"""Massive-backed holiday + earnings-calendar provider (audit §6, §11.1).

Two endpoints, two very different verdicts — which is exactly why this
adapter exists as a capability PROBE and not as an assumption:

  - ``GET /v1/marketstatus/upcoming`` — **entitled (200)**. Returns upcoming
    exchange holidays and early closes:

        [{"date": "2026-11-26", "exchange": "NYSE", "name": "Thanksgiving",
          "status": "closed"},
         {"date": "2026-11-27", "exchange": "NYSE", "name": "Thanksgiving",
          "status": "early-close", "open": "09:30", "close": "13:00"}]

    Each row becomes a ``MARKET_HOLIDAY`` candidate: CONFIRMED, source kind
    STRUCTURED_PROVIDER. Rows are per-exchange, so NYSE and NASDAQ both
    appear for the same day and the event_key carries the exchange.

  - ``GET /benzinga/v1/earnings`` — **403 today** (audit §13: every Benzinga
    endpoint is outside this subscription). The 403 lands as
    ``earnings_calendar: False`` — a PROVEN absence, logged once and NEVER
    raised into the ingestion loop (audit §6 "SUBSCRIPTION_DENIED as a
    first-class verdict"). If the add-on is ever purchased the probe flips to
    True on its own and :meth:`fetch_events` starts emitting real CONFIRMED
    earnings dates with no code change — the fallback chain in audit §6 is
    ordered data, not branching code.

Until then upcoming earnings dates come from
:mod:`libs.event_calendar.sec_edgar`'s cadence estimate, carrying
``EventStatus.ESTIMATED`` all the way to the UI (§11: "Do not fabricate an
exact event date when only an estimate exists").
"""
import logging
from datetime import date, datetime, timezone
from typing import Sequence

import httpx

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from .provider import (
    EASTERN,
    US_EVENT_TIMEZONE,
    CalendarProviderError,
    EventCandidate,
    MarketDay,
    blank_capabilities,
)

logger = logging.getLogger(__name__)

MASSIVE_BASE_URL = "https://api.massive.com"
DEFAULT_TIMEOUT_SECONDS = 15.0

HOLIDAYS_PATH = "/v1/marketstatus/upcoming"
EARNINGS_PATH = "/benzinga/v1/earnings"

#: Cheap symbol for the earnings entitlement probe.
PROBE_SYMBOL = "AAPL"

SOURCE_NAME = "massive_calendar"

_REDACTED = "***redacted***"

#: Holidays have no wall-clock instant; anchor them at the regular open so
#: horizon math and ordering work. The DATE is the fact; the time is a
#: presentation anchor, and ``session`` stays UNKNOWN to say so.
HOLIDAY_ANCHOR_ET = (9, 30)

#: Benzinga's ``time`` field ("bmo"/"amc"/"dmt") -> our session taxonomy.
_BENZINGA_SESSION = {
    "bmo": EventSession.BEFORE_MARKET,
    "before": EventSession.BEFORE_MARKET,
    "amc": EventSession.AFTER_MARKET,
    "after": EventSession.AFTER_MARKET,
    "dmt": EventSession.DURING_MARKET,
    "during": EventSession.DURING_MARKET,
}

#: Eastern anchor time per session when the source gives only a date + a
#: BMO/AMC marker. Mirrors sec_edgar's cadence anchors so an ESTIMATED and a
#: CONFIRMED earnings event for the same ticker sort consistently.
_SESSION_ANCHOR_ET = {
    EventSession.BEFORE_MARKET: (7, 0),
    EventSession.AFTER_MARKET: (16, 5),
    EventSession.DURING_MARKET: (12, 0),
    EventSession.UNKNOWN: (12, 0),
}

#: Whether the SUBSCRIPTION_DENIED verdict has already been logged this
#: process. The 403 is a permanent plan fact, not a per-request event — log
#: it once, exactly as libs/market_data/alpaca.py does for unservable symbols.
_EARNINGS_DENIED_LOGGED = False


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    """Eastern wall-clock -> UTC instant (DST resolved from the date)."""
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


def _as_date(raw: object) -> date | None:
    """An ISO date (or the date half of an ISO datetime) -> date, else None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


class MassiveCalendarProvider:
    """EventCalendarProvider over Massive's holiday + earnings endpoints."""

    name = SOURCE_NAME

    def __init__(
        self,
        api_key: str,
        base_url: str = MASSIVE_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`transport` is injectable so tests use httpx.MockTransport.

        Blank keys are refused at construction (the Massive market-data
        adapter's rule): the adapter can never fire keyless.
        """
        if not api_key or not api_key.strip():
            raise CalendarProviderError(
                "MassiveCalendarProvider requires a non-empty API key "
                "(MASSIVE_API_KEY / settings.massive_api_key); there is NO "
                "synthetic fallback"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        # Header-first auth; flips to "query" only after the server rejects
        # header auth — the exact fallback MassiveProvider._request performs.
        self._auth_mode = "header"

    # ------------------------------------------------------------------
    # HTTP plumbing (MassiveProvider._send/_request, same taxonomy)
    # ------------------------------------------------------------------

    def _send(self, url: str, params: dict | None) -> httpx.Response:
        request_url = httpx.URL(url)
        if params:
            request_url = request_url.copy_merge_params(
                {k: str(v) for k, v in params.items()}
            )
        headers: dict[str, str] = {}
        if self._auth_mode == "header":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            request_url = request_url.copy_set_param("apiKey", self.api_key)
        # Log the path only — the query string could carry the apiKey.
        logger.debug(
            "Massive calendar GET %s (auth=%s key=%s)",
            request_url.path, self._auth_mode, _REDACTED,
        )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                return client.get(request_url, headers=headers)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"Massive calendar request failed: {exc!r}"
            ) from exc

    def _request(self, path: str, params: dict | None = None) -> httpx.Response:
        """One Massive call: 401 -> one apiKey-param retry; 403 -> capability."""
        url = f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path

        response = self._send(url, params)
        if response.status_code == 401 and self._auth_mode == "header":
            logger.debug(
                "Massive rejected header auth (HTTP 401) for %s; retrying once "
                "with the apiKey query parameter", endpoint,
            )
            self._auth_mode = "query"
            response = self._send(url, params)
            if response.status_code == 401:
                self._auth_mode = "header"  # neither form worked

        if response.status_code == 401:
            raise CalendarProviderError(
                f"Massive rejected the API key (HTTP 401) for {endpoint} — "
                "check MASSIVE_API_KEY (the key itself is never logged or echoed)"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"Massive returned HTTP 403 for {endpoint}: the configured "
                "plan/subscription does not include this endpoint. There is "
                f"NO synthetic fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"Massive calendar returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response

    @staticmethod
    def _rows(response: httpx.Response) -> list[dict]:
        """The payload as a list of objects.

        Massive serves this family bare-list, but a ``{"results": [...]}``
        envelope is accepted too so a server-side shape change degrades to
        parsing rather than to zero rows.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise CalendarProviderError(
                "Massive calendar returned unparseable JSON for "
                f"{httpx.URL(str(response.url)).path}"
            ) from exc
        if isinstance(payload, dict):
            payload = payload.get("results")
        if not isinstance(payload, list):
            raise CalendarProviderError(
                "Massive calendar returned a non-list payload for "
                f"{httpx.URL(str(response.url)).path}"
            )
        return [row for row in payload if isinstance(row, dict)]

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Probe holidays (expected 200) and Benzinga earnings (expected 403).

        Never raises: a probe fault becomes the error STRING, which the UI
        renders as "availability unknown" — distinct from the ``False`` that
        means "proven not in the plan" (audit §6).
        """
        global _EARNINGS_DENIED_LOGGED
        report = blank_capabilities()

        try:
            self._request(HOLIDAYS_PATH)
        except CapabilityNotAvailable as exc:
            logger.warning("Massive capability 'market_holidays' unavailable: %s", exc)
            report["market_holidays"] = False
        except MarketDataError as exc:
            report["market_holidays"] = str(exc)
        else:
            report["market_holidays"] = True

        try:
            self._request(EARNINGS_PATH, params={"ticker": PROBE_SYMBOL, "limit": 1})
        except CapabilityNotAvailable as exc:
            if not _EARNINGS_DENIED_LOGGED:
                logger.warning(
                    "Massive capability 'earnings_calendar' unavailable "
                    "(SUBSCRIPTION_DENIED — upcoming earnings dates will be "
                    "ESTIMATED from SEC filing cadence): %s", exc,
                )
                _EARNINGS_DENIED_LOGGED = True
            report["earnings_calendar"] = False
        except MarketDataError as exc:
            report["earnings_calendar"] = str(exc)
        else:
            report["earnings_calendar"] = True

        return report

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """Holiday candidates always; earnings candidates only if entitled.

        A 403 on either endpoint yields FEWER candidates and a capability
        verdict — never an exception into the ingestion loop, and never a
        fabricated date to fill the gap (§8 "calendar ingestion should
        survive individual provider failures").
        """
        candidates: list[EventCandidate] = []
        candidates.extend(self._fetch_holidays(start, end))
        candidates.extend(self._fetch_earnings(tickers, start, end))
        return candidates

    def _fetch_holidays(
        self, start: datetime, end: datetime
    ) -> list[EventCandidate]:
        try:
            rows = self._rows(self._request(HOLIDAYS_PATH))
        except CapabilityNotAvailable as exc:
            logger.warning("Massive holidays unavailable (403): %s", exc)
            return []
        except MarketDataError as exc:
            logger.warning("Massive holidays fetch failed: %s", exc)
            return []

        window_start, window_end = start.date(), end.date()
        out: list[EventCandidate] = []
        for row in rows:
            day = _as_date(row.get("date"))
            if day is None:
                logger.debug("Massive holiday row skipped: bad date %r", row.get("date"))
                continue
            if day < window_start or day > window_end:
                continue
            exchange = str(row.get("exchange") or "US").strip().upper() or "US"
            status = str(row.get("status") or "").strip()
            name = str(row.get("name") or "").strip() or "Market holiday"
            title = f"{exchange} {name}"
            if status and status.lower().replace("_", "-") != "closed":
                # early-close days are still market-moving: say so in the title
                title = f"{exchange} {name} ({status})"
            out.append(
                EventCandidate(
                    event_key=f"MARKET_HOLIDAY:{exchange}:{day.isoformat()}",
                    event_type=EventType.MARKET_HOLIDAY,
                    title=title,
                    scheduled_at=_et_to_utc(day, HOLIDAY_ANCHOR_ET),
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.STRUCTURED_PROVIDER,
                    source_name=SOURCE_NAME,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=EventSession.UNKNOWN,
                    agency=exchange,
                    raw=dict(row),
                )
            )
        return out

    def _fetch_earnings(
        self, tickers: Sequence[str], start: datetime, end: datetime
    ) -> list[EventCandidate]:
        """Benzinga earnings — empty (with a logged verdict) while 403.

        The mapping below is live code, not a stub: the moment the add-on is
        entitled the same call returns 200 and these candidates flow, with
        ``date_confirmed`` deciding CONFIRMED vs ESTIMATED so an unconfirmed
        vendor row is never promoted to a fact.
        """
        global _EARNINGS_DENIED_LOGGED
        symbols = [t.strip().upper() for t in tickers if t and t.strip()]
        if not symbols:
            return []
        window_start, window_end = start.date(), end.date()
        out: list[EventCandidate] = []
        for symbol in symbols:
            try:
                rows = self._rows(
                    self._request(
                        EARNINGS_PATH,
                        params={
                            "ticker": symbol,
                            "date_from": window_start.isoformat(),
                            "date_to": window_end.isoformat(),
                        },
                    )
                )
            except CapabilityNotAvailable as exc:
                if not _EARNINGS_DENIED_LOGGED:
                    logger.warning(
                        "Massive earnings calendar denied (403) — upcoming "
                        "earnings dates fall back to the SEC cadence "
                        "ESTIMATE: %s", exc,
                    )
                    _EARNINGS_DENIED_LOGGED = True
                return out  # plan-wide fact: no point asking for other tickers
            except MarketDataError as exc:
                logger.warning(
                    "Massive earnings fetch failed for %s: %s", symbol, exc
                )
                continue
            for row in rows:
                candidate = self._earnings_candidate(symbol, row, window_start, window_end)
                if candidate is not None:
                    out.append(candidate)
        return out

    @staticmethod
    def _earnings_candidate(
        symbol: str, row: dict, window_start: date, window_end: date
    ) -> EventCandidate | None:
        day = _as_date(row.get("date"))
        if day is None or day < window_start or day > window_end:
            return None
        ticker = str(row.get("ticker") or symbol).strip().upper() or symbol
        session = _BENZINGA_SESSION.get(
            str(row.get("time") or "").strip().lower(), EventSession.UNKNOWN
        )
        confirmed = row.get("date_confirmed")
        is_confirmed = confirmed is True or str(confirmed).strip() in {"1", "true", "True"}
        fiscal_quarter = row.get("period")
        fiscal_year = row.get("period_year") or row.get("fiscal_year")
        return EventCandidate(
            event_key=f"EARNINGS:{ticker}:{day.isoformat()}",
            event_type=EventType.EARNINGS,
            title=f"{ticker} earnings release",
            scheduled_at=_et_to_utc(day, _SESSION_ANCHOR_ET[session]),
            status=EventStatus.CONFIRMED if is_confirmed else EventStatus.ESTIMATED,
            source=EventSourceKind.STRUCTURED_PROVIDER,
            source_name=SOURCE_NAME,
            ticker=ticker,
            event_timezone=US_EVENT_TIMEZONE,
            session=session,
            source_event_id=str(row.get("id")) if row.get("id") is not None else None,
            fiscal_quarter=(
                int(fiscal_quarter)
                if isinstance(fiscal_quarter, (int, float))
                or (isinstance(fiscal_quarter, str) and fiscal_quarter.isdigit())
                else None
            ),
            fiscal_year=(
                int(fiscal_year)
                if isinstance(fiscal_year, (int, float))
                or (isinstance(fiscal_year, str) and fiscal_year.isdigit())
                else None
            ),
            raw=dict(row),
        )

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """Massive's holiday feed carries no per-day open/close table.

        Raises rather than returning ``[]``: an empty list here would be read
        as "the exchange never opens", and the caller must fall through to
        the Alpaca calendar instead.
        """
        raise CapabilityNotAvailable(
            "Massive serves market HOLIDAYS, not per-session open/close rows "
            "— use the alpaca_calendar provider for market_calendar"
        )
