"""Alpaca trading-calendar provider (audit §5.2, §11.1; spec §10).

Serves the ONE thing the platform is missing to classify an earnings release
as before/after market correctly: the exchange session table. Today
``routers/analysis.py::_last_expected_trading_date`` does Mon-Fri arithmetic
and its own docstring admits "Holidays are not modeled" — this adapter is
what fixes that (audit §5.2).

Endpoint (Trading API, the same paper keys the broker already uses):

    GET https://paper-api.alpaca.markets/v2/calendar?start=YYYY-MM-DD&end=...

    -> [{"date": "2026-11-27", "open": "09:30", "close": "13:00",
         "session_open": "0700", "session_close": "1700",
         "settlement_date": "2026-12-01"}, ...]

``open``/``close`` are Eastern WALL-CLOCK times for the regular session;
``session_open``/``session_close`` are the extended-hours window in HHMM.
Everything is converted to UTC through ``America/New_York`` so a November
session and a July session are both correct across the DST boundary (§10:
"Avoid date-only logic for market-moving events").

An early close is derived from the data, not from a holiday list: any day
whose regular close is not 16:00 ET is flagged ``is_early_close``.

Alpaca serves NO earnings calendar, so ``earnings_calendar`` and
``earnings_history`` are reported ``False`` — a documented absence, not an
omitted key (audit §6).
"""
import logging
from datetime import date, datetime, timedelta
from typing import Sequence

import httpx

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError

from .provider import (
    EASTERN,
    CalendarProviderError,
    EventCandidate,
    MarketDay,
    blank_capabilities,
)

logger = logging.getLogger(__name__)

#: Trading API host. Reference data (the calendar) is identical across
#: paper/live, so paper keys are correct here — the same choice
#: libs/market_data/alpaca.py makes for the option-contracts endpoint.
ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"

DEFAULT_TIMEOUT_SECONDS = 15.0

#: Regular-session close. A day closing earlier is an early close (half day).
REGULAR_CLOSE = "16:00"

#: Calendar probe window — two days is enough to prove entitlement.
PROBE_DAYS = 2

#: The exchange this adapter reports sessions for.
EXCHANGE = "US"

SOURCE_NAME = "alpaca_calendar"


def _parse_hhmm(raw: object) -> tuple[int, int] | None:
    """"09:30" or "0930" -> (9, 30); anything else -> None (honest absence)."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace(":", "")
    if len(text) != 4 or not text.isdigit():
        return None
    hour, minute = int(text[:2]), int(text[2:])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    """Eastern wall-clock (day, hour, minute) as a UTC instant.

    ZoneInfo resolves the UTC offset from the DATE, so 09:30 in January is
    14:30Z and 09:30 in July is 13:30Z — the DST correctness the whole event
    clock depends on (§10).
    """
    from datetime import timezone

    local = datetime(day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


class AlpacaCalendarProvider:
    """EventCalendarProvider serving exchange sessions from Alpaca."""

    name = SOURCE_NAME

    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        base_url: str = ALPACA_TRADING_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`transport` is injectable so tests use httpx.MockTransport.

        Blank credentials are refused at construction, mirroring
        ``AlpacaMarketDataProvider``: the adapter can never fire keyless.
        """
        if not api_key_id or not api_key_id.strip():
            raise CalendarProviderError(
                "Alpaca calendar requires credentials — set ALPACA_API_KEY_ID "
                "(the key itself is never logged)"
            )
        if not api_secret_key or not api_secret_key.strip():
            raise CalendarProviderError(
                "Alpaca calendar requires credentials — set "
                "ALPACA_API_SECRET_KEY (the key itself is never logged)"
            )
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "APCA-API-KEY-ID": api_key_id.strip(),
                "APCA-API-SECRET-KEY": api_secret_key.strip(),
            },
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    # ------------------------------------------------------------------
    # Transport (same taxonomy as libs/market_data/alpaca.py)
    # ------------------------------------------------------------------

    def _request(self, path: str, params: dict | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path  # path only: never log the query/keys
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"Alpaca calendar request failed for {endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 401:
            raise CalendarProviderError(
                f"Alpaca rejected the API key (HTTP 401) for {endpoint} — "
                "check ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (the keys "
                "are never logged or echoed)"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"Alpaca returned HTTP 403 for {endpoint}: the account's "
                "subscription does not include this endpoint. There is NO "
                f"synthetic fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"Alpaca calendar returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response

    @staticmethod
    def _rows(response: httpx.Response) -> list[dict]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise CalendarProviderError(
                "Alpaca calendar returned unparseable JSON for "
                f"{httpx.URL(str(response.url)).path}"
            ) from exc
        if not isinstance(payload, list):
            raise CalendarProviderError(
                "Alpaca calendar returned a non-list payload for "
                f"{httpx.URL(str(response.url)).path}"
            )
        return [row for row in payload if isinstance(row, dict)]

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Probe /v2/calendar; every other capability is a documented False.

        Alpaca does not sell an earnings calendar at any tier probed
        (audit §13), so those keys are ``False`` by fact, not by probe.
        """
        report = blank_capabilities()
        today = datetime.now(EASTERN).date()
        try:
            self._request(
                "/v2/calendar",
                params={
                    "start": today.isoformat(),
                    "end": (today + timedelta(days=PROBE_DAYS)).isoformat(),
                },
            )
        except CapabilityNotAvailable as exc:
            logger.warning("Alpaca capability 'market_calendar' unavailable: %s", exc)
            report["market_calendar"] = False
        except MarketDataError as exc:
            report["market_calendar"] = str(exc)
        else:
            report["market_calendar"] = True
        return report

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """Exchange sessions in ``[start, end]``, oldest first.

        A row with an unparseable date or open/close is SKIPPED with a
        warning rather than defaulted to 09:30-16:00 — a fabricated session
        would silently misclassify every earnings release on that day.
        """
        rows = self._rows(
            self._request(
                "/v2/calendar",
                params={"start": start.isoformat(), "end": end.isoformat()},
            )
        )
        days: list[MarketDay] = []
        for row in rows:
            raw_date = row.get("date")
            if not isinstance(raw_date, str):
                continue
            try:
                session_date = date.fromisoformat(raw_date.strip())
            except ValueError:
                logger.warning("Alpaca calendar row skipped: bad date %r", raw_date)
                continue
            open_hm = _parse_hhmm(row.get("open"))
            close_hm = _parse_hhmm(row.get("close"))
            if open_hm is None or close_hm is None:
                logger.warning(
                    "Alpaca calendar row skipped for %s: unparseable session "
                    "times (open=%r close=%r)",
                    session_date, row.get("open"), row.get("close"),
                )
                continue
            ext_open_hm = _parse_hhmm(row.get("session_open"))
            ext_close_hm = _parse_hhmm(row.get("session_close"))
            close_text = str(row.get("close", "")).strip()
            days.append(
                MarketDay(
                    session_date=session_date,
                    exchange=EXCHANGE,
                    open_utc=_et_to_utc(session_date, open_hm),
                    close_utc=_et_to_utc(session_date, close_hm),
                    session_open_utc=(
                        _et_to_utc(session_date, ext_open_hm)
                        if ext_open_hm is not None else None
                    ),
                    session_close_utc=(
                        _et_to_utc(session_date, ext_close_hm)
                        if ext_close_hm is not None else None
                    ),
                    is_early_close=close_text.replace(":", "")
                    != REGULAR_CLOSE.replace(":", ""),
                    source=SOURCE_NAME,
                )
            )
        days.sort(key=lambda d: d.session_date)
        return days

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """Alpaca's calendar carries no discrete events — always empty.

        Holidays reach the platform as the ABSENCE of a session row plus
        Massive's holiday feed; emitting a synthetic MARKET_HOLIDAY here from
        gaps would invent names ("Thanksgiving") the source never sent.
        """
        return []
