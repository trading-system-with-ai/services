"""Bureau of Labor Statistics release calendar (spec §8, §38-§41; audit §6).

BLS publishes one schedule page per news release, each a single
``<table class="release-list">`` whose rows are
``Reference Month | Release Date | Release Time``::

    <tr><td>July 2026</td><td>Aug. 12, 2026</td><td>08:30 AM</td></tr>

Four releases are mapped to typed events (§5's taxonomy — never a generic
"macro event"):

  ``cpi.htm``     -> :attr:`EventType.CPI`
  ``ppi.htm``     -> :attr:`EventType.PPI`
  ``empsit.htm``  -> :attr:`EventType.EMPLOYMENT_REPORT`
  ``jolts.htm``   -> :attr:`EventType.JOLTS`

Real Earnings (``realer.htm``) is deliberately SKIPPED: it is not a
market-moving print and the spec's macro table does not list it.

THE RELEASE TIME IS PARSED, NEVER ASSUMED. Verified live 2026-08-19: CPI,
PPI and the Employment Situation drop at 08:30 ET but **JOLTS drops at
10:00 ET** — i.e. AFTER the open, which makes it a ``DURING_MARKET`` event
rather than ``BEFORE_MARKET``. Hardcoding 08:30 would put every JOLTS event
90 minutes early and in the wrong session bucket, so the third column is
read from the page and a row whose time cannot be parsed is DROPPED rather
than defaulted (audit §6: fail loudly and empty).

AS-OF (§14/§96). BLS publishes its schedule a full year ahead, so a release
row is knowable long before it happens: unlike an RSS item, there is no
publication instant to compare against ``as_of``. Candidates are therefore
NOT filtered by ``as_of`` here — the honest statement is "this schedule was
public well before as_of". What ``as_of`` does gate is macro OBSERVATIONS
(the actual numbers), which is enforced in the macro packet layer, not here.

HTML SCRAPERS MUST FAIL LOUDLY AND EMPTY (audit §6): a layout change yields
zero rows plus a warning, never a partially-parsed release at a guessed time.
"""
import logging
import re
from datetime import date, datetime, timezone
from typing import Sequence

import httpx

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError
from libs.trading_core.models.enums import (
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
    classify_session_et,
)

logger = logging.getLogger(__name__)

BLS_BASE_URL = "https://www.bls.gov"
SCHEDULE_PATH = "/schedule/news_release/{slug}.htm"

DEFAULT_TIMEOUT_SECONDS = 15.0

SOURCE_NAME = "bls"
AGENCY = "Bureau of Labor Statistics"

#: The schedule pages this adapter reads, in taxonomy order. ``realer`` (Real
#: Earnings) is intentionally absent — see the module docstring.
SCHEDULE_SLUGS: dict[str, EventType] = {
    "cpi": EventType.CPI,
    "ppi": EventType.PPI,
    "empsit": EventType.EMPLOYMENT_REPORT,
    "jolts": EventType.JOLTS,
}

#: Human title stem per event type — the reference period is appended, so a
#: row renders as "CPI — July 2026" (§10: the period is part of the identity;
#: two CPI releases differ by which month they describe).
TITLE_STEM: dict[EventType, str] = {
    EventType.CPI: "CPI",
    EventType.PPI: "PPI",
    EventType.EMPLOYMENT_REPORT: "Employment Situation",
    EventType.JOLTS: "JOLTS",
}

#: The BLS series each release headlines, recorded on the candidate for the
#: macro layer to join against (``EventCandidate.series_id``).
HEADLINE_SERIES: dict[EventType, str] = {
    EventType.CPI: "CUSR0000SA0",
    EventType.PPI: "WPSFD4",
    EventType.EMPLOYMENT_REPORT: "CES0000000001",
    EventType.JOLTS: "JTS000000000000000JOL",
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

#: The schedule table. One per page, verified live 2026-08-19.
_TABLE_RE = re.compile(
    r'<table[^>]*class="[^"]*\brelease-list\b[^"]*"[^>]*>(.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

#: "January 2026" / "2nd Quarter 2026" reference-period cell.
_REF_MONTH_RE = re.compile(r"^([A-Za-z]{3,9})\s+(20\d{2})$")

#: "Feb. 13, 2026" and the zero-padded "Jan. 09, 2026" both appear live.
_RELEASE_DATE_RE = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(20\d{2})$")

#: "08:30 AM" / "10:00 AM".
_RELEASE_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*([AP])\.?M\.?$", re.IGNORECASE)


def _clean(cell: str) -> str:
    """Tag-strip a table cell and collapse whitespace/entities."""
    text = _TAG_RE.sub(" ", cell or "")
    text = text.replace("&nbsp;", " ").replace("\xa0", " ").replace("&amp;", "&")
    return " ".join(text.split())


def _month_number(name: str) -> int | None:
    return _MONTHS.get((name or "").strip().lower().rstrip("."))


def _parse_release_date(text: str) -> date | None:
    match = _RELEASE_DATE_RE.match(text.strip())
    if not match:
        return None
    month = _month_number(match.group(1))
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def _parse_release_time(text: str) -> tuple[int, int] | None:
    """``"08:30 AM"`` -> ``(8, 30)``; ``"10:00 AM"`` -> ``(10, 0)``.

    Returns ``None`` for anything unrecognised so the caller can DROP the row
    — a release at an unknown time is not a fact we may invent.
    """
    match = _RELEASE_TIME_RE.match(text.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (1 <= hour <= 12) or not (0 <= minute <= 59):
        return None
    meridiem = match.group(3).upper()
    if meridiem == "A":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour, minute


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    """Eastern wall-clock -> UTC (DST resolved from the date).

    A February 08:30 ET release is 13:30Z and an August one is 12:30Z; every
    T-minus alert is an hour wrong for half the year if this is wrong.
    """
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


def parse_schedule_page(html: str) -> list[dict]:
    """Rows of ``{reference_period, release_date, release_time, time_text}``.

    Returns ``[]`` when the table is missing or its layout changed. Rows whose
    date OR time cannot be parsed are skipped individually with a warning:
    one reformatted row must not cost the other twelve, but neither may it be
    emitted at a guessed instant.
    """
    table = _TABLE_RE.search(html or "")
    if not table:
        return []
    out: list[dict] = []
    for row_html in _ROW_RE.findall(table.group(1)):
        cells = [_clean(c) for c in _CELL_RE.findall(row_html)]
        if len(cells) < 3:
            continue
        period_text, date_text, time_text = cells[0], cells[1], cells[2]
        if period_text.lower().startswith("reference"):
            continue  # the <thead> row
        release_date = _parse_release_date(date_text)
        if release_date is None:
            logger.warning("BLS schedule row has an unparseable date %r", date_text)
            continue
        release_time = _parse_release_time(time_text)
        if release_time is None:
            logger.warning(
                "BLS schedule row for %s has an unparseable release time %r — "
                "dropping the row rather than assuming a time",
                date_text, time_text,
            )
            continue
        out.append(
            {
                "reference_period": period_text,
                "release_date": release_date,
                "release_time": release_time,
                "time_text": time_text,
            }
        )
    return out


def reference_period_code(text: str) -> str | None:
    """``"July 2026"`` -> ``"2026-07"``; unrecognised -> ``None``.

    The canonical period key the macro layer joins observations on.
    """
    match = _REF_MONTH_RE.match((text or "").strip())
    if not match:
        return None
    month = _month_number(match.group(1))
    if month is None:
        return None
    return f"{int(match.group(2)):04d}-{month:02d}"


class BlsCalendarProvider:
    """Release calendar for CPI, PPI, the Employment Situation and JOLTS.

    Keyless: BLS schedule pages are free public HTML. A contact User-Agent is
    still sent on every request — the platform's rule for every government
    source (SEC's fair-access policy, and simple courtesy to BLS).
    """

    name = "bls"

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str = BLS_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": user_agent},
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"BLS request failed for {endpoint}: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"bls.gov returned HTTP 403 for {endpoint}: the request was "
                "refused (User-Agent or IP block). There is NO synthetic "
                f"fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"bls.gov returned HTTP {response.status_code} for {endpoint}: "
                f"{response.text[:300]}"
            )
        return response.text

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Probe ONE schedule page (CPI). Never raises.

        ``macro_calendar`` is True only when the page fetches AND parses to at
        least one row: a 200 that yields nothing is a layout change, which is
        an availability problem even though HTTP succeeded.
        """
        report = blank_capabilities()
        try:
            html = self._get_text(SCHEDULE_PATH.format(slug="cpi"))
        except CapabilityNotAvailable as exc:
            logger.warning("BLS capability 'macro_calendar' unavailable: %s", exc)
            report["macro_calendar"] = False
        except MarketDataError as exc:
            report["macro_calendar"] = str(exc)
        else:
            if parse_schedule_page(html):
                report["macro_calendar"] = True
            else:
                report["macro_calendar"] = (
                    "bls.gov CPI schedule returned HTTP 200 but no release "
                    "rows could be parsed — the page layout changed"
                )
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
        """Every mapped BLS release in ``[start, end]``.

        ``tickers`` is ignored — macro prints are market-wide. Each schedule
        page is fetched inside its own try: a layout change on JOLTS must not
        cost the platform its CPI calendar (§8 failure isolation).

        ``as_of`` is accepted for Protocol conformance and deliberately NOT
        applied: BLS publishes this schedule a year ahead, so every row was
        knowable well before any realistic ``as_of`` (module docstring).
        """
        out: list[EventCandidate] = []
        for slug, event_type in SCHEDULE_SLUGS.items():
            try:
                out.extend(self.fetch_release_schedule(slug, event_type))
            except MarketDataError as exc:
                logger.warning("BLS %s schedule unavailable: %s", slug, exc)
        return [c for c in out if start <= c.scheduled_at <= end]

    def fetch_release_schedule(
        self, slug: str, event_type: EventType
    ) -> list[EventCandidate]:
        """Candidates from one schedule page."""
        path = SCHEDULE_PATH.format(slug=slug)
        rows = parse_schedule_page(self._get_text(path))
        if not rows:
            logger.warning(
                "BLS %s schedule page yielded no rows — refusing to emit any "
                "%s event rather than guess dates", slug, event_type.value,
            )
            return []
        url = f"{self.base_url}{path}"
        stem = TITLE_STEM.get(event_type, event_type.value)
        out: list[EventCandidate] = []
        for row in rows:
            release_date: date = row["release_date"]
            scheduled_at = _et_to_utc(release_date, row["release_time"])
            period_text = row["reference_period"]
            out.append(
                EventCandidate(
                    event_key=f"{event_type.value}:{release_date.isoformat()}",
                    event_type=event_type,
                    title=f"{stem} — {period_text}",
                    scheduled_at=scheduled_at,
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.GOVERNMENT_AGENCY,
                    source_name=SOURCE_NAME,
                    event_timezone=US_EVENT_TIMEZONE,
                    # Parsed, not assumed: JOLTS at 10:00 ET is DURING_MARKET.
                    session=classify_session_et(scheduled_at),
                    source_url=url,
                    agency=AGENCY,
                    series_id=HEADLINE_SERIES.get(event_type),
                    release_period=reference_period_code(period_text),
                    raw={
                        "reference_period": period_text,
                        "release_time_text": row["time_text"],
                        "url": url,
                    },
                )
            )
        return out

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """BLS publishes its own release schedule, not the exchange's."""
        raise CapabilityNotAvailable(
            "the Bureau of Labor Statistics does not serve exchange sessions "
            "— use the alpaca_calendar provider for market_calendar"
        )
