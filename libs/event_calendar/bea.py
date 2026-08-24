"""Bureau of Economic Analysis release calendar (spec §8, §38-§41).

BEA's release schedule (https://www.bea.gov/news/schedule) is one
``<table id="release-schedule-table">`` whose rows look like::

    <tr class="scheduled-releases-type-press">
      <td class="scheduled-date no-wrap">
        <div class="release-date">August 26</div>
        <small class="text-muted">8:30 AM</small></td>
      <td …>News</td>
      <td class="release-title …">GDP (Second Estimate) and Corporate
          Profits, 2nd Quarter 2026</td>
    </tr>

Two layout facts (verified live 2026-08-19) drive this parser and are the
reason it is fixture-pinned:

1. **The year is NOT in the row.** Each date cell reads "August 26"; the only
   year on the page is the column header ``<th>Year 2026</th>``. The year is
   therefore taken from that header, and because the table runs forward in
   time, a row whose month goes BACKWARDS relative to its predecessor has
   wrapped into the next year (a December -> January boundary). Reading the
   year from "today" instead would silently mis-date every row each January.

2. **BEA never writes "Gross Domestic Product" on this page.** GDP releases
   are titled ``GDP (Advance Estimate), 3rd Quarter 2026`` /
   ``GDP (Second Estimate) and Corporate Profits, …`` / ``GDP (Third
   Estimate), Industries, …``. Matching on the spelled-out phrase would find
   ZERO rows. PCE arrives as ``Personal Income and Outlays, July 2026``.

Only these two families become events (:attr:`EventType.GDP` and
:attr:`EventType.PCE`). Regional/annual products — "GDP by County", "Real
Personal Consumption Expenditures by State", "Activities of U.S.
Multinational Enterprises" — are SKIPPED: they are not market-moving prints
and typing them as GDP/PCE would put false catalysts on the calendar.

BEA's *statistics API* needs a free key (``settings.bea_api_key``); this
SCHEDULE page does not. Absent a key the platform still shows every BEA date
and only the ACTUALS are unavailable — see :mod:`libs.event_calendar.macro_data`.
"""
import html as html_module
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

BEA_BASE_URL = "https://www.bea.gov"
SCHEDULE_PATH = "/news/schedule"

#: BEA's PAST releases. The schedule page above is forward-looking ONLY — it
#: starts at today and never lists a release that already happened — so on a
#: fresh install GDP and PCE had ZERO history and every §15 comparison
#: answered "no comparable event", which is true of the rows but false of the
#: world (found live 2026-08-22). The archive is the same site's index of
#: releases already published, and it carries a full ISO timestamp per row.
#:
#: The query parameter selects one product family; without it the page mixes
#: in blogs and working papers. It also selects only ONE family, so the GDP
#: filter (451) carries no Personal Income release at all — PCE needs its own
#: id (476). Filtering to 451 alone is why PCE had zero stored history while
#: GDP had six.
ARCHIVE_PATHS = (
    "/news/archive?field_related_product_target_id=451",  # Gross Domestic Product
    "/news/archive?field_related_product_target_id=476",  # Personal Income (PCE)
)

#: Kept as the GDP filter's own name for the tests and callers that reference
#: a single archive path.
ARCHIVE_PATH = ARCHIVE_PATHS[0]

#: How many archive pages to walk. Each holds ~10 releases and GDP/PCE run
#: ~24 releases a year between them, so 4 pages ≈ 18 months of history —
#: enough for §15 to find a previous comparable event several cycles back
#: without hammering a free public site.
ARCHIVE_PAGES = 4

DEFAULT_TIMEOUT_SECONDS = 15.0

SOURCE_NAME = "bea"
AGENCY = "Bureau of Economic Analysis"


def _event_title(event_type: EventType, release_title: str) -> str:
    """Title for the candidate.

    BEA's own title already names the release ("GDP (Advance Estimate), 3rd
    Quarter 2026"), so it is used verbatim for GDP rather than prefixed into
    "GDP — GDP (…)". PCE rows read "Personal Income and Outlays, July 2026",
    which does not say PCE anywhere, so the ticker-style tag is added.
    """
    if event_type is EventType.PCE:
        return f"PCE — {release_title}"
    return release_title


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_TABLE_RE = re.compile(
    r'<table[^>]*id="release-schedule-table"[^>]*>(.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_YEAR_HEADER_RE = re.compile(r"Year\s+(20\d{2})", re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_DATE_RE = re.compile(
    r'class="release-date"[^>]*>\s*([A-Za-z]{3,9})\s+(\d{1,2})\s*<', re.IGNORECASE
)
_TIME_RE = re.compile(
    r'class="text-muted"[^>]*>\s*(\d{1,2}):(\d{2})\s*([AP])\.?M\.?\s*<', re.IGNORECASE
)
_TITLE_RE = re.compile(
    r'class="[^"]*\brelease-title\b[^"]*"[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")

#: A GDP headline release. Anchored at the START so "GDP by County" and
#: "Real GDP by State" (regional annual products) do not match.
#:
#: BEA changed its headline wording: 2026 releases read "GDP (Advance
#: Estimate), 1st Quarter 2026" while everything before roughly 2026 reads
#: "Gross Domestic Product, 3rd Quarter 2025 (Initial Estimate)" — different
#: word, different clause order, and an Initial/Updated estimate vocabulary
#: alongside Advance/Second/Third. Matching only the modern form silently
#: dropped 14 of the 20 rows on the first archive page, which is how a series
#: published since 1947 came to have no history at all. The estimate clause is
#: matched anywhere in the title rather than immediately after the name.
_GDP_RE = re.compile(
    r"^(?:GDP|Gross\s+Domestic\s+Product)\b(?![^,(]*\bby\b)"
    r"(?=.*\((?:advance|initial|second|preliminary|third|updated|final)\s+estimate\))",
    re.IGNORECASE | re.DOTALL,
)

#: The PCE release. "Personal Income and Outlays, July 2026" — but NOT
#: "…by State", which is a separate regional annual product.
_PCE_RE = re.compile(r"^Personal\s+Income\s+and\s+Outlays\s*,", re.IGNORECASE)

#: Trailing reference period: "…, 2nd Quarter 2026" or "…, July 2026".
#: ``and Year`` is optional: BEA writes the Q4 release as "4th Quarter and
#: Year 2025", and requiring the year to follow "Quarter" directly left those
#: three releases with a NULL period. That is not cosmetic — §15 rejects a
#: candidate whose ``release_period`` EQUALS the event's, which is what stops
#: the Second Estimate of a quarter from being compared against the Advance
#: Estimate of the same quarter. With both NULL that guard cannot fire.
_QUARTER_PERIOD_RE = re.compile(
    r"(\d)(?:st|nd|rd|th)\s+Quarter\s+(?:and\s+Year\s+)?(20\d{2})", re.IGNORECASE
)
_MONTH_PERIOD_RE = re.compile(r"([A-Za-z]{3,9})\s+(20\d{2})\s*$")


def _clean(text: str) -> str:
    stripped = _TAG_RE.sub(" ", text or "")
    stripped = stripped.replace("&nbsp;", " ").replace("\xa0", " ").replace("&amp;", "&")
    return " ".join(stripped.split())


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


def _to_24h(hour: int, minute: int, meridiem: str) -> tuple[int, int] | None:
    if not (1 <= hour <= 12) or not (0 <= minute <= 59):
        return None
    if meridiem.upper() == "A":
        return (0 if hour == 12 else hour), minute
    return (12 if hour == 12 else hour + 12), minute


def parse_archive_page(html: str) -> list[dict]:
    """Past BEA releases from one archive page.

    Rows look like::

        <tr class="release-row">
          <td class="views-field-title"><a href="...">GDP (Advance Estimate), ...</a></td>
          <td class="views-field-created"><time datetime="2026-02-20T08:30:00-05:00">...</time></td>
        </tr>

    The ``datetime`` attribute is an ISO instant WITH the release hour and an
    explicit offset, so unlike the schedule page there is nothing to infer:
    a row whose timestamp will not parse is skipped, never dated by guess.
    """
    rows: list[dict] = []
    for block in re.findall(r'<tr[^>]*class="[^"]*release-row[^"]*"[^>]*>(.*?)</tr>',
                            html, re.S | re.IGNORECASE):
        title_match = re.search(r'views-field-title[^>]*>\s*<a[^>]*>(.*?)</a>',
                                block, re.S | re.IGNORECASE)
        time_match = re.search(r'<time[^>]*datetime="([^"]+)"', block, re.IGNORECASE)
        if not title_match or not time_match:
            continue
        title = re.sub(r"<[^>]+>", "", title_match.group(1))
        title = html_module.unescape(title).strip()
        try:
            stamp = datetime.fromisoformat(time_match.group(1))
        except ValueError:
            logger.warning("BEA archive row has an unparseable time: %r",
                           time_match.group(1))
            continue
        if stamp.tzinfo is None:
            continue  # no offset means no instant — refuse rather than assume
        rows.append({"title": title, "published_at": stamp.astimezone(timezone.utc)})
    return rows


def classify_release_title(title: str) -> EventType | None:
    """Which typed event a BEA release title denotes, or ``None`` to skip."""
    text = (title or "").strip()
    if _GDP_RE.match(text):
        return EventType.GDP
    if _PCE_RE.match(text):
        return EventType.PCE
    return None


def release_period_code(title: str) -> str | None:
    """``"…, 2nd Quarter 2026"`` -> ``"2026-Q2"``; ``"…, July 2026"`` -> ``"2026-07"``."""
    text = (title or "").strip()
    quarter = _QUARTER_PERIOD_RE.search(text)
    if quarter:
        return f"{int(quarter.group(2)):04d}-Q{int(quarter.group(1))}"
    month = _MONTH_PERIOD_RE.search(text)
    if month:
        number = _MONTHS.get(month.group(1).lower())
        if number:
            return f"{int(month.group(2)):04d}-{number:02d}"
    return None


def parse_schedule_page(html: str) -> list[dict]:
    """Rows of ``{release_date, release_time, time_text, title}``.

    Returns ``[]`` when the table or its year header is missing — without the
    header year no row can be dated, and a guessed year is a fabricated date.
    """
    table = _TABLE_RE.search(html or "")
    if not table:
        return []
    body = table.group(1)
    year_match = _YEAR_HEADER_RE.search(body)
    if not year_match:
        logger.warning(
            "BEA schedule table has no 'Year YYYY' header — refusing to date "
            "rows whose cells carry no year"
        )
        return []
    year = int(year_match.group(1))

    out: list[dict] = []
    previous_month = 0
    for row_html in _ROW_RE.findall(body):
        date_match = _DATE_RE.search(row_html)
        title_match = _TITLE_RE.search(row_html)
        if not date_match or not title_match:
            continue  # the <thead> row, or a layout we do not recognise
        month = _MONTHS.get(date_match.group(1).lower())
        if month is None:
            continue
        # The table runs forward in time; a month that moves BACKWARDS has
        # wrapped past December into the next year.
        if month < previous_month:
            year += 1
        previous_month = month
        try:
            release_date = date(year, month, int(date_match.group(2)))
        except ValueError:
            logger.warning("BEA schedule row has an invalid date: %r", date_match.group(0))
            continue
        time_match = _TIME_RE.search(row_html)
        if not time_match:
            logger.warning(
                "BEA schedule row on %s has no parseable release time — "
                "dropping the row rather than assuming one", release_date,
            )
            continue
        release_time = _to_24h(
            int(time_match.group(1)), int(time_match.group(2)), time_match.group(3)
        )
        if release_time is None:
            continue
        out.append(
            {
                "release_date": release_date,
                "release_time": release_time,
                "time_text": (
                    f"{int(time_match.group(1))}:{time_match.group(2)} "
                    f"{time_match.group(3).upper()}M"
                ),
                "title": _clean(title_match.group(1)),
            }
        )
    return out


class BeaCalendarProvider:
    """GDP and PCE release dates from BEA's public schedule page.

    Keyless — the schedule is free HTML. (BEA's statistics API, which serves
    the ACTUAL numbers, does need a free key; that is a separate concern in
    :mod:`libs.event_calendar.macro_data`.)
    """

    name = "bea"

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str = BEA_BASE_URL,
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

    def _get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"BEA request failed for {endpoint}: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"bea.gov returned HTTP 403 for {endpoint}: the request was "
                "refused (User-Agent or IP block). There is NO synthetic "
                f"fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"bea.gov returned HTTP {response.status_code} for {endpoint}: "
                f"{response.text[:300]}"
            )
        return response.text

    def capabilities(self) -> dict[str, bool | str]:
        """Probe the schedule page. Never raises."""
        report = blank_capabilities()
        try:
            html = self._get_text(SCHEDULE_PATH)
        except CapabilityNotAvailable as exc:
            logger.warning("BEA capability 'macro_calendar' unavailable: %s", exc)
            report["macro_calendar"] = False
        except MarketDataError as exc:
            report["macro_calendar"] = str(exc)
        else:
            if parse_schedule_page(html):
                report["macro_calendar"] = True
            else:
                report["macro_calendar"] = (
                    "bea.gov schedule returned HTTP 200 but no release rows "
                    "could be parsed — the page layout changed"
                )
        return report

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """GDP and PCE releases in ``[start, end]``.

        ``as_of`` is accepted for Protocol conformance and not applied: BEA
        publishes this schedule months ahead (same reasoning as BLS).
        """
        try:
            rows = parse_schedule_page(self._get_text(SCHEDULE_PATH))
        except MarketDataError as exc:
            logger.warning("BEA schedule unavailable: %s", exc)
            return []
        if not rows:
            logger.warning(
                "BEA schedule page yielded no rows — refusing to emit any BEA "
                "event rather than guess dates"
            )
            return []

        url = f"{self.base_url}{SCHEDULE_PATH}"
        out: list[EventCandidate] = []
        for row in rows:
            event_type = classify_release_title(row["title"])
            if event_type is None:
                continue  # regional/annual product — not a market print
            release_date: date = row["release_date"]
            scheduled_at = _et_to_utc(release_date, row["release_time"])
            if not (start <= scheduled_at <= end):
                continue
            out.append(
                EventCandidate(
                    event_key=f"{event_type.value}:{release_date.isoformat()}",
                    event_type=event_type,
                    title=_event_title(event_type, row["title"]),
                    scheduled_at=scheduled_at,
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.GOVERNMENT_AGENCY,
                    source_name=SOURCE_NAME,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=classify_session_et(scheduled_at),
                    source_url=url,
                    agency=AGENCY,
                    release_period=release_period_code(row["title"]),
                    raw={"title": row["title"], "release_time_text": row["time_text"]},
                )
            )

        # PAST releases from the archive. The schedule page above is
        # forward-looking only, so without this GDP/PCE could never acquire
        # the history §15 needs to name a previous comparable event — the
        # calendar would keep answering "no comparable event" for a series
        # that has been published quarterly for decades.
        #
        # Archive failure is a DEGRADATION, not a failure: the upcoming
        # releases parsed above are still returned.
        out.extend(self._archive_candidates(start=start, end=end))
        return out

    def _archive_candidates(
        self, *, start: datetime, end: datetime
    ) -> list[EventCandidate]:
        """Past GDP/PCE releases inside ``[start, end]``, from the archive.

        One pass per product family: the archive's filter selects exactly one,
        so GDP and PCE each need their own. A family that fails to load is
        skipped rather than aborting the rest — partial history beats none.
        """
        found: list[EventCandidate] = []
        seen: set[str] = set()
        for archive_path in ARCHIVE_PATHS:
            found.extend(
                self._archive_family(
                    archive_path, start=start, end=end, seen=seen
                )
            )
        return found

    def _archive_family(
        self,
        archive_path: str,
        *,
        start: datetime,
        end: datetime,
        seen: set[str],
    ) -> list[EventCandidate]:
        """One product family's past releases. ``seen`` is shared across
        families so the same release can never be emitted twice."""
        archive_url = f"{self.base_url}{archive_path}"
        found: list[EventCandidate] = []
        for page in range(ARCHIVE_PAGES):
            path = f"{archive_path}&page={page}"
            try:
                rows = parse_archive_page(self._get_text(path))
            except MarketDataError as exc:
                logger.info("BEA archive page %d unavailable: %s", page, exc)
                break
            if not rows:
                break
            for row in rows:
                event_type = classify_release_title(row["title"])
                if event_type is None:
                    continue
                scheduled_at = row["published_at"]
                if not (start <= scheduled_at <= end):
                    continue
                release_date = scheduled_at.astimezone(EASTERN).date()
                key = f"{event_type.value}:{release_date.isoformat()}"
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    EventCandidate(
                        event_key=key,
                        event_type=event_type,
                        title=_event_title(event_type, row["title"]),
                        scheduled_at=scheduled_at,
                        # CONFIRMED and not ESTIMATED: this release ALREADY
                        # HAPPENED and BEA is stating when. That is a fact,
                        # not a derivation.
                        status=EventStatus.CONFIRMED,
                        source=EventSourceKind.GOVERNMENT_AGENCY,
                        source_name=SOURCE_NAME,
                        event_timezone=US_EVENT_TIMEZONE,
                        session=classify_session_et(scheduled_at),
                        source_url=archive_url,
                        agency=AGENCY,
                        release_period=release_period_code(row["title"]),
                        raw={"title": row["title"], "origin": "archive"},
                    )
                )
        return found

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        raise CapabilityNotAvailable(
            "the Bureau of Economic Analysis does not serve exchange sessions "
            "— use the alpaca_calendar provider for market_calendar"
        )
