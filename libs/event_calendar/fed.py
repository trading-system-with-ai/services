"""Federal Reserve event provider — FOMC calendar + speeches (spec §9).

§9 is explicit: "Do not label every Fed event as an FOMC meeting." This
adapter therefore emits FIVE distinct typed events from two free primary
sources, never one generic "Fed event":

  ``https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm``
    A year panel per year, each row a meeting rendered as a month cell and a
    day cell: ``January 27-28*`` (a two-day meeting; ``*`` marks a meeting
    with a Summary of Economic Projections), ``April 28-29``, and month
    ranges that cross a boundary (``Apr/May`` + ``30-1``). Each meeting
    becomes:

      FOMC_MEETING           day 1, 09:00 ET  — the meeting convenes
      FOMC_DECISION          last day, 14:00 ET — the statement drops; this
                             is the market-moving instant
      FOMC_PRESS_CONFERENCE  last day, 14:30 ET — the Chair's presser
      FOMC_MINUTES           the "Minutes: (Released Month DD, YYYY ...)"
                             link when the page carries one (CONFIRMED,
                             14:00 ET); otherwise decision + 21 days, marked
                             ESTIMATED/DERIVED because three weeks is the
                             convention, not a promise.

    Unscheduled meetings and notation votes are SKIPPED with a log: they are
    not part of the forward calendar and dating them would be a guess.

  ``https://www.federalreserve.gov/feeds/speeches.xml``
    RSS 2.0. ``<title>`` is "Speaker Name: Speech Title"; ``pubDate`` is
    RFC-2822. Each item becomes a FED_SPEECH carrying speaker and topic
    (§9: "Store speaker, topic, event type and official timestamp").

HTML SCRAPERS MUST FAIL LOUDLY AND EMPTY (audit §6): a layout change yields
zero rows plus a warning — never a partially-parsed meeting at the wrong
time, which would put an alert on a day the Fed is not meeting.
"""
import logging
import re
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Sequence
from xml.etree import ElementTree

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
    classify_session_et,
)

logger = logging.getLogger(__name__)

FED_BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_PATH = "/monetarypolicy/fomccalendars.htm"
SPEECHES_RSS_PATH = "/feeds/speeches.xml"
PRESS_MONETARY_RSS_PATH = "/feeds/press_monetary.xml"

DEFAULT_TIMEOUT_SECONDS = 15.0

SOURCE_NAME_FOMC = "fed_fomc"
SOURCE_NAME_RSS = "fed_rss"
SOURCE_NAME_DERIVED = "derived_cadence"

#: Eastern wall-clock anchors. The Fed publishes the statement at 14:00 ET
#: and the Chair's press conference starts at 14:30 ET; the meeting itself
#: convenes in the morning. These are the published conventions — the page
#: does not carry per-meeting times, so they are anchors, not scraped facts.
MEETING_START_ET = (9, 0)
DECISION_ET = (14, 0)
PRESS_CONFERENCE_ET = (14, 30)
MINUTES_ET = (14, 0)

#: Minutes appear three weeks after the decision. Used ONLY when the page has
#: no released-minutes link yet, and the result is ESTIMATED/DERIVED.
MINUTES_LAG_DAYS = 21

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

#: A year panel heading: "2026 FOMC Meetings".
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

#: The month cell: "January", "Apr/May", "January/February".
_MONTH_CELL_RE = re.compile(
    r"^\s*([A-Za-z]{3,9})\s*(?:/\s*([A-Za-z]{3,9}))?\s*$"
)

#: The day cell: "27-28*", "28-29", "17", "30-1" (crossing a month), and the
#: same forms with a trailing footnote marker.
_DAY_CELL_RE = re.compile(
    r"^\s*(\d{1,2})\s*(?:[-–—]\s*(\d{1,2}))?\s*(\*)?\s*$"
)

#: "(Released January 6, 2027)" inside a Minutes link's text.
_RELEASED_RE = re.compile(
    r"released\s+([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})", re.IGNORECASE
)

#: Rows the forward calendar must not turn into scheduled events.
_SKIP_MARKERS = ("unscheduled", "notation vote", "conference call")

#: LIVE layout (verified 2026-08-19): one ``<div class="panel ...">`` per year
#: headed ``<h4><a id="…">2026 FOMC Meetings</a></h4>``; inside, one
#: ``<div class="row fomc-meeting" ">`` per meeting (note the stray quote the
#: Fed's template emits) holding ``fomc-meeting__month`` (``<strong>January</strong>``,
#: or ``April/May`` for a meeting crossing months), ``fomc-meeting__date``
#: ("27-28", "16-17*", "30-1") and, for past meetings, a Minutes cell ending
#: "(Released February 18, 2026)". These regexes read THAT markup directly;
#: the HTMLParser below stays as the fallback for older layouts.
_PANEL_HEAD_RE = re.compile(
    r"<h4>\s*(?:<a[^>]*>)?\s*(20\d{2})\s+FOMC\s+Meetings\s*(?:</a>)?\s*</h4>",
    re.IGNORECASE,
)
_ROW_SPLIT_RE = re.compile(r'<div[^>]*class="[^"]*\brow\s+fomc-meeting\b', re.IGNORECASE)
_ROW_MONTH_RE = re.compile(
    r"fomc-meeting__month[^>]*>\s*(?:<strong>\s*)?([A-Za-z]{3,9}(?:\s*/\s*[A-Za-z]{3,9})?)",
    re.IGNORECASE,
)
_ROW_DATE_RE = re.compile(r"fomc-meeting__date[^>]*>\s*([^<]{1,20}?)\s*<", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _rows_from_markup(html: str) -> list[dict]:
    """Extract ``{year, month, day, minutes}`` rows from the live FOMC page
    markup by class name. Returns [] when the class names are absent so the
    caller can fall back to the generic HTMLParser path."""
    heads = list(_PANEL_HEAD_RE.finditer(html or ""))
    if not heads:
        return []
    rows: list[dict] = []
    for idx, head in enumerate(heads):
        year = int(head.group(1))
        panel_end = heads[idx + 1].start() if idx + 1 < len(heads) else len(html)
        panel = html[head.end():panel_end]
        chunks = _ROW_SPLIT_RE.split(panel)[1:]  # drop the pre-row preamble
        for chunk in chunks:
            month = _ROW_MONTH_RE.search(chunk)
            day = _ROW_DATE_RE.search(chunk)
            if not month or not day:
                continue
            text = _TAG_RE.sub(" ", chunk)
            released = _RELEASED_RE.search(text)
            rows.append(
                {
                    "year": year,
                    "month": month.group(1).strip(),
                    "day": day.group(1).strip(),
                    "minutes": released.group(0) if released else "",
                    "blob": " ".join(text.split()).lower()[:400],
                }
            )
    return rows

#: RSS speech title: "Chair Jerome H. Powell: Economic Outlook".
_SPEECH_TITLE_RE = re.compile(r"^\s*([^:]{2,80}?)\s*:\s*(.+?)\s*$", re.DOTALL)
#: The LIVE feed (verified 2026-08-19) titles items "Surname, Speech Title"
#: (e.g. "Cook, Outlook for the U.S. and Alaskan Economies") with no author
#: element; the speaker is the capitalised word run before the first comma.
_SPEECH_TITLE_COMMA_RE = re.compile(r"^\s*([A-Z][A-Za-z.\-' ]{1,40}?)\s*,\s*(.+?)\s*$", re.DOTALL)
#: Last-resort speaker hint from the speech URL, e.g. .../speech/cook20260805a.htm
_SPEECH_LINK_SLUG_RE = re.compile(r"/speech/([a-z]+)\d{8}[a-z]?\.htm", re.IGNORECASE)

#: Speakers whose remarks move rates markets on their own (spec §13's
#: importance model gives these a bump; the provider just records the fact).
_SENIOR_SPEAKER_MARKERS = ("chair", "vice chair", "powell")


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    """Eastern wall-clock -> UTC instant (DST resolved from the date).

    This is the single most correctness-critical line in the module: a
    January decision is 19:00Z and a July decision is 18:00Z, and getting it
    wrong would put every T-minus alert an hour off for half the year.
    """
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


def _month_number(name: object) -> int | None:
    if not isinstance(name, str):
        return None
    return _MONTHS.get(name.strip().lower().rstrip("."))


def _slug(text: str, limit: int = 40) -> str:
    """Lowercase alnum-and-dashes slug for a deterministic event_key."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return cleaned[:limit].strip("-")


class _FomcCalendarParser(HTMLParser):
    """Collects (year, month_cell, day_cell, minutes_text) tuples.

    Written against the federalreserve.gov layout — panels of
    ``div.panel`` per year, each meeting a ``div.fomc-meeting`` with
    ``.fomc-meeting__month`` / ``.fomc-meeting__date`` children and an
    optional Minutes anchor — but deliberately class-TOLERANT: it tracks the
    most recent year seen in a heading and accumulates the text of every
    element carrying a ``fomc-meeting*`` class, so a renamed wrapper element
    does not silently zero the parse. Anything it cannot resolve to a real
    (year, month, day) is dropped by the caller.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._year: int | None = None
        self._current: dict | None = None
        self._capture: str | None = None
        self._buffer: list[str] = []
        self._depth = 0
        self._meeting_depth: int | None = None
        # Depth at which the current capture container opened. Text keeps
        # accumulating into the same field until that container closes, so a
        # nested <a>/<em> inside the Minutes cell does not truncate the
        # "(Released February 18, 2026)" the parse depends on.
        self._capture_depth: int | None = None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> str:
        for key, value in attrs:
            if key == "class" and value:
                return value
        return ""

    def _flush(self) -> None:
        if self._capture is None or self._current is None:
            self._buffer = []
            self._capture = None
            self._capture_depth = None
            return
        text = " ".join(" ".join(self._buffer).split())
        if text:
            existing = self._current.get(self._capture, "")
            self._current[self._capture] = f"{existing} {text}".strip()
        self._buffer = []
        self._capture = None
        self._capture_depth = None

    def _begin(self, field: str) -> None:
        """Start (or continue) accumulating text into `field`."""
        if self._capture == field:
            return  # already inside this cell; keep one continuous buffer
        self._flush()
        self._capture = field
        self._capture_depth = self._depth

    # -- HTMLParser -------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        classes = self._classes(attrs)
        if "fomc-meeting" in classes and self._meeting_depth is None:
            self._flush()
            self._current = {"year": self._year, "month": "", "day": "", "minutes": ""}
            self._meeting_depth = self._depth
            return
        if self._current is not None:
            if "fomc-meeting__month" in classes:
                self._begin("month")
            elif "fomc-meeting__date" in classes:
                self._begin("day")
            elif "fomc-meeting__minutes" in classes or (
                tag == "a" and self._capture is None
            ):
                self._begin("minutes")
        elif tag in {"h1", "h2", "h3", "h4", "h5"}:
            self._flush()
            self._capture = "heading"
            self._capture_depth = self._depth
            self._current = {"__heading__": True}

    def handle_endtag(self, tag: str) -> None:
        # Only the element that OPENED the capture closes it; inner tags are
        # part of the same cell.
        if self._capture is not None and self._depth <= (self._capture_depth or 0):
            if self._current is not None and self._current.get("__heading__"):
                text = " ".join(" ".join(self._buffer).split())
                match = _YEAR_RE.search(text)
                if match:
                    self._year = int(match.group(1))
                self._buffer = []
                self._capture = None
                self._capture_depth = None
                self._current = None
            else:
                self._flush()
        if self._meeting_depth is not None and self._depth <= self._meeting_depth:
            self._flush()
            if self._current is not None and not self._current.get("__heading__"):
                self.rows.append(self._current)
            self._current = None
            self._meeting_depth = None
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buffer.append(data)

    def close(self) -> None:  # pragma: no cover — trivial finaliser
        super().close()
        if self._current is not None and not self._current.get("__heading__"):
            self.rows.append(self._current)
            self._current = None


def parse_fomc_calendar(html: str) -> list[dict]:
    """FOMC calendar HTML -> ``[{year, start, end, has_sep, minutes_date}]``.

    PURE, so the whole parse is fixture-testable without a network. Returns
    an EMPTY list on a layout it cannot read — the audit's "fail loudly and
    empty", never a partial meeting at a guessed date.
    """
    rows = _rows_from_markup(html or "")
    if not rows:
        parser = _FomcCalendarParser()
        try:
            parser.feed(html or "")
            parser.close()
        except Exception as exc:  # pragma: no cover — html.parser is forgiving
            logger.warning("FOMC calendar HTML could not be parsed: %s", exc)
            return []
        rows = parser.rows

    meetings: list[dict] = []
    for row in rows:
        year = row.get("year")
        month_text = str(row.get("month") or "")
        day_text = str(row.get("day") or "")
        minutes_text = str(row.get("minutes") or "")
        blob = str(row.get("blob") or f"{month_text} {day_text} {minutes_text}").lower()
        if any(marker in blob for marker in _SKIP_MARKERS):
            logger.info(
                "FOMC row skipped (not a scheduled meeting): %r", blob.strip()[:80]
            )
            continue
        if not isinstance(year, int):
            continue
        month_match = _MONTH_CELL_RE.match(month_text)
        day_match = _DAY_CELL_RE.match(day_text)
        if month_match is None or day_match is None:
            logger.debug(
                "FOMC row skipped (unreadable cells): month=%r day=%r",
                month_text, day_text,
            )
            continue
        first_month = _month_number(month_match.group(1))
        second_month = _month_number(month_match.group(2))
        if first_month is None:
            continue
        first_day = int(day_match.group(1))
        last_day = int(day_match.group(2)) if day_match.group(2) else first_day
        try:
            start = date(year, first_month, first_day)
        except ValueError:
            logger.warning(
                "FOMC row skipped (impossible start date): %s %s %s",
                year, month_text, day_text,
            )
            continue
        # A month-crossing meeting: "Apr/May 30-1" ends in the SECOND month,
        # and "December 15-16" of a December/January pair rolls the year.
        end_month = second_month if (second_month and last_day < first_day) else first_month
        end_year = year + 1 if (end_month < first_month) else year
        try:
            end = date(end_year, end_month, last_day)
        except ValueError:
            logger.warning(
                "FOMC row skipped (impossible end date): %s %s %s",
                year, month_text, day_text,
            )
            continue
        if end < start:
            logger.warning(
                "FOMC row skipped (end before start): %s..%s", start, end
            )
            continue
        released = _RELEASED_RE.search(minutes_text)
        minutes_date: date | None = None
        if released:
            m = _month_number(released.group(1))
            if m is not None:
                try:
                    minutes_date = date(int(released.group(3)), m, int(released.group(2)))
                except ValueError:
                    minutes_date = None
        meetings.append(
            {
                "year": year,
                "start": start,
                "end": end,
                "has_sep": bool(day_match.group(3)),
                "minutes_date": minutes_date,
            }
        )
    meetings.sort(key=lambda m: m["start"])
    return meetings


def parse_speeches_rss(xml_text: str) -> list[dict]:
    """Speeches RSS -> ``[{title, speaker, topic, link, published_at}]``.

    PURE. An item without a parseable ``pubDate`` is dropped: a speech with
    no official timestamp cannot be placed on a timeline, and inventing one
    would be exactly the fabrication §7 forbids.
    """
    try:
        root = ElementTree.fromstring(xml_text or "")
    except ElementTree.ParseError as exc:
        logger.warning("Fed speeches RSS could not be parsed: %s", exc)
        return []
    items: list[dict] = []
    for item in root.iter("item"):
        raw_title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()
        if not raw_title or not pub_raw:
            continue
        try:
            published = parsedate_to_datetime(pub_raw)
        except (TypeError, ValueError):
            logger.debug("Fed RSS item skipped: unparseable pubDate %r", pub_raw)
            continue
        if published is None:
            continue
        if published.tzinfo is None:
            # The Fed's feed stamps -0400/-0500; a naive value is its own
            # local clock, not UTC.
            published = published.replace(tzinfo=EASTERN)
        speaker, topic = "", raw_title
        comma_match = _SPEECH_TITLE_COMMA_RE.match(raw_title)
        colon_match = _SPEECH_TITLE_RE.match(raw_title)
        if comma_match and (
            not colon_match or raw_title.find(",") < raw_title.find(":")
        ):
            # Live format "Surname, Title" (the comma precedes any colon that
            # may belong to the speech title itself).
            speaker, topic = comma_match.group(1).strip(), comma_match.group(2).strip()
        elif colon_match:
            speaker, topic = colon_match.group(1).strip(), colon_match.group(2).strip()
        if not speaker:
            description = (item.findtext("description") or "").strip()
            desc_match = _SPEECH_TITLE_RE.match(description)
            if desc_match:
                speaker = desc_match.group(1).strip()
        if not speaker and link:
            slug_match = _SPEECH_LINK_SLUG_RE.search(link)
            if slug_match:
                speaker = slug_match.group(1).capitalize()
        items.append(
            {
                "title": raw_title,
                "speaker": speaker,
                "topic": topic,
                "link": link,
                "published_at": published.astimezone(timezone.utc),
            }
        )
    return items


def is_senior_speaker(speaker: str | None) -> bool:
    """Whether `speaker` is the Chair or a Vice Chair (spec §13 importance)."""
    text = (speaker or "").lower()
    return any(marker in text for marker in _SENIOR_SPEAKER_MARKERS)


class FedProvider:
    """EventCalendarProvider over the Fed's public calendar and RSS feeds."""

    name = "fed"

    def __init__(
        self,
        base_url: str = FED_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = "trading-system-with-ai/0.1 (catalyst research)",
    ) -> None:
        """`transport` is injectable so tests use httpx.MockTransport.

        No credentials: these are public pages. A descriptive User-Agent is
        still sent as a courtesy to a free public service.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": user_agent},
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
    # Transport
    # ------------------------------------------------------------------

    def _get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"Federal Reserve request failed for {endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"federalreserve.gov returned HTTP 403 for {endpoint}: the "
                "request was refused (User-Agent or IP block). There is NO "
                f"synthetic fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"federalreserve.gov returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response.text

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Probe the FOMC calendar page. Never raises.

        ``fed_events`` is True only when the page both fetches AND parses to
        at least one meeting: a 200 that yields zero meetings is a layout
        change, which is an availability problem even though HTTP succeeded.
        """
        report = blank_capabilities()
        try:
            html = self._get_text(FOMC_CALENDAR_PATH)
        except CapabilityNotAvailable as exc:
            logger.warning("Fed capability 'fed_events' unavailable: %s", exc)
            report["fed_events"] = False
        except MarketDataError as exc:
            report["fed_events"] = str(exc)
        else:
            if parse_fomc_calendar(html):
                report["fed_events"] = True
            else:
                report["fed_events"] = (
                    "federalreserve.gov FOMC calendar returned HTTP 200 but no "
                    "meeting rows could be parsed — the page layout changed"
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
        """FOMC events + speeches in ``[start, end]``.

        `tickers` is ignored: Fed events are market-wide (§12 ranks them
        below position/pool/watchlist events, but they apply to everyone).
        Each source is fetched inside its own try: a dead RSS feed must not
        cost the platform the FOMC calendar (§8).
        """
        out: list[EventCandidate] = []
        try:
            out.extend(self.fetch_fomc_events(start, end))
        except MarketDataError as exc:
            logger.warning("FOMC calendar unavailable: %s", exc)
        try:
            out.extend(self.fetch_speeches(start, end, as_of=as_of))
        except MarketDataError as exc:
            logger.warning("Fed speeches RSS unavailable: %s", exc)
        return out

    def fetch_fomc_events(
        self, start: datetime, end: datetime
    ) -> list[EventCandidate]:
        """The four typed FOMC events per scheduled meeting (§9)."""
        meetings = parse_fomc_calendar(self._get_text(FOMC_CALENDAR_PATH))
        if not meetings:
            logger.warning(
                "FOMC calendar page yielded no meetings — refusing to emit "
                "any FOMC event rather than guess dates"
            )
            return []
        out: list[EventCandidate] = []
        for meeting in meetings:
            out.extend(self._meeting_candidates(meeting))
        return [c for c in out if start <= c.scheduled_at <= end]

    def _meeting_candidates(self, meeting: dict) -> list[EventCandidate]:
        first: date = meeting["start"]
        last: date = meeting["end"]
        has_sep: bool = meeting["has_sep"]
        minutes_date: date | None = meeting["minutes_date"]
        url = f"{self.base_url}{FOMC_CALENDAR_PATH}"
        span = (
            first.isoformat() if first == last
            else f"{first.isoformat()}..{last.isoformat()}"
        )
        sep_note = " (with Summary of Economic Projections)" if has_sep else ""

        decision_at = _et_to_utc(last, DECISION_ET)
        candidates = [
            EventCandidate(
                event_key=f"FOMC_MEETING:{first.isoformat()}",
                event_type=EventType.FOMC_MEETING,
                title=f"FOMC meeting ({span}){sep_note}",
                scheduled_at=_et_to_utc(first, MEETING_START_ET),
                status=EventStatus.CONFIRMED,
                source=EventSourceKind.FEDERAL_RESERVE,
                source_name=SOURCE_NAME_FOMC,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.BEFORE_MARKET,
                source_url=url,
                agency="Federal Reserve",
                topic="Summary of Economic Projections" if has_sep else None,
                raw={"start": first.isoformat(), "end": last.isoformat(), "sep": has_sep},
            ),
            EventCandidate(
                event_key=f"FOMC_DECISION:{last.isoformat()}",
                event_type=EventType.FOMC_DECISION,
                title=f"FOMC rate decision{sep_note}",
                scheduled_at=decision_at,
                status=EventStatus.CONFIRMED,
                source=EventSourceKind.FEDERAL_RESERVE,
                source_name=SOURCE_NAME_FOMC,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.DURING_MARKET,
                source_url=url,
                agency="Federal Reserve",
                raw={"meeting_end": last.isoformat(), "sep": has_sep},
            ),
            EventCandidate(
                event_key=f"FOMC_PRESS_CONFERENCE:{last.isoformat()}",
                event_type=EventType.FOMC_PRESS_CONFERENCE,
                title="FOMC Chair press conference",
                scheduled_at=_et_to_utc(last, PRESS_CONFERENCE_ET),
                status=EventStatus.CONFIRMED,
                source=EventSourceKind.FEDERAL_RESERVE,
                source_name=SOURCE_NAME_FOMC,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.DURING_MARKET,
                source_url=url,
                agency="Federal Reserve",
                raw={"meeting_end": last.isoformat()},
            ),
        ]

        if minutes_date is not None:
            minutes_day, minutes_status = minutes_date, EventStatus.CONFIRMED
            minutes_source = EventSourceKind.FEDERAL_RESERVE
            minutes_source_name = SOURCE_NAME_FOMC
            minutes_title = "FOMC minutes"
        else:
            # Three weeks is the Fed's convention, not a published date, so
            # the row says ESTIMATED/DERIVED and never alerts (§11).
            minutes_day = last + timedelta(days=MINUTES_LAG_DAYS)
            minutes_status = EventStatus.ESTIMATED
            minutes_source = EventSourceKind.DERIVED
            minutes_source_name = SOURCE_NAME_DERIVED
            minutes_title = "FOMC minutes (estimated: 3 weeks after the decision)"
        candidates.append(
            EventCandidate(
                event_key=f"FOMC_MINUTES:{minutes_day.isoformat()}",
                event_type=EventType.FOMC_MINUTES,
                title=minutes_title,
                scheduled_at=_et_to_utc(minutes_day, MINUTES_ET),
                status=minutes_status,
                source=minutes_source,
                source_name=minutes_source_name,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.DURING_MARKET,
                source_url=url,
                agency="Federal Reserve",
                raw={
                    "meeting_end": last.isoformat(),
                    "released": minutes_date.isoformat() if minutes_date else None,
                },
            )
        )
        return candidates

    def fetch_speeches(
        self,
        start: datetime,
        end: datetime,
        *,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """FED_SPEECH candidates from the speeches RSS feed (§9)."""
        items = parse_speeches_rss(self._get_text(SPEECHES_RSS_PATH))
        out: list[EventCandidate] = []
        for item in items:
            published: datetime = item["published_at"]
            if as_of is not None and published > as_of:
                continue  # §96: not published yet at as_of
            if not (start <= published <= end):
                continue
            speaker = item["speaker"] or "Federal Reserve"
            day = published.astimezone(EASTERN).date()
            out.append(
                EventCandidate(
                    event_key=(
                        f"FED_SPEECH:{day.isoformat()}:{_slug(speaker)}:"
                        f"{_slug(item['topic'])}"
                    ),
                    event_type=EventType.FED_SPEECH,
                    title=item["title"],
                    scheduled_at=published,
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.FEDERAL_RESERVE,
                    source_name=SOURCE_NAME_RSS,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=classify_session_et(published),
                    source_url=item["link"] or None,
                    agency="Federal Reserve",
                    speaker=speaker,
                    topic=item["topic"],
                    raw={"senior_speaker": is_senior_speaker(speaker)},
                )
            )
        return out

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """The Fed publishes its own calendar, not the exchange's."""
        raise CapabilityNotAvailable(
            "the Federal Reserve does not serve exchange sessions — use the "
            "alpaca_calendar provider for market_calendar"
        )
