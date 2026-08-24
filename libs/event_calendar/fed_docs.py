"""Federal Reserve DOCUMENTS provider — statements, minutes, speeches (§42-§45).

:mod:`libs.event_calendar.fed` answers *when*: it scrapes the FOMC calendar
and the speeches feed into dated ``EventCandidate`` rows. This module answers
*what was actually said*. They are deliberately separate adapters over the
same host because they fail differently and are called at different times: a
calendar scrape runs every ingestion tick and must never be blocked by a
statement page that has not been published yet, and a document fetch runs
once per meeting and must store text VERBATIM.

§44 is the rule that shapes every decision here: **the source document is
authoritative**. This provider therefore

* stores the statement's paragraphs exactly as the Fed wrote them (entities
  decoded, whitespace collapsed inside a paragraph, nothing paraphrased),
* parses only what can be read literally off the page — the vote line, the
  target range, the dissenter names — and returns ``None`` rather than a
  guess when the sentence the Fed used does not match,
* NEVER derives a hawkish/dovish score (§43). Scoring belongs nowhere in
  this codebase; dimensions are reported separately by
  :mod:`libs.trading_core.events.fed_intel`, which reads what this module
  stores.

LIVE LAYOUT (verified 2026-08-19, fixtures in tests/fixtures/events/):

``/newsevents/pressreleases/monetary{YYYYMMDD}a.htm``
    ``<div id="article">`` wrapping a ``div.heading.col-xs-12.col-sm-8.col-md-8``
    (``p.article__time`` "July 29, 2026", ``h3.title`` "Federal Reserve issues
    FOMC statement", ``p.releaseTime`` "For release at 2:00 p.m. EDT", plus a
    social-share ``<ul>`` that is pure navigation) and a sibling
    ``div.col-xs-12.col-sm-8.col-md-8`` holding the statement's ``<p>``
    paragraphs. The body ends with a "For media inquiries" paragraph and an
    "Implementation Note issued …" link — both boilerplate, both stripped.

``/monetarypolicy/fomcminutes{YYYYMMDD}.htm``
    ``<div id="article" class="col-xs-12 col-sm-8 col-md-9">`` (note: a
    DIFFERENT class list from the press release — hence the container match is
    by ``id="article"`` first, class second), sections introduced by a leading
    ``<strong>`` inside the paragraph ("Developments in Financial Markets and
    Open Market Operations", "Committee Policy Actions", …), and numbered
    footnotes at the end.

``/feeds/press_monetary.xml``
    RSS 2.0, CDATA everywhere. "Federal Reserve issues FOMC statement" is the
    statement release; "Minutes of the Federal Open Market Committee, July
    28-29, 2026" is the minutes release — and note that its ``<link>`` points
    at a PRESS RELEASE page (``monetary20260819a.htm``), not at the
    ``fomcminutes…`` document, so the RSS is used for the released_at INSTANT
    and the canonical URL is still built from the meeting date.
    "Minutes of the Board's discount rate meetings on …" is a different
    document entirely and classifies as OTHER, never MINUTES.

Boilerplate stripping is conservative on purpose: dropping a real statement
paragraph would corrupt the §44 diff, so a paragraph is dropped only when it
matches one of the few known navigation/footer shapes.
"""
import html as html_module
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from xml.etree import ElementTree

import httpx

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError

from .fed import DECISION_ET, FED_BASE_URL, PRESS_MONETARY_RSS_PATH, _et_to_utc
from .provider import EASTERN, CalendarProviderError, blank_capabilities

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 20.0

#: Path templates. ``a`` is the Fed's first-release-of-the-day suffix; the
#: statement is always the "a" release on a decision day.
STATEMENT_PATH_TEMPLATE = "/newsevents/pressreleases/monetary{yyyymmdd}a.htm"
MINUTES_PATH_TEMPLATE = "/monetarypolicy/fomcminutes{yyyymmdd}.htm"

SOURCE_NAME = "fed_docs"

DOC_TYPE_STATEMENT = "STATEMENT"
DOC_TYPE_MINUTES = "MINUTES"
DOC_TYPE_SPEECH = "SPEECH"

RSS_KIND_STATEMENT = "STATEMENT"
RSS_KIND_MINUTES = "MINUTES"
RSS_KIND_OTHER = "OTHER"


class FedDocsError(CalendarProviderError):
    """A Fed document request could not be answered honestly.

    Subclasses the calendar error (itself a ``MarketDataError``) so the
    gateway seam's single ``except MarketDataError`` keeps working — a
    document fetch failing is the same class of event as a calendar scrape
    failing, and neither may be papered over with placeholder text.
    """


class DocumentNotFound(FedDocsError):
    """HTTP 404 — the document does not exist (yet) at that URL.

    Distinguished from the generic error because it is the EXPECTED answer
    for a meeting whose statement has not dropped: the caller records
    "not published" rather than "the Fed is down".
    """


# ---------------------------------------------------------------------------
# Parsed shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FomcDocument:
    """A Fed document reduced to verbatim text plus its provenance.

    ``text`` is ``"\\n\\n".join(paragraphs)`` — reconstructable, so storing
    both is redundant by design: ``paragraphs`` is what the sentence-level
    diff consumes and ``text`` is what a human (or the LLM) reads, and the
    two can never drift because the constructor derives one from the other.
    """

    doc_type: str
    url: str
    title: str
    paragraphs: list[str]
    text: str
    released_at: datetime | None = None
    meeting_date: date | None = None
    speaker: str | None = None
    raw_html_len: int = 0
    source_name: str = SOURCE_NAME

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "url": self.url,
            "title": self.title,
            "paragraphs": list(self.paragraphs),
            "text": self.text,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "meeting_date": self.meeting_date.isoformat() if self.meeting_date else None,
            "speaker": self.speaker,
            "raw_html_len": self.raw_html_len,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class FomcStatement(FomcDocument):
    """A policy statement: the document plus the two facts it states outright.

    ``vote`` and ``target_range`` are parsed, never inferred. ``vote["text"]``
    and ``target_range["text"]`` carry the sentence they were read from so the
    UI can show the source line next to the number (§44: the document, not our
    summary of it, is the authority).
    """

    vote: dict = field(default_factory=dict)
    target_range: dict | None = None

    def to_dict(self) -> dict:
        out = super().to_dict()
        out["vote"] = dict(self.vote)
        out["target_range"] = dict(self.target_range) if self.target_range else None
        return out


@dataclass(frozen=True)
class RssItem:
    """One ``press_monetary.xml`` entry, classified by what it announces."""

    title: str
    url: str
    published_at: datetime
    kind: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "kind": self.kind,
        }


# ---------------------------------------------------------------------------
# HTML -> paragraphs
# ---------------------------------------------------------------------------

#: Paragraph shapes that are page furniture rather than document text. Kept
#: SHORT and anchored: a broad filter would silently eat a real paragraph and
#: corrupt the diff, which is a worse failure than leaving one boilerplate
#: line in the text.
_BOILERPLATE_PREFIXES = (
    "for media inquiries",
    "implementation note issued",
    "for release at",
    "last update",
    "share",
    "return to text",
)

_BOILERPLATE_EXACT = ("", " ")

#: "Last Update: July 29, 2026" and the footnote back-links.
_FOOTNOTE_RE = re.compile(r"^\d+\.\s")

#: Tags whose text is navigation, never document body.
_SKIP_CONTAINERS = {"script", "style", "ul", "ol", "nav", "form", "select", "button"}

#: Block tags that end a paragraph even without a closing </p>.
_PARAGRAPH_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li"}


class _ArticleParser(HTMLParser):
    """Collects ``<p>``/heading text from the Fed's article container.

    Two-stage on purpose. First it looks for the container the Fed actually
    uses — ``id="article"`` — and collects only paragraphs beneath it; once
    that container has been seen, nothing outside it is ever collected again.
    That ordering matters: the full page wraps the article in a site shell
    whose header ("An official website of the United States Government"), nav
    blurb and footer address are ALSO ``col-sm-8`` columns, so a class-only
    match pulls six chrome paragraphs into what the §44 diff treats as the
    Committee's words.

    Only when no ``id="article"`` exists at all — a layout change, or a caller
    handing us a pre-trimmed fragment — does the class heuristic, and then
    :func:`_paragraphs_anywhere`, take over. That fallback is why a rename
    degrades to "slightly noisier text" instead of "empty document", which for
    a diff is the difference between a usable answer and a wrong one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        #: ``(inside_article, text)`` pairs. The flag is what lets the
        #: finaliser DISCARD the site chrome the shell emits before
        #: ``id="article"`` opens — "An official website of the United States
        #: Government" and the .gov/HTTPS explainers are ``<p>`` elements too,
        #: and four of them ahead of the statement would be diffed as
        #: paragraphs the Committee added.
        self.paragraphs: list[tuple[tuple[bool, bool], str]] = []
        self.headings: list[tuple[tuple[bool, bool], str]] = []
        self.article_time: str = ""
        self.release_time: str = ""
        self._depth = 0
        self._article_depth: int | None = None
        #: Set once ``id="article"`` has been entered. From then on the
        #: class heuristic is disabled and paragraphs outside the article are
        #: dropped — the site chrome must never reach the diff.
        self._article_seen = False
        #: True while inside the ``id="article"`` container specifically.
        self._strict = False
        self._skip_depth: int | None = None
        self._buffer: list[str] = []
        self._in_paragraph: str | None = None
        self._paragraph_class: str = ""

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
        for key, value in attrs:
            if key == name:
                return value or ""
        return ""

    def _flush(self) -> None:
        tag, self._in_paragraph = self._in_paragraph, None
        text = _normalize_space("".join(self._buffer))
        self._buffer = []
        cls = self._paragraph_class
        self._paragraph_class = ""
        if not text:
            return
        if "article__time" in cls:
            self.article_time = self.article_time or text
            return
        if "releaseTime" in cls:
            self.release_time = self.release_time or text
            return
        # Recorded as TWO independent facts because ``id="article"`` may not
        # have been seen yet when this paragraph flushes: the shell's header
        # column closes BEFORE the article opens, so a single boolean decided
        # here is decided too early. :meth:`resolve` picks between them once
        # the whole document has been read.
        inside = (self._strict, self._article_depth is not None)
        if tag and tag.startswith("h"):
            self.headings.append((inside, text))
            return
        self.paragraphs.append((inside, text))

    #: Tags that never nest and that the Fed routinely leaves unclosed
    #: (``<p class="releaseTime">For release at 2:00 p.m. EDT`` has no
    #: ``</p>`` on the live page). Counting them would drift the depth by one
    #: for the rest of the document, which is how the article container ends
    #: up "never closing" and the page footer ends up inside the statement.
    _VOID_FOR_DEPTH = {"p", "br", "img", "input", "meta", "link", "hr"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self._VOID_FOR_DEPTH:
            self._depth += 1
        if self._skip_depth is not None:
            return
        if tag in _SKIP_CONTAINERS:
            # A share menu or a script: everything inside is furniture.
            if self._in_paragraph:
                self._flush()
            self._skip_depth = self._depth
            return
        if self._attr(attrs, "id") == "article":
            # ALWAYS wins, even over a class-matched container already open:
            # the shell's header column opens BEFORE the article does, so
            # first-match-wins would anchor on the chrome. ``_strict`` marks
            # everything from here on as document text, and :meth:`resolve`
            # then discards every paragraph collected without that mark —
            # including the ones already flushed from the header column.
            self._article_depth = self._depth
            self._article_seen = True
            self._strict = True
        elif self._article_depth is None and not self._article_seen:
            # Only reached on a page with no id="article" at all.
            if "col-sm-8" in self._attr(attrs, "class"):
                self._article_depth = self._depth
        if tag in _PARAGRAPH_TAGS:
            if self._in_paragraph:
                self._flush()
            self._in_paragraph = tag
            self._paragraph_class = self._attr(attrs, "class")
        elif tag == "br" and self._in_paragraph:
            # Minutes put the section heading and its first sentence in ONE
            # <p> separated by <br/>; a space keeps them one paragraph, which
            # is what the source document does.
            self._buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._in_paragraph == tag:
            self._flush()
        if tag in self._VOID_FOR_DEPTH:
            return
        if self._skip_depth is not None and self._depth <= self._skip_depth:
            self._skip_depth = None
        self._depth = max(0, self._depth - 1)
        # AFTER the decrement: ``_article_depth`` was recorded as the depth
        # the container's own start tag was at, so the container is left only
        # once the depth drops BELOW it. Checking before the decrement leaves
        # ``_strict`` stuck on and lets the page footer count as document text.
        if self._article_depth is not None and self._depth < self._article_depth:
            self._article_depth = None
            self._strict = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth is not None or not self._in_paragraph:
            return
        self._buffer.append(data)

    # -- results ----------------------------------------------------------

    def resolve(self) -> tuple[list[str], list[str]]:
        """``(paragraphs, headings)`` — article-only when there was an article.

        When ``id="article"`` was found anywhere on the page, ONLY what sat
        inside it survives. When it was not, everything does, and the caller's
        boilerplate filter is the only cleanup — the honest degradation for an
        unrecognised layout.
        """
        for index in (0, 1) if self._article_seen else (1,):
            body = [text for flags, text in self.paragraphs if flags[index]]
            if body:
                return (
                    body,
                    [text for flags, text in self.headings if flags[index]],
                )
        # Neither filter kept anything: hand back everything and let the
        # caller's boilerplate filter do what it can. An empty result would
        # diff as "the whole statement was removed".
        return (
            [text for _, text in self.paragraphs],
            [text for _, text in self.headings],
        )

    def close(self) -> None:  # pragma: no cover — trivial finaliser
        super().close()
        if self._in_paragraph:
            self._flush()


_TAG_RE = re.compile(r"<[^>]+>")
_P_BLOCK_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def _normalize_space(text: str) -> str:
    """Collapse runs of whitespace (including NBSP) to single spaces."""
    return " ".join((text or "").replace(" ", " ").split()).strip()


def _paragraphs_anywhere(html: str) -> list[str]:
    """Fallback: every ``<p>`` on the page, tags stripped.

    Used when the article container cannot be located. Deliberately dumb — the
    boilerplate filter downstream does the cleanup — because a clever
    heuristic that guessed wrong would drop statement text.
    """
    out: list[str] = []
    for match in _P_BLOCK_RE.finditer(html or ""):
        text = _normalize_space(html_module.unescape(_TAG_RE.sub(" ", match.group(1))))
        if text:
            out.append(text)
    return out


def _is_boilerplate(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _BOILERPLATE_EXACT:
        return True
    if _FOOTNOTE_RE.match(lowered) and "return to text" in lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES)


def parse_article(html: str) -> dict:
    """HTML -> ``{title, paragraphs, headings, article_time, release_time}``.

    PURE and stdlib-only (``html.parser``): no bs4/lxml dependency creeps into
    a library that a scheduled ingestion job imports.
    """
    parser = _ArticleParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception as exc:  # pragma: no cover — HTMLParser is forgiving
        logger.warning("Fed document HTML could not be parsed: %s", exc)

    body, headings = parser.resolve()
    title = next((h for h in headings if h), "")
    paragraphs = [p for p in body if not _is_boilerplate(p)]
    if len(paragraphs) < 2:
        # The container was not found (or was empty): take every <p>. The
        # largest <p>-dense block IS the whole page here, because the fetch
        # already narrowed us to one document.
        fallback = [p for p in _paragraphs_anywhere(html) if not _is_boilerplate(p)]
        if len(fallback) > len(paragraphs):
            paragraphs = fallback
    return {
        "title": title,
        "paragraphs": paragraphs,
        "headings": headings,
        "article_time": parser.article_time,
        "release_time": parser.release_time,
    }


# ---------------------------------------------------------------------------
# Statement facts: the vote and the target range
# ---------------------------------------------------------------------------

#: "…approved the following statement for release by a 9 – 3 vote" — the Fed
#: uses an EN DASH surrounded by spaces on the live page, a hyphen on older
#: ones, so the separator class covers both.
_VOTE_COUNT_RE = re.compile(
    r"by\s+an?\s+(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+vote", re.IGNORECASE
)

#: "Voting against the monetary policy action were Beth M. Hammack, Neel
#: Kashkari, and Lorie K. Logan, who preferred…" / "Voting against this
#: action: None."
#:
#: The name list ends at ", who …" — the reason clause the Fed appends to
#: every dissent — or at the end of the paragraph. It deliberately does NOT
#: end at a period: the names carry middle initials ("Beth M. Hammack"), and a
#: period terminator truncates the list to "Beth M", which would then be
#: stored and rendered as the committee's dissent. A trailing "." is stripped
#: by :func:`_split_names` instead, where it is unambiguous.
_AGAINST_RE = re.compile(
    r"voting\s+against\s+(?:the\s+monetary\s+policy\s+action|this\s+action)\s*"
    r"(?:were|was|:)?\s*(.+?)(?:,\s+who\b|$)",
    re.IGNORECASE | re.DOTALL,
)

#: "…maintain the target range for the federal funds rate at 3-1/2 to 3-3/4
#: percent" and the ``lower/raise … to X to Y percent`` forms.
_TARGET_RANGE_RE = re.compile(
    r"target range for the federal funds rate (?:at|to)\s+"
    r"([0-9]+(?:-[0-9]+/[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s+to\s+"
    r"([0-9]+(?:-[0-9]+/[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s+percent",
    re.IGNORECASE,
)

_FRACTION_RE = re.compile(r"^(\d+)(?:-(\d+)/(\d+))?$")


def parse_fed_fraction(token: str) -> float | None:
    """``"3-3/4"`` -> ``3.75``; ``"4"`` -> ``4.0``; ``"3.75"`` -> ``3.75``.

    The Fed writes rates as mixed fractions, and a float comparison of
    "3-1/2" against "3.5" is the kind of silent mismatch that would report a
    rate change where none happened.
    """
    token = (token or "").strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        pass
    match = _FRACTION_RE.match(token)
    if not match:
        return None
    whole = int(match.group(1))
    if match.group(2) is None:
        return float(whole)
    numerator, denominator = int(match.group(2)), int(match.group(3))
    if denominator == 0:
        return None
    return whole + numerator / denominator


def parse_vote(paragraphs: list[str]) -> dict:
    """The vote as the statement states it.

    Returns ``{"for": int|None, "against": int|None, "dissenters": [...],
    "text": str}``. Every field is independently optional: an older statement
    carries the dissent sentence without the "by a 9 – 3 vote" preamble, and a
    unanimous one carries neither, so ``for``/``against`` stay ``None`` rather
    than being back-filled from a committee-size assumption.
    """
    vote: dict = {"for": None, "against": None, "dissenters": [], "text": ""}
    lines: list[str] = []
    for para in paragraphs:
        count = _VOTE_COUNT_RE.search(para)
        if count and vote["for"] is None:
            vote["for"] = int(count.group(1))
            vote["against"] = int(count.group(2))
            lines.append(para)
        against = _AGAINST_RE.search(para)
        if against:
            names = _split_names(against.group(1))
            if names:
                vote["dissenters"] = names
            if para not in lines:
                lines.append(para)
    vote["text"] = "\n\n".join(lines)
    vote["unanimous"] = (
        vote["against"] == 0 if vote["against"] is not None else (not vote["dissenters"])
    ) and bool(lines or vote["for"] is not None)
    return vote


_NAME_SPLIT_RE = re.compile(r",\s*and\s+|\s+and\s+|,\s*", re.IGNORECASE)


def _split_names(blob: str) -> list[str]:
    """"A, B, and C" -> ``["A", "B", "C"]``; "None" -> ``[]``."""
    cleaned = _normalize_space(blob).rstrip(".")
    if not cleaned or cleaned.lower() in {"none", "no one"}:
        return []
    parts = [p.strip(" .") for p in _NAME_SPLIT_RE.split(cleaned)]
    return [p for p in parts if p and p.lower() != "none"]


def parse_target_range(paragraphs: list[str]) -> dict | None:
    """``{"low_pct", "high_pct", "text"}`` or ``None`` when unstated.

    ``None`` is a real answer: a statement that does not restate the range
    (they exist) must not be given last meeting's range, or the rate-change
    computation would report a hold that the document never claimed.
    """
    for para in paragraphs:
        match = _TARGET_RANGE_RE.search(para)
        if not match:
            continue
        low = parse_fed_fraction(match.group(1))
        high = parse_fed_fraction(match.group(2))
        if low is None or high is None:
            continue
        return {"low_pct": low, "high_pct": high, "text": para}
    return None


# ---------------------------------------------------------------------------
# Minutes sections
# ---------------------------------------------------------------------------

#: A minutes section heading is a paragraph that OPENS with a ``<strong>``:
#: ``<p><strong>Committee Policy Actions</strong><br/>In support of…``. The
#: Fed uses no ``<h4>`` inside the minutes body, so the bold run is the only
#: structural marker there is. Anchored to "opens with" so the many inline
#: ``<strong>`` spans inside body prose are not mistaken for headings.
#: ``[^>]*?(?<!center")>`` excludes the centred signature block
#: (``<p style="text-align:center"><strong>Joshua Gallin</strong>``), which is
#: the Secretary's name, not a section.
_MINUTES_SECTION_RE = re.compile(
    r"<p\b(?![^>]*text-align:\s*center)[^>]*>"
    r"\s*(?:<a\b[^>]*>\s*</a>\s*)?((?:<strong\b[^>]*>.*?</strong>\s*)+)",
    re.IGNORECASE | re.DOTALL,
)

#: Bold runs that are not section headings: the meeting's date line, and the
#: vote sentences, which belong to the "Committee Policy Actions" section.
_NOT_A_SECTION = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+\d|voting\s+(?:for|against)\b|none\b|\W*$)", re.IGNORECASE
)


def article_fragment(html: str) -> str:
    """The ``<div id="article">…</div>`` slice, or the whole document.

    Regex-scoped rather than parsed because the caller wants the RAW markup
    (the bold section headings), and because the Fed's own template leaves
    tags unclosed — a strict parse would be less robust here, not more. Div
    nesting is counted, so an inner ``</div>`` does not truncate the slice.
    """
    text = html or ""
    start = text.lower().find('id="article"')
    if start < 0:
        return text
    open_at = text.rfind("<", 0, start)
    if open_at < 0:
        return text
    depth = 0
    for match in re.finditer(r"<(/?)div\b[^>]*>", text[open_at:], re.IGNORECASE):
        depth += -1 if match.group(1) else 1
        if depth == 0:
            return text[open_at : open_at + match.end()]
    return text[open_at:]


def minutes_sections(html: str) -> list[str]:
    """Section headings of an FOMC minutes page, in document order.

    Deterministic and derived from the document itself — never a fixed list of
    expected section names, which would silently drop a section the Fed added
    and report the minutes as complete anyway. Scoped to the article container
    first: the site's own nav menus are bold-in-paragraph too, and unscoped
    this returns forty-eight "sections", most of them links in the sidebar.
    """
    out: list[str] = []
    for match in _MINUTES_SECTION_RE.finditer(article_fragment(html)):
        heading = _normalize_space(
            html_module.unescape(_TAG_RE.sub("", match.group(1)))
        ).rstrip(":. ")
        if not heading or _NOT_A_SECTION.match(heading):
            continue
        if heading not in out:
            out.append(heading)
    return out


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

_RSS_STATEMENT_RE = re.compile(r"issues\s+FOMC\s+statement", re.IGNORECASE)
#: "Minutes of the Federal Open Market Committee, July 28–29, 2026". The
#: qualifier matters: "Minutes of the Board's discount rate meetings on …" is
#: a DIFFERENT document that appears in the same feed and must not be mistaken
#: for FOMC minutes.
_RSS_MINUTES_RE = re.compile(
    r"minutes\s+of\s+the\s+federal\s+open\s+market\s+committee", re.IGNORECASE
)
#: The meeting dates inside a minutes title: "July 28–29, 2026",
#: "March 17-18, 2026", "April 28-29, 2026" (en dash or hyphen).
_RSS_MINUTES_DATES_RE = re.compile(
    r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:\s*[-–—]\s*(\d{1,2}))?,?\s+(20\d{2})"
)


def classify_press_monetary(title: str) -> str:
    """RSS title -> ``STATEMENT`` / ``MINUTES`` / ``OTHER``."""
    text = title or ""
    if _RSS_STATEMENT_RE.search(text):
        return RSS_KIND_STATEMENT
    if _RSS_MINUTES_RE.search(text):
        return RSS_KIND_MINUTES
    return RSS_KIND_OTHER


def parse_press_monetary_rss(xml_text: str) -> list[RssItem]:
    """``press_monetary.xml`` -> classified items, newest first.

    PURE. An item without a parseable ``pubDate`` is DROPPED: the whole point
    of consulting the feed is to learn the release INSTANT (a statement is
    only visible once released, §14), and an item with no timestamp cannot
    answer that.
    """
    try:
        root = ElementTree.fromstring((xml_text or "").lstrip("﻿"))
    except ElementTree.ParseError as exc:
        logger.warning("Fed press_monetary RSS could not be parsed: %s", exc)
        return []
    items: list[RssItem] = []
    for node in root.iter("item"):
        title = _normalize_space(node.findtext("title") or "")
        url = (node.findtext("link") or "").strip()
        pub_raw = (node.findtext("pubDate") or "").strip()
        if not title or not pub_raw:
            continue
        try:
            published = parsedate_to_datetime(pub_raw)
        except (TypeError, ValueError):
            logger.debug("Fed RSS item skipped: unparseable pubDate %r", pub_raw)
            continue
        if published is None:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=EASTERN)
        items.append(
            RssItem(
                title=title,
                url=url,
                published_at=published.astimezone(timezone.utc),
                kind=classify_press_monetary(title),
            )
        )
    items.sort(key=lambda item: item.published_at, reverse=True)
    return items


def minutes_meeting_dates(title: str) -> tuple[date, date] | None:
    """Meeting start/end dates named in an FOMC minutes RSS title.

    "Minutes of the Federal Open Market Committee, July 28–29, 2026" ->
    ``(2026-07-28, 2026-07-29)``. Used to match a minutes RELEASE back to the
    meeting whose minutes they are, which is how a released_at instant is
    attached to a document whose own page carries no machine-readable time.
    """
    match = _RSS_MINUTES_DATES_RE.search(title or "")
    if not match:
        return None
    from .fed import _month_number

    month = _month_number(match.group(1))
    if not month:
        return None
    year = int(match.group(4))
    try:
        start = date(year, month, int(match.group(2)))
        end = date(year, month, int(match.group(3))) if match.group(3) else start
    except ValueError:
        return None
    return (start, end)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FedDocumentsProvider:
    """Fetches and parses FOMC statements, minutes and speeches.

    NOT an ``EventCalendarProvider``: it serves documents, not dated
    candidates, and is therefore absent from ``_PROVIDERS`` in the package
    registry (which maps names to calendar sources). It is reached through
    :func:`libs.event_calendar.fed_documents_provider` instead, so a caller
    cannot accidentally get it back from ``get_provider("fed")`` and then find
    ``fetch_events`` missing.

    Every request carries the operator's contact User-Agent — the Fed asks
    for one and returns 403 to anonymous scrapers.
    """

    name = "fed_docs"

    def __init__(
        self,
        user_agent: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        base_url: str = FED_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        agent = (user_agent or "").strip()
        if not agent:
            raise ValueError(
                "FedDocumentsProvider needs a contact User-Agent "
                "(settings.sec_user_agent); federalreserve.gov refuses "
                "anonymous requests"
            )
        self.user_agent = agent
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": agent},
            follow_redirects=True,
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    # -- transport --------------------------------------------------------

    def _get_text(self, url: str) -> str:
        endpoint = httpx.URL(url).path
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise FedDocsError(
                f"Federal Reserve document request failed for {endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 404:
            raise DocumentNotFound(
                f"federalreserve.gov has no document at {endpoint} (HTTP 404) "
                "— it has most likely not been published yet"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"federalreserve.gov returned HTTP 403 for {endpoint}: the "
                "request was refused (User-Agent or IP block). There is NO "
                f"synthetic fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise FedDocsError(
                f"federalreserve.gov returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response.text

    # -- URL builders -----------------------------------------------------

    def statement_url(self, decision_date: date) -> str:
        """The statement page for a decision date (the meeting's LAST day)."""
        return self.base_url + STATEMENT_PATH_TEMPLATE.format(
            yyyymmdd=decision_date.strftime("%Y%m%d")
        )

    def minutes_url(self, meeting_end_date: date) -> str:
        """The minutes page, keyed by the meeting's END date, not its release."""
        return self.base_url + MINUTES_PATH_TEMPLATE.format(
            yyyymmdd=meeting_end_date.strftime("%Y%m%d")
        )

    # -- capabilities -----------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state report over the fixed capability keys. Never raises.

        Probes the press-monetary feed rather than a statement page: the feed
        exists on every day of the year, whereas a statement URL is a 404 on
        every day but eight, and a probe that 404s two-thirds of the time
        would report a healthy source as broken.
        """
        report = blank_capabilities()
        try:
            xml_text = self._get_text(self.base_url + PRESS_MONETARY_RSS_PATH)
        except CapabilityNotAvailable as exc:
            logger.warning("Fed documents unavailable (403): %s", exc)
            report["fed_events"] = False
        except MarketDataError as exc:
            report["fed_events"] = str(exc)
        else:
            if parse_press_monetary_rss(xml_text):
                report["fed_events"] = True
            else:
                report["fed_events"] = (
                    "federalreserve.gov press_monetary.xml returned HTTP 200 "
                    "but no items could be parsed — the feed format changed"
                )
        return report

    # -- RSS --------------------------------------------------------------

    def list_press_monetary(self, as_of: datetime | None = None) -> list[RssItem]:
        """Monetary-policy press releases, newest first, cut at ``as_of``.

        ``as_of`` implements §14: a replay standing at a past instant must not
        see a release published after it, so items are filtered on
        ``published_at``, the Fed's own timestamp — never on fetch time.
        """
        items = parse_press_monetary_rss(
            self._get_text(self.base_url + PRESS_MONETARY_RSS_PATH)
        )
        if as_of is None:
            return items
        cut = _as_utc(as_of)
        return [item for item in items if item.published_at <= cut]

    def released_at_for(self, url: str, *, items: list[RssItem] | None = None):
        """The RSS ``pubDate`` for a document URL, or ``None`` if not listed.

        The feed only reaches back a few months, so ``None`` here means "the
        official instant is unknown", which the caller renders as the 14:00 ET
        convention marked as such — never as a fact.
        """
        for item in items or []:
            if item.url == url:
                return item.published_at
        return None

    # -- documents --------------------------------------------------------

    def fetch_statement(
        self,
        decision_date: date,
        *,
        as_of: datetime | None = None,
        rss_items: list[RssItem] | None = None,
    ) -> FomcStatement | None:
        """The policy statement for ``decision_date``.

        ``released_at`` prefers the RSS ``pubDate`` (the Fed's own publication
        instant) and falls back to the 14:00 ET convention that
        :mod:`libs.event_calendar.fed` already uses for the FOMC_DECISION
        event — the two must agree, hence the shared ``DECISION_ET`` constant
        and ``_et_to_utc`` rather than a second copy of the DST arithmetic.

        Returns ``None`` when the statement was released after ``as_of``: a
        point-in-time replay must not be able to read a document that did not
        exist yet (§14/§96). Raises :class:`DocumentNotFound` on 404.
        """
        url = self.statement_url(decision_date)
        cut = _as_utc(as_of) if as_of is not None else None
        released = self.released_at_for(url, items=rss_items)
        if released is None:
            released = _et_to_utc(decision_date, DECISION_ET)
        if cut is not None and released > cut:
            logger.debug(
                "statement %s released %s is after as_of %s — withheld",
                url, released, cut,
            )
            return None

        html_text = self._get_text(url)
        article = parse_article(html_text)
        paragraphs = article["paragraphs"]
        return FomcStatement(
            doc_type=DOC_TYPE_STATEMENT,
            url=url,
            title=article["title"] or "Federal Reserve issues FOMC statement",
            paragraphs=paragraphs,
            text="\n\n".join(paragraphs),
            released_at=released,
            meeting_date=decision_date,
            raw_html_len=len(html_text or ""),
            vote=parse_vote(paragraphs),
            target_range=parse_target_range(paragraphs),
        )

    def fetch_minutes(
        self,
        meeting_end_date: date,
        *,
        as_of: datetime | None = None,
        rss_items: list[RssItem] | None = None,
    ) -> FomcDocument | None:
        """The minutes of the meeting that ended on ``meeting_end_date``.

        ``released_at`` comes from the RSS item whose title names this
        meeting — the minutes page itself carries no machine-readable release
        instant, and the release is three weeks after the meeting, so dating
        them by the meeting date would make an as-of replay show minutes
        twenty-one days before they existed. When the feed does not reach back
        far enough, ``released_at`` is ``None``: an honest unknown.
        """
        url = self.minutes_url(meeting_end_date)
        released = None
        for item in rss_items or []:
            if item.kind != RSS_KIND_MINUTES:
                continue
            dates = minutes_meeting_dates(item.title)
            if dates and dates[1] == meeting_end_date:
                released = item.published_at
                break
        cut = _as_utc(as_of) if as_of is not None else None
        if cut is not None and released is not None and released > cut:
            return None

        html_text = self._get_text(url)
        article = parse_article(html_text)
        paragraphs = article["paragraphs"]
        return FomcDocument(
            doc_type=DOC_TYPE_MINUTES,
            url=url,
            title=article["title"] or "Minutes of the Federal Open Market Committee",
            paragraphs=paragraphs,
            text="\n\n".join(paragraphs),
            released_at=released,
            meeting_date=meeting_end_date,
            raw_html_len=len(html_text or ""),
        )

    def fetch_speech(
        self, url: str, *, as_of: datetime | None = None
    ) -> FomcDocument | None:
        """A speech page, by the URL the FED_SPEECH event already carries.

        The speaker comes from the page's own byline when it has one and from
        the URL slug otherwise (``/speech/cook20260805a.htm`` -> "Cook"),
        matching what ``libs.event_calendar.fed`` derives from the RSS so the
        two never disagree about who spoke. ``released_at`` is the page's
        date line at 12:00 ET when present — speeches carry no published time,
        so midday is an explicit anchor, not a claimed timestamp.
        """
        html_text = self._get_text(url)
        article = parse_article(html_text)
        paragraphs = article["paragraphs"]
        released = _speech_released_at(article["article_time"])
        if as_of is not None and released is not None and released > _as_utc(as_of):
            return None
        return FomcDocument(
            doc_type=DOC_TYPE_SPEECH,
            url=url,
            title=article["title"],
            paragraphs=paragraphs,
            text="\n\n".join(paragraphs),
            released_at=released,
            speaker=_speech_speaker(article, url),
            raw_html_len=len(html_text or ""),
        )


_SPEECH_SLUG_RE = re.compile(r"/speech/([a-z]+)\d{8}[a-z]?\.htm", re.IGNORECASE)
#: "Governor Lisa D. Cook", "Chair Jerome H. Powell", "Vice Chair for
#: Supervision …" — the byline paragraph on a speech page.
_SPEECH_BYLINE_RE = re.compile(
    r"^(?:Chair(?:man)?|Vice\s+Chair(?:\s+for\s+\w+)?|Governor|President)\s+(.+)$",
    re.IGNORECASE,
)
_ARTICLE_DATE_RE = re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})")
#: Speeches carry no clock time; noon ET is the explicit anchor.
_SPEECH_ANCHOR_ET = (12, 0)


def _speech_released_at(article_time: str) -> datetime | None:
    match = _ARTICLE_DATE_RE.search(article_time or "")
    if not match:
        return None
    from .fed import _month_number

    month = _month_number(match.group(1))
    if not month:
        return None
    try:
        day = date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None
    return _et_to_utc(day, _SPEECH_ANCHOR_ET)


def _speech_speaker(article: dict, url: str) -> str | None:
    for text in list(article.get("headings") or []) + list(
        article.get("paragraphs") or []
    )[:3]:
        match = _SPEECH_BYLINE_RE.match(text.strip())
        if match:
            return _normalize_space(match.group(0))
    slug = _SPEECH_SLUG_RE.search(url or "")
    return slug.group(1).capitalize() if slug else None


def _as_utc(moment: datetime) -> datetime:
    """A naive instant is Eastern wall-clock (what the Fed publishes in)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=EASTERN).astimezone(timezone.utc)
    return moment.astimezone(timezone.utc)
