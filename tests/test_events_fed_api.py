"""Fed API — GET/POST /api/events/{id}/fed (Phase H, U3; event spec §9,
§42-§45, §46, §96; audit catalyst-event-audit.md §11.9).

WHY THE PROVIDER IS A FAKE AND THE DOCUMENTS ARE REAL. Every statement,
minutes page and speech below is parsed from the LIVE fixtures U1 downloaded
from federalreserve.gov with the contact User-Agent — the June 2026 statement
(a 12-0 vote) and the July 2026 statement (9-3, two dissents named in the
document) — through U1's OWN parser. What is faked is only the HTTP transport:
these tests are about the SEAM, and U1's suite already pins the parser against
those same bytes. So the diff counts asserted here are the diff of two real
FOMC statements, not of a hand-written pair engineered to produce them.

The guarantees these tests defend, in the order they appear:

1. **A GET NEVER FETCHES.** Asserted against an EXPLODING provider on an event
   whose documents are already stored. federalreserve.gov rate-limits by
   User-Agent; a read that lazily fetched four documents per page load would
   get this platform's contact address throttled, and the throttle would land
   on the backfill that could have repaired it.
2. **Only POST writes**, and it writes the documents plus the §45 minute
   windows, reports every outcome by name, and leaves a DATA_BACKFILL audit
   row carrying ``kind: "event_fed"``.
3. **THE §44 DIFF IS OVER STORED TEXT AND IS DETERMINISTIC.** The same two
   statements always produce the same ADDED/REMOVED/CHANGED/UNCHANGED counts,
   because difflib is deterministic and the inputs are stored rather than
   refetched.
4. **THE AS-OF GATE HIDES A LATER STATEMENT** (§96): the same stored documents
   answered at two instants give two different "previous statements", because
   the gate is in the query rather than applied to an answer that already saw
   them.
5. **TWO REACTION WINDOWS, SEPARATED** (§45). Planted minute bars that RISE
   through 14:00-14:30 and FALL through 14:30-15:30 come back as two objects
   with opposite signs — the exact case one blended number would erase.
6. **NO SINGLE HAWKISH/DOVISH SCORE ANYWHERE** (§43), asserted mechanically
   over every key of the live payload.
7. **The evidence bundle carries ``macro_context.fed``** and POST /analysis on
   an FOMC event still succeeds against the stub LLM.

Uses the shared ``client`` fixture (conftest.py).
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from apps.gateway import event_fed as seam
from apps.gateway.db import (
    AuditEvent,
    EventRow,
    FedDocumentRow,
    SessionLocal,
    StockBar1mRow,
    StockBarDaily,
)
from libs.event_calendar.fed_docs import (
    FomcDocument,
    FomcStatement,
    RssItem,
    parse_article,
    parse_target_range,
    parse_vote,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

EASTERN = ZoneInfo("America/New_York")
FIXTURES = Path(__file__).parent / "fixtures" / "events"

#: The two real meetings the fixtures cover. June's statement was UNANIMOUS
#: (12-0) and July's carried two named dissenters (9-3) — which is why the
#: COMMITTEE_DISPERSION dimension below has something real to report.
JUNE = date(2026, 6, 17)
JULY = date(2026, 7, 29)

#: A fixed anchor rather than ``now()``: every as-of assertion here is a
#: statement about ordering between instants, and a drifting clock would make
#: "the July statement is not yet visible" rot overnight.
NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
NOW_Q = "2026-08-19T12:00:00Z"

#: An instant BETWEEN the two decisions — the July statement did not exist.
BEFORE_JULY = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
BEFORE_JULY_Q = "2026-07-01T12:00:00Z"


def _et(day: date, hour: int, minute: int = 0) -> datetime:
    """An ET wall clock as the UTC instant the DB stores."""
    return datetime(
        day.year, day.month, day.day, hour, minute, tzinfo=EASTERN
    ).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# The fixtures, parsed by U1's own parser
# ---------------------------------------------------------------------------


def _statement(day: date) -> FomcStatement:
    """One live-derived statement fixture as U1's value type."""
    html = (FIXTURES / f"fomc_statement_{day.isoformat()}.html").read_text()
    article = parse_article(html)
    paragraphs = article["paragraphs"]
    return FomcStatement(
        doc_type="STATEMENT",
        url=(
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            f"monetary{day:%Y%m%d}a.htm"
        ),
        title=article["title"],
        paragraphs=paragraphs,
        text="\n\n".join(paragraphs),
        released_at=_et(day, 14, 0),
        meeting_date=day,
        raw_html_len=len(html),
        vote=parse_vote(paragraphs),
        target_range=parse_target_range(paragraphs),
    )


def _minutes(day: date, *, released_at: datetime) -> FomcDocument:
    html = (FIXTURES / f"fomc_minutes_{day.isoformat()}.html").read_text()
    article = parse_article(html)
    paragraphs = article["paragraphs"]
    return FomcDocument(
        doc_type="MINUTES",
        url=f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{day:%Y%m%d}.htm",
        title=article["title"],
        paragraphs=paragraphs,
        text="\n\n".join(paragraphs),
        released_at=released_at,
        meeting_date=day,
        raw_html_len=len(html),
    )


def _speech(url: str, *, speaker: str, released_at: datetime) -> FomcDocument:
    """A speech document. The body is deliberately short: the packet quotes a
    speech's SPEAKER and TITLE, never its text, so a long fixture would test
    nothing the statement fixtures do not already cover."""
    paragraphs = [
        "The labor market has cooled while inflation remains above our "
        "objective, and I judge the current stance appropriate for now.",
    ]
    return FomcDocument(
        doc_type="SPEECH",
        url=url,
        title="Outlook for the U.S. Economy",
        paragraphs=paragraphs,
        text="\n\n".join(paragraphs),
        released_at=released_at,
        speaker=speaker,
        raw_html_len=4096,
    )


class FakeFedDocs:
    """A Fed documents client serving the live fixtures and counting calls.

    Honours the as-of gate the REAL provider honours — ``fetch_statement``
    returns ``None`` without touching the transport when the document was
    released after ``as_of`` — because the backfill's NOT_YET_RELEASED branch
    is a real code path and a fake that always answered would never exercise it.
    """

    name = "fed_docs"

    def __init__(self, *, statements: dict[date, FomcStatement] | None = None,
                 minutes: dict[date, FomcDocument] | None = None,
                 speeches: dict[str, FomcDocument] | None = None) -> None:
        self.statements = dict(statements or {})
        self.minutes = dict(minutes or {})
        self.speeches = dict(speeches or {})
        self.calls: list[tuple[str, object]] = []

    def list_press_monetary(self, as_of=None):
        self.calls.append(("rss", as_of))
        items = []
        for day, doc in sorted(self.statements.items()):
            items.append(
                RssItem(
                    title="Federal Reserve issues FOMC statement",
                    url=doc.url,
                    published_at=doc.released_at,
                    kind="STATEMENT",
                )
            )
        return [
            item for item in items if as_of is None or item.published_at <= as_of
        ]

    def fetch_statement(self, decision_date, *, as_of=None, rss_items=None):
        self.calls.append(("statement", decision_date))
        doc = self.statements.get(decision_date)
        if doc is None:
            return None
        if as_of is not None and doc.released_at > as_of:
            return None
        return doc

    def fetch_minutes(self, meeting_end_date, *, as_of=None, rss_items=None):
        self.calls.append(("minutes", meeting_end_date))
        doc = self.minutes.get(meeting_end_date)
        if doc is None:
            return None
        if as_of is not None and doc.released_at is not None and doc.released_at > as_of:
            return None
        return doc

    def fetch_speech(self, url, *, as_of=None):
        self.calls.append(("speech", url))
        doc = self.speeches.get(url)
        if doc is None:
            return None
        if as_of is not None and doc.released_at is not None and doc.released_at > as_of:
            return None
        return doc


class ExplodingFedDocs:
    """Every method raises. Installed under GET to prove the read is store-only."""

    name = "fed_docs"

    def _boom(self, *a, **k):
        raise AssertionError("the read path fetched from federalreserve.gov")

    list_press_monetary = _boom
    fetch_statement = _boom
    fetch_minutes = _boom
    fetch_speech = _boom


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


async def _add_event(
    *,
    event_type: EventType,
    when: datetime,
    key: str | None = None,
    title: str = "",
    source_url: str | None = None,
    speaker: str | None = None,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key or f"{event_type.value}:{when.astimezone(EASTERN).date()}",
            event_type=event_type.value,
            title=title or f"{event_type.value}",
            ticker=None,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=EventSession.DURING_MARKET.value,
            status=EventStatus.CONFIRMED.value,
            source=EventSourceKind.FEDERAL_RESERVE.value,
            source_name="fed_fomc",
            source_url=source_url,
            agency="Federal Reserve",
            speaker=speaker,
            importance=95,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _add_meeting_events() -> dict[str, int]:
    """The June and July decisions plus the September meeting under analysis.

    Exactly the shape ``libs.event_calendar.fed`` writes: the decision key
    carries the meeting's END date and the instant is 14:00 ET.
    """
    return {
        "june": await _add_event(
            event_type=EventType.FOMC_DECISION,
            when=_et(JUNE, 14, 0),
            key=f"FOMC_DECISION:{JUNE.isoformat()}",
            title="FOMC rate decision",
        ),
        "july": await _add_event(
            event_type=EventType.FOMC_DECISION,
            when=_et(JULY, 14, 0),
            key=f"FOMC_DECISION:{JULY.isoformat()}",
            title="FOMC rate decision",
        ),
        "september": await _add_event(
            event_type=EventType.FOMC_DECISION,
            when=_et(date(2026, 9, 16), 14, 0),
            key="FOMC_DECISION:2026-09-16",
            title="FOMC rate decision (with Summary of Economic Projections)",
        ),
    }


async def _store_documents(*, july_released: datetime | None = None) -> None:
    """The two statements and June's minutes, stored as the backfill stores them."""
    async with SessionLocal() as s:
        for day in (JUNE, JULY):
            doc = _statement(day)
            released = july_released if (day == JULY and july_released) else doc.released_at
            s.add(
                FedDocumentRow(
                    doc_type="STATEMENT",
                    meeting_date=day,
                    url=doc.url,
                    title=doc.title,
                    released_at=released,
                    text=doc.text,
                    paragraphs=list(doc.paragraphs),
                    parsed={"vote": dict(doc.vote), "target_range": doc.target_range},
                    provider="fed_docs",
                )
            )
        mins = _minutes(JUNE, released_at=_et(date(2026, 7, 8), 14, 0))
        s.add(
            FedDocumentRow(
                doc_type="MINUTES",
                meeting_date=JUNE,
                url=mins.url,
                title=mins.title,
                released_at=mins.released_at,
                text=mins.text,
                paragraphs=list(mins.paragraphs),
                parsed={},
                provider="fed_docs",
            )
        )
        await s.commit()


async def _store_speech(*, url: str, speaker: str, released_at: datetime) -> None:
    doc = _speech(url, speaker=speaker, released_at=released_at)
    async with SessionLocal() as s:
        s.add(
            FedDocumentRow(
                doc_type="SPEECH",
                meeting_date=None,
                url=doc.url,
                title=doc.title,
                released_at=doc.released_at,
                text=doc.text,
                paragraphs=list(doc.paragraphs),
                parsed={"speaker": doc.speaker},
                provider="fed_docs",
            )
        )
        await s.commit()


async def _plant_minute_bars(
    *, day: date, symbol: str, statement_pct: float, presser_pct: float
) -> None:
    """Minute bars that move ``statement_pct`` over 14:00-14:30 and
    ``presser_pct`` over 14:30-15:30, on the same day.

    Chosen so the two windows have OPPOSITE signs: that is the case one
    blended number would erase, and it is the whole reason §45 asks for two.
    """
    base = 100.0
    rows: list[tuple[datetime, float]] = []
    start = _et(day, 14, 0)
    for minute in range(0, 31):
        frac = minute / 30
        rows.append((start + timedelta(minutes=minute), base * (1 + statement_pct / 100 * frac)))
    mid_close = rows[-1][1]
    mid = _et(day, 14, 30)
    for minute in range(1, 61):
        frac = minute / 60
        rows.append((mid + timedelta(minutes=minute), mid_close * (1 + presser_pct / 100 * frac)))
    async with SessionLocal() as s:
        for ts, close in rows:
            s.add(
                StockBar1mRow(
                    ticker=symbol,
                    ts=ts,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=10_000,
                )
            )
        await s.commit()


async def _plant_daily_bars(*, symbol: str, around: date, closes: list[float]) -> None:
    """Daily bars spanning the decision — the §45 fallback's input."""
    day = around - timedelta(days=len(closes) // 2)
    async with SessionLocal() as s:
        for close in closes:
            while day.weekday() >= 5:
                day += timedelta(days=1)
            s.add(
                StockBarDaily(
                    ticker=symbol,
                    ts=day,
                    open=round(close * 0.995, 4),
                    high=round(close * 1.01, 4),
                    low=round(close * 0.99, 4),
                    close=close,
                    volume=1_000_000.0,
                )
            )
            day += timedelta(days=1)
        await s.commit()


async def _event_row(event_id: int) -> EventRow:
    async with SessionLocal() as s:
        return await s.get(EventRow, event_id)


def _walk(value, path=""):
    """Every (path, key) pair in a nested payload — the §43 grep, mechanised."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.{key}", str(key)
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk(item, f"{path}[{idx}]")


# ---------------------------------------------------------------------------
# 1. The read is store-only
# ---------------------------------------------------------------------------


async def test_get_fed_never_fetches_and_serves_the_stored_documents(
    client, monkeypatch
):
    """§27 / audit §7.2 rule 1 — with an EXPLODING provider installed, the GET
    still answers the whole packet off stored rows.

    The seam imports the factory lazily inside ``backfill_fed``, so this patch
    covers the only path that could ever reach the network — and the assertion
    that matters is not that the patch was used but that nothing raised."""
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(
        calendar_pkg, "fed_documents_provider", lambda *a, **k: ExplodingFedDocs()
    )
    ids = await _add_meeting_events()
    await _store_documents()

    r = await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    packet = body["packet"]
    assert packet["previous_statement"]["available"] is True
    # The PREVIOUS decision at 2026-08-19 is July, not September's own row.
    assert packet["previous_statement"]["meeting_date"] == JULY.isoformat()
    assert packet["previous_statement"]["compared_to"]["meeting_date"] == JUNE.isoformat()


async def test_get_on_a_non_fed_event_is_200_with_a_reason_not_404(client):
    """A row that exists but has no Committee behind it is honest absence, not
    a 404 — 404 would say the event does not exist."""
    event_id = await _add_event(
        event_type=EventType.CPI,
        when=_et(date(2026, 8, 12), 8, 30),
        key="CPI:2026-08-12",
    )
    r = await client.get(f"/api/events/{event_id}/fed?as_of={NOW_Q}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "not a Federal Reserve event" in body["reason"]


async def test_get_with_nothing_stored_names_the_backfill(client):
    """No documents stored: coverage says so and names the POST that fixes it —
    never an empty diff presented as "nothing changed"."""
    ids = await _add_meeting_events()
    r = await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    body = r.json()
    coverage = body["packet"]["coverage"]
    assert coverage["previous_statement"] is False
    assert "Backfill" in coverage["previous_statement_reason"]
    assert body["packet"]["statement_diff"]["counts"]["TOTAL"] == 0


async def test_get_with_no_registered_decision_says_run_the_ingest(client):
    """A speech with no FOMC_DECISION anywhere in the registry: the reason
    names the ingest rather than the backfill, because there is no meeting to
    fetch documents FOR."""
    event_id = await _add_event(
        event_type=EventType.FED_SPEECH,
        when=_et(date(2026, 8, 5), 12, 0),
        key="FED_SPEECH:2026-08-05:cook:outlook",
        speaker="Cook",
    )
    body = (await client.get(f"/api/events/{event_id}/fed?as_of={NOW_Q}")).json()
    assert body["available"] is True
    assert "calendar ingest" in body["packet"]["coverage"]["previous_decision_reason"]


# ---------------------------------------------------------------------------
# 2. The §44 diff over stored text
# ---------------------------------------------------------------------------


async def test_diff_counts_are_deterministic_over_the_two_live_statements(client):
    """The REAL June-to-July diff: 1 ADDED, 2 CHANGED, 6 UNCHANGED, 0 REMOVED.

    These numbers are a fact about two documents the Federal Reserve published,
    not about a fixture engineered to produce them — the ADDED sentence is the
    "Voting against" line that exists in July because two governors dissented
    and does not exist in June because nobody did.
    """
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    counts = body["packet"]["statement_diff"]["counts"]
    assert counts == {"ADDED": 1, "REMOVED": 0, "CHANGED": 2, "UNCHANGED": 6, "TOTAL": 9}

    added = [
        item
        for item in body["packet"]["statement_diff"]["items"]
        if item["status"] == "ADDED"
    ]
    assert len(added) == 1
    assert "Voting against" in added[0]["current_text"]
    assert "COMMITTEE_DISPERSION" in added[0]["dimensions"]


async def test_the_diff_is_stable_across_two_identical_reads(client):
    """difflib is deterministic and the inputs are STORED, so the same request
    twice is the same bytes. A diff that re-fetched its inputs could not
    promise this."""
    ids = await _add_meeting_events()
    await _store_documents()
    url = f"/api/events/{ids['september']}/fed?as_of={NOW_Q}"
    first = (await client.get(url)).json()["packet"]["statement_diff"]
    second = (await client.get(url)).json()["packet"]["statement_diff"]
    assert first == second


async def test_vote_and_target_range_come_from_the_document(client):
    """§44: the numbers are PARSED from the Fed's own sentence, and the
    sentence rides along so the UI can show the source line beside them."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    prev = body["packet"]["previous_statement"]
    assert prev["vote"]["for"] == 9
    assert prev["vote"]["against"] == 3
    assert prev["vote"]["unanimous"] is False
    assert len(prev["vote"]["dissenters"]) == 3
    assert prev["target_range"]["low_pct"] == 3.5
    assert prev["target_range"]["high_pct"] == 3.75
    assert "3-1/2 to 3-3/4 percent" in prev["target_range"]["text"]
    # June was UNANIMOUS — the dimension that has something to say about it.
    assert prev["compared_to"]["vote"]["unanimous"] is True
    assert prev["compared_to"]["vote"]["against"] == 0


async def test_dimensions_are_reported_separately_with_the_policy_rate_change(client):
    """§43 in its positive form: eight dimensions, each with its own status and
    its own sentences. POLICY_RATE additionally carries the change in basis
    points — 3.50-3.75 held at 3.50-3.75 is HOLD, 0 bp, which is a MEASURED
    fact about two parsed ranges rather than a reading of the language."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    dims = body["packet"]["dimensions"]
    assert set(dims) >= {
        "POLICY_RATE",
        "INFLATION",
        "EMPLOYMENT",
        "GROWTH",
        "BALANCE_SHEET",
        "FORWARD_GUIDANCE",
        "RISK_BALANCE",
        "COMMITTEE_DISPERSION",
    }
    assert dims["POLICY_RATE"]["policy_rate_change"]["change_bp"] == 0
    assert dims["POLICY_RATE"]["policy_rate_change"]["direction"] == "HOLD"
    assert dims["COMMITTEE_DISPERSION"]["current_vote"]["against"] == 3
    assert dims["COMMITTEE_DISPERSION"]["previous_vote"]["against"] == 0


async def test_no_single_hawkish_or_dovish_score_anywhere_in_the_payload(client):
    """§43, asserted MECHANICALLY over every key of the live payload rather
    than by reading the code.

    The only permitted match is the disclaimer TEXT that states such a score
    must not exist — a value, never a key. A key named ``hawkish_score`` is
    exactly the thing this platform refuses to compute, and the refusal has to
    be enforced by a test or it becomes a comment."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    offenders = [
        path
        for path, key in _walk(body)
        if any(bad in key.lower() for bad in ("score", "hawk", "dove"))
    ]
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# 3. The as-of gate (§96)
# ---------------------------------------------------------------------------


async def test_as_of_before_july_hides_the_july_statement(client):
    """THE SAME STORED ROWS, two instants, two different answers.

    At 2026-08-19 the previous statement is July's (9-3). At 2026-07-01 July
    had not happened, so the previous statement is JUNE's (12-0) and there is
    no statement before it stored — the diff has nothing to compare against and
    says so rather than diffing June against itself."""
    ids = await _add_meeting_events()
    await _store_documents()

    late = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()["packet"]
    early = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={BEFORE_JULY_Q}")
    ).json()["packet"]

    assert late["previous_statement"]["meeting_date"] == JULY.isoformat()
    assert late["previous_statement"]["vote"]["against"] == 3

    assert early["previous_statement"]["meeting_date"] == JUNE.isoformat()
    assert early["previous_statement"]["vote"]["unanimous"] is True
    assert early["coverage"]["compared_statement"] is False


async def test_a_document_with_an_unknown_release_instant_is_not_visible(client):
    """``released_at IS NULL`` means UNKNOWN, and an unknown instant cannot be
    resolved into "before your as_of". Admitting it would put June's minutes on
    the reader's screen three weeks before they were written (§44 rule 18)."""
    ids = await _add_meeting_events()
    await _store_documents()
    async with SessionLocal() as s:
        row = (
            await s.execute(
                select(FedDocumentRow).where(FedDocumentRow.doc_type == "MINUTES")
            )
        ).scalars().one()
        row.released_at = None
        await s.commit()

    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    assert body["packet"]["previous_minutes"]["available"] is False


async def test_the_minutes_block_carries_dimension_tagged_key_paragraphs(client):
    """The minutes run to thousands of words; the packet keeps only the
    sentences that carry a dimension tag, in document order. A mechanical
    selection a reader can audit — no summarisation happens anywhere."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    minutes = body["packet"]["previous_minutes"]
    assert minutes["available"] is True
    assert minutes["meeting_date"] == JUNE.isoformat()
    assert minutes["key_paragraphs"], "the June minutes tagged nothing"
    assert all(item["dimensions"] for item in minutes["key_paragraphs"])
    # Document order, never re-ranked.
    idxs = [item["idx"] for item in minutes["key_paragraphs"]]
    assert idxs == sorted(idxs)


async def test_a_future_as_of_is_422(client):
    ids = await _add_meeting_events()
    r = await client.get(f"/api/events/{ids['september']}/fed?as_of=2099-01-01T00:00:00Z")
    assert r.status_code == 422
    assert "no Fed documents exist for it" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 4. §45 — two reaction windows, separated
# ---------------------------------------------------------------------------


async def test_statement_and_press_conference_windows_are_separate_and_opposite(
    client,
):
    """THE CASE ONE BLENDED NUMBER WOULD ERASE. SPY rises +0.60% over
    14:00-14:30 (the statement) and falls -0.40% over 14:30-15:30 (the Chair's
    Q&A). Both signs survive into the payload, in two objects that are never
    added together."""
    ids = await _add_meeting_events()
    await _store_documents()
    await _plant_minute_bars(day=JULY, symbol="SPY", statement_pct=0.60, presser_pct=-0.40)

    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    reaction = body["packet"]["previous_reaction"]
    assert reaction["basis"] == "1m_bars"
    assert reaction["separated"] is True
    assert reaction["unit"] == "percent"

    statement = reaction["statement"]["SPY"]
    presser = reaction["press_conference"]["SPY"]
    assert statement["return_pct"] == pytest.approx(0.60, abs=0.02)
    assert presser["return_pct"] == pytest.approx(-0.40, abs=0.02)
    assert statement["return_pct"] > 0 > presser["return_pct"]

    assert reaction["windows"]["statement"]["label_et"] == "14:00-14:30 ET"
    assert reaction["windows"]["press_conference"]["label_et"] == "14:30-15:30 ET"


async def test_daily_fallback_says_it_could_not_separate_the_windows(client):
    """With NO minute bars stored, the payload falls back to daily and states
    plainly that the two windows could not be told apart — it does not quietly
    report the blend under the statement's name."""
    ids = await _add_meeting_events()
    await _store_documents()
    await _plant_daily_bars(
        symbol="SPY", around=JULY, closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    )
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    reaction = body["packet"]["previous_reaction"]
    assert reaction["basis"] == "daily"
    assert reaction["separated"] is False
    assert reaction["label"] == "daily (no intraday bars)"
    assert reaction["statement"] == {}


async def test_no_bars_at_all_is_a_named_absence_never_a_zero_return(client):
    """§44 rule 18: nothing stored is a REASON, not a 0.0% move."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    reaction = body["packet"]["previous_reaction"]
    assert reaction["available"] is False
    assert "no stored bars" in reaction["reason"]
    assert reaction["statement"] == {}


async def test_market_pricing_is_unavailable_and_says_so(client):
    """This platform subscribes to no fed funds futures feed, so there is no
    branch that could produce implied odds of a cut. The status is the fixed
    string and the disclaimer repeats it at top level."""
    ids = await _add_meeting_events()
    await _store_documents()
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    pricing = body["packet"]["market_pricing"]
    assert pricing["status"] == "UNAVAILABLE"
    assert pricing["proxy"] is None
    assert "fed funds futures" in body["disclaimer"].lower()


# ---------------------------------------------------------------------------
# 5. Speeches since the previous decision
# ---------------------------------------------------------------------------


async def test_speeches_since_the_previous_decision_are_listed_in_order(client):
    """Speaker, title, instant and URL — never the speech's TEXT. A speech is
    context for the statement, not a second document to diff."""
    ids = await _add_meeting_events()
    await _store_documents()
    await _store_speech(
        url="https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm",
        speaker="Cook",
        released_at=_et(date(2026, 8, 5), 12, 0),
    )
    await _store_speech(
        url="https://www.federalreserve.gov/newsevents/speech/waller20260812a.htm",
        speaker="Waller",
        released_at=_et(date(2026, 8, 12), 12, 0),
    )
    # Given BEFORE the July decision — belongs to the previous cycle, excluded.
    await _store_speech(
        url="https://www.federalreserve.gov/newsevents/speech/powell20260710a.htm",
        speaker="Powell",
        released_at=_et(date(2026, 7, 10), 12, 0),
    )
    body = (
        await client.get(f"/api/events/{ids['september']}/fed?as_of={NOW_Q}")
    ).json()
    speeches = body["packet"]["subsequent_speeches"]
    assert [s["speaker"] for s in speeches] == ["Cook", "Waller"]
    assert speeches[0]["at"] < speeches[1]["at"]
    assert all("text" not in s for s in speeches)


# ---------------------------------------------------------------------------
# 6. The backfill — the only path that fetches
# ---------------------------------------------------------------------------


async def test_backfill_stores_both_statements_the_minutes_and_reports_outcomes(
    client, monkeypatch
):
    """POST fetches the RSS ONCE, both statements, the minutes and every
    speech, stores them, and names every outcome."""
    ids = await _add_meeting_events()
    await _add_event(
        event_type=EventType.FED_SPEECH,
        when=_et(date(2026, 8, 5), 12, 0),
        key="FED_SPEECH:2026-08-05:cook:outlook",
        speaker="Cook",
        source_url="https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm",
    )
    # The July meeting's minutes were released on 2026-08-19 at 14:00 ET —
    # AFTER this test's as_of of 12:00Z the same day, which is 08:00 ET. That
    # is the real calendar, not a contrivance, and it exercises the branch a
    # made-up earlier date would have skipped: the minutes are reported
    # NOT_YET_RELEASED and nothing is stored for them.
    fake = FakeFedDocs(
        statements={JUNE: _statement(JUNE), JULY: _statement(JULY)},
        minutes={JULY: _minutes(JUNE, released_at=_et(date(2026, 8, 19), 14, 0))},
        speeches={
            "https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm": _speech(
                "https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm",
                speaker="Cook",
                released_at=_et(date(2026, 8, 5), 12, 0),
            )
        },
    )
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(
        calendar_pkg, "fed_documents_provider", lambda *a, **k: fake
    )

    r = await client.post(f"/api/events/{ids['september']}/fed/backfill?as_of={NOW_Q}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["counts"]["documents"] == 3  # two statements + the speech

    kinds = {row["kind"]: row for row in body["documents"]}
    assert kinds["rss"]["status"] == "OK"
    assert kinds["previous_statement"]["status"] == "STORED"
    assert kinds["previous_statement"]["meeting_date"] == JULY.isoformat()
    assert kinds["compared_statement"]["meeting_date"] == JUNE.isoformat()
    assert kinds["speech"]["speaker"] == "Cook"
    # The minutes of the July meeting were not out at 08:00 ET on release day.
    assert kinds["minutes"]["status"] == "NOT_YET_RELEASED"
    assert "had not been published" in kinds["minutes"]["reason"]

    # ONE RSS call, not one per document.
    assert sum(1 for call in fake.calls if call[0] == "rss") == 1

    async with SessionLocal() as s:
        stored = (await s.execute(select(FedDocumentRow))).scalars().all()
    assert len(stored) == 3
    assert {row.provider for row in stored} == {"fed_docs"}
    # NOTHING was written for the document that had not been published.
    assert all(row.doc_type != "MINUTES" for row in stored)


async def test_backfill_stores_the_minutes_once_they_have_been_released(
    client, monkeypatch
):
    """The counterpart to the test above: at 20:00Z on release day the July
    minutes ARE public, so they are fetched and stored with the RSS's own
    publication instant — not with the meeting date, which is three weeks
    earlier and would make an as-of replay show them before they existed."""
    ids = await _add_meeting_events()
    released = _et(date(2026, 8, 19), 14, 0)
    fake = FakeFedDocs(
        statements={JUNE: _statement(JUNE), JULY: _statement(JULY)},
        minutes={JULY: _minutes(JUNE, released_at=released)},
    )
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(calendar_pkg, "fed_documents_provider", lambda *a, **k: fake)

    body = (
        await client.post(
            f"/api/events/{ids['september']}/fed/backfill?as_of=2026-08-19T20:00:00Z"
        )
    ).json()
    kinds = {row["kind"]: row for row in body["documents"]}
    assert kinds["minutes"]["status"] == "STORED"
    assert kinds["minutes"]["released_at"] == released.isoformat()
    assert kinds["minutes"]["paragraphs"] > 100  # the real June minutes

    async with SessionLocal() as s:
        row = (
            await s.execute(
                select(FedDocumentRow).where(FedDocumentRow.doc_type == "MINUTES")
            )
        ).scalars().one()
    # Keyed on the MEETING, dated by the RELEASE — the two are three weeks
    # apart and the packet needs both.
    assert row.meeting_date == JULY
    assert row.released_at.replace(tzinfo=timezone.utc) == released


async def test_backfill_is_idempotent_on_the_document_url(client, monkeypatch):
    """One Fed URL is one document forever, so a second press overwrites the
    same rows rather than creating a second copy the diff would have to choose
    between."""
    ids = await _add_meeting_events()
    fake = FakeFedDocs(statements={JUNE: _statement(JUNE), JULY: _statement(JULY)})
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(calendar_pkg, "fed_documents_provider", lambda *a, **k: fake)

    first = await client.post(
        f"/api/events/{ids['september']}/fed/backfill?as_of={NOW_Q}"
    )
    second = await client.post(
        f"/api/events/{ids['september']}/fed/backfill?as_of={NOW_Q}"
    )
    assert first.status_code == second.status_code == 200
    statuses = {row["kind"]: row["status"] for row in second.json()["documents"]}
    assert statuses["previous_statement"] == "UPDATED"

    async with SessionLocal() as s:
        rows = (
            await s.execute(
                select(FedDocumentRow).where(FedDocumentRow.doc_type == "STATEMENT")
            )
        ).scalars().all()
    assert len(rows) == 2


async def test_backfill_honours_as_of_without_issuing_a_request(client, monkeypatch):
    """A statement released after ``as_of`` is NOT_YET_RELEASED. This is what
    makes a point-in-time replay cheap as well as correct: the RSS instant
    already answers the question, so no page is fetched."""
    ids = await _add_meeting_events()
    fake = FakeFedDocs(statements={JUNE: _statement(JUNE), JULY: _statement(JULY)})
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(calendar_pkg, "fed_documents_provider", lambda *a, **k: fake)

    body = (
        await client.post(
            f"/api/events/{ids['september']}/fed/backfill?as_of={BEFORE_JULY_Q}"
        )
    ).json()
    kinds = {row["kind"]: row for row in body["documents"]}
    # At 2026-07-01 the PREVIOUS decision is June, and the one before it is not
    # registered — so the statement fetched is June's and there is nothing to
    # compare it against.
    assert kinds["previous_statement"]["meeting_date"] == JUNE.isoformat()
    assert kinds["compared_statement"]["status"] == "SKIPPED"

    async with SessionLocal() as s:
        rows = (await s.execute(select(FedDocumentRow))).scalars().all()
    assert {row.meeting_date for row in rows if row.doc_type == "STATEMENT"} == {JUNE}


async def test_backfill_writes_a_data_backfill_audit_row(client, monkeypatch):
    """Rule 12 / ADR-003: the audit row rides in the SAME transaction as the
    data, carrying ``kind: "event_fed"``."""
    ids = await _add_meeting_events()
    fake = FakeFedDocs(statements={JUNE: _statement(JUNE), JULY: _statement(JULY)})
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(calendar_pkg, "fed_documents_provider", lambda *a, **k: fake)

    await client.post(f"/api/events/{ids['september']}/fed/backfill?as_of={NOW_Q}")
    async with SessionLocal() as s:
        rows = (
            await s.execute(
                select(AuditEvent).where(AuditEvent.action == "DATA_BACKFILL")
            )
        ).scalars().all()
    fed_rows = [r for r in rows if (r.details or {}).get("kind") == "event_fed"]
    assert len(fed_rows) == 1
    assert fed_rows[0].details["counts"]["documents"] == 2
    assert fed_rows[0].details["previous_decision"] == f"FOMC_DECISION:{JULY.isoformat()}"


async def test_backfill_on_a_non_fed_event_spends_nothing(client, monkeypatch):
    """No Committee, no catalogue, no requests."""
    event_id = await _add_event(
        event_type=EventType.CPI,
        when=_et(date(2026, 8, 12), 8, 30),
        key="CPI:2026-08-12",
    )
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(
        calendar_pkg, "fed_documents_provider", lambda *a, **k: ExplodingFedDocs()
    )
    body = (await client.post(f"/api/events/{event_id}/fed/backfill")).json()
    assert body["available"] is False
    assert body["counts"] == {"documents": 0, "bars": 0}


async def test_backfill_with_no_registered_decision_refuses_before_fetching(
    client, monkeypatch
):
    """There is no meeting to fetch documents for, so nothing is fetched — the
    exploding provider proves it."""
    event_id = await _add_event(
        event_type=EventType.FOMC_DECISION,
        when=_et(date(2026, 9, 16), 14, 0),
        key="FOMC_DECISION:2026-09-16",
    )
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(
        calendar_pkg, "fed_documents_provider", lambda *a, **k: ExplodingFedDocs()
    )
    body = (await client.post(f"/api/events/{event_id}/fed/backfill")).json()
    assert body["available"] is False
    assert "calendar ingest" in body["reason"]


async def test_a_document_fetch_failure_is_a_named_row_not_a_500(
    client, monkeypatch
):
    """One dead document never costs the other three: the minutes fail, the
    statements still land."""
    from libs.event_calendar.fed_docs import FedDocsError

    ids = await _add_meeting_events()
    fake = FakeFedDocs(statements={JUNE: _statement(JUNE), JULY: _statement(JULY)})

    def boom(*a, **k):
        raise FedDocsError("federalreserve.gov answered 503")

    fake.fetch_minutes = boom
    import libs.event_calendar as calendar_pkg

    monkeypatch.setattr(calendar_pkg, "fed_documents_provider", lambda *a, **k: fake)

    r = await client.post(f"/api/events/{ids['september']}/fed/backfill?as_of={NOW_Q}")
    assert r.status_code == 200
    kinds = {row["kind"]: row for row in r.json()["documents"]}
    assert kinds["minutes"]["status"] == "ERROR"
    assert "503" in kinds["minutes"]["reason"]
    assert kinds["previous_statement"]["status"] == "STORED"


# ---------------------------------------------------------------------------
# 7. The evidence bundle and the analysis prompt
# ---------------------------------------------------------------------------


async def test_the_evidence_bundle_carries_macro_context_fed(client):
    """§46: the Fed packet rides as a KEY under macro_context, where a reader
    already looks for "what is the macro state of the world"."""
    ids = await _add_meeting_events()
    await _store_documents()
    r = await client.get(f"/api/events/{ids['september']}/evidence?as_of={NOW_Q}")
    assert r.status_code == 200
    bundle = r.json()["bundle"]
    fed = bundle["macro_context"]["fed"]
    assert fed["available"] is True
    assert fed["kind"] == "fomc_packet"
    assert fed["packet"]["statement_diff"]["counts"]["ADDED"] == 1
    assert bundle["coverage"]["macro_context_fed"]["available"] is True


async def test_a_non_fed_bundle_has_no_fed_key_at_all(client):
    """Not a dead stub on six thousand earnings rows — the key is ABSENT, so
    the bundle digest of every non-Fed event is unchanged by Phase H."""
    event_id = await _add_event(
        event_type=EventType.CPI,
        when=_et(date(2026, 8, 12), 8, 30),
        key="CPI:2026-08-12",
    )
    bundle = (
        await client.get(f"/api/events/{event_id}/evidence?as_of={NOW_Q}")
    ).json()["bundle"]
    assert "fed" not in bundle["macro_context"]
    assert "macro_context_fed" not in bundle["coverage"]


async def test_post_analysis_on_an_fomc_event_succeeds_against_the_stub(client):
    """End to end: the bundle carries the Fed packet, the prompt carries the
    §44 instruction, and the stub LLM answers a valid analysis."""
    ids = await _add_meeting_events()
    await _store_documents()
    r = await client.post(f"/api/events/{ids['september']}/analysis?as_of={NOW_Q}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "OK"


def test_the_fed_prompt_instruction_forbids_a_single_label():
    """§44/§43 in the prompt layer: an FOMC bundle is asked to explain the diff
    DIMENSION BY DIMENSION and told never to collapse it into one label. An
    earnings bundle's prompt bytes are UNCHANGED, so no stored analysis is
    invalidated by this addition."""
    from libs.llm.event_analysis import build_user_message

    fed = build_user_message({"event": {"event_type": "FOMC_DECISION"}})
    assert "DIMENSION BY DIMENSION" in fed
    assert "never collapse" in fed.lower()
    assert "LLM ANALYSIS" in fed

    earnings = build_user_message({"event": {"event_type": "EARNINGS"}})
    assert "FEDERAL RESERVE event" not in earnings


# ---------------------------------------------------------------------------
# 8. The seam's own units
# ---------------------------------------------------------------------------


async def test_meeting_date_is_read_from_the_event_key_not_the_timestamp(client):
    """The registry drops ``EventCandidate.raw``, so the key and the timestamp
    are the only meeting facts a stored row carries — and the KEY wins, because
    a MINUTES row's key names its release day while its timestamp does too, and
    the reconciliation happens in ``previous_decision_row`` rather than by
    trusting either one alone."""
    event_id = await _add_event(
        event_type=EventType.FOMC_DECISION,
        when=_et(JULY, 14, 0),
        key=f"FOMC_DECISION:{JULY.isoformat()}",
    )
    row = await _event_row(event_id)
    assert seam.meeting_date_of(row) == JULY

    speech_id = await _add_event(
        event_type=EventType.FED_SPEECH,
        when=_et(date(2026, 8, 5), 12, 0),
        key="FED_SPEECH:2026-08-05:cook:outlook",
    )
    assert seam.meeting_date_of(await _event_row(speech_id)) is None


async def test_previous_decision_is_bounded_by_the_event_not_by_now(client):
    """An as-of replay of a FUTURE meeting run from today still compares
    against the last decision that has actually occurred — the bound is the
    EARLIER of the event's instant and ``as_of``."""
    ids = await _add_meeting_events()
    september = await _event_row(ids["september"])
    async with SessionLocal() as s:
        prev = await seam.previous_decision_row(s, september, as_of=NOW)
        assert prev.event_key == f"FOMC_DECISION:{JULY.isoformat()}"
        early = await seam.previous_decision_row(s, september, as_of=BEFORE_JULY)
        assert early.event_key == f"FOMC_DECISION:{JUNE.isoformat()}"


async def test_a_decision_is_never_its_own_predecessor(client):
    """Asking the JULY row for its previous decision gives JUNE, not July."""
    ids = await _add_meeting_events()
    july = await _event_row(ids["july"])
    async with SessionLocal() as s:
        prev = await seam.previous_decision_row(s, july, as_of=NOW)
    assert prev.event_key == f"FOMC_DECISION:{JUNE.isoformat()}"
