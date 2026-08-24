"""Fed intelligence — the gateway seam (Phase H; event spec §9, §42-§45;
audit catalyst-event-audit.md §11.9).

THE SPLIT THIS MODULE EXISTS TO KEEP, stated once. Every sentence of the
statement diff, every dimension row and every window return is computed by
``libs/trading_core/events/fed_intel.py``, which is pure stdlib and may not
import ``apps/`` or ``libs.market_data`` (audit §7.4). Every byte of Fed HTML
is fetched and parsed by ``libs/event_calendar/fed_docs.py``. This module is
the only place the three halves meet: it resolves which meeting an event
belongs to, fetches and STORES the documents, reads them back with their
minute bars, hands value types to the pure builder and renders the frozen
result as JSON. It parses no HTML and computes no return.

READ AND WRITE ARE TWO DIFFERENT FUNCTIONS AND THAT IS THE CONTRACT (§27;
audit §7.2 rule 1). :func:`build_fed_payload` is DB-ONLY: it holds no provider
handle, cannot reach federalreserve.gov, and answers with honest absence when
nothing has been fetched. :func:`backfill_fed` is the USER action that spends
the requests. The separation earns its keep differently here than it does for
macro: the Fed serves no API and rate-limits by User-Agent, so a read endpoint
that lazily fetched four documents per page load would get the platform's
contact address throttled — and the throttle would land on the backfill that
could have repaired it.

WHY THE DOCUMENTS ARE STORED RATHER THAN RE-FETCHED. §44 makes the source
document authoritative and the diff deterministic. A diff whose inputs come
off the wire at read time is not deterministic: the Fed edits pages, retires
URLs and answers 403 to a stranger, so the same event would diff differently
on Tuesday. Storing the paragraphs is what makes the answer reproducible, and
it is the only way an as-of replay can show the June statement as it stood in
June rather than as the site serves it today.

WHICH TWO STATEMENTS THE DIFF COMPARES, and why neither is the current one.
This event has not happened yet — there is no statement to diff. The question
a trader has going into an FOMC meeting is what the Committee changed LAST
time, so the packet diffs the PREVIOUS decision's statement against the one
before it. The seam's whole job on the read path is finding those two decision
events and their stored documents.

AS-OF IS APPLIED IN THE QUERY AND AGAIN IN THE LIBRARY. Documents are loaded
with ``released_at <= as_of`` in SQL (a statement released after the instant
under replay must not reach the library at all), and ``build_fed_packet``
re-gates every document whose ``released_at`` it can see. Two gates for one
rule is deliberate: the SQL gate is the one that matters for cost, the library
gate is the one that survives a caller bug. A document whose release instant
is UNKNOWN (``released_at IS NULL`` — the minutes, when the press RSS does not
reach back) is treated as NOT VISIBLE by the query, because "we do not know
when this became public" cannot be resolved into "before your as_of".

NO SINGLE HAWKISH/DOVISH SCORE, ANYWHERE (§43). There is no key in this
payload, no column in ``fed_documents`` and no branch in the code below that
collapses eight policy dimensions into one number. Dimensions are reported
side by side and the disclaimer saying so travels in the packet.

TWO REACTION WINDOWS, NEVER BLENDED (§45). When minute bars are stored around
the previous decision, the payload reports 14:00-14:30 ET (the statement) and
14:30-15:30 ET (the Chair's press conference) as separate objects: the market
reversing between the two is the single most informative thing an FOMC replay
can show, and one blended number erases it. When only daily bars exist, the
payload says ``basis: "daily"`` and ``separated: false`` rather than quietly
reporting the blend as if it were the statement.

PROVENANCE IS LABELLED AT BLOCK LEVEL (§49, §91): the Fed's own words and
votes are DATA, the diff and the window returns are QUANT (this platform's
arithmetic). Nothing in this payload is LLM-generated.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.event_calendar.provider import CalendarProviderError
from libs.market_data import CapabilityNotAvailable, MarketDataError
from libs.trading_core.events import EASTERN
from libs.trading_core.events.evidence import TIER_DATA, TIER_QUANT, json_safe
from libs.trading_core.events.fed_intel import (
    FED_INTEL_MODEL_VERSION,
    MARKET_PRICING_NOTE,
    NO_SINGLE_SCORE_NOTE,
    SOURCE_AUTHORITATIVE_NOTE,
    build_fed_packet,
    fomc_reaction_daily,
    fomc_reaction_windows,
)
from libs.trading_core.events.reaction import DailyBar
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import EventSession, EventType

from . import audit
from .db import EventRow, FedDocumentRow, StockBar1mRow, StockBarDaily
from .event_price import _as_utc, event_date_et
from .event_macro import _to_daily_bars
from .event_replay import ensure_event_window_bars, to_minute_bars

logger = logging.getLogger(__name__)

__all__ = [
    "FED_DISCLAIMER",
    "FED_REACTION_SYMBOLS",
    "DOC_TYPE_MINUTES",
    "DOC_TYPE_SPEECH",
    "DOC_TYPE_STATEMENT",
    "FED_EVENT_TYPES",
    "backfill_fed",
    "build_fed_payload",
    "fed_context_section",
    "is_fed_event",
    "load_documents",
    "meeting_date_of",
    "previous_decision_row",
]

#: Document vocabulary, mirrored from ``libs.event_calendar.fed_docs`` rather
#: than imported, so this module stays importable when the HTTP client's
#: dependencies are not installed — the READ path must never need the fetcher.
DOC_TYPE_STATEMENT = "STATEMENT"
DOC_TYPE_MINUTES = "MINUTES"
DOC_TYPE_SPEECH = "SPEECH"

#: The event types that get a Fed packet. FED_SPEECH is included because a
#: speech is read against the statement it follows — "what did the Committee
#: say, and what has this speaker said since" is one question — and a speech
#: page with no policy context is the least useful card on the site.
FED_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.FOMC_MEETING,
        EventType.FOMC_DECISION,
        EventType.FOMC_PRESS_CONFERENCE,
        EventType.FOMC_MINUTES,
        EventType.FED_SPEECH,
    }
)

#: The types whose event_key carries the MEETING's date and which therefore
#: identify a meeting on their own.
_MEETING_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.FOMC_MEETING,
        EventType.FOMC_DECISION,
        EventType.FOMC_PRESS_CONFERENCE,
        EventType.FOMC_MINUTES,
    }
)

#: The cross-asset panel the §45 windows are measured over. Deliberately FIVE
#: and deliberately not the eight-symbol macro panel: an FOMC decision is a
#: RATES event, so the long bond (TLT), the dollar (UUP) and gold (GLD) carry
#: as much of the reaction as the index does, while the belly of the curve
#: (IEF/SHY) and oil add rows nobody reads in a thirty-minute window. Every
#: symbol here costs a minute-bar window per backfill — roughly a thousand
#: rows — so the list is short on purpose.
FED_REACTION_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD", "UUP")

#: ET wall clocks the §45 windows are anchored to. Mirrored from
#: ``libs.event_calendar.fed`` (DECISION_ET / PRESS_CONFERENCE_ET) — the
#: statement drops at 14:00 and the Chair takes the podium at 14:30 — with the
#: presser's END at 15:30 because the Q&A runs about an hour. They are
#: CONVENTIONS, which is why the payload labels them rather than claiming them
#: as scraped facts.
STATEMENT_WINDOW_START_ET = (14, 0)
PRESS_CONFERENCE_START_ET = (14, 30)
PRESS_CONFERENCE_END_ET = (15, 30)

#: How far back a decision search looks for the previous meeting. Meetings are
#: roughly six weeks apart and the calendar has gaps (an unscheduled meeting is
#: skipped by the ingest, deliberately), so 120 days covers two cadences
#: without ever reaching into the meeting before last.
PREVIOUS_DECISION_LOOKBACK_DAYS = 120

#: The disclaimer every Fed payload carries at top level, so a reader who never
#: opens ``packet.disclaimers`` still cannot mistake the diff for a verdict.
FED_DISCLAIMER = " ".join(
    (
        SOURCE_AUTHORITATIVE_NOTE,
        NO_SINGLE_SCORE_NOTE,
        MARKET_PRICING_NOTE,
    )
)


# ---------------------------------------------------------------------------
# Which meeting is this event about
# ---------------------------------------------------------------------------


def _event_type_of(event_row: EventRow) -> EventType | None:
    """The row's typed event type, or ``None`` for a value the enum dropped.

    ``None`` rather than a raise, for the same reason ``event_macro`` does it:
    a registry row whose type string no longer parses is a data problem, and a
    read answering "this is not a Fed event" is a better failure than a 500
    out of the enum constructor.
    """
    try:
        return EventType(event_row.event_type)
    except ValueError:
        return None


def is_fed_event(event_row: EventRow) -> bool:
    """Whether this row gets a §42-§45 Fed packet."""
    etype = _event_type_of(event_row)
    return etype is not None and etype in FED_EVENT_TYPES


def meeting_date_of(event_row: EventRow) -> date | None:
    """The MEETING date this row is about, or ``None`` for a speech.

    Read from ``event_key`` first and from ``scheduled_at`` only as a fallback,
    and the order is load-bearing. The four FOMC_* rows for one meeting are
    keyed ``FOMC_DECISION:2026-07-29`` / ``FOMC_MINUTES:2026-08-19`` and so on
    — the key's date is the ROW's own ET date, which for the decision and the
    press conference IS the meeting's end but for the minutes is the release
    day three weeks later. The two are reconciled by
    :func:`previous_decision_row`, which searches the DECISION rows rather than
    trusting any single key to name the meeting.

    ``raw`` is not consulted because the registry does not store it: the
    ingest drops ``EventCandidate.raw`` when it writes ``events``, so the key
    and the timestamp are the only meeting facts a stored row carries.
    """
    etype = _event_type_of(event_row)
    if etype is None or etype not in _MEETING_EVENT_TYPES:
        return None
    key = (event_row.event_key or "").strip()
    if ":" in key:
        tail = key.rsplit(":", 1)[-1]
        try:
            return date.fromisoformat(tail)
        except ValueError:
            pass
    return event_date_et(event_row)


def _et_instant(day: date, hm: tuple[int, int]) -> datetime:
    """An ET wall clock on ``day`` as its UTC instant.

    Built in Eastern and converted, never by adding a fixed offset: a January
    decision is 19:00Z and a July decision is 18:00Z, and offset arithmetic
    would put the statement window an hour off for half the year.
    """
    return datetime(day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN).astimezone(
        timezone.utc
    )


async def previous_decision_row(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> EventRow | None:
    """The last FOMC_DECISION strictly BEFORE this event, at ``as_of``.

    "Before this event" and not "before now": an as-of replay run against a
    date in the past must find the decision that had happened THEN. The bound
    is the earlier of the event's own instant and ``as_of``, so replaying a
    future meeting from today still compares against the last decision that
    has actually occurred.

    A decision event is EXCLUDED from being its own predecessor by the strict
    ``<`` on ``scheduled_at``; a MINUTES row three weeks after its meeting
    correctly finds that meeting's decision, because the decision precedes the
    minutes' release.
    """
    moment = _as_utc(as_of)
    own = _as_utc(event_row.scheduled_at)
    bound = min(own, moment)
    floor = bound - timedelta(days=PREVIOUS_DECISION_LOOKBACK_DAYS)
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == EventType.FOMC_DECISION.value,
                    EventRow.scheduled_at < bound,
                    EventRow.scheduled_at >= floor,
                    EventRow.id != event_row.id,
                )
                .order_by(EventRow.scheduled_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .all()
    )
    return rows[0] if rows else None


async def _decision_before(
    session: AsyncSession, decision_row: EventRow
) -> EventRow | None:
    """The FOMC_DECISION immediately before ``decision_row`` — the diff's
    comparison document. No as-of bound is needed: ``decision_row`` is already
    as-of gated, and anything earlier than it is earlier than the gate."""
    floor = _as_utc(decision_row.scheduled_at) - timedelta(
        days=PREVIOUS_DECISION_LOOKBACK_DAYS
    )
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == EventType.FOMC_DECISION.value,
                    EventRow.scheduled_at < decision_row.scheduled_at,
                    EventRow.scheduled_at >= floor,
                )
                .order_by(EventRow.scheduled_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .all()
    )
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Stored documents
# ---------------------------------------------------------------------------


async def load_documents(
    session: AsyncSession,
    *,
    doc_type: str,
    meeting_dates: Sequence[date] = (),
    as_of: datetime | None = None,
) -> dict[date, FedDocumentRow]:
    """Stored documents of one type, keyed by meeting date, gated at ``as_of``.

    THE GATE IS IN THE QUERY (§14/§96), not applied to the result: the pure
    library must never see a document the caller could not have seen, and
    filtering afterwards leaves a window in which some later code path reads
    the ungated list.

    ``released_at IS NULL`` is EXCLUDED by the gate rather than admitted. An
    unknown publication instant cannot be resolved into "before your as_of",
    and admitting it would let the minutes appear on the meeting day, three
    weeks before they were written. Ungated reads (``as_of=None``) keep the
    NULL rows, which is what the backfill's idempotence check wants.
    """
    dates = [d for d in meeting_dates if d is not None]
    if not dates:
        return {}
    stmt = select(FedDocumentRow).where(
        FedDocumentRow.doc_type == doc_type,
        FedDocumentRow.meeting_date.in_(dates),
    )
    if as_of is not None:
        moment = _as_utc(as_of)
        stmt = stmt.where(
            FedDocumentRow.released_at.is_not(None),
            FedDocumentRow.released_at <= moment,
        )
    rows = (await session.execute(stmt.order_by(FedDocumentRow.meeting_date))).scalars().all()
    out: dict[date, FedDocumentRow] = {}
    for row in rows:
        if row.meeting_date is not None:
            out[row.meeting_date] = row
    return out


async def _load_speeches(
    session: AsyncSession, *, start: datetime, end: datetime
) -> list[FedDocumentRow]:
    """Stored SPEECH documents released in ``(start, end]``, oldest first."""
    rows = (
        (
            await session.execute(
                select(FedDocumentRow)
                .where(
                    FedDocumentRow.doc_type == DOC_TYPE_SPEECH,
                    FedDocumentRow.released_at.is_not(None),
                    FedDocumentRow.released_at > start,
                    FedDocumentRow.released_at <= end,
                )
                .order_by(FedDocumentRow.released_at)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


def _doc_to_mapping(row: FedDocumentRow | None) -> dict[str, Any] | None:
    """An ORM row as the mapping ``build_fed_packet`` expects.

    Field names are U1's ``FomcStatement`` names EXACTLY (``paragraphs``,
    ``vote``, ``target_range``, ``released_at``, ``meeting_date``), because the
    pure builder reads them by those names and a rename here would silently
    empty the diff rather than fail.

    ``released_at`` is RE-STAMPED to UTC: SQLite hands a
    ``DateTime(timezone=True)`` column back NAIVE, and ``build_fed_packet``'s
    as-of gate DROPS a document whose instant is naive (correctly — guessing
    the zone of a release instant is how a 14:00 ET statement becomes visible
    at 14:00 UTC, four hours early). Without this line every stored document
    would vanish from the packet under the test harness.
    """
    if row is None:
        return None
    parsed = dict(row.parsed or {})
    return {
        "url": row.url,
        "title": row.title,
        "doc_type": row.doc_type,
        "meeting_date": row.meeting_date,
        "released_at": _as_utc(row.released_at) if row.released_at else None,
        "paragraphs": list(row.paragraphs or []),
        "text": row.text or "",
        "vote": dict(parsed.get("vote") or {}),
        "target_range": parsed.get("target_range"),
        "speaker": parsed.get("speaker"),
    }


# ---------------------------------------------------------------------------
# The §45 reaction windows
# ---------------------------------------------------------------------------


async def _minute_bars_around(
    session: AsyncSession,
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    as_of: datetime,
) -> dict[str, list]:
    """Stored minute bars per symbol in ``[start, end]``, gated at ``as_of``.

    Symbols with NO stored bars are omitted entirely rather than mapped to an
    empty list, so the caller can tell "nothing is stored for this decision"
    (fall back to daily) from "SPY moved 0.0%" — which are different claims and
    only one of them is ever true (§44 rule 18).
    """
    moment = _as_utc(as_of)
    out: dict[str, list] = {}
    for symbol in symbols:
        rows = (
            (
                await session.execute(
                    select(StockBar1mRow)
                    .where(
                        StockBar1mRow.ticker == symbol,
                        StockBar1mRow.ts >= start,
                        StockBar1mRow.ts <= end,
                        StockBar1mRow.ts <= moment,
                    )
                    .order_by(StockBar1mRow.ts)
                )
            )
            .scalars()
            .all()
        )
        if rows:
            out[symbol] = to_minute_bars(list(rows))
    return out


async def _daily_bars_around(
    session: AsyncSession,
    symbols: Sequence[str],
    *,
    decision_day: date,
    as_of: datetime,
) -> dict[str, list[DailyBar]]:
    """Stored daily bars per symbol spanning the decision, gated at ``as_of``.

    The window opens thirty days before the decision (the pre-event close has
    to exist through a holiday week) and closes at ``as_of``'s ET date — a
    daily bar is not knowable until its session ends, so gating on the bar's
    own date rather than on an instant is the honest rule for a daily series.
    """
    lo = decision_day - timedelta(days=30)
    hi = _as_utc(as_of).astimezone(EASTERN).date()
    out: dict[str, list[DailyBar]] = {}
    for symbol in symbols:
        rows = (
            (
                await session.execute(
                    select(StockBarDaily)
                    .where(
                        StockBarDaily.ticker == symbol,
                        StockBarDaily.ts >= lo,
                        StockBarDaily.ts <= hi,
                    )
                    .order_by(StockBarDaily.ts)
                )
            )
            .scalars()
            .all()
        )
        bars = _to_daily_bars(rows)
        if bars:
            out[symbol] = bars
    return out


async def _reactions_for(
    session: AsyncSession, decision_row: EventRow | None, *, as_of: datetime
) -> dict[str, Any]:
    """The §45 reaction block for the previous decision — 1m if stored, else
    daily, else empty.

    THE TWO WINDOWS ARE PREFERRED AND THE DAILY IS A STATED FALLBACK. A daily
    bar spans the statement AND the press conference and cannot tell them
    apart, so it is used only when no minute bar exists — and when it is used
    the payload says ``separated: false`` with the label "daily (no intraday
    bars)" rather than reporting the blend under the statement's name.

    Returns ``{}`` when neither is stored, which the pure builder renders as
    ``previous_reaction.available: false`` with a reason. Nothing is
    fabricated to fill the silence.
    """
    if decision_row is None:
        return {}
    day = meeting_date_of(decision_row) or event_date_et(decision_row)
    decision_at = _et_instant(day, STATEMENT_WINDOW_START_ET)
    presser_at = _et_instant(day, PRESS_CONFERENCE_START_ET)
    presser_end = _et_instant(day, PRESS_CONFERENCE_END_ET)

    minute = await _minute_bars_around(
        session,
        FED_REACTION_SYMBOLS,
        start=decision_at,
        end=presser_end,
        as_of=as_of,
    )
    if minute:
        return fomc_reaction_windows(
            minute,
            decision_at_utc=decision_at,
            press_conf_at_utc=presser_at,
            press_conf_end_utc=presser_end,
        )

    daily = await _daily_bars_around(
        session, FED_REACTION_SYMBOLS, decision_day=day, as_of=as_of
    )
    if daily:
        return fomc_reaction_daily(
            daily,
            decision_at_utc=decision_at,
            session=EventSession.DURING_MARKET,
        )
    return {}


# ---------------------------------------------------------------------------
# Macro prints (pass-through)
# ---------------------------------------------------------------------------


async def _macro_prints(session: AsyncSession, *, as_of: datetime) -> dict[str, Any]:
    """The inflation / labor / growth backdrop, from Phase G's stored prints.

    PASS-THROUGH, NOT A SECOND COMPUTATION. The latest visible print per role
    is read out of ``macro_observations`` through the SAME pure helpers the
    macro packet uses (``series_for`` + ``derive_prints``), so the number on
    the Fed card and the number on the CPI card can never disagree. A role
    with nothing stored is absent from the dict, and the packet's
    ``data.available`` goes false — never a zero.

    Wrapped in a guard because Phase G's module is a soft dependency of this
    one: an FOMC packet with no macro backdrop is still the whole §44 diff, and
    an ImportError here must not cost the reader the statement.
    """
    moment = _as_utc(as_of)
    try:
        from libs.trading_core.events.macro import derive_prints, series_for

        from .event_macro import load_observations, _to_observation
    except Exception:  # noqa: BLE001 — Phase G is optional to this payload
        return {}

    roles = {
        "inflation": EventType.CPI,
        "labor": EventType.EMPLOYMENT_REPORT,
        "growth": EventType.GDP,
    }
    out: dict[str, Any] = {}
    for role, etype in roles.items():
        specs = series_for(etype)
        if not specs:
            continue
        spec = specs[0]
        stored = await load_observations(session, [spec.series_id])
        rows = stored.get(spec.series_id) or []
        if not rows:
            continue
        schedule = {
            row.period: _as_utc(row.release_at)
            for row in rows
            if row.release_at is not None
        }
        prints = derive_prints(
            [_to_observation(row) for row in rows], spec, schedule=schedule or None
        )
        visible = [
            p
            for p in prints
            if p.release_at is not None and _as_utc(p.release_at) <= moment
        ]
        if not visible:
            continue
        latest = visible[-1]
        out[role] = {
            "series_id": latest.series_id,
            "label": spec.label,
            "period": latest.period,
            "value": latest.value,
            "value_raw": latest.value_raw,
            "unit": latest.unit,
            "transform": latest.transform,
            "release_at": _as_utc(latest.release_at).isoformat(),
            "release_time_basis": latest.release_time_basis,
            "reason": latest.reason,
        }
    return out


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


def _event_ref(event_row: EventRow) -> dict[str, Any]:
    return {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "title": event_row.title,
        "scheduled_at": _as_utc(event_row.scheduled_at).isoformat(),
        "speaker": event_row.speaker,
    }


async def build_fed_payload(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> dict[str, Any]:
    """The whole §42-§45 Fed block for one event, as of one instant.

    Order of operations is the contract: resolve the previous decision and the
    one before it -> load their STORED statements, the previous meeting's
    minutes and the speeches since, each gated at ``as_of`` in SQL -> measure
    the previous decision's two reaction windows over as-of-gated bars -> hand
    all of it to the pure builder, which applies the gate a second time and
    computes the diff. Nothing is measured before the gate and nothing is
    parsed here.

    THIS FUNCTION NEVER FETCHES (§27; audit §7.2 rule 1). It takes no provider
    name and imports no HTTP client; a meeting nobody backfilled answers
    ``coverage.previous_statement: false`` naming the POST that would fix it.
    federalreserve.gov rate-limits by User-Agent, so a read that lazily
    fetched four documents per page load would get this platform's contact
    address throttled — and the throttle would land on the backfill.

    A NON-FED event answers ``{"available": false, "reason": ...}`` rather than
    404: the row exists, it simply has no Committee behind it, and a 404 would
    say the event does not exist.
    """
    moment = _as_utc(as_of)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "as_of": moment.isoformat(),
        "model_version": FED_INTEL_MODEL_VERSION,
    }
    if not is_fed_event(event_row):
        return {
            **base,
            "available": False,
            "reason": (
                f"not a Federal Reserve event ({event_row.event_type}) — the "
                "§42 packet describes an FOMC meeting or a Fed speech"
            ),
            "disclaimer": FED_DISCLAIMER,
        }

    prev_decision = await previous_decision_row(session, event_row, as_of=moment)
    prev_prev_decision = (
        await _decision_before(session, prev_decision)
        if prev_decision is not None
        else None
    )

    prev_day = meeting_date_of(prev_decision) if prev_decision is not None else None
    prev_prev_day = (
        meeting_date_of(prev_prev_decision) if prev_prev_decision is not None else None
    )

    statements = await load_documents(
        session,
        doc_type=DOC_TYPE_STATEMENT,
        meeting_dates=[d for d in (prev_day, prev_prev_day) if d is not None],
        as_of=moment,
    )
    minutes = await load_documents(
        session,
        doc_type=DOC_TYPE_MINUTES,
        meeting_dates=[d for d in (prev_prev_day, prev_day) if d is not None],
        as_of=moment,
    )
    # The minutes a reader can HAVE at as_of are the previous meeting's when
    # three weeks have passed, and otherwise the meeting before it. Preferring
    # the more recent one and falling back is what makes the card useful in the
    # three-week gap rather than empty.
    minutes_row = None
    minutes_day = None
    for day in (prev_day, prev_prev_day):
        if day is not None and day in minutes:
            minutes_row, minutes_day = minutes[day], day
            break

    speech_rows: list[FedDocumentRow] = []
    if prev_decision is not None:
        speech_rows = await _load_speeches(
            session, start=_as_utc(prev_decision.scheduled_at), end=moment
        )

    reactions = await _reactions_for(session, prev_decision, as_of=moment)
    macro_prints = await _macro_prints(session, as_of=moment)

    packet = build_fed_packet(
        current_event=_event_ref(event_row),
        previous_decision=(
            _event_ref(prev_decision) if prev_decision is not None else None
        ),
        prev_statement=_doc_to_mapping(statements.get(prev_day)),
        prev_prev_statement=_doc_to_mapping(statements.get(prev_prev_day)),
        prev_minutes=_doc_to_mapping(minutes_row),
        speeches_since=[
            {
                "speaker": (row.parsed or {}).get("speaker"),
                "title": row.title,
                "released_at": _as_utc(row.released_at) if row.released_at else None,
                "url": row.url,
            }
            for row in speech_rows
        ],
        macro_prints=macro_prints,
        reactions=reactions,
        as_of=moment,
    )

    coverage = dict(packet.get("coverage") or {})
    if not coverage.get("previous_statement"):
        coverage["previous_statement_reason"] = (
            "no FOMC statement is stored for the previous decision at this "
            "as_of — press Backfill to fetch it from federalreserve.gov"
        )
    if prev_decision is None:
        coverage["previous_decision_reason"] = (
            "no FOMC_DECISION event is registered in the "
            f"{PREVIOUS_DECISION_LOOKBACK_DAYS} days before this event — run "
            "the calendar ingest first"
        )
    packet["coverage"] = coverage

    return json_safe(
        {
            **base,
            "available": True,
            "packet": packet,
            "reaction_symbols": list(FED_REACTION_SYMBOLS),
            "minutes_meeting_date": minutes_day,
            "tiers": {"documents": TIER_DATA, "diff": TIER_QUANT, "reaction": TIER_QUANT},
            "disclaimer": FED_DISCLAIMER,
        }
    )


async def fed_context_section(
    session: AsyncSession, event_row: EventRow, as_of: datetime
) -> dict[str, Any] | None:
    """The evidence bundle's ``macro_context.fed`` block, or ``None`` (§46).

    ``None`` for a non-Fed event rather than an ``available: false`` stub: this
    block rides INSIDE the macro_context Phase G already fills, and adding a
    dead key to every earnings bundle would change the bundle digest for six
    thousand rows to say "this is not an FOMC meeting", which the event type
    beside it already says.

    Catches its own failures for the same reason every other section does — one
    dead block must never cost the reader the other six.
    """
    if not is_fed_event(event_row):
        return None
    try:
        payload = await build_fed_payload(session, event_row, as_of=as_of)
    except Exception as exc:  # noqa: BLE001 — never sink the bundle
        logger.warning("fed context for event %s failed: %s", event_row.id, exc)
        return {
            "tier": TIER_QUANT,
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "disclaimer": FED_DISCLAIMER,
            "model_version": FED_INTEL_MODEL_VERSION,
        }
    packet = payload.get("packet") or {}
    return {
        "tier": TIER_QUANT,
        "available": bool(payload.get("available")),
        "reason": payload.get("reason"),
        "kind": "fomc_packet",
        "event_type": event_row.event_type,
        "packet": packet,
        "coverage": packet.get("coverage"),
        "disclaimer": FED_DISCLAIMER,
        "model_version": FED_INTEL_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# The backfill (USER-triggered writes)
# ---------------------------------------------------------------------------


async def _upsert_document(
    session: AsyncSession,
    doc: Any,
    *,
    provider: str,
    event_id: int | None,
    meeting_date: date | None,
) -> str:
    """Store one fetched document, keyed on its URL. ``"STORED"`` | ``"UPDATED"``.

    IDEMPOTENT ON THE URL because one Fed URL is one document forever. A second
    press of the button re-parses the same page and overwrites the same row
    rather than accumulating a second copy that would make the diff pick an
    arbitrary one of two identical statements.

    The UPDATE path is deliberate rather than a no-op: the Fed does correct a
    page occasionally (a misspelled dissenter's name), and §44's rule is that
    the SOURCE document is authoritative — so the freshest parse of the same
    URL wins. What it can never do is create a SECOND row for the same words.
    """
    existing = (
        (
            await session.execute(
                select(FedDocumentRow).where(FedDocumentRow.url == doc.url)
            )
        )
        .scalars()
        .first()
    )
    parsed: dict[str, Any] = {}
    vote = getattr(doc, "vote", None)
    if vote:
        parsed["vote"] = dict(vote)
    target = getattr(doc, "target_range", None)
    if target:
        parsed["target_range"] = dict(target)
    if getattr(doc, "speaker", None):
        parsed["speaker"] = doc.speaker

    released = doc.released_at
    if released is not None:
        released = _as_utc(released)
    resolved_meeting = meeting_date if meeting_date is not None else doc.meeting_date

    if existing is None:
        session.add(
            FedDocumentRow(
                doc_type=doc.doc_type,
                meeting_date=resolved_meeting,
                event_id=event_id,
                url=doc.url,
                title=doc.title or "",
                released_at=released,
                text=doc.text or "",
                paragraphs=list(doc.paragraphs or []),
                parsed=parsed,
                provider=provider,
            )
        )
        return "STORED"

    existing.doc_type = doc.doc_type
    existing.meeting_date = resolved_meeting
    if event_id is not None:
        existing.event_id = event_id
    existing.title = doc.title or ""
    existing.released_at = released
    existing.text = doc.text or ""
    existing.paragraphs = list(doc.paragraphs or [])
    existing.parsed = parsed
    existing.provider = provider
    existing.fetched_at = datetime.now(timezone.utc)
    return "UPDATED"


def _outcome(
    kind: str, *, status: str, reason: str | None = None, **extra: Any
) -> dict[str, Any]:
    return {"kind": kind, "status": status, "reason": reason, **extra}


async def backfill_fed(
    session: AsyncSession,
    event_row: EventRow,
    *,
    settings,
    as_of: datetime | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    """USER action: fetch and store this meeting's documents and its bars.

    FOUR FETCH GROUPS, EACH INDEPENDENT, EVERY OUTCOME REPORTED. The previous
    decision's statement, the one before it (the diff needs both), the previous
    meeting's minutes, and the speeches given since — plus the §45 minute-bar
    windows for the five reaction symbols. A failure in any one of them is a
    named row in the response rather than an exception, because a packet with
    the diff but no minutes is still most of the §44 answer and refusing the
    whole thing would be the worse trade.

    ONE RSS CALL, NOT FOUR. ``list_press_monetary`` is fetched once and the
    resulting items are passed into every ``fetch_statement`` / ``fetch_minutes``
    call, because that feed is where the Fed's own publication instants come
    from and re-fetching it per document would quadruple the request count for
    an identical answer. It is also what lets ``fetch_statement`` decline to
    issue an HTTP request at all for a statement released after ``as_of``.

    IDEMPOTENT ON THE DOCUMENT URL. Two presses re-parse the same pages and
    overwrite the same rows; the minute-bar helper refuses to refetch a window
    it already stored. Pressing twice costs requests and changes nothing else.

    Writes a SYSTEM-of-record ``DATA_BACKFILL`` audit row carrying
    ``kind: "event_fed"`` in the same transaction as the data (rule 12,
    ADR-003).
    """
    moment = _as_utc(as_of) if as_of is not None else datetime.now(timezone.utc)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "as_of": moment.isoformat(),
    }
    if not is_fed_event(event_row):
        return {
            **base,
            "available": False,
            "reason": f"not a Federal Reserve event ({event_row.event_type})",
            "documents": [],
            "bars": [],
            "counts": {"documents": 0, "bars": 0},
        }

    prev_decision = await previous_decision_row(session, event_row, as_of=moment)
    if prev_decision is None:
        return {
            **base,
            "available": False,
            "reason": (
                "no FOMC_DECISION event is registered in the "
                f"{PREVIOUS_DECISION_LOOKBACK_DAYS} days before this event — "
                "there is no meeting to fetch documents for; run the calendar "
                "ingest first"
            ),
            "documents": [],
            "bars": [],
            "counts": {"documents": 0, "bars": 0},
        }
    prev_prev_decision = await _decision_before(session, prev_decision)

    client = provider
    if client is None:
        try:
            from libs.event_calendar import fed_documents_provider

            client = fed_documents_provider(settings)
        except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
            return {
                **base,
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "documents": [],
                "bars": [],
                "counts": {"documents": 0, "bars": 0},
            }

    outcomes: list[dict[str, Any]] = []
    stored = 0

    # --- 0. the publication feed, fetched ONCE ----------------------------
    rss_items: list[Any] = []
    try:
        rss_items = list(client.list_press_monetary(as_of=moment))
    except (CalendarProviderError, MarketDataError) as exc:
        outcomes.append(_outcome("rss", status="ERROR", reason=str(exc)))
    except Exception as exc:  # noqa: BLE001
        outcomes.append(
            _outcome("rss", status="ERROR", reason=f"{type(exc).__name__}: {exc}")
        )
    else:
        outcomes.append(_outcome("rss", status="OK", items=len(rss_items)))

    # --- 1 & 2. the two statements the diff compares ----------------------
    for label, row in (
        ("previous_statement", prev_decision),
        ("compared_statement", prev_prev_decision),
    ):
        if row is None:
            outcomes.append(
                _outcome(
                    label,
                    status="SKIPPED",
                    reason="no earlier FOMC_DECISION is registered to compare against",
                )
            )
            continue
        day = meeting_date_of(row)
        if day is None:
            outcomes.append(
                _outcome(label, status="SKIPPED", reason="decision row carries no date")
            )
            continue
        try:
            doc = client.fetch_statement(day, as_of=moment, rss_items=rss_items)
        except (CalendarProviderError, MarketDataError, CapabilityNotAvailable) as exc:
            outcomes.append(
                _outcome(label, status="ERROR", reason=str(exc), meeting_date=day.isoformat())
            )
            continue
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                _outcome(
                    label,
                    status="ERROR",
                    reason=f"{type(exc).__name__}: {exc}",
                    meeting_date=day.isoformat(),
                )
            )
            continue
        if doc is None:
            outcomes.append(
                _outcome(
                    label,
                    status="NOT_YET_RELEASED",
                    reason=f"released after as_of {moment.isoformat()}",
                    meeting_date=day.isoformat(),
                )
            )
            continue
        status = await _upsert_document(
            session, doc, provider=doc.source_name, event_id=row.id, meeting_date=day
        )
        stored += 1
        outcomes.append(
            _outcome(
                label,
                status=status,
                url=doc.url,
                meeting_date=day.isoformat(),
                paragraphs=len(doc.paragraphs or []),
            )
        )

    # --- 3. the previous meeting's minutes --------------------------------
    minutes_day = meeting_date_of(prev_decision)
    if minutes_day is None:
        outcomes.append(
            _outcome("minutes", status="SKIPPED", reason="decision row carries no date")
        )
    else:
        try:
            doc = client.fetch_minutes(minutes_day, as_of=moment, rss_items=rss_items)
        except (CalendarProviderError, MarketDataError, CapabilityNotAvailable) as exc:
            outcomes.append(
                _outcome(
                    "minutes",
                    status="ERROR",
                    reason=str(exc),
                    meeting_date=minutes_day.isoformat(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                _outcome(
                    "minutes",
                    status="ERROR",
                    reason=f"{type(exc).__name__}: {exc}",
                    meeting_date=minutes_day.isoformat(),
                )
            )
        else:
            if doc is None:
                outcomes.append(
                    _outcome(
                        "minutes",
                        status="NOT_YET_RELEASED",
                        reason=(
                            "the minutes of this meeting had not been published "
                            f"at as_of {moment.isoformat()}"
                        ),
                        meeting_date=minutes_day.isoformat(),
                    )
                )
            else:
                status = await _upsert_document(
                    session,
                    doc,
                    provider=doc.source_name,
                    event_id=None,
                    meeting_date=minutes_day,
                )
                stored += 1
                outcomes.append(
                    _outcome(
                        "minutes",
                        status=status,
                        url=doc.url,
                        meeting_date=minutes_day.isoformat(),
                        paragraphs=len(doc.paragraphs or []),
                        released_at=(
                            _as_utc(doc.released_at).isoformat()
                            if doc.released_at
                            else None
                        ),
                    )
                )

    # --- 4. the speeches given since the previous decision ----------------
    speech_events = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == EventType.FED_SPEECH.value,
                    EventRow.scheduled_at > prev_decision.scheduled_at,
                    EventRow.scheduled_at <= moment,
                    EventRow.source_url.is_not(None),
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    for row in speech_events:
        url = (row.source_url or "").strip()
        if not url:
            continue
        try:
            doc = client.fetch_speech(url, as_of=moment)
        except (CalendarProviderError, MarketDataError, CapabilityNotAvailable) as exc:
            outcomes.append(_outcome("speech", status="ERROR", reason=str(exc), url=url))
            continue
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                _outcome(
                    "speech",
                    status="ERROR",
                    reason=f"{type(exc).__name__}: {exc}",
                    url=url,
                )
            )
            continue
        if doc is None:
            outcomes.append(
                _outcome(
                    "speech",
                    status="NOT_YET_RELEASED",
                    reason=f"released after as_of {moment.isoformat()}",
                    url=url,
                )
            )
            continue
        status = await _upsert_document(
            session, doc, provider=doc.source_name, event_id=row.id, meeting_date=None
        )
        stored += 1
        outcomes.append(
            _outcome("speech", status=status, url=doc.url, speaker=doc.speaker)
        )

    # --- 5. the §45 minute windows around the previous decision -----------
    provider_name = getattr(settings, "market_data_provider", "") or ""
    bars_written = 0
    bar_results: list[dict[str, Any]] = []
    for symbol in FED_REACTION_SYMBOLS:
        try:
            result = await ensure_event_window_bars(
                session, symbol, prev_decision, provider_name, now=moment
            )
        except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
            bar_results.append(
                {
                    "symbol": symbol,
                    "fetched": False,
                    "bars": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        bars_written += int(result.get("bars") or 0)
        bar_results.append(
            {
                "symbol": symbol,
                "fetched": bool(result.get("fetched")),
                "bars": int(result.get("bars") or 0),
                "stored_bars": result.get("stored_bars"),
                "reason": result.get("reason"),
            }
        )

    counts = {"documents": stored, "bars": bars_written}
    await audit.record(
        session,
        actor_type=ActorType.USER,
        action=AuditAction.DATA_BACKFILL,
        entity_type="event",
        entity_id=str(event_row.id),
        details={
            "kind": "event_fed",
            "event_key": event_row.event_key,
            "event_type": event_row.event_type,
            "as_of": moment.isoformat(),
            "counts": counts,
            "previous_decision": prev_decision.event_key,
        },
    )
    await session.commit()

    return json_safe(
        {
            **base,
            "available": True,
            "counts": counts,
            "previous_decision": _event_ref(prev_decision),
            "documents": outcomes,
            "bars": bar_results,
            "disclaimer": FED_DISCLAIMER,
        }
    )
