"""Event calendar ingestion seam — the gateway half of the event registry
(event spec §7-§13; audit §5.1 "Gateway seams", §11.1 Phase B).

Shape copied deliberately from :mod:`apps.gateway.risk_snapshot`, because
that module already solved every problem this one has:

- the tick is SPLIT OUT of the loop (:func:`run_calendar_ingest`) so tests
  can drive one deterministic ingestion without a background task — httpx
  ``ASGITransport`` does not run the lifespan, so the loop never starts under
  the suite (``main.py::lifespan``);
- the cadence is computed against the **America/New_York** day, not a stored
  date column (:func:`new_york_today`);
- a source that cannot answer produces a NAMED SKIP, never a fabricated row;
- ``asyncio.CancelledError`` is re-raised and everything else is logged and
  swallowed, so one bad provider can never end the loop.

WHAT ONE TICK DOES

1. Resolve the ticker universe: watchlist ∪ trading pool ∪ open positions.
   These are the only symbols the platform is allowed to store bars for
   (plan §4.2) and the only ones an earnings card can be relevant to (§12).
2. For each configured provider, decide whether its cadence is due
   (:data:`PROVIDER_MIN_INTERVAL_HOURS`) from ``event_ingest_state.last_ok_at``
   — SEC and the Fed are daily sources, exchange calendars weekly. ``force=True``
   (the manual refresh button) ignores the cadence.
3. Fetch in a worker thread (the providers are sync httpx, house rule) inside
   a per-provider ``try``. A 403/timeout/parse failure updates THAT provider's
   ingest-state row with the honest error and leaves every other provider's
   rows committed (§8 "calendar ingestion should survive individual provider
   failures").
4. Upsert each candidate through U1's :func:`merge`: match on ``event_key``
   first, then the ``same_event`` drift windows (EARNINGS ±21d, FOMC_MINUTES
   ±7d) so an ESTIMATED date being replaced by the confirmed one updates the
   card instead of duplicating it. Created rows audit EVENT_DISCOVERED; a
   merge whose ``change`` is a real date/status move audits EVENT_UPDATED.
5. Upsert the market calendar (±400 days) and re-classify sessions against it
   — a 13:00 ET release on a half-day is AFTER_MARKET, which the default
   09:30-16:00 session would get wrong.
6. Re-score importance for events in the horizon with the DB's own relevance
   inputs (position > pool > watchlist > market-wide > other).
7. T-minus alert: CONFIRMED/REVISED events landing inside
   ``settings.event_horizon_alert_days`` write EVENT_APPROACHING **exactly
   once per (event_key, horizon)** — checked against the AUDIT TABLE, not an
   in-process set, so a restart cannot re-alert (ADR-007 gives no leader
   election; correctness may not ride on single-process state). ESTIMATED
   events NEVER alert (§11).
8. One CALENDAR_INGESTED audit row per tick carrying the per-provider counts
   and the capability report the API surfaces.

NO FABRICATION: every number in the returned dict is a count of rows that
actually exist. A provider that could not answer contributes a named skip and
an error string, never an invented event.
"""
import asyncio
import logging
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.common.telemetry import REGISTRY
from libs.event_calendar import configured_providers
from libs.event_calendar.provider import (
    CapabilityNotAvailable,
    MarketDataError,
    MarketDay,
    ProviderNotConfigured,
)
from libs.trading_core.events import (
    EARNINGS_DRIFT_WINDOW,
    EASTERN,
    MINUTES_DRIFT_WINDOW,
    Event,
    EventCandidate,
    classify_session,
    default_relevance_tier,
    link_previous_events,
    merge,
    same_event,
    score_importance,
)
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from . import audit
from .db import (
    AuditEvent,
    EventIngestStateRow,
    EventRow,
    MarketCalendarRow,
    Position,
    SessionLocal,
    TradingPoolItem,
    WatchlistItem,
)

logger = logging.getLogger(__name__)

#: The exchange clock. Same constant name and zone as risk_snapshot.py:142 —
#: the NY calendar day is what "one ingest per day" and the T-minus horizon
#: are both statements about.
NEW_YORK = EASTERN

#: How far ahead/behind a tick looks for events, in days. Ahead: wide enough
#: that a quarterly earnings cadence estimate and the next two FOMC meetings
#: are always inside it. Behind: ~3.3 years so the last 12 quarterly releases
#: (spec §19 "LAST 4 / 8 / 12" event-move distributions, Phase C) are kept —
#: the SEC submissions fetch is one request regardless, this only widens the
#: filter. Still narrow enough that no provider is asked for a decade.
LOOKAHEAD_DAYS = 120
LOOKBACK_DAYS = 1200

#: Market-calendar horizon (±400 days): far enough back for the previous
#: event's session to be classifiable, far enough forward to cover any
#: estimated date the cadence model produces.
CALENDAR_LOOKBACK_DAYS = 400
CALENDAR_LOOKAHEAD_DAYS = 400

#: Minimum hours between successful fetches, per provider. SEC EDGAR and the
#: Fed publish on a daily cadence and rate-limit hard; exchange calendars and
#: holiday tables change a handful of times a year. Re-asking more often
#: costs the source's goodwill and buys nothing. ``force=True`` bypasses it.
PROVIDER_MIN_INTERVAL_HOURS: dict[str, float] = {
    "sec_edgar": 20.0,
    "fed": 20.0,
    "alpaca_calendar": 24.0 * 7,
    "massive_calendar": 24.0 * 7,
    "stub": 0.0,
}
#: Default for a provider not named above (daily-source assumption).
DEFAULT_MIN_INTERVAL_HOURS = 20.0

#: The audit ``entity_type`` every event row is recorded under. entity_id is
#: the numeric ``events.id`` (``audit_events.entity_id`` is VARCHAR(64) in
#: Postgres and long FED_SPEECH natural keys overflowed it live on
#: 2026-08-19); the natural ``event_key`` always travels in ``details`` so
#: the trail stays joinable across environments.
ENTITY_TYPE = "event"

#: ``change`` words from U1's merge that represent a real, user-visible move
#: and therefore an EVENT_UPDATED audit row. "reverified"/"metadata" are
#: bookkeeping — auditing them would bury the real changes in noise.
MATERIAL_CHANGES = frozenset({"rescheduled", "confirmed", "revised", "canceled"})

#: SQL PREFILTER for the drift-window lookup, taken from U1's own constants
#: rather than restated. The prefilter only narrows which rows are loaded;
#: ``same_event`` remains the sole authority on whether two rows are the same
#: event. Deriving the bound here means a change to U1's window can never
#: leave this query silently too narrow to find the row the matcher would
#: have accepted.
_DRIFT_WINDOWS: dict[EventType, timedelta] = {
    EventType.EARNINGS: EARNINGS_DRIFT_WINDOW,
    EventType.FOMC_MINUTES: MINUTES_DRIFT_WINDOW,
}

CALENDAR_INGESTS_TOTAL = REGISTRY.counter(
    "calendar_ingests_total",
    "Completed event-calendar ingestion ticks (event spec §8).",
)
CALENDAR_PROVIDER_FAILURES_TOTAL = REGISTRY.counter(
    "calendar_provider_failures_total",
    "Event-calendar provider fetches that failed, labeled by provider "
    "(audit §6: failure isolation is per-adapter, not per-loop).",
    ("provider",),
)
EVENTS_DISCOVERED_TOTAL = REGISTRY.counter(
    "events_discovered_total",
    "Event rows created by calendar ingestion (event spec §5).",
)
EVENT_ALERTS_TOTAL = REGISTRY.counter(
    "event_alerts_total",
    "EVENT_APPROACHING audit rows written — exactly one per (event, horizon).",
)


def new_york_today() -> date:
    """Today's date on the exchange calendar the cadence rules use."""
    return datetime.now(NEW_YORK).date()


def _utc(value: datetime | None) -> datetime | None:
    """Read a stored timestamp as UTC.

    SQLite drops the timezone on a ``DateTime(timezone=True)`` column (the
    same caveat handled at risk_snapshot.py:1379 and order_sync.py:171), so a
    naive value read back is UTC by construction, not by assumption.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# ORM <-> domain
# ---------------------------------------------------------------------------


def row_to_event(row: EventRow) -> Event:
    """Hydrate the pure :class:`Event` value from its stored row.

    Enum columns are stored as their string values; an unrecognised value
    would mean the enum shrank under a live table, which is a programming
    error worth raising on rather than silently coercing.
    """
    return Event(
        event_id=row.id,
        event_key=row.event_key,
        event_type=EventType(row.event_type),
        title=row.title,
        ticker=row.ticker,
        company_id=row.company_id,
        scheduled_at=_utc(row.scheduled_at),
        event_timezone=row.event_timezone,
        session=EventSession(row.session),
        status=EventStatus(row.status),
        source=EventSourceKind(row.source),
        source_name=row.source_name,
        source_url=row.source_url,
        source_event_id=row.source_event_id,
        last_verified_at=_utc(row.last_verified_at),
        previous_event_id=row.previous_event_id,
        comparison_reason=row.comparison_reason,
        importance=row.importance,
        series_id=row.series_id,
        agency=row.agency,
        release_period=row.release_period,
        fiscal_quarter=row.fiscal_quarter,
        fiscal_year=row.fiscal_year,
        speaker=row.speaker,
        topic=row.topic,
        revision_history=tuple(row.revision_history or ()),
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )


def apply_event_to_row(row: EventRow, event: Event) -> None:
    """Write a merged :class:`Event` back onto its row.

    ``event_key`` IS in this list on purpose: U1's merge re-keys the row when
    an accepted date move changes the ET date the natural key embeds. A row
    that kept its old key after a confirmed reschedule would be re-created
    under the new key on the very next tick — the duplicate card the ±21-day
    drift window exists to prevent.
    """
    row.event_key = event.event_key
    row.event_type = event.event_type.value
    row.title = event.title
    row.ticker = event.ticker
    row.company_id = event.company_id
    row.scheduled_at = event.scheduled_at
    row.event_timezone = event.event_timezone
    row.session = event.session.value
    row.status = event.status.value
    row.source = event.source.value
    row.source_name = event.source_name
    row.source_url = event.source_url
    row.source_event_id = event.source_event_id
    row.last_verified_at = event.last_verified_at
    row.importance = event.importance
    row.series_id = event.series_id
    row.agency = event.agency
    row.release_period = event.release_period
    row.fiscal_quarter = event.fiscal_quarter
    row.fiscal_year = event.fiscal_year
    row.speaker = event.speaker
    row.topic = event.topic
    row.revision_history = [dict(entry) for entry in event.revision_history]
    row.updated_at = event.updated_at


def _new_row(event: Event) -> EventRow:
    row = EventRow(
        event_key=event.event_key,
        event_type=event.event_type.value,
        title=event.title,
        ticker=event.ticker,
        company_id=event.company_id,
        scheduled_at=event.scheduled_at,
        event_timezone=event.event_timezone,
        session=event.session.value,
        status=event.status.value,
        source=event.source.value,
        source_name=event.source_name,
        source_url=event.source_url,
        source_event_id=event.source_event_id,
        last_verified_at=event.last_verified_at,
        importance=event.importance,
        series_id=event.series_id,
        agency=event.agency,
        release_period=event.release_period,
        fiscal_quarter=event.fiscal_quarter,
        fiscal_year=event.fiscal_year,
        speaker=event.speaker,
        topic=event.topic,
        revision_history=[dict(entry) for entry in event.revision_history],
    )
    if event.created_at is not None:
        row.created_at = event.created_at
        row.updated_at = event.updated_at or event.created_at
    return row


# ---------------------------------------------------------------------------
# Relevance (§12) — the DB inputs the pure importance model cannot see
# ---------------------------------------------------------------------------


async def relevance_inputs(session: AsyncSession) -> dict[str, set[str]]:
    """The three ticker sets §12 ranks events by.

    Read once per tick (and once per API request) rather than per event: the
    lists are tiny and the alternative is N queries per horizon page.
    """
    positions = set(
        (
            await session.execute(
                select(Position.ticker).where(Position.status == "OPEN")
            )
        )
        .scalars()
        .all()
    )
    pool = set((await session.execute(select(TradingPoolItem.ticker))).scalars().all())
    watchlist = set((await session.execute(select(WatchlistItem.ticker))).scalars().all())
    return {"positions": positions, "pool": pool, "watchlist": watchlist}


def relevance_tier(event_type: EventType, ticker: str | None, sets: dict[str, set[str]]) -> str:
    """§12 ladder: POSITION > TRADING_POOL > WATCHLIST > MARKET_WIDE > OTHER.

    A macro or Fed event has no ticker to be relevant *to*, so it falls to
    MARKET_WIDE by type (U1's ``default_relevance_tier``) — it is important
    because of what it is, not because of what the user holds.
    """
    symbol = (ticker or "").strip().upper()
    if symbol:
        if symbol in sets["positions"]:
            return "POSITION"
        if symbol in sets["pool"]:
            return "TRADING_POOL"
        if symbol in sets["watchlist"]:
            return "WATCHLIST"
    return default_relevance_tier(event_type)


async def ingest_tickers(session: AsyncSession) -> list[str]:
    """Watchlist ∪ trading pool ∪ open positions, sorted and deduped.

    The universe the platform is allowed to hold data for (plan §4.2). An
    empty universe is a legitimate state (fresh install) — the macro/Fed
    providers still contribute market-wide events.
    """
    sets = await relevance_inputs(session)
    return sorted(sets["positions"] | sets["pool"] | sets["watchlist"])


# ---------------------------------------------------------------------------
# Ingest state (cadence + capability memory)
# ---------------------------------------------------------------------------


async def _ingest_state(session: AsyncSession, key: str) -> EventIngestStateRow:
    row = await session.get(EventIngestStateRow, key)
    if row is None:
        row = EventIngestStateRow(key=key, meta={})
        session.add(row)
    return row


def _cadence_due(row: EventIngestStateRow, now: datetime, provider_name: str) -> bool:
    """Has this provider's re-fetch cadence come due?

    Measured from ``last_ok_at`` (the last SUCCESS), never ``last_fetched_at``:
    a provider that has been failing must keep being retried, otherwise a
    transient 500 would silently mute a source for a day.
    """
    last_ok = _utc(row.last_ok_at)
    if last_ok is None:
        return True
    hours = PROVIDER_MIN_INTERVAL_HOURS.get(provider_name, DEFAULT_MIN_INTERVAL_HOURS)
    return now - last_ok >= timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Market calendar
# ---------------------------------------------------------------------------


async def _upsert_market_calendar(
    session: AsyncSession, days: Iterable[MarketDay], *, now: datetime
) -> int:
    """Upsert session rows by ``session_date`` (the PK). Returns rows written."""
    written = 0
    for day in days:
        row = await session.get(MarketCalendarRow, day.session_date)
        if row is None:
            row = MarketCalendarRow(session_date=day.session_date)
            session.add(row)
        row.exchange = day.exchange or "US"
        row.open_utc = day.open_utc
        row.close_utc = day.close_utc
        row.session_open_utc = day.session_open_utc
        row.session_close_utc = day.session_close_utc
        row.is_early_close = bool(day.is_early_close)
        row.source = day.source or "unknown"
        row.fetched_at = now
        written += 1
    return written


async def market_calendar_map(
    session: AsyncSession, start: date, end: date
) -> dict[date, MarketCalendarRow]:
    """Stored sessions in ``[start, end]``, keyed by ET calendar date."""
    rows = (
        (
            await session.execute(
                select(MarketCalendarRow).where(
                    MarketCalendarRow.session_date >= start,
                    MarketCalendarRow.session_date <= end,
                )
            )
        )
        .scalars()
        .all()
    )
    return {row.session_date: row for row in rows}


def _session_for(
    scheduled_at: datetime, calendar: dict[date, MarketCalendarRow]
) -> EventSession:
    """Classify against that ET day's real session when one is stored.

    This is the whole reason the market_calendar table exists: on a 13:00 ET
    half-day close, a 13:30 release is AFTER_MARKET, and the default
    09:30-16:00 session would call it DURING_MARKET.
    """
    day = scheduled_at.astimezone(NEW_YORK).date()
    row = calendar.get(day)
    if row is None:
        return classify_session(scheduled_at)
    return classify_session(scheduled_at, _utc(row.open_utc), _utc(row.close_utc))


# ---------------------------------------------------------------------------
# Upsert one candidate
# ---------------------------------------------------------------------------


async def _find_existing(
    session: AsyncSession, candidate: EventCandidate
) -> EventRow | None:
    """The stored row this candidate describes, or None.

    Two lookups, in order: the exact natural key (the fast, deterministic
    path), then U1's ``same_event`` drift window over same-type rows near the
    candidate's date — that second pass is what absorbs an ESTIMATED earnings
    date being replaced by a confirmed one three days later without leaving a
    duplicate card behind.
    """
    row = (
        await session.execute(
            select(EventRow).where(EventRow.event_key == candidate.event_key)
        )
    ).scalar_one_or_none()
    if row is not None:
        return row

    window = _DRIFT_WINDOWS.get(candidate.event_type)
    if window is None:
        return None
    query = select(EventRow).where(
        EventRow.event_type == candidate.event_type.value,
        EventRow.scheduled_at >= candidate.scheduled_at - window,
        EventRow.scheduled_at <= candidate.scheduled_at + window,
    )
    if candidate.event_type is EventType.EARNINGS:
        if not candidate.ticker:
            return None
        query = query.where(EventRow.ticker == candidate.ticker)
    near = (await session.execute(query)).scalars().all()
    for row in near:
        if same_event(row_to_event(row), candidate):
            return row
    return None


async def upsert_candidate(
    session: AsyncSession,
    candidate: EventCandidate,
    *,
    now: datetime,
    calendar: dict[date, MarketCalendarRow] | None = None,
) -> tuple[EventRow, str]:
    """Create or merge one candidate. Returns ``(row, change)``.

    ``change`` is ``"created"`` for a brand-new row, else U1's merge verdict.
    Sessions are re-classified against the stored market calendar *after* the
    merge so a half-day correction applies to whatever date won.
    """
    calendar = calendar or {}
    existing = await _find_existing(session, candidate)
    if existing is None:
        event = candidate.to_event(now=now)
        if event.session is EventSession.UNKNOWN:
            event = replace(event, session=_session_for(event.scheduled_at, calendar))
        row = _new_row(event)
        session.add(row)
        await session.flush()
        return row, "created"

    merged, change = merge(row_to_event(existing), candidate, now)
    resolved = _session_for(merged.scheduled_at, calendar)
    if calendar and resolved is not merged.session:
        merged = replace(merged, session=resolved)
    apply_event_to_row(existing, merged)
    return existing, change or "reverified"


# ---------------------------------------------------------------------------
# Importance (§13)
# ---------------------------------------------------------------------------


async def rescore_importance(
    session: AsyncSession, rows: Sequence[EventRow], sets: dict[str, set[str]]
) -> int:
    """Recompute importance for ``rows`` with the DB's relevance inputs.

    Rescored on every tick because relevance MOVES: promoting a ticker into
    the trading pool must raise its earnings card without waiting for the
    provider to re-emit the event.
    """
    changed = 0
    for row in rows:
        try:
            event_type = EventType(row.event_type)
        except ValueError:  # pragma: no cover — enum drift under a live table
            continue
        tier = relevance_tier(event_type, row.ticker, sets)
        score = score_importance(
            event_type, relevance_tier=tier, speaker=row.speaker
        ).score
        if row.importance != score:
            row.importance = score
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Previous-comparable linkage (§15) — persisted, never crossing types
# ---------------------------------------------------------------------------

#: Types whose §15 chain the tick maintains. EARNINGS is the one the Catalyst
#: page reads (a "Previous Event" tab needs the previous PRINT); the two FOMC
#: types are included because their chain is unambiguous — each decision has
#: exactly one predecessor. FED_SPEECH is deliberately EXCLUDED:
#: ``previous_comparable`` flags a same-speaker link low-confidence, and
#: persisting a low-confidence guess as a column would let it be read later
#: with the caveat stripped off. Macro types are excluded for the same reason
#: the price tab does not link them yet — the comparison there is
#: release-period arithmetic that Phase G owns.
LINKED_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.EARNINGS,
    EventType.FOMC_DECISION,
    EventType.FOMC_MINUTES,
)


async def link_previous(
    session: AsyncSession, rows: Sequence[EventRow]
) -> int:
    """Persist ``previous_event_id`` + ``comparison_reason`` for ``rows`` (§15).

    THE MATCHING RULES ARE NOT HERE. This function batches by (type, ticker),
    hands each batch to the pure :func:`link_previous_events`, and writes the
    answer back — so the definition of "comparable" lives in ONE place and a
    stored link can never disagree with the one the price tab computes on the
    fly from the same function.

    BATCHED PER (type, ticker), which is what makes the "never crosses types"
    guarantee structural rather than merely tested: an EARNINGS row is only
    ever offered EARNINGS candidates for the SAME ticker, so no ordering
    accident inside the matcher could link NVDA's print to AMD's, or an
    earnings print to an FOMC decision. (``previous_comparable`` also checks
    the type itself; two independent guarantees for a cross-link that would be
    silently wrong rather than loudly broken.)

    The pool is the FULL stored history of that (type, ticker), not the
    ingest window: an event at the edge of the ±window must still find the
    print before it, which may be a quarter earlier — otherwise the first
    card of every quarter would report "no previous comparable event" until a
    later tick happened to load both.

    WRITES ONLY WHAT CHANGED, and records nothing. Re-linking an unchanged
    chain on every tick would either write a no-op UPDATE or, worse, an audit
    row per event per tick — the "no audit spam" rule. Returns how many rows
    actually moved.
    """
    by_group: dict[tuple[str, str | None], list[EventRow]] = {}
    for row in rows:
        try:
            event_type = EventType(row.event_type)
        except ValueError:  # pragma: no cover — enum drift under a live table
            continue
        if event_type not in LINKED_EVENT_TYPES:
            continue
        # The ticker is part of the key for EARNINGS only; the FOMC types are
        # global series with no ticker, and keying them by their (NULL) ticker
        # would still put them all in one batch, which is what they want.
        by_group.setdefault((row.event_type, row.ticker), []).append(row)

    changed = 0
    for (event_type_value, ticker), group in by_group.items():
        pool_rows = (
            (
                await session.execute(
                    select(EventRow).where(
                        EventRow.event_type == event_type_value,
                        EventRow.ticker == ticker,
                    )
                )
            )
            .scalars()
            .all()
        )
        by_key = {r.event_key: r for r in pool_rows}
        links = link_previous_events([row_to_event(r) for r in pool_rows])
        subjects = {r.event_key for r in group}
        for event_key, previous_key, reason in links:
            row = by_key.get(event_key)
            if row is None or event_key not in subjects:
                # Only rows this tick touched are written; the rest of the
                # pool is present to be MATCHED AGAINST, not to be rewritten.
                continue
            previous_row = by_key.get(previous_key) if previous_key else None
            previous_id = previous_row.id if previous_row is not None else None
            if row.previous_event_id != previous_id or row.comparison_reason != reason:
                row.previous_event_id = previous_id
                row.comparison_reason = reason
                changed += 1
    return changed


# ---------------------------------------------------------------------------
# T-minus alerting (§11) — exactly once per (event_key, horizon)
# ---------------------------------------------------------------------------


async def _already_alerted(session: AsyncSession, event_id: int, horizon: int) -> bool:
    """Has EVENT_APPROACHING already been written for this (event, horizon)?

    Checked against the AUDIT TABLE rather than an in-process set, which is
    what makes the exactly-once guarantee survive a restart AND a second
    replica (ADR-007: no leader election, so correctness may not depend on
    single-process state). The horizon lives in ``details`` so a later
    T-1 alert is a distinct fact from today's T-7.
    """
    rows = (
        (
            await session.execute(
                select(AuditEvent.details).where(
                    AuditEvent.action == AuditAction.EVENT_APPROACHING.value,
                    AuditEvent.entity_type == ENTITY_TYPE,
                    AuditEvent.entity_id == str(event_id),
                )
            )
        )
        .scalars()
        .all()
    )
    return any((details or {}).get("horizon") == horizon for details in rows)


async def emit_approaching_alerts(
    session: AsyncSession, *, now: datetime, horizon_days: int
) -> int:
    """Write EVENT_APPROACHING for events entering the T-minus window.

    Only CONFIRMED and REVISED events qualify. An ESTIMATED date is a
    derivation, not a fact, and §11 forbids presenting it as one — alerting
    on it would do exactly that. A CANCELED event obviously does not
    approach.
    """
    horizon_end = now + timedelta(days=horizon_days)
    rows = (
        (
            await session.execute(
                select(EventRow).where(
                    EventRow.status.in_(
                        (EventStatus.CONFIRMED.value, EventStatus.REVISED.value)
                    ),
                    EventRow.scheduled_at > now,
                    EventRow.scheduled_at <= horizon_end,
                )
            )
        )
        .scalars()
        .all()
    )
    written = 0
    for row in rows:
        if await _already_alerted(session, row.id, horizon_days):
            continue
        scheduled = _utc(row.scheduled_at)
        days_to_event = (scheduled - now) / timedelta(days=1)
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.EVENT_APPROACHING,
            entity_type=ENTITY_TYPE,
            entity_id=str(row.id),
            details={
                "event_key": row.event_key,
                "horizon": horizon_days,
                "type": row.event_type,
                "ticker": row.ticker or "",
                "title": row.title,
                "scheduled_at": scheduled.isoformat(),
                "status": row.status,
                "days_to_event": round(days_to_event, 2),
            },
        )
        written += 1
    EVENT_ALERTS_TOTAL.inc(written)
    return written


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------


def _provider_name(provider) -> str:
    return getattr(provider, "name", provider.__class__.__name__)


def _probe_capabilities(provider) -> dict[str, bool | str]:
    """Capability report, never raising into the tick.

    A probe that itself fails is reported as the error string on every key —
    "availability unknown", which is a different fact from the ``False``
    that means "probed and proven absent" (audit §6).
    """
    try:
        return dict(provider.capabilities())
    except Exception as exc:  # noqa: BLE001 — a probe fault must not end the tick
        logger.warning(
            "event_calendar_capability_probe_failed",
            extra={"extra_fields": {"provider": _provider_name(provider), "error": str(exc)}},
        )
        return {"error": str(exc)}


async def run_calendar_ingest(
    session: AsyncSession,
    *,
    now: datetime,
    providers: list | None = None,
    tickers: list[str] | None = None,
    force: bool = False,
) -> dict:
    """ONE ingestion tick. Returns the honest per-provider report.

    ``providers`` and ``tickers`` are injection seams for tests and for the
    manual refresh; both default to what the configuration and the database
    say. ``force=True`` ignores every provider's re-fetch cadence (the
    refresh button) but changes nothing else — a forced tick that finds the
    same events still creates zero rows, because identity is the natural key,
    not the fetch.

    The caller owns the transaction: this function adds rows and audit
    records to ``session`` and never commits, so the ingest and its audit
    trail land atomically (ADR-003).
    """
    settings = get_settings()
    if now.tzinfo is None:
        raise ValueError("run_calendar_ingest requires a timezone-aware `now`")
    now = now.astimezone(timezone.utc)

    if providers is None:
        try:
            providers = configured_providers(settings)
        except ProviderNotConfigured:
            providers = []
    symbols = list(tickers) if tickers is not None else await ingest_tickers(session)

    report: dict[str, dict] = {}
    skipped: list[dict] = []
    created = 0
    updated = 0
    calendar_days = 0

    if not providers:
        # Honest absence, not an error: the platform still serves whatever is
        # already stored, and the API explains why nothing new arrived.
        logger.info(
            "event_calendar_skipped",
            extra={"extra_fields": {"reason": "NO_PROVIDERS_CONFIGURED"}},
        )
        skipped.append({"provider": "", "reason": "NO_PROVIDERS_CONFIGURED"})

    window_start = now - timedelta(days=LOOKBACK_DAYS)
    window_end = now + timedelta(days=LOOKAHEAD_DAYS)
    cal_start = (now - timedelta(days=CALENDAR_LOOKBACK_DAYS)).astimezone(NEW_YORK).date()
    cal_end = (now + timedelta(days=CALENDAR_LOOKAHEAD_DAYS)).astimezone(NEW_YORK).date()

    for provider in providers:
        name = _provider_name(provider)
        state = await _ingest_state(session, name)
        if not force and not _cadence_due(state, now, name):
            logger.info(
                "event_calendar_skipped",
                extra={"extra_fields": {"reason": "CADENCE_NOT_DUE", "provider": name}},
            )
            skipped.append({"provider": name, "reason": "CADENCE_NOT_DUE"})
            report[name] = {
                "skipped": "CADENCE_NOT_DUE",
                "created": 0,
                "updated": 0,
                "capabilities": dict(state.meta.get("capabilities") or {}),
                "last_ok_at": (_utc(state.last_ok_at) or now).isoformat()
                if state.last_ok_at
                else None,
            }
            continue

        state.last_fetched_at = now
        capabilities = await asyncio.to_thread(_probe_capabilities, provider)

        # --- events -------------------------------------------------------
        candidates: list[EventCandidate] = []
        error: str | None = None
        try:
            candidates = await asyncio.to_thread(
                provider.fetch_events,
                tickers=symbols,
                start=window_start,
                end=window_end,
            )
        except CapabilityNotAvailable as exc:
            error = f"CAPABILITY_NOT_AVAILABLE: {exc}"
        except MarketDataError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — per-adapter isolation (§8)
            error = str(exc)
        if error is not None:
            CALENDAR_PROVIDER_FAILURES_TOTAL.inc(provider=name)
            logger.warning(
                "event_calendar_provider_failed",
                extra={"extra_fields": {"provider": name, "error": error}},
                exc_info=True,
            )

        # --- market calendar ----------------------------------------------
        days: list[MarketDay] = []
        calendar_error: str | None = None
        try:
            days = await asyncio.to_thread(
                provider.fetch_market_calendar, cal_start, cal_end
            )
        except CapabilityNotAvailable:
            # Not a failure: SEC EDGAR and the Fed simply do not serve
            # exchange sessions, and say so with this exception by design.
            calendar_error = None
        except MarketDataError as exc:
            calendar_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            calendar_error = str(exc)
        if calendar_error is not None:
            CALENDAR_PROVIDER_FAILURES_TOTAL.inc(provider=name)
            logger.warning(
                "event_calendar_sessions_failed",
                extra={"extra_fields": {"provider": name, "error": calendar_error}},
            )
        if days:
            calendar_days += await _upsert_market_calendar(session, days, now=now)

        calendar = await market_calendar_map(session, cal_start, cal_end)

        provider_created = 0
        provider_updated = 0
        for candidate in candidates:
            row, change = await upsert_candidate(
                session, candidate, now=now, calendar=calendar
            )
            if change == "created":
                provider_created += 1
                await audit.record(
                    session,
                    actor_type=ActorType.SYSTEM,
                    action=AuditAction.EVENT_DISCOVERED,
                    entity_type=ENTITY_TYPE,
                    entity_id=str(row.id),
                    details={
                        "event_key": row.event_key,
                        "type": row.event_type,
                        "ticker": row.ticker or "",
                        "scheduled_at": _utc(row.scheduled_at).isoformat(),
                        "status": row.status,
                        "source_name": row.source_name,
                    },
                )
            elif change in MATERIAL_CHANGES:
                provider_updated += 1
                await audit.record(
                    session,
                    actor_type=ActorType.SYSTEM,
                    action=AuditAction.EVENT_UPDATED,
                    entity_type=ENTITY_TYPE,
                    entity_id=str(row.id),
                    details={
                        "event_key": row.event_key,
                        "change": change,
                        "type": row.event_type,
                        "ticker": row.ticker or "",
                        "scheduled_at": _utc(row.scheduled_at).isoformat(),
                        "status": row.status,
                        "source_name": row.source_name,
                    },
                )

        created += provider_created
        updated += provider_updated
        EVENTS_DISCOVERED_TOTAL.inc(provider_created)

        combined_error = error or calendar_error
        state.last_error = combined_error
        if combined_error is None:
            state.last_ok_at = now
        state.meta = {
            "capabilities": capabilities,
            "created": provider_created,
            "updated": provider_updated,
            "candidates": len(candidates),
            "sessions": len(days),
        }
        report[name] = {
            "created": provider_created,
            "updated": provider_updated,
            "candidates": len(candidates),
            "sessions": len(days),
            "capabilities": capabilities,
            "error": combined_error,
            "last_ok_at": now.isoformat() if combined_error is None else (
                _utc(state.last_ok_at).isoformat() if state.last_ok_at else None
            ),
        }
        if combined_error is not None:
            skipped.append({"provider": name, "reason": "PROVIDER_ERROR"})

    # --- importance + alerts ----------------------------------------------
    sets = await relevance_inputs(session)
    horizon_days = int(settings.event_horizon_alert_days)
    scored_rows = (
        (
            await session.execute(
                select(EventRow).where(
                    EventRow.scheduled_at >= now - timedelta(days=LOOKBACK_DAYS),
                    EventRow.scheduled_at <= now + timedelta(days=LOOKAHEAD_DAYS),
                )
            )
        )
        .scalars()
        .all()
    )
    rescored = await rescore_importance(session, scored_rows, sets)
    # §15 — the previous-comparable chain, persisted so the Catalyst page's
    # "Previous Event" tab reads a column instead of recomputing the match on
    # every request. Runs after the upserts and the rescoring because a
    # re-keyed or newly confirmed row must be linked in its FINAL state.
    linked = await link_previous(session, scored_rows)
    alerts = await emit_approaching_alerts(session, now=now, horizon_days=horizon_days)

    result = {
        "as_of": now.isoformat(),
        "providers": report,
        "created": created,
        "updated": updated,
        "rescored": rescored,
        "linked": linked,
        "calendar_days": calendar_days,
        "alerts": alerts,
        "skipped": skipped,
    }
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.CALENDAR_INGESTED,
        entity_type="event_calendar",
        entity_id=str(new_york_today()),
        details={
            "providers": {
                key: {
                    k: v
                    for k, v in value.items()
                    if k in ("created", "updated", "candidates", "sessions", "error", "skipped")
                }
                for key, value in report.items()
            },
            "capabilities": {key: value.get("capabilities", {}) for key, value in report.items()},
            "created": created,
            "updated": updated,
            "alerts": alerts,
            "skipped": skipped,
        },
    )
    CALENDAR_INGESTS_TOTAL.inc()
    logger.info(
        "event_calendar_ingested",
        extra={
            "extra_fields": {
                "created": created,
                "updated": updated,
                "alerts": alerts,
                "providers": list(report),
            }
        },
    )
    return result


async def run_scheduled_ingest(force: bool = False) -> dict:
    """ONE tick of :func:`event_calendar_loop`, owning its own session.

    Split out from the loop (the monitor.py / risk_snapshot.py pattern) so
    tests drive a single tick deterministically — httpx ASGITransport does
    not run the lifespan, so the background task never starts under the
    suite.
    """
    async with SessionLocal() as session:
        try:
            result = await run_calendar_ingest(
                session, now=datetime.now(timezone.utc), force=force
            )
        except Exception:
            await session.rollback()
            raise
        await session.commit()
        return result


async def event_calendar_loop() -> None:
    """Sleep -> ingest the event calendar, forever (event spec §8).

    Started by the gateway lifespan when
    ``settings.event_calendar_interval_seconds`` > 0 (0 disables it), exactly
    like the position monitor and the risk-snapshot writer.

    RESILIENCE: every exception from a tick is logged with its traceback and
    swallowed; the next tick runs normally. ``asyncio.CancelledError`` is
    always re-raised so graceful shutdown is never swallowed. Individual
    provider failures never reach here — they are isolated inside the tick
    (§8) — so an exception at this level means the ingest itself broke.
    """
    interval = get_settings().event_calendar_interval_seconds
    logger.info(
        "event_calendar_loop_started",
        extra={"extra_fields": {"interval_seconds": interval}},
    )
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await run_scheduled_ingest()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("event_calendar_tick_failed")
    except asyncio.CancelledError:
        logger.info("event_calendar_loop_stopped")
        raise


__all__ = [
    "ENTITY_TYPE",
    "LOOKAHEAD_DAYS",
    "LOOKBACK_DAYS",
    "MATERIAL_CHANGES",
    "NEW_YORK",
    "PROVIDER_MIN_INTERVAL_HOURS",
    "apply_event_to_row",
    "emit_approaching_alerts",
    "event_calendar_loop",
    "ingest_tickers",
    "link_previous",
    "market_calendar_map",
    "new_york_today",
    "relevance_inputs",
    "relevance_tier",
    "rescore_importance",
    "row_to_event",
    "run_calendar_ingest",
    "run_scheduled_ingest",
    "upsert_candidate",
]
