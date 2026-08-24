"""Event registry API — the Catalyst page's backend (event spec §5-§13, §15;
audit §5.1, §11.1 Phase B).

Named ``events``, not ``catalyst``, deliberately (audit §11.1 point 4):
``GET /api/watchlist/{ticker}/catalyst`` already ships (routers/analysis.py,
pinned by tests/test_catalyst.py) and is a read over the latest LLM
recommendation. This surface is the typed event REGISTRY and supersedes that
one later; the collision is resolved here rather than discovered in a later
phase.

Endpoints:

- ``GET  /api/events``               horizon + relevance + type filtered feed
- ``GET  /api/events/calendar``      stored exchange sessions
- ``GET  /api/events/{event_id}``    one event + its previous comparable (§15)
- ``GET  /api/events/{id}/price-context``  pre-event positioning + previous
  event reactions, as of an instant (§17, §31, §32; Phase E1)
- ``GET  /api/events/{id}/fundamentals``   point-in-time §28 snapshot,
  previous-vs-current change and §30 valuation, as of an instant (Phase E2)
- ``GET  /api/events/{id}/replay``          the §20 replay bundle: release,
  minute-bar immediate reaction, daily subsequent reaction (Phase C)
- ``GET  /api/events/{id}/history``         the §60 LAST N EARNINGS table
- ``POST /api/events/refresh``       force one ingestion tick (USER actor)
- ``POST /api/events/{id}/replay/backfill``   USER: fetch ONE event's minute
  window (§20; reads never fetch minutes, only this does)
- ``POST /api/events/{id}/history/backfill``  USER: fetch the last N events'
  minute windows (bounded, default 4, max 12)
- ``GET  /api/events/{id}/news``             the §21-§27 news evidence for
  the event's window, as of an instant (Phase D; never fetches)
- ``POST /api/events/{id}/news/backfill``    USER: fetch the event's news
  window from every configured provider (Phase D)
- ``GET  /api/events/{id}/risk``             the §63 event-risk snapshot +
  the §66 options panel, as of an instant (Phase K) — SHADOW, never a gate
- ``POST /api/events/{id}/confirm``  USER-confirmed date (source rank 0, §78)
- ``POST /api/events/{id}/cancel``   USER cancellation

HONEST ABSENCE, NOT 503. Unlike the market-data endpoints, this router
answers 200 with ``events: []`` when nothing is configured and nothing is
stored: the whole point of §11's ESTIMATED status is that the platform tells
you what it does and does not know about a date, and a 503 would hide the
capability report that explains WHY the calendar is empty. Nothing is
fabricated to fill the silence — an empty list plus a capability block IS the
honest answer.

Every date the payload carries is stamped twice: ``scheduled_at_utc`` (the
instant) and ``scheduled_at_local`` (the event's own zone, with offset).
``is_estimated`` travels with every row so no consumer can render a derived
date as a fact (§7, §11).
"""
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.event_calendar import configured_provider_names
from libs.trading_core.events import (
    EASTERN,
    IMPORTANCE_MODEL_VERSION,
    MIN_MEANINGFUL_N,
    lifecycle,
    merge,
    previous_comparable,
    relevance_rank,
    score_importance,
    to_local,
)
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from .. import audit
from ..db import (
    EventIngestStateRow,
    EventRow,
    MarketCalendarRow,
    Position,
    get_session,
)
from ..deps import require_llm_provider
from ..event_calendar import (
    ENTITY_TYPE,
    apply_event_to_row,
    relevance_inputs,
    relevance_tier,
    row_to_event,
    run_calendar_ingest,
)
from ..event_news import (
    build_event_news,
    ensure_event_news_window,
    news_provider_names,
)
from ..event_prediction_markets import (
    prediction_markets_section,
    refresh_event_prediction_markets,
)
from ..event_price import build_price_context
from ..event_research import run_event_research, web_research_section
from ..event_timeline import attach_card_summaries, build_event_timeline
from ..event_replay import (
    DEFAULT_HISTORY_BACKFILL,
    MAX_HISTORY_BACKFILL,
    backfill_history_windows,
    build_event_history,
    build_event_replay_payload,
    ensure_event_window_bars,
)
from ..fundamentals import build_fundamentals_context, fundamentals_provider_name
from .. import event_fed, event_macro, event_options, event_risk, event_study

router = APIRouter(prefix="/api/events", tags=["events"])

#: The display timezone every catalyst surface renders in (§10). Stored
#: instants are UTC; this is the zone the UI shows and the horizon buckets
#: are computed in.
DISPLAY_TIMEZONE = "America/New_York"

#: Named horizons (§11). "custom" requires explicit start/end.
HORIZONS: dict[str, int | None] = {"today": 0, "7d": 7, "30d": 30, "custom": None}

#: The user is the actor of record for a confirm/cancel; the platform is
#: single-user today (same constant as trading_pool.CURRENT_USER).
CURRENT_USER = "current-user"

#: Ceiling on the §86 study's ``min_n`` knob. It only raises the bar at which a
#: correlation is flagged NOT_MEANINGFUL, so a large value is harmless — but an
#: unbounded integer in a query string is a 422 waiting to be an OverflowError,
#: and no reader needs a stricter floor than "every event this install has".
MAX_MIN_N = 1000


# ---------------------------------------------------------------------------
# Horizon math (§11) — bucketed on the NEW YORK day, not on UTC
# ---------------------------------------------------------------------------


def resolve_horizon(
    horizon: str, start: str | None, end: str | None, now: datetime
) -> tuple[datetime, datetime, str]:
    """``(start_utc, end_utc, label)`` for a named or custom horizon.

    Bucketed on the America/New_York CALENDAR DAY, not on a UTC offset from
    ``now``: "today" must mean the NY trading day the user is looking at, and
    "next 7 days" must end at the close of the NY day 7 days out. Doing this
    in UTC would silently shift the window by 4-5 hours and move an AMC
    earnings release into the wrong bucket — and by a *different* amount in
    July than in December, because the ET offset changes with DST. Deriving
    both edges from the local date and converting back is what makes the
    window DST-correct by construction.

    Raises ``ValueError`` for an unknown horizon or a malformed/incoherent
    custom range; the endpoint turns that into a 422.
    """
    if horizon not in HORIZONS:
        raise ValueError(
            f"unknown horizon {horizon!r}; expected one of {sorted(HORIZONS)}"
        )
    if horizon == "custom":
        if not start or not end:
            raise ValueError("horizon=custom requires both start and end (ISO dates)")
        try:
            start_dt = _parse_boundary(start, end_of_day=False)
            end_dt = _parse_boundary(end, end_of_day=True)
        except ValueError as exc:
            raise ValueError(f"invalid custom range: {exc}") from exc
        if end_dt <= start_dt:
            raise ValueError("custom range end must be after start")
        return start_dt, end_dt, f"{start} → {end}"

    days = HORIZONS[horizon]
    today_local = now.astimezone(EASTERN).date()
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=EASTERN)
    end_local = datetime.combine(
        today_local + timedelta(days=days), datetime.max.time(), tzinfo=EASTERN
    )
    label = {"today": "Today", "7d": "Next 7 days", "30d": "Next 30 days"}[horizon]
    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
        label,
    )


def _parse_boundary(value: str, *, end_of_day: bool) -> datetime:
    """Parse an ISO date or datetime into a UTC boundary instant.

    A bare date is interpreted in NEW YORK (the display zone) rather than
    UTC: a user typing "2026-09-16" means that NY calendar day, and reading
    it as UTC would clip the AMC releases off its end.
    """
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not an ISO date or datetime") from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc)
    if len(text) == 10:  # bare YYYY-MM-DD
        clock = datetime.max.time() if end_of_day else datetime.min.time()
        return datetime.combine(parsed.date(), clock, tzinfo=EASTERN).astimezone(
            timezone.utc
        )
    return parsed.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def _csv(value: str | None) -> list[str]:
    return [piece.strip() for piece in (value or "").split(",") if piece.strip()]


# ---------------------------------------------------------------------------
# EventOut
# ---------------------------------------------------------------------------


def event_out(
    row: EventRow,
    *,
    now: datetime,
    sets: dict[str, set[str]],
    exposure: dict[str, dict] | None = None,
) -> dict:
    """One event as the API renders it (§6, §11, §12, §13).

    Both timestamps travel: ``scheduled_at_utc`` is the instant the platform
    compares on, ``scheduled_at_local`` the wall clock the event asserts
    ("CPI at 08:30 ET"), which is the one a user reasons about and the one
    that stays fixed across DST while the UTC instant moves.

    ``importance_components`` is the whole §13 contract: the score is never
    a number without its arithmetic. ``is_estimated`` is duplicated out of
    ``status`` on purpose — a consumer that forgets to check the enum still
    cannot render a derived date as a fact (§7, §11).
    """
    event_type = EventType(row.event_type)
    tier = relevance_tier(event_type, row.ticker, sets)
    scored = score_importance(event_type, relevance_tier=tier, speaker=row.speaker)
    scheduled = row.scheduled_at
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=timezone.utc)
    else:
        scheduled = scheduled.astimezone(timezone.utc)
    verified = row.last_verified_at
    if verified is not None and verified.tzinfo is None:
        verified = verified.replace(tzinfo=timezone.utc)

    out = {
        "event_id": row.id,
        "event_key": row.event_key,
        "event_type": row.event_type,
        "title": row.title,
        "ticker": row.ticker,
        "company_id": row.company_id,
        "scheduled_at_utc": scheduled.isoformat(),
        "scheduled_at_local": to_local(scheduled, row.event_timezone).isoformat(),
        "event_timezone": row.event_timezone,
        "session": row.session,
        "status": row.status,
        "is_estimated": row.status == EventStatus.ESTIMATED.value,
        "source": row.source,
        "source_name": row.source_name,
        "source_url": row.source_url,
        "source_event_id": row.source_event_id,
        "last_verified_at": verified.isoformat() if verified else None,
        "previous_event_id": row.previous_event_id,
        "comparison_reason": row.comparison_reason,
        "days_to_event": (scheduled - now) / timedelta(days=1),
        "lifecycle": lifecycle(scheduled, now).value,
        "relevance_tier": tier,
        # The STORED score (what ingestion last persisted) is null until a
        # tick has scored the row; the live recomputation below is what the
        # UI renders. Both are exposed so a stale persisted score is visible
        # rather than silently papered over.
        "importance": scored.score,
        "importance_stored": row.importance,
        "importance_components": dict(scored.components),
        "importance_raw_total": scored.raw_total,
        "importance_was_clamped": scored.was_clamped,
        "importance_model_version": IMPORTANCE_MODEL_VERSION,
        "series_id": row.series_id,
        "agency": row.agency,
        "release_period": row.release_period,
        "fiscal_quarter": row.fiscal_quarter,
        "fiscal_year": row.fiscal_year,
        "speaker": row.speaker,
        "topic": row.topic,
        "revision_history": list(row.revision_history or []),
        "exposure": None,
    }
    if exposure and row.ticker:
        out["exposure"] = exposure.get(row.ticker.upper())
    return out


async def _exposure_map(session: AsyncSession) -> dict[str, dict]:
    """Open-position exposure per ticker (§12/§54 "exposure if any").

    ``position_market_value`` is deliberately the COST basis
    (quantity × avg_price × multiplier), not a live mark: this router reads
    stored rows only and never touches a market-data provider, so quoting a
    "market value" from a live price here would both break that separation
    and put a synthetic number in an unconfigured install's response. The key
    name is the contract's; the value is honestly labelled by ``basis``.
    """
    rows = (
        (
            await session.execute(select(Position).where(Position.status == "OPEN"))
        )
        .scalars()
        .all()
    )
    out: dict[str, dict] = {}
    for row in rows:
        key = row.ticker.upper()
        bucket = out.setdefault(
            key,
            {"position_qty": 0, "position_market_value": 0.0, "basis": "COST"},
        )
        bucket["position_qty"] += row.quantity
        bucket["position_market_value"] += (
            row.quantity * row.avg_price * (row.multiplier or 1)
        )
    return out


# ---------------------------------------------------------------------------
# Capability + freshness blocks
# ---------------------------------------------------------------------------


async def capability_report(session: AsyncSession) -> tuple[dict, dict]:
    """``(capabilities, freshness)`` — the last KNOWN provider state.

    Read from ``event_ingest_state`` (written by each ingestion tick), never
    by probing here: an API request must not fire six HTTP probes, and a
    remembered verdict with its timestamp is more honest than a fresh one
    that hides how long ago it was true. Providers that are configured but
    have never run appear with ``last_ok_at: null`` and an explicit
    ``NEVER_RUN`` note rather than being omitted — an absent provider and an
    untried one are different facts.
    """
    settings = get_settings()
    try:
        configured = configured_provider_names(settings)
    except Exception:  # noqa: BLE001 — a registry fault must not 500 the feed
        configured = []
    rows = (
        (await session.execute(select(EventIngestStateRow))).scalars().all()
    )
    by_key = {row.key: row for row in rows}

    capabilities: dict[str, dict] = {}
    per_provider: dict[str, dict] = {}
    for name in configured:
        row = by_key.get(name)
        meta = dict(row.meta or {}) if row is not None else {}
        capabilities[name] = dict(meta.get("capabilities") or {})
        last_ok = row.last_ok_at if row is not None else None
        last_fetch = row.last_fetched_at if row is not None else None
        per_provider[name] = {
            "configured": True,
            "last_ok_at": _iso(last_ok),
            "last_fetched_at": _iso(last_fetch),
            "last_error": row.last_error if row is not None else None,
            "note": None if last_fetch is not None else "NEVER_RUN",
        }
    last_ingest = max(
        (row.last_ok_at for row in rows if row.last_ok_at is not None),
        default=None,
    )
    freshness = {
        "last_ingest_at": _iso(last_ingest),
        "per_provider": per_provider,
        "configured_providers": configured,
    }
    return capabilities, freshness


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# GET /api/events
# ---------------------------------------------------------------------------


@router.get("")
async def list_events(
    session: AsyncSession = Depends(get_session),
    horizon: str = Query(default="7d"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    types: str | None = Query(default=None, description="CSV of EventType names"),
    tickers: str | None = Query(default=None, description="CSV of symbols"),
    include_estimated: bool = Query(default=True),
    include_canceled: bool = Query(default=False),
    relevance: str | None = Query(default=None, description="CSV of relevance tiers"),
    summaries: bool = Query(
        default=False,
        description=(
            "Attach the §54 card summary (analysis freshness, implied move "
            "with its basis, historical move median with its n) to each "
            "event. OFF by default so the feed's payload is unchanged for "
            "every existing caller."
        ),
    ),
) -> dict:
    """The catalyst feed (§11 horizon, §12 relevance ordering).

    Sorted by (relevance tier rank, scheduled_at) so a position's earnings
    outranks an unrelated macro print at the same hour — the §12 ladder,
    applied as the sort key rather than left to the client.

    Returns 200 with an empty list when nothing matches or nothing is
    configured; ``capabilities``/``freshness`` explain why. Unknown horizon
    or a malformed custom range is a 422 (a client bug, not an empty result).
    """
    now = datetime.now(timezone.utc)
    try:
        start_utc, end_utc, label = resolve_horizon(horizon, start, end, now)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    query = select(EventRow).where(
        EventRow.scheduled_at >= start_utc, EventRow.scheduled_at <= end_utc
    )
    wanted_types = [t.upper() for t in _csv(types)]
    if wanted_types:
        unknown = [t for t in wanted_types if t not in EventType.__members__]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"unknown event type(s): {', '.join(sorted(unknown))}",
            )
        query = query.where(EventRow.event_type.in_(wanted_types))
    wanted_tickers = [t.upper() for t in _csv(tickers)]
    if wanted_tickers:
        query = query.where(EventRow.ticker.in_(wanted_tickers))
    if not include_estimated:
        query = query.where(EventRow.status != EventStatus.ESTIMATED.value)
    if not include_canceled:
        query = query.where(EventRow.status != EventStatus.CANCELED.value)

    rows = (await session.execute(query)).scalars().all()
    sets = await relevance_inputs(session)
    exposure = await _exposure_map(session)

    events = [event_out(row, now=now, sets=sets, exposure=exposure) for row in rows]
    wanted_relevance = {r.upper() for r in _csv(relevance)}
    if wanted_relevance:
        events = [e for e in events if e["relevance_tier"] in wanted_relevance]
    events.sort(
        key=lambda e: (relevance_rank(e["relevance_tier"]), e["scheduled_at_utc"])
    )

    # §54 card summaries, OPT-IN. Off, not one extra query runs and the
    # payload is byte-identical to what every existing caller already parses;
    # on, four bulk queries add a ``summary`` block to each row. The flag
    # rather than always-on because the feed is also polled by the alert path
    # and by the header count, neither of which draws a card.
    if summaries:
        await attach_card_summaries(session, events)

    counts_by_type: dict[str, int] = {}
    counts_by_relevance: dict[str, int] = {}
    for event in events:
        counts_by_type[event["event_type"]] = counts_by_type.get(event["event_type"], 0) + 1
        tier = event["relevance_tier"]
        counts_by_relevance[tier] = counts_by_relevance.get(tier, 0) + 1

    capabilities, freshness = await capability_report(session)
    return {
        "as_of": now.isoformat(),
        "horizon": {
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "label": label,
            "key": horizon,
        },
        "display_timezone": DISPLAY_TIMEZONE,
        "events": events,
        "counts": {
            "total": len(events),
            "by_type": counts_by_type,
            "by_relevance": counts_by_relevance,
            "estimated": sum(1 for e in events if e["is_estimated"]),
            "confirmed": sum(
                1 for e in events if e["status"] == EventStatus.CONFIRMED.value
            ),
        },
        "capabilities": capabilities,
        "freshness": freshness,
    }


# ---------------------------------------------------------------------------
# GET /api/events/calendar
# ---------------------------------------------------------------------------


@router.get("/calendar")
async def get_market_calendar(
    session: AsyncSession = Depends(get_session),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> dict:
    """Stored exchange sessions (§10; the half-day table §6 classification needs).

    Defaults to a ±30 NY-day window around today. Returns only rows that were
    actually ingested — an absent date is an absent row, never a synthesised
    "probably 09:30-16:00" session, because a guessed session is exactly what
    would misclassify a half-day release.
    """
    today = datetime.now(EASTERN).date()
    try:
        start_date = date.fromisoformat(start) if start else today - timedelta(days=30)
        end_date = date.fromisoformat(end) if end else today + timedelta(days=30)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"start/end must be ISO dates: {exc}"
        ) from exc
    if end_date < start_date:
        raise HTTPException(status_code=422, detail="end must not precede start")

    rows = (
        (
            await session.execute(
                select(MarketCalendarRow)
                .where(
                    MarketCalendarRow.session_date >= start_date,
                    MarketCalendarRow.session_date <= end_date,
                )
                .order_by(MarketCalendarRow.session_date)
            )
        )
        .scalars()
        .all()
    )
    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "display_timezone": DISPLAY_TIMEZONE,
        "sessions": [
            {
                "session_date": row.session_date.isoformat(),
                "exchange": row.exchange,
                "open_utc": _iso(row.open_utc),
                "close_utc": _iso(row.close_utc),
                "session_open_utc": _iso(row.session_open_utc),
                "session_close_utc": _iso(row.session_close_utc),
                "is_early_close": bool(row.is_early_close),
                "source": row.source,
                "fetched_at": _iso(row.fetched_at),
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# POST /api/events/refresh
# ---------------------------------------------------------------------------


@router.post("/refresh")
async def refresh_events(session: AsyncSession = Depends(get_session)) -> dict:
    """Force ONE ingestion tick, ignoring every provider's re-fetch cadence.

    The user asked, so the cadence gate is bypassed — but nothing else
    changes: identity is still the natural key, so a refresh that finds the
    same events creates ZERO rows. That idempotence is what makes the button
    safe to press repeatedly.
    """
    now = datetime.now(timezone.utc)
    result = await run_calendar_ingest(session, now=now, force=True)
    await session.commit()
    return result


# ---------------------------------------------------------------------------
# GET /api/events/{event_id}
# ---------------------------------------------------------------------------


@router.get("/{event_id}")
async def get_event(
    event_id: int, session: AsyncSession = Depends(get_session)
) -> dict:
    """One event plus its previous comparable event and WHY that one (§15).

    The comparison reason is returned alongside the previous event because
    §15 requires the platform to justify the comparison rather than assert
    it — "prior quarterly earnings" and "prior speech by the same speaker
    (low confidence)" carry very different weight.
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    now = datetime.now(timezone.utc)
    sets = await relevance_inputs(session)
    exposure = await _exposure_map(session)
    payload = event_out(row, now=now, sets=sets, exposure=exposure)

    # Candidate pool: same type, strictly earlier. previous_comparable
    # narrows further (ticker, series, speaker) and is the single authority
    # on what "comparable" means — this query only avoids loading the table.
    earlier = (
        (
            await session.execute(
                select(EventRow).where(
                    EventRow.event_type == row.event_type,
                    EventRow.scheduled_at < row.scheduled_at,
                )
            )
        )
        .scalars()
        .all()
    )
    previous, reason = previous_comparable(
        row_to_event(row), [row_to_event(other) for other in earlier]
    )
    previous_payload = None
    if previous is not None and previous.event_id is not None:
        previous_row = await session.get(EventRow, previous.event_id)
        if previous_row is not None:
            previous_payload = event_out(
                previous_row, now=now, sets=sets, exposure=exposure
            )
    payload["previous_event"] = previous_payload
    payload["comparison_reason"] = reason or row.comparison_reason
    return payload


# ---------------------------------------------------------------------------
# POST /api/events/{event_id}/confirm  and  /cancel
# ---------------------------------------------------------------------------


class ConfirmEventRequest(BaseModel):
    """A user asserting the real date from an IR page or SEC filing (§7).

    The USER source outranks every automated one (rank 0, §78), so this both
    flips ESTIMATED -> CONFIRMED and pins the date against any later cadence
    estimate — which is exactly the audit's step 4 in the earnings fallback
    chain ("user-confirmed, which overwrites the estimate").
    """

    scheduled_at: str = Field(description="ISO-8601 instant; tz-aware or ET-local")
    session: str | None = Field(default=None, description="EventSession name")
    source_url: str | None = Field(default=None)


@router.post("/{event_id}/confirm")
async def confirm_event(
    event_id: int,
    body: ConfirmEventRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER-confirm an event's date (§7, §78).

    Applied through the SAME merge rules every provider goes through, with
    ``EventSourceKind.USER`` — the authority is data (rank 0), not a bypass.
    That matters: the merge is what appends the replaced value to
    ``revision_history`` and what re-keys the row when the confirmation moves
    it to another ET day.
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    try:
        scheduled = _parse_instant(body.scheduled_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    event_session = EventSession.UNKNOWN
    if body.session:
        try:
            event_session = EventSession(body.session.strip().upper())
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown session {body.session!r}; expected one of "
                    f"{[s.value for s in EventSession]}"
                ),
            ) from exc

    now = datetime.now(timezone.utc)
    existing = row_to_event(row)
    candidate = _user_candidate(
        existing,
        scheduled_at=scheduled,
        status=EventStatus.CONFIRMED,
        session=event_session,
        source_url=body.source_url,
        now=now,
    )
    merged, change = merge(existing, candidate, now)
    apply_event_to_row(row, merged)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.EVENT_UPDATED,
        entity_type=ENTITY_TYPE,
        entity_id=row.event_key,
        details={
            "change": change or "reverified",
            "type": row.event_type,
            "ticker": row.ticker or "",
            "scheduled_at": _iso(row.scheduled_at),
            "status": row.status,
            "source_name": row.source_name,
            "user_confirmed": True,
            "source_url": body.source_url,
        },
    )
    await session.commit()
    sets = await relevance_inputs(session)
    payload = event_out(row, now=now, sets=sets, exposure=await _exposure_map(session))
    payload["change"] = change or "reverified"
    return payload


@router.post("/{event_id}/cancel")
async def cancel_event(
    event_id: int, session: AsyncSession = Depends(get_session)
) -> dict:
    """USER-cancel an event (§7: CANCELED only ever arrives explicitly).

    Never inferred from an event vanishing from a provider's feed — a feed
    going quiet is a fact about the feed, not about the event.
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    now = datetime.now(timezone.utc)
    existing = row_to_event(row)
    candidate = _user_candidate(
        existing,
        scheduled_at=existing.scheduled_at,
        status=EventStatus.CANCELED,
        session=existing.session,
        source_url=None,
        now=now,
    )
    merged, change = merge(existing, candidate, now)
    apply_event_to_row(row, merged)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.EVENT_UPDATED,
        entity_type=ENTITY_TYPE,
        entity_id=row.event_key,
        details={
            "change": change or "canceled",
            "type": row.event_type,
            "ticker": row.ticker or "",
            "scheduled_at": _iso(row.scheduled_at),
            "status": row.status,
            "source_name": row.source_name,
            "user_canceled": True,
        },
    )
    await session.commit()
    sets = await relevance_inputs(session)
    payload = event_out(row, now=now, sets=sets, exposure=await _exposure_map(session))
    payload["change"] = change or "canceled"
    return payload


def _parse_instant(value: str) -> datetime:
    """Parse a user-supplied instant; a bare local time is read as ET.

    Naive input is NOT assumed UTC (§10: guessing the zone is how a BMO
    release becomes an AMC release). The display zone is Eastern, so that is
    what an offsetless string from this UI means.
    """
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"scheduled_at {value!r} is not an ISO-8601 instant") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    return parsed.astimezone(timezone.utc)


def _user_candidate(
    existing,
    *,
    scheduled_at: datetime,
    status: EventStatus,
    session: EventSession,
    source_url: str | None,
    now: datetime,
):
    """Build the USER-authored candidate the merge rules consume.

    Imported lazily from the domain package so this router keeps the same
    single definition of EventCandidate as the providers and the seam.
    """
    from libs.trading_core.events import EventCandidate
    from libs.trading_core.events.taxonomy import event_key as build_key

    try:
        key = build_key(
            existing.event_type,
            scheduled_at=scheduled_at,
            ticker=existing.ticker,
            release_period=existing.release_period,
            speaker=existing.speaker,
            title=existing.title,
            subtype=existing.topic,
        )
    except ValueError:
        # A type whose key needs a field this row does not carry keeps its
        # existing key rather than inventing one.
        key = existing.event_key
    return EventCandidate(
        event_key=key,
        event_type=existing.event_type,
        title=existing.title,
        scheduled_at=scheduled_at,
        status=status,
        source=EventSourceKind.USER,
        source_name="user",
        ticker=existing.ticker,
        company_id=existing.company_id,
        event_timezone=existing.event_timezone,
        session=session,
        source_url=source_url,
        source_event_id=existing.source_event_id,
        last_verified_at=now,
        series_id=existing.series_id,
        agency=existing.agency,
        release_period=existing.release_period,
        fiscal_quarter=existing.fiscal_quarter,
        fiscal_year=existing.fiscal_year,
        speaker=existing.speaker,
        topic=existing.topic,
    )


# ---------------------------------------------------------------------------
# GET /api/events/{event_id}/price-context   (Phase E1, §17, §31, §32)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/price-context")
async def get_event_price_context(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Pre-event positioning and previous-event reactions for one event (§17,
    §31, §32; audit §11.2 Phase E1).

    ``as_of`` is what makes this endpoint answerable historically (§14, §85,
    §96): the seam gates the BARS at that instant, so asking for a past
    earnings date returns the analysis that would have existed then rather
    than today's hindsight. It defaults to now only at THIS boundary — the
    seam itself requires it (audit §7.2 rule 2). A future ``as_of`` is a 422:
    a request for prices that do not exist yet is a mistake worth reporting,
    not a request to silently clamp to now.

    200 IN EVERY DEGRADED CASE, exactly like the feed above. An unconfigured
    market-data provider yields ``bars: {"available": false, "reason": ...}``
    with the event's registry facts intact; a macro/Fed event (no ticker)
    yields ``{"available": false, "reason": "no_ticker"}`` — Phase G gives
    those their §39 multi-asset proxies. Only a missing event is a 404, and
    only a malformed/future ``as_of`` is a 422.

    Computed on demand: the whole payload is a few hundred stored bars and a
    handful of registry rows, so it needs no table of its own (the §72 cache
    with its persist-dedupe guard belongs to the Phase F package, not here).
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    now = datetime.now(timezone.utc)
    if as_of is None:
        as_of_dt = now
    else:
        try:
            as_of_dt = _parse_instant(as_of)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if as_of_dt > now:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"as_of {as_of_dt.isoformat()} is in the future "
                    f"(now {now.isoformat()}); no prices exist for it"
                ),
            )

    settings = get_settings()
    return await build_price_context(
        session,
        row,
        as_of=as_of_dt,
        provider_name=settings.market_data_provider,
    )


# ---------------------------------------------------------------------------
# GET /api/events/{event_id}/fundamentals   (Phase E2, §16, §28, §29, §30, §33)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/fundamentals")
async def get_event_fundamentals(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Point-in-time fundamentals for one event (§16, §28, §29, §30, §33, §35;
    audit §7, §11.3 Phase E2).

    ``as_of`` is what makes this answerable historically (§85, §96): the seam
    gates the STATEMENTS on ``acceptance_datetime <= as_of`` — never on the
    fiscal period end, which is weeks earlier and would let a quarter inform
    an analysis run before it was ever filed — and gates the price used for
    the multiples on the §14 bar rule. Asking for a past earnings date returns
    the fundamentals picture that existed then, previous-vs-current delta
    included. It defaults to now only at THIS boundary; the seam itself
    requires it (audit §7.2 rule 2). A future ``as_of`` is a 422: no filing
    exists for it, and silently clamping to now would answer a different
    question than the one asked.

    200 IN EVERY DEGRADED CASE, exactly like the price-context route above. A
    provider whose plan excludes fundamentals (403) yields ``statements:
    {"available": false, "reason": ...}`` with the registry facts intact; a
    macro/Fed event yields ``{"available": false, "reason": "no_ticker"}`` —
    a CPI release has no balance sheet. CONSENSUS is ALWAYS
    ``available: false`` with its reason (§33/§98, audit §7.3): Benzinga
    estimates are 403 at any instant, so an EPS surprise is not merely
    un-backtestable, it is unavailable, and the payload says so rather than
    omitting the block. Only a missing event is a 404, and only a malformed or
    future ``as_of`` is a 422.

    Computed on demand over stored filings and stored bars; the §72 cached
    analysis package with its persist-dedupe guard belongs to Phase F.
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    now = datetime.now(timezone.utc)
    if as_of is None:
        as_of_dt = now
    else:
        try:
            as_of_dt = _parse_instant(as_of)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if as_of_dt > now:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"as_of {as_of_dt.isoformat()} is in the future "
                    f"(now {now.isoformat()}); no filings exist for it"
                ),
            )

    settings = get_settings()
    return await build_fundamentals_context(
        session,
        row,
        as_of=as_of_dt,
        # Statements come from Massive when its key is configured (the only
        # provider serving financials); prices stay on the market-data
        # provider — see fundamentals.fundamentals_provider_name.
        provider_name=fundamentals_provider_name(settings),
        price_provider_name=settings.market_data_provider,
    )


# ---------------------------------------------------------------------------
# Event replay (Phase C, §17, §20, §60)
# ---------------------------------------------------------------------------


async def _event_or_404(session: AsyncSession, event_id: int) -> EventRow:
    """The row, or the router's standard 404. Only a MISSING event is a 404.

    Every other degradation on these four routes — no provider, no ticker, no
    stored minutes, an event that has not happened yet — is a 200 with a
    reason, exactly like the price-context and fundamentals routes above.
    """
    row = await session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")
    return row


def _resolve_as_of(as_of: str | None, now: datetime, *, subject: str) -> datetime:
    """``as_of`` as an instant, defaulting to ``now``; 422 on future/malformed.

    The same rule the price-context and fundamentals routes apply, stated once
    here because four more routes now need it: a future ``as_of`` is a 422
    rather than a silent clamp to now, because clamping answers a DIFFERENT
    question than the one asked and the caller has no way to notice.
    ``subject`` names what does not exist yet ("no prices", "no minute bars")
    so the error says which fact the request outran.
    """
    if as_of is None:
        return now
    try:
        parsed = _parse_instant(as_of)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if parsed > now:
        raise HTTPException(
            status_code=422,
            detail=(
                f"as_of {parsed.isoformat()} is in the future "
                f"(now {now.isoformat()}); {subject}"
            ),
        )
    return parsed


@router.get("/{event_id}/replay")
async def get_event_replay(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §20 event replay for one event, as of an instant (Phase C).

    Four blocks in the §20 order: what was knowable BEFORE the release
    (references to the price-context and fundamentals endpoints at the same
    ``as_of``), the RELEASE itself, the IMMEDIATE reaction from minute bars,
    and the SUBSEQUENT daily reaction with its abnormal-vs-SPY overlay.

    THIS ROUTE NEVER FETCHES MINUTE BARS. It reads what is stored and gates it
    at ``as_of``; an event with no stored window answers 200 with
    ``immediate_reaction: {"available": false, "reason": ...}`` and a complete
    daily reaction. Fetching a ~1,000-bar window on a GET would mean opening a
    page issues provider calls nobody asked for — ``POST
    /api/events/{id}/replay/backfill`` is the USER action that does it.

    A FUTURE EVENT IS 200 + ``available: false``, not a 404: the Catalyst page
    opens on UPCOMING earnings, and "this has not happened yet as of
    <instant>" is the honest answer, with the event's registry facts attached.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no reaction exists for it")
    settings = get_settings()
    return await build_event_replay_payload(
        session,
        row,
        as_of=as_of_dt,
        provider_name=settings.market_data_provider,
    )


@router.get("/{event_id}/history")
async def get_event_history(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    last: int | None = Query(
        default=None,
        ge=1,
        le=MAX_HISTORY_BACKFILL,
        description="Trim to the newest N past events (the §60 4/8/12 toggle)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §60 "LAST N EARNINGS" table for this event's ticker (Phase C).

    One row per past CONFIRMED/REVISED print knowable at ``as_of``, with the
    daily gap/1D/5D/abnormal columns, ``intraday_30m`` from stored minute bars
    where they exist, and the three columns that are unavailable at EVERY
    instant rather than merely absent here: ``eps_surprise`` and
    ``rev_surprise`` (no consensus vendor at any tier, §33/§98) and
    ``implied_move`` (Phase I, §36). Those carry ``available: false`` with
    their reason rather than being omitted — a table that simply lacked the
    columns would read as "we did not think surprise mattered".

    ``last`` trims BEFORE the summary is computed, so the §19/§64 distribution
    describes exactly the rows on screen. Like ``/replay``, this route reads
    stored minutes only; ``POST /api/events/{id}/history/backfill`` fills
    them.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no history exists for it")
    settings = get_settings()
    return await build_event_history(
        session,
        row,
        as_of=as_of_dt,
        provider_name=settings.market_data_provider,
        last=last,
    )


@router.post("/{event_id}/replay/backfill")
async def backfill_event_replay_bars(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch and store ONE event's minute-bar window (Phase C).

    A POST because it WRITES — this is the only path that inserts
    ``stock_bars_1m``, and it does so together with a SYSTEM ``DATA_BACKFILL``
    audit row in the same transaction (rule 12, ADR-003).

    IDEMPOTENT AND SELF-LIMITING. An event window is a closed interval in the
    past, so a window already stored is returned as ``fetched: false`` with
    ``reason: "window already stored"`` rather than refetched, and a window
    the provider could not serve is throttled per (ticker, event) so a
    repeated press cannot hammer a vendor for minutes that do not exist.

    200 IN EVERY DEGRADED CASE. An unconfigured provider, a plan without
    minute data (403) and a macro event with no ticker all answer 200 with
    ``fetched: false`` and the reason — a button press must report why nothing
    arrived, not 5xx. Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    settings = get_settings()
    return await ensure_event_window_bars(
        session,
        row.ticker or "",
        row,
        settings.market_data_provider,
        now=datetime.now(timezone.utc),
    )


@router.post("/{event_id}/history/backfill")
async def backfill_event_history_bars(
    event_id: int,
    last: int = Query(
        default=DEFAULT_HISTORY_BACKFILL,
        ge=1,
        le=MAX_HISTORY_BACKFILL,
        description=(
            "How many of the newest past events to fetch minute windows for "
            f"(default {DEFAULT_HISTORY_BACKFILL}, max {MAX_HISTORY_BACKFILL})"
        ),
    ),
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant selecting the past-event pool; now (UTC) by default",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch the last N past events' minute windows (Phase C).

    BOUNDED TWICE, on purpose. FastAPI's ``le=`` rejects an out-of-range
    ``last`` at the boundary with a 422, and the seam clamps again — the
    ceiling is a property of the operation (twelve paginated provider fetches
    is already a lot), not of one caller's query string, so it holds for the
    internal caller too.

    Each window is attempted independently and every outcome is reported in
    ``results``: one event whose bars the provider will not serve does not
    cost the other three theirs, which is the same per-item isolation the
    calendar ingest applies to providers (§8).
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no past events exist for it")
    settings = get_settings()
    return await backfill_history_windows(
        session,
        row,
        as_of=as_of_dt,
        provider_name=settings.market_data_provider,
        last=last,
        now=now,
    )


# ---------------------------------------------------------------------------
# News evidence (Phase D, §21-§27, §59, §81, §96)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/news")
async def get_event_news(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §21-§27 news evidence for one event's window, as of an instant.

    The window is the INTER-EVENT period — from one day before the previous
    comparable event to ``as_of`` — which is what makes the §26 counts mean
    anything: "eleven developments" is a statement about what has happened
    since the last print, not about a fixed trailing month that would double-
    count news for two events three weeks apart.

    THIS ROUTE NEVER FETCHES (§27). It reads stored articles and gates them at
    ``as_of``; an event whose window has nothing stored answers 200 with
    ``available: false`` and an ``unavailable`` entry naming the backfill as
    the remedy. That is stricter than ``/fundamentals``, which tops its mirror
    up on read, and deliberately so: a news window is dozens of paginated
    requests across two vendors, so opening the Catalyst page must not issue
    them. ``POST /api/events/{id}/news/backfill`` is the USER action that does.

    ``as_of`` is what makes this answerable historically (§96): an article
    published after it is excluded before any count, cluster, novelty
    measurement or score is computed, and the §22 decay is measured from it.
    It defaults to now only at THIS boundary; the seam itself requires it
    (audit §7.2 rule 2). A future ``as_of`` is a 422 — no article exists for
    it, and silently clamping would answer a different question than the one
    asked.

    200 IN EVERY DEGRADED CASE, like every other route here. A macro/Fed event
    yields ``{"available": false, "reason": "no_ticker"}`` — a CPI release has
    no issuer whose coverage this would be. Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no news exists for it")
    return await build_event_news(session, row, as_of=as_of_dt)


@router.post("/{event_id}/news/backfill")
async def backfill_event_news(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch this event's news window from every provider (§21).

    A POST because it WRITES — this is Phase D's only path into
    ``news_articles``, and it writes together with a SYSTEM ``NEWS_INGESTED``
    audit row in the same transaction (rule 12, ADR-003).

    BOTH VENDORS, EACH ISOLATED. Alpaca and Massive syndicate different wires,
    so a window built from one alone is a partial view of the tape; both are
    asked when both are configured and their results merge on ``source_id``
    (the column's UNIQUE key), so a story both carry is stored once. One
    vendor 403ing does not cost the other its articles — it becomes a named
    entry in ``providers[]``.

    THROTTLED PER TICKER, NOT PER EVENT. Two events on one symbol share almost
    all of their articles, so a per-event key would fetch the same hundred
    rows twice. Unlike the minute-bar backfill there is no "already stored"
    short circuit: a news window is open at its right edge and gains articles
    all day, so refusing to refetch would freeze the tape at whatever the
    first press caught.

    200 IN EVERY DEGRADED CASE. An unconfigured provider, a plan without news
    (403) and a macro event with no ticker all answer 200 with
    ``fetched: false`` and the reason — a button press must report why nothing
    arrived, not 5xx. Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    settings = get_settings()
    return await ensure_event_news_window(
        session,
        row,
        provider_names=news_provider_names(settings),
        now=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# External web research + prediction markets (Catalyst research upgrade;
# plan §5, Phases 1/12/21)
#
# THE READ/WRITE SPLIT IS THE COST BOUNDARY. The two GETs read stored rows and
# are free and poll-safe; the two POSTs are the only paths that spend a search
# quota or reach an external market API. That split is what makes Phase 21's
# "Refresh Sources" (these POSTs) mean something different from opening a tab
# — a React Query poll on the Catalyst page can never bill the operator.
#
# RESEARCH ONLY. Nothing these routes write is read by instrument selection,
# scoring, sizing, the gate chain, or execution; that isolation is enforced
# structurally by tests/test_research_safety_adversarial.py, not by convention.
# ---------------------------------------------------------------------------


@router.get("/{event_id}/research")
async def get_event_research(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Stored web-research evidence for one event, as of an instant.

    NEVER FETCHES (audit §7.2 rule 1). An event with no stored run answers 200
    with ``available: false`` and ``reason: "NEVER_RUN"`` — an honest runtime
    state, distinct from the provider being unconfigured (the providers
    endpoint owns that) and from a run that found nothing
    (``NO_EVIDENCE_ACCEPTED``). ``POST /research/backfill`` is the USER action
    that spends.

    ``as_of`` makes it answerable historically: a document published after it
    is withheld and counted, so a replay sees the evidence that was knowable
    then rather than everything a later refresh found.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no research exists for it")
    return await web_research_section(session, row, as_of=as_of_dt)


@router.post("/{event_id}/research/backfill")
async def backfill_event_research(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: run this event's bounded external web research (§21).

    A POST because it WRITES and because it SPENDS: this is the only path in
    the platform that issues paid search queries. Every bound is a named
    constant (``MAX_QUERIES_PER_EVENT``, ``MAX_RESULTS_PER_QUERY``,
    ``MAX_UNIQUE_DOCUMENTS``, ``MAX_ACCEPTED_EVIDENCE``) and the run row
    records what was actually spent, so cost is auditable rather than
    estimated.

    200 IN EVERY DEGRADED CASE — unconfigured, throttled, no queries planned,
    or every query failing. A button press must report why nothing arrived.
    Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    settings = get_settings()
    return await run_event_research(
        session,
        row,
        provider_name=settings.web_search_provider,
        now=datetime.now(timezone.utc),
    )


@router.get("/{event_id}/prediction-markets")
async def get_event_prediction_markets(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Stored prediction-market matches and pricing, as of an instant.

    Every price here is MARKET-IMPLIED PROBABILITY — what the contract costs,
    never a claim about the outcome's actual likelihood — and each entry
    carries the depth facts (spread, liquidity known or not) so a thin
    market's 70c is not read with a deep market's authority.

    NEVER FETCHES. The distinct honest states are ``NEVER_RUN``,
    ``NO_RELEVANT_PREDICTION_MARKET`` (matching ran and accepted nothing — a
    valid and common outcome) and ``MARKET_METADATA_UNAVAILABLE``.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(
        as_of, now, subject="no prediction-market data exists for it"
    )
    return await prediction_markets_section(session, row, as_of=as_of_dt)


@router.post("/{event_id}/prediction-markets/backfill")
async def backfill_event_prediction_markets(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: discover, match and observe this event's markets.

    READ-ONLY against the venue: public discovery, pricing and history only.
    No wallet, no signing, no order — the provider protocol has no method
    that could place one.

    Accepting NO market is a success, not a failure: it stores the decisions
    and reports ``NO_RELEVANT_PREDICTION_MARKET`` rather than forcing a
    loosely-related contract into the evidence bundle.
    """
    row = await _event_or_404(session, event_id)
    settings = get_settings()
    return await refresh_event_prediction_markets(
        session,
        row,
        provider_name=settings.prediction_markets_provider,
        now=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Event analysis packages (Phase F, §16, §46-§52, §69-§71, §99)
#
# FOUR ROUTES, ONE WRITER. Only the POST calls the model; the three GETs read
# what is stored and say so when nothing is. That is the same read/write split
# ``/news`` and ``/news/backfill`` already draw, and for a stronger reason
# here: an LLM call costs money and takes seconds, so opening the Catalyst
# page must never trigger one. The evidence route goes further and never
# touches the model at all — the bundle IS an answer, and a reader who wants
# only the filed numbers should not have to pay for prose to see them.
# ---------------------------------------------------------------------------


@router.get("/{event_id}/evidence")
async def get_event_evidence(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §46 evidence bundle for one event — DATA and QUANT tiers only.

    NO LLM IS INVOLVED AND NONE IS REQUIRED. This is the document the model
    would be handed: previous event and its reaction, point-in-time
    fundamentals, price and positioning, the news window, the §35
    expectations-gap INPUTS (inputs, never a labelled regime — labelling is
    the model's job and this route does not do the model's job). It answers
    200 with no LLM configured at all, because every number in it was measured
    or filed, not generated.

    NO FETCHES (§27; audit §7.2 rule 1). It composes STORED filings, STORED
    bars and STORED articles. A section with nothing behind it reports
    ``available: false`` with a reason and, where a user action would fix it,
    names the backfill — it does not quietly omit the section, which would
    read as "there is nothing to say" rather than "we have not looked".

    ``as_of`` is what makes it answerable historically (§85, §96): filings are
    gated on their acceptance instant, bars on the §14 rule, articles on
    ``published_at``. A future ``as_of`` is a 422 rather than a silent clamp.
    Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no evidence exists for it")

    from ..event_analysis import build_bundle, _event_status_badge

    bundle, digest = await build_bundle(
        session, row, as_of=as_of_dt, settings=get_settings()
    )
    return {
        "event_id": row.id,
        "event_key": row.event_key,
        "ticker": row.ticker,
        "as_of": as_of_dt.isoformat(),
        "bundle": bundle,
        "bundle_digest": digest,
        "event_status_badge": _event_status_badge(row),
        "tiers": {
            "bundle": "DATA/QUANT — measured or filed facts and arithmetic over them",
            "prior_analyses": "LLM_PRIOR — past opinions, not evidence (§70)",
        },
    }


@router.get("/{event_id}/analysis")
async def get_event_analysis(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The latest STORED analysis package for one event — never calls the LLM.

    404 ``ANALYSIS_NOT_FOUND`` when none exists, and that is deliberately a
    404 rather than a 200 with ``available: false``: unlike a missing filing
    or an empty news window, "nobody has run an analysis yet" is not a
    degradation of this platform's data — it is a resource that does not exist
    and that a POST creates. The UI turns the 404 into the "Generate analysis"
    call to action rather than an error.

    There is no ``as_of`` parameter, on purpose. A stored package IS its own
    as-of (it carries the instant its bundle was assembled as of), and
    accepting one here would imply this route could answer for an arbitrary
    instant — which it cannot without calling the model, which it must not.
    Use ``GET .../analyses`` to see every instant on record.

    THE PRIMARY PAYLOAD IS THE LAST GOOD ANALYSIS, NOT THE LAST ROW. A
    provider timeout writes a FAILED row that is newer than the research it
    did not replace; serving that row as "the analysis" tells a reader the
    platform has nothing when it in fact has a complete note on disk. So an
    ``OK`` row wins when one exists, and the newer failure is not hidden
    either — it rides along as ``last_attempt`` (status, error, instant,
    provider, model) so the UI can show "the newest attempt failed" ABOVE the
    analysis rather than passing off an older answer as current. With no
    ``OK`` row on file the newest row is returned exactly as before, FAILED or
    INVALID status and all, because then the failure IS the whole story.
    """
    row = await _event_or_404(session, event_id)

    from ..event_analysis import (
        STATUS_OK,
        latest_analysis,
        latest_ok_analysis,
        serialize_analysis,
    )

    stored = await latest_analysis(session, event_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ANALYSIS_NOT_FOUND",
                "message": (
                    f"no analysis has been generated for event {event_id} "
                    f"({row.event_key}) — POST /api/events/{event_id}/analysis "
                    "to produce one"
                ),
            },
        )
    if stored.status == STATUS_OK:
        return serialize_analysis(stored, row, cached=True)

    good = await latest_ok_analysis(session, event_id)
    if good is None:
        return serialize_analysis(stored, row, cached=True)
    return serialize_analysis(good, row, cached=True, last_attempt=stored)


@router.post("/{event_id}/analysis")
async def generate_event_analysis(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    force: bool = Query(
        default=False,
        description="Re-run even when a stored analysis covers this exact evidence",
    ),
    _: None = Depends(require_llm_provider),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: produce the §48 analysis package for one event.

    A POST because it WRITES — the row lands together with its
    ``EVENT_ANALYSIS_GENERATED`` audit record in the same transaction (rule
    12, ADR-003) — and because it SPENDS: this is the only route in the
    catalyst surface that calls a model.

    503 ``LLM_NOT_CONFIGURED`` when no usable provider exists, via the shared
    dependency every LLM route uses. That is the one case here that is NOT a
    200: an install with no model configured cannot produce an analysis at
    all, and reporting a fabricated-looking empty package would be worse than
    saying so (§44 rule 18).

    EVERY OTHER FAILURE IS A 200 WITH A STATUS. A provider that 403s, times
    out or refuses returns ``status: "FAILED"`` with the honest error and the
    evidence bundle intact — the reader still gets the filed numbers. A model
    that quoted a figure the bundle does not contain returns ``status:
    "INVALID"`` with the text AND the violations list, because hiding a
    misquote destroys the evidence that it happened (§99).

    CACHED BY EVIDENCE, NOT BY CLOCK (§72). If an OK package already exists
    for this event with the same bundle digest, prompt version and model, it
    is returned with ``cached: true`` and no call is made. ``force=true``
    re-runs and INSERTS a new row; the previous answer is never deleted (it
    is demoted to SUPERSEDED), so a regression between model versions stays
    diagnosable and §69 event memory keeps its series.

    AND THAT IS WHY THE DEFAULT ``as_of`` IS TRUNCATED TO THE MINUTE. The
    bundle carries its own ``as_of``, so the instant is part of what the
    digest covers — which is correct, because a bundle answering a different
    moment IS a different document. But with a microsecond-resolution default
    no two presses could ever share a digest, the cache above would miss every
    single time, and the UI's ordinary "generate" button would spend a model
    call on every press while reporting a duplicate answer as fresh. Truncating
    only the DEFAULT keeps the honest as-of semantics (an explicitly passed
    instant is used exactly as given, to the microsecond) while making the
    common path cacheable. A minute is the right grain because nothing in the
    bundle — filings gated on acceptance, daily bars, an article's publication
    — moves faster than that.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no evidence exists for it")
    if as_of is None:
        as_of_dt = as_of_dt.replace(second=0, microsecond=0)

    from ..event_analysis import get_or_create_analysis

    return await get_or_create_analysis(
        session,
        row,
        as_of=as_of_dt,
        settings=get_settings(),
        force=force,
    )


@router.get("/{event_id}/analyses")
async def list_event_analyses(
    event_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Every analysis ever written for one event, newest first (§69).

    SUMMARIES, NOT PACKAGES — id, instant, status, regime, confidence and the
    executive paragraph. Shipping N full bundles to draw a list of dates would
    be megabytes of JSON per render, and the full package is one
    ``GET .../analysis`` away.

    FAILED and INVALID rows are listed, not filtered. The history of what the
    platform tried and what came back is part of what makes the analysis
    auditable (§99); a list that showed only the successes would let a model
    that fails four times in five look perfectly reliable.
    """
    row = await _event_or_404(session, event_id)

    from ..event_analysis import list_analyses

    items = await list_analyses(session, event_id)
    return {
        "event_id": row.id,
        "event_key": row.event_key,
        "ticker": row.ticker,
        "count": len(items),
        "analyses": items,
    }


# ---------------------------------------------------------------------------
# Options / implied move (Phase I, §18, §36, §37, §66)
#
# THREE ROUTES, THE SAME READ/WRITE SPLIT the replay and news surfaces draw,
# and here it is at its sharpest: one event's straddle costs a dated contract
# listing plus two paginated bar fetches, and the history table shows up to
# eight prior prints. A GET that backfilled them all would issue seventeen
# provider calls because somebody opened a tab. So the GET reads STORED bars
# and STORED metrics, and the two POSTs are the USER actions that fetch.
#
# The ONE provider call a GET makes is the LIVE chain for an UPCOMING event —
# a single snapshot for a symbol the platform already tracks, which cannot be
# stored as history (a price taken before the print is not a reconstruction of
# what the market charged) and which is precisely the question an upcoming
# print raises. Any failure there is a NO_DATA block with a reason, never a
# 5xx and never a number.
#
# TWO PROVIDERS, TWO CAPABILITIES. Historical bars and dated contracts come
# from ``option_history_provider_name`` (Massive whenever its key is
# configured — the only vendor with an as-of contract reference), while the
# LIVE chain keeps using ``market_data_provider``, because a current snapshot
# with greeks is what Alpaca actually sells.
# ---------------------------------------------------------------------------


@router.get("/{event_id}/options")
async def get_event_options(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §18/§36 implied-move context for one event, as of an instant.

    ``current`` is this event's own implied move, and its BASIS depends on
    where the event sits relative to ``as_of``: an UPCOMING print is priced
    off the LIVE chain (``LIVE_CHAIN_SNAPSHOT``), a past one off the stored
    reconstruction from daily option closes
    (``HISTORICAL_DAILY_CLOSE_APPROXIMATION``). The two are never blended and
    the label travels with every number (§37).

    ``history`` renders each previous comparable print's stored straddle —
    implied vs actual vs the §66 ratio — and ``stats`` gives the distribution
    of both. ``comparison`` is the line the tab exists for: what the market is
    charging now against what this stock has actually done.

    ``disclaimer`` carries the §37 wording verbatim, so a consumer that
    renders the number cannot render it without the caveat.

    THIS ROUTE DOES NOT BACKFILL OPTION BARS. An event with no stored metrics
    answers 200 with ``current: null`` (or the live block) and
    ``coverage.reason`` naming ``POST /api/events/{id}/options/backfill`` as
    the remedy. That is the same rule ``/news`` applies and for a stronger
    reason: an option backfill is a contract probe plus two bar fetches per
    event.

    200 IN EVERY DEGRADED CASE. A macro event with no ticker, an unconfigured
    provider, a vendor without dated option data and a symbol with no listed
    contracts each get their own reason. Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no option market exists for it")
    settings = get_settings()
    return await event_options.build_event_options_payload(
        session,
        row,
        as_of=as_of_dt,
        provider_name=settings.market_data_provider,
    )


@router.post("/{event_id}/options/backfill")
async def backfill_event_options(
    event_id: int,
    force: bool = Query(
        default=False,
        description=(
            "Re-fetch even when this event already has an OK metrics row. "
            "A NO_DATA row is always retried without it."
        ),
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch and store ONE event's ATM straddle (§18, §36).

    A POST because it WRITES — this is the only path that inserts
    ``option_daily_bars`` or ``event_option_metrics``, and it does so with a
    SYSTEM ``DATA_BACKFILL`` audit row in the same transaction (rule 12,
    ADR-003).

    POINT-IN-TIME BY CONSTRUCTION. The contracts are listed with ``as_of`` set
    to the event's PRE-event session, so a strike created in reaction to the
    print — which today's contract universe would happily return — cannot
    enter the straddle.

    NEVER A FABRICATED PREMIUM. If either leg has no bar on an anchor session
    the stored row is ``status: NO_DATA`` with the reason; a missing put
    treated as zero would halve the implied move and the mistake would look
    like a cheap option rather than like absent data.

    200 IN EVERY DEGRADED CASE. Alpaca's honest ``CapabilityNotAvailable``, an
    unconfigured provider, a macro event with no ticker and an event older
    than the stored equity bars all answer 200 with ``fetched: false`` and the
    reason. Only a missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    settings = get_settings()
    return await event_options.backfill_event_options(
        session,
        row,
        provider_name=event_options.option_history_provider_name(settings),
        now=datetime.now(timezone.utc),
        force=force,
    )


@router.post("/{event_id}/options/history/backfill")
async def backfill_event_options_history(
    event_id: int,
    last: int = Query(
        default=event_options.DEFAULT_HISTORY_BACKFILL,
        ge=1,
        le=event_options.MAX_HISTORY_BACKFILL,
        description=(
            "How many of the newest previous comparable events to price "
            f"(default {event_options.DEFAULT_HISTORY_BACKFILL}, max "
            f"{event_options.MAX_HISTORY_BACKFILL})"
        ),
    ),
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant selecting the past-event pool; now (UTC) by default",
    ),
    force: bool = Query(
        default=False,
        description=(
            "Re-fetch events that already have an OK metrics row. Without it "
            "the re-run costs provider calls only for the events still "
            "missing — NO_DATA rows are always retried."
        ),
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: price the last N previous events' straddles (§66).

    This is what fills the implied-vs-actual table. One press walks the
    previous comparable EARNINGS prints for this ticker — the same pool the
    price tab uses, gated at ``as_of`` — and backfills each one's straddle.

    BOUNDED TWICE, on purpose. FastAPI's ``le=`` rejects an out-of-range
    ``last`` at the boundary with a 422, and the seam clamps again: the
    ceiling is a property of the operation (twelve contract probes plus
    twenty-four paginated bar fetches is already a lot), not of one caller's
    query string, so it holds for the internal caller too.

    Each event is attempted INDEPENDENTLY and every outcome appears in
    ``events`` — ``{event_id, event_key, status, reason, stored_bars}`` per
    print, tallied in ``counts`` — so one event the vendor will not serve does
    not cost the other three theirs, the same per-item isolation the calendar
    ingest applies across providers (§8).

    PACED AND REPEATABLE. The walk waits between events rather than firing
    ~32 requests in a burst (which Massive answered with HTTP 429 until the
    run gave up), and every failure is STORED as a NO_DATA row naming its
    reason — so pressing the button again retries exactly the events that are
    still missing. ``force=true`` re-fetches the ones that already succeeded.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no previous events exist for it")
    settings = get_settings()
    return await event_options.backfill_options_history(
        session,
        row,
        as_of=as_of_dt,
        provider_name=event_options.option_history_provider_name(settings),
        last=last,
        now=now,
        force=force,
    )


# ---------------------------------------------------------------------------
# GET /api/events/{id}/timeline (Phase J, §57)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/timeline")
async def get_event_timeline(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Everything that happened between the last comparable event and ``as_of``.

    The §57 "Since Last Event" rail: one chronological list merging four
    STORED sources for this issuer — material news developments (the §23
    clusters, categorised and scored by the same pure layer the news tab
    renders), financial statements that became public, other registry events
    for the ticker, and this event's own stored analyses. The two ends of the
    rail are the previous comparable event and this one, so the answer reads
    as "since the last print, before the next".

    THIS ROUTE NEVER FETCHES (§27; audit §7.2 rule 1), and neither does the
    seam behind it — ``event_timeline`` holds no provider handle at all. A
    window with nothing stored is an empty ``items`` list with its counts at
    zero, not a lazy backfill: ``POST /api/events/{id}/news/backfill`` is the
    USER action that fills the news half, and the timeline shows what is
    there.

    ``as_of`` is the window's right edge AND a hard bound on every item (§96):
    news is gated on ``published_at``, filings on ``acceptance_datetime`` (the
    instant a filing became public, never its period end), events and analyses
    on their own instants, and the merged list is filtered once more so a kind
    added later cannot leak a future row. A future ``as_of`` is a 422 rather
    than a silent clamp, exactly as on the other read routes.

    A macro or Fed event answers 200 with ``available: false`` and ``reason:
    "no_ticker"`` — a CPI release has no issuer whose news, filings or sibling
    events these would be — with its own analyses still on the rail. Only a
    missing event is a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no timeline exists for it")
    return await build_event_timeline(session, row, as_of=as_of_dt)


# ---------------------------------------------------------------------------
# GET/POST /api/events/{id}/macro (Phase G, §8, §38-§41)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/macro")
async def get_event_macro(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §38 macro packet for one release, as of an instant.

    Four blocks: the PREVIOUS release visible at ``as_of`` (its reference
    period, its release instant and basis, and the actual per role), the
    CURRENT release this event is, the RECENT TREND per series, and the §39
    cross-asset reaction to the previous release — SPY/QQQ/TLT/IEF/SHY/GLD/
    USO/UUP in percent alongside the 2Y and 10Y in basis points, which are
    kept in separate objects because they are not the same kind of number.
    §40's related-evidence window (other macro prints and Fed events between
    the last release and now) rides along as the deterministic factual set;
    picking the themes out of it is the model's job, not a keyword filter's.

    THIS ROUTE NEVER FETCHES (§27; audit §7.2 rule 1), and the seam behind it
    holds no provider handle at all. That matters more here than on any other
    read: BLS's unregistered API allows roughly twenty-five requests a DAY, so
    a read that lazily topped up eight series would exhaust the budget on one
    page load and then serve errors to the backfill that could have fixed it.
    A series nobody has fetched is ``coverage.actuals.available: false``
    naming the POST below, not a lazy fan-out.

    NO CONSENSUS, IN EVERY BRANCH (§33). This platform subscribes to no macro
    forecast source, so ``consensus`` and ``surprise`` are the fixed
    unavailable strings and the payload carries the disclaimer at top level.
    There is no code path behind this route that computes a surprise.

    A NON-MACRO event answers 200 with ``available: false`` and a reason — the
    row exists, it simply has no statistical release behind it. Only a missing
    event is a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no macro release exists for it")
    return await event_macro.build_macro_payload(session, row, as_of=as_of_dt)


@router.post("/{event_id}/macro/backfill")
async def backfill_event_macro(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant the fetch is anchored to; now (UTC) by default",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch this release's series, the yield curve and the bars.

    The one path in Phase G that spends network requests. Three independent
    fetches — the event type's BLS series (one request each, three years, the
    ceiling of the unregistered v1 API), the Treasury par yield curve for this
    year and last, and daily bars for the eight §39 reference symbols from the
    configured equity provider — and every outcome, success or failure, is a
    named row in the response. A CPI packet with no yield curve is still most
    of the §38 answer, so one dead source never costs the other two.

    PACED, NOT RETRIED. BLS allows roughly 25 requests per day unregistered
    and this asks for at most four; a 429 is reported as a reason and the
    operator presses again tomorrow, because retrying would spend the very
    budget the retry is trying to recover.

    IDEMPOTENT. Observations upsert on (series_id, period) — an agency
    revising July's CPI is restating one fact, not publishing a second — and
    curves on the session date; bars are inserted only for dates not already
    stored and are never rewritten. Pressing twice costs provider calls and
    changes nothing else.

    Writes a SYSTEM-of-record ``DATA_BACKFILL`` audit row carrying
    ``kind: "event_macro"`` and the counts, in the same transaction as the
    data (rule 12, ADR-003).
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no macro release exists for it")
    settings = get_settings()
    return await event_macro.backfill_macro(
        session, row, settings=settings, as_of=as_of_dt
    )


# ---------------------------------------------------------------------------
# GET/POST /api/events/{id}/fed (Phase H, §9, §42-§45)
# ---------------------------------------------------------------------------


@router.get("/{event_id}/fed")
async def get_event_fed(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §42-§45 Fed packet for one FOMC row or Fed speech, as of an instant.

    Six blocks: the PREVIOUS decision's statement stored VERBATIM with its vote
    and target range, the §44 SENTENCE-LEVEL DIFF against the statement before
    it (ADDED / REMOVED / CHANGED / UNCHANGED, computed with stdlib difflib so
    the same two documents always diff the same way), the eight POLICY
    DIMENSIONS reported SIDE BY SIDE, the previous meeting's MINUTES, the
    SPEECHES given since, and the §45 pair of REACTION WINDOWS.

    NO SINGLE HAWKISH/DOVISH SCORE, BY DESIGN (§43). There is no key in this
    payload that collapses the eight dimensions into one number, and there is
    no code path behind this route that could compute one. A statement can
    tighten its inflation language while softening its guidance in the same
    paragraph; one label would have to erase one of those, and which one it
    erased would be invisible. The disclaimer saying so travels in the packet.

    TWO WINDOWS, NEVER BLENDED (§45). When minute bars are stored the reaction
    is reported as 14:00-14:30 ET (the statement) and 14:30-15:30 ET (the
    Chair's press conference) as SEPARATE objects — the market reversing
    between the two is the most informative thing an FOMC replay can show.
    When only daily bars exist the payload says ``basis: "daily"`` and
    ``separated: false`` rather than quietly reporting the blend as the
    statement's move.

    MARKET PRICING IS UNAVAILABLE AND SAYS SO. This platform subscribes to no
    fed funds futures feed, so ``market_pricing.status`` is the fixed
    UNAVAILABLE string; the 2Y yield change rides along only when a daily
    reaction computed one, explicitly LABELLED as a proxy rather than as
    implied odds.

    THIS ROUTE NEVER FETCHES (§27; audit §7.2 rule 1) and the seam behind it
    holds no HTTP client. federalreserve.gov rate-limits by User-Agent, so a
    read that lazily fetched four documents per page load would get this
    platform's contact address throttled — and the throttle would land on the
    backfill that could have repaired it. A meeting nobody has backfilled is
    ``coverage.previous_statement: false`` naming the POST below.

    A NON-FED event answers 200 with ``available: false`` and a reason — the
    row exists, it simply has no Committee behind it. Only a missing event is
    a 404.
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no Fed documents exist for it")
    return await event_fed.build_fed_payload(session, row, as_of=as_of_dt)


@router.post("/{event_id}/fed/backfill")
async def backfill_event_fed(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant the fetch is anchored to; now (UTC) by default",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """USER action: fetch this meeting's Fed documents and its reaction bars.

    The one path in Phase H that spends network requests. Five fetch groups —
    the press_monetary RSS feed (ONCE, because it carries the Fed's own
    publication instants for every document below), the previous decision's
    statement, the statement before it (the §44 diff needs both), the previous
    meeting's minutes, and every FED_SPEECH since — plus the §45 minute-bar
    windows for SPY/QQQ/TLT/GLD/UUP around the previous decision. Every
    outcome, success or failure, is a named row in the response: a packet with
    the diff but no minutes is still most of the §44 answer, so one dead
    document never costs the other four.

    KEYLESS BUT NEVER ANONYMOUS. federalreserve.gov answers 403 to a request
    without a User-Agent, so every fetch carries the operator's contact address
    (``settings.sec_user_agent``, the single place a contact is configured).

    IDEMPOTENT ON THE DOCUMENT URL — one Fed URL is one document forever, so a
    second press re-parses the same pages and overwrites the same rows rather
    than accumulating a second copy the diff would have to choose between. The
    minute-bar helper refuses to refetch a window it already stored.

    AS-OF IS HONOURED BY THE FETCH ITSELF, not just by the store: a statement
    whose RSS publication instant is later than ``as_of`` is reported
    NOT_YET_RELEASED without an HTTP request being issued at all, which is what
    makes a point-in-time replay cheap as well as correct.

    Writes a SYSTEM-of-record ``DATA_BACKFILL`` audit row carrying
    ``kind: "event_fed"`` and the counts, in the same transaction as the data
    (rule 12, ADR-003).
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no Fed documents exist for it")
    settings = get_settings()
    return await event_fed.backfill_fed(
        session, row, settings=settings, as_of=as_of_dt
    )


# ---------------------------------------------------------------------------
# GET /api/events/{id}/risk (Phase K, §62-§67) — SHADOW
# ---------------------------------------------------------------------------


@router.get("/{event_id}/risk")
async def get_event_risk(
    event_id: int,
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §63 event-risk snapshot for one event + the §66 options panel.

    SHADOW, ALWAYS (§65). ``enforcement`` is the literal string ``"SHADOW"``
    and nothing this route computes has ever resized, rejected or paused a
    trade. It exists so a human can read the state, the drivers behind it and
    the caveats against it BEFORE any of it is promoted into enforcement —
    which is a separate, explicit decision that no backtest has yet earned.

    NO LLM ASSIGNS THE STATE (§63). ``event_risk_state`` comes out of a
    deterministic table in ``libs/trading_core/risk/event_risk.py``: expected
    move, imminence and exposure share, each with the threshold it crossed
    named in ``drivers``. Running this route twice on the same stored rows
    returns the same state, forever. A model's *narrative* about an event
    lives on the ANALYSIS tab, and the separation is the point.

    UNKNOWN IS NOT LOW. With neither a stored straddle for this print nor a
    realized move from a single previous one, the state is ``UNKNOWN`` with a
    ``reason``, and ``coverage.reason`` names the backfill that would fill the
    gap. A state of LOW would claim a measurement nobody made — the most
    dangerous of the five values to fabricate, because it is the one that
    reads as permission.

    EVERY HISTORICAL STATISTIC CARRIES ITS ``n`` (§64). ``historical`` is
    ``{median_abs, p75_abs, p90_abs, max_abs, n}`` in one mapping, so no
    consumer can render "median move 7.1%" without "based on 8 events" in
    hand; with ``n = 0`` all four statistics are ``null`` rather than zero.

    THIS ROUTE NEVER FETCHES. Both the straddle and the previous prints'
    realized moves are read from ``event_option_metrics``, written by the
    Phase I backfill; the exposure and NAV are cost-basis reads of the open
    book. A catalyst page must not spend a provider call, and an implied move
    quoted from a live chain on every page load would be a different number
    every refresh with no record of either.

    NAV IS COST BASIS HERE and says so in ``nav_basis``. The risk view's NAV
    marks the book to stored closes; this surface reads registry rows only, so
    exposure and NAV are both taken at cost and their ratio is a magnitude
    check for a threshold bump, never a valuation.

    A non-single-name event (a CPI release, an FOMC decision) answers 200 with
    ``available: false`` and a reason: the row exists, it simply has no issuer
    whose position this would be. Only a missing event is a 404. An FOMC
    decision within three days rides along in ``market_wide`` on EVERY
    response — it moves the whole book at once, so it is reported beside a
    ticker's own state and never folded into it (§62).
    """
    row = await _event_or_404(session, event_id)
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no event risk exists for it")
    nav = await event_risk.nav_at_cost(session)
    payload = await event_risk.event_risk_payload(
        session, row, as_of=as_of_dt, nav=nav
    )
    payload["nav_at_cost"] = nav
    payload["nav_basis"] = "COST"
    return payload


# ---------------------------------------------------------------------------
# GET /api/events/study (Phase L, §86, §92) — the measurement harness
# ---------------------------------------------------------------------------


@router.get("/study")
async def get_event_study(
    event_type: str | None = Query(
        default=None,
        description=(
            "narrow the sample to one EventType (EARNINGS, GUIDANCE, ...); "
            "omit for every single-name event with a stored analysis"
        ),
    ),
    min_n: int | None = Query(
        default=None,
        ge=MIN_MEANINGFUL_N,
        le=MAX_MIN_N,
        description=(
            "raise the bar at which a correlation is flagged NOT_MEANINGFUL. "
            f"Cannot be set below the library floor of {MIN_MEANINGFUL_N} — "
            "this parameter may make the table stricter, never more quotable."
        ),
    ),
    as_of: str | None = Query(
        default=None,
        description="ISO-8601 instant to answer as of; defaults to now (UTC)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §86 measurement: how each pre-event feature ranked against what
    the market actually did, over THIS installation's own stored history.

    §86 IS TWO PROHIBITIONS AND ONE INSTRUCTION — *"Do NOT assume they are
    predictive. Measure."* This route is the measurement, and it is
    deliberately the least persuasive surface on the platform. It returns
    Spearman's rho and ``n`` per feature per horizon, the coverage behind each
    column, a fixed caveat block, and nothing that reads as a conclusion:
    there is no verdict field, no ranking by |rho| and no threshold at which a
    feature becomes "validated". A reader who wants to know whether news
    materiality predicts earnings reactions gets the numbers and has to decide
    for themselves — which, at this sample size, is the only honest offer.

    NOT_MEANINGFUL BELOW n = 12 (§92). Every cell carries its own ``n`` and a
    flag when the pairing count is below the floor, because a rho over four
    events is ±0.8 about as often as not and a table that renders it like a
    real result has already misled. ``min_n`` may RAISE that floor and, by
    construction, cannot lower it — the one thing this knob must never do is
    make a tiny sample look quotable.

    NO P-VALUE IS COMPUTED ANYWHERE BEHIND THIS ROUTE (§92). |rho| and n are
    the whole report. A significance figure over a watchlist's worth of prints
    would print like evidence without being any, which is precisely the fake
    precision §92 names.

    THIS ROUTE NEVER FETCHES, and here that is a correctness property rather
    than a courtesy. Features are read out of each event's EARLIEST STORED
    BUNDLE — the evidence exactly as it stood before the print, with every
    as-of gate already applied when it was assembled. Re-assembling them today
    would measure each feature against bars, filings and articles that already
    contain the reaction, which would inflate the very correlations this report
    exists to state honestly. Outcomes, by contrast, ARE measured with
    hindsight from stored daily bars, because the realised reaction is the
    thing being predicted; the discipline belongs on the feature side and lives
    there.

    LIVE OPTION SNAPSHOTS ARE EXCLUDED (§85). Only the
    ``HISTORICAL_DAILY_CLOSE_APPROXIMATION`` basis feeds ``implied_move_pct``
    and ``iv_before``: a ``LIVE_CHAIN_SNAPSHOT`` row is written whenever
    somebody opened the options tab, which for a past event may be days after
    it printed, and §85 states plainly that a live chain cannot be
    reconstructed point-in-time. Correlating a possibly-post-event snapshot
    with the move it was taken after would be the largest look-ahead this
    payload could carry — and it would surface as the strongest column in the
    table.

    TWO §86 CANDIDATES ARE NOT MEASURED AND ARE NAMED ANYWAY, in
    ``report.not_measurable``: estimate revision (no consensus vendor in the
    subscription — §33) and valuation expansion (the filing history is only as
    deep as the backfill, so the measurement would describe the backfill). A
    table that silently dropped them would read as a claim that §86's list was
    fully covered.

    HONEST EMPTINESS, NOT A 404. An install that has analysed no events
    answers 200 with ``insufficient_data: true``, zero-coverage columns and the
    full caveat block. "Nothing has been measured yet" is a complete and
    correct §86 answer, and the feature list plus the caveats are the useful
    half of the response even then.

    RESEARCH ONLY (§87). Nothing in this payload is wired to a signal, a trade
    plan or an order, and no code path reads it into one.
    """
    now = datetime.now(timezone.utc)
    as_of_dt = _resolve_as_of(as_of, now, subject="no events have occurred yet")
    if event_type is not None:
        try:
            event_type = EventType(event_type).value
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown event_type {event_type!r}; expected one of "
                    f"{sorted(t.value for t in EventType)}"
                ),
            ) from exc
    return await event_study.build_study_payload(
        session, as_of=as_of_dt, event_type=event_type, min_n=min_n
    )


# STATIC PATH, PARAMETERISED NEIGHBOUR. ``GET /{event_id}`` is registered far
# above (it has to be — every per-event tab hangs off it), and Starlette
# matches routes in REGISTRATION order, so an appended ``/study`` would be
# swallowed by it and answered 422 "study is not a valid integer". The rest of
# the file solves this by declaring its static paths (``/calendar``) BEFORE the
# parameterised one; this route is appended at the end instead, so it moves
# itself to the front of the table here rather than being wedged into the
# middle of the per-event section where a reader looking for Phase L would not
# find it. Only the ordering moves — the route, its path and its handler are
# exactly what the decorator above declared.
router.routes.insert(0, router.routes.pop())
