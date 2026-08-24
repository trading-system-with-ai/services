"""Event-risk SEAM — the gateway's DB side of §62-§67, SHADOW ONLY (Phase K).

The pure classifier lives in ``libs/trading_core/risk/event_risk.py`` and has
no database, no clock and no network. This module is the other half: it reads
the stored event registry and the stored option metrics, assembles an
:class:`~libs.trading_core.risk.event_risk.EventRiskInputs`, and hands the
snapshot back. The split is the same one the rest of the platform uses (a pure
library plus a gateway seam) and it is what lets the whole §63 table be tested
without a session.

**NOTHING HERE CAN CHANGE AN ORDER.** Every consumer of this module is a
SHADOW surface (§65): the ``shadow.event`` block in the RISK_DECISION audit
row, the ``event_risk`` panel on a trade plan, and ``GET
/api/events/{id}/risk``. The caps :func:`shadow_event_block` returns join the
hypothetical shadow verdict the statistical layer already computes — they are
NEVER passed to ``assess(extra_caps=...)``, which stays empty in ``apps/``
because no backtest has validated these thresholds yet. The enforcement mode
travels in the payload as the literal string ``"SHADOW"`` so a reader of a
stored audit row can never mistake a warning for a resize.

**THIS MODULE NEVER FETCHES** (§27; audit §7.2 rule 1). It runs on GETs and,
worse, inside the order gate chain, where an HTTP round trip would put a
provider's latency and a provider's outage on the path of a trading decision.
Every number comes from a stored row: the event registry (``events``), the
option metrics the Phase I backfill wrote (``event_option_metrics``), and the
open positions. An event nobody has backfilled option metrics for produces an
honest ``UNKNOWN`` naming the backfill, not a guessed move.

**NO LLM ANYWHERE** (§63). There is no provider call, no prompt and no import
of an analysis module in this file; the state comes out of the deterministic
table in the pure library and nothing else. ``tests/test_event_risk_api.py``
asserts that statically.

**HISTORY COMES FROM REALIZED MOVES, NOT FROM A REACTION RE-COMPUTATION.**
``EventOptionMetricRow.actual_move_pct`` is the stored realized move of each
previous comparable print, written by the same Phase I backfill that stored
the straddle it is compared against. Using it here means the §64 median on the
Risk tab and the §66 median on the Options tab are the SAME number computed
the SAME way — two different medians for one ticker's prints would be a bug
the user would have to arbitrate. When no metrics are stored the sample is
simply empty (``n=0``) and says so; it is never backfilled from a live price
series on a read.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.events.implied_move import (
    BASIS_HISTORICAL,
    BASIS_LIVE,
    STATUS_NO_DATA,
)
from libs.trading_core.models.enums import EventStatus, EventType
from libs.trading_core.risk.event_risk import (
    EVENT_RISK_MODEL_VERSION,
    EventRiskInputs,
    EventRiskPolicy,
    EventRiskThresholds,
    STATE_UNKNOWN,
    classify_event_risk,
    event_risk_caps,
)

from .db import EventOptionMetricRow, EventRow, Position

#: How far ahead :func:`upcoming_event_for` looks for the ticker's next print.
#: Two weeks is the §65 Trade Plan panel's horizon: a print a month out does
#: not change how a position is sized today, and surfacing it would put a
#: permanent low-grade warning on every plan for every ticker, which is how a
#: warning stops being read. A parameter, never a literal at the call site.
DEFAULT_HORIZON_DAYS = 14

#: How close a MARKET-WIDE FOMC decision must be to be flagged beside a
#: single-name event (§62: a macro print moves every position, so it is
#: reported as a SEPARATE flag rather than folded into the ticker's own state
#: — blending them would make "EARNINGS risk HIGH" mean two different things).
FOMC_FLAG_DAYS = 3.0

#: Event types this seam treats as a company-specific catalyst for a ticker.
#: EARNINGS only, for now: it is the one type with a stored realized-move
#: history and a stored straddle, which is what §63 needs to classify anything
#: at all. A GUIDANCE or PRODUCT_LAUNCH row would classify as UNKNOWN forever,
#: and an UNKNOWN chip on every plan teaches the user to ignore the chip.
TICKER_EVENT_TYPES = (EventType.EARNINGS.value,)

#: Statuses an upcoming event may carry. ESTIMATED is INCLUDED deliberately:
#: §7 forbids presenting a derived date as a fact, not knowing about it. The
#: snapshot carries ``is_estimated`` into the classifier, which turns it into
#: a caveat — "earnings in 1.3 days" that is really a cadence guess must be
#: labelled, but hiding it entirely would let a position walk into a print the
#: platform did know about.
UPCOMING_STATUSES = (
    EventStatus.CONFIRMED.value,
    EventStatus.ESTIMATED.value,
    EventStatus.REVISED.value,
)

#: How many previous comparable prints the historical sample draws from. Eight
#: matches ``event_options.DEFAULT_HISTORY`` and the §60 history table, so the
#: Risk tab's "based on N events" and the Options tab's history are the same
#: window rather than two samples the user has to reconcile.
HISTORY_EVENTS = 8

#: The reason a snapshot carries when the ticker has no upcoming event at all.
NO_EVENT_REASON = "no upcoming event within the horizon for this ticker"

#: The sentence every shadow surface carries. Stated once so the audit row,
#: the plan panel and the API cannot drift into three different promises.
SHADOW_NOTE = (
    "SHADOW (§65): the event-risk state, its warning and any cap are "
    "RESEARCH DEFAULTS that changed nothing. The approved quantity above is "
    "Tier 0's and is unaffected by this block."
)


#: The §66 explainer, stated once so the API and the UI cannot drift. It is
#: the single most useful sentence on the options panel: it names the failure
#: mode that surprises people who were RIGHT about the direction.
LONG_CALL_EXPLAINER = (
    "a long call can lose despite correct direction if realized move < priced "
    "implied move"
)

#: What the §66 options block says about the crush it cannot compute. An
#: expected IV crush is a forecast of a number nobody has published; this
#: platform stores realized crushes from past prints and refuses to project
#: one forward (§44 rule 18 — the honest absence, never a fabricated number).
CRUSH_NO_DATA = "NO_DATA"


def _as_utc(value: datetime | None) -> datetime | None:
    """Stored instants are UTC; SQLite hands them back naive (same convention
    as ``event_price._as_utc`` / ``routers/events._iso``)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite(value: object) -> float | None:
    """``float(value)`` when it is a real finite number, else ``None``.

    Booleans are rejected (``True`` is not a price), matching the pure
    library's own guard — a ``True`` that slipped through as ``1.0`` would be
    a 1% move nobody measured.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pct_from_fraction(value: object) -> float | None:
    """A stored FRACTION converted to the classifier's PERCENT number.

    THE UNIT SEAM, and the reason this function exists rather than a bare
    ``_finite`` call at each read site. The event-options layer stores every
    ``*_pct`` column as a FRACTION of spot — ``implied_move.pct`` is
    ``points / spot`` and ``actual_move_pct`` is ``post.close / pre_close -
    1.0``, so an 8.8% print is persisted as ``0.088`` (see
    :mod:`libs.trading_core.events.implied_move`, whose docstrings say "a
    fraction, not percentage points: the UI multiplies by 100"). The Phase K
    classifier documents the OPPOSITE convention — every move it takes is a
    PERCENT number, ``8.8`` meaning 8.8%.

    Both conventions are correct in their own module; what is not correct is
    handing one to the other unconverted. Doing so understates every event by
    100x: a real 8.8% earnings print arrives as ``0.088``, lands under the
    4.0% MODERATE floor, and an EXTREME event renders as "implied move 0.09%"
    in a panel whose whole purpose is to warn about the print. That is a
    silent, safety-relevant failure — the number still looks like a number —
    so the multiplication happens HERE, once, at the boundary where the
    fraction leaves storage, and never inline at a call site where the next
    reader would have to know both conventions to spot its absence.
    """
    out = _finite(value)
    return None if out is None else out * 100.0


def days_between(now: datetime, when: datetime | None) -> float | None:
    """Signed days from ``now`` to ``when`` — positive means "in the future".

    Fractional on purpose: §65's panel prints "Earnings in 1.3 days", and the
    imminence rule in the classifier is a ≤3-day window where rounding to
    whole days would move an event across the boundary.
    """
    when_utc = _as_utc(when)
    if when_utc is None:
        return None
    return (when_utc - _as_utc(now)).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def upcoming_event_for(
    session: AsyncSession,
    ticker: str,
    *,
    now: datetime,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> EventRow | None:
    """The ticker's NEAREST upcoming catalyst inside the horizon, or ``None``.

    Nearest, not most important: the §63 question is "what is this position
    about to sit through", and that is answered by the next print regardless
    of how the importance scorer ranked it. CANCELED rows are excluded (a
    canceled print is not a risk) and ESTIMATED ones are kept with their
    status travelling onward, per :data:`UPCOMING_STATUSES`.

    Strictly future: an event at exactly ``now`` has already happened for the
    purposes of a trade being placed now, and §67 puts post-event handling in
    a different module.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None
    now_utc = _as_utc(now)
    horizon = max(0, int(horizon_days))
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.ticker == symbol,
                    EventRow.event_type.in_(TICKER_EVENT_TYPES),
                    EventRow.status.in_(UPCOMING_STATUSES),
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        delta = days_between(now_utc, row.scheduled_at)
        if delta is None or delta <= 0.0:
            continue
        if delta > horizon:
            break  # ordered by date — everything after is further out
        return row
    return None


async def market_wide_flag(
    session: AsyncSession, *, now: datetime, within_days: float = FOMC_FLAG_DAYS
) -> dict | None:
    """The §62 MARKET-WIDE flag: an FOMC decision inside ``within_days``.

    Reported SEPARATELY from the ticker's own state and never folded into it.
    An FOMC decision is a risk to every position in the book at once, so
    expressing it as a bump to one ticker's event state would both overstate
    that ticker's idiosyncratic risk and understate the book's; the §65 panel
    prints it as its own line for exactly that reason.

    ``None`` when no meeting is close — an honest absence, not a false flag.
    """
    now_utc = _as_utc(now)
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == EventType.FOMC_DECISION.value,
                    EventRow.status.in_(UPCOMING_STATUSES),
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        delta = days_between(now_utc, row.scheduled_at)
        if delta is None or delta <= 0.0:
            continue
        if delta > within_days:
            break
        return {
            "event_id": row.id,
            "event_key": row.event_key,
            "event_type": row.event_type,
            "title": row.title,
            "scheduled_at_utc": _as_utc(row.scheduled_at).isoformat(),
            "days_away": delta,
            "is_estimated": row.status == EventStatus.ESTIMATED.value,
            "note": (
                "MARKET-WIDE: an FOMC decision moves every position in the "
                "book, so it is reported beside the ticker's own event state "
                "and never folded into it (§62)."
            ),
        }
    return None


async def _previous_prints(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> list[EventRow]:
    """Past comparable EARNINGS rows for this event's ticker, oldest first.

    Delegates to :func:`apps.gateway.event_price._past_comparable_rows` — the
    SAME point-in-time pool the price tab and the options tab walk. Reusing it
    is what keeps "the last 8 prints" from meaning three different sets of
    events on three tabs of the same page.
    """
    from .event_price import _past_comparable_rows  # local: avoids an import cycle

    return await _past_comparable_rows(session, event_row, as_of)


async def _stored_metrics(
    session: AsyncSession, event_ids: Sequence[int]
) -> dict[int, EventOptionMetricRow]:
    """``{event_id: row}`` for the HISTORICAL basis — ONE query for the whole
    sample, matching ``event_options._stored_metrics_for``'s reason: eight
    per-row lookups is eight queries on an order path."""
    ids = [int(i) for i in event_ids]
    if not ids:
        return {}
    rows = (
        (
            await session.execute(
                select(EventOptionMetricRow).where(
                    EventOptionMetricRow.event_id.in_(ids),
                    EventOptionMetricRow.basis == BASIS_HISTORICAL,
                )
            )
        )
        .scalars()
        .all()
    )
    return {row.event_id: row for row in rows}


async def _latest_metric(
    session: AsyncSession, event_id: int
) -> EventOptionMetricRow | None:
    """This event's own most recent stored metrics row, ANY basis.

    Newest ``as_of`` wins. A LIVE_CHAIN_SNAPSHOT written minutes ago and a
    HISTORICAL_DAILY_CLOSE_APPROXIMATION reconstruction can both exist for one
    event (that is what ``UNIQUE(event_id, basis)`` is for); the fresher one is
    the better description of what the market is pricing right now, and the
    basis travels into the payload so nobody has to guess which it was.
    """
    rows = (
        (
            await session.execute(
                select(EventOptionMetricRow)
                .where(EventOptionMetricRow.event_id == int(event_id))
                .order_by(EventOptionMetricRow.as_of.desc())
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        return row
    return None


async def position_exposure_for(
    session: AsyncSession, ticker: str
) -> float | None:
    """Absolute USD exposure of the OPEN position in ``ticker``, at COST.

    Cost basis (``quantity × avg_price × multiplier``), matching
    ``routers/events._exposure_map``'s deliberate choice: this module reads
    stored rows only and never touches a market-data provider, so a "market
    value" quoted here would both break that separation and put a synthetic
    number into an unconfigured install's audit row. It is an exposure SCALE
    for a percent-of-NAV bump, not a mark.

    ``None`` — never ``0.0`` — when nothing is held: the classifier
    distinguishes "no exposure measured" from "zero exposure", and a
    fabricated zero would silently make every snapshot look small.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None
    rows = (
        (
            await session.execute(
                select(Position).where(
                    Position.ticker == symbol, Position.status == "OPEN"
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return None
    total = math.fsum(
        abs(float(row.quantity) * float(row.avg_price) * float(row.multiplier or 1))
        for row in rows
    )
    return total if math.isfinite(total) else None


async def nav_at_cost(session: AsyncSession) -> float | None:
    """A COST-BASIS NAV: portfolio cash + Σ|open position cost|.

    NOT the risk view's NAV, and labelled as such wherever it is used. The
    risk engine's NAV marks every position to its latest stored close and
    reads the SPY regime on the way; that is the right number for a sizing
    decision and the wrong cost for a catalyst page, which is a stored-rows
    read that must not walk the bar tables for a percent-of-NAV denominator.

    Consistency is why it is cost basis on BOTH sides: the exposure numerator
    from :func:`position_exposure_for` is cost too, so ``exposure / nav`` is a
    ratio of two comparable quantities rather than a mark divided by a book
    value. The ratio is a magnitude check for a threshold bump — "is this
    position a tenth of the account or a quarter of it" — not a valuation.

    ``None`` when the result is not a positive finite number; the classifier
    then reports ``exposure_share: None`` with its caveat rather than a share
    of nothing.
    """
    from .db import get_or_create_portfolio  # local: keeps the import surface flat

    portfolio = await get_or_create_portfolio(session)
    cash = _finite(portfolio.cash) or 0.0
    rows = (
        (
            await session.execute(select(Position).where(Position.status == "OPEN"))
        )
        .scalars()
        .all()
    )
    total = cash + math.fsum(
        abs(float(row.quantity) * float(row.avg_price) * float(row.multiplier or 1))
        for row in rows
    )
    return total if math.isfinite(total) and total > 0.0 else None


def greeks_from_rows(
    rows: Sequence[Mapping[str, object]] | None, ticker: str
) -> dict[str, float] | None:
    """Net gamma/vega/theta of the open OPTION positions in ``ticker``.

    Reads the per-position rows ``routers/portfolio.portfolio_greeks_read``
    already produced for the book the order chain is judging — the §16 view.
    Wiring the existing view rather than recomputing is the point: the Risk
    tab's "position sensitivity" and the portfolio greeks page then describe
    the same position with the same numbers.

    Rows with ``data_ok: false`` are SKIPPED, not read as zeros. That row's
    documented zeros exist so a missing contract does not corrupt the
    portfolio TOTAL; borrowing them here would turn "the contract fell off
    today's chain" into "this position has no convexity", which is the
    opposite claim. If no usable row survives, the answer is ``None`` — the
    classifier renders that as an absent greeks block, never as a dict of
    zeros (an all-zero greeks block reads as a measured stock position).
    """
    symbol = (ticker or "").strip().upper()
    if not rows or not symbol:
        return None
    gamma = vega = theta = 0.0
    seen = False
    for row in rows:
        if str(row.get("ticker") or "").strip().upper() != symbol:
            continue
        if not row.get("data_ok"):
            continue
        g = _finite(row.get("gamma"))
        v = _finite(row.get("vega_usd"))
        t = _finite(row.get("theta_usd_per_day"))
        if g is None and v is None and t is None:
            continue
        seen = True
        gamma += g or 0.0
        vega += v or 0.0
        theta += t or 0.0
    if not seen:
        return None
    return {"gamma": gamma, "vega": vega, "theta": theta}


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------


async def snapshot_for_event(
    session: AsyncSession,
    event_row: EventRow,
    *,
    now: datetime,
    position_exposure_usd: float | None = None,
    nav: float | None = None,
    option_greeks: Mapping[str, float | None] | None = None,
    thresholds: EventRiskThresholds = EventRiskThresholds(),
) -> dict:
    """The §63 snapshot for ONE stored event row, plus its provenance.

    Returns ``classify_event_risk(...)``'s exact 15 keys with three seam keys
    added — ``event_id``, ``event_key`` and ``coverage`` — so a consumer can
    both render the state and answer "why is n=0" without a second request.
    ``coverage`` names the backfill when the sample is empty, which is the
    difference between "this stock does not move" and "nobody has looked".
    """
    now_utc = _as_utc(now)
    scheduled = _as_utc(event_row.scheduled_at)
    prints = await _previous_prints(session, event_row, as_of=now_utc)
    selected = prints[-HISTORY_EVENTS:] if HISTORY_EVENTS else []
    metrics = await _stored_metrics(session, [row.id for row in selected])

    moves: list[float] = []
    with_metrics = 0
    for row in selected:
        metric = metrics.get(row.id)
        if metric is None:
            continue
        with_metrics += 1
        if metric.status == STATUS_NO_DATA:
            # NO_DATA is the server retracting its own computation; a number
            # sitting beside it is withdrawn, so it must not enter the sample.
            continue
        # Stored as a FRACTION; the classifier takes PERCENT numbers.
        move = _pct_from_fraction(metric.actual_move_pct)
        if move is not None:
            moves.append(move)

    own = await _latest_metric(session, event_row.id)
    implied_pct: float | None = None
    implied_basis: str | None = None
    if own is not None and own.status != STATUS_NO_DATA:
        # Stored as a FRACTION; the classifier takes PERCENT numbers.
        implied_pct = _pct_from_fraction(own.implied_move_pct)
        if implied_pct is not None:
            implied_basis = own.basis

    is_estimated = event_row.status == EventStatus.ESTIMATED.value
    inputs = EventRiskInputs(
        event_type=event_row.event_type,
        time_to_event_days=days_between(now_utc, scheduled),
        historical_moves=moves,
        implied_move_pct=implied_pct,
        implied_basis=implied_basis,
        position_exposure_usd=_finite(position_exposure_usd),
        portfolio_nav_usd=_finite(nav),
        option_gamma=_finite((option_greeks or {}).get("gamma")),
        option_vega=_finite((option_greeks or {}).get("vega")),
        option_theta=_finite((option_greeks or {}).get("theta")),
        is_estimated=is_estimated,
    )
    snapshot = classify_event_risk(inputs, thresholds=thresholds)
    snapshot["event_id"] = event_row.id
    snapshot["event_key"] = event_row.event_key
    snapshot["ticker"] = event_row.ticker
    snapshot["scheduled_at_utc"] = scheduled.isoformat() if scheduled else None
    snapshot["is_estimated"] = is_estimated
    snapshot["coverage"] = {
        "history_events": len(prints),
        "history_with_metrics": with_metrics,
        "history_moves_used": len(moves),
        "own_metrics": own is not None,
        "reason": (
            None
            if moves or implied_pct is not None
            else (
                "no stored option metrics for this event or its previous "
                "prints — use POST /api/events/{id}/options/backfill and POST "
                "/api/events/{id}/options/history/backfill"
            )
        ),
    }
    return snapshot


async def snapshot_for(
    session: AsyncSession,
    ticker: str,
    *,
    now: datetime,
    position_exposure_usd: float | None = None,
    nav: float | None = None,
    option_greeks: Mapping[str, float | None] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict | None:
    """The §63 snapshot for a TICKER's next event, or ``None`` when there is
    none inside the horizon.

    ``None`` rather than an UNKNOWN snapshot on purpose: "this ticker has no
    print coming" is a different fact from "there is a print and we could not
    measure it", and only the second one deserves a chip on a trade plan.
    """
    event_row = await upcoming_event_for(
        session, ticker, now=now, horizon_days=horizon_days
    )
    if event_row is None:
        return None
    return await snapshot_for_event(
        session,
        event_row,
        now=now,
        position_exposure_usd=position_exposure_usd,
        nav=nav,
        option_greeks=option_greeks,
    )


# ---------------------------------------------------------------------------
# §65 — the SHADOW block for the order path
# ---------------------------------------------------------------------------


async def shadow_event_block(
    session: AsyncSession,
    *,
    ticker: str,
    requested_qty: int,
    price: float | None,
    nav: float | None,
    position_exposure_usd: float | None = None,
    option_greeks: Mapping[str, float | None] | None = None,
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    policy: EventRiskPolicy = EventRiskPolicy(),
) -> tuple[dict, list]:
    """``(block, caps)`` for the RISK_DECISION ``shadow.event`` key — §65.

    ``block`` is what lands in the audit row and ``caps`` are real
    :class:`~libs.trading_core.risk.pretrade.QuantityCap` rows for the
    HYPOTHETICAL shadow verdict the statistical layer already computes. They
    join that ONE verdict exactly the way the Phase D stress caps do, because
    a trade gets one shadow answer rather than one per layer.

    **The caps are returned, never applied.** Nothing in this call reaches
    ``assess``; the caller passes them to ``_pretrade_statistical_shadow``'s
    ``extra_caps`` — the hypothetical parameter — and ``assess(extra_caps=())``
    stays empty (§65: promotion out of SHADOW is an explicit human step, and
    no backtest has validated these thresholds).

    ``verdict`` is deliberately named ``would_warn`` / ``would_cap_qty``: the
    subjunctive is the contract. ``would_warn`` is true from the policy's
    ``warn_from`` rung upward (HIGH by default), which is a WIDER set than the
    rung that emits a cap — §65's WARN tier exists precisely so a MODERATE
    event is visible without pretending to a resize.

    Returns ``({"event": None...}, [])`` — never raises for a data gap — when
    the ticker has no upcoming event. Genuine infrastructure faults propagate
    and the ORDER PATH's own ``try/except`` turns them into a note; that split
    is deliberate, so a missing table is loud in the logs while a missing
    event is quiet in the payload.
    """
    stamp = now or datetime.now(timezone.utc)
    snapshot = await snapshot_for(
        session,
        ticker,
        now=stamp,
        position_exposure_usd=position_exposure_usd,
        nav=nav,
        option_greeks=option_greeks,
        horizon_days=horizon_days,
    )
    market_wide = await market_wide_flag(session, now=stamp)
    if snapshot is None:
        return (
            {
                "snapshot": None,
                "caps": [],
                "verdict": {"would_warn": False, "would_cap_qty": None},
                "market_wide": market_wide,
                "enforcement": policy.mode,
                "model_version": EVENT_RISK_MODEL_VERSION,
                "reason": NO_EVENT_REASON,
                "note": SHADOW_NOTE,
            },
            [],
        )

    caps = event_risk_caps(
        snapshot,
        requested_qty=int(requested_qty),
        price=_finite(price),
        nav=_finite(nav),
        policy=policy,
    )
    state = snapshot.get("event_risk_state")
    would_warn = _warns(state, policy)
    would_cap_qty = min((cap.cap_qty for cap in caps), default=None)
    return (
        {
            "snapshot": snapshot,
            "caps": [
                {
                    "code": cap.code,
                    "layer": cap.layer,
                    "cap_qty": cap.cap_qty,
                    "sentence": cap.sentence,
                    "measured": dict(cap.measured),
                }
                for cap in caps
            ],
            "verdict": {"would_warn": would_warn, "would_cap_qty": would_cap_qty},
            "market_wide": market_wide,
            "enforcement": policy.mode,
            "model_version": EVENT_RISK_MODEL_VERSION,
            "reason": None,
            "note": SHADOW_NOTE,
        },
        list(caps),
    )


def _warns(state: object, policy: EventRiskPolicy) -> bool:
    """Whether ``state`` is at or above the policy's WARN rung.

    ``UNKNOWN`` never warns: a warning implies a measurement, and UNKNOWN's
    whole content is that no measurement was made. It is rendered as its own
    chip instead (§63) — loud in a different way, and honest.
    """
    from libs.trading_core.risk.event_risk import STATE_LADDER

    if state == STATE_UNKNOWN or state not in STATE_LADDER:
        return False
    warn_from = policy.warn_from
    if warn_from not in STATE_LADDER:
        return False
    return STATE_LADDER.index(str(state)) >= STATE_LADDER.index(warn_from)


# ---------------------------------------------------------------------------
# §66 — the options block for GET /api/events/{id}/risk
# ---------------------------------------------------------------------------


async def options_risk_block(
    session: AsyncSession, event_row: EventRow, *, now: datetime
) -> dict:
    """The §66 options panel: what the market is charging, what it charged
    before, and the trap that costs a directionally-correct trade money.

    Four honest parts:

    - ``event_iv`` / ``implied_move_pct``: this event's own stored straddle,
      with its ``basis`` — LIVE_CHAIN_SNAPSHOT and
      HISTORICAL_DAILY_CLOSE_APPROXIMATION are different claims (§37) and are
      labelled, never blended. Both are PERCENT numbers here
      (:func:`_pct_from_fraction`), matching the snapshot beside them, so the
      whole payload speaks one unit convention;
    - ``expected_iv_crush``: the fixed string ``NO_DATA``. This platform
      subscribes to no forward vol surface, and an "expected crush" would be a
      forecast of a number nobody published. The honest absence is the answer;
    - ``historical_iv_crush``: the REALIZED crushes stored for previous
      prints, with their ``n`` (§64) — the only crush evidence that exists;
    - ``explainer``: :data:`LONG_CALL_EXPLAINER`, carried into the payload so
      a consumer that renders the implied move cannot render it without the
      sentence that explains how being right still loses.
    """
    from libs.trading_core.risk.event_risk import historical_event_risk

    own = await _latest_metric(session, event_row.id)
    prints = await _previous_prints(session, event_row, as_of=_as_utc(now))
    selected = prints[-HISTORY_EVENTS:] if HISTORY_EVENTS else []
    metrics = await _stored_metrics(session, [row.id for row in selected])

    crushes: list[float] = []
    implieds: list[float] = []
    for row in selected:
        metric = metrics.get(row.id)
        if metric is None or metric.status == STATUS_NO_DATA:
            continue
        # Both columns are stored FRACTIONS (`iv_crush` is
        # `iv_after / iv_before - 1`), and `historical_event_risk` reports
        # PERCENT numbers like every other statistic on this payload.
        crush = _pct_from_fraction(metric.iv_crush_pct)
        if crush is not None:
            crushes.append(crush)
        implied = _pct_from_fraction(metric.implied_move_pct)
        if implied is not None:
            implieds.append(implied)

    own_ok = own is not None and own.status != STATUS_NO_DATA
    return {
        # PERCENT numbers, matching the snapshot beside them: this payload
        # speaks ONE unit convention end to end, so a consumer never has to
        # know which key came from which storage column to render it.
        "event_iv": _pct_from_fraction(own.iv_before) if own_ok else None,
        "implied_move_pct": (
            _pct_from_fraction(own.implied_move_pct) if own_ok else None
        ),
        "implied_basis": own.basis if own_ok else None,
        "implied_status": own.status if own is not None else None,
        "is_live_basis": bool(own_ok and own.basis == BASIS_LIVE),
        # §44 rule 18: the crush this platform cannot know, named rather than
        # invented. A number here would be a forecast wearing a measurement's
        # clothes.
        "expected_iv_crush": CRUSH_NO_DATA,
        "expected_iv_crush_note": (
            "no forward volatility surface is subscribed, so the IV crush this "
            "print will produce is NOT forecast. The realized crushes below "
            "are what past prints actually did."
        ),
        "historical_iv_crush": historical_event_risk(crushes),
        "historical_implied_move": historical_event_risk(implieds),
        "explainer": LONG_CALL_EXPLAINER,
    }


async def event_risk_payload(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    nav: float | None = None,
    position_exposure_usd: float | None = None,
    option_greeks: Mapping[str, float | None] | None = None,
) -> dict:
    """``GET /api/events/{id}/risk``'s body — the §63 snapshot + §66 options.

    ``available`` is false with a reason for an event that has no issuer whose
    risk this could be (a CPI release has no position and no straddle). The
    row still answers 200: the event exists, it simply is not a single-name
    catalyst, and a 404 would claim otherwise.
    """
    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return {
            "event_id": event_row.id,
            "event_key": event_row.event_key,
            "as_of": _as_utc(as_of).isoformat(),
            "available": False,
            "reason": (
                "this event has no ticker — event risk is measured for ONE "
                "position against ONE issuer's print"
            ),
            "snapshot": None,
            "options": None,
            "market_wide": await market_wide_flag(session, now=as_of),
            "enforcement": EventRiskPolicy().mode,
            "model_version": EVENT_RISK_MODEL_VERSION,
            "note": SHADOW_NOTE,
        }

    exposure = (
        position_exposure_usd
        if position_exposure_usd is not None
        else await position_exposure_for(session, ticker)
    )
    snapshot = await snapshot_for_event(
        session,
        event_row,
        now=as_of,
        position_exposure_usd=exposure,
        nav=nav,
        option_greeks=option_greeks,
    )
    return {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": ticker,
        "as_of": _as_utc(as_of).isoformat(),
        "available": True,
        "reason": None,
        "snapshot": snapshot,
        "options": await options_risk_block(session, event_row, now=as_of),
        "market_wide": await market_wide_flag(session, now=as_of),
        "enforcement": EventRiskPolicy().mode,
        "model_version": EVENT_RISK_MODEL_VERSION,
        "note": SHADOW_NOTE,
    }


async def plan_event_risk(
    session: AsyncSession,
    ticker: str,
    *,
    now: datetime | None = None,
) -> dict | None:
    """The §65 Trade Plan panel's block, computed FRESH on every read.

    Never stored inside ``preview``: a plan generated last Tuesday would
    otherwise keep announcing "earnings in 1.3 days" forever, and a stale
    countdown is worse than none — it is a false statement with a number
    attached. The exposure and NAV are deliberately omitted here (a plan is a
    research artifact, not a position), so ``exposure_share`` is an honest
    ``None`` with its caveat rather than a share of a portfolio the plan does
    not own.

    ``None`` when the ticker has no print inside the horizon — the panel then
    does not render at all, rather than rendering an empty one.
    """
    stamp = now or datetime.now(timezone.utc)
    snapshot = await snapshot_for(session, ticker, now=stamp)
    if snapshot is None:
        return None
    return {
        "snapshot": snapshot,
        "enforcement": EventRiskPolicy().mode,
        "model_version": EVENT_RISK_MODEL_VERSION,
        "computed_at": _as_utc(stamp).isoformat(),
        "note": SHADOW_NOTE,
    }
