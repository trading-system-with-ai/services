"""Position monitor API (development plan §11, §37, §38).

``GET /api/positions`` lists paper positions with their live exit-engine read:
every OPEN stock row is evaluated by
:func:`libs.trading_core.exits.evaluate_exit` and every OPEN option row by
:func:`libs.trading_core.exits.evaluate_option_exit` — the SAME engines the
backtest validated (plan §21), never reimplemented here — and carries the
full per-rule reason list, "OK:"-prefixed for rules that did NOT fire, so the
user can always see why the system is still holding (§37). Option rows'
``exit_reasons`` therefore include the option families (§11.3 PREMIUM
hard stop, §11.7 DTE exit) alongside the shared underlying-driven rules.
CLOSED rows carry ``realized_pnl`` and honest nulls elsewhere (§44 rule 18).
Read-only: no audit events (rule 12 covers state changes and decisions).

Option rows (plan §12.1 conventions): ``quantity`` counts CONTRACTS,
``avg_price`` is the entry premium PER SHARE (mid at fill), ``market_value``
is ``qty * current_mid * 100``, ``max_loss`` the premium paid. The
``contract`` block carries the identity plus the live read — remaining
``dte`` from ``opt_expiry`` and ``current_mid`` from the SAME contract in
today's regenerated chain (shared helper in routers/options.py); both are
honest nulls when unavailable (e.g. the contract expired off the chain).

``POST /api/positions/check-exits`` runs the same evaluation for every OPEN
position and EXECUTES the triggered exits for BOTH instrument kinds through
the shared sell-to-close flow: each fires an EXIT_GENERATED audit event
(SYSTEM) and the sell (ORDER_REQUESTED actor SYSTEM) in the same
transaction. Option exits close at the current chain mid, falling back to
INTRINSIC value when the contract is missing from today's chain (documented
in routers/orders.py). Mechanical exits are NOT blocked by the §18 kill
switch: exits reduce risk, and risk protection outranks the pause (§18
risk-priority).
"""
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.exits import (
    ExitDecision,
    OptionState,
    PositionState,
    evaluate_exit,
    evaluate_option_exit,
)
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import Position, StockBarDaily, get_session
from .options import build_option_chain
from .orders import (
    POSITION_OPEN,
    execute_sell_to_close,
    execution_lock,
    find_option_contract,
    is_option_position,
    option_intrinsic_value,
)

router = APIRouter(prefix="/api/positions", tags=["positions"])

VALID_STATUS = ("OPEN", "CLOSED", "ALL")


async def _stored_bars(session: AsyncSession, ticker: str) -> list[StockBarDaily]:
    """All stored daily bars for `ticker`, oldest first (no lazy backfill —
    the monitor only reads what execution already stored)."""
    rows = await session.execute(
        select(StockBarDaily)
        .where(StockBarDaily.ticker == ticker)
        .order_by(StockBarDaily.ts)
    )
    return list(rows.scalars().all())


def _bars_held(position: Position, bars: list[StockBarDaily]) -> int:
    """Bars since entry; the entry bar (``entry_bar_date``) is bar 0 (§11).

    A missing ``entry_bar_date`` conservatively counts 0 — the time stop can
    then never fire early, while every price-based rule still protects.
    """
    if position.entry_bar_date is None:
        return 0
    entry = date.fromisoformat(position.entry_bar_date)
    return sum(1 for b in bars if b.ts > entry)


def _option_live_read(
    position: Position, spot: float
) -> tuple[int | None, float | None]:
    """Live ``(dte_remaining, current_mid)`` for one OPEN option position.

    ``dte`` counts calendar days from today to ``opt_expiry``, clamped at 0
    once expired (the remaining time cannot be negative); ``current_mid`` is
    the SAME contract's mid in today's regenerated chain via the SHARED
    helper (routers/options.py) — an honest ``None`` when the contract is
    missing from the chain (e.g. expired), which makes the §11.3 premium
    stop report "insufficient data" loudly rather than pretending (§44
    rule 18).
    """
    dte: int | None = None
    if position.opt_expiry:
        today = datetime.now(timezone.utc).date()
        dte = max(0, (date.fromisoformat(position.opt_expiry) - today).days)
    _, chain = build_option_chain(position.ticker, spot)
    contract = find_option_contract(chain, position)
    return dte, contract.mid if contract is not None else None


def _evaluate_open_position(
    position: Position,
    bars: list[StockBarDaily],
    option_read: tuple[int | None, float | None] | None = None,
) -> tuple[ExitDecision | None, str | None]:
    """Run the shared exit engine for one OPEN position.

    Stock rows go through :func:`evaluate_exit` unchanged; option rows
    through :func:`evaluate_option_exit` with ``option_read`` — the
    ``(dte, current_mid)`` pair from :func:`_option_live_read` (computed by
    the caller so the chain is built once per row). Returns ``(decision,
    None)`` when evaluable, else ``(None, reason)`` — an honest explanation
    of WHY no evaluation was possible (§44 rule 18): no stored bars, a
    legacy stock row without a positive ``stop_distance``, or an option row
    without a positive entry premium.
    """
    if not bars:
        return None, (
            f"no stored bars for {position.ticker} — exit rules cannot be "
            "evaluated (DATA_ISSUE)"
        )
    is_option = is_option_position(position)
    if is_option and position.avg_price <= 0.0:
        return None, (
            "option position has no entry premium recorded — exit rules "
            "cannot be evaluated (DATA_ISSUE)"
        )
    if not is_option and position.stop_distance <= 0.0:
        return None, (
            "position has no stop_distance recorded — exit rules cannot be "
            "evaluated (DATA_ISSUE)"
        )
    entry = (
        date.fromisoformat(position.entry_bar_date)
        if position.entry_bar_date is not None
        else None
    )
    closes_since_entry = [b.close for b in bars if entry is None or b.ts >= entry]
    # Fold the current close into the peak BEFORE evaluating, exactly as the
    # backtest engine does (see PositionState docs).
    peak = max(closes_since_entry) if closes_since_entry else bars[-1].close
    # PositionState tracks the UNDERLYING (see its docs): for an option row
    # entry_price is the underlying close on the entry bar — avg_price is
    # the PREMIUM and belongs to OptionState.entry_premium instead.
    if is_option:
        underlying_entry = next(
            (b.close for b in bars if b.ts.isoformat() == position.entry_bar_date),
            closes_since_entry[0] if closes_since_entry else bars[-1].close,
        )
    else:
        underlying_entry = position.avg_price
    state = PositionState(
        entry_price=underlying_entry,
        stop_distance=position.stop_distance,  # ignored by the option engine
        entry_edge=position.entry_edge,
        bars_held=_bars_held(position, bars),
        highest_close_since_entry=max(peak, bars[-1].close),
    )
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    if is_option:
        dte, current_mid = option_read if option_read is not None else (None, None)
        decision = evaluate_option_exit(
            state,
            OptionState(
                entry_premium=position.avg_price,
                current_mid=current_mid,
                dte=dte,
            ),
            closes,
            highs,
            lows,
            volumes=volumes,
        )
    else:
        decision = evaluate_exit(state, closes, highs, lows, volumes=volumes)
    return decision, None


@router.get("")
async def list_positions(
    status: str = Query(default="OPEN"),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List positions with the full exit-engine read per OPEN row (§37).

    ``status`` filters OPEN (default) | CLOSED | ALL. Read-only — no audit.
    Every OPEN row's ``exit_reasons`` lists EVERY §11 rule with real numbers,
    "OK:"-prefixed when not firing (§37/§38) — including the option families
    (§11.3 premium hard stop, §11.7 DTE exit) for option rows; CLOSED rows
    carry ``realized_pnl`` and honest nulls elsewhere (§44 rule 18).

    Option rows: ``quantity`` = contracts, ``avg_price`` = entry premium per
    share, ``market_value`` = qty * current_mid * 100, ``max_loss`` = premium
    paid; ``current_price`` stays the UNDERLYING's last stored close (the
    underlying series drives the shared exit rules) while the option's own
    quote lives in ``contract.current_mid``. ``stop_price`` on an option row
    is the §11.3 PREMIUM stop per share (from the exit engine), not an
    underlying level.
    """
    status = status.upper()
    if status not in VALID_STATUS:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {', '.join(VALID_STATUS)}",
        )

    stmt = select(Position).order_by(Position.ticker, Position.id)
    if status != "ALL":
        stmt = stmt.where(Position.status == status)
    positions = (await session.execute(stmt)).scalars().all()

    out: list[dict] = []
    for pos in positions:
        is_option = is_option_position(pos)
        contract_out = None
        if pos.opt_expiry is not None:
            contract_out = {
                "expiry": pos.opt_expiry,
                "strike": pos.opt_strike,
                "right": pos.opt_right,
                "multiplier": pos.multiplier or 1,
                # Live fields; filled below for OPEN rows (honest nulls on
                # CLOSED rows — no live read, §44 rule 18).
                "dte": None,
                "current_mid": None,
                "premium_pnl_pct": None,
            }
        row = {
            "id": pos.id,
            "ticker": pos.ticker,
            "instrument": pos.instrument,
            "status": pos.status,
            "quantity": pos.quantity,
            "avg_price": pos.avg_price,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
            "current_price": None,
            "market_value": None,
            "unrealized_pnl": None,
            "unrealized_pnl_pct": None,
            "realized_pnl": pos.realized_pnl,
            "max_loss": pos.max_loss,
            # Stock: the fixed §11.3 underlying stop. Options: null here —
            # the premium stop is filled from the exit-engine read below.
            "stop_price": (
                pos.avg_price - pos.stop_distance
                if pos.stop_distance > 0 and not is_option
                else None
            ),
            "trail_price": None,
            "entry_edge": pos.entry_edge,
            "current_edge": None,
            "signal_decay": None,
            "bars_held": None,
            "time_stop_remaining": None,
            "exit_status": None,
            "exit_reasons": [],
            "contract": contract_out,
        }
        if pos.status == POSITION_OPEN:
            bars = await _stored_bars(session, pos.ticker)
            option_read = None
            if bars:
                price = bars[-1].close
                row["current_price"] = price
                row["bars_held"] = _bars_held(pos, bars)
                if is_option:
                    # Live option read via the shared chain helper (§9);
                    # market value carries the x100 multiplier (§12.1) and is
                    # an honest null when the contract has no current mid.
                    option_read = _option_live_read(pos, price)
                    dte, current_mid = option_read
                    mult = pos.multiplier or 1
                    if contract_out is not None:
                        contract_out["dte"] = dte
                        contract_out["current_mid"] = current_mid
                    if current_mid is not None:
                        row["market_value"] = pos.quantity * current_mid * mult
                        row["unrealized_pnl"] = (
                            (current_mid - pos.avg_price) * pos.quantity * mult
                        )
                        if pos.avg_price > 0:
                            pnl_pct = current_mid / pos.avg_price - 1.0
                            row["unrealized_pnl_pct"] = pnl_pct
                            if contract_out is not None:
                                contract_out["premium_pnl_pct"] = pnl_pct
                else:
                    row["market_value"] = pos.quantity * price
                    row["unrealized_pnl"] = (price - pos.avg_price) * pos.quantity
                    row["unrealized_pnl_pct"] = (
                        (price - pos.avg_price) / pos.avg_price
                        if pos.avg_price
                        else None
                    )
            decision, why_not = _evaluate_open_position(pos, bars, option_read)
            if decision is not None:
                row["stop_price"] = decision.stop_price
                row["trail_price"] = decision.trail_price
                row["current_edge"] = decision.current_edge
                row["signal_decay"] = (
                    pos.entry_edge - decision.current_edge
                    if decision.current_edge is not None
                    else None
                )
                row["time_stop_remaining"] = decision.time_stop_remaining
                row["exit_status"] = (
                    "EXIT_SIGNALED" if decision.should_exit else "HOLD"
                )
                row["exit_reasons"] = decision.reasons
            else:
                # §37: even when unevaluable, the user must see WHY.
                row["exit_reasons"] = [why_not]
        out.append(row)
    return out


@router.post("/check-exits")
async def check_exits(session: AsyncSession = Depends(get_session)) -> dict:
    """Evaluate + EXECUTE §11 exits for every OPEN position — one transaction.

    Covers BOTH instrument kinds: stock rows via ``evaluate_exit``, option
    rows via ``evaluate_option_exit`` (§11.3 premium hard stop / §11.7 DTE
    exit in front of the shared underlying rules). Each triggered exit
    writes EXIT_GENERATED (SYSTEM, with the rule and the full reason list)
    and runs the shared sell-to-close flow (``system_generated`` ->
    ORDER_REQUESTED actor SYSTEM) in the SAME transaction (rule 12) —
    option exits fill at the current chain mid, or at intrinsic value when
    the contract is missing from today's chain (documented in
    routers/orders.py). Positions that hold answer with their full
    "OK:"-prefixed reason list (§37). Mechanical exits are deliberately NOT
    blocked by the §18 kill switch: an exit reduces risk, and risk protection
    outranks the pause (§18 risk-priority). Runs under the paper-execution
    lock shared with approve/close, so a concurrent manual close can never
    double-sell the same position (§42 analogue).
    """
    async with execution_lock():
        return await _check_exits_locked(session)


async def _check_exits_locked(session: AsyncSession) -> dict:
    """The check-exits flow proper — caller holds the paper-execution lock."""
    positions = (
        (
            await session.execute(
                select(Position)
                .where(Position.status == POSITION_OPEN)
                .order_by(Position.ticker, Position.id)
            )
        )
        .scalars()
        .all()
    )

    exits_triggered: list[dict] = []
    held: list[dict] = []
    for pos in positions:
        bars = await _stored_bars(session, pos.ticker)
        option_read = None
        if is_option_position(pos) and bars:
            option_read = _option_live_read(pos, bars[-1].close)
        decision, why_not = _evaluate_open_position(pos, bars, option_read)
        if decision is None:
            # Unevaluable is NOT an exit — but the gap is surfaced (§44 r18).
            held.append({"ticker": pos.ticker, "reasons": [why_not]})
            continue
        if not decision.should_exit:
            held.append({"ticker": pos.ticker, "reasons": decision.reasons})
            continue

        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.EXIT_GENERATED,
            entity_type="position",
            entity_id=str(pos.id),
            details={
                "ticker": pos.ticker,
                "instrument": pos.instrument,
                "rule": decision.triggered_rule,
                "reasons": decision.reasons,
            },
        )
        if is_option_position(pos):
            # Same reference logic as /close: current chain mid, or the
            # documented intrinsic fallback when the contract is gone.
            _dte, current_mid = option_read if option_read is not None else (None, None)
            if current_mid is not None:
                reference, source = current_mid, "chain mid"
            else:
                reference = option_intrinsic_value(pos, bars[-1].close)
                source = "intrinsic (contract missing from today's chain)"
        else:
            reference, source = bars[-1].close, "last stored close"
        order, _realized = await execute_sell_to_close(
            session,
            pos,
            pos.quantity,  # mechanical exits always close in full (§11)
            reference,
            reason=f"exit engine: {decision.triggered_rule}",
            system_generated=True,
            reference_source=source,
        )
        exits_triggered.append(
            {
                "ticker": pos.ticker,
                "rule": decision.triggered_rule,
                "order_id": order.id,
            }
        )

    await session.commit()
    return {
        "checked": len(positions),
        "exits_triggered": exits_triggered,
        "held": held,
    }
