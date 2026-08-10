"""Position monitor API (development plan §11, §37, §38).

``GET /api/positions`` lists paper positions with their live exit-engine read:
every OPEN row is evaluated by :func:`libs.trading_core.exits.evaluate_exit`
— the SAME engine the backtest validated (plan §21), never reimplemented here
— and carries the full per-rule reason list, "OK:"-prefixed for rules that
did NOT fire, so the user can always see why the system is still holding
(§37). CLOSED rows carry ``realized_pnl`` and honest nulls elsewhere (§44
rule 18). Read-only: no audit events (rule 12 covers state changes and
decisions).

``POST /api/positions/check-exits`` runs the same evaluation for every OPEN
position and EXECUTES the triggered exits: each fires an EXIT_GENERATED audit
event (SYSTEM) and the shared sell-to-close flow (ORDER_REQUESTED actor
SYSTEM) in the same transaction. Mechanical exits are NOT blocked by the §18
kill switch: exits reduce risk, and risk protection outranks the pause (§18
risk-priority).
"""
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.exits import ExitDecision, PositionState, evaluate_exit
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import Position, StockBarDaily, get_session
from .orders import POSITION_OPEN, execute_sell_to_close, execution_lock

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


def _evaluate_open_position(
    position: Position, bars: list[StockBarDaily]
) -> tuple[ExitDecision | None, str | None]:
    """Run the shared exit engine for one OPEN position.

    Returns ``(decision, None)`` when evaluable, else ``(None, reason)`` —
    an honest explanation of WHY no evaluation was possible (§44 rule 18):
    no stored bars, or a legacy row without a positive ``stop_distance``.
    """
    if not bars:
        return None, (
            f"no stored bars for {position.ticker} — exit rules cannot be "
            "evaluated (DATA_ISSUE)"
        )
    if position.stop_distance <= 0.0:
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
    state = PositionState(
        entry_price=position.avg_price,
        stop_distance=position.stop_distance,
        entry_edge=position.entry_edge,
        bars_held=_bars_held(position, bars),
        highest_close_since_entry=max(peak, bars[-1].close),
    )
    decision = evaluate_exit(
        state,
        [b.close for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
        volumes=[b.volume for b in bars],
    )
    return decision, None


@router.get("")
async def list_positions(
    status: str = Query(default="OPEN"),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List positions with the full exit-engine read per OPEN row (§37).

    ``status`` filters OPEN (default) | CLOSED | ALL. Read-only — no audit.
    Every OPEN row's ``exit_reasons`` lists EVERY §11 rule with real numbers,
    "OK:"-prefixed when not firing (§37/§38); CLOSED rows carry
    ``realized_pnl`` and honest nulls elsewhere (§44 rule 18).
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
        row = {
            "id": pos.id,
            "ticker": pos.ticker,
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
            "stop_price": (
                pos.avg_price - pos.stop_distance if pos.stop_distance > 0 else None
            ),
            "trail_price": None,
            "entry_edge": pos.entry_edge,
            "current_edge": None,
            "signal_decay": None,
            "bars_held": None,
            "time_stop_remaining": None,
            "exit_status": None,
            "exit_reasons": [],
        }
        if pos.status == POSITION_OPEN:
            bars = await _stored_bars(session, pos.ticker)
            if bars:
                price = bars[-1].close
                row["current_price"] = price
                row["market_value"] = pos.quantity * price
                row["unrealized_pnl"] = (price - pos.avg_price) * pos.quantity
                row["unrealized_pnl_pct"] = (
                    (price - pos.avg_price) / pos.avg_price if pos.avg_price else None
                )
                row["bars_held"] = _bars_held(pos, bars)
            decision, why_not = _evaluate_open_position(pos, bars)
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

    Each triggered exit writes EXIT_GENERATED (SYSTEM, with the rule and the
    full reason list) and runs the shared sell-to-close flow
    (``system_generated`` -> ORDER_REQUESTED actor SYSTEM) in the SAME
    transaction (rule 12). Positions that hold answer with their full
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
        decision, why_not = _evaluate_open_position(pos, bars)
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
                "rule": decision.triggered_rule,
                "reasons": decision.reasons,
            },
        )
        order, _realized = await execute_sell_to_close(
            session,
            pos,
            pos.quantity,  # mechanical exits always close in full (§11)
            bars[-1].close,
            reason=f"exit engine: {decision.triggered_rule}",
            system_generated=True,
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
