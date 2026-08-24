"""Trading Pool API.

Hard rules enforced here (development plan §4.3, rules 6 & 18):
- only symbols currently on the Watchlist may be promoted;
- promotion never means immediate purchase — trading_enabled starts False;
- only USER actions may promote/remove/toggle.

Promotion readiness checks (§4.3): before a symbol enters the pool it is
evaluated against MIN_HISTORY / BACKTEST_COMPLETED / BACKTEST_TRADES / LIQUIDITY.
Any failure blocks with 422 UNLESS the user explicitly acknowledges the risk
characteristics (``acknowledge_risks``) — and either way the full check
results and the acknowledged flag land in the TRADING_POOL_ADD audit details,
so an acknowledged override stays permanently visible in the audit trail
(§4.3, §38).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction, InstrumentType
from libs.trading_core.risk.liquidity import (
    LiquidityLimits,
    evaluate_underlying_liquidity,
    liquidity_report_detail,
)
from libs.trading_core.signals import RegimeParams

from .. import audit
from ..db import (
    BacktestRecord,
    StockBarDaily,
    TradingPoolItem,
    WatchlistItem,
    get_session,
)
from ..schemas import (
    TradingPoolAddRequest,
    TradingPoolItemOut,
    TradingPoolPromotedOut,
    TradingToggleRequest,
)
from .backtests import STATUS_COMPLETED

router = APIRouter(prefix="/api/trading-pool", tags=["trading-pool"])

CURRENT_USER = "local-user"

# V1 account constraints: long-only. Anything else is rejected at the boundary.
PERMITTED_STRATEGIES = {
    InstrumentType.LONG_STOCK.value,
    InstrumentType.LONG_CALL.value,
    InstrumentType.LONG_PUT.value,
}

# §4.3 MIN_HISTORY threshold: enough stored daily bars for the slowest signal
# the engines run (the 200-bar SMA) — taken from the REAL parameter object,
# never a duplicated constant (plan §6.2).
MIN_HISTORY_BARS = RegimeParams().sma_slow

# §4.3 LIQUIDITY (risk-engine audit §7.3 / §10 B0): the SAME pure evaluation
# the §10 gate chain runs — ADV20 from the stored daily volumes; no order size
# (readiness is not an order) and no live quote here — in REPORT mode:
# research defaults, UNVALIDATED, so the check still passes and the detail
# carries the measured numbers + the hypothetical verdict until promoted.
LIQUIDITY_LIMITS = LiquidityLimits()


async def promotion_checks(session: AsyncSession, ticker: str) -> list[dict]:
    """Run the §4.3 promotion readiness checks for ``ticker``, in order.

    Returns one ``{"name", "passed", "detail"}`` dict per check, with honest
    numeric details (§44 rule 18):

    - MIN_HISTORY: stored daily bars >= ``RegimeParams.sma_slow`` (200);
    - BACKTEST_COMPLETED: at least one COMPLETED backtest exists;
    - BACKTEST_TRADES: the LATEST COMPLETED backtest produced >= 1 closed
      trade — 0 trades means the strategy has no trade evidence at all;
    - LIQUIDITY: REPORT mode (audit §7.3 / B0) — ADV20 measured from the
      stored daily volumes via the shared pure evaluator; passes regardless
      of the hypothetical verdict, which the detail states with the numbers.
    """
    checks: list[dict] = []

    bar_count = (
        await session.execute(
            select(func.count())
            .select_from(StockBarDaily)
            .where(StockBarDaily.ticker == ticker)
        )
    ).scalar_one()
    checks.append(
        {
            "name": "MIN_HISTORY",
            "passed": bar_count >= MIN_HISTORY_BARS,
            "detail": (
                f"{bar_count} stored daily bars; promotion requires >= "
                f"{MIN_HISTORY_BARS} (RegimeParams.sma_slow)"
            ),
        }
    )

    latest = (
        await session.execute(
            select(BacktestRecord)
            .where(
                BacktestRecord.ticker == ticker,
                BacktestRecord.status == STATUS_COMPLETED,
            )
            .order_by(BacktestRecord.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    checks.append(
        {
            "name": "BACKTEST_COMPLETED",
            "passed": latest is not None,
            "detail": (
                f"latest COMPLETED backtest id {latest.id}"
                if latest is not None
                else f"no COMPLETED backtest exists for {ticker}"
            ),
        }
    )

    if latest is not None:
        # IS/OOS segmentation removed 2026-08-16 (user decision): the evidence
        # bar is now >= 1 closed trade over the whole tested period. Legacy
        # rows (pre-removal) stored {"full", "in_sample", "out_of_sample"} —
        # read their full segment; new rows store the flat dict.
        stored = latest.metrics or {}
        flat = stored.get("full", stored) if isinstance(stored, dict) else {}
        trade_count = flat.get("num_trades", 0)
        checks.append(
            {
                "name": "BACKTEST_TRADES",
                "passed": trade_count >= 1,
                "detail": (
                    f"latest COMPLETED backtest (id {latest.id}) has "
                    f"{trade_count} closed trade(s)"
                    + ("" if trade_count >= 1 else " — no trade evidence")
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "BACKTEST_TRADES",
                "passed": False,
                "detail": (
                    "0 trades — no COMPLETED backtest to evaluate, so no "
                    "trade evidence"
                ),
            }
        )

    # LIQUIDITY (REPORT mode): the last `adv_window` stored volumes, oldest
    # first, through the shared evaluator; fewer bars -> honest UNAVAILABLE.
    volume_rows = (
        await session.execute(
            select(StockBarDaily.volume)
            .where(StockBarDaily.ticker == ticker)
            .order_by(StockBarDaily.ts.desc())
            .limit(LIQUIDITY_LIMITS.adv_window)
        )
    ).scalars().all()
    liquidity = evaluate_underlying_liquidity(
        list(reversed(volume_rows)), None, None, None, LIQUIDITY_LIMITS
    )
    checks.append(
        {
            "name": "LIQUIDITY",
            "passed": True,  # REPORT mode: never blocks until promoted (Q3)
            "detail": (
                liquidity_report_detail(liquidity, LIQUIDITY_LIMITS)
                + " (readiness check: no order size, no live quote)"
            ),
        }
    )
    return checks


@router.get("", response_model=list[TradingPoolItemOut])
async def list_trading_pool(session: AsyncSession = Depends(get_session)):
    rows = await session.execute(select(TradingPoolItem).order_by(TradingPoolItem.ticker))
    return rows.scalars().all()


@router.post("", response_model=TradingPoolPromotedOut, status_code=201)
async def promote_to_trading_pool(
    req: TradingPoolAddRequest, session: AsyncSession = Depends(get_session)
):
    watch = await session.execute(select(WatchlistItem).where(WatchlistItem.ticker == req.ticker))
    if watch.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=422,
            detail=f"{req.ticker} is not on the Watchlist; only Watchlist symbols may be promoted",
        )

    existing = await session.execute(
        select(TradingPoolItem).where(TradingPoolItem.ticker == req.ticker)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"{req.ticker} already in trading pool")

    bad = set(req.allowed_strategies) - PERMITTED_STRATEGIES
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"strategies not permitted by account constraints: {sorted(bad)}",
        )

    # §4.3 promotion readiness checks: any failure blocks unless the user
    # explicitly acknowledges the risk characteristics.
    checks = await promotion_checks(session, req.ticker)
    if not all(c["passed"] for c in checks) and not req.acknowledge_risks:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "promotion checks failed — review and acknowledge to proceed"
                ),
                "checks": checks,
            },
        )

    item = TradingPoolItem(
        ticker=req.ticker,
        trading_enabled=False,  # promotion is authorization, not an order
        allowed_strategies=req.allowed_strategies,
        promoted_by=CURRENT_USER,
    )
    session.add(item)
    # The audit details ALWAYS carry the full check results and the
    # acknowledged flag — an acknowledged override of failed checks is
    # permanently visible in the audit trail (§4.3, §38).
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_POOL_ADD,
        entity_type="trading_pool",
        entity_id=req.ticker,
        details={
            "allowed_strategies": req.allowed_strategies,
            "promotion_checks": checks,
            "risks_acknowledged": req.acknowledge_risks,
        },
    )
    await session.commit()
    await session.refresh(item)
    return {
        "ticker": item.ticker,
        "trading_enabled": item.trading_enabled,
        "allowed_strategies": item.allowed_strategies,
        "promoted_by": item.promoted_by,
        "created_at": item.created_at,
        "promotion_checks": checks,
        "risks_acknowledged": req.acknowledge_risks,
    }


@router.post("/{ticker}/trading", response_model=TradingPoolItemOut)
async def toggle_trading(
    ticker: str, req: TradingToggleRequest, session: AsyncSession = Depends(get_session)
):
    ticker = ticker.upper()
    row = await session.execute(select(TradingPoolItem).where(TradingPoolItem.ticker == ticker))
    item = row.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail=f"{ticker} not in trading pool")

    item.trading_enabled = req.enabled
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_POOL_TOGGLE,
        entity_type="trading_pool",
        entity_id=ticker,
        details={"trading_enabled": req.enabled},
    )
    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/{ticker}", status_code=204)
async def remove_from_trading_pool(ticker: str, session: AsyncSession = Depends(get_session)):
    ticker = ticker.upper()
    row = await session.execute(select(TradingPoolItem).where(TradingPoolItem.ticker == ticker))
    item = row.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail=f"{ticker} not in trading pool")

    await session.delete(item)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_POOL_REMOVE,
        entity_type="trading_pool",
        entity_id=ticker,
    )
    await session.commit()
