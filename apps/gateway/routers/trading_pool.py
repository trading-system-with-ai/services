"""Trading Pool API.

Hard rules enforced here (development plan §4.3, rules 6 & 18):
- only symbols currently on the Watchlist may be promoted;
- promotion never means immediate purchase — trading_enabled starts False;
- only USER actions may promote/remove/toggle.

Promotion readiness checks (§4.3): before a symbol enters the pool it is
evaluated against MIN_HISTORY / BACKTEST_COMPLETED / OOS_STATS / LIQUIDITY.
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

# §4.3 LIQUIDITY: a documented placeholder until real market data exists.
LIQUIDITY_STUB_DETAIL = (
    "stub data — real liquidity checks arrive with the Massive integration"
)


async def promotion_checks(session: AsyncSession, ticker: str) -> list[dict]:
    """Run the §4.3 promotion readiness checks for ``ticker``, in order.

    Returns one ``{"name", "passed", "detail"}`` dict per check, with honest
    numeric details (§44 rule 18):

    - MIN_HISTORY: stored daily bars >= ``RegimeParams.sma_slow`` (200);
    - BACKTEST_COMPLETED: at least one COMPLETED backtest exists;
    - OOS_STATS: the LATEST COMPLETED backtest produced >= 1 out-of-sample
      trade — 0 trades means the strategy has NO out-of-sample evidence;
    - LIQUIDITY: always passes at V1 — a documented stub until real market
      data (Massive) arrives (§4.3).
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
        oos_trades = (latest.metrics or {}).get("out_of_sample", {}).get(
            "num_trades", 0
        )
        checks.append(
            {
                "name": "OOS_STATS",
                "passed": oos_trades >= 1,
                "detail": (
                    f"latest COMPLETED backtest (id {latest.id}) has "
                    f"{oos_trades} out-of-sample trade(s)"
                    + ("" if oos_trades >= 1 else " — no out-of-sample evidence")
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "OOS_STATS",
                "passed": False,
                "detail": (
                    "0 out-of-sample trades — no COMPLETED backtest to "
                    "evaluate, so no out-of-sample evidence"
                ),
            }
        )

    checks.append(
        {"name": "LIQUIDITY", "passed": True, "detail": LIQUIDITY_STUB_DETAIL}
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
