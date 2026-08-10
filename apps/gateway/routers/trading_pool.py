"""Trading Pool API.

Hard rules enforced here (development plan §4.3, rules 6 & 18):
- only symbols currently on the Watchlist may be promoted;
- promotion never means immediate purchase — trading_enabled starts False;
- only USER actions may promote/remove/toggle.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction, InstrumentType

from .. import audit
from ..db import TradingPoolItem, WatchlistItem, get_session
from ..schemas import TradingPoolAddRequest, TradingPoolItemOut, TradingToggleRequest

router = APIRouter(prefix="/api/trading-pool", tags=["trading-pool"])

CURRENT_USER = "local-user"

# V1 account constraints: long-only. Anything else is rejected at the boundary.
PERMITTED_STRATEGIES = {
    InstrumentType.LONG_STOCK.value,
    InstrumentType.LONG_CALL.value,
    InstrumentType.LONG_PUT.value,
}


@router.get("", response_model=list[TradingPoolItemOut])
async def list_trading_pool(session: AsyncSession = Depends(get_session)):
    rows = await session.execute(select(TradingPoolItem).order_by(TradingPoolItem.ticker))
    return rows.scalars().all()


@router.post("", response_model=TradingPoolItemOut, status_code=201)
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

    item = TradingPoolItem(
        ticker=req.ticker,
        trading_enabled=False,  # promotion is authorization, not an order
        allowed_strategies=req.allowed_strategies,
        promoted_by=CURRENT_USER,
    )
    session.add(item)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_POOL_ADD,
        entity_type="trading_pool",
        entity_id=req.ticker,
        details={"allowed_strategies": req.allowed_strategies},
    )
    await session.commit()
    await session.refresh(item)
    return item


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
