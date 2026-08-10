"""Watchlist API.

Authorization rule (development plan §4.2, rule 5): only USER actions may
add/remove Watchlist symbols. There is deliberately no code path for SYSTEM
or LLM actors to call these endpoints' service logic.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import TradingPoolItem, WatchlistItem, get_session
from ..schemas import TickerRequest, WatchlistItemOut

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

# Single-user V1: a fixed user identity until auth-service lands.
CURRENT_USER = "local-user"


@router.get("", response_model=list[WatchlistItemOut])
async def list_watchlist(session: AsyncSession = Depends(get_session)):
    rows = await session.execute(select(WatchlistItem).order_by(WatchlistItem.ticker))
    return rows.scalars().all()


async def add_ticker_to_watchlist(
    session: AsyncSession, ticker: str, note: str = ""
) -> WatchlistItem:
    """Shared USER watchlist-insertion semantics (plan §4.2, rule 5).

    The single code path by which a ticker enters the Watchlist: used by both
    POST /api/watchlist and the recommendation promote endpoint, so the two
    can never diverge. Raises 409 when the ticker is already listed; audits
    WATCHLIST_ADD as ActorType.USER in the caller's transaction. Flushes,
    never commits — the caller controls the transaction so the insertion can
    be grouped with its own state changes + audit events (rule 12).
    """
    existing = await session.execute(select(WatchlistItem).where(WatchlistItem.ticker == ticker))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"{ticker} already on watchlist")

    item = WatchlistItem(ticker=ticker, added_by=CURRENT_USER, note=note)
    session.add(item)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.WATCHLIST_ADD,
        entity_type="watchlist",
        entity_id=ticker,
        details={"note": note},
    )
    await session.flush()
    return item


@router.post("", response_model=WatchlistItemOut, status_code=201)
async def add_to_watchlist(req: TickerRequest, session: AsyncSession = Depends(get_session)):
    item = await add_ticker_to_watchlist(session, req.ticker, req.note)
    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/{ticker}", status_code=204)
async def remove_from_watchlist(ticker: str, session: AsyncSession = Depends(get_session)):
    ticker = ticker.upper()
    row = await session.execute(select(WatchlistItem).where(WatchlistItem.ticker == ticker))
    item = row.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail=f"{ticker} not on watchlist")

    # Removing from Watchlist also revokes Trading Pool membership: a symbol
    # can never be authorized to trade while outside the Watchlist.
    pool_row = await session.execute(select(TradingPoolItem).where(TradingPoolItem.ticker == ticker))
    pool_item = pool_row.scalar_one_or_none()
    if pool_item is not None:
        await session.delete(pool_item)
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.TRADING_POOL_REMOVE,
            entity_type="trading_pool",
            entity_id=ticker,
            details={"reason": "cascade: removed from watchlist"},
        )

    await session.delete(item)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.WATCHLIST_REMOVE,
        entity_type="watchlist",
        entity_id=ticker,
    )
    await session.commit()
