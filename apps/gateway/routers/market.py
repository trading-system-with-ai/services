"""Market overview API (development plan §22.1, §6.1).

Quotes come from the configured MarketDataProvider (stub until the MASSIVE
integration lands). The market regime is COMPUTED by the Market Regime Engine
(libs.trading_core.signals.classify_regime, plan §6.1) from stored SPY daily
bars — no placeholder.

SPY/QQQ/VIX are system reference symbols: exempt from the watchlist-only
historical data rule (plan §4.2) because the Market Regime Engine requires
index data regardless of what the user watches (plan §6.1) — see ADR-005.
Their bars use the same lazy backfill path as watchlist analysis, so the first
overview request writes a SYSTEM-attributed DATA_BACKFILL audit event in the
same transaction as the inserted bars (rule 12); every later request is
read-only.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.signals import classify_regime

from ..db import get_session
from .analysis import ensure_daily_bars

router = APIRouter(prefix="/api/market", tags=["market"])

# Headline dashboard indices: SPY/QQQ for direction, VIX for volatility.
# System reference symbols, exempt from the watchlist-only data rule (ADR-005).
INDEX_SYMBOLS = ["SPY", "QQQ", "VIX"]

# The regime engine reads the broad-market index (plan §6.1).
REGIME_SYMBOL = "SPY"


@router.get("/overview")
async def market_overview(session: AsyncSession = Depends(get_session)) -> dict:
    settings = get_settings()
    provider = get_provider(settings.market_data_provider)
    quotes = provider.get_quotes(INDEX_SYMBOLS)

    # Real market regime from SPY daily bars (plan §6.1; ADR-005 exemption).
    bars = await ensure_daily_bars(session, REGIME_SYMBOL, settings.market_data_provider)
    regime = classify_regime(
        [b.close for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
    )

    return {
        "provider": settings.market_data_provider,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": False,
        "market_regime": regime.classification.value,
        "indices": [
            {
                "symbol": q.symbol,
                "price": q.price,
                "change_pct": q.change_pct,
                "ts": q.ts.isoformat(),
            }
            for q in quotes
        ],
    }
