"""Market overview API (development plan §22.1).

Read-only: serves quotes from the configured MarketDataProvider (stub until
the MASSIVE integration lands), so it writes no audit events. The regime
value is a fixed placeholder until the regime engine (plan §7) computes it.
"""
from datetime import datetime, timezone

from fastapi import APIRouter

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.models import MarketRegime

router = APIRouter(prefix="/api/market", tags=["market"])

# Headline dashboard indices: SPY/QQQ for direction, VIX for volatility.
INDEX_SYMBOLS = ["SPY", "QQQ", "VIX"]

# Placeholder until the regime engine derives this from market data (plan §7).
PLACEHOLDER_REGIME = MarketRegime.NEUTRAL_RANGE


@router.get("/overview")
async def market_overview() -> dict:
    settings = get_settings()
    provider = get_provider(settings.market_data_provider)
    quotes = provider.get_quotes(INDEX_SYMBOLS)
    return {
        "provider": settings.market_data_provider,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": False,
        "market_regime": PLACEHOLDER_REGIME.value,
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
