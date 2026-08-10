"""Symbol analysis API (development plan §4.2, §6).

Serves `GET /api/watchlist/{ticker}/analysis`: stored daily bars plus the full
technical read — indicators, market-regime classification, and directional
scoring. Historical data may exist ONLY for Watchlist symbols (plan §4.2), so
non-watchlist tickers 404; the sole exception is the system reference indices
used by the Market Regime Engine (ADR-005, see routers/market.py).

Bars are lazily backfilled from the configured MarketDataProvider on first
request; the bulk insert and its SYSTEM-attributed DATA_BACKFILL audit event
share one transaction (rule 12, ADR-003). All analytics come exclusively from
libs.trading_core (features + signals), so backtest and live share this exact
code (plan §21).
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.features import atr, macd, realized_vol, rsi, sma
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.signals import classify_regime, score_direction

from .. import audit
from ..db import StockBarDaily, WatchlistItem, get_session

router = APIRouter(prefix="/api/watchlist", tags=["analysis"])

# Tunable parameters (plan §6.2: parameters, never hardcoded truths).
BACKFILL_DAYS = 600  # bars fetched on first request; > sma_slow(200) + warmup
SERIES_BARS = 250  # chart series length (~one trading year)
SMA_FAST = 20
SMA_MID = 50
SMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
REALIZED_VOL_PERIOD = 20


async def ensure_daily_bars(
    session: AsyncSession,
    ticker: str,
    provider_name: str,
    days: int = BACKFILL_DAYS,
) -> list[StockBarDaily]:
    """Return stored daily bars for `ticker` (oldest first), lazily backfilling.

    On first request for a symbol — no stored bars — the configured provider's
    `get_daily_bars` history is bulk-inserted together with a SYSTEM-attributed
    DATA_BACKFILL audit event in the SAME transaction (rule 12, ADR-003), then
    committed. Subsequent calls read the stored bars and write nothing, so a
    symbol is backfilled exactly once.

    Watchlist gating (plan §4.2) is the CALLER's responsibility — the market
    overview path uses this same function for the exempt system reference
    symbols (ADR-005).
    """
    rows = await session.execute(
        select(StockBarDaily)
        .where(StockBarDaily.ticker == ticker)
        .order_by(StockBarDaily.ts)
    )
    stored = list(rows.scalars().all())
    if stored:
        return stored

    provider = get_provider(provider_name)
    fetched = provider.get_daily_bars(ticker, days)
    if not fetched:
        raise HTTPException(
            status_code=502, detail=f"provider {provider_name!r} returned no bars for {ticker}"
        )
    orm_bars = [
        StockBarDaily(
            ticker=ticker,
            ts=b.ts,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
        )
        for b in fetched
    ]
    session.add_all(orm_bars)
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.DATA_BACKFILL,
        entity_type="stock_bars_daily",
        entity_id=ticker,
        details={
            "bars": len(fetched),
            "provider": provider_name,
            "first": fetched[0].ts.isoformat(),
            "last": fetched[-1].ts.isoformat(),
        },
    )
    await session.commit()
    return orm_bars


@router.get("/{ticker}/analysis")
async def get_symbol_analysis(
    ticker: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """Full technical analysis for one Watchlist symbol (plan §6).

    404s for tickers not on the Watchlist: historical data may exist only for
    Watchlist symbols (plan §4.2).
    """
    ticker = ticker.upper()
    row = await session.execute(
        select(WatchlistItem).where(WatchlistItem.ticker == ticker)
    )
    if row.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{ticker} is not on the watchlist; historical data exists "
                "only for Watchlist symbols"
            ),
        )

    settings = get_settings()
    bars = await ensure_daily_bars(session, ticker, settings.market_data_provider)

    dates = [b.ts for b in bars]
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    last_close = closes[-1]

    # Indicators (libs.trading_core.features — shared by backtest and live, plan §21).
    sma_fast_series = sma(closes, SMA_FAST)
    sma_mid_series = sma(closes, SMA_MID)
    atr_last = atr(highs, lows, closes, period=ATR_PERIOD)[-1]
    macd_res = macd(closes, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)

    regime = classify_regime(closes, highs, lows)
    signal = score_direction(closes, highs, lows, volumes=volumes)

    tail = bars[-SERIES_BARS:]
    offset = len(bars) - len(tail)

    return {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": settings.market_data_provider,
        "bars": {
            "count": len(bars),
            "first": dates[0].isoformat(),
            "last": dates[-1].isoformat(),
        },
        "price": last_close,
        "indicators": {
            "sma20": sma_fast_series[-1],
            "sma50": sma_mid_series[-1],
            "sma200": sma(closes, SMA_SLOW)[-1],
            "rsi14": rsi(closes, period=RSI_PERIOD)[-1],
            "atr14": atr_last,
            "atr_pct": atr_last / last_close if atr_last is not None else None,
            "macd": macd_res["macd"][-1],
            "macd_signal": macd_res["signal"][-1],
            "macd_histogram": macd_res["histogram"][-1],
            "realized_vol20": realized_vol(closes, period=REALIZED_VOL_PERIOD)[-1],
        },
        "regime": {
            "classification": regime.classification.value,
            "features": regime.features,
        },
        "signal": {
            "bull_score": signal.bull_score,
            "bear_score": signal.bear_score,
            "directional_edge": signal.directional_edge,
            "bias": signal.bias.value,
            "components": [
                {
                    "name": c.name,
                    "side": c.side,
                    "triggered": c.triggered,
                    "weight": c.weight,
                    "detail": c.detail,
                }
                for c in signal.components
            ],
        },
        "series": {
            "dates": [b.ts.isoformat() for b in tail],
            "close": [b.close for b in tail],
            "sma20": sma_fast_series[offset:],
            "sma50": sma_mid_series[offset:],
        },
    }
