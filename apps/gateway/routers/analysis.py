"""Symbol analysis API (development plan §4.2, §6).

Serves `GET /api/watchlist/{ticker}/analysis`: stored daily bars plus the full
technical read — indicators, market-regime classification, and directional
scoring. Research surfaces are OPEN to any ticker (2026-08-20, §4.2 amended):
bars lazily backfill on first read; the system reference indices remain
used by the Market Regime Engine (ADR-005, see routers/market.py).

Bars are lazily backfilled from the configured MarketDataProvider on first
request; the bulk insert and its SYSTEM-attributed DATA_BACKFILL audit event
share one transaction (rule 12, ADR-003). All analytics come exclusively from
libs.trading_core (features + signals), so backtest and live share this exact
code (plan §21).
"""
import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import ProviderNotConfigured, get_provider
from libs.trading_core.features import atr, macd, realized_vol, rsi, sma
from libs.trading_core.models import (
    ActorType,
    AuditAction,
    DirectionalBias,
    MarketRegime,
    OpportunityStatus,
)
from libs.trading_core.signals import (
    DirectionalParams,
    DirectionalResult,
    RegimeParams,
    RegimeResult,
    classify_regime,
    edge_legend,
    score_direction,
)
from libs.trading_core.tradeability import assess_tradeability

from .. import audit
from ..db import (
    BacktestRecord,
    NewsArticleRow,
    Recommendation,
    StockBarDaily,
    WatchlistItem,
    get_session,
)
from ..deps import market_data_unavailable, require_market_data_provider

router = APIRouter(prefix="/api/watchlist", tags=["analysis"])

logger = logging.getLogger(__name__)

# Tunable parameters (plan §6.2: parameters, never hardcoded truths).
BACKFILL_DAYS = 600  # bars fetched on first request; > sma_slow(200) + warmup
SERIES_BARS = 250  # chart series length (~one trading year)
BARS_LIMIT_MIN = 10  # /bars: fewer is not a chart
BARS_LIMIT_MAX = 600  # /bars: capped at the backfill depth — no more exists
SMA_FAST = 20
SMA_MID = 50
SMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
REALIZED_VOL_PERIOD = 20


#: Bars are COMPLETE trading days only. A bar whose Eastern date is today may
#: still be forming while the market trades; storing it would freeze a
#: provisional number as history (and nothing ever rewrites a stored bar).
EASTERN = ZoneInfo("America/New_York")

#: How long to wait between refresh ATTEMPTS per symbol. A market holiday
#: looks exactly like a missing bar — the provider legitimately has nothing
#: newer — and without a throttle every request that day would re-ask.
REFRESH_ATTEMPT_SECONDS = 30 * 60

#: Bars fetched on a refresh: enough to cover any realistic gap (vacation,
#: paused deployment) with buffer; the insert dedupes against stored dates.
REFRESH_FETCH_DAYS = 15

# ticker -> last refresh attempt (UTC). In-process; a restart just retries.
_refresh_attempts: dict[str, datetime] = {}


def _complete_days_only(bars: list) -> list:
    """Drop any bar dated today-or-later in Eastern time (still forming)."""
    today_eastern = datetime.now(EASTERN).date()
    return [b for b in bars if b.ts < today_eastern]


def _last_expected_trading_date(today_eastern: date) -> date:
    """The most recent WEEKDAY strictly before today (Eastern).

    Holidays are not modeled: on one, this expects a bar the provider will
    not have, the refresh finds nothing new, and the attempt throttle keeps
    that harmless. Never a fabricated calendar — just Mon–Fri arithmetic.
    """
    d = today_eastern - timedelta(days=1)
    while d.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        d -= timedelta(days=1)
    return d


def _stale_trading_days(newest_bar: date, today_eastern: date) -> int:
    """WEEKDAYS the newest stored bar lags the last expected trading day.

    0 = current. Same Mon–Fri arithmetic (and the same unmodeled-holiday
    honesty) as :func:`_last_expected_trading_date`; the tradeability
    layer's ``max_stale_trading_days`` tolerance absorbs single holidays.
    """
    expected = _last_expected_trading_date(today_eastern)
    d, lag = newest_bar, 0
    while d < expected:
        d += timedelta(days=1)
        if d.weekday() < 5:
            lag += 1
    return lag


async def ensure_daily_bars(
    session: AsyncSession,
    ticker: str,
    provider_name: str,
    days: int = BACKFILL_DAYS,
) -> list[StockBarDaily]:
    """Return stored daily bars for `ticker` (oldest first), lazily backfilled
    and kept fresh.

    FIRST request for a symbol — no stored bars — bulk-inserts the configured
    provider's history with a SYSTEM DATA_BACKFILL audit event in the SAME
    transaction (rule 12, ADR-003). LATER requests serve the stored bars, and
    when the newest stored bar is older than the last expected trading day
    (weekday arithmetic, Eastern) the missing tail is fetched and APPENDED —
    same audit action with ``mode: "refresh"``. Without this, every signal,
    regime read and backtest would run forever on data frozen at first fetch,
    and the data-quality gate would eventually (correctly) refuse to trade on
    it. Refresh attempts are throttled per symbol (REFRESH_ATTEMPT_SECONDS)
    so a market holiday — indistinguishable from a missing bar — does not
    re-ask the provider on every request.

    COMPLETE DAYS ONLY: bars dated today (Eastern) are dropped before storing
    — an intraday daily bar is provisional, and stored bars are never
    rewritten. A refresh failure serves the stored bars rather than failing
    the request: yesterday's real close beats no answer, and the gap is
    logged loudly.

    Membership gating is the CALLER's responsibility, and since 2026-08-20
    (§4.2 amended) backtests.py is the ONLY caller that gates: research reads
    backfill for any symbol. First-fetch is deliberately unthrottled (the
    refresh throttle covers only the stored-bars branch), so storage/audit
    growth is driven by browsing breadth, not watchlist size.

    UNCONFIGURED PROVIDER: when `provider_name` is blank the registry raises
    :class:`ProviderNotConfigured` and this function re-raises it as the
    shared 503 ``MARKET_DATA_NOT_CONFIGURED`` — the SAME body every other
    market-data route returns. Because every backfill in the codebase funnels
    through here, that one translation covers the analysis, bars, options,
    backtest and order-gate paths at once: no bar is ever invented, and stored
    bars (real data already fetched) still serve normally.
    """
    rows = await session.execute(
        select(StockBarDaily)
        .where(StockBarDaily.ticker == ticker)
        .order_by(StockBarDaily.ts)
    )
    stored = list(rows.scalars().all())
    if stored:
        return await _refresh_if_stale(session, ticker, provider_name, stored)

    try:
        provider = get_provider(provider_name)
    except ProviderNotConfigured as exc:
        raise market_data_unavailable(exc) from exc
    # Fetch two extra: a provider series can include today's still-forming bar
    # AND run one date ahead of the Eastern trading day (a UTC-dated series
    # just after midnight UTC). Both get dropped as incomplete; the trim below
    # still stores exactly `days` COMPLETE days.
    fetched = _complete_days_only(provider.get_daily_bars(ticker, days + 2))[-days:]
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


async def _refresh_if_stale(
    session: AsyncSession,
    ticker: str,
    provider_name: str,
    stored: list[StockBarDaily],
) -> list[StockBarDaily]:
    """Append missing complete trading days to `stored`; serve stored on fail.

    Append-only by construction: only bars strictly NEWER than the newest
    stored date are inserted — a stored bar is real data already served and
    is never rewritten (§44 rule 18).
    """
    newest = stored[-1].ts
    today_eastern = datetime.now(EASTERN).date()
    if newest >= _last_expected_trading_date(today_eastern):
        return stored

    now = datetime.now(timezone.utc)
    last_attempt = _refresh_attempts.get(ticker)
    if (
        last_attempt is not None
        and (now - last_attempt).total_seconds() < REFRESH_ATTEMPT_SECONDS
    ):
        return stored  # holiday or recent failure — do not hammer the provider
    _refresh_attempts[ticker] = now

    try:
        provider = get_provider(provider_name)
        gap_days = max((today_eastern - newest).days + 5, REFRESH_FETCH_DAYS)
        fetched = _complete_days_only(provider.get_daily_bars(ticker, gap_days))
    except ProviderNotConfigured:
        # Stored real bars still serve (see docstring); nothing to fetch with.
        return stored
    except Exception:
        logger.exception("bar_refresh_failed", extra={"extra_fields": {"ticker": ticker}})
        return stored

    new_bars = [b for b in fetched if b.ts > newest]
    if not new_bars:
        return stored  # holiday / provider not yet updated — honest no-op

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
        for b in new_bars
    ]
    session.add_all(orm_bars)
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.DATA_BACKFILL,
        entity_type="stock_bars_daily",
        entity_id=ticker,
        details={
            "mode": "refresh",
            "bars": len(new_bars),
            "provider": provider_name,
            "first": new_bars[0].ts.isoformat(),
            "last": new_bars[-1].ts.isoformat(),
            "previous_newest": newest.isoformat(),
        },
    )
    await session.commit()
    return stored + orm_bars


# The Market Regime Engine reads the broad-market index (plan §6.1). SPY is a
# system reference symbol (ADR-005) — always maintained regardless of what
# anyone browses (the watchlist-only READ rule itself ended 2026-08-20).
REGIME_REFERENCE_SYMBOL = "SPY"


async def market_regime_from_spy(session: AsyncSession) -> RegimeResult:
    """Current broad-market regime from stored SPY daily bars (plan §6.1).

    Shared helper so the market overview, portfolio risk and order preview
    paths all see the SAME regime read. SPY bars lazily backfill via
    :func:`ensure_daily_bars` (ADR-005 exemption), so the first caller also
    writes the SYSTEM DATA_BACKFILL audit event; later calls are read-only.
    """
    settings = get_settings()
    bars = await ensure_daily_bars(
        session, REGIME_REFERENCE_SYMBOL, settings.market_data_provider
    )
    return classify_regime(
        [b.close for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
    )


# Regimes that agree with each directional bias (plan §31 v0 mapping).
_BULL_REGIMES = frozenset({MarketRegime.STRONG_BULL, MarketRegime.MILD_BULL})
_BEAR_REGIMES = frozenset({MarketRegime.STRONG_BEAR, MarketRegime.MILD_BEAR})


def _opportunity_status(
    bar_count: int,
    regime: MarketRegime,
    signal: DirectionalResult,
    regime_params: RegimeParams,
    directional_params: DirectionalParams,
) -> OpportunityStatus:
    """v0 opportunity-status mapping (plan §31), first match wins:

    1. DATA_ISSUE     — fewer stored bars than ``regime_params.sma_slow``: the
       slow trend structure (and hence the regime) is undefined.
    2. ENTRY_READY    — bias != NEUTRAL AND the regime agrees with the bias
       direction (BULL bias with STRONG/MILD_BULL; BEAR bias mirrored with
       STRONG/MILD_BEAR).
    3. SETUP_FORMING  — ``|directional_edge| >= bias_threshold / 2``: material
       one-sided evidence that has not (yet) aligned into an entry.
    4. WATCH          — some nonzero bull or bear score exists.
    5. NO_SIGNAL      — otherwise.

    All cutoffs come from the shared signal engines' parameter objects — no
    threshold is hardcoded here (plan §6.2).
    """
    if bar_count < regime_params.sma_slow:
        return OpportunityStatus.DATA_ISSUE
    if signal.bias is DirectionalBias.BULL and regime in _BULL_REGIMES:
        return OpportunityStatus.ENTRY_READY
    if signal.bias is DirectionalBias.BEAR and regime in _BEAR_REGIMES:
        return OpportunityStatus.ENTRY_READY
    if abs(signal.directional_edge) >= directional_params.bias_threshold / 2.0:
        return OpportunityStatus.SETUP_FORMING
    if signal.bull_score > 0.0 or signal.bear_score > 0.0:
        return OpportunityStatus.WATCH
    return OpportunityStatus.NO_SIGNAL


@router.get("/overview")
async def get_watchlist_overview(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """One dashboard row per Watchlist symbol (plan §31), ticker-ordered.

    Each row carries the last close, the current regime and directional read
    (computed by the SAME libs.trading_core.signals code the backtest replays,
    plan §21), the v0 opportunity status (see :func:`_opportunity_status`),
    and the symbol's latest backtest outcome: ``backtest_status`` is
    ``"NONE"`` (never run) | ``"COMPLETED"`` | ``"FAILED"`` with
    ``last_backtest_id`` pointing at that latest record (null when NONE).

    Bars are lazily backfilled per symbol via the shared helper, so a fresh
    symbol's first overview also writes its DATA_BACKFILL audit event.

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured: every column of this dashboard — last price, regime, scores,
    opportunity status — is market data or computed from it, so an
    unconfigured install shows no rows rather than rows of fiction.
    """
    require_market_data_provider()
    settings = get_settings()
    regime_params = RegimeParams()
    directional_params = DirectionalParams()

    rows = await session.execute(select(WatchlistItem).order_by(WatchlistItem.ticker))

    # One SPY read for all rows (§9 market-regime input); best-effort — a
    # fault degrades every row's tradeability honestly, never the endpoint.
    try:
        market_regime = (await market_regime_from_spy(session)).classification
    except Exception:
        logger.exception("market_regime_read_failed")
        market_regime = None

    today_eastern = datetime.now(EASTERN).date()
    overview: list[dict] = []
    for item in rows.scalars().all():
        bars = await ensure_daily_bars(session, item.ticker, settings.market_data_provider)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]

        regime = classify_regime(closes, highs, lows, params=regime_params)
        signal = score_direction(
            closes, highs, lows, volumes=volumes, params=directional_params
        )
        status = _opportunity_status(
            len(bars), regime.classification, signal, regime_params, directional_params
        )
        tradeability = assess_tradeability(
            bar_count=len(bars),
            stale_trading_days=_stale_trading_days(bars[-1].ts, today_eastern),
            market_regime=market_regime,
            symbol_regime=regime.classification,
            vol_regime=None,
            vol_unavailable_reason="ATM IV not evaluated in the overview",
        )

        latest = await session.execute(
            select(BacktestRecord)
            .where(BacktestRecord.ticker == item.ticker)
            .order_by(BacktestRecord.id.desc())
            .limit(1)
        )
        last_backtest = latest.scalars().first()

        overview.append(
            {
                "ticker": item.ticker,
                "price": closes[-1],
                "regime": regime.classification.value,
                "bull_score": signal.bull_score,
                "bear_score": signal.bear_score,
                "directional_edge": signal.directional_edge,
                "bias": signal.bias.value,
                "edge_class": signal.classification.value,
                "tradeability": tradeability.state.value,
                "opportunity_status": status.value,
                "backtest_status": (
                    last_backtest.status if last_backtest is not None else "NONE"
                ),
                "last_backtest_id": (
                    last_backtest.id if last_backtest is not None else None
                ),
            }
        )
    return overview


@router.get("/{ticker}/analysis")
async def get_symbol_analysis(
    ticker: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """Full technical analysis for one Watchlist symbol (plan §6).

    OPEN to any ticker (2026-08-20, §4.2 amended): bars lazily backfill on
    first request; watchlist membership means continuous tracking + backtest
    eligibility, not read access. 503 ``MARKET_DATA_NOT_CONFIGURED`` when no
    market data provider is configured — every number here is market data or
    computed from it, so there is nothing honest to serve (§44 rule 18).
    """
    require_market_data_provider()
    ticker = ticker.upper()
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

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

    # Layer 2 — tradeability (§9/§10): direction-agnostic environment verdict.
    # Market regime read is best-effort here: an SPY fetch fault must not take
    # the symbol analysis down, and a missing read honestly degrades the
    # verdict to DATA_INSUFFICIENT instead of pretending an answer.
    try:
        market_regime = (await market_regime_from_spy(session)).classification
    except Exception:
        logger.exception("market_regime_read_failed")
        market_regime = None
    tradeability = assess_tradeability(
        bar_count=len(bars),
        stale_trading_days=_stale_trading_days(
            dates[-1], datetime.now(EASTERN).date()
        ),
        market_regime=market_regime,
        symbol_regime=regime.classification,
        vol_regime=None,
        vol_unavailable_reason=(
            "ATM IV is not evaluated in this view — the volatility regime is "
            "classified on the Options/Trade Plan paths"
        ),
    )

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
        "tradeability": {
            "state": tradeability.state.value,
            "reasons": tradeability.reasons,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in tradeability.checks
            ],
            "version": tradeability.version,
        },
        "signal": {
            # Deterministic market-data calculation — no LLM is involved in
            # this score (upgrade §3). The UI labels the panel from this flag
            # rather than inventing its own provenance claim.
            "deterministic": True,
            "bull_score": signal.bull_score,
            "bear_score": signal.bear_score,
            "directional_edge": signal.directional_edge,
            "bias": signal.bias.value,
            "classification": signal.classification.value,
            "weights_version": signal.weights_version,
            "classification_version": signal.classification_version,
            # §8 threshold legend, derived from the classifier's own params —
            # displayed bands can never drift from the classification code.
            "edge_legend": edge_legend(),
            "components": [
                {
                    "name": c.name,
                    "side": c.side,
                    "triggered": c.triggered,
                    "weight": c.weight,
                    "contribution": c.contribution,
                    "max_contribution": c.max_contribution,
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


@router.get("/{ticker}/catalyst")
async def get_symbol_catalyst(
    ticker: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """LLM catalyst context for one Watchlist symbol (upgrade §11/§38 —
    Phase E). READ-ONLY over stored data: the latest stored LLM
    interpretation for the ticker plus the stored news articles citing it.
    Never calls the LLM (generation happens on the recommendations refresh
    path) and never mixes market-derived numbers in (§25: this payload is
    interpretation + citations, nothing else).

    Honest empties: ``llm: null`` when no interpretation was ever generated
    for the symbol; ``articles: []`` when no stored news cites it. OPEN to
    any ticker (2026-08-20, §4.2 amended).
    """
    ticker = ticker.upper()
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

    latest = (
        await session.execute(
            select(Recommendation)
            .where(Recommendation.ticker == ticker)
            .order_by(Recommendation.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    articles = (
        (
            await session.execute(
                select(NewsArticleRow)
                .order_by(NewsArticleRow.published_at.desc())
                .limit(200)
            )
        )
        .scalars()
        .all()
    )
    cited = [a for a in articles if ticker in (a.tickers or [])][:10]

    latest_source: str | None = None
    published = [a.published_at for a in cited]
    if latest is not None:
        for ev in latest.evidence or []:
            ts = ev.get("published_at")
            if ts:
                try:
                    published.append(datetime.fromisoformat(ts))
                except ValueError:
                    pass
    if published:
        latest_source = max(published).isoformat()

    return {
        "ticker": ticker,
        # §25/§11: this block is INTERPRETIVE, generated by the LLM from the
        # cited articles — the UI labels it LLM-GENERATED off this flag and
        # must never style it as market data.
        "generated": True,
        "llm": (
            {
                "generated_at": latest.ts.isoformat(),
                "model": latest.llm_model,  # "" = pre-upgrade row, unknown
                "status": latest.status,
                "sentiment": latest.sentiment,
                "impact": latest.impact,
                "novelty": latest.novelty,
                "source_reliability": latest.source_reliability,
                "horizon": latest.horizon,
                "catalyst_type": latest.catalyst_type,
                "reason_codes": latest.reason_codes,
                "summary": latest.summary,
                "evidence": latest.evidence,
            }
            if latest is not None
            else None
        ),
        "articles": [
            {
                "title": a.title,
                "publisher": a.publisher,
                "published_at": a.published_at.isoformat(),
                "url": a.url,
            }
            for a in cited
        ],
        # §38 freshness: the newest source timestamp across citations and
        # stored articles — never implied to be live market data.
        "latest_source_published_at": latest_source,
    }


@router.get("/{ticker}/bars")
async def get_symbol_bars(
    ticker: str,
    limit: int = Query(default=SERIES_BARS, ge=BARS_LIMIT_MIN, le=BARS_LIMIT_MAX),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Raw daily OHLCV bars for one Watchlist symbol (plan §33, Price tab).

    Returns the most recent `limit` stored bars, oldest first. OPEN to any
    ticker (2026-08-20, §4.2 amended) — shares the analysis lazy-backfill path, so
    the first request for a fresh symbol writes its one DATA_BACKFILL audit
    event and later requests are read-only. 503
    ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is configured:
    bars are raw market data, never synthesized.
    """
    require_market_data_provider()
    ticker = ticker.upper()
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

    settings = get_settings()
    bars = await ensure_daily_bars(session, ticker, settings.market_data_provider)
    return {
        "ticker": ticker,
        "source": settings.market_data_provider,
        "bars": [
            {
                "date": b.ts.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars[-limit:]
        ],
    }
