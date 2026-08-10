"""Market Regime Engine v0 (development plan §6.1).

Pure, deterministic classification of the current market regime from daily
OHLC series — no DB, no FastAPI — shared verbatim by the backtest and live
paths (plan §21). All indicator math comes exclusively from
:mod:`libs.trading_core.features`; every threshold and period is a parameter
on :class:`RegimeParams`, never a hardcoded truth (plan §6.2).

Posture: ``TRANSITION`` defaults to NO TRADE (plan §6.1). Whenever the engine
cannot confidently classify — insufficient history, or a volatility
dislocation — it answers ``TRANSITION`` so downstream strategy/risk layers
stand down instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from libs.trading_core.features import atr, sma
from libs.trading_core.models import MarketRegime


@dataclass(frozen=True)
class RegimeParams:
    """Backtest parameters for the regime classifier (plan §6.1, §6.2).

    Every value here is a tunable backtest parameter — the defaults are
    starting points for optimization, not truths (plan §6.2).

    - ``sma_fast`` / ``sma_mid`` / ``sma_slow``: periods of the three simple
      moving averages whose ordering defines trend structure.
    - ``slope_lookback``: bars over which the fast SMA's slope is measured
      (``sma_fast[t] - sma_fast[t - slope_lookback]``) for the STRONG regimes.
    - ``atr_period``: ATR period for the volatility-dislocation guard.
    - ``extreme_atr_pct``: ATR / last-close ratio above which the market is
      considered dislocated and the regime is forced to ``TRANSITION``
      (no-trade posture, plan §6.1).
    """

    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200
    slope_lookback: int = 5
    atr_period: int = 14
    extreme_atr_pct: float = 0.06


@dataclass
class RegimeResult:
    """Regime classification plus every input the classifier used.

    ``features`` exists for explainability (plan §6.1): it records the exact
    computed values the decision was based on, so any classification can be
    audited and reproduced after the fact.
    """

    classification: MarketRegime
    features: dict[str, float | bool | None] = field(default_factory=dict)


def classify_regime(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    params: RegimeParams = RegimeParams(),
) -> RegimeResult:
    """Classify the current market regime from aligned OHLC series (plan §6.1).

    v0 rules, evaluated in order (first match wins):

    1. Fewer than ``params.sma_slow`` bars -> ``TRANSITION``. Insufficient
       history means the slow trend structure is undefined; unknown = no-trade
       posture (plan §6.1).
    2. ``ATR / last close > params.extreme_atr_pct`` -> ``TRANSITION``.
       A volatility dislocation invalidates trend-following assumptions
       regardless of SMA structure (vol guard, plan §6.1).
    3. ``close > sma_fast > sma_mid > sma_slow`` AND the fast SMA's slope over
       ``params.slope_lookback`` bars is positive -> ``STRONG_BULL``
       (fully stacked rising trend).
    4. Mirrored: ``close < sma_fast < sma_mid < sma_slow`` AND fast-SMA slope
       negative -> ``STRONG_BEAR``.
    5. ``close > sma_mid`` AND ``close > sma_slow`` -> ``MILD_BULL``
       (above both major trend anchors but not fully stacked).
    6. Mirrored: ``close < sma_mid`` AND ``close < sma_slow`` -> ``MILD_BEAR``.
    7. Otherwise -> ``NEUTRAL_RANGE`` (close sits between the major SMAs; no
       directional trend claim is justified).

    Returns a :class:`RegimeResult` whose ``features`` dict contains every
    computed input (explainability): bar count, last close, the three SMAs,
    the fast-SMA slope, ATR, ATR/close ratio and the guard flags.
    """
    if not (len(closes) == len(highs) == len(lows)):
        raise ValueError(
            "closes, highs and lows must have equal length, got "
            f"{len(closes)}/{len(highs)}/{len(lows)}"
        )
    n = len(closes)
    last_close = closes[-1] if n else None

    features: dict[str, float | bool | None] = {
        "bars": float(n),
        "close": last_close,
        "sma_fast": None,
        "sma_mid": None,
        "sma_slow": None,
        "sma_fast_slope": None,
        "atr": None,
        "atr_pct": None,
        "extreme_vol": False,
        "sufficient_bars": n >= params.sma_slow,
    }

    # Rule 1 — insufficient history = no-trade posture (plan §6.1).
    if n < params.sma_slow:
        return RegimeResult(MarketRegime.TRANSITION, features)

    sma_fast_series = sma(closes, params.sma_fast)
    sma_fast_last = sma_fast_series[-1]
    sma_mid_last = sma(closes, params.sma_mid)[-1]
    sma_slow_last = sma(closes, params.sma_slow)[-1]

    slope_idx = n - 1 - params.slope_lookback
    slope: float | None = None
    if slope_idx >= 0:
        prior = sma_fast_series[slope_idx]
        if sma_fast_last is not None and prior is not None:
            slope = sma_fast_last - prior

    atr_last = atr(highs, lows, closes, period=params.atr_period)[-1]
    atr_pct: float | None = None
    if atr_last is not None and last_close is not None and last_close != 0.0:
        atr_pct = atr_last / last_close

    features.update(
        {
            "sma_fast": sma_fast_last,
            "sma_mid": sma_mid_last,
            "sma_slow": sma_slow_last,
            "sma_fast_slope": slope,
            "atr": atr_last,
            "atr_pct": atr_pct,
        }
    )

    # Rule 2 — volatility dislocation guard (plan §6.1): extreme ATR relative
    # to price forces TRANSITION (no trade) regardless of SMA structure.
    if atr_pct is not None and atr_pct > params.extreme_atr_pct:
        features["extreme_vol"] = True
        return RegimeResult(MarketRegime.TRANSITION, features)

    assert last_close is not None  # n >= sma_slow >= 1

    # Rules 3-4 — STRONG regimes need the full SMA stack AND slope agreement.
    if (
        sma_fast_last is not None
        and sma_mid_last is not None
        and sma_slow_last is not None
    ):
        if (
            slope is not None
            and last_close > sma_fast_last > sma_mid_last > sma_slow_last
            and slope > 0.0
        ):
            return RegimeResult(MarketRegime.STRONG_BULL, features)
        if (
            slope is not None
            and last_close < sma_fast_last < sma_mid_last < sma_slow_last
            and slope < 0.0
        ):
            return RegimeResult(MarketRegime.STRONG_BEAR, features)

    # Rules 5-6 — MILD regimes: above (below) both major trend anchors.
    if sma_mid_last is not None and sma_slow_last is not None:
        if last_close > sma_mid_last and last_close > sma_slow_last:
            return RegimeResult(MarketRegime.MILD_BULL, features)
        if last_close < sma_mid_last and last_close < sma_slow_last:
            return RegimeResult(MarketRegime.MILD_BEAR, features)

    # Rule 7 — no directional claim is justified.
    return RegimeResult(MarketRegime.NEUTRAL_RANGE, features)
