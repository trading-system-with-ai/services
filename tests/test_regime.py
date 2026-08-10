"""Tests for the Market Regime Engine v0 (development plan §6.1).

All series are deterministic synthetic builders (linear/geometric trends and
a noise-free oscillation) so every expected classification can be reasoned
about by hand; determinism and parameterization (plan §6.2: thresholds are
parameters, never truths) are asserted explicitly.
"""
import pytest

from libs.trading_core.models import MarketRegime
from libs.trading_core.signals import RegimeParams, classify_regime


# ---------------------------------------------------------------------------
# Deterministic synthetic series builders
# ---------------------------------------------------------------------------


def linear_trend(n: int, start: float = 100.0, step: float = 0.5, spread: float = 0.5):
    """Straight-line closes with a fixed high/low band around each close."""
    closes = [start + step * i for i in range(n)]
    return closes, [c + spread for c in closes], [c - spread for c in closes]


def geometric_trend(n: int, start: float = 100.0, ratio: float = 1.003, spread_pct: float = 0.005):
    """Constant-percentage-growth closes with a proportional high/low band."""
    closes = [start * ratio**i for i in range(n)]
    return closes, [c * (1.0 + spread_pct) for c in closes], [c * (1.0 - spread_pct) for c in closes]


def oscillation(n: int, mid: float = 100.0, amplitude: float = 1.0, spread: float = 0.5, phase: int = 0):
    """Noise-free period-4 oscillation: mid, mid+a, mid, mid-a, ...

    ``phase`` rotates the pattern so tests can pin exactly where the series
    ends (on the midline, on a peak, or in a trough).
    """
    pattern = [0.0, amplitude, 0.0, -amplitude]
    closes = [mid + pattern[(i + phase) % 4] for i in range(n)]
    return closes, [c + spread for c in closes], [c - spread for c in closes]


# ---------------------------------------------------------------------------
# Trend classifications (§6.1 rules 3-6)
# ---------------------------------------------------------------------------


def test_strong_linear_uptrend_is_strong_bull():
    closes, highs, lows = linear_trend(260)
    result = classify_regime(closes, highs, lows)
    assert result.classification is MarketRegime.STRONG_BULL
    # Explainability: the features carry the full SMA stack the rule used.
    f = result.features
    assert f["close"] > f["sma_fast"] > f["sma_mid"] > f["sma_slow"]
    assert f["sma_fast_slope"] > 0.0
    assert f["sufficient_bars"] is True
    assert f["extreme_vol"] is False


def test_strong_geometric_uptrend_is_strong_bull():
    closes, highs, lows = geometric_trend(260)
    assert classify_regime(closes, highs, lows).classification is MarketRegime.STRONG_BULL


def test_strong_downtrend_is_strong_bear_mirrored():
    # Linear decline that stays well above zero (500 -> 241 over 260 bars).
    closes, highs, lows = linear_trend(260, start=500.0, step=-1.0)
    result = classify_regime(closes, highs, lows)
    assert result.classification is MarketRegime.STRONG_BEAR
    f = result.features
    assert f["close"] < f["sma_fast"] < f["sma_mid"] < f["sma_slow"]
    assert f["sma_fast_slope"] < 0.0


def test_geometric_downtrend_is_strong_bear():
    closes, highs, lows = geometric_trend(260, start=500.0, ratio=0.997)
    assert classify_regime(closes, highs, lows).classification is MarketRegime.STRONG_BEAR


def test_flat_oscillation_is_neutral_range():
    # 253 bars of the period-4 oscillation end exactly on the 100.0 midline
    # (252 % 4 == 0): the close sits ON the fast/slow SMAs and BETWEEN the
    # mid SMA (99.98) and the slow SMA (100.0), so neither the bull nor the
    # bear condition of rules 3-6 holds -> NEUTRAL_RANGE (§6.1 rule 7).
    closes, highs, lows = oscillation(253)
    result = classify_regime(closes, highs, lows)
    assert result.classification is MarketRegime.NEUTRAL_RANGE


def test_oscillation_ending_on_peak_is_mild_bull():
    # 250 bars end on a peak (close 101 > both major SMAs) but the SMA stack
    # is not ordered and the fast slope is flat -> MILD_BULL, not STRONG.
    closes, highs, lows = oscillation(250)
    assert classify_regime(closes, highs, lows).classification is MarketRegime.MILD_BULL


def test_oscillation_ending_in_trough_is_mild_bear():
    # 252 bars end in a trough (close 99 < both major SMAs) -> MILD_BEAR.
    closes, highs, lows = oscillation(252)
    assert classify_regime(closes, highs, lows).classification is MarketRegime.MILD_BEAR


# ---------------------------------------------------------------------------
# No-trade guards (§6.1 rules 1-2: TRANSITION defaults to NO TRADE)
# ---------------------------------------------------------------------------


def test_insufficient_history_is_transition():
    # 50 bars < sma_slow (200): unknown structure = no-trade posture (§6.1).
    closes, highs, lows = linear_trend(50)
    result = classify_regime(closes, highs, lows)
    assert result.classification is MarketRegime.TRANSITION
    f = result.features
    assert f["sufficient_bars"] is False
    assert f["bars"] == 50
    assert f["close"] == closes[-1]  # explainability even on the early exit
    assert f["sma_slow"] is None


def test_extreme_atr_is_transition():
    # Flat closes with an 16-point daily range: ATR/close = 0.16 > 0.06
    # -> volatility dislocation guard forces TRANSITION (§6.1 rule 2).
    n = 250
    closes = [100.0] * n
    highs = [108.0] * n
    lows = [92.0] * n
    result = classify_regime(closes, highs, lows)
    assert result.classification is MarketRegime.TRANSITION
    assert result.features["extreme_vol"] is True
    assert result.features["atr_pct"] == pytest.approx(0.16)


def test_extreme_atr_threshold_is_a_parameter():
    # Same dislocated series, but with the guard relaxed the flat structure
    # classifies NEUTRAL_RANGE — the threshold is a parameter, not a truth
    # (plan §6.2).
    n = 250
    closes = [100.0] * n
    highs = [108.0] * n
    lows = [92.0] * n
    params = RegimeParams(extreme_atr_pct=0.5)
    result = classify_regime(closes, highs, lows, params)
    assert result.classification is MarketRegime.NEUTRAL_RANGE
    assert result.features["extreme_vol"] is False


def test_sma_slow_period_is_a_parameter():
    # 50 bars are insufficient under the default sma_slow=200, but a fully
    # parameterized classifier with a shorter stack classifies the trend.
    closes, highs, lows = linear_trend(50)
    params = RegimeParams(sma_fast=5, sma_mid=10, sma_slow=20)
    result = classify_regime(closes, highs, lows, params)
    assert result.classification is MarketRegime.STRONG_BULL


# ---------------------------------------------------------------------------
# Structural contracts
# ---------------------------------------------------------------------------


def test_determinism_identical_calls_identical_results():
    closes, highs, lows = linear_trend(260)
    assert classify_regime(closes, highs, lows) == classify_regime(closes, highs, lows)
    closes, highs, lows = oscillation(253)
    assert classify_regime(closes, highs, lows) == classify_regime(closes, highs, lows)


def test_misaligned_series_are_rejected():
    closes, highs, lows = linear_trend(260)
    with pytest.raises(ValueError):
        classify_regime(closes, highs[:-1], lows)


def test_features_are_complete_for_explainability():
    # Every input the classifier used must be present in the features dict
    # (plan §6.1: decisions must be auditable after the fact).
    closes, highs, lows = linear_trend(260)
    f = classify_regime(closes, highs, lows).features
    assert set(f) == {
        "bars",
        "close",
        "sma_fast",
        "sma_mid",
        "sma_slow",
        "sma_fast_slope",
        "atr",
        "atr_pct",
        "extreme_vol",
        "sufficient_bars",
    }
    assert all(v is not None for v in f.values())
