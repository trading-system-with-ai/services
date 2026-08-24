"""Tests for the Directional Signal Engine v0 (development plan §6.2).

All series are deterministic synthetic builders (linear/geometric trends,
a noise-free oscillation and a trend+triangle-wave combination that produces
confirmed pivots). Score arithmetic is re-derived from the returned
components (self-consistency), and parameterization is asserted explicitly
(plan §6.2: weights and thresholds are parameters, never truths).
"""
import pytest

from libs.trading_core.models import DirectionalBias
from libs.trading_core.signals import DirectionalParams, score_direction

approx = pytest.approx

# Feature pairs always evaluated; volume_expansion only when volumes given.
BASE_COMPONENTS = {
    "close_vs_sma_fast",
    "close_vs_sma_mid",
    "close_vs_sma_slow",
    "sma_fast_slope",
    "macd_cross",
    "macd_zero",
    "rsi_zone",
    "pivot_structure",
}


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


def trending_wave(
    n: int,
    start: float = 100.0,
    step: float = 0.5,
    wave_amp: float = 3.0,
    wave_period: int = 12,
    spread: float = 0.5,
):
    """Linear trend plus a noise-free triangle wave.

    The wave's descent rate (2*amp/half-period = 1.0/bar) exceeds the trend
    slope, so the series has genuine local peaks/troughs that become
    CONFIRMED pivots — which a pure monotonic trend never produces — giving
    the structure component real HH+HL / LH+LL patterns to detect.
    """
    closes = []
    half = wave_period // 2
    for i in range(n):
        phase = i % wave_period
        tri = phase if phase <= half else wave_period - phase
        closes.append(start + step * i + (-wave_amp + 2.0 * wave_amp * tri / half))
    return closes, [c + spread for c in closes], [c - spread for c in closes]


def rising_volumes(n: int, start: float = 1000.0, step: float = 5.0):
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# Directional calls on trends (§6.2)
# ---------------------------------------------------------------------------


def test_strong_uptrend_is_bull():
    # 259 bars end on a wave peak: all SMA, slope, MACD and structure (HH+HL)
    # components trigger bull; volumes are expanding.
    closes, highs, lows = trending_wave(259)
    result = score_direction(closes, highs, lows, rising_volumes(259))
    assert result.bias is DirectionalBias.BULL
    assert result.directional_edge > 0.0
    assert result.bull_score >= 75.0
    assert result.bear_score <= 25.0
    triggered_bull = {c.name for c in result.components if c.side == "bull" and c.triggered}
    assert {"close_vs_sma_slow", "sma_fast_slope", "pivot_structure"} <= triggered_bull


def test_strong_downtrend_is_bear_mirrored():
    # Mirrored decline (400 -> ~274) ending in a wave trough: LH+LL structure.
    closes, highs, lows = trending_wave(253, start=400.0, step=-0.5)
    result = score_direction(closes, highs, lows, rising_volumes(253))
    assert result.bias is DirectionalBias.BEAR
    assert result.directional_edge < 0.0
    assert result.bear_score >= 75.0
    assert result.bull_score <= 25.0
    triggered_bear = {c.name for c in result.components if c.side == "bear" and c.triggered}
    assert {"close_vs_sma_slow", "sma_fast_slope", "pivot_structure"} <= triggered_bear


def test_geometric_uptrend_is_bull_without_volumes():
    closes, highs, lows = geometric_trend(260)
    result = score_direction(closes, highs, lows)
    assert result.bias is DirectionalBias.BULL
    assert result.bull_score > result.bear_score


def test_flat_oscillation_is_neutral():
    # 253 bars end exactly on the 100.0 midline: neither side accumulates
    # enough evidence to clear the bias threshold. (Exact bull==bear symmetry
    # was an artifact of the old equal weights: DIFFERENT components trigger
    # on each side, and §6 grouped weights value them differently — the small
    # residual edge stays far inside the NEUTRAL band.)
    closes, highs, lows = oscillation(253)
    result = score_direction(closes, highs, lows)
    assert result.bias is DirectionalBias.NEUTRAL
    assert abs(result.directional_edge) < DirectionalParams().bias_threshold / 2


# ---------------------------------------------------------------------------
# Insufficient data (§6.2: no information => scores near 0, NEUTRAL)
# ---------------------------------------------------------------------------


def test_50_bars_is_neutral_with_insufficient_components():
    # 50 bars (ending on the midline): the slow-SMA and pivot-structure
    # components cannot be evaluated yet and must say so; the remaining
    # symmetric evidence keeps the bias NEUTRAL.
    closes, highs, lows = oscillation(50, phase=1)
    result = score_direction(closes, highs, lows)
    assert result.bias is DirectionalBias.NEUTRAL
    by_name = {}
    for c in result.components:
        by_name.setdefault(c.name, []).append(c)
    for name in ("close_vs_sma_slow", "pivot_structure"):
        for comp in by_name[name]:
            assert not comp.triggered
            assert "insufficient data" in comp.detail


def test_fewer_bars_than_every_period_scores_zero_neutral():
    # Documented contract (§6.2): with fewer bars than the longest required
    # period, EVERY component is listed but untriggered, both scores are 0
    # and the bias is NEUTRAL — correct no-information behaviour, not an
    # error.
    closes, highs, lows = linear_trend(10)
    result = score_direction(closes, highs, lows, [1000.0] * 10)
    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.directional_edge == 0.0
    assert result.bias is DirectionalBias.NEUTRAL
    assert {c.name for c in result.components} == BASE_COMPONENTS | {"volume_expansion"}
    for comp in result.components:
        assert not comp.triggered
        assert "insufficient data" in comp.detail


# ---------------------------------------------------------------------------
# Component listing and the volume component (§6.2)
# ---------------------------------------------------------------------------


def test_all_components_listed_triggered_or_not():
    closes, highs, lows = trending_wave(259)
    result = score_direction(closes, highs, lows)
    names = {c.name for c in result.components}
    assert names == BASE_COMPONENTS  # no volumes given -> no volume component
    # Every feature appears exactly once per side.
    for name in names:
        sides = sorted(c.side for c in result.components if c.name == name)
        assert sides == ["bear", "bull"]


def test_volume_component_skipped_entirely_without_volumes():
    closes, highs, lows = trending_wave(259)
    without = score_direction(closes, highs, lows)
    with_vol = score_direction(closes, highs, lows, rising_volumes(259))
    assert not any(c.name == "volume_expansion" for c in without.components)
    vol_comps = [c for c in with_vol.components if c.name == "volume_expansion"]
    # When given, the component sits in BOTH sides' denominators (§6.2).
    assert sorted(c.side for c in vol_comps) == ["bear", "bull"]
    assert all(c.triggered for c in vol_comps)  # rising volumes expand


# ---------------------------------------------------------------------------
# Score arithmetic (self-consistency with the returned components)
# ---------------------------------------------------------------------------


def _side_score(components, side: str) -> float:
    total = sum(c.weight for c in components if c.side == side)
    hit = sum(c.weight for c in components if c.side == side and c.triggered)
    return hit / total * 100.0


@pytest.mark.parametrize("with_volumes", [False, True], ids=["no-volumes", "volumes"])
def test_scores_recomputable_from_components(with_volumes):
    closes, highs, lows = trending_wave(259)
    volumes = rising_volumes(259) if with_volumes else None
    result = score_direction(closes, highs, lows, volumes)
    assert result.bull_score == approx(_side_score(result.components, "bull"))
    assert result.bear_score == approx(_side_score(result.components, "bear"))
    assert result.directional_edge == approx(result.bull_score - result.bear_score)


# ---------------------------------------------------------------------------
# Determinism and parameterization (§6.2)
# ---------------------------------------------------------------------------


def test_determinism_identical_calls_identical_results():
    closes, highs, lows = trending_wave(259)
    volumes = rising_volumes(259)
    assert score_direction(closes, highs, lows, volumes) == score_direction(
        closes, highs, lows, volumes
    )


# Explicit EQUAL-weight parameter set: these tests verify the scoring
# MECHANICS (weight doubling arithmetic), which is clearest against a 1.0
# baseline — the §6 grouped defaults are characterized elsewhere.
EQUAL_WEIGHTS = dict(
    weight_sma_fast=1.0, weight_sma_mid=1.0, weight_sma_slow=1.0,
    weight_sma_slope=1.0, weight_macd_cross=1.0, weight_macd_zero=1.0,
    weight_rsi_zone=1.0, weight_structure=1.0, weight_volume=1.0,
)


def test_doubling_a_triggered_weight_raises_the_score_as_expected():
    closes, highs, lows = trending_wave(259)
    base = score_direction(closes, highs, lows, params=DirectionalParams(**EQUAL_WEIGHTS))
    # Baseline: the slow-SMA bull component is triggered with weight 1.0.
    slow_bull = next(
        c for c in base.components if c.name == "close_vs_sma_slow" and c.side == "bull"
    )
    assert slow_bull.triggered and slow_bull.weight == 1.0

    doubled = score_direction(
        closes, highs, lows,
        params=DirectionalParams(**{**EQUAL_WEIGHTS, "weight_sma_slow": 2.0}),
    )
    bull_hit = sum(c.weight for c in base.components if c.side == "bull" and c.triggered)
    bull_total = sum(c.weight for c in base.components if c.side == "bull")
    bear_hit = sum(c.weight for c in base.components if c.side == "bear" and c.triggered)
    bear_total = sum(c.weight for c in base.components if c.side == "bear")
    # The weight applies to the feature PAIR: +1 weight on both denominators,
    # +1 on the bull numerator (triggered), +0 on the bear numerator (not).
    assert doubled.bull_score == approx((bull_hit + 1.0) / (bull_total + 1.0) * 100.0)
    assert doubled.bear_score == approx(bear_hit / (bear_total + 1.0) * 100.0)
    assert doubled.bull_score > base.bull_score  # untriggered rsi_zone dilutes less


def test_doubling_an_untriggered_weight_lowers_the_score_as_expected():
    closes, highs, lows = trending_wave(259)
    base = score_direction(closes, highs, lows, params=DirectionalParams(**EQUAL_WEIGHTS))
    # Baseline: RSI is overbought in this trend, so rsi_zone is untriggered
    # on both sides with weight 1.0.
    rsi_comps = [c for c in base.components if c.name == "rsi_zone"]
    assert all(not c.triggered and c.weight == 1.0 for c in rsi_comps)

    doubled = score_direction(
        closes, highs, lows,
        params=DirectionalParams(**{**EQUAL_WEIGHTS, "weight_rsi_zone": 2.0}),
    )
    bull_hit = sum(c.weight for c in base.components if c.side == "bull" and c.triggered)
    bull_total = sum(c.weight for c in base.components if c.side == "bull")
    assert doubled.bull_score == approx(bull_hit / (bull_total + 1.0) * 100.0)
    assert doubled.bull_score < base.bull_score


def test_bias_threshold_is_a_parameter():
    closes, highs, lows = trending_wave(259)
    base = score_direction(closes, highs, lows)
    assert base.bias is DirectionalBias.BULL
    # The same evidence with a stricter threshold is not enough to call BULL.
    strict = score_direction(
        closes, highs, lows, params=DirectionalParams(bias_threshold=base.directional_edge + 1.0)
    )
    assert strict.bias is DirectionalBias.NEUTRAL
    assert strict.directional_edge == approx(base.directional_edge)


def test_misaligned_series_are_rejected():
    closes, highs, lows = trending_wave(100)
    with pytest.raises(ValueError):
        score_direction(closes, highs[:-1], lows)
    with pytest.raises(ValueError):
        score_direction(closes, highs, lows, [1.0] * 99)
