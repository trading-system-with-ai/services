"""Distribution diagnostics tests (Phase B contract §2.6, spec §15).

Estimators under test (population moments about the sample mean, divisor n):

    m_r = Sum_t (x_t - mean)^r / n
    g1  = m3 / m2^{3/2}          (skewness, biased Fisher-Pearson)
    g2  = m4 / m2^2 - 3          (excess kurtosis)
    JB  = n/6 * (g1^2 + g2^2/4)
    jb_p = exp(-JB/2)            (chi^2(2) survival, closed form)
"""
from __future__ import annotations

import math

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.diagnostics import (
    FLAG_HEAVY_TAIL,
    FLAG_LEFT_SKEWED,
    FLAG_NORMAL_LIKE,
    FLAG_UNSTABLE,
    TRUST_HIGH,
    TRUST_LOW,
    TRUST_REDUCED,
    DistributionParams,
    distribution_diagnostics,
    jarque_bera_p_value,
)

P10 = DistributionParams(min_obs=10)


def _moments(values: list[float]) -> tuple[float, float, float]:
    """Hand-recomputed g1, g2, JB straight from the definitions."""
    n = len(values)
    mean = math.fsum(values) / n
    dev = [v - mean for v in values]
    m2 = math.fsum(d ** 2 for d in dev) / n
    m3 = math.fsum(d ** 3 for d in dev) / n
    m4 = math.fsum(d ** 4 for d in dev) / n
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2 - 3.0
    return g1, g2, (n / 6.0) * (g1 ** 2 + g2 ** 2 / 4.0)


# ---------------------------------------------------------------------------
# Symmetric series -> g1 == 0
# ---------------------------------------------------------------------------


def test_symmetric_series_has_zero_skew() -> None:
    # -3..3 repeated: mean = 0 by symmetry, and every odd deviation power
    # cancels pairwise -> m3 = 0 -> g1 = 0 EXACTLY.
    series = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10
    r = distribution_diagnostics(series, params=P10)
    assert r.n == 70
    assert r.mean == pytest.approx(0.0, abs=1e-15)
    assert r.skew == pytest.approx(0.0, abs=1e-12)

    # Hand: m2 = 2*(9+4+1)/7 = 28/7 = 4 ; m4 = 2*(81+16+1)/7 = 196/7 = 28
    # g2 = 28/16 - 3 = 1.75 - 3 = -1.25  (a flat/uniform-ish shape)
    assert r.excess_kurtosis == pytest.approx(-1.25, abs=1e-12)
    g1, g2, jb = _moments(series)
    assert r.skew == pytest.approx(g1, abs=1e-12)
    assert r.excess_kurtosis == pytest.approx(g2, abs=1e-12)
    assert r.jarque_bera == pytest.approx(jb, rel=1e-12)


def test_jb_p_is_the_closed_form_exp_minus_jb_over_two() -> None:
    series = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10
    r = distribution_diagnostics(series, params=P10)
    assert r.jb_p == pytest.approx(math.exp(-r.jarque_bera / 2.0), rel=1e-15)

    # Standalone closed form, hand-checked at two points:
    assert jarque_bera_p_value(0.0) == pytest.approx(1.0, abs=1e-15)      # exp(0)
    # JB = 2*ln(2) = 1.3862943611 -> p = exp(-ln 2) = 0.5
    assert jarque_bera_p_value(2.0 * math.log(2.0)) == pytest.approx(0.5, rel=1e-15)
    # JB = 5.991 (the classic chi2(2) 5% critical value) -> p ~= 0.05
    assert jarque_bera_p_value(5.991464547) == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Heavy tail / left skew labelling
# ---------------------------------------------------------------------------


def test_heavy_tail_series_is_flagged() -> None:
    # Mostly zeros with a few symmetric large outliers: symmetric (g1 = 0)
    # but very peaked-with-fat-tails -> g2 well above heavy_tail_kurtosis=1.0.
    series = [0.0] * 96 + [-12.0, 12.0, -11.0, 11.0]
    r = distribution_diagnostics(series, params=P10)
    g1, g2, _ = _moments(series)
    assert r.excess_kurtosis == pytest.approx(g2, rel=1e-12)
    assert g2 > 1.0
    assert FLAG_HEAVY_TAIL in r.flags
    assert r.skew == pytest.approx(0.0, abs=1e-9)     # still symmetric
    assert FLAG_LEFT_SKEWED not in r.flags
    assert r.primary == FLAG_HEAVY_TAIL
    assert r.gaussian_trust == TRUST_LOW
    assert r.has(FLAG_HEAVY_TAIL)


def test_left_skewed_series_is_flagged() -> None:
    # Many small gains, a couple of large losses -> long LEFT tail, g1 < -0.5.
    series = [1.0] * 40 + [-9.0, -8.0]
    r = distribution_diagnostics(series, params=P10)
    g1, _, _ = _moments(series)
    assert r.skew == pytest.approx(g1, rel=1e-12)
    assert g1 < -0.5
    assert FLAG_LEFT_SKEWED in r.flags
    assert r.primary == FLAG_LEFT_SKEWED      # priority over HEAVY_TAIL
    assert r.gaussian_trust == TRUST_LOW


def test_flag_thresholds_are_parameters() -> None:
    series = [1.0] * 40 + [-9.0, -8.0]
    g1, g2, _ = _moments(series)
    # Loosen left_skew below the actual g1 -> the flag must disappear.
    loose = DistributionParams(min_obs=10, left_skew=g1 - 1.0, heavy_tail_kurtosis=g2 + 1.0)
    r = distribution_diagnostics(series, params=loose)
    assert FLAG_LEFT_SKEWED not in r.flags
    assert FLAG_HEAVY_TAIL not in r.flags


# ---------------------------------------------------------------------------
# gaussian_trust mapping (spec §15)
# ---------------------------------------------------------------------------


def test_gaussian_trust_reduced_when_only_the_p_value_rejects() -> None:
    # The uniform-ish -3..3 series: no heavy tail (g2 = -1.25 < 1.0), no left
    # skew (g1 = 0), but n=70 makes JB = 70/6 * (0 + 1.25^2/4) = 4.5573
    # -> p = exp(-2.2786) = 0.1024 >= 0.05, so it reads NORMAL_LIKE / HIGH.
    series = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10
    r = distribution_diagnostics(series, params=P10)
    assert r.jarque_bera == pytest.approx(70.0 / 6.0 * (1.25 ** 2 / 4.0), rel=1e-12)
    assert r.jarque_bera == pytest.approx(4.557291666, abs=1e-8)
    assert r.jb_p == pytest.approx(math.exp(-4.557291666 / 2.0), abs=1e-9)
    assert r.jb_p > 0.05
    assert r.primary == FLAG_NORMAL_LIKE
    assert r.gaussian_trust == TRUST_HIGH

    # Lengthen the SAME shape to n=350: JB scales linearly with n
    # -> JB = 350/6 * 0.390625 = 22.786 -> p = exp(-11.39) = 1.1e-5 < 0.05.
    # No heavy/skew flag applies, so trust must be REDUCED (not LOW).
    long_series = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 50
    lr = distribution_diagnostics(long_series, params=P10)
    assert lr.jarque_bera == pytest.approx(350.0 / 6.0 * (1.25 ** 2 / 4.0), rel=1e-12)
    assert lr.jb_p < 0.05
    assert FLAG_HEAVY_TAIL not in lr.flags and FLAG_LEFT_SKEWED not in lr.flags
    assert lr.gaussian_trust == TRUST_REDUCED


# ---------------------------------------------------------------------------
# UNSTABLE: honest nulls (contract §1, §3.6)
# ---------------------------------------------------------------------------


def test_below_min_obs_is_unstable_and_unavailable() -> None:
    r = distribution_diagnostics([1.0, 2.0, 3.0], params=P10)
    assert r.primary == FLAG_UNSTABLE
    assert r.flags == (FLAG_UNSTABLE,)
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.skew is None and r.excess_kurtosis is None
    assert r.jarque_bera is None and r.jb_p is None
    assert r.reason == "n=3 < min_obs=10"
    assert r.gaussian_trust == TRUST_LOW      # cannot vouch for Gaussian
    assert r.is_available is False
    # mean/stdev are still honest when n >= 2
    assert r.mean == pytest.approx(2.0, abs=1e-15)


def test_zero_variance_is_unstable() -> None:
    r = distribution_diagnostics([2.0] * 50, params=P10)
    assert r.primary == FLAG_UNSTABLE
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.skew is None                      # 0/0 has no value
    assert r.reason and "variance" in r.reason
    assert r.mean == pytest.approx(2.0, abs=1e-15)
    assert r.stdev == pytest.approx(0.0, abs=1e-15)
    assert r.gaussian_trust == TRUST_LOW


def test_non_finite_input_raises() -> None:
    with pytest.raises(ValueError):
        distribution_diagnostics([1.0, 2.0, math.nan] * 10, params=P10)
    with pytest.raises(ValueError):
        distribution_diagnostics([1.0, 2.0, math.inf] * 10, params=P10)


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        DistributionParams(min_obs=2)          # needs >= 3 for a 3rd moment
    with pytest.raises(ValueError):
        DistributionParams(normal_p=0.0)
    with pytest.raises(ValueError):
        DistributionParams(normal_p=1.0)


def test_scale_and_location_invariance_of_shape_statistics() -> None:
    # g1/g2 are invariant to x -> a*x + b (a > 0): a genuinely useful
    # property to pin, since it is what lets one function serve P&L and returns.
    series = [1.0] * 40 + [-9.0, -8.0]
    base = distribution_diagnostics(series, params=P10)
    moved = distribution_diagnostics([3.0 * v + 17.0 for v in series], params=P10)
    assert moved.skew == pytest.approx(base.skew, rel=1e-12)
    assert moved.excess_kurtosis == pytest.approx(base.excess_kurtosis, rel=1e-12)
    assert moved.jb_p == pytest.approx(base.jb_p, rel=1e-12)
