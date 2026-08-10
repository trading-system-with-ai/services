"""Tests for Volatility Regime v0 (development plan §7).

Pins every threshold band (level and ratio paths), the exact boundary
values, the honest-null handling of an unavailable realized vol (ratio is
None, never a fake number), the precedence order (EXTREME > HIGH > LOW >
NORMAL), the explainability of the features dict, and input validation.
The level thresholds are PROVISIONAL until IV history enables IV Rank
(plan §7); the defaults are asserted so a silent retune fails a test.
"""
import pytest

from libs.trading_core.models import IVRegime
from libs.trading_core.volatility import (
    VolRegimeParams,
    VolRegimeResult,
    classify_vol_regime,
)

# ---------------------------------------------------------------------------
# Parameters (plan §7 provisional defaults; §6.2: parameters, never truths)
# ---------------------------------------------------------------------------


def test_default_params_match_plan_7_provisional_thresholds():
    p = VolRegimeParams()
    assert (p.low_iv, p.high_iv, p.extreme_iv) == (0.20, 0.35, 0.60)
    assert (p.low_ratio, p.high_ratio, p.extreme_ratio) == (1.1, 1.5, 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_iv": 0.0},
        {"low_iv": 0.40},  # low >= high
        {"high_iv": 0.70},  # high >= extreme
        {"low_ratio": 0.0},
        {"low_ratio": 1.6},  # low >= high
        {"high_ratio": 2.5},  # high >= extreme
    ],
)
def test_invalid_params_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        VolRegimeParams(**kwargs)


# ---------------------------------------------------------------------------
# Input validation (checked at the door)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atm_iv", [0.0, -0.25])
def test_non_positive_atm_iv_raises(atm_iv):
    with pytest.raises(ValueError):
        classify_vol_regime(atm_iv, 0.20)


@pytest.mark.parametrize("rv", [0.0, -0.10])
def test_non_positive_rv_raises_pass_none_instead(rv):
    """A degenerate RV must be passed as None (honest null), never 0.0."""
    with pytest.raises(ValueError):
        classify_vol_regime(0.30, rv)


# ---------------------------------------------------------------------------
# LOW band (level AND ratio must agree — or ratio be unavailable)
# ---------------------------------------------------------------------------


def test_low_when_level_and_ratio_both_low():
    # 0.15 <= 0.20 and ratio 0.15/0.15 = 1.0 <= 1.1.
    r = classify_vol_regime(0.15, 0.15)
    assert r.regime is IVRegime.LOW
    assert r.features["iv_rv_ratio"] == pytest.approx(1.0)
    assert set(r.features["thresholds_fired"]) == {"low_iv", "low_ratio"}


def test_low_boundary_exact_level_and_exact_ratio():
    # atm_iv exactly at low_iv (<= is inclusive) and ratio exactly at
    # low_ratio: 0.20 / (0.20/1.1) = 1.1 <= 1.1 -> LOW.
    rv = 0.20 / 1.1
    r = classify_vol_regime(0.20, rv)
    assert r.regime is IVRegime.LOW
    assert r.features["iv_rv_ratio"] == pytest.approx(1.1)


def test_low_level_but_ratio_above_low_ratio_is_normal():
    # Level says LOW (0.18 <= 0.20) but ratio 0.18/0.14 = 1.286 > 1.1: the
    # ratio blocks the LOW verdict -> NORMAL (plan §7: the ratio confirms
    # LOW, it does not price premium on its own).
    r = classify_vol_regime(0.18, 0.14)
    assert r.regime is IVRegime.NORMAL
    assert "low_iv" in r.features["thresholds_fired"]
    assert "low_ratio" not in r.features["thresholds_fired"]


def test_low_with_rv_none_falls_back_to_level_alone():
    # Honest null: no RV -> ratio None; 0.15 <= 0.20 alone gives LOW.
    r = classify_vol_regime(0.15, None)
    assert r.regime is IVRegime.LOW
    assert r.features["rv"] is None
    assert r.features["iv_rv_ratio"] is None
    assert r.features["thresholds_fired"] == ["low_iv"]


# ---------------------------------------------------------------------------
# NORMAL band (nothing fires)
# ---------------------------------------------------------------------------


def test_normal_mid_band_level_and_ratio():
    # 0.20 < 0.25 < 0.35, ratio 0.25/0.22 = 1.136 < 1.5 -> NORMAL.
    r = classify_vol_regime(0.25, 0.22)
    assert r.regime is IVRegime.NORMAL
    assert r.features["thresholds_fired"] == []


def test_normal_just_above_low_and_just_below_high():
    assert classify_vol_regime(0.2001, None).regime is IVRegime.NORMAL
    assert classify_vol_regime(0.3499, None).regime is IVRegime.NORMAL


# ---------------------------------------------------------------------------
# HIGH band (level OR ratio escalates)
# ---------------------------------------------------------------------------


def test_high_by_level_boundary_inclusive():
    # atm_iv exactly at high_iv fires (>=), even with a tame ratio.
    r = classify_vol_regime(0.35, 0.30)  # ratio 1.167 < 1.5
    assert r.regime is IVRegime.HIGH
    assert r.features["thresholds_fired"] == ["high_iv"]


def test_high_by_ratio_boundary_inclusive():
    # Level is mid-band (0.28125 < 0.35) but the ratio is EXACTLY 1.5 and
    # fires (>=). 0.28125 (9/32) and 0.1875 (3/16) are exact binary
    # fractions, so 0.28125/0.1875 == 1.5 bit-exactly.
    r = classify_vol_regime(0.28125, 0.1875)
    assert r.regime is IVRegime.HIGH
    assert r.features["iv_rv_ratio"] == 1.5
    assert r.features["thresholds_fired"] == ["high_ratio"]


def test_high_with_rv_none_level_alone():
    r = classify_vol_regime(0.40, None)
    assert r.regime is IVRegime.HIGH
    assert r.features["iv_rv_ratio"] is None


# ---------------------------------------------------------------------------
# EXTREME band (level OR ratio; outranks everything)
# ---------------------------------------------------------------------------


def test_extreme_by_level_boundary_inclusive():
    r = classify_vol_regime(0.60, None)
    assert r.regime is IVRegime.EXTREME
    # 0.60 also clears high_iv — both true comparisons are reported, the
    # regime is the highest-precedence band (plan §37 explainability).
    assert set(r.features["thresholds_fired"]) == {"extreme_iv", "high_iv"}


def test_extreme_by_ratio_boundary_inclusive():
    # Level mid-band (0.30) but ratio exactly 2.0 -> EXTREME. A ratio of
    # 2.0 also clears high_ratio (1.5); both true comparisons are reported.
    r = classify_vol_regime(0.30, 0.15)
    assert r.regime is IVRegime.EXTREME
    assert r.features["iv_rv_ratio"] == pytest.approx(2.0)
    assert set(r.features["thresholds_fired"]) == {"extreme_ratio", "high_ratio"}


def test_extreme_ratio_outranks_low_level():
    # A LOW IV level with a blown-out ratio is still EXTREME: 0.20/0.09 =
    # 2.22 >= 2.0. Precedence: EXTREME wins over the low_iv comparison.
    r = classify_vol_regime(0.20, 0.09)
    assert r.regime is IVRegime.EXTREME
    assert "extreme_ratio" in r.features["thresholds_fired"]
    assert "low_iv" in r.features["thresholds_fired"]


# ---------------------------------------------------------------------------
# rv=None never escalates (plan §7: "Do not assume IV > RV automatically
# means options are overpriced" — and no ratio means no ratio verdict at all)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("atm_iv", "expected"),
    [
        (0.15, IVRegime.LOW),
        (0.20, IVRegime.LOW),  # boundary, inclusive
        (0.25, IVRegime.NORMAL),
        (0.35, IVRegime.HIGH),  # boundary, inclusive
        (0.45, IVRegime.HIGH),
        (0.60, IVRegime.EXTREME),  # boundary, inclusive
        (0.75, IVRegime.EXTREME),
    ],
)
def test_rv_none_level_only_bands(atm_iv, expected):
    r = classify_vol_regime(atm_iv, None)
    assert r.regime is expected
    assert r.features["iv_rv_ratio"] is None
    ratio_names = {"low_ratio", "high_ratio", "extreme_ratio"}
    assert not ratio_names & set(r.features["thresholds_fired"])


# ---------------------------------------------------------------------------
# Explainability and custom parameters
# ---------------------------------------------------------------------------


def test_features_always_report_inputs():
    r = classify_vol_regime(0.28, 0.25)
    assert isinstance(r, VolRegimeResult)
    assert r.features["atm_iv"] == 0.28
    assert r.features["rv"] == 0.25
    assert r.features["iv_rv_ratio"] == pytest.approx(0.28 / 0.25)
    assert isinstance(r.features["thresholds_fired"], list)


def test_custom_params_are_respected():
    # Retuned thresholds move the bands (plan §6.2: parameters, not truths):
    # with extreme_iv=0.50 an IV of 0.55 becomes EXTREME instead of HIGH.
    assert classify_vol_regime(0.55, None).regime is IVRegime.HIGH
    p = VolRegimeParams(extreme_iv=0.50)
    assert classify_vol_regime(0.55, None, params=p).regime is IVRegime.EXTREME
    # And a looser low_ratio admits a ratio the defaults would reject.
    assert classify_vol_regime(0.18, 0.14).regime is IVRegime.NORMAL
    p2 = VolRegimeParams(low_ratio=1.3)
    assert classify_vol_regime(0.18, 0.14, params=p2).regime is IVRegime.LOW


def test_determinism():
    assert classify_vol_regime(0.31, 0.24) == classify_vol_regime(0.31, 0.24)
