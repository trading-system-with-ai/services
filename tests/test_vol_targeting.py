"""Volatility-targeting exposure multiplier tests (development plan §14).

The multiplier is ``clamp(target_vol / forecast_vol, min, max)`` with an
honest 1.0 default when there is no usable forecast. §14: the result only
scales a risk budget — hard caps are enforced downstream by the risk engine
regardless (covered in tests/test_risk_engine.py).
"""
import pytest

from libs.trading_core.allocation import VolTargetParams, exposure_multiplier


def test_no_forecast_means_no_adjustment():
    # None or non-positive forecast -> honest 1.0, never a guessed scale.
    assert exposure_multiplier(None) == 1.0
    assert exposure_multiplier(0.0) == 1.0
    assert exposure_multiplier(-0.05) == 1.0


def test_target_over_forecast_arithmetic():
    # target 0.12 / forecast 0.24 = 0.5 — high vol halves exposure.
    assert exposure_multiplier(0.24) == pytest.approx(0.5)
    # target 0.12 / forecast 0.12 = 1.0 — on-target vol changes nothing.
    assert exposure_multiplier(0.12) == pytest.approx(1.0)
    # target 0.12 / forecast 0.15 = 0.8.
    assert exposure_multiplier(0.15) == pytest.approx(0.8)


def test_upward_clamp_at_max_multiplier():
    # target 0.12 / forecast 0.06 = 2.0 raw, clamped to max 1.2 (§14: cap
    # upward leverage in calm markets).
    assert exposure_multiplier(0.06) == pytest.approx(1.2)
    # Just inside the cap: 0.12 / 0.11 ~= 1.0909 < 1.2 -> unclamped.
    assert exposure_multiplier(0.11) == pytest.approx(0.12 / 0.11)
    # Exactly at the cap boundary: 0.12 / 0.10 = 1.2 -> exactly max.
    assert exposure_multiplier(0.10) == pytest.approx(1.2)


def test_downward_clamp_at_min_multiplier():
    # target 0.12 / forecast 1.20 = 0.1 raw, floored at 0.25 so a vol
    # spike shrinks sizing sanely instead of collapsing it to zero.
    assert exposure_multiplier(1.20) == pytest.approx(0.25)
    # Exactly at the floor boundary: 0.12 / 0.48 = 0.25.
    assert exposure_multiplier(0.48) == pytest.approx(0.25)
    # Just above the floor: 0.12 / 0.40 = 0.30 -> unclamped.
    assert exposure_multiplier(0.40) == pytest.approx(0.30)


def test_custom_params_are_obeyed():
    # All thresholds are parameters (house rule), never hardcoded truths.
    params = VolTargetParams(
        target_vol=0.20, max_multiplier=1.5, min_multiplier=0.5
    )
    assert exposure_multiplier(0.10, params) == pytest.approx(1.5)  # 2.0 raw
    assert exposure_multiplier(0.20, params) == pytest.approx(1.0)
    assert exposure_multiplier(0.80, params) == pytest.approx(0.5)  # 0.25 raw
    assert exposure_multiplier(None, params) == 1.0


def test_invalid_params_rejected_early():
    with pytest.raises(ValueError):
        VolTargetParams(target_vol=0.0)
    with pytest.raises(ValueError):
        VolTargetParams(target_vol=-0.1)
    with pytest.raises(ValueError):
        VolTargetParams(min_multiplier=0.0)
    with pytest.raises(ValueError):
        VolTargetParams(min_multiplier=1.5, max_multiplier=1.2)


def test_defaults_documented_values():
    params = VolTargetParams()
    assert params.target_vol == 0.12
    assert params.max_multiplier == 1.2
    assert params.min_multiplier == 0.25
