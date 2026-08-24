"""Model dispersion & model-risk state tests (Phase B contract §2.7; spec
§39, §40, §59).

Dispersion never averages: it reports min, max and ratio = max/min over the
comparable views (ACTIVE or DEGRADED with a finite positive value), and
flags MODEL_DISPERSION_HIGH iff ratio > params.high_ratio (default 1.5,
STRICTLY greater).
"""
from __future__ import annotations

import pytest

from libs.trading_core.risk.models.base import ModelHealth, ModelMeta, ModelResult
from libs.trading_core.risk.models.diagnostics import TRUST_HIGH, TRUST_LOW
from libs.trading_core.risk.models.ensemble import (
    FLAG_DISPERSION_HIGH,
    RISK_ELEVATED,
    RISK_HIGH,
    RISK_LOW,
    EnsembleParams,
    dispersion,
    model_risk_state,
)


def _meta(name: str = "view") -> ModelMeta:
    return ModelMeta(
        model_name=name,
        model_version="1.0.0",
        params={},
        return_type=None,
        frequency="1D",
        lookback=None,
        data_source=None,
        as_of=None,
        confidence=None,
        horizon_days=1,
        distribution=None,
    )


def view(
    value: float | None,
    health: ModelHealth = ModelHealth.ACTIVE,
    *,
    name: str = "view",
    reason: str | None = None,
) -> ModelResult:
    """A ModelResult honouring base.py's own shape rules (non-ACTIVE needs a
    reason; UNAVAILABLE/FAILED need value=None)."""
    if health is not ModelHealth.ACTIVE and reason is None:
        reason = f"synthetic {health}"
    return ModelResult(
        value=value,
        health=health,
        reason=reason,
        sample_size=100,
        meta=_meta(name),
    )


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_ratio_is_max_over_min_hand_checked() -> None:
    # Spec §39 shape: Historical VaR 1800 vs Gaussian VaR 1200.
    # ratio = 1800/1200 = 1.5 exactly.
    d = dispersion({"gaussian": view(1200.0), "historical": view(1800.0)})
    assert d.ratio == pytest.approx(1.5, rel=1e-15)
    assert d.min_name == "gaussian" and d.min_value == 1200.0
    assert d.max_name == "historical" and d.max_value == 1800.0
    assert d.n_views == 2 and d.n_comparable == 2
    assert d.health is ModelHealth.ACTIVE
    # 1.5 is NOT > 1.5 -> the flag must NOT fire (strict inequality).
    assert d.flag is None
    assert d.is_high is False


def test_flag_fires_strictly_above_the_threshold() -> None:
    # 1801/1200 = 1.50083... > 1.5 -> flag.
    d = dispersion({"gaussian": view(1200.0), "historical": view(1801.0)})
    assert d.ratio == pytest.approx(1801.0 / 1200.0, rel=1e-15)
    assert d.flag == FLAG_DISPERSION_HIGH
    assert d.is_high is True

    # The spec §40 example: 1200 vs 5100 -> 4.25.
    wide = dispersion({"a": view(1200.0), "b": view(5100.0)})
    assert wide.ratio == pytest.approx(4.25, rel=1e-15)
    assert wide.is_high is True


def test_high_ratio_is_a_parameter() -> None:
    views = {"a": view(100.0), "b": view(140.0)}   # ratio = 1.4
    assert dispersion(views).flag is None                                  # 1.4 < 1.5
    loose = EnsembleParams(high_ratio=1.2)
    assert dispersion(views, params=loose).flag == FLAG_DISPERSION_HIGH     # 1.4 > 1.2
    strict = EnsembleParams(high_ratio=2.0)
    assert dispersion(views, params=strict).flag is None                    # 1.4 < 2.0


def test_never_averages_the_views() -> None:
    # The mean of 1200 and 1800 is 1500; it must appear nowhere.
    d = dispersion({"a": view(1200.0), "b": view(1800.0)})
    assert d.min_value == 1200.0 and d.max_value == 1800.0
    assert 1500.0 not in (d.ratio, d.min_value, d.max_value)


def test_unavailable_and_failed_views_are_ignored_but_visible() -> None:
    d = dispersion(
        {
            "a": view(100.0),
            "b": view(140.0),
            "dead": view(None, ModelHealth.UNAVAILABLE),
            "broken": view(None, ModelHealth.FAILED),
        }
    )
    # Only the two comparable views drive the ratio: 140/100 = 1.4.
    assert d.ratio == pytest.approx(1.4, rel=1e-15)
    assert d.n_views == 4
    assert d.n_comparable == 2
    excluded = dict(d.excluded)
    assert set(excluded) == {"dead", "broken"}      # the omission is visible
    assert all(why for why in excluded.values())


def test_degraded_views_still_count_as_comparable() -> None:
    d = dispersion({"a": view(100.0), "b": view(200.0, ModelHealth.DEGRADED)})
    assert d.n_comparable == 2
    assert d.ratio == pytest.approx(2.0, rel=1e-15)


def test_single_comparable_view_is_degraded_not_active() -> None:
    # One view is not an ensemble: saying "no disagreement" would be dishonest.
    d = dispersion({"a": view(100.0), "dead": view(None, ModelHealth.UNAVAILABLE)})
    assert d.n_comparable == 1
    assert d.health in (ModelHealth.DEGRADED, ModelHealth.UNAVAILABLE)
    assert d.reason
    if d.ratio is not None:
        assert d.ratio == pytest.approx(1.0, rel=1e-15)
        assert d.flag is None


def test_no_comparable_views_is_unavailable() -> None:
    d = dispersion({"dead": view(None, ModelHealth.UNAVAILABLE)})
    assert d.ratio is None
    assert d.health is ModelHealth.UNAVAILABLE
    assert d.reason
    assert d.is_available is False
    assert dispersion({}).health is ModelHealth.UNAVAILABLE


def test_non_positive_values_are_excluded() -> None:
    # A ratio over a zero or negative denominator is meaningless.
    d = dispersion({"a": view(100.0), "zero": view(0.0), "neg": view(-50.0)})
    assert d.n_comparable == 1
    assert set(dict(d.excluded)) == {"zero", "neg"}


# ---------------------------------------------------------------------------
# Model-risk state rule table (contract §2.7)
# ---------------------------------------------------------------------------


def test_no_triggers_is_low() -> None:
    s = model_risk_state(views={"a": view(100.0), "b": view(110.0)})
    assert s.state == RISK_LOW
    assert s.reasons == ()          # LOW must carry no reasons
    assert s.n_triggers == 0
    assert s.is_low is True


def test_exactly_one_trigger_is_elevated() -> None:
    # Only trigger: dispersion high (200/100 = 2.0 > 1.5).
    d = dispersion({"a": view(100.0), "b": view(200.0)})
    assert d.is_high
    s = model_risk_state(views={"a": view(100.0), "b": view(200.0)}, dispersion_result=d)
    assert s.state == RISK_ELEVATED
    assert s.n_triggers == 1
    assert s.triggers["dispersion_high"] is True
    assert len(s.reasons) == 1
    assert "dispersion" in s.reasons[0]


def test_two_triggers_is_high() -> None:
    # Trigger 1: dispersion high. Trigger 4: a DEGRADED view.
    views = {"a": view(100.0), "b": view(200.0, ModelHealth.DEGRADED)}
    d = dispersion(views)
    assert d.is_high
    s = model_risk_state(views=views, dispersion_result=d)
    assert s.n_triggers == 2
    assert s.state == RISK_HIGH
    assert s.triggers["dispersion_high"] and s.triggers["sample_degraded"]


def test_any_failed_view_is_high_even_as_the_only_trigger() -> None:
    # A failed estimator is an UNKNOWN, not a wide number: it outranks the count.
    s = model_risk_state(views={"a": view(None, ModelHealth.FAILED), "b": view(100.0)})
    assert s.state == RISK_HIGH
    assert s.reasons and "FAILED" in s.reasons[0]   # prepended, reads first


def test_gaussian_trust_low_only_counts_when_a_gaussian_view_is_live() -> None:
    views = {"gauss": view(100.0), "hist": view(110.0)}
    # LOW trust WITH a live Gaussian view -> the trigger fires.
    s = model_risk_state(views=views, gaussian_trust=TRUST_LOW, gaussian_views=("gauss",))
    assert s.triggers["gaussian_trust_low"] is True
    assert s.state == RISK_ELEVATED

    # Same trust, but the Gaussian view is dead -> the warning is moot.
    dead = {"gauss": view(None, ModelHealth.UNAVAILABLE), "hist": view(110.0)}
    s2 = model_risk_state(views=dead, gaussian_trust=TRUST_LOW, gaussian_views=("gauss",))
    assert s2.triggers["gaussian_trust_low"] is False

    # HIGH trust -> never fires.
    s3 = model_risk_state(views=views, gaussian_trust=TRUST_HIGH, gaussian_views=("gauss",))
    assert s3.triggers["gaussian_trust_low"] is False
    assert s3.state == RISK_LOW


def test_core_view_unavailable_or_absent_triggers() -> None:
    s = model_risk_state(
        views={"hist": view(None, ModelHealth.UNAVAILABLE), "other": view(100.0)},
        core_views=("hist",),
    )
    assert s.triggers["core_unavailable"] is True
    assert "hist" in s.reasons[0]

    # A core view that was never computed at all is exactly as unavailable.
    s2 = model_risk_state(views={"other": view(100.0)}, core_views=("hist",))
    assert s2.triggers["core_unavailable"] is True


def test_high_triggers_is_a_parameter() -> None:
    views = {"a": view(100.0), "b": view(200.0, ModelHealth.DEGRADED)}
    d = dispersion(views)
    # With high_triggers=3, two triggers no longer reach HIGH.
    lenient = EnsembleParams(high_triggers=3)
    s = model_risk_state(views=views, dispersion_result=d, params=lenient)
    assert s.n_triggers == 2
    assert s.state == RISK_ELEVATED


def test_rejects_non_model_result_views() -> None:
    with pytest.raises(ValueError):
        model_risk_state(views={"a": 100.0})       # type: ignore[dict-item]
    with pytest.raises(ValueError):
        dispersion({"a": 100.0})                   # type: ignore[dict-item]
