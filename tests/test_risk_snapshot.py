"""Tests for libs/trading_core/risk/snapshot.py (Phase B contract §2.11;
spec §45, §55). Every number is hand-checked in a comment."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta, timezone

import pytest

from libs.trading_core.greeks import PortfolioGreeks
from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMeta,
    active,
    degraded,
    failed,
    unavailable,
)
from libs.trading_core.risk.snapshot import (
    SNAPSHOT_VERSION,
    DataQuality,
    PortfolioRiskSnapshot,
    TtlPolicy,
)

AS_OF = datetime(2026, 8, 17, 21, 0, 0, tzinfo=timezone.utc)


def _meta(name: str, conf: float = 0.95) -> ModelMeta:
    return ModelMeta(model_name=name, model_version="1.0.0", confidence=conf, horizon_days=1)


def _dq(valid: bool = True, **kw) -> DataQuality:
    base = dict(as_of=date(2026, 8, 15), oldest_bar=date(2026, 5, 20), newest_bar=date(2026, 8, 15), n_obs=60)
    base.update(kw)
    if not valid:
        base.setdefault("reasons", ("tickers_missing=('XYZ',)",))
    return DataQuality(valid=valid, **base)


def _snap(**kw) -> PortfolioRiskSnapshot:
    fields = dict(
        as_of=AS_OF, nav=100_000.0, cash=40_000.0, cash_reserved=5_000.0,
        gross_exposure=60_000.0, delta_adjusted_exposure=55_000.0,
        heat_pct=0.03, heat_state="NORMAL", data_quality=_dq(),
    )
    fields.update(kw)
    return PortfolioRiskSnapshot(**fields)


# --- construction --------------------------------------------------------

def test_construct_with_model_results_and_version():
    var95 = active(_meta("historical_var"), 1_250.0, 60)          # VaR95 = $1,250 loss
    es95 = active(_meta("historical_es"), 1_800.0, 60)            # ES95 = $1,800 >= VaR95
    vol = degraded(_meta("portfolio_vol"), "n=60 == min_obs", 900.0, 60)
    s = _snap(volatility=vol, var={"HISTORICAL:0.95:1": var95}, es={"HISTORICAL:0.95:1": es95},
              model_health={"historical_var": ModelHealth.ACTIVE})
    assert s.snapshot_version == SNAPSHOT_VERSION == "b.1"
    assert s.var["HISTORICAL:0.95:1"].value == 1250.0
    assert s.es["HISTORICAL:0.95:1"].value >= s.var["HISTORICAL:0.95:1"].value  # contract §3.1
    assert s.risk_state == "NORMAL"          # defaults to heat_state
    assert s.correlation_state is None       # Phase C
    assert s.result_keys() == ("HISTORICAL:0.95:1",)
    assert isinstance(s.ttl, TtlPolicy)
    assert s.ttl.statistical_seconds == 86400.0 and s.ttl.greeks_seconds == 120.0


def test_mapping_fields_are_copied_and_typed():
    src = {"HISTORICAL:0.95:1": active(_meta("historical_var"), 1.0, 60)}
    s = _snap(var=src)
    src["GAUSSIAN:0.95:1"] = active(_meta("gaussian_var"), 2.0, 60)
    assert list(s.var) == ["HISTORICAL:0.95:1"]     # caller mutation does not leak
    with pytest.raises(ValueError):
        _snap(var={"X": 3.0})                       # untyped JSON not accepted
    with pytest.raises(ValueError):
        _snap(model_health={"m": "GREEN"})          # unknown health state


def test_malformed_inputs_raise():
    with pytest.raises(ValueError):
        _snap(as_of=date(2026, 8, 17))              # must be datetime (spec §55)
    with pytest.raises(ValueError):
        _snap(nav=float("nan"))
    with pytest.raises(ValueError):
        TtlPolicy(statistical_seconds=0)
    with pytest.raises(ValueError):
        DataQuality(as_of=None, oldest_bar=date(2026, 2, 1), newest_bar=date(2026, 1, 1))
    with pytest.raises(ValueError):
        DataQuality(as_of=None, oldest_bar=None, newest_bar=None, valid=False)  # needs a reason


def test_immutability():
    s = _snap()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.nav = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.ttl.greeks_seconds = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.data_quality.valid = False  # type: ignore[misc]


# --- staleness (spec §55) ------------------------------------------------

def test_is_stale_statistical_boundary():
    s = _snap()
    # TTL = 86400 s: as_of + 86400 s is exactly TTL old -> not stale
    assert s.is_stale(AS_OF + timedelta(seconds=86400)) is False
    assert s.is_stale(AS_OF + timedelta(seconds=86400), kind="statistical") is False
    # TTL + 1 s = 86401 s -> stale
    assert s.is_stale(AS_OF + timedelta(seconds=86401)) is True
    assert s.is_stale(AS_OF) is False              # age 0
    assert s.age_seconds(AS_OF + timedelta(minutes=2)) == 120.0


def test_is_stale_greeks_boundary():
    s = _snap()
    # greeks TTL = 120 s: exactly 120 s -> not stale; 121 s -> stale
    assert s.is_stale(AS_OF + timedelta(seconds=120), kind="greeks") is False
    assert s.is_stale(AS_OF + timedelta(seconds=121), kind="greeks") is True
    # the same instant is fresh for statistical (121 <= 86400)
    assert s.is_stale(AS_OF + timedelta(seconds=121), kind="statistical") is False


def test_is_stale_custom_ttl_and_bad_inputs():
    s = _snap(ttl=TtlPolicy(statistical_seconds=10, greeks_seconds=1))
    assert s.is_stale(AS_OF + timedelta(seconds=10)) is False
    assert s.is_stale(AS_OF + timedelta(seconds=10.5)) is True
    assert s.is_stale(AS_OF + timedelta(seconds=1), kind="greeks") is False
    assert s.is_stale(AS_OF + timedelta(seconds=2), kind="greeks") is True
    with pytest.raises(ValueError):
        s.is_stale(AS_OF, kind="garch")
    with pytest.raises(ValueError):
        s.is_stale(datetime(2026, 8, 17, 21, 0, 0))    # naive vs aware
    with pytest.raises(ValueError):
        s.is_stale(date(2026, 8, 18))  # type: ignore[arg-type]


# --- health (spec §41) ---------------------------------------------------

def test_health_summary_and_overall_worst_of():
    var95 = active(_meta("historical_var"), 1_250.0, 60)
    var99 = degraded(_meta("historical_var", 0.99), "tail_size=1 < 3", 2_100.0, 60)
    es99 = unavailable(_meta("historical_es", 0.99), "n=17 < min_obs=60", 17)
    s = _snap(volatility=active(_meta("portfolio_vol"), 900.0, 60),
              var={"HISTORICAL:0.95:1": var95, "HISTORICAL:0.99:1": var99},
              es={"HISTORICAL:0.99:1": es99})
    hs = s.health_summary()
    assert hs == {
        "volatility": ModelHealth.ACTIVE,
        "var:HISTORICAL:0.95:1": ModelHealth.ACTIVE,
        "var:HISTORICAL:0.99:1": ModelHealth.DEGRADED,
        "es:HISTORICAL:0.99:1": ModelHealth.UNAVAILABLE,
    }
    # worst of {ACTIVE, ACTIVE, DEGRADED, UNAVAILABLE} = UNAVAILABLE
    assert s.overall_health() is ModelHealth.UNAVAILABLE


def test_overall_health_includes_model_health_ledger_and_data_quality():
    ok = active(_meta("historical_var"), 1.0, 60)
    s = _snap(var={"HISTORICAL:0.95:1": ok})
    assert s.overall_health() is ModelHealth.ACTIVE
    # ledger says a model FAILED -> worst-of = FAILED
    s2 = _snap(var={"HISTORICAL:0.95:1": ok}, model_health={"gaussian_var": ModelHealth.FAILED})
    assert s2.health_summary() == {"var:HISTORICAL:0.95:1": ModelHealth.ACTIVE}   # ledger not merged
    assert s2.overall_health() is ModelHealth.FAILED
    # invalid data quality lifts ACTIVE to DEGRADED, but never lowers a worse state
    s3 = _snap(var={"HISTORICAL:0.95:1": ok}, data_quality=_dq(valid=False))
    assert s3.overall_health() is ModelHealth.DEGRADED
    s4 = _snap(var={"HISTORICAL:0.95:1": failed(_meta("historical_var"), "boom")},
               data_quality=_dq(valid=False))
    assert s4.overall_health() is ModelHealth.FAILED


def test_overall_health_empty_snapshot_is_unavailable_and_greeks_carry_no_health():
    g = PortfolioGreeks(net_delta_shares=10.0, delta_adjusted_notional=1000.0,
                        net_gamma=0.0, net_theta_per_day=0.0, net_vega=0.0, per_position=())
    s = _snap(greeks=g)
    assert s.health_summary() == {}
    assert s.overall_health() is ModelHealth.UNAVAILABLE


def test_health_summary_duck_types_sibling_results():
    # Sibling result dataclasses (drawdown/contribution/…) expose `.health: ModelHealth`;
    # a stand-in object with the same attribute is picked up under its field key.
    class _Stub:
        health = ModelHealth.DEGRADED

    s = _snap(drawdown=_Stub(), contributions_es=_Stub())  # type: ignore[arg-type]
    assert s.health_summary() == {"drawdown": ModelHealth.DEGRADED,
                                  "contributions_es": ModelHealth.DEGRADED}
    assert s.overall_health() is ModelHealth.DEGRADED
