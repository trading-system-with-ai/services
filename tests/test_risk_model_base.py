"""Tests for the risk model base (contract §2.2; spec §4, §41, §44, §57, §70).

Pins: registry duplicate/replace, worst-of health ordering, validate can
never upgrade (contract §3 invariant 8), ModelResult/ModelMeta immutability
and shape rules, ModelMeta fields per spec §44, mode defaults (SHADOW).
"""
from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from libs.trading_core.risk.models import base
from libs.trading_core.risk.models.base import (
    REGISTRY,
    BaseRiskModel,
    ModelHealth,
    ModelMeta,
    ModelMode,
    ModelResult,
    RiskModel,
    active,
    clear_for_tests,
    combine_health,
    degraded,
    downgrade,
    failed,
    get,
    health_rank,
    names,
    register,
    unavailable,
    validate_never_upgrades,
)


META = ModelMeta(model_name="toy", model_version="1.0.0", params={"k": 3})


class ToyModel(BaseRiskModel):
    name = "toy"
    version = "1.0.0"
    distribution = "EMPIRICAL"
    return_type = "SIMPLE"
    frequency = "1D"
    data_source = "stock_bars_daily"

    def __init__(self, *, min_obs: int = 60, mode: ModelMode | None = None) -> None:
        super().__init__(mode=mode)
        self.min_obs = min_obs

    def params(self):
        return {"min_obs": self.min_obs}

    def calculate(self, pnl, *, as_of=None) -> ModelResult:  # type: ignore[override]
        n = len(pnl)
        meta = self.metadata(lookback=n or None, as_of=as_of, confidence=0.95, horizon_days=1)
        if n < self.min_obs:
            return unavailable(meta, f"n={n} < min_obs={self.min_obs}", n)
        return active(meta, -min(pnl), n, diagnostics={"n": n})


class OtherToy(ToyModel):
    name = "toy2"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Empty the registry for this module, then RESTORE what was there.

    `clear_for_tests()` empties a process-global registry that the risk
    modules populate at IMPORT time. Clearing it on the way out therefore
    leaked an empty registry into every test module collected after this
    one — `tests/test_risk_garch.py::test_the_model_is_registered_in_
    research_mode` fails when run after this file. Alphabetical collection
    hid it in the full suite, but any explicit ordering reproduced it.
    Snapshot-and-restore keeps this module's isolation while leaving the
    global registry exactly as it was found.
    """
    from libs.trading_core.risk.models.base import REGISTRY

    saved = dict(REGISTRY)
    clear_for_tests()
    try:
        yield
    finally:
        clear_for_tests()
        REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Enums / ordering
# ---------------------------------------------------------------------------


def test_health_enum_values_and_rank():
    # spec §41 states, worst-of rank ACTIVE=0 < DEGRADED=1 < UNAVAILABLE=2 < FAILED=3
    assert [h.value for h in ModelHealth] == ["ACTIVE", "DEGRADED", "UNAVAILABLE", "FAILED"]
    assert [health_rank(h) for h in ModelHealth] == [0, 1, 2, 3]
    assert ModelHealth.ACTIVE == "ACTIVE"  # StrEnum: audit-exact string
    assert [m.value for m in ModelMode] == ["RESEARCH", "SHADOW", "PRODUCTION"]


def test_combine_health_worst_of_ordering():
    A, D, U, F = ModelHealth.ACTIVE, ModelHealth.DEGRADED, ModelHealth.UNAVAILABLE, ModelHealth.FAILED
    assert combine_health(A) is A
    assert combine_health(A, A) is A
    assert combine_health(A, D) is D
    assert combine_health(D, A) is D  # order-independent
    assert combine_health(A, D, U) is U
    assert combine_health(U, D) is U
    assert combine_health(A, D, U, F) is F
    assert combine_health(F, A) is F
    assert combine_health("DEGRADED", "ACTIVE") is D  # accepts raw strings
    with pytest.raises(ValueError):
        combine_health()


# ---------------------------------------------------------------------------
# ModelMeta (spec §44)
# ---------------------------------------------------------------------------


def test_model_meta_has_every_spec_44_field():
    # spec §44 store list -> contract field names:
    # model_name, model_version, parameters->params, data_window->lookback (+ frequency,
    # return_type), data_source, as_of_timestamp->as_of, confidence_level->confidence,
    # horizon->horizon_days, distribution.  (diagnostics live on ModelResult.)
    names_ = {f.name for f in dataclasses.fields(ModelMeta)}
    spec_44 = {
        "model_name", "model_version", "params", "return_type", "frequency",
        "lookback", "data_source", "as_of", "confidence", "horizon_days", "distribution",
    }
    # Every §44 field is still present and still spelled the same...
    assert spec_44 <= names_
    # ...and the ONLY thing beyond them is the §5 tier, which is additive and
    # defaults to None. Pinned as an exact set so a future field cannot be
    # added to the provenance record without this test being updated.
    assert names_ - spec_44 == {"tier"}
    m = ModelMeta(
        model_name="historical_var", model_version="1.0.0",
        params={"confidence": 0.95, "horizon_days": 1, "lookback": 250},
        return_type="SIMPLE", frequency="1D", lookback=250,
        data_source="stock_bars_daily", as_of=date(2026, 8, 17),
        confidence=0.95, horizon_days=1, distribution="EMPIRICAL",
    )
    assert m.lookback == 250 and m.as_of == date(2026, 8, 17) and m.distribution == "EMPIRICAL"


def test_model_meta_frozen_and_params_defensively_copied():
    src = {"a": 1}
    m = ModelMeta("x", "1.0.0", params=src)
    src["a"] = 999  # caller mutation must not leak into recorded provenance
    assert m.params == {"a": 1}
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.model_name = "y"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"model_name": ""},
        {"model_version": ""},
        {"confidence": 0.5},   # boundary excluded: (0.5, 1)
        {"confidence": 1.0},
        {"confidence": 0.3},
        {"horizon_days": 0},
        {"lookback": 0},
    ],
)
def test_model_meta_rejects_malformed(kw):
    base_kw = {"model_name": "x", "model_version": "1.0.0"}
    base_kw.update(kw)
    with pytest.raises(ValueError):
        ModelMeta(**base_kw)


# ---------------------------------------------------------------------------
# ModelResult shape rules
# ---------------------------------------------------------------------------


def test_model_result_immutable_and_diagnostics_default_empty():
    r = ModelResult(value=12.5, health=ModelHealth.ACTIVE, reason=None, sample_size=60, meta=META)
    assert r.diagnostics == {}
    assert r.is_available is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.value = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.health = ModelHealth.FAILED  # type: ignore[misc]
    # diagnostics defensively copied
    d = {"n": 60}
    r2 = ModelResult(value=1.0, health="ACTIVE", reason=None, sample_size=60, meta=META, diagnostics=d)
    d["n"] = 0
    assert r2.diagnostics == {"n": 60}
    assert r2.health is ModelHealth.ACTIVE  # raw string coerced to enum
    # equality is by value (frozen dataclass)
    assert r == ModelResult(value=12.5, health="ACTIVE", reason=None, sample_size=60, meta=META)


def test_model_result_honest_null_rules():
    # UNAVAILABLE / FAILED must carry value None + a reason; non-ACTIVE needs a reason
    with pytest.raises(ValueError):
        ModelResult(value=0.0, health=ModelHealth.UNAVAILABLE, reason="n=1 < 60", sample_size=1, meta=META)
    with pytest.raises(ValueError):
        ModelResult(value=None, health=ModelHealth.UNAVAILABLE, reason=None, sample_size=1, meta=META)
    with pytest.raises(ValueError):
        ModelResult(value=1.0, health=ModelHealth.DEGRADED, reason="", sample_size=61, meta=META)
    with pytest.raises(ValueError):
        ModelResult(value=float("nan"), health=ModelHealth.ACTIVE, reason=None, sample_size=61, meta=META)
    with pytest.raises(ValueError):
        ModelResult(value=1.0, health=ModelHealth.ACTIVE, reason=None, sample_size=-1, meta=META)
    with pytest.raises(ValueError):
        ModelResult(value=1.0, health="BOGUS", reason=None, sample_size=1, meta=META)
    # negative value is legal (tail of losses can be a gain, contract §1: no flooring at 0)
    r = ModelResult(value=-3.0, health=ModelHealth.ACTIVE, reason=None, sample_size=61, meta=META)
    assert r.value == -3.0


def test_constructors_uniform_shapes():
    a = active(META, 5.0, 100, diagnostics={"tail_size": 5})
    assert (a.value, a.health, a.reason, a.sample_size, a.diagnostics) == (5.0, ModelHealth.ACTIVE, None, 100, {"tail_size": 5})
    d = degraded(META, "small tail: k=2", 4.0, 70)
    assert (d.value, d.health, d.reason, d.sample_size) == (4.0, ModelHealth.DEGRADED, "small tail: k=2", 70)
    u = unavailable(META, "n=17 < min_obs=60", 17)
    assert (u.value, u.health, u.reason, u.sample_size) == (None, ModelHealth.UNAVAILABLE, "n=17 < min_obs=60", 17)
    assert u.is_available is False
    f = failed(META, "estimator raised ZeroDivisionError")
    assert (f.value, f.health, f.sample_size) == (None, ModelHealth.FAILED, 0)
    with pytest.raises(ValueError):
        unavailable(META, "", 17)
    with pytest.raises(ValueError):
        degraded(META, "x", None, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        active(META, None, 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate never upgrades (contract §3.8, spec §57)
# ---------------------------------------------------------------------------


def test_validate_never_upgrades_guard():
    a = active(META, 5.0, 100)
    d = degraded(META, "small tail", 5.0, 100)
    u = unavailable(META, "n=10 < 60", 10)
    # same or worse: allowed, returns `after`
    assert validate_never_upgrades(a, a) is a
    assert validate_never_upgrades(a, d) is d
    assert validate_never_upgrades(d, u) is u
    assert validate_never_upgrades(a, u) is u
    # any upgrade raises
    with pytest.raises(ValueError):
        validate_never_upgrades(d, a)
    with pytest.raises(ValueError):
        validate_never_upgrades(u, d)
    with pytest.raises(ValueError):
        validate_never_upgrades(u, a)
    # cannot conjure a value: before UNAVAILABLE(None) -> after FAILED... value None ok
    f = failed(META, "boom")
    assert validate_never_upgrades(u, f) is f


def test_downgrade_helper():
    a = active(META, 5.0, 100)
    d = downgrade(a, ModelHealth.DEGRADED, "kupiec p=0.02")
    assert d.health is ModelHealth.DEGRADED and d.value == 5.0 and d.reason == "kupiec p=0.02"
    d2 = downgrade(d, ModelHealth.DEGRADED, "dispersion 1.8 > 1.5")
    assert d2.reason == "kupiec p=0.02; dispersion 1.8 > 1.5"
    u = downgrade(d2, ModelHealth.UNAVAILABLE, "stale")
    assert u.health is ModelHealth.UNAVAILABLE and u.value is None
    # asking for a BETTER health is a no-op on health (worst-of), never an upgrade
    still = downgrade(u, ModelHealth.ACTIVE, "note")
    assert still.health is ModelHealth.UNAVAILABLE and still.value is None
    with pytest.raises(ValueError):
        downgrade(a, ModelHealth.DEGRADED, "")
    # meta preserved untouched
    assert u.meta == META


def test_base_model_validate_is_identity_and_subclass_may_downgrade():
    m = ToyModel()
    r = m.calculate([1.0, -2.0, 3.0] * 30)  # n=90 >= 60 -> ACTIVE, value = -min = 2.0
    assert r.health is ModelHealth.ACTIVE and r.value == 2.0 and r.sample_size == 90
    assert m.validate(r) is r
    assert m.diagnostics(r) == {"n": 90}

    class Strict(ToyModel):
        name = "strict"

        def validate(self, result):
            return validate_never_upgrades(result, downgrade(result, ModelHealth.DEGRADED, "shadow-only"))

    class Cheater(ToyModel):
        name = "cheat"

        def validate(self, result):
            return validate_never_upgrades(result, active(result.meta, 1.0, result.sample_size))

    assert Strict().validate(r).health is ModelHealth.DEGRADED
    u = ToyModel().calculate([1.0] * 5)  # n=5 < 60 -> UNAVAILABLE
    assert u.health is ModelHealth.UNAVAILABLE and u.value is None and u.reason == "n=5 < min_obs=60"
    with pytest.raises(ValueError):
        Cheater().validate(u)


# ---------------------------------------------------------------------------
# BaseRiskModel metadata / mode
# ---------------------------------------------------------------------------


def test_base_model_defaults_shadow_and_metadata_from_class_attrs():
    m = ToyModel(min_obs=30)
    assert m.mode is ModelMode.SHADOW  # spec §70: never a decision input until promoted
    assert isinstance(m, RiskModel)   # runtime Protocol check
    meta = m.metadata()
    assert meta.model_name == "toy" and meta.model_version == "1.0.0"
    assert meta.params == {"min_obs": 30}
    assert (meta.return_type, meta.frequency, meta.data_source, meta.distribution) == (
        "SIMPLE", "1D", "stock_bars_daily", "EMPIRICAL")
    assert meta.as_of is None and meta.lookback is None
    # per-call overrides; params overrides MERGE
    meta2 = m.metadata(as_of=date(2026, 8, 17), lookback=250, confidence=0.99, horizon_days=10,
                       params={"lambda": 0.94})
    assert meta2.params == {"min_obs": 30, "lambda": 0.94}
    assert (meta2.as_of, meta2.lookback, meta2.confidence, meta2.horizon_days) == (
        date(2026, 8, 17), 250, 0.99, 10)
    # explicit mode
    assert ToyModel(mode=ModelMode.PRODUCTION).mode is ModelMode.PRODUCTION
    assert ToyModel(mode="RESEARCH").mode is ModelMode.RESEARCH
    with pytest.raises(ValueError):
        ToyModel(mode="LIVE")
    # class-level default is untouched by an instance override
    assert ToyModel.mode is ModelMode.SHADOW


def test_base_model_requires_name_and_calculate():
    class NoName(BaseRiskModel):
        def calculate(self, *a, **k):
            raise NotImplementedError

    with pytest.raises(ValueError):
        NoName()

    class NoCalc(BaseRiskModel):
        name = "nocalc"

    with pytest.raises(TypeError):  # abstract method not implemented
        NoCalc()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_get_names_duplicate_replace():
    assert names() == ()
    m1 = register(ToyModel())
    assert get("toy") is m1
    assert names() == ("toy",)
    with pytest.raises(ValueError):
        register(ToyModel())  # duplicate name
    assert get("toy") is m1  # unchanged after failed register
    m2 = register(ToyModel(min_obs=10), replace=True)
    assert get("toy") is m2 and m2 is not m1
    register(OtherToy())
    assert names() == ("toy", "toy2")  # sorted
    assert REGISTRY["toy2"].name == "toy2"
    with pytest.raises(KeyError):
        get("missing")
    clear_for_tests()
    assert names() == () and REGISTRY == {}


def test_registry_rejects_non_models():
    with pytest.raises(ValueError):
        register(object())  # type: ignore[arg-type]

    class Bare:
        name = ""
        version = "1.0.0"
        mode = ModelMode.SHADOW

        def calculate(self, *a, **k): ...
        def validate(self, r): return r
        def diagnostics(self, r): return {}
        def metadata(self): return META

    with pytest.raises(ValueError):
        register(Bare())  # empty name


def test_package_init_reexports_every_base_name_unchanged():
    """The package re-exports the base abstractions as the SAME objects.

    The package ``__all__`` is a SUPERSET of ``base.__all__``: the Phase B
    integration also exports the model modules' public names (contract
    §2.1–§2.11), so callers need not know the module layout. What must never
    drift is identity — ``pkg.ModelHealth`` has to BE ``base.ModelHealth``,
    or two enums would compare unequal across modules.
    """
    import libs.trading_core.risk.models as pkg

    for n in ("ModelHealth", "ModelMode", "ModelMeta", "ModelResult", "RiskModel",
              "BaseRiskModel", "register", "get", "names", "clear_for_tests",
              "unavailable", "degraded", "combine_health", "validate_never_upgrades"):
        assert getattr(pkg, n) is getattr(base, n)
    missing = set(base.__all__) - set(pkg.__all__)
    assert not missing, f"package init dropped base exports: {sorted(missing)}"
    for n in base.__all__:
        assert getattr(pkg, n) is getattr(base, n)
