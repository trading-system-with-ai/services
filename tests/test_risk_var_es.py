"""VaR & ES estimators — hand-checked (Phase B contract §2.3, invariants §3).

Every number below is computed by hand in a comment; nothing is asserted
against "whatever the code returns". The reference series is

    PNL = [-50, -30, -20, -10, -5, 5, 10, 20, 30, 120]          (n = 10)

whose losses L = -pnl sorted DESCENDING are

    L = [50, 30, 20, 10, 5, -5, -10, -20, -30, -120]

so L(1)=50, L(2)=30, L(3)=20, L(4)=10, L(5)=5, L(6)=-5, ...

Sample moments (ddof=1):
    Sum(pnl) = -50-30-20-10-5+5+10+20+30+120 = 70   =>  mu = 70/10 = 7
    deviations d = pnl - 7 = [-57,-37,-27,-17,-12,-2,3,13,23,113]
    Sum d^2 = 3249+1369+729+289+144+4+9+169+529+12769 = 19260
    var = 19260/9 = 2140            sigma = sqrt(2140) = 46.26013402488151
"""
from __future__ import annotations

import math
from datetime import date
from statistics import NormalDist

import pytest

from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMode,
    clear_for_tests,
    get,
    names,
    validate_never_upgrades,
)
from libs.trading_core.risk.models.var_es import (
    DEFAULT_MIN_OBS_95,
    DEFAULT_MIN_OBS_99,
    DISTRIBUTION_EMPIRICAL,
    DISTRIBUTION_EMPIRICAL_VOL_SCALED,
    DISTRIBUTION_NORMAL,
    MODEL_VERSION,
    SCALING_NONE,
    SCALING_SQRT_TIME,
    GaussianESModel,
    GaussianVaRModel,
    HistoricalESModel,
    HistoricalVaRModel,
    conditional_es,
    conditional_var,
    default_min_obs,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    register_models,
    sorted_losses,
    tail_size,
)

PNL = [-50.0, -30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0, 120.0]
LOSSES_DESC = [50.0, 30.0, 20.0, 10.0, 5.0, -5.0, -10.0, -20.0, -30.0, -120.0]
MU = 7.0
SIGMA = math.sqrt(2140.0)  # = 46.26013402488151

# min_obs=5 lets the 10-point series produce a number; n=10 >= 2*5 => ACTIVE.
SMALL = {"min_obs": 5}


# ---------------------------------------------------------------------------
# Primitives: loss ordering and tail size
# ---------------------------------------------------------------------------


def test_sorted_losses_negates_and_sorts_descending():
    # L = -pnl, descending. Hand-listed at the top of this module.
    assert sorted_losses(PNL) == LOSSES_DESC


def test_tail_size_matches_contract_worked_examples():
    # Contract 2.3 states these exactly: n=600 -> 30 @95%, 6 @99%;
    # n=250 -> 13 @95%, 3 @99%. Naive float ceil would give 31 and 7 for
    # n=600 because 600*(1-0.95) = 30.00000000000003 in binary floating
    # point; tail_size snaps that back to the exact integer.
    assert tail_size(600, 0.95) == 30      # 600 * 0.05 = 30 exactly
    assert tail_size(600, 0.99) == 6       # 600 * 0.01 = 6 exactly
    assert tail_size(250, 0.95) == 13      # 250 * 0.05 = 12.5 -> ceil 13
    assert tail_size(250, 0.99) == 3       # 250 * 0.01 = 2.5  -> ceil 3
    assert tail_size(100, 0.95) == 5       # 100 * 0.05 = 5 exactly
    assert tail_size(1000, 0.99) == 10     # 1000 * 0.01 = 10 exactly
    assert tail_size(10, 0.95) == 1        # 10 * 0.05 = 0.5 -> ceil 1
    assert tail_size(10, 0.7) == 3         # 10 * 0.3 = 3 exactly
    assert tail_size(10, 0.8) == 2         # 10 * 0.2 = 2 exactly


def test_tail_size_clamped_to_at_least_one_and_at_most_n():
    assert tail_size(0, 0.95) == 0         # no observations, no tail
    assert tail_size(1, 0.95) == 1         # ceil(0.05) = 1
    assert tail_size(3, 0.6) == 2          # 3 * 0.4 = 1.2 -> ceil 2 (<= n)


# ---------------------------------------------------------------------------
# Historical VaR / ES — hand-computed
# ---------------------------------------------------------------------------


def test_historical_var_and_es_at_70_pct_hand_computed():
    # k = ceil(10 * 0.30) = 3.
    # VaR = L(3) = 20.
    # ES  = (L1 + L2 + L3)/3 = (50 + 30 + 20)/3 = 100/3 = 33.3333...
    var = historical_var(PNL, 0.70, **SMALL)
    es = historical_es(PNL, 0.70, **SMALL)
    assert var.value == 20.0
    assert es.value == pytest.approx(100.0 / 3.0, rel=0, abs=1e-12)
    assert var.diagnostics["tail_size"] == 3
    assert es.diagnostics["tail_size"] == 3
    assert var.health is ModelHealth.ACTIVE and var.reason is None


def test_historical_var_and_es_at_80_pct_hand_computed():
    # k = ceil(10 * 0.20) = 2.
    # VaR = L(2) = 30.  ES = (50 + 30)/2 = 40.
    assert historical_var(PNL, 0.80, **SMALL).value == 30.0
    assert historical_es(PNL, 0.80, **SMALL).value == 40.0


def test_historical_var_and_es_at_95_and_99_k_equals_one():
    # n=10: k95 = ceil(10*0.05) = ceil(0.5) = 1; k99 = ceil(0.1) = 1.
    # With k=1 both VaR and ES are the single worst loss L(1) = 50, and
    # ES == VaR exactly (contract 3.1: equality iff k=1 or ties).
    for alpha in (0.95, 0.99):
        var = historical_var(PNL, alpha, **SMALL)
        es = historical_es(PNL, alpha, **SMALL)
        assert var.diagnostics["tail_size"] == 1
        assert var.value == 50.0
        assert es.value == 50.0


def test_historical_tail_can_be_negative_and_is_reported_honestly():
    # An all-gains book: pnl = [1..10] => losses desc = [-1,-2,...,-10].
    # k = ceil(10*0.3) = 3 => VaR = L(3) = -3, ES = (-1-2-3)/3 = -2.
    # Contract 1: the negative number is reported, never floored at 0.
    gains = [float(i) for i in range(1, 11)]
    assert historical_var(gains, 0.70, **SMALL).value == -3.0
    assert historical_es(gains, 0.70, **SMALL).value == -2.0


# ---------------------------------------------------------------------------
# Gaussian VaR / ES — hand-computed against NormalDist
# ---------------------------------------------------------------------------


def test_gaussian_var_and_es_at_95_hand_computed():
    # mu = 7, sigma = sqrt(2140) = 46.26013402488151.
    # z_0.95 = NormalDist().inv_cdf(0.95) = 1.6448536269514715
    # VaR = -7 + 1.6448536269514715 * 46.26013402488151 = 69.09114923408752
    # phi(z) = 0.10314917819929559; /0.05 = 2.0629835639859117
    # ES  = -7 + 46.26013402488151 * 2.0629835639859117 = 88.42137093013326
    z = NormalDist().inv_cdf(0.95)
    assert z == pytest.approx(1.6448536269514715, abs=1e-15)
    var = gaussian_var(PNL, 0.95, **SMALL)
    es = gaussian_es(PNL, 0.95, **SMALL)
    assert var.value == pytest.approx(69.09114923408752, abs=1e-12)
    assert es.value == pytest.approx(88.42137093013326, abs=1e-12)
    # and the closed forms restated independently:
    assert var.value == pytest.approx(-MU + z * SIGMA, abs=1e-12)
    assert es.value == pytest.approx(
        -MU + SIGMA * NormalDist().pdf(z) / 0.05, abs=1e-12
    )
    assert var.diagnostics["mu"] == pytest.approx(7.0, abs=1e-12)
    assert var.diagnostics["sigma"] == pytest.approx(SIGMA, abs=1e-12)
    assert var.diagnostics["z"] == pytest.approx(z, abs=1e-15)


def test_gaussian_var_and_es_at_99_hand_computed():
    # z_0.99 = 2.3263478740408408
    # VaR = -7 + 2.3263478740408408 * 46.26013402488151 = 100.61716444162747
    # ES  = -7 + 46.26013402488151 * phi(z)/0.01       = 116.29316703821705
    var = gaussian_var(PNL, 0.99, min_obs=5)
    es = gaussian_es(PNL, 0.99, min_obs=5)
    assert var.value == pytest.approx(100.61716444162747, abs=1e-12)
    assert es.value == pytest.approx(116.29316703821705, abs=1e-12)


def test_gaussian_uses_ddof_one():
    # If ddof were 0 the variance would be 19260/10 = 1926 (sigma = 43.886...)
    # rather than 19260/9 = 2140 (sigma = 46.260...). Pin the ddof=1 value.
    assert gaussian_var(PNL, 0.95, **SMALL).diagnostics["sigma"] == pytest.approx(
        math.sqrt(19260.0 / 9.0), abs=1e-12
    )


# ---------------------------------------------------------------------------
# Contract 3 invariants
# ---------------------------------------------------------------------------


def test_invariant_1_es_ge_var_historical_and_gaussian():
    # Contract 3.1 on several series and confidences.
    series = [
        PNL,
        [float(i) for i in range(1, 11)],
        [-1.0, -1.0, -1.0, 2.0, 2.0, 2.0, -8.0, 4.0, 0.0, 1.0],
    ]
    for pnl in series:
        for alpha in (0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
            hv = historical_var(pnl, alpha, **SMALL).value
            he = historical_es(pnl, alpha, **SMALL).value
            gv = gaussian_var(pnl, alpha, **SMALL).value
            ge = gaussian_es(pnl, alpha, **SMALL).value
            assert he >= hv - 1e-12, (pnl, alpha, he, hv)
            assert ge >= gv - 1e-12, (pnl, alpha, ge, gv)


def test_invariant_2_monotone_in_confidence():
    # Contract 3.2: VaR_0.99 >= VaR_0.95 and ES_0.99 >= ES_0.95.
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        lo = fn(PNL, 0.95, **SMALL).value
        hi = fn(PNL, 0.99, **SMALL).value
        assert hi >= lo - 1e-12, (fn.__name__, lo, hi)
    # A denser grid on the historical pair (step functions, still monotone).
    for fn in (historical_var, historical_es):
        vals = [fn(PNL, a, **SMALL).value for a in (0.55, 0.6, 0.7, 0.8, 0.9, 0.95)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_invariant_4_scale_by_k():
    # Contract 3.4: scaling the P&L by k scales VaR/ES by k (k > 0).
    k = 3.5
    scaled = [k * p for p in PNL]
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        base = fn(PNL, 0.80, **SMALL).value
        got = fn(scaled, 0.80, **SMALL).value
        assert got == pytest.approx(k * base, rel=1e-12), fn.__name__


def test_invariant_4_shift_by_constant():
    # Contract 3.4: adding a constant gain c to every P&L shifts the loss
    # distribution down by c, so VaR/ES fall by exactly c (historical: the
    # order of losses is unchanged; Gaussian: mu -> mu + c, sigma unchanged).
    c = 11.0
    shifted = [p + c for p in PNL]
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        base = fn(PNL, 0.80, **SMALL).value
        got = fn(shifted, 0.80, **SMALL).value
        assert got == pytest.approx(base - c, abs=1e-12), fn.__name__
    # Gaussian sigma is genuinely untouched by the shift.
    assert gaussian_var(shifted, 0.80, **SMALL).diagnostics["sigma"] == pytest.approx(
        SIGMA, abs=1e-12
    )


def test_invariant_6_below_min_obs_is_unavailable_not_an_exception():
    # Contract 3.6: value None, health UNAVAILABLE, reason with real numbers.
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        r = fn(PNL, 0.95)  # default min_obs = 60 at 95%, n = 10
        assert r.value is None
        assert r.health is ModelHealth.UNAVAILABLE
        assert r.reason and "n=10" in r.reason and "min_obs=60" in r.reason
        assert r.sample_size == 10


# ---------------------------------------------------------------------------
# Horizon scaling
# ---------------------------------------------------------------------------


def test_sqrt_time_scaling_historical_labelled():
    # Historical: VaR_h = VaR_1 * sqrt(h). At 80%, VaR_1 = 30, h = 4 =>
    # 30 * 2 = 60. ES_1 = 40 => 80.
    v1 = historical_var(PNL, 0.80, **SMALL)
    v4 = historical_var(PNL, 0.80, 4, **SMALL)
    assert v1.diagnostics["scaling"] == SCALING_NONE
    assert v4.diagnostics["scaling"] == SCALING_SQRT_TIME
    assert v4.meta.params["scaling"] == SCALING_SQRT_TIME
    assert v4.meta.horizon_days == 4
    assert v4.value == pytest.approx(30.0 * 2.0, abs=1e-12) == pytest.approx(60.0)
    assert historical_es(PNL, 0.80, 4, **SMALL).value == pytest.approx(80.0, abs=1e-12)
    # A non-square horizon still scales by sqrt(h): h=2 => 30*sqrt(2).
    assert historical_var(PNL, 0.80, 2, **SMALL).value == pytest.approx(
        30.0 * math.sqrt(2.0), abs=1e-12
    )
    assert v4.meta.distribution == DISTRIBUTION_EMPIRICAL


def test_sqrt_time_scaling_gaussian_splits_drift_and_shock():
    # Gaussian: VaR_h = -mu*h + z*sigma*sqrt(h). With mu=7, h=4, alpha=0.95:
    #   -7*4 + 1.6448536269514715 * 46.26013402488151 * 2
    # = -28 + 152.18229846817504 = 124.18229846817504
    z = NormalDist().inv_cdf(0.95)
    expected = -MU * 4 + z * SIGMA * 2.0
    assert expected == pytest.approx(124.18229846817504, abs=1e-11)
    got = gaussian_var(PNL, 0.95, 4, **SMALL)
    assert got.value == pytest.approx(expected, abs=1e-11)
    assert got.diagnostics["scaling"] == SCALING_SQRT_TIME
    # ES_h = -mu*h + sigma*sqrt(h)*phi(z)/(1-alpha)
    es_expected = -MU * 4 + SIGMA * 2.0 * NormalDist().pdf(z) / 0.05
    assert gaussian_es(PNL, 0.95, 4, **SMALL).value == pytest.approx(
        es_expected, abs=1e-11
    )
    # The 1-day number is preserved in diagnostics for reproducibility.
    assert got.diagnostics["one_day"] == pytest.approx(-MU + z * SIGMA, abs=1e-12)


def test_horizon_one_is_unscaled():
    for fn in (historical_var, gaussian_var):
        assert fn(PNL, 0.80, 1, **SMALL).value == pytest.approx(
            fn(PNL, 0.80, **SMALL).value, abs=1e-15
        )


# ---------------------------------------------------------------------------
# min_obs defaults, DEGRADED band, health text
# ---------------------------------------------------------------------------


def test_default_min_obs_steps_up_at_99():
    assert default_min_obs(0.95) == DEFAULT_MIN_OBS_95 == 60
    assert default_min_obs(0.975) == 60
    assert default_min_obs(0.99) == DEFAULT_MIN_OBS_99 == 250
    assert default_min_obs(0.995) == 250
    # 250 at 99% is chosen so the tail holds at least 3 observations.
    assert tail_size(250, 0.99) == 3


def test_degraded_band_between_min_obs_and_twice_min_obs():
    # min_obs=60 at 95%: n=80 is in [60, 120) => DEGRADED with a real reason;
    # n=120 = 2*60 => ACTIVE. Build deterministic series of each length.
    n80 = [float((i % 7) - 3) for i in range(80)]
    n120 = [float((i % 7) - 3) for i in range(120)]
    d = historical_var(n80, 0.95)
    assert d.value is not None                      # a number WAS produced
    assert d.health is ModelHealth.DEGRADED
    assert "n=80" in d.reason and "120" in d.reason and "k=" in d.reason
    a = historical_var(n120, 0.95)
    assert a.health is ModelHealth.ACTIVE and a.reason is None
    # boundary: n=60 exactly is DEGRADED (>= min_obs, < 2*min_obs)
    n60 = [float((i % 7) - 3) for i in range(60)]
    assert historical_var(n60, 0.95).health is ModelHealth.DEGRADED
    # n=59 is UNAVAILABLE
    n59 = [float((i % 7) - 3) for i in range(59)]
    u = historical_var(n59, 0.95)
    assert u.health is ModelHealth.UNAVAILABLE and u.value is None
    assert "n=59" in u.reason and "min_obs=60" in u.reason


def test_diagnostics_shape_on_every_estimator():
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        d = fn(PNL, 0.80, **SMALL).diagnostics
        assert d["n"] == 10
        assert d["tail_size"] == 2
        assert d["mu"] == pytest.approx(7.0, abs=1e-12)
        assert d["sigma"] == pytest.approx(SIGMA, abs=1e-12)
        assert d["scaling"] == SCALING_NONE
        assert d["horizon_days"] == 1


def test_unavailable_result_still_carries_diagnostics_and_meta():
    r = historical_var(PNL, 0.95)  # n=10 < 60
    assert r.diagnostics["n"] == 10 and r.diagnostics["tail_size"] == 1
    assert r.meta.model_name == "historical_var"
    assert r.meta.model_version == MODEL_VERSION
    assert r.meta.confidence == 0.95 and r.meta.horizon_days == 1
    assert r.meta.distribution == DISTRIBUTION_EMPIRICAL


def test_meta_records_params_for_reproducibility():
    r = gaussian_es(PNL, 0.95, 3, min_obs=5, as_of=date(2026, 8, 18))
    assert r.meta.params == {
        "confidence": 0.95,
        "horizon_days": 3,
        "min_obs": 5,
        "scaling": SCALING_SQRT_TIME,
    }
    assert r.meta.as_of == date(2026, 8, 18)
    assert r.meta.distribution == DISTRIBUTION_NORMAL
    assert r.meta.lookback == 10
    assert r.sample_size == 10


# ---------------------------------------------------------------------------
# Malformed input raises; missing data does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 0.4, 0.0, 1.0, 1.5, -0.2, float("nan")])
def test_confidence_outside_range_raises(alpha):
    with pytest.raises(ValueError):
        historical_var(PNL, alpha, **SMALL)


@pytest.mark.parametrize("h", [0, -1, 1.5, True])
def test_bad_horizon_raises(h):
    with pytest.raises(ValueError):
        historical_var(PNL, 0.95, h, **SMALL)


def test_non_finite_pnl_raises():
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            historical_var([*PNL[:-1], bad], 0.80, **SMALL)


def test_bad_min_obs_raises():
    with pytest.raises(ValueError):
        historical_var(PNL, 0.80, min_obs=0)


def test_empty_series_is_unavailable_not_an_exception():
    r = historical_var([], 0.95, min_obs=1)
    assert r.value is None and r.health is ModelHealth.UNAVAILABLE
    assert r.sample_size == 0


# ---------------------------------------------------------------------------
# Conditional (filtered-HS) VaR / ES
# ---------------------------------------------------------------------------


def _alternating(n: int, amp: float = 100.0) -> list[float]:
    """Constant-|pnl| alternating series: EWMA sigma is constant, so the
    Hull-White ratios sigma_now/sigma_t are all exactly 1."""
    return [amp if i % 2 == 0 else -amp for i in range(n)]


def test_conditional_var_equals_plain_hs_when_volatility_is_constant():
    # With |pnl| constant the EWMA seed is amp^2 and the recursion's fixed
    # point is amp^2, so sigma_t == sigma_now for every t and the scaled
    # series equals pnl[init_obs:] exactly. Conditional VaR must then equal
    # historical VaR on that tail.
    pnl = _alternating(40, 100.0)
    tail = pnl[10:]  # init_obs = 10
    cond = conditional_var(pnl, 0.80, lam=0.94, init_obs=10, min_obs=5)
    plain = historical_var(tail, 0.80, min_obs=5)
    assert cond.value == pytest.approx(plain.value, abs=1e-9)
    # losses of the 30-point tail: 15 entries of +100, 15 of -100.
    # k = ceil(30*0.2) = 6 => VaR = L(6) = 100.
    assert cond.diagnostics["tail_size"] == 6
    assert cond.value == pytest.approx(100.0, abs=1e-9)


def test_conditional_labels_distribution_and_lambda():
    pnl = _alternating(40, 100.0)
    for fn in (conditional_var, conditional_es):
        r = fn(pnl, 0.80, lam=0.9, init_obs=10, min_obs=5)
        assert r.meta.distribution == DISTRIBUTION_EMPIRICAL_VOL_SCALED
        assert r.meta.params["lambda"] == 0.9
        assert r.meta.params["init_obs"] == 10
        # bookkeeping: 40 inputs, 10 dropped as warm-up, 30 used.
        assert r.diagnostics["n_input"] == 40
        assert r.diagnostics["n_scaled"] == 30
        assert r.diagnostics["dropped_warmup"] == 10
        assert r.sample_size == 30


def test_conditional_es_ge_conditional_var():
    pnl = [float(((i * 37) % 23) - 11) for i in range(60)]
    v = conditional_var(pnl, 0.90, init_obs=10, min_obs=5)
    e = conditional_es(pnl, 0.90, init_obs=10, min_obs=5)
    assert e.value >= v.value - 1e-12


def test_conditional_upweights_a_recent_volatility_burst():
    # Quiet history then a loud recent stretch: sigma_now >> sigma_t for the
    # old days, so the old losses are inflated and conditional VaR exceeds
    # plain historical VaR on the same window.
    quiet = [1.0 if i % 2 == 0 else -1.0 for i in range(40)]
    loud = [40.0 if i % 2 == 0 else -40.0 for i in range(10)]
    pnl = quiet + loud
    cond = conditional_var(pnl, 0.90, init_obs=10, min_obs=5)
    plain = historical_var(pnl[10:], 0.90, min_obs=5)
    assert cond.value > plain.value


# ---------------------------------------------------------------------------
# Model classes & registry
# ---------------------------------------------------------------------------


def test_model_classes_are_registered_under_contract_names():
    # register_models() is idempotent, so restoring after another module's
    # clear_for_tests() is safe and required here (test order independence).
    register_models()
    for name, cls in [
        ("historical_var", HistoricalVaRModel),
        ("historical_es", HistoricalESModel),
        ("gaussian_var", GaussianVaRModel),
        ("gaussian_es", GaussianESModel),
    ]:
        m = get(name)
        assert isinstance(m, cls)
        assert m.name == name
        assert m.version == MODEL_VERSION == "1.0.0"
        # spec 70: shadow by default, so it can never feed a veto.
        assert m.mode is ModelMode.SHADOW


def test_register_models_is_idempotent():
    register_models()
    before = names()
    assert register_models() == (
        "gaussian_es",
        "gaussian_var",
        "historical_es",
        "historical_var",
    )
    assert names() == before  # no duplicates, no exception
    # and it restores the registry after a clear (test-isolation friendly)
    clear_for_tests()
    assert names() == ()
    register_models()
    assert set(names()) >= {
        "gaussian_es",
        "gaussian_var",
        "historical_es",
        "historical_var",
    }


def test_model_calculate_matches_the_free_function():
    m = HistoricalVaRModel(confidence=0.80, min_obs=5)
    assert m.calculate(PNL).value == historical_var(PNL, 0.80, min_obs=5).value == 30.0
    g = GaussianESModel(confidence=0.95, horizon_days=4, min_obs=5)
    assert g.calculate(PNL).value == pytest.approx(
        gaussian_es(PNL, 0.95, 4, min_obs=5).value, abs=1e-12
    )
    # params() feeds ModelMeta.params (spec 44 provenance).
    assert m.params() == {"confidence": 0.80, "horizon_days": 1, "min_obs": 5}


def test_model_binds_default_min_obs_from_confidence():
    assert HistoricalVaRModel(confidence=0.95).min_obs == 60
    assert HistoricalVaRModel(confidence=0.99).min_obs == 250


def test_validate_never_upgrades_health():
    # Contract 3.8 / invariant: validate() is identity here and may never
    # turn an UNAVAILABLE calculation into an ACTIVE one.
    m = HistoricalVaRModel(confidence=0.95)  # min_obs 60 > n=10
    r = m.calculate(PNL)
    assert r.health is ModelHealth.UNAVAILABLE
    v = m.validate(r)
    assert v.health is ModelHealth.UNAVAILABLE and v.value is None
    # Attempting an upgrade through the guard is refused.
    good = HistoricalVaRModel(confidence=0.80, min_obs=5).calculate(PNL)
    assert good.health is ModelHealth.ACTIVE
    with pytest.raises(ValueError):
        validate_never_upgrades(r, good)


def test_model_rejects_bad_construction_params():
    with pytest.raises(ValueError):
        HistoricalVaRModel(confidence=1.2)
    with pytest.raises(ValueError):
        GaussianVaRModel(horizon_days=0)
    with pytest.raises(ValueError):
        GaussianESModel(min_obs=-1)
