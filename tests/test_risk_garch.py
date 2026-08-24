"""Tests for ``libs/trading_core/risk/models/garch.py`` — GARCH(1,1)
conditional volatility, RESEARCH (Phase E design §9.3; risk spec §12, §13,
§14, §57, §58, §70).

The centrepiece is a **parameter-recovery test**: a GARCH(1,1) series is
simulated with KNOWN ``(omega, alpha, beta)`` from a seeded stdlib RNG, and
the estimator must recover the two parameters that matter to the tolerance
the design fixes (``|Δalpha| < 0.03``, ``|Δbeta| < 0.05``). That is the only
honest way to test an MLE: not "does it return a number" but "does it return
the RIGHT number when the truth is known".

Everything else pins the properties a risk reviewer would ask about:

- the constraints (``omega > 0``, ``alpha, beta >= 0``, ``alpha+beta < 1``)
  hold on ADVERSARIAL input — a constant series, a handful of observations,
  a series of spikes — because a fit that quietly returns a non-stationary
  parameterisation would poison every forecast built on it;
- the honest-null and DEGRADED paths (spec §13: never fabricate a GARCH
  result), each with a reason carrying real numbers;
- the closed-form multi-step forecast against the iterative recursion it
  claims to replace;
- the half-life arithmetic;
- the FHS scaling shape (identical to the EWMA filter's, so the historical
  VaR/ES estimators consume either unchanged);
- the §13/§58 **fallback**: when GARCH is not ACTIVE the conditional view
  comes from EWMA, and the caller is told why.

The simulator below is deliberately written out in the test rather than
imported: the fit must recover parameters from an INDEPENDENT construction
of the process, not from the library's own recursion.
"""
from __future__ import annotations

import math
import random

import pytest

from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMode,
    ModelResult,
    get,
)
from libs.trading_core.risk.models.garch import (
    DEFAULT_DEGRADED_PERSISTENCE,
    DEFAULT_LJUNG_BOX_LAGS,
    DEFAULT_LJUNG_BOX_P,
    DEFAULT_MIN_OBS,
    DISTRIBUTION_GAUSSIAN_GARCH,
    MODEL_NAME_GARCH,
    PERSISTENCE_MAX,
    Garch11Model,
    GarchFit,
    GarchParams,
    conditional_scaled_pnl_source,
    conditional_volatility_source,
    fit_garch,
    garch_forecast_variance,
    garch_scaled_pnl,
    garch_scaling,
    garch_variance_path,
    garch_volatility_forecast,
    ljung_box,
    SOURCE_EWMA,
    SOURCE_GARCH,
)
from libs.trading_core.risk.models.var_es import historical_es, historical_var
from libs.trading_core.risk.models.volatility import volatility_scaled_pnl

# --- the seeded simulation the design fixes (design §9.3) ------------------
TRUE_OMEGA = 1e-6
TRUE_ALPHA = 0.08
TRUE_BETA = 0.90
SIM_N = 3000
SIM_SEED = 20260818


def simulate_garch(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int,
    *,
    var0: float | None = None,
) -> list[float]:
    """A GARCH(1,1) path with Gaussian innovations, written out independently.

    ``sigma^2_t = omega + alpha r^2_{t-1} + beta sigma^2_{t-1}``,
    ``r_t = sigma_t z_t``, ``z_t ~ N(0,1)`` from ``random.Random(seed)``.
    Seeded from the unconditional variance unless ``var0`` is given (the
    near-integrated cases have no finite unconditional variance to start at).
    """
    rng = random.Random(seed)
    variance = var0 if var0 is not None else omega / (1.0 - alpha - beta)
    out: list[float] = []
    previous = 0.0
    for _ in range(n):
        variance = omega + alpha * previous * previous + beta * variance
        previous = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(previous)
    return out


@pytest.fixture(scope="module")
def simulated() -> list[float]:
    return simulate_garch(SIM_N, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, SIM_SEED)


@pytest.fixture(scope="module")
def simulated_fit(simulated: list[float]) -> GarchFit:
    return fit_garch(simulated)


# ---------------------------------------------------------------------------
# Parameter recovery — the test that actually validates the estimator
# ---------------------------------------------------------------------------


def test_recovers_the_simulated_parameters(simulated_fit: GarchFit) -> None:
    """n=3000, true (omega, alpha, beta) = (1e-6, 0.08, 0.90):
    ``|Δalpha| < 0.03`` and ``|Δbeta| < 0.05`` (design §9.3)."""
    assert simulated_fit.params is not None, simulated_fit.reason
    params = simulated_fit.params
    assert abs(params.alpha - TRUE_ALPHA) < 0.03, f"alpha={params.alpha}"
    assert abs(params.beta - TRUE_BETA) < 0.05, f"beta={params.beta}"


def test_recovered_persistence_is_stationary(simulated_fit: GarchFit) -> None:
    """``alpha + beta < 1`` — the covariance-stationarity constraint, which
    is what makes the unconditional variance and the half-life meaningful."""
    assert simulated_fit.persistence is not None
    assert simulated_fit.persistence < 1.0
    assert simulated_fit.persistence == pytest.approx(
        TRUE_ALPHA + TRUE_BETA, abs=0.05
    )


def test_the_seeded_fit_is_active_with_no_reason(simulated_fit: GarchFit) -> None:
    """A well-specified series that passes every diagnostic is ACTIVE — the
    DEGRADED paths below are therefore real signals, not the default."""
    assert simulated_fit.health is ModelHealth.ACTIVE
    assert simulated_fit.reason is None
    assert simulated_fit.converged


def test_the_fit_is_deterministic(simulated: list[float]) -> None:
    """Same window ⇒ bit-identical parameters (spec §44 reproducibility)."""
    a = fit_garch(simulated)
    b = fit_garch(simulated)
    assert a.params == b.params
    assert a.loglik == b.loglik
    assert a.sigma2_series == b.sigma2_series


def test_recovered_unconditional_variance_is_near_the_sample_second_moment(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """A sanity check a reviewer can do by hand: the fitted ``V_L`` should be
    the same order as the realised mean ``r^2`` over the window."""
    sample_second_moment = math.fsum(r * r for r in simulated) / len(simulated)
    assert simulated_fit.unconditional_var is not None
    ratio = simulated_fit.unconditional_var / sample_second_moment
    assert 0.5 < ratio < 2.0, ratio


# ---------------------------------------------------------------------------
# Diagnostics (spec §13 — a fitted GARCH is never automatically trusted)
# ---------------------------------------------------------------------------


def test_diagnostics_carry_the_reviewers_numbers(simulated_fit: GarchFit) -> None:
    diagnostics = simulated_fit.diagnostics
    for key in (
        "n",
        "omega",
        "alpha",
        "beta",
        "persistence",
        "half_life",
        "unconditional_var",
        "unconditional_vol_annualized",
        "sigma2_next",
        "loglik",
        "converged",
        "iterations",
        "n_evals",
        "optimizer",
        "ljung_box_q_sq",
        "ljung_box_p",
        "ljung_box_lags",
        "std_resid_mean",
        "std_resid_var",
        "mean_model",
        "distribution",
    ):
        assert key in diagnostics, key
    assert diagnostics["n"] == SIM_N
    assert diagnostics["ljung_box_lags"] == DEFAULT_LJUNG_BOX_LAGS
    assert diagnostics["mean_model"] == "ZERO"
    assert diagnostics["distribution"] == DISTRIBUTION_GAUSSIAN_GARCH


def test_standardized_residuals_have_unit_variance(simulated_fit: GarchFit) -> None:
    """If the model captured the conditional variance, ``z_t = r_t/sigma_t``
    should have variance ≈ 1 — the basic goodness-of-filter check."""
    assert simulated_fit.diagnostics["std_resid_var"] == pytest.approx(1.0, abs=0.1)
    assert simulated_fit.diagnostics["std_resid_mean"] == pytest.approx(0.0, abs=0.1)
    assert len(simulated_fit.std_residuals) == SIM_N


def test_ljung_box_on_the_seeded_fit_finds_no_remaining_clustering(
    simulated_fit: GarchFit,
) -> None:
    """The whole point of the model: after filtering, ``z^2_t`` should show
    no autocorrelation, i.e. Ljung-Box p well above the 0.05 threshold."""
    assert simulated_fit.diagnostics["ljung_box_p"] > DEFAULT_LJUNG_BOX_P


def test_ljung_box_detects_clustering_when_it_is_really_there() -> None:
    """The diagnostic is not a rubber stamp: fed a strongly autocorrelated
    series it returns a tiny p-value.

    ``values`` alternates between two long blocks (a step function), whose
    lag-k autocorrelations are near 1 for small k.
    """
    values = [1.0] * 100 + [5.0] * 100 + [1.0] * 100 + [5.0] * 100
    q, p = ljung_box(values, lags=10)
    assert q > 100.0
    assert p < 1e-6


def test_ljung_box_hand_checked_on_a_short_series() -> None:
    """Q must match the formula ``n(n+2) Σ rho_k^2/(n-k)`` computed here
    independently, so the implementation is not testing itself."""
    values = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.8, -1.2, 0.3, 0.9, -0.4]
    lags = 3
    n = len(values)
    mean = sum(values) / n
    devs = [v - mean for v in values]
    denom = sum(d * d for d in devs)
    expected = 0.0
    for k in range(1, lags + 1):
        rho = sum(devs[t] * devs[t - k] for t in range(k, n)) / denom
        expected += rho * rho / (n - k)
    expected *= n * (n + 2)
    q, _p = ljung_box(values, lags=lags)
    assert q == pytest.approx(expected, rel=1e-12)


def test_ljung_box_requires_more_observations_than_lags() -> None:
    with pytest.raises(ValueError, match="needs n > lags"):
        ljung_box([1.0, 2.0, 3.0], lags=10)


def test_ljung_box_requires_variation() -> None:
    with pytest.raises(ValueError, match="non-zero variance"):
        ljung_box([2.0] * 50, lags=5)


# ---------------------------------------------------------------------------
# Constraint respect on adversarial series
# ---------------------------------------------------------------------------


def _assert_feasible(fit: GarchFit, label: str) -> None:
    """Whatever the health, a returned parameterisation is always feasible."""
    if fit.params is None:
        assert fit.health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED), label
        assert fit.reason, label
        return
    params = fit.params
    assert params.omega > 0.0, f"{label}: omega={params.omega}"
    assert params.alpha >= 0.0, f"{label}: alpha={params.alpha}"
    assert params.beta >= 0.0, f"{label}: beta={params.beta}"
    assert params.persistence < 1.0, f"{label}: persistence={params.persistence}"
    assert params.persistence <= PERSISTENCE_MAX, label


def test_constant_series_stays_feasible_and_never_raises() -> None:
    """A constant |return| has no volatility dynamics at all; the fit must
    still be feasible and must say something honest."""
    fit = fit_garch([0.01] * 300)
    _assert_feasible(fit, "constant")
    assert fit.health is not ModelHealth.FAILED
    assert fit.reason  # a constant series is never a clean ACTIVE fit


def test_all_zero_series_is_unavailable_not_a_crash() -> None:
    """Every return exactly 0 ⇒ the Gaussian likelihood is undefined
    (``ln sigma^2`` of nothing). Honest null with the real number, no
    exception (spec §13: never fabricate a GARCH result)."""
    fit = fit_garch([0.0] * 400)
    assert fit.health is ModelHealth.UNAVAILABLE
    assert fit.params is None
    assert "no variation" in fit.reason
    assert fit.sigma2_series == ()


def test_tiny_sample_is_unavailable_with_the_numbers() -> None:
    fit = fit_garch([0.01, -0.02, 0.015, -0.005] * 3)
    assert fit.health is ModelHealth.UNAVAILABLE
    assert fit.params is None
    assert fit.reason == f"n=12 < min_obs={DEFAULT_MIN_OBS}"


def test_unavailable_below_250_observations_by_default() -> None:
    """The documented ``min_obs=250`` boundary, checked on both sides."""
    series = simulate_garch(400, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 99)
    below = fit_garch(series[:249])
    at = fit_garch(series[:250])
    assert below.health is ModelHealth.UNAVAILABLE
    assert "n=249 < min_obs=250" in below.reason
    assert at.params is not None
    assert at.n == 250


def test_spike_series_stays_feasible() -> None:
    """A quiet series punctuated by huge jumps — the pathological case for a
    variance recursion. Feasible parameters, honest health, no exception."""
    series = [0.001] * 290 + [0.5, -0.6, 0.55, -0.7, 0.9] * 2
    fit = fit_garch(series)
    _assert_feasible(fit, "spikes")
    assert fit.health is not ModelHealth.FAILED


def test_alternating_extremes_stay_feasible() -> None:
    series = [0.2 if i % 2 else -0.2 for i in range(300)]
    fit = fit_garch(series)
    _assert_feasible(fit, "alternating")


def test_a_single_huge_outlier_stays_feasible() -> None:
    series = simulate_garch(400, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 7)
    series[200] = 100.0
    fit = fit_garch(series)
    _assert_feasible(fit, "outlier")


def test_non_finite_return_raises() -> None:
    """Malformed input raises; missing data degrades (house rule §9)."""
    with pytest.raises(ValueError, match="returns\\[3\\] must be a finite number"):
        fit_garch([0.01, 0.02, 0.03, float("nan")] + [0.01] * 300)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_obs": 1},
        {"max_iter": 0},
        {"ljung_box_lags": 0},
        {"ljung_box_p": 0.0},
        {"ljung_box_p": 1.0},
        {"degraded_persistence": 1.5},
    ],
)
def test_malformed_parameters_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        fit_garch([0.01] * 300, **kwargs)


# ---------------------------------------------------------------------------
# DEGRADED paths (spec §13 MODEL_DEGRADED)
# ---------------------------------------------------------------------------


def test_near_igarch_series_is_degraded_on_persistence() -> None:
    """A near-integrated process (true ``alpha+beta = 0.9999``, ``omega``
    vanishing) drives the estimate onto the stationarity ceiling. The fit is
    still feasible, but DEGRADED with the persistence quoted — never
    presented as a trustworthy stationary model."""
    series = simulate_garch(1500, 1e-9, 0.10, 0.8999, 424242, var0=1e-4)
    fit = fit_garch(series)
    assert fit.params is not None
    assert fit.persistence >= DEFAULT_DEGRADED_PERSISTENCE
    assert fit.health is ModelHealth.DEGRADED
    assert "persistence=" in fit.reason
    assert "near-integrated" in fit.reason
    _assert_feasible(fit, "near-IGARCH")


def test_degraded_persistence_threshold_is_a_parameter() -> None:
    """Every threshold is a documented parameter (house rule §9): raising it
    above the fitted persistence removes that trigger."""
    series = simulate_garch(1500, 1e-9, 0.10, 0.8999, 424242, var0=1e-4)
    strict = fit_garch(series)
    assert strict.health is ModelHealth.DEGRADED
    lax = fit_garch(series, degraded_persistence=0.9999999)
    assert lax.persistence == pytest.approx(strict.persistence)
    assert "persistence=" not in (lax.reason or "")


def test_ljung_box_threshold_can_force_a_degraded_verdict(
    simulated: list[float],
) -> None:
    """The clean fit becomes DEGRADED when the p-value bar is raised above
    its own p — proving the Ljung-Box trigger is wired to health, not just
    recorded in diagnostics."""
    clean = fit_garch(simulated)
    assert clean.health is ModelHealth.ACTIVE
    p_value = clean.diagnostics["ljung_box_p"]
    strict = fit_garch(simulated, ljung_box_p=min(p_value + 0.01, 0.999999))
    assert strict.health is ModelHealth.DEGRADED
    assert "Ljung-Box" in strict.reason
    assert "volatility clustering remains" in strict.reason


def test_non_convergence_is_degraded_not_silently_accepted(
    simulated: list[float],
) -> None:
    """Capping the optimiser at one iteration cannot converge; the fit is
    returned (feasible, usable) but DEGRADED with the optimiser's own
    reason — spec §57: calculation never claims health it did not earn."""
    fit = fit_garch(simulated, max_iter=1)
    assert fit.params is not None
    assert fit.health is ModelHealth.DEGRADED
    assert "did not converge" in fit.reason
    assert not fit.converged
    _assert_feasible(fit, "max_iter=1")


def test_degraded_fit_still_produces_a_forecast(simulated: list[float]) -> None:
    """DEGRADED means caveated, not absent (spec §58: do not halt all risk
    control because one advanced model is imperfect)."""
    fit = fit_garch(simulated, max_iter=1)
    result = garch_volatility_forecast(simulated, 1, fit=fit)
    assert result.health is ModelHealth.DEGRADED
    assert result.value is not None and result.value > 0.0
    assert result.reason == fit.reason


# ---------------------------------------------------------------------------
# Variance path & the walk-forward property
# ---------------------------------------------------------------------------


def test_variance_path_hand_checked() -> None:
    """``omega=1e-6, alpha=0.1, beta=0.8``, ``var0=4e-4``, ``r=[0.02]``:
    ``path = [4e-4, 1e-6 + 0.1(0.0004) + 0.8(0.0004) = 3.61e-4]``."""
    params = GarchParams(omega=1e-6, alpha=0.1, beta=0.8)
    path = garch_variance_path([0.02], params, var0=4e-4)
    assert path[0] == pytest.approx(4e-4)
    assert path[1] == pytest.approx(1e-6 + 0.1 * 0.0004 + 0.8 * 0.0004)
    assert path[1] == pytest.approx(3.61e-4)


def test_variance_path_length_and_walk_forward_safety(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """``path[t]`` uses ``returns[< t]`` only — a spike planted at ``t``
    cannot change the variance forecast FOR ``t`` (contract §3 invariant 5,
    spec §43 no hindsight)."""
    params = simulated_fit.params
    assert params is not None
    seed = simulated_fit.diagnostics["seed_variance"]
    base = garch_variance_path(simulated, params, var0=seed)
    assert len(base) == len(simulated) + 1

    spiked = list(simulated)
    spiked[500] = 10.0
    poisoned = garch_variance_path(spiked, params, var0=seed)
    assert poisoned[500] == base[500]        # the forecast FOR t is untouched
    assert poisoned[501] != base[501]        # the next one legitimately moves


def test_in_sample_sigma2_series_matches_the_path(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    params = simulated_fit.params
    assert params is not None
    path = garch_variance_path(
        simulated, params, var0=simulated_fit.diagnostics["seed_variance"]
    )
    assert list(simulated_fit.sigma2_series) == path[: len(simulated)]
    assert simulated_fit.sigma2_next == path[len(simulated)]


def test_variance_path_rejects_a_bad_var0() -> None:
    params = GarchParams(omega=1e-6, alpha=0.1, beta=0.8)
    with pytest.raises(ValueError, match="var0 must be"):
        garch_variance_path([0.01], params, var0=-1.0)


# ---------------------------------------------------------------------------
# Forecast: closed form vs iterative recursion (design §9.3)
# ---------------------------------------------------------------------------


def test_closed_form_forecast_equals_the_iterative_recursion(
    simulated_fit: GarchFit,
) -> None:
    """``V_L + (alpha+beta)^{k-1}(sigma^2_{t+1} - V_L)`` must reproduce
    iterating ``sigma^2_{k+1} = omega + (alpha+beta) sigma^2_k`` step by
    step — the identity that lets the library skip the loop."""
    params = simulated_fit.params
    assert params is not None
    horizon = 30
    closed = garch_forecast_variance(simulated_fit, horizon)

    iterative = []
    variance = simulated_fit.sigma2_next
    for _ in range(horizon):
        iterative.append(variance)
        variance = params.omega + params.persistence * variance

    assert len(closed) == horizon
    for k, (c, i) in enumerate(zip(closed, iterative)):
        assert c == pytest.approx(i, rel=1e-12, abs=1e-30), f"step {k}"


def test_forecast_first_step_is_sigma2_next(simulated_fit: GarchFit) -> None:
    assert garch_forecast_variance(simulated_fit, 1)[0] == pytest.approx(
        simulated_fit.sigma2_next
    )


def test_forecast_decays_monotonically_toward_the_unconditional_variance(
    simulated_fit: GarchFit,
) -> None:
    """The mean-reversion property: from a starting point below ``V_L`` the
    path rises toward it and never overshoots."""
    params = simulated_fit.params
    assert params is not None
    v_l = params.unconditional_variance
    path = garch_forecast_variance(simulated_fit, 400)
    start = path[0]
    if start < v_l:
        assert all(a <= b + 1e-18 for a, b in zip(path, path[1:]))
        assert all(v <= v_l + 1e-18 for v in path)
    else:
        assert all(a >= b - 1e-18 for a, b in zip(path, path[1:]))
        assert all(v >= v_l - 1e-18 for v in path)
    assert path[-1] == pytest.approx(v_l, rel=0.05)


def test_forecast_without_a_fit_raises() -> None:
    """An UNAVAILABLE fit has no forecast; the caller reports the honest null
    rather than receiving a fabricated number."""
    fit = fit_garch([0.01] * 10)
    with pytest.raises(ValueError, match="no GARCH forecast without a fit"):
        garch_forecast_variance(fit, 5)


def test_forecast_horizon_must_be_positive(simulated_fit: GarchFit) -> None:
    with pytest.raises(ValueError, match="h must be an int >= 1"):
        garch_forecast_variance(simulated_fit, 0)


# ---------------------------------------------------------------------------
# Half-life & persistence arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha, beta, expected_half_life",
    [
        (0.05, 0.90, math.log(0.5) / math.log(0.95)),   # ≈ 13.51 days
        (0.08, 0.90, math.log(0.5) / math.log(0.98)),   # ≈ 34.31 days
        (0.10, 0.80, math.log(0.5) / math.log(0.90)),   # ≈  6.58 days
        (0.25, 0.25, 1.0),                              # persistence 0.5 ⇒ exactly 1 day
    ],
)
def test_half_life_arithmetic(alpha: float, beta: float, expected_half_life: float) -> None:
    """``half_life = ln 0.5 / ln(alpha + beta)`` — days for a variance shock
    to decay by half."""
    params = GarchParams(omega=1e-6, alpha=alpha, beta=beta)
    assert params.persistence == pytest.approx(alpha + beta)
    assert params.half_life_days == pytest.approx(expected_half_life, rel=1e-12)


def test_half_life_of_one_day_is_exactly_persistence_one_half() -> None:
    """Hand-check: ``ln 0.5 / ln 0.5 = 1``."""
    assert GarchParams(omega=1e-6, alpha=0.2, beta=0.3).half_life_days == pytest.approx(1.0)


def test_zero_persistence_has_an_infinite_half_life() -> None:
    """A shock that does not survive at all has no meaningful halving time —
    reported as ``inf``, not as a fake 0."""
    assert GarchParams(omega=1e-4, alpha=0.0, beta=0.0).half_life_days == math.inf


def test_unconditional_variance_arithmetic() -> None:
    """``V_L = omega/(1 - alpha - beta)``: ``1e-6/0.02 = 5e-5``."""
    params = GarchParams(omega=1e-6, alpha=0.08, beta=0.90)
    assert params.unconditional_variance == pytest.approx(5e-5, rel=1e-12)


def test_the_fits_half_life_matches_its_own_parameters(simulated_fit: GarchFit) -> None:
    params = simulated_fit.params
    assert params is not None
    assert simulated_fit.half_life_days == pytest.approx(
        math.log(0.5) / math.log(params.persistence), rel=1e-12
    )
    assert simulated_fit.diagnostics["half_life"] == pytest.approx(
        simulated_fit.half_life_days
    )


@pytest.mark.parametrize(
    "omega, alpha, beta",
    [
        (0.0, 0.1, 0.8),      # omega must be > 0
        (-1e-6, 0.1, 0.8),
        (1e-6, -0.1, 0.8),    # alpha >= 0
        (1e-6, 0.1, -0.8),    # beta >= 0
        (1e-6, 0.5, 0.5),     # persistence must be < 1
        (1e-6, 0.9, 0.2),
    ],
)
def test_garch_params_rejects_an_infeasible_parameterisation(
    omega: float, alpha: float, beta: float
) -> None:
    with pytest.raises(ValueError):
        GarchParams(omega=omega, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# garch_volatility_forecast (ModelResult contract)
# ---------------------------------------------------------------------------


def test_volatility_forecast_is_a_research_labelled_model_result(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    result = garch_volatility_forecast(simulated, 1, fit=simulated_fit)
    assert isinstance(result, ModelResult)
    assert result.health is ModelHealth.ACTIVE
    assert result.value == pytest.approx(math.sqrt(simulated_fit.sigma2_next))
    assert result.sample_size == SIM_N
    assert result.meta.model_name == MODEL_NAME_GARCH
    assert result.meta.model_version == "1.0.0"
    assert result.meta.distribution == DISTRIBUTION_GAUSSIAN_GARCH
    assert result.meta.horizon_days == 1
    assert result.meta.params["mean_model"] == "ZERO"
    assert result.meta.params["innovations"] == "GAUSSIAN"
    assert result.meta.params["mode"] == str(ModelMode.RESEARCH)


def test_multi_day_forecast_sums_the_variance_term_structure(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """``sigma_h = sqrt(Σ_k sigma^2_{t+k})`` — NOT ``sqrt(h) x sigma_1``,
    because GARCH knows the term structure. The two differ whenever the
    current variance is away from ``V_L``, and ``params.scaling`` says so."""
    one = garch_volatility_forecast(simulated, 1, fit=simulated_fit)
    five = garch_volatility_forecast(simulated, 5, fit=simulated_fit)
    path = garch_forecast_variance(simulated_fit, 5)
    assert five.value == pytest.approx(math.sqrt(math.fsum(path)))
    assert five.meta.params["scaling"] == "GARCH_TERM_STRUCTURE"
    assert five.value > one.value
    assert five.diagnostics["forecast_variance_total"] == pytest.approx(
        math.fsum(path)
    )


def test_forecast_on_a_short_series_is_an_honest_null() -> None:
    result = garch_volatility_forecast([0.01, -0.02] * 20, 1)
    assert result.health is ModelHealth.UNAVAILABLE
    assert result.value is None
    assert f"min_obs={DEFAULT_MIN_OBS}" in result.reason


def test_forecast_on_a_flat_series_is_an_honest_null() -> None:
    result = garch_volatility_forecast([0.0] * 400, 1)
    assert result.health is ModelHealth.UNAVAILABLE
    assert result.value is None
    assert "no variation" in result.reason


# ---------------------------------------------------------------------------
# Filtered historical simulation (garch_scaled_pnl)
# ---------------------------------------------------------------------------


def test_garch_scaled_pnl_keeps_every_observation(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """Unlike the EWMA filter there is no warm-up to drop: the recursion is
    seeded from the window's own second moment, so the output length equals
    the input length."""
    scaled = garch_scaled_pnl(simulated, fit=simulated_fit)
    assert len(scaled) == len(simulated)


def test_garch_scaled_pnl_applies_sigma_now_over_sigma_t(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """Hand-check the scaling on the first three entries."""
    scaled = garch_scaled_pnl(simulated, fit=simulated_fit)
    sigma_now = math.sqrt(simulated_fit.sigma2_next)
    for t in range(3):
        expected = simulated[t] * sigma_now / math.sqrt(simulated_fit.sigma2_series[t])
        assert scaled[t] == pytest.approx(expected, rel=1e-12)


def test_garch_scaling_is_exactly_linear_for_a_given_fit(
    simulated: list[float], simulated_fit: GarchFit
) -> None:
    """Doubling the P&L doubles the filtered series EXACTLY: the ratio
    ``sigma_now/sigma_t`` is scale-free, so VaR/ES over it scale by the same
    factor (contract §3 invariant 4)."""
    base = garch_scaled_pnl(simulated, fit=simulated_fit)
    doubled = garch_scaled_pnl([2.0 * v for v in simulated], fit=simulated_fit)
    assert len(base) == len(doubled)
    for a, b in zip(base, doubled):
        assert b == 2.0 * a


def test_the_estimator_is_equivariant_under_rescaling(simulated: list[float]) -> None:
    """Refitting on ``2 x pnl`` must give the SAME dynamics: ``alpha`` and
    ``beta`` are scale-free while ``omega`` (a variance) scales by ``2^2``.

    The agreement is ~1e-6, not exact, because these are two independent
    Nelder–Mead optimisations — which is precisely why the test above pins
    the *filter* with a shared fit and this one pins the *estimator*
    separately at the tolerance a derivative-free optimiser can deliver.
    """
    base = fit_garch(simulated)
    rescaled = fit_garch([2.0 * v for v in simulated])
    assert base.params is not None and rescaled.params is not None
    assert rescaled.params.alpha == pytest.approx(base.params.alpha, rel=1e-4)
    assert rescaled.params.beta == pytest.approx(base.params.beta, rel=1e-4)
    assert rescaled.params.omega == pytest.approx(4.0 * base.params.omega, rel=1e-4)


def test_garch_scaling_has_the_same_shape_as_the_ewma_filter(
    simulated: list[float],
) -> None:
    """Design §9.3: "FHS shape identical to ``volatility_scaled_pnl``" — the
    same call signature style, the same list-of-floats output, and both feed
    the historical estimators unchanged."""
    garch_series = garch_scaled_pnl(simulated)
    ewma_series = volatility_scaled_pnl(simulated)
    assert isinstance(garch_series, list) and isinstance(ewma_series, list)
    assert all(isinstance(v, float) for v in garch_series)
    for series in (garch_series, ewma_series):
        var = historical_var(series, 0.95)
        es = historical_es(series, 0.95)
        assert var.value is not None and es.value is not None
        assert es.value >= var.value          # contract §3 invariant 1


def test_garch_scaling_bookkeeping(simulated: list[float], simulated_fit: GarchFit) -> None:
    scaling = garch_scaling(simulated, fit=simulated_fit)
    assert scaling.n_input == SIM_N
    assert scaling.n_used == SIM_N
    assert scaling.dropped == 0
    assert scaling.sigma_now == pytest.approx(math.sqrt(simulated_fit.sigma2_next))
    assert scaling.fit is simulated_fit


def test_garch_scaled_pnl_is_empty_without_a_fit() -> None:
    """No parameters ⇒ no filtered series (honest null), and the reason
    travels on the fit."""
    scaling = garch_scaling([0.01, -0.01] * 30)
    assert scaling.scaled == ()
    assert scaling.sigma_now is None
    assert scaling.fit.health is ModelHealth.UNAVAILABLE
    assert garch_scaled_pnl([0.01, -0.01] * 30) == []


# ---------------------------------------------------------------------------
# Fallback hierarchy (spec §13/§58; design §9.3)
# ---------------------------------------------------------------------------


def test_fallback_picks_garch_when_the_fit_is_active(simulated: list[float]) -> None:
    source, result, reason = conditional_volatility_source(simulated)
    assert source == SOURCE_GARCH
    assert result.meta.model_name == MODEL_NAME_GARCH
    assert result.health is ModelHealth.ACTIVE
    assert "GARCH(1,1) ACTIVE" in reason
    assert "persistence=" in reason
    assert "Ljung-Box" in reason


def test_fallback_picks_ewma_when_the_series_is_too_short_for_garch() -> None:
    """The commonest real case: a book with 120 days of history. GARCH is
    UNAVAILABLE, EWMA answers, and the reason names both facts."""
    series = simulate_garch(120, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 5150)
    source, result, reason = conditional_volatility_source(series)
    assert source == SOURCE_EWMA
    assert result.meta.model_name == "ewma_volatility"
    assert result.health is ModelHealth.ACTIVE
    assert result.value is not None and result.value > 0.0
    assert "GARCH not ACTIVE (health=UNAVAILABLE)" in reason
    assert "n=120 < min_obs=250" in reason
    assert "falling back to EWMA(lambda=0.94)" in reason


def test_fallback_picks_ewma_when_the_garch_fit_is_degraded(
    simulated: list[float],
) -> None:
    """Spec §13: a model that fails diagnostics falls back to a simpler one.
    A DEGRADED GARCH is NOT used by the selector, and the reason quotes the
    diagnostic that rejected it."""
    p_value = fit_garch(simulated).diagnostics["ljung_box_p"]
    source, result, reason = conditional_volatility_source(
        simulated, ljung_box_p=min(p_value + 0.01, 0.999999)
    )
    assert source == SOURCE_EWMA
    assert result.meta.model_name == "ewma_volatility"
    assert "GARCH not ACTIVE (health=DEGRADED)" in reason
    assert "Ljung-Box" in reason
    assert "falling back to EWMA" in reason


def test_fallback_picks_ewma_when_the_series_has_no_variation() -> None:
    source, _result, reason = conditional_volatility_source([0.0] * 400)
    assert source == SOURCE_EWMA
    assert "no variation" in reason


def test_fallback_reports_honestly_when_even_ewma_cannot_answer() -> None:
    """Nothing halts and nothing is fabricated: the EWMA branch returns its
    own UNAVAILABLE with its own reason (spec §58)."""
    source, result, reason = conditional_volatility_source([0.01, -0.02, 0.03])
    assert source == SOURCE_EWMA
    assert result.health is ModelHealth.UNAVAILABLE
    assert result.value is None
    assert "init_obs" in result.reason
    assert "GARCH not ACTIVE" in reason


def test_fallback_respects_the_ewma_parameters() -> None:
    series = simulate_garch(120, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 5150)
    _source, result, reason = conditional_volatility_source(series, lam=0.97)
    assert result.meta.params["lambda"] == 0.97
    assert "lambda=0.97" in reason


def test_scaled_pnl_fallback_mirrors_the_forecast_fallback(
    simulated: list[float],
) -> None:
    """The conditional VaR/ES input follows the SAME rule, so a snapshot
    cannot end up with a GARCH volatility tile beside an EWMA-filtered VaR."""
    source, scaled, reason = conditional_scaled_pnl_source(simulated)
    assert source == SOURCE_GARCH
    assert len(scaled) == SIM_N
    assert "ACTIVE" in reason

    short = simulate_garch(120, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 5150)
    source_short, scaled_short, reason_short = conditional_scaled_pnl_source(short)
    assert source_short == SOURCE_EWMA
    assert scaled_short == volatility_scaled_pnl(short)
    assert "falling back to EWMA" in reason_short


# ---------------------------------------------------------------------------
# The registered model (spec §4, §70)
# ---------------------------------------------------------------------------


def test_the_model_is_registered_in_research_mode() -> None:
    """RESEARCH is one step BELOW SHADOW: this model is not even the shadow
    conditional-volatility forecaster (spec §70)."""
    model = get(MODEL_NAME_GARCH)
    assert isinstance(model, Garch11Model)
    assert model.mode is ModelMode.RESEARCH
    assert model.mode is not ModelMode.PRODUCTION
    assert model.name == MODEL_NAME_GARCH
    assert model.version == "1.0.0"


def test_the_model_calculates_the_same_number_as_the_function(
    simulated: list[float],
) -> None:
    model = Garch11Model()
    from_model = model.calculate(simulated)
    from_function = garch_volatility_forecast(simulated, 1)
    assert from_model.value == pytest.approx(from_function.value)
    assert from_model.health is from_function.health


def test_the_model_exposes_its_parameters(simulated: list[float]) -> None:
    model = Garch11Model(horizon_days=5, min_obs=300)
    params = model.params()
    assert params["horizon_days"] == 5
    assert params["min_obs"] == 300
    assert params["ljung_box_lags"] == DEFAULT_LJUNG_BOX_LAGS
    result = model.calculate(simulated)
    assert result.meta.horizon_days == 5
    assert result.meta.params["min_obs"] == 300


def test_model_validate_never_upgrades_health() -> None:
    """Contract §3 invariant 8, inherited from ``BaseRiskModel``."""
    model = Garch11Model()
    result = model.calculate([0.01, -0.02] * 20)
    assert result.health is ModelHealth.UNAVAILABLE
    assert model.validate(result).health is ModelHealth.UNAVAILABLE


# ---------------------------------------------------------------------------
# Result-shape guards
# ---------------------------------------------------------------------------


def test_unavailable_fit_carries_no_parameters() -> None:
    with pytest.raises(ValueError, match="requires params=None"):
        GarchFit(
            params=GarchParams(omega=1e-6, alpha=0.1, beta=0.8),
            loglik=None,
            converged=False,
            iterations=0,
            persistence=0.9,
            unconditional_var=1e-5,
            half_life_days=6.0,
            sigma2_series=(),
            std_residuals=(),
            n=10,
            health=ModelHealth.UNAVAILABLE,
            reason="too short",
        )


def test_non_active_fit_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        GarchFit(
            params=None,
            loglik=None,
            converged=False,
            iterations=0,
            persistence=None,
            unconditional_var=None,
            half_life_days=None,
            sigma2_series=(),
            std_residuals=(),
            n=10,
            health=ModelHealth.UNAVAILABLE,
            reason=None,
        )


def test_is_available_tracks_whether_parameters_exist(
    simulated_fit: GarchFit,
) -> None:
    assert simulated_fit.is_available
    assert not fit_garch([0.01] * 10).is_available
