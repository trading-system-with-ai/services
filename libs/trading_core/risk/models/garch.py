"""GARCH(1,1) conditional volatility with Gaussian innovations — RESEARCH
(Phase E design §9.3; risk spec §12, §13, §14, §57, §58, §63).

Pure stdlib, deterministic, no numpy/scipy (house rule): the MLE is
maximised by the stdlib Nelder–Mead in ``risk/optim.py`` and the Ljung–Box
p-value comes from ``models/_chi2.py``. **RESEARCH mode** (spec §70): every
model class here is constructed with ``ModelMode.RESEARCH``, which is
strictly below SHADOW — these numbers are studied and displayed, they are
not even the shadow forecaster yet. The production conditional-volatility
forecaster remains EWMA (``models/volatility.py``), and
:func:`conditional_volatility_source` is the seam that says which one a
caller actually got.

The model
---------
For a zero-mean return series ``r_t``:

    r_t = sigma_t * z_t,   z_t ~ N(0, 1) i.i.d.
    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

with ``omega > 0``, ``alpha >= 0``, ``beta >= 0`` and persistence
``alpha + beta < 1`` (covariance stationarity). The unconditional variance
is ``V_L = omega / (1 - alpha - beta)`` and the shock half-life is
``ln 0.5 / ln(alpha + beta)`` days.

**Zero mean is an assumption, not an oversight.** Daily equity returns have
a mean two orders of magnitude below their standard deviation, and the risk
question is about the size of moves; estimating a mean would add a
parameter that the likelihood barely identifies. It is documented on every
result (``params.mean_model = "ZERO"``) so no reader assumes otherwise.

Estimation
----------
Conditional (quasi-)maximum likelihood. The negative log-likelihood, up to
a constant, is

    -2 * logL = sum_t [ ln sigma^2_t + r^2_t / sigma^2_t ]

recursed forward from ``sigma^2_0 = mean(r^2)`` (the sample second moment —
a *backcast* that uses only the estimation window, never future data). The
constant ``n ln(2 pi)`` is added back in the reported ``loglik`` so the
number is a real log-likelihood comparable across fits of equal length.

Constraints are enforced by an **unconstrained reparameterisation**, so the
optimiser can never propose an infeasible point and the fit never needs a
penalty term:

    omega       = softplus(u0)                      (> 0)
    persistence = PERSISTENCE_MAX * logistic(u1)    (in (0, PERSISTENCE_MAX))
    alpha       = persistence * logistic(u2)        (>= 0)
    beta        = persistence * (1 - logistic(u2))  (>= 0)

``PERSISTENCE_MAX = 0.999999`` keeps ``V_L`` finite by construction. A fit
that lands ON the ceiling is a near-IGARCH corner and is reported DEGRADED
(``persistence >= degraded_persistence``, default 0.999), never silently
accepted as stationary.

Start values (design §9.3): ``alpha = 0.06``, ``beta = 0.90``,
``omega = (1 - alpha - beta) * sample variance`` — the EWMA-shaped prior
that fits most daily equity series. The optimiser is then restarted once
from its own solution with a smaller initial simplex, which is the standard
cheap guard against a premature Nelder–Mead stop; both runs are
deterministic, so the fit is reproducible bit-for-bit.

Diagnostics (spec §13 — a fitted GARCH is never automatically trusted)
----------------------------------------------------------------------
``GarchFit.diagnostics`` carries the numbers a model-risk reviewer needs:
sample size, persistence, half-life, unconditional variance and annualized
unconditional volatility, the optimiser's iteration/evaluation counts and
convergence reason, the mean and variance of the standardized residuals
(both should be ~0 and ~1), and the **Ljung–Box Q statistic on the SQUARED
standardized residuals** at ``m = 10`` lags with its chi2(m) p-value. That
last one is the actual test of the model's purpose: if volatility
clustering remains in ``z^2_t`` after filtering, the GARCH did not capture
it, and the fit is DEGRADED.

Health (spec §41, §57 — calculation never claims health beyond
ACTIVE-if-computed; these downgrades are all data/fit facts, not opinions):

- ``UNAVAILABLE``: ``n < min_obs`` (default 250), or the series carries no
  variation at all (every return 0) so the likelihood is undefined.
- ``DEGRADED``: the optimiser did not converge, or
  ``persistence >= 0.999``, or Ljung-Box ``p < 0.05``, or the fitted
  ``omega`` underflowed to ~0 (a degenerate corner where the unconditional
  variance is not identified). The reason names every trigger with numbers.
- ``FAILED``: a numeric error inside the fit. Never raised at the caller —
  data problems degrade, they do not explode (house rule).

Fallback (spec §13/§58, design §9.3)
------------------------------------
:func:`conditional_volatility_source` implements the hierarchy in one
place: try GARCH; if its health is not ACTIVE, return the EWMA forecast
instead, with a ``reason`` that quotes why GARCH was rejected. The caller
records ``"GARCH"`` or ``"EWMA"`` in the snapshot, so the displayed number
always names its own forecaster. Nothing halts because GARCH failed.

Sign/unit conventions are the library's (contract §1): ``garch_scaled_pnl``
takes a gain-positive P&L series in USD and returns the same series
rescaled to today's conditional volatility, in USD — identical in shape to
``volatility.volatility_scaled_pnl`` so the historical VaR/ES estimators
consume either one unchanged (``distribution="EMPIRICAL_GARCH_SCALED"``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..optim import DEFAULT_MAX_ITER as _OPTIM_MAX_ITER
from ..optim import NMResult, nelder_mead
from ._chi2 import chi2_sf
from .base import (
    BaseRiskModel,
    ModelHealth,
    ModelMeta,
    ModelMode,
    ModelResult,
    ModelTier,
    active,
    degraded,
    failed,
    register,
    unavailable,
)
from .volatility import (
    DEFAULT_INIT_OBS,
    DEFAULT_LAMBDA,
    ewma_volatility_forecast,
    volatility_scaled_pnl,
)

#: Estimator version (contract §4): arithmetic change ⇒ MAJOR bump.
MODEL_VERSION = "1.0.0"

#: Registry / ``ModelMeta.model_name`` label.
MODEL_NAME_GARCH = "garch11"

#: Distribution labels (contract §1 provenance).
DISTRIBUTION_GAUSSIAN_GARCH = "GAUSSIAN_GARCH"
DISTRIBUTION_EMPIRICAL_GARCH_SCALED = "EMPIRICAL_GARCH_SCALED"

#: Forecaster names returned by :func:`conditional_volatility_source`.
SOURCE_GARCH = "GARCH"
SOURCE_EWMA = "EWMA"

#: Documented defaults (design §9.3). Every one is a keyword parameter.
DEFAULT_MIN_OBS = 250               # GARCH needs a real sample; below this ⇒ UNAVAILABLE
DEFAULT_MAX_ITER = _OPTIM_MAX_ITER  # Nelder–Mead iteration cap per restart
DEFAULT_LJUNG_BOX_LAGS = 10         # m in the Ljung–Box Q on standardized residuals²
DEFAULT_LJUNG_BOX_P = 0.05          # p below this ⇒ DEGRADED (clustering remains)
DEFAULT_DEGRADED_PERSISTENCE = 0.999  # α+β at/above this ⇒ DEGRADED (near-IGARCH)
DEFAULT_ANNUALIZATION_DAYS = 252
DEFAULT_FREQUENCY = "1D"

#: Starting parameters for the MLE (EWMA-shaped prior, design §9.3).
INIT_ALPHA = 0.06
INIT_BETA = 0.90

#: Ceiling on α+β imposed by the transform; keeps V_L finite by construction.
PERSISTENCE_MAX = 0.999999

#: Below this the fitted ω is numerically indistinguishable from 0 and the
#: unconditional variance is not identified ⇒ DEGRADED.
OMEGA_FLOOR_RATIO = 1e-12   # relative to the sample variance


# ---------------------------------------------------------------------------
# Guards (same shape as volatility.py)
# ---------------------------------------------------------------------------


def _check_finite(values: Sequence[float], *, what: str) -> list[float]:
    out: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}[{i}] must be a finite number, got {v!r}")
        out.append(float(v))
    return out


def _check_positive_int(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _check_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be a float in (0, 1), got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchParams:
    """Fitted GARCH(1,1) parameters (design §9.3).

    Constraints are checked at construction — a ``GarchParams`` instance is
    always a valid, covariance-stationary parameterisation:
    ``omega > 0``, ``alpha >= 0``, ``beta >= 0``, ``alpha + beta < 1``.
    The estimator can only produce such points (its transform enforces
    them), so a violation here is a programming error, not a data problem.
    """

    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name in ("omega", "alpha", "beta"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
            object.__setattr__(self, name, float(v))
        if self.omega <= 0.0:
            raise ValueError(f"omega must be > 0, got {self.omega}")
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(
                f"persistence alpha+beta must be < 1 (covariance stationarity), "
                f"got {self.alpha} + {self.beta} = {self.alpha + self.beta}"
            )

    @property
    def persistence(self) -> float:
        """``alpha + beta`` — how long a shock survives (always < 1)."""
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """``V_L = omega / (1 - alpha - beta)`` — finite by the constraint."""
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life_days(self) -> float:
        """``ln 0.5 / ln(persistence)`` — days for a variance shock to halve.

        ``inf`` when ``persistence == 0`` (a shock does not survive at all,
        so "halving time" is degenerate); otherwise finite and positive.
        """
        p = self.persistence
        if p <= 0.0:
            return math.inf
        return math.log(0.5) / math.log(p)

    def as_dict(self) -> dict[str, float]:
        return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}


# ---------------------------------------------------------------------------
# Unconstrained transform
# ---------------------------------------------------------------------------


def _softplus(u: float) -> float:
    if u > 30.0:
        return u
    if u < -30.0:
        return math.exp(u)
    return math.log1p(math.exp(u))


def _inv_softplus(y: float) -> float:
    if y > 30.0:
        return y
    return math.log(math.expm1(y))


def _logistic(u: float) -> float:
    if u >= 0.0:
        return 1.0 / (1.0 + math.exp(-u))
    e = math.exp(u)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _unpack(u: Sequence[float]) -> tuple[float, float, float]:
    """Unconstrained ``u`` → ``(omega, alpha, beta)`` inside the feasible set."""
    omega = _softplus(u[0])
    persistence = PERSISTENCE_MAX * _logistic(u[1])
    weight = _logistic(u[2])
    return omega, persistence * weight, persistence * (1.0 - weight)


def _pack(omega: float, alpha: float, beta: float) -> list[float]:
    """``(omega, alpha, beta)`` → the unconstrained coordinates (inverse of
    :func:`_unpack`); used only for the deterministic start point."""
    persistence = alpha + beta
    return [
        _inv_softplus(omega),
        _logit(persistence / PERSISTENCE_MAX),
        _logit(alpha / persistence),
    ]


# ---------------------------------------------------------------------------
# Likelihood & recursion
# ---------------------------------------------------------------------------


def garch_variance_path(
    returns: Sequence[float],
    params: GarchParams,
    *,
    var0: float | None = None,
) -> list[float]:
    """Conditional variances ``sigma^2_t`` for ``t = 0 … n`` (length ``n + 1``).

    ``path[t]`` for ``t < n`` is the variance of observation ``t`` given
    information STRICTLY BEFORE it (``returns[< t]`` only — walk-forward
    safe by construction, contract §3 invariant 5), and ``path[n]`` is the
    one-step-ahead forecast for the NEXT period. The recursion is seeded
    with ``var0`` (default: the sample second moment ``mean(r^2)`` of the
    given window — a backcast from the estimation window itself, never
    future data).

    Hand-check (``omega=1e-6, alpha=0.1, beta=0.8``, ``var0=4e-4``,
    ``r=[0.02]``): ``path = [4e-4, 1e-6 + 0.1*4e-4 + 0.8*4e-4 = 3.61e-4]``.
    """
    values = _check_finite(returns, what="returns")
    if not isinstance(params, GarchParams):
        raise ValueError("params must be a GarchParams")
    if var0 is None:
        n = len(values)
        seed = math.fsum(r * r for r in values) / n if n else 0.0
    else:
        if isinstance(var0, bool) or not isinstance(var0, (int, float)) or not math.isfinite(var0) or var0 < 0.0:
            raise ValueError(f"var0 must be a finite number >= 0, got {var0!r}")
        seed = float(var0)
    path = [seed]
    var = seed
    for r in values:
        var = params.omega + params.alpha * r * r + params.beta * var
        path.append(var)
    return path


def _negative_loglik(returns: Sequence[float], omega: float, alpha: float, beta: float, var0: float) -> float:
    """``0.5 * sum[ln sigma^2_t + r^2_t/sigma^2_t]`` — the objective, up to
    the constant ``0.5 n ln(2 pi)``. ``inf`` on any non-positive/overflowing
    variance (infeasible point; the simplex is pushed back)."""
    var = var0
    total = 0.0
    for r in returns:
        if var <= 0.0 or not math.isfinite(var):
            return math.inf
        total += math.log(var) + (r * r) / var
        var = omega + alpha * r * r + beta * var
    if not math.isfinite(total):
        return math.inf
    return 0.5 * total


def ljung_box(values: Sequence[float], *, lags: int = DEFAULT_LJUNG_BOX_LAGS) -> tuple[float, float]:
    """Ljung–Box ``(Q, p)`` on ``values`` at ``m = lags`` (χ²(m) p-value).

    ``Q = n(n+2) * sum_{k=1..m} rho_k^2 / (n - k)`` where ``rho_k`` is the
    sample autocorrelation of ``values`` at lag ``k`` (about its own mean).
    Applied by :func:`fit_garch` to the SQUARED standardized residuals: a
    small p-value says volatility clustering survived the filter.

    Requires ``n > lags`` and non-zero variance; otherwise ``ValueError``
    (the caller — ``fit_garch`` — checks first and degrades instead).
    """
    lags = _check_positive_int(lags, name="lags")
    series = _check_finite(values, what="values")
    n = len(series)
    if n <= lags:
        raise ValueError(f"ljung_box needs n > lags, got n={n}, lags={lags}")
    mean = math.fsum(series) / n
    devs = [v - mean for v in series]
    denom = math.fsum(d * d for d in devs)
    if denom <= 0.0:
        raise ValueError("ljung_box needs non-zero variance in values")
    q = 0.0
    for k in range(1, lags + 1):
        num = math.fsum(devs[t] * devs[t - k] for t in range(k, n))
        rho = num / denom
        q += (rho * rho) / (n - k)
    q *= n * (n + 2)
    return q, chi2_sf(q, lags)


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchFit:
    """Outcome of :func:`fit_garch` (design §9.3; spec §13 persisted diagnostics).

    - ``params``: the fitted :class:`GarchParams`, or ``None`` when the fit
      produced no number (UNAVAILABLE / FAILED — honest null, never a
      fabricated parameterisation);
    - ``loglik``: the Gaussian log-likelihood at the optimum (constant
      included), ``None`` when there is no fit;
    - ``converged`` / ``iterations``: the optimiser's own verdict and cost;
    - ``persistence`` / ``unconditional_var`` / ``half_life_days``: the
      headline stability numbers (``None`` without a fit);
    - ``sigma2_series``: in-sample conditional variances, one per input
      return, each using information strictly before its observation;
    - ``std_residuals``: ``r_t / sigma_t`` for the same indices;
    - ``diagnostics``: the reviewer's numbers (see the module docstring);
    - ``health`` / ``reason``: spec §41 states; every non-ACTIVE state
      carries a reason with real numbers.
    """

    params: GarchParams | None
    loglik: float | None
    converged: bool
    iterations: int
    persistence: float | None
    unconditional_var: float | None
    half_life_days: float | None
    sigma2_series: tuple[float, ...]
    std_residuals: tuple[float, ...]
    n: int
    health: ModelHealth
    reason: str | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED) and self.params is not None:
            raise ValueError(f"health={health} requires params=None (honest null)")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def is_available(self) -> bool:
        """True iff a parameterisation was produced (ACTIVE or DEGRADED)."""
        return self.params is not None

    @property
    def sigma2_next(self) -> float | None:
        """One-step-ahead conditional variance ``sigma^2_{t+1}``; ``None``
        without a fit."""
        return self.diagnostics.get("sigma2_next")


def _unavailable_fit(n: int, reason: str, diagnostics: Mapping[str, Any]) -> GarchFit:
    return GarchFit(
        params=None,
        loglik=None,
        converged=False,
        iterations=0,
        persistence=None,
        unconditional_var=None,
        half_life_days=None,
        sigma2_series=(),
        std_residuals=(),
        n=n,
        health=ModelHealth.UNAVAILABLE,
        reason=reason,
        diagnostics=dict(diagnostics),
    )


def fit_garch(
    returns: Sequence[float],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
) -> GarchFit:
    """Fit GARCH(1,1) by conditional Gaussian MLE (design §9.3).

    Deterministic and reproducible: the start point, the two Nelder–Mead
    runs and every tie-break are fixed, so the same window always yields
    bit-identical parameters.

    Constraints (``omega > 0``, ``alpha, beta >= 0``, ``alpha + beta < 1``)
    hold for EVERY returned fit — they are built into the parameterisation,
    not checked afterwards. Even on adversarial input (a constant series,
    a series of spikes, a near-integrated series) the result is a feasible
    point, possibly DEGRADED.

    Health (see the module docstring): ``n < min_obs`` or a series with no
    variation ⇒ UNAVAILABLE; non-convergence, ``persistence >=
    degraded_persistence``, Ljung–Box ``p < ljung_box_p``, or a fitted
    ``omega`` at the numerical floor ⇒ DEGRADED with every trigger named;
    an internal numeric error ⇒ FAILED. **Never raises for data problems**
    (spec §13 "never fabricate GARCH results"); malformed arguments
    (non-finite return, ``min_obs < 2``) still raise ``ValueError``.
    """
    min_obs = _check_positive_int(min_obs, name="min_obs", minimum=2)
    max_iter = _check_positive_int(max_iter, name="max_iter")
    ljung_box_lags = _check_positive_int(ljung_box_lags, name="ljung_box_lags")
    ljung_box_p = _check_probability(ljung_box_p, name="ljung_box_p")
    degraded_persistence = _check_probability(
        degraded_persistence, name="degraded_persistence"
    )
    annualization_days = _check_positive_int(annualization_days, name="annualization_days")
    values = _check_finite(returns, what="returns")
    n = len(values)
    base_diagnostics: dict[str, Any] = {
        "n": n,
        "min_obs": min_obs,
        "mean_model": "ZERO",
        "distribution": DISTRIBUTION_GAUSSIAN_GARCH,
        "ljung_box_lags": ljung_box_lags,
    }
    if n < min_obs:
        return _unavailable_fit(n, f"n={n} < min_obs={min_obs}", base_diagnostics)

    second_moment = math.fsum(r * r for r in values) / n
    if second_moment <= 0.0:
        return _unavailable_fit(
            n,
            f"no variation in the series (mean r^2 = {second_moment}) — "
            "the Gaussian likelihood is undefined",
            base_diagnostics,
        )

    try:
        mean = math.fsum(values) / n
        sample_var = math.fsum((r - mean) * (r - mean) for r in values) / (n - 1)
        seed_var = second_moment  # backcast from this window only (no hindsight)
        omega0 = max((1.0 - INIT_ALPHA - INIT_BETA) * sample_var, second_moment * 1e-6)
        x0 = _pack(omega0, INIT_ALPHA, INIT_BETA)

        def objective(u: Sequence[float]) -> float:
            omega, alpha, beta = _unpack(u)
            return _negative_loglik(values, omega, alpha, beta, seed_var)

        first: NMResult = nelder_mead(objective, x0, step=0.5, max_iter=max_iter)
        # One deterministic restart from the solution with a tighter simplex:
        # the standard cheap guard against a premature Nelder–Mead stop.
        second: NMResult = nelder_mead(objective, list(first.x), step=0.1, max_iter=max_iter)
        best = second if second.fval <= first.fval else first

        omega, alpha, beta = _unpack(best.x)
        params = GarchParams(omega=omega, alpha=alpha, beta=beta)
        path = garch_variance_path(values, params, var0=seed_var)
        sigma2_series = tuple(path[:n])
        sigma2_next = path[n]
        std_residuals = tuple(
            r / math.sqrt(s2) for r, s2 in zip(values, sigma2_series)
        )
        loglik = -best.fval - 0.5 * n * math.log(2.0 * math.pi)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:  # pragma: no cover - defensive
        return GarchFit(
            params=None,
            loglik=None,
            converged=False,
            iterations=0,
            persistence=None,
            unconditional_var=None,
            half_life_days=None,
            sigma2_series=(),
            std_residuals=(),
            n=n,
            health=ModelHealth.FAILED,
            reason=f"GARCH fit raised {type(exc).__name__}: {exc}",
            diagnostics=base_diagnostics,
        )

    # --- diagnostics -------------------------------------------------------
    z2 = [z * z for z in std_residuals]
    try:
        lb_q, lb_p = ljung_box(z2, lags=ljung_box_lags)
    except ValueError as exc:
        lb_q, lb_p = None, None
        lb_note: str | None = f"Ljung-Box unavailable: {exc}"
    else:
        lb_note = None
    z_mean = math.fsum(std_residuals) / n
    z_var = math.fsum((z - z_mean) * (z - z_mean) for z in std_residuals) / (n - 1)
    converged = best.converged
    diagnostics: dict[str, Any] = dict(base_diagnostics)
    diagnostics.update(
        {
            "omega": params.omega,
            "alpha": params.alpha,
            "beta": params.beta,
            "persistence": params.persistence,
            "half_life": params.half_life_days,
            "unconditional_var": params.unconditional_variance,
            "unconditional_vol": math.sqrt(params.unconditional_variance),
            "unconditional_vol_annualized": math.sqrt(
                params.unconditional_variance * annualization_days
            ),
            "annualization_days": annualization_days,
            "sigma2_next": sigma2_next,
            "sigma_next": math.sqrt(sigma2_next),
            "sample_variance": sample_var,
            "seed_variance": seed_var,
            "loglik": loglik,
            "converged": converged,
            "iterations": best.iterations,
            "n_evals": best.n_evals,
            "optimizer": "nelder_mead",
            "optimizer_reason": best.reason,
            "ljung_box_q_sq": lb_q,
            "ljung_box_p": lb_p,
            "std_resid_mean": z_mean,
            "std_resid_var": z_var,
        }
    )
    if lb_note:
        diagnostics["ljung_box_note"] = lb_note

    # --- health (every trigger named with its number) ----------------------
    triggers: list[str] = []
    if not converged:
        triggers.append(f"optimizer did not converge ({best.reason})")
    if params.persistence >= degraded_persistence:
        triggers.append(
            f"persistence={params.persistence:.6f} >= {degraded_persistence} "
            "(near-integrated: the unconditional variance is barely identified)"
        )
    if lb_p is not None and lb_p < ljung_box_p:
        triggers.append(
            f"Ljung-Box(m={ljung_box_lags}) on standardized residuals^2: "
            f"Q={lb_q:.4f}, p={lb_p:.4g} < {ljung_box_p} "
            "(volatility clustering remains after filtering)"
        )
    if lb_p is None:
        triggers.append(lb_note or "Ljung-Box unavailable")
    if params.omega <= sample_var * OMEGA_FLOOR_RATIO:
        triggers.append(
            f"omega={params.omega:.3e} at the numerical floor "
            f"({OMEGA_FLOOR_RATIO:g} x sample variance {sample_var:.3e}) — "
            "the unconditional variance is not identified"
        )
    health = ModelHealth.DEGRADED if triggers else ModelHealth.ACTIVE
    reason = "; ".join(triggers) if triggers else None

    return GarchFit(
        params=params,
        loglik=loglik,
        converged=converged,
        iterations=best.iterations,
        persistence=params.persistence,
        unconditional_var=params.unconditional_variance,
        half_life_days=params.half_life_days,
        sigma2_series=sigma2_series,
        std_residuals=std_residuals,
        n=n,
        health=health,
        reason=reason,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def garch_forecast_variance(fit: GarchFit, h: int) -> list[float]:
    """Multi-step variance forecasts ``[sigma^2_{t+1} … sigma^2_{t+h}]``
    in CLOSED FORM (design §9.3).

    Under GARCH(1,1) the ``k``-step-ahead expected variance is

        E[sigma^2_{t+k}] = V_L + (alpha + beta)^{k-1} * (sigma^2_{t+1} - V_L)

    — the one-step forecast decays geometrically toward the unconditional
    variance at the persistence rate. Mathematically identical to iterating
    ``sigma^2_{k+1} = omega + (alpha + beta) sigma^2_k`` (pinned by test),
    but computed directly so a long horizon costs no accumulated rounding.

    ``ValueError`` when ``fit`` has no parameters (an UNAVAILABLE/FAILED fit
    has no forecast — the caller reports the honest null) or ``h < 1``.
    """
    if not isinstance(fit, GarchFit):
        raise ValueError("fit must be a GarchFit")
    h = _check_positive_int(h, name="h")
    params = fit.params
    sigma2_next = fit.sigma2_next
    if params is None or sigma2_next is None:
        raise ValueError(
            f"no GARCH forecast without a fit (health={fit.health}, reason={fit.reason})"
        )
    v_l = params.unconditional_variance
    persistence = params.persistence
    gap = sigma2_next - v_l
    return [v_l + (persistence ** k) * gap for k in range(h)]


def garch_volatility_forecast(
    returns: Sequence[float],
    horizon_days: int = 1,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
    as_of: date | datetime | None = None,
    fit: GarchFit | None = None,
) -> ModelResult:
    """GARCH(1,1) volatility forecast over ``horizon_days`` (design §9.3).

    ``sigma_h = sqrt(sum_{k=1..h} sigma^2_{t+k})`` from
    :func:`garch_forecast_variance` — the standard deviation of the SUM of
    the next ``h`` (conditionally independent, zero-mean) returns, in the
    units of the input (return units for returns, USD/day-summed for a P&L
    series). At ``h = 1`` this is exactly ``sigma_{t+1}``.

    Note this is NOT the ``sqrt(h)`` scaling used for the historical VaR
    rows (contract §1): GARCH knows the term structure of variance, so the
    horizon aggregation is the real sum, and ``params.scaling`` records
    ``"GARCH_TERM_STRUCTURE"`` to distinguish the two on sight.

    ``mode`` is RESEARCH and ``distribution`` is ``"GAUSSIAN_GARCH"``.
    Health mirrors the underlying :class:`GarchFit`: UNAVAILABLE/FAILED
    give ``value=None`` with the fit's reason; DEGRADED still reports the
    number, caveated. Pass an already-computed ``fit`` to avoid re-fitting.
    """
    horizon_days = _check_positive_int(horizon_days, name="horizon_days")
    if fit is None:
        fit = fit_garch(
            returns,
            min_obs=min_obs,
            max_iter=max_iter,
            ljung_box_lags=ljung_box_lags,
            ljung_box_p=ljung_box_p,
            degraded_persistence=degraded_persistence,
            annualization_days=annualization_days,
        )
    elif not isinstance(fit, GarchFit):
        raise ValueError("fit must be a GarchFit or None")
    n = fit.n
    meta = ModelMeta(
        model_name=MODEL_NAME_GARCH,
        model_version=MODEL_VERSION,
        params={
            "min_obs": min_obs,
            "max_iter": max_iter,
            "ljung_box_lags": ljung_box_lags,
            "ljung_box_p": ljung_box_p,
            "degraded_persistence": degraded_persistence,
            "mean_model": "ZERO",
            "innovations": "GAUSSIAN",
            "scaling": "GARCH_TERM_STRUCTURE",
            "mode": str(ModelMode.RESEARCH),
        },
        frequency=DEFAULT_FREQUENCY,
        lookback=n if n >= 1 else None,
        as_of=as_of,
        # §5: GARCH(1,1) is the CONDITIONAL volatility model — Tier 2.
        tier=ModelTier.TIER_2,
        horizon_days=horizon_days,
        distribution=DISTRIBUTION_GAUSSIAN_GARCH,
    )
    diagnostics = dict(fit.diagnostics)
    diagnostics["horizon_days"] = horizon_days
    if fit.params is None:
        reason = fit.reason or "GARCH fit produced no parameters"
        if fit.health is ModelHealth.FAILED:
            return failed(meta, reason, n, diagnostics=diagnostics)
        return unavailable(meta, reason, n, diagnostics=diagnostics)
    path = garch_forecast_variance(fit, horizon_days)
    variance = math.fsum(path)
    sigma = math.sqrt(variance)
    diagnostics["forecast_variance_total"] = variance
    diagnostics["forecast_variance_path"] = tuple(path)
    diagnostics["annualized"] = math.sqrt(path[0] * annualization_days)
    if fit.health is ModelHealth.DEGRADED:
        return degraded(meta, fit.reason or "degraded fit", sigma, n, diagnostics=diagnostics)
    return active(meta, sigma, n, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Filtered historical simulation with GARCH sigma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchScaling:
    """Result of :func:`garch_scaling` (the detail behind
    :func:`garch_scaled_pnl`) — deliberately the same shape as
    ``volatility.VolatilityScaling`` so the two filters are interchangeable.

    - ``scaled``: ``pnl_t x sigma_now / sigma_t`` for every kept ``t``;
    - ``sigma_now``: the one-step-ahead GARCH forecast (``None`` without a fit);
    - ``n_input`` / ``n_used`` / ``dropped``: bookkeeping;
    - ``fit``: the underlying :class:`GarchFit` (health/reason/diagnostics).
    """

    scaled: tuple[float, ...]
    sigma_now: float | None
    n_input: int
    n_used: int
    dropped: int
    fit: GarchFit


def garch_scaling(
    pnl: Sequence[float],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    fit: GarchFit | None = None,
) -> GarchScaling:
    """GARCH-filtered historical simulation with bookkeeping (design §9.3).

    Identical in FORM to the Hull–White EWMA filter in
    ``volatility.volatility_scaling`` — only the conditional-volatility
    model differs: ``sigma_t`` is the GARCH conditional standard deviation
    of day ``t`` (information strictly before ``t``) and ``sigma_now`` is
    the one-step-ahead forecast. Entries with ``sigma_t <= 0`` are dropped,
    never imputed (they cannot occur for a valid fit, where ``omega > 0``
    makes every variance strictly positive — the guard is defensive).

    Unlike the EWMA filter there is NO warm-up to drop: the recursion is
    seeded with the window's own second moment, so every input index has a
    conditional variance. When the fit produced no parameters the result is
    empty (``scaled=()``) and ``fit`` carries the honest reason.
    """
    values = _check_finite(pnl, what="pnl")
    n = len(values)
    if fit is None:
        fit = fit_garch(
            values,
            min_obs=min_obs,
            max_iter=max_iter,
            ljung_box_lags=ljung_box_lags,
            ljung_box_p=ljung_box_p,
            degraded_persistence=degraded_persistence,
        )
    elif not isinstance(fit, GarchFit):
        raise ValueError("fit must be a GarchFit or None")
    sigma2_next = fit.sigma2_next
    if fit.params is None or sigma2_next is None:
        return GarchScaling(
            scaled=(), sigma_now=None, n_input=n, n_used=0, dropped=n, fit=fit
        )
    sigma_now = math.sqrt(sigma2_next)
    scaled: list[float] = []
    for value, var_t in zip(values, fit.sigma2_series):
        if var_t <= 0.0:  # pragma: no cover - impossible for omega > 0
            continue
        scaled.append(value * sigma_now / math.sqrt(var_t))
    return GarchScaling(
        scaled=tuple(scaled),
        sigma_now=sigma_now,
        n_input=n,
        n_used=len(scaled),
        dropped=n - len(scaled),
        fit=fit,
    )


def garch_scaled_pnl(
    pnl: Sequence[float],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    fit: GarchFit | None = None,
) -> list[float]:
    """GARCH-filtered P&L ``pnl_t x sigma_now / sigma_t`` (design §9.3).

    The GARCH counterpart of ``volatility.volatility_scaled_pnl`` and
    deliberately the SAME shape: feed it to ``var_es.historical_var`` /
    ``historical_es`` and the result is the conditional VaR/ES with
    ``distribution="EMPIRICAL_GARCH_SCALED"``. Empty when the fit produced
    no parameters (honest null; see :func:`garch_scaling` for the reason).
    """
    return list(
        garch_scaling(
            pnl,
            min_obs=min_obs,
            max_iter=max_iter,
            ljung_box_lags=ljung_box_lags,
            ljung_box_p=ljung_box_p,
            degraded_persistence=degraded_persistence,
            fit=fit,
        ).scaled
    )


# ---------------------------------------------------------------------------
# Fallback selector (spec §13/§58 fallback hierarchy; design §9.3)
# ---------------------------------------------------------------------------


def conditional_volatility_source(
    returns: Sequence[float],
    *,
    horizon_days: int = 1,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
    as_of: date | datetime | None = None,
) -> tuple[str, ModelResult, str]:
    """Pick the conditional-volatility forecaster: GARCH if it is ACTIVE,
    otherwise EWMA (spec §13/§58 fallback hierarchy; design §9.3).

    Returns ``(source, result, reason)`` where ``source`` is
    ``"GARCH"`` or ``"EWMA"`` and ``reason`` always states WHY that
    forecaster is the one you got:

    - GARCH health ACTIVE ⇒ ``("GARCH", garch_result, "GARCH(1,1) ACTIVE:
      persistence=…, Ljung-Box p=…")``;
    - anything else ⇒ ``("EWMA", ewma_result, "GARCH not ACTIVE
      (health=…): <the fit's reason> — falling back to EWMA(lambda=…)")``.

    A DEGRADED GARCH fit is NOT used: spec §13 says a model that fails
    diagnostics falls back to a simpler one, and the simpler one here (EWMA)
    has no diagnostics to fail. The degraded GARCH number is still available
    to a caller that wants it (call :func:`garch_volatility_forecast`
    directly) — it is just not what the fallback hierarchy hands back.

    The EWMA branch is itself honest: if the series is too short even for
    EWMA, its ``ModelResult`` is UNAVAILABLE with its own reason and the
    caller reports no conditional volatility at all. Nothing here raises for
    a data problem, and nothing halts because GARCH failed (spec §58).
    """
    garch_result = garch_volatility_forecast(
        returns,
        horizon_days,
        min_obs=min_obs,
        max_iter=max_iter,
        ljung_box_lags=ljung_box_lags,
        ljung_box_p=ljung_box_p,
        degraded_persistence=degraded_persistence,
        as_of=as_of,
    )
    if garch_result.health is ModelHealth.ACTIVE:
        persistence = garch_result.diagnostics.get("persistence")
        lb_p = garch_result.diagnostics.get("ljung_box_p")
        reason = (
            f"GARCH(1,1) ACTIVE: persistence={persistence:.6f}, "
            f"Ljung-Box(m={ljung_box_lags}) p={lb_p:.4g}"
            if persistence is not None and lb_p is not None
            else "GARCH(1,1) ACTIVE"
        )
        return SOURCE_GARCH, garch_result, reason
    ewma_result = ewma_volatility_forecast(
        returns, lam=lam, init_obs=init_obs, as_of=as_of
    )
    reason = (
        f"GARCH not ACTIVE (health={garch_result.health}): "
        f"{garch_result.reason} — falling back to EWMA(lambda={lam})"
    )
    return SOURCE_EWMA, ewma_result, reason


def conditional_scaled_pnl_source(
    pnl: Sequence[float],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    max_iter: int = DEFAULT_MAX_ITER,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
    degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
) -> tuple[str, list[float], str]:
    """The same fallback rule for the FILTERED P&L series feeding conditional
    VaR/ES: GARCH-scaled when the fit is ACTIVE, EWMA-scaled otherwise.

    Returns ``(source, scaled_pnl, reason)``. Distribution labels for the
    caller: ``"EMPIRICAL_GARCH_SCALED"`` for ``"GARCH"``,
    ``"EMPIRICAL_VOL_SCALED"`` for ``"EWMA"``.
    """
    scaling = garch_scaling(
        pnl,
        min_obs=min_obs,
        max_iter=max_iter,
        ljung_box_lags=ljung_box_lags,
        ljung_box_p=ljung_box_p,
        degraded_persistence=degraded_persistence,
    )
    if scaling.fit.health is ModelHealth.ACTIVE:
        return (
            SOURCE_GARCH,
            list(scaling.scaled),
            f"GARCH(1,1) ACTIVE: persistence={scaling.fit.persistence:.6f}",
        )
    reason = (
        f"GARCH not ACTIVE (health={scaling.fit.health}): "
        f"{scaling.fit.reason} — falling back to EWMA(lambda={lam})"
    )
    return SOURCE_EWMA, volatility_scaled_pnl(pnl, lam=lam, init_obs=init_obs), reason


# ---------------------------------------------------------------------------
# RiskModel class (spec §4 registry; RESEARCH mode)
# ---------------------------------------------------------------------------


class Garch11Model(BaseRiskModel):
    """Registered wrapper around :func:`garch_volatility_forecast`
    (``"garch11"``).

    ``mode`` defaults to :attr:`ModelMode.RESEARCH` — one step BELOW the
    library's SHADOW default (spec §70): this model is not even the shadow
    conditional-volatility forecaster; EWMA is. Promotion to SHADOW requires
    the §63 comparison (design §9.4: Kupiec p at least EWMA's over ≥ 250
    forecast days, Christoffersen p ≥ 0.05, no FAILED diagnostics) and is a
    user action, not an import.
    """

    name = MODEL_NAME_GARCH
    version = MODEL_VERSION
    mode = ModelMode.RESEARCH
    #: Spec §5: conditional volatility model.
    tier = ModelTier.TIER_2
    distribution = DISTRIBUTION_GAUSSIAN_GARCH
    frequency = DEFAULT_FREQUENCY

    def __init__(
        self,
        *,
        horizon_days: int = 1,
        min_obs: int = DEFAULT_MIN_OBS,
        max_iter: int = DEFAULT_MAX_ITER,
        ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
        ljung_box_p: float = DEFAULT_LJUNG_BOX_P,
        degraded_persistence: float = DEFAULT_DEGRADED_PERSISTENCE,
        mode: Any = None,
    ) -> None:
        super().__init__(mode=mode if mode is not None else ModelMode.RESEARCH)
        self.horizon_days = _check_positive_int(horizon_days, name="horizon_days")
        self.min_obs = _check_positive_int(min_obs, name="min_obs", minimum=2)
        self.max_iter = _check_positive_int(max_iter, name="max_iter")
        self.ljung_box_lags = _check_positive_int(ljung_box_lags, name="ljung_box_lags")
        self.ljung_box_p = _check_probability(ljung_box_p, name="ljung_box_p")
        self.degraded_persistence = _check_probability(
            degraded_persistence, name="degraded_persistence"
        )

    def params(self) -> Mapping[str, Any]:
        return {
            "horizon_days": self.horizon_days,
            "min_obs": self.min_obs,
            "max_iter": self.max_iter,
            "ljung_box_lags": self.ljung_box_lags,
            "ljung_box_p": self.ljung_box_p,
            "degraded_persistence": self.degraded_persistence,
            "mean_model": "ZERO",
            "innovations": "GAUSSIAN",
        }

    def calculate(
        self,
        returns: Sequence[float],
        *,
        as_of: date | datetime | None = None,
    ) -> ModelResult:
        """Fit and forecast with the bound parameters (never claims more than
        ACTIVE-if-computed; validation is a separate step, spec §57)."""
        return garch_volatility_forecast(
            returns,
            self.horizon_days,
            min_obs=self.min_obs,
            max_iter=self.max_iter,
            ljung_box_lags=self.ljung_box_lags,
            ljung_box_p=self.ljung_box_p,
            degraded_persistence=self.degraded_persistence,
            as_of=as_of,
        )


MODEL_CLASSES: tuple[type[BaseRiskModel], ...] = (Garch11Model,)


def register_models() -> tuple[str, ...]:
    """Register the GARCH model; IDEMPOTENT (same contract as
    ``var_es.register_models``). Called once at import time."""
    for cls in MODEL_CLASSES:
        register(cls(), replace=True)
    return tuple(sorted(cls.name for cls in MODEL_CLASSES))


register_models()


__all__ = [
    "DEFAULT_ANNUALIZATION_DAYS",
    "DEFAULT_DEGRADED_PERSISTENCE",
    "DEFAULT_FREQUENCY",
    "DEFAULT_LJUNG_BOX_LAGS",
    "DEFAULT_LJUNG_BOX_P",
    "DEFAULT_MAX_ITER",
    "DEFAULT_MIN_OBS",
    "DISTRIBUTION_EMPIRICAL_GARCH_SCALED",
    "DISTRIBUTION_GAUSSIAN_GARCH",
    "INIT_ALPHA",
    "INIT_BETA",
    "MODEL_NAME_GARCH",
    "MODEL_VERSION",
    "OMEGA_FLOOR_RATIO",
    "PERSISTENCE_MAX",
    "SOURCE_EWMA",
    "SOURCE_GARCH",
    "Garch11Model",
    "GarchFit",
    "GarchParams",
    "GarchScaling",
    "conditional_scaled_pnl_source",
    "conditional_volatility_source",
    "fit_garch",
    "garch_forecast_variance",
    "garch_scaled_pnl",
    "garch_scaling",
    "garch_variance_path",
    "garch_volatility_forecast",
    "ljung_box",
    "register_models",
]
