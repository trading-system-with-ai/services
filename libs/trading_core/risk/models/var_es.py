"""Value-at-Risk & Expected Shortfall (risk spec §6, §7, §8, §12; Phase B
design contract §2.3).

Pure, deterministic, stdlib-only (house rule): ``math`` and
``statistics.NormalDist`` only, ``math.fsum`` for every sum feeding a
statistic, no numpy. Everything here is SHADOW/RESEARCH — nothing alters a
Tier 0 decision (``risk/engine.py`` stays byte-identical).

Sign convention (contract §1). A **P&L series** is gain-positive
(``pnl[t] > 0`` = money made). A **loss** is ``L_t = −pnl_t``. VaR and ES
are reported as **losses: positive = money lost**. If the α-tail of losses
is negative (the book gains even in its worst tail) the negative number is
reported honestly — never floored at 0. The UI formats; the estimator does
not lie.

The ONE quantile convention for the whole platform (contract §2.3), so VaR,
ES and the Euler ES contributions in ``contribution.py`` stay mutually
coherent:

- Sort losses **descending**: ``L(1) ≥ L(2) ≥ … ≥ L(n)``.
- Tail size ``k = ceil(n·(1 − α))``. (n=600: 30 @95%, 6 @99%;
  n=250: 13 @95%, 3 @99%; n=10: 1 @95%.)
- **Historical VaR_α = L(k)** — the k-th largest loss (empirical upper
  quantile, ``P(L ≥ VaR) ≥ 1 − α``).
- **Historical ES_α = mean(L(1..k))** — the plain average of the k largest
  losses. Hence ``ES ≥ VaR`` always, with equality iff ``k = 1`` or the
  tail is tied. ES as a plain tail average is exactly what makes
  ``Σ_i RC^ES_i = ES_α`` hold in ``contribution.py`` (contract §3.3).
- **Gaussian VaR_α = −μ + z_α·σ**, **Gaussian ES_α = −μ + σ·φ(z_α)/(1 − α)**,
  with ``μ``/``σ`` the sample mean and standard deviation (ddof=1) of the
  P&L, ``z_α = NormalDist().inv_cdf(α)`` and ``φ`` the standard normal pdf.

Horizon (contract §1, §2.3). Only 1 day is *estimated*. ``horizon_days
h > 1`` is **√h scaling of the 1-day number**, labelled
``scaling="SQRT_TIME"`` in the diagnostics and meta params so a reader can
never mistake it for an estimate:

- Historical: ``VaR_h = VaR_1 × √h`` (the whole number scales — an
  empirical quantile carries no separable drift term).
- Gaussian: the drift and the shock scale differently under i.i.d.
  aggregation, so ``VaR_h = −μ·h + z·σ·√h`` and
  ``ES_h = −μ·h + σ·φ(z)/(1 − α)·√h`` — computed from the 1-day μ and σ,
  never re-estimated on overlapping h-day windows.

Health (contract §1 honest nulls, §2.3). ``min_obs`` defaults to **60** for
``α < 0.99`` and **250** for ``α ≥ 0.99`` (so ``k ≥ 3`` at 99%):

- ``n < min_obs`` ⇒ ``UNAVAILABLE``, ``value=None``, ``reason`` carrying the
  real numbers (``"n=17 < min_obs=60 (k=1)"``). Never a fabricated 0.
- ``min_obs ≤ n < 2·min_obs`` ⇒ ``DEGRADED`` — the number is real but the
  tail is thin (``"small sample: n=80 < 2*min_obs=120 (tail k=4)"``).
- ``n ≥ 2·min_obs`` ⇒ ``ACTIVE``.

Diagnostics on every result: ``n``, ``tail_size`` (k), ``mu``, ``sigma``,
``scaling`` ("NONE" for h=1, "SQRT_TIME" for h>1), ``horizon_days``.

Conditional VaR/ES (spec §12) are NOT a separate estimator: they are the
historical estimators above applied to
``volatility.volatility_scaled_pnl`` (Hull–White filtered historical
simulation), labelled ``distribution="EMPIRICAL_VOL_SCALED"`` with
``params.lambda``. See :func:`conditional_var` / :func:`conditional_es`.

Malformed input raises ``ValueError`` (α outside (0.5, 1), ``horizon_days
< 1``, non-finite P&L, ``min_obs < 1``); missing data never raises — it
degrades health.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from statistics import NormalDist
from typing import Any

from libs.trading_core.risk.models.base import (
    BaseRiskModel,
    ModelHealth,
    ModelMeta,
    ModelResult,
    ModelTier,
    active,
    degraded,
    register,
    unavailable,
)
from libs.trading_core.risk.models.volatility import (
    DEFAULT_INIT_OBS,
    DEFAULT_LAMBDA,
    volatility_scaled_pnl,
)

#: Estimator version (contract §4): arithmetic change ⇒ MAJOR bump.
MODEL_VERSION = "1.0.0"

#: Registry / ``ModelMeta.model_name`` labels (contract §2.3).
MODEL_NAME_HISTORICAL_VAR = "historical_var"
MODEL_NAME_HISTORICAL_ES = "historical_es"
MODEL_NAME_GAUSSIAN_VAR = "gaussian_var"
MODEL_NAME_GAUSSIAN_ES = "gaussian_es"
MODEL_NAME_CONDITIONAL_VAR = "conditional_var"
MODEL_NAME_CONDITIONAL_ES = "conditional_es"

#: Distribution labels recorded in ``ModelMeta.distribution`` (spec §44).
DISTRIBUTION_EMPIRICAL = "EMPIRICAL"
DISTRIBUTION_NORMAL = "NORMAL"
DISTRIBUTION_EMPIRICAL_VOL_SCALED = "EMPIRICAL_VOL_SCALED"

#: Documented defaults (contract §2.3). Every one is a keyword parameter.
DEFAULT_MIN_OBS_95 = 60      # α < HIGH_CONFIDENCE
DEFAULT_MIN_OBS_99 = 250     # α >= HIGH_CONFIDENCE ⇒ k >= 3 at 99%
HIGH_CONFIDENCE = 0.99       # the boundary at which min_obs steps up
DEGRADED_MULTIPLE = 2        # ACTIVE needs n >= DEGRADED_MULTIPLE * min_obs
DEFAULT_FREQUENCY = "1D"

#: ``scaling`` diagnostic values (contract §1: multi-day is labelled, never gated).
SCALING_NONE = "NONE"
SCALING_SQRT_TIME = "SQRT_TIME"


# ---------------------------------------------------------------------------
# Guards & shared helpers
# ---------------------------------------------------------------------------


def _check_confidence(confidence: float) -> float:
    """``confidence ∈ (0.5, 1)`` (contract §1); anything else is malformed."""
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not (0.5 < confidence < 1.0)
    ):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence!r}")
    return float(confidence)


def _check_horizon(horizon_days: int) -> int:
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or horizon_days < 1
    ):
        raise ValueError(f"horizon_days must be an int >= 1, got {horizon_days!r}")
    return horizon_days


def _check_finite(values: Sequence[float], *, what: str = "pnl") -> None:
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}[{i}] must be a finite number, got {v!r}")


def default_min_obs(confidence: float) -> int:
    """Documented ``min_obs`` default for ``confidence`` (contract §2.3).

    ``250`` at or above 99% (so the 99% tail holds ``k = ceil(250·0.01) = 3``
    observations), ``60`` below it. Callers may always override.
    """
    return (
        DEFAULT_MIN_OBS_99
        if _check_confidence(confidence) >= HIGH_CONFIDENCE
        else DEFAULT_MIN_OBS_95
    )


def _resolve_min_obs(confidence: float, min_obs: int | None) -> int:
    if min_obs is None:
        return default_min_obs(confidence)
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 1:
        raise ValueError(f"min_obs must be an int >= 1 or None, got {min_obs!r}")
    return min_obs


#: Relative tolerance for snapping ``n·(1 − α)`` to an integer in :func:`tail_size`.
#: ``1 − 0.95`` is 0.05000000000000004 in binary floating point, so the naive
#: ``ceil(600 · (1 − 0.95))`` is ``ceil(30.00000000000003) = 31`` — one more
#: tail observation than the contract's own worked example (n=600 ⇒ k=30).
#: A representation error of a few ULPs must not change the estimator, so a
#: product within this relative tolerance of a whole number IS that whole
#: number. Real fractional tails (12.5 → 13) are untouched: they sit half a
#: unit away, ~14 orders of magnitude beyond the tolerance.
TAIL_SIZE_REL_TOL = 1e-9


def tail_size(n: int, confidence: float) -> int:
    """``k = ceil(n·(1 − α))`` — the platform tail-size convention (contract §2.3).

    The ceiling is taken on the EXACT product, not on its binary
    floating-point image: ``n·(1 − α)`` is snapped to a whole number when it
    is within ``TAIL_SIZE_REL_TOL`` of one (see that constant for why). This
    reproduces the contract's worked examples exactly.

    Hand-check: ``n=600, α=0.95`` ⇒ ``ceil(30) = 30``; ``n=600, α=0.99`` ⇒
    ``ceil(6) = 6``; ``n=250, α=0.95`` ⇒ ``ceil(12.5) = 13``; ``n=250,
    α=0.99`` ⇒ ``ceil(2.5) = 3``; ``n=10, α=0.95`` ⇒ ``ceil(0.5) = 1``. The
    result is clamped to ``[1, n]`` for ``n ≥ 1`` so a tail is never empty.
    """
    if n <= 0:
        return 0
    raw = n * (1.0 - _check_confidence(confidence))
    nearest = round(raw)
    if nearest >= 0 and math.isclose(raw, nearest, rel_tol=TAIL_SIZE_REL_TOL, abs_tol=0.0):
        k = int(nearest)
    else:
        k = math.ceil(raw)
    return max(1, min(n, k))


def sorted_losses(pnl: Sequence[float]) -> list[float]:
    """Losses ``L_t = −pnl_t`` sorted DESCENDING (``L(1)`` is the worst loss).

    The single ordering used by VaR, ES and the Euler ES tail set, so the
    three agree on which dates are "the tail".
    """
    return sorted((-float(p) for p in pnl), reverse=True)


def _health_for(n: int, min_obs: int, k: int) -> tuple[ModelHealth, str | None]:
    """Sample-size health band (contract §2.3).

    ``n < min_obs`` ⇒ UNAVAILABLE; ``n < 2·min_obs`` ⇒ DEGRADED (thin tail);
    otherwise ACTIVE. Reasons carry the real numbers.
    """
    if n < min_obs:
        return (
            ModelHealth.UNAVAILABLE,
            f"n={n} < min_obs={min_obs} (tail k={k})",
        )
    threshold = DEGRADED_MULTIPLE * min_obs
    if n < threshold:
        return (
            ModelHealth.DEGRADED,
            f"small sample: n={n} < {DEGRADED_MULTIPLE}*min_obs={threshold} (small tail: k={k})",
        )
    return ModelHealth.ACTIVE, None


def _mean_and_stdev(values: Sequence[float]) -> tuple[float | None, float | None]:
    """``(mean, stdev)`` with ddof=1, two-pass, ``math.fsum``.

    ``(None, None)`` for an empty series and ``(mean, None)`` for a single
    observation — ddof=1 has no answer with ``n < 2``, and a fabricated 0
    would be a lie (contract §1).
    """
    n = len(values)
    if n == 0:
        return None, None
    mean = math.fsum(values) / n
    if n < 2:
        return mean, None
    var = math.fsum((v - mean) * (v - mean) for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _scaling_label(horizon_days: int) -> str:
    return SCALING_NONE if horizon_days == 1 else SCALING_SQRT_TIME


def _meta(
    name: str,
    *,
    distribution: str,
    confidence: float,
    horizon_days: int,
    min_obs: int,
    n: int,
    as_of: date | datetime | None,
    extra_params: Mapping[str, Any] | None = None,
    tier: ModelTier = ModelTier.TIER_1,
) -> ModelMeta:
    params: dict[str, Any] = {
        "confidence": confidence,
        "horizon_days": horizon_days,
        "min_obs": min_obs,
        "scaling": _scaling_label(horizon_days),
    }
    if extra_params:
        params.update(extra_params)
    return ModelMeta(
        model_name=name,
        model_version=MODEL_VERSION,
        params=params,
        frequency=DEFAULT_FREQUENCY,
        lookback=n if n >= 1 else None,
        as_of=as_of,
        confidence=confidence,
        horizon_days=horizon_days,
        distribution=distribution,
        tier=tier,
    )


def _diagnostics(
    *,
    n: int,
    k: int,
    mu: float | None,
    sigma: float | None,
    horizon_days: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "n": n,
        "tail_size": k,
        "mu": mu,
        "sigma": sigma,
        "scaling": _scaling_label(horizon_days),
        "horizon_days": horizon_days,
    }
    if extra:
        diag.update(extra)
    return diag


def _result(
    meta: ModelMeta,
    value: float,
    n: int,
    health: ModelHealth,
    reason: str | None,
    diagnostics: Mapping[str, Any],
) -> ModelResult:
    """ACTIVE or DEGRADED result (UNAVAILABLE is built by the callers directly)."""
    if health is ModelHealth.DEGRADED:
        assert reason is not None
        return degraded(meta, reason, value, n, diagnostics=diagnostics)
    return active(meta, value, n, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Historical (empirical) estimators
# ---------------------------------------------------------------------------


def _historical(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int,
    *,
    min_obs: int | None,
    as_of: date | datetime | None,
    name: str,
    distribution: str,
    extra_params: Mapping[str, Any] | None,
    expected_shortfall: bool,
    tier: ModelTier = ModelTier.TIER_1,
) -> ModelResult:
    """Shared body of :func:`historical_var` / :func:`historical_es`.

    ``tier`` (spec §5) is TIER_1 for the unconditional historical
    estimators and is overridden to TIER_2 by :func:`_conditional`, whose
    input has already been through a CONDITIONAL volatility filter — the
    quantile convention is the same, the model family is not.
    """
    confidence = _check_confidence(confidence)
    horizon_days = _check_horizon(horizon_days)
    resolved_min_obs = _resolve_min_obs(confidence, min_obs)
    _check_finite(pnl)

    n = len(pnl)
    k = tail_size(n, confidence)
    meta = _meta(
        name,
        distribution=distribution,
        confidence=confidence,
        horizon_days=horizon_days,
        min_obs=resolved_min_obs,
        n=n,
        as_of=as_of,
        extra_params=extra_params,
        tier=tier,
    )
    mu, sigma = _mean_and_stdev(pnl)
    diagnostics = _diagnostics(
        n=n, k=k, mu=mu, sigma=sigma, horizon_days=horizon_days
    )
    health, reason = _health_for(n, resolved_min_obs, k)
    if health is ModelHealth.UNAVAILABLE:
        assert reason is not None
        return unavailable(meta, reason, n, diagnostics=diagnostics)

    losses = sorted_losses(pnl)
    if expected_shortfall:
        one_day = math.fsum(losses[:k]) / k
    else:
        one_day = losses[k - 1]
    value = one_day * math.sqrt(horizon_days)
    diagnostics["one_day"] = one_day
    return _result(meta, value, n, health, reason, diagnostics)


def historical_var(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Historical (empirical) VaR — the k-th largest loss (contract §2.3).

    ``L_t = −pnl_t`` sorted descending, ``k = ceil(n·(1 − α))``,
    ``VaR_α = L(k)``; multi-day is ``VaR_1 × √h`` labelled
    ``scaling="SQRT_TIME"`` (``distribution="EMPIRICAL"``).

    Hand-check (``pnl = [−50, −30, −20, −10, −5, 5, 10, 20, 30, 120]``,
    ``min_obs=5``): losses descending are
    ``[50, 30, 20, 10, 5, −5, −10, −20, −30, −120]``; at α=0.8,
    ``k = ceil(10·0.2) = 2`` ⇒ ``VaR = L(2) = 30``.

    Health: ``n < min_obs`` ⇒ UNAVAILABLE (``value=None``); ``n <
    2·min_obs`` ⇒ DEGRADED; else ACTIVE. ``min_obs=None`` resolves to
    :func:`default_min_obs` (60, or 250 at α ≥ 99%).
    """
    return _historical(
        pnl,
        confidence,
        horizon_days,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_HISTORICAL_VAR,
        distribution=DISTRIBUTION_EMPIRICAL,
        extra_params=None,
        expected_shortfall=False,
    )


def historical_es(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Historical (empirical) ES — mean of the k largest losses (contract §2.3).

    ``ES_α = (L(1) + … + L(k)) / k`` with the same ``k = ceil(n·(1 − α))`` as
    :func:`historical_var`, so ``ES ≥ VaR`` always (equality iff ``k = 1`` or
    the tail is tied) and ``Σ_i RC^ES_i = ES_α`` holds exactly in
    ``contribution.py``. Multi-day is ``ES_1 × √h`` (``scaling="SQRT_TIME"``).

    Hand-check (same 10-point series, α=0.7, ``min_obs=5``): losses
    descending are ``[50, 30, 20, 10, 5, −5, −10, −20, −30, −120]``;
    ``k = ceil(10·0.3) = 3`` (the exact product — see :func:`tail_size`) ⇒
    ``ES = (50 + 30 + 20)/3 = 100/3 = 33.333…``, and the matching
    ``VaR = L(3) = 20`` is below it as invariant §3.1 requires.
    """
    return _historical(
        pnl,
        confidence,
        horizon_days,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_HISTORICAL_ES,
        distribution=DISTRIBUTION_EMPIRICAL,
        extra_params=None,
        expected_shortfall=True,
    )


# ---------------------------------------------------------------------------
# Gaussian (parametric) estimators
# ---------------------------------------------------------------------------


def _gaussian(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int,
    *,
    min_obs: int | None,
    as_of: date | datetime | None,
    name: str,
    expected_shortfall: bool,
) -> ModelResult:
    """Shared body of :func:`gaussian_var` / :func:`gaussian_es`."""
    confidence = _check_confidence(confidence)
    horizon_days = _check_horizon(horizon_days)
    resolved_min_obs = _resolve_min_obs(confidence, min_obs)
    _check_finite(pnl)

    n = len(pnl)
    k = tail_size(n, confidence)
    meta = _meta(
        name,
        distribution=DISTRIBUTION_NORMAL,
        confidence=confidence,
        horizon_days=horizon_days,
        min_obs=resolved_min_obs,
        n=n,
        as_of=as_of,
    )
    mu, sigma = _mean_and_stdev(pnl)
    normal = NormalDist()
    z = normal.inv_cdf(confidence)
    diagnostics = _diagnostics(
        n=n, k=k, mu=mu, sigma=sigma, horizon_days=horizon_days, extra={"z": z}
    )
    health, reason = _health_for(n, resolved_min_obs, k)
    if health is ModelHealth.UNAVAILABLE:
        assert reason is not None
        return unavailable(meta, reason, n, diagnostics=diagnostics)
    if mu is None or sigma is None:  # pragma: no cover - min_obs >= 1 guards n >= 2 in practice
        return unavailable(
            meta,
            f"sample moments need n >= 2 (ddof=1), got n={n}",
            n,
            diagnostics=diagnostics,
        )

    # Drift scales with h, the shock with √h (i.i.d. aggregation).
    drift = -mu * horizon_days
    shock = sigma * math.sqrt(horizon_days)
    if expected_shortfall:
        multiplier = normal.pdf(z) / (1.0 - confidence)
        diagnostics["es_multiplier"] = multiplier
        value = drift + shock * multiplier
        one_day = -mu + sigma * multiplier
    else:
        value = drift + shock * z
        one_day = -mu + sigma * z
    diagnostics["one_day"] = one_day
    return _result(meta, value, n, health, reason, diagnostics)


def gaussian_var(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Parametric normal VaR ``−μ + z_α·σ`` (contract §2.3).

    ``μ``/``σ`` are the sample mean and standard deviation (ddof=1,
    ``math.fsum``) of the gain-positive P&L; ``z_α =
    NormalDist().inv_cdf(α)``. Multi-day: ``−μ·h + z_α·σ·√h``
    (``scaling="SQRT_TIME"``, ``distribution="NORMAL"``).

    Hand-check (``pnl = [−50, −30, −20, −10, −5, 5, 10, 20, 30, 120]``,
    ``min_obs=5``): ``μ = 70/10 = 7``; ``Σ(p − μ)² = 19260`` so
    ``σ² = 19260/9 = 2140`` and ``σ = 46.26013402488151``; at α=0.95,
    ``z = 1.6448536269514715`` ⇒ ``VaR = −7 + 1.6448536269514715 ×
    46.26013402488151 = 69.09114923408752``.

    Diagnostics carry ``z`` alongside ``mu``/``sigma`` so the number is
    reproducible by hand.
    """
    return _gaussian(
        pnl,
        confidence,
        horizon_days,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_GAUSSIAN_VAR,
        expected_shortfall=False,
    )


def gaussian_es(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Parametric normal ES ``−μ + σ·φ(z_α)/(1 − α)`` (contract §2.3).

    ``φ`` is the standard normal pdf and ``z_α = NormalDist().inv_cdf(α)``;
    ``μ``/``σ`` sample (ddof=1). Because ``φ(z)/(1 − α) > z`` for every
    ``α ∈ (0.5, 1)``, Gaussian ES ≥ Gaussian VaR always (contract §3.1).
    Multi-day: ``−μ·h + σ·√h·φ(z)/(1 − α)``.

    Hand-check (same series, α=0.95): ``φ(1.6448536269514715) =
    0.10314917819929559``, ``/0.05 = 2.0629835639859117`` ⇒
    ``ES = −7 + 46.26013402488151 × 2.0629835639859117 =
    88.42137093013326``. Diagnostics carry ``es_multiplier = φ(z)/(1 − α)``.
    """
    return _gaussian(
        pnl,
        confidence,
        horizon_days,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_GAUSSIAN_ES,
        expected_shortfall=True,
    )


# ---------------------------------------------------------------------------
# Conditional (filtered-HS) convenience wrappers — spec §12
# ---------------------------------------------------------------------------


def _conditional(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int,
    *,
    lam: float,
    init_obs: int,
    min_obs: int | None,
    as_of: date | datetime | None,
    name: str,
    expected_shortfall: bool,
) -> ModelResult:
    """Historical estimator over the Hull–White volatility-scaled P&L."""
    scaled = volatility_scaled_pnl(pnl, lam=lam, init_obs=init_obs)
    result = _historical(
        scaled,
        confidence,
        horizon_days,
        min_obs=min_obs,
        as_of=as_of,
        name=name,
        distribution=DISTRIBUTION_EMPIRICAL_VOL_SCALED,
        extra_params={"lambda": lam, "init_obs": init_obs},
        expected_shortfall=expected_shortfall,
        # §5: the filter in front of this estimator is EWMA — a CONDITIONAL
        # volatility model — so the view is Tier 2 even though the quantile
        # arithmetic behind it is the Tier 1 historical one.
        tier=ModelTier.TIER_2,
    )
    # Record the filtering that happened BEFORE the estimator saw the data:
    # sample_size is the scaled length, so say what was dropped.
    diagnostics = dict(result.diagnostics)
    diagnostics["n_input"] = len(pnl)
    diagnostics["n_scaled"] = len(scaled)
    diagnostics["dropped_warmup"] = len(pnl) - len(scaled)
    diagnostics["lambda"] = lam
    diagnostics["init_obs"] = init_obs
    return ModelResult(
        value=result.value,
        health=result.health,
        reason=result.reason,
        sample_size=result.sample_size,
        meta=result.meta,
        diagnostics=diagnostics,
    )


def conditional_var(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Conditional (filtered-HS) VaR — spec §12; contract §2.3 / §2.4.

    :func:`historical_var` applied to
    ``volatility.volatility_scaled_pnl(pnl, lam=lam, init_obs=init_obs)``
    — Hull–White filtering rescales every past P&L to TODAY's volatility
    (``pnl*_t = pnl_t × σ_now/σ_t``), so a quiet-market history does not
    understate a turbulent present. No separate estimator: the quantile
    convention is exactly the historical one.

    ``distribution="EMPIRICAL_VOL_SCALED"``; ``params.lambda`` and
    ``params.init_obs`` record the filter. Warm-up entries are dropped by
    the scaler, so ``sample_size`` is the SCALED length (``n_input``,
    ``n_scaled`` and ``dropped_warmup`` are in the diagnostics) and the
    ``min_obs`` bands apply to it.
    """
    return _conditional(
        pnl,
        confidence,
        horizon_days,
        lam=lam,
        init_obs=init_obs,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_CONDITIONAL_VAR,
        expected_shortfall=False,
    )


def conditional_es(
    pnl: Sequence[float],
    confidence: float,
    horizon_days: int = 1,
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
    min_obs: int | None = None,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Conditional (filtered-HS) ES — :func:`historical_es` over the
    volatility-scaled P&L. See :func:`conditional_var` for the filtering."""
    return _conditional(
        pnl,
        confidence,
        horizon_days,
        lam=lam,
        init_obs=init_obs,
        min_obs=min_obs,
        as_of=as_of,
        name=MODEL_NAME_CONDITIONAL_ES,
        expected_shortfall=True,
    )


# ---------------------------------------------------------------------------
# RiskModel classes (spec §4 registry; contract §2.2, §2.3)
# ---------------------------------------------------------------------------


class _VarEsModel(BaseRiskModel):
    """Shared plumbing for the four VaR/ES models.

    Parameters (``confidence``, ``horizon_days``, ``min_obs``) are bound at
    construction and recorded in ``ModelMeta.params``; ``calculate(pnl)``
    takes only the data. ``mode`` defaults to SHADOW (spec §70) — these
    numbers can never feed a veto until promoted.
    """

    version = MODEL_VERSION
    frequency = DEFAULT_FREQUENCY
    #: Spec §5: unconditional estimators over a fixed historical window.
    tier = ModelTier.TIER_1

    def __init__(
        self,
        *,
        confidence: float = 0.95,
        horizon_days: int = 1,
        min_obs: int | None = None,
        mode: Any = None,
    ) -> None:
        super().__init__(mode=mode)
        self.confidence = _check_confidence(confidence)
        self.horizon_days = _check_horizon(horizon_days)
        self.min_obs = _resolve_min_obs(self.confidence, min_obs)

    def params(self) -> Mapping[str, Any]:
        return {
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "min_obs": self.min_obs,
        }

    def _estimator(self, pnl: Sequence[float], as_of: date | datetime | None) -> ModelResult:
        raise NotImplementedError  # pragma: no cover - abstract-ish

    def calculate(
        self,
        pnl: Sequence[float],
        *,
        as_of: date | datetime | None = None,
    ) -> ModelResult:
        """Estimate on ``pnl`` with the bound parameters (never claims more
        than ACTIVE-if-computed; validation is a separate step, spec §57)."""
        return self._estimator(pnl, as_of)


class HistoricalVaRModel(_VarEsModel):
    """Registered wrapper around :func:`historical_var` (``"historical_var"``)."""

    name = MODEL_NAME_HISTORICAL_VAR
    distribution = DISTRIBUTION_EMPIRICAL

    def _estimator(self, pnl: Sequence[float], as_of: date | datetime | None) -> ModelResult:
        return historical_var(
            pnl, self.confidence, self.horizon_days, min_obs=self.min_obs, as_of=as_of
        )


class HistoricalESModel(_VarEsModel):
    """Registered wrapper around :func:`historical_es` (``"historical_es"``)."""

    name = MODEL_NAME_HISTORICAL_ES
    distribution = DISTRIBUTION_EMPIRICAL

    def _estimator(self, pnl: Sequence[float], as_of: date | datetime | None) -> ModelResult:
        return historical_es(
            pnl, self.confidence, self.horizon_days, min_obs=self.min_obs, as_of=as_of
        )


class GaussianVaRModel(_VarEsModel):
    """Registered wrapper around :func:`gaussian_var` (``"gaussian_var"``)."""

    name = MODEL_NAME_GAUSSIAN_VAR
    distribution = DISTRIBUTION_NORMAL

    def _estimator(self, pnl: Sequence[float], as_of: date | datetime | None) -> ModelResult:
        return gaussian_var(
            pnl, self.confidence, self.horizon_days, min_obs=self.min_obs, as_of=as_of
        )


class GaussianESModel(_VarEsModel):
    """Registered wrapper around :func:`gaussian_es` (``"gaussian_es"``)."""

    name = MODEL_NAME_GAUSSIAN_ES
    distribution = DISTRIBUTION_NORMAL

    def _estimator(self, pnl: Sequence[float], as_of: date | datetime | None) -> ModelResult:
        return gaussian_es(
            pnl, self.confidence, self.horizon_days, min_obs=self.min_obs, as_of=as_of
        )


#: The four models this module registers, in registry-name order.
MODEL_CLASSES: tuple[type[_VarEsModel], ...] = (
    GaussianESModel,
    GaussianVaRModel,
    HistoricalESModel,
    HistoricalVaRModel,
)


def register_models() -> tuple[str, ...]:
    """Register the four VaR/ES models; IDEMPOTENT (safe to call repeatedly).

    Called once at import time so ``get("historical_var")`` works for any
    importer. Re-registration ``replace``s the existing instance rather than
    raising, so a test that calls ``clear_for_tests()`` can restore the
    registry by calling this again. Returns the registered names, sorted.
    """
    for cls in MODEL_CLASSES:
        register(cls(), replace=True)
    return tuple(sorted(cls.name for cls in MODEL_CLASSES))


register_models()


__all__ = [
    "DEFAULT_MIN_OBS_95",
    "DEFAULT_MIN_OBS_99",
    "DEGRADED_MULTIPLE",
    "DISTRIBUTION_EMPIRICAL",
    "DISTRIBUTION_EMPIRICAL_VOL_SCALED",
    "DISTRIBUTION_NORMAL",
    "HIGH_CONFIDENCE",
    "MODEL_CLASSES",
    "MODEL_NAME_CONDITIONAL_ES",
    "MODEL_NAME_CONDITIONAL_VAR",
    "MODEL_NAME_GAUSSIAN_ES",
    "MODEL_NAME_GAUSSIAN_VAR",
    "MODEL_NAME_HISTORICAL_ES",
    "MODEL_NAME_HISTORICAL_VAR",
    "MODEL_VERSION",
    "SCALING_NONE",
    "SCALING_SQRT_TIME",
    "GaussianESModel",
    "GaussianVaRModel",
    "HistoricalESModel",
    "HistoricalVaRModel",
    "conditional_es",
    "conditional_var",
    "default_min_obs",
    "gaussian_es",
    "gaussian_var",
    "historical_es",
    "historical_var",
    "register_models",
    "sorted_losses",
    "tail_size",
]
