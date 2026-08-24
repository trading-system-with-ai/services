"""Single-factor (SPY) risk diagnostic — spec §11; compliance §3 row 11.

Pure stdlib, deterministic, no I/O. **RESEARCH** (spec §70) — one step below
the SHADOW default the rest of this package uses. Nothing here gates: there
is no ``QuantityCap``, no limit object, no registered model, and no call
path into ``assess()``. It answers one question for display only:

    *How much of this book's P&L variation is just the market?*

The compliance audit (§3 row 11) records the promise this module closes: the
audit committed to a "SPY-β single-factor diagnostic" as a zero-cost
addition to the §11 concentration work, and it was never built. Note what it
does NOT do — §11's ``max_factor_...`` concentration cap remains
REJECT-documented, because a cap needs a validated factor taxonomy and this
is one regression against one proxy series.

Estimators, both hand-checkable
-------------------------------

**Beta (OLS slope through cov/var, ``ddof=1``)**::

    beta = cov(r_asset, r_factor) / var(r_factor)
    cov  = Σ_t (a_t − ā)(f_t − f̄) / (n − 1)
    var  = Σ_t (f_t − f̄)²        / (n − 1)

The ``n − 1`` divisors cancel in the ratio, so ``beta`` is identical to the
population form; ``ddof=1`` is stated because ``r2`` below reports sample
variances the caller may want to reconcile, and the platform reports sample
(ddof=1) variances everywhere else (``portfolio_volatility``, Gaussian VaR).

**R²** is the squared Pearson correlation of the two series, which for a
univariate OLS fit with an intercept is exactly the fraction of the asset's
variance the factor explains::

    r2 = cov² / (var_factor × var_asset)     ∈ [0, 1]

**Factor-explained variance share** (the portfolio number)::

    beta_p = cov(pnl_total, f) / var(f)          — regress the BOOK on the factor
    share  = var(beta_p × f_t) / var(pnl_total)
           = beta_p² × var(f) / var(pnl_total)

which equals the portfolio regression's R² identically — the two routes are
written as one so the reported ``share`` and ``r2`` can never disagree. It
is the fraction of book P&L variance attributable to the factor; ``1 −
share`` is idiosyncratic. Because ``share`` is a variance RATIO it is in
``[0, 1]`` by construction, and is clamped there only to absorb float error.

Honest nulls (never a fabricated 0.0, never an exception for missing data)
-------------------------------------------------------------------------

- ``n < params.min_obs`` (60) ⇒ ``health=UNAVAILABLE``, every statistic
  ``None``, a ``reason`` carrying the real numbers;
- ``min_obs ≤ n < degraded_multiple × min_obs`` ⇒ ``health=DEGRADED`` with
  the numbers stated — the estimate is reported, and labelled thin;
- a CONSTANT factor series (``var(f) ≤ zero_variance``) ⇒ ``UNAVAILABLE``:
  a factor that never moved explains nothing, and the division would
  otherwise be the number reported;
- a constant ASSET/book series ⇒ ``beta`` is a defined 0.0 but ``r2`` /
  ``share`` are ``None`` (``0/0``), stated in the reason.

**Malformed input raises** (``ValueError``), missing data never does: a
length mismatch between the asset and factor series is a caller bug, not a
data gap, and so is a non-finite value.

Every threshold is a parameter on :class:`FactorParams` (house rule: never a
hardcoded truth). **RESEARCH DEFAULTS — UNVALIDATED**: ``min_obs=60`` is the
platform's standing minimum for a shape/fit call (it matches
``DistributionParams.min_obs``), not a validated choice for beta stability.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .base import ModelHealth, ModelMeta

#: Model name / version recorded in ``ModelMeta`` (contract §4).
MODEL_VERSION = "1.0.0"
MODEL_NAME = "single_factor_beta"

#: The factor this diagnostic is written for (spec §11). It is a LABEL, not a
#: fetch: the caller supplies the return series and names it here, so a book
#: measured against a different proxy is never silently read as SPY.
DEFAULT_FACTOR = "SPY"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorParams:
    """Every threshold of this module. **RESEARCH DEFAULTS — UNVALIDATED**.

    - ``min_obs`` (60): fewer paired observations than this and no beta is
      reported — ``UNAVAILABLE`` with the real numbers in ``reason``;
    - ``degraded_multiple`` (2.0): ``n < degraded_multiple × min_obs`` ⇒
      ``DEGRADED`` — the number is reported AND labelled thin;
    - ``zero_variance`` (1e-18): a series whose sample variance is at or
      below this counts as constant.
    """

    min_obs: int = 60
    degraded_multiple: float = 2.0
    zero_variance: float = 1e-18

    def __post_init__(self) -> None:
        if isinstance(self.min_obs, bool) or not isinstance(self.min_obs, int) or self.min_obs < 3:
            raise ValueError(f"min_obs must be an int >= 3, got {self.min_obs!r}")
        if not (self.degraded_multiple >= 1.0):
            raise ValueError(
                f"degraded_multiple must be >= 1, got {self.degraded_multiple}"
            )
        if not (self.zero_variance >= 0.0):
            raise ValueError(f"zero_variance must be >= 0, got {self.zero_variance}")


DEFAULT_FACTOR_PARAMS = FactorParams()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionBeta:
    """One position's exposure to the factor (spec §11).

    ``beta``/``r2`` are ``None`` exactly when the regression was not
    computable for this position; ``reason`` then says why in real numbers.
    """

    label: str
    beta: float | None
    r2: float | None
    n: int
    health: ModelHealth
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "health", ModelHealth(self.health))
        if self.health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={self.health} requires a non-empty reason")
        for name in ("beta", "r2"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be finite or None, got {v!r}")


@dataclass(frozen=True)
class FactorRiskResult:
    """How much of the BOOK is the factor (spec §11) — display only.

    - ``portfolio_beta``: ``beta_p`` from regressing total book P&L on the
      factor — the book's dollar sensitivity to a 1-unit factor move;
    - ``explained_variance_share``: ``var(beta_p × f) / var(pnl_total)`` ∈
      ``[0, 1]`` — the fraction of book P&L variance the factor explains;
    - ``idiosyncratic_share``: ``1 − explained_variance_share`` (``None``
      when the share is);
    - ``positions``: per-position :class:`PositionBeta`, in the caller's
      mapping order;
    - ``factor``: the label of the series actually used (never assumed);
    - ``health``/``reason``: honest, with the real numbers.
    """

    portfolio_beta: float | None
    explained_variance_share: float | None
    idiosyncratic_share: float | None
    positions: tuple[PositionBeta, ...]
    factor: str
    n: int
    health: ModelHealth
    reason: str | None
    meta: ModelMeta

    def __post_init__(self) -> None:
        object.__setattr__(self, "health", ModelHealth(self.health))
        object.__setattr__(self, "positions", tuple(self.positions))
        if self.health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={self.health} requires a non-empty reason")
        for name in (
            "portfolio_beta",
            "explained_variance_share",
            "idiosyncratic_share",
        ):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be finite or None, got {v!r}")

    @property
    def is_available(self) -> bool:
        return self.explained_variance_share is not None


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _check_finite(values: Sequence[float], *, what: str) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}: every value must be a finite number, got {v!r}")
        out.append(float(v))
    return out


def _moments(a: Sequence[float], f: Sequence[float]) -> tuple[float, float, float]:
    """``(cov, var_f, var_a)``, all sample (``ddof=1``) — the whole regression.

    The ``n − 1`` divisor is applied to each, so ``cov/var_f`` (beta) is
    divisor-free while the variances remain the sample variances the rest of
    the platform reports.
    """
    n = len(f)
    mean_a = math.fsum(a) / n
    mean_f = math.fsum(f) / n
    da = [x - mean_a for x in a]
    df = [y - mean_f for y in f]
    denom = n - 1
    cov = math.fsum(x * y for x, y in zip(da, df)) / denom
    var_f = math.fsum(y * y for y in df) / denom
    var_a = math.fsum(x * x for x in da) / denom
    return cov, var_f, var_a


def beta_vs_factor(
    asset_returns: Sequence[float],
    factor_returns: Sequence[float],
    *,
    params: FactorParams = DEFAULT_FACTOR_PARAMS,
) -> dict[str, Any]:
    """``{"beta", "r2", "n", "health", "reason"}`` — OLS of asset on factor.

    ``beta = cov/var_f`` and ``r2 = cov²/(var_f · var_a)`` with sample
    (``ddof=1``) moments, per the module docstring. The two series are
    positionally paired: element ``t`` of each is the SAME period, which is
    the caller's responsibility to arrange (this function has no dates to
    align on and will not guess).

    Raises ``ValueError`` on a length mismatch or a non-finite value —
    malformed input. Returns honest nulls for missing data: ``n <
    min_obs`` ⇒ ``UNAVAILABLE``; a constant factor ⇒ ``UNAVAILABLE`` (it
    explains nothing and the division is undefined); a constant asset ⇒
    ``beta = 0.0`` with ``r2 = None`` (``0/0``).
    """
    if len(asset_returns) != len(factor_returns):
        raise ValueError(
            f"asset_returns and factor_returns must be the same length, got "
            f"{len(asset_returns)} and {len(factor_returns)}"
        )
    a = _check_finite(asset_returns, what="asset_returns")
    f = _check_finite(factor_returns, what="factor_returns")
    n = len(a)

    def _null(reason: str, health: ModelHealth = ModelHealth.UNAVAILABLE) -> dict[str, Any]:
        return {"beta": None, "r2": None, "n": n, "health": health, "reason": reason}

    if n < params.min_obs:
        return _null(f"n={n} < min_obs={params.min_obs}")

    cov, var_f, var_a = _moments(a, f)
    if var_f <= params.zero_variance:
        return _null(
            f"factor series is constant (sample variance {var_f:.3e} <= "
            f"{params.zero_variance:.0e}); a factor that never moved explains nothing"
        )

    health = ModelHealth.ACTIVE
    reason: str | None = None
    if n < params.degraded_multiple * params.min_obs:
        health = ModelHealth.DEGRADED
        reason = (
            f"n={n} < {params.degraded_multiple:g} x min_obs={params.min_obs}: "
            f"beta is reported but thinly estimated"
        )

    beta = cov / var_f
    if var_a <= params.zero_variance:
        # A flat asset genuinely has zero sensitivity; R² is 0/0, not 0.
        return {
            "beta": 0.0,
            "r2": None,
            "n": n,
            "health": ModelHealth.DEGRADED if health is ModelHealth.ACTIVE else health,
            "reason": (
                f"asset series is constant (sample variance {var_a:.3e}); beta is "
                f"0.0 by construction and r2 is undefined (0/0)"
            ),
        }
    r2 = cov * cov / (var_f * var_a)
    # A variance ratio is in [0, 1]; clamp float error only, never round.
    r2 = max(0.0, min(1.0, r2))
    return {"beta": beta, "r2": r2, "n": n, "health": health, "reason": reason}


def factor_risk_share(
    positions_pnl: Mapping[str, Sequence[float]],
    factor_returns: Sequence[float],
    *,
    params: FactorParams = DEFAULT_FACTOR_PARAMS,
    factor: str = DEFAULT_FACTOR,
    as_of: date | None = None,
) -> FactorRiskResult:
    """Per-position betas plus the book's factor-explained variance share.

    ``positions_pnl`` maps a position label to its per-period P&L series
    (USD, gain-positive), every series the SAME length as
    ``factor_returns`` and positionally aligned to it. ``pnl_total`` is the
    per-period SUM across positions — the book's P&L — and the portfolio
    numbers come from regressing that total on the factor::

        beta_p = cov(pnl_total, f) / var(f)
        share  = beta_p² × var(f) / var(pnl_total)      ∈ [0, 1]

    ``share`` is identically the portfolio regression's R², and is computed
    once from :func:`beta_vs_factor` so the two can never disagree.

    Raises ``ValueError`` on malformed input: an empty mapping, a length
    mismatch anywhere, or a non-finite value. Missing data degrades
    honestly instead — too few observations or a constant factor gives
    ``UNAVAILABLE`` with every statistic ``None`` and the real numbers in
    ``reason``, and a per-position regression that fails leaves THAT
    position's ``beta`` ``None`` without taking the book's number down.

    RESEARCH (spec §70): a display diagnostic. No cap, no gate, no registry.
    """
    if not positions_pnl:
        raise ValueError("positions_pnl must contain at least one position")
    f = _check_finite(factor_returns, what="factor_returns")
    n = len(f)
    for label, series in positions_pnl.items():
        if len(series) != n:
            raise ValueError(
                f"position {label!r} has {len(series)} observations but "
                f"factor_returns has {n} (series must be positionally aligned)"
            )

    columns = {
        label: _check_finite(series, what=f"positions_pnl[{label!r}]")
        for label, series in positions_pnl.items()
    }
    meta = ModelMeta(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        params={
            "min_obs": params.min_obs,
            "degraded_multiple": params.degraded_multiple,
            "zero_variance": params.zero_variance,
            "factor": factor,
        },
        lookback=n or None,
        as_of=as_of,
    )

    positions = tuple(
        PositionBeta(
            label=label,
            beta=(res := beta_vs_factor(series, f, params=params))["beta"],
            r2=res["r2"],
            n=res["n"],
            health=res["health"],
            reason=res["reason"],
        )
        for label, series in columns.items()
    )

    # The book is the per-period SUM of its positions.
    pnl_total = [math.fsum(col[t] for col in columns.values()) for t in range(n)]
    book = beta_vs_factor(pnl_total, f, params=params)
    share = book["r2"]
    return FactorRiskResult(
        portfolio_beta=book["beta"],
        explained_variance_share=share,
        idiosyncratic_share=None if share is None else 1.0 - share,
        positions=positions,
        factor=factor,
        n=n,
        health=book["health"],
        reason=book["reason"],
        meta=meta,
    )
