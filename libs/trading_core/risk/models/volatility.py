"""Volatility models — sample covariance, portfolio σ, EWMA, filtered P&L
(risk spec §5 Tier 1, §12/§14 conditional volatility; Phase B design
contract §2.4).

Pure, deterministic, stdlib-only (house rule): ``math``/``dataclasses``
only, ``math.fsum`` for every sum feeding a statistic, no numpy. Everything
here is SHADOW/RESEARCH — nothing alters a Tier 0 decision.

Estimators (each hand-checkable; the docstrings restate them):

- **Sample covariance** (``sample_covariance``): for aligned return columns
  ``x``, ``y`` of length ``n``, ``cov = Σ_t (x_t − x̄)(y_t − ȳ) / (n − 1)``
  (ddof=1). Each pair is computed ONCE for ``i ≤ j`` and mirrored, so the
  matrix is symmetric EXACTLY (bit-for-bit), not merely within 1e-12.
- **Portfolio volatility** (``portfolio_volatility``): sample standard
  deviation (ddof=1) of the book P&L series, USD/day; the diagnostic
  ``annualized_usd = σ·√annualization_days`` (default 252).
- **EWMA variance** (``ewma_variance``; RiskMetrics form used by spec §12/§14
  as the pre-GARCH conditional-volatility baseline):
  ``σ²_t = λ·σ²_{t−1} + (1 − λ)·r²_{t−1}``. The recursion is ZERO-MEAN
  (uses ``r²``, not ``(r − r̄)²``), and its seed is the matching zero-mean
  sample second moment of the first ``init_obs`` returns,
  ``σ²_{init_obs} = Σ_{t<init_obs} r²_t / init_obs``. Entries before
  ``init_obs`` are ``None`` (warm-up). The forecast at index ``t`` uses ONLY
  ``returns[< t]`` — walk-forward safe by construction (contract §3.5); a
  spike at ``t`` cannot change ``σ²_t``. Constant ``|r|`` ⇒ constant σ
  (contract §3.9), which the zero-mean seed guarantees exactly.
- **EWMA volatility forecast** (``ewma_volatility_forecast``): one more
  step of the same recursion — ``σ_{next} = √(λ·σ²_{n−1} + (1 − λ)·r²_{n−1})``
  — in the units of the input (returns or USD).
- **Volatility-scaled P&L** (``volatility_scaled_pnl``; Hull–White 1998
  filtered historical simulation, spec §12 "conditional VaR / ES"):
  ``pnl*_t = pnl_t × σ_now / σ_t`` where ``σ_t`` is the EWMA forecast for
  day ``t`` (information ``< t``) and ``σ_now`` is the forecast for the NEXT
  period. Warm-up entries (``σ_t`` is ``None``) and any leading entries with
  ``σ_t = 0`` (no volatility information yet — only possible while every
  earlier return is exactly 0) are DROPPED, never imputed; the result is a
  contiguous tail of the input, deterministic, of length
  ``n − dropped``. Conditional VaR/ES (``var_es.conditional_var`` /
  ``conditional_es``) are the plain historical estimators over this series.

Health (contract §1 honest nulls): ``n < min_obs`` ⇒ ``UNAVAILABLE`` with
the real numbers in ``reason``; never a fabricated 0. Malformed input
(non-finite values, ``lam ∉ (0, 1)``, ``init_obs < 1``, ``min_obs < 2``)
raises ``ValueError``.

Out of scope for Phase B (contract §2.4 / §5): shrinkage or robust
covariance, GARCH.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMeta,
    ModelResult,
    ModelTier,
    active,
    unavailable,
)
from libs.trading_core.risk.returns import ReturnMatrix

#: Estimator version (contract §4): arithmetic change ⇒ MAJOR bump.
MODEL_VERSION = "1.0.0"

#: Registry / ``ModelMeta.model_name`` labels of this module's estimators.
MODEL_NAME_COVARIANCE = "sample_covariance"
MODEL_NAME_PORTFOLIO_VOL = "portfolio_volatility"
MODEL_NAME_EWMA_VOL = "ewma_volatility"

#: Documented defaults (contract §2.4). Every one is a keyword parameter.
DEFAULT_MIN_OBS = 60            # covariance / portfolio σ minimum sample
DEFAULT_LAMBDA = 0.94           # RiskMetrics daily decay
DEFAULT_INIT_OBS = 20           # EWMA warm-up length (seed window)
DEFAULT_ANNUALIZATION_DAYS = 252
DEFAULT_FREQUENCY = "1D"


# ---------------------------------------------------------------------------
# Shared guards
# ---------------------------------------------------------------------------


def _check_finite(values: Sequence[float], *, what: str) -> None:
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}[{i}] must be a finite number, got {v!r}")


def _check_min_obs(min_obs: int) -> None:
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError(f"min_obs must be an int >= 2 (ddof=1 needs two observations), got {min_obs!r}")


def _check_ewma_params(lam: float, init_obs: int) -> None:
    if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam!r}")
    if isinstance(init_obs, bool) or not isinstance(init_obs, int) or init_obs < 1:
        raise ValueError(f"init_obs must be an int >= 1, got {init_obs!r}")


def _sample_mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, variance)`` with ddof=1, two-pass, ``math.fsum``; ``len >= 2``."""
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) * (v - mean) for v in values) / (n - 1)
    return mean, var


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovarianceResult:
    """Sample covariance of a :class:`ReturnMatrix` (contract §2.4).

    - ``tickers``: column order (identical to the input matrix);
    - ``matrix``: ``matrix[i][j] = cov(tickers[i], tickers[j])`` (ddof=1),
      symmetric exactly; EMPTY (``()``) when ``health`` is UNAVAILABLE;
    - ``n``: observations used (``sample_size``);
    - ``health`` / ``reason``: ``ModelResult``-like — UNAVAILABLE carries a
      reason with the real numbers, ACTIVE carries ``None``;
    - ``meta``: provenance (``ModelMeta``, name ``"sample_covariance"``);
    - ``diagnostics``: small scalars (``n``, ``n_tickers``, ``ddof``).
    """

    tickers: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    n: int
    health: ModelHealth
    reason: str | None
    meta: ModelMeta
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if health is ModelHealth.UNAVAILABLE and self.matrix:
            raise ValueError("UNAVAILABLE covariance must carry an empty matrix")
        if self.matrix:
            width = len(self.tickers)
            if len(self.matrix) != width or any(len(r) != width for r in self.matrix):
                raise ValueError(
                    f"matrix must be {width}x{width} to match tickers, got "
                    f"{len(self.matrix)} rows of {[len(r) for r in self.matrix]}"
                )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def is_available(self) -> bool:
        return bool(self.matrix) or (self.health is ModelHealth.ACTIVE and not self.tickers)

    @property
    def sample_size(self) -> int:
        return self.n

    def entry(self, a: str, b: str) -> float:
        """``cov(a, b)``; ``KeyError`` on an unknown ticker or when UNAVAILABLE."""
        if not self.matrix:
            raise KeyError(f"covariance unavailable: {self.reason}")
        try:
            i = self.tickers.index(a)
            j = self.tickers.index(b)
        except ValueError:
            raise KeyError(f"unknown ticker in ({a!r}, {b!r}); have {self.tickers}") from None
        return self.matrix[i][j]

    def variance(self, ticker: str) -> float:
        """Diagonal entry ``cov(ticker, ticker)``."""
        return self.entry(ticker, ticker)


def sample_covariance(
    matrix: ReturnMatrix,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> CovarianceResult:
    """Sample covariance matrix of the aligned return columns (contract §2.4).

    Estimator: ``cov_ij = Σ_t (r_{t,i} − r̄_i)(r_{t,j} − r̄_j) / (n − 1)``
    (ddof=1, two-pass means, ``math.fsum``); computed once per ``i ≤ j``
    and mirrored, so ``matrix[i][j] is matrix[j][i]`` bit-for-bit.

    Health: ``n = matrix.n_obs < min_obs`` ⇒ UNAVAILABLE (empty matrix,
    ``reason="n=… < min_obs=…"``); otherwise ACTIVE. No shrinkage (out of
    scope). ``min_obs < 2`` or a non-finite cell is malformed (``ValueError``).
    """
    _check_min_obs(min_obs)
    n = matrix.n_obs
    tickers = tuple(matrix.tickers)
    meta = ModelMeta(
        model_name=MODEL_NAME_COVARIANCE,
        model_version=MODEL_VERSION,
        params={"min_obs": min_obs, "ddof": 1},
        return_type=matrix.return_type,
        frequency=matrix.frequency,
        lookback=n if n >= 1 else None,
        data_source=matrix.source,
        as_of=matrix.as_of,
        # §5: an unconditional sample covariance over a fixed window.
        tier=ModelTier.TIER_1,
    )
    diagnostics: dict[str, Any] = {"n": n, "n_tickers": len(tickers), "ddof": 1}
    if n < min_obs:
        return CovarianceResult(
            tickers=tickers,
            matrix=(),
            n=n,
            health=ModelHealth.UNAVAILABLE,
            reason=f"n={n} < min_obs={min_obs}",
            meta=meta,
            diagnostics=diagnostics,
        )
    columns = [matrix.column(t) for t in tickers]
    for t, col in zip(tickers, columns):
        _check_finite(col, what=f"returns[{t}]")
    means = [math.fsum(col) / n for col in columns]
    devs = [[v - m for v in col] for col, m in zip(columns, means)]
    width = len(tickers)
    cells: list[list[float]] = [[0.0] * width for _ in range(width)]
    for i in range(width):
        for j in range(i, width):
            c = math.fsum(x * y for x, y in zip(devs[i], devs[j])) / (n - 1)
            cells[i][j] = c
            cells[j][i] = c
    return CovarianceResult(
        tickers=tickers,
        matrix=tuple(tuple(row) for row in cells),
        n=n,
        health=ModelHealth.ACTIVE,
        reason=None,
        meta=meta,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Portfolio volatility
# ---------------------------------------------------------------------------


def portfolio_volatility(
    pnl: Sequence[float],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Sample standard deviation of the book P&L, USD/day (contract §2.4).

    Estimator: ``μ = Σ pnl / n``; ``σ = √(Σ (pnl_t − μ)² / (n − 1))``
    (ddof=1, ``math.fsum``). Diagnostics: ``n``, ``mu``, ``variance``,
    ``annualized_usd = σ·√annualization_days``, ``annualization_days``.

    Health: ``n < min_obs`` ⇒ UNAVAILABLE (``value=None``, reason with the
    numbers). ``σ = 0`` (constant P&L) is reported as ``0.0`` — it is a real
    number, not a gap. Non-finite P&L is malformed (``ValueError``).
    """
    _check_min_obs(min_obs)
    if isinstance(annualization_days, bool) or not isinstance(annualization_days, int) or annualization_days < 1:
        raise ValueError(f"annualization_days must be an int >= 1, got {annualization_days!r}")
    _check_finite(pnl, what="pnl")
    n = len(pnl)
    meta = ModelMeta(
        model_name=MODEL_NAME_PORTFOLIO_VOL,
        model_version=MODEL_VERSION,
        params={
            "min_obs": min_obs,
            "ddof": 1,
            "annualization_days": annualization_days,
        },
        frequency=DEFAULT_FREQUENCY,
        lookback=n if n >= 1 else None,
        as_of=as_of,
        horizon_days=1,
        # §5: unconditional sample sigma of the book P&L.
        tier=ModelTier.TIER_1,
    )
    if n < min_obs:
        return unavailable(
            meta,
            f"n={n} < min_obs={min_obs}",
            n,
            diagnostics={"n": n},
        )
    mu, var = _sample_mean_and_variance(pnl)
    sigma = math.sqrt(var)
    return active(
        meta,
        sigma,
        n,
        diagnostics={
            "n": n,
            "mu": mu,
            "variance": var,
            "annualized_usd": sigma * math.sqrt(annualization_days),
            "annualization_days": annualization_days,
        },
    )


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


def _ewma_variance_path(
    returns: Sequence[float], lam: float, init_obs: int
) -> list[float | None]:
    """EWMA variance path of length ``n + 1``.

    ``path[t]`` for ``t < n`` is the forecast for observation ``t`` (uses
    ``returns[< t]`` only); ``path[n]`` is the forecast for the NEXT period.
    Entries ``t < init_obs`` are ``None``; ``path[init_obs]`` is the
    zero-mean seed ``Σ_{t<init_obs} r²_t / init_obs``; afterwards
    ``path[t+1] = λ·path[t] + (1 − λ)·r²_t``. Empty (all ``None`` up to
    index ``n``) when ``n < init_obs`` — the seed window is incomplete.
    """
    n = len(returns)
    path: list[float | None] = [None] * (n + 1)
    if n < init_obs:
        return path
    var = math.fsum(r * r for r in returns[:init_obs]) / init_obs
    path[init_obs] = var
    one_minus = 1.0 - lam
    for t in range(init_obs, n):
        r = returns[t]
        var = lam * var + one_minus * (r * r)
        path[t + 1] = var
    return path


def ewma_variance(
    returns: Sequence[float],
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
) -> list[float | None]:
    """Walk-forward EWMA variance forecasts, one per input index (contract §2.4).

    ``out[t]`` is the variance forecast for period ``t`` computed from
    ``returns[< t]`` ONLY:

        out[t] = None                                   for t < init_obs
        out[init_obs] = Σ_{s < init_obs} r_s² / init_obs   (zero-mean seed)
        out[t + 1] = λ·out[t] + (1 − λ)·r_t²            for t ≥ init_obs

    Length equals ``len(returns)``; if ``len(returns) <= init_obs`` every
    entry is ``None`` (no forecast has a complete seed window). Constant
    ``|r|`` gives a constant path (seed = r², recursion fixed point).
    ``lam ∉ (0, 1)``, ``init_obs < 1`` or a non-finite return ⇒ ``ValueError``.

    Hand-check (``init_obs=2``, ``λ=0.94``, ``r=[0.01, −0.02, 0.03, 0.01]``):
    ``out = [None, None, (0.0001+0.0004)/2 = 0.00025,
    0.94·0.00025 + 0.06·0.0009 = 0.000289]``.
    """
    _check_ewma_params(lam, init_obs)
    _check_finite(returns, what="returns")
    return _ewma_variance_path(returns, lam, init_obs)[: len(returns)]


def ewma_volatility_forecast(
    returns: Sequence[float],
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """EWMA volatility forecast for the NEXT period (contract §2.4).

    ``σ_next = √(λ·σ²_{n−1} + (1 − λ)·r²_{n−1})`` — the element after the
    last of :func:`ewma_variance` — in the units of the input (return units
    for returns, USD/day for a P&L series). Diagnostics: ``n``, ``lambda``,
    ``init_obs``, ``variance`` (σ²_next), ``half_life_days = ln 0.5 / ln λ``.

    Health: ``n < init_obs`` ⇒ UNAVAILABLE (``"n=… < init_obs=…"``);
    otherwise ACTIVE (a seed window of exactly ``init_obs`` returns already
    yields a forecast — the caller decides how much history it trusts via
    ``init_obs``). Malformed parameters ⇒ ``ValueError``.
    """
    _check_ewma_params(lam, init_obs)
    _check_finite(returns, what="returns")
    n = len(returns)
    meta = ModelMeta(
        model_name=MODEL_NAME_EWMA_VOL,
        model_version=MODEL_VERSION,
        params={"lambda": lam, "init_obs": init_obs},
        frequency=DEFAULT_FREQUENCY,
        lookback=n if n >= 1 else None,
        as_of=as_of,
        horizon_days=1,
        distribution=None,
        # §5: EWMA is a CONDITIONAL volatility model — Tier 2, beside GARCH.
        tier=ModelTier.TIER_2,
    )
    if n < init_obs:
        return unavailable(
            meta,
            f"n={n} < init_obs={init_obs}",
            n,
            diagnostics={"n": n, "lambda": lam, "init_obs": init_obs},
        )
    var_next = _ewma_variance_path(returns, lam, init_obs)[n]
    assert var_next is not None  # n >= init_obs guarantees the seed exists
    return active(
        meta,
        math.sqrt(var_next),
        n,
        diagnostics={
            "n": n,
            "lambda": lam,
            "init_obs": init_obs,
            "variance": var_next,
            "half_life_days": math.log(0.5) / math.log(lam),
        },
    )


# ---------------------------------------------------------------------------
# Filtered historical simulation (Hull–White volatility scaling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityScaling:
    """Result of :func:`volatility_scaling` (the detail behind
    :func:`volatility_scaled_pnl`).

    - ``scaled``: ``pnl_t × σ_now / σ_t`` for every kept ``t`` (in order);
    - ``sigma_now``: the EWMA forecast for the next period (``None`` when
      the series is shorter than ``init_obs``);
    - ``n_input`` / ``n_used`` / ``dropped``: bookkeeping (``dropped =
      n_input − n_used`` = warm-up plus any leading zero-σ entries);
    - ``lam`` / ``init_obs``: the parameters used.
    """

    scaled: tuple[float, ...]
    sigma_now: float | None
    n_input: int
    n_used: int
    dropped: int
    lam: float
    init_obs: int


def volatility_scaling(
    pnl: Sequence[float],
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
) -> VolatilityScaling:
    """Hull–White filtered-HS scaling with bookkeeping (contract §2.4).

    ``σ_t = √ewma_variance(pnl)[t]`` (forecast for ``t`` from ``pnl[< t]``),
    ``σ_now = √(λ·σ²_{n−1} + (1 − λ)·pnl²_{n−1})`` (forecast for the next
    period). Kept entries: every ``t ≥ init_obs`` with ``σ_t > 0``; because
    the recursion is monotone-positive once any ``r_t ≠ 0`` has entered,
    the ``σ_t = 0`` entries (if any) form a leading run right after the
    warm-up, so the kept set is a contiguous tail — DROPPED, never imputed.
    ``scaled_t = pnl_t × σ_now / σ_t``. When ``n <= init_obs`` nothing is
    kept (``scaled = ()``); when every input is 0, ``σ_now = 0`` and again
    nothing is kept.
    """
    _check_ewma_params(lam, init_obs)
    _check_finite(pnl, what="pnl")
    n = len(pnl)
    path = _ewma_variance_path(pnl, lam, init_obs)
    var_now = path[n]
    if var_now is None:
        return VolatilityScaling(
            scaled=(), sigma_now=None, n_input=n, n_used=0, dropped=n,
            lam=lam, init_obs=init_obs,
        )
    sigma_now = math.sqrt(var_now)
    scaled: list[float] = []
    for t in range(init_obs, n):
        var_t = path[t]
        assert var_t is not None
        if var_t <= 0.0:
            continue
        scaled.append(pnl[t] * sigma_now / math.sqrt(var_t))
    return VolatilityScaling(
        scaled=tuple(scaled),
        sigma_now=sigma_now,
        n_input=n,
        n_used=len(scaled),
        dropped=n - len(scaled),
        lam=lam,
        init_obs=init_obs,
    )


def volatility_scaled_pnl(
    pnl: Sequence[float],
    *,
    lam: float = DEFAULT_LAMBDA,
    init_obs: int = DEFAULT_INIT_OBS,
) -> list[float]:
    """Filtered-HS P&L ``pnl_t × σ_now / σ_t`` (contract §2.4; Hull–White).

    Thin list view of :func:`volatility_scaling` — see it for the exact
    rule (warm-up and leading zero-σ entries dropped; contiguous tail; the
    forecast for ``t`` never uses ``pnl[t]``). Consumed by
    ``var_es.conditional_var`` / ``conditional_es``. Limit sanity (contract
    §3.9): as ``λ → 1`` the σ path is flat, ratios → 1 and the output tends
    to ``pnl[init_obs:]``; with constant ``|pnl|`` the ratios are exactly 1.
    """
    return list(volatility_scaling(pnl, lam=lam, init_obs=init_obs).scaled)


__all__ = [
    "DEFAULT_ANNUALIZATION_DAYS",
    "DEFAULT_INIT_OBS",
    "DEFAULT_LAMBDA",
    "DEFAULT_MIN_OBS",
    "MODEL_NAME_COVARIANCE",
    "MODEL_NAME_EWMA_VOL",
    "MODEL_NAME_PORTFOLIO_VOL",
    "MODEL_VERSION",
    "CovarianceResult",
    "VolatilityScaling",
    "ewma_variance",
    "ewma_volatility_forecast",
    "portfolio_volatility",
    "sample_covariance",
    "volatility_scaled_pnl",
    "volatility_scaling",
]
