"""Risk contribution — volatility (Euler/covariance) and ES (Euler tail
average), plus marginal & incremental ES (risk spec §9, §10, §33; Phase B
design contract §2.5).

Pure stdlib, deterministic, no I/O. SHADOW/RESEARCH: nothing here alters a
Tier 0 decision.

Inputs (contract §1, §2.5): per-position P&L series in USD per horizon,
gain-positive, on the SAME dates (``positions_pnl[key][t]``). The
portfolio series is DERIVED as ``pnl_p[t] = math.fsum(pnl_i[t] over i)``;
a caller-supplied ``portfolio_pnl`` (e.g. ``BookPnl.total``) is only
CHECKED against it — ``|pnl_p[t] − portfolio_pnl[t]| ≤ sum_tolerance ×
scale`` for every ``t`` with ``scale = max(1, max_t |pnl_p[t]|,
max_t |portfolio_pnl[t]|)`` — and a mismatch is malformed input
(``ValueError``), because contributions of a series that is not the sum of
its parts cannot add up. Contributions are reported as LOSSES (positive =
risk added), like VaR/ES.

Estimators (every number hand-checkable):

- **Volatility contribution** (``method="VOL"``):
  ``RC^σ_i = cov(pnl_i, pnl_p) / σ_p`` with the sample covariance
  ``cov = Σ_t (pnl_i,t − mean_i)(pnl_p,t − mean_p) / (n − 1)`` and the
  sample stdev ``σ_p = sqrt(Σ_t (pnl_p,t − mean_p)² / (n − 1))`` (ddof=1,
  ``math.fsum``). ``total = σ_p``. Because ``Σ_i cov(pnl_i, pnl_p) =
  var(pnl_p)``, ``Σ_i RC^σ_i = σ_p`` up to fsum rounding (contract §3.3).
- **ES contribution (Euler)** (``method="ES"``) at confidence ``α``: with
  losses ``L_t = −pnl_p,t`` and the ONE platform tail convention (contract
  §2.3) — ``k = ceil(n·(1 − α))``, tail set ``T`` = the dates of the ``k``
  largest portfolio losses, ties broken by date order (earlier first,
  stable) — ``RC^ES_i = mean_{t∈T}(−pnl_i,t)`` and ``total = ES_α =
  mean_{t∈T}(L_t)``. Since ``Σ_i (−pnl_i,t) = L_t`` on every tail date,
  ``Σ_i RC^ES_i = ES_α`` exactly (up to fsum rounding). This module keeps
  its own private tail helper (``_tail_indices``) that MUST match
  ``var_es.py``; the integrator pins the equality with a cross-module test.
- **Marginal ES** of a candidate held at quantity ``q`` (``q ≠ 0``):
  join ``pnl_new = pnl_book + pnl_cand`` (the book WITHOUT the candidate
  plus the candidate's P&L at quantity ``q``), take the tail of
  ``pnl_new``, ``RC^ES_cand = mean_{t∈T}(−pnl_cand,t)``, and report
  ``RC^ES_cand / q`` per unit — "how much ES one more unit adds at the
  margin" (spec §9). Returned as a ``ModelResult`` (``value`` = per-unit
  marginal ES; ``diagnostics`` carry ``contribution``, ``quantity``,
  ``tail_size``).
- **Incremental ES**: ``ES_α(pnl_book + pnl_cand) − ES_α(pnl_book)``, both
  recomputed on the joined series with the same ``k`` (spec §9 "incremental
  risk framework"). Returned as ``IncrementalResult(before, after, delta)``.

Shares: ``share_i = contribution_i / total`` — ``None`` when ``total ≤ 0``
(a non-positive σ_p or ES makes "share of risk" meaningless; the raw
contributions are still reported).

Health (contract §1, §2.3, §2.5) — parameters on ``ContributionParams``:
``n < min_obs`` ⇒ ``UNAVAILABLE`` (``total=None``, empty ``per_position``,
reason with the real numbers, no exception); ``min_obs ≤ n <
degraded_multiple × min_obs`` ⇒ ``DEGRADED`` ("small tail: k=…" for ES,
"small sample" for VOL); otherwise ``ACTIVE``. ``min_obs`` defaults follow
the VaR/ES grid — 60 for α < 0.99, 250 for α ≥ 0.99 (so k ≥ 3 at 99%) —
and 60 for the volatility contribution; pass ``min_obs=`` to override.
``σ_p = 0`` (constant portfolio P&L) ⇒ ``UNAVAILABLE`` for VOL (0/0 is not
a contribution). An empty book ⇒ ``UNAVAILABLE`` ("no positions").
Malformed input — mismatched lengths, α outside (0.5, 1), non-finite
values, ``q = 0`` — raises ``ValueError``.

Every result carries a ``ModelMeta`` (``model_version="1.0.0"``; bump per
contract §4) so the number is reproducible (spec §44).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, NamedTuple

from .base import ModelHealth, ModelMeta, ModelResult, ModelTier, active, degraded, unavailable
from .var_es import tail_size  # noqa: F401  — the ONE tail-size convention (contract §2.3/§3.3)

#: Contribution method labels (contract §2.5).
METHOD_VOL = "VOL"
METHOD_ES = "ES"

#: Model names / version recorded in ``ModelMeta`` (contract §4).
MODEL_VERSION = "1.0.0"
MODEL_NAME_VOL = "vol_contribution"
MODEL_NAME_ES = "es_contribution"
MODEL_NAME_MARGINAL_ES = "marginal_es"
MODEL_NAME_INCREMENTAL_ES = "incremental_es"

#: Distribution label of the ES estimators (empirical tail average).
DISTRIBUTION_EMPIRICAL = "EMPIRICAL"


# ---------------------------------------------------------------------------
# Parameters & results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributionParams:
    """Every threshold of this module (house rule: never a magic number).

    - ``min_obs_vol`` (60): minimum observations for the volatility
      contribution;
    - ``min_obs_95`` (60) / ``min_obs_99`` (250): default ES ``min_obs``
      for ``α < 0.99`` / ``α ≥ 0.99`` (contract §2.3 grid; k ≥ 3 at 99%);
    - ``degraded_multiple`` (2.0): ``n < degraded_multiple × min_obs`` ⇒
      DEGRADED;
    - ``sum_tolerance`` (1e-6): relative tolerance of the portfolio-series
      check (contract §2.5 "assert within 1e-6·scale").
    """

    min_obs_vol: int = 60
    min_obs_95: int = 60
    min_obs_99: int = 250
    degraded_multiple: float = 2.0
    sum_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("min_obs_vol", "min_obs_95", "min_obs_99"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 2:
                raise ValueError(f"{name} must be an int >= 2, got {v!r}")
        if not (self.degraded_multiple >= 1.0):
            raise ValueError(
                f"degraded_multiple must be >= 1, got {self.degraded_multiple}"
            )
        if not (self.sum_tolerance >= 0.0):
            raise ValueError(f"sum_tolerance must be >= 0, got {self.sum_tolerance}")

    def default_min_obs(self, confidence: float) -> int:
        """ES ``min_obs`` for ``confidence``: ``min_obs_99`` if α ≥ 0.99 else ``min_obs_95``."""
        return self.min_obs_99 if confidence >= 0.99 else self.min_obs_95


DEFAULT_PARAMS = ContributionParams()


class PositionContribution(NamedTuple):
    """One row of ``ContributionResult.per_position`` — ``(key, contribution,
    share)`` (contract §2.5); attribute access and tuple unpacking both work."""

    key: str
    contribution: float
    share: float | None


@dataclass(frozen=True)
class ContributionResult:
    """Risk contribution of every position to one total (contract §2.5).

    - ``total``: ``σ_p`` (VOL) or ``ES_α`` (ES), USD per horizon, loss-
      positive; ``None`` when UNAVAILABLE;
    - ``per_position``: ``(key, contribution, share)`` rows in input order
      (empty when UNAVAILABLE); ``Σ contribution == total`` up to fsum
      rounding; ``share = contribution / total`` or ``None`` if
      ``total ≤ 0``;
    - ``method``: ``"VOL"`` | ``"ES"``; ``confidence`` / ``tail_size`` are
      ``None`` for VOL;
    - ``health`` / ``reason`` / ``sample_size`` / ``meta`` as in
      ``ModelResult`` (contract §2.2).
    """

    total: float | None
    per_position: tuple[PositionContribution, ...]
    method: str
    confidence: float | None
    tail_size: int | None
    health: ModelHealth
    reason: str | None
    sample_size: int
    meta: ModelMeta

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if self.method not in (METHOD_VOL, METHOD_ES):
            raise ValueError(f"method must be {METHOD_VOL!r} or {METHOD_ES!r}, got {self.method!r}")
        if health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED):
            if self.total is not None or self.per_position:
                raise ValueError(f"health={health} requires total=None and no per_position rows")
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if self.total is not None and not math.isfinite(self.total):
            raise ValueError(f"total must be finite, got {self.total!r}")
        object.__setattr__(self, "per_position", tuple(self.per_position))

    @property
    def is_available(self) -> bool:
        return self.total is not None

    def contribution_of(self, key: str) -> float | None:
        """Contribution of ``key`` (``None`` if absent / UNAVAILABLE)."""
        for row in self.per_position:
            if row.key == key:
                return row.contribution
        return None

    def share_of(self, key: str) -> float | None:
        for row in self.per_position:
            if row.key == key:
                return row.share
        return None


@dataclass(frozen=True)
class IncrementalResult:
    """Incremental ES of a candidate (contract §2.5): ``delta = after − before``
    with ``before = ES_α(book)`` and ``after = ES_α(book + candidate)``,
    both on the same ``n`` and ``k``. All three ``None`` when UNAVAILABLE."""

    before: float | None
    after: float | None
    delta: float | None
    confidence: float
    tail_size: int | None
    health: ModelHealth
    reason: str | None
    sample_size: int
    meta: ModelMeta

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED) and (
            self.before is not None or self.after is not None or self.delta is not None
        ):
            raise ValueError(f"health={health} requires before/after/delta = None")
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")

    @property
    def is_available(self) -> bool:
        return self.delta is not None


# ---------------------------------------------------------------------------
# Private helpers — tail convention (MUST equal var_es.py, contract §2.3)
# ---------------------------------------------------------------------------


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number in (0.5, 1), got {confidence!r}")
    if not (0.5 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    return float(confidence)


def _check_finite(values: Sequence[float], *, what: str) -> None:
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}: every value must be a finite number, got {v!r}")


# ``tail_size`` is IMPORTED from ``var_es`` (above) and re-exported here, not
# reimplemented. Contract §3.3 requires ``Σ_i RC^ES_i == ES_α`` EXACTLY, which
# holds only if this module and ``var_es`` select the SAME k dates. A second
# local implementation (``n − floor(n·α)``) agreed on the platform grid but
# diverged elsewhere (e.g. n=90, α=0.7 ⇒ 27 vs 28), silently breaking the
# invariant; there is now exactly one definition of the tail.


def _tail_indices(portfolio_pnl: Sequence[float], confidence: float) -> tuple[int, list[int]]:
    """``(k, indices)`` — the ``k`` dates of the ``k`` largest portfolio
    losses ``L_t = −pnl_t``, ordered by loss descending, ties by date order
    (earlier ``t`` first — stable, deterministic)."""
    n = len(portfolio_pnl)
    k = tail_size(n, confidence)
    if k == 0:
        return 0, []
    order = sorted(range(n), key=lambda t: (portfolio_pnl[t], t))  # smallest pnl = largest loss first
    return k, order[:k]


def _tail_mean_loss(pnl: Sequence[float], indices: Sequence[int]) -> float:
    """``mean_{t∈indices}(−pnl_t)`` with ``math.fsum``."""
    return math.fsum(-pnl[t] for t in indices) / len(indices)


def _historical_es(pnl: Sequence[float], confidence: float) -> tuple[float, int]:
    """``(ES_α, k)`` = mean of the ``k`` largest losses (contract §2.3)."""
    k, idx = _tail_indices(pnl, confidence)
    return _tail_mean_loss(pnl, idx), k


def _portfolio_series(
    positions_pnl: Mapping[str, Sequence[float]],
    portfolio_pnl: Sequence[float] | None,
    tolerance: float,
) -> tuple[list[str], list[list[float]], list[float]]:
    """Validate shapes and derive ``pnl_p[t] = fsum_i pnl_i[t]``.

    Returns ``(keys, columns, pnl_p)`` (keys in input order). ``ValueError``
    on ragged lengths, non-finite values, or a supplied ``portfolio_pnl``
    that differs from the derived sum by more than ``tolerance × scale``.
    """
    keys = [str(k) for k in positions_pnl]
    columns: list[list[float]] = []
    n: int | None = None
    for key in positions_pnl:
        col = [float(v) for v in positions_pnl[key]]
        _check_finite(col, what=f"positions_pnl[{key!r}]")
        if n is None:
            n = len(col)
        elif len(col) != n:
            raise ValueError(
                f"positions_pnl[{key!r}] has {len(col)} observations, expected {n} (all series must share dates)"
            )
        columns.append(col)
    if n is None:
        n = len(portfolio_pnl) if portfolio_pnl is not None else 0
    pnl_p = [math.fsum(col[t] for col in columns) for t in range(n)] if columns else [0.0] * n
    if portfolio_pnl is not None:
        given = [float(v) for v in portfolio_pnl]
        _check_finite(given, what="portfolio_pnl")
        if len(given) != n:
            raise ValueError(
                f"portfolio_pnl has {len(given)} observations, positions have {n}"
            )
        scale = max(1.0, max((abs(v) for v in pnl_p), default=0.0), max((abs(v) for v in given), default=0.0))
        for t, (a, b) in enumerate(zip(pnl_p, given)):
            if abs(a - b) > tolerance * scale:
                raise ValueError(
                    f"portfolio_pnl[{t}]={b!r} != sum of positions {a!r} "
                    f"(|diff|={abs(a - b)!r} > {tolerance}×scale={tolerance * scale!r})"
                )
    return keys, columns, pnl_p


def _rows(keys: Sequence[str], contributions: Sequence[float], total: float) -> tuple[PositionContribution, ...]:
    return tuple(
        PositionContribution(key=k, contribution=c, share=(c / total if total > 0.0 else None))
        for k, c in zip(keys, contributions)
    )


def _sample_health(n: int, min_obs: int, params: ContributionParams, *, small_reason: str) -> tuple[ModelHealth, str | None]:
    if n < min_obs:
        return ModelHealth.UNAVAILABLE, f"n={n} < min_obs={min_obs}"
    if n < params.degraded_multiple * min_obs:
        return ModelHealth.DEGRADED, small_reason
    return ModelHealth.ACTIVE, None


def _meta(name: str, *, params: Mapping[str, Any], n: int, as_of: date | None,
          confidence: float | None = None, distribution: str | None = None) -> ModelMeta:
    return ModelMeta(
        model_name=name,
        model_version=MODEL_VERSION,
        params=params,
        return_type=None,
        frequency="1D",
        lookback=n if n >= 1 else None,
        data_source=None,
        as_of=as_of,
        confidence=confidence,
        horizon_days=1,
        distribution=distribution,
        # §5: Euler decomposition of an unconditional historical estimate.
        tier=ModelTier.TIER_1,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def volatility_contributions(
    positions_pnl: Mapping[str, Sequence[float]],
    *,
    portfolio_pnl: Sequence[float] | None = None,
    min_obs: int | None = None,
    params: ContributionParams = DEFAULT_PARAMS,
    as_of: date | None = None,
) -> ContributionResult:
    """Volatility risk contribution ``RC^σ_i = cov(pnl_i, pnl_p) / σ_p``
    (ddof=1); ``total = σ_p`` (spec §10; contract §2.5).

    ``portfolio_pnl`` (optional) is checked against the derived sum
    (``ValueError`` on mismatch). ``min_obs`` defaults to
    ``params.min_obs_vol`` (60). ``σ_p = 0`` ⇒ UNAVAILABLE.
    """
    if min_obs is None:
        min_obs = params.min_obs_vol
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError(f"min_obs must be an int >= 2, got {min_obs!r}")
    keys, columns, pnl_p = _portfolio_series(positions_pnl, portfolio_pnl, params.sum_tolerance)
    n = len(pnl_p)
    meta = _meta(MODEL_NAME_VOL, params={"min_obs": min_obs, "ddof": 1}, n=n, as_of=as_of)

    def _na(reason: str) -> ContributionResult:
        return ContributionResult(None, (), METHOD_VOL, None, None, ModelHealth.UNAVAILABLE, reason, n, meta)

    if not keys:
        return _na("no positions")
    health, reason = _sample_health(n, min_obs, params, small_reason=f"small sample: n={n} < {params.degraded_multiple:g}×min_obs={min_obs}")
    if health is ModelHealth.UNAVAILABLE:
        return _na(reason or "insufficient data")

    mean_p = math.fsum(pnl_p) / n
    dev_p = [v - mean_p for v in pnl_p]
    var_p = math.fsum(d * d for d in dev_p) / (n - 1)
    if var_p <= 0.0:
        return _na(f"portfolio sigma = 0 (constant P&L over n={n})")
    sigma_p = math.sqrt(var_p)
    contributions: list[float] = []
    for col in columns:
        mean_i = math.fsum(col) / n
        cov = math.fsum((col[t] - mean_i) * dev_p[t] for t in range(n)) / (n - 1)
        contributions.append(cov / sigma_p)
    return ContributionResult(
        total=sigma_p,
        per_position=_rows(keys, contributions, sigma_p),
        method=METHOD_VOL,
        confidence=None,
        tail_size=None,
        health=health,
        reason=reason,
        sample_size=n,
        meta=meta,
    )


def es_contributions(
    positions_pnl: Mapping[str, Sequence[float]],
    confidence: float,
    *,
    portfolio_pnl: Sequence[float] | None = None,
    min_obs: int | None = None,
    params: ContributionParams = DEFAULT_PARAMS,
    as_of: date | None = None,
) -> ContributionResult:
    """Euler ES contribution ``RC^ES_i = mean_{t∈T}(−pnl_i,t)`` over the
    tail set ``T`` of the ``k = ceil(n(1−α))`` largest portfolio losses;
    ``total = ES_α = mean_{t∈T}(−pnl_p,t)`` — sums exactly (spec §10, §33;
    contract §2.5).

    ``min_obs`` defaults to ``params.default_min_obs(confidence)`` (60 for
    α < 0.99, 250 for α ≥ 0.99). Diagnostics-grade fields: ``tail_size=k``.
    """
    confidence = _check_confidence(confidence)
    if min_obs is None:
        min_obs = params.default_min_obs(confidence)
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError(f"min_obs must be an int >= 2, got {min_obs!r}")
    keys, columns, pnl_p = _portfolio_series(positions_pnl, portfolio_pnl, params.sum_tolerance)
    n = len(pnl_p)
    k, tail = _tail_indices(pnl_p, confidence)
    meta = _meta(
        MODEL_NAME_ES,
        params={"confidence": confidence, "horizon_days": 1, "min_obs": min_obs, "tail_size": k},
        n=n, as_of=as_of, confidence=confidence, distribution=DISTRIBUTION_EMPIRICAL,
    )

    def _na(reason: str) -> ContributionResult:
        return ContributionResult(None, (), METHOD_ES, confidence, k if k else None, ModelHealth.UNAVAILABLE, reason, n, meta)

    if not keys:
        return _na("no positions")
    health, reason = _sample_health(n, min_obs, params, small_reason=f"small tail: k={k} (n={n} < {params.degraded_multiple:g}×min_obs={min_obs})")
    if health is ModelHealth.UNAVAILABLE:
        return _na(f"{reason} (k={k})")
    total = _tail_mean_loss(pnl_p, tail)
    contributions = [_tail_mean_loss(col, tail) for col in columns]
    return ContributionResult(
        total=total,
        per_position=_rows(keys, contributions, total),
        method=METHOD_ES,
        confidence=confidence,
        tail_size=k,
        health=health,
        reason=reason,
        sample_size=n,
        meta=meta,
    )


def marginal_es(
    candidate_pnl: Sequence[float],
    portfolio_pnl: Sequence[float],
    confidence: float,
    quantity: float,
    *,
    min_obs: int | None = None,
    params: ContributionParams = DEFAULT_PARAMS,
    as_of: date | None = None,
) -> ModelResult:
    """Marginal ES per unit of a candidate held at ``quantity`` (spec §9;
    contract §2.5).

    ``candidate_pnl`` is the candidate's P&L series AT ``quantity`` units;
    ``portfolio_pnl`` is the book WITHOUT the candidate. On the joined
    series ``pnl_new = portfolio + candidate`` take the tail ``T`` (``k =
    ceil(n(1−α))``), ``RC^ES_cand = mean_{t∈T}(−candidate_pnl_t)`` and
    ``value = RC^ES_cand / quantity``. Diagnostics: ``contribution``
    (``RC^ES_cand``), ``quantity``, ``tail_size``, ``n``, ``es_after``
    (ES of the joined series). ``quantity == 0`` is malformed
    (``ValueError``); a short candidate passes a negative quantity.
    """
    confidence = _check_confidence(confidence)
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or not math.isfinite(quantity) or quantity == 0:
        raise ValueError(f"quantity must be a finite non-zero number, got {quantity!r}")
    if min_obs is None:
        min_obs = params.default_min_obs(confidence)
    cand = [float(v) for v in candidate_pnl]
    book = [float(v) for v in portfolio_pnl]
    _check_finite(cand, what="candidate_pnl")
    _check_finite(book, what="portfolio_pnl")
    if len(cand) != len(book):
        raise ValueError(f"candidate_pnl has {len(cand)} observations, portfolio_pnl has {len(book)}")
    n = len(book)
    joined = [b + c for b, c in zip(book, cand)]
    k, tail = _tail_indices(joined, confidence)
    meta = _meta(
        MODEL_NAME_MARGINAL_ES,
        params={"confidence": confidence, "horizon_days": 1, "min_obs": min_obs, "quantity": float(quantity), "tail_size": k},
        n=n, as_of=as_of, confidence=confidence, distribution=DISTRIBUTION_EMPIRICAL,
    )
    health, reason = _sample_health(n, min_obs, params, small_reason=f"small tail: k={k} (n={n} < {params.degraded_multiple:g}×min_obs={min_obs})")
    if health is ModelHealth.UNAVAILABLE:
        return unavailable(meta, f"{reason} (k={k})", n, diagnostics={"n": n, "tail_size": k})
    contribution = _tail_mean_loss(cand, tail)
    diag = {
        "contribution": contribution,
        "quantity": float(quantity),
        "tail_size": k,
        "n": n,
        "es_after": _tail_mean_loss(joined, tail),
    }
    value = contribution / float(quantity)
    if health is ModelHealth.DEGRADED:
        return degraded(meta, reason or "small tail", value, n, diagnostics=diag)
    return active(meta, value, n, diagnostics=diag)


def incremental_es(
    portfolio_pnl: Sequence[float],
    candidate_pnl: Sequence[float],
    confidence: float,
    *,
    min_obs: int | None = None,
    params: ContributionParams = DEFAULT_PARAMS,
    as_of: date | None = None,
) -> IncrementalResult:
    """Incremental ES ``delta = ES_α(book + candidate) − ES_α(book)`` (spec
    §9; contract §2.5), both historical tail averages (contract §2.3) on
    the same ``n`` and ``k``. ``delta`` may be negative (a hedge lowers ES).
    """
    confidence = _check_confidence(confidence)
    if min_obs is None:
        min_obs = params.default_min_obs(confidence)
    cand = [float(v) for v in candidate_pnl]
    book = [float(v) for v in portfolio_pnl]
    _check_finite(cand, what="candidate_pnl")
    _check_finite(book, what="portfolio_pnl")
    if len(cand) != len(book):
        raise ValueError(f"candidate_pnl has {len(cand)} observations, portfolio_pnl has {len(book)}")
    n = len(book)
    k = tail_size(n, confidence)
    meta = _meta(
        MODEL_NAME_INCREMENTAL_ES,
        params={"confidence": confidence, "horizon_days": 1, "min_obs": min_obs, "tail_size": k},
        n=n, as_of=as_of, confidence=confidence, distribution=DISTRIBUTION_EMPIRICAL,
    )
    health, reason = _sample_health(n, min_obs, params, small_reason=f"small tail: k={k} (n={n} < {params.degraded_multiple:g}×min_obs={min_obs})")
    if health is ModelHealth.UNAVAILABLE:
        return IncrementalResult(None, None, None, confidence, k if k else None, ModelHealth.UNAVAILABLE, f"{reason} (k={k})", n, meta)
    before, _ = _historical_es(book, confidence)
    after, _ = _historical_es([b + c for b, c in zip(book, cand)], confidence)
    return IncrementalResult(
        before=before,
        after=after,
        delta=after - before,
        confidence=confidence,
        tail_size=k,
        health=health,
        reason=reason,
        sample_size=n,
        meta=meta,
    )


__all__ = [
    "ContributionParams",
    "ContributionResult",
    "DEFAULT_PARAMS",
    "DISTRIBUTION_EMPIRICAL",
    "IncrementalResult",
    "METHOD_ES",
    "METHOD_VOL",
    "MODEL_NAME_ES",
    "MODEL_NAME_INCREMENTAL_ES",
    "MODEL_NAME_MARGINAL_ES",
    "MODEL_NAME_VOL",
    "MODEL_VERSION",
    "PositionContribution",
    "es_contributions",
    "incremental_es",
    "marginal_es",
    "tail_size",
    "volatility_contributions",
]
