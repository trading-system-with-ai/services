"""Distribution diagnostics — skewness, excess kurtosis, Jarque–Bera, and the
"how much do we trust Gaussian VaR" signal (risk spec §15; Phase B design
contract §2.6).

Pure stdlib, deterministic, no I/O. SHADOW/RESEARCH: nothing here alters a
Tier 0 decision.

Input is a P&L series (USD per horizon, gain-positive) OR a return series —
the estimators are scale- and location-free apart from ``mean``/``stdev``,
so the same function serves both; the caller records which in ``meta``.

Estimators (contract §2.6 — every number hand-checkable):

- **Central moments about the sample mean, POPULATION form** (divisor ``n``,
  *not* ``n − 1``)::

      mean = Σ_t x_t / n
      m_r  = Σ_t (x_t − mean)^r / n        for r = 2, 3, 4

- **Sample skewness** ``g1 = m3 / m2^{3/2}`` (Fisher–Pearson, the *biased*
  ``g1``; no ``√(n(n−1))/(n−2)`` adjustment — that would be ``G1``).
- **Excess kurtosis** ``g2 = m4 / m2² − 3`` (0 for a Gaussian; the biased
  ``g2``, not ``G2``).
- **Jarque–Bera** ``JB = n/6 · (g1² + g2²/4)``.
- **p-value**: JB is asymptotically ``χ²(2)``, whose survival function has
  the closed form ``P(X > x) = exp(−x/2)`` — so ``jb_p = exp(−JB/2)``
  exactly, with no incomplete gamma (contract §2.6). It is clamped to
  ``[0, 1]`` only by construction (``JB ≥ 0`` ⇒ ``p ∈ (0, 1]``). Note this
  is the asymptotic p-value: at small ``n`` JB over-rejects, which is one
  more reason ``min_obs`` gates the label.

``stdev`` on the result is the SAMPLE standard deviation (ddof=1,
``statistics.stdev``) because that is what the rest of the platform reports
(``portfolio_volatility``, Gaussian VaR); the moments feeding ``g1``/``g2``
use the population divisor as documented above. The two are intentionally
different and both are stated so a reader can hand-check either.

Labels (thresholds are parameters on ``DistributionParams``, never inline):

- ``UNSTABLE``  — ``n < min_obs`` OR variance ≈ 0 (``m2 ≤ zero_variance``).
  Nothing else can be said honestly; ``g1``/``g2``/``JB`` are ``None``
  because they are ``0/0``.
- ``LEFT_SKEWED`` — ``g1 < left_skew`` (default −0.5): the loss tail is the
  long one.
- ``HEAVY_TAIL``  — ``g2 > heavy_tail_kurtosis`` (default 1.0).
- ``NORMAL_LIKE`` — ``jb_p ≥ normal_p`` (default 0.05) and neither heavy
  nor left-skewed.

A series may carry several flags (e.g. both ``HEAVY_TAIL`` and
``LEFT_SKEWED``); ``flags`` is the full tuple in priority order and
``primary`` is the first by the priority ``UNSTABLE > LEFT_SKEWED >
HEAVY_TAIL > NORMAL_LIKE``. When a series is neither normal-like nor heavy
nor skewed (``jb_p < normal_p`` on its own) ``flags`` is empty and
``primary`` is ``None`` — an honest "not Gaussian, but not in a way this
tool names".

``gaussian_trust`` (spec §15 "reduce trust in Gaussian VaR"):

- ``LOW``     if ``HEAVY_TAIL`` or ``LEFT_SKEWED`` (or ``UNSTABLE``);
- ``REDUCED`` if ``jb_p < normal_p`` only;
- ``HIGH``    otherwise.

Health: ``UNSTABLE`` ⇒ ``UNAVAILABLE`` with ``value=None`` and a reason
carrying the real numbers; ``min_obs ≤ n < degraded_multiple × min_obs`` ⇒
``DEGRADED``; otherwise ``ACTIVE``. Malformed input (non-finite value)
raises ``ValueError``; missing data never raises.

Every result carries a ``ModelMeta`` (``model_version="1.0.0"``; bump per
contract §4) so the number is reproducible (spec §44).
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .base import ModelHealth, ModelMeta, ModelTier

#: Model name / version recorded in ``ModelMeta`` (contract §4).
MODEL_VERSION = "1.0.0"
MODEL_NAME = "distribution_diagnostics"

#: Distribution flags (contract §2.6).
FLAG_NORMAL_LIKE = "NORMAL_LIKE"
FLAG_HEAVY_TAIL = "HEAVY_TAIL"
FLAG_LEFT_SKEWED = "LEFT_SKEWED"
FLAG_UNSTABLE = "UNSTABLE"

#: Priority order for ``primary`` (contract §2.6): first match wins.
FLAG_PRIORITY: tuple[str, ...] = (
    FLAG_UNSTABLE,
    FLAG_LEFT_SKEWED,
    FLAG_HEAVY_TAIL,
    FLAG_NORMAL_LIKE,
)

#: ``gaussian_trust`` levels (spec §15).
TRUST_HIGH = "HIGH"
TRUST_REDUCED = "REDUCED"
TRUST_LOW = "LOW"


# ---------------------------------------------------------------------------
# Parameters & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionParams:
    """Every threshold of this module (house rule: never a magic number).

    - ``min_obs`` (60): below this the sample cannot support a shape call ⇒
      ``UNSTABLE`` / ``UNAVAILABLE``;
    - ``heavy_tail_kurtosis`` (1.0): ``g2 >`` this ⇒ ``HEAVY_TAIL``;
    - ``left_skew`` (−0.5): ``g1 <`` this ⇒ ``LEFT_SKEWED``;
    - ``normal_p`` (0.05): ``jb_p ≥`` this (and no heavy/skew flag) ⇒
      ``NORMAL_LIKE``; ``jb_p <`` this alone ⇒ ``gaussian_trust=REDUCED``;
    - ``zero_variance`` (1e-18): ``m2 ≤`` this counts as a constant series;
    - ``degraded_multiple`` (2.0): ``n < degraded_multiple × min_obs`` ⇒
      ``DEGRADED``.
    """

    min_obs: int = 60
    heavy_tail_kurtosis: float = 1.0
    left_skew: float = -0.5
    normal_p: float = 0.05
    zero_variance: float = 1e-18
    degraded_multiple: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.min_obs, bool) or not isinstance(self.min_obs, int) or self.min_obs < 3:
            raise ValueError(f"min_obs must be an int >= 3, got {self.min_obs!r}")
        if not (0.0 < self.normal_p < 1.0):
            raise ValueError(f"normal_p must be in (0, 1), got {self.normal_p}")
        if not (self.zero_variance >= 0.0):
            raise ValueError(f"zero_variance must be >= 0, got {self.zero_variance}")
        if not (self.degraded_multiple >= 1.0):
            raise ValueError(f"degraded_multiple must be >= 1, got {self.degraded_multiple}")
        if not math.isfinite(self.heavy_tail_kurtosis):
            raise ValueError(f"heavy_tail_kurtosis must be finite, got {self.heavy_tail_kurtosis}")
        if not math.isfinite(self.left_skew):
            raise ValueError(f"left_skew must be finite, got {self.left_skew}")


DEFAULT_PARAMS = DistributionParams()


@dataclass(frozen=True)
class DistributionResult:
    """Shape of one P&L / return series (contract §2.6).

    ``mean``/``stdev`` are always reported when ``n ≥ 2`` (they are honest
    even for a constant series); ``skew``/``excess_kurtosis``/
    ``jarque_bera``/``jb_p`` are ``None`` when the series is ``UNSTABLE``
    (``0/0`` has no value). ``flags`` is the full tuple in priority order,
    ``primary`` the first of them (``None`` when no flag applies).
    """

    n: int
    mean: float | None
    stdev: float | None
    skew: float | None
    excess_kurtosis: float | None
    jarque_bera: float | None
    jb_p: float | None
    flags: tuple[str, ...]
    primary: str | None
    gaussian_trust: str
    health: ModelHealth
    reason: str | None
    meta: ModelMeta

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        object.__setattr__(self, "flags", tuple(self.flags))
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if self.gaussian_trust not in (TRUST_HIGH, TRUST_REDUCED, TRUST_LOW):
            raise ValueError(f"gaussian_trust must be HIGH/REDUCED/LOW, got {self.gaussian_trust!r}")
        if self.primary is not None and self.primary not in self.flags:
            raise ValueError(f"primary={self.primary!r} must be one of flags={self.flags!r}")
        for name in ("mean", "stdev", "skew", "excess_kurtosis", "jarque_bera", "jb_p"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be finite or None, got {v!r}")

    @property
    def is_available(self) -> bool:
        """True when the shape statistics were computed (not ``UNSTABLE``)."""
        return self.skew is not None

    def has(self, flag: str) -> bool:
        return flag in self.flags


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def _check_finite(values: Sequence[float], *, what: str) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what}: every value must be a finite number, got {v!r}")
        out.append(float(v))
    return out


def jarque_bera_p_value(jb: float) -> float:
    """``P(χ²(2) > JB) = exp(−JB/2)`` — the closed form used platform-wide
    (contract §2.6). ``JB = 0`` ⇒ ``p = 1``; ``JB = 5.99`` ⇒ ``p ≈ 0.0500``.
    """
    if jb < 0.0:
        raise ValueError(f"jarque_bera must be >= 0, got {jb}")
    return math.exp(-jb / 2.0)


def distribution_diagnostics(
    pnl_or_returns: Sequence[float],
    *,
    params: DistributionParams = DEFAULT_PARAMS,
    as_of: date | None = None,
    label: str | None = None,
) -> DistributionResult:
    """Skewness, excess kurtosis, Jarque–Bera and Gaussian trust for one
    series (spec §15; contract §2.6).

    Moments are population-form about the sample mean (divisor ``n``):
    ``g1 = m3/m2^{3/2}``, ``g2 = m4/m2² − 3``, ``JB = n/6·(g1² + g2²/4)``,
    ``jb_p = exp(−JB/2)``. ``stdev`` is the ddof=1 sample stdev.

    ``n < params.min_obs`` or a (near-)constant series ⇒ ``UNSTABLE``,
    ``health=UNAVAILABLE``, shape statistics ``None``, ``reason`` with the
    real numbers — never an exception, never a fabricated 0. Non-finite
    input raises ``ValueError``.

    ``label`` (optional) is recorded in ``meta.params["series"]`` so a
    snapshot can say which series was diagnosed ("book_pnl", "SPY returns").
    """
    values = _check_finite(pnl_or_returns, what="pnl_or_returns")
    n = len(values)

    meta_params: dict[str, Any] = {
        "min_obs": params.min_obs,
        "heavy_tail_kurtosis": params.heavy_tail_kurtosis,
        "left_skew": params.left_skew,
        "normal_p": params.normal_p,
    }
    if label is not None:
        meta_params["series"] = label
    meta = ModelMeta(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        params=meta_params,
        return_type=None,
        frequency="1D",
        lookback=n if n >= 1 else None,
        data_source=None,
        as_of=as_of,
        confidence=None,
        horizon_days=1,
        distribution="EMPIRICAL",
        # §5: unconditional moments / normality test over a fixed window.
        tier=ModelTier.TIER_1,
    )

    mean = math.fsum(values) / n if n >= 1 else None
    stdev = statistics.stdev(values) if n >= 2 else None

    def _unstable(reason: str) -> DistributionResult:
        return DistributionResult(
            n=n,
            mean=mean,
            stdev=stdev,
            skew=None,
            excess_kurtosis=None,
            jarque_bera=None,
            jb_p=None,
            flags=(FLAG_UNSTABLE,),
            primary=FLAG_UNSTABLE,
            gaussian_trust=TRUST_LOW,
            health=ModelHealth.UNAVAILABLE,
            reason=reason,
            meta=meta,
        )

    if n < params.min_obs:
        return _unstable(f"n={n} < min_obs={params.min_obs}")

    # Population central moments about the sample mean (divisor n).
    assert mean is not None  # n >= min_obs >= 3
    dev = [v - mean for v in values]
    m2 = math.fsum(d * d for d in dev) / n
    if m2 <= params.zero_variance:
        return _unstable(f"variance ~ 0 (m2={m2!r} <= {params.zero_variance}) over n={n}")
    m3 = math.fsum(d * d * d for d in dev) / n
    m4 = math.fsum(d * d * d * d for d in dev) / n

    g1 = m3 / (m2 ** 1.5)
    g2 = m4 / (m2 * m2) - 3.0
    jb = (n / 6.0) * (g1 * g1 + (g2 * g2) / 4.0)
    jb_p = jarque_bera_p_value(jb)

    flags: list[str] = []
    if g1 < params.left_skew:
        flags.append(FLAG_LEFT_SKEWED)
    if g2 > params.heavy_tail_kurtosis:
        flags.append(FLAG_HEAVY_TAIL)
    if jb_p >= params.normal_p and not flags:
        flags.append(FLAG_NORMAL_LIKE)
    ordered = tuple(f for f in FLAG_PRIORITY if f in flags)
    primary = ordered[0] if ordered else None

    if FLAG_HEAVY_TAIL in ordered or FLAG_LEFT_SKEWED in ordered:
        trust = TRUST_LOW
    elif jb_p < params.normal_p:
        trust = TRUST_REDUCED
    else:
        trust = TRUST_HIGH

    if n < params.degraded_multiple * params.min_obs:
        health: ModelHealth = ModelHealth.DEGRADED
        reason: str | None = (
            f"small sample: n={n} < {params.degraded_multiple:g}×min_obs={params.min_obs}"
        )
    else:
        health, reason = ModelHealth.ACTIVE, None

    return DistributionResult(
        n=n,
        mean=mean,
        stdev=stdev,
        skew=g1,
        excess_kurtosis=g2,
        jarque_bera=jb,
        jb_p=jb_p,
        flags=ordered,
        primary=primary,
        gaussian_trust=trust,
        health=health,
        reason=reason,
        meta=meta,
    )


__all__ = [
    "DEFAULT_PARAMS",
    "DistributionParams",
    "DistributionResult",
    "FLAG_HEAVY_TAIL",
    "FLAG_LEFT_SKEWED",
    "FLAG_NORMAL_LIKE",
    "FLAG_PRIORITY",
    "FLAG_UNSTABLE",
    "MODEL_NAME",
    "MODEL_VERSION",
    "TRUST_HIGH",
    "TRUST_LOW",
    "TRUST_REDUCED",
    "distribution_diagnostics",
    "jarque_bera_p_value",
]
