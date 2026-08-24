"""Model ensemble — dispersion across risk views and the portfolio model-risk
state (risk spec §39, §40, §41, §59; Phase B design contract §2.7).

Pure stdlib, deterministic, no I/O. SHADOW/RESEARCH: nothing here alters a
Tier 0 decision.

The point of this module is the spec §40 mandate: **model disagreement is
information, not noise to be averaged away**. Given several views of the
same quantity (Historical VaR $1,800, Gaussian VaR $1,200, …) this module
NEVER produces "average risk = $1,500". It reports which view is the
smallest, which is the largest, and how far apart they are — and raises a
flag when that spread is wide enough to mean the number itself is not to be
trusted.

Estimators (every number hand-checkable):

- **Dispersion** (contract §2.7). Over the *comparable* views — those whose
  ``health`` is ``ACTIVE`` or ``DEGRADED`` **and** whose ``value`` is a
  finite number ``> 0`` — let ``lo = min(values)`` and ``hi =
  max(values)``::

      ratio = hi / lo

  ``min_name`` / ``max_name`` are the keys attaining them; ties are broken
  by the mapping's iteration order (first key wins — stable and
  deterministic given a ``dict``). ``flag = "MODEL_DISPERSION_HIGH"`` iff
  ``ratio > params.high_ratio`` (default 1.5, i.e. the widest view is more
  than 50% above the narrowest), else ``None``. A single comparable view
  gives ``ratio = 1.0`` and no flag but ``DEGRADED`` health — one view is
  not an ensemble, and saying "no disagreement" would be dishonest.
  Fewer than ``params.min_views`` (default 2) comparable views ⇒
  ``ratio=None``, ``health=UNAVAILABLE``, ``reason`` with the real counts.

  Views that are ``UNAVAILABLE``/``FAILED``, or carry ``value=None``, or a
  non-positive value, are **ignored** for the ratio (a ratio through zero
  or across a sign change is meaningless) but are counted in
  ``n_views`` / ``n_excluded`` and named in ``excluded`` so the omission is
  visible rather than silent.

- **Model risk state** (contract §2.7, spec §59). A rule table over four
  boolean triggers, each of which appends a human-readable reason:

      1. ``dispersion_high``  — the dispersion flag fired;
      2. ``gaussian_trust_low`` — ``distribution.gaussian_trust == "LOW"``
         *and* at least one Gaussian view is comparable (a Gaussian-trust
         warning about a Gaussian model nobody is using is not model risk);
      3. ``core_unavailable`` — at least one *core* view (the caller names
         them; typically Historical VaR/ES) is ``UNAVAILABLE``;
      4. ``sample_degraded`` — at least one view is ``DEGRADED``.

  Then::

      any view FAILED                     -> HIGH   (overrides the count)
      n_triggers >= params.high_triggers  -> HIGH   (default 2)
      n_triggers == 1                     -> ELEVATED
      n_triggers == 0                     -> LOW

  A ``FAILED`` view is HIGH on its own because a failed estimator is an
  unknown, not a wide number. ``reasons`` lists the triggers that actually
  fired, with the real numbers — never a fixed sentence.

Both results carry ``health`` so ``PortfolioRiskSnapshot.health_summary()``
(contract §2.11) can fold them in, and a ``ModelMeta`` so the number is
reproducible (spec §44; ``model_version="1.0.0"``, bump per contract §4).

Malformed input (``high_ratio ≤ 1``, a non-``ModelResult`` view) raises
``ValueError``; missing data never raises — it degrades health.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .base import ModelHealth, ModelMeta, ModelResult

#: Model names / version recorded in ``ModelMeta`` (contract §4).
MODEL_VERSION = "1.0.0"
MODEL_NAME_DISPERSION = "model_dispersion"
MODEL_NAME_MODEL_RISK = "model_risk_state"

#: The spec §40 flag.
FLAG_DISPERSION_HIGH = "MODEL_DISPERSION_HIGH"

#: Model-risk states (spec §59).
RISK_LOW = "LOW"
RISK_ELEVATED = "ELEVATED"
RISK_HIGH = "HIGH"

#: ``gaussian_trust`` value that counts as a trigger (see ``diagnostics.py``).
TRUST_LOW = "LOW"

#: Healths whose views take part in the dispersion ratio (contract §2.7).
COMPARABLE_HEALTH = (ModelHealth.ACTIVE, ModelHealth.DEGRADED)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleParams:
    """Every threshold of this module (house rule: never a magic number).

    - ``high_ratio`` (1.5): ``ratio > high_ratio`` ⇒
      ``MODEL_DISPERSION_HIGH`` (spec §40 — the example there,
      $1,200 vs $5,100, is a ratio of 4.25);
    - ``min_views`` (2): fewer comparable views than this ⇒ dispersion
      ``UNAVAILABLE`` (a ratio needs two numbers);
    - ``degraded_views`` (2): fewer comparable views than this ⇒ the
      dispersion result is ``DEGRADED`` even when a ratio exists (with
      ``min_views=1`` a lone view yields ``ratio=1.0``, which must not read
      as ACTIVE "models agree");
    - ``high_triggers`` (2): number of model-risk triggers at or above
      which the state is ``HIGH`` (spec §59; one trigger ⇒ ``ELEVATED``);
    - ``backtest_red_triggers`` (1, ADDITIVE — Phase E design §9.4): how
      many RED walk-forward backtest verdicts on CORE views it takes to
      count as ONE model-risk trigger. A model whose 99 % VaR is being
      breached far more often than 1 % of days is *mis-calibrated*, which
      is model risk by definition (spec §59 "failed diagnostics"); 1 is the
      strictest honest setting — a single core view failing its coverage
      test is already evidence. Set higher to require corroboration. The
      caller supplies the verdicts (``backtest_red_count``); this module
      never reads a database.
    """

    high_ratio: float = 1.5
    min_views: int = 2
    degraded_views: int = 2
    high_triggers: int = 2
    backtest_red_triggers: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.high_ratio, (int, float)) or isinstance(self.high_ratio, bool):
            raise ValueError(f"high_ratio must be a number > 1, got {self.high_ratio!r}")
        if not math.isfinite(self.high_ratio) or self.high_ratio <= 1.0:
            raise ValueError(f"high_ratio must be > 1, got {self.high_ratio}")
        for name in ("min_views", "degraded_views", "high_triggers",
                     "backtest_red_triggers"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")


DEFAULT_PARAMS = EnsembleParams()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersionResult:
    """Disagreement across risk views (contract §2.7; spec §39, §40).

    - ``ratio``: ``max/min`` over comparable views (``None`` when fewer
      than ``params.min_views`` are comparable) — never an average;
    - ``min_name`` / ``max_name`` / ``min_value`` / ``max_value``: the
      views attaining the extremes (``None`` when ``ratio`` is);
    - ``flag``: ``"MODEL_DISPERSION_HIGH"`` or ``None``;
    - ``n_views``: how many views were offered; ``n_comparable``: how many
      took part; ``excluded``: ``(name, why)`` rows for the rest — the
      omission is always visible.
    """

    ratio: float | None
    min_name: str | None
    max_name: str | None
    min_value: float | None
    max_value: float | None
    flag: str | None
    n_views: int
    n_comparable: int
    excluded: tuple[tuple[str, str], ...]
    health: ModelHealth
    reason: str | None
    meta: ModelMeta

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        object.__setattr__(self, "excluded", tuple(self.excluded))
        if health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED) and self.ratio is not None:
            raise ValueError(f"health={health} requires ratio=None")
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if self.ratio is not None and (not math.isfinite(self.ratio) or self.ratio < 1.0):
            raise ValueError(f"ratio must be finite and >= 1, got {self.ratio!r}")
        if self.flag is not None and self.flag != FLAG_DISPERSION_HIGH:
            raise ValueError(f"flag must be {FLAG_DISPERSION_HIGH!r} or None, got {self.flag!r}")

    @property
    def is_available(self) -> bool:
        return self.ratio is not None

    @property
    def is_high(self) -> bool:
        """True iff the ``MODEL_DISPERSION_HIGH`` flag fired."""
        return self.flag == FLAG_DISPERSION_HIGH


@dataclass(frozen=True)
class ModelRiskState:
    """Portfolio-level model risk (spec §59; contract §2.7).

    ``state`` is ``"LOW" | "ELEVATED" | "HIGH"``; ``reasons`` names every
    trigger that actually fired (empty iff ``LOW``); ``triggers`` is the
    boolean rule table as evaluated, so a reader can replay the decision.
    """

    state: str
    reasons: tuple[str, ...]
    triggers: Mapping[str, bool]
    n_triggers: int
    health: ModelHealth
    meta: ModelMeta
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "triggers", dict(self.triggers))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        if self.state not in (RISK_LOW, RISK_ELEVATED, RISK_HIGH):
            raise ValueError(f"state must be LOW/ELEVATED/HIGH, got {self.state!r}")
        if self.state == RISK_LOW and self.reasons:
            raise ValueError(f"state=LOW must carry no reasons, got {self.reasons!r}")
        if self.state != RISK_LOW and not self.reasons:
            raise ValueError(f"state={self.state} requires at least one reason")

    @property
    def is_low(self) -> bool:
        return self.state == RISK_LOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(name: str, *, params: Mapping[str, Any], n: int,
          as_of: date | datetime | None) -> ModelMeta:
    return ModelMeta(
        model_name=name,
        model_version=MODEL_VERSION,
        params=params,
        return_type=None,
        frequency=None,
        lookback=None,
        data_source=None,
        as_of=as_of,
        confidence=None,
        horizon_days=None,
        distribution=None,
    )


def _comparable(views: Mapping[str, ModelResult]) -> tuple[
    list[tuple[str, float]], list[tuple[str, str]]
]:
    """Split ``views`` into ``(usable, excluded)``.

    Usable = health ACTIVE/DEGRADED with a finite value > 0 (contract §2.7).
    Excluded rows are ``(name, why)`` with the real health/value so the
    caller can show *why* a view sat out.
    """
    usable: list[tuple[str, float]] = []
    excluded: list[tuple[str, str]] = []
    for name, view in views.items():
        if not isinstance(view, ModelResult):
            raise ValueError(f"views[{name!r}] must be a ModelResult, got {type(view).__name__}")
        key = str(name)
        health = ModelHealth(view.health)
        if health not in COMPARABLE_HEALTH:
            excluded.append((key, f"health={health}"))
            continue
        value = view.value
        if value is None:
            excluded.append((key, "value=None"))
            continue
        value = float(value)
        if not math.isfinite(value):
            excluded.append((key, f"value={value!r} not finite"))
            continue
        if value <= 0.0:
            excluded.append((key, f"value={value:g} <= 0 (ratio undefined)"))
            continue
        usable.append((key, value))
    return usable, excluded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispersion(
    views: Mapping[str, ModelResult],
    *,
    params: EnsembleParams = DEFAULT_PARAMS,
    as_of: date | datetime | None = None,
) -> DispersionResult:
    """Model disagreement across ``views`` (spec §39, §40; contract §2.7).

    ``ratio = max(values) / min(values)`` over views that are ACTIVE or
    DEGRADED with a finite positive value; ``flag =
    "MODEL_DISPERSION_HIGH"`` iff ``ratio > params.high_ratio`` (1.5).
    Views that are UNAVAILABLE/FAILED or carry ``value=None`` or a
    non-positive value are ignored for the ratio and reported in
    ``excluded``. **Never averages** — that is the whole point of §40.

    Fewer than ``params.min_views`` comparable views ⇒ ``ratio=None``,
    ``health=UNAVAILABLE``, ``reason`` with the real counts (no exception).
    Fewer than ``params.degraded_views`` ⇒ ``DEGRADED``.

    Hand-check: views ``{"HIST": 1800, "GAUSS": 1200}`` ⇒ ratio =
    1800/1200 = 1.5, which is NOT ``> 1.5`` ⇒ no flag. Adding
    ``{"STRESS": 5100}`` ⇒ ratio = 5100/1200 = 4.25 ⇒ flag.
    """
    usable, excluded = _comparable(views)
    n_views = len(views)
    n_comparable = len(usable)
    meta = _meta(
        MODEL_NAME_DISPERSION,
        params={
            "high_ratio": params.high_ratio,
            "min_views": params.min_views,
            "n_views": n_views,
        },
        n=n_comparable,
        as_of=as_of,
    )

    if n_comparable < params.min_views:
        return DispersionResult(
            ratio=None,
            min_name=None,
            max_name=None,
            min_value=None,
            max_value=None,
            flag=None,
            n_views=n_views,
            n_comparable=n_comparable,
            excluded=tuple(excluded),
            health=ModelHealth.UNAVAILABLE,
            reason=(
                f"only {n_comparable} comparable view(s) < min_views="
                f"{params.min_views} (of {n_views} offered)"
            ),
            meta=meta,
        )

    # Ties resolved by iteration order: strict < / > keeps the FIRST extreme.
    min_name, min_value = usable[0]
    max_name, max_value = usable[0]
    for name, value in usable[1:]:
        if value < min_value:
            min_name, min_value = name, value
        if value > max_value:
            max_name, max_value = name, value

    ratio = max_value / min_value
    flag = FLAG_DISPERSION_HIGH if ratio > params.high_ratio else None

    if n_comparable < params.degraded_views:
        health: ModelHealth = ModelHealth.DEGRADED
        reason: str | None = (
            f"only {n_comparable} comparable view(s) < degraded_views="
            f"{params.degraded_views}: dispersion is not meaningful"
        )
    elif excluded:
        health = ModelHealth.DEGRADED
        reason = (
            f"{len(excluded)} of {n_views} view(s) excluded: "
            + ", ".join(f"{name} ({why})" for name, why in excluded)
        )
    else:
        health, reason = ModelHealth.ACTIVE, None

    return DispersionResult(
        ratio=ratio,
        min_name=min_name,
        max_name=max_name,
        min_value=min_value,
        max_value=max_value,
        flag=flag,
        n_views=n_views,
        n_comparable=n_comparable,
        excluded=tuple(excluded),
        health=health,
        reason=reason,
        meta=meta,
    )


def model_risk_state(
    views: Mapping[str, ModelResult],
    *,
    dispersion_result: DispersionResult | None = None,
    gaussian_trust: str | None = None,
    gaussian_views: Sequence[str] = (),
    core_views: Sequence[str] = (),
    backtest_red_count: int = 0,
    backtest_red_reasons: Sequence[str] = (),
    params: EnsembleParams = DEFAULT_PARAMS,
    as_of: date | datetime | None = None,
) -> ModelRiskState:
    """Portfolio model-risk state ``LOW | ELEVATED | HIGH`` (spec §59;
    contract §2.7).

    Rule table — each trigger appends a reason carrying the real numbers:

    1. ``dispersion_high``    — ``dispersion_result.is_high``;
    2. ``gaussian_trust_low`` — ``gaussian_trust == "LOW"`` AND at least
       one name in ``gaussian_views`` is a comparable view (ACTIVE or
       DEGRADED with a value): a Gaussian-trust warning only matters when a
       Gaussian model is in play;
    3. ``core_unavailable``   — some name in ``core_views`` is UNAVAILABLE
       (or absent from ``views`` entirely — a core view that was never
       computed is exactly as unavailable);
    4. ``sample_degraded``    — some view is DEGRADED;
    5. ``backtest_red``       — ADDITIVE (Phase E design §9.4):
       ``backtest_red_count >= params.backtest_red_triggers``. The caller
       counts RED walk-forward backtest verdicts on the views IT considers
       core and passes the count plus one reason each; this module reads no
       database and holds no opinion about which view is core. A
       mis-calibrated VaR (breaching far more or far less often than its
       confidence promises) is model risk in the plainest sense — spec §59
       lists "failed diagnostics" among the inputs. Defaults to 0, so every
       existing caller keeps its exact previous behaviour.

    Then: **any FAILED view ⇒ HIGH** (a failed estimator is an unknown, not
    a wide number, so it outranks the count); else ``n_triggers ≥
    params.high_triggers`` (2) ⇒ HIGH; ``== 1`` ⇒ ELEVATED; ``0`` ⇒ LOW.

    ``gaussian_trust`` comes from ``diagnostics.DistributionResult``
    (pass ``result.gaussian_trust``); ``core_views`` / ``gaussian_views``
    are key names of ``views`` — the caller decides which views are core
    (typically the historical VaR/ES pair), because "core" is a policy
    choice, not a statistic.

    Health: ``ACTIVE`` — this is a *classification* of other results, and
    it is always computable (an empty ensemble is honestly ``LOW`` with no
    triggers, and ``diagnostics["n_views"]=0`` says so).
    """
    for name, view in views.items():
        if not isinstance(view, ModelResult):
            raise ValueError(f"views[{name!r}] must be a ModelResult, got {type(view).__name__}")

    healths = {str(name): ModelHealth(v.health) for name, v in views.items()}
    reasons: list[str] = []

    # -- trigger 1: dispersion --------------------------------------------
    dispersion_high = bool(dispersion_result is not None and dispersion_result.is_high)
    if dispersion_high:
        assert dispersion_result is not None
        reasons.append(
            f"model dispersion high: ratio={dispersion_result.ratio:.3g} "
            f"({dispersion_result.max_name}/{dispersion_result.min_name}) "
            f"> {params.high_ratio:g}"
        )

    # -- trigger 2: Gaussian trust ----------------------------------------
    live_gaussian = [
        n for n in (str(g) for g in gaussian_views)
        if healths.get(n) in COMPARABLE_HEALTH and views[n].value is not None
    ]
    gaussian_trust_low = bool(gaussian_trust == TRUST_LOW and live_gaussian)
    if gaussian_trust_low:
        reasons.append(
            f"gaussian_trust=LOW with Gaussian view(s) active: {', '.join(live_gaussian)}"
        )

    # -- trigger 3: a core view is unavailable -----------------------------
    missing_core = [
        n for n in (str(c) for c in core_views)
        if healths.get(n, ModelHealth.UNAVAILABLE) is ModelHealth.UNAVAILABLE
    ]
    core_unavailable = bool(missing_core)
    if core_unavailable:
        reasons.append(f"core view(s) UNAVAILABLE: {', '.join(missing_core)}")

    # -- trigger 4: a degraded sample --------------------------------------
    degraded_names = [n for n, h in healths.items() if h is ModelHealth.DEGRADED]
    sample_degraded = bool(degraded_names)
    if sample_degraded:
        reasons.append(f"view(s) DEGRADED: {', '.join(degraded_names)}")

    # -- trigger 5: RED backtest verdicts on core views (Phase E §9.4) ----
    if isinstance(backtest_red_count, bool) or not isinstance(backtest_red_count, int) \
            or backtest_red_count < 0:
        raise ValueError(
            f"backtest_red_count must be an int >= 0, got {backtest_red_count!r}"
        )
    backtest_red = backtest_red_count >= params.backtest_red_triggers
    if backtest_red:
        supplied = tuple(str(r) for r in backtest_red_reasons)
        reasons.append(
            f"{backtest_red_count} RED backtest verdict(s) on core view(s) "
            f">= backtest_red_triggers={params.backtest_red_triggers}"
            + (": " + "; ".join(supplied) if supplied else "")
        )

    triggers = {
        "dispersion_high": dispersion_high,
        "gaussian_trust_low": gaussian_trust_low,
        "core_unavailable": core_unavailable,
        "sample_degraded": sample_degraded,
        "backtest_red": backtest_red,
    }
    n_triggers = sum(1 for fired in triggers.values() if fired)

    failed_names = [n for n, h in healths.items() if h is ModelHealth.FAILED]
    if failed_names:
        # FAILED outranks the count: prepend so the reason reads first.
        reasons.insert(0, f"view(s) FAILED: {', '.join(failed_names)}")
        state = RISK_HIGH
    elif n_triggers >= params.high_triggers:
        state = RISK_HIGH
    elif n_triggers >= 1:
        # Any fired trigger below the HIGH threshold is ELEVATED. With the
        # default high_triggers=2 this is exactly the contract's "exactly
        # one"; with a raised threshold it keeps every counted trigger
        # visible instead of falling through to LOW (which would contradict
        # the non-empty `reasons` and trip ModelRiskState's own invariant).
        state = RISK_ELEVATED
    else:
        state = RISK_LOW

    meta = _meta(
        MODEL_NAME_MODEL_RISK,
        params={
            "high_triggers": params.high_triggers,
            "high_ratio": params.high_ratio,
            "backtest_red_triggers": params.backtest_red_triggers,
            "core_views": tuple(str(c) for c in core_views),
            "gaussian_views": tuple(str(g) for g in gaussian_views),
        },
        n=len(views),
        as_of=as_of,
    )
    return ModelRiskState(
        state=state,
        reasons=tuple(reasons),
        triggers=triggers,
        n_triggers=n_triggers,
        health=ModelHealth.ACTIVE,
        meta=meta,
        diagnostics={
            "n_views": len(views),
            "failed": tuple(failed_names),
            "degraded": tuple(degraded_names),
            "core_unavailable": tuple(missing_core),
            "backtest_red_count": backtest_red_count,
            "dispersion_ratio": (
                dispersion_result.ratio if dispersion_result is not None else None
            ),
        },
    )


__all__ = [
    "COMPARABLE_HEALTH",
    "DEFAULT_PARAMS",
    "DispersionResult",
    "EnsembleParams",
    "FLAG_DISPERSION_HIGH",
    "MODEL_NAME_DISPERSION",
    "MODEL_NAME_MODEL_RISK",
    "MODEL_VERSION",
    "ModelRiskState",
    "RISK_ELEVATED",
    "RISK_HIGH",
    "RISK_LOW",
    "TRUST_LOW",
    "dispersion",
    "model_risk_state",
]
