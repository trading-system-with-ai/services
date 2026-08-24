"""Risk model abstraction, health, metadata & registry (spec §4, §41, §44,
§57, §70; Phase B design contract §2.2).

Pure stdlib, deterministic, no I/O. Every statistical model in
``libs/trading_core/risk/models/`` returns a ``ModelResult`` built from the
constructors in this module so that health / reason / metadata have ONE
shape across VaR, ES, volatility, contribution, diagnostics and drawdown.

Conventions (contract §1, §2.2):

- **Health** (spec §41): ``ACTIVE`` (computed, nothing to flag),
  ``DEGRADED`` (computed but caveated — e.g. small tail), ``UNAVAILABLE``
  (not computed — insufficient data; ``value is None``), ``FAILED``
  (estimator error; ``value is None``). Every non-ACTIVE result carries a
  non-empty ``reason`` with the real numbers (``"n=17 < min_obs=60"``).
  Missing data NEVER raises — it degrades health; malformed input raises
  ``ValueError``.
- **Worst-of ordering** ``ACTIVE < DEGRADED < UNAVAILABLE < FAILED`` is
  used by ``combine_health`` and by the validate guard.
- **Calculation vs validation** (spec §57): ``calculate`` never claims more
  than ACTIVE-if-computed; ``validate`` is a separate step that may only
  DOWNGRADE health (``validate_never_upgrades`` — contract §3 invariant 8).
- **Mode** (spec §70): ``RESEARCH`` (offline study only, numbers may be
  shown in research views), ``SHADOW`` (calculate, display, log the
  hypothetical approve/reject — DO NOT alter trading), ``PRODUCTION``
  (eligible to feed a decision). Promotion is SHADOW → PRODUCTION only
  after validation (spec §68). The Tier 0 engine consults ``mode``: a
  model that is not ``PRODUCTION`` cannot be wired into a veto. Every model
  defaults to SHADOW.
- **Metadata** (spec §44): ``ModelMeta`` carries everything needed to
  reproduce a number — name, version, params, return type, frequency,
  lookback, data source, as_of, confidence, horizon, distribution.
  ``model_version`` starts at ``"1.0.0"``; any estimator arithmetic change
  bumps MAJOR, a parameter-default change bumps MINOR (contract §4).

Nothing here alters a Tier 0 decision.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # annotation only — keeps this module import-cycle-free
    from ..returns import ReturnType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelHealth(StrEnum):
    """Spec §41 model health. Ordered worst-of: ACTIVE < DEGRADED < UNAVAILABLE < FAILED."""

    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


_HEALTH_RANK: Mapping[ModelHealth, int] = {
    ModelHealth.ACTIVE: 0,
    ModelHealth.DEGRADED: 1,
    ModelHealth.UNAVAILABLE: 2,
    ModelHealth.FAILED: 3,
}

# Health states in which the estimator produced NO number (value must be None).
_NO_VALUE_HEALTH: frozenset[ModelHealth] = frozenset(
    {ModelHealth.UNAVAILABLE, ModelHealth.FAILED}
)


def health_rank(health: ModelHealth) -> int:
    """Severity rank of a health state (0 = ACTIVE … 3 = FAILED)."""
    return _HEALTH_RANK[ModelHealth(health)]


class ModelMode(StrEnum):
    """Spec §70 lifecycle mode. Only PRODUCTION may feed a decision."""

    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"


class ModelTier(StrEnum):
    """Spec §5 model CLASSIFICATION — the machine-readable artefact.

    §5 prescribes an ORDERING of model families, and the platform obeyed it
    exactly; what was missing until now was the classification itself, so a
    reader could not ask a model what tier it is. This enum is that answer.
    It is ORTHOGONAL to :class:`ModelMode` (lifecycle) and to the engine's
    ``LAYER_*`` taxonomy (where a constraint binds): a Tier 1 model can be
    SHADOW, and a Tier 2 model can be RESEARCH.

    - ``TIER_0`` — the hard limits that actually decide today
      (``risk/engine.py``'s nine-gate ladder). NO statistical model in
      ``risk/models/`` carries this tier: Tier 0 is not a fitted model, and
      stamping one TIER_0 would misdescribe the decision hierarchy §72
      lays out. It exists in the enum so the vocabulary is complete and a
      reader can see the tier a model is NOT.
    - ``TIER_1`` — unconditional / historical estimators over a fixed
      window: historical & Gaussian VaR/ES, portfolio σ and covariance,
      Euler risk contributions, distribution diagnostics, drawdown, and the
      stress catalogue (§5 lists stress testing in the first tier — it is a
      deterministic reprice of the current book, not a fitted forecast).
    - ``TIER_2`` — CONDITIONAL volatility models: EWMA and GARCH(1,1), and
      every VaR/ES view filtered through one of them.
    - ``TIER_3`` — the extreme-tail and dependence-structure family
      (§16/§17/§20/§35). Every member is documented-deferred or
      documented-rejected: NO implementation exists anywhere in ``libs/``
      or ``apps/``, and the adversarial suite enforces that absence with a
      source scan, which is why this docstring does not name the models —
      naming them here would trip the very tripwire that protects the
      deferral. Present in the enum for the same reason as TIER_0: so the
      vocabulary is complete and a reader can see the tier nothing is in.

    Nothing consults this field to make a decision — it is provenance
    (§44), served beside the number and persisted with it.
    """

    TIER_0 = "TIER_0"
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


# ---------------------------------------------------------------------------
# Metadata & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMeta:
    """Spec §44 provenance — everything needed to reproduce the number.

    ``params`` is shallow-copied into a fresh ``dict`` at construction so a
    caller mutating its own mapping afterwards cannot alter the recorded
    provenance; treat it as read-only. Validation is minimal and structural
    (malformed ⇒ ``ValueError``): non-empty name/version, ``confidence`` in
    (0.5, 1), ``horizon_days >= 1``, ``lookback >= 1`` when given.
    """

    model_name: str
    model_version: str
    params: Mapping[str, Any] = field(default_factory=dict)
    return_type: "ReturnType | None" = None
    frequency: str | None = None
    lookback: int | None = None
    data_source: str | None = None
    as_of: date | datetime | None = None
    confidence: float | None = None
    horizon_days: int | None = None
    distribution: str | None = None
    #: Spec §5 classification (ADDITIVE, default ``None``). ``None`` means
    #: "this meta was built before/outside the taxonomy", never TIER_0 —
    #: defaulting an unclassified model into the tier that DECIDES would be
    #: the one dishonest default available here.
    tier: "ModelTier | None" = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")
        if self.confidence is not None and not (0.5 < self.confidence < 1.0):
            raise ValueError(
                f"confidence must be in (0.5, 1), got {self.confidence}"
            )
        if self.horizon_days is not None and self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")
        if self.lookback is not None and self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if self.tier is not None:
            # A bad spelling is malformed input, not a silent None: the whole
            # point of the field is that it can be trusted when present.
            object.__setattr__(self, "tier", ModelTier(self.tier))
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class ModelResult:
    """One model output (contract §2.2).

    Shape rules enforced at construction (uniform across all models):

    - ``health`` in {UNAVAILABLE, FAILED} ⇒ ``value is None`` (honest null).
    - ``health != ACTIVE`` ⇒ ``reason`` is a non-empty string.
    - ``value`` when present is a finite float (NaN/inf is malformed input
      to a result, never a reportable number).
    - ``sample_size >= 0`` (observations actually used).
    - ``diagnostics`` is shallow-copied into a fresh ``dict``; small scalars
      only (``tail_size``, ``n``, ``annualized_usd`` …).
    """

    value: float | None
    health: ModelHealth
    reason: str | None
    sample_size: int
    meta: ModelMeta
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(
                self.value, (int, float)
            ):
                raise ValueError(f"value must be a float or None, got {self.value!r}")
            if not math.isfinite(self.value):
                raise ValueError(f"value must be finite, got {self.value!r}")
            object.__setattr__(self, "value", float(self.value))
        if health in _NO_VALUE_HEALTH and self.value is not None:
            raise ValueError(f"health={health} requires value=None, got {self.value}")
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if isinstance(self.sample_size, bool) or not isinstance(
            self.sample_size, int
        ) or self.sample_size < 0:
            raise ValueError(f"sample_size must be an int >= 0, got {self.sample_size!r}")
        if not isinstance(self.meta, ModelMeta):
            raise ValueError("meta must be a ModelMeta")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def is_available(self) -> bool:
        """True iff a number was produced (ACTIVE or DEGRADED)."""
        return self.value is not None


# ---------------------------------------------------------------------------
# Result constructors — every model module builds results through these
# ---------------------------------------------------------------------------


def _require_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return reason


def active(
    meta: ModelMeta,
    value: float,
    sample_size: int,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ModelResult:
    """ACTIVE result: computed, nothing to flag (``reason=None``)."""
    if value is None:
        raise ValueError("active() requires a value")
    return ModelResult(
        value=value,
        health=ModelHealth.ACTIVE,
        reason=None,
        sample_size=sample_size,
        meta=meta,
        diagnostics=diagnostics or {},
    )


def degraded(
    meta: ModelMeta,
    reason: str,
    value: float,
    sample_size: int,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ModelResult:
    """DEGRADED result: a number was computed but is caveated (``reason`` says why)."""
    if value is None:
        raise ValueError("degraded() requires a value; use unavailable() when none exists")
    return ModelResult(
        value=value,
        health=ModelHealth.DEGRADED,
        reason=_require_reason(reason),
        sample_size=sample_size,
        meta=meta,
        diagnostics=diagnostics or {},
    )


def unavailable(
    meta: ModelMeta,
    reason: str,
    sample_size: int,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ModelResult:
    """UNAVAILABLE result: not computed (insufficient data). ``value=None``, never 0."""
    return ModelResult(
        value=None,
        health=ModelHealth.UNAVAILABLE,
        reason=_require_reason(reason),
        sample_size=sample_size,
        meta=meta,
        diagnostics=diagnostics or {},
    )


def failed(
    meta: ModelMeta,
    reason: str,
    sample_size: int = 0,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ModelResult:
    """FAILED result: the estimator errored (e.g. non-convergence). ``value=None``."""
    return ModelResult(
        value=None,
        health=ModelHealth.FAILED,
        reason=_require_reason(reason),
        sample_size=sample_size,
        meta=meta,
        diagnostics=diagnostics or {},
    )


# ---------------------------------------------------------------------------
# Health algebra & validation guard
# ---------------------------------------------------------------------------


def combine_health(*healths: ModelHealth) -> ModelHealth:
    """Worst-of several health states (ACTIVE < DEGRADED < UNAVAILABLE < FAILED).

    Used when one metric depends on several inputs (e.g. a snapshot line
    built from a covariance result and a P&L series). At least one health
    is required — an empty combination has no honest answer.
    """
    if not healths:
        raise ValueError("combine_health requires at least one health")
    return max((ModelHealth(h) for h in healths), key=health_rank)


def validate_never_upgrades(before: ModelResult, after: ModelResult) -> ModelResult:
    """Guard for ``RiskModel.validate`` implementations (spec §57; contract §3.8).

    Returns ``after`` unchanged if its health is the same or worse than
    ``before``'s; raises ``ValueError`` if validation tried to UPGRADE
    health (e.g. UNAVAILABLE → ACTIVE). Also refuses a validation that
    conjures a value the calculation did not produce.
    """
    if health_rank(after.health) < health_rank(before.health):
        raise ValueError(
            "validate() must never upgrade health: "
            f"{before.health} -> {after.health}"
        )
    if before.value is None and after.value is not None:
        raise ValueError("validate() must never introduce a value the calculation did not produce")
    return after


def downgrade(result: ModelResult, to: ModelHealth, reason: str) -> ModelResult:
    """Return ``result`` downgraded to ``combine_health(result.health, to)``.

    ``reason`` (non-empty) is appended to any existing reason with ``"; "``.
    Downgrading to UNAVAILABLE/FAILED drops the value (honest null). The
    output is checked with ``validate_never_upgrades`` so a caller passing a
    *better* health than the current one is a no-op on health, never an
    upgrade.
    """
    _require_reason(reason)
    new_health = combine_health(result.health, to)
    new_reason = f"{result.reason}; {reason}" if result.reason else reason
    new_value = None if new_health in _NO_VALUE_HEALTH else result.value
    after = _dc_replace(result, health=new_health, reason=new_reason, value=new_value)
    return validate_never_upgrades(result, after)


# ---------------------------------------------------------------------------
# Model protocol & base class
# ---------------------------------------------------------------------------


@runtime_checkable
class RiskModel(Protocol):
    """Spec §4 model abstraction. ``mode`` is consulted by the engine: only
    ``PRODUCTION`` models may ever be wired into a veto."""

    name: str
    version: str
    mode: ModelMode

    def calculate(self, *args: Any, **kwargs: Any) -> ModelResult: ...

    def validate(self, result: ModelResult) -> ModelResult: ...  # may downgrade; never upgrades

    def diagnostics(self, result: ModelResult) -> Mapping[str, Any]: ...

    def metadata(self) -> ModelMeta: ...


class BaseRiskModel(ABC):
    """Convenience base implementing ``RiskModel``.

    Subclasses set the class attributes ``name`` (registry key, e.g.
    ``"historical_var"``), ``version`` (``"1.0.0"``; bump per contract §4)
    and optionally ``distribution``, ``return_type``, ``frequency``,
    ``data_source``; override ``params()`` to expose their parameter
    dataclass as a mapping; and implement ``calculate``.

    Defaults: ``mode = SHADOW`` (spec §70 — never a decision input until
    promoted); ``validate`` = identity (subclasses may downgrade, and MUST
    route through ``validate_never_upgrades`` / ``downgrade``);
    ``diagnostics(result)`` = ``result.diagnostics``; ``metadata()`` builds
    a ``ModelMeta`` from the class attributes + ``params()``, with keyword
    overrides for the per-call fields (``as_of``, ``lookback``,
    ``confidence``, ``horizon_days``, …) so a model's ``calculate`` can
    stamp the exact inputs it used.
    """

    name: str = ""
    version: str = "1.0.0"
    mode: ModelMode = ModelMode.SHADOW
    #: Spec §5 tier (ADDITIVE). Subclasses set it; ``metadata()`` stamps it
    #: onto every ``ModelMeta`` they build, so a registered model always
    #: answers "what tier am I?" without the caller knowing the taxonomy.
    tier: ModelTier | None = None
    distribution: str | None = None
    return_type: "ReturnType | None" = None
    frequency: str | None = None
    data_source: str | None = None

    def __init__(self, *, mode: ModelMode | None = None) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a non-empty class attribute 'name'")
        if not self.version:
            raise ValueError(f"{type(self).__name__} must define a non-empty class attribute 'version'")
        if mode is not None:
            self.mode = ModelMode(mode)

    def params(self) -> Mapping[str, Any]:
        """Estimator parameters recorded in ``ModelMeta.params`` (override)."""
        return {}

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> ModelResult:
        """Compute the model output; never claims health beyond ACTIVE-if-computed."""

    def validate(self, result: ModelResult) -> ModelResult:
        """Identity by default. Overrides may downgrade only (spec §57)."""
        return validate_never_upgrades(result, result)

    def diagnostics(self, result: ModelResult) -> Mapping[str, Any]:
        return result.diagnostics

    def metadata(self, **overrides: Any) -> ModelMeta:
        """``ModelMeta`` from class attributes + ``params()``; ``overrides``
        set/replace any ``ModelMeta`` field (``params`` overrides are MERGED
        on top of ``params()``)."""
        params: dict[str, Any] = dict(self.params())
        extra = overrides.pop("params", None)
        if extra:
            params.update(extra)
        fields: dict[str, Any] = {
            "model_name": self.name,
            "model_version": self.version,
            "params": params,
            "return_type": self.return_type,
            "frequency": self.frequency,
            "data_source": self.data_source,
            "distribution": self.distribution,
            "tier": self.tier,
        }
        fields.update(overrides)
        return ModelMeta(**fields)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r}, mode={self.mode})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, RiskModel] = {}


def register(model: RiskModel, *, replace: bool = False) -> RiskModel:
    """Register ``model`` under ``model.name``.

    Raises ``ValueError`` on an empty name, on an object that does not
    satisfy ``RiskModel``, or on a duplicate name unless ``replace=True``.
    Returns the model so it can be used as a decorator-ish one-liner
    (``MODEL = register(HistoricalVaRModel())``).
    """
    if not isinstance(model, RiskModel):
        raise ValueError(f"{model!r} does not implement RiskModel")
    name = getattr(model, "name", "")
    if not isinstance(name, str) or not name:
        raise ValueError("model.name must be a non-empty string")
    if name in REGISTRY and not replace:
        raise ValueError(f"model {name!r} already registered (pass replace=True to override)")
    REGISTRY[name] = model
    return model


def get(name: str) -> RiskModel:
    """Registered model by name; ``KeyError`` (with the known names) if absent."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no risk model registered as {name!r}; known: {names()}") from None


def names() -> tuple[str, ...]:
    """Registered model names, sorted (deterministic)."""
    return tuple(sorted(REGISTRY))


def clear_for_tests() -> None:
    """Empty the registry (test isolation only)."""
    REGISTRY.clear()


# ---------------------------------------------------------------------------
# Tier lookup by model NAME (spec §5)
# ---------------------------------------------------------------------------

#: Model names whose tier is TIER_2 (CONDITIONAL volatility models and every
#: view filtered through one). Everything else statistical in this package is
#: TIER_1; TIER_0 is the hard-limit engine (no model) and TIER_3 is deferred.
#:
#: Kept as a NAME set rather than derived from the registry because the tier
#: question is also asked about numbers that never went through a registered
#: model — the validation rows are keyed by the estimator's name, and the
#: conditional views are stamped ``garch_<name>`` by the snapshot builder
#: when GARCH is the live filter.
_TIER_2_NAME_PARTS: frozenset[str] = frozenset(
    {"conditional", "garch", "ewma"}
)


def tier_for_model_name(name: str) -> ModelTier | None:
    """The spec §5 tier of a model NAME, or ``None`` when it is not one.

    Resolution order, most authoritative first:

    1. a REGISTERED model's own ``tier`` (the model answers for itself);
    2. the conditional-family name test — any name containing
       ``conditional``, ``garch`` or ``ewma`` is TIER_2, which covers the
       ``garch_historical_var`` names the snapshot builder synthesises and
       the ``garch_var`` / ``conditional_var`` keys the validation grid
       uses, neither of which is a registry entry;
    3. otherwise TIER_1 for a non-empty name — every remaining estimator in
       this package is an unconditional one.

    An empty / non-string name is ``None`` (honest null): a tier is never
    guessed for something that is not a model.
    """
    if not isinstance(name, str) or not name:
        return None
    model = REGISTRY.get(name)
    tier = getattr(model, "tier", None) if model is not None else None
    if tier is not None:
        return ModelTier(tier)
    lowered = name.lower()
    if any(part in lowered for part in _TIER_2_NAME_PARTS):
        return ModelTier.TIER_2
    return ModelTier.TIER_1


__all__ = [
    "BaseRiskModel",
    "ModelHealth",
    "ModelMeta",
    "ModelMode",
    "ModelResult",
    "ModelTier",
    "REGISTRY",
    "RiskModel",
    "active",
    "clear_for_tests",
    "combine_health",
    "degraded",
    "downgrade",
    "failed",
    "get",
    "health_rank",
    "names",
    "register",
    "tier_for_model_name",
    "unavailable",
    "validate_never_upgrades",
]
