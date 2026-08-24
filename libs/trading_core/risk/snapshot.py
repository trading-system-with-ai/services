"""Typed portfolio risk snapshot (risk spec §45, §55; Phase B design
contract §2.11).

Pure stdlib, deterministic, no I/O. This module is the *container* the
gateway snapshot builder fills from the Phase B estimators; it computes
nothing statistical itself. Every metric field holds a typed result
(:class:`~libs.trading_core.risk.models.base.ModelResult` or one of the
typed result dataclasses of the sibling modules) — never untyped JSON. The
``to_api_dict()`` serialiser lives in the gateway (contract §2.11), not
here.

Field types (contract §2.11). Result types from modules written alongside
this one are referenced by name only (``TYPE_CHECKING`` imports) so this
module stays import-cycle-free and importable before its siblings exist;
the expected concrete types are:

- ``volatility``       : ``ModelResult | None``  (``risk/models/volatility.py``)
- ``var`` / ``es``     : ``Mapping[str, ModelResult]`` keyed
  ``"<METHOD>:<confidence>:<horizon_days>"``, e.g. ``"HISTORICAL:0.95:1"``,
  ``"GAUSSIAN:0.99:1"``, ``"HISTORICAL_VOL_SCALED:0.95:1"``
  (``risk/models/var_es.py``)
- ``drawdown``         : ``DrawdownResult | None``  (``risk/models/drawdown.py``)
- ``greeks``           : ``PortfolioGreeks | None``  (``libs/trading_core/greeks.py``)
- ``contributions_vol`` / ``contributions_es`` : ``ContributionResult | None``
  (``risk/models/contribution.py``; ``method="VOL"`` / ``"ES"``)
- ``distribution``     : ``DistributionResult | None``  (``risk/models/diagnostics.py``)
- ``dispersion``       : ``DispersionResult | None``  (``risk/models/ensemble.py``)
- ``model_risk``       : ``ModelRiskState | None``  (``risk/models/ensemble.py``)
- ``stress``      : ``StressResult | None``  (``risk/models/stress.py``;
  the §25/§26 catalogue run over this book, whose ``worst`` row is the
  headline stress loss the API serves)
- ``correlation_state``: ``CorrelationState | None``  (``libs/trading_core/
  correlation.py``; the §19 regime of the book's tickers on LOG returns).
  Declared since Phase B, actually POPULATED by the builder since the §45
  fix — before that it was always ``None`` while the same object reached
  the wire through a separate dict, so the typed snapshot and the API
  disagreed about a field both claimed to carry.
- ``data_quality``     : :class:`DataQuality`
- ``model_health``     : ``Mapping[str, ModelHealth]`` — the builder's own
  per-model health ledger (e.g. ``{"historical_var": ACTIVE, ...}``)
- ``risk_state``       : ``str`` — the Tier 0 heat state today (``heat_state``)
- ``ttl``              : :class:`TtlPolicy`

Conventions:

- **Sign / units** are inherited from the contained results (contract §1):
  VaR/ES/contributions are USD losses per horizon, positive = money lost;
  exposures are USD; ``heat_pct`` is a fraction of NAV.
- **Staleness** (spec §55, model-specific TTL): ``is_stale(now, kind)`` is
  ``(now - as_of) > ttl(kind)`` — a snapshot exactly ``ttl`` old is NOT
  stale, ``ttl + 1s`` is. ``kind`` is ``"statistical"`` (VaR/ES/vol/…,
  default 86400 s = one trading day, they are daily-close numbers) or
  ``"greeks"`` (default 120 s — live NBBO greeks decay in seconds/minutes).
  ``now`` and ``as_of`` must agree on tz-awareness (else ``ValueError``).
- **Health** (spec §41): ``health_summary()`` collects the health of every
  contained result (duck-typed on a ``.health`` attribute holding a
  ``ModelHealth``); ``overall_health()`` is the WORST-OF over that summary
  plus the builder's ``model_health`` ledger (``ACTIVE < DEGRADED <
  UNAVAILABLE < FAILED``), further degraded to at least ``DEGRADED`` when
  ``data_quality.valid`` is False (missing tickers etc. — contract §2.9).
  A snapshot containing no results at all is ``UNAVAILABLE`` (honest null,
  never a fabricated ACTIVE).
- **Versioning**: :data:`SNAPSHOT_VERSION` = ``"b.1"`` is stamped on every
  snapshot; bump when a field is added/removed/re-typed so persisted
  snapshots (spec §44, migration 018) stay interpretable.
- **Immutability**: frozen dataclasses; ``Mapping`` fields are shallow-
  copied into fresh dicts at construction; tuple fields are tuples.

Nothing here alters a Tier 0 decision.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from libs.trading_core.greeks import PortfolioGreeks
from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMeta,
    ModelResult,
    ModelTier,
    active,
    combine_health,
    degraded,
    unavailable,
)

if TYPE_CHECKING:  # annotation only — keeps this module import-cycle-free
    from libs.trading_core.correlation import CorrelationState
    from libs.trading_core.risk.models.contribution import ContributionResult
    from libs.trading_core.risk.models.diagnostics import DistributionResult
    from libs.trading_core.risk.models.drawdown import DrawdownResult
    from libs.trading_core.risk.models.ensemble import DispersionResult, ModelRiskState
    from libs.trading_core.risk.models.stress import StressResult

#: Snapshot schema version (contract §2.11 / §4). Bump on any field change.
SNAPSHOT_VERSION = "b.1"

#: Staleness kinds accepted by :meth:`PortfolioRiskSnapshot.is_stale`.
STALENESS_KIND_STATISTICAL = "statistical"
STALENESS_KIND_GREEKS = "greeks"
_STALENESS_KINDS: frozenset[str] = frozenset(
    {STALENESS_KIND_STATISTICAL, STALENESS_KIND_GREEKS}
)


# ---------------------------------------------------------------------------
# Policies & data quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TtlPolicy:
    """Model-specific time-to-live (spec §55).

    - ``statistical_seconds`` (default 86400): VaR / ES / volatility /
      contribution / drawdown — daily-close statistics, valid for a day.
    - ``greeks_seconds`` (default 120): live portfolio greeks from the NBBO
      chain — decay in minutes.

    Both must be ``> 0`` (a zero/negative TTL would make every snapshot
    stale at birth — malformed configuration, ``ValueError``).
    """

    statistical_seconds: float = 86400.0
    greeks_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in ("statistical_seconds", "greeks_seconds"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
            object.__setattr__(self, name, float(v))

    def seconds_for(self, kind: str) -> float:
        """TTL in seconds for ``kind`` (``"statistical"`` | ``"greeks"``); ``ValueError`` otherwise."""
        if kind == STALENESS_KIND_STATISTICAL:
            return self.statistical_seconds
        if kind == STALENESS_KIND_GREEKS:
            return self.greeks_seconds
        raise ValueError(
            f"unknown staleness kind {kind!r}; expected one of {sorted(_STALENESS_KINDS)}"
        )


@dataclass(frozen=True)
class DataQuality:
    """Input-data quality of a snapshot (contract §2.11).

    - ``as_of``: the return-window end used by the estimators (last bar
      date) — may differ from the live ``PortfolioRiskSnapshot.as_of``;
    - ``oldest_bar`` / ``newest_bar``: first/last dates of the aligned
      return window (``None`` when there was no window);
    - ``tickers_missing``: tickers with no return column (positions
      EXCLUDED from the book — contract §2.9), sorted;
    - ``n_obs``: aligned observations actually available (``>= 0``);
    - ``valid``: the builder's verdict that inputs were sufficient for the
      core views (``False`` ⇒ snapshot ``overall_health()`` ≥ DEGRADED);
    - ``reasons``: real-number explanations when not fully valid
      (``"tickers_missing=('XYZ',)"``, ``"n_obs=17 < min_obs=60"``).
    """

    as_of: date | datetime | None
    oldest_bar: date | None
    newest_bar: date | None
    tickers_missing: tuple[str, ...] = ()
    n_obs: int = 0
    valid: bool = True
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.n_obs, bool) or not isinstance(self.n_obs, int) or self.n_obs < 0:
            raise ValueError(f"n_obs must be an int >= 0, got {self.n_obs!r}")
        if (
            self.oldest_bar is not None
            and self.newest_bar is not None
            and self.oldest_bar > self.newest_bar
        ):
            raise ValueError(
                f"oldest_bar {self.oldest_bar} is after newest_bar {self.newest_bar}"
            )
        object.__setattr__(self, "tickers_missing", tuple(self.tickers_missing))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not self.valid and not self.reasons:
            raise ValueError("valid=False requires at least one reason")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _fresh_result_map(name: str, m: Mapping[str, ModelResult]) -> dict[str, ModelResult]:
    out: dict[str, ModelResult] = {}
    for k, v in m.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"{name} keys must be non-empty strings, got {k!r}")
        if not isinstance(v, ModelResult):
            raise ValueError(f"{name}[{k!r}] must be a ModelResult, got {type(v).__name__}")
        out[k] = v
    return out


@dataclass(frozen=True)
class PortfolioRiskSnapshot:
    """One typed portfolio risk snapshot (spec §45; contract §2.11).

    See the module docstring for the concrete type of every field. All
    ``Mapping`` fields are shallow-copied into fresh dicts; ``as_of`` must
    be a ``datetime`` (live snapshot time — spec §55 requires it).
    """

    as_of: datetime
    nav: float
    cash: float
    cash_reserved: float
    gross_exposure: float
    delta_adjusted_exposure: float
    heat_pct: float
    heat_state: str
    volatility: ModelResult | None = None
    var: Mapping[str, ModelResult] = field(default_factory=dict)
    es: Mapping[str, ModelResult] = field(default_factory=dict)
    drawdown: DrawdownResult | None = None
    greeks: PortfolioGreeks | None = None
    contributions_vol: ContributionResult | None = None
    contributions_es: ContributionResult | None = None
    distribution: DistributionResult | None = None
    dispersion: DispersionResult | None = None
    model_risk: ModelRiskState | None = None
    correlation_state: CorrelationState | None = None
    #: §45 (ADDITIVE): the stress catalogue run over this book. Declared and
    #: POPULATED from the start, unlike ``correlation_state``, which spent
    #: all of Phase C declared-but-never-passed while the same object
    #: reached the wire through a separate dict — the exact defect this
    #: field exists to avoid repeating. ``None`` only when the stress layer
    #: produced nothing at all (it raised, or no catalogue was built); an
    #: UNAVAILABLE run is a real ``StressResult`` carrying its reason.
    stress: StressResult | None = None
    data_quality: DataQuality = field(
        default_factory=lambda: DataQuality(as_of=None, oldest_bar=None, newest_bar=None)
    )
    model_health: Mapping[str, ModelHealth] = field(default_factory=dict)
    risk_state: str = ""
    ttl: TtlPolicy = field(default_factory=TtlPolicy)
    snapshot_version: str = SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, datetime):
            raise ValueError(f"as_of must be a datetime, got {type(self.as_of).__name__}")
        for name in (
            "nav", "cash", "cash_reserved", "gross_exposure",
            "delta_adjusted_exposure", "heat_pct",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
            object.__setattr__(self, name, float(v))
        if self.volatility is not None and not isinstance(self.volatility, ModelResult):
            raise ValueError("volatility must be a ModelResult or None")
        if self.greeks is not None and not isinstance(self.greeks, PortfolioGreeks):
            raise ValueError("greeks must be a PortfolioGreeks or None")
        if not isinstance(self.data_quality, DataQuality):
            raise ValueError("data_quality must be a DataQuality")
        if not isinstance(self.ttl, TtlPolicy):
            raise ValueError("ttl must be a TtlPolicy")
        if not self.snapshot_version:
            raise ValueError("snapshot_version must be non-empty")
        object.__setattr__(self, "var", _fresh_result_map("var", self.var))
        object.__setattr__(self, "es", _fresh_result_map("es", self.es))
        if not self.risk_state:
            object.__setattr__(self, "risk_state", self.heat_state)
        mh: dict[str, ModelHealth] = {}
        for k, v in self.model_health.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"model_health keys must be non-empty strings, got {k!r}")
            mh[k] = ModelHealth(v)  # ValueError on an unknown state
        object.__setattr__(self, "model_health", mh)

    # -- staleness (spec §55) ------------------------------------------------

    def age_seconds(self, now: datetime) -> float:
        """``(now - as_of)`` in seconds (negative if ``now`` precedes ``as_of``).

        ``ValueError`` if ``now`` is not a datetime or its tz-awareness
        differs from ``as_of`` (comparing naive with aware is malformed).
        """
        if not isinstance(now, datetime):
            raise ValueError(f"now must be a datetime, got {type(now).__name__}")
        if (now.tzinfo is None) != (self.as_of.tzinfo is None):
            raise ValueError("now and as_of must both be naive or both tz-aware")
        return (now - self.as_of).total_seconds()

    def is_stale(self, now: datetime, kind: str = STALENESS_KIND_STATISTICAL) -> bool:
        """True iff ``(now - as_of) > ttl.seconds_for(kind)``.

        Exactly ``ttl`` old ⇒ NOT stale; ``ttl + 1s`` ⇒ stale. ``kind`` is
        ``"statistical"`` (default) or ``"greeks"``; anything else is a
        ``ValueError``.
        """
        return self.age_seconds(now) > self.ttl.seconds_for(kind)

    # -- health (spec §41) ---------------------------------------------------

    def health_summary(self) -> Mapping[str, ModelHealth]:
        """Health of every contained result, keyed by field (dict, insertion order).

        Keys: ``"volatility"``, ``"var:<key>"``, ``"es:<key>"``,
        ``"drawdown"``, ``"contributions_vol"``, ``"contributions_es"``,
        ``"distribution"``, ``"dispersion"``, ``"model_risk"`` — a field
        that is ``None`` (or whose result carries no ``ModelHealth``
        ``.health``) is simply absent. ``greeks`` carries no health and is
        never listed; the builder's ``model_health`` ledger is NOT merged
        here (see :meth:`overall_health`).
        """
        out: dict[str, ModelHealth] = {}

        def _put(key: str, obj: object) -> None:
            if obj is None:
                return
            h = getattr(obj, "health", None)
            if h is None:
                return
            try:
                out[key] = ModelHealth(h)
            except ValueError:
                return  # a foreign health enum — not a ModelHealth, skip honestly

        _put("volatility", self.volatility)
        for k, r in self.var.items():
            _put(f"var:{k}", r)
        for k, r in self.es.items():
            _put(f"es:{k}", r)
        _put("drawdown", self.drawdown)
        _put("contributions_vol", self.contributions_vol)
        _put("contributions_es", self.contributions_es)
        _put("distribution", self.distribution)
        _put("dispersion", self.dispersion)
        _put("model_risk", self.model_risk)
        return out

    def overall_health(self) -> ModelHealth:
        """Worst-of :meth:`health_summary` ∪ ``model_health`` values
        (``ACTIVE < DEGRADED < UNAVAILABLE < FAILED``); at least ``DEGRADED``
        when ``data_quality.valid`` is False; ``UNAVAILABLE`` when the
        snapshot contains no result and no ledger entry at all.
        """
        healths: list[ModelHealth] = list(self.health_summary().values())
        healths.extend(self.model_health.values())
        if not healths:
            return ModelHealth.UNAVAILABLE
        if not self.data_quality.valid:
            healths.append(ModelHealth.DEGRADED)
        return combine_health(*healths)

    # -- convenience ---------------------------------------------------------

    def result_keys(self) -> Sequence[str]:
        """Sorted ``var`` and ``es`` keys (deterministic listing)."""
        return tuple(sorted(set(self.var) | set(self.es)))


# ---------------------------------------------------------------------------
# Diversification ratio (spec §34; audit.md:215 P1)
# ---------------------------------------------------------------------------

#: Model identity persisted with the metric (contract §4).
DIVERSIFICATION_MODEL_NAME = "diversification_ratio"
DIVERSIFICATION_MODEL_VERSION = "1.0.0"

#: Default minimum aligned observations. RESEARCH DEFAULT — UNVALIDATED.
#: Matches the 95% grid's ``min_obs`` (contract §2.3) so the ratio is never
#: reported on a shorter window than the VaR it sits beside.
DIVERSIFICATION_MIN_OBS = 60


def diversification_ratio(
    per_position_pnl: Mapping[str, Sequence[float]],
    portfolio_pnl: Sequence[float],
    *,
    min_obs: int = DIVERSIFICATION_MIN_OBS,
    as_of: date | datetime | None = None,
) -> ModelResult:
    """Diversification ratio ``DR = (Σ_i w_i σ_i) / σ_p`` (spec §34).

    THE EXACT ESTIMATOR (there is more than one convention; this is ours):

        DR = Σ_i stdev(pnl_i) / stdev(pnl_total)

    with ``stdev`` the SAMPLE standard deviation (``ddof=1``, i.e.
    ``statistics.stdev``) over the SAME aligned window, and every
    ``w_i = 1``. The weights are 1 — not a capital or NAV fraction — because
    the per-position series this platform builds are already **USD P&L**
    (``risk/pnl_series.py``), so ``σ_i`` is already the USD volatility that
    position contributes. Multiplying by a capital weight a second time
    would double-count the position size and produce a number with no
    interpretation. The denominator is the stdev of the **realised total**
    series (``portfolio_pnl``), not a reconstructed ``√(wᵀΣw)``, so the
    ratio inherits the same correlation structure the book actually had.

    Interpretation. ``σ_p ≤ Σ σ_i`` by the triangle inequality, so
    ``DR ≥ 1`` for any imperfectly correlated book; ``DR = 1`` exactly when
    every position is perfectly positively correlated (no diversification at
    all — the §19 failure mode); ``DR`` grows as positions offset. A single
    position gives ``DR = 1`` by construction and is reported as such, with
    the reason saying so — that is a fact about the book, not a model
    failure.

    Honest nulls (contract §1): fewer than ``min_obs`` observations, fewer
    than one position, a non-finite input, a zero portfolio σ (a perfectly
    hedged book has no denominator — the ratio is undefined, not infinite),
    or a series whose length disagrees with ``portfolio_pnl`` ⇒
    ``value=None``, ``health=UNAVAILABLE``, ``reason`` carrying the real
    numbers. Never a fabricated 1.0.

    SHADOW/RESEARCH: this number decides nothing.
    """
    n_total = len(portfolio_pnl)
    keys = sorted(per_position_pnl)
    meta = ModelMeta(
        model_name=DIVERSIFICATION_MODEL_NAME,
        model_version=DIVERSIFICATION_MODEL_VERSION,
        params={
            "min_obs": int(min_obs),
            "ddof": 1,
            "estimator": "sum_stdev_positions / stdev_total",
            "weights": "1.0 per position (the per-position series are USD P&L)",
        },
        as_of=as_of,
        lookback=n_total if n_total >= 1 else None,
        # §5: a ratio of unconditional sample sigmas over a fixed window.
        tier=ModelTier.TIER_1,
    )
    diagnostics: dict[str, Any] = {"n_positions": len(keys), "n": n_total}

    if not keys:
        return unavailable(meta, "no per-position P&L series (empty book)", 0,
                           diagnostics=diagnostics)
    if n_total < min_obs:
        return unavailable(
            meta, f"n={n_total} < min_obs={min_obs}", n_total, diagnostics=diagnostics
        )
    for key in keys:
        series = per_position_pnl[key]
        if len(series) != n_total:
            return unavailable(
                meta,
                f"series {key!r} has {len(series)} observations, "
                f"portfolio has {n_total} — the window is not aligned",
                n_total,
                diagnostics=diagnostics,
            )
        for v in series:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"per_position_pnl[{key!r}] contains a non-finite value {v!r}")
    for v in portfolio_pnl:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"portfolio_pnl contains a non-finite value {v!r}")

    sigma_p = statistics.stdev(portfolio_pnl)
    sigma_sum = math.fsum(statistics.stdev(per_position_pnl[k]) for k in keys)
    diagnostics["sigma_portfolio_usd"] = sigma_p
    diagnostics["sigma_sum_usd"] = sigma_sum
    if sigma_p <= 0.0:
        return unavailable(
            meta,
            f"portfolio sigma is {sigma_p} — a book with no variance has no "
            "diversification ratio (the denominator is zero)",
            n_total,
            diagnostics=diagnostics,
        )
    value = sigma_sum / sigma_p
    if len(keys) == 1:
        return degraded(
            meta,
            "a single position cannot be diversified: DR = 1 by construction",
            value,
            n_total,
            diagnostics=diagnostics,
        )
    if n_total < 2 * min_obs:
        return degraded(
            meta,
            f"n={n_total} < 2x min_obs={2 * min_obs} — the ratio is estimated "
            "on a short window",
            value,
            n_total,
            diagnostics=diagnostics,
        )
    return active(meta, value, n_total, diagnostics=diagnostics)


__all__ = [
    "DIVERSIFICATION_MIN_OBS",
    "DIVERSIFICATION_MODEL_NAME",
    "DIVERSIFICATION_MODEL_VERSION",
    "DataQuality",
    "PortfolioRiskSnapshot",
    "diversification_ratio",
    "SNAPSHOT_VERSION",
    "STALENESS_KIND_GREEKS",
    "STALENESS_KIND_STATISTICAL",
    "TtlPolicy",
]
