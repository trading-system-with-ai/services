"""Statistical risk snapshot builder — the ONE gateway seam between the
stored book and the pure Phase B risk library (risk spec §6–§10, §39–§45,
§55, §56, §70; Phase B design contract §6).

SHADOW, always. Nothing computed here alters a Tier 0 decision, the
``GATE_ORDER`` chain, or an approved quantity. The order path records this
module's output under the RISK_DECISION ``shadow`` block and moves on
unchanged; the risk view renders it under two additive keys.

WHAT THIS MODULE DOES (and deliberately does not do):

- It FETCHES inputs (open positions, their last stored closes, the stored
  daily bar history, broker/simulator cash) and calls the pure library
  (:mod:`libs.trading_core.risk`) for every statistic. No estimator
  arithmetic lives here — that is the library's job and its tests own it.
- It reuses the portfolio view's own helpers
  (``open_positions_with_prices`` / ``position_market_value`` /
  ``portfolio_greeks_read`` / ``open_csp_cash_reserved`` /
  ``stored_bars_by_ticker``, plan §21) so this snapshot can never disagree
  with the book the user sees or the book the risk engine judges.
- It PERSISTS (``persist=True``) one ``risk_snapshots`` row plus one
  ``risk_metrics`` row per (metric, model, confidence, horizon) carrying the
  FULL §44 ``ModelMeta`` inline, plus ``risk_contributions`` rows. It calls
  ``session.add`` + ``flush`` only — **the CALLER commits**, so a
  PRE_TRADE build shares one transaction with the decision that caused it
  (house rule: every risk decision on a write path is audited in the same
  transaction) and the read view commits its own session at the end.
- It writes NO audit event. A snapshot is a measurement, not a decision
  (house rule: read views write no audit events).

HONEST NULLS (§44 rule 18). Every gap is named, never filled:

- no cash (no account) / fewer than ``MIN_OBS_STATISTICAL`` aligned
  observations / a ticker with no stored bars ⇒
  ``DataQuality(valid=False, reasons=[...])`` with the real numbers, and the
  affected models return ``UNAVAILABLE`` with a reason of their own;
- a position whose greeks are unknown (``data_ok`` False, or a delta that
  cannot be recovered) is EXCLUDED from the book P&L and named in
  ``positions_excluded`` with the reason — never priced at zero delta;
- every model call is wrapped: an exception becomes a ``FAILED``
  ``ModelResult`` carrying the exception text. The view never 500s because
  a statistic misbehaved.

DELTA RECOVERY (contract §2.9). ``PositionRiskInput.delta`` is the SIGNED
per-share delta, recovered uniformly for every instrument as
``delta_shares / (quantity × multiplier)`` from the shared
``portfolio_greeks_read`` per-position rows — the one place the platform
decides a position's delta (stock +1, short stock −1, long options the
chain delta, income legs the NEGATED short-leg delta, spreads the net).
Positions are stored with a POSITIVE ``quantity`` (the sign lives in the
per-share delta, exactly as ``greeks.py`` does it), so the product
``quantity × multiplier × delta × spot`` is the same delta-adjusted
exposure ``aggregate_greeks`` reports — one exposure per book.

UNITS (contract §1). VaR / ES / contributions are USD LOSSES per horizon
(positive = money lost); ``*_pct`` fields in the API dicts are FRACTIONS of
NAV; ``pct_nav`` is ``value / NAV`` when ``NAV > 0`` and ``None`` otherwise
(dividing by a zero or negative NAV would be a fabricated ratio).
"""
from __future__ import annotations

import dataclasses

import asyncio
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.common.telemetry import REGISTRY
from libs.trading_core.risk import (
    DataQuality,
    ModelHealth,
    ModelResult,
    PortfolioRiskSnapshot,
    PositionRiskInput,
    RiskLimits,
    TtlPolicy,
    align,
    book_pnl_series,
    conditional_es,
    conditional_var,
    dispersion,
    distribution_diagnostics,
    drawdown,
    es_contributions,
    gaussian_es,
    gaussian_var,
    heat_state,
    historical_es,
    historical_var,
    model_risk_state,
    portfolio_heat,
    portfolio_volatility,
    reconstructed_book_drawdown,
    returns_from_closes,
    volatility_contributions,
)
from libs.trading_core.risk.models.base import (
    ModelMeta,
    ModelMode,
    active,
    combine_health,
    degraded,
    tier_for_model_name,
)
# Spec §34 (audit.md:215, P1): the diversification ratio lives beside the
# snapshot container it is served on. Imported from the LEAF module so the
# gateway keeps its one-way dependency on the pure library.
from libs.trading_core.risk.snapshot import (
    DIVERSIFICATION_MIN_OBS,
    DIVERSIFICATION_MODEL_NAME,
    DIVERSIFICATION_MODEL_VERSION,
    diversification_ratio,
)
# Spec §11 (compliance §3 row 11): the single-factor (SPY) RESEARCH
# diagnostic. Imported from the LEAF module for the same one-way-dependency
# reason as the diversification ratio above. It registers no model, derives
# no cap and gates nothing — it is served for display only.
from libs.trading_core.risk.models.factor import (
    DEFAULT_FACTOR,
    DEFAULT_FACTOR_PARAMS,
    MODEL_NAME as FACTOR_MODEL_NAME,
    MODEL_VERSION as FACTOR_MODEL_VERSION,
    factor_risk_share,
)
from libs.trading_core.risk.pnl_series import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL_CONST_IV,
    book_method_summary,
)
from libs.trading_core.risk.returns import RETURN_TYPE_LOG, RETURN_TYPE_SIMPLE
# Phase D §8.5: the stress engine and the leg-aware revaluation types. Both
# are LEAF modules of the same pure library (`options.reval` imports only
# `options.bs`; `risk.models.stress` imports `options.reval` and, for the cap
# only, `risk.pretrade` from INSIDE the function) — importing them here keeps
# the gateway the single seam and adds no cycle.
from libs.trading_core.options.reval import (
    METHOD_DELTA_LINEAR as REVAL_METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL as REVAL_METHOD_FULL_REVAL,
    OptionLeg,
    StockLeg,
)
from libs.trading_core.risk.models import stress as stress_models
from libs.trading_core.risk.models import garch as garch_models
# Phase C §7.4: the correlation regime lives in the (leaf) correlation
# module, which imports `risk.returns`; importing it HERE — never from
# `libs.trading_core.risk` — keeps that one-way dependency intact.
from libs.trading_core.correlation import correlation_regime

from .db import (
    AtmIvDailyRow,
    RiskContributionRow,
    RiskMetricRow,
    RiskSnapshotRow,
    SessionLocal,
    StressRunRow,
)

logger = logging.getLogger("apps.gateway.risk_snapshot")

# --- Policy constants (parameters, never hardcoded truths — plan §6.2) -----

#: The exchange calendar the SCHEDULED once-per-day rule is measured on.
NEW_YORK = ZoneInfo("America/New_York")

#: Stored daily bars fetched per underlying (≈ 2.5 years of trading days —
#: the contract §2.3 lookback that makes a 30-observation 95% tail).
DEFAULT_LOOKBACK_BARS = 600

#: Minimum aligned observations for the statistical layer to claim valid
#: inputs (contract §2.3 ``min_obs`` for the 95% grid).
MIN_OBS_STATISTICAL = 60

#: The confidence grid the platform reports (contract §1).
CONFIDENCE_95 = 0.95
CONFIDENCE_99 = 0.99

#: §10: the tail size below which a 99 % ES contribution is called NOISY.
#: RESEARCH DEFAULT — UNVALIDATED, like every other threshold in this layer
#: (house rule: never a magic number). Ten tail observations is the point
#: the audit's own "noisy at 99%" objection was made about: at n ≈ 600 the
#: 99 % tail holds ~6 days, so the per-position split of an average over
#: six numbers is dominated by which six days happened to land there.
#: A parameter, not a hardcoded truth — the 95 % block is unaffected.
ES99_NOISY_TAIL_MIN = 10

#: Horizon of every estimated number (contract §1: only 1D is estimated).
HORIZON_DAYS = 1

#: --- §6/§12 multi-day DISPLAY horizons (RESEARCH) -----------------------
#: Additional horizons served as √h-SCALED display rows beside the 1-day
#: numbers. NOTHING here is estimated: the library computes ``VaR_1 × √h``
#: and labels it ``scaling="SQRT_TIME"`` (``var_es.py`` ``_scaling_label``),
#: which is exactly what the audit promised as a RESEARCH display and never
#: built (compliance §6/§12, Tier B). Empirical multi-day estimation stays
#: DEFERRED on ~600 observations — overlapping windows would manufacture
#: precision the sample does not contain.
#:
#: RESEARCH DEFAULTS — UNVALIDATED. √h scaling assumes i.i.d. returns; a
#: book with volatility clustering or mean reversion violates it, and no
#: backtest of these horizons has been run (the walk-forward runner scores
#: the 1-day grid only). Do not gate on these numbers.
DISPLAY_HORIZONS: tuple[int, ...] = (5, 10)

#: Label stamped on the API row and the persisted params for a horizon that
#: was scaled rather than estimated (mirrors ``var_es.SCALING_SQRT_TIME``).
SCALING_SQRT_TIME = "SQRT_TIME"

#: The §34 estimator, stated on the wire so a reader never has to guess which
#: of the several diversification-ratio conventions produced the number.
DIVERSIFICATION_ESTIMATOR = (
    "DR = sum_i stdev(pnl_i) / stdev(pnl_total), ddof=1, w_i = 1 "
    "(the per-position series are already USD P&L); >= 1 for an imperfectly "
    "correlated book, = 1 when every position is perfectly correlated"
)

#: `risk_metrics.metric` discriminator for the §34 row (VARCHAR(32); the
#: column has no CHECK constraint, so no migration is needed for a new name).
METRIC_DIVERSIFICATION_RATIO = "DIVERSIFICATION_RATIO"

#: `risk_metrics.metric` discriminator for the §13 GARCH FIT row (same
#: VARCHAR(32), same no-migration reasoning). Written only when GARCH is the
#: LIVE conditional source, so the row's presence is itself the statement
#: "GARCH was driving the conditional views on this snapshot".
METRIC_COND_VOL_FIT = "COND_VOL_FIT"

#: The §13 fit-row's `model_name`. The registry name is `garch11`; this row
#: describes that model's FIT, so it carries the same identity.
MODEL_NAME_GARCH_FIT = "garch11"

#: The `GarchFit.diagnostics` keys §13 requires to be persisted. Named
#: explicitly rather than dumping the whole diagnostics dict, so the spec's
#: list is visible in the code and a missing key is an honest null rather
#: than a silently absent one.
GARCH_FIT_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "omega",
    "alpha",
    "beta",
    "persistence",
    "half_life",
    "ljung_box_q_sq",
    "ljung_box_p",
    "converged",
)

#: Label for the §12 GARCH horizon sigmas — the CLOSED-FORM variance term
#: structure, deliberately distinct from the historical rows' SQRT_TIME.
SOURCE_GARCH_TERM_STRUCTURE = "GARCH_TERM_STRUCTURE"

#: Mode label for the multi-day display rows — one step BELOW SHADOW (spec
#: §70): these are not even shadow gates, they are a display.
#: The ticker whose stored bars stand in for "the market" in the §11
#: single-factor diagnostic. A documented parameter, not a hidden truth: the
#: label travels with the result (``statistical.factor.factor``) so a book
#: measured against a different proxy can never be read as SPY.
#: **RESEARCH DEFAULT — UNVALIDATED** as a factor model for this book.
FACTOR_TICKER = DEFAULT_FACTOR

MODE_RESEARCH = "RESEARCH"

#: SHADOW mode label stamped on the API block (spec §70).
MODE_SHADOW = "SHADOW"

#: --- Phase D stress policy (design §8.5) — parameters, never magic -------

#: Calendar days per year used to turn a contract's DTE into the pricer's
#: ``t_years``. CALENDAR, not trading: Black-Scholes time is wall-clock time
#: and the chain's ``dte`` is a calendar count (routers/options.py) — mixing
#: 252 in here would silently shorten every expiry by ~40 %.
DAYS_PER_YEAR = 365.0

#: Risk-free rate / dividend yield handed to the pricer for scenario
#: revaluation. The SAME defaults ``libs/trading_core/options/bs.py`` uses, so
#: a stress reprice and a greek read never disagree about the carry.
#: RESEARCH DEFAULTS — UNVALIDATED (no term-structure read exists yet).
STRESS_RATE = 0.04
STRESS_DIVIDEND_YIELD = 0.0

#: Leg-key suffixes for the two legs of a spread position. A spread is ONE
#: stored row but TWO option legs, and ``scenario_pnl`` rejects duplicate
#: keys — these suffixes keep both legs addressable and traceable back to the
#: row (``AAPL#12:long`` / ``AAPL#12:short``).
SPREAD_LONG_SUFFIX = ":long"
SPREAD_SHORT_SUFFIX = ":short"

#: Snapshot build triggers (mirrors migrations/018_risk_snapshots.sql).
TRIGGER_SCHEDULED = "SCHEDULED"
#: ON_DEMAND builds persist at most this often (seconds); see step (h).
ON_DEMAND_PERSIST_MIN_INTERVAL_SECONDS = 900
#: Reason shown for model risk LOW on an EMPTY book (nothing to model).
EMPTY_BOOK_NOTE = "no open positions — nothing to model"
TRIGGER_ON_DEMAND = "ON_DEMAND"
TRIGGER_PRE_TRADE = "PRE_TRADE"

#: Model-view keys in the snapshot's ``var`` / ``es`` maps and their API
#: ordering (contract §6: HISTORICAL 0.95, HISTORICAL 0.99, GAUSSIAN 0.95,
#: GAUSSIAN 0.99, HISTORICAL_VOL_SCALED 0.95).
VIEW_ORDER: tuple[tuple[str, float], ...] = (
    ("HISTORICAL", CONFIDENCE_95),
    ("HISTORICAL", CONFIDENCE_99),
    ("GAUSSIAN", CONFIDENCE_95),
    ("GAUSSIAN", CONFIDENCE_99),
    ("HISTORICAL_VOL_SCALED", CONFIDENCE_95),
)

#: ADDITIVE (§6/§12): the √h-scaled DISPLAY rows, appended AFTER every 1-day
#: row so the existing array prefix is byte-identical and any consumer
#: reading ``var[0]`` still gets HISTORICAL 0.95 1-day. Only the HISTORICAL
#: view is scaled: the Gaussian rows would need a drift term over h days
#: (``-mu*h``) that the 1-day number does not carry, and the conditional
#: (vol-scaled) view already has a term structure of its own — scaling it by
#: √h would compose two different volatility models into a number with no
#: interpretation. HISTORICAL √h is the one the audit promised.
DISPLAY_VIEW_ORDER: tuple[tuple[str, float, int], ...] = tuple(
    ("HISTORICAL", CONFIDENCE_95, h) for h in DISPLAY_HORIZONS
)

#: Every (label, confidence, horizon) row the API serves and the persister
#: writes, in contract order: the 1-day grid first, then the display rows.
ALL_VIEW_ORDER: tuple[tuple[str, float, int], ...] = tuple(
    (label, conf, HORIZON_DAYS) for label, conf in VIEW_ORDER
) + DISPLAY_VIEW_ORDER

#: The views ``dispersion`` compares (contract §6 / spec §39): the 95% 1-day
#: VaR views only — comparing a 95% number with a 99% one would measure the
#: confidence grid, not model disagreement.
DISPERSION_VIEW_KEYS: tuple[str, ...] = (
    "HISTORICAL:0.95:1",
    "GAUSSIAN:0.95:1",
    "HISTORICAL_VOL_SCALED:0.95:1",
)

#: §40 (compliance §3 Tier C): the key the WORST STRESS LOSS enters the
#: dispersion set under. It is a PSEUDO-view — no ``var_views`` entry
#: produces it — built by :func:`_stress_dispersion_view` from the stress
#: result's absolute worst loss.
#:
#: WHY A STRESS ROW BELONGS IN A "MODEL DISPERSION" NUMBER. Spec §40's own
#: worked example spans exactly this: its widest view ($5,100) is a stress
#: number and its narrowest ($1,200) a statistical one, giving the 4.25
#: ratio the section quotes. Excluding stress therefore made the reported
#: disagreement UNDERSTATE the true spread on a stress-dominated book —
#: precisely the book where a wide spread matters most. Dispersion now
#: spans the STATISTICAL and STRESS families rather than the statistical
#: one alone, and ``min_model`` / ``max_model`` name which family won.
#:
#: The comparison stays honest because both sides are the same quantity in
#: the same units: a 1-day USD LOSS on today's book, positive = money lost.
#: They are not the same PROBABILITY — a 95% tail average and a scenario
#: reprice answer different questions — which is why this widens the ratio
#: on purpose rather than by accident.
DISPERSION_STRESS_KEY = "STRESS:WORST:1"

#: Views ``model_risk_state`` treats as CORE (their absence is a trigger) and
#: as Gaussian (a LOW gaussian_trust only matters when one is live).
CORE_VIEW_NAMES: tuple[str, ...] = ("historical_var_95", "historical_es_95")
GAUSSIAN_VIEW_NAMES: tuple[str, ...] = ("gaussian_var_95", "gaussian_es_95")

# --- Telemetry (plan §41; contract §6 "Observability") ---------------------

RISK_SNAPSHOT_AGE_SECONDS = REGISTRY.gauge(
    "risk_snapshot_age_seconds",
    "Age in seconds of the newest SCHEDULED statistical risk snapshot build "
    "(risk spec §55; Phase B contract §6). Only SCHEDULED builds count: "
    "on-demand page loads and pre-trade builds must not mask a dead "
    "scheduled writer, which is exactly what this gauge exists to reveal.",
)
RISK_MODEL_LATENCY_SECONDS = REGISTRY.histogram(
    "risk_model_latency_seconds",
    "Wall time of one stage of a statistical risk snapshot build, in SECONDS "
    "(risk spec §41; Phase B contract §6).",
    ("stage",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
RISK_SNAPSHOT_BUILDS_TOTAL = REGISTRY.counter(
    "risk_snapshot_builds_total",
    "Statistical risk snapshot builds completed, by trigger "
    "(SCHEDULED | ON_DEMAND | PRE_TRADE).",
    ("trigger",),
)
RISK_SNAPSHOT_FAILURES_TOTAL = REGISTRY.counter(
    "risk_snapshot_failures_total",
    "Statistical risk snapshot builds that raised before producing a "
    "snapshot (the caller degrades to honest nulls; no Tier 0 effect).",
)


def _set_model_health_gauge(model_health: Mapping[str, ModelHealth]) -> None:
    """§65 ``model_health_state`` gauge, deferred to ``risk_validation``.

    The instrument is DEFINED there (with the three §65 counters, so all four
    read as one family in the exposition) and merely SET here. The import is
    function-local for the same reason every other ``risk_validation`` import
    in this module is: `risk_snapshot` is the module the gateway wires first,
    and keeping the edge lazy means neither module can ever become the
    other's import-time prerequisite.
    """
    from .risk_validation import set_model_health_gauge

    set_model_health_gauge(model_health)


def _inc_garch_fit_failures(*, site: str, health: str) -> None:
    """§65 GARCH fit-failure counter; same deferral as above."""
    from .risk_validation import GARCH_FIT_FAILURES_TOTAL as counter

    counter.inc(site=site, health=health)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RiskSnapshotBuild:
    """One completed build: the typed snapshot plus its serialised views.

    - ``snapshot``: the typed :class:`PortfolioRiskSnapshot` (spec §45);
    - ``row_id``: ``risk_snapshots.id`` when persisted, else ``None``
      (the row is flushed, not committed — the CALLER commits);
    - ``api``: the contract §6 ``"statistical"`` block, ready to serve;
    - ``drawdown_api``: the contract §6 ``"drawdown"`` block;
    - ``positions_excluded``: ``[{"key", "reason"}]`` — every open position
      left out of the book P&L, named with why (honest gap, never zeros);
    - ``latency_seconds``: wall time of the whole build.

    ADDITIVE (Phase C contract §7.5) — the INPUTS the build already assembled,
    exposed so a pre-trade caller can compare a candidate against this exact
    book instead of rebuilding (and silently disagreeing with) it:

    - ``book``: the :class:`BookPnl` the statistics were measured on, or
      ``None`` when there was no priceable position;
    - ``returns``: the aligned SIMPLE :class:`ReturnMatrix` the book was
      priced on (``None`` when no ticker had two stored bars) — the same
      dates as ``book``, which is what :func:`pretrade.proposed_book`
      requires;
    - ``positions``: the :class:`PositionRiskInput` rows that entered the
      book (excluded positions are NOT here — they are in
      ``positions_excluded``);
    - ``capital_weights``: ``key -> |market value| / NAV`` (``None`` per key
      when the value or NAV is unknown);
    - ``nav`` / ``cash``: the NAV and account cash this build measured, so a
      caller's percent-of-NAV numbers use the same denominator;
    - ``correlation_state``: the §19 :class:`CorrelationState` of the book's
      tickers on LOG returns, or ``None`` with fewer than two tickers.

    These are references to the very objects the statistics used — read
    them, never mutate them.
    """

    snapshot: PortfolioRiskSnapshot
    row_id: int | None
    api: dict
    drawdown_api: dict
    positions_excluded: list[dict]
    latency_seconds: float
    book: Any | None = None
    returns: Any | None = None
    positions: tuple[PositionRiskInput, ...] = ()
    capital_weights: Mapping[str, float | None] = field(default_factory=dict)
    nav: float = 0.0
    cash: float | None = None
    correlation_state: Any | None = None
    # ADDITIVE (Phase D contract §8.5): the book as revaluation legs and the
    # stress run over it, so a pre-trade caller adds its candidate's legs to
    # THIS book instead of rebuilding one that could disagree.
    stock_legs: tuple[Any, ...] = ()
    option_legs: tuple[Any, ...] = ()
    stress: Any | None = None
    #: The catalogue this build ran — the SAME scenarios a pre-trade cap
    #: search must re-evaluate at each candidate quantity (a cap measured
    #: against a different catalogue would not be the same limit).
    scenarios: tuple[Any, ...] = ()


@dataclass
class _Stage:
    """Per-stage latency recorder (histogram label ``stage``)."""

    started: float = field(default_factory=time.perf_counter)

    def observe(self, stage: str) -> float:
        elapsed = time.perf_counter() - self.started
        RISK_MODEL_LATENCY_SECONDS.observe(elapsed, stage=stage)
        self.started = time.perf_counter()
        return elapsed


# ---------------------------------------------------------------------------
# Safe model invocation (contract §6: never 500 the view)
# ---------------------------------------------------------------------------


def _failed_result(name: str, exc: Exception, *, confidence: float | None = None) -> ModelResult:
    """A ``FAILED`` :class:`ModelResult` carrying the exception text verbatim.

    An estimator that raised is an UNKNOWN, not a zero (spec §41): value
    None, health FAILED, reason = the real error. The snapshot's
    ``overall_health`` and ``model_risk_state`` both see it.
    """
    return ModelResult(
        value=None,
        health=ModelHealth.FAILED,
        reason=f"{type(exc).__name__}: {exc}",
        sample_size=0,
        meta=ModelMeta(
            model_name=name,
            model_version="0.0.0",
            params={"error": f"{type(exc).__name__}: {exc}"},
            confidence=confidence,
            horizon_days=HORIZON_DAYS,
        ),
    )


def _safe(name: str, fn: Callable[[], Any], *, confidence: float | None = None) -> Any:
    """Call ``fn`` and turn any exception into a FAILED result / ``None``.

    Model modules raise only on MALFORMED input (contract §1) — a bug or an
    impossible book — and that must never take down a read view. The
    exception text is preserved so the failure is diagnosable from the API
    payload, and logged once with a traceback.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — deliberate: SHADOW must not 500
        logger.exception("risk_model_failed", extra={"extra_fields": {"model": name}})
        return _failed_result(name, exc, confidence=confidence)


# ---------------------------------------------------------------------------
# Serialisation helpers (contract §6 shapes)
# ---------------------------------------------------------------------------


def _pct_nav(value: float | None, nav: float) -> float | None:
    """``value / NAV`` as a FRACTION, or ``None`` when NAV ≤ 0.

    A ratio against a zero or negative NAV is not a small number, it is a
    meaningless one — honest null rather than a fabricated percentage.
    """
    if value is None or nav is None or nav <= 0 or not math.isfinite(nav):
        return None
    return value / nav


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _jsonable(value: Any) -> Any:
    """Plain-JSON form of a params / diagnostics value (dates -> ISO)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tier_of(meta: ModelMeta) -> str | None:
    """The §5 tier of the model behind ``meta``, as a plain string.

    Falls back to the NAME lookup when the meta itself carries no tier: a
    conditional view is stamped ``garch_<name>`` by this builder and is not
    a registry entry, so its own meta is copied from the historical
    estimator underneath it. The name lookup is what knows the filter in
    front of it changed the family (spec §5 TIER_2).
    """
    tier = meta.tier
    if tier is None:
        tier = tier_for_model_name(meta.model_name)
    return str(tier) if tier is not None else None


def _meta_params(meta: ModelMeta) -> dict:
    """The FULL §44 provenance of one model, as a plain dict for JSON columns.

    Everything needed to reproduce the number without a join: the
    estimator's own params plus return type / frequency / lookback / data
    source / distribution.
    """
    return {
        **{str(k): _jsonable(v) for k, v in meta.params.items()},
        "return_type": meta.return_type,
        "frequency": meta.frequency,
        "lookback": meta.lookback,
        "data_source": meta.data_source,
        "distribution": meta.distribution,
        "as_of": _iso(meta.as_of),
        # §5 (ADDITIVE): the model's tier, persisted with the number so a
        # replayed `risk_metrics` row can answer "what family produced
        # this?" without a join. `None` for a meta built outside the
        # taxonomy — never a guessed TIER_0.
        "tier": _tier_of(meta),
    }


def _scaling(result: ModelResult) -> str | None:
    """The ``scaling`` label for the API row: ``None`` for a 1-day number.

    The library labels an unscaled number ``"NONE"``; contract §6 renders
    that as JSON ``null`` (nothing was scaled) and keeps ``"SQRT_TIME"``
    verbatim when it was.
    """
    label = result.diagnostics.get("scaling")
    return None if label in (None, "NONE") else str(label)


def _metric_row_payload(
    result: ModelResult,
    model_label: str,
    confidence: float,
    horizon_days: int = HORIZON_DAYS,
) -> dict:
    """One VaR/ES row of the contract §6 ``var`` / ``es`` arrays.

    ``mode`` is SHADOW for the 1-day grid and RESEARCH for a √h-scaled
    multi-day display row (§6/§12): a scaled number is a display, not even a
    shadow gate, and the row says so rather than leaving the reader to infer
    it from ``scaling``.
    """
    return {
        "model": model_label,
        "model_name": result.meta.model_name,
        "model_version": result.meta.model_version,
        "distribution": result.meta.distribution,
        # ADDITIVE (§5): the model's tier, beside the model that produced
        # the number. Display/provenance only — nothing reads it to decide.
        "tier": _tier_of(result.meta),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "value_usd": result.value,
        "pct_nav": None,  # filled by the caller (needs NAV)
        "health": str(result.health),
        "reason": result.reason,
        "sample_size": result.sample_size,
        "tail_size": result.diagnostics.get("tail_size"),
        "scaling": _scaling(result),
        "mode": MODE_SHADOW if horizon_days == HORIZON_DAYS else MODE_RESEARCH,
    }


def _diversification_api(result: ModelResult | None) -> dict:
    """The §34 ``statistical.diversification_ratio`` block (always an object).

    ``{value, health, reason, n_obs}`` — an honest null inside a present
    block, never a missing key: "we could not measure this, here is why" is
    the answer, not silence. ``value`` is a pure ratio (NOT a ``*_pct``
    fraction and NOT USD), ``>= 1`` for an imperfectly correlated book.
    """
    if result is None:
        return {
            "value": None,
            "health": str(ModelHealth.UNAVAILABLE),
            "reason": "the diversification ratio was not computed",
            "n_obs": 0,
            "model_name": DIVERSIFICATION_MODEL_NAME,
            "model_version": DIVERSIFICATION_MODEL_VERSION,
            "estimator": DIVERSIFICATION_ESTIMATOR,
            "mode": MODE_SHADOW,
        }
    return {
        "value": result.value,
        "health": str(result.health),
        "reason": result.reason,
        "n_obs": result.sample_size,
        "model_name": result.meta.model_name,
        "model_version": result.meta.model_version,
        "estimator": DIVERSIFICATION_ESTIMATOR,
        "mode": MODE_SHADOW,
    }


def _factor_api(result: Any | None, *, reason: str | None = None) -> dict:
    """The §11 ``statistical.factor`` block (always an object).

    ``asdict`` of the :class:`FactorRiskResult` with ``positions`` flattened
    into JSON objects and ``health`` stringified — the same treatment
    ``correlation_state`` gets, and for the same reason: a tuple of frozen
    dataclasses is not a shape a client can rely on.

    ``result is None`` means the diagnostic was not COMPUTED at all (no SPY
    series, no book, or the estimator raised). That is served as a present
    block carrying an honest null and the real ``reason`` — never a missing
    key and never a fabricated 0.0, which would read as "the book has no
    market exposure" when the truth is "we did not measure it".

    RESEARCH (spec §70): nothing here decides anything.
    """
    if result is None:
        return {
            "portfolio_beta": None,
            "explained_variance_share": None,
            "idiosyncratic_share": None,
            "positions": [],
            "factor": DEFAULT_FACTOR,
            "n": 0,
            "health": str(ModelHealth.UNAVAILABLE),
            "reason": reason or "the factor diagnostic was not computed",
            "model_name": FACTOR_MODEL_NAME,
            "model_version": FACTOR_MODEL_VERSION,
            "mode": MODE_RESEARCH,
        }
    return {
        "portfolio_beta": result.portfolio_beta,
        "explained_variance_share": result.explained_variance_share,
        "idiosyncratic_share": result.idiosyncratic_share,
        "positions": [
            {
                "label": p.label,
                "beta": p.beta,
                "r2": p.r2,
                "n": p.n,
                "health": str(p.health),
                "reason": p.reason,
            }
            for p in result.positions
        ],
        "factor": result.factor,
        "n": result.n,
        "health": str(result.health),
        "reason": result.reason,
        "model_name": result.meta.model_name,
        "model_version": result.meta.model_version,
        "mode": MODE_RESEARCH,
    }


def _conditional_horizon_sigmas(
    fit: Any | None, *, source: str, reason: str
) -> dict:
    """The §12 ``statistical.conditional_horizon_sigmas`` block.

    ``{h5_usd, h10_usd, source, reason}``. On the GARCH branch the sigmas are
    ``sqrt(Σ_{k=1..h} sigma²_{t+k})`` from
    :func:`garch_forecast_variance` — the closed-form TERM STRUCTURE, which
    is why ``source`` reads ``GARCH_TERM_STRUCTURE`` and not ``SQRT_TIME``:
    GARCH mean-reverts toward the unconditional variance, so its h-day sigma
    is generally NOT ``sigma_1 x sqrt(h)`` and the two must never be confused.

    Honest nulls with the real reason when GARCH is not the live conditional
    source or the fit carries no parameters. EWMA has no term structure to
    report (a flat-variance filter would give exactly √h, which would be a
    restatement of the display rows, not new information) — that is stated
    rather than silently filled in.
    """
    block: dict[str, Any] = {
        f"h{h}_usd": None for h in DISPLAY_HORIZONS
    }
    if fit is None or source != garch_models.SOURCE_GARCH:
        block["source"] = None
        block["reason"] = (
            f"conditional source is {source or 'unknown'}, not GARCH — an EWMA "
            "filter carries no variance term structure. " + reason
        ).strip()
        return block
    try:
        horizon = max(DISPLAY_HORIZONS)
        variances = garch_models.garch_forecast_variance(fit, horizon)
    except Exception as exc:  # a parameterless fit: honest null, never a guess
        block["source"] = None
        block["reason"] = f"no GARCH forecast: {type(exc).__name__}: {exc}"
        return block
    for h in DISPLAY_HORIZONS:
        total = math.fsum(variances[:h])
        block[f"h{h}_usd"] = math.sqrt(total) if total > 0.0 else None
    block["source"] = SOURCE_GARCH_TERM_STRUCTURE
    block["reason"] = None
    return block


def _stress_dispersion_view(stress_result, *, as_of) -> ModelResult | None:
    """The §40 stress PSEUDO-VIEW for the dispersion set, or ``None``.

    Wraps the stress run's absolute worst LOSS in a :class:`ModelResult` so
    the ensemble — which compares ``ModelResult`` values and nothing else —
    can weigh it against the VaR views without learning what a scenario is.

    ``None`` (the view is simply absent, never a zero) when:

    - there is no stress result at all (the layer raised, or no catalogue);
    - no row produced a number (``worst_loss_usd is None``) — an
      UNAVAILABLE catalogue is a gap, and a gap is not a $0 loss;
    - the worst loss is ``<= 0``, i.e. every scenario was a GAIN. The
      ensemble ignores non-positive values anyway (a ratio through zero is
      meaningless), so emitting one would only inflate ``n_excluded`` with
      a view that was never comparable.

    Health is the stress run's OWN health, so a DEGRADED stress run makes
    the dispersion result DEGRADED rather than silently lending it the
    confidence of the statistical views beside it.
    """
    if stress_result is None:
        return None
    worst_loss = stress_result.worst_loss_usd
    if worst_loss is None or worst_loss <= 0.0:
        return None
    health = ModelHealth(stress_result.health)
    if health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED):
        return None
    meta = ModelMeta(
        model_name=stress_models.MODEL_NAME,
        model_version=stress_models.MODEL_VERSION,
        params={
            "catalogue_version": stress_result.catalogue_version,
            "scenario": stress_result.worst.name if stress_result.worst else None,
            "estimator": "worst absolute scenario loss over the catalogue",
        },
        as_of=as_of,
        horizon_days=HORIZON_DAYS,
        # NOT a confidence level: a scenario reprice carries no tail
        # probability, and stamping 0.95 on it would let a reader compare it
        # with the VaR views as though they shared a quantile.
        confidence=None,
        tier=stress_models.MODEL_TIER,
    )
    n = len(stress_result.rows)
    diagnostics = {"n_scenarios": n, "scenario": meta.params.get("scenario")}
    if health is ModelHealth.ACTIVE:
        return active(meta, worst_loss, n, diagnostics=diagnostics)
    return degraded(
        meta,
        stress_result.reason or "stress catalogue degraded",
        worst_loss,
        n,
        diagnostics=diagnostics,
    )


def _es99_noise_warning(result, *, min_tail: int = ES99_NOISY_TAIL_MIN):
    """DEGRADE a 99 % contribution result whose tail is too thin (§10).

    The audit anticipated exactly this number and called it "noisy at
    99%". Suppressing it would lose information; serving it bare would
    lend it the confidence of the 95 % block beside it. So it ships with a
    health that SAYS SO: ``tail_size < min_tail`` ⇒ DEGRADED, carrying the
    real ``k`` and ``n`` in the reason.

    Never upgrades: an already-DEGRADED or UNAVAILABLE result keeps its
    worse health (``combine_health`` semantics), and its existing reason is
    preserved with the noise note appended. ``None`` in, ``None`` out.
    """
    if result is None or result.tail_size is None:
        return result
    if result.tail_size >= min_tail:
        return result
    note = (
        f"noisy at 99%: only k={result.tail_size} tail observations "
        f"(n={result.sample_size}) < {min_tail} — the per-position split of "
        "an average over so few days is dominated by which days landed in "
        "the tail; read it as an indication, not a measurement"
    )
    health = combine_health(ModelHealth(result.health), ModelHealth.DEGRADED)
    reason = f"{result.reason}; {note}" if result.reason else note
    return dataclasses.replace(result, health=health, reason=reason)


def _contribution_block(
    result, nav: float, capital_weights: Mapping[str, float],
    meta_by_key: Mapping[str, tuple[str, str]], *, with_confidence: bool,
) -> dict | None:
    """The contract §6 ``contributions.es`` / ``.vol`` block, or ``None``.

    ``None`` only when the estimator itself was never run (no result at
    all); an UNAVAILABLE result is still reported — with its health and
    reason — because "we could not measure this, here is why" is the honest
    answer, not silence.
    """
    if result is None:
        return None
    rows = []
    for row in result.per_position:
        ticker, instrument = meta_by_key.get(row.key, ("", ""))
        rows.append(
            {
                "key": row.key,
                "ticker": ticker,
                "instrument": instrument,
                "contribution_usd": row.contribution,
                "share": row.share,
                "capital_weight": capital_weights.get(row.key),
            }
        )
    block: dict = {
        "total_usd": result.total,
        "pct_nav": _pct_nav(result.total, nav),
        "health": str(result.health),
        "reason": result.reason,
        "rows": rows,
    }
    if with_confidence:
        return {"confidence": result.confidence, **block}
    return block


# ---------------------------------------------------------------------------
# Correlation regime (spec §19; Phase C contract §7.4)
# ---------------------------------------------------------------------------


def log_matrix_from_bars(
    bars_by_ticker: Mapping[str, Sequence[tuple[date, float]]]
) -> Any | None:
    """Aligned **LOG** :class:`ReturnMatrix` over the given stored bars.

    The correlation sibling of the SIMPLE matrix the book P&L is priced on:
    the SAME bars, the SAME inner-join alignment, the log convention every
    correlation on this platform uses (contract §1). A ticker with fewer
    than two bars, or whose returns raise, is dropped — an honest gap the
    regime reports as a smaller ``n_pairs``, never a fabricated column.

    ``None`` when nothing could be built. Shared by the snapshot builder and
    the pre-trade path (plan §21: one construction, not two) so the regime a
    preview reports is the regime the risk view reports.
    """
    series = []
    for ticker in sorted(bars_by_ticker):
        bars = bars_by_ticker.get(ticker) or []
        if len(bars) < 2:
            continue
        built = _safe(
            f"log_returns:{ticker}",
            lambda t=ticker, b=bars: returns_from_closes(
                t, b, return_type=RETURN_TYPE_LOG
            ),
        )
        if isinstance(built, ModelResult):  # the safe wrapper caught a raise
            continue
        series.append(built)
    return align(series) if series else None


def correlation_state_for(
    bars_by_ticker: Mapping[str, Sequence[tuple[date, float]]]
) -> Any | None:
    """The §19 :class:`CorrelationState` of these tickers, or ``None``.

    ``None`` — not an UNAVAILABLE state — when fewer than two tickers have
    usable bars: a correlation needs a pair, and a one-name book has no
    regime to report (the API serialises the ``None`` as a null, which is
    the honest answer, rather than a state object full of nulls).
    """
    matrix = log_matrix_from_bars(bars_by_ticker)
    if matrix is None or len(matrix.tickers) < 2:
        return None
    return correlation_regime(matrix)


def correlation_state_api(state: Any | None) -> dict | None:
    """Serialise a :class:`CorrelationState` for the wire (contract §6).

    ``asdict`` of the frozen dataclass with ``worst_pairs`` flattened into
    JSON arrays (a tuple of tuples is not a JSON shape a client can rely
    on). ``None`` passes straight through as a null.
    """
    if state is None:
        return None
    out = dataclasses.asdict(state)
    out["worst_pairs"] = [[a, b, rho] for a, b, rho in state.worst_pairs]
    # ADDITIVE (§18, compliance §3 row 18): the rolling SPEARMAN average over
    # the SAME short window as ``current_avg``. Read EXPLICITLY rather than
    # from ``asdict``: it is a non-field attribute on the frozen dataclass
    # (see ``CorrelationState``), so ``asdict`` does not see it and the key
    # would silently never appear on the wire. ``None`` is honest — a rank
    # correlation needs a pair and a window, exactly like the Pearson twin.
    out["current_avg_spearman"] = getattr(state, "current_avg_spearman", None)
    return out


# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------


def option_leg_fields_by_key(
    option_legs: Sequence[OptionLeg],
) -> dict[str, dict[str, float | str]]:
    """The design §10.1 leg fields, keyed by POSITION key, from the stress
    legs the builder already resolved (design §10.3).

    THE SAME CHAIN RESOLUTION, READ ONCE. ``stress_legs_from_book`` has
    already resolved every option position against today's chain through
    ``find_option_contract`` / ``find_spread_short_leg``; this function only
    re-shapes those legs. It performs NO chain call of its own, so the P&L
    series, the greeks panel and the stress rows can never be anchored to
    different contracts.

    ONE LEG PER POSITION KEY. A spread is one position row and TWO
    revaluation legs (``…:long`` / ``…:short``), while the book P&L carries
    it as ONE ``PositionRiskInput`` under the bare key — that key is what
    ``contributions``, ``capital_weights``, ``meta_by_key`` and every
    incremental/marginal ES consumer look the row up by, so splitting it
    here would break those surfaces. A suffixed leg is therefore SKIPPED and
    its position stays DELTA_LINEAR on its NET delta, labelled as such (see
    ``_position_inputs``). Honest and conservative: a net-delta spread row
    reports the estimator that actually priced it.
    """
    out: dict[str, dict[str, float | str]] = {}
    for leg in option_legs:
        if leg.key.endswith(SPREAD_LONG_SUFFIX) or leg.key.endswith(
            SPREAD_SHORT_SUFFIX
        ):
            continue  # a spread's two legs are one un-splittable book row
        if leg.iv0 is None or leg.t_years <= 0.0:
            continue  # no vol / expired ⇒ nothing to revalue; stays linear
        out[leg.key] = {
            "strike": leg.strike,
            "right": leg.right,
            "t_years": leg.t_years,
            "iv0": leg.iv0,
            "mark0": leg.mark0,
        }
    return out


def _position_inputs(
    pairs: Sequence[tuple[Any, float | None]],
    greeks_rows: Sequence[dict],
    leg_fields: Mapping[str, Mapping[str, float | str]] | None = None,
) -> tuple[list[PositionRiskInput], list[dict]]:
    """Build one :class:`PositionRiskInput` per OPEN position, or exclude it.

    The signed per-share delta is recovered from the shared
    ``portfolio_greeks_read`` row as ``delta_shares / (quantity ×
    multiplier)`` — the module docstring explains why that is the one rule
    for every instrument. A row with ``data_ok`` False, a null
    ``equivalent_shares``, a zero size, or a non-positive spot cannot be
    priced and is EXCLUDED with the row's own ``note`` as the reason (the
    server-generated string, rendered verbatim by the UI).

    ``leg_fields`` (design §10.3, ADDITIVE) maps a position key to the five
    option leg fields from :func:`option_leg_fields_by_key` — the SAME chain
    resolution the stress legs and the greeks panel used, never a second
    chain call. A key present there gets its ``PositionRiskInput`` built
    with those fields and is priced FULL_REVAL_CONST_IV; every other row
    (stock, a spread's net row, an option whose chain gave no IV) is built
    exactly as before and stays DELTA_LINEAR. Omitting the argument
    reproduces the pre-batch behaviour byte for byte.
    """
    legs: Mapping[str, Mapping[str, float | str]] = leg_fields or {}
    inputs: list[PositionRiskInput] = []
    excluded: list[dict] = []
    for (pos, price), row in zip(pairs, greeks_rows):
        key = f"{pos.ticker}#{pos.id}"
        multiplier = pos.multiplier or 1
        scale = pos.quantity * multiplier
        if not row.get("data_ok"):
            excluded.append(
                {
                    "key": key,
                    "reason": row.get("note")
                    or "no delta (position greeks are unavailable)",
                }
            )
            continue
        delta_shares = row.get("equivalent_shares")
        if delta_shares is None or scale == 0:
            excluded.append(
                {
                    "key": key,
                    "reason": (
                        f"no delta (equivalent_shares={delta_shares!r}, "
                        f"quantity={pos.quantity}, multiplier={multiplier})"
                    ),
                }
            )
            continue
        if price is None or not math.isfinite(price) or price <= 0:
            excluded.append(
                {
                    "key": key,
                    "reason": (
                        f"no spot ({pos.ticker} has no usable stored close: "
                        f"{price!r})"
                    ),
                }
            )
            continue
        leg = legs.get(key) or {}
        inputs.append(
            PositionRiskInput(
                key=key,
                ticker=pos.ticker,
                instrument=pos.instrument,
                quantity=pos.quantity,
                multiplier=multiplier,
                spot=price,
                delta=delta_shares / scale,
                max_loss=pos.max_loss,
                # ADDITIVE (design §10.1). Absent ⇒ every field stays None
                # ⇒ the row is DELTA_LINEAR exactly as before.
                strike=leg.get("strike"),  # type: ignore[arg-type]
                right=leg.get("right"),  # type: ignore[arg-type]
                t_years=leg.get("t_years"),  # type: ignore[arg-type]
                iv0=leg.get("iv0"),  # type: ignore[arg-type]
                mark0=leg.get("mark0"),  # type: ignore[arg-type]
            )
        )
    return inputs, excluded


# ---------------------------------------------------------------------------
# Phase D — stress legs from the open book (design §8.5)
# ---------------------------------------------------------------------------


def _t_years(dte: int | None) -> float:
    """A contract's time to expiry in YEARS from its chain DTE.

    ``max(dte, 0) / 365`` — calendar days (see :data:`DAYS_PER_YEAR`). A
    missing DTE is 0.0, which the pricer reads as expiry: the leg is then
    valued at intrinsic rather than at a guessed tenor (honest, never a
    fabricated 30 days).
    """
    if dte is None:
        return 0.0
    return max(int(dte), 0) / DAYS_PER_YEAR


def stress_legs_from_book(
    pairs: Sequence[tuple[Any, float | None]],
    greeks_rows: Sequence[dict],
) -> tuple[list[StockLeg], list[OptionLeg], list[dict]]:
    """The open book as revaluation legs (design §8.5): stock legs, option
    legs, and the positions that could NOT be turned into a leg.

    THE SAME CHAIN RESOLUTION THE RISK VIEW USES. Option marks and IVs come
    from ``find_option_contract`` / ``find_spread_short_leg`` against the
    chain ``option_chain_or_none`` regenerates — the identical helpers
    ``portfolio_greeks_read`` calls — so a stress reprice can never be
    anchored to a contract the greeks panel disagrees about (plan §21).

    Sign convention (design §8.2: ``quantity`` is SIGNED):

    - LONG_STOCK ``+shares``; SHORT_STOCK ``−shares``;
    - LONG_CALL / LONG_PUT ``+contracts``;
    - income legs (covered call, cash-secured put) ``−contracts`` — the
      platform stores them with a POSITIVE quantity and the short-ness in the
      instrument, exactly as ``portfolio_greeks_read`` negates their greeks;
    - a spread is ONE row and TWO legs: ``+contracts`` at ``opt_strike`` and
      ``−contracts`` at ``short_strike``, keyed ``…:long`` / ``…:short``.

    Anchoring (design §8.2): ``mark0`` is the contract MID from the chain and
    ``iv0`` the PROVIDER's IV (or ``None``). ``reval`` holds the basis
    ``mark0 − model(iv0)`` constant across scenarios, so the zero scenario
    prices to exactly 0.0 P&L even when the provider's mid and the
    Black-Scholes model disagree — the scenario measures the MOVE, never the
    model's opinion of today's price.

    Honest gaps, never guesses:

    - no stored close (spot unknown) ⇒ excluded with the reason;
    - a contract missing from today's chain (expired off it) ⇒ excluded;
    - a contract quoted but with NO provider IV ⇒ the leg is STILL built with
      ``iv0=None`` and the chain's ``delta`` as ``delta0``: ``scenario_pnl``
      then prices it DELTA_LINEAR and LABELS it, which is a degraded number
      the coverage counts expose — strictly better than dropping the leg and
      understating the book's loss silently;
    - a spread with only one quoted leg ⇒ the whole row is excluded (half a
      spread is not a position).

    Returns ``(stock_legs, option_legs, excluded)``; ``excluded`` rows are
    ``{"key", "reason"}`` in the same shape ``positions_excluded`` uses.
    """
    from .routers.options import option_chain_or_none  # local: router cycle
    from .routers.portfolio import (
        find_option_contract,
        find_spread_short_leg,
        is_income_position,
        is_option_position,
        is_short_stock_position,
        is_spread_position,
    )

    stock_legs: list[StockLeg] = []
    option_legs: list[OptionLeg] = []
    excluded: list[dict] = []
    chains: dict[str, list] = {}  # one chain build per ticker, like the greeks read

    def _chain(ticker: str, spot: float) -> list:
        if ticker not in chains:
            chains[ticker] = option_chain_or_none(ticker, spot) or []
        return chains[ticker]

    def _option_leg(
        key: str, pos, contract, *, quantity: int, spot: float
    ) -> OptionLeg:
        return OptionLeg(
            key=key,
            ticker=pos.ticker,
            right=contract.right,
            strike=contract.strike,
            t_years=_t_years(contract.dte),
            quantity=quantity,
            spot0=spot,
            mark0=contract.mid,
            iv0=contract.iv,
            delta0=contract.delta,
            multiplier=pos.multiplier or 100,
            r=STRESS_RATE,
            q=STRESS_DIVIDEND_YIELD,
        )

    for (pos, price), _row in zip(pairs, greeks_rows):
        key = f"{pos.ticker}#{pos.id}"
        if price is None or not math.isfinite(price) or price <= 0:
            excluded.append(
                {
                    "key": key,
                    "reason": (
                        f"no spot ({pos.ticker} has no usable stored close: "
                        f"{price!r}) — the leg cannot be anchored"
                    ),
                }
            )
            continue
        try:
            if is_spread_position(pos):
                chain = _chain(pos.ticker, price)
                long_leg = find_option_contract(chain, pos)
                short_leg = find_spread_short_leg(chain, pos)
                if long_leg is None or short_leg is None:
                    excluded.append(
                        {
                            "key": key,
                            "reason": (
                                f"spread leg(s) missing from today's chain (long "
                                f"{pos.opt_strike} / short {pos.short_strike} exp "
                                f"{pos.opt_expiry}) — no stress leg"
                            ),
                        }
                    )
                    continue
                option_legs.append(
                    _option_leg(
                        key + SPREAD_LONG_SUFFIX,
                        pos,
                        long_leg,
                        quantity=pos.quantity,
                        spot=price,
                    )
                )
                option_legs.append(
                    _option_leg(
                        key + SPREAD_SHORT_SUFFIX,
                        pos,
                        short_leg,
                        quantity=-pos.quantity,
                        spot=price,
                    )
                )
            elif is_income_position(pos):
                chain = _chain(pos.ticker, price)
                contract = find_option_contract(chain, pos)
                if contract is None:
                    excluded.append(
                        {
                            "key": key,
                            "reason": (
                                f"short {pos.opt_right} {pos.opt_strike} exp "
                                f"{pos.opt_expiry} missing from today's chain — "
                                "no stress leg"
                            ),
                        }
                    )
                    continue
                option_legs.append(
                    _option_leg(
                        key, pos, contract, quantity=-pos.quantity, spot=price
                    )
                )
            elif is_option_position(pos):
                chain = _chain(pos.ticker, price)
                contract = find_option_contract(chain, pos)
                if contract is None:
                    excluded.append(
                        {
                            "key": key,
                            "reason": (
                                f"contract {pos.opt_right} {pos.opt_strike} exp "
                                f"{pos.opt_expiry} missing from today's chain "
                                "(e.g. expired) — no stress leg"
                            ),
                        }
                    )
                    continue
                option_legs.append(
                    _option_leg(
                        key, pos, contract, quantity=pos.quantity, spot=price
                    )
                )
            elif is_short_stock_position(pos):
                stock_legs.append(
                    StockLeg(
                        key=key,
                        ticker=pos.ticker,
                        quantity=-pos.quantity,
                        spot0=price,
                    )
                )
            else:
                stock_legs.append(
                    StockLeg(
                        key=key,
                        ticker=pos.ticker,
                        quantity=pos.quantity,
                        spot0=price,
                    )
                )
        except ValueError as exc:
            # A malformed leg (a non-positive strike, a NaN mid) is a DATA
            # fact, not a crash: name it and move on with the rest of the
            # book, exactly as a missing contract is named.
            excluded.append({"key": key, "reason": f"unpriceable leg: {exc}"})
    return stock_legs, option_legs, excluded


def _scenario_row_api(row) -> dict:
    """One :class:`ScenarioResult` on the wire (design §8.5).

    ``pnl_usd`` is GAIN-POSITIVE (a stress LOSS is negative); ``loss_usd`` /
    ``loss_pct_nav`` restate it in the VaR/ES sign (positive = money lost) so
    the UI never has to negate a server number itself. Honest nulls: an
    UNAVAILABLE row carries ``None`` on every number and its ``reason``.
    """
    return {
        "name": row.name,
        "kind": row.kind,
        "validated": row.validated,
        "pnl_usd": row.pnl_usd,
        "pnl_pct_nav": row.pnl_pct_nav,
        "loss_usd": row.loss_usd,
        "loss_pct_nav": (None if row.pnl_pct_nav is None else -row.pnl_pct_nav),
        "method_coverage": dict(row.method_coverage),
        "health": str(row.health),
        "reason": row.reason,
        "params": _jsonable(row.params),
    }


def _stress_api(
    result,
    *,
    n_stock_legs: int,
    n_option_legs: int,
    positions_excluded: Sequence[dict] = (),
) -> dict:
    """The contract §6/§8.5 ``statistical.stress`` block.

    ALWAYS an object (never null): an empty book has a real, measurable
    stress P&L of exactly 0.0 and the catalogue rows say so. The honest nulls
    live INSIDE the rows — a historical window outside the stored history is
    an UNAVAILABLE row with its reason, never a fabricated zero.

    ``per_position`` is the WORST row's per-leg P&L (spec §52: an option row
    shows its own scenario loss); ``method_coverage`` is that same row's, so
    the headline number's pricing quality travels with it.

    ``positions_excluded`` is the STRESS view's own gap list — a position with
    no stress leg (contract off today's chain, no spot). It is deliberately
    NOT the statistical block's list: a position can be in one view and out of
    the other, and one merged list would misreport both.
    """
    worst = result.worst if result is not None else None
    return {
        "mode": MODE_SHADOW,
        # ADDITIVE (§5): stress is a FIRST-tier model in the spec's own
        # list — a deterministic reprice of today's book, no fitted
        # parameter. Served from the library constant so the wire and the
        # module cannot drift apart.
        "tier": str(stress_models.MODEL_TIER),
        "catalogue_version": (
            result.catalogue_version if result is not None else stress_models.CATALOGUE_VERSION
        ),
        "model_version": (
            result.model_version if result is not None else stress_models.MODEL_VERSION
        ),
        "health": str(result.health) if result is not None else str(ModelHealth.UNAVAILABLE),
        "reason": result.reason if result is not None else "no stress run",
        "n_stock_legs": n_stock_legs,
        "n_option_legs": n_option_legs,
        "method_coverage": (
            dict(worst.method_coverage)
            if worst is not None
            else {REVAL_METHOD_FULL_REVAL: 0, REVAL_METHOD_DELTA_LINEAR: 0}
        ),
        "rows": [_scenario_row_api(r) for r in (result.rows if result is not None else ())],
        "worst": _scenario_row_api(worst) if worst is not None else None,
        "per_position": dict(worst.per_key) if worst is not None else {},
        "positions_excluded": list(positions_excluded),
    }

async def _scheduled_nav_series(session: AsyncSession) -> list[tuple[date, float]]:
    """The persisted SCHEDULED NAV path — LAST row per NY calendar date,
    oldest first (contract §6: live drawdown reads these rows).

    Only SCHEDULED rows: ON_DEMAND and PRE_TRADE builds fire at arbitrary
    moments (a page load, an order preview) and would turn the NAV path into
    a sampling artefact of user activity. Rows with a NULL nav (no account
    at build time) are skipped — an unknown NAV is not a NAV.
    """
    rows = (
        await session.execute(
            select(RiskSnapshotRow)
            .where(RiskSnapshotRow.trigger == TRIGGER_SCHEDULED)
            .order_by(RiskSnapshotRow.as_of)
        )
    ).scalars().all()
    by_day: dict[date, float] = {}
    for row in rows:
        if row.nav is None:
            continue
        as_of = row.as_of
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        by_day[as_of.astimezone(NEW_YORK).date()] = row.nav
    return [(day, by_day[day]) for day in sorted(by_day)]


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


async def build_risk_snapshot(
    session: AsyncSession,
    *,
    trigger: str,
    cash: float | None,
    trading_enabled: bool | None = None,
    persist: bool = True,
) -> RiskSnapshotBuild:
    """Build (and optionally persist) ONE statistical risk snapshot.

    ``trigger`` is SCHEDULED | ON_DEMAND | PRE_TRADE. ``cash`` is the
    account cash the caller already resolved (the broker's LIVE cash, or the
    simulator's ledger — the platform stores no copy of a real account);
    ``None`` means there is no account, which is a data-quality fact, not an
    error: the build still runs and reports honest nulls.

    ``persist=True`` adds the ``risk_snapshots`` / ``risk_metrics`` /
    ``risk_contributions`` rows and FLUSHES them (so ``row_id`` is real) —
    **the caller commits**, keeping a PRE_TRADE build in the same
    transaction as the decision it accompanies.

    ``trading_enabled`` (the §18 kill-switch state) is accepted so callers
    hand over the same book context the Tier 0 snapshot used; Phase B does
    not consume it (SHADOW builds never decide anything) — Phase C's
    hypothetical statistical verdicts will, so the parameter is kept
    optional rather than removed.

    Never raises for a data gap or a misbehaving estimator (both degrade to
    typed honest nulls); a genuine infrastructure failure propagates to the
    caller, which is expected to degrade its own payload.
    """
    from .routers.portfolio import (  # local import: avoids a router import cycle
        open_csp_cash_reserved,
        open_positions_with_prices,
        portfolio_greeks_read,
        position_market_value,
        stored_bars_by_ticker,
    )
    from .deps import market_data_configured

    build_started = time.perf_counter()
    stage = _Stage()
    limits = RiskLimits()
    now = datetime.now(timezone.utc)
    have_market_data = market_data_configured()

    # -- (a) the book: positions, market values, NAV, heat -------------------
    pairs = await open_positions_with_prices(session)
    values = [
        position_market_value(pos, price, market_data=have_market_data)
        for pos, price in pairs
    ]
    nav = (cash or 0.0) + math.fsum(v for v in values if v is not None)
    position_risks = [
        _PositionRiskLite(pos.ticker, value if value is not None else 0.0, pos.max_loss)
        for (pos, _price), value in zip(pairs, values)
    ]
    heat = portfolio_heat(position_risks, nav) if nav > 0 else 0.0
    state = heat_state(heat, limits)
    cash_reserved = open_csp_cash_reserved(pairs)
    gross_exposure = math.fsum(abs(v) for v in values if v is not None)
    stage.observe("book")

    # -- (d0) the book as revaluation legs (Phase D §8.5; design §10.3) -----
    # MOVED EARLIER by compliance batch 2 (design §10.3), not duplicated:
    # the very same `stress_legs_from_book` call that has always fed the
    # stress engine now ALSO feeds the P&L series, so the option marks and
    # IVs behind VaR/ES/RC are byte-identical to the ones behind the stress
    # rows and the greeks panel. Resolving it here rather than at (f2) is
    # what makes "one chain read, one set of contracts" true — a second call
    # could disagree with the first if the chain moved mid-build.
    greeks, greeks_rows = portfolio_greeks_read(pairs)
    legs = _safe("stress_legs", lambda: stress_legs_from_book(pairs, greeks_rows))
    if isinstance(legs, ModelResult):
        # The leg layer raised: stress reports UNAVAILABLE and the P&L
        # series keeps its pre-batch DELTA_LINEAR behaviour. A failure in
        # the NEW path degrades to the OLD numbers — never to a 500 and
        # never to a changed decision (SHADOW).
        stock_legs: list[StockLeg] = []
        option_legs: list[OptionLeg] = []
        stress_excluded: list[dict] = []
    else:
        stock_legs, option_legs, stress_excluded = legs
    leg_fields = _safe(
        "pnl_leg_fields", lambda: option_leg_fields_by_key(option_legs)
    )
    if isinstance(leg_fields, ModelResult):
        leg_fields = {}

    # -- (d) per-position risk inputs (delta from the SHARED greeks read) ----
    positions, positions_excluded = _position_inputs(
        pairs, greeks_rows, leg_fields
    )
    delta_adjusted_exposure = greeks.delta_adjusted_notional
    capital_weights: dict[str, float] = {}
    meta_by_key: dict[str, tuple[str, str]] = {}
    for (pos, _price), value in zip(pairs, values):
        key = f"{pos.ticker}#{pos.id}"
        meta_by_key[key] = (pos.ticker, pos.instrument)
        capital_weights[key] = abs(value) / nav if (value is not None and nav > 0) else None
    stage.observe("greeks")

    # -- (b)/(c) stored bars -> aligned SIMPLE return matrix ------------------
    tickers = sorted({p.ticker for p in positions})
    bars_by_ticker = await stored_bars_by_ticker(
        session, tickers, lookback_bars=DEFAULT_LOOKBACK_BARS
    )
    series = []
    tickers_no_bars: list[str] = []
    for ticker in tickers:
        bars = bars_by_ticker.get(ticker) or []
        if len(bars) < 2:
            tickers_no_bars.append(ticker)
            continue
        built = _safe(
            f"returns:{ticker}",
            lambda t=ticker, b=bars: returns_from_closes(
                t, b, return_type=RETURN_TYPE_SIMPLE
            ),
        )
        if isinstance(built, ModelResult):  # the safe wrapper caught a raise
            tickers_no_bars.append(ticker)
            continue
        series.append(built)
    matrix = align(series) if series else None
    n_obs = matrix.n_obs if matrix is not None else 0
    window_start = matrix.dates[0] if (matrix is not None and matrix.n_obs) else None
    window_end = matrix.dates[-1] if (matrix is not None and matrix.n_obs) else None
    stage.observe("returns")

    # -- (c2) correlation regime (spec §19; Phase C contract §7.4/§7.5) ------
    # A SECOND alignment of the SAME bars on LOG returns: correlation of
    # returns uses the log convention platform-wide (contract §1), while the
    # P&L matrix above must stay SIMPLE (``pnl = exposure × r`` is exact for
    # stock). `correlation_regime` raises on the wrong convention, so the two
    # can never be crossed by accident. UNAVAILABLE below two tickers.
    correlation_state = _safe(
        "correlation_regime", lambda: correlation_state_for(bars_by_ticker)
    )
    if isinstance(correlation_state, ModelResult):  # the wrapper caught a raise
        correlation_state = None
    stage.observe("correlation")

    # -- (e) book P&L (DELTA_LINEAR) -----------------------------------------
    book = None
    if matrix is not None and positions:
        book = _safe("book_pnl_series", lambda: book_pnl_series(positions, matrix))
        if isinstance(book, ModelResult):
            book = None
    pnl_total: list[float] = list(book.total) if book is not None else []
    per_position_pnl: dict[str, list[float]] = dict(book.per_position) if book else {}
    for key in book.keys_excluded if book is not None else ():
        if not any(e["key"] == key for e in positions_excluded):
            positions_excluded.append(
                {"key": key, "reason": "no stored bars for the underlying"}
            )
    tickers_missing = sorted(
        set(tickers_no_bars) | set(book.tickers_missing if book is not None else ())
    )
    stage.observe("pnl")

    # -- (f) the models -------------------------------------------------------
    # Conditional (filtered-HS) views follow the spec §13/§58 fallback: GARCH
    # scaling when the RESEARCH fit is ACTIVE, EWMA scaling otherwise — one
    # decision, named in ``conditional_source`` so the tile can say which
    # forecaster is behind the number (Phase E, contract §9.3/§9.5).
    conditional_source: dict = {
        "source": garch_models.SOURCE_EWMA,
        "reason": "no book P&L series (empty book) — EWMA is the default filter",
    }
    garch_scaled: list[float] | None = None
    #: §12 GARCH term structure of the DISPLAY horizons. Populated only on the
    #: GARCH branch; honest nulls with a reason otherwise (see below).
    garch_fit = None
    garch_fit_health: str | None = None
    if pnl_total:
        try:
            # `garch_scaling` rather than `conditional_scaled_pnl_source`: the
            # two apply the SAME rule (ACTIVE fit ⇒ GARCH, else EWMA), but the
            # scaling object also hands back the FIT, which §12's term
            # structure needs and which the tuple-returning helper discards.
            # Deciding the source here keeps that one decision in one place.
            scaling = garch_models.garch_scaling(pnl_total)
            garch_fit = scaling.fit
            garch_fit_health = str(scaling.fit.health)
            if scaling.fit.health is ModelHealth.ACTIVE:
                garch_scaled = list(scaling.scaled)
                conditional_source = {
                    "source": garch_models.SOURCE_GARCH,
                    "reason": (
                        "GARCH(1,1) ACTIVE: "
                        f"persistence={scaling.fit.persistence:.6f}"
                    ),
                }
            else:
                # §65: a non-ACTIVE fit is a fit FAILURE for observability
                # purposes — the number the operator wants alerted on is "how
                # often did GARCH not produce a usable forecast", not "how
                # often did the optimiser raise".
                _inc_garch_fit_failures(
                    site="snapshot", health=garch_fit_health
                )
                conditional_source = {
                    "source": garch_models.SOURCE_EWMA,
                    "reason": (
                        f"GARCH not ACTIVE (health={scaling.fit.health}): "
                        f"{scaling.fit.reason} — falling back to EWMA"
                    ),
                }
        except Exception as exc:  # the seam degrades, never fails the build
            _inc_garch_fit_failures(site="snapshot", health="RAISED")
            conditional_source = {
                "source": garch_models.SOURCE_EWMA,
                "reason": f"fallback selector raised {type(exc).__name__}: {exc}",
            }

    def _garch_conditional(fn, series, c, h, **kw):
        result = fn(series, c, h, **kw)
        return dataclasses.replace(
            result,
            meta=dataclasses.replace(
                result.meta,
                model_name=f"garch_{result.meta.model_name}",
                distribution=garch_models.DISTRIBUTION_EMPIRICAL_GARCH_SCALED,
                params={**dict(result.meta.params), "conditional_source": "GARCH"},
            ),
        )

    var_views: dict[str, ModelResult] = {}
    es_views: dict[str, ModelResult] = {}
    for label, conf in VIEW_ORDER:
        key = f"{label}:{conf}:{HORIZON_DAYS}"
        if label == "HISTORICAL":
            var_fn, es_fn = historical_var, historical_es
        elif label == "GAUSSIAN":
            var_fn, es_fn = gaussian_var, gaussian_es
        elif garch_scaled is not None:
            var_fn = lambda s, c, h, **kw: _garch_conditional(historical_var, garch_scaled, c, h, **kw)  # noqa: E731
            es_fn = lambda s, c, h, **kw: _garch_conditional(historical_es, garch_scaled, c, h, **kw)  # noqa: E731
        else:
            var_fn, es_fn = conditional_var, conditional_es
        var_views[key] = _safe(
            f"{label.lower()}_var",
            lambda f=var_fn, c=conf: f(pnl_total, c, HORIZON_DAYS, as_of=window_end),
            confidence=conf,
        )
        es_views[key] = _safe(
            f"{label.lower()}_es",
            lambda f=es_fn, c=conf: f(pnl_total, c, HORIZON_DAYS, as_of=window_end),
            confidence=conf,
        )

    # ADDITIVE (§6/§12): the √h-scaled HISTORICAL display rows. The library
    # does the scaling and labels it (`scaling="SQRT_TIME"`); this loop only
    # asks for the horizon. Keys are the SAME `<METHOD>:<conf>:<horizon>`
    # scheme, so `"HISTORICAL:0.95:5"` sits beside `"HISTORICAL:0.95:1"` and
    # nothing about the 1-day entries changes.
    for label, conf, horizon in DISPLAY_VIEW_ORDER:
        key = f"{label}:{conf}:{horizon}"
        var_views[key] = _safe(
            f"{label.lower()}_var_{horizon}d",
            lambda c=conf, h=horizon: historical_var(pnl_total, c, h, as_of=window_end),
            confidence=conf,
        )
        es_views[key] = _safe(
            f"{label.lower()}_es_{horizon}d",
            lambda c=conf, h=horizon: historical_es(pnl_total, c, h, as_of=window_end),
            confidence=conf,
        )

    volatility = _safe(
        "portfolio_volatility",
        lambda: portfolio_volatility(pnl_total, as_of=window_end),
    )
    stage.observe("var_es")

    contributions_es = None
    contributions_vol = None
    contributions_es99 = None
    if per_position_pnl:
        contributions_es = _safe(
            "es_contributions",
            lambda: es_contributions(
                per_position_pnl, CONFIDENCE_95, portfolio_pnl=pnl_total,
                as_of=window_end,
            ),
            confidence=CONFIDENCE_95,
        )
        contributions_vol = _safe(
            "volatility_contributions",
            lambda: volatility_contributions(
                per_position_pnl, portfolio_pnl=pnl_total, as_of=window_end
            ),
        )
        # §10 (compliance §3 Tier C): the SAME Euler decomposition at 99 %.
        # The audit's objection to a 99 % RC was that it is NOISY, not that
        # it is wrong — so it ships WITH the warning rather than absent,
        # which is the honest-null rule applied to precision instead of
        # availability. `_es99_health_note` below turns a thin tail into a
        # DEGRADED health carrying the real k.
        contributions_es99 = _safe(
            "es_contributions_99",
            lambda: es_contributions(
                per_position_pnl, CONFIDENCE_99, portfolio_pnl=pnl_total,
                as_of=window_end,
            ),
            confidence=CONFIDENCE_99,
        )
        # A raise inside a contribution estimator yields a ModelResult, which
        # is not a ContributionResult — report it as "no block" rather than
        # serving a shape the contract does not define.
        if isinstance(contributions_es, ModelResult):
            contributions_es = None
        if isinstance(contributions_vol, ModelResult):
            contributions_vol = None
        if isinstance(contributions_es99, ModelResult):
            contributions_es99 = None
        else:
            contributions_es99 = _es99_noise_warning(contributions_es99)
    stage.observe("contributions")

    distribution = _safe(
        "distribution_diagnostics",
        lambda: distribution_diagnostics(pnl_total, as_of=window_end),
    )
    if isinstance(distribution, ModelResult):
        distribution = None
    # -- (e2) Phase D stress: the catalogue and the run over the legs ------
    # SHADOW (design §8.5). The legs were resolved in (d0) above, from the
    # SAME chain resolution the greeks read used, so a stress reprice and the
    # greeks panel can never be anchored to different contracts. `_safe`
    # wraps the whole layer: a stress failure degrades to
    # `stress_result=None` (the API block then reports UNAVAILABLE with the
    # reason) and never 500s the view or touches a Tier 0 number.
    # The scenario catalogue is derived from the SAME stored bars the return
    # matrix was built on (`bars_by_ticker`): a historical window's shock is
    # the real cumulative return over that window, and a window outside the
    # stored history comes back as an UNAVAILABLE row rather than a guess.
    #
    # ORDER (§40): this runs BEFORE `dispersion` because the worst stress
    # loss is now one of the views dispersion compares. It reads nothing
    # dispersion produces, so moving it earlier changes no number it emits —
    # the stress rows are byte-identical to what stage (f2) produced.
    scenarios = _safe(
        "stress_catalogue",
        lambda: stress_models.default_scenarios(bars_by_ticker),
    )
    if isinstance(scenarios, ModelResult):
        scenarios = ()
    stress_result = _safe(
        "run_stress",
        lambda: stress_models.run_stress(
            stock_legs,
            option_legs,
            scenarios,
            nav=nav if nav > 0 else None,
        ),
    )
    if isinstance(stress_result, ModelResult):
        stress_result = None

    # §40 (compliance §3 Tier C): the dispersion input set now spans the
    # STATISTICAL and STRESS families. The stress entry is a PSEUDO-view
    # built from the run's absolute worst loss — the widest view in §40's
    # own worked example — and is simply ABSENT when the stress layer
    # produced no comparable number, so a book with no stress coverage
    # yields exactly the three-view ratio it always did.
    dispersion_views = {
        k: var_views[k] for k in DISPERSION_VIEW_KEYS if k in var_views
    }
    stress_view = _safe(
        "stress_dispersion_view",
        lambda: _stress_dispersion_view(stress_result, as_of=window_end),
    )
    if isinstance(stress_view, ModelResult):
        dispersion_views[DISPERSION_STRESS_KEY] = stress_view
    dispersion_result = _safe(
        "dispersion",
        lambda: dispersion(dispersion_views, as_of=window_end),
    )
    if isinstance(dispersion_result, ModelResult):
        dispersion_result = None

    # -- (f2) §34 diversification ratio (audit.md:215, P1) -------------------
    # DR = Σ_i stdev(pnl_i) / stdev(pnl_total), ddof=1, over the SAME aligned
    # window as every other view. w_i = 1 because the per-position series are
    # already USD P&L. Honest null below `min_obs`; SHADOW — decides nothing.
    diversification = _safe(
        "diversification_ratio",
        lambda: diversification_ratio(
            per_position_pnl,
            pnl_total,
            min_obs=MIN_OBS_STATISTICAL,
            as_of=window_end,
        ),
    )

    # -- (f2b) §11 single-factor (SPY) diagnostic ----------------------------
    # RESEARCH (spec §70): display only — no cap, no gate, no registered
    # model. `factor_risk_share` pairs its inputs POSITIONALLY and cannot
    # align dates itself, so the SPY series is inner-joined onto the book's
    # OWN P&L dates here and the P&L columns are subset to the surviving
    # dates. Zipping the two raw series would silently regress the book on
    # a misaligned market — the numbers would look plausible and mean
    # nothing, which is the one failure mode this diagnostic must not have.
    factor_result = None
    factor_reason: str | None = None
    if not per_position_pnl:
        factor_reason = "no book P&L series (empty book)"
    else:
        factor_bars = await stored_bars_by_ticker(
            session, [FACTOR_TICKER], lookback_bars=DEFAULT_LOOKBACK_BARS
        )
        spy_bars = factor_bars.get(FACTOR_TICKER) or []
        if len(spy_bars) < 2:
            factor_reason = (
                f"no stored {FACTOR_TICKER} bars ({len(spy_bars)} found); the "
                f"factor series is unavailable, which is not a statement "
                f"about this book's market exposure"
            )
        else:
            factor_series = _safe(
                f"returns:{FACTOR_TICKER}",
                lambda b=spy_bars: returns_from_closes(
                    FACTOR_TICKER, b, return_type=RETURN_TYPE_SIMPLE
                ),
            )
            if isinstance(factor_series, ModelResult):
                factor_reason = f"{FACTOR_TICKER} return series could not be built"
            else:
                by_date = dict(zip(factor_series.dates, factor_series.values))
                book_dates = list(book.dates) if book is not None else []
                keep = [i for i, d in enumerate(book_dates) if d in by_date]
                if len(keep) < DEFAULT_FACTOR_PARAMS.min_obs:
                    factor_reason = (
                        f"only {len(keep)} of {len(book_dates)} book P&L dates "
                        f"have a {FACTOR_TICKER} return (min_obs="
                        f"{DEFAULT_FACTOR_PARAMS.min_obs})"
                    )
                else:
                    aligned_factor = [by_date[book_dates[i]] for i in keep]
                    aligned_pnl = {
                        key: [col[i] for i in keep]
                        for key, col in per_position_pnl.items()
                    }
                    factor_result = _safe(
                        "factor_risk_share",
                        lambda: factor_risk_share(
                            aligned_pnl,
                            aligned_factor,
                            factor=FACTOR_TICKER,
                            as_of=book_dates[keep[-1]],
                        ),
                    )
                    if isinstance(factor_result, ModelResult):
                        factor_result = None
                        factor_reason = "the factor diagnostic raised"

    # -- (f3) §12 GARCH term structure of the DISPLAY horizons --------------
    # sigma_h = sqrt(Σ_{k=1..h} sigma²_{t+k}) — the REAL variance aggregation
    # GARCH knows, not the √h scaling the historical display rows use. Served
    # only when GARCH is the live conditional source: a term structure from a
    # fit that lost the fallback would contradict the VaR rows beside it.
    conditional_horizon_sigmas = _conditional_horizon_sigmas(
        garch_fit,
        source=str(conditional_source.get("source")),
        reason=str(conditional_source.get("reason") or ""),
    )

    named_views: dict[str, ModelResult] = {
        "historical_var_95": var_views[f"HISTORICAL:{CONFIDENCE_95}:{HORIZON_DAYS}"],
        "historical_es_95": es_views[f"HISTORICAL:{CONFIDENCE_95}:{HORIZON_DAYS}"],
        "gaussian_var_95": var_views[f"GAUSSIAN:{CONFIDENCE_95}:{HORIZON_DAYS}"],
        "gaussian_es_95": es_views[f"GAUSSIAN:{CONFIDENCE_95}:{HORIZON_DAYS}"],
        "conditional_var_95": var_views[
            f"HISTORICAL_VOL_SCALED:{CONFIDENCE_95}:{HORIZON_DAYS}"
        ],
        "portfolio_volatility": volatility,
    }
    # Phase E (design §9.4): the newest PERSISTED validation verdicts feed the
    # `backtest_red_triggers` rule-table parameter. READ ONLY — a page load
    # never recomputes a walk-forward backtest (the runner is driven by the
    # SCHEDULED tick and by POST /api/risk/validation/run). No run yet ⇒ zero
    # triggers, which is the honest "not validated", never a fabricated pass.
    from .risk_validation import (
        latest_backtest_rows,
        red_verdict_count,
        validation_api_from_rows,
    )

    backtest_rows = await latest_backtest_rows(session)
    backtest_red_count, backtest_red_reasons = red_verdict_count(backtest_rows)
    validation_api_block = validation_api_from_rows(backtest_rows)

    model_risk = _safe(
        "model_risk_state",
        lambda: model_risk_state(
            named_views,
            dispersion_result=dispersion_result,
            gaussian_trust=(
                distribution.gaussian_trust if distribution is not None else None
            ),
            gaussian_views=GAUSSIAN_VIEW_NAMES,
            core_views=CORE_VIEW_NAMES,
            backtest_red_count=backtest_red_count,
            backtest_red_reasons=backtest_red_reasons,
            as_of=window_end,
        ),
    )
    if isinstance(model_risk, ModelResult):
        model_risk = None
    if model_risk is not None and not per_position_pnl:
        # An EMPTY book carries no model risk: every core view is UNAVAILABLE
        # only because there is nothing to model, not because a model failed
        # or the data ran short. Say LOW instead of ELEVATED (the rule table
        # cannot tell the two apart; the builder can). The dataclass keeps
        # LOW reason-free by construction; the API layer names the real
        # cause (see _statistical_api EMPTY_BOOK_NOTE).
        model_risk = dataclasses.replace(model_risk, state="LOW", reasons=())
    stage.observe("ensemble")

    # -- (f) drawdown: the persisted SCHEDULED NAV path + reconstruction -----
    nav_series = await _scheduled_nav_series(session)
    dd = _safe("drawdown", lambda: drawdown(nav_series))
    if isinstance(dd, ModelResult):
        dd = None
    reconstructed = None
    if pnl_total and nav > 0:
        reconstructed = _safe(
            "reconstructed_book_drawdown",
            lambda: reconstructed_book_drawdown(
                pnl_total, nav, dates=list(matrix.dates) if matrix else None
            ),
        )
        if isinstance(reconstructed, ModelResult):
            reconstructed = None
    stage.observe("drawdown")

    # -- (f2) Phase D stress: the catalogue and the run over the legs ------
    # SHADOW (design §8.5). The legs were resolved in (d0) above, from the
    # SAME chain resolution the greeks read used, so a stress reprice and the
    # greeks panel can never be anchored to different contracts. `_safe`
    # wraps the whole layer: a stress failure degrades to
    # `stress_result=None` (the API block then reports UNAVAILABLE with the
    # reason) and never 500s the view or touches a Tier 0 number.
    # `stress_excluded` is NOT merged into `positions_excluded`: that list
    # is the STATISTICAL view's, and the two exclusions are different facts. A position can have
    # a known delta (so it IS in the book P&L) while its contract has rolled
    # off today's chain (so it has no stress leg), and the reverse happens
    # too. Merging them would make `data_quality.keys_excluded` claim
    # positions were dropped from a view they are actually in. The stress
    # gaps travel on the stress block, next to the numbers they explain.
    # The scenario catalogue is derived from the SAME stored bars the return
    # matrix was built on (`bars_by_ticker`): a historical window's shock is
    # the real cumulative return over that window, and a window outside the
    # stored history comes back as an UNAVAILABLE row rather than a guess.
    # The RUN itself moved UP to stage (e2), before `dispersion`, because
    # §40 now counts the worst stress loss as a dispersion view and the
    # ensemble cannot compare a number that has not been computed yet. Only
    # the SERIALISATION stayed here, beside the leg counts it reports.
    stress_api = _stress_api(
        stress_result,
        n_stock_legs=len(stock_legs),
        n_option_legs=len(option_legs),
        positions_excluded=stress_excluded,
    )
    stage.observe("stress")

    # -- (g) data quality + the typed snapshot -------------------------------
    reasons: list[str] = []
    if cash is None:
        reasons.append("no account: cash is unknown (NAV cannot be measured)")
    if tickers_missing:
        reasons.append(f"tickers_missing={tuple(tickers_missing)}")
    if positions and n_obs < MIN_OBS_STATISTICAL:
        reasons.append(f"n_obs={n_obs} < min_obs={MIN_OBS_STATISTICAL}")
    data_quality = DataQuality(
        as_of=window_end,
        oldest_bar=window_start,
        newest_bar=window_end,
        tickers_missing=tuple(tickers_missing),
        n_obs=n_obs,
        valid=not reasons,
        reasons=tuple(reasons),
    )
    model_health: dict[str, ModelHealth] = {
        name: result.health for name, result in named_views.items()
    }
    # §65: publish the per-model health as an ordinal gauge at EVERY build.
    # A gauge, not a counter: health is a current state, and `max()` over the
    # label set is the book's worst model health. Never fails the build — an
    # observability write must not be able to take down a read view.
    try:
        _set_model_health_gauge(model_health)
    except Exception:  # noqa: BLE001 — telemetry is never load-bearing
        logger.exception("model_health_gauge_failed")
    snapshot = PortfolioRiskSnapshot(
        as_of=now,
        nav=nav,
        cash=cash if cash is not None else 0.0,
        cash_reserved=cash_reserved,
        gross_exposure=gross_exposure,
        delta_adjusted_exposure=delta_adjusted_exposure,
        heat_pct=heat,
        heat_state=state,
        volatility=volatility,
        var=var_views,
        es=es_views,
        drawdown=dd,
        greeks=greeks,
        contributions_vol=contributions_vol,
        contributions_es=contributions_es,
        distribution=distribution,
        dispersion=dispersion_result,
        model_risk=model_risk,
        # §45 BUG FIX: `correlation_state` has been a DECLARED field of the
        # dataclass since Phase B, but the builder never passed it — so the
        # typed snapshot always said None while the very same object reached
        # the wire through `_statistical_api`'s separate argument. Two
        # surfaces, one number, disagreeing. Passing it here makes the typed
        # snapshot and the API dict agree; SHADOW either way.
        correlation_state=correlation_state,
        # §45 (compliance §3 Tier C): the stress result was travelling to
        # the wire ONLY through the untyped `stress_api` dict, so the typed
        # snapshot could not answer "what was the worst stress loss?" —
        # exactly the shape of the `correlation_state` defect fixed above.
        # It is the SAME object `_stress_api` serialised, so the typed
        # snapshot and the API block cannot disagree.
        stress=stress_result,
        data_quality=data_quality,
        model_health=model_health,
        risk_state=state,
        ttl=TtlPolicy(),
    )

    # -- (i) the contract §6 API blocks --------------------------------------
    api = _statistical_api(
        snapshot,
        nav=nav,
        now=now,
        positions_excluded=positions_excluded,
        capital_weights=capital_weights,
        meta_by_key=meta_by_key,
        book=book,
        correlation_state=correlation_state,
        stress=stress_api,
        validation=validation_api_block,
        conditional_source=conditional_source,
        diversification=diversification,
        conditional_horizon_sigmas=conditional_horizon_sigmas,
        factor=factor_result,
        factor_reason=factor_reason,
        contributions_es99=contributions_es99,
    )
    drawdown_api = _drawdown_api(dd, reconstructed, nav_series)

    # -- (h) persistence (flush only; the CALLER commits) --------------------
    # ON_DEMAND builds (the risk view; the UI polls it every 15 s) persist at
    # most once per ON_DEMAND_PERSIST_MIN_INTERVAL_SECONDS — otherwise every
    # page refresh would write a snapshot + metrics + contributions + the
    # whole stress catalogue (QA finding, Phase D). The build itself always
    # runs; only the write is deduplicated, and the API says so
    # (``persisted`` false, ``snapshot_id`` null).
    row_id: int | None = None
    persisted = False
    if persist and trigger == TRIGGER_ON_DEMAND:
        newest = await latest_snapshot_row(session)
        if newest is not None:
            newest_as_of = newest.as_of
            if newest_as_of.tzinfo is None:  # sqlite returns naive UTC
                newest_as_of = newest_as_of.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - newest_as_of).total_seconds()
            if age < ON_DEMAND_PERSIST_MIN_INTERVAL_SECONDS:
                persist = False
    if persist:
        persisted = True
        row_id = await _persist(
            session,
            snapshot=snapshot,
            trigger=trigger,
            nav=nav,
            n_positions=len(pairs),
            window_start=window_start,
            window_end=window_end,
            book=book,
            capital_weights=capital_weights,
            meta_by_key=meta_by_key,
            reconstructed=reconstructed,
            stress=stress_result,
            diversification=diversification,
            contributions_es99=contributions_es99,
            # §13: only on the GARCH branch. `conditional_source` is the one
            # place that decision is made (stage (e1)); reading it here
            # rather than re-testing the fit's health keeps the persisted
            # row and the served `conditional_source` block in agreement.
            garch_fit=(
                garch_fit
                if conditional_source.get("source") == garch_models.SOURCE_GARCH
                else None
            ),
        )
        api["snapshot_id"] = row_id
    api["persisted"] = persisted
    stage.observe("persist")

    latency = time.perf_counter() - build_started
    RISK_SNAPSHOT_BUILDS_TOTAL.inc(trigger=trigger)
    # Scrape-time age of the newest SCHEDULED snapshot (spec §55): a
    # CALLBACK, not a value written here — a gauge frozen at 0.0 by the last
    # build would say "fresh" forever after the builder stopped running, and
    # ON_DEMAND / PRE_TRADE builds (page loads, previews) must not mask a
    # dead scheduled writer either — which is exactly the failure the metric
    # exists to reveal (QA finding, Phase B verification).
    if trigger == TRIGGER_SCHEDULED:
        _as_of = snapshot.as_of
        RISK_SNAPSHOT_AGE_SECONDS.set_callback(
            lambda: (datetime.now(timezone.utc) - _as_of).total_seconds()
        )
    return RiskSnapshotBuild(
        snapshot=snapshot,
        row_id=row_id,
        api=api,
        drawdown_api=drawdown_api,
        positions_excluded=positions_excluded,
        latency_seconds=latency,
        # ADDITIVE (Phase C §7.5): the inputs, so a pre-trade caller compares
        # its candidate against THIS book rather than rebuilding one.
        book=book,
        returns=matrix,
        positions=tuple(positions),
        capital_weights=dict(capital_weights),
        nav=nav,
        cash=cash,
        correlation_state=correlation_state,
        # ADDITIVE (Phase D §8.5): the legs and the run, for the pre-trade
        # cap search — the same objects the numbers above were measured on.
        stock_legs=tuple(stock_legs),
        option_legs=tuple(option_legs),
        stress=stress_result,
        scenarios=tuple(scenarios),
    )


class _PositionRiskLite:
    """Duck-typed :class:`~libs.trading_core.risk.PositionRisk` for heat.

    ``portfolio_heat`` reads ``market_value`` and ``max_loss`` only; a tiny
    local carrier avoids importing the Tier 0 dataclass into a SHADOW path
    (and can never be mistaken for one).
    """

    __slots__ = ("ticker", "market_value", "max_loss")

    def __init__(self, ticker: str, market_value: float, max_loss: float) -> None:
        self.ticker = ticker
        self.market_value = market_value
        self.max_loss = max_loss


def _statistical_api(
    snapshot: PortfolioRiskSnapshot,
    *,
    nav: float,
    now: datetime,
    positions_excluded: list[dict],
    capital_weights: Mapping[str, float],
    meta_by_key: Mapping[str, tuple[str, str]],
    book,
    correlation_state: Any | None = None,
    stress: dict | None = None,
    validation: dict | None = None,
    conditional_source: Mapping[str, Any] | None = None,
    diversification: ModelResult | None = None,
    factor: Any | None = None,
    factor_reason: str | None = None,
    conditional_horizon_sigmas: Mapping[str, Any] | None = None,
    contributions_es99: Any | None = None,
) -> dict:
    """Serialise the snapshot into the contract §6 ``"statistical"`` block.

    Every key of the contract is ALWAYS present; values are honest nulls
    when the underlying number could not be measured. Methodology fields
    (``model_name``, ``model_version``, ``distribution``, ``confidence``,
    ``horizon_days``, ``sample_size``, ``health``, ``reason``) travel with
    every number so the UI's "How is this calculated?" panel (spec §50)
    renders server-generated strings verbatim rather than inventing labels.
    """
    dq = snapshot.data_quality
    var_rows = []
    es_rows = []
    for label, conf, horizon in ALL_VIEW_ORDER:
        key = f"{label}:{conf}:{horizon}"
        for source, sink in ((snapshot.var, var_rows), (snapshot.es, es_rows)):
            result = source.get(key)
            if result is None:
                continue
            row = _metric_row_payload(result, label, conf, horizon)
            row["pct_nav"] = _pct_nav(result.value, nav)
            sink.append(row)

    vol = snapshot.volatility
    volatility_block = None
    if vol is not None:
        annualized = vol.diagnostics.get("annualized_usd")
        volatility_block = {
            "value_usd": vol.value,
            "pct_nav": _pct_nav(vol.value, nav),
            "annualized_pct_nav": _pct_nav(annualized, nav),
            "health": str(vol.health),
            "reason": vol.reason,
            "sample_size": vol.sample_size,
            "model_name": vol.meta.model_name,
            "model_version": vol.meta.model_version,
        }

    dist = snapshot.distribution
    distribution_block = None
    if dist is not None:
        distribution_block = {
            "primary": dist.primary,
            "flags": list(dist.flags),
            "skew": dist.skew,
            "excess_kurtosis": dist.excess_kurtosis,
            "jarque_bera": dist.jarque_bera,
            "jb_p": dist.jb_p,
            "gaussian_trust": dist.gaussian_trust,
            "n": dist.n,
            "health": str(dist.health),
            "reason": dist.reason,
        }

    disp = snapshot.dispersion
    dispersion_block = None
    if disp is not None:
        dispersion_block = {
            "ratio": disp.ratio,
            "high": disp.is_high,
            "min_model": disp.min_name,
            "max_model": disp.max_name,
            "n_comparable": disp.n_comparable,
            "health": str(disp.health),
            "reason": disp.reason,
        }

    mr = snapshot.model_risk
    model_risk_block = None
    if mr is not None:
        reasons = list(mr.reasons)
        if not reasons and book is not None and not book.per_position:
            reasons = [EMPTY_BOOK_NOTE]
        elif not reasons and book is None:
            reasons = [EMPTY_BOOK_NOTE]
        model_risk_block = {"state": mr.state, "reasons": reasons}

    return {
        "mode": MODE_SHADOW,
        "snapshot_id": None,  # set by the builder when persisted
        "snapshot_version": snapshot.snapshot_version,
        "as_of": snapshot.as_of.isoformat(),
        "stale": snapshot.is_stale(now),
        # The BOOK-LEVEL summary (design §10.3): FULL_REVAL_CONST_IV when at
        # least one position was revalued through Black-Scholes, else
        # DELTA_LINEAR. `data_quality.pnl_method_by_key` below says which
        # rows — a one-word summary of a mixed book is only honest with the
        # per-key map served beside it.
        "pnl_method": book.method if book is not None else METHOD_DELTA_LINEAR,
        "n_obs": dq.n_obs,
        "window_start": _iso(dq.oldest_bar),
        "window_end": _iso(dq.newest_bar),
        "data_quality": {
            "valid": dq.valid,
            "reasons": list(dq.reasons),
            "tickers_missing": list(dq.tickers_missing),
            "keys_excluded": [e["key"] for e in positions_excluded],
            # ADDITIVE (design §10.3): position key -> the estimator that
            # actually priced its P&L series. Empty on an empty book. An
            # option position absent from this map with the book present was
            # priced DELTA_LINEAR because its chain gave no IV — the honest
            # fallback, labelled rather than hidden.
            "pnl_method_by_key": (
                dict(book.method_by_key) if book is not None else {}
            ),
            "pnl_method_counts": (
                dict(book.method_counts)
                if book is not None
                else {METHOD_FULL_REVAL_CONST_IV: 0, METHOD_DELTA_LINEAR: 0}
            ),
        },
        "model_health": {
            name: str(health) for name, health in snapshot.model_health.items()
        },
        "model_risk": model_risk_block,
        "dispersion": dispersion_block,
        "distribution": distribution_block,
        "volatility": volatility_block,
        "var": var_rows,
        "es": es_rows,
        "contributions": {
            "es": _contribution_block(
                snapshot.contributions_es, nav, capital_weights, meta_by_key,
                with_confidence=True,
            ),
            "vol": _contribution_block(
                snapshot.contributions_vol, nav, capital_weights, meta_by_key,
                with_confidence=False,
            ),
            # ADDITIVE (§10): the SAME Euler decomposition at 99 %, shipped
            # WITH the audit's own "noisy at 99%" caveat carried in `health`
            # / `reason` rather than omitted. `null` on an empty book, like
            # the 95 % block beside it. Nothing reads it to decide.
            "es99": _contribution_block(
                contributions_es99, nav, capital_weights, meta_by_key,
                with_confidence=True,
            ),
        },
        # ADDITIVE (Phase C contract §7.5): the §19 correlation regime of the
        # book's tickers on LOG returns. Null with fewer than two tickers —
        # a correlation needs a pair (honest null, not a state of nulls).
        "correlation_state": correlation_state_api(correlation_state),
        # ADDITIVE (Phase D contract §8.5): the stress catalogue run over the
        # current book. ALWAYS an object — the honest nulls live inside the
        # rows (an unavailable window is a row with a reason, not a missing
        # block). SHADOW: no scenario here decides anything.
        "stress": stress if stress is not None else _stress_api(
            None, n_stock_legs=0, n_option_legs=0
        ),
        # ADDITIVE (Phase E contract §9.4): the VaR/ES walk-forward backtest
        # verdicts, read from the NEWEST PERSISTED rows. NULL before any run
        # has happened — an honest "never validated". Deliberately NOT
        # computed here: a page read must never pay for a walk-forward
        # backtest, and a number computed on the read path would silently
        # differ from the persisted history the UI shows next to it.
        "validation": validation,
        # ADDITIVE (Phase E §9.3/§9.5): which filter is behind the conditional
        # VaR/ES rows and the σ tile — GARCH (RESEARCH, when its fit is
        # ACTIVE) or EWMA (the §13/§58 fallback), with the reason.
        "conditional_source": conditional_source,
        # ADDITIVE (§34, audit.md:215 P1): DR = Σ_i σ_i / σ_p over the book
        # P&L series. A pure RATIO (not a fraction, not USD): 1.0 means no
        # diversification at all. ALWAYS an object — honest nulls inside.
        "diversification_ratio": _diversification_api(diversification),
        # ADDITIVE (§11, compliance §3 row 11): the single-factor (SPY)
        # RESEARCH diagnostic. ALWAYS an object — an unmeasurable book is a
        # block with a null and a real reason, never a missing key. Derives
        # no cap and gates nothing.
        "factor": _factor_api(factor, reason=factor_reason),
        # ADDITIVE (§12): the GARCH variance TERM STRUCTURE at the display
        # horizons, served only when GARCH is the live conditional source.
        # Deliberately NOT the √h number on the historical display rows —
        # `source` names which aggregation produced it.
        "conditional_horizon_sigmas": (
            dict(conditional_horizon_sigmas)
            if conditional_horizon_sigmas is not None
            else _conditional_horizon_sigmas(None, source="", reason="")
        ),
        "positions_excluded": list(positions_excluded),
    }


def _drawdown_api(dd, reconstructed, nav_series: Sequence[tuple[date, float]]) -> dict:
    """Serialise the contract §6 ``"drawdown"`` block.

    ``nav_series`` names its own provenance (``risk_snapshots SCHEDULED``)
    so nobody mistakes a two-day-old platform's drawdown for a real account
    history; ``reconstructed`` carries the library's honest
    ``RECONSTRUCTED_CURRENT_BOOK`` label — today's book replayed over the
    return window, NOT a NAV path that ever existed.
    """
    return {
        "nav_series": {
            "n": len(nav_series),
            "since": _iso(nav_series[0][0]) if nav_series else None,
            "source": "risk_snapshots SCHEDULED",
        },
        "current_pct": dd.current_dd_pct if dd is not None else None,
        "max_pct": dd.max_dd_pct if dd is not None else None,
        "peak_date": _iso(dd.peak_date) if dd is not None else None,
        "trough_date": _iso(dd.trough_date) if dd is not None else None,
        "peak_nav": dd.peak_nav if dd is not None else None,
        "health": str(dd.health) if dd is not None else str(ModelHealth.UNAVAILABLE),
        "reason": dd.reason if dd is not None else "no drawdown result",
        "reconstructed": (
            {
                "label": reconstructed.method,
                "current_pct": reconstructed.current_dd_pct,
                "max_pct": reconstructed.max_dd_pct,
                "n_obs": reconstructed.n_obs,
                "health": str(reconstructed.health),
                "reason": reconstructed.reason,
            }
            if reconstructed is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Persistence (migration 018 rows; flush only — the CALLER commits)
# ---------------------------------------------------------------------------


def garch_fit_metric_row(
    fit,
    *,
    snapshot_id: int,
    as_of: datetime,
) -> RiskMetricRow | None:
    """The §13 ``COND_VOL_FIT`` row for a live GARCH fit, or ``None``.

    THE GAP THIS CLOSES. The snapshot's GARCH branch used to DISCARD the
    ``GarchFit`` once it had the scaled series: ω/α/β, persistence,
    half-life, the Ljung-Box statistic on standardized residuals² and the
    optimiser's own verdict existed for the length of one function call and
    were never written down. Spec §13 is explicit that a fitted GARCH is
    never to be trusted without its diagnostics, and the compliance audit
    found them present only on the VALIDATION path — so a reader of a
    persisted snapshot could not tell whether the conditional VaR beside it
    came from a well-identified fit or a near-integrated one.

    ``None`` (no row at all, rather than a row of nulls) when there is no
    fit, or when the fit produced no parameters: an UNAVAILABLE/FAILED fit
    means EWMA is driving the conditional views, and a ``COND_VOL_FIT`` row
    would then claim a GARCH fit was in force when it was not. The caller
    writes this row ONLY on the GARCH branch for the same reason.

    ``value`` is the one-step-ahead conditional volatility σ_{t+1} in USD
    per day — the number the fit exists to produce — so the row is a metric
    and not merely a diagnostics envelope. ``value_pct_nav`` is left to the
    caller's ``_pct_nav``. Health/reason are the FIT's own, so a DEGRADED
    fit (near-integrated, or Ljung-Box rejecting) says so on the row.
    """
    if fit is None or not getattr(fit, "is_available", False):
        return None
    diagnostics = dict(getattr(fit, "diagnostics", {}) or {})
    sigma_next = diagnostics.get("sigma_next")
    # Every §13 key, explicitly — an absent key becomes an honest None here
    # rather than vanishing from the persisted record.
    payload: dict[str, Any] = {
        key: _jsonable(diagnostics.get(key)) for key in GARCH_FIT_DIAGNOSTIC_KEYS
    }
    # The reproducibility extras that make the row replayable on its own.
    payload.update(
        {
            "loglik": _jsonable(fit.loglik),
            "iterations": _jsonable(fit.iterations),
            "unconditional_var": _jsonable(fit.unconditional_var),
            "sigma2_next": _jsonable(diagnostics.get("sigma2_next")),
            "n": _jsonable(fit.n),
            "ljung_box_note": diagnostics.get("ljung_box_note"),
        }
    )
    return RiskMetricRow(
        snapshot_id=snapshot_id,
        metric=METRIC_COND_VOL_FIT,
        model_name=MODEL_NAME_GARCH_FIT,
        model_version=garch_models.MODEL_VERSION,
        confidence=None,          # a volatility fit carries no tail probability
        horizon_days=HORIZON_DAYS,
        distribution=garch_models.DISTRIBUTION_GAUSSIAN_GARCH,
        value=_jsonable(sigma_next),
        value_pct_nav=None,       # USD/day, not a loss — never divided by NAV
        health=str(fit.health),
        reason=fit.reason,
        sample_size=int(fit.n),
        params={
            "estimator": "GARCH(1,1), Gaussian innovations, zero mean",
            "optimizer": diagnostics.get("optimizer"),
            "annualization_days": _jsonable(diagnostics.get("annualization_days")),
            "tier": str(garch_models.Garch11Model.tier),
            "mode": str(ModelMode.RESEARCH),
        },
        diagnostics=payload,
        as_of=as_of,
    )


async def _persist(
    session: AsyncSession,
    *,
    snapshot: PortfolioRiskSnapshot,
    trigger: str,
    nav: float,
    n_positions: int,
    window_start: date | None,
    window_end: date | None,
    book,
    capital_weights: Mapping[str, float],
    meta_by_key: Mapping[str, tuple[str, str]],
    reconstructed,
    stress=None,
    diversification: ModelResult | None = None,
    contributions_es99=None,
    garch_fit=None,
) -> int:
    """Add one snapshot row + its metric/contribution rows and FLUSH.

    Flush (not commit) is the whole point: a PRE_TRADE build lands in the
    same transaction as the RISK_DECISION audit event and any fill, so a
    rolled-back order leaves no orphan snapshot claiming it measured a book
    that never traded.
    """
    dq = snapshot.data_quality
    dist = snapshot.distribution
    disp = snapshot.dispersion
    dd = snapshot.drawdown
    row = RiskSnapshotRow(
        as_of=snapshot.as_of,
        snapshot_version=snapshot.snapshot_version,
        trigger=trigger,
        nav=nav,
        cash=snapshot.cash,
        cash_reserved=snapshot.cash_reserved,
        gross_exposure=snapshot.gross_exposure,
        delta_adjusted_exposure=snapshot.delta_adjusted_exposure,
        heat_pct=snapshot.heat_pct,
        heat_state=snapshot.heat_state,
        n_positions=n_positions,
        n_obs=dq.n_obs,
        window_start=window_start,
        window_end=window_end,
        # design §10.3: the SAME book-level summary the API serves, so a
        # replayed row and the live view can never disagree about which
        # estimator produced the VaR/ES on it.
        pnl_method=book.method if book is not None else METHOD_DELTA_LINEAR,
        data_quality_valid=dq.valid,
        data_quality={
            "valid": dq.valid,
            "reasons": list(dq.reasons),
            "tickers_missing": list(dq.tickers_missing),
            "n_obs": dq.n_obs,
            "pnl_method_by_key": (
                dict(book.method_by_key) if book is not None else {}
            ),
        },
        model_health={k: str(v) for k, v in snapshot.model_health.items()},
        model_risk_state=snapshot.model_risk.state if snapshot.model_risk else None,
        dispersion_ratio=disp.ratio if disp is not None else None,
        dispersion_high=disp.is_high if disp is not None else None,
        distribution_primary=dist.primary if dist is not None else None,
        gaussian_trust=dist.gaussian_trust if dist is not None else None,
        drawdown_current_pct=(
            dd.current_dd_pct
            if dd is not None and dd.current_dd_pct is not None
            else (reconstructed.current_dd_pct if reconstructed is not None else None)
        ),
        drawdown_max_pct=(
            dd.max_dd_pct
            if dd is not None and dd.max_dd_pct is not None
            else (reconstructed.max_dd_pct if reconstructed is not None else None)
        ),
        risk_state=snapshot.risk_state,
    )
    session.add(row)
    await session.flush()

    for metric, views in (("VAR", snapshot.var), ("ES", snapshot.es)):
        for label, conf, horizon in ALL_VIEW_ORDER:
            key = f"{label}:{conf}:{horizon}"
            result = views.get(key)
            if result is None:
                continue
            session.add(
                RiskMetricRow(
                    snapshot_id=row.id,
                    metric=metric,
                    model_name=result.meta.model_name,
                    model_version=result.meta.model_version,
                    confidence=conf,
                    horizon_days=horizon,
                    distribution=result.meta.distribution,
                    value=result.value,
                    value_pct_nav=_pct_nav(result.value, nav),
                    health=str(result.health),
                    reason=result.reason,
                    sample_size=result.sample_size,
                    params=_meta_params(result.meta),
                    diagnostics=_jsonable(result.diagnostics),
                    as_of=snapshot.as_of,
                )
            )
    vol = snapshot.volatility
    if vol is not None:
        session.add(
            RiskMetricRow(
                snapshot_id=row.id,
                metric="VOLATILITY",
                model_name=vol.meta.model_name,
                model_version=vol.meta.model_version,
                confidence=None,
                horizon_days=HORIZON_DAYS,
                distribution=vol.meta.distribution,
                value=vol.value,
                value_pct_nav=_pct_nav(vol.value, nav),
                health=str(vol.health),
                reason=vol.reason,
                sample_size=vol.sample_size,
                params=_meta_params(vol.meta),
                diagnostics=_jsonable(vol.diagnostics),
                as_of=snapshot.as_of,
            )
        )

    # §34: ONE extra risk_metrics row. `metric` is DIVERSIFICATION_RATIO and
    # `value` is a pure RATIO — not a USD loss like VAR/ES and not USD/day
    # like VOLATILITY — so `value_pct_nav` stays NULL rather than dividing a
    # ratio by NAV, which would be arithmetic without meaning. `confidence`
    # is NULL (a diversification ratio has no tail) and `horizon_days` is the
    # 1-day window the per-position series are measured on.
    if diversification is not None:
        session.add(
            RiskMetricRow(
                snapshot_id=row.id,
                metric=METRIC_DIVERSIFICATION_RATIO,
                model_name=diversification.meta.model_name,
                model_version=diversification.meta.model_version,
                confidence=None,
                horizon_days=HORIZON_DAYS,
                distribution=diversification.meta.distribution,
                value=diversification.value,
                value_pct_nav=None,
                health=str(diversification.health),
                reason=diversification.reason,
                sample_size=diversification.sample_size,
                params=_meta_params(diversification.meta),
                diagnostics=_jsonable(diversification.diagnostics),
                as_of=snapshot.as_of,
            )
        )

    # §10: the ES-99 rows persist under the SAME `method="ES"` as the 95 %
    # rows and are told apart by `confidence` — which the row already
    # carries — rather than by a new method spelling that every existing
    # reader would have to learn. `result.confidence` is 0.99 for these.
    # §13: ONE `COND_VOL_FIT` row when GARCH is the LIVE conditional source,
    # carrying the fit diagnostics the snapshot used to throw away. The
    # caller passes `garch_fit` only on the GARCH branch, so the row's mere
    # presence records which filter was in force (the EWMA branch writes
    # none). A fit that produced no parameters yields `None` here.
    garch_row = garch_fit_metric_row(
        garch_fit, snapshot_id=row.id, as_of=snapshot.as_of
    )
    if garch_row is not None:
        session.add(garch_row)

    for method, result in (
        ("ES", snapshot.contributions_es),
        ("ES", contributions_es99),
        ("VOL", snapshot.contributions_vol),
    ):
        if result is None:
            continue
        for contrib in result.per_position:
            ticker, instrument = meta_by_key.get(contrib.key, ("", ""))
            session.add(
                RiskContributionRow(
                    snapshot_id=row.id,
                    method=method,
                    confidence=result.confidence,
                    position_key=contrib.key,
                    ticker=ticker,
                    instrument=instrument,
                    contribution=contrib.contribution,
                    share=contrib.share,
                    capital_weight=capital_weights.get(contrib.key),
                )
            )

    # Phase D (design §8.4): one `stress_runs` row per SCENARIO of this
    # build — the catalogue history. UNAVAILABLE rows are persisted too, with
    # their reason: "that window is outside our stored history" is a fact
    # worth keeping, and a silently absent row would later read as "never
    # run" (spec §56).
    if stress is not None:
        for srow in stress.rows:
            session.add(
                StressRunRow(
                    snapshot_id=row.id,
                    scenario=srow.name,
                    kind=srow.kind,
                    validated=srow.validated,
                    pnl_usd=srow.pnl_usd,
                    pnl_pct_nav=srow.pnl_pct_nav,
                    method_full_reval=int(
                        srow.method_coverage.get(REVAL_METHOD_FULL_REVAL, 0)
                    ),
                    method_delta_linear=int(
                        srow.method_coverage.get(REVAL_METHOD_DELTA_LINEAR, 0)
                    ),
                    health=str(srow.health),
                    reason=srow.reason,
                    params=_jsonable(srow.params),
                    per_position=_jsonable(dict(srow.per_key)),
                    as_of=snapshot.as_of,
                )
            )
    await session.flush()
    return row.id


# ---------------------------------------------------------------------------
# atm_iv_daily (spec §24; audit §7.1) — best-effort, never raises
# ---------------------------------------------------------------------------


async def record_atm_iv(
    session: AsyncSession,
    ticker: str,
    *,
    bar_date: date,
    atm_iv: float | None,
    spot: float | None,
    expiry: date | None = None,
    dte: int | None = None,
    source: str,
) -> None:
    """Upsert one ``atm_iv_daily`` row for ``(ticker, bar_date)``.

    INTERNALLY CALCULATED provenance: ``source`` labels which chain the IV
    was derived from (``"alpaca_chain"``, ``"stub_chain"``, …) so it is
    never mistaken for vendor IV history (data-source architecture §12).

    BEST-EFFORT BY CONTRACT: this is a side observation of a read that had
    the number in hand anyway (spec §24 needs IV history for empirical IV
    shocks and IV rank). It must never fail the caller — a chain view, an
    order gate — so every exception is swallowed after one log line, and a
    null IV or spot is simply not recorded (there is nothing to record).

    Upsert by SELECT-then-INSERT/UPDATE rather than a dialect-specific
    ``ON CONFLICT``: the platform runs sqlite in tests and postgres live,
    and one portable path is worth more here than one round trip. The row
    is added/updated on the session; the CALLER commits (a read path that
    commits nothing simply leaves it, which is the honest outcome for a
    best-effort observation).
    """
    if atm_iv is None or spot is None:
        return
    try:
        if not math.isfinite(atm_iv) or not math.isfinite(spot) or spot <= 0:
            return
        existing = (
            await session.execute(
                select(AtmIvDailyRow).where(
                    AtmIvDailyRow.ticker == ticker,
                    AtmIvDailyRow.bar_date == bar_date,
                )
            )
        ).scalars().first()
        if existing is None:
            session.add(
                AtmIvDailyRow(
                    ticker=ticker,
                    bar_date=bar_date,
                    atm_iv=float(atm_iv),
                    spot=float(spot),
                    expiry=expiry,
                    dte=dte,
                    source=source,
                )
            )
        else:
            existing.atm_iv = float(atm_iv)
            existing.spot = float(spot)
            existing.expiry = expiry
            existing.dte = dte
            existing.source = source
        await session.flush()
    except Exception:  # noqa: BLE001 — best-effort by contract (docstring)
        logger.exception(
            "atm_iv_record_failed",
            extra={"extra_fields": {"ticker": ticker, "bar_date": str(bar_date)}},
        )


async def latest_snapshot_row(
    session: AsyncSession, *, trigger: str | None = None
) -> RiskSnapshotRow | None:
    """The newest persisted ``risk_snapshots`` row (optionally by trigger).

    ``None`` when none exists — an honest "never built" rather than a
    fabricated empty snapshot.
    """
    stmt = select(RiskSnapshotRow)
    if trigger is not None:
        stmt = stmt.where(RiskSnapshotRow.trigger == trigger)
    stmt = stmt.order_by(RiskSnapshotRow.as_of.desc(), RiskSnapshotRow.id.desc()).limit(1)
    return (await session.execute(stmt)).scalars().first()


async def latest_worst_scenario_per_position(
    session: AsyncSession,
) -> tuple[dict[str, float], str | None]:
    """The WORST persisted scenario's per-leg P&L for the newest snapshot.

    Returns ``(per_key_pnl, scenario_name)`` — the per-leg map of the row
    with the SMALLEST ``pnl_usd`` among the newest snapshot's persisted
    ``stress_runs`` rows, so a positions view can show each option row its own
    scenario loss (spec §52) without re-running the catalogue on a read.

    ``({}, None)`` when no snapshot has been built, when its stress rows were
    all unavailable, or when nothing was persisted — an honest "not measured
    yet", never a zero that would read as "this position loses nothing".

    Keys are the LEG keys the stress engine used: a position key
    (``"AAPL#12"``) for stock and single options, and the two suffixed leg
    keys (``"AAPL#12:long"`` / ``":short"``) for a spread. A caller wanting
    one number per POSITION sums the legs whose key shares the position
    prefix — :func:`worst_scenario_pnl_for_key` does exactly that.
    """
    snapshot = await latest_snapshot_row(session)
    if snapshot is None:
        return {}, None
    rows = (
        await session.execute(
            select(StressRunRow)
            .where(StressRunRow.snapshot_id == snapshot.id)
            .where(StressRunRow.pnl_usd.is_not(None))
            .order_by(StressRunRow.pnl_usd, StressRunRow.id)
            .limit(1)
        )
    ).scalars().first()
    if rows is None:
        return {}, None
    per_position = rows.per_position or {}
    return (
        {k: v for k, v in per_position.items() if isinstance(v, (int, float))},
        rows.scenario,
    )


def worst_scenario_pnl_for_key(
    per_position: Mapping[str, float], key: str
) -> float | None:
    """One POSITION's worst-scenario P&L from the per-LEG map.

    A spread is one position and two legs, so its number is the SUM of the
    ``…:long`` and ``…:short`` legs — the position's net P&L under that
    scenario, which is the only figure a position row should show.

    ``None`` when the key contributed no leg: the position was excluded from
    the stress run (no spot, a contract off today's chain) and saying so is
    the honest answer, not 0.0.
    """
    if key in per_position:
        return per_position[key]
    prefix = key + ":"
    legs = [v for k, v in per_position.items() if k.startswith(prefix)]
    if not legs:
        return None
    return math.fsum(legs)


# ---------------------------------------------------------------------------
# Background loop — ONE SCHEDULED snapshot per America/New_York calendar day
# ---------------------------------------------------------------------------


def new_york_today() -> date:
    """Today's date on the exchange calendar the SCHEDULED rule uses."""
    return datetime.now(NEW_YORK).date()


async def _scheduled_exists_today(session: AsyncSession, day: date) -> bool:
    """True when a SCHEDULED row already exists for ``day`` (NY calendar).

    Compared in NY local time rather than by a stored date column: the row
    carries an instant (``as_of``), and "one per trading day" is a statement
    about the exchange's day, not UTC's.
    """
    rows = (
        await session.execute(
            select(RiskSnapshotRow.as_of).where(
                RiskSnapshotRow.trigger == TRIGGER_SCHEDULED
            )
        )
    ).scalars().all()
    for as_of in rows:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        if as_of.astimezone(NEW_YORK).date() == day:
            return True
    return False


async def run_scheduled_snapshot() -> dict:
    """ONE tick of :func:`risk_snapshot_loop`: persist today's SCHEDULED row.

    Split out from the loop (monitor.py pattern) so tests drive a single
    tick deterministically without a background task.

    Skips — with a log line and a named ``skipped`` reason, never a
    fabricated row — when a SCHEDULED row already exists for today's
    America/New_York date, when no market-data provider is configured
    (every return series would be missing, so the snapshot would measure
    nothing), or when cash cannot be resolved (no account ⇒ no NAV ⇒ the
    row would poison the drawdown NAV series with a zero).

    Cash is resolved exactly as the risk view resolves it: the simulator's
    ledger under ``BROKER_PROVIDER=simulated``, the broker's LIVE cash
    otherwise (§14: broker CASH, never buying power; the platform stores no
    copy of a real account).
    """
    from .deps import (
        broker_configured,
        market_data_configured,
        simulated_broker_mode,
    )
    from .db import get_or_create_portfolio, get_or_create_system_state
    from .risk_validation import (
        TRIGGER_SCHEDULED as VALIDATION_TRIGGER_SCHEDULED,
        run_model_backtests,
    )
    from .routers.portfolio import _to_thread_get_account

    day = new_york_today()
    if not market_data_configured():
        logger.info(
            "risk_snapshot_skipped",
            extra={"extra_fields": {"reason": "MARKET_DATA_NOT_CONFIGURED", "day": str(day)}},
        )
        return {"skipped": "MARKET_DATA_NOT_CONFIGURED", "day": str(day)}

    async with SessionLocal() as session:
        if await _scheduled_exists_today(session, day):
            logger.info(
                "risk_snapshot_skipped",
                extra={"extra_fields": {"reason": "ALREADY_BUILT_TODAY", "day": str(day)}},
            )
            return {"skipped": "ALREADY_BUILT_TODAY", "day": str(day)}

        cash: float | None = None
        if broker_configured() and simulated_broker_mode():
            portfolio = await get_or_create_portfolio(session)
            cash = portfolio.cash
        elif broker_configured():
            try:
                account = await _to_thread_get_account()
            except Exception as exc:  # noqa: BLE001 — a broker fault is a skip
                logger.warning(
                    "risk_snapshot_skipped",
                    extra={
                        "extra_fields": {
                            "reason": "BROKER_UNREADABLE",
                            "error": str(exc),
                            "day": str(day),
                        }
                    },
                )
                return {"skipped": "BROKER_UNREADABLE", "day": str(day)}
            cash = account.cash
        if cash is None:
            logger.info(
                "risk_snapshot_skipped",
                extra={"extra_fields": {"reason": "NO_ACCOUNT", "day": str(day)}},
            )
            return {"skipped": "NO_ACCOUNT", "day": str(day)}

        state = await get_or_create_system_state(session)
        try:
            build = await build_risk_snapshot(
                session,
                trigger=TRIGGER_SCHEDULED,
                cash=cash,
                trading_enabled=state.trading_enabled,
                persist=True,
            )
        except Exception:
            RISK_SNAPSHOT_FAILURES_TOTAL.inc()
            await session.rollback()
            raise

        # Phase E (design §9.4): ONE walk-forward validation run per NY day,
        # right after the SCHEDULED snapshot and in the SAME transaction, on
        # the SAME book P&L series the snapshot was measured on — so a
        # verdict can never describe a different book than the numbers it
        # judges. Guarded by the snapshot's own once-per-day rule (we only
        # reach here after `_scheduled_exists_today` said no), so the daily
        # cadence is inherited rather than re-implemented.
        #
        # NEVER fails the tick: a validation run is a measurement of past
        # forecasts: if it raises, the snapshot it accompanies is still a
        # good snapshot. The failure is logged with its traceback and the
        # rows simply do not appear (the API block then keeps serving the
        # previous run — honestly stamped with ITS as_of).
        validation_seconds: float | None = None
        validation_rows = 0
        if build.book is not None and build.book.total:
            try:
                run = await run_model_backtests(
                    session,
                    book_pnl=list(build.book.total),
                    dates=list(build.book.dates) if build.book.dates else None,
                    nav=build.snapshot.nav,
                    snapshot_id=build.row_id,
                    trigger=VALIDATION_TRIGGER_SCHEDULED,
                )
            except Exception:  # noqa: BLE001 — never fail the snapshot tick
                logger.exception(
                    "risk_validation_failed",
                    extra={"extra_fields": {"day": str(day), "snapshot_id": build.row_id}},
                )
            else:
                validation_seconds = round(run.seconds, 4)
                validation_rows = len(run.rows)

        await session.commit()

    logger.info(
        "risk_snapshot_scheduled",
        extra={
            "extra_fields": {
                "day": str(day),
                "snapshot_id": build.row_id,
                "nav": build.snapshot.nav,
                "n_obs": build.snapshot.data_quality.n_obs,
                "health": str(build.snapshot.overall_health()),
                "latency_seconds": round(build.latency_seconds, 4),
                # Design §9.4 asks for the validation runtime to be MEASURED
                # and LOGGED, not merely bounded by construction.
                "validation_seconds": validation_seconds,
                "validation_rows": validation_rows,
            }
        },
    )
    return {
        "snapshot_id": build.row_id,
        "day": str(day),
        "nav": build.snapshot.nav,
        "validation_rows": validation_rows,
        "validation_seconds": validation_seconds,
    }


async def risk_snapshot_loop() -> None:
    """Sleep -> build ONE SCHEDULED snapshot per NY day, forever.

    Started by the gateway lifespan when
    ``settings.risk_snapshot_interval_seconds`` > 0 (0 disables it), exactly
    like the position monitor and the reconciliation loop. Tests drive
    :func:`run_scheduled_snapshot` directly — httpx ASGITransport does not
    run the lifespan, so the task never starts under the suite.

    RESILIENCE: every exception from a tick is logged with its traceback and
    swallowed; the next tick runs normally. ``asyncio.CancelledError`` is
    always re-raised so graceful shutdown is never swallowed.
    """
    interval = get_settings().risk_snapshot_interval_seconds
    logger.info(
        "risk_snapshot_loop_started",
        extra={"extra_fields": {"interval_seconds": interval}},
    )
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await run_scheduled_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception:
                RISK_SNAPSHOT_FAILURES_TOTAL.inc()
                logger.exception("risk_snapshot_tick_failed")
    except asyncio.CancelledError:
        logger.info("risk_snapshot_loop_stopped")
        raise


__all__ = [
    "DEFAULT_LOOKBACK_BARS",
    "MIN_OBS_STATISTICAL",
    "NEW_YORK",
    "RiskSnapshotBuild",
    "TRIGGER_ON_DEMAND",
    "TRIGGER_PRE_TRADE",
    "TRIGGER_SCHEDULED",
    "build_risk_snapshot",
    "latest_snapshot_row",
    "new_york_today",
    "record_atm_iv",
    "risk_snapshot_loop",
    "run_scheduled_snapshot",
]
