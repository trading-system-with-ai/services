"""VaR/ES MODEL VALIDATION runner — the gateway seam between the persisted
book P&L history and the pure walk-forward backtest library
(``libs/trading_core/risk/validation.py``).

Risk spec §42 (backtest the risk FORECASTS, not the trading returns), §43
(walk-forward only — no hindsight), §56 (never store only the latest value),
§57 (model VALIDATION is a separate concern from model OUTPUT), §59 (model
risk is itself a risk), §63 (required model comparison), §68 (validation
acceptance); Phase B/E design contract §9.4.

WHAT THIS MODULE IS
-------------------
It answers one question per model view: *when this estimator said "the 95%
one-day VaR is $X", how often did the book actually lose more than $X, and
was the miss rate consistent with 5 %?* It does that the only honest way
(§43): the forecast for day ``t`` is estimated on the ``window``
observations STRICTLY BEFORE ``t`` and then compared with ``pnl[t]``. No
statistic here ever sees the day it forecasts, and no threshold anywhere is
fitted to the outcome (audit §9: walk-forward here is *validation*, never
parameter search).

It is a RUNNER, not an estimator. Every number comes from
``risk/validation.py`` (Kupiec POF, Christoffersen independence, ES severity)
over forecasts produced by ``risk/models/var_es.py`` and the two conditional
filters. No arithmetic lives here.

THE VIEW GRID (design §9.4)
---------------------------
======================  ==========  ============================  ========
view                    confidence  distribution                  mode
======================  ==========  ============================  ========
historical_var          0.95, 0.99  EMPIRICAL                     SHADOW
gaussian_var            0.95, 0.99  NORMAL                        SHADOW
conditional_var (EWMA)  0.95        EMPIRICAL_VOL_SCALED          SHADOW
garch_var               0.95        EMPIRICAL_GARCH_SCALED        RESEARCH
======================  ==========  ============================  ========

Each row is scored against the MATCHING ES estimator (historical ES against
historical VaR, Gaussian ES against Gaussian VaR, and so on) so the
``es_severity_ratio`` compares a model with itself rather than mixing
families.

NO LOOK-AHEAD INSIDE THE FILTERS. This is the subtle part and the reason the
conditional views are not simply ``conditional_var(whole_series)``: a
filtered-HS estimator rescales history to *today's* volatility, so running it
once over the full series and slicing would leak the end of the sample into
every early forecast. Instead the filter runs INSIDE each rolling window —
``volatility_scaled_pnl(window)`` estimates σ on that window only, and its
"σ_now" is the window's own last-day forecast, which is the forecast for the
very day being scored. The GARCH view does the same with ``garch_scaled_pnl``.

BOUNDED RUNTIME (design §9.4). A GARCH MLE per forecast day would be hundreds
of Nelder–Mead fits per run. The runner therefore REFITS at most every
``GARCH_REFIT_EVERY`` (20) forecast steps and reuses the parameters in
between, re-running only the cheap variance recursion on the current window.
That is a documented approximation, and it is *conservative for validation*:
between refits the parameters are STALER than a live model's would be, never
fresher, so the GARCH view is never flattered by information it would not
have had. The stride is recorded in every GARCH row's ``params``.

HONEST NULLS (§44 rule 18). Fewer than ``MIN_FORECASTS`` (60) usable pairs ⇒
the row is persisted with ``verdict=UNAVAILABLE``, ``health=UNAVAILABLE``,
NULL statistics and a ``reason`` carrying the real numbers. A GARCH fit whose
health is not ACTIVE ⇒ an UNAVAILABLE row quoting the fit's own reason. A row
is never skipped: a missing row would later read as "never run".

WHAT THIS MODULE NEVER DOES. It writes no audit event (a validation run is a
measurement, not a decision — house rule: read views write no audit events).
It commits nothing: rows are ``session.add`` + ``flush``, and the CALLER
commits, exactly like ``risk_snapshot._persist``. And nothing it produces
alters a Tier 0 decision — the only consumer is the SHADOW ``model_risk``
display's ``backtest_red_triggers`` parameter and the read-only
``statistical.validation`` block.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.telemetry import REGISTRY
from libs.trading_core.risk.models.base import ModelHealth, tier_for_model_name
from libs.trading_core.risk.models.garch import (
    DEFAULT_MIN_OBS as GARCH_DEFAULT_MIN_OBS,
    DISTRIBUTION_EMPIRICAL_GARCH_SCALED,
    GarchFit,
    fit_garch,
    garch_scaling,
)
from libs.trading_core.risk.models.var_es import (
    DISTRIBUTION_EMPIRICAL,
    DISTRIBUTION_EMPIRICAL_VOL_SCALED,
    DISTRIBUTION_NORMAL,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
)
from libs.trading_core.risk.models.volatility import (
    DEFAULT_INIT_OBS,
    DEFAULT_LAMBDA,
    volatility_scaled_pnl,
)
from libs.trading_core.risk.validation import (
    VALIDATION_VERSION,
    BacktestParams,
    BacktestVerdict,
    exceedances,
    walk_forward,
)

from .db import RiskModelBacktestRow

logger = logging.getLogger("apps.gateway.risk_validation")


# ---------------------------------------------------------------------------
# Parameters — every threshold documented, never a magic number (house rule)
# ---------------------------------------------------------------------------

#: Rolling estimation window, in observations (design §9.4). 250 ≈ one
#: trading year: long enough that the 95 % tail holds ~12 observations, short
#: enough that the window still describes a recognisable regime.
DEFAULT_WINDOW = 250

#: Minimum usable forecast/realized pairs before any coverage test is
#: reported (design §9.4). BELOW the library's own 250-day default on
#: purpose: this platform's stored history is ~600 bars, so a 250-window
#: walk-forward yields ~350 forecasts at best and far fewer on a young book.
#: 60 is the smallest sample at which a 95 % Kupiec test has any power at all
#: (expected 3 exceedances) — and every row says how many it actually had, so
#: a thin sample is visible rather than implied.
MIN_FORECASTS = 60

#: Kupiec traffic-light cut-offs (Basel-style; ``risk/validation.py``).
GREEN_P = 0.05
RED_P = 0.01

#: Christoffersen clustering cut-off.
CLUSTER_P = 0.05

#: GARCH refit stride (design §9.4 "keep runtime bounded"). The MLE is re-run
#: on every ``GARCH_REFIT_EVERY``-th forecast step; in between, the previous
#: window's parameters are reused and only the variance recursion re-runs on
#: the current window. See the module docstring for why this is conservative.
GARCH_REFIT_EVERY = 20

#: Observations a GARCH fit needs before it is attempted at all. The library
#: default (250) equals the walk-forward window, so a 250-observation window
#: is exactly at the boundary — kept explicit so the two can be tuned apart.
GARCH_MIN_OBS = GARCH_DEFAULT_MIN_OBS

#: EWMA filter parameters for the conditional view (RiskMetrics daily decay).
EWMA_LAMBDA = DEFAULT_LAMBDA
EWMA_INIT_OBS = DEFAULT_INIT_OBS

#: The confidence grid and horizon this platform reports (contract §1).
CONFIDENCE_95 = 0.95
CONFIDENCE_99 = 0.99
HORIZON_DAYS = 1

#: Model-name keys used on the rows and in the API block.
MODEL_HISTORICAL_VAR = "historical_var"
MODEL_GAUSSIAN_VAR = "gaussian_var"
MODEL_CONDITIONAL_VAR = "conditional_var"
MODEL_GARCH_VAR = "garch_var"

#: Mode label per view (spec §70): GARCH is RESEARCH, strictly below SHADOW.
MODE_SHADOW = "SHADOW"
MODE_RESEARCH = "RESEARCH"

#: The two views the §63 comparison weighs against each other.
COMPARISON_EWMA_KEY = MODEL_CONDITIONAL_VAR
COMPARISON_GARCH_KEY = MODEL_GARCH_VAR

#: §63 promotion criterion — the documented, RESEARCH-only sentence recorded
#: verbatim on every comparison so the bar never drifts between runs. The
#: promotion itself is a USER action; this runner only reports whether the
#: numbers currently clear the bar.
COMPARISON_CRITERION = (
    "RESEARCH: GARCH may move RESEARCH -> SHADOW only if, over at least 250 "
    "forecast days, its Kupiec p is at least EWMA's, its Christoffersen p is "
    "at least 0.05, and its diagnostics never FAILED in the window. The "
    "promotion itself is a user action, never automatic."
)

#: Forecast days the §63 criterion demands before a promotion is even
#: considered (a full trading year of out-of-sample days).
COMPARISON_MIN_FORECASTS = 250

#: Version of THIS runner's row layout (contract §4).
RUNNER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

RISK_VALIDATION_SECONDS = REGISTRY.histogram(
    "risk_validation_seconds",
    "Wall time of one VaR/ES model-validation run (walk-forward backtest of "
    "the whole view grid), in SECONDS. Design §9.4 requires this runtime to "
    "stay bounded — the GARCH view refits on a stride rather than per day.",
    (),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0),
)
RISK_VALIDATION_RUNS_TOTAL = REGISTRY.counter(
    "risk_validation_runs_total",
    "VaR/ES model-validation runs completed, by trigger "
    "(SCHEDULED | ON_DEMAND).",
    ("trigger",),
)

# --- Spec §65 instruments (compliance Tier A) ------------------------------
# The audit's point: without these there is no alertable TIME SERIES of how
# often a model breached, which is exactly the evidence the 20-day shadow
# window is supposed to accumulate. All four are OBSERVATIONS of numbers the
# platform already computes — none of them changes a decision.

VAR_EXCEEDANCES_TOTAL = REGISTRY.counter(
    "var_exceedances_total",
    "VaR exceedances counted by a walk-forward validation run, by model and "
    "confidence (spec §42/§65). An exceedance is a forecast day whose "
    "realised loss exceeded the VaR forecast made WITHOUT seeing that day. "
    "Incremented by each row's exceedance count as the run persists it, so "
    "the counter is the running total ACROSS runs, not a gauge of the newest "
    "verdict — re-running validation on the same window adds to it again; "
    "alert on rate(), not on the absolute value. "
    "`confidence` is a label because the grid scores the same model at both "
    "0.95 and 0.99, and those are different tests with different expected "
    "exceedance rates (5% vs 1%) — summing them would produce a series that "
    "matches neither.",
    ("model", "confidence"),
)
ES_EXCEEDANCES_TOTAL = REGISTRY.counter(
    "es_exceedances_total",
    "Exceedance days the ES severity ratio was actually scored on, by model "
    "and confidence (spec §42/§65). A SUBSET of var_exceedances_total for "
    "the same (model, confidence): the grid scores one row per VaR view "
    "against its own matching ES estimator, and an exceedance whose ES "
    "forecast was UNAVAILABLE is counted by the VaR test but contributes to "
    "no severity ratio. The two counters are equal on a clean run; a gap "
    "means ES forecasts went missing, which is the condition worth alerting "
    "on.",
    ("model", "confidence"),
)
GARCH_FIT_FAILURES_TOTAL = REGISTRY.counter(
    "garch_fit_failures_total",
    "GARCH(1,1) fits that did NOT produce a usable ACTIVE forecast, by call "
    "site and resulting health (spec §65). Counts the operational fact the "
    "operator cares about — 'GARCH gave us nothing usable' — which includes "
    "an UNAVAILABLE fit on a short sample and a DEGRADED one, not only a "
    "raised exception (health=RAISED covers that case).",
    ("site", "health"),
)
MODEL_HEALTH_STATE = REGISTRY.gauge(
    "model_health_state",
    "Health of one risk model as an ordinal (spec §41/§65): "
    "ACTIVE=0, DEGRADED=1, UNAVAILABLE=2, FAILED=3. Higher is worse, so "
    "max() over the label set is the book's worst model health and a "
    "threshold alert reads naturally. Set at every snapshot build.",
    ("model",),
)

#: The §65 ordinal encoding of ModelHealth. A DICTIONARY, not `list.index`,
#: so adding a health state later is a deliberate edit here rather than a
#: silent renumbering of every existing alert threshold.
MODEL_HEALTH_ORDINAL: dict[str, int] = {
    str(ModelHealth.ACTIVE): 0,
    str(ModelHealth.DEGRADED): 1,
    str(ModelHealth.UNAVAILABLE): 2,
    str(ModelHealth.FAILED): 3,
}


def set_model_health_gauge(model_health: Mapping[str, Any]) -> None:
    """Publish one ``model_health_state`` sample per model (spec §65).

    ``model_health`` is the builder's own per-model ledger
    (``{"historical_var_95": ModelHealth.ACTIVE, ...}``). An unrecognised
    health string is SKIPPED rather than mapped to a guessed ordinal — a
    fabricated 0 would read as ACTIVE, which is the one wrong answer.

    Label cardinality is bounded by the fixed view grid (plan §41), so this
    never grows with the number of positions.
    """
    for name, health in model_health.items():
        ordinal = MODEL_HEALTH_ORDINAL.get(str(health))
        if ordinal is None:
            continue
        MODEL_HEALTH_STATE.set(float(ordinal), model=str(name))

#: Run triggers recorded in ``params.trigger``.
TRIGGER_SCHEDULED = "SCHEDULED"
TRIGGER_ON_DEMAND = "ON_DEMAND"


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _View:
    """One (model, confidence) pair of the validation grid.

    ``var_of`` / ``es_of`` take a rolling WINDOW of P&L (the observations
    strictly before the day being forecast) and return a ``ModelResult``.
    They are the seam that keeps every filter inside its own window.
    """

    key: str
    model_name: str
    model_version: str
    distribution: str
    confidence: float
    mode: str


def _window_min_obs(window: int) -> int:
    """``min_obs`` handed to an estimator running on a rolling window.

    The estimator's own default (60, or 250 at 99 %) is written for a FULL
    history, not for a rolling slice: at ``window=250`` the 99 % default would
    make every single forecast UNAVAILABLE and the whole 99 % row vacuous.
    The rolling window IS the sample, so the requirement is stated once here —
    every forecast must use the complete window, nothing less. A window that
    is short by even one observation would be a different (smaller) estimator
    than the one being validated.
    """
    return window


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestRowResult:
    """One scored view, before persistence — the shape the API block serves.

    Mirrors ``RiskModelBacktestRow`` one-to-one plus the display-only
    ``mode``. Every statistic is ``None`` when it could not be computed.
    """

    model_name: str
    model_version: str
    distribution: str
    confidence: float
    horizon_days: int
    window_obs: int
    n_forecasts: int
    exceedances: int
    rate: float | None
    expected_rate: float
    kupiec_lr: float | None
    kupiec_p: float | None
    christoffersen_lr: float | None
    christoffersen_p: float | None
    es_severity_ratio: float | None
    verdict: str
    health: str
    reason: str | None
    mode: str
    params: Mapping[str, object]

    def api(self) -> dict:
        """The design §9.4 ``statistical.validation.rows`` entry."""
        # ADDITIVE (§5): the tier of the model this row VALIDATES — derived
        # from the model name rather than stored, so a persisted row
        # (`_row_api`) and a freshly scored one cannot disagree about it. A
        # validation row has no tier of its own; it carries the underlying
        # model's.
        tier = tier_for_model_name(self.model_name)
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "distribution": self.distribution,
            "tier": str(tier) if tier is not None else None,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "window": self.window_obs,
            "n_forecasts": self.n_forecasts,
            "exceedances": self.exceedances,
            "rate": self.rate,
            "expected_rate": self.expected_rate,
            "kupiec_p": self.kupiec_p,
            "christoffersen_p": self.christoffersen_p,
            "es_severity_ratio": self.es_severity_ratio,
            "verdict": self.verdict,
            "health": self.health,
            "reason": self.reason,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ValidationRun:
    """One completed validation run: the scored rows plus their comparison.

    - ``rows``: one :class:`BacktestRowResult` per view, in grid order;
    - ``comparison``: the §63 EWMA-vs-GARCH dict (always present, honest
      nulls inside when a side is missing);
    - ``as_of``: the instant the run was made;
    - ``n_obs``: length of the book P&L series it ran on;
    - ``seconds``: measured wall time (design §9.4 "measure and log").
    """

    rows: tuple[BacktestRowResult, ...]
    comparison: Mapping[str, object]
    as_of: datetime
    n_obs: int
    window: int
    seconds: float

    def api(self) -> dict:
        """The design §9.4 ``statistical.validation`` block."""
        return {
            "mode": MODE_SHADOW,
            "as_of": self.as_of.isoformat(),
            "window": self.window,
            "min_forecasts": MIN_FORECASTS,
            "n_obs": self.n_obs,
            "rows": [row.api() for row in self.rows],
            "comparison": dict(self.comparison),
        }


# ---------------------------------------------------------------------------
# Walk-forward estimators (every filter runs INSIDE the window)
# ---------------------------------------------------------------------------


def _ewma_scaled_window(window_pnl: Sequence[float]) -> list[float]:
    """EWMA-filtered window — σ estimated on THIS window only (no look-ahead).

    ``volatility_scaled_pnl`` rescales each observation of the window to the
    window's own last-day σ forecast, which is precisely the forecast for the
    day being scored. Running the filter over the full series instead and
    slicing would leak the end of the sample backwards; that is the bug this
    function exists to prevent.
    """
    return volatility_scaled_pnl(
        window_pnl, lam=EWMA_LAMBDA, init_obs=EWMA_INIT_OBS
    )


class _GarchWindowFilter:
    """GARCH-filtered windows with a bounded refit stride (design §9.4).

    Refits the MLE on every ``refit_every``-th call and reuses the parameters
    in between, re-running only the (cheap, closed-form) variance recursion on
    the current window. Stateful by design — one instance per walk-forward
    pass, consumed in strict chronological order.

    ``last_fit`` keeps the most recent fit so the caller can report its health
    and reason honestly instead of inventing one, and ``n_fits`` records how
    many MLEs the run actually paid for.
    """

    def __init__(self, *, refit_every: int = GARCH_REFIT_EVERY,
                 min_obs: int = GARCH_MIN_OBS) -> None:
        if isinstance(refit_every, bool) or not isinstance(refit_every, int) or refit_every < 1:
            raise ValueError(f"refit_every must be an int >= 1, got {refit_every!r}")
        self.refit_every = refit_every
        self.min_obs = min_obs
        self.step = 0
        self.n_fits = 0
        self.params = None
        self.last_fit: GarchFit | None = None
        self.n_failed = 0

    def scaled(self, window_pnl: Sequence[float]) -> list[float]:
        """The window rescaled to its own one-step-ahead GARCH σ.

        Empty when no usable parameterisation exists (the caller turns that
        into an UNAVAILABLE forecast, never a fabricated number).
        """
        # The stride is unconditional. Refitting because the LAST attempt
        # failed would make a pathological series (a flat book, say) pay the
        # full per-day MLE cost — exactly the runtime blow-up the stride
        # exists to prevent — while producing nothing usable either way.
        refit = self.step % self.refit_every == 0
        self.step += 1
        if refit:
            fit = fit_garch(window_pnl, min_obs=self.min_obs)
            self.n_fits += 1
            self.last_fit = fit
            if fit.health is not ModelHealth.ACTIVE:
                # §65: every MLE that did not yield a usable ACTIVE fit is
                # counted, including one that still produced parameters but
                # is DEGRADED — the operator's question is "how often is
                # GARCH not giving us a clean forecast", and a DEGRADED fit
                # that silently kept scoring is precisely the case a
                # failures-only counter would hide.
                GARCH_FIT_FAILURES_TOTAL.inc(
                    site="validation", health=str(fit.health)
                )
            if fit.params is None:
                self.params = None
                self.n_failed += 1
                return []
            self.params = fit.params
        if self.params is None:
            # No usable parameterisation yet (the stride's last MLE produced
            # none). An empty series makes the estimator UNAVAILABLE, so the
            # day is SKIPPED in the exceedance report — never scored against
            # a fabricated forecast.
            return []
        # Reuse (or the fresh) parameters, but always re-run the recursion on
        # THIS window so sigma_now is the current day's forecast.
        scaling = garch_scaling(
            window_pnl,
            min_obs=self.min_obs,
            fit=_refit_with(self.params, window_pnl, self.min_obs),
        )
        if scaling.sigma_now is None:
            return []
        return list(scaling.scaled)


def _refit_with(params, window_pnl: Sequence[float], min_obs: int) -> GarchFit:
    """A :class:`GarchFit` for ``window_pnl`` at FIXED ``params``.

    The variance recursion is re-run on the current window (so ``sigma2_next``
    is today's forecast) while ω, α, β stay the ones the last MLE produced.
    That is the whole point of the refit stride: the expensive part is the
    optimisation, not the recursion.
    """
    from libs.trading_core.risk.models.garch import garch_variance_path

    values = [float(v) for v in window_pnl]
    n = len(values)
    # Seed with the window's own second moment — the same backcast fit_garch
    # uses, so a reused-parameter recursion and a fresh fit agree on t=0.
    path = garch_variance_path(values, params)
    sigma2_series = tuple(path[:n])
    sigma2_next = path[n]
    std_resid = tuple(
        v / math.sqrt(s) if s > 0.0 else 0.0
        for v, s in zip(values, sigma2_series)
    )
    return GarchFit(
        params=params,
        loglik=None,
        converged=True,
        iterations=0,
        persistence=params.persistence,
        unconditional_var=params.unconditional_variance,
        half_life_days=params.half_life_days,
        sigma2_series=sigma2_series,
        std_residuals=std_resid,
        n=n,
        health=ModelHealth.DEGRADED,
        reason=(
            "parameters reused from an earlier refit (bounded-runtime stride, "
            f"every {GARCH_REFIT_EVERY} forecast steps); the variance "
            "recursion was re-run on this window"
        ),
        diagnostics={"sigma2_next": sigma2_next, "n": n, "reused_params": True},
    )


# ---------------------------------------------------------------------------
# Scoring one view
# ---------------------------------------------------------------------------


def _score_view(
    view: _View,
    pnl: Sequence[float],
    *,
    window: int,
    dates: Sequence[date] | None,
    params_extra: Mapping[str, object],
    var_estimator,
    es_estimator,
    unavailable_reason: str | None = None,
) -> BacktestRowResult:
    """Walk-forward one view and turn its exceedance report into a row.

    ``unavailable_reason`` short-circuits the whole pass (used when the GARCH
    fit was not ACTIVE): the row is still produced, with zero forecasts and
    the real reason, because a skipped row would read as "never run".
    """
    backtest_params = BacktestParams(
        min_forecasts=MIN_FORECASTS,
        green_p=GREEN_P,
        red_p=RED_P,
        cluster_p=CLUSTER_P,
    )
    base_params: dict[str, object] = {
        "window": window,
        "min_forecasts": MIN_FORECASTS,
        "green_p": GREEN_P,
        "red_p": RED_P,
        "cluster_p": CLUSTER_P,
        "runner_version": RUNNER_VERSION,
        "validation_version": VALIDATION_VERSION,
        "mode": view.mode,
        "walk_forward": True,
        **dict(params_extra),
    }

    if unavailable_reason is not None:
        return BacktestRowResult(
            model_name=view.model_name,
            model_version=view.model_version,
            distribution=view.distribution,
            confidence=view.confidence,
            horizon_days=HORIZON_DAYS,
            window_obs=window,
            n_forecasts=0,
            exceedances=0,
            rate=None,
            expected_rate=1.0 - view.confidence,
            kupiec_lr=None,
            kupiec_p=None,
            christoffersen_lr=None,
            christoffersen_p=None,
            es_severity_ratio=None,
            verdict=str(BacktestVerdict.UNAVAILABLE),
            health=str(ModelHealth.UNAVAILABLE),
            reason=unavailable_reason,
            mode=view.mode,
            params=base_params,
        )

    var_series = walk_forward(
        pnl,
        window=window,
        estimator=var_estimator,
        confidence=view.confidence,
        dates=dates,
    )
    es_series = walk_forward(
        pnl,
        window=window,
        estimator=es_estimator,
        confidence=view.confidence,
        dates=dates,
    )
    report = exceedances(
        var_series,
        confidence=view.confidence,
        es_forecasts=es_series,
        params=backtest_params,
    )
    base_params["n_skipped"] = report.n_skipped
    base_params["transitions"] = list(report.transitions) if report.transitions else None
    base_params["conditional_coverage_p"] = report.conditional_coverage_p
    base_params["clustered"] = report.clustered
    base_params["es_n"] = report.es_n

    return BacktestRowResult(
        model_name=view.model_name,
        model_version=view.model_version,
        distribution=view.distribution,
        confidence=view.confidence,
        horizon_days=HORIZON_DAYS,
        window_obs=window,
        n_forecasts=report.n,
        exceedances=report.x,
        rate=report.rate,
        expected_rate=report.expected_rate,
        kupiec_lr=report.kupiec_lr,
        kupiec_p=report.kupiec_p,
        christoffersen_lr=report.christoffersen_lr,
        christoffersen_p=report.christoffersen_p,
        es_severity_ratio=report.es_severity_ratio,
        verdict=str(report.verdict),
        health=str(report.health),
        reason=report.reason,
        mode=view.mode,
        params=base_params,
    )


# ---------------------------------------------------------------------------
# §63 comparison
# ---------------------------------------------------------------------------


def ewma_vs_garch(rows: Sequence[BacktestRowResult]) -> dict:
    """The §63 EWMA-vs-GARCH comparison dict (design §9.4).

    Reports both Kupiec p-values, which model the numbers currently favour,
    whether the promotion bar is cleared, and the criterion sentence VERBATIM
    so the bar cannot drift between runs. ``preferred`` is ``None`` when
    either side has no p-value — a comparison with a missing half is not a
    preference, and saying "EWMA" by default would be a fabricated verdict.

    NOTHING here promotes anything. The GARCH view stays RESEARCH until a
    USER moves it; this dict is the evidence they would weigh.
    """
    by_model = {row.model_name: row for row in rows if row.confidence == CONFIDENCE_95}
    ewma = by_model.get(COMPARISON_EWMA_KEY)
    garch = by_model.get(COMPARISON_GARCH_KEY)

    ewma_p = ewma.kupiec_p if ewma is not None else None
    garch_p = garch.kupiec_p if garch is not None else None
    garch_cp = garch.christoffersen_p if garch is not None else None
    garch_n = garch.n_forecasts if garch is not None else 0

    preferred: str | None = None
    if ewma_p is not None and garch_p is not None:
        # "Preferred" = the better-calibrated coverage, ties to the INCUMBENT
        # (EWMA): a research model does not win a tie against the model that
        # is already the platform's conditional forecaster.
        preferred = COMPARISON_GARCH_KEY if garch_p > ewma_p else COMPARISON_EWMA_KEY

    unmet: list[str] = []
    if garch is None or garch_p is None:
        unmet.append("GARCH view produced no Kupiec p-value")
    else:
        if garch_n < COMPARISON_MIN_FORECASTS:
            unmet.append(
                f"n_forecasts={garch_n} < {COMPARISON_MIN_FORECASTS} required forecast days"
            )
        if ewma_p is None:
            unmet.append("EWMA view produced no Kupiec p-value to compare against")
        elif garch_p < ewma_p:
            unmet.append(f"GARCH Kupiec p={garch_p:.4g} < EWMA p={ewma_p:.4g}")
        if garch_cp is None or garch_cp < CLUSTER_P:
            unmet.append(
                "GARCH Christoffersen p="
                + ("None" if garch_cp is None else f"{garch_cp:.4g}")
                + f" < {CLUSTER_P}"
            )
        if garch.health == str(ModelHealth.FAILED):
            unmet.append("GARCH diagnostics FAILED in the window")

    return {
        "ewma_kupiec_p": ewma_p,
        "garch_kupiec_p": garch_p,
        "garch_christoffersen_p": garch_cp,
        "garch_n_forecasts": garch_n,
        "preferred": preferred,
        "criterion": COMPARISON_CRITERION,
        "criterion_met": not unmet,
        "criterion_unmet_reasons": unmet,
        "promotion": (
            "NONE — GARCH stays RESEARCH; promotion is a user action (§63)."
        ),
    }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def compute_model_backtests(
    book_pnl: Sequence[float],
    *,
    dates: Sequence[date] | None = None,
    window: int = DEFAULT_WINDOW,
    as_of: datetime | None = None,
) -> ValidationRun:
    """Score the whole view grid walk-forward. PURE — no session, no I/O.

    Split out from :func:`run_model_backtests` so the arithmetic is testable
    (and timeable) without a database, exactly as the design's persistence
    pattern separates measuring from persisting.

    ``window`` observations are consumed before the first forecast, so a
    series of ``n`` produces at most ``n - window`` forecasts; below
    ``MIN_FORECASTS`` of them every row is an honest UNAVAILABLE with the
    real numbers.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError(f"window must be an int >= 2, got {window!r}")
    values = [float(v) for v in book_pnl]
    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise ValueError(f"book_pnl[{i}] must be finite, got {v!r}")
    if dates is not None and len(dates) != len(values):
        raise ValueError(
            f"dates length {len(dates)} must equal book_pnl length {len(values)}"
        )
    started = time.perf_counter()
    now = as_of or datetime.now(timezone.utc)
    min_obs = _window_min_obs(window)

    rows: list[BacktestRowResult] = []

    # --- unconditional views: historical + Gaussian, 95 % and 99 % --------
    for confidence in (CONFIDENCE_95, CONFIDENCE_99):
        rows.append(
            _score_view(
                _View(
                    key=f"{MODEL_HISTORICAL_VAR}:{confidence}",
                    model_name=MODEL_HISTORICAL_VAR,
                    model_version="1.0.0",
                    distribution=DISTRIBUTION_EMPIRICAL,
                    confidence=confidence,
                    mode=MODE_SHADOW,
                ),
                values,
                window=window,
                dates=dates,
                params_extra={"estimator": "historical", "min_obs": min_obs},
                var_estimator=(
                    lambda w, c=confidence: historical_var(
                        w, c, HORIZON_DAYS, min_obs=min_obs
                    )
                ),
                es_estimator=(
                    lambda w, c=confidence: historical_es(
                        w, c, HORIZON_DAYS, min_obs=min_obs
                    )
                ),
            )
        )
    for confidence in (CONFIDENCE_95, CONFIDENCE_99):
        rows.append(
            _score_view(
                _View(
                    key=f"{MODEL_GAUSSIAN_VAR}:{confidence}",
                    model_name=MODEL_GAUSSIAN_VAR,
                    model_version="1.0.0",
                    distribution=DISTRIBUTION_NORMAL,
                    confidence=confidence,
                    mode=MODE_SHADOW,
                ),
                values,
                window=window,
                dates=dates,
                params_extra={"estimator": "gaussian", "min_obs": min_obs},
                var_estimator=(
                    lambda w, c=confidence: gaussian_var(
                        w, c, HORIZON_DAYS, min_obs=min_obs
                    )
                ),
                es_estimator=(
                    lambda w, c=confidence: gaussian_es(
                        w, c, HORIZON_DAYS, min_obs=min_obs
                    )
                ),
            )
        )

    # --- EWMA-filtered conditional view (95 %) ----------------------------
    # The filter runs INSIDE each window (see module docstring). The scaled
    # window is SHORTER than the raw one (the EWMA warm-up is dropped), so
    # the estimator's min_obs must be the scaled length, not the window.
    ewma_min_obs = max(2, min_obs - EWMA_INIT_OBS)

    def _ewma_var(w: Sequence[float]) -> object:
        return historical_var(
            _ewma_scaled_window(w), CONFIDENCE_95, HORIZON_DAYS, min_obs=ewma_min_obs
        )

    def _ewma_es(w: Sequence[float]) -> object:
        return historical_es(
            _ewma_scaled_window(w), CONFIDENCE_95, HORIZON_DAYS, min_obs=ewma_min_obs
        )

    rows.append(
        _score_view(
            _View(
                key=f"{MODEL_CONDITIONAL_VAR}:{CONFIDENCE_95}",
                model_name=MODEL_CONDITIONAL_VAR,
                model_version="1.0.0",
                distribution=DISTRIBUTION_EMPIRICAL_VOL_SCALED,
                confidence=CONFIDENCE_95,
                mode=MODE_SHADOW,
            ),
            values,
            window=window,
            dates=dates,
            params_extra={
                "estimator": "ewma_filtered_hs",
                "lambda": EWMA_LAMBDA,
                "init_obs": EWMA_INIT_OBS,
                "min_obs": ewma_min_obs,
                "filter_scope": "PER_WINDOW",
            },
            var_estimator=_ewma_var,
            es_estimator=_ewma_es,
        )
    )

    # --- GARCH-filtered conditional view (95 %, RESEARCH) -----------------
    # The design's "skip with reason if the fit is not ACTIVE" is enforced
    # PER WINDOW, not by one probe on the last window: each rolling window
    # fits (or reuses) its own parameters, and a window whose fit produced
    # nothing yields an empty filtered series ⇒ an UNAVAILABLE forecast for
    # that day ⇒ a SKIPPED pair in the exceedance report. That is both the
    # honest answer and the walk-forward one — a probe on the LAST window
    # would let the most recent days decide whether the earliest forecasts
    # are reported, which is precisely the hindsight §43 forbids.
    #
    # The whole row degrades to UNAVAILABLE only in the two cases where no
    # window could ever fit: no forecast day exists at all, or the window is
    # shorter than the GARCH minimum sample.
    garch_reason: str | None = None
    if len(values) <= window:
        garch_reason = (
            f"n={len(values)} <= window={window}: no forecast day exists to score"
        )
    elif window < GARCH_MIN_OBS:
        garch_reason = (
            f"window={window} < garch_min_obs={GARCH_MIN_OBS}: no rolling window "
            "is long enough to fit GARCH(1,1) — the EWMA-filtered view remains "
            "the conditional forecaster (spec §13/§58 fallback)"
        )

    garch_filter = _GarchWindowFilter()

    def _garch_var(w: Sequence[float]) -> object:
        return historical_var(
            garch_filter.scaled(w), CONFIDENCE_95, HORIZON_DAYS, min_obs=min_obs
        )

    def _garch_es(w: Sequence[float]) -> object:
        # A SECOND pass over the same windows would refit on a different
        # stride phase, so the ES filter gets its own instance seeded
        # identically — deterministic and aligned with the VaR pass.
        return historical_es(
            garch_es_filter.scaled(w), CONFIDENCE_95, HORIZON_DAYS, min_obs=min_obs
        )

    garch_es_filter = _GarchWindowFilter()

    rows.append(
        _score_view(
            _View(
                key=f"{MODEL_GARCH_VAR}:{CONFIDENCE_95}",
                model_name=MODEL_GARCH_VAR,
                model_version="1.0.0",
                distribution=DISTRIBUTION_EMPIRICAL_GARCH_SCALED,
                confidence=CONFIDENCE_95,
                mode=MODE_RESEARCH,
            ),
            values,
            window=window,
            dates=dates,
            params_extra={
                "estimator": "garch_filtered_hs",
                "refit_every": GARCH_REFIT_EVERY,
                "garch_min_obs": GARCH_MIN_OBS,
                "min_obs": min_obs,
                "filter_scope": "PER_WINDOW",
                "refit_note": (
                    "parameters are refit every "
                    f"{GARCH_REFIT_EVERY} forecast steps and reused in between "
                    "(bounded runtime, design §9.4); the variance recursion is "
                    "re-run on every window, so between refits the parameters "
                    "are staler than a live model's, never fresher"
                ),
            },
            var_estimator=_garch_var,
            es_estimator=_garch_es,
            unavailable_reason=garch_reason,
        )
    )

    # The fit count is only known after both passes have run; the row is
    # frozen, so rebuild it with the completed params rather than mutating.
    garch_row = rows[-1]
    rows[-1] = dataclasses.replace(
        garch_row,
        params={
            **dict(garch_row.params),
            "n_garch_fits": garch_filter.n_fits + garch_es_filter.n_fits,
            "n_garch_windows_unfittable": (
                garch_filter.n_failed + garch_es_filter.n_failed
            ),
            "garch_last_fit_health": (
                str(garch_filter.last_fit.health)
                if garch_filter.last_fit is not None
                else None
            ),
            "garch_last_fit_reason": (
                garch_filter.last_fit.reason
                if garch_filter.last_fit is not None
                else None
            ),
        },
    )

    seconds = time.perf_counter() - started
    RISK_VALIDATION_SECONDS.observe(seconds)
    return ValidationRun(
        rows=tuple(rows),
        comparison=ewma_vs_garch(rows),
        as_of=now,
        n_obs=len(values),
        window=window,
        seconds=seconds,
    )


async def run_model_backtests(
    session: AsyncSession,
    *,
    book_pnl: Sequence[float],
    dates: Sequence[date] | None = None,
    nav: float | None = None,
    snapshot_id: int | None = None,
    window: int = DEFAULT_WINDOW,
    as_of: datetime | None = None,
    trigger: str = TRIGGER_ON_DEMAND,
) -> ValidationRun:
    """Score the grid and PERSIST one ``risk_model_backtests`` row per view.

    ``session.add`` + ``flush`` only — **the CALLER commits** (the
    ``risk_snapshot._persist`` pattern), so a SCHEDULED run lands in the same
    transaction as the snapshot that triggered it. Writes NO audit event: a
    validation run is a measurement, not a decision.

    ``nav`` is accepted and recorded in ``params`` for context (the rows are
    USD-denominated, and knowing the book size behind a $ VaR is what makes
    an old row interpretable) — no number here is divided by it.

    Every view produces a row, including UNAVAILABLE ones with their reason:
    a missing row would later read as "never run" (spec §56, §44 rule 18).
    """
    run = compute_model_backtests(
        book_pnl, dates=dates, window=window, as_of=as_of
    )
    for row in run.rows:
        session.add(
            RiskModelBacktestRow(
                as_of=run.as_of,
                snapshot_id=snapshot_id,
                model_name=row.model_name,
                model_version=row.model_version,
                distribution=row.distribution,
                confidence=row.confidence,
                horizon_days=row.horizon_days,
                window_obs=row.window_obs,
                n_forecasts=row.n_forecasts,
                exceedances=row.exceedances,
                rate=row.rate,
                expected_rate=row.expected_rate,
                kupiec_lr=row.kupiec_lr,
                kupiec_p=row.kupiec_p,
                christoffersen_lr=row.christoffersen_lr,
                christoffersen_p=row.christoffersen_p,
                es_severity_ratio=row.es_severity_ratio,
                verdict=row.verdict,
                health=row.health,
                reason=row.reason,
                params={
                    **dict(row.params),
                    "trigger": trigger,
                    "nav": nav,
                    "n_obs": run.n_obs,
                    "seconds": round(run.seconds, 4),
                    "comparison": (
                        dict(run.comparison)
                        if row.model_name in (COMPARISON_EWMA_KEY, COMPARISON_GARCH_KEY)
                        else None
                    ),
                },
            )
        )
        # §65: count the exceedances THIS row reports, as the row is
        # persisted — the counter and the stored history can never disagree.
        #
        # The grid scores one row per VaR view, each against its OWN matching
        # ES estimator (module docstring), so a row carries both numbers:
        #
        #   var_exceedances := report.x   — forecast days whose realised loss
        #                                    exceeded the VaR forecast;
        #   es_exceedances  := params.es_n — of those days, the ones the ES
        #                                    severity ratio was actually
        #                                    computed on (an exceedance whose
        #                                    ES forecast was UNAVAILABLE is
        #                                    counted by the VaR test but
        #                                    scored by neither).
        #
        # The two are equal on a clean run and diverge exactly when ES
        # forecasts went missing — which is the divergence worth alerting on,
        # and the reason these are two counters rather than one.
        #
        # UNAVAILABLE rows report `n_forecasts=0` with `exceedances=0`: no day
        # was scored, so nothing is added. Incrementing by that 0 would be
        # harmless arithmetic but is skipped for clarity of intent — "no run"
        # and "a run with no breach" must stay distinguishable in the logs
        # even though they coincide in the counter.
        if row.n_forecasts:
            labels = {"model": row.model_name, "confidence": str(row.confidence)}
            VAR_EXCEEDANCES_TOTAL.inc(float(row.exceedances), **labels)
            es_n = row.params.get("es_n")
            if isinstance(es_n, int) and not isinstance(es_n, bool):
                ES_EXCEEDANCES_TOTAL.inc(float(es_n), **labels)
    await session.flush()
    RISK_VALIDATION_RUNS_TOTAL.inc(trigger=trigger)
    logger.info(
        "risk_validation_run",
        extra={
            "extra_fields": {
                "trigger": trigger,
                "snapshot_id": snapshot_id,
                "n_obs": run.n_obs,
                "window": window,
                "n_rows": len(run.rows),
                "seconds": round(run.seconds, 4),
                "verdicts": {r.model_name + f":{r.confidence}": r.verdict for r in run.rows},
            }
        },
    )
    return run


# ---------------------------------------------------------------------------
# Reads (the API block NEVER recomputes — design §9.4)
# ---------------------------------------------------------------------------


async def latest_backtest_rows(
    session: AsyncSession,
) -> list[RiskModelBacktestRow]:
    """The rows of the NEWEST validation run, in grid order.

    "Newest run" = the newest ``as_of`` instant; all of that run's rows are
    returned together so a reader never mixes a fresh historical row with a
    stale GARCH one. Empty list when nothing has ever been run — an honest
    "never validated", not a fabricated GREEN.
    """
    newest = (
        await session.execute(
            select(RiskModelBacktestRow.as_of)
            .order_by(RiskModelBacktestRow.as_of.desc(), RiskModelBacktestRow.id.desc())
            .limit(1)
        )
    ).scalars().first()
    if newest is None:
        return []
    rows = (
        await session.execute(
            select(RiskModelBacktestRow)
            .where(RiskModelBacktestRow.as_of == newest)
            .order_by(RiskModelBacktestRow.id.asc())
        )
    ).scalars().all()
    return list(rows)


def _as_utc(value: datetime | None) -> datetime | None:
    """Stored instants are UTC; SQLite hands them back NAIVE.

    The repo-wide convention (``risk_snapshot._scheduled_exists_today`` does
    the same): a tz-naive column value IS UTC, so stamp it rather than let a
    consumer read it in local time and silently shift the run by hours.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _row_api(row: RiskModelBacktestRow) -> dict:
    """One persisted row on the wire (design §9.4).

    Key-for-key identical to :meth:`BacktestRowResult.api` — the freshly
    scored row and the replayed one must serialise the same shape, or the
    UI would have to know which path it is reading. The §5 ``tier`` is
    derived from ``model_name`` on BOTH paths for the same reason: derived
    on read, it cannot go stale against a re-classified model, and the two
    paths cannot disagree about it.
    """
    params = row.params if isinstance(row.params, dict) else {}
    tier = tier_for_model_name(row.model_name)
    return {
        "model_name": row.model_name,
        "model_version": row.model_version,
        "distribution": row.distribution,
        "tier": str(tier) if tier is not None else None,
        "confidence": row.confidence,
        "horizon_days": row.horizon_days,
        "window": row.window_obs,
        "n_forecasts": row.n_forecasts,
        "exceedances": row.exceedances,
        "rate": row.rate,
        "expected_rate": row.expected_rate,
        "kupiec_p": row.kupiec_p,
        "christoffersen_p": row.christoffersen_p,
        "es_severity_ratio": row.es_severity_ratio,
        "verdict": row.verdict,
        "health": row.health,
        "reason": row.reason,
        "mode": params.get("mode", MODE_SHADOW),
    }


def validation_api_from_rows(rows: Sequence[RiskModelBacktestRow]) -> dict | None:
    """The design §9.4 ``statistical.validation`` block, from PERSISTED rows.

    ``None`` when nothing has been persisted — the read view says "no
    validation run yet" rather than running one, because a page load must
    never pay for a walk-forward backtest (design §9.4: "never recomputed on
    a page read") and a fabricated block would claim a validation that never
    happened.

    The comparison is read off whichever row carries it (the runner stamps it
    on the two conditional views), never recomputed here.
    """
    if not rows:
        return None
    comparison: dict | None = None
    for row in rows:
        params = row.params if isinstance(row.params, dict) else {}
        candidate = params.get("comparison")
        if isinstance(candidate, dict):
            comparison = candidate
            break
    first = rows[0]
    first_params = first.params if isinstance(first.params, dict) else {}
    as_of = _as_utc(first.as_of)
    return {
        "mode": MODE_SHADOW,
        "as_of": as_of.isoformat() if as_of is not None else None,
        "window": first.window_obs,
        "min_forecasts": first_params.get("min_forecasts", MIN_FORECASTS),
        "n_obs": first_params.get("n_obs"),
        "rows": [_row_api(row) for row in rows],
        "comparison": comparison,
    }


async def validation_api(session: AsyncSession) -> dict | None:
    """``statistical.validation`` for the risk view — a pure READ.

    Reads the newest persisted run and serialises it. Never computes, never
    writes, never raises for a missing run (``None`` is the honest answer).
    """
    return validation_api_from_rows(await latest_backtest_rows(session))


def red_verdict_count(
    rows: Sequence[RiskModelBacktestRow] | Sequence[BacktestRowResult],
    *,
    core_models: Sequence[str] = (MODEL_HISTORICAL_VAR,),
) -> tuple[int, tuple[str, ...]]:
    """RED verdicts on CORE views → ``(count, reasons)`` for model risk.

    Design §9.4: "a RED verdict on a core view counts as one trigger". CORE
    is a POLICY choice, not a statistic — by default only the historical VaR
    views, because those are the views the platform's own risk numbers are
    read off. A RED on the RESEARCH GARCH view is interesting but must not
    raise the platform's model-risk state: nothing consumes it.

    Returns the number of RED core rows and one reason string each, with the
    real rate against the expected one so the trigger is auditable.
    """
    count = 0
    reasons: list[str] = []
    core = set(core_models)
    for row in rows:
        if row.model_name not in core:
            continue
        if str(row.verdict) != str(BacktestVerdict.RED):
            continue
        count += 1
        rate = "None" if row.rate is None else f"{row.rate:.3g}"
        kp = "None" if row.kupiec_p is None else f"{row.kupiec_p:.3g}"
        reasons.append(
            f"backtest RED on {row.model_name} @ {row.confidence:g}: "
            f"exceedance rate {rate} vs expected {row.expected_rate:.3g} "
            f"over n={row.n_forecasts} (Kupiec p={kp})"
        )
    return count, tuple(reasons)


__all__ = [
    "COMPARISON_CRITERION",
    "DEFAULT_WINDOW",
    "GARCH_REFIT_EVERY",
    "MIN_FORECASTS",
    "MODEL_CONDITIONAL_VAR",
    "MODEL_GARCH_VAR",
    "MODEL_GAUSSIAN_VAR",
    "MODEL_HISTORICAL_VAR",
    "TRIGGER_ON_DEMAND",
    "TRIGGER_SCHEDULED",
    "BacktestRowResult",
    "ValidationRun",
    "compute_model_backtests",
    "ewma_vs_garch",
    "latest_backtest_rows",
    "red_verdict_count",
    "run_model_backtests",
    "validation_api",
    "validation_api_from_rows",
]
