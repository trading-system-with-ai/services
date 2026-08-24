"""Risk Engine v0 (development plan §12, §13, §17) + Phase B statistical library.

Pure, deterministic library code — no DB, no FastAPI — architecturally
independent from the strategy engine (plan §17): it receives a request plus
a portfolio snapshot and decides; it never computes signals itself. Risk
limits have PRIORITY over strategy confidence (plan §44 rule 20).

Two layers live here, and the boundary is load-bearing:

- **Tier 0 decision engine** (``engine.py``, ``liquidity.py``) — the only
  code that can gate a trade. Byte-identical across Phase B.
- **Phase B statistical library** (``returns``, ``pnl_series``, ``models/*``,
  ``validation``, ``snapshot``) — SHADOW/RESEARCH per spec §70: it measures
  and reports (VaR, ES, volatility, contributions, drawdown, model risk) and
  can never veto. Every model defaults to ``ModelMode.SHADOW``; the engine
  would have to consult ``mode`` before wiring one into a decision.
- **Phase C pre-trade layer** (``pretrade``) — SHADOW too (spec §70): it
  compares the book before and after a proposed trade, derives hypothetical
  ``QuantityCap`` rows from ``StatisticalLimits`` (RESEARCH DEFAULTS,
  UNVALIDATED) and reports the verdict the statistical layer ALONE would
  have reached. It becomes binding only where a caller deliberately passes
  those caps to ``assess(extra_caps=...)`` — an explicit promotion step.
- **Phase D stress layer** (``models/stress``, with
  ``options/iv`` + ``options/reval`` underneath) — SHADOW as well. It
  revalues the option book FULLY under historical, hypothetical and
  user-defined scenarios (spec §21–§27) and derives a hypothetical
  ``QuantityCap`` on the ``"STRESS"`` layer from ``StressLimits``
  (RESEARCH DEFAULT, UNVALIDATED). Spec §27 grants the stress test veto
  authority; granting it is the PRODUCTION promotion step, not this import.
- **Phase E conditional volatility** (``optim``, ``models/garch``,
  ``models/_chi2``) — RESEARCH, a step BELOW SHADOW (spec §70). GARCH(1,1)
  is fitted, diagnosed (Ljung–Box on standardized residuals², persistence,
  half-life) and reported, but EWMA remains the conditional-volatility
  forecaster; ``conditional_volatility_source`` implements the §13/§58
  fallback hierarchy in one place and always names which model produced the
  number. Promotion to SHADOW requires the §63 exceedance comparison and is
  a user action.

Import-cycle note: ``correlation.py`` imports ``risk.returns``, so this
package must never import ``correlation`` at module level. The Phase B
modules import only ``base``/``returns``/each other, never this ``__init__``
or ``engine``. ``pretrade`` follows the same rule and imports no ``engine``
either: the dependency runs one way, which is why ``assess`` types its
``extra_caps`` structurally (``ExtraCap``) instead of importing
``QuantityCap``.
"""
from .engine import (  # noqa: F401
    LAYER_HARD_LIMIT,
    BindingConstraint,
    ExtraCap,
    IncomeRiskRequest,
    PortfolioSnapshot,
    PositionRisk,
    RiskAssessment,
    RiskLimits,
    RiskRequest,
    assess,
    assess_income,
    heat_state,
    portfolio_heat,
)
from .liquidity import (  # noqa: F401  (audit §7.3 / B0: REPORT-mode gate)
    LiquidityLimits,
    LiquidityReport,
    evaluate_underlying_liquidity,
)

# --- Phase B: returns layer (contract §2.1) --------------------------------
from .returns import (  # noqa: F401
    ReturnMatrix,
    ReturnSeries,
    ReturnType,
    align,
    log_returns,
    returns_from_closes,
    simple_returns,
)

# --- Phase B: book P&L construction (contract §2.9) ------------------------
from .pnl_series import (  # noqa: F401
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL_CONST_IV,
    BookPnl,
    PositionRiskInput,
    book_method_summary,
    book_pnl_series,
    position_pnl_series,
)

# --- Phase B: models (contract §2.2–§2.8) ----------------------------------
# Importing the package registers the four VaR/ES models (idempotent).
from .models import (  # noqa: F401
    REGISTRY,
    BaseRiskModel,
    ContributionResult,
    CovarianceResult,
    DispersionResult,
    DistributionResult,
    DrawdownResult,
    GaussianESModel,
    GaussianVaRModel,
    HistoricalESModel,
    HistoricalVaRModel,
    ModelHealth,
    ModelMeta,
    ModelMode,
    ModelResult,
    ModelRiskState,
    PositionContribution,
    RiskModel,
    conditional_es,
    conditional_var,
    dispersion,
    distribution_diagnostics,
    drawdown,
    es_contributions,
    ewma_variance,
    ewma_volatility_forecast,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    incremental_es,
    marginal_es,
    model_risk_state,
    portfolio_volatility,
    reconstructed_book_drawdown,
    sample_covariance,
    tail_size,
    volatility_contributions,
    volatility_scaled_pnl,
)

# --- Single-factor (SPY) diagnostic (spec §11) — RESEARCH ------------------
# Compliance §3 row 11 records the audit's unbuilt "SPY-β single-factor
# diagnostic"; this is it. RESEARCH, a step below SHADOW: it registers no
# model, derives no ``QuantityCap`` and is not consulted by ``assess`` — the
# §11 ``max_factor_...`` concentration cap stays REJECT-documented, because
# a cap needs a validated taxonomy and this is one regression against one
# proxy series.
from .models.factor import (  # noqa: F401
    DEFAULT_FACTOR,
    FactorParams,
    FactorRiskResult,
    PositionBeta,
    beta_vs_factor,
    factor_risk_share,
)

# --- Phase D: stress engine (design §8.3) — SHADOW -------------------------
# Scenario revaluation of the CURRENT book (full revaluation of every option
# leg that has an IV) plus the hypothetical STRESS cap. Spec §27 gives the
# stress test veto authority; that veto is the PRODUCTION promotion, not
# this import — ``StressLimits.mode`` is ``"SHADOW"`` and nothing here
# reaches ``assess`` unless a caller passes the cap as ``extra_caps``.
from .models.stress import (  # noqa: F401
    CATALOGUE_VERSION,
    CODE_STRESS_LOSS,
    DEFAULT_HISTORICAL_WINDOWS,
    DEFAULT_HYPOTHETICAL_SCENARIOS,
    IV_SHOCK_SOURCE_RV_PROXY,
    IV_SHOCK_SOURCE_SPECIFIED,
    KIND_HISTORICAL,
    KIND_HYPOTHETICAL,
    KIND_IV_GRID,
    KIND_USER,
    HistoricalShockParams,
    HistoricalWindow,
    Scenario,
    ScenarioResult,
    StressLimits,
    StressResult,
    auto_worst_windows,
    default_scenarios,
    historical_shocks_from_closes,
    run_scenario,
    run_stress,
    stress_caps,
)

# --- Phase E: GARCH(1,1) conditional volatility (design §9.3) — RESEARCH ---
# One step below SHADOW (spec §70): the fitted model is studied and
# displayed, EWMA remains the conditional-volatility forecaster, and
# ``conditional_volatility_source`` implements the §13/§58 fallback in one
# place so the number always names its own model. ``nelder_mead`` and
# ``chi2_sf`` are numerical primitives, not risk models — no thresholds.
from .optim import NMResult, nelder_mead  # noqa: F401
from .models.garch import (  # noqa: F401
    DISTRIBUTION_EMPIRICAL_GARCH_SCALED,
    DISTRIBUTION_GAUSSIAN_GARCH,
    MODEL_NAME_GARCH,
    SOURCE_EWMA,
    SOURCE_GARCH,
    Garch11Model,
    GarchFit,
    GarchParams,
    GarchScaling,
    conditional_scaled_pnl_source,
    conditional_volatility_source,
    fit_garch,
    garch_forecast_variance,
    garch_scaled_pnl,
    garch_scaling,
    garch_variance_path,
    garch_volatility_forecast,
    ljung_box,
)
from .models._chi2 import chi2_cdf, chi2_sf  # noqa: F401

# --- Phase B: walk-forward backtest (contract §2.10) -----------------------
from .validation import (  # noqa: F401
    BacktestParams,
    BacktestVerdict,
    ExceedanceReport,
    ForecastSeries,
    christoffersen_independence,
    exceedances,
    kupiec_pof,
    volatility_forecast_error,
    walk_forward,
)

# --- Phase C: pre-trade portfolio risk (contract §7.1–§7.2) ----------------
# SHADOW: these produce hypothetical verdicts and caps; nothing here changes
# a Tier 0 decision unless a caller passes the caps to ``assess(extra_caps=)``.
from .pretrade import (  # noqa: F401
    CAP_LAYERS,
    CODE_BUCKET_ES_CONTRIBUTION,
    CODE_ES_CONTRIBUTION,
    CODE_INCREMENTAL_ES,
    CODE_PORTFOLIO_ES,
    LAYER_CONCENTRATION,
    LAYER_STATISTICAL,
    LAYER_STRESS,
    MODE_PRODUCTION,
    MODE_SHADOW,
    CandidateSpec,
    MetricPair,
    QuantityCap,
    RiskComparison,
    ShadowVerdict,
    StatisticalLimits,
    compare,
    proposed_book,
    shadow_verdict,
    statistical_caps,
)

# --- Phase K: event risk (event spec §62–§67) — SHADOW ---------------------
# Discrete JUMP risk around a scheduled catalyst, beside (never inside) the
# statistical layer. §63's state is assigned by a documented deterministic
# TABLE — no LLM anywhere in the call path — and every historical statistic
# carries its sample size ``n`` (§64). ``event_risk_caps`` emits the same
# ``QuantityCap`` shape Phase C/D use, but §65 says SHADOW until a backtest
# validates the rules: emitting a cap is not binding one, and the gateway
# deliberately does not pass these to ``assess(extra_caps=...)``.
from .event_risk import (  # noqa: F401
    BASIS_HISTORICAL_MEDIAN,
    BASIS_IMPLIED,
    BASIS_NONE,
    CODE_EVENT_EXPOSURE,
    EVENT_RISK_MODEL_VERSION,
    SENSITIVITY_HIGH,
    SENSITIVITY_LOW,
    SENSITIVITY_MODERATE,
    STATE_EXTREME,
    STATE_HIGH,
    STATE_LADDER,
    STATE_LOW,
    STATE_MODERATE,
    STATE_UNKNOWN,
    EventRiskInputs,
    EventRiskPolicy,
    EventRiskSnapshot,
    EventRiskThresholds,
    classify_event_risk,
    event_risk_caps,
    historical_event_risk,
)

# --- Phase B: typed snapshot (contract §2.11) ------------------------------
from .snapshot import (  # noqa: F401
    SNAPSHOT_VERSION,
    DataQuality,
    PortfolioRiskSnapshot,
    TtlPolicy,
)

__all__ = [
    # --- Tier 0 decision engine (pre-existing; PRODUCTION) ---
    "LiquidityLimits",
    "LiquidityReport",
    "evaluate_underlying_liquidity",
    "IncomeRiskRequest",
    "PortfolioSnapshot",
    "PositionRisk",
    "RiskAssessment",
    "RiskLimits",
    "RiskRequest",
    "BindingConstraint",
    "ExtraCap",
    "LAYER_HARD_LIMIT",
    "assess",
    "assess_income",
    "heat_state",
    "portfolio_heat",
    # --- Phase B returns layer ---
    "ReturnMatrix",
    "ReturnSeries",
    "ReturnType",
    "align",
    "log_returns",
    "returns_from_closes",
    "simple_returns",
    # --- Phase B book P&L ---
    "BookPnl",
    "PositionRiskInput",
    "book_pnl_series",
    "position_pnl_series",
    "METHOD_DELTA_LINEAR",
    "METHOD_FULL_REVAL_CONST_IV",
    "book_method_summary",
    # --- Phase B model framework ---
    "REGISTRY",
    "BaseRiskModel",
    "ModelHealth",
    "ModelMeta",
    "ModelMode",
    "ModelResult",
    "RiskModel",
    # --- Phase B estimators ---
    "ContributionResult",
    "CovarianceResult",
    "DispersionResult",
    "DistributionResult",
    "DrawdownResult",
    "GaussianESModel",
    "GaussianVaRModel",
    "HistoricalESModel",
    "HistoricalVaRModel",
    "ModelRiskState",
    "PositionContribution",
    "conditional_es",
    "conditional_var",
    "dispersion",
    "distribution_diagnostics",
    "drawdown",
    "es_contributions",
    "ewma_variance",
    "ewma_volatility_forecast",
    "gaussian_es",
    "gaussian_var",
    "historical_es",
    "historical_var",
    "incremental_es",
    "marginal_es",
    "model_risk_state",
    "portfolio_volatility",
    "reconstructed_book_drawdown",
    "sample_covariance",
    "tail_size",
    "volatility_contributions",
    "volatility_scaled_pnl",
    # --- Single-factor diagnostic (spec §11, RESEARCH) ---
    "DEFAULT_FACTOR",
    "FactorParams",
    "FactorRiskResult",
    "PositionBeta",
    "beta_vs_factor",
    "factor_risk_share",
    # --- Phase B validation ---
    "BacktestParams",
    "BacktestVerdict",
    "ExceedanceReport",
    "ForecastSeries",
    "christoffersen_independence",
    "exceedances",
    "kupiec_pof",
    "volatility_forecast_error",
    "walk_forward",
    # --- Phase C pre-trade (SHADOW) ---
    "CAP_LAYERS",
    "CODE_BUCKET_ES_CONTRIBUTION",
    "CODE_ES_CONTRIBUTION",
    "CODE_INCREMENTAL_ES",
    "CODE_PORTFOLIO_ES",
    "CandidateSpec",
    "LAYER_CONCENTRATION",
    "LAYER_STATISTICAL",
    "LAYER_STRESS",
    "MODE_PRODUCTION",
    "MODE_SHADOW",
    "MetricPair",
    "QuantityCap",
    "RiskComparison",
    "ShadowVerdict",
    "StatisticalLimits",
    "compare",
    "proposed_book",
    "shadow_verdict",
    "statistical_caps",
    # --- Phase D stress engine (SHADOW) ---
    "CATALOGUE_VERSION",
    "CODE_STRESS_LOSS",
    "DEFAULT_HISTORICAL_WINDOWS",
    "DEFAULT_HYPOTHETICAL_SCENARIOS",
    "IV_SHOCK_SOURCE_RV_PROXY",
    "IV_SHOCK_SOURCE_SPECIFIED",
    "KIND_HISTORICAL",
    "KIND_HYPOTHETICAL",
    "KIND_IV_GRID",
    "KIND_USER",
    "HistoricalShockParams",
    "HistoricalWindow",
    "Scenario",
    "ScenarioResult",
    "StressLimits",
    "StressResult",
    "auto_worst_windows",
    "default_scenarios",
    "historical_shocks_from_closes",
    "run_scenario",
    "run_stress",
    "stress_caps",
    # --- Phase E conditional volatility (RESEARCH) ---
    "DISTRIBUTION_EMPIRICAL_GARCH_SCALED",
    "DISTRIBUTION_GAUSSIAN_GARCH",
    "MODEL_NAME_GARCH",
    "NMResult",
    "SOURCE_EWMA",
    "SOURCE_GARCH",
    "Garch11Model",
    "GarchFit",
    "GarchParams",
    "GarchScaling",
    "chi2_cdf",
    "chi2_sf",
    "conditional_scaled_pnl_source",
    "conditional_volatility_source",
    "fit_garch",
    "garch_forecast_variance",
    "garch_scaled_pnl",
    "garch_scaling",
    "garch_variance_path",
    "garch_volatility_forecast",
    "ljung_box",
    "nelder_mead",
    # --- Phase K event risk (SHADOW) ---
    "BASIS_HISTORICAL_MEDIAN",
    "BASIS_IMPLIED",
    "BASIS_NONE",
    "CODE_EVENT_EXPOSURE",
    "EVENT_RISK_MODEL_VERSION",
    "SENSITIVITY_HIGH",
    "SENSITIVITY_LOW",
    "SENSITIVITY_MODERATE",
    "STATE_EXTREME",
    "STATE_HIGH",
    "STATE_LADDER",
    "STATE_LOW",
    "STATE_MODERATE",
    "STATE_UNKNOWN",
    "EventRiskInputs",
    "EventRiskPolicy",
    "EventRiskSnapshot",
    "EventRiskThresholds",
    "classify_event_risk",
    "event_risk_caps",
    "historical_event_risk",
    # --- Phase B snapshot ---
    "DataQuality",
    "PortfolioRiskSnapshot",
    "SNAPSHOT_VERSION",
    "TtlPolicy",
]
