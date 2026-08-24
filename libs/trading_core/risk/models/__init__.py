"""Statistical risk models (Phase B; spec §4 registry, contract §2.2–§2.8).

Everything in this package is SHADOW/RESEARCH — nothing here alters a
Tier 0 decision (``risk/engine.py`` stays byte-identical).

This init exports the base abstractions AND the Phase B model modules'
public names, so callers can ``from libs.trading_core.risk.models import
historical_var, es_contributions, drawdown`` without knowing the module
layout. Importing this package therefore imports every model module, which
registers the four VaR/ES models (``register_models()`` is idempotent).

Import order matters and is fixed here: ``base`` (no intra-package
imports) → ``volatility`` (base, returns) → ``var_es`` (base, volatility) →
``contribution`` (base, var_es for the ONE ``tail_size``) → the leaf
modules → ``stress`` (base + the ``options`` leaf package) → ``garch``
(base, volatility, the private ``_chi2`` primitive and the leaf
``risk/optim``). No module imports ``risk/__init__`` or ``risk/engine``,
and ``stress``'s single reference back to ``pretrade`` is function-local,
so this package stays cycle-free in BOTH import orders (``risk.models``
first or ``risk.pretrade`` first).

Phase E note (spec §70): the GARCH model registers in ``RESEARCH`` mode,
one step below the SHADOW default the rest of this package uses. EWMA is
still the conditional-volatility forecaster; ``conditional_volatility_source``
is the single place that implements the §13/§58 fallback and names which
model produced the number a caller is holding.
"""
from .base import (  # noqa: F401
    REGISTRY,
    BaseRiskModel,
    ModelHealth,
    ModelMeta,
    ModelMode,
    ModelResult,
    ModelTier,
    RiskModel,
    active,
    clear_for_tests,
    combine_health,
    degraded,
    downgrade,
    failed,
    get,
    health_rank,
    names,
    register,
    tier_for_model_name,
    unavailable,
    validate_never_upgrades,
)
from .volatility import (  # noqa: F401
    CovarianceResult,
    VolatilityScaling,
    ewma_variance,
    ewma_volatility_forecast,
    portfolio_volatility,
    sample_covariance,
    volatility_scaled_pnl,
    volatility_scaling,
)
from .var_es import (  # noqa: F401
    DISTRIBUTION_EMPIRICAL,
    DISTRIBUTION_EMPIRICAL_VOL_SCALED,
    DISTRIBUTION_NORMAL,
    GaussianESModel,
    GaussianVaRModel,
    HistoricalESModel,
    HistoricalVaRModel,
    conditional_es,
    conditional_var,
    default_min_obs,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    register_models,
    sorted_losses,
    tail_size,
)
from .contribution import (  # noqa: F401
    ContributionParams,
    ContributionResult,
    IncrementalResult,
    METHOD_ES,
    METHOD_VOL,
    PositionContribution,
    es_contributions,
    incremental_es,
    marginal_es,
    volatility_contributions,
)
from .diagnostics import (  # noqa: F401
    DistributionParams,
    DistributionResult,
    distribution_diagnostics,
    jarque_bera_p_value,
)
from .ensemble import (  # noqa: F401
    DispersionResult,
    EnsembleParams,
    FLAG_DISPERSION_HIGH,
    ModelRiskState,
    RISK_ELEVATED,
    RISK_HIGH,
    RISK_LOW,
    dispersion,
    model_risk_state,
)
# --- Single-factor (SPY) diagnostic (spec §11) — RESEARCH ------------------
# ``factor`` imports ``base`` only, so it adds no edge to the dependency
# graph and this package stays cycle-free in both import orders. RESEARCH
# (spec §70), a step BELOW SHADOW: it registers no model, derives no cap and
# is never consulted by ``assess`` — a pure display diagnostic answering
# "how much of this book is just the market?".
from .factor import (  # noqa: F401
    DEFAULT_FACTOR,
    DEFAULT_FACTOR_PARAMS,
    MODEL_NAME as FACTOR_MODEL_NAME,
    FactorParams,
    FactorRiskResult,
    PositionBeta,
    beta_vs_factor,
    factor_risk_share,
)
from .drawdown import (  # noqa: F401
    METHOD_NAV_PATH,
    METHOD_RECONSTRUCTED,
    DrawdownParams,
    DrawdownResult,
    drawdown,
    reconstructed_book_drawdown,
)

# --- Phase D: stress engine (design §8.3) ----------------------------------
# ``stress`` imports ``base`` and the ``options`` leaf package only; its one
# reference back to ``pretrade`` (for the shared ``QuantityCap`` shape and
# the verified bisection helper) is a FUNCTION-LOCAL import inside
# ``stress_caps``, so this package still depends on nothing above it.
from .stress import (  # noqa: F401
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
    LAYER_STRESS,
    SCENARIO_KINDS,
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
# ``garch`` imports ``base``, ``volatility``, the private ``_chi2`` primitive
# and ``risk/optim`` (a leaf module with no intra-package imports), so it
# adds no new edge to the dependency graph and this package stays cycle-free
# in both import orders. Its model is registered in RESEARCH mode — one step
# BELOW the library's SHADOW default: EWMA remains the conditional-volatility
# forecaster, and ``conditional_volatility_source`` is the seam that says
# which one a caller actually got (spec §13/§58 fallback hierarchy).
from .garch import (  # noqa: F401
    DISTRIBUTION_EMPIRICAL_GARCH_SCALED,
    DISTRIBUTION_GAUSSIAN_GARCH,
    MODEL_NAME_GARCH,
    PERSISTENCE_MAX,
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
from .garch import register_models as register_garch_models  # noqa: F401
from ._chi2 import (  # noqa: F401
    chi2_cdf,
    chi2_sf,
    regularized_gamma_p,
    regularized_gamma_q,
)

# The four VaR/ES models self-register at ``var_es`` import time; call again
# so a test that emptied the registry restores it by re-importing (idempotent).
# Same for the RESEARCH-mode GARCH model.
register_models()
register_garch_models()

__all__ = [
    # base
    "REGISTRY",
    "BaseRiskModel",
    "ModelHealth",
    "ModelMeta",
    "ModelMode",
    "ModelResult",
    "ModelTier",
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
    # volatility
    "CovarianceResult",
    "VolatilityScaling",
    "ewma_variance",
    "ewma_volatility_forecast",
    "portfolio_volatility",
    "sample_covariance",
    "volatility_scaled_pnl",
    "volatility_scaling",
    # var_es
    "DISTRIBUTION_EMPIRICAL",
    "DISTRIBUTION_EMPIRICAL_VOL_SCALED",
    "DISTRIBUTION_NORMAL",
    "GaussianESModel",
    "GaussianVaRModel",
    "HistoricalESModel",
    "HistoricalVaRModel",
    "conditional_es",
    "conditional_var",
    "default_min_obs",
    "gaussian_es",
    "gaussian_var",
    "historical_es",
    "historical_var",
    "register_models",
    "sorted_losses",
    "tail_size",
    # contribution
    "ContributionParams",
    "ContributionResult",
    "IncrementalResult",
    "METHOD_ES",
    "METHOD_VOL",
    "PositionContribution",
    "es_contributions",
    "incremental_es",
    "marginal_es",
    "volatility_contributions",
    # diagnostics
    "DistributionParams",
    "DistributionResult",
    "distribution_diagnostics",
    "jarque_bera_p_value",
    # ensemble
    "DispersionResult",
    "EnsembleParams",
    "FLAG_DISPERSION_HIGH",
    "ModelRiskState",
    "RISK_ELEVATED",
    "RISK_HIGH",
    "RISK_LOW",
    "dispersion",
    "model_risk_state",
    # drawdown
    "METHOD_NAV_PATH",
    "METHOD_RECONSTRUCTED",
    "DrawdownParams",
    "DrawdownResult",
    "drawdown",
    "reconstructed_book_drawdown",
    # factor (spec §11, RESEARCH)
    "DEFAULT_FACTOR",
    "DEFAULT_FACTOR_PARAMS",
    "FACTOR_MODEL_NAME",
    "FactorParams",
    "FactorRiskResult",
    "PositionBeta",
    "beta_vs_factor",
    "factor_risk_share",
    # stress (Phase D)
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
    "LAYER_STRESS",
    "SCENARIO_KINDS",
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
    # garch (Phase E, RESEARCH) + the chi2 primitive behind its diagnostics
    "DISTRIBUTION_EMPIRICAL_GARCH_SCALED",
    "DISTRIBUTION_GAUSSIAN_GARCH",
    "MODEL_NAME_GARCH",
    "PERSISTENCE_MAX",
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
    "register_garch_models",
    "regularized_gamma_p",
    "regularized_gamma_q",
]
