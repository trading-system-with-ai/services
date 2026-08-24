"""Options pricing library (development plan §9 — contract analytics).

Pure, deterministic stdlib-only code — no DB, no FastAPI, no numpy/scipy.
The standard normal CDF is built from :func:`math.erf` per house rules:
``N(x) = 0.5 * (1 + erf(x / sqrt(2)))``.

Conventions (documented in full on :mod:`libs.trading_core.options.bs`):

- ``theta`` is PER CALENDAR DAY (annualized theta / 365), negative for
  long options.
- ``vega`` is per 1 IV POINT (a 0.01 change in implied volatility).
- ``delta`` is signed: calls positive, puts negative.

Phase D adds two modules on top of the pricer, both stdlib-only and both
importing ONLY ``bs`` (no risk-package import, so ``options`` stays a leaf
the risk library may depend on):

- ``iv`` — the implied-volatility bisection solver. Whatever it returns is
  INTERNALLY CALCULATED (``method="BISECTION"``) and must be labelled as
  such wherever it is stored or displayed; it is never vendor IV
  (``docs/data-source-architecture.md`` §12).
- ``reval`` — basis-anchored scenario revaluation of option and stock legs.
  IV shocks are RELATIVE and multiplicative on the IV level, the basis
  ``mark0 − model0`` is held constant so a zero scenario is exactly 0.0,
  and a leg with no IV is priced DELTA_LINEAR and LABELLED so
  (``method_coverage``) rather than silently counted as full revaluation.
"""
from .bs import Greeks, bs_greeks, bs_price  # noqa: F401

# --- Phase D: implied vol (design §8.1) — INTERNALLY CALCULATED ------------
from .iv import (  # noqa: F401
    DEFAULT_HI,
    DEFAULT_LO,
    DEFAULT_MAX_ITER,
    DEFAULT_TOL,
    METHOD_BISECTION,
    IVResult,
    implied_vol,
)

# --- Phase D: scenario revaluation (design §8.2) ---------------------------
from .reval import (  # noqa: F401
    DAYS_PER_YEAR,
    DEFAULT_MULTIPLIER,
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL,
    LegBaseline,
    OptionLeg,
    ScenarioPnl,
    StockLeg,
    leg_baseline,
    reval_leg,
    scenario_pnl,
)

__all__ = [
    # --- Black-Scholes pricer ---
    "Greeks",
    "bs_greeks",
    "bs_price",
    # --- Phase D implied vol ---
    "DEFAULT_HI",
    "DEFAULT_LO",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOL",
    "METHOD_BISECTION",
    "IVResult",
    "implied_vol",
    # --- Phase D scenario revaluation ---
    "DAYS_PER_YEAR",
    "DEFAULT_MULTIPLIER",
    "METHOD_DELTA_LINEAR",
    "METHOD_FULL_REVAL",
    "LegBaseline",
    "OptionLeg",
    "ScenarioPnl",
    "StockLeg",
    "leg_baseline",
    "reval_leg",
    "scenario_pnl",
]
