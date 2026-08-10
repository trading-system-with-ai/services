"""Options pricing library (development plan §9 — contract analytics).

Pure, deterministic stdlib-only code — no DB, no FastAPI, no numpy/scipy.
The standard normal CDF is built from :func:`math.erf` per house rules:
``N(x) = 0.5 * (1 + erf(x / sqrt(2)))``.

Conventions (documented in full on :mod:`libs.trading_core.options.bs`):

- ``theta`` is PER CALENDAR DAY (annualized theta / 365), negative for
  long options.
- ``vega`` is per 1 IV POINT (a 0.01 change in implied volatility).
- ``delta`` is signed: calls positive, puts negative.
"""
from .bs import Greeks, bs_greeks, bs_price  # noqa: F401

__all__ = [
    "Greeks",
    "bs_greeks",
    "bs_price",
]
