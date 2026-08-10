"""Black-Scholes(-Merton) pricing and greeks (development plan §9).

Pure stdlib, deterministic, no numpy/scipy (house rule): the standard normal
CDF is ``N(x) = 0.5 * (1 + erf(x / sqrt(2)))`` via :func:`math.erf`, and the
standard normal PDF is ``phi(x) = exp(-x^2 / 2) / sqrt(2*pi)``.

Model: European option on a continuous-dividend-yield underlying, with

    d1 = (ln(S/K) + (r - q + iv^2 / 2) * T) / (iv * sqrt(T))
    d2 = d1 - iv * sqrt(T)

Greek CONVENTIONS (precise — consumers such as the contract selector §9.1
depend on these units):

- ``delta``: signed dPrice/dSpot. Calls in (0, +1), puts in (-1, 0).
- ``gamma``: dDelta/dSpot per $1 of spot; identical for calls and puts.
- ``theta``: PER CALENDAR DAY — the annualized Black-Scholes theta divided
  by 365. Negative for long options in the typical case (time decay costs
  the holder money); deep-ITM European puts / high-dividend calls can have
  a positive theta — the sign is whatever the model produces, NOT forced.
- ``vega``: per 1 IV POINT, i.e. the price change for a 0.01 (one
  percentage point) move in implied volatility — annualized vega / 100.
  Identical for calls and puts.

Edge handling:

- ``t_years <= 0`` (at/after expiry): price is intrinsic value, delta is
  0 or +/-1 (calls: +1 if spot > strike else 0; puts: -1 if spot < strike
  else 0 — an exactly at-the-money expiring option reports delta 0), and
  gamma/theta/vega are 0. Honest terminal values, never NaN (house rule).
- ``iv <= 0``, ``spot <= 0`` or ``strike <= 0``: ``ValueError`` — always,
  even at expiry, so bad inputs never silently produce a number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Calendar days per year used to convert annualized theta to per-day theta.
_DAYS_PER_YEAR = 365.0


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (house rule: no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Greeks:
    """Black-Scholes price and greeks in the module's documented units.

    ``theta`` per calendar day (annual / 365); ``vega`` per 1 IV point
    (per 0.01 of implied volatility); ``delta`` signed (calls +, puts -).
    """

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float


def _validate(spot: float, strike: float, iv: float, right: str) -> None:
    """Reject non-positive spot/strike/iv and unknown rights (ValueError)."""
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if iv <= 0.0:
        raise ValueError(f"iv must be > 0, got {iv}")
    if right not in ("C", "P"):
        raise ValueError(f'right must be "C" or "P", got {right!r}')


def _intrinsic(spot: float, strike: float, right: str) -> float:
    """Intrinsic value at expiry (t_years <= 0)."""
    if right == "C":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _expiry_delta(spot: float, strike: float, right: str) -> float:
    """Terminal delta: 0 or +/-1; exactly at-the-money reports 0."""
    if right == "C":
        return 1.0 if spot > strike else 0.0
    return -1.0 if spot < strike else 0.0


def _d1_d2(
    spot: float, strike: float, t_years: float, iv: float, r: float, q: float
) -> tuple[float, float]:
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years) / (
        iv * sqrt_t
    )
    return d1, d1 - iv * sqrt_t


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    right: str,
    r: float = 0.04,
    q: float = 0.0,
) -> float:
    """European Black-Scholes-Merton price (plan §9).

    ``right`` is ``"C"`` or ``"P"``; ``r`` is the continuously-compounded
    risk-free rate and ``q`` the continuous dividend yield, both annualized.
    ``t_years <= 0`` returns intrinsic value; ``iv <= 0``, ``spot <= 0`` or
    ``strike <= 0`` raise ``ValueError``.
    """
    _validate(spot, strike, iv, right)
    if t_years <= 0.0:
        return _intrinsic(spot, strike, right)
    d1, d2 = _d1_d2(spot, strike, t_years, iv, r, q)
    disc_s = spot * math.exp(-q * t_years)
    disc_k = strike * math.exp(-r * t_years)
    if right == "C":
        return disc_s * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
    return disc_k * _norm_cdf(-d2) - disc_s * _norm_cdf(-d1)


def bs_greeks(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    right: str,
    r: float = 0.04,
    q: float = 0.0,
) -> Greeks:
    """Black-Scholes price plus greeks in the documented units (plan §9).

    Units — see the module docstring for the full statement:

    - ``delta`` signed (calls +, puts -), ``e^{-qT} * N(d1)`` for calls and
      ``-e^{-qT} * N(-d1)`` for puts.
    - ``gamma`` per $1 of spot, same for both rights.
    - ``theta`` PER CALENDAR DAY: annualized theta / 365 (negative for long
      options in the typical time-decay case).
    - ``vega`` per 1 IV POINT (0.01 of vol): annualized vega / 100, same
      for both rights.

    ``t_years <= 0``: intrinsic price, delta 0 / +/-1, all other greeks 0.
    ``iv <= 0``, ``spot <= 0`` or ``strike <= 0``: ``ValueError``.
    """
    _validate(spot, strike, iv, right)
    if t_years <= 0.0:
        return Greeks(
            price=_intrinsic(spot, strike, right),
            delta=_expiry_delta(spot, strike, right),
            gamma=0.0,
            theta=0.0,
            vega=0.0,
        )

    d1, d2 = _d1_d2(spot, strike, t_years, iv, r, q)
    sqrt_t = math.sqrt(t_years)
    exp_qt = math.exp(-q * t_years)
    exp_rt = math.exp(-r * t_years)
    disc_s = spot * exp_qt
    disc_k = strike * exp_rt
    pdf_d1 = _norm_pdf(d1)

    if right == "C":
        price = disc_s * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
        delta = exp_qt * _norm_cdf(d1)
        theta_annual = (
            -disc_s * pdf_d1 * iv / (2.0 * sqrt_t)
            - r * disc_k * _norm_cdf(d2)
            + q * disc_s * _norm_cdf(d1)
        )
    else:
        price = disc_k * _norm_cdf(-d2) - disc_s * _norm_cdf(-d1)
        delta = -exp_qt * _norm_cdf(-d1)
        theta_annual = (
            -disc_s * pdf_d1 * iv / (2.0 * sqrt_t)
            + r * disc_k * _norm_cdf(-d2)
            - q * disc_s * _norm_cdf(-d1)
        )

    gamma = exp_qt * pdf_d1 / (spot * iv * sqrt_t)
    vega_annual = disc_s * pdf_d1 * sqrt_t

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_annual / _DAYS_PER_YEAR,
        vega=vega_annual / 100.0,
    )
