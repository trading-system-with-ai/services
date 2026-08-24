"""Portfolio Greeks aggregation (development plan §16).

Pure, deterministic, dependency-free arithmetic — no DB, no FastAPI, no
market data. The caller (gateway) computes per-position Greeks upstream
(:func:`libs.trading_core.options.bs.bs_greeks` for options, delta 1.0 for
stock) and this module only SUMS them into portfolio-level exposures, so
every number here is hand-checkable:

- ``net_delta_shares`` — plan §16 "Equivalent Shares": Σ qty * multiplier *
  delta. A 100-share stock position contributes 100; four 0.62-delta calls
  contribute 4 * 100 * 0.62 = 248.
- ``delta_adjusted_notional`` — Σ qty * multiplier * delta * spot, the
  dollar exposure the delta represents.
- ``net_gamma`` / ``net_theta_per_day`` / ``net_vega`` — the same
  qty * multiplier scaling applied to each per-share greek. Theta is $ per
  day; vega is $ per one IV point (both per share on input).

Honest inputs, honest outputs: nothing is estimated or defaulted here — a
position with unknown greeks must be resolved by the caller BEFORE
aggregation, never smuggled in as zeros.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Instrument vocabulary (plan §16). Spread rows contribute their NET
# per-share greeks (Phase 1); income rows their short leg's NEGATED greeks
# (Phase 2 — this entry was missing until 2026-08-17: an open income
# position crashed the aggregate view); SHORT_STOCK contributes delta −1
# per share (Phase 3).
VALID_INSTRUMENTS = (
    "LONG_STOCK",
    "LONG_CALL",
    "LONG_PUT",
    "BULL_CALL_SPREAD",
    "BEAR_PUT_SPREAD",
    "COVERED_CALL",
    "CASH_SECURED_PUT",
    "SHORT_STOCK",
)


@dataclass(frozen=True)
class PositionGreeksInput:
    """One position's per-share greeks, as computed upstream (plan §16).

    - ``quantity``: shares for stock, contracts for options.
    - ``multiplier``: 1 for stock, 100 for standard equity options.
    - ``delta``: per share, 1.0 for long stock, Black-Scholes delta for
      options (negative for long puts).
    - ``theta_per_day``: $ per share per calendar day (typically negative
      for long options).
    - ``vega``: $ per share per one IV point.
    """

    ticker: str
    instrument: str
    quantity: int
    multiplier: int
    spot: float
    delta: float
    gamma: float
    theta_per_day: float
    vega: float

    def __post_init__(self) -> None:
        if self.instrument not in VALID_INSTRUMENTS:
            raise ValueError(
                f"instrument must be one of {VALID_INSTRUMENTS}, "
                f"got {self.instrument!r}"
            )
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier}")


@dataclass(frozen=True)
class PositionGreeksContribution:
    """One position's fully-scaled contribution to the portfolio totals.

    Every field is the input's per-share greek times ``quantity *
    multiplier`` (times ``spot`` for the notional), so the portfolio sums
    below can be audited line by line (plan §36 spirit).
    """

    ticker: str
    instrument: str
    quantity: int
    multiplier: int
    delta_shares: float
    delta_notional: float
    gamma: float
    theta_per_day: float
    vega: float


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregated portfolio-level greeks (plan §16).

    - ``net_delta_shares``: Σ qty * mult * delta (§16 "Equivalent Shares").
    - ``delta_adjusted_notional``: Σ qty * mult * delta * spot ($).
    - ``net_gamma``: Σ qty * mult * gamma.
    - ``net_theta_per_day``: Σ qty * mult * theta_per_day ($/day).
    - ``net_vega``: Σ qty * mult * vega ($ per IV point).
    - ``per_position``: each input's contribution, in input order.
    """

    net_delta_shares: float
    delta_adjusted_notional: float
    net_gamma: float
    net_theta_per_day: float
    net_vega: float
    per_position: tuple[PositionGreeksContribution, ...]


def position_contribution(
    position: PositionGreeksInput,
) -> PositionGreeksContribution:
    """Scale one position's per-share greeks to its full size (plan §16)."""
    scale = position.quantity * position.multiplier
    delta_shares = scale * position.delta
    return PositionGreeksContribution(
        ticker=position.ticker,
        instrument=position.instrument,
        quantity=position.quantity,
        multiplier=position.multiplier,
        delta_shares=delta_shares,
        delta_notional=delta_shares * position.spot,
        gamma=scale * position.gamma,
        theta_per_day=scale * position.theta_per_day,
        vega=scale * position.vega,
    )


def aggregate_greeks(
    inputs: Sequence[PositionGreeksInput],
) -> PortfolioGreeks:
    """Sum per-position greek contributions into portfolio totals (§16).

    Pure arithmetic: an empty book aggregates to all-zero totals with an
    empty ``per_position`` — never ``None`` (the zeros are true sums, not
    fabricated placeholders).
    """
    contributions = tuple(position_contribution(p) for p in inputs)
    return PortfolioGreeks(
        net_delta_shares=sum(c.delta_shares for c in contributions),
        delta_adjusted_notional=sum(c.delta_notional for c in contributions),
        net_gamma=sum(c.gamma for c in contributions),
        net_theta_per_day=sum(c.theta_per_day for c in contributions),
        net_vega=sum(c.vega for c in contributions),
        per_position=contributions,
    )
