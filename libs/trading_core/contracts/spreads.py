"""Vertical spread selector — §9-S (execution-chains roadmap Phase 1).

Pure, deterministic library code, same charter as the §9 single-leg
selector it builds on: no DB, no FastAPI, no market data fetching.

Composition rule (§21: reuse, never reimplement): the LONG leg is exactly
the §9 selector's rank-1 contract for the direction — every §9.1 filter and
§9.2 ranking applies unchanged. The SHORT leg is then chosen from the SAME
expiry, OTM-ward of the long strike, at a width targeted as a fraction of
spot (research parameters, §6.2) — from the REAL strike grid only.

The short leg is SOLD, so its quote quality matters more, not less:
``price_basis`` must be a real NBBO quote (a day-close mid has an unknown
spread — fail-closed, §9.1), plus its own OI and spread caps.

Every produced candidate carries the defined-risk arithmetic (net debit =
max loss, width − debit = max profit, breakeven) and None-safe NET greeks
(either leg missing a greek -> None, never zero-filled) with §37 rationale
lines. NO ELIGIBLE SPREAD is a valid output with named reasons, never an
error.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .selector import ContractQuote, ScoredContract, SelectorParams, select_contracts

#: Instrument names this selector serves (mirrors InstrumentType values;
#: kept as strings so this module stays import-light).
BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"


@dataclass(frozen=True)
class SpreadParams:
    """§9-S vertical-spread parameters (research parameters, §6.2 — to be
    swept by the backtester once the spread backtest leg lands, never
    permanent truths).

    - ``width_pct_min`` / ``width_pct_target`` / ``width_pct_max``: the
      short strike's distance from the LONG strike, as a fraction of spot.
      The nearest real strike to ``target`` inside [min, max] wins.
    - ``short_min_open_interest`` / ``short_max_spread_pct``: liquidity
      floors for the SHORT leg (it must be re-buyable to close).
    """

    width_pct_min: float = 0.02
    width_pct_target: float = 0.05
    width_pct_max: float = 0.15
    short_min_open_interest: int = 50
    short_max_spread_pct: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 < self.width_pct_min <= self.width_pct_target <= self.width_pct_max):
            raise ValueError(
                "require 0 < width_pct_min <= width_pct_target <= "
                f"width_pct_max, got {self.width_pct_min!r} / "
                f"{self.width_pct_target!r} / {self.width_pct_max!r}"
            )
        if self.short_min_open_interest < 0:
            raise ValueError(
                "short_min_open_interest must be >= 0, got "
                f"{self.short_min_open_interest!r}"
            )
        if self.short_max_spread_pct <= 0.0:
            raise ValueError(
                "short_max_spread_pct must be > 0, got "
                f"{self.short_max_spread_pct!r}"
            )


def _net_greek(long_v: float | None, short_v: float | None) -> float | None:
    """Long minus short, None-safe: either side unknown -> None (§ honest
    nulls — a net greek computed from an invented zero is a fabrication)."""
    if long_v is None or short_v is None:
        return None
    return long_v - short_v


@dataclass
class SpreadCandidate:
    """One fully-specified vertical spread (defined risk, §37-explained).

    Premiums are PER SHARE (quote convention). ``max_loss`` == ``net_debit``
    by construction — the §12 risk engine can size this without new theory.
    """

    instrument: str  # BULL_CALL_SPREAD | BEAR_PUT_SPREAD
    long_leg: ContractQuote
    short_leg: ContractQuote
    net_debit: float
    width: float
    max_loss: float
    max_profit: float
    breakeven: float
    net_delta: float | None
    net_gamma: float | None
    net_theta: float | None
    net_vega: float | None
    rationale: list[str] = field(default_factory=list)


@dataclass
class SpreadSelection:
    """Selector verdict: a candidate, or named reasons why none exists."""

    candidate: SpreadCandidate | None
    fail_reasons: list[str] = field(default_factory=list)
    #: The §9 verdict backing the long leg (None when no leg was eligible).
    long_leg_scored: ScoredContract | None = None


def _summarize_long_failures(scored: list[ScoredContract]) -> str:
    """Top §9.1 blockers across the chain, with counts (§37)."""
    counter: Counter[str] = Counter()
    for s in scored:
        for reason in s.fail_reasons:
            counter[reason.split(":")[0]] += 1
    top = ", ".join(f"{name} ({n})" for name, n in counter.most_common(3))
    return top or "empty chain"


def select_vertical_spread(
    chain: list[ContractQuote],
    instrument: str,
    spot: float,
    params: SelectorParams = SelectorParams(),
    spread_params: SpreadParams = SpreadParams(),
) -> SpreadSelection:
    """Select a vertical debit spread from a REAL chain snapshot (§9-S).

    ``instrument`` is ``"BULL_CALL_SPREAD"`` (long the §9 call, short a
    higher-strike call) or ``"BEAR_PUT_SPREAD"`` (long the §9 put, short a
    lower-strike put). Deterministic; NO-ELIGIBLE is a valid output.
    """
    if instrument == BULL_CALL_SPREAD:
        direction, otm_sign = "BULL", +1.0
    elif instrument == BEAR_PUT_SPREAD:
        direction, otm_sign = "BEAR", -1.0
    else:
        raise ValueError(
            f"instrument must be {BULL_CALL_SPREAD!r} or {BEAR_PUT_SPREAD!r}, "
            f"got {instrument!r}"
        )
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot!r}")

    # --- LONG leg: the §9 rank-1 contract, rules unchanged (§21) -----------
    scored = select_contracts(chain, direction, params)
    long_scored = next((s for s in scored if s.rank == 1), None)
    if long_scored is None:
        return SpreadSelection(
            candidate=None,
            fail_reasons=[
                f"no §9-eligible long leg for {direction}: top blockers — "
                + _summarize_long_failures(scored)
            ],
        )
    long_leg = long_scored.contract

    # --- SHORT leg: same expiry, OTM-ward, width targeted vs spot ----------
    lo = spread_params.width_pct_min * spot
    hi = spread_params.width_pct_max * spot
    target = long_leg.strike + otm_sign * spread_params.width_pct_target * spot

    rejected: Counter[str] = Counter()
    candidates: list[ContractQuote] = []
    for c in chain:
        if c.right != long_leg.right or c.expiry != long_leg.expiry:
            continue
        distance = otm_sign * (c.strike - long_leg.strike)
        if distance <= 0.0:
            continue  # not OTM-ward of the long strike
        if not (lo <= distance <= hi):
            rejected["width outside band"] += 1
            continue
        if c.price_basis != "quote":
            # Selling a leg priced off a day close would book an UNKNOWN
            # spread as income — fail closed (§9.1).
            rejected["no real NBBO quote (price_basis != quote)"] += 1
            continue
        if c.mid <= 0.0:
            rejected["non-positive mid"] += 1
            continue
        if c.open_interest < spread_params.short_min_open_interest:
            rejected["short-leg open interest below floor"] += 1
            continue
        if c.spread_pct > spread_params.short_max_spread_pct:
            rejected["short-leg bid-ask spread above cap"] += 1
            continue
        candidates.append(c)

    if not candidates:
        detail = (
            "; rejected — "
            + ", ".join(f"{k} ({n})" for k, n in rejected.most_common())
            if rejected
            else ""
        )
        return SpreadSelection(
            candidate=None,
            fail_reasons=[
                f"no eligible short leg within "
                f"[{spread_params.width_pct_min:.0%}, "
                f"{spread_params.width_pct_max:.0%}] of spot from the long "
                f"strike {long_leg.strike:g} ({long_leg.right}, "
                f"{long_leg.expiry.isoformat()}){detail}"
            ],
            long_leg_scored=long_scored,
        )

    # Nearest real strike to the target width; stable tie-break by strike so
    # identical input -> identical output.
    short_leg = min(candidates, key=lambda c: (abs(c.strike - target), c.strike))

    # --- Defined-risk arithmetic -------------------------------------------
    net_debit = long_leg.mid - short_leg.mid
    width = abs(short_leg.strike - long_leg.strike)
    if net_debit <= 0.0:
        return SpreadSelection(
            candidate=None,
            fail_reasons=[
                f"net debit {net_debit:.4f} <= 0 from these quotes (long mid "
                f"{long_leg.mid:.4f} vs short mid {short_leg.mid:.4f}) — a "
                "vertical DEBIT spread cannot cost nothing; quote anomaly, "
                "fail closed."
            ],
            long_leg_scored=long_scored,
        )
    if net_debit >= width:
        return SpreadSelection(
            candidate=None,
            fail_reasons=[
                f"net debit {net_debit:.4f} >= width {width:.4f}: max profit "
                "would be <= 0 — the spread has no upside at these quotes."
            ],
            long_leg_scored=long_scored,
        )

    max_profit = width - net_debit
    breakeven = long_leg.strike + otm_sign * net_debit

    candidate = SpreadCandidate(
        instrument=instrument,
        long_leg=long_leg,
        short_leg=short_leg,
        net_debit=net_debit,
        width=width,
        max_loss=net_debit,
        max_profit=max_profit,
        breakeven=breakeven,
        net_delta=_net_greek(long_leg.delta, short_leg.delta),
        net_gamma=_net_greek(long_leg.gamma, short_leg.gamma),
        net_theta=_net_greek(long_leg.theta, short_leg.theta),
        net_vega=_net_greek(long_leg.vega, short_leg.vega),
        rationale=[
            (
                f"§9-S {instrument}: long {long_leg.right} {long_leg.strike:g} "
                f"@ {long_leg.mid:.4f} (§9 rank-1) + short {short_leg.right} "
                f"{short_leg.strike:g} @ {short_leg.mid:.4f} "
                f"({short_leg.expiry.isoformat()}, {long_leg.dte}d)."
            ),
            (
                f"net debit {net_debit:.4f}/share = MAX LOSS (defined risk); "
                f"width {width:g} -> max profit {max_profit:.4f}/share; "
                f"breakeven {breakeven:.4f} at expiry."
            ),
            (
                "short leg hedges the premium: net theta "
                + (
                    f"{_net_greek(long_leg.theta, short_leg.theta):+.4f}/day"
                    if long_leg.theta is not None and short_leg.theta is not None
                    else "unknown (a leg's theta not provided)"
                )
                + ", net vega "
                + (
                    f"{_net_greek(long_leg.vega, short_leg.vega):+.4f}"
                    if long_leg.vega is not None and short_leg.vega is not None
                    else "unknown (a leg's vega not provided)"
                )
                + "."
            ),
        ],
    )
    return SpreadSelection(
        candidate=candidate, fail_reasons=[], long_leg_scored=long_scored
    )
