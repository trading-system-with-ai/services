"""Income-leg selection — covered calls / cash-secured puts (Phase 2).

Pure, deterministic library code (no DB, no FastAPI, no fetching), same
charter as the §9 selector. These are INCOME OVERLAYS, not §8 directional
entries: a covered call sells upside above an EXISTING holding; a CSP
sells downside against LOCKED cash. Selection follows the most widely
cited mechanical standards: 30–45 DTE, |delta| in the 0.15–0.35 band
(target 0.25) — far enough to keep assignment odds modest, near enough
that the credit is worth collecting — with basic §9.1-style liquidity
gates on the leg being SOLD (real NBBO only, OI floor, spread cap).

NO ELIGIBLE CONTRACT is a valid output with named reasons (§44 rule 18).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from libs.trading_core.contracts import ContractQuote


@dataclass(frozen=True)
class IncomeParams:
    """Income-leg thresholds (§6.2 research parameters)."""

    dte_min: int = 30
    dte_max: int = 45
    abs_delta_min: float = 0.15
    abs_delta_max: float = 0.35
    abs_delta_target: float = 0.25
    min_open_interest: int = 50
    max_spread_pct: float = 0.15

    def __post_init__(self) -> None:
        if not (1 <= self.dte_min <= self.dte_max):
            raise ValueError(
                f"require 1 <= dte_min <= dte_max, got {self.dte_min}/{self.dte_max}"
            )
        if not (
            0.0 < self.abs_delta_min
            <= self.abs_delta_target
            <= self.abs_delta_max
            < 1.0
        ):
            raise ValueError(
                "require 0 < abs_delta_min <= abs_delta_target <= "
                f"abs_delta_max < 1, got {self.abs_delta_min}/"
                f"{self.abs_delta_target}/{self.abs_delta_max}"
            )


@dataclass
class IncomeSelection:
    """Selector verdict: the contract to SELL, or named blockers."""

    contract: ContractQuote | None
    fail_reasons: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def _select_income_leg(
    chain: list[ContractQuote],
    right: str,
    spot: float,
    params: IncomeParams,
    otm_required: bool,
) -> IncomeSelection:
    rejected: Counter[str] = Counter()
    candidates: list[ContractQuote] = []
    for c in chain:
        if c.right != right:
            continue
        if not (params.dte_min <= c.dte <= params.dte_max):
            rejected["DTE outside window"] += 1
            continue
        if otm_required and (
            (right == "C" and c.strike <= spot)
            or (right == "P" and c.strike >= spot)
        ):
            # Selling ITM premium flips the trade's character (§4: income,
            # not leverage) — OTM only.
            rejected["not OTM"] += 1
            continue
        if c.price_basis != "quote":
            rejected["no real NBBO quote"] += 1
            continue
        if c.mid <= 0.0 or c.bid <= 0.0:
            # SOLD at (near) the bid: a zero bid means the credit is zero.
            rejected["no bid to sell into"] += 1
            continue
        if c.delta is None:
            rejected["delta not provided"] += 1
            continue
        if not (params.abs_delta_min <= abs(c.delta) <= params.abs_delta_max):
            rejected["|delta| outside band"] += 1
            continue
        if c.open_interest < params.min_open_interest:
            rejected["open interest below floor"] += 1
            continue
        if c.spread_pct > params.max_spread_pct:
            rejected["bid-ask spread above cap"] += 1
            continue
        candidates.append(c)

    if not candidates:
        detail = ", ".join(f"{k} ({n})" for k, n in rejected.most_common())
        return IncomeSelection(
            contract=None,
            fail_reasons=[
                f"no eligible short {right} in the {params.dte_min}-"
                f"{params.dte_max} DTE / |Δ| {params.abs_delta_min:.2f}-"
                f"{params.abs_delta_max:.2f} band"
                + (f"; rejected — {detail}" if detail else "")
            ],
        )

    # Nearest |delta| to target; stable tie-breaks (dte, strike).
    pick = min(
        candidates,
        key=lambda c: (
            abs(abs(c.delta) - params.abs_delta_target),
            c.dte,
            c.strike,
        ),
    )
    annualized = (
        (pick.mid / spot) * (365.0 / pick.dte) if spot > 0 and pick.dte > 0 else 0.0
    )
    return IncomeSelection(
        contract=pick,
        rationale=[
            (
                f"income leg: short {right} {pick.strike:g} exp "
                f"{pick.expiry.isoformat()} ({pick.dte}d), Δ {pick.delta:+.2f} "
                f"(target |Δ| {params.abs_delta_target:.2f}), credit mid "
                f"{pick.mid:.4f}/share (~{annualized:.1%} annualized on spot)."
            ),
        ],
    )


def select_covered_call(
    chain: list[ContractQuote], spot: float, params: IncomeParams = IncomeParams()
) -> IncomeSelection:
    """The call to SELL against 100 shares/contract of an existing holding."""
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    return _select_income_leg(chain, "C", spot, params, otm_required=True)


def select_cash_secured_put(
    chain: list[ContractQuote], spot: float, params: IncomeParams = IncomeParams()
) -> IncomeSelection:
    """The put to SELL against locked cash of strike*100/contract."""
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    return _select_income_leg(chain, "P", spot, params, otm_required=True)
