"""Risk Engine v0 tests (development plan §12, §13, §42).

Every deterministic case hand-computes the approved quantity in a comment so
the cap arithmetic is auditable; the randomized property test (plan §42)
asserts the risk invariants that must hold for EVERY approval: per-trade
risk never exceeds ``abs_max_trade_risk`` or the tier budget, portfolio heat
stays strictly below the reject threshold, and cash respects the regime
floor. Risk limits have priority over strategy confidence (plan §44 rule 20).
"""
import random

import pytest

from libs.trading_core.models import MarketRegime, RiskDecision
from libs.trading_core.risk import (
    PortfolioSnapshot,
    PositionRisk,
    RiskLimits,
    RiskRequest,
    assess,
    heat_state,
    portfolio_heat,
)

NAV = 100_000.0


def snap(
    nav: float = NAV,
    cash: float = NAV,
    positions: list[PositionRisk] | None = None,
    regime: MarketRegime = MarketRegime.STRONG_BULL,
    trading_enabled: bool = True,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        cash=cash,
        positions=positions or [],
        regime=regime,
        trading_enabled=trading_enabled,
    )


def req(
    ticker: str = "XOM",
    entry_price: float = 10.0,
    stop_distance: float = 1.0,
    edge: float = 90.0,
    quantity_requested: int | None = None,
) -> RiskRequest:
    return RiskRequest(
        ticker=ticker,
        entry_price=entry_price,
        stop_distance=stop_distance,
        edge=edge,
        quantity_requested=quantity_requested,
    )


# ---------------------------------------------------------------------------
# Step 1 — kill switch (plan §18, §44 rule 20: risk > confidence)
# ---------------------------------------------------------------------------


def test_kill_switch_rejects_even_very_strong_signal():
    # edge 90 = VERY_STRONG, but the kill switch has absolute priority.
    result = assess(req(edge=90.0), snap(trading_enabled=False))
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["KILL_SWITCH_ACTIVE"]
    assert result.approved_quantity == 0
    assert result.trade_risk_usd == 0.0
    assert result.explanations  # non-APPROVE must explain itself


# ---------------------------------------------------------------------------
# Step 2 — portfolio heat gate (plan §12.5)
# ---------------------------------------------------------------------------


def test_heat_at_reject_threshold_rejects_outright():
    # 8,000 / 100,000 = 8.00% == heat_reject -> reject NEW risk.
    positions = [PositionRisk("SPY", 40_000.0, 8_000.0)]
    result = assess(req(edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["HEAT_LIMIT"]
    assert result.heat_before_pct == pytest.approx(0.08)
    assert result.heat_after_pct == pytest.approx(0.08)  # nothing added
    assert result.explanations


def test_heat_above_reject_threshold_rejects_outright():
    positions = [PositionRisk("SPY", 40_000.0, 9_000.0)]  # 9% heat
    result = assess(req(edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["HEAT_LIMIT"]


def test_heat_just_below_reject_resizes_into_headroom():
    # Existing heat 7,900/100,000 = 7.9%. VERY_STRONG budget wants
    # floor(100,000 * 0.0125 / 1.0) = 1,250 shares. Heat headroom is
    # 8,000 - 7,900 = $100, but heat_after must stay STRICTLY below 8%,
    # so 100 shares (exactly 8.00%) is disallowed -> 99 shares.
    positions = [PositionRisk("SPY", 40_000.0, 7_900.0)]
    result = assess(req(edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 99
    assert "RESIZED_BY_HEAT_LIMIT" in result.reason_codes
    assert result.heat_after_pct == pytest.approx(0.07999)
    assert result.heat_after_pct < 0.08
    assert result.explanations


# ---------------------------------------------------------------------------
# Step 3 — signal strength tiers and budgets (plan §12.2)
# ---------------------------------------------------------------------------


def test_edge_below_weak_threshold_is_no_signal():
    result = assess(req(edge=10.0), snap())
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["SIGNAL_TOO_WEAK"]
    assert result.signal_strength is None
    assert result.risk_budget_pct is None
    assert result.explanations


@pytest.mark.parametrize(
    ("edge", "strength", "budget", "qty"),
    [
        # qty = floor(100,000 * budget / stop 1.0)
        (30.0, "WEAK", 0.005, 500),
        (50.0, "MODERATE", 0.0075, 750),
        (70.0, "STRONG", 0.01, 1_000),
        (90.0, "VERY_STRONG", 0.0125, 1_250),
    ],
)
def test_budget_tiers(edge, strength, budget, qty):
    result = assess(req(edge=edge), snap())
    assert result.decision is RiskDecision.APPROVE
    assert result.signal_strength == strength
    assert result.risk_budget_pct == pytest.approx(budget)
    assert result.approved_quantity == qty
    assert result.trade_risk_usd == pytest.approx(qty * 1.0)


def test_negative_edge_uses_absolute_value():
    # A bearish edge of -70 is still a STRONG signal (plan §12.2).
    result = assess(req(edge=-70.0), snap())
    assert result.signal_strength == "STRONG"
    assert result.approved_quantity == 1_000


def test_abs_max_trade_risk_overrides_any_budget():
    # Custom tier budget 5% must be clamped to abs_max_trade_risk 1.5%:
    # "No confidence score may override" (plan §12.2).
    limits = RiskLimits(budget_very_strong=0.05)
    result = assess(req(edge=90.0), snap(), limits)
    assert result.risk_budget_pct == pytest.approx(0.015)
    # floor(100,000 * 0.015 / 1.0) = 1,500; single-name risk cap is also
    # exactly 1,500 so it does not bind.
    assert result.approved_quantity == 1_500
    assert result.decision is RiskDecision.APPROVE


def test_custom_strength_thresholds_and_budgets_obeyed():
    # All thresholds are parameters (house rule): with weak lowered to 10
    # an edge of 15 becomes a valid WEAK signal with a 0.2% budget.
    limits = RiskLimits(strength_weak=10.0, budget_weak=0.002)
    result = assess(req(edge=15.0), snap(), limits)
    assert result.decision is RiskDecision.APPROVE
    assert result.signal_strength == "WEAK"
    # floor(100,000 * 0.002 / 1.0) = 200
    assert result.approved_quantity == 200


# ---------------------------------------------------------------------------
# Step 4 — base sizing (plan §12.1)
# ---------------------------------------------------------------------------


def test_stop_distance_must_be_positive():
    with pytest.raises(ValueError):
        assess(req(stop_distance=0.0), snap())
    with pytest.raises(ValueError):
        assess(req(stop_distance=-1.0), snap())


def test_quantity_requested_below_budget_is_plain_approve():
    # Budget allows 1,250; caller wants 300 -> approving the full request
    # is not a resize.
    result = assess(req(edge=90.0, quantity_requested=300), snap())
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 300


def test_quantity_requested_above_budget_gets_budget_quantity():
    result = assess(req(edge=90.0, quantity_requested=5_000), snap())
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 1_250


def test_budget_too_small_for_one_share_rejects_with_reason():
    # WEAK budget = $500; a $600 stop distance buys zero shares.
    result = assess(req(edge=30.0, stop_distance=600.0), snap())
    assert result.decision is RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert result.reason_codes == ["BUDGET_TOO_SMALL"]
    assert result.explanations


# ---------------------------------------------------------------------------
# Step 5a/5b — single-name caps (plan §12.3), each individually binding
# ---------------------------------------------------------------------------


def test_single_name_risk_cap_binds():
    # Existing NVDA max_loss $1,000 counts toward the 1.5% ($1,500) cap:
    # headroom = 1,500 - 1,000 = $500 -> floor(500 / 1.0) = 500 shares,
    # down from the VERY_STRONG budget quantity 1,250.
    positions = [PositionRisk("NVDA", 5_000.0, 1_000.0)]
    result = assess(req(ticker="NVDA", edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 500
    assert "RESIZED_BY_SINGLE_NAME_RISK_CAP" in result.reason_codes
    # Bucket cap was NOT the binder: 3,000 - 1,000 = 2,000 headroom > 500.
    assert not any(c.startswith("BUCKET_LIMIT") for c in result.reason_codes)
    assert not any("RESIZED_BY_BUCKET" in c for c in result.reason_codes)
    assert result.explanations


def test_single_name_capital_cap_binds():
    # Existing XOM market value $15,000; capital cap 20% of NAV = $20,000.
    # Headroom = $5,000 / entry $100 = 50 shares, down from the budget
    # quantity floor(1,250 / 1.0) = 1,250. (Existing XOM max_loss $200
    # leaves risk-cap headroom 1,300 > 1,250, so only capital binds.)
    positions = [PositionRisk("XOM", 15_000.0, 200.0)]
    result = assess(
        req(ticker="XOM", entry_price=100.0, edge=90.0),
        snap(positions=positions),
    )
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 50
    assert result.reason_codes == ["RESIZED_BY_SINGLE_NAME_CAPITAL_CAP"]
    assert result.explanations


# ---------------------------------------------------------------------------
# Step 5c — correlation bucket (plan §12.4)
# ---------------------------------------------------------------------------


def test_bucket_cap_binds_across_nvda_and_amd():
    # NVDA and AMD share TECH_MEGA. Existing NVDA bucket max_loss $2,500;
    # bucket cap 3% of NAV = $3,000 -> headroom $500 -> 500 AMD shares,
    # down from the budget quantity 1,250.
    positions = [PositionRisk("NVDA", 10_000.0, 2_500.0)]
    result = assess(req(ticker="AMD", edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 500
    assert "RESIZED_BY_BUCKET_LIMIT_TECH_MEGA" in result.reason_codes
    assert result.explanations


def test_non_bucket_ticker_unaffected_by_bucket_positions():
    # Same NVDA exposure, but XOM is not in TECH_MEGA: full budget approved.
    positions = [PositionRisk("NVDA", 10_000.0, 2_500.0)]
    result = assess(req(ticker="XOM", edge=90.0), snap(positions=positions))
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 1_250
    assert not any("BUCKET" in c for c in result.reason_codes)


# ---------------------------------------------------------------------------
# Step 5e — regime cash floor (plan §13): regime-dependent outcome for the
# SAME request
# ---------------------------------------------------------------------------


def test_cash_floor_is_regime_dependent():
    # nav 100,000, cash 40,000. STRONG budget qty = floor(1,000 / 1.0) =
    # 1,000 shares at $20 = $20,000 spend.
    request = req(entry_price=20.0, edge=70.0)

    # STRONG_BULL floor 15%: cash_after = 20,000 = 20% >= 15% -> APPROVE.
    bull = assess(request, snap(cash=40_000.0, regime=MarketRegime.STRONG_BULL))
    assert bull.decision is RiskDecision.APPROVE
    assert bull.approved_quantity == 1_000
    assert bull.cash_after_pct == pytest.approx(0.20)

    # MILD_BULL floor 25%: spend cap = 40,000 - 25,000 = 15,000 ->
    # floor(15,000 / 20) = 750 shares -> RESIZE.
    mild = assess(request, snap(cash=40_000.0, regime=MarketRegime.MILD_BULL))
    assert mild.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert mild.approved_quantity == 750
    assert mild.reason_codes == ["RESIZED_BY_CASH_FLOOR"]
    assert mild.cash_after_pct == pytest.approx(0.25)

    # STRONG_BEAR floor 60%: cash 40% is already below the floor -> the
    # same request is rejected purely by the regime floor.
    bear = assess(request, snap(cash=40_000.0, regime=MarketRegime.STRONG_BEAR))
    assert bear.decision is RiskDecision.REJECT
    assert bear.approved_quantity == 0
    assert bear.reason_codes == ["CASH_FLOOR"]
    assert bear.explanations


# ---------------------------------------------------------------------------
# Helpers (plan §12.5): heat and its 4% / 6% / 8% states
# ---------------------------------------------------------------------------


def test_portfolio_heat_arithmetic():
    positions = [
        PositionRisk("A", 10_000.0, 1_000.0),
        PositionRisk("B", 10_000.0, 1_500.0),
    ]
    assert portfolio_heat(positions, NAV) == pytest.approx(0.025)
    assert portfolio_heat([], NAV) == 0.0
    with pytest.raises(ValueError):
        portfolio_heat(positions, 0.0)


@pytest.mark.parametrize(
    ("heat", "state"),
    [
        (0.0, "NORMAL"),
        (0.039, "NORMAL"),
        (0.04, "ELEVATED"),
        (0.059, "ELEVATED"),
        (0.06, "HIGH"),
        (0.079, "HIGH"),
        (0.08, "BLOCKED"),
        (0.12, "BLOCKED"),
    ],
)
def test_heat_states_at_4_6_8_percent(heat, state):
    assert heat_state(heat) == state


def test_heat_state_uses_custom_limits():
    limits = RiskLimits(heat_elevated=0.01, heat_high=0.02, heat_reject=0.03)
    assert heat_state(0.015, limits) == "ELEVATED"
    assert heat_state(0.05, limits) == "BLOCKED"


# ---------------------------------------------------------------------------
# Determinism (plan §21 spirit: same inputs, same decision)
# ---------------------------------------------------------------------------


def test_assess_is_deterministic():
    positions = [PositionRisk("NVDA", 10_000.0, 2_500.0)]
    request = req(ticker="AMD", edge=90.0)
    first = assess(request, snap(positions=positions))
    second = assess(request, snap(positions=positions))
    assert first == second


# ---------------------------------------------------------------------------
# PROPERTY — randomized risk invariants (plan §42)
# ---------------------------------------------------------------------------


def _expected_budget(edge: float, limits: RiskLimits) -> float | None:
    """Independent re-derivation of the tier budget for the invariant check."""
    a = abs(edge)
    if a >= limits.strength_very_strong:
        return min(limits.budget_very_strong, limits.abs_max_trade_risk)
    if a >= limits.strength_strong:
        return min(limits.budget_strong, limits.abs_max_trade_risk)
    if a >= limits.strength_moderate:
        return min(limits.budget_moderate, limits.abs_max_trade_risk)
    if a >= limits.strength_weak:
        return min(limits.budget_weak, limits.abs_max_trade_risk)
    return None


def test_property_risk_invariants_hold_for_random_portfolios():
    """For every approval (plan §42): per-trade risk <= abs_max_trade_risk
    and <= the tier budget; heat_after strictly < heat_reject; cash_after
    respects the regime floor; every REJECT carries a reason code."""
    rng = random.Random(20260810)
    limits = RiskLimits()
    tickers = ["NVDA", "AMD", "MSFT", "AAPL", "XOM", "JPM", "SPY", "CVX"]
    approvals = rejects = 0

    for _ in range(200):
        nav = rng.uniform(50_000.0, 500_000.0)
        cash = rng.uniform(0.0, nav)
        positions = [
            PositionRisk(
                ticker=rng.choice(tickers),
                market_value=rng.uniform(0.0, nav * 0.15),
                max_loss=rng.uniform(0.0, nav * 0.02),
            )
            for _ in range(rng.randrange(0, 5))
        ]
        regime = rng.choice(list(MarketRegime))
        snapshot = PortfolioSnapshot(
            nav=nav,
            cash=cash,
            positions=positions,
            regime=regime,
            trading_enabled=rng.random() > 0.1,
        )
        entry = rng.uniform(5.0, 500.0)
        request = RiskRequest(
            ticker=rng.choice(tickers),
            entry_price=entry,
            stop_distance=rng.uniform(0.25, entry * 0.2),
            edge=rng.uniform(-100.0, 100.0),
            quantity_requested=rng.choice([None, rng.randrange(0, 2_000)]),
        )

        result = assess(request, snapshot, limits)
        qty = result.approved_quantity
        total_ml = sum(p.max_loss for p in positions)

        if result.decision in (
            RiskDecision.APPROVE,
            RiskDecision.APPROVE_WITH_RESIZE,
        ):
            approvals += 1
            assert qty > 0
            risk = qty * request.stop_distance
            # Absolute per-trade ceiling (plan §12.2).
            assert risk <= nav * limits.abs_max_trade_risk + 1e-9
            # Never above the step-4 tier budget (plan §12.1/§12.2).
            budget = _expected_budget(request.edge, limits)
            assert budget is not None
            assert risk <= nav * budget + 1e-9
            # Heat stays strictly below the reject threshold (plan §12.5),
            # recomputed from raw inputs.
            assert (total_ml + risk) / nav < limits.heat_reject
            # Cash respects the regime floor (plan §13).
            cash_after = cash - qty * request.entry_price
            assert cash_after / nav >= limits.cash_floors[regime] - 1e-9
            assert result.cash_after_pct == pytest.approx(cash_after / nav)
            if request.quantity_requested is not None:
                assert qty <= request.quantity_requested
        else:
            rejects += 1
            assert result.decision is RiskDecision.REJECT
            assert qty == 0
            assert result.reason_codes  # every REJECT explains itself

        if result.decision is not RiskDecision.APPROVE:
            assert result.explanations

    # The generator must actually exercise both outcomes.
    assert approvals > 20
    assert rejects > 20
