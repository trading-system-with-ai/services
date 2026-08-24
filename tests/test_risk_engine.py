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


# ---------------------------------------------------------------------------
# ADDITIVE §14 / §16 parameters (appended; all tests above are UNCHANGED).
# budget_multiplier scales the tier budget BEFORE the absolute cap; the
# portfolio greek limits check the post-trade book at the APPROVED quantity.
# ---------------------------------------------------------------------------

from libs.trading_core.greeks import (  # noqa: E402
    PortfolioGreeks,
    PositionGreeksInput,
)


def flat_greeks(
    delta_notional: float = 0.0,
    theta_per_day: float = 0.0,
    vega: float = 0.0,
) -> PortfolioGreeks:
    """Current-book greeks with only the exercised exposures non-zero."""
    return PortfolioGreeks(
        net_delta_shares=0.0,
        delta_adjusted_notional=delta_notional,
        net_gamma=0.0,
        net_theta_per_day=theta_per_day,
        net_vega=vega,
        per_position=(),
    )


def candidate(
    delta: float = 0.0,
    spot: float = 10.0,
    theta_per_day: float = 0.0,
    vega: float = 0.0,
    quantity: int = 25,
    multiplier: int = 100,
    instrument: str = "LONG_CALL",
) -> PositionGreeksInput:
    """Candidate per-share greeks; ``quantity`` is the requested basis —
    the engine must rescale to the APPROVED quantity."""
    return PositionGreeksInput(
        ticker="XOM",
        instrument=instrument,
        quantity=quantity,
        multiplier=multiplier,
        spot=spot,
        delta=delta,
        gamma=0.0,
        theta_per_day=theta_per_day,
        vega=vega,
    )


# --- §14 budget multiplier -------------------------------------------------


def test_budget_multiplier_half_halves_budget_bound_quantity():
    # Reference: VERY_STRONG budget 1.25% of 100,000 / stop 1.0 = 1,250.
    # Multiplier 0.5 -> effective budget 0.625% -> floor(625 / 1.0) = 625.
    base = assess(req(edge=90.0), snap())
    assert base.approved_quantity == 1_250
    halved = assess(req(edge=90.0), snap(), budget_multiplier=0.5)
    assert halved.decision is RiskDecision.APPROVE
    assert halved.approved_quantity == 625
    assert halved.risk_budget_pct == pytest.approx(0.00625)
    assert halved.trade_risk_usd == pytest.approx(625.0)


def test_budget_multiplier_never_overrides_abs_max_trade_risk():
    # 0.0125 * 10 = 12.5% raw, but §14 NEVER overrides hard caps:
    # effective = min(0.125, abs_max_trade_risk 0.015) = 1.5% -> floor(
    # 1,500 / 1.0) = 1,500 shares (single-name risk headroom is exactly
    # 1,500 as well, so the abs cap is what binds the budget).
    result = assess(req(edge=90.0), snap(), budget_multiplier=10.0)
    assert result.risk_budget_pct == pytest.approx(0.015)
    assert result.approved_quantity == 1_500
    assert result.decision is RiskDecision.APPROVE


def test_budget_multiplier_must_be_positive():
    with pytest.raises(ValueError):
        assess(req(edge=90.0), snap(), budget_multiplier=0.0)
    with pytest.raises(ValueError):
        assess(req(edge=90.0), snap(), budget_multiplier=-0.5)


# --- §16 portfolio greek limits, each firing individually ------------------


def test_portfolio_delta_limit_rejects_with_post_trade_numbers():
    # Book delta-adjusted notional $140,000. Approved qty is the budget
    # quantity 1,250 (stock, mult 1); its contribution is 1,250 * 1 * 1.0
    # * $10 = $12,500 -> post-trade $152,500 > cap 1.50 * 100,000 =
    # $150,000 -> PORTFOLIO_DELTA_LIMIT, rejected outright.
    result = assess(
        req(edge=90.0, entry_price=10.0, stop_distance=1.0),
        snap(),
        portfolio_greeks=flat_greeks(delta_notional=140_000.0),
        new_position_greeks=candidate(
            delta=1.0, spot=10.0, quantity=1_250, multiplier=1,
            instrument="LONG_STOCK",
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert result.reason_codes == ["PORTFOLIO_DELTA_LIMIT"]
    joined = " ".join(result.explanations)
    assert "152,500.00" in joined  # §36-style real numbers
    assert "150,000.00" in joined


def test_portfolio_theta_limit_rejects_individually():
    # Budget: floor(100,000 * 0.0125 / stop 50) = 25 contracts. Theta
    # contribution 25 * 100 * (-0.05) = -$125/day; book -$90/day -> post
    # -$215/day. Cap 0.001 * 100,000 = $100/day -> |−215| > 100 -> reject.
    # Delta (25*100*0.5*$10 = $12,500 < $150,000) and vega (25*100*0.1 =
    # $250 < $1,000) stay inside their caps: ONLY theta fires.
    result = assess(
        req(edge=90.0, entry_price=250.0, stop_distance=50.0),
        snap(),
        portfolio_greeks=flat_greeks(theta_per_day=-90.0),
        new_position_greeks=candidate(
            delta=0.5, theta_per_day=-0.05, vega=0.1
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert result.reason_codes == ["PORTFOLIO_THETA_LIMIT"]
    assert "215.00" in " ".join(result.explanations)


def test_portfolio_vega_limit_rejects_individually():
    # Book vega $900; contribution 25 * 100 * 0.2 = $500 -> post $1,400 >
    # cap 0.01 * 100,000 = $1,000 -> reject. Delta ($7,500 vs cap $150,000)
    # and theta (-$50 vs $100/day) stay inside: ONLY vega fires.
    result = assess(
        req(edge=90.0, entry_price=250.0, stop_distance=50.0),
        snap(),
        portfolio_greeks=flat_greeks(vega=900.0),
        new_position_greeks=candidate(
            delta=0.3, theta_per_day=-0.02, vega=0.2
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["PORTFOLIO_VEGA_LIMIT"]
    assert "1,400.00" in " ".join(result.explanations)


def test_all_three_greek_limits_fire_together_when_all_breach():
    # delta: 149,000 + 25*100*0.5*10 = 161,500 > 150,000
    # theta: -95 + 25*100*(-0.05) = -220 -> 220 > 100
    # vega:  990 + 25*100*0.1 = 1,240 > 1,000
    result = assess(
        req(edge=90.0, entry_price=250.0, stop_distance=50.0),
        snap(),
        portfolio_greeks=flat_greeks(
            delta_notional=149_000.0, theta_per_day=-95.0, vega=990.0
        ),
        new_position_greeks=candidate(
            delta=0.5, spot=10.0, theta_per_day=-0.05, vega=0.1
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == [
        "PORTFOLIO_DELTA_LIMIT",
        "PORTFOLIO_THETA_LIMIT",
        "PORTFOLIO_VEGA_LIMIT",
    ]


def test_greek_check_scales_to_approved_quantity_not_requested_basis():
    # Requested 50 contracts, but the budget approves only 25. At the
    # REQUESTED basis vega would breach: 50*100*0.03 = $150 -> 900 + 150 =
    # $1,050 > $1,000. At the APPROVED 25: 25*100*0.03 = $75 -> $975 <=
    # $1,000 -> no breach. The engine must scale to the approved quantity.
    result = assess(
        req(
            edge=90.0,
            entry_price=250.0,
            stop_distance=50.0,
            quantity_requested=50,
        ),
        snap(),
        portfolio_greeks=flat_greeks(vega=900.0),
        new_position_greeks=candidate(vega=0.03, quantity=50),
    )
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 25
    assert not any("PORTFOLIO" in c for c in result.reason_codes)


def test_exactly_at_greek_limit_passes():
    # 25 * 100 * 0.04 = $100 -> post vega 900 + 100 = $1,000 == cap:
    # exactly AT a greek limit is allowed; only strictly above rejects.
    result = assess(
        req(edge=90.0, entry_price=250.0, stop_distance=50.0),
        snap(),
        portfolio_greeks=flat_greeks(vega=900.0),
        new_position_greeks=candidate(vega=0.04),
    )
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 25
    assert result.reason_codes == []


def test_greek_limits_are_parameters():
    # Tighter vega cap 0.1% of NAV = $100: contribution 25*100*0.1 = $250
    # now breaches on an otherwise-empty book (thresholds parameterized).
    limits = RiskLimits(max_net_vega_pct_nav=0.001)
    result = assess(
        req(edge=90.0, entry_price=250.0, stop_distance=50.0),
        snap(),
        limits,
        portfolio_greeks=flat_greeks(),
        new_position_greeks=candidate(vega=0.1),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["PORTFOLIO_VEGA_LIMIT"]


# --- zero behavior change when the new params are omitted ------------------


def test_omitted_new_params_leave_reference_cases_identical():
    # A plain approve and a bucket-resize case, each computed with the new
    # parameters omitted and with their explicit defaults: the assessments
    # must be IDENTICAL (dataclass equality over every field).
    positions = [PositionRisk("NVDA", 10_000.0, 2_500.0)]
    for request, snapshot in (
        (req(edge=90.0), snap()),
        (req(ticker="AMD", edge=90.0), snap(positions=positions)),
    ):
        reference = assess(request, snapshot)
        explicit = assess(
            request,
            snapshot,
            budget_multiplier=1.0,
            portfolio_greeks=None,
            new_position_greeks=None,
        )
        assert reference == explicit


def test_greek_checks_require_both_arguments():
    # Supplying only ONE of the two greeks inputs runs no checks — the
    # post-trade book would be unknowable — so the result is identical to
    # omitting both, even with an absurd book exposure.
    reference = assess(req(edge=90.0), snap())
    only_portfolio = assess(
        req(edge=90.0),
        snap(),
        portfolio_greeks=flat_greeks(delta_notional=1e9),
    )
    only_candidate = assess(
        req(edge=90.0),
        snap(),
        new_position_greeks=candidate(
            delta=1.0, quantity=1_250, multiplier=1, instrument="LONG_STOCK"
        ),
    )
    assert only_portfolio == reference
    assert only_candidate == reference


# ===========================================================================
# assess_income — covered call / cash-secured put OPENS (risk-engine audit
# §8 item 3, §10 Phase B0; spec §2, §72). Every case hand-computes the
# approved contracts in a comment. NAV 100,000 unless stated.
# ===========================================================================

from libs.trading_core.risk import IncomeRiskRequest, assess_income  # noqa: E402


def csp(
    contracts: int = 1,
    strike: float = 100.0,
    credit: float = 2.0,
    ticker: str = "XOM",
) -> IncomeRiskRequest:
    """CSP bases: risk (strike − credit) × 100, capital strike × 100."""
    return IncomeRiskRequest(
        ticker=ticker,
        instrument="CASH_SECURED_PUT",
        contracts=contracts,
        risk_per_contract=(strike - credit) * 100,
        capital_per_contract=strike * 100,
    )


def cc(contracts: int = 1, ticker: str = "XOM") -> IncomeRiskRequest:
    """Covered call bases: risk 0 / capital 0 (stock row carries the heat)."""
    return IncomeRiskRequest(
        ticker=ticker,
        instrument="COVERED_CALL",
        contracts=contracts,
        risk_per_contract=0.0,
        capital_per_contract=0.0,
    )


def test_income_kill_switch_rejects_both_instruments():
    for request in (csp(), cc()):
        result = assess_income(request, snap(trading_enabled=False))
        assert result.decision is RiskDecision.REJECT
        assert result.reason_codes == ["KILL_SWITCH_ACTIVE"]
        assert result.approved_quantity == 0
        assert result.signal_strength is None
        assert result.risk_budget_pct is None


def test_income_heat_gate_rejects_at_threshold_even_zero_basis_cc():
    # Heat 8,000 / 100,000 = 8.00% >= heat_reject 8% -> REJECT, even for a
    # covered call whose own risk basis is zero (the gate is on the BOOK).
    positions = [PositionRisk("XOM", 50_000.0, 8_000.0)]
    result = assess_income(cc(), snap(positions=positions))
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["HEAT_LIMIT"]
    assert result.heat_before_pct == pytest.approx(0.08)


def test_income_abs_trade_risk_cap_resizes_csp():
    # NAV 1,000,000 cash 1,000,000. Strike 100 / credit 2 -> risk 9,800 per
    # contract; abs cap 1.5% * 1,000,000 = 15,000 -> floor(15,000 / 9,800)
    # = 1 contract. Single-name risk cap is ALSO 15,000 (same 1.5%), so it
    # cannot bind further; capital 10,000 per contract vs 200,000 cap ok;
    # cash floor 15% -> (1,000,000 − 150,000)/10,000 = 85 ok.
    result = assess_income(csp(contracts=3), snap(nav=1_000_000.0, cash=1_000_000.0))
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 1
    assert result.reason_codes == ["RESIZED_BY_ABS_TRADE_RISK_CAP"]
    assert result.trade_risk_usd == pytest.approx(9_800.0)
    assert result.heat_after_pct == pytest.approx(0.0098)
    assert result.cash_after_pct == pytest.approx(0.99)
    assert "$29,400.00" in result.explanations[0]  # 3 * 9,800 named
    assert result.signal_strength is None and result.risk_budget_pct is None


def test_income_abs_trade_risk_cap_zeroes_to_reject_on_small_nav():
    # NAV 100,000: abs cap 1,500 < 9,800 per contract -> 0 -> REJECT with
    # the bare code (not RESIZED_BY_).
    result = assess_income(csp(contracts=1), snap())
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["ABS_TRADE_RISK_CAP"]
    assert result.approved_quantity == 0
    assert result.trade_risk_usd == 0.0


def test_income_single_name_risk_cap_binds_csp():
    # NAV 1,000,000, XOM already carries 6,000 max loss (0.6%). Single-name
    # cap 15,000 - 6,000 = 9,000 headroom < 9,800 -> 0. Abs cap alone would
    # allow 1 (15,000 / 9,800). Order: abs cap first (3 -> 1, RESIZED), then
    # single-name (1 -> 0, bare code) -> REJECT with both reasons.
    positions = [PositionRisk("XOM", 60_000.0, 6_000.0)]
    result = assess_income(
        csp(contracts=3), snap(nav=1_000_000.0, cash=940_000.0, positions=positions)
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == [
        "RESIZED_BY_ABS_TRADE_RISK_CAP",
        "SINGLE_NAME_RISK_CAP",
    ]
    assert result.approved_quantity == 0


def test_income_single_name_capital_cap_binds_on_capital_basis():
    # Small risk basis so only capital binds: strike 100 / credit 99 -> risk
    # 100 per contract (abs cap 15,000/100 = 150 fine). NAV 1,000,000, XOM
    # market value 195,000; capital cap 200,000 - 195,000 = 5,000 -> floor(
    # 5,000 / 10,000) = 0 -> SINGLE_NAME_CAPITAL_CAP REJECT.
    positions = [PositionRisk("XOM", 195_000.0, 1_000.0)]
    result = assess_income(
        csp(contracts=1, strike=100.0, credit=99.0),
        snap(nav=1_000_000.0, cash=805_000.0, positions=positions),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["SINGLE_NAME_CAPITAL_CAP"]
    assert "$195,000.00" in result.explanations[0]


def test_income_bucket_cap_binds_on_risk_basis():
    # NVDA and AMD share TECH_MEGA (bucket cap 3% = 30,000 on NAV 1,000,000).
    # AMD carries 25,000 max loss -> bucket headroom 5,000 < 9,800 -> 0.
    # abs cap: 15,000/9,800 = 1 (no resize for 1 requested); single-name NVDA
    # 15,000/9,800 = 1 ok; then bucket -> 0 -> REJECT.
    positions = [PositionRisk("AMD", 100_000.0, 25_000.0)]
    result = assess_income(
        csp(contracts=1, ticker="NVDA"),
        snap(nav=1_000_000.0, cash=900_000.0, positions=positions),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["BUCKET_LIMIT_TECH_MEGA"]


def test_income_heat_headroom_strictly_below_reject():
    # NAV 1,000,000, book heat 70,200 (7.02%): headroom to 8% = 9,800 —
    # EXACTLY one contract's risk, but heat must stay STRICTLY below 8%, so
    # 1 contract (heat 80,000 = 8.00%) is not allowed -> 0 -> REJECT.
    # (abs cap 15,000/9,800 = 1; single-name XOM has 0 -> 1; no bucket.)
    positions = [PositionRisk("SPY", 500_000.0, 70_200.0)]
    result = assess_income(
        csp(contracts=1), snap(nav=1_000_000.0, cash=500_000.0, positions=positions)
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["HEAT_LIMIT"]
    # One dollar less of book heat and the contract fits: 70,199 + 9,800 =
    # 79,999 < 80,000.
    positions = [PositionRisk("SPY", 500_000.0, 70_199.0)]
    result = assess_income(
        csp(contracts=1), snap(nav=1_000_000.0, cash=500_000.0, positions=positions)
    )
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 1
    assert result.heat_after_pct == pytest.approx(0.079999)


def test_income_cash_floor_binds_on_reservation_basis():
    # NAV 1,000,000, STRONG_BULL floor 15% = 150,000. Usable cash 165,000 ->
    # (165,000 − 150,000) / 10,000 = 1 contract; risk basis 100 per contract
    # (strike 100 / credit 99) so nothing else binds for 3 requested ->
    # RESIZED_BY_CASH_FLOOR to 1; cash after = 155,000 / 1,000,000 = 15.5%.
    positions = [PositionRisk("SPY", 835_000.0, 1_000.0)]
    result = assess_income(
        csp(contracts=3, strike=100.0, credit=99.0),
        snap(nav=1_000_000.0, cash=165_000.0, positions=positions),
    )
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 1
    assert result.reason_codes == ["RESIZED_BY_CASH_FLOOR"]
    assert result.cash_after_pct == pytest.approx(0.155)
    # A regime with a 60% floor rejects outright (165,000 < 600,000).
    result = assess_income(
        csp(contracts=3, strike=100.0, credit=99.0),
        snap(
            nav=1_000_000.0,
            cash=165_000.0,
            positions=positions,
            regime=MarketRegime.STRONG_BEAR,
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["CASH_FLOOR"]


def test_income_greek_breach_rejects_with_negated_short_leg():
    # Covered call, zero bases -> only greeks can bind. The caller passes the
    # short call's greeks NEGATED: delta −0.30 per share, spot 100, mult 100,
    # 5 contracts -> −0.30 * 100 * 5 * 100 = −15,000 delta notional. Book
    # delta notional −140,000 -> post −155,000; |155,000| > 150% * 100,000
    # = 150,000 -> PORTFOLIO_DELTA_LIMIT REJECT. Theta/vega flat.
    result = assess_income(
        cc(contracts=5),
        snap(),
        portfolio_greeks=flat_greeks(delta_notional=-140_000.0),
        new_position_greeks=candidate(
            delta=-0.30, spot=100.0, quantity=1, instrument="COVERED_CALL"
        ),
    )
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["PORTFOLIO_DELTA_LIMIT"]
    assert "$155,000.00" in result.explanations[0]
    # Short-premium hedges the book: book +140,000, short call −15,000 ->
    # 125,000 -> passes.
    result = assess_income(
        cc(contracts=5),
        snap(),
        portfolio_greeks=flat_greeks(delta_notional=140_000.0),
        new_position_greeks=candidate(
            delta=-0.30, spot=100.0, quantity=1, instrument="COVERED_CALL"
        ),
    )
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 5


def test_income_greek_check_needs_both_inputs():
    reference = assess_income(cc(contracts=5), snap())
    only_book = assess_income(
        cc(contracts=5), snap(), portfolio_greeks=flat_greeks(delta_notional=1e9)
    )
    assert only_book == reference
    assert reference.decision is RiskDecision.APPROVE


def test_income_covered_call_zero_basis_only_kill_heat_greeks_can_bind():
    # A covered call carries risk 0 / capital 0: with an EMPTY cash balance,
    # a heavy same-name book and a filled bucket, every sizing clamp is
    # skipped (nothing to divide by, nothing to add) and 7 contracts are
    # approved with heat / cash unchanged.
    positions = [
        PositionRisk("NVDA", 900_000.0, 14_000.0),  # 1.4% single-name risk
        PositionRisk("AMD", 50_000.0, 15_000.0),  # TECH_MEGA bucket 2.9%
        PositionRisk("SPY", 50_000.0, 40_000.0),  # book heat 6.9% (< 8%)
    ]
    result = assess_income(
        cc(contracts=7, ticker="NVDA"),
        snap(nav=1_000_000.0, cash=0.0, positions=positions),
    )
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 7
    assert result.reason_codes == []
    assert result.trade_risk_usd == 0.0
    assert result.heat_after_pct == pytest.approx(0.069)
    assert result.cash_after_pct == pytest.approx(0.0)


def test_income_zero_contracts_rejects_with_reason():
    result = assess_income(cc(contracts=0), snap())
    assert result.decision is RiskDecision.REJECT
    assert result.reason_codes == ["ZERO_QUANTITY_REQUESTED"]


def test_income_request_validation():
    with pytest.raises(ValueError):
        assess_income(
            IncomeRiskRequest("XOM", "LONG_CALL", 1, 100.0, 100.0), snap()
        )
    with pytest.raises(ValueError):
        assess_income(
            IncomeRiskRequest("XOM", "CASH_SECURED_PUT", 1, -1.0, 100.0), snap()
        )
    with pytest.raises(ValueError):
        assess_income(
            IncomeRiskRequest("XOM", "CASH_SECURED_PUT", 1, 100.0, -1.0), snap()
        )
    with pytest.raises(ValueError):
        assess_income(
            IncomeRiskRequest("XOM", "CASH_SECURED_PUT", -1, 100.0, 100.0), snap()
        )
    # Non-finite bases are refused too (a NaN basis compares False against
    # every clamp and would otherwise skip them all — QA probe 2026-08-17).
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assess_income(
                IncomeRiskRequest("XOM", "CASH_SECURED_PUT", 1, bad, 100.0), snap()
            )
        with pytest.raises(ValueError):
            assess_income(
                IncomeRiskRequest("XOM", "CASH_SECURED_PUT", 1, 100.0, bad), snap()
            )


def test_income_thresholds_are_parameters():
    # Raising abs_max_trade_risk to 10% lets the 100,000-NAV CSP through:
    # 10,000 / 9,800 = 1; single-name risk also raised to 10%; heat reject
    # raised to 12% (headroom 12,000 > 9,800; the default 8,000 would bind);
    # capital 10,000 vs 20% cap fine; cash floor 15% -> (100,000 −
    # 15,000)/10,000 = 8.
    limits = RiskLimits(
        abs_max_trade_risk=0.10, single_name_risk=0.10, heat_reject=0.12
    )
    result = assess_income(csp(contracts=1), snap(), limits)
    assert result.decision is RiskDecision.APPROVE
    assert result.approved_quantity == 1
    assert result.trade_risk_usd == pytest.approx(9_800.0)


def test_income_does_not_touch_assess():
    # The additive engine leaves the reference stock case byte-identical.
    reference = assess(req(edge=90.0), snap())
    assert reference.approved_quantity == 1_250
    assert reference.decision is RiskDecision.APPROVE


def test_income_contracts_must_be_a_real_int():
    """QA follow-up: a float or bool quantity must not flow through to
    ``approved_quantity`` — the ledger counts whole contracts."""
    for bad in (1.5, True, "2"):
        with pytest.raises(ValueError):
            assess_income(
                IncomeRiskRequest(
                    ticker="XOM",
                    instrument="CASH_SECURED_PUT",
                    contracts=bad,  # type: ignore[arg-type]
                    risk_per_contract=9_800.0,
                    capital_per_contract=10_000.0,
                ),
                snap(),
            )


# ---------------------------------------------------------------------------
# Phase C (design contract §7.3) — ADDITIVE engine surface, SHADOW by default.
#
# Three things are pinned below and nothing else may move:
#   1. ``assess(extra_caps=...)`` applies caller-supplied caps through the
#      SAME clamp closure as the hard caps, AFTER the cash floor and BEFORE
#      the greek check, recording the cap's own layer;
#   2. ``binding_constraints`` is a TOTAL mapping of ``reason_codes`` for
#      EVERY decision of BOTH pipelines;
#   3. with default arguments nothing changed at all — a battery of 200+
#      seeded calls must produce byte-identical decisions.
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from libs.trading_core.risk import (  # noqa: E402
    LAYER_HARD_LIMIT,
    BindingConstraint,
    IncomeRiskRequest,
    RiskAssessment,
    assess_income,
)
from libs.trading_core.risk.engine import _binding_constraints  # noqa: E402
from libs.trading_core.risk.pretrade import (  # noqa: E402
    LAYER_CONCENTRATION,
    LAYER_STATISTICAL,
    QuantityCap,
)


# ---------------------------------------------------------------------------
# The byte-identity battery (contract §7.3)
#
# The battery is a GENERATOR of engine INPUTS, not a copy of engine logic:
# it produces seeded (request, snapshot, limits, multiplier, greeks) tuples
# spanning every branch — kill switch, heat gate, weak signal, each cap,
# the greek reject, the zero-quantity paths — and the test records what the
# CURRENT engine returns for each. Because the expectations are the engine's
# own output rather than re-derived arithmetic, the test proves exactly one
# thing, which is the thing the contract asks for: adding ``extra_caps`` and
# the two new fields did not perturb any default-argument decision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    """One battery case: the inputs of a single ``assess`` call."""

    request: RiskRequest
    snapshot: PortfolioSnapshot
    limits: RiskLimits
    budget_multiplier: float

    def run(self, **kwargs) -> RiskAssessment:
        return assess(
            self.request,
            self.snapshot,
            self.limits,
            budget_multiplier=self.budget_multiplier,
            **kwargs,
        )

    def fingerprint(self, result: RiskAssessment) -> tuple:
        """The four fields the contract requires to be byte-identical."""
        return (
            result.decision,
            result.approved_quantity,
            tuple(result.reason_codes),
            tuple(result.explanations),
        )


def battery(n: int = 240, seed: int = 20260818) -> list[_Case]:
    """Generate ``n`` seeded, deterministic ``assess`` input tuples.

    Ranges are chosen so the whole decision tree is exercised: edges below
    and above every strength threshold, heats on both sides of the reject
    gate, cash from 0 to full NAV, every regime, positions in and out of the
    TECH_MEGA bucket, stop distances that both do and do not divide the
    budget, explicit requested quantities including 0, and occasional
    non-default limits. Deterministic for a fixed seed, so the "before" and
    "after" runs see identical inputs.
    """
    rng = random.Random(seed)
    regimes = list(MarketRegime)
    tickers = ["XOM", "NVDA", "AAPL", "KO", "QQQ"]
    cases: list[_Case] = []
    for i in range(n):
        nav = rng.choice([25_000.0, 100_000.0, 250_000.0])
        n_pos = rng.randint(0, 4)
        positions = [
            PositionRisk(
                ticker=rng.choice(tickers),
                market_value=round(rng.uniform(0.0, nav * 0.25), 2),
                max_loss=round(rng.uniform(0.0, nav * 0.03), 2),
            )
            for _ in range(n_pos)
        ]
        request = RiskRequest(
            ticker=rng.choice(tickers),
            entry_price=round(rng.uniform(5.0, 400.0), 2),
            stop_distance=round(rng.uniform(0.25, 40.0), 2),
            edge=round(rng.uniform(-100.0, 100.0), 1),
            quantity_requested=rng.choice([None, 0, 1, 7, 50, 5_000]),
        )
        snapshot = PortfolioSnapshot(
            nav=nav,
            cash=round(rng.uniform(0.0, nav), 2),
            positions=positions,
            regime=rng.choice(regimes),
            # Every ~12th case trips the kill switch.
            trading_enabled=(i % 12 != 0),
        )
        limits = (
            RiskLimits()
            if i % 5
            else RiskLimits(
                abs_max_trade_risk=round(rng.uniform(0.005, 0.05), 4),
                single_name_risk=round(rng.uniform(0.005, 0.05), 4),
                heat_reject=round(rng.uniform(0.02, 0.15), 4),
            )
        )
        cases.append(
            _Case(
                request=request,
                snapshot=snapshot,
                limits=limits,
                budget_multiplier=rng.choice([1.0, 0.5, 0.75, 1.5, 2.0]),
            )
        )
    return cases


def test_battery_covers_the_whole_decision_tree():
    """The byte-identity proof is only worth as much as its coverage."""
    results = [c.run() for c in battery()]
    decisions = {r.decision for r in results}
    assert decisions == {
        RiskDecision.APPROVE,
        RiskDecision.APPROVE_WITH_RESIZE,
        RiskDecision.REJECT,
    }
    codes = {code for r in results for code in r.reason_codes}
    # Every early-reject branch and at least the main sizing caps must fire.
    for expected in (
        "KILL_SWITCH_ACTIVE",
        "SIGNAL_TOO_WEAK",
        "RESIZED_BY_SINGLE_NAME_RISK_CAP",
        "RESIZED_BY_CASH_FLOOR",
    ):
        assert expected in codes, f"battery never exercised {expected}"
    assert len(results) >= 200


def test_assess_with_default_arguments_is_byte_identical():
    """Contract §7.3: ``extra_caps`` defaults to ``()`` and MUST change nothing.

    The battery is run twice — once plainly and once passing the new
    argument explicitly at its default — and the decision, approved
    quantity, reason codes and explanations must match EXACTLY, field for
    field, on all 240 cases. Any drift in the sizing pipeline, in a clamp
    sentence, or in the ordering of reason codes fails here.
    """
    cases = battery()
    assert len(cases) >= 200
    plain = [c.fingerprint(c.run()) for c in cases]
    explicit = [c.fingerprint(c.run(extra_caps=())) for c in cases]
    assert plain == explicit

    # And the reference case from the pre-Phase-C suite is untouched.
    reference = assess(req(edge=90.0), snap())
    assert reference.decision is RiskDecision.APPROVE
    assert reference.approved_quantity == 1_250
    assert reference.reason_codes == []


def test_extra_caps_never_raise_the_quantity():
    """A cap can only REDUCE — the statistical layer cannot grant risk."""
    for cap_qty in (2_000, 10_000):
        result = assess(
            req(edge=90.0),
            snap(),
            extra_caps=[
                QuantityCap(
                    code="PORTFOLIO_ES_LIMIT",
                    layer=LAYER_STATISTICAL,
                    cap_qty=cap_qty,
                    sentence="ES headroom is ample.",
                )
            ],
        )
        assert result.approved_quantity == 1_250  # unchanged
        assert result.decision is RiskDecision.APPROVE
        assert result.reason_codes == []


# ---------------------------------------------------------------------------
# extra_caps: clamp, code, layer, sentence (contract §7.3)
# ---------------------------------------------------------------------------


def test_extra_cap_resizes_and_records_code_layer_and_sentence():
    # Baseline approves 1 250 shares (100 000 * 1.25% / 1.00).
    cap = QuantityCap(
        code="PORTFOLIO_ES_LIMIT",
        layer=LAYER_STATISTICAL,
        cap_qty=400,
        sentence=(
            "Portfolio ES-95 (1D) would be $6,200.00 (6.20% of NAV) at 1,250 "
            "shares, above the 5.00%-of-NAV limit ($5,000.00); quantity "
            "reduced from 1,250 to 400."
        ),
    )
    result = assess(req(edge=90.0), snap(), extra_caps=[cap])
    assert result.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert result.approved_quantity == 400
    assert result.reason_codes == ["RESIZED_BY_PORTFOLIO_ES_LIMIT"]
    # The cap's own sentence is recorded verbatim (audit-exact, spec §47).
    assert result.explanations[0] == cap.sentence
    assert result.binding_constraints == (
        BindingConstraint("RESIZED_BY_PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL),
    )
    assert result.requested_quantity is None  # none was requested


def test_extra_cap_at_zero_rejects_with_the_bare_code():
    cap = QuantityCap(
        code="ES_CONTRIBUTION_CAP",
        layer=LAYER_CONCENTRATION,
        cap_qty=0,
        sentence="XOM would hold 91.0% of ES-95 contributions; no size is safe.",
    )
    result = assess(req(edge=90.0), snap(), extra_caps=[cap])
    assert result.decision is RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert result.reason_codes == ["ES_CONTRIBUTION_CAP"]
    assert result.binding_constraints == (
        BindingConstraint("ES_CONTRIBUTION_CAP", LAYER_CONCENTRATION),
    )


def test_extra_caps_apply_in_order_and_only_the_binding_ones_are_recorded():
    caps = [
        QuantityCap("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL, 900, "ES caps at 900."),
        QuantityCap("ES_CONTRIBUTION_CAP", LAYER_CONCENTRATION, 300, "RC caps at 300."),
        # Already above the running quantity (300) — must not bind.
        QuantityCap("INCREMENTAL_ES_CAP", LAYER_STATISTICAL, 800, "Increment caps at 800."),
    ]
    result = assess(req(edge=90.0), snap(), extra_caps=caps)
    assert result.approved_quantity == 300
    assert result.reason_codes == [
        "RESIZED_BY_PORTFOLIO_ES_LIMIT",
        "RESIZED_BY_ES_CONTRIBUTION_CAP",
    ]
    assert [bc.layer for bc in result.binding_constraints] == [
        LAYER_STATISTICAL,
        LAYER_CONCENTRATION,
    ]


def test_extra_caps_run_after_the_cash_floor_and_before_the_greeks():
    """Contract §7.3 placement, proven by consequence rather than by reading
    the source: the cash floor still binds first (its code comes earlier),
    and the greek check evaluates the CAPPED quantity (the same greeks that
    reject at 1 250 pass at 100)."""
    # Cash 20 000 with a 15% floor on 100 000 NAV leaves 5 000 -> 500 shares
    # at $10 entry; the statistical cap then cuts 500 to 100.
    cash_snap = snap(cash=20_000.0)
    caps = [QuantityCap("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL, 100, "ES caps at 100.")]
    result = assess(req(edge=90.0), cash_snap, extra_caps=caps)
    assert result.approved_quantity == 100
    assert result.reason_codes == [
        "RESIZED_BY_CASH_FLOOR",
        "RESIZED_BY_PORTFOLIO_ES_LIMIT",
    ]

    # Greeks priced so 1 250 shares breach the delta limit but 100 do not:
    # the limit is 150% of 100 000 = 150 000 of |delta notional|, and the
    # candidate carries 1.0 delta x 200 spot x multiplier 1 = 200 per share.
    book_greeks = flat_greeks()
    cand_greeks = candidate(delta=1.0, spot=200.0, multiplier=1, quantity=1)
    uncapped = assess(
        req(edge=90.0), snap(),
        portfolio_greeks=book_greeks, new_position_greeks=cand_greeks,
    )
    # 1 250 x 200 = 250 000 > 150 000 -> the greek check rejects.
    assert uncapped.decision is RiskDecision.REJECT
    assert "PORTFOLIO_DELTA_LIMIT" in uncapped.reason_codes

    capped = assess(
        req(edge=90.0), snap(),
        portfolio_greeks=book_greeks, new_position_greeks=cand_greeks,
        extra_caps=[QuantityCap("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL, 100, "ES caps at 100.")],
    )
    # 100 x 200 = 20 000 <= 150 000 -> the greek check now passes, which is
    # only possible if the cap was applied BEFORE it.
    assert capped.decision is RiskDecision.APPROVE_WITH_RESIZE
    assert capped.approved_quantity == 100
    assert "PORTFOLIO_DELTA_LIMIT" not in capped.reason_codes


def test_a_hard_limit_still_wins_over_a_looser_statistical_cap():
    """Spec §38: the statistical layer never softens a hard limit."""
    result = assess(
        req(edge=90.0),
        snap(cash=20_000.0),  # cash floor caps at 500
        extra_caps=[QuantityCap("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL, 900, "ES caps at 900.")],
    )
    assert result.approved_quantity == 500
    assert result.reason_codes == ["RESIZED_BY_CASH_FLOOR"]
    assert result.binding_constraints == (
        BindingConstraint("RESIZED_BY_CASH_FLOOR", LAYER_HARD_LIMIT),
    )


def test_extra_caps_do_not_reach_the_early_reject_branches():
    """A kill switch outranks everything, statistical caps included."""
    result = assess(
        req(edge=90.0),
        snap(trading_enabled=False),
        extra_caps=[QuantityCap("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL, 900, "ES caps at 900.")],
    )
    assert result.reason_codes == ["KILL_SWITCH_ACTIVE"]
    assert result.binding_constraints == (
        BindingConstraint("KILL_SWITCH_ACTIVE", LAYER_HARD_LIMIT),
    )


# ---------------------------------------------------------------------------
# binding_constraints: a TOTAL mapping for every code the engine can emit
# ---------------------------------------------------------------------------

#: Every reason code either pipeline can produce (contract §7.3). Kept as an
#: explicit list so a NEW code added to the engine without a decision about
#: its layer fails the totality test below rather than silently defaulting.
ALL_ENGINE_REASON_CODES = [
    "KILL_SWITCH_ACTIVE",
    "HEAT_LIMIT",
    "SIGNAL_TOO_WEAK",
    "BUDGET_TOO_SMALL",
    "ZERO_QUANTITY_REQUESTED",
    "ABS_TRADE_RISK_CAP",
    "SINGLE_NAME_RISK_CAP",
    "SINGLE_NAME_CAPITAL_CAP",
    "BUCKET_LIMIT_TECH_MEGA",
    "CASH_FLOOR",
    "PORTFOLIO_DELTA_LIMIT",
    "PORTFOLIO_THETA_LIMIT",
    "PORTFOLIO_VEGA_LIMIT",
]


def test_binding_constraint_mapping_is_total_over_every_engine_code():
    """Every code, bare and RESIZED_BY_-prefixed, maps to HARD_LIMIT."""
    codes = ALL_ENGINE_REASON_CODES + [
        f"RESIZED_BY_{c}" for c in ALL_ENGINE_REASON_CODES
    ]
    constraints = _binding_constraints(codes)
    assert len(constraints) == len(codes)
    assert [bc.code for bc in constraints] == codes
    assert {bc.layer for bc in constraints} == {LAYER_HARD_LIMIT}


def test_binding_constraint_mapping_is_total_over_unknown_codes_too():
    """A code the mapping has never seen resolves to HARD_LIMIT — the mapping
    can never raise, and a future Tier 0 code is a hard limit by default."""
    assert _binding_constraints(["SOME_FUTURE_TIER_0_CODE"]) == (
        BindingConstraint("SOME_FUTURE_TIER_0_CODE", LAYER_HARD_LIMIT),
    )
    assert _binding_constraints([]) == ()


def test_binding_constraint_strips_the_resize_prefix_when_resolving_a_layer():
    """``RESIZED_BY_<CAP>`` and ``<CAP>`` are one constraint at two severities."""
    caps = {"PORTFOLIO_ES_LIMIT": LAYER_STATISTICAL}
    assert _binding_constraints(["RESIZED_BY_PORTFOLIO_ES_LIMIT"], caps) == (
        BindingConstraint("RESIZED_BY_PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL),
    )
    assert _binding_constraints(["PORTFOLIO_ES_LIMIT"], caps) == (
        BindingConstraint("PORTFOLIO_ES_LIMIT", LAYER_STATISTICAL),
    )


def test_binding_constraints_are_populated_for_every_assess_decision():
    """Contract §7.3: 'populated for EVERY decision'. The battery checks all
    240 cases — the constraints must mirror ``reason_codes`` one for one."""
    for case in battery():
        result = case.run()
        assert [bc.code for bc in result.binding_constraints] == result.reason_codes
        assert all(bc.layer == LAYER_HARD_LIMIT for bc in result.binding_constraints)
        assert result.requested_quantity == case.request.quantity_requested


def test_binding_constraints_are_populated_for_every_assess_income_decision():
    cases = [
        # (contracts, risk, capital, snapshot) spanning approve/resize/reject.
        (1, 100.0, 1_000.0, snap()),
        (50, 500.0, 2_000.0, snap()),
        (1, 9_800.0, 10_000.0, snap()),
        (2, 100.0, 1_000.0, snap(trading_enabled=False)),
        (0, 100.0, 1_000.0, snap()),
        (3, 0.0, 0.0, snap()),  # covered-call shape: both bases zero
        (5, 200.0, 500.0, snap(cash=1_000.0)),
        (
            4,
            300.0,
            900.0,
            snap(positions=[PositionRisk("XOM", 0.0, 8_000.0)]),
        ),
    ]
    seen_decisions = set()
    for contracts, rp, cp, snapshot in cases:
        result = assess_income(
            IncomeRiskRequest("XOM", "CASH_SECURED_PUT", contracts, rp, cp),
            snapshot,
        )
        seen_decisions.add(result.decision)
        assert [bc.code for bc in result.binding_constraints] == result.reason_codes
        assert all(
            bc.layer == LAYER_HARD_LIMIT for bc in result.binding_constraints
        )
        assert result.requested_quantity == contracts
    # The set of cases really did span all three decisions.
    assert seen_decisions == {
        RiskDecision.APPROVE,
        RiskDecision.APPROVE_WITH_RESIZE,
        RiskDecision.REJECT,
    }


def test_requested_quantity_sits_next_to_approved_quantity():
    """Spec §47: 'Requested: 4 contracts / Approved: 2 contracts'."""
    result = assess(req(edge=90.0, quantity_requested=100), snap())
    assert result.requested_quantity == 100
    assert result.approved_quantity == 100  # budget allows 1 250; 100 requested

    income = assess_income(
        IncomeRiskRequest("XOM", "CASH_SECURED_PUT", 4, 500.0, 2_000.0), snap()
    )
    assert income.requested_quantity == 4
    assert income.approved_quantity <= 4


def test_new_risk_assessment_fields_default_so_old_construction_sites_work():
    """Additive-only: the two fields have defaults, so every pre-Phase-C
    construction of a ``RiskAssessment`` still type-checks and runs."""
    a = RiskAssessment(
        decision=RiskDecision.APPROVE,
        approved_quantity=10,
        signal_strength="STRONG",
        risk_budget_pct=0.01,
        trade_risk_usd=100.0,
        reason_codes=[],
        explanations=[],
        heat_before_pct=0.01,
        heat_after_pct=0.02,
        cash_after_pct=0.5,
    )
    assert a.requested_quantity is None
    assert a.binding_constraints == ()
