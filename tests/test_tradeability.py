"""Tradeability layer v1 (upgrade 2026-08-12 §9/§10).

Pins the four states, the verdict precedence, the §10 "strong direction but
blocked environment" scenario, and that every check is reported PASS or not
(auditability contract). Direction never appears in the inputs — the layer
is direction-agnostic by construction (§9).
"""
from libs.trading_core.models import IVRegime, MarketRegime, TradeabilityState
from libs.trading_core.tradeability import (
    TradeabilityParams,
    assess_tradeability,
)

BULL = MarketRegime.STRONG_BULL
TRANSITION = MarketRegime.TRANSITION


def _assess(**overrides):
    """Baseline: fresh, deep data in a clean bull environment, NORMAL vol."""
    kwargs = dict(
        bar_count=600,
        stale_trading_days=0,
        market_regime=BULL,
        symbol_regime=MarketRegime.MILD_BULL,
        vol_regime=IVRegime.NORMAL,
    )
    kwargs.update(overrides)
    return assess_tradeability(**kwargs)


def test_clean_environment_is_tradeable():
    decision = _assess()
    assert decision.state is TradeabilityState.TRADEABLE
    assert decision.reasons == []
    assert decision.version == TradeabilityParams().version
    # Auditability: every check reported, all PASS.
    assert {c.name for c in decision.checks} == {
        "DATA_QUALITY", "DATA_FRESHNESS", "MARKET_REGIME",
        "SYMBOL_REGIME", "VOLATILITY_REGIME",
    }
    assert all(c.status == "PASS" for c in decision.checks)


def test_transition_symbol_regime_blocks():
    decision = _assess(symbol_regime=TRANSITION)
    assert decision.state is TradeabilityState.BLOCKED
    assert any("SYMBOL_REGIME" in r and "TRANSITION" in r for r in decision.reasons)


def test_transition_market_regime_blocks():
    decision = _assess(market_regime=TRANSITION)
    assert decision.state is TradeabilityState.BLOCKED


def test_extreme_volatility_blocks():
    decision = _assess(vol_regime=IVRegime.EXTREME)
    assert decision.state is TradeabilityState.BLOCKED
    assert any("EXTREME volatility" in r for r in decision.reasons)


def test_high_volatility_is_conditional():
    decision = _assess(vol_regime=IVRegime.HIGH)
    assert decision.state is TradeabilityState.CONDITIONAL


def test_unknown_volatility_is_conditional_with_stated_reason():
    decision = _assess(vol_regime=None, vol_unavailable_reason="plan-gated")
    assert decision.state is TradeabilityState.CONDITIONAL
    assert any("plan-gated" in r for r in decision.reasons)


def test_too_few_bars_is_data_insufficient():
    decision = _assess(bar_count=150)
    assert decision.state is TradeabilityState.DATA_INSUFFICIENT


def test_stale_bars_are_data_insufficient():
    decision = _assess(stale_trading_days=3)
    assert decision.state is TradeabilityState.DATA_INSUFFICIENT
    # One lagging trading day is within default tolerance (holiday honesty).
    assert _assess(stale_trading_days=1).state is TradeabilityState.TRADEABLE


def test_missing_regime_read_is_data_insufficient():
    decision = _assess(market_regime=None)
    assert decision.state is TradeabilityState.DATA_INSUFFICIENT


def test_precedence_insufficient_beats_block_beats_condition():
    """First-match precedence (§9): the verdict names the most fundamental
    problem — you cannot call an environment BLOCKED on data you don't have."""
    decision = _assess(
        bar_count=10, symbol_regime=TRANSITION, vol_regime=IVRegime.HIGH
    )
    assert decision.state is TradeabilityState.DATA_INSUFFICIENT
    # All three problems still individually reported (§26 evidence).
    assert len(decision.reasons) == 3

    decision = _assess(symbol_regime=TRANSITION, vol_regime=IVRegime.HIGH)
    assert decision.state is TradeabilityState.BLOCKED
    assert len(decision.reasons) == 2


def test_rdw_scenario_strong_direction_with_blocked_environment():
    """§10: Bull 77.8 / Edge +66.7 alongside TRANSITION + EXTREME vol is a
    valid state — tradeability BLOCKED with both causes named. The layer
    never sees the direction at all: direction != permission (§9)."""
    decision = _assess(symbol_regime=TRANSITION, vol_regime=IVRegime.EXTREME)
    assert decision.state is TradeabilityState.BLOCKED
    assert len(decision.reasons) == 2
    joined = " ".join(decision.reasons)
    assert "TRANSITION" in joined and "EXTREME" in joined


def test_thresholds_are_parameters():
    params = TradeabilityParams(
        min_bars=50, max_stale_trading_days=5,
        blocked_regimes=(MarketRegime.TRANSITION, MarketRegime.STRONG_BEAR),
        version="tradeability-test",
    )
    decision = _assess(
        bar_count=60, stale_trading_days=4,
        market_regime=MarketRegime.STRONG_BEAR, params=params,
    )
    assert decision.state is TradeabilityState.BLOCKED
    assert decision.version == "tradeability-test"
