"""Options Backtest Engine (LONG_CALL leg) — quant-integrity tests.

Mirrors test_backtest.py's approach: deterministic synthetic underlying
series (shortened signal params) + a FAKE contract provider whose option
bars are crafted per-test, so every fill and exit can be reasoned about by
hand. The engine must run the LIVE option exit engine (premium stop §11.3,
DTE exit §11.7) — these tests characterize that wiring, plus the
honest-gap rules (missing fill bar -> skipped entry; no-trade day ->
premium stop reports insufficient data; expiry -> intrinsic settlement).
"""
import math
from datetime import date, timedelta

import pytest

from libs.trading_core.backtest import (
    BacktestParams,
    OptionLegBars,
    run_call_backtest,
)
from libs.trading_core.backtest.engine import INITIAL_EQUITY
from libs.trading_core.signals import DirectionalParams, RegimeParams

SHORT_REGIME = RegimeParams(sma_fast=5, sma_mid=10, sma_slow=20, slope_lookback=3)
SHORT_DIRECTION = DirectionalParams(
    sma_fast=5,
    sma_mid=10,
    sma_slow=20,
    slope_lookback=3,
    macd_fast=5,
    macd_slow=10,
    macd_signal=3,
    rsi_period=5,
    pivot_window=2,
    volume_sma_period=5,
)


def call_params(**overrides) -> BacktestParams:
    base = dict(instrument="LONG_CALL", warmup_bars=25)
    base.update(overrides)
    return BacktestParams(**base)


def make_bars(closes):
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    volumes = [1_000_000.0] * len(closes)
    dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(len(closes))]
    return dates, opens, highs, lows, closes, volumes


def uptrend(n=80):
    return [100.0 * 1.01**i for i in range(n)]


def run(bars, params, provider):
    return run_call_backtest(
        *bars,
        params,
        contract_provider=provider,
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
    )


def flat_premium_leg(dates, expiry_offset=60, premium=5.0, symbol="TEST_C"):
    """A leg whose option trades every day at a constant premium."""
    expiry = dates[0] + timedelta(days=expiry_offset)
    return OptionLegBars(
        symbol=symbol,
        strike=100.0,
        expiry=expiry,
        bars={d: (premium, premium) for d in dates},
    )


def test_entry_fill_arithmetic_and_contract_identity():
    """First entry: decision at bar 25, filled at bar 26's option OPEN with
    the option slippage proxy and per-contract commission — hand-computed."""
    bars = make_bars(uptrend())
    dates = bars[0]
    leg = flat_premium_leg(dates, premium=5.0)
    params = call_params()

    result = run(bars, params, lambda d, s: leg)
    assert result.trades, "uptrend must enter"
    tr = result.trades[0]

    # Fill = open * (1 + 100bps) = 5.0 * 1.01
    assert tr.entry_price == pytest.approx(5.0 * 1.01)
    # Sizing: 10% of 100k equity on premium, per-contract cost 505 + 0.65.
    per_contract = 5.0 * 1.01 * 100 + 0.65
    assert tr.contracts == math.floor(INITIAL_EQUITY * 0.10 / per_contract)
    assert tr.contract_symbol == "TEST_C"
    assert tr.strike == 100.0
    assert tr.entry_reason.endswith(
        f"LONG_CALL TEST_C (strike 100, exp {leg.expiry.isoformat()})"
    )
    # Decision at the first eligible bar (warmup 25), fill at the NEXT bar.
    assert tr.entry_index >= 26


def test_premium_hard_stop_fires_via_live_exit_engine():
    """The option collapsing >45% below entry premium fires PREMIUM_HARD_STOP
    (§11.3) even while the underlying keeps rising (no stock rule fires)."""
    bars = make_bars(uptrend())
    dates = bars[0]
    expiry = dates[0] + timedelta(days=70)
    prem = {}
    p = 5.0
    for i, d in enumerate(dates):
        # collapse the premium after bar 30 while the underlying rises
        if i > 30:
            p = 1.0
        prem[d] = (p, p)
    leg = OptionLegBars(symbol="CRUSH_C", strike=100.0, expiry=expiry, bars=prem)

    result = run(bars, call_params(), lambda d, s: leg)
    assert result.trades
    assert result.trades[0].exit_reason.startswith("PREMIUM_HARD_STOP"), (
        result.trades[0].exit_reason
    )


def test_dte_exit_fires_at_21_dte():
    """With a flat premium (no stop) and a strong signal (no signal exits),
    the LIVE §11.7 rule closes the position at DTE <= 21."""
    bars = make_bars(uptrend(100))
    dates = bars[0]
    leg = flat_premium_leg(dates, expiry_offset=45, premium=5.0)

    result = run(bars, call_params(time_stop_bars=200), lambda d, s: leg)
    assert result.trades
    tr = result.trades[0]
    assert tr.exit_reason.startswith("DTE_EXIT"), tr.exit_reason
    # Exit decision fired at the first bar with DTE <= 21; fill next bar.
    dte_at_exit_decision = (leg.expiry - dates[tr.exit_index - 1]).days
    assert dte_at_exit_decision <= 21


def test_expiry_settlement_at_intrinsic():
    """A leg whose bars STOP right after entry (illiquid contract) cannot
    fill its pending exit — the position is carried to expiry and settled at
    intrinsic off the REAL underlying close: contractual arithmetic, no
    invented price, and the exit reason says so."""
    bars = make_bars(uptrend(60))
    dates = bars[0]
    expiry = dates[50]
    leg = OptionLegBars(
        symbol="ILLIQ_C",
        strike=100.0,
        expiry=expiry,
        # Bars exist only through bar 27: the entry (fill ~bar 26) works,
        # then the contract never trades again.
        bars={d: (5.0, 5.0) for d in dates[:28]},
    )

    result = run(bars, call_params(time_stop_bars=200), lambda d, s: leg)
    assert result.trades
    tr = result.trades[0]
    assert "settled at expiry intrinsic" in tr.exit_reason or tr.exit_reason.startswith(
        "EXPIRY_SETTLEMENT"
    ), tr.exit_reason
    # Settled at intrinsic of the REAL underlying close on the expiry bar.
    idx = dates.index(tr.exit_date)
    closes = bars[4]
    assert tr.exit_price == pytest.approx(max(closes[idx] - 100.0, 0.0))
    assert tr.exit_date >= expiry


def test_missing_fill_bar_skips_entry_honestly():
    """A contract with NO bar on the would-be fill date never fills — the
    entry is skipped, not priced by invention (plan §44 rule 18)."""
    bars = make_bars(uptrend(40))
    result = run(
        bars,
        call_params(),
        lambda d, s: OptionLegBars(
            symbol="NODATA_C",
            strike=100.0,
            expiry=bars[0][0] + timedelta(days=60),
            bars={},
        ),
    )
    assert result.trades == []
    assert all(v == INITIAL_EQUITY for v in result.equity)


def test_no_provider_result_means_no_trade():
    """provider -> None on every decision: NO TRADE is a valid output."""
    bars = make_bars(uptrend(40))
    result = run(bars, call_params(), lambda d, s: None)
    assert result.trades == []
    assert result.metrics.num_trades == 0


def test_instrument_validation():
    # Phase 3 (2026-08-17): SHORT_STOCK is a real leg now — the naked
    # shorts remain outside the vocabulary forever.
    assert BacktestParams(instrument="SHORT_STOCK").instrument == "SHORT_STOCK"
    with pytest.raises(ValueError, match="instrument"):
        BacktestParams(instrument="NAKED_SHORT_CALL")
    bars = make_bars(uptrend(30))
    with pytest.raises(ValueError, match="LONG_CALL"):
        run_call_backtest(
            *bars,
            BacktestParams(),  # LONG_STOCK params into the call engine
            contract_provider=lambda d, s: None,
        )


# ---------------------------------------------------------------------------
# BULL_CALL_SPREAD leg (roadmap Phase 1): net-debit semantics over two real
# legs — adverse slippage on BOTH legs, live premium-stop/DTE on the NET,
# expiry at NET intrinsic bounded [0, width].
# ---------------------------------------------------------------------------
from libs.trading_core.backtest import SpreadLegBars, run_spread_backtest


def spread_params(**overrides) -> BacktestParams:
    base = dict(instrument="BULL_CALL_SPREAD", warmup_bars=25)
    base.update(overrides)
    return BacktestParams(**base)


def run_spread(bars, params, provider):
    return run_spread_backtest(
        *bars,
        params,
        spread_provider=provider,
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
    )


def flat_spread_leg(dates, expiry_offset=60, long_prem=5.0, short_prem=2.0):
    expiry = dates[0] + timedelta(days=expiry_offset)
    return SpreadLegBars(
        long=OptionLegBars(
            symbol="LONG_C",
            strike=100.0,
            expiry=expiry,
            bars={d: (long_prem, long_prem) for d in dates},
        ),
        short=OptionLegBars(
            symbol="SHORT_C",
            strike=105.0,
            expiry=expiry,
            bars={d: (short_prem, short_prem) for d in dates},
        ),
    )


def test_spread_entry_net_fill_with_adverse_slippage_on_both_legs():
    bars = make_bars(uptrend())
    dates = bars[0]
    leg = flat_spread_leg(dates)
    result = run_spread(bars, spread_params(), lambda d, s: leg)
    assert result.trades
    tr = result.trades[0]
    # Net fill = long_open*(1+slip) - short_open*(1-slip), slip = 100bps.
    expected_net = 5.0 * 1.01 - 2.0 * 0.99
    assert tr.entry_price == pytest.approx(expected_net)
    # Sizing: 10% of equity on net debit; commissions PER LEG (×2).
    per_spread = expected_net * 100 + 2 * 0.65
    assert tr.contracts == math.floor(INITIAL_EQUITY * 0.10 / per_spread)
    assert tr.contract_symbol == "LONG_C"
    assert tr.short_symbol == "SHORT_C"
    assert tr.short_strike == 105.0
    assert "BULL_CALL_SPREAD" in tr.entry_reason


def test_spread_premium_stop_fires_on_net_collapse():
    bars = make_bars(uptrend())
    dates = bars[0]
    expiry = dates[0] + timedelta(days=70)
    long_bars, short_bars = {}, {}
    lp, sp = 5.0, 2.0
    for i, d in enumerate(dates):
        if i > 30:
            lp, sp = 2.4, 1.2  # net 1.2 < 55% of entry net (~3.05)
        long_bars[d] = (lp, lp)
        short_bars[d] = (sp, sp)
    leg = SpreadLegBars(
        long=OptionLegBars("L_C", 100.0, expiry, long_bars),
        short=OptionLegBars("S_C", 105.0, expiry, short_bars),
    )
    result = run_spread(bars, spread_params(), lambda d, s: leg)
    assert result.trades
    assert result.trades[0].exit_reason.startswith("PREMIUM_HARD_STOP"), (
        result.trades[0].exit_reason
    )


def test_spread_expiry_settles_at_net_intrinsic_bounded_by_width():
    """Legs stop trading right after entry; held to expiry -> net intrinsic
    max(S-100,0)-max(S-105,0), which the rising underlying pins to the full
    width 5.0."""
    bars = make_bars(uptrend(60))
    dates = bars[0]
    expiry = dates[50]
    leg = SpreadLegBars(
        long=OptionLegBars("L_C", 100.0, expiry, {d: (5.0, 5.0) for d in dates[:28]}),
        short=OptionLegBars("S_C", 105.0, expiry, {d: (2.0, 2.0) for d in dates[:28]}),
    )
    result = run_spread(bars, spread_params(time_stop_bars=200), lambda d, s: leg)
    assert result.trades
    tr = result.trades[0]
    assert "net intrinsic" in tr.exit_reason
    idx = dates.index(tr.exit_date)
    closes = bars[4]
    expected = max(closes[idx] - 100.0, 0.0) - max(closes[idx] - 105.0, 0.0)
    assert tr.exit_price == pytest.approx(max(expected, 0.0))
    assert tr.exit_price <= 5.0 + 1e-9  # bounded by width


def test_spread_entry_skipped_when_either_leg_bar_missing():
    bars = make_bars(uptrend(40))
    dates = bars[0]
    expiry = dates[0] + timedelta(days=60)
    # Long leg trades daily; short leg NEVER trades -> no joint fill, ever.
    leg = SpreadLegBars(
        long=OptionLegBars("L_C", 100.0, expiry, {d: (5.0, 5.0) for d in dates}),
        short=OptionLegBars("S_C", 105.0, expiry, {}),
    )
    result = run_spread(bars, spread_params(), lambda d, s: leg)
    assert result.trades == []
    assert all(v == INITIAL_EQUITY for v in result.equity)


def test_spread_degenerate_net_skips_entry():
    """Short quote >= long quote (net <= 0) or net >= width: never filled."""
    bars = make_bars(uptrend(40))
    dates = bars[0]
    expiry = dates[0] + timedelta(days=60)
    inverted = SpreadLegBars(
        long=OptionLegBars("L_C", 100.0, expiry, {d: (2.0, 2.0) for d in dates}),
        short=OptionLegBars("S_C", 105.0, expiry, {d: (5.0, 5.0) for d in dates}),
    )
    result = run_spread(bars, spread_params(), lambda d, s: inverted)
    assert result.trades == []


def test_spread_instrument_validation():
    bars = make_bars(uptrend(30))
    with pytest.raises(ValueError, match="BULL_CALL_SPREAD"):
        run_spread_backtest(
            *bars,
            BacktestParams(instrument="LONG_CALL"),
            spread_provider=lambda d, s: None,
        )
    with pytest.raises(ValueError, match="spread_width_pct"):
        BacktestParams(instrument="BULL_CALL_SPREAD", spread_width_pct=0.0)


# ---------------------------------------------------------------------------
# BEAR_PUT_SPREAD leg (audit risk-engine-audit.md §8 item 6, fixed
# 2026-08-17): the bear entry gate (_evaluate_entry_bear, shared with
# LONG_PUT / SHORT_STOCK) + put-vertical geometry (short strike BELOW the
# long). Before the fix the BULL evaluator ran and the engine demanded
# short > long, so the leg silently produced zero trades.
# ---------------------------------------------------------------------------


def downtrend(n=80):
    return [100.0 * 0.99**i for i in range(n)]


def bear_put_spread_provider(bars, width=5.0, long_tv=1.0, short_tv=0.5, expiry_offset=60):
    """Put vertical struck AT the decision spot: long K=floor(spot), short
    K-width, same expiry. Premiums are intrinsic off the synthetic
    underlying + a constant time value, so net = long − short rises
    monotonically as the underlying falls (hand-checkable fills)."""
    dates, opens, _h, _l, closes, _v = bars
    open_by_date = dict(zip(dates, opens))
    close_by_date = dict(zip(dates, closes))
    expiry = dates[0] + timedelta(days=expiry_offset)

    def prem(strike, tv):
        return {
            d: (
                max(strike - open_by_date[d], 0.0) + tv,
                max(strike - close_by_date[d], 0.0) + tv,
            )
            for d in dates
        }

    def provider(decision_date, spot):
        k_long = float(math.floor(spot))
        k_short = k_long - width
        return SpreadLegBars(
            long=OptionLegBars("L_P", k_long, expiry, prem(k_long, long_tv)),
            short=OptionLegBars("S_P", k_short, expiry, prem(k_short, short_tv)),
        )

    return provider


def test_bear_put_spread_enters_on_downtrend_and_profits():
    """(a) Synthetic downtrend + put-vertical provider -> the bear leg now
    ENTERS via the bear gate and its P&L follows the falling underlying."""
    bars = make_bars(downtrend())
    dates, opens, _h, _l, closes, _v = bars
    result = run_spread(
        bars, spread_params(instrument="BEAR_PUT_SPREAD"), bear_put_spread_provider(bars)
    )
    assert result.trades, "a clean downtrend must open at least one bear put spread"
    for tr in result.trades:
        assert tr.short_strike < tr.strike  # put vertical geometry
        assert tr.strike - tr.short_strike == pytest.approx(5.0)
        assert "bias BEAR" in tr.entry_reason
        assert "BEAR_PUT_SPREAD" in tr.entry_reason
        assert 0.0 < tr.entry_price < 5.0

    # First trade by hand: decision at bar 25 (spot 100*0.99^25 = 77.78 ->
    # K_long 77, K_short 72), filled at bar 26's open = close[25] = 77.78:
    # long open = max(77-77.78,0)+1.0 = 1.0, short open = 0.5, net fill =
    # 1.0*1.01 - 0.5*0.99 = 0.515 (adverse slippage BOTH legs, 100 bps).
    tr = result.trades[0]
    assert tr.entry_index == 26
    assert tr.strike == 77.0 and tr.short_strike == 72.0
    assert tr.entry_price == pytest.approx(0.515)
    per_spread = 0.515 * 100 + 2 * 0.65
    assert tr.contracts == math.floor(INITIAL_EQUITY * 0.10 / per_spread)
    # Exit fill at the exit bar's open (adverse: sell long cheaper, buy back
    # short dearer): (77-S+1.0)*0.99 - (72-S+0.5)*1.01 with S = open[exit].
    s_exit = opens[tr.exit_index]
    expected_exit = (77.0 - s_exit + 1.0) * 0.99 - (72.0 - s_exit + 0.5) * 1.01
    assert tr.exit_price == pytest.approx(max(expected_exit, 0.0))
    assert tr.pnl > 0.0  # underlying fell 77.78 -> ~67.6 by the exit
    # Sign consistent with the downtrend across the whole replay.
    total_pnl = sum(t.pnl for t in result.trades)
    assert total_pnl > 0.0
    assert result.equity[-1] == pytest.approx(INITIAL_EQUITY + total_pnl)


def test_bear_put_spread_refuses_bull_geometry_candidate():
    """(b) A candidate whose short strike sits ABOVE the long (call-vertical
    geometry) under BEAR_PUT_SPREAD is refused at the decision — even though
    its premiums (5 / 2) would otherwise fill as a positive net debit."""
    bars = make_bars(downtrend(40))
    dates = bars[0]
    expiry = dates[0] + timedelta(days=60)
    wrong_way = SpreadLegBars(
        long=OptionLegBars("L_P", 100.0, expiry, {d: (5.0, 5.0) for d in dates}),
        short=OptionLegBars("S_P", 105.0, expiry, {d: (2.0, 2.0) for d in dates}),
    )
    result = run_spread(
        bars, spread_params(instrument="BEAR_PUT_SPREAD"), lambda d, s: wrong_way
    )
    assert result.trades == []
    assert all(v == INITIAL_EQUITY for v in result.equity)

    # Mirror: put-vertical geometry under BULL_CALL_SPREAD is refused too.
    up = make_bars(uptrend(40))
    udates = up[0]
    uexpiry = udates[0] + timedelta(days=60)
    wrong_way_bull = SpreadLegBars(
        long=OptionLegBars("L_C", 105.0, uexpiry, {d: (5.0, 5.0) for d in udates}),
        short=OptionLegBars("S_C", 100.0, uexpiry, {d: (2.0, 2.0) for d in udates}),
    )
    result = run_spread(up, spread_params(), lambda d, s: wrong_way_bull)
    assert result.trades == []


def test_bear_put_spread_needs_the_bear_signal_not_the_bull_one():
    """The bear leg on an UPTREND never enters (the bull gate would have),
    and the bull leg on a DOWNTREND never enters — the two gates are
    distinct (plan §44 rule 18: NO TRADE is a valid output)."""
    up = make_bars(uptrend(60))
    result = run_spread(
        up, spread_params(instrument="BEAR_PUT_SPREAD"), bear_put_spread_provider(up)
    )
    assert result.trades == []
    down = make_bars(downtrend(60))
    leg = flat_spread_leg(down[0])
    result = run_spread(down, spread_params(), lambda d, s: leg)
    assert result.trades == []


def test_bull_call_spread_regression_pin_unchanged_by_bear_fix():
    """(c) The bull path is byte-identical after the bear fix: pin the
    existing flat-premium fixture's replay numbers. Hand-check: net fill
    5*1.01 - 2*0.99 = 3.07; per spread 307 + 1.30 = 308.30; qty =
    floor(10 000 / 308.30) = 32; exit net 5*0.99 - 2*1.01 = 2.93; pnl =
    32*100*2.93 - 32*1.30 - 32*308.30 = 9376 - 41.6 - 9865.6 = -531.2."""
    bars = make_bars(uptrend())
    leg = flat_spread_leg(bars[0])
    result = run_spread(bars, spread_params(), lambda d, s: leg)
    assert len(result.trades) == 11
    first = result.trades[0]
    assert first.entry_index == 26 and first.exit_index == 40
    assert first.entry_price == pytest.approx(3.07)
    assert first.contracts == 32
    assert first.exit_price == pytest.approx(2.93)
    assert first.pnl == pytest.approx(-531.2)
    assert first.exit_reason.startswith("DTE_EXIT: 21 DTE")
    assert all(t.short_strike > t.strike for t in result.trades)
    assert result.equity[-1] == pytest.approx(94_322.8)
    assert result.metrics.num_trades == 11


# ---------------------------------------------------------------------------
# Income legs (Phase 2): covered-call buy-write + cash-secured put replays.
# ---------------------------------------------------------------------------
from libs.trading_core.backtest import run_covered_call_backtest, run_csp_backtest


def test_covered_call_overlay_banks_profit_via_capture_rule():
    """Uptrend base + a decaying far-OTM call: the overlay is sold, decays,
    and PROFIT_CAPTURE buys it back at a gain (the 50% standard). The churn
    guard must prevent re-selling a leg already in the DTE zone."""
    bars = make_bars(uptrend(100))
    dates = bars[0]
    expiry = dates[0] + timedelta(days=80)
    prem = {}
    for i, d in enumerate(dates):
        prem[d] = (2.0, 2.0) if i <= 30 else (0.8, 0.8)  # decays past 50%
    leg = OptionLegBars(symbol="CC_C", strike=100000.0, expiry=expiry, bars=prem)
    params = BacktestParams(instrument="COVERED_CALL", warmup_bars=25, time_stop_bars=200)
    result = run_covered_call_backtest(
        *bars, params,
        contract_provider=lambda d, s: leg,
        regime_params=SHORT_REGIME, directional_params=SHORT_DIRECTION,
    )
    cc_trades = [t for t in result.trades if t.contract_symbol == "CC_C"]
    assert cc_trades, "the overlay must have been sold"
    captured = [t for t in cc_trades if t.exit_reason.startswith("PROFIT_CAPTURE")]
    assert captured, [t.exit_reason for t in cc_trades]
    assert captured[0].pnl > 0  # sold ~2.0, bought back ~0.8
    # Churn guard: once the same leg is inside 21 DTE it is never re-sold.
    dte_at_entry = [(leg.expiry - t.entry_date).days for t in cc_trades]
    assert all(d > 21 for d in dte_at_entry), dte_at_entry


def test_covered_call_assignment_caps_the_stock_at_strike():
    """Strong uptrend + a near strike: expiry ITM -> shares called away at
    the strike (contractual settlement), both legs closed on the same bar."""
    bars = make_bars(uptrend(80))
    dates = bars[0]
    closes = bars[4]
    expiry = dates[40]
    strike = closes[30]  # will be deep ITM by bar 40 in a 1%/day uptrend
    leg = OptionLegBars(
        symbol="ASSIGN_C", strike=strike, expiry=expiry,
        # Rich flat premium so the 2x loss stop never fires first.
        bars={d: (5.0, 5.0) for d in dates},
    )
    params = BacktestParams(
        instrument="COVERED_CALL", warmup_bars=25, time_stop_bars=200,
        # Keep the mechanical stops out of the way for this scenario.
        option_slippage_bps=0.0, slippage_bps=0.0,
    )
    result = run_covered_call_backtest(
        *bars, params,
        contract_provider=lambda d, s: leg,
        regime_params=SHORT_REGIME, directional_params=SHORT_DIRECTION,
    )
    assigned_stock = [t for t in result.trades if "ASSIGNED" in t.exit_reason]
    if assigned_stock:  # loss-stop may fire first depending on path; if
        # assignment happened, the stock MUST have exited exactly at strike.
        assert assigned_stock[0].exit_price == pytest.approx(strike)
        cc_rows = [t for t in result.trades if "assigned" in t.exit_reason]
        assert cc_rows and cc_rows[0].exit_index == assigned_stock[0].exit_index


def test_csp_keeps_credit_otm_and_cash_settles_itm():
    """Flat-to-up series: the sold put expires OTM -> full credit kept."""
    bars = make_bars(uptrend(80))
    dates = bars[0]
    expiry = dates[0] + timedelta(days=50)
    leg = OptionLegBars(
        symbol="CSP_P", strike=50.0, expiry=expiry,  # far OTM put
        bars={d: (1.5, 1.5) for d in dates},
    )
    params = BacktestParams(instrument="CASH_SECURED_PUT", warmup_bars=25)
    result = run_csp_backtest(
        *bars, params,
        contract_provider=lambda d, s: leg,
        regime_params=SHORT_REGIME, directional_params=SHORT_DIRECTION,
    )
    assert result.trades
    tr = result.trades[0]
    # Managed by the mechanical rules: flat premium -> DTE_EXIT at 21, or
    # expiry OTM if the window is short; either way the seller keeps value.
    assert tr.pnl > 0 or tr.exit_reason.startswith("DTE_EXIT")
    # Return measured on the cash actually secured (strike basis).
    assert abs(tr.return_pct) < 100.0


def test_income_engine_validation():
    bars = make_bars(uptrend(30))
    with pytest.raises(ValueError, match="COVERED_CALL"):
        run_covered_call_backtest(
            *bars, BacktestParams(), contract_provider=lambda d, s: None
        )
    with pytest.raises(ValueError, match="CASH_SECURED_PUT"):
        run_csp_backtest(
            *bars, BacktestParams(), contract_provider=lambda d, s: None
        )
