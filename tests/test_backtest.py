"""Quant-integrity tests for Backtest Engine V1 (development plan §20).

Every series is a deterministic synthetic builder so each expected trade can
be reasoned about by hand. The suite pins the plan §20.3 bias controls:

- NO LOOK-AHEAD: a trade fully closed before bar N is identical whether the
  engine saw N bars or N+100 bars.
- Explicit next-open fill model with slippage + commission (plan §44 rule 11),
  verified against a hand-computed trade.
- NO TRADE is a valid output (plan §44 rule 18) and produces honest nulls.
- Long-only invariant (plan §5), determinism, and no NaN/Infinity anywhere.

Signal thresholds are shortened via RegimeParams / DirectionalParams — every
threshold is a parameter (plan §6.2) — so purpose-built series stay small.
"""
import dataclasses
import math
from datetime import date, timedelta

import pytest

from libs.trading_core.backtest import BacktestParams, run_backtest
from libs.trading_core.backtest.engine import INITIAL_EQUITY
from libs.trading_core.signals import DirectionalParams, RegimeParams

# ---------------------------------------------------------------------------
# Shortened signal parameters (plan §6.2: thresholds are parameters) so the
# purpose-built series need ~25 warmup bars instead of 200.
# ---------------------------------------------------------------------------

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
SHORT_PARAMS = BacktestParams(warmup_bars=25)


def make_bars(closes: list[float], spread: float = 0.5):
    """Deterministic bars from a closes series: each bar opens at the prior
    close, with a fixed high/low band around the open->close range."""
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + spread for o, c in zip(opens, closes)]
    lows = [min(o, c) - spread for o, c in zip(opens, closes)]
    volumes = [1_000_000.0] * len(closes)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(closes))]
    return dates, opens, highs, lows, closes, volumes


def run_short(bars, params: BacktestParams = SHORT_PARAMS):
    return run_backtest(
        *bars,
        params,
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
    )


def uptrend_closes(n: int = 80) -> list[float]:
    return [100.0 * 1.01**i for i in range(n)]


def wavy_closes(n: int) -> list[float]:
    """Rising trend modulated by a sine wave: several entries and exits."""
    return [100.0 * (1.003**i) * (1.0 + 0.05 * math.sin(i / 8.0)) for i in range(n)]


def walk_assert_finite(obj) -> None:
    """Recursively assert every float in a result is finite (None is allowed,
    NaN/Infinity never — plan §44 rule 18)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            walk_assert_finite(getattr(obj, f.name))
    elif isinstance(obj, dict):
        for value in obj.values():
            walk_assert_finite(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            walk_assert_finite(value)
    elif isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float in result: {obj!r}"


# ---------------------------------------------------------------------------
# NO-LOOK-AHEAD PROPERTY (plan §20.3) — the most important test in the suite.
# ---------------------------------------------------------------------------


def test_no_look_ahead_closed_trades_identical_on_truncated_series():
    """A trade fully closed before bar 300 must be bit-identical whether the
    engine was given 300 bars or 400 bars (plan §20.3: decisions at bar t see
    only data <= t, so future bars cannot change past trades)."""
    bars_400 = make_bars(wavy_closes(400))
    bars_300 = tuple(series[:300] for series in bars_400)

    full = run_short(bars_400)
    head = run_short(bars_300)

    closed_early = [
        t for t in head.trades if not t.exit_reason.startswith("END_OF_DATA")
    ]
    assert len(closed_early) >= 3  # the series produces several round trips
    # Dataclass equality: every field (indices, dates, prices, shares, pnl,
    # reasons) must match exactly — not approximately.
    assert full.trades[: len(closed_early)] == closed_early


def test_no_look_ahead_equity_prefix_identical():
    """The daily equity marks over the shared prefix are identical too: bar
    t's mark uses only fills decided on bars < t and bar t's close."""
    bars_400 = make_bars(wavy_closes(400))
    bars_300 = tuple(series[:300] for series in bars_400)

    full = run_short(bars_400)
    head = run_short(bars_300)

    assert head.equity == full.equity[:300]
    assert head.dates == full.dates[:300]


# ---------------------------------------------------------------------------
# Entries and NO TRADE (plan §44 rule 18)
# ---------------------------------------------------------------------------


def test_uptrend_produces_trade_and_positive_return():
    result = run_short(make_bars(uptrend_closes()))
    assert len(result.trades) >= 1
    full = result.metrics["full"]
    assert full.num_trades >= 1
    assert full.total_return_pct > 0.0
    assert full.exposure_pct > 0.0
    # Warmup is respected: the earliest possible fill is warmup_bars + 1.
    assert result.trades[0].entry_index >= SHORT_PARAMS.warmup_bars + 1


def test_downtrend_and_flat_series_produce_no_trades_with_null_metrics():
    """NO TRADE is a valid output (plan §44 rule 18): bearish and flat series
    never qualify for a long entry, and undefined metrics are None."""
    downtrend = [100.0 * 0.99**i for i in range(80)]
    flat = [100.0] * 80
    for closes in (downtrend, flat):
        result = run_short(make_bars(closes))
        assert result.trades == []
        for segment in ("full", "in_sample", "out_of_sample"):
            m = result.metrics[segment]
            assert m.num_trades == 0
            assert m.win_rate is None
            assert m.profit_factor is None
            assert m.expectancy_pct is None
            assert m.avg_trade_pct is None
            assert m.avg_hold_bars is None
            assert m.exposure_pct == 0.0
            assert m.total_return_pct == 0.0
        # Flat equity: zero-variance daily returns => Sharpe/Sortino are None,
        # never a division blow-up.
        assert result.metrics["full"].sharpe is None
        assert result.metrics["full"].sortino is None
        assert all(value == INITIAL_EQUITY for value in result.equity)


# ---------------------------------------------------------------------------
# Fill model (plan §44 rule 11) — hand-computed trade
# ---------------------------------------------------------------------------


def fill_model_bars():
    """Rise 2%/bar through bar 34, then crash 10%/bar: forces exactly one
    entry (decision at bar 25, fill at open of 26) and one signal exit."""
    closes = [100.0 * 1.02**i for i in range(35)]
    while len(closes) < 45:
        closes.append(closes[-1] * 0.90)
    return make_bars(closes)


def test_fill_model_matches_hand_computed_trade():
    dates, opens, highs, lows, closes, volumes = fill_model_bars()
    params = SHORT_PARAMS
    result = run_short((dates, opens, highs, lows, closes, volumes), params)

    assert len(result.trades) == 1
    trade = result.trades[0]

    # Entry: decision at the close of bar 25 (first bar >= warmup_bars),
    # filled at the NEXT bar's open plus slippage (plan §44 rule 11).
    assert trade.entry_index == params.warmup_bars + 1 == 26
    slip = params.slippage_bps / 10_000.0
    expected_entry = opens[26] * (1.0 + slip)
    assert trade.entry_price == pytest.approx(expected_entry, rel=1e-12)

    # Shares: floor(equity * position_pct / fill price), trimmed only if the
    # commission would push cash negative.
    qty = math.floor(INITIAL_EQUITY * params.position_pct / expected_entry)
    while qty > 0 and qty * (expected_entry + params.commission_per_share) > INITIAL_EQUITY:
        qty -= 1
    assert trade.shares == qty

    # Exit: sell fill at the exit bar's open minus slippage.
    assert trade.exit_reason.startswith("SIGNAL_DECAY")
    expected_exit = opens[trade.exit_index] * (1.0 - slip)
    assert trade.exit_price == pytest.approx(expected_exit, rel=1e-12)

    # PnL: commission charged per share BOTH ways.
    cost = qty * (expected_entry + params.commission_per_share)
    proceeds = qty * (expected_exit - params.commission_per_share)
    expected_pnl = proceeds - cost
    assert trade.pnl == pytest.approx(expected_pnl, rel=1e-12)
    assert trade.return_pct == pytest.approx(expected_pnl / cost * 100.0, rel=1e-12)
    assert trade.bars_held == trade.exit_index - trade.entry_index

    # After the exit the account is flat cash for the rest of the series.
    assert result.equity[-1] == pytest.approx(INITIAL_EQUITY + expected_pnl, rel=1e-12)


# ---------------------------------------------------------------------------
# Costs monotonicity (plan §44 rule 11: costs are real and explicit)
# ---------------------------------------------------------------------------


def test_raising_slippage_never_increases_total_return():
    bars = make_bars(uptrend_closes())
    returns = [
        run_short(bars, BacktestParams(warmup_bars=25, slippage_bps=bps))
        .metrics["full"]
        .total_return_pct
        for bps in (0.0, 5.0, 50.0, 200.0)
    ]
    assert all(a >= b for a, b in zip(returns, returns[1:]))


def test_raising_commission_never_increases_total_return():
    bars = make_bars(uptrend_closes())
    returns = [
        run_short(bars, BacktestParams(warmup_bars=25, commission_per_share=c))
        .metrics["full"]
        .total_return_pct
        for c in (0.0, 0.005, 0.05, 0.5)
    ]
    assert all(a >= b for a, b in zip(returns, returns[1:]))


# ---------------------------------------------------------------------------
# Exit rules (plan §11.5 ATR trail, §11.6 time stop)
# ---------------------------------------------------------------------------


def test_atr_trail_exit_triggers_on_drop_below_trail():
    """Steep +2/bar trend, then a single -2 close: the drop breaches
    peak - 0.5 * atr14 while every directional component stays bullish, so
    ATR_TRAIL (priority 3) fires — not SIGNAL_FLIP/DECAY (plan §11.5)."""
    closes = [100.0 + 2.0 * i for i in range(41)]
    closes[40] = closes[39] - 2.0
    while len(closes) < 48:
        closes.append(closes[-1] - 1.0)
    params = BacktestParams(warmup_bars=25, atr_trail_k=0.5, exit_edge_threshold=0.0)
    result = run_short(make_bars(closes, spread=0.3), params)

    assert result.trades, "the uptrend must produce an entry"
    trade = result.trades[0]
    assert trade.exit_reason.startswith("ATR_TRAIL")
    # Explainability (plan §38): the reason carries the real numbers used.
    assert "peak" in trade.exit_reason and "atr14" in trade.exit_reason
    # The decision was at the -2 close of bar 40; the fill is the next open.
    assert trade.exit_index == 41


def test_time_stop_exit_triggers_when_position_goes_nowhere():
    """Rise 1.5%/bar into the entry, then perfectly flat closes: no trail
    breach, edge stays positive, but after time_stop_bars without a
    min_move_atr * atr14 move the TIME_STOP abandons the trade (plan §11.6)."""
    closes = [100.0 * 1.015**i for i in range(26)]
    while len(closes) < 45:
        closes.append(closes[25])
    params = BacktestParams(warmup_bars=25, time_stop_bars=10, exit_edge_threshold=0.0)
    result = run_short(make_bars(closes), params)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason.startswith("TIME_STOP")
    assert f"held {params.time_stop_bars} bars >= {params.time_stop_bars}" in trade.exit_reason
    # Decision at bars_held == time_stop_bars, filled one bar later.
    assert trade.bars_held == params.time_stop_bars + 1


# ---------------------------------------------------------------------------
# Structural contracts: determinism, alignment, finiteness, long-only
# ---------------------------------------------------------------------------


def test_determinism_identical_runs_identical_results():
    bars = make_bars(wavy_closes(300))
    assert run_short(bars) == run_short(bars)


def test_arrays_aligned_and_result_free_of_nan_and_infinity():
    for closes in (wavy_closes(400), uptrend_closes(), [100.0] * 80):
        bars = make_bars(closes)
        result = run_short(bars)
        n = len(closes)
        assert len(result.dates) == len(result.equity) == len(result.drawdown) == n
        assert result.dates == bars[0]
        # drawdown = equity / running_max - 1 is never positive.
        assert all(dd <= 0.0 for dd in result.drawdown)
        walk_assert_finite(result)


def test_long_only_invariant_no_shorting_no_negative_cash():
    """Plan §5: long stock only. Shares are always positive and equity (cash
    + position marked to market) never goes negative — cash is never used to
    short and the account never borrows."""
    for closes in (wavy_closes(400), uptrend_closes(), [100.0 * 0.99**i for i in range(80)]):
        result = run_short(make_bars(closes))
        for trade in result.trades:
            assert trade.shares >= 1
            assert trade.exit_index > trade.entry_index or trade.exit_reason.startswith(
                "END_OF_DATA"
            )
        assert min(result.equity) > 0.0
        assert 0.0 <= result.metrics["full"].exposure_pct <= 100.0


def test_in_sample_out_of_sample_segmentation():
    """OOS boundary = floor(n * oos_split); trades belong to the segment of
    their ENTRY bar; the engine only reports the OOS segment, it never
    optimizes on it (plan §44 rule 16)."""
    bars = make_bars(wavy_closes(400))
    result = run_short(bars)
    boundary = math.floor(400 * SHORT_PARAMS.oos_split)
    assert result.oos_start_date == bars[0][boundary]
    assert set(result.metrics) == {"full", "in_sample", "out_of_sample"}
    m = result.metrics
    assert m["in_sample"].num_trades + m["out_of_sample"].num_trades == m["full"].num_trades
    assert m["in_sample"].num_trades == sum(
        1 for t in result.trades if t.entry_index < boundary
    )
    assert m["out_of_sample"].num_trades == sum(
        1 for t in result.trades if t.entry_index >= boundary
    )


# ---------------------------------------------------------------------------
# Parameter validation (plan §44 rule 2: parameters, checked at the door)
# ---------------------------------------------------------------------------


def test_default_params_match_the_v1_schema():
    p = BacktestParams()
    assert (
        p.position_pct,
        p.commission_per_share,
        p.slippage_bps,
        p.entry_edge_threshold,
        p.exit_edge_threshold,
        p.atr_trail_k,
        p.time_stop_bars,
        p.min_move_atr,
        p.oos_split,
        p.warmup_bars,
    ) == (1.0, 0.005, 5.0, 25.0, 10.0, 3.0, 20, 1.0, 0.7, 200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_pct": 0.0},
        {"position_pct": 1.5},
        {"commission_per_share": -0.01},
        {"slippage_bps": -1.0},
        {"entry_edge_threshold": 10.0, "exit_edge_threshold": 25.0},
        {"atr_trail_k": 0.0},
        {"time_stop_bars": 0},
        {"min_move_atr": -0.5},
        {"oos_split": 0.0},
        {"oos_split": 1.0},
        {"warmup_bars": 0},
    ],
)
def test_invalid_params_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        BacktestParams(**kwargs)


def test_misaligned_series_are_rejected():
    dates, opens, highs, lows, closes, volumes = make_bars(uptrend_closes(40))
    with pytest.raises(ValueError):
        run_backtest(dates, opens, highs, lows, closes[:-1], volumes)
    with pytest.raises(ValueError):
        run_backtest([], [], [], [], [], [])


# ---------------------------------------------------------------------------
# Audit explainability (plan §38): every trade must say WHY it was entered.
# Regression: the fill block once dropped pending_entry, leaving "".
# ---------------------------------------------------------------------------


def test_every_trade_has_a_populated_entry_reason():
    result = run_short(make_bars(wavy_closes(400)))
    assert result.trades  # the series produces trades
    for trade in result.trades:
        assert trade.entry_reason.strip(), "entry_reason must never be empty"
        # Reasons carry real numbers (edge value) and the regime context.
        assert "edge" in trade.entry_reason.lower()
        assert any(ch.isdigit() for ch in trade.entry_reason)
