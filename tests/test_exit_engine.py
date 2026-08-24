"""Tests for Exit Engine v0 (development plan §11).

Every series is a purpose-built deterministic builder so each expected
trigger can be hand-computed in the comments. The suite pins:

- each of the five §11 rules fired by a series built for exactly that rule;
- the priority order (HARD_STOP > SIGNAL_FLIP > SIGNAL_DECAY > ATR_TRAIL >
  TIME_STOP, mirroring the backtest engine);
- short-history behaviour: signal/ATR rules report "insufficient data" and
  stay quiet while HARD_STOP keeps working off raw prices (a data gap must
  never disable the hard stop, plan §11.3);
- explainability (plan §37/§38): EVERY rule reports a reason with real
  numbers, non-triggered rules prefixed "OK:";
- §21 reuse: current_edge is exactly what the live score_direction returns;
- determinism and input validation.
"""
import math

import pytest

from libs.trading_core.exits import (
    ExitDecision,
    ExitParams,
    OptionState,
    PositionState,
    evaluate_exit,
    evaluate_option_exit,
)
from libs.trading_core.signals import DirectionalParams, score_direction

# The §11 rules in engine priority order.
RULES = ("HARD_STOP", "SIGNAL_FLIP", "SIGNAL_DECAY", "ATR_TRAIL", "TIME_STOP")

# The option-position rules in priority order (plan §11.3 / §11.7): the
# premium hard stop REPLACES the underlying HARD_STOP, DTE_EXIT comes next,
# then the same underlying-driven families as stock.
OPTION_RULES = (
    "PREMIUM_HARD_STOP",
    "DTE_EXIT",
    "SIGNAL_FLIP",
    "SIGNAL_DECAY",
    "ATR_TRAIL",
    "TIME_STOP",
)

# Shortened signal parameters (plan §6.2: thresholds are parameters) so the
# purpose-built series need ~25 bars instead of 200 (same set as the
# backtest tests).
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


def make_bars(closes: list[float], spread: float = 0.5):
    """Deterministic OHLCV bars from a closes series: each bar opens at the
    prior close, with a fixed high/low band around the open->close range."""
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + spread for o, c in zip(opens, closes)]
    lows = [min(o, c) - spread for o, c in zip(opens, closes)]
    volumes = [1_000_000.0] * len(closes)
    return closes, highs, lows, volumes


def rule_names(decision: ExitDecision) -> list[str]:
    """Extract the rule name that heads each reason line, OK-prefixed or not."""
    return [r.removeprefix("OK: ").split(":")[0] for r in decision.reasons]


# ---------------------------------------------------------------------------
# Parameters (plan §11 defaults; §44 rule 2: parameters, checked at the door)
# ---------------------------------------------------------------------------


def test_default_params_match_plan_11():
    p = ExitParams()
    assert (
        p.exit_edge_threshold,
        p.atr_trail_k,
        p.time_stop_bars,
        p.min_move_atr,
        p.atr_period,
    ) == (10.0, 3.0, 20, 1.0, 14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atr_trail_k": 0.0},
        {"atr_trail_k": -1.0},
        {"time_stop_bars": 0},
        {"min_move_atr": -0.5},
        {"atr_period": 0},
    ],
)
def test_invalid_params_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ExitParams(**kwargs)


# ---------------------------------------------------------------------------
# Holding: every rule explained with "OK:" (plan §37 — the user must see why
# the position is still held)
# ---------------------------------------------------------------------------


def test_holding_uptrend_reports_ok_for_all_five_rules():
    """+1%/bar uptrend, stop far below, young position: nothing triggers and
    all five rules report an OK line with real numbers, in priority order."""
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    state = PositionState(
        entry_price=closes[20],  # ~122.02
        stop_distance=10.0,
        entry_edge=40.0,
        bars_held=9,
        highest_close_since_entry=closes[-1],
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )

    assert d.should_exit is False
    assert d.triggered_rule is None
    assert len(d.reasons) == 5
    assert rule_names(d) == list(RULES)
    assert all(r.startswith("OK: ") for r in d.reasons)
    assert all(any(ch.isdigit() for ch in r) for r in d.reasons)
    # Hand-computed edge with SHORT_DIRECTION on a clean uptrend under the §6
    # grouped weights (v1): bull side triggers close>sma5/10/20 (SMA group,
    # 20), slope>0 (10), macd>signal + macd>0 (MACD group, 10) = 40 of the 75
    # total (RSI=100 sits OUTSIDE the [50,70] bull zone — 5 untriggered; a
    # monotonic series confirms no pivots — 20; flat volume never expands —
    # 10), bear side 0: edge = 40/75*100 - 0 = 53.333.
    assert d.current_edge == pytest.approx(160.0 / 3.0)
    assert d.stop_price == pytest.approx(closes[20] - 10.0)
    assert d.trail_price is not None and d.trail_price < closes[-1]
    assert d.time_stop_remaining == 20 - 9 == 11


# ---------------------------------------------------------------------------
# Rule 1: HARD_STOP (plan §11.3 adapted to stock)
# ---------------------------------------------------------------------------


def test_hard_stop_triggers_with_hand_computed_numbers():
    """entry 100, stop_distance 8 -> stop 92.00; last close 91.50 <= 92.00
    breaches the hard stop (plan §11.3)."""
    closes, highs, lows, volumes = make_bars([100.0] * 28 + [95.0, 91.5])
    state = PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=5,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )

    assert d.should_exit is True
    assert d.triggered_rule == "HARD_STOP"
    assert d.stop_price == pytest.approx(92.0)
    assert d.reasons[0].startswith("HARD_STOP: stop 92.0000 breached: close 91.5000")
    assert "stop_distance 8.0000" in d.reasons[0]


# ---------------------------------------------------------------------------
# Rule 2: SIGNAL_FLIP (plan §11.1)
# ---------------------------------------------------------------------------


def test_signal_flip_triggers_on_bear_bias():
    """-1%/bar downtrend flips the live signal to BEAR while the (huge) stop
    holds. Hand-computed edge: the bear side triggers 5 of 9 components
    (close<sma5/10/20, slope<0, macd<0; RSI=0 is OUTSIDE the [30,50] bear
    zone, no confirmed pivots, flat volume). The MACD line of a decaying
    exponential is -k*0.99^t — negative but RISING toward zero — so it sits
    ABOVE its lagging signal line and the bull macd_cross triggers (1 of 9):
    edge = 1/9*100 - 5/9*100 = -44.444 <= -25 -> bias BEAR."""
    closes, highs, lows, volumes = make_bars([100.0 * 0.99**i for i in range(30)])
    state = PositionState(
        entry_price=90.0,
        stop_distance=50.0,  # stop 40.0 — never breached (close ~74.66)
        entry_edge=35.0,
        bars_held=5,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )

    assert d.triggered_rule == "SIGNAL_FLIP"
    assert d.should_exit is True
    # Grouped weights (v1): the flipped series triggers the bear SMA group
    # (20) + negative slope (10) = 30 of 75, bull 0 -> edge = -30/75*100.
    assert d.current_edge == pytest.approx(-40.0)
    assert d.reasons[0].startswith("OK: HARD_STOP")
    assert "bias BEAR" in d.reasons[1]
    # Collect-all behaviour: decay and trail conditions are ALSO breached
    # (edge -66.7 < 10; close 74.66 < peak 100 - 3*ATR), reported un-prefixed,
    # but the flip outranks them.
    assert d.reasons[2].startswith("SIGNAL_DECAY:")
    assert d.reasons[3].startswith("ATR_TRAIL:")


# ---------------------------------------------------------------------------
# Rule 3: SIGNAL_DECAY (plan §11.1 — exit easier than entry)
# ---------------------------------------------------------------------------


def test_signal_decay_triggers_on_flat_edge():
    """Perfectly flat series: RSI=50 sits inside BOTH inclusive zones, every
    other component ties -> bull = bear = 1/9*100, edge exactly 0.0 < 10.0
    exit threshold -> SIGNAL_DECAY (bias NEUTRAL, so no flip). ATR is
    exactly 1.0 (every TR = high-low = 1.0), so trail = 100 - 3*1 = 97 holds."""
    closes, highs, lows, volumes = make_bars([100.0] * 30)
    state = PositionState(
        entry_price=100.0,
        stop_distance=5.0,  # stop 95.0 — not breached at close 100.0
        entry_edge=30.0,
        bars_held=10,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )

    assert d.triggered_rule == "SIGNAL_DECAY"
    assert d.current_edge == pytest.approx(0.0)
    assert "edge 0.0 < exit threshold 10.0" in d.reasons[2]
    assert d.reasons[0].startswith("OK: HARD_STOP")
    assert d.reasons[1].startswith("OK: SIGNAL_FLIP")
    assert d.trail_price == pytest.approx(97.0)


# ---------------------------------------------------------------------------
# Rule 4: ATR_TRAIL (plan §11.5)
# ---------------------------------------------------------------------------


def test_atr_trail_triggers_on_drop_below_trail():
    """+2/bar staircase then one -2 close. With spread 0.3 every true range
    is exactly 2.6 (high-low = 2 + 2*0.3), so ATR14 = 2.6. Peak close is
    178.0 (bar 39); with atr_trail_k=0.5 the trail is 178.0 - 0.5*2.6 =
    176.70, and the last close 176.0 < 176.70 -> ATR_TRAIL. The signal
    stays BULL (one down bar can't flip a staircase), so no flip/decay."""
    closes = [100.0 + 2.0 * i for i in range(41)]
    closes[40] = closes[39] - 2.0  # 176.0
    closes, highs, lows, volumes = make_bars(closes, spread=0.3)
    state = PositionState(
        entry_price=120.0,
        stop_distance=15.0,  # stop 105.0 — far below
        entry_edge=50.0,
        bars_held=10,
        highest_close_since_entry=178.0,
    )
    d = evaluate_exit(
        state,
        closes,
        highs,
        lows,
        volumes,
        params=ExitParams(atr_trail_k=0.5),
        directional_params=SHORT_DIRECTION,
    )

    assert d.triggered_rule == "ATR_TRAIL"
    assert d.should_exit is True
    assert d.trail_price == pytest.approx(176.7)
    assert "close 176.0000 < trail 176.7000" in d.reasons[3]
    assert "peak 178.0000" in d.reasons[3]
    assert "atr14" in d.reasons[3]
    assert d.reasons[0].startswith("OK: HARD_STOP")
    assert d.reasons[1].startswith("OK: SIGNAL_FLIP")


# ---------------------------------------------------------------------------
# Rule 5: TIME_STOP (plan §11.6)
# ---------------------------------------------------------------------------


def test_time_stop_triggers_when_position_goes_nowhere():
    """Flat series, 20 bars held: move = 100.0 - 100.0 = 0.0 < 1.0 * ATR
    (exactly 1.0) -> TIME_STOP. exit_edge_threshold=0.0 keeps SIGNAL_DECAY
    quiet (flat edge is exactly 0.0, not < 0.0); the trail (97.0) and the
    stop (95.0) both hold."""
    closes, highs, lows, volumes = make_bars([100.0] * 30)
    state = PositionState(
        entry_price=100.0,
        stop_distance=5.0,
        entry_edge=30.0,
        bars_held=20,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(
        state,
        closes,
        highs,
        lows,
        volumes,
        params=ExitParams(exit_edge_threshold=0.0),
        directional_params=SHORT_DIRECTION,
    )

    assert d.triggered_rule == "TIME_STOP"
    assert d.time_stop_remaining == 0
    assert "held 20 bars >= 20" in d.reasons[4]
    assert "move 0.0000" in d.reasons[4]
    assert "atr14 1.0000" in d.reasons[4]
    assert d.reasons[2].startswith("OK: SIGNAL_DECAY")
    assert d.reasons[3].startswith("OK: ATR_TRAIL")


def test_time_stop_remaining_counts_down_and_floors_at_zero():
    """time_stop_remaining = max(0, time_stop_bars - bars_held), reported on
    every call; a position that HAS moved (uptrend) never time-stops even at
    remaining 0."""
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    for bars_held, expected in ((0, 20), (7, 13), (19, 1), (20, 0), (33, 0)):
        state = PositionState(
            entry_price=100.0,
            stop_distance=10.0,
            entry_edge=40.0,
            bars_held=bars_held,
            highest_close_since_entry=closes[-1],
        )
        d = evaluate_exit(
            state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
        )
        assert d.time_stop_remaining == expected
        # The uptrend moved ~33 points >= 1 ATR, so the time stop never fires.
        assert d.should_exit is False
        assert d.reasons[4].startswith("OK: TIME_STOP")


# ---------------------------------------------------------------------------
# Priority order (mirrors the backtest engine's exit priority)
# ---------------------------------------------------------------------------


def test_priority_hard_stop_beats_flip_decay_and_trail():
    """+2/bar staircase then a -78 crash to 100.0: the crash breaches the
    hard stop (130.0), flips the signal to BEAR, decays the edge AND drops
    below the trail — every condition is reported breached (no OK prefix),
    but HARD_STOP is priority 1 and wins."""
    closes = [100.0 + 2.0 * i for i in range(40)]  # ... 176, 178
    closes.append(100.0)
    closes, highs, lows, volumes = make_bars(closes)
    state = PositionState(
        entry_price=150.0,
        stop_distance=20.0,  # stop 130.0, breached by close 100.0
        entry_edge=45.0,
        bars_held=5,
        highest_close_since_entry=178.0,
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )

    assert d.triggered_rule == "HARD_STOP"
    assert d.should_exit is True
    assert d.reasons[0].startswith("HARD_STOP: stop 130.0000 breached")
    # The lower-priority rules were still evaluated and also breached.
    assert d.reasons[1].startswith("SIGNAL_FLIP: bias BEAR")
    assert d.reasons[2].startswith("SIGNAL_DECAY:")
    assert d.reasons[3].startswith("ATR_TRAIL: close 100.0000 < trail")


# ---------------------------------------------------------------------------
# Short history (plan §11.3: a data gap must never disable the hard stop)
# ---------------------------------------------------------------------------


def test_short_history_hard_stop_still_fires():
    """Two bars only: every signal component and the ATR are in warmup, so
    rules 2-5 report insufficient data and stay quiet — but close 91.5 <=
    stop 92.0 fires the HARD_STOP off raw prices."""
    closes, highs, lows, volumes = make_bars([100.0, 91.5])
    state = PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=1,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(state, closes, highs, lows, volumes)  # default 200-bar params

    assert d.should_exit is True
    assert d.triggered_rule == "HARD_STOP"
    assert d.current_edge is None
    assert d.trail_price is None
    for i in (1, 2, 3, 4):  # SIGNAL_FLIP, SIGNAL_DECAY, ATR_TRAIL, TIME_STOP
        assert "insufficient data" in d.reasons[i]
        assert not d.reasons[i].startswith("OK: ")


def test_short_history_holds_without_false_triggers():
    """Two bars, stop NOT breached: nothing may trigger — in particular the
    all-warmup edge of 0.0 must NOT masquerade as SIGNAL_DECAY."""
    closes, highs, lows, volumes = make_bars([100.0, 99.0])
    state = PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=1,
        highest_close_since_entry=100.0,
    )
    d = evaluate_exit(state, closes, highs, lows, volumes)

    assert d.should_exit is False
    assert d.triggered_rule is None
    assert d.current_edge is None
    assert d.trail_price is None
    assert d.time_stop_remaining == 19
    assert d.reasons[0].startswith("OK: HARD_STOP: close 99.0000 > stop 92.0000")
    assert sum("insufficient data" in r for r in d.reasons) == 4


# ---------------------------------------------------------------------------
# §21 reuse: the exits run the exact live signal code
# ---------------------------------------------------------------------------


def test_current_edge_is_exactly_the_live_signal_edge():
    closes, highs, lows, volumes = make_bars(
        [100.0 * (1.002**i) * (1.0 + 0.03 * math.sin(i / 4.0)) for i in range(40)]
    )
    state = PositionState(
        entry_price=100.0,
        stop_distance=50.0,
        entry_edge=40.0,
        bars_held=3,
        highest_close_since_entry=max(closes),
    )
    d = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )
    live = score_direction(closes, highs, lows, volumes=volumes, params=SHORT_DIRECTION)
    assert d.current_edge == live.directional_edge  # bit-identical, not approx


# ---------------------------------------------------------------------------
# Determinism and reporting invariants
# ---------------------------------------------------------------------------


def test_determinism_identical_inputs_identical_decisions():
    closes, highs, lows, volumes = make_bars(
        [100.0 * (1.002**i) * (1.0 + 0.03 * math.sin(i / 4.0)) for i in range(40)]
    )

    def run():
        state = PositionState(
            entry_price=105.0,
            stop_distance=12.0,
            entry_edge=40.0,
            bars_held=8,
            highest_close_since_entry=max(closes),
        )
        return evaluate_exit(
            state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
        )

    assert run() == run()  # dataclass equality: every field and every reason


def test_stop_price_and_remaining_always_reported():
    """stop_price = entry - stop_distance and time_stop_remaining are
    reported on EVERY decision, triggered or not, long or short history."""
    for closes_raw, bars_held in (([100.0, 99.0], 1), ([100.0 * 1.01**i for i in range(30)], 4)):
        closes, highs, lows, volumes = make_bars(closes_raw)
        state = PositionState(
            entry_price=100.0,
            stop_distance=7.5,
            entry_edge=30.0,
            bars_held=bars_held,
            highest_close_since_entry=max(closes),
        )
        d = evaluate_exit(state, closes, highs, lows, volumes)
        assert d.stop_price == pytest.approx(92.5)
        assert d.time_stop_remaining == 20 - bars_held
        assert len(d.reasons) == 5
        assert rule_names(d) == list(RULES)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def good_state() -> PositionState:
    return PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=1,
        highest_close_since_entry=100.0,
    )


def test_empty_closes_rejected():
    with pytest.raises(ValueError):
        evaluate_exit(good_state(), [], [], [])


def test_misaligned_arrays_rejected():
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    with pytest.raises(ValueError):
        evaluate_exit(good_state(), closes, highs[:-1], lows, volumes)
    with pytest.raises(ValueError):
        evaluate_exit(good_state(), closes, highs, lows[:-1], volumes)
    with pytest.raises(ValueError):
        evaluate_exit(good_state(), closes, highs, lows, volumes[:-1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("stop_distance", 0.0),
        ("stop_distance", -1.0),
        ("bars_held", -1),
    ],
)
def test_invalid_position_state_rejected(field, value):
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    state = good_state()
    setattr(state, field, value)
    with pytest.raises(ValueError):
        evaluate_exit(state, closes, highs, lows, volumes)


# ===========================================================================
# Option exits (plan §11.3 premium hard stop, §11.7 DTE exit, §21 reuse)
# ===========================================================================


def option_rule_names(decision: ExitDecision) -> list[str]:
    return [r.removeprefix("OK: ").split(":")[0] for r in decision.reasons]


def uptrend_setup():
    """The holding series from the stock tests: +1%/bar uptrend where no
    underlying rule fires (bars_held 9, stop far below, edge 66.7)."""
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    state = PositionState(
        entry_price=closes[20],
        stop_distance=10.0,
        entry_edge=40.0,
        bars_held=9,
        highest_close_since_entry=closes[-1],
    )
    return state, closes, highs, lows, volumes


def downtrend_setup():
    """The SIGNAL_FLIP series from the stock tests: -1%/bar downtrend that
    flips the live signal to BEAR while the (huge) underlying stop holds."""
    closes, highs, lows, volumes = make_bars([100.0 * 0.99**i for i in range(30)])
    state = PositionState(
        entry_price=90.0,
        stop_distance=50.0,
        entry_edge=35.0,
        bars_held=5,
        highest_close_since_entry=100.0,
    )
    return state, closes, highs, lows, volumes


def healthy_option() -> OptionState:
    """Premium up, plenty of DTE: neither option rule fires."""
    return OptionState(entry_premium=2.0, current_mid=2.5, dte=45)


# ---------------------------------------------------------------------------
# Parameters (plan §11.3 research parameter, §11.7 threshold)
# ---------------------------------------------------------------------------


def test_option_params_defaults_match_plan_11_3_and_11_7():
    p = ExitParams()
    assert p.premium_hard_stop_pct == 0.45
    assert p.dte_exit_threshold == 21


@pytest.mark.parametrize(
    "kwargs",
    [
        {"premium_hard_stop_pct": 0.0},
        {"premium_hard_stop_pct": -0.1},
        {"premium_hard_stop_pct": 1.5},
        {"dte_exit_threshold": -1},
    ],
)
def test_invalid_option_params_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ExitParams(**kwargs)


def test_option_params_boundary_values_accepted():
    # A 100% premium stop (fires only at a worthless option) and a 0-DTE
    # threshold (fires only on expiry day) are extreme but valid tunings.
    p = ExitParams(premium_hard_stop_pct=1.0, dte_exit_threshold=0)
    assert p.premium_hard_stop_pct == 1.0
    assert p.dte_exit_threshold == 0


# ---------------------------------------------------------------------------
# Holding: all six option rules explained with "OK:" (plan §37)
# ---------------------------------------------------------------------------


def test_option_holding_reports_ok_for_all_six_rules():
    state, closes, highs, lows, volumes = uptrend_setup()
    d = evaluate_option_exit(
        state,
        healthy_option(),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )

    assert d.should_exit is False
    assert d.triggered_rule is None
    assert len(d.reasons) == 6
    assert option_rule_names(d) == list(OPTION_RULES)
    assert all(r.startswith("OK: ") for r in d.reasons)
    # stop_price reports the PREMIUM stop level for options (§11.3):
    # 2.0 * (1 - 0.45).
    assert d.stop_price == pytest.approx(1.1)
    assert d.current_edge == pytest.approx(160.0 / 3.0)  # same live signal
    assert d.time_stop_remaining == 11


# ---------------------------------------------------------------------------
# PREMIUM_HARD_STOP (plan §11.3) — replaces the underlying HARD_STOP
# ---------------------------------------------------------------------------


def test_premium_hard_stop_fires_at_exactly_minus_45_pct():
    """entry premium 2.00, default premium_hard_stop_pct 0.45: the stop sits
    at 2.0 * (1 - 0.45); a mid EXACTLY at the stop fires (<=, boundary
    inclusive — computed with the engine's own formula so the comparison is
    bit-exact), and a mid a hair above holds."""
    state, closes, highs, lows, volumes = uptrend_setup()
    stop = 2.0 * (1.0 - ExitParams().premium_hard_stop_pct)

    d = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=stop, dte=45),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert d.should_exit is True
    assert d.triggered_rule == "PREMIUM_HARD_STOP"
    assert d.reasons[0].startswith("PREMIUM_HARD_STOP: premium stop")
    assert "entry premium 2.0000" in d.reasons[0]
    assert "(1 - 0.45)" in d.reasons[0]
    assert d.stop_price == stop

    held = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=stop + 1e-9, dte=45),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert held.triggered_rule is None
    assert held.reasons[0].startswith("OK: PREMIUM_HARD_STOP")


def test_premium_hard_stop_exact_boundary_with_binary_exact_params():
    """With premium_hard_stop_pct=0.5 and entry 2.0 the stop is EXACTLY 1.0
    (all values binary-exact): mid 1.0 fires, mid 1.000000001 holds."""
    state, closes, highs, lows, volumes = uptrend_setup()
    params = ExitParams(premium_hard_stop_pct=0.5)
    fired = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=1.0, dte=45),
        closes,
        highs,
        lows,
        volumes,
        params=params,
        directional_params=SHORT_DIRECTION,
    )
    assert fired.triggered_rule == "PREMIUM_HARD_STOP"
    assert fired.stop_price == 1.0
    held = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=1.000000001, dte=45),
        closes,
        highs,
        lows,
        volumes,
        params=params,
        directional_params=SHORT_DIRECTION,
    )
    assert held.triggered_rule is None


def test_worthless_option_mid_zero_fires_premium_stop():
    state, closes, highs, lows, volumes = uptrend_setup()
    d = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=0.0, dte=45),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert d.triggered_rule == "PREMIUM_HARD_STOP"


# ---------------------------------------------------------------------------
# DTE_EXIT (plan §11.7) — 21 fires, 22 holds
# ---------------------------------------------------------------------------


def test_dte_21_fires_and_22_holds():
    state, closes, highs, lows, volumes = uptrend_setup()

    d21 = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=2.5, dte=21),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert d21.should_exit is True
    assert d21.triggered_rule == "DTE_EXIT"
    assert "21 DTE <= threshold 21" in d21.reasons[1]
    assert d21.reasons[0].startswith("OK: PREMIUM_HARD_STOP")

    d22 = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=2.5, dte=22),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert d22.should_exit is False
    assert d22.triggered_rule is None
    assert d22.reasons[1] == "OK: DTE_EXIT: 22 DTE > threshold 21"


# ---------------------------------------------------------------------------
# Priority: PREMIUM_HARD_STOP > DTE_EXIT > signal rules (plan §11)
# ---------------------------------------------------------------------------


def test_priority_premium_stop_beats_dte_beats_signal_rules():
    """Downtrend flips the signal to BEAR; dte 10 breaches §11.7; mid 1.0
    breaches the premium stop. All three report breached (no OK prefix) and
    the premium stop wins; removing each condition promotes the next."""
    state, closes, highs, lows, volumes = downtrend_setup()

    all_breached = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=1.0, dte=10),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert all_breached.triggered_rule == "PREMIUM_HARD_STOP"
    assert all_breached.reasons[0].startswith("PREMIUM_HARD_STOP:")
    assert all_breached.reasons[1].startswith("DTE_EXIT: 10 DTE <= threshold 21")
    assert all_breached.reasons[2].startswith("SIGNAL_FLIP: bias BEAR")

    dte_and_signal = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=2.5, dte=10),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert dte_and_signal.triggered_rule == "DTE_EXIT"
    assert dte_and_signal.reasons[0].startswith("OK: PREMIUM_HARD_STOP")
    assert dte_and_signal.reasons[2].startswith("SIGNAL_FLIP: bias BEAR")

    signal_only = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=2.5, dte=45),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert signal_only.triggered_rule == "SIGNAL_FLIP"
    assert signal_only.reasons[0].startswith("OK: PREMIUM_HARD_STOP")
    assert signal_only.reasons[1].startswith("OK: DTE_EXIT")


# ---------------------------------------------------------------------------
# Honest nulls: missing mid / missing DTE report loudly, never silently
# ---------------------------------------------------------------------------


def test_none_current_mid_reports_insufficient_data_but_dte_still_fires():
    state, closes, highs, lows, volumes = uptrend_setup()
    d = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=None, dte=21),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    # The premium stop is skipped but NEVER silently (plan §11.3):
    assert "insufficient data" in d.reasons[0]
    assert not d.reasons[0].startswith("OK:")
    assert d.reasons[0].startswith("PREMIUM_HARD_STOP:")
    assert "not triggered" in d.reasons[0]
    # ... while DTE_EXIT was still evaluated and fires:
    assert d.should_exit is True
    assert d.triggered_rule == "DTE_EXIT"


def test_none_mid_and_none_dte_hold_with_both_reported():
    state, closes, highs, lows, volumes = uptrend_setup()
    d = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=None, dte=None),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert d.should_exit is False
    assert d.triggered_rule is None
    assert "insufficient data" in d.reasons[0]
    assert "insufficient data" in d.reasons[1]
    assert len(d.reasons) == 6


# ---------------------------------------------------------------------------
# The underlying HARD_STOP does NOT apply to option positions (plan §11.3:
# the premium stop replaces it)
# ---------------------------------------------------------------------------


def test_underlying_hard_stop_absent_for_options():
    """A series that breaches the underlying stock stop (close 91.5 <= entry
    100 - stop_distance 8) fires HARD_STOP for stock — but the option
    evaluation has NO underlying HARD_STOP rule at all: the premium stop
    replaced it, and state.stop_distance is ignored."""
    closes, highs, lows, volumes = make_bars([100.0] * 28 + [95.0, 91.5])
    state = PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=5,
        highest_close_since_entry=100.0,
    )
    stock = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )
    assert stock.triggered_rule == "HARD_STOP"

    option = evaluate_option_exit(
        state,
        healthy_option(),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    names = option_rule_names(option)
    assert names == list(OPTION_RULES)
    assert "HARD_STOP" not in names  # only PREMIUM_HARD_STOP exists
    assert option.triggered_rule != "HARD_STOP"
    # No reason line mentions the underlying stop_distance construction.
    assert not any("stop_distance" in r for r in option.reasons)
    # The reported stop is the premium stop, not entry - stop_distance.
    assert option.stop_price == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# §21 shared internals: underlying rules bit-identical across both engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup", [uptrend_setup, downtrend_setup])
def test_underlying_rules_bit_identical_between_stock_and_option(setup):
    state, closes, highs, lows, volumes = setup()
    stock = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )
    option = evaluate_option_exit(
        state,
        healthy_option(),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    # The four underlying-driven reason lines are STRING-identical (§21:
    # shared internals, not a reimplementation) ...
    assert option.reasons[2:] == stock.reasons[1:]
    # ... and every shared numeric output is bit-identical, not approx.
    assert option.current_edge == stock.current_edge
    assert option.trail_price == stock.trail_price
    assert option.time_stop_remaining == stock.time_stop_remaining


def test_underlying_rules_bit_identical_on_sinusoid_series():
    closes, highs, lows, volumes = make_bars(
        [100.0 * (1.002**i) * (1.0 + 0.03 * math.sin(i / 4.0)) for i in range(40)]
    )
    state = PositionState(
        entry_price=100.0,
        stop_distance=50.0,
        entry_edge=40.0,
        bars_held=3,
        highest_close_since_entry=max(closes),
    )
    stock = evaluate_exit(
        state, closes, highs, lows, volumes, directional_params=SHORT_DIRECTION
    )
    option = evaluate_option_exit(
        state,
        healthy_option(),
        closes,
        highs,
        lows,
        volumes,
        directional_params=SHORT_DIRECTION,
    )
    assert option.reasons[2:] == stock.reasons[1:]
    assert option.current_edge == stock.current_edge
    assert option.trail_price == stock.trail_price


def test_option_short_history_insufficient_data_but_premium_stop_works():
    """Two bars: signal/ATR rules report insufficient data exactly like the
    stock engine, but the premium stop still protects the position off the
    raw option quote (plan §11.3)."""
    closes, highs, lows, volumes = make_bars([100.0, 99.0])
    state = PositionState(
        entry_price=100.0,
        stop_distance=8.0,
        entry_edge=30.0,
        bars_held=1,
        highest_close_since_entry=100.0,
    )
    d = evaluate_option_exit(
        state,
        OptionState(entry_premium=2.0, current_mid=1.0, dte=45),
        closes,
        highs,
        lows,
        volumes,
    )
    assert d.triggered_rule == "PREMIUM_HARD_STOP"
    assert d.current_edge is None
    assert d.trail_price is None
    assert sum("insufficient data" in r for r in d.reasons) == 4


# ---------------------------------------------------------------------------
# Determinism and input validation (option path)
# ---------------------------------------------------------------------------


def test_option_determinism():
    state, closes, highs, lows, volumes = downtrend_setup()

    def run():
        return evaluate_option_exit(
            state,
            OptionState(entry_premium=2.0, current_mid=1.5, dte=30),
            closes,
            highs,
            lows,
            volumes,
            directional_params=SHORT_DIRECTION,
        )

    assert run() == run()


@pytest.mark.parametrize(
    "option",
    [
        OptionState(entry_premium=0.0, current_mid=1.0, dte=30),
        OptionState(entry_premium=-2.0, current_mid=1.0, dte=30),
        OptionState(entry_premium=2.0, current_mid=-0.5, dte=30),
        OptionState(entry_premium=2.0, current_mid=1.0, dte=-1),
    ],
)
def test_invalid_option_state_rejected(option):
    state, closes, highs, lows, volumes = uptrend_setup()
    with pytest.raises(ValueError):
        evaluate_option_exit(state, option, closes, highs, lows, volumes)


def test_option_empty_and_misaligned_series_rejected():
    with pytest.raises(ValueError):
        evaluate_option_exit(good_state(), healthy_option(), [], [], [])
    closes, highs, lows, volumes = make_bars([100.0 * 1.01**i for i in range(30)])
    with pytest.raises(ValueError):
        evaluate_option_exit(
            good_state(), healthy_option(), closes, highs[:-1], lows, volumes
        )
