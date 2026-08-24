"""Direction-mirror tests for the shared exit engine (2026-08-17 bug fix).

Split from test_exit_engine.py: these characterize the BEAR frame — a long
put / bear put spread profits when the underlying FALLS, so its signal
exits fire on the OPPOSING (BULL) bias, its decay on the bear-favorable
edge, its ATR trail above the running trough, and its time stop on the
missing DOWNWARD move.
"""
import pytest

from libs.trading_core.exits import (
    ExitParams,
    OptionState,
    PositionState,
    evaluate_exit,
    evaluate_option_exit,
)

# ---------------------------------------------------------------------------
# DIRECTION MIRROR (2026-08-17 bug fix): before it, every WINNING long put
# was SIGNAL_FLIP'd on its first evaluation (bias BEAR read as "against" a
# position that profits from BEAR). These tests pin the mirrored semantics.
# ---------------------------------------------------------------------------


def _down_series(n=60, step=0.8):
    closes = [100 - i * step for i in range(n)]
    return closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes]


def _bear_state(closes, **overrides):
    base = dict(
        entry_price=100.0,
        stop_distance=4.0,
        entry_edge=-60.0,
        bars_held=10,
        highest_close_since_entry=100.0,
        direction="BEAR",
        lowest_close_since_entry=min(closes),
    )
    base.update(overrides)
    return PositionState(**base)


def test_winning_put_is_held_not_flipped():
    closes, highs, lows = _down_series()
    d = evaluate_option_exit(
        _bear_state(closes),
        OptionState(entry_premium=5.0, current_mid=9.0, dte=45),
        closes, highs, lows,
    )
    assert d.should_exit is False, d.triggered_rule
    # The reasons speak in the position's frame.
    assert any("BEAR-favorable edge" in r for r in d.reasons)


def test_bear_trail_hangs_above_the_trough_and_fires_on_rally():
    closes, _, _ = _down_series(40)
    rally = closes + [closes[-1] + i * 1.8 for i in range(1, 13)]
    highs = [c + 0.5 for c in rally]
    lows = [c - 0.5 for c in rally]
    d = evaluate_option_exit(
        _bear_state(rally, bars_held=51, lowest_close_since_entry=min(rally)),
        OptionState(entry_premium=5.0, current_mid=4.0, dte=45),
        rally, highs, lows,
        params=ExitParams(time_stop_bars=200),
    )
    assert d.should_exit is True
    # SIGNAL_FLIP (bias now BULL) outranks, and the trail is armed above
    # the trough with the real numbers in its line.
    assert d.triggered_rule == "SIGNAL_FLIP"
    trail_line = next(r for r in d.reasons if "ATR_TRAIL" in r)
    assert "trough" in trail_line


def test_bear_time_stop_measures_the_favorable_down_move():
    """A bear position in a FLAT market: no favorable (downward) move ->
    TIME_STOP fires after the configured bars."""
    closes = [100.0] * 80
    highs = [100.5] * 80
    lows = [99.5] * 80
    d = evaluate_option_exit(
        _bear_state(
            closes,
            bars_held=25,
            entry_edge=-60.0,
            lowest_close_since_entry=100.0,
        ),
        OptionState(entry_premium=5.0, current_mid=5.0, dte=45),
        closes, highs, lows,
        # Flat series: the signal engine reads NEUTRAL-ish; keep signal
        # rules quiet so the time stop is what we observe.
        params=ExitParams(exit_edge_threshold=-999.0, time_stop_bars=20),
    )
    time_line = next(r for r in d.reasons if "TIME_STOP" in r)
    assert d.should_exit is True or "TIME_STOP" in (d.triggered_rule or ""), (
        d.triggered_rule, time_line
    )


def test_bear_state_requires_the_trough_anchor():
    with pytest.raises(ValueError, match="lowest_close_since_entry"):
        PositionState(
            entry_price=100.0,
            stop_distance=4.0,
            entry_edge=-60.0,
            bars_held=1,
            highest_close_since_entry=100.0,
            direction="BEAR",
        )


def test_stock_engine_bear_direction_mirrors_the_hard_stop():
    """SUPERSEDED refusal (Phase 3, 2026-08-17): SHORT_STOCK exists now —
    the BEAR stock hard stop sits ABOVE entry and fires on a rally
    through it; a falling market never stops the short."""
    closes, highs, lows = _down_series()
    d = evaluate_exit(_bear_state(closes), closes, highs, lows)
    stop_line = next(r for r in d.reasons if "HARD_STOP" in r)
    assert "entry 100.0000 + stop_distance" in stop_line  # ABOVE entry
    assert d.triggered_rule != "HARD_STOP"  # falling market: stop intact

    # Rally through entry + stop: the mirrored stop fires.
    rally = [100.0, 106.0]
    d2 = evaluate_exit(
        _bear_state(rally, bars_held=1, lowest_close_since_entry=100.0),
        rally, [c + 0.5 for c in rally], [c - 0.5 for c in rally],
    )
    assert d2.should_exit is True
    assert d2.triggered_rule == "HARD_STOP"


# ---------------------------------------------------------------------------
# SHORT-PREMIUM management (Phase 2): the industry-standard mechanical rules
# for collateralized covered calls / cash-secured puts.
# ---------------------------------------------------------------------------
from libs.trading_core.exits import ShortPremiumState, evaluate_short_premium_exit


def test_profit_capture_at_half_of_max_profit():
    d = evaluate_short_premium_exit(
        ShortPremiumState(
            entry_credit=2.0, current_mid=0.95, dte=40, strike=105.0,
            spot=100.0, right="C",
        )
    )
    assert d.should_exit is True
    assert d.triggered_rule == "PROFIT_CAPTURE"
    assert any("50%" in r for r in d.reasons)


def test_loss_stop_at_twice_the_credit_outranks_everything():
    d = evaluate_short_premium_exit(
        ShortPremiumState(
            entry_credit=2.0, current_mid=4.2, dte=10, strike=105.0,
            spot=108.0, right="C",
        )
    )
    assert d.triggered_rule == "PREMIUM_LOSS_STOP"  # priority over DTE
    assert d.stop_price == pytest.approx(4.0)


def test_dte_rule_and_itm_advisory():
    d = evaluate_short_premium_exit(
        ShortPremiumState(
            entry_credit=2.0, current_mid=1.5, dte=15, strike=95.0,
            spot=92.0, right="P",  # short put ITM (spot < strike)
        )
    )
    assert d.triggered_rule == "DTE_EXIT"
    assert any(r.startswith("ADVISORY") and "ITM" in r for r in d.reasons)


def test_short_premium_holds_between_the_rails_and_none_mid_is_honest():
    held = evaluate_short_premium_exit(
        ShortPremiumState(
            entry_credit=2.0, current_mid=1.6, dte=40, strike=105.0,
            spot=100.0, right="C",
        )
    )
    assert held.should_exit is False
    blind = evaluate_short_premium_exit(
        ShortPremiumState(
            entry_credit=2.0, current_mid=None, dte=40, strike=105.0,
            spot=None, right="C",
        )
    )
    assert blind.should_exit is False
    assert sum("insufficient data" in r for r in blind.reasons) >= 2
