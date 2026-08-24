"""Scenario revaluation tests (Phase D design §8.2 / §8.7).

Hand-computed baseline used across the file (r = q = 0 so every number is
checkable with erf alone):

    LONG CALL: S0 = 100, K = 100, T0 = 1.0, iv0 = 0.20, right = C
      d1 = (0 + 0.5*0.04*1)/(0.20) =  0.10 ;  d2 = -0.10
      model0 = 100*(N(0.10) - N(-0.10)) = 7.96556745540580
      mark0  = 8.00           (the chain mid we anchor on)
      basis  = 8.00 - 7.96556745540580 = 0.03443254459420

    Every scenario price is bs_price(S1, K, T1, iv1) + 0.0344325445942,
    so the ZERO scenario reprices to exactly 8.00 and the leg's P&L is
    exactly 0.0 (not "about zero") — that is the point of the basis.

IV shock convention (design §8.2): RELATIVE and MULTIPLICATIVE on the LEVEL.
    iv_shock = +0.20  ->  iv1 = 0.20 * 1.20 = 0.24
    iv_shock = -0.40  ->  iv1 = 0.20 * 0.60 = 0.12

Sign convention: quantity is SIGNED in contracts, short legs NEGATIVE, and
P&L is gain-positive USD.
"""
from __future__ import annotations

import math

import pytest

from libs.trading_core.options.bs import bs_price
from libs.trading_core.options.reval import (
    DAYS_PER_YEAR,
    IV_FLOOR,
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL,
    OptionLeg,
    StockLeg,
    leg_baseline,
    reval_leg,
    scenario_pnl,
)

# --- the hand-computed anchor ---------------------------------------------
MODEL0_ATM_CALL = 7.96556745540580
MARK0 = 8.00
BASIS = MARK0 - MODEL0_ATM_CALL  # 0.03443254459420


def make_call(quantity: int = 1, mark0: float = MARK0, **kw) -> OptionLeg:
    """The canonical ATM long call from the module docstring."""
    params = dict(
        key="AAPL#1",
        ticker="AAPL",
        right="C",
        strike=100.0,
        t_years=1.0,
        quantity=quantity,
        spot0=100.0,
        mark0=mark0,
        iv0=0.20,
        r=0.0,
        q=0.0,
    )
    params.update(kw)
    return OptionLeg(**params)


# ---------------------------------------------------------------------------
# leg_baseline — the basis anchor
# ---------------------------------------------------------------------------


def test_leg_baseline_hand_computed_basis() -> None:
    base = leg_baseline(make_call())
    assert base.model0 == pytest.approx(MODEL0_ATM_CALL, abs=1e-12)
    assert base.basis == pytest.approx(BASIS, abs=1e-12)
    assert base.mark0 == MARK0
    assert base.price0 == MARK0  # the baseline price IS the market mark
    assert base.method == METHOD_FULL_REVAL
    # model0 + basis reconstructs the mark exactly.
    assert base.model0 + base.basis == pytest.approx(MARK0, abs=1e-12)


def test_leg_baseline_without_iv_is_delta_linear() -> None:
    base = leg_baseline(make_call(iv0=None, delta0=0.54))
    assert base.model0 is None
    assert base.basis is None
    assert base.method == METHOD_DELTA_LINEAR
    assert base.price0 == MARK0


def test_leg_baseline_at_expiry_carries_no_basis() -> None:
    """A settled option is worth its payoff — a bid/ask residual is not
    carried through settlement."""
    leg = make_call(t_years=0.0, mark0=12.0, strike=90.0)
    base = leg_baseline(leg)
    assert base.model0 == pytest.approx(10.0)  # intrinsic 100 - 90
    assert base.basis == 0.0


# ---------------------------------------------------------------------------
# The zero-scenario identity — EXACTLY 0.0
# ---------------------------------------------------------------------------


def test_zero_scenario_is_exactly_zero_for_one_leg() -> None:
    res = scenario_pnl([], [make_call(quantity=3)])
    assert res.total == 0.0  # bit-exact, not approx
    assert res.per_key["AAPL#1"] == 0.0
    assert res.method_by_key["AAPL#1"] == METHOD_FULL_REVAL


def test_zero_scenario_is_exactly_zero_for_a_mixed_book() -> None:
    """Stock + long call + short put + an IV-less leg: still exactly 0."""
    legs = [
        make_call(quantity=2),
        make_call(key="AAPL#2", right="P", strike=95.0, quantity=-4, mark0=3.1),
        make_call(key="MSFT#7", ticker="MSFT", quantity=1, iv0=None, delta0=0.6),
    ]
    stocks = [StockLeg(key="NVDA#3", ticker="NVDA", quantity=-150, spot0=42.5)]
    res = scenario_pnl(stocks, legs)
    assert res.total == 0.0
    assert all(v == 0.0 for v in res.per_key.values())
    assert res.method_coverage == {METHOD_FULL_REVAL: 3, METHOD_DELTA_LINEAR: 1}
    assert res.fully_revalued is False


def test_zero_scenario_is_zero_regardless_of_how_wrong_the_mark_is() -> None:
    """The basis absorbs ANY model/market gap — that is its job."""
    for mark0 in (0.01, 7.9655674554058, 8.0, 50.0):
        res = scenario_pnl([], [make_call(mark0=mark0)])
        assert res.total == 0.0, mark0


def test_empty_book_is_exactly_zero_not_missing() -> None:
    res = scenario_pnl([], [], spot_shock=-0.25, iv_shock=1.0, days_forward=30)
    assert res.total == 0.0
    assert res.per_key == {}
    assert res.method_coverage == {METHOD_FULL_REVAL: 0, METHOD_DELTA_LINEAR: 0}


# ---------------------------------------------------------------------------
# reval_leg — the basis is HELD, and expiry is intrinsic
# ---------------------------------------------------------------------------


def test_reval_leg_holds_the_basis_constant() -> None:
    """price1 = bs_price(S1, K, T1, iv1) + basis, with the SAME basis."""
    leg = make_call()
    # scenario: S -10 %, IV +50 % (0.20 -> 0.30), 30 days forward
    s1 = 90.0
    iv1 = 0.30
    t1 = 1.0 - 30.0 / DAYS_PER_YEAR
    expected = bs_price(s1, 100.0, t1, iv1, "C", 0.0, 0.0) + BASIS
    got = reval_leg(leg, spot1=s1, iv1=iv1, days_forward=30.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_reval_leg_at_expiry_is_intrinsic_with_no_basis() -> None:
    """T1 <= 0 settles at the payoff: 115 - 100 = 15, NOT 15 + basis."""
    leg = make_call()  # T0 = 1.0
    got = reval_leg(leg, spot1=115.0, iv1=0.20, days_forward=365.0)
    assert got == pytest.approx(15.0, abs=1e-12)
    assert got != pytest.approx(15.0 + BASIS, abs=1e-9)
    # OTM at expiry is worth exactly zero.
    otm = reval_leg(leg, spot1=80.0, iv1=0.20, days_forward=400.0)
    assert otm == 0.0


def test_reval_leg_past_expiry_put_is_intrinsic() -> None:
    put = make_call(right="P", strike=100.0, mark0=6.0)
    got = reval_leg(put, spot1=88.0, iv1=0.20, days_forward=365.0)
    assert got == pytest.approx(12.0, abs=1e-12)


def test_reval_leg_without_iv_raises_rather_than_faking_full_reval() -> None:
    leg = make_call(iv0=None, delta0=0.5)
    with pytest.raises(ValueError, match="requires iv0"):
        reval_leg(leg, spot1=100.0, iv1=0.2, days_forward=0.0)


def test_expiry_scenario_through_scenario_pnl_uses_intrinsic() -> None:
    """A 2-contract long call held to expiry with spot at 115."""
    res = scenario_pnl(
        [], [make_call(quantity=2)], spot_shock=0.15, days_forward=365.0
    )
    # price1 = 15.0 (intrinsic, no basis); pnl = 2 * 100 * (15.0 - 8.00)
    assert res.total == pytest.approx(2 * 100 * (15.0 - 8.00), abs=1e-9)
    assert res.total == pytest.approx(1400.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Hand-computed scenario numbers
# ---------------------------------------------------------------------------


def test_hand_computed_down_move_with_iv_spike() -> None:
    """S -10 %, IV +20 % (0.20 -> 0.24), no time: full revaluation."""
    leg = make_call(quantity=1)
    res = scenario_pnl([], [leg], spot_shock=-0.10, iv_shock=0.20)
    expected_price1 = bs_price(90.0, 100.0, 1.0, 0.24, "C", 0.0, 0.0) + BASIS
    expected = 1 * 100 * (expected_price1 - MARK0)
    assert res.total == pytest.approx(expected, abs=1e-9)
    # Direction sanity: a long ATM call loses on a 10 % drop even with a
    # 20 % relative IV bid at this tenor.
    assert res.total < 0.0


def test_iv_shock_is_relative_multiplicative_not_additive() -> None:
    """+0.20 means iv1 = iv0 * 1.20 = 0.24, NOT iv0 + 0.20 = 0.40."""
    leg = make_call()
    res = scenario_pnl([], [leg], iv_shock=0.20)
    as_multiplicative = 100 * (
        bs_price(100.0, 100.0, 1.0, 0.24, "C", 0.0, 0.0) + BASIS - MARK0
    )
    as_additive = 100 * (
        bs_price(100.0, 100.0, 1.0, 0.40, "C", 0.0, 0.0) + BASIS - MARK0
    )
    assert res.total == pytest.approx(as_multiplicative, abs=1e-9)
    assert res.total != pytest.approx(as_additive, abs=1.0)


def test_iv_crush_hurts_a_long_option_and_helps_a_short_one() -> None:
    long_leg = make_call(quantity=1)
    short_leg = make_call(key="AAPL#s", quantity=-1)
    long_res = scenario_pnl([], [long_leg], iv_shock=-0.40)
    short_res = scenario_pnl([], [short_leg], iv_shock=-0.40)
    assert long_res.total < 0.0
    assert short_res.total > 0.0
    assert long_res.total == pytest.approx(-short_res.total, abs=1e-9)


def test_time_decay_only_hurts_a_long_option() -> None:
    res = scenario_pnl([], [make_call()], days_forward=30.0)
    assert res.total < 0.0  # theta, isolated


def test_iv_shock_floor_does_not_raise_at_minus_one() -> None:
    """A -100 % IV shock degrades to a near-zero vol, never a ValueError
    from the pricer (iv <= 0)."""
    res = scenario_pnl([], [make_call()], iv_shock=-1.0)
    assert math.isfinite(res.total)
    # iv1 is floored at IV_FLOOR, so the ATM call is worth ~0 and the long
    # loses essentially the whole model value (the basis cancels).
    expected = 100 * (
        bs_price(100.0, 100.0, 1.0, IV_FLOOR, "C", 0.0, 0.0) - MODEL0_ATM_CALL
    )
    assert res.total == pytest.approx(expected, abs=1e-9)
    assert res.total == pytest.approx(-796.55, abs=0.01)


# ---------------------------------------------------------------------------
# Stock legs — exactly linear
# ---------------------------------------------------------------------------


def test_stock_leg_is_exactly_linear() -> None:
    long_stock = StockLeg(key="AAPL#s", ticker="AAPL", quantity=100, spot0=150.0)
    res = scenario_pnl([long_stock], [], spot_shock=-0.10)
    assert res.total == pytest.approx(100 * 150.0 * -0.10, abs=1e-9)
    assert res.total == pytest.approx(-1500.0, abs=1e-9)
    assert res.method_by_key["AAPL#s"] == METHOD_FULL_REVAL


def test_short_stock_gains_on_a_drop() -> None:
    short = StockLeg(key="TSLA#9", ticker="TSLA", quantity=-50, spot0=200.0)
    res = scenario_pnl([short], [], spot_shock=-0.20)
    assert res.total == pytest.approx(-50 * 200.0 * -0.20, abs=1e-9)
    assert res.total == pytest.approx(2000.0, abs=1e-9)


def test_stock_ignores_iv_and_time() -> None:
    leg = StockLeg(key="A#1", ticker="A", quantity=10, spot0=50.0)
    a = scenario_pnl([leg], [], spot_shock=0.03)
    b = scenario_pnl([leg], [], spot_shock=0.03, iv_shock=2.0, days_forward=90)
    assert a.total == b.total == pytest.approx(15.0, abs=1e-9)


def test_stock_scales_exactly_with_quantity() -> None:
    base = StockLeg(key="A#1", ticker="A", quantity=1, spot0=50.0)
    one = scenario_pnl([base], [], spot_shock=-0.08).total
    for q in (2, 7, 33, 500):
        many = scenario_pnl([base.scaled(q)], [], spot_shock=-0.08).total
        assert many == pytest.approx(one * q, rel=1e-12)


# ---------------------------------------------------------------------------
# Spreads and income structures — leg signs net correctly
# ---------------------------------------------------------------------------


def test_debit_call_spread_legs_net() -> None:
    """Long 100C / short 110C, 1 spread. The spread's P&L must equal the
    sum of the two legs, and it must be BOUNDED by the spread width."""
    long_leg = make_call(key="X#1:long", quantity=1, strike=100.0, mark0=8.0)
    short_leg = make_call(key="X#1:short", quantity=-1, strike=110.0, mark0=3.5)
    res = scenario_pnl([], [long_leg, short_leg], spot_shock=0.30, days_forward=365.0)
    # At expiry with spot 130: long worth 30, short worth -20 -> net 10.
    # Debit paid per share = 8.0 - 3.5 = 4.5 -> P&L = (10 - 4.5) * 100 = 550.
    assert res.per_key["X#1:long"] == pytest.approx(100 * (30.0 - 8.0), abs=1e-9)
    assert res.per_key["X#1:short"] == pytest.approx(-100 * (20.0 - 3.5), abs=1e-9)
    assert res.total == pytest.approx(550.0, abs=1e-9)
    # Max value of a 10-wide spread is the width; the gain cannot exceed it.
    assert res.total <= (110.0 - 100.0) * 100 - (8.0 - 3.5) * 100 + 1e-9


def test_covered_call_short_leg_sign() -> None:
    """Long 100 shares + short 1 call: the short call LOSES on a rally."""
    stock = StockLeg(key="A#1", ticker="AAPL", quantity=100, spot0=100.0)
    short_call = make_call(key="A#1:cc", quantity=-1, strike=105.0, mark0=2.0)
    res = scenario_pnl([stock], [short_call], spot_shock=0.20)
    assert res.per_key["A#1"] == pytest.approx(2000.0, abs=1e-9)  # stock gains
    assert res.per_key["A#1:cc"] < 0.0  # short call loses
    # The short call caps the upside: total < the naked stock gain.
    assert res.total < res.per_key["A#1"]


def test_cash_secured_put_short_leg_gains_when_it_expires_worthless() -> None:
    short_put = make_call(
        key="A#csp", right="P", strike=95.0, quantity=-2, mark0=4.0
    )
    res = scenario_pnl([], [short_put], spot_shock=0.10, days_forward=365.0)
    # Spot 110 > 95 -> put expires at 0; short collected 4.00 per share.
    assert res.total == pytest.approx(2 * 100 * 4.0, abs=1e-9)
    assert res.total == pytest.approx(800.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Linearity in quantity (spec §67 property)
# ---------------------------------------------------------------------------


def test_option_pnl_is_linear_in_quantity() -> None:
    one = scenario_pnl(
        [], [make_call(quantity=1)], spot_shock=-0.12, iv_shock=0.35, days_forward=10
    ).total
    for q in (2, 5, 13, 100):
        many = scenario_pnl(
            [],
            [make_call(quantity=q)],
            spot_shock=-0.12,
            iv_shock=0.35,
            days_forward=10,
        ).total
        assert many == pytest.approx(one * q, rel=1e-12)


def test_pnl_flips_sign_with_quantity_sign() -> None:
    long_res = scenario_pnl([], [make_call(quantity=4)], spot_shock=-0.07).total
    short_res = scenario_pnl([], [make_call(quantity=-4)], spot_shock=-0.07).total
    assert long_res == pytest.approx(-short_res, abs=1e-9)


def test_scaled_helper_matches_an_explicitly_built_leg() -> None:
    scaled = make_call(quantity=1).scaled(7)
    explicit = make_call(quantity=7)
    assert scaled == explicit


# ---------------------------------------------------------------------------
# Per-ticker shocks & the spot0 override
# ---------------------------------------------------------------------------


def test_per_ticker_shock_overrides_the_uniform_one() -> None:
    a = StockLeg(key="A#1", ticker="AAPL", quantity=10, spot0=100.0)
    b = StockLeg(key="B#1", ticker="MSFT", quantity=10, spot0=100.0)
    res = scenario_pnl(
        [a, b], [], spot_shock=-0.05, spot_shock_by_ticker={"AAPL": -0.20}
    )
    assert res.per_key["A#1"] == pytest.approx(-200.0)  # -20 % override
    assert res.per_key["B#1"] == pytest.approx(-50.0)   # -5 % uniform


def test_spot0_by_ticker_overrides_the_leg_baseline_spot() -> None:
    leg = StockLeg(key="A#1", ticker="AAPL", quantity=10, spot0=100.0)
    res = scenario_pnl([leg], [], spot0_by_ticker={"AAPL": 200.0}, spot_shock=-0.10)
    assert res.per_key["A#1"] == pytest.approx(-200.0)


# ---------------------------------------------------------------------------
# The DELTA_LINEAR fallback is LABELLED, never silently mixed in
# ---------------------------------------------------------------------------


def test_missing_iv_falls_back_to_delta_linear_and_says_so() -> None:
    leg = make_call(iv0=None, delta0=0.55, quantity=2)
    res = scenario_pnl([], [leg], spot_shock=-0.10, iv_shock=0.5, days_forward=7)
    # 2 * 100 * 0.55 * 100.0 * -0.10
    assert res.total == pytest.approx(-1100.0, abs=1e-9)
    assert res.method_by_key["AAPL#1"] == METHOD_DELTA_LINEAR
    assert res.method_coverage == {METHOD_FULL_REVAL: 0, METHOD_DELTA_LINEAR: 1}
    assert res.fully_revalued is False
    assert len(res.notes) == 1
    note = res.notes[0]
    assert "no iv0" in note and "DELTA_LINEAR" in note
    assert "cannot see the IV shock" in note


def test_delta_linear_fallback_ignores_iv_and_time_as_documented() -> None:
    leg = make_call(iv0=None, delta0=0.55)
    a = scenario_pnl([], [leg], spot_shock=-0.10).total
    b = scenario_pnl([], [leg], spot_shock=-0.10, iv_shock=3.0, days_forward=180).total
    assert a == b


def test_leg_with_neither_iv_nor_delta_contributes_zero_with_a_note() -> None:
    leg = make_call(iv0=None, delta0=None)
    res = scenario_pnl([], [leg], spot_shock=-0.30)
    assert res.per_key["AAPL#1"] == 0.0
    assert res.method_by_key["AAPL#1"] == METHOD_DELTA_LINEAR
    assert "unvaluable" in res.notes[0]
    assert "understated" in res.notes[0]


def test_method_coverage_counts_a_mixed_book() -> None:
    legs = [
        make_call(key="A#1"),
        make_call(key="A#2"),
        make_call(key="A#3", iv0=None, delta0=0.4),
    ]
    stocks = [StockLeg(key="S#1", ticker="AAPL", quantity=5, spot0=100.0)]
    res = scenario_pnl(stocks, legs, spot_shock=-0.05)
    assert res.method_coverage == {METHOD_FULL_REVAL: 3, METHOD_DELTA_LINEAR: 1}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"key": ""},
        {"ticker": ""},
        {"right": "X"},
        {"strike": 0.0},
        {"strike": -1.0},
        {"spot0": 0.0},
        {"mark0": math.nan},
        {"quantity": 1.5},
        {"quantity": True},
        {"multiplier": 0},
        {"iv0": 0.0},
        {"iv0": -0.2},
        {"t_years": math.inf},
    ],
)
def test_option_leg_rejects_malformed_input(kw: dict) -> None:
    with pytest.raises(ValueError):
        make_call(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"key": ""},
        {"ticker": ""},
        {"quantity": 2.5},
        {"spot0": 0.0},
        {"spot0": -3.0},
    ],
)
def test_stock_leg_rejects_malformed_input(kw: dict) -> None:
    params = dict(key="A#1", ticker="AAPL", quantity=10, spot0=100.0)
    params.update(kw)
    with pytest.raises(ValueError):
        StockLeg(**params)


def test_duplicate_leg_key_raises() -> None:
    with pytest.raises(ValueError, match="duplicate leg key"):
        scenario_pnl([], [make_call(), make_call()])


def test_duplicate_key_across_stock_and_option_raises() -> None:
    stock = StockLeg(key="AAPL#1", ticker="AAPL", quantity=1, spot0=100.0)
    with pytest.raises(ValueError, match="duplicate leg key"):
        scenario_pnl([stock], [make_call(key="AAPL#1")])


def test_negative_days_forward_raises() -> None:
    with pytest.raises(ValueError, match="days_forward must be >= 0"):
        scenario_pnl([], [make_call()], days_forward=-1.0)


def test_shock_to_or_below_minus_one_raises_for_options() -> None:
    with pytest.raises(ValueError, match="cannot go negative"):
        scenario_pnl([], [make_call()], spot_shock=-1.0)


def test_scaled_rejects_a_negative_factor() -> None:
    with pytest.raises(ValueError):
        make_call().scaled(-1)


def test_scenario_pnl_result_is_frozen_and_copies_its_mappings() -> None:
    res = scenario_pnl([], [make_call()], spot_shock=0.01)
    with pytest.raises(Exception):
        res.total = 1.0  # type: ignore[misc]
    assert isinstance(res.per_key, dict)
    assert isinstance(res.notes, tuple)
