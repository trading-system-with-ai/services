"""§9-S vertical spread selector tests (execution-chains roadmap Phase 1).

Synthetic chains built from the same quote factory conventions as the §9
selector tests, so every selection and every rejection can be hand-checked.
"""
from datetime import date

import pytest

from libs.trading_core.contracts import (
    ContractQuote,
    SelectorParams,
    SpreadParams,
    select_vertical_spread,
)

EXPIRY = date(2026, 9, 24)
SPOT = 100.0


def cq(**overrides) -> ContractQuote:
    """A call passing every §9 default filter; override fields to vary."""
    base = dict(
        expiry=EXPIRY,
        dte=45,
        strike=100.0,
        right="C",
        bid=4.9,
        ask=5.1,
        mid=5.0,
        spread_pct=0.04,
        last=5.0,
        volume=200,
        open_interest=500,
        iv=0.30,
        delta=0.55,
        gamma=0.02,
        theta=-0.05,
        vega=0.12,
    )
    base.update(overrides)
    return ContractQuote(**base)


def bull_chain() -> list[ContractQuote]:
    """Long candidate at 100 (passes §9) + short candidates OTM-ward.

    Strikes 103/105/108 sit at 3%/5%/8% of spot beyond the long strike —
    inside the default [2%, 15%] width band; 105 is exactly the 5% target.
    OTM legs carry sub-§9 deltas so only the 100 strike can be the long leg.
    """
    return [
        cq(strike=100.0),
        cq(strike=103.0, mid=3.6, bid=3.5, ask=3.7, delta=0.38, theta=-0.04, vega=0.10),
        cq(strike=105.0, mid=2.8, bid=2.7, ask=2.9, delta=0.30, theta=-0.035, vega=0.09),
        cq(strike=108.0, mid=1.9, bid=1.8, ask=2.0, delta=0.22, theta=-0.03, vega=0.07),
    ]


def test_bull_call_spread_happy_path_hand_checked():
    sel = select_vertical_spread(bull_chain(), "BULL_CALL_SPREAD", SPOT)
    assert sel.fail_reasons == []
    c = sel.candidate
    assert c is not None
    # Long leg = §9 rank-1 (the 100 strike, the only §9-eligible delta).
    assert c.long_leg.strike == 100.0
    assert sel.long_leg_scored is not None and sel.long_leg_scored.rank == 1
    # Short leg = nearest to the 5% target -> the 105 strike.
    assert c.short_leg.strike == 105.0
    # Defined-risk arithmetic, hand-computed.
    assert c.net_debit == pytest.approx(5.0 - 2.8)
    assert c.width == pytest.approx(5.0)
    assert c.max_loss == pytest.approx(2.2)
    assert c.max_profit == pytest.approx(5.0 - 2.2)
    assert c.breakeven == pytest.approx(100.0 + 2.2)
    # Net greeks = long - short.
    assert c.net_delta == pytest.approx(0.55 - 0.30)
    assert c.net_theta == pytest.approx(-0.05 - (-0.035))
    assert c.net_vega == pytest.approx(0.12 - 0.09)
    # §37: the rationale carries the real numbers.
    joined = " ".join(c.rationale)
    assert "MAX LOSS" in joined and "breakeven" in joined


def test_bear_put_spread_mirror():
    chain = [
        cq(right="P", strike=100.0, delta=-0.55),
        cq(right="P", strike=95.0, mid=2.8, bid=2.7, ask=2.9, delta=-0.30),
        cq(right="P", strike=97.0, mid=3.6, bid=3.5, ask=3.7, delta=-0.38),
    ]
    sel = select_vertical_spread(chain, "BEAR_PUT_SPREAD", SPOT)
    c = sel.candidate
    assert c is not None
    assert c.long_leg.strike == 100.0
    assert c.short_leg.strike == 95.0  # nearest to the 5% target, OTM-ward DOWN
    assert c.breakeven == pytest.approx(100.0 - c.net_debit)
    assert c.max_profit == pytest.approx(5.0 - c.net_debit)


def test_no_eligible_long_leg_names_the_blockers():
    chain = [cq(delta=0.10), cq(strike=105.0, delta=0.05)]  # all below §9 window
    sel = select_vertical_spread(chain, "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "no §9-eligible long leg" in sel.fail_reasons[0]
    assert "|delta|" in sel.fail_reasons[0] or "delta" in sel.fail_reasons[0]


def test_no_short_leg_in_width_band():
    # Only strike beyond the long leg is 30% away — outside width_pct_max.
    chain = [cq(strike=100.0), cq(strike=130.0, mid=0.5, delta=0.05)]
    sel = select_vertical_spread(chain, "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "no eligible short leg" in sel.fail_reasons[0]
    assert sel.long_leg_scored is not None  # the long leg WAS found


def test_short_leg_filters_day_close_oi_and_spread():
    base_long = cq(strike=100.0)
    # 105 with day_close basis -> rejected (unknown spread cannot be sold).
    day_close = cq(strike=105.0, mid=2.8, delta=0.30, price_basis="day_close")
    sel = select_vertical_spread([base_long, day_close], "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "price_basis" in sel.fail_reasons[0]
    # 105 with thin OI -> rejected under the short-leg floor.
    thin = cq(strike=105.0, mid=2.8, delta=0.30, open_interest=10)
    sel = select_vertical_spread([base_long, thin], "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "open interest" in sel.fail_reasons[0]
    # 105 with a fat spread -> rejected under the short-leg cap.
    fat = cq(strike=105.0, mid=2.8, delta=0.30, spread_pct=0.40)
    sel = select_vertical_spread([base_long, fat], "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "spread above cap" in sel.fail_reasons[0]


def test_degenerate_quotes_fail_closed():
    # Short mid >= long mid -> net credit "debit spread": quote anomaly.
    chain = [cq(strike=100.0), cq(strike=105.0, mid=6.0, bid=5.9, ask=6.1, delta=0.30)]
    sel = select_vertical_spread(chain, "BULL_CALL_SPREAD", SPOT)
    assert sel.candidate is None
    assert "net debit" in sel.fail_reasons[0]


def test_net_greeks_are_none_safe():
    chain = [
        cq(strike=100.0),
        cq(strike=105.0, mid=2.8, delta=0.30, theta=None, vega=None),
    ]
    sel = select_vertical_spread(chain, "BULL_CALL_SPREAD", SPOT)
    c = sel.candidate
    assert c is not None
    assert c.net_theta is None and c.net_vega is None  # never zero-filled
    assert c.net_delta is not None
    assert "unknown" in c.rationale[2]


def test_determinism_and_validation():
    a = select_vertical_spread(bull_chain(), "BULL_CALL_SPREAD", SPOT)
    b = select_vertical_spread(bull_chain(), "BULL_CALL_SPREAD", SPOT)
    assert a.candidate.short_leg.strike == b.candidate.short_leg.strike
    assert a.candidate.net_debit == b.candidate.net_debit
    with pytest.raises(ValueError, match="instrument"):
        select_vertical_spread(bull_chain(), "IRON_CONDOR", SPOT)
    with pytest.raises(ValueError, match="spot"):
        select_vertical_spread(bull_chain(), "BULL_CALL_SPREAD", 0.0)
    with pytest.raises(ValueError, match="width_pct"):
        SpreadParams(width_pct_min=0.2, width_pct_max=0.1)
