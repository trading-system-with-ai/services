"""Portfolio Greeks aggregation tests (development plan §16).

Every case hand-computes the sums in comments so the qty * multiplier *
per-share-greek arithmetic is auditable line by line.
"""
import pytest

from libs.trading_core.greeks import (
    PositionGreeksInput,
    aggregate_greeks,
    position_contribution,
)


def stock(qty: int = 100, spot: float = 50.0) -> PositionGreeksInput:
    """Long stock: delta exactly 1.0 per share, no gamma/theta/vega."""
    return PositionGreeksInput(
        ticker="NVDA",
        instrument="LONG_STOCK",
        quantity=qty,
        multiplier=1,
        spot=spot,
        delta=1.0,
        gamma=0.0,
        theta_per_day=0.0,
        vega=0.0,
    )


def call(qty: int = 4, spot: float = 50.0) -> PositionGreeksInput:
    """Long calls: 0.62 delta, standard 100 multiplier."""
    return PositionGreeksInput(
        ticker="NVDA",
        instrument="LONG_CALL",
        quantity=qty,
        multiplier=100,
        spot=spot,
        delta=0.62,
        gamma=0.03,
        theta_per_day=-0.05,
        vega=0.11,
    )


def test_mixed_stock_and_option_book_hand_computed():
    # §16 "Equivalent Shares" reference case:
    #   stock: 100 sh * 1 * 1.0            = 100 delta-shares
    #   calls:   4 ct * 100 * 0.62         = 248 delta-shares
    #   net_delta_shares                    = 348
    result = aggregate_greeks([stock(), call()])
    assert result.net_delta_shares == pytest.approx(348.0)

    # delta_adjusted_notional = Σ qty*mult*delta*spot
    #   stock: 100 * 1 * 1.0 * 50   =  5,000
    #   calls: 4 * 100 * 0.62 * 50  = 12,400
    #   total                       = 17,400
    assert result.delta_adjusted_notional == pytest.approx(17_400.0)

    # net_gamma = 100*1*0.0 + 4*100*0.03 = 12.0
    assert result.net_gamma == pytest.approx(12.0)

    # net_theta_per_day = 100*1*0.0 + 4*100*(-0.05) = -20.0 $/day
    assert result.net_theta_per_day == pytest.approx(-20.0)

    # net_vega = 100*1*0.0 + 4*100*0.11 = 44.0 $ per IV point
    assert result.net_vega == pytest.approx(44.0)


def test_per_position_echoes_each_contribution_in_order():
    result = aggregate_greeks([stock(), call()])
    assert len(result.per_position) == 2

    s, c = result.per_position
    assert s.ticker == "NVDA" and s.instrument == "LONG_STOCK"
    assert s.quantity == 100 and s.multiplier == 1
    assert s.delta_shares == pytest.approx(100.0)  # 100 * 1 * 1.0
    assert s.delta_notional == pytest.approx(5_000.0)  # 100 * 50
    assert s.gamma == 0.0 and s.theta_per_day == 0.0 and s.vega == 0.0

    assert c.instrument == "LONG_CALL"
    assert c.delta_shares == pytest.approx(248.0)  # 4 * 100 * 0.62
    assert c.delta_notional == pytest.approx(12_400.0)  # 248 * 50
    assert c.gamma == pytest.approx(12.0)  # 4 * 100 * 0.03
    assert c.theta_per_day == pytest.approx(-20.0)  # 4 * 100 * -0.05
    assert c.vega == pytest.approx(44.0)  # 4 * 100 * 0.11

    # The totals are exactly the sum of the echoed contributions.
    assert result.net_delta_shares == pytest.approx(
        s.delta_shares + c.delta_shares
    )
    assert result.delta_adjusted_notional == pytest.approx(
        s.delta_notional + c.delta_notional
    )


def test_long_put_contributes_negative_delta():
    # 2 puts * 100 * (-0.40) = -80 delta-shares; theta/vega still add.
    put = PositionGreeksInput(
        ticker="SPY",
        instrument="LONG_PUT",
        quantity=2,
        multiplier=100,
        spot=400.0,
        delta=-0.40,
        gamma=0.01,
        theta_per_day=-0.08,
        vega=0.20,
    )
    result = aggregate_greeks([put])
    assert result.net_delta_shares == pytest.approx(-80.0)
    # -80 delta-shares * $400 spot = -$32,000 delta-adjusted notional.
    assert result.delta_adjusted_notional == pytest.approx(-32_000.0)
    assert result.net_theta_per_day == pytest.approx(-16.0)  # 200 * -0.08
    assert result.net_vega == pytest.approx(40.0)  # 200 * 0.20


def test_puts_offset_stock_delta():
    # 100 sh stock (+100) + 2 puts * 100 * -0.40 (-80) = net +20.
    put = PositionGreeksInput(
        ticker="NVDA",
        instrument="LONG_PUT",
        quantity=2,
        multiplier=100,
        spot=50.0,
        delta=-0.40,
        gamma=0.02,
        theta_per_day=-0.03,
        vega=0.15,
    )
    result = aggregate_greeks([stock(), put])
    assert result.net_delta_shares == pytest.approx(20.0)
    # 5,000 (stock) + (-80 * 50) = 5,000 - 4,000 = 1,000.
    assert result.delta_adjusted_notional == pytest.approx(1_000.0)


def test_empty_book_aggregates_to_zeros():
    result = aggregate_greeks([])
    assert result.net_delta_shares == 0.0
    assert result.delta_adjusted_notional == 0.0
    assert result.net_gamma == 0.0
    assert result.net_theta_per_day == 0.0
    assert result.net_vega == 0.0
    assert result.per_position == ()


def test_position_contribution_matches_single_input_aggregate():
    c = position_contribution(call())
    agg = aggregate_greeks([call()])
    assert agg.net_delta_shares == pytest.approx(c.delta_shares)
    assert agg.per_position == (c,)


def test_invalid_instrument_and_multiplier_rejected():
    with pytest.raises(ValueError):
        PositionGreeksInput(
            ticker="X",
            instrument="SHORT_CALL",  # not in v0 vocabulary
            quantity=1,
            multiplier=100,
            spot=10.0,
            delta=0.5,
            gamma=0.0,
            theta_per_day=0.0,
            vega=0.0,
        )
    with pytest.raises(ValueError):
        PositionGreeksInput(
            ticker="X",
            instrument="LONG_STOCK",
            quantity=1,
            multiplier=0,
            spot=10.0,
            delta=1.0,
            gamma=0.0,
            theta_per_day=0.0,
            vega=0.0,
        )
