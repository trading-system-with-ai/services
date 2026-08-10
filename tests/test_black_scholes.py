"""Black-Scholes pricing library tests (development plan §9).

Anchors are textbook values (Hull): S=100, K=100, T=1, r=0.05, q=0,
iv=0.20 -> call 10.4506, put 5.5735, call delta N(0.35) ~= 0.6368.
Put-call parity C - P = S*e^{-qT} - K*e^{-rT} is checked across a parameter
grid at 1e-9 — it must hold exactly in the model, not just approximately.
"""
import math

import pytest

from libs.trading_core.options import Greeks, bs_greeks, bs_price

# Textbook anchor parameters: S=100, K=100, T=1, r=0.05, q=0, iv=0.20.
ANCHOR = dict(spot=100.0, strike=100.0, t_years=1.0, iv=0.20, r=0.05, q=0.0)


# ---------------------------------------------------------------------------
# Textbook anchors
# ---------------------------------------------------------------------------


def test_anchor_call_price():
    assert bs_price(right="C", **ANCHOR) == pytest.approx(10.4506, abs=1e-3)


def test_anchor_put_price():
    assert bs_price(right="P", **ANCHOR) == pytest.approx(5.5735, abs=1e-3)


def test_anchor_call_delta():
    # d1 = (0 + (0.05 + 0.02)) / 0.20 = 0.35; N(0.35) ~= 0.6368.
    g = bs_greeks(right="C", **ANCHOR)
    assert g.delta == pytest.approx(0.6368, abs=1e-3)


def test_anchor_put_delta_is_call_delta_minus_one():
    # With q=0: delta_put = delta_call - 1 (both from N(d1)).
    call = bs_greeks(right="C", **ANCHOR)
    put = bs_greeks(right="P", **ANCHOR)
    assert put.delta == pytest.approx(call.delta - 1.0, abs=1e-12)
    assert put.delta < 0.0 < call.delta  # signed convention


def test_greeks_price_matches_bs_price():
    for right in ("C", "P"):
        assert bs_greeks(right=right, **ANCHOR).price == pytest.approx(
            bs_price(right=right, **ANCHOR), abs=1e-12
        )


def test_anchor_theta_negative_per_calendar_day():
    # Long ATM options decay; theta is per calendar day (annual / 365), so
    # its magnitude must be small — an ATM 1y call loses cents/day, not $6+.
    for right in ("C", "P"):
        g = bs_greeks(right=right, **ANCHOR)
        assert g.theta < 0.0
        assert abs(g.theta) < 0.1


def test_anchor_vega_per_iv_point():
    # Annualized ATM vega ~= S * phi(d1) * sqrt(T) ~= 37.52; per 1 IV point
    # (0.01 vol) it is ~0.3752. Cross-check with a finite difference.
    g = bs_greeks(right="C", **ANCHOR)
    assert g.vega == pytest.approx(0.3752, abs=1e-3)
    bumped = dict(ANCHOR, iv=ANCHOR["iv"] + 0.01)
    fd = bs_price(right="C", **bumped) - bs_price(right="C", **ANCHOR)
    assert g.vega == pytest.approx(fd, abs=2e-3)


# ---------------------------------------------------------------------------
# Put-call parity across a parameter grid (tolerance 1e-9)
# ---------------------------------------------------------------------------


def test_put_call_parity_grid():
    for spot in (80.0, 100.0, 123.45):
        for strike in (90.0, 100.0, 110.0):
            for t in (0.05, 0.5, 1.0, 2.0):
                for iv in (0.10, 0.35):
                    for r in (0.0, 0.04):
                        for q in (0.0, 0.02):
                            c = bs_price(spot, strike, t, iv, "C", r=r, q=q)
                            p = bs_price(spot, strike, t, iv, "P", r=r, q=q)
                            parity = spot * math.exp(-q * t) - strike * math.exp(
                                -r * t
                            )
                            assert c - p == pytest.approx(parity, abs=1e-9)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_call_price_increases_in_iv():
    ivs = [0.05, 0.10, 0.20, 0.40, 0.80]
    prices = [bs_price(100.0, 100.0, 0.5, iv, "C", r=0.04, q=0.0) for iv in ivs]
    assert prices == sorted(prices)
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_call_price_increases_in_t():
    ts = [0.05, 0.25, 0.5, 1.0, 2.0]
    prices = [bs_price(100.0, 105.0, t, 0.20, "C", r=0.04, q=0.0) for t in ts]
    assert all(b > a for a, b in zip(prices, prices[1:]))


# ---------------------------------------------------------------------------
# Gamma / vega parity between rights
# ---------------------------------------------------------------------------


def test_gamma_and_vega_identical_for_call_and_put():
    for spot, strike, t, iv, r, q in [
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.0),
        (95.0, 110.0, 0.3, 0.45, 0.04, 0.01),
        (150.0, 120.0, 1.7, 0.15, 0.0, 0.03),
    ]:
        c = bs_greeks(spot, strike, t, iv, "C", r=r, q=q)
        p = bs_greeks(spot, strike, t, iv, "P", r=r, q=q)
        assert c.gamma == pytest.approx(p.gamma, abs=1e-12)
        assert c.vega == pytest.approx(p.vega, abs=1e-12)
        assert c.gamma > 0.0
        assert c.vega > 0.0


# ---------------------------------------------------------------------------
# Expiry edge cases (t_years <= 0)
# ---------------------------------------------------------------------------


def test_expiry_itm_call_intrinsic_and_delta_one():
    assert bs_price(110.0, 100.0, 0.0, 0.20, "C") == 10.0
    g = bs_greeks(110.0, 100.0, 0.0, 0.20, "C")
    assert g == Greeks(price=10.0, delta=1.0, gamma=0.0, theta=0.0, vega=0.0)


def test_expiry_otm_call_zero():
    assert bs_price(90.0, 100.0, 0.0, 0.20, "C") == 0.0
    g = bs_greeks(90.0, 100.0, 0.0, 0.20, "C")
    assert g == Greeks(price=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0)


def test_expiry_itm_put_intrinsic_and_delta_minus_one():
    assert bs_price(90.0, 100.0, 0.0, 0.20, "P") == 10.0
    g = bs_greeks(90.0, 100.0, 0.0, 0.20, "P")
    assert g == Greeks(price=10.0, delta=-1.0, gamma=0.0, theta=0.0, vega=0.0)


def test_expiry_otm_put_zero():
    g = bs_greeks(110.0, 100.0, 0.0, 0.20, "P")
    assert g == Greeks(price=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0)


def test_expiry_atm_reports_delta_zero():
    # Documented convention: exactly at-the-money at expiry -> delta 0.
    assert bs_greeks(100.0, 100.0, 0.0, 0.20, "C").delta == 0.0
    assert bs_greeks(100.0, 100.0, 0.0, 0.20, "P").delta == 0.0


def test_negative_t_treated_as_expired():
    assert bs_price(110.0, 100.0, -0.5, 0.20, "C") == 10.0
    assert bs_greeks(90.0, 100.0, -1.0, 0.20, "P").delta == -1.0


# ---------------------------------------------------------------------------
# ValueError cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(spot=0.0), dict(spot=-100.0),
    dict(strike=0.0), dict(strike=-100.0),
    dict(iv=0.0), dict(iv=-0.20),
])
def test_invalid_inputs_raise(kwargs):
    args = dict(ANCHOR, **kwargs)
    with pytest.raises(ValueError):
        bs_price(right="C", **args)
    with pytest.raises(ValueError):
        bs_greeks(right="P", **args)


def test_invalid_inputs_raise_even_at_expiry():
    # Honest errors, never a silent number: bad iv/spot/strike raise even
    # when t_years <= 0 would otherwise short-circuit to intrinsic.
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.0, 0.0, "C")
    with pytest.raises(ValueError):
        bs_greeks(-1.0, 100.0, 0.0, 0.20, "P")


def test_invalid_right_raises():
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.20, "X")
    with pytest.raises(ValueError):
        bs_greeks(100.0, 100.0, 1.0, 0.20, "call")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    for right in ("C", "P"):
        first_p = bs_price(right=right, **ANCHOR)
        first_g = bs_greeks(right=right, **ANCHOR)
        for _ in range(3):
            assert bs_price(right=right, **ANCHOR) == first_p
            assert bs_greeks(right=right, **ANCHOR) == first_g
