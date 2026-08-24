"""Implied-volatility solver tests (Phase D design §8.1 / §8.7).

Hand-computed anchor (r = q = 0, so the discount factors vanish and the
formula collapses to a number that can be checked with erf alone):

    S = K = 100, T = 1, sigma = 0.20, right = C

    d1 = (ln(1) + (0 - 0 + 0.5*0.04)*1) / (0.20*1) = 0.02/0.20 =  0.10
    d2 = d1 - 0.20                                                = -0.10
    N(0.10)  = 0.5*(1 + erf(0.10/sqrt(2))) = 0.5398278372770290
    N(-0.10) = 1 - N(0.10)                 = 0.4601721627229710
    price    = 100*N(d1) - 100*N(d2) = 100*(0.5398278... - 0.4601721...)
             = 100*0.0796556745540580 = 7.96556745540580

    -> implied_vol(7.96556745540580, ...) must return 0.20.

The round-trip grid below is the real acceptance test: price a contract at
a known sigma, invert the price, and require |sigma_out - sigma_in| < 1e-6
across strikes, vols, tenors and both rights.
"""
from __future__ import annotations

import math

import pytest

from libs.trading_core.options.bs import bs_greeks, bs_price
from libs.trading_core.options.iv import (
    DEFAULT_HI,
    METHOD_BISECTION,
    IVResult,
    implied_vol,
)

# --- the hand-computed anchor ---------------------------------------------
ANCHOR_PRICE = 7.96556745540580


def test_hand_computed_atm_call_round_trip() -> None:
    """The docstring's 7.9655674554058 inverts back to sigma = 0.20."""
    price = bs_price(100.0, 100.0, 1.0, 0.20, "C", 0.0, 0.0)
    assert price == pytest.approx(ANCHOR_PRICE, abs=1e-10)

    res = implied_vol(price, 100.0, 100.0, 1.0, "C", r=0.0, q=0.0)
    assert res.converged is True
    assert res.reason is None
    assert res.method == METHOD_BISECTION
    assert res.iv == pytest.approx(0.20, abs=1e-6)
    assert res.iterations > 0


#: Below this vega (price change per 1 IV point) a mark carries no
#: recoverable volatility information: the whole [lo, hi] bracket prices
#: inside a few float ULPs, so bisection cannot separate 0.05 from 0.06.
#: This is a property of the PRICE, not of the solver, and the grid test
#: asserts it explicitly rather than hiding those points.
VEGA_INFORMATION_FLOOR = 1e-8


def test_round_trip_grid_within_1e_6() -> None:
    """|sigma_out - sigma_in| < 1e-6 across a strike/vol/tenor/right grid.

    Two documented exceptions, both asserted rather than skipped:

    - the price is at/below the model floor at ``sigma = lo`` (a deep-ITM
      option whose value is all intrinsic) — the solver must return
      ``None``, not a clamped ``lo``;
    - the contract's vega is below :data:`VEGA_INFORMATION_FLOOR` (a
      deep-OTM wing worth ~1e-15) — the mark contains no vol information at
      all, so any sigma in a wide band reproduces it to float precision.
      The test then requires only that the solved sigma REPRODUCES THE
      PRICE, which is the strongest true statement available.
    """
    spot = 100.0
    worst = 0.0
    n_exact = 0
    n_uninformative = 0
    for strike in (60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0):
        for sigma in (0.05, 0.10, 0.20, 0.35, 0.60, 1.20):
            for t_years in (7 / 365, 30 / 365, 0.5, 1.0, 2.0):
                for right in ("C", "P"):
                    price = bs_price(spot, strike, t_years, sigma, right)
                    res = implied_vol(price, spot, strike, t_years, right)
                    if res.iv is None:
                        # Only legitimate at/below the sigma=lo model floor.
                        floor = bs_price(spot, strike, t_years, 1e-4, right)
                        assert price <= floor, (
                            f"unsolvable but above floor: K={strike} "
                            f"sigma={sigma} T={t_years} {right}: {res.reason}"
                        )
                        continue
                    vega = bs_greeks(spot, strike, t_years, sigma, right).vega
                    if vega < VEGA_INFORMATION_FLOOR:
                        # No vol information in the price: require only that
                        # the solved sigma reprices the contract.
                        n_uninformative += 1
                        repriced = bs_price(spot, strike, t_years, res.iv, right)
                        assert abs(repriced - price) <= 1e-12
                        continue
                    err = abs(res.iv - sigma)
                    worst = max(worst, err)
                    n_exact += 1
                    assert err < 1e-6, (
                        f"K={strike} sigma={sigma} T={t_years} {right}: "
                        f"solved {res.iv}, err {err}"
                    )
    assert n_exact > 300, f"grid degenerated to {n_exact} informative points"
    assert worst < 1e-6
    # The uninformative corner is real but small — if it ever swallows the
    # grid, this test has stopped testing the solver.
    assert n_uninformative < n_exact / 10


def test_solver_is_deterministic() -> None:
    """Same inputs, same answer — bit-for-bit (house rule: deterministic)."""
    price = bs_price(100.0, 105.0, 0.25, 0.33, "P")
    a = implied_vol(price, 100.0, 105.0, 0.25, "P")
    b = implied_vol(price, 100.0, 105.0, 0.25, "P")
    assert a == b


# --- guards: honest nulls, never a fabricated number ----------------------


def test_expired_option_returns_none_with_reason() -> None:
    res = implied_vol(5.0, 100.0, 95.0, 0.0, "C")
    assert res.iv is None
    assert res.converged is False
    assert res.iterations == 0
    assert "expired" in res.reason and "t_years=0" in res.reason


def test_negative_t_years_returns_none() -> None:
    res = implied_vol(5.0, 100.0, 95.0, -0.5, "C")
    assert res.iv is None
    assert "expired" in res.reason


def test_price_at_or_below_intrinsic_returns_none() -> None:
    """A call marked at exactly intrinsic has no positive implied vol."""
    intrinsic = 100.0 - 90.0  # 10.0
    res = implied_vol(intrinsic, 100.0, 90.0, 0.5, "C", r=0.0, q=0.0)
    assert res.iv is None
    assert "model floor" in res.reason
    assert "10.0" in res.reason or "10.000000" in res.reason


def test_price_below_intrinsic_returns_none() -> None:
    res = implied_vol(3.0, 100.0, 90.0, 0.5, "C", r=0.0, q=0.0)
    assert res.iv is None
    assert "model floor" in res.reason


def test_zero_and_negative_price_return_none() -> None:
    for bad in (0.0, -1.5):
        res = implied_vol(bad, 100.0, 100.0, 0.5, "C")
        assert res.iv is None
        assert "not a tradeable mark" in res.reason


def test_price_above_the_sigma_ceiling_returns_none() -> None:
    """A price beyond bs_price(sigma=5.0) is data, not volatility."""
    ceiling = bs_price(100.0, 100.0, 1.0, DEFAULT_HI, "C")
    res = implied_vol(ceiling + 1.0, 100.0, 100.0, 1.0, "C")
    assert res.iv is None
    assert "ceiling" in res.reason
    assert "5" in res.reason
    # Exactly AT the ceiling is also rejected (>=): reporting hi would be
    # a clamped, fabricated number.
    at = implied_vol(ceiling, 100.0, 100.0, 1.0, "C")
    assert at.iv is None


def test_custom_bracket_is_honoured() -> None:
    """A caller-narrowed bracket rejects a vol outside it, honestly."""
    price = bs_price(100.0, 100.0, 1.0, 0.80, "C")
    res = implied_vol(price, 100.0, 100.0, 1.0, "C", hi=0.50)
    assert res.iv is None
    assert "ceiling" in res.reason
    # Widen it back and the same price solves.
    ok = implied_vol(price, 100.0, 100.0, 1.0, "C", hi=2.0)
    assert ok.iv == pytest.approx(0.80, abs=1e-6)


def test_tolerance_controls_accuracy_and_non_convergence_is_labelled() -> None:
    """A starved iteration budget reports the bracket width, not silence."""
    price = bs_price(100.0, 100.0, 1.0, 0.20, "C")
    res = implied_vol(price, 100.0, 100.0, 1.0, "C", max_iter=3)
    assert res.converged is False
    assert res.iterations == 3
    assert res.iv is not None  # still a bracketed estimate...
    assert "bracket width" in res.reason  # ...but labelled as unconverged


def test_looser_tolerance_uses_fewer_iterations() -> None:
    price = bs_price(100.0, 100.0, 1.0, 0.20, "C")
    tight = implied_vol(price, 100.0, 100.0, 1.0, "C", tol=1e-10)
    loose = implied_vol(price, 100.0, 100.0, 1.0, "C", tol=1e-4)
    assert loose.iterations < tight.iterations
    assert loose.converged is True
    assert abs(loose.iv - 0.20) < 1e-4


# --- malformed input still raises (contract §1) ---------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0.0},
        {"spot": -10.0},
        {"strike": 0.0},
        {"strike": -5.0},
        {"right": "X"},
        {"price": math.nan},
        {"lo": 0.0},
        {"lo": 1.0, "hi": 0.5},
        {"tol": 0.0},
        {"tol": -1e-6},
        {"max_iter": 0},
        {"max_iter": True},
    ],
)
def test_malformed_input_raises_value_error(kwargs: dict) -> None:
    base = dict(
        price=5.0, spot=100.0, strike=100.0, t_years=0.5, right="C"
    )
    base.update(kwargs)
    positional = (
        base.pop("price"),
        base.pop("spot"),
        base.pop("strike"),
        base.pop("t_years"),
        base.pop("right"),
    )
    with pytest.raises(ValueError):
        implied_vol(*positional, **base)


def test_result_is_frozen_and_carries_the_provenance_label() -> None:
    res = IVResult(iv=0.2, iterations=5, converged=True)
    assert res.method == METHOD_BISECTION  # INTERNALLY CALCULATED, never vendor
    with pytest.raises(Exception):
        res.iv = 0.3  # type: ignore[misc]


def test_dividend_and_rate_are_honoured() -> None:
    """Non-zero r/q must be passed through — solving with the wrong ones
    would produce a different (and wrong) vol."""
    price = bs_price(100.0, 100.0, 1.0, 0.25, "C", r=0.05, q=0.02)
    right = implied_vol(price, 100.0, 100.0, 1.0, "C", r=0.05, q=0.02)
    assert right.iv == pytest.approx(0.25, abs=1e-6)
    wrong = implied_vol(price, 100.0, 100.0, 1.0, "C", r=0.0, q=0.0)
    assert abs(wrong.iv - 0.25) > 1e-3
