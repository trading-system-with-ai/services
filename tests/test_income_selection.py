"""Income-leg selection tests (Phase 2): covered-call / CSP contract choice
by the mechanical standards (30-45 DTE, |delta| 0.15-0.35 band, OTM only,
sellable liquidity)."""
from datetime import date

import pytest

from libs.trading_core.contracts import ContractQuote
from libs.trading_core.strategies import (
    IncomeParams,
    select_cash_secured_put,
    select_covered_call,
)

EXPIRY = date(2026, 10, 2)
SPOT = 100.0


def cq(**overrides) -> ContractQuote:
    base = dict(
        expiry=EXPIRY, dte=38, strike=108.0, right="C",
        bid=1.4, ask=1.6, mid=1.5, spread_pct=0.13, last=1.5,
        volume=80, open_interest=300, iv=0.30, delta=0.25,
        gamma=0.02, theta=-0.04, vega=0.10,
    )
    base.update(overrides)
    return ContractQuote(**base)


def test_covered_call_picks_nearest_to_target_delta_otm():
    chain = [
        cq(strike=104.0, delta=0.40),           # above the band -> rejected
        cq(strike=108.0, delta=0.26),           # nearest to 0.25 target
        cq(strike=112.0, delta=0.17),
        cq(strike=98.0, delta=0.55),            # ITM -> rejected
    ]
    sel = select_covered_call(chain, SPOT)
    assert sel.contract is not None
    assert sel.contract.strike == 108.0
    assert "annualized" in sel.rationale[0]


def test_csp_mirror_and_no_bid_rejected():
    chain = [
        cq(right="P", strike=92.0, delta=-0.26),
        cq(right="P", strike=88.0, delta=-0.16),
        cq(right="P", strike=95.0, delta=-0.30, bid=0.0, mid=0.4),  # no bid
        cq(right="P", strike=104.0, delta=-0.60),  # ITM put -> rejected
    ]
    sel = select_cash_secured_put(chain, SPOT)
    assert sel.contract is not None
    assert sel.contract.strike == 92.0


def test_named_blockers_when_nothing_qualifies():
    chain = [
        cq(dte=10),                       # DTE window
        cq(strike=109.0, delta=None),     # no delta
        cq(strike=110.0, open_interest=5),
        cq(strike=111.0, price_basis="day_close"),
    ]
    sel = select_covered_call(chain, SPOT)
    assert sel.contract is None
    msg = sel.fail_reasons[0]
    for needle in ("DTE outside window", "delta not provided",
                   "open interest below floor", "no real NBBO quote"):
        assert needle in msg, msg


def test_params_validation():
    with pytest.raises(ValueError, match="dte_min"):
        IncomeParams(dte_min=0)
    with pytest.raises(ValueError, match="abs_delta"):
        IncomeParams(abs_delta_min=0.4, abs_delta_max=0.2)
    with pytest.raises(ValueError, match="spot"):
        select_covered_call([], 0.0)
