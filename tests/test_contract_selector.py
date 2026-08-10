"""Contract Selector v0 tests (development plan §9, §34).

Hand-built micro-chains exercise each §9.1 filter in isolation (the exact
reason string, with real numbers, is asserted), and the §9.2 ranking
arithmetic is hand-computed in comments for a 3-candidate case so the score
composition is auditable. Every contract must come back with a verdict, in
input order — the UI needs All / Eligible / Recommended views (§34).
"""
from datetime import date

import pytest

from libs.trading_core.contracts import (
    ContractQuote,
    ScoredContract,
    SelectorParams,
    select_contracts,
)

EXPIRY = date(2026, 9, 24)


def cq(**overrides) -> ContractQuote:
    """A call passing every default filter; override fields to break one."""
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
        theta=-0.05,  # |theta|/mid = 0.01 <= 0.02 default cap
        vega=0.12,
    )
    base.update(overrides)
    return ContractQuote(**base)


# ---------------------------------------------------------------------------
# Side gate (§5 long-only): BULL -> calls, BEAR -> puts
# ---------------------------------------------------------------------------


def test_bull_selects_only_calls():
    call = cq(right="C", delta=0.55)
    put = cq(right="P", delta=-0.55)
    out = select_contracts([call, put], "BULL")
    assert out[0].eligible is True
    assert out[1].eligible is False
    assert "wrong side for BULL direction" in out[1].fail_reasons


def test_bear_selects_only_puts():
    call = cq(right="C", delta=0.55)
    put = cq(right="P", delta=-0.55)
    out = select_contracts([call, put], "BEAR")
    assert out[0].eligible is False
    assert "wrong side for BEAR direction" in out[0].fail_reasons
    assert out[1].eligible is True


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        select_contracts([cq()], "SIDEWAYS")


# ---------------------------------------------------------------------------
# §9.1 filters — each failing individually, exact reason string checked
# ---------------------------------------------------------------------------


def test_dte_below_window():
    out = select_contracts([cq(dte=20)], "BULL")
    assert out[0].eligible is False
    assert out[0].fail_reasons == ["DTE 20 outside [30, 90]"]


def test_dte_above_window():
    out = select_contracts([cq(dte=120)], "BULL")
    assert out[0].fail_reasons == ["DTE 120 outside [30, 90]"]


def test_abs_delta_below_window():
    out = select_contracts([cq(delta=0.30)], "BULL")
    assert out[0].fail_reasons == ["|delta| 0.30 outside [0.40, 0.75]"]


def test_abs_delta_above_window():
    out = select_contracts([cq(delta=0.80)], "BULL")
    assert out[0].fail_reasons == ["|delta| 0.80 outside [0.40, 0.75]"]


def test_open_interest_floor():
    out = select_contracts([cq(open_interest=50)], "BULL")
    assert out[0].fail_reasons == ["open interest 50 < 100"]


def test_volume_floor():
    out = select_contracts([cq(volume=5)], "BULL")
    assert out[0].fail_reasons == ["volume 5 < 10"]


def test_spread_cap():
    out = select_contracts([cq(spread_pct=0.20)], "BULL")
    assert out[0].fail_reasons == ["spread_pct 0.2000 > 0.1000"]


def test_theta_burden_cap():
    # |theta| / mid = 0.15 / 5.0 = 0.03/day > 0.02 cap.
    out = select_contracts([cq(theta=-0.15)], "BULL")
    assert out[0].fail_reasons == ["theta burden 0.0300/day > 0.0200"]


def test_mid_must_be_positive():
    # mid <= 0 fails on its own reason and skips the theta-burden division.
    out = select_contracts(
        [cq(bid=0.0, ask=0.0, mid=0.0, spread_pct=0.0)], "BULL"
    )
    assert out[0].eligible is False
    assert out[0].fail_reasons == ["mid 0.0000 <= 0"]


def test_multiple_failures_all_reported():
    out = select_contracts([cq(dte=20, volume=5, open_interest=50)], "BULL")
    assert out[0].fail_reasons == [
        "DTE 20 outside [30, 90]",
        "open interest 50 < 100",
        "volume 5 < 10",
    ]


def test_boundaries_are_inclusive():
    for c in (
        cq(dte=30), cq(dte=90),
        cq(delta=0.40), cq(delta=0.75),
        cq(open_interest=100), cq(volume=10),
        cq(spread_pct=0.10),
        cq(theta=-0.10),  # burden exactly 0.02
    ):
        out = select_contracts([c], "BULL")
        assert out[0].eligible is True, out[0].fail_reasons


def test_ineligible_has_honest_nulls():
    out = select_contracts([cq(dte=20)], "BULL")
    assert out[0].score is None
    assert out[0].rank is None
    assert out[0].components is None


# ---------------------------------------------------------------------------
# §9.2 ranking — hand-computed 3-candidate case
# ---------------------------------------------------------------------------

# Defaults: delta_mid = (0.40 + 0.75)/2 = 0.575, half_width = 0.175.
#
# A: spread 0.02, theta -0.05/mid 5 (burden ratio 0.5), delta 0.55
#    liquidity = 1 - 0.02/0.10             = 0.8
#    theta_burden = (0.05/5)/0.02          = 0.5
#    delta_fit = 1 - |0.55-0.575|/0.175    = 6/7  ~= 0.857142857
#    score = 0.8 - 0.5 + 6/7               ~= 1.157142857
# B: spread 0.05, theta -0.02 (burden 0.2), delta 0.575
#    liquidity = 0.5, theta_burden = 0.2, delta_fit = 1.0
#    score = 0.5 - 0.2 + 1.0                = 1.3
# C: spread 0.00, theta -0.10 (burden 1.0), delta 0.70
#    liquidity = 1.0, theta_burden = 1.0
#    delta_fit = 1 - |0.70-0.575|/0.175    = 2/7  ~= 0.285714286
#    score = 1.0 - 1.0 + 2/7               ~= 0.285714286
# Ranks by score: B (1.3) > A (~1.157) > C (~0.286).

def rank_chain() -> list[ContractQuote]:
    a = cq(spread_pct=0.02, theta=-0.05, delta=0.55, strike=95.0)
    b = cq(spread_pct=0.05, theta=-0.02, delta=0.575, strike=100.0)
    c = cq(spread_pct=0.0, bid=5.0, ask=5.0, theta=-0.10, delta=0.70,
           strike=105.0)
    return [a, b, c]


def test_ranking_arithmetic_hand_computed():
    out = select_contracts(rank_chain(), "BULL")
    a, b, c = out
    assert all(s.eligible for s in out)

    assert a.components["liquidity"] == pytest.approx(0.8)
    assert a.components["theta_burden"] == pytest.approx(0.5)
    assert a.components["delta_fit"] == pytest.approx(6.0 / 7.0)
    assert a.score == pytest.approx(0.8 - 0.5 + 6.0 / 7.0)

    assert b.components == pytest.approx(
        {"liquidity": 0.5, "theta_burden": 0.2, "delta_fit": 1.0}
    )
    assert b.score == pytest.approx(1.3)

    assert c.components["liquidity"] == pytest.approx(1.0)
    assert c.components["theta_burden"] == pytest.approx(1.0)
    assert c.components["delta_fit"] == pytest.approx(2.0 / 7.0)
    assert c.score == pytest.approx(2.0 / 7.0)

    assert (a.rank, b.rank, c.rank) == (2, 1, 3)


def test_weights_change_the_winner():
    # Boosting w_liquidity to 10 promotes C (tightest spread):
    #   A: 10*0.8 - 0.5 + 6/7 ~= 8.357;  B: 10*0.5 - 0.2 + 1 = 5.8;
    #   C: 10*1.0 - 1.0 + 2/7 ~= 9.286.
    params = SelectorParams(w_liquidity=10.0)
    out = select_contracts(rank_chain(), "BULL", params)
    a, b, c = out
    assert c.rank == 1
    assert c.score == pytest.approx(10.0 - 1.0 + 2.0 / 7.0)
    assert (a.rank, b.rank) == (2, 3)


def test_top_n_limits_ranks_but_keeps_scores():
    params = SelectorParams(top_n=1)
    out = select_contracts(rank_chain(), "BULL", params)
    a, b, c = out
    assert b.rank == 1
    assert a.rank is None and c.rank is None
    assert a.eligible and c.eligible
    assert a.score is not None and c.score is not None


# ---------------------------------------------------------------------------
# All contracts returned with verdicts (§34), edge chains, determinism
# ---------------------------------------------------------------------------


def test_every_contract_returned_in_input_order():
    chain = rank_chain() + [cq(dte=20), cq(right="P", delta=-0.55)]
    out = select_contracts(chain, "BULL")
    assert len(out) == len(chain)
    for scored, original in zip(out, chain):
        assert isinstance(scored, ScoredContract)
        assert scored.contract is original
    assert [s.eligible for s in out] == [True, True, True, False, False]


def test_empty_chain():
    assert select_contracts([], "BULL") == []


def test_all_ineligible_chain_has_no_ranks():
    chain = [cq(dte=20), cq(volume=5), cq(right="P", delta=-0.55)]
    out = select_contracts(chain, "BULL")
    assert len(out) == 3
    assert all(not s.eligible for s in out)
    assert all(s.rank is None and s.score is None for s in out)


def test_determinism():
    chain = rank_chain() + [cq(dte=20), cq(open_interest=0)]
    first = select_contracts(chain, "BULL")
    for _ in range(3):
        assert select_contracts(chain, "BULL") == first
