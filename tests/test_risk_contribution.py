"""Risk contribution tests (Phase B contract §2.5; invariants §3.3, §3.4).

Canonical fixture — 3 positions x 8 days (USD/day, gain-positive):

    t :        0     1     2     3     4     5     6     7
    A :      2.0  -3.0   1.0   4.0  -6.0   0.0   2.0  -1.0    sum = -1.0
    B :     -1.0   2.0   0.0  -2.0   3.0   1.0  -1.0   1.0    sum =  3.0
    C :      0.0  -1.0   2.0   1.0  -4.0   2.0   0.0  -2.0    sum = -2.0
    P = A+B+C:
             1.0  -2.0   3.0   3.0  -7.0   3.0   1.0  -2.0    sum =  0.0

  mean_P = 0/8 = 0  ->  deviations ARE the values themselves.
  Sum P^2 = 1+4+9+9+49+9+1+4 = 86
  var_P (ddof=1) = 86/7 ;  sigma_P = sqrt(86/7) = sqrt(12.285714285714286)
                        = 3.5050983275...

  ES tail (contract §2.3): losses L = -P =
            -1.0   2.0  -3.0  -3.0   7.0  -3.0  -1.0   2.0
  k = ceil(8 * (1 - 0.75)) = ceil(2.0) = 2
  Two largest losses: 7.0 (t=4) and 2.0 (t=1 or t=7 -> tie broken by DATE
  ORDER, so t=1 wins).  Tail set T = {4, 1} -> ES = (7.0 + 2.0)/2 = 4.5
  Per position on T:   A: (-(-6) + -(-3))/2 = (6 + 3)/2 = 4.5
                       B: (-3 + -2)/2 = -2.5
                       C: (4 + 1)/2 = 2.5
  Sum = 4.5 - 2.5 + 2.5 = 4.5 == ES  (contract §3.3, exactly)
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.contribution import (
    METHOD_ES,
    METHOD_VOL,
    ContributionParams,
    es_contributions,
    incremental_es,
    marginal_es,
    tail_size,
    volatility_contributions,
)

A = [2.0, -3.0, 1.0, 4.0, -6.0, 0.0, 2.0, -1.0]
B = [-1.0, 2.0, 0.0, -2.0, 3.0, 1.0, -1.0, 1.0]
C = [0.0, -1.0, 2.0, 1.0, -4.0, 2.0, 0.0, -2.0]
POSITIONS = {"A": A, "B": B, "C": C}
P = [a + b + c for a, b, c in zip(A, B, C)]

# n=8 is far below the production min_obs (60/250); these params let the
# ARITHMETIC be tested on a hand-checkable sample. min_obs itself is
# exercised separately below.
SMALL = ContributionParams(min_obs_vol=5, min_obs_95=5, min_obs_99=5, degraded_multiple=1.0)

SIGMA_P = math.sqrt(86.0 / 7.0)  # mean_P = 0, so Sum(P - mean)^2 = Sum P^2 = 86


def test_fixture_arithmetic_is_what_the_docstring_claims() -> None:
    assert P == [1.0, -2.0, 3.0, 3.0, -7.0, 3.0, 1.0, -2.0]
    assert math.fsum(P) == 0.0
    assert math.fsum(v * v for v in P) == 86.0


# ---------------------------------------------------------------------------
# Volatility contributions
# ---------------------------------------------------------------------------


def test_volatility_total_is_sigma_p_and_rcs_sum_to_it() -> None:
    r = volatility_contributions(POSITIONS, params=SMALL)
    assert r.method == METHOD_VOL
    assert r.confidence is None and r.tail_size is None
    assert r.sample_size == 8
    assert r.total == pytest.approx(SIGMA_P, rel=1e-15)
    assert r.total == pytest.approx(3.5050983275, abs=1e-9)

    # Contract §3.3: Sum_i RC_i == sigma_p within 1e-9 * max(1, |total|).
    total_rc = math.fsum(row.contribution for row in r.per_position)
    assert abs(total_rc - r.total) <= 1e-9 * max(1.0, abs(r.total))

    assert [row.key for row in r.per_position] == ["A", "B", "C"]  # input order


def test_volatility_rc_per_position_hand_computed() -> None:
    # mean_A = -1/8 = -0.125, mean_P = 0.
    # cov(A,P) = Sum_t (A_t - mean_A)(P_t - 0) / 7
    #          = [Sum_t A_t*P_t - mean_A * Sum_t P_t] / 7
    #          = Sum_t A_t*P_t / 7          (since Sum_t P_t = 0)
    # A*P = 2*1 + (-3)(-2) + 1*3 + 4*3 + (-6)(-7) + 0*3 + 2*1 + (-1)(-2)
    #     = 2 + 6 + 3 + 12 + 42 + 0 + 2 + 2 = 69   -> cov = 69/7
    # RC_A = cov/sigma_P = (69/7)/sqrt(86/7)
    r = volatility_contributions(POSITIONS, params=SMALL)
    assert math.fsum(a * p for a, p in zip(A, P)) == 69.0
    assert r.contribution_of("A") == pytest.approx((69.0 / 7.0) / SIGMA_P, rel=1e-14)

    # B*P = -1 + (-4) + 0 + (-6) + (-21) + 3 + (-1) + (-2) = -32 -> cov = -32/7
    assert math.fsum(b * p for b, p in zip(B, P)) == -32.0
    assert r.contribution_of("B") == pytest.approx((-32.0 / 7.0) / SIGMA_P, rel=1e-14)

    # C*P = 0 + 2 + 6 + 3 + 28 + 6 + 0 + 4 = 49 -> cov = 49/7 = 7
    assert math.fsum(c * p for c, p in zip(C, P)) == 49.0
    assert r.contribution_of("C") == pytest.approx(7.0 / SIGMA_P, rel=1e-14)

    # 69 - 32 + 49 = 86 == Sum P^2  -> the sum identity, exactly.
    assert 69.0 - 32.0 + 49.0 == 86.0


def test_volatility_shares_sum_to_one() -> None:
    r = volatility_contributions(POSITIONS, params=SMALL)
    assert r.total is not None and r.total > 0
    for row in r.per_position:
        assert row.share == pytest.approx(row.contribution / r.total, rel=1e-15)
    assert math.fsum(row.share for row in r.per_position) == pytest.approx(1.0, abs=1e-12)
    # B hedges the book: negative covariance -> negative contribution & share.
    assert r.contribution_of("B") < 0.0
    assert r.share_of("B") < 0.0


# ---------------------------------------------------------------------------
# ES contributions (Euler tail average)
# ---------------------------------------------------------------------------


def test_tail_size_is_ceil_n_times_one_minus_alpha() -> None:
    assert tail_size(8, 0.75) == 2      # ceil(8*0.25) = 2
    assert tail_size(600, 0.95) == 30   # contract §2.3 worked examples
    assert tail_size(600, 0.99) == 6
    assert tail_size(250, 0.95) == 13   # ceil(12.5)
    assert tail_size(250, 0.99) == 3    # ceil(2.5)


def test_es_contributions_hand_computed_tail() -> None:
    r = es_contributions(POSITIONS, confidence=0.75, params=SMALL)
    assert r.method == METHOD_ES
    assert r.confidence == 0.75
    assert r.tail_size == 2                     # k = ceil(8 * 0.25)
    # ES = (7.0 + 2.0)/2 = 4.5  (largest losses at t=4 and t=1)
    assert r.total == pytest.approx(4.5, abs=1e-12)
    # A: (6 + 3)/2 = 4.5 ; B: (-3 + -2)/2 = -2.5 ; C: (4 + 1)/2 = 2.5
    assert r.contribution_of("A") == pytest.approx(4.5, abs=1e-12)
    assert r.contribution_of("B") == pytest.approx(-2.5, abs=1e-12)
    assert r.contribution_of("C") == pytest.approx(2.5, abs=1e-12)


def test_es_contributions_sum_to_es_exactly_invariant_3_3() -> None:
    for alpha in (0.75, 0.8, 0.9):
        r = es_contributions(POSITIONS, confidence=alpha, params=SMALL)
        total_rc = math.fsum(row.contribution for row in r.per_position)
        assert abs(total_rc - r.total) <= 1e-9 * max(1.0, abs(r.total))


def test_es_tail_tie_breaks_by_date_order() -> None:
    # Losses 2.0 occur at t=1 and t=7; the earlier date must be chosen, which
    # is what makes A's contribution 4.5 rather than the t=7 alternative
    # (A_7 = -1 -> (6 + 1)/2 = 3.5). Pinning A distinguishes the two tails.
    r = es_contributions(POSITIONS, confidence=0.75, params=SMALL)
    assert r.contribution_of("A") == pytest.approx(4.5, abs=1e-12)


def test_es_shares() -> None:
    r = es_contributions(POSITIONS, confidence=0.75, params=SMALL)
    assert r.share_of("A") == pytest.approx(4.5 / 4.5, abs=1e-12)      # 1.0
    assert r.share_of("B") == pytest.approx(-2.5 / 4.5, abs=1e-12)
    assert r.share_of("C") == pytest.approx(2.5 / 4.5, abs=1e-12)
    assert math.fsum(row.share for row in r.per_position) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Contract §3.4 — scaling
# ---------------------------------------------------------------------------


def test_scaling_by_k_scales_contributions_by_k() -> None:
    k = 3.0
    scaled = {key: [k * v for v in series] for key, series in POSITIONS.items()}
    base_vol = volatility_contributions(POSITIONS, params=SMALL)
    up_vol = volatility_contributions(scaled, params=SMALL)
    assert up_vol.total == pytest.approx(k * base_vol.total, rel=1e-14)
    for a, b in zip(base_vol.per_position, up_vol.per_position):
        assert b.contribution == pytest.approx(k * a.contribution, rel=1e-14)

    base_es = es_contributions(POSITIONS, confidence=0.75, params=SMALL)
    up_es = es_contributions(scaled, confidence=0.75, params=SMALL)
    assert up_es.total == pytest.approx(k * base_es.total, rel=1e-14)


# ---------------------------------------------------------------------------
# Marginal & incremental ES
# ---------------------------------------------------------------------------


def test_incremental_es_arithmetic() -> None:
    # book = A + B ; candidate = C. after == ES(A+B+C) == 4.5 (above).
    book = [a + b for a, b in zip(A, B)]
    # book = 1, -1, 1, 2, -3, 1, 1, 0 ; losses = -1, 1, -1, -2, 3, -1, -1, 0
    # k=2 -> two largest losses: 3.0 (t=4) and 1.0 (t=1, earliest of the ties)
    # ES(book) = (3.0 + 1.0)/2 = 2.0
    r = incremental_es(book, C, 0.75, params=SMALL)
    assert r.before == pytest.approx(2.0, abs=1e-12)
    assert r.after == pytest.approx(4.5, abs=1e-12)
    assert r.delta == pytest.approx(2.5, abs=1e-12)   # 4.5 - 2.0
    assert r.tail_size == 2
    assert r.confidence == 0.75


def test_incremental_es_of_a_hedge_is_negative() -> None:
    # A perfect hedge of the book: candidate = -book -> joined series is all
    # zeros, ES(after) = 0, so delta = -ES(before) < 0.
    book = [a + b for a, b in zip(A, B)]
    hedge = [-v for v in book]
    r = incremental_es(book, hedge, 0.75, params=SMALL)
    assert r.after == pytest.approx(0.0, abs=1e-12)
    assert r.before == pytest.approx(2.0, abs=1e-12)
    assert r.delta == pytest.approx(-2.0, abs=1e-12)


def test_marginal_es_is_per_unit_contribution() -> None:
    # Candidate C at quantity q=2 means candidate_pnl IS the P&L of 2 units.
    # Joined = book + C ; tail as in the fixture -> RC_C = 2.5 ; per unit
    # 2.5 / 2 = 1.25.
    book = [a + b for a, b in zip(A, B)]
    r = marginal_es(C, book, 0.75, 2.0, params=SMALL)
    assert r.value == pytest.approx(2.5 / 2.0, abs=1e-12)
    assert r.health in (ModelHealth.ACTIVE, ModelHealth.DEGRADED)

    # At q=1 the per-unit number is the whole contribution.
    r1 = marginal_es(C, book, 0.75, 1.0, params=SMALL)
    assert r1.value == pytest.approx(2.5, abs=1e-12)


def test_marginal_es_rejects_zero_quantity() -> None:
    book = [a + b for a, b in zip(A, B)]
    with pytest.raises(ValueError):
        marginal_es(C, book, 0.75, 0.0, params=SMALL)


# ---------------------------------------------------------------------------
# Mismatch, malformed input, honest nulls
# ---------------------------------------------------------------------------


def test_portfolio_mismatch_raises_value_error() -> None:
    # A supplied portfolio series that is NOT the sum of the parts cannot
    # have contributions that add up -> malformed input (contract §2.5).
    bad = [v + 5.0 for v in P]
    with pytest.raises(ValueError):
        volatility_contributions(POSITIONS, portfolio_pnl=bad, params=SMALL)
    with pytest.raises(ValueError):
        es_contributions(POSITIONS, confidence=0.75, portfolio_pnl=bad, params=SMALL)

    # The exact sum IS accepted (within tolerance).
    ok = volatility_contributions(POSITIONS, portfolio_pnl=P, params=SMALL)
    assert ok.total == pytest.approx(SIGMA_P, rel=1e-14)


def test_ragged_and_malformed_series_raise() -> None:
    with pytest.raises(ValueError):
        volatility_contributions({"A": A, "B": B[:-1]}, params=SMALL)
    with pytest.raises(ValueError):
        volatility_contributions({"A": [1.0, math.nan, 2.0, 3.0, 4.0, 5.0]}, params=SMALL)
    with pytest.raises(ValueError):  # confidence outside (0.5, 1)
        es_contributions(POSITIONS, confidence=0.5, params=SMALL)


def test_below_min_obs_is_unavailable_not_zero() -> None:
    # Contract §3.6: honest null, no exception.
    r = volatility_contributions(POSITIONS)  # default min_obs_vol = 60 > 8
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.total is None
    assert r.per_position == ()
    assert r.reason and "n=8" in r.reason
    assert r.is_available is False

    e = es_contributions(POSITIONS, confidence=0.99)  # default min_obs_99 = 250
    assert e.health is ModelHealth.UNAVAILABLE
    assert e.total is None
    assert e.reason
