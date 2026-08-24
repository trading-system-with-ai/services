"""Pre-trade portfolio risk tests (Phase B design contract §7.1–§7.2; risk
spec §8, §9, §11, §37, §46, §47, §70).

Canonical fixture — 3 positions x 8 days, plus one candidate priced PER UNIT.
Every P&L number below is EXACT in binary floating point: the return matrix
is built by dividing the intended P&L by an exposure that is a power of two
(16384 / 8192 / 4096 / 1024), so ``pnl = exposure x r`` reproduces the
integers with no rounding at all and every hand-check is literal.

    t :          0     1     2     3     4     5     6     7
    AAA#1 :    2.0  -3.0   1.0   4.0  -6.0   0.0   2.0  -1.0   (exposure 16384)
    BBB#2 :   -1.0   2.0   0.0  -2.0   3.0   1.0  -1.0   1.0   (exposure  8192)
    CCC#3 :    0.0  -1.0   2.0   1.0  -4.0   2.0   0.0  -2.0   (exposure  4096)
    BOOK  :    1.0  -2.0   3.0   3.0  -7.0   3.0   1.0  -2.0

Two candidates on ticker DDD, both 1024 of exposure per UNIT:

    RISK-ADDING (``ADDER``, correlated with the book, used for the caps):
      per unit :  1.0  -2.0   1.0   2.0  -4.0   1.0   0.0  -1.0
    HEDGE (``HEDGE``, anti-correlated, used to show a NEGATIVE increment):
      per unit : -1.0   2.0  -1.0  -3.0   5.0  -1.0   0.0   2.0

ES-95 on n=8: ``k = ceil(8 x 0.05) = 1`` -> ES_95 = the single worst loss.
The book's worst day is t=4 (P = -7.0), so ES_95(book) = 7.0. Adding q units
of ADDER makes day t=4 equal ``-7 - 4q``, and it stays the worst day, so:

    ES_95(book + q x ADDER) = 7 + 4q      (linear, hence MONOTONE in q)

which is what makes the cap arithmetic below hand-checkable to the unit.

n=8 is far below the production ``min_obs`` (60/250); ``SMALL`` /
``LIMITS_SMALL`` lower it so the ARITHMETIC can be tested on a sample a
human can verify. ``min_obs`` itself is exercised in its own test.
"""
from __future__ import annotations

import dataclasses
import math
from datetime import date, timedelta

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.contribution import (
    ContributionParams,
    es_contributions,
)
from libs.trading_core.risk.models.var_es import historical_es, tail_size
from libs.trading_core.risk.pnl_series import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL_CONST_IV,
    PositionRiskInput,
    book_method_summary,
    book_pnl_series,
)
from libs.trading_core.risk.pretrade import (
    CODE_BUCKET_ES_CONTRIBUTION,
    CODE_ES_CONTRIBUTION,
    CODE_INCREMENTAL_ES,
    CODE_PORTFOLIO_ES,
    DECISION_APPROVE,
    DECISION_APPROVE_WITH_RESIZE,
    DECISION_REJECT,
    LAYER_CONCENTRATION,
    LAYER_STATISTICAL,
    MAX_BISECTION_STEPS,
    MODE_SHADOW,
    CandidateSpec,
    QuantityCap,
    StatisticalLimits,
    _largest_passing,
    compare,
    proposed_book,
    shadow_verdict,
    statistical_caps,
)
from libs.trading_core.risk.returns import ReturnMatrix

# --- the fixture ----------------------------------------------------------

AAA = [2.0, -3.0, 1.0, 4.0, -6.0, 0.0, 2.0, -1.0]
BBB = [-1.0, 2.0, 0.0, -2.0, 3.0, 1.0, -1.0, 1.0]
CCC = [0.0, -1.0, 2.0, 1.0, -4.0, 2.0, 0.0, -2.0]
BOOK = [1.0, -2.0, 3.0, 3.0, -7.0, 3.0, 1.0, -2.0]

ADDER = [1.0, -2.0, 1.0, 2.0, -4.0, 1.0, 0.0, -1.0]
HEDGE = [-1.0, 2.0, -1.0, -3.0, 5.0, -1.0, 0.0, 2.0]

DATES = tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(8))
NAV = 1_000.0

# Exposures are powers of two so pnl/exposure is exact in binary.
EXP_AAA, EXP_BBB, EXP_CCC, EXP_CAND = 16384.0, 8192.0, 4096.0, 1024.0
SPOT = 128.0

SMALL = ContributionParams(
    min_obs_vol=5, min_obs_95=5, min_obs_99=5, degraded_multiple=1.0
)
LIMITS_SMALL = StatisticalLimits(min_obs=5)

POSITIONS = [
    # exposure = quantity x multiplier x delta x spot: 128 x 1 x 1 x 128 = 16384
    PositionRiskInput("AAA#1", "AAA", "LONG_STOCK", 128, 1, SPOT, 1.0, 500.0),
    PositionRiskInput("BBB#2", "BBB", "LONG_STOCK", 64, 1, SPOT, 1.0, 300.0),
    PositionRiskInput("CCC#3", "CCC", "LONG_STOCK", 32, 1, SPOT, 1.0, 200.0),
]


def matrix(candidate_unit_pnl: list[float] = ADDER) -> ReturnMatrix:
    """SIMPLE return matrix whose columns reproduce the fixture P&L exactly."""
    rows = tuple(
        zip(
            [x / EXP_AAA for x in AAA],
            [x / EXP_BBB for x in BBB],
            [x / EXP_CCC for x in CCC],
            [x / EXP_CAND for x in candidate_unit_pnl],
        )
    )
    return ReturnMatrix(
        dates=DATES,
        tickers=("AAA", "BBB", "CCC", "DDD"),
        rows=rows,
        return_type="SIMPLE",
    )


def candidate(quantity_requested: int = 10, ticker: str = "DDD") -> CandidateSpec:
    """One UNIT of the candidate carries 1024 of exposure (8 x 128 x 1.0)."""
    return CandidateSpec(
        key="DDD#cand",
        ticker=ticker,
        instrument="LONG_STOCK",
        multiplier=8,
        spot=SPOT,
        delta=1.0,
        max_loss_per_unit=50.0,
        capital_per_unit=100.0,
        quantity_requested=quantity_requested,
    )


def book(candidate_unit_pnl: list[float] = ADDER):
    m = matrix(candidate_unit_pnl)
    return book_pnl_series(POSITIONS, m), m


# ---------------------------------------------------------------------------
# The fixture is what the docstring claims
# ---------------------------------------------------------------------------


def test_fixture_pnl_is_exact() -> None:
    b, _ = book()
    assert b.per_position["AAA#1"] == AAA
    assert b.per_position["BBB#2"] == BBB
    assert b.per_position["CCC#3"] == CCC
    assert b.total == BOOK
    # k = ceil(8 * 0.05) = 1 -> ES_95 is the single worst loss, 7.0 at t=4.
    assert tail_size(8, 0.95) == 1
    assert historical_es(BOOK, 0.95, min_obs=5).value == 7.0


# ---------------------------------------------------------------------------
# proposed_book (contract §7.1)
# ---------------------------------------------------------------------------


def test_proposed_book_prices_the_candidate_at_quantity() -> None:
    b, m = book()
    after = proposed_book(b, candidate(), 3, m)
    # 3 units x 1024 exposure = 3072; pnl_t = 3072 * r_t = 3 * ADDER_t.
    assert after.per_position["DDD#cand"] == [3.0 * x for x in ADDER]
    assert after.total == [p + 3.0 * a for p, a in zip(BOOK, ADDER)]
    # The book's own positions are carried through untouched.
    assert after.per_position["AAA#1"] == AAA
    assert after.dates == b.dates and after.method == b.method


def test_proposed_book_labels_the_candidate_with_the_estimator_that_priced_it(
) -> None:
    """design §10.3: the after-book's labels are the book's PLUS the
    candidate's own, and ``method`` is the summary of the JOINED map.

    A DELTA_LINEAR book joined to a full-revalued option candidate is a
    FULL_REVAL_CONST_IV book: reporting the pre-trade book's stale
    DELTA_LINEAR would mislabel a series that really was revalued, and
    dropping the map entirely would lose the book's own labels too.
    """
    b, m = book()
    assert b.method == METHOD_DELTA_LINEAR

    # A STOCK candidate changes nothing: still all delta-linear.
    stock_after = proposed_book(b, candidate(), 3, m)
    assert stock_after.method == METHOD_DELTA_LINEAR
    assert stock_after.method_by_key == {**b.method_by_key,
                                         "DDD#cand": METHOD_DELTA_LINEAR}

    # An OPTION candidate carrying its five leg fields flips the summary.
    opt = CandidateSpec(
        key="DDD#cand", ticker="DDD", instrument="LONG_CALL", multiplier=100,
        spot=SPOT, delta=0.55, max_loss_per_unit=50.0, capital_per_unit=100.0,
        quantity_requested=10, strike=SPOT, right="C", t_years=0.25,
        iv0=0.30, mark0=6.0,
    )
    after = proposed_book(b, opt, 3, m)
    assert after.method == METHOD_FULL_REVAL_CONST_IV
    assert after.method_by_key["DDD#cand"] == METHOD_FULL_REVAL_CONST_IV
    # The book's own rows keep THEIR labels — they were carried through
    # unchanged, so their labels must be too.
    for key, label in b.method_by_key.items():
        assert after.method_by_key[key] == label
    assert after.method_counts[METHOD_FULL_REVAL_CONST_IV] == 1
    # The label is not decoration: the series really differs from the
    # delta-linear one at the same quantity.
    lin = dataclasses.replace(opt, strike=None, right=None, t_years=None,
                              iv0=None, mark0=None)
    assert (after.per_position["DDD#cand"]
            != proposed_book(b, lin, 3, m).per_position["DDD#cand"])
    # And the summary agrees with the shared helper.
    assert after.method == book_method_summary(after.method_by_key)


def test_proposed_book_excluded_candidate_carries_no_method_label() -> None:
    """An unpriceable candidate (no column) was priced by NOTHING, so it
    gets no label — honest gap, never a default."""
    b, m = book()
    after = proposed_book(b, candidate(ticker="ZZZ"), 3, m)
    assert "DDD#cand" in after.keys_excluded
    assert "DDD#cand" not in after.method_by_key
    assert after.method_by_key == dict(b.method_by_key)


def test_proposed_book_at_zero_leaves_the_total_unchanged() -> None:
    b, m = book()
    after = proposed_book(b, candidate(), 0, m)
    assert after.total == BOOK
    assert after.per_position["DDD#cand"] == [0.0] * 8


def test_proposed_book_refuses_negative_quantity_and_key_collision() -> None:
    b, m = book()
    with pytest.raises(ValueError, match="quantity must be an int >= 0"):
        proposed_book(b, candidate(), -1, m)
    collide = CandidateSpec(
        key="AAA#1", ticker="DDD", instrument="LONG_STOCK", multiplier=8,
        spot=SPOT, delta=1.0, max_loss_per_unit=50.0, capital_per_unit=100.0,
        quantity_requested=1,
    )
    with pytest.raises(ValueError, match="already exists in the book"):
        proposed_book(b, collide, 1, m)


def test_proposed_book_excludes_a_candidate_with_no_returns_column() -> None:
    """Honest gap (contract §7.2): no column -> excluded and NAMED, not zeros."""
    b, m = book()
    after = proposed_book(b, candidate(ticker="ZZZ"), 5, m)
    assert "ZZZ" in after.tickers_missing
    assert "DDD#cand" in after.keys_excluded
    assert "DDD#cand" not in after.per_position
    assert after.total == BOOK  # unchanged: nothing was fabricated


# ---------------------------------------------------------------------------
# compare (spec §46; contract §7.1)
# ---------------------------------------------------------------------------


def comparison(unit_pnl=ADDER, quantity=3, **kwargs):
    b, m = book(unit_pnl)
    params = dict(
        returns=m,
        nav=NAV,
        heat_before=0.032,
        heat_after=0.040,
        cash_before=0.42,
        cash_after=0.38,
        positions=POSITIONS,
        limits=LIMITS_SMALL,
        contribution_params=SMALL,
    )
    params.update(kwargs)
    return compare(b, candidate(), quantity, **params)


def test_compare_hand_checked_on_the_3x8_book() -> None:
    c = comparison(quantity=3)
    # n=8 with min_obs=5 sits in the thin-sample band (n < 2*min_obs), so the
    # comparison honestly reports DEGRADED and says why — the numbers are
    # real, the sample is small, and the caller is told (contract §2.3).
    assert c.health is ModelHealth.DEGRADED
    assert "n=8" in c.reason
    assert c.quantity == 3 and c.n_obs == 8 and c.tail_size_95 == 1

    # Tier 0 numbers are passed through verbatim, never recomputed here.
    assert c.heat_pct == (0.032, 0.040)
    assert c.cash_pct == (0.42, 0.38)

    # ES_95 before = 7.0 (worst day t=4). After 3 units: t=4 becomes
    # -7 - 3*4 = -19, still the worst day -> ES_95 after = 19.0.
    assert c.es_hist_95.before.value == 7.0
    assert c.es_hist_95.after.value == 19.0
    assert c.es_hist_95.delta_usd == 12.0
    assert c.es_hist_95.delta_pct_nav == 12.0 / NAV

    # k=1 -> VaR_95 == ES_95 (the single worst loss IS its own average).
    assert c.var_hist_95.before.value == 7.0
    assert c.var_hist_95.after.value == 19.0

    # k = ceil(8 * 0.01) = 1 too, so the 99% pair equals the 95% pair here.
    assert tail_size(8, 0.99) == 1
    assert c.es_hist_99.before.value == 7.0
    assert c.es_hist_99.after.value == 19.0


def test_incremental_es_equals_es_after_minus_es_before_exactly() -> None:
    """Contract §7.1 / spec §8 'Delta ES = Y - X' — EXACTLY, not approximately."""
    for q in range(0, 6):
        c = comparison(quantity=q)
        before = c.es_hist_95.before.value
        after = c.es_hist_95.after.value
        assert c.incremental_es_95_usd == after - before
        assert c.incremental_es_95_usd == float(4 * q)  # ES(q) = 7 + 4q
        assert c.incremental_es_95_pct_nav == c.incremental_es_95_usd / NAV


def test_marginal_es_is_the_candidate_euler_contribution_over_quantity() -> None:
    """Contract §7.1 / spec §9: marginal = RC^ES_candidate / q, hand-checked."""
    q = 3
    c = comparison(quantity=q)
    b, m = book()
    after = proposed_book(b, candidate(), q, m)
    contrib = es_contributions(
        after.per_position, 0.95, min_obs=5, params=SMALL
    )
    rc = contrib.contribution_of("DDD#cand")
    # Tail of the joined series is the single day t=4; the candidate's P&L
    # there is 3 * -4.0 = -12.0, so RC = -(-12.0) = 12.0 and marginal = 4.0
    # per unit — exactly the slope of ES(q) = 7 + 4q.
    assert rc == 12.0
    assert c.marginal_es_95_per_unit == rc / q == 4.0


def test_marginal_es_is_none_at_zero_quantity() -> None:
    """Honest null: there is no 'per unit' when no unit is held."""
    assert comparison(quantity=0).marginal_es_95_per_unit is None


def test_a_hedge_lowers_es_and_reports_a_negative_increment() -> None:
    """Spec §8: a trade can REDUCE portfolio ES; the sign is reported honestly."""
    c = comparison(unit_pnl=HEDGE, quantity=1)
    # After 1 HEDGE unit the book is [0, 0, 2, 0, -2, 2, 1, 0]; the worst
    # loss is 2.0 at t=4, so ES_95 falls from 7.0 to 2.0.
    assert c.es_hist_95.before.value == 7.0
    assert c.es_hist_95.after.value == 2.0
    assert c.incremental_es_95_usd == -5.0


def test_es_shares_and_bucket_shares_after(  ) -> None:
    """Spec §11 / §46: candidate share, worst single-name share, bucket share."""
    c = comparison(quantity=2, buckets={"TECH": ("AAA", "DDD")})
    # At q=2 the joined tail is t=4 alone; contributions are the losses
    # there: AAA 6, BBB -3, CCC 4, DDD 2*4 = 8 -> total 15 = ES_95(2).
    assert c.es_hist_95.after.value == 15.0
    assert c.candidate_es_share_after == pytest.approx(8.0 / 15.0)
    # The largest single-name share after is the candidate's own 8/15.
    assert c.max_single_es_share_after == pytest.approx(8.0 / 15.0)
    # Before the trade the tail is t=4 too: AAA 6 / BBB -3 / CCC 4, total 7;
    # the worst single name is AAA at 6/7.
    assert c.max_single_es_share_before == pytest.approx(6.0 / 7.0)
    # The TECH bucket holds AAA + the candidate: (6 + 8)/15.
    assert c.bucket_es_share_after["TECH"] == pytest.approx(14.0 / 15.0)


def test_bucket_shares_only_cover_buckets_the_candidate_belongs_to() -> None:
    c = comparison(quantity=2, buckets={"TECH": ("AAA", "DDD"), "ENERGY": ("BBB",)})
    assert set(c.bucket_es_share_after) == {"TECH"}


def test_net_delta_notional_adds_the_candidate_exposure() -> None:
    c = comparison(quantity=3, delta_notional_before=50_000.0)
    # 3 units x 8 multiplier x 1.0 delta x 128 spot = 3072.
    assert c.net_delta_notional == (50_000.0, 53_072.0)


def test_net_delta_notional_after_is_none_when_before_is_unknown() -> None:
    """No guessing: an unknown 'before' cannot produce a known 'after'."""
    assert comparison(quantity=3).net_delta_notional == (None, None)


def test_contributions_after_sum_to_es_after() -> None:
    """Contract §3.3 carried into Phase C: Sum RC_i == ES_95(after) exactly."""
    c = comparison(quantity=4)
    contrib = c.contributions_es_95_after
    total = math.fsum(row.contribution for row in contrib.per_position)
    assert total == contrib.total == c.es_hist_95.after.value == 23.0


# ---------------------------------------------------------------------------
# UNAVAILABLE (contract §7.2) — no view, no caps, honest reason
# ---------------------------------------------------------------------------


def test_compare_unavailable_when_the_candidate_has_no_returns() -> None:
    b, m = book()
    c = compare(
        b, candidate(ticker="ZZZ"), 3,
        returns=m, nav=NAV, heat_before=0.03, heat_after=0.04,
        cash_before=0.4, cash_after=0.35, positions=POSITIONS,
        limits=LIMITS_SMALL, contribution_params=SMALL,
    )
    assert c.health is ModelHealth.UNAVAILABLE
    assert "ZZZ" in c.reason and "no returns column" in c.reason
    assert c.es_hist_95.before is None and c.es_hist_95.delta_usd is None
    assert c.incremental_es_95_usd is None
    assert c.marginal_es_95_per_unit is None
    assert c.candidate_es_share_after is None
    assert c.bucket_es_share_after == {}
    # Tier 0's own numbers still come through — a statistical gap never
    # blanks the hard-limit view (spec §38).
    assert c.heat_pct == (0.03, 0.04)


def test_compare_unavailable_below_min_obs() -> None:
    b, m = book()
    c = compare(
        b, candidate(), 3, returns=m, nav=NAV, heat_before=0.03,
        heat_after=0.04, cash_before=0.4, cash_after=0.35,
        positions=POSITIONS, limits=StatisticalLimits(min_obs=60),
    )
    assert c.health is ModelHealth.UNAVAILABLE
    assert c.reason == "n=8 < min_obs=60"


def test_unavailable_view_produces_no_caps() -> None:
    """Contract §7.2: 'a missing statistical view NEVER produces a cap'."""
    b, m = book()
    caps, health, reason = statistical_caps(
        b, candidate(ticker="ZZZ"), returns=m, nav=NAV,
        positions=POSITIONS, limits=LIMITS_SMALL, contribution_params=SMALL,
    )
    assert caps == [] and health is ModelHealth.UNAVAILABLE
    assert "ZZZ" in reason

    caps, health, reason = statistical_caps(
        b, candidate(), returns=m, nav=NAV, positions=POSITIONS,
        limits=StatisticalLimits(min_obs=60),
    )
    assert caps == [] and health is ModelHealth.UNAVAILABLE
    assert reason == "n=8 < min_obs=60"


# ---------------------------------------------------------------------------
# statistical_caps (spec §11, §37; contract §7.2)
# ---------------------------------------------------------------------------


def caps_for(limits: StatisticalLimits, *, requested: int = 10, buckets=None):
    b, m = book()
    return statistical_caps(
        b, candidate(requested), returns=m, nav=NAV, positions=POSITIONS,
        buckets=buckets or {}, limits=limits, contribution_params=SMALL,
    )


def test_no_cap_when_every_limit_is_satisfied_at_the_requested_quantity() -> None:
    # ES_95(10) = 47 = 4.7% of NAV; all four limits set above what q=10 does.
    caps, health, reason = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.10,
            max_single_position_es_share=0.95,
            max_bucket_es_share=0.99,
            max_incremental_es95_pct_nav=0.10,
            min_obs=5,
        ),
        buckets={"TECH": ("AAA", "DDD")},
    )
    assert caps == [] and health is ModelHealth.ACTIVE and reason is None


def test_portfolio_es_cap_bisects_to_the_largest_passing_quantity() -> None:
    """Monotone case, hand-checked: ES(q) = 7 + 4q, limit 2.3% of 1000 = $23.

    7 + 4q <= 23  <=>  q <= 4  -> the largest passing quantity is 4.
    """
    caps, health, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.023,
            max_single_position_es_share=0.99,
            max_bucket_es_share=0.99,
            max_incremental_es95_pct_nav=0.99,
            min_obs=5,
        )
    )
    assert health is ModelHealth.ACTIVE
    assert len(caps) == 1
    cap = caps[0]
    assert cap.code == CODE_PORTFOLIO_ES
    assert cap.layer == LAYER_STATISTICAL
    assert cap.cap_qty == 4
    assert cap.measured["es95_usd_at_requested"] == 47.0  # 7 + 4*10
    assert cap.measured["es95_usd_at_cap"] == 23.0  # 7 + 4*4, exactly at the limit
    assert cap.measured["limit_usd"] == 23.0
    # Spec §47: the sentence carries the REAL numbers.
    assert "$47.00" in cap.sentence and "4.70% of NAV" in cap.sentence
    assert "reduced from 10 to 4" in cap.sentence


def test_incremental_es_cap_bisects() -> None:
    """Increment(q) = 4q; limit 1.2% of 1000 = $12 -> 4q <= 12 <=> q <= 3."""
    caps, _, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.99,
            max_single_position_es_share=0.99,
            max_bucket_es_share=0.99,
            max_incremental_es95_pct_nav=0.012,
            min_obs=5,
        )
    )
    assert len(caps) == 1
    cap = caps[0]
    assert cap.code == CODE_INCREMENTAL_ES and cap.layer == LAYER_STATISTICAL
    assert cap.cap_qty == 3
    assert cap.measured["es95_before_usd"] == 7.0
    assert cap.measured["incremental_usd_at_requested"] == 40.0
    assert cap.measured["incremental_usd_at_cap"] == 12.0


def test_es_contribution_cap_is_a_concentration_layer_cap() -> None:
    """Candidate share(q) = 4q/(7+4q); limit 0.60 -> 4q <= 0.6(7+4q)
    <=> 1.6q <= 4.2 <=> q <= 2.625 -> largest integer 2 (share 8/15 = 53.3%)."""
    caps, _, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.99,
            max_single_position_es_share=0.60,
            max_bucket_es_share=0.99,
            max_incremental_es95_pct_nav=0.99,
            min_obs=5,
        )
    )
    assert len(caps) == 1
    cap = caps[0]
    assert cap.code == CODE_ES_CONTRIBUTION
    assert cap.layer == LAYER_CONCENTRATION
    assert cap.cap_qty == 2
    assert cap.measured["es_share_at_requested"] == pytest.approx(40.0 / 47.0)
    assert cap.measured["es_share_at_cap"] == pytest.approx(8.0 / 15.0)


def test_bucket_es_cap_names_the_bucket_in_its_code() -> None:
    """Bucket AAA+DDD share(q) = (6 + 4q)/(7 + 4q); limit 0.94:
    6 + 4q <= 0.94(7 + 4q) <=> 0.24q <= 0.58 <=> q <= 2.41 -> 2 (14/15 = 93.3%)."""
    caps, _, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.99,
            max_single_position_es_share=0.99,
            max_bucket_es_share=0.94,
            max_incremental_es95_pct_nav=0.99,
            min_obs=5,
        ),
        buckets={"TECH": ("AAA", "DDD")},
    )
    assert len(caps) == 1
    cap = caps[0]
    assert cap.code == f"{CODE_BUCKET_ES_CONTRIBUTION}:TECH"
    assert cap.layer == LAYER_CONCENTRATION
    assert cap.cap_qty == 2
    assert cap.measured["bucket_es_share_at_cap"] == pytest.approx(14.0 / 15.0)


def test_all_four_limits_can_bind_at_once_in_deterministic_order() -> None:
    caps, health, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.023,
            max_single_position_es_share=0.60,
            max_bucket_es_share=0.94,
            max_incremental_es95_pct_nav=0.012,
            min_obs=5,
        ),
        buckets={"TECH": ("AAA", "DDD")},
    )
    assert health is ModelHealth.ACTIVE
    assert [c.code for c in caps] == [
        CODE_PORTFOLIO_ES,
        CODE_ES_CONTRIBUTION,
        f"{CODE_BUCKET_ES_CONTRIBUTION}:TECH",
        CODE_INCREMENTAL_ES,
    ]
    assert [c.cap_qty for c in caps] == [4, 2, 2, 3]


def test_a_cap_never_exceeds_the_requested_quantity() -> None:
    """A cap only ever REDUCES: with a request of 2 the ES cap that would
    allow 4 reports 2, not 4."""
    caps, _, _ = caps_for(
        StatisticalLimits(
            max_portfolio_es95_pct_nav=0.023,
            max_single_position_es_share=0.99,
            max_bucket_es_share=0.99,
            max_incremental_es95_pct_nav=0.99,
            min_obs=5,
        ),
        requested=2,
    )
    # ES_95(2) = 15 <= 23, so the limit is satisfied at the request: no cap.
    assert caps == []


# ---------------------------------------------------------------------------
# The bisection itself (contract §7.2: <= 20 steps, with a step-down guard)
# ---------------------------------------------------------------------------


def test_bisection_finds_the_largest_passing_q_on_a_monotone_predicate() -> None:
    for cutoff in (0, 1, 4, 7, 9, 10):
        assert _largest_passing(10, lambda q, c=cutoff: q <= c) == cutoff
    assert _largest_passing(10, lambda q: True) == 10
    assert _largest_passing(10, lambda q: False) == 0
    assert _largest_passing(0, lambda q: True) == 0


def test_bisection_stays_within_the_step_budget() -> None:
    probes: list[int] = []

    def passes(q: int) -> bool:
        probes.append(q)
        return q <= 3

    assert _largest_passing(1_000_000, passes) == 3
    # requested + bisection + verification, all bounded by the contract's 20.
    assert len(probes) <= 2 * MAX_BISECTION_STEPS + 1


def test_step_down_guard_never_returns_an_unverified_quantity() -> None:
    """Constructed NON-MONOTONE predicate (contract §7.2 'non-monotone corner').

    Passing set {0, 1, 2, 3, 4, 6, 7} on a request of 10. Bisection probes
    10 (fail), 5 (fail), 2 (pass -> lo=2), 3 (pass -> lo=3), 4 (pass ->
    lo=4) and returns 4. It is NOT the global maximum (7 also passes) — no
    bounded bisection can see past the failing 5 — but it IS verified, and
    under-approving is the safe direction for a risk cap.
    """
    passing = {0, 1, 2, 3, 4, 6, 7}
    probes: list[int] = []

    def passes(q: int) -> bool:
        probes.append(q)
        return q in passing

    result = _largest_passing(10, passes)
    assert result == 4
    assert probes == [10, 5, 2, 3, 4]
    assert passes(result)  # the invariant: whatever comes back was CHECKED

    # And when NOTHING in the searched interval passes, the honest answer is
    # 0 rather than an unverified guess.
    assert _largest_passing(10, lambda q: q == 9) == 0


def test_every_cap_quantity_actually_satisfies_its_limit() -> None:
    """The property the whole search exists for: a cap must never hand back
    a quantity that still breaches (contract §7.2)."""
    limits = StatisticalLimits(
        max_portfolio_es95_pct_nav=0.023,
        max_single_position_es_share=0.60,
        max_bucket_es_share=0.94,
        max_incremental_es95_pct_nav=0.012,
        min_obs=5,
    )
    caps, _, _ = caps_for(limits, buckets={"TECH": ("AAA", "DDD")})
    b, m = book()
    for cap in caps:
        after = proposed_book(b, candidate(10), cap.cap_qty, m)
        contrib = es_contributions(
            after.per_position, 0.95, min_obs=5, params=SMALL
        )
        es = contrib.total
        if cap.code == CODE_PORTFOLIO_ES:
            assert es <= limits.max_portfolio_es95_pct_nav * NAV
        elif cap.code == CODE_ES_CONTRIBUTION:
            assert contrib.share_of("DDD#cand") <= limits.max_single_position_es_share
        elif cap.code == CODE_INCREMENTAL_ES:
            assert es - 7.0 <= limits.max_incremental_es95_pct_nav * NAV
        else:
            share = (
                contrib.contribution_of("AAA#1") + contrib.contribution_of("DDD#cand")
            ) / es
            assert share <= limits.max_bucket_es_share


# ---------------------------------------------------------------------------
# shadow_verdict (spec §47, §70; contract §7.2)
# ---------------------------------------------------------------------------


def cap(code: str, qty: int, layer: str = LAYER_STATISTICAL) -> QuantityCap:
    return QuantityCap(code=code, layer=layer, cap_qty=qty, sentence=f"{code} caps at {qty}.")


def test_shadow_verdict_orders_binding_caps_most_restrictive_first() -> None:
    """Contract §7.2 hand-check: approved 10 with caps A(7), B(3), C(12) ->
    quantity 3, APPROVE_WITH_RESIZE, binding ('B', 'A') — C never binds."""
    caps = [cap("A", 7), cap("B", 3), cap("C", 12)]
    v = shadow_verdict(10, caps)
    assert v.hypothetical_quantity == 3
    assert v.hypothetical_decision == DECISION_APPROVE_WITH_RESIZE
    assert v.binding == ("B", "A")
    assert v.caps == tuple(caps)  # every cap kept, binding or not
    assert v.mode == MODE_SHADOW
    assert v.changes_quantity is True


def test_shadow_verdict_approves_when_no_cap_binds() -> None:
    v = shadow_verdict(5, [cap("A", 5), cap("B", 9)])
    assert v.hypothetical_decision == DECISION_APPROVE
    assert v.hypothetical_quantity == 5
    assert v.binding == ()
    assert v.changes_quantity is False


def test_shadow_verdict_rejects_at_zero() -> None:
    v = shadow_verdict(4, [cap("A", 3), cap("B", 0)])
    assert v.hypothetical_decision == DECISION_REJECT
    assert v.hypothetical_quantity == 0
    assert v.binding == ("B", "A")


def test_shadow_verdict_with_no_caps_approves_the_tier_0_quantity() -> None:
    v = shadow_verdict(7, [])
    assert v.hypothetical_decision == DECISION_APPROVE
    assert v.hypothetical_quantity == 7 and v.binding == () and v.caps == ()


def test_shadow_verdict_of_a_tier_0_reject_stays_a_reject() -> None:
    """The statistical layer can never GRANT risk Tier 0 refused."""
    v = shadow_verdict(0, [cap("A", 9)])
    assert v.hypothetical_quantity == 0
    assert v.hypothetical_decision == DECISION_REJECT
    assert v.binding == ()  # nothing "binds" below zero


def test_shadow_verdict_ties_keep_emission_order() -> None:
    v = shadow_verdict(10, [cap("FIRST", 2), cap("SECOND", 2), cap("THIRD", 1)])
    assert v.binding == ("THIRD", "FIRST", "SECOND")


def test_shadow_verdict_refuses_a_negative_approved_quantity() -> None:
    with pytest.raises(ValueError, match="approved_qty must be an int >= 0"):
        shadow_verdict(-1, [])


# ---------------------------------------------------------------------------
# Parameters, validation and the SHADOW contract
# ---------------------------------------------------------------------------


def test_statistical_limits_defaults_are_the_documented_research_values() -> None:
    """Contract §7.2 — RESEARCH DEFAULTS, UNVALIDATED, and every one a parameter."""
    limits = StatisticalLimits()
    assert limits.max_portfolio_es95_pct_nav == 0.05
    assert limits.max_single_position_es_share == 0.35
    assert limits.max_bucket_es_share == 0.50
    assert limits.max_incremental_es95_pct_nav == 0.015
    assert limits.min_obs == 60
    # SHADOW until a human promotes it (spec §70).
    assert limits.mode == MODE_SHADOW and limits.is_shadow is True


def test_statistical_limits_reject_malformed_values() -> None:
    for bad in ({"max_portfolio_es95_pct_nav": 0.0},
                {"max_single_position_es_share": -0.1},
                {"max_bucket_es_share": float("nan")},
                {"max_incremental_es95_pct_nav": float("inf")},
                {"min_obs": 1},
                {"mode": "PRODUCTION_MAYBE"}):
        with pytest.raises(ValueError):
            StatisticalLimits(**bad)


def test_quantity_cap_refuses_an_unexplained_or_mislabelled_cap() -> None:
    with pytest.raises(ValueError, match="layer must be"):
        QuantityCap(code="X", layer="HARD_LIMIT", cap_qty=1, sentence="s")
    with pytest.raises(ValueError, match="sentence must be"):
        QuantityCap(code="X", layer=LAYER_STATISTICAL, cap_qty=1, sentence="")
    with pytest.raises(ValueError, match="cap_qty must be an int >= 0"):
        QuantityCap(code="X", layer=LAYER_STATISTICAL, cap_qty=-1, sentence="s")


def test_candidate_spec_validates_its_inputs() -> None:
    for bad in ({"multiplier": 0}, {"spot": 0.0}, {"spot": float("nan")},
                {"delta": float("inf")}, {"max_loss_per_unit": -1.0},
                {"capital_per_unit": -1.0}, {"quantity_requested": -1}):
        kwargs = dict(
            key="K", ticker="T", instrument="LONG_STOCK", multiplier=1,
            spot=100.0, delta=1.0, max_loss_per_unit=1.0,
            capital_per_unit=1.0, quantity_requested=1,
        )
        kwargs.update(bad)
        with pytest.raises(ValueError):
            CandidateSpec(**kwargs)


def test_candidate_exposure_and_position_at_quantity() -> None:
    c = candidate()
    assert c.exposure_at(3) == 3 * 8 * 1.0 * 128.0 == 3072.0
    pos = c.position_at(3)
    assert pos.quantity == 3 and pos.exposure == 3072.0
    assert pos.max_loss == 150.0  # 3 x max_loss_per_unit
    with pytest.raises(ValueError, match="quantity must be an int >= 0"):
        c.position_at(-1)


def test_compare_requires_a_positive_nav() -> None:
    b, m = book()
    with pytest.raises(ValueError, match="nav must be > 0"):
        compare(b, candidate(), 1, returns=m, nav=0.0, heat_before=0.0,
                heat_after=0.0, cash_before=0.0, cash_after=0.0)
