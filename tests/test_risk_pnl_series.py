"""Book P&L construction tests — DELTA_LINEAR (risk spec §8, §21; Phase B
design contract §2.9).

Every number is hand-checked in a comment: stock long/short sign, option
×100 with delta (long call positive delta, long put negative delta), the
missing-ticker exclusion (named, never zero-filled), ``total == fsum`` of
the included parts, the SIMPLE-only return-type guard, and the contract
§3.4 scaling invariant (k × quantity ⇒ k × P&L).
"""
import math
from datetime import date

import pytest

from libs.trading_core.risk.pnl_series import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL_CONST_IV,
    BookPnl,
    PositionRiskInput,
    book_method_summary,
    book_pnl_series,
    position_pnl_series,
)
from libs.trading_core.risk.returns import ReturnMatrix

D1, D2, D3 = date(2026, 8, 4), date(2026, 8, 5), date(2026, 8, 6)

# AAPL simple returns: +1%, -2%, +0.5%; MSFT: -1%, +3%, 0%
MATRIX = ReturnMatrix(
    dates=(D1, D2, D3),
    tickers=("AAPL", "MSFT"),
    rows=((0.01, -0.01), (-0.02, 0.03), (0.005, 0.0)),
    return_type="SIMPLE",
)
LOG_MATRIX = ReturnMatrix(
    dates=(D1, D2, D3),
    tickers=("AAPL", "MSFT"),
    rows=((0.01, -0.01), (-0.02, 0.03), (0.005, 0.0)),
    return_type="LOG",
)


def stock(key, ticker, qty, spot):
    return PositionRiskInput(
        key=key, ticker=ticker, instrument="LONG_STOCK" if qty > 0 else "SHORT_STOCK",
        quantity=qty, multiplier=1, spot=spot, delta=1.0 if qty > 0 else 1.0,
        max_loss=abs(qty) * spot * 0.05,
    )


# ---------------------------------------------------------------------------
# position_pnl_series
# ---------------------------------------------------------------------------


def test_long_stock_pnl_sign_and_arithmetic():
    # 100 sh AAPL @ 200, delta 1 -> exposure = 100*1*1*200 = 20 000
    # pnl = 20000 * [0.01, -0.02, 0.005] = [200, -400, 100]
    pos = stock("AAPL#1", "AAPL", 100, 200.0)
    assert pos.exposure == 20_000.0
    assert position_pnl_series(pos, MATRIX) == pytest.approx(
        [200.0, -400.0, 100.0], abs=1e-9
    )


def test_short_stock_pnl_is_negated():
    # -100 sh AAPL @ 200 -> exposure = -20 000; a -2% day GAINS 400
    pos = stock("AAPL#2", "AAPL", -100, 200.0)
    assert pos.exposure == -20_000.0
    assert position_pnl_series(pos, MATRIX) == pytest.approx(
        [-200.0, 400.0, -100.0], abs=1e-9
    )


def test_long_call_option_times_100_with_delta():
    # 4 contracts, multiplier 100, delta 0.62, spot 200
    # exposure = 4*100*0.62*200 = 49 600; pnl = 49600 * [0.01,-0.02,0.005]
    #          = [496, -992, 248]
    pos = PositionRiskInput(
        key="AAPL#C", ticker="AAPL", instrument="LONG_CALL", quantity=4,
        multiplier=100, spot=200.0, delta=0.62, max_loss=2_400.0,
    )
    assert pos.exposure == pytest.approx(49_600.0)
    assert position_pnl_series(pos, MATRIX) == pytest.approx(
        [496.0, -992.0, 248.0], abs=1e-9
    )


def test_long_put_negative_delta_gains_on_down_day():
    # 2 contracts on MSFT, delta -0.40, spot 50
    # exposure = 2*100*(-0.40)*50 = -4 000; MSFT r = [-0.01, 0.03, 0.0]
    # pnl = [40, -120, 0]
    pos = PositionRiskInput(
        key="MSFT#P", ticker="MSFT", instrument="LONG_PUT", quantity=2,
        multiplier=100, spot=50.0, delta=-0.40, max_loss=600.0,
    )
    assert pos.exposure == pytest.approx(-4_000.0)
    assert position_pnl_series(pos, MATRIX) == pytest.approx(
        [40.0, -120.0, 0.0], abs=1e-9
    )


def test_covered_call_short_leg_delta_already_negated_by_caller():
    # Caller passes the SHORT leg's negated delta (as greeks.py): -1
    # contract with delta -0.30 negated -> quantity -1, delta 0.30, or
    # quantity 1, delta -0.30 — either way exposure = -1*100*0.30*100 = -3000
    a = PositionRiskInput("CC#1", "AAPL", "COVERED_CALL", -1, 100, 100.0, 0.30, 0.0)
    b = PositionRiskInput("CC#2", "AAPL", "COVERED_CALL", 1, 100, 100.0, -0.30, 0.0)
    assert a.exposure == pytest.approx(-3_000.0)
    assert b.exposure == pytest.approx(-3_000.0)
    # -3000 * 0.01 = -30 on the +1% day
    assert position_pnl_series(a, MATRIX)[0] == pytest.approx(-30.0)


def test_scaling_invariant_k_times_quantity():
    # contract §3.4: k * quantity => k * pnl exactly (linear estimator)
    p1 = stock("A#1", "AAPL", 10, 200.0)
    p5 = stock("A#5", "AAPL", 50, 200.0)
    s1 = position_pnl_series(p1, MATRIX)
    s5 = position_pnl_series(p5, MATRIX)
    assert s5 == pytest.approx([5 * v for v in s1], rel=1e-12)


def test_position_return_type_guard():
    pos = stock("AAPL#1", "AAPL", 100, 200.0)
    with pytest.raises(ValueError, match="SIMPLE"):
        position_pnl_series(pos, LOG_MATRIX)


def test_position_missing_ticker_is_key_error():
    pos = stock("TSLA#1", "TSLA", 10, 300.0)
    with pytest.raises(KeyError):
        position_pnl_series(pos, MATRIX)


def test_position_input_validation():
    with pytest.raises(ValueError, match="multiplier"):
        PositionRiskInput("k", "AAPL", "LONG_STOCK", 1, 0, 100.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="spot"):
        PositionRiskInput("k", "AAPL", "LONG_STOCK", 1, 1, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="spot"):
        PositionRiskInput("k", "AAPL", "LONG_STOCK", 1, 1, math.inf, 1.0, 0.0)
    with pytest.raises(ValueError, match="delta"):
        PositionRiskInput("k", "AAPL", "LONG_STOCK", 1, 1, 100.0, math.nan, 0.0)
    with pytest.raises(ValueError, match="max_loss"):
        PositionRiskInput("k", "AAPL", "LONG_STOCK", 1, 1, 100.0, 1.0, math.inf)


# ---------------------------------------------------------------------------
# book_pnl_series
# ---------------------------------------------------------------------------


def test_book_total_is_fsum_of_parts_and_dates_from_matrix():
    # AAPL 100 sh @200 -> [200, -400, 100]
    # MSFT -50 sh @400 -> exposure -20000; MSFT r [-0.01, 0.03, 0] -> [200, -600, 0]
    # AAPL call 4x d0.62 @200 -> [496, -992, 248]
    # total = [896, -1992, 348]
    positions = [
        stock("AAPL#1", "AAPL", 100, 200.0),
        stock("MSFT#1", "MSFT", -50, 400.0),
        PositionRiskInput("AAPL#C", "AAPL", "LONG_CALL", 4, 100, 200.0, 0.62, 2_400.0),
    ]
    book = book_pnl_series(positions, MATRIX)
    assert isinstance(book, BookPnl)
    assert book.method == METHOD_DELTA_LINEAR == "DELTA_LINEAR"
    assert book.dates == (D1, D2, D3)
    assert book.n_obs == 3 and book.as_of == D3
    assert set(book.per_position) == {"AAPL#1", "MSFT#1", "AAPL#C"}
    assert book.per_position["MSFT#1"] == pytest.approx([200.0, -600.0, 0.0], abs=1e-9)
    assert book.total == pytest.approx([896.0, -1992.0, 348.0], abs=1e-9)
    for t in range(3):
        parts = [s[t] for s in book.per_position.values()]
        assert book.total[t] == math.fsum(parts)
    assert book.tickers_missing == ()
    assert book.keys_excluded == ()


def test_book_missing_ticker_excluded_and_named_not_zero_filled():
    positions = [
        stock("AAPL#1", "AAPL", 100, 200.0),
        stock("TSLA#1", "TSLA", 10, 300.0),
        stock("NVDA#1", "NVDA", 5, 100.0),
        stock("TSLA#2", "TSLA", 20, 300.0),
    ]
    book = book_pnl_series(positions, MATRIX)
    assert set(book.per_position) == {"AAPL#1"}  # excluded, not zero rows
    assert book.tickers_missing == ("NVDA", "TSLA")  # sorted, distinct
    assert book.keys_excluded == ("TSLA#1", "NVDA#1", "TSLA#2")  # input order
    # total is only the included AAPL leg: [200, -400, 100]
    assert book.total == pytest.approx([200.0, -400.0, 100.0], abs=1e-9)


def test_book_all_missing_or_empty_gives_zero_total_of_matrix_length():
    book = book_pnl_series([stock("TSLA#1", "TSLA", 10, 300.0)], MATRIX)
    assert book.per_position == {}
    assert book.total == [0.0, 0.0, 0.0]
    assert book.tickers_missing == ("TSLA",)
    empty = book_pnl_series([], MATRIX)
    assert empty.total == [0.0, 0.0, 0.0] and empty.tickers_missing == ()


def test_book_return_type_guard_and_duplicate_keys():
    with pytest.raises(ValueError, match="SIMPLE"):
        book_pnl_series([stock("AAPL#1", "AAPL", 1, 200.0)], LOG_MATRIX)
    with pytest.raises(ValueError, match="unique"):
        book_pnl_series(
            [stock("AAPL#1", "AAPL", 1, 200.0), stock("AAPL#1", "AAPL", 2, 200.0)],
            MATRIX,
        )


def test_book_scaling_invariant_k_times_book():
    positions = [
        stock("AAPL#1", "AAPL", 100, 200.0),
        stock("MSFT#1", "MSFT", -50, 400.0),
    ]
    scaled = [
        stock("AAPL#1", "AAPL", 300, 200.0),
        stock("MSFT#1", "MSFT", -150, 400.0),
    ]
    b1 = book_pnl_series(positions, MATRIX)
    b3 = book_pnl_series(scaled, MATRIX)
    assert b3.total == pytest.approx([3 * v for v in b1.total], rel=1e-12)


# ---------------------------------------------------------------------------
# Compliance batch 2 — FULL_REVAL_CONST_IV (design §10.1, §10.4 invariants)
# ---------------------------------------------------------------------------


def _old_delta_linear_series(pos, returns):
    """The PRE-BATCH estimator, transcribed verbatim from the Phase B
    contract §2.9 so the byte-identity claim is checked against an
    independent copy rather than against the function under test."""
    exposure = pos.quantity * pos.multiplier * pos.delta * pos.spot
    return [exposure * r for r in returns.column(pos.ticker)]


def call(key="AAPL#9", qty=1, *, iv0=0.30, t_years=0.25, strike=100.0,
         spot=100.0, mark0=6.2, right="C", delta=0.55):
    """A long call carrying its five design §10.1 leg fields."""
    return PositionRiskInput(
        key=key, ticker="AAPL", instrument="LONG_CALL", quantity=qty,
        multiplier=100, spot=spot, delta=delta, max_loss=abs(qty) * mark0 * 100,
        strike=strike, right=right, t_years=t_years, iv0=iv0, mark0=mark0,
    )


def _symmetric_matrix(r: float, ticker: str = "AAPL"):
    """A three-row matrix: 0, +r, -r — the §10.4 convexity grid."""
    return ReturnMatrix(
        dates=(D1, D2, D3), tickers=(ticker,),
        rows=((0.0,), (r,), (-r,)), return_type="SIMPLE",
    )


# --- (1) stock-only inputs are byte-identical to the old function ----------


def test_stock_only_output_is_byte_identical_to_the_pre_batch_estimator():
    """§10.4's strongest pin: a stock row never reaches the revaluation
    branch, so EVERY float must compare EQUAL (not approx) to the old
    transcribed estimator — per position AND on the book total."""
    positions = [
        stock("AAPL#1", "AAPL", 100, 200.0),
        stock("MSFT#1", "MSFT", -50, 400.0),
        stock("AAPL#2", "AAPL", 7, 133.37),
    ]
    for pos in positions:
        assert position_pnl_series(pos, MATRIX) == _old_delta_linear_series(
            pos, MATRIX
        )

    book = book_pnl_series(positions, MATRIX)
    expected_total = [
        math.fsum(_old_delta_linear_series(p, MATRIX)[t] for p in positions)
        for t in range(MATRIX.n_obs)
    ]
    assert book.total == expected_total  # exact ==, not approx
    for pos in positions:
        assert book.per_position[pos.key] == _old_delta_linear_series(pos, MATRIX)
    # A stock-only book keeps the old label, so a persisted row and a served
    # chip are unchanged for every user who holds no options.
    assert book.method == METHOD_DELTA_LINEAR
    assert set(book.method_by_key.values()) == {METHOD_DELTA_LINEAR}


def test_option_without_leg_fields_is_byte_identical_too():
    """The fallback path is the OLD path — an option row whose chain gave no
    IV must not shift by an ulp just because the branch exists."""
    no_iv = PositionRiskInput(
        key="AAPL#3", ticker="AAPL", instrument="LONG_CALL", quantity=2,
        multiplier=100, spot=100.0, delta=0.55, max_loss=1240.0,
    )
    assert position_pnl_series(no_iv, MATRIX) == _old_delta_linear_series(
        no_iv, MATRIX
    )
    assert no_iv.pnl_method == METHOD_DELTA_LINEAR
    assert no_iv.can_full_reval is False


# --- (2) convexity on a symmetric grid -------------------------------------


def test_long_call_is_convex_in_r_on_a_symmetric_grid():
    """§10.4: pnl(+r) + pnl(−r) > 0 for a long option — the gamma the
    DELTA_LINEAR series is blind to. The linear twin sums to exactly 0."""
    pos = call()
    for r in (0.005, 0.01, 0.03, 0.05):
        series = position_pnl_series(pos, _symmetric_matrix(r))
        assert series[0] == 0.0  # r = 0 ⇒ EXACTLY zero (the basis cancels)
        assert series[1] + series[2] > 0.0, r
        # the delta-linear estimator on the same grid is exactly symmetric
        linear = _old_delta_linear_series(pos, _symmetric_matrix(r))
        assert linear[1] + linear[2] == pytest.approx(0.0, abs=1e-12)


def test_long_put_is_convex_and_short_call_is_concave():
    """Convexity is a property of being LONG the option, not of the right:
    a long put is convex too, and a SHORT call (negative quantity) has the
    mirrored, NEGATIVE curvature — the sign comes from quantity alone."""
    long_put = call(right="P", strike=100.0, delta=-0.45, mark0=5.1)
    short_call = call(qty=-1)
    grid = _symmetric_matrix(0.03)
    put_series = position_pnl_series(long_put, grid)
    short_series = position_pnl_series(short_call, grid)
    assert put_series[1] + put_series[2] > 0.0
    assert short_series[1] + short_series[2] < 0.0
    # the short leg is exactly minus the long leg of the same contract
    long_call_series = position_pnl_series(call(qty=1), grid)
    assert short_series == pytest.approx(
        [-v for v in long_call_series], rel=1e-12
    )


def test_book_mean_pnl_over_a_symmetric_grid_is_positive_for_a_long_book():
    """§10.4 as written: a LONG option book's mean P&L over a symmetric ±r
    grid is > 0 — a statement about the book total, not one position."""
    grid = _symmetric_matrix(0.04)
    book = book_pnl_series(
        [call("AAPL#1", 3), call("AAPL#2", 1, strike=105.0, delta=0.38, mark0=3.7)],
        grid,
    )
    assert math.fsum(book.total) / len(book.total) > 0.0


# --- (3) first-order agreement with DELTA_LINEAR as r -> 0 -----------------


def test_first_order_agreement_with_delta_linear_at_tiny_r():
    """§10.4: at r = 1e-6 the revalued P&L matches delta × spot × qty ×
    multiplier × r to rel 1e-3. The delta MUST be the model's own — the
    chain delta of a different contract would not be the derivative of the
    function being differenced, and the test would be checking nothing."""
    from libs.trading_core.options.bs import bs_greeks

    eps = 1e-6
    for strike, iv0, t_years, right in (
        (100.0, 0.30, 0.25, "C"),
        (110.0, 0.45, 0.08, "C"),
        (95.0, 0.22, 1.00, "P"),
    ):
        greeks = bs_greeks(100.0, strike, t_years, iv0, right)
        pos = call(
            strike=strike, iv0=iv0, t_years=t_years, right=right,
            delta=greeks.delta, mark0=max(greeks.price, 0.01),
        )
        grid = ReturnMatrix(
            dates=(D1,), tickers=("AAPL",), rows=((eps,),), return_type="SIMPLE"
        )
        revalued = position_pnl_series(pos, grid)[0]
        linear = _old_delta_linear_series(pos, grid)[0]
        assert revalued == pytest.approx(linear, rel=1e-3), (strike, right)


# --- (4) method_by_key labelling, including the no-iv0 fallback ------------


def test_method_by_key_labels_every_included_row_and_summarises_the_book():
    """§10.3: per-key truth beside a book-level summary; the no-iv0 option
    is labelled DELTA_LINEAR rather than swept into the FULL_REVAL claim."""
    no_iv = PositionRiskInput(
        key="AAPL#no_iv", ticker="AAPL", instrument="LONG_CALL", quantity=1,
        multiplier=100, spot=100.0, delta=0.55, max_loss=620.0,
        strike=100.0, right="C", t_years=0.25, iv0=None, mark0=6.2,
    )
    positions = [stock("AAPL#1", "AAPL", 100, 200.0), call("AAPL#2"), no_iv]
    book = book_pnl_series(positions, MATRIX)

    assert book.method_by_key == {
        "AAPL#1": METHOD_DELTA_LINEAR,
        "AAPL#2": METHOD_FULL_REVAL_CONST_IV,
        "AAPL#no_iv": METHOD_DELTA_LINEAR,
    }
    # The book-level summary is the STRONGER label when anything revalued.
    assert book.method == METHOD_FULL_REVAL_CONST_IV
    assert book.method_counts == {
        METHOD_FULL_REVAL_CONST_IV: 1, METHOD_DELTA_LINEAR: 2,
    }
    # The fallback row's numbers ARE the old numbers.
    assert book.per_position["AAPL#no_iv"] == _old_delta_linear_series(
        no_iv, MATRIX
    )


@pytest.mark.parametrize(
    "override, why",
    [
        ({"iv0": None}, "no chain IV"),
        ({"strike": None}, "no strike"),
        ({"right": None}, "no right"),
        ({"t_years": None}, "no tenor"),
        ({"mark0": None}, "no mark"),
        ({"t_years": 0.0}, "expired (t_years == 0)"),
        ({"t_years": -0.5}, "past expiry"),
    ],
)
def test_any_missing_or_dead_leg_field_falls_back_and_says_so(override, why):
    """The dispatch predicate is all-five-and-alive: drop ANY field, or let
    the tenor die, and the row is DELTA_LINEAR with the old numbers."""
    fields = dict(
        key="AAPL#1", ticker="AAPL", instrument="LONG_CALL", quantity=1,
        multiplier=100, spot=100.0, delta=0.55, max_loss=620.0,
        strike=100.0, right="C", t_years=0.25, iv0=0.30, mark0=6.2,
    )
    fields.update(override)
    pos = PositionRiskInput(**fields)
    assert pos.can_full_reval is False, why
    assert pos.pnl_method == METHOD_DELTA_LINEAR
    assert position_pnl_series(pos, MATRIX) == _old_delta_linear_series(
        pos, MATRIX
    )
    assert book_pnl_series([pos], MATRIX).method == METHOD_DELTA_LINEAR


def test_book_method_summary_helper_is_the_one_rule():
    """The gateway, the API dict and the persisted column all call this —
    so it is pinned directly, not only through a book."""
    assert book_method_summary({}) == METHOD_DELTA_LINEAR
    assert book_method_summary({"a": METHOD_DELTA_LINEAR}) == METHOD_DELTA_LINEAR
    assert (
        book_method_summary(
            {"a": METHOD_DELTA_LINEAR, "b": METHOD_FULL_REVAL_CONST_IV}
        )
        == METHOD_FULL_REVAL_CONST_IV
    )


def test_excluded_positions_carry_no_method_label():
    """A row nothing priced gets no label — an honest gap, not a claim that
    it was DELTA_LINEAR'd."""
    off_matrix = PositionRiskInput(
        key="TSLA#1", ticker="TSLA", instrument="LONG_CALL", quantity=1,
        multiplier=100, spot=300.0, delta=0.5, max_loss=1000.0,
        strike=300.0, right="C", t_years=0.25, iv0=0.40, mark0=20.0,
    )
    book = book_pnl_series([off_matrix, call("AAPL#1")], MATRIX)
    assert "TSLA#1" in book.keys_excluded
    assert "TSLA#1" not in book.method_by_key
    assert book.method_by_key == {"AAPL#1": METHOD_FULL_REVAL_CONST_IV}


# --- (5) malformed leg data is rejected, never silently downgraded ---------


@pytest.mark.parametrize(
    "override, match",
    [
        ({"strike": 0.0}, "strike"),
        ({"strike": -5.0}, "strike"),
        ({"strike": float("nan")}, "strike"),
        ({"right": "X"}, "right"),
        ({"iv0": 0.0}, "iv0"),
        ({"iv0": -0.2}, "iv0"),
        ({"t_years": float("inf")}, "t_years"),
        ({"mark0": float("nan")}, "mark0"),
    ],
)
def test_malformed_leg_fields_raise_rather_than_fall_back(override, match):
    fields = dict(
        key="AAPL#1", ticker="AAPL", instrument="LONG_CALL", quantity=1,
        multiplier=100, spot=100.0, delta=0.55, max_loss=620.0,
        strike=100.0, right="C", t_years=0.25, iv0=0.30, mark0=6.2,
    )
    fields.update(override)
    with pytest.raises(ValueError, match=match):
        PositionRiskInput(**fields)


# --- (6) the estimator's own arithmetic, hand-checkable --------------------


def test_full_reval_equals_the_documented_bs_difference_and_scales_linearly():
    """The §10.1 formula, recomputed independently: qty × mult × [BS(S(1+r))
    − BS(S)], with mark0 ABSENT from the arithmetic (the basis cancels)."""
    from libs.trading_core.options.bs import bs_price

    pos = call(qty=3)
    grid = _symmetric_matrix(0.02)
    price0 = bs_price(100.0, 100.0, 0.25, 0.30, "C")
    expected = [
        3 * 100 * (bs_price(100.0 * (1.0 + r), 100.0, 0.25, 0.30, "C") - price0)
        for r in (0.0, 0.02, -0.02)
    ]
    assert position_pnl_series(pos, grid) == expected  # exact ==

    # The mark is provenance only: change it and NOTHING moves (the basis
    # would have cancelled either way — this is the Phase D lesson pinned).
    assert position_pnl_series(call(qty=3, mark0=99.0), grid) == expected

    # Contract §3.4 scaling holds for the new estimator too: k × quantity
    # ⇒ k × P&L (the model is priced once, the size multiplies it).
    single = position_pnl_series(call(qty=1), grid)
    assert position_pnl_series(pos, grid) == pytest.approx(
        [3 * v for v in single], rel=1e-12
    )


def test_total_is_still_fsum_of_the_parts_on_a_mixed_book():
    """`total` is method-agnostic — which is why the Euler contribution
    identity survives the mixed book (asserted end-to-end in the snapshot
    tests). Pinned here at the source."""
    positions = [stock("AAPL#1", "AAPL", 100, 200.0), call("AAPL#2", 2),
                 stock("MSFT#1", "MSFT", -50, 400.0)]
    book = book_pnl_series(positions, MATRIX)
    for t in range(MATRIX.n_obs):
        assert book.total[t] == math.fsum(
            series[t] for series in book.per_position.values()
        )


def test_wiped_out_underlying_prices_at_the_option_floor_instead_of_raising():
    """r <= −1 is not a tradable bar (``returns_from_closes`` requires every
    close > 0, so the real pipeline cannot produce one), but a hand-built
    matrix can, and one such row must not destroy the whole book's series.

    The floor is the ANALYTIC LIMIT of ``bs_price`` as spot → 0+, so it is
    CONTINUOUS with the branch beside it: 0 for a call, and the DISCOUNTED
    strike ``K·e^(−rT)`` for a put — never the undiscounted ``K``, which
    would overstate a long put by ``K(1 − e^(−rT))`` (about $199 per
    contract at K=100... $99 here) and make the series jump across r = −1.
    """
    grid = ReturnMatrix(
        dates=(D1, D2), tickers=("AAPL",), rows=((-1.0,), (-1.5,)),
        return_type="SIMPLE",
    )
    from libs.trading_core.options.bs import bs_price
    from libs.trading_core.risk.pnl_series import (
        FULL_REVAL_DIVIDEND_YIELD,
        FULL_REVAL_RATE,
    )

    price0 = bs_price(
        100.0, 100.0, 0.25, 0.30, "C", FULL_REVAL_RATE, FULL_REVAL_DIVIDEND_YIELD
    )
    series = position_pnl_series(call(qty=1), grid)
    assert series == [100 * (0.0 - price0)] * 2  # the long call goes to zero

    put = call(right="P", strike=100.0, delta=-0.45, mark0=5.1)
    put0 = bs_price(
        100.0, 100.0, 0.25, 0.30, "P", FULL_REVAL_RATE, FULL_REVAL_DIVIDEND_YIELD
    )
    floor = 100.0 * math.exp(-FULL_REVAL_RATE * 0.25)
    assert position_pnl_series(put, grid) == [100 * (floor - put0)] * 2

    # CONTINUITY: the floor is the limit, not a separate rule. A bar a hair
    # above r = −1 must price within a cent of the r = −1 bar.
    near = ReturnMatrix(
        dates=(D1,), tickers=("AAPL",), rows=((-1.0 + 1e-9,),),
        return_type="SIMPLE",
    )
    assert position_pnl_series(put, near)[0] == pytest.approx(
        100 * (floor - put0), abs=1e-2
    )
