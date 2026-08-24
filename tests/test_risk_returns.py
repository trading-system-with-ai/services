"""Returns layer tests (risk spec §3; Phase B design contract §2.1).

Every number is hand-checked in a comment. Pins: simple/log arithmetic on
a 5-close vector, the INNER-JOIN-ON-RETURN-DATES alignment rule with a
three-ticker gap (per-ticker returns on that ticker's own consecutive bars,
never compounded across another ticker's gap), strictly-increasing-date
validation, window()/column(), provenance metadata, and contract invariant
§3.7 (``log_returns`` moved from ``correlation.py`` is byte-identical and
still importable from ``correlation``).
"""
import math
from datetime import date

import pytest

from libs.trading_core import correlation
from libs.trading_core.risk.returns import (
    ReturnMatrix,
    ReturnSeries,
    align,
    log_returns,
    returns_from_closes,
    simple_returns,
)

D1, D2, D3, D4, D5 = (
    date(2026, 8, 3),
    date(2026, 8, 4),
    date(2026, 8, 5),
    date(2026, 8, 6),
    date(2026, 8, 7),
)
CLOSES = [100.0, 110.0, 99.0, 108.9, 108.9]


# ---------------------------------------------------------------------------
# simple_returns / log_returns arithmetic
# ---------------------------------------------------------------------------


def test_simple_returns_hand_computed():
    # 110/100-1 = 0.10; 99/110-1 = -0.10; 108.9/99-1 = 0.10; 108.9/108.9-1 = 0
    out = simple_returns(CLOSES)
    assert len(out) == 4
    assert out == pytest.approx([0.10, -0.10, 0.10, 0.0], abs=1e-12)


def test_log_returns_hand_computed():
    # ln(1.1), ln(0.9), ln(1.1), ln(1) = 0
    out = log_returns(CLOSES)
    assert out == pytest.approx(
        [math.log(1.1), math.log(0.9), math.log(1.1), 0.0], abs=1e-12
    )
    # log of a simple return: ln(1 + r_simple) == r_log for each t
    for rs, rl in zip(simple_returns(CLOSES), out):
        assert math.log1p(rs) == pytest.approx(rl, abs=1e-12)


def test_short_and_empty_inputs_yield_empty():
    assert simple_returns([]) == []
    assert simple_returns([100.0]) == []
    assert log_returns([]) == []
    assert log_returns([100.0]) == []


@pytest.mark.parametrize("fn", [simple_returns, log_returns])
def test_nonpositive_close_is_value_error_with_contract_text(fn):
    with pytest.raises(ValueError, match=r"closes must all be > 0, got 0\.0"):
        fn([100.0, 0.0, 110.0])
    with pytest.raises(ValueError, match=r"closes must all be > 0, got -5\.0"):
        fn([100.0, -5.0])


# ---------------------------------------------------------------------------
# Contract invariant §3.7 — log_returns moved, correlation re-exports it
# ---------------------------------------------------------------------------


def test_correlation_log_returns_is_the_same_function():
    assert correlation.log_returns is log_returns
    # identical output on the historic correlation.py test vector
    assert correlation.log_returns([100.0, 110.0, 121.0]) == pytest.approx(
        [math.log(1.1), math.log(1.1)], abs=1e-12
    )
    with pytest.raises(ValueError, match="closes must all be > 0"):
        correlation.log_returns([100.0, 0.0])


# ---------------------------------------------------------------------------
# returns_from_closes
# ---------------------------------------------------------------------------


def test_returns_from_closes_dates_on_later_bar_and_metadata():
    bars = list(zip([D1, D2, D3, D4, D5], CLOSES))
    s = returns_from_closes("AAPL", bars, return_type="SIMPLE")
    assert s.ticker == "AAPL"
    assert s.dates == (D2, D3, D4, D5)  # return t is dated on the LATER bar
    assert s.values == pytest.approx((0.10, -0.10, 0.10, 0.0), abs=1e-12)
    assert s.return_type == "SIMPLE"
    assert s.frequency == "1D"
    assert s.source == "stock_bars_daily"
    assert s.n_obs == 4
    assert s.as_of == D5

    sl = returns_from_closes("AAPL", bars, return_type="LOG")
    assert sl.return_type == "LOG"
    assert sl.values[0] == pytest.approx(math.log(1.1), abs=1e-12)


def test_returns_from_closes_empty_and_single_bar():
    e = returns_from_closes("X", [], return_type="SIMPLE")
    assert e.n_obs == 0 and e.as_of is None and e.dates == ()
    one = returns_from_closes("X", [(D1, 100.0)], return_type="SIMPLE")
    assert one.n_obs == 0 and one.window(5).n_obs == 0


def test_returns_from_closes_rejects_unsorted_and_duplicate_dates():
    with pytest.raises(ValueError, match="strictly increasing"):
        returns_from_closes(
            "X", [(D1, 100.0), (D3, 101.0), (D2, 102.0)], return_type="SIMPLE"
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        returns_from_closes(
            "X", [(D1, 100.0), (D1, 101.0)], return_type="SIMPLE"
        )


def test_returns_from_closes_rejects_bad_close_and_bad_type():
    with pytest.raises(ValueError, match="closes must all be > 0"):
        returns_from_closes("X", [(D1, 100.0), (D2, 0.0)], return_type="LOG")
    with pytest.raises(ValueError, match="return_type"):
        returns_from_closes("X", [(D1, 100.0), (D2, 101.0)], return_type="PCT")


# ---------------------------------------------------------------------------
# ReturnSeries container
# ---------------------------------------------------------------------------


def test_return_series_window_last_n():
    s = ReturnSeries("A", (D2, D3, D4, D5), (0.1, -0.1, 0.1, 0.0), "SIMPLE")
    w = s.window(2)
    assert w.dates == (D4, D5)
    assert w.values == (0.1, 0.0)
    assert w.ticker == "A" and w.return_type == "SIMPLE"
    assert s.window(10).n_obs == 4  # more than available -> everything
    assert s.window(0).n_obs == 0
    with pytest.raises(ValueError):
        s.window(-1)


def test_return_series_validation():
    with pytest.raises(ValueError, match="equal length"):
        ReturnSeries("A", (D2, D3), (0.1,), "SIMPLE")
    with pytest.raises(ValueError, match="strictly increasing"):
        ReturnSeries("A", (D3, D2), (0.1, 0.2), "SIMPLE")
    with pytest.raises(ValueError, match="strictly increasing"):
        ReturnSeries("A", (D2, D2), (0.1, 0.2), "SIMPLE")
    with pytest.raises(ValueError, match="return_type"):
        ReturnSeries("A", (D2,), (0.1,), "ARITH")


# ---------------------------------------------------------------------------
# align — inner join on return dates, never compounded across a gap
# ---------------------------------------------------------------------------


def _three_ticker_series():
    # A: full history D1..D5. Closes 100,110,99,108.9,108.9
    a = returns_from_closes(
        "A", list(zip([D1, D2, D3, D4, D5], CLOSES)), return_type="SIMPLE"
    )
    # B: MISSING D3. Closes D1=50, D2=51, D4=54.06, D5=54.06
    #   B returns (own consecutive bars): D2: 51/50-1 = 0.02;
    #   D4: 54.06/51-1 = 0.06 (spans B's gap D2->D4, its own bar-to-bar);
    #   D5: 0.0
    b = returns_from_closes(
        "B", [(D1, 50.0), (D2, 51.0), (D4, 54.06), (D5, 54.06)],
        return_type="SIMPLE",
    )
    # C: full history, closes 10, 10, 10, 20, 10
    #   returns D2: 0, D3: 0, D4: 1.0, D5: -0.5
    c = returns_from_closes(
        "C", [(D1, 10.0), (D2, 10.0), (D3, 10.0), (D4, 20.0), (D5, 10.0)],
        return_type="SIMPLE",
    )
    return a, b, c


def test_align_inner_join_drops_gap_date_for_all_and_keeps_own_bar_returns():
    a, b, c = _three_ticker_series()
    m = align([a, b, c])
    assert m.tickers == ("A", "B", "C")
    # A has returns on D2,D3,D4,D5; B on D2,D4,D5; C on D2..D5 -> D2,D4,D5
    assert m.dates == (D2, D4, D5)
    assert m.n_obs == 3
    assert m.as_of == D5
    # A on D4 is A's OWN D3->D4 return 108.9/99-1 = 0.10, NOT compounded
    # D2->D4 (108.9/110-1 = -0.01)
    assert m.column("A") == pytest.approx([0.10, 0.10, 0.0], abs=1e-12)
    # B on D4 is B's own D2->D4 return 54.06/51-1 = 0.06
    assert m.column("B") == pytest.approx([0.02, 0.06, 0.0], abs=1e-12)
    assert m.column("C") == pytest.approx([0.0, 1.0, -0.5], abs=1e-12)
    # rows[t][i] layout
    assert m.rows[1] == pytest.approx((0.10, 0.06, 1.0), abs=1e-12)
    assert m.return_type == "SIMPLE"
    assert m.frequency == "1D" and m.source == "stock_bars_daily"


def test_align_single_series_is_identity_and_column_order_is_input_order():
    a, b, c = _three_ticker_series()
    m1 = align([a])
    assert m1.dates == a.dates and m1.column("A") == list(a.values)
    m = align([c, a])
    assert m.tickers == ("C", "A")
    assert m.dates == (D2, D3, D4, D5)
    assert m.rows[2] == pytest.approx((1.0, 0.10), abs=1e-12)


def test_align_no_common_dates_yields_empty_matrix_not_error():
    a = ReturnSeries("A", (D2, D3), (0.1, 0.2), "SIMPLE")
    b = ReturnSeries("B", (D4, D5), (0.1, 0.2), "SIMPLE")
    m = align([a, b])
    assert m.n_obs == 0 and m.rows == () and m.as_of is None
    assert m.tickers == ("A", "B")
    assert m.column("A") == []


def test_align_rejects_empty_duplicates_and_mixed_conventions():
    a, b, c = _three_ticker_series()
    with pytest.raises(ValueError, match="at least one"):
        align([])
    with pytest.raises(ValueError, match="unique"):
        align([a, a])
    a_log = returns_from_closes(
        "A", list(zip([D1, D2, D3, D4, D5], CLOSES)), return_type="LOG"
    )
    with pytest.raises(ValueError, match="mixed return types"):
        align([a_log, b])
    b_weekly = ReturnSeries("B", b.dates, b.values, "SIMPLE", frequency="1W")
    with pytest.raises(ValueError, match="mixed frequencies"):
        align([a, b_weekly])


def test_align_source_provenance_kept_or_joined():
    a, b, _ = _three_ticker_series()
    b2 = ReturnSeries("B", b.dates, b.values, "SIMPLE", source="alpaca_bars")
    assert align([a, b]).source == "stock_bars_daily"
    assert align([a, b2]).source == "alpaca_bars+stock_bars_daily"


# ---------------------------------------------------------------------------
# ReturnMatrix container
# ---------------------------------------------------------------------------


def test_matrix_window_and_column_and_key_error():
    a, b, c = _three_ticker_series()
    m = align([a, b, c])
    w = m.window(2)
    assert w.dates == (D4, D5)
    assert w.column("B") == pytest.approx([0.06, 0.0], abs=1e-12)
    assert w.tickers == m.tickers and w.return_type == "SIMPLE"
    assert m.window(99).n_obs == 3
    assert m.window(0).n_obs == 0
    with pytest.raises(ValueError):
        m.window(-1)
    with pytest.raises(KeyError):
        m.column("ZZZ")


def test_matrix_validation():
    with pytest.raises(ValueError, match="unique"):
        ReturnMatrix((D2,), ("A", "A"), ((0.1, 0.2),), "SIMPLE")
    with pytest.raises(ValueError, match="equal length"):
        ReturnMatrix((D2, D3), ("A",), ((0.1,),), "SIMPLE")
    with pytest.raises(ValueError, match="cells"):
        ReturnMatrix((D2,), ("A", "B"), ((0.1,),), "SIMPLE")
    with pytest.raises(ValueError, match="strictly increasing"):
        ReturnMatrix((D3, D2), ("A",), ((0.1,), (0.2,)), "SIMPLE")
    with pytest.raises(ValueError, match="return_type"):
        ReturnMatrix((D2,), ("A",), ((0.1,),), "PCT")


def test_containers_are_frozen():
    s = ReturnSeries("A", (D2,), (0.1,), "SIMPLE")
    with pytest.raises(Exception):
        s.ticker = "B"  # type: ignore[misc]
    m = ReturnMatrix((D2,), ("A",), ((0.1,),), "SIMPLE")
    with pytest.raises(Exception):
        m.tickers = ("B",)  # type: ignore[misc]
