"""NAV drawdown tests (Phase B contract §2.8).

Every number is hand-computed in a comment. The canonical 6-point path::

    idx  date        nav    running_max   dd = nav/max - 1
    0    2026-01-02  100.0  100.0          0.0
    1    2026-01-05  105.0  105.0          0.0
    2    2026-01-06  110.0  110.0          0.0        <- peak
    3    2026-01-07   99.0  110.0         99/110-1  = -0.10
    4    2026-01-08   88.0  110.0         88/110-1  = -0.20   <- trough (max dd)
    5    2026-01-09   99.0  110.0         99/110-1  = -0.10   <- current

  max_dd_pct     = -0.20 exactly (88/110 = 0.8)
  current_dd_pct = -0.10 exactly (99/110 = 0.9)
  peak_nav 110.0 on 2026-01-06; trough 2026-01-08; current_nav 99.0
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.drawdown import (
    METHOD_NAV_PATH,
    METHOD_RECONSTRUCTED,
    DrawdownParams,
    drawdown,
    reconstructed_book_drawdown,
)

PATH: list[tuple[date, float]] = [
    (date(2026, 1, 2), 100.0),
    (date(2026, 1, 5), 105.0),
    (date(2026, 1, 6), 110.0),
    (date(2026, 1, 7), 99.0),
    (date(2026, 1, 8), 88.0),
    (date(2026, 1, 9), 99.0),
]


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


def test_six_point_path_hand_computed() -> None:
    r = drawdown(PATH)

    # 88/110 - 1 = 0.8 - 1 = -0.2 exactly (both are exact binary-ish ratios
    # only to within float epsilon, so compare with a tight tolerance).
    assert r.max_dd_pct == pytest.approx(-0.20, abs=1e-12)
    # 99/110 - 1 = 0.9 - 1 = -0.1
    assert r.current_dd_pct == pytest.approx(-0.10, abs=1e-12)
    assert r.peak_nav == 110.0
    assert r.peak_date == date(2026, 1, 6)
    assert r.trough_date == date(2026, 1, 8)
    assert r.current_nav == 99.0
    assert r.n_obs == 6
    assert r.method == METHOD_NAV_PATH
    # n=6 >= degraded_multiple(2.0) * min_obs(2) = 4  -> ACTIVE
    assert r.health is ModelHealth.ACTIVE
    assert r.reason is None
    assert r.is_available is True
    assert r.meta.as_of == date(2026, 1, 9)
    assert r.meta.model_version == "1.0.0"


def test_max_dd_is_measured_from_the_peak_in_force_at_the_trough() -> None:
    # The identity that makes peak_date/peak_nav meaningful:
    #   max_dd == nav_at_trough / peak_nav - 1
    r = drawdown(PATH)
    nav_at_trough = dict(PATH)[r.trough_date]  # 88.0
    assert nav_at_trough == 88.0
    assert r.max_dd_pct == pytest.approx(nav_at_trough / r.peak_nav - 1.0, abs=1e-15)


def test_drawdowns_are_never_positive_and_flat_path_is_zero() -> None:
    # Strictly rising: every nav IS its own running max -> dd_t == 0 for all t.
    rising = [(date(2026, 2, d), 100.0 + d) for d in (2, 3, 4, 5)]
    r = drawdown(rising)
    assert r.max_dd_pct == 0.0
    assert r.current_dd_pct == 0.0
    assert r.health is ModelHealth.ACTIVE  # a true 0.0, not a missing value
    assert r.is_available is True

    # Flat: 100/100 - 1 = 0 everywhere.
    flat = [(date(2026, 2, d), 100.0) for d in (2, 3, 4, 5)]
    assert drawdown(flat).max_dd_pct == 0.0


def test_recovery_to_a_new_peak_resets_current_but_not_max() -> None:
    #  100, 80, 130 ->  dd: 0, 80/100-1=-0.2, 130 is a NEW max so dd=0
    path = [
        (date(2026, 3, 2), 100.0),
        (date(2026, 3, 3), 80.0),
        (date(2026, 3, 4), 130.0),
        (date(2026, 3, 5), 130.0),
    ]
    r = drawdown(path)
    assert r.max_dd_pct == pytest.approx(-0.2, abs=1e-15)
    assert r.current_dd_pct == 0.0          # fully recovered
    assert r.trough_date == date(2026, 3, 3)
    assert r.peak_nav == 100.0              # the peak the -20% was measured FROM
    assert r.peak_date == date(2026, 3, 2)


def test_tie_on_max_drawdown_resolves_to_the_earliest_date() -> None:
    # 100, 90, 90 -> dd = 0, -0.1, -0.1. Tie: earliest (2026-04-03) wins.
    path = [
        (date(2026, 4, 2), 100.0),
        (date(2026, 4, 3), 90.0),
        (date(2026, 4, 6), 90.0),
    ]
    r = drawdown(path)
    assert r.max_dd_pct == pytest.approx(-0.1, abs=1e-15)
    assert r.trough_date == date(2026, 4, 3)


# ---------------------------------------------------------------------------
# Health / honest nulls (contract §1, §3.6)
# ---------------------------------------------------------------------------


def test_n_below_two_is_unavailable_not_zero() -> None:
    for short in ([], [(date(2026, 1, 2), 100.0)]):
        r = drawdown(short)
        assert r.health is ModelHealth.UNAVAILABLE
        assert r.max_dd_pct is None       # never a fabricated 0.0
        assert r.current_dd_pct is None
        assert r.peak_date is None and r.trough_date is None
        assert r.peak_nav is None and r.current_nav is None
        assert r.is_available is False
        assert r.reason and f"min_obs=2" in r.reason
        assert f"n={len(short)}" in r.reason


def test_short_path_is_degraded_with_real_numbers() -> None:
    # n=2 and n=3 are < 2.0 * 2 = 4  -> DEGRADED but still a number.
    r = drawdown(PATH[:3])
    assert r.health is ModelHealth.DEGRADED
    assert r.max_dd_pct == 0.0            # 100,105,110 only rises
    assert r.reason and "n=3" in r.reason


def test_min_obs_is_a_parameter() -> None:
    strict = DrawdownParams(min_obs=10)
    r = drawdown(PATH, params=strict)     # n=6 < 10
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.reason == "n=6 < min_obs=10"


# ---------------------------------------------------------------------------
# Malformed input -> ValueError (contract §1: only malformed input raises)
# ---------------------------------------------------------------------------


def test_malformed_input_raises() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        drawdown([(date(2026, 1, 5), 100.0), (date(2026, 1, 2), 90.0)])
    with pytest.raises(ValueError, match="strictly increasing"):  # duplicate date
        drawdown([(date(2026, 1, 5), 100.0), (date(2026, 1, 5), 90.0)])
    with pytest.raises(ValueError, match="> 0"):
        drawdown([(date(2026, 1, 2), 100.0), (date(2026, 1, 5), 0.0)])
    with pytest.raises(ValueError, match="> 0"):
        drawdown([(date(2026, 1, 2), 100.0), (date(2026, 1, 5), -5.0)])
    with pytest.raises(ValueError, match="finite"):
        drawdown([(date(2026, 1, 2), 100.0), (date(2026, 1, 5), math.nan)])
    with pytest.raises(ValueError, match="datetime.date"):
        drawdown([("2026-01-02", 100.0), (date(2026, 1, 5), 90.0)])


# ---------------------------------------------------------------------------
# Reconstructed book drawdown (labelled, NOT account history)
# ---------------------------------------------------------------------------


def test_reconstructed_path_rebuilds_nav_and_is_labelled() -> None:
    # pnl of today's book; anchor nav_now = 99.0 at the LAST date.
    # Walking back:  nav_5 = 99
    #                nav_4 = 99 - 11   = 88     (pnl_5 = +11)
    #                nav_3 = 88 - (-11)= 99     (pnl_4 = -11)
    #                nav_2 = 99 - (-11)= 110    (pnl_3 = -11)
    #                nav_1 = 110 - 5   = 105    (pnl_2 = +5)
    #                nav_0 = 105 - 5   = 100    (pnl_1 = +5)
    # -> exactly the PATH above, so the same -0.2 / -0.1 must come out.
    pnl = [
        (date(2026, 1, 2), 0.0),    # first point: anchors nav_0 = 100
        (date(2026, 1, 5), 5.0),
        (date(2026, 1, 6), 5.0),
        (date(2026, 1, 7), -11.0),
        (date(2026, 1, 8), -11.0),
        (date(2026, 1, 9), 11.0),
    ]
    r = reconstructed_book_drawdown(pnl, 99.0)
    assert r.method == METHOD_RECONSTRUCTED          # the honest label
    assert r.meta.params["method"] == "RECONSTRUCTED_CURRENT_BOOK"
    assert r.meta.model_name == "reconstructed_book_drawdown"
    assert r.max_dd_pct == pytest.approx(-0.20, abs=1e-12)
    assert r.current_dd_pct == pytest.approx(-0.10, abs=1e-12)
    assert r.peak_nav == pytest.approx(110.0, abs=1e-12)
    assert r.current_nav == pytest.approx(99.0, abs=1e-12)
    assert r.trough_date == date(2026, 1, 8)
    assert r.n_obs == 6


def test_reconstructed_nav_differences_equal_pnl_exactly() -> None:
    # The construction guarantees nav_t - nav_{t-1} == pnl_t.
    pnl = [(date(2026, 5, d), v) for d, v in ((4, 0.0), (5, 3.0), (6, -7.0), (7, 2.0))]
    r = reconstructed_book_drawdown(pnl, 1000.0)
    # nav_3 = 1000, nav_2 = 998, nav_1 = 1005, nav_0 = 1002
    # running max = 1002, 1005, 1005, 1005
    # dd = 0, 0, 998/1005-1, 1000/1005-1 ; max dd = 998/1005 - 1
    assert r.max_dd_pct == pytest.approx(998.0 / 1005.0 - 1.0, abs=1e-15)
    assert r.current_dd_pct == pytest.approx(1000.0 / 1005.0 - 1.0, abs=1e-15)
    assert r.current_nav == pytest.approx(1000.0, abs=1e-12)


def test_reconstructed_accepts_plain_floats_with_dates() -> None:
    dates = [date(2026, 6, d) for d in (1, 2, 3)]
    a = reconstructed_book_drawdown([0.0, -10.0, 4.0], 94.0, dates=dates)
    b = reconstructed_book_drawdown(list(zip(dates, [0.0, -10.0, 4.0])), 94.0)
    assert a.max_dd_pct == pytest.approx(b.max_dd_pct, abs=1e-15)
    # nav_2 = 94, nav_1 = 90, nav_0 = 100 -> dd = 0, -0.1, 94/100-1 = -0.06
    assert a.max_dd_pct == pytest.approx(-0.10, abs=1e-15)
    assert a.current_dd_pct == pytest.approx(-0.06, abs=1e-15)


def test_reconstructed_honest_nulls_and_malformed() -> None:
    r = reconstructed_book_drawdown([(date(2026, 1, 2), 1.0)], 100.0)
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.max_dd_pct is None
    assert r.method == METHOD_RECONSTRUCTED       # label survives the null

    with pytest.raises(ValueError, match="nav_now"):
        reconstructed_book_drawdown([(date(2026, 1, 2), 1.0)], 0.0)
    with pytest.raises(ValueError, match="len"):
        reconstructed_book_drawdown([1.0, 2.0], 100.0, dates=[date(2026, 1, 2)])
    # Cumulative P&L exceeding nav_now would imply a non-positive past NAV.
    with pytest.raises(ValueError, match="<= 0"):
        reconstructed_book_drawdown(
            [(date(2026, 1, 2), 0.0), (date(2026, 1, 5), 500.0)], 100.0
        )
