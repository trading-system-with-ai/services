"""Tests for ``libs/trading_core/risk/models/_chi2.py`` — the general χ²
survival function behind the Ljung–Box diagnostic (Phase E design §9.2).

The load-bearing assertions are the two **closed-form pins**: Phase B's
``validation.py`` keeps its exact ``erfc(sqrt(x/2))`` (df=1) and
``exp(-x/2)`` (df=2) formulas — they are exact, cheaper, and changing them
would change persisted Kupiec/Christoffersen p-values (contract §4). The
general implementation must agree with both to 1e-12, which is what makes
it safe to use the SAME function for the df=10 case the closed forms cannot
cover.

Everything else here is a textbook table value or a mathematical property
(monotonicity, complementarity, the boundaries), so a reader can check the
numbers against a χ² table rather than against the code.
"""
from __future__ import annotations

import math

import pytest

from libs.trading_core.risk.models._chi2 import (
    MAX_ITER,
    chi2_cdf,
    chi2_sf,
    regularized_gamma_p,
    regularized_gamma_q,
)
from libs.trading_core.risk.validation import chi2_1_sf, chi2_2_sf

#: A spread of statistics covering the whole useful range: from "obviously
#: not significant" through the 5 % and 1 % critical points into the deep
#: tail where the p-value underflows.
X_GRID = [
    0.0,
    1e-9,
    0.001,
    0.1,
    0.5,
    1.0,
    2.0,
    2.7055,   # χ²(1) 10 % point
    3.8415,   # χ²(1)  5 % point
    5.9915,   # χ²(2)  5 % point
    6.6349,   # χ²(1)  1 % point
    9.2103,   # χ²(2)  1 % point
    10.0,
    18.307,   # χ²(10) 5 % point
    25.0,
    50.0,
    100.0,
    500.0,
]


# ---------------------------------------------------------------------------
# The two closed-form pins (design §9.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", X_GRID)
def test_df1_matches_the_erfc_closed_form_to_1e_12(x: float) -> None:
    """``chi2_sf(x, 1) == erfc(sqrt(x/2))`` — the Kupiec / Christoffersen
    p-value that ``validation.py`` computes in closed form."""
    expected = math.erfc(math.sqrt(x / 2.0))
    actual = chi2_sf(x, 1)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("x", X_GRID)
def test_df2_matches_the_exp_closed_form_to_1e_12(x: float) -> None:
    """``chi2_sf(x, 2) == exp(-x/2)`` — the Jarque–Bera / Christoffersen
    conditional-coverage p-value."""
    expected = math.exp(-x / 2.0)
    actual = chi2_sf(x, 2)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("x", X_GRID)
def test_the_phase_b_helpers_agree_with_the_general_function(x: float) -> None:
    """The Phase B helpers themselves — not just their formulas — stay
    consistent with the general implementation, so a future reader can trust
    either entry point."""
    assert chi2_1_sf(x) == pytest.approx(chi2_sf(x, 1), rel=1e-12, abs=1e-300)
    assert chi2_2_sf(x) == pytest.approx(chi2_sf(x, 2), rel=1e-12, abs=1e-300)


# ---------------------------------------------------------------------------
# Table values for the df the closed forms cannot cover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, df, expected",
    [
        (18.307, 10, 0.05),    # design §9.2 names this one explicitly
        (23.209, 10, 0.01),
        (15.987, 10, 0.10),
        (3.940, 10, 0.95),
        (16.919, 9, 0.05),
        (11.070, 5, 0.05),
        (15.086, 5, 0.01),
        (12.592, 6, 0.05),
        (31.410, 20, 0.05),
        (124.342, 100, 0.05),
    ],
)
def test_textbook_critical_points(x: float, df: int, expected: float) -> None:
    """Standard χ² table values, matched to 5e-4 (the tables' own precision)."""
    assert chi2_sf(x, df) == pytest.approx(expected, abs=5e-4)


def test_the_design_document_example_exactly() -> None:
    """``sf(18.307, 10) ≈ 0.05`` within 5e-4 (design §9.2)."""
    assert abs(chi2_sf(18.307, 10) - 0.05) < 5e-4


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("df", [1, 2, 3, 5, 10, 20, 100])
def test_monotone_non_increasing_in_x(df: int) -> None:
    """A larger test statistic can never be MORE probable under the null."""
    previous = chi2_sf(X_GRID[0], df)
    for x in X_GRID[1:]:
        current = chi2_sf(x, df)
        assert current <= previous + 1e-15, f"df={df}, x={x}"
        previous = current


@pytest.mark.parametrize("x", [0.5, 2.0, 5.0, 10.0, 18.307])
def test_monotone_non_decreasing_in_df(x: float) -> None:
    """For a FIXED statistic, more degrees of freedom means a larger tail."""
    previous = chi2_sf(x, 1)
    for df in (2, 3, 5, 10, 20):
        current = chi2_sf(x, df)
        assert current >= previous - 1e-15, f"x={x}, df={df}"
        previous = current


@pytest.mark.parametrize("df", [1, 2, 3, 7, 10, 30])
@pytest.mark.parametrize("x", [0.0, 0.5, 3.0, 12.0, 40.0])
def test_cdf_and_sf_are_complementary(x: float, df: int) -> None:
    assert chi2_cdf(x, df) + chi2_sf(x, df) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("df", [1, 2, 5, 10])
def test_boundaries(df: int) -> None:
    """``x = 0`` ⇒ the whole mass is in the tail; a huge ``x`` underflows to
    0.0 rather than raising."""
    assert chi2_sf(0.0, df) == 1.0
    assert chi2_cdf(0.0, df) == 0.0
    assert chi2_sf(1e6, df) == 0.0
    assert 0.0 <= chi2_sf(1e6, df) <= 1.0


@pytest.mark.parametrize("df", [1, 2, 3, 10, 25])
def test_values_stay_in_the_unit_interval(df: int) -> None:
    for x in X_GRID:
        p = chi2_sf(x, df)
        assert 0.0 <= p <= 1.0, f"df={df}, x={x}, p={p}"


def test_median_of_chi2_1_is_the_known_0_4549() -> None:
    """A well-known landmark: the χ²(1) median is ≈ 0.4549 (sf = 0.5)."""
    assert chi2_sf(0.454936, 1) == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# The gamma primitives themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 12.5])
@pytest.mark.parametrize("x", [0.0, 0.25, 1.0, 4.0, 20.0])
def test_gamma_p_and_q_are_complementary(a: float, x: float) -> None:
    assert regularized_gamma_p(a, x) + regularized_gamma_q(a, x) == pytest.approx(
        1.0, abs=1e-13
    )


def test_gamma_p_at_a_one_is_the_exponential_cdf() -> None:
    """``P(1, x) = 1 - e^{-x}`` exactly — the a=1 special case."""
    for x in (0.0, 0.1, 1.0, 5.0, 30.0):
        assert regularized_gamma_p(1.0, x) == pytest.approx(
            1.0 - math.exp(-x), rel=1e-13, abs=1e-300
        )


def test_both_expansion_branches_are_exercised_and_agree_at_the_seam() -> None:
    """The series is used for ``x < a + 1`` and the continued fraction above
    it; they must join smoothly at the switch point."""
    a = 5.0
    below = regularized_gamma_q(a, a + 1.0 - 1e-9)
    above = regularized_gamma_q(a, a + 1.0 + 1e-9)
    assert below == pytest.approx(above, rel=1e-8)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_tiny_negative_x_from_rounding_is_clamped_to_zero() -> None:
    """Matches ``validation.chi2_1_sf``'s existing tolerance: a statistic
    that rounds to -1e-12 is 0, not an error."""
    assert chi2_sf(-1e-12, 1) == 1.0


def test_materially_negative_x_raises() -> None:
    with pytest.raises(ValueError, match="x must be >= 0"):
        chi2_sf(-0.5, 1)


@pytest.mark.parametrize("df", [0, -1, 1.5, True])
def test_bad_df_raises(df) -> None:
    with pytest.raises(ValueError, match="df must be an int >= 1"):
        chi2_sf(1.0, df)


@pytest.mark.parametrize("a", [0.0, -1.0, float("nan")])
def test_bad_a_raises(a: float) -> None:
    with pytest.raises(ValueError):
        regularized_gamma_p(a, 1.0)


def test_non_finite_x_raises() -> None:
    with pytest.raises(ValueError, match="x must be a finite number"):
        chi2_sf(float("inf"), 1)


def test_max_iter_is_a_documented_parameter() -> None:
    """Every threshold is a documented parameter (house rule §9)."""
    assert MAX_ITER == 1000
