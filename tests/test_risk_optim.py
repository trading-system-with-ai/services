"""Tests for ``libs/trading_core/risk/optim.py`` — the deterministic
Nelder–Mead simplex minimiser behind the GARCH MLE (Phase E design §9.1).

The two textbook cases the design names:

- a **quadratic** (convex, well conditioned) — the minimiser must land on
  the analytic minimum to the requested tolerance;
- **Rosenbrock** (the classic banana valley) — the hard case for a
  derivative-free method; it must reach ``(1, 1)`` from the standard
  ``(-1.2, 1)`` start.

Plus the properties the risk library depends on: determinism (a persisted
fitted parameter must be reproducible, spec §44), infeasible points
handled as ``+inf`` rather than crashes, and malformed arguments raising.
"""
from __future__ import annotations

import math

import pytest

from libs.trading_core.risk.optim import (
    DEFAULT_MAX_ITER,
    NMResult,
    nelder_mead,
)


# ---------------------------------------------------------------------------
# Quadratic
# ---------------------------------------------------------------------------


def test_quadratic_1d_finds_the_analytic_minimum() -> None:
    """``f(x) = (x - 3)^2 + 7`` has its minimum at ``x = 3``, ``f = 7``."""
    result = nelder_mead(lambda x: (x[0] - 3.0) ** 2 + 7.0, [0.0])
    assert result.converged, result.reason
    assert result.reason == "converged"
    assert result.x[0] == pytest.approx(3.0, abs=1e-6)
    assert result.fval == pytest.approx(7.0, abs=1e-10)
    assert result.n_params == 1


def test_quadratic_2d_shifted_minimum() -> None:
    """``f = (x - 1)^2 + 3(y + 2)^2`` ⇒ minimum ``(1, -2)``, ``f = 0``."""

    def f(v):
        return (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2

    result = nelder_mead(f, [0.0, 0.0])
    assert result.converged, result.reason
    assert result.x[0] == pytest.approx(1.0, abs=1e-6)
    assert result.x[1] == pytest.approx(-2.0, abs=1e-6)
    assert result.fval == pytest.approx(0.0, abs=1e-12)


def test_quadratic_3d_with_coordinates_on_different_scales() -> None:
    """The relative initial step handles a 1e-6-scale coordinate next to
    O(1) ones — the situation the GARCH ``omega`` creates."""

    def f(v):
        return (v[0] - 1e-6) ** 2 / 1e-12 + (v[1] - 0.5) ** 2 + (v[2] + 4.0) ** 2

    result = nelder_mead(f, [2e-6, 0.0, 0.0], tol=1e-12, tol_x=1e-12)
    assert result.converged, result.reason
    assert result.x[0] == pytest.approx(1e-6, rel=1e-4)
    assert result.x[1] == pytest.approx(0.5, abs=1e-6)
    assert result.x[2] == pytest.approx(-4.0, abs=1e-6)


def test_starting_at_the_minimum_converges_immediately() -> None:
    """A start already at the optimum still returns it (the initial simplex
    is evaluated, then the tolerance test passes)."""
    result = nelder_mead(lambda x: x[0] * x[0], [0.0], step=1e-9)
    assert result.converged
    assert result.fval == pytest.approx(0.0, abs=1e-16)


# ---------------------------------------------------------------------------
# Rosenbrock
# ---------------------------------------------------------------------------


def _rosenbrock(v) -> float:
    """``f(x, y) = (1 - x)^2 + 100 (y - x^2)^2``; minimum 0 at ``(1, 1)``."""
    return (1.0 - v[0]) ** 2 + 100.0 * (v[1] - v[0] * v[0]) ** 2


def test_rosenbrock_from_the_standard_start() -> None:
    """From ``(-1.2, 1)`` — the textbook start — Nelder–Mead reaches ``(1, 1)``."""
    result = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, max_iter=4000)
    assert result.converged, result.reason
    assert result.x[0] == pytest.approx(1.0, abs=1e-5)
    assert result.x[1] == pytest.approx(1.0, abs=1e-5)
    assert result.fval == pytest.approx(0.0, abs=1e-10)


def test_rosenbrock_from_the_origin() -> None:
    """A second start reaches the same optimum — not a lucky initial simplex."""
    result = nelder_mead(_rosenbrock, [0.0, 0.0], step=0.5, max_iter=4000)
    assert result.converged, result.reason
    assert result.x[0] == pytest.approx(1.0, abs=1e-5)
    assert result.x[1] == pytest.approx(1.0, abs=1e-5)


def test_rosenbrock_needs_both_convergence_criteria() -> None:
    """The value spread alone is not enough on a flat valley: the simplex
    diameter criterion is what forces the vertices together.

    With ``tol_x`` deliberately huge the run stops early on the value
    criterion alone and lands FAR from ``(1, 1)`` — proving the diameter
    test in the real defaults is load-bearing, not decoration.
    """
    lax = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, tol=1e-2, tol_x=1e9)
    strict = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, max_iter=4000)
    assert lax.converged and strict.converged
    assert abs(lax.x[0] - 1.0) > abs(strict.x[0] - 1.0)


# ---------------------------------------------------------------------------
# Determinism, cost, infeasible regions
# ---------------------------------------------------------------------------


def test_repeated_calls_are_bit_identical() -> None:
    """No randomness anywhere: a persisted fit must be reproducible (spec §44)."""
    a = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, max_iter=4000)
    b = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, max_iter=4000)
    assert a == b
    assert a.x == b.x and a.fval == b.fval and a.n_evals == b.n_evals


def test_max_iter_reached_reports_not_converged_with_the_numbers() -> None:
    """Hitting the cap is honest: ``converged=False`` and a reason quoting
    the achieved spread and diameter (no silent 'best effort ACTIVE')."""
    result = nelder_mead(_rosenbrock, [-1.2, 1.0], step=0.5, max_iter=3)
    assert not result.converged
    assert result.iterations == 3
    assert result.reason.startswith("max_iter reached")
    assert "f_spread=" in result.reason and "simplex_diameter=" in result.reason


def test_n_evals_counts_the_initial_simplex() -> None:
    """The honest cost number includes the ``n + 1`` initial evaluations."""
    result = nelder_mead(lambda v: v[0] ** 2 + v[1] ** 2, [1.0, 1.0], max_iter=1)
    assert result.n_evals >= 3  # x0 plus two displaced vertices


def test_infeasible_points_are_pushed_back_not_crashes() -> None:
    """``inf``/``nan`` outside the feasible region is 'worse than
    everything' — exactly what a constrained likelihood needs.

    ``f`` here is ``(x - 2)^2`` on ``x > 0`` and ``+inf`` elsewhere; the
    simplex must stay inside and find ``x = 2``.
    """

    def f(v):
        if v[0] <= 0.0:
            return math.inf
        return (v[0] - 2.0) ** 2

    result = nelder_mead(f, [0.5], step=0.5)
    assert result.converged, result.reason
    assert result.x[0] == pytest.approx(2.0, abs=1e-6)


def test_nan_is_treated_as_worse_than_everything() -> None:
    def f(v):
        if v[0] < -1.0:
            return float("nan")
        return (v[0] - 1.0) ** 2

    result = nelder_mead(f, [0.0])
    assert result.x[0] == pytest.approx(1.0, abs=1e-6)


def test_infinite_objective_at_x0_raises() -> None:
    """The optimiser cannot start from a point it cannot evaluate — a caller
    bug, not a data gap, so it raises rather than degrading."""
    with pytest.raises(ValueError, match="finite at x0"):
        nelder_mead(lambda v: math.inf, [0.0])


# ---------------------------------------------------------------------------
# Malformed arguments (house rule: bad input raises)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": 0.0},
        {"step": -1.0},
        {"tol": 0.0},
        {"tol_x": -1e-9},
        {"max_iter": 0},
        {"max_iter": 1.5},
    ],
)
def test_malformed_parameters_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        nelder_mead(lambda v: v[0] ** 2, [1.0], **kwargs)


def test_empty_x0_raises() -> None:
    with pytest.raises(ValueError, match="at least one coordinate"):
        nelder_mead(lambda v: 0.0, [])


def test_non_finite_x0_raises() -> None:
    with pytest.raises(ValueError, match="x0\\[1\\] must be finite"):
        nelder_mead(lambda v: v[0] ** 2, [0.0, math.inf])


def test_non_callable_objective_raises() -> None:
    with pytest.raises(ValueError, match="f must be callable"):
        nelder_mead("not a function", [0.0])  # type: ignore[arg-type]


def test_objective_returning_a_non_number_raises() -> None:
    with pytest.raises(ValueError, match="objective must return a real number"):
        nelder_mead(lambda v: "nope", [0.0])  # type: ignore[return-value]


def test_result_is_frozen_and_typed() -> None:
    result = nelder_mead(lambda v: v[0] ** 2, [1.0])
    assert isinstance(result, NMResult)
    assert isinstance(result.x, tuple)
    with pytest.raises(Exception):
        result.fval = 0.0  # type: ignore[misc]


def test_default_max_iter_is_the_documented_two_thousand() -> None:
    """Every threshold is a documented parameter (house rule §9)."""
    assert DEFAULT_MAX_ITER == 2000
