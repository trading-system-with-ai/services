"""Nelder–Mead simplex minimiser — the optimiser behind the GARCH(1,1) MLE
(Phase E design §9.1; risk spec §12).

Pure stdlib, deterministic, no numpy/scipy (house rule). There is **no
randomness anywhere**: the initial simplex is built deterministically from
``x0`` and ``step``, every tie in the ordering is broken by index, and the
same inputs always produce the same output — a requirement for a risk model
whose fitted parameters are persisted and must be reproducible (spec §44).

The algorithm (Nelder & Mead 1965, standard coefficients)
---------------------------------------------------------
Minimise ``f: R^n -> R`` over a simplex of ``n + 1`` vertices.

Initial simplex: ``x0`` plus, for each coordinate ``i``, a vertex with
``x[i]`` displaced by ``step_i``. The displacement is *relative* where the
coordinate is non-zero (``step_i = step * |x0_i|``) and absolute where it is
zero (``step_i = step``), so a parameter of order 1e-6 and one of order 1
are perturbed on their own scales. This matters for the GARCH objective,
whose ``omega`` is ~1e-6 while ``alpha``/``beta`` are ~0.1–0.9 — although in
practice that objective is optimised in an unconstrained transformed space
where all coordinates are O(1).

Each iteration, with vertices sorted best (lowest ``f``) to worst:

1. **Reflect** the worst vertex through the centroid of the other ``n``:
   ``xr = xc + alpha_r (xc - x_worst)``, ``alpha_r = 1``. Accept if it is at
   least as good as the second-worst but not better than the best.
2. **Expand** (``gamma = 2``) when the reflection is the new best: try
   ``xe = xc + gamma (xr - xc)`` and keep whichever of ``xe``/``xr`` is
   lower.
3. **Contract** (``rho = 0.5``) when the reflection is still worse than the
   second-worst: outside contraction ``xc + rho (xr - xc)`` when ``f(xr)``
   beats the worst, inside contraction ``xc + rho (x_worst - xc)``
   otherwise. Accept if it improves on the relevant reference point.
4. **Shrink** (``sigma = 0.5``) toward the best vertex when nothing else
   helped.

Convergence (both criteria, standard): the simplex is converged when the
spread of function values ``f_worst - f_best`` is ``<= tol * (|f_best| +
tol)`` **and** the largest vertex-to-best coordinate distance is
``<= tol_x``. Requiring both prevents declaring victory on a long flat
valley (Rosenbrock) where the values agree but the vertices do not.

Non-finite objective values (``nan``/``inf``) are not an error: they are
treated as ``+inf`` (worse than everything), which is exactly what a
constrained likelihood needs when a trial point leaves the feasible region.
A ``+inf`` at ``x0`` itself, however, is a caller bug and raises — the
optimiser cannot start from a point it cannot evaluate.

Nothing here is a risk model: it holds no thresholds and makes no decision.
It is a numerical utility used by ``risk/models/garch.py`` (RESEARCH).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Standard Nelder–Mead coefficients (documented parameters, not magic).
REFLECTION = 1.0
EXPANSION = 2.0
CONTRACTION = 0.5
SHRINK = 0.5

#: Documented defaults (design §9.1). Every one is a keyword parameter.
DEFAULT_STEP = 0.1          # initial simplex displacement (relative where possible)
DEFAULT_TOL = 1e-8          # convergence tolerance on the spread of f values
DEFAULT_TOL_X = 1e-8        # convergence tolerance on the simplex diameter
DEFAULT_MAX_ITER = 2000     # iteration cap; hitting it ⇒ converged=False

#: ``reason`` strings (stable — callers/tests match on them).
REASON_CONVERGED = "converged"
REASON_MAX_ITER = "max_iter reached"


@dataclass(frozen=True)
class NMResult:
    """Outcome of :func:`nelder_mead`.

    - ``x``: the best vertex found (tuple, never mutated by the caller);
    - ``fval``: ``f(x)`` at that vertex (finite; ``inf`` only if every
      evaluated point was infeasible);
    - ``iterations``: simplex iterations performed (not function
      evaluations);
    - ``n_evals``: objective evaluations, including the initial simplex —
      the honest cost number;
    - ``converged``: both tolerance criteria met before ``max_iter``;
    - ``reason``: ``"converged"`` or ``"max_iter reached"`` with the real
      numbers appended (spread / diameter), so a DEGRADED model can quote it.
    """

    x: tuple[float, ...]
    fval: float
    iterations: int
    n_evals: int
    converged: bool
    reason: str

    @property
    def n_params(self) -> int:
        return len(self.x)


def _safe(value: float) -> float:
    """Objective value with non-finite results mapped to ``+inf``.

    A trial point outside the feasible region (a likelihood that overflowed,
    a log of a non-positive variance) must be *worse than everything*, not a
    crash — that is how the simplex is pushed back into the region.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"objective must return a real number, got {value!r}")
    v = float(value)
    if math.isnan(v):
        return math.inf
    return v


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    step: float = DEFAULT_STEP,
    tol: float = DEFAULT_TOL,
    tol_x: float = DEFAULT_TOL_X,
    max_iter: int = DEFAULT_MAX_ITER,
) -> NMResult:
    """Minimise ``f`` from ``x0`` with the Nelder–Mead simplex (design §9.1).

    Deterministic: the initial simplex, every ordering tie-break (by vertex
    index) and every acceptance test are fixed, so repeated calls with the
    same arguments return bit-identical results.

    Parameters
    ----------
    f
        Objective ``f(x) -> float``. May return ``inf``/``nan`` for an
        infeasible ``x`` (treated as ``+inf``); must be finite at ``x0``.
    x0
        Starting point; ``len(x0) >= 1``, all entries finite.
    step
        Initial simplex displacement, ``> 0``. Applied as ``step * |x0_i|``
        where ``x0_i != 0`` and as ``step`` where it is 0, so coordinates on
        very different scales are perturbed on their own scale.
    tol, tol_x
        Convergence tolerances (``> 0``) on the function-value spread and
        the simplex diameter; BOTH must hold.
    max_iter
        Iteration cap (``>= 1``); reaching it returns the best vertex with
        ``converged=False`` and a reason quoting the achieved tolerances.

    Malformed arguments raise ``ValueError`` (house rule: bad input raises,
    missing data degrades — there is no "missing data" here).
    """
    if not callable(f):
        raise ValueError("f must be callable")
    point0 = [float(v) for v in x0]
    n = len(point0)
    if n < 1:
        raise ValueError("x0 must have at least one coordinate")
    for i, v in enumerate(point0):
        if not math.isfinite(v):
            raise ValueError(f"x0[{i}] must be finite, got {v!r}")
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not (step > 0.0) or not math.isfinite(step):
        raise ValueError(f"step must be a finite float > 0, got {step!r}")
    for name, value in (("tol", tol), ("tol_x", tol_x)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value > 0.0):
            raise ValueError(f"{name} must be a float > 0, got {value!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"max_iter must be an int >= 1, got {max_iter!r}")

    n_evals = 0

    def evaluate(point: list[float]) -> float:
        nonlocal n_evals
        n_evals += 1
        return _safe(f(list(point)))

    f0 = evaluate(point0)
    if not math.isfinite(f0):
        raise ValueError(f"objective must be finite at x0, got {f0}")

    # --- initial simplex (deterministic) -----------------------------------
    simplex: list[list[float]] = [list(point0)]
    values: list[float] = [f0]
    for i in range(n):
        vertex = list(point0)
        delta = step * abs(vertex[i]) if vertex[i] != 0.0 else step
        vertex[i] = vertex[i] + delta
        simplex.append(vertex)
        values.append(evaluate(vertex))

    def order() -> None:
        """Sort vertices best→worst; ties broken by current index (stable)."""
        idx = sorted(range(len(values)), key=lambda i: (values[i], i))
        simplex[:] = [simplex[i] for i in idx]
        values[:] = [values[i] for i in idx]

    order()

    def spread() -> float:
        return values[-1] - values[0] if math.isfinite(values[-1]) else math.inf

    def diameter() -> float:
        best = simplex[0]
        return max(
            (max(abs(v[k] - best[k]) for k in range(n)) for v in simplex[1:]),
            default=0.0,
        )

    def converged_now() -> bool:
        return spread() <= tol * (abs(values[0]) + tol) and diameter() <= tol_x

    iterations = 0
    converged = converged_now()
    while not converged and iterations < max_iter:
        iterations += 1
        # centroid of all but the worst vertex
        centroid = [math.fsum(v[k] for v in simplex[:-1]) / n for k in range(n)]
        worst = simplex[-1]
        f_worst = values[-1]
        f_best = values[0]
        f_second_worst = values[-2]

        reflected = [centroid[k] + REFLECTION * (centroid[k] - worst[k]) for k in range(n)]
        f_reflected = evaluate(reflected)

        if f_best <= f_reflected < f_second_worst:
            simplex[-1], values[-1] = reflected, f_reflected
        elif f_reflected < f_best:
            expanded = [centroid[k] + EXPANSION * (reflected[k] - centroid[k]) for k in range(n)]
            f_expanded = evaluate(expanded)
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
        else:
            if f_reflected < f_worst:
                # outside contraction
                contracted = [
                    centroid[k] + CONTRACTION * (reflected[k] - centroid[k]) for k in range(n)
                ]
                f_contracted = evaluate(contracted)
                accept = f_contracted <= f_reflected
            else:
                # inside contraction
                contracted = [
                    centroid[k] + CONTRACTION * (worst[k] - centroid[k]) for k in range(n)
                ]
                f_contracted = evaluate(contracted)
                accept = f_contracted < f_worst
            if accept:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                best = list(simplex[0])
                for i in range(1, len(simplex)):
                    shrunk = [best[k] + SHRINK * (simplex[i][k] - best[k]) for k in range(n)]
                    simplex[i] = shrunk
                    values[i] = evaluate(shrunk)
        order()
        converged = converged_now()

    final_spread = spread()
    final_diameter = diameter()
    if converged:
        reason = REASON_CONVERGED
    else:
        reason = (
            f"{REASON_MAX_ITER}: iterations={iterations}, "
            f"f_spread={final_spread:.3e} > tol={tol:.3e}, "
            f"simplex_diameter={final_diameter:.3e} > tol_x={tol_x:.3e}"
        )
    return NMResult(
        x=tuple(simplex[0]),
        fval=values[0],
        iterations=iterations,
        n_evals=n_evals,
        converged=converged,
        reason=reason,
    )


__all__ = [
    "CONTRACTION",
    "DEFAULT_MAX_ITER",
    "DEFAULT_STEP",
    "DEFAULT_TOL",
    "DEFAULT_TOL_X",
    "EXPANSION",
    "NMResult",
    "REASON_CONVERGED",
    "REASON_MAX_ITER",
    "REFLECTION",
    "SHRINK",
    "nelder_mead",
]
