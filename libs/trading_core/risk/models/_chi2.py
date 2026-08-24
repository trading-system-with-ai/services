"""χ² survival function for ANY degrees of freedom, via the regularized
incomplete gamma function (Phase E design §9.2).

Pure stdlib, deterministic, no numpy/scipy (house rule). Private to the
risk library (leading underscore): it is a numerical primitive, not a risk
model — it holds no thresholds and decides nothing.

Why this module exists
----------------------
Phase B needed only two χ² tail probabilities and both have exact closed
forms, which is why ``risk/validation.py`` uses them and keeps using them:

- ``df = 1``: ``P(X > x) = erfc(sqrt(x/2))``  (Kupiec POF, Christoffersen
  independence);
- ``df = 2``: ``P(X > x) = exp(-x/2)``        (Jarque–Bera, Christoffersen
  conditional coverage).

Phase E adds the Ljung–Box statistic on standardized residuals², which is
χ² with ``m`` lags (default ``m = 10``) — no elementary closed form. Rather
than special-case one more df, this module implements the general function
and the Phase B closed forms are *pinned against it* by test
(``tests/test_risk_chi2.py``): ``chi2_sf(x, 1) == erfc(sqrt(x/2))`` and
``chi2_sf(x, 2) == exp(-x/2)`` to 1e-12. The closed forms stay in
``validation.py`` — they are exact, cheaper, and changing them would change
persisted Kupiec/Christoffersen p-values (contract §4 versioning).

The algorithm (Numerical Recipes ``gammp``/``gammq``)
----------------------------------------------------
``P(a, x) = γ(a, x) / Γ(a)`` (lower regularized), ``Q(a, x) = 1 - P(a, x)``
(upper regularized). Two complementary expansions, each convergent exactly
where the other is not:

- **Series** (used for ``x < a + 1``):
  ``P(a, x) = x^a e^{-x} / Γ(a) · Σ_{k≥0} x^k / (a(a+1)…(a+k))``.
- **Continued fraction** (Lentz's modified algorithm, used for
  ``x >= a + 1``):
  ``Q(a, x) = x^a e^{-x} / Γ(a) · 1/(x+1-a- 1·(1-a)/(x+3-a- 2·(2-a)/(x+5-a-…)))``.

Both are evaluated in log space for the prefactor
(``a·ln x - x - lgamma(a)``) so that a large ``x`` (a very significant test
statistic) underflows to 0.0 rather than raising, and the answer stays
accurate to ~1e-14 relative over the range that matters for diagnostics.

Then ``chi2_sf(x, df) = Q(df/2, x/2)`` — the χ²(df) upper tail.

Guards: ``a <= 0`` or a negative ``x`` is malformed (``ValueError``);
``x = 0`` gives ``P = 0`` / ``Q = 1`` exactly; a tiny negative ``x`` from
floating-point rounding (``> -1e-9``) is clamped to 0, matching the
tolerance already used by ``validation.chi2_1_sf``. Non-convergence after
``MAX_ITER`` terms raises ``ValueError`` (a numerical bug, not a data gap) —
callers that must not raise wrap the call and report FAILED health.
"""
from __future__ import annotations

import math

#: Iteration cap for both expansions (documented parameter, not magic).
MAX_ITER = 1000

#: Relative convergence tolerance for both expansions.
EPS = 3e-16

#: Smallest number used to avoid a zero denominator in Lentz's algorithm.
FPMIN = 1e-300

#: Rounding slack: a statistic this slightly negative is treated as 0
#: (same convention as ``validation.chi2_1_sf``).
NEGATIVE_TOLERANCE = 1e-9


def _check_a(a: float) -> float:
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a):
        raise ValueError(f"a must be a finite number, got {a!r}")
    if a <= 0.0:
        raise ValueError(f"a must be > 0, got {a}")
    return float(a)


def _check_x(x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"x must be a finite number, got {x!r}")
    v = float(x)
    if v < 0.0:
        if v < -NEGATIVE_TOLERANCE:
            raise ValueError(f"x must be >= 0, got {x}")
        v = 0.0
    return v


def _gamma_series(a: float, x: float) -> float:
    """Lower regularized ``P(a, x)`` by the series (accurate for ``x < a+1``)."""
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * EPS:
            log_prefactor = a * math.log(x) - x - math.lgamma(a)
            if log_prefactor < -745.0:  # exp() underflows to 0.0 below this
                return 0.0
            return total * math.exp(log_prefactor)
    raise ValueError(f"gamma series did not converge for a={a}, x={x} in {MAX_ITER} terms")


def _gamma_continued_fraction(a: float, x: float) -> float:
    """Upper regularized ``Q(a, x)`` by the continued fraction (``x >= a+1``).

    Modified Lentz algorithm: ``b`` is the running denominator term, ``c``
    and ``d`` the two recurrences, ``h`` the accumulated value.
    """
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b if b != 0.0 else 1.0 / FPMIN
    h = d
    for i in range(1, MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            log_prefactor = a * math.log(x) - x - math.lgamma(a)
            if log_prefactor < -745.0:
                return 0.0
            return math.exp(log_prefactor) * h
    raise ValueError(
        f"gamma continued fraction did not converge for a={a}, x={x} in {MAX_ITER} terms"
    )


def regularized_gamma_p(a: float, x: float) -> float:
    """Lower regularized incomplete gamma ``P(a, x) = γ(a, x)/Γ(a)`` ∈ [0, 1].

    Series for ``x < a + 1``, ``1 - Q`` from the continued fraction
    otherwise. ``x = 0`` ⇒ ``0.0`` exactly. ``a <= 0`` or ``x < 0``
    (beyond rounding slack) ⇒ ``ValueError``.
    """
    a = _check_a(a)
    x = _check_x(x)
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_continued_fraction(a, x)


def regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma ``Q(a, x) = Γ(a, x)/Γ(a)`` ∈ [0, 1].

    The complement of :func:`regularized_gamma_p`, computed by whichever
    expansion is stable in the region so no accuracy is lost to
    cancellation: continued fraction for ``x >= a + 1`` (directly ``Q``),
    ``1 - series`` below it.
    """
    a = _check_a(a)
    x = _check_x(x)
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def chi2_sf(x: float, df: int) -> float:
    """χ²(df) survival function ``P(X > x) = Q(df/2, x/2)`` (design §9.2).

    ``df >= 1`` (integer degrees of freedom — every statistic in this
    library has integer df). ``x = 0`` ⇒ ``1.0``; a large ``x`` underflows
    to ``0.0`` rather than raising. Monotone non-increasing in ``x`` and
    non-decreasing in ``df`` for fixed ``x``.

    Hand-checks pinned by ``tests/test_risk_chi2.py``:
    ``chi2_sf(x, 1) == math.erfc(math.sqrt(x/2))``,
    ``chi2_sf(x, 2) == math.exp(-x/2)`` (both to 1e-12), and the textbook
    5 % point of χ²(10): ``chi2_sf(18.307, 10) ≈ 0.05``.
    """
    if isinstance(df, bool) or not isinstance(df, int) or df < 1:
        raise ValueError(f"df must be an int >= 1, got {df!r}")
    return regularized_gamma_q(df / 2.0, _check_x(x) / 2.0)


def chi2_cdf(x: float, df: int) -> float:
    """χ²(df) CDF ``P(X <= x) = P(df/2, x/2)`` — the complement of
    :func:`chi2_sf`, computed by the stable branch (never ``1 - sf``)."""
    if isinstance(df, bool) or not isinstance(df, int) or df < 1:
        raise ValueError(f"df must be an int >= 1, got {df!r}")
    return regularized_gamma_p(df / 2.0, _check_x(x) / 2.0)


__all__ = [
    "EPS",
    "MAX_ITER",
    "NEGATIVE_TOLERANCE",
    "chi2_cdf",
    "chi2_sf",
    "regularized_gamma_p",
    "regularized_gamma_q",
]
