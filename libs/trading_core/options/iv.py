"""Implied volatility solver — INTERNALLY CALCULATED (Phase D design §8.1;
risk spec §22, §24; data-source provenance rule).

Pure stdlib, deterministic, no numpy/scipy (house rule): the solver is a
plain **bisection** on :func:`~libs.trading_core.options.bs.bs_price`, which
is strictly increasing in ``sigma`` for every European option with
``t_years > 0`` (vega ``= S e^{-qT} phi(d1) sqrt(T) > 0``). Monotonicity is
what makes bisection both correct and unconditionally convergent here —
Newton would be faster but can leave the bracket on a near-zero vega wing,
and this platform prefers a slower estimator that cannot diverge.

**Provenance (binding).** A number this module returns is INTERNAL
DETERMINISTIC, never vendor IV. ``IVResult.method`` is ``"BISECTION"`` and
callers persisting or displaying it must label it as internally calculated
(``docs/data-source-architecture.md`` §12) — the platform must never let a
solved IV masquerade as a broker/vendor quote.

Algorithm::

    f(sigma) = bs_price(spot, strike, t_years, sigma, right, r, q) - price

    f is strictly increasing on (0, inf).
    f(lo) > 0  ->  the price is BELOW the model floor at sigma=lo  -> None
    f(hi) < 0  ->  the price is ABOVE the sigma=hi ceiling          -> None
    otherwise  ->  bisect [lo, hi] until hi - lo <= tol or |f(mid)| <= 0
                   (price tolerance is NOT used as the stop: the bracket
                   width is, so the reported ``iv`` is accurate to ``tol``
                   in VOLATILITY units, which is what the caller cares
                   about and what the round-trip test pins).

Honest nulls (house rule) — every failure returns ``iv=None`` with a
``reason`` carrying the real numbers, and NEVER raises for a
merely-unsolvable price:

- ``t_years <= 0`` — an expired option has no volatility to imply;
- ``price <= 0`` — a non-positive mark is not a tradeable price;
- ``price <= intrinsic`` (the discounted no-arbitrage floor at ``lo``) —
  below the model's own minimum, so no positive sigma reproduces it;
- ``price >= bs_price(hi)`` — above the ``sigma = hi`` ceiling (default
  500 % vol); reporting ``hi`` would be a fabricated number.

Known limitation (measured, not hidden). On a contract whose **vega is
essentially zero** — a deep-OTM long-dated wing worth ~1e-15 — the entire
``[lo, hi]`` bracket prices within a few float ULPs of each other, so no
bisection can recover the vol that produced the mark: the price simply does
not contain it. The solver still returns a bracketed sigma that *reprices
the contract*, and ``tests/test_options_iv.py`` asserts exactly that weaker
(and true) property for those points. Callers that care should check vega,
not this module: a leg worth 1e-15 contributes nothing to a stress number
either way.

Malformed input still raises ``ValueError`` (contract §1: bad input never
silently produces a number) — non-positive spot/strike, unknown ``right``,
a non-positive or inverted ``[lo, hi]`` bracket, ``tol <= 0`` or
``max_iter < 1``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bs import bs_price

#: Solver label recorded on every result (provenance §12: INTERNAL
#: DETERMINISTIC — this is never a vendor IV).
METHOD_BISECTION = "BISECTION"

#: Default bracket. ``lo`` is a hair above zero (``bs_price`` rejects
#: ``iv <= 0``); ``hi = 5.0`` is 500 % annualized vol — above any listed
#: equity option's traded IV, so a price beyond it is data, not vol.
DEFAULT_LO = 1e-4
DEFAULT_HI = 5.0

#: Default half-width the bracket must reach, in VOLATILITY units.
DEFAULT_TOL = 1e-8

#: Default iteration budget. Bisection halves the bracket each step, so
#: ``(hi - lo) / 2^n <= tol`` needs ``n >= log2(5 / 1e-8) ~= 29`` — 100 is
#: a generous ceiling that also covers a widened caller-supplied bracket.
DEFAULT_MAX_ITER = 100


@dataclass(frozen=True)
class IVResult:
    """The outcome of one implied-volatility solve.

    - ``iv``: the solved volatility, or ``None`` when the price is not
      invertible under the model (honest null — never a clamped ``lo``/
      ``hi`` and never a fabricated 0);
    - ``iterations``: bisection steps actually taken (0 on a guard exit);
    - ``converged``: True only when the bracket reached ``tol``;
    - ``reason``: why ``iv`` is ``None`` (or why a solve stopped early),
      with the real numbers; ``None`` on a clean solve;
    - ``method``: always ``"BISECTION"`` — the provenance label.
    """

    iv: float | None
    iterations: int
    converged: bool
    reason: str | None = None
    method: str = METHOD_BISECTION


def _intrinsic(spot: float, strike: float, right: str) -> float:
    """Undiscounted intrinsic value — the price floor used in messages."""
    if right == "C":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    right: str,
    *,
    r: float = 0.04,
    q: float = 0.0,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> IVResult:
    """Solve ``bs_price(..., iv) == price`` for ``iv`` by bisection (§8.1).

    ``price`` is the option's mark PER SHARE (not per contract) in the same
    units as ``spot``/``strike``. ``right`` is ``"C"`` or ``"P"``.

    Returns an :class:`IVResult`; ``iv is None`` whenever the price lies
    outside the model's ``[lo, hi]`` price range or the option has expired
    — with a ``reason`` that states the numbers. Raises ``ValueError`` only
    on malformed input (bad spot/strike/right/bracket/tolerance).

    Hand-check: an ATM call with ``S=K=100``, ``T=1``, ``r=q=0`` and
    ``sigma=0.20`` prices at ``100*(2*N(0.1)-1) = 7.9656`` — feeding that
    price back returns ``0.20`` to within ``tol``.
    """
    # --- malformed input: always raises (contract §1) ---------------------
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if right not in ("C", "P"):
        raise ValueError(f'right must be "C" or "P", got {right!r}')
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price}")
    if lo <= 0.0 or hi <= lo:
        raise ValueError(f"bracket must satisfy 0 < lo < hi, got lo={lo}, hi={hi}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"max_iter must be an int >= 1, got {max_iter!r}")

    # --- honest nulls: unsolvable prices degrade, never raise -------------
    if t_years <= 0.0:
        return IVResult(
            iv=None,
            iterations=0,
            converged=False,
            reason=(
                f"expired: t_years={t_years:g} <= 0 — an expired option has "
                f"no implied volatility (price is intrinsic "
                f"{_intrinsic(spot, strike, right):.4f})"
            ),
        )
    if price <= 0.0:
        return IVResult(
            iv=None,
            iterations=0,
            converged=False,
            reason=f"price={price:g} <= 0 is not a tradeable mark",
        )

    floor_price = bs_price(spot, strike, t_years, lo, right, r, q)
    if price <= floor_price:
        return IVResult(
            iv=None,
            iterations=0,
            converged=False,
            reason=(
                f"price={price:.6f} <= model floor {floor_price:.6f} at "
                f"sigma={lo:g} (intrinsic {_intrinsic(spot, strike, right):.6f}) "
                f"— no positive volatility reproduces it"
            ),
        )
    ceiling_price = bs_price(spot, strike, t_years, hi, right, r, q)
    if price >= ceiling_price:
        return IVResult(
            iv=None,
            iterations=0,
            converged=False,
            reason=(
                f"price={price:.6f} >= model ceiling {ceiling_price:.6f} at "
                f"sigma={hi:g} — above the sigma={hi:g} ceiling"
            ),
        )

    # --- bisection on a bracket that is now guaranteed to straddle -------
    a, b = lo, hi
    iterations = 0
    mid = 0.5 * (a + b)
    for _ in range(max_iter):
        if b - a <= tol:
            break
        iterations += 1
        mid = 0.5 * (a + b)
        f_mid = bs_price(spot, strike, t_years, mid, right, r, q) - price
        if f_mid == 0.0:
            return IVResult(iv=mid, iterations=iterations, converged=True, reason=None)
        if f_mid > 0.0:
            b = mid
        else:
            a = mid

    mid = 0.5 * (a + b)
    converged = (b - a) <= tol
    reason = (
        None
        if converged
        else (
            f"bracket width {b - a:.3e} > tol={tol:g} after "
            f"max_iter={max_iter} bisection steps"
        )
    )
    # Not converged is still a usable estimate (the bracket brackets the
    # root), so the value is reported WITH the reason — the caller decides.
    return IVResult(iv=mid, iterations=iterations, converged=converged, reason=reason)


__all__ = [
    "DEFAULT_HI",
    "DEFAULT_LO",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOL",
    "METHOD_BISECTION",
    "IVResult",
    "implied_vol",
]
