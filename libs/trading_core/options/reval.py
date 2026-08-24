"""Leg-aware scenario revaluation, basis-anchored (Phase D design §8.2;
risk spec §21, §22, §25, §26).

Pure stdlib, deterministic, no I/O (house rule). This module answers ONE
question: *what would this book be worth if the underlying moved by X, IV
moved by Y and Z days passed?* — by **full revaluation** of each option leg
through :func:`~libs.trading_core.options.bs.bs_price`, not by multiplying
delta by the move (spec §21: "Do NOT treat an option as if it were simply
stock multiplied by Delta for all risk calculations").

The basis anchor (why scenario 0 gives EXACTLY 0)
--------------------------------------------------
A model price never equals the market mark: bid/ask, American early
exercise, discrete dividends and a wrong ``r`` all show up as a residual.
If we revalued naively (``bs_price(S1, iv1, T1) - mark0``), a *zero*
scenario would report that residual as scenario P&L — a fabricated number
that would pollute every stress row.

So each leg carries a **basis**, computed once at the baseline::

    model0 = bs_price(spot0_implied_by_the_leg's own mark, ...)   # see below
    basis  = mark0 - model0

and every scenario price is ``bs_price(S1, K, T1, iv1) + basis``. The
basis is held CONSTANT across scenarios (a documented modelling
assumption — it is a *price* residual, not a *vol* residual, so it does not
scale with the move). Consequences, both intended:

- the zero scenario (0 spot shock, 0 IV shock, 0 days) returns P&L ``0.0``
  — bit-exact, not "approximately zero". Getting that exactness right takes
  one extra care: P&L is measured as ``price1 - (model0 + basis)``, NOT as
  ``price1 - mark0``. The two are algebraically identical, but floating-
  point addition does not re-associate, so ``(model0 + basis) - mark0`` can
  be ~1e-16 instead of 0 for some marks. Differencing against the same
  reconstructed baseline makes the cancellation exact for every mark;
- the reported P&L is a pure *model delta*, so two scenarios differ only by
  what the scenario changed.

At/after expiry (``T1 <= 0``) the leg is worth **intrinsic**, with NO
basis: a residual that exists because of bid/ask spreads has no meaning at
settlement, and adding it would let an expired option settle above/below
its terminal payoff.

Baseline model price. ``leg_baseline`` needs the spot the mark was struck
at; it takes it from :class:`OptionLeg.spot0` so the leg is self-contained
(the gateway builds legs from the same chain snapshot the risk view uses).

Method labelling (honest nulls, spec §22)
------------------------------------------
Full revaluation needs an IV. When a leg has ``iv0 is None`` (the chain
gave no IV and the solver could not invert the mark), the leg is priced
**DELTA_LINEAR** — ``qty * multiplier * delta0 * spot0 * shock`` — and the
result records ``method_by_key[key] = "DELTA_LINEAR"`` plus a note. It is
never silently mixed into a FULL_REVAL claim: :class:`ScenarioPnl` exposes
``method_coverage`` so a caller (and the UI) can see how much of the book
was actually revalued. A DELTA_LINEAR leg ignores IV and time by
construction — the fallback cannot see vega or theta, and the note says so.

Sign conventions (identical to the rest of the risk library)
------------------------------------------------------------
``quantity`` is SIGNED in CONTRACTS: long legs positive, short legs
(spread short legs, covered calls, cash-secured puts) NEGATIVE. P&L is
``quantity * multiplier * (price1 - price0)`` — a short call that gains
value produces a loss automatically, no special-casing. Stock legs are
``quantity * spot0 * shock`` with ``quantity`` signed in SHARES (short
stock negative) and multiplier 1.

Everything is USD per scenario, gain-positive (contract §1).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .bs import bs_price

#: Revaluation method labels (design §8.2). ``FULL_REVAL`` = the leg was
#: repriced through Black-Scholes under the scenario state; ``DELTA_LINEAR``
#: = the honest fallback used when the leg has no IV (it cannot see IV or
#: time — the note on the result says so).
METHOD_FULL_REVAL = "FULL_REVAL"
METHOD_DELTA_LINEAR = "DELTA_LINEAR"

#: Calendar days per year — the SAME convention ``bs.py`` uses for theta,
#: so ``days_forward`` and the pricer agree on what a day is.
DAYS_PER_YEAR = 365.0

#: Default option contract multiplier (shares per contract).
DEFAULT_MULTIPLIER = 100

#: Floor applied to a shocked IV. ``bs_price`` rejects ``iv <= 0``, so a
#: ``-100 %`` IV shock must degrade to a near-zero volatility rather than
#: raise. At 1e-6 the option is worth its (discounted) intrinsic to well
#: within a cent, which is the honest limit of "vol went to zero".
IV_FLOOR = 1e-6


def _check_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionLeg:
    """One option leg of the book, per design §8.2.

    - ``key``: unique per position row (e.g. ``"AAPL#12"``; a spread's two
      legs use distinct keys such as ``"AAPL#12:long"`` / ``"…:short"``);
    - ``ticker``: the UNDERLYING whose shock drives this leg;
    - ``right``: ``"C"`` or ``"P"``;
    - ``strike``: per share;
    - ``t_years``: time to expiry at the baseline (DTE / 365);
    - ``quantity``: SIGNED contracts — short legs NEGATIVE;
    - ``multiplier``: shares per contract (100);
    - ``spot0``: the underlying price the ``mark0`` was struck at;
    - ``mark0``: the leg's mark PER SHARE (chain mid), used as the anchor;
    - ``iv0``: the baseline implied vol (vendor or internally solved).
      ``None`` ⇒ this leg falls back to DELTA_LINEAR and is labelled;
    - ``delta0``: per-share delta used ONLY by the DELTA_LINEAR fallback.
      ``None`` with ``iv0 is None`` ⇒ the leg cannot be valued at all and
      is reported with a note (contributing 0.0, never a guess);
    - ``r`` / ``q``: annualized rate and dividend yield for the pricer.
    """

    key: str
    ticker: str
    right: str
    strike: float
    t_years: float
    quantity: int
    spot0: float
    mark0: float
    iv0: float | None = None
    delta0: float | None = None
    multiplier: int = DEFAULT_MULTIPLIER
    r: float = 0.04
    q: float = 0.0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be a non-empty string")
        if not self.ticker:
            raise ValueError(f"{self.key}: ticker must be a non-empty string")
        if self.right not in ("C", "P"):
            raise ValueError(f'{self.key}: right must be "C" or "P", got {self.right!r}')
        if _check_finite(f"{self.key}.strike", self.strike) <= 0.0:
            raise ValueError(f"{self.key}: strike must be > 0, got {self.strike}")
        if _check_finite(f"{self.key}.spot0", self.spot0) <= 0.0:
            raise ValueError(f"{self.key}: spot0 must be > 0, got {self.spot0}")
        _check_finite(f"{self.key}.t_years", self.t_years)
        _check_finite(f"{self.key}.mark0", self.mark0)
        _check_finite(f"{self.key}.r", self.r)
        _check_finite(f"{self.key}.q", self.q)
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError(f"{self.key}: quantity must be an int, got {self.quantity!r}")
        if (
            isinstance(self.multiplier, bool)
            or not isinstance(self.multiplier, int)
            or self.multiplier <= 0
        ):
            raise ValueError(
                f"{self.key}: multiplier must be an int > 0, got {self.multiplier!r}"
            )
        if self.iv0 is not None:
            if _check_finite(f"{self.key}.iv0", self.iv0) <= 0.0:
                raise ValueError(f"{self.key}: iv0 must be > 0 or None, got {self.iv0}")
        if self.delta0 is not None:
            _check_finite(f"{self.key}.delta0", self.delta0)

    @property
    def can_full_reval(self) -> bool:
        """True when the leg has the IV full revaluation needs."""
        return self.iv0 is not None

    def scaled(self, factor: int) -> "OptionLeg":
        """The same leg at ``quantity * factor`` — how a candidate's
        per-unit legs are grown to a trade quantity (design §8.3 cap
        search). ``factor`` must be an int ≥ 0."""
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 0:
            raise ValueError(f"factor must be an int >= 0, got {factor!r}")
        return OptionLeg(
            key=self.key,
            ticker=self.ticker,
            right=self.right,
            strike=self.strike,
            t_years=self.t_years,
            quantity=self.quantity * factor,
            spot0=self.spot0,
            mark0=self.mark0,
            iv0=self.iv0,
            delta0=self.delta0,
            multiplier=self.multiplier,
            r=self.r,
            q=self.q,
        )


@dataclass(frozen=True)
class StockLeg:
    """One stock position (design §8.2).

    ``quantity`` is SIGNED in SHARES (short stock negative); ``spot0`` is
    the baseline price. Stock is linear: scenario P&L is
    ``quantity * spot0 * shock`` — exact, no model, no basis.
    """

    key: str
    ticker: str
    quantity: int
    spot0: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be a non-empty string")
        if not self.ticker:
            raise ValueError(f"{self.key}: ticker must be a non-empty string")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError(f"{self.key}: quantity must be an int, got {self.quantity!r}")
        if _check_finite(f"{self.key}.spot0", self.spot0) <= 0.0:
            raise ValueError(f"{self.key}: spot0 must be > 0, got {self.spot0}")

    def scaled(self, factor: int) -> "StockLeg":
        """The same leg at ``quantity * factor`` (cap search helper)."""
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 0:
            raise ValueError(f"factor must be an int >= 0, got {factor!r}")
        return StockLeg(
            key=self.key,
            ticker=self.ticker,
            quantity=self.quantity * factor,
            spot0=self.spot0,
        )


# ---------------------------------------------------------------------------
# Baseline & single-leg revaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegBaseline:
    """The leg's anchor: its model price at the baseline state and the
    residual that reconciles the model with the market mark (design §8.2).

    ``price0 = model0 + basis == mark0`` by construction — which is exactly
    why a zero scenario returns 0.0. ``basis`` is ``None`` (and ``model0``
    ``None``) when the leg has no IV: the DELTA_LINEAR fallback has no
    model price to anchor.
    """

    key: str
    model0: float | None
    basis: float | None
    mark0: float
    method: str

    @property
    def price0(self) -> float:
        """The baseline price actually used — always the market mark."""
        return self.mark0


def leg_baseline(leg: OptionLeg) -> LegBaseline:
    """Compute ``model0`` and the constant ``basis = mark0 - model0`` (§8.2).

    An expired leg (``t_years <= 0``) is anchored on intrinsic with
    ``basis = 0.0``: at settlement the payoff IS the price, so no residual
    is carried (see the module docstring).
    """
    if leg.iv0 is None:
        return LegBaseline(
            key=leg.key,
            model0=None,
            basis=None,
            mark0=leg.mark0,
            method=METHOD_DELTA_LINEAR,
        )
    if leg.t_years <= 0.0:
        model0 = bs_price(
            leg.spot0, leg.strike, 0.0, leg.iv0, leg.right, leg.r, leg.q
        )
        return LegBaseline(
            key=leg.key,
            model0=model0,
            basis=0.0,
            mark0=leg.mark0,
            method=METHOD_FULL_REVAL,
        )
    model0 = bs_price(
        leg.spot0, leg.strike, leg.t_years, leg.iv0, leg.right, leg.r, leg.q
    )
    return LegBaseline(
        key=leg.key,
        model0=model0,
        basis=leg.mark0 - model0,
        mark0=leg.mark0,
        method=METHOD_FULL_REVAL,
    )


def reval_leg(
    leg: OptionLeg,
    *,
    spot1: float,
    iv1: float,
    days_forward: float,
    baseline: LegBaseline | None = None,
) -> float:
    """The leg's scenario price PER SHARE under ``(spot1, iv1, +days)``.

    ``price1 = bs_price(spot1, K, max(T0 - days/365, 0), iv1) + basis``,
    except at/after expiry where it is **intrinsic with no basis** (§8.2).

    Raises ``ValueError`` when the leg has no ``iv0`` — the caller must use
    the DELTA_LINEAR path instead (:func:`scenario_pnl` does this and
    labels it); silently pretending a full revaluation happened would break
    the provenance of ``method_coverage``.
    """
    if leg.iv0 is None:
        raise ValueError(
            f"{leg.key}: reval_leg requires iv0; leg has none — use the "
            f"DELTA_LINEAR fallback and label it"
        )
    if iv1 <= 0.0:
        raise ValueError(f"{leg.key}: iv1 must be > 0, got {iv1}")
    if spot1 <= 0.0:
        raise ValueError(f"{leg.key}: spot1 must be > 0, got {spot1}")
    base = baseline if baseline is not None else leg_baseline(leg)
    t1 = leg.t_years - float(days_forward) / DAYS_PER_YEAR
    if t1 <= 0.0:
        # Settlement: the payoff IS the price. No basis (module docstring).
        return bs_price(spot1, leg.strike, 0.0, iv1, leg.right, leg.r, leg.q)
    basis = base.basis if base.basis is not None else 0.0
    return bs_price(spot1, leg.strike, t1, iv1, leg.right, leg.r, leg.q) + basis


def _baseline_price_for_diff(leg: OptionLeg, base: LegBaseline) -> float:
    """The baseline price P&L is measured AGAINST, in the same arithmetic
    form :func:`reval_leg` produces a scenario price in.

    This is ``model0 + basis``, which is *algebraically* ``mark0`` but not
    always ``mark0`` bit-for-bit: floating-point addition does not
    re-associate, so for some marks ``(model0 + basis) - mark0`` is ~1e-16
    rather than 0. Measuring the scenario price against the reconstructed
    baseline instead makes the zero scenario cancel EXACTLY, which is the
    guarantee the whole basis anchor exists to provide (module docstring).

    An expired leg is anchored on intrinsic with no basis, matching
    :func:`reval_leg`'s settlement branch.
    """
    if base.model0 is None:  # DELTA_LINEAR leg — no model anchor
        return leg.mark0
    if leg.t_years <= 0.0:
        return base.model0
    return base.model0 + (base.basis or 0.0)


# ---------------------------------------------------------------------------
# Scenario aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioPnl:
    """The book's P&L under ONE scenario state (design §8.2).

    - ``total``: USD, gain-positive, ``math.fsum`` of ``per_key``;
    - ``per_key``: every leg's contribution in USD, in input order (stock
      legs first, then option legs);
    - ``method_by_key``: ``FULL_REVAL`` or ``DELTA_LINEAR`` per OPTION leg
      (stock legs are exact and are labelled ``FULL_REVAL`` — a linear
      instrument priced linearly IS its full revaluation);
    - ``method_coverage``: ``{"FULL_REVAL": n, "DELTA_LINEAR": n}`` counts;
    - ``notes``: one honest sentence per degraded leg (no IV, unvaluable).
    """

    total: float
    per_key: Mapping[str, float]
    method_by_key: Mapping[str, str]
    method_coverage: Mapping[str, int]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_key", dict(self.per_key))
        object.__setattr__(self, "method_by_key", dict(self.method_by_key))
        object.__setattr__(self, "method_coverage", dict(self.method_coverage))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def fully_revalued(self) -> bool:
        """True when no leg fell back to DELTA_LINEAR."""
        return self.method_coverage.get(METHOD_DELTA_LINEAR, 0) == 0


def scenario_pnl(
    stock_legs: Sequence[StockLeg],
    option_legs: Sequence[OptionLeg],
    *,
    spot0_by_ticker: Mapping[str, float] | None = None,
    spot_shock_by_ticker: Mapping[str, float] | None = None,
    spot_shock: float = 0.0,
    iv_shock: float = 0.0,
    days_forward: float = 0.0,
) -> ScenarioPnl:
    """Revalue the whole book under one scenario state (design §8.2).

    Scenario parameterisation:

    - ``spot_shock`` is a FRACTIONAL move applied to EVERY underlying
      (``-0.05`` = every name down 5 %; the uniform-beta-1 assumption, and
      it is documented as such by the caller's scenario);
    - ``spot_shock_by_ticker`` overrides it per ticker (a ticker absent
      from the mapping keeps ``spot_shock``);
    - ``iv_shock`` is **RELATIVE and multiplicative on the IV LEVEL**:
      ``+0.20`` ⇒ ``iv1 = iv0 * 1.20`` (design §8.2 — not "+20 vol
      points"). ``iv1`` is floored at a hair above 0 so a ``-1.0`` shock
      degrades to a near-zero vol instead of raising;
    - ``days_forward`` moves time forward in CALENDAR days.

    ``spot0_by_ticker`` overrides the per-leg ``spot0`` when supplied (the
    gateway passes one live spot map; a ticker absent from it falls back to
    the leg's own ``spot0``).

    Guarantees (pinned by tests): the zero scenario returns ``0.0``
    exactly; P&L is linear in each leg's ``quantity``.
    """
    _check_finite("spot_shock", spot_shock)
    _check_finite("iv_shock", iv_shock)
    _check_finite("days_forward", days_forward)
    if days_forward < 0.0:
        raise ValueError(f"days_forward must be >= 0, got {days_forward}")
    spots0 = dict(spot0_by_ticker or {})
    shocks = dict(spot_shock_by_ticker or {})

    per_key: dict[str, float] = {}
    method_by_key: dict[str, str] = {}
    notes: list[str] = []
    n_full = 0
    n_linear = 0

    def _spot0_of(ticker: str, fallback: float) -> float:
        s = spots0.get(ticker, fallback)
        if s <= 0.0:
            raise ValueError(f"spot0 for {ticker} must be > 0, got {s}")
        return float(s)

    def _shock_of(ticker: str) -> float:
        return float(shocks.get(ticker, spot_shock))

    # --- stock legs: exact and linear ------------------------------------
    for sleg in stock_legs:
        if sleg.key in per_key:
            raise ValueError(f"duplicate leg key {sleg.key!r}")
        s0 = _spot0_of(sleg.ticker, sleg.spot0)
        pnl = sleg.quantity * s0 * _shock_of(sleg.ticker)
        per_key[sleg.key] = pnl
        method_by_key[sleg.key] = METHOD_FULL_REVAL
        n_full += 1

    # --- option legs: full revaluation, or the labelled fallback ---------
    for oleg in option_legs:
        if oleg.key in per_key:
            raise ValueError(f"duplicate leg key {oleg.key!r}")
        s0 = _spot0_of(oleg.ticker, oleg.spot0)
        shock = _shock_of(oleg.ticker)
        s1 = s0 * (1.0 + shock)
        if s1 <= 0.0:
            raise ValueError(
                f"{oleg.key}: spot shock {shock:g} drives {oleg.ticker} to "
                f"{s1:g} <= 0 — a stock cannot go negative"
            )
        if oleg.iv0 is not None:
            base = leg_baseline(oleg)
            iv1 = max(oleg.iv0 * (1.0 + iv_shock), IV_FLOOR)
            price1 = reval_leg(
                oleg, spot1=s1, iv1=iv1, days_forward=days_forward, baseline=base
            )
            # P&L in the MODEL-DELTA form (see the module docstring): the
            # basis cancels algebraically, so subtracting ``price0`` — which
            # is ``model0 + basis`` — is done as ``price1 - price0`` with the
            # SAME reconstructed price0, never against ``mark0`` directly.
            # ``(model0 + basis) - mark0`` is not bit-exactly 0 in floating
            # point for every mark (it reassociates), and the zero-scenario
            # identity must be EXACT, not approximate.
            price0 = _baseline_price_for_diff(oleg, base)
            pnl = oleg.quantity * oleg.multiplier * (price1 - price0)
            per_key[oleg.key] = pnl
            method_by_key[oleg.key] = METHOD_FULL_REVAL
            n_full += 1
            continue
        # --- DELTA_LINEAR fallback: no IV, so no model price -------------
        if oleg.delta0 is None:
            per_key[oleg.key] = 0.0
            method_by_key[oleg.key] = METHOD_DELTA_LINEAR
            n_linear += 1
            notes.append(
                f"{oleg.key}: no iv0 and no delta0 — leg contributes 0.0 "
                f"(unvaluable; the book's stress loss is understated)"
            )
            continue
        pnl = oleg.quantity * oleg.multiplier * oleg.delta0 * s0 * shock
        per_key[oleg.key] = pnl
        method_by_key[oleg.key] = METHOD_DELTA_LINEAR
        n_linear += 1
        notes.append(
            f"{oleg.key}: no iv0 — priced DELTA_LINEAR (delta0="
            f"{oleg.delta0:.4f}); this leg cannot see the IV shock "
            f"{iv_shock:+.2f} or {days_forward:g} days of decay"
        )

    total = math.fsum(per_key.values())
    return ScenarioPnl(
        total=total,
        per_key=per_key,
        method_by_key=method_by_key,
        method_coverage={
            METHOD_FULL_REVAL: n_full,
            METHOD_DELTA_LINEAR: n_linear,
        },
        notes=tuple(notes),
    )


__all__ = [
    "DAYS_PER_YEAR",
    "DEFAULT_MULTIPLIER",
    "IV_FLOOR",
    "METHOD_DELTA_LINEAR",
    "METHOD_FULL_REVAL",
    "LegBaseline",
    "OptionLeg",
    "ScenarioPnl",
    "StockLeg",
    "leg_baseline",
    "reval_leg",
    "scenario_pnl",
]
