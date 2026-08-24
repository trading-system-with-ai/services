"""Book P&L series construction — DELTA_LINEAR + FULL_REVAL_CONST_IV (risk
spec §8, §21, §22; Phase B design contract §2.9, design §10).

Pure, deterministic, stdlib-only (house rule): no DB, no market data, no
ORM types. The gateway snapshot builder hands in plain
:class:`PositionRiskInput` rows plus a ``SIMPLE`` :class:`~libs.trading_core
.risk.returns.ReturnMatrix` and receives per-position and total daily P&L
series that the VaR / ES / volatility / contribution estimators consume.

Estimator (``method = "DELTA_LINEAR"``, contract §2.9), hand-checkable:

    exposure = quantity × multiplier × delta × spot        (USD)
    pnl_t    = exposure × r_simple_t                       (USD per day)

- ``quantity`` is SIGNED (short legs / short stock negative), so a short
  stock position gains when the underlying falls;
- ``delta`` is per share, 1.0 for stock, the chain delta for options
  (negative for long puts), and the short leg's delta is NEGATED by the
  caller exactly as in ``greeks.py`` — this module never re-signs;
- ``pnl_t > 0`` is a GAIN (contract §1 sign convention);
- returns must be ``SIMPLE`` (``pnl = exposure × r`` is exact for stock
  under simple returns; ``LOG`` is a ``ValueError`` — mixing conventions is
  malformed input, not a data gap).

Second estimator (``method = "FULL_REVAL_CONST_IV"``, design §10.1) —
compliance batch 2, SHADOW. When a position carries its option leg fields
(``strike``, ``right``, ``t_years``, ``iv0``, ``mark0`` — all five present,
``t_years > 0``, ``iv0 > 0``), its daily P&L is measured by FULL
REVALUATION through :func:`~libs.trading_core.options.bs.bs_price` instead
of by delta:

    pnl_t = quantity × multiplier
            × [ BS(spot·(1 + r_t), K, T0, iv0, right) − BS(spot, K, T0, iv0, right) ]

Exactly what this estimator does and does NOT see, stated once so no
consumer has to guess:

- it sees S-CONVEXITY (gamma): a long call gains more on +r than it loses
  on −r, which the DELTA_LINEAR series is blind to by construction;
- ``iv0`` is held CONSTANT across the whole history — ``CONST_IV``. There
  is no stored IV history yet, so a vega P&L would have to be invented;
  the label says the number is a constant-vol reprice, never "the option's
  full P&L";
- ``T`` is held at ``T0`` on every observation — no theta. Each row is a
  1-day *price* move under a spot shock, not a 1-day *hold*; letting T roll
  would mix decay into a series the VaR estimators read as pure market
  risk.

THE BASIS CANCELS EXACTLY (the Phase D lesson, design §8.2). A model price
never equals the market ``mark0``; the residual ``basis = mark0 − model0``
is held constant, so it appears in both terms of the difference above and
cancels. The cancellation is made EXACT — not "within 1e-16" — by
differencing against the SAME reconstructed baseline value rather than
against ``mark0``: floating-point addition does not re-associate, so
``(model0 + basis) − mark0`` can be a stray ulp while ``price1 − price0``
with both terms built the same way is bit-exact. ``r_t = 0`` therefore
gives ``0.0`` exactly, and ``mark0`` never enters the arithmetic at all.

Fallback and labelling (honest nulls, spec §22): a position missing ANY of
the five leg fields — a stock, an option whose chain gave no IV, a spread
carried as one net row — is priced DELTA_LINEAR exactly as before and
LABELLED as such in :attr:`BookPnl.method_by_key`. Stock output is
byte-identical to the pre-batch function: the new path is not even reached.

Honest gaps (contract §1): a position whose ticker has NO column in the
matrix cannot be priced and is EXCLUDED from the book — its ticker is named
in ``tickers_missing`` (and its key in ``keys_excluded``) so the snapshot
builder marks data quality DEGRADED; nothing is filled with zeros. The
total is ``math.fsum`` over the INCLUDED positions per date.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from libs.trading_core.options.bs import bs_price
from libs.trading_core.risk.returns import RETURN_TYPE_SIMPLE, ReturnMatrix

#: The delta-approximation P&L method (contract §2.9) — stock (exact) and
#: every option leg that cannot be revalued (no IV, no strike, expired).
METHOD_DELTA_LINEAR = "DELTA_LINEAR"

#: Full Black-Scholes revaluation under the spot path, IV held at ``iv0``
#: and T held at ``T0`` (design §10.1). A DISTINCT label from Phase D's
#: ``reval.METHOD_FULL_REVAL``: that one revalues under an IV *and* a time
#: shock, this one is spot-only and says so in its name.
METHOD_FULL_REVAL_CONST_IV = "FULL_REVAL_CONST_IV"

#: The carry the revaluation prices under, stated EXPLICITLY rather than
#: inherited from ``bs_price``'s defaults: the terminal floor below is the
#: analytic limit of that pricer, so it has to know the rate the pricer
#: used. These are the same values Phase D's stress legs carry
#: (``risk_snapshot.STRESS_RATE`` / ``STRESS_DIVIDEND_YIELD``), so the P&L
#: series and a stress reprice of the SAME contract agree on carry.
FULL_REVAL_RATE = 0.04
FULL_REVAL_DIVIDEND_YIELD = 0.0


@dataclass(frozen=True)
class PositionRiskInput:
    """One position as the gateway builder passes it (contract §2.9).

    Replay-importable plain data — no ORM types — so a stored snapshot's
    inputs can be re-fed to reproduce its numbers (spec §44).

    - ``key``: unique per position row (e.g. ``"AAPL#12"``);
    - ``ticker``: the underlying whose returns drive the position;
    - ``instrument``: an ``InstrumentType`` value (carried for labelling);
    - ``quantity``: SIGNED — shares for stock, contracts for options; short
      legs / short stock negative;
    - ``multiplier``: 1 for stock, 100 for standard equity options;
    - ``spot``: underlying last close, > 0;
    - ``delta``: per-share delta (stock 1.0; options from the chain; short
      leg already NEGATED by the caller as in ``greeks.py``);
    - ``max_loss``: the position's defined max loss (carried through for
      the snapshot; not used by the estimator).

    OPTIONAL OPTION LEG FIELDS (design §10.1, additive — every one defaults
    to ``None``, and ``None`` anywhere means the row keeps its exact
    pre-batch DELTA_LINEAR behaviour). They are the SAME five values the
    Phase D :class:`~libs.trading_core.options.reval.OptionLeg` carries, and
    the gateway fills them from the SAME chain resolution — never a second
    chain read:

    - ``strike``: per share, > 0;
    - ``right``: ``"C"`` or ``"P"``;
    - ``t_years``: time to expiry at the baseline (DTE / 365), > 0 to
      revalue — an expired leg (``t_years <= 0``) has no convexity left to
      measure and falls back;
    - ``iv0``: the baseline implied vol, > 0 to revalue. ``None`` is the
      honest chain gap that keeps the row DELTA_LINEAR and LABELLED;
    - ``mark0``: the leg mark per share. Carried for provenance and for the
      basis identity of design §10.1 — the estimator differences two model
      prices, so the basis cancels and ``mark0`` never enters the
      arithmetic (see the module docstring).

    ``can_full_reval`` is the single dispatch predicate; nothing else in
    this module re-decides it.

    ``ValueError`` on malformed input: ``multiplier < 1``, non-finite or
    non-positive ``spot``, non-finite ``delta`` / ``max_loss``, and — when
    given — a non-finite or non-positive ``strike``, a ``right`` outside
    ``{"C", "P"}``, a non-finite ``t_years`` / ``mark0``, or a non-finite
    or non-positive ``iv0``. Malformed leg data is rejected, never quietly
    downgraded to DELTA_LINEAR: a wrong strike would otherwise vanish into
    a plausible-looking number.
    """

    key: str
    ticker: str
    instrument: str
    quantity: int
    multiplier: int
    spot: float
    delta: float
    max_loss: float
    # --- optional option leg (design §10.1); None ⇒ DELTA_LINEAR ----------
    strike: float | None = None
    right: str | None = None
    t_years: float | None = None
    iv0: float | None = None
    mark0: float | None = None

    def __post_init__(self) -> None:
        if self.multiplier < 1:
            raise ValueError(
                f"{self.key}: multiplier must be >= 1, got {self.multiplier}"
            )
        if not math.isfinite(self.spot) or self.spot <= 0:
            raise ValueError(
                f"{self.key}: spot must be a finite number > 0, got {self.spot}"
            )
        if not math.isfinite(self.delta):
            raise ValueError(f"{self.key}: delta must be finite, got {self.delta}")
        if not math.isfinite(self.max_loss):
            raise ValueError(
                f"{self.key}: max_loss must be finite, got {self.max_loss}"
            )
        # Optional leg fields: absent is fine, PRESENT-AND-WRONG is not.
        if self.strike is not None:
            if not math.isfinite(self.strike) or self.strike <= 0:
                raise ValueError(
                    f"{self.key}: strike must be a finite number > 0 or None, "
                    f"got {self.strike}"
                )
        if self.right is not None and self.right not in ("C", "P"):
            raise ValueError(
                f'{self.key}: right must be "C", "P" or None, got {self.right!r}'
            )
        if self.t_years is not None and not math.isfinite(self.t_years):
            raise ValueError(
                f"{self.key}: t_years must be finite or None, got {self.t_years}"
            )
        if self.iv0 is not None:
            if not math.isfinite(self.iv0) or self.iv0 <= 0:
                raise ValueError(
                    f"{self.key}: iv0 must be a finite number > 0 or None, got "
                    f"{self.iv0}"
                )
        if self.mark0 is not None and not math.isfinite(self.mark0):
            raise ValueError(
                f"{self.key}: mark0 must be finite or None, got {self.mark0}"
            )

    @property
    def exposure(self) -> float:
        """Delta-adjusted dollar exposure ``quantity × multiplier × delta × spot``
        (the ``greeks.py`` ``delta_adjusted_notional`` term for one row)."""
        return self.quantity * self.multiplier * self.delta * self.spot

    @property
    def can_full_reval(self) -> bool:
        """True when this row carries everything design §10.1 needs to
        reprice it: all five leg fields present, a LIVE tenor
        (``t_years > 0``) and a usable vol (``iv0 > 0``).

        The single dispatch predicate — :func:`position_pnl_series` and
        :func:`book_pnl_series` both read it rather than re-deriving the
        rule, so the series and its label can never disagree.
        """
        return (
            self.strike is not None
            and self.right is not None
            and self.t_years is not None
            and self.iv0 is not None
            and self.mark0 is not None
            and self.t_years > 0.0
            and self.iv0 > 0.0
        )

    @property
    def pnl_method(self) -> str:
        """The label this row's series carries — the estimator that will
        actually price it (design §10.1)."""
        return (
            METHOD_FULL_REVAL_CONST_IV
            if self.can_full_reval
            else METHOD_DELTA_LINEAR
        )


@dataclass(frozen=True)
class BookPnl:
    """Per-position and total daily P&L of a book (contract §2.9).

    - ``dates[t]``: the return date of row ``t`` (from the matrix);
    - ``per_position[key][t]``: USD P&L of that INCLUDED position on
      ``dates[t]`` (gain-positive);
    - ``total[t]``: ``math.fsum`` of ``per_position[*][t]``;
    - ``method``: the BOOK-LEVEL summary (design §10.3) —
      ``"FULL_REVAL_CONST_IV"`` when AT LEAST ONE included position was
      full-revalued, ``"DELTA_LINEAR"`` otherwise. A mixed book therefore
      reports the STRONGER label, which is only honest because
      ``method_by_key`` sits right next to it and says exactly which rows
      it applies to. A one-word summary of a mixed book cannot be precise;
      the per-key map is the precise answer, and every surface that serves
      ``method`` serves ``method_by_key`` with it;
    - ``method_by_key[key]``: the estimator that actually priced that
      INCLUDED row — ``"FULL_REVAL_CONST_IV"`` or ``"DELTA_LINEAR"``.
      Excluded keys are absent (they were priced by nothing);
    - ``tickers_missing``: sorted distinct tickers with no matrix column
      (their positions are EXCLUDED — an honest gap, not zeros);
    - ``keys_excluded``: the excluded position keys, in input order.

    An empty book (no included positions) has ``total == [0.0] * n_obs``,
    ``method == "DELTA_LINEAR"`` (nothing was revalued) and an empty
    ``method_by_key``.
    """

    dates: tuple[date, ...]
    per_position: dict[str, list[float]]
    total: list[float]
    method: str
    tickers_missing: tuple[str, ...]
    keys_excluded: tuple[str, ...] = ()
    #: ADDITIVE (design §10.2/§10.3). Defaults to an empty map so a
    #: :class:`BookPnl` built positionally by older code still constructs.
    method_by_key: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.dates)

    @property
    def as_of(self) -> date | None:
        return self.dates[-1] if self.dates else None

    @property
    def method_counts(self) -> dict[str, int]:
        """How many INCLUDED rows each estimator priced (design §10.3) —
        both labels always present, so a reader never has to distinguish
        "zero full-revals" from "the key is missing"."""
        counts = {METHOD_FULL_REVAL_CONST_IV: 0, METHOD_DELTA_LINEAR: 0}
        for method in self.method_by_key.values():
            counts[method] = counts.get(method, 0) + 1
        return counts


def book_method_summary(method_by_key: Mapping[str, str]) -> str:
    """The book-level ``method`` label from the per-key map (design §10.3).

    ``FULL_REVAL_CONST_IV`` when at least one row full-revalued, else
    ``DELTA_LINEAR``. One function so the library, the gateway API and the
    persisted ``risk_snapshots.pnl_method`` column can never drift apart.
    """
    if any(m == METHOD_FULL_REVAL_CONST_IV for m in method_by_key.values()):
        return METHOD_FULL_REVAL_CONST_IV
    return METHOD_DELTA_LINEAR


def _require_simple(returns: ReturnMatrix) -> None:
    if returns.return_type != RETURN_TYPE_SIMPLE:
        raise ValueError(
            f"P&L construction requires {RETURN_TYPE_SIMPLE!r} returns, got "
            f"{returns.return_type!r} (pnl = exposure × r is exact only for "
            "simple returns)"
        )


def _full_reval_series(
    pos: PositionRiskInput, column: Sequence[float]
) -> list[float]:
    """FULL_REVAL_CONST_IV daily P&L of one option position (design §10.1).

    ``pnl_t = quantity × multiplier × [BS(spot·(1+r_t)) − BS(spot)]`` with
    ``K``, ``right``, ``T0`` and ``iv0`` held fixed on both sides.

    The baseline ``BS(spot)`` is computed ONCE and every observation
    differences against that same value, which is what makes ``r_t = 0``
    return ``0.0`` exactly and makes the ``mark0 − model0`` basis cancel
    exactly rather than to within an ulp (the Phase D lesson, design §8.2).
    ``mark0`` is deliberately absent from the arithmetic: adding it and
    subtracting it again is what introduces the ulp.

    A return of ``r_t <= -1`` (the underlying at or below zero) is not a
    tradable state and ``bs_price`` rejects the non-positive spot. It is
    also UNREACHABLE from the real pipeline — ``returns_from_closes``
    requires every close ``> 0``, so ``r_t = c_t / c_{t-1} - 1 > -1``
    strictly — but a hand-built matrix can carry one, and losing a whole
    book's series to one impossible bar would be worse than pricing it.
    Such an observation is priced at the ANALYTIC LIMIT of ``bs_price`` as
    ``spot -> 0+``, so the floor is continuous with the branch beside it:
    ``0`` for a call, and ``K * exp(-r * T)`` — the DISCOUNTED strike, not
    ``K`` — for a put. (Undiscounted ``K`` would overstate a long put by
    ``K * (1 - exp(-r * T))``: about $199 per contract at ``K=200``,
    ``T=0.25``, and it would make the series jump discontinuously across
    ``r = -1``.)

    Caller guarantees ``pos.can_full_reval`` (all five fields, live tenor,
    positive vol), so the ``float()`` narrowings below cannot see ``None``.
    """
    strike = float(pos.strike)  # type: ignore[arg-type]
    right = str(pos.right)
    t_years = float(pos.t_years)  # type: ignore[arg-type]
    iv0 = float(pos.iv0)  # type: ignore[arg-type]
    scale = pos.quantity * pos.multiplier
    price0 = bs_price(
        pos.spot,
        strike,
        t_years,
        iv0,
        right,
        FULL_REVAL_RATE,
        FULL_REVAL_DIVIDEND_YIELD,
    )
    # The spot -> 0+ limit of `bs_price` under the SAME carry (see the
    # docstring): a call is worthless, a put is worth the DISCOUNTED strike.
    wiped_out = (
        strike * math.exp(-FULL_REVAL_RATE * t_years) if right == "P" else 0.0
    )
    out: list[float] = []
    for r in column:
        spot1 = pos.spot * (1.0 + r)
        if spot1 <= 0.0:
            price1 = wiped_out
        else:
            price1 = bs_price(
                spot1,
                strike,
                t_years,
                iv0,
                right,
                FULL_REVAL_RATE,
                FULL_REVAL_DIVIDEND_YIELD,
            )
        out.append(scale * (price1 - price0))
    return out


def position_pnl_series(
    pos: PositionRiskInput, returns: ReturnMatrix
) -> list[float]:
    """Daily P&L of one position — DELTA_LINEAR or FULL_REVAL_CONST_IV.

    DISPATCH (design §10.1), on the position's own data and nothing else:

    - ``pos.can_full_reval`` (``strike``, ``right``, ``t_years``, ``iv0``
      and ``mark0`` all present, ``t_years > 0``, ``iv0 > 0``) ⇒ FULL
      REVALUATION, ``pnl_t = quantity × multiplier × [BS(spot·(1+r_t)) −
      BS(spot)]`` with T and IV held at their baseline (S-convexity only —
      no theta, no vega; see the module docstring);
    - otherwise ⇒ the unchanged DELTA_LINEAR estimator,
      ``pnl_t = (quantity × multiplier × delta × spot) × r_simple_t``. A
      stock row never reaches the revaluation branch, so its output is
      byte-identical to the pre-batch function.

    ``pos.pnl_method`` names which branch ran. ``ValueError`` if the matrix
    is not ``SIMPLE``; ``KeyError`` (from :meth:`ReturnMatrix.column`) if
    the position's ticker has no column — :func:`book_pnl_series` handles
    that case by exclusion instead.

    Hand-check (DELTA_LINEAR): 100 sh long stock at spot 200, delta 1 →
    exposure 20 000; ``r = 0.01`` → ``pnl = 200.0``. Short 100 sh →
    exposure −20 000 → ``pnl = −200.0`` on the same day.

    Hand-check (FULL_REVAL_CONST_IV): 1 long ATM call, spot 100, K 100,
    T 0.25, iv 0.30 → ``r = 0`` gives EXACTLY ``0.0``, and the ±1 % pair
    sums to a POSITIVE number (gamma), where DELTA_LINEAR would sum to 0.
    """
    _require_simple(returns)
    column = returns.column(pos.ticker)
    if pos.can_full_reval:
        return _full_reval_series(pos, column)
    exposure = pos.exposure
    return [exposure * r for r in column]


def book_pnl_series(
    positions: Sequence[PositionRiskInput], returns: ReturnMatrix
) -> BookPnl:
    """Per-position and total P&L of a book (contract §2.9, design §10.2).

    Each position is priced by :func:`position_pnl_series`, which dispatches
    per row: an option row carrying its leg fields is FULL-REVALUED, every
    other row stays DELTA_LINEAR. The row's label is recorded in
    ``method_by_key`` and the book's ``method`` is the §10.3 summary
    (``FULL_REVAL_CONST_IV`` when at least one row revalued).

    ``total[t] = math.fsum(pnl_i[t])`` over the included positions —
    unchanged, and method-agnostic, which is why the Euler contribution
    identity (Σ RC = ES) still holds exactly on a mixed book.

    Positions whose ticker has no column in ``returns`` are EXCLUDED and
    named in ``tickers_missing`` / ``keys_excluded`` (honest gap; the
    caller degrades snapshot health) and carry NO ``method_by_key`` entry.
    ``ValueError`` if ``returns`` is not ``SIMPLE`` or two positions share
    a ``key``.
    """
    _require_simple(returns)
    keys = [p.key for p in positions]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"position keys must be unique, duplicated: {dupes}")

    per_position: dict[str, list[float]] = {}
    method_by_key: dict[str, str] = {}
    missing: set[str] = set()
    excluded: list[str] = []
    for pos in positions:
        if pos.ticker not in returns.tickers:
            missing.add(pos.ticker)
            excluded.append(pos.key)
            continue
        per_position[pos.key] = position_pnl_series(pos, returns)
        method_by_key[pos.key] = pos.pnl_method

    n = returns.n_obs
    series = list(per_position.values())
    total = [math.fsum(s[t] for s in series) for t in range(n)]
    return BookPnl(
        dates=returns.dates,
        per_position=per_position,
        total=total,
        method=book_method_summary(method_by_key),
        tickers_missing=tuple(sorted(missing)),
        keys_excluded=tuple(excluded),
        method_by_key=method_by_key,
    )
