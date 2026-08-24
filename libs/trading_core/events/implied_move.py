"""Implied move & IV-crush arithmetic (event spec §18, §36, §37, §47, §66;
audit options section; Phase I unit U2).

Pure stdlib, deterministic, **no I/O** — like the rest of
``libs/trading_core/events/`` this module may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4). The gateway
seam (``apps/gateway/event_options.py``) fetches option quotes or daily
bars and hands this module plain floats; nothing here knows what a provider
is.

Four ideas carry the module:

1. **The straddle IS the implied move** (§18, §36). The market's own price
   for "the underlying moves *something* by expiry" is the at-the-money
   call plus the at-the-money put. :func:`straddle_implied_move` divides
   that sum by spot and stops — no vol model, no annualization, no
   distributional assumption. :func:`implied_move_from_iv` gives the
   independent ``iv * sqrt(dte/365)`` cross-check for the cases where an
   IV is available, and the two are deliberately kept as separate methods
   (``"ATM_STRADDLE"`` / ``"IV_SQRT_T"``) so a payload never blends them.

2. **A move is pricing, not a forecast** (§37). :data:`DISCLAIMER` is a
   fixed string that the API payload and the UI both render verbatim. It
   lives here, next to the arithmetic, so there is exactly one wording.

3. **The basis label travels with the number** (audit; house rule). An
   implied move reconstructed from option *daily closes* is a different
   claim from one read off a live chain snapshot: the close-based figure
   mixes a stale mark with a stale spot and never saw a bid/ask.
   :data:`BASIS_HISTORICAL` and :data:`BASIS_LIVE` are the only two labels,
   and :func:`build_summary` refuses to emit a number without one.

4. **Absence is a value** (house rule; §85). Every numeric field is
   ``float | None``. A missing leg does NOT become a zero and does NOT
   halve the straddle: :func:`build_summary` returns status ``NO_DATA`` or
   ``PARTIAL`` with a note naming the leg. There is no NaN and no ``inf``
   anywhere in a returned value — :func:`_finite` is the single gate.

Expiry selection (§18). :func:`select_event_expiry` takes the first expiry
at or after the event date, with one session-dependent correction: an
``AFTER_MARKET`` release on a Friday is priced *after* that Friday's
options have already expired, so a same-day expiry cannot span the event
and is skipped. ``DURING_MARKET`` is treated the same way — by the close
the news is already in the price, so the same-day contract expires on
information the event has already delivered. ``BEFORE_MARKET`` and
``UNKNOWN`` keep the same-day expiry (a BMO print genuinely is priced by
that day's close; UNKNOWN takes the more inclusive branch and the caller
labels it).

Rate convention. Every IV solve here uses ``r=0.04`` by default and
``q=0.0``, matching :mod:`libs.trading_core.options.iv`. The platform has
no dividend-yield seam and does not invent one, so ``q=0`` is a documented
approximation that biases call IV slightly high and put IV slightly low on
a dividend payer — it is not a measured yield.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from libs.trading_core.options.iv import implied_vol

__all__ = [
    "BASIS_HISTORICAL",
    "BASIS_LIVE",
    "CLASSIFICATION_FAIR",
    "CLASSIFICATION_OVER",
    "CLASSIFICATION_UNDER",
    "DAYS_PER_YEAR",
    "DEFAULT_RATE",
    "DISCLAIMER",
    "IMPLIED_MOVE_MODEL_VERSION",
    "METHOD_IV_SQRT_T",
    "METHOD_STRADDLE",
    "OVER_PRICED_RATIO",
    "RATIO_BAND_EPSILON",
    "STATUS_NO_DATA",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "UNDER_PRICED_RATIO",
    "ImpliedMove",
    "ImpliedMoveSummary",
    "build_summary",
    "event_iv",
    "historical_move_stats",
    "implied_move_from_iv",
    "implied_vs_realized",
    "iv_crush",
    "nearest_strike",
    "select_event_expiry",
    "straddle_implied_move",
]

# ---------------------------------------------------------------------------
# Fixed strings & constants (these travel into the API payload and the UI)
# ---------------------------------------------------------------------------

#: Label for numbers derived from a CURRENT option-chain snapshot: real
#: bid/ask-derived marks, spot and quote from the same instant.
BASIS_LIVE = "LIVE_CHAIN_SNAPSHOT"

#: Label for numbers reconstructed from option DAILY CLOSES around a past
#: event. An approximation on purpose: a daily close is a last print, not a
#: mid, and the underlying close is a different instant from the option's
#: last trade. Never presented as a live quote (audit options section).
BASIS_HISTORICAL = "HISTORICAL_DAILY_CLOSE_APPROXIMATION"

#: §37 wording. Rendered verbatim in the API ``disclaimer`` field and in the
#: UI Options tab — one string, one place, so the two cannot drift.
DISCLAIMER = (
    "Implied move is option-market pricing, not a forecast of the move."
)

#: :attr:`ImpliedMove.method` values. Two methods, never blended.
METHOD_STRADDLE = "ATM_STRADDLE"
METHOD_IV_SQRT_T = "IV_SQRT_T"

#: :attr:`ImpliedMoveSummary.status` values.
STATUS_OK = "OK"
STATUS_PARTIAL = "PARTIAL"
STATUS_NO_DATA = "NO_DATA"

#: :func:`implied_vs_realized` classification labels and their thresholds on
#: ``actual / implied``. Below 0.8 the market paid for more movement than it
#: got (the option was OVER_PRICED); above 1.2 the move exceeded what was
#: paid for (UNDER_PRICED); the band between is FAIR. The thresholds are a
#: display convention, not a signal — §37 forbids reading either label as a
#: prediction about the next event.
UNDER_PRICED_RATIO = 1.2
OVER_PRICED_RATIO = 0.8
CLASSIFICATION_UNDER = "UNDER_PRICED"
CLASSIFICATION_FAIR = "FAIR"
CLASSIFICATION_OVER = "OVER_PRICED"

#: Comparison slack on the two band edges. Binary floating point cannot
#: represent 0.08/0.10 exactly — it evaluates to 0.7999999999999999 — so a
#: naive ``ratio < 0.8`` would label a textbook boundary case OVER_PRICED
#: and the docstring's "exactly 0.8 is FAIR" would be a lie. One part in
#: 1e-9 is far below any move the platform can measure and far above double
#: rounding error, so it changes only the label of a tie.
RATIO_BAND_EPSILON = 1e-9

#: Calendar days per year for the ``sqrt(t)`` scaling — calendar, not
#: trading, because an option's time value decays over weekends too.
DAYS_PER_YEAR = 365.0

#: Risk-free rate used by :func:`event_iv` when the caller supplies none.
#: The platform has no treasury-yield seam wired into this layer, so 4 % is
#: a documented constant, not a measured rate (contract: do not add a seam
#: just for this).
DEFAULT_RATE = 0.04

#: Bumped whenever the arithmetic changes, so a stored metric row says which
#: version produced it (mirrors ``REPLAY_MODEL_VERSION``).
IMPLIED_MOVE_MODEL_VERSION = "implied_move-1.0.0"

#: Sessions whose same-day expiry cannot span the event — see the module
#: docstring. Compared against the session's string value so the caller may
#: pass an :class:`~libs.trading_core.models.enums.EventSession` or a plain
#: string without this module importing the enum's identity.
_SAME_DAY_EXPIRY_INVALID = frozenset({"AFTER_MARKET", "DURING_MARKET"})


# ---------------------------------------------------------------------------
# Small helpers — every one of them refuses to fabricate
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """``None`` unless the value is a finite float (no NaN, no ±inf)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _positive(value: float | None) -> float | None:
    """``None`` unless the value is finite and strictly positive.

    Used for prices, spots and strikes: a zero or negative mark is
    malformed data, not a number to divide by.
    """
    number = _finite(value)
    if number is None or number <= 0.0:
        return None
    return number


def _session_value(session: object) -> str:
    """The session as an upper-case string, whatever type it arrived as.

    Accepts an :class:`EventSession` (a ``StrEnum``), a plain string or
    ``None`` — ``None`` reads as ``"UNKNOWN"``, the inclusive branch.
    """
    if session is None:
        return "UNKNOWN"
    value = getattr(session, "value", session)
    return str(value).upper()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpliedMove:
    """One implied move, in dollars and as a fraction of spot (§18, §36).

    ``pct`` is a FRACTION (0.062 = 6.2 %), matching the rest of the event
    layer's return conventions — the UI multiplies by 100, the payload does
    not. ``points`` is the same move in dollars per share of the underlying
    (``pct * spot``), which is what a trader compares against a strike
    width. ``method`` is :data:`METHOD_STRADDLE` or :data:`METHOD_IV_SQRT_T`
    and never a blend of the two.
    """

    points: float | None
    pct: float | None
    method: str
    spot: float | None = None
    inputs: Mapping[str, float | None] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", dict(self.inputs))

    def to_dict(self) -> dict:
        """JSON-ready mapping. Keys are stable API field names."""
        return {
            "points": self.points,
            "pct": self.pct,
            "method": self.method,
            "spot": self.spot,
            "inputs": dict(self.inputs),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImpliedMoveSummary:
    """The full pre/post option picture for one event (§18, §36, §37, §66).

    ``pre`` is the implied move priced BEFORE the event (the number the
    market charged for the print); ``post`` is the same straddle re-priced
    after it, which is what makes ``iv_crush_pct`` meaningful. ``ratio`` is
    ``|actual| / implied`` and ``classification`` its banded label — both
    ``None`` when either side is missing, never a zero.

    ``status`` is :data:`STATUS_OK` (both legs on both sides),
    :data:`STATUS_PARTIAL` (a usable pre-event move but an incomplete post
    side, or vice versa) or :data:`STATUS_NO_DATA` (no implied move at all).
    ``notes`` names every absence — a caller can always answer "why is this
    None" from the payload alone.
    """

    basis: str
    status: str
    pre: ImpliedMove | None = None
    post: ImpliedMove | None = None
    iv_before: float | None = None
    iv_after: float | None = None
    iv_crush_pct: float | None = None
    actual_move_pct: float | None = None
    ratio: float | None = None
    classification: str | None = None
    strike: float | None = None
    expiry: date | None = None
    event_date: date | None = None
    session: str | None = None
    dte_days: float | None = None
    disclaimer: str = DISCLAIMER
    model_version: str = IMPLIED_MOVE_MODEL_VERSION
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", dict(self.notes))

    def to_dict(self) -> dict:
        """JSON-ready mapping, in payload order.

        Dates render as ISO strings so the gateway can hand this straight to
        FastAPI without a second walk.
        """
        return {
            "basis": self.basis,
            "status": self.status,
            "pre": self.pre.to_dict() if self.pre is not None else None,
            "post": self.post.to_dict() if self.post is not None else None,
            "iv_before": self.iv_before,
            "iv_after": self.iv_after,
            "iv_crush_pct": self.iv_crush_pct,
            "actual_move_pct": self.actual_move_pct,
            "ratio": self.ratio,
            "classification": self.classification,
            "strike": self.strike,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "event_date": (
                self.event_date.isoformat() if self.event_date else None
            ),
            "session": self.session,
            "dte_days": self.dte_days,
            "disclaimer": self.disclaimer,
            "model_version": self.model_version,
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# §18 — expiry and strike selection
# ---------------------------------------------------------------------------


def select_event_expiry(
    event_date: date,
    expiries: Iterable[date],
    *,
    session: object = None,
) -> date | None:
    """The first listed expiry that still spans the event (§18).

    Returns the earliest expiry ``>= event_date``, except for an
    ``AFTER_MARKET`` or ``DURING_MARKET`` release, where the same-day
    contract expires before (or on the same close as) the news and therefore
    cannot price it — those take the earliest expiry ``> event_date``. A
    ``BEFORE_MARKET`` release IS priced by that day's close, so its same-day
    expiry is kept, and an ``UNKNOWN`` session takes that same inclusive
    branch (the caller labels the assumption; this module does not guess a
    session it was not given).

    ``None`` when no candidate qualifies — an empty list, or every expiry
    already past. Duplicates and unsorted input are fine; non-``date``
    entries raise ``TypeError`` (contract: bad input never silently
    produces a number).
    """
    candidates: list[date] = []
    for expiry in expiries:
        if not isinstance(expiry, date):
            raise TypeError(f"expiries must be date objects, got {expiry!r}")
        candidates.append(expiry)
    if not candidates:
        return None
    strict = _session_value(session) in _SAME_DAY_EXPIRY_INVALID
    qualifying = [
        expiry
        for expiry in candidates
        if (expiry > event_date if strict else expiry >= event_date)
    ]
    if not qualifying:
        return None
    return min(qualifying)


def nearest_strike(spot: float, strikes: Iterable[float]) -> float | None:
    """The listed strike closest to ``spot`` — the ATM leg (§18).

    Ties (spot exactly between two strikes) resolve to the LOWER strike, a
    fixed rule so the same chain always yields the same straddle: an
    arbitrary tiebreak would make a stored metric irreproducible.

    Non-finite or non-positive strikes are skipped rather than raising (a
    chain row with a null strike is data noise, not a programming error);
    ``None`` when nothing usable survives or ``spot`` is not a positive
    finite number.
    """
    reference = _positive(spot)
    if reference is None:
        return None
    usable = [s for s in (_positive(x) for x in strikes) if s is not None]
    if not usable:
        return None
    return min(usable, key=lambda s: (abs(s - reference), s))


# ---------------------------------------------------------------------------
# §36 — the two implied-move estimators
# ---------------------------------------------------------------------------


def straddle_implied_move(
    call_px: float | None, put_px: float | None, spot: float | None
) -> ImpliedMove:
    """ATM straddle implied move: ``(call + put) / spot`` (§18, §36).

    The market's own price for "it moves by expiry", read directly off the
    two at-the-money legs — no vol model and no distributional assumption,
    which is exactly why §37 calls it *pricing*.

    Returns an :class:`ImpliedMove` with ``pct``/``points`` ``None`` and a
    ``reason`` naming the missing or malformed input whenever a leg or the
    spot is absent, zero, negative or non-finite. A missing leg is never
    treated as a free option: half a straddle is not an implied move.
    """
    call = _finite(call_px)
    put = _finite(put_px)
    reference = _positive(spot)
    inputs = {"call_px": call, "put_px": put, "spot": reference}

    missing = [
        name
        for name, value in (("call", call), ("put", put))
        if value is None or value < 0.0
    ]
    if missing:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_STRADDLE,
            spot=reference,
            inputs=inputs,
            reason=(
                f"missing_or_negative_leg: {', '.join(missing)} — a straddle "
                "needs both legs; half of one is not an implied move"
            ),
        )
    if reference is None:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_STRADDLE,
            spot=None,
            inputs=inputs,
            reason=f"spot_not_positive: got {spot!r}",
        )

    assert call is not None and put is not None  # narrowed by `missing`
    points = _finite(call + put)
    if points is None:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_STRADDLE,
            spot=reference,
            inputs=inputs,
            reason="straddle_sum_not_finite",
        )
    return ImpliedMove(
        points=points,
        pct=_finite(points / reference),
        method=METHOD_STRADDLE,
        spot=reference,
        inputs=inputs,
        reason=None,
    )


def implied_move_from_iv(
    iv: float | None, dte_days: float | None, *, spot: float | None = None
) -> ImpliedMove:
    """IV-based implied move: ``iv * sqrt(dte / 365)`` (§36 cross-check).

    The one-standard-deviation move to expiry under the lognormal
    convention, kept as an INDEPENDENT estimator of the straddle: the two
    disagree by roughly the usual ~0.8 factor (a straddle prices the mean
    absolute move, ``sqrt(2/pi) * sigma``, not one sigma), so a payload
    showing both must label which is which — hence the distinct
    :data:`METHOD_IV_SQRT_T`.

    ``dte_days`` is CALENDAR days (weekends decay too). ``points`` is filled
    only when a positive ``spot`` is supplied; otherwise the percentage
    stands alone. Non-positive ``iv`` or ``dte_days`` yields ``None`` with a
    reason — a zero-DTE option has no time value left to imply a move from.
    """
    vol = _finite(iv)
    days = _finite(dte_days)
    reference = _positive(spot)
    inputs = {"iv": vol, "dte_days": days, "spot": reference}

    if vol is None or vol <= 0.0:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_IV_SQRT_T,
            spot=reference,
            inputs=inputs,
            reason=f"iv_not_positive: got {iv!r}",
        )
    if days is None or days <= 0.0:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_IV_SQRT_T,
            spot=reference,
            inputs=inputs,
            reason=(
                f"dte_not_positive: got {dte_days!r} — an expired option has "
                "no time value left to imply a move from"
            ),
        )

    pct = _finite(vol * math.sqrt(days / DAYS_PER_YEAR))
    if pct is None:
        return ImpliedMove(
            points=None,
            pct=None,
            method=METHOD_IV_SQRT_T,
            spot=reference,
            inputs=inputs,
            reason="iv_move_not_finite",
        )
    return ImpliedMove(
        points=_finite(pct * reference) if reference is not None else None,
        pct=pct,
        method=METHOD_IV_SQRT_T,
        spot=reference,
        inputs=inputs,
        reason=None,
    )


def event_iv(
    price: float | None,
    spot: float | None,
    strike: float | None,
    t_years: float | None,
    right: str,
    *,
    r: float = DEFAULT_RATE,
    q: float = 0.0,
) -> float | None:
    """Implied volatility of one option mark, or ``None`` (§36).

    A thin, honest wrapper over
    :func:`libs.trading_core.options.iv.implied_vol`: the solver's
    ``IVResult.iv`` when it solved, ``None`` on every unsolvable price. The
    rate defaults to :data:`DEFAULT_RATE` (4 %, a documented constant, not a
    measured yield) and ``q`` to 0 — see the module docstring for the bias
    that introduces on a dividend payer.

    Unlike the solver this never raises: malformed input (non-positive
    spot/strike, unknown ``right``, non-finite price) returns ``None``,
    because the caller is a backfill loop over vendor rows, not a
    programmer's expression. The solved IV is INTERNAL DETERMINISTIC and
    must be labelled as such wherever it is displayed
    (``docs/data-source-architecture.md`` §12) — it is never a vendor IV.
    """
    mark = _finite(price)
    reference = _positive(spot)
    k = _positive(strike)
    t = _finite(t_years)
    if mark is None or reference is None or k is None or t is None:
        return None
    if right not in ("C", "P"):
        return None
    if t <= 0.0 or mark <= 0.0:
        return None
    try:
        result = implied_vol(mark, reference, k, t, right, r=r, q=q)
    except ValueError:
        return None
    return _finite(result.iv)


def iv_crush(iv_before: float | None, iv_after: float | None) -> float | None:
    """Fractional change in implied volatility across the event (§36, §66).

    ``iv_after / iv_before - 1`` — negative for the usual post-print crush
    (``-0.40`` means IV fell 40 %). A fraction, not percentage points: the
    UI multiplies by 100.

    ``None`` when either side is missing or ``iv_before`` is not strictly
    positive; a zero base is malformed data, not a divide-by-zero to catch.
    """
    before = _positive(iv_before)
    after = _finite(iv_after)
    if before is None or after is None or after < 0.0:
        return None
    return _finite(after / before - 1.0)


# ---------------------------------------------------------------------------
# §66 — implied versus realized
# ---------------------------------------------------------------------------


def implied_vs_realized(
    implied_pct: float | None, actual_move_pct: float | None
) -> tuple[float | None, str | None]:
    """``(ratio, classification)`` for ``|actual| / implied`` (§66).

    Both inputs are FRACTIONS of spot. The sign of the realized move is
    dropped on purpose: a straddle is direction-agnostic, so the comparable
    quantity is the absolute move.

    Classification bands — a display convention, never a prediction (§37):

    - ratio ``> 1.2`` → :data:`CLASSIFICATION_UNDER` (the move exceeded what
      the option market charged for it);
    - ratio ``< 0.8`` → :data:`CLASSIFICATION_OVER`;
    - anything in ``[0.8, 1.2]`` → :data:`CLASSIFICATION_FAIR`.

    Boundaries belong to FAIR: exactly 0.8 and exactly 1.2 are FAIR, so the
    two extreme labels are strictly beyond the band. The comparison carries
    :data:`RATIO_BAND_EPSILON` of slack because ``0.08 / 0.10`` is
    ``0.7999999999999999`` in binary floating point — without it a textbook
    boundary case would be labelled OVER_PRICED.

    Returns ``(None, None)`` when either input is missing or ``implied_pct``
    is not strictly positive — there is no ratio against a zero implied
    move, and no label without a ratio.
    """
    implied = _positive(implied_pct)
    actual = _finite(actual_move_pct)
    if implied is None or actual is None:
        return None, None
    ratio = _finite(abs(actual) / implied)
    if ratio is None:
        return None, None
    if ratio > UNDER_PRICED_RATIO + RATIO_BAND_EPSILON:
        return ratio, CLASSIFICATION_UNDER
    if ratio < OVER_PRICED_RATIO - RATIO_BAND_EPSILON:
        return ratio, CLASSIFICATION_OVER
    return ratio, CLASSIFICATION_FAIR


def historical_move_stats(moves: Sequence[float] | None) -> dict:
    """|Move| distribution over past events (§19, §64, §66).

    Keys are always present so a caller never branches on shape:
    ``{"median_abs", "p90_abs", "max_abs", "n"}``. ``n`` counts the USABLE
    (finite) inputs, not the length of the argument — a list with three
    ``None`` holes reports ``n=0`` and all-``None`` statistics rather than a
    median over the survivors' positions.

    Percentiles use the NEAREST-RANK definition (rank ``ceil(p/100 * n)``,
    no interpolation), matching
    :func:`libs.trading_core.events.reaction.percentile_nearest_rank`, so a
    reported p90 is always a move that actually happened. All values are
    absolute; a single-element sample reports that element for all three
    statistics, which is honest given ``n=1`` travels alongside.
    """
    usable = [
        abs(value)
        for value in (_finite(m) for m in (moves or ()))
        if value is not None
    ]
    n = len(usable)
    if n == 0:
        return {"median_abs": None, "p90_abs": None, "max_abs": None, "n": 0}
    ordered = sorted(usable)
    return {
        "median_abs": _nearest_rank(ordered, 50.0),
        "p90_abs": _nearest_rank(ordered, 90.0),
        "max_abs": ordered[-1],
        "n": n,
    }


def _nearest_rank(ordered: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an ALREADY-ASCENDING non-empty sample.

    Local rather than imported so this module stays importable without
    pulling in ``reaction``'s indicator dependencies; the definition is
    identical (``reaction.percentile_nearest_rank``) and pinned by test.
    """
    rank = math.ceil(pct / 100.0 * len(ordered))
    rank = max(1, min(len(ordered), rank))
    return float(ordered[rank - 1])


# ---------------------------------------------------------------------------
# The builder — the only place statuses and notes are decided
# ---------------------------------------------------------------------------


def build_summary(
    pre_call_close: float | None,
    pre_put_close: float | None,
    spot: float | None,
    post_call_close: float | None,
    post_put_close: float | None,
    post_spot: float | None,
    *,
    strike: float | None,
    expiry: date | None,
    event_date: date | None,
    session: object = None,
    actual_move_pct: float | None = None,
    dte_days: float | None = None,
    r: float = DEFAULT_RATE,
) -> ImpliedMoveSummary:
    """Assemble the pre/post implied-move picture for one event (§18, §36).

    Inputs are option marks (per share, the same units as ``spot``) for the
    ATM straddle on ``strike``/``expiry``, taken from the last knowable
    pre-event close and the first post-event close respectively, plus the
    underlying at each of those two instants. The basis is always
    :data:`BASIS_HISTORICAL`: this builder reconstructs a past event from
    closes. Live-chain callers build their own :class:`ImpliedMoveSummary`
    with :data:`BASIS_LIVE`.

    What it decides, and nothing more:

    - the pre-event straddle move (:func:`straddle_implied_move`);
    - the post-event straddle, whose only purpose here is the IV pair;
    - ``iv_before``/``iv_after`` via :func:`event_iv` on the CALL leg at
      ``dte_days`` and at ``max(dte - 1, 0)`` calendar days respectively
      (the post mark is one session later), and their
      :func:`iv_crush` — both ``None`` unless ``dte_days`` is supplied and
      positive, since an IV solve needs a time to expiry;
    - the §66 ratio and classification against ``actual_move_pct``.

    Status: :data:`STATUS_OK` when the pre-event move computed AND the post
    side produced a usable straddle; :data:`STATUS_PARTIAL` when the
    pre-event move computed but the post side did not (the implied move is
    still real — only the crush is missing); :data:`STATUS_NO_DATA` when no
    pre-event implied move exists. Every ``None`` gets a ``notes`` entry.
    """
    notes: dict[str, str] = {}
    strike_value = _positive(strike)
    if strike is not None and strike_value is None:
        notes["strike"] = f"strike_not_positive: got {strike!r}"

    pre = straddle_implied_move(pre_call_close, pre_put_close, spot)
    if pre.reason is not None:
        notes["pre"] = pre.reason

    post = straddle_implied_move(post_call_close, post_put_close, post_spot)
    if post.reason is not None:
        notes["post"] = post.reason

    # --- IV pair: only meaningful with a real time-to-expiry -------------
    dte = _finite(dte_days)
    iv_before: float | None = None
    iv_after: float | None = None
    if dte is None or dte <= 0.0:
        notes["iv"] = (
            f"dte_missing_or_not_positive: got {dte_days!r} — an implied "
            "volatility solve needs a positive time to expiry"
        )
    elif strike_value is None:
        notes.setdefault(
            "iv", "strike_missing: an implied volatility solve needs a strike"
        )
    else:
        iv_before = event_iv(
            pre_call_close, spot, strike_value, dte / DAYS_PER_YEAR, "C", r=r
        )
        if iv_before is None:
            notes["iv_before"] = (
                "unsolvable: the pre-event call close is outside the "
                "Black-Scholes price range for any positive volatility"
            )
        post_dte = max(dte - 1.0, 0.0)
        if post_dte <= 0.0:
            notes["iv_after"] = (
                "expired_after_event: the straddle expires within one "
                "session of the event, so no post-event volatility remains"
            )
        else:
            iv_after = event_iv(
                post_call_close,
                post_spot,
                strike_value,
                post_dte / DAYS_PER_YEAR,
                "C",
                r=r,
            )
            if iv_after is None:
                notes["iv_after"] = (
                    "unsolvable: the post-event call close is outside the "
                    "Black-Scholes price range for any positive volatility"
                )

    crush = iv_crush(iv_before, iv_after)
    if crush is None and "iv_crush" not in notes:
        notes["iv_crush"] = (
            "needs_both_iv: implied volatility is missing on at least one "
            "side of the event"
        )

    actual = _finite(actual_move_pct)
    if actual_move_pct is not None and actual is None:
        notes["actual_move_pct"] = (
            f"not_finite: got {actual_move_pct!r}"
        )
    elif actual is None:
        notes["actual_move_pct"] = "not_supplied"

    ratio, classification = implied_vs_realized(pre.pct, actual)
    if ratio is None:
        notes.setdefault(
            "ratio",
            "needs_implied_and_actual: the implied-vs-realized ratio needs a "
            "positive pre-event implied move and a realized move",
        )

    if pre.pct is None:
        status = STATUS_NO_DATA
    elif post.pct is None:
        status = STATUS_PARTIAL
    else:
        status = STATUS_OK

    return ImpliedMoveSummary(
        basis=BASIS_HISTORICAL,
        status=status,
        pre=pre,
        post=post,
        iv_before=iv_before,
        iv_after=iv_after,
        iv_crush_pct=crush,
        actual_move_pct=actual,
        ratio=ratio,
        classification=classification,
        strike=strike_value,
        expiry=expiry,
        event_date=event_date,
        session=_session_value(session) if session is not None else None,
        dte_days=dte,
        notes=notes,
    )
