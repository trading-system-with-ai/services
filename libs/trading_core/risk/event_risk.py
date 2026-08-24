"""Event risk — discrete JUMP risk around a scheduled catalyst (event spec
§62–§67; Phase K contract U1).

Pure, deterministic, stdlib-only (house rule). This module does NOT replace
VaR/ES/GARCH: those measure the DIFFUSIVE risk of a book that trades every
day. An earnings print is a different animal — a single scheduled instant at
which the underlying gaps by an amount the daily-return history never
sampled — so §62 asks for a SEPARATE view sitting beside the statistical
one, not folded into it.

Three hard rules, each of which is load-bearing and each of which is
re-proved by a test in ``tests/test_risk_event.py``:

1.  **SHADOW by construction** (§65: "Do not initially block trades until
    backtesting validates event rules. Start in SHADOW mode."). Nothing
    here calls, patches or is called by
    :func:`~libs.trading_core.risk.engine.assess`. :func:`event_risk_caps`
    emits :class:`~libs.trading_core.risk.pretrade.QuantityCap` rows — the
    ONE cap shape ``assess(extra_caps=...)`` understands — but emitting a
    cap and BINDING a cap are different acts: a cap binds only where a
    caller deliberately passes it through ``extra_caps``, and the gateway
    deliberately does not. :class:`EventRiskPolicy` records ``mode="SHADOW"``
    until a human promotes it.
2.  **No LLM assigns the state** (§63: "Do not let LLM alone assign this
    state."). :func:`classify_event_risk` is a table. Given the same inputs
    it returns the same state, forever, and the table is written out below
    in prose so a reader can check the code against the documentation
    without running it. There is no model call, no prompt, no network.
3.  **Every historical statistic carries its sample size** (§64: "Be
    explicit about sample size: based on 8 events. Never imply statistical
    certainty from eight observations."). :func:`historical_event_risk`
    returns ``n`` in the same mapping as the statistics — they cannot be
    read apart — and ``n`` travels into the snapshot, into the caveats and
    into every cap sentence that leans on a historical number.

The honesty rule that follows from (3) and matters most: when there is no
implied move AND no historical sample, the state is
:data:`STATE_UNKNOWN`, not ``LOW``. An absent measurement is not a
measurement of absence. "We don't know how much this thing moves" is the
single most dangerous state to render as "LOW", so it gets its own label
and a reason string naming what was missing.

Percentile convention: NEAREST-RANK (rank ``ceil(p/100 · n)``, no
interpolation), identical to
:func:`libs.trading_core.events.reaction.percentile_nearest_rank` and
:func:`libs.trading_core.events.implied_move.historical_move_stats`. A
reported p90 is therefore always a move that ACTUALLY HAPPENED, which is
the property that lets the UI say "based on 8 events" without hedging.

Units. Every "move" and every "pct" in this module is a PERCENT number:
``7.1`` means 7.1%, not 0.071. ``historical_moves`` are SIGNED (a −7.1%
print is ``-7.1``); the statistics take absolute values, because §64 asks
for a magnitude distribution. ``exposure_share`` is likewise a percent of
NAV. Mixing the two conventions is the easiest bug to write here, so
:func:`classify_event_risk` never multiplies a share by a move.

RESEARCH DEFAULTS, UNVALIDATED: every threshold on
:class:`EventRiskThresholds` and :class:`EventRiskPolicy` is a parameter
with a default chosen from the §63/§65 sketch, not from a backtest. §65
requires exactly that backtest before any of this may gate a trade.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .pretrade import LAYER_CONCENTRATION, MODE_SHADOW, QuantityCap

#: Version stamped into every snapshot so a stored payload names the table
#: that produced it (the thresholds are RESEARCH DEFAULTS and will move).
EVENT_RISK_MODEL_VERSION = "event-risk-1.0.0"

#: §63 states, ordered least → most severe. :data:`STATE_UNKNOWN` is this
#: module's addition and is deliberately NOT on the ladder: it means "not
#: measured", which is not a point on a severity scale.
STATE_LOW = "LOW"
STATE_MODERATE = "MODERATE"
STATE_HIGH = "HIGH"
STATE_EXTREME = "EXTREME"
STATE_UNKNOWN = "UNKNOWN"

#: The severity ladder a bump walks along (index = severity).
STATE_LADDER = (STATE_LOW, STATE_MODERATE, STATE_HIGH, STATE_EXTREME)

#: §66 options sensitivity label. Driven by the position's option greeks,
#: NOT by the underlying's expected move — an event can be EXTREME for a
#: stock holder and irrelevant to someone holding no convexity, and the two
#: facts deserve two fields rather than one blended number.
SENSITIVITY_LOW = "LOW"
SENSITIVITY_MODERATE = "MODERATE"
SENSITIVITY_HIGH = "HIGH"

#: ``expected_move_basis`` values — which number the classifier actually
#: used. §64 honesty: the caller must always be able to tell an implied
#: move (forward-looking, one market price) from a historical median
#: (backward-looking, n observations) without guessing.
BASIS_IMPLIED = "IMPLIED"
BASIS_HISTORICAL_MEDIAN = "HISTORICAL_MEDIAN"
BASIS_NONE = "NONE"

#: Cap codes emitted by :func:`event_risk_caps`. The stem is what the engine
#: would record if a caller ever promoted the cap; the ``layer`` is
#: ``CONCENTRATION`` because what the cap limits is the SIZE of a single
#: exposure across a known jump, not a statistical tail estimate.
CODE_EVENT_EXPOSURE = "EVENT_EXPOSURE_CAP"

__all__ = [
    "BASIS_HISTORICAL_MEDIAN",
    "BASIS_IMPLIED",
    "BASIS_NONE",
    "CODE_EVENT_EXPOSURE",
    "EVENT_RISK_MODEL_VERSION",
    "SENSITIVITY_HIGH",
    "SENSITIVITY_LOW",
    "SENSITIVITY_MODERATE",
    "STATE_EXTREME",
    "STATE_HIGH",
    "STATE_LADDER",
    "STATE_LOW",
    "STATE_MODERATE",
    "STATE_UNKNOWN",
    "EventRiskInputs",
    "EventRiskPolicy",
    "EventRiskSnapshot",
    "EventRiskThresholds",
    "classify_event_risk",
    "event_risk_caps",
    "historical_event_risk",
]


def _finite(value: object) -> float | None:
    """``float(value)`` when it is a real finite number, else ``None``.

    ``bool`` is rejected on purpose: ``True`` is not a 1% move, and a
    silently-coerced flag is exactly the kind of input that would make a
    fabricated statistic look measured.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nearest_rank(ordered: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an ALREADY-ASCENDING non-empty sample.

    Same definition as ``events.reaction.percentile_nearest_rank`` and
    ``events.implied_move._nearest_rank``; kept local so this module stays
    importable from the risk package without pulling the events package in
    (the risk library must not depend on the events library), and pinned
    against theirs by test.
    """
    rank = math.ceil(pct / 100.0 * len(ordered))
    rank = max(1, min(len(ordered), rank))
    return float(ordered[rank - 1])


# ---------------------------------------------------------------------------
# §64 — the historical distribution, sample size attached
# ---------------------------------------------------------------------------


def historical_event_risk(moves: Sequence[float] | None) -> dict:
    """|Move| distribution over previous events (§64).

    Returns ``{"median_abs", "p75_abs", "p90_abs", "max_abs", "n"}`` — the
    four §64 metrics plus, in the SAME mapping, the sample size behind
    them. That co-location is the point: a caller cannot render a median
    without having ``n`` in its hand, so "7.1%" can never reach a screen
    without "based on 8 events" being available to put next to it.

    ``moves`` are SIGNED percent numbers (``-7.1`` is a 7.1% drop); the
    statistics are absolute, per §64's "median absolute move".

    ``n`` counts the USABLE (finite, non-bool) inputs, not ``len(moves)``.
    A list of eight entries with three ``None`` holes reports ``n=5`` and
    statistics over the five real prints — never a median computed over
    positions that include a hole. With no usable input at all, every
    statistic is ``None`` and ``n`` is ``0``: there is no such thing as a
    median of nothing, and ``0.0`` would read as "this never moves".

    Hand-check on ``[-8, 4, -6, 12, 2]`` (n=5, absolutes sorted
    ``[2, 4, 6, 8, 12]``): median rank ``ceil(0.5·5)=3`` → 6; p75 rank
    ``ceil(0.75·5)=4`` → 8; p90 rank ``ceil(0.9·5)=5`` → 12; max 12.
    """
    usable = [
        abs(value)
        for value in (_finite(m) for m in (moves or ()))
        if value is not None
    ]
    n = len(usable)
    if n == 0:
        return {
            "median_abs": None,
            "p75_abs": None,
            "p90_abs": None,
            "max_abs": None,
            "n": 0,
        }
    ordered = sorted(usable)
    return {
        "median_abs": _nearest_rank(ordered, 50.0),
        "p75_abs": _nearest_rank(ordered, 75.0),
        "p90_abs": _nearest_rank(ordered, 90.0),
        "max_abs": ordered[-1],
        "n": n,
    }


# ---------------------------------------------------------------------------
# Inputs, thresholds, policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRiskInputs:
    """Everything the §63 snapshot is computed from — and nothing else.

    Deliberately a plain value object with no I/O: the gateway seam
    (Phase K U2) does the DB work and hands the numbers here, which is what
    lets the whole classifier be tested without a database.

    - ``event_type``: the taxonomy string (``"EARNINGS"``, ``"FOMC_DECISION"``
      …), carried through to the payload for labelling;
    - ``time_to_event_days``: signed days from "now" to the scheduled
      instant; ``None`` when unknown. Negative means the event has passed —
      the classifier treats a passed event as no longer imminent rather
      than as "0 days away" (§67: this module is PRE_EVENT);
    - ``historical_moves``: SIGNED percent moves of PREVIOUS comparable
      events, most-recent-last (order is irrelevant to the statistics);
    - ``implied_move_pct``: the move the option market is pricing for THIS
      event, percent, ``None`` when no straddle is available;
    - ``implied_basis``: how that implied number was derived (e.g.
      ``"straddle_atm"``), carried for provenance;
    - ``position_exposure_usd``: absolute USD exposure of the position that
      would sit through the event; ``None`` when unknown;
    - ``portfolio_nav_usd``: NAV the exposure is measured against;
    - ``option_gamma`` / ``option_vega`` / ``option_theta``: NET greeks of
      the POSITION (already signed and already scaled by quantity and
      multiplier by the caller — this module never re-signs and never
      re-scales anything, exactly like ``pretrade.CandidateSpec``);
    - ``is_estimated``: the event date is an ESTIMATED cadence guess, not a
      CONFIRMED announcement. It never changes the state — an estimated
      date is not less risky — it adds a caveat, because a "1.3 days" that
      is really "some time in the next three weeks" must not be read as
      precision the row does not have.

    ``ValueError`` on a non-finite ``portfolio_nav_usd`` or a negative one:
    an exposure share divided by a bad NAV is worse than no share at all.
    """

    event_type: str | None = None
    time_to_event_days: float | None = None
    historical_moves: Sequence[float] = ()
    implied_move_pct: float | None = None
    implied_basis: str | None = None
    position_exposure_usd: float | None = None
    portfolio_nav_usd: float | None = None
    option_gamma: float | None = None
    option_vega: float | None = None
    option_theta: float | None = None
    is_estimated: bool = False

    def __post_init__(self) -> None:
        nav = self.portfolio_nav_usd
        if nav is not None:
            if isinstance(nav, bool) or not math.isfinite(float(nav)) or float(nav) < 0.0:
                raise ValueError(
                    f"portfolio_nav_usd must be a finite number >= 0 or None, got {nav!r}"
                )
        object.__setattr__(self, "historical_moves", tuple(self.historical_moves or ()))


@dataclass(frozen=True)
class EventRiskThresholds:
    """The §63 classification table, as parameters (RESEARCH DEFAULTS).

    Written out in prose here so the docstring and the code can be checked
    against each other. Given ``expected_move`` (percent) and
    ``exposure_share`` (percent of NAV):

    **Base state from the expected move** — the size of the jump the market
    or history says to expect:

        ``expected_move >= extreme_move_pct`` (12%)  → EXTREME
        ``expected_move >= high_move_pct``    (8%)   → HIGH
        ``expected_move >= moderate_move_pct``(4%)   → MODERATE
        otherwise                                    → LOW

    **Imminence bump** — a jump you are still holding TOMORROW is a
    different risk from one three weeks out. When
    ``0 <= time_to_event_days <= imminent_days`` (3d) the base state is
    bumped ONE level; when the event is further out, or already passed, or
    the date is unknown, it is not. Bumping on imminence rather than
    scaling the move is deliberate: the *size* of the expected gap does not
    shrink because it is far away, only the chance you are still holding it
    does, and that is a decision the trader makes, not the classifier.

    **Exposure bump** — the same 10% gap is a scratch on a 1%-of-NAV
    position and a catastrophe on a 30% one:

        ``exposure_share >= exposure_two_level_pct`` (25%) → bump TWO levels
        ``exposure_share >= exposure_one_level_pct`` (10%) → bump ONE level

    Bumps are cumulative and saturate at EXTREME. Nothing ever bumps a
    state DOWN — the table is monotone in every driver, which is the
    property that makes it explainable ("it is HIGH *because* …") and the
    one an eventual backtest can falsify cleanly.

    **Option greeks bump NOTHING.** They set ``sensitivity`` (§66) instead,
    on its own axis. See :class:`EventRiskSnapshot`.
    """

    moderate_move_pct: float = 4.0
    high_move_pct: float = 8.0
    extreme_move_pct: float = 12.0
    imminent_days: float = 3.0
    exposure_one_level_pct: float = 10.0
    exposure_two_level_pct: float = 25.0
    #: |vega| and |gamma| above which the §66 sensitivity label rises. Both
    #: are per-position NET greeks in the caller's own units, so these are
    #: the crudest defaults in the file and exist to be overridden.
    vega_moderate: float = 50.0
    vega_high: float = 200.0
    gamma_moderate: float = 5.0
    gamma_high: float = 20.0

    def __post_init__(self) -> None:
        if not (self.moderate_move_pct <= self.high_move_pct <= self.extreme_move_pct):
            raise ValueError(
                "move thresholds must be non-decreasing: moderate <= high <= extreme, got "
                f"{self.moderate_move_pct}, {self.high_move_pct}, {self.extreme_move_pct}"
            )
        if self.exposure_one_level_pct > self.exposure_two_level_pct:
            raise ValueError(
                "exposure_one_level_pct must be <= exposure_two_level_pct, got "
                f"{self.exposure_one_level_pct} > {self.exposure_two_level_pct}"
            )
        for name in (
            "moderate_move_pct",
            "high_move_pct",
            "extreme_move_pct",
            "imminent_days",
            "exposure_one_level_pct",
            "exposure_two_level_pct",
            "vega_moderate",
            "vega_high",
            "gamma_moderate",
            "gamma_high",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True)
class EventRiskPolicy:
    """What the event layer WOULD do about a state — SHADOW (§65).

    §65 lists WARN / RESIZE / REJECT as the available actions and then
    forbids using the last two until a backtest validates the rules. This
    dataclass encodes exactly that: ``mode`` is :data:`MODE_SHADOW`, and
    :func:`event_risk_caps` returns caps that a caller would have to hand to
    ``assess(extra_caps=...)`` on purpose for anything to bind. The gateway
    does not.

    The caps themselves express "cap the exposure crossing this event at X%
    of NAV":

        EXTREME → ``extreme_max_exposure_pct`` (5% of NAV)
        HIGH    → ``high_max_exposure_pct``   (10% of NAV)
        below HIGH → no cap at all (WARN territory only)

    ``warn_from`` names the least-severe state that produces a WARN. The
    numbers are RESEARCH DEFAULTS, UNVALIDATED — 5% and 10% are round
    numbers from the §65 sketch, not backtested thresholds, and the whole
    reason this ships in SHADOW is that nobody has earned the right to
    assert them yet.
    """

    high_max_exposure_pct: float = 10.0
    extreme_max_exposure_pct: float = 5.0
    warn_from: str = STATE_HIGH
    mode: str = MODE_SHADOW

    def __post_init__(self) -> None:
        if self.warn_from not in STATE_LADDER:
            raise ValueError(
                f"warn_from must be one of {STATE_LADDER}, got {self.warn_from!r}"
            )
        for name in ("high_max_exposure_pct", "extreme_max_exposure_pct"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if self.extreme_max_exposure_pct > self.high_max_exposure_pct:
            raise ValueError(
                "extreme_max_exposure_pct must be <= high_max_exposure_pct "
                f"(EXTREME is stricter), got {self.extreme_max_exposure_pct} > "
                f"{self.high_max_exposure_pct}"
            )


# ---------------------------------------------------------------------------
# §63 — the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRiskSnapshot:
    """The §63 event-risk snapshot: one event, one position, one verdict.

    Two independent axes, and keeping them independent is the design:

    - ``event_risk_state`` — how big the jump is relative to the position.
      Driven by expected move, imminence and exposure share;
    - ``sensitivity`` (§66) — how much the POSITION's convexity amplifies
      that jump. Driven by option gamma/vega. A stock holder with a HIGH
      event has ``sensitivity == "LOW"`` and that is correct, not a gap:
      they eat the gap linearly. Folding greeks into the state would make
      "HIGH" mean two different things depending on the instrument, and the
      §65 Trade Plan panel prints both lines separately for exactly that
      reason.

    ``drivers`` is the list of sentences that JUSTIFY the state — one per
    rule that fired, each carrying the real number, so the UI can answer
    "why HIGH?" without re-deriving anything. ``caveats`` is the list of
    reasons to trust it LESS: the sample size (§64), an ESTIMATED date, a
    missing NAV. Both are plain strings and both are always present (possibly
    empty), never ``None``.
    """

    event_type: str | None
    time_to_event_days: float | None
    historical: Mapping[str, float | int | None]
    implied: Mapping[str, object]
    expected_move_pct: float | None
    expected_move_basis: str
    position_exposure_usd: float | None
    exposure_share: float | None
    option_greeks: Mapping[str, float | None] | None
    event_risk_state: str
    sensitivity: str
    drivers: Sequence[str] = ()
    caveats: Sequence[str] = ()
    reason: str | None = None
    model_version: str = EVENT_RISK_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "historical", dict(self.historical))
        object.__setattr__(self, "implied", dict(self.implied))
        if self.option_greeks is not None:
            object.__setattr__(self, "option_greeks", dict(self.option_greeks))
        object.__setattr__(self, "drivers", tuple(self.drivers))
        object.__setattr__(self, "caveats", tuple(self.caveats))

    def to_dict(self) -> dict:
        """JSON-ready mapping, in payload order. Keys are stable API names.

        ``historical`` always carries ``n`` (§64) — including in the
        ``UNKNOWN`` case, where every statistic is ``None`` and ``n`` is
        ``0``, which is what tells a reader the difference between "no
        data" and "no movement".
        """
        return {
            "event_type": self.event_type,
            "time_to_event_days": self.time_to_event_days,
            "historical": dict(self.historical),
            "implied": dict(self.implied),
            "expected_move_pct": self.expected_move_pct,
            "expected_move_basis": self.expected_move_basis,
            "position_exposure_usd": self.position_exposure_usd,
            "exposure_share": self.exposure_share,
            "option_greeks": (
                dict(self.option_greeks) if self.option_greeks is not None else None
            ),
            "event_risk_state": self.event_risk_state,
            "sensitivity": self.sensitivity,
            "drivers": list(self.drivers),
            "caveats": list(self.caveats),
            "reason": self.reason,
            "model_version": self.model_version,
        }


def _sensitivity_from_greeks(
    gamma: float | None, vega: float | None, thresholds: EventRiskThresholds
) -> tuple[str, list[str]]:
    """§66 sensitivity label from the position's NET greeks.

    Magnitudes only — a short-gamma position is at least as exposed to a
    gap as a long-gamma one, so the label answers "how much does convexity
    amplify this print" and the SIGN of the exposure is a separate
    question the greeks themselves already answer. Absent greeks yield
    ``LOW``: a position with no options has no option sensitivity, which
    is a measurement, not a gap.
    """
    drivers: list[str] = []
    level = 0
    abs_vega = abs(vega) if vega is not None else None
    abs_gamma = abs(gamma) if gamma is not None else None
    if abs_vega is not None:
        if abs_vega >= thresholds.vega_high:
            level = max(level, 2)
            drivers.append(
                f"net vega {vega:+.1f} at/above {thresholds.vega_high:.1f} — an IV "
                "crush after the print moves this position hard (§66)"
            )
        elif abs_vega >= thresholds.vega_moderate:
            level = max(level, 1)
            drivers.append(
                f"net vega {vega:+.1f} at/above {thresholds.vega_moderate:.1f} — "
                "exposed to post-event IV crush (§66)"
            )
    if abs_gamma is not None:
        if abs_gamma >= thresholds.gamma_high:
            level = max(level, 2)
            drivers.append(
                f"net gamma {gamma:+.2f} at/above {thresholds.gamma_high:.2f} — "
                "the gap is amplified non-linearly (§66)"
            )
        elif abs_gamma >= thresholds.gamma_moderate:
            level = max(level, 1)
            drivers.append(
                f"net gamma {gamma:+.2f} at/above {thresholds.gamma_moderate:.2f} — "
                "convexity amplifies the gap (§66)"
            )
    return (SENSITIVITY_LOW, SENSITIVITY_MODERATE, SENSITIVITY_HIGH)[level], drivers


def classify_event_risk(
    inputs: EventRiskInputs, *, thresholds: EventRiskThresholds = EventRiskThresholds()
) -> dict:
    """The §63 classifier — a deterministic table, never a model (§63).

    Returns :meth:`EventRiskSnapshot.to_dict`. Same inputs ⇒ same output,
    always; there is no randomness, no clock read and no LLM anywhere in
    this call path, which is precisely what §63's "Do not let LLM alone
    assign this state" requires and what the test suite asserts by running
    the function twice and comparing payloads.

    The expected move (§63 ``current_implied_move`` vs §64 history):

    - ``implied_move_pct`` when present → basis :data:`BASIS_IMPLIED`. The
      option market's forward-looking price for THIS print beats a median
      of past ones, because it already contains whatever the past ones
      taught it plus everything since;
    - otherwise the historical ``median_abs`` → basis
      :data:`BASIS_HISTORICAL_MEDIAN`, and the sample size rides along in
      both ``historical["n"]`` and a caveat;
    - neither available → basis :data:`BASIS_NONE`, state
      :data:`STATE_UNKNOWN`, ``reason`` naming what was missing. **The
      classifier never guesses.** UNKNOWN is not LOW; it is "we did not
      measure this", and it is rendered differently on purpose.

    Then the :class:`EventRiskThresholds` table runs: base state from the
    expected move, +1 level when the event is within ``imminent_days``,
    +1/+2 levels from ``exposure_share``, saturating at EXTREME.
    ``exposure_share`` is ``100 · exposure / NAV`` and is ``None`` (with a
    caveat, and no bump) whenever either side is missing or NAV is 0 — a
    share divided by nothing is not 0%, and treating it as 0% would silently
    make every unfunded snapshot look small.

    Worked example, the §65 panel case. ``implied_move_pct=8.8``,
    ``time_to_event_days=1.3``, exposure 12 000 on 200 000 NAV: base HIGH
    (8.8 ≥ 8), imminent (1.3 ≤ 3) → +1 → EXTREME, exposure share 6.0%
    (< 10) → no further bump. State EXTREME with three drivers.
    """
    hist = historical_event_risk(inputs.historical_moves)
    n = int(hist["n"])
    implied_pct = _finite(inputs.implied_move_pct)
    implied = {"pct": implied_pct, "basis": inputs.implied_basis}

    drivers: list[str] = []
    caveats: list[str] = []

    # --- exposure share (None-safe; never fabricate a 0%) ------------------
    exposure = _finite(inputs.position_exposure_usd)
    nav = _finite(inputs.portfolio_nav_usd)
    exposure_share: float | None
    if exposure is None or nav is None or nav <= 0.0:
        exposure_share = None
        if exposure is None and nav is None:
            caveats.append(
                "position exposure and NAV unknown — exposure share not computed "
                "(not assumed small)"
            )
        elif exposure is None:
            caveats.append(
                "position exposure unknown — exposure share not computed "
                "(not assumed small)"
            )
        else:
            caveats.append(
                "portfolio NAV unknown or zero — exposure share not computed "
                "(not assumed small)"
            )
    else:
        exposure_share = 100.0 * abs(exposure) / nav

    # --- §66 sensitivity: its own axis, never a state bump -----------------
    gamma = _finite(inputs.option_gamma)
    vega = _finite(inputs.option_vega)
    theta = _finite(inputs.option_theta)
    has_greeks = any(g is not None for g in (gamma, vega, theta))
    option_greeks = (
        {"gamma": gamma, "vega": vega, "theta": theta} if has_greeks else None
    )
    sensitivity, greek_drivers = _sensitivity_from_greeks(gamma, vega, thresholds)
    drivers.extend(greek_drivers)

    if inputs.is_estimated:
        caveats.append(
            "event date is ESTIMATED, not confirmed — the timing of this risk is "
            "itself uncertain"
        )

    # --- the expected move and its basis (§63/§64) -------------------------
    median_abs = hist["median_abs"]
    if implied_pct is not None:
        expected_move = abs(implied_pct)
        basis = BASIS_IMPLIED
        drivers.append(
            f"implied move {expected_move:.2f}% priced by the option market"
            + (f" ({inputs.implied_basis})" if inputs.implied_basis else "")
        )
    elif median_abs is not None:
        expected_move = float(median_abs)
        basis = BASIS_HISTORICAL_MEDIAN
        drivers.append(
            f"no implied move available — using historical median absolute move "
            f"{expected_move:.2f}% based on {n} event(s)"
        )
    else:
        expected_move = None  # type: ignore[assignment]
        basis = BASIS_NONE

    # §64: the sample size travels with any historical number, ALWAYS.
    if n == 0:
        caveats.append("no historical event moves available (n=0)")
    else:
        caveats.append(f"historical statistics based on {n} event(s)")
        if n < 8:
            caveats.append(
                f"small sample: {n} event(s) cannot establish a distribution — "
                "treat these percentiles as anecdotes, not statistics"
            )

    if expected_move is None:
        # Honest UNKNOWN (never a guessed LOW).
        snapshot = EventRiskSnapshot(
            event_type=inputs.event_type,
            time_to_event_days=_finite(inputs.time_to_event_days),
            historical=hist,
            implied=implied,
            expected_move_pct=None,
            expected_move_basis=BASIS_NONE,
            position_exposure_usd=exposure,
            exposure_share=exposure_share,
            option_greeks=option_greeks,
            event_risk_state=STATE_UNKNOWN,
            sensitivity=sensitivity,
            drivers=drivers,
            caveats=caveats,
            reason=(
                "no implied move and no historical event moves (n=0) — event risk "
                "NOT measured; UNKNOWN is not LOW"
            ),
        )
        return snapshot.to_dict()

    # --- base state from the expected move ---------------------------------
    if expected_move >= thresholds.extreme_move_pct:
        level = 3
        drivers.append(
            f"expected move {expected_move:.2f}% at/above the EXTREME threshold "
            f"{thresholds.extreme_move_pct:.1f}%"
        )
    elif expected_move >= thresholds.high_move_pct:
        level = 2
        drivers.append(
            f"expected move {expected_move:.2f}% at/above the HIGH threshold "
            f"{thresholds.high_move_pct:.1f}%"
        )
    elif expected_move >= thresholds.moderate_move_pct:
        level = 1
        drivers.append(
            f"expected move {expected_move:.2f}% at/above the MODERATE threshold "
            f"{thresholds.moderate_move_pct:.1f}%"
        )
    else:
        level = 0
        drivers.append(
            f"expected move {expected_move:.2f}% below the MODERATE threshold "
            f"{thresholds.moderate_move_pct:.1f}%"
        )

    # --- imminence bump -----------------------------------------------------
    ttl = _finite(inputs.time_to_event_days)
    if ttl is None:
        caveats.append("time to event unknown — imminence not applied")
    elif ttl < 0.0:
        drivers.append(
            f"event was {abs(ttl):.1f} day(s) ago — no imminence bump (this module "
            "measures PRE-EVENT risk, §67)"
        )
    elif ttl <= thresholds.imminent_days:
        level += 1
        drivers.append(
            f"event is {ttl:.1f} day(s) away, within the {thresholds.imminent_days:.0f}-day "
            "imminence window — one level up"
        )

    # --- exposure bump ------------------------------------------------------
    if exposure_share is not None:
        if exposure_share >= thresholds.exposure_two_level_pct:
            level += 2
            drivers.append(
                f"position is {exposure_share:.1f}% of NAV, at/above "
                f"{thresholds.exposure_two_level_pct:.0f}% — two levels up"
            )
        elif exposure_share >= thresholds.exposure_one_level_pct:
            level += 1
            drivers.append(
                f"position is {exposure_share:.1f}% of NAV, at/above "
                f"{thresholds.exposure_one_level_pct:.0f}% — one level up"
            )

    state = STATE_LADDER[min(level, len(STATE_LADDER) - 1)]

    snapshot = EventRiskSnapshot(
        event_type=inputs.event_type,
        time_to_event_days=ttl,
        historical=hist,
        implied=implied,
        expected_move_pct=expected_move,
        expected_move_basis=basis,
        position_exposure_usd=exposure,
        exposure_share=exposure_share,
        option_greeks=option_greeks,
        event_risk_state=state,
        sensitivity=sensitivity,
        drivers=drivers,
        caveats=caveats,
        reason=None,
    )
    return snapshot.to_dict()


# ---------------------------------------------------------------------------
# §65 — the SHADOW cap
# ---------------------------------------------------------------------------


def event_risk_caps(
    snapshot: Mapping[str, object],
    *,
    requested_qty: int,
    price: float | None,
    nav: float | None,
    policy: EventRiskPolicy = EventRiskPolicy(),
) -> list[QuantityCap]:
    """The hypothetical event cap for a proposed trade — SHADOW ONLY (§65).

    Mirrors ``pretrade.statistical_caps`` and ``models.stress.stress_caps``:
    same :class:`~libs.trading_core.risk.pretrade.QuantityCap` shape, same
    "a limit that is satisfied produces NO cap" rule, same fail-open on a
    missing view. A cap emitted here changes nothing by itself — it becomes
    binding only if a caller passes it to ``assess(extra_caps=...)``, and
    per §65 the gateway must not, because no backtest has validated these
    thresholds yet.

    The limit is a straight exposure ceiling — "however large the gap turns
    out to be, do not have more than X% of NAV crossing it":

        ``cap_qty = floor(policy_pct/100 · nav / price)``

    which is exact and needs no bisection: exposure is linear in quantity,
    so the largest passing quantity is a division, not a search. That is the
    honest reason this module does not reuse ``pretrade._largest_passing`` —
    there is nothing non-monotone to guard against.

    Emits a cap only for :data:`STATE_HIGH` and :data:`STATE_EXTREME`.
    Below HIGH the policy is WARN-only (§65's "WARN" rung), and
    :data:`STATE_UNKNOWN` emits NOTHING: an unmeasured event must not
    produce a number that looks measured, and in SHADOW fail-open is the
    deliberate choice (same open item Phase C and Phase D recorded — the
    PRODUCTION promotion decides the fail-closed rule).

    Returns ``[]`` — never raises — when ``requested_qty <= 0``, ``price``
    or ``nav`` is missing/non-positive, or the state is below HIGH. The
    cap's ``sentence`` carries the real numbers AND the sample size behind
    the expected move (§64), so a promoted cap could never resize a trade
    with an unexplained justification.

    Hand-check: EXTREME, nav 200 000, price 50, requested 100. Budget
    ``5% · 200 000 = 10 000``; ``cap_qty = floor(10 000/50) = 200``; 200 ≥
    100 so the limit is SATISFIED and **no cap is emitted**. Drop nav to
    50 000: budget 2 500, ``cap_qty = 50 < 100`` → one cap at 50.
    """
    if isinstance(requested_qty, bool) or not isinstance(requested_qty, int):
        raise ValueError(f"requested_qty must be an int, got {requested_qty!r}")
    if requested_qty <= 0:
        return []

    state = snapshot.get("event_risk_state") if snapshot else None
    if state == STATE_EXTREME:
        pct = policy.extreme_max_exposure_pct
    elif state == STATE_HIGH:
        pct = policy.high_max_exposure_pct
    else:
        # LOW / MODERATE → WARN-only; UNKNOWN → nothing (never guess).
        return []

    price_f = _finite(price)
    nav_f = _finite(nav)
    if price_f is None or price_f <= 0.0 or nav_f is None or nav_f <= 0.0:
        # A view that could not be computed must never produce a cap.
        return []

    budget_usd = pct / 100.0 * nav_f
    cap_qty = int(math.floor(budget_usd / price_f))
    cap_qty = max(0, min(cap_qty, requested_qty))
    if cap_qty >= requested_qty:
        return []  # the limit is satisfied — no cap

    expected = _finite(snapshot.get("expected_move_pct"))
    basis = snapshot.get("expected_move_basis") or BASIS_NONE
    historical = snapshot.get("historical") or {}
    n = historical.get("n") if isinstance(historical, Mapping) else None
    ttl = _finite(snapshot.get("time_to_event_days"))
    event_type = snapshot.get("event_type") or "event"

    when = f"in {ttl:.1f} day(s)" if ttl is not None and ttl >= 0.0 else "upcoming"
    move_txt = f"{expected:.2f}%" if expected is not None else "unknown"
    basis_txt = (
        f"implied move {move_txt}"
        if basis == BASIS_IMPLIED
        else f"historical median move {move_txt} based on {n} event(s)"
    )
    sentence = (
        f"{state} event risk: {event_type} {when} with {basis_txt}. SHADOW policy "
        f"caps exposure across the event at {pct:.0f}% of NAV "
        f"(${budget_usd:,.0f}), i.e. {cap_qty} unit(s) at ${price_f:,.2f} vs "
        f"{requested_qty} requested. RESEARCH DEFAULT, UNVALIDATED — this cap is "
        f"not enforced ({policy.mode})."
    )
    return [
        QuantityCap(
            code=CODE_EVENT_EXPOSURE,
            layer=LAYER_CONCENTRATION,
            cap_qty=cap_qty,
            sentence=sentence,
            measured={
                "requested_qty": float(requested_qty),
                "cap_qty": float(cap_qty),
                "price": price_f,
                "nav": nav_f,
                "max_exposure_pct_nav": float(pct),
                "budget_usd": budget_usd,
                "requested_exposure_usd": float(requested_qty) * price_f,
                "expected_move_pct": expected,
                "historical_n": float(n) if isinstance(n, (int, float)) else None,
                "time_to_event_days": ttl,
            },
        )
    ]
