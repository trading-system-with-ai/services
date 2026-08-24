"""Exit Engine v0 (development plan §11).

Pure, deterministic exit decisions for open paper positions — no DB, no
FastAPI. The engine REUSES the live directional signal code
(:func:`libs.trading_core.signals.score_direction`) verbatim (plan §21 —
MANDATORY; signals are never reimplemented here), and mirrors the backtest
engine's exit priority (:mod:`libs.trading_core.backtest.engine`) so paper
positions are managed by the same rules the backtest validated.

Stock rule set, in priority order (first match wins; plan §11):

1. HARD_STOP    (§11.3, adapted to stock) — close at or below the fixed
                per-share dollar stop set at position open.
2. SIGNAL_FLIP  (§11.1) — the directional engine now reads BEAR.
3. SIGNAL_DECAY (§11.1) — the directional edge fell below the exit
                threshold, which is deliberately LOWER than the entry
                threshold: exits are easier than entries.
4. ATR_TRAIL    (§11.5) — close below the ATR trailing stop off the highest
                close since entry.
5. TIME_STOP    (§11.6) — the position has gone nowhere for too long.

Option rule set (:func:`evaluate_option_exit`, plan §11.3 / §11.7):

1. PREMIUM_HARD_STOP (§11.3) — the option mid fell to or below the
                premium stop; this REPLACES the underlying HARD_STOP,
                which does not apply to option positions.
2. DTE_EXIT     (§11.7) — too few days to expiry: gamma/theta territory.
3-6.            the SAME underlying-driven rules as stock (SIGNAL_FLIP,
                SIGNAL_DECAY, ATR_TRAIL, TIME_STOP), produced by the SAME
                internal evaluations — shared code, never duplicated
                (plan §21).

EVERY rule is evaluated on every call and reported with the real numbers it
used — including the rules that did NOT fire, prefixed "OK:" — so the user
can always see why a position is still held (plan §37, §38).

A data gap must never disable a hard stop silently: while the signal engine
or the ATR still lack history, those rules report "insufficient data" and
stay untriggered, but the stock HARD_STOP keeps protecting the position off
raw prices (plan §11.3). For options, a missing current mid makes
PREMIUM_HARD_STOP report "insufficient data" — loudly, never silently —
while DTE_EXIT and the underlying rules keep protecting the position. Risk
protection outranks strategy confidence, always.
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.trading_core.features import atr
from libs.trading_core.models import DirectionalBias
from libs.trading_core.signals import DirectionalParams, score_direction


@dataclass(frozen=True)
class ExitParams:
    """Exit-engine parameters (plan §11; §6.2: every threshold is a
    documented backtest parameter, never a hardcoded truth).

    - ``exit_edge_threshold``: directional edge below which SIGNAL_DECAY
      fires (plan §11.1). Deliberately LOWER than the live entry threshold
      so exits are easier than entries; the ``exit <= entry`` pairing is
      enforced where both thresholds are known together
      (:class:`libs.trading_core.backtest.BacktestParams`).
    - ``atr_trail_k``: ATR multiple of the trailing stop (plan §11.5).
    - ``time_stop_bars``: bars after which a going-nowhere position is
      abandoned (plan §11.6).
    - ``min_move_atr``: minimum favourable move, in ATR multiples, required
      to escape the time stop (plan §11.6).
    - ``atr_period``: ATR period used by ATR_TRAIL and TIME_STOP (plan
      §11.5 / §11.6 pin these exits to atr14 by default).
    - ``premium_hard_stop_pct``: option positions only — fraction of the
      entry premium whose loss fires PREMIUM_HARD_STOP (plan §11.3; the
      0.45 default is a RESEARCH PARAMETER for the 40-50% band §11.3
      studies, to be tuned by backtest).
    - ``dte_exit_threshold``: option positions only — DTE at or below which
      DTE_EXIT fires (plan §11.7: close or roll around 21 DTE rather than
      hold into the gamma/theta zone).
    """

    exit_edge_threshold: float = 10.0
    atr_trail_k: float = 3.0
    time_stop_bars: int = 20
    min_move_atr: float = 1.0
    atr_period: int = 14
    premium_hard_stop_pct: float = 0.45
    dte_exit_threshold: int = 21
    # SHORT-PREMIUM management (Phase 2 — covered calls / cash-secured
    # puts; the most widely cited mechanical standards):
    # - close at 50% of max profit (buy back once mid <= 50% of the credit);
    # - stop the loss once mid >= 2x the credit received;
    # - never hold short premium into the §11.7 gamma zone (same 21 DTE).
    short_profit_capture_pct: float = 0.50
    short_loss_multiple: float = 2.0

    def __post_init__(self) -> None:
        if self.atr_trail_k <= 0.0:
            raise ValueError(f"atr_trail_k must be > 0, got {self.atr_trail_k!r}")
        if not isinstance(self.time_stop_bars, int) or self.time_stop_bars < 1:
            raise ValueError(
                f"time_stop_bars must be an integer >= 1, got {self.time_stop_bars!r}"
            )
        if self.min_move_atr < 0.0:
            raise ValueError(f"min_move_atr must be >= 0, got {self.min_move_atr!r}")
        if not isinstance(self.atr_period, int) or self.atr_period < 1:
            raise ValueError(
                f"atr_period must be an integer >= 1, got {self.atr_period!r}"
            )
        if not (0.0 < self.premium_hard_stop_pct <= 1.0):
            raise ValueError(
                "premium_hard_stop_pct must be in (0, 1], got "
                f"{self.premium_hard_stop_pct!r}"
            )
        if not isinstance(self.dte_exit_threshold, int) or self.dte_exit_threshold < 0:
            raise ValueError(
                "dte_exit_threshold must be an integer >= 0, got "
                f"{self.dte_exit_threshold!r}"
            )
        if not (0.0 < self.short_profit_capture_pct < 1.0):
            raise ValueError(
                "short_profit_capture_pct must be in (0, 1), got "
                f"{self.short_profit_capture_pct!r}"
            )
        if self.short_loss_multiple <= 1.0:
            raise ValueError(
                "short_loss_multiple must be > 1, got "
                f"{self.short_loss_multiple!r}"
            )


@dataclass
class PositionState:
    """Caller-maintained state of one open position's UNDERLYING (plan §11).

    - ``entry_price``: fill price of the underlying at position open (for
      option positions: the underlying spot at entry, used by the
      underlying-driven rules).
    - ``stop_distance``: per-share dollar risk fixed at position open (the
      §10 gate chain sizes it as an ATR multiple at entry); the hard stop
      sits at ``entry_price - stop_distance`` and NEVER widens (plan §11.3).
      IGNORED for option positions — PREMIUM_HARD_STOP replaces the
      underlying hard stop there (:func:`evaluate_option_exit`).
    - ``entry_edge``: directional edge at entry, kept for decay context in
      the explanations (plan §38).
    - ``bars_held``: bars since entry; the entry bar itself is 0.
    - ``highest_close_since_entry``: running peak close. To produce trails
      identical to the backtest engine (which folds the current close into
      its peak BEFORE evaluating exits), callers must update this with the
      current close before calling :func:`evaluate_exit`.
    """

    entry_price: float
    stop_distance: float
    entry_edge: float
    bars_held: int
    highest_close_since_entry: float
    #: Which way the position PROFITS ("BULL" for long stock/calls/call
    #: spreads, "BEAR" for long puts/put spreads). Every underlying signal
    #: rule mirrors on it: SIGNAL_FLIP fires on the OPPOSING bias,
    #: SIGNAL_DECAY on the position-favorable edge falling, ATR_TRAIL
    #: trails ABOVE the trough for BEAR, TIME_STOP measures the favorable
    #: move. Default BULL = pre-2026-08-17 behavior, byte-identical.
    direction: str = "BULL"
    #: Running LOWEST close since entry — the BEAR trail anchor (mirror of
    #: ``highest_close_since_entry``); REQUIRED when direction == "BEAR".
    lowest_close_since_entry: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("BULL", "BEAR"):
            raise ValueError(
                f"direction must be 'BULL' or 'BEAR', got {self.direction!r}"
            )
        if self.direction == "BEAR" and self.lowest_close_since_entry is None:
            raise ValueError(
                "BEAR positions require lowest_close_since_entry (the trail "
                "anchor mirrors to the trough)"
            )


@dataclass
class OptionState:
    """Caller-maintained state of one open LONG option position (plan §11.3,
    §11.7). Complements :class:`PositionState` (which keeps tracking the
    UNDERLYING) for :func:`evaluate_option_exit`.

    - ``entry_premium``: premium paid at entry, PER SHARE (matching quote
      convention, not per 100-share contract).
    - ``current_mid``: current option mid, per share; ``None`` when the
      quote is unavailable (honest null — PREMIUM_HARD_STOP then reports
      "insufficient data" loudly rather than pretending).
    - ``dte``: calendar days to expiry; ``None`` when unknown.
    """

    entry_premium: float
    current_mid: float | None
    dte: int | None


@dataclass
class ExitDecision:
    """One fully explainable exit evaluation (plan §37, §38).

    - ``should_exit`` / ``triggered_rule``: the FIRST matching rule in
      priority order — stock: "HARD_STOP" | "SIGNAL_FLIP" | "SIGNAL_DECAY" |
      "ATR_TRAIL" | "TIME_STOP"; options: "PREMIUM_HARD_STOP" | "DTE_EXIT"
      followed by the same signal/trail/time rules — or ``False`` / ``None``
      to keep holding.
    - ``reasons``: one human-readable line per rule evaluated — every rule,
      every call — with the real numbers used. Rules that did not fire are
      prefixed ``"OK:"`` (plan §37: the user must see why the position is
      still held); rules whose inputs are still warming up say
      "insufficient data" and never trigger.
    - ``current_edge``: live directional edge; ``None`` while the signal
      engine has no evaluable component (short history).
    - ``stop_price``: the protective stop level — stock: ``entry_price -
      stop_distance``; options: the premium stop ``entry_premium * (1 -
      premium_hard_stop_pct)`` per share. Always reported.
    - ``trail_price``: the ATR trailing stop on the UNDERLYING, ``None``
      until ATR is computable.
    - ``time_stop_remaining``: ``max(0, time_stop_bars - bars_held)``,
      always reported.
    """

    should_exit: bool
    triggered_rule: str | None
    reasons: list[str]
    current_edge: float | None
    stop_price: float
    trail_price: float | None
    time_stop_remaining: int | None


# ---------------------------------------------------------------------------
# Shared internals (plan §21: one implementation, reused — never duplicated)
# ---------------------------------------------------------------------------


@dataclass
class _UnderlyingRules:
    """Internal: the underlying-driven rule evaluations shared VERBATIM by
    :func:`evaluate_exit` and :func:`evaluate_option_exit` (plan §21)."""

    current_edge: float | None
    trail_price: float | None
    time_stop_remaining: int
    evaluations: list[tuple[str, bool, str]]  # (rule, triggered, reason)


def _validate_series(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None,
) -> int:
    """Common series validation for both engines; returns ``len(closes)``."""
    n = len(closes)
    if n == 0:
        raise ValueError("closes must not be empty")
    if not (len(highs) == len(lows) == n):
        raise ValueError(
            "closes, highs and lows must have equal length, got "
            f"{n}/{len(highs)}/{len(lows)}"
        )
    if volumes is not None and len(volumes) != n:
        raise ValueError(
            f"volumes must align with closes, got {len(volumes)}/{n}"
        )
    return n


def _underlying_evaluations(
    state: PositionState,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None,
    params: ExitParams,
    directional_params: DirectionalParams | None,
) -> _UnderlyingRules:
    """Evaluate the underlying-driven rule families — SIGNAL_FLIP,
    SIGNAL_DECAY, ATR_TRAIL, TIME_STOP — off the UNDERLYING series.

    This is the single shared implementation behind both
    :func:`evaluate_exit` and :func:`evaluate_option_exit` (plan §21):
    given the same series, state and parameters, both public entry points
    produce bit-identical evaluations for these rules. Assumes the series
    were validated by :func:`_validate_series`.
    """
    if state.bars_held < 0:
        raise ValueError(f"bars_held must be >= 0, got {state.bars_held!r}")

    n = len(closes)
    close = closes[-1]

    # --- Live signal (plan §21: the exact live code, never reimplemented) --
    direction = score_direction(
        closes,
        highs,
        lows,
        volumes=volumes,
        params=directional_params if directional_params is not None else DirectionalParams(),
    )
    # The signal is "ready" once at least one component was actually
    # evaluated; with every component still in warmup the 0/0 scores carry
    # no information and must not fire signal exits (plan §11.1).
    signal_ready = any(
        not c.detail.startswith("insufficient data") for c in direction.components
    )
    edge = direction.directional_edge
    current_edge = edge if signal_ready else None

    # Direction mirror (2026-08-17): every rule below is expressed in the
    # POSITION's frame. favorable_edge is +edge for BULL, -edge for BEAR;
    # the opposing bias flips the position; the trail hangs below the peak
    # (BULL) or above the trough (BEAR — the industry-standard short-side
    # trailing stop); the time stop measures the FAVORABLE move.
    bear = state.direction == "BEAR"
    favorable_edge = -edge if bear else edge
    opposing_bias = DirectionalBias.BULL if bear else DirectionalBias.BEAR

    # --- ATR for the trail and the time stop (plan §11.5 / §11.6) ----------
    atr_last = atr(highs, lows, closes, period=params.atr_period)[-1]
    if atr_last is None:
        trail_price = None
    elif bear:
        trail_price = (
            (state.lowest_close_since_entry or 0.0)
            + params.atr_trail_k * atr_last
        )
    else:
        trail_price = (
            state.highest_close_since_entry - params.atr_trail_k * atr_last
        )
    time_stop_remaining = max(0, params.time_stop_bars - state.bars_held)

    evaluations: list[tuple[str, bool, str]] = []

    # SIGNAL_FLIP (§11.1).
    if not signal_ready:
        evaluations.append(
            (
                "SIGNAL_FLIP",
                False,
                "SIGNAL_FLIP: insufficient data: no directional component "
                f"evaluable on {n} bars; not triggered",
            )
        )
    elif direction.bias is opposing_bias:
        evaluations.append(
            (
                "SIGNAL_FLIP",
                True,
                f"SIGNAL_FLIP: bias {direction.bias.value} opposes this "
                f"{state.direction} position, edge {edge:.1f} (bull "
                f"{direction.bull_score:.1f} vs bear {direction.bear_score:.1f})",
            )
        )
    else:
        evaluations.append(
            (
                "SIGNAL_FLIP",
                False,
                f"OK: SIGNAL_FLIP: bias {direction.bias.value}, edge {edge:.1f}",
            )
        )

    # SIGNAL_DECAY (§11.1) — the exit threshold is deliberately lower than
    # the entry threshold: exits are easier than entries.
    if not signal_ready:
        evaluations.append(
            (
                "SIGNAL_DECAY",
                False,
                "SIGNAL_DECAY: insufficient data: no directional component "
                f"evaluable on {n} bars; not triggered",
            )
        )
    elif favorable_edge < params.exit_edge_threshold:
        evaluations.append(
            (
                "SIGNAL_DECAY",
                True,
                f"SIGNAL_DECAY: {state.direction}-favorable edge "
                f"{favorable_edge:.1f} < exit threshold "
                f"{params.exit_edge_threshold:.1f} (entry edge "
                f"{state.entry_edge:.1f}; exit easier than entry, plan §11.1)",
            )
        )
    else:
        evaluations.append(
            (
                "SIGNAL_DECAY",
                False,
                f"OK: SIGNAL_DECAY: {state.direction}-favorable edge "
                f"{favorable_edge:.1f} >= exit threshold "
                f"{params.exit_edge_threshold:.1f}",
            )
        )

    # ATR_TRAIL (§11.5).
    if trail_price is None:
        evaluations.append(
            (
                "ATR_TRAIL",
                False,
                f"ATR_TRAIL: insufficient data: atr{params.atr_period} needs "
                f"{params.atr_period + 1} bars, have {n}; not triggered",
            )
        )
    else:
        if bear:
            trail_detail = (
                f"(trough {state.lowest_close_since_entry:.4f} + "
                f"{params.atr_trail_k:.2f} * atr{params.atr_period} {atr_last:.4f})"
            )
            triggered = close > trail_price
            relation = ">" if triggered else "<="
        else:
            trail_detail = (
                f"(peak {state.highest_close_since_entry:.4f} - "
                f"{params.atr_trail_k:.2f} * atr{params.atr_period} {atr_last:.4f})"
            )
            triggered = close < trail_price
            relation = "<" if triggered else ">="
        if triggered:
            evaluations.append(
                (
                    "ATR_TRAIL",
                    True,
                    f"ATR_TRAIL: close {close:.4f} {relation} trail "
                    f"{trail_price:.4f} {trail_detail}",
                )
            )
        else:
            evaluations.append(
                (
                    "ATR_TRAIL",
                    False,
                    f"OK: ATR_TRAIL: close {close:.4f} {relation} trail "
                    f"{trail_price:.4f} {trail_detail}",
                )
            )

    # TIME_STOP (§11.6) — needs the ATR for its minimum-move bar.
    if atr_last is None:
        evaluations.append(
            (
                "TIME_STOP",
                False,
                f"TIME_STOP: insufficient data: atr{params.atr_period} needs "
                f"{params.atr_period + 1} bars, have {n} (held "
                f"{state.bars_held}/{params.time_stop_bars} bars, "
                f"{time_stop_remaining} remaining); not triggered",
            )
        )
    else:
        # Favorable move in the POSITION's frame: up for BULL, down for BEAR.
        move = (state.entry_price - close) if bear else (close - state.entry_price)
        min_move = params.min_move_atr * atr_last
        if state.bars_held >= params.time_stop_bars and move < min_move:
            evaluations.append(
                (
                    "TIME_STOP",
                    True,
                    f"TIME_STOP: held {state.bars_held} bars >= "
                    f"{params.time_stop_bars}, move {move:.4f} < "
                    f"{params.min_move_atr:.2f} * atr{params.atr_period} "
                    f"{atr_last:.4f} = {min_move:.4f}",
                )
            )
        elif state.bars_held >= params.time_stop_bars:
            evaluations.append(
                (
                    "TIME_STOP",
                    False,
                    f"OK: TIME_STOP: held {state.bars_held} bars >= "
                    f"{params.time_stop_bars} but move {move:.4f} >= required "
                    f"{min_move:.4f} ({params.min_move_atr:.2f} * "
                    f"atr{params.atr_period} {atr_last:.4f})",
                )
            )
        else:
            evaluations.append(
                (
                    "TIME_STOP",
                    False,
                    f"OK: TIME_STOP: held {state.bars_held} bars < "
                    f"{params.time_stop_bars} ({time_stop_remaining} bars "
                    f"remaining), move {move:.4f} vs required {min_move:.4f}",
                )
            )

    return _UnderlyingRules(
        current_edge=current_edge,
        trail_price=trail_price,
        time_stop_remaining=time_stop_remaining,
        evaluations=evaluations,
    )


def _decide(
    evaluations: list[tuple[str, bool, str]],
    shared: _UnderlyingRules,
    stop_price: float,
) -> ExitDecision:
    """Fold ordered rule evaluations into the first-match decision."""
    triggered_rule = next((name for name, hit, _ in evaluations if hit), None)
    return ExitDecision(
        should_exit=triggered_rule is not None,
        triggered_rule=triggered_rule,
        reasons=[reason for _, _, reason in evaluations],
        current_edge=shared.current_edge,
        stop_price=stop_price,
        trail_price=shared.trail_price,
        time_stop_remaining=shared.time_stop_remaining,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def evaluate_exit(
    state: PositionState,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None = None,
    params: ExitParams = ExitParams(),
    directional_params: DirectionalParams | None = None,
) -> ExitDecision:
    """Evaluate EVERY §11 exit rule for one open LONG_STOCK position at the
    close of the current bar (``closes[-1]``), collecting a reason line per
    rule; the FIRST match in priority order triggers (mirrors the backtest
    engine's exit priority, plan §20 / §11):

    1. HARD_STOP    (§11.3) — ``close <= entry_price - stop_distance``.
    2. SIGNAL_FLIP  (§11.1) — live bias == BEAR.
    3. SIGNAL_DECAY (§11.1) — ``edge < exit_edge_threshold`` (exits easier
       than entries).
    4. ATR_TRAIL    (§11.5) — ``close < highest_close_since_entry -
       atr_trail_k * ATR(atr_period)``.
    5. TIME_STOP    (§11.6) — ``bars_held >= time_stop_bars`` and the move
       since entry is ``< min_move_atr * ATR(atr_period)``.

    Rules 2-5 come from the SAME internal evaluations
    :func:`evaluate_option_exit` uses (plan §21 — shared, never duplicated).
    ``directional_params`` passes through UNCHANGED to
    :func:`score_direction` — the exits run the exact live signal code, never
    a reimplementation (plan §21); ``None`` means the engine's defaults.

    Insufficient-data behaviour: while the signal engine has no evaluable
    component or the ATR is still warming up, the affected rules report
    "insufficient data" and DO NOT trigger — but HARD_STOP always works off
    raw prices: a data gap must never disable the hard stop (plan §11.3).

    Raises ``ValueError`` on empty ``closes``, misaligned series,
    ``stop_distance <= 0`` or ``bars_held < 0``.
    """
    _validate_series(closes, highs, lows, volumes)
    if state.stop_distance <= 0.0:
        raise ValueError(f"stop_distance must be > 0, got {state.stop_distance!r}")

    close = closes[-1]
    # Phase 3 (2026-08-17): BEAR = SHORT_STOCK — the hard stop MIRRORS to
    # sit ABOVE the entry (a short is stopped by the price RISING through
    # entry + stop_distance). Every other §11 rule mirrors inside the
    # shared evaluations. Long stock (BULL) is byte-identical to before.
    if state.direction == "BEAR":
        stop_price = state.entry_price + state.stop_distance
    else:
        stop_price = state.entry_price - state.stop_distance
    shared = _underlying_evaluations(
        state, closes, highs, lows, volumes, params, directional_params
    )

    # (1) HARD_STOP (§11.3) — raw prices only, immune to data gaps.
    bear_stock = state.direction == "BEAR"
    stop_detail = (
        f"entry {state.entry_price:.4f} "
        + ("+" if bear_stock else "-")
        + f" stop_distance {state.stop_distance:.4f}"
    )
    breached = close >= stop_price if bear_stock else close <= stop_price
    rel_hit = ">=" if bear_stock else "<="
    rel_ok = "<" if bear_stock else ">"
    if breached:
        hard_stop = (
            "HARD_STOP",
            True,
            f"HARD_STOP: stop {stop_price:.4f} breached: close "
            f"{close:.4f} {rel_hit} {stop_detail}",
        )
    else:
        hard_stop = (
            "HARD_STOP",
            False,
            f"OK: HARD_STOP: close {close:.4f} {rel_ok} stop {stop_price:.4f} "
            f"({stop_detail})",
        )

    return _decide([hard_stop, *shared.evaluations], shared, stop_price)


def evaluate_option_exit(
    state: PositionState,
    option: OptionState,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None = None,
    params: ExitParams = ExitParams(),
    directional_params: DirectionalParams | None = None,
) -> ExitDecision:
    """Evaluate every exit rule for one open LONG option position (LONG_CALL
    / LONG_PUT) at the close of the current UNDERLYING bar; the FIRST match
    in priority order triggers (plan §11.3 / §11.7 / §11):

    1. PREMIUM_HARD_STOP (§11.3) — ``current_mid <= entry_premium *
       (1 - premium_hard_stop_pct)``. This REPLACES the underlying
       HARD_STOP, which does NOT apply to option positions: the option's
       max loss is the premium paid, and the §11.3 stop is defined on that
       premium, not on the underlying's stop_distance
       (``state.stop_distance`` is ignored here). ``current_mid is None``
       reports "insufficient data" — loudly, NEVER silently — and does not
       trigger, while every later rule still protects the position.
    2. DTE_EXIT (§11.7) — ``dte <= dte_exit_threshold``: never hold long
       premium into the gamma/theta zone; ``dte is None`` reports
       "insufficient data" and does not trigger.
    3-6. SIGNAL_FLIP, SIGNAL_DECAY, ATR_TRAIL, TIME_STOP — the SAME
       underlying-driven evaluations as :func:`evaluate_exit`, produced by
       the shared internals (plan §21): given the same underlying series,
       state and parameters they are bit-identical to the stock engine's.

    An option exit can only ever map to SELL_TO_CLOSE — Sell-to-Open does
    not exist anywhere in this system (plan §5).

    ``ExitDecision.stop_price`` reports the PREMIUM stop level (per share),
    the option counterpart of the stock hard stop.

    Raises ``ValueError`` on empty/misaligned series, ``entry_premium <= 0``,
    a negative ``current_mid``, a negative ``dte`` or ``bars_held < 0``.
    """
    _validate_series(closes, highs, lows, volumes)
    if option.entry_premium <= 0.0:
        raise ValueError(
            f"entry_premium must be > 0, got {option.entry_premium!r}"
        )
    if option.current_mid is not None and option.current_mid < 0.0:
        raise ValueError(
            f"current_mid must be >= 0 when provided, got {option.current_mid!r}"
        )
    if option.dte is not None and option.dte < 0:
        raise ValueError(f"dte must be >= 0 when provided, got {option.dte!r}")

    premium_stop = option.entry_premium * (1.0 - params.premium_hard_stop_pct)
    shared = _underlying_evaluations(
        state, closes, highs, lows, volumes, params, directional_params
    )

    evaluations: list[tuple[str, bool, str]] = []

    # (1) PREMIUM_HARD_STOP (§11.3) — replaces the underlying HARD_STOP.
    stop_detail = (
        f"entry premium {option.entry_premium:.4f} * "
        f"(1 - {params.premium_hard_stop_pct:.2f})"
    )
    if option.current_mid is None:
        evaluations.append(
            (
                "PREMIUM_HARD_STOP",
                False,
                "PREMIUM_HARD_STOP: insufficient data: option mid "
                f"unavailable, premium stop {premium_stop:.4f} "
                f"({stop_detail}) could not be checked; not triggered",
            )
        )
    elif option.current_mid <= premium_stop:
        evaluations.append(
            (
                "PREMIUM_HARD_STOP",
                True,
                f"PREMIUM_HARD_STOP: premium stop {premium_stop:.4f} "
                f"breached: mid {option.current_mid:.4f} <= {stop_detail}",
            )
        )
    else:
        evaluations.append(
            (
                "PREMIUM_HARD_STOP",
                False,
                f"OK: PREMIUM_HARD_STOP: mid {option.current_mid:.4f} > "
                f"premium stop {premium_stop:.4f} ({stop_detail})",
            )
        )

    # (2) DTE_EXIT (§11.7) — never hold long premium into the gamma/theta zone.
    if option.dte is None:
        evaluations.append(
            (
                "DTE_EXIT",
                False,
                "DTE_EXIT: insufficient data: days to expiry unavailable "
                f"(threshold {params.dte_exit_threshold} DTE); not triggered",
            )
        )
    elif option.dte <= params.dte_exit_threshold:
        evaluations.append(
            (
                "DTE_EXIT",
                True,
                f"DTE_EXIT: {option.dte} DTE <= threshold "
                f"{params.dte_exit_threshold} (plan §11.7: gamma/theta zone)",
            )
        )
    else:
        evaluations.append(
            (
                "DTE_EXIT",
                False,
                f"OK: DTE_EXIT: {option.dte} DTE > threshold "
                f"{params.dte_exit_threshold}",
            )
        )

    # (3-6) The shared underlying-driven rules (plan §21). NOTE: no
    # underlying HARD_STOP here — PREMIUM_HARD_STOP replaced it (§11.3).
    evaluations.extend(shared.evaluations)

    return _decide(evaluations, shared, premium_stop)


@dataclass(frozen=True)
class ShortPremiumState:
    """One open COLLATERALIZED short-premium position (Phase 2).

    - ``entry_credit``: premium RECEIVED per share at open.
    - ``current_mid``: the short contract's current mid per share; ``None``
      when unquoted (honest null — value rules report "insufficient data").
    - ``dte``: calendar days to expiry; ``None`` when unknown.
    - ``strike`` / ``spot``: for the assignment-risk warning (ITM check);
      ``right`` is "C" (covered call) or "P" (cash-secured put).
    """

    entry_credit: float
    current_mid: float | None
    dte: int | None
    strike: float
    spot: float | None
    right: str

    def __post_init__(self) -> None:
        if self.entry_credit <= 0.0:
            raise ValueError(f"entry_credit must be > 0, got {self.entry_credit!r}")
        if self.right not in ("C", "P"):
            raise ValueError(f"right must be 'C' or 'P', got {self.right!r}")


def evaluate_short_premium_exit(
    state: ShortPremiumState,
    params: ExitParams = ExitParams(),
) -> ExitDecision:
    """Mechanical management for a collateralized SHORT premium leg
    (Phase 2 — covered call / cash-secured put), industry-standard rules
    in priority order:

    1. PREMIUM_LOSS_STOP — ``mid >= short_loss_multiple * credit`` (the
       position moved against the seller; defend before assignment risk
       compounds).
    2. DTE_EXIT — ``dte <= dte_exit_threshold`` (§11.7 gamma-zone rule,
       identical for sellers).
    3. PROFIT_CAPTURE — ``mid <= (1 - short_profit_capture_pct) * credit``
       (the standard "close at 50% of max profit").

    An ITM short (assignment risk) adds a LOUD advisory line but does not
    itself trigger — assignment economics on a COVERED position are
    defined, and the human decides. Buying back is the only action any of
    these can lead to (BUY_TO_CLOSE); nothing here can open exposure.
    """
    evaluations: list[tuple[str, bool, str]] = []
    credit = state.entry_credit

    if state.current_mid is None:
        evaluations.append(
            (
                "PREMIUM_LOSS_STOP",
                False,
                "PREMIUM_LOSS_STOP: insufficient data: the short contract "
                "has no current mid; not triggered",
            )
        )
        evaluations.append(
            (
                "PROFIT_CAPTURE",
                False,
                "PROFIT_CAPTURE: insufficient data: no current mid; "
                "not triggered",
            )
        )
    else:
        loss_level = params.short_loss_multiple * credit
        if state.current_mid >= loss_level:
            evaluations.append(
                (
                    "PREMIUM_LOSS_STOP",
                    True,
                    f"PREMIUM_LOSS_STOP: mid {state.current_mid:.4f} >= "
                    f"{params.short_loss_multiple:.1f}x credit {credit:.4f} "
                    f"= {loss_level:.4f}",
                )
            )
        else:
            evaluations.append(
                (
                    "PREMIUM_LOSS_STOP",
                    False,
                    f"OK: PREMIUM_LOSS_STOP: mid {state.current_mid:.4f} < "
                    f"{loss_level:.4f}",
                )
            )
        capture_level = (1.0 - params.short_profit_capture_pct) * credit
        if state.current_mid <= capture_level:
            evaluations.append(
                (
                    "PROFIT_CAPTURE",
                    True,
                    f"PROFIT_CAPTURE: mid {state.current_mid:.4f} <= "
                    f"{1.0 - params.short_profit_capture_pct:.0%} of credit "
                    f"{credit:.4f} — {params.short_profit_capture_pct:.0%} "
                    "of max profit captured (standard management)",
                )
            )
        else:
            evaluations.append(
                (
                    "PROFIT_CAPTURE",
                    False,
                    f"OK: PROFIT_CAPTURE: mid {state.current_mid:.4f} > "
                    f"{capture_level:.4f}",
                )
            )

    if state.dte is None:
        evaluations.append(
            ("DTE_EXIT", False, "DTE_EXIT: insufficient data: dte unknown; not triggered")
        )
    elif state.dte <= params.dte_exit_threshold:
        evaluations.append(
            (
                "DTE_EXIT",
                True,
                f"DTE_EXIT: {state.dte} DTE <= threshold "
                f"{params.dte_exit_threshold} (plan §11.7: gamma/theta zone "
                "— identical for premium sellers)",
            )
        )
    else:
        evaluations.append(
            ("DTE_EXIT", False, f"OK: DTE_EXIT: {state.dte} DTE > threshold {params.dte_exit_threshold}")
        )

    # Assignment-risk advisory (§37): loud, never a trigger by itself.
    reasons = []
    if state.spot is not None:
        itm = (
            state.spot > state.strike
            if state.right == "C"
            else state.spot < state.strike
        )
        if itm:
            reasons.append(
                f"ADVISORY: short {state.right} {state.strike:g} is ITM "
                f"(spot {state.spot:.4f}) — assignment possible; economics "
                "are DEFINED on a collateralized position, human decides."
            )

    # Priority: loss stop -> DTE -> profit capture.
    order = ("PREMIUM_LOSS_STOP", "DTE_EXIT", "PROFIT_CAPTURE")
    triggered = next(
        (name for name in order for (n, hit, _r) in evaluations if n == name and hit),
        None,
    )
    reasons.extend(r for (_n, _hit, r) in evaluations)
    capture_level = (
        (1.0 - params.short_profit_capture_pct) * credit
    )
    return ExitDecision(
        should_exit=triggered is not None,
        triggered_rule=triggered,
        reasons=reasons,
        current_edge=None,  # mechanical management: no signal engine here
        stop_price=params.short_loss_multiple * credit,
        trail_price=None,
        time_stop_remaining=None,
    )
