"""Previous-event market reaction & pre-event price context (event spec §14,
§17, §19, §20, §31, §32, §64, §85, §96; audit §7, §11.2; Phase E1 unit U1).

Pure stdlib, deterministic, **no I/O** — like the rest of
``libs/trading_core/events/`` this module may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4). It is handed
plain :class:`DailyBar` values by the gateway seam
(``apps/gateway/event_price.py``) and hands back frozen result objects.

Three ideas carry the whole module:

1. **The as-of gate is on bars, not on queries** (§14, §96). A daily bar
   dated *d* only exists for an analysis run at instant *t* once that day's
   regular session has closed — 16:00 ET. :func:`as_of_bar_filter` is THE
   place that rule lives; every other function assumes its input has already
   been through it, so "what did we know at 15:59 ET" is answered by
   filtering the bars, never by trimming the answer afterwards.

2. **The reaction window depends on the session** (§17). An AMC release on D
   is priced on D+1; a BMO release on D is priced on D itself; a
   DURING_MARKET release is priced on D against D-1's close; an UNKNOWN
   session gets the conservative two-day span D-1 → the first bar after D,
   explicitly flagged via ``basis`` so no caller mistakes it for a measured
   window. :func:`first_reaction_index` returns the pair of bar indices plus
   that ``basis`` label, and nothing else in the module guesses.

3. **Absence is a value** (house rule; §85). Every numeric field is
   ``float | None`` and every ``None`` has a companion string in the
   result's ``reasons`` mapping. There is no zero standing in for "we have
   no bars", no NaN, no ``inf``: a zero or non-finite pre-event close makes
   the return ``None`` with ``"pre_event_close_not_positive"``, not a
   division blow-up.

Statistics (§19, §64) always travel with their sample size and are labelled
absolute-move distributions, never probabilities: :class:`HistoryStats`
carries ``n`` / ``n_available`` and the percentiles use the nearest-rank
definition (hand-checkable, no interpolation, no numpy).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from libs.trading_core.events.taxonomy import EASTERN
from libs.trading_core.features.indicators import atr, realized_vol, sma
from libs.trading_core.models.enums import EventSession

__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_LAST_N",
    "EXCURSION_HORIZON",
    "PRE_CONTEXT_DEFAULT_BARS",
    "REGULAR_CLOSE_ET",
    "AbnormalResult",
    "DailyBar",
    "HistoryStats",
    "PriceContext",
    "ReactionResult",
    "abnormal_vs",
    "as_of_bar_filter",
    "event_reaction",
    "first_reaction_index",
    "history_stats",
    "percentile_nearest_rank",
    "pre_event_price_context",
]

#: Regular US equity close in ET. A daily bar for date *d* is only knowable
#: from 16:00 ET on *d* (§14/§96) — 15:59 still sees an unfinished session.
REGULAR_CLOSE_ET = (16, 0)

#: Reaction horizons in trading days after the event (§17).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

#: Excursion window (max favorable / adverse) in trading days.
EXCURSION_HORIZON = 10

#: Sample windows for the §19/§64 history strip ("last 4/8/12 events").
DEFAULT_LAST_N: tuple[int, ...] = (4, 8, 12)

#: Lookback when no previous comparable event anchors the price context
#: (~3 months of trading days).
PRE_CONTEXT_DEFAULT_BARS = 63

#: Basis labels returned by :func:`first_reaction_index` — fixed strings that
#: travel into the API payload and the UI tooltip (§85: the window is always
#: shown, never implied).
BASIS_AFTER_MARKET = "after_market_next_day"
BASIS_BEFORE_MARKET = "before_market_same_day"
BASIS_DURING_MARKET = "during_market_same_day"
BASIS_UNKNOWN = "unknown_session_two_day_span"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyBar:
    """One daily OHLCV bar, dated on its ET session date.

    The pure-library mirror of the ``stock_bars_daily`` ORM row; the gateway
    converts, this layer never touches the DB. ``volume`` is ``float`` so a
    provider that reports fractional/notional volume is not silently
    truncated.
    """

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class ReactionResult:
    """How the underlying moved around one event (§17).

    ``returns[k]`` is the k-trading-day return measured from
    ``pre_event_close`` to the close of the (k-1)-th bar after the reaction
    bar, so ``returns[1]`` is the reaction bar's own close. ``abs_returns``
    mirrors it in absolute value (the §19 distribution is over |move|).
    ``bars_available`` is ``False`` when the window could not be located at
    all — then every numeric field is ``None`` and ``reasons`` says why.
    """

    event_date_et: date | None = None
    session: EventSession | None = None
    basis: str | None = None
    bars_available: bool = False
    pre_event_close: float | None = None
    pre_event_date: date | None = None
    react_open: float | None = None
    react_close: float | None = None
    react_date: date | None = None
    gap_return: float | None = None
    returns: Mapping[int, float | None] = field(default_factory=dict)
    abs_returns: Mapping[int, float | None] = field(default_factory=dict)
    #: ``{k: date}`` — the bar date each measured k-day window ENDS on. It is
    #: what :func:`abnormal_vs` aligns the benchmark to, so the two series are
    #: compared over one calendar span rather than two equal bar counts.
    window_end_dates: Mapping[int, date] = field(default_factory=dict)
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "returns", dict(self.returns))
        object.__setattr__(self, "abs_returns", dict(self.abs_returns))
        object.__setattr__(self, "window_end_dates", dict(self.window_end_dates))
        object.__setattr__(self, "reasons", dict(self.reasons))

    @property
    def horizons(self) -> tuple[int, ...]:
        """Horizons this result was computed for, ascending."""
        return tuple(sorted(self.returns))


@dataclass(frozen=True)
class AbnormalResult:
    """Stock move minus benchmark move on the SAME calendar windows (§17).

    ``abnormal[k] = stock_return[k] - benchmark_return[k]``. The benchmark's
    windows are aligned by DATE, not by index: the benchmark trades on days
    the stock may be halted for, and vice versa, so indices drift.
    """

    abnormal: Mapping[int, float | None] = field(default_factory=dict)
    abnormal_gap: float | None = None
    benchmark_returns: Mapping[int, float | None] = field(default_factory=dict)
    benchmark_gap_return: float | None = None
    benchmark_available: bool = False
    basis: str | None = None
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "abnormal", dict(self.abnormal))
        object.__setattr__(self, "benchmark_returns", dict(self.benchmark_returns))
        object.__setattr__(self, "reasons", dict(self.reasons))


@dataclass(frozen=True)
class HistoryStats:
    """Distribution of |move| over the last ``n`` comparable events (§19, §64).

    ``n`` is the window that was ASKED for, ``n_available`` the number of
    events that actually produced a usable return — the UI prints both
    ("based on 8 events") because a median over 3 of a requested 12 is a
    different claim. ``positive_frequency`` is a FREQUENCY (count/n), never a
    probability: §64 forbids presenting a historical count as a forecast.
    """

    horizon: int
    n: int
    n_available: int
    median_abs: float | None = None
    mean_abs: float | None = None
    p75_abs: float | None = None
    p90_abs: float | None = None
    max_abs: float | None = None
    positive_count: int | None = None
    positive_frequency: float | None = None
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", dict(self.reasons))


@dataclass(frozen=True)
class PriceContext:
    """Where the underlying stands going INTO the event (§31, §32).

    Everything is measured on bars at or before ``as_of_date_et`` (the caller
    has already applied :func:`as_of_bar_filter`), anchored on the previous
    comparable event's date when one exists — "the stock has run up 14% since
    the last print" is the §32 framing, and ``run_up_pct`` carries exactly
    that number under its spec label.
    """

    as_of_date_et: date | None = None
    anchor_date_et: date | None = None
    anchor_close: float | None = None
    anchor_basis: str | None = None
    last_close: float | None = None
    bars_through: date | None = None
    n_bars: int = 0
    since_anchor_return: float | None = None
    run_up_pct: float | None = None
    benchmark_return: float | None = None
    relative_return: float | None = None
    max_drawdown: float | None = None
    realized_vol_20d: float | None = None
    realized_vol_since_anchor: float | None = None
    volume_trend: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    sma20_distance_pct: float | None = None
    sma50_distance_pct: float | None = None
    sma200_distance_pct: float | None = None
    atr14: float | None = None
    atr_pct: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    distance_from_52w_high_pct: float | None = None
    distance_from_52w_low_pct: float | None = None
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", dict(self.reasons))


# ---------------------------------------------------------------------------
# Small numeric helpers — every one of them refuses to fabricate
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """``None`` unless the value is a finite float (no NaN, no ±inf)."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _pct_change(later: float | None, earlier: float | None) -> float | None:
    """``later / earlier - 1``, or ``None`` when the base is unusable.

    A zero or negative base is malformed price data, not a divide-by-zero to
    be papered over with ``inf`` — the caller records the reason instead.
    """
    if later is None or earlier is None:
        return None
    if earlier <= 0.0:
        return None
    return _finite(later / earlier - 1.0)


def percentile_nearest_rank(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile (no interpolation), ``0 < pct <= 100``.

    Rank ``ceil(pct/100 * n)`` into the ascending sample, clamped to
    ``[1, n]``. Chosen over an interpolating definition because the result is
    always an OBSERVED move: "p90 of the last 8 prints" names a move that
    actually happened, which is what §19's honesty requirement wants from a
    sample this small. ``None`` for an empty sample.
    """
    if not 0.0 < pct <= 100.0:
        raise ValueError(f"pct must be in (0, 100], got {pct!r}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    rank = math.ceil(pct / 100.0 * n)
    rank = max(1, min(n, rank))
    return float(ordered[rank - 1])


def _check_sorted(bars: Sequence[DailyBar], *, what: str) -> None:
    """Bars must be strictly increasing by date (same rule as returns.py)."""
    for i in range(1, len(bars)):
        if not bars[i].date > bars[i - 1].date:
            raise ValueError(
                f"{what} must be strictly increasing by date, got "
                f"{bars[i - 1].date} followed by {bars[i].date} at index {i}"
            )


# ---------------------------------------------------------------------------
# §14/§96 — the look-ahead gate
# ---------------------------------------------------------------------------


def as_of_bar_filter(
    bars: Sequence[DailyBar], as_of_utc: datetime
) -> list[DailyBar]:
    """Bars knowable at ``as_of_utc`` (spec §14, §96) — THE look-ahead gate.

    A bar dated *d* is kept iff ``d < as_of_et_date``, or ``d ==
    as_of_et_date`` and the ET wall clock has reached the 16:00 regular close.
    At 15:59 ET the same-day bar does not yet exist as a settled daily bar and
    including it would let a backtest trade on a close it could not have seen.

    ``as_of_utc`` must be timezone-aware (any zone; it is converted to ET) —
    a naive datetime is refused rather than assumed, exactly as
    ``taxonomy.require_utc`` does, because guessing the zone here silently
    shifts the boundary by hours. Order is preserved; the input is not
    mutated.
    """
    if as_of_utc.tzinfo is None or as_of_utc.tzinfo.utcoffset(as_of_utc) is None:
        raise ValueError(
            f"as_of_utc must be timezone-aware; got naive {as_of_utc!r}"
        )
    local = as_of_utc.astimezone(EASTERN)
    as_of_date = local.date()
    close_reached = (local.hour, local.minute) >= REGULAR_CLOSE_ET
    kept: list[DailyBar] = []
    for bar in bars:
        if bar.date < as_of_date:
            kept.append(bar)
        elif bar.date == as_of_date and close_reached:
            kept.append(bar)
    return kept


# ---------------------------------------------------------------------------
# §17 — reaction windows
# ---------------------------------------------------------------------------


def _last_index_on_or_before(bars: Sequence[DailyBar], day: date) -> int | None:
    for i in range(len(bars) - 1, -1, -1):
        if bars[i].date <= day:
            return i
    return None


def _last_index_before(bars: Sequence[DailyBar], day: date) -> int | None:
    for i in range(len(bars) - 1, -1, -1):
        if bars[i].date < day:
            return i
    return None


def _first_index_on_or_after(bars: Sequence[DailyBar], day: date) -> int | None:
    for i, bar in enumerate(bars):
        if bar.date >= day:
            return i
    return None


def _first_index_after(bars: Sequence[DailyBar], day: date) -> int | None:
    for i, bar in enumerate(bars):
        if bar.date > day:
            return i
    return None


def first_reaction_index(
    bars: Sequence[DailyBar],
    event_date_et: date,
    session: EventSession,
) -> tuple[int, int, str] | None:
    """Locate ``(pre_idx, react_idx, basis)`` for an event on ``event_date_et``.

    Bars must be sorted ascending; the bar dates ARE the trading days, so
    holidays and weekends need no calendar — a Friday AMC print reacts on the
    next bar, which is Monday's, and a BMO print on a market holiday reacts on
    the next bar that exists.

    Per session (§17):

    - ``AFTER_MARKET``  pre = the last bar dated ``<= D`` (that day's close,
      which is the last price before the release), react = the next bar.
    - ``BEFORE_MARKET`` pre = the last bar dated ``< D``, react = the bar
      dated D, or the next bar when D has no bar (holiday).
    - ``DURING_MARKET`` pre = the last bar ``< D``, react = the bar D itself
      (or the next existing bar) — the move is inside D's own range.
    - ``UNKNOWN``       pre = the last bar ``< D``, react = the first bar
      ``> D``: a deliberately conservative two-day span that cannot miss the
      move whichever session it really was, flagged by its ``basis`` so the
      UI can say so.

    ``None`` when either leg is missing — the event is too recent (no bar
    after it yet) or predates the bar history. The caller turns that into an
    explicit reason; this function never invents an index.
    """
    if not bars:
        return None
    if session is EventSession.AFTER_MARKET:
        pre_idx = _last_index_on_or_before(bars, event_date_et)
        if pre_idx is None:
            return None
        react_idx = pre_idx + 1
        basis = BASIS_AFTER_MARKET
    elif session is EventSession.BEFORE_MARKET:
        pre_idx = _last_index_before(bars, event_date_et)
        if pre_idx is None:
            return None
        react_candidate = _first_index_on_or_after(bars, event_date_et)
        if react_candidate is None:
            return None
        react_idx = react_candidate
        basis = BASIS_BEFORE_MARKET
    elif session is EventSession.DURING_MARKET:
        pre_idx = _last_index_before(bars, event_date_et)
        if pre_idx is None:
            return None
        react_candidate = _first_index_on_or_after(bars, event_date_et)
        if react_candidate is None:
            return None
        react_idx = react_candidate
        basis = BASIS_DURING_MARKET
    else:
        pre_idx = _last_index_before(bars, event_date_et)
        if pre_idx is None:
            return None
        react_candidate = _first_index_after(bars, event_date_et)
        if react_candidate is None:
            return None
        react_idx = react_candidate
        basis = BASIS_UNKNOWN
    if react_idx >= len(bars) or react_idx <= pre_idx:
        return None
    return pre_idx, react_idx, basis


def _unavailable_reaction(
    event_date_et: date,
    session: EventSession,
    horizons: Sequence[int],
    reason: str,
) -> ReactionResult:
    return ReactionResult(
        event_date_et=event_date_et,
        session=session,
        basis=None,
        bars_available=False,
        returns={k: None for k in horizons},
        abs_returns={k: None for k in horizons},
        reasons={"bars": reason},
    )


def event_reaction(
    bars: Sequence[DailyBar],
    event_date_et: date,
    session: EventSession,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    excursion_horizon: int = EXCURSION_HORIZON,
) -> ReactionResult:
    """Measure the move around one event (§17).

    ``gap_return`` is the overnight/opening gap ``react_open / pre_close - 1``
    — for an AMC print that is the market's whole first verdict. ``returns[k]``
    runs from ``pre_event_close`` to the close ``k`` trading days into the
    reaction (``k = 1`` is the reaction bar itself), so 1D and the gap answer
    two different questions and both are reported.

    Excursions scan the closes of bars ``1..excursion_horizon`` and report the
    best and worst close-basis move; they are ``None`` (with a reason) when no
    bar past the reaction exists — an excursion over one bar is just the 1D
    return dressed up.

    Missing data is always a reason, never a zero: no bars, no reaction bar
    (event too recent), a non-positive pre-event close, or a horizon that runs
    past the end of the data each produce ``None`` plus an entry in
    ``reasons``.
    """
    horizons = tuple(sorted({int(k) for k in horizons}))
    for k in horizons:
        if k < 1:
            raise ValueError(f"horizons must all be >= 1, got {k}")
    _check_sorted(bars, what="bars")
    if not bars:
        return _unavailable_reaction(
            event_date_et, session, horizons, "no_bars_available"
        )

    located = first_reaction_index(bars, event_date_et, session)
    if located is None:
        first_bar, last_bar = bars[0].date, bars[-1].date
        if event_date_et < first_bar:
            reason = f"bars unavailable before {first_bar.isoformat()}"
        else:
            reason = (
                "no bar after the event yet; bars end "
                f"{last_bar.isoformat()}"
            )
        return _unavailable_reaction(event_date_et, session, horizons, reason)

    pre_idx, react_idx, basis = located
    pre_bar = bars[pre_idx]
    react_bar = bars[react_idx]
    pre_close = _finite(pre_bar.close)
    reasons: dict[str, str] = {}

    if pre_close is None or pre_close <= 0.0:
        reasons["pre_event_close"] = "pre_event_close_not_positive"
        return ReactionResult(
            event_date_et=event_date_et,
            session=session,
            basis=basis,
            bars_available=True,
            pre_event_close=None,
            pre_event_date=pre_bar.date,
            react_open=_finite(react_bar.open),
            react_close=_finite(react_bar.close),
            react_date=react_bar.date,
            gap_return=None,
            returns={k: None for k in horizons},
            abs_returns={k: None for k in horizons},
            reasons={
                **reasons,
                **{
                    f"return_{k}D": "pre_event_close_not_positive"
                    for k in horizons
                },
                "gap_return": "pre_event_close_not_positive",
            },
        )

    gap_return = _pct_change(_finite(react_bar.open), pre_close)
    if gap_return is None:
        reasons["gap_return"] = "react_open_unavailable"

    returns: dict[int, float | None] = {}
    abs_returns: dict[int, float | None] = {}
    window_end_dates: dict[int, date] = {}
    for k in horizons:
        idx = react_idx + k - 1
        if idx >= len(bars):
            returns[k] = None
            abs_returns[k] = None
            reasons[f"return_{k}D"] = "insufficient_bars_after_event"
            continue
        window_end_dates[k] = bars[idx].date
        value = _pct_change(_finite(bars[idx].close), pre_close)
        returns[k] = value
        abs_returns[k] = abs(value) if value is not None else None
        if value is None:
            reasons[f"return_{k}D"] = "close_unavailable"

    path: list[float] = []
    last_excursion_idx = min(react_idx + excursion_horizon - 1, len(bars) - 1)
    for idx in range(react_idx, last_excursion_idx + 1):
        value = _pct_change(_finite(bars[idx].close), pre_close)
        if value is not None:
            path.append(value)
    if path:
        mfe: float | None = max(path)
        mae: float | None = min(path)
    else:
        mfe = mae = None
        reasons["excursion"] = "insufficient_bars_after_event"

    return ReactionResult(
        event_date_et=event_date_et,
        session=session,
        basis=basis,
        bars_available=True,
        pre_event_close=pre_close,
        pre_event_date=pre_bar.date,
        react_open=_finite(react_bar.open),
        react_close=_finite(react_bar.close),
        react_date=react_bar.date,
        gap_return=gap_return,
        returns=returns,
        abs_returns=abs_returns,
        window_end_dates=window_end_dates,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        reasons=reasons,
    )


def abnormal_vs(
    stock: ReactionResult,
    bench_bars: Sequence[DailyBar],
    event_date_et: date,
    session: EventSession,
) -> AbnormalResult:
    """Stock reaction minus the benchmark's, on the same CALENDAR windows.

    The benchmark's own pre/react bars are found by DATE — its pre bar is the
    last bench bar on or before the stock's pre-event bar date, its k-day bar
    the last bench bar on or before the stock's k-day bar date. Aligning by
    index instead would silently compare a 5-day stock window against a 6-day
    benchmark window whenever the two histories disagree about a trading day.

    ``abnormal[k]`` is ``None`` whenever either leg is (with a reason);
    ``benchmark_available`` is ``False`` when the benchmark has no usable
    window at all, which is honest degradation rather than an abnormal return
    equal to the raw one.
    """
    horizons = stock.horizons or DEFAULT_HORIZONS
    reasons: dict[str, str] = {}

    if not stock.bars_available or stock.pre_event_date is None:
        return AbnormalResult(
            abnormal={k: None for k in horizons},
            benchmark_returns={k: None for k in horizons},
            benchmark_available=False,
            basis=stock.basis,
            reasons={"benchmark": "stock_reaction_unavailable"},
        )
    if not bench_bars:
        return AbnormalResult(
            abnormal={k: None for k in horizons},
            benchmark_returns={k: None for k in horizons},
            benchmark_available=False,
            basis=stock.basis,
            reasons={"benchmark": "no_benchmark_bars_available"},
        )
    _check_sorted(bench_bars, what="bench_bars")

    bench_pre_idx = _last_index_on_or_before(bench_bars, stock.pre_event_date)
    if bench_pre_idx is None:
        return AbnormalResult(
            abnormal={k: None for k in horizons},
            benchmark_returns={k: None for k in horizons},
            benchmark_available=False,
            basis=stock.basis,
            reasons={
                "benchmark": (
                    "benchmark bars unavailable before "
                    f"{bench_bars[0].date.isoformat()}"
                )
            },
        )
    bench_pre_close = _finite(bench_bars[bench_pre_idx].close)
    if bench_pre_close is None or bench_pre_close <= 0.0:
        return AbnormalResult(
            abnormal={k: None for k in horizons},
            benchmark_returns={k: None for k in horizons},
            benchmark_available=False,
            basis=stock.basis,
            reasons={"benchmark": "benchmark_pre_close_not_positive"},
        )

    # The stock's own window dates: k-D ends on the (react_idx + k - 1)-th
    # stock bar. We re-derive those dates from the stock result's window so
    # the benchmark is measured over the identical calendar span.
    bench_gap: float | None = None
    bench_react_idx = _last_index_on_or_before(bench_bars, stock.react_date) if stock.react_date else None
    if bench_react_idx is not None and stock.react_date is not None:
        if bench_bars[bench_react_idx].date == stock.react_date:
            bench_gap = _pct_change(
                _finite(bench_bars[bench_react_idx].open), bench_pre_close
            )
        else:
            reasons["benchmark_gap"] = (
                "benchmark has no bar on "
                f"{stock.react_date.isoformat()}"
            )
    else:
        reasons["benchmark_gap"] = "benchmark_react_bar_unavailable"

    abnormal_gap: float | None = None
    if bench_gap is not None and stock.gap_return is not None:
        abnormal_gap = _finite(stock.gap_return - bench_gap)
    elif stock.gap_return is None:
        reasons.setdefault("abnormal_gap", "stock_gap_unavailable")
    else:
        reasons.setdefault("abnormal_gap", "benchmark_gap_unavailable")

    bench_returns: dict[int, float | None] = {}
    abnormal: dict[int, float | None] = {}
    for k in horizons:
        window_end = _stock_window_end_date(stock, k)
        if window_end is None:
            bench_returns[k] = None
            abnormal[k] = None
            reasons[f"abnormal_{k}D"] = "stock_window_unavailable"
            continue
        idx = _last_index_on_or_before(bench_bars, window_end)
        if idx is None or idx < bench_pre_idx:
            bench_returns[k] = None
            abnormal[k] = None
            reasons[f"abnormal_{k}D"] = (
                "benchmark has no bar on or before "
                f"{window_end.isoformat()}"
            )
            continue
        bench_value = _pct_change(_finite(bench_bars[idx].close), bench_pre_close)
        bench_returns[k] = bench_value
        stock_value = stock.returns.get(k)
        if bench_value is None or stock_value is None:
            abnormal[k] = None
            reasons[f"abnormal_{k}D"] = (
                "stock_return_unavailable"
                if stock_value is None
                else "benchmark_return_unavailable"
            )
        else:
            abnormal[k] = _finite(stock_value - bench_value)

    return AbnormalResult(
        abnormal=abnormal,
        abnormal_gap=abnormal_gap,
        benchmark_returns=bench_returns,
        benchmark_gap_return=bench_gap,
        benchmark_available=True,
        basis=stock.basis,
        reasons=reasons,
    )


def _stock_window_end_date(stock: ReactionResult, k: int) -> date | None:
    """The calendar date the stock's k-day window ends on, if it was measured.

    Only windows the stock actually resolved carry a date; an unmeasured
    horizon has no date to align the benchmark to, and inventing one (e.g.
    "react date + k calendar days") would measure the benchmark over a window
    the stock never saw.
    """
    return stock.window_end_dates.get(k)


def history_stats(
    reactions: Sequence[ReactionResult],
    *,
    horizon: int = 1,
    last_n: Sequence[int] = DEFAULT_LAST_N,
) -> dict[str, HistoryStats]:
    """|Move| distribution over the last N comparable events (§19, §64).

    ``reactions`` may be in any order — they are sorted by event date and the
    LAST ``n`` are taken, so a caller passing the full history gets the
    trailing window it asked for. Events without a usable return at ``horizon``
    are counted out of ``n_available`` (never imputed), and a window with fewer
    than two usable events yields all-``None`` statistics with the reason
    ``"insufficient_sample"``: a "median" of one print is a single number
    wearing a statistic's clothes.

    Keys are the window labels ``"last4"``/``"last8"``/``"last12"``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    ordered = sorted(
        reactions,
        key=lambda r: (r.event_date_et is None, r.event_date_et or date.min),
    )
    out: dict[str, HistoryStats] = {}
    for n in last_n:
        n = int(n)
        if n < 1:
            raise ValueError(f"last_n values must be >= 1, got {n}")
        window = ordered[-n:] if n <= len(ordered) else list(ordered)
        values = [
            r.returns.get(horizon)
            for r in window
            if r.bars_available and r.returns.get(horizon) is not None
        ]
        usable = [v for v in values if v is not None]
        label = f"last{n}"
        if len(usable) < 2:
            out[label] = HistoryStats(
                horizon=horizon,
                n=n,
                n_available=len(usable),
                reasons={
                    "sample": (
                        "insufficient_sample: "
                        f"{len(usable)} usable event(s), need >= 2"
                    )
                },
            )
            continue
        abs_values = [abs(v) for v in usable]
        ordered_abs = sorted(abs_values)
        positive = sum(1 for v in usable if v > 0.0)
        out[label] = HistoryStats(
            horizon=horizon,
            n=n,
            n_available=len(usable),
            median_abs=percentile_nearest_rank(ordered_abs, 50.0),
            mean_abs=_finite(math.fsum(abs_values) / len(abs_values)),
            p75_abs=percentile_nearest_rank(ordered_abs, 75.0),
            p90_abs=percentile_nearest_rank(ordered_abs, 90.0),
            max_abs=max(abs_values),
            positive_count=positive,
            positive_frequency=_finite(positive / len(usable)),
        )
    return out


# ---------------------------------------------------------------------------
# §31/§32 — pre-event price context
# ---------------------------------------------------------------------------


def _sma_distance(
    closes: Sequence[float], period: int, last_close: float
) -> tuple[float | None, float | None]:
    """``(sma, distance_pct)`` from ``indicators.sma`` — never reimplemented."""
    if len(closes) < period:
        return None, None
    value = sma(closes, period)[-1]
    if value is None or value <= 0.0:
        return None, None
    return float(value), _pct_change(last_close, float(value))


def _max_drawdown(closes: Sequence[float]) -> float | None:
    """Largest peak-to-trough close-basis decline, as a negative fraction."""
    peak: float | None = None
    worst = 0.0
    for close in closes:
        if close <= 0.0:
            continue
        if peak is None or close > peak:
            peak = close
        drop = close / peak - 1.0
        if drop < worst:
            worst = drop
    if peak is None:
        return None
    return _finite(worst)


def pre_event_price_context(
    bars: Sequence[DailyBar],
    *,
    anchor_date_et: date | None = None,
    as_of_date_et: date,
    bench_bars: Sequence[DailyBar] | None = None,
    default_lookback_bars: int = PRE_CONTEXT_DEFAULT_BARS,
) -> PriceContext:
    """Where the stock stands going into the event (§31, §32).

    The anchor is the PREVIOUS comparable event's date — the run-up is
    measured from that event's pre-event close, which is what "up 14% since
    the last print" means. With no anchor (first tracked event) the window
    falls back to the last ``default_lookback_bars`` bars and
    ``anchor_basis`` says so, so the UI never labels a 63-bar window as
    "since the last earnings".

    ``benchmark_return`` / ``relative_return`` use the benchmark over the SAME
    calendar window; ``relative_return`` is the arithmetic difference (§32's
    "vs SPY"), not a ratio. Windowed indicators come from
    ``features.indicators`` (``sma``, ``atr``, ``realized_vol``) and are
    ``None`` with a reason whenever the window is longer than the history —
    a 200-day SMA over 120 bars is not a 200-day SMA.
    """
    _check_sorted(bars, what="bars")
    reasons: dict[str, str] = {}
    usable = [b for b in bars if b.date <= as_of_date_et]
    if not usable:
        return PriceContext(
            as_of_date_et=as_of_date_et,
            anchor_date_et=anchor_date_et,
            n_bars=0,
            reasons={"bars": "no_bars_available"},
        )

    closes = [float(b.close) for b in usable]
    highs = [float(b.high) for b in usable]
    lows = [float(b.low) for b in usable]
    volumes = [float(b.volume) for b in usable]
    last_close = _finite(closes[-1])
    bars_through = usable[-1].date
    n_bars = len(usable)

    # --- anchor -----------------------------------------------------------
    anchor_idx: int | None = None
    anchor_basis: str
    if anchor_date_et is not None:
        anchor_idx = _last_index_on_or_before(usable, anchor_date_et)
        if anchor_idx is None:
            anchor_basis = "anchor_before_first_bar"
            reasons["anchor"] = (
                "bars unavailable before "
                f"{usable[0].date.isoformat()}"
            )
        else:
            anchor_basis = "previous_event"
    else:
        anchor_basis = f"default_{default_lookback_bars}_bars"
        anchor_idx = max(0, n_bars - default_lookback_bars - 1)
        if n_bars < 2:
            anchor_idx = None
            reasons["anchor"] = "insufficient_bars_for_default_window"

    anchor_close: float | None = None
    anchor_date_used: date | None = None
    if anchor_idx is not None:
        anchor_close = _finite(usable[anchor_idx].close)
        anchor_date_used = usable[anchor_idx].date

    since_anchor = _pct_change(last_close, anchor_close)
    if since_anchor is None:
        reasons.setdefault("since_anchor_return", "anchor_close_unavailable")

    # --- benchmark over the same calendar window --------------------------
    bench_return: float | None = None
    relative_return: float | None = None
    if bench_bars:
        _check_sorted(bench_bars, what="bench_bars")
        bench_usable = [b for b in bench_bars if b.date <= as_of_date_et]
        if not bench_usable:
            reasons["benchmark_return"] = "no_benchmark_bars_available"
        elif anchor_date_used is None:
            reasons["benchmark_return"] = "anchor_unavailable"
        else:
            b_anchor = _last_index_on_or_before(bench_usable, anchor_date_used)
            b_last = _last_index_on_or_before(bench_usable, bars_through)
            if b_anchor is None or b_last is None:
                reasons["benchmark_return"] = (
                    "benchmark bars unavailable before "
                    f"{bench_usable[0].date.isoformat()}"
                )
            else:
                bench_return = _pct_change(
                    _finite(bench_usable[b_last].close),
                    _finite(bench_usable[b_anchor].close),
                )
                if bench_return is None:
                    reasons["benchmark_return"] = "benchmark_close_unavailable"
        if bench_return is not None and since_anchor is not None:
            relative_return = _finite(since_anchor - bench_return)
        else:
            reasons.setdefault("relative_return", "benchmark_return_unavailable")
    else:
        reasons["benchmark_return"] = "no_benchmark_bars_provided"
        reasons["relative_return"] = "no_benchmark_bars_provided"

    # --- window statistics ------------------------------------------------
    window_closes = closes[anchor_idx:] if anchor_idx is not None else closes
    max_dd = _max_drawdown(window_closes)
    if max_dd is None:
        reasons.setdefault("max_drawdown", "no_usable_closes_in_window")

    rv20: float | None = None
    if n_bars >= 21:
        rv20 = _finite(realized_vol(closes, 20)[-1])
    if rv20 is None:
        reasons["realized_vol_20d"] = f"needs 21 bars, have {n_bars}"

    rv_anchor: float | None = None
    if len(window_closes) >= 3:
        # A sample stdev over the whole window: period = number of returns.
        period = len(window_closes) - 1
        rv_anchor = _finite(realized_vol(window_closes, period)[-1])
    if rv_anchor is None:
        reasons["realized_vol_since_anchor"] = (
            f"needs 3 bars since the anchor, have {len(window_closes)}"
        )

    volume_trend: float | None = None
    if n_bars >= 80:
        recent = math.fsum(volumes[-20:]) / 20.0
        prior = math.fsum(volumes[-80:-20]) / 60.0
        volume_trend = _pct_change(recent, prior)
        if volume_trend is None:
            reasons["volume_trend"] = "prior_60d_average_volume_not_positive"
    else:
        reasons["volume_trend"] = f"needs 80 bars, have {n_bars}"

    sma20, d20 = _sma_distance(closes, 20, last_close) if last_close else (None, None)
    sma50, d50 = _sma_distance(closes, 50, last_close) if last_close else (None, None)
    sma200, d200 = _sma_distance(closes, 200, last_close) if last_close else (None, None)
    for period, value in ((20, sma20), (50, sma50), (200, sma200)):
        if value is None:
            reasons[f"sma{period}"] = f"needs {period} bars, have {n_bars}"

    atr14: float | None = None
    atr_pct: float | None = None
    if n_bars >= 15:
        atr14 = _finite(atr(highs, lows, closes, 14)[-1])
    if atr14 is None:
        reasons["atr14"] = f"needs 15 bars, have {n_bars}"
    else:
        if last_close is not None and last_close > 0.0:
            atr_pct = _finite(atr14 / last_close)
        if atr_pct is None:
            reasons["atr_pct"] = "last_close_not_positive"

    # --- 52-week band -----------------------------------------------------
    window_52w = usable[-252:] if n_bars >= 1 else []
    high_52w = max((b.high for b in window_52w), default=None)
    low_52w = min((b.low for b in window_52w), default=None)
    dist_high = _pct_change(last_close, high_52w)
    dist_low = _pct_change(last_close, low_52w)
    if n_bars < 252:
        reasons["52w_window"] = f"partial 52w window: {n_bars} bars"
    if dist_high is None:
        reasons.setdefault("distance_from_52w_high_pct", "52w_high_unavailable")
    if dist_low is None:
        reasons.setdefault("distance_from_52w_low_pct", "52w_low_unavailable")

    return PriceContext(
        as_of_date_et=as_of_date_et,
        anchor_date_et=anchor_date_used,
        anchor_close=anchor_close,
        anchor_basis=anchor_basis,
        last_close=last_close,
        bars_through=bars_through,
        n_bars=n_bars,
        since_anchor_return=since_anchor,
        run_up_pct=since_anchor,
        benchmark_return=bench_return,
        relative_return=relative_return,
        max_drawdown=max_dd,
        realized_vol_20d=rv20,
        realized_vol_since_anchor=rv_anchor,
        volume_trend=volume_trend,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        sma20_distance_pct=d20,
        sma50_distance_pct=d50,
        sma200_distance_pct=d200,
        atr14=atr14,
        atr_pct=atr_pct,
        high_52w=_finite(high_52w) if high_52w is not None else None,
        low_52w=_finite(low_52w) if low_52w is not None else None,
        distance_from_52w_high_pct=dist_high,
        distance_from_52w_low_pct=dist_low,
        reasons=reasons,
    )
