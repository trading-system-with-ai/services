"""Backtest Engine V1 (development plan Phase 3, §20).

Pure, deterministic, dependency-free replay of the shared signal engines over
daily OHLCV series — no DB, no FastAPI. The engine imports its signal logic
EXCLUSIVELY from :mod:`libs.trading_core.signals` so backtest and live run the
exact same code (plan §21 — MANDATORY; never reimplement signals here).

V1 SCOPE — LONG STOCK ONLY (plan §5): the engine is long-only and never
shorts. Long Call / Long Put backtesting is deliberately DEFERRED until real
option-chain data exists; we never fabricate option prices from stock bars
(plan §5, §20).

Bias controls (plan §20.3), the heart of the replay loop:

- Walk forward bar by bar. The decision at bar ``t`` sees ONLY data up to and
  including ``t`` (``closes[:t+1]`` etc.). Passing full arrays to the signal
  engines is forbidden — no look-ahead, ever.
- A decision at the close of ``t`` fills at the OPEN of ``t+1`` (explicit
  fill model, plan §44 rule 11 — never same-bar, never midpoint), with
  slippage and per-share commission applied both ways.
- The engine only REPORTS in-sample / out-of-sample metrics; it never
  optimizes anything on the OOS segment (plan §44 rule 16).
- NO TRADE is a valid output (plan §44 rule 18): a series that never
  qualifies produces zero trades and honest null metrics.

Every threshold is a parameter on :class:`BacktestParams` (plan §6.2, §44
rule 2); every trade carries human-readable entry/exit reasons with the real
numbers used (audit explainability, plan §38).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import mean, stdev
from typing import Sequence

from libs.trading_core.features import atr
from libs.trading_core.models import DirectionalBias, MarketRegime
from libs.trading_core.signals import (
    DirectionalParams,
    RegimeParams,
    classify_regime,
    score_direction,
)

#: Fixed notional starting capital of every backtest run. All reported
#: metrics are relative (percentages / ratios), so the scale only influences
#: integer share flooring; it is a documented engine constant, not a tunable
#: strategy threshold (plan §44 rule 2 governs thresholds, not numeraires).
INITIAL_EQUITY: float = 100_000.0

#: ATR period for the trailing stop and time stop (plan §11.5 / §11.6 pin
#: these exits to atr14). Kept as a named module constant because the V1
#: BacktestParams schema (plan §20) does not carry it; it becomes a parameter
#: when the schema does.
ATR_PERIOD: int = 14

#: Trading days per year used to annualize CAGR / Sharpe / Sortino.
TRADING_DAYS_PER_YEAR: int = 252

#: Regimes in which a long-stock entry is allowed (plan §5, §11.1).
_BULL_REGIMES = frozenset({MarketRegime.STRONG_BULL, MarketRegime.MILD_BULL})


@dataclass(frozen=True)
class BacktestParams:
    """Backtest parameters for the V1 long-stock engine (plan §20, §44 rule 2).

    Every field is a tunable backtest parameter — the defaults are starting
    points for optimization, never truths (plan §6.2):

    - ``position_pct``: fraction of current equity deployed per entry, in
      ``(0, 1]``. Shares = ``floor(equity * position_pct / fill_price)``.
    - ``commission_per_share``: commission charged per share on BOTH the buy
      and the sell fill (explicit cost model, plan §44 rule 11).
    - ``slippage_bps``: slippage in basis points applied against us on every
      fill — buys at ``open * (1 + bps/10000)``, sells at
      ``open * (1 - bps/10000)`` (plan §44 rule 11).
    - ``entry_edge_threshold``: minimum ``directional_edge`` required (with a
      BULL bias and a bull regime) to enter (plan §11.1).
    - ``exit_edge_threshold``: edge below which the SIGNAL_DECAY exit fires.
      Exits must be easier than entries (plan §11.1), so it must be <=
      ``entry_edge_threshold``.
    - ``atr_trail_k``: ATR-multiple of the trailing stop — exit when close
      drops below ``highest_close_since_entry - atr_trail_k * atr14``
      (plan §11.5).
    - ``time_stop_bars``: bars after which a position that has not moved is
      abandoned (plan §11.6).
    - ``min_move_atr``: the "has not moved" bar for the time stop — the trade
      must be up at least ``min_move_atr * atr14`` to escape it (plan §11.6).
    - ``oos_split``: in-sample fraction of the series, in ``(0, 1)``; the
      out-of-sample segment starts at bar ``floor(n_bars * oos_split)`` and
      is only ever reported on, never optimized against (plan §44 rule 16).
    - ``warmup_bars``: bars withheld from trading at the start of the series
      so every indicator is fully formed before the first decision
      (plan §20.3).
    """

    position_pct: float = 1.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    entry_edge_threshold: float = 25.0
    exit_edge_threshold: float = 10.0
    atr_trail_k: float = 3.0
    time_stop_bars: int = 20
    min_move_atr: float = 1.0
    oos_split: float = 0.7
    warmup_bars: int = 200

    def __post_init__(self) -> None:
        if not (0.0 < self.position_pct <= 1.0):
            raise ValueError(
                f"position_pct must be in (0, 1], got {self.position_pct!r}"
            )
        if self.commission_per_share < 0.0:
            raise ValueError(
                "commission_per_share must be >= 0, got "
                f"{self.commission_per_share!r}"
            )
        if self.slippage_bps < 0.0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps!r}")
        if self.exit_edge_threshold > self.entry_edge_threshold:
            raise ValueError(
                "exit_edge_threshold must be <= entry_edge_threshold (exits are "
                f"easier than entries, plan §11.1), got "
                f"{self.exit_edge_threshold!r} > {self.entry_edge_threshold!r}"
            )
        if self.atr_trail_k <= 0.0:
            raise ValueError(f"atr_trail_k must be > 0, got {self.atr_trail_k!r}")
        if not isinstance(self.time_stop_bars, int) or self.time_stop_bars < 1:
            raise ValueError(
                f"time_stop_bars must be an integer >= 1, got {self.time_stop_bars!r}"
            )
        if self.min_move_atr < 0.0:
            raise ValueError(f"min_move_atr must be >= 0, got {self.min_move_atr!r}")
        if not (0.0 < self.oos_split < 1.0):
            raise ValueError(f"oos_split must be in (0, 1), got {self.oos_split!r}")
        if not isinstance(self.warmup_bars, int) or self.warmup_bars < 1:
            raise ValueError(
                f"warmup_bars must be an integer >= 1, got {self.warmup_bars!r}"
            )


@dataclass
class Trade:
    """One completed long-stock round trip (plan §20).

    ``entry_index`` / ``exit_index`` are the FILL bars (the ``t+1`` of the
    decision bar, plan §44 rule 11); prices are actual fill prices including
    slippage. ``entry_reason`` / ``exit_reason`` are human-readable with the
    real numbers the decision used (audit explainability, plan §38).
    ``pnl`` is net of commissions; ``return_pct`` is ``pnl`` over the entry
    cost basis (shares * entry_price + entry commission), in percent.
    """

    entry_index: int
    entry_date: date
    entry_price: float
    exit_index: int
    exit_date: date
    exit_price: float
    shares: int
    bars_held: int
    return_pct: float
    pnl: float
    entry_reason: str
    exit_reason: str


@dataclass
class SegmentMetrics:
    """Performance metrics of one segment (full / in_sample / out_of_sample).

    Exactly the SEG shape of the backtest record JSON (plan §20): a metric
    that is undefined for the segment (e.g. no trades, too few returns, zero
    variance) is ``None`` — never NaN and never Infinity (plan §44 rule 18:
    honest nulls over fabricated numbers).

    - ``total_return_pct``: equity change over the segment's slice, percent.
    - ``cagr_pct``: annualized growth from segment start/end equity over
      ``segment bars / 252`` years; ``None`` if the slice has < 2 points.
    - ``sharpe``: ``mean(daily_returns) / stdev(daily_returns) * sqrt(252)``
      (sample stdev); ``None`` with < 2 returns or zero stdev.
    - ``sortino``: same numerator over the downside deviation
      ``sqrt(mean(min(r, 0)^2))``; ``None`` with < 2 returns or no downside.
    - ``max_drawdown_pct``: most negative ``equity / running_max - 1`` within
      the segment slice, percent (<= 0).
    - ``win_rate``: winning trades / total trades, in [0, 1]; ``None`` if no
      trades.
    - ``profit_factor``: gross wins / gross losses; ``None`` if no losses.
    - ``expectancy_pct``: mean trade ``return_pct`` (the per-trade
      expectancy); ``None`` if no trades.
    - ``avg_trade_pct``: mean trade ``return_pct``; ``None`` if no trades.
    - ``avg_hold_bars``: mean ``bars_held``; ``None`` if no trades.
    - ``num_trades``: trades whose ENTRY bar lies in the segment.
    - ``exposure_pct``: percent of the segment's bars with a position on.
    """

    total_return_pct: float
    cagr_pct: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown_pct: float
    win_rate: float | None
    profit_factor: float | None
    expectancy_pct: float | None
    avg_trade_pct: float | None
    avg_hold_bars: float | None
    num_trades: int
    exposure_pct: float


@dataclass
class BacktestResult:
    """Full output of one backtest run (plan §20).

    ``dates`` / ``equity`` / ``drawdown`` are aligned per-bar arrays over the
    ENTIRE input series (daily mark-to-market ``cash + shares * close``;
    ``drawdown = equity / running_max - 1``). ``metrics`` holds exactly the
    keys ``"full"``, ``"in_sample"`` and ``"out_of_sample"``;
    ``oos_start_date`` is the first bar of the out-of-sample segment
    (plan §44 rule 16: the OOS segment is report-only).
    """

    trades: list[Trade]
    dates: list[date]
    equity: list[float]
    drawdown: list[float]
    metrics: dict[str, SegmentMetrics]
    oos_start_date: date | None


def _downside_deviation(returns: Sequence[float]) -> float:
    """Root-mean-square of the negative returns (0.0 if none are negative)."""
    return math.sqrt(
        math.fsum(min(r, 0.0) ** 2 for r in returns) / len(returns)
    )


def _segment_metrics(
    equity: Sequence[float],
    held: Sequence[bool],
    trades: Sequence[Trade],
) -> SegmentMetrics:
    """Compute one segment's metrics from its equity slice, per-bar position
    flags and the trades ENTERED inside it (plan §20).

    Every division is guarded: undefined metrics are ``None``, never
    NaN/Infinity (plan §44 rule 18).
    """
    n = len(equity)

    total_return_pct = 0.0
    if n >= 2 and equity[0] != 0.0:
        total_return_pct = (equity[-1] / equity[0] - 1.0) * 100.0

    cagr_pct: float | None = None
    if n >= 2 and equity[0] > 0.0 and equity[-1] > 0.0:
        years = (n - 1) / TRADING_DAYS_PER_YEAR
        cagr_pct = ((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) * 100.0

    returns = [
        equity[t] / equity[t - 1] - 1.0
        for t in range(1, n)
        if equity[t - 1] != 0.0
    ]
    sharpe: float | None = None
    sortino: float | None = None
    if len(returns) >= 2:
        sd = stdev(returns)
        if sd > 0.0:
            sharpe = mean(returns) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)
        dd = _downside_deviation(returns)
        if dd > 0.0:
            sortino = mean(returns) / dd * math.sqrt(TRADING_DAYS_PER_YEAR)

    max_drawdown_pct = 0.0
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        if running_max > 0.0:
            max_drawdown_pct = min(max_drawdown_pct, (value / running_max - 1.0) * 100.0)

    num_trades = len(trades)
    win_rate: float | None = None
    profit_factor: float | None = None
    expectancy_pct: float | None = None
    avg_trade_pct: float | None = None
    avg_hold_bars: float | None = None
    if num_trades:
        wins = sum(1 for t in trades if t.pnl > 0.0)
        win_rate = wins / num_trades
        gross_wins = math.fsum(t.pnl for t in trades if t.pnl > 0.0)
        gross_losses = -math.fsum(t.pnl for t in trades if t.pnl < 0.0)
        if gross_losses > 0.0:
            profit_factor = gross_wins / gross_losses
        expectancy_pct = mean(t.return_pct for t in trades)
        avg_trade_pct = expectancy_pct
        avg_hold_bars = mean(float(t.bars_held) for t in trades)

    exposure_pct = (sum(held) / n * 100.0) if n else 0.0

    return SegmentMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_drawdown_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        avg_trade_pct=avg_trade_pct,
        avg_hold_bars=avg_hold_bars,
        num_trades=num_trades,
        exposure_pct=exposure_pct,
    )


def run_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(),
    *,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Replay the shared signal engines over daily bars, long stock only
    (plan §20; V1 scope plan §5 — no shorting, ever).

    Replay semantics (plan §20.3 bias controls):

    - At bar ``t`` (``t >= params.warmup_bars``) signals are evaluated on the
      slices ``closes[:t+1]`` / ``highs[:t+1]`` / ``lows[:t+1]`` /
      ``volumes[:t+1]`` ONLY — the engines never see a future bar.
    - A decision at the close of ``t`` fills at the OPEN of ``t+1``
      (plan §44 rule 11): buys at ``open * (1 + slippage_bps/10000)``, sells
      at ``open * (1 - slippage_bps/10000)``, plus ``commission_per_share``
      each way.
    - ENTRY (only when flat): regime in {STRONG_BULL, MILD_BULL} AND
      bias == BULL AND ``directional_edge >= entry_edge_threshold``
      (plan §11.1). Shares = ``floor(equity * position_pct / fill_price)``
      (trimmed only if the commission would push cash below zero — the
      account never borrows, plan §5).
    - EXIT (any position, first match wins, in priority order):
      SIGNAL_FLIP (bias == BEAR) -> SIGNAL_DECAY
      (``edge < exit_edge_threshold``, easier than entry, plan §11.1) ->
      ATR_TRAIL (``close < highest_close_since_entry - atr_trail_k * atr14``,
      plan §11.5) -> TIME_STOP (``bars_held >= time_stop_bars`` and the move
      since entry is ``< min_move_atr * atr14``, plan §11.6). At the FINAL
      bar an open position is marked to that bar's close with exit_reason
      END_OF_DATA (a valuation, not a fill: no slippage, no exit commission).
    - Equity is marked to market daily (``cash + shares * close``);
      ``drawdown = equity / running_max - 1``.
    - IS/OOS (plan §44 rule 16): the OOS segment starts at bar
      ``floor(n_bars * oos_split)``; a trade belongs to the segment of its
      ENTRY bar; segment equity metrics are computed over the segment's slice
      of the equity curve. The engine only reports OOS results — nothing is
      ever fitted or optimized on them.

    ``regime_params`` / ``directional_params`` pass through to the shared
    signal engines (plan §21: same code as live; plan §6.2: every threshold a
    parameter). NO TRADE is a valid outcome (plan §44 rule 18).
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError(
            "dates, opens, highs, lows, closes and volumes must have equal "
            f"length, got {len(dates)}/{len(opens)}/{len(highs)}/{len(lows)}/"
            f"{n}/{len(volumes)}"
        )
    if n < 1:
        raise ValueError("run_backtest needs at least 1 bar")

    slip = params.slippage_bps / 10_000.0
    commission = params.commission_per_share

    # ATR is a recursive (Wilder) indicator seeded at a fixed index, so the
    # value at bar t of the full-series computation is bit-identical to the
    # last value of the computation over bars[:t+1] — precomputing it once is
    # an optimization, not look-ahead (plan §20.3).
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)

    cash = INITIAL_EQUITY
    shares = 0
    entry_index = -1
    entry_price = 0.0
    entry_cost = 0.0
    entry_reason = ""
    peak_close = -math.inf
    pending_entry: str | None = None
    pending_exit: str | None = None

    trades: list[Trade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def close_trade(t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, shares
        proceeds = shares * exit_price - shares * exit_commission
        cash += proceeds
        pnl = proceeds - entry_cost
        trades.append(
            Trade(
                entry_index=entry_index,
                entry_date=dates[entry_index],
                entry_price=entry_price,
                exit_index=t,
                exit_date=dates[t],
                exit_price=exit_price,
                shares=shares,
                bars_held=t - entry_index,
                return_pct=(pnl / entry_cost * 100.0) if entry_cost > 0.0 else 0.0,
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=reason,
            )
        )
        shares = 0

    for t in range(n):
        # --- 1. Fill the decision made at the close of t-1 at this open ----
        # (plan §44 rule 11: explicit next-open fill model).
        if pending_entry is not None:
            fill = opens[t] * (1.0 + slip)
            # Flat => equity == cash, and nothing moved cash since the
            # decision at the close of t-1: size off it directly.
            qty = math.floor(cash * params.position_pct / fill) if fill > 0.0 else 0
            # Never let costs push cash below zero — a long-only cash account
            # cannot borrow (plan §5).
            while qty > 0 and qty * (fill + commission) > cash:
                qty -= 1
            if qty > 0:
                cash -= qty * (fill + commission)
                shares = qty
                entry_index = t
                entry_price = fill
                entry_cost = qty * (fill + commission)
                entry_reason = pending_entry
                peak_close = -math.inf
            pending_entry = None
        elif pending_exit is not None and shares > 0:
            close_trade(t, opens[t] * (1.0 - slip), commission, pending_exit)
            pending_exit = None

        held_during_bar = shares > 0

        # --- 2. Decide at the close of t (fills at the open of t+1) --------
        if shares > 0:
            peak_close = max(peak_close, closes[t])
            if t == n - 1:
                # Priority (5) END_OF_DATA (plan §20): the final bar marks any
                # open position to its close — a valuation, not a fill.
                close_trade(
                    t,
                    closes[t],
                    0.0,
                    f"END_OF_DATA: marked to final close {closes[t]:.4f}",
                )
            else:
                pending_exit = _evaluate_exit(
                    t,
                    closes,
                    highs,
                    lows,
                    volumes,
                    atr14,
                    params,
                    directional_params,
                    entry_index,
                    entry_price,
                    peak_close,
                )
        elif pending_entry is None and params.warmup_bars <= t < n - 1:
            pending_entry = _evaluate_entry(
                t, closes, highs, lows, volumes, params, regime_params, directional_params
            )

        equity.append(cash + shares * closes[t])
        held_flags.append(held_during_bar)

    # --- Drawdown over the full curve: equity / running_max - 1 ------------
    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    # --- IS/OOS segmentation (plan §44 rule 16: report-only) ----------------
    boundary = math.floor(n * params.oos_split)
    oos_start_date = dates[boundary] if boundary < n else None
    metrics = {
        "full": _segment_metrics(equity, held_flags, trades),
        "in_sample": _segment_metrics(
            equity[:boundary],
            held_flags[:boundary],
            [tr for tr in trades if tr.entry_index < boundary],
        ),
        "out_of_sample": _segment_metrics(
            equity[boundary:],
            held_flags[boundary:],
            [tr for tr in trades if tr.entry_index >= boundary],
        ),
    }

    return BacktestResult(
        trades=trades,
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=metrics,
        oos_start_date=oos_start_date,
    )


def _evaluate_entry(
    t: int,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    params: BacktestParams,
    regime_params: RegimeParams,
    directional_params: DirectionalParams,
) -> str | None:
    """Entry decision at the close of bar ``t`` on data ``[:t+1]`` ONLY
    (plan §20.3). Returns the human-readable entry reason, or ``None``.

    Long stock only (plan §5): regime in {STRONG_BULL, MILD_BULL} AND
    bias == BULL AND ``directional_edge >= entry_edge_threshold``
    (plan §11.1).
    """
    regime = classify_regime(
        closes[: t + 1], highs[: t + 1], lows[: t + 1], params=regime_params
    )
    if regime.classification not in _BULL_REGIMES:
        return None
    direction = score_direction(
        closes[: t + 1],
        highs[: t + 1],
        lows[: t + 1],
        volumes=volumes[: t + 1],
        params=directional_params,
    )
    if (
        direction.bias is DirectionalBias.BULL
        and direction.directional_edge >= params.entry_edge_threshold
    ):
        return (
            f"edge {direction.directional_edge:.1f} >= "
            f"{params.entry_edge_threshold:.1f}, regime "
            f"{regime.classification.value}, bias BULL"
        )
    return None


def _evaluate_exit(
    t: int,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    atr14: list[float | None],
    params: BacktestParams,
    directional_params: DirectionalParams,
    entry_index: int,
    entry_price: float,
    peak_close: float,
) -> str | None:
    """Exit decision at the close of bar ``t`` on data ``[:t+1]`` ONLY
    (plan §20.3). First match wins, in priority order (plan §11):

    1. SIGNAL_FLIP  — bias == BEAR.
    2. SIGNAL_DECAY — ``edge < exit_edge_threshold`` (easier than entry,
       plan §11.1).
    3. ATR_TRAIL    — ``close < peak_close - atr_trail_k * atr14`` (plan §11.5).
    4. TIME_STOP    — held ``>= time_stop_bars`` bars without a
       ``min_move_atr * atr14`` move (plan §11.6).

    (Priority 5, END_OF_DATA, is handled by the replay loop at the final
    bar.) Returns the human-readable exit reason, or ``None``.
    """
    direction = score_direction(
        closes[: t + 1],
        highs[: t + 1],
        lows[: t + 1],
        volumes=volumes[: t + 1],
        params=directional_params,
    )
    edge = direction.directional_edge

    # (1) SIGNAL_FLIP — the directional evidence now points down.
    if direction.bias is DirectionalBias.BEAR:
        return f"SIGNAL_FLIP: bias BEAR, edge {edge:.1f}"

    # (2) SIGNAL_DECAY — the edge no longer justifies holding (plan §11.1).
    if edge < params.exit_edge_threshold:
        return (
            f"SIGNAL_DECAY: edge {edge:.1f} < {params.exit_edge_threshold:.1f}"
        )

    atr_t = atr14[t]
    if atr_t is not None:
        # (3) ATR_TRAIL (plan §11.5).
        trail = peak_close - params.atr_trail_k * atr_t
        if closes[t] < trail:
            return (
                f"ATR_TRAIL: close {closes[t]:.4f} < trail {trail:.4f} "
                f"(peak {peak_close:.4f} - {params.atr_trail_k:.2f} * "
                f"atr14 {atr_t:.4f})"
            )

        # (4) TIME_STOP (plan §11.6).
        bars_held = t - entry_index
        move = closes[t] - entry_price
        if bars_held >= params.time_stop_bars and move < params.min_move_atr * atr_t:
            return (
                f"TIME_STOP: held {bars_held} bars >= {params.time_stop_bars}, "
                f"move {move:.4f} < {params.min_move_atr:.2f} * atr14 {atr_t:.4f}"
            )

    return None
