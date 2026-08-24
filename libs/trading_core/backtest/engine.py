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
- Exits run the SHARED live exit engine (libs.trading_core.exits) with the
  SHARED §12.1 stop sizing (ATR_STOP_MULTIPLE) — backtest and live cannot
  drift apart (§21; user parity mandate 2026-08-16).
- IS/OOS segmentation was REMOVED 2026-08-16 (user decision): with purely
  manual parameter iteration the platform reports full-period metrics only;
  the split returns if/when ML-driven parameter search is introduced.
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

from libs.trading_core.exits import ExitParams, PositionState, evaluate_exit
from libs.trading_core.features import atr
from libs.trading_core.models import DirectionalBias, MarketRegime
from libs.trading_core.risk.engine import ATR_STOP_MULTIPLE
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
#: Bear-side mirror (2026-08-17): the LONG_PUT / BEAR_PUT_SPREAD entry gate.
_BEAR_REGIMES = frozenset({MarketRegime.STRONG_BEAR, MarketRegime.MILD_BEAR})

#: Valid fill-model names (plan §20.2). Order is the optimism ordering:
#: OPTIMISTIC (frictionless best case) -> CONSERVATIVE (default slippage) ->
#: WORST (adversarial slippage floor).
FILL_MODELS: tuple[str, ...] = ("OPTIMISTIC", "CONSERVATIVE", "WORST")


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
    - ``warmup_bars``: bars withheld from trading at the start of the series
      so every indicator is fully formed before the first decision
      (plan §20.3).
    - ``fill_model``: how pessimistic the fill price is (plan §20.2 — "Never
      treat historical mid as guaranteed fill"). One of :data:`FILL_MODELS`,
      mapped onto daily-bar data (we have no historical bid/ask yet) as an
      effective slippage in bps applied both ways around the next open:

      - ``"OPTIMISTIC"``   -> 0 bps (frictionless best case: the raw next open),
      - ``"CONSERVATIVE"`` -> ``slippage_bps`` (the pre-existing default
        behavior — a CONSERVATIVE run is bit-identical to the engine before
        fill models existed),
      - ``"WORST"``        -> ``max(slippage_bps, worst_slippage_bps)``
        (an adversarial floor; never better than CONSERVATIVE).

      Once real quote data lands, ``"WORST"`` becomes buy-at-ask /
      sell-at-bid instead of a bps proxy (plan §20.2). Commission is
      unchanged by the fill model.
    - ``worst_slippage_bps``: the slippage floor (bps, >= 0) the ``"WORST"``
      model enforces via ``max(slippage_bps, worst_slippage_bps)``.
    """

    position_pct: float = 1.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    entry_edge_threshold: float = 25.0
    exit_edge_threshold: float = 10.0
    atr_trail_k: float = 3.0
    time_stop_bars: int = 20
    min_move_atr: float = 1.0
    warmup_bars: int = 200
    fill_model: str = "CONSERVATIVE"
    worst_slippage_bps: float = 25.0
    # --- Instrument leg (user mandate 2026-08-17: options join the backtest).
    # "LONG_STOCK" (V1 engine), "LONG_CALL", or "BULL_CALL_SPREAD" (options
    # engine over REAL historical contract bars — see backtest/options.py).
    # LONG_PUT / BEAR_PUT_SPREAD arrive with the bear-side signal mirror.
    instrument: str = "LONG_STOCK"
    # LONG_CALL leg parameters (research parameters, §6.2). Contract choice
    # is moneyness/DTE-based because historical greeks/OI do not exist
    # (data-source-architecture.md) — deterministic from REAL strike grids
    # and REAL underlying closes, never invented data.
    target_dte_min: int = 30
    target_dte_max: int = 90
    strike_otm_pct: float = 0.0  # 0 = ATM; 0.05 = 5% OTM
    option_premium_pct: float = 0.10  # equity fraction spent on premium
    commission_per_contract: float = 0.65
    # §20.2 proxy for option spreads (NO historical NBBO exists): bps applied
    # both ways around the option bar price, by fill model.
    option_slippage_bps: float = 100.0
    worst_option_slippage_bps: float = 300.0
    # BULL_CALL_SPREAD leg: short strike ≈ long strike + this fraction of
    # spot (nearest REAL strike, same expiry) — mirrors §9-S width_pct_target.
    spread_width_pct: float = 0.05

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
        if self.instrument not in (
            "LONG_STOCK",
            "LONG_CALL",
            "LONG_PUT",
            "BULL_CALL_SPREAD",
            "BEAR_PUT_SPREAD",
            "COVERED_CALL",
            "CASH_SECURED_PUT",
            "SHORT_STOCK",
            # AUTO: the §8 matrix picks the instrument daily from the live
            # signal stack (backtest/auto.py, Phase B 2026-08-20).
            "AUTO",
        ):
            raise ValueError(
                "instrument must be one of ('LONG_STOCK', 'LONG_CALL', "
                "'LONG_PUT', 'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', "
                "'COVERED_CALL', 'CASH_SECURED_PUT', 'SHORT_STOCK', 'AUTO'), "
                f"got {self.instrument!r}"
            )
        if not isinstance(self.target_dte_min, int) or self.target_dte_min < 1:
            raise ValueError(
                f"target_dte_min must be an integer >= 1, got {self.target_dte_min!r}"
            )
        if (
            not isinstance(self.target_dte_max, int)
            or self.target_dte_max < self.target_dte_min
        ):
            raise ValueError(
                "target_dte_max must be an integer >= target_dte_min, got "
                f"{self.target_dte_max!r}"
            )
        if not (-0.5 <= self.strike_otm_pct <= 0.5):
            raise ValueError(
                f"strike_otm_pct must be in [-0.5, 0.5], got {self.strike_otm_pct!r}"
            )
        if not (0.0 < self.option_premium_pct <= 1.0):
            raise ValueError(
                f"option_premium_pct must be in (0, 1], got {self.option_premium_pct!r}"
            )
        if self.commission_per_contract < 0.0:
            raise ValueError(
                "commission_per_contract must be >= 0, got "
                f"{self.commission_per_contract!r}"
            )
        if self.option_slippage_bps < 0.0:
            raise ValueError(
                f"option_slippage_bps must be >= 0, got {self.option_slippage_bps!r}"
            )
        if self.worst_option_slippage_bps < 0.0:
            raise ValueError(
                "worst_option_slippage_bps must be >= 0, got "
                f"{self.worst_option_slippage_bps!r}"
            )
        if not (0.0 < self.spread_width_pct <= 0.5):
            raise ValueError(
                f"spread_width_pct must be in (0, 0.5], got "
                f"{self.spread_width_pct!r}"
            )
        if not isinstance(self.warmup_bars, int) or self.warmup_bars < 1:
            raise ValueError(
                f"warmup_bars must be an integer >= 1, got {self.warmup_bars!r}"
            )
        if self.fill_model not in FILL_MODELS:
            raise ValueError(
                f"fill_model must be one of {list(FILL_MODELS)}, got "
                f"{self.fill_model!r}"
            )
        if self.worst_slippage_bps < 0.0:
            raise ValueError(
                f"worst_slippage_bps must be >= 0, got {self.worst_slippage_bps!r}"
            )

    def effective_slippage_bps(self) -> float:
        """Effective slippage in bps under this params' ``fill_model``
        (plan §20.2 mapped to daily-bar data — see the class docstring):
        OPTIMISTIC -> 0, CONSERVATIVE -> ``slippage_bps``, WORST ->
        ``max(slippage_bps, worst_slippage_bps)``."""
        if self.fill_model == "OPTIMISTIC":
            return 0.0
        if self.fill_model == "WORST":
            return max(self.slippage_bps, self.worst_slippage_bps)
        return self.slippage_bps

    def effective_option_slippage_bps(self) -> float:
        """§20.2 option-spread proxy under this params' fill model — same
        optimism ordering as the stock helper (no historical NBBO exists;
        see data-source-architecture.md)."""
        if self.fill_model == "OPTIMISTIC":
            return 0.0
        if self.fill_model == "WORST":
            return max(self.option_slippage_bps, self.worst_option_slippage_bps)
        return self.option_slippage_bps


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
    ``drawdown = equity / running_max - 1``). ``metrics`` covers the whole
    period (IS/OOS segmentation removed 2026-08-16 — see module docstring).
    """

    trades: list[Trade]
    dates: list[date]
    equity: list[float]
    drawdown: list[float]
    metrics: SegmentMetrics


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
      (plan §44 rule 11): buys at ``open * (1 + slip)``, sells at
      ``open * (1 - slip)``, plus ``commission_per_share`` each way, where
      ``slip`` is the EFFECTIVE slippage chosen by ``params.fill_model``
      (plan §20.2 — "Never treat historical mid as guaranteed fill"):
      OPTIMISTIC -> 0 bps, CONSERVATIVE -> ``slippage_bps``, WORST ->
      ``max(slippage_bps, worst_slippage_bps)``. The mapping is a bps proxy
      because only daily OHLCV exists — there is no historical bid/ask yet;
      when real quote data lands, WORST becomes buy-at-ask / sell-at-bid
      (plan §20.2). Commission is identical across fill models.
    - ENTRY (only when flat): regime in {STRONG_BULL, MILD_BULL} AND
      bias == BULL AND ``directional_edge >= entry_edge_threshold``
      (plan §11.1) AND atr14 is computable (the §12.1 stop must be sizable —
      live refuses an unsized stock entry, so the replay does too). Shares =
      ``floor(equity * position_pct / fill_price)`` (trimmed only if the
      commission would push cash below zero — the account never borrows,
      plan §5). The stop distance is fixed at entry:
      ``ATR_STOP_MULTIPLE * atr14`` at the decision bar (§12.1, the SAME
      shared constant the live gate chain uses).
    - EXIT: the SHARED live exit engine (libs.trading_core.exits
      .evaluate_exit — §21, never reimplemented), first match in live
      priority order: HARD_STOP (close <= entry - stop_distance, §11.3) ->
      SIGNAL_FLIP -> SIGNAL_DECAY (easier than entry, §11.1) -> ATR_TRAIL
      (§11.5) -> TIME_STOP (§11.6). At the FINAL bar an open position is
      marked to that bar's close with exit_reason END_OF_DATA (a valuation,
      not a fill: no slippage, no exit commission).
    - Equity is marked to market daily (``cash + shares * close``);
      ``drawdown = equity / running_max - 1``.

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

    # Effective slippage from the fill model (plan §20.2), computed ONCE and
    # applied symmetrically to buy and sell fills below.
    slip = params.effective_slippage_bps() / 10_000.0
    commission = params.commission_per_share

    # ATR is a recursive (Wilder) indicator seeded at a fixed index, so the
    # value at bar t of the full-series computation is bit-identical to the
    # last value of the computation over bars[:t+1] — precomputing it once is
    # an optimization, not look-ahead (plan §20.3).
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)

    # Exit rules: the SHARED live engine's parameters, mapped 1:1 from
    # BacktestParams (§21 — same logic, same knobs as live).
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    shares = 0
    entry_index = -1
    entry_price = 0.0
    entry_cost = 0.0
    entry_reason = ""
    entry_edge = 0.0
    entry_stop_distance = 0.0
    peak_close = -math.inf
    # (reason, edge_at_decision, stop_distance sized at the decision bar)
    pending_entry: tuple[str, float, float] | None = None
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
                entry_reason, entry_edge, entry_stop_distance = pending_entry
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
                # SHARED live exit engine (§21): peak_close already folds in
                # the current close, matching the live caller convention
                # documented on PositionState.highest_close_since_entry.
                decision = evaluate_exit(
                    PositionState(
                        entry_price=entry_price,
                        stop_distance=entry_stop_distance,
                        entry_edge=entry_edge,
                        bars_held=t - entry_index,
                        highest_close_since_entry=peak_close,
                    ),
                    closes[: t + 1],
                    highs[: t + 1],
                    lows[: t + 1],
                    volumes=volumes[: t + 1],
                    params=exit_params,
                    directional_params=directional_params,
                )
                pending_exit = (
                    next(
                        r
                        for r in decision.reasons
                        if r.startswith(decision.triggered_rule or "")
                    )
                    if decision.should_exit
                    else None
                )
        elif pending_entry is None and params.warmup_bars <= t < n - 1:
            # §12.1 parity: live refuses a LONG_STOCK entry whose stop cannot
            # be sized (no ATR) — the replay refuses identically.
            atr_t = atr14[t]
            if atr_t is not None:
                entry_eval = _evaluate_entry(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, edge = entry_eval
                    pending_entry = (reason, edge, ATR_STOP_MULTIPLE * atr_t)

        equity.append(cash + shares * closes[t])
        held_flags.append(held_during_bar)

    # --- Drawdown over the full curve: equity / running_max - 1 ------------
    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
    )


def run_short_stock_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="SHORT_STOCK"),
    *,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Replay margin-backed SHORT STOCK (roadmap Phase 3) — the exact bear
    mirror of :func:`run_backtest`, same shared engines (§21).

    - ENTRY (only when flat): the bear mirror — regime in {STRONG_BEAR,
      MILD_BEAR} AND bias == BEAR AND ``directional_edge <=
      -entry_edge_threshold`` (:func:`_evaluate_entry_bear`, the same
      helper the LONG_PUT leg replays) AND atr14 computable. The short
      SELL fills at ``open * (1 - slip)`` (slippage against the seller)
      and the PROCEEDS are credited to cash; shares =
      ``floor(equity * position_pct / fill)`` — the shorted notional is
      capped by the same position_pct that caps a long, the replay's
      margin-usage model.
    - EXIT: the SHARED live exit engine with ``direction="BEAR"`` — the
      hard stop sits ABOVE entry (``close >= entry + stop_distance``), the
      ATR trail hangs above the running trough, signal exits fire on the
      OPPOSING (BULL) bias. The cover BUY fills at ``open * (1 + slip)``
      and cash is debited. END_OF_DATA marks to the final close.
    - Equity while short = ``cash - shares * close`` (the liability is
      marked live), so equity moves by exactly the short's P&L.
    - ``return_pct`` is measured on the SHORTED NOTIONAL at entry
      (qty * fill) — the capital the position_pct cap allocated.
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError(
            "dates, opens, highs, lows, closes and volumes must have equal "
            f"length, got {len(dates)}/{len(opens)}/{len(highs)}/{len(lows)}/"
            f"{n}/{len(volumes)}"
        )
    if n < 1:
        raise ValueError("run_short_stock_backtest needs at least 1 bar")

    slip = params.effective_slippage_bps() / 10_000.0
    commission = params.commission_per_share
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    shares = 0
    entry_index = -1
    entry_price = 0.0
    entry_notional = 0.0
    entry_commission_paid = 0.0
    entry_reason = ""
    entry_edge = 0.0
    entry_stop_distance = 0.0
    trough_close = math.inf
    pending_entry: tuple[str, float, float] | None = None
    pending_exit: str | None = None

    trades: list[Trade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def cover_trade(t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, shares
        cost = shares * exit_price + shares * exit_commission
        cash -= cost
        pnl = (
            shares * (entry_price - exit_price)
            - entry_commission_paid
            - shares * exit_commission
        )
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
                return_pct=(
                    pnl / entry_notional * 100.0 if entry_notional > 0.0 else 0.0
                ),
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=reason,
            )
        )
        shares = 0

    for t in range(n):
        # --- 1. Fill the decision made at the close of t-1 at this open ----
        if pending_entry is not None:
            fill = opens[t] * (1.0 - slip)  # the short SELL fills lower
            qty = math.floor(cash * params.position_pct / fill) if fill > 0.0 else 0
            if qty > 0:
                cash += qty * fill - qty * commission  # proceeds credited
                shares = qty
                entry_index = t
                entry_price = fill
                entry_notional = qty * fill
                entry_commission_paid = qty * commission
                entry_reason, entry_edge, entry_stop_distance = pending_entry
                trough_close = math.inf
            pending_entry = None
        elif pending_exit is not None and shares > 0:
            cover_trade(t, opens[t] * (1.0 + slip), commission, pending_exit)
            pending_exit = None

        held_during_bar = shares > 0

        # --- 2. Decide at the close of t (fills at the open of t+1) --------
        if shares > 0:
            trough_close = min(trough_close, closes[t])
            if t == n - 1:
                cover_trade(
                    t,
                    closes[t],
                    0.0,
                    f"END_OF_DATA: marked to final close {closes[t]:.4f}",
                )
            else:
                decision = evaluate_exit(
                    PositionState(
                        entry_price=entry_price,
                        stop_distance=entry_stop_distance,
                        entry_edge=entry_edge,
                        bars_held=t - entry_index,
                        highest_close_since_entry=max(closes[entry_index : t + 1]),
                        direction="BEAR",
                        lowest_close_since_entry=trough_close,
                    ),
                    closes[: t + 1],
                    highs[: t + 1],
                    lows[: t + 1],
                    volumes=volumes[: t + 1],
                    params=exit_params,
                    directional_params=directional_params,
                )
                pending_exit = (
                    next(
                        r
                        for r in decision.reasons
                        if r.startswith(decision.triggered_rule or "")
                    )
                    if decision.should_exit
                    else None
                )
        elif pending_entry is None and params.warmup_bars <= t < n - 1:
            atr_t = atr14[t]
            if atr_t is not None:
                entry_eval = _evaluate_entry_bear(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, edge = entry_eval
                    pending_entry = (reason, edge, ATR_STOP_MULTIPLE * atr_t)

        equity.append(cash - shares * closes[t])
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
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
) -> tuple[str, float] | None:
    """Entry decision at the close of bar ``t`` on data ``[:t+1]`` ONLY
    (plan §20.3). Returns ``(human-readable entry reason, edge)`` — the edge
    seeds PositionState.entry_edge for the shared exit engine — or ``None``.

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
            f"{regime.classification.value}, bias BULL",
            direction.directional_edge,
        )
    return None


def _evaluate_entry_bear(
    t: int,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    params: BacktestParams,
    regime_params: RegimeParams,
    directional_params: DirectionalParams,
) -> tuple[str, float] | None:
    """The bear-side mirror of :func:`_evaluate_entry` (2026-08-17): regime
    in {STRONG_BEAR, MILD_BEAR} AND bias == BEAR AND ``directional_edge <=
    -entry_edge_threshold`` — the LONG_PUT / BEAR_PUT_SPREAD entry gate,
    with the same §20.3 no-look-ahead slices."""
    regime = classify_regime(
        closes[: t + 1], highs[: t + 1], lows[: t + 1], params=regime_params
    )
    if regime.classification not in _BEAR_REGIMES:
        return None
    direction = score_direction(
        closes[: t + 1],
        highs[: t + 1],
        lows[: t + 1],
        volumes=volumes[: t + 1],
        params=directional_params,
    )
    if (
        direction.bias is DirectionalBias.BEAR
        and direction.directional_edge <= -params.entry_edge_threshold
    ):
        return (
            f"edge {direction.directional_edge:.1f} <= "
            f"-{params.entry_edge_threshold:.1f}, regime "
            f"{regime.classification.value}, bias BEAR",
            direction.directional_edge,
        )
    return None


