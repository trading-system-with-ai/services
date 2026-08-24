"""Options Backtest Engine — LONG_CALL leg (user mandate 2026-08-17).

Replays the SAME bull entry signal as the stock engine, but expresses it by
buying a REAL historical call contract. The fabrication ban holds
throughout (data-source-architecture.md "Historical options data"):

- Contract choice is deterministic over the REAL strike grid / expirations
  that existed at the decision date (injected via ``contract_provider`` —
  the engine itself never touches a network).
- Entries, marks and exits price off the contract's REAL daily bars
  (Alpaca serves full contract-life bars from ~Feb 2024). A day the
  contract did not trade simply has no bar: entry fills are SKIPPED when
  the fill bar is missing, marks carry the last REAL traded price, and the
  premium stop honestly reports "insufficient data" on such days.
- The spread is a §20.2 bps proxy (``option_slippage_bps`` by fill model):
  historical NBBO does not exist at this provider, and we say so rather
  than invent quotes.

Exits run the LIVE option exit engine — :func:`evaluate_option_exit`
(§21: PREMIUM_HARD_STOP §11.3, DTE_EXIT §11.7, then the shared underlying
signal/trail/time rules) — never a reimplementation. Two engine-only
terminal cases exist on top: EXPIRY_SETTLEMENT (still held at expiry:
settled at intrinsic value off the REAL underlying close — contractual
arithmetic, not a price guess) and END_OF_DATA (marked to the last real
option price).

NO TRADE is a valid output (plan §44 rule 18).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

from libs.trading_core.exits import (
    ExitParams,
    OptionState,
    PositionState,
    ShortPremiumState,
    evaluate_exit,
    evaluate_option_exit,
    evaluate_short_premium_exit,
)
from libs.trading_core.features import atr
from libs.trading_core.risk.engine import ATR_STOP_MULTIPLE
from libs.trading_core.signals import DirectionalParams, RegimeParams

from .engine import (
    ATR_PERIOD,
    INITIAL_EQUITY,
    BacktestParams,
    BacktestResult,
    _evaluate_entry,
    _evaluate_entry_bear,
    _segment_metrics,
)

#: One option contract = 100 underlying shares (OCC standard).
CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class OptionLegBars:
    """One selected call contract's REAL daily history.

    ``bars`` maps trading date -> (open, close) of the contract's actual
    daily bar; dates with no trades are simply absent (honest gaps, never
    interpolated).
    """

    symbol: str
    strike: float
    expiry: date
    bars: dict[date, tuple[float, float]]


#: Called at a bull entry decision: (decision_date, spot_close) -> the
#: contract to buy, with its real bars, or None when no §-eligible contract
#: exists / no data is available (the entry is then skipped).
ContractProvider = Callable[[date, float], OptionLegBars | None]


@dataclass
class OptionTrade:
    """One completed long-call round trip.

    Field names shared with the stock ``Trade`` (entry_index/date, exit_*,
    bars_held, return_pct, pnl, reasons) so :func:`_segment_metrics` reads
    both; premiums are PER SHARE (quote convention), ``pnl`` is per the
    whole position net of commissions.
    """

    entry_index: int
    entry_date: date
    entry_price: float  # premium per share, actual fill incl. slippage
    exit_index: int
    exit_date: date
    exit_price: float  # premium per share (or intrinsic at settlement)
    contracts: int
    bars_held: int
    return_pct: float
    pnl: float
    entry_reason: str
    exit_reason: str
    contract_symbol: str = ""
    strike: float = 0.0
    contract_expiry: date | None = None
    # BULL_CALL_SPREAD trades: the SHORT leg's identity (empty for singles);
    # entry/exit prices above are then NET debit per share.
    short_symbol: str = ""
    short_strike: float = 0.0


def run_call_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="LONG_CALL"),
    *,
    contract_provider: ContractProvider,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Replay the bull signal, expressed as LONG_CALL over real contract bars.

    Same §20.3 bias controls as the stock engine: decisions at bar ``t`` see
    underlying data ``[:t+1]`` only; a decision at the close of ``t`` fills
    at the option's next-bar OPEN with the §20.2 option slippage proxy and
    per-contract commission both ways. The contract itself is chosen by
    ``contract_provider`` AT the decision date (it must only use
    information available then — the router's resolver picks by
    moneyness/DTE from the real grid).
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError(
            "dates, opens, highs, lows, closes and volumes must have equal "
            f"length, got {len(dates)}/{len(opens)}/{len(highs)}/{len(lows)}/"
            f"{n}/{len(volumes)}"
        )
    if n < 1:
        raise ValueError("run_call_backtest needs at least 1 bar")
    if params.instrument not in ("LONG_CALL", "LONG_PUT"):
        raise ValueError(
            "run_call_backtest requires instrument='LONG_CALL' or "
            f"'LONG_PUT', got {params.instrument!r}"
        )
    bear = params.instrument == "LONG_PUT"

    slip = params.effective_option_slippage_bps() / 10_000.0
    commission = params.commission_per_contract
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    contracts = 0
    leg: OptionLegBars | None = None
    entry_index = -1
    entry_premium = 0.0
    entry_cost = 0.0
    entry_reason = ""
    entry_edge = 0.0
    entry_stop_distance = 0.0
    entry_underlying = 0.0
    peak_close = -math.inf
    trough_close = math.inf  # BEAR trail anchor (mirror of peak_close)
    last_option_price = 0.0  # last REAL traded price, for marking gaps
    # (leg, reason, edge, stop_distance) decided at the close of t-1
    pending_entry: tuple[OptionLegBars, str, float, float] | None = None
    pending_exit: str | None = None

    trades: list[OptionTrade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def close_trade(t: int, exit_premium: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, contracts, leg
        proceeds = contracts * CONTRACT_MULTIPLIER * exit_premium - contracts * exit_commission
        cash += proceeds
        pnl = proceeds - entry_cost
        trades.append(
            OptionTrade(
                entry_index=entry_index,
                entry_date=dates[entry_index],
                entry_price=entry_premium,
                exit_index=t,
                exit_date=dates[t],
                exit_price=exit_premium,
                contracts=contracts,
                bars_held=t - entry_index,
                return_pct=(pnl / entry_cost * 100.0) if entry_cost > 0.0 else 0.0,
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=reason,
                contract_symbol=leg.symbol if leg else "",
                strike=leg.strike if leg else 0.0,
                contract_expiry=leg.expiry if leg else None,
            )
        )
        contracts = 0
        leg = None

    for t in range(n):
        today = dates[t]

        # --- 1. Fill the decision made at the close of t-1 at this open ----
        if pending_entry is not None:
            cand, reason, edge, stop_distance = pending_entry
            bar = cand.bars.get(today)
            if bar is not None and bar[0] > 0.0:
                fill = bar[0] * (1.0 + slip)
                per_contract = fill * CONTRACT_MULTIPLIER + commission
                qty = math.floor(cash * params.option_premium_pct / per_contract)
                while qty > 0 and qty * per_contract > cash:
                    qty -= 1
                if qty > 0:
                    cash -= qty * per_contract
                    contracts = qty
                    leg = cand
                    entry_index = t
                    entry_premium = fill
                    entry_cost = qty * per_contract
                    entry_reason = (
                        f"{reason} -> {params.instrument} {cand.symbol} "
                        f"(strike {cand.strike:g}, exp {cand.expiry.isoformat()})"
                    )
                    entry_edge = edge
                    entry_stop_distance = stop_distance
                    entry_underlying = opens[t]
                    peak_close = -math.inf
                    trough_close = math.inf
                    last_option_price = fill
            # Missing fill bar (contract did not trade at the open of t):
            # the entry is SKIPPED — never filled at an invented price.
            pending_entry = None
        elif pending_exit is not None and contracts > 0 and leg is not None:
            bar = leg.bars.get(today)
            if bar is not None and bar[0] > 0.0:
                close_trade(t, bar[0] * (1.0 - slip), commission, pending_exit)
                pending_exit = None
            elif today >= leg.expiry:
                # No tradable bar left before expiry: contractual settlement
                # at intrinsic off the REAL underlying close (§ arithmetic,
                # not a price guess). No slippage/commission on settlement.
                intrinsic = (
                    max(leg.strike - closes[t], 0.0)
                    if bear
                    else max(closes[t] - leg.strike, 0.0)
                )
                close_trade(t, intrinsic, 0.0, pending_exit + " -> settled at expiry intrinsic")
                pending_exit = None
            # else: keep the pending exit and try the next bar.

        held_during_bar = contracts > 0

        # --- 2. Decide at the close of t --------------------------------
        if contracts > 0 and leg is not None:
            peak_close = max(peak_close, closes[t])
            trough_close = min(trough_close, closes[t])
            bar = leg.bars.get(today)
            if bar is not None and bar[1] > 0.0:
                last_option_price = bar[1]

            if today >= leg.expiry:
                # Held through expiry: settle at intrinsic (see above).
                intrinsic = (
                    max(leg.strike - closes[t], 0.0)
                    if bear
                    else max(closes[t] - leg.strike, 0.0)
                )
                close_trade(
                    t,
                    intrinsic,
                    0.0,
                    (
                        f"EXPIRY_SETTLEMENT: intrinsic max({leg.strike:g} - "
                        f"{closes[t]:.4f}, 0) = {intrinsic:.4f}"
                        if bear
                        else f"EXPIRY_SETTLEMENT: intrinsic max({closes[t]:.4f} - "
                        f"{leg.strike:g}, 0) = {intrinsic:.4f}"
                    ),
                )
            elif t == n - 1:
                close_trade(
                    t,
                    last_option_price,
                    0.0,
                    f"END_OF_DATA: marked to last real option price "
                    f"{last_option_price:.4f}",
                )
            else:
                decision = evaluate_option_exit(
                    PositionState(
                        entry_price=entry_underlying,
                        stop_distance=entry_stop_distance,
                        entry_edge=entry_edge,
                        bars_held=t - entry_index,
                        highest_close_since_entry=peak_close,
                        direction="BEAR" if bear else "BULL",
                        lowest_close_since_entry=trough_close if bear else None,
                    ),
                    OptionState(
                        entry_premium=entry_premium,
                        # None on no-trade days: the premium stop reports
                        # "insufficient data" loudly instead of pretending.
                        current_mid=bar[1] if bar is not None and bar[1] > 0.0 else None,
                        dte=(leg.expiry - today).days,
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
                entry_fn = _evaluate_entry_bear if bear else _evaluate_entry
                entry_eval = entry_fn(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, edge = entry_eval
                    cand = contract_provider(today, closes[t])
                    if cand is not None and cand.expiry > today:
                        pending_entry = (
                            cand,
                            reason,
                            edge,
                            ATR_STOP_MULTIPLE * atr_t,
                        )
                    # No eligible contract / no data -> entry skipped.

        # --- 3. Mark to market -------------------------------------------
        equity.append(cash + contracts * CONTRACT_MULTIPLIER * last_option_price)
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,  # type: ignore[arg-type]  # duck-typed for metrics
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
    )


@dataclass(frozen=True)
class SpreadLegBars:
    """Both legs of one vertical debit spread — SAME expiry, real bars each.

    Geometry by leg (enforced at the entry decision, audit §8 item 6):
    BULL_CALL_SPREAD short.strike > long.strike; BEAR_PUT_SPREAD
    short.strike < long.strike. Either way the LONG leg is the dearer one,
    so net = long − short is a positive debit.
    """

    long: OptionLegBars
    short: OptionLegBars


#: Called at an entry decision (bull or bear): (decision_date, spot_close)
#: -> both legs with their real bars, or None (entry skipped).
SpreadProvider = Callable[[date, float], SpreadLegBars | None]


def run_spread_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="BULL_CALL_SPREAD"),
    *,
    spread_provider: SpreadProvider,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Replay the bull signal as a BULL CALL SPREAD over real contract bars
    — or, for ``instrument="BEAR_PUT_SPREAD"``, the bear signal
    (:func:`_evaluate_entry_bear`, the same gate as LONG_PUT / SHORT_STOCK)
    as a BEAR PUT SPREAD (long the higher put, short the lower put; audit
    §8 item 6 fix 2026-08-17).

    NET-DEBIT semantics throughout (execution-chains roadmap Phase 1): the
    net debit paid at entry IS the max loss, so the §11.3 premium stop and
    §11.7 DTE exit transfer verbatim — the LIVE ``evaluate_option_exit``
    runs on (net entry debit, current net mid, dte). Slippage is ADVERSE ON
    BOTH LEGS both ways (§20.2): buy the long leg dearer + sell the short
    leg cheaper at entry; reversed at exit. A day where EITHER leg did not
    trade has no observable net — entries skip, marks carry the last real
    joint observation, the premium stop reports "insufficient data".
    Held through expiry -> settled at NET intrinsic
    ``max(S-K_long,0) - max(S-K_short,0)`` (bounded [0, width]) off the
    real underlying close — contractual arithmetic, never a price guess.
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError(
            "dates, opens, highs, lows, closes and volumes must have equal "
            f"length, got {len(dates)}/{len(opens)}/{len(highs)}/{len(lows)}/"
            f"{n}/{len(volumes)}"
        )
    if n < 1:
        raise ValueError("run_spread_backtest needs at least 1 bar")
    if params.instrument not in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
        raise ValueError(
            "run_spread_backtest requires instrument='BULL_CALL_SPREAD' or "
            f"'BEAR_PUT_SPREAD', got {params.instrument!r}"
        )
    bear = params.instrument == "BEAR_PUT_SPREAD"

    slip = params.effective_option_slippage_bps() / 10_000.0
    commission = params.commission_per_contract  # PER LEG, both ways
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    contracts = 0
    legs: SpreadLegBars | None = None
    entry_index = -1
    entry_net = 0.0
    entry_cost = 0.0
    entry_reason = ""
    entry_edge = 0.0
    entry_stop_distance = 0.0
    entry_underlying = 0.0
    peak_close = -math.inf
    trough_close = math.inf  # BEAR trail anchor
    last_net = 0.0  # last REAL joint observation, floored at 0 for marking
    pending_entry: tuple[SpreadLegBars, str, float, float] | None = None
    pending_exit: str | None = None

    trades: list[OptionTrade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def joint_bar(d: date) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(long (open, close), short (open, close)) when BOTH legs traded."""
        if legs is None:
            return None
        lb = legs.long.bars.get(d)
        sb = legs.short.bars.get(d)
        if lb is None or sb is None:
            return None
        return lb, sb

    def close_trade(t: int, exit_net: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, contracts, legs
        proceeds = contracts * CONTRACT_MULTIPLIER * exit_net - contracts * exit_commission
        cash += proceeds
        pnl = proceeds - entry_cost
        trades.append(
            OptionTrade(
                entry_index=entry_index,
                entry_date=dates[entry_index],
                entry_price=entry_net,
                exit_index=t,
                exit_date=dates[t],
                exit_price=exit_net,
                contracts=contracts,
                bars_held=t - entry_index,
                return_pct=(pnl / entry_cost * 100.0) if entry_cost > 0.0 else 0.0,
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=reason,
                contract_symbol=legs.long.symbol if legs else "",
                strike=legs.long.strike if legs else 0.0,
                contract_expiry=legs.long.expiry if legs else None,
                short_symbol=legs.short.symbol if legs else "",
                short_strike=legs.short.strike if legs else 0.0,
            )
        )
        contracts = 0
        legs = None

    for t in range(n):
        today = dates[t]

        # --- 1. Fill the decision made at the close of t-1 -----------------
        if pending_entry is not None:
            cand, reason, edge, stop_distance = pending_entry
            lb = cand.long.bars.get(today)
            sb = cand.short.bars.get(today)
            if lb is not None and sb is not None and lb[0] > 0.0 and sb[0] > 0.0:
                # Adverse on both legs: pay up for the long, receive less
                # for the short (§20.2).
                net_fill = lb[0] * (1.0 + slip) - sb[0] * (1.0 - slip)
                width = abs(cand.short.strike - cand.long.strike)
                if 0.0 < net_fill < width:
                    per_spread = net_fill * CONTRACT_MULTIPLIER + 2.0 * commission
                    qty = math.floor(cash * params.option_premium_pct / per_spread)
                    while qty > 0 and qty * per_spread > cash:
                        qty -= 1
                    if qty > 0:
                        cash -= qty * per_spread
                        contracts = qty
                        legs = cand
                        entry_index = t
                        entry_net = net_fill
                        entry_cost = qty * per_spread
                        entry_reason = (
                            f"{reason} -> {params.instrument} long "
                            f"{cand.long.symbol} / short {cand.short.symbol} "
                            f"(width {width:g}, exp {cand.long.expiry.isoformat()})"
                        )
                        entry_edge = edge
                        entry_stop_distance = stop_distance
                        entry_underlying = opens[t]
                        peak_close = -math.inf
                        trough_close = math.inf
                        last_net = net_fill
                # Degenerate fill (net <= 0 or >= width) or missing leg bar:
                # entry skipped — never filled at an invented net.
            pending_entry = None
        elif pending_exit is not None and contracts > 0 and legs is not None:
            jb = joint_bar(today)
            if jb is not None and jb[0][0] > 0.0:
                # Adverse both ways: sell the long cheaper, buy back the
                # short dearer.
                exit_net = jb[0][0] * (1.0 - slip) - jb[1][0] * (1.0 + slip)
                close_trade(t, max(exit_net, 0.0), 2.0 * commission, pending_exit)
                pending_exit = None
            elif today >= legs.long.expiry:
                if bear:
                    intrinsic = max(legs.long.strike - closes[t], 0.0) - max(
                        legs.short.strike - closes[t], 0.0
                    )
                else:
                    intrinsic = max(closes[t] - legs.long.strike, 0.0) - max(
                        closes[t] - legs.short.strike, 0.0
                    )
                close_trade(
                    t,
                    max(intrinsic, 0.0),
                    0.0,
                    pending_exit + " -> settled at expiry net intrinsic",
                )
                pending_exit = None

        held_during_bar = contracts > 0

        # --- 2. Decide at the close of t -----------------------------------
        if contracts > 0 and legs is not None:
            peak_close = max(peak_close, closes[t])
            trough_close = min(trough_close, closes[t])
            jb = joint_bar(today)
            net_close: float | None = None
            if jb is not None:
                net_close = jb[0][1] - jb[1][1]
                # Real joint observation; asynchronous quote noise can dip
                # microscopically below zero — floor for marking only.
                last_net = max(net_close, 0.0)

            if today >= legs.long.expiry:
                if bear:
                    intrinsic = max(legs.long.strike - closes[t], 0.0) - max(
                        legs.short.strike - closes[t], 0.0
                    )
                    detail = (
                        f"max({legs.long.strike:g} - {closes[t]:.4f}, 0) - "
                        f"max({legs.short.strike:g} - {closes[t]:.4f}, 0)"
                    )
                else:
                    intrinsic = max(closes[t] - legs.long.strike, 0.0) - max(
                        closes[t] - legs.short.strike, 0.0
                    )
                    detail = (
                        f"max({closes[t]:.4f} - {legs.long.strike:g}, 0) - "
                        f"max({closes[t]:.4f} - {legs.short.strike:g}, 0)"
                    )
                close_trade(
                    t,
                    max(intrinsic, 0.0),
                    0.0,
                    f"EXPIRY_SETTLEMENT: net intrinsic {detail} = "
                    f"{max(intrinsic, 0.0):.4f}",
                )
            elif t == n - 1:
                close_trade(
                    t,
                    last_net,
                    0.0,
                    f"END_OF_DATA: marked to last real net {last_net:.4f}",
                )
            else:
                decision = evaluate_option_exit(
                    PositionState(
                        entry_price=entry_underlying,
                        stop_distance=entry_stop_distance,
                        entry_edge=entry_edge,
                        bars_held=t - entry_index,
                        highest_close_since_entry=peak_close,
                        direction="BEAR" if bear else "BULL",
                        lowest_close_since_entry=trough_close if bear else None,
                    ),
                    OptionState(
                        entry_premium=entry_net,
                        current_mid=net_close if net_close is not None and net_close > 0.0 else None,
                        dte=(legs.long.expiry - today).days,
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
                # Bear leg uses the SAME bear gate as LONG_PUT / SHORT_STOCK
                # (audit §8 item 6: previously the BULL evaluator ran here,
                # so BEAR_PUT_SPREAD could never enter).
                entry_fn = _evaluate_entry_bear if bear else _evaluate_entry
                entry_eval = entry_fn(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, edge = entry_eval
                    cand = spread_provider(today, closes[t])
                    # Debit-spread geometry: put vertical shorts BELOW the
                    # long strike, call vertical shorts ABOVE it (mirrors the
                    # router's resolver). Wrong-way candidates are refused.
                    geometry_ok = (
                        cand is not None
                        and cand.long.expiry > today
                        and cand.long.expiry == cand.short.expiry
                        and (
                            cand.short.strike < cand.long.strike
                            if bear
                            else cand.short.strike > cand.long.strike
                        )
                    )
                    if geometry_ok:
                        pending_entry = (cand, reason, edge, ATR_STOP_MULTIPLE * atr_t)

        equity.append(cash + contracts * CONTRACT_MULTIPLIER * last_net)
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,  # type: ignore[arg-type]
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
    )


def run_csp_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="CASH_SECURED_PUT"),
    *,
    contract_provider: ContractProvider,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Replay the CASH-SECURED PUT income strategy over real put bars.

    Entry = the SAME bull signal as the stock engine (selling a put is a
    bullish income entry); the put is SOLD at its REAL next-bar open with
    adverse slippage (credit received LOWER); sizing reserves strike×100
    per contract from cash (position_pct caps the reserved fraction).
    Management is the LIVE mechanical standard (evaluate_short_premium_exit:
    50% capture / 2x stop / 21 DTE) on real closes; buybacks fill at real
    opens with adverse slippage (pay MORE). Held to expiry: OTM keeps the
    full credit; ITM settles as CASH P&L = credit − intrinsic — a
    documented CASH-SETTLED ASSIGNMENT APPROXIMATION (no share delivery in
    replay; the wheel arrives later). Honest gaps as everywhere: no bar,
    no fill/mark.
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError("misaligned input series")
    if n < 1:
        raise ValueError("run_csp_backtest needs at least 1 bar")
    if params.instrument != "CASH_SECURED_PUT":
        raise ValueError(
            "run_csp_backtest requires instrument='CASH_SECURED_PUT', got "
            f"{params.instrument!r}"
        )

    slip = params.effective_option_slippage_bps() / 10_000.0
    commission = params.commission_per_contract
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    contracts = 0
    leg: OptionLegBars | None = None
    entry_index = -1
    entry_credit = 0.0
    entry_reason = ""
    last_mid = 0.0
    pending_entry: tuple[OptionLegBars, str] | None = None
    pending_exit: str | None = None

    trades: list[OptionTrade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def close_short(t: int, buyback: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, contracts, leg
        cost = contracts * CONTRACT_MULTIPLIER * buyback + exit_commission
        cash -= cost
        credit_total = contracts * CONTRACT_MULTIPLIER * entry_credit
        pnl = credit_total - cost
        basis = leg.strike * CONTRACT_MULTIPLIER * contracts if leg else 0.0
        trades.append(
            OptionTrade(
                entry_index=entry_index,
                entry_date=dates[entry_index],
                entry_price=entry_credit,
                exit_index=t,
                exit_date=dates[t],
                exit_price=buyback,
                contracts=contracts,
                bars_held=t - entry_index,
                # Return on the CASH SECURED (the capital actually at work).
                return_pct=(pnl / basis * 100.0) if basis > 0 else 0.0,
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=reason,
                contract_symbol=leg.symbol if leg else "",
                strike=leg.strike if leg else 0.0,
                contract_expiry=leg.expiry if leg else None,
            )
        )
        contracts = 0
        leg = None

    for t in range(n):
        today = dates[t]

        if pending_entry is not None:
            cand, reason = pending_entry
            bar = cand.bars.get(today)
            if bar is not None and bar[0] > 0.0:
                credit = bar[0] * (1.0 - slip)  # sold: receive LESS
                reserve_per = cand.strike * CONTRACT_MULTIPLIER
                qty = math.floor(cash * params.position_pct / reserve_per)
                if qty > 0 and credit > 0.0:
                    contracts = qty
                    leg = cand
                    entry_index = t
                    entry_credit = credit
                    cash += qty * CONTRACT_MULTIPLIER * credit - commission * qty
                    entry_reason = (
                        f"{reason} -> CASH_SECURED_PUT short {cand.symbol} "
                        f"(strike {cand.strike:g}, exp {cand.expiry.isoformat()}, "
                        f"reserve {reserve_per * qty:,.0f})"
                    )
                    last_mid = credit
            pending_entry = None
        elif pending_exit is not None and contracts > 0 and leg is not None:
            bar = leg.bars.get(today)
            if bar is not None and bar[0] > 0.0:
                close_short(
                    t, bar[0] * (1.0 + slip), commission * contracts, pending_exit
                )
                pending_exit = None
            elif today >= leg.expiry:
                intrinsic = max(leg.strike - closes[t], 0.0)
                close_short(
                    t, intrinsic, 0.0,
                    pending_exit + " -> cash-settled at expiry intrinsic",
                )
                pending_exit = None

        held_during_bar = contracts > 0

        if contracts > 0 and leg is not None:
            bar = leg.bars.get(today)
            if bar is not None and bar[1] > 0.0:
                last_mid = bar[1]
            if today >= leg.expiry:
                intrinsic = max(leg.strike - closes[t], 0.0)
                close_short(
                    t,
                    intrinsic,
                    0.0,
                    "EXPIRY_SETTLEMENT: cash-settled assignment approximation "
                    f"— intrinsic max({leg.strike:g} - {closes[t]:.4f}, 0) = "
                    f"{intrinsic:.4f}",
                )
            elif t == n - 1:
                close_short(
                    t, last_mid, 0.0,
                    f"END_OF_DATA: marked to last real mid {last_mid:.4f}",
                )
            else:
                decision = evaluate_short_premium_exit(
                    ShortPremiumState(
                        entry_credit=entry_credit,
                        current_mid=(
                            bar[1] if bar is not None and bar[1] > 0.0 else None
                        ),
                        dte=(leg.expiry - today).days,
                        strike=leg.strike,
                        spot=closes[t],
                        right="P",
                    ),
                    exit_params,
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
            if atr14[t] is not None:
                entry_eval = _evaluate_entry(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, _edge = entry_eval
                    cand = contract_provider(today, closes[t])
                    if cand is not None and cand.expiry > today:
                        pending_entry = (cand, reason)

        # Equity: cash (credit already inside) minus the buyback liability.
        equity.append(cash - contracts * CONTRACT_MULTIPLIER * last_mid)
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,  # type: ignore[arg-type]
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
    )


def run_covered_call_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="COVERED_CALL"),
    *,
    contract_provider: ContractProvider,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
) -> BacktestResult:
    """Buy-write replay: the V1 stock leg (same entries, same LIVE exits)
    plus a rolling SHORT CALL overlay on the held shares (Phase 2).

    THE COLLATERAL LAW IN REPLAY FORM: the call is only ever sold against
    shares actually held (contracts = shares // 100); when the STOCK exit
    fires, the call is bought back on the SAME fill bar (atomic unwind —
    replay never strands a naked short). Overlay management is the live
    mechanical standard (50% capture / 2x stop / 21 DTE) on real bars;
    held-to-expiry: OTM expires worthless (credit kept), ITM = shares
    CALLED AWAY at the strike (contractual settlement — the stock leg
    closes at the strike, capped upside is the strategy's real cost).
    Stock fills follow the stock engine's §20.2 model; option fills the
    option slippage proxy. Honest gaps: no call bar, no sell/mark.
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError("misaligned input series")
    if n < 1:
        raise ValueError("run_covered_call_backtest needs at least 1 bar")
    if params.instrument != "COVERED_CALL":
        raise ValueError(
            "run_covered_call_backtest requires instrument='COVERED_CALL', "
            f"got {params.instrument!r}"
        )

    stock_slip = params.effective_slippage_bps() / 10_000.0
    opt_slip = params.effective_option_slippage_bps() / 10_000.0
    stock_comm = params.commission_per_share
    opt_comm = params.commission_per_contract
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
    stock_entry_index = -1
    stock_entry_price = 0.0
    stock_entry_cost = 0.0
    stock_entry_reason = ""
    stock_entry_edge = 0.0
    stock_stop_distance = 0.0
    peak_close = -math.inf

    cc: OptionLegBars | None = None
    cc_contracts = 0
    cc_entry_index = -1
    cc_credit = 0.0
    cc_last_mid = 0.0
    cc_entry_reason = ""

    pending_stock_entry: tuple[str, float, float] | None = None
    pending_stock_exit: str | None = None
    pending_cc_sell = False
    pending_cc_buyback: str | None = None

    trades: list[OptionTrade] = []
    equity: list[float] = []
    held_flags: list[bool] = []

    def record_cc(t: int, buyback: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, cc, cc_contracts
        cost = cc_contracts * CONTRACT_MULTIPLIER * buyback + exit_commission
        cash -= cost
        credit_total = cc_contracts * CONTRACT_MULTIPLIER * cc_credit
        pnl = credit_total - cost
        basis = credit_total if credit_total > 0 else 1.0
        trades.append(
            OptionTrade(
                entry_index=cc_entry_index,
                entry_date=dates[cc_entry_index],
                entry_price=cc_credit,
                exit_index=t,
                exit_date=dates[t],
                exit_price=buyback,
                contracts=cc_contracts,
                bars_held=t - cc_entry_index,
                return_pct=pnl / basis * 100.0,
                pnl=pnl,
                entry_reason=cc_entry_reason,
                exit_reason=reason,
                contract_symbol=cc.symbol if cc else "",
                strike=cc.strike if cc else 0.0,
                contract_expiry=cc.expiry if cc else None,
            )
        )
        cc = None
        cc_contracts = 0

    def record_stock(t: int, fill: float, exit_commission: float, reason: str) -> None:
        nonlocal cash, shares
        proceeds = shares * fill - exit_commission
        cash += proceeds
        pnl = proceeds - stock_entry_cost
        trades.append(
            OptionTrade(
                entry_index=stock_entry_index,
                entry_date=dates[stock_entry_index],
                entry_price=stock_entry_price,
                exit_index=t,
                exit_date=dates[t],
                exit_price=fill,
                contracts=shares,  # SHARES for the stock leg (named in reason)
                bars_held=t - stock_entry_index,
                return_pct=(pnl / stock_entry_cost * 100.0) if stock_entry_cost > 0 else 0.0,
                pnl=pnl,
                entry_reason=stock_entry_reason,
                exit_reason=reason,
            )
        )
        shares = 0

    for t in range(n):
        today = dates[t]

        # --- fills decided at the close of t-1 ------------------------------
        if pending_stock_entry is not None:
            reason, edge, stop_distance = pending_stock_entry
            fill = opens[t] * (1.0 + stock_slip)
            qty = math.floor(cash * params.position_pct / fill) if fill > 0 else 0
            while qty > 0 and qty * (fill + stock_comm) > cash:
                qty -= 1
            if qty > 0:
                cash -= qty * (fill + stock_comm)
                shares = qty
                stock_entry_index = t
                stock_entry_price = fill
                stock_entry_cost = qty * (fill + stock_comm)
                stock_entry_reason = f"{reason} -> LONG_STOCK {qty} shares (buy-write base)"
                stock_entry_edge = edge
                stock_stop_distance = stop_distance
                peak_close = -math.inf
            pending_stock_entry = None
        elif pending_stock_exit is not None and shares > 0:
            # COLLATERAL LAW: unwind the call on the SAME bar, first.
            if cc is not None and cc_contracts > 0:
                bar = cc.bars.get(today)
                buyback = (
                    bar[0] * (1.0 + opt_slip)
                    if bar is not None and bar[0] > 0.0
                    else max(opens[t] - cc.strike, 0.0)  # intrinsic fallback
                )
                record_cc(
                    t, buyback, opt_comm * cc_contracts,
                    "UNWIND: stock exit fired — call bought back on the same "
                    "bar (collateral law; no naked short in replay)",
                )
            record_stock(t, opens[t] * (1.0 - stock_slip), stock_comm * shares, pending_stock_exit)
            pending_stock_exit = None
        if pending_cc_sell and shares >= CONTRACT_MULTIPLIER and cc is None:
            cand = contract_provider(today, opens[t])
            # CHURN GUARD: never sell an overlay already inside the DTE
            # management zone — it would be bought right back by DTE_EXIT,
            # bleeding slippage+commission every cycle.
            if (
                cand is not None
                and cand.expiry > today
                and (cand.expiry - today).days > exit_params.dte_exit_threshold
            ):
                bar = cand.bars.get(today)
                if bar is not None and bar[0] > 0.0:
                    qty = shares // CONTRACT_MULTIPLIER
                    credit = bar[0] * (1.0 - opt_slip)
                    if credit > 0.0:
                        cc = cand
                        cc_contracts = qty
                        cc_entry_index = t
                        cc_credit = credit
                        cc_last_mid = credit
                        cash += qty * CONTRACT_MULTIPLIER * credit - opt_comm * qty
                        cc_entry_reason = (
                            f"COVERED_CALL short {cand.symbol} x{qty} against "
                            f"{shares} held shares (strike {cand.strike:g}, "
                            f"exp {cand.expiry.isoformat()})"
                        )
            pending_cc_sell = False
        elif pending_cc_buyback is not None and cc is not None:
            bar = cc.bars.get(today)
            if bar is not None and bar[0] > 0.0:
                record_cc(t, bar[0] * (1.0 + opt_slip), opt_comm * cc_contracts, pending_cc_buyback)
                pending_cc_buyback = None
            elif today >= cc.expiry:
                intrinsic = max(closes[t] - cc.strike, 0.0)
                record_cc(t, intrinsic, 0.0, pending_cc_buyback + " -> settled at expiry intrinsic")
                pending_cc_buyback = None

        held_during_bar = shares > 0

        # --- decisions at the close of t ------------------------------------
        if shares > 0:
            peak_close = max(peak_close, closes[t])

            # Overlay first: expiry / management on the short call.
            if cc is not None:
                bar = cc.bars.get(today)
                if bar is not None and bar[1] > 0.0:
                    cc_last_mid = bar[1]
                if today >= cc.expiry:
                    if closes[t] > cc.strike:
                        # ASSIGNMENT: shares called away at the strike —
                        # contractual settlement; the credit stays banked.
                        assigned_strike = cc.strike
                        record_cc(
                            t, 0.0, 0.0,
                            "EXPIRY: ITM — assigned (credit kept; shares "
                            "called away at the strike)",
                        )
                        record_stock(
                            t, assigned_strike, 0.0,
                            "ASSIGNED: shares called away at the covered "
                            "call's strike",
                        )
                    else:
                        record_cc(
                            t, 0.0, 0.0,
                            "EXPIRY: OTM — expired worthless, full credit kept",
                        )
                elif t < n - 1:
                    decision = evaluate_short_premium_exit(
                        ShortPremiumState(
                            entry_credit=cc_credit,
                            current_mid=(
                                bar[1] if bar is not None and bar[1] > 0.0 else None
                            ),
                            dte=(cc.expiry - today).days,
                            strike=cc.strike,
                            spot=closes[t],
                            right="C",
                        ),
                        exit_params,
                    )
                    if decision.should_exit:
                        pending_cc_buyback = next(
                            r
                            for r in decision.reasons
                            if r.startswith(decision.triggered_rule or "")
                        )

            # Stock leg: the LIVE exit engine, unchanged (§21).
            if shares > 0 and t == n - 1:
                if cc is not None and cc_contracts > 0:
                    record_cc(
                        t, cc_last_mid, 0.0,
                        f"END_OF_DATA: call marked to last real mid {cc_last_mid:.4f}",
                    )
                record_stock(
                    t, closes[t], 0.0,
                    f"END_OF_DATA: marked to final close {closes[t]:.4f}",
                )
            elif shares > 0:
                decision = evaluate_exit(
                    PositionState(
                        entry_price=stock_entry_price,
                        stop_distance=stock_stop_distance,
                        entry_edge=stock_entry_edge,
                        bars_held=t - stock_entry_index,
                        highest_close_since_entry=peak_close,
                    ),
                    closes[: t + 1],
                    highs[: t + 1],
                    lows[: t + 1],
                    volumes=volumes[: t + 1],
                    params=exit_params,
                    directional_params=directional_params,
                )
                if decision.should_exit:
                    pending_stock_exit = next(
                        r
                        for r in decision.reasons
                        if r.startswith(decision.triggered_rule or "")
                    )
                elif (
                    cc is None
                    and not pending_cc_sell
                    and shares >= CONTRACT_MULTIPLIER
                ):
                    # No call open and the stock stays: sell the overlay.
                    pending_cc_sell = True
        elif pending_stock_entry is None and params.warmup_bars <= t < n - 1:
            atr_t = atr14[t]
            if atr_t is not None:
                entry_eval = _evaluate_entry(
                    t, closes, highs, lows, volumes, params, regime_params, directional_params
                )
                if entry_eval is not None:
                    reason, edge = entry_eval
                    pending_stock_entry = (reason, edge, ATR_STOP_MULTIPLE * atr_t)

        # Equity: cash + shares at close − call buyback liability.
        liability = cc_contracts * CONTRACT_MULTIPLIER * cc_last_mid if cc is not None else 0.0
        equity.append(cash + shares * closes[t] - liability)
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    return BacktestResult(
        trades=trades,  # type: ignore[arg-type]
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),
    )
