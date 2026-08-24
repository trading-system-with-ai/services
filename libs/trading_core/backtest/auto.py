"""AUTO-instrument backtest engine (Phase B of the auto-strategy program,
docs/auto-strategy-portfolio-design.md, user mandate 2026-08-20).

The user picks NO instrument: at every FLAT moment the entry instrument is
chosen by the LIVE §8 matrix from the live signal stack — direction (bias),
strength tier (|edge| banded 25/40/60/80), volatility regime (REAL stored
ATM IV only; unknown coerces to NORMAL exactly like the live chain), and
the account/user permissions. Score bands ARE the user's a/b/c/d model;
the IV condition is the institutional refinement the matrix already
embodies (never buy premium blind).

§21 ONE PIPELINE, strictly: signals via classify_regime/score_direction,
tiering via risk.engine.strength_tier, instrument via
strategies.select_instrument, exits via the SHARED live exit engines —
nothing re-derived here. SWITCHING IS EXIT-MEDIATED by design: a held
position is closed only by the live exit engine (SIGNAL_FLIP / DECAY /
stops already encode direction change); the next flat bar re-enters
whatever §8 then says. No parallel exit math, no churn on tier flicker.

Phase B scope: LONG_STOCK, SHORT_STOCK, LONG_CALL, LONG_PUT. Spreads and
the income overlay (covered calls in the neutral band) arrive in Phase D —
constructing this engine with ``permissions.defined_risk_spreads=True``
raises, loudly, rather than silently degrading a documented capability.

Accounting, fills, slippage, commissions, settlement and END_OF_DATA
semantics are copied VERBATIM from the single-leg engines (engine.py /
options.py) per position kind — one behavior, two entry paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from libs.trading_core.exits import (
    ExitParams,
    OptionState,
    PositionState,
    evaluate_exit,
    evaluate_option_exit,
)
from libs.trading_core.features.indicators import atr, realized_vol
from libs.trading_core.models.enums import DirectionalBias, InstrumentType
from libs.trading_core.risk.engine import (
    ATR_STOP_MULTIPLE,
    RiskLimits,
    strength_tier,
)
from libs.trading_core.signals import (
    DirectionalParams,
    RegimeParams,
    classify_regime,
    score_direction,
)
from libs.trading_core.strategies import AccountPermissions, select_instrument
from libs.trading_core.volatility import VolRegimeParams, classify_vol_regime

from .engine import (
    ATR_PERIOD,
    _BEAR_REGIMES,
    _BULL_REGIMES,
    INITIAL_EQUITY,
    BacktestParams,
    BacktestResult,
    Trade,
    _segment_metrics,
)
from .options import CONTRACT_MULTIPLIER, ContractProvider, OptionLegBars, OptionTrade


@dataclass(frozen=True)
class AutoDecision:
    """One day's §8 verdict while FLAT — the audit trail of AUTO mode."""

    day: date
    edge: float
    tier: str | None
    vol_regime: str | None
    instrument: str
    rationale: str


def run_auto_backtest(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    params: BacktestParams = BacktestParams(instrument="AUTO"),
    *,
    permissions: AccountPermissions,
    call_provider: ContractProvider | None = None,
    put_provider: ContractProvider | None = None,
    iv_series: list[float | None] | None = None,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
    risk_limits: RiskLimits = RiskLimits(),
    vol_params: VolRegimeParams = VolRegimeParams(),
) -> tuple[BacktestResult, list[AutoDecision]]:
    """Replay the full §8 decision stack over daily bars.

    ``iv_series`` is per-bar REAL ATM IV (stored atm_iv_daily rows aligned
    to ``dates``) or ``None`` entries where no real IV exists — an unknown
    vol regime coerces to NORMAL inside ``select_instrument`` with a
    rationale line, byte-identical to the live chain's behavior. NOTHING is
    estimated: before IV history accumulates, AUTO simply behaves like the
    live platform does without chain data (stock over options).

    Entry gating mirrors the single-leg engines exactly: bull entries need
    regime ∈ {STRONG_BULL, MILD_BULL}, bias BULL, edge >= threshold; bear
    entries the mirror. A NEUTRAL bias or an unsupportive regime enters
    nothing (NO TRADE is a valid outcome, §44 rule 18). An option entry
    whose provider returns no §-eligible contract, or whose contract has no
    real bar at the fill open, is SKIPPED — never filled at an invented
    price.

    Returns ``(BacktestResult, decisions)`` — decisions is the per-entry-day
    §8 audit trail (edge, tier, vol regime, instrument, rationale).
    """
    n = len(closes)
    if not (len(dates) == len(opens) == len(highs) == len(lows) == n == len(volumes)):
        raise ValueError(
            "dates, opens, highs, lows, closes and volumes must have equal "
            f"length, got {len(dates)}/{len(opens)}/{len(highs)}/{len(lows)}/"
            f"{n}/{len(volumes)}"
        )
    if n < 1:
        raise ValueError("run_auto_backtest needs at least 1 bar")
    if params.instrument != "AUTO":
        raise ValueError(
            f"run_auto_backtest requires instrument='AUTO', got {params.instrument!r}"
        )
    if iv_series is not None and len(iv_series) != n:
        raise ValueError(
            f"iv_series must align with bars: {len(iv_series)} != {n}"
        )
    if permissions.defined_risk_spreads:
        raise ValueError(
            "AUTO does not support defined_risk_spreads yet (Phase D, "
            "docs/auto-strategy-portfolio-design.md) — disable the spread "
            "permission for AUTO runs"
        )

    stock_slip = params.effective_slippage_bps() / 10_000.0
    option_slip = params.effective_option_slippage_bps() / 10_000.0
    stock_commission = params.commission_per_share
    option_commission = params.commission_per_contract
    atr14 = atr(highs, lows, closes, period=ATR_PERIOD)
    rv = realized_vol(closes)
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    cash = INITIAL_EQUITY
    # Position state: exactly one of these is active at a time.
    kind: str = "NONE"  # NONE | STOCK_LONG | STOCK_SHORT | OPTION
    shares = 0
    contracts = 0
    leg: OptionLegBars | None = None
    option_bear = False
    entry_index = -1
    entry_price = 0.0  # stock fill, or option premium fill
    entry_cost = 0.0  # long-stock / option cost basis
    entry_notional = 0.0  # short-stock notional
    entry_commission_paid = 0.0  # short-stock entry commission
    entry_reason = ""
    entry_edge = 0.0
    entry_stop_distance = 0.0
    entry_underlying = 0.0
    peak_close = -math.inf
    trough_close = math.inf
    last_option_price = 0.0

    # (kind, leg|None, bear, reason, edge, stop_distance)
    pending_entry: tuple[str, OptionLegBars | None, bool, str, float, float] | None = None
    pending_exit: str | None = None

    trades: list[Trade | OptionTrade] = []
    equity: list[float] = []
    held_flags: list[bool] = []
    decisions: list[AutoDecision] = []

    def reset_position() -> None:
        nonlocal kind, shares, contracts, leg
        kind = "NONE"
        shares = 0
        contracts = 0
        leg = None

    def close_stock_long(t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
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
        reset_position()

    def close_stock_short(t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
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
        reset_position()

    def close_option(t: int, exit_premium: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
        proceeds = contracts * CONTRACT_MULTIPLIER * exit_premium - contracts * exit_commission
        cash += proceeds
        pnl = proceeds - entry_cost
        trades.append(
            OptionTrade(
                entry_index=entry_index,
                entry_date=dates[entry_index],
                entry_price=entry_price,
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
        reset_position()

    for t in range(n):
        today = dates[t]

        # --- 1. Fill the decision made at the close of t-1 at this open ----
        if pending_entry is not None:
            p_kind, p_leg, p_bear, p_reason, p_edge, p_stop = pending_entry
            if p_kind == "STOCK_LONG":
                fill = opens[t] * (1.0 + stock_slip)
                qty = math.floor(cash * params.position_pct / fill) if fill > 0.0 else 0
                while qty > 0 and qty * (fill + stock_commission) > cash:
                    qty -= 1
                if qty > 0:
                    cash -= qty * (fill + stock_commission)
                    kind, shares = "STOCK_LONG", qty
                    entry_index, entry_price = t, fill
                    entry_cost = qty * (fill + stock_commission)
                    entry_reason, entry_edge, entry_stop_distance = p_reason, p_edge, p_stop
                    peak_close = -math.inf
            elif p_kind == "STOCK_SHORT":
                fill = opens[t] * (1.0 - stock_slip)  # the short SELL fills lower
                qty = math.floor(cash * params.position_pct / fill) if fill > 0.0 else 0
                if qty > 0:
                    cash += qty * fill - qty * stock_commission
                    kind, shares = "STOCK_SHORT", qty
                    entry_index, entry_price = t, fill
                    entry_notional = qty * fill
                    entry_commission_paid = qty * stock_commission
                    entry_reason, entry_edge, entry_stop_distance = p_reason, p_edge, p_stop
                    trough_close = math.inf
            else:  # OPTION
                assert p_leg is not None
                bar = p_leg.bars.get(today)
                if bar is not None and bar[0] > 0.0:
                    fill = bar[0] * (1.0 + option_slip)
                    per_contract = fill * CONTRACT_MULTIPLIER + option_commission
                    qty = math.floor(cash * params.option_premium_pct / per_contract)
                    while qty > 0 and qty * per_contract > cash:
                        qty -= 1
                    if qty > 0:
                        cash -= qty * per_contract
                        kind, contracts, leg = "OPTION", qty, p_leg
                        option_bear = p_bear
                        entry_index, entry_price = t, fill
                        entry_cost = qty * per_contract
                        entry_reason = (
                            f"{p_reason} -> "
                            f"{'LONG_PUT' if p_bear else 'LONG_CALL'} {p_leg.symbol} "
                            f"(strike {p_leg.strike:g}, exp {p_leg.expiry.isoformat()})"
                        )
                        entry_edge, entry_stop_distance = p_edge, p_stop
                        entry_underlying = opens[t]
                        peak_close = -math.inf
                        trough_close = math.inf
                        last_option_price = fill
                # Missing fill bar: entry SKIPPED, never invented.
            pending_entry = None
        elif pending_exit is not None and kind != "NONE":
            if kind == "STOCK_LONG":
                close_stock_long(t, opens[t] * (1.0 - stock_slip), stock_commission, pending_exit)
                pending_exit = None
            elif kind == "STOCK_SHORT":
                close_stock_short(t, opens[t] * (1.0 + stock_slip), stock_commission, pending_exit)
                pending_exit = None
            else:
                assert leg is not None
                bar = leg.bars.get(today)
                if bar is not None and bar[0] > 0.0:
                    close_option(t, bar[0] * (1.0 - option_slip), option_commission, pending_exit)
                    pending_exit = None
                elif today >= leg.expiry:
                    intrinsic = (
                        max(leg.strike - closes[t], 0.0)
                        if option_bear
                        else max(closes[t] - leg.strike, 0.0)
                    )
                    close_option(t, intrinsic, 0.0, pending_exit + " -> settled at expiry intrinsic")
                    pending_exit = None
                # else: keep the pending exit for the next bar.

        held_during_bar = kind != "NONE"

        # --- 2. Decide at the close of t (fills at the open of t+1) --------
        if kind != "NONE" and pending_exit is not None:
            # LATCH: a decided-but-unfilled exit keeps its rule (same
            # verifier catch as the portfolio engine) — only the option
            # mark advances while the fill retries.
            if kind == "OPTION" and leg is not None:
                bar = leg.bars.get(today)
                if bar is not None and bar[1] > 0.0:
                    last_option_price = bar[1]
                if t == n - 1:
                    close_option(
                        t, last_option_price, 0.0,
                        pending_exit
                        + f" -> END_OF_DATA: marked to last real option "
                        f"price {last_option_price:.4f}",
                    )
                    pending_exit = None
        elif kind == "STOCK_LONG":
            peak_close = max(peak_close, closes[t])
            if t == n - 1:
                close_stock_long(t, closes[t], 0.0, f"END_OF_DATA: marked to final close {closes[t]:.4f}")
            else:
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
                    next(r for r in decision.reasons if r.startswith(decision.triggered_rule or ""))
                    if decision.should_exit
                    else None
                )
        elif kind == "STOCK_SHORT":
            trough_close = min(trough_close, closes[t])
            if t == n - 1:
                close_stock_short(t, closes[t], 0.0, f"END_OF_DATA: marked to final close {closes[t]:.4f}")
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
                    next(r for r in decision.reasons if r.startswith(decision.triggered_rule or ""))
                    if decision.should_exit
                    else None
                )
        elif kind == "OPTION":
            assert leg is not None
            peak_close = max(peak_close, closes[t])
            trough_close = min(trough_close, closes[t])
            bar = leg.bars.get(today)
            if bar is not None and bar[1] > 0.0:
                last_option_price = bar[1]
            if today >= leg.expiry:
                intrinsic = (
                    max(leg.strike - closes[t], 0.0)
                    if option_bear
                    else max(closes[t] - leg.strike, 0.0)
                )
                close_option(
                    t,
                    intrinsic,
                    0.0,
                    (
                        f"EXPIRY_SETTLEMENT: intrinsic max({leg.strike:g} - {closes[t]:.4f}, 0) = {intrinsic:.4f}"
                        if option_bear
                        else f"EXPIRY_SETTLEMENT: intrinsic max({closes[t]:.4f} - {leg.strike:g}, 0) = {intrinsic:.4f}"
                    ),
                )
            elif t == n - 1:
                close_option(
                    t, last_option_price, 0.0,
                    f"END_OF_DATA: marked to last real option price {last_option_price:.4f}",
                )
            else:
                decision = evaluate_option_exit(
                    PositionState(
                        entry_price=entry_underlying,
                        stop_distance=entry_stop_distance,
                        entry_edge=entry_edge,
                        bars_held=t - entry_index,
                        highest_close_since_entry=peak_close,
                        direction="BEAR" if option_bear else "BULL",
                        lowest_close_since_entry=trough_close if option_bear else None,
                    ),
                    OptionState(
                        entry_premium=entry_price,
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
                    next(r for r in decision.reasons if r.startswith(decision.triggered_rule or ""))
                    if decision.should_exit
                    else None
                )
        elif pending_entry is None and params.warmup_bars <= t < n - 1:
            atr_t = atr14[t]
            if atr_t is not None:
                # --- The live decision stack, on [:t+1] slices only --------
                regime = classify_regime(
                    closes[: t + 1], highs[: t + 1], lows[: t + 1], params=regime_params
                )
                sig = score_direction(
                    closes[: t + 1],
                    highs[: t + 1],
                    lows[: t + 1],
                    volumes=volumes[: t + 1],
                    params=directional_params,
                )
                edge = sig.directional_edge
                bull_ok = (
                    regime.classification in _BULL_REGIMES
                    and sig.bias is DirectionalBias.BULL
                    and edge >= params.entry_edge_threshold
                )
                bear_ok = (
                    regime.classification in _BEAR_REGIMES
                    and sig.bias is DirectionalBias.BEAR
                    and edge <= -params.entry_edge_threshold
                )
                if bull_ok or bear_ok:
                    tier = strength_tier(edge, risk_limits)
                    iv_t = iv_series[t] if iv_series is not None else None
                    vol_regime = None
                    if iv_t is not None and iv_t > 0.0:
                        rv_t = rv[t]
                        vol_regime = classify_vol_regime(
                            iv_t,
                            rv_t if rv_t is not None and rv_t > 0.0 else None,
                            vol_params,
                        ).regime
                    verdict = select_instrument(sig.bias, tier, vol_regime, permissions)
                    instrument = verdict.instrument
                    gate_reason = (
                        f"edge {edge:+.1f} (tier {tier}), regime "
                        f"{regime.classification.value}, vol "
                        f"{vol_regime.value if vol_regime is not None else 'UNKNOWN->NORMAL'}"
                    )
                    decisions.append(
                        AutoDecision(
                            day=today,
                            edge=edge,
                            tier=tier,
                            vol_regime=vol_regime.value if vol_regime is not None else None,
                            instrument=instrument.value,
                            rationale="; ".join(verdict.rationale),
                        )
                    )
                    auto_reason = f"AUTO[{gate_reason}]"
                    if instrument is InstrumentType.LONG_STOCK:
                        pending_entry = ("STOCK_LONG", None, False, auto_reason, edge, ATR_STOP_MULTIPLE * atr_t)
                    elif instrument is InstrumentType.SHORT_STOCK:
                        pending_entry = ("STOCK_SHORT", None, True, auto_reason, edge, ATR_STOP_MULTIPLE * atr_t)
                    elif instrument is InstrumentType.LONG_CALL and call_provider is not None:
                        cand = call_provider(today, closes[t])
                        if cand is not None and cand.expiry > today:
                            pending_entry = ("OPTION", cand, False, auto_reason, edge, ATR_STOP_MULTIPLE * atr_t)
                    elif instrument is InstrumentType.LONG_PUT and put_provider is not None:
                        cand = put_provider(today, closes[t])
                        if cand is not None and cand.expiry > today:
                            pending_entry = ("OPTION", cand, True, auto_reason, edge, ATR_STOP_MULTIPLE * atr_t)
                    # NO_TRADE / unavailable provider: nothing entered.

        # --- 3. Mark to market -------------------------------------------
        if kind == "STOCK_LONG":
            equity.append(cash + shares * closes[t])
        elif kind == "STOCK_SHORT":
            equity.append(cash - shares * closes[t])
        elif kind == "OPTION":
            equity.append(cash + contracts * CONTRACT_MULTIPLIER * last_option_price)
        else:
            equity.append(cash)
        held_flags.append(held_during_bar)

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    result = BacktestResult(
        trades=trades,  # type: ignore[arg-type]  # duck-typed for metrics
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, trades),  # type: ignore[arg-type]
    )
    return result, decisions
