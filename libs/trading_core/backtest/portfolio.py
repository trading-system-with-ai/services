"""Portfolio-level AUTO backtest (Phase C of the auto-strategy program,
docs/auto-strategy-portfolio-design.md, user mandate 2026-08-20).

Replays the WHOLE watchlist (or any subset) against ONE shared cash
ledger. Every day, every FLAT symbol runs the same §8 decision stack the
single-symbol AUTO engine runs (direction × strength tier × vol regime ×
permissions — libs/trading_core/backtest/auto.py documents the semantics);
every HELD symbol is managed by the SHARED live exit engines. What the
portfolio layer adds is exactly capital:

- SIZING IS THE LIVE §12 RULE (§21): a stock entry risks
  ``tier_budget × prior-bar equity`` over its ATR stop distance (the
  PREVIOUS bar's close mark — same-morning marks would be look-ahead) —
  ``shares = floor(budget_pct(tier) * equity / stop_distance)`` with the
  same 0.5/0.75/1.0/1.25% tier budgets the live risk engine uses —
  capped by ``position_pct × equity`` notional and by free cash. Option
  entries spend ``option_premium_pct × equity``, capped by free cash
  (the single-leg engines' rule, applied to portfolio equity).
- CONTENTION IS |EDGE| PRIORITY: same-day entry candidates fill in
  descending |edge| order until cash, the gross budget or
  ``max_positions`` runs out — deterministic. NOTE: ``decisions`` records
  SELECTION INTENT (what §8 chose), not fills — a candidate crowded out
  by capital appears as a decision with no matching trade.
- ``cash_floor_pct`` of equity is never deployed (0 = fully investable),
  and TOTAL |notional| is capped at ``max_gross_pct`` of prior equity —
  the gross budget is what actually bounds chained shorts, whose
  proceeds inflate cash and would defeat any cash-only floor.

Daily output is the user's ask verbatim: per-symbol allocation percent
(signed — a short contributes negatively) plus the cash percent, for
every bar. Bars are aligned on the INTERSECTION of the symbols' trading
dates (documented; a symbol missing a shared date would otherwise need
invented marks, §44 rule 18).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    _tier_budget,
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

from .auto import AutoDecision
from .engine import (
    ATR_PERIOD,
    _BEAR_REGIMES,
    _BULL_REGIMES,
    INITIAL_EQUITY,
    BacktestParams,
    SegmentMetrics,
    Trade,
    _segment_metrics,
)
from .options import CONTRACT_MULTIPLIER, ContractProvider, OptionLegBars, OptionTrade


@dataclass(frozen=True)
class SymbolBars:
    """One symbol's aligned daily series (same order as the shared dates)."""

    ticker: str
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]


@dataclass(frozen=True)
class RebalanceEvent:
    """One capital event in the portfolio replay — the explainability the
    user mandated 2026-08-20: WHEN the book changed and WHY it was sized
    that way. ``action`` ∈ ENTER | EXIT | SKIP. ENTER events carry the
    full sizing arithmetic in ``sizing`` (tier budget × prior equity ÷
    stop, then each cap that trimmed it, with real numbers); EXIT events
    carry the shared exit engine's rule verbatim; SKIP events name the
    capital constraint that crowded a selected candidate out — the
    difference between "the matrix said no" and "capital said no"."""

    day: date
    ticker: str
    action: str
    instrument: str
    quantity: int
    price: float | None
    reason: str
    sizing: str
    cash_after: float
    equity_prev: float


@dataclass
class PortfolioBacktestResult:
    """Full portfolio replay output (docs/auto-strategy-portfolio-design.md §C).

    ``allocations[t]`` maps ticker -> signed percent of that bar's equity
    (short positions negative); ``cash_pct[t]`` completes the picture —
    cash_pct + Σ allocations ≈ 100 by construction. ``trades`` and
    ``decisions`` carry the owning ticker.
    """

    dates: list[date]
    equity: list[float]
    drawdown: list[float]
    metrics: SegmentMetrics
    allocations: list[dict[str, float]]
    cash_pct: list[float]
    trades: list[tuple[str, Trade | OptionTrade]]
    decisions: list[tuple[str, AutoDecision]]
    journal: list[RebalanceEvent]


@dataclass
class _Slot:
    """One symbol's position state inside the shared ledger."""

    bars: SymbolBars
    atr14: list[float | None]
    rv: list[float | None]
    iv_series: list[float | None] | None
    call_provider: ContractProvider | None
    put_provider: ContractProvider | None

    kind: str = "NONE"  # NONE | STOCK_LONG | STOCK_SHORT | OPTION
    shares: int = 0
    contracts: int = 0
    leg: OptionLegBars | None = None
    option_bear: bool = False
    entry_index: int = -1
    entry_price: float = 0.0
    entry_cost: float = 0.0
    entry_notional: float = 0.0
    entry_commission_paid: float = 0.0
    entry_reason: str = ""
    entry_edge: float = 0.0
    entry_stop_distance: float = 0.0
    entry_underlying: float = 0.0
    peak_close: float = -math.inf
    trough_close: float = math.inf
    last_option_price: float = 0.0
    # (kind, leg|None, bear, reason, edge, stop_distance, tier)
    pending_entry: tuple | None = None
    pending_exit: str | None = None

    def value(self, t: int) -> float:
        """This slot's signed mark-to-market contribution to equity."""
        if self.kind == "STOCK_LONG":
            return self.shares * self.bars.closes[t]
        if self.kind == "STOCK_SHORT":
            return -self.shares * self.bars.closes[t]
        if self.kind == "OPTION":
            return self.contracts * CONTRACT_MULTIPLIER * self.last_option_price
        return 0.0

    def reset(self) -> None:
        self.kind = "NONE"
        self.shares = 0
        self.contracts = 0
        self.leg = None


def run_portfolio_backtest(
    dates: list[date],
    symbols: list[SymbolBars],
    params: BacktestParams = BacktestParams(instrument="AUTO"),
    *,
    permissions: AccountPermissions,
    iv_series_by_ticker: dict[str, list[float | None]] | None = None,
    call_providers: dict[str, ContractProvider] | None = None,
    put_providers: dict[str, ContractProvider] | None = None,
    cash_floor_pct: float = 0.0,
    max_positions: int | None = None,
    max_gross_pct: float = 1.0,
    regime_params: RegimeParams = RegimeParams(),
    directional_params: DirectionalParams = DirectionalParams(),
    risk_limits: RiskLimits = RiskLimits(),
    vol_params: VolRegimeParams = VolRegimeParams(),
) -> PortfolioBacktestResult:
    """Replay the §8 decision stack for every symbol against one cash ledger.

    ``dates`` is the SHARED (intersected) trading calendar; every
    ``SymbolBars`` must align to it exactly. See the module docstring for
    sizing, contention and allocation semantics.
    """
    n = len(dates)
    if n < 1:
        raise ValueError("run_portfolio_backtest needs at least 1 bar")
    if not symbols:
        raise ValueError("run_portfolio_backtest needs at least 1 symbol")
    for sb in symbols:
        for name, series in (
            ("opens", sb.opens), ("highs", sb.highs), ("lows", sb.lows),
            ("closes", sb.closes), ("volumes", sb.volumes),
        ):
            if len(series) != n:
                raise ValueError(
                    f"{sb.ticker}.{name} must align with dates: {len(series)} != {n}"
                )
    if params.instrument != "AUTO":
        raise ValueError(
            f"run_portfolio_backtest requires instrument='AUTO', got {params.instrument!r}"
        )
    if permissions.defined_risk_spreads:
        raise ValueError(
            "the portfolio backtest does not support defined_risk_spreads yet "
            "(Phase D, docs/auto-strategy-portfolio-design.md)"
        )
    if not (0.0 <= cash_floor_pct < 1.0):
        raise ValueError(f"cash_floor_pct must be in [0, 1), got {cash_floor_pct!r}")
    if max_positions is not None and max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions!r}")
    if max_gross_pct <= 0.0:
        raise ValueError(f"max_gross_pct must be > 0, got {max_gross_pct!r}")

    stock_slip = params.effective_slippage_bps() / 10_000.0
    option_slip = params.effective_option_slippage_bps() / 10_000.0
    stock_commission = params.commission_per_share
    option_commission = params.commission_per_contract
    exit_params = ExitParams(
        exit_edge_threshold=params.exit_edge_threshold,
        atr_trail_k=params.atr_trail_k,
        time_stop_bars=params.time_stop_bars,
        min_move_atr=params.min_move_atr,
        atr_period=ATR_PERIOD,
    )

    iv_map = iv_series_by_ticker or {}
    calls = call_providers or {}
    puts = put_providers or {}
    slots: list[_Slot] = [
        _Slot(
            bars=sb,
            atr14=atr(sb.highs, sb.lows, sb.closes, period=ATR_PERIOD),
            rv=realized_vol(sb.closes),
            iv_series=iv_map.get(sb.ticker),
            call_provider=calls.get(sb.ticker),
            put_provider=puts.get(sb.ticker),
        )
        for sb in symbols
    ]

    cash = INITIAL_EQUITY
    equity_prev = INITIAL_EQUITY
    trades: list[tuple[str, Trade | OptionTrade]] = []
    decisions: list[tuple[str, AutoDecision]] = []
    journal: list[RebalanceEvent] = []
    equity: list[float] = []
    held_flags: list[bool] = []
    allocations: list[dict[str, float]] = []
    cash_pct: list[float] = []

    def record_stock_long_close(s: _Slot, t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
        proceeds = s.shares * exit_price - s.shares * exit_commission
        cash += proceeds
        pnl = proceeds - s.entry_cost
        trades.append((s.bars.ticker, Trade(
            entry_index=s.entry_index, entry_date=dates[s.entry_index],
            entry_price=s.entry_price, exit_index=t, exit_date=dates[t],
            exit_price=exit_price, shares=s.shares, bars_held=t - s.entry_index,
            return_pct=(pnl / s.entry_cost * 100.0) if s.entry_cost > 0.0 else 0.0,
            pnl=pnl, entry_reason=s.entry_reason, exit_reason=reason,
        )))
        journal.append(RebalanceEvent(
            day=dates[t], ticker=s.bars.ticker, action="EXIT",
            instrument="LONG_STOCK", quantity=s.shares, price=exit_price,
            reason=reason, sizing="", cash_after=cash, equity_prev=equity_prev,
        ))
        s.reset()

    def record_stock_short_close(s: _Slot, t: int, exit_price: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
        cost = s.shares * exit_price + s.shares * exit_commission
        cash -= cost
        pnl = (
            s.shares * (s.entry_price - exit_price)
            - s.entry_commission_paid
            - s.shares * exit_commission
        )
        trades.append((s.bars.ticker, Trade(
            entry_index=s.entry_index, entry_date=dates[s.entry_index],
            entry_price=s.entry_price, exit_index=t, exit_date=dates[t],
            exit_price=exit_price, shares=s.shares, bars_held=t - s.entry_index,
            return_pct=(pnl / s.entry_notional * 100.0) if s.entry_notional > 0.0 else 0.0,
            pnl=pnl, entry_reason=s.entry_reason, exit_reason=reason,
        )))
        journal.append(RebalanceEvent(
            day=dates[t], ticker=s.bars.ticker, action="EXIT",
            instrument="SHORT_STOCK", quantity=s.shares, price=exit_price,
            reason=reason, sizing="", cash_after=cash, equity_prev=equity_prev,
        ))
        s.reset()

    def record_option_close(s: _Slot, t: int, exit_premium: float, exit_commission: float, reason: str) -> None:
        nonlocal cash
        proceeds = s.contracts * CONTRACT_MULTIPLIER * exit_premium - s.contracts * exit_commission
        cash += proceeds
        pnl = proceeds - s.entry_cost
        trades.append((s.bars.ticker, OptionTrade(
            entry_index=s.entry_index, entry_date=dates[s.entry_index],
            entry_price=s.entry_price, exit_index=t, exit_date=dates[t],
            exit_price=exit_premium, contracts=s.contracts, bars_held=t - s.entry_index,
            return_pct=(pnl / s.entry_cost * 100.0) if s.entry_cost > 0.0 else 0.0,
            pnl=pnl, entry_reason=s.entry_reason, exit_reason=reason,
            contract_symbol=s.leg.symbol if s.leg else "",
            strike=s.leg.strike if s.leg else 0.0,
            contract_expiry=s.leg.expiry if s.leg else None,
        )))
        journal.append(RebalanceEvent(
            day=dates[t], ticker=s.bars.ticker, action="EXIT",
            instrument="LONG_PUT" if s.option_bear else "LONG_CALL",
            quantity=s.contracts, price=exit_premium,
            reason=reason, sizing="", cash_after=cash, equity_prev=equity_prev,
        ))
        s.reset()

    for t in range(n):
        today = dates[t]

        # --- 1a. Fill EXITS first: they free the cash entries compete for --
        for s in slots:
            if s.pending_exit is None or s.kind == "NONE":
                continue
            if s.kind == "STOCK_LONG":
                record_stock_long_close(s, t, s.bars.opens[t] * (1.0 - stock_slip), stock_commission, s.pending_exit)
                s.pending_exit = None
            elif s.kind == "STOCK_SHORT":
                record_stock_short_close(s, t, s.bars.opens[t] * (1.0 + stock_slip), stock_commission, s.pending_exit)
                s.pending_exit = None
            else:
                assert s.leg is not None
                bar = s.leg.bars.get(today)
                if bar is not None and bar[0] > 0.0:
                    record_option_close(s, t, bar[0] * (1.0 - option_slip), option_commission, s.pending_exit)
                    s.pending_exit = None
                elif today >= s.leg.expiry:
                    intrinsic = (
                        max(s.leg.strike - s.bars.closes[t], 0.0)
                        if s.option_bear
                        else max(s.bars.closes[t] - s.leg.strike, 0.0)
                    )
                    record_option_close(s, t, intrinsic, 0.0, s.pending_exit + " -> settled at expiry intrinsic")
                    s.pending_exit = None
                else:
                    # exit decided but unfillable today — journal the stall
                    # (verifier catch: a frozen book must not be silent).
                    journal.append(RebalanceEvent(
                        day=today, ticker=s.bars.ticker, action="SKIP",
                        instrument="LONG_PUT" if s.option_bear else "LONG_CALL",
                        quantity=0, price=None,
                        reason=(
                            f"exit pending: no real option bar for "
                            f"{s.leg.symbol} at the fill open — retry next bar "
                            f"(rule: {s.pending_exit})"
                        ),
                        sizing="", cash_after=cash, equity_prev=equity_prev,
                    ))

        # --- 1b. Fill ENTRIES by |edge| priority under the shared budget ---
        floor_cash = cash_floor_pct * equity_prev
        # GROSS-EXPOSURE BUDGET (verifier catch 2026-08-20): a short fill
        # CREDITS cash, so a cash-based cap alone lets chained shorts
        # compound the book to multiples of equity. Every entry is charged
        # against |notional| under max_gross_pct of prior equity instead.
        gross = sum(abs(x.value(t)) for x in slots if x.kind != "NONE")
        gross_budget = max_gross_pct * equity_prev
        candidates = sorted(
            (s for s in slots if s.pending_entry is not None),
            key=lambda s: -abs(s.pending_entry[4]),
        )
        def skip(s_, instr_, why_):
            journal.append(RebalanceEvent(
                day=today, ticker=s_.bars.ticker, action="SKIP",
                instrument=instr_, quantity=0, price=None,
                reason=why_, sizing="", cash_after=cash, equity_prev=equity_prev,
            ))

        for s in candidates:
            p_kind, p_leg, p_bear, p_reason, p_edge, p_stop, p_tier = s.pending_entry
            s.pending_entry = None
            instr_name = (
                "LONG_STOCK" if p_kind == "STOCK_LONG"
                else "SHORT_STOCK" if p_kind == "STOCK_SHORT"
                else ("LONG_PUT" if p_bear else "LONG_CALL")
            )
            open_positions = sum(1 for x in slots if x.kind != "NONE")
            if max_positions is not None and open_positions >= max_positions:
                skip(s, instr_name, f"contention: max_positions {max_positions} reached")
                continue
            investable = cash - floor_cash
            if investable <= 0.0:
                skip(s, instr_name, (
                    f"contention: investable ${investable:,.0f} <= 0 "
                    f"(cash ${cash:,.0f} - floor ${floor_cash:,.0f})"
                ))
                continue
            if p_kind in ("STOCK_LONG", "STOCK_SHORT"):
                is_short = p_kind == "STOCK_SHORT"
                fill = s.bars.opens[t] * (1.0 - stock_slip if is_short else 1.0 + stock_slip)
                if fill <= 0.0 or p_stop <= 0.0:
                    skip(s, instr_name, "unfillable: non-positive fill price or stop distance")
                    continue
                # LIVE §12 sizing: tier budget of prior-bar equity over the stop.
                budget_pct = _tier_budget(p_tier, risk_limits)
                budget = budget_pct * equity_prev
                risk_qty = math.floor(budget / p_stop)
                qty = risk_qty
                notional_cap = math.floor(params.position_pct * equity_prev / fill)
                qty = min(qty, notional_cap)
                gross_room = gross_budget - gross
                if gross_room <= 0.0:
                    skip(s, instr_name, (
                        f"contention: gross budget exhausted "
                        f"(gross ${gross:,.0f} >= {max_gross_pct:.0%} × ${equity_prev:,.0f})"
                    ))
                    continue
                gross_cap = math.floor(gross_room / fill)
                qty = min(qty, gross_cap)
                if not is_short:
                    while qty > 0 and qty * (fill + stock_commission) > investable:
                        qty -= 1
                else:
                    # shorting posts no cash but is capped by the same notional
                    # discipline: never short more notional than investable.
                    while qty > 0 and qty * fill > investable:
                        qty -= 1
                if qty <= 0:
                    skip(s, instr_name, (
                        f"contention: sized to 0 shares (risk qty {risk_qty}, "
                        f"caps: position_pct {notional_cap}, gross {gross_cap}, "
                        f"investable ${investable:,.0f} @ ${fill:,.2f})"
                    ))
                    continue
                sizing_math = (
                    f"tier {p_tier} budget {budget_pct:.2%} × equity "
                    f"${equity_prev:,.0f} = ${budget:,.0f} risk ÷ stop "
                    f"${p_stop:,.2f} = {risk_qty} sh; caps: position_pct→"
                    f"{min(risk_qty, notional_cap)}, gross→"
                    f"{min(risk_qty, notional_cap, gross_cap)}, cash→{qty} "
                    f"@ ${fill:,.2f}"
                )
                gross += qty * fill
                if not is_short:
                    cash -= qty * (fill + stock_commission)
                    s.kind = "STOCK_LONG"
                else:
                    cash += qty * fill - qty * stock_commission
                    s.kind = "STOCK_SHORT"
                    s.entry_notional = qty * fill
                    s.entry_commission_paid = qty * stock_commission
                s.shares = qty
                s.entry_index, s.entry_price = t, fill
                if not is_short:
                    s.entry_cost = qty * (fill + stock_commission)
                s.entry_reason, s.entry_edge, s.entry_stop_distance = p_reason, p_edge, p_stop
                s.peak_close = -math.inf
                s.trough_close = math.inf
                journal.append(RebalanceEvent(
                    day=today, ticker=s.bars.ticker, action="ENTER",
                    instrument=instr_name, quantity=qty, price=fill,
                    reason=p_reason, sizing=sizing_math,
                    cash_after=cash, equity_prev=equity_prev,
                ))
            else:  # OPTION
                assert p_leg is not None
                bar = p_leg.bars.get(today)
                if bar is None or bar[0] <= 0.0:
                    skip(s, instr_name, (
                        f"no real option bar for {p_leg.symbol} at the fill open "
                        "— entry skipped, never invented"
                    ))
                    continue
                fill = bar[0] * (1.0 + option_slip)
                per_contract = fill * CONTRACT_MULTIPLIER + option_commission
                gross_room = gross_budget - gross
                if gross_room <= 0.0:
                    skip(s, instr_name, (
                        f"contention: gross budget exhausted "
                        f"(gross ${gross:,.0f} >= {max_gross_pct:.0%} × ${equity_prev:,.0f})"
                    ))
                    continue
                budget = min(params.option_premium_pct * equity_prev, investable, gross_room)
                qty = math.floor(budget / per_contract)
                while qty > 0 and qty * per_contract > investable:
                    qty -= 1
                if qty <= 0:
                    skip(s, instr_name, (
                        f"contention: premium budget ${budget:,.0f} buys 0 contracts "
                        f"@ ${per_contract:,.0f}/contract"
                    ))
                    continue
                sizing_math = (
                    f"premium budget min({params.option_premium_pct:.0%} × "
                    f"${equity_prev:,.0f}, investable ${investable:,.0f}, gross room "
                    f"${gross_room:,.0f}) = ${budget:,.0f} ÷ ${per_contract:,.0f}"
                    f"/contract = {qty} contracts"
                )
                cash -= qty * per_contract
                gross += qty * fill * CONTRACT_MULTIPLIER
                s.kind, s.contracts, s.leg = "OPTION", qty, p_leg
                s.option_bear = p_bear
                s.entry_index, s.entry_price = t, fill
                s.entry_cost = qty * per_contract
                s.entry_reason = (
                    f"{p_reason} -> {'LONG_PUT' if p_bear else 'LONG_CALL'} "
                    f"{p_leg.symbol} (strike {p_leg.strike:g}, exp {p_leg.expiry.isoformat()})"
                )
                s.entry_edge, s.entry_stop_distance = p_edge, p_stop
                s.entry_underlying = s.bars.opens[t]
                s.peak_close = -math.inf
                s.trough_close = math.inf
                s.last_option_price = fill
                journal.append(RebalanceEvent(
                    day=today, ticker=s.bars.ticker, action="ENTER",
                    instrument=instr_name, quantity=qty, price=fill,
                    reason=s.entry_reason, sizing=sizing_math,
                    cash_after=cash, equity_prev=equity_prev,
                ))

        any_held = any(s.kind != "NONE" for s in slots)

        # --- 2. Decide at the close of t, per symbol -----------------------
        for s in slots:
            b = s.bars
            if s.kind != "NONE" and s.pending_exit is not None:
                # LATCH (verifier catch): an exit already decided but not yet
                # filled keeps ITS rule — re-evaluating daily would overwrite
                # the recorded reason with a later bar's text, or silently
                # cancel a decided exit. The peaks still advance at fill/
                # settlement paths; expiry settlement below still applies.
                if s.kind == "OPTION" and s.leg is not None:
                    bar = s.leg.bars.get(today)
                    if bar is not None and bar[1] > 0.0:
                        s.last_option_price = bar[1]
                    if t == n - 1:
                        # final bar: a stalled exit still marks out honestly
                        record_option_close(
                            s, t, s.last_option_price, 0.0,
                            s.pending_exit
                            + f" -> END_OF_DATA: marked to last real option "
                            f"price {s.last_option_price:.4f}",
                        )
                        s.pending_exit = None
                continue
            if s.kind == "STOCK_LONG":
                s.peak_close = max(s.peak_close, b.closes[t])
                if t == n - 1:
                    record_stock_long_close(s, t, b.closes[t], 0.0, f"END_OF_DATA: marked to final close {b.closes[t]:.4f}")
                else:
                    d = evaluate_exit(
                        PositionState(
                            entry_price=s.entry_price, stop_distance=s.entry_stop_distance,
                            entry_edge=s.entry_edge, bars_held=t - s.entry_index,
                            highest_close_since_entry=s.peak_close,
                        ),
                        b.closes[: t + 1], b.highs[: t + 1], b.lows[: t + 1],
                        volumes=b.volumes[: t + 1], params=exit_params,
                        directional_params=directional_params,
                    )
                    s.pending_exit = (
                        next(r for r in d.reasons if r.startswith(d.triggered_rule or ""))
                        if d.should_exit else None
                    )
            elif s.kind == "STOCK_SHORT":
                s.trough_close = min(s.trough_close, b.closes[t])
                if t == n - 1:
                    record_stock_short_close(s, t, b.closes[t], 0.0, f"END_OF_DATA: marked to final close {b.closes[t]:.4f}")
                else:
                    d = evaluate_exit(
                        PositionState(
                            entry_price=s.entry_price, stop_distance=s.entry_stop_distance,
                            entry_edge=s.entry_edge, bars_held=t - s.entry_index,
                            highest_close_since_entry=max(b.closes[s.entry_index : t + 1]),
                            direction="BEAR", lowest_close_since_entry=s.trough_close,
                        ),
                        b.closes[: t + 1], b.highs[: t + 1], b.lows[: t + 1],
                        volumes=b.volumes[: t + 1], params=exit_params,
                        directional_params=directional_params,
                    )
                    s.pending_exit = (
                        next(r for r in d.reasons if r.startswith(d.triggered_rule or ""))
                        if d.should_exit else None
                    )
            elif s.kind == "OPTION":
                assert s.leg is not None
                s.peak_close = max(s.peak_close, b.closes[t])
                s.trough_close = min(s.trough_close, b.closes[t])
                bar = s.leg.bars.get(today)
                if bar is not None and bar[1] > 0.0:
                    s.last_option_price = bar[1]
                if today >= s.leg.expiry:
                    intrinsic = (
                        max(s.leg.strike - b.closes[t], 0.0)
                        if s.option_bear
                        else max(b.closes[t] - s.leg.strike, 0.0)
                    )
                    record_option_close(
                        s, t, intrinsic, 0.0,
                        f"EXPIRY_SETTLEMENT: intrinsic = {intrinsic:.4f}",
                    )
                elif t == n - 1:
                    record_option_close(
                        s, t, s.last_option_price, 0.0,
                        f"END_OF_DATA: marked to last real option price {s.last_option_price:.4f}",
                    )
                else:
                    d = evaluate_option_exit(
                        PositionState(
                            entry_price=s.entry_underlying, stop_distance=s.entry_stop_distance,
                            entry_edge=s.entry_edge, bars_held=t - s.entry_index,
                            highest_close_since_entry=s.peak_close,
                            direction="BEAR" if s.option_bear else "BULL",
                            lowest_close_since_entry=s.trough_close if s.option_bear else None,
                        ),
                        OptionState(
                            entry_premium=s.entry_price,
                            current_mid=bar[1] if bar is not None and bar[1] > 0.0 else None,
                            dte=(s.leg.expiry - today).days,
                        ),
                        b.closes[: t + 1], b.highs[: t + 1], b.lows[: t + 1],
                        volumes=b.volumes[: t + 1], params=exit_params,
                        directional_params=directional_params,
                    )
                    s.pending_exit = (
                        next(r for r in d.reasons if r.startswith(d.triggered_rule or ""))
                        if d.should_exit else None
                    )
            elif s.pending_entry is None and params.warmup_bars <= t < n - 1:
                atr_t = s.atr14[t]
                if atr_t is None:
                    continue
                regime = classify_regime(
                    b.closes[: t + 1], b.highs[: t + 1], b.lows[: t + 1], params=regime_params
                )
                sig = score_direction(
                    b.closes[: t + 1], b.highs[: t + 1], b.lows[: t + 1],
                    volumes=b.volumes[: t + 1], params=directional_params,
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
                if not (bull_ok or bear_ok):
                    continue
                tier = strength_tier(edge, risk_limits)
                if tier is None:
                    continue
                iv_t = s.iv_series[t] if s.iv_series is not None else None
                vol_regime = None
                if iv_t is not None and iv_t > 0.0:
                    rv_t = s.rv[t]
                    vol_regime = classify_vol_regime(
                        iv_t, rv_t if rv_t is not None and rv_t > 0.0 else None, vol_params
                    ).regime
                verdict = select_instrument(sig.bias, tier, vol_regime, permissions)
                instrument = verdict.instrument
                decisions.append((b.ticker, AutoDecision(
                    day=today, edge=edge, tier=tier,
                    vol_regime=vol_regime.value if vol_regime is not None else None,
                    instrument=instrument.value, rationale="; ".join(verdict.rationale),
                )))
                auto_reason = (
                    f"AUTO[edge {edge:+.1f} (tier {tier}), regime "
                    f"{regime.classification.value}, vol "
                    f"{vol_regime.value if vol_regime is not None else 'UNKNOWN->NORMAL'}]"
                )
                stop = ATR_STOP_MULTIPLE * atr_t
                if instrument is InstrumentType.LONG_STOCK:
                    s.pending_entry = ("STOCK_LONG", None, False, auto_reason, edge, stop, tier)
                elif instrument is InstrumentType.SHORT_STOCK:
                    s.pending_entry = ("STOCK_SHORT", None, True, auto_reason, edge, stop, tier)
                elif instrument is InstrumentType.LONG_CALL and s.call_provider is not None:
                    cand = s.call_provider(today, b.closes[t])
                    if cand is not None and cand.expiry > today:
                        s.pending_entry = ("OPTION", cand, False, auto_reason, edge, stop, tier)
                elif instrument is InstrumentType.LONG_PUT and s.put_provider is not None:
                    cand = s.put_provider(today, b.closes[t])
                    if cand is not None and cand.expiry > today:
                        s.pending_entry = ("OPTION", cand, True, auto_reason, edge, stop, tier)

        # --- 3. Mark to market + allocation table --------------------------
        eq = cash + sum(s.value(t) for s in slots)
        equity.append(eq)
        held_flags.append(any_held)
        if eq > 0.0:
            allocations.append({
                s.bars.ticker: s.value(t) / eq * 100.0 for s in slots if s.kind != "NONE"
            })
            cash_pct.append(cash / eq * 100.0)
        else:
            allocations.append({})
            cash_pct.append(0.0)
        equity_prev = eq

    drawdown: list[float] = []
    running_max = -math.inf
    for value in equity:
        running_max = max(running_max, value)
        drawdown.append(value / running_max - 1.0 if running_max > 0.0 else 0.0)

    plain_trades = [tr for _, tr in trades]
    return PortfolioBacktestResult(
        dates=list(dates),
        equity=equity,
        drawdown=drawdown,
        metrics=_segment_metrics(equity, held_flags, plain_trades),  # type: ignore[arg-type]
        allocations=allocations,
        cash_pct=cash_pct,
        trades=trades,
        decisions=decisions,
        journal=journal,
    )
