"""Order preview + paper execution API — the §10 entry gate chain, fully
explainable (§33, §36), and the V1 paper fill model (plan §11).

``POST /api/orders/preview`` walks a proposed LONG_STOCK entry through the
nine §10 gates in their exact order and answers with per-gate PASS / FAIL /
SKIPPED status plus why-trade / why-not-trade narratives (§33 — both lists are
always present). Evaluation stops populating later gates after the first FAIL
("no rejected ticker may produce an order", §42): the remaining gates report
SKIPPED with "not evaluated: earlier gate failed".

``POST /api/orders/approve`` ALWAYS re-runs that same chain server-side —
client previews are never trusted (§42) — and only a fully passing chain may
fill. ``POST /api/orders/close`` sells an open position to close; closing is
allowed even while global trading is paused, because closing REDUCES risk and
risk protection outranks the pause (§18 risk-priority).

Paper fill model (plan §11; parameters live on Settings, §6.2): fills simulate
off the last STORED daily close, moved AGAINST the trader by
``paper_slippage_bps`` — BUY fills at ``close * (1 + bps/10000)``, SELL at
``close * (1 - bps/10000)`` — plus ``paper_commission_per_share * quantity``
commission charged on BOTH sides. The only order sides are BUY_TO_OPEN and
SELL_TO_CLOSE — Sell-to-Open does not exist in this system (§5).

V1 scope: VOLATILITY and LIQUIDITY are SKIPPED (no option/quote data until the
Massive integration lands, plan §22.1) and CONTRACT_SELECTION is SKIPPED for a
stock order — the gates still appear so the chain's shape never changes.

Risk approval calls ``libs.trading_core.risk.assess`` — the risk engine is
never reimplemented here, and risk limits have PRIORITY over strategy
confidence (§44 rule 20). Every chain run writes exactly ONE SYSTEM-attributed
RISK_DECISION audit event — even when vetoed at gate one — in the same
transaction, so every decision is auditable (§38, rule 12).
"""
import asyncio
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.features import atr
from libs.trading_core.models import (
    ActorType,
    AuditAction,
    DirectionalBias,
    InstrumentType,
    MarketRegime,
    RiskDecision,
)
from libs.trading_core.risk import (
    PortfolioSnapshot,
    PositionRisk,
    RiskAssessment,
    RiskLimits,
    RiskRequest,
    assess,
)
from libs.trading_core.signals import RegimeParams, classify_regime, score_direction

from .. import audit
from ..db import (
    Order,
    Position,
    StockBarDaily,
    TradingPoolItem,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
    utcnow,
)
from ..schemas import TickerRequest
from .analysis import ensure_daily_bars, market_regime_from_spy
from .portfolio import open_positions_with_prices

router = APIRouter(prefix="/api/orders", tags=["orders"])

# The only order sides that exist in this system (§5): a long-only account
# BUYS to open and SELLS to close. Sell-to-Open does not exist, ever.
BUY_TO_OPEN = "BUY_TO_OPEN"
SELL_TO_CLOSE = "SELL_TO_CLOSE"

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"
POSITION_CLOSED = "CLOSED"

# Single-user V1: fixed user identity until auth-service lands (matches
# routers/trading_control.py).
CURRENT_USER = "local-user"

# ---------------------------------------------------------------------------
# Execution serialization (§42 duplicate protection; V1 no-pyramiding).
#
# Every paper-execution mutation (approve, close, check-exits) runs its
# check-then-act sequence — idempotency lookup, open-position check, cash
# guard — and its fill inside ONE critical section. Without it, two rapid
# approves can BOTH pass the open-position check before either commits and
# double-fill (pyramiding, forbidden in V1), or a duplicate client_order_id
# can crash into the UNIQUE constraint with a 500 instead of replaying the
# existing order (§42). V1 runs a single process, so an in-process asyncio
# lock closes the race completely; the UNIQUE constraint on
# orders.client_order_id stays as the multi-process backstop.
#
# The lock is per event loop (test suites create one loop per test; asyncio
# primitives may not be shared across loops).
# ---------------------------------------------------------------------------
_EXECUTION_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def execution_lock() -> asyncio.Lock:
    """The current event loop's paper-execution lock (see block comment)."""
    loop = asyncio.get_running_loop()
    lock = _EXECUTION_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _EXECUTION_LOCKS[loop] = lock
    return lock

# --- V1 parameters (plan §6.2: parameters, never hardcoded truths) ----------
# Stop distance for a LONG_STOCK entry = ATR_STOP_MULTIPLE * ATR14 (plan §12.1).
ATR_STOP_MULTIPLE = 2.0
ATR_PERIOD = 14
# DATA_QUALITY staleness bound: last stored bar must be within this many
# calendar days of today (stub data always passes; the check must still exist).
MAX_BAR_AGE_DAYS = 5

# The §10 gate chain, in its exact order.
GATE_ORDER = (
    "TRADING_POOL_AUTHORIZATION",
    "DATA_QUALITY",
    "REGIME",
    "DIRECTIONAL_SIGNAL",
    "VOLATILITY",
    "INSTRUMENT",
    "LIQUIDITY",
    "CONTRACT_SELECTION",
    "RISK_APPROVAL",
)

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

# Exact V1 skip details (contract-fixed strings).
SKIP_EARLIER_FAIL = "not evaluated: earlier gate failed"
SKIP_NO_OPTION_DATA = "no option/quote data yet — arrives with the Massive integration"
SKIP_STOCK_ORDER = "stock order — no contract selection needed"

# Regimes that veto a new LONG_STOCK entry (§6.1: TRANSITION defaults to NO
# TRADE; bear regimes give a long-only account no edge, §5).
_BEAR_REGIMES = frozenset({MarketRegime.MILD_BEAR, MarketRegime.STRONG_BEAR})


class OrderPreviewRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)


@dataclass
class GateChainResult:
    """One full §10 gate-chain evaluation, shared by preview and approve.

    ``preview`` is the exact dict ``POST /api/orders/preview`` responds with;
    ``veto_gate`` names the first FAILing gate (``None`` when the chain fully
    passes — only then may an approval fill, §42). ``entry_price`` is the last
    stored close, ``stop_distance`` the §12.1 stop (2 * ATR14), ``edge`` the
    directional edge at evaluation, and ``last_bar_date`` the last stored bar
    date (YYYY-MM-DD) — the entry-bar anchor for ``bars_held`` (plan §11).

    The chain has already recorded its SYSTEM RISK_DECISION audit event on
    the session but NOT committed — the caller owns the transaction, so the
    audit lands atomically with whatever state change follows (rule 12).
    """

    preview: dict
    assessment: RiskAssessment | None
    entry_price: float | None
    stop_distance: float | None
    edge: float | None
    last_bar_date: str | None
    veto_gate: str | None

    @property
    def failed(self) -> bool:
        """True when any gate FAILed — no order may be produced (§42)."""
        return self.veto_gate is not None


async def run_gate_chain(
    session: AsyncSession, ticker: str, quantity: int | None
) -> GateChainResult:
    """Evaluate the §10 gate chain for a proposed LONG_STOCK entry (§33, §42).

    Places no order and never commits. Records exactly one SYSTEM
    RISK_DECISION audit event — veto or approval — on the session; the caller
    commits it in the same transaction as any state change (§38, rule 12).
    """
    settings = get_settings()
    limits = RiskLimits()
    regime_params = RegimeParams()

    gates: list[dict] = []

    def gate(name: str, status: str, detail: str) -> None:
        gates.append({"name": name, "status": status, "detail": detail})

    vetoed = False  # first FAIL stops evaluating later gates (§10, §42)
    signal_edge: float | None = None
    signal_bias: str | None = None
    entry_price: float | None = None
    stop_distance: float | None = None
    assessment: RiskAssessment | None = None

    # ------------------------------------------------------------------
    # Gate 1 — TRADING_POOL_AUTHORIZATION (§32, §18): the ticker must be in
    # the Trading Pool, per-symbol trading enabled, AND the global kill
    # switch must allow trading. FAIL names every missing authorization.
    # ------------------------------------------------------------------
    pool_row = (
        await session.execute(
            select(TradingPoolItem).where(TradingPoolItem.ticker == ticker)
        )
    ).scalar_one_or_none()
    state = await get_or_create_system_state(session)
    missing: list[str] = []
    if pool_row is None:
        missing.append(
            f"{ticker} is not in the Trading Pool — only pool symbols may trade (§32)"
        )
    elif not pool_row.trading_enabled:
        missing.append(f"trading is not enabled for {ticker} in the Trading Pool (§32)")
    if not state.trading_enabled:
        missing.append(
            "the global kill switch is engaged — trading is paused system-wide (§18)"
        )
    if missing:
        gate("TRADING_POOL_AUTHORIZATION", FAIL, "; ".join(missing))
        vetoed = True
    else:
        gate(
            "TRADING_POOL_AUTHORIZATION",
            PASS,
            f"{ticker} is in the Trading Pool with trading enabled and the "
            "global kill switch allows trading",
        )

    # ------------------------------------------------------------------
    # Gate 2 — DATA_QUALITY: stored bars exist (lazy backfill first, §4.2),
    # cover the regime engine's slow SMA, and are fresh (parameterized).
    # ------------------------------------------------------------------
    bars: list = []
    if vetoed:
        gate("DATA_QUALITY", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        problems: list[str] = []
        try:
            bars = await ensure_daily_bars(
                session, ticker, settings.market_data_provider
            )
        except HTTPException as exc:
            problems.append(f"no stored bars for {ticker}: {exc.detail}")
        age_days = 0
        if bars:
            if len(bars) < regime_params.sma_slow:
                problems.append(
                    f"only {len(bars)} stored bars; the regime engine needs at "
                    f"least {regime_params.sma_slow}"
                )
            last_ts = bars[-1].ts
            age_days = (datetime.now(timezone.utc).date() - last_ts).days
            if age_days > MAX_BAR_AGE_DAYS:
                problems.append(
                    f"last bar {last_ts.isoformat()} is {age_days} calendar days "
                    f"old (limit {MAX_BAR_AGE_DAYS})"
                )
        if problems:
            gate("DATA_QUALITY", FAIL, "; ".join(problems))
            vetoed = True
        else:
            gate(
                "DATA_QUALITY",
                PASS,
                f"{len(bars)} stored bars through {bars[-1].ts.isoformat()} "
                f"({age_days} calendar day(s) old, limit {MAX_BAR_AGE_DAYS})",
            )

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    if not vetoed:
        entry_price = closes[-1]
        atr_last = atr(highs, lows, closes, period=ATR_PERIOD)[-1]
        if atr_last is not None:
            stop_distance = ATR_STOP_MULTIPLE * atr_last

    # ------------------------------------------------------------------
    # Gate 3 — REGIME on the symbol's OWN bars (§6.1): TRANSITION defaults
    # to NO TRADE; bear regimes veto a long-only LONG_STOCK entry (§5).
    # ------------------------------------------------------------------
    if vetoed:
        gate("REGIME", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        regime = classify_regime(closes, highs, lows, params=regime_params)
        cls = regime.classification
        if cls is MarketRegime.TRANSITION:
            gate("REGIME", FAIL, "regime TRANSITION defaults to NO TRADE (§6.1)")
            vetoed = True
        elif cls in _BEAR_REGIMES:
            gate(
                "REGIME",
                FAIL,
                f"{ticker} regime is {cls.value}: a bear regime does not permit "
                "a new LONG_STOCK entry (long-only account, §5)",
            )
            vetoed = True
        else:
            gate("REGIME", PASS, f"{ticker} regime {cls.value} permits a long entry")

    # ------------------------------------------------------------------
    # Gate 4 — DIRECTIONAL_SIGNAL (§6.2): the bias must be BULL for a
    # LONG_STOCK entry (long-only account, §5). Detail carries the numbers.
    # ------------------------------------------------------------------
    if vetoed:
        gate("DIRECTIONAL_SIGNAL", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        sig = score_direction(closes, highs, lows, volumes=volumes)
        signal_edge = sig.directional_edge
        signal_bias = sig.bias.value
        numbers = (
            f"edge {sig.directional_edge:+.1f} "
            f"(bull {sig.bull_score:.1f} / bear {sig.bear_score:.1f})"
        )
        if sig.bias is not DirectionalBias.BULL:
            gate(
                "DIRECTIONAL_SIGNAL",
                FAIL,
                f"bias is {sig.bias.value} with {numbers}; LONG_STOCK requires "
                "a BULL bias (long-only account, §5)",
            )
            vetoed = True
        else:
            gate("DIRECTIONAL_SIGNAL", PASS, f"BULL bias with {numbers}")

    # ------------------------------------------------------------------
    # Gates 5–8 — VOLATILITY / INSTRUMENT / LIQUIDITY / CONTRACT_SELECTION.
    # VOLATILITY and LIQUIDITY are V1 skips (no option/quote data, §22.1);
    # INSTRUMENT exists so the account constraint is explicit (§5);
    # CONTRACT_SELECTION does not apply to a stock order.
    # ------------------------------------------------------------------
    gate("VOLATILITY", SKIPPED, SKIP_EARLIER_FAIL if vetoed else SKIP_NO_OPTION_DATA)
    if vetoed:
        gate("INSTRUMENT", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        gate(
            "INSTRUMENT",
            PASS,
            "LONG_STOCK is permitted by the V1 long-only account constraints (§5)",
        )
    gate("LIQUIDITY", SKIPPED, SKIP_EARLIER_FAIL if vetoed else SKIP_NO_OPTION_DATA)
    gate(
        "CONTRACT_SELECTION",
        SKIPPED,
        SKIP_EARLIER_FAIL if vetoed else SKIP_STOCK_ORDER,
    )

    # ------------------------------------------------------------------
    # Gate 9 — RISK_APPROVAL: build the portfolio snapshot and call the risk
    # engine (§12, §17). Risk limits have PRIORITY over strategy confidence
    # (§44 rule 20) — a REJECT here vetoes regardless of signal strength.
    # ------------------------------------------------------------------
    if vetoed:
        gate("RISK_APPROVAL", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        portfolio = await get_or_create_portfolio(session)
        pairs = await open_positions_with_prices(session)
        nav = portfolio.cash + sum(
            pos.quantity * price for pos, price in pairs if price is not None
        )
        position_risks = [
            PositionRisk(
                ticker=pos.ticker,
                market_value=(pos.quantity * price) if price is not None else 0.0,
                max_loss=pos.max_loss,
            )
            for pos, price in pairs
        ]
        spy_regime = (await market_regime_from_spy(session)).classification
        snapshot = PortfolioSnapshot(
            nav=nav,
            cash=portfolio.cash,
            positions=position_risks,
            regime=spy_regime,
            trading_enabled=state.trading_enabled,
        )
        assessment = assess(
            RiskRequest(
                ticker=ticker,
                entry_price=entry_price,
                stop_distance=stop_distance,
                edge=signal_edge,
                quantity_requested=quantity,
            ),
            snapshot,
            limits,
        )
        if assessment.decision is RiskDecision.REJECT:
            gate(
                "RISK_APPROVAL",
                FAIL,
                f"risk engine REJECT ({', '.join(assessment.reason_codes)}) — "
                "risk limits have priority over strategy confidence (§44 rule 20)",
            )
            vetoed = True
        elif assessment.decision is RiskDecision.APPROVE_WITH_RESIZE:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE_WITH_RESIZE: quantity resized to "
                f"{assessment.approved_quantity} "
                f"({', '.join(assessment.reason_codes)})",
            )
        else:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE: {assessment.approved_quantity} shares, "
                f"${assessment.trade_risk_usd:,.2f} at risk",
            )

    # §33: aggregate the passing/failing gate details + risk explanations;
    # both lists are ALWAYS present.
    why_trade = [f"{g['name']}: {g['detail']}" for g in gates if g["status"] == PASS]
    why_not_trade = [f"{g['name']}: {g['detail']}" for g in gates if g["status"] == FAIL]
    if assessment is not None:
        if assessment.decision is RiskDecision.REJECT:
            why_not_trade.extend(assessment.explanations)
        else:
            why_trade.extend(assessment.explanations)

    # Exactly ONE RISK_DECISION audit event per chain run — even an early veto
    # is a decision and must be auditable (§38, rule 12). Recorded on the
    # session only; the CALLER commits, so the event shares one transaction
    # with any state change (preview: none; approve: the fill).
    veto_gate = next((g["name"] for g in gates if g["status"] == FAIL), None)
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.RISK_DECISION,
        entity_type="order_preview",
        entity_id=ticker,
        details={
            "decision": assessment.decision.value if assessment is not None else "VETOED",
            "veto_gate": veto_gate,
            "gates": {g["name"]: g["status"] for g in gates},
            "reason_codes": (
                list(assessment.reason_codes)
                if assessment is not None
                else ([f"VETO_{veto_gate}"] if veto_gate else [])
            ),
        },
    )

    risk_out = None
    if assessment is not None:
        risk_out = {
            "decision": assessment.decision.value,
            "approved_quantity": assessment.approved_quantity,
            "signal_strength": assessment.signal_strength,
            "risk_budget_pct": assessment.risk_budget_pct,
            "trade_risk_usd": assessment.trade_risk_usd,
            "reason_codes": assessment.reason_codes,
            "explanations": assessment.explanations,
            "heat_before_pct": assessment.heat_before_pct,
            "heat_after_pct": assessment.heat_after_pct,
            "cash_after_pct": assessment.cash_after_pct,
        }

    preview = {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "signal": {
            "edge": signal_edge,
            "bias": signal_bias,
            "strength": assessment.signal_strength if assessment is not None else None,
        },
        "proposed": {
            "instrument": InstrumentType.LONG_STOCK.value,
            "entry_price": entry_price,
            "stop_distance": stop_distance,
            "quantity_requested": quantity,
        },
        "risk": risk_out,
        "why_trade": why_trade,
        "why_not_trade": why_not_trade,
    }
    return GateChainResult(
        preview=preview,
        assessment=assessment,
        entry_price=entry_price,
        stop_distance=stop_distance,
        edge=signal_edge,
        last_bar_date=bars[-1].ts.isoformat() if bars else None,
        veto_gate=veto_gate,
    )


@router.post("/preview")
async def preview_order(
    req: OrderPreviewRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Evaluate the §10 gate chain for a proposed LONG_STOCK entry (§33, §42).

    Places no order. Writes exactly one SYSTEM RISK_DECISION audit event —
    veto or approval — committed in the same transaction (§38, rule 12).
    """
    result = await run_gate_chain(session, req.ticker, req.quantity)
    await session.commit()
    return result.preview


# ---------------------------------------------------------------------------
# Paper execution (plan §11, §42)
# ---------------------------------------------------------------------------


class OrderApproveRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)
    # Idempotency key (§42): replaying the same key returns the existing
    # order — a duplicate request can never fill twice.
    client_order_id: str | None = Field(default=None, min_length=1, max_length=64)


class OrderCloseRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)  # default: close in full
    reason: str | None = None


def _order_payload(order: Order) -> dict:
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "ticker": order.ticker,
        "side": order.side,
        "quantity": order.quantity,
        "fill_price": order.fill_price,
        "commission": order.commission,
        "status": order.status,
        "created_at": order.created_at.isoformat(),
    }


def _position_payload(position: Position) -> dict:
    return {
        "id": position.id,
        "ticker": position.ticker,
        "status": position.status,
        "quantity": position.quantity,
        "avg_price": position.avg_price,
        "stop_price": (
            position.avg_price - position.stop_distance
            if position.stop_distance > 0
            else None
        ),
        "max_loss": position.max_loss,
        "realized_pnl": position.realized_pnl,
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
    }


async def _open_position(session: AsyncSession, ticker: str) -> Position | None:
    """The OPEN position in `ticker`, if any (at most one — no pyramiding, V1)."""
    return (
        (
            await session.execute(
                select(Position)
                .where(Position.ticker == ticker, Position.status == POSITION_OPEN)
                .order_by(Position.id)
            )
        )
        .scalars()
        .first()
    )


async def _last_stored_close(session: AsyncSession, ticker: str) -> float | None:
    """Last stored daily close for `ticker` — the paper fill reference price."""
    return (
        (
            await session.execute(
                select(StockBarDaily.close)
                .where(StockBarDaily.ticker == ticker)
                .order_by(StockBarDaily.ts.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def execute_sell_to_close(
    session: AsyncSession,
    position: Position,
    quantity: int,
    last_close: float,
    *,
    reason: str | None = None,
    system_generated: bool = False,
) -> tuple[Order, float]:
    """Fill a SELL_TO_CLOSE against `position` at the paper fill model (§11).

    SELL fill = ``last_close * (1 - paper_slippage_bps/10000)`` (slippage
    always moves AGAINST the trader); commission =
    ``paper_commission_per_share * quantity``, charged on this side too.
    ``realized_pnl = (fill - avg_price) * quantity - commission`` — the
    buy-side commission is NOT re-charged here because it already left cash
    at open. Cash is credited the net proceeds; the position shrinks (its
    ``max_loss`` scales down proportionally so portfolio heat, plan §12.5,
    reflects only the risk still open) and flips to CLOSED with ``closed_at``
    when quantity reaches 0. Records the full ORDER_REQUESTED (SYSTEM when
    ``system_generated``, else USER, plan §11) -> ORDER_SUBMITTED ->
    ORDER_FILLED audit chain on the session and NEVER commits — the caller
    owns the transaction (rule 12).

    Deliberately checks NEITHER the kill switch nor the §10 gates: closing
    REDUCES risk, and risk protection outranks the pause (§18 risk-priority).
    The side is SELL_TO_CLOSE — Sell-to-Open does not exist in this
    system (§5).
    """
    settings = get_settings()
    fill = last_close * (1.0 - settings.paper_slippage_bps / 10000.0)
    commission = settings.paper_commission_per_share * quantity
    proceeds = quantity * fill - commission
    realized = (fill - position.avg_price) * quantity - commission

    portfolio = await get_or_create_portfolio(session)
    portfolio.cash += proceeds
    portfolio.updated_at = utcnow()

    order = Order(
        ticker=position.ticker,
        side=SELL_TO_CLOSE,
        quantity=quantity,
        fill_price=fill,
        commission=commission,
        status="FILLED",
    )
    session.add(order)
    await session.flush()

    remaining = position.quantity - quantity
    position.max_loss = (
        position.max_loss * remaining / position.quantity
        if position.quantity > 0
        else 0.0
    )
    position.quantity = remaining
    position.realized_pnl = (position.realized_pnl or 0.0) + realized
    if remaining <= 0:
        position.status = POSITION_CLOSED
        position.closed_at = utcnow()
        position.max_loss = 0.0

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM if system_generated else ActorType.USER,
        actor_id="" if system_generated else CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "ticker": position.ticker,
            "side": SELL_TO_CLOSE,
            "quantity": quantity,
            "reason": reason,
            "system_generated": system_generated,
        },
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_model": "paper: last stored close * (1 - slippage_bps/10000)",
            "last_close": last_close,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_share": settings.paper_commission_per_share,
        },
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_price": fill,
            "commission": commission,
            "position_id": position.id,
            "realized_pnl": realized,
        },
    )
    return order, realized


@router.post("/approve")
async def approve_order(
    req: OrderApproveRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Approve and paper-fill a BUY_TO_OPEN entry — ONE transaction (§11, §42).

    The server ALWAYS re-runs the full §10 gate chain at approval time —
    client previews are never trusted (§42); any FAILing gate answers 422
    with the fresh preview embedded, and no order row may exist for a
    rejected ticker. Flow: idempotency lookup (a duplicate
    ``client_order_id`` returns the EXISTING order, 200, no second fill,
    §42) -> open-position check (an existing OPEN position in the ticker is
    409 — no pyramiding in V1) -> gate chain -> fill.

    Fill model (§11): BUY fill = last stored close *
    ``(1 + paper_slippage_bps/10000)`` plus
    ``paper_commission_per_share * quantity`` commission; quantity =
    min(requested, risk-approved) — risk limits outrank strategy confidence
    (§44 rule 20). Cash is guarded again at fill time (INSUFFICIENT_CASH ->
    422) even though the risk chain already enforced the cash floor (§13).
    The Order + Position + cash decrement + audit chain ORDER_REQUESTED
    (USER) -> ORDER_SUBMITTED -> ORDER_FILLED (+ the chain's RISK_DECISION)
    all commit atomically (rule 12). The whole flow runs under the
    paper-execution lock so two rapid approves can never double-fill (§42;
    V1 no-pyramiding).
    """
    async with execution_lock():
        return await _approve_order_locked(req, session)


async def _approve_order_locked(
    req: OrderApproveRequest, session: AsyncSession
) -> dict:
    """The approve flow proper — caller holds the paper-execution lock."""
    ticker = req.ticker

    # --- Idempotency (§42): same client_order_id -> the EXISTING order. ----
    if req.client_order_id is not None:
        existing = (
            (
                await session.execute(
                    select(Order).where(Order.client_order_id == req.client_order_id)
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            # Replay: no re-evaluation, no second fill, no new audit events.
            # The position reflects its CURRENT state; preview is an honest
            # null (§44 rule 18) — nothing was re-evaluated on this call.
            position = (
                (
                    await session.execute(
                        select(Position)
                        .where(Position.ticker == existing.ticker)
                        .order_by(Position.id.desc())
                    )
                )
                .scalars()
                .first()
            )
            return {
                "order": _order_payload(existing),
                "position": _position_payload(position) if position else None,
                "preview": None,
            }

    # --- No pyramiding in V1: one OPEN position per ticker. ---------------
    if await _open_position(session, ticker) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{ticker} already has an OPEN position — pyramiding is not "
                "supported in V1; close it before opening again"
            ),
        )

    # --- Re-run the FULL §10 chain (§42: client previews are never trusted).
    chain = await run_gate_chain(session, ticker, req.quantity)
    if chain.failed or chain.assessment is None:
        # The chain already recorded its RISK_DECISION; commit it so the veto
        # stays auditable (rule 12), then answer 422 with the fresh preview.
        await session.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"approval denied: gate {chain.veto_gate} failed — no "
                    "rejected ticker may produce an order (§42)"
                ),
                "preview": chain.preview,
            },
        )

    approved = chain.assessment.approved_quantity
    quantity = min(req.quantity, approved) if req.quantity is not None else approved

    # --- Paper fill (§11): slippage against the trader + commission. -------
    settings = get_settings()
    fill = chain.entry_price * (1.0 + settings.paper_slippage_bps / 10000.0)
    commission = settings.paper_commission_per_share * quantity
    cost = quantity * fill + commission

    portfolio = await get_or_create_portfolio(session)
    if cost > portfolio.cash:
        # The §13 cash floor was already enforced by RISK_APPROVAL, but the
        # fill adds slippage + commission — guard anyway (risk outranks all).
        await session.commit()  # keep the RISK_DECISION auditable (rule 12)
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"INSUFFICIENT_CASH: cost ${cost:,.2f} exceeds cash "
                    f"${portfolio.cash:,.2f}"
                ),
                "preview": chain.preview,
            },
        )

    order = Order(
        client_order_id=req.client_order_id,
        ticker=ticker,
        side=BUY_TO_OPEN,
        quantity=quantity,
        fill_price=fill,
        commission=commission,
        status="FILLED",
    )
    session.add(order)
    portfolio.cash -= cost
    portfolio.updated_at = utcnow()
    position = Position(
        ticker=ticker,
        quantity=quantity,
        avg_price=fill,
        max_loss=quantity * chain.stop_distance,
        stop_distance=chain.stop_distance,
        entry_edge=chain.edge,
        entry_bar_date=chain.last_bar_date,
        status=POSITION_OPEN,
    )
    session.add(position)
    await session.flush()

    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "ticker": ticker,
            "side": BUY_TO_OPEN,
            "quantity_requested": req.quantity,
            "client_order_id": req.client_order_id,
        },
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_model": "paper: last stored close * (1 + slippage_bps/10000)",
            "last_close": chain.entry_price,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_share": settings.paper_commission_per_share,
        },
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_price": fill,
            "commission": commission,
            "quantity": quantity,
            "position_id": position.id,
        },
    )
    await session.commit()

    return {
        "order": _order_payload(order),
        "position": {
            "id": position.id,
            "ticker": position.ticker,
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            "stop_price": position.avg_price - position.stop_distance,
            "max_loss": position.max_loss,
        },
        "preview": chain.preview,
    }


@router.post("/close")
async def close_position(
    req: OrderCloseRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Sell-to-close an OPEN paper position, fully or partially (§11).

    Closing is ALLOWED while global trading is paused: the §18 kill switch
    blocks NEW risk, and closing REDUCES risk — risk protection has priority
    (§18 risk-priority), so no gate chain and no kill-switch check runs here.
    404 when no OPEN position exists in the ticker; 422 when the requested
    quantity exceeds the open quantity. Fill, cash credit, position update
    and the ORDER_REQUESTED (USER) -> ORDER_SUBMITTED -> ORDER_FILLED audit
    chain commit in ONE transaction (rule 12), under the paper-execution
    lock so two rapid closes can never double-credit cash (§42 analogue).
    """
    async with execution_lock():
        return await _close_position_locked(req, session)


async def _close_position_locked(
    req: OrderCloseRequest, session: AsyncSession
) -> dict:
    """The close flow proper — caller holds the paper-execution lock."""
    ticker = req.ticker
    position = await _open_position(session, ticker)
    if position is None:
        raise HTTPException(status_code=404, detail=f"no OPEN position in {ticker}")

    quantity = req.quantity if req.quantity is not None else position.quantity
    if quantity > position.quantity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"cannot close {quantity} shares of {ticker}: only "
                f"{position.quantity} open"
            ),
        )

    last_close = await _last_stored_close(session, ticker)
    if last_close is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"no stored bars for {ticker} — the paper fill model has no "
                "reference price (honest error, §44 rule 18)"
            ),
        )

    order, realized = await execute_sell_to_close(
        session,
        position,
        quantity,
        last_close,
        reason=req.reason,
        system_generated=False,
    )
    await session.commit()

    return {
        "order": _order_payload(order),
        "position": _position_payload(position),
        "realized_pnl": realized,
    }
