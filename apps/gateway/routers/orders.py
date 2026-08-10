"""Order preview API — the §10 entry gate chain, fully explainable (§33, §36).

``POST /api/orders/preview`` walks a proposed LONG_STOCK entry through the
nine §10 gates in their exact order and answers with per-gate PASS / FAIL /
SKIPPED status plus why-trade / why-not-trade narratives (§33 — both lists are
always present). Evaluation stops populating later gates after the first FAIL
("no rejected ticker may produce an order", §42): the remaining gates report
SKIPPED with "not evaluated: earlier gate failed".

V1 scope: VOLATILITY and LIQUIDITY are SKIPPED (no option/quote data until the
Massive integration lands, plan §22.1) and CONTRACT_SELECTION is SKIPPED for a
stock order — the gates still appear so the chain's shape never changes.

Risk approval calls ``libs.trading_core.risk.assess`` — the risk engine is
never reimplemented here, and risk limits have PRIORITY over strategy
confidence (§44 rule 20). Every preview writes exactly ONE SYSTEM-attributed
RISK_DECISION audit event — even when vetoed at gate one — in the same
transaction, so every decision is auditable (§38, rule 12).
"""
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
    TradingPoolItem,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
)
from ..schemas import TickerRequest
from .analysis import ensure_daily_bars, market_regime_from_spy
from .portfolio import open_positions_with_prices

router = APIRouter(prefix="/api/orders", tags=["orders"])

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


@router.post("/preview")
async def preview_order(
    req: OrderPreviewRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Evaluate the §10 gate chain for a proposed LONG_STOCK entry (§33, §42).

    Places no order. Writes exactly one SYSTEM RISK_DECISION audit event —
    veto or approval — committed in the same transaction (§38, rule 12).
    """
    ticker = req.ticker
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
                quantity_requested=req.quantity,
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

    # Exactly ONE RISK_DECISION audit event per preview — even an early veto
    # is a decision and must be auditable (§38, rule 12); committed here in
    # the same transaction as this (read-mostly) request.
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
    await session.commit()

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

    return {
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
            "quantity_requested": req.quantity,
        },
        "risk": risk_out,
        "why_trade": why_trade,
        "why_not_trade": why_not_trade,
    }
