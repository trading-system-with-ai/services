"""Income strategies API — covered calls / cash-secured puts (Phase 2).

COLLATERAL IS THE LAW OF THIS MODULE: no short leg exists here without its
backing named and LOCKED first —

- COVERED_CALL: an OPEN LONG_STOCK position in the same ticker must have
  FREE shares >= 100 per contract (free = held minus shares already
  pinned under other open covered calls). The link is stored
  (``collateral_position_id``) and the stock close path refuses to sell
  pinned shares until the call is bought back.
- CASH_SECURED_PUT: ``strike * 100 * contracts`` is reserved
  (``cash_reserved``) and excluded from deployable cash until the put is
  bought back.

Fills follow the platform's venue split (§11): the internal simulator
(BROKER_PROVIDER=simulated) credits the premium at mid*(1-slippage) with
per-contract commission; the real broker path submits through the
adapter's collateral-attested ``submit_short_open_order`` /
``submit_short_close_order``. Selling premium can only ever be closed by
BUYING BACK (BUY_TO_CLOSE) — nothing in this module can grow exposure
after open. Management is the shared engine's mechanical standards
(evaluate_short_premium_exit: 50% profit capture / 2x loss stop / 21 DTE,
ITM assignment ADVISORY) — the exit sweep evaluates these rows too.

Kill switch: OPENING income positions requires trading enabled (they add
obligations); BUYBACKS are always allowed (risk-reducing, §18 priority).
"""
from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.broker.alpaca import occ_option_symbol
from libs.common.config import get_settings
from libs.trading_core.greeks import PositionGreeksInput
from libs.trading_core.models import (
    ActorType,
    AuditAction,
    InstrumentType,
    RiskDecision,
)
from libs.trading_core.risk import (
    IncomeRiskRequest,
    RiskAssessment,
    RiskLimits,
    assess_income,
)
from libs.trading_core.strategies import (
    IncomeParams,
    select_cash_secured_put,
    select_covered_call,
)

from .. import audit
from ..broker_exec import (
    BrokerError,
    BrokerRejected,
    broker_order_details,
    new_client_order_id,
    submit_short_close_and_poll,
    submit_short_open_and_poll,
)
from ..db import (
    Order,
    Position,
    TradingPoolItem,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
    utcnow,
)
from ..deps import (
    account_permissions_from_settings,
    require_broker,
    require_market_data_provider,
    resolve_broker,
    simulated_broker_mode,
)
from ..risk_inputs import build_portfolio_snapshot
from .options import build_option_chain
from .orders import (
    OPTION_MULTIPLIER,
    POSITION_OPEN,
    _broker_cash_for_sizing,
    _last_stored_close,
    execution_lock,
)
from .portfolio import find_option_contract, portfolio_greeks_read
from .watchlist import CURRENT_USER

router = APIRouter(prefix="/api/income", tags=["income"])

INCOME_PARAMS = IncomeParams()

#: Shares of stock collateral pinned per short call contract (OCC standard).
SHARES_PER_CONTRACT = 100


class IncomeOpenRequest(BaseModel):
    """POST /api/income/covered-call | /cash-secured-put body."""

    ticker: str
    contracts: int = Field(default=1, ge=1, le=100)


class BuybackRequest(BaseModel):
    reason: str | None = None


def _require_income_permission(kind: str) -> None:
    """Phase 2 unlock gate: refused until the WHOLE chain lands and the
    §33 forbidden set shrinks. Kept in ONE place so the eventual unlock is
    one edit wide."""
    perms = account_permissions_from_settings()
    allowed = perms.covered_call if kind == "covered_call" else perms.cash_secured_put
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"{kind} is disabled in account permissions — enable it in "
                "Settings (Account Permissions) to trade collateralized "
                "short premium."
            ),
        )


async def _require_trading_enabled(session: AsyncSession) -> None:
    state = await get_or_create_system_state(session)
    if not state.trading_enabled:
        raise HTTPException(
            status_code=409,
            detail=(
                "trading is paused (§18 kill switch) — opening income "
                "positions adds obligations and is blocked; buybacks remain "
                "allowed."
            ),
        )


def pinned_shares(stock_position: Position, open_calls: list[Position]) -> int:
    """Shares of `stock_position` already backing OPEN covered calls."""
    return sum(
        SHARES_PER_CONTRACT * cc.quantity
        for cc in open_calls
        if cc.collateral_position_id == stock_position.id
        and cc.status == POSITION_OPEN
    )


async def _open_income_rows(session: AsyncSession, ticker: str) -> list[Position]:
    rows = await session.execute(
        select(Position).where(
            Position.ticker == ticker,
            Position.status == POSITION_OPEN,
            Position.instrument.in_(
                [
                    InstrumentType.COVERED_CALL.value,
                    InstrumentType.CASH_SECURED_PUT.value,
                ]
            ),
        )
    )
    return list(rows.scalars().all())


#: Audit entity type for the income risk decisions (entity_id = ticker), so
#: the alerts feed resolves the symbol like it does for ``order_preview``.
RISK_DECISION_ENTITY = "income_open"


async def _require_pool_authorization(
    session: AsyncSession, *, ticker: str, kind: str, contracts: int
) -> None:
    """Trading Pool authorization — gate 1 semantics mirrored from
    ``routers/orders.run_gate_chain`` (§21, §32; audit §8 item 3, §10 Phase
    B0): the symbol must be a Trading Pool row with per-symbol trading
    enabled. A refusal is a risk decision on a write path, so it is recorded
    as ONE SYSTEM RISK_DECISION (decision ``VETOED``, veto gate named),
    committed, and answered 422 like the order path's approve denial.
    """
    pool_row = (
        await session.execute(
            select(TradingPoolItem).where(TradingPoolItem.ticker == ticker)
        )
    ).scalar_one_or_none()
    missing: list[str] = []
    if pool_row is None:
        missing.append(
            f"{ticker} is not in the Trading Pool — only pool symbols may trade (§32)"
        )
    elif not pool_row.trading_enabled:
        missing.append(
            f"trading is not enabled for {ticker} in the Trading Pool (§32)"
        )
    if not missing:
        return
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.RISK_DECISION,
        entity_type=RISK_DECISION_ENTITY,
        entity_id=ticker,
        details={
            "decision": "VETOED",
            "mode": "execution",
            "ticker": ticker,
            "instrument": kind,
            "veto_gate": "TRADING_POOL_AUTHORIZATION",
            "reason_codes": ["VETO_TRADING_POOL_AUTHORIZATION"],
            "explanations": missing,
            "quantity_requested": contracts,
            "approved_quantity": 0,
        },
    )
    await session.commit()  # the veto stays auditable (rule 12)
    raise HTTPException(
        status_code=422,
        detail={
            "message": (
                f"{kind} open denied: gate TRADING_POOL_AUTHORIZATION failed "
                "— no unauthorized ticker may produce an order (§21, §42)"
            ),
            "veto_gate": "TRADING_POOL_AUTHORIZATION",
            "reason_codes": ["VETO_TRADING_POOL_AUTHORIZATION"],
            "explanations": missing,
        },
    )


def _short_leg_greeks(
    ticker: str, kind: str, leg, spot: float
) -> PositionGreeksInput | None:
    """The candidate SHORT leg's per-share greeks, NEGATED (plan §16;
    ``libs.trading_core.greeks`` conventions — the same sign flip
    ``portfolio_greeks_read`` applies to open income rows). ``None`` when
    the provider reports no greeks for the contract: an honest skip (no
    greek check), never zeros presented as a flat contribution."""
    if None in (leg.delta, leg.gamma, leg.theta, leg.vega):
        return None
    return PositionGreeksInput(
        ticker=ticker,
        instrument=kind,
        quantity=1,  # requested basis; assess_income scales by approved qty
        multiplier=OPTION_MULTIPLIER,
        spot=spot,
        delta=-leg.delta,
        gamma=-leg.gamma,
        theta_per_day=-leg.theta,
        vega=-leg.vega,
    )


async def _income_risk_gate(
    session: AsyncSession,
    *,
    kind: str,
    ticker: str,
    leg,
    contracts: int,
    spot: float,
) -> tuple[RiskAssessment, dict]:
    """Tier 0 for income opens (audit §8 item 3, §10 Phase B0; spec §2, §72):
    usable cash -> shared portfolio snapshot -> greeks -> ``assess_income``
    -> ONE SYSTEM RISK_DECISION on the session (committed by the caller in
    the same transaction as the fill; on REJECT committed here before the
    422 so the decision stays auditable, rule 12).

    Risk / capital bases per contract (audit §10 Phase B0):

    - CASH_SECURED_PUT: risk ``(strike − credit) × 100`` (stock to zero, the
      row's ``max_loss`` basis at the chain mid), capital ``strike × 100``
      (the cash reservation).
    - COVERED_CALL: risk 0 / capital 0 — the stock row already carries the
      heat and the pinned shares are already deployed capital; only the
      kill switch, the heat gate and the greek limits can bind.

    Returns ``(assessment, risk_block)`` where ``risk_block`` is the
    response's ``risk`` object (requested vs approved, reasons).
    """
    state = await get_or_create_system_state(session)
    # §14: usable cash. Simulated -> the local ledger; a real broker -> its
    # LIVE cash (never buying power). Unreadable broker -> FAIL CLOSED
    # (§28): deployable cash cannot be verified, nothing is sized.
    broker_cash, broker_cash_error = await _broker_cash_for_sizing()
    if broker_cash_error is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "BROKER_ERROR",
                "message": (
                    "the broker account could not be fetched "
                    f"({broker_cash_error}) — deployable cash cannot be "
                    "verified and sizing from local cash alone is forbidden "
                    "with a real broker configured (§14, §28: failing "
                    "closed; no income position may be opened until the "
                    "broker answers)."
                ),
            },
        )
    if broker_cash is not None:
        account_cash = broker_cash
    else:
        account_cash = (await get_or_create_portfolio(session)).cash
    inputs = await build_portfolio_snapshot(
        session, cash=account_cash, trading_enabled=state.trading_enabled
    )
    snapshot = inputs.snapshot

    # §16 greeks: the current book (shared helper) + the short leg NEGATED.
    # Either side unknowable -> BOTH None (honest skip, no greek check).
    candidate = _short_leg_greeks(ticker, kind, leg, spot)
    book_greeks = portfolio_greeks_read(inputs.pairs)[0] if candidate else None
    greeks_checked = candidate is not None and book_greeks is not None

    if kind == InstrumentType.CASH_SECURED_PUT.value:
        # Risk basis at the EXPECTED credit — the chain mid net of the paper
        # slippage the simulated fill applies (mid × (1 − bps/1e4)) — so the
        # assessed heat is never BELOW the heat the Position row will book
        # ((strike − fill) × 100, see _fill_short_open). Broker fills are
        # unknowable ahead; this is the conservative estimate (QA finding).
        expected_credit = leg.mid * (
            1.0 - get_settings().paper_slippage_bps / 10000.0
        )
        risk_per_contract = max(leg.strike - expected_credit, 0.0) * OPTION_MULTIPLIER
        capital_per_contract = leg.strike * OPTION_MULTIPLIER
    else:
        risk_per_contract = 0.0
        capital_per_contract = 0.0
    assessment = assess_income(
        IncomeRiskRequest(
            ticker=ticker,
            instrument=kind,
            contracts=contracts,
            risk_per_contract=risk_per_contract,
            capital_per_contract=capital_per_contract,
        ),
        snapshot,
        RiskLimits(),
        portfolio_greeks=book_greeks if greeks_checked else None,
        new_position_greeks=candidate if greeks_checked else None,
    )
    details = {
        "decision": assessment.decision.value,
        "mode": "execution",
        "ticker": ticker,
        "instrument": kind,
        "veto_gate": (
            "RISK_APPROVAL"
            if assessment.decision is RiskDecision.REJECT
            else None
        ),
        "reason_codes": list(assessment.reason_codes),
        "explanations": list(assessment.explanations),
        "quantity_requested": contracts,
        "approved_quantity": assessment.approved_quantity,
        "trade_risk_usd": assessment.trade_risk_usd,
        "risk_per_contract": risk_per_contract,
        "capital_per_contract": capital_per_contract,
        "heat_before_pct": assessment.heat_before_pct,
        "heat_after_pct": assessment.heat_after_pct,
        "cash_after_pct": assessment.cash_after_pct,
        "usable_cash": inputs.usable_cash,
        "cash_reserved_total": inputs.cash_reserved_total,
        "nav": snapshot.nav,
        "greeks_checked": greeks_checked,
        "greeks_note": (
            None
            if greeks_checked
            else "provider reports no greeks for the short leg — greek "
            "limits not checked (honest skip)"
        ),
    }
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.RISK_DECISION,
        entity_type=RISK_DECISION_ENTITY,
        entity_id=ticker,
        details=details,
    )
    risk_block = {
        "decision": assessment.decision.value,
        "quantity_requested": contracts,
        "approved_quantity": assessment.approved_quantity,
        "reason_codes": list(assessment.reason_codes),
        "explanations": list(assessment.explanations),
        "trade_risk_usd": assessment.trade_risk_usd,
        "heat_before_pct": assessment.heat_before_pct,
        "heat_after_pct": assessment.heat_after_pct,
        "cash_after_pct": assessment.cash_after_pct,
        "greeks_checked": greeks_checked,
    }
    if assessment.decision is RiskDecision.REJECT:
        await session.commit()  # the REJECT stays auditable (rule 12)
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"{kind} open rejected by the risk engine "
                    f"({', '.join(assessment.reason_codes)}) — risk limits "
                    "have priority over everything (§44 rule 20)"
                ),
                "veto_gate": "RISK_APPROVAL",
                "risk": risk_block,
                "reason_codes": list(assessment.reason_codes),
                "explanations": list(assessment.explanations),
            },
        )
    return assessment, risk_block


@router.post("/covered-call")
async def open_covered_call(
    req: IncomeOpenRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Sell a covered call against an existing LONG_STOCK holding.

    Chain: permission -> kill switch -> Trading Pool authorization (gate 1,
    §21/§32) -> collateral (free shares) -> live §-selection (30-45 DTE, |Δ|
    0.15-0.35, OTM, sellable) -> Tier 0 risk gate (``assess_income``, audit
    §10 Phase B0: kill switch / heat / greek limits; one RISK_DECISION audit
    in the fill's transaction) -> venue fill -> Position row linked to its
    collateral. 422 with named reasons at every refusal (§44 rule 18).
    """
    require_market_data_provider()
    require_broker()
    _require_income_permission("covered_call")
    ticker = req.ticker.strip().upper()
    kind = InstrumentType.COVERED_CALL.value
    async with execution_lock():
        await _require_trading_enabled(session)
        await _require_pool_authorization(
            session, ticker=ticker, kind=kind, contracts=req.contracts
        )

        # --- Collateral: free shares in an OPEN LONG_STOCK row. ------------
        stock = (
            (
                await session.execute(
                    select(Position).where(
                        Position.ticker == ticker,
                        Position.status == POSITION_OPEN,
                        Position.instrument == InstrumentType.LONG_STOCK.value,
                    )
                )
            )
            .scalars()
            .first()
        )
        if stock is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"no OPEN LONG_STOCK position in {ticker} — a covered "
                    "call sells upside against shares you hold (§5: the "
                    "short leg must be covered)."
                ),
            )
        open_income = await _open_income_rows(session, ticker)
        free = stock.quantity - pinned_shares(stock, open_income)
        needed = SHARES_PER_CONTRACT * req.contracts
        if free < needed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"insufficient free shares: {req.contracts} contract(s) "
                    f"need {needed} shares, but only {free} of "
                    f"{stock.quantity} held shares are not already pinned "
                    "under open covered calls."
                ),
            )

        # --- Live selection (§42: fresh chain for execution). --------------
        spot = await _last_stored_close(session, ticker)
        if spot is None:
            raise HTTPException(
                status_code=422,
                detail=f"no stored bars for {ticker} — no spot reference.",
            )
        _, chain = build_option_chain(ticker, spot, max_age_seconds=0.0)
        selection = select_covered_call(chain, spot, INCOME_PARAMS)
        if selection.contract is None:
            raise HTTPException(
                status_code=422,
                detail="; ".join(selection.fail_reasons),
            )
        leg = selection.contract

        # --- Tier 0 risk gate (audit §8 item 3): REJECT -> 422 above; a
        # RESIZE fills the APPROVED contracts (never more than the pinned
        # shares allow — collateral was checked for the larger request).
        assessment, risk_block = await _income_risk_gate(
            session,
            kind=kind,
            ticker=ticker,
            leg=leg,
            contracts=req.contracts,
            spot=spot,
        )
        result = await _fill_short_open(
            session,
            kind=kind,
            ticker=ticker,
            leg=leg,
            contracts=assessment.approved_quantity,
            rationale=selection.rationale,
            collateral_position_id=stock.id,
            cash_reserved=None,
        )
        result["risk"] = risk_block
        return result


@router.post("/cash-secured-put")
async def open_cash_secured_put(
    req: IncomeOpenRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Sell a cash-secured put: strike*100 per contract is LOCKED first.

    Chain: permission -> kill switch -> Trading Pool authorization (gate 1,
    §21/§32) -> live selection -> cash collateral law -> Tier 0 risk gate
    (``assess_income``, audit §10 Phase B0: risk basis (strike − credit) ×
    100, capital basis strike × 100 per contract; one RISK_DECISION audit in
    the fill's transaction) -> venue fill.
    """
    require_market_data_provider()
    require_broker()
    _require_income_permission("cash_secured_put")
    ticker = req.ticker.strip().upper()
    kind = InstrumentType.CASH_SECURED_PUT.value
    async with execution_lock():
        await _require_trading_enabled(session)
        await _require_pool_authorization(
            session, ticker=ticker, kind=kind, contracts=req.contracts
        )

        spot = await _last_stored_close(session, ticker)
        if spot is None:
            raise HTTPException(
                status_code=422,
                detail=f"no stored bars for {ticker} — no spot reference.",
            )
        _, chain = build_option_chain(ticker, spot, max_age_seconds=0.0)
        selection = select_cash_secured_put(chain, spot, INCOME_PARAMS)
        if selection.contract is None:
            raise HTTPException(
                status_code=422, detail="; ".join(selection.fail_reasons)
            )
        leg = selection.contract
        reserve = leg.strike * OPTION_MULTIPLIER * req.contracts

        # --- Cash collateral (simulated ledger; broker mode verifies at
        # the broker which enforces its own buying power). -------------------
        if simulated_broker_mode():
            portfolio = await get_or_create_portfolio(session)
            already_reserved = sum(
                (row.cash_reserved or 0.0)
                for row in await _all_open_csp_rows(session)
            )
            available = portfolio.cash - already_reserved
            if available < reserve:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"insufficient free cash to secure the put: need "
                        f"${reserve:,.2f} (strike {leg.strike:g} x 100 x "
                        f"{req.contracts}), have ${available:,.2f} free "
                        f"(${already_reserved:,.2f} already reserved under "
                        "other cash-secured puts)."
                    ),
                )

        # --- Tier 0 risk gate (audit §8 item 3): REJECT -> 422 above; a
        # RESIZE reserves strike*100 for the APPROVED contracts only (the
        # collateral law above was checked for the larger request).
        assessment, risk_block = await _income_risk_gate(
            session,
            kind=kind,
            ticker=ticker,
            leg=leg,
            contracts=req.contracts,
            spot=spot,
        )
        approved = assessment.approved_quantity
        result = await _fill_short_open(
            session,
            kind=kind,
            ticker=ticker,
            leg=leg,
            contracts=approved,
            rationale=selection.rationale,
            collateral_position_id=None,
            cash_reserved=leg.strike * OPTION_MULTIPLIER * approved,
        )
        result["risk"] = risk_block
        return result


async def _all_open_csp_rows(session: AsyncSession) -> list[Position]:
    rows = await session.execute(
        select(Position).where(
            Position.status == POSITION_OPEN,
            Position.instrument == InstrumentType.CASH_SECURED_PUT.value,
        )
    )
    return list(rows.scalars().all())


async def _fill_short_open(
    session: AsyncSession,
    *,
    kind: str,
    ticker: str,
    leg,
    contracts: int,
    rationale: list[str],
    collateral_position_id: int | None,
    cash_reserved: float | None,
) -> dict:
    """Common venue fill + Position row + audits for both income kinds.

    V1 SCOPE: the SIMULATED venue only. The real-broker path exists at the
    adapter (collateral-attested submit_short_open_order) but its §26
    settle/sweep integration is the next chunk — an honest 422 names that
    instead of half-executing.
    """
    if not simulated_broker_mode():
        return await _fill_short_open_via_broker(
            session,
            kind=kind,
            ticker=ticker,
            leg=leg,
            contracts=contracts,
            rationale=rationale,
            collateral_position_id=collateral_position_id,
            cash_reserved=cash_reserved,
        )
    settings = get_settings()
    fill = leg.mid * (1.0 - settings.paper_slippage_bps / 10000.0)
    commission = settings.paper_commission_per_contract * contracts
    credit = contracts * fill * OPTION_MULTIPLIER - commission

    portfolio = await get_or_create_portfolio(session)
    portfolio.cash += credit
    portfolio.updated_at = utcnow()

    occ = occ_option_symbol(ticker, leg.expiry, leg.strike, leg.right)
    order = Order(
        ticker=ticker,
        instrument=kind,
        side="SELL_TO_OPEN",
        quantity=contracts,
        fill_price=fill,
        commission=commission,
        status="FILLED",
        opt_expiry=leg.expiry.isoformat(),
        opt_strike=leg.strike,
        opt_right=leg.right,
    )
    session.add(order)

    position = Position(
        ticker=ticker,
        instrument=kind,
        quantity=contracts,
        avg_price=fill,  # CREDIT received per share
        # CSP: defined max loss = (strike - credit) * 100 * qty (stock to
        # zero); covered call: NO incremental defined loss beyond the stock
        # row already counted in heat — 0.0, documented.
        max_loss=(
            (leg.strike - fill) * OPTION_MULTIPLIER * contracts
            if kind == InstrumentType.CASH_SECURED_PUT.value
            else 0.0
        ),
        stop_distance=fill,  # credit basis for the 2x loss stop
        entry_edge=0.0,
        entry_bar_date=None,
        status=POSITION_OPEN,
        opt_expiry=leg.expiry.isoformat(),
        opt_strike=leg.strike,
        opt_right=leg.right,
        multiplier=OPTION_MULTIPLIER,
        collateral_position_id=collateral_position_id,
        cash_reserved=cash_reserved,
    )
    session.add(position)
    await session.flush()
    order.position_id = position.id

    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.ORDER_FILLED,
        entity_type="position",
        entity_id=str(position.id),
        details={
            "kind": kind,
            "ticker": ticker,
            "occ_symbol": occ,
            "contracts": contracts,
            "credit_per_share": fill,
            "net_credit": credit,
            "collateral_position_id": collateral_position_id,
            "cash_reserved": cash_reserved,
            "rationale": rationale,
            "venue": "simulated",
        },
    )
    await session.commit()
    await session.refresh(position)
    return {
        "position": {
            "id": position.id,
            "ticker": ticker,
            "instrument": kind,
            "contracts": contracts,
            "occ_symbol": occ,
            "credit_per_share": fill,
            "net_credit": credit,
            "collateral_position_id": collateral_position_id,
            "cash_reserved": cash_reserved,
            "rationale": rationale,
        },
    }


async def _fill_short_open_via_broker(
    session: AsyncSession,
    *,
    kind: str,
    ticker: str,
    leg,
    contracts: int,
    rationale: list[str],
    collateral_position_id: int | None,
    cash_reserved: float | None,
) -> dict:
    """Real-broker short open, §11 lifecycle: T1 — durable PENDING_SUBMIT
    order row + request audit BEFORE the submit leaves the process; T2 —
    the settled truth (broker's fill/status), the Position row and the
    outcome audits. No local cash mutation: the credit landed in the REAL
    account. The collateral attestation travels to the adapter
    (covered_by), whose OCC gate + paper guard are the last line."""
    occ = occ_option_symbol(ticker, leg.expiry, leg.strike, leg.right)
    covered_by = (
        f"position:{collateral_position_id}"
        if collateral_position_id is not None
        else f"cash:{cash_reserved:.2f}"
    )
    broker = resolve_broker()
    client_order_id = new_client_order_id("sto")

    order = Order(
        client_order_id=client_order_id,
        ticker=ticker,
        instrument=kind,
        side="SELL_TO_OPEN",
        quantity=contracts,
        fill_price=0.0,
        commission=0.0,
        status="PENDING_SUBMIT",
        opt_expiry=leg.expiry.isoformat(),
        opt_strike=leg.strike,
        opt_right=leg.right,
    )
    session.add(order)
    await session.flush()
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "kind": kind,
            "ticker": ticker,
            "occ_symbol": occ,
            "side": "SELL_TO_OPEN",
            "contracts": contracts,
            "covered_by": covered_by,
            "client_order_id": client_order_id,
            "venue": get_settings().broker_provider,
        },
    )
    await session.commit()  # T1: durable before the network

    try:
        broker_order, adopted = await submit_short_open_and_poll(
            broker, client_order_id, occ, contracts, covered_by
        )
    except BrokerRejected as exc:
        order.status = "REJECTED"
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_REJECTED,
            entity_type="order",
            entity_id=str(order.id),
            details={"reason": str(exc), "rejected_by": "broker"},
        )
        await session.commit()
        raise HTTPException(
            status_code=422, detail=f"the broker rejected the short open: {exc}"
        ) from exc
    except BrokerError as exc:
        await session.commit()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "BROKER_ERROR",
                "message": (
                    f"the broker call failed: {exc}. The order may or may "
                    "not exist at the broker; the PENDING_SUBMIT row is "
                    "durable and GET /api/broker/reconcile surfaces the "
                    "truth — do not blindly retry."
                ),
            },
        ) from exc

    filled = broker_order.filled_quantity
    fill_price = broker_order.filled_avg_price
    order.status = broker_order.status
    order.broker_order_id = broker_order.broker_order_id or None
    order.filled_quantity = filled
    order.fill_price = fill_price if fill_price is not None else 0.0
    await session.flush()

    position = None
    if filled > 0 and fill_price is not None:
        position = Position(
            ticker=ticker,
            instrument=kind,
            quantity=filled,
            avg_price=fill_price,  # CREDIT per share, the broker's own
            max_loss=(
                (leg.strike - fill_price) * OPTION_MULTIPLIER * filled
                if kind == InstrumentType.CASH_SECURED_PUT.value
                else 0.0
            ),
            stop_distance=fill_price,
            entry_edge=0.0,
            entry_bar_date=None,
            status=POSITION_OPEN,
            opt_expiry=leg.expiry.isoformat(),
            opt_strike=leg.strike,
            opt_right=leg.right,
            multiplier=OPTION_MULTIPLIER,
            collateral_position_id=collateral_position_id,
            cash_reserved=cash_reserved,
        )
        session.add(position)
        await session.flush()
        order.position_id = position.id

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED if position else AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "kind": kind,
            "occ_symbol": occ,
            "rationale": rationale,
            "outcome": (
                "position opened at the broker's credit"
                if position
                else "no full fill applied yet — GET /api/broker/reconcile "
                "surfaces the truth"
            ),
            **broker_order_details(broker_order, adopted=adopted),
        },
    )
    await session.commit()
    return {
        "position": (
            {
                "id": position.id,
                "ticker": ticker,
                "instrument": kind,
                "contracts": position.quantity,
                "occ_symbol": occ,
                "credit_per_share": position.avg_price,
                "net_credit": position.avg_price * position.quantity * OPTION_MULTIPLIER,
                "collateral_position_id": collateral_position_id,
                "cash_reserved": cash_reserved,
                "rationale": rationale,
            }
            if position
            else None
        ),
        "order_status": broker_order.status,
    }


@router.post("/{position_id}/buyback")
async def buyback(
    position_id: int,
    req: BuybackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Buy back an open income short leg (BUY_TO_CLOSE) — always allowed
    (risk-reducing; the §18 kill switch does not block it). Releases the
    collateral (share pin / cash reservation) in the same transaction."""
    require_market_data_provider()
    require_broker()
    async with execution_lock():
        position = await session.get(Position, position_id)
        if (
            position is None
            or position.status != POSITION_OPEN
            or position.instrument
            not in (
                InstrumentType.COVERED_CALL.value,
                InstrumentType.CASH_SECURED_PUT.value,
            )
        ):
            raise HTTPException(
                status_code=404,
                detail=f"no OPEN income position #{position_id}",
            )
        if not simulated_broker_mode():
            occ = occ_option_symbol(
                position.ticker,
                date.fromisoformat(position.opt_expiry or ""),
                position.opt_strike or 0.0,
                position.opt_right or "C",
            )
            broker = resolve_broker()
            client_order_id = new_client_order_id(f"btc-{position.id}")
            order = Order(
                client_order_id=client_order_id,
                ticker=position.ticker,
                instrument=position.instrument,
                side="BUY_TO_CLOSE",
                quantity=position.quantity,
                fill_price=0.0,
                commission=0.0,
                status="PENDING_SUBMIT",
                opt_expiry=position.opt_expiry,
                opt_strike=position.opt_strike,
                opt_right=position.opt_right,
                position_id=position.id,
            )
            session.add(order)
            await session.flush()
            await session.commit()  # T1: durable before the network
            try:
                broker_order, adopted = await submit_short_close_and_poll(
                    broker, client_order_id, occ, position.quantity
                )
            except BrokerRejected as exc:
                order.status = "REJECTED"
                await session.commit()
                raise HTTPException(
                    status_code=422,
                    detail=f"the broker rejected the buyback: {exc}",
                ) from exc
            except BrokerError as exc:
                await session.commit()
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "BROKER_ERROR",
                        "message": (
                            f"the broker call failed: {exc}; the "
                            "PENDING_SUBMIT row is durable — "
                            "GET /api/broker/reconcile surfaces the truth."
                        ),
                    },
                ) from exc
            filled = broker_order.filled_quantity
            fill_price = broker_order.filled_avg_price
            order.status = broker_order.status
            order.broker_order_id = broker_order.broker_order_id or None
            order.filled_quantity = filled
            order.fill_price = fill_price if fill_price is not None else 0.0
            realized = None
            if filled >= position.quantity and fill_price is not None:
                realized = (
                    (position.avg_price - fill_price)
                    * position.quantity
                    * OPTION_MULTIPLIER
                )
                position.realized_pnl = (position.realized_pnl or 0.0) + realized
                position.quantity = 0
                position.status = "CLOSED"
                position.closed_at = utcnow()
                position.max_loss = 0.0
            await audit.record(
                session,
                actor_type=ActorType.USER,
                actor_id=CURRENT_USER,
                action=AuditAction.ORDER_FILLED,
                entity_type="position",
                entity_id=str(position.id),
                details={
                    "kind": position.instrument,
                    "action": "BUY_TO_CLOSE",
                    "realized_pnl": realized,
                    "reason": req.reason,
                    **broker_order_details(broker_order, adopted=adopted),
                },
            )
            await session.commit()
            return {
                "position_id": position.id,
                "status": position.status,
                "realized_pnl": realized,
                "buyback_price": fill_price,
                "reference_source": "broker fill",
            }

        spot = await _last_stored_close(session, position.ticker)
        if spot is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"no stored bars for {position.ticker} — no reference "
                    "price to buy back at (honest error, §44 rule 18)."
                ),
            )
        _, chain = build_option_chain(position.ticker, spot, max_age_seconds=0.0)
        contract = find_option_contract(chain, position)
        if contract is not None:
            reference, source = contract.mid, "chain mid"
        else:
            # Intrinsic fallback (e.g. expired off the chain) — documented.
            strike = position.opt_strike or 0.0
            if (position.opt_right or "C") == "C":
                reference = max(spot - strike, 0.0)
            else:
                reference = max(strike - spot, 0.0)
            source = "intrinsic (contract missing from today's chain)"

        settings = get_settings()
        fill = reference * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_contract * position.quantity
        cost = position.quantity * fill * OPTION_MULTIPLIER + commission
        realized = (
            (position.avg_price - fill)
            * position.quantity
            * OPTION_MULTIPLIER
            - commission
        )

        portfolio = await get_or_create_portfolio(session)
        portfolio.cash -= cost
        portfolio.updated_at = utcnow()

        order = Order(
            ticker=position.ticker,
            instrument=position.instrument,
            side="BUY_TO_CLOSE",
            quantity=position.quantity,
            fill_price=fill,
            commission=commission,
            status="FILLED",
            opt_expiry=position.opt_expiry,
            opt_strike=position.opt_strike,
            opt_right=position.opt_right,
            position_id=position.id,
        )
        session.add(order)

        position.realized_pnl = (position.realized_pnl or 0.0) + realized
        position.quantity = 0
        position.status = "CLOSED"
        position.closed_at = utcnow()
        position.max_loss = 0.0
        # Collateral release happens by construction: a CLOSED row no longer
        # pins shares (pinned_shares filters OPEN) nor reserves cash.

        await audit.record(
            session,
            actor_type=ActorType.USER,
            actor_id=CURRENT_USER,
            action=AuditAction.ORDER_FILLED,
            entity_type="position",
            entity_id=str(position.id),
            details={
                "kind": position.instrument,
                "action": "BUY_TO_CLOSE",
                "reference_price": reference,
                "reference_source": source,
                "fill": fill,
                "realized_pnl": realized,
                "reason": req.reason,
                "venue": "simulated",
            },
        )
        await session.commit()
        return {
            "position_id": position.id,
            "status": "CLOSED",
            "realized_pnl": realized,
            "buyback_price": fill,
            "reference_source": source,
        }
