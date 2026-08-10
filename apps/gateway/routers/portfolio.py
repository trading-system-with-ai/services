"""Portfolio risk state API (development plan §12.5, §13, §35).

``GET /api/portfolio/risk`` reports the paper portfolio exactly as the risk
engine sees it: NAV, cash versus the regime cash floor (plan §13), portfolio
heat and its state (plan §12.5), correlation-bucket utilization (plan §12.4)
and the active risk limits. Every number is computed with the SAME
``libs.trading_core.risk`` helpers and :class:`RiskLimits` defaults the order
path uses, so this view can never disagree with the engine.

Position market values come from the last STORED daily close. A position whose
ticker has no stored bars surfaces ``market_price``/``market_value`` as null
with a ``DATA_ISSUE`` note (honest nulls, plan §44 rule 18) and contributes 0
toward NAV — the gap is shown, never papered over.

Read-only: the endpoint changes no state and makes no decision, so it writes
no audit event (rule 12 covers state changes and decisions).
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.risk import PositionRisk, RiskLimits, heat_state, portfolio_heat

from ..db import (
    Position,
    StockBarDaily,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
)
from .analysis import market_regime_from_spy

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"


async def open_positions_with_prices(
    session: AsyncSession,
) -> list[tuple[Position, float | None]]:
    """OPEN positions (ticker-ordered) paired with the last stored daily close.

    The price is ``None`` — an honest null (plan §44 rule 18) — when the
    ticker has no stored bars; callers surface that as a DATA_ISSUE and count
    0 toward NAV. Shared by the portfolio risk view and the order preview
    path so both build the identical portfolio picture.
    """
    rows = await session.execute(
        select(Position)
        .where(Position.status == POSITION_OPEN)
        .order_by(Position.ticker, Position.id)
    )
    out: list[tuple[Position, float | None]] = []
    for pos in rows.scalars().all():
        price = (
            await session.execute(
                select(StockBarDaily.close)
                .where(StockBarDaily.ticker == pos.ticker)
                .order_by(StockBarDaily.ts.desc())
                .limit(1)
            )
        ).scalars().first()
        out.append((pos, price))
    return out


@router.get("/risk")
async def get_portfolio_risk(session: AsyncSession = Depends(get_session)) -> dict:
    """Current portfolio risk state (plan §12.5, §13, §35). Read-only."""
    limits = RiskLimits()
    portfolio = await get_or_create_portfolio(session)
    state = await get_or_create_system_state(session)
    # Broad-market regime from SPY via the shared helper (plan §6.1, ADR-005).
    regime = (await market_regime_from_spy(session)).classification

    pairs = await open_positions_with_prices(session)
    # NAV is derived, never stored: cash + market value of OPEN positions;
    # bar-less tickers count 0 (surfaced as DATA_ISSUE below).
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

    # Heat and its state from the SAME risk-lib helpers the engine uses (§12.5).
    heat = portfolio_heat(position_risks, nav)
    heat_risk_usd = sum(pos.max_loss for pos, _ in pairs)
    max_new_risk_usd = max(0.0, limits.heat_reject * nav - heat_risk_usd)

    positions_out = []
    for pos, price in pairs:
        positions_out.append(
            {
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "market_price": price,
                "market_value": (pos.quantity * price) if price is not None else None,
                "max_loss": pos.max_loss,
                "opened_at": pos.opened_at.isoformat(),
                # Honest data gap (plan §44 rule 18): no stored bars for this
                # ticker, so its market value is unknown and counts 0 to NAV.
                "note": None if price is not None else "DATA_ISSUE",
            }
        )

    buckets_out = []
    for name, members in limits.correlation_buckets.items():
        bucket_risk_usd = sum(
            pos.max_loss for pos, _ in pairs if pos.ticker in members
        )
        bucket_risk_pct = bucket_risk_usd / nav
        buckets_out.append(
            {
                "name": name,
                "tickers": list(members),
                "risk_usd": bucket_risk_usd,
                "risk_pct": bucket_risk_pct,
                "cap_pct": limits.bucket_risk,
                "utilization_pct": bucket_risk_pct / limits.bucket_risk,
            }
        )

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "nav": nav,
        "cash": portfolio.cash,
        "cash_pct": portfolio.cash / nav,
        "market_regime": regime.value,
        "cash_floor_pct": limits.cash_floors[regime],
        "trading_enabled": state.trading_enabled,
        "portfolio_heat_pct": heat,
        "heat_state": heat_state(heat, limits),
        "max_new_risk_usd": max_new_risk_usd,
        "max_new_risk_pct": max_new_risk_usd / nav,
        "positions": positions_out,
        "buckets": buckets_out,
        "limits": {
            "single_name_risk_pct": limits.single_name_risk,
            "single_name_capital_pct": limits.single_name_capital,
            "bucket_risk_pct": limits.bucket_risk,
            "heat_elevated_pct": limits.heat_elevated,
            "heat_high_pct": limits.heat_high,
            "heat_reject_pct": limits.heat_reject,
            "abs_max_trade_risk_pct": limits.abs_max_trade_risk,
        },
    }
