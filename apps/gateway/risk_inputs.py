"""Shared risk-engine inputs for every WRITE path (risk-engine audit §8
item 3, §10 Phase B0; plan §17, §21).

The :class:`~libs.trading_core.risk.PortfolioSnapshot` the risk engine
decides against is built HERE, once, for the stock/option order path
(routers/orders.py) and the income opens (routers/income.py), so both judge
the identical book — the same helpers the portfolio risk view uses
(``open_positions_with_prices`` / ``position_market_value`` /
``market_regime_from_spy``, plan §21) and never a re-implementation.

Collateral already pledged to open cash-secured puts is NOT deployable
(audit §8 item 3): the snapshot's ``cash`` is ``usable_cash = cash − Σ
cash_reserved`` over the OPEN CASH_SECURED_PUT rows, so the §13 cash floor
measures capital that can actually be deployed. NAV is NOT netted:
``nav = cash + Σ market values`` — pledged collateral is still the
account's asset (the short put's negative market value already sits in Σ
market values), and this is the SAME NAV the portfolio risk view reports,
so every %-of-NAV cap and the §14 vol-targeting multiplier see one NAV per
book (QA finding, 2026-08-17). ``cash`` is the caller's account cash — the
broker's LIVE cash in broker mode, the simulator's ledger in simulated mode
(§14: broker CASH, never buying power; the platform stores no copy of a
real account).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.risk import PortfolioSnapshot, PositionRisk

from .db import Position
from .routers.analysis import market_regime_from_spy
from .routers.portfolio import (
    open_csp_cash_reserved,
    open_positions_with_prices,
    position_market_value,
)


@dataclass
class SnapshotInputs:
    """The risk snapshot plus the intermediates callers keep using.

    - ``snapshot``: the :class:`PortfolioSnapshot` for ``assess`` /
      ``assess_income``.
    - ``pairs`` / ``values``: the OPEN positions with their last stored close
      and per-position market values (the §14 vol-targeting and §16 greek
      helpers consume these; None value = honest data gap counted as 0).
    - ``cash_reserved_total``: Σ ``cash_reserved`` over OPEN CASH_SECURED_PUT
      rows — collateral pledged to open puts.
    - ``usable_cash``: ``cash − cash_reserved_total`` — the deployable cash
      the snapshot carries as ``snapshot.cash`` (NAV itself is un-netted).
    """

    snapshot: PortfolioSnapshot
    pairs: list[tuple[Position, float | None]]
    values: list[float | None]
    cash_reserved_total: float
    usable_cash: float


async def build_portfolio_snapshot(
    session: AsyncSession, *, cash: float, trading_enabled: bool
) -> SnapshotInputs:
    """Build the risk engine's :class:`PortfolioSnapshot` from the OPEN book
    (plan §12.5, §13, §17; audit §10 Phase B0).

    ``cash`` is the account cash the caller already resolved (broker LIVE
    cash or the simulator ledger, §14). Open CSP collateral is netted out of
    the DEPLOYABLE cash (``usable_cash = cash − Σ cash_reserved``); NAV =
    cash + Σ market values, un-netted, identical to the risk view (options
    at book value, bar-less stock counts 0 — see ``position_market_value``). The regime is the SPY read shared with the
    portfolio risk view (plan §6.1); ``trading_enabled`` mirrors the kill
    switch (plan §18).
    """
    pairs = await open_positions_with_prices(session)
    values = [position_market_value(pos, price) for pos, price in pairs]
    cash_reserved_total = open_csp_cash_reserved(pairs)
    usable_cash = cash - cash_reserved_total
    # NAV is un-netted (pledged collateral is still an asset; the view
    # reports the same number) — only the deployable cash is netted.
    nav = cash + math.fsum(v for v in values if v is not None)
    position_risks = [
        PositionRisk(
            ticker=pos.ticker,
            market_value=value if value is not None else 0.0,
            max_loss=pos.max_loss,
        )
        for (pos, _price), value in zip(pairs, values)
    ]
    spy_regime = (await market_regime_from_spy(session)).classification
    snapshot = PortfolioSnapshot(
        nav=nav,
        cash=usable_cash,
        positions=position_risks,
        regime=spy_regime,
        trading_enabled=trading_enabled,
    )
    return SnapshotInputs(
        snapshot=snapshot,
        pairs=pairs,
        values=values,
        cash_reserved_total=cash_reserved_total,
        usable_cash=usable_cash,
    )
