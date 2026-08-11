"""Portfolio risk state API (development plan §12.5, §13, §14, §16, §35).

``GET /api/portfolio/risk`` reports the paper portfolio exactly as the risk
engine sees it: NAV, cash versus the regime cash floor (plan §13), portfolio
heat and its state (plan §12.5), correlation-bucket utilization (plan §12.4)
— the STATIC config buckets plus DYNAMIC buckets measured from stored bars —
the aggregated portfolio Greeks against their §16 limits, the §14
vol-targeting read, and the active risk limits. Every number is computed with
the SAME ``libs.trading_core`` helpers (``risk``, ``greeks``, ``allocation``,
``correlation``) and :class:`RiskLimits` defaults the order path uses, so
this view can never disagree with the engine.

Position market values come from the last STORED daily close for stock; an
option position is carried at its premium BOOK value (``qty * avg_price *
multiplier`` — the documented V1 approximation, exact for heat purposes
because a long option's max_loss IS that premium, §12.1), matching
:func:`position_market_value`, which the order path's risk snapshot also
uses. A stock position whose ticker has no stored bars surfaces
``market_price``/``market_value`` as null with a ``DATA_ISSUE`` note (honest
nulls, plan §44 rule 18) and contributes 0 toward NAV — the gap is shown,
never papered over.

Per-position Greeks (plan §16): stock rows contribute delta 1.0 / gamma 0 /
theta 0 / vega 0 per share at spot = last stored close; option rows locate
the SAME contract in today's regenerated chain (the shared §9 helper in
routers/options.py) and use its per-share greeks. A position whose greeks
cannot be known — no stored bars, or a contract missing from today's chain
(e.g. expired) — is reported with ``data_ok: false`` and contributes ZEROS
to the totals, with the gap named in its ``note``.

Read-only: the endpoint changes no state and makes no decision, so it writes
no audit event (rule 12 covers state changes and decisions).
"""
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.allocation import VolTargetParams, exposure_multiplier
from libs.trading_core.contracts import ContractQuote
from libs.trading_core.correlation import build_dynamic_buckets
from libs.trading_core.features import realized_vol
from libs.trading_core.greeks import (
    PortfolioGreeks,
    PositionGreeksInput,
    aggregate_greeks,
)
from libs.trading_core.models import InstrumentType
from libs.trading_core.risk import PositionRisk, RiskLimits, heat_state, portfolio_heat

from ..db import (
    Position,
    StockBarDaily,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
)
from .analysis import market_regime_from_spy
from .options import REALIZED_VOL_PERIOD, build_option_chain

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"

# Option instruments (the §8 matrix outputs that carry an opt_* contract).
_OPTION_INSTRUMENTS = frozenset(
    {InstrumentType.LONG_CALL.value, InstrumentType.LONG_PUT.value}
)

# --- Parameter seams (plan §6.2: parameters, never hardcoded truths) --------
# §14 vol-targeting parameters shared by this view and the order gate chain.
VOL_TARGET_PARAMS = VolTargetParams()
# §12.4 dynamic-bucket thresholds ("thresholds require validation" — these
# defaults mirror libs.trading_core.correlation and are starting points).
DYNAMIC_BUCKET_THRESHOLD = 0.70
DYNAMIC_BUCKET_WINDOW = 60


def is_option_position(position: Position) -> bool:
    """True when `position` holds a long option (LONG_CALL / LONG_PUT)."""
    return position.instrument in _OPTION_INSTRUMENTS


def position_market_value(position: Position, price: float | None) -> float | None:
    """One OPEN position's market value for the NAV / risk snapshot (§12.5).

    Stock: ``quantity * last stored close`` (``None`` when the ticker has no
    stored bars — honest null, the caller counts 0 toward NAV). Options:
    ``quantity * avg_price * multiplier`` — the premium PAID (book value), a
    documented V1 approximation until option marks are wired into the
    portfolio view; it is exact for heat purposes because a long option's
    max_loss IS that premium (§12.1). Shared by this view and the order
    path's risk snapshot so both build the identical portfolio picture.
    """
    if is_option_position(position):
        return position.quantity * position.avg_price * (position.multiplier or 1)
    if price is None:
        return None
    return position.quantity * price


def find_option_contract(
    chain: list[ContractQuote], position: Position
) -> ContractQuote | None:
    """Locate `position`'s EXACT contract (expiry + strike + right) in a
    freshly regenerated chain; ``None`` when it is no longer quoted (e.g.
    expired off the chain) — callers fall back honestly (intrinsic value for
    closes, zero-with-note for greeks)."""
    for c in chain:
        if (
            c.expiry.isoformat() == position.opt_expiry
            and c.right == position.opt_right
            and position.opt_strike is not None
            and abs(c.strike - position.opt_strike) < 1e-9
        ):
            return c
    return None


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


async def stored_closes_by_ticker(
    session: AsyncSession, tickers: Iterable[str]
) -> dict[str, list[float]]:
    """Stored daily closes (oldest first) per ticker — the input series for
    the §12.4 dynamic buckets and the §14 RV20 forecast. Reads only what is
    already stored (no lazy backfill: this is a portfolio read, not a data
    request)."""
    out: dict[str, list[float]] = {}
    for ticker in sorted(set(tickers)):
        rows = await session.execute(
            select(StockBarDaily.close)
            .where(StockBarDaily.ticker == ticker)
            .order_by(StockBarDaily.ts)
        )
        out[ticker] = list(rows.scalars().all())
    return out


def portfolio_greeks_read(
    pairs: list[tuple[Position, float | None]],
) -> tuple[PortfolioGreeks, list[dict]]:
    """Aggregate portfolio Greeks + per-position rows for OPEN positions (§16).

    Stock rows contribute per-share delta 1.0 / gamma 0 / theta 0 / vega 0 at
    spot = last stored close. Option rows locate the SAME contract in today's
    regenerated chain (shared §9 helper, routers/options.py) for their
    per-share greeks. A position whose greeks cannot be known — no stored
    bars (spot unknown) or a contract missing from today's chain (e.g.
    expired) — gets ``data_ok: false`` and contributes ZEROS to the totals,
    with the gap named in ``note`` (honest, §44 rule 18). Aggregation is
    :func:`libs.trading_core.greeks.aggregate_greeks` — never reimplemented
    here (plan §21). Shared by this view and the order gate chain so the risk
    engine judges the same book the user sees.
    """
    inputs: list[PositionGreeksInput] = []
    rows: list[dict] = []
    ok_row_indexes: list[int] = []
    chains: dict[str, list[ContractQuote]] = {}  # one chain build per ticker
    for pos, price in pairs:
        note: str | None = None
        per_share: tuple[float, float, float, float] | None = None
        if price is None:
            note = (
                f"DATA_ISSUE: no stored bars for {pos.ticker} — spot unknown; "
                "greeks contribute zeros"
            )
        elif is_option_position(pos):
            if pos.ticker not in chains:
                _, chains[pos.ticker] = build_option_chain(pos.ticker, price)
            contract = find_option_contract(chains[pos.ticker], pos)
            if contract is None:
                note = (
                    f"DATA_ISSUE: contract {pos.opt_right} {pos.opt_strike} "
                    f"exp {pos.opt_expiry} missing from today's chain (e.g. "
                    "expired); greeks contribute zeros"
                )
            else:
                per_share = (
                    contract.delta,
                    contract.gamma,
                    contract.theta,
                    contract.vega,
                )
        else:
            per_share = (1.0, 0.0, 0.0, 0.0)

        if per_share is None:
            rows.append(
                {
                    "ticker": pos.ticker,
                    "instrument": pos.instrument,
                    "equivalent_shares": 0.0,
                    "delta_notional_usd": 0.0,
                    "gamma": 0.0,
                    "theta_usd_per_day": 0.0,
                    "vega_usd": 0.0,
                    "data_ok": False,
                    "note": note,
                }
            )
        else:
            delta, gamma, theta, vega = per_share
            ok_row_indexes.append(len(rows))
            rows.append({})  # filled from the aggregate's contribution below
            inputs.append(
                PositionGreeksInput(
                    ticker=pos.ticker,
                    instrument=pos.instrument,
                    quantity=pos.quantity,
                    multiplier=pos.multiplier or 1,
                    spot=price,
                    delta=delta,
                    gamma=gamma,
                    theta_per_day=theta,
                    vega=vega,
                )
            )

    greeks = aggregate_greeks(inputs)
    for idx, c in zip(ok_row_indexes, greeks.per_position):
        rows[idx] = {
            "ticker": c.ticker,
            "instrument": c.instrument,
            "equivalent_shares": c.delta_shares,
            "delta_notional_usd": c.delta_notional,
            "gamma": c.gamma,
            "theta_usd_per_day": c.theta_per_day,
            "vega_usd": c.vega,
            "data_ok": True,
            "note": None,
        }
    return greeks, rows


def vol_targeting_block(
    nav: float,
    pairs: list[tuple[Position, float | None]],
    closes_by_ticker: Mapping[str, list[float]],
) -> dict:
    """The §14 vol-targeting read: forecast vol -> exposure multiplier.

    CRUDE v0 FORECAST PROXY (documented; plan §14): ``forecast_vol`` is the
    NAV-weighted average of each open position's underlying RV20
    (:func:`libs.trading_core.features.realized_vol` over stored closes),
    each weight being the position's market value / NAV — so CASH implicitly
    weighs in at ZERO vol, damping the forecast for a mostly-cash book. A
    position whose market value is unknown (no stored bars) or whose RV20 is
    not computable (short history) also weighs in at zero. With no open
    positions the forecast is an honest ``None`` -> multiplier 1.0 (no
    adjustment, never a guess). A real portfolio-vol forecast (correlations,
    option deltas) replaces this proxy later.

    The multiplier is :func:`libs.trading_core.allocation.exposure_multiplier`
    and feeds ``assess(budget_multiplier=...)``, which hard-caps the budget
    at ``abs_max_trade_risk`` — vol targeting can NEVER override hard risk
    caps (§14, §44 rule 20). Shared by this view and the order gate chain so
    the reported multiplier IS the one the risk engine applies.
    """
    params = VOL_TARGET_PARAMS
    forecast: float | None = None
    if pairs:
        weighted = 0.0
        any_rv = False
        for pos, price in pairs:
            value = position_market_value(pos, price)
            if value is None:
                continue  # no stored bars: counts 0 toward NAV, zero weight
            closes = closes_by_ticker.get(pos.ticker, [])
            rv = (
                realized_vol(closes, period=REALIZED_VOL_PERIOD)[-1]
                if closes
                else None
            )
            if rv is None:
                continue  # RV20 not computable: zero-vol weight (documented)
            weighted += (value / nav) * rv
            any_rv = True
        forecast = weighted if any_rv else None
    multiplier = exposure_multiplier(forecast, params)
    if pairs:
        note = (
            "crude v0 proxy: NAV-weighted average of per-ticker RV20 over "
            "open positions; cash (and any position without a computable "
            "RV20) weighs in at zero vol. Scales the §12.2 tier budget only "
            "— hard caps are never overridden (§14)."
        )
    else:
        note = (
            "no open positions — no vol forecast (honest null); "
            "multiplier 1.0 means no adjustment (§14)"
        )
    return {
        "target_vol": params.target_vol,
        "forecast_vol": forecast,
        "multiplier": multiplier,
        "max_multiplier": params.max_multiplier,
        "note": note,
    }


@router.get("/risk")
async def get_portfolio_risk(session: AsyncSession = Depends(get_session)) -> dict:
    """Current portfolio risk state (plan §12.5, §13, §14, §16, §35). Read-only."""
    limits = RiskLimits()
    portfolio = await get_or_create_portfolio(session)
    state = await get_or_create_system_state(session)
    # Broad-market regime from SPY via the shared helper (plan §6.1, ADR-005).
    regime = (await market_regime_from_spy(session)).classification

    pairs = await open_positions_with_prices(session)
    # NAV is derived, never stored: cash + market value of OPEN positions
    # (options at premium book value, §12.1 — see position_market_value);
    # bar-less stock counts 0 (surfaced as DATA_ISSUE below).
    values = [position_market_value(pos, price) for pos, price in pairs]
    nav = portfolio.cash + sum(v for v in values if v is not None)
    position_risks = [
        PositionRisk(
            ticker=pos.ticker,
            market_value=value if value is not None else 0.0,
            max_loss=pos.max_loss,
        )
        for (pos, _price), value in zip(pairs, values)
    ]

    # Heat and its state from the SAME risk-lib helpers the engine uses (§12.5).
    heat = portfolio_heat(position_risks, nav)
    heat_risk_usd = sum(pos.max_loss for pos, _ in pairs)
    max_new_risk_usd = max(0.0, limits.heat_reject * nav - heat_risk_usd)

    positions_out = []
    for (pos, price), value in zip(pairs, values):
        positions_out.append(
            {
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "market_price": price,
                "market_value": value,
                "max_loss": pos.max_loss,
                "opened_at": pos.opened_at.isoformat(),
                # Honest data gap (plan §44 rule 18): no stored bars for this
                # ticker, so its market price is unknown; a bar-less STOCK
                # position also counts 0 to NAV.
                "note": None if price is not None else "DATA_ISSUE",
            }
        )

    # §12.4 buckets: STATIC config rows first (RiskLimits), then DYNAMIC rows
    # from connected components of rolling-60d correlation > 0.70 among the
    # open-position tickers' stored closes (libs.trading_core.correlation —
    # never reimplemented here). Both kinds are measured against the SAME
    # bucket_risk cap.
    buckets_out = []
    for name, members in limits.correlation_buckets.items():
        bucket_risk_usd = sum(
            pos.max_loss for pos, _ in pairs if pos.ticker in members
        )
        bucket_risk_pct = bucket_risk_usd / nav
        buckets_out.append(
            {
                "name": name,
                "kind": "STATIC",
                "tickers": list(members),
                "risk_usd": bucket_risk_usd,
                "risk_pct": bucket_risk_pct,
                "cap_pct": limits.bucket_risk,
                "utilization_pct": bucket_risk_pct / limits.bucket_risk,
            }
        )
    closes_by_ticker = await stored_closes_by_ticker(
        session, (pos.ticker for pos, _ in pairs)
    )
    for members in build_dynamic_buckets(
        closes_by_ticker,
        threshold=DYNAMIC_BUCKET_THRESHOLD,
        window=DYNAMIC_BUCKET_WINDOW,
    ):
        tickers = sorted(members)
        bucket_risk_usd = sum(
            pos.max_loss for pos, _ in pairs if pos.ticker in members
        )
        bucket_risk_pct = bucket_risk_usd / nav
        buckets_out.append(
            {
                "name": "DYNAMIC:" + "+".join(tickers),
                "kind": "DYNAMIC",
                "tickers": tickers,
                "risk_usd": bucket_risk_usd,
                "risk_pct": bucket_risk_pct,
                "cap_pct": limits.bucket_risk,
                "utilization_pct": bucket_risk_pct / limits.bucket_risk,
            }
        )

    # §16 portfolio greeks vs their limits; breaches carry the real numbers.
    greeks, greeks_rows = portfolio_greeks_read(pairs)
    delta_notional = greeks.delta_adjusted_notional
    breaches: list[str] = []
    delta_cap = limits.max_delta_notional_pct_nav * nav
    if abs(delta_notional) > delta_cap:
        breaches.append(
            f"|delta-adjusted notional| ${abs(delta_notional):,.2f} "
            f"({abs(delta_notional) / nav:.2%} of NAV) exceeds the "
            f"{limits.max_delta_notional_pct_nav:.2%}-of-NAV limit "
            f"(${delta_cap:,.2f})"
        )
    theta_cap = limits.max_net_theta_pct_nav * nav
    if abs(greeks.net_theta_per_day) > theta_cap:
        breaches.append(
            f"|net theta| ${abs(greeks.net_theta_per_day):,.2f}/day "
            f"({abs(greeks.net_theta_per_day) / nav:.4%} of NAV) exceeds the "
            f"{limits.max_net_theta_pct_nav:.4%}-of-NAV limit "
            f"(${theta_cap:,.2f}/day)"
        )
    vega_cap = limits.max_net_vega_pct_nav * nav
    if abs(greeks.net_vega) > vega_cap:
        breaches.append(
            f"|net vega| ${abs(greeks.net_vega):,.2f} per IV point "
            f"({abs(greeks.net_vega) / nav:.2%} of NAV) exceeds the "
            f"{limits.max_net_vega_pct_nav:.2%}-of-NAV limit "
            f"(${vega_cap:,.2f})"
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
        "greeks": {
            "net_delta_shares": greeks.net_delta_shares,
            "delta_adjusted_notional_usd": delta_notional,
            "delta_notional_pct_nav": delta_notional / nav,
            "net_gamma": greeks.net_gamma,
            "net_theta_usd_per_day": greeks.net_theta_per_day,
            "net_vega_usd": greeks.net_vega,
            "limits": {
                "max_delta_notional_pct_nav": limits.max_delta_notional_pct_nav,
                "max_net_theta_pct_nav": limits.max_net_theta_pct_nav,
                "max_net_vega_pct_nav": limits.max_net_vega_pct_nav,
            },
            "breaches": breaches,
            "per_position": greeks_rows,
        },
        "vol_targeting": vol_targeting_block(nav, pairs, closes_by_ticker),
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
