"""THE §42 REPLAY TEST — a day-by-day lifecycle replay over stored bars.

A 320-bar deterministic synthetic series (steady uptrend, then a sharp
breakdown) is written STRAIGHT into ``stock_bars_daily`` — the same table
``ensure_daily_bars`` reads — so every endpoint sees the advancing history
naturally. The first 260 bars are seeded up front; the last 60 replay one
bar per "day": insert the bar, preview, approve when (and only when) the
whole §10 chain PASSes with no position open, then run the mechanical exit
check. The §42 invariants are asserted the whole way:

(a) never more than one OPEN position;
(b) an approve NEVER succeeds while a position is open (409) or when any
    gate fails (422) — attempted EVERY day;
(c) cash conservation each day: cash changes only on fill days, by exactly
    the audited fill amounts;
(d) every fill carries its ORDER_REQUESTED -> ORDER_SUBMITTED ->
    ORDER_FILLED audit chain with the fill numbers matching the order row;
(e) after force-closing any leftover position,
    ``final_cash == initial_cash + Σ realized_pnl`` over all CLOSED
    positions, to the cent;
(f) at least one entry AND at least one mechanical exit occurred.

Documented seams used (never behavior hacks):

- ``orders.MAX_BAR_AGE_DAYS`` is widened via its module-level parameter seam
  (its own docstring invites tests to substitute thresholds): replayed bars
  carry their real historical dates, which are legitimately "stale" against
  the wall clock mid-replay.
- Paper commissions are zeroed through the Settings parameters (plan §6.2:
  parameters, never truths) so invariant (e) holds EXACTLY as stated:
  ``realized_pnl`` deliberately books the open-side commission at open
  (documented in ``execute_sell_to_close``), so with nonzero commissions the
  identity would gain a Σ(open-side commissions) term. Slippage keeps its
  default — fills still move against the trader.
- Series shape: 290 uptrend bars (+0.6%/day drift, small deterministic
  wobble) then 30 breakdown bars (-3.5%/day). Verified against the real
  libs: the first replay day reads STRONG_BULL / BULL edge 66.7 / LONG_STOCK
  (the ticker's seeded base IV keeps the §7 regime off LOW, so the §8 BULL
  column can never map to an option here) -> a guaranteed entry; the
  breakdown then guarantees a mechanical exit (ATR_TRAIL/HARD_STOP family)
  within days, and the §42 flow is deterministic end to end.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from apps.gateway.execution import gate_chain
from sqlalchemy import select

from apps.gateway.db import Order, Position, SessionLocal, StockBarDaily
from apps.gateway.routers import orders as orders_router
from libs.common.config import get_settings

TICKER = "RPLY"

# --- Series parameters (deterministic; see module docstring) ----------------
TOTAL_BARS = 320
REPLAY_BARS = 60
UP_BARS = 290  # bars 0..289 rise; 290..319 break down
UP_DRIFT = 0.006
DOWN_DRIFT = -0.035
WOBBLE = 0.002  # keeps RV20 > 0 without ever flipping a day's direction
RANGE_PCT = 0.006
BASE_PRICE = 80.0
BASE_VOLUME = 1_000_000.0

OPTION_INSTRUMENTS = {"LONG_CALL", "LONG_PUT"}


def build_bars():
    """320 deterministic synthetic daily bars on the last 320 weekdays."""
    d = datetime.now(timezone.utc).date()
    dates = []
    while len(dates) < TOTAL_BARS:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()

    bars = []
    price = BASE_PRICE
    prev = price
    for t, ts in enumerate(dates):
        drift = UP_DRIFT if t < UP_BARS else DOWN_DRIFT
        price *= math.exp(drift + WOBBLE * math.sin(t / 3.0))
        close = round(price, 4)
        bars.append(
            dict(
                ts=ts,
                open=prev,
                high=round(max(prev, close) * (1 + RANGE_PCT), 4),
                low=round(min(prev, close) * (1 - RANGE_PCT), 4),
                close=close,
                volume=BASE_VOLUME + (50_000.0 if t % 2 == 0 else -50_000.0),
            )
        )
        prev = close
    return bars


async def insert_bar(bar):
    """Upsert one daily bar into the SAME table ensure_daily_bars reads.

    Dates are unique per ticker in this replay, so the upsert is a plain
    insert guarded by the (ticker, ts) UNIQUE constraint."""
    async with SessionLocal() as s:
        s.add(StockBarDaily(ticker=TICKER, **bar))
        await s.commit()


async def db_open_positions():
    async with SessionLocal() as s:
        rows = await s.execute(select(Position).where(Position.status == "OPEN"))
        return list(rows.scalars().all())


async def db_orders_after(last_id):
    async with SessionLocal() as s:
        rows = await s.execute(
            select(Order).where(Order.id > last_id).order_by(Order.id)
        )
        return list(rows.scalars().all())


def order_cash_delta(order):
    """The exact cash effect of one paper fill (§11 fill model)."""
    mult = 100 if order.instrument in OPTION_INSTRUMENTS else 1
    if order.side == "BUY_TO_OPEN":
        return -(order.quantity * order.fill_price * mult + order.commission)
    return order.quantity * order.fill_price * mult - order.commission


async def get_cash(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    return r.json()["cash"]


async def assert_order_audit_chain(client, order):
    """(d): the full ORDER_* audit chain exists and matches the fill row."""
    r = await client.get("/api/audit", params={"entity_id": str(order.id)})
    assert r.status_code == 200
    events = [e for e in r.json() if e["entity_type"] == "order"]
    actions = {e["action"] for e in events}
    assert {"ORDER_REQUESTED", "ORDER_SUBMITTED", "ORDER_FILLED"} <= actions, (
        f"order {order.id} is missing audit events: {actions}"
    )
    filled = next(e for e in events if e["action"] == "ORDER_FILLED")
    assert filled["details"]["fill_price"] == pytest.approx(order.fill_price)
    assert filled["details"]["commission"] == pytest.approx(order.commission)


async def authorize(client):
    """Watchlist -> Trading Pool -> per-symbol enable -> global resume."""
    r = await client.post("/api/watchlist", json={"ticker": TICKER})
    assert r.status_code == 201
    # acknowledge_risks: no COMPLETED backtest exists for TICKER at promote
    # time, so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": TICKER, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{TICKER}/trading", json={"enabled": True})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200


async def test_replay_lifecycle(client, monkeypatch):
    # Documented seams (module docstring): stale-bar bound + zero commissions.
    settings = get_settings()
    monkeypatch.setattr(settings, "paper_commission_per_share", 0.0)
    monkeypatch.setattr(settings, "paper_commission_per_contract", 0.0)
    monkeypatch.setattr(gate_chain, "MAX_BAR_AGE_DAYS", 100_000)

    bars = build_bars()
    await authorize(client)

    # This replay OWNS the ticker's history — it advances it bar by bar. The
    # freshness refresh (ensure_daily_bars) would see the synthetic series as
    # "stale" against the real calendar and append stub bars on top, colliding
    # with the replay's own upcoming dates. Pre-arming the per-symbol attempt
    # throttle keeps the refresh a no-op for the test's lifetime.
    from datetime import datetime as _dt, timezone as _tz

    from apps.gateway.routers import analysis as analysis_router

    monkeypatch.setitem(
        analysis_router._refresh_attempts, TICKER, _dt.now(_tz.utc)
    )

    # Seed the pre-replay history in one shot; the ticker now has stored
    # bars, so no lazy backfill can ever overwrite the synthetic series.
    async with SessionLocal() as s:
        s.add_all(
            StockBarDaily(ticker=TICKER, **b)
            for b in bars[: TOTAL_BARS - REPLAY_BARS]
        )
        await s.commit()

    initial_cash = await get_cash(client)
    expected_cash = initial_cash
    last_order_id = 0
    entries = 0
    mechanical_exits = 0

    for i in range(TOTAL_BARS - REPLAY_BARS, TOTAL_BARS):
        await insert_bar(bars[i])

        open_before = await db_open_positions()
        assert len(open_before) <= 1, f"day {i}: pyramiding detected"  # (a)

        # Preview decides whether today is an entry day: only a chain with
        # ZERO failing gates may fill (§42).
        r = await client.post("/api/orders/preview", json={"ticker": TICKER})
        assert r.status_code == 200, r.text
        preview = r.json()
        chain_passes = all(g["status"] != "FAIL" for g in preview["gates"])
        if chain_passes:
            assert preview["risk"] is not None
            assert preview["risk"]["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE"}

        # (b): approve is attempted EVERY day; it may only ever succeed on a
        # fully-passing chain with no open position.
        r = await client.post("/api/orders/approve", json={"ticker": TICKER})
        if open_before:
            assert r.status_code == 409, (
                f"day {i}: approve succeeded over an OPEN position: {r.text}"
            )
        elif chain_passes:
            assert r.status_code == 200, f"day {i}: {r.text}"
            entries += 1
        else:
            assert r.status_code == 422, (
                f"day {i}: approve succeeded through a failing gate: {r.text}"
            )

        # Mechanical exits run every day (§11).
        r = await client.post("/api/positions/check-exits")
        assert r.status_code == 200
        mechanical_exits += len(r.json()["exits_triggered"])

        assert len(await db_open_positions()) <= 1, f"day {i}"  # (a)

        # (c) + (d): reconcile cash against the audited fills, every day.
        new_orders = await db_orders_after(last_order_id)
        if not open_before and not chain_passes:
            # §42: a rejected chain may produce NO order of any kind today.
            assert new_orders == [], f"day {i}: order rows from a vetoed chain"
        if open_before:
            # With a position open the only legal fill today is a close.
            assert all(o.side == "SELL_TO_CLOSE" for o in new_orders), f"day {i}"
        for order in new_orders:
            await assert_order_audit_chain(client, order)
            expected_cash += order_cash_delta(order)
            last_order_id = order.id
        cash_now = await get_cash(client)
        assert cash_now == pytest.approx(expected_cash, abs=1e-6), (
            f"day {i}: cash {cash_now} != expected {expected_cash} "
            f"(new fills: {[o.id for o in new_orders]})"
        )

    # (e): force-close anything still open, then check the ledger identity.
    if await db_open_positions():
        r = await client.post(
            "/api/orders/close",
            json={"ticker": TICKER, "reason": "replay end: force close"},
        )
        assert r.status_code == 200, r.text
        for order in await db_orders_after(last_order_id):
            await assert_order_audit_chain(client, order)
            expected_cash += order_cash_delta(order)
            last_order_id = order.id
    assert await db_open_positions() == []

    final_cash = await get_cash(client)
    assert final_cash == pytest.approx(expected_cash, abs=1e-6)

    async with SessionLocal() as s:
        closed = list(
            (
                await s.execute(
                    select(Position).where(Position.status == "CLOSED")
                )
            ).scalars().all()
        )
    assert closed, "the replay must have completed at least one round trip"
    realized_total = sum(p.realized_pnl or 0.0 for p in closed)
    # To the cent (commissions are zeroed via the documented seam, so the
    # identity holds exactly; see module docstring).
    assert final_cash == pytest.approx(initial_cash + realized_total, abs=0.005)

    # (f): the series shape guarantees both deterministically.
    assert entries >= 1, "no entry ever fired — the series seams regressed"
    assert mechanical_exits >= 1, (
        "no mechanical exit ever fired — the series seams regressed"
    )
