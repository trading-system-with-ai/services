"""§30 End-to-End Acceptance Test — all 20 steps, one flow.

Watchlist → historical data → backtest → Trading Pool → enable → signal →
LONG_CALL selection → real contract → risk sizing → permission gate →
BUY_TO_OPEN at the broker → fill → local/broker reconciliation → monitor →
exit trigger → SELL_TO_CLOSE → fill → close → realized P&L → audit trail.

VENUE HONESTY: the broker is the REAL AlpacaPaperBroker over MockTransport —
the paper-host guard, OCC symbol addressing, status mapping and
client_order_id plumbing are all production code. Market data is the
deterministic stub (CALL_TICKER "GW" yields BULL/LOW → LONG_CALL, same seam
as test_option_execution): the LIVE-Massive variant of steps 2/6/8 requires
an options-capable Massive plan and runs as a manual acceptance once the
plan includes option chains. Everything downstream of the data — every gate,
sizing rule, broker exchange, exit rule, cash cent and audit row — is the
real pipeline end to end.
"""
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import update

from apps.gateway.db import Order, Position, SessionLocal

from tests.test_broker_execution import (
    FakeAlpaca,
    PAPER_ACCOUNT,
    _broker_client,
    cash,
    rows,
)

pytestmark = pytest.mark.anyio

CALL_TICKER = "GW"  # deterministic BULL/LOW -> LONG_CALL (see module docstring)


class EchoAlpaca(FakeAlpaca):
    """Echoes each submission back as an immediate full fill.

    ``fill_prices`` is consumed one POST at a time (buy leg, then sell leg),
    so the test controls both premiums while quantity comes from whatever the
    RISK ENGINE actually submitted — sizing is under test, not scripted.
    """

    def __init__(self, fill_prices: list[str]):
        super().__init__({})
        self.fill_prices = list(fill_prices)
        self.book: dict[str, dict] = {}
        self.seq = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json as _json

        path = request.url.path
        if path == "/v2/orders" and request.method == "POST":
            self.calls.append((request.method, path))
            body = _json.loads(request.content)
            self.posted.append(body)
            self.seq += 1
            fill = {
                "id": f"brk-acc-{self.seq:04d}",
                "client_order_id": body["client_order_id"],
                "symbol": body["symbol"],
                "side": body["side"],
                "status": "filled",
                "qty": body["qty"],
                "filled_qty": body["qty"],
                "filled_avg_price": self.fill_prices.pop(0),
                "submitted_at": "2026-08-11T14:30:00Z",
            }
            self.book[body["client_order_id"]] = fill
            return httpx.Response(200, json=fill)
        if path == "/v2/orders:by_client_order_id":
            self.calls.append((request.method, path))
            cid = request.url.params.get("client_order_id", "")
            if cid in self.book:
                return httpx.Response(200, json=self.book[cid])
            return httpx.Response(404, json={"message": "order not found"})
        return super().handler(request)


async def test_section_30_acceptance_all_twenty_steps(monkeypatch):
    fake = EchoAlpaca(fill_prices=[])  # premiums set after the preview reads
    async with _broker_client(fake, monkeypatch) as client:
        # ---- 1. Add to Watchlist -------------------------------------------
        r = await client.post("/api/watchlist", json={"ticker": CALL_TICKER})
        assert r.status_code == 201

        # ---- 2. Historical data lands in the local store -------------------
        r = await client.get(f"/api/watchlist/{CALL_TICKER}/analysis")
        assert r.status_code == 200
        assert r.json()["bars"]["count"] >= 200  # enough for every indicator

        # ---- 3. Backtest runs to COMPLETED ---------------------------------
        r = await client.post("/api/backtests", json={"ticker": CALL_TICKER})
        assert r.status_code == 200
        assert r.json()["status"] == "COMPLETED"

        # ---- 4. Promote to Trading Pool (checks evaluated + recorded) ------
        r = await client.post(
            "/api/trading-pool",
            json={"ticker": CALL_TICKER, "acknowledge_risks": True},
        )
        assert r.status_code == 201
        checks = {c["name"]: c["passed"] for c in r.json()["promotion_checks"]}
        assert checks["MIN_HISTORY"] and checks["BACKTEST_COMPLETED"]

        # ---- 5. Enable trading ---------------------------------------------
        r = await client.post(
            f"/api/trading-pool/{CALL_TICKER}/trading", json={"enabled": True}
        )
        assert r.status_code == 200
        assert (await client.post("/api/trading/resume", json={})).status_code == 200

        # ---- 6-10. Signal -> LONG_CALL -> real contract -> sizing -> gates -
        r = await client.post("/api/orders/preview", json={"ticker": CALL_TICKER})
        assert r.status_code == 200
        preview = r.json()
        assert preview["signal"]["bias"] == "BULL"  # 6: signal valid
        assert preview["proposed"]["instrument"] == "LONG_CALL"  # 7
        contract = preview["proposed"]["contract"]  # 8: a real chain contract
        # Compact OCC (2026-08-17): 15 + len(root), no padding.
        assert contract is not None and len(contract["option_symbol"]) == 17
        risk = preview["risk"]  # 9: risk engine sized it
        assert risk["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE"}
        assert risk["approved_quantity"] >= 1
        gates = {g["name"]: g["status"] for g in preview["gates"]}
        assert gates["INSTRUMENT"] == "PASS"  # 10: permission gate
        assert gates["RISK_APPROVAL"] == "PASS"

        # Broker premiums: buy AT the chain mid (so the monitor holds in step
        # 14 — no instant premium stop), sell 2.30 higher.
        entry_premium = round(contract["mid"], 2)
        exit_premium = round(entry_premium + 2.30, 2)
        fake.fill_prices = [f"{entry_premium:.2f}", f"{exit_premium:.2f}"]

        # ---- 11-12. BUY_TO_OPEN reaches Alpaca; the fill comes back --------
        cash_start = await cash(client)
        r = await client.post("/api/orders/approve", json={"ticker": CALL_TICKER})
        assert r.status_code == 200, r.text
        assert len(fake.posted) == 1
        assert fake.posted[0]["side"] == "buy"
        assert fake.posted[0]["symbol"] == contract["option_symbol"]  # OCC addressed
        order = (await rows(Order))[0]
        qty = order.filled_quantity
        assert qty == risk["approved_quantity"] and order.status == "FILLED"
        position = (await rows(Position))[0]
        assert position.quantity == qty and position.multiplier == 100
        assert position.avg_price == pytest.approx(entry_premium)
        # Cash lives ONLY at the broker (static fake -> unchanged reading);
        # the platform's record of the buy is the position itself.
        assert await cash(client) == pytest.approx(cash_start)

        # ---- 13. Local position matches the broker's ----------------------
        fake.positions = [
            {
                "symbol": contract["option_symbol"],
                "qty": str(qty),
                "avg_entry_price": f"{entry_premium:.2f}",
                "market_value": f"{qty * entry_premium * 100:.2f}",
            }
        ]
        r = await client.get("/api/broker/reconcile")
        assert r.status_code == 200
        recon = r.json()
        assert recon["mismatches"] == [] and recon["in_sync"] is True

        # ---- 14. The monitor evaluates and HOLDS a healthy position --------
        r = await client.post("/api/positions/check-exits")
        assert r.status_code == 200
        body = r.json()
        assert body["checked"] == 1 and body["exits_triggered"] == []

        # ---- 15-18. Exit condition -> SELL_TO_CLOSE -> fill -> closed ------
        near_expiry = (
            datetime.now(timezone.utc).date() + timedelta(days=10)
        ).isoformat()
        async with SessionLocal() as s:
            await s.execute(
                update(Position)
                .where(Position.id == position.id)
                .values(opt_expiry=near_expiry)
            )
            await s.commit()
        r = await client.post("/api/positions/check-exits")
        assert r.status_code == 200
        triggered = r.json()["exits_triggered"]
        assert len(triggered) == 1  # 15: an exit rule fired
        assert len(fake.posted) == 2 and fake.posted[1]["side"] == "sell"  # 16
        sell = (await rows(Order))[1]
        assert sell.side == "SELL_TO_CLOSE" and sell.status == "FILLED"  # 17
        closed = (await rows(Position))[0]
        assert closed.status == "CLOSED" and closed.quantity == 0  # 18

        # ---- 19. Realized P&L reconciles to the cent -----------------------
        # (Cash itself lives at the broker and is read live — the platform's
        # own economics record is the realized P&L on the position.)
        assert closed.realized_pnl == pytest.approx(
            (exit_premium - entry_premium) * qty * 100
        )

        # ---- 20. The full audit trail exists -------------------------------
        r = await client.get("/api/audit")
        actions = {e["action"] for e in r.json()}
        assert {
            "DATA_BACKFILL",
            "BACKTEST_STARTED",
            "BACKTEST_COMPLETED",
            "TRADING_POOL_ADD",
            "RISK_DECISION",
            "ORDER_REQUESTED",
            "ORDER_SUBMITTED",
            "ORDER_FILLED",
            "EXIT_GENERATED",
        } <= actions
