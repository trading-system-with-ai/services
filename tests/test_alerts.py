"""Alerts feed tests — GET /api/alerts (§18/§29/§38).

Every alert here is produced by driving the REAL flows through the existing
APIs (pause/resume, order preview/approve, check-exits); audit rows are
hand-inserted ONLY for the actions no V1 flow can produce
(KILL_SWITCH_TRIGGERED, ORDER_REJECTED, BACKTEST_FAILED — the last only
fails on an engine exception the stub data never triggers).

Determinism: GOOGL is the stub provider's deterministic bull ticker (same
property tests/test_order_preview.py and tests/test_position_monitor.py rely
on), so previews/approvals never depend on signal luck; the forced exit uses
the direct-UPDATE hard-stop trick from tests/test_position_monitor.py.
"""
from sqlalchemy import update

from apps.gateway.db import AuditEvent, Position, SessionLocal

BULL_TICKER = "GOOGL"

ALERT_KEYS = {"id", "ts", "severity", "title", "ticker", "action", "correlation_id"}
ALERT_ACTIONS = {
    "TRADING_PAUSED",
    "KILL_SWITCH_TRIGGERED",
    "ORDER_REJECTED",
    "RISK_DECISION",
    "EXIT_GENERATED",
    "BACKTEST_FAILED",
    "ORDER_FILLED",
    "TRADING_RESUMED",
}


async def authorize(client, ticker):
    """Watchlist -> Trading Pool -> per-symbol enable -> global resume."""
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    # acknowledge_risks: the ticker has no stored history/backtest at promote
    # time, so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200


async def alerts(client, **params):
    r = await client.get("/api/alerts", params=params)
    assert r.status_code == 200
    body = r.json()
    for a in body:  # every item carries the exact contract shape
        assert set(a) == ALERT_KEYS
    return body


async def test_pause_is_critical_with_reason_then_resume_is_info(client):
    r = await client.post("/api/trading/pause", json={"reason": "data feed stale"})
    assert r.status_code == 200

    body = await alerts(client)
    assert [a["action"] for a in body] == ["TRADING_PAUSED"]
    a = body[0]
    assert a["severity"] == "CRITICAL"
    assert a["title"] == "Trading paused — data feed stale"
    assert a["ticker"] == ""  # not symbol-scoped
    assert isinstance(a["id"], int)
    assert a["ts"]
    assert a["correlation_id"]  # bound from the request-ID middleware (§41)

    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200
    body = await alerts(client)
    # newest-first: the resume lands on top
    assert [a["action"] for a in body] == ["TRADING_RESUMED", "TRADING_PAUSED"]
    assert body[0]["severity"] == "INFO"
    assert body[0]["title"] == "Trading resumed"
    assert body[0]["ticker"] == ""


async def test_veto_preview_is_warning_naming_the_failing_gate(client):
    """A watchlist-only (non-pool) ticker's APPROVE attempt vetoes at
    TRADING_POOL_AUTHORIZATION (execution mode — research no longer gates on
    the pool, upgrade §15) — that RISK_DECISION IS an alert."""
    r = await client.post("/api/watchlist", json={"ticker": "MSFT"})
    assert r.status_code == 201
    r = await client.post("/api/orders/approve", json={"ticker": "MSFT"})
    assert r.status_code == 422
    gates = r.json()["detail"]["preview"]["gates"]
    assert gates[0]["name"] == "TRADING_POOL_AUTHORIZATION"
    assert gates[0]["status"] == "FAIL"

    body = await alerts(client)
    risk = [a for a in body if a["action"] == "RISK_DECISION"]
    assert len(risk) == 1
    a = risk[0]
    assert a["severity"] == "WARNING"
    assert a["ticker"] == "MSFT"
    assert "MSFT" in a["title"]
    assert "TRADING_POOL_AUTHORIZATION" in a["title"]  # the failing gate
    # The WATCHLIST_ADD that set this up is audited but is NOT an alert.
    assert all(x["action"] != "WATCHLIST_ADD" for x in body)


async def test_fully_approving_preview_is_not_an_alert(client):
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/orders/preview", json={"ticker": BULL_TICKER})
    assert r.status_code == 200
    assert r.json()["risk"]["decision"] in ("APPROVE", "APPROVE_WITH_RESIZE")

    # The RISK_DECISION row exists in the audit trail (§38)...
    r = await client.get("/api/audit", params={"entity_id": BULL_TICKER})
    assert any(e["action"] == "RISK_DECISION" for e in r.json())
    # ...but an approving preview is routine, never an alert.
    body = await alerts(client)
    assert all(a["action"] != "RISK_DECISION" for a in body)


async def test_order_filled_alert_names_side_qty_ticker_and_price(client):
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
    assert r.status_code == 200, r.text
    order = r.json()["order"]

    body = await alerts(client)
    fills = [a for a in body if a["action"] == "ORDER_FILLED"]
    assert len(fills) == 1
    a = fills[0]
    assert a["severity"] == "INFO"
    assert a["ticker"] == BULL_TICKER
    title = a["title"]
    assert "BUY_TO_OPEN" in title
    assert str(order["quantity"]) in title
    assert BULL_TICKER in title
    assert f"{order['fill_price']:.2f}" in title
    # The approving RISK_DECISION the approve chain re-ran is still no alert.
    assert all(x["action"] != "RISK_DECISION" for x in body)


async def test_forced_mechanical_exit_is_warning_with_rule_name(client):
    """Force HARD_STOP via the direct-UPDATE trick (sky-high entry, tiny
    stop — tests/test_position_monitor.py): the EXIT_GENERATED lands as a
    WARNING naming the rule, and the system sell fill as a newer INFO."""
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
    assert r.status_code == 200, r.text
    position_id = r.json()["position"]["id"]

    async with SessionLocal() as s:
        await s.execute(
            update(Position)
            .where(Position.id == position_id)
            .values(avg_price=1_000_000.0, stop_distance=1.0)
        )
        await s.commit()

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    assert len(r.json()["exits_triggered"]) == 1

    body = await alerts(client)
    exits = [a for a in body if a["action"] == "EXIT_GENERATED"]
    assert len(exits) == 1
    a = exits[0]
    assert a["severity"] == "WARNING"
    assert a["ticker"] == BULL_TICKER
    assert BULL_TICKER in a["title"]
    assert "HARD_STOP" in a["title"]  # the §11 rule that fired

    # Entry fill + system exit fill, both INFO and symbol-scoped (the close
    # fill's ticker comes from the Order-row enrichment).
    fills = [x for x in body if x["action"] == "ORDER_FILLED"]
    assert len(fills) == 2
    assert all(x["severity"] == "INFO" for x in fills)
    assert all(x["ticker"] == BULL_TICKER for x in fills)
    assert "SELL_TO_CLOSE" in fills[0]["title"]  # the exit fill is newer

    # Newest-first across the whole feed (ids are the audit row ids).
    ids = [x["id"] for x in body]
    assert ids == sorted(ids, reverse=True)
    # The exit fires before its fill: EXIT_GENERATED sits below the sell fill.
    assert body.index(fills[0]) < body.index(a)


async def test_flowless_actions_classify_from_hand_inserted_rows(client):
    """KILL_SWITCH_TRIGGERED and ORDER_REJECTED have no producing V1 flow,
    and BACKTEST_FAILED needs an engine exception stub data never raises —
    the ONLY hand-inserted audit rows in this suite."""
    async with SessionLocal() as s:
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                action="KILL_SWITCH_TRIGGERED",
                entity_type="system_state",
                entity_id="global",
                details={"reason": "daily loss limit hit"},
            )
        )
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                action="ORDER_REJECTED",
                entity_type="order",
                entity_id="999",  # no such Order row — enrichment must cope
                details={"ticker": "NVDA", "reason": "broker rejected"},
            )
        )
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                action="BACKTEST_FAILED",
                entity_type="backtests",
                entity_id="NVDA",
                details={"backtest_id": 1, "error": "engine exploded"},
            )
        )
        await s.commit()

    body = await alerts(client)
    by_action = {a["action"]: a for a in body}

    kill = by_action["KILL_SWITCH_TRIGGERED"]
    assert kill["severity"] == "CRITICAL"
    assert kill["ticker"] == ""
    assert "daily loss limit hit" in kill["title"]

    rejected = by_action["ORDER_REJECTED"]
    assert rejected["severity"] == "CRITICAL"
    assert rejected["ticker"] == "NVDA"
    assert "NVDA" in rejected["title"]
    assert "broker rejected" in rejected["title"]

    failed = by_action["BACKTEST_FAILED"]
    assert failed["severity"] == "WARNING"
    assert failed["ticker"] == "NVDA"
    assert "engine exploded" in failed["title"]


async def test_non_alert_actions_never_appear(client):
    """WATCHLIST_ADD / DATA_BACKFILL (and every other unlisted action) exist
    in the audit trail but never in the alerts feed."""
    await authorize(client, BULL_TICKER)  # writes WATCHLIST_ADD + pool events
    # The preview's DATA_QUALITY gate lazily backfills bars -> DATA_BACKFILL.
    r = await client.post("/api/orders/preview", json={"ticker": BULL_TICKER})
    assert r.status_code == 200

    r = await client.get("/api/audit")
    audited = {e["action"] for e in r.json()}
    assert "WATCHLIST_ADD" in audited
    assert "DATA_BACKFILL" in audited

    body = await alerts(client)
    surfaced = {a["action"] for a in body}
    assert "WATCHLIST_ADD" not in surfaced
    assert "DATA_BACKFILL" not in surfaced
    assert surfaced <= ALERT_ACTIONS


async def test_limit_validation_and_newest_first_paging(client):
    for bad in (0, 201, -5):
        r = await client.get("/api/alerts", params={"limit": bad})
        assert r.status_code == 422

    r = await client.post("/api/trading/pause", json={"reason": "first pause"})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200
    r = await client.post("/api/trading/pause", json={"reason": "second pause"})
    assert r.status_code == 200

    body = await alerts(client, limit=2)
    assert len(body) == 2
    assert [a["action"] for a in body] == ["TRADING_PAUSED", "TRADING_RESUMED"]
    assert "second pause" in body[0]["title"]

    body = await alerts(client, limit=1)
    assert len(body) == 1
    assert "second pause" in body[0]["title"]

    body = await alerts(client)  # default limit 50 covers all three
    assert [a["action"] for a in body] == [
        "TRADING_PAUSED",
        "TRADING_RESUMED",
        "TRADING_PAUSED",
    ]
    assert "first pause" in body[2]["title"]
