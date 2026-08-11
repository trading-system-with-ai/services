"""GET /api/audit filtering (action, actor_type, entity_id) + /api/audit/actions.

Events are seeded exclusively through the existing state-changing APIs, so the
tests exercise the same audit-in-transaction paths as production (rule 12):

  WATCHLIST_ADD        NVDA    USER
  WATCHLIST_ADD        AAPL    USER
  TRADING_POOL_ADD     AAPL    USER
  TRADING_POOL_REMOVE  AAPL    SYSTEM  (cascade on watchlist removal)
  WATCHLIST_REMOVE     AAPL    USER
  TRADING_PAUSED       global  USER
  TRADING_RESUMED      global  USER
"""

SEEDED_ACTIONS = {
    "WATCHLIST_ADD",
    "WATCHLIST_REMOVE",
    "TRADING_POOL_ADD",
    "TRADING_POOL_REMOVE",
    "TRADING_PAUSED",
    "TRADING_RESUMED",
}


async def seed_audit_events(client):
    r = await client.post("/api/watchlist", json={"ticker": "NVDA"})
    assert r.status_code == 201
    r = await client.post("/api/watchlist", json={"ticker": "AAPL"})
    assert r.status_code == 201
    # acknowledge_risks: AAPL has no stored history/backtest at promote time,
    # so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": "AAPL", "acknowledge_risks": True}
    )
    assert r.status_code == 201
    # Removing a pooled symbol from the watchlist cascades a SYSTEM-attributed
    # TRADING_POOL_REMOVE alongside the USER WATCHLIST_REMOVE.
    r = await client.delete("/api/watchlist/AAPL")
    assert r.status_code == 204
    r = await client.post("/api/trading/pause", json={"reason": "filter test"})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume")
    assert r.status_code == 200


async def test_action_filter_returns_only_that_action(client):
    await seed_audit_events(client)

    r = await client.get("/api/audit", params={"action": "WATCHLIST_ADD"})
    assert r.status_code == 200
    events = r.json()
    assert len(events) == 2
    assert all(e["action"] == "WATCHLIST_ADD" for e in events)
    assert {e["entity_id"] for e in events} == {"NVDA", "AAPL"}


async def test_actor_type_filter(client):
    await seed_audit_events(client)

    r = await client.get("/api/audit", params={"actor_type": "SYSTEM"})
    assert r.status_code == 200
    events = r.json()
    # The cascade trading-pool removal is the only SYSTEM event seeded.
    assert len(events) == 1
    assert events[0]["action"] == "TRADING_POOL_REMOVE"
    assert events[0]["entity_id"] == "AAPL"

    r = await client.get("/api/audit", params={"actor_type": "USER"})
    events = r.json()
    assert len(events) == 6
    assert all(e["actor_type"] == "USER" for e in events)

    r = await client.get("/api/audit", params={"actor_type": "LLM"})
    assert r.json() == []


async def test_entity_id_and_action_combine_with_and_semantics(client):
    await seed_audit_events(client)

    r = await client.get(
        "/api/audit", params={"entity_id": "AAPL", "action": "WATCHLIST_ADD"}
    )
    assert r.status_code == 200
    events = r.json()
    assert len(events) == 1
    assert events[0]["action"] == "WATCHLIST_ADD"
    assert events[0]["entity_id"] == "AAPL"

    # Each filter matches some events on its own, but their conjunction is empty.
    r = await client.get(
        "/api/audit", params={"entity_id": "NVDA", "action": "WATCHLIST_REMOVE"}
    )
    assert r.json() == []

    # All three filters together.
    r = await client.get(
        "/api/audit",
        params={"entity_id": "AAPL", "action": "TRADING_POOL_REMOVE", "actor_type": "SYSTEM"},
    )
    events = r.json()
    assert len(events) == 1
    assert events[0]["actor_type"] == "SYSTEM"


async def test_actions_endpoint_returns_distinct_sorted_set(client):
    await seed_audit_events(client)

    r = await client.get("/api/audit/actions")
    assert r.status_code == 200
    actions = r.json()
    assert actions == sorted(SEEDED_ACTIONS)


async def test_actions_endpoint_empty_when_no_events(client):
    r = await client.get("/api/audit/actions")
    assert r.status_code == 200
    assert r.json() == []


async def test_invalid_filter_values_rejected(client):
    r = await client.get("/api/audit", params={"action": "NOT_AN_ACTION"})
    assert r.status_code == 422
    r = await client.get("/api/audit", params={"actor_type": "ROBOT"})
    assert r.status_code == 422
