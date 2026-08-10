async def test_fresh_db_status_is_disabled(client):
    r = await client.get("/api/trading/status")
    assert r.status_code == 200
    body = r.json()
    assert body["trading_enabled"] is False
    assert body["reason"] == "startup default: trading disabled"
    assert body["updated_at"] is None


async def test_pause_requires_nonempty_reason(client):
    r = await client.post("/api/trading/pause", json={})
    assert r.status_code == 422

    r = await client.post("/api/trading/pause", json={"reason": ""})
    assert r.status_code == 422

    r = await client.post("/api/trading/pause", json={"reason": "   "})
    assert r.status_code == 422

    # nothing may have flipped the switch as a side effect
    r = await client.get("/api/trading/status")
    assert r.json()["trading_enabled"] is False


async def test_pause_resume_round_trip(client):
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["trading_enabled"] is True
    assert body["updated_by"] == "local-user"
    assert body["updated_at"] is not None

    r = await client.get("/api/trading/status")
    assert r.json()["trading_enabled"] is True

    r = await client.post("/api/trading/pause", json={"reason": "macro event risk"})
    assert r.status_code == 200
    body = r.json()
    assert body["trading_enabled"] is False
    assert body["reason"] == "macro event risk"
    assert body["updated_by"] == "local-user"
    assert body["updated_at"] is not None

    # state persists across requests (fresh session each call)
    r = await client.get("/api/trading/status")
    body = r.json()
    assert body["trading_enabled"] is False
    assert body["reason"] == "macro event risk"


async def test_pause_and_resume_are_audited_as_user(client):
    await client.post("/api/trading/pause", json={"reason": "audit test"})
    await client.post("/api/trading/resume", json={})

    r = await client.get("/api/audit")
    events = r.json()
    by_action = {e["action"]: e for e in events}
    assert "TRADING_PAUSED" in by_action
    assert "TRADING_RESUMED" in by_action
    for action in ("TRADING_PAUSED", "TRADING_RESUMED"):
        event = by_action[action]
        assert event["actor_type"] == "USER"
        assert event["actor_id"] == "local-user"
        assert event["entity_type"] == "system_state"
    assert by_action["TRADING_PAUSED"]["details"] == {"reason": "audit test"}
