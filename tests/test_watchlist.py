async def test_add_list_remove_watchlist(client):
    r = await client.post("/api/watchlist", json={"ticker": "nvda", "note": "ai leader"})
    assert r.status_code == 201
    assert r.json()["ticker"] == "NVDA"  # normalized to uppercase

    r = await client.get("/api/watchlist")
    assert [w["ticker"] for w in r.json()] == ["NVDA"]

    r = await client.delete("/api/watchlist/NVDA")
    assert r.status_code == 204

    r = await client.get("/api/watchlist")
    assert r.json() == []


async def test_duplicate_add_rejected(client):
    await client.post("/api/watchlist", json={"ticker": "AAPL"})
    r = await client.post("/api/watchlist", json={"ticker": "AAPL"})
    assert r.status_code == 409


async def test_invalid_ticker_rejected(client):
    for bad in ["", "123", "toolongticker", "NV DA", "$PY"]:
        r = await client.post("/api/watchlist", json={"ticker": bad})
        assert r.status_code == 422, bad


async def test_watchlist_mutations_are_audited(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.delete("/api/watchlist/NVDA")

    r = await client.get("/api/audit")
    actions = [e["action"] for e in r.json()]
    assert "WATCHLIST_ADD" in actions
    assert "WATCHLIST_REMOVE" in actions
    # all watchlist mutations must be USER-attributed
    for e in r.json():
        if e["action"].startswith("WATCHLIST"):
            assert e["actor_type"] == "USER"
