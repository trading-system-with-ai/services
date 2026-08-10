"""Authorization-boundary tests — these encode the platform's core safety rules."""


async def test_non_watchlist_symbol_cannot_enter_trading_pool(client):
    r = await client.post("/api/trading-pool", json={"ticker": "TSLA"})
    assert r.status_code == 422
    assert "not on the Watchlist" in r.json()["detail"]


async def test_promotion_starts_with_trading_disabled(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post("/api/trading-pool", json={"ticker": "NVDA"})
    assert r.status_code == 201
    assert r.json()["trading_enabled"] is False  # authorization != immediate purchase


async def test_short_strategies_rejected_by_account_constraints(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/trading-pool",
        json={"ticker": "NVDA", "allowed_strategies": ["SHORT_STOCK"]},
    )
    assert r.status_code == 422
    r = await client.post(
        "/api/trading-pool",
        json={"ticker": "NVDA", "allowed_strategies": ["NAKED_SHORT_CALL"]},
    )
    assert r.status_code == 422


async def test_trading_toggle_is_audited(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.post("/api/trading-pool", json={"ticker": "NVDA"})
    r = await client.post("/api/trading-pool/NVDA/trading", json={"enabled": True})
    assert r.json()["trading_enabled"] is True

    r = await client.get("/api/audit", params={"entity_id": "NVDA"})
    actions = [e["action"] for e in r.json()]
    assert "TRADING_POOL_TOGGLE" in actions


async def test_watchlist_removal_cascades_out_of_trading_pool(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.post("/api/trading-pool", json={"ticker": "NVDA"})

    await client.delete("/api/watchlist/NVDA")

    r = await client.get("/api/trading-pool")
    assert r.json() == []
    # cascade removal must itself be audited
    r = await client.get("/api/audit", params={"entity_id": "NVDA"})
    actions = [e["action"] for e in r.json()]
    assert "TRADING_POOL_REMOVE" in actions
