"""Free-tier EOD options view (Massive Options Basic: contracts reference +
previous-day bars). Pins the honest-data contract: reference + EOD only,
missing capabilities NAMED, nothing approximated, day-cached.
"""
from apps.gateway.routers import options as options_router


async def test_eod_view_open_for_non_watchlist_ticker(client):
    # 2026-08-20 (§4.2 amended): research surfaces serve any ticker.
    r = await client.get("/api/watchlist/VZ/options/eod")
    assert r.status_code == 200


async def test_eod_view_contract(client):
    r = await client.post("/api/watchlist", json={"ticker": "VZ"})
    assert r.status_code == 201

    r = await client.get("/api/watchlist/VZ/options/eod")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["ticker"] == "VZ"
    assert body["data_recency"] == "end_of_day"  # §25/§37 provenance label
    assert body["spot_reference"] > 0
    assert "not a live quote" in body["spot_reference_note"]
    # What this VIEW does not contain is NAMED, never approximated — a fact
    # about the view, not a plan claim (quotes/greeks live on the chain view
    # when the provider serves them).
    assert set(body["not_in_this_view"]) == {
        "bid/ask quotes", "greeks", "implied volatility", "open interest",
    }

    # Expirations are grouped reference data.
    assert body["expirations"], "stub chain universe has front expiries"
    for e in body["expirations"]:
        assert e["dte"] >= 0
        assert e["strikes"] > 0
        assert e["calls"] > 0 and e["puts"] > 0

    # Target expiry honors the minimum-DTE preference when available.
    target = next(
        (e for e in body["expirations"] if e["dte"] >= options_router.EOD_MIN_DTE),
        body["expirations"][-1],
    )
    assert body["target_expiry"] == target["date"]

    # ATM contracts: <= 2 strikes x call+put, each with an EOD bar or an
    # honest null (never zeros).
    atm = body["atm_contracts"]
    assert 0 < len(atm) <= options_router.EOD_ATM_STRIKES * 2
    assert {c["contract_type"] for c in atm} == {"call", "put"}
    for c in atm:
        assert c["expiration_date"] == body["target_expiry"]
        assert c["ticker"].startswith("O:VZ")
        prev = c["prev_day"]
        if prev is not None:
            assert prev["close"] > 0
            assert prev["date"]


async def test_eod_view_is_day_cached(client):
    """Second read serves the cache — the free tier's 5 calls/minute budget
    is spent at most once per (ticker, day)."""
    await client.post("/api/watchlist", json={"ticker": "VZ"})
    first = (await client.get("/api/watchlist/VZ/options/eod")).json()
    assert (
        options_router._eod_cache
    ), "first read must populate the day cache"
    second = (await client.get("/api/watchlist/VZ/options/eod")).json()
    assert first == second
