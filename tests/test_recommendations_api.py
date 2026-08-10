"""Recommendation API tests — THE AUTHORITY-BOUNDARY PROPERTY (plan §4.1, §42, §44 rule 5).

The central safety rule of this iteration: the LLM proposes, the user curates.
POST /api/recommendations/refresh (the only LLM-driven write path) must be
provably unable to touch the Watchlist, Trading Pool, orders or positions —
the first test asserts exact row counts of all four tables before/after.
The ONLY recommendation -> watchlist path is the explicit USER promote action.
"""
import math
from datetime import datetime

from sqlalchemy import func, select

from apps.gateway.db import (
    Order,
    Position,
    SessionLocal,
    TradingPoolItem,
    WatchlistItem,
)

EXECUTION_TABLES = {
    "watchlist": WatchlistItem,
    "trading_pool": TradingPoolItem,
    "orders": Order,
    "positions": Position,
}


async def table_counts() -> dict[str, int]:
    """Row counts of every table a recommendation must never touch."""
    async with SessionLocal() as s:
        return {
            name: (await s.execute(select(func.count()).select_from(model))).scalar_one()
            for name, model in EXECUTION_TABLES.items()
        }


async def audit_events(client, action=None):
    r = await client.get("/api/audit")
    assert r.status_code == 200
    events = r.json()
    if action is not None:
        events = [e for e in events if e["action"] == action]
    return events


def assert_matches_contract(rec: dict) -> None:
    """Validate one recommendation against the plan §4.1 API contract."""
    assert isinstance(rec["id"], int)
    datetime.fromisoformat(rec["ts"])  # iso8601
    assert isinstance(rec["ticker"], str) and rec["ticker"]
    assert rec["company"] is None or isinstance(rec["company"], str)
    assert -1.0 <= rec["sentiment"] <= 1.0
    for score in ("impact", "novelty", "source_reliability"):
        assert 0.0 <= rec[score] <= 1.0
    for score in ("sentiment", "impact", "novelty", "source_reliability"):
        assert math.isfinite(rec[score])  # no NaN/Infinity, ever
    assert isinstance(rec["horizon"], str) and rec["horizon"]
    assert isinstance(rec["catalyst_type"], str) and rec["catalyst_type"]
    assert isinstance(rec["reason_codes"], list)
    assert all(isinstance(c, str) for c in rec["reason_codes"])
    assert isinstance(rec["summary"], str)
    assert isinstance(rec["evidence"], list)
    for item in rec["evidence"]:
        assert isinstance(item["source"], str) and item["source"]
        datetime.fromisoformat(item["published_at"])
        assert isinstance(item["snippet"], str) and item["snippet"]
    assert rec["status"] in {"PENDING", "DISMISSED", "PROMOTED"}


async def test_refresh_creates_pending_rows_and_llm_cannot_act(client):
    """§42/§44 rule 5: refresh writes ONLY recommendation + audit rows. The
    watchlist, trading pool, orders and positions tables are ALL unchanged —
    the "LLM cannot act" property."""
    before = await table_counts()

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["created"], "stub provider should propose candidates"
    for rec in body["created"]:
        assert_matches_contract(rec)
        assert rec["status"] == "PENDING"

    # THE AUTHORITY-BOUNDARY PROPERTY: zero execution-side writes.
    after = await table_counts()
    assert after == before
    assert after == {"watchlist": 0, "trading_pool": 0, "orders": 0, "positions": 0}

    # Every created row is LLM-attributed in the audit trail, with the full
    # §4.1 score schema + provider name in details.
    events = await audit_events(client, "RECOMMENDATION_CREATED")
    assert len(events) == len(body["created"])
    created_ids = {str(rec["id"]) for rec in body["created"]}
    for e in events:
        assert e["actor_type"] == "LLM"
        assert e["actor_id"] == "stub"  # provider name (settings.llm_provider)
        assert e["entity_id"] in created_ids
        for key in ("provider", "sentiment", "impact", "novelty", "source_reliability",
                    "horizon", "catalyst_type", "reason_codes", "summary"):
            assert key in e["details"]
        assert e["details"]["provider"] == "stub"

    # And no LLM actor ever appears on a watchlist/pool/order audit event.
    for e in await audit_events(client):
        if e["actor_type"] == "LLM":
            assert e["entity_type"] == "recommendation"


async def test_refresh_twice_skips_already_pending(client):
    first = (await client.post("/api/recommendations/refresh")).json()
    first_tickers = {rec["ticker"] for rec in first["created"]}
    assert first_tickers

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    second = r.json()
    second_tickers = {rec["ticker"] for rec in second["created"]}

    # Already-PENDING tickers are never re-proposed...
    assert first_tickers.isdisjoint(second_tickers)
    # ...and each is reported as skipped, with the PENDING reason.
    skipped = {s["ticker"]: s["reason"] for s in second["skipped"]}
    for ticker in first_tickers:
        assert ticker in skipped
        assert "PENDING" in skipped[ticker]


async def test_watchlisted_tickers_never_recommended(client):
    watchlisted = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
    for ticker in watchlisted:
        r = await client.post("/api/watchlist", json={"ticker": ticker})
        assert r.status_code == 201

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    body = r.json()

    created_tickers = {rec["ticker"] for rec in body["created"]}
    assert created_tickers.isdisjoint(set(watchlisted))
    skipped = {s["ticker"]: s["reason"] for s in body["skipped"]}
    for ticker in watchlisted:
        assert ticker in skipped
        assert "watchlist" in skipped[ticker]


async def test_dismiss_flips_status_with_user_audit(client):
    created = (await client.post("/api/recommendations/refresh")).json()["created"]
    rec = created[0]

    r = await client.post(f"/api/recommendations/{rec['id']}/dismiss")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == rec["id"]
    assert body["status"] == "DISMISSED"
    assert_matches_contract(body)

    events = await audit_events(client, "RECOMMENDATION_DISMISSED")
    assert len(events) == 1
    assert events[0]["actor_type"] == "USER"
    assert events[0]["entity_id"] == str(rec["id"])

    # Dismissing a non-PENDING row -> 409.
    r = await client.post(f"/api/recommendations/{rec['id']}/dismiss")
    assert r.status_code == 409

    # Unknown id -> 404.
    r = await client.post("/api/recommendations/999999/dismiss")
    assert r.status_code == 404


async def test_promote_is_the_only_path_to_watchlist(client):
    created = (await client.post("/api/recommendations/refresh")).json()["created"]
    rec = created[0]

    r = await client.post(f"/api/recommendations/{rec['id']}/promote")
    assert r.status_code == 200
    body = r.json()
    assert body["watchlist_ticker"] == rec["ticker"]
    assert body["recommendation"]["id"] == rec["id"]
    assert body["recommendation"]["status"] == "PROMOTED"

    # The watchlist gained the ticker — via the same semantics as POST
    # /api/watchlist, with the note referencing the recommendation.
    r = await client.get("/api/watchlist")
    rows = {w["ticker"]: w for w in r.json()}
    assert rec["ticker"] in rows
    assert f"recommendation #{rec['id']}" in rows[rec["ticker"]]["note"]

    # WATCHLIST_ADD is USER-attributed and references the recommendation id.
    adds = await audit_events(client, "WATCHLIST_ADD")
    assert len(adds) == 1
    assert adds[0]["actor_type"] == "USER"
    assert adds[0]["entity_id"] == rec["ticker"]
    assert f"recommendation #{rec['id']}" in adds[0]["details"]["note"]

    # The PROMOTED status flip is USER-audited too.
    promoted = await audit_events(client, "RECOMMENDATION_PROMOTED")
    assert len(promoted) == 1
    assert promoted[0]["actor_type"] == "USER"
    assert promoted[0]["entity_id"] == str(rec["id"])

    # Promoting again (row no longer PENDING) -> 409.
    r = await client.post(f"/api/recommendations/{rec['id']}/promote")
    assert r.status_code == 409


async def test_promote_conflicts_with_manual_watchlist_add(client):
    created = (await client.post("/api/recommendations/refresh")).json()["created"]
    rec = created[0]

    # The user manually adds the ticker in the meantime.
    r = await client.post("/api/watchlist", json={"ticker": rec["ticker"]})
    assert r.status_code == 201

    r = await client.post(f"/api/recommendations/{rec['id']}/promote")
    assert r.status_code == 409

    # The row stays PENDING — the failed promote committed nothing.
    pending = (await client.get("/api/recommendations", params={"status": "PENDING"})).json()
    assert rec["id"] in {row["id"] for row in pending}


async def test_get_filters_by_status_and_matches_contract(client):
    created = (await client.post("/api/recommendations/refresh")).json()["created"]
    assert len(created) >= 3
    to_dismiss, to_promote = created[0], created[1]
    await client.post(f"/api/recommendations/{to_dismiss['id']}/dismiss")
    await client.post(f"/api/recommendations/{to_promote['id']}/promote")

    # Default filter is PENDING.
    pending = (await client.get("/api/recommendations")).json()
    pending_ids = {row["id"] for row in pending}
    assert to_dismiss["id"] not in pending_ids
    assert to_promote["id"] not in pending_ids
    assert all(row["status"] == "PENDING" for row in pending)

    dismissed = (await client.get("/api/recommendations", params={"status": "DISMISSED"})).json()
    assert [row["id"] for row in dismissed] == [to_dismiss["id"]]

    promoted = (await client.get("/api/recommendations", params={"status": "PROMOTED"})).json()
    assert [row["id"] for row in promoted] == [to_promote["id"]]

    everything = (await client.get("/api/recommendations", params={"status": "ALL"})).json()
    assert {row["id"] for row in everything} == {row["id"] for row in created}
    for row in everything:
        assert_matches_contract(row)

    r = await client.get("/api/recommendations", params={"status": "bogus"})
    assert r.status_code == 422
