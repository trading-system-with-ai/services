"""LLM catalyst context endpoint (upgrade §11/§25/§38 — Phase E).

GET /api/watchlist/{ticker}/catalyst is READ-ONLY over stored LLM output and
stored news — never a live LLM call, never mixed with market-derived
numbers. Pins the §38 freshness surface (generated_at, model recorded at
generation time, latest source timestamp) and the honest empty states.
"""
from datetime import datetime


async def refresh(client):
    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    return r.json()["created"]


async def test_catalyst_open_for_non_watchlist_ticker(client):
    # 2026-08-20 (§4.2 amended): research surfaces serve any ticker; honest
    # empties (llm null, articles []) for a symbol with no stored output.
    r = await client.get("/api/watchlist/ZZZZ/catalyst")
    assert r.status_code == 200
    assert r.json()["llm"] is None


async def test_catalyst_empty_state_is_honest(client):
    """A watchlisted symbol with no LLM output and no cited news answers
    nulls/empties — never fabricated interpretation."""
    r = await client.post("/api/watchlist", json={"ticker": "GOOGL"})
    assert r.status_code == 201
    r = await client.get("/api/watchlist/GOOGL/catalyst")
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is True  # the BLOCK is interpretive by contract
    assert body["llm"] is None
    assert body["articles"] == []
    assert body["latest_source_published_at"] is None


async def test_catalyst_reports_stored_interpretation_with_provenance(client):
    """After a refresh, the proposed ticker's catalyst view carries the §11
    fields, the §38 model recorded AT GENERATION, and cited articles."""
    created = await refresh(client)
    assert created
    ticker = created[0]["ticker"]
    # The recommendation rows themselves now record the generating model.
    assert created[0]["llm_model"].startswith("stub")

    # Catalyst is a research surface: watchlist-gated, so add the symbol.
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201

    r = await client.get(f"/api/watchlist/{ticker}/catalyst")
    assert r.status_code == 200
    body = r.json()

    llm = body["llm"]
    assert llm is not None
    assert llm["model"].startswith("stub")  # recorded at generation (§38)
    datetime.fromisoformat(llm["generated_at"])
    assert -1.0 <= llm["sentiment"] <= 1.0
    for key in ("impact", "novelty", "source_reliability"):
        assert 0.0 <= llm[key] <= 1.0
    assert llm["summary"]
    assert llm["evidence"], "grounded interpretation must cite sources"

    # Stub news cites the proposed tickers, so articles surface here with
    # verbatim provenance fields.
    for a in body["articles"]:
        assert a["title"] and a["url"]
        datetime.fromisoformat(a["published_at"])

    # §38: the latest source timestamp is real and parseable — the UI shows
    # it NEXT TO generated_at so an old summary can never read as live.
    assert body["latest_source_published_at"] is not None
    datetime.fromisoformat(body["latest_source_published_at"])
