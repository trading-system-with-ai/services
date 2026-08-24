"""Phase 8 — news ingestion, dedup, and GROUNDED LLM enrichment.

The chain under test: real articles from the market data provider →
deduplicated into ``news_articles`` by the provider's own article id → the
LLM sees ONLY stored articles → server-side grounding validation drops any
draft whose evidence cites an unknown url or whose ticker is absent from the
cited articles. A fabricated citation can never reach the user.
"""
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import select

from apps.gateway.db import NewsArticleRow, Recommendation, SessionLocal
from libs.llm.provider import RecommendationDraft
from libs.market_data.massive import MassiveProvider
from libs.market_data.provider import CapabilityNotAvailable

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Massive news parsing (MockTransport — never the network)
# ---------------------------------------------------------------------------


def _news_row(**overrides) -> dict:
    row = {
        "id": "art-001",
        "publisher": {"name": "Reuters"},
        "title": "NVDA raises guidance",
        "author": "Wire",
        "published_utc": "2026-08-12T13:30:00Z",
        "article_url": "https://news.example.com/nvda-guidance",
        "tickers": ["NVDA"],
        "description": "The company raised full-year guidance.",
    }
    row.update(overrides)
    return row


def _provider(handler) -> MassiveProvider:
    return MassiveProvider(api_key="test-key", transport=httpx.MockTransport(handler))


def test_massive_news_parses_verbatim_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/reference/news"
        assert request.url.params.get("order") == "desc"
        return httpx.Response(
            200,
            json={"results": [_news_row(), _news_row(id="art-002", tickers=["AMD", "NVDA"])]},
        )

    articles = _provider(handler).get_news(limit=10)
    assert len(articles) == 2
    a = articles[0]
    assert a.source_id == "art-001"
    assert a.publisher == "Reuters"
    assert a.url == "https://news.example.com/nvda-guidance"
    assert a.tickers == ("NVDA",)
    assert a.published_at == datetime(2026, 8, 12, 13, 30, tzinfo=timezone.utc)
    assert articles[1].tickers == ("AMD", "NVDA")


def test_massive_news_skips_uncitable_rows_never_patches():
    """An article without id/title/url/timestamp cannot ground anything."""
    rows = [
        _news_row(),
        _news_row(id=None),
        _news_row(article_url=None),
        _news_row(published_utc="not-a-date"),
        "not-a-dict",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": rows})

    articles = _provider(handler).get_news()
    assert [a.source_id for a in articles] == ["art-001"]


def test_massive_news_403_is_capability_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "not entitled"})

    with pytest.raises(CapabilityNotAvailable):
        _provider(handler).get_news()


def test_probe_includes_news_capability():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/reference/news":
            return httpx.Response(403, json={"message": "not entitled"})
        return httpx.Response(200, json={"results": [], "status": "OK"})

    report = _provider(handler).probe_capabilities()
    assert report["news"] is False  # plan-gated, honestly reported
    assert report["stock_history"] is True


# ---------------------------------------------------------------------------
# Refresh: ingest → dedup → grounded enrichment (stub providers end to end)
# ---------------------------------------------------------------------------


async def test_refresh_ingests_news_and_grounds_recommendations(client):
    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200, r.text
    body = r.json()

    # Articles were ingested and reported.
    assert body["news"]["fetched"] > 0
    assert body["news"]["new"] == body["news"]["fetched"]  # fresh table
    async with SessionLocal() as s:
        stored = (await s.execute(select(NewsArticleRow))).scalars().all()
    assert len(stored) == body["news"]["new"]

    # Every created recommendation is grounded: its evidence cites a STORED
    # article url, and its ticker appears in that article's ticker list.
    assert body["created"], "stub news + stub LLM must yield recommendations"
    stored_by_url = {a.url: set(a.tickers) for a in stored}
    for rec in body["created"]:
        assert rec["evidence"], "every recommendation must carry evidence"
        for ev in rec["evidence"]:
            assert ev["source"] in stored_by_url
        assert any(
            rec["ticker"] in stored_by_url[ev["source"]] for ev in rec["evidence"]
        )

    # Audit: the ingestion itself is on the record.
    events = (await client.get("/api/audit")).json()
    assert any(e["action"] == "NEWS_INGESTED" for e in events)


async def test_second_refresh_dedups_articles(client):
    first = (await client.post("/api/recommendations/refresh")).json()
    assert first["news"]["new"] > 0

    second = (await client.post("/api/recommendations/refresh")).json()
    # Same stub feed -> same provider article ids -> nothing new stored.
    assert second["news"]["fetched"] == first["news"]["fetched"]
    assert second["news"]["new"] == 0
    async with SessionLocal() as s:
        count = len((await s.execute(select(NewsArticleRow))).scalars().all())
    assert count == first["news"]["new"]  # no duplicates, ever

    # And no duplicate PENDING recommendations either (excluded + reported).
    tickers = [r["ticker"] for r in first["created"]]
    for t in tickers:
        assert any(
            s_["ticker"] == t and "PENDING" in s_["reason"]
            for s_ in second["skipped"]
        )


async def test_ungrounded_drafts_are_dropped(client, monkeypatch):
    """A draft citing an unknown url, and one whose ticker is not in its cited
    article, must both be DROPPED — fiction wearing a citation."""

    class LyingProvider:
        def enrich(self, articles, exclude_tickers, as_of, limit=5):
            real_url = articles[0].url
            real_tickers = set(articles[0].tickers)
            fabricated = RecommendationDraft(
                ticker="TSLA",
                company=None,
                sentiment=0.9,
                impact=0.9,
                novelty=0.9,
                source_reliability=0.9,
                horizon="1-5d",
                catalyst_type="fабricated",
                reason_codes=["MADE_UP"],
                summary="cites a url the platform never stored",
                evidence=[
                    {
                        "source": "https://fake.example.com/not-ingested",
                        "published_at": as_of.isoformat(),
                        "snippet": "…",
                    }
                ],
            )
            wrong_ticker = RecommendationDraft(
                ticker="ZZZZ",  # not in the cited article's ticker list
                company=None,
                sentiment=0.5,
                impact=0.5,
                novelty=0.5,
                source_reliability=0.5,
                horizon="1-5d",
                catalyst_type="mismatch",
                reason_codes=["WRONG_TICKER"],
                summary="cites a real article about someone else",
                evidence=[
                    {
                        "source": real_url,
                        "published_at": as_of.isoformat(),
                        "snippet": "…",
                    }
                ],
            )
            assert "ZZZZ" not in real_tickers
            return [fabricated, wrong_ticker]

    monkeypatch.setattr(
        "apps.gateway.routers.recommendations.get_recommendation_provider",
        lambda name: LyingProvider(),
    )

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["created"] == []  # neither lie was stored
    reasons = {s_["ticker"]: s_["reason"] for s_ in body["skipped"]}
    assert reasons.get("TSLA") == "evidence not grounded in stored news"
    assert reasons.get("ZZZZ") == "ticker absent from cited articles"
    async with SessionLocal() as s:
        recs = (await s.execute(select(Recommendation))).scalars().all()
    assert recs == []


async def test_news_capability_gap_is_an_honest_503(client, monkeypatch):
    class NoNewsProvider:
        def get_news(self, limit=50, published_after=None):
            raise CapabilityNotAvailable("news endpoint not in plan")

    monkeypatch.setattr(
        "apps.gateway.routers.recommendations.get_market_data_provider",
        lambda name: NoNewsProvider(),
    )
    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "NEWS_NOT_AVAILABLE"
    assert "no synthetic fallback" in detail["message"]


async def test_concurrent_refreshes_never_collide(client):
    """Two simultaneous refreshes must both answer 200: the refresh lock
    serialises them, the loser of the ingest race sees dedup — never a 500
    from the UNIQUE constraint."""
    import asyncio

    r1, r2 = await asyncio.gather(
        client.post("/api/recommendations/refresh"),
        client.post("/api/recommendations/refresh"),
    )
    assert r1.status_code == 200 and r2.status_code == 200
    bodies = sorted((r1.json(), r2.json()), key=lambda b: -b["news"]["new"])
    assert bodies[0]["news"]["new"] > 0  # one did the ingest
    assert bodies[1]["news"]["new"] == 0  # the other saw dedup
    async with SessionLocal() as s:
        stored = (await s.execute(select(NewsArticleRow))).scalars().all()
    assert len(stored) == bodies[0]["news"]["new"]  # exactly once
