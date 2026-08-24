"""Repro: provider stamps retrieved_at AFTER the captured `now`."""
from datetime import datetime, timedelta, timezone
from collections import Counter
import pytest

from apps.gateway.db import EventRow, EventSearchRunRow, SearchEvidenceRow, SessionLocal
from tests.test_events_research_api import NOW, research_client, _add_event  # noqa
from libs.web_search.provider import SearchResult


class LateStamp:
    """Brave-shaped: undated results, retrieved_at strictly after `now`."""
    name = "brave"
    def capabilities(self): return {}
    def _mk(self, query, rtype, limit):
        stamp = NOW + timedelta(seconds=2)
        return [
            SearchResult(
                provider="brave", provider_result_id=None, query=query,
                title=f"Real headline {i} for {query}",
                url=f"https://reuters.com/{abs(hash(query))%1000}/a{i}",
                snippet="Analysts expect guidance to be raised this quarter.",
                publisher="Reuters",
                published_at=None,          # Brave omitted page_age
                retrieved_at=stamp,          # stamped during the search
                result_type=rtype, rank=i,
            ) for i in range(3)
        ]
    def search_web(self, query, **kw): return self._mk(query, "web", kw.get("limit", 5))
    def search_news(self, query, **kw): return self._mk(query, "news", kw.get("limit", 5))


async def test_repro(research_client, monkeypatch):
    import apps.gateway.event_research as er
    from libs import web_search as ws
    from apps.gateway.event_research import reset_research_throttle, run_event_research, web_research_section

    monkeypatch.setattr(ws, "get_provider", lambda name: LateStamp())
    event_id = await _add_event()
    reset_research_throttle()
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        rep = await run_event_research(s, row, provider_name="brave", now=NOW)
    print("REPORT:", {k: rep[k] for k in ("status","queries_executed","results_considered","results_accepted","skipped")})

    async with SessionLocal() as s:
        rows = (await s.execute(SearchEvidenceRow.__table__.select())).all()
        print("reject reasons:", Counter(r.reject_reason for r in rows))
        row = await s.get(EventRow, event_id)
        sec = await web_research_section(s, row, as_of=NOW + timedelta(hours=1))
        print("GET available:", sec["available"], "reason:", sec.get("reason"))
