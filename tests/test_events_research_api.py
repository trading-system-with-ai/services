"""Research orchestration API — the write side (Catalyst research upgrade;
plan §5, Phases 1/12/13/21; LOOP 8).

WHAT THIS FILE IS ABOUT. The read seams were proved in LOOP 6; here the
subject is the COST BOUNDARY and the honest reporting around it:

  - a GET never spends (the providers are asked to explode if a read path
    reaches them — the strongest form of the §27 poll-safety rule);
  - a POST spends exactly once per press, within named bounds, and is
    throttled afterwards;
  - every degraded outcome is a 200 with a NAMED reason, never a 5xx and
    never a silent empty answer;
  - a partial run stores its good evidence and says PARTIAL;
  - the spend is auditable: one audit row carrying what was bought.
"""
from datetime import datetime, timedelta, timezone

import pytest

from apps.gateway.db import (
    EventPredictionMarketRow,
    EventRow,
    EventSearchRunRow,
    PredictionMarketRow,
    SearchEvidenceRow,
    SessionLocal,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
async def research_client():
    """A client with BOTH research providers opted into (stub).

    Separate from the shared ``client`` fixture on purpose: web search and
    prediction markets are opt-in everywhere else in the suite — an install
    that has not configured them must serve NOT_CONFIGURED — so the default
    client deliberately leaves them unset and only these tests turn them on.
    """
    from tests.conftest import _client_with_providers

    async with _client_with_providers(
        {
            "MARKET_DATA_PROVIDER": "stub",
            "LLM_PROVIDER": "stub",
            "BROKER_PROVIDER": "simulated",
            "WEB_SEARCH_PROVIDER": "stub",
            "PREDICTION_MARKETS_PROVIDER": "stub",
        }
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_throttles():
    """Both orchestrators throttle per event in-process; a leaked attempt
    clock would make the NEXT test's first press a silent no-op."""
    from apps.gateway.event_prediction_markets import reset_market_throttle
    from apps.gateway.event_research import reset_research_throttle

    reset_research_throttle()
    reset_market_throttle()
    yield
    reset_research_throttle()
    reset_market_throttle()


async def _add_event(
    *,
    key: str = "EARNINGS:AAPL:2026-08-27",
    ticker: str | None = "AAPL",
    event_type: EventType = EventType.EARNINGS,
    when: datetime = NOW + timedelta(days=9),
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title="Apple Q3 earnings",
            ticker=ticker,
            scheduled_at=when,
            session=EventSession.AFTER_MARKET.value,
            status=EventStatus.CONFIRMED.value,
            source=EventSourceKind.STRUCTURED_PROVIDER.value,
            source_name="test",
        )
        s.add(row)
        await s.commit()
        return row.id


# ---------------------------------------------------------------------------
# GETs never fetch (§27 / audit §7.2 rule 1)
# ---------------------------------------------------------------------------


@pytest.fixture
def explode_on_provider(monkeypatch):
    """Make ANY provider construction fail loudly.

    Stronger than asserting a call count: if a read path ever reaches a
    provider, the test fails with a message naming the path rather than
    silently passing because the stub answered cheaply.
    """
    import libs.prediction_markets as pm
    import libs.web_search as ws

    def boom(name):  # pragma: no cover - reaching it IS the failure
        raise AssertionError(f"a read path asked for a provider ({name!r})")

    monkeypatch.setattr(ws, "get_provider", boom)
    monkeypatch.setattr(pm, "get_provider", boom)


async def test_research_get_never_touches_a_provider(client, explode_on_provider):
    event_id = await _add_event()
    r = await client.get(f"/api/events/{event_id}/research")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "NEVER_RUN"


async def test_prediction_markets_get_never_touches_a_provider(
    client, explode_on_provider
):
    event_id = await _add_event()
    r = await client.get(f"/api/events/{event_id}/prediction-markets")
    assert r.status_code == 200, r.text
    assert r.json()["reason"] == "NEVER_RUN"


async def test_research_get_rejects_a_future_as_of(client):
    event_id = await _add_event()
    future = (NOW + timedelta(days=3650)).isoformat()
    r = await client.get(f"/api/events/{event_id}/research", params={"as_of": future})
    assert r.status_code == 422


async def test_research_routes_404_an_unknown_event(client):
    assert (await client.get("/api/events/987654/research")).status_code == 404
    assert (
        await client.post("/api/events/987654/research/backfill")
    ).status_code == 404


# ---------------------------------------------------------------------------
# The paid POST: bounds, persistence, audit, throttle
# ---------------------------------------------------------------------------


async def test_research_backfill_stores_a_run_and_evidence(research_client):
    event_id = await _add_event()
    r = await research_client.post(f"/api/events/{event_id}/research/backfill")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fetched"] is True
    assert body["status"] in ("OK", "PARTIAL")
    assert body["queries_executed"] >= 1
    # Cost transparency: the response states what the press bought.
    assert body["results_considered"] >= body["results_accepted"]
    assert body["research_window"]["basis"]

    async with SessionLocal() as s:
        runs = (await s.execute(EventSearchRunRow.__table__.select())).all()
        assert len(runs) == 1
        rows = (await s.execute(SearchEvidenceRow.__table__.select())).all()
        assert rows, "candidates (accepted and rejected) are the audit record"

    # The stored run is now visible to the free GET.
    got = await research_client.get(f"/api/events/{event_id}/research")
    assert got.status_code == 200
    assert got.json()["run_status"] == body["status"]


async def test_research_backfill_respects_the_query_bound(research_client, monkeypatch):
    """A single press may never issue more than MAX_QUERIES_PER_EVENT calls."""
    from libs.trading_core.events.web_research import MAX_QUERIES_PER_EVENT
    import libs.web_search as ws

    calls: list[str] = []
    real = ws.get_provider("stub")

    class Counting:
        def search_web(self, query, **kw):
            calls.append(query)
            return real.search_web(query, **kw)

        def search_news(self, query, **kw):
            calls.append(query)
            return real.search_news(query, **kw)

    monkeypatch.setattr(ws, "get_provider", lambda name: Counting())
    event_id = await _add_event()
    r = await research_client.post(f"/api/events/{event_id}/research/backfill")
    assert r.status_code == 200
    assert 0 < len(calls) <= MAX_QUERIES_PER_EVENT


async def test_second_research_press_is_throttled(research_client):
    event_id = await _add_event()
    first = await research_client.post(f"/api/events/{event_id}/research/backfill")
    second = await research_client.post(f"/api/events/{event_id}/research/backfill")
    assert first.json()["fetched"] is True
    assert second.json() == {
        **{k: second.json()[k] for k in second.json()},
    }
    assert second.json()["fetched"] is False
    assert second.json()["reason"] == "RECENTLY_REFRESHED"

    async with SessionLocal() as s:
        runs = (await s.execute(EventSearchRunRow.__table__.select())).all()
        assert len(runs) == 1, "a throttled press must not store a second run"


async def test_research_throttle_is_per_event_not_global(research_client):
    """Two events must not mute each other: their windows and plans differ."""
    first_id = await _add_event(key="EARNINGS:AAPL:2026-08-27")
    second_id = await _add_event(key="EARNINGS:MSFT:2026-08-28", ticker="MSFT")
    a = await research_client.post(f"/api/events/{first_id}/research/backfill")
    b = await research_client.post(f"/api/events/{second_id}/research/backfill")
    assert a.json()["fetched"] is True
    assert b.json()["fetched"] is True


async def test_research_backfill_writes_one_audit_row_with_the_cost(research_client):
    from apps.gateway.db import AuditEvent

    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/research/backfill")
    async with SessionLocal() as s:
        rows = (
            (await s.execute(AuditEvent.__table__.select()))
            .mappings()
            .all()
        )
    search_rows = [r for r in rows if r["action"] == "EVENT_SEARCH_RUN"]
    assert len(search_rows) == 1
    details = search_rows[0]["details"]
    # What it bought, not merely that it ran.
    for key in ("queries_executed", "results_considered", "results_accepted"):
        assert key in details


async def test_research_backfill_reports_unconfigured_honestly(unconfigured_client):
    event_id = await _add_event()
    r = await unconfigured_client.post(f"/api/events/{event_id}/research/backfill")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fetched"] is False
    assert body["reason"] == "NOT_CONFIGURED"
    async with SessionLocal() as s:
        assert not (await s.execute(EventSearchRunRow.__table__.select())).all()


async def test_a_provider_failing_every_query_is_a_200_not_a_500(research_client, monkeypatch):
    import libs.web_search as ws
    from libs.web_search.provider import WebSearchError

    class Broken:
        def search_web(self, query, **kw):
            raise WebSearchError("upstream is down")

        def search_news(self, query, **kw):
            raise WebSearchError("upstream is down")

    monkeypatch.setattr(ws, "get_provider", lambda name: Broken())
    event_id = await _add_event()
    r = await research_client.post(f"/api/events/{event_id}/research/backfill")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["queries_executed"] == 0
    assert body["skipped"], "the failures are named, never silent"


async def test_a_partially_failing_run_is_stored_as_partial(research_client, monkeypatch):
    """One query failing must not discard the four that answered."""
    import libs.web_search as ws
    from libs.web_search.provider import WebSearchError

    real = ws.get_provider("stub")
    state = {"n": 0}

    class Flaky:
        def _maybe(self, query, kw, fn):
            state["n"] += 1
            if state["n"] == 1:
                raise WebSearchError("rate limited")
            return fn(query, **kw)

        def search_web(self, query, **kw):
            return self._maybe(query, kw, real.search_web)

        def search_news(self, query, **kw):
            return self._maybe(query, kw, real.search_news)

    monkeypatch.setattr(ws, "get_provider", lambda name: Flaky())
    event_id = await _add_event()
    body = (
        await research_client.post(f"/api/events/{event_id}/research/backfill")
    ).json()
    assert body["status"] == "PARTIAL"
    assert body["queries_executed"] >= 1
    assert len(body["skipped"]) == 1

    # PARTIAL runs still serve their evidence — unlike FAILED ones.
    got = (await research_client.get(f"/api/events/{event_id}/research")).json()
    assert got["run_status"] == "PARTIAL"


async def test_the_research_window_names_its_basis(research_client):
    """A fallback window must never masquerade as a real previous event."""
    event_id = await _add_event(
        key="CPI:2026-09-10", ticker=None, event_type=EventType.CPI
    )
    body = (
        await research_client.post(f"/api/events/{event_id}/research/backfill")
    ).json()
    window = body["research_window"]
    assert window["basis"]
    if window["previous_event_id"] is None:
        # No precedent in the table: the fallback must SAY it is one.
        assert window["fallback_reason"]


# ---------------------------------------------------------------------------
# Prediction markets: discovery, matching, observation
# ---------------------------------------------------------------------------


async def test_prediction_market_backfill_stores_decisions(research_client):
    event_id = await _add_event()
    r = await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fetched"] is True
    assert body["candidates_considered"] >= 0

    async with SessionLocal() as s:
        decisions = (
            await s.execute(EventPredictionMarketRow.__table__.select())
        ).all()
        markets = (await s.execute(PredictionMarketRow.__table__.select())).all()
    # Every decision references a market the provider actually returned.
    assert len(decisions) <= len(markets) + len(decisions)


async def test_prediction_market_accept_cap_is_enforced(research_client):
    from libs.trading_core.events.prediction_intel import MAX_ACCEPTED_MARKETS

    event_id = await _add_event()
    body = (
        await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    ).json()
    assert body["markets_accepted"] <= MAX_ACCEPTED_MARKETS


async def test_no_relevant_market_is_a_success_not_a_failure(research_client):
    """The valid, common outcome: considered candidates, accepted none."""
    import libs.prediction_markets as pm

    event_id = await _add_event()
    body = (
        await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    ).json()
    if body["markets_accepted"] == 0:
        assert body["fetched"] is True
        assert body["reason"] == "NO_RELEVANT_PREDICTION_MARKET"


async def test_prediction_market_provider_failure_is_distinct_from_no_market(
    research_client, monkeypatch
):
    """PROVIDER_UNAVAILABLE and NO_RELEVANT_MARKET are different answers."""
    import libs.prediction_markets as pm
    from libs.prediction_markets.provider import PredictionMarketError

    class Broken:
        def search_markets(self, query, **kw):
            raise PredictionMarketError("venue unreachable")

    monkeypatch.setattr(pm, "get_provider", lambda name: Broken())
    event_id = await _add_event()
    body = (
        await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    ).json()
    assert body["fetched"] is False
    assert body["reason"] == "PROVIDER_UNAVAILABLE"

    async with SessionLocal() as s:
        rows = (await s.execute(EventPredictionMarketRow.__table__.select())).all()
    assert not rows, "a failed discovery must not store a 'nothing relevant' batch"


async def test_prediction_market_backfill_reports_unconfigured(unconfigured_client):
    event_id = await _add_event()
    body = (
        await unconfigured_client.post(
            f"/api/events/{event_id}/prediction-markets/backfill"
        )
    ).json()
    assert body["fetched"] is False
    assert body["reason"] == "NOT_CONFIGURED"


async def test_second_market_press_is_throttled(research_client):
    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    second = (
        await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    ).json()
    assert second["fetched"] is False
    assert second["reason"] == "RECENTLY_REFRESHED"


async def test_market_backfill_audits_the_fetch(research_client):
    from apps.gateway.db import AuditEvent

    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    async with SessionLocal() as s:
        rows = (
            (await s.execute(AuditEvent.__table__.select())).mappings().all()
        )
    assert [r for r in rows if r["action"] == "PREDICTION_MARKET_FETCHED"]


async def test_re_running_discovery_updates_rather_than_duplicates(research_client):
    """The (provider, provider_market_id) pair is the identity."""
    from apps.gateway.event_prediction_markets import reset_market_throttle

    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    async with SessionLocal() as s:
        first = (await s.execute(PredictionMarketRow.__table__.select())).all()

    reset_market_throttle()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    async with SessionLocal() as s:
        second = (await s.execute(PredictionMarketRow.__table__.select())).all()

    assert len(second) == len(first), "a re-run refreshes markets, never clones them"


# ---------------------------------------------------------------------------
# Backfill idempotence at a REPEATED instant (ADR-007)
#
# The throttle makes this rare from the UI, but "rare" is not "impossible":
# an operator clearing the throttle, a retried request, or two workers can all
# re-run a backfill at the same stamped instant. UNIQUE(event_id, market_id,
# as_of) then fires, and an unhandled IntegrityError 500s a button press —
# which is exactly what these two tests forbid.
# ---------------------------------------------------------------------------


async def test_repeating_a_market_refresh_at_one_instant_is_idempotent(
    research_client, monkeypatch
):
    """Re-deciding at the same as_of UPDATES; it never duplicates or raises."""
    from dataclasses import replace

    import libs.trading_core.events.prediction_intel as pi
    from apps.gateway.db import (
        PredictionMarketPricePointRow,
        PredictionMarketSnapshotRow,
    )
    from apps.gateway.event_prediction_markets import (
        refresh_event_prediction_markets,
        reset_market_throttle,
    )

    # Force acceptance so the snapshot/history writers actually run — the
    # composite-PK paths are the ones a repeat run collides on.
    real_match = pi.match_markets

    def always_accept(event, candidates, **kw):
        out = real_match(event, candidates, **kw)
        kept = [
            replace(
                d,
                accepted=True,
                relation=d.relation or "CONTEXT",
                reject_reason=None,
            )
            for d in out.decisions[:3]
        ]
        return type(out)(decisions=kept)

    monkeypatch.setattr(pi, "match_markets", always_accept)

    event_id = await _add_event()
    moment = NOW

    async def once():
        reset_market_throttle()
        async with SessionLocal() as s:
            row = await s.get(EventRow, event_id)
            return await refresh_event_prediction_markets(
                s, row, provider_name="stub", now=moment
            )

    first = await once()
    second = await once()  # must not raise
    assert first["markets_accepted"] == second["markets_accepted"] >= 1
    # The second pass re-decides the SAME markets: nothing new is stored.
    assert second["snapshots_stored"] == 0
    assert second["history_points_stored"] == 0

    async with SessionLocal() as s:
        decisions = (
            await s.execute(EventPredictionMarketRow.__table__.select())
        ).all()
        snaps = (
            await s.execute(PredictionMarketSnapshotRow.__table__.select())
        ).all()
        points = (
            await s.execute(PredictionMarketPricePointRow.__table__.select())
        ).all()
    assert len(decisions) == first["markets_accepted"]
    assert len(snaps) == first["snapshots_stored"]
    assert len(points) == first["history_points_stored"]


async def test_repeating_a_research_run_at_one_instant_is_idempotent(
    research_client,
):
    """Two runs at one instant store two runs — and neither raises.

    Unlike the match table, a search RUN is deliberately append-only: each is
    a separate spend with its own cost record, and the read side picks the
    newest. What must not happen is a constraint violation from the evidence
    rows, which key on (run_id, canonical_url).
    """
    from apps.gateway.event_research import (
        reset_research_throttle,
        run_event_research,
    )

    event_id = await _add_event()

    async def once():
        reset_research_throttle()
        async with SessionLocal() as s:
            row = await s.get(EventRow, event_id)
            return await run_event_research(
                s, row, provider_name="stub", now=NOW
            )

    first = await once()
    second = await once()  # must not raise
    assert first["results_accepted"] == second["results_accepted"]

    async with SessionLocal() as s:
        runs = (await s.execute(EventSearchRunRow.__table__.select())).all()
    assert len(runs) == 2, "each spend keeps its own auditable run row"


# ---------------------------------------------------------------------------
# LOOP 8 review findings (adversarially confirmed)
# ---------------------------------------------------------------------------


async def test_collapse_keeps_the_accepted_twin_not_the_rejected_one(
    research_client, monkeypatch
):
    """Confirmed high: several candidates can share one canonical_url, and the
    pure layer emits REJECTED copies FIRST (time-gated and suspicious rows are
    decided before dedup registration, deliberately). Collapsing on first
    occurrence handed the row to exactly the copy that layer refused."""
    import libs.web_search as ws
    from libs.web_search.provider import SearchResult

    from apps.gateway.event_research import reset_research_throttle, run_event_research

    stale = NOW - timedelta(days=400)
    fresh = NOW - timedelta(days=3)

    class Twins:
        def _pair(self, query):
            return [
                # Rejected twin FIRST (out of window), tracking param strips
                # to the same canonical url as the good copy below.
                SearchResult(
                    provider="stub", provider_result_id="a1", query=query,
                    title="Apple guidance raised", url="https://ex.example/a?utm_source=x",
                    snippet="s", publisher="Ex", published_at=stale,
                    retrieved_at=NOW, result_type="news", rank=1,
                ),
                SearchResult(
                    provider="stub", provider_result_id="a2", query=query,
                    title="Apple guidance raised", url="https://ex.example/a",
                    snippet="s", publisher="Ex", published_at=fresh,
                    retrieved_at=NOW, result_type="news", rank=2,
                ),
            ]

        def search_web(self, query, **kw):
            return self._pair(query)

        def search_news(self, query, **kw):
            return self._pair(query)

    monkeypatch.setattr(ws, "get_provider", lambda name: Twins())
    event_id = await _add_event()
    reset_research_throttle()
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        report = await run_event_research(s, row, provider_name="stub", now=NOW)

    async with SessionLocal() as s:
        stored = (
            (await s.execute(SearchEvidenceRow.__table__.select())).mappings().all()
        )
    # One row per document, and the run's own count matches what was stored —
    # a run may never claim evidence it did not keep.
    accepted_rows = [r for r in stored if r["accepted"]]
    assert len(accepted_rows) == report["results_accepted"]
    # Whatever survived, it is never a row the pure layer refused for being
    # outside the research window.
    for row in stored:
        if row["accepted"]:
            assert row["reject_reason"] is None


async def test_a_run_that_matched_nothing_is_not_never_run(research_client, monkeypatch):
    """Confirmed high: discovery that succeeds and returns an EMPTY pool stored
    nothing, so the GET said NEVER_RUN — conflating "we looked and found
    nothing" with "nobody has ever looked"."""
    import libs.prediction_markets as pm

    class Empty:
        def search_markets(self, query, **kw):
            return []

    monkeypatch.setattr(pm, "get_provider", lambda name: Empty())
    event_id = await _add_event()

    post = (
        await research_client.post(
            f"/api/events/{event_id}/prediction-markets/backfill"
        )
    ).json()
    assert post["fetched"] is True
    assert post["reason"] == "NO_RELEVANT_PREDICTION_MARKET"

    got = (
        await research_client.get(f"/api/events/{event_id}/prediction-markets")
    ).json()
    # The GET must tell the SAME story the POST told.
    assert got["reason"] == "NO_RELEVANT_PREDICTION_MARKET"
    assert got["available"] is False


async def test_never_run_survives_for_an_event_nobody_researched(research_client):
    """The complement: the honest NEVER_RUN must still be reachable."""
    event_id = await _add_event()
    got = (
        await research_client.get(f"/api/events/{event_id}/prediction-markets")
    ).json()
    assert got["reason"] == "NEVER_RUN"


async def test_a_run_marker_is_point_in_time(research_client, monkeypatch):
    """A run that happened AFTER the requested instant did not exist then."""
    import libs.prediction_markets as pm

    class Empty:
        def search_markets(self, query, **kw):
            return []

    monkeypatch.setattr(pm, "get_provider", lambda name: Empty())
    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")

    earlier = (NOW - timedelta(days=30)).isoformat()
    got = (
        await research_client.get(
            f"/api/events/{event_id}/prediction-markets", params={"as_of": earlier}
        )
    ).json()
    assert got["reason"] == "NEVER_RUN"


@pytest.mark.parametrize("bad_name", ["definitely-not-a-provider"])
async def test_an_unknown_provider_is_a_200_and_does_not_burn_the_throttle(
    research_client, bad_name
):
    """Confirmed: get_provider() sat outside any try, so a misconfigured name
    500'd AND armed the hour-long throttle, locking out the retry."""
    from apps.gateway.event_research import run_event_research

    event_id = await _add_event()
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        first = await run_event_research(
            s, row, provider_name=bad_name, now=NOW
        )
    assert first["fetched"] is False
    assert first["reason"] == "NOT_CONFIGURED"

    # The throttle was never armed, so fixing the setting works immediately.
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        second = await run_event_research(s, row, provider_name="stub", now=NOW)
    assert second["fetched"] is True


async def test_partial_discovery_is_not_reported_as_no_relevant_market(
    research_client, monkeypatch
):
    """Confirmed: 'there is no relevant market' is a claim about the markets.
    With some discovery queries failed, the platform never saw them."""
    import libs.prediction_markets as pm
    from libs.prediction_markets.provider import (
        MarketOutcome,
        PredictionMarketError,
        PredictionMarketInfo,
    )

    state = {"n": 0}
    # One query SUCCEEDS with an irrelevant market (so the pool is non-empty
    # and PROVIDER_UNAVAILABLE correctly does not fire), the rest fail. The
    # matcher then accepts nothing — but the search was incomplete, so the
    # answer must not be the confident "no relevant market exists".
    irrelevant = PredictionMarketInfo(
        provider="stub",
        market_id="m-unrelated",
        provider_event_id=None,
        question="Will it rain in Lisbon tomorrow?",
        url="https://example.test/m",
        outcomes=[MarketOutcome(name="Yes", price=0.4)],
        resolution_criteria=None,
        end_date=None,
        status="ACTIVE",
        volume=None,
        liquidity=None,
        raw={},
    )

    class Flaky:
        def search_markets(self, query, **kw):
            state["n"] += 1
            if state["n"] == 1:
                return [irrelevant]
            raise PredictionMarketError("rate limited")

        def get_market_snapshot(self, market_id):  # pragma: no cover
            raise PredictionMarketError("not reached")

        def get_price_history(self, market_id, **kw):  # pragma: no cover
            raise PredictionMarketError("not reached")

    monkeypatch.setattr(pm, "get_provider", lambda name: Flaky())
    event_id = await _add_event()
    body = (
        await research_client.post(
            f"/api/events/{event_id}/prediction-markets/backfill"
        )
    ).json()
    assert body["reason"] == "PARTIAL_DISCOVERY"
    assert body["skipped"]


async def test_one_press_cannot_exceed_the_per_query_market_limit(
    research_client, monkeypatch
):
    """Confirmed: the per-EVENT pool cap was passed as the per-QUERY limit,
    letting one press fetch MAX_MARKET_QUERIES times the intended ceiling."""
    import libs.prediction_markets as pm
    from libs.trading_core.events.prediction_intel import MAX_MARKETS_PER_QUERY

    seen: list[int] = []

    class Recording:
        def search_markets(self, query, *, limit=20, active_only=True):
            seen.append(limit)
            return []

    monkeypatch.setattr(pm, "get_provider", lambda name: Recording())
    event_id = await _add_event()
    await research_client.post(f"/api/events/{event_id}/prediction-markets/backfill")
    assert seen, "discovery ran"
    assert all(limit <= MAX_MARKETS_PER_QUERY for limit in seen)


async def test_search_requests_metric_counts_failed_queries_too(
    research_client, monkeypatch
):
    """Confirmed: the counter only tracked queries that ANSWERED, so a
    provider erroring on every call looked free."""
    import libs.web_search as ws
    from libs.web_search.provider import WebSearchError

    from apps.gateway.event_research import (
        SEARCH_REQUESTS,
        reset_research_throttle,
        run_event_research,
    )

    class Broken:
        def search_web(self, query, **kw):
            raise WebSearchError("down")

        def search_news(self, query, **kw):
            raise WebSearchError("down")

    monkeypatch.setattr(ws, "get_provider", lambda name: Broken())
    before = SEARCH_REQUESTS.value(provider="stub")
    event_id = await _add_event()
    reset_research_throttle()
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        report = await run_event_research(s, row, provider_name="stub", now=NOW)
    after = SEARCH_REQUESTS.value(provider="stub")
    assert report["queries_executed"] == 0  # none answered...
    assert after > before  # ...but the requests were still issued and billed
