"""Integration tests for the §41 observability slice.

Request-ID middleware (honor/generate/echo), /metrics exposition with ROUTE
TEMPLATE labels (cardinality control), /metrics self-exclusion, and the
§38+§41 correlation closure: an audited action inherits the request's
X-Request-ID as its correlation_id.
"""
from sqlalchemy import select

from apps.gateway import audit
from apps.gateway.db import AuditEvent, SessionLocal
from libs.common.telemetry import request_id_var
from libs.trading_core.models import ActorType, AuditAction

# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------


async def test_request_id_generated_when_absent(client):
    r = await client.get("/api/watchlist")
    assert r.status_code == 200
    rid = r.headers.get("x-request-id")
    assert rid is not None
    assert len(rid) == 32  # uuid4().hex
    int(rid, 16)  # hex-parsable

    # a second request gets a DIFFERENT generated id
    r2 = await client.get("/api/watchlist")
    assert r2.headers["x-request-id"] != rid


async def test_request_id_honored_when_client_sends_one(client):
    r = await client.get(
        "/api/watchlist", headers={"X-Request-ID": "client-supplied-id-42"}
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "client-supplied-id-42"


# ---------------------------------------------------------------------------
# /metrics exposition
# ---------------------------------------------------------------------------


async def test_metrics_counts_requests_with_route_template(client):
    await client.post("/api/watchlist", json={"ticker": "AAPL"})
    r = await client.get("/api/watchlist/AAPL/analysis")
    assert r.status_code == 200
    await client.get("/api/watchlist")

    m = await client.get("/metrics")
    assert m.status_code == 200
    assert m.headers["content-type"].startswith("text/plain")
    body = m.text

    assert "# TYPE http_requests_total counter" in body
    # ROUTE TEMPLATE label, never the concrete ticker (§41 cardinality control)
    assert (
        'http_requests_total{method="GET",path="/api/watchlist/{ticker}/analysis",status="200"}'
        in body
    )
    assert 'path="/api/watchlist/AAPL/analysis"' not in body
    assert 'http_requests_total{method="GET",path="/api/watchlist",status="200"}' in body

    # latency histogram, labeled by method + template only
    assert "# TYPE http_request_duration_ms histogram" in body
    assert (
        'http_request_duration_ms_bucket{method="GET",path="/api/watchlist/{ticker}/analysis",le="+Inf"}'
        in body
    )
    assert (
        'http_request_duration_ms_count{method="GET",path="/api/watchlist/{ticker}/analysis"}'
        in body
    )


async def test_metrics_includes_uptime_and_freshness_gauges(client):
    # with a watchlist ticker whose bars were just stub-backfilled, the
    # newest bar is at most a weekend away from today
    await client.post("/api/watchlist", json={"ticker": "AAPL"})
    await client.get("/api/watchlist/AAPL/analysis")

    body = (await client.get("/metrics")).text
    assert "# TYPE process_uptime_seconds gauge" in body
    uptime = _sample_value(body, "process_uptime_seconds")
    assert uptime >= 0.0

    assert "# TYPE watchlist_bars_max_age_days gauge" in body
    age = _sample_value(body, "watchlist_bars_max_age_days")
    assert 0.0 <= age <= 4.0  # stub series ends today or the prior weekday

    # honest stub gauge: chains are generated on demand, age pinned to 0 (§41)
    assert "on demand" in _help_line(body, "option_chain_age_seconds").lower()
    assert _sample_value(body, "option_chain_age_seconds") == 0.0


async def test_metrics_excludes_itself_from_request_metrics(client):
    await client.get("/metrics")
    body = (await client.get("/metrics")).text
    assert 'path="/metrics"' not in body


# ---------------------------------------------------------------------------
# Correlation closure (§38 + §41)
# ---------------------------------------------------------------------------


async def test_audited_action_lands_request_id_in_correlation_id(client):
    r = await client.post(
        "/api/watchlist",
        json={"ticker": "MSFT"},
        headers={"X-Request-ID": "corr-abc-123"},
    )
    assert r.status_code == 201
    assert r.headers["x-request-id"] == "corr-abc-123"

    rows = await client.get("/api/audit", params={"entity_id": "MSFT"})
    events = [e for e in rows.json() if e["action"] == "WATCHLIST_ADD"]
    assert len(events) == 1
    assert events[0]["correlation_id"] == "corr-abc-123"


async def test_generated_request_id_also_reaches_audit_row(client):
    r = await client.post("/api/watchlist", json={"ticker": "NVDA"})
    generated = r.headers["x-request-id"]

    rows = await client.get("/api/audit", params={"entity_id": "NVDA"})
    events = [e for e in rows.json() if e["action"] == "WATCHLIST_ADD"]
    assert events[0]["correlation_id"] == generated


async def test_record_outside_request_context_keeps_empty_correlation(client):
    # scripts / startup: no request in flight -> honest empty string
    assert request_id_var.get() == ""
    async with SessionLocal() as session:
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.DATA_BACKFILL,
            entity_type="test",
            entity_id="SCRIPT",
        )
        await session.commit()
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(AuditEvent).where(AuditEvent.entity_id == "SCRIPT")
            )
        ).scalar_one()
        assert row.correlation_id == ""


async def test_explicitly_passed_correlation_id_wins(client):
    token = request_id_var.set("ambient-request-id")
    try:
        async with SessionLocal() as session:
            event = await audit.record(
                session,
                actor_type=ActorType.SYSTEM,
                action=AuditAction.DATA_BACKFILL,
                entity_type="test",
                entity_id="EXPLICIT",
                correlation_id="explicit-id-wins",
            )
            assert event.correlation_id == "explicit-id-wins"
    finally:
        request_id_var.reset(token)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sample_value(body: str, name: str) -> float:
    """Value of an UNLABELED sample line ``<name> <value>``."""
    for line in body.splitlines():
        if line.startswith(f"{name} "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"no sample line for {name!r}")


def _help_line(body: str, name: str) -> str:
    for line in body.splitlines():
        if line.startswith(f"# HELP {name} "):
            return line
    raise AssertionError(f"no HELP line for {name!r}")