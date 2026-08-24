"""Research trade plan lifecycle (upgrade §19/§40/§41/§45 — Phase D).

§45 workflow proofs pinned here:

- Watchlist symbol CAN generate a research plan (no pool membership).
- Apply promotes to the Trading Pool with trading DISABLED.
- Apply does NOT place an order.
- Applying a new plan supersedes the previous ACTIVE plan (§40).
- The full user decision chain is audited (§36).
"""
from sqlalchemy import select

from apps.gateway.db import Order, SessionLocal, TradingPoolItem

BULL_TICKER = "GOOGL"  # deterministic stub bull (see test_order_preview.py)


async def watchlist(client, ticker):
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201


async def generate(client, ticker, acknowledge=False):
    r = await client.post("/api/plans/generate", json={"ticker": ticker})
    assert r.status_code == 201, r.text
    return r.json()


async def apply_plan(client, plan_id, acknowledge=True):
    return await client.post(
        f"/api/plans/{plan_id}/apply", json={"acknowledge_risks": acknowledge}
    )


async def audit_events(client, entity_id):
    r = await client.get("/api/audit", params={"entity_id": entity_id})
    return r.json()


async def test_apply_off_watchlist_422_and_pools_nothing(client):
    """2026-08-20 verifier catch: with generation OPEN, apply was the one
    path that could pool a non-watchlist symbol, bypassing the direct
    promote endpoint's membership 422. Membership is a hard precondition
    that acknowledge_risks cannot bypass."""
    r = await client.post("/api/plans/generate", json={"ticker": "GOOGL"})
    assert r.status_code == 201
    plan_id = r.json()["id"]
    r = await client.post(
        f"/api/plans/{plan_id}/apply", json={"acknowledge_risks": True}
    )
    assert r.status_code == 422
    assert "only Watchlist symbols may be promoted" in str(r.json()["detail"])
    r = await client.get("/api/trading-pool")
    assert all(row["ticker"] != "GOOGL" for row in r.json())


async def test_generate_open_off_watchlist_and_not_pool_gated(client):
    # 2026-08-20 (§4.2 amended): plan GENERATION is open research; only
    # backtests stay member-only. Execution remains behind the §10 chain.
    r = await client.post("/api/plans/generate", json={"ticker": "GOOGL"})
    assert r.status_code == 201

    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)  # NOT in the pool — §15
    assert plan["status"] == "GENERATED"
    assert plan["ticker"] == BULL_TICKER
    # The stored preview is the full research payload (§16 chain).
    assert plan["preview"]["mode"] == "research"
    assert plan["preview"]["gates"][0]["name"] == "DATA_QUALITY"
    assert plan["preview"]["execution_authorization"]["authorized"] is False
    # §41 version metadata.
    assert set(plan["versions"]) == {
        "score_weight_version",
        "edge_classification_version",
        "tradeability_version",
    }
    assert all(plan["versions"].values())
    assert plan["market_data_as_of"]  # the bars the research saw
    # Audited as a USER action (§36).
    events = await audit_events(client, str(plan["id"]))
    assert [e["action"] for e in events] == ["PLAN_GENERATED"]
    assert events[0]["actor_type"] == "USER"


async def test_apply_promotes_pool_disabled_and_places_no_order(client):
    """§19/§45: Trading Pool YES · Plan ACTIVE · Trading Enabled NO · no order."""
    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)

    r = await apply_plan(client, plan["id"])  # acknowledge (§4.3 checks fail: no backtest)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan"]["status"] == "ACTIVE"
    assert body["plan"]["applied_at"] is not None
    assert body["trading_pool"] is True
    assert body["trading_enabled"] is False  # §20: applying never enables execution
    assert body["order_placed"] is False  # §19: applying never places an order

    async with SessionLocal() as s:
        pool = (
            await s.execute(
                select(TradingPoolItem).where(TradingPoolItem.ticker == BULL_TICKER)
            )
        ).scalar_one()
        assert pool.trading_enabled is False
        orders = (await s.execute(select(Order))).scalars().all()
        assert orders == []  # NO order rows from the apply

    # The user decision chain is fully audited (§36).
    plan_events = [e["action"] for e in await audit_events(client, str(plan["id"]))]
    assert "PLAN_APPLIED" in plan_events
    pool_events = await audit_events(client, BULL_TICKER)
    add = [e for e in pool_events if e["action"] == "TRADING_POOL_ADD"]
    assert len(add) == 1
    assert add[0]["actor_type"] == "USER"
    assert add[0]["details"]["via"] == "plan_apply"
    assert add[0]["details"]["risks_acknowledged"] is True


async def test_apply_without_acknowledgment_422s_on_failed_checks(client):
    """§4.3 stays sovereign through the apply path: failed promotion checks
    block unless explicitly acknowledged, with the checks listed."""
    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)

    r = await apply_plan(client, plan["id"], acknowledge=False)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "checks" in detail
    assert any(not c["passed"] for c in detail["checks"])

    # Plan unchanged, nothing promoted.
    r = await client.get(f"/api/plans/{plan['id']}")
    assert r.json()["status"] == "GENERATED"
    async with SessionLocal() as s:
        pool = (
            await s.execute(
                select(TradingPoolItem).where(TradingPoolItem.ticker == BULL_TICKER)
            )
        ).scalar_one_or_none()
        assert pool is None


async def test_new_apply_supersedes_previous_active_plan(client):
    """§40: one ACTIVE plan per symbol; the old one is SUPERSEDED and points
    at its successor."""
    await watchlist(client, BULL_TICKER)
    first = await generate(client, BULL_TICKER)
    assert (await apply_plan(client, first["id"])).status_code == 200

    second = await generate(client, BULL_TICKER)
    r = await apply_plan(client, second["id"])
    assert r.status_code == 200
    assert r.json()["superseded_plan_id"] == first["id"]

    r = await client.get(f"/api/plans/{first['id']}")
    old = r.json()
    assert old["status"] == "SUPERSEDED"
    assert old["superseded_by"] == second["id"]
    r = await client.get(f"/api/plans/{second['id']}")
    assert r.json()["status"] == "ACTIVE"

    events = [e["action"] for e in await audit_events(client, str(first["id"]))]
    assert "PLAN_SUPERSEDED" in events


async def test_cancel_and_lifecycle_guards(client):
    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)

    r = await client.post(f"/api/plans/{plan['id']}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "CANCELLED"

    # A cancelled plan cannot be applied or re-cancelled.
    assert (await apply_plan(client, plan["id"])).status_code == 409
    assert (await client.post(f"/api/plans/{plan['id']}/cancel")).status_code == 409

    # An ACTIVE plan cannot be applied twice.
    second = await generate(client, BULL_TICKER)
    assert (await apply_plan(client, second["id"])).status_code == 200
    assert (await apply_plan(client, second["id"])).status_code == 409


async def test_stale_plan_apply_409s_and_revalidate_recomputes(client, monkeypatch):
    """§42: stale research cannot become the ACTIVE plan — apply answers
    PLAN_REVALIDATION_REQUIRED; revalidate creates a FRESH plan on current
    data which then applies normally."""
    from apps.gateway.routers import plans as plans_router

    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)

    # Frozen stub bars lag the real clock by months — with the REAL (§42)
    # tolerance restored, this plan is honestly stale.
    monkeypatch.setattr(plans_router, "PLAN_STALENESS_TOLERANCE_TRADING_DAYS", 1)

    r = await client.get(f"/api/plans/{plan['id']}")
    reval = r.json()["revalidation"]
    assert reval["revalidation_required"] is True
    assert reval["stale_market_data"] is True

    r = await apply_plan(client, plan["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "PLAN_REVALIDATION_REQUIRED"
    assert detail["revalidation"]["stale_market_data"] is True

    # §42 "Recompute": revalidate produces a NEW GENERATED plan on today's
    # chain run, linked to its predecessor, old plan untouched.
    r = await client.post(f"/api/plans/{plan['id']}/revalidate")
    assert r.status_code == 201
    body = r.json()
    assert body["revalidated_from"] == plan["id"]
    fresh = body["plan"]
    assert fresh["id"] != plan["id"]
    assert fresh["status"] == "GENERATED"
    assert body["previous"]["id"] == plan["id"]
    assert (await client.get(f"/api/plans/{plan['id']}")).json()["status"] == "GENERATED"

    # The fresh plan's data is as current as the provider allows, so with the
    # suite's tolerance restored it applies cleanly.
    monkeypatch.setattr(
        plans_router, "PLAN_STALENESS_TOLERANCE_TRADING_DAYS", 100_000
    )
    assert (await apply_plan(client, fresh["id"])).status_code == 200


async def test_config_version_drift_requires_revalidation(client, monkeypatch):
    """§42: a §41 configuration-version change invalidates reproducibility —
    the plan reports config_changed and apply refuses."""
    from apps.gateway.routers import plans as plans_router

    await watchlist(client, BULL_TICKER)
    plan = await generate(client, BULL_TICKER)

    monkeypatch.setattr(
        plans_router,
        "_current_versions",
        lambda: {
            "score_weight_version": "score-weights-v2-test-drift",  # drifted
            "edge_classification_version": "edge-class-v1",
            "tradeability_version": "tradeability-v1",
        },
    )
    r = await client.get(f"/api/plans/{plan['id']}")
    reval = r.json()["revalidation"]
    assert reval["revalidation_required"] is True
    assert reval["stale_market_data"] is False  # suite tolerance is huge
    assert "score_weight_version" in reval["config_changed"]
    assert (
        reval["config_changed"]["score_weight_version"]["plan"]
        == "score-weights-v1-grouped"
    )

    assert (await apply_plan(client, plan["id"])).status_code == 409


async def test_list_and_get(client):
    await watchlist(client, BULL_TICKER)
    p1 = await generate(client, BULL_TICKER)
    p2 = await generate(client, BULL_TICKER)

    r = await client.get("/api/plans")
    ids = [p["id"] for p in r.json()]
    assert ids == [p2["id"], p1["id"]]  # newest first

    r = await client.get("/api/plans", params={"ticker": BULL_TICKER.lower()})
    assert len(r.json()) == 2  # ticker filter is case-insensitive

    assert (await client.get("/api/plans/999999")).status_code == 404
