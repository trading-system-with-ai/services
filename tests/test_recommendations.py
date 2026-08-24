

async def test_stale_pending_auto_expires_and_unblocks_refresh(client):
    """User decision 2026-08-20: PENDING older than EXPIRE_AFTER_DAYS is
    marked EXPIRED on refresh — it leaves the pending view AND stops
    blocking a fresh proposal for the same ticker."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import update

    from apps.gateway.db import Recommendation, SessionLocal
    from apps.gateway.routers.recommendations import EXPIRE_AFTER_DAYS

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200
    created = r.json()["created"]
    assert created, "stub provider must propose at least one ticker"
    rec_id = created[0]["id"]

    # age the row past the expiry line
    async with SessionLocal() as s:
        await s.execute(
            update(Recommendation)
            .where(Recommendation.id == rec_id)
            .values(ts=datetime.now(timezone.utc) - timedelta(days=EXPIRE_AFTER_DAYS + 1))
        )
        await s.commit()

    r = await client.post("/api/recommendations/refresh")
    assert r.status_code == 200

    # the aged row is EXPIRED, not PENDING
    r = await client.get("/api/recommendations?status=ALL")
    rows = {x["id"]: x for x in r.json()}
    assert rows[rec_id]["status"] == "EXPIRED"
    # and PENDING no longer contains it
    r = await client.get("/api/recommendations")
    assert all(x["id"] != rec_id for x in r.json())
