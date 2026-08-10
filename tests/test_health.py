async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_readyz_checks_database(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"
