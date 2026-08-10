import os

# Must be set before any app import so the engine binds to the test database.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"

import pytest
from httpx import ASGITransport, AsyncClient

from apps.gateway.db import Base, engine
from apps.gateway.main import app


@pytest.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
