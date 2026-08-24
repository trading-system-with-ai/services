"""Every response carries the wire contract's version.

The frontend ships from its own repository and updates independently of this
one, so the two halves can legitimately run different versions. Without a
marker the symptom of a mismatch is an EMPTY PANEL rather than an error:
TypeScript cannot catch a field that is simply absent from the JSON — it reads
as ``undefined`` and renders as nothing.

A client that compares this header against what it was built for can say "the
API moved" instead of showing a reader a blank where a number should be.
"""
import pytest

from apps.gateway.main import API_VERSION

pytestmark = pytest.mark.anyio


async def test_every_response_carries_the_api_version(client):
    for path in ("/api/health", "/api/config/providers", "/api/events?horizon=today"):
        response = await client.get(path)
        assert response.headers.get("X-API-Version") == API_VERSION, path


async def test_an_error_response_carries_it_too(client):
    """A client hitting a 404 because the route MOVED needs the version most —
    that is exactly the mismatch this header exists to explain."""
    response = await client.get("/api/events/999999999")
    assert response.status_code >= 400
    assert response.headers.get("X-API-Version") == API_VERSION


def test_the_version_is_a_two_part_number():
    """major.minor — minor for additive changes, major for breaking ones."""
    major, _, minor = API_VERSION.partition(".")
    assert major.isdigit() and minor.isdigit(), API_VERSION
