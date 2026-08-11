"""ADVERSARIAL safety tests for the broker (plan §5, §11, §18, §44 rule 18).

These tests do not confirm the happy path — the existing broker suites already
do. They ATTACK the three properties the user's requirement rests on, each one
phrased as "here is the bypass someone would actually try":

a. LIVE-TRADING IMPOSSIBILITY — no spelling of the live host, and no scheme or
   port trick, may construct the adapter. See also the parametrised host matrix
   in tests/test_broker_alpaca.py.
b. NO SILENT SIMULATION — with the broker unset, every execution path must
   write NOTHING; and with a real broker selected, a broker FAULT must surface
   as an error rather than degrading into an internally simulated fill.
c. NO SHORTING — no path may emit a sell exceeding the OPEN long, and the
   adapter has no vocabulary for opening a short at all (§5).
"""
import os
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from apps.gateway.db import Base, Order, Position, SessionLocal, engine
from apps.gateway.main import app
from libs.broker import BUY_TO_OPEN, SELL_TO_CLOSE, BrokerError
from libs.broker.alpaca import PAPER_HOST, AlpacaPaperBroker
from libs.common.config import get_settings

BULL_TICKER = "GOOGL"

PAPER_ACCOUNT = {
    "account_number": "PA3ABCDEF",
    "cash": "100000.00",
    "equity": "100000.00",
    "buying_power": "200000.00",
    "currency": "USD",
    "is_paper": True,
    "status": "ACTIVE",
}

_ENV_VARS = ("MARKET_DATA_PROVIDER", "LLM_PROVIDER", "BROKER_PROVIDER")


# ===========================================================================
# (a) LIVE-TRADING IMPOSSIBILITY
# ===========================================================================


def test_no_registered_broker_can_be_built_against_the_live_host(monkeypatch):
    """Even with the settings pointed AT the live endpoint, nothing constructs.

    This is the configuration-only attack the requirement names: an operator
    (or a leaked env file) sets the live URL and expects live trading. The
    registry must fail loudly instead.
    """
    from libs.broker import get_broker

    for live_url in (
        "https://api.alpaca.markets",
        "https://API.ALPACA.MARKETS",
        "https://api.alpaca.markets/",
        "https://paper-api.alpaca.markets@api.alpaca.markets",
    ):
        monkeypatch.setenv("ALPACA_PAPER_BASE_URL", live_url)
        monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
        monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
        get_settings.cache_clear()
        with pytest.raises(ValueError):
            get_broker("alpaca_paper")
    get_settings.cache_clear()


def test_accepted_base_url_is_always_the_canonical_paper_url():
    """Whatever spelling is supplied, the stored URL is the paper URL.

    Normalisation matters for safety: it means no later code can be fooled by
    a variant spelling, and userinfo (a classic host-confusion vector) is
    dropped rather than carried into the request.
    """
    broker = AlpacaPaperBroker(
        api_key="k",
        api_secret="s",
        # Live host in the USERINFO — the real host here is the paper host,
        # and normalisation must discard the decoy entirely.
        base_url="https://api.alpaca.markets\\@paper-api.alpaca.markets",
    )
    assert broker.base_url == f"https://{PAPER_HOST}"
    assert "api.alpaca.markets\\@" not in broker.base_url


def test_requests_are_only_ever_sent_to_the_paper_host():
    """The URL every request actually goes to, observed at the transport."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=PAPER_ACCOUNT)

    broker = AlpacaPaperBroker(
        api_key="k", api_secret="s", transport=httpx.MockTransport(handler)
    )
    broker.get_account()
    assert seen and all(url.startswith(f"https://{PAPER_HOST}/") for url in seen)


# ===========================================================================
# (c) NO SHORTING — the adapter has no vocabulary for it (§5)
# ===========================================================================


@pytest.mark.parametrize(
    "side",
    ["SELL_TO_OPEN", "sell_to_open", "SHORT", "sell", "SELL", "BUY_TO_CLOSE"],
)
def test_no_side_string_can_open_a_short(side):
    """Only the two §5 sides exist; everything else raises before any POST."""
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        posted.append({})
        return httpx.Response(200, json={})

    broker = AlpacaPaperBroker(
        api_key="k", api_secret="s", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ValueError):
        broker.submit_order("cid", BULL_TICKER, side, 1)
    assert posted == [], "an unknown side must never reach the broker"


def test_the_only_sides_are_buy_to_open_and_sell_to_close():
    from libs.broker.alpaca import _SIDE_TO_ALPACA

    assert set(_SIDE_TO_ALPACA) == {BUY_TO_OPEN, SELL_TO_CLOSE}
    # A "sell" exists ONLY as the close of a long.
    assert _SIDE_TO_ALPACA[SELL_TO_CLOSE] == "sell"


@pytest.mark.parametrize("quantity", [0, -1, -100])
def test_non_positive_quantity_never_reaches_the_broker(quantity):
    """A negative quantity is the other way to express a short."""
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        posted.append({})
        return httpx.Response(200, json={})

    broker = AlpacaPaperBroker(
        api_key="k", api_secret="s", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ValueError):
        broker.submit_order("cid", BULL_TICKER, SELL_TO_CLOSE, quantity)
    assert posted == []


# ===========================================================================
# (b) NO SILENT SIMULATION, and (c) at the API layer
# ===========================================================================


class _Broker:
    """The real adapter over a scripted transport, for gateway-level tests."""

    def __init__(self, on_order=None):
        self.posted: list[dict] = []
        self._on_order = on_order

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json as _json

        path = request.url.path
        if path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        if path == "/v2/positions":
            return httpx.Response(200, json=[])
        if path == "/v2/orders:by_client_order_id":
            return httpx.Response(404, json={"message": "not found"})
        if path == "/v2/orders" and request.method == "POST":
            self.posted.append(_json.loads(request.content))
            if self._on_order is not None:
                return self._on_order()
            return httpx.Response(
                200,
                json={
                    # Unique per submission: broker_order_id is UNIQUE in the
                    # schema, so a fixed id would collide on the second order.
                    "id": f"brk-{len(self.posted)}",
                    "client_order_id": self.posted[-1]["client_order_id"],
                    "symbol": self.posted[-1]["symbol"],
                    "side": self.posted[-1]["side"],
                    "status": "filled",
                    "qty": self.posted[-1]["qty"],
                    "filled_qty": self.posted[-1]["qty"],
                    "filled_avg_price": "200.00",
                    "submitted_at": "2026-08-10T14:30:00Z",
                },
            )
        raise AssertionError(f"unexpected {request.method} {path}")

    def adapter(self) -> AlpacaPaperBroker:
        return AlpacaPaperBroker(
            api_key="test-key",
            api_secret="test-secret",
            transport=httpx.MockTransport(self.handler),
        )


@asynccontextmanager
async def _client(monkeypatch, broker: _Broker | None):
    """A gateway client; `broker` None means BROKER_PROVIDER is UNSET."""
    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    if broker is None:
        os.environ.pop("BROKER_PROVIDER", None)
    else:
        os.environ["BROKER_PROVIDER"] = "alpaca_paper"
        monkeypatch.setattr(
            "apps.gateway.deps.get_broker", lambda name: broker.adapter()
        )
    get_settings.cache_clear()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


async def _authorize(client, ticker=BULL_TICKER):
    await client.post("/api/watchlist", json={"ticker": ticker})
    await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    await client.post("/api/trading/resume", json={})


async def _rows(model):
    async with SessionLocal() as s:
        return list((await s.execute(select(model).order_by(model.id))).scalars().all())


async def _cash(client):
    """Cash as the API reports it.

    Read through the API rather than straight off the table: the portfolio row
    is created lazily on first access, so a direct query before anything has
    touched it returns None and would make a "cash did not move" assertion
    vacuously pass.
    """
    return (await client.get("/api/portfolio/risk")).json()["cash"]


async def test_unset_broker_writes_nothing_across_every_execution_path(monkeypatch):
    """approve, close, check-exits and the monitor sweep all place NOTHING.

    The single most important negative test in the suite: an unconfigured
    install must never produce an Order or Position row, and cash must not
    move by a cent — no silent fallback to the internal simulator.
    """
    async with _client(monkeypatch, None) as client:
        await _authorize(client)
        cash_before = await _cash(client)

        assert (
            await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
        ).status_code == 503
        assert (
            await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        ).status_code == 503

        # check-exits answers 200-with-skipped (it is a sweep, not a placement)
        sweep = await client.post("/api/positions/check-exits")
        assert sweep.status_code == 200
        assert sweep.json()["skipped"] == "BROKER_NOT_CONFIGURED"

        # the background monitor's sweep, invoked directly
        from apps.gateway.routers.positions import run_exit_sweep

        async with SessionLocal() as session:
            result = await run_exit_sweep(session)
        assert result["skipped"] == "BROKER_NOT_CONFIGURED"
        assert result["exits_triggered"] == []

        assert await _rows(Order) == []
        assert await _rows(Position) == []
        assert await _cash(client) == cash_before


async def test_a_broker_fault_surfaces_and_never_degrades_to_a_fake_fill(monkeypatch):
    """A broker error must be an error — never an internally simulated fill."""
    broker = _Broker(on_order=lambda: httpx.Response(500, text="upstream boom"))
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        cash_before = await _cash(client)

        r = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
        assert r.status_code == 502
        assert r.json()["detail"]["code"] == "BROKER_ERROR"

        # NOTHING was invented: no position, no cash movement, and no order row
        # claiming a fill that did not happen.
        assert await _rows(Position) == []
        assert await _cash(client) == cash_before
        assert all(o.status != "FILLED" for o in await _rows(Order))


async def test_broker_rejection_does_not_fall_back_to_the_simulator(monkeypatch):
    broker = _Broker(on_order=lambda: httpx.Response(422, text="insufficient bp"))
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        cash_before = await _cash(client)
        r = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
        assert r.status_code == 422
        assert await _rows(Position) == []
        assert await _cash(client) == cash_before


async def test_closing_more_than_held_is_refused_and_sends_no_sell(monkeypatch):
    """(c) The over-close attack: sell more than the OPEN long.

    A sell exceeding the held quantity would leave the account SHORT — the one
    thing a long-only account (§5) must never do. It must be refused BEFORE
    anything reaches the broker.
    """
    broker = _Broker()
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        r = await client.post(
            "/api/orders/approve", json={"ticker": BULL_TICKER, "quantity": 5}
        )
        assert r.status_code == 200
        held = r.json()["position"]["quantity"]
        assert held > 0
        sells_before = sum(1 for p in broker.posted if p["side"] == "sell")

        over = await client.post(
            "/api/orders/close", json={"ticker": BULL_TICKER, "quantity": held + 1}
        )
        assert over.status_code == 422
        assert "only" in str(over.json()["detail"])

        # No sell was submitted, and the position is untouched.
        assert sum(1 for p in broker.posted if p["side"] == "sell") == sells_before
        positions = await _rows(Position)
        assert [p.quantity for p in positions] == [held]
        assert positions[0].status == "OPEN"


async def test_closing_with_no_open_position_sends_no_sell(monkeypatch):
    """Selling with nothing held is the purest form of opening a short."""
    broker = _Broker()
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 404
        assert broker.posted == []


async def test_every_sell_the_broker_receives_is_covered_by_an_open_long(monkeypatch):
    """End-to-end: buy 5, close 5 — the sell never exceeds what is held."""
    broker = _Broker()
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        r = await client.post(
            "/api/orders/approve", json={"ticker": BULL_TICKER, "quantity": 5}
        )
        held = r.json()["position"]["quantity"]

        assert (
            await client.post(
                "/api/orders/close",
                json={"ticker": BULL_TICKER, "quantity": held},
            )
        ).status_code == 200

        buys = [p for p in broker.posted if p["side"] == "buy"]
        sells = [p for p in broker.posted if p["side"] == "sell"]
        assert sum(int(p["qty"]) for p in sells) <= sum(int(p["qty"]) for p in buys)


async def test_paper_flag_is_reverified_before_every_submission():
    """(a) layer 2: a live account is refused even on the paper host."""
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={**PAPER_ACCOUNT, "is_paper": False})
        posted.append({})
        return httpx.Response(200, json={})

    broker = AlpacaPaperBroker(
        api_key="k", api_secret="s", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BrokerError, match="NOT a paper account"):
        broker.submit_order("cid", BULL_TICKER, BUY_TO_OPEN, 1)
    assert posted == [], "no order may be POSTed to a non-paper account"


async def test_the_sell_path_itself_refuses_an_oversell_bypassing_the_router(
    monkeypatch,
):
    """(c) defence in depth: the invariant does not rely on the caller.

    ``_sell_to_close_via_broker`` is called directly here with a quantity
    larger than the OPEN position — the exact mistake a future caller could
    make — and must refuse before anything reaches the broker. The router's
    own over-close check is bypassed on purpose: this asserts the guarantee
    holds at the submission boundary, not just at the edge.
    """
    from apps.gateway.routers.orders import _sell_to_close_via_broker

    broker = _Broker()
    async with _client(monkeypatch, broker) as client:
        await _authorize(client)
        r = await client.post(
            "/api/orders/approve", json={"ticker": BULL_TICKER, "quantity": 5}
        )
        held = r.json()["position"]["quantity"]
        sells_before = sum(1 for p in broker.posted if p["side"] == "sell")

        async with SessionLocal() as session:
            position = (
                await session.execute(
                    select(Position).where(Position.status == "OPEN")
                )
            ).scalars().one()
            with pytest.raises(Exception) as excinfo:
                await _sell_to_close_via_broker(session, position, held + 50)

        assert "long-only" in str(excinfo.value)
        # No sell left the process.
        assert sum(1 for p in broker.posted if p["side"] == "sell") == sells_before
