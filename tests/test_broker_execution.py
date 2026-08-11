"""Gateway broker wiring — approve / close / exit sweep (plan §11, §44 rule 18).

THE THREE EXECUTION VENUES, and what each must do:

1. UNSET (a fresh install): approve and close answer 503
   BROKER_NOT_CONFIGURED and write NOTHING — no Order row, no Position row, no
   cash movement. Crucially they must not fall back to the internal simulator:
   a simulated fill reported as a broker fill is an execution that never
   happened, presented as one that did.
2. ``simulated``: the internal §11 fill model, byte-identical to its behaviour
   before the broker existed. Proven here by re-running the hand-computed fill
   arithmetic from tests/test_orders_execution.py under an explicit opt-in.
3. ``alpaca_paper`` with a mocked transport: the order goes to the broker with
   OUR client_order_id, and whatever the broker says is what gets recorded —
   a full fill opens a full position, a PARTIAL fill opens a PARTIAL position,
   a zero fill opens NO position, a rejection opens nothing and audits
   ORDER_REJECTED. Nothing is ever assumed.

The network is never touched: every broker call runs through an
``httpx.MockTransport`` wired into the real ``AlpacaPaperBroker``, so the real
adapter, the real paper-host guard and the real status mapping are all in the
path — only the wire is fake.
"""
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from apps.gateway.db import Base, Order, Position, SessionLocal, engine
from apps.gateway.deps import BROKER_NOT_CONFIGURED
from apps.gateway.main import app
from libs.broker.alpaca import AlpacaPaperBroker
from libs.common.config import get_settings

# The deterministic stub (RNG seeded by the symbol) gives GOOGL a bull regime
# and a BULL bias, so the §10 chain approves — the same property the existing
# execution tests rely on.
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


# ---------------------------------------------------------------------------
# Broker double: the REAL adapter over a mocked transport
# ---------------------------------------------------------------------------


class FakeAlpaca:
    """Records every request and answers with scripted order payloads.

    Deliberately wraps the real :class:`AlpacaPaperBroker` rather than
    implementing the Protocol from scratch: the paper-host guard, the
    account-is-paper precheck, the status mapping and the client_order_id
    plumbing are all part of what these tests are asserting about, so they must
    be the production code, not a stand-in.
    """

    def __init__(self, order_payload: dict, *, poll_payload: dict | None = None):
        self.calls: list[tuple[str, str]] = []
        self.posted: list[dict] = []
        # What POST /v2/orders answers with (or an httpx.Response for errors).
        self.order_payload = order_payload
        # What the GET-by-client-id poll answers with after a submission;
        # defaults to the submission response (nothing changed while polling).
        self.poll_payload = poll_payload
        # Orders the broker ALREADY holds, keyed by our client_order_id — used
        # to exercise the adopt-instead-of-resubmit path.
        self.existing: dict[str, dict] = {}
        self.positions: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path

        if path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        if path == "/v2/positions":
            return httpx.Response(200, json=self.positions)
        if path == "/v2/orders:by_client_order_id":
            client_order_id = request.url.params.get("client_order_id", "")
            if client_order_id in self.existing:
                return httpx.Response(200, json=self.existing[client_order_id])
            if self.posted and self.poll_payload is not None:
                return httpx.Response(200, json=self.poll_payload)
            if self.posted:
                return httpx.Response(200, json=self._submitted_body())
            return httpx.Response(404, json={"message": "order not found"})
        if path == "/v2/orders" and request.method == "POST":
            body = json.loads(request.content)
            self.posted.append(body)
            if isinstance(self.order_payload, httpx.Response):
                return self.order_payload
            return httpx.Response(200, json=self._submitted_body())
        raise AssertionError(f"unexpected broker request {request.method} {path}")

    def _submitted_body(self) -> dict:
        body = dict(self.order_payload)
        if self.posted:
            body.setdefault("client_order_id", self.posted[-1]["client_order_id"])
            body["client_order_id"] = self.posted[-1]["client_order_id"]
        return body

    @property
    def submit_count(self) -> int:
        return len(self.posted)

    def broker(self) -> AlpacaPaperBroker:
        return AlpacaPaperBroker(
            api_key="test-key",
            api_secret="test-secret",
            transport=httpx.MockTransport(self.handler),
        )


def order_body(**overrides) -> dict:
    payload = {
        "id": "brk-order-0001",
        "client_order_id": "will-be-overwritten",
        "symbol": BULL_TICKER,
        "side": "buy",
        "status": "filled",
        "qty": "10",
        "filled_qty": "10",
        "filled_avg_price": "200.00",
        "submitted_at": "2026-08-10T14:30:00Z",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Clients. The shared conftest fixtures cover UNSET and "simulated"; the real
# broker path needs its own, with the adapter swapped for the mocked one.
# ---------------------------------------------------------------------------

_ENV_VARS = ("MARKET_DATA_PROVIDER", "LLM_PROVIDER", "BROKER_PROVIDER")


@asynccontextmanager
async def _broker_client(fake: FakeAlpaca, monkeypatch):
    """A client on BROKER_PROVIDER=alpaca_paper, wired to `fake`'s transport."""
    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["BROKER_PROVIDER"] = "alpaca_paper"
    get_settings.cache_clear()
    # Only the CONSTRUCTION of the adapter is replaced: everything downstream
    # is the real AlpacaPaperBroker talking to a MockTransport.
    monkeypatch.setattr("apps.gateway.deps.get_broker", lambda name: fake.broker())
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


@pytest.fixture
def fake_broker():
    return FakeAlpaca(order_body())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def authorize(client, ticker=BULL_TICKER):
    """Watchlist -> Trading Pool -> per-symbol enable -> global resume."""
    assert (await client.post("/api/watchlist", json={"ticker": ticker})).status_code == 201
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    assert r.status_code == 200
    assert (await client.post("/api/trading/resume", json={})).status_code == 200


async def approve(client, ticker=BULL_TICKER, quantity=None, client_order_id=None):
    body = {"ticker": ticker}
    if quantity is not None:
        body["quantity"] = quantity
    if client_order_id is not None:
        body["client_order_id"] = client_order_id
    return await client.post("/api/orders/approve", json=body)


async def rows(model):
    async with SessionLocal() as s:
        return list((await s.execute(select(model).order_by(model.id))).scalars().all())


async def audit_actions_for(client, entity_id):
    r = await client.get("/api/audit", params={"entity_id": str(entity_id)})
    assert r.status_code == 200
    return list(reversed(r.json()))


async def cash(client):
    return (await client.get("/api/portfolio/risk")).json()["cash"]


# ===========================================================================
# 1. UNCONFIGURED — the platform places nothing and closes nothing
# ===========================================================================


async def test_approve_503_when_nothing_is_configured(unconfigured_client):
    """A fresh install refuses to place an order — and NOT with a silent
    internal fill.

    Market data is unset here too and is checked first, so the code is
    whichever guard fired; the point of THIS test is the refusal itself.
    ``test_broker_unconfigured_alone_still_refuses`` isolates the broker guard
    with market data available, which is the case a fallback would hide.
    """
    r = await approve(unconfigured_client)

    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] in {BROKER_NOT_CONFIGURED, "MARKET_DATA_NOT_CONFIGURED"}
    assert "PROVIDER" in detail["message"]


async def test_close_503_when_no_broker_configured(unconfigured_client):
    r = await unconfigured_client.post("/api/orders/close", json={"ticker": BULL_TICKER})

    assert r.status_code == 503
    # Market data is also unset in this fixture and is checked first, so the
    # code may be either guard — what matters is that it REFUSES.
    assert r.json()["detail"]["code"] in {
        BROKER_NOT_CONFIGURED,
        "MARKET_DATA_NOT_CONFIGURED",
    }


async def test_unconfigured_approve_writes_no_rows(unconfigured_client):
    """The 503 is a real refusal, not a status code on a completed write."""
    r = await approve(unconfigured_client)
    assert r.status_code == 503

    assert await rows(Order) == []
    assert await rows(Position) == []
    # …and cash is untouched.
    assert await cash(unconfigured_client) == get_settings().paper_initial_cash


async def test_broker_unconfigured_alone_still_refuses(monkeypatch):
    """Market data configured, broker NOT: approve must still 503.

    The important case, and the one a fallback would hide: everything needed
    to compute a simulated fill is available, and the platform still refuses
    because there is nowhere to actually place the order.
    """
    fake = FakeAlpaca(order_body())
    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["BROKER_PROVIDER"] = ""
    get_settings.cache_clear()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await authorize(client)
            r = await approve(client)

            assert r.status_code == 503
            assert r.json()["detail"]["code"] == BROKER_NOT_CONFIGURED
            assert await rows(Order) == []
            assert await rows(Position) == []
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()
    assert fake.submit_count == 0


async def test_check_exits_skips_when_broker_unconfigured(monkeypatch):
    """The sweep SELLS. With nowhere to sell, it must change nothing."""
    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["BROKER_PROVIDER"] = ""
    get_settings.cache_clear()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # A real OPEN position, seeded directly so the sweep has work to do.
            async with SessionLocal() as s:
                s.add(
                    Position(
                        ticker=BULL_TICKER,
                        quantity=10,
                        avg_price=100.0,
                        max_loss=200.0,
                        stop_distance=20.0,
                        entry_edge=0.5,
                        entry_bar_date="2026-08-07",
                    )
                )
                await s.commit()

            r = await client.post("/api/positions/check-exits")
            assert r.status_code == 200
            body = r.json()
            assert body["skipped"] == BROKER_NOT_CONFIGURED
            assert body["checked"] == 0
            assert body["exits_triggered"] == []

            # NOTHING changed: the position is still open, no sell was written.
            (position,) = await rows(Position)
            assert position.status == "OPEN"
            assert position.quantity == 10
            assert await rows(Order) == []
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


async def test_monitor_sweep_skips_when_broker_unconfigured(caplog):
    """The background monitor takes the same skip, and stays alive."""
    from apps.gateway import monitor

    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["BROKER_PROVIDER"] = ""
    get_settings.cache_clear()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        sweeps_before = monitor.STATE.sweeps_total
        with caplog.at_level(logging.WARNING, logger="position_monitor"):
            result = await monitor.run_sweep_and_update()

        assert result["skipped"] == BROKER_NOT_CONFIGURED
        # A skipped sweep is not a sweep: the counter must not advance.
        assert monitor.STATE.sweeps_total == sweeps_before
        assert any(
            rec.message == "exit_sweep_skipped_no_broker" for rec in caplog.records
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


async def test_broker_status_never_503s_and_explains_the_absence(unconfigured_client):
    """The surface that EXPLAINS the unconfigured state must work in it."""
    r = await unconfigured_client.get("/api/broker/status")

    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is False
    assert body["provider"] == ""  # verbatim, never a cosmetic default
    assert body["mode"] is None
    assert body["account"] is None
    assert "BROKER_PROVIDER" in body["error"]


async def test_config_reports_broker_unconfigured(unconfigured_client):
    providers = (await unconfigured_client.get("/api/config")).json()["providers"]

    assert providers["broker"] == ""
    assert providers["broker_configured"] is False


# ===========================================================================
# 2. SIMULATED — the existing behaviour, byte-identical, behind an opt-in
# ===========================================================================


async def test_simulated_mode_fill_model_is_unchanged(client):
    """The hand-computed §11 fill arithmetic still holds under "simulated".

    This is deliberately the SAME computation as
    tests/test_orders_execution.py::test_approve_happy_path_fill_model_and_audit_chain.
    Adding the broker must not have moved a single decimal of the internal
    model — it only made reaching it an explicit choice.
    """
    settings = get_settings()
    assert settings.broker_provider == "simulated"  # the fixture's opt-in

    await authorize(client)
    cash_before = await cash(client)

    r = await approve(client)
    assert r.status_code == 200
    body = r.json()
    order, preview = body["order"], body["preview"]
    qty = order["quantity"]

    last_close = preview["proposed"]["entry_price"]
    expected_fill = last_close * (1.0 + settings.paper_slippage_bps / 10000.0)
    expected_commission = settings.paper_commission_per_share * qty

    assert order["fill_price"] == pytest.approx(expected_fill)
    assert order["commission"] == pytest.approx(expected_commission)
    assert order["status"] == "FILLED"
    assert body["position"]["avg_price"] == pytest.approx(expected_fill)
    assert await cash(client) == pytest.approx(
        cash_before - (qty * expected_fill + expected_commission)
    )

    # An internal fill is NOT a broker fill and never claims to be one.
    assert order["broker"] is None
    (row,) = await rows(Order)
    assert row.broker_order_id is None
    assert row.broker_status is None


async def test_simulated_mode_reports_no_broker_account(client):
    """"simulated" is configured, but there is no account to report."""
    body = (await client.get("/api/broker/status")).json()

    assert body["configured"] is True
    assert body["provider"] == "simulated"
    # NOT "paper": there is no broker account, and calling the internal
    # simulator one would be the exact conflation this platform refuses.
    assert body["mode"] is None
    assert body["account"] is None
    assert "INTERNAL" in body["error"]


# ===========================================================================
# 3. ALPACA PAPER (mocked transport) — the broker decides, we record
# ===========================================================================


async def test_approve_submits_with_our_client_order_id(monkeypatch):
    """§42 across the network: our idempotency key IS the broker's."""
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        r = await approve(client, quantity=10, client_order_id="my-key-001")
        assert r.status_code == 200

        assert fake.submit_count == 1
        submitted = fake.posted[0]
        assert submitted["client_order_id"] == "my-key-001"
        assert submitted["symbol"] == BULL_TICKER
        assert submitted["side"] == "buy"
        assert submitted["qty"] == "10"

        # The broker was ASKED first — the adopt-not-resubmit lookup (§42).
        assert ("GET", "/v2/orders:by_client_order_id") in fake.calls


async def test_full_fill_opens_position_and_audits_broker_order_id(monkeypatch):
    fake = FakeAlpaca(
        order_body(status="filled", qty="10", filled_qty="10", filled_avg_price="200.00")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="fill-1")
        assert r.status_code == 200
        body = r.json()

        order = body["order"]
        assert order["status"] == "FILLED"
        assert order["quantity"] == 10
        assert order["filled_quantity"] == 10
        # The price is the BROKER's average, not a modelled one.
        assert order["fill_price"] == pytest.approx(200.00)
        assert order["broker"]["broker_order_id"] == "brk-order-0001"
        assert order["broker"]["broker_status"] == "filled"

        position = body["position"]
        assert position["quantity"] == 10
        assert position["avg_price"] == pytest.approx(200.00)
        assert await cash(client) == pytest.approx(cash_before - 10 * 200.00)

        # Every broker interaction is auditable with the BROKER's own ids and
        # its own status word (§38, rule 12).
        events = await audit_actions_for(client, order["id"])
        submitted = next(e for e in events if e["action"] == "ORDER_SUBMITTED")
        assert submitted["details"]["broker_order_id"] == "brk-order-0001"
        assert submitted["details"]["broker_status"] == "filled"
        filled = next(e for e in events if e["action"] == "ORDER_FILLED")
        assert filled["details"]["filled_quantity"] == 10
        assert filled["details"]["partial"] is False


async def test_partial_fill_opens_a_partial_position(monkeypatch):
    """PARTIAL FILLS ARE FIRST-CLASS: the position is what FILLED, not what
    was asked for. Opening 10 because we requested 10 would be a position we
    do not actually hold."""
    fake = FakeAlpaca(
        order_body(
            status="partially_filled",
            qty="10",
            filled_qty="4",
            filled_avg_price="199.50",
        )
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="partial-1")
        assert r.status_code == 200
        body = r.json()

        order = body["order"]
        assert order["quantity"] == 10  # requested
        assert order["filled_quantity"] == 4  # actual
        assert order["status"] == "PARTIALLY_FILLED"
        assert order["broker"]["broker_status"] == "partially_filled"

        # The POSITION carries the FILLED quantity.
        assert body["position"]["quantity"] == 4
        assert body["position"]["avg_price"] == pytest.approx(199.50)
        (position,) = await rows(Position)
        assert position.quantity == 4

        # Cash reflects only what actually filled.
        assert await cash(client) == pytest.approx(cash_before - 4 * 199.50)

        events = await audit_actions_for(client, order["id"])
        filled = next(e for e in events if e["action"] == "ORDER_FILLED")
        assert filled["details"]["partial"] is True
        assert filled["details"]["filled_quantity"] == 4
        assert filled["details"]["requested_quantity"] == 10


async def test_zero_fill_accepted_opens_no_position(monkeypatch):
    """An ACCEPTED order with nothing filled is not a trade. No position, no
    cash movement — and the order row records that it is live at the broker."""
    fake = FakeAlpaca(
        order_body(status="accepted", qty="10", filled_qty="0", filled_avg_price=None)
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="zero-1")
        assert r.status_code == 200
        body = r.json()

        assert body["position"] is None
        assert await rows(Position) == []
        assert await cash(client) == pytest.approx(cash_before)

        order = body["order"]
        assert order["status"] == "ACCEPTED"
        assert order["filled_quantity"] == 0
        assert order["broker"]["broker_status"] == "accepted"

        # Audited as exactly that — never a silent gap.
        events = await audit_actions_for(client, order["id"])
        assert not [e for e in events if e["action"] == "ORDER_FILLED"]
        outcomes = [
            e["details"].get("outcome", "")
            for e in events
            if e["action"] == "ORDER_SUBMITTED"
        ]
        assert any("no quantity filled" in o for o in outcomes)


async def test_rejected_order_writes_no_position_and_audits_rejection(monkeypatch):
    fake = FakeAlpaca(
        httpx.Response(403, json={"code": 40310000, "message": "insufficient buying power"})
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="REJ-1")
        assert r.status_code == 422
        assert "insufficient buying power" in r.json()["detail"]["message"]

        assert await rows(Position) == []
        assert await cash(client) == pytest.approx(cash_before)

        # NB: the audit filter upper-cases entity_id, hence the uppercase key.
        events = await audit_actions_for(client, "REJ-1")
        rejected = [e for e in events if e["action"] == "ORDER_REJECTED"]
        assert len(rejected) == 1
        assert "insufficient buying power" in rejected[0]["details"]["reason"]
        assert rejected[0]["details"]["rejected_by"] == "broker"


async def test_duplicate_client_order_id_adopts_without_a_second_post(monkeypatch):
    """A submission whose response we lost still reached the broker. Asking
    first, and adopting what is there, is what stops a doubled position."""
    fake = FakeAlpaca(order_body())
    fake.existing["already-there"] = order_body(
        id="brk-order-EXISTING",
        client_order_id="already-there",
        status="filled",
        qty="10",
        filled_qty="10",
        filled_avg_price="201.25",
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)

        r = await approve(client, quantity=10, client_order_id="already-there")
        assert r.status_code == 200
        body = r.json()

        # NO second POST — the existing order was adopted.
        assert fake.submit_count == 0
        assert body["order"]["broker"]["broker_order_id"] == "brk-order-EXISTING"
        assert body["order"]["fill_price"] == pytest.approx(201.25)
        assert body["position"]["quantity"] == 10

        events = await audit_actions_for(client, body["order"]["id"])
        submitted = next(e for e in events if e["action"] == "ORDER_SUBMITTED")
        assert submitted["details"]["adopted_existing"] is True


async def test_local_duplicate_guard_still_applies(monkeypatch):
    """The LOCAL guard (§42) is kept as well: a replayed API call returns the
    existing row without touching the broker at all."""
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)

        r1 = await approve(client, quantity=10, client_order_id="dup-key")
        assert r1.status_code == 200
        calls_after_first = len(fake.calls)

        r2 = await approve(client, quantity=10, client_order_id="dup-key")
        assert r2.status_code == 200
        assert r2.json()["order"]["id"] == r1.json()["order"]["id"]

        assert len(await rows(Order)) == 1
        assert len(fake.calls) == calls_after_first  # broker never re-contacted


async def test_long_put_submits_the_occ_contract_symbol(monkeypatch):
    """A LONG_PUT is a real broker order on its OCC contract symbol.

    The account is options Level 3, so long calls and puts execute at the
    broker like any other order — they simply address a contract instead of
    the underlying. The symbol is the assertion that matters: submitting the
    bare ticker for an option entry would buy SHARES instead of the contract
    the §8 matrix and the §9 selector actually chose.
    """
    fake = FakeAlpaca(
        order_body(status="filled", qty="1", filled_qty="1", filled_avg_price="4.20")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        # A BEAR view on a long-only account can only be a long put (§5).
        r = await client.post(
            "/api/orders/approve",
            json={"ticker": BULL_TICKER, "quantity": 1, "direction": "BEAR"},
        )

        assert r.status_code == 200, r.text
        assert r.json()["preview"]["proposed"]["instrument"] == "LONG_PUT"

        assert fake.submit_count == 1
        symbol = fake.posted[0]["symbol"]
        # OCC layout: 6-char padded root, YYMMDD, C/P, strike in thousandths.
        assert symbol.startswith(f"{BULL_TICKER:<6}"), symbol
        assert len(symbol) == 21, symbol
        assert symbol[12] == "P", symbol  # a put, not a call
        assert symbol[13:].isdigit(), symbol
        assert symbol != BULL_TICKER  # never the bare underlying

        # The audit trail records the exact string sent to the broker, so the
        # contract that traded is identifiable after the fact (§38).
        audit_rows = (await client.get("/api/audit")).json()
        requested = [a for a in audit_rows if a["action"] == "ORDER_REQUESTED"]
        assert requested and requested[0]["details"]["broker_symbol"] == symbol

        positions = await rows(Position)
        assert len(positions) == 1
        assert positions[0].instrument == "LONG_PUT"
        assert positions[0].multiplier == 100  # contracts, not shares


async def test_close_routes_through_the_broker(monkeypatch):
    """A manual close is a broker SELL, and the position follows the fill."""
    fake = FakeAlpaca(
        order_body(status="filled", qty="10", filled_qty="10", filled_avg_price="200.00")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        r = await approve(client, quantity=10, client_order_id="buy-1")
        assert r.status_code == 200

        fake.order_payload = order_body(
            id="brk-order-SELL",
            side="sell",
            status="filled",
            qty="10",
            filled_qty="10",
            filled_avg_price="210.00",
        )
        fake.poll_payload = None
        fake.posted.clear()

        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 200
        body = r.json()

        assert fake.posted[-1]["side"] == "sell"
        assert body["order"]["side"] == "SELL_TO_CLOSE"
        assert body["order"]["broker"]["broker_order_id"] == "brk-order-SELL"
        assert body["position"]["status"] == "CLOSED"
        # Realized against the BROKER's prices on both legs.
        assert body["realized_pnl"] == pytest.approx((210.00 - 200.00) * 10)


async def test_partial_sell_fill_leaves_an_honestly_partial_position(monkeypatch):
    """A partially filled exit must NOT flatten the local row: the broker
    still holds the rest, and pretending otherwise is the §18 divergence."""
    fake = FakeAlpaca(
        order_body(status="filled", qty="10", filled_qty="10", filled_avg_price="200.00")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="buy-2")).status_code == 200

        fake.order_payload = order_body(
            id="brk-order-PARTIALSELL",
            side="sell",
            status="partially_filled",
            qty="10",
            filled_qty="3",
            filled_avg_price="205.00",
        )
        fake.posted.clear()

        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 200
        body = r.json()

        assert body["order"]["filled_quantity"] == 3
        assert body["position"]["status"] == "OPEN"
        assert body["position"]["quantity"] == 7
        assert body["realized_pnl"] == pytest.approx((205.00 - 200.00) * 3)


async def test_exit_sweep_uses_the_same_broker_path(monkeypatch):
    """Mechanical exits are broker orders too (§18). A sweep that only moved
    local rows is the reconciliation failure this rule exists to prevent."""
    fake = FakeAlpaca(
        order_body(status="filled", qty="10", filled_qty="10", filled_avg_price="200.00")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="buy-3")).status_code == 200

        # Force an exit: a position whose stop is far above the current price
        # trips the §11.1 hard stop on the next evaluation.
        async with SessionLocal() as s:
            position = (await s.execute(select(Position))).scalars().first()
            position.avg_price = 10_000.0
            position.stop_distance = 1.0
            await s.commit()

        fake.order_payload = order_body(
            id="brk-order-EXIT",
            side="sell",
            status="filled",
            qty="10",
            filled_qty="10",
            filled_avg_price="150.00",
        )
        fake.posted.clear()

        r = await client.post("/api/positions/check-exits")
        assert r.status_code == 200
        assert r.json()["exits_triggered"], "the stop should have fired"

        # The exit went to the BROKER — not merely to the local rows.
        assert fake.posted and fake.posted[-1]["side"] == "sell"
        sell = [o for o in await rows(Order) if o.side == "SELL_TO_CLOSE"]
        assert len(sell) == 1
        assert sell[0].broker_order_id == "brk-order-EXIT"
        assert sell[0].filled_quantity == 10


async def test_rejected_exit_leaves_the_position_open_and_is_reported(monkeypatch):
    """A refused exit must NOT close the local row.

    Marking the position closed because we TRIED to exit would create a local
    flat against a real broker long — the §18 divergence, arrived at from the
    protective side. The sweep reports the failure instead.
    """
    fake = FakeAlpaca(
        order_body(status="filled", qty="10", filled_qty="10", filled_avg_price="200.00")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="buy-4")).status_code == 200

        async with SessionLocal() as s:
            position = (await s.execute(select(Position))).scalars().first()
            position.avg_price = 10_000.0
            position.stop_distance = 1.0
            await s.commit()

        # The broker refuses the closing order.
        fake.order_payload = httpx.Response(
            403, json={"message": "position is not closeable right now"}
        )
        fake.posted.clear()

        r = await client.post("/api/positions/check-exits")
        assert r.status_code == 200
        body = r.json()

        assert body["exits_triggered"] == []
        (failure,) = body["exits_failed"]
        assert failure["ticker"] == BULL_TICKER
        assert "not closeable" in failure["reason"]

        # The position is UNCHANGED — still open, still ours, still theirs.
        (position,) = await rows(Position)
        assert position.status == "OPEN"
        assert position.quantity == 10


async def test_broker_status_reports_the_paper_account(monkeypatch):
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        body = (await client.get("/api/broker/status")).json()

        assert body["configured"] is True
        assert body["provider"] == "alpaca_paper"
        assert body["mode"] == "paper"
        assert body["account"]["is_paper"] is True
        assert body["account"]["account_number"] == "PA3ABCDEF"
        assert body["error"] is None


async def test_broker_status_never_echoes_credentials(monkeypatch):
    """Account numbers are identifiers; keys are not, and never appear."""
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        raw = (await client.get("/api/broker/status")).text.lower()

        assert "test-key" not in raw
        assert "test-secret" not in raw
        for fragment in ("api_key", "secret", "apca"):
            assert fragment not in raw


async def test_option_round_trip_conserves_cash_with_the_100x_multiplier(monkeypatch):
    """Open and close a long put at the broker; cash must reconcile exactly.

    An option is quoted PER SHARE and trades in 100-share contracts, so every
    cash movement carries the multiplier. This is the arithmetic most likely to
    be wrong by 100x in either direction, so it is pinned end to end rather
    than per-leg: buy 2 contracts at 4.20, sell them at 6.50, and the account
    must end up exactly 2 * (6.50 - 4.20) * 100 = $460 richer.
    """
    fake = FakeAlpaca(
        order_body(status="filled", qty="2", filled_qty="2", filled_avg_price="4.20")
    )
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = (await client.get("/api/portfolio/risk")).json()["cash"]

        r = await client.post(
            "/api/orders/approve",
            json={"ticker": BULL_TICKER, "quantity": 2, "direction": "BEAR"},
        )
        assert r.status_code == 200, r.text

        cash_after_buy = (await client.get("/api/portfolio/risk")).json()["cash"]
        # 2 contracts * $4.20/share * 100 shares = $840 debited, not $8.40.
        assert cash_after_buy == pytest.approx(cash_before - 840.0, abs=0.01)

        position = (await rows(Position))[0]
        assert position.multiplier == 100
        assert position.max_loss == pytest.approx(840.0)  # premium IS max loss

        # Sell the same contracts back at a higher premium. A distinct broker
        # order id, as a real broker would issue for the closing leg.
        fake.order_payload = order_body(
            id="brk-order-0002",
            status="filled",
            qty="2",
            filled_qty="2",
            filled_avg_price="6.50",
        )
        fake.posted.clear()
        c = await client.post(
            "/api/orders/close", json={"ticker": BULL_TICKER}
        )
        assert c.status_code == 200, c.text

        # The close addressed the CONTRACT, not the underlying.
        assert fake.posted[0]["symbol"] == fake_symbol_of(position)
        assert fake.posted[0]["side"] == "sell"

        cash_final = (await client.get("/api/portfolio/risk")).json()["cash"]
        assert cash_final == pytest.approx(cash_before + 460.0, abs=0.01)

        closed = (await rows(Position))[0]
        assert closed.status == "CLOSED"
        assert closed.realized_pnl == pytest.approx(460.0, abs=0.01)


def fake_symbol_of(position) -> str:
    """The OCC symbol the close path should have addressed."""
    from datetime import date

    from libs.broker.alpaca import occ_option_symbol

    return occ_option_symbol(
        position.ticker,
        date.fromisoformat(position.opt_expiry),
        position.opt_strike,
        position.opt_right,
    )
