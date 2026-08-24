"""Broker reconciliation — GET /api/broker/reconcile (plan §18, §44 rule 18).

THE BEHAVIOUR UNDER TEST, and why it is what it is: when the broker's ledger
and ours disagree, the platform HALTS. It does not pick a winner and it does
not auto-correct. A mismatch means one of two records is wrong and nobody knows
which; overwriting either one would write a guess into the exact rows that are
supposed to be the source of truth — and the likely causes (an unrecorded fill,
an exit that only moved local rows, a manual trade in the broker UI) are
precisely the conditions under which continuing to trade compounds the damage.

So: audit the disagreement, pause trading with an explicit reason, and let a
human decide. Plan §18 lists reconciliation mismatch as a documented
kill-switch trigger; these tests hold that line.
"""
import os
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from apps.gateway.db import (
    Base,
    Portfolio,
    Position,
    SessionLocal,
    SystemState,
    engine,
)
from apps.gateway.main import app
from apps.gateway.routers.broker import (
    MISMATCH_CASH,
    MISMATCH_MISSING_AT_BROKER,
    MISMATCH_MISSING_LOCALLY,
    MISMATCH_QUANTITY,
    PAUSE_REASON_PREFIX,
)
from libs.broker.alpaca import AlpacaPaperBroker
from libs.common.config import get_settings

_ENV_VARS = ("MARKET_DATA_PROVIDER", "LLM_PROVIDER", "BROKER_PROVIDER")

LOCAL_CASH = 90_000.0


def account_payload(cash: float = LOCAL_CASH) -> dict:
    return {
        "account_number": "PA3ABCDEF",
        "cash": f"{cash:.2f}",
        "equity": f"{cash + 2000:.2f}",
        "buying_power": f"{cash * 2:.2f}",
        "currency": "USD",
        "is_paper": True,
    }


def broker_position(symbol: str, qty: int, price: float = 200.0) -> dict:
    return {
        "symbol": symbol,
        "qty": str(qty),
        "avg_entry_price": f"{price:.2f}",
        "market_value": f"{qty * price:.2f}",
    }


class FakeLedger:
    """The real Alpaca adapter over a mocked transport, serving one ledger."""

    def __init__(self, positions: list[dict], cash: float = LOCAL_CASH):
        self.positions = positions
        self.cash = cash

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=account_payload(self.cash))
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=self.positions)
        raise AssertionError(f"unexpected request {request.method} {request.url.path}")

    def broker(self) -> AlpacaPaperBroker:
        return AlpacaPaperBroker(
            api_key="test-key",
            api_secret="test-secret",
            transport=httpx.MockTransport(self.handler),
        )


@asynccontextmanager
async def _client(monkeypatch, ledger: FakeLedger | None, provider="alpaca_paper"):
    previous = {name: os.environ.get(name) for name in _ENV_VARS}
    os.environ["MARKET_DATA_PROVIDER"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["BROKER_PROVIDER"] = provider
    get_settings.cache_clear()
    if ledger is not None:
        monkeypatch.setattr(
            "apps.gateway.deps.get_broker", lambda name: ledger.broker()
        )
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


async def seed_local(positions: list[tuple[str, int]], cash: float = LOCAL_CASH):
    """Local OPEN positions + portfolio cash, written straight to the DB."""
    async with SessionLocal() as s:
        s.add(Portfolio(id=1, cash=cash))
        # Trading starts ENABLED so a pause is observable as a change.
        s.add(
            SystemState(
                id=1, trading_enabled=True, reason="enabled for the test", updated_by="test"
            )
        )
        for ticker, quantity in positions:
            s.add(
                Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_price=200.0,
                    max_loss=400.0,
                    stop_distance=20.0,
                    entry_edge=0.5,
                    entry_bar_date="2026-08-07",
                )
            )
        await s.commit()


async def trading_enabled(client) -> bool:
    return (await client.get("/api/trading/status")).json()["trading_enabled"]


async def kill_switch_events(client) -> list[dict]:
    r = await client.get("/api/audit", params={"action": "KILL_SWITCH_TRIGGERED"})
    assert r.status_code == 200
    return r.json()


# ===========================================================================
# In sync
# ===========================================================================


async def test_in_sync_reports_no_mismatches_and_does_not_pause(monkeypatch):
    ledger = FakeLedger([broker_position("AAPL", 10), broker_position("MSFT", 3)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10), ("MSFT", 3)])

        r = await client.get("/api/broker/reconcile")
        assert r.status_code == 200
        body = r.json()

        assert body["configured"] is True
        assert body["in_sync"] is True
        assert body["mismatches"] == []
        assert body["as_of"]
        assert body["broker"]["account"]["account_number"] == "PA3ABCDEF"
        assert {p["symbol"]: p["quantity"] for p in body["local"]["positions"]} == {
            "AAPL": 10,
            "MSFT": 3,
        }

        # Agreement changes nothing: trading stays exactly as it was.
        assert await trading_enabled(client) is True
        assert await kill_switch_events(client) == []


async def test_cash_within_tolerance_is_in_sync(monkeypatch):
    """Cents of float/rounding drift are not a divergence."""
    ledger = FakeLedger([], cash=LOCAL_CASH + 0.4)
    async with _client(monkeypatch, ledger) as client:
        await seed_local([], cash=LOCAL_CASH)

        body = (await client.get("/api/broker/reconcile")).json()
        assert body["in_sync"] is True


# ===========================================================================
# Mismatches: reported, audited, and trading paused
# ===========================================================================


async def test_quantity_mismatch_is_reported_audited_and_pauses_trading(monkeypatch):
    ledger = FakeLedger([broker_position("AAPL", 4)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10)])
        assert await trading_enabled(client) is True

        body = (await client.get("/api/broker/reconcile")).json()

        assert body["in_sync"] is False
        (mismatch,) = [
            m for m in body["mismatches"] if m["kind"] == MISMATCH_QUANTITY
        ]
        assert mismatch["symbol"] == "AAPL"
        assert mismatch["broker"] == 4
        assert mismatch["local"] == 10
        assert "partial" in mismatch["detail"]

        # PAUSED (§18) — with an explicit, human-readable reason.
        assert await trading_enabled(client) is False
        status = (await client.get("/api/trading/status")).json()
        assert status["reason"].startswith(PAUSE_REASON_PREFIX)
        assert "AAPL" in status["reason"]

        # AUDITED as a SYSTEM event naming the trigger.
        (event,) = await kill_switch_events(client)
        assert event["actor_type"] == "SYSTEM"
        assert event["details"]["trigger"] == "BROKER_RECONCILIATION_MISMATCH"
        assert event["details"]["auto_corrected"] is False
        assert event["details"]["mismatches"][0]["symbol"] == "AAPL"


async def test_missing_position_at_broker_is_reported_and_pauses(monkeypatch):
    """We think we hold it; the broker does not. Either an entry never filled
    or an exit only moved local rows — both are exactly what §18 fears."""
    ledger = FakeLedger([])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10)])

        body = (await client.get("/api/broker/reconcile")).json()

        (mismatch,) = body["mismatches"]
        assert mismatch["kind"] == MISMATCH_MISSING_AT_BROKER
        assert mismatch["symbol"] == "AAPL"
        assert mismatch["broker"] is None
        assert mismatch["local"] == 10
        assert body["in_sync"] is False

        assert await trading_enabled(client) is False
        assert len(await kill_switch_events(client)) == 1


async def test_position_missing_locally_is_reported_and_pauses(monkeypatch):
    """The broker holds something we have no row for — a fill we never
    recorded, or a trade placed outside the platform."""
    ledger = FakeLedger([broker_position("TSLA", 5)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([])

        body = (await client.get("/api/broker/reconcile")).json()

        (mismatch,) = body["mismatches"]
        assert mismatch["kind"] == MISMATCH_MISSING_LOCALLY
        assert mismatch["symbol"] == "TSLA"
        assert mismatch["broker"] == 5
        assert mismatch["local"] is None

        assert await trading_enabled(client) is False


async def test_cash_is_never_compared_because_no_copy_exists(monkeypatch):
    """THE PLATFORM STORES NO COPY OF CASH (user rule: the account is the
    broker's), so there is nothing for cash to disagree WITH: reconciliation
    reports local cash as null and never emits a CASH_MISMATCH — the position
    ledger is what it compares."""
    ledger = FakeLedger([], cash=LOCAL_CASH - 5_000.0)
    async with _client(monkeypatch, ledger) as client:
        await seed_local([], cash=LOCAL_CASH)  # a stale row changes nothing

        body = (await client.get("/api/broker/reconcile")).json()

        assert body["local"]["cash"] is None
        assert [m for m in body["mismatches"] if m["kind"] == MISMATCH_CASH] == []
        assert body["in_sync"] is True
        assert await trading_enabled(client) is True


async def test_mismatch_does_not_auto_correct_either_ledger(monkeypatch):
    """THE CENTRAL GUARANTEE: halting is the action; rewriting is not.

    Neither the local position nor local cash may be "fixed" to agree with the
    broker. A human decides which ledger is right.
    """
    ledger = FakeLedger([broker_position("AAPL", 4)], cash=LOCAL_CASH - 5_000.0)
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10)], cash=LOCAL_CASH)

        assert (await client.get("/api/broker/reconcile")).json()["in_sync"] is False

        async with SessionLocal() as s:
            position = (await s.execute(select(Position))).scalars().one()
            portfolio = await s.get(Portfolio, 1)
            # Untouched, both of them.
            assert position.quantity == 10
            assert position.status == "OPEN"
            assert portfolio.cash == pytest.approx(LOCAL_CASH)


async def test_repeated_reconcile_stays_paused_and_keeps_reporting(monkeypatch):
    """Reconciling again while paused reports the same truth — and does not
    quietly resume anything."""
    ledger = FakeLedger([broker_position("AAPL", 4)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10)])

        first = (await client.get("/api/broker/reconcile")).json()
        second = (await client.get("/api/broker/reconcile")).json()

        assert first["mismatches"] == second["mismatches"]
        assert await trading_enabled(client) is False
        # Each detection is its own audit event — the history is not collapsed.
        assert len(await kill_switch_events(client)) == 2


async def test_option_positions_reconcile_by_occ_symbol(monkeypatch):
    """Options are broker-executed (§30.13): a local option position compares
    against the broker's holding of the SAME OCC contract symbol — matching
    quantity is in_sync, an absent contract is a real MISSING_AT_BROKER."""
    occ = "MSFT260918C00400000"

    def option_row():
        return Position(
            ticker="MSFT",
            instrument="LONG_CALL",
            quantity=2,
            avg_price=3.5,
            max_loss=700.0,
            stop_distance=3.5,
            entry_edge=0.4,
            entry_bar_date="2026-08-07",
            opt_expiry="2026-09-18",
            opt_strike=400.0,
            opt_right="C",
            multiplier=100,
        )

    # Broker holds the same contract, same quantity -> in sync.
    ledger = FakeLedger([broker_position("AAPL", 10), broker_position(occ, 2)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([("AAPL", 10)])
        async with SessionLocal() as s:
            s.add(option_row())
            await s.commit()

        body = (await client.get("/api/broker/reconcile")).json()
        assert body["in_sync"] is True
        assert await trading_enabled(client) is True


async def test_option_position_missing_at_broker_is_a_mismatch(monkeypatch):
    """The inverse: a local option long the broker does not hold pauses
    trading like any other divergence — an option fill that never landed at
    the broker is exactly the §18 failure reconciliation exists to catch."""
    ledger = FakeLedger([])
    async with _client(monkeypatch, ledger) as client:
        async with SessionLocal() as s:
            s.add(
                Position(
                    ticker="MSFT",
                    instrument="LONG_CALL",
                    quantity=2,
                    avg_price=3.5,
                    max_loss=700.0,
                    stop_distance=3.5,
                    entry_edge=0.4,
                    entry_bar_date="2026-08-07",
                    opt_expiry="2026-09-18",
                    opt_strike=400.0,
                    opt_right="C",
                    multiplier=100,
                )
            )
            await s.commit()

        body = (await client.get("/api/broker/reconcile")).json()
        assert body["in_sync"] is False
        kinds = {m["kind"] for m in body["mismatches"]}
        assert "MISSING_AT_BROKER" in kinds
        assert any(m["symbol"] == "MSFT260918C00400000" for m in body["mismatches"])
        assert await trading_enabled(client) is False  # §18 kill switch fired


# ===========================================================================
# Unconfigured: nothing to reconcile, and nothing paused
# ===========================================================================


async def test_unconfigured_reports_not_configured_and_pauses_nothing(
    unconfigured_client,
):
    """An ABSENT ledger is not a disagreement between ledgers."""
    await seed_local([("AAPL", 10)])

    r = await unconfigured_client.get("/api/broker/reconcile")
    assert r.status_code == 200
    body = r.json()

    assert body["configured"] is False
    assert body["broker"] is None
    assert body["mismatches"] == []
    # NOT in sync either — we simply do not know, and say so.
    assert body["in_sync"] is False
    assert "BROKER_PROVIDER" in body["message"]

    # Local facts are still reported; nothing was paused.
    assert body["local"]["positions"] == [{"symbol": "AAPL", "quantity": 10}]
    assert await trading_enabled(unconfigured_client) is True
    assert await kill_switch_events(unconfigured_client) == []


async def test_simulated_mode_has_no_ledger_to_reconcile(monkeypatch):
    async with _client(monkeypatch, None, provider="simulated") as client:
        await seed_local([("AAPL", 10)])

        body = (await client.get("/api/broker/reconcile")).json()

        assert body["configured"] is False
        assert body["broker"] is None
        assert body["mismatches"] == []
        assert "simulated" in body["message"]
        assert await trading_enabled(client) is True


async def test_broker_read_failure_is_not_treated_as_a_mismatch(monkeypatch):
    """A fault teaches us nothing about the broker's ledger. Pausing on an
    unknown would be acting on a guess in the opposite direction."""

    class Broken(FakeLedger):
        def handler(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

    async with _client(monkeypatch, Broken([])) as client:
        await seed_local([("AAPL", 10)])

        body = (await client.get("/api/broker/reconcile")).json()

        assert body["configured"] is True
        assert body["broker"] is None
        assert body["mismatches"] == []
        assert body["in_sync"] is False
        assert "NOT a mismatch" in body["message"]
        # Nothing paused: we did not learn that anything is wrong.
        assert await trading_enabled(client) is True
        assert await kill_switch_events(client) == []


# ===========================================================================
# The pause has TEETH: it must actually stop the next entry
# ===========================================================================


async def test_pause_from_a_mismatch_actually_blocks_the_next_approve(monkeypatch):
    """The consequence the pause exists for, verified end to end.

    Flipping ``system_state.trading_enabled`` and writing an audit row is only
    bookkeeping unless it STOPS something. Here a mismatch pauses trading and
    the very next entry attempt is refused by the §10 kill-switch gate — the
    property the reconciliation design is actually claiming.
    """
    ticker = "AAPL"
    # The broker holds 4; we think we hold 10 — a QUANTITY mismatch.
    ledger = FakeLedger([broker_position(ticker, 4)])
    async with _client(monkeypatch, ledger) as client:
        await seed_local([(ticker, 10)])
        # Authorize a DIFFERENT symbol for entry, so the refusal can only come
        # from the global kill switch and not from a per-symbol restriction.
        entry = "GOOGL"
        await client.post("/api/watchlist", json={"ticker": entry})
        await client.post(
            "/api/trading-pool", json={"ticker": entry, "acknowledge_risks": True}
        )
        await client.post(f"/api/trading-pool/{entry}/trading", json={"enabled": True})
        assert await trading_enabled(client) is True

        assert (await client.get("/api/broker/reconcile")).json()["in_sync"] is False
        assert await trading_enabled(client) is False

        # THE POINT: the next entry is refused, and the §10 chain names the
        # kill switch as the reason.
        r = await client.post("/api/orders/approve", json={"ticker": entry})
        assert r.status_code == 422
        gates = {g["name"]: g for g in r.json()["detail"]["preview"]["gates"]}
        pool_gate = gates["TRADING_POOL_AUTHORIZATION"]
        assert pool_gate["status"] == "FAIL"
        assert "kill switch" in pool_gate["detail"]

        # And nothing was placed.
        async with SessionLocal() as s:
            opened = (
                await s.execute(select(Position).where(Position.ticker == entry))
            ).scalars().all()
        assert list(opened) == []
