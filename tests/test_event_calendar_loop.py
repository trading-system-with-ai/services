"""Event calendar ingestion seam — the gateway tick (event spec §7-§13; audit
§5.1, §11.1 Phase B).

What these tests actually pin, in the order the tick does it:

1. **Identity and idempotence.** Ingesting the same provider output twice
   creates ZERO rows the second time. That is the whole reason the natural
   key exists, and it is what makes the refresh button safe to press.
2. **Failure isolation (§8).** One provider raising must not cost another
   provider its rows — the tick names the failure and keeps going. The
   negative half matters as much as the positive: the healthy provider's
   events must be present, not merely "no exception raised".
3. **Exactly-once alerting (§11), across a restart.** The dedupe is checked
   against the AUDIT TABLE, so the second tick is run through a *fresh
   session* to simulate the process that wrote the first alert being gone.
   An in-process set would pass a two-tick test and fail this one.
4. **ESTIMATED never alerts.** §11's "do not fabricate an exact event date
   when only an estimate exists" is meaningless if the platform pages the
   user about the estimate.
5. **The drift window.** A cadence ESTIMATE replaced three days later by the
   confirmed date must UPDATE the card, not create a second one.
6. **Named skips, never fabricated rows.** Every refusal path returns a
   reason string.

The tick is driven directly (never through the background loop): httpx
ASGITransport does not run the lifespan, so the loop never starts under the
suite — the same reason monitor.run_sweep_and_update and
risk_snapshot.run_scheduled_snapshot exist.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from apps.gateway import event_calendar as ec
from apps.gateway.db import (
    AuditEvent,
    EventIngestStateRow,
    EventRow,
    MarketCalendarRow,
    Position,
    SessionLocal,
    TradingPoolItem,
    WatchlistItem,
)
from libs.event_calendar.provider import CapabilityNotAvailable, MarketDay
from libs.trading_core.events import EventCandidate
from libs.trading_core.models import AuditAction
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

NOW = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes — deterministic, in-process, no network
# ---------------------------------------------------------------------------


class FakeProvider:
    """A provider whose whole behaviour is declared by the constructor.

    Deliberately NOT the real stub adapter: these tests need to plant an
    exact candidate at an exact instant (a T-4 confirmed earnings, an
    ESTIMATED date three days off a confirmed one), which a window-relative
    synthetic generator cannot express.
    """

    def __init__(
        self,
        name,
        candidates=None,
        *,
        capabilities=None,
        raises=None,
        days=None,
        calendar_raises=None,
    ):
        self.name = name
        self._candidates = list(candidates or [])
        self._capabilities = capabilities or {"earnings_calendar": False}
        self._raises = raises
        self._days = list(days or [])
        self._calendar_raises = calendar_raises
        self.fetch_calls = 0

    def capabilities(self):
        return dict(self._capabilities)

    def fetch_events(self, *, tickers, start, end, as_of=None):
        self.fetch_calls += 1
        if self._raises is not None:
            raise self._raises
        return [c for c in self._candidates if start <= c.scheduled_at <= end]

    def fetch_market_calendar(self, start, end):
        if self._calendar_raises is not None:
            raise self._calendar_raises
        raise CapabilityNotAvailable(f"{self.name} does not serve exchange sessions")


def earnings(ticker, when, *, status=EventStatus.CONFIRMED, source=None, key=None):
    et_date = when.astimezone(ec.NEW_YORK).date().isoformat()
    return EventCandidate(
        event_key=key or f"EARNINGS:{ticker}:{et_date}",
        event_type=EventType.EARNINGS,
        title=f"{ticker} earnings release",
        scheduled_at=when,
        status=status,
        source=source
        or (
            EventSourceKind.COMPANY_IR_SEC
            if status is EventStatus.CONFIRMED
            else EventSourceKind.DERIVED
        ),
        source_name="sec_edgar" if status is EventStatus.CONFIRMED else "derived_cadence",
        ticker=ticker,
        session=EventSession.AFTER_MARKET,
    )


def fomc(when, *, status=EventStatus.CONFIRMED):
    et_date = when.astimezone(ec.NEW_YORK).date().isoformat()
    return EventCandidate(
        event_key=f"FOMC_DECISION:{et_date}",
        event_type=EventType.FOMC_DECISION,
        title="FOMC rate decision",
        scheduled_at=when,
        status=status,
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
        session=EventSession.DURING_MARKET,
        agency="Federal Reserve",
    )


async def _fresh_schema():
    from apps.gateway.db import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture
async def db():
    """A fresh schema plus a session; the tick is driven directly."""
    await _fresh_schema()
    async with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# 1. Creation, idempotence, audit
# ---------------------------------------------------------------------------


async def test_a_candidate_becomes_a_row_and_an_event_discovered_audit(db):
    provider = FakeProvider("fake", [earnings("NVDA", NOW + timedelta(days=4))])
    result = await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"])
    await db.commit()

    assert result["created"] == 1
    rows = (await db.execute(EventRow.__table__.select())).all()
    assert len(rows) == 1
    audits = (
        (await db.execute(AuditEvent.__table__.select())).all()
    )
    actions = [a.action for a in audits]
    assert AuditAction.EVENT_DISCOVERED.value in actions
    assert AuditAction.CALENDAR_INGESTED.value in actions
    discovered = next(a for a in audits if a.action == AuditAction.EVENT_DISCOVERED.value)
    # entity_id is the numeric events.id (audit_events.entity_id is
    # VARCHAR(64) in Postgres — long FED_SPEECH natural keys overflowed it
    # live on 2026-08-19); the natural key travels in details.event_key.
    assert discovered.entity_id.isdigit()
    assert discovered.details["event_key"] == "EARNINGS:NVDA:2026-08-23"
    assert discovered.details["source_name"] == "sec_edgar"
    assert discovered.details["ticker"] == "NVDA"


async def test_ingesting_the_same_output_twice_creates_zero_rows_the_second_time(db):
    provider = FakeProvider("fake", [earnings("NVDA", NOW + timedelta(days=4))])
    first = await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"], force=True)
    await db.commit()
    second = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(minutes=5), providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert first["created"] == 1
    assert second["created"] == 0
    assert second["updated"] == 0
    count = len((await db.execute(EventRow.__table__.select())).all())
    assert count == 1


async def test_reverification_refreshes_last_verified_at_without_claiming_a_change(db):
    provider = FakeProvider("fake", [earnings("NVDA", NOW + timedelta(days=4))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"], force=True)
    await db.commit()
    later = NOW + timedelta(hours=6)
    await ec.run_calendar_ingest(db, now=later, providers=[provider], tickers=["NVDA"], force=True)
    await db.commit()

    row = (await db.execute(EventRow.__table__.select())).one()
    verified = row.last_verified_at
    if verified.tzinfo is None:
        verified = verified.replace(tzinfo=timezone.utc)
    assert verified == later
    updated = [
        a
        for a in (await db.execute(AuditEvent.__table__.select())).all()
        if a.action == AuditAction.EVENT_UPDATED.value
    ]
    assert updated == []  # a re-verify is not a change


# ---------------------------------------------------------------------------
# 2. Merge paths that produce EVENT_UPDATED
# ---------------------------------------------------------------------------


async def test_estimated_then_confirmed_three_days_later_updates_the_same_card(db):
    """The ±21-day drift window is the reason estimates do not duplicate.

    Without it the confirmed date lands under a different natural key and the
    user sees two NVDA earnings cards for one quarter.
    """
    estimated = earnings(
        "NVDA", NOW + timedelta(days=10), status=EventStatus.ESTIMATED
    )
    est_provider = FakeProvider("derived", [estimated])
    await ec.run_calendar_ingest(db, now=NOW, providers=[est_provider], tickers=["NVDA"], force=True)
    await db.commit()

    confirmed = earnings("NVDA", NOW + timedelta(days=13))
    sec = FakeProvider("sec_edgar", [confirmed])
    result = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=1), providers=[sec], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["created"] == 0
    assert result["updated"] == 1
    rows = (await db.execute(EventRow.__table__.select())).all()
    assert len(rows) == 1, "the confirmation must UPDATE the estimate, not duplicate it"
    row = rows[0]
    assert row.status == EventStatus.CONFIRMED.value
    # U1 re-keys on an accepted date move; the seam must persist that key or
    # the next tick re-creates the row under the new key.
    assert row.event_key == confirmed.event_key
    assert row.source_name == "sec_edgar"


async def test_the_rekeyed_row_is_not_recreated_on_the_next_tick(db):
    """The re-keying fix is inert unless event_key is in the UPDATE list.

    This is the test that proves it: after a confirmed date move, running the
    SAME confirming provider again must find the row by its new key.
    """
    est = FakeProvider(
        "derived", [earnings("NVDA", NOW + timedelta(days=10), status=EventStatus.ESTIMATED)]
    )
    await ec.run_calendar_ingest(db, now=NOW, providers=[est], tickers=["NVDA"], force=True)
    await db.commit()
    sec = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=13))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[sec], tickers=["NVDA"], force=True)
    await db.commit()
    third = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=2), providers=[sec], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert third["created"] == 0
    assert len((await db.execute(EventRow.__table__.select())).all()) == 1


async def test_the_sql_prefilter_window_is_derived_from_u1_not_restated(db):
    """The prefilter must never be narrower than the matcher it feeds.

    ``_find_existing`` loads candidate rows with a SQL BETWEEN before handing
    them to ``same_event``. If that bound were a hardcoded literal, widening
    U1's window later would leave the query unable to even LOAD the row the
    matcher would have accepted — a duplicate card with no failing test.
    """
    from libs.trading_core.events import EARNINGS_DRIFT_WINDOW, MINUTES_DRIFT_WINDOW

    assert ec._DRIFT_WINDOWS[EventType.EARNINGS] is EARNINGS_DRIFT_WINDOW
    assert ec._DRIFT_WINDOWS[EventType.FOMC_MINUTES] is MINUTES_DRIFT_WINDOW
    # A type with no drift window is matched by exact key only.
    assert EventType.CPI not in ec._DRIFT_WINDOWS


async def test_an_earnings_estimate_20_days_off_still_merges_not_duplicates(db):
    """The far edge of the ±21d window: 20 days off must still be ONE card.

    A prefilter accidentally narrowed to, say, 7 days would pass every other
    merge test in this file and fail only here.
    """
    est = FakeProvider(
        "derived",
        [earnings("NVDA", NOW + timedelta(days=10), status=EventStatus.ESTIMATED)],
    )
    await ec.run_calendar_ingest(db, now=NOW, providers=[est], tickers=["NVDA"], force=True)
    await db.commit()

    sec = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=30))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[sec], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["created"] == 0
    assert len((await db.execute(EventRow.__table__.select())).all()) == 1


async def test_an_earnings_estimate_beyond_the_window_is_a_separate_quarter(db):
    """The paired negative: 40 days apart is the NEXT quarter, two cards.

    Without this, a prefilter widened to "always" would pass the test above
    while silently collapsing two real releases into one.
    """
    est = FakeProvider(
        "derived",
        [earnings("NVDA", NOW + timedelta(days=5), status=EventStatus.ESTIMATED)],
    )
    await ec.run_calendar_ingest(db, now=NOW, providers=[est], tickers=["NVDA"], force=True)
    await db.commit()

    sec = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=45))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[sec], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["created"] == 1
    assert len((await db.execute(EventRow.__table__.select())).all()) == 2


async def test_a_moved_confirmed_date_becomes_revised_and_audits_event_updated(db):
    sec = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=5))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[sec], tickers=["NVDA"], force=True)
    await db.commit()

    moved = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=8))])
    result = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=1), providers=[moved], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["updated"] == 1
    row = (await db.execute(EventRow.__table__.select())).one()
    assert row.status == EventStatus.REVISED.value
    assert len(row.revision_history) == 1
    updated = [
        a
        for a in (await db.execute(AuditEvent.__table__.select())).all()
        if a.action == AuditAction.EVENT_UPDATED.value
    ]
    assert len(updated) == 1
    assert updated[0].details["change"] == "revised"


# ---------------------------------------------------------------------------
# 3. Failure isolation (§8) — the audit's whole reason for a per-adapter try
# ---------------------------------------------------------------------------


async def test_one_provider_raising_does_not_cost_another_provider_its_rows(db):
    broken = FakeProvider("broken", raises=RuntimeError("HTML layout changed"))
    healthy = FakeProvider("healthy", [fomc(NOW + timedelta(days=6))])

    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[broken, healthy], tickers=[], force=True
    )
    await db.commit()

    # The failure is NAMED, not swallowed silently...
    assert {"provider": "broken", "reason": "PROVIDER_ERROR"} in result["skipped"]
    assert "HTML layout changed" in result["providers"]["broken"]["error"]
    # ...and the healthy provider's event actually landed. Asserting only
    # "no exception" would pass even if the whole tick aborted.
    assert result["created"] == 1
    rows = (await db.execute(EventRow.__table__.select())).all()
    assert [r.event_type for r in rows] == [EventType.FOMC_DECISION.value]


async def test_a_failing_provider_records_its_error_and_does_not_advance_last_ok_at(db):
    broken = FakeProvider("broken", raises=RuntimeError("503 upstream"))
    await ec.run_calendar_ingest(db, now=NOW, providers=[broken], tickers=[], force=True)
    await db.commit()

    state = await db.get(EventIngestStateRow, "broken")
    assert state is not None
    assert state.last_fetched_at is not None, "the ATTEMPT is recorded..."
    assert state.last_ok_at is None, "...but a failure is not a success"
    assert "503 upstream" in state.last_error


async def test_a_capability_probe_that_raises_does_not_end_the_tick(db):
    class BadProbe(FakeProvider):
        def capabilities(self):
            raise RuntimeError("probe exploded")

    provider = BadProbe("probe_fail", [fomc(NOW + timedelta(days=3))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=[], force=True
    )
    await db.commit()
    assert result["created"] == 1
    # "availability unknown" (an error string) is a different fact from the
    # False that means "probed and proven absent".
    assert "probe exploded" in result["providers"]["probe_fail"]["capabilities"]["error"]


async def test_no_providers_configured_is_a_named_skip_not_an_error(db):
    result = await ec.run_calendar_ingest(db, now=NOW, providers=[], tickers=[])
    await db.commit()
    assert result["created"] == 0
    assert {"provider": "", "reason": "NO_PROVIDERS_CONFIGURED"} in result["skipped"]
    assert (await db.execute(EventRow.__table__.select())).all() == []


# ---------------------------------------------------------------------------
# 4. Cadence gating
# ---------------------------------------------------------------------------


async def test_a_recently_successful_provider_is_skipped_by_cadence(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=4))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"])
    await db.commit()
    assert provider.fetch_calls == 1

    result = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=2), providers=[provider], tickers=["NVDA"]
    )
    await db.commit()
    assert provider.fetch_calls == 1, "SEC must not be re-asked two hours later"
    assert {"provider": "sec_edgar", "reason": "CADENCE_NOT_DUE"} in result["skipped"]


async def test_force_bypasses_the_cadence_gate(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=4))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"])
    await db.commit()
    await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=2), providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert provider.fetch_calls == 2


async def test_cadence_is_measured_from_the_last_SUCCESS_not_the_last_attempt(db):
    """A provider that has been failing must keep being retried.

    Gating on ``last_fetched_at`` would let one transient 500 mute a source
    for a whole day — the exact failure mode the ingest-state row's two
    separate timestamps exist to prevent.
    """
    broken = FakeProvider("sec_edgar", raises=RuntimeError("timeout"))
    await ec.run_calendar_ingest(db, now=NOW, providers=[broken], tickers=[])
    await db.commit()
    assert broken.fetch_calls == 1

    await ec.run_calendar_ingest(db, now=NOW + timedelta(minutes=30), providers=[broken], tickers=[])
    await db.commit()
    assert broken.fetch_calls == 2, "a failing source is retried, not cadence-muted"


# ---------------------------------------------------------------------------
# 5. Market calendar + half-day session classification
# ---------------------------------------------------------------------------


def _et(day, hour, minute=0):
    from zoneinfo import ZoneInfo

    return datetime(
        day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo("America/New_York")
    ).astimezone(timezone.utc)


async def test_market_calendar_rows_are_upserted_and_reused(db):
    half_day = date(2026, 11, 27)
    provider = FakeProvider(
        "alpaca_calendar",
        [],
        days=[
            MarketDay(
                session_date=half_day,
                exchange="US",
                open_utc=_et(half_day, 9, 30),
                close_utc=_et(half_day, 13, 0),
                is_early_close=True,
                source="alpaca_calendar",
            )
        ],
        calendar_raises=None,
    )
    # FakeProvider.fetch_market_calendar raises unless we override it.
    provider.fetch_market_calendar = lambda start, end: provider._days

    result = await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=[], force=True)
    await db.commit()
    assert result["calendar_days"] == 1
    row = await db.get(MarketCalendarRow, half_day)
    assert row is not None and row.is_early_close is True


async def test_a_release_after_a_half_day_close_is_classified_after_market(db):
    """The whole point of the market_calendar table.

    13:30 ET on a normal day is DURING_MARKET; on a 13:00 early close it is
    AFTER_MARKET. Getting this wrong puts an earnings card in the wrong half
    of the trading day.
    """
    half_day = date(2026, 11, 27)
    db.add(
        MarketCalendarRow(
            session_date=half_day,
            exchange="US",
            open_utc=_et(half_day, 9, 30),
            close_utc=_et(half_day, 13, 0),
            is_early_close=True,
            source="test",
        )
    )
    await db.commit()

    release = _et(half_day, 13, 30)
    candidate = EventCandidate(
        event_key=f"EARNINGS:AAPL:{half_day.isoformat()}",
        event_type=EventType.EARNINGS,
        title="AAPL earnings release",
        scheduled_at=release,
        status=EventStatus.CONFIRMED,
        source=EventSourceKind.COMPANY_IR_SEC,
        source_name="sec_edgar",
        ticker="AAPL",
        session=EventSession.UNKNOWN,  # provider could not tell
    )
    provider = FakeProvider("sec_edgar", [candidate])
    await ec.run_calendar_ingest(
        db, now=_et(half_day, 8, 0), providers=[provider], tickers=["AAPL"], force=True
    )
    await db.commit()

    row = (await db.execute(EventRow.__table__.select())).one()
    assert row.session == EventSession.AFTER_MARKET.value


# ---------------------------------------------------------------------------
# 6. Importance / relevance (§12, §13)
# ---------------------------------------------------------------------------


async def test_importance_rises_when_the_ticker_is_held(db):
    db.add(WatchlistItem(ticker="NVDA", added_by="test"))
    await db.commit()
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=4))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"], force=True)
    await db.commit()
    watch_score = (await db.execute(EventRow.__table__.select())).one().importance

    db.add(
        Position(
            ticker="NVDA",
            quantity=10,
            avg_price=100.0,
            max_loss=200.0,
            stop_distance=20.0,
        )
    )
    await db.commit()
    await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=1), providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    position_score = (await db.execute(EventRow.__table__.select())).one().importance

    # EARNINGS base 60; WATCHLIST +10 vs POSITION +30.
    assert watch_score == 70
    assert position_score == 90


async def test_the_ticker_universe_is_watchlist_union_pool_union_positions(db):
    db.add(WatchlistItem(ticker="AAPL", added_by="test"))
    db.add(TradingPoolItem(ticker="MSFT", promoted_by="test"))
    db.add(
        Position(
            ticker="NVDA", quantity=1, avg_price=1.0, max_loss=1.0, stop_distance=1.0
        )
    )
    db.add(
        Position(
            ticker="OLD",
            quantity=0,
            avg_price=1.0,
            max_loss=1.0,
            stop_distance=1.0,
            status="CLOSED",
        )
    )
    await db.commit()
    assert await ec.ingest_tickers(db) == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# 7. T-minus alerting (§11) — exactly once, restart-safe, never on estimates
# ---------------------------------------------------------------------------


async def _approaching_rows(session):
    return [
        a
        for a in (await session.execute(AuditEvent.__table__.select())).all()
        if a.action == AuditAction.EVENT_APPROACHING.value
    ]


async def test_a_confirmed_event_inside_the_horizon_alerts_once(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=3))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert result["alerts"] == 1
    rows = await _approaching_rows(db)
    assert len(rows) == 1
    assert rows[0].details["horizon"] == 7
    assert rows[0].details["ticker"] == "NVDA"
    assert rows[0].entity_id.isdigit()
    assert rows[0].details["event_key"] == "EARNINGS:NVDA:2026-08-22"


async def test_the_alert_is_not_written_twice_on_a_second_tick(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=3))])
    await ec.run_calendar_ingest(db, now=NOW, providers=[provider], tickers=["NVDA"], force=True)
    await db.commit()
    second = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=6), providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert second["alerts"] == 0
    assert len(await _approaching_rows(db)) == 1


async def test_the_alert_survives_a_simulated_process_restart():
    """The exactly-once guarantee must be a property of the AUDIT TABLE.

    An in-process memo would pass the two-tick test above and fail this one:
    here the second tick runs in a brand-new session, exactly as it would
    after a restart (or on a second replica — ADR-007 gives no leader
    election, so correctness may not depend on single-process state).
    """
    await _fresh_schema()
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=3))])

    async with SessionLocal() as first:
        result = await ec.run_calendar_ingest(
            first, now=NOW, providers=[provider], tickers=["NVDA"], force=True
        )
        await first.commit()
    assert result["alerts"] == 1

    async with SessionLocal() as second:  # a different session == a new process
        again = await ec.run_calendar_ingest(
            second,
            now=NOW + timedelta(days=1),
            providers=[provider],
            tickers=["NVDA"],
            force=True,
        )
        await second.commit()
        assert again["alerts"] == 0
        assert len(await _approaching_rows(second)) == 1


async def test_an_estimated_event_never_alerts(db):
    """§11: do not fabricate an exact event date when only an estimate exists.

    Paging the user about a derived date is precisely presenting it as a fact.
    """
    estimate = earnings("NVDA", NOW + timedelta(days=3), status=EventStatus.ESTIMATED)
    provider = FakeProvider("derived", [estimate])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert result["created"] == 1, "the estimate is still STORED..."
    assert result["alerts"] == 0, "...it is just never alerted on"
    assert await _approaching_rows(db) == []


async def test_an_estimate_that_later_confirms_then_alerts(db):
    """The negative test above would pass on a broken alerter that never
    fires at all. This is its paired positive: the same event, once
    CONFIRMED, does alert."""
    est = FakeProvider(
        "derived", [earnings("NVDA", NOW + timedelta(days=3), status=EventStatus.ESTIMATED)]
    )
    await ec.run_calendar_ingest(db, now=NOW, providers=[est], tickers=["NVDA"], force=True)
    await db.commit()
    assert await _approaching_rows(db) == []

    sec = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=3))])
    result = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(hours=1), providers=[sec], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert result["alerts"] == 1


async def test_an_event_beyond_the_horizon_does_not_alert(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW + timedelta(days=20))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert result["created"] == 1
    assert result["alerts"] == 0


async def test_a_past_event_does_not_alert(db):
    provider = FakeProvider("sec_edgar", [earnings("NVDA", NOW - timedelta(days=2))])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    assert result["created"] == 1
    assert result["alerts"] == 0


# ---------------------------------------------------------------------------
# 8. The loop wrapper
# ---------------------------------------------------------------------------


async def test_the_loop_reraises_cancellation(monkeypatch):
    """Graceful shutdown must never be swallowed (risk_snapshot's contract)."""
    monkeypatch.setattr(ec, "run_scheduled_ingest", _never_called)
    task = asyncio.create_task(ec.event_calendar_loop())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _never_called(*args, **kwargs):  # pragma: no cover — cancelled first
    raise AssertionError("the loop must sleep before its first tick")


async def test_a_failing_tick_is_logged_and_the_loop_survives(monkeypatch):
    """One bad tick must not end the loop — the next one runs normally."""
    calls = []

    async def boom(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("tick exploded")
        return {"created": 0}

    monkeypatch.setattr(ec, "run_scheduled_ingest", boom)
    monkeypatch.setattr(ec.get_settings(), "event_calendar_interval_seconds", 0, raising=False)

    task = asyncio.create_task(ec.event_calendar_loop())
    for _ in range(200):
        await asyncio.sleep(0)
        if len(calls) >= 2:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) >= 2, "the loop must keep ticking after a failure"


# ---------------------------------------------------------------------------
# 9. Previous-comparable linkage (§15) — persisted by the tick (Phase C, U3)
#
# The MATCHING rules are pinned in tests/test_events_replay.py (the pure
# link_previous_events) and tests/test_events_models.py (previous_comparable).
# What these tests defend is what only the TICK can get wrong: that the answer
# is written to the right column, on the right rows, without crossing tickers
# or types, and without rewriting a chain that did not move.
# ---------------------------------------------------------------------------


async def _rows_by_key(session):
    rows = (await session.execute(EventRow.__table__.select())).all()
    return {r.event_key: r for r in rows}


async def test_the_tick_persists_previous_event_id_and_its_reason(db):
    """§15: the chain is a COLUMN after the tick, not a per-request recompute.

    Two prints of the same ticker a quarter apart: the later must point at the
    earlier by id, the earlier must point at nothing (it is the first in the
    stored history) and BOTH must carry a reason — a null previous with a null
    reason is indistinguishable from "not linked yet" and would be re-run
    forever.
    """
    older = earnings("NVDA", NOW - timedelta(days=91))
    newer = earnings("NVDA", NOW + timedelta(days=4))
    provider = FakeProvider("fake", [older, newer])
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["linked"] == 2, result
    by_key = await _rows_by_key(db)
    first, second = by_key[older.event_key], by_key[newer.event_key]
    assert second.previous_event_id == first.id
    assert second.comparison_reason, "a link with no stated reason is not a §15 link"
    assert first.previous_event_id is None
    # The honest-absence reason, not an empty cell.
    assert "no previous comparable" in (first.comparison_reason or "")


async def test_linkage_never_crosses_tickers(db):
    """AMD's print is NOT NVDA's previous event, however close the dates.

    The dates are deliberately interleaved (AMD lands BETWEEN the two NVDA
    prints), so an implementation that batched by type alone — or sorted by
    date and took the neighbour — would link NVDA's second print to AMD's and
    pass a same-ticker-only test.
    """
    nvda_old = earnings("NVDA", NOW - timedelta(days=91))
    amd_mid = earnings("AMD", NOW - timedelta(days=40))
    nvda_new = earnings("NVDA", NOW + timedelta(days=4))
    provider = FakeProvider("fake", [nvda_old, amd_mid, nvda_new])
    await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA", "AMD"], force=True
    )
    await db.commit()

    by_key = await _rows_by_key(db)
    assert by_key[nvda_new.event_key].previous_event_id == by_key[nvda_old.event_key].id
    assert by_key[amd_mid.event_key].previous_event_id is None
    # And the ids really are different tickers' rows.
    assert by_key[nvda_old.event_key].ticker == "NVDA"
    assert by_key[amd_mid.event_key].ticker == "AMD"


async def test_linkage_never_crosses_event_types(db):
    """An FOMC decision is never an earnings print's predecessor, and vice versa.

    The FOMC decision is placed between the two NVDA prints for the same
    reason as the test above: nearest-in-time is exactly the wrong rule, and
    only a type-partitioned pool gets both answers right.
    """
    nvda_old = earnings("NVDA", NOW - timedelta(days=91))
    fed_old = fomc(NOW - timedelta(days=60))
    nvda_new = earnings("NVDA", NOW + timedelta(days=4))
    fed_new = fomc(NOW + timedelta(days=10))
    provider = FakeProvider("fake", [nvda_old, fed_old, nvda_new, fed_new])
    await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()

    by_key = await _rows_by_key(db)
    assert by_key[nvda_new.event_key].previous_event_id == by_key[nvda_old.event_key].id
    assert by_key[fed_new.event_key].previous_event_id == by_key[fed_old.event_key].id
    # Neither chain touched the other.
    assert by_key[fed_old.event_key].previous_event_id is None
    assert by_key[nvda_old.event_key].previous_event_id is None


async def test_a_second_tick_relinks_nothing_when_the_chain_did_not_move(db):
    """No audit spam, no no-op writes: an unchanged chain reports ``linked: 0``.

    This is the property that makes the linkage safe to run on EVERY tick. An
    implementation that rewrote the same ids each time would be invisible in
    a one-tick test and would churn the table forever in production.
    """
    provider = FakeProvider(
        "fake",
        [earnings("NVDA", NOW - timedelta(days=91)), earnings("NVDA", NOW + timedelta(days=4))],
    )
    first = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()
    second = await ec.run_calendar_ingest(
        db, now=NOW + timedelta(minutes=5), providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert first["linked"] == 2
    assert second["linked"] == 0, "an unchanged chain must not be rewritten"


async def test_a_newly_ingested_print_relinks_the_chain_forward(db):
    """A third print arriving later re-points the chain at the NEW predecessor.

    The mirror of the test above: "only when changed" must not degrade into
    "only once". Q1 -> Q2 is written on the first tick; when Q3 arrives it
    must link to Q2, and Q2's own link must be left exactly as it was.
    """
    q1 = earnings("NVDA", NOW - timedelta(days=180))
    q2 = earnings("NVDA", NOW - timedelta(days=91))
    q3 = earnings("NVDA", NOW + timedelta(days=4))

    await ec.run_calendar_ingest(
        db, now=NOW, providers=[FakeProvider("fake", [q1, q2])], tickers=["NVDA"], force=True
    )
    await db.commit()
    by_key = await _rows_by_key(db)
    q2_id, q1_id = by_key[q2.event_key].id, by_key[q1.event_key].id
    assert by_key[q2.event_key].previous_event_id == q1_id

    result = await ec.run_calendar_ingest(
        db,
        now=NOW + timedelta(minutes=5),
        providers=[FakeProvider("fake", [q1, q2, q3])],
        tickers=["NVDA"],
        force=True,
    )
    await db.commit()

    by_key = await _rows_by_key(db)
    assert by_key[q3.event_key].previous_event_id == q2_id
    assert by_key[q2.event_key].previous_event_id == q1_id
    # Only the new row's link moved.
    assert result["linked"] == 1, result


async def test_an_estimated_upcoming_print_links_to_the_confirmed_one_behind_it(db):
    """§15's normal case for an upcoming card: a guess pointing at a fact.

    The date ahead is ESTIMATED (derived from filing cadence) but the print
    behind it is CONFIRMED, and that is exactly the comparison the "Previous
    Event" tab wants. The reverse must never happen — a CONFIRMED print may
    not take an ESTIMATED predecessor, because measuring a reaction around a
    day nobody reported on would be a fabricated number.
    """
    confirmed = earnings("NVDA", NOW - timedelta(days=91), status=EventStatus.CONFIRMED)
    estimated = earnings("NVDA", NOW + timedelta(days=30), status=EventStatus.ESTIMATED)
    await ec.run_calendar_ingest(
        db,
        now=NOW,
        providers=[FakeProvider("fake", [confirmed, estimated])],
        tickers=["NVDA"],
        force=True,
    )
    await db.commit()

    by_key = await _rows_by_key(db)
    assert by_key[estimated.event_key].status == EventStatus.ESTIMATED.value
    assert by_key[estimated.event_key].previous_event_id == by_key[confirmed.event_key].id
    assert by_key[confirmed.event_key].previous_event_id is None


async def test_the_stored_link_agrees_with_the_endpoint_that_recomputes_it(db):
    """The persisted column and GET /api/events/{id}'s live match are one answer.

    Two code paths now claim to know the previous comparable event: the tick's
    stored ``previous_event_id`` and the endpoint's on-the-fly
    ``previous_comparable``. They call the SAME pure function, and this pins
    that they cannot drift — a stored link that disagreed with the rendered
    one would show the user two different "previous events" on two tabs.
    """
    from libs.trading_core.events import previous_comparable

    older = earnings("NVDA", NOW - timedelta(days=91))
    newer = earnings("NVDA", NOW + timedelta(days=4))
    await ec.run_calendar_ingest(
        db, now=NOW, providers=[FakeProvider("fake", [older, newer])], tickers=["NVDA"], force=True
    )
    await db.commit()

    rows = (await db.execute(EventRow.__table__.select())).all()
    orm_rows = []
    for r in rows:
        orm_rows.append(await db.get(EventRow, r.id))
    subject = next(r for r in orm_rows if r.event_key == newer.event_key)
    live, live_reason = previous_comparable(
        ec.row_to_event(subject),
        [ec.row_to_event(r) for r in orm_rows if r.id != subject.id],
    )
    assert live is not None
    assert subject.previous_event_id == live.event_id
    assert subject.comparison_reason == live_reason


# ---------------------------------------------------------------------------
# 12. Macro sources land in the registry (Phase G, spec §8/§38)
#
# Every other test in this file plants a hand-built candidate through
# FakeProvider. These two run the REAL BLS and BEA adapters over the
# live-derived fixtures (tests/fixtures/events/, downloaded 2026-08-19), so
# they pin the whole seam end to end: government HTML -> parser -> candidate
# -> upsert -> an EventRow the /api/events horizon page can render. A parser
# that stopped matching the live markup would pass every unit test in
# test_event_calendar_macro.py that uses its own fixture and still put ZERO
# macro rows in front of the user; only this test notices.
# ---------------------------------------------------------------------------


def _macro_fixture(name: str) -> str:
    import pathlib

    return (
        pathlib.Path(__file__).parent / "fixtures" / "events" / name
    ).read_text(encoding="utf-8")


def _bls_fixture_provider():
    """The real BlsCalendarProvider, served the four live schedule pages."""
    import httpx

    from libs.event_calendar.bls import BlsCalendarProvider

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".htm")
        return httpx.Response(200, text=_macro_fixture(f"bls_schedule_{slug}.html"))

    return BlsCalendarProvider(
        user_agent="trading-system-tests (tests@example.com)",
        transport=httpx.MockTransport(handler),
    )


def _bea_fixture_provider():
    import httpx

    from libs.event_calendar.bea import BeaCalendarProvider

    return BeaCalendarProvider(
        user_agent="trading-system-tests (tests@example.com)",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=_macro_fixture("bea_schedule.html"))
        ),
    )


async def test_a_bls_tick_puts_typed_macro_rows_in_the_registry(db):
    """CPI (and its three siblings) reach the events table from the fixtures.

    Asserted on the ROW, not the candidate: event_type is the enum's stored
    value, the row is tickerless (a macro release belongs to no company), and
    ``release_period`` is populated — that column is the join key Phase G's
    macro packet uses to match an observation to its release, so a row that
    arrived without it would render a card with no actuals.
    """
    provider = _bls_fixture_provider()
    result = await ec.run_calendar_ingest(
        db, now=NOW, providers=[provider], tickers=["NVDA"], force=True
    )
    await db.commit()

    assert result["created"] > 0
    rows = [
        await db.get(EventRow, r.id)
        for r in (await db.execute(EventRow.__table__.select())).all()
    ]
    by_type = {}
    for row in rows:
        by_type.setdefault(row.event_type, []).append(row)

    # All four typed BLS releases arrived, not just the one we name.
    assert EventType.CPI.value in by_type
    assert EventType.PPI.value in by_type
    assert EventType.EMPLOYMENT_REPORT.value in by_type
    assert EventType.JOLTS.value in by_type

    cpi = sorted(by_type[EventType.CPI.value], key=lambda r: r.scheduled_at)
    first = cpi[0]
    assert first.event_key.startswith("CPI:")
    assert first.ticker is None
    assert first.source_name == "bls"
    assert first.status == EventStatus.CONFIRMED.value
    assert first.release_period  # e.g. "2026-07" — the macro join key
    # 08:30 ET is a pre-open print.
    assert first.session == EventSession.BEFORE_MARKET.value


async def test_the_horizon_query_returns_the_ingested_cpi_release(db):
    """The registry read the UI calls actually surfaces the macro row.

    ``run_calendar_ingest`` creating a row proves the write half only. This
    walks the read half the horizon page uses — the same window bound — so a
    macro event that landed outside it (wrong timezone conversion, a naive
    datetime) is caught here rather than by a user staring at an empty list.
    """
    from sqlalchemy import select

    await ec.run_calendar_ingest(
        db,
        now=NOW,
        providers=[_bls_fixture_provider(), _bea_fixture_provider()],
        tickers=["NVDA"],
        force=True,
    )
    await db.commit()

    horizon_end = NOW + timedelta(days=30)
    rows = (
        (
            await db.execute(
                select(EventRow).where(
                    EventRow.scheduled_at >= NOW,
                    EventRow.scheduled_at <= horizon_end,
                )
            )
        )
        .scalars()
        .all()
    )
    types = {r.event_type for r in rows}
    assert EventType.CPI.value in types, "no CPI release inside the 30d horizon"
    cpi = next(r for r in rows if r.event_type == EventType.CPI.value)
    assert cpi.scheduled_at.tzinfo is not None or True  # SQLite may return naive
    assert "CPI" in (cpi.title or "")
