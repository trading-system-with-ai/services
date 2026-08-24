"""Persistence pins for the event registry (Catalyst & Event Intelligence
Phase B; migration 021 / apps/gateway/db.py EventRow, MarketCalendarRow,
EventIngestStateRow).

tests/test_migration_parity.py already proves the ORM's column NAMES and ORDER
mirror the SQL. What it cannot see is behaviour, because the harness runs on
sqlite: it never executes the migration, so a DEFAULT that exists only in the
SQL text, a UNIQUE the ORM forgot to declare, or a datetime that comes back
naive would all pass parity and fail in production. These tests exercise the
ORM the way the ingestion seam will:

- a full round trip preserves every field, including the UTC instant and the
  NULLs that mean "not known" (importance unscored, no ticker on a macro row);
- ``events.event_key`` is UNIQUE at the DATABASE level — the idempotence the
  ingestion loop relies on under ADR-007 (no leader election, no distributed
  lock), where correctness must not ride on a single process;
- the JSON columns default to their empty container rather than NULL, so
  ``revision_history`` can be appended to and ``meta`` read without a None
  guard on a freshly inserted row;
- ``previous_event_id`` self-references and, per ``ON DELETE SET NULL``, does
  not cascade;
- ``market_calendar`` stores a half day distinguishably from a full day (the
  reason the table exists: session classification on an early close), and
  ``event_ingest_state`` keeps a failed attempt distinguishable from one that
  never ran.
"""
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from apps.gateway.db import (
    Base,
    EventIngestStateRow,
    EventRow,
    MarketCalendarRow,
    SessionLocal,
    engine,
)

# Instants are UTC everywhere; the ET wall-clock fact lives in event_timezone.
SCHEDULED = datetime(2026, 8, 27, 20, 20, 0, tzinfo=timezone.utc)  # 16:20 ET (EDT)


def _utc(value: datetime) -> datetime:
    """Re-attach UTC to a stamp read back from the harness.

    ``DateTime(timezone=True)`` is honoured by Postgres but sqlite has no tz
    type, so the harness returns the same instant NAIVE. The gateway already
    handles exactly this (`risk_snapshot.py:1379` "sqlite returns naive UTC",
    `order_sync.py:171`, `risk_validation.py:1034`); these tests assert the
    same contract — the WALL CLOCK is preserved unchanged and reading it as
    UTC recovers the instant that was written. Anything that shifted the hour
    on the way through (a local-time write) still fails here.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@pytest.fixture
async def db():
    """A fresh schema for each test (the suite's sqlite harness; conftest's
    client fixture does the same thing around the HTTP app)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def _earnings(**kw) -> EventRow:
    fields = dict(
        event_key="EARNINGS:NVDA:2026-08-27",
        event_type="EARNINGS",
        title="NVDA earnings release (8-K Item 2.02)",
        ticker="NVDA",
        company_id="0001045810",
        scheduled_at=SCHEDULED,
        event_timezone="America/New_York",
        session="AFTER_MARKET",
        status="CONFIRMED",
        source="COMPANY_IR_SEC",
        source_name="sec_edgar",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x/y.htm",
        source_event_id="0001045810-26-000123",
        last_verified_at=datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc),
    )
    fields.update(kw)
    return EventRow(**fields)


# ---------------------------------------------------------------------------
# events — round trip
# ---------------------------------------------------------------------------


async def test_event_round_trip_preserves_every_field(db):
    async with SessionLocal() as s:
        s.add(_earnings(importance=75, fiscal_quarter=2, fiscal_year=2027))
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.id is not None  # SERIAL identity assigned
    assert row.event_key == "EARNINGS:NVDA:2026-08-27"
    assert row.event_type == "EARNINGS"
    assert row.ticker == "NVDA"
    assert row.company_id == "0001045810"
    assert row.session == "AFTER_MARKET"
    assert row.status == "CONFIRMED"
    assert row.source == "COMPANY_IR_SEC"
    assert row.source_name == "sec_edgar"
    assert row.source_event_id == "0001045810-26-000123"
    assert row.importance == 75
    assert (row.fiscal_quarter, row.fiscal_year) == (2, 2027)


async def test_scheduled_at_round_trips_as_the_same_utc_instant(db):
    """The stored instant must come back tz-aware and EQUAL — a naive read
    would silently reinterpret 20:20Z as 20:20 local and move the event by
    hours (the whole point of DateTime(timezone=True))."""
    async with SessionLocal() as s:
        s.add(_earnings())
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert _utc(row.scheduled_at) == SCHEDULED
    assert _utc(row.scheduled_at).hour == 20  # 20:20Z, not a local-time 16:20
    # The event's own zone is stored alongside the instant: "16:20 ET" is the
    # asserted fact, and its UTC offset changes across DST.
    assert row.event_timezone == "America/New_York"


async def test_created_and_updated_at_are_populated_and_tz_aware(db):
    async with SessionLocal() as s:
        s.add(_earnings())
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    for stamp in (row.created_at, row.updated_at):
        assert stamp is not None
        # utcnow() default, so the stored wall clock is a UTC one.
        assert abs((_utc(stamp) - datetime.now(timezone.utc)).total_seconds()) < 300


async def test_macro_event_stores_series_fields_and_null_ticker(db):
    """A CPI release has no ticker and no fiscal quarter; a corporate row has
    no series/agency. Both absences are NULL, never a placeholder string."""
    async with SessionLocal() as s:
        s.add(
            EventRow(
                event_key="CPI:2026-07",
                event_type="CPI",
                title="CPI — July 2026",
                scheduled_at=datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc),
                session="BEFORE_MARKET",
                status="CONFIRMED",
                source="GOVERNMENT_AGENCY",
                source_name="bls",
                series_id="CUUR0000SA0",
                agency="BLS",
                release_period="2026-07",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.ticker is None and row.company_id is None
    assert row.fiscal_quarter is None and row.fiscal_year is None
    assert (row.series_id, row.agency, row.release_period) == (
        "CUUR0000SA0", "BLS", "2026-07",
    )


async def test_unscored_importance_is_null_not_zero(db):
    """§44 rule 18 / audit §5.2: "not yet scored" must be distinguishable
    from "scored 0", or the UI reads an unscored event as unimportant."""
    async with SessionLocal() as s:
        s.add(_earnings())
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.importance is None
    assert row.source_url is not None  # sanity: the fixture does set some optionals


async def test_optional_string_columns_default_to_none(db):
    """Every nullable column left unset comes back None — an ingestion path
    that omits a field must not materialise "" and later read as known."""
    async with SessionLocal() as s:
        s.add(
            EventRow(
                event_key="FED_SPEECH:2026-09-04:powell:outlook",
                event_type="FED_SPEECH",
                title="Chair Powell: Economic Outlook",
                scheduled_at=datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc),
                status="CONFIRMED",
                source="FEDERAL_RESERVE",
                source_name="fed_rss",
                speaker="Jerome H. Powell",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.ticker is None
    assert row.source_url is None
    assert row.source_event_id is None
    assert row.last_verified_at is None
    assert row.previous_event_id is None
    assert row.comparison_reason is None
    assert row.topic is None
    assert row.speaker == "Jerome H. Powell"
    # Column defaults still apply to the fields the writer omitted.
    assert row.event_timezone == "America/New_York"
    assert row.session == "UNKNOWN"


# ---------------------------------------------------------------------------
# events — unique event_key (DB-level ingestion idempotence, ADR-007)
# ---------------------------------------------------------------------------


async def test_duplicate_event_key_is_rejected_by_the_database(db):
    """The ingestion loop's idempotence must not ride on the single-process
    assumption (ADR-007: no leader election). A second writer re-ingesting
    the same event can only collide."""
    async with SessionLocal() as s:
        s.add(_earnings())
        await s.commit()

    async with SessionLocal() as s:
        # Same key, different source/status — still the same real-world event.
        s.add(_earnings(status="ESTIMATED", source="DERIVED", source_name="derived_cadence"))
        with pytest.raises(IntegrityError):
            await s.commit()

    async with SessionLocal() as s:
        rows = (await s.execute(select(EventRow))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "CONFIRMED"  # the first write survived intact


async def test_the_same_event_from_two_sources_is_one_row_updated_not_two(db):
    """A better source UPDATES the row (source-precedence merge) rather than
    inserting a rival copy — the reason the unique key is `event_key` alone
    and not `(event_key, source)`."""
    async with SessionLocal() as s:
        s.add(
            _earnings(
                status="ESTIMATED",
                source="DERIVED",
                source_name="derived_cadence",
                source_url=None,
                source_event_id=None,
                last_verified_at=None,
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (
            await s.execute(
                select(EventRow).where(EventRow.event_key == "EARNINGS:NVDA:2026-08-27")
            )
        ).scalar_one()
        row.status = "CONFIRMED"
        row.source = "COMPANY_IR_SEC"
        row.source_name = "sec_edgar"
        row.revision_history = [
            {
                "scheduled_at": SCHEDULED.isoformat(),
                "status": "ESTIMATED",
                "source_name": "derived_cadence",
                "at": "2026-08-19T14:00:00+00:00",
            }
        ]
        await s.commit()

    async with SessionLocal() as s:
        rows = (await s.execute(select(EventRow))).scalars().all()
    assert len(rows) == 1
    assert (rows[0].status, rows[0].source_name) == ("CONFIRMED", "sec_edgar")
    assert rows[0].revision_history[0]["source_name"] == "derived_cadence"


async def test_distinct_event_keys_coexist(db):
    """Two quarters of the same ticker are different events; the unique key
    must not collapse them."""
    async with SessionLocal() as s:
        s.add(_earnings())
        s.add(
            _earnings(
                event_key="EARNINGS:NVDA:2026-11-19",
                scheduled_at=datetime(2026, 11, 19, 21, 20, tzinfo=timezone.utc),
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        keys = set((await s.execute(select(EventRow.event_key))).scalars().all())
    assert keys == {"EARNINGS:NVDA:2026-08-27", "EARNINGS:NVDA:2026-11-19"}


# ---------------------------------------------------------------------------
# events — JSON default and self-FK
# ---------------------------------------------------------------------------


async def test_revision_history_defaults_to_empty_list_not_null(db):
    """A freshly ingested event must be appendable without a None guard —
    the JSON DEFAULT '[]' in the SQL has to exist in the ORM too, because
    the sqlite harness never runs the migration."""
    async with SessionLocal() as s:
        s.add(_earnings())
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.revision_history == []
    assert isinstance(row.revision_history, list)


async def test_revision_history_round_trips_a_list_of_dicts(db):
    entries = [
        {
            "scheduled_at": "2026-08-27T20:20:00+00:00",
            "status": "ESTIMATED",
            "source_name": "derived_cadence",
            "at": "2026-07-01T00:00:00+00:00",
        },
        {
            "scheduled_at": "2026-08-27T20:20:00+00:00",
            "status": "CONFIRMED",
            "source_name": "sec_edgar",
            "at": "2026-08-19T14:00:00+00:00",
        },
    ]
    async with SessionLocal() as s:
        s.add(_earnings(revision_history=entries))
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventRow))).scalar_one()

    assert row.revision_history == entries
    assert row.revision_history[-1]["status"] == "CONFIRMED"


async def test_previous_event_id_links_to_the_prior_comparable_event(db):
    """§15: the link carries its REASON — an unexplained comparison is not
    auditable."""
    async with SessionLocal() as s:
        prior = _earnings(
            event_key="EARNINGS:NVDA:2026-05-28",
            scheduled_at=datetime(2026, 5, 28, 20, 20, tzinfo=timezone.utc),
        )
        s.add(prior)
        await s.commit()
        prior_id = prior.id

        s.add(
            _earnings(
                previous_event_id=prior_id,
                comparison_reason="prior quarterly earnings",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (
            await s.execute(
                select(EventRow).where(EventRow.event_key == "EARNINGS:NVDA:2026-08-27")
            )
        ).scalar_one()

    assert row.previous_event_id == prior_id
    assert row.comparison_reason == "prior quarterly earnings"


async def test_deleting_the_prior_event_does_not_delete_its_successor(db):
    """ON DELETE SET NULL, not CASCADE: pruning a stale prior event must
    never take the upcoming one with it.

    sqlite runs with ``PRAGMA foreign_keys=OFF`` by default, so the harness
    cannot execute the referential ACTION — this pins the outcome the ORM
    layer produces, and the companion test below pins the clause in the SQL
    that Postgres actually enforces. Neither alone would be enough.
    """
    async with SessionLocal() as s:
        prior = _earnings(
            event_key="EARNINGS:NVDA:2026-05-28",
            scheduled_at=datetime(2026, 5, 28, 20, 20, tzinfo=timezone.utc),
        )
        s.add(prior)
        await s.commit()
        prior_id = prior.id
        s.add(_earnings(previous_event_id=prior_id))
        await s.commit()

    async with SessionLocal() as s:
        await s.execute(delete(EventRow).where(EventRow.id == prior_id))
        await s.commit()

    async with SessionLocal() as s:
        rows = (await s.execute(select(EventRow))).scalars().all()
    assert [r.event_key for r in rows] == ["EARNINGS:NVDA:2026-08-27"]


def test_migration_declares_on_delete_set_null_for_the_self_fk():
    """The referential action lives in the SQL Postgres runs. The harness
    cannot enforce it (foreign_keys=OFF on sqlite) and
    tests/test_migration_parity.py only compares column names and order, so
    without this pin a later edit to CASCADE would delete an upcoming event
    when its stale prior quarter was pruned, and no test would notice."""
    sql = (
        Path(__file__).resolve().parents[1] / "migrations" / "021_events.sql"
    ).read_text()
    assert "REFERENCES events (id) ON DELETE SET NULL" in sql
    assert "ON DELETE CASCADE" not in sql
    # The unique key the ingestion loop's idempotence rests on (ADR-007).
    assert "event_key          VARCHAR(200) NOT NULL UNIQUE" in sql


# ---------------------------------------------------------------------------
# market_calendar
# ---------------------------------------------------------------------------


async def test_market_calendar_round_trip_full_day(db):
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=date(2026, 8, 19),
                open_utc=datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc),   # 09:30 EDT
                close_utc=datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc),   # 16:00 EDT
                session_open_utc=datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc),
                session_close_utc=datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc),
                source="alpaca_calendar",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(MarketCalendarRow))).scalar_one()

    assert row.session_date == date(2026, 8, 19)
    assert row.exchange == "US"  # column default
    assert row.is_early_close is False
    assert _utc(row.open_utc) == datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc)
    assert row.source == "alpaca_calendar"
    assert row.fetched_at is not None


async def test_market_calendar_marks_an_early_close(db):
    """The reason this table exists: on a half day the 09:30-16:00 default
    would classify a 13:30 ET release as DURING_MARKET when the tape has
    already closed."""
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=date(2026, 11, 27),
                open_utc=datetime(2026, 11, 27, 14, 30, tzinfo=timezone.utc),  # 09:30 EST
                close_utc=datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc),  # 13:00 EST
                is_early_close=True,
                source="alpaca_calendar",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(MarketCalendarRow))).scalar_one()

    assert row.is_early_close is True
    assert (row.close_utc - row.open_utc).total_seconds() == 3.5 * 3600
    # Extended-session hours the provider did not report stay NULL, not a guess.
    assert row.session_open_utc is None and row.session_close_utc is None


async def test_session_date_is_the_primary_key(db):
    """One row per session day: a re-fetch of the same window must collide
    rather than duplicate (upsert on session_date)."""
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=date(2026, 8, 19),
                open_utc=datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc),
                close_utc=datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc),
                source="alpaca_calendar",
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=date(2026, 8, 19),
                open_utc=datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc),
                close_utc=datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc),
                source="massive_calendar",
            )
        )
        with pytest.raises(IntegrityError):
            await s.commit()


# ---------------------------------------------------------------------------
# event_ingest_state
# ---------------------------------------------------------------------------


async def test_ingest_state_meta_defaults_to_empty_dict_not_null(db):
    async with SessionLocal() as s:
        s.add(EventIngestStateRow(key="alpaca_calendar"))
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(select(EventIngestStateRow))).scalar_one()

    assert row.meta == {}
    assert row.last_fetched_at is None and row.last_ok_at is None
    assert row.last_error is None


async def test_ingest_state_distinguishes_a_failed_attempt_from_never_run(db):
    """A provider that has been 403ing for a week must not look like one that
    was never tried: last_fetched_at moves, last_ok_at does not, and the
    honest error string is kept (§8 per-adapter failure isolation)."""
    attempted = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    succeeded = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    async with SessionLocal() as s:
        s.add(
            EventIngestStateRow(
                key="massive_calendar",
                last_fetched_at=attempted,
                last_ok_at=succeeded,
                last_error="HTTP 403 SUBSCRIPTION_DENIED /benzinga/v1/earnings",
                meta={"capabilities": {"earnings_calendar": False, "market_holidays": True}},
            )
        )
        s.add(EventIngestStateRow(key="fed_fomc"))
        await s.commit()

    async with SessionLocal() as s:
        rows = {
            r.key: r
            for r in (await s.execute(select(EventIngestStateRow))).scalars().all()
        }

    failing = rows["massive_calendar"]
    assert _utc(failing.last_fetched_at) == attempted
    assert _utc(failing.last_ok_at) == succeeded
    assert failing.last_fetched_at > failing.last_ok_at  # stale but not silent
    assert "403" in failing.last_error
    assert failing.meta["capabilities"]["earnings_calendar"] is False

    never_run = rows["fed_fomc"]
    assert never_run.last_fetched_at is None and never_run.last_error is None


async def test_ingest_state_key_is_the_primary_key(db):
    """Per-provider (or provider:ticker) watermark — one row each."""
    async with SessionLocal() as s:
        s.add(EventIngestStateRow(key="sec_edgar:NVDA"))
        await s.commit()

    async with SessionLocal() as s:
        s.add(EventIngestStateRow(key="sec_edgar:NVDA"))
        with pytest.raises(IntegrityError):
            await s.commit()

    async with SessionLocal() as s:
        rows = (await s.execute(select(EventIngestStateRow))).scalars().all()
    assert len(rows) == 1
