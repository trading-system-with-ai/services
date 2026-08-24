"""Event registry API — GET/POST /api/events (event spec §6, §11, §12, §13,
§15; audit §5.1, §11.1 Phase B).

The tests are grouped by the guarantee they defend, not by endpoint:

1. **Honest absence.** An unconfigured install with an empty table gets 200
   and an empty list plus a capability block that explains why — never a 500
   and never an invented date.
2. **Horizon math across the NY midnight.** The window is bucketed on the
   America/New_York calendar day, so an AMC release at 20:15 ET (00:15 UTC
   the NEXT day) must land in *today's* bucket. Both a DST date (August) and
   a non-DST date (January) are pinned, because a UTC-offset implementation
   is wrong by a different number of hours in each.
3. **§12 relevance ordering.** POSITION > TRADING_POOL > WATCHLIST >
   MARKET_WIDE, applied as the SORT KEY rather than left to the client.
4. **§7/§11 estimate handling.** ``is_estimated`` on every row;
   ``include_estimated=false`` hides them.
5. **User confirmation (§7, §78).** The confirm endpoint flips
   ESTIMATED -> CONFIRMED through the same merge rules a provider goes
   through, with the USER source, and audits it.
6. **Refresh idempotence.** A second refresh against the same provider output
   creates zero rows.

Uses the shared ``client`` fixture (conftest.py): providers "stub",
execution "simulated". The event registry's own provider list is injected per
test rather than taken from settings, so no test depends on network state.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from apps.gateway import event_calendar as ec
from apps.gateway.db import (
    AuditEvent,
    EventRow,
    MarketCalendarRow,
    Position,
    SessionLocal,
    TradingPoolItem,
    WatchlistItem,
)
from apps.gateway.routers.events import resolve_horizon
from libs.trading_core.models import AuditAction
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

EASTERN = ZoneInfo("America/New_York")


def _et(y, m, d, hour, minute=0) -> datetime:
    return datetime(y, m, d, hour, minute, tzinfo=EASTERN).astimezone(timezone.utc)


async def _add_event(
    *,
    key,
    event_type=EventType.EARNINGS,
    title="Test event",
    ticker=None,
    when=None,
    status=EventStatus.CONFIRMED,
    source=EventSourceKind.COMPANY_IR_SEC,
    source_name="sec_edgar",
    session=EventSession.AFTER_MARKET,
    speaker=None,
    release_period=None,
) -> int:
    """Insert one event row directly and return its id.

    Direct inserts rather than an ingestion tick: these tests are about what
    the API does with stored rows, and going through a provider would couple
    every assertion to the ingest's own window arithmetic.
    """
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title=title,
            ticker=ticker,
            scheduled_at=when or (datetime.now(timezone.utc) + timedelta(days=2)),
            event_timezone="America/New_York",
            session=session.value,
            status=status.value,
            source=source.value,
            source_name=source_name,
            speaker=speaker,
            release_period=release_period,
            revision_history=[],
        )
        s.add(row)
        await s.commit()
        return row.id


def _in_days(days: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)


# ---------------------------------------------------------------------------
# 1. Honest absence
# ---------------------------------------------------------------------------


async def test_empty_registry_answers_200_with_an_explanation_not_500(client):
    r = await client.get("/api/events")
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == []
    assert body["counts"]["total"] == 0
    assert body["display_timezone"] == "America/New_York"
    # The capability/freshness blocks are what tell the user WHY it is empty.
    assert "capabilities" in body
    assert "freshness" in body
    assert "configured_providers" in body["freshness"]


async def test_unconfigured_install_still_answers_the_feed(unconfigured_client):
    """Deliberately not a 503: a 503 would hide the capability report, which
    is the only thing that explains an empty calendar."""
    r = await unconfigured_client.get("/api/events")
    assert r.status_code == 200
    assert r.json()["events"] == []


async def test_a_provider_that_has_never_run_is_reported_as_never_run(client):
    """An absent provider and an untried one are different facts."""
    r = await client.get("/api/events")
    per_provider = r.json()["freshness"]["per_provider"]
    for name, block in per_provider.items():
        assert block["configured"] is True
        assert block["note"] == "NEVER_RUN"
        assert block["last_ok_at"] is None


# ---------------------------------------------------------------------------
# 2. Horizon math (§11)
# ---------------------------------------------------------------------------


async def test_unknown_horizon_is_422(client):
    r = await client.get("/api/events?horizon=next-quarter")
    assert r.status_code == 422
    assert "unknown horizon" in r.json()["detail"]


async def test_custom_horizon_without_a_range_is_422(client):
    r = await client.get("/api/events?horizon=custom")
    assert r.status_code == 422


async def test_custom_horizon_with_an_inverted_range_is_422(client):
    r = await client.get(
        "/api/events?horizon=custom&start=2026-09-10&end=2026-09-01"
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    "day,offset_hours",
    [
        # August: EDT, UTC-4. A 20:15 ET release is 00:15 UTC the NEXT day.
        (date(2026, 8, 19), 4),
        # January: EST, UTC-5. Same wall clock, a different UTC instant —
        # which is exactly why the bucket must be computed in ET.
        (date(2026, 1, 21), 5),
    ],
)
def test_today_bucket_covers_the_whole_new_york_day_across_dst(day, offset_hours):
    """An AMC release belongs to the NY day it happened on, not the UTC one.

    ``resolve_horizon`` is exercised directly with a pinned ``now`` because
    the endpoint's ``now`` is the wall clock and cannot be frozen without a
    time-machine dependency the project does not carry.
    """
    now = datetime(day.year, day.month, day.day, 15, 0, tzinfo=EASTERN).astimezone(
        timezone.utc
    )
    start_utc, end_utc, label = resolve_horizon("today", None, None, now)
    assert label == "Today"
    # The window opens at 00:00 ET and closes at 23:59:59.999999 ET.
    assert start_utc == datetime(
        day.year, day.month, day.day, 0, 0, tzinfo=EASTERN
    ).astimezone(timezone.utc)
    assert start_utc.hour == offset_hours  # proves the offset really differs

    amc = datetime(day.year, day.month, day.day, 20, 15, tzinfo=EASTERN).astimezone(
        timezone.utc
    )
    assert amc.astimezone(timezone.utc).date() != day, (
        "the fixture must actually straddle UTC midnight or it proves nothing"
    )
    assert start_utc <= amc <= end_utc


def test_the_7d_bucket_ends_at_the_close_of_the_seventh_new_york_day():
    now = datetime(2026, 8, 19, 15, 0, tzinfo=EASTERN).astimezone(timezone.utc)
    start_utc, end_utc, label = resolve_horizon("7d", None, None, now)
    assert label == "Next 7 days"
    last_moment = datetime(2026, 8, 26, 23, 59, tzinfo=EASTERN).astimezone(timezone.utc)
    assert start_utc <= last_moment <= end_utc
    just_after = datetime(2026, 8, 27, 0, 30, tzinfo=EASTERN).astimezone(timezone.utc)
    assert just_after > end_utc


def test_a_bare_custom_date_is_read_in_new_york_not_utc():
    """Reading "2026-09-16" as UTC would clip that day's AMC releases off."""
    now = datetime(2026, 8, 19, 15, 0, tzinfo=EASTERN).astimezone(timezone.utc)
    start_utc, end_utc, _ = resolve_horizon("custom", "2026-09-16", "2026-09-16", now)
    amc = datetime(2026, 9, 16, 20, 15, tzinfo=EASTERN).astimezone(timezone.utc)
    assert start_utc <= amc <= end_utc


async def test_events_outside_the_horizon_are_excluded(client):
    await _add_event(key="EARNINGS:NVDA:in3", ticker="NVDA", when=_in_days(3))
    await _add_event(key="EARNINGS:AAPL:in20", ticker="AAPL", when=_in_days(20))

    seven = (await client.get("/api/events?horizon=7d")).json()
    assert [e["ticker"] for e in seven["events"]] == ["NVDA"]

    thirty = (await client.get("/api/events?horizon=30d")).json()
    assert {e["ticker"] for e in thirty["events"]} == {"NVDA", "AAPL"}


# ---------------------------------------------------------------------------
# 3. Relevance ordering (§12)
# ---------------------------------------------------------------------------


async def test_relevance_ordering_puts_position_above_pool_above_watchlist(client):
    async with SessionLocal() as s:
        s.add(WatchlistItem(ticker="WATCH", added_by="test"))
        s.add(TradingPoolItem(ticker="POOL", promoted_by="test"))
        s.add(
            Position(
                ticker="HELD",
                quantity=10,
                avg_price=100.0,
                max_loss=200.0,
                stop_distance=20.0,
            )
        )
        await s.commit()

    # Planted in DELIBERATELY REVERSE chronological order so a naive
    # sort-by-date would produce the opposite of the expected answer — the
    # test would pass by accident otherwise.
    await _add_event(key="E:WATCH", ticker="WATCH", when=_in_days(1))
    await _add_event(key="E:POOL", ticker="POOL", when=_in_days(2))
    await _add_event(key="E:HELD", ticker="HELD", when=_in_days(3))
    await _add_event(
        key="FOMC_DECISION:x",
        event_type=EventType.FOMC_DECISION,
        title="FOMC decision",
        when=_in_days(4),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
        session=EventSession.DURING_MARKET,
    )

    body = (await client.get("/api/events?horizon=7d")).json()
    tiers = [e["relevance_tier"] for e in body["events"]]
    assert tiers == ["POSITION", "TRADING_POOL", "WATCHLIST", "MARKET_WIDE"]
    assert [e["ticker"] for e in body["events"]][:3] == ["HELD", "POOL", "WATCH"]
    assert body["counts"]["by_relevance"]["POSITION"] == 1


async def test_two_events_in_the_same_tier_sort_by_date(client):
    async with SessionLocal() as s:
        s.add(WatchlistItem(ticker="AAA", added_by="test"))
        s.add(WatchlistItem(ticker="BBB", added_by="test"))
        await s.commit()
    await _add_event(key="E:BBB", ticker="BBB", when=_in_days(5))
    await _add_event(key="E:AAA", ticker="AAA", when=_in_days(2))
    body = (await client.get("/api/events?horizon=7d")).json()
    assert [e["ticker"] for e in body["events"]] == ["AAA", "BBB"]


async def test_relevance_filter_narrows_the_feed(client):
    async with SessionLocal() as s:
        s.add(WatchlistItem(ticker="WATCH", added_by="test"))
        await s.commit()
    await _add_event(key="E:WATCH", ticker="WATCH", when=_in_days(2))
    await _add_event(
        key="CPI:2026-08",
        event_type=EventType.CPI,
        title="CPI",
        when=_in_days(3),
        source=EventSourceKind.GOVERNMENT_AGENCY,
        source_name="bls",
        session=EventSession.BEFORE_MARKET,
        release_period="2026-08",
    )
    body = (await client.get("/api/events?horizon=7d&relevance=MARKET_WIDE")).json()
    assert [e["event_type"] for e in body["events"]] == ["CPI"]


async def test_exposure_is_attached_for_held_tickers_only(client):
    async with SessionLocal() as s:
        s.add(
            Position(
                ticker="HELD",
                quantity=7,
                avg_price=50.0,
                max_loss=100.0,
                stop_distance=10.0,
            )
        )
        s.add(WatchlistItem(ticker="WATCH", added_by="test"))
        await s.commit()
    await _add_event(key="E:HELD", ticker="HELD", when=_in_days(1))
    await _add_event(key="E:WATCH", ticker="WATCH", when=_in_days(2))

    body = (await client.get("/api/events?horizon=7d")).json()
    by_ticker = {e["ticker"]: e for e in body["events"]}
    assert by_ticker["HELD"]["exposure"]["position_qty"] == 7
    assert by_ticker["HELD"]["exposure"]["position_market_value"] == pytest.approx(350.0)
    # Labelled honestly: this is a cost basis, not a live mark — the router
    # reads stored rows only and never touches a market-data provider.
    assert by_ticker["HELD"]["exposure"]["basis"] == "COST"
    assert by_ticker["WATCH"]["exposure"] is None


# ---------------------------------------------------------------------------
# 4. Estimates, types, tickers (§7, §11)
# ---------------------------------------------------------------------------


async def test_include_estimated_false_hides_estimated_events(client):
    await _add_event(key="E:CONF", ticker="CONF", when=_in_days(2))
    await _add_event(
        key="E:EST",
        ticker="EST",
        when=_in_days(3),
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
    )

    with_est = (await client.get("/api/events?horizon=7d")).json()
    assert with_est["counts"]["estimated"] == 1
    assert {e["ticker"] for e in with_est["events"]} == {"CONF", "EST"}

    without = (await client.get("/api/events?horizon=7d&include_estimated=false")).json()
    assert [e["ticker"] for e in without["events"]] == ["CONF"]


async def test_every_row_carries_is_estimated_matching_its_status(client):
    """A consumer that forgets to read the enum still cannot render a derived
    date as a fact."""
    await _add_event(key="E:CONF", ticker="CONF", when=_in_days(2))
    await _add_event(
        key="E:EST",
        ticker="EST",
        when=_in_days(3),
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
    )
    body = (await client.get("/api/events?horizon=7d")).json()
    for event in body["events"]:
        assert event["is_estimated"] == (event["status"] == "ESTIMATED")


async def test_canceled_events_are_hidden_by_default_and_shown_on_request(client):
    await _add_event(
        key="E:GONE",
        ticker="GONE",
        when=_in_days(2),
        status=EventStatus.CANCELED,
    )
    assert (await client.get("/api/events?horizon=7d")).json()["events"] == []
    shown = (await client.get("/api/events?horizon=7d&include_canceled=true")).json()
    assert [e["ticker"] for e in shown["events"]] == ["GONE"]


async def test_type_and_ticker_filters(client):
    await _add_event(key="E:NVDA", ticker="NVDA", when=_in_days(2))
    await _add_event(
        key="CPI:2026-08",
        event_type=EventType.CPI,
        title="CPI",
        when=_in_days(3),
        source=EventSourceKind.GOVERNMENT_AGENCY,
        source_name="bls",
        release_period="2026-08",
    )
    by_type = (await client.get("/api/events?horizon=7d&types=CPI")).json()
    assert [e["event_type"] for e in by_type["events"]] == ["CPI"]
    by_ticker = (await client.get("/api/events?horizon=7d&tickers=NVDA")).json()
    assert [e["ticker"] for e in by_ticker["events"]] == ["NVDA"]


async def test_an_unknown_event_type_filter_is_422_not_a_silent_empty_list(client):
    """A typo'd filter must be a visible client error; returning [] would look
    identical to "no such events" and hide the bug."""
    r = await client.get("/api/events?horizon=7d&types=EARNINGZ")
    assert r.status_code == 422
    assert "EARNINGZ" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Payload shape (§6, §10, §13)
# ---------------------------------------------------------------------------


async def test_event_out_carries_both_timestamps_and_the_importance_arithmetic(client):
    when = _et(2026, 12, 15, 16, 5)  # December => EST, UTC-5
    async with SessionLocal() as s:
        s.add(
            Position(
                ticker="NVDA",
                quantity=1,
                avg_price=1.0,
                max_loss=1.0,
                stop_distance=1.0,
            )
        )
        await s.commit()
    await _add_event(key="EARNINGS:NVDA:2026-12-15", ticker="NVDA", when=when)

    body = (
        await client.get(
            "/api/events?horizon=custom&start=2026-12-01&end=2026-12-31"
        )
    ).json()
    event = body["events"][0]
    assert event["scheduled_at_utc"].endswith("+00:00")
    assert event["scheduled_at_local"].startswith("2026-12-15T16:05")
    assert "-05:00" in event["scheduled_at_local"]  # EST offset, not EDT
    assert event["event_timezone"] == "America/New_York"
    # §13: the score is never a number without its arithmetic.
    assert event["importance_components"] == {"event_type": 60, "relevance": 30}
    assert sum(event["importance_components"].values()) == event["importance_raw_total"]
    assert event["importance"] == 90
    assert event["importance_was_clamped"] is False
    assert event["importance_model_version"]
    assert event["lifecycle"] in (
        "SCHEDULED",
        "PRE_EVENT",
        "LIVE",
        "POST_EVENT",
        "ARCHIVED",
    )


async def test_a_clamped_score_reports_its_unclamped_total_honestly(client):
    """"90 + 30 = 120 -> 100" must be visible, not presented as if the
    components summed to 100."""
    async with SessionLocal() as s:
        s.add(
            Position(
                ticker="SPY", quantity=1, avg_price=1.0, max_loss=1.0, stop_distance=1.0
            )
        )
        await s.commit()
    await _add_event(
        key="FOMC_DECISION:soon",
        event_type=EventType.FOMC_DECISION,
        title="FOMC decision",
        ticker="SPY",
        when=_in_days(2),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
    )
    event = (await client.get("/api/events?horizon=7d")).json()["events"][0]
    assert event["importance_raw_total"] == 120
    assert event["importance"] == 100
    assert event["importance_was_clamped"] is True


# ---------------------------------------------------------------------------
# 6. GET /{event_id} + previous comparable (§15)
# ---------------------------------------------------------------------------


async def test_event_detail_returns_the_previous_comparable_and_its_reason(client):
    await _add_event(
        key="EARNINGS:NVDA:past", ticker="NVDA", when=_in_days(-91)
    )
    current_id = await _add_event(
        key="EARNINGS:NVDA:next", ticker="NVDA", when=_in_days(3)
    )
    r = await client.get(f"/api/events/{current_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["event_id"] == current_id
    assert body["previous_event"]["event_key"] == "EARNINGS:NVDA:past"
    assert body["comparison_reason"] == "prior quarterly earnings"


async def test_a_first_ever_event_has_an_honest_null_previous(client):
    event_id = await _add_event(key="EARNINGS:NEW:1", ticker="NEW", when=_in_days(3))
    body = (await client.get(f"/api/events/{event_id}")).json()
    assert body["previous_event"] is None
    assert body["comparison_reason"] is None


async def test_previous_comparable_never_crosses_tickers(client):
    await _add_event(key="EARNINGS:AAPL:past", ticker="AAPL", when=_in_days(-30))
    event_id = await _add_event(key="EARNINGS:NVDA:next", ticker="NVDA", when=_in_days(3))
    body = (await client.get(f"/api/events/{event_id}")).json()
    assert body["previous_event"] is None


async def test_unknown_event_id_is_404(client):
    r = await client.get("/api/events/99999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Confirm / cancel (§7, §78)
# ---------------------------------------------------------------------------


async def test_confirm_flips_estimated_to_confirmed_with_the_user_source(client):
    event_id = await _add_event(
        key="EARNINGS:NVDA:2026-11-18",
        ticker="NVDA",
        when=_et(2026, 11, 18, 16, 5),
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
    )
    r = await client.post(
        f"/api/events/{event_id}/confirm",
        json={
            "scheduled_at": "2026-11-19T16:05:00",
            "session": "AFTER_MARKET",
            "source_url": "https://investor.nvidia.com/events",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "CONFIRMED"
    assert body["is_estimated"] is False
    assert body["source"] == EventSourceKind.USER.value
    assert body["source_name"] == "user"
    assert body["source_url"] == "https://investor.nvidia.com/events"
    # A naive instant from this UI means EASTERN, never UTC (§10).
    assert body["scheduled_at_local"].startswith("2026-11-19T16:05")
    # The merge re-keys on the accepted date move, and the row was persisted
    # with the new key — otherwise the next tick recreates the card.
    assert body["event_key"] == "EARNINGS:NVDA:2026-11-19"
    # And the previous value is preserved for the §84 audit trail.
    assert len(body["revision_history"]) == 1

    async with SessionLocal() as s:
        audits = [
            a
            for a in (await s.execute(AuditEvent.__table__.select())).all()
            if a.action == AuditAction.EVENT_UPDATED.value
        ]
    assert len(audits) == 1
    assert audits[0].actor_type == "USER"
    assert audits[0].details["user_confirmed"] is True


async def test_confirm_with_a_bad_instant_is_422(client):
    event_id = await _add_event(key="E:X", ticker="X", when=_in_days(3))
    r = await client.post(
        f"/api/events/{event_id}/confirm", json={"scheduled_at": "next tuesday"}
    )
    assert r.status_code == 422


async def test_confirm_with_an_unknown_session_is_422(client):
    event_id = await _add_event(key="E:X", ticker="X", when=_in_days(3))
    r = await client.post(
        f"/api/events/{event_id}/confirm",
        json={"scheduled_at": "2026-11-19T16:05:00", "session": "LUNCHTIME"},
    )
    assert r.status_code == 422


async def test_confirm_on_an_unknown_event_is_404(client):
    r = await client.post(
        "/api/events/424242/confirm", json={"scheduled_at": "2026-11-19T16:05:00"}
    )
    assert r.status_code == 404


async def test_cancel_marks_the_event_canceled_and_audits_it(client):
    event_id = await _add_event(key="E:GONE", ticker="GONE", when=_in_days(4))
    r = await client.post(f"/api/events/{event_id}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "CANCELED"

    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        assert row.status == EventStatus.CANCELED.value
        audits = [
            a
            for a in (await s.execute(AuditEvent.__table__.select())).all()
            if a.action == AuditAction.EVENT_UPDATED.value
        ]
    assert len(audits) == 1
    assert audits[0].details["user_canceled"] is True

    # A canceled event drops out of the default feed.
    assert (await client.get("/api/events?horizon=7d")).json()["events"] == []


# ---------------------------------------------------------------------------
# 8. POST /refresh
# ---------------------------------------------------------------------------


async def test_refresh_ingests_and_is_idempotent(client, monkeypatch):
    """The refresh button must be safe to press repeatedly: identity is the
    natural key, so the second press creates ZERO rows."""
    from libs.event_calendar.stub import StubEventCalendarProvider

    provider = StubEventCalendarProvider()
    monkeypatch.setattr(
        "apps.gateway.event_calendar.configured_providers", lambda settings: [provider]
    )
    async with SessionLocal() as s:
        s.add(WatchlistItem(ticker="NVDA", added_by="test"))
        await s.commit()

    first = (await client.post("/api/events/refresh")).json()
    assert first["created"] > 0
    second = (await client.post("/api/events/refresh")).json()
    assert second["created"] == 0

    # The stub plants its synthetic events at fixed offsets from the WINDOW
    # START, which the tick sets 400 days back — so they land in the past, not
    # the next 7 days. That is a property of the stub, not of the API, and the
    # custom horizon is what reaches them. Asserting against a rolling "30d"
    # here would have made this test pass or fail by accident.
    from apps.gateway.event_calendar import LOOKBACK_DAYS

    window_start = (
        datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    ).astimezone(EASTERN).date()
    feed = (
        await client.get(
            f"/api/events?horizon=custom&start={window_start.isoformat()}"
            f"&end={(window_start + timedelta(days=40)).isoformat()}"
        )
    ).json()
    assert feed["counts"]["total"] > 0
    # The stub is SYNTHETIC and says so in every title — no test may mistake
    # its output for real data.
    assert all("SYNTHETIC" in e["title"] for e in feed["events"])


async def test_refresh_surfaces_the_capability_report_into_the_feed(client, monkeypatch):
    """A 403 on the earnings calendar must reach the UI as a probed fact
    (capability False), which is what the honest-absence banner renders."""

    class DeniedEarnings:
        name = "massive_calendar"

        def capabilities(self):
            return {"earnings_calendar": False, "market_holidays": True}

        def fetch_events(self, *, tickers, start, end, as_of=None):
            return []

        def fetch_market_calendar(self, start, end):
            from libs.event_calendar.provider import CapabilityNotAvailable

            raise CapabilityNotAvailable("not offered")

    monkeypatch.setattr(
        "apps.gateway.event_calendar.configured_providers",
        lambda settings: [DeniedEarnings()],
    )
    monkeypatch.setattr(
        "apps.gateway.routers.events.configured_provider_names",
        lambda settings: ["massive_calendar"],
    )
    await client.post("/api/events/refresh")
    body = (await client.get("/api/events")).json()
    assert body["capabilities"]["massive_calendar"]["earnings_calendar"] is False
    assert body["freshness"]["per_provider"]["massive_calendar"]["last_ok_at"]


async def test_refresh_with_no_providers_returns_a_named_skip_not_an_error(
    client, monkeypatch
):
    monkeypatch.setattr(
        "apps.gateway.event_calendar.configured_providers", lambda settings: []
    )
    r = await client.post("/api/events/refresh")
    assert r.status_code == 200
    assert {"provider": "", "reason": "NO_PROVIDERS_CONFIGURED"} in r.json()["skipped"]


# ---------------------------------------------------------------------------
# 9. GET /calendar
# ---------------------------------------------------------------------------


async def test_market_calendar_endpoint_returns_stored_sessions(client):
    half_day = date(2026, 11, 27)
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=half_day,
                exchange="US",
                open_utc=_et(2026, 11, 27, 9, 30),
                close_utc=_et(2026, 11, 27, 13, 0),
                is_early_close=True,
                source="alpaca_calendar",
            )
        )
        await s.commit()
    body = (
        await client.get("/api/events/calendar?start=2026-11-01&end=2026-11-30")
    ).json()
    assert len(body["sessions"]) == 1
    session_row = body["sessions"][0]
    assert session_row["session_date"] == "2026-11-27"
    assert session_row["is_early_close"] is True
    # Extended hours were never reported by this source: honest nulls, never
    # a guessed 04:00/20:00.
    assert session_row["session_open_utc"] is None


async def test_market_calendar_with_no_rows_is_an_empty_list_not_a_guess(client):
    body = (await client.get("/api/events/calendar")).json()
    assert body["sessions"] == []


async def test_market_calendar_rejects_a_malformed_range(client):
    assert (await client.get("/api/events/calendar?start=nope")).status_code == 422
    assert (
        await client.get("/api/events/calendar?start=2026-12-01&end=2026-11-01")
    ).status_code == 422


# ---------------------------------------------------------------------------
# 10. Alert classification (the ALERT_RULES entry)
# ---------------------------------------------------------------------------


async def test_event_approaching_surfaces_in_the_alerts_feed(client):
    """ADR-006: alerts are a classification OVER the audit trail. The rule
    must turn the ingestion's audit row into a readable alert."""
    async with SessionLocal() as s:
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                actor_id="",
                action=AuditAction.EVENT_APPROACHING.value,
                entity_type=ec.ENTITY_TYPE,
                entity_id="101",
                details={
                    "horizon": 7,
                    "type": "EARNINGS",
                    "ticker": "NVDA",
                    "days_to_event": 2.9,
                    "scheduled_at": "2026-08-22T20:05:00+00:00",
                    "status": "CONFIRMED",
                },
                correlation_id="",
            )
        )
        await s.commit()

    alerts = (await client.get("/api/alerts")).json()
    approaching = [a for a in alerts if a["action"] == "EVENT_APPROACHING"]
    assert len(approaching) == 1
    assert approaching[0]["severity"] == "INFO"
    assert approaching[0]["ticker"] == "NVDA"
    # Floored, not rounded: 2.9 days away is "in 2 days", and rounding up
    # would move the event a day further out than it actually is.
    assert approaching[0]["title"] == "NVDA event in 2 days"


async def test_a_macro_event_alert_names_its_type_since_it_has_no_ticker(client):
    async with SessionLocal() as s:
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                actor_id="",
                action=AuditAction.EVENT_APPROACHING.value,
                entity_type=ec.ENTITY_TYPE,
                entity_id="102",
                details={
                    "horizon": 7,
                    "type": "FOMC_DECISION",
                    "ticker": "",
                    "days_to_event": 1.2,
                },
                correlation_id="",
            )
        )
        await s.commit()
    alerts = (await client.get("/api/alerts")).json()
    approaching = [a for a in alerts if a["action"] == "EVENT_APPROACHING"]
    assert approaching[0]["title"] == "FOMC_DECISION event in 1 day"


async def test_a_sparse_alert_row_degrades_instead_of_crashing(client):
    """Every other title builder tolerates missing details; this one must too."""
    async with SessionLocal() as s:
        s.add(
            AuditEvent(
                actor_type="SYSTEM",
                actor_id="",
                action=AuditAction.EVENT_APPROACHING.value,
                entity_type=ec.ENTITY_TYPE,
                entity_id="103",
                details={"event_key": "CPI:2026-09"},
                correlation_id="",
            )
        )
        await s.commit()
    alerts = (await client.get("/api/alerts")).json()
    approaching = [a for a in alerts if a["action"] == "EVENT_APPROACHING"]
    assert approaching[0]["title"] == "CPI:2026-09 event approaching"


# ---------------------------------------------------------------------------
# 7. §54 card summaries — the OPT-IN flag on the feed (Phase J)
#
# The behaviour under the flag is pinned in tests/test_events_timeline_api.py,
# next to the seam that computes it. What belongs HERE is the property that is
# about THIS endpoint's contract with its existing callers: the flag is off by
# default and, off, the response is exactly what it always was.
# ---------------------------------------------------------------------------


async def test_the_feed_carries_no_summary_key_unless_it_is_asked_for(client):
    """OFF BY DEFAULT, and the default response is unchanged.

    ``summary`` is additive and opt-in because the feed is also polled by the
    alert path and the header count, neither of which draws a card — and
    because a field that appeared unbidden would change the payload every
    existing consumer already parses. Asserted as an EQUALITY between the
    default and the explicitly-off response, not merely as the absence of one
    key: a flag that also reordered rows or grew a counts entry would be just
    as much of a regression, and only comparing the whole shape catches it.
    """
    await _add_event(key="EARNINGS:NVDA:2026-08-27", ticker="NVDA", when=_in_days(2))

    default = (await client.get("/api/events")).json()
    explicit_off = (await client.get("/api/events?summaries=false")).json()

    assert default["events"], "the fixture event must be inside the 7d horizon"
    assert all("summary" not in event for event in default["events"])
    assert set(default) == set(explicit_off)
    assert default["counts"] == explicit_off["counts"]
    assert [e["event_id"] for e in default["events"]] == [
        e["event_id"] for e in explicit_off["events"]
    ]


async def test_summaries_true_adds_a_summary_to_every_row_and_nothing_else(client):
    """ON, each row gains exactly one key, with every field honestly null.

    Present-with-nulls rather than omitted, because "the platform looked and
    found nothing" and "the platform did not look" are different facts and the
    card renders them differently (§44 rule 18). Nothing is invented to fill
    the block: an event with no stored analysis and no stored option metrics
    reports NONE and nulls, never a zero.
    """
    await _add_event(key="EARNINGS:NVDA:2026-08-27", ticker="NVDA", when=_in_days(2))

    off = (await client.get("/api/events")).json()
    on = (await client.get("/api/events?summaries=true")).json()

    plain = off["events"][0]
    summarised = on["events"][0]
    assert set(summarised) - set(plain) == {"summary"}
    summary = summarised["summary"]
    assert summary["analysis_status"] == "NONE"
    assert summary["implied_move_pct"] is None
    assert summary["implied_move_basis"] is None
    assert summary["historical_move_median_abs"] is None
    assert summary["historical_move_n"] is None
    assert summary["previous_event_actual_move_pct"] is None
    # Wherever an implied move can appear, so does the wording that says what
    # it is not (§54, §66): option-market pricing, not a forecast.
    assert "not a forecast" in summary["implied_move_note"]


async def test_an_empty_feed_with_summaries_on_is_still_an_empty_list(client):
    """No rows, no queries, no crash — the flag must not need a row to exist."""
    body = (await client.get("/api/events?summaries=true")).json()

    assert body["events"] == []
    assert body["counts"]["total"] == 0
