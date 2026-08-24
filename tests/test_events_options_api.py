"""Event options API — GET/POST /api/events/{id}/options (Phase I, U3; event
spec §18, §36, §37, §66, §85, §96; audit §7.3, §11 options section).

WHY EVERY EQUITY BAR HERE IS SEEDED AND EVERY OPTION BAR IS FETCHED. The
underlying's closes are inserted by these tests so the anchor sessions and the
realized move are arithmetic a reader can verify (100.00 -> 108.00 is +8%,
stated, not computed by the fixture's generator). The OPTION premiums come from
the stub provider on purpose: they are what the backfill's whole job is to
fetch and store, and a test that pre-seeded them would prove nothing about the
contract-selection and storage path.

The guarantees these tests defend, in the order they appear:

1. **A GET NEVER FETCHES OPTION BARS.** The provider is monkeypatched to
   EXPLODE if any option method is called, and the read endpoint is exercised
   against it for a PAST event. This is the load-bearing performance property:
   an eight-row history table must not become seventeen provider calls.
2. **Only POST writes**, and it writes BOTH tables plus a DATA_BACKFILL audit
   row carrying ``kind: "event_option_straddle"``.
3. **The straddle is the arithmetic the library specifies**, pinned against
   the stub's own documented premium formula rather than against whatever the
   endpoint returned — otherwise the assertion is a mirror.
4. **NEVER A FABRICATED PREMIUM** (§44 rule 18). A provider that serves the
   call but not the put yields ``NO_DATA`` and a NULL ``implied_move_pct``,
   never half a straddle. A vendor without the capability at all
   (``CapabilityNotAvailable``, which is exactly what Alpaca raises) is 200
   with a named reason and NO stored row.
5. **The two bases are never blended** (§37). An upcoming event is priced off
   the LIVE chain and labelled ``LIVE_CHAIN_SNAPSHOT``; a past one off stored
   closes and labelled ``HISTORICAL_DAILY_CLOSE_APPROXIMATION``. The §37
   disclaimer wording is in the payload.
6. **Point-in-time**: contracts are listed with ``as_of`` = the PRE-event
   session, so a strike created in reaction to the print cannot enter the
   straddle. Pinned by capturing the ``as_of`` the seam actually passes.

Uses the shared ``client`` fixture (conftest.py).
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from apps.gateway import event_options as eo
from apps.gateway.db import (
    AuditEvent,
    EventOptionMetricRow,
    EventRow,
    OptionDailyBarRow,
    SessionLocal,
    StockBarDaily,
)
from libs.market_data import CapabilityNotAvailable, MarketDataError
from libs.market_data.stub import StubProvider
from libs.trading_core.events.implied_move import (
    BASIS_HISTORICAL,
    BASIS_LIVE,
    DISCLAIMER,
    STATUS_NO_DATA,
    STATUS_OK,
    STATUS_PARTIAL,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _et(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    """An ET wall-clock instant as its UTC equivalent (what the DB stores)."""
    return datetime(y, m, d, hour, minute, tzinfo=EASTERN).astimezone(timezone.utc)


def _weekdays(start: date, count: int) -> list[date]:
    """Consecutive WEEKDAYS from ``start`` — the bar dates ARE trading days,
    exactly as the pure library assumes (it never consults a calendar)."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


async def _seed_daily(ticker: str, *, start: date, closes: list[float]) -> list[date]:
    """One daily equity bar per weekday with the given closes.

    ``open`` is 1% below ``close`` so the gap and the 1D return are DIFFERENT
    numbers and no assertion can confuse the two.
    """
    days = _weekdays(start, len(closes))
    async with SessionLocal() as s:
        for day, close in zip(days, closes):
            s.add(
                StockBarDaily(
                    ticker=ticker,
                    ts=day,
                    open=round(close * 0.99, 6),
                    high=round(close * 1.02, 6),
                    low=round(close * 0.97, 6),
                    close=close,
                    volume=1_000_000.0,
                )
            )
        await s.commit()
    return days


async def _add_event(
    *,
    key: str,
    ticker: str | None,
    when: datetime,
    event_type: EventType = EventType.EARNINGS,
    session: EventSession = EventSession.AFTER_MARKET,
    status: EventStatus = EventStatus.CONFIRMED,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title="Earnings",
            ticker=ticker,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=session.value,
            status=status.value,
            source=EventSourceKind.COMPANY_IR_SEC.value,
            source_name="sec_edgar",
            source_url="https://example.invalid/ir",
            revision_history=[],
        )
        s.add(row)
        await s.commit()
        return row.id


#: The AMC release day and the session that reacts to it. Mon 2026-02-02
#: 16:30 ET is after the close, so the pre-event anchor is Monday's own close
#: and the reaction is Tuesday's — the §17 rule this whole file depends on.
RELEASE_DAY = date(2026, 2, 2)
REACT_DAY = date(2026, 2, 3)

#: An ``as_of`` comfortably after the fixture's last bar and comfortably before
#: the suite's real clock — the endpoints 422 a future ``as_of`` on purpose.
PAST_AS_OF = "2026-03-01T00:00:00Z"


async def _past_event_fixture(ticker: str = "ACME") -> dict:
    """One PAST AMC earnings print with the underlying seeded flat then up 8%.

    Flat at 100.00 through the release day makes the pre-event close exactly
    100.00, and the step to 108.00 on the reaction day makes the realized move
    exactly +8% — both numbers this file states rather than derives.
    """
    days = _weekdays(date(2026, 1, 5), 30)
    closes = [100.0] * len(days)
    react_index = days.index(REACT_DAY)
    for i in range(react_index, len(days)):
        closes[i] = 108.0
    await _seed_daily(ticker, start=date(2026, 1, 5), closes=closes)
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{RELEASE_DAY.isoformat()}",
        ticker=ticker,
        when=_et(2026, 2, 2, 16, 30),
    )
    return {"event_id": event_id, "ticker": ticker, "days": days}


class ExplodingOptionProvider:
    """A provider that fails the test if any OPTION method is called.

    Used to prove the GET route fetches nothing. Asserting "the row count did
    not change" would pass an implementation that fetched and discarded; only
    refusing to serve does.
    """

    def list_option_contracts(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("GET /options must not list contracts")

    def get_option_history_bars(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("GET /options must not fetch option bars")

    def get_option_chain(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("GET /options must not fetch a chain for a PAST event")


class NoCapabilityProvider:
    """Alpaca's honest refusal, in the shape the seam must survive.

    ``CapabilityNotAvailable`` from the CONTRACT LISTING is the first thing the
    backfill hits on a provider without dated option reference data, and it
    must become a reason rather than either a 5xx or an empty-list "this symbol
    has no options" — those are different facts.
    """

    def list_option_contracts(self, *a, **k):
        raise CapabilityNotAvailable(
            "massive: /v3/reference/options/contracts is not on this plan"
        )

    def get_option_history_bars(self, *a, **k):  # pragma: no cover - unreachable
        raise CapabilityNotAvailable("no historical option aggregates")


class OneLeggedProvider:
    """Serves the CALL's bars and nothing for the PUT.

    The single most important negative fixture in the file: a straddle with one
    leg is not an implied move, and the temptation to treat a missing leg as
    zero would halve every number that touched it — a mistake that reads as a
    cheap option rather than as absent data.
    """

    def __init__(self) -> None:
        self._stub = StubProvider()

    def list_option_contracts(self, *a, **k):
        return self._stub.list_option_contracts(*a, **k)

    def get_option_history_bars(self, option_ticker: str, start: date, end: date):
        if "P" in option_ticker.split(":", 1)[-1][6:]:
            return []
        return self._stub.get_option_history_bars(option_ticker, start, end)


class RecordingProvider:
    """The stub, with every option call's arguments recorded.

    Exists for the point-in-time assertion: the guarantee that contracts are
    listed AS OF the pre-event session is a property of the ARGUMENT the seam
    passes, and no response shape can prove it.
    """

    def __init__(self) -> None:
        self._stub = StubProvider()
        self.contract_calls: list[dict] = []
        self.bar_calls: list[tuple[str, date, date]] = []

    def list_option_contracts(self, underlying: str, **kwargs):
        self.contract_calls.append({"underlying": underlying, **kwargs})
        return self._stub.list_option_contracts(underlying, **kwargs)

    def get_option_history_bars(self, option_ticker: str, start: date, end: date):
        self.bar_calls.append((option_ticker, start, end))
        return self._stub.get_option_history_bars(option_ticker, start, end)


# ---------------------------------------------------------------------------
# POST /options/backfill — the write path
# ---------------------------------------------------------------------------


async def test_backfill_stores_option_bars_and_metrics(client):
    """The happy path: both tables written, status OK, numbers present."""
    fixture = await _past_event_fixture()
    r = await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["fetched"] is True
    assert body["basis"] == BASIS_HISTORICAL
    assert body["status"] == STATUS_OK
    assert body["stored_bars"] > 0
    # The stub ladder is 90..110 by 5 around a nominal 100 spot, and the
    # pre-event close is exactly 100.00 — so the ATM strike is 100.0, stated
    # rather than read back from the response.
    assert body["strike"] == 100.0
    assert body["call_ticker"].startswith("O:ACME")
    assert body["put_ticker"].startswith("O:ACME")
    assert body["implied_move_pct"] is not None and body["implied_move_pct"] > 0.0

    async with SessionLocal() as s:
        bars = (await s.execute(select(OptionDailyBarRow))).scalars().all()
        metrics = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .all()
        )
    assert len(bars) == body["stored_bars"]
    assert {b.option_ticker for b in bars} == {body["call_ticker"], body["put_ticker"]}
    # Every stored premium is a real positive number — the storage path DROPS
    # non-positive closes rather than persisting a zero that a later straddle
    # would read as a free option.
    assert all(b.close > 0.0 for b in bars)
    assert all(b.provider == "stub" for b in bars)

    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.basis == BASIS_HISTORICAL
    assert metric.status == STATUS_OK
    assert metric.strike == 100.0
    assert metric.spot == 100.0
    assert metric.pre_call_close is not None and metric.pre_call_close > 0.0
    assert metric.pre_put_close is not None and metric.pre_put_close > 0.0


async def test_backfill_straddle_matches_the_stub_premium_formula(client):
    """The stored implied move IS ``(call + put) / spot`` on the anchor session.

    Recomputed here from the stub's OWN documented bar series rather than from
    the endpoint's answer — an assertion that read the response back would be a
    mirror and would pass any arithmetic at all.
    """
    fixture = await _past_event_fixture()
    await client.post(f"/api/events/{fixture['event_id']}/options/backfill")

    async with SessionLocal() as s:
        metric = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .one()
        )

    stub = StubProvider()
    window_start = RELEASE_DAY - timedelta(days=eo.BARS_BEFORE_DAYS)
    window_end = REACT_DAY + timedelta(days=eo.BARS_AFTER_DAYS)
    call_bars = {
        b.ts: b.close
        for b in stub.get_option_history_bars(
            metric.call_ticker, window_start, window_end
        )
    }
    put_bars = {
        b.ts: b.close
        for b in stub.get_option_history_bars(
            metric.put_ticker, window_start, window_end
        )
    }
    # The AMC pre-event anchor is the RELEASE DAY's own close (§17): the 16:00
    # print is the last price before a 16:30 release.
    expected_pct = (call_bars[RELEASE_DAY] + put_bars[RELEASE_DAY]) / 100.0
    assert metric.pre_call_close == pytest.approx(call_bars[RELEASE_DAY])
    assert metric.pre_put_close == pytest.approx(put_bars[RELEASE_DAY])
    assert metric.implied_move_pct == pytest.approx(expected_pct)
    assert metric.implied_move_points == pytest.approx(expected_pct * 100.0)
    # And the post side is the REACTION day, which is what makes the crush
    # measurable at all.
    assert metric.post_call_close == pytest.approx(call_bars[REACT_DAY])
    assert metric.post_put_close == pytest.approx(put_bars[REACT_DAY])


async def test_backfill_records_the_realized_move_and_ratio(client):
    """+8% realized against the straddle gives the §66 ratio and a label."""
    fixture = await _past_event_fixture()
    await client.post(f"/api/events/{fixture['event_id']}/options/backfill")

    async with SessionLocal() as s:
        metric = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .one()
        )
    # 100.00 -> 108.00 across the anchor pair. Stated, not derived.
    assert metric.actual_move_pct == pytest.approx(0.08)
    assert metric.implied_realized_ratio == pytest.approx(
        0.08 / metric.implied_move_pct
    )
    assert metric.classification in ("UNDER_PRICED", "FAIR", "OVER_PRICED")


async def test_backfill_writes_a_data_backfill_audit_row(client):
    """The write is audited in the same transaction (rule 12, ADR-003)."""
    fixture = await _past_event_fixture()
    await client.post(f"/api/events/{fixture['event_id']}/options/backfill")

    async with SessionLocal() as s:
        rows = (
            (
                await s.execute(
                    select(AuditEvent)
                    .where(AuditEvent.entity_type == "option_daily_bars")
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    details = rows[0].details
    assert details["kind"] == "event_option_straddle"
    assert details["event_id"] == fixture["event_id"]
    assert details["basis"] == BASIS_HISTORICAL
    assert details["strike"] == 100.0
    assert details["status"] == STATUS_OK
    assert details["bars"] > 0


async def test_backfill_is_idempotent_on_the_metrics_row(client):
    """A second press UPDATES the one row; UNIQUE(event_id, basis) holds.

    Two rows for one (event, basis) would mean the payload had to pick one,
    and "the newest" is a rule nothing else in the platform enforces.
    """
    fixture = await _past_event_fixture()
    first = (
        await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    ).json()
    second = (
        await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    ).json()
    assert first["status"] == second["status"] == STATUS_OK
    assert first["implied_move_pct"] == pytest.approx(second["implied_move_pct"])

    async with SessionLocal() as s:
        metrics = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .all()
        )
        bars = (
            (await s.execute(select(OptionDailyBarRow)))
            .scalars()
            .all()
        )
    assert len(metrics) == 1
    # The bar upsert lands on (option_ticker, bar_date) too — a refetch can
    # overwrite a session, never duplicate it.
    assert len(bars) == first["stored_bars"]


async def test_contracts_are_listed_as_of_the_pre_event_session(client):
    """§96: the contract universe asked for is the PRE-event one.

    This is the point-in-time guarantee and it is a property of the ARGUMENT,
    not of the response — a strike created in reaction to the print is exactly
    what today's universe would return, and no payload shape would reveal it.
    """
    fixture = await _past_event_fixture()
    recorder = RecordingProvider()
    async with SessionLocal() as s:
        row = await s.get(EventRow, fixture["event_id"])
        await eo.backfill_event_options(
            s, row, provider=recorder, provider_name="stub"
        )

    assert recorder.contract_calls, "the seam must probe for contracts"
    for call in recorder.contract_calls:
        # AMC release => the pre-event session IS the release day's close.
        assert call["as_of"] == RELEASE_DAY
        assert call["underlying"] == "ACME"
    # The bar window must span BOTH anchors, or the crush is unmeasurable.
    for _ticker, start, end in recorder.bar_calls:
        assert start <= RELEASE_DAY
        assert end >= REACT_DAY


async def test_expiry_probe_selects_a_contract_that_spans_the_event(client):
    """§18: the chosen expiry is on/after the event, never before it."""
    fixture = await _past_event_fixture()
    body = (
        await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    ).json()
    expiry = date.fromisoformat(body["expiry"])
    assert expiry > RELEASE_DAY, (
        "an AFTER_MARKET print cannot be priced by a contract expiring at that "
        "same close — §18's strict branch"
    )
    assert expiry.weekday() == 4, "the probe walks weekly Fridays"


# ---------------------------------------------------------------------------
# Honest absence — the §44 rule 18 paths
# ---------------------------------------------------------------------------


async def test_capability_not_available_is_200_with_a_reason_and_a_no_data_row(client):
    """Alpaca's refusal is a REASON, not a 5xx and not an empty answer.

    UPDATED: it now stores a NO_DATA row carrying that reason. The row asserts
    NO price whatsoever — every premium column is NULL — so it still cannot be
    mistaken for a measurement of the option market. What it records is that
    the attempt HAPPENED and came back empty, which is the distinction a run
    that wrote nothing at all could not make: a history five rows short looked
    identical to a history nobody had ever backfilled, and that is how a
    rate-limited run lost six of eight events in silence.
    """
    fixture = await _past_event_fixture()
    async with SessionLocal() as s:
        row = await s.get(EventRow, fixture["event_id"])
        result = await eo.backfill_event_options(
            s, row, provider=NoCapabilityProvider(), provider_name="alpaca"
        )
    assert result["fetched"] is False
    assert result["status"] == STATUS_NO_DATA
    assert result["stored_bars"] == 0
    assert "capability_not_available" in result["reason"]
    assert "/v3/reference/options/contracts" in result["reason"]

    async with SessionLocal() as s:
        metrics = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .all()
        )
    assert len(metrics) == 1
    stored = metrics[0]
    assert stored.event_id == fixture["event_id"]
    assert stored.basis == BASIS_HISTORICAL
    assert stored.status == STATUS_NO_DATA
    assert "capability_not_available" in stored.notes["reason"]
    # NOT A PRICE. The row records an absence and prices nothing.
    assert stored.implied_move_pct is None
    assert stored.pre_call_close is None
    assert stored.pre_put_close is None
    assert stored.strike is None


async def test_a_missing_leg_is_no_data_never_half_a_straddle(client):
    """The put has no bars => NO_DATA and a NULL implied move.

    The mistake this guards against is arithmetic, not cosmetic: treating the
    absent put as 0.0 would report an implied move roughly half its true size,
    and the result would look like a cheap option rather than like missing data.
    """
    fixture = await _past_event_fixture()
    async with SessionLocal() as s:
        row = await s.get(EventRow, fixture["event_id"])
        result = await eo.backfill_event_options(
            s, row, provider=OneLeggedProvider(), provider_name="stub"
        )
    assert result["status"] == STATUS_NO_DATA
    assert result["implied_move_pct"] is None
    assert "put" in result["reason"]

    async with SessionLocal() as s:
        metric = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .one()
        )
    assert metric.status == STATUS_NO_DATA
    assert metric.implied_move_pct is None
    assert metric.implied_move_points is None
    assert metric.pre_put_close is None
    # The CALL's real bars are still stored — the leg that traded is data, and
    # discarding it would lose a fact because its partner was missing.
    assert metric.pre_call_close is not None and metric.pre_call_close > 0.0
    assert metric.notes, "every None must carry its reason"


async def test_macro_event_without_a_ticker_says_so(client):
    """A CPI release has no issuer whose options these would be."""
    event_id = await _add_event(
        key="CPI:2026-02",
        ticker=None,
        when=_et(2026, 2, 11, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
    )
    r = await client.post(f"/api/events/{event_id}/options/backfill")
    assert r.status_code == 200
    body = r.json()
    assert body["fetched"] is False
    assert body["reason"] == "no_ticker"
    assert body["status"] == STATUS_NO_DATA


async def test_event_without_stored_equity_bars_reports_that(client):
    """No underlying bars => no anchor => an honest reason, not a guess.

    The straddle percentage divides a premium by the spot on the SAME session;
    without stored closes there is no spot, and inventing one is exactly the
    fabrication §44 forbids.
    """
    event_id = await _add_event(
        key="EARNINGS:NOBARS:2026-02-02",
        ticker="NOBARS",
        when=_et(2026, 2, 2, 16, 30),
    )
    body = (await client.post(f"/api/events/{event_id}/options/backfill")).json()
    assert body["fetched"] is False
    assert body["status"] == STATUS_NO_DATA
    assert "no stored daily bars" in body["reason"]


async def test_backfill_never_404s_for_a_real_event_but_does_for_a_missing_one(client):
    r = await client.post("/api/events/999999/options/backfill")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /options — the read path
# ---------------------------------------------------------------------------


async def test_get_returns_the_disclaimer_and_basis_for_a_past_event(client):
    """§37: the wording is IN the payload, beside a labelled basis."""
    fixture = await _past_event_fixture()
    await client.post(f"/api/events/{fixture['event_id']}/options/backfill")

    r = await client.get(
        f"/api/events/{fixture['event_id']}/options?as_of={PAST_AS_OF}"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER
    assert "not a forecast" in body["disclaimer"]
    assert body["event_id"] == fixture["event_id"]
    assert body["ticker"] == "ACME"
    assert body["is_upcoming"] is False

    current = body["current"]
    assert current["basis"] == BASIS_HISTORICAL
    assert current["status"] == STATUS_OK
    assert current["strike"] == 100.0
    assert current["implied_move_pct"] > 0.0
    assert current["actual_move_pct"] == pytest.approx(0.08)
    assert current["event_key"] == f"EARNINGS:ACME:{RELEASE_DAY.isoformat()}"
    assert current["event_date"] == RELEASE_DAY.isoformat()


async def test_get_never_fetches_option_data_for_a_past_event(client, monkeypatch):
    """A GET on a PAST event issues NO provider call at all.

    The exploding provider is installed at the factory, so any path that
    reached for the network — contracts, bars or a chain — fails loudly rather
    than silently costing seventeen requests per page open.
    """
    fixture = await _past_event_fixture()
    await client.post(f"/api/events/{fixture['event_id']}/options/backfill")

    monkeypatch.setattr(
        eo, "get_provider", lambda name: ExplodingOptionProvider()
    )
    r = await client.get(
        f"/api/events/{fixture['event_id']}/options?as_of={PAST_AS_OF}"
    )
    assert r.status_code == 200
    assert r.json()["current"]["status"] == STATUS_OK


async def test_get_without_a_backfill_says_what_to_press(client):
    """No stored metrics => ``current: null`` and a reason naming the remedy.

    Not an empty object and not a zeroed straddle: the tab must be able to
    distinguish "this event's options were never fetched" from "this event had
    no implied move", and only a named reason does that.
    """
    fixture = await _past_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fixture['event_id']}/options?as_of={PAST_AS_OF}"
        )
    ).json()
    assert body["current"] is None
    assert body["coverage"]["stored_metrics"] is False
    assert "options/backfill" in body["coverage"]["reason"]
    assert body["history"] == []
    assert body["stats"]["actual"]["n"] == 0
    assert body["stats"]["actual"]["median_abs"] is None
    assert body["comparison"]["implied_pct"] is None


async def test_get_history_stats_and_comparison_over_prior_events(client):
    """The §66 table: prior prints' implied vs actual, with the distribution.

    Three prior events with hand-chosen realized moves (+4%, +10%, -6%) so the
    median |move| is exactly 6% — a number this test states, and which the
    nearest-rank definition must reproduce over three samples.
    """
    ticker = "HIST"
    # One quarter apart, each AMC, each with its own realized move.
    plan = [
        (date(2025, 5, 5), date(2025, 5, 6), 0.04),
        (date(2025, 8, 4), date(2025, 8, 5), 0.10),
        (date(2025, 11, 3), date(2025, 11, 4), -0.06),
    ]
    days = _weekdays(date(2025, 4, 1), 260)
    closes = [100.0] * len(days)
    for release, react, move in plan:
        idx = days.index(react)
        level = closes[idx - 1] * (1.0 + move)
        for i in range(idx, len(days)):
            closes[i] = round(level, 6)
        # Settle back to a flat level before the next print so each event's
        # anchor pair is exactly its own move and nothing compounds into the
        # next quarter's measurement.
        next_release = next(
            (r for r, _a, _m in plan if r > release), None
        )
        if next_release is not None:
            reset = days.index(next_release)
            for i in range(idx + 5, reset + 1):
                closes[i] = round(level, 6)
    await _seed_daily(ticker, start=date(2025, 4, 1), closes=closes)

    prior_ids = []
    for release, _react, _move in plan:
        prior_ids.append(
            await _add_event(
                key=f"EARNINGS:{ticker}:{release.isoformat()}",
                ticker=ticker,
                when=datetime.combine(
                    release, time(16, 30), tzinfo=EASTERN
                ).astimezone(timezone.utc),
            )
        )
    current_id = await _add_event(
        key=f"EARNINGS:{ticker}:2026-02-02",
        ticker=ticker,
        when=_et(2026, 2, 2, 16, 30),
    )

    r = await client.post(
        f"/api/events/{current_id}/options/history/backfill?last=4&as_of={PAST_AS_OF}"
    )
    assert r.status_code == 200, r.text
    hist = r.json()
    assert hist["event_count"] == 3
    assert hist["counts"]["ok"] == 3
    assert hist["stored_bars"] > 0

    body = (
        await client.get(f"/api/events/{current_id}/options?as_of={PAST_AS_OF}")
    ).json()
    assert len(body["history"]) == 3
    # Newest first — the order the UI lists them and the order the chart draws.
    dates = [row["event_date"] for row in body["history"]]
    assert dates == sorted(dates, reverse=True)
    for row in body["history"]:
        assert row["basis"] == BASIS_HISTORICAL
        assert row["event_key"].startswith(f"EARNINGS:{ticker}:")
        assert row["implied_move_pct"] is not None
        assert row["actual_move_pct"] is not None

    stats = body["stats"]["actual"]
    assert stats["n"] == 3
    # |+4%|, |+10%|, |-6%| sorted -> 0.04, 0.06, 0.10; nearest-rank median is
    # the 2nd of 3 = 0.06, and the p90 is the 3rd = 0.10.
    assert stats["median_abs"] == pytest.approx(0.06, abs=1e-6)
    assert stats["p90_abs"] == pytest.approx(0.10, abs=1e-6)
    assert stats["max_abs"] == pytest.approx(0.10, abs=1e-6)
    assert body["stats"]["implied"]["n"] == 3

    comparison = body["comparison"]
    assert comparison["hist_median_abs"] == pytest.approx(0.06, abs=1e-6)
    assert comparison["hist_p90_abs"] == pytest.approx(0.10, abs=1e-6)
    assert comparison["hist_max_abs"] == pytest.approx(0.10, abs=1e-6)


async def test_history_omits_events_with_no_stored_metrics(client):
    """An unfetched prior print is ABSENT from history, never a blank row.

    A row of nulls would enter the chart as a zero-height bar and read as "the
    market priced nothing", which is a claim about the option market rather
    than about this platform's backfill state.
    """
    ticker = "SPARSE"
    days = _weekdays(date(2025, 10, 1), 120)
    await _seed_daily(ticker, start=date(2025, 10, 1), closes=[100.0] * len(days))
    await _add_event(
        key=f"EARNINGS:{ticker}:2025-11-03",
        ticker=ticker,
        when=_et(2025, 11, 3, 16, 30),
    )
    current_id = await _add_event(
        key=f"EARNINGS:{ticker}:2026-02-02",
        ticker=ticker,
        when=_et(2026, 2, 2, 16, 30),
    )
    body = (
        await client.get(f"/api/events/{current_id}/options?as_of={PAST_AS_OF}")
    ).json()
    assert body["coverage"]["history_events"] == 1
    assert body["coverage"]["history_with_metrics"] == 0
    assert body["history"] == []


async def test_get_rejects_a_future_as_of(client):
    """A future ``as_of`` is a 422, never a silent clamp to now.

    Clamping answers a DIFFERENT question than the one asked and the caller has
    no way to notice — the same rule every other event route applies.
    """
    fixture = await _past_event_fixture()
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    r = await client.get(
        f"/api/events/{fixture['event_id']}/options",
        params={"as_of": future},  # let httpx encode the offset's "+"
    )
    assert r.status_code == 422
    assert "no option market exists for it" in r.text


async def test_get_404s_for_a_missing_event(client):
    r = await client.get("/api/events/999999/options")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# The LIVE basis — upcoming events
# ---------------------------------------------------------------------------


async def test_upcoming_event_is_priced_off_the_live_chain(client):
    """§37: an upcoming print gets the LIVE basis, never the historical one."""
    ticker = "SOON"
    # Bars up to yesterday, an event next week: this is the state the Catalyst
    # page opens in, and the whole reason the live path exists.
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=60)
    days = _weekdays(start, 40)
    await _seed_daily(ticker, start=start, closes=[100.0] * len(days))
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{(today + timedelta(days=7)).isoformat()}",
        ticker=ticker,
        when=datetime.combine(
            today + timedelta(days=7), time(16, 30), tzinfo=EASTERN
        ).astimezone(timezone.utc),
    )

    body = (await client.get(f"/api/events/{event_id}/options")).json()
    assert body["is_upcoming"] is True
    current = body["current"]
    assert current["basis"] == BASIS_LIVE
    # PARTIAL, never OK: the event has not happened, so the crush, the realized
    # move and the ratio are genuinely unknown and OK would claim otherwise.
    assert current["status"] == STATUS_PARTIAL
    assert current["implied_move_pct"] is not None
    assert current["implied_move_pct"] > 0.0
    assert current["actual_move_pct"] is None
    assert current["iv_crush_pct"] is None
    assert current["implied_realized_ratio"] is None
    assert any("has not happened yet" in note for note in current["notes"])


async def test_live_path_stores_nothing(client):
    """A snapshot before the print is not a reconstruction of what was charged.

    Filing the live number as history would make today's guess indistinguishable
    from tomorrow's measurement — which is exactly what the two-basis split
    exists to prevent.
    """
    ticker = "NOSTORE"
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=60)
    days = _weekdays(start, 40)
    await _seed_daily(ticker, start=start, closes=[100.0] * len(days))
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{(today + timedelta(days=7)).isoformat()}",
        ticker=ticker,
        when=datetime.combine(
            today + timedelta(days=7), time(16, 30), tzinfo=EASTERN
        ).astimezone(timezone.utc),
    )
    await client.get(f"/api/events/{event_id}/options")

    async with SessionLocal() as s:
        metrics = (
            (await s.execute(select(EventOptionMetricRow)))
            .scalars()
            .all()
        )
    assert metrics == []


async def test_live_chain_failure_is_no_data_not_a_number(client, monkeypatch):
    """A vendor that refuses the chain yields NO_DATA with a reason.

    Never a 5xx (a quote outage must not take the Catalyst page with it) and
    never a number — the UI treats a finite value beside NO_DATA as the server
    retracting its own computation, so the server must not send one.
    """

    class NoChain:
        def get_option_chain(self, *a, **k):
            raise CapabilityNotAvailable("alpaca: options snapshots not on this plan")

    ticker = "NOCHAIN"
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=60)
    days = _weekdays(start, 40)
    await _seed_daily(ticker, start=start, closes=[100.0] * len(days))
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{(today + timedelta(days=7)).isoformat()}",
        ticker=ticker,
        when=datetime.combine(
            today + timedelta(days=7), time(16, 30), tzinfo=EASTERN
        ).astimezone(timezone.utc),
    )
    monkeypatch.setattr(eo, "get_provider", lambda name: NoChain())

    r = await client.get(f"/api/events/{event_id}/options")
    assert r.status_code == 200
    body = r.json()
    current = body["current"]
    assert current["basis"] == BASIS_LIVE
    assert current["status"] == STATUS_NO_DATA
    assert current["implied_move_pct"] is None
    assert "capability_not_available" in body["coverage"]["reason"]
    # And the comparison does not smuggle the retracted number back in.
    assert body["comparison"]["implied_pct"] is None


async def test_upcoming_macro_event_without_a_ticker_is_no_data(client):
    """No issuer, no chain — and the reason says which fact is missing."""
    today = datetime.now(timezone.utc).date()
    event_id = await _add_event(
        key="FOMC_DECISION:2099-01-01",
        ticker=None,
        when=datetime.combine(
            today + timedelta(days=14), time(14, 0), tzinfo=EASTERN
        ).astimezone(timezone.utc),
        event_type=EventType.FOMC_DECISION,
        session=EventSession.DURING_MARKET,
    )
    body = (await client.get(f"/api/events/{event_id}/options")).json()
    assert body["current"]["status"] == STATUS_NO_DATA
    assert body["coverage"]["reason"] == "no_ticker"


# ---------------------------------------------------------------------------
# POST /options/history/backfill
# ---------------------------------------------------------------------------


async def test_history_backfill_is_bounded_at_the_edges(client):
    """``last`` outside [1, 12] is a 422 at the boundary, not a clamp."""
    fixture = await _past_event_fixture()
    over = await client.post(
        f"/api/events/{fixture['event_id']}/options/history/backfill?last=13"
    )
    assert over.status_code == 422
    under = await client.post(
        f"/api/events/{fixture['event_id']}/options/history/backfill?last=0"
    )
    assert under.status_code == 422
    ok = await client.post(
        f"/api/events/{fixture['event_id']}/options/history/backfill"
        f"?last={eo.MAX_HISTORY_BACKFILL}&as_of={PAST_AS_OF}"
    )
    assert ok.status_code == 200


async def test_history_backfill_with_no_previous_events_says_so(client):
    """An event with no precedent is an honest empty result, not an error."""
    fixture = await _past_event_fixture()
    body = (
        await client.post(
            f"/api/events/{fixture['event_id']}/options/history/backfill"
            f"?as_of={PAST_AS_OF}"
        )
    ).json()
    assert body["events"] == []
    assert body["event_count"] == 0
    assert body["counts"] == {"ok": 0, "no_data": 0, "failed": 0, "skipped": 0}
    assert body["stored_bars"] == 0
    assert body["status"] == STATUS_NO_DATA
    assert "no previous comparable events" in body["reason"]
    assert body["results"] == []


async def test_history_backfill_isolates_each_event(client):
    """One event's failure does not cost the others their straddles (§8).

    The fixture gives the FIRST prior print no equity bars at all, so its
    anchor cannot resolve; the second is fully seeded. A shared try/except
    around the loop would lose both.
    """
    ticker = "ISOL"
    # Bars only from November: the August print predates them entirely.
    days = _weekdays(date(2025, 11, 3), 90)
    closes = [100.0] * len(days)
    react = days.index(date(2025, 11, 4))
    for i in range(react, len(days)):
        closes[i] = 105.0
    await _seed_daily(ticker, start=date(2025, 11, 3), closes=closes)

    await _add_event(
        key=f"EARNINGS:{ticker}:2025-08-04",
        ticker=ticker,
        when=_et(2025, 8, 4, 16, 30),
    )
    await _add_event(
        key=f"EARNINGS:{ticker}:2025-11-03",
        ticker=ticker,
        when=_et(2025, 11, 3, 16, 30),
    )
    current_id = await _add_event(
        key=f"EARNINGS:{ticker}:2026-02-02",
        ticker=ticker,
        when=_et(2026, 2, 2, 16, 30),
    )

    body = (
        await client.post(
            f"/api/events/{current_id}/options/history/backfill"
            f"?last=4&as_of={PAST_AS_OF}"
        )
    ).json()
    assert body["event_count"] == 2
    # The per-event list is the load-bearing part: a caller that only got a
    # count could not tell WHICH print is still missing.
    assert [e["event_key"] for e in body["events"]] == [
        f"EARNINGS:{ticker}:2025-11-03",
        f"EARNINGS:{ticker}:2025-08-04",
    ]
    assert body["counts"]["ok"] == 1
    assert body["counts"]["no_data"] == 1
    statuses = {r["event_key"]: r["status"] for r in body["results"]}
    assert statuses[f"EARNINGS:{ticker}:2025-11-03"] == STATUS_OK
    assert statuses[f"EARNINGS:{ticker}:2025-08-04"] == STATUS_NO_DATA
    # The survivor really stored something; the failure really did not.
    stored = {r["event_key"]: r["stored_bars"] for r in body["results"]}
    assert stored[f"EARNINGS:{ticker}:2025-11-03"] > 0
    assert stored[f"EARNINGS:{ticker}:2025-08-04"] == 0


async def test_history_backfill_404s_for_a_missing_event(client):
    r = await client.post("/api/events/999999/options/history/backfill")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Rate-limit survivability: pacing, the honest NO_DATA row, and the re-run
#
# The LIVE failure these defend: POST /options/history/backfill?last=8 fired
# ~32 provider requests back-to-back, Massive answered the tail with HTTP 429,
# and six of eight events ended with no bars AND NO ROW — the history table
# was six rows short and nothing anywhere said why. Three properties fix it:
# the walk paces itself, every failure is written down, and a re-run retries
# exactly the gaps.
# ---------------------------------------------------------------------------


class RateLimitedForTicker:
    """The stub, except one UNDERLYING's bars always raise a rate-limit error.

    Shaped like the live fault: the CONTRACT listing succeeds (the reference
    endpoint was not the one being throttled), and the BAR fetch is what the
    exhausted 429 backoff finally raises out of.
    """

    def __init__(self, blocked: str) -> None:
        self._stub = StubProvider()
        self.blocked = blocked
        self.bar_calls: list[str] = []

    def list_option_contracts(self, underlying: str, **kwargs):
        return self._stub.list_option_contracts(underlying, **kwargs)

    def get_option_history_bars(self, option_ticker: str, start: date, end: date):
        self.bar_calls.append(option_ticker)
        underlying = option_ticker.split(":", 1)[-1][:len(self.blocked)]
        if underlying == self.blocked:
            raise MarketDataError(
                "Massive rate limit (HTTP 429) persisted after 4 retries for "
                f"/v2/aggs/ticker/{option_ticker}/range/1/day"
            )
        return self._stub.get_option_history_bars(option_ticker, start, end)


async def _two_prior_events(ticker: str) -> int:
    """Two seeded prior prints plus the current one; returns the current id."""
    days = _weekdays(date(2025, 10, 1), 200)
    closes = [100.0] * len(days)
    react = days.index(date(2025, 11, 4))
    for i in range(react, len(days)):
        closes[i] = 105.0
    await _seed_daily(ticker, start=date(2025, 10, 1), closes=closes)
    for release in (date(2025, 11, 3), date(2026, 2, 2)):
        await _add_event(
            key=f"EARNINGS:{ticker}:{release.isoformat()}",
            ticker=ticker,
            when=datetime.combine(release, time(16, 30), tzinfo=EASTERN).astimezone(
                timezone.utc
            ),
        )
    return await _add_event(
        key=f"EARNINGS:{ticker}:2026-05-04",
        ticker=ticker,
        when=_et(2026, 5, 4, 16, 30),
    )


async def test_history_backfill_paces_between_events_but_not_before_the_first(
    client, monkeypatch
):
    """The burst is what earned the 429; the pause is the fix.

    Pinned on the CONSTANT rather than on a literal, and on the CALL COUNT:
    N events must sleep N-1 times, never N — a one-event run should not pay a
    toll for a burst it is not creating.
    """
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(eo.asyncio, "sleep", _record)
    current_id = await _two_prior_events("PACE")

    body = (
        await client.post(
            f"/api/events/{current_id}/options/history/backfill"
            f"?last=4&as_of=2026-04-01T00:00:00Z"
        )
    ).json()
    assert body["event_count"] == 2
    assert slept == [eo.HISTORY_BACKFILL_PACING_SECONDS]  # 2 events -> 1 pause


async def test_rate_limited_event_persists_a_no_data_row_naming_the_reason(
    client, monkeypatch
):
    """The live failure, reproduced: a 429 the adapter could not outlast.

    Before this change the event stored NOTHING — the response said
    "provider_error" and the database kept no trace, so the missing history
    row was indistinguishable from one nobody had backfilled. Now the gap is
    a row, and the row says why.
    """
    monkeypatch.setattr(eo, "HISTORY_BACKFILL_PACING_SECONDS", 0.0)
    ticker = "RLIM"
    current_id = await _two_prior_events(ticker)
    provider = RateLimitedForTicker(ticker)

    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        body = await eo.backfill_options_history(
            s,
            row,
            as_of=datetime(2026, 4, 1, tzinfo=timezone.utc),
            provider=provider,
            provider_name="massive",
            last=4,
        )

    assert body["event_count"] == 2
    assert body["counts"]["failed"] == 2
    assert body["counts"]["ok"] == 0
    for outcome in body["events"]:
        assert outcome["status"] == STATUS_NO_DATA
        assert "429" in outcome["reason"]
        assert outcome["stored_bars"] == 0
        assert outcome["event_id"] is not None
        assert outcome["event_key"].startswith(f"EARNINGS:{ticker}:")

    # The honest gap is IN THE DATABASE, not only in the response body.
    async with SessionLocal() as s:
        rows = (
            (await s.execute(select(EventOptionMetricRow))).scalars().all()
        )
    assert len(rows) == 2
    for stored in rows:
        assert stored.status == STATUS_NO_DATA
        assert "429" in stored.notes["reason"]
        assert stored.implied_move_pct is None


async def test_rerun_overwrites_a_no_data_row_when_the_provider_recovers(
    client, monkeypatch
):
    """The whole point of writing the gap down: the re-run can close it.

    First pass rate-limited -> NO_DATA rows. Second pass with a healthy
    provider must OVERWRITE them with real straddles, without ``force`` — a
    NO_DATA row is a record of a gap, not of an answer.
    """
    monkeypatch.setattr(eo, "HISTORY_BACKFILL_PACING_SECONDS", 0.0)
    ticker = "RERUN"
    current_id = await _two_prior_events(ticker)
    as_of = datetime(2026, 4, 1, tzinfo=timezone.utc)

    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        first = await eo.backfill_options_history(
            s, row, as_of=as_of, provider=RateLimitedForTicker(ticker),
            provider_name="massive", last=4,
        )
    assert first["counts"]["failed"] == 2

    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        second = await eo.backfill_options_history(
            s, row, as_of=as_of, provider=StubProvider(),
            provider_name="stub", last=4,
        )
    assert second["counts"]["ok"] == 2
    assert second["counts"]["failed"] == 0
    assert second["stored_bars"] > 0

    async with SessionLocal() as s:
        rows = (
            (await s.execute(select(EventOptionMetricRow))).scalars().all()
        )
    # OVERWRITTEN, not duplicated — UNIQUE(event_id, basis) still holds.
    assert len(rows) == 2
    for stored in rows:
        assert stored.status != STATUS_NO_DATA
        assert stored.implied_move_pct is not None


async def test_rerun_skips_ok_events_unless_force_is_set(client, monkeypatch):
    """An OK row is an answer; re-fetching it costs provider calls for nothing.

    So the second press is a no-op that reports ``skipped`` — until
    ``force=true``, which re-does the work. Proven on the PROVIDER CALL COUNT:
    an assertion on the stored row could not tell a skip from a rewrite.
    """
    monkeypatch.setattr(eo, "HISTORY_BACKFILL_PACING_SECONDS", 0.0)
    ticker = "FORCE"
    current_id = await _two_prior_events(ticker)
    as_of = datetime(2026, 4, 1, tzinfo=timezone.utc)

    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        first = await eo.backfill_options_history(
            s, row, as_of=as_of, provider=StubProvider(), provider_name="stub", last=4
        )
    assert first["counts"]["ok"] == 2

    quiet = RecordingProvider()
    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        second = await eo.backfill_options_history(
            s, row, as_of=as_of, provider=quiet, provider_name="stub", last=4
        )
    assert second["counts"]["skipped"] == 2
    assert second["stored_bars"] == 0
    assert quiet.bar_calls == []  # nothing re-fetched
    assert all(e["skipped"] for e in second["events"])

    forced = RecordingProvider()
    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        third = await eo.backfill_options_history(
            s, row, as_of=as_of, provider=forced, provider_name="stub", last=4,
            force=True,
        )
    assert third["counts"]["ok"] == 2
    assert third["counts"]["skipped"] == 0
    assert third["force"] is True
    assert len(forced.bar_calls) == 4  # two legs x two events, really re-fetched


async def test_force_query_param_reaches_both_backfill_endpoints(client):
    """The flag is reachable over HTTP, not only from the seam."""
    fixture = await _past_event_fixture("FQP")
    single = await client.post(
        f"/api/events/{fixture['event_id']}/options/backfill?force=true"
    )
    assert single.status_code == 200, single.text
    assert single.json()["status"] == STATUS_OK

    hist = await client.post(
        f"/api/events/{fixture['event_id']}/options/history/backfill"
        f"?as_of={PAST_AS_OF}&force=true"
    )
    assert hist.status_code == 200, hist.text
    assert hist.json()["force"] is True


async def test_a_stored_no_data_row_is_counted_but_never_charted(client, monkeypatch):
    """The gap is visible in ``coverage``, absent from ``history``.

    A NO_DATA row must NOT become a history entry: a row of nulls enters the
    §66 chart as a zero-height bar and reads as "the market priced nothing",
    which is a claim about the option market rather than about this platform's
    backfill state. But it must not vanish either — the count is what lets the
    UI tell "we looked and came back empty" from "nobody has looked".
    """
    monkeypatch.setattr(eo, "HISTORY_BACKFILL_PACING_SECONDS", 0.0)
    ticker = "GAPCT"
    current_id = await _two_prior_events(ticker)
    as_of = datetime(2026, 4, 1, tzinfo=timezone.utc)

    async with SessionLocal() as s:
        row = await s.get(EventRow, current_id)
        await eo.backfill_options_history(
            s, row, as_of=as_of, provider=RateLimitedForTicker(ticker),
            provider_name="massive", last=4,
        )

    body = (
        await client.get(
            f"/api/events/{current_id}/options?as_of=2026-04-01T00:00:00Z"
        )
    ).json()
    assert body["history"] == []
    assert body["coverage"]["history_events"] == 2
    assert body["coverage"]["history_with_metrics"] == 0
    assert body["coverage"]["history_attempted_no_data"] == 2
    # No fabricated distribution off zero rows.
    assert body["stats"]["actual"]["n"] == 0


async def test_single_backfill_skips_nothing_by_default(client):
    """``skip_if_ok`` belongs to the HISTORY walk, not to the single-event
    button: pressing it is an explicit request to refresh THIS event, and a
    later backfill legitimately sees a post-event bar the first one missed."""
    fixture = await _past_event_fixture("SINGLE")
    first = await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    assert first.json()["fetched"] is True
    second = await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    assert second.json()["fetched"] is True
    assert second.json().get("skipped") is not True


# ---------------------------------------------------------------------------
# Seam-level units
# ---------------------------------------------------------------------------


async def test_anchor_sessions_follow_the_session_rule(client):
    """§17: an AMC print anchors on its OWN close, a BMO print on the day before.

    Getting this backwards puts the reaction INSIDE the pre-event straddle,
    which reads as an implied move that already knew the answer — the single
    most consequential off-by-one in the whole seam.
    """
    from apps.gateway.event_price import to_daily_bars

    days = _weekdays(date(2026, 1, 26), 10)
    closes = [100.0 + i for i in range(len(days))]
    await _seed_daily("ANCH", start=date(2026, 1, 26), closes=closes)

    amc = await _add_event(
        key="EARNINGS:ANCH:AMC", ticker="ANCH", when=_et(2026, 2, 2, 16, 30)
    )
    bmo = await _add_event(
        key="EARNINGS:ANCH:BMO",
        ticker="ANCH",
        when=_et(2026, 2, 2, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    async with SessionLocal() as s:
        bars = await eo._equity_bars(s, "ANCH")
        amc_row = await s.get(EventRow, amc)
        bmo_row = await s.get(EventRow, bmo)

    amc_anchors, reason = eo.anchor_sessions(bars, amc_row)
    assert reason is None
    assert amc_anchors["pre_date"] == RELEASE_DAY
    assert amc_anchors["post_date"] == REACT_DAY

    bmo_anchors, reason = eo.anchor_sessions(bars, bmo_row)
    assert reason is None
    assert bmo_anchors["pre_date"] == date(2026, 1, 30)  # the Friday before
    assert bmo_anchors["post_date"] == RELEASE_DAY  # the release day itself


def test_expiry_candidates_are_weekly_fridays_from_the_event():
    """The probe list, pinned: Fridays, and the event day when it IS a Friday."""
    # Mon 2026-02-02 -> the following Friday and the two after it.
    assert eo._expiry_candidates(date(2026, 2, 2)) == [
        date(2026, 2, 6),
        date(2026, 2, 13),
        date(2026, 2, 20),
    ]
    # A Friday event includes ITSELF: whether a same-day expiry may price the
    # print is §18's question (it may, for a BEFORE_MARKET release), and
    # excluding it here would pre-empt that rule.
    assert eo._expiry_candidates(date(2026, 2, 6))[0] == date(2026, 2, 6)


def test_option_history_provider_prefers_massive_when_keyed():
    """Historical bars follow the FUNDAMENTALS rule, not market_data_provider.

    Massive is the only vendor with an as-of contract reference; routing this
    to Alpaca because it happens to be the quote provider would turn every
    reconstruction into a CapabilityNotAvailable.
    """

    class S:
        massive_api_key = "k"
        market_data_provider = "alpaca"

    class Unkeyed:
        massive_api_key = ""
        market_data_provider = "stub"

    assert eo.option_history_provider_name(S()) == "massive"
    assert eo.option_history_provider_name(Unkeyed()) == "stub"


async def test_stored_bars_are_read_back_when_a_refetch_returns_nothing(client):
    """A second backfill whose provider serves ``[]`` keeps the stored answer.

    A delisted contract 404s on refetch, and pricing the straddle off the fetch
    alone would downgrade a complete stored answer to NO_DATA. The store is the
    accumulated truth; the fetch only tops it up.
    """
    fixture = await _past_event_fixture()
    first = (
        await client.post(f"/api/events/{fixture['event_id']}/options/backfill")
    ).json()
    assert first["status"] == STATUS_OK

    class EmptyBars:
        def __init__(self) -> None:
            self._stub = StubProvider()

        def list_option_contracts(self, *a, **k):
            return self._stub.list_option_contracts(*a, **k)

        def get_option_history_bars(self, *a, **k):
            return []

    async with SessionLocal() as s:
        row = await s.get(EventRow, fixture["event_id"])
        second = await eo.backfill_event_options(
            s, row, provider=EmptyBars(), provider_name="stub"
        )
    assert second["stored_bars"] == 0
    assert second["status"] == STATUS_OK
    assert second["implied_move_pct"] == pytest.approx(first["implied_move_pct"])


async def test_past_macro_event_does_not_offer_a_useless_backfill(client):
    """A ticker-less PAST event says ``no_ticker``, not "press backfill".

    That button answers ``no_ticker`` too, so naming it would send the user to
    a remedy that cannot work — a reason that points somewhere useless is worse
    than one that names what is actually missing.
    """
    event_id = await _add_event(
        key="CPI:2026-01",
        ticker=None,
        when=_et(2026, 1, 13, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
    )
    body = (
        await client.get(f"/api/events/{event_id}/options?as_of={PAST_AS_OF}")
    ).json()
    assert body["is_upcoming"] is False
    assert body["current"] is None
    assert body["coverage"]["reason"] == "no_ticker"
    assert "options/backfill" not in (body["coverage"]["reason"] or "")
