"""Event replay API — GET/POST /api/events/{id}/replay|history (Phase C, U3;
event spec §14, §17, §19, §20, §60, §85, §96; audit §7, §11.4).

WHY EVERY BAR HERE IS SEEDED, NEVER FETCHED. Both the daily and the MINUTE
series are inserted by these tests before the endpoint is called. Seeding
pins the arithmetic to numbers computable by hand (a +2% gap is a +2% gap),
and it keeps the stub provider's synthetic random walk out of the assertions —
a test whose expected +30m move comes from the same generator that produced
the bars proves nothing about the anchor rule. The tests that deliberately
have NO seeded minute bars are the ones about the fetch path and about honest
absence, which is where the provider belongs.

The guarantees these tests defend, in the order they appear:

1. **The as-of gate is real for MINUTES too** (§14, §96), and pinned in
   PAIRS, never singly: at 09:44 ET the +30m bar is absent AND at 10:05 ET it
   is present. A gate that returned nothing at all would pass the first half
   of every pair and fail the second.
2. **A GET never fetches minute bars.** The provider is monkeypatched to
   EXPLODE if called, and the read endpoints are exercised against it. This
   is the contract's load-bearing performance property: twelve events on a
   history page must not become twelve provider fetches.
3. **Only POST writes**, idempotently, with a DATA_BACKFILL audit row carrying
   ``kind: "intraday_event_window"`` — and a second press stores nothing.
4. **A future event is 200 + ``available: false``**, with its registry facts
   attached. The Catalyst page opens on UPCOMING earnings; a 404 there would
   hide the card the user just clicked.
5. **Honest absence, never a zero** (§44 rule 18, §85). No provider, no
   ticker, no stored minutes, no consensus and no implied move each get their
   own shape and their own reason; the §60 table's three permanently
   unavailable columns are PRESENT and marked, never omitted.
6. **The window spans the right days.** An AMC release needs tomorrow's open;
   a BMO one is answered by its own morning. Fetching the wrong span is the
   difference between a measured reaction and no reaction at all.

Uses the shared ``client`` / ``unconfigured_client`` fixtures (conftest.py).
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from apps.gateway import event_replay as er
from apps.gateway.db import (
    AuditEvent,
    EventRow,
    MarketCalendarRow,
    SessionLocal,
    StockBar1mRow,
    StockBarDaily,
)
from libs.market_data import CapabilityNotAvailable
from libs.market_data.provider import IntradayBar
from libs.trading_core.models import AuditAction
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


def _et_on(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=EASTERN).astimezone(
        timezone.utc
    )


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


async def _seed_daily(
    ticker: str, *, start: date, closes: list[float]
) -> list[date]:
    """One daily bar per weekday with the given closes.

    ``open`` is 1% below ``close`` so the gap return and the 1D return are
    DIFFERENT numbers and a test can tell which one an implementation
    reported.
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


async def _seed_minutes(
    ticker: str, bars: list[tuple[datetime, float]], *, volume: int = 1_000
) -> None:
    """Insert minute bars from ``(ts_utc, close)`` pairs.

    ``open``/``high``/``low`` are pinned TO the close so every window's move is
    exactly ``close/reference - 1`` and the expected numbers stay arithmetic a
    reader can verify, not a fixture's secret.
    """
    async with SessionLocal() as s:
        for ts, close in bars:
            s.add(
                StockBar1mRow(
                    ticker=ticker,
                    ts=ts,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=volume,
                )
            )
        await s.commit()


async def _add_event(
    *,
    key: str,
    ticker: str | None,
    when: datetime,
    event_type: EventType = EventType.EARNINGS,
    session: EventSession = EventSession.AFTER_MARKET,
    status: EventStatus = EventStatus.CONFIRMED,
    title: str = "Earnings",
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title=title,
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


#: Where the multi-event HISTORY fixtures start. A 200-weekday span is ~9.5
#: months, so this is anchored well in the past: ``as_of`` may never be in the
#: future (the endpoints 422 it on purpose), and a fixture whose newest print
#: is next month would make the whole file fail by the calendar rather than by
#: the code. The single-event AMC fixture below has a much shorter span and
#: keeps its own 2026 dates.
HISTORY_START = date(2025, 1, 6)

#: An ``as_of`` comfortably after every history fixture's last print and
#: comfortably before the suite's real clock. Named so a reader can see at a
#: glance that no history test smuggles in a future instant.
HISTORY_AS_OF = "2026-01-05T00:00:00Z"

#: The reaction-day session used by the AMC fixtures below: Tue 2026-02-03.
#: Its own release is Mon 2026-02-02 at 16:30 ET (after the close), so the
#: whole reaction lands on the following morning.
RELEASE_DAY = date(2026, 2, 2)
REACT_DAY = date(2026, 2, 3)


def _amc_minute_bars(
    *,
    pre_close: float = 100.0,
    after_hours: float = 104.0,
    open_price: float = 102.0,
    at_5m: float = 103.0,
    at_30m: float = 105.0,
    at_60m: float = 106.0,
) -> list[tuple[datetime, float]]:
    """A minimal AMC tape: one after-hours print, then the next morning.

    Deliberately SPARSE rather than a full 390-bar session — the anchor rules
    care about which bar is at or before each mark, and a hand-listed tape
    makes the expected answer readable. Every price is distinct so no two
    windows can be confused for one another.
    """
    return [
        # release day, one regular bar and one after-hours print
        (_et_on(RELEASE_DAY, 15, 59), pre_close),
        (_et_on(RELEASE_DAY, 17, 0), after_hours),
        # reaction morning
        (_et_on(REACT_DAY, 9, 30), open_price),
        (_et_on(REACT_DAY, 9, 35), at_5m),
        (_et_on(REACT_DAY, 10, 0), at_30m),
        (_et_on(REACT_DAY, 10, 30), at_60m),
    ]


async def _amc_fixture(ticker: str = "ACME") -> dict:
    """One past AMC earnings print with BOTH series seeded.

    The daily closes are flat at 100.0 through the release day so
    ``pre_event_close`` is exactly 100.0 and every minute-bar move below is
    ``price/100 - 1`` — arithmetic the test states rather than computes.
    """
    days = _weekdays(date(2026, 1, 5), 40)
    closes = [100.0] * len(days)
    release_index = days.index(RELEASE_DAY)
    # After the print the stock steps to 105 and stays there, so the daily 1D
    # return (+5%) differs from every intraday window and cannot be confused
    # with one.
    for i in range(release_index + 1, len(days)):
        closes[i] = 105.0
    await _seed_daily(ticker, start=date(2026, 1, 5), closes=closes)
    await _seed_daily("SPY", start=date(2026, 1, 5), closes=[200.0] * len(days))
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{RELEASE_DAY.isoformat()}",
        ticker=ticker,
        when=_et(2026, 2, 2, 16, 30),
        session=EventSession.AFTER_MARKET,
    )
    await _seed_minutes(ticker, _amc_minute_bars())
    return {"event_id": event_id, "ticker": ticker, "days": days}


class ExplodingProvider:
    """A provider that fails the test if any bar method is called.

    Used to prove the GET routes fetch NOTHING. Asserting "the response was
    fast" or "the row count did not change" would both pass an implementation
    that fetched and discarded; only refusing to serve does.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_intraday_bars(self, *args, **kwargs):  # pragma: no cover — must not run
        self.calls.append("intraday")
        raise AssertionError("a GET must never fetch minute bars")

    def get_daily_bars(self, *args, **kwargs):  # pragma: no cover
        self.calls.append("daily")
        raise AssertionError("a GET must never fetch daily bars in these tests")


class RecordingProvider:
    """Serves a fixed minute tape and records every window it was asked for."""

    def __init__(self, bars: list[IntradayBar] | None = None, *, raises=None):
        self.bars = list(bars or [])
        self.raises = raises
        self.windows: list[tuple[str, datetime, datetime]] = []

    def get_intraday_bars(self, symbol, start, end, *, timeframe="1Min"):
        self.windows.append((symbol, start, end))
        if self.raises is not None:
            raise self.raises
        return [b for b in self.bars if start <= b.ts <= end]


def _intraday(ts: datetime, close: float, volume: int = 500) -> IntradayBar:
    return IntradayBar(ts=ts, open=close, high=close, low=close, close=close, volume=volume)


@pytest.fixture(autouse=True)
def _clear_backfill_throttle():
    """The attempt throttle is PROCESS-LOCAL, so it leaks across tests.

    One test's failed backfill would otherwise silently suppress the next
    test's fetch for six hours of wall clock — a cross-test dependency that
    shows up as an inexplicable ``bars: 0``.
    """
    er._backfill_attempts.clear()
    yield
    er._backfill_attempts.clear()


# ---------------------------------------------------------------------------
# 1. GET /replay — the §20 bundle
# ---------------------------------------------------------------------------


async def test_replay_returns_the_four_section_20_blocks_in_order(client):
    """§20 is an ORDER, not a bag of keys: what was knowable before, the
    release, the immediate reaction, the subsequent one."""
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    keys = list(body)
    for name in (
        "information_before",
        "release",
        "immediate_reaction",
        "subsequent_reaction",
    ):
        assert name in body, name
    assert keys.index("information_before") < keys.index("release")
    assert keys.index("release") < keys.index("immediate_reaction")
    assert keys.index("immediate_reaction") < keys.index("subsequent_reaction")


async def test_replay_release_block_carries_both_clocks_and_the_source(client):
    """§10: the instant AND the ET wall clock. A UI showing only UTC would
    print "21:30" for a 16:30 ET release and read as after-hours-plus-five."""
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    release = r.json()["release"]
    assert release["timestamp_utc"] == _et(2026, 2, 2, 16, 30).isoformat()
    assert release["timestamp_et"].startswith("2026-02-02T16:30")
    assert release["session"] == EventSession.AFTER_MARKET.value
    assert release["source_name"] == "sec_edgar"
    assert release["source_url"] == "https://example.invalid/ir"


async def test_replay_measures_the_amc_gap_and_windows_from_the_seeded_tape(client):
    """The anchor rule, end to end, on numbers stated in the fixture.

    pre_event_close is 100.0 (the flat daily series), the next open is 102.0
    and the +5/+30/+60 prints are 103/105/106 — so the gap is +2% and the
    windows are +3%, +5% and +6%. An implementation that anchored on the
    after-hours print (104) or on the open instead of the pre-close would miss
    every one of these.
    """
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    ir = r.json()["immediate_reaction"]
    assert ir["available"] is True
    assert ir["basis"] == "after_market_next_open_anchor"
    assert ir["confidence"] == "high"
    assert ir["pre_event_close"] == pytest.approx(100.0)
    assert ir["after_hours_move"] == pytest.approx(0.04)
    assert ir["open_price"] == pytest.approx(102.0)
    assert ir["gap_at_open"] == pytest.approx(0.02)
    assert ir["windows"]["5m"]["move"] == pytest.approx(0.03)
    assert ir["windows"]["30m"]["move"] == pytest.approx(0.05)
    assert ir["windows"]["60m"]["move"] == pytest.approx(0.06)


async def test_replay_carries_the_daily_subsequent_reaction_beside_the_intraday(client):
    """§20's fourth block: the daily 1D/3D/5D/10D and the SPY overlay.

    Pinned in the SAME test as the intraday numbers above so a reader can see
    they are different measurements of the same event — the daily 1D is +5%
    (100 -> 105 close to close), which is none of the intraday windows.
    """
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    sub = r.json()["subsequent_reaction"]
    assert sub["available"] is True
    assert sub["reaction"]["returns"]["1D"] == pytest.approx(0.05)
    assert sub["provenance"] == "QUANT"
    # SPY is flat, so the abnormal 1D equals the raw 1D.
    assert sub["abnormal"]["abnormal"]["1D"] == pytest.approx(0.05)


async def test_replay_labels_provenance_data_for_bars_and_quant_for_metrics(client):
    """§91: a price is DATA, arithmetic over it is QUANT. Un-labelled numbers
    are the failure this project exists to avoid."""
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    body = r.json()
    assert body["provenance"]["minute_bars"] == "DATA"
    assert body["provenance"]["daily_bars"] == "DATA"
    assert body["provenance"]["metrics"] == "QUANT"
    assert body["immediate_reaction"]["provenance"] == "QUANT"


async def test_replay_information_before_references_the_sibling_endpoints(client):
    """§20's first block is REFERENCES, not copies — and each carries the same
    ``as_of``, so following one cannot answer a different question."""
    fx = await _amc_fixture()
    as_of = "2026-03-01T00:00:00Z"
    r = await client.get(f"/api/events/{fx['event_id']}/replay", params={"as_of": as_of})
    info = r.json()["information_before"]
    assert info["price_context"]["endpoint"] == (
        f"/api/events/{fx['event_id']}/price-context"
    )
    assert info["fundamentals"]["endpoint"] == f"/api/events/{fx['event_id']}/fundamentals"
    assert info["price_context"]["as_of"] == info["fundamentals"]["as_of"]
    # Phase D has not landed, so the news block is an explicit absence.
    assert info["news_window"]["available"] is False
    assert "Phase D" in info["news_window"]["reason"]


async def test_replay_of_a_future_event_is_200_with_a_reason_not_a_404(client):
    """The Catalyst page opens on UPCOMING earnings. "It has not happened yet"
    is an answer; a 404 would hide the card the user just clicked."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-12-01",
        ticker="ACME",
        when=_et(2026, 12, 1, 16, 30),
    )
    r = await client.get(
        f"/api/events/{event_id}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "has not occurred" in body["reason"]
    # The registry facts travel WITH the refusal.
    assert body["event"]["event_key"] == "EARNINGS:ACME:2026-12-01"
    assert body["event"]["ticker"] == "ACME"
    assert body["event"]["session"] == EventSession.AFTER_MARKET.value


async def test_an_event_that_has_occurred_by_a_later_as_of_replays(client):
    """The paired half of the test above: the SAME event resolves once
    ``as_of`` passes it. A gate that always refused would pass only the first."""
    fx = await _amc_fixture()
    early = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-02-01T00:00:00Z"}
    )
    late = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert early.json()["available"] is False
    assert late.json()["available"] is True


async def test_replay_of_a_macro_event_says_no_ticker(client):
    """A CPI print moves an index, not a single name (§39 / Phase G)."""
    event_id = await _add_event(
        key="CPI:2026-01",
        ticker=None,
        when=_et(2026, 1, 13, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
        title="CPI",
    )
    r = await client.get(
        f"/api/events/{event_id}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert r.status_code == 200
    assert r.json() == {**r.json(), "available": False, "reason": "no_ticker"}


async def test_replay_without_stored_minutes_still_answers_the_daily_half(client):
    """The normal state of a fresh install: no minute bars, complete daily
    reaction, and a reason that names the backfill route.

    This is the shape the UI keys on to decide whether to offer the "Load
    minute bars" button, so the reason string is part of the contract.
    """
    days = _weekdays(date(2026, 1, 5), 40)
    closes = [100.0] * len(days)
    for i in range(days.index(RELEASE_DAY) + 1, len(days)):
        closes[i] = 105.0
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=closes)
    await _seed_daily("SPY", start=date(2026, 1, 5), closes=[200.0] * len(days))
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    body = r.json()
    assert body["available"] is True
    assert body["immediate_reaction"]["available"] is False
    assert "backfill" in body["immediate_reaction"]["reason"]
    assert body["subsequent_reaction"]["reaction"]["returns"]["1D"] == pytest.approx(0.05)


async def test_replay_404s_only_for_a_missing_event(client):
    r = await client.get("/api/events/999999/replay")
    assert r.status_code == 404


async def test_replay_rejects_a_future_as_of_with_422(client):
    """A request for prices that do not exist yet is a mistake worth
    reporting, not a request to silently clamp to now."""
    fx = await _amc_fixture()
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    r = await client.get(f"/api/events/{fx['event_id']}/replay", params={"as_of": future})
    assert r.status_code == 422
    assert "future" in str(r.json()["detail"])


async def test_replay_rejects_a_malformed_as_of_with_422(client):
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "not-a-date"}
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 2. The as-of gate on MINUTE bars (§14, §96) — always in pairs
# ---------------------------------------------------------------------------


async def test_minute_bars_after_as_of_are_invisible_to_the_replay(client):
    """PAIR 1 of 2. At 09:44 ET the 10:00 bar has not happened.

    The +30m mark is NOT null — the library's rule is "the last bar at or
    before the mark", so it falls back to the 09:35 print — but it must be
    measured off that OLDER bar and must SAY SO via ``lag_seconds``. That is
    the honest shape: "+30m, from a bar 25 minutes stale" is a weaker claim
    than "+30m", and §85 requires the payload to carry the difference. What
    the gate must guarantee is that the 10:00 price is nowhere in the answer.
    """
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay",
        params={"as_of": _et_on(REACT_DAY, 9, 44).isoformat()},
    )
    ir = r.json()["immediate_reaction"]
    assert ir["available"] is True
    assert ir["windows"]["5m"]["move"] == pytest.approx(0.03)
    assert ir["windows"]["5m"]["lag_seconds"] == 0
    # The +30m mark is filled from the 09:35 bar, NOT from the future 10:00
    # one — so it reports the 5-minute price and 25 minutes of staleness.
    assert ir["windows"]["30m"]["bar_ts_utc"] == _et_on(REACT_DAY, 9, 35).isoformat()
    assert ir["windows"]["30m"]["move"] == pytest.approx(0.03)
    assert ir["windows"]["30m"]["lag_seconds"] == 25 * 60
    # THE GATE: the 10:00 close (+5%) has not leaked into any window.
    assert all(
        w["move"] != pytest.approx(0.05) for w in ir["windows"].values()
    ), "a bar after as_of must not reach any window"


async def test_the_same_bar_is_visible_once_as_of_passes_it(client):
    """PAIR 2 of 2. At 10:05 ET the 10:00 bar exists and +30m resolves."""
    fx = await _amc_fixture()
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay",
        params={"as_of": _et_on(REACT_DAY, 10, 5).isoformat()},
    )
    ir = r.json()["immediate_reaction"]
    # Now the 10:00 bar IS knowable, so +30m is measured off it with no lag.
    assert ir["windows"]["30m"]["bar_ts_utc"] == _et_on(REACT_DAY, 10, 0).isoformat()
    assert ir["windows"]["30m"]["move"] == pytest.approx(0.05)
    assert ir["windows"]["30m"]["lag_seconds"] == 0
    # ...and the 10:30 bar still has not happened, so +60m falls back to
    # 10:00 and says how stale it is.
    assert ir["windows"]["60m"]["bar_ts_utc"] == _et_on(REACT_DAY, 10, 0).isoformat()
    assert ir["windows"]["60m"]["lag_seconds"] == 30 * 60


async def test_the_as_of_gate_reports_how_many_minute_bars_it_could_see(client):
    """``minute_bars_stored`` is the count AFTER the gate, so the freshness
    block describes the evidence the answer actually rests on."""
    fx = await _amc_fixture()
    early = await client.get(
        f"/api/events/{fx['event_id']}/replay",
        params={"as_of": _et_on(REACT_DAY, 9, 44).isoformat()},
    )
    late = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert early.json()["data_freshness"]["minute_bars_stored"] == 4
    assert late.json()["data_freshness"]["minute_bars_stored"] == 6


async def test_the_history_table_gates_its_minute_bars_at_as_of_too(client):
    """The contract requires the gate in BOTH endpoints; a table that showed
    an intraday_30m the replay refused would leak the future through the
    other tab."""
    fx = await _amc_fixture()
    later_id = await _add_event(
        key="EARNINGS:ACME:2026-05-04",
        ticker="ACME",
        when=_et(2026, 5, 4, 16, 30),
    )
    early = await client.get(
        f"/api/events/{later_id}/history",
        params={"as_of": _et_on(REACT_DAY, 9, 44).isoformat()},
    )
    late = await client.get(
        f"/api/events/{later_id}/history", params={"as_of": "2026-06-01T00:00:00Z"}
    )
    early_row = next(r for r in early.json()["rows"] if r["event_id"] == fx["event_id"])
    late_row = next(r for r in late.json()["rows"] if r["event_id"] == fx["event_id"])
    # At 09:44 the 10:00 bar does not exist, so the table's +30m cell falls
    # back to the 09:35 print (+3%) exactly as the replay's does — the two
    # views must agree, and neither may show the 10:00 price.
    assert early_row["intraday_30m"]["move"] == pytest.approx(0.03)
    # Once as_of passes 10:00 the same cell becomes the real +30m (+5%).
    assert late_row["intraday_30m"]["available"] is True
    assert late_row["intraday_30m"]["move"] == pytest.approx(0.05)
    assert late_row["intraday_30m"]["bar_ts_utc"] == _et_on(REACT_DAY, 10, 0).isoformat()


# ---------------------------------------------------------------------------
# 3. A GET never fetches (the contract's performance property)
# ---------------------------------------------------------------------------


async def test_get_replay_never_calls_the_provider(client, monkeypatch):
    """Proven by a provider that RAISES, not by counting rows: an
    implementation that fetched and discarded would pass a row count."""
    fx = await _amc_fixture()
    exploding = ExplodingProvider()
    monkeypatch.setattr(er, "get_provider", lambda name: exploding)
    r = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert r.status_code == 200
    assert r.json()["immediate_reaction"]["available"] is True
    assert exploding.calls == []


async def test_get_history_never_calls_the_provider_even_with_twelve_events(
    client, monkeypatch
):
    """THE reason the backfill is a POST. Twelve past prints on one page must
    not become twelve paginated provider fetches."""
    await _seed_daily("ACME", start=HISTORY_START, closes=[100.0] * 200)
    await _seed_daily("SPY", start=HISTORY_START, closes=[200.0] * 200)
    days = _weekdays(HISTORY_START, 200)
    for i in range(12):
        await _add_event(
            key=f"EARNINGS:ACME:past-{i}",
            ticker="ACME",
            when=_et_on(days[5 + i * 5], 16, 30),
        )
    subject = await _add_event(
        key="EARNINGS:ACME:subject", ticker="ACME", when=_et_on(days[150], 16, 30)
    )
    exploding = ExplodingProvider()
    monkeypatch.setattr(er, "get_provider", lambda name: exploding)
    r = await client.get(
        f"/api/events/{subject}/history", params={"as_of": HISTORY_AS_OF}
    )
    assert r.status_code == 200
    assert r.json()["n_rows"] == 12
    assert exploding.calls == []


# ---------------------------------------------------------------------------
# 4. POST /replay/backfill — the only writer of stock_bars_1m
# ---------------------------------------------------------------------------


async def _stored_minutes(ticker: str) -> list[StockBar1mRow]:
    async with SessionLocal() as s:
        rows = (
            await s.execute(
                StockBar1mRow.__table__.select().order_by(StockBar1mRow.ts)
            )
        ).all()
    return [r for r in rows if r.ticker == ticker]


async def test_backfill_fetches_the_window_and_stores_it_with_an_audit_row(
    client, monkeypatch
):
    """Rule 12 / ADR-003: the bars and their DATA_BACKFILL audit land in ONE
    transaction, and the audit says WHICH event's window it was."""
    days = _weekdays(date(2026, 1, 5), 40)
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * len(days))
    await _seed_daily("SPY", start=date(2026, 1, 5), closes=[200.0] * len(days))
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    provider = RecordingProvider(
        [_intraday(ts, close) for ts, close in _amc_minute_bars()]
    )
    monkeypatch.setattr(er, "get_provider", lambda name: provider)

    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.status_code == 200
    body = r.json()
    assert body["fetched"] is True
    assert body["bars"] == 6
    assert body["ticker"] == "ACME"

    assert len(await _stored_minutes("ACME")) == 6

    async with SessionLocal() as s:
        audits = (await s.execute(AuditEvent.__table__.select())).all()
    backfills = [
        a
        for a in audits
        if a.action == AuditAction.DATA_BACKFILL.value
        and (a.details or {}).get("kind") == "intraday_event_window"
    ]
    assert len(backfills) == 1
    assert backfills[0].details["event_key"] == "EARNINGS:ACME:2026-02-02"
    assert backfills[0].details["ticker"] == "ACME"
    assert backfills[0].details["bars"] == 6
    assert backfills[0].entity_type == "stock_bars_1m"


async def test_a_second_backfill_of_the_same_window_stores_nothing(client, monkeypatch):
    """An event window is a CLOSED interval in the past — refetching it can
    only rewrite minutes that cannot have changed."""
    fx = await _amc_fixture()
    provider = RecordingProvider(
        [_intraday(ts, close) for ts, close in _amc_minute_bars()]
    )
    monkeypatch.setattr(er, "get_provider", lambda name: provider)

    r = await client.post(f"/api/events/{fx['event_id']}/replay/backfill")
    body = r.json()
    assert body["fetched"] is False
    assert "already stored" in body["reason"]
    assert provider.windows == [], "a stored window must not be refetched"
    assert len(await _stored_minutes("ACME")) == 6


async def test_backfill_asks_for_the_amc_two_day_window(client, monkeypatch):
    """An AMC release's reaction is TOMORROW's open, so the window must reach
    20:00 ET on the next session — a same-day window would have no open to
    measure and every intraday number would be an honest null."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{event_id}/replay/backfill")

    (symbol, start, end) = provider.windows[0]
    assert symbol == "ACME"
    assert start == _et_on(RELEASE_DAY, 4, 0)
    assert end == _et_on(REACT_DAY, 20, 0)


async def test_backfill_asks_for_a_one_day_window_for_a_bmo_release(client, monkeypatch):
    """A BMO print is fully answered by its OWN morning; fetching a second
    day would be a thousand bars nobody reads."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02-bmo",
        ticker="ACME",
        when=_et(2026, 2, 2, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{event_id}/replay/backfill")

    (_symbol, start, end) = provider.windows[0]
    assert start == _et_on(RELEASE_DAY, 4, 0)
    assert end == _et_on(RELEASE_DAY, 20, 0)


async def test_the_window_always_opens_at_04_00_et_so_premarket_is_covered(
    client, monkeypatch
):
    """A BMO release at 07:00 ET moves the PRE-MARKET tape. A window opening
    at 09:30 would measure the reaction from a price that already contained
    it."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:bmo-premarket",
        ticker="ACME",
        when=_et(2026, 2, 2, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{event_id}/replay/backfill")
    (_s, start, _e) = provider.windows[0]
    assert start.astimezone(EASTERN).hour == 4
    assert start < _et(2026, 2, 2, 7, 0)


async def test_the_amc_window_uses_the_stored_market_calendar_over_a_weekday_guess(
    client, monkeypatch
):
    """THE REASON market_calendar EXISTS. A Friday AMC print reacts on Monday
    — but after a stacked holiday it reacts on TUESDAY, and only the stored
    session grid knows that. The basis string says which rule was used, so a
    fallback window can never be mistaken for a calendar-backed one.
    """
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    friday = date(2026, 2, 6)
    tuesday = date(2026, 2, 10)  # Monday the 9th is a holiday in this fixture
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=tuesday,
                exchange="US",
                open_utc=_et_on(tuesday, 9, 30),
                close_utc=_et_on(tuesday, 16, 0),
                source="test",
            )
        )
        await s.commit()
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-06", ticker="ACME", when=_et_on(friday, 16, 30)
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(f"/api/events/{event_id}/replay/backfill")

    (_s, _start, end) = provider.windows[0]
    assert end == _et_on(tuesday, 20, 0), "the holiday Monday must be skipped"
    assert r.json()["window_basis"] == "spans_next_session:market_calendar"


async def test_without_a_stored_calendar_the_window_names_its_weekday_fallback(
    client, monkeypatch
):
    """The paired half: with no session grid the window is the next WEEKDAY
    and SAYS SO. A named approximation of which minutes to FETCH is honest; an
    unnamed one would let a holiday-shifted window read as a measurement."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-06", ticker="ACME", when=_et_on(date(2026, 2, 6), 16, 30)
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(f"/api/events/{event_id}/replay/backfill")

    assert r.json()["window_basis"] == "spans_next_session:next_weekday_fallback"
    (_s, _start, end) = provider.windows[0]
    assert end == _et_on(date(2026, 2, 9), 20, 0)  # the Monday


async def test_backfill_drops_provider_bars_outside_the_requested_window(
    client, monkeypatch
):
    """A paginating provider may overshoot. A bar outside this event's window
    is not this window's to store — it would be counted as evidence for a
    reaction it has nothing to do with."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    good = [_intraday(ts, close) for ts, close in _amc_minute_bars()]
    stray = [
        _intraday(_et_on(date(2026, 1, 30), 10, 0), 99.0),  # before the window
        _intraday(_et_on(date(2026, 2, 5), 10, 0), 99.0),  # after it
    ]
    provider = RecordingProvider(good + stray)
    # The RecordingProvider filters by window itself; bypass that so the stray
    # bars really reach the seam under test.
    provider.get_intraday_bars = lambda *a, **k: good + stray  # type: ignore[assignment]
    monkeypatch.setattr(er, "get_provider", lambda name: provider)

    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.json()["bars"] == 6
    assert len(await _stored_minutes("ACME")) == 6


async def test_backfill_deduplicates_a_timestamp_the_provider_served_twice(
    client, monkeypatch
):
    """The PK is (ticker, ts). A duplicated instant must be dropped BEFORE the
    insert, or the whole transaction dies on a constraint violation and the
    user's button press 500s."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    bars = [_intraday(ts, close) for ts, close in _amc_minute_bars()]
    provider = RecordingProvider([])
    provider.get_intraday_bars = lambda *a, **k: bars + bars  # type: ignore[assignment]
    monkeypatch.setattr(er, "get_provider", lambda name: provider)

    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.status_code == 200
    assert r.json()["bars"] == 6
    assert len(await _stored_minutes("ACME")) == 6


async def test_backfill_with_an_unconfigured_provider_is_200_with_a_reason(
    unconfigured_client,
):
    """A button press must report why nothing arrived, not 5xx."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    r = await unconfigured_client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.status_code == 200
    body = r.json()
    assert body["fetched"] is False
    assert body["bars"] == 0
    assert body["reason"]
    assert await _stored_minutes("ACME") == []


async def test_backfill_translates_a_403_capability_refusal_into_a_reason(
    client, monkeypatch
):
    """A plan without minute data is a named refusal (§16 capability), not an
    outage and not an empty success."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    provider = RecordingProvider(
        raises=CapabilityNotAvailable("minute bars not in subscription (403)")
    )
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.status_code == 200
    assert r.json()["fetched"] is False
    assert "403" in r.json()["reason"]


async def test_an_empty_provider_window_is_a_named_absence_and_writes_no_audit(
    client, monkeypatch
):
    """A holiday, a halted symbol or a range before the symbol listed. Nothing
    was written, so there is nothing to audit — an audit trail of no-ops is
    noise that hides the real backfills."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    monkeypatch.setattr(er, "get_provider", lambda name: RecordingProvider([]))
    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.json()["fetched"] is True
    assert r.json()["bars"] == 0
    assert "no minute bars" in r.json()["reason"]
    async with SessionLocal() as s:
        audits = (await s.execute(AuditEvent.__table__.select())).all()
    assert [a for a in audits if a.action == AuditAction.DATA_BACKFILL.value] == []


async def test_a_repeated_failing_backfill_is_throttled_per_event(client, monkeypatch):
    """An event window that came back empty will keep coming back empty — the
    interval is closed and in the past. Re-asking is cost with no possible new
    answer."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{event_id}/replay/backfill")
    second = await client.post(f"/api/events/{event_id}/replay/backfill")

    assert len(provider.windows) == 1, "the second attempt must not reach the provider"
    assert "recently attempted" in second.json()["reason"]


async def test_the_throttle_is_per_event_not_per_ticker(client, monkeypatch):
    """Two prints of the SAME ticker are two different windows. A per-ticker
    throttle would silently deny the second event its bars."""
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 60)
    first = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    second = await _add_event(
        key="EARNINGS:ACME:2026-02-17", ticker="ACME", when=_et(2026, 2, 17, 16, 30)
    )
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{first}/replay/backfill")
    await client.post(f"/api/events/{second}/replay/backfill")
    assert len(provider.windows) == 2


async def test_backfill_of_a_macro_event_says_no_ticker(client):
    event_id = await _add_event(
        key="CPI:2026-01",
        ticker=None,
        when=_et(2026, 1, 13, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
        title="CPI",
    )
    r = await client.post(f"/api/events/{event_id}/replay/backfill")
    assert r.status_code == 200
    assert r.json()["reason"] == "no_ticker"
    assert r.json()["fetched"] is False


async def test_backfill_404s_only_for_a_missing_event(client):
    r = await client.post("/api/events/999999/replay/backfill")
    assert r.status_code == 404


async def test_backfilled_bars_are_immediately_visible_to_the_replay(client, monkeypatch):
    """The round trip the UI performs: press "Load minute bars", refetch, see
    numbers. If the write and the read disagreed about the window, the button
    would appear to do nothing."""
    days = _weekdays(date(2026, 1, 5), 40)
    closes = [100.0] * len(days)
    for i in range(days.index(RELEASE_DAY) + 1, len(days)):
        closes[i] = 105.0
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=closes)
    await _seed_daily("SPY", start=date(2026, 1, 5), closes=[200.0] * len(days))
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-02", ticker="ACME", when=_et(2026, 2, 2, 16, 30)
    )
    before = await client.get(
        f"/api/events/{event_id}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert before.json()["immediate_reaction"]["available"] is False

    provider = RecordingProvider(
        [_intraday(ts, close) for ts, close in _amc_minute_bars()]
    )
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    await client.post(f"/api/events/{event_id}/replay/backfill")

    after = await client.get(
        f"/api/events/{event_id}/replay", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    ir = after.json()["immediate_reaction"]
    assert ir["available"] is True
    assert ir["gap_at_open"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 5. GET /history — the §60 table
# ---------------------------------------------------------------------------


async def _history_fixture(n_past: int = 6) -> dict:
    """``n_past`` past AMC prints for one ticker plus an upcoming subject."""
    days = _weekdays(HISTORY_START, 200)
    await _seed_daily("ACME", start=HISTORY_START, closes=[100.0] * 200)
    await _seed_daily("SPY", start=HISTORY_START, closes=[200.0] * 200)
    past_ids = []
    for i in range(n_past):
        past_ids.append(
            await _add_event(
                key=f"EARNINGS:ACME:past-{i}",
                ticker="ACME",
                when=_et_on(days[5 + i * 10], 16, 30),
            )
        )
    subject = await _add_event(
        key="EARNINGS:ACME:subject", ticker="ACME", when=_et_on(days[150], 16, 30)
    )
    return {"subject": subject, "past_ids": past_ids, "days": days}


async def test_history_returns_one_row_per_past_comparable_print(client):
    fx = await _history_fixture(6)
    r = await client.get(
        f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n_rows"] == 6
    assert [row["event_id"] for row in body["rows"]] == fx["past_ids"]


async def test_history_carries_the_section_60_column_order(client):
    """The UI renders the columns the payload names, so the ORDER lives here
    rather than being hardcoded client-side where it could drift."""
    fx = await _history_fixture(3)
    r = await client.get(
        f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF}
    )
    columns = r.json()["columns"]
    for name in (
        "date_et",
        "session",
        "status",
        "eps_surprise",
        "rev_surprise",
        "implied_move",
        "actual_move_abs",
        "gap",
        "intraday_30m",
        "ret_1d",
        "ret_5d",
        "abnormal_1d",
    ):
        assert name in columns, name
    assert columns.index("eps_surprise") < columns.index("gap")


async def test_history_marks_surprise_and_implied_move_unavailable_not_absent(client):
    """§33/§98 and §36/§60. A table that simply LACKED these columns would
    read as "we did not think surprise mattered"; ``available: false`` with a
    reason says the platform knows it is missing and why."""
    fx = await _history_fixture(2)
    r = await client.get(
        f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF}
    )
    row = r.json()["rows"][0]
    assert row["eps_surprise"] == {"available": False, "reason": "CONSENSUS DATA UNAVAILABLE"}
    assert row["rev_surprise"]["available"] is False
    assert row["implied_move"]["available"] is False
    assert "Phase I" in row["implied_move"]["reason"]
    assert set(r.json()["not_backtestable"]) >= {
        "eps_surprise",
        "rev_surprise",
        "implied_move",
    }


async def test_history_intraday_30m_is_an_honest_absence_without_stored_bars(client):
    """Minute bars are backfilled ONE EVENT AT A TIME on user action, so most
    rows of a freshly loaded table legitimately have none."""
    fx = await _history_fixture(3)
    r = await client.get(
        f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF}
    )
    for row in r.json()["rows"]:
        assert row["intraday_30m"]["available"] is False
        assert row["intraday_30m"]["reason"]


async def test_history_shows_intraday_30m_for_the_one_event_that_has_bars(client):
    """The mixed state the §60 table is normally in: eleven honest absences
    and one measured number, side by side and distinguishable."""
    fx = await _amc_fixture()
    subject = await _add_event(
        key="EARNINGS:ACME:2026-05-04", ticker="ACME", when=_et(2026, 5, 4, 16, 30)
    )
    other = await _add_event(
        key="EARNINGS:ACME:2026-01-12", ticker="ACME", when=_et(2026, 1, 12, 16, 30)
    )
    r = await client.get(
        f"/api/events/{subject}/history", params={"as_of": "2026-06-01T00:00:00Z"}
    )
    rows = {row["event_id"]: row for row in r.json()["rows"]}
    assert rows[fx["event_id"]]["intraday_30m"]["move"] == pytest.approx(0.05)
    assert rows[fx["event_id"]]["intraday_30m"]["basis"] == "after_market_next_open_anchor"
    assert rows[other]["intraday_30m"]["available"] is False


async def test_history_last_trims_to_the_newest_n_rows(client):
    """The §60 4/8/12 toggle. Trimming to the NEWEST is what makes "last 4"
    mean the four most recent prints rather than the four oldest."""
    fx = await _history_fixture(10)
    r = await client.get(
        f"/api/events/{fx['subject']}/history",
        params={"as_of": HISTORY_AS_OF, "last": 4},
    )
    body = r.json()
    assert body["n_rows"] == 4
    assert [row["event_id"] for row in body["rows"]] == fx["past_ids"][-4:]


async def test_history_summary_describes_exactly_the_rows_shown(client):
    """§19/§64: the distribution must be OF the visible table. A summary over
    ten prints printed above four rows is a different claim than the one the
    reader is checking."""
    fx = await _history_fixture(10)
    r = await client.get(
        f"/api/events/{fx['subject']}/history",
        params={"as_of": HISTORY_AS_OF, "last": 4},
    )
    body = r.json()
    stats = body["summary"]["1D"]
    # Every window's n_available is bounded by the rows actually rendered.
    for window in stats.values():
        assert window["n_available"] <= body["n_rows"]


async def test_history_rejects_a_last_above_the_section_60_ceiling(client):
    """Twelve is the largest §60 toggle. Bounded at the boundary so no query
    string can ask for more windows than the UI can display."""
    fx = await _history_fixture(3)
    r = await client.get(
        f"/api/events/{fx['subject']}/history",
        params={"as_of": HISTORY_AS_OF, "last": 99},
    )
    assert r.status_code == 422


async def test_history_excludes_an_estimated_past_date(client):
    """§15: an ESTIMATED past date is a DERIVATION. Measuring a reaction
    around a day nobody reported on would be a fabricated number wearing a
    measurement's clothes."""
    days = _weekdays(HISTORY_START, 200)
    await _seed_daily("ACME", start=HISTORY_START, closes=[100.0] * 200)
    await _seed_daily("SPY", start=HISTORY_START, closes=[200.0] * 200)
    confirmed = await _add_event(
        key="EARNINGS:ACME:confirmed", ticker="ACME", when=_et_on(days[10], 16, 30)
    )
    await _add_event(
        key="EARNINGS:ACME:estimated",
        ticker="ACME",
        when=_et_on(days[20], 16, 30),
        status=EventStatus.ESTIMATED,
    )
    subject = await _add_event(
        key="EARNINGS:ACME:subject", ticker="ACME", when=_et_on(days[100], 16, 30)
    )
    r = await client.get(
        f"/api/events/{subject}/history", params={"as_of": HISTORY_AS_OF}
    )
    assert [row["event_id"] for row in r.json()["rows"]] == [confirmed]


async def test_history_excludes_a_print_not_yet_knowable_at_as_of(client):
    """A print scheduled after ``as_of`` is not history yet, however firmly
    the registry knows about it today."""
    days = _weekdays(HISTORY_START, 200)
    await _seed_daily("ACME", start=HISTORY_START, closes=[100.0] * 200)
    await _seed_daily("SPY", start=HISTORY_START, closes=[200.0] * 200)
    early = await _add_event(
        key="EARNINGS:ACME:early", ticker="ACME", when=_et_on(days[10], 16, 30)
    )
    later = await _add_event(
        key="EARNINGS:ACME:later", ticker="ACME", when=_et_on(days[60], 16, 30)
    )
    subject = await _add_event(
        key="EARNINGS:ACME:subject", ticker="ACME", when=_et_on(days[150], 16, 30)
    )
    cut = _et_on(days[30], 16, 0).isoformat()
    narrow = await client.get(f"/api/events/{subject}/history", params={"as_of": cut})
    wide = await client.get(
        f"/api/events/{subject}/history", params={"as_of": HISTORY_AS_OF}
    )
    assert [row["event_id"] for row in narrow.json()["rows"]] == [early]
    assert [row["event_id"] for row in wide.json()["rows"]] == [early, later]


async def test_history_of_a_macro_event_says_no_ticker(client):
    event_id = await _add_event(
        key="CPI:2026-01",
        ticker=None,
        when=_et(2026, 1, 13, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
        title="CPI",
    )
    r = await client.get(
        f"/api/events/{event_id}/history", params={"as_of": "2026-03-01T00:00:00Z"}
    )
    assert r.status_code == 200
    assert r.json()["available"] is False
    assert r.json()["reason"] == "no_ticker"


async def test_history_stays_200_with_no_market_data_provider(unconfigured_client):
    """A 503 here would hide the registry facts, which are real rows and are
    not hidden because a quote vendor is unpaid."""
    days = _weekdays(HISTORY_START, 200)
    await _add_event(
        key="EARNINGS:ACME:past", ticker="ACME", when=_et_on(days[10], 16, 30)
    )
    subject = await _add_event(
        key="EARNINGS:ACME:subject", ticker="ACME", when=_et_on(days[100], 16, 30)
    )
    r = await unconfigured_client.get(
        f"/api/events/{subject}/history", params={"as_of": HISTORY_AS_OF}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n_rows"] == 1
    assert body["unavailable"], "an unavailable bar source must be named"
    assert body["rows"][0]["bars_available"] is False


async def test_history_404s_only_for_a_missing_event(client):
    r = await client.get("/api/events/999999/history")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 6. POST /history/backfill — bounded, isolated, newest-first
# ---------------------------------------------------------------------------


async def test_history_backfill_defaults_to_the_last_four_events(client, monkeypatch):
    """The default matches the §60 table's smallest toggle, so the common
    button press is the cheap one."""
    fx = await _history_fixture(10)
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(
        f"/api/events/{fx['subject']}/history/backfill",
        params={"as_of": HISTORY_AS_OF},
    )
    body = r.json()
    assert body["last"] == 4
    assert body["events_attempted"] == 4
    assert len(provider.windows) == 4


async def test_history_backfill_fetches_the_newest_events_not_the_oldest(
    client, monkeypatch
):
    """"Backfill last 4" must fill the four rows the reader is looking at.
    Filling the four OLDEST would leave the visible table exactly as empty."""
    fx = await _history_fixture(10)
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(
        f"/api/events/{fx['subject']}/history/backfill",
        params={"as_of": HISTORY_AS_OF, "last": 3},
    )
    attempted = [item["event_id"] for item in r.json()["results"]]
    assert attempted == list(reversed(fx["past_ids"][-3:]))


async def test_history_backfill_rejects_a_last_above_twelve(client):
    """Bounded at the FastAPI boundary: twelve paginated provider fetches is
    already a lot, and no query string may ask for more."""
    fx = await _history_fixture(3)
    r = await client.post(
        f"/api/events/{fx['subject']}/history/backfill",
        params={"as_of": HISTORY_AS_OF, "last": 50},
    )
    assert r.status_code == 422


async def test_history_backfill_isolates_one_events_failure_from_the_others(
    client, monkeypatch
):
    """Per-item isolation, the same rule the calendar ingest applies to
    providers (§8): one window the vendor will not serve must not cost the
    other three theirs."""
    fx = await _history_fixture(4)
    failing_id = fx["past_ids"][-2]
    async with SessionLocal() as s:
        failing_row = await s.get(EventRow, failing_id)
        failing_day = failing_row.scheduled_at.date()

    class PartlyFailing:
        """Refuses exactly ONE event's window, by date.

        Keyed on the WINDOW rather than on a call counter: the seam builds a
        provider per event, so a counter would reset every time and never
        refuse anything — which is precisely how this test first passed
        vacuously.
        """

        def get_intraday_bars(self, symbol, start, end, *, timeframe="1Min"):
            if start.astimezone(EASTERN).date() == failing_day:
                raise CapabilityNotAvailable("no minute data for this range (403)")
            return [_intraday(start + timedelta(hours=6), 100.0)]

    monkeypatch.setattr(er, "get_provider", lambda name: PartlyFailing())
    r = await client.post(
        f"/api/events/{fx['subject']}/history/backfill",
        params={"as_of": HISTORY_AS_OF, "last": 4},
    )
    results = r.json()["results"]
    assert len(results) == 4
    failures = [item for item in results if not item["fetched"]]
    assert len(failures) == 1
    assert "403" in failures[0]["reason"]
    assert sum(item["bars"] for item in results) == 3
    assert failing_id in {item["event_id"] for item in results}


async def test_history_backfill_reports_every_windows_outcome(client, monkeypatch):
    """The UI needs to say which of the four filled. A single aggregate count
    cannot tell "three of four succeeded" from "one returned three bars"."""
    fx = await _history_fixture(3)
    provider = RecordingProvider([])
    monkeypatch.setattr(er, "get_provider", lambda name: provider)
    r = await client.post(
        f"/api/events/{fx['subject']}/history/backfill",
        params={"as_of": HISTORY_AS_OF, "last": 3},
    )
    body = r.json()
    assert body["events_available"] == 3
    assert len(body["results"]) == 3
    for item in body["results"]:
        assert "event_key" in item
        assert "reason" in item or item["fetched"] is True


async def test_history_backfill_404s_only_for_a_missing_event(client):
    r = await client.post("/api/events/999999/history/backfill")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Persistence details — the ORM mirror and the stored row
# ---------------------------------------------------------------------------


async def test_stored_minute_rows_round_trip_through_the_pure_value(client):
    """SQLite hands ``TIMESTAMPTZ`` back NAIVE and ``MinuteBar`` REFUSES a
    naive timestamp. Both rules are right; together they would silently drop
    every stored bar unless the seam re-stamps the instant. This test is the
    tripwire for that."""
    await _seed_minutes("ACME", _amc_minute_bars())
    async with SessionLocal() as s:
        rows = list(
            (
                await s.execute(
                    StockBar1mRow.__table__.select().order_by(StockBar1mRow.ts)
                )
            ).all()
        )
        bars = er.to_minute_bars(rows)
    assert len(bars) == 6
    assert all(bar.ts_utc.tzinfo is not None for bar in bars)
    assert bars == sorted(bars, key=lambda b: b.ts_utc)
    assert bars[0].ts_utc == _et_on(RELEASE_DAY, 15, 59)


async def test_the_minute_bar_primary_key_refuses_a_duplicate_instant(client):
    """(ticker, ts) is the PK the migration declares. Storing the same minute
    twice would make a window's bar count — and therefore its volume
    comparison — a function of how many times somebody pressed a button."""
    import sqlalchemy.exc

    await _seed_minutes("ACME", [(_et_on(RELEASE_DAY, 15, 59), 100.0)])
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        await _seed_minutes("ACME", [(_et_on(RELEASE_DAY, 15, 59), 101.0)])


async def test_the_same_instant_for_two_tickers_is_two_rows(client):
    """The paired half: the PK is COMPOSITE, so one minute of NVDA and the
    same minute of AMD coexist."""
    await _seed_minutes("ACME", [(_et_on(RELEASE_DAY, 15, 59), 100.0)])
    await _seed_minutes("OTHER", [(_et_on(RELEASE_DAY, 15, 59), 55.0)])
    assert len(await _stored_minutes("ACME")) == 1
    assert len(await _stored_minutes("OTHER")) == 1


async def test_minute_volume_is_stored_as_a_whole_share_count(client):
    """``stock_bars_1m.volume`` is BIGINT. A float here would be a different
    column than the migration declares, and the parity pin would not catch a
    value that merely round-trips."""
    await _seed_minutes("ACME", [(_et_on(RELEASE_DAY, 15, 59), 100.0)], volume=123_456)
    rows = await _stored_minutes("ACME")
    assert rows[0].volume == 123_456
    assert isinstance(rows[0].volume, int)


async def test_the_event_block_has_one_shape_across_every_branch(client):
    """§20's ``event`` ref is the SAME object on all three answers.

    The bundle's own ``to_dict`` emits a narrower ``event`` block than the
    seam's reference does, so a naive spread drops ``scheduled_at_utc`` from
    the AVAILABLE replay while the future-event refusal and the history table
    keep it. A UI reading that field would then work on an upcoming event and
    break on a past one — the worst kind of inconsistency, because it looks
    like a data problem rather than a shape problem.
    """
    fx = await _amc_fixture()
    future_id = await _add_event(
        key="EARNINGS:ACME:2026-12-01", ticker="ACME", when=_et(2026, 12, 1, 16, 30)
    )
    available = await client.get(
        f"/api/events/{fx['event_id']}/replay", params={"as_of": "2026-06-01T00:00:00Z"}
    )
    refused = await client.get(
        f"/api/events/{future_id}/replay", params={"as_of": "2026-06-01T00:00:00Z"}
    )
    history = await client.get(
        f"/api/events/{fx['event_id']}/history", params={"as_of": "2026-06-01T00:00:00Z"}
    )
    assert available.json()["available"] is True
    assert refused.json()["available"] is False
    shapes = [set(r.json()["event"]) for r in (available, refused, history)]
    assert shapes[0] == shapes[1] == shapes[2]
    assert "scheduled_at_utc" in shapes[0]
    assert "event_key" in shapes[0]


async def test_the_history_tables_calendar_lookup_is_one_query_not_one_per_row(client):
    """The §60 table resolves twelve windows against ONE calendar read.

    Each row needs the next trading day after its own release, and asking per
    row turned a twelve-row page into twelve identical range queries over the
    same small grid. Counted rather than timed: a timing assertion would be
    flaky, while the statement count is exactly the property that matters.
    """
    from sqlalchemy import event as sa_event

    from apps.gateway.db import engine

    fx = await _history_fixture(12)
    # Warm every lazy load so only the endpoint's own statements are counted.
    await client.get(f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF})

    calendar_reads = []

    @sa_event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        if "market_calendar" in statement:
            calendar_reads.append(statement)

    try:
        r = await client.get(
            f"/api/events/{fx['subject']}/history", params={"as_of": HISTORY_AS_OF}
        )
    finally:
        sa_event.remove(engine.sync_engine, "before_cursor_execute", _count)

    assert r.json()["n_rows"] == 12
    assert len(calendar_reads) == 1, (
        f"twelve rows must cost ONE calendar query, got {len(calendar_reads)}"
    )


async def test_a_prefetched_calendar_grid_picks_the_same_next_session(client):
    """The batched grid and the per-event query must answer IDENTICALLY.

    The optimization above is only safe if both branches apply the same
    bounds. Pinned against a stored holiday, where the two rules would
    visibly diverge if the filtered list drifted from the SQL predicate.
    """
    await _seed_daily("ACME", start=date(2026, 1, 5), closes=[100.0] * 40)
    friday = date(2026, 2, 6)
    tuesday = date(2026, 2, 10)  # the Monday is a holiday in this fixture
    async with SessionLocal() as s:
        s.add(
            MarketCalendarRow(
                session_date=tuesday,
                exchange="US",
                open_utc=_et_on(tuesday, 9, 30),
                close_utc=_et_on(tuesday, 16, 0),
                source="test",
            )
        )
        await s.commit()
    event_id = await _add_event(
        key="EARNINGS:ACME:2026-02-06", ticker="ACME", when=_et_on(friday, 16, 30)
    )
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        queried = await er.event_window(s, row)
        batched = await er.event_window(s, row, sessions=[tuesday])
    assert queried == batched
    assert queried[1] == _et_on(tuesday, 20, 0)
    assert queried[2] == "spans_next_session:market_calendar"
