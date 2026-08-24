"""Daily-bar freshness (§15): stored history is APPENDED to, never frozen.

Before this existed, ``ensure_daily_bars`` served whatever the first backfill
stored, forever — every signal, regime read and backtest would age with it
until the data-quality gate (correctly) refused to trade. These tests pin the
refresh contract:

- stale stored bars -> the missing COMPLETE trading days are fetched and
  appended (audited as DATA_BACKFILL mode=refresh), never rewritten;
- fresh stored bars -> the provider is not asked at all;
- a bar dated today (Eastern) is provisional and never stored;
- refresh attempts are throttled per symbol (holidays look like missing bars);
- a refresh failure serves the stored bars — yesterday's real close beats
  no answer.
"""
from datetime import date, datetime, timedelta

import pytest

from apps.gateway.db import SessionLocal, StockBarDaily
from apps.gateway.routers import analysis
from apps.gateway.routers.analysis import (
    EASTERN,
    _last_expected_trading_date,
    ensure_daily_bars,
)

pytestmark = pytest.mark.anyio


class Bar:
    def __init__(self, ts: date, close: float):
        self.ts = ts
        self.open = close - 1.0
        self.high = close + 1.0
        self.low = close - 2.0
        self.close = close
        self.volume = 1000.0


class FakeProvider:
    """Serves a fixed daily series ending at `end` (inclusive)."""

    def __init__(self, end: date, days: int = 30):
        self.calls = 0
        all_days: list[date] = []
        d = end
        while len(all_days) < days:
            if d.weekday() < 5:
                all_days.append(d)
            d -= timedelta(days=1)
        self.series = [Bar(day, 100.0 + i) for i, day in enumerate(reversed(all_days))]

    def get_daily_bars(self, ticker: str, days: int):
        self.calls += 1
        return self.series[-days:] if days < len(self.series) else self.series


def weekdays_back(start: date, n: int) -> date:
    d = start
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


async def seed_bars(ticker: str, newest: date, count: int = 5) -> None:
    days: list[date] = []
    d = newest
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    async with SessionLocal() as s:
        for i, day in enumerate(reversed(days)):
            s.add(
                StockBarDaily(
                    ticker=ticker, ts=day, open=1, high=2, low=0.5,
                    close=50.0 + i, volume=10,
                )
            )
        await s.commit()


@pytest.fixture(autouse=True)
def reset_refresh_throttle():
    analysis._refresh_attempts.clear()
    yield
    analysis._refresh_attempts.clear()


def test_last_expected_trading_date_skips_weekends():
    assert _last_expected_trading_date(date(2026, 8, 10)) == date(2026, 8, 7)  # Mon -> Fri
    assert _last_expected_trading_date(date(2026, 8, 9)) == date(2026, 8, 7)  # Sun -> Fri
    assert _last_expected_trading_date(date(2026, 8, 8)) == date(2026, 8, 7)  # Sat -> Fri
    assert _last_expected_trading_date(date(2026, 8, 12)) == date(2026, 8, 11)  # Wed -> Tue


async def test_stale_bars_are_refreshed_append_only(client, monkeypatch):
    """Newest stored bar 3 weekdays old -> exactly the missing tail appends."""
    today = datetime.now(EASTERN).date()
    stale_newest = weekdays_back(today, 3)
    await seed_bars("RFRSH", stale_newest)

    provider = FakeProvider(end=weekdays_back(today, 1))
    monkeypatch.setattr(analysis, "get_provider", lambda name: provider)

    async with SessionLocal() as s:
        bars = await ensure_daily_bars(s, "RFRSH", "stub")

    assert provider.calls == 1
    dates = [b.ts for b in bars]
    assert dates == sorted(dates) and len(dates) == len(set(dates))  # append-only, no dupes
    assert bars[-1].ts == weekdays_back(today, 1)  # caught up to last complete day
    assert bars[-1].ts > stale_newest

    # Audited as a refresh, not a fresh backfill.
    r = await client.get("/api/audit", params={"entity_id": "RFRSH"})
    refreshes = [
        e for e in r.json()
        if e["action"] == "DATA_BACKFILL" and e["details"].get("mode") == "refresh"
    ]
    assert len(refreshes) == 1
    assert refreshes[0]["details"]["previous_newest"] == stale_newest.isoformat()


async def test_fresh_bars_do_not_ask_the_provider(client, monkeypatch):
    today = datetime.now(EASTERN).date()
    await seed_bars("FRESH", _last_expected_trading_date(today))

    provider = FakeProvider(end=today)
    monkeypatch.setattr(analysis, "get_provider", lambda name: provider)

    async with SessionLocal() as s:
        bars = await ensure_daily_bars(s, "FRESH", "stub")
    assert provider.calls == 0
    assert bars[-1].ts == _last_expected_trading_date(today)


async def test_todays_partial_bar_is_never_stored(client, monkeypatch):
    """A provider series that includes TODAY (Eastern) — still forming — must
    be trimmed to complete days on both backfill and refresh."""
    today = datetime.now(EASTERN).date()
    provider = FakeProvider(end=today)  # includes today's provisional bar
    monkeypatch.setattr(analysis, "get_provider", lambda name: provider)

    async with SessionLocal() as s:
        bars = await ensure_daily_bars(s, "PARTL", "stub")  # first backfill
    assert all(b.ts < today for b in bars)


async def test_refresh_attempts_are_throttled(client, monkeypatch):
    """Stale + provider has nothing newer (holiday shape): asked once, then
    the throttle holds — the second call serves stored bars quietly."""
    today = datetime.now(EASTERN).date()
    stale_newest = weekdays_back(today, 3)
    await seed_bars("THROT", stale_newest)

    provider = FakeProvider(end=stale_newest)  # nothing newer available
    monkeypatch.setattr(analysis, "get_provider", lambda name: provider)

    async with SessionLocal() as s:
        await ensure_daily_bars(s, "THROT", "stub")
        await ensure_daily_bars(s, "THROT", "stub")
    assert provider.calls == 1  # second call inside the throttle window


async def test_refresh_failure_serves_stored_bars(client, monkeypatch):
    today = datetime.now(EASTERN).date()
    stale_newest = weekdays_back(today, 3)
    await seed_bars("RFAIL", stale_newest)

    class ExplodingProvider:
        def get_daily_bars(self, ticker, days):
            raise RuntimeError("provider down")

    monkeypatch.setattr(analysis, "get_provider", lambda name: ExplodingProvider())

    async with SessionLocal() as s:
        bars = await ensure_daily_bars(s, "RFAIL", "stub")
    assert bars[-1].ts == stale_newest  # stored real data still serves
