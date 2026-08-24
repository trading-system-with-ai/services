"""Macro API — GET/POST /api/events/{id}/macro (Phase G, U3; event spec §8,
§33, §38-§41, §39, §46, §85, §96; audit §6 macro rows, §7.2, §11.9).

WHY EVERY BLS AND TREASURY CALL HERE IS A FAKE AND EVERY EQUITY BAR IS A STUB
BAR. The statistics themselves are staged — a hand-written index series whose
MoM a reader can verify by eye (324.000 -> 324.648 is exactly +0.20%) — because
the point of these tests is the SEAM, not the parser: U1's suite already pins
the real BLS JSON and the real Treasury CSV against live-derived fixtures, and
re-parsing them here would test the same code twice while making these
assertions unreadable. What is NOT faked is the storage path, the release-date
join, the as-of gate and the reaction arithmetic: those are the real functions
production runs.

The guarantees these tests defend, in the order they appear:

1. **A GET NEVER FETCHES.** Asserted against an EXPLODING macro provider, an
   exploding Treasury client and an exploding equity provider, on an event
   whose series are already stored. This is not a performance nicety here: BLS
   allows roughly 25 requests a DAY unregistered, so a read that lazily topped
   up would burn the budget on a page load and then fail the backfill that
   could have repaired it. The read path holds no provider handle at all, and
   this test is what stops that rotting into a lazy top-up later.
2. **Only POST writes**, and it writes THREE tables — observations, yield
   curves and daily bars — plus a DATA_BACKFILL audit row carrying
   ``kind: "event_macro"``.
3. **The release INSTANT is joined from the schedule, and its basis is
   stored.** A period the registry has a CONFIRMED release row for gets that
   row's instant with basis SCHEDULED; a period with no schedule row gets
   period-end + 45 days with basis ESTIMATED. Storing the instant without the
   basis would let a guess be read as a publication time.
4. **THE AS-OF GATE HIDES A LATER PRINT** (§96). The same stored series
   answered at two instants gives two different "previous releases", because
   the gate is applied to the observations by their release instants — not by
   trimming an answer that already saw them.
5. **NEVER A FABRICATED CONSENSUS** (§33). The literal string
   ``CONSENSUS DATA UNAVAILABLE`` is present, ``surprise`` is the unavailable
   string, and no branch of the payload carries a number for either.
6. **An asset with no bars is a NAMED ABSENCE, never a 0.0% return** (§44 rule
   18), and a tenor Treasury did not quote is absent from the curve rather
   than present at zero.
7. **The bundle's ``macro_context`` is filled for BOTH shapes**: an earnings
   bundle gets the upcoming CPI within 14 days; a CPI bundle gets its own
   packet, with the three issuer sections honestly skipped for want of a
   ticker.

Uses the shared ``client`` fixture (conftest.py).
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from apps.gateway import event_macro as seam
from apps.gateway.db import (
    AuditEvent,
    EventRow,
    MacroObservationRow,
    SessionLocal,
    StockBarDaily,
    TreasuryYieldRow,
)
from libs.common.config import get_settings
from libs.market_data import CapabilityNotAvailable, MarketDataError
from libs.trading_core.events.macro import (
    CONSENSUS_UNAVAILABLE_REASON,
    RELEASE_BASIS_ESTIMATED,
    RELEASE_BASIS_SCHEDULED,
    TENOR_2Y,
    TENOR_10Y,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

EASTERN = ZoneInfo("America/New_York")

#: The BLS headline CPI series id — the one series every CPI assertion below
#: reads, taken from the pure catalogue rather than retyped so a catalogue
#: edit breaks these tests loudly instead of silently testing nothing.
HEADLINE = "CUSR0000SA0"

#: A fixed anchor rather than ``now()``: every as-of assertion here is a
#: statement about ordering between instants, and a drifting clock would make
#: "the August print is not yet visible" rot overnight.
NOW = datetime(2025, 9, 20, 12, 0, tzinfo=timezone.utc)

#: ``NOW`` as a QUERY-SAFE string. ``isoformat()`` ends in ``+00:00`` and a
#: raw ``+`` in a query string decodes to a SPACE, which is not an instant —
#: the route answers 422 and the test looks like a seam bug. ``Z`` says the
#: same thing and survives the URL.
NOW_Q = "2025-09-20T12:00:00Z"

#: Hand-written index levels whose transforms a reader can verify by eye:
#: 324.000 -> 324.648 is +0.20% exactly, and 324.648 -> 325.621 is +0.30%.
#: A generated series would make every assertion below a mirror of the code
#: under test.
CPI_LEVELS: dict[str, float] = {
    "2025-04": 322.000,
    "2025-05": 323.000,
    "2025-06": 324.000,
    "2025-07": 324.648,
    "2025-08": 325.621,
}


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _et(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    """An ET wall-clock instant as its UTC equivalent (what the DB stores)."""
    return datetime(y, m, d, hour, minute, tzinfo=EASTERN).astimezone(timezone.utc)


def _weekdays(start: date, count: int) -> list[date]:
    """Consecutive WEEKDAYS — the bar dates ARE trading days, exactly as the
    pure reaction library assumes (it never consults a calendar)."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


async def _add_cpi_event(
    *,
    period: str,
    release: datetime,
    status: EventStatus = EventStatus.CONFIRMED,
    event_type: EventType = EventType.CPI,
) -> int:
    """One stored BLS schedule row, shaped exactly as the U1 adapter writes it."""
    async with SessionLocal() as s:
        row = EventRow(
            event_key=f"{event_type.value}:{release.astimezone(EASTERN).date()}",
            event_type=event_type.value,
            title=f"{event_type.value} — {period}",
            ticker=None,
            scheduled_at=release,
            event_timezone="America/New_York",
            session=EventSession.BEFORE_MARKET.value,
            status=status.value,
            source=EventSourceKind.GOVERNMENT_AGENCY.value,
            source_name="bls",
            source_url="https://www.bls.gov/schedule/news_release/cpi.htm",
            agency="Bureau of Labor Statistics",
            series_id=HEADLINE,
            release_period=period,
            importance=90,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _add_earnings_event(*, ticker: str, when: datetime) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=f"EARNINGS:{ticker}:{when.astimezone(EASTERN).date()}",
            event_type=EventType.EARNINGS.value,
            title=f"{ticker} earnings",
            ticker=ticker,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=EventSession.AFTER_MARKET.value,
            status=EventStatus.CONFIRMED.value,
            source=EventSourceKind.COMPANY_IR_SEC.value,
            source_name="sec_edgar",
            importance=60,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_observations(
    levels: dict[str, float],
    *,
    series_id: str = HEADLINE,
    releases: dict[str, datetime] | None = None,
) -> None:
    """Stored observations, each with the release instant the join produced."""
    releases = releases or {}
    async with SessionLocal() as s:
        for period, value in levels.items():
            instant = releases.get(period)
            s.add(
                MacroObservationRow(
                    series_id=series_id,
                    period=period,
                    value=value,
                    release_at=instant,
                    release_basis=(
                        RELEASE_BASIS_SCHEDULED
                        if instant is not None
                        else RELEASE_BASIS_ESTIMATED
                    ),
                    provider="bls",
                )
            )
        await s.commit()


async def _seed_bars(ticker: str, *, start: date, closes: list[float]) -> list[date]:
    """One daily bar per weekday. ``open`` is 1% under ``close`` so the gap and
    the 1D return are different numbers and no assertion can confuse them."""
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


async def _seed_curves(rows: dict[date, dict[str, float]]) -> None:
    async with SessionLocal() as s:
        for day, tenors in rows.items():
            s.add(
                TreasuryYieldRow(
                    curve_date=day, tenors=dict(tenors), provider="treasury"
                )
            )
        await s.commit()


async def _event_row(event_id: int) -> EventRow:
    async with SessionLocal() as s:
        return await s.get(EventRow, event_id)


# ---------------------------------------------------------------------------
# Fakes — staged sources, never the real network
# ---------------------------------------------------------------------------


class _Obs:
    """The shape ``MacroDataProvider.get_series`` returns (U1's value type)."""

    def __init__(self, series_id: str, period: str, value: float | None) -> None:
        self.series_id = series_id
        self.period = period
        self.value = value


class FakeMacroProvider:
    """A BLS client serving staged levels and counting its calls."""

    name = "bls"

    def __init__(self, by_series: dict[str, dict[str, float]]) -> None:
        self.by_series = by_series
        self.calls: list[tuple[str, int, int]] = []

    def get_series(self, series_id, *, start_year, end_year):
        self.calls.append((series_id, start_year, end_year))
        levels = self.by_series.get(series_id)
        if levels is None:
            raise MarketDataError(f"no staged series {series_id}")
        return [_Obs(series_id, period, value) for period, value in levels.items()]


class _Curve:
    def __init__(self, day: date, tenors: dict[str, float]) -> None:
        self.date = day
        self.tenors = tenors


class FakeTreasury:
    def __init__(self, by_year: dict[int, list[_Curve]]) -> None:
        self.by_year = by_year
        self.calls: list[int] = []

    def get_yield_curve(self, year: int):
        self.calls.append(year)
        return self.by_year.get(year, [])


class ExplodingMacroProvider:
    name = "bls"

    def get_series(self, *a, **k):  # pragma: no cover — must never be reached
        raise AssertionError("a READ fetched macro data (§27; 25 req/day budget)")


class ExplodingTreasury:
    def get_yield_curve(self, *a, **k):  # pragma: no cover
        raise AssertionError("a READ fetched the yield curve (§27)")


class ExplodingEquityProvider:
    name = "exploding"

    def get_daily_bars(self, *a, **k):  # pragma: no cover
        raise AssertionError("a READ fetched daily bars (§27)")

    def get_quotes(self, *a, **k):  # pragma: no cover
        raise AssertionError("a READ fetched quotes")


class _StubBar:
    def __init__(self, day: date, close: float) -> None:
        self.date = day
        self.open = round(close * 0.99, 6)
        self.high = round(close * 1.02, 6)
        self.low = round(close * 0.97, 6)
        self.close = close
        self.volume = 1_000_000.0


class FakeEquityProvider:
    """Serves bars for the symbols it was given and refuses the rest.

    The refusal is the point: on a real subscription UUP or USO is exactly the
    symbol a thin plan does not carry, and the backfill must record that as one
    symbol's reason rather than failing the other seven.
    """

    name = "fake"

    def __init__(self, symbols: dict[str, list[_StubBar]]) -> None:
        self.symbols = symbols
        self.calls: list[str] = []

    def get_daily_bars(self, ticker: str, days: int):
        self.calls.append(ticker)
        if ticker not in self.symbols:
            raise CapabilityNotAvailable(f"{ticker} not in this plan")
        return self.symbols[ticker][-days:]


def _stub_bars(start: date, closes: list[float]) -> list[_StubBar]:
    return [_StubBar(d, c) for d, c in zip(_weekdays(start, len(closes)), closes)]


# ---------------------------------------------------------------------------
# 1. The read never fetches
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_macro_never_fetches_anything(client, monkeypatch):
    """§27 / audit §7.2 rule 1 — and the 25-requests-a-day reason it matters.

    Three exploding sources, one fully-populated event: a read that reached
    ANY of them raises rather than quietly succeeding, so this cannot rot into
    a lazy top-up the way an assertion on call counts could.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})
    await _seed_bars("SPY", start=date(2025, 8, 3), closes=[100.0] * 20)

    monkeypatch.setattr(
        "libs.event_calendar.macro_data_provider",
        lambda *a, **k: ExplodingMacroProvider(),
    )
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider",
        lambda *a, **k: ExplodingEquityProvider(),
    )

    r = await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["packet"]["previous_release"]["period"] == "2025-07"


@pytest.mark.anyio
async def test_get_macro_on_an_unbackfilled_event_names_the_button(client):
    """Honest absence, with the remedy in the reason (§44 rule 18).

    Nothing is stored, so the answer is not an error and not an empty object:
    it is a 200 whose coverage says the actuals are unavailable and names the
    POST that would fill them.
    """
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    r = await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["coverage"]["actuals"]["available"] is False
    assert "backfill" in body["coverage"]["actuals"]["reason"]
    # The CURRENT release is still named — its date is a registry fact that
    # does not depend on any statistic having been fetched.
    assert body["packet"]["current_release"]["period"] == "2025-08"


@pytest.mark.anyio
async def test_get_macro_on_a_non_macro_event_is_200_not_404(client):
    """The row exists; it simply has no statistical release behind it."""
    event_id = await _add_earnings_event(ticker="ACME", when=_et(2025, 9, 10, 16, 30))
    r = await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "not a macro event" in body["reason"]


@pytest.mark.anyio
async def test_get_macro_404s_only_for_a_missing_event(client):
    r = await client.get("/api/events/999999/macro")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_a_future_as_of_is_422_not_a_silent_clamp(client):
    """Clamping answers a DIFFERENT question than the one asked (§96)."""
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    r = await client.get(f"/api/events/{event_id}/macro?as_of={future}")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 2. The transforms and the previous release
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_previous_release_quotes_the_mom_the_levels_imply(client):
    """324.000 -> 324.648 is +0.20%, and the payload must say 0.20.

    Pinned against arithmetic stated in this file, not against whatever the
    endpoint returned — otherwise the assertion is a mirror. ``value_raw``
    carries the level the transform ran on, so the number is checkable by eye
    from the payload alone.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})

    r = await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    body = r.json()
    headline = body["packet"]["previous_release"]["actual"]["headline"]
    assert headline["period"] == "2025-07"
    assert headline["value_raw"] == pytest.approx(324.648)
    assert headline["prior"] == pytest.approx(324.000)
    assert headline["value"] == pytest.approx(0.20, abs=0.005)
    assert headline["unit"] == "percent"
    assert headline["release_time_basis"] == RELEASE_BASIS_SCHEDULED


@pytest.mark.anyio
async def test_as_of_hides_a_print_released_after_it(client):
    """§96 — the same stored series, two instants, two different answers.

    The August print's release instant is 2025-09-11 08:30 ET. Asked as of
    2025-09-20 the previous release is August; asked as of 2025-09-01 it is
    July, because the gate runs on the OBSERVATIONS by their release
    instants — not by trimming an answer that already saw them.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    release_aug = _et(2025, 9, 11, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=release_aug)
    await _seed_observations(
        CPI_LEVELS, releases={"2025-07": release_jul, "2025-08": release_aug}
    )

    late = (await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")).json()
    assert late["packet"]["previous_release"]["period"] == "2025-08"

    early_q = "2025-09-01T12:00:00Z"
    early = (
        await client.get(f"/api/events/{event_id}/macro?as_of={early_q}")
    ).json()
    assert early["packet"]["previous_release"]["period"] == "2025-07"
    # And no August number leaked into the trend either.
    trend = early["packet"]["recent_trend"][f"{HEADLINE}:headline"]
    assert all(p["period"] != "2025-08" for p in trend["prints"])


@pytest.mark.anyio
async def test_recent_trend_carries_a_direction_over_the_visible_prints(client):
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    trend = body["packet"]["recent_trend"][f"{HEADLINE}:headline"]
    assert trend["n_points"] >= 3
    assert trend["direction"] in {"rising", "falling", "flat"}
    assert trend["unit"] == "percent"


# ---------------------------------------------------------------------------
# 3. No fabricated consensus (§33)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_consensus_and_no_surprise_in_any_branch(client):
    """§33/§98 — the literal string, in a payload where everything worked.

    A macro block that is otherwise complete is exactly the shape that invites
    a reader to supply an expectation from memory, so the absence is stated
    three times: on the previous release, on the current one, and as a
    top-level disclaimer nobody has to open ``coverage`` to see.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    packet = body["packet"]
    assert packet["previous_release"]["consensus"] == CONSENSUS_UNAVAILABLE_REASON
    assert packet["current_release"]["consensus"] == CONSENSUS_UNAVAILABLE_REASON
    assert "UNAVAILABLE" in packet["previous_release"]["surprise"]
    assert body["coverage"]["consensus"]["available"] is False
    assert "CONSENSUS DATA UNAVAILABLE" in body["disclaimer"]
    # No numeric surprise anywhere in the payload, under any key.
    import json as _json

    text = _json.dumps(body)
    assert '"surprise": 0' not in text
    assert '"eps_surprise"' not in text


# ---------------------------------------------------------------------------
# 4. The §39 cross-asset reaction
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reaction_measures_present_assets_and_names_the_absent_ones(client):
    """§39/§44 rule 18 — an asset with no bars is a REASON, never a 0.0%.

    SPY rises 1% on the release day and QQQ falls 2%; the other six reference
    symbols have no bars at all. The two present symbols carry real returns,
    the six absent ones are listed with a reason naming the backfill, and no
    zero appears anywhere for them.
    """
    release_jul = _et(2025, 8, 12, 8, 30)  # a Wednesday, BEFORE_MARKET
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})

    # 2025-08-10 is a Sunday, so the weekdays run Mon 11, Tue 12 (the release
    # day), Wed 13, ... The BEFORE_MARKET pre-event close is therefore Monday's
    # and the reaction close is Tuesday's — the release-day bar itself.
    await _seed_bars(
        "SPY", start=date(2025, 8, 10), closes=[100.0, 101.0] + [101.0] * 8
    )
    await _seed_bars(
        "QQQ", start=date(2025, 8, 10), closes=[200.0, 196.0] + [196.0] * 8
    )

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    reaction = body["previous_release_reaction"]
    assert reaction["available"] is True

    spy = reaction["assets"]["SPY"]
    # BEFORE_MARKET: the pre-event close is 2025-08-11 (100.0) and the
    # reaction close is the release day itself (101.0) -> +1% as a FRACTION.
    assert spy["pre_event_close"] == pytest.approx(100.0)
    assert spy["returns"]["1D"] == pytest.approx(0.01, abs=1e-9)
    assert spy["returns_unit"] == "fraction"
    assert spy["is_proxy"] is False

    qqq = reaction["assets"]["QQQ"]
    assert qqq["returns"]["1D"] == pytest.approx(-0.02, abs=1e-9)

    absent = {row["symbol"]: row["reason"] for row in reaction["unavailable"]}
    # DIA and VIXY are in the roster but unseeded here — they belong in the
    # NAMED-ABSENT list, which is the whole point: an unfetched index is
    # reported as absent with a reason, never as a 0.0% reaction.
    assert set(absent) == {"DIA", "VIXY", "TLT", "IEF", "SHY", "GLD", "USO", "UUP"}
    for reason in absent.values():
        assert reason and "0" != reason
        assert "no stored daily bars" in reason


@pytest.mark.anyio
async def test_yield_changes_are_basis_points_and_a_missing_tenor_is_absent(client):
    """§39 — bp and % never share a column, and an unquoted tenor is not 0.

    The 2Y goes 4.20 -> 4.28 across the release: eight basis points, stated as
    8.0 rather than as 0.08. The 10Y is deliberately UNQUOTED on the day
    before, and the payload must say so rather than reporting a 428bp move
    from an imagined zero.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})
    await _seed_bars("SPY", start=date(2025, 8, 10), closes=[100.0] * 10)
    await _seed_curves(
        {
            date(2025, 8, 11): {TENOR_2Y: 4.20},  # NO 10Y that day
            date(2025, 8, 12): {TENOR_2Y: 4.28, TENOR_10Y: 4.55},
        }
    )

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    yields = body["previous_release_reaction"]["yields"]
    two = yields[TENOR_2Y]
    assert two["before"] == pytest.approx(4.20)
    assert two["after"] == pytest.approx(4.28)
    assert two["change_bp"] == pytest.approx(8.0, abs=1e-6)
    assert two["change_unit"] == "basis_points"
    assert two["level_unit"] == "percent"

    ten = yields[TENOR_10Y]
    assert ten["change_bp"] is None
    assert ten["before"] is None
    assert ten["reason"]


@pytest.mark.anyio
async def test_proxies_are_badged_as_proxies(client):
    """TLT is a fund holding long Treasuries, not "long rates" (§39)."""
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})
    await _seed_bars("TLT", start=date(2025, 8, 10), closes=[90.0] * 10)

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    tlt = body["previous_release_reaction"]["assets"]["TLT"]
    assert tlt["is_proxy"] is True
    assert tlt["role"] == "long_duration_proxy"
    assert "proxy" in body["previous_release_reaction"]["proxy_note"].lower()


# ---------------------------------------------------------------------------
# 5. §40 — the related-evidence window
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_related_evidence_lists_events_between_the_releases(client):
    """§40 — the deterministic factual set, with NO keyword filter applied.

    A PPI print lands between the July CPI release and ``as_of``; a JOLTS
    print lands BEFORE the window opens. Only the first is in the list, and
    the event under analysis is never its own related evidence.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})
    await _add_cpi_event(
        period="2025-07",
        release=_et(2025, 8, 14, 8, 30),
        event_type=EventType.PPI,
    )
    await _add_cpi_event(
        period="2025-06",
        release=_et(2025, 8, 4, 10, 0),  # before the window opens
        event_type=EventType.JOLTS,
    )

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    related = body["related_evidence"]
    types = [item["event_type"] for item in related["events"]]
    assert "PPI" in types
    assert "JOLTS" not in types
    assert all(item["event_id"] != event_id for item in related["events"])
    # §40 explicitly forbids a rigid keyword list here.
    assert "keyword" in related["note"].lower()


# ---------------------------------------------------------------------------
# 6. The backfill — the only path that spends requests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_backfill_stores_observations_curves_bars_and_an_audit_row(
    client, monkeypatch
):
    """One press writes THREE tables and one DATA_BACKFILL row (rule 12).

    The release instants come from the REGISTRY, not from the statistics
    source, which is the join this whole seam exists to make: BLS's data API
    returns no timestamps at all.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))

    macro = FakeMacroProvider({HEADLINE: CPI_LEVELS})
    treasury = FakeTreasury(
        {2025: [_Curve(date(2025, 8, 11), {TENOR_2Y: 4.20, TENOR_10Y: 4.50})]}
    )
    equity = FakeEquityProvider(
        {"SPY": _stub_bars(date(2025, 8, 3), [100.0] * 15)}
    )
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider", lambda *a, **k: equity
    )

    async with SessionLocal() as s:
        result = await seam.backfill_macro(
            s,
            await s.get(EventRow, event_id),
            settings=get_settings(),
            as_of=NOW,
            macro_provider=macro,
            treasury=treasury,
        )

    assert result["available"] is True
    assert result["counts"]["observations"] == len(CPI_LEVELS)
    assert result["counts"]["yield_curves"] == 1
    assert result["counts"]["bars"] > 0
    # BLS's three-year ceiling is what was asked for, not ten.
    assert macro.calls[0][2] - macro.calls[0][1] + 1 == seam.BACKFILL_SERIES_YEARS

    async with SessionLocal() as s:
        stored = (
            (
                await s.execute(
                    select(MacroObservationRow).where(
                        MacroObservationRow.series_id == HEADLINE
                    )
                )
            )
            .scalars()
            .all()
        )
        by_period = {row.period: row for row in stored}
        curves = (await s.execute(select(TreasuryYieldRow))).scalars().all()
        spy = (
            (
                await s.execute(
                    select(StockBarDaily).where(StockBarDaily.ticker == "SPY")
                )
            )
            .scalars()
            .all()
        )
        audits = (
            (
                await s.execute(
                    select(AuditEvent).where(AuditEvent.action == "DATA_BACKFILL")
                )
            )
            .scalars()
            .all()
        )

    assert set(by_period) == set(CPI_LEVELS)
    assert by_period["2025-07"].value == pytest.approx(324.648)
    assert by_period["2025-07"].provider == "bls"
    assert len(curves) == 1
    assert curves[0].tenors[TENOR_2Y] == pytest.approx(4.20)
    assert spy
    macro_audits = [a for a in audits if (a.details or {}).get("kind") == "event_macro"]
    assert len(macro_audits) == 1
    assert macro_audits[0].details["counts"]["observations"] == len(CPI_LEVELS)


@pytest.mark.anyio
async def test_the_release_instant_is_scheduled_when_the_registry_knows_it(
    client, monkeypatch
):
    """The join, and the basis that makes it readable (§8, §85).

    July has a stored CONFIRMED release row, so its observation gets that
    instant with basis SCHEDULED. April has none, so it gets period-end + 45
    days with basis ESTIMATED. Storing the instant WITHOUT the basis is what
    would let a guess be quoted as a publication time.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))

    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider",
        lambda *a, **k: FakeEquityProvider({}),
    )
    async with SessionLocal() as s:
        await seam.backfill_macro(
            s,
            await s.get(EventRow, event_id),
            settings=get_settings(),
            as_of=NOW,
            macro_provider=FakeMacroProvider({HEADLINE: CPI_LEVELS}),
            treasury=FakeTreasury({}),
        )

    async with SessionLocal() as s:
        july = await s.get(MacroObservationRow, (HEADLINE, "2025-07"))
        april = await s.get(MacroObservationRow, (HEADLINE, "2025-04"))

    assert july.release_basis == RELEASE_BASIS_SCHEDULED
    assert july.release_at.replace(tzinfo=timezone.utc) == release_jul
    assert april.release_basis == RELEASE_BASIS_ESTIMATED
    # 2025-04-30 + 45 days = 2025-06-14.
    assert april.release_at.date() == date(2025, 6, 14)


@pytest.mark.anyio
async def test_backfill_is_idempotent_and_a_revision_overwrites(client, monkeypatch):
    """One fact per (series, period) — a re-fetch restates, never duplicates.

    The second press serves a REVISED July level, which is exactly what BLS
    does every February. One row survives, carrying the new number: a second
    row would double every MoM computed over the series.
    """
    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider",
        lambda *a, **k: FakeEquityProvider({}),
    )

    async def _run(levels):
        async with SessionLocal() as s:
            return await seam.backfill_macro(
                s,
                await s.get(EventRow, event_id),
                settings=get_settings(),
                as_of=NOW,
                macro_provider=FakeMacroProvider({HEADLINE: levels}),
                treasury=FakeTreasury({}),
            )

    await _run(CPI_LEVELS)
    revised = dict(CPI_LEVELS, **{"2025-07": 324.700})
    await _run(revised)

    async with SessionLocal() as s:
        rows = (
            (
                await s.execute(
                    select(MacroObservationRow).where(
                        MacroObservationRow.series_id == HEADLINE,
                        MacroObservationRow.period == "2025-07",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].value == pytest.approx(324.700)


@pytest.mark.anyio
async def test_one_dead_source_never_costs_the_other_two(client, monkeypatch):
    """A CPI packet with no yield curve is still most of the §38 answer.

    Treasury raises, one equity symbol is off-plan, and the series still land:
    every failure is a named row in the response instead of an exception.
    """
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))

    class _Dead:
        def get_yield_curve(self, year):
            raise MarketDataError("treasury 503")

    equity = FakeEquityProvider({"SPY": _stub_bars(date(2025, 8, 3), [100.0] * 10)})
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider", lambda *a, **k: equity
    )
    async with SessionLocal() as s:
        result = await seam.backfill_macro(
            s,
            await s.get(EventRow, event_id),
            settings=get_settings(),
            as_of=NOW,
            macro_provider=FakeMacroProvider({HEADLINE: CPI_LEVELS}),
            treasury=_Dead(),
        )

    assert result["counts"]["observations"] == len(CPI_LEVELS)
    assert result["counts"]["yield_curves"] == 0
    assert "treasury 503" in result["yield_curve"]["reason"]
    by_symbol = {row["symbol"]: row for row in result["bars"]}
    assert by_symbol["SPY"]["status"] == "OK"
    assert by_symbol["UUP"]["status"] == "ERROR"
    assert "not in this plan" in by_symbol["UUP"]["reason"]


@pytest.mark.anyio
async def test_an_event_type_with_no_adapter_says_so_without_spending_a_call(
    client, monkeypatch
):
    """GDP needs a BEA key and RETAIL_SALES a Census adapter (§44 rule 18).

    Their release DATES are tracked either way; their actuals are honestly
    unavailable. The macro provider must not be called at all — there is no
    series id to ask for, and inventing one would be a 404 dressed as coverage.
    """
    event_id = await _add_cpi_event(
        period="2025-Q2",
        release=_et(2025, 8, 26, 8, 30),
        event_type=EventType.GDP,
    )
    macro = FakeMacroProvider({})
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider",
        lambda *a, **k: FakeEquityProvider({}),
    )
    async with SessionLocal() as s:
        result = await seam.backfill_macro(
            s,
            await s.get(EventRow, event_id),
            settings=get_settings(),
            as_of=NOW,
            macro_provider=macro,
            treasury=FakeTreasury({}),
        )

    assert macro.calls == []
    assert result["series"][0]["status"] == "UNAVAILABLE"
    assert "no data source configured" in result["series"][0]["reason"]

    body = (
        await client.get(f"/api/events/{event_id}/macro?as_of={NOW_Q}")
    ).json()
    assert body["coverage"]["actuals"]["available"] is False


@pytest.mark.anyio
async def test_backfill_endpoint_is_reachable_and_returns_counts(client, monkeypatch):
    """The POST route itself, end to end through FastAPI."""
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    monkeypatch.setattr(
        "libs.event_calendar.macro_data_provider",
        lambda *a, **k: FakeMacroProvider({HEADLINE: CPI_LEVELS}),
    )
    monkeypatch.setattr(
        "libs.event_calendar.treasury.TreasuryYields",
        lambda **k: FakeTreasury({}),
    )
    monkeypatch.setattr(
        "apps.gateway.event_macro.get_provider",
        lambda *a, **k: FakeEquityProvider({}),
    )
    r = await client.post(f"/api/events/{event_id}/macro/backfill")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["counts"]["observations"] == len(CPI_LEVELS)


# ---------------------------------------------------------------------------
# 7. The evidence bundle's macro_context (§46)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_an_earnings_bundle_carries_the_upcoming_macro_releases(client):
    """§46 — reporting the day before CPI is a different trade (the whole point).

    The CPI eight days out is listed with its days_to; a CPI ninety days out is
    outside the fourteen-day horizon and is not.
    """
    from apps.gateway import event_evidence

    earnings_at = _et(2025, 9, 3, 16, 30)
    earnings_id = await _add_earnings_event(ticker="ACME", when=earnings_at)
    await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _add_cpi_event(period="2025-11", release=_et(2025, 12, 10, 8, 30))

    async with SessionLocal() as s:
        bundle = await event_evidence.build_evidence_bundle(
            s,
            await s.get(EventRow, earnings_id),
            as_of=earnings_at,
            settings=get_settings(),
        )

    context = bundle["macro_context"]
    assert context["kind"] == "upcoming_macro_releases"
    assert context["available"] is True
    periods = [item["event_type"] for item in context["upcoming"]]
    assert periods == ["CPI"]
    assert 7.0 < context["upcoming"][0]["days_to"] < 9.0
    assert context["consensus_status"] == CONSENSUS_UNAVAILABLE_REASON


@pytest.mark.anyio
async def test_an_estimated_macro_date_is_not_offered_as_context(client):
    """§7 — a derived date must not be quoted as a scheduled release.

    The model writes prose about "CPI two days after this print"; that is a
    materially different sentence when the CPI date is this platform's guess.
    """
    from apps.gateway import event_evidence

    earnings_at = _et(2025, 9, 3, 16, 30)
    earnings_id = await _add_earnings_event(ticker="ACME", when=earnings_at)
    await _add_cpi_event(
        period="2025-08",
        release=_et(2025, 9, 11, 8, 30),
        status=EventStatus.ESTIMATED,
    )

    async with SessionLocal() as s:
        bundle = await event_evidence.build_evidence_bundle(
            s,
            await s.get(EventRow, earnings_id),
            as_of=earnings_at,
            settings=get_settings(),
        )
    assert bundle["macro_context"]["upcoming"] == []


@pytest.mark.anyio
async def test_a_macro_bundle_carries_its_packet_and_skips_the_issuer_sections(client):
    """A CPI bundle's evidence IS the packet; the three issuer seams are skipped.

    Skipped rather than asked-and-refused: each already answers ``no_ticker``
    correctly, but asking costs a bar load and a news query to receive a
    sentence the coverage map states for free.
    """
    from apps.gateway import event_evidence

    release_jul = _et(2025, 8, 12, 8, 30)
    await _add_cpi_event(period="2025-07", release=release_jul)
    event_id = await _add_cpi_event(period="2025-08", release=_et(2025, 9, 11, 8, 30))
    await _seed_observations(CPI_LEVELS, releases={"2025-07": release_jul})

    async with SessionLocal() as s:
        bundle = await event_evidence.build_evidence_bundle(
            s, await s.get(EventRow, event_id), as_of=NOW, settings=get_settings()
        )

    context = bundle["macro_context"]
    assert context["kind"] == "macro_event_packet"
    assert context["packet"]["previous_release"]["period"] == "2025-07"
    for section in ("price_analysis", "fundamentals", "news"):
        assert bundle["coverage"][section]["available"] is False
        assert "no ticker" in bundle["coverage"][section]["reason"]
    assert bundle["coverage"]["macro_context"]["available"] is True


@pytest.mark.anyio
async def test_the_macro_prompt_asks_the_component_question(client):
    """§41 — and only for a macro release, so earnings prompt bytes are stable.

    The cache key is the bundle, and a prompt edit that touched every event
    would invalidate every stored analysis for a sentence only macro rows need.
    """
    from libs.llm.event_analysis import build_user_message

    macro_bundle = {"event": {"event_type": "CPI"}, "as_of": "2025-09-20T12:00:00+00:00"}
    earnings_bundle = {"event": {"event_type": "EARNINGS"}, "as_of": "x"}

    macro_msg = build_user_message(macro_bundle)
    earnings_msg = build_user_message(earnings_bundle)

    assert "LLM ANALYSIS" in macro_msg
    assert "COMPONENT" in macro_msg
    assert "LLM ANALYSIS" not in earnings_msg
    assert "COMPONENT" not in earnings_msg


# ---------------------------------------------------------------------------
# 8. A macro release is a first-class row in the ordinary feed
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_macro_release_appears_in_the_ordinary_events_feed(client):
    """A CPI print is an EVENT, not a separate surface (§8, §12 MARKET_WIDE).

    The whole Phase G design rests on macro releases being rows in the same
    registry as earnings — that is what lets ``macro_context`` be a query over
    ``events`` rather than a second calendar, and what makes the §40 related
    window a plain scan. If a ticker-less row ever stopped reaching the feed,
    every macro surface would go quiet at once while every test that only
    exercises the macro endpoint kept passing. This is the tripwire for that.
    """
    release = datetime.now(timezone.utc) + timedelta(days=5)
    await _add_cpi_event(period="2025-08", release=release)

    body = (await client.get("/api/events?horizon=30d")).json()
    keys = {row["event_key"] for row in body["events"]}
    assert any(key.startswith("CPI:") for key in keys)
    cpi = next(row for row in body["events"] if row["event_key"].startswith("CPI:"))
    assert cpi["ticker"] is None
    assert cpi["event_type"] == "CPI"
