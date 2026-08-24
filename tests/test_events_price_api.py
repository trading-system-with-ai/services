"""Price context API — GET /api/events/{id}/price-context (event spec §14,
§17, §19, §31, §32, §64, §85, §96; audit §7, §11.2 Phase E1).

WHY THE BARS ARE SEEDED, NOT FETCHED. Every test below inserts its own
``stock_bars_daily`` rows before calling the endpoint. ``ensure_daily_bars``
serves stored bars whenever any exist, so seeding both pins the arithmetic to
hand-computable numbers AND keeps the stub provider's synthetic random walk out
of the assertions — a test whose expected 1D return comes from the same
generator that produced the bars proves nothing about the reaction window.
The one test that deliberately has NO stored bars is the provider-outage test,
which is precisely about the fetch path failing.

The guarantees these tests defend, in the order they appear:

1. **The as-of gate is real** (§14, §96). Two paired assertions, never one:
   at 15:59 ET the same-day bar is absent AND at 16:00 ET it is present; with
   ``as_of`` before an event's reaction bar the reaction is unavailable with a
   reason AND with a later ``as_of`` the same event resolves. A gate that
   simply returned nothing would pass the first half of each pair and fail the
   second.
2. **Session-correct windows** (§17). An AMC print reacts on the NEXT bar; a
   BMO print reacts on the SAME date's bar. The seeded closes are chosen so
   the two answers differ, so an implementation that ignored ``session``
   could not pass both.
3. **Honest absence, never a zero** (§44 rule 18, §85). An event predating the
   bar history returns ``bars_available: false`` with the boundary date in its
   reason; an unconfigured provider leaves the endpoint at 200 with a reason;
   a macro event with no ticker says ``no_ticker``; every ``null`` metric has
   a companion entry.
4. **Point-in-time event selection.** A past earnings row scheduled after
   ``as_of`` is not history yet; an ESTIMATED past date is not a comparable.
5. **§19/§64 sample discipline.** Statistics always carry ``n_available`` and
   ``positive_count``, and a window with fewer than two usable events is
   ``null`` with ``insufficient_sample`` rather than a "median" of one print.

Uses the shared ``client`` / ``unconfigured_client`` fixtures (conftest.py).
"""
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apps.gateway.db import EventRow, SessionLocal, StockBarDaily
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


#: A hand-built run of consecutive WEEKDAY sessions with known closes. Weekends
#: are skipped so the bar dates ARE trading days, exactly as the library
#: assumes (it never consults a calendar — the bars are the calendar).
def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


async def _seed_bars(
    ticker: str,
    *,
    start: date,
    closes: list[float],
    volume: float = 1_000_000.0,
) -> list[date]:
    """Insert one bar per weekday from ``start`` with the given closes.

    ``open`` is deliberately NOT equal to ``close``: it is set 1% below, so the
    gap return (react_open / pre_close - 1) and the 1D return (react_close /
    pre_close - 1) are DIFFERENT numbers and a test can tell which one an
    implementation reported.
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
                    volume=volume,
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
            revision_history=[],
        )
        s.add(row)
        await s.commit()
        return row.id


def _flat_closes(n: int, first: float = 100.0, step: float = 0.0) -> list[float]:
    return [first + step * i for i in range(n)]


async def _standard_fixture(
    ticker: str = "ACME",
    *,
    bench: bool = True,
) -> dict:
    """A 60-session history plus one past AMC print and one upcoming print.

    Bar dates run from Mon 2026-01-05. The past event is on session index 20
    (2026-02-02); its reaction bar is index 21. Closes step by +1 except across
    the reaction, where a +10 jump makes the measured move unmistakable.
    """
    closes = _flat_closes(60, first=100.0, step=1.0)
    for i in range(21, 60):
        closes[i] += 10.0
    days = await _seed_bars(ticker, start=date(2026, 1, 5), closes=closes)
    if bench:
        # SPY drifts +0.1/day with no jump: the abnormal return is therefore
        # dominated by the stock's own move, and a benchmark-blind
        # implementation cannot produce the same number.
        await _seed_bars(
            "SPY", start=date(2026, 1, 5), closes=_flat_closes(60, 500.0, 0.1)
        )
    past_id = await _add_event(
        key=f"EARNINGS:{ticker}:{days[20].isoformat()}",
        ticker=ticker,
        when=_et(days[20].year, days[20].month, days[20].day, 16, 30),
    )
    upcoming_day = days[-1] + timedelta(days=7)
    upcoming_id = await _add_event(
        key=f"EARNINGS:{ticker}:{upcoming_day.isoformat()}",
        ticker=ticker,
        when=_et(upcoming_day.year, upcoming_day.month, upcoming_day.day, 16, 30),
    )
    return {
        "days": days,
        "closes": closes,
        "past_id": past_id,
        "upcoming_id": upcoming_id,
        "ticker": ticker,
    }


def _after_close(day: date) -> str:
    """ISO instant at 16:30 ET on ``day`` — after the regular close, so that
    day's bar is knowable."""
    return datetime(day.year, day.month, day.day, 16, 30, tzinfo=EASTERN).isoformat()


# ---------------------------------------------------------------------------
# 1. Shape, provenance and the 404 / 422 boundaries
# ---------------------------------------------------------------------------


async def test_payload_carries_block_level_provenance_labels(client):
    """§91: bars are DATA, everything derived from them is QUANT. A payload
    without the labels lets the UI render a computed number as a vendor fact."""
    fx = await _standard_fixture()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context",
        params={"as_of": _after_close(fx["days"][-1])},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"] == {"bars": "DATA", "metrics": "QUANT"}


async def test_payload_carries_every_contract_block(client):
    fx = await _standard_fixture()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context",
        params={"as_of": _after_close(fx["days"][-1])},
    )
    body = r.json()
    for key in (
        "event_id",
        "ticker",
        "as_of",
        "provenance",
        "data_freshness",
        "pre_event",
        "previous_events",
        "history_stats",
        "not_backtestable",
        "unavailable",
    ):
        assert key in body, key
    assert body["event_id"] == fx["upcoming_id"]
    assert body["ticker"] == "ACME"


async def test_data_freshness_reports_the_real_last_bar_and_the_source(client):
    fx = await _standard_fixture()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context",
        params={"as_of": _after_close(fx["days"][-1])},
    )
    fresh = r.json()["data_freshness"]
    assert fresh["bars_through"] == fx["days"][-1].isoformat()
    assert fresh["n_bars"] == 60
    # The provider NAME, not a boolean: the UI labels which vendor's closes
    # these are (iex is not consolidated tape, §17).
    assert fresh["bars_source"] == "stub"


async def test_missing_event_is_404(client):
    r = await client.get("/api/events/999999/price-context")
    assert r.status_code == 404


async def test_as_of_in_the_future_is_422_not_silently_clamped(client):
    """A request for prices that do not exist yet is a mistake worth
    reporting; clamping to now would answer a different question than asked."""
    fx = await _standard_fixture()
    future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context", params={"as_of": future}
    )
    assert r.status_code == 422
    assert "future" in str(r.json()["detail"]).lower()


async def test_malformed_as_of_is_422(client):
    fx = await _standard_fixture()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context",
        params={"as_of": "not-a-timestamp"},
    )
    assert r.status_code == 422


async def test_as_of_defaults_to_now_when_omitted(client):
    """Optional at the HTTP boundary only — the seam itself requires it
    (audit §7.2 rule 2)."""
    fx = await _standard_fixture()
    r = await client.get(f"/api/events/{fx['upcoming_id']}/price-context")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    stamped = datetime.fromisoformat(body["as_of"])
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 120


async def test_not_backtestable_labels_travel_in_the_payload(client):
    """§85 / audit §7.3: the option-side context this tab will later carry is
    not reconstructable point-in-time, and the payload says so rather than
    letting an empty list imply everything here is."""
    fx = await _standard_fixture()
    r = await client.get(
        f"/api/events/{fx['upcoming_id']}/price-context",
        params={"as_of": _after_close(fx["days"][-1])},
    )
    labels = r.json()["not_backtestable"]
    assert "historical_implied_move" in labels
    assert "historical_atm_iv" in labels


# ---------------------------------------------------------------------------
# 2. Non-ticker events (Phase G owns multi-asset)
# ---------------------------------------------------------------------------


async def test_macro_event_without_a_ticker_answers_available_false(client):
    cpi_id = await _add_event(
        key="CPI:2026-03",
        ticker=None,
        when=_et(2026, 3, 11, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
        title="CPI March 2026",
    )
    r = await client.get(f"/api/events/{cpi_id}/price-context")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "no_ticker"


async def test_macro_event_answer_carries_no_fabricated_price_block(client):
    """The honest shape is a REFUSAL, not an empty pre_event block full of
    nulls that a client would render as "0% run-up"."""
    fomc_id = await _add_event(
        key="FOMC_DECISION:2026-03-18",
        ticker=None,
        when=_et(2026, 3, 18, 14, 0),
        event_type=EventType.FOMC_DECISION,
        session=EventSession.DURING_MARKET,
        title="FOMC decision",
    )
    body = (await client.get(f"/api/events/{fomc_id}/price-context")).json()
    assert "pre_event" not in body
    assert "previous_events" not in body


# ---------------------------------------------------------------------------
# 3. §17 reaction windows — session-correct, and hand-computed
# ---------------------------------------------------------------------------


async def test_after_market_event_reacts_on_the_next_bar(client):
    """AMC on D: the pre-event close is D's close, the reaction is D+1."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    past = body["previous_events"][0]
    reaction = past["reaction"]
    assert reaction["basis"] == "after_market_next_day"
    assert reaction["pre_event_date"] == fx["days"][20].isoformat()
    assert reaction["react_date"] == fx["days"][21].isoformat()
    pre, react = fx["closes"][20], fx["closes"][21]
    assert reaction["pre_event_close"] == pre
    assert math.isclose(reaction["returns"]["1D"], react / pre - 1, rel_tol=1e-9)


async def test_before_market_event_reacts_on_the_same_dated_bar(client):
    """BMO on D: pre = D-1's close, react = D's own bar. Asserted against the
    SAME seeded history as the AMC test, so the two answers differ by exactly
    the session — an implementation ignoring ``session`` cannot pass both."""
    days = await _seed_bars(
        "BMO", start=date(2026, 1, 5), closes=_flat_closes(30, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(30, 500.0, 0.1))
    await _add_event(
        key=f"EARNINGS:BMO:{days[10].isoformat()}",
        ticker="BMO",
        when=_et(days[10].year, days[10].month, days[10].day, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:BMO:{later.isoformat()}",
        ticker="BMO",
        when=_et(later.year, later.month, later.day, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    reaction = body["previous_events"][0]["reaction"]
    assert reaction["basis"] == "before_market_same_day"
    assert reaction["pre_event_date"] == days[9].isoformat()
    assert reaction["react_date"] == days[10].isoformat()


async def test_gap_return_and_1d_return_are_different_measurements(client):
    """The gap (react OPEN vs pre close) and 1D (react CLOSE vs pre close)
    answer different questions; the seeded opens sit 1% below the closes so a
    payload that conflated them would be caught here."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    reaction = body["previous_events"][0]["reaction"]
    pre = fx["closes"][20]
    assert math.isclose(
        reaction["gap_return"], (fx["closes"][21] * 0.99) / pre - 1, rel_tol=1e-9
    )
    assert reaction["gap_return"] != reaction["returns"]["1D"]


async def test_every_reported_horizon_is_present_and_labelled(client):
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    assert body["horizons"] == ["1D", "3D", "5D", "10D"]
    returns = body["previous_events"][0]["reaction"]["returns"]
    assert set(returns) == {"1D", "3D", "5D", "10D"}
    for k in ("1D", "3D", "5D", "10D"):
        assert returns[k] is not None


async def test_abnormal_return_is_the_stock_minus_spy_on_the_same_window(client):
    """§17: the benchmark is measured over the stock's OWN calendar window,
    so the abnormal 1D is exactly stock_1D - spy_1D and nothing else."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    past = body["previous_events"][0]
    abnormal = past["abnormal_vs_spy"]
    assert abnormal["benchmark"] == "SPY"
    assert abnormal["benchmark_available"] is True
    stock_1d = past["reaction"]["returns"]["1D"]
    bench_1d = abnormal["benchmark_returns"]["1D"]
    assert math.isclose(abnormal["abnormal"]["1D"], stock_1d - bench_1d, rel_tol=1e-9)
    # And the benchmark genuinely moved, so the subtraction is not a no-op.
    assert bench_1d != 0.0


# ---------------------------------------------------------------------------
# 4. The as-of gate (§14, §96) — every assertion PAIRED
# ---------------------------------------------------------------------------


async def test_as_of_at_1559_et_excludes_that_days_bar(client):
    """PAIRED with the 16:00 test below. At 15:59 the session has not closed,
    so the day's daily bar does not exist yet."""
    fx = await _standard_fixture()
    last = fx["days"][-1]
    at_1559 = datetime(last.year, last.month, last.day, 15, 59, tzinfo=EASTERN)
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": at_1559.isoformat()},
        )
    ).json()
    assert body["data_freshness"]["bars_through"] == fx["days"][-2].isoformat()
    assert body["data_freshness"]["n_bars"] == 59


async def test_as_of_at_1600_et_includes_that_days_bar(client):
    """The other half of the pair: a gate that simply dropped the last bar
    always would pass the 15:59 test and fail this one."""
    fx = await _standard_fixture()
    last = fx["days"][-1]
    at_1600 = datetime(last.year, last.month, last.day, 16, 0, tzinfo=EASTERN)
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": at_1600.isoformat()},
        )
    ).json()
    assert body["data_freshness"]["bars_through"] == last.isoformat()
    assert body["data_freshness"]["n_bars"] == 60


async def test_the_as_of_gate_is_applied_to_the_spy_bars_too(client):
    """PAIRED with the test below. The contract requires the look-ahead gate on
    BOTH bar lists: abnormal returns are stock MINUS SPY, so a benchmark that
    still sees the closing bar at 15:59 leaks the future into every abnormal
    number even when the stock side is gated correctly. Pinning the benchmark's
    own freshness is the only way that asymmetry shows up as a failure."""
    fx = await _standard_fixture()
    last = fx["days"][-1]
    at_1559 = datetime(last.year, last.month, last.day, 15, 59, tzinfo=EASTERN)
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": at_1559.isoformat()},
        )
    ).json()
    freshness = body["data_freshness"]
    assert freshness["benchmark"] == "SPY"
    assert freshness["benchmark_bars_through"] == fx["days"][-2].isoformat()
    # The benchmark is gated to exactly the same instant as the stock.
    assert freshness["benchmark_bars_through"] == freshness["bars_through"]


async def test_the_spy_gate_opens_at_the_close_like_the_stock_gate(client):
    """The other half: a benchmark list truncated unconditionally would pass
    the 15:59 assertion above and fail here."""
    fx = await _standard_fixture()
    last = fx["days"][-1]
    at_1600 = datetime(last.year, last.month, last.day, 16, 0, tzinfo=EASTERN)
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": at_1600.isoformat()},
        )
    ).json()
    freshness = body["data_freshness"]
    assert freshness["benchmark_bars_through"] == last.isoformat()
    assert freshness["benchmark_bars_through"] == freshness["bars_through"]


async def test_as_of_before_the_react_bar_leaves_the_reaction_unavailable(client):
    """Look-ahead, the direct form: standing at the past event's own evening,
    the market has not yet reacted, and the payload must say so rather than
    reporting the reaction the platform can see today."""
    fx = await _standard_fixture()
    event_day = fx["days"][20]
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(event_day)},
        )
    ).json()
    past = body["previous_events"][0]
    assert past["bars_available"] is False
    assert "no bar after the event yet" in past["reason"]
    assert past["reaction"]["returns"]["1D"] is None


async def test_the_same_event_resolves_once_as_of_passes_its_react_bar(client):
    """The paired half: proves the previous test's absence is the GATE, not a
    permanently broken code path."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][21])},
        )
    ).json()
    past = body["previous_events"][0]
    assert past["bars_available"] is True
    assert past["reaction"]["returns"]["1D"] is not None
    # 3D still needs bars the caller could not have seen at that instant.
    assert past["reaction"]["returns"]["3D"] is None
    assert (
        past["reaction"]["reasons"]["return_3D"] == "insufficient_bars_after_event"
    )


async def test_an_event_scheduled_after_as_of_is_not_history_yet(client):
    """A registry row the platform knows about TODAY was not knowable at a
    past ``as_of`` — point-in-time applies to the event pool, not only to the
    bars."""
    fx = await _standard_fixture()
    # as_of one day before the past print: it must not appear at all.
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][19])},
        )
    ).json()
    assert body["previous_events"] == []
    assert body["anchor_event"] is None


async def test_the_same_event_appears_once_as_of_reaches_it(client):
    """Paired with the test above."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][21])},
        )
    ).json()
    assert [e["event_id"] for e in body["previous_events"]] == [fx["past_id"]]


async def test_pre_event_context_is_measured_only_through_as_of(client):
    """The run-up is a function of the bars the caller could see, so moving
    as_of back must move ``last_close`` back with it."""
    fx = await _standard_fixture()
    late = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()["pre_event"]
    early = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][40])},
        )
    ).json()["pre_event"]
    assert late["last_close"] == fx["closes"][-1]
    assert early["last_close"] == fx["closes"][40]
    assert early["run_up_pct"] != late["run_up_pct"]


# ---------------------------------------------------------------------------
# 5. Honest absence — never a zero
# ---------------------------------------------------------------------------


async def test_event_older_than_the_bar_history_says_so_with_the_boundary_date(
    client,
):
    """The 48 seeded CONFIRMED earnings rows go back to Nov-2023 while the
    bars start in 2024 — the oldest prints per ticker MUST render as
    "bars unavailable before <first bar>" and never as a 0.0% reaction."""
    days = await _seed_bars(
        "OLD", start=date(2026, 1, 5), closes=_flat_closes(30, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(30, 500.0, 0.1))
    await _add_event(
        key="EARNINGS:OLD:2025-11-04",
        ticker="OLD",
        when=_et(2025, 11, 4, 16, 30),
    )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:OLD:{later.isoformat()}",
        ticker="OLD",
        when=_et(later.year, later.month, later.day, 16, 30),
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    past = body["previous_events"][0]
    assert past["bars_available"] is False
    assert past["reason"] == f"bars unavailable before {days[0].isoformat()}"
    assert past["reaction"]["returns"]["1D"] is None
    assert past["reaction"]["gap_return"] is None


async def test_unconfigured_provider_leaves_the_endpoint_at_200_with_a_reason(
    unconfigured_client,
):
    """A 503 here would hide the event's real registry facts behind a quote
    vendor's absence. The bars block explains itself instead."""
    upcoming = await _add_event(
        key="EARNINGS:NOPROV:2026-06-10",
        ticker="NOPROV",
        when=_et(2026, 6, 10, 16, 30),
    )
    r = await unconfigured_client.get(
        f"/api/events/{upcoming}/price-context",
        params={"as_of": _et(2026, 6, 9, 17, 0).isoformat()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["bars"]["available"] is False
    assert body["bars"]["reason"]
    assert any(entry["field"] == "bars" for entry in body["unavailable"])


async def test_unconfigured_provider_still_reports_the_registry_identity(
    unconfigured_client,
):
    upcoming = await _add_event(
        key="EARNINGS:NOPROV2:2026-06-10",
        ticker="NOPROV2",
        when=_et(2026, 6, 10, 16, 30),
    )
    body = (
        await unconfigured_client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _et(2026, 6, 9, 17, 0).isoformat()},
        )
    ).json()
    assert body["event_id"] == upcoming
    assert body["ticker"] == "NOPROV2"
    assert body["provenance"]["bars"] == "DATA"


async def test_short_history_reports_sma200_as_null_with_a_reason(client):
    """A 200-day SMA over 30 bars is not a 200-day SMA. The null must travel
    with the arithmetic that explains it."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    pre = body["pre_event"]
    assert pre["sma200"] is None
    assert "needs 200 bars" in pre["reasons"]["sma200"]
    # And the same fact is reachable from the flat list the UI renders from.
    fields = {entry["field"] for entry in body["unavailable"]}
    assert "pre_event.sma200" in fields
    # The DISTANCE tile is the one the UI actually renders, so it must carry
    # an explanation of its own rather than leaving the user to notice that a
    # different field is also missing.
    assert pre["sma200_distance_pct"] is None
    assert "needs 200 bars" in pre["reasons"]["sma200_distance_pct"]
    assert "pre_event.sma200_distance_pct" in fields


async def test_every_null_pre_event_metric_carries_a_reason(client):
    """The house rule, asserted structurally: no metric may be null in
    silence. Checked over the numeric fields, with a floor on how many were
    actually examined so an empty-payload regression cannot pass vacuously."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    pre = body["pre_event"]
    reasons = pre["reasons"]
    checked = 0
    nulls = 0
    for key, value in pre.items():
        if key in ("reasons", "anchor_basis", "benchmark") or key.endswith(
            ("_date_et", "_through")
        ):
            continue
        checked += 1
        if value is None:
            nulls += 1
            # Keyed on the field ITSELF, not on any key that merely contains
            # its name: "sma200" must not be allowed to excuse a bare
            # "sma200_distance_pct".
            assert key in reasons, f"{key} is null with no reason of its own"
            assert reasons[key], f"{key} has an empty reason"
    assert checked >= 20
    # A payload where nothing happened to be null would satisfy the loop
    # vacuously; this history is 60 bars, so sma200 and friends ARE null.
    assert nulls >= 1


async def test_no_metric_is_ever_nan_or_infinite(client):
    """JSON has no NaN; a serialiser that emitted one would produce invalid
    JSON, and a Python float('inf') would silently become Infinity. Every
    number in the payload is asserted finite."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    seen = 0

    def walk(node):
        nonlocal seen
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            seen += 1
            assert math.isfinite(node), node

    walk(body)
    assert seen >= 30


# ---------------------------------------------------------------------------
# 6. §15 comparability and §31/§32 anchoring
# ---------------------------------------------------------------------------


async def test_an_estimated_past_date_is_not_a_comparable_event(client):
    """§15: an ESTIMATED date is a cadence DERIVATION. Measuring a market
    reaction around a day nobody reported on would be a fabricated number
    wearing a measurement's clothes."""
    days = await _seed_bars(
        "EST", start=date(2026, 1, 5), closes=_flat_closes(40, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(40, 500.0, 0.1))
    await _add_event(
        key=f"EARNINGS:EST:{days[10].isoformat()}",
        ticker="EST",
        when=_et(days[10].year, days[10].month, days[10].day, 16, 30),
        status=EventStatus.ESTIMATED,
    )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:EST:{later.isoformat()}",
        ticker="EST",
        when=_et(later.year, later.month, later.day, 16, 30),
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    assert body["previous_events"] == []
    assert body["anchor_event"] is None


async def test_a_revised_past_date_IS_a_comparable_event(client):
    """The paired half of the ESTIMATED test: REVISED is a confirmed date that
    moved, and it stays comparable."""
    days = await _seed_bars(
        "REV", start=date(2026, 1, 5), closes=_flat_closes(40, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(40, 500.0, 0.1))
    past = await _add_event(
        key=f"EARNINGS:REV:{days[10].isoformat()}",
        ticker="REV",
        when=_et(days[10].year, days[10].month, days[10].day, 16, 30),
        status=EventStatus.REVISED,
    )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:REV:{later.isoformat()}",
        ticker="REV",
        when=_et(later.year, later.month, later.day, 16, 30),
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    assert [e["event_id"] for e in body["previous_events"]] == [past]


async def test_another_tickers_earnings_never_enters_the_history(client):
    fx = await _standard_fixture()
    await _seed_bars(
        "OTHER", start=date(2026, 1, 5), closes=_flat_closes(60, 200.0, 1.0)
    )
    await _add_event(
        key=f"EARNINGS:OTHER:{fx['days'][15].isoformat()}",
        ticker="OTHER",
        when=_et(
            fx["days"][15].year, fx["days"][15].month, fx["days"][15].day, 16, 30
        ),
    )
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    tickers = {e["event_key"].split(":")[1] for e in body["previous_events"]}
    assert tickers == {"ACME"}


async def test_run_up_is_anchored_on_the_previous_print_when_one_exists(client):
    """§32's framing is "since the last print", so the anchor basis must say
    ``previous_event`` and the anchor date must BE that print's pre-event
    date — not a generic 3-month lookback."""
    fx = await _standard_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    pre = body["pre_event"]
    assert pre["anchor_basis"] == "previous_event"
    assert pre["anchor_date_et"] == fx["days"][20].isoformat()
    assert math.isclose(
        pre["run_up_pct"], fx["closes"][-1] / fx["closes"][20] - 1, rel_tol=1e-9
    )
    assert body["anchor_event"]["event_id"] == fx["past_id"]


async def test_with_no_previous_print_the_anchor_says_it_is_a_default_window(
    client,
):
    """Rendering a plain 63-bar lookback as "since the last earnings" would be
    a fabricated claim about the measurement itself."""
    days = await _seed_bars(
        "FIRST", start=date(2026, 1, 5), closes=_flat_closes(40, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(40, 500.0, 0.1))
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:FIRST:{later.isoformat()}",
        ticker="FIRST",
        when=_et(later.year, later.month, later.day, 16, 30),
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    assert body["previous_events"] == []
    assert body["anchor_event"] is None
    assert body["pre_event"]["anchor_basis"] == "default_63_bars"


async def test_spy_relative_return_is_the_arithmetic_difference(client):
    """§32's "vs SPY" is a difference of returns, not a ratio."""
    fx = await _standard_fixture()
    pre = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()["pre_event"]
    assert pre["benchmark"] == "SPY"
    assert math.isclose(
        pre["relative_return"],
        pre["since_anchor_return"] - pre["benchmark_return"],
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# 7. §19/§64 history statistics — sample size always travels
# ---------------------------------------------------------------------------


async def _multi_event_fixture(ticker: str = "MULTI") -> dict:
    """Five past AMC prints spread over a 60-session history, so last4 and
    last8 have genuinely different sample sizes."""
    closes = _flat_closes(60, 100.0, 0.5)
    for idx, bump in ((6, 4.0), (16, -3.0), (26, 5.0), (36, -2.0), (46, 6.0)):
        for j in range(idx, 60):
            closes[j] += bump
    days = await _seed_bars(ticker, start=date(2026, 1, 5), closes=closes)
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(60, 500.0, 0.1))
    past_ids = []
    for idx in (5, 15, 25, 35, 45):
        past_ids.append(
            await _add_event(
                key=f"EARNINGS:{ticker}:{days[idx].isoformat()}",
                ticker=ticker,
                when=_et(days[idx].year, days[idx].month, days[idx].day, 16, 30),
            )
        )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:{ticker}:{later.isoformat()}",
        ticker=ticker,
        when=_et(later.year, later.month, later.day, 16, 30),
    )
    return {"days": days, "past_ids": past_ids, "upcoming_id": upcoming}


async def test_history_stats_are_reported_for_1d_and_5d(client):
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    stats = body["history_stats"]
    assert set(stats) == {"1D", "5D"}
    assert set(stats["1D"]) == {"last4", "last8", "last12"}


async def test_history_stats_always_carry_the_sample_size(client):
    """§19/§64: a distribution without its n is an unfalsifiable claim."""
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    last8 = body["history_stats"]["1D"]["last8"]
    assert last8["n"] == 8
    assert last8["n_available"] == 5  # only five prints actually exist
    assert last8["horizon"] == "1D"


async def test_positive_frequency_travels_with_its_count_not_as_a_probability(
    client,
):
    """"5/8" is a historical count; a bare 0.625 reads as a forecast (§64)."""
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    last8 = body["history_stats"]["1D"]["last8"]
    assert last8["positive_count"] is not None
    assert 0 <= last8["positive_count"] <= last8["n_available"]
    assert math.isclose(
        last8["positive_frequency"],
        last8["positive_count"] / last8["n_available"],
        rel_tol=1e-9,
    )


async def test_a_single_usable_event_is_insufficient_sample_not_a_median(client):
    """A "median" of one print is a single number in a statistic's clothes."""
    fx = await _standard_fixture()  # exactly one past print
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    last4 = body["history_stats"]["1D"]["last4"]
    assert last4["n_available"] == 1
    assert last4["median_abs"] is None
    assert "insufficient_sample" in last4["reasons"]["sample"]


async def test_history_stats_are_absolute_move_distributions(client):
    """The published statistics are |move| distributions: p90 >= median >= 0
    for any sample, which a signed distribution would violate."""
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    last8 = body["history_stats"]["1D"]["last8"]
    assert last8["median_abs"] >= 0.0
    assert last8["p90_abs"] >= last8["median_abs"]
    assert last8["max_abs"] >= last8["p90_abs"]


async def test_history_stats_shrink_when_as_of_hides_later_prints(client):
    """The statistics are themselves point-in-time: at an earlier as_of fewer
    prints have happened, and n_available must fall accordingly."""
    fx = await _multi_event_fixture()
    late = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()["history_stats"]["1D"]["last8"]
    early = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][30])},
        )
    ).json()["history_stats"]["1D"]["last8"]
    assert late["n_available"] == 5
    assert early["n_available"] < late["n_available"]


# ---------------------------------------------------------------------------
# 8. Ordering and the previous-comparable flag
# ---------------------------------------------------------------------------


async def test_previous_events_are_ordered_oldest_first(client):
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    dates = [e["date_et"] for e in body["previous_events"]]
    assert dates == sorted(dates)
    assert len(dates) == 5


async def test_exactly_one_previous_event_is_flagged_as_the_comparable(client):
    """§15 names ONE precedent (the nearest qualifying print) and says why;
    the rest are history for the table."""
    fx = await _multi_event_fixture()
    body = (
        await client.get(
            f"/api/events/{fx['upcoming_id']}/price-context",
            params={"as_of": _after_close(fx["days"][-1])},
        )
    ).json()
    flagged = [e for e in body["previous_events"] if e["is_previous_comparable"]]
    assert len(flagged) == 1
    assert flagged[0]["event_id"] == fx["past_ids"][-1]
    assert body["anchor_event"]["comparison_reason"]


async def test_bmo_run_up_is_anchored_on_the_pre_print_close_not_the_print_day(
    client,
):
    """A BMO print on D is priced INTO D's own bar, so D's close already
    contains that print's reaction. Anchoring the run-up there would measure
    "since the last earnings" from a close that already moved on the news —
    understating, or inverting, the very move the tile claims to report. The
    honest anchor is D-1's close, which is the reaction's own
    ``pre_event_date``.

    The seeded history steps +1/day, so the two candidate anchors differ by a
    known amount and an implementation using the event DATE would land on the
    other number rather than merely being imprecise.
    """
    days = await _seed_bars(
        "BMOANC", start=date(2026, 1, 5), closes=_flat_closes(40, 100.0, 1.0)
    )
    await _seed_bars("SPY", start=date(2026, 1, 5), closes=_flat_closes(40, 500.0, 0.1))
    await _add_event(
        key=f"EARNINGS:BMOANC:{days[10].isoformat()}",
        ticker="BMOANC",
        when=_et(days[10].year, days[10].month, days[10].day, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    later = days[-1] + timedelta(days=7)
    upcoming = await _add_event(
        key=f"EARNINGS:BMOANC:{later.isoformat()}",
        ticker="BMOANC",
        when=_et(later.year, later.month, later.day, 7, 0),
        session=EventSession.BEFORE_MARKET,
    )
    body = (
        await client.get(
            f"/api/events/{upcoming}/price-context",
            params={"as_of": _after_close(days[-1])},
        )
    ).json()
    pre = body["pre_event"]
    assert pre["anchor_basis"] == "previous_event"
    # days[9], the last close BEFORE the BMO print — not days[10].
    assert pre["anchor_date_et"] == days[9].isoformat()
    assert pre["anchor_close"] == 109.0
    last_close = 100.0 + 39.0
    assert math.isclose(pre["run_up_pct"], last_close / 109.0 - 1, rel_tol=1e-9)
    # And the event-date anchor would have produced a DIFFERENT number, so
    # this assertion actually discriminates between the two implementations.
    assert not math.isclose(
        pre["run_up_pct"], last_close / 110.0 - 1, rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# Macro events get a comparable pool too (found live 2026-08-22)
#
# ``_past_comparable_rows`` was pinned to ``event_type == EARNINGS`` and
# returned [] whenever the event had no ticker. Every macro release therefore
# reached ``previous_comparable`` with an empty pool and reported "no
# comparable event" — true of the rows it was handed, false of the world, for
# series published for decades. The pure resolver was always correct; the
# gateway seam never let it see a candidate.
# ---------------------------------------------------------------------------


async def _set_release_period(event_id: int, period: str) -> None:
    async with SessionLocal() as s:
        row = await s.get(EventRow, event_id)
        row.release_period = period
        await s.commit()


async def test_macro_event_resolves_a_previous_comparable_release(client):
    """A GDP release must anchor on the prior GDP release, not answer 'none'."""
    prior = await _add_event(
        key="GDP:2026-06-25",
        ticker=None,
        when=datetime(2026, 6, 25, 12, 30, tzinfo=timezone.utc),
        event_type=EventType.GDP,
        session=EventSession.BEFORE_MARKET,
        title="GDP (Third Estimate), 1st Quarter 2026",
    )
    await _set_release_period(prior, "2026-Q1")
    current = await _add_event(
        key="GDP:2026-07-30",
        ticker=None,
        when=datetime(2026, 7, 30, 12, 30, tzinfo=timezone.utc),
        event_type=EventType.GDP,
        session=EventSession.BEFORE_MARKET,
        title="GDP (Advance Estimate), 2nd Quarter 2026",
    )
    await _set_release_period(current, "2026-Q2")

    res = await client.get(
        f"/api/events/{current}/price-context",
        params={"as_of": _after_close(date(2026, 6, 30))},
    )
    assert res.status_code == 200
    anchor = res.json().get("anchor_event") or {}
    assert anchor.get("event_id") == prior


async def test_macro_comparable_pool_never_crosses_event_type(client):
    """§15: a PPI release is not a precedent for a GDP release, even though
    both are macro, both are tickerless, and both sit in the same window."""
    ppi = await _add_event(
        key="PPI:2026-06-13",
        ticker=None,
        when=datetime(2026, 6, 13, 12, 30, tzinfo=timezone.utc),
        event_type=EventType.PPI,
        session=EventSession.BEFORE_MARKET,
        title="Producer Price Index, July 2026",
    )
    await _set_release_period(ppi, "2026-05")
    current = await _add_event(
        key="GDP:2026-07-30",
        ticker=None,
        when=datetime(2026, 7, 30, 12, 30, tzinfo=timezone.utc),
        event_type=EventType.GDP,
        session=EventSession.BEFORE_MARKET,
        title="GDP (Advance Estimate), 2nd Quarter 2026",
    )
    await _set_release_period(current, "2026-Q2")

    res = await client.get(
        f"/api/events/{current}/price-context",
        params={"as_of": _after_close(date(2026, 6, 30))},
    )
    assert res.status_code == 200
    assert not (res.json().get("anchor_event") or {}).get("event_id")


async def test_earnings_comparable_pool_still_requires_the_same_ticker(client):
    """Loosening the type pin must not loosen ticker scoping: another
    company's print is not this company's precedent."""
    other = await _add_event(
        key="EARNINGS:DELL:2026-05-28",
        ticker="DELL",
        when=datetime(2026, 5, 28, 20, 5, tzinfo=timezone.utc),
        event_type=EventType.EARNINGS,
    )
    current = await _add_event(
        key="EARNINGS:HPE:2026-07-02",
        ticker="HPE",
        when=datetime(2026, 7, 2, 20, 5, tzinfo=timezone.utc),
        event_type=EventType.EARNINGS,
    )
    res = await client.get(
        f"/api/events/{current}/price-context",
        params={"as_of": _after_close(date(2026, 6, 30))},
    )
    assert res.status_code == 200
    anchor = res.json().get("anchor_event") or {}
    assert anchor.get("event_id") != other
    assert not anchor.get("event_id")
