"""Fundamentals API — GET /api/events/{id}/fundamentals (event spec §16, §28,
§29, §30, §33, §35, §85, §96; audit §7, §11.3 Phase E2).

WHY THE STATEMENTS ARE SEEDED, NOT FETCHED. Almost every test below inserts
its own ``fundamental_statements`` rows before calling the endpoint.
``ensure_fundamentals`` tops the mirror up only when a refresh is DUE, so
seeding a fresh row both pins the arithmetic to hand-computable numbers AND
keeps the stub provider's synthetic series out of the assertions — a test
whose expected gross margin comes from the same generator that produced the
revenue proves nothing about the formula. The tests that deliberately seed
NOTHING are the ingestion tests, which are precisely about the fetch path.

The guarantees these tests defend, in the order they appear:

1. **The as-of gate is on ``acceptance_datetime`` and nothing else** (§85,
   §96; audit §7.1). Always a PAIRED assertion, never one half: a filing
   accepted one hour after ``as_of`` is invisible AND the same filing is
   visible one hour later. A gate that returned nothing would pass the first
   half and fail the second. The companion pair plants a quarter whose PERIOD
   ended long before ``as_of`` but whose acceptance is after it — the exact
   look-ahead an ``end_date`` filter would let through — and proves it stays
   out.
2. **The previous column is what was known THEN.** The previous snapshot is
   taken at the previous comparable event's own instant, so a filing accepted
   between the two events appears in CURRENT and not in PREVIOUS, and the
   delta describes a change that actually happened.
3. **Honest absence, never a zero** (§44 rule 18, §85, §33). A metric whose
   input the provider does not report is ``null`` WITH a reason; a ticker
   with no filings is ``available: false`` at 200; a macro event says
   ``no_ticker``; an unconfigured provider still answers 200. Consensus is
   ``available: false`` in EVERY payload, including the fully-populated one —
   §33/§98 make it unavailable at any instant, not merely un-backtestable.
4. **The price leg is gated too.** The multiples are priced off the last close
   knowable at ``as_of`` under the §14 16:00-ET rule, so a P/E in an answer
   dated last October cannot be computed from today's close.
5. **Ingestion is idempotent and throttled.** Re-fetching the same periods
   writes no duplicate rows (the natural key) and no second audit event; a
   restatement of the same period OVERWRITES with the later acceptance
   instant (audit §7.3).

Uses the shared ``client`` / ``unconfigured_client`` fixtures (conftest.py).
"""
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from apps.gateway import fundamentals as fundamentals_seam
from apps.gateway.db import (
    AuditEvent,
    EventRow,
    FundamentalStatementRow,
    SessionLocal,
    StockBarDaily,
)
from libs.trading_core.events.fundamentals import METRIC_ORDER
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


def _utc(y: int, m: int, d: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(y, m, d, hour, minute, tzinfo=timezone.utc)


#: Hand-built statement values whose ratios are exact decimals. Revenue 1000,
#: gross profit 400 and net income 100 make gross margin 0.40 and net margin
#: 0.10 with no floating-point argument, so a test that asserts 0.40 is
#: asserting the FORMULA and not the seed.
def _values(
    *,
    revenue: float = 1000.0,
    gross_profit: float = 400.0,
    operating_income: float = 250.0,
    net_income: float = 100.0,
    eps: float = 1.0,
    shares: float = 100.0,
    ocf: float = 180.0,
    assets: float = 2000.0,
    current_assets: float = 800.0,
    current_liabilities: float = 400.0,
    equity: float = 1000.0,
    long_term_debt: float = 500.0,
) -> dict:
    """The flattened provider field names, with ONLY the fields Massive really
    reports — no cash, no capex, no D&A (contract §3). A helper that filled
    those in would let free_cash_flow pass here and fail in production."""
    return {
        "income_statement.revenues": revenue,
        "income_statement.gross_profit": gross_profit,
        "income_statement.operating_income_loss": operating_income,
        "income_statement.net_income_loss": net_income,
        "income_statement.diluted_earnings_per_share": eps,
        "income_statement.diluted_average_shares": shares,
        "cash_flow_statement.net_cash_flow_from_operating_activities": ocf,
        "balance_sheet.assets": assets,
        "balance_sheet.current_assets": current_assets,
        "balance_sheet.current_liabilities": current_liabilities,
        "balance_sheet.equity": equity,
        "balance_sheet.long_term_debt": long_term_debt,
    }


async def _seed_statement(
    ticker: str,
    *,
    fiscal_year: int,
    fiscal_period: str,
    end_date: date,
    accepted: datetime,
    timeframe: str = "quarterly",
    values: dict | None = None,
    filing_date: date | None = None,
    source_filing_url: str | None = "https://example.test/filing",
    fetched_at: datetime | None = None,
) -> int:
    """Insert one statement row and return its id."""
    async with SessionLocal() as s:
        row = FundamentalStatementRow(
            ticker=ticker,
            cik="0000000001",
            timeframe=timeframe,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            start_date=end_date - timedelta(days=90),
            end_date=end_date,
            filing_date=(
                filing_date
                if filing_date is not None
                else (accepted.date() if accepted is not None else None)
            ),
            acceptance_datetime=accepted,
            source_filing_url=source_filing_url,
            values=values if values is not None else _values(),
            raw_fields_count=len(values if values is not None else _values()),
            fetched_at=fetched_at or _utc(2026, 1, 1),
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_bars(
    ticker: str, *, start: date, closes: list[float], volume: float = 1_000_000.0
) -> list[date]:
    """One bar per weekday from ``start``. ``open`` is 1% below ``close`` so a
    test can tell which one an implementation reported."""
    days: list[date] = []
    day = start
    while len(days) < len(closes):
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    async with SessionLocal() as s:
        for when, close in zip(days, closes):
            s.add(
                StockBarDaily(
                    ticker=ticker,
                    ts=when,
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
            session=session.value,
            status=status.value,
            source=EventSourceKind.STRUCTURED_PROVIDER.value,
            source_name="test",
        )
        s.add(row)
        await s.commit()
        return row.id


def _metric(payload: dict, name: str):
    return payload["current"]["metrics"][name]


def _change(payload: dict, name: str) -> dict:
    return next(c for c in payload["changes"] if c["metric"] == name)


@pytest.fixture(autouse=True)
def _clear_fundamentals_throttle():
    """The seam throttles provider ATTEMPTS per ticker in a process-local
    dict. Left alone, the first test to touch a ticker would suppress the
    fetch in every later test — an ordering dependency that would make a
    green suite meaningless. Cleared on both sides so neither direction
    leaks."""
    fundamentals_seam._refresh_attempts.clear()
    yield
    fundamentals_seam._refresh_attempts.clear()


@pytest.fixture
def seeded_only(monkeypatch):
    """Serve ONLY the statements a test seeded — no provider top-up.

    A seeded quarter's acceptance instant is deliberately months old (that is
    what the as-of tests are about), so ``_refresh_is_due`` is permanently
    true for it and ``ensure_fundamentals`` would mirror the stub's synthetic
    series in alongside it. The synthetic revenue then wins "newest quarter"
    and every hand-computed margin below becomes an assertion about the stub's
    random walk instead of about the formula — the same trap
    ``tests/test_events_price_api.py`` avoids by seeding its bars.

    Patching the PROVIDER rather than ``ensure_fundamentals`` keeps the real
    ingest path (the stored-rows read, the refresh predicate, the throttle) in
    the call graph for every one of these tests; only the vendor's answer is
    empty. The ingestion tests in section 7 deliberately do NOT take this
    fixture — they are about the fetch path itself.
    """

    class _NoFilings:
        def get_financials(self, ticker, *, timeframe="quarterly", limit=12):
            return []

    monkeypatch.setattr(fundamentals_seam, "get_provider", lambda name: _NoFilings())


# ---------------------------------------------------------------------------
# 1. The as-of gate — always paired
# ---------------------------------------------------------------------------


async def test_filing_accepted_one_hour_after_as_of_is_invisible(client, seeded_only):
    """§96's sentinel: a filing accepted at T+1h must not inform T."""
    await _seed_statement(
        "AOF",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 30, 21, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:AOF:2025-10-30", ticker="AOF", when=_et(2025, 10, 30, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-10-30T20:00:00%2B00:00"
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is False
    assert payload["current"]["metrics"]["revenue"] is None
    assert payload["current"]["reasons"]["revenue"]


async def test_the_same_filing_is_visible_one_hour_later(client, seeded_only):
    """The other half of the pair: the gate must ADMIT, not merely refuse.

    Without this assertion a seam that always returned nothing would look
    point-in-time correct.
    """
    await _seed_statement(
        "AOF2",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 30, 21, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:AOF2:2025-10-30", ticker="AOF2", when=_et(2025, 10, 30, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-10-30T22:00:00%2B00:00"
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is True
    assert _metric(payload, "revenue") == 1000.0


async def test_gate_ignores_period_end_entirely(client, seeded_only):
    """THE look-ahead an ``end_date`` filter would let through.

    The period ended eight months before ``as_of``, so any implementation
    filtering on the fiscal period would happily serve it — but it was not
    ACCEPTED until a day later, and on the as-of date it did not exist.
    """
    await _seed_statement(
        "LATE",
        fiscal_year=2025,
        fiscal_period="Q1",
        end_date=date(2025, 3, 31),
        accepted=_utc(2025, 12, 2, 12, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:LATE:2025-12-01", ticker="LATE", when=_et(2025, 12, 1, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-12-01T23:00:00%2B00:00"
    )
    payload = r.json()
    assert payload["available"] is False
    assert _metric(payload, "revenue") is None

    later = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-12-03T00:00:00%2B00:00"
    )
    assert later.json()["available"] is True


async def test_row_without_acceptance_datetime_is_excluded_with_a_reason(client, seeded_only):
    """An unknown publication time cannot be proven to precede as_of.

    Admitting it "because the period is old" is the same leak through the
    back door, so the row is stored (it is real) and refused with a stated
    reason rather than silently dropped.
    """
    await _seed_statement(
        "NOACC",
        fiscal_year=2025,
        fiscal_period="Q2",
        end_date=date(2025, 6, 30),
        accepted=None,
    )
    event_id = await _add_event(
        key="EARNINGS:NOACC:2025-11-01", ticker="NOACC", when=_et(2025, 11, 1, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2026-01-01T00:00:00%2B00:00"
    )
    payload = r.json()
    assert payload["available"] is False
    # The row IS stored — the absence is about the gate, not about ingestion.
    assert payload["statements"]["available"] is True
    assert payload["statements"]["count"] == 1
    joined = " ".join(payload["current"]["reasons"].values())
    assert "acceptance_datetime" in joined


async def test_only_the_newest_visible_quarter_drives_the_snapshot(client, seeded_only):
    """Two quarters visible; the snapshot must use the LATER period."""
    await _seed_statement(
        "TWOQ",
        fiscal_year=2025,
        fiscal_period="Q2",
        end_date=date(2025, 6, 30),
        accepted=_utc(2025, 7, 25, 20, 0),
        values=_values(revenue=900.0),
    )
    await _seed_statement(
        "TWOQ",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 25, 20, 0),
        values=_values(revenue=1000.0),
    )
    event_id = await _add_event(
        key="EARNINGS:TWOQ:2025-11-01", ticker="TWOQ", when=_et(2025, 11, 1, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-11-01T23:00:00%2B00:00"
    )
    payload = r.json()
    assert _metric(payload, "revenue") == 1000.0
    assert payload["current"]["quarterly"]["fiscal_period"] == "Q3"
    assert payload["current"]["quarterly"]["acceptance_datetime"].startswith(
        "2025-10-25T20:00"
    )


# ---------------------------------------------------------------------------
# 2. Previous-vs-current — the previous column is what was known THEN
# ---------------------------------------------------------------------------


async def _seed_two_quarter_pair(ticker: str) -> tuple[int, int]:
    """Q2 accepted before the July print, Q3 accepted before the October one.

    Returns ``(previous_event_id, current_event_id)``. The two revenues differ
    by exactly 100 on a base of 900, so the delta and the percent change are
    both exact.
    """
    await _seed_statement(
        ticker,
        fiscal_year=2025,
        fiscal_period="Q2",
        end_date=date(2025, 6, 30),
        accepted=_utc(2025, 7, 24, 20, 0),
        values=_values(revenue=900.0, gross_profit=315.0, net_income=90.0, eps=0.9),
    )
    await _seed_statement(
        ticker,
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(revenue=1000.0, gross_profit=400.0, net_income=100.0, eps=1.0),
    )
    previous_id = await _add_event(
        key=f"EARNINGS:{ticker}:2025-07-25",
        ticker=ticker,
        when=_et(2025, 7, 25, 16, 30),
    )
    current_id = await _add_event(
        key=f"EARNINGS:{ticker}:2025-10-24",
        ticker=ticker,
        when=_et(2025, 10, 24, 16, 30),
    )
    return previous_id, current_id


async def test_previous_snapshot_is_taken_at_the_previous_events_instant(client, seeded_only):
    """The Q3 filing must NOT leak into the previous column.

    It was accepted in October; at the July print only Q2 existed. Reading
    "previous" off an earlier row of TODAY's data would put Q3's revenue on
    both sides and report a delta of zero.
    """
    previous_id, current_id = await _seed_two_quarter_pair("PREV")
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    payload = r.json()
    assert payload["previous_event"]["event_id"] == previous_id
    assert payload["previous_event"]["snapshot"]["metrics"]["revenue"] == 900.0
    assert _metric(payload, "revenue") == 1000.0


async def test_change_row_carries_delta_and_direction(client, seeded_only):
    """§29: delta is absolute, direction is an arrow, both travel together."""
    _, current_id = await _seed_two_quarter_pair("DELTA")
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    change = _change(r.json(), "revenue")
    assert change["previous"] == 900.0
    assert change["current"] == 1000.0
    assert change["delta"] == pytest.approx(100.0)
    assert change["direction"] == "up"
    assert change["arrow"] == "↑"


async def test_margin_change_carries_delta_bps(client, seeded_only):
    """§29's "Δ: +70 bps" — a ratio's change is reported in basis points.

    Gross margin moves 0.35 -> 0.40, which is exactly +500 bps on seeds
    chosen so the number is not an artifact of rounding.
    """
    _, current_id = await _seed_two_quarter_pair("BPS")
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    change = _change(r.json(), "gross_margin")
    assert change["previous"] == pytest.approx(0.35)
    assert change["current"] == pytest.approx(0.40)
    assert change["delta_bps"] == pytest.approx(500.0)


async def test_dollar_metric_has_a_present_but_null_delta_bps(client, seeded_only):
    """Basis points do not apply to a revenue in dollars — and the key is
    present anyway, so the UI never has to guess whether it was omitted or
    could not be computed."""
    _, current_id = await _seed_two_quarter_pair("NOBPS")
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    change = _change(r.json(), "revenue")
    assert "delta_bps" in change
    assert change["delta_bps"] is None


async def test_every_metric_in_the_canonical_order_has_a_change_row(client, seeded_only):
    """The §58 comparison table renders METRIC_ORDER; a metric missing from
    ``changes`` would silently vanish from the UI."""
    _, current_id = await _seed_two_quarter_pair("ORDER")
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    payload = r.json()
    assert [c["metric"] for c in payload["changes"]] == list(METRIC_ORDER)
    assert payload["metric_order"] == list(METRIC_ORDER)


async def test_no_previous_event_yields_rows_with_a_reason_not_an_empty_table(client, seeded_only):
    """A first-ever print still renders every metric — it just has no delta."""
    await _seed_statement(
        "FIRST",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:FIRST:2025-10-24", ticker="FIRST", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    payload = r.json()
    assert payload["previous_event"] is None
    assert len(payload["changes"]) == len(METRIC_ORDER)
    assert _change(payload, "revenue")["reason"] == "no_previous_snapshot"
    assert any(
        entry["field"] == "previous_snapshot" for entry in payload["unavailable"]
    )


async def test_an_estimated_past_date_is_not_a_comparable(client, seeded_only):
    """§15: an ESTIMATED past date is derived, not observed. Snapshotting at a
    day nobody reported on would anchor the whole comparison to a fiction."""
    await _seed_statement(
        "EST",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    await _add_event(
        key="EARNINGS:EST:2025-07-25",
        ticker="EST",
        when=_et(2025, 7, 25, 16, 30),
        status=EventStatus.ESTIMATED,
    )
    current_id = await _add_event(
        key="EARNINGS:EST:2025-10-24", ticker="EST", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
    )
    assert r.json()["previous_event"] is None


async def test_a_past_event_after_as_of_is_not_history_yet(client, seeded_only):
    """Point-in-time event selection: the registry knows about the July print
    today, but at an ``as_of`` in June it had not happened."""
    await _seed_statement(
        "FUT",
        fiscal_year=2025,
        fiscal_period="Q1",
        end_date=date(2025, 3, 31),
        accepted=_utc(2025, 4, 25, 20, 0),
    )
    await _add_event(
        key="EARNINGS:FUT:2025-07-25", ticker="FUT", when=_et(2025, 7, 25, 16, 30)
    )
    current_id = await _add_event(
        key="EARNINGS:FUT:2025-10-24", ticker="FUT", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(
        f"/api/events/{current_id}/fundamentals?as_of=2025-06-01T00:00:00%2B00:00"
    )
    assert r.json()["previous_event"] is None


# ---------------------------------------------------------------------------
# 3. Honest absence — never a zero, always a reason
# ---------------------------------------------------------------------------


async def test_provider_gaps_are_null_with_a_named_reason_never_zero(client, seeded_only):
    """Massive reports no capex, no cash and no D&A. Every metric that needs
    one is ``null`` WITH the structural reason — a 0.0 here would be a
    fabricated number wearing a computed number's clothes (§44 rule 18)."""
    await _seed_statement(
        "GAPS",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:GAPS:2025-10-24", ticker="GAPS", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    for name in ("free_cash_flow", "capex", "cash", "net_debt", "roic", "quick_ratio"):
        assert payload["current"]["metrics"][name] is None, name
        assert payload["current"]["reasons"][name], name
        assert "not reported" in payload["current"]["reasons"][name], name


async def test_every_null_metric_has_a_companion_reason(client, seeded_only):
    """The payload-wide invariant, asserted rather than merely intended."""
    await _seed_statement(
        "NULLS",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:NULLS:2025-10-24", ticker="NULLS", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    for name, value in payload["current"]["metrics"].items():
        if value is None:
            assert payload["current"]["reasons"].get(name), name


async def test_no_metric_is_a_nan_or_an_infinity(client, seeded_only):
    """A NaN serialises as the bare token ``NaN``, which is not valid JSON and
    which every consumer reads differently. Nothing here may produce one."""
    await _seed_statement(
        "FINITE",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(revenue=0.0, equity=0.0, current_liabilities=0.0),
    )
    event_id = await _add_event(
        key="EARNINGS:FINITE:2025-10-24",
        ticker="FINITE",
        when=_et(2025, 10, 24, 16, 30),
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    for name, value in payload["current"]["metrics"].items():
        if value is not None:
            assert math.isfinite(value), name


async def test_zero_revenue_makes_margins_null_not_a_division_result(client, seeded_only):
    """A zero denominator is an absence, not a margin of zero."""
    await _seed_statement(
        "ZEROREV",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(revenue=0.0),
    )
    event_id = await _add_event(
        key="EARNINGS:ZEROREV:2025-10-24",
        ticker="ZEROREV",
        when=_et(2025, 10, 24, 16, 30),
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    assert _metric(payload, "gross_margin") is None
    assert payload["current"]["reasons"]["gross_margin"]


async def test_a_ticker_with_no_filings_answers_200_with_available_false(client, seeded_only):
    """No filings is a fact about the feed, not a server error."""
    event_id = await _add_event(
        key="EARNINGS:NOFIL:2025-10-24", ticker="NOFIL", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of=2020-01-02T00:00:00%2B00:00"
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is False
    assert all(v is None for v in payload["current"]["metrics"].values())


async def test_macro_event_says_no_ticker(client, seeded_only):
    """A CPI release has no balance sheet; substituting an index proxy would
    invent an issuer the event does not have."""
    event_id = await _add_event(
        key="CPI:2025-10",
        ticker=None,
        when=_et(2025, 10, 15, 8, 30),
        event_type=EventType.CPI,
        session=EventSession.BEFORE_MARKET,
        title="CPI",
    )
    r = await client.get(f"/api/events/{event_id}/fundamentals")
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is False
    assert payload["reason"] == "no_ticker"


async def test_unconfigured_provider_still_answers_200(unconfigured_client):
    """A read endpoint must not 5xx because a vendor is unpaid — the reason
    IS the §16 capability answer, and a 503 would hide it."""
    event_id = await _add_event(
        key="EARNINGS:UNCONF:2025-10-24",
        ticker="UNCONF",
        when=_et(2025, 10, 24, 16, 30),
    )
    r = await unconfigured_client.get(f"/api/events/{event_id}/fundamentals")
    assert r.status_code == 200
    payload = r.json()
    assert payload["statements"]["available"] is False
    assert payload["statements"]["reason"]
    assert all(v is None for v in payload["current"]["metrics"].values())


async def test_consensus_is_always_unavailable_with_its_reason(client, seeded_only):
    """§33/§98 + audit §7.3: Benzinga estimates 403 at ANY instant, so the
    block is present and false even in a fully-populated payload. Omitting it
    would read as "not applicable"; computing an EPS surprise would be a
    fabrication."""
    await _seed_statement(
        "CONS",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:CONS:2025-10-24", ticker="CONS", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    assert payload["available"] is True
    assert payload["consensus"]["available"] is False
    assert "CONSENSUS DATA UNAVAILABLE" in payload["consensus"]["reason"]
    assert "403" in payload["consensus"]["reason"]


async def test_provenance_and_not_backtestable_labels_travel(client, seeded_only):
    """§91: statements are DATA, derived numbers are QUANT. §85: what this
    view cannot reconstruct point-in-time is named, because an empty list
    would claim the whole tab is backtestable."""
    await _seed_statement(
        "PROV",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:PROV:2025-10-24", ticker="PROV", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    assert payload["provenance"] == {"statements": "DATA", "metrics": "QUANT"}
    assert "consensus_eps" in payload["not_backtestable"]
    assert "restated_prior_periods" in payload["not_backtestable"]


# ---------------------------------------------------------------------------
# 4. Valuation — the price leg is gated too
# ---------------------------------------------------------------------------


async def test_multiples_use_the_last_close_knowable_at_as_of(client, seeded_only):
    """§14's 16:00-ET rule reaches the valuation leg.

    At 15:59 ET the same-day bar is not a settled close, so the price is the
    PREVIOUS session's 50.0 — and P/E is 50/1 = 50 on a TTM EPS of 1.0, not
    the 60 today's unsettled bar would give.
    """
    await _seed_statement(
        "PRICE",
        fiscal_year=2025,
        fiscal_period="TTM",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        timeframe="ttm",
        values=_values(eps=1.0),
    )
    await _seed_statement(
        "PRICE",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(eps=1.0),
    )
    await _seed_bars("PRICE", start=date(2025, 10, 27), closes=[50.0, 60.0])
    event_id = await _add_event(
        key="EARNINGS:PRICE:2025-10-24", ticker="PRICE", when=_et(2025, 10, 24, 16, 30)
    )
    before = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-28T15:59:00-04:00"
        )
    ).json()
    assert before["current"]["price"] == 50.0
    assert _metric(before, "pe_ttm") == pytest.approx(50.0)

    after = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-28T16:00:00-04:00"
        )
    ).json()
    assert after["current"]["price"] == 60.0
    assert _metric(after, "pe_ttm") == pytest.approx(60.0)


async def test_market_cap_is_price_times_diluted_shares(client, seeded_only):
    """The one arithmetic expression in the seam, pinned. 50.0 × 100 shares."""
    await _seed_statement(
        "CAP",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(shares=100.0),
    )
    # A run long enough that ``ensure_daily_bars`` sees fresh-enough history
    # and does not append the stub's own series on top of the seeded closes.
    await _seed_bars(
        "CAP", start=date(2025, 10, 27), closes=[50.0] * 5, volume=1_000_000.0
    )
    event_id = await _add_event(
        key="EARNINGS:CAP:2025-10-24", ticker="CAP", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-28T16:00:00-04:00"
        )
    ).json()
    assert payload["current"]["market_cap"] == pytest.approx(5000.0)


async def test_without_a_price_the_multiples_are_null_with_a_reason(client, seeded_only):
    """No bars means no valuation — and the statement metrics that need no
    price MUST survive, or a missing quote would erase a filed revenue."""
    await _seed_statement(
        "NOPRICE",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:NOPRICE:2025-10-24",
        ticker="NOPRICE",
        when=_et(2025, 10, 24, 16, 30),
    )
    # An as_of far before the stub universe leaves no bar knowable.
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    if payload["current"]["price"] is None:
        assert _metric(payload, "pe_ttm") is None
        assert payload["current"]["reasons"]["pe_ttm"]
        assert any(entry["field"] == "price" for entry in payload["unavailable"])
    # Either way the price-free metrics are intact.
    assert _metric(payload, "revenue") == 1000.0
    assert _metric(payload, "gross_margin") == pytest.approx(0.40)


async def test_valuation_block_carries_sector_and_peers_as_unavailable(client, seeded_only):
    """§30 asks for a sector/peer comparison; Phase G/J owns it. Naming the
    absence keeps the payload from implying the comparison was attempted."""
    await _seed_statement(
        "PEER",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:PEER:2025-10-24", ticker="PEER", when=_et(2025, 10, 24, 16, 30)
    )
    valuation = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()["valuation"]
    assert valuation["sector"]["available"] is False
    assert valuation["peers"]["available"] is False
    assert "Phase G/J" in valuation["sector"]["reason"]
    assert valuation["multiples"]["ev_ebitda"]["available"] is False
    assert valuation["multiples"]["fcf_yield"]["available"] is False


async def test_valuation_as_of_is_an_iso_string_not_a_datetime(client, seeded_only):
    """The library returns a real ``datetime`` there (it is a value object).
    A payload that leaked it would serialise inconsistently across clients."""
    await _seed_statement(
        "ISO",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:ISO:2025-10-24", ticker="ISO", when=_et(2025, 10, 24, 16, 30)
    )
    valuation = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()["valuation"]
    assert isinstance(valuation["as_of"], str)
    assert valuation["as_of"].endswith("+00:00")


async def test_own_history_is_priced_at_each_events_own_date(client, seeded_only):
    """§30's "against its OWN history" is only true if each historical point
    used the price that prevailed THEN. Two past prints at different closes
    must give a history of two distinct P/E values."""
    ticker = "HIST"
    for period, end, accepted in (
        ("Q1", date(2025, 3, 31), _utc(2025, 4, 20, 20, 0)),
        ("Q2", date(2025, 6, 30), _utc(2025, 7, 20, 20, 0)),
        ("Q3", date(2025, 9, 30), _utc(2025, 10, 20, 20, 0)),
    ):
        for timeframe in ("quarterly", "ttm"):
            await _seed_statement(
                ticker,
                fiscal_year=2025,
                fiscal_period=period if timeframe == "quarterly" else "TTM",
                end_date=end,
                accepted=accepted,
                timeframe=timeframe,
                values=_values(eps=1.0),
            )
    # A long bar run so every event date has a knowable close.
    await _seed_bars(
        ticker, start=date(2025, 4, 1), closes=[10.0 + i for i in range(160)]
    )
    await _add_event(
        key=f"EARNINGS:{ticker}:2025-04-21",
        ticker=ticker,
        when=_et(2025, 4, 21, 16, 30),
    )
    await _add_event(
        key=f"EARNINGS:{ticker}:2025-07-21",
        ticker=ticker,
        when=_et(2025, 7, 21, 16, 30),
    )
    current_id = await _add_event(
        key=f"EARNINGS:{ticker}:2025-10-21",
        ticker=ticker,
        when=_et(2025, 10, 21, 16, 30),
    )
    valuation = (
        await client.get(
            f"/api/events/{current_id}/fundamentals?as_of=2025-10-22T23:00:00%2B00:00"
        )
    ).json()["valuation"]
    pe = valuation["multiples"]["pe_ttm"]
    assert pe["history_n"] == 2
    assert pe["min"] is not None and pe["max"] is not None
    assert pe["min"] < pe["max"]  # different dates priced differently
    assert pe["percentile"] is not None


# ---------------------------------------------------------------------------
# 5. Freshness, momentum and the shape contract
# ---------------------------------------------------------------------------


async def test_freshness_reports_the_filing_the_snapshot_used(client, seeded_only):
    """"Fundamentals: period ending …, filed …, accepted …" (§71) needs all
    three, plus when THIS platform stored the row — a different fact from
    every date the filer asserts."""
    await _seed_statement(
        "FRESH",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        filing_date=date(2025, 10, 23),
        fetched_at=_utc(2026, 2, 1, 9, 0),
    )
    event_id = await _add_event(
        key="EARNINGS:FRESH:2025-10-24", ticker="FRESH", when=_et(2025, 10, 24, 16, 30)
    )
    freshness = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()["freshness"]
    assert freshness["period_end"] == "2025-09-30"
    assert freshness["latest_filing_date"] == "2025-10-23"
    assert freshness["acceptance_datetime"].startswith("2025-10-23T20:00")
    assert freshness["fetched_at"].startswith("2026-02-01T09:00")
    assert freshness["statements_stored"] == 1
    assert freshness["source_filing_url"] == "https://example.test/filing"


async def test_fundamental_momentum_is_quant_labelled_with_its_counts(client, seeded_only):
    """§35: a label with its own arithmetic attached can be argued with; a
    label without one is an opinion in a number's clothing."""
    _, current_id = await _seed_two_quarter_pair("MOM")
    momentum = (
        await client.get(
            f"/api/events/{current_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()["fundamental_momentum"]
    assert momentum["provenance"] == "QUANT"
    assert momentum["label"].startswith("fundamentals_")
    assert momentum["improved"] + momentum["weakened"] + momentum["unchanged"] == (
        momentum["compared"]
    )


async def test_payload_identifies_the_event_and_echoes_as_of(client, seeded_only):
    """Every payload names what it is about and when it claims to hold."""
    event_id = await _add_event(
        key="EARNINGS:ECHO:2025-10-24", ticker="ECHO", when=_et(2025, 10, 24, 16, 30)
    )
    payload = (
        await client.get(
            f"/api/events/{event_id}/fundamentals?as_of=2025-10-25T23:00:00%2B00:00"
        )
    ).json()
    assert payload["event_id"] == event_id
    assert payload["event_key"] == "EARNINGS:ECHO:2025-10-24"
    assert payload["ticker"] == "ECHO"
    assert payload["as_of"] == "2025-10-25T23:00:00+00:00"
    assert payload["model_version"]


# ---------------------------------------------------------------------------
# 6. Routing contract
# ---------------------------------------------------------------------------


async def test_unknown_event_is_404(client, seeded_only):
    r = await client.get("/api/events/999999/fundamentals")
    assert r.status_code == 404


async def test_malformed_as_of_is_422(client, seeded_only):
    event_id = await _add_event(
        key="EARNINGS:BAD:2025-10-24", ticker="BAD", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(f"/api/events/{event_id}/fundamentals?as_of=not-a-date")
    assert r.status_code == 422


async def test_future_as_of_is_422(client, seeded_only):
    """No filing exists for tomorrow; silently clamping to now would answer a
    different question than the one asked."""
    event_id = await _add_event(
        key="EARNINGS:FUTQ:2025-10-24", ticker="FUTQ", when=_et(2025, 10, 24, 16, 30)
    )
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    r = await client.get(
        f"/api/events/{event_id}/fundamentals?as_of={future.replace('+', '%2B')}"
    )
    assert r.status_code == 422
    assert "future" in r.json()["detail"]


async def test_as_of_defaults_to_now(client, seeded_only):
    """Omitting ``as_of`` is legal at THIS boundary only; the seam itself
    requires the instant (audit §7.2 rule 2)."""
    event_id = await _add_event(
        key="EARNINGS:DEF:2025-10-24", ticker="DEF", when=_et(2025, 10, 24, 16, 30)
    )
    r = await client.get(f"/api/events/{event_id}/fundamentals")
    assert r.status_code == 200
    assert r.json()["as_of"]


# ---------------------------------------------------------------------------
# 7. Ingestion — idempotence, throttling, restatement
# ---------------------------------------------------------------------------


async def test_ensure_fundamentals_stores_rows_and_audits_once(client):
    """First call for a symbol mirrors the vendor's filings with ONE SYSTEM
    DATA_BACKFILL event in the same transaction (rule 12, ADR-003)."""
    now = _utc(2026, 3, 2, 12, 0)
    async with SessionLocal() as s:
        rows = await fundamentals_seam.ensure_fundamentals(s, "GW", "stub", now=now)
    assert rows
    assert all(row.ticker == "GW" for row in rows)
    async with SessionLocal() as s:
        events = (
            (
                await s.execute(
                    select(AuditEvent).where(
                        AuditEvent.entity_type == "fundamental_statements",
                        AuditEvent.entity_id == "GW",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(events) == 1
    assert events[0].details["kind"] == "fundamentals"
    assert events[0].details["ticker"] == "GW"
    assert events[0].details["rows"] == len(rows)


async def test_refetching_the_same_periods_writes_no_duplicates(client):
    """The natural key makes ingestion idempotent at the DATABASE level, so a
    second replica re-ingesting the same tick can only collide."""
    first_now = _utc(2026, 3, 2, 12, 0)
    async with SessionLocal() as s:
        first = await fundamentals_seam.ensure_fundamentals(
            s, "GOOGL", "stub", now=first_now
        )
    fundamentals_seam._refresh_attempts.clear()
    # Far enough later that the attempt throttle has expired.
    async with SessionLocal() as s:
        second = await fundamentals_seam.ensure_fundamentals(
            s, "GOOGL", "stub", now=first_now + timedelta(days=1)
        )
    assert len(second) == len(first)
    async with SessionLocal() as s:
        count = len(
            (
                await s.execute(
                    select(FundamentalStatementRow).where(
                        FundamentalStatementRow.ticker == "GOOGL"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert count == len(first)


async def test_an_unchanged_refetch_writes_no_second_audit_row(client):
    """An audit trail of no-ops is noise; ``fetched_at`` that ticks when
    nothing changed is a freshness stamp nobody can read."""
    first_now = _utc(2026, 3, 2, 12, 0)
    async with SessionLocal() as s:
        await fundamentals_seam.ensure_fundamentals(s, "IBM", "stub", now=first_now)
    fundamentals_seam._refresh_attempts.clear()
    async with SessionLocal() as s:
        await fundamentals_seam.ensure_fundamentals(
            s, "IBM", "stub", now=first_now + timedelta(days=1)
        )
    async with SessionLocal() as s:
        events = (
            (
                await s.execute(
                    select(AuditEvent).where(
                        AuditEvent.entity_type == "fundamental_statements",
                        AuditEvent.entity_id == "IBM",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(events) == 1


async def test_a_restatement_overwrites_and_moves_the_acceptance_instant(client):
    """Audit §7.3: the vendor serves only its CURRENT XBRL view, so claiming
    to store both the original and the restatement would be a fiction. The
    moved acceptance instant IS the flag that a period was restated."""
    await _seed_statement(
        "RESTATE",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
        values=_values(revenue=1000.0),
    )

    class _Restated:
        ticker = "RESTATE"
        cik = "0000000001"
        timeframe = "quarterly"
        fiscal_year = 2025
        fiscal_period = "Q3"
        start_date = date(2025, 7, 1)
        end_date = date(2025, 9, 30)
        filing_date = date(2026, 1, 15)
        acceptance_datetime = _utc(2026, 1, 15, 20, 0)
        source_filing_url = "https://example.test/restated"
        values = _values(revenue=950.0)
        raw_fields_count = 12

    async with SessionLocal() as s:
        rows = (
            (
                await s.execute(
                    select(FundamentalStatementRow).where(
                        FundamentalStatementRow.ticker == "RESTATE"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        changed = fundamentals_seam._apply_update(
            rows[0], _Restated(), now=_utc(2026, 1, 16, 9, 0)
        )
        assert changed is True
        await s.commit()

    async with SessionLocal() as s:
        rows = (
            (
                await s.execute(
                    select(FundamentalStatementRow).where(
                        FundamentalStatementRow.ticker == "RESTATE"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1  # overwritten, not duplicated
    assert rows[0].values["income_statement.revenues"] == 950.0
    assert rows[0].acceptance_datetime.replace(tzinfo=timezone.utc) == _utc(
        2026, 1, 15, 20, 0
    )


async def test_a_second_call_inside_the_throttle_does_not_reach_the_provider(client):
    """A quarterly filer is "stale" for ~89 days out of 90. Without the
    attempt throttle every request would re-ask the vendor for a filing that
    does not exist yet."""
    calls: list[str] = []

    class _CountingProvider:
        def get_financials(self, ticker, *, timeframe="quarterly", limit=12):
            calls.append(timeframe)
            return []

    original = fundamentals_seam.get_provider
    fundamentals_seam.get_provider = lambda name: _CountingProvider()
    try:
        now = _utc(2026, 3, 2, 12, 0)
        async with SessionLocal() as s:
            await fundamentals_seam.ensure_fundamentals(s, "THROT", "stub", now=now)
        first = len(calls)
        assert first == 2  # quarterly + ttm
        async with SessionLocal() as s:
            await fundamentals_seam.ensure_fundamentals(
                s, "THROT", "stub", now=now + timedelta(minutes=5)
            )
        assert len(calls) == first  # throttled — no new provider call
    finally:
        fundamentals_seam.get_provider = original


async def test_a_403_capability_refusal_serves_stored_rows_and_raises_nothing(
    client,
):
    """§16: the answer to a missing capability is an explicit refusal, never
    invented numbers — and never a 5xx out of a read endpoint."""
    from libs.market_data import CapabilityNotAvailable

    await _seed_statement(
        "DENIED",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )

    class _DenyingProvider:
        def get_financials(self, ticker, *, timeframe="quarterly", limit=12):
            raise CapabilityNotAvailable("financials not in subscription (403)")

    original = fundamentals_seam.get_provider
    fundamentals_seam.get_provider = lambda name: _DenyingProvider()
    try:
        async with SessionLocal() as s:
            rows = await fundamentals_seam.ensure_fundamentals(
                s, "DENIED", "stub", now=_utc(2026, 3, 2, 12, 0)
            )
        assert len(rows) == 1  # the stored row still serves
    finally:
        fundamentals_seam.get_provider = original


async def test_a_provider_exception_serves_stored_rows(client):
    """An unexpected vendor failure must not take the endpoint down with it."""

    class _BrokenProvider:
        def get_financials(self, ticker, *, timeframe="quarterly", limit=12):
            raise RuntimeError("connection reset")

    original = fundamentals_seam.get_provider
    fundamentals_seam.get_provider = lambda name: _BrokenProvider()
    try:
        async with SessionLocal() as s:
            rows = await fundamentals_seam.ensure_fundamentals(
                s, "BROKEN", "stub", now=_utc(2026, 3, 2, 12, 0)
            )
        assert rows == []
    finally:
        fundamentals_seam.get_provider = original


async def test_ensure_fundamentals_ignores_a_blank_ticker(client):
    """A macro event's empty ticker must never reach the vendor as a symbol."""
    async with SessionLocal() as s:
        assert (
            await fundamentals_seam.ensure_fundamentals(
                s, "", "stub", now=_utc(2026, 3, 2, 12, 0)
            )
            == []
        )


async def test_stored_acceptance_instants_reach_the_pure_layer_aware(client):
    """THE silent-failure guard, and the reason ``_StoredStatement`` exists.

    SQLite hands ``TIMESTAMPTZ`` columns back NAIVE, and the pure layer
    REFUSES a naive acceptance instant rather than assuming UTC (it cannot
    prove an unknown zone precedes ``as_of``). Both rules are right; together,
    without the seam's re-stamp, they drop every stored row with no error and
    no reason — the payload would say "no statement was public at as_of"
    forever, for every ticker, and every as-of test above would still pass
    because they would all be asserting on an empty answer.

    So this asserts the boundary directly: what the seam hands the library is
    aware-UTC, and the library's own gate then ADMITS it.
    """
    from libs.trading_core.events.fundamentals import select_statements_as_of

    await _seed_statement(
        "AWARE",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=_utc(2025, 10, 23, 20, 0),
    )
    async with SessionLocal() as s:
        rows = await fundamentals_seam._stored_rows(s, "AWARE")
    statements = fundamentals_seam.to_statements(rows)
    assert len(statements) == 1
    accepted = statements[0].acceptance_datetime
    assert accepted is not None
    assert accepted.tzinfo is not None
    assert accepted.utcoffset() == timedelta(0)

    quarterly, _, visible = select_statements_as_of(
        statements, _utc(2025, 10, 24, 0, 0)
    )
    assert quarterly is not None
    assert len(visible) == 1


async def test_a_provider_row_with_no_acceptance_stays_none_after_conversion(
    client,
):
    """The re-stamp must not INVENT an instant either. After conversion a
    ``None`` really does mean the provider omitted it — which is what makes
    the pure layer's refusal a statement about the feed rather than about
    SQLite."""
    await _seed_statement(
        "NOACC2",
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        accepted=None,
    )
    async with SessionLocal() as s:
        rows = await fundamentals_seam._stored_rows(s, "NOACC2")
    assert fundamentals_seam.to_statements(rows)[0].acceptance_datetime is None


async def test_stored_rows_are_never_filtered_by_date_in_sql(client):
    """The as-of gate lives in the pure layer, ONCE. A WHERE clause here would
    be a second, untested copy of the §85 rule — and the two would drift.

    Proven by loading a ticker whose only filing is far in the future and
    asserting the SEAM still returns it: the row is loaded, and the LIBRARY
    is what refuses it.
    """
    await _seed_statement(
        "SQLGATE",
        fiscal_year=2030,
        fiscal_period="Q1",
        end_date=date(2030, 3, 31),
        accepted=_utc(2030, 4, 20, 20, 0),
    )
    async with SessionLocal() as s:
        rows = await fundamentals_seam._stored_rows(s, "SQLGATE")
    assert len(rows) == 1


def test_fundamentals_provider_prefers_massive_when_its_key_is_configured():
    """Statements live on Massive only (audit §3); the market-data provider is
    Alpaca. The seam must not ask Alpaca for financials when a Massive key
    exists (live 2026-08-19: statements_stored=0 with provider=alpaca)."""
    from types import SimpleNamespace

    from apps.gateway.fundamentals import fundamentals_provider_name

    assert fundamentals_provider_name(
        SimpleNamespace(massive_api_key="k", market_data_provider="alpaca")
    ) == "massive"
    assert fundamentals_provider_name(
        SimpleNamespace(massive_api_key="   ", market_data_provider="alpaca")
    ) == "alpaca"
    assert fundamentals_provider_name(
        SimpleNamespace(massive_api_key="", market_data_provider="")
    ) == ""
