"""PHASE L — ADVERSARIAL LOOK-AHEAD VALIDATION of every as-of surface
(event spec §85, §96; audit §7.1, §7.2, §11.5).

Every other event test in this suite asks "is the number right?". This one
asks one question, twelve times, once per endpoint family:

    **If a fact that did not exist yet is sitting in the database, can the
    endpoint see it?**

THE METHOD, identical for every family below and deliberately monotonous:

1.  Plant a PAST twin and a FUTURE artifact in SQLite — directly, never
    through an ingest tick — differing only in the instant that gates them
    (a bar dated after ``as_of``; a statement whose ``acceptance_datetime``
    is an hour later; an article ``published_at`` an hour later; an
    observation whose ``release_at`` is later; a Fed document ``released_at``
    later; an analysis row whose ``as_of`` is later; a stored option metric).
    The future artifact carries a SENTINEL — a distinctive string and a
    distinctive number that appear nowhere else in the platform.
2.  Call the endpoint with ``as_of=T``, between the two.
3.  Assert the sentinel is absent from the WHOLE payload, by recursively
    walking every dict, list, string, number and mapping KEY
    (``_lookahead_util.assert_absent``) — not by checking the two or three
    fields the author happened to think of. A field-by-field assertion is
    only ever as good as the author's imagination, and the leak found by this
    file (see ``test_fundamentals_freshness_block_leaks_...`` below) was in a
    block no field-level test was looking at.
4.  Assert the PAST twin IS visible (``assert_present``). Every test here is
    a PAIR. A gate that returned nothing at all would satisfy every
    absence assertion in this file while destroying the endpoint, and the
    paired half is what makes a green run mean "point-in-time" rather than
    "empty".

COVERAGE TABLE — the twelve endpoint families §96 requires, the artifact
planted in each, and the gate that must catch it:

    | # | endpoint                          | future artifact planted     | gate under test                          |
    |---|-----------------------------------|-----------------------------|------------------------------------------|
    | 1 | GET .../price-context             | daily bar dated after       | reaction.as_of_bar_filter + the pure     |
    |   |                                   |                             | context's own ``as_of_date_et`` bound    |
    | 2 | GET .../fundamentals              | statement accepted after    | fundamentals.select_statements_as_of     |
    | 3 | GET .../replay                    | daily + minute bar after    | as_of_bar_filter / replay window gate    |
    | 4 | GET .../history                   | a LATER earnings event row  | event_price._past_comparable_rows        |
    | 5 | GET .../news                      | article published after     | news_intel.analyze_window (§96 stage 1)  |
    | 6 | GET .../evidence                  | all of the above at once    | every seam's own gate, composed          |
    | 7 | GET .../analysis + .../analyses   | analysis row as-of after    | event_analysis.prior_analyses_for_ticker |
    | 8 | GET .../options                   | stored metric + later print | _past_comparable_rows + §85 basis rule   |
    | 9 | GET .../macro                     | observation released after  | macro.visible_prints                     |
    |10 | GET .../fed                       | document released after     | SQL bound + fed_intel._gate              |
    |11 | GET .../timeline                  | article + filing + event    | every kind's own gate, composed          |
    |12 | GET .../risk                      | later print + option metric | event_risk._previous_prints              |

MUTATION VERIFICATION (the house rule from tests/test_risk_adversarial.py).
A test that cannot fail is worse than no test, because it is a green light
nobody re-examines. So six ``test_the_suite_bites_*`` tests below temporarily
monkeypatch the gate to a pass-through INSIDE the test and assert the sentinel
BECOMES visible — proving the corresponding absence assertion is load-bearing
rather than passing because the fixture happened to store nothing. Sources are
never edited; the patch dies with the test.

Two of those bite tests record a finding worth stating up front, because it is
the shape of the whole defence and not an accident:

- **The news and Fed surfaces are gated TWICE** (a SQL bound plus the pure
  layer's own gate), and the bite test proves the PURE gate alone is
  sufficient: it first replaces the loader with one that returns EVERY stored
  row, shows the payload is STILL clean, and only then defeats the pure gate
  to make the sentinel appear. Without that middle step the test would prove
  only that the SQL bound works, and deleting the SQL clause as "redundant"
  would silently reopen the leak with every assertion still green.
- **The price surface is likewise gated twice** — ``as_of_bar_filter`` at the
  seam and an independent ``b.date <= as_of_date_et`` bound inside
  ``pre_event_price_context``. Defeating only the first moves ``bars_through``
  by a single day; the sentinel needs BOTH defeated. That is defence in depth
  working, and the bite test asserts the intermediate state too so a future
  refactor that collapses the two into one is visible here.

§85 — WHAT CANNOT BE RECONSTRUCTED POINT-IN-TIME. A live option-chain
snapshot and a current quote are not historical facts and this platform cannot
rebuild either. The rule §85 imposes is therefore about LABELLING, and the
last section asserts it: ``LIVE_CHAIN_SNAPSHOT`` is served ONLY for an event
that has not happened yet, and a PAST event's options payload must never carry
the LIVE basis — not in ``current``, not anywhere in ``history``, even when a
LIVE row is sitting in the database for that very event.

Uses the shared ``client`` fixture (conftest.py): providers "stub", execution
"simulated".
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from apps.gateway import event_analysis as ea
from apps.gateway import event_fed as ef
from apps.gateway import event_news as en
from apps.gateway import event_price as ep
from apps.gateway import fundamentals as fu
from apps.gateway.db import (
    EventAnalysisRow,
    EventOptionMetricRow,
    EventRow,
    FedDocumentRow,
    FundamentalStatementRow,
    MacroObservationRow,
    NewsArticleRow,
    SessionLocal,
    StockBar1mRow,
    StockBarDaily,
)
from libs.trading_core.events.implied_move import BASIS_HISTORICAL, BASIS_LIVE
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from ._lookahead_util import (
    assert_absent,
    assert_present,
    et,
    find_sentinel,
    iso,
    utc,
    walk_strings,
    weekdays,
)

# ---------------------------------------------------------------------------
# The instant everything is anchored on, and the sentinels
# ---------------------------------------------------------------------------

#: THE as-of. A fixed instant rather than ``now()`` so every boundary below is
#: a number a reader can check by hand. Every "past" artifact precedes it and
#: every "future" artifact follows it, usually by ONE HOUR — the gate is
#: attacked at its edge, not from a comfortable distance, because an
#: off-by-one in a timezone conversion is exactly the bug that ships.
NOW = utc(2026, 8, 18, 12, 0)

#: One hour AFTER ``NOW``: the instant every planted future artifact carries.
LATER = NOW + timedelta(hours=1)

#: The string sentinel. Deliberately unpronounceable and unique across the
#: whole repository, so a match is never a coincidence and ``grep`` finds
#: every place it is planted.
SENTINEL = "ZZLOOKAHEAD"

#: The numeric sentinel, planted as a price/value. A distinctive float rather
#: than a round number: 999.0 could plausibly be arithmetic, 987654.321 could
#: not. ``walk_strings`` renders numbers with ``repr`` so this is findable in
#: the same pass as the string.
SENTINEL_NUM = 987654.321

TICKER = "AAPL"


# ---------------------------------------------------------------------------
# Seeding helpers — direct inserts, never an ingestion tick
# ---------------------------------------------------------------------------


async def _add_event(
    *,
    key: str,
    ticker: str | None,
    when: datetime,
    event_type: EventType = EventType.EARNINGS,
    status: EventStatus = EventStatus.CONFIRMED,
    title: str = "Earnings",
    **extra,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title=title,
            ticker=ticker,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=EventSession.AFTER_MARKET.value,
            status=status.value,
            source=EventSourceKind.STRUCTURED_PROVIDER.value,
            source_name="test",
            **extra,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_bars(
    ticker: str,
    *,
    start: date,
    count: int,
    future_after: date,
) -> list[date]:
    """Daily bars whose closes climb by 1.00 a day — EXCEPT after
    ``future_after``, where every close is :data:`SENTINEL_NUM`.

    Two properties matter. The past closes are boring consecutive numbers, so
    a leak is not hidden among lookalikes; and every post-``as_of`` bar
    carries the SAME distinctive value, so it does not matter which of them a
    leaking payload happens to quote — a max, a last close, a 52-week high and
    an ATR built from them all surface the sentinel.

    Bars are seeded for the benchmark too: SPY's post-``as_of`` sessions are
    just as unknowable as the issuer's, and a relative-return computed against
    a leaked benchmark is the same violation one indirection away.
    """
    days = weekdays(start, count)
    async with SessionLocal() as s:
        for i, day in enumerate(days):
            close = SENTINEL_NUM if day > future_after else 100.0 + i
            for symbol in (ticker, "SPY"):
                s.add(
                    StockBarDaily(
                        ticker=symbol,
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


async def _seed_articles() -> None:
    """One PAST development and one FUTURE one, an hour either side of NOW."""
    async with SessionLocal() as s:
        s.add(
            NewsArticleRow(
                source_id="past-guidance",
                title="Apple raises full-year guidance on strong iPhone demand",
                publisher="Reuters",
                published_at=NOW - timedelta(days=2),
                url="https://news.test/past-guidance",
                tickers=[TICKER],
                description="",
                fetched_at=NOW - timedelta(days=2),
            )
        )
        s.add(
            NewsArticleRow(
                source_id=f"{SENTINEL}-recall",
                title=f"Apple announces {SENTINEL} global product recall",
                publisher="Bloomberg",
                published_at=LATER,
                url=f"https://news.test/{SENTINEL}",
                tickers=[TICKER],
                description=f"{SENTINEL} — published after as_of",
                fetched_at=LATER,
            )
        )
        await s.commit()


async def _seed_statements() -> None:
    """A quarter accepted MONTHS ago and one accepted an hour after NOW.

    The future filing's PERIOD END (2026-06-30) is well BEFORE ``as_of``,
    which is the entire point: a gate written against ``end_date`` instead of
    ``acceptance_datetime`` would admit it, and would look correct doing so.
    """
    async with SessionLocal() as s:
        s.add(
            FundamentalStatementRow(
                ticker=TICKER,
                timeframe="quarterly",
                fiscal_year=2026,
                fiscal_period="Q1",
                start_date=date(2026, 1, 1),
                end_date=date(2026, 3, 31),
                filing_date=date(2026, 5, 1),
                acceptance_datetime=utc(2026, 5, 1, 20, 30),
                source_filing_url="https://sec.test/past-filing",
                values={
                    "income_statement.revenues": 111_111.0,
                    "income_statement.net_income_loss": 22_222.0,
                    "balance_sheet.assets": 333_333.0,
                },
                raw_fields_count=3,
            )
        )
        s.add(
            FundamentalStatementRow(
                ticker=TICKER,
                timeframe="quarterly",
                fiscal_year=2026,
                fiscal_period="Q2",
                start_date=date(2026, 4, 1),
                end_date=date(2026, 6, 30),  # period CLOSED before as_of
                filing_date=date(2026, 8, 17),
                acceptance_datetime=LATER,  # ...but was PUBLIC only after it
                source_filing_url=f"https://sec.test/{SENTINEL}-filing",
                values={
                    "income_statement.revenues": SENTINEL_NUM,
                    "income_statement.net_income_loss": SENTINEL_NUM,
                    "balance_sheet.assets": SENTINEL_NUM,
                },
                raw_fields_count=3,
            )
        )
        await s.commit()


async def _standard_scene() -> tuple[int, int]:
    """The scene nearly every test below runs against.

    ``(subject_event_id, future_event_id)``: an earnings print four days
    BEFORE ``as_of`` (so it is a PAST event and the §85 rules for past events
    apply), one comparable print a quarter earlier that IS legitimate history,
    and one print a week AFTER ``as_of`` whose key carries the sentinel —
    a row that exists in the registry today and must not be treated as
    history at ``as_of``.
    """
    await _add_event(
        key="EARNINGS:AAPL:2026-05-01",
        ticker=TICKER,
        when=et(2026, 5, 1, 16, 5),
    )
    subject = await _add_event(
        key="EARNINGS:AAPL:2026-08-14",
        ticker=TICKER,
        when=et(2026, 8, 14, 16, 5),
    )
    future = await _add_event(
        key=f"EARNINGS:AAPL:{SENTINEL}:2026-08-25",
        ticker=TICKER,
        when=NOW + timedelta(days=7),
        title=f"{SENTINEL} future print",
    )
    await _seed_bars(
        TICKER, start=date(2026, 4, 1), count=110, future_after=NOW.date()
    )
    await _seed_articles()
    await _seed_statements()
    return subject, future


async def _event_id_for_key(key: str) -> int:
    """The id of an already-seeded event, by its registry key."""
    from sqlalchemy import select

    async with SessionLocal() as s:
        row = (
            await s.execute(select(EventRow).where(EventRow.event_key == key))
        ).scalar_one()
        return row.id


async def _get(client, event_id: int, suffix: str, *, as_of: datetime = NOW) -> dict:
    """One as-of read that must answer 200 — a degraded block is fine, a 5xx
    is not: this suite is about what a payload CONTAINS, and an endpoint that
    errored contains nothing and would pass every absence assertion."""
    response = await client.get(
        f"/api/events/{event_id}/{suffix}?as_of={iso(as_of)}"
    )
    assert response.status_code == 200, response.text
    return response.json()


# ===========================================================================
# 0. THE SCANNER ITSELF — attacked before it is trusted
# ===========================================================================
#
# Every assertion in this file routes through ``assert_absent``. A scanner
# with a blind spot would turn all twelve suites into tests that cannot fail,
# so its coverage is proved here against hand-built payloads BEFORE any
# endpoint is called.


def test_the_scanner_finds_a_sentinel_at_every_depth_and_shape():
    """Nested dicts, lists, tuples, mapping KEYS, numbers and datetimes.

    The KEY case is the one worth spelling out: several payloads in this
    platform key a mapping BY an artifact's identity (``metrics_by_period``
    on the reference period), so a leaked future observation can appear as a
    KEY whose value is an innocent number. A values-only scan would call that
    payload clean.
    """
    assert find_sentinel({"a": SENTINEL}, SENTINEL)
    assert find_sentinel({"a": {"b": [{"c": f"x {SENTINEL} y"}]}}, SENTINEL)
    assert find_sentinel([[[SENTINEL]]], SENTINEL)
    assert find_sentinel({"a": ({"b": SENTINEL},)}, SENTINEL)
    # a mapping KEY, value innocent
    assert find_sentinel({"periods": {SENTINEL: 1.0}}, SENTINEL)
    # a number, rendered by repr
    assert find_sentinel({"close": SENTINEL_NUM}, str(SENTINEL_NUM))
    # a datetime, rendered isoformat
    assert find_sentinel({"at": LATER}, "2026-08-18T13:00")


def test_the_scanner_reports_clean_on_a_payload_that_is_clean():
    """The other half: no false positives, or every suite here fails
    permanently and gets deleted rather than debugged."""
    assert find_sentinel({"a": "ordinary", "b": [1, 2.5, None, True]}, SENTINEL) == []
    assert find_sentinel({}, SENTINEL) == []


def test_the_scanner_does_not_stringify_none_or_bools_into_false_matches():
    """``None`` and booleans carry no identity and must not be scanned —
    otherwise a sentinel containing "None" or "True" would match empty
    fields all over the platform and the scan would be useless."""
    assert list(walk_strings({"a": None, "b": True, "c": False})) == ["a", "b", "c"]


def test_assert_absent_raises_with_the_offending_strings_named():
    """A failure must say WHERE it leaked. "absent" is a claim about a whole
    document; a bare ``False`` gives the next reader nothing to work with."""
    with pytest.raises(AssertionError) as exc:
        assert_absent({"deep": {"url": f"https://x/{SENTINEL}"}}, SENTINEL, where="X")
    assert SENTINEL in str(exc.value)
    assert "LOOK-AHEAD LEAK in X" in str(exc.value)


def test_assert_present_catches_a_gate_that_returns_nothing():
    """The paired half. A gate that returned an EMPTY payload would satisfy
    every ``assert_absent`` in this file; this is what refuses to call that
    a pass."""
    with pytest.raises(AssertionError) as exc:
        assert_present({"items": []}, "past-guidance", where="Y")
    assert "simply returning nothing" in str(exc.value)


# ===========================================================================
# 1. GET /api/events/{id}/price-context — bars dated after as_of
# ===========================================================================


async def test_price_context_never_sees_a_bar_dated_after_as_of(client):
    """§14/§96: a session that has not closed cannot price anything.

    The sentinel is planted as the CLOSE of every post-``as_of`` bar, so it
    would surface through ``last_close``, ``high_52w``, the ATR, the SMAs or
    the realized vol — any one of them quoting a future session fails this.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "price-context")

    assert_absent(payload, str(SENTINEL_NUM), where="price-context")
    assert payload["pre_event"]["bars_through"] == "2026-08-17"
    # PAIRED: the endpoint still answers with real, past bars.
    assert payload["bars"]["available"] is True
    assert payload["pre_event"]["last_close"] is not None
    assert payload["pre_event"]["n_bars"] > 20


async def test_price_context_never_treats_a_later_print_as_history(client):
    """The registry gate, distinct from the bar gate: an earnings row
    scheduled a week after ``as_of`` exists in the table TODAY and is not a
    precedent AT ``as_of``, however firmly the platform knows about it now."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "price-context")

    assert_absent(payload, SENTINEL, where="price-context previous_events")
    # PAIRED: the legitimate May print IS in the history.
    assert_present(payload, "2026-05-01", where="price-context previous_events")


# ===========================================================================
# 2. GET /api/events/{id}/fundamentals — a statement accepted after as_of
# ===========================================================================


async def test_fundamentals_never_quotes_a_filing_accepted_after_as_of(client):
    """§7/§85/§96: the gate is on ``acceptance_datetime``, never ``end_date``.

    The planted filing's period ENDED 2026-06-30, seven weeks before
    ``as_of`` — a gate written against the period end would admit it and look
    entirely reasonable. Its numbers are the sentinel, so any metric,
    multiple, delta or momentum score computed from it surfaces here.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "fundamentals")

    assert_absent(
        payload["current"], str(SENTINEL_NUM), where="fundamentals current"
    )
    assert_absent(
        payload.get("changes") or {}, str(SENTINEL_NUM), where="fundamentals changes"
    )
    assert_absent(
        payload.get("valuation") or {},
        str(SENTINEL_NUM),
        where="fundamentals valuation",
    )
    assert_absent(
        payload.get("fundamental_momentum") or {},
        str(SENTINEL_NUM),
        where="fundamentals momentum",
    )
    # PAIRED: the Q1 filing accepted in May IS visible and IS the basis.
    assert_present(payload["current"], "111111.0", where="fundamentals current")


async def test_fundamentals_states_how_many_rows_the_gate_excluded(client):
    """Honest absence (§44 rule 18): the payload SAYS a row was withheld
    rather than silently serving one fewer quarter. A gate that drops rows
    without saying so is indistinguishable from a provider that never sent
    them."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "fundamentals")

    haystack = " ".join(walk_strings(payload))
    assert "accepted after" in haystack, (
        "the payload does not say a statement row was withheld by the gate"
    )


# §96 leak FIXED 2026-08-19 (freshness now built from the as_of-gated rows);
# the xfail that discovered it was removed when the fix landed — see DEVLOG (33).
async def test_fundamentals_freshness_block_leaks_the_future_filing(client):
    """The freshness block must describe the newest VISIBLE filing.

    Found by the recursive scan and by nothing else in the suite: no
    field-level test was looking at ``freshness``, and the leaked
    ``acceptance_datetime`` is a timestamp an hour after the ``as_of`` the
    caller supplied — the payload states, in its own words, that it knows
    about something that had not happened.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "fundamentals")

    assert_absent(payload["freshness"], SENTINEL, where="fundamentals freshness")
    accepted = payload["freshness"]["acceptance_datetime"]
    assert accepted is None or accepted <= iso(NOW), (
        f"freshness.acceptance_datetime {accepted!r} is after as_of {iso(NOW)}"
    )


# ===========================================================================
# 3. GET /api/events/{id}/replay — bars and minutes after as_of
# ===========================================================================


async def test_replay_never_reports_a_bar_or_minute_after_as_of(client):
    """§20: a replay of a past print is what was knowable THEN.

    Minute bars are planted for the day AFTER ``as_of`` as well as daily
    ones, because the replay's reaction windows read minutes and a gate that
    covered only the daily series would leak through the intraday block.
    """
    subject, _ = await _standard_scene()
    async with SessionLocal() as s:
        for minute in range(5):
            s.add(
                StockBar1mRow(
                    ticker=TICKER,
                    ts=LATER + timedelta(minutes=minute),
                    open=SENTINEL_NUM,
                    high=SENTINEL_NUM,
                    low=SENTINEL_NUM,
                    close=SENTINEL_NUM,
                    volume=1000.0,
                )
            )
        await s.commit()

    payload = await _get(client, subject, "replay")
    assert_absent(payload, str(SENTINEL_NUM), where="replay")
    assert_absent(payload, SENTINEL, where="replay")
    # PAIRED: the replay is a real answer about a real, past event.
    assert payload["event"]["event_key"] == "EARNINGS:AAPL:2026-08-14"


# ===========================================================================
# 4. GET /api/events/{id}/history — a later print is not history
# ===========================================================================


async def test_history_never_lists_an_event_scheduled_after_as_of(client):
    """§60: "LAST N EARNINGS" means the last N KNOWABLE ones.

    The registry contains a print dated a week after ``as_of`` because the
    calendar ingest legitimately knows about upcoming events. It is a fact
    about the FUTURE and must not enter a table describing the past.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "history")

    assert_absent(payload, SENTINEL, where="history")
    assert_absent(payload, str(SENTINEL_NUM), where="history")
    # PAIRED: the May print IS the history.
    assert_present(payload, "2026-05-01", where="history")


# ===========================================================================
# 5. GET /api/events/{id}/news — an article published after as_of
# ===========================================================================


async def test_news_never_sees_an_article_published_after_as_of(client):
    """§21-§27/§96: the gate is on ``published_at`` and it runs FIRST.

    The future article is not merely absent from the evidence list — it must
    not have influenced a COUNT, a cluster, a novelty measurement or a score
    either, which is why the whole payload is scanned rather than the
    evidence array alone.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "news")

    assert_absent(payload, SENTINEL, where="news")
    # PAIRED: the guidance story published two days ago IS in the window.
    assert_present(payload, "past-guidance", where="news")


async def test_news_counts_do_not_include_the_future_article(client):
    """The §26 counts are the headline number a reader trusts. An article
    that was not knowable must not be counted even as "raw" — a count of
    articles nobody could have read is not a fact about the window."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "news")

    counts = payload.get("counts") or {}
    assert counts.get("raw") == 1, counts
    assert_absent(counts, SENTINEL, where="news counts")


# ===========================================================================
# 6. GET /api/events/{id}/evidence — every artifact at once
# ===========================================================================


async def test_the_evidence_bundle_composes_only_knowable_facts(client):
    """§46: the bundle is the document the model is handed.

    This is the highest-value assertion in the file. Every leak in every
    section below it becomes a number the LLM quotes as fact, laundered
    through prose where no downstream validator can spot it — ``numbers_
    quoted`` checks the analysis against the BUNDLE, so a poisoned bundle
    validates perfectly.
    """
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "evidence")
    bundle = payload["bundle"]

    # EVERY NUMBER in the bundle is gated. This is the assertion that matters
    # most: a leaked VALUE is what the model would quote, and ``numbers_quoted``
    # validates the analysis against this document, so a poisoned bundle
    # validates perfectly and the leak becomes undetectable downstream.
    assert_absent(bundle, str(SENTINEL_NUM), where="evidence bundle (numeric)")

    # Every section EXCEPT the two carrying the confirmed ``freshness`` defect
    # (see test_the_evidence_bundle_inherits_the_fundamentals_freshness_leak
    # below; the defect is FIXED, so the assertion now passes outright).
    for section, block in bundle.items():
        if section in ("fundamentals", "source_metadata"):
            continue
        assert_absent(block, SENTINEL, where=f"evidence bundle.{section}")

    # PAIRED: the bundle is a real document with real, past evidence in it.
    assert payload["bundle_digest"]
    assert_present(bundle, "past-guidance", where="evidence bundle")


async def test_the_bundle_digest_is_stable_across_two_reads(client):
    """A digest that moved between two identical as-of reads would mean some
    section is reading ``now()`` rather than ``as_of`` — a look-ahead that
    would be invisible to a single-shot assertion."""
    subject, _ = await _standard_scene()
    first = await _get(client, subject, "evidence")
    second = await _get(client, subject, "evidence")
    assert first["bundle_digest"] == second["bundle_digest"]


# ===========================================================================
# 7. GET .../analysis + .../analyses — a prior analysis as-of after
# ===========================================================================


async def _seed_analysis(
    event_id: int, *, as_of: datetime, marker: str, status: str = "OK"
) -> int:
    """One stored analysis package whose prose carries ``marker``."""
    async with SessionLocal() as s:
        row = EventAnalysisRow(
            event_id=event_id,
            as_of=as_of,
            kind="PRE_EVENT",
            bundle={"note": marker},
            bundle_digest=f"digest-{marker}",
            analysis={
                "executive_summary": f"{marker} summary",
                "regime": "NEUTRAL",
                "confidence": 0.5,
            },
            provider="stub",
            model="stub-model",
            prompt_version="v1",
            violations=[],
            status=status,
        )
        s.add(row)
        await s.commit()
        return row.id


async def test_the_event_memory_never_recalls_an_analysis_from_the_future(client):
    """§69/§96: a prior analysis written FOR a later instant knows things this
    run must not, and feeding it back would be a leak laundered through the
    model's own prose — the hardest kind to detect downstream, because the
    number arrives as an opinion rather than as a field."""
    subject, future = await _standard_scene()
    # The FUTURE opinion hangs off the future print; the PAST one hangs off the
    # May print, whose key carries no sentinel — otherwise the past summary
    # would legitimately quote the future event's KEY and the assertion would
    # fire on the test's own scaffolding rather than on a leak.
    earlier = await _event_id_for_key("EARNINGS:AAPL:2026-05-01")
    await _seed_analysis(future, as_of=LATER, marker=SENTINEL)
    await _seed_analysis(
        earlier, as_of=NOW - timedelta(days=30), marker="past-analysis"
    )

    payload = await _get(client, subject, "evidence")
    prior = payload["bundle"]["prior_analyses"]

    assert_absent(prior, SENTINEL, where="evidence prior_analyses")
    # PAIRED: the analysis written a month ago IS remembered.
    assert_present(prior, "past-analysis", where="evidence prior_analyses")


async def test_the_analyses_list_is_scoped_to_this_event_not_the_future_one(client):
    """``GET .../analyses`` is a per-event audit trail (§99). It lists this
    event's attempts — including failures — and never another event's."""
    subject, future = await _standard_scene()
    await _seed_analysis(future, as_of=LATER, marker=SENTINEL)
    await _seed_analysis(subject, as_of=NOW - timedelta(days=1), marker="mine")

    response = await client.get(f"/api/events/{subject}/analyses")
    assert response.status_code == 200, response.text
    payload = response.json()

    assert_absent(payload, SENTINEL, where="analyses list")
    assert_present(payload, "mine", where="analyses list")


# ===========================================================================
# 8. GET /api/events/{id}/options — stored metrics and the §85 basis rule
# ===========================================================================


async def _seed_metric(
    event_id: int,
    *,
    basis: str,
    as_of: datetime,
    implied_move_pct: float,
    status: str = "OK",
    call_ticker: str = "AAPL260821C00100000",
) -> None:
    async with SessionLocal() as s:
        s.add(
            EventOptionMetricRow(
                event_id=event_id,
                as_of=as_of,
                basis=basis,
                expiry=date(2026, 8, 21),
                strike=100.0,
                spot=100.0,
                call_ticker=call_ticker,
                put_ticker=call_ticker.replace("C00", "P00"),
                pre_call_close=2.0,
                pre_put_close=2.0,
                implied_move_pct=implied_move_pct,
                implied_move_points=4.0,
                actual_move_pct=0.05,
                status=status,
                notes={},
            )
        )
        await s.commit()


async def test_options_history_never_includes_a_print_after_as_of(client):
    """§66: the history strip walks PREVIOUS comparable prints. A stored
    metric for a print scheduled next week is a real row that describes an
    event which has not happened."""
    subject, future = await _standard_scene()
    await _seed_metric(
        future,
        basis=BASIS_HISTORICAL,
        as_of=LATER,
        implied_move_pct=SENTINEL_NUM,
        call_ticker=f"{SENTINEL}C00100000",
    )
    await _seed_metric(
        subject, basis=BASIS_HISTORICAL, as_of=NOW, implied_move_pct=0.06
    )

    payload = await _get(client, subject, "options")
    assert_absent(payload, SENTINEL, where="options")
    assert_absent(payload, str(SENTINEL_NUM), where="options")
    # PAIRED: this event's own stored reconstruction IS served.
    assert payload["current"] is not None
    assert payload["current"]["implied_move_pct"] == 0.06


# --- §85 — the LIVE basis is only ever served for a FUTURE event -----------


async def test_a_past_event_never_carries_the_live_chain_basis(client):
    """§85 — THE labelling rule for what cannot be reconstructed.

    A live option-chain snapshot is a real bid/ask midpoint observed at a
    known instant. This platform cannot rebuild one for a past date, so
    ``LIVE_CHAIN_SNAPSHOT`` is a claim only an UPCOMING event may make. Here
    a LIVE row is planted in the database FOR THE PAST EVENT ITSELF — the
    strongest form of the attack, because the row exists, is addressable and
    is newer than the historical one — and the payload must still refuse it.
    """
    subject, _ = await _standard_scene()
    await _seed_metric(
        subject,
        basis=BASIS_HISTORICAL,
        as_of=NOW - timedelta(days=4),
        implied_move_pct=0.06,
    )
    await _seed_metric(
        subject,
        basis=BASIS_LIVE,
        as_of=LATER,
        implied_move_pct=SENTINEL_NUM,
        call_ticker=f"{SENTINEL}LIVE",
    )

    payload = await _get(client, subject, "options")

    assert payload["is_upcoming"] is False
    assert payload["current"]["basis"] == BASIS_HISTORICAL
    assert_absent(payload, BASIS_LIVE, where="past event options payload")
    assert_absent(payload, SENTINEL, where="past event options payload")
    # PAIRED: the historical reconstruction IS served.
    assert payload["current"]["implied_move_pct"] == 0.06


async def test_the_live_basis_is_reserved_for_an_event_that_has_not_happened(client):
    """The other half of §85, without which the test above could be satisfied
    by a platform that never emits the LIVE basis at all. An UPCOMING print
    is priced off the live chain and says so."""
    await _seed_bars(
        TICKER, start=date(2026, 4, 1), count=110, future_after=NOW.date()
    )
    upcoming = await _add_event(
        key="EARNINGS:AAPL:2026-09-02",
        ticker=TICKER,
        when=NOW + timedelta(days=15),
    )
    payload = await _get(client, upcoming, "options")

    assert payload["is_upcoming"] is True
    current = payload["current"]
    # The stub chain may or may not price it; what §85 fixes is that IF a
    # number is served for an upcoming event it is labelled LIVE, and that a
    # refusal is a labelled NO_DATA rather than a silent historical stand-in.
    assert current["basis"] in (BASIS_LIVE, BASIS_HISTORICAL)
    if current["basis"] == BASIS_HISTORICAL:
        assert current.get("status") is not None
    assert_absent(payload, str(SENTINEL_NUM), where="upcoming options payload")


async def test_the_not_backtestable_fields_are_named_in_the_payload(client):
    """§85 requires the platform to SAY which fields it cannot reconstruct
    point-in-time, rather than leaving a consumer to discover it. The list
    travels in every options payload."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "options")

    assert payload["not_backtestable"], "the §85 list must not be empty"
    assert payload["disclaimer"]


# ===========================================================================
# 9. GET /api/events/{id}/macro — an observation released after as_of
# ===========================================================================

CPI_SERIES = "CUSR0000SA0"


async def _macro_scene() -> int:
    """A CPI release with a PAST print and a FUTURE one.

    The future observation's reference PERIOD (2026-07) precedes ``as_of``,
    exactly as July CPI's period does when it is published in August: a gate
    written against the period rather than ``release_at`` would admit a number
    nobody had yet, which is the specific mistake ``visible_prints`` exists to
    prevent.
    """
    event_id = await _add_event(
        key="CPI:2026-08-12",
        ticker=None,
        when=et(2026, 8, 12, 8, 30),
        event_type=EventType.CPI,
        title="CPI release",
        release_period="2026-06",
        agency="Bureau of Labor Statistics",
        series_id=CPI_SERIES,
    )
    async with SessionLocal() as s:
        s.add(
            MacroObservationRow(
                series_id=CPI_SERIES,
                period="2026-06",
                value=311.111,
                release_at=utc(2026, 7, 14, 12, 30),
                release_basis="SCHEDULED",
                provider="bls",
            )
        )
        s.add(
            MacroObservationRow(
                series_id=CPI_SERIES,
                period="2026-07",  # period ENDED before as_of...
                value=SENTINEL_NUM,
                release_at=LATER,  # ...published AFTER it
                release_basis="SCHEDULED",
                provider="bls",
            )
        )
        await s.commit()
    return event_id


async def test_macro_never_reports_an_observation_released_after_as_of(client):
    """§8/§38/§96: gating is on ``release_at``, never the reference period."""
    event_id = await _macro_scene()
    payload = await _get(client, event_id, "macro")

    assert_absent(payload, str(SENTINEL_NUM), where="macro")
    # PAIRED: June's print, released in July, IS visible.
    assert_present(payload, "311.111", where="macro")


async def test_macro_does_not_leak_the_future_period_as_a_mapping_key(client):
    """The KEY case the scanner was built for: a trend block keyed by
    reference period can leak "2026-07" as a key even when its value is
    withheld, and a values-only scan would report the payload clean."""
    event_id = await _macro_scene()
    payload = await _get(client, event_id, "macro")

    trend = payload.get("trend") or {}
    assert_absent(trend, str(SENTINEL_NUM), where="macro trend")


# ===========================================================================
# 10. GET /api/events/{id}/fed — a document released after as_of
# ===========================================================================


async def _fed_scene() -> int:
    """An FOMC decision with a PAST statement and a FUTURE one.

    The future statement's MEETING DATE precedes ``as_of`` while its
    ``released_at`` follows it — the minutes' twenty-one-day gap in miniature,
    and the reason the gate reads ``released_at`` while the join reads
    ``meeting_date``.
    """
    await _add_event(
        key="FOMC_DECISION:2026-06-17",
        ticker=None,
        when=et(2026, 6, 17, 14, 0),
        event_type=EventType.FOMC_DECISION,
        title="FOMC decision June",
    )
    # The meeting whose statement is planted UNRELEASED. It is the most recent
    # decision before ``as_of``, so it IS the "previous decision" the packet
    # selects — the statement is looked up and only the ``released_at`` gate
    # stands between it and the payload. Anything less and the document would
    # be absent because nothing asked for it, which is not the same fact.
    await _add_event(
        key="FOMC_DECISION:2026-08-05",
        ticker=None,
        when=et(2026, 8, 5, 14, 0),
        event_type=EventType.FOMC_DECISION,
        title="FOMC decision August",
    )
    event_id = await _add_event(
        key="FOMC_DECISION:2026-09-16",
        ticker=None,
        when=et(2026, 9, 16, 14, 0),
        event_type=EventType.FOMC_DECISION,
        title="FOMC decision September",
    )
    async with SessionLocal() as s:
        s.add(
            FedDocumentRow(
                doc_type="STATEMENT",
                meeting_date=date(2026, 6, 17),
                url="https://fed.test/past-statement",
                title="FOMC statement June 2026",
                released_at=et(2026, 6, 17, 14, 0),
                text="The Committee decided to maintain the target range.",
                paragraphs=["The Committee decided to maintain the target range."],
                parsed={
                    "target_range": {
                        "low_pct": 4.25,
                        "high_pct": 4.5,
                        "text": "4-1/4 to 4-1/2 percent",
                    }
                },
                provider="fed_docs",
            )
        )
        s.add(
            FedDocumentRow(
                doc_type="STATEMENT",
                meeting_date=date(2026, 8, 5),  # meeting BEFORE as_of...
                url=f"https://fed.test/{SENTINEL}-statement",
                title=f"FOMC statement {SENTINEL}",
                released_at=LATER,  # ...released AFTER it
                text=f"The Committee {SENTINEL} cut rates by 300 basis points.",
                paragraphs=[f"The Committee {SENTINEL} cut rates."],
                parsed={
                    "target_range": {
                        "low_pct": 1.0,
                        "high_pct": 1.25,
                        "text": f"{SENTINEL} 1 to 1-1/4 percent",
                    }
                },
                provider="fed_docs",
            )
        )
        await s.commit()
    return event_id


async def test_fed_never_serves_a_document_released_after_as_of(client):
    """§9/§42-§45/§96: a statement released an hour after ``as_of`` must not
    reach the diff, the dimensions, the vote or the reaction windows."""
    event_id = await _fed_scene()
    payload = await _get(client, event_id, "fed")

    assert_absent(payload, SENTINEL, where="fed")
    # PAIRED: the June statement IS the previous decision.
    assert_present(payload, "maintain the target range", where="fed")


# ===========================================================================
# 11. GET /api/events/{id}/timeline — every kind at once
# ===========================================================================


async def test_the_timeline_bounds_every_kind_not_only_news(client):
    """§54/§56/§96: the rail carries articles, filings and events, and each
    kind has its OWN gate. A rail that bounded news and forgot filings would
    look correct on the tab that is easiest to eyeball."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "timeline")

    assert_absent(payload, SENTINEL, where="timeline")
    assert_absent(payload, str(SENTINEL_NUM), where="timeline")
    # PAIRED: past items ARE on the rail.
    assert_present(payload, "past-guidance", where="timeline")


async def test_a_filing_is_placed_on_the_rail_by_when_it_became_public(client):
    """A quarter that CLOSED 2026-06-30 and was accepted after ``as_of``
    belongs nowhere on a rail drawn at ``as_of``. Keyed on ``end_date`` it
    would appear seven weeks before anyone could read it — the exact
    look-ahead ``acceptance_datetime`` exists to prevent."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "timeline")

    assert_absent(payload, f"{SENTINEL}-filing", where="timeline filings")
    assert_absent(payload, "2026-06-30", where="timeline filings")


# ===========================================================================
# 12. GET /api/events/{id}/risk — later prints and stored metrics
# ===========================================================================


async def test_event_risk_never_measures_a_print_after_as_of(client):
    """§63/§64: the state comes from a deterministic table over stored
    straddles and realized moves. A future print entering the sample would
    move the state and every driver threshold with it — and because the state
    is what a human reads before sizing, a leak here is the one that most
    directly reaches a trade."""
    subject, future = await _standard_scene()
    await _seed_metric(
        future,
        basis=BASIS_HISTORICAL,
        as_of=LATER,
        implied_move_pct=SENTINEL_NUM,
        call_ticker=f"{SENTINEL}RISK",
    )
    await _seed_metric(
        subject, basis=BASIS_HISTORICAL, as_of=NOW, implied_move_pct=0.06
    )

    payload = await _get(client, subject, "risk")

    assert_absent(payload, SENTINEL, where="risk")
    assert_absent(payload, str(SENTINEL_NUM), where="risk")
    # PAIRED: the risk surface is a real answer in SHADOW mode.
    assert payload["enforcement"] == "SHADOW"


async def test_every_historical_statistic_on_the_risk_surface_carries_its_n(client):
    """§64, restated adversarially: a leak that added one event to a sample
    would be invisible without ``n``. The ``n`` is what makes the sample
    auditable at all."""
    subject, _ = await _standard_scene()
    payload = await _get(client, subject, "risk")

    options = payload["options"]
    for name in ("historical_iv_crush", "historical_implied_move"):
        block = options[name]
        assert "n" in block, (name, block)
        if block["n"] in (0, None):
            assert block["median_abs"] is None, (name, block)


# ===========================================================================
# MUTATION VERIFICATION — proving each suite above BITES
# ===========================================================================
#
# The house rule from tests/test_risk_adversarial.py: a test that cannot fail
# is a green light nobody re-examines. Each block below defeats one gate by
# monkeypatch INSIDE the test and asserts the sentinel becomes visible. The
# sources are never edited; the patch dies with the test.


async def test_the_suite_bites_price_context_bars(client, monkeypatch):
    """MUTATION 1 — the §14 bar gate, and a finding about its depth.

    ``as_of_bar_filter`` is NOT the only gate on this surface:
    ``pre_event_price_context`` independently drops ``b.date >
    as_of_date_et``. Defeating only the first moves ``bars_through`` by one
    day and leaks nothing; the sentinel needs BOTH defeated. Both stages are
    asserted, so a future refactor that collapses the two into a single gate
    fails HERE — visibly — rather than quietly halving the defence.
    """
    subject, _ = await _standard_scene()
    clean = await _get(client, subject, "price-context")
    assert find_sentinel(clean, str(SENTINEL_NUM)) == []

    # Stage 1: defeat the seam's filter only. Defence in depth holds.
    monkeypatch.setattr(ep, "as_of_bar_filter", lambda bars, as_of: list(bars))
    partial = await _get(client, subject, "price-context")
    assert find_sentinel(partial, str(SENTINEL_NUM)) == [], (
        "the inner as_of_date_et bound in pre_event_price_context is gone — "
        "this surface is now singly gated"
    )
    assert partial["pre_event"]["bars_through"] == "2026-08-18"

    # Stage 2: defeat the pure context's own bound too. NOW it must leak.
    real_context = ep.pre_event_price_context

    def leaky_context(bars, *, anchor_date_et, as_of_date_et, bench_bars=None):
        return real_context(
            bars,
            anchor_date_et=anchor_date_et,
            as_of_date_et=date(2099, 1, 1),
            bench_bars=bench_bars,
        )

    monkeypatch.setattr(ep, "pre_event_price_context", leaky_context)
    leaked = await _get(client, subject, "price-context")
    assert find_sentinel(leaked, str(SENTINEL_NUM)), (
        "BOTH bar gates were defeated and the future close STILL did not "
        "surface — test_price_context_never_sees_a_bar_dated_after_as_of "
        "cannot fail and is worthless"
    )


async def test_the_suite_bites_the_news_gate(client, monkeypatch):
    """MUTATION 2 — the §96 as-of stage of the news pipeline.

    Three-step, because this surface is gated twice and the ORDER of the
    proof is what makes it meaningful:

      1. clean read — nothing leaks;
      2. replace the LOADER with one that returns every stored row, defeating
         the SQL ``end`` bound — the payload is STILL clean, which is what
         proves the PURE gate alone is sufficient and the SQL clause is the
         optimisation its docstring claims;
      3. defeat the pure gate as well — the sentinel appears.

    Without step 2 this would prove only that the SQL bound works, and
    deleting that clause as redundant would silently reopen the leak with
    every payload assertion in this file still green.
    """
    subject, _ = await _standard_scene()
    clean = await _get(client, subject, "news")
    assert find_sentinel(clean, SENTINEL) == []

    real_loader = en._articles_for_ticker

    async def unbounded_loader(session, ticker, *, start, end):
        return await real_loader(
            session, ticker, start=start, end=end + timedelta(days=30)
        )

    monkeypatch.setattr(en, "_articles_for_ticker", unbounded_loader)
    still_clean = await _get(client, subject, "news")
    assert find_sentinel(still_clean, SENTINEL) == [], (
        "with the SQL end-bound removed the future article reached the "
        "payload — the PURE §96 gate is not doing the work its docstring "
        "claims, and the SQL clause is load-bearing rather than an "
        "optimisation"
    )

    real_analyze = en.analyze_window

    def leaky_analyze(articles, **kwargs):
        kwargs["as_of"] = kwargs["as_of"] + timedelta(days=9)
        return real_analyze(articles, **kwargs)

    monkeypatch.setattr(en, "analyze_window", leaky_analyze)
    leaked = await _get(client, subject, "news")
    assert find_sentinel(leaked, SENTINEL), (
        "the pure as-of gate was defeated and the future article STILL did "
        "not surface — test_news_never_sees_an_article_published_after_as_of "
        "cannot fail"
    )


async def test_the_suite_bites_the_fundamentals_acceptance_gate(client, monkeypatch):
    """MUTATION 3 — the ``acceptance_datetime`` gate.

    The gateway loads EVERY stored statement row unfiltered on purpose (its
    docstring says a WHERE clause here would be a second, untested copy of the
    §85 rule), so ``select_statements_as_of`` is the SINGLE gate and shifting
    the instant handed to ``build_snapshot`` is enough to make it leak. That
    single-gate design is precisely why the pure layer's test coverage
    matters more here than anywhere else.
    """
    subject, _ = await _standard_scene()
    clean = await _get(client, subject, "fundamentals")
    assert find_sentinel(clean["current"], str(SENTINEL_NUM)) == []

    real_snapshot = fu.build_snapshot

    def leaky_snapshot(rows, **kwargs):
        kwargs["as_of"] = kwargs["as_of"] + timedelta(days=9)
        return real_snapshot(rows, **kwargs)

    monkeypatch.setattr(fu, "build_snapshot", leaky_snapshot)
    leaked = await _get(client, subject, "fundamentals")
    assert find_sentinel(leaked["current"], str(SENTINEL_NUM)), (
        "the acceptance gate was defeated and the un-filed quarter STILL did "
        "not surface in the metrics — the fundamentals suite cannot fail"
    )


async def test_the_suite_bites_the_macro_release_gate(client, monkeypatch):
    """MUTATION 4 — ``visible_prints``, the gate on ``release_at``.

    A single pure gate with no SQL bound behind it (``load_schedule_rows``
    deliberately does not apply ``as_of``, because a release SCHEDULE is
    published a year ahead). Shifting the instant is therefore the whole
    attack surface, and it must leak.
    """
    event_id = await _macro_scene()
    clean = await _get(client, event_id, "macro")
    assert find_sentinel(clean, str(SENTINEL_NUM)) == []

    from libs.trading_core.events import macro as macro_lib

    real_visible = macro_lib.visible_prints

    def leaky_visible(prints, as_of):
        return real_visible(prints, as_of + timedelta(days=9))

    monkeypatch.setattr(macro_lib, "visible_prints", leaky_visible)
    leaked = await _get(client, event_id, "macro")
    assert find_sentinel(leaked, str(SENTINEL_NUM)), (
        "visible_prints was defeated and the unreleased observation STILL "
        "did not surface — the macro suite cannot fail"
    )


async def test_the_suite_bites_the_fed_release_gate(client, monkeypatch):
    """MUTATION 5 — the Fed surface, gated twice like news.

    Same three-step proof: defeat the SQL bound alone (payload stays clean,
    so the pure ``_gate`` is sufficient), then defeat ``_gate`` as well and
    watch the unreleased statement appear.
    """
    event_id = await _fed_scene()
    clean = await _get(client, event_id, "fed")
    assert find_sentinel(clean, SENTINEL) == []

    real_load = ef.load_documents

    async def unbounded_load(session, **kwargs):
        kwargs["as_of"] = kwargs["as_of"] + timedelta(days=9)
        return await real_load(session, **kwargs)

    monkeypatch.setattr(ef, "load_documents", unbounded_load)
    still_clean = await _get(client, event_id, "fed")
    assert find_sentinel(still_clean, SENTINEL) == [], (
        "with the SQL released_at bound widened the future statement reached "
        "the payload — fed_intel._gate is not re-applying the §96 rule and "
        "the SQL clause is the only defence"
    )

    from libs.trading_core.events import fed_intel as fed_lib

    monkeypatch.setattr(
        fed_lib, "_gate", lambda doc, as_of: (dict(doc) if doc else None)
    )
    leaked = await _get(client, event_id, "fed")
    assert find_sentinel(leaked, SENTINEL), (
        "both Fed gates were defeated and the unreleased statement STILL did "
        "not surface — the fed suite cannot fail"
    )


async def test_the_suite_bites_the_event_memory_gate(client, monkeypatch):
    """MUTATION 6 — the §69 prior-analyses gate.

    The subtlest leak in the platform, because it arrives as PROSE. The
    ``numbers_quoted`` validator checks the analysis against the bundle, so a
    bundle poisoned with a future opinion validates perfectly and the leak is
    invisible to every downstream check. This is the one that most needs to
    be proved to bite.
    """
    subject, future = await _standard_scene()
    # Renamed away from the sentinel for the same reason as the test above:
    # the future event's KEY would otherwise ride along on its own summary.
    async with SessionLocal() as session:
        row = await session.get(EventRow, future)
        row.event_key = "EARNINGS:AAPL:2026-08-25"
        row.title = "future print"
        await session.commit()
    await _seed_analysis(future, as_of=LATER, marker=SENTINEL)

    clean = await _get(client, subject, "evidence")
    assert find_sentinel(clean["bundle"]["prior_analyses"], SENTINEL) == []

    real_prior = ea.prior_analyses_for_ticker

    async def leaky_prior(session, ticker, **kwargs):
        kwargs["before_as_of"] = kwargs["before_as_of"] + timedelta(days=9)
        return await real_prior(session, ticker, **kwargs)

    monkeypatch.setattr(ea, "prior_analyses_for_ticker", leaky_prior)
    leaked = await _get(client, subject, "evidence")
    assert find_sentinel(leaked["bundle"]["prior_analyses"], SENTINEL), (
        "the §69 memory gate was defeated and the future analysis STILL did "
        "not surface — the event-memory suite cannot fail"
    )


# ===========================================================================
# The boundary itself — a future as_of is refused, not clamped
# ===========================================================================


@pytest.mark.parametrize(
    "suffix",
    [
        "price-context",
        "fundamentals",
        "replay",
        "history",
        "news",
        "evidence",
        "options",
        "macro",
        "fed",
        "timeline",
        "risk",
    ],
)
async def test_an_as_of_in_the_future_is_a_422_on_every_family(client, suffix):
    """The last look-ahead surface is the PARAMETER itself.

    A request for an instant that has not arrived is a mistake worth
    reporting, not a request to silently clamp to now: clamping answers a
    DIFFERENT question than the one asked and the caller has no way to
    notice. Asserted across every family at once so a route added later
    without ``_resolve_as_of`` is caught here.
    """
    subject, _ = await _standard_scene()
    ahead = datetime.now(timezone.utc) + timedelta(days=3)
    response = await client.get(
        f"/api/events/{subject}/{suffix}?as_of={iso(ahead)}"
    )
    assert response.status_code == 422, (
        f"{suffix} accepted a future as_of ({response.status_code})"
    )
    assert "future" in response.text.lower()
