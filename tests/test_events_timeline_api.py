"""Event timeline + §54 card summaries — GET /api/events/{id}/timeline and the
``summaries=true`` flag on GET /api/events (event spec §54, §56, §57, §96;
audit §7.1, §7.2, §11.5 Phase J).

The guarantees these tests defend, in the order they appear:

1. **The rail is bracketed by the two prints.** ``anchors.previous_event`` is
   the last comparable event and ``anchors.next_event`` is this one, so
   "since the last print" is a claim the payload can actually support. With
   no predecessor the window falls back to a stated 120 days rather than to a
   silent "all of history", and ``window.basis`` names which of the two
   happened.
2. **``as_of`` bounds every kind, not just news** (§96). Asserted PAIRED
   throughout: the post-``as_of`` article/filing/event is invisible at one
   instant AND visible at a later one. A gate that returned nothing would
   pass the first half of each pair and fail the second.
3. **The GET never fetches** (§27; audit §7.2 rule 1). Proved by patching the
   news seam's ``get_provider`` to explode — a read that reached a vendor
   would raise rather than quietly succeed, so this cannot rot into a no-op.
4. **A filing is dated by when it became PUBLIC**, not by when its period
   ended. A quarter closing 2026-06-30 and accepted 2026-07-28 belongs on the
   rail in July; keyed on ``end_date`` it would appear three weeks before
   anyone could read it, which is the exact look-ahead ``acceptance_datetime``
   exists to prevent.
5. **Summaries are OPT-IN and byte-identical when off.** The whole list
   payload is compared key-for-key against the un-flagged one, because a card
   feature that silently changed an existing consumer's response shape would
   be a regression nothing else in the suite would catch.
6. **A summary never quotes a retracted number** (§44 rule 18). A NO_DATA
   option row contributes nothing — not even a leftover ``implied_move_pct``
   — and a median always travels with its ``n``.

Uses the shared ``client`` fixture (conftest.py): providers "stub", execution
"simulated".
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from apps.gateway import event_news as news_seam
from apps.gateway.db import (
    EventAnalysisRow,
    EventOptionMetricRow,
    EventRow,
    FundamentalStatementRow,
    NewsArticleRow,
    SessionLocal,
)
from apps.gateway.event_timeline import MAX_TIMELINE_ITEMS, _sort_key, _truncate
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

#: A fixed instant rather than ``now()`` so every window boundary and decay
#: factor below is a number a reader can check by hand instead of a value that
#: drifts overnight.
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _iso(when: datetime) -> str:
    """An instant as the query-string form the UI sends."""
    return when.isoformat().replace("+00:00", "Z")


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
    previous_event_id: int | None = None,
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
            previous_event_id=previous_event_id,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_article(
    source_id: str,
    title: str,
    *,
    published_at: datetime,
    publisher: str = "Reuters",
    tickers: tuple[str, ...] = ("AAPL",),
) -> int:
    async with SessionLocal() as s:
        row = NewsArticleRow(
            source_id=source_id,
            title=title,
            publisher=publisher,
            published_at=published_at,
            url=f"https://news.test/{source_id}",
            tickers=list(tickers),
            description="",
            fetched_at=published_at,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_filing(
    *,
    ticker: str = "AAPL",
    fiscal_period: str = "Q2",
    fiscal_year: int = 2026,
    end: date,
    accepted: datetime | None,
    timeframe: str = "quarterly",
    url: str | None = "https://sec.test/filing",
) -> int:
    async with SessionLocal() as s:
        row = FundamentalStatementRow(
            ticker=ticker,
            timeframe=timeframe,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            start_date=end - timedelta(days=90),
            end_date=end,
            acceptance_datetime=accepted,
            source_filing_url=url,
            values={"income_statement.revenues": 1.0},
            raw_fields_count=1,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_analysis(
    event_id: int,
    *,
    as_of: datetime,
    created_at: datetime,
    status: str = "OK",
    regime: str | None = "HIGH_BAR",
    confidence: str | None = "MEDIUM",
    digest: str = "d0",
) -> int:
    async with SessionLocal() as s:
        row = EventAnalysisRow(
            event_id=event_id,
            as_of=as_of,
            kind="PRE_EVENT",
            bundle={"facts": []},
            bundle_digest=digest,
            analysis=(
                None
                if regime is None
                else {
                    "expectations_gap_regime": regime,
                    "confidence": confidence,
                    "executive_summary": "…",
                }
            ),
            model="stub-model",
            prompt_version="v1",
            status=status,
            created_at=created_at,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_metric(
    event_id: int,
    *,
    basis: str = "live_chain",
    as_of: datetime = NOW,
    implied_move_pct: float | None = None,
    actual_move_pct: float | None = None,
    status: str = "OK",
) -> int:
    async with SessionLocal() as s:
        row = EventOptionMetricRow(
            event_id=event_id,
            as_of=as_of,
            basis=basis,
            implied_move_pct=implied_move_pct,
            actual_move_pct=actual_move_pct,
            status=status,
            notes={},
        )
        s.add(row)
        await s.commit()
        return row.id


async def _seed_two_stories(ticker: str = "AAPL", *, at: datetime = NOW) -> None:
    """Two distinct MATERIAL developments plus one syndicated copy of the first.

    Hand-written rather than generated so the expected outcome is readable by
    eye: a GUIDANCE story carried twice minutes apart by two publishers, and a
    separate REGULATION story. §23 folds the syndicated pair into ONE
    development of two articles and must not fuse the antitrust probe into it
    — which is exactly what the timeline's ``article_count`` then reports.
    """
    await _seed_article(
        "guid-reuters",
        "Apple raises full-year guidance on strong iPhone demand",
        publisher="Reuters",
        published_at=at - timedelta(days=2),
        tickers=(ticker,),
    )
    await _seed_article(
        "guid-benzinga",
        "Apple lifts full year guidance on strong iPhone demand",
        publisher="Benzinga",
        published_at=at - timedelta(days=2) + timedelta(minutes=20),
        tickers=(ticker,),
    )
    await _seed_article(
        "reg-bloomberg",
        "DOJ opens antitrust probe into Apple App Store",
        publisher="Bloomberg",
        published_at=at - timedelta(days=5),
        tickers=(ticker,),
    )


async def _timeline(client, event_id: int, *, as_of: datetime | None = NOW) -> dict:
    url = f"/api/events/{event_id}/timeline"
    if as_of is not None:
        url += f"?as_of={_iso(as_of)}"
    response = await client.get(url)
    assert response.status_code == 200, response.text
    return response.json()


def _kinds(payload: dict, kind: str) -> list[dict]:
    return [item for item in payload["items"] if item["kind"] == kind]


def _titles(payload: dict) -> list[str]:
    return [item.get("title") or "" for item in payload["items"]]


class _ExplodingProvider:
    """Any provider call is a test failure — proves the GET never fetches."""

    def get_news_window(self, **kwargs):  # noqa: D102
        raise AssertionError("the timeline read path must never call a provider")

    def get_news(self, *args, **kwargs):  # noqa: D102
        raise AssertionError("the timeline read path must never call a provider")


@pytest.fixture(autouse=True)
def _clear_news_throttle():
    """The news seam throttles provider attempts in a process-local dict."""
    news_seam._fetch_attempts.clear()
    yield
    news_seam._fetch_attempts.clear()


async def _earnings_pair(*, ticker: str = "AAPL") -> tuple[int, int]:
    """A previous CONFIRMED print and the upcoming one, chained.

    The shape almost every test below wants: the rail's two ends, with the
    upcoming event's ``previous_event_id`` pointing at the old one so the §54
    median walk has a chain to follow as well.
    """
    previous = await _add_event(
        key=f"EARNINGS:{ticker}:2026-05-01",
        ticker=ticker,
        when=_utc(2026, 5, 1, 20, 30),
    )
    upcoming = await _add_event(
        key=f"EARNINGS:{ticker}:2026-08-27",
        ticker=ticker,
        when=_utc(2026, 8, 27, 20, 30),
        previous_event_id=previous,
    )
    return previous, upcoming


# ---------------------------------------------------------------------------
# 1. The anchors and the window — what "since the last event" is measured over
# ---------------------------------------------------------------------------


async def test_the_rail_is_bracketed_by_the_previous_and_next_prints(client):
    previous, upcoming = await _earnings_pair()

    payload = await _timeline(client, upcoming)

    anchors = payload["anchors"]
    assert anchors["previous_event"]["event_id"] == previous
    assert anchors["previous_event"]["event_key"] == "EARNINGS:AAPL:2026-05-01"
    assert anchors["next_event"]["event_id"] == upcoming
    assert anchors["as_of"] == payload["as_of"]
    assert payload["window"]["basis"] == "previous_earnings:EARNINGS:AAPL:2026-05-01"
    assert payload["window"]["start"] == _utc(2026, 5, 1, 20, 30).isoformat()


async def test_an_anchor_day_is_the_events_own_zone_not_utc(client):
    """A 20:30 ET print is 00:30 UTC the NEXT day; the label must not slip.

    The one-line bug this pins is a ``.date()`` taken on the UTC instant,
    which would move every after-market earnings anchor a calendar day
    forward — off by one for precisely the events this rail brackets.
    """
    _, upcoming = await _earnings_pair()

    payload = await _timeline(client, upcoming)

    # 2026-05-01 20:30 UTC is 16:30 ET the SAME day.
    assert payload["anchors"]["previous_event"]["date_et"] == "2026-05-01"
    assert payload["anchors"]["previous_event"]["event_timezone"] == "America/New_York"


async def test_with_no_previous_event_the_window_falls_back_to_a_stated_120_days(
    client,
):
    """A newly covered symbol gets a NAMED fallback, not silent all-of-history."""
    upcoming = await _add_event(
        key="EARNINGS:NEW:2026-08-27", ticker="NEW", when=_utc(2026, 8, 27, 20, 30)
    )

    payload = await _timeline(client, upcoming)

    assert payload["anchors"]["previous_event"] is None
    assert payload["window"]["basis"] == "default_120d"
    assert payload["window"]["start"] == (NOW - timedelta(days=120)).isoformat()
    assert payload["window"]["days"] == pytest.approx(120.0)


async def test_an_estimated_past_date_does_not_anchor_the_window(client):
    """An ESTIMATED predecessor is a DERIVATION, not a day anybody reported on.

    Framing four months of news against a date the platform guessed would put
    a fiction at the start of the rail, so the window falls back instead —
    and says so.
    """
    await _add_event(
        key="EARNINGS:AAPL:2026-05-01",
        ticker="AAPL",
        when=_utc(2026, 5, 1, 20, 30),
        status=EventStatus.ESTIMATED,
    )
    upcoming = await _add_event(
        key="EARNINGS:AAPL:2026-08-27", ticker="AAPL", when=_utc(2026, 8, 27, 20, 30)
    )

    payload = await _timeline(client, upcoming)

    assert payload["anchors"]["previous_event"] is None
    assert payload["window"]["basis"] == "default_120d"


# ---------------------------------------------------------------------------
# 2. Ordering and the kinds on the rail
# ---------------------------------------------------------------------------


async def test_items_are_chronological_and_carry_every_kind(client):
    previous, upcoming = await _earnings_pair()
    await _seed_two_stories()
    await _seed_filing(end=date(2026, 6, 30), accepted=_utc(2026, 7, 28, 21, 5))
    await _add_event(
        key="CORPORATE_EVENT:AAPL:2026-06-10",
        ticker="AAPL",
        when=_utc(2026, 6, 10, 13, 30),
        event_type=EventType.CORPORATE_EVENT,
        title="Dividend declared",
    )
    await _seed_analysis(
        upcoming, as_of=_utc(2026, 8, 15), created_at=_utc(2026, 8, 15, 9)
    )

    payload = await _timeline(client, upcoming)

    stamps = [item["at"] for item in payload["items"]]
    assert stamps == sorted(stamps), "the rail must be ascending by instant"
    assert {item["kind"] for item in payload["items"]} == {
        "NEWS",
        "FILING",
        "EVENT",
        "ANALYSIS",
    }
    assert payload["counts"]["total"] == len(payload["items"])
    assert payload["counts"]["by_kind"]["FILING"] == 1
    assert payload["counts"]["by_kind"]["EVENT"] == 1
    assert payload["counts"]["by_kind"]["ANALYSIS"] == 1


async def test_a_syndicated_pair_is_one_development_that_two_outlets_carried(client):
    """§23: duplicated coverage must not inflate importance."""
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()

    payload = await _timeline(client, upcoming)

    news = _kinds(payload, "NEWS")
    assert len(news) == 2, [item["title"] for item in news]
    guidance = next(item for item in news if "guidance" in item["title"].lower())
    assert guidance["article_count"] == 2
    assert guidance["publisher"] == "Reuters"
    assert guidance["evidence_id"] == "news:guid-reuters"
    assert guidance["url"] == "https://news.test/guid-reuters"
    assert guidance["cluster_id"]
    assert guidance["score"] is not None


async def test_news_items_are_counted_by_their_category(client):
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()

    payload = await _timeline(client, upcoming)

    by_category = payload["counts"]["by_category"]
    assert sum(by_category.values()) == payload["counts"]["by_kind"]["NEWS"]
    assert set(by_category) == {
        item["category"] for item in _kinds(payload, "NEWS")
    }


async def test_the_two_anchors_are_never_repeated_as_items(client):
    """Drawing the last print as an item too would imply two prints happened."""
    previous, upcoming = await _earnings_pair()

    payload = await _timeline(client, upcoming)

    event_ids = {item.get("event_id") for item in _kinds(payload, "EVENT")}
    assert previous not in event_ids
    assert upcoming not in event_ids


async def test_another_issuers_events_are_not_on_this_rail(client):
    _, upcoming = await _earnings_pair()
    await _add_event(
        key="CORPORATE_EVENT:MSFT:2026-06-10",
        ticker="MSFT",
        when=_utc(2026, 6, 10, 13, 30),
        event_type=EventType.CORPORATE_EVENT,
    )
    await _seed_article(
        "msft-news",
        "Microsoft raises full-year guidance",
        published_at=NOW - timedelta(days=3),
        tickers=("MSFT",),
    )

    payload = await _timeline(client, upcoming)

    assert _kinds(payload, "EVENT") == []
    assert all("Microsoft" not in title for title in _titles(payload))


async def test_only_ok_analyses_reach_the_rail(client):
    """An INVALID answer quoted a number that was not in its evidence (§99).

    It stays listed on ``/analyses`` — the record of what the model tried is
    part of what makes this auditable — but a timeline of findings that
    included it would read as a finding.
    """
    _, upcoming = await _earnings_pair()
    await _seed_analysis(
        upcoming,
        as_of=_utc(2026, 8, 10),
        created_at=_utc(2026, 8, 10, 9),
        status="INVALID",
        digest="bad",
    )
    await _seed_analysis(
        upcoming,
        as_of=_utc(2026, 8, 12),
        created_at=_utc(2026, 8, 12, 9),
        status="FAILED",
        regime=None,
        digest="worse",
    )
    await _seed_analysis(
        upcoming,
        as_of=_utc(2026, 8, 15),
        created_at=_utc(2026, 8, 15, 9),
        digest="good",
    )

    payload = await _timeline(client, upcoming)

    analyses = _kinds(payload, "ANALYSIS")
    assert len(analyses) == 1
    assert analyses[0]["regime"] == "HIGH_BAR"
    assert analyses[0]["confidence"] == "MEDIUM"
    assert analyses[0]["at"] == _utc(2026, 8, 15, 9).isoformat()


# ---------------------------------------------------------------------------
# 3. The as-of gate (§96) — asserted PAIRED for every kind
# ---------------------------------------------------------------------------


async def test_an_article_published_after_as_of_is_excluded(client):
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()
    await _seed_article(
        "later-buyback",
        "Apple announces a $100 billion buyback",
        published_at=NOW + timedelta(hours=1),
    )

    payload = await _timeline(client, upcoming, as_of=NOW)

    assert all("buyback" not in title.lower() for title in _titles(payload))


async def test_the_same_article_is_on_the_rail_one_hour_later(client):
    """The paired half: without it, a gate that returned nothing would pass."""
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()
    await _seed_article(
        "later-buyback",
        "Apple announces a $100 billion buyback",
        published_at=NOW + timedelta(hours=1),
    )

    payload = await _timeline(client, upcoming, as_of=NOW + timedelta(hours=2))

    assert any("buyback" in title.lower() for title in _titles(payload))


async def test_a_filing_accepted_after_as_of_is_excluded_and_later_included(client):
    _, upcoming = await _earnings_pair()
    await _seed_filing(
        end=date(2026, 6, 30), accepted=NOW + timedelta(hours=1), fiscal_period="Q2"
    )

    before = await _timeline(client, upcoming, as_of=NOW)
    after = await _timeline(client, upcoming, as_of=NOW + timedelta(hours=2))

    assert _kinds(before, "FILING") == []
    assert len(_kinds(after, "FILING")) == 1


async def test_a_registry_event_after_as_of_is_excluded_and_later_included(client):
    _, upcoming = await _earnings_pair()
    await _add_event(
        key="CORPORATE_EVENT:AAPL:2026-08-18",
        ticker="AAPL",
        when=NOW + timedelta(hours=1),
        event_type=EventType.CORPORATE_EVENT,
    )

    before = await _timeline(client, upcoming, as_of=NOW)
    after = await _timeline(client, upcoming, as_of=NOW + timedelta(hours=2))

    assert _kinds(before, "EVENT") == []
    assert len(_kinds(after, "EVENT")) == 1


async def test_an_analysis_created_after_as_of_is_excluded_and_later_included(client):
    _, upcoming = await _earnings_pair()
    await _seed_analysis(
        upcoming,
        as_of=_utc(2026, 8, 18),
        created_at=NOW + timedelta(hours=1),
    )

    before = await _timeline(client, upcoming, as_of=NOW)
    after = await _timeline(client, upcoming, as_of=NOW + timedelta(hours=2))

    assert _kinds(before, "ANALYSIS") == []
    assert len(_kinds(after, "ANALYSIS")) == 1


async def test_every_item_is_at_or_before_as_of(client):
    """The blanket invariant, over a rail carrying all four kinds at once."""
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()
    await _seed_filing(end=date(2026, 6, 30), accepted=_utc(2026, 7, 28, 21, 5))
    await _add_event(
        key="CORPORATE_EVENT:AAPL:2026-06-10",
        ticker="AAPL",
        when=_utc(2026, 6, 10, 13, 30),
        event_type=EventType.CORPORATE_EVENT,
    )
    await _seed_analysis(
        upcoming, as_of=_utc(2026, 8, 15), created_at=_utc(2026, 8, 15, 9)
    )

    payload = await _timeline(client, upcoming, as_of=NOW)

    assert payload["items"]
    assert all(item["at"] <= payload["as_of"] for item in payload["items"])


async def test_a_future_as_of_is_a_422_not_a_silent_clamp(client):
    """Clamping would answer a DIFFERENT question with no way to notice."""
    _, upcoming = await _earnings_pair()

    response = await client.get(
        f"/api/events/{upcoming}/timeline"
        f"?as_of={_iso(datetime.now(timezone.utc) + timedelta(days=3))}"
    )

    assert response.status_code == 422
    assert "future" in response.json()["detail"]


async def test_a_missing_event_is_the_only_404(client):
    response = await client.get("/api/events/999999/timeline")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 4. A filing is dated by when it became PUBLIC
# ---------------------------------------------------------------------------


async def test_a_filing_sits_at_its_acceptance_instant_not_its_period_end(client):
    """Keyed on ``end_date`` it would land three weeks before anyone could read it."""
    _, upcoming = await _earnings_pair()
    await _seed_filing(end=date(2026, 6, 30), accepted=_utc(2026, 7, 28, 21, 5))

    payload = await _timeline(client, upcoming)

    filing = _kinds(payload, "FILING")[0]
    assert filing["at"] == _utc(2026, 7, 28, 21, 5).isoformat()
    assert filing["period_end"] == "2026-06-30"
    assert filing["fiscal_period"] == "Q2"
    assert filing["fiscal_year"] == 2026
    assert filing["timeframe"] == "quarterly"
    assert filing["source_url"] == "https://sec.test/filing"


async def test_a_filing_with_no_acceptance_instant_is_left_off_the_rail(client):
    """The row is real and stays STORED; its publication instant is unknown.

    Dating it from ``filing_date`` would be a guess, and a guessed publication
    instant is exactly the look-ahead the acceptance gate exists to stop.
    """
    _, upcoming = await _earnings_pair()
    await _seed_filing(end=date(2026, 6, 30), accepted=None)

    payload = await _timeline(client, upcoming)

    assert _kinds(payload, "FILING") == []


# ---------------------------------------------------------------------------
# 5. The read never fetches (§27; audit §7.2 rule 1)
# ---------------------------------------------------------------------------


async def test_the_timeline_never_calls_a_provider(client, monkeypatch):
    _, upcoming = await _earnings_pair()
    await _seed_two_stories()
    monkeypatch.setattr(
        news_seam, "get_provider", lambda *a, **k: _ExplodingProvider()
    )

    payload = await _timeline(client, upcoming)

    assert _kinds(payload, "NEWS")


async def test_an_empty_window_is_zero_counts_not_a_lazy_backfill(client):
    _, upcoming = await _earnings_pair()

    payload = await _timeline(client, upcoming)

    assert payload["items"] == []
    assert payload["counts"]["total"] == 0
    assert payload["counts"]["by_kind"] == {
        "NEWS": 0,
        "FILING": 0,
        "EVENT": 0,
        "ANALYSIS": 0,
    }
    assert payload["available"] is True


# ---------------------------------------------------------------------------
# 6. A macro event has no issuer
# ---------------------------------------------------------------------------


async def test_a_macro_event_says_no_ticker_at_200(client):
    """A CPI release has no issuer whose news, filings or siblings these are."""
    cpi = await _add_event(
        key="CPI:2026-08",
        ticker=None,
        when=_utc(2026, 8, 27, 12, 30),
        event_type=EventType.CPI,
        title="CPI",
    )

    payload = await _timeline(client, cpi)

    assert payload["available"] is False
    assert payload["reason"] == "no_ticker"
    assert _kinds(payload, "NEWS") == []
    assert _kinds(payload, "FILING") == []


async def test_a_macro_events_own_analyses_are_still_on_its_rail(client):
    """It has no tape, but the platform's own view of it is still a fact."""
    cpi = await _add_event(
        key="CPI:2026-08",
        ticker=None,
        when=_utc(2026, 8, 27, 12, 30),
        event_type=EventType.CPI,
        title="CPI",
    )
    await _seed_analysis(
        cpi, as_of=_utc(2026, 8, 15), created_at=_utc(2026, 8, 15, 9)
    )

    payload = await _timeline(client, cpi)

    assert len(_kinds(payload, "ANALYSIS")) == 1


# ---------------------------------------------------------------------------
# 7. Truncation
# ---------------------------------------------------------------------------


#: Distinct MATERIAL subjects the §23 clustering will not fuse. Near-identical
#: headlines collapse into ONE story by design (that is what §23 is FOR), so a
#: fixture built by numbering the same sentence produces one item and proves
#: nothing about volume.
_DISTINCT_STORIES: tuple[str, ...] = (
    "Apple raises full-year guidance ahead of {n} launch cycle",
    "DOJ opens antitrust probe {n} into Apple App Store terms",
    "Apple to acquire sensor maker Lumira {n} in cash deal",
    "Apple names {n} chief financial officer in executive shake-up",
    "Apple recalls {n} battery units after regulator complaint",
    "Apple announces {n} billion dollar buyback and dividend raise",
    "Court rules against Apple in {n} patent infringement suit",
    "Apple cuts full year outlook citing {n} supply constraints",
)


async def test_a_dense_window_stays_inside_the_cap(client):
    """225 stored articles must not become 225 rail items.

    The end-to-end half of the truncation guarantee. It asserts the INVARIANT
    (never more than the cap, still ascending) rather than that the flag
    flipped, because how many of 225 real articles survive §23 clustering and
    the §26 material cut is the pure layer's judgement, not this seam's — and
    a test that pinned that number would be asserting the lexicon's behaviour
    from the wrong file. :func:`test_the_cap_drops_the_lowest_scoring_news`
    pins the cut itself, directly, where the arithmetic is visible.
    """
    _, upcoming = await _earnings_pair()
    for index in range(MAX_TIMELINE_ITEMS + 25):
        template = _DISTINCT_STORIES[index % len(_DISTINCT_STORIES)]
        await _seed_article(
            f"probe-{index}",
            template.format(n=f"round {index}"),
            publisher=f"Outlet{index % 7}",
            published_at=NOW - timedelta(days=100, minutes=index * 90),
        )

    payload = await _timeline(client, upcoming)

    assert len(payload["items"]) <= MAX_TIMELINE_ITEMS
    assert payload["max_items"] == MAX_TIMELINE_ITEMS
    stamps = [item["at"] for item in payload["items"]]
    assert stamps == sorted(stamps)


def test_the_cap_drops_the_lowest_scoring_news_and_keeps_everything_else():
    """The cut, exercised directly, where its arithmetic is readable.

    Two properties matter and neither is visible end-to-end without a corpus
    of hundreds of genuinely distinct stories: the non-NEWS kinds SURVIVE (a
    filing has no "less important" ordering to exploit, and dropping the
    oldest N would silently delete the beginning of the period), and the news
    that survives is the HIGH-scoring news rather than the most recent.
    """
    news = [
        {
            "kind": "NEWS",
            "at": f"2026-06-{(index % 28) + 1:02d}T12:00:00+00:00",
            "score": index / 1000.0,
            "evidence_id": f"news:n{index}",
        }
        for index in range(MAX_TIMELINE_ITEMS + 50)
    ]
    filing = {"kind": "FILING", "at": "2026-06-01T00:00:00+00:00", "statement_id": 1}
    analysis = {"kind": "ANALYSIS", "at": "2026-06-02T00:00:00+00:00", "id": 9}

    kept, truncated = _truncate(sorted([*news, filing, analysis], key=_sort_key))

    assert truncated is True
    assert len(kept) == MAX_TIMELINE_ITEMS
    assert filing in kept and analysis in kept
    kept_scores = [item["score"] for item in kept if item["kind"] == "NEWS"]
    # The bottom of the ranking went; the top stayed.
    assert min(kept_scores) > 0.0
    assert max(kept_scores) == pytest.approx((MAX_TIMELINE_ITEMS + 49) / 1000.0)
    assert [item["at"] for item in kept] == sorted(item["at"] for item in kept)


def test_a_list_inside_the_cap_is_returned_untouched():
    """No reordering, no flag, no copy-shuffling for the ordinary case."""
    items = [
        {"kind": "NEWS", "at": "2026-06-01T00:00:00+00:00", "score": 0.9},
        {"kind": "FILING", "at": "2026-06-02T00:00:00+00:00"},
    ]

    kept, truncated = _truncate(items)

    assert truncated is False
    assert kept == items


# ---------------------------------------------------------------------------
# 8. §54 card summaries on GET /api/events
# ---------------------------------------------------------------------------


async def _list(client, *, summaries: bool | None = None) -> dict:
    url = "/api/events?horizon=30d"
    if summaries is not None:
        url += f"&summaries={'true' if summaries else 'false'}"
    response = await client.get(url)
    assert response.status_code == 200, response.text
    return response.json()


def _summary(payload: dict, event_id: int) -> dict:
    return next(e for e in payload["events"] if e["event_id"] == event_id)["summary"]


async def _upcoming_soon(*, ticker: str = "AAPL") -> tuple[int, int]:
    """A previous print and one inside the default 30-day horizon.

    Anchored on ``now()`` rather than on :data:`NOW` because the list endpoint
    buckets on the real clock — an event fixed at a 2026 date would fall out
    of the horizon the moment the machine's date moved past it, which is the
    kind of test that rots quietly.
    """
    now = datetime.now(timezone.utc)
    previous = await _add_event(
        key=f"EARNINGS:{ticker}:prev",
        ticker=ticker,
        when=now - timedelta(days=90),
    )
    upcoming = await _add_event(
        key=f"EARNINGS:{ticker}:next",
        ticker=ticker,
        when=now + timedelta(days=5),
        previous_event_id=previous,
    )
    return previous, upcoming


async def test_the_list_payload_is_unchanged_when_summaries_are_not_asked_for(client):
    """An opt-in feature that changed the default response would be a regression.

    Compared key-for-key rather than just checking for the absence of
    ``summary``: the flag also adds queries, and a payload that grew a
    ``counts`` key or reordered rows would break a consumer just as surely.
    """
    _, upcoming = await _upcoming_soon()
    await _seed_metric(upcoming, implied_move_pct=0.07)

    default = await _list(client)
    explicit_off = await _list(client, summaries=False)

    for payload in (default, explicit_off):
        assert payload["events"], "the fixture event must be in the horizon"
        for event in payload["events"]:
            assert "summary" not in event
    assert [e["event_id"] for e in default["events"]] == [
        e["event_id"] for e in explicit_off["events"]
    ]
    assert set(default) == set(explicit_off)
    assert default["counts"] == explicit_off["counts"]


async def test_every_event_gets_a_summary_key_when_asked_for(client):
    """Present-with-nulls, never absent: the two mean different things."""
    _, upcoming = await _upcoming_soon()

    payload = await _list(client, summaries=True)

    summary = _summary(payload, upcoming)
    assert summary["analysis_status"] == "NONE"
    assert summary["analysis_as_of"] is None
    assert summary["implied_move_pct"] is None
    assert summary["implied_move_basis"] is None
    assert summary["historical_move_median_abs"] is None
    assert summary["historical_move_n"] is None
    assert summary["previous_event_actual_move_pct"] is None
    assert "not a forecast" in summary["implied_move_note"]


async def test_a_fresh_analysis_is_ready_and_an_old_one_is_stale(client):
    """Two different facts, and a trader acts on them differently."""
    _, fresh_event = await _upcoming_soon(ticker="AAPL")
    _, stale_event = await _upcoming_soon(ticker="MSFT")
    now = datetime.now(timezone.utc)
    await _seed_analysis(
        fresh_event, as_of=now - timedelta(days=1), created_at=now, digest="fresh"
    )
    await _seed_analysis(
        stale_event,
        as_of=now - timedelta(days=30),
        created_at=now - timedelta(days=30),
        digest="stale",
    )

    payload = await _list(client, summaries=True)

    assert _summary(payload, fresh_event)["analysis_status"] == "READY"
    assert _summary(payload, stale_event)["analysis_status"] == "STALE"
    assert _summary(payload, stale_event)["analysis_as_of"] is not None


async def test_an_invalid_analysis_does_not_make_a_card_say_ready(client):
    """An answer that quoted an invented number is not an answer (§99)."""
    _, upcoming = await _upcoming_soon()
    await _seed_analysis(
        upcoming,
        as_of=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        status="INVALID",
        digest="bad",
    )

    payload = await _list(client, summaries=True)

    assert _summary(payload, upcoming)["analysis_status"] == "NONE"


async def test_the_implied_move_travels_with_its_basis(client):
    """§37: a live chain and a reconstruction are different claims."""
    _, upcoming = await _upcoming_soon()
    await _seed_metric(
        upcoming, basis="daily_close_reconstruction", implied_move_pct=0.0642
    )

    payload = await _list(client, summaries=True)

    summary = _summary(payload, upcoming)
    assert summary["implied_move_pct"] == pytest.approx(0.0642)
    assert summary["implied_move_basis"] == "daily_close_reconstruction"
    assert summary["implied_move_as_of"] is not None


async def test_a_no_data_option_row_is_a_retraction_not_a_quote(client):
    """The seam stored the ABSENCE; resurrecting a stale number would undo it."""
    _, upcoming = await _upcoming_soon()
    await _seed_metric(
        upcoming, basis="live_chain", implied_move_pct=0.99, status="NO_DATA"
    )

    payload = await _list(client, summaries=True)

    summary = _summary(payload, upcoming)
    assert summary["implied_move_pct"] is None
    assert summary["implied_move_basis"] is None


async def test_the_historical_median_walks_the_previous_event_chain_with_its_n(client):
    """A median without its n is a single observation in a statistic's clothes."""
    now = datetime.now(timezone.utc)
    oldest = await _add_event(
        key="EARNINGS:AAPL:q1", ticker="AAPL", when=now - timedelta(days=270)
    )
    middle = await _add_event(
        key="EARNINGS:AAPL:q2",
        ticker="AAPL",
        when=now - timedelta(days=180),
        previous_event_id=oldest,
    )
    previous = await _add_event(
        key="EARNINGS:AAPL:q3",
        ticker="AAPL",
        when=now - timedelta(days=90),
        previous_event_id=middle,
    )
    upcoming = await _add_event(
        key="EARNINGS:AAPL:q4",
        ticker="AAPL",
        when=now + timedelta(days=5),
        previous_event_id=previous,
    )
    await _seed_metric(oldest, basis="hist", actual_move_pct=-0.02)
    await _seed_metric(middle, basis="hist", actual_move_pct=0.08)
    await _seed_metric(previous, basis="hist", actual_move_pct=0.04)

    payload = await _list(client, summaries=True)

    summary = _summary(payload, upcoming)
    # |−0.02|, |0.08|, |0.04| -> median 0.04. The sign is dropped on purpose:
    # "how big does this typically move" is a magnitude question.
    assert summary["historical_move_median_abs"] == pytest.approx(0.04)
    assert summary["historical_move_n"] == 3
    assert summary["previous_event_actual_move_pct"] == pytest.approx(0.04)


async def test_a_no_data_past_row_is_not_counted_in_the_median(client):
    now = datetime.now(timezone.utc)
    previous = await _add_event(
        key="EARNINGS:AAPL:p", ticker="AAPL", when=now - timedelta(days=90)
    )
    upcoming = await _add_event(
        key="EARNINGS:AAPL:n",
        ticker="AAPL",
        when=now + timedelta(days=5),
        previous_event_id=previous,
    )
    await _seed_metric(
        previous, basis="hist", actual_move_pct=0.5, status="NO_DATA"
    )

    payload = await _list(client, summaries=True)

    summary = _summary(payload, upcoming)
    assert summary["historical_move_median_abs"] is None
    assert summary["historical_move_n"] is None
    assert summary["previous_event_actual_move_pct"] is None


async def test_the_previous_event_chain_walk_survives_a_cycle(client):
    """A self-referential chain must terminate, not spin the request forever."""
    now = datetime.now(timezone.utc)
    upcoming = await _add_event(
        key="EARNINGS:LOOP:n", ticker="LOOP", when=now + timedelta(days=5)
    )
    async with SessionLocal() as s:
        row = await s.get(EventRow, upcoming)
        row.previous_event_id = upcoming
        await s.commit()

    payload = await _list(client, summaries=True)

    assert _summary(payload, upcoming)["historical_move_n"] is None


async def test_summaries_do_not_change_the_events_other_fields(client):
    """``summary`` is ADDITIVE — nothing existing is rewritten to make room."""
    _, upcoming = await _upcoming_soon()

    off = await _list(client)
    on = await _list(client, summaries=True)

    plain = next(e for e in off["events"] if e["event_id"] == upcoming)
    summarised = next(e for e in on["events"] if e["event_id"] == upcoming)
    for key, value in plain.items():
        if key == "days_to_event":
            continue  # recomputed from a live clock on each request
        assert summarised[key] == value, key
    assert set(summarised) - set(plain) == {"summary"}


async def test_the_identity_keys_are_both_flat_and_nested_and_agree(client):
    """The tab header reads them flat; every other event payload nests them.

    Duplicated on purpose (see the seam), so the property worth pinning is
    that the two copies can never disagree — which is what a reader who saw
    both would want to know.
    """
    _, upcoming = await _earnings_pair()

    payload = await _timeline(client, upcoming)

    assert payload["event_id"] == payload["event"]["event_id"] == upcoming
    assert payload["event_key"] == payload["event"]["event_key"]
    assert payload["ticker"] == payload["event"]["ticker"] == "AAPL"


async def test_two_cards_sharing_an_ancestor_each_keep_their_full_history(client):
    """A shared ancestor must not cost one of the two cards its chain.

    THE BUG THIS PINS. A 30-day horizon that straddles a quarter boundary
    carries two prints of the SAME ticker, and both chains pass through the
    same older rows. A previous-event walk whose frontier maps each row to ONE
    owner lets the second card overwrite the first, and one of them comes back
    with no history at all — a card reading "—" beside an identical card
    reading "median 4% (n=3)", with nothing in the payload to explain why.
    Both cards are asserted, because the bug always leaves one of them right.
    """
    now = datetime.now(timezone.utc)
    oldest = await _add_event(
        key="EARNINGS:AAPL:q1", ticker="AAPL", when=now - timedelta(days=200)
    )
    shared = await _add_event(
        key="EARNINGS:AAPL:q2",
        ticker="AAPL",
        when=now - timedelta(days=110),
        previous_event_id=oldest,
    )
    # Both of these are inside the 30-day horizon and both chain through
    # ``shared`` — the collision the walk has to survive.
    first = await _add_event(
        key="EARNINGS:AAPL:q3",
        ticker="AAPL",
        when=now + timedelta(days=3),
        previous_event_id=shared,
    )
    second = await _add_event(
        key="EARNINGS:AAPL:q4",
        ticker="AAPL",
        when=now + timedelta(days=10),
        previous_event_id=shared,
    )
    await _seed_metric(oldest, basis="hist", actual_move_pct=-0.06)
    await _seed_metric(shared, basis="hist", actual_move_pct=0.02)

    payload = await _list(client, summaries=True)

    for event_id in (first, second):
        summary = _summary(payload, event_id)
        assert summary["historical_move_n"] == 2, event_id
        assert summary["historical_move_median_abs"] == pytest.approx(0.04), event_id
        assert summary["previous_event_actual_move_pct"] == pytest.approx(0.02)
