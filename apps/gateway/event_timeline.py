"""Event timeline + calendar-card summaries — the gateway seam (Phase J, U1;
event spec §54, §56, §57; audit §7.1, §7.2, §11.5).

TWO READS, ONE RULE: NEITHER OF THEM FETCHES. :func:`build_event_timeline`
answers "what has happened to this issuer since its last print" and
:func:`attach_card_summaries` answers "what does the platform already know
about each row in the calendar feed". Both are STORED-ROWS-ONLY, and that is
not an optimisation — it is the audit §7.2 rule 1 split every read surface in
this router obeys. Opening the Catalyst page or scrolling the calendar must
not issue a vendor call; the POST backfills (``/news/backfill``,
``/options/backfill``, ``/replay/backfill``) are the USER actions that fetch.
This module holds no provider handle at all, which is what makes that
checkable rather than merely intended.

THE TIMELINE CLASSIFIES NOTHING ITSELF (audit §7.4). Which headlines are one
story, what category they fall in and what they score is
``libs/trading_core/events/news_intel.analyze_window``'s judgement, reached
through the SAME loader ``event_news.build_event_news`` uses
(``_articles_for_ticker`` + ``to_raw_articles``). Re-implementing either half
here would put a second, drift-free-by-luck copy of the §22-§26 pipeline in
the codebase, and the timeline's category chips would then be allowed to
disagree with the news tab's badges for the very same article — the one
inconsistency a reader would never think to check.

THE WINDOW IS THE INTER-EVENT PERIOD, exactly as §21's news window is: from
the previous comparable event's instant to ``as_of``. That is what makes
"nine developments since the last print" a statement about the period rather
than about the calendar. With no previous event — a newly covered symbol, a
registry without history — it falls back to ``as_of`` minus
:data:`~apps.gateway.event_news.DEFAULT_WINDOW_DAYS`, and ``window.basis``
says which of the two happened so no reader has to infer it.

``as_of`` IS A HARD BOUND ON EVERY ITEM (§96). News is gated by the pure
layer on ``published_at``; filings on ``acceptance_datetime`` (the instant a
filing became PUBLIC, never ``end_date``, the instant its period closed);
registry events and analyses on their own instants. The bound is re-applied
to the assembled list as a final pass, so a kind added later cannot leak a
future row past it by forgetting its own filter. A timeline that showed an
item the caller could not have known is a look-ahead leak with a date on it.

WHAT THE CARD SUMMARIES WILL AND WILL NOT SAY (§44 rule 18, §54). Every field
is either a stored number or ``None``; nothing is computed to fill a hole.
``analysis_status`` is a three-value vocabulary rather than a boolean because
"analysed last month" and "never analysed" are different facts a trader acts
on differently. ``historical_move_median_abs`` always travels with its
``historical_move_n`` — a median of one is a number pretending to be a
distribution — and ``implied_move_pct`` always travels with its
``implied_move_basis``, because §37 is explicit that a live-chain reading and
a reconstruction from daily closes are different claims. A NO_DATA option row
contributes NOTHING: it is the seam's retraction of a number, and quoting a
stale ``implied_move_pct`` off it would turn that retraction into a claim.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.events import to_local
from libs.trading_core.events.news_intel import analyze_window
from libs.trading_core.models.enums import EventStatus

from .db import (
    EventAnalysisRow,
    EventOptionMetricRow,
    EventRow,
    FundamentalStatementRow,
)
from .event_news import (
    DEFAULT_WINDOW_DAYS,
    _articles_for_ticker,
    _previous_anchor_row,
    to_raw_articles,
)
from .event_price import _as_utc

#: The most items one timeline renders. Past this the payload is truncated —
#: NEWS first and by score, since that is the only kind with a ranking the
#: library computed — and ``truncated`` says so. A cap rather than a page
#: because the timeline is a shape to read at a glance, not a feed to scroll:
#: a mega-cap's four-month window can carry a thousand articles, and shipping
#: them would make the browser, not the trader, the thing that decides what is
#: visible.
MAX_TIMELINE_ITEMS = 200

#: How far back the previous-event walk goes when pooling past option metrics
#: for the §54 "historical event move" line. Eight prints is two years of
#: quarterly earnings — enough for a median to mean something, bounded so a
#: pathological ``previous_event_id`` chain cannot make a calendar render walk
#: the whole table.
MAX_PREVIOUS_WALK = 8

#: An analysis is READY only while it is this fresh. Older OK rows are STALE,
#: not absent: the platform HAS an answer for this event and the trader should
#: know it exists AND that it predates a week of news.
ANALYSIS_FRESH_DAYS = 7

#: Analyses that count as "the platform has an answer". INVALID and FAILED
#: rows deliberately do not — an analysis whose numbers failed the fact-index
#: check is evidence about the model, not an answer about the event (§99).
ANALYSIS_OK_STATUS = "OK"

#: Option metric rows in this status carry no numbers worth quoting. The seam
#: writes one when a vendor would not serve the straddle, precisely so the
#: absence is stored rather than retried forever; reading a leftover
#: ``implied_move_pct`` off it would resurrect a number the seam retracted.
OPTION_NO_DATA_STATUS = "NO_DATA"

#: The kinds a timeline item can be. Frozen here so the router's docstring,
#: the counts block and the UI's filter chips all name the same four things.
TIMELINE_KINDS = ("NEWS", "FILING", "EVENT", "ANALYSIS")


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


# ---------------------------------------------------------------------------
# The timeline (§57)
# ---------------------------------------------------------------------------


def _anchor_out(row: EventRow | None) -> dict | None:
    """One end of the timeline as the UI renders it, or ``None``.

    Deliberately NOT the full ``event_out`` shape. The anchors are two labels
    on a rail — "LAST EARNINGS 2026-05-01 AMC" and the same for the next one —
    and shipping the whole registry row twice (importance components, revision
    history, exposure) to draw them would be most of the payload. ``date_et``
    is pre-formatted because every consumer of it renders a day, and
    ``is_estimated`` travels so an anchor on a DERIVED date can never be drawn
    as a reported one (§7, §11).
    """
    if row is None:
        return None
    when = _as_utc(row.scheduled_at)
    # The DAY is taken in the event's OWN zone, not in UTC, and that is the
    # whole reason this field is spelled ``date_et``. An AMC print at 20:15 ET
    # is 00:15 UTC the NEXT calendar day, so a UTC ``.date()`` would label the
    # last earnings anchor with tomorrow — off by one for exactly the events
    # this rail exists to bracket.
    local = to_local(when, row.event_timezone)
    return {
        "event_id": row.id,
        "event_key": row.event_key,
        "event_type": row.event_type,
        "title": row.title,
        "date_et": local.date().isoformat(),
        "scheduled_at_local": local.isoformat(),
        "event_timezone": row.event_timezone,
        "scheduled_at_utc": when.isoformat(),
        "session": row.session,
        "status": row.status,
        "is_estimated": row.status == EventStatus.ESTIMATED.value,
    }


async def _news_items(
    session: AsyncSession,
    ticker: str,
    *,
    start: datetime,
    as_of: datetime,
) -> list[dict]:
    """MATERIAL developments in the window, one item per §23 cluster.

    The clustering, the categories and the scores are ``analyze_window``'s —
    this function only chooses which of its output belongs on a timeline and
    flattens each cluster into a row. MATERIAL only, and that is the §26 cut
    rather than a display preference: a timeline is the answer to "what
    happened", and eleven hundred syndicated price-move blurbs are not eleven
    hundred things that happened.

    ``article_count`` travels with each item so a story carried by six
    publishers reads as ONE development that six outlets covered, which is
    what §23 means by refusing to let duplication inflate importance.
    """
    rows = await _articles_for_ticker(session, ticker, start=start, end=as_of)
    if not rows:
        return []
    articles = to_raw_articles(rows)
    result = analyze_window(
        articles,
        ticker=ticker,
        as_of=as_of,
        window_start=start,
    )
    items: list[dict] = []
    for cluster in result.clusters:
        if not cluster.material:
            continue
        canonical = cluster.canonical
        published = canonical.published_at
        if published is None:
            # Cannot be placed on a rail without an instant, and inventing one
            # would put a real development at a fictional moment. The news tab
            # still shows it; the timeline honestly cannot.
            continue
        items.append(
            {
                "kind": "NEWS",
                "at": _iso(published),
                "category": cluster.materiality,
                # The DISPLAY title is the publisher's own words (the §81
                # sanitised copy is for the LLM path and is not what a human
                # reads); the UI escapes it. ``suspicious_instruction`` travels
                # so a headline shaped like a prompt injection is visible as
                # such rather than silently laundered away.
                "title": canonical.title,
                "publisher": canonical.publisher,
                "url": canonical.url,
                "evidence_id": "news:" + canonical.source_id,
                "cluster_id": cluster.cluster_id,
                "score": None if cluster.score is None else cluster.score.score,
                "article_count": cluster.article_count,
                "suspicious_instruction": bool(
                    canonical.to_ref().get("suspicious_instruction")
                ),
            }
        )
    return items


async def _filing_items(
    session: AsyncSession,
    ticker: str,
    *,
    start: datetime,
    as_of: datetime,
) -> list[dict]:
    """Financial statements that became PUBLIC in the window.

    Gated on ``acceptance_datetime``, never on ``end_date`` — the whole point
    of that column (§96, audit §7.1). A quarter ending 2026-06-30 is filed in
    late July, and a timeline keyed on the period end would place the filing
    three weeks before anybody could read it. Rows with a NULL acceptance
    instant are EXCLUDED rather than dated from ``filing_date``: the row is
    still real and still stored, but its publication instant is unknown and a
    guessed one is exactly the look-ahead this gate exists to stop.
    """
    rows = (
        (
            await session.execute(
                select(FundamentalStatementRow)
                .where(
                    FundamentalStatementRow.ticker == ticker,
                    FundamentalStatementRow.acceptance_datetime.is_not(None),
                    FundamentalStatementRow.acceptance_datetime >= start,
                    FundamentalStatementRow.acceptance_datetime <= as_of,
                )
                .order_by(FundamentalStatementRow.acceptance_datetime.asc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "kind": "FILING",
            "at": _iso(row.acceptance_datetime),
            "fiscal_period": row.fiscal_period,
            "fiscal_year": row.fiscal_year,
            "timeframe": row.timeframe,
            "period_end": row.end_date.isoformat() if row.end_date else None,
            "source_url": row.source_filing_url,
            "statement_id": row.id,
        }
        for row in rows
    ]


async def _event_items(
    session: AsyncSession,
    ticker: str,
    *,
    start: datetime,
    as_of: datetime,
    exclude_ids: set[int],
) -> list[dict]:
    """OTHER registry events for the same issuer inside the window.

    The two anchors are excluded because they are already drawn as the rail's
    ends — repeating them as items would show the last print twice and imply
    two prints happened. Everything else the registry knows about this ticker
    belongs here: a dividend, a shareholder meeting, a guidance update all
    frame the period between two earnings and are exactly what "since the last
    event" is asking about.
    """
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.ticker == ticker,
                    EventRow.scheduled_at >= start,
                    EventRow.scheduled_at <= as_of,
                )
                .order_by(EventRow.scheduled_at.asc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "kind": "EVENT",
            "at": _iso(row.scheduled_at),
            "event_id": row.id,
            "event_key": row.event_key,
            "event_type": row.event_type,
            "title": row.title,
            "status": row.status,
            "is_estimated": row.status == EventStatus.ESTIMATED.value,
        }
        for row in rows
        if row.id not in exclude_ids
    ]


async def _analysis_items(
    session: AsyncSession,
    event_id: int,
    *,
    start: datetime,
    as_of: datetime,
) -> list[dict]:
    """This event's own OK analyses — the platform's memory of its own work.

    On the rail because the §57 question is "what has happened since the last
    print", and "we formed a view on 2026-08-04 and called the regime
    HIGH_BAR" is one of the things that happened. Only OK rows: a FAILED
    attempt has no view to place and an INVALID one quoted a number that was
    not in its evidence, so putting either on a timeline of findings would
    read as a finding (§99 keeps them listed on the /analyses history
    instead, which is where the record of what the model tried belongs).

    Placed at ``created_at``, not ``as_of``: this is the instant the platform
    formed the view, and a historical re-run written today genuinely happened
    today even though it answers about October.
    """
    rows = (
        (
            await session.execute(
                select(EventAnalysisRow)
                .where(
                    EventAnalysisRow.event_id == event_id,
                    EventAnalysisRow.status == ANALYSIS_OK_STATUS,
                    EventAnalysisRow.created_at >= start,
                    EventAnalysisRow.created_at <= as_of,
                )
                .order_by(EventAnalysisRow.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    items: list[dict] = []
    for row in rows:
        analysis = row.analysis or {}
        items.append(
            {
                "kind": "ANALYSIS",
                "at": _iso(row.created_at),
                "id": row.id,
                "regime": analysis.get("expectations_gap_regime"),
                "confidence": analysis.get("confidence"),
                "as_of_analysis": _iso(row.as_of),
                "model": row.model,
            }
        )
    return items


def _sort_key(item: dict) -> tuple:
    """Ascending by instant, then by kind, then by a stable tiebreak.

    The secondary keys exist so two items stamped the SAME second (a filing
    and the article about it, routinely) cannot swap places between two reads
    of identical data. An unstable order would make the UI's collapsed groups
    reshuffle on every refresh for no reason a reader could explain.
    """
    return (
        item.get("at") or "",
        item.get("kind") or "",
        str(item.get("evidence_id") or item.get("event_id") or item.get("id") or ""),
    )


def _truncate(items: list[dict]) -> tuple[list[dict], bool]:
    """Cap the list at :data:`MAX_TIMELINE_ITEMS`, dropping the LOWEST news.

    Only NEWS is droppable, and only by score, for a simple reason: it is the
    only kind the library ranked. A filing, a registry event and an analysis
    are each singular facts with no "less important" ordering to exploit, and
    there are never many of them — the count that explodes on a mega-cap is
    articles. Dropping the bottom of a scored ranking is a defensible cut;
    dropping "the oldest 400 things" would silently delete the beginning of
    the period, which is where the story usually starts.
    """
    if len(items) <= MAX_TIMELINE_ITEMS:
        return items, False
    news = [item for item in items if item["kind"] == "NEWS"]
    other = [item for item in items if item["kind"] != "NEWS"]
    room = max(MAX_TIMELINE_ITEMS - len(other), 0)
    news.sort(key=lambda item: (-(item.get("score") or 0.0), item.get("at") or ""))
    kept = other + news[:room]
    kept.sort(key=_sort_key)
    return kept[:MAX_TIMELINE_ITEMS], True


async def build_event_timeline(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> dict:
    """Everything that happened between the last comparable event and ``as_of``.

    DB-ONLY, NEVER FETCHES (§27; audit §7.2 rule 1). Four stored sources are
    merged onto one rail — material news developments, filings that became
    public, other registry events for the issuer, and this event's own stored
    analyses — sorted ascending and counted by kind and by news category.

    ``as_of`` is REQUIRED, is the window's right edge and is a hard bound on
    every item (audit §7.2 rule 2, §96). It is re-applied to the merged list
    after each source has already applied it, so a kind added to this function
    later cannot leak a future row by forgetting its own filter.

    A macro or Fed event has no issuer, and therefore no news, no filings and
    no sibling events: it answers with ``available: false``, ``reason:
    "no_ticker"``, its own analyses still on the rail, and empty everything
    else. Substituting an index proxy would attribute a different issuer's
    tape to it (§39 multi-asset proxies are Phase G).
    """
    moment = _as_utc(as_of)
    ticker = (event_row.ticker or "").strip().upper()

    previous = await _previous_anchor_row(session, event_row, moment)
    if previous is not None:
        start = _as_utc(previous.scheduled_at)
        basis = f"previous_{(event_row.event_type or 'event').lower()}:{previous.event_key}"
    else:
        start = moment - timedelta(days=DEFAULT_WINDOW_DAYS)
        basis = f"default_{DEFAULT_WINDOW_DAYS}d"
    if start > moment:
        # Only reachable by asking about an instant before the previous event.
        # Clamp rather than invert: an empty window is the honest shape for
        # "nothing had happened yet", and a reversed one is not a window.
        start = moment
        basis = f"{basis}:clamped_to_as_of"

    items: list[dict] = []
    if ticker:
        items.extend(await _news_items(session, ticker, start=start, as_of=moment))
        items.extend(await _filing_items(session, ticker, start=start, as_of=moment))
        items.extend(
            await _event_items(
                session,
                ticker,
                start=start,
                as_of=moment,
                exclude_ids={
                    event_row.id,
                    *({previous.id} if previous is not None else set()),
                },
            )
        )
    items.extend(
        await _analysis_items(session, event_row.id, start=start, as_of=moment)
    )

    # The final as-of pass. Every source above already gated itself; this
    # catches the one that stops doing so. An item with no instant at all is
    # dropped for the same reason — it cannot be placed, and placing it
    # anywhere would be a date the platform made up.
    end_iso = moment.isoformat()
    items = [
        item for item in items if item.get("at") is not None and item["at"] <= end_iso
    ]
    items.sort(key=_sort_key)
    items, truncated = _truncate(items)

    by_kind = {kind: 0 for kind in TIMELINE_KINDS}
    by_category: dict[str, int] = {}
    for item in items:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        if item["kind"] == "NEWS":
            category = item.get("category") or "OTHER"
            by_category[category] = by_category.get(category, 0) + 1

    return {
        # The identity is BOTH nested and flat, and the duplication is
        # deliberate rather than an oversight: ``event`` is the block every
        # other event payload in this router carries (``/news``, ``/replay``,
        # ``/options``), so a consumer that already knows that shape reads
        # this one for free — while the three flat keys are what the timeline
        # tab itself renders in its header, and making it dig one level for
        # the ticker it puts in a title would be a shape nobody would guess.
        # They are one expression each, so they cannot disagree.
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": event_row.ticker,
        "event": {
            "event_id": event_row.id,
            "event_key": event_row.event_key,
            "event_type": event_row.event_type,
            "title": event_row.title,
            "ticker": event_row.ticker,
            "scheduled_at_utc": _iso(event_row.scheduled_at),
        },
        "as_of": end_iso,
        "available": bool(ticker),
        "reason": None if ticker else "no_ticker",
        "window": {
            "start": start.isoformat(),
            "end": end_iso,
            "basis": basis,
            "days": (moment - start) / timedelta(days=1),
        },
        "anchors": {
            "previous_event": _anchor_out(previous),
            "as_of": end_iso,
            "next_event": _anchor_out(event_row),
        },
        "items": items,
        "counts": {
            "total": len(items),
            "by_kind": by_kind,
            "by_category": by_category,
        },
        "truncated": truncated,
        "max_items": MAX_TIMELINE_ITEMS,
        "provenance": {
            "news": "QUANT",
            "filings": "DATA",
            "events": "DATA",
            "analyses": "LLM",
        },
    }


# ---------------------------------------------------------------------------
# Calendar card summaries (§54)
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    """The middle value, or the mean of the middle two. ``None`` when empty.

    Spelled out rather than imported from ``statistics`` only so the empty
    case returns ``None`` instead of raising: an empty pool is the ORDINARY
    state for a newly covered ticker, and a card must render it as "—", not
    as a 500.
    """
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


async def _previous_event_chain(
    session: AsyncSession, event_ids: list[int], *, depth: int
) -> dict[int, list[int]]:
    """For each event, its ``previous_event_id`` chain, nearest ancestor first.

    Walked BREADTH-FIRST across every card at once — one query per LEVEL, not
    one per card — because the calendar feed can carry two hundred rows and a
    per-row walk would be sixteen hundred round trips to draw one page.

    THE FRONTIER MAPS ONE ROW TO MANY OWNERS, and that is the whole subtlety.
    Two cards on the same page routinely share an ancestor: a ticker's Q3 and
    Q4 prints are both in a 30-day horizon whenever the horizon straddles a
    quarter, and both chains pass through the same Q2 row. A frontier keyed
    ``row_id -> owner`` would let the second card overwrite the first, and one
    of the two would silently lose its entire history — a card reading "no
    past moves" beside an identical card reading "median 4% (n=3)", with
    nothing in the payload to explain the difference. ``set`` of owners is
    what keeps both walks alive through the shared row.

    ``seen`` is per-owner and the depth cap is absolute, so a cyclic or
    self-referential ``previous_event_id`` terminates instead of spinning.
    """
    if not event_ids:
        return {}

    chains: dict[int, list[int]] = {event_id: [] for event_id in event_ids}
    seen: dict[int, set[int]] = {event_id: {event_id} for event_id in event_ids}

    rows = (
        await session.execute(
            select(EventRow.id, EventRow.previous_event_id).where(
                EventRow.id.in_(event_ids)
            )
        )
    ).all()

    # row id -> the cards still walking THROUGH it.
    frontier: dict[int, set[int]] = {}
    for row_id, previous_id in rows:
        if previous_id is not None:
            frontier.setdefault(previous_id, set()).add(row_id)

    for _ in range(depth):
        if not frontier:
            break
        found = (
            await session.execute(
                select(EventRow.id, EventRow.previous_event_id).where(
                    EventRow.id.in_(list(frontier))
                )
            )
        ).all()
        next_frontier: dict[int, set[int]] = {}
        for row_id, previous_id in found:
            for owner in frontier.get(row_id, ()):
                if row_id in seen[owner]:
                    continue
                seen[owner].add(row_id)
                chains[owner].append(row_id)
                if previous_id is not None and previous_id not in seen[owner]:
                    next_frontier.setdefault(previous_id, set()).add(owner)
        frontier = next_frontier

    return chains


async def attach_card_summaries(session: AsyncSession, events: list[dict]) -> None:
    """Add a ``summary`` block to each event dict, in place (§54).

    THE CALENDAR CARD'S FOUR EXTRA LINES, and every one of them is a stored
    fact or ``None``. What a card may say is: whether an analysis exists and
    how fresh it is, what the option market charged for this event and on what
    BASIS, how big this issuer's past moves on this event type actually were
    (a median with its n), and what the previous print did. Nothing here is
    derived from a live price, because this seam holds no provider handle.

    BULK, NOT N+1. Four queries total regardless of how many events are on the
    page, plus at most :data:`MAX_PREVIOUS_WALK` more for the previous-event
    walk. A per-card query would be four hundred round trips for a thirty-day
    calendar, which is the difference between a page that opens and one that
    times out.

    ``summary`` is added to EVERY event, never omitted for the empty case: a
    card that gets no key cannot distinguish "the server does not compute this"
    from "there is nothing", and the UI renders those two differently (§44
    rule 18). The FIELDS inside are null when unknown.
    """
    if not events:
        return

    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    if not event_ids:
        for event in events:
            event["summary"] = _empty_summary()
        return

    now = datetime.now(timezone.utc)

    # --- 1. analyses: the newest OK row per event ---------------------------
    analysis_rows = (
        (
            await session.execute(
                select(EventAnalysisRow)
                .where(
                    EventAnalysisRow.event_id.in_(event_ids),
                    EventAnalysisRow.status == ANALYSIS_OK_STATUS,
                )
                .order_by(EventAnalysisRow.as_of.desc(), EventAnalysisRow.id.desc())
            )
        )
        .scalars()
        .all()
    )
    newest_analysis: dict[int, EventAnalysisRow] = {}
    for row in analysis_rows:
        newest_analysis.setdefault(row.event_id, row)

    # --- 2. option metrics for the events themselves ------------------------
    metric_rows = (
        (
            await session.execute(
                select(EventOptionMetricRow)
                .where(EventOptionMetricRow.event_id.in_(event_ids))
                .order_by(
                    EventOptionMetricRow.as_of.desc(), EventOptionMetricRow.id.desc()
                )
            )
        )
        .scalars()
        .all()
    )
    implied: dict[int, EventOptionMetricRow] = {}
    for row in metric_rows:
        if row.status == OPTION_NO_DATA_STATUS or row.implied_move_pct is None:
            # A retraction is not a number. See OPTION_NO_DATA_STATUS.
            continue
        implied.setdefault(row.event_id, row)

    # --- 3. the previous-event chains, and their realised moves -------------
    chains = await _previous_event_chain(
        session, event_ids, depth=MAX_PREVIOUS_WALK
    )
    chain_ids = sorted({row_id for chain in chains.values() for row_id in chain})
    past_metrics: dict[int, list[EventOptionMetricRow]] = {}
    if chain_ids:
        past_rows = (
            (
                await session.execute(
                    select(EventOptionMetricRow).where(
                        EventOptionMetricRow.event_id.in_(chain_ids),
                        EventOptionMetricRow.actual_move_pct.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in past_rows:
            if row.status == OPTION_NO_DATA_STATUS:
                continue
            past_metrics.setdefault(row.event_id, []).append(row)

    for event in events:
        event_id = event.get("event_id")
        summary = _empty_summary()

        analysis = newest_analysis.get(event_id)
        if analysis is not None:
            when = _as_utc(analysis.as_of)
            fresh = (now - when) <= timedelta(days=ANALYSIS_FRESH_DAYS)
            summary["analysis_status"] = "READY" if fresh else "STALE"
            summary["analysis_as_of"] = when.isoformat()
            summary["analysis_id"] = analysis.id

        metric = implied.get(event_id)
        if metric is not None:
            summary["implied_move_pct"] = metric.implied_move_pct
            summary["implied_move_basis"] = metric.basis
            summary["implied_move_as_of"] = _iso(metric.as_of)

        chain = chains.get(event_id) or []
        moves: list[float] = []
        for row_id in chain:
            for row in past_metrics.get(row_id, ()):
                if row.actual_move_pct is not None:
                    moves.append(abs(row.actual_move_pct))
        if moves:
            summary["historical_move_median_abs"] = _median(moves)
            # n travels with the median, always. A median of one is a single
            # observation wearing a statistic's clothes, and a card that shows
            # the number without its n invites exactly that reading.
            summary["historical_move_n"] = len(moves)

        # The immediately previous print, which is the one a reader compares
        # against — separate from the median, because "last time it moved 9%"
        # and "it typically moves 4%" are different sentences.
        if chain:
            previous_rows = past_metrics.get(chain[0], ())
            for row in previous_rows:
                if row.actual_move_pct is not None:
                    summary["previous_event_actual_move_pct"] = row.actual_move_pct
                    break

        event["summary"] = summary


def _empty_summary() -> dict:
    """The all-absent summary — every key present, every value honest.

    The keys are ALWAYS present and the values are null when unknown, which is
    the §44 rule 18 shape: a UI can then render "—" for a metric the platform
    looked for and did not find, and render nothing at all when the whole
    ``summary`` key is missing because summaries were not requested. Those are
    genuinely different states and the payload keeps them different.
    """
    return {
        "analysis_status": "NONE",
        "analysis_as_of": None,
        "analysis_id": None,
        "implied_move_pct": None,
        "implied_move_basis": None,
        "implied_move_as_of": None,
        "implied_move_note": (
            "implied move is option-market pricing, not a forecast"
        ),
        "historical_move_median_abs": None,
        "historical_move_n": None,
        "previous_event_actual_move_pct": None,
    }
