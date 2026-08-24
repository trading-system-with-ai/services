"""News evidence — the gateway seam (Phase D, U3; event spec §21-§27, §59,
§81, §91, §96; audit §5.1, §7, §9.3, §11.5).

THE SPLIT THIS MODULE EXISTS TO KEEP, exactly as ``fundamentals.py`` keeps it
for filings and ``event_price.py`` for prices. Every judgement in the payload —
what is a duplicate, what is one story, what category a headline falls in, how
novel it is, what it scores — is made by
``libs/trading_core/events/news_intel.py``, which is pure stdlib and may not
import ``apps/`` or ``libs.market_data`` (audit §7.4). This module is the only
place the two halves meet: it fetches articles through the market-data
providers, MIRRORS them into ``news_articles``, reads them back, hands the ORM
rows to the library and renders the frozen result as JSON. It classifies
nothing itself.

INGESTION AND ANALYSIS ARE SEPARATE CALL GRAPHS (audit §7.2 rule 1).
:func:`ensure_event_news_window` holds the provider handles and writes rows; it
takes ``now``, never an ``as_of``. :func:`build_event_news` takes ``as_of``,
reads STORED rows only and never touches a provider at all — not even to top
the mirror up, which is the ONE place this seam is stricter than
``fundamentals.py``. The reason is §27 and the audit's read-path rule: opening
the Catalyst page must not issue vendor calls nobody asked for, and a news
window is dozens of paginated requests across two vendors rather than one.
``POST /api/events/{id}/news/backfill`` is the USER action that fetches; the
GET reads what is there and says so when there is nothing. That is the same
division ``/replay`` and ``/replay/backfill`` already draw for minute bars.

BOTH VENDORS, NOT ONE (§21; audit §5.1). Alpaca and Massive syndicate
different wires — Benzinga through Alpaca, its own publisher set through
Massive — so a window fetched from one alone is a partial view of the tape and
the §26 counts computed over it are wrong in a way nothing in the payload could
reveal. Every configured provider that implements ``get_news_window`` is asked
(``news_provider_names``), their results are merged, and the merge is safe
because ``source_id`` is a UNIQUE column: a story both vendors carry lands
once. A provider that 403s or is unconfigured becomes a NAMED SKIP in
``providers[]``, never an exception and never silence.

THE AS-OF GATE IS ON ``published_at``, AND IT IS THE PURE LAYER'S (§96; audit
§7.1). This module does not implement it — ``analyze_window`` drops
``published_at > as_of`` before anything else runs — but it must not defeat it,
so the SQL here caps the read at ``as_of`` as an OPTIMISATION and the library
still re-applies the gate to every row it is handed. Two copies of a filter is
usually a smell; here the outer one is a bounded read and the inner one is the
contract, and a test patches the loader to return EVERY stored row so the
inner gate is proved sufficient on its own — otherwise deleting the SQL bound
as a redundant clause would silently reopen the leak with every payload
assertion still green.

WHAT IS PERSISTED BACK, AND WHAT DELIBERATELY IS NOT (audit §7.1, §96;
migration 023). ``cluster_id``, ``materiality``, ``materiality_score``,
``source_quality`` and the per-ticker ``relevance`` entry are written onto the
article rows. All five are AS-OF-INDEPENDENT: they are functions of the
article's own text, its own publisher and its own tags, so they mean the same
thing at every instant. ``novelty``, ``decay`` and the composite evidence
``score`` are NOT persisted and have no columns: each depends on the ``as_of``
and on which OTHER articles shared the window, so storing one would freeze a
single request's viewpoint onto the row and let the next read at a different
``as_of`` inherit it — a look-ahead leak wearing a cache's clothes.

PROVENANCE IS LABELLED AT BLOCK LEVEL (§91): the articles are DATA (the
publisher's own words, cited back to ``url``), every score and category is
QUANT (this platform's lexicon and arithmetic). Nothing here is LLM-generated,
and §81 governs the other direction — article text is UNTRUSTED input, so the
strings meant for a model come pre-sanitised under ``safe_title`` /
``safe_description`` with ``suspicious_instruction`` flagged and never obeyed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import (
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.events.news_intel import (
    MATERIAL_SCORE_THRESHOLD,
    NEWS_MODEL_VERSION,
    RawArticle,
    analyze_window,
    materiality_of,
    source_quality,
    ticker_relevance,
)
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import EventStatus

from . import audit
from .db import EventRow, NewsArticleRow
from .event_price import _as_utc

logger = logging.getLogger(__name__)

#: The audit ``entity_type`` for a news backfill — the table it writes, the
#: same convention ``fundamentals.ENTITY_TYPE`` and ``ensure_daily_bars``
#: follow. It matches the entity_type the Phase 8 recommendations ingest
#: already uses, deliberately: both paths write the same mirror.
ENTITY_TYPE = "news_articles"

#: How far back the window opens when there is no previous comparable event to
#: anchor it (§21). A hundred and twenty days is roughly one earnings cycle
#: plus a month of lead-in: long enough that a quarter's worth of developments
#: is in view, short enough that the §22 decay (14-day half-life, floor 0.2)
#: has already flattened everything older into the floor. A longer default
#: would fetch thousands of articles to rank a tail that cannot move.
DEFAULT_WINDOW_DAYS = 120

#: The window opens one day BEFORE the previous comparable event rather than
#: at it. A release moves the tape for hours afterwards, and the coverage that
#: frames it — the previews published the evening before — belongs to the
#: story of the period that follows, not to the one that closed.
WINDOW_LEAD_DAYS = 1

#: Articles requested per provider per window. Five hundred is the provider
#: protocol's own default and is a ceiling, not a target: a heavily-covered
#: mega-cap over a full quarter can exceed it, and the payload says so
#: (``truncated`` in the fetch report) rather than pretending the window was
#: exhausted.
FETCH_LIMIT = 500

#: Minimum spacing between provider ATTEMPTS per ticker, in seconds. Six hours,
#: matching ``fundamentals.REFRESH_ATTEMPT_SECONDS``. News differs from filings
#: in that a window is never "already complete" — new articles land in it all
#: day — so unlike ``ensure_event_window_bars`` there is no already-stored
#: short circuit, and this throttle is the only thing standing between a
#: repeatedly-pressed button and two vendors.
FETCH_ATTEMPT_SECONDS = 6 * 60 * 60

#: Per-ticker last provider attempt (success or failure), process-local like
#: ``fundamentals._refresh_attempts`` and ``analysis._refresh_attempts``. A
#: restart re-attempts, which is correct: a cold process has no evidence the
#: vendor is still failing. Keyed on the TICKER, not on (ticker, event),
#: because the window is the ticker's tape — two events on one symbol share
#: almost all of their articles, so a per-event key would fetch the same
#: hundred rows twice.
_fetch_attempts: dict[str, datetime] = {}

#: How many ranked evidence rows the payload carries (§27). Fifty is far more
#: than the UI lists and enough for "View Evidence" to resolve without a
#: second query, while keeping a mega-cap quarter's payload bounded. The
#: COUNTS above it are computed over everything, so truncating the list never
#: changes the §26 headline.
EVIDENCE_LIMIT = 50

#: The event statuses whose dates are FACTS rather than derivations (§15) —
#: an ESTIMATED past date is this platform's own guess, and anchoring a window
#: to it would open the window on a day nobody reported on. Mirrors
#: ``event_price.COMPARABLE_STATUSES``; spelled here so the news window's
#: anchor rule is readable without a second file.
ANCHOR_STATUSES: frozenset[str] = frozenset(
    {EventStatus.CONFIRMED.value, EventStatus.REVISED.value}
)


# ---------------------------------------------------------------------------
# Provider selection — BOTH vendors when both are configured (§21)
# ---------------------------------------------------------------------------


def news_provider_names(settings) -> list[str]:
    """Every configured provider that can serve a news WINDOW, in fetch order.

    The news counterpart of ``fundamentals.fundamentals_provider_name``, and
    deliberately a LIST rather than a single name — that is the whole
    difference. Statements have exactly one source (only Massive serves
    ``/vX/reference/financials``), so choosing one provider is the honest
    answer there. News does not: Alpaca carries the Benzinga wire and Massive
    carries its own publisher set, they overlap partially, and a window built
    from one alone is a partial view of the tape whose §26 counts are wrong in
    a way nothing in the payload could reveal.

    So both are returned when both are configured, market-data provider first
    (it is the platform's primary and the one whose absence is already
    reported everywhere else). Duplicates are dropped — a deployment whose
    ``market_data_provider`` IS "massive" gets one entry, not two. Merging is
    safe because ``source_id`` is UNIQUE: a story both vendors carry is stored
    once.

    Returns ``[]`` when nothing is configured, which the caller reports as a
    named skip rather than an error.
    """
    names: list[str] = []
    primary = (getattr(settings, "market_data_provider", "") or "").strip()
    if primary:
        names.append(primary)
    if (getattr(settings, "massive_api_key", "") or "").strip():
        names.append("massive")
    # dict.fromkeys keeps first-seen order while dropping the duplicate a
    # massive-primary deployment would otherwise produce.
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# The window (§21) — anchored on the previous comparable event
# ---------------------------------------------------------------------------


async def _previous_anchor_row(
    session: AsyncSession, event_row: EventRow, before: datetime
) -> EventRow | None:
    """The newest same-type, same-ticker event strictly before ``before``.

    The anchor for the window's start. Narrowed to CONFIRMED/REVISED for the
    §15 reason ``event_price._past_comparable_rows`` narrows: an ESTIMATED
    past date is a derivation, and opening a news window at a day nobody
    reported on would frame the whole period against a fiction.

    ``before`` is the event's own instant for the read path and ``now`` for
    the ingest path — the two differ, and passing it in rather than reading
    ``event_row.scheduled_at`` twice is what lets a FUTURE event still anchor
    on its last real predecessor.
    """
    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return None
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.ticker == ticker,
                    EventRow.event_type == event_row.event_type,
                    EventRow.scheduled_at < event_row.scheduled_at,
                )
                .order_by(EventRow.scheduled_at.desc())
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if row.id == event_row.id or row.status not in ANCHOR_STATUSES:
            continue
        when = _as_utc(row.scheduled_at)
        if when >= _as_utc(event_row.scheduled_at) or when > before:
            continue
        return row
    return None


async def news_window(
    session: AsyncSession, event_row: EventRow, *, end: datetime
) -> tuple[datetime, datetime, str]:
    """``(start_utc, end_utc, basis)`` — the tape one event's news covers (§21).

    THE WINDOW IS THE INTER-EVENT PERIOD, which is what makes the §26 counts
    mean anything. "Eleven developments" is a statement about what has happened
    SINCE THE LAST PRINT; measured over a fixed trailing 30 days it would be a
    statement about the calendar, and two events three weeks apart would report
    overlapping news as if it were new twice.

    The start is the previous comparable event's instant minus
    :data:`WINDOW_LEAD_DAYS` (the previews published the evening before frame
    the period that follows). With no such predecessor — a newly-covered
    symbol, or a registry that has not ingested history yet — it falls back to
    ``end`` minus :data:`DEFAULT_WINDOW_DAYS`, and the ``basis`` string says
    which of the two happened so no reader has to infer it.

    ``end`` is supplied by the caller and is the ONE thing that differs between
    the two call graphs: ``now`` when ingesting, ``as_of`` when reading. A
    reversed window cannot arise — the anchor is strictly before the event and
    the fallback is measured backwards from ``end`` — but a pathological
    registry (an anchor after ``as_of``) is clamped rather than allowed to
    invert, because a reversed window makes the provider raise.
    """
    end_utc = _as_utc(end)
    anchor = await _previous_anchor_row(session, event_row, end_utc)
    if anchor is not None:
        start = _as_utc(anchor.scheduled_at) - timedelta(days=WINDOW_LEAD_DAYS)
        basis = f"previous_{(event_row.event_type or 'event').lower()}:{anchor.event_key}"
    else:
        start = end_utc - timedelta(days=DEFAULT_WINDOW_DAYS)
        basis = f"default_{DEFAULT_WINDOW_DAYS}d"
    if start > end_utc:
        # The anchor is later than the read instant (only reachable by asking
        # for an as_of before the previous event). Clamp rather than invert:
        # get_news_window raises on a reversed window, and an empty window is
        # the honest shape for "nothing had happened yet".
        start = end_utc
        basis = f"{basis}:clamped_to_as_of"
    return start, end_utc, basis


# ---------------------------------------------------------------------------
# Ingestion — the only half of this module that holds a provider handle
# ---------------------------------------------------------------------------


def _to_row(article, *, now: datetime) -> NewsArticleRow:
    """One provider article as an unsaved mirror row.

    Every field is the provider's verbatim (migration 012's contract): nothing
    is normalised, translated or truncated on the way in, because the store is
    what LLM evidence is grounded against and a rewritten headline would make
    the citation check meaningless. The Phase D evidence columns are left at
    their defaults — NULL means "not yet analysed", which is the truth until
    :func:`build_event_news` runs.
    """
    return NewsArticleRow(
        source_id=article.source_id,
        title=article.title,
        publisher=article.publisher,
        published_at=_as_utc(article.published_at),
        url=article.url,
        tickers=list(article.tickers),
        description=article.description,
        fetched_at=now,
    )


async def ensure_event_news_window(
    session: AsyncSession,
    event_row: EventRow,
    *,
    provider_names: list[str],
    now: datetime,
) -> dict:
    """Fetch and store the news window for ONE event. The honest report.

    The news counterpart of ``ensure_event_window_bars`` and
    ``ensure_fundamentals``, and deliberately the only path in Phase D that
    writes ``news_articles`` — so the NEWS_INGESTED audit trail and the
    grounding story stay single-sourced (rule 12, ADR-003). It shares the
    table with the Phase 8 recommendations refresh, which is why the upsert
    key is the same one that path uses: ``source_id``, the provider's own
    article id and the column's UNIQUE constraint.

    EVERY CONFIGURED PROVIDER IS ASKED, and each is isolated. Alpaca 403ing
    does not cost Massive its articles; both failing is two named skips and
    ``fetched: 0``, not an exception. That per-item isolation is the same rule
    the calendar ingest applies across providers (§8).

    ``now`` is REQUIRED and is the only clock this function reads — the
    throttle, ``fetched_at`` and the window's end all come from it — so a test
    drives the cadence without patching time. It is NOT an ``as_of``:
    ingestion has no as-of (audit §7.2 rule 1); it writes what the vendors
    serve and lets :func:`build_event_news` decide what was knowable when.

    NO "ALREADY STORED" SHORT CIRCUIT, unlike the minute-bar backfill. A
    minute window is a closed interval in the past and cannot change; a news
    window is open at its right edge and gains articles all day, so refusing
    to refetch a window that already has rows would freeze the tape at
    whatever the first press happened to catch. The per-ticker throttle is
    what bounds the cost instead.

    EVERY PROVIDER FAILURE IS A NAMED SKIP, NEVER AN EXCEPTION. An
    unconfigured provider, a plan without news (403 ->
    :class:`CapabilityNotAvailable`), a transport error and an unexpected
    exception all appear in ``providers[]`` with their reason. The caller is a
    USER-pressed button; it must report why nothing arrived, not 5xx.
    """
    ticker = (event_row.ticker or "").strip().upper()
    base: dict = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": ticker or None,
    }
    if not ticker:
        # A CPI release or an FOMC decision has no issuer whose tape this
        # would be. Phase G gives macro events their §39 proxies; inventing a
        # ticker here would attribute an index's coverage to an event that has
        # none.
        return {
            **base,
            "fetched": False,
            "articles": 0,
            "stored": 0,
            "reason": "no_ticker",
            "providers": [],
        }

    start, end, basis = await news_window(session, event_row, end=now)
    base |= {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "window_basis": basis,
    }

    if not provider_names:
        return {
            **base,
            "fetched": False,
            "articles": 0,
            "stored": 0,
            "reason": "no market data provider is configured for news",
            "providers": [],
        }

    last_attempt = _fetch_attempts.get(ticker)
    if (
        last_attempt is not None
        and (now - last_attempt).total_seconds() < FETCH_ATTEMPT_SECONDS
    ):
        return {
            **base,
            "fetched": False,
            "articles": 0,
            "stored": 0,
            "reason": "news recently fetched for this ticker",
            "providers": [],
        }
    _fetch_attempts[ticker] = now

    fetched: dict[str, object] = {}
    reports: list[dict] = []
    for name in provider_names:
        try:
            provider = get_provider(name)
            window_articles = list(
                provider.get_news_window(
                    tickers=[ticker], start=start, end=end, limit=FETCH_LIMIT
                )
            )
        except (ProviderNotConfigured, CapabilityNotAvailable, MarketDataError) as exc:
            # Named, expected refusals: no key, no subscription, vendor error.
            # The OTHER providers still run.
            logger.info(
                "news_window_unavailable",
                extra={
                    "extra_fields": {
                        "ticker": ticker,
                        "provider": name,
                        "reason": str(exc),
                    }
                },
            )
            reports.append({"provider": name, "fetched": False, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
            logger.exception(
                "news_window_failed",
                extra={"extra_fields": {"ticker": ticker, "provider": name}},
            )
            reports.append({"provider": name, "fetched": False, "reason": str(exc)})
            continue

        kept = 0
        for article in window_articles:
            source_id = (getattr(article, "source_id", "") or "").strip()
            stamp = getattr(article, "published_at", None)
            if not source_id or stamp is None:
                # An article with no id cannot be deduplicated and one with no
                # instant cannot be placed in a window or gated at an as_of.
                # Both are dropped rather than stored with a substituted value.
                continue
            if source_id in fetched:
                # Both vendors carry this story. One article is one article —
                # first parse wins, matching the provider protocol's own
                # de-dup rule.
                continue
            fetched[source_id] = article
            kept += 1
        reports.append(
            {
                "provider": name,
                "fetched": True,
                "articles": len(window_articles),
                "new_to_merge": kept,
                "truncated": len(window_articles) >= FETCH_LIMIT,
            }
        )

    if not fetched:
        return {
            **base,
            "fetched": any(r.get("fetched") for r in reports),
            "articles": 0,
            "stored": 0,
            "reason": "no articles were served for this window",
            "providers": reports,
        }

    existing = set(
        (
            await session.execute(
                select(NewsArticleRow.source_id).where(
                    NewsArticleRow.source_id.in_(list(fetched))
                )
            )
        )
        .scalars()
        .all()
    )
    stored = 0
    for source_id, article in fetched.items():
        if source_id in existing:
            # Already mirrored — by an earlier backfill or by the Phase 8
            # recommendations refresh. The stored row is the same article;
            # rewriting it would churn ``fetched_at`` for no new fact and
            # could overwrite text a reader has already cited.
            continue
        session.add(_to_row(article, now=now))
        stored += 1

    if not stored:
        # Everything the vendors served is already on disk. No audit row (an
        # audit trail of no-ops is noise) and no commit.
        return {
            **base,
            "fetched": True,
            "articles": len(fetched),
            "stored": 0,
            "reason": "all fetched articles were already stored",
            "providers": reports,
        }

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.NEWS_INGESTED,
        entity_type=ENTITY_TYPE,
        entity_id=ticker,
        details={
            "kind": "event_window",
            "ticker": ticker,
            "event_key": event_row.event_key,
            "event_id": event_row.id,
            "fetched": len(fetched),
            "stored": stored,
            "providers": provider_names,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "window_basis": basis,
        },
    )
    try:
        await session.commit()
    except IntegrityError:
        # Cross-process backstop, the same one the Phase 8 ingest carries:
        # another writer landed one of these source_ids between the existence
        # check above and this commit. Their rows ARE the articles — roll ours
        # back and report the collision rather than 500ing a button press.
        await session.rollback()
        logger.info(
            "news_window_conflict", extra={"extra_fields": {"ticker": ticker}}
        )
        return {
            **base,
            "fetched": True,
            "articles": len(fetched),
            "stored": 0,
            "reason": "another writer stored these articles first",
            "providers": reports,
        }
    return {
        **base,
        "fetched": True,
        "articles": len(fetched),
        "stored": stored,
        "providers": reports,
    }


# ---------------------------------------------------------------------------
# Stored reads — the analysis half never holds a provider handle
# ---------------------------------------------------------------------------


def _tagged(row: NewsArticleRow, ticker: str) -> bool:
    """Whether a stored row's ``tickers`` array contains ``ticker``.

    The Python half of the containment test. On Postgres the SQL below narrows
    with ``tickers @> '["T"]'`` against the migration-023 GIN index; SQLite
    (the test harness) has no JSONB operator at all, so the read falls back to
    loading the window and filtering here. Same predicate either way, spelled
    once, so the two dialects cannot disagree about what "tagged" means.
    """
    return any(
        isinstance(tag, str) and tag.strip().upper() == ticker
        for tag in (row.tickers or [])
    )


async def _articles_for_ticker(
    session: AsyncSession, ticker: str, *, start: datetime, end: datetime
) -> list[NewsArticleRow]:
    """Stored articles tagged ``ticker`` published in ``[start, end]``.

    THE ``end`` BOUND IS AN OPTIMISATION, NOT THE AS-OF GATE. The gate lives
    in ``analyze_window`` (§96) and is re-applied to every row handed to it;
    this bound only keeps the read from loading a mega-cap's entire mirror to
    throw most of it away. If this clause were deleted the payload would be
    identical — a property the test suite asserts by planting a post-``as_of``
    article and checking it is absent from the counts, not merely from the SQL.

    The ticker filter is dialect-aware: JSONB containment on Postgres (the
    migration-023 GIN index), a Python pass on SQLite. Nothing else differs.
    """
    stmt = (
        select(NewsArticleRow)
        .where(
            NewsArticleRow.published_at >= start,
            NewsArticleRow.published_at <= end,
        )
        .order_by(NewsArticleRow.published_at.desc())
    )
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        # ``tickers @> '["AAPL"]'`` — the containment test migration 023's GIN
        # index serves. The operator is applied to the BARE column, not to a
        # ``CAST(tickers AS JSONB)``: the column already IS jsonb in Postgres
        # (migration 012), and wrapping an indexed column in a cast is exactly
        # what makes the planner ignore the index and sequential-scan the
        # mirror instead. ``type_coerce`` re-labels the expression's type for
        # SQLAlchemy without emitting any SQL.
        from sqlalchemy import type_coerce
        from sqlalchemy.dialects.postgresql import JSONB

        stmt = stmt.where(
            type_coerce(NewsArticleRow.tickers, JSONB).contains([ticker])
        )
    rows = list((await session.execute(stmt)).scalars().all())
    if dialect == "postgresql":
        return rows
    return [row for row in rows if _tagged(row, ticker)]


def to_raw_articles(rows) -> list[RawArticle]:
    """Stored rows as the pure library's input, order preserved.

    Only the instant is transformed, and it must be: SQLite hands
    ``TIMESTAMPTZ`` columns back NAIVE (the convention ``event_price._as_utc``
    exists for) and ``RawArticle`` REFUSES a naive ``published_at`` rather
    than assuming UTC — correctly, because an unknown zone cannot be proven to
    precede ``as_of`` and guessing would move the §96 boundary by hours. Each
    rule is right and together they would reject every row the database
    returns; re-stamping here, at the ORM boundary, is what keeps the pure
    layer's refusal meaningful.

    ``id`` travels so :func:`_persist_evidence` can write the classification
    back onto the row it came from without a second lookup.
    """
    return [
        RawArticle(
            id=row.id,
            source_id=row.source_id,
            title=row.title or "",
            description=row.description or "",
            publisher=row.publisher or "",
            published_at=_as_utc(row.published_at) if row.published_at else None,
            url=row.url or "",
            tickers=tuple(row.tickers or ()),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Persisting the as-of-INDEPENDENT half of the analysis back onto the rows
# ---------------------------------------------------------------------------


async def _persist_evidence(
    session: AsyncSession, rows, articles, result, *, ticker: str
) -> int:
    """Write ``cluster_id``/``materiality``/``source_quality``/``relevance``.

    ONLY AS-OF-INDEPENDENT FIELDS (audit §7.1, §96; migration 023). Each of
    the four describes the ARTICLE — the category its own text falls in, the
    weight of its own publisher, the story its own source_id anchors, its
    relevance to a named ticker — and means the same thing at every instant.
    ``novelty``, ``decay`` and the composite ``score`` are absent by design:
    they are functions of ``as_of`` and of which other articles shared the
    window, so writing one would let a later read at a different ``as_of``
    inherit this request's viewpoint.

    ``articles`` is the SAME ``RawArticle`` list handed to ``analyze_window``,
    not a second conversion of ``rows``. Rebuilding them here would put a
    duplicate ORM-to-value mapping in the module, and the two copies would be
    free to disagree — the stored ``materiality`` could then differ from the
    one in the payload for the very same article, which is the one thing this
    write must never do.

    ``relevance`` is MERGED, never replaced: it is a per-ticker map, and an
    article tagged AAPL and MSFT is scored for whichever event is being read
    without erasing the other's entry.

    Returns how many rows actually changed; an unchanged row is left alone so
    a repeated read writes nothing and commits nothing.
    """
    by_source = {row.source_id: row for row in rows}
    article_by_source = {article.source_id: article for article in articles}
    cluster_of: dict[str, str] = {}
    for cluster in result.clusters:
        for member in cluster.members:
            cluster_of[member.source_id] = cluster.cluster_id
        # Near-duplicates folded away by dedupe belong to the canonical
        # article's story too — that is what "duplicate_of" MEANS, and leaving
        # them unlabelled would make a syndicated copy look unanalysed.
        for duplicate_id in cluster.duplicates:
            cluster_of[duplicate_id] = cluster.cluster_id

    changed = 0
    for source_id, cluster_id in cluster_of.items():
        row = by_source.get(source_id)
        article = article_by_source.get(source_id)
        if row is None or article is None:
            continue
        materiality = materiality_of(article)
        quality = source_quality(article.publisher)
        relevance = ticker_relevance(article, ticker)
        merged = dict(row.relevance or {})
        row_changed = False
        if row.cluster_id != cluster_id:
            row.cluster_id = cluster_id
            row_changed = True
        if row.materiality != materiality.category:
            row.materiality = materiality.category
            row_changed = True
        if row.materiality_score != materiality.score:
            row.materiality_score = materiality.score
            row_changed = True
        if row.source_quality != quality:
            row.source_quality = quality
            row_changed = True
        if merged.get(ticker) != relevance:
            merged[ticker] = relevance
            # Reassigned rather than mutated in place: SQLAlchemy does not
            # track mutation of a plain JSON dict, so an in-place update would
            # be silently dropped at flush.
            row.relevance = merged
            row_changed = True
        if row_changed:
            changed += 1
    if changed:
        await session.commit()
    return changed


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


async def build_event_news(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> dict:
    """The whole news evidence block for one event, as of one instant (§21-§27).

    Order of operations is the contract: resolve the window -> load STORED
    rows in it -> hand them to the pure layer, which gates on ``published_at``
    and computes -> persist the as-of-independent half back -> render. Nothing
    is classified before the gate, so no score can come from an article the
    caller could not have seen.

    THIS FUNCTION NEVER FETCHES (§27; audit §7.2 rule 1). Not even to top the
    mirror up, which is where it is stricter than ``fundamentals.py``: a news
    window is dozens of paginated requests across two vendors, and opening the
    Catalyst page must not issue them. An event with nothing stored answers
    200 with ``available: false`` and the reason, plus an ``unavailable``
    entry naming the backfill as the remedy. ``POST
    /api/events/{id}/news/backfill`` is the USER action that fetches.

    ``as_of`` is REQUIRED (audit §7.2 rule 2 — a seam that defaults it to
    ``now()`` cannot answer a historical question) and is BOTH the window's
    right edge and the gate: asking about a past earnings date returns the
    news picture that existed then, with the decay measured from then, and an
    article published the next morning invisible.

    Non-ticker events (macro, Fed) return ``{"available": false, "reason":
    "no_ticker"}``: a CPI release has no issuer whose coverage this would be,
    and substituting an index proxy would attribute someone else's tape to it.
    """
    moment = _as_utc(as_of)
    ticker = (event_row.ticker or "").strip().upper()
    payload_base = {
        "event": {
            "event_id": event_row.id,
            "event_key": event_row.event_key,
            "event_type": event_row.event_type,
            "title": event_row.title,
            "ticker": event_row.ticker,
            "scheduled_at_utc": _as_utc(event_row.scheduled_at).isoformat()
            if event_row.scheduled_at
            else None,
        },
        "as_of": moment.isoformat(),
        "model_version": NEWS_MODEL_VERSION,
    }
    if not ticker:
        return {
            **payload_base,
            "available": False,
            "reason": "no_ticker",
            "provenance": {"articles": "DATA", "scores": "QUANT"},
            "unavailable": [
                {
                    "field": "news",
                    "reason": (
                        "this event has no ticker — a macro or Fed release has "
                        "no issuer whose news window this would be (§39 "
                        "multi-asset proxies are Phase G)"
                    ),
                }
            ],
        }

    start, end, basis = await news_window(session, event_row, end=moment)
    window = {"start": start.isoformat(), "end": end.isoformat(), "basis": basis}

    rows = await _articles_for_ticker(session, ticker, start=start, end=end)
    unavailable: list[dict] = []
    if not rows:
        unavailable.append(
            {
                "field": "articles",
                "reason": (
                    f"no news articles are stored for {ticker} in this window "
                    "— press \"Fetch news for this window\" (POST "
                    "/api/events/{id}/news/backfill) to ask the configured "
                    "providers"
                ),
            }
        )

    # Converted ONCE and reused for the write-back below: a second conversion
    # would be a second ORM-to-value mapping free to drift from this one, and
    # the stored classification would then be allowed to disagree with the
    # payload's for the same article.
    articles = to_raw_articles(rows)
    result = analyze_window(
        articles,
        ticker=ticker,
        as_of=moment,
        window_start=start,
    )
    analysis = result.to_dict()

    # The as-of-INDEPENDENT half of the analysis goes back onto the rows; the
    # as-of-dependent half (novelty, decay, score) deliberately does not.
    await _persist_evidence(session, rows, articles, result, ticker=ticker)

    # --- freshness (§27) --------------------------------------------------
    # Computed over the STORED window rows, not over the ranked evidence: an
    # article excluded by relevance is still evidence that the mirror is warm,
    # and reporting "no news since March" because the newest piece was
    # off-topic would be a false staleness signal.
    newest_article = max(
        (_as_utc(row.published_at) for row in rows if row.published_at is not None),
        default=None,
    )
    last_fetch = max(
        (_as_utc(row.fetched_at) for row in rows if row.fetched_at is not None),
        default=None,
    )
    freshness = {
        "newest_article_at": newest_article.isoformat() if newest_article else None,
        "last_fetch_at": last_fetch.isoformat() if last_fetch else None,
        "articles_stored": len(rows),
    }

    return {
        **payload_base,
        "available": bool(rows),
        "provenance": {"articles": "DATA", "scores": "QUANT"},
        "window": window,
        "counts": analysis["counts"],
        "themes": analysis["themes"],
        "clusters": analysis["clusters"],
        # Ranked, truncated for transport. The COUNTS above are computed over
        # everything, so the §26 headline never changes with this cut.
        "evidence": analysis["evidence"][:EVIDENCE_LIMIT],
        "evidence_total": len(analysis["evidence"]),
        "evidence_limit": EVIDENCE_LIMIT,
        "excluded": analysis["excluded"],
        "material_threshold": MATERIAL_SCORE_THRESHOLD,
        "untrusted_text_policy": analysis["untrusted_text_policy"],
        "freshness": freshness,
        "unavailable": unavailable,
    }
