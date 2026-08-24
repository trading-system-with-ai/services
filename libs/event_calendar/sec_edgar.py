"""SEC EDGAR submissions provider — authoritative PAST earnings + the
cadence ESTIMATE for the next one (audit §6 fallback chain steps 2 and 3).

Why this module carries the earnings calendar: the 2026-08-19 entitlement
probe found **no** subscribed source for upcoming earnings dates (audit §13
— every Benzinga endpoint 403). EDGAR is free, authoritative and
point-in-time safe, so the platform derives what it can honestly derive:

  step 2 — CONFIRMED PAST releases. A company announces results in an 8-K
           tagged **Item 2.02** ("Results of Operations and Financial
           Condition"). The filing's ``acceptanceDateTime`` is the instant
           the release became public — the true §85 point-in-time key, not
           the period end and not ``filingDate`` (which is a calendar date
           and would put an after-close release on the wrong session).

  step 3 — ESTIMATED next release, from that history alone. See
           :func:`estimate_next_earnings`. It is written with
           ``EventStatus.ESTIMATED`` / ``EventSourceKind.DERIVED`` and can
           never be promoted to a fact by this module; only a subscribed
           calendar (step 1) or the user (step 4) may confirm it.

Endpoints:

  - ``GET https://www.sec.gov/files/company_tickers.json`` — ticker -> CIK.
    Cached for the process lifetime: the mapping changes on the order of
    days and this is a courtesy to a free public service.
  - ``GET https://data.sec.gov/submissions/CIK##########.json`` — the
    filing index. ``filings.recent`` holds ~1000 recent filings as PARALLEL
    ARRAYS (``accessionNumber[i]`` describes the same filing as ``form[i]``);
    ``filings.files`` lists older pages, fetched only when ``recent`` did not
    yield enough earnings history.

SEC requires a descriptive contact ``User-Agent`` and rate-limits to 10
requests/second. Both are honoured: the User-Agent is a required constructor
argument (no silent default that would get the platform blocked) and calls
are spaced by :data:`MIN_REQUEST_INTERVAL_SECONDS`. A 403 (bad User-Agent)
or 429 (too fast) becomes a capability verdict, never a crash — the loop
must survive one source failing (§8).
"""
import logging
import re
import statistics
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import httpx
from dataclasses import replace

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from .provider import (
    EASTERN,
    US_EVENT_TIMEZONE,
    CalendarProviderError,
    EventCandidate,
    MarketDay,
    blank_capabilities,
    classify_session_et,
)

logger = logging.getLogger(__name__)

SEC_WWW_BASE_URL = "https://www.sec.gov"
SEC_DATA_BASE_URL = "https://data.sec.gov"
COMPANY_TICKERS_PATH = "/files/company_tickers.json"

DEFAULT_TIMEOUT_SECONDS = 15.0

#: SEC's published fair-access limit is 10 requests/second. 0.12s between
#: calls keeps a comfortable margin without making ingestion slow.
MIN_REQUEST_INTERVAL_SECONDS = 0.12

#: The 8-K item that means "we just reported results".
EARNINGS_ITEM = "2.02"

#: How many past earnings events the history walk aims for before it stops
#: paging into ``filings.files`` (3 years of quarters).
DEFAULT_HISTORY_TARGET = 12

#: Older-page fetches per ticker. Bounded so a pathological filer cannot make
#: one ticker monopolise the tick.
MAX_OLDER_PAGES = 4

SOURCE_NAME = "sec_edgar"
DERIVED_SOURCE_NAME = "derived_cadence"

#: Eastern anchor time for an ESTIMATED release, by the modal session of the
#: recent history. Deliberately NOT the exact time of the last release: the
#: estimate claims a session, not a minute, and a spuriously precise
#: timestamp would read as a confirmed fact on the card.
SESSION_ANCHOR_ET: dict[EventSession, tuple[int, int]] = {
    EventSession.BEFORE_MARKET: (7, 0),
    EventSession.AFTER_MARKET: (16, 5),
    EventSession.DURING_MARKET: (12, 0),
    EventSession.UNKNOWN: (12, 0),
}

#: A year of quarters: 364 days = exactly 52 weeks, so "same quarter last
#: year + 364d" lands on the SAME WEEKDAY, which is how issuers actually
#: schedule (e.g. "the Thursday of week 4 after quarter end").
YEAR_OF_WEEKS_DAYS = 364

#: How far a candidate may sit from "exactly one year before the last
#: release" and still count as that release's year-ago twin. A quarter is
#: ~91 days, so 45 days keeps the match inside the right quarter while
#: tolerating the weeks of drift issuers actually show.
YEAR_ANCHOR_TOLERANCE_DAYS = 45

#: Gaps used for the median fallback when there is no year-ago anchor.
MEDIAN_GAP_SAMPLE = 3

#: Minimum history before any estimate is emitted. One past release says
#: nothing about cadence; a wrong estimate is worse than an honest absence.
MIN_HISTORY_FOR_ESTIMATE = 2

#: History needed before the (stronger) year-ago anchor is trusted.
MIN_HISTORY_FOR_YEAR_ANCHOR = 4

#: Sessions sampled for the modal-session vote.
MODAL_SESSION_SAMPLE = 4

#: Process-lifetime ticker -> CIK cache, guarded because provider instances
#: are per-request but the map is global and expensive.
_CIK_CACHE: dict[str, str] = {}
_CIK_CACHE_LOCK = threading.Lock()

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


def reset_cik_cache() -> None:
    """Drop the process-lifetime ticker->CIK map (tests; operator refresh)."""
    with _CIK_CACHE_LOCK:
        _CIK_CACHE.clear()


def _parse_acceptance(raw: object) -> datetime | None:
    """EDGAR ``acceptanceDateTime`` -> a UTC instant.

    EDGAR stamps ``"2026-08-27T16:05:12.000Z"`` (UTC, explicit) but has also
    served ``"2026-08-27T16:05:12-04:00"`` and a naive form. A naive value is
    read as EASTERN — EDGAR's own local clock — because assuming UTC would
    shift an after-close release four hours into the next morning and flip
    its session from AFTER_MARKET to BEFORE_MARKET.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    return parsed.astimezone(timezone.utc)


def _accession_nodashes(accession: str) -> str:
    return accession.replace("-", "")


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    """Eastern wall-clock -> UTC instant (DST resolved from the date)."""
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


def _next_weekday(day: date) -> date:
    """Snap Sat/Sun forward to Monday. Issuers do not report at weekends."""
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _modal_session(sessions: Sequence[EventSession]) -> EventSession:
    """The most common session, ties broken toward the most RECENT.

    `sessions` is newest-first, so on a 2-2 tie the newer habit wins — an
    issuer that moved from BMO to AMC last year is estimated AMC.
    """
    counts: dict[EventSession, int] = {}
    for s in sessions:
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return EventSession.UNKNOWN
    best = max(counts.values())
    for s in sessions:  # newest-first order breaks the tie
        if counts[s] == best:
            return s
    return EventSession.UNKNOWN  # pragma: no cover — unreachable


#: Two Item 2.02 filings closer than this are the SAME quarterly release:
#: the first is the earnings press release, later ones are follow-ups (a
#: preliminary-results update, an investor-day deck furnished under 2.02, an
#: amendment). Matches the registry's EARNINGS same_event window.
RELEASE_CLUSTER_DAYS = 21


def cluster_releases(
    history: Sequence[EventCandidate], *, window_days: int = RELEASE_CLUSTER_DAYS
) -> list[EventCandidate]:
    """Collapse Item 2.02 filings into one candidate per quarterly release.

    PURE. Input in any order; output NEWEST FIRST with the EARLIEST filing
    of each cluster kept — the release is the first 2.02 of the quarter, and
    a later 2.02 inside the window must not read as a "rescheduled" earnings
    date (live 2026-08-19: SMCI's follow-up 8-Ks were turning CONFIRMED rows
    into REVISED). Dropped filings stay visible in ``raw['follow_ups']`` of
    the kept candidate so nothing is silently lost.
    """
    ordered = sorted(history, key=lambda c: c.scheduled_at)
    kept: list[EventCandidate] = []
    follow_ups: dict[int, list[str]] = {}
    for cand in ordered:
        if kept and (cand.scheduled_at - kept[-1].scheduled_at) <= timedelta(days=window_days):
            follow_ups.setdefault(len(kept) - 1, []).append(
                cand.source_event_id or cand.scheduled_at.isoformat()
            )
            continue
        kept.append(cand)
    out: list[EventCandidate] = []
    for idx, cand in enumerate(kept):
        extra = follow_ups.get(idx)
        if extra:
            raw = dict(cand.raw or {})
            raw["follow_ups"] = list(extra)
            cand = replace(cand, raw=raw)
        out.append(cand)
    out.reverse()
    return out


def estimate_next_earnings(
    history: Sequence[EventCandidate],
    *,
    ticker: str | None = None,
    now: datetime | None = None,
) -> EventCandidate | None:
    """The next earnings date derived from filing cadence (audit §6 step 3).

    PURE — no I/O, so it is unit-testable on hand-built histories and can be
    called by the gateway on rows already in the database.

    ``history`` is past EARNINGS candidates for ONE ticker in any order.
    Rules, in the order the audit specifies:

    1. Fewer than :data:`MIN_HISTORY_FOR_ESTIMATE` past releases -> ``None``.
       An honest absence beats a guess drawn from one observation.
    2. With >= :data:`MIN_HISTORY_FOR_YEAR_ANCHOR` releases, anchor on the
       SAME FISCAL QUARTER LAST YEAR + 364 days — the issuer's real schedule
       is a weekday in a fixed week after quarter end, and 52 weeks preserves
       the weekday.
    3. Otherwise: last release + the MEDIAN of the last
       :data:`MEDIAN_GAP_SAMPLE` gaps. The median (not the mean) so one
       delayed quarter cannot drag the estimate.
    4. Snap a weekend result forward to Monday.
    5. Time of day = the anchor for the MODAL session of the last
       :data:`MODAL_SESSION_SAMPLE` releases — a session claim, not a minute.

    The result is always ``ESTIMATED`` / ``DERIVED`` / ``derived_cadence``.
    """
    past = sorted(history, key=lambda c: c.scheduled_at)
    if len(past) < MIN_HISTORY_FOR_ESTIMATE:
        return None
    symbol = (ticker or past[-1].ticker or "").strip().upper()
    if not symbol:
        return None

    last = past[-1]
    estimated_at: datetime | None = None
    method = "median_gap"

    if len(past) >= MIN_HISTORY_FOR_YEAR_ANCHOR:
        # The NEXT release's analogue one year ago, located by CALENDAR
        # PROXIMITY rather than by counting back four positions.
        #
        # WHY NOT past[-4]. That indexing silently assumes the issuer filed
        # exactly four times in the last year. Real histories break it: HPE's
        # record carries both 2025-09-03 and 2025-10-15, so past[-4] pointed
        # at October and the estimate skipped the imminent September report
        # (stored 2026-10-14 while HPE's own IR said 2026-09-02). An
        # off-by-one in WHICH release we anchor on is a whole QUARTER of
        # error, and it lands on the side that HIDES a near-term event.
        #
        # Two cases, both handled without counting positions:
        #  - the last release HAS a year-ago twin (a regular filer): the next
        #    release's analogue is whatever the issuer filed after that twin;
        #  - it does NOT (the history starts mid-cycle): the analogue is
        #    simply the earliest release after `target`.
        target = last.scheduled_at - timedelta(days=YEAR_OF_WEEKS_DAYS)
        twin = min(
            past[:-1],
            key=lambda c: abs((c.scheduled_at - target).total_seconds()),
        )
        twin_is_real = (
            abs((twin.scheduled_at - target).days) <= YEAR_ANCHOR_TOLERANCE_DAYS
        )
        # The anchor is drawn from the HISTORY BEFORE the last release
        # (past[:-1]) in both branches: anchoring on the last release itself
        # would just re-date it a year out, which is a cadence claim the
        # history has not evidenced.
        if twin_is_real:
            following = [
                c for c in past[:-1] if c.scheduled_at > twin.scheduled_at
            ]
        else:
            following = [c for c in past[:-1] if c.scheduled_at > target]
        anchor = following[0] if following else None
        estimated_at = (
            anchor.scheduled_at + timedelta(days=YEAR_OF_WEEKS_DAYS)
            if anchor is not None
            else None
        )
        method = "same_quarter_last_year+364d"
        if estimated_at is None or estimated_at <= last.scheduled_at:
            # Either no year-ago analogue exists (a gap in the record), or
            # the anchor lands at/before the most recent release (an issuer
            # that pulled its schedule forward). Neither can be the NEXT
            # event, so fall through to the gap median.
            estimated_at = None
            method = "median_gap"

    if estimated_at is None:
        gaps = [
            (past[i].scheduled_at - past[i - 1].scheduled_at).total_seconds()
            for i in range(1, len(past))
        ][-MEDIAN_GAP_SAMPLE:]
        if not gaps:
            return None
        estimated_at = last.scheduled_at + timedelta(
            seconds=statistics.median(gaps)
        )

    # An estimate at or before `now` is not a FUTURE event (live 2026-08-19:
    # an issuer whose year-ago anchor had already passed produced nothing).
    # Roll forward by the median gap — a bounded walk, never a guess past
    # two years.
    if now is not None and estimated_at <= now:
        gaps_all = [
            (past[i].scheduled_at - past[i - 1].scheduled_at).total_seconds()
            for i in range(1, len(past))
        ][-MEDIAN_GAP_SAMPLE:]
        step = timedelta(seconds=statistics.median(gaps_all)) if gaps_all else None
        if step is None or step.total_seconds() <= 0:
            return None
        rolled = 0
        while estimated_at <= now and rolled < 8:
            estimated_at += step
            rolled += 1
        if estimated_at <= now:
            return None
        method = f"{method}+rolled_forward_x{rolled}"

    session = _modal_session(
        [c.session for c in reversed(past[-MODAL_SESSION_SAMPLE:])]
    )
    est_day = _next_weekday(estimated_at.astimezone(EASTERN).date())
    scheduled_at = _et_to_utc(est_day, SESSION_ANCHOR_ET[session])

    return EventCandidate(
        event_key=f"EARNINGS:{symbol}:{est_day.isoformat()}",
        event_type=EventType.EARNINGS,
        title=f"{symbol} earnings (estimated from filing cadence)",
        scheduled_at=scheduled_at,
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name=DERIVED_SOURCE_NAME,
        ticker=symbol,
        company_id=last.company_id,
        event_timezone=US_EVENT_TIMEZONE,
        session=session,
        last_verified_at=now,
        raw={
            "method": method,
            "history_size": len(past),
            "last_release_utc": last.scheduled_at.isoformat(),
        },
    )


class SecEdgarProvider:
    """EventCalendarProvider over SEC EDGAR submissions. Free, authoritative."""

    name = SOURCE_NAME

    def __init__(
        self,
        user_agent: str,
        www_base_url: str = SEC_WWW_BASE_URL,
        data_base_url: str = SEC_DATA_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        history_target: int = DEFAULT_HISTORY_TARGET,
        cik_map: dict[str, str] | None = None,
    ) -> None:
        """`transport` is injectable so tests use httpx.MockTransport.

        `user_agent` is REQUIRED and must carry contact information — SEC
        blocks anonymous scrapers with a 403. Refusing a blank value at
        construction is deliberate: a silent default would get the whole
        platform's IP throttled and the failure would look like an outage.

        `cik_map` lets a caller supply ticker->CIK directly (the audit's
        "fallback: caller may pass cik map"), skipping the company_tickers
        fetch entirely.
        """
        if not user_agent or not user_agent.strip():
            raise CalendarProviderError(
                "SEC EDGAR requires a contact User-Agent (settings.sec_user_agent, "
                "e.g. 'trading-system-with-ai/0.1 (you@example.com)') — SEC "
                "returns HTTP 403 without one"
            )
        self.user_agent = user_agent.strip()
        self.www_base_url = www_base_url.rstrip("/")
        self.data_base_url = data_base_url.rstrip("/")
        self.history_target = history_target
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_at = 0.0
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        if cik_map:
            with _CIK_CACHE_LOCK:
                _CIK_CACHE.update(
                    {k.strip().upper(): str(v).zfill(10) for k, v in cik_map.items()}
                )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    # ------------------------------------------------------------------
    # Transport (rate-limited; SEC's failure taxonomy)
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Space requests to honour SEC's 10 req/s fair-access limit."""
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def _request(self, url: str) -> httpx.Response:
        endpoint = httpx.URL(url).path  # path only, never a query string
        self._throttle()
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"SEC EDGAR request failed for {endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"SEC EDGAR returned HTTP 403 for {endpoint}: SEC requires a "
                "descriptive contact User-Agent (set SEC_USER_AGENT). There "
                f"is NO synthetic fallback: {response.text[:300]}"
            )
        if response.status_code == 429:
            raise CalendarProviderError(
                f"SEC EDGAR rate limited (HTTP 429) on {endpoint} — the "
                "10 req/s fair-access limit was exceeded; back off and retry "
                "on the next ingestion tick"
            )
        if response.status_code == 404:
            raise CalendarProviderError(
                f"SEC EDGAR has no document at {endpoint} (HTTP 404)"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"SEC EDGAR returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response

    def _json(self, url: str) -> dict:
        response = self._request(url)
        try:
            payload = response.json()
        except ValueError as exc:
            raise CalendarProviderError(
                f"SEC EDGAR returned unparseable JSON for {httpx.URL(url).path}"
            ) from exc
        if not isinstance(payload, dict):
            raise CalendarProviderError(
                f"SEC EDGAR returned a non-object payload for {httpx.URL(url).path}"
            )
        return payload

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def _load_cik_map(self) -> dict[str, str]:
        """ticker -> zero-padded 10-digit CIK, cached for the process life.

        ``company_tickers.json`` is an OBJECT keyed by row index:
        ``{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}``.
        """
        with _CIK_CACHE_LOCK:
            if _CIK_CACHE:
                return dict(_CIK_CACHE)
        payload = self._json(f"{self.www_base_url}{COMPANY_TICKERS_PATH}")
        mapping: dict[str, str] = {}
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker")
            cik = row.get("cik_str")
            if isinstance(ticker, str) and ticker.strip() and cik is not None:
                try:
                    mapping[ticker.strip().upper()] = str(int(cik)).zfill(10)
                except (TypeError, ValueError):
                    continue
        if not mapping:
            raise CalendarProviderError(
                "SEC company_tickers.json contained no usable ticker->CIK rows"
            )
        with _CIK_CACHE_LOCK:
            _CIK_CACHE.update(mapping)
            return dict(_CIK_CACHE)

    def resolve_cik(self, ticker: str) -> str | None:
        """The 10-digit CIK for `ticker`, or None if SEC does not list it."""
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return None
        with _CIK_CACHE_LOCK:
            cached = _CIK_CACHE.get(symbol)
        if cached:
            return cached
        return self._load_cik_map().get(symbol)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Probe the submissions endpoint with one always-present CIK.

        EDGAR has no upcoming-earnings feed at all, so ``earnings_calendar``
        stays ``False`` while ``earnings_history`` reflects the live probe —
        the distinction the whole fallback chain rests on.
        """
        report = blank_capabilities()
        try:
            # Apple, CIK 0000320193 — the canonical always-present filer.
            self._json(f"{self.data_base_url}/submissions/CIK0000320193.json")
        except CapabilityNotAvailable as exc:
            logger.warning("SEC capability 'earnings_history' unavailable: %s", exc)
            report["earnings_history"] = False
        except MarketDataError as exc:
            report["earnings_history"] = str(exc)
        else:
            report["earnings_history"] = True
        return report

    # ------------------------------------------------------------------
    # Filings -> earnings candidates
    # ------------------------------------------------------------------

    def fetch_earnings_history(
        self,
        ticker: str,
        *,
        as_of: datetime | None = None,
        limit: int | None = None,
    ) -> list[EventCandidate]:
        """Past CONFIRMED earnings releases for `ticker`, NEWEST FIRST.

        Every candidate is one 8-K carrying Item 2.02. ``as_of`` drops
        filings accepted after it (§96 look-ahead): at as_of=T the platform
        must not know about a release that happened at T+1h.
        """
        symbol = (ticker or "").strip().upper()
        cik = self.resolve_cik(symbol)
        if cik is None:
            logger.info("SEC EDGAR does not list ticker %r — no earnings history", symbol)
            return []
        want = limit or self.history_target
        payload = self._json(f"{self.data_base_url}/submissions/CIK{cik}.json")

        recent = payload.get("filings")
        recent_block = recent.get("recent") if isinstance(recent, dict) else None
        events = self._from_filings_block(
            symbol, cik, recent_block if isinstance(recent_block, dict) else {}, as_of
        )

        # Older pages only if `recent` did not reach the target — one fetch
        # per page, bounded, and only when it can actually help.
        files = recent.get("files") if isinstance(recent, dict) else None
        if len(events) < want and isinstance(files, list):
            for page in files[:MAX_OLDER_PAGES]:
                if len(events) >= want:
                    break
                if not isinstance(page, dict):
                    continue
                name = page.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                try:
                    older = self._json(f"{self.data_base_url}/submissions/{name.strip()}")
                except MarketDataError as exc:
                    logger.warning(
                        "SEC EDGAR older filings page %r unreadable: %s", name, exc
                    )
                    break
                events.extend(self._from_filings_block(symbol, cik, older, as_of))

        events.sort(key=lambda c: c.scheduled_at, reverse=True)
        return events[:want]

    def _from_filings_block(
        self,
        symbol: str,
        cik: str,
        block: dict,
        as_of: datetime | None,
    ) -> list[EventCandidate]:
        """Parse one PARALLEL-ARRAY filings block into earnings candidates.

        The arrays are index-aligned: ``form[i]``, ``items[i]``,
        ``acceptanceDateTime[i]`` all describe filing ``accessionNumber[i]``.
        A row whose arrays are short or whose acceptance instant is
        unparseable is SKIPPED — never defaulted to ``filingDate``, which is
        a date and would misplace an after-close release.
        """
        accessions = block.get("accessionNumber")
        forms = block.get("form")
        if not isinstance(accessions, list) or not isinstance(forms, list):
            return []
        items_arr = block.get("items") if isinstance(block.get("items"), list) else []
        accept_arr = (
            block.get("acceptanceDateTime")
            if isinstance(block.get("acceptanceDateTime"), list) else []
        )
        filed_arr = (
            block.get("filingDate") if isinstance(block.get("filingDate"), list) else []
        )
        primary_arr = (
            block.get("primaryDocument")
            if isinstance(block.get("primaryDocument"), list) else []
        )

        def at(arr: list, i: int) -> object:
            return arr[i] if i < len(arr) else None

        out: list[EventCandidate] = []
        for i in range(min(len(accessions), len(forms))):
            form = at(forms, i)
            # "8-K/A" is an AMENDMENT to an already-captured release: taking it
            # too would create a second earnings event for the same quarter.
            if form != "8-K":
                continue
            items = at(items_arr, i)
            if not isinstance(items, str) or EARNINGS_ITEM not in items:
                continue
            accession = at(accessions, i)
            if not isinstance(accession, str) or not _ACCESSION_RE.match(accession):
                continue
            accepted = _parse_acceptance(at(accept_arr, i))
            if accepted is None:
                logger.debug(
                    "SEC filing %s skipped: unparseable acceptanceDateTime %r",
                    accession, at(accept_arr, i),
                )
                continue
            if as_of is not None and accepted > as_of:
                # §96: at as_of=T this filing had not happened yet.
                continue
            primary = at(primary_arr, i)
            doc = primary if isinstance(primary, str) and primary.strip() else ""
            url = (
                f"{self.www_base_url}/Archives/edgar/data/{int(cik)}/"
                f"{_accession_nodashes(accession)}/{doc}"
                if doc else
                f"{self.www_base_url}/Archives/edgar/data/{int(cik)}/"
                f"{_accession_nodashes(accession)}"
            )
            release_day = accepted.astimezone(EASTERN).date()
            out.append(
                EventCandidate(
                    event_key=f"EARNINGS:{symbol}:{release_day.isoformat()}",
                    event_type=EventType.EARNINGS,
                    title=f"{symbol} earnings release (8-K Item 2.02)",
                    scheduled_at=accepted,
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.COMPANY_IR_SEC,
                    source_name=SOURCE_NAME,
                    ticker=symbol,
                    company_id=cik,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=classify_session_et(accepted),
                    source_url=url,
                    source_event_id=accession,
                    raw={
                        "form": form,
                        "items": items,
                        "filingDate": at(filed_arr, i),
                        "acceptanceDateTime": at(accept_arr, i),
                        "cik": cik,
                    },
                )
            )
        return cluster_releases(out)

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
        include_next: bool = False,
    ) -> list[EventCandidate]:
        """Past releases inside ``[start, end]`` plus the cadence estimate.

        The history walk always fetches the FULL recent history (the estimate
        needs it) but only past events inside the window are returned. The
        ESTIMATED next release is returned when it falls in the window, or
        unconditionally with ``include_next=True`` — the gateway uses that to
        seed a card the user can confirm before the window reaches it.

        One ticker failing never fails the batch: its error is logged and the
        other tickers' candidates are still returned (§8).
        """
        out: list[EventCandidate] = []
        for ticker in tickers:
            symbol = (ticker or "").strip().upper()
            if not symbol:
                continue
            try:
                history = self.fetch_earnings_history(symbol, as_of=as_of)
            except MarketDataError as exc:
                logger.warning("SEC EDGAR earnings history failed for %s: %s", symbol, exc)
                continue
            out.extend(c for c in history if start <= c.scheduled_at <= end)
            estimate = estimate_next_earnings(history, ticker=symbol, now=as_of)
            if estimate is not None and (
                include_next or start <= estimate.scheduled_at <= end
            ):
                out.append(estimate)
        return out

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """EDGAR knows filings, not exchange sessions."""
        raise CapabilityNotAvailable(
            "SEC EDGAR does not serve exchange sessions — use the "
            "alpaca_calendar provider for market_calendar"
        )
