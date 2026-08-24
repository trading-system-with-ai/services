"""Massive-backed market data provider (development plan §22.1).

Massive is the ONLY supported source of REAL market data on this platform.
This adapter speaks the Massive REST API over raw httpx — deliberately not an
SDK, matching every other provider in this codebase (``libs/broker/alpaca.py``,
``libs/llm/anthropic.py``, ``libs/llm/openai.py``). The API key comes from
configuration (``settings.massive_api_key`` / ``MASSIVE_API_KEY``) — never
hardcoded here, never logged, never echoed into an error message.

NO SYNTHETIC FALLBACK, EVER (§44 rule 18). Every number returned by this
module was reported by Massive. When Massive cannot answer — network fault,
bad key, plan-gated endpoint, unknown symbol — the answer is an explicit
error or an honest absence, never an invented value:

  - transport/HTTP faults, bad payloads -> :class:`MarketDataError`;
  - HTTP 403 (endpoint not in the subscribed plan) ->
    :class:`CapabilityNotAvailable` naming the endpoint (§16 capability
    detection);
  - a symbol the snapshot does not know -> that symbol is SKIPPED with a
    WARNING (the platform renders a missing quote as absent);
  - an option contract without a usable quote or without greeks -> that
    contract is SKIPPED with a debug log (the §9 selector needs real quotes
    and real greeks; zero-filling would fabricate them).

Endpoints (verified against the Massive docs before implementation):

  - Daily bars:   ``GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}``
    (https://massive.com/docs/rest/stocks/aggregates/custom-bars.md) —
    ``results[]`` rows carry ``o,h,l,c,v,vw,t,n`` with ``t`` in unix ms.
  - Stock quote:  ``GET /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}``
    (https://massive.com/docs/rest/stocks/snapshots/single-ticker-snapshot.md)
    — ``ticker`` object with ``day``, ``prevDay``, ``lastTrade``,
    ``todaysChange``, ``todaysChangePerc``, ``updated`` (unix ns).
  - Index quote:  ``GET /v3/snapshot/indices?ticker=I:<NAME>``
    (https://massive.com/docs/rest/indices/snapshots/indices-snapshot.md) —
    indices (VIX, SPX, ...) use the ``I:`` ticker form and are NOT served by
    the stocks snapshot; ``results[]`` rows carry ``value``,
    ``session.change_percent`` and ``last_updated`` (unix ns).
  - Option chain: ``GET /v3/snapshot/options/{underlying}?limit=250``
    (https://massive.com/docs/rest/options/snapshots/option-chain-snapshot.md)
    — ``results[]`` rows carry ``details`` (ticker/contract_type/
    strike_price/expiration_date), ``greeks``, ``implied_volatility``,
    ``open_interest``, ``last_quote`` (bid/ask/midpoint), ``day``,
    ``last_trade`` and ``underlying_asset.price``; paginated via
    ``next_url``.

Auth is header-first (``Authorization: Bearer <key>``); if the deployment
rejects header auth with a 401 the adapter falls back ONCE to the documented
``apiKey`` query parameter and remembers whichever form worked. The key never
appears in a log line either way.
"""
import logging
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Sequence

import httpx

from .provider import (
    Bar,
    CapabilityNotAvailable,
    FinancialStatement,
    IntradayBar,
    MarketDataError,
    NewsArticle,
    OptionContractRef,
    OptionQuote,
    Quote,
    require_aware_utc,
)

logger = logging.getLogger(__name__)

MASSIVE_BASE_URL = "https://api.massive.com"
DEFAULT_TIMEOUT_SECONDS = 15.0

# Rate-limit backoff after HTTP 429. Massive answers a burst of option-bar
# fetches (four per event, eight events in one history backfill) with 429s
# that outlast a single pause, so the adapter retries a BOUNDED number of
# times with growing waits instead of once. Every value is a parameter, never
# a hardcoded truth (§6.2): the constructor takes both, so a caller with a
# thinner plan can widen the ladder without editing this module.
#
# ``DEFAULT_RETRY_AFTER_SECONDS`` is the wait for the FIRST retry when the
# server sends no Retry-After header; the rest of the ladder follows it. The
# server's own Retry-After always wins when present — it is the vendor
# stating when the window reopens — capped at
# ``RATE_LIMIT_MAX_DELAY_SECONDS`` so a hostile or mistaken header cannot
# park a request for an hour.
DEFAULT_RETRY_AFTER_SECONDS = 2.0
DEFAULT_MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0, 12.0, 25.0)
RATE_LIMIT_MAX_DELAY_SECONDS = 30.0

# Option-chain pagination: the server pages at 250 contracts; we follow
# next_url for at most this many pages (8 x 250 = 2000 contracts) so a
# pathological chain cannot loop forever. Hitting the cap logs a WARNING —
# a truncated chain is a data-completeness fact the operator should see.
CHAIN_PAGE_LIMIT = 250
DEFAULT_MAX_CHAIN_PAGES = 8

# from-date buffer for daily bars: fetch `days * 1.6` CALENDAR days so
# weekends and holidays still leave at least `days` TRADING bars to trim to.
CALENDAR_BUFFER = 1.6

# Intraday aggregates: the server's per-page maximum for the aggs endpoint,
# and the next_url page cap. One event window is ~2 sessions of minutes, so
# the cap is head-room against a server that never stops sending a cursor,
# not a working limit; hitting it logs a WARNING so the truncation is visible.
INTRADAY_PAGE_LIMIT = 50_000
DEFAULT_MAX_INTRADAY_PAGES = 20

# Windowed news (Phase D §21): the server's per-page maximum for
# /v2/reference/news, and the next_url page cap PER TICKER. An event window is
# ~120 days of one ticker's coverage, which fits in a page or two; the cap is
# head-room against a server that never stops sending a cursor, not a working
# limit, and hitting it logs a WARNING so the truncation is visible.
NEWS_PAGE_LIMIT = 1000
DEFAULT_MAX_NEWS_PAGES = 10

# Dated contract reference (Phase I §36): the server's page size for
# /v3/reference/options/contracts and the next_url page cap for ONE expiry.
# A single expiry of a liquid name is a few hundred strikes, so two pages is
# the working case and the cap is head-room against a server that never stops
# sending a cursor. Hitting it logs a WARNING — a truncated strike grid could
# silently exclude the ATM strike, which is exactly the one §36 needs.
CONTRACTS_PAGE_LIMIT = 250
DEFAULT_MAX_CONTRACT_PAGES = 8

# Historical option aggregates: the server's per-page maximum for
# /v2/aggs/ticker/{option}/range/1/day. One contract's whole life is a few
# hundred sessions, so a single page covers any event window; the cap exists
# so a cursor loop cannot run forever.
OPTION_BAR_PAGE_LIMIT = 5000
DEFAULT_MAX_OPTION_BAR_PAGES = 5

# Massive's timespan vocabulary, keyed by the platform's provider-neutral
# timeframe strings. Alpaca spells a minute bar "1Min"; the aggs URL spells
# the same thing as multiplier=1 / timespan=minute. A timeframe absent here is
# refused by name rather than silently served at the wrong resolution — bars
# at a resolution the caller did not ask for are wrong numbers, not missing
# ones.
INTRADAY_TIMEFRAMES: dict[str, tuple[int, str]] = {
    "1Min": (1, "minute"),
    "5Min": (5, "minute"),
    "15Min": (15, "minute"),
    "1Hour": (1, "hour"),
}

# Massive bar/snapshot timestamps are unix epoch instants; the trading DATE of
# an instant is its US/Eastern (exchange time) date.
EASTERN = ZoneInfo("America/New_York")

# Symbols the platform spells bare but Massive serves ONLY as indices in the
# "I:" ticker form on /v3/snapshot/indices (verified:
# https://massive.com/docs/rest/indices/snapshots/indices-snapshot.md).
# "^"-prefixed symbols are always treated as indices; this set catches the
# bare spellings the watchlist uses. A parameter, never a hardcoded truth.
DEFAULT_INDEX_SYMBOLS = frozenset({"VIX"})

# Cheap, always-listed symbol used by probe_capabilities().
PROBE_SYMBOL = "SPY"
PROBE_HISTORY_DAYS = 5

# Statement blocks Massive nests under ``financials`` (verified against the
# AAPL sample, 2026-08-19). Flattened keys are "<block>.<field>" so a consumer
# always knows which statement a number came from; an unexpected extra block
# is carried through too — the provider's vocabulary, not ours.
FINANCIALS_TIMEFRAMES = ("quarterly", "annual", "ttm")
#: Massive pages financials at 100 rows; the platform never needs more than a
#: few years of quarters in one call.
FINANCIALS_MAX_LIMIT = 100

# spread_pct assigned to a contract that has a midpoint but NO two-sided
# bid/ask market: (2*mid - 0) / mid — the widest spread a market centred on
# mid can have (bid 0, ask 2*mid). Deliberately the CONSERVATIVE unknown: it
# can only make the §9 liquidity filter reject the contract; an unquoted
# contract can never look tradeable. (A 0.0 here would fabricate perfect
# liquidity — exactly the kind of invented number this platform refuses.)
UNQUOTED_SPREAD_PCT = 2.0
#: Quotes-less plans price chain rows from the snapshot's day bar; a bar
#: older than this many days is an expired session, not a usable price
#: (weekend/holiday gaps stay inside the window).
DAY_BAR_MAX_AGE_DAYS = 5

_REDACTED = "***redacted***"

# OCC-style option ticker, e.g. "O:AAPL211022C000150000":
# root, YYMMDD expiry, C/P right, strike in thousandths zero-padded to 8
# (Massive doc examples pad to 9; the unit is thousandths either way).
_OCC_TICKER_RE = re.compile(
    r"^O:(?P<root>.+?)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8,9})$"
)


def _parse_occ_ticker(ticker: str) -> tuple[date, str, float] | None:
    """Parse an ``O:``-prefixed OCC option ticker to (expiry, right, strike).

    Returns None when the ticker does not match the OCC layout — the caller
    then relies on the explicit ``details`` fields instead of guessing.
    """
    match = _OCC_TICKER_RE.match(ticker or "")
    if match is None:
        return None
    raw = match.group("expiry")
    try:
        expiry = date(2000 + int(raw[0:2]), int(raw[2:4]), int(raw[4:6]))
    except ValueError:
        return None
    strike = int(match.group("strike")) / 1000.0
    return expiry, match.group("right"), strike


def _normalise_right(value: object) -> str | None:
    """A contract right as ``"C"``/``"P"``; None when it is neither.

    Accepts both vocabularies in play: the OCC letter the ticker carries and
    the ``"call"``/``"put"`` words the reference endpoint spells out. None is
    returned for anything else (including ``None`` itself) so callers can tell
    "the row did not say" from "the row said put" — defaulting to a call would
    price the wrong leg of a straddle.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if text in ("C", "CALL"):
        return "C"
    if text in ("P", "PUT"):
        return "P"
    return None


def _as_float(value: object) -> float | None:
    """A JSON number as float; None for anything else (bool included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ts_from_unix_ns(value: object) -> datetime | None:
    """A unix-nanosecond timestamp as an aware UTC datetime; None if unusable."""
    ns = _as_float(value)
    if ns is None or ns <= 0:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _ts_from_unix_ms(value: object) -> datetime | None:
    """A unix-MILLISECOND timestamp as an aware UTC datetime; None if unusable.

    Separate from :func:`_ts_from_unix_ns` on purpose: the aggregates endpoint
    stamps ``t`` in milliseconds while the snapshot endpoints use nanoseconds,
    and reading one with the other's scale would place a 2024 bar in 1970 or
    the far future — a wrong number rather than a missing one.
    """
    ms = _as_float(value)
    if ms is None or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _parse_iso_date(value: object) -> date | None:
    """An ISO ``YYYY-MM-DD`` string as a date; None for anything unusable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_iso_instant(value: object) -> datetime | None:
    """An ISO-8601 instant as an aware UTC datetime; None if unusable.

    Massive stamps ``acceptance_datetime`` as ``"2024-08-02T06:01:36Z"`` —
    sometimes with sub-second digits beyond microsecond precision, which
    ``datetime.fromisoformat`` rejects, so the fraction is trimmed. A naive
    string (no offset) is treated as UTC: Massive documents these instants as
    UTC, and inventing a local zone would shift the as-of key.
    """
    if not isinstance(value, str) or not value:
        return None
    raw = value.strip().replace("Z", "+00:00")
    if "." in raw:
        head, _, tail = raw.partition(".")
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        raw = f"{head}.{digits[:6]}{rest}" if digits else head + rest
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_option_day_bar(option_ticker: str, row: dict) -> Bar | None:
    """One option ``/v2/aggs`` daily row as a :class:`Bar`, or None.

    ``t`` is unix MILLISECONDS at the session start, and its US/Eastern date is
    the contract's trading date — the same convention
    :meth:`MassiveProvider.get_daily_bars` uses for stocks, so option and stock
    bars for the same session share a key and can be lined up without a
    timezone reinterpretation in between.

    Any missing OHLC field skips the row rather than defaulting it: a
    fabricated premium is the one number the §36 straddle must never contain.
    Volume alone is tolerated as 0.0 when absent, matching the daily stock
    adapter — a contract can legitimately have a bar with no reported volume,
    and volume is not what the straddle is priced from.
    """
    ts_ms = _as_float(row.get("t"))
    o = _as_float(row.get("o"))
    h = _as_float(row.get("h"))
    low = _as_float(row.get("l"))
    c = _as_float(row.get("c"))
    if None in (ts_ms, o, h, low, c):
        logger.warning(
            "Massive option aggregates: skipping bar with missing fields for %s",
            option_ticker,
        )
        return None
    return Bar(
        ts=datetime.fromtimestamp(ts_ms / 1000.0, tz=EASTERN).date(),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=_as_float(row.get("v")) or 0.0,
    )


def _parse_intraday_agg(symbol: str, row: dict) -> IntradayBar | None:
    """One aggregates ``results[]`` row as an :class:`IntradayBar`, or None.

    Volume is REQUIRED. The daily adapter tolerates a missing ``v`` (it
    defaults to 0.0) because nothing downstream compares daily volumes;
    intraday volume feeds event-replay comparisons, where a fabricated zero
    would read as "nobody traded that minute" and quietly skew an average.
    """
    ts = _ts_from_unix_ms(row.get("t"))
    o = _as_float(row.get("o"))
    h = _as_float(row.get("h"))
    low = _as_float(row.get("l"))
    c = _as_float(row.get("c"))
    v = _as_float(row.get("v"))
    if ts is None or None in (o, h, low, c, v) or c <= 0 or v < 0:
        logger.debug("Massive %s intraday bar skipped: incomplete row %r", symbol, row)
        return None
    return IntradayBar(ts=ts, open=o, high=h, low=low, close=c, volume=int(round(v)))


def _parse_news_row(row: object) -> NewsArticle | None:
    """One ``results[]`` news row as a :class:`NewsArticle`, or None if unusable.

    Shared by :meth:`MassiveProvider.get_news` and
    :meth:`MassiveProvider.get_news_window` so the recency feed and the
    windowed feed normalise IDENTICALLY — the same article fetched two ways
    must produce the same ``source_id``, or dedup downstream would keep both
    copies.

    Rows missing the fields that make an article REFERENCEABLE (id, title,
    URL, timestamp) return None and are skipped with a warning, never patched:
    an uncitable article cannot ground a recommendation.
    """
    if not isinstance(row, dict):
        return None
    source_id = row.get("id")
    title = row.get("title")
    url = row.get("article_url")
    published_raw = row.get("published_utc")
    if not (
        isinstance(source_id, str)
        and source_id
        and isinstance(title, str)
        and title
        and isinstance(url, str)
        and url
        and isinstance(published_raw, str)
    ):
        logger.warning("Skipping Massive news row missing id/title/url/timestamp")
        return None
    try:
        published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Skipping Massive news row with bad timestamp")
        return None
    publisher = ""
    pub = row.get("publisher")
    if isinstance(pub, dict) and isinstance(pub.get("name"), str):
        publisher = pub["name"]
    tickers_raw = row.get("tickers")
    tickers = tuple(
        t for t in tickers_raw if isinstance(t, str) and t
    ) if isinstance(tickers_raw, list) else ()
    description = (
        row["description"] if isinstance(row.get("description"), str) else ""
    )
    return NewsArticle(
        source_id=source_id,
        title=title,
        publisher=publisher,
        published_at=published_at,
        url=url,
        tickers=tickers,
        description=description,
    )


def _retry_after_seconds(response: httpx.Response, default: float) -> float:
    """Seconds to wait per the Retry-After header, else `default`."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _backoff_delay(
    response: httpx.Response,
    attempt: int,
    *,
    first_delay: float,
    ladder: Sequence[float] = RATE_LIMIT_BACKOFF_SECONDS,
    cap: float = RATE_LIMIT_MAX_DELAY_SECONDS,
) -> float:
    """How long to wait before retry number ``attempt`` (1-based).

    The server's ``Retry-After`` wins when it sends one — it is the vendor
    stating when the window reopens, and guessing longer wastes the caller's
    time while guessing shorter earns another 429. Otherwise the wait walks
    the fixed ladder, whose first rung is ``first_delay`` so the constructor
    parameter still governs the common case. Both forms are capped: an
    unbounded sleep inside a request handler is a hang, not patience.
    """
    if attempt <= 1:
        fallback = first_delay
    else:
        index = min(attempt, len(ladder)) - 1
        fallback = ladder[index]
    return min(max(0.0, _retry_after_seconds(response, fallback)), cap)


class MassiveProvider:
    """MarketDataProvider backed by the Massive REST API. Real data only."""

    def __init__(
        self,
        api_key: str,
        base_url: str = MASSIVE_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retry_after_default_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        max_rate_limit_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
        max_chain_pages: int = DEFAULT_MAX_CHAIN_PAGES,
        max_intraday_pages: int = DEFAULT_MAX_INTRADAY_PAGES,
        max_news_pages: int = DEFAULT_MAX_NEWS_PAGES,
        max_contract_pages: int = DEFAULT_MAX_CONTRACT_PAGES,
        max_option_bar_pages: int = DEFAULT_MAX_OPTION_BAR_PAGES,
        index_symbols: frozenset[str] = DEFAULT_INDEX_SYMBOLS,
    ) -> None:
        """`transport` is injectable so tests can mock the network (httpx.MockTransport).

        Raises :class:`MarketDataError` when `api_key` is missing/blank —
        mirroring the LLM providers, the adapter can never fire keyless, and
        there is NO synthetic fallback to fall back to.
        """
        if not api_key or not api_key.strip():
            raise MarketDataError(
                "MassiveProvider requires a non-empty API key (MASSIVE_API_KEY / "
                "settings.massive_api_key); Massive is the only market data "
                "source and there is NO synthetic fallback"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_after_default_seconds = retry_after_default_seconds
        self.max_rate_limit_retries = max(0, int(max_rate_limit_retries))
        self.max_chain_pages = max_chain_pages
        self.max_intraday_pages = max_intraday_pages
        self.max_news_pages = max_news_pages
        self.max_contract_pages = max_contract_pages
        self.max_option_bar_pages = max_option_bar_pages
        self.index_symbols = frozenset(s.upper() for s in index_symbols)
        self._transport = transport
        # Header-first auth; flips to "query" only after the server rejects
        # header auth AND accepts the apiKey query parameter (see _request).
        self._auth_mode = "header"

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _send(self, url: str, params: dict | None) -> httpx.Response:
        """One GET with the current auth form. The key is never logged."""
        request_url = httpx.URL(url)
        if params:
            request_url = request_url.copy_merge_params(
                {k: str(v) for k, v in params.items()}
            )
        headers: dict[str, str] = {}
        if self._auth_mode == "header":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            request_url = request_url.copy_set_param("apiKey", self.api_key)
        # Log the path only — the query string could carry the apiKey.
        logger.debug(
            "Massive GET %s (auth=%s key=%s)",
            request_url.path, self._auth_mode, _REDACTED,
        )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                return client.get(request_url, headers=headers)
        except httpx.HTTPError as exc:
            raise MarketDataError(f"Massive API request failed: {exc!r}") from exc

    def _request(
        self, path: str, params: dict | None = None, *, allow_404: bool = False
    ) -> httpx.Response:
        """One Massive call with the documented failure taxonomy.

        - 401 on header auth -> ONE fallback attempt with the ``apiKey`` query
          parameter (Polygon-heritage deployments accept either); a 401 on
          both forms raises :class:`MarketDataError` naming MASSIVE_API_KEY.
        - 429 -> up to ``max_rate_limit_retries`` retries with a growing
          wait (Retry-After when the server sends it, else
          ``retry_after_default_seconds`` then the 5s/12s/25s ladder, capped
          at ``RATE_LIMIT_MAX_DELAY_SECONDS``), then :class:`MarketDataError`.
          One retry was not enough: a history backfill fires four option-bar
          fetches per event back-to-back, and the vendor's window stays shut
          across several of them.
        - 403 -> :class:`CapabilityNotAvailable` naming the endpoint and the
          likely plan limitation (§16 capability detection).
        - 404 is returned to the caller when `allow_404` (an unknown symbol is
          an honest absence, not a fault); any other >= 400 raises.

        `path` may be a full URL (option-chain ``next_url`` pagination).
        """
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        endpoint = httpx.URL(url).path

        response = self._send(url, params)
        if response.status_code == 401 and self._auth_mode == "header":
            logger.debug(
                "Massive rejected header auth (HTTP 401) for %s; retrying once "
                "with the apiKey query parameter", endpoint,
            )
            self._auth_mode = "query"
            response = self._send(url, params)
            if response.status_code == 401:
                self._auth_mode = "header"  # neither form worked; stay header-first

        attempt = 0
        while response.status_code == 429 and attempt < self.max_rate_limit_retries:
            attempt += 1
            delay = _backoff_delay(
                response, attempt, first_delay=self.retry_after_default_seconds
            )
            logger.warning(
                "Massive rate limited (HTTP 429) on %s; retry %d/%d in %.1fs",
                endpoint, attempt, self.max_rate_limit_retries, delay,
            )
            if delay > 0:
                time.sleep(delay)
            response = self._send(url, params)
        if response.status_code == 429:
            # The message shape is unchanged apart from the retry count: the
            # caller (and the operator reading the log) needs "429 persisted",
            # and how hard the adapter tried before giving up.
            raise MarketDataError(
                f"Massive rate limit (HTTP 429) persisted after "
                f"{self.max_rate_limit_retries} retries for {endpoint}"
            )

        if response.status_code == 401:
            raise MarketDataError(
                f"Massive rejected the API key (HTTP 401) for {endpoint} — "
                "check MASSIVE_API_KEY (the key itself is never logged or echoed)"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"Massive returned HTTP 403 for {endpoint}: the configured "
                "plan/subscription does not include this endpoint. Upgrade the "
                "Massive plan to enable it — there is NO synthetic fallback: "
                f"{response.text[:300]}"
            )
        if response.status_code == 404 and allow_404:
            return response
        if response.status_code >= 400:
            raise MarketDataError(
                f"Massive API returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketDataError(
                f"Massive API returned a non-JSON body: {response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise MarketDataError(
                f"Massive API returned an unexpected payload shape: "
                f"{type(payload).__name__}"
            )
        return payload

    # ------------------------------------------------------------------
    # Daily bars
    # ------------------------------------------------------------------

    def get_daily_bars(self, symbol: str, days: int) -> list[Bar]:
        """Last `days` daily bars for `symbol`, oldest first — Massive data only.

        ``GET /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}`` with a
        ``days * CALENDAR_BUFFER`` calendar window (weekends/holidays), then
        trimmed to the LAST `days` bars. Empty results return ``[]`` — callers
        treat that as honest no-data, never a cue to synthesize history.
        """
        if days <= 0:
            return []
        to_date = datetime.now(EASTERN).date()
        from_date = to_date - timedelta(days=math.ceil(days * CALENDAR_BUFFER))
        payload = self._json(
            self._request(
                f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date.isoformat()}"
                f"/{to_date.isoformat()}",
                params={
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": min(50_000, max(days * 2, 10)),
                },
            )
        )
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return []  # a listed-but-quiet symbol: honestly no bars

        bars: list[Bar] = []
        for row in results:
            if not isinstance(row, dict):
                logger.warning("Massive aggregates: skipping malformed row for %s", symbol)
                continue
            ts_ms = _as_float(row.get("t"))
            o = _as_float(row.get("o"))
            h = _as_float(row.get("h"))
            low = _as_float(row.get("l"))
            c = _as_float(row.get("c"))
            if None in (ts_ms, o, h, low, c):
                logger.warning(
                    "Massive aggregates: skipping bar with missing fields for %s", symbol
                )
                continue
            # t is unix ms at the session start; its US/Eastern date is the
            # trading date of the bar.
            bar_date = datetime.fromtimestamp(ts_ms / 1000.0, tz=EASTERN).date()
            bars.append(
                Bar(
                    ts=bar_date,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=_as_float(row.get("v")) or 0.0,
                )
            )
        bars.sort(key=lambda b: b.ts)  # asc requested; sorted defensively
        return bars[-days:]

    # ------------------------------------------------------------------
    # Intraday bars (Phase C event replay)
    # ------------------------------------------------------------------

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
    ) -> list[IntradayBar]:
        """Intraday bars over ``[start, end]``, ascending and de-duplicated.

        ``GET /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/
        {to}?adjusted=true&sort=asc&limit=50000`` following ``next_url`` — the
        SAME pagination the option chain uses, so the cursor handling has one
        shape in this adapter rather than two.

        The window is expressed to the endpoint as UNIX MILLISECONDS rather
        than dates: the aggs range accepts either, and a date would round the
        window to whole Eastern sessions, which is exactly wrong for an event
        window that starts at 04:00 ET on one day and ends at 20:00 ET on the
        next. Milliseconds keep the caller's instants intact.

        Massive's ``t`` is the bar's START instant in unix ms, which is what
        :class:`IntradayBar.ts` carries (aware UTC). Rows outside the
        requested window are dropped — a paginating server may overshoot, and
        a bar outside the window the caller asked for is not theirs to
        interpret.

        A plan without the aggs endpoint answers HTTP 403, which surfaces as
        :class:`CapabilityNotAvailable` naming the endpoint (§16) — never
        empty bars, which would read as "the market was closed".
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )
        try:
            multiplier, timespan = INTRADAY_TIMEFRAMES[timeframe]
        except KeyError:
            known = ", ".join(sorted(INTRADAY_TIMEFRAMES))
            raise CapabilityNotAvailable(
                f"Massive intraday bars: timeframe {timeframe!r} has no "
                f"aggregates mapping on this adapter (known: {known}). Bars at "
                "a different resolution will NOT be substituted."
            ) from None

        from_ms = int(start_utc.timestamp() * 1000)
        to_ms = int(end_utc.timestamp() * 1000)
        url: str | None = (
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
            f"{from_ms}/{to_ms}"
        )
        params: dict | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": INTRADAY_PAGE_LIMIT,
        }
        # ts -> bar: de-duplication across pages, FIRST parse wins (a repeated
        # ts on a page boundary is a pagination artefact, not two minutes).
        by_ts: dict[datetime, IntradayBar] = {}
        pages = 0
        while url is not None and pages < self.max_intraday_pages:
            payload = self._json(self._request(url, params=params, allow_404=True))
            pages += 1
            results = payload.get("results")
            for row in results if isinstance(results, list) else []:
                if not isinstance(row, dict):
                    continue
                bar = _parse_intraday_agg(symbol, row)
                if bar is None:
                    continue
                if bar.ts < start_utc or bar.ts > end_utc:
                    continue  # a server overshoot, not the caller's window
                by_ts.setdefault(bar.ts, bar)
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None  # next_url already carries the cursor and limit
        if url is not None:
            logger.warning(
                "Massive intraday bars for %s truncated at %d pages "
                "(max_intraday_pages=%d); more bars exist in the window",
                symbol, pages, self.max_intraday_pages,
            )
        return [by_ts[ts] for ts in sorted(by_ts)]

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------

    def _is_index(self, symbol: str) -> bool:
        return symbol.startswith("^") or symbol.upper() in self.index_symbols

    def _stock_quote(self, symbol: str) -> Quote | None:
        """One stock/ETF quote from the single-ticker snapshot, or None.

        None (with a WARNING naming the symbol and why) when the snapshot does
        not know the symbol or carries no usable price — the platform renders
        a missing quote as absent, never as an invented number.
        """
        response = self._request(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
            allow_404=True,
        )
        if response.status_code == 404:
            logger.warning(
                "No Massive quote for %r: the stocks snapshot does not know it "
                "(HTTP 404); rendering it as absent", symbol,
            )
            return None
        payload = self._json(response)
        snapshot = payload.get("ticker")
        if not isinstance(snapshot, dict):
            logger.warning(
                "No Massive quote for %r: snapshot response carried no ticker "
                "data; rendering it as absent", symbol,
            )
            return None

        last_trade = snapshot.get("lastTrade") or {}
        day = snapshot.get("day") or {}
        prev_day = snapshot.get("prevDay") or {}

        # Price: the last trade, else today's session close. Both are Massive
        # numbers; if neither exists there is no price to report.
        price = _as_float(last_trade.get("p"))
        if price is None or price <= 0:
            price = _as_float(day.get("c"))
        if price is None or price <= 0:
            logger.warning(
                "No Massive quote for %r: snapshot has no last trade and no "
                "session close; rendering it as absent", symbol,
            )
            return None

        # Change: the snapshot's own todaysChangePerc, else derived from the
        # previous close — arithmetic on Massive numbers, never a guess.
        change_pct = _as_float(snapshot.get("todaysChangePerc"))
        if change_pct is None:
            prev_close = _as_float(prev_day.get("c"))
            if prev_close is None or prev_close <= 0:
                logger.warning(
                    "No Massive quote for %r: snapshot has neither "
                    "todaysChangePerc nor a previous close; rendering it as "
                    "absent", symbol,
                )
                return None
            change_pct = (price / prev_close - 1.0) * 100.0

        # `updated` / lastTrade.t are unix ns; fall back to the fetch time
        # (an honest "as of now", mirroring the Alpaca timestamp policy).
        ts = (
            _ts_from_unix_ns(snapshot.get("updated"))
            or _ts_from_unix_ns(last_trade.get("t"))
            or datetime.now(timezone.utc)
        )
        return Quote(symbol=symbol, price=price, change_pct=change_pct, ts=ts)

    def _index_quote(self, symbol: str) -> Quote | None:
        """One index quote (VIX, ^SPX, ...) via ``GET /v3/snapshot/indices``.

        Indices are not served by the stocks snapshot; Massive spells them
        ``I:<NAME>`` on the indices snapshot endpoint (verified:
        https://massive.com/docs/rest/indices/snapshots/indices-snapshot.md).
        """
        index_ticker = f"I:{symbol.lstrip('^').upper()}"
        payload = self._json(
            self._request("/v3/snapshot/indices", params={"ticker": index_ticker})
        )
        results = payload.get("results")
        rows = [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []
        if not rows:
            logger.warning(
                "No Massive quote for %r: the indices snapshot returned no "
                "result for %s; rendering it as absent", symbol, index_ticker,
            )
            return None
        row = rows[0]
        if row.get("error"):
            logger.warning(
                "No Massive quote for %r: indices snapshot reports %s (%s); "
                "rendering it as absent",
                symbol, row.get("error"), row.get("message", ""),
            )
            return None

        price = _as_float(row.get("value"))
        if price is None or price <= 0:
            logger.warning(
                "No Massive quote for %r: indices snapshot carries no value; "
                "rendering it as absent", symbol,
            )
            return None
        session = row.get("session") or {}
        change_pct = _as_float(session.get("change_percent"))
        if change_pct is None:
            prev_close = _as_float(session.get("previous_close"))
            if prev_close is None or prev_close <= 0:
                logger.warning(
                    "No Massive quote for %r: indices snapshot has neither "
                    "change_percent nor previous_close; rendering it as absent",
                    symbol,
                )
                return None
            change_pct = (price / prev_close - 1.0) * 100.0
        ts = _ts_from_unix_ns(row.get("last_updated")) or datetime.now(timezone.utc)
        return Quote(symbol=symbol, price=price, change_pct=change_pct, ts=ts)

    def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Current quotes; symbols Massive cannot quote are honestly ABSENT.

        Stocks/ETFs use the single-ticker snapshot; index symbols ("^"-prefixed
        or in ``index_symbols``, e.g. VIX) use the indices snapshot in the
        ``I:`` ticker form. Per-symbol absences (unknown symbol, no price, an
        asset class the plan does not include) are skipped with a WARNING so
        the quotes that ARE real still flow. If EVERY symbol failed on a
        plan-gated endpoint the :class:`CapabilityNotAvailable` is re-raised —
        an all-403 batch is a capability problem, not an empty market.
        """
        quotes: list[Quote] = []
        capability_errors: list[CapabilityNotAvailable] = []
        for symbol in symbols:
            try:
                quote = (
                    self._index_quote(symbol)
                    if self._is_index(symbol)
                    else self._stock_quote(symbol)
                )
            except CapabilityNotAvailable as exc:
                logger.warning(
                    "No Massive quote for %r: %s; rendering it as absent",
                    symbol, exc,
                )
                capability_errors.append(exc)
                continue
            if quote is not None:
                quotes.append(quote)
        if symbols and not quotes and capability_errors:
            raise capability_errors[0]
        return quotes

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------

    def get_option_chain(
        self, symbol: str, spot: float, as_of: date
    ) -> list[OptionQuote]:
        """The CURRENT option chain snapshot for `symbol` (plan §9).

        ``GET /v3/snapshot/options/{symbol}?limit=250`` following ``next_url``
        pagination (capped at ``max_chain_pages``). The endpoint serves
        current state ONLY, so an `as_of` that is not today raises
        :class:`MarketDataError` — a historical chain we do not have is
        reported as such, never approximated.

        `spot` (the caller's last stored close) is accepted for interface
        compatibility but nothing is computed from it: every price, greek and
        IV in the rows is Massive's own, and the response's fresher
        ``underlying_asset.price`` supersedes the passed spot (their drift is
        logged at debug). Contracts without a usable quote (no bid/ask pair
        and no midpoint, or a non-positive mid) and contracts missing any
        greek or the IV are SKIPPED with a debug log: the §9 selector needs
        real quotes and real greeks, and zero-filling would fabricate them.
        """
        today = datetime.now(EASTERN).date()
        if as_of != today:
            raise MarketDataError(
                f"Massive option-chain snapshots are current-state only: "
                f"requested as_of={as_of.isoformat()} but today is "
                f"{today.isoformat()}. Historical chains are not served by "
                "/v3/snapshot/options and will not be approximated."
            )

        chain: list[OptionQuote] = []
        underlying_price: float | None = None
        url: str | None = f"/v3/snapshot/options/{symbol}"
        params: dict | None = {"limit": CHAIN_PAGE_LIMIT}
        pages = 0
        while url is not None and pages < self.max_chain_pages:
            payload = self._json(self._request(url, params=params))
            pages += 1
            results = payload.get("results")
            for row in results if isinstance(results, list) else []:
                if not isinstance(row, dict):
                    continue
                parsed = self._parse_chain_row(symbol, row, as_of)
                if parsed is None:
                    continue
                quote, row_underlying = parsed
                chain.append(quote)
                if row_underlying is not None:
                    underlying_price = row_underlying
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None  # next_url already carries the cursor and limit
        if url is not None:
            logger.warning(
                "Massive option chain for %s truncated at %d pages "
                "(max_chain_pages=%d); more contracts exist beyond the cap",
                symbol, pages, self.max_chain_pages,
            )

        if underlying_price is not None and spot > 0:
            # The snapshot's own underlying price is fresher than the caller's
            # stored close; surface the drift rather than silently differing.
            logger.debug(
                "Massive %s chain: underlying_asset.price=%.4f vs passed "
                "spot=%.4f (%.2f%% drift); the snapshot price is authoritative",
                symbol, underlying_price, spot,
                (underlying_price / spot - 1.0) * 100.0,
            )

        # Deterministic ORDER (not data): the API's paging order is not
        # contractual, and the §9 selector breaks ties by chain order.
        chain.sort(key=lambda q: (q.expiry, q.strike, q.right))
        return chain

    def _parse_chain_row(
        self, symbol: str, row: dict, as_of: date
    ) -> tuple[OptionQuote, float | None] | None:
        """One snapshot row -> (OptionQuote, underlying price) or None (skip).

        The OCC ticker (``O:AAPL211022C000150000``) is parsed and
        CROSS-CHECKED against the explicit ``details`` fields; a row where the
        two disagree is skipped with a WARNING — inconsistent source data is
        never guessed into a contract.
        """
        details = row.get("details") or {}
        ticker = str(details.get("ticker", ""))
        occ = _parse_occ_ticker(ticker)

        # Explicit details fields (authoritative when present).
        expiry: date | None = None
        raw_expiry = details.get("expiration_date")
        if isinstance(raw_expiry, str) and raw_expiry:
            try:
                expiry = date.fromisoformat(raw_expiry)
            except ValueError:
                expiry = None
        contract_type = str(details.get("contract_type", "")).lower()
        right = {"call": "C", "put": "P"}.get(contract_type)
        strike = _as_float(details.get("strike_price"))

        if occ is not None:
            occ_expiry, occ_right, occ_strike = occ
            mismatch = (
                (expiry is not None and expiry != occ_expiry)
                or (right is not None and right != occ_right)
                or (strike is not None and abs(strike - occ_strike) > 1e-6)
            )
            if mismatch:
                logger.warning(
                    "Massive chain row for %s skipped: OCC ticker %r "
                    "(expiry=%s right=%s strike=%s) disagrees with details "
                    "(expiry=%s right=%s strike=%s); refusing to guess",
                    symbol, ticker, occ_expiry, occ_right, occ_strike,
                    expiry, right, strike,
                )
                return None
            expiry = expiry or occ_expiry
            right = right or occ_right
            strike = strike if strike is not None else occ_strike

        if expiry is None or right is None or strike is None or strike <= 0:
            logger.debug(
                "Massive chain row for %s skipped: contract identity "
                "incomplete (ticker=%r)", symbol, ticker,
            )
            return None
        dte = (expiry - as_of).days
        if dte < 0:
            logger.debug(
                "Massive chain row %r skipped: already expired (%s)", ticker, expiry
            )
            return None

        # Quote: (bid+ask)/2 when a two-sided market exists, else the
        # reported midpoint. Plans without the NBBO quotes entitlement (the
        # snapshot then carries NO last_quote block at all — observed live on
        # a quotes-less options tier) fall back to the DAY bar's close: a
        # REAL traded session price, never an invention. The spread is then
        # recorded as the conservative worst case, so unknown bid/ask quality
        # can only ever REJECT in the §9 selector — the chain stays visible
        # with real greeks/IV/OI, and selection stays honest about quotes.
        last_quote = row.get("last_quote") or {}
        day = row.get("day") or {}
        bid = _as_float(last_quote.get("bid"))
        ask = _as_float(last_quote.get("ask"))
        midpoint = _as_float(last_quote.get("midpoint"))
        day_close = _as_float(day.get("close"))
        day_updated = _ts_from_unix_ns(day.get("last_updated"))
        price_basis = "quote"
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            spread_pct = max(0.0, (ask - bid) / mid) if mid > 0 else 0.0
        elif midpoint is not None:
            mid = midpoint
            # No two-sided market: an absent side is recorded as 0.0 ("no
            # bid"/"no offer") and the spread as the conservative worst case,
            # so unknown liquidity can only ever REJECT in the selector.
            spread_pct = UNQUOTED_SPREAD_PCT
        elif day_close is not None and day_close > 0 and (
            day_updated is not None
            and (datetime.now(timezone.utc) - day_updated).days
            <= DAY_BAR_MAX_AGE_DAYS
        ):
            mid = day_close
            spread_pct = UNQUOTED_SPREAD_PCT
            price_basis = "day_close"
            logger.debug(
                "Massive chain row %r: no last_quote on this plan — using the "
                "day-bar close %.4f (updated %s) with worst-case spread",
                ticker, day_close, day_updated.isoformat(),
            )
        else:
            logger.debug(
                "Massive chain row %r skipped: no bid/ask, no midpoint and no "
                "fresh day close (unquotable — not a selector candidate)",
                ticker,
            )
            return None
        if mid <= 0:
            logger.debug(
                "Massive chain row %r skipped: non-positive mid %.4f "
                "(unquotable — not a selector candidate)", ticker, mid,
            )
            return None

        # Greeks/IV: Massive's numbers or nothing. NEVER zero-filled — a
        # fabricated greek would flow straight into §9 ranking and §13 limits.
        greeks = row.get("greeks") or {}
        delta = _as_float(greeks.get("delta"))
        gamma = _as_float(greeks.get("gamma"))
        theta = _as_float(greeks.get("theta"))
        vega = _as_float(greeks.get("vega"))
        iv = _as_float(row.get("implied_volatility"))
        if None in (delta, gamma, theta, vega, iv):
            logger.debug(
                "Massive chain row %r skipped: greeks/IV missing (the §9 "
                "selector requires them; they are never fabricated)", ticker,
            )
            return None

        last_trade = row.get("last_trade") or {}
        underlying = row.get("underlying_asset") or {}
        quote = OptionQuote(
            expiry=expiry,
            dte=dte,
            strike=strike,
            right=right,
            bid=bid if bid is not None else 0.0,
            ask=ask if ask is not None else 0.0,
            mid=mid,
            spread_pct=spread_pct,
            last=_as_float(last_trade.get("price")),
            # Absent volume/OI read as 0 — "none reported", the conservative
            # value the §9 liquidity floors can only ever reject on.
            volume=int(_as_float(day.get("volume")) or 0),
            open_interest=int(_as_float(row.get("open_interest")) or 0),
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            price_basis=price_basis,
        )
        return quote, _as_float(underlying.get("price"))


    # ------------------------------------------------------------------
    # News (Phase 8 ingestion — REAL articles only)
    # ------------------------------------------------------------------

    def get_news(
        self,
        limit: int = 50,
        published_after: datetime | None = None,
    ) -> list[NewsArticle]:
        """Most recent news articles (``GET /v2/reference/news``), newest first.

        Every article is the provider's data verbatim — id, title, publisher,
        timestamp, URL, tickers, description. Rows missing the fields that
        make an article REFERENCEABLE (id, title, URL, timestamp) are skipped
        with a warning, never patched: an uncitable article cannot ground a
        recommendation. 403 raises :class:`CapabilityNotAvailable` (§16 —
        the plan does not include news; there is no synthetic fallback).
        """
        params: dict[str, str] = {
            "limit": str(max(1, min(int(limit), 100))),
            "order": "desc",
            "sort": "published_utc",
        }
        if published_after is not None:
            params["published_utc.gte"] = published_after.isoformat()
        body = self._json(self._request("/v2/reference/news", params=params))
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            return []

        articles: list[NewsArticle] = []
        for row in results:
            article = _parse_news_row(row)
            if article is not None:
                articles.append(article)
        return articles

    def get_news_window(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> list[NewsArticle]:
        """Every article for `tickers` in ``[start, end]``, newest first (§21).

        ``GET /v2/reference/news?ticker=<T>&published_utc.gte=&published_utc.lte=
        &limit=1000&order=desc&sort=published_utc`` following ``next_url`` —
        the SAME cursor handling the option chain and intraday aggregates use,
        so this adapter has one pagination shape rather than three.

        ONE REQUEST PER TICKER, unlike the Alpaca adapter: Massive's ``ticker``
        parameter matches a single symbol, so a basket is a loop and the
        per-ticker result sets are MERGED here. A syndicated article tagged
        with two of the requested tickers therefore arrives twice, and the
        ``source_id`` de-duplication below is what makes it one article again
        — the merge is exactly why that de-duplication is not optional.

        `limit` caps the MERGED result, and each ticker's pagination stops
        early once the merge is already full, so a large basket cannot fan out
        into unbounded round-trips. Because the merged list is sorted newest
        first before truncation, a `limit` that bites keeps the most recent
        articles across the whole basket rather than draining it ticker by
        ticker.
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )
        symbols = list(dict.fromkeys(
            t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()
        ))
        if not symbols or limit <= 0:
            # No symbols to ask about, or no room for an answer. Answered here
            # rather than by the server, whose news endpoint without a ticker
            # returns the entire market's firehose.
            return []

        # source_id -> article: de-duplication across pages AND across the
        # per-ticker requests, FIRST parse wins.
        by_id: dict[str, NewsArticle] = {}
        for symbol in symbols:
            if len(by_id) >= limit:
                break
            url: str | None = "/v2/reference/news"
            params: dict | None = {
                "ticker": symbol,
                "published_utc.gte": start_utc.isoformat(),
                "published_utc.lte": end_utc.isoformat(),
                "limit": NEWS_PAGE_LIMIT,
                "order": "desc",
                "sort": "published_utc",
            }
            pages = 0
            while url is not None and pages < self.max_news_pages:
                payload = self._json(self._request(url, params=params))
                pages += 1
                results = payload.get("results")
                for row in results if isinstance(results, list) else []:
                    article = _parse_news_row(row)
                    if article is None:
                        continue
                    by_id.setdefault(article.source_id, article)
                if len(by_id) >= limit:
                    url = None
                    break
                next_url = payload.get("next_url")
                url = next_url if isinstance(next_url, str) and next_url else None
                params = None  # next_url already carries the cursor and limit
            if url is not None:
                logger.warning(
                    "Massive windowed news for %s truncated at %d pages "
                    "(max_news_pages=%d); more articles exist in the window",
                    symbol, pages, self.max_news_pages,
                )
        ordered = sorted(
            by_id.values(), key=lambda a: (a.published_at, a.source_id), reverse=True
        )
        return ordered[:limit]

    # ------------------------------------------------------------------
    # Fundamentals (Phase E2 — filed statements, point-in-time)
    # ------------------------------------------------------------------

    def get_financials(
        self, ticker: str, *, timeframe: str = "quarterly", limit: int = 12
    ) -> list[FinancialStatement]:
        """Filed financial statements for `ticker`, NEWEST FIRST (§28, audit §11.3).

        ``GET /vX/reference/financials?ticker=&timeframe=&limit=&order=desc&
        sort=filing_date`` — the ONE fundamentals endpoint this plan is
        entitled to (Benzinga consensus/estimates are 403 across the board,
        §33). Each result carries period bounds, ``filing_date``,
        ``acceptance_datetime`` — the point-in-time key the whole as-of
        contract rests on (§7.1/§85) — and a ``financials`` object of
        statement blocks whose fields are ``{value, unit, label, order}``.

        The blocks are FLATTENED to ``"<block>.<field>" -> float``. Only
        numeric values survive: a field whose value is a string, null or a
        nested object is SKIPPED, so a consumer that finds a key absent knows
        the filer did not report a usable number rather than reading a
        fabricated 0.0 (§44 rule 18). ``raw_fields_count`` records how many
        fields the row carried before that filter.

        Rows missing what makes a statement PLACEABLE (period start/end) are
        skipped with a warning — an undatable statement cannot be filtered
        as-of, and a guessed period would corrupt every YoY comparison. Rows
        WITHOUT ``acceptance_datetime`` are kept (the field is honestly
        ``None``): excluding them here would hide the gap, while the pure
        as-of layer excludes them with an explicit reason.

        403 raises :class:`CapabilityNotAvailable` (§16 — the plan does not
        include fundamentals; there is no synthetic fallback).
        """
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return []
        period = (timeframe or "").strip().lower()
        if period not in FINANCIALS_TIMEFRAMES:
            raise ValueError(
                f"unsupported financials timeframe {timeframe!r} — Massive serves "
                f"{', '.join(FINANCIALS_TIMEFRAMES)}"
            )
        params = {
            "ticker": symbol,
            "timeframe": period,
            "limit": str(max(1, min(int(limit), FINANCIALS_MAX_LIMIT))),
            "order": "desc",
            "sort": "filing_date",
        }
        body = self._json(self._request("/vX/reference/financials", params=params))
        results = body.get("results")
        if not isinstance(results, list):
            return []

        statements: list[FinancialStatement] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            start = _parse_iso_date(row.get("start_date"))
            end = _parse_iso_date(row.get("end_date"))
            if start is None or end is None:
                logger.warning(
                    "Skipping Massive financials row for %s missing start/end date",
                    symbol,
                )
                continue
            values, raw_count = self._flatten_financials(row.get("financials"))
            fiscal_year_raw = row.get("fiscal_year")
            fiscal_year: int | None = None
            if isinstance(fiscal_year_raw, (int, str)) and not isinstance(
                fiscal_year_raw, bool
            ):
                try:
                    fiscal_year = int(fiscal_year_raw)
                except ValueError:
                    fiscal_year = None
            statements.append(
                FinancialStatement(
                    ticker=symbol,
                    cik=row["cik"] if isinstance(row.get("cik"), str) else None,
                    timeframe=(
                        row["timeframe"]
                        if isinstance(row.get("timeframe"), str) and row["timeframe"]
                        else period
                    ),
                    fiscal_year=fiscal_year,
                    fiscal_period=(
                        row["fiscal_period"]
                        if isinstance(row.get("fiscal_period"), str)
                        else ""
                    ),
                    start_date=start,
                    end_date=end,
                    filing_date=_parse_iso_date(row.get("filing_date")),
                    acceptance_datetime=_parse_iso_instant(
                        row.get("acceptance_datetime")
                    ),
                    source_filing_url=(
                        row["source_filing_url"]
                        if isinstance(row.get("source_filing_url"), str)
                        else None
                    ),
                    values=values,
                    raw_fields_count=raw_count,
                )
            )

        # Newest first. `sort=filing_date&order=desc` already asks for that,
        # but the ordering the platform DEPENDS on is by period end — a late
        # amended filing must not reorder the fiscal series — so it is
        # enforced here rather than trusted from the wire.
        statements.sort(key=lambda s: (s.end_date, s.filing_date or s.end_date),
                        reverse=True)
        return statements

    @staticmethod
    def _flatten_financials(blocks: object) -> tuple[dict[str, float], int]:
        """``{block: {field: {value: ...}}}`` -> ``{"block.field": float}``.

        Returns the flattened numeric mapping and the count of fields seen
        BEFORE the numeric filter, so callers can tell an empty statement from
        one whose numbers were all unparseable.
        """
        values: dict[str, float] = {}
        raw_count = 0
        if not isinstance(blocks, dict):
            return values, raw_count
        for block_name, fields in blocks.items():
            if not isinstance(block_name, str) or not isinstance(fields, dict):
                continue
            for field_name, field in fields.items():
                if not isinstance(field_name, str):
                    continue
                raw_count += 1
                if not isinstance(field, dict):
                    continue
                number = _as_float(field.get("value"))
                if number is None or not math.isfinite(number):
                    continue
                values[f"{block_name}.{field_name}"] = number
        return values, raw_count

    # ------------------------------------------------------------------
    # Capability probe (§16: detect, never fall back)
    # ------------------------------------------------------------------

    def get_option_contracts(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        max_pages: int = 2,
    ) -> list[dict]:
        """Option CONTRACT REFERENCE rows for `underlying` (no quotes).

        ``GET /v3/reference/options/contracts`` — included in ALL Massive
        options plans including Basic (verified against
        https://massive.com/docs/rest/options/contracts/all-contracts.md),
        unlike the chain snapshot which needs Starter+. Returns
        ``{"ticker", "contract_type", "strike_price", "expiration_date",
        "shares_per_contract"}`` dicts, non-expired only, following
        ``next_url`` up to ``max_pages`` (1000 rows/page). Reference data
        ONLY: what contracts exist — never prices, greeks or IV.
        """
        params: dict | None = {
            "underlying_ticker": underlying,
            "expired": "false",
            "limit": 1000,
            "order": "asc",
            "sort": "expiration_date",
        }
        if expiration_gte is not None:
            params["expiration_date.gte"] = expiration_gte.isoformat()
        if expiration_lte is not None:
            params["expiration_date.lte"] = expiration_lte.isoformat()

        contracts: list[dict] = []
        url: str | None = "/v3/reference/options/contracts"
        pages = 0
        while url is not None and pages < max_pages:
            payload = self._json(self._request(url, params=params))
            pages += 1
            results = payload.get("results")
            for row in results if isinstance(results, list) else []:
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker", ""))
                contract_type = str(row.get("contract_type", "")).lower()
                strike = _as_float(row.get("strike_price"))
                raw_expiry = row.get("expiration_date")
                try:
                    expiry = (
                        date.fromisoformat(raw_expiry)
                        if isinstance(raw_expiry, str)
                        else None
                    )
                except ValueError:
                    expiry = None
                if (
                    not ticker
                    or contract_type not in ("call", "put")
                    or strike is None
                    or strike <= 0
                    or expiry is None
                ):
                    logger.debug(
                        "Massive contract row for %s skipped: incomplete "
                        "identity (ticker=%r)", underlying, row.get("ticker"),
                    )
                    continue
                contracts.append(
                    {
                        "ticker": ticker,
                        "contract_type": contract_type,
                        "strike_price": strike,
                        "expiration_date": expiry,
                        "shares_per_contract": _as_float(
                            row.get("shares_per_contract")
                        ) or 100.0,
                    }
                )
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None  # next_url carries the cursor
        return contracts

    def get_option_prev_bar(self, option_ticker: str) -> dict | None:
        """Previous-session EOD bar for one option contract.

        ``GET /v2/aggs/ticker/{optionsTicker}/prev`` — included in ALL
        Massive options plans including Basic (End-of-Day recency there;
        verified against
        https://massive.com/docs/rest/options/aggregates/previous-day-bar.md).
        Returns ``{"open","high","low","close","volume","vwap","date"}`` or
        ``None`` when the contract has no previous-session bar (an illiquid
        contract that did not trade is an honest absence, never a zero).
        """
        payload = self._json(
            self._request(
                f"/v2/aggs/ticker/{option_ticker}/prev", params={"adjusted": "true"}
            )
        )
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        row = results[0]
        if not isinstance(row, dict):
            return None
        close = _as_float(row.get("c"))
        if close is None or close <= 0:
            return None
        ts_ms = row.get("t")
        bar_date: date | None = None
        if isinstance(ts_ms, (int, float)) and ts_ms > 0:
            bar_date = datetime.fromtimestamp(
                ts_ms / 1000.0, tz=timezone.utc
            ).astimezone(EASTERN).date()
        return {
            "open": _as_float(row.get("o")),
            "high": _as_float(row.get("h")),
            "low": _as_float(row.get("l")),
            "close": close,
            "volume": _as_float(row.get("v")) or 0.0,
            "vwap": _as_float(row.get("vw")),
            "date": bar_date.isoformat() if bar_date else None,
        }

    # ------------------------------------------------------------------
    # Historical options (Phase I §36 implied move)
    # ------------------------------------------------------------------

    def list_option_contracts(
        self,
        underlying: str,
        *,
        expiration_date: date,
        as_of: date,
        right: str | None = None,
        limit: int = CONTRACTS_PAGE_LIMIT,
    ) -> list[OptionContractRef]:
        """Contracts on `underlying` for one expiry, as they were listed on `as_of`.

        ``GET /v3/reference/options/contracts?underlying_ticker=…&
        expiration_date=…&as_of=…&contract_type=…&limit=…`` following
        ``next_url`` (capped at ``max_contract_pages``). Verified live on the
        base plan.

        ``as_of`` goes ON THE WIRE — it is the whole reason this method is not
        a filter over :meth:`get_option_contracts`, whose ``expired=false``
        query answers "what is listed today". A strike listed the morning
        AFTER an earnings release must not appear in that event's pre-event
        straddle universe, and only the server knows when each contract was
        first listed.

        Rows are validated against the OCC ticker: the ``details`` fields are
        preferred, and :func:`_parse_occ_ticker` fills in whichever of
        expiry/right/strike the row omitted. A row whose ticker and fields
        DISAGREE on the expiry or right is skipped with a WARNING rather than
        being resolved by preference — a mismatched identity means one of the
        two is wrong, and pricing the wrong contract is a wrong number.
        """
        symbol = (underlying or "").strip().upper()
        if not symbol:
            raise MarketDataError(
                "list_option_contracts requires a non-empty underlying ticker"
            )
        params: dict | None = {
            "underlying_ticker": symbol,
            "expiration_date": expiration_date.isoformat(),
            "as_of": as_of.isoformat(),
            "limit": max(1, min(int(limit), CONTRACTS_PAGE_LIMIT)),
            "order": "asc",
            "sort": "strike_price",
        }
        wanted = _normalise_right(right)
        if wanted is not None:
            params["contract_type"] = "call" if wanted == "C" else "put"

        refs: list[OptionContractRef] = []
        seen: set[str] = set()
        url: str | None = "/v3/reference/options/contracts"
        pages = 0
        while url is not None and pages < self.max_contract_pages:
            payload = self._json(self._request(url, params=params, allow_404=True))
            pages += 1
            results = payload.get("results")
            for row in results if isinstance(results, list) else []:
                if not isinstance(row, dict):
                    continue
                ref = self._parse_contract_ref(symbol, row)
                if ref is None or ref.ticker in seen:
                    continue
                if wanted is not None and ref.right != wanted:
                    continue  # server filter is authoritative; belt-and-braces
                seen.add(ref.ticker)
                refs.append(ref)
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None  # next_url already carries the cursor and limit
        if url is not None:
            logger.warning(
                "Massive contract reference for %s %s truncated at %d pages "
                "(max_contract_pages=%d); strikes beyond the cap are MISSING",
                symbol, expiration_date.isoformat(), pages, self.max_contract_pages,
            )
        # Deterministic order (not data): strike then right, so the ATM pick
        # is reproducible when two rows tie.
        refs.sort(key=lambda r: (r.strike, r.right))
        return refs

    @staticmethod
    def _parse_contract_ref(underlying: str, row: dict) -> OptionContractRef | None:
        """One ``/v3/reference/options/contracts`` row as an OptionContractRef.

        None (with a log line) whenever the row cannot be identified with
        confidence: no ticker, an unusable strike, an unknown contract_type,
        or an OCC ticker that contradicts the row's own expiry/right. Skipping
        is the honest answer — a contract we cannot name is one we cannot
        price.
        """
        ticker = row.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            logger.debug("Massive contract row for %s skipped: no ticker", underlying)
            return None
        ticker = ticker.strip()
        parsed = _parse_occ_ticker(ticker)

        expiry = _parse_iso_date(row.get("expiration_date"))
        right = _normalise_right(row.get("contract_type"))
        strike = _as_float(row.get("strike_price"))

        if parsed is not None:
            occ_expiry, occ_right, occ_strike = parsed
            if expiry is not None and expiry != occ_expiry:
                logger.warning(
                    "Massive contract %s skipped: expiration_date %s contradicts "
                    "the OCC ticker (%s)", ticker, expiry, occ_expiry,
                )
                return None
            if right is not None and right != occ_right:
                logger.warning(
                    "Massive contract %s skipped: contract_type %s contradicts "
                    "the OCC ticker (%s)", ticker, right, occ_right,
                )
                return None
            expiry = expiry or occ_expiry
            right = right or occ_right
            if strike is None or strike <= 0:
                strike = occ_strike

        if expiry is None or right is None or strike is None or strike <= 0:
            logger.debug(
                "Massive contract row for %s skipped: incomplete identity "
                "(ticker=%r)", underlying, ticker,
            )
            return None
        return OptionContractRef(
            ticker=ticker,
            underlying=underlying,
            expiry=expiry,
            right=right,
            strike=strike,
        )

    def get_option_history_bars(
        self, option_ticker: str, start: date, end: date
    ) -> list[Bar]:
        """Daily bars for ONE option contract over ``[start, end]``, oldest first.

        ``GET /v2/aggs/ticker/{option_ticker}/range/1/day/{from}/{to}
        ?adjusted=true&sort=asc&limit=5000`` following ``next_url`` (capped at
        ``max_option_bar_pages``). Verified live on the base plan for
        ``O:``-prefixed contract symbols.

        ``option_ticker`` is passed through VERBATIM — it is the symbol
        :meth:`list_option_contracts` returned, and re-deriving it here from
        parts is how a strike padded to nine digits instead of eight becomes a
        silent empty result.

        A 404 (a symbol this deployment does not know) and an empty
        ``results`` both return ``[]``: the contract served no bars in the
        window. Rows missing any of ``t/o/h/l/c`` are skipped with a WARNING
        rather than zero-filled — an option premium of 0.0 that we invented
        would make a straddle look free.
        """
        ticker = (option_ticker or "").strip()
        if not ticker:
            raise MarketDataError(
                "get_option_history_bars requires a non-empty option ticker"
            )
        if end < start:
            raise ValueError(
                f"get_option_history_bars window is reversed: start={start} > "
                f"end={end}; an empty window is expressed as start == end"
            )

        bars: list[Bar] = []
        seen_dates: set[date] = set()
        url: str | None = (
            f"/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        params: dict | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": OPTION_BAR_PAGE_LIMIT,
        }
        pages = 0
        while url is not None and pages < self.max_option_bar_pages:
            payload = self._json(self._request(url, params=params, allow_404=True))
            pages += 1
            results = payload.get("results")
            for row in results if isinstance(results, list) else []:
                if not isinstance(row, dict):
                    logger.warning(
                        "Massive option aggregates: skipping malformed row for %s",
                        ticker,
                    )
                    continue
                bar = _parse_option_day_bar(ticker, row)
                if bar is None or bar.ts in seen_dates:
                    continue
                seen_dates.add(bar.ts)
                bars.append(bar)
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None  # next_url already carries the cursor and limit
        if url is not None:
            logger.warning(
                "Massive option bars for %s truncated at %d pages "
                "(max_option_bar_pages=%d); later sessions are MISSING",
                ticker, pages, self.max_option_bar_pages,
            )
        bars.sort(key=lambda b: b.ts)  # asc requested; sorted defensively
        return bars

    def probe_capabilities(self) -> dict[str, bool | str]:
        """Cheap live probes of the capabilities this platform needs.

        Returns ``{"stock_history": ..., "stock_realtime": ..., "option_chain":
        ..., "option_contracts": ..., "news": ..., "financials": ...}`` where
        each value is
        ``True`` (works), ``False`` (HTTP 403 — the
        plan does not include it; the hint is logged as a WARNING) or the
        error string (network/API fault — availability unknown). §16: the
        answer to a missing capability is this report, never synthetic data.

        ``option_chain`` (the Starter+ snapshot with quotes/greeks/IV) and
        ``option_contracts`` (the Basic-tier reference list) are probed
        SEPARATELY: a Basic options plan answers chain=False,
        contracts=True — and the platform serves the honest EOD reference
        view off the latter.
        """
        to_date = datetime.now(EASTERN).date()
        from_date = to_date - timedelta(days=PROBE_HISTORY_DAYS)
        probes: dict[str, tuple[str, dict | None]] = {
            "stock_history": (
                f"/v2/aggs/ticker/{PROBE_SYMBOL}/range/1/day/"
                f"{from_date.isoformat()}/{to_date.isoformat()}",
                {"adjusted": "true", "sort": "asc", "limit": PROBE_HISTORY_DAYS},
            ),
            "stock_realtime": (
                f"/v2/snapshot/locale/us/markets/stocks/tickers/{PROBE_SYMBOL}",
                None,
            ),
            "option_chain": (
                f"/v3/snapshot/options/{PROBE_SYMBOL}",
                {"limit": 1},
            ),
            "option_contracts": (
                "/v3/reference/options/contracts",
                {"underlying_ticker": PROBE_SYMBOL, "limit": 1},
            ),
            "news": ("/v2/reference/news", {"limit": "1"}),
            "financials": (
                "/vX/reference/financials",
                {"ticker": PROBE_SYMBOL, "limit": "1"},
            ),
        }
        report: dict[str, bool | str] = {}
        for name, (path, params) in probes.items():
            try:
                self._request(path, params=params)
            except CapabilityNotAvailable as exc:
                logger.warning("Massive capability %r unavailable: %s", name, exc)
                report[name] = False
            except MarketDataError as exc:
                report[name] = str(exc)
            else:
                report[name] = True
        return report
