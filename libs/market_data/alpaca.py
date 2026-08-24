"""Alpaca-backed market data provider (data_source.md upgrade).

ALPACA IS THE AUTHORITATIVE SOURCE FOR MARKET DATA on this platform
(data_source.md §1/§2): stocks, options and news all come from the Alpaca
Market Data API; Massive is reserved for company fundamentals only. The
broker/trading side already speaks Alpaca (libs/broker/alpaca.py) — this
module is the MARKET DATA half, using the SAME account credentials
(``settings.alpaca_api_key_id`` / ``alpaca_api_secret_key``), which work
against ``data.alpaca.markets`` per Alpaca's docs and were verified live.

NO SYNTHETIC FALLBACK, EVER (§44 rule 18). Every number returned here was
reported by Alpaca. When Alpaca cannot answer — network fault, bad key,
plan-gated endpoint, unknown symbol — the answer is an explicit error or an
honest absence, never an invented value:

  - transport/HTTP faults, bad payloads -> :class:`MarketDataError`;
  - HTTP 403 (endpoint not in the subscription) ->
    :class:`CapabilityNotAvailable` naming the endpoint;
  - a symbol Alpaca does not serve (e.g. the VIX index — Alpaca has no
    index feed) -> that symbol is SKIPPED with a WARNING (honest absence);
  - an option contract without a usable NBBO quote or without greeks/IV ->
    SKIPPED with a debug log (the §9 selector needs real quotes and real
    greeks; zero-filling would fabricate them).

Endpoints (each verified LIVE against a paid Alpaca market-data subscription
on 2026-08-13, and against https://docs.alpaca.markets/reference). NOTE: some
of these — full option chains and historical minute bars in particular —
require a paid data plan; the free tier will refuse them, and the adapter
surfaces that refusal rather than substituting anything:

  - Daily bars:      ``GET /v2/stocks/{symbol}/bars`` on data.alpaca.markets
                     (timeframe=1Day, adjustment=split, RFC-3339 ``t``,
                     o/h/l/c/v/n/vw; next_page_token pagination).
  - Stock quotes:    ``GET /v2/stocks/snapshots?symbols=...`` — one call for
                     all symbols; latestTrade.p + prevDailyBar.c give price
                     and day change.
  - Option chain:    ``GET /v1beta1/options/snapshots/{underlying}`` —
                     snapshots keyed by BARE OCC symbol (no "O:" prefix),
                     each with latestQuote (bp/ap/bs/as — REAL NBBO),
                     latestTrade, dailyBar, greeks (delta/gamma/theta/vega/
                     rho) and impliedVolatility. NO open_interest here —
                     merged from the Trading API contracts endpoint.
  - Contracts:       ``GET {paper-api}/v2/options/contracts`` (Trading API,
                     same keys) — reference rows incl. open_interest and
                     close_price.
  - Single-contract
    snapshot:        ``GET /v1beta1/options/snapshots?symbols=...`` — used
                     for the EOD previous-day bar read.
  - News:            ``GET /v1beta1/news`` — real articles with id,
                     headline, source, created_at, url, symbols.

Auth: headers ``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY``. The key never
appears in a log line or error message.
"""
import logging
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

ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
#: Trading API host for the option-contracts REFERENCE endpoint (paper keys
#: work here; reference data is identical across paper/live).
ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_RETRY_AFTER_SECONDS = 2.0

#: Options feed: Algo Trader Plus includes OPRA (real-time). The parameter
#: exists so a downgraded account can be pointed at "indicative" explicitly —
#: never silently.
DEFAULT_OPTIONS_FEED = "opra"

#: Chain pagination: Alpaca pages option snapshots; follow next_page_token up
#: to this many pages so a pathological chain cannot loop forever.
CHAIN_PAGE_LIMIT = 1000  # rows per page (server max for this endpoint family)
DEFAULT_MAX_CHAIN_PAGES = 8

#: Daily-bars calendar buffer: fetch days*1.6 calendar days so weekends and
#: holidays still leave at least `days` trading bars to trim to.
CALENDAR_BUFFER = 1.6

#: Stocks feed. "iex" is what the account's stock endpoints already use for
#: bars; it is stated explicitly on intraday requests because the default feed
#: differs by subscription tier, and a silently different feed would change
#: which minutes exist in a replay window.
STOCKS_FEED = "iex"

#: Intraday bars: the server's per-page maximum for this endpoint. Fewer
#: round-trips per event window; the page_token loop handles the rest.
INTRADAY_PAGE_LIMIT = 10_000

#: Intraday pagination cap. One event window is at most ~2 sessions of minutes
#: (< 2000 bars), so this is generous head-room, not a working limit — it
#: exists only so a server that kept returning a cursor could not loop
#: forever. Hitting it logs a WARNING and the truncation is visible.
DEFAULT_MAX_INTRADAY_PAGES = 20

#: Windowed news (Phase D §21): the server's per-page maximum for
#: /v1beta1/news. Fewer round-trips per event window; the page_token loop
#: handles the rest.
NEWS_PAGE_LIMIT = 50

#: Windowed-news pagination cap. An event window is ~120 days of one ticker's
#: coverage (a few hundred articles at most), so this is head-room against a
#: server that never stops sending a cursor, not a working limit; hitting it
#: logs a WARNING so the truncation is visible rather than silent.
DEFAULT_MAX_NEWS_PAGES = 40

#: The exchange clock. Alpaca daily bars stamp t at 04:00Z (midnight ET);
#: the bar's trading DATE is its Eastern date.
EASTERN = ZoneInfo("America/New_York")

#: Symbols Alpaca cannot serve (no index feed). Requests for them are
#: answered with an honest absence, never a proxy. Parameter, not a truth.
DEFAULT_UNSERVABLE_SYMBOLS = frozenset({"VIX", "^VIX", "^SPX", "^GSPC"})

#: Cheap, always-listed symbol used by probe_capabilities().
PROBE_SYMBOL = "SPY"
PROBE_HISTORY_DAYS = 5

#: Alpaca sells no company fundamentals at ANY tier (data_source.md §1/§2:
#: Massive is reserved for fundamentals precisely because of this gap). The
#: probe reports it as a plain False rather than firing a request that could
#: only 404 — a structural absence, not a plan the operator could upgrade.
#: The key exists so the capability set is IDENTICAL across providers and the
#: Settings view never has to special-case which provider it is talking to.
FINANCIALS_AVAILABLE = False

#: Symbols already warned as unservable — the condition is permanent, so it
#: is logged once per process, not once per request.
_WARNED_UNSERVABLE: set[str] = set()

#: Open interest is a once-per-day OCC number: underlying -> (eastern_day,
#: {occ_symbol: oi}). Module-level because provider instances are
#: per-request; bounded by the set of viewed underlyings (old days evicted).
_OI_DAY_CACHE: dict[str, tuple[date, dict[str, int]]] = {}

#: Bare OCC option symbol (Alpaca form, no "O:" prefix):
#: root (1-6 chars), YYMMDD, C/P, strike in thousandths padded to 8.
_OCC_BARE_RE = re.compile(
    r"^(?P<root>[A-Z][A-Z0-9.]{0,5})(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$"
)


def _parse_bare_occ(symbol: str) -> tuple[date, str, float] | None:
    """Parse a bare OCC option symbol to (expiry, right, strike) or None."""
    m = _OCC_BARE_RE.match(symbol or "")
    if m is None:
        return None
    raw = m.group("expiry")
    try:
        expiry = date(2000 + int(raw[0:2]), int(raw[2:4]), int(raw[4:6]))
    except ValueError:
        return None
    return expiry, m.group("right"), int(m.group("strike")) / 1000.0


def _as_float(value: object) -> float | None:
    """A JSON number as float; None for anything else (bool included).

    Alpaca's Trading API stringifies some numerics (open_interest,
    close_price, strike_price) — numeric STRINGS parse too.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _ts_from_rfc3339(value: object) -> datetime | None:
    """An RFC-3339 timestamp as an aware UTC datetime; None if unusable.

    Alpaca stamps nanosecond precision ("...834543155Z");
    ``datetime.fromisoformat`` (3.11+) accepts 'Z' but not >6 fractional
    digits, so the fraction is trimmed to microseconds first.
    """
    if not isinstance(value, str) or not value:
        return None
    raw = value.replace("Z", "+00:00")
    m = re.match(r"^(.*?\.)(\d+)(\+00:00)$", raw)
    if m and len(m.group(2)) > 6:
        raw = f"{m.group(1)}{m.group(2)[:6]}{m.group(3)}"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    """An aware UTC instant as the RFC-3339 form Alpaca's start/end expect."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_intraday_row(symbol: str, row: dict) -> IntradayBar | None:
    """One ``bars[]`` row as an :class:`IntradayBar`, or None if unusable.

    Volume is required and must be a whole share count: unlike the daily
    adapter (whose Bar.volume is a float and defaults to 0.0), an intraday bar
    with no volume cannot be volume-zeroed, because event replay COMPARES
    volumes and a fabricated zero would read as "nobody traded that minute".
    """
    ts = _ts_from_rfc3339(row.get("t"))
    o, h = _as_float(row.get("o")), _as_float(row.get("h"))
    low, c = _as_float(row.get("l")), _as_float(row.get("c"))
    v = _as_float(row.get("v"))
    if ts is None or None in (o, h, low, c, v) or c <= 0 or v < 0:
        logger.debug("Alpaca %s intraday bar skipped: incomplete row %r", symbol, row)
        return None
    return IntradayBar(
        ts=ts, open=o, high=h, low=low, close=c, volume=int(round(v))
    )


def _parse_news_row(row: dict) -> NewsArticle | None:
    """One ``news[]`` row as a :class:`NewsArticle`, or None if unusable.

    Shared by :meth:`AlpacaMarketDataProvider.get_news` and
    :meth:`AlpacaMarketDataProvider.get_news_window` so the recency feed and
    the windowed feed normalise IDENTICALLY — the same article fetched two
    ways must produce the same ``source_id``, or dedup downstream would keep
    both copies.

    Rows missing any citable field (id, headline, url, created_at) return
    None and are SKIPPED by the caller, never patched: an article that cannot
    be cited cannot ground a recommendation. ``source_id`` is prefixed
    ``alpaca:`` so the dedup keyspace can never collide with another
    provider's ids.
    """
    if not isinstance(row, dict):
        return None
    raw_id = row.get("id")
    title = row.get("headline")
    url = row.get("url")
    published_at = _ts_from_rfc3339(row.get("created_at"))
    if (
        raw_id is None
        or not isinstance(title, str) or not title
        or not isinstance(url, str) or not url
        or published_at is None
    ):
        logger.debug("Alpaca news row skipped: missing citable field (id=%r)", raw_id)
        return None
    tickers_raw = row.get("symbols")
    tickers = tuple(
        t for t in tickers_raw if isinstance(t, str) and t
    ) if isinstance(tickers_raw, list) else ()
    source = row.get("source")
    summary = row.get("summary")
    return NewsArticle(
        source_id=f"alpaca:{raw_id}",
        title=title,
        publisher=source if isinstance(source, str) else "",
        published_at=published_at,
        url=url,
        tickers=tickers,
        description=summary if isinstance(summary, str) else "",
    )


def _retry_after_seconds(response: httpx.Response, default: float) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


class AlpacaMarketDataProvider:
    """MarketDataProvider backed by the Alpaca Market Data API. Real data only."""

    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        data_base_url: str = ALPACA_DATA_BASE_URL,
        trading_base_url: str = ALPACA_TRADING_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retry_after_default_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        options_feed: str = DEFAULT_OPTIONS_FEED,
        max_chain_pages: int = DEFAULT_MAX_CHAIN_PAGES,
        max_intraday_pages: int = DEFAULT_MAX_INTRADAY_PAGES,
        max_news_pages: int = DEFAULT_MAX_NEWS_PAGES,
        unservable_symbols: frozenset[str] = DEFAULT_UNSERVABLE_SYMBOLS,
    ) -> None:
        if not api_key_id or not api_key_id.strip():
            raise MarketDataError(
                "Alpaca market data requires credentials — set "
                "ALPACA_API_KEY_ID (the key itself is never logged)"
            )
        if not api_secret_key or not api_secret_key.strip():
            raise MarketDataError(
                "Alpaca market data requires credentials — set "
                "ALPACA_API_SECRET_KEY (the key itself is never logged)"
            )
        self.data_base_url = data_base_url.rstrip("/")
        self.trading_base_url = trading_base_url.rstrip("/")
        self.retry_after_default_seconds = retry_after_default_seconds
        self.options_feed = options_feed
        self.max_chain_pages = max_chain_pages
        self.max_intraday_pages = max_intraday_pages
        self.max_news_pages = max_news_pages
        self.unservable_symbols = unservable_symbols
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "APCA-API-KEY-ID": api_key_id.strip(),
                "APCA-API-SECRET-KEY": api_secret_key.strip(),
            },
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        # Providers are constructed per get_provider() call (matching the
        # Massive adapter's precedent); explicit close-on-collect keeps the
        # socket pool from lingering until interpreter teardown.
        self.close()

    # ------------------------------------------------------------------
    # Transport with the documented failure taxonomy
    # ------------------------------------------------------------------

    def _request(
        self, url: str, params: dict | None = None, *, allow_404: bool = False
    ) -> httpx.Response:
        """One Alpaca call. 429 -> one Retry-After retry; 401/403 -> the
        documented taxonomy; 404 returned to the caller when `allow_404`
        (an unknown symbol is an honest absence, not a fault)."""
        endpoint = httpx.URL(url).path
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise MarketDataError(
                f"Alpaca request failed for {endpoint}: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code == 429:
            delay = _retry_after_seconds(response, self.retry_after_default_seconds)
            logger.warning(
                "Alpaca rate limited (HTTP 429) on %s; retrying once in %.1fs",
                endpoint, delay,
            )
            if delay > 0:
                time.sleep(delay)
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                raise MarketDataError(
                    f"Alpaca retry failed for {endpoint}: {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code == 429:
                raise MarketDataError(
                    f"Alpaca rate limit (HTTP 429) persisted after one retry "
                    f"for {endpoint}"
                )

        if response.status_code == 401:
            raise MarketDataError(
                f"Alpaca rejected the API key (HTTP 401) for {endpoint} — "
                "check ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (the keys "
                "are never logged or echoed)"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"Alpaca returned HTTP 403 for {endpoint}: the account's "
                "market data subscription does not include this endpoint. "
                "There is NO synthetic fallback: "
                f"{response.text[:300]}"
            )
        if response.status_code == 404 and allow_404:
            return response
        if response.status_code >= 400:
            raise MarketDataError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"{endpoint}: {response.text[:300]}"
            )
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketDataError(
                f"Alpaca returned unparseable JSON for "
                f"{httpx.URL(str(response.url)).path}"
            ) from exc
        if not isinstance(payload, dict):
            raise MarketDataError(
                f"Alpaca returned a non-object payload for "
                f"{httpx.URL(str(response.url)).path}"
            )
        return payload

    # ------------------------------------------------------------------
    # Stocks
    # ------------------------------------------------------------------

    def get_daily_bars(self, symbol: str, days: int) -> list[Bar]:
        """Last `days` daily bars, oldest first (data_source.md §2.1).

        ``GET /v2/stocks/{symbol}/bars`` with timeframe=1Day and
        adjustment=split (split-adjusted continuity, matching the platform's
        prior bar semantics), following next_page_token. The bar's trading
        DATE is its Eastern date (Alpaca stamps 04:00Z = midnight ET).
        """
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=max(int(days * CALENDAR_BUFFER), days + 5))
        bars: list[Bar] = []
        params: dict | None = {
            "timeframe": "1Day",
            "start": from_date.isoformat(),
            "end": to_date.isoformat(),
            "adjustment": "split",
            "limit": 10_000,
            "sort": "asc",
        }
        url = f"{self.data_base_url}/v2/stocks/{symbol}/bars"
        pages = 0
        page_token: str | None = None
        exhausted_with_token = False
        while pages < self.max_chain_pages:
            call_params = dict(params or {})
            if page_token:
                call_params["page_token"] = page_token
            payload = self._json(self._request(url, params=call_params, allow_404=True))
            pages += 1
            rows = payload.get("bars")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                ts = _ts_from_rfc3339(row.get("t"))
                o, h = _as_float(row.get("o")), _as_float(row.get("h"))
                low, c = _as_float(row.get("l")), _as_float(row.get("c"))
                v = _as_float(row.get("v"))
                if ts is None or None in (o, h, low, c) or c <= 0:
                    logger.debug("Alpaca %s bar skipped: incomplete row %r", symbol, row)
                    continue
                bars.append(
                    Bar(
                        ts=ts.astimezone(EASTERN).date(),
                        open=o, high=h, low=low, close=c,
                        volume=v if v is not None else 0.0,
                    )
                )
            token = payload.get("next_page_token")
            if not isinstance(token, str) or not token:
                break
            page_token = token
            exhausted_with_token = pages >= self.max_chain_pages
        if exhausted_with_token:
            logger.warning(
                "Alpaca daily bars for %s truncated at %d pages with more "
                "pages remaining", symbol, pages,
            )
        bars.sort(key=lambda b: b.ts)
        return bars[-days:]

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
    ) -> list[IntradayBar]:
        """Intraday bars over ``[start, end]``, ascending and de-duplicated.

        ``GET /v2/stocks/{symbol}/bars?timeframe=1Min&start=&end=&feed=iex&
        adjustment=split&limit=10000`` following ``next_page_token`` until the
        server stops sending one (verified live 2026-08-19).

        EXTENDED HOURS ARE INCLUDED, deliberately. Alpaca serves sparse
        pre-market and after-hours minutes alongside the complete regular
        session (e.g. AAPL 2024-05-02 carried 29 bars between 20:31Z and
        20:59Z), and those are precisely the minutes an after-market earnings
        release moves. Filtering them out here would delete the reaction the
        caller came for; classifying them is the consumer's job.

        ``adjustment=split`` matches :meth:`get_daily_bars`, so a minute close
        and a daily close for the same session are on the SAME price basis and
        a gap computed across the two is real rather than a split artefact.

        A range earlier than the symbol's history returns ``bars: null``,
        which is an honest ``[]`` — never extrapolated backwards. Rows missing
        a timestamp, an OHLC field or the volume field are SKIPPED with a
        debug log: a minute bar with an invented volume would corrupt the
        volume comparisons event replay makes, and 0 must keep meaning
        "Alpaca reported zero trades".
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )

        url = f"{self.data_base_url}/v2/stocks/{symbol}/bars"
        base_params = {
            "timeframe": timeframe,
            "start": _rfc3339(start_utc),
            "end": _rfc3339(end_utc),
            "feed": STOCKS_FEED,
            "adjustment": "split",
            "limit": INTRADAY_PAGE_LIMIT,
            "sort": "asc",
        }
        # ts -> bar: de-duplication across pages, FIRST parse wins. A repeated
        # ts on a page boundary is a pagination artefact, not two minutes.
        by_ts: dict[datetime, IntradayBar] = {}
        pages = 0
        page_token: str | None = None
        while pages < self.max_intraday_pages:
            call_params = dict(base_params)
            if page_token:
                call_params["page_token"] = page_token
            payload = self._json(self._request(url, params=call_params, allow_404=True))
            pages += 1
            rows = payload.get("bars")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                bar = _parse_intraday_row(symbol, row)
                if bar is None:
                    continue
                by_ts.setdefault(bar.ts, bar)
            token = payload.get("next_page_token")
            if not isinstance(token, str) or not token:
                page_token = None
                break
            page_token = token
        if page_token:
            logger.warning(
                "Alpaca intraday bars for %s truncated at %d pages "
                "(max_intraday_pages=%d); more bars exist in %s..%s",
                symbol, pages, self.max_intraday_pages,
                base_params["start"], base_params["end"],
            )
        return [by_ts[ts] for ts in sorted(by_ts)]

    def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Current quotes via ONE multi-symbol snapshot call.

        ``GET /v2/stocks/snapshots?symbols=...`` — price = latestTrade.p,
        day change vs prevDailyBar.c, ts = latestTrade.t. Symbols Alpaca
        cannot serve (indices like VIX — no index feed on Alpaca) are
        SKIPPED with a warning: an honest absence, never a proxy value.
        """
        servable = [s for s in symbols if s.upper() not in self.unservable_symbols]
        for s in symbols:
            if s.upper() in self.unservable_symbols:
                # A PERMANENT condition (Alpaca has no index feed): logged
                # once per symbol per process, not per request.
                if s.upper() not in _WARNED_UNSERVABLE:
                    _WARNED_UNSERVABLE.add(s.upper())
                    logger.warning(
                        "Alpaca serves no data for %s (no index feed) — "
                        "omitted from quotes (honest absence, no proxy; "
                        "logged once)", s,
                    )
        if not servable:
            return []

        payload = self._json(
            self._request(
                f"{self.data_base_url}/v2/stocks/snapshots",
                params={"symbols": ",".join(servable)},
            )
        )
        quotes: list[Quote] = []
        for symbol in servable:
            snap = payload.get(symbol)
            if not isinstance(snap, dict):
                logger.warning("Alpaca snapshot missing for %s — skipped", symbol)
                continue
            trade = snap.get("latestTrade") or {}
            prev = snap.get("prevDailyBar") or {}
            price = _as_float(trade.get("p"))
            prev_close = _as_float(prev.get("c"))
            ts = _ts_from_rfc3339(trade.get("t"))
            if price is None or price <= 0 or ts is None:
                logger.warning(
                    "Alpaca snapshot for %s has no usable latest trade — skipped",
                    symbol,
                )
                continue
            if prev_close is None or prev_close <= 0:
                # No previous close -> the day change is UNKNOWN. Quote
                # requires change_pct, and 0.00% would fabricate an
                # "unchanged" reading Alpaca never reported — skip the
                # symbol instead (honest absence), exactly as the Massive
                # adapter does for the same gap.
                logger.warning(
                    "Alpaca snapshot for %s has no usable previous close — "
                    "day change unknowable; skipped (never reported as 0.00%%)",
                    symbol,
                )
                continue
            change_pct = (price / prev_close - 1.0) * 100.0
            quotes.append(
                Quote(symbol=symbol, price=price, change_pct=change_pct, ts=ts)
            )
        return quotes

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def get_option_chain(
        self, symbol: str, spot: float, as_of: date
    ) -> list[OptionQuote]:
        """The CURRENT option chain for `symbol` (plan §9, data_source.md §8).

        ``GET /v1beta1/options/snapshots/{symbol}`` (OPRA feed) merged with
        open interest from the Trading API contracts endpoint. Current-state
        only: a historical `as_of` raises — a chain we do not have is
        reported as such, never approximated.

        Row admission (never zero-filled):
        - a REAL two-sided NBBO (bp>0 and ap>0) prices the row
          (price_basis "quote"); otherwise a positive dailyBar close prices
          it as an EOD close (price_basis "day_close", bid/ask honest
          zeros + worst-case spread so unknown liquidity can only REJECT
          in the §9 selector);
        - greeks (delta/gamma/theta/vega) AND impliedVolatility must all be
          present, or the row is skipped with a debug log;
        - open interest missing from the contracts merge reads 0 —
          "none reported", which §9 liquidity floors can only reject on.
        """
        today = datetime.now(EASTERN).date()
        if as_of != today:
            raise MarketDataError(
                f"Alpaca option-chain snapshots are current-state only: "
                f"requested as_of={as_of.isoformat()} but today is "
                f"{today.isoformat()}. Historical chains are not served by "
                "/v1beta1/options/snapshots and will not be approximated."
            )

        oi_by_symbol = self._open_interest_map(symbol)

        chain: list[OptionQuote] = []
        url = f"{self.data_base_url}/v1beta1/options/snapshots/{symbol}"
        page_token: str | None = None
        pages = 0
        exhausted_with_token = False
        while pages < self.max_chain_pages:
            params: dict = {"limit": CHAIN_PAGE_LIMIT, "feed": self.options_feed}
            if page_token:
                params["page_token"] = page_token
            payload = self._json(self._request(url, params=params))
            pages += 1
            snaps = payload.get("snapshots")
            for occ, row in (snaps or {}).items() if isinstance(snaps, dict) else []:
                quote = self._parse_chain_row(occ, row, as_of, oi_by_symbol)
                if quote is not None:
                    chain.append(quote)
            token = payload.get("next_page_token")
            if not isinstance(token, str) or not token:
                break
            page_token = token
            # Truncation means: the budget is spent AND the server still
            # offered another page. A chain completing exactly on the last
            # allowed page is NOT truncated.
            exhausted_with_token = pages >= self.max_chain_pages
        if exhausted_with_token:
            logger.warning(
                "Alpaca option chain for %s truncated at %d pages "
                "(max_chain_pages=%d); more contracts exist beyond the cap",
                symbol, pages, self.max_chain_pages,
            )

        # Deterministic ORDER (not data): §9 selector breaks ties by order.
        chain.sort(key=lambda q: (q.expiry, q.strike, q.right))
        return chain

    def _parse_chain_row(
        self,
        occ: str,
        row: object,
        as_of: date,
        oi_by_symbol: dict[str, int],
    ) -> OptionQuote | None:
        from libs.trading_core.contracts import ContractQuote  # local: avoid cycle

        if not isinstance(row, dict):
            return None
        parsed = _parse_bare_occ(occ)
        if parsed is None:
            logger.debug("Alpaca chain row %r skipped: unparseable OCC symbol", occ)
            return None
        expiry, right, strike = parsed
        dte = (expiry - as_of).days
        if dte < 0:
            return None

        latest_quote = row.get("latestQuote") or {}
        bid = _as_float(latest_quote.get("bp"))
        ask = _as_float(latest_quote.get("ap"))
        day = row.get("day") or row.get("dailyBar") or {}
        day_close = _as_float(day.get("c"))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 0.0
            price_basis = "quote"
        elif ask is not None and ask > 0 and (bid is None or bid <= 0):
            # ONE-SIDED NBBO: OPRA reports bid 0 when NO BID exists — a real
            # market state, and the offer is a real quote (observed live:
            # deep OTM wings quoted ask-only after hours). Keep the real ask;
            # bid 0.0 is the reported no-bid; mid is the canonical midpoint
            # of a no-bid market (ask/2) and the spread lands exactly at the
            # documented worst case, so §9 can only ever REJECT on it.
            bid = 0.0
            mid = ask / 2.0
            spread_pct = 2.0
            price_basis = "quote"
        elif day_close is not None and day_close > 0:
            # No two-sided NBBO right now (e.g. pre-market): the session
            # close is a REAL traded price; spread unknown -> conservative
            # worst case, so it can only ever REJECT in the selector.
            bid, ask = 0.0, 0.0
            mid = day_close
            spread_pct = 2.0
            price_basis = "day_close"
        else:
            logger.debug(
                "Alpaca chain row %r skipped: no two-sided NBBO and no "
                "session close (unquotable)", occ,
            )
            return None
        if mid <= 0:
            return None

        # Greeks/IV: Alpaca's values or honest None — the feed omits them on
        # deep ITM/OTM contracts. The row is KEPT (its quote is real chain
        # data the user should see); the §9 selector rejects greekless rows
        # with a named reason, and nothing is ever zero-filled.
        greeks = row.get("greeks") or {}
        delta = _as_float(greeks.get("delta"))
        gamma = _as_float(greeks.get("gamma"))
        theta = _as_float(greeks.get("theta"))
        vega = _as_float(greeks.get("vega"))
        iv = _as_float(row.get("impliedVolatility"))
        if None in (delta, gamma, theta, vega, iv):
            delta = gamma = theta = vega = iv = None
            logger.debug(
                "Alpaca chain row %r has no greeks/IV — kept with honest "
                "nulls (selector will reject it by name)", occ,
            )

        last_trade = row.get("latestTrade") or {}
        volume = _as_float(day.get("v"))
        return ContractQuote(
            expiry=expiry,
            dte=dte,
            strike=strike,
            right=right,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_pct=spread_pct,
            last=_as_float(last_trade.get("p")),
            volume=int(volume) if volume is not None else 0,
            open_interest=oi_by_symbol.get(occ, 0),
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            price_basis=price_basis,
        )

    def _open_interest_map(self, underlying: str) -> dict[str, int]:
        """OCC symbol -> open interest, DAY-CACHED per underlying.

        Open interest is computed by OCC once per day (the contracts
        endpoint even stamps ``open_interest_date``), so re-fetching it on
        every chain rebuild only re-reads the same numbers — one contracts
        call per (underlying, Eastern day) suffices. The cache is
        process-wide (module-level) because provider instances are
        constructed per request.

        Source: the Trading API contracts
        endpoint (snapshots do not carry OI). A fetch fault degrades to an
        EMPTY map with a warning — OI then reads 0 ("none reported"),
        which the §9 liquidity floor can only reject on; it never invents.
        Faults are NOT cached: the next chain rebuild retries.
        """
        today = datetime.now(EASTERN).date()
        cached = _OI_DAY_CACHE.get(underlying)
        if cached is not None and cached[0] == today:
            return cached[1]
        try:
            contracts = self.get_option_contracts(underlying)
        except (CapabilityNotAvailable, MarketDataError) as exc:
            logger.warning(
                "Alpaca open-interest merge unavailable for %s (%s) — chain "
                "rows will read OI 0 (none reported)", underlying, exc,
            )
            return {}
        result: dict[str, int] = {}
        for c in contracts:
            oi = _as_float(c.get("open_interest"))
            if oi is not None and oi >= 0:
                result[str(c["ticker"])] = int(oi)
        # Evict other days (bounded by the set of viewed underlyings).
        for key in [k for k, v in _OI_DAY_CACHE.items() if v[0] != today]:
            del _OI_DAY_CACHE[key]
        _OI_DAY_CACHE[underlying] = (today, result)
        return result

    def get_option_contracts(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        max_pages: int = 2,
    ) -> list[dict]:
        """Option CONTRACT REFERENCE rows (Trading API, same credentials).

        ``GET {trading}/v2/options/contracts`` — active contracts with
        open_interest and close_price. Returns the platform-normal shape
        ``{"ticker","contract_type","strike_price","expiration_date",
        "shares_per_contract","open_interest","close_price",
        "close_price_date"}``; tickers are Alpaca's BARE OCC symbols.
        """
        params: dict = {
            "underlying_symbols": underlying,
            "status": "active",
            "limit": 10_000,
        }
        if expiration_gte is not None:
            params["expiration_date_gte"] = expiration_gte.isoformat()
        if expiration_lte is not None:
            params["expiration_date_lte"] = expiration_lte.isoformat()

        contracts: list[dict] = []
        pages = 0
        page_token: str | None = None
        url = f"{self.trading_base_url}/v2/options/contracts"
        while pages < max_pages:
            call_params = dict(params)
            if page_token:
                call_params["page_token"] = page_token
            payload = self._json(self._request(url, params=call_params))
            pages += 1
            rows = payload.get("option_contracts")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("symbol", ""))
                contract_type = str(row.get("type", "")).lower()
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
                        "Alpaca contract row skipped: incomplete identity %r",
                        row.get("symbol"),
                    )
                    continue
                contracts.append(
                    {
                        "ticker": ticker,
                        "contract_type": contract_type,
                        "strike_price": strike,
                        "expiration_date": expiry,
                        "shares_per_contract": _as_float(row.get("size")) or 100.0,
                        "open_interest": _as_float(row.get("open_interest")),
                        "close_price": _as_float(row.get("close_price")),
                        "close_price_date": row.get("close_price_date"),
                    }
                )
            token = payload.get("next_page_token")
            if not isinstance(token, str) or not token:
                break
            page_token = token
        return contracts

    def get_option_contracts_window(
        self,
        underlying: str,
        as_of: date,
        dte_min: int,
        dte_max: int,
        spot: float,
        contract_type: str = "call",
    ) -> list[dict]:
        """Contracts that EXISTED at ``as_of`` with expiry in the DTE window,
        strikes within ±25% of ``spot`` — the historical-backtest contract
        universe (user mandate 2026-08-17).

        Queries BOTH statuses: contracts from past dates are ``inactive``
        now, and near-window ones may still be ``active``. Adjusted contracts
        (leading digit, e.g. ``1AAPL…``) are corporate-action artifacts and
        NOT valid data-API symbols — filtered out. Returns
        ``[{"ticker","strike_price","expiration_date"}, ...]``.
        """
        lo = as_of + timedelta(days=dte_min)
        hi = as_of + timedelta(days=dte_max)
        out: list[dict] = []
        url = f"{self.trading_base_url}/v2/options/contracts"
        for status in ("inactive", "active"):
            params: dict = {
                "underlying_symbols": underlying,
                "status": status,
                "type": contract_type,
                "expiration_date_gte": lo.isoformat(),
                "expiration_date_lte": hi.isoformat(),
                "strike_price_gte": f"{spot * 0.75:.2f}",
                "strike_price_lte": f"{spot * 1.25:.2f}",
                "limit": 10_000,
            }
            pages = 0
            page_token: str | None = None
            while pages < 4:
                call_params = dict(params)
                if page_token:
                    call_params["page_token"] = page_token
                payload = self._json(self._request(url, params=call_params))
                pages += 1
                rows = payload.get("option_contracts")
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    ticker = str(row.get("symbol", ""))
                    if not ticker or ticker[0].isdigit():
                        continue  # adjusted contract — not a data-API symbol
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
                    if strike is None or strike <= 0 or expiry is None:
                        continue
                    out.append(
                        {
                            "ticker": ticker,
                            "strike_price": strike,
                            "expiration_date": expiry,
                        }
                    )
                token = payload.get("next_page_token")
                if not isinstance(token, str) or not token:
                    break
                page_token = token
        return out

    def get_option_daily_bars(
        self, option_ticker: str, start: date, end: date
    ) -> dict[date, tuple[float, float]]:
        """REAL daily bars of one contract over [start, end] — the options
        backtest's price source (full contract life served from ~Feb 2024;
        see data-source-architecture.md). Returns {date: (open, close)};
        days the contract did not trade are simply absent.
        """
        url = f"{self.data_base_url}/v1beta1/options/bars"
        params = {
            "symbols": option_ticker,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10_000,
        }
        payload = self._json(self._request(url, params=params))
        rows = (payload.get("bars") or {}).get(option_ticker) or []
        out: dict[date, tuple[float, float]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            ts = row.get("t")
            o = _as_float(row.get("o"))
            c = _as_float(row.get("c"))
            if not isinstance(ts, str) or o is None or c is None:
                continue
            try:
                bar_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(EASTERN).date()
            except ValueError:
                continue
            out[bar_date] = (o, c)
        return out

    def get_option_prev_bar(self, option_ticker: str) -> dict | None:
        """Previous-session EOD bar for one contract, from its snapshot's
        prevDailyBar (``GET /v1beta1/options/snapshots?symbols=...``).
        None when the contract has no previous-session bar — an illiquid
        contract that did not trade is an honest absence, never a zero.
        """
        payload = self._json(
            self._request(
                f"{self.data_base_url}/v1beta1/options/snapshots",
                params={"symbols": option_ticker, "feed": self.options_feed},
            )
        )
        snaps = payload.get("snapshots")
        row = snaps.get(option_ticker) if isinstance(snaps, dict) else None
        if not isinstance(row, dict):
            return None
        prev = row.get("prevDailyBar") or {}
        close = _as_float(prev.get("c"))
        if close is None or close <= 0:
            return None
        ts = _ts_from_rfc3339(prev.get("t"))
        return {
            "open": _as_float(prev.get("o")),
            "high": _as_float(prev.get("h")),
            "low": _as_float(prev.get("l")),
            "close": close,
            "volume": _as_float(prev.get("v")) or 0.0,
            "vwap": _as_float(prev.get("vw")),
            "date": ts.astimezone(EASTERN).date().isoformat() if ts else None,
        }

    # ------------------------------------------------------------------
    # News (data_source.md: Alpaca News API — included with market data)
    # ------------------------------------------------------------------

    def get_news(self, limit: int = 50) -> list[NewsArticle]:
        """Latest real news articles (``GET /v1beta1/news``).

        Every field is Alpaca's verbatim; rows missing any citable field
        (id, headline, url, created_at) are SKIPPED, never patched.
        ``source_id`` is prefixed ``alpaca:`` so the dedup keyspace can
        never collide with previously stored provider ids.
        """
        payload = self._json(
            self._request(
                f"{self.data_base_url}/v1beta1/news",
                params={"limit": max(1, min(int(limit), 50)), "sort": "desc"},
            )
        )
        articles: list[NewsArticle] = []
        rows = payload.get("news")
        for row in rows if isinstance(rows, list) else []:
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

        ``GET /v1beta1/news?symbols=<CSV>&start=&end=&limit=50&sort=desc&
        include_content=false`` following ``next_page_token`` until the server
        stops sending one.

        ONE request per page, not per ticker: Alpaca's ``symbols`` parameter
        takes a comma-separated list and tags each article with every symbol
        it matched, so asking for the basket at once costs a fraction of the
        round-trips and returns each syndicated article ONCE rather than once
        per ticker. ``include_content=false`` keeps the full article body off
        the wire — the evidence engine scores titles and summaries, and the
        body would multiply the payload for nothing.

        ``sort=desc`` makes the first page the most recent, so a `limit` that
        bites keeps the NEWEST articles, matching :meth:`get_news`. Duplicate
        ``source_id`` values (a page boundary repeating a row) are collapsed,
        first parse wins.
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )
        symbols = [
            t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()
        ]
        if not symbols or limit <= 0:
            # No symbols to ask about, or no room for an answer. An empty
            # request is answered here rather than by the server, which would
            # read `symbols=` as "every symbol" and return the whole firehose.
            return []

        url = f"{self.data_base_url}/v1beta1/news"
        base_params = {
            "symbols": ",".join(dict.fromkeys(symbols)),
            "start": _rfc3339(start_utc),
            "end": _rfc3339(end_utc),
            "limit": NEWS_PAGE_LIMIT,
            "sort": "desc",
            "include_content": "false",
        }
        # source_id -> article: de-duplication across pages, FIRST parse wins.
        by_id: dict[str, NewsArticle] = {}
        pages = 0
        page_token: str | None = None
        while pages < self.max_news_pages:
            call_params = dict(base_params)
            if page_token:
                call_params["page_token"] = page_token
            payload = self._json(self._request(url, params=call_params))
            pages += 1
            rows = payload.get("news")
            for row in rows if isinstance(rows, list) else []:
                article = _parse_news_row(row)
                if article is None:
                    continue
                by_id.setdefault(article.source_id, article)
            if len(by_id) >= limit:
                page_token = None
                break
            token = payload.get("next_page_token")
            if not isinstance(token, str) or not token:
                page_token = None
                break
            page_token = token
        if page_token:
            logger.warning(
                "Alpaca windowed news for %s truncated at %d pages "
                "(max_news_pages=%d); more articles exist in %s..%s",
                base_params["symbols"], pages, self.max_news_pages,
                base_params["start"], base_params["end"],
            )
        ordered = sorted(
            by_id.values(), key=lambda a: (a.published_at, a.source_id), reverse=True
        )
        return ordered[:limit]

    # ------------------------------------------------------------------
    # Point-in-time option reference — not served by Alpaca (Phase I §36)
    # ------------------------------------------------------------------

    def list_option_contracts(
        self,
        underlying: str,
        *,
        expiration_date: date,
        as_of: date,
        right: str | None = None,
        limit: int = 250,
    ) -> list[OptionContractRef]:
        """Always raises :class:`CapabilityNotAvailable` — Alpaca has no as-of view.

        Alpaca's contracts endpoint (``GET /v2/options/contracts``) answers
        "what is listed NOW"; it takes no as-of parameter and drops expired
        contracts, so it cannot say which strikes existed BEFORE a past
        earnings release — which is the only question §36 asks of it.
        Filtering today's universe locally would look like an answer and be a
        wrong one: strikes listed in reaction to the event would slip into
        that event's pre-event straddle.

        The refusal is explicit and names the missing endpoint so the caller
        renders "Unavailable — reason" (§16, §44 rule 18). Returning ``[]``
        would read as "no contracts existed for that expiry", a false claim
        about the market rather than an honest one about the provider.
        """
        raise CapabilityNotAvailable(
            "Alpaca does not serve POINT-IN-TIME option contract reference "
            "data: GET /v2/options/contracts is current-listing only (no "
            "as_of parameter, expired contracts dropped), so contracts for "
            f"{(underlying or '').strip().upper() or '?'} expiring "
            f"{expiration_date.isoformat()} as of {as_of.isoformat()} cannot "
            "be established. Configure the Massive provider "
            "(/v3/reference/options/contracts) for historical option "
            "reference; there is NO synthetic fallback."
        )

    def get_option_history_bars(
        self, option_ticker: str, start: date, end: date
    ) -> list[Bar]:
        """Always raises :class:`CapabilityNotAvailable` — see the note below.

        Alpaca DOES serve historical option bars, and this adapter already
        reads them in :meth:`get_option_daily_bars` for the options backtest.
        This method still refuses, because the §36 pipeline needs BOTH halves
        of the capability from ONE provider: contracts that existed as of a
        past date, and that contract's bars. Alpaca cannot answer the first
        (see :meth:`list_option_contracts`), and serving the second here would
        invite a caller to pair Alpaca bars with Massive contract identities —
        a cross-provider blend that data_source.md §33 forbids precisely
        because it corrupts price provenance.

        Callers wanting Alpaca's option bars for the BACKTEST path call
        :meth:`get_option_daily_bars`, which is unchanged and still real.
        """
        raise CapabilityNotAvailable(
            "Alpaca does not serve the Phase I historical-option capability: "
            "/v1beta1/options/bars has no point-in-time contract reference to "
            f"pair with, so bars for {(option_ticker or '?').strip()} over "
            f"[{start.isoformat()}, {end.isoformat()}] are not served through "
            "this seam (no cross-provider blending, data_source.md §33). "
            "Configure the Massive provider for historical option history; "
            "there is NO synthetic fallback."
        )

    # ------------------------------------------------------------------
    # Fundamentals — not sold by Alpaca at any tier
    # ------------------------------------------------------------------

    def get_financials(
        self, ticker: str, *, timeframe: str = "quarterly", limit: int = 12
    ) -> list[FinancialStatement]:
        """Always raises :class:`CapabilityNotAvailable` — Alpaca has no financials.

        Alpaca's Market Data API serves prices, options and news; company
        filings are not in its catalogue at any subscription tier, which is
        exactly why data_source.md §2 keeps Massive for fundamentals. The
        refusal is explicit and names the gap so a caller running on Alpaca
        renders "Unavailable — reason" (§16, §44 rule 18); returning ``[]``
        would read as "this company filed nothing", which is a false claim
        about the company rather than an honest one about the provider.
        """
        raise CapabilityNotAvailable(
            "Alpaca does not serve company fundamentals at any subscription "
            f"tier — no statements for {(ticker or '').strip().upper() or '?'} "
            f"({timeframe}, limit {limit}). Configure the Massive provider for "
            "financial statements; there is NO synthetic fallback."
        )

    # ------------------------------------------------------------------
    # Capability probe (§16: detect, never fall back)
    # ------------------------------------------------------------------

    def probe_capabilities(self) -> dict[str, bool | str]:
        """Cheap live probes with the platform's EXACT capability keys.

        ``{"stock_history", "stock_realtime", "option_chain",
        "option_contracts", "news", "financials"}`` -> True (works), False
        (HTTP 403 — not in the subscription) or the error string (fault —
        availability unknown). §16: the answer to a missing capability is this
        report, never synthetic data.

        ``financials`` is the constant ``FINANCIALS_AVAILABLE`` (False): there
        is no Alpaca endpoint to probe, so the honest answer is known without
        a request. It is reported anyway so both providers answer the SAME key
        set and a caller never has to ask which provider it is holding.
        """
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=PROBE_HISTORY_DAYS)
        probes: dict[str, tuple[str, dict | None]] = {
            "stock_history": (
                f"{self.data_base_url}/v2/stocks/{PROBE_SYMBOL}/bars",
                {
                    "timeframe": "1Day",
                    "start": from_date.isoformat(),
                    "end": to_date.isoformat(),
                    "limit": PROBE_HISTORY_DAYS,
                },
            ),
            "stock_realtime": (
                f"{self.data_base_url}/v2/stocks/snapshots",
                {"symbols": PROBE_SYMBOL},
            ),
            "option_chain": (
                f"{self.data_base_url}/v1beta1/options/snapshots/{PROBE_SYMBOL}",
                {"limit": 10, "feed": self.options_feed},
            ),
            "option_contracts": (
                f"{self.trading_base_url}/v2/options/contracts",
                {"underlying_symbols": PROBE_SYMBOL, "limit": 1},
            ),
            "news": (f"{self.data_base_url}/v1beta1/news", {"limit": 1}),
        }
        report: dict[str, bool | str] = {}
        for name, (url, params) in probes.items():
            try:
                self._request(url, params=params)
            except CapabilityNotAvailable as exc:
                logger.warning("Alpaca capability %r unavailable: %s", name, exc)
                report[name] = False
            except MarketDataError as exc:
                report[name] = str(exc)
            else:
                report[name] = True
        report["financials"] = FINANCIALS_AVAILABLE
        return report
