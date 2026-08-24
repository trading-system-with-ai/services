"""Polymarket adapter — PUBLIC READ APIs only (Catalyst research upgrade;
plan §3, Phases 3/11; LOOP 4).

Two official public surfaces, both keyless:

- **Gamma** (https://gamma-api.polymarket.com): discovery (`/public-search`),
  market metadata (`/markets/{id}`) — question, outcomes, resolution
  description, lifecycle, volume/liquidity.
- **CLOB** (https://clob.polymarket.com): pricing — order book (`/book`),
  last trade (`/last-trade-price`), dated price history (`/prices-history`).

READ ONLY, BY CONSTRUCTION: no wallet, no signing, no order placement, no
USDC, no private credentials — none of those surfaces exist in this class
and the research-safety word-ban test keeps it that way.

KEYLESS DOES NOT MEAN ANONYMOUS: construction requires a contact User-Agent
(the SEC EDGAR discipline) and calls are courtesy-paced. Reliability per
plan Phase 11: request timeout, 429 -> one capped Retry-After retry, one
jittered-backoff retry for transient faults (5xx / transport) on these safe
GETs, schema-tolerant parsing (Gamma serves JSON-encoded STRING arrays for
outcomes/outcomePrices/clobTokenIds), and PARTIAL failure handling — a dead
order book degrades the snapshot's bid/ask to None, it does not sink the
snapshot, and nothing here can break an event page (the gateway seam treats
PredictionMarketError as research degradation).

NO FABRICATION (§44 rule 18): unparseable prices, absent volume/liquidity
and missing history are ``None``/empty — never zero, never interpolated.
"""
import json
import logging
import math
import random
import re
import threading
import time as time_module
from datetime import datetime, timezone

import httpx

from .provider import (
    CAPABILITY_KEYS,
    MARKET_STATUS_ACTIVE,
    MARKET_STATUS_CLOSED,
    MARKET_STATUS_RESOLVED,
    MARKET_STATUS_UNKNOWN,
    CapabilityNotAvailable,
    MarketOutcome,
    MarketSnapshot,
    PredictionMarketError,
    PredictionMarketInfo,
    PricePoint,
    blank_capabilities,
)

logger = logging.getLogger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRY_AFTER_SECONDS = 1.0
#: Ceiling on an honored Retry-After (the Brave adapter's rationale: the
#: header is remote-controlled text; 'inf' or hours must not be obeyed).
MAX_RETRY_AFTER_SECONDS = 60.0
#: One transient-fault retry (5xx / transport) waits this base plus jitter.
TRANSIENT_RETRY_BASE_SECONDS = 0.5
#: Courtesy pacing between calls to a free public API.
MIN_REQUEST_INTERVAL_SECONDS = 0.2

#: History fidelity, in minutes (the CLOB parameter's unit). Hourly points
#: are plenty for inter-event repricing narratives and keep payloads small.
HISTORY_FIDELITY_MINUTES = 60

#: DAILY fidelity, for the long view. The venue caps a `startTs`/`endTs`
#: request by SPAN — 14 days at hourly, and daily fidelity does not extend it
#: ("interval is too long" past roughly a fortnight either way). The `interval`
#: parameter has no such cap, so the full life of a contract is reachable only
#: as `interval=max&fidelity=1440`: 265 daily points back to a market's first
#: trade, versus the 14 days the bounded form allows.
#:
#: This matters for reading a market rather than merely quoting it. A contract
#: that moved from 20c to 60c did so on a day, and that day is the question
#: worth asking; a fortnight of hourly points cannot show it if the move
#: happened last quarter.
HISTORY_FIDELITY_DAILY_MINUTES = 1440
HISTORY_INTERVAL_MAX = "max"

#: The widest span the bounded `startTs`/`endTs` form accepts. Probed against
#: the live venue (2026-08-22/23): hourly works at 14 days and 400s at 16, and
#: daily fidelity does NOT widen it. Past this, the request must switch to the
#: uncapped `interval` form above.
MAX_BOUNDED_HISTORY_DAYS = 14

#: /public-search discovery cap per call (the deterministic candidate pool
#: is bounded upstream too; this bounds the raw payload).
SEARCH_PAGE_LIMIT = 20


def _retry_after_seconds(response: httpx.Response, default: float) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(0.0, value), MAX_RETRY_AFTER_SECONDS)


def _as_float(raw: object) -> float | None:
    """Provider number fields arrive as numbers OR strings OR garbage.
    Unparseable -> None, NEVER 0 (§44 rule 18)."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(float(raw)) else None
    if isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    return None


def _first_float(item: dict, *keys: str) -> float | None:
    """The first key whose value PARSES — fallback on parse failure, never
    on truthiness: a provider-STATED 0 is a real number (absent != 0), and
    a garbage primary field must not block a valid secondary one."""
    for key in keys:
        if key in item:
            value = _as_float(item.get(key))
            if value is not None:
                return value
            # Key present but unparseable/absent-valued: try the next field.
    return None


def _as_aware_utc(value: datetime) -> datetime:
    """Naive instants are treated as UTC (the platform convention —
    prediction_intel/_brave share it): a naive bound must never be
    interpreted in the HOST's timezone, which would make the fetched
    history window machine-dependent."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


#: Gamma market ids are opaque tokens ([0-9A-Za-z_-]); anything else in a
#: path segment is either corruption or traversal-shaped and is refused.
_SAFE_MARKET_ID = re.compile(r"^[0-9A-Za-z_-]+$")


def _json_string_list(raw: object) -> list:
    """Gamma serves list fields (outcomes, outcomePrices, clobTokenIds) as
    JSON-ENCODED STRINGS. Accept a real list too; anything else -> []."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_end_date(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class PolymarketProvider:
    """PredictionMarketProvider over Polymarket's public Gamma + CLOB APIs."""

    name = "polymarket"

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_after_default_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        gamma_base_url: str = GAMMA_BASE_URL,
        clob_base_url: str = CLOB_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise PredictionMarketError(
                "Polymarket requires a contact User-Agent (set SEC_USER_AGENT "
                "— the shared operator contact string); anonymous scraping of "
                "a free public API is not how this platform behaves"
            )
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.clob_base_url = clob_base_url.rstrip("/")
        self.retry_after_default_seconds = retry_after_default_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._pace_lock = threading.Lock()
        self._last_request_at = 0.0
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent.strip(),
                "Accept": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover — close is best effort
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    # ------------------------------------------------------------------
    # Transport with the documented failure taxonomy (plan Phase 11)
    # ------------------------------------------------------------------

    def _pace(self) -> None:
        with self._pace_lock:
            now = time_module.monotonic()
            wait = self.min_request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                time_module.sleep(wait)
            self._last_request_at = time_module.monotonic()

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        """One Polymarket GET. Bounded retries for SAFE reads only:
        429 -> one capped Retry-After retry; transient fault (transport
        error or 5xx) -> one jittered-backoff retry; anything persisting
        raises :class:`PredictionMarketError` — research degradation, never
        a page failure."""
        self._pace()
        response: httpx.Response | None = None
        fault: str | None = None
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            fault = f"{type(exc).__name__}: {exc}"

        if response is not None and response.status_code == 429:
            delay = _retry_after_seconds(response, self.retry_after_default_seconds)
            logger.warning(
                "Polymarket rate limited (HTTP 429) on %s; retrying once in %.1fs",
                url, delay,
            )
            if delay > 0:
                time_module.sleep(delay)
            self._pace()
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                raise PredictionMarketError(
                    f"Polymarket retry failed for {url}: {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code == 429:
                raise PredictionMarketError(
                    f"Polymarket rate limit (HTTP 429) persisted after one "
                    f"retry for {url}"
                )
        elif fault is not None or (
            response is not None and response.status_code >= 500
        ):
            # Transient fault: one jittered-backoff retry (safe GET).
            reason = fault or f"HTTP {response.status_code}"
            delay = TRANSIENT_RETRY_BASE_SECONDS * (1.0 + random.random())
            logger.warning(
                "Polymarket transient fault on %s (%s); retrying once in %.2fs",
                url, reason, delay,
            )
            time_module.sleep(delay)
            self._pace()
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                raise PredictionMarketError(
                    f"Polymarket retry failed for {url}: {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code >= 500:
                raise PredictionMarketError(
                    f"Polymarket fault (HTTP {response.status_code}) persisted "
                    f"after one retry for {url}"
                )

        if response.status_code == 404:
            raise PredictionMarketError(
                f"Polymarket has no resource at {url} (HTTP 404)"
            )
        if response.status_code >= 400:
            raise PredictionMarketError(
                f"Polymarket returned HTTP {response.status_code} for {url}: "
                f"{response.text[:300]}"
            )
        return response

    def _json(self, response: httpx.Response, url: str):
        try:
            return response.json()
        except ValueError as exc:
            raise PredictionMarketError(
                f"Polymarket returned an unparseable body for {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Parsing — schema-tolerant, per-item failure isolation
    # ------------------------------------------------------------------

    def _parse_market(
        self, item: object, *, event_id: str | None = None
    ) -> PredictionMarketInfo:
        if not isinstance(item, dict):
            raise ValueError(f"market item is {type(item).__name__}, not dict")
        market_id = item.get("id")
        if market_id is None or not str(market_id).strip():
            raise ValueError("market item has no id")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("market item has no question")

        outcome_names = [
            str(name) for name in _json_string_list(item.get("outcomes"))
        ]
        outcome_prices = [
            _as_float(price) for price in _json_string_list(item.get("outcomePrices"))
        ]
        outcomes = tuple(
            MarketOutcome(
                name=name,
                price=outcome_prices[i] if i < len(outcome_prices) else None,
            )
            for i, name in enumerate(outcome_names)
        )

        active = item.get("active")
        closed = item.get("closed")
        resolved_marker = item.get("umaResolutionStatus")
        if isinstance(resolved_marker, str) and "resolved" in resolved_marker.lower():
            status = MARKET_STATUS_RESOLVED
        elif closed is True:
            status = MARKET_STATUS_CLOSED
        elif active is True:
            status = MARKET_STATUS_ACTIVE
        else:
            status = MARKET_STATUS_UNKNOWN

        # The venue's grouping id. /markets/{id} nests it under `events`;
        # a market inside a SEARCH response has no such key, so the caller
        # passes the id down from the event wrapper it was found in. Without
        # that path every stored row carried a NULL group and the platform
        # could not tell one distribution's brackets from another's.
        events = item.get("events")
        provider_event_id = event_id
        if provider_event_id is None and isinstance(events, list) and events:
            if isinstance(events[0], dict):
                raw_event_id = events[0].get("id")
                if raw_event_id is not None:
                    provider_event_id = str(raw_event_id)

        slug = item.get("slug")
        description = item.get("description")
        return PredictionMarketInfo(
            provider=self.name,
            market_id=str(market_id),
            provider_event_id=provider_event_id,
            question=question.strip(),
            url=(
                f"https://polymarket.com/market/{slug}"
                if isinstance(slug, str) and slug.strip() else None
            ),
            outcomes=outcomes,
            resolution_criteria=(
                description.strip()
                if isinstance(description, str) and description.strip() else None
            ),
            end_date=_parse_end_date(item.get("endDate")),
            status=status,
            volume=_first_float(item, "volumeNum", "volume"),
            liquidity=_first_float(item, "liquidityNum", "liquidity"),
            raw=item,
        )

    def _clob_token_ids(self, raw_market: dict) -> list[str]:
        return [str(t) for t in _json_string_list(raw_market.get("clobTokenIds"))]

    # ------------------------------------------------------------------
    # PredictionMarketProvider
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state probe (audit §6). Gamma probes cover discovery and
        metadata; the CLOB's public ``/time`` health check answers for both
        pricing capabilities (they share the host and auth-free access).
        Never raises."""
        report = blank_capabilities()

        def probe(key: str, url: str, params: dict | None = None) -> None:
            try:
                self._get(url, params)
            except CapabilityNotAvailable:
                report[key] = False
            except Exception as exc:
                report[key] = f"{type(exc).__name__}: {exc}"
            else:
                report[key] = True

        probe("market_search", f"{self.gamma_base_url}/public-search",
              {"q": "test"})
        probe("market_metadata", f"{self.gamma_base_url}/markets",
              {"limit": 1})
        probe("market_snapshot", f"{self.clob_base_url}/time")
        report["price_history"] = report["market_snapshot"]
        return report

    def search_markets(
        self,
        query: str,
        *,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[PredictionMarketInfo]:
        if limit <= 0:
            return []
        response = self._get(
            f"{self.gamma_base_url}/public-search",
            {"q": query, "limit_per_type": min(limit, SEARCH_PAGE_LIMIT)},
        )
        payload = self._json(response, "/public-search")
        if not isinstance(payload, dict):
            # A non-object body is a FAULT, never dressed up as the honest
            # NO_RELEVANT_MARKET empty answer — the two states must differ.
            raise PredictionMarketError(
                "Polymarket returned a non-object body for /public-search: "
                f"{type(payload).__name__}"
            )
        # WHOLE VENUE EVENTS, NEVER HALF OF ONE.
        #
        # Polymarket publishes a distribution as one contract per range, and
        # /public-search already returns them GROUPED under their event. The
        # previous code flattened that grouping away and stopped at `limit`
        # mid-event, so a seven-bracket GDP series could arrive as four
        # brackets — which is not a smaller answer, it is a WRONG one: the
        # brackets that survive are the ones the search happened to list
        # first, and reading them as a distribution understates wherever the
        # mass actually sits.
        #
        # So the unit of acceptance here is the EVENT: take events whole
        # while they fit, and stop before one that would not. `limit` becomes
        # a floor-shaped bound rather than a hard cut through a group.
        markets: list[PredictionMarketInfo] = []
        seen_ids: set[str] = set()
        events = payload.get("events")
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict):
                continue
            # The venue's own grouping id, carried from the WRAPPER — a
            # market nested in a search response has no `events` key of its
            # own, which is why every stored row had a NULL group until now.
            raw_event_id = event.get("id")
            event_id = str(raw_event_id) if raw_event_id is not None else None
            raw_markets = (
                event.get("markets")
                if isinstance(event.get("markets"), list)
                else []
            )
            group: list[PredictionMarketInfo] = []
            # Dedup WITHIN the group as well as across groups: the venue can
            # list the same market twice inside one event, and a duplicate
            # bracket would double-count that range in the distribution.
            group_ids: set[str] = set()
            for raw_market in raw_markets:
                try:
                    market = self._parse_market(raw_market, event_id=event_id)
                except ValueError as exc:
                    # One malformed market costs itself, never the search.
                    logger.warning(
                        "Polymarket search result unparseable; skipped: %s", exc
                    )
                    continue
                if market.market_id in seen_ids or market.market_id in group_ids:
                    continue
                if active_only and market.status != MARKET_STATUS_ACTIVE:
                    continue
                group_ids.add(market.market_id)
                group.append(market)
            if not group:
                continue
            # Stop BEFORE splitting a group. An event that does not fit is
            # left out entirely rather than half-admitted.
            if markets and len(markets) + len(group) > limit:
                break
            for market in group:
                seen_ids.add(market.market_id)
            markets.extend(group)
            if len(markets) >= limit:
                break
        return markets

    def get_market(self, market_id: str) -> PredictionMarketInfo:
        if not _SAFE_MARKET_ID.fullmatch(market_id or ""):
            # A traversal-shaped or garbled id must never reach the URL
            # path — refused in the taxonomy, not interpolated.
            raise PredictionMarketError(
                f"Polymarket market id {market_id!r} is not a valid id"
            )
        response = self._get(f"{self.gamma_base_url}/markets/{market_id}")
        payload = self._json(response, f"/markets/{market_id}")
        if isinstance(payload, list):  # tolerate list-wrapped answers
            payload = payload[0] if payload else None
        if not isinstance(payload, dict):
            raise PredictionMarketError(
                f"Polymarket returned no market object for id {market_id!r}"
            )
        try:
            return self._parse_market(payload)
        except ValueError as exc:
            raise PredictionMarketError(
                f"Polymarket market {market_id!r} is unparseable: {exc}"
            ) from exc

    def get_market_snapshot(self, market_id: str) -> MarketSnapshot:
        """Current pricing: Gamma outcome prices + CLOB book/last-trade for
        the FIRST outcome's token (the market's headline leg). PARTIAL
        failure handling (plan Phase 11): a dead book or missing last-trade
        degrades those fields to None — the snapshot survives on whatever
        Gamma still states."""
        info = self.get_market(market_id)
        raw = info.raw if isinstance(info.raw, dict) else {}
        tokens = self._clob_token_ids(raw)
        primary_token = tokens[0] if tokens else None

        best_bid = best_ask = midpoint = spread = last_trade = None
        if primary_token:
            try:
                book = self._json(
                    self._get(
                        f"{self.clob_base_url}/book",
                        {"token_id": primary_token},
                    ),
                    "/book",
                )
                if isinstance(book, dict):
                    # `or []` (not a .get default): a served "bids": null has
                    # the key PRESENT, and iterating None would sink the
                    # snapshot a dead book is only allowed to degrade.
                    bids = [
                        _as_float(level.get("price"))
                        for level in (book.get("bids") or [])
                        if isinstance(level, dict)
                    ]
                    asks = [
                        _as_float(level.get("price"))
                        for level in (book.get("asks") or [])
                        if isinstance(level, dict)
                    ]
                    bids = [b for b in bids if b is not None]
                    asks = [a for a in asks if a is not None]
                    best_bid = max(bids) if bids else None
                    best_ask = min(asks) if asks else None
                    if best_bid is not None and best_ask is not None:
                        midpoint = round((best_bid + best_ask) / 2.0, 4)
                        spread = round(best_ask - best_bid, 4)
            except PredictionMarketError as exc:
                logger.warning(
                    "Polymarket book unavailable for market %s: %s",
                    market_id, exc,
                )
            try:
                trade = self._json(
                    self._get(
                        f"{self.clob_base_url}/last-trade-price",
                        {"token_id": primary_token},
                    ),
                    "/last-trade-price",
                )
                if isinstance(trade, dict):
                    last_trade = _as_float(trade.get("price"))
            except PredictionMarketError as exc:
                logger.warning(
                    "Polymarket last trade unavailable for market %s: %s",
                    market_id, exc,
                )

        outcome_prices = {
            outcome.name: outcome.price
            for outcome in info.outcomes
            if outcome.price is not None  # unpriced = ABSENT, never 0
        }
        return MarketSnapshot(
            provider=self.name,
            market_id=info.market_id,
            observed_at=datetime.now(timezone.utc),
            outcome_prices=outcome_prices,
            best_bid=best_bid,
            best_ask=best_ask,
            midpoint=midpoint,
            spread=spread,
            last_trade_price=last_trade,
            volume=info.volume,
            liquidity=info.liquidity,
            open_interest=None,  # Polymarket's public read APIs do not state OI
        )

    def get_price_history(
        self,
        market_id: str,
        *,
        outcome: str,
        start: datetime,
        end: datetime,
    ) -> list[PricePoint]:
        # Naive bounds are UTC by platform convention — .timestamp() on a
        # naive datetime would interpret it in the HOST's timezone and make
        # the fetched window machine-dependent.
        start = _as_aware_utc(start)
        end = _as_aware_utc(end)
        if start >= end:
            return []
        info = self.get_market(market_id)
        raw = info.raw if isinstance(info.raw, dict) else {}
        tokens = self._clob_token_ids(raw)
        outcome_index = next(
            (
                i for i, o in enumerate(info.outcomes)
                if o.name.strip().lower() == outcome.strip().lower()
            ),
            None,
        )
        if outcome_index is None or outcome_index >= len(tokens):
            return []  # honest absence: no such outcome leg to chart
        # TWO SHAPES, ONE ENDPOINT. A bounded window keeps hourly detail; the
        # full history is only reachable through `interval`, which the venue
        # does not span-cap, and only at daily fidelity. Callers asking for
        # more than the bounded form allows get the long view rather than the
        # 400 the bounded form would return.
        span_days = (end - start).total_seconds() / 86400.0
        if span_days > MAX_BOUNDED_HISTORY_DAYS:
            params = {
                "market": tokens[outcome_index],
                "interval": HISTORY_INTERVAL_MAX,
                "fidelity": HISTORY_FIDELITY_DAILY_MINUTES,
            }
        else:
            params = {
                "market": tokens[outcome_index],
                "startTs": int(start.timestamp()),
                "endTs": int(end.timestamp()),
                "fidelity": HISTORY_FIDELITY_MINUTES,
            }
        response = self._get(f"{self.clob_base_url}/prices-history", params)
        payload = self._json(response, "/prices-history")
        history = payload.get("history") if isinstance(payload, dict) else None
        points: list[PricePoint] = []
        for item in history if isinstance(history, list) else []:
            if not isinstance(item, dict):
                continue
            price = _as_float(item.get("p"))
            ts_raw = item.get("t")
            # The timestamp gets the SAME guards as the price: bool is an
            # int subclass (`true` would fabricate a 1970 point), JSON NaN/
            # Infinity parse as floats, and an out-of-range epoch raises
            # from fromtimestamp — each malformed point costs only itself.
            if (
                price is None
                or isinstance(ts_raw, bool)
                or not isinstance(ts_raw, (int, float))
                or not math.isfinite(ts_raw)
            ):
                continue
            try:
                ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                continue
            points.append(PricePoint(ts=ts, price=price))
        points.sort(key=lambda p: p.ts)
        return points
