"""Brave Search API adapter (Catalyst research upgrade; plan §2, LOOP 2).

The real web-search provider, backed by Brave's official Search API
(https://api.search.brave.com). Keyed: ``BRAVE_API_KEY`` authenticates via
the ``X-Subscription-Token`` header and is NEVER logged, echoed in an error,
or sent anywhere but api.search.brave.com over TLS.

Transport discipline (the Alpaca adapter's shape, both patterns combined):

- ONE ``_request()`` chokepoint every call goes through — 429 answers one
  Retry-After retry then raises; 401 names the env var without echoing the
  key; 403 is :class:`CapabilityNotAvailable` (the plan does not include the
  endpoint — e.g. the news vertical on a lower tier), never a fault dressed
  as data.
- PROACTIVE pacing (the SEC EDGAR shape): Brave's entry plans allow ~1
  request/second, so calls are spaced at least
  :data:`MIN_REQUEST_INTERVAL_SECONDS` apart instead of provoking 429s and
  apologising.

RESULTS ARE UNTRUSTED TEXT, RETURNED VERBATIM (provider contract): titles
and snippets go to the pure research layer for sanitization; nothing here
interprets them.

TIME HONESTY (§44 rule 18): ``published_at`` comes only from Brave's
``page_age`` timestamp. Brave's human-relative ``age`` string ("2 days ago")
is NEVER parsed into a fake instant, and a result with no ``page_age`` keeps
``published_at=None`` — the as-of gate downstream treats that absence
conservatively. Brave's ``freshness`` window is sent as a HINT when the
caller bounds the search, but the deterministic ``published_at <= as_of``
gate in the research layer is the real look-ahead enforcement, never this
parameter.
"""
import logging
import math
import threading
import time as time_module
from datetime import datetime, timezone
from typing import Sequence

import httpx

from .provider import (
    CapabilityNotAvailable,
    RESULT_TYPE_NEWS,
    RESULT_TYPE_WEB,
    SearchResult,
    WebSearchError,
    blank_capabilities,
)

logger = logging.getLogger(__name__)

BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRY_AFTER_SECONDS = 1.0

#: Proactive pacing: Brave's entry plans allow ~1 request/second. Spacing
#: calls slightly wider than 1s keeps the platform inside the published
#: limit instead of relying on 429 recovery (the SEC EDGAR discipline).
MIN_REQUEST_INTERVAL_SECONDS = 1.05

#: Server-side per-request result caps (Brave's documented maxima).
MAX_COUNT_WEB = 20
MAX_COUNT_NEWS = 50

#: Domain HINTS folded into the query string (``site:``/``-site:``) are
#: capped so a long domain list cannot blow Brave's query-length limit. The
#: hint is best-effort; the research layer's deterministic domain policy over
#: the RESULTS is the enforcement point (provider contract).
MAX_DOMAIN_HINTS = 3


#: Ceiling on an honored Retry-After. The header is server/proxy-controlled
#: text: 'inf' would make time.sleep raise OverflowError out of the failure
#: taxonomy, and a huge finite value would block the worker for hours on a
#: header nobody audited. Beyond the cap, the default backoff is the answer.
MAX_RETRY_AFTER_SECONDS = 60.0


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


def _parse_page_age(raw: object) -> datetime | None:
    """Brave's ``page_age`` timestamp as an aware-UTC instant, or None.

    Brave serves it zoneless (e.g. ``2026-08-12T09:30:00``); it is treated as
    UTC — the provider's own convention. Anything unparseable is an honest
    ``None``, never a guess (§44 rule 18).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc_date(value: datetime) -> str:
    """The bound's UTC calendar date. A NAIVE bound is treated as UTC — the
    same convention :func:`_parse_page_age` applies — never as host-local
    time, which would make the freshness window drift with the machine's
    timezone and silently narrow/widen what Brave returns."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date().isoformat()


def _freshness_param(
    start_time: datetime | None, end_time: datetime | None
) -> str | None:
    """Brave's date-range ``freshness`` value for the caller's bounds, or
    None when unbounded. End-only bounds are inexpressible in this parameter
    and are omitted — the research layer's as-of gate does the real work."""
    if start_time is None:
        return None
    end = end_time or datetime.now(timezone.utc)
    return f"{_as_utc_date(start_time)}to{_as_utc_date(end)}"


def _with_domain_hints(
    query: str,
    domains: Sequence[str] | None,
    exclude_domains: Sequence[str] | None,
) -> str:
    parts = [query]
    for domain in list(domains or [])[:MAX_DOMAIN_HINTS]:
        parts.append(f"site:{domain}")
    for domain in list(exclude_domains or [])[:MAX_DOMAIN_HINTS]:
        parts.append(f"-site:{domain}")
    return " ".join(parts)


class BraveSearchProvider:
    """WebSearchProvider backed by the Brave Search API. Real results only."""

    name = "brave"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_after_default_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise WebSearchError(
                "Brave Search requires an API key — set BRAVE_API_KEY (the "
                "key is never logged or echoed)"
            )
        self.retry_after_default_seconds = retry_after_default_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._pace_lock = threading.Lock()
        self._last_request_at = 0.0
        self._client = httpx.Client(
            base_url=BRAVE_API_BASE_URL,
            headers={
                "X-Subscription-Token": api_key.strip(),
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
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
    # Transport with the documented failure taxonomy
    # ------------------------------------------------------------------

    def _pace(self) -> None:
        """Space calls MIN_REQUEST_INTERVAL_SECONDS apart (SEC EDGAR shape)."""
        with self._pace_lock:
            now = time_module.monotonic()
            wait = self.min_request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                time_module.sleep(wait)
            self._last_request_at = time_module.monotonic()

    def _request(self, path: str, params: dict) -> httpx.Response:
        """One Brave call. 429 -> one Retry-After retry; 401/403 -> the
        documented taxonomy; anything else >= 400 raises with the body
        excerpt (the key itself can never appear — it lives in a header)."""
        self._pace()
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise WebSearchError(
                f"Brave request failed for {path}: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code == 429:
            delay = _retry_after_seconds(response, self.retry_after_default_seconds)
            logger.warning(
                "Brave rate limited (HTTP 429) on %s; retrying once in %.1fs",
                path, delay,
            )
            if delay > 0:
                time_module.sleep(delay)
            self._pace()
            try:
                response = self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                raise WebSearchError(
                    f"Brave retry failed for {path}: {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code == 429:
                raise WebSearchError(
                    f"Brave rate limit (HTTP 429) persisted after one retry "
                    f"for {path}"
                )

        if response.status_code == 401:
            raise WebSearchError(
                f"Brave rejected the API key (HTTP 401) for {path} — check "
                "BRAVE_API_KEY (the key is never logged or echoed)"
            )
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"Brave returned HTTP 403 for {path}: the subscription does "
                "not include this endpoint. There is NO synthetic fallback: "
                f"{response.text[:300]}"
            )
        if response.status_code >= 400:
            raise WebSearchError(
                f"Brave API returned HTTP {response.status_code} for {path}: "
                f"{response.text[:300]}"
            )
        return response

    # ------------------------------------------------------------------
    # WebSearchProvider
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state probe (audit §6): one minimal request per vertical.
        True = works; False = HTTP 403 (proven absence — not in the plan);
        an error string = fault, availability unknown. Never raises."""
        report = blank_capabilities()
        probes = {
            "web_search": ("/web/search", {"q": "test", "count": 1}),
            "news_search": ("/news/search", {"q": "test", "count": 1}),
        }
        for key, (path, params) in probes.items():
            try:
                self._request(path, params)
            except CapabilityNotAvailable:
                report[key] = False
            except Exception as exc:
                report[key] = f"{type(exc).__name__}: {exc}"
            else:
                report[key] = True
        return report

    def search_web(
        self,
        query: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        q = _with_domain_hints(query, domains, exclude_domains)
        params: dict = {"q": q, "count": min(limit, MAX_COUNT_WEB)}
        freshness = _freshness_param(start_time, end_time)
        if freshness:
            params["freshness"] = freshness
        if country:
            params["country"] = country
        if language:
            params["search_lang"] = language
        response = self._request("/web/search", params)
        payload = self._json(response, "/web/search")
        raw_results = payload.get("web", {})
        if not isinstance(raw_results, dict):
            raw_results = {}
        # Bound by what was actually asked of the server — a server that
        # over-serves must not widen what the platform accepts.
        return self._parse_results(
            raw_results.get("results"), query=query, result_type=RESULT_TYPE_WEB,
            limit=params["count"],
        )

    def search_news(
        self,
        query: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        q = _with_domain_hints(query, domains, exclude_domains)
        params: dict = {"q": q, "count": min(limit, MAX_COUNT_NEWS)}
        freshness = _freshness_param(start_time, end_time)
        if freshness:
            params["freshness"] = freshness
        if country:
            params["country"] = country
        if language:
            params["search_lang"] = language
        response = self._request("/news/search", params)
        payload = self._json(response, "/news/search")
        # The news vertical serves results at the top level, unlike /web.
        # Bounded by what was asked of the server (see search_web).
        return self._parse_results(
            payload.get("results"), query=query, result_type=RESULT_TYPE_NEWS,
            limit=params["count"],
        )

    # ------------------------------------------------------------------
    # Parsing — degrade to fewer results, never crash on one bad item
    # ------------------------------------------------------------------

    def _json(self, response: httpx.Response, path: str) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                f"Brave returned an unparseable body for {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise WebSearchError(
                f"Brave returned a non-object body for {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    def _parse_results(
        self, raw: object, *, query: str, result_type: str, limit: int
    ) -> list[SearchResult]:
        if not isinstance(raw, list):
            return []
        retrieved_at = datetime.now(timezone.utc)
        results: list[SearchResult] = []
        for rank, item in enumerate(raw):
            if len(results) >= limit:
                break
            try:
                results.append(
                    self._parse_item(
                        item, query=query, result_type=result_type,
                        rank=rank, retrieved_at=retrieved_at,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                # One malformed item costs itself, never the response
                # (the openai/alpaca per-entry discipline).
                logger.warning(
                    "Brave %s result %d unparseable; skipped: %s",
                    result_type, rank, exc,
                )
        return results

    def _parse_item(
        self,
        item: object,
        *,
        query: str,
        result_type: str,
        rank: int,
        retrieved_at: datetime,
    ) -> SearchResult:
        if not isinstance(item, dict):
            raise TypeError(f"result item is {type(item).__name__}, not dict")
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("result has no url")
        title = item.get("title")
        snippet = item.get("description")
        meta_url = item.get("meta_url")
        hostname = (
            meta_url.get("hostname") if isinstance(meta_url, dict) else None
        )
        profile = item.get("profile")
        profile_name = (
            profile.get("name") if isinstance(profile, dict) else None
        )
        source = item.get("source") if isinstance(item.get("source"), str) else None
        publisher = profile_name or source or hostname
        return SearchResult(
            provider=self.name,
            provider_result_id=None,  # Brave assigns no stable result ids
            query=query,
            title=title if isinstance(title, str) else "",
            url=url.strip(),
            snippet=snippet if isinstance(snippet, str) else "",
            publisher=publisher if isinstance(publisher, str) else None,
            published_at=_parse_page_age(item.get("page_age")),
            retrieved_at=retrieved_at,
            result_type=result_type,
            rank=rank,
        )
