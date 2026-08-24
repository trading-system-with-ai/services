"""Deterministic SYNTHETIC web-search provider — development and tests only.

OPT-IN ONLY, NEVER A FALLBACK (§44 rule 18): this provider is reachable only
by explicitly setting ``WEB_SEARCH_PROVIDER=stub``. An unconfigured install
reports web research honestly unavailable rather than serving invented
results, and no code path substitutes the stub when Brave cannot answer.

Determinism: every field derives from ``zlib.crc32`` of the query (the same
seam :mod:`libs.llm.stub` uses), and every timestamp derives from the CALLER'S
window bounds. When the caller passes no bounds at all the window anchors on
the ``STUB_ANCHOR_DATE`` seam — frozen by the test harness (conftest), today
(UTC) in a live dev install — the exact discipline
``libs/market_data/stub.py::_default_series_end`` set, so tests are
byte-identical across runs while a dev install's results stay current.

The FULL synthetic shape needs ``limit >= 5`` (smaller limits truncate it):

- result index 2 carries ``published_at=None`` (provider omitted the
  publication time — the conservative-exclusion path);
- result indexes 3 and 4 share a canonical target (``.../dup``) through
  differently-decorated URLs, so the dedup layer has something real to fold;
- domains are obviously fake (``stub-web.example``) so a stub row that ever
  reached a real evidence table is immediately recognizable, exactly as the
  option-bar stub writes ``provider='stub'``.
"""
import zlib
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

from .provider import (
    CAPABILITY_KEYS,
    RESULT_TYPE_NEWS,
    RESULT_TYPE_WEB,
    SearchResult,
)

#: Span of the synthetic window when the caller leaves one bound open.
_ANCHOR_SPAN = timedelta(days=30)


def _anchor_end() -> datetime:
    """Midnight UTC of ``STUB_ANCHOR_DATE`` when set (the frozen test seam) —
    else now (UTC). Resolved at CALL time, never import time, so a test that
    changes settings is honored (the market-data stub's seam discipline)."""
    from libs.common.config import get_settings

    anchor = get_settings().stub_anchor_date
    if anchor:
        try:
            return datetime.combine(
                date.fromisoformat(anchor), time(0, 0), tzinfo=timezone.utc
            )
        except ValueError:
            pass  # a malformed anchor must never break the provider
    return datetime.now(timezone.utc)

#: Synthetic publishers, chosen by hash — plainly fake names.
_PUBLISHERS = ("Stub Wire", "Stub Business Daily", "Stub Markets Desk")


class StubWebSearchProvider:
    """Deterministic synthetic search results (see module docstring)."""

    name = "stub"

    def capabilities(self) -> dict[str, bool | str]:
        return {k: True for k in CAPABILITY_KEYS}

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
        return self._results(
            query, RESULT_TYPE_WEB, start_time, end_time, limit
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
        return self._results(
            query, RESULT_TYPE_NEWS, start_time, end_time, limit
        )

    def _results(
        self,
        query: str,
        result_type: str,
        start_time: datetime | None,
        end_time: datetime | None,
        limit: int,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        # Open bounds close around what the caller DID say: start-only spans
        # forward from start, end-only spans back from end, and only the
        # fully-unbounded call falls to the anchor — so a "results since X"
        # call can never collide with a fixed past anchor.
        if end_time is not None:
            end = end_time
        elif start_time is not None:
            end = start_time + _ANCHOR_SPAN
        else:
            end = _anchor_end()
        start = start_time or (end - _ANCHOR_SPAN)
        if start >= end:
            # A caller-inverted window is a caller bug surfaced honestly as
            # zero results, never as results stamped outside the window.
            return []
        seed = zlib.crc32(f"{result_type}|{query}".encode("utf-8"))
        count = min(limit, 5)
        span = (end - start) / (count + 1)
        results: list[SearchResult] = []
        for i in range(count):
            h = zlib.crc32(f"{seed}|{i}".encode("utf-8"))
            # Result index 2 omits its publication time — the provider-omitted
            # path the as-of gate must treat conservatively.
            published = None if i == 2 else start + span * (i + 1)
            # Indexes 3 and 4 point at the same target through different URLs
            # (query-string decoration) so downstream dedup has work to do.
            slug = "dup" if i >= 3 else f"doc-{h % 100_000}"
            decoration = f"?utm_source=stub{i}" if i == 4 else ""
            results.append(
                SearchResult(
                    provider=self.name,
                    provider_result_id=f"stub-{result_type}-{h}",
                    query=query,
                    title=f"[SYNTHETIC] {query} development {i + 1}",
                    url=f"https://stub-web.example/{seed % 1000}/{slug}{decoration}",
                    snippet=(
                        f"Synthetic stub coverage #{i + 1} for '{query}'. "
                        "Not real reporting; deterministic test data."
                    ),
                    publisher=_PUBLISHERS[h % len(_PUBLISHERS)],
                    published_at=published,
                    retrieved_at=end,
                    result_type=result_type,
                    rank=i,
                )
            )
        return results
