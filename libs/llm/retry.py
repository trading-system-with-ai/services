"""One-retry HTTP POST for the event-analysis provider calls (Phase 19.2).

The Catalyst research upgrade makes a single analysis request carry the whole
§46 bundle — web research and prediction markets included — so one transient
network hiccup or rate-limit response now throws away a materially expensive
assembly. This module adds the smallest useful remedy: ONE retry, after a
bounded backoff, on failures that are plausibly transient. It deliberately
does NOT grow into a general retry framework:

  - one retry, never a loop — the second failure is the answer;
  - only transport errors and rate-limit/server statuses retry; a 4xx is a
    request that will fail identically twice and is never retried;
  - ``Retry-After`` is honoured when it is a finite number of seconds, capped
    (an analysis call is a user-facing request, not a batch job — a server
    asking for a ten-minute wait gets the capped backoff, not obedience);
  - scope is the ``analyze_event`` calls ONLY. The discovery calls
    (``generate``/``enrich``) keep their fail-fast behavior: they run in
    scheduled batches that re-run anyway, and doubling their spend on a flaky
    day is worse than a missed cycle.

Lives in its own module (not ``provider.py``) because it imports httpx:
``provider.py`` is the pure interface the gateway may import on installs
whose HTTP stack is absent or mid-upgrade.
"""
import math
import time
from typing import Any, Callable

import httpx

from .provider import ProviderError

#: Statuses worth one retry: rate limit (429), transient server-side failures,
#: and 529 (Anthropic's documented "overloaded"). Everything else returns to
#: the caller unchanged, where the adapters' own status handling applies.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

#: Backoff before the single retry when the response names no usable
#: ``Retry-After``.
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

#: Cap on an honoured ``Retry-After``. Also the guard against the
#: ``Retry-After: inf`` family (isfinite below): a header must never be able
#: to park the request thread indefinitely.
MAX_RETRY_AFTER_SECONDS = 30.0

#: The default sleeper, as a module attribute so tests can replace it (and
#: record the chosen delay) without patching the stdlib for the whole process.
_sleep: Callable[[float], None] = time.sleep


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """A usable ``Retry-After`` delay in seconds, or None.

    Numeric-seconds form only (the HTTP-date form is not worth parsing for a
    single bounded retry); non-finite and negative values are ignored rather
    than obeyed.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def post_json_with_retry(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
    provider_name: str,
    sleep: Callable[[float], None] | None = None,
) -> httpx.Response:
    """POST ``payload``; retry ONCE on transport failure or a retryable status.

    Returns the final :class:`httpx.Response` — including a retryable-status
    response whose retry also failed; the caller's own non-200 handling turns
    that into its error. A transport failure on the retry raises
    :class:`ProviderError`. ``sleep`` is injectable for tests; None means
    :func:`time.sleep`.
    """
    # Module attribute resolved at call time — the injectable seam for tests.
    do_sleep = sleep if sleep is not None else _sleep
    response: httpx.Response | None = None
    for attempt in (0, 1):
        try:
            with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            if attempt == 0:
                do_sleep(DEFAULT_RETRY_BACKOFF_SECONDS)
                continue
            raise ProviderError(
                f"{provider_name} API request failed (after one retry): {exc!r}"
            ) from exc
        if response.status_code in RETRYABLE_STATUSES and attempt == 0:
            do_sleep(_retry_after_seconds(response) or DEFAULT_RETRY_BACKOFF_SECONDS)
            continue
        return response
    assert response is not None  # loop always ends on a returned response
    return response
