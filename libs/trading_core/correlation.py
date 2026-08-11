"""Rolling correlation and dynamic correlation buckets (plan §12.4).

Pure, deterministic, dependency-free — plain-list inputs, hand-checkable
Pearson arithmetic. The static ``RiskLimits.correlation_buckets`` config is
the v0 mechanism; this module provides the DATA-DRIVEN alternative §12.4
calls for: measure rolling correlations of daily log returns and group
tickers whose correlation exceeds a threshold into shared-risk buckets.

Every parameter is explicit and tunable (house rule; §12.4: "thresholds
require validation" — the 0.70 / 60-day defaults are starting points to be
validated against history, never hardcoded truths):

- ``window`` (default 60 trading days): the number of most-recent aligned
  log returns the Pearson correlation is computed over.
- ``threshold`` (default 0.70): tickers with pairwise correlation STRICTLY
  above this share a bucket.

Honest nulls: insufficient history or a zero-variance return series yields
``None``, never a fabricated 0.0 correlation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def log_returns(closes: Sequence[float]) -> list[float]:
    """Daily log returns ``ln(close[t] / close[t-1])`` (plan §12.4).

    Returns a list one shorter than ``closes`` (no return exists for the
    first bar). Every close must be > 0 — log returns are undefined
    otherwise, and a silent fallback would poison the correlations.
    """
    for c in closes:
        if c <= 0:
            raise ValueError(f"closes must all be > 0, got {c}")
    return [
        math.log(closes[t] / closes[t - 1]) for t in range(1, len(closes))
    ]


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation of two equal-length samples; None on zero variance."""
    n = len(a)
    mean_a = math.fsum(a) / n
    mean_b = math.fsum(b) / n
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    var_a = math.fsum(x * x for x in da)
    var_b = math.fsum(y * y for y in db)
    if var_a <= 0.0 or var_b <= 0.0:
        return None  # zero variance -> correlation undefined (honest null)
    cov = math.fsum(x * y for x, y in zip(da, db))
    return cov / math.sqrt(var_a * var_b)


def rolling_correlation(
    a_closes: Sequence[float],
    b_closes: Sequence[float],
    window: int = 60,
) -> float | None:
    """Pearson correlation of the last ``window`` daily log returns (§12.4).

    The two close series are assumed bar-aligned at the RECENT end (both
    ending on the current bar); the trailing ``window`` returns of each are
    compared. Returns ``None`` — honest null, never a fake 0.0 — when either
    series has fewer than ``window + 1`` closes (insufficient data) or when
    either return window has zero variance (correlation undefined).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(a_closes) < window + 1 or len(b_closes) < window + 1:
        return None
    ra = log_returns(a_closes[-(window + 1):])
    rb = log_returns(b_closes[-(window + 1):])
    return _pearson(ra, rb)


def build_dynamic_buckets(
    closes_by_ticker: Mapping[str, Sequence[float]],
    threshold: float = 0.70,
    window: int = 60,
) -> list[frozenset[str]]:
    """Group tickers into correlation buckets via connected components (§12.4).

    Builds the graph whose edges connect ticker pairs with
    ``rolling_correlation > threshold`` (strict; ``None`` correlations —
    insufficient data or zero variance — never create an edge), then returns
    its connected components. Transitivity is deliberate: if A~B and B~C
    exceed the threshold, {A, B, C} share ONE bucket cap even if A~C does
    not, matching the shared-risk intent of §12.4.

    Singleton components are EXCLUDED — a ticker correlated with nothing
    needs no bucket cap beyond its single-name limits (§12.3).

    Deterministic output regardless of mapping iteration order: members are
    compared in sorted ticker order and buckets are sorted by their sorted
    member tuple.
    """
    tickers = sorted(closes_by_ticker)
    parent: dict[str, str] = {t: t for t in tickers}

    def find(t: str) -> str:
        root = t
        while parent[root] != root:
            root = parent[root]
        while parent[t] != root:  # path compression
            parent[t], t = root, parent[t]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic root choice: smaller ticker wins.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            corr = rolling_correlation(
                closes_by_ticker[a], closes_by_ticker[b], window=window
            )
            if corr is not None and corr > threshold:
                union(a, b)

    components: dict[str, set[str]] = {}
    for t in tickers:
        components.setdefault(find(t), set()).add(t)

    buckets = [
        frozenset(members)
        for members in components.values()
        if len(members) > 1  # singletons excluded
    ]
    buckets.sort(key=lambda bucket: tuple(sorted(bucket)))
    return buckets
