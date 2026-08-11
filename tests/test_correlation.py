"""Rolling correlation and dynamic bucket tests (development plan §12.4).

Synthetic price series with known return structure make every correlation
hand-checkable: scaled copies share IDENTICAL log returns (corr exactly 1),
exp-negated returns are exact mirrors (corr exactly -1), and independent
seeded walks must show only sampling noise.
"""
import math
import random

import pytest

from libs.trading_core.correlation import (
    build_dynamic_buckets,
    log_returns,
    rolling_correlation,
)


def walk(seed: int, n: int = 61, start: float = 100.0) -> list[float]:
    """Deterministic geometric random walk with ~2% daily vol."""
    rng = random.Random(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0, 0.02)))
    return closes


def from_returns(returns: list[float], start: float = 100.0) -> list[float]:
    """Rebuild a close series from log returns (inverse of log_returns)."""
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * math.exp(r))
    return closes


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------


def test_log_returns_hand_computed():
    # ln(110/100) and ln(121/110) = ln(1.1) twice.
    out = log_returns([100.0, 110.0, 121.0])
    assert out == pytest.approx([math.log(1.1), math.log(1.1)])
    assert len(out) == 2  # one shorter than the closes


def test_log_returns_rejects_nonpositive_closes():
    with pytest.raises(ValueError):
        log_returns([100.0, 0.0, 110.0])
    with pytest.raises(ValueError):
        log_returns([100.0, -5.0])


def test_log_returns_short_series():
    assert log_returns([100.0]) == []
    assert log_returns([]) == []


# ---------------------------------------------------------------------------
# rolling_correlation
# ---------------------------------------------------------------------------


def test_perfectly_correlated_scaled_copy_is_one():
    # b = 2 * a bar-for-bar -> identical log returns -> Pearson exactly 1.
    a = walk(seed=1)
    b = [2.0 * x for x in a]
    assert rolling_correlation(a, b, window=60) == pytest.approx(1.0)


def test_anti_correlated_mirror_is_minus_one():
    # b's log returns are the exact negation of a's -> Pearson exactly -1.
    a = walk(seed=2)
    b = from_returns([-r for r in log_returns(a)])
    assert rolling_correlation(a, b, window=60) == pytest.approx(-1.0)


def test_independent_walks_have_small_correlation():
    # Two independent seeded walks: only sampling noise remains. With 60
    # samples the standard error is ~1/sqrt(60) ~= 0.13, so |corr| < 0.35
    # is a comfortable deterministic bound for these fixed seeds.
    corr = rolling_correlation(walk(seed=11), walk(seed=97), window=60)
    assert corr is not None
    assert abs(corr) < 0.35


def test_insufficient_data_returns_none():
    # window=60 needs 61 closes for 60 returns; 60 closes is not enough.
    short = walk(seed=3, n=60)
    full = walk(seed=4, n=61)
    assert rolling_correlation(short, full, window=60) is None
    assert rolling_correlation(full, short, window=60) is None
    assert rolling_correlation([], full, window=60) is None


def test_zero_variance_returns_none():
    # Constant closes -> all-zero returns -> variance 0 -> honest None,
    # never a fabricated 0.0 correlation.
    flat = [100.0] * 61
    assert rolling_correlation(flat, walk(seed=5), window=60) is None
    assert rolling_correlation(walk(seed=5), flat, window=60) is None


def test_only_the_trailing_window_is_used():
    # Extra OLD history on one side must not change the result: series are
    # aligned at the recent end, and only the last `window` returns count.
    a = walk(seed=6)
    b = [2.0 * x for x in a]
    b_with_history = [55.0, 61.0, 58.0] + b
    assert rolling_correlation(a, b_with_history, window=60) == pytest.approx(
        1.0
    )


def test_window_must_be_at_least_two():
    with pytest.raises(ValueError):
        rolling_correlation(walk(seed=7), walk(seed=8), window=1)


# ---------------------------------------------------------------------------
# build_dynamic_buckets
# ---------------------------------------------------------------------------


def correlated_family() -> dict[str, list[float]]:
    """AAA/BBB/CCC share the same return stream (BBB scaled, CCC lightly
    noised); ZZZ is an independent walk."""
    base = walk(seed=42)
    rng = random.Random(7)
    noised = from_returns(
        [r + rng.gauss(0.0, 0.004) for r in log_returns(base)]
    )
    return {
        "AAA": base,
        "BBB": [3.0 * x for x in base],  # corr(AAA, BBB) = 1.0 exactly
        "CCC": noised,  # high but imperfect corr with AAA/BBB
        "ZZZ": walk(seed=1234),  # independent
    }


def test_three_correlated_plus_one_independent_gives_one_bucket_of_three():
    closes = correlated_family()
    # Sanity on the synthetic structure: family pairs above threshold,
    # the independent walk below it against every family member.
    assert rolling_correlation(closes["AAA"], closes["CCC"]) > 0.70
    for member in ("AAA", "BBB", "CCC"):
        corr = rolling_correlation(closes[member], closes["ZZZ"])
        assert corr is not None and corr <= 0.70

    buckets = build_dynamic_buckets(closes, threshold=0.70, window=60)
    # One bucket of the three correlated names; ZZZ is a singleton and
    # singletons are excluded.
    assert buckets == [frozenset({"AAA", "BBB", "CCC"})]


def test_threshold_is_strict_and_tunable():
    closes = correlated_family()
    # An impossible threshold (> 1) admits no edges: corr must be STRICTLY
    # above the threshold, and Pearson never exceeds 1.
    assert build_dynamic_buckets(closes, threshold=1.0, window=60) == []
    # A negative threshold links everything into one bucket of four.
    assert build_dynamic_buckets(closes, threshold=-1.0, window=60) == [
        frozenset({"AAA", "BBB", "CCC", "ZZZ"})
    ]


def test_insufficient_history_creates_no_edges():
    # 30 closes cannot support a 60-return window: every pairwise
    # correlation is None -> no edges -> no buckets (honest null, §12.4).
    a = walk(seed=9, n=30)
    closes = {"AAA": a, "BBB": [2.0 * x for x in a]}
    assert build_dynamic_buckets(closes, threshold=0.70, window=60) == []


def test_buckets_are_deterministic_regardless_of_input_order():
    closes = correlated_family()
    reversed_order = dict(reversed(list(closes.items())))
    assert build_dynamic_buckets(closes) == build_dynamic_buckets(
        reversed_order
    )
    # And repeated calls agree exactly.
    assert build_dynamic_buckets(closes) == build_dynamic_buckets(closes)


def test_two_disjoint_pairs_sorted_deterministically():
    a = walk(seed=21)
    c = walk(seed=77)
    closes = {
        "DDD": [2.0 * x for x in c],
        "AAA": a,
        "BBB": [1.5 * x for x in a],
        "CCC": c,
    }
    # Sanity: the cross-pair correlation stays below threshold.
    corr = rolling_correlation(a, c)
    assert corr is not None and corr <= 0.70
    buckets = build_dynamic_buckets(closes, threshold=0.70, window=60)
    # Sorted by sorted member tuple: {AAA,BBB} before {CCC,DDD}.
    assert buckets == [
        frozenset({"AAA", "BBB"}),
        frozenset({"CCC", "DDD"}),
    ]
