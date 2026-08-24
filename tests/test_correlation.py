"""Rolling correlation and dynamic bucket tests (development plan §12.4).

Synthetic price series with known return structure make every correlation
hand-checkable: scaled copies share IDENTICAL log returns (corr exactly 1),
exp-negated returns are exact mirrors (corr exactly -1), and independent
seeded walks must show only sampling noise.
"""
import dataclasses
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


# ---------------------------------------------------------------------------
# Correlation regime (risk spec §19; Phase B design contract §7.4)
#
# Canonical fixture — 2 tickers x 24 daily LOG returns, split in half:
#
#   OLDER 12 (t = 0..11), exact mirrors -> pairwise correlation EXACTLY -1
#     AAA :  0.01 -0.02  0.03 -0.01  0.02 -0.03  0.01 -0.02  0.03 -0.01  0.02 -0.03
#     BBB : -0.01  0.02 -0.03  0.01 -0.02  0.03 -0.01  0.02 -0.03  0.01 -0.02  0.03
#
#   RECENT 12 (t = 12..23), IDENTICAL columns -> correlation EXACTLY +1
#     both: -0.05 -0.04 -0.03 -0.02 -0.01  0.01  0.02  0.03  0.04  0.05 -0.06 -0.07
#
# With long_window=24 / short_window=12 the three averages are (each one
# hand-verified against the plain Pearson formula, not just against the code):
#
#   normal_avg  (all 24 obs) = 0.5409052092228864
#   current_avg (last 12)    = 1.0            <- diversification has failed
#   delta                    = 0.4590947907771136
#
# Stress window at stress_quantile=0.5: the equal-weight portfolio return is
# (AAA + BBB)/2, which is 0.0 on every OLDER day (the mirrors cancel) and
# equals the shared value on every RECENT day. The 12 worst days are
# therefore the 5 negative recent days (t=12..16), the 2 most negative
# recent days (t=22, 23), and 5 of the zero-valued older days broken by DATE
# ORDER (t=0..4) -> indices [0,1,2,3,4,12,13,14,15,16,22,23], and
#
#   stress_avg = 0.6141866197906573
#
# delta 0.459 >= converging_delta 0.15 AND current 1.0 >= converging_level
# 0.80  ->  state CONVERGING (the §19 "0.61 -> 0.84" shape, exaggerated so
# every number is exact).
# ---------------------------------------------------------------------------

from datetime import date, timedelta  # noqa: E402

from libs.trading_core.correlation import (  # noqa: E402
    STATE_CONVERGING,
    STATE_ELEVATED,
    STATE_NORMAL,
    STATE_UNAVAILABLE,
    CorrelationRegimeParams,
    CorrelationState,
    correlation_regime,
)
from libs.trading_core.risk.returns import ReturnMatrix  # noqa: E402

OLD_A = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.02, -0.03]
OLD_B = [-0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03]
NEW = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, -0.06, -0.07]

REGIME_A = OLD_A + NEW
REGIME_B = OLD_B + list(NEW)

NORMAL_AVG = 0.5409052092228864
STRESS_AVG = 0.6141866197906573

# long_window covers the whole fixture; short_window is the recent half.
REGIME_PARAMS = CorrelationRegimeParams(
    long_window=24, short_window=12, stress_quantile=0.5, min_stress_obs=10
)


def regime_matrix(
    columns: dict[str, list[float]] | None = None,
    return_type: str = "LOG",
) -> ReturnMatrix:
    """LOG return matrix over consecutive dates from the given columns."""
    columns = columns or {"AAA": REGIME_A, "BBB": REGIME_B}
    tickers = tuple(columns)
    n = len(next(iter(columns.values())))
    return ReturnMatrix(
        dates=tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(n)),
        tickers=tickers,
        rows=tuple(zip(*(columns[t] for t in tickers))),
        return_type=return_type,
    )


def test_correlation_regime_is_hand_checked_end_to_end():
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    assert isinstance(state, CorrelationState)
    assert state.normal_avg == pytest.approx(NORMAL_AVG)
    assert state.current_avg == 1.0  # identical columns: exactly 1
    assert state.delta == pytest.approx(1.0 - NORMAL_AVG)
    assert state.stress_avg == pytest.approx(STRESS_AVG)
    assert state.state == STATE_CONVERGING
    assert state.n_pairs == 1
    assert state.n_obs_long == 24
    assert state.n_obs_short == 12
    assert state.n_obs_stress == 12
    assert state.worst_pairs == (("AAA", "BBB", 1.0),)
    assert "converging level" in state.reason and "regime shift" in state.reason
    assert state.is_available is True


def test_stress_window_is_the_worst_days_of_the_equal_weight_portfolio():
    """Contract §7.4: the stress sample is conditioned on portfolio losses,
    not on the calendar — a different sample from the trailing window."""
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    # 24 obs x 0.5 = 12 stress days, and their correlation (0.614) sits
    # between the full-sample 0.541 and the recent 1.0 — a real third view.
    assert state.n_obs_stress == 12
    assert state.stress_avg == pytest.approx(STRESS_AVG)
    assert state.normal_avg < state.stress_avg < state.current_avg


def test_stress_average_is_none_when_the_stress_sample_is_too_small():
    """Honest null (contract §7.4: '>= 10 days else None') — the rest of the
    state still stands; only the number that could not be computed is None."""
    params = CorrelationRegimeParams(
        long_window=24, short_window=12, stress_quantile=0.2, min_stress_obs=10
    )
    # 24 x 0.2 = 4 stress days < 10.
    state = correlation_regime(regime_matrix(), params=params)
    assert state.stress_avg is None
    assert state.n_obs_stress == 4
    assert state.current_avg == 1.0  # everything else is unaffected
    assert state.state == STATE_CONVERGING


def test_state_is_normal_when_correlation_has_not_moved():
    """Identical windows -> delta 0 -> NORMAL."""
    stable = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03] * 4
    # A companion that is NOT highly correlated (rho ~ -0.65): under the
    # revised rule a persistently HIGH correlation is CONVERGING even with
    # delta 0, so "not moved" must be tested on a diversified pair.
    other = [-0.01, 0.03, 0.005, 0.02, -0.03, 0.01] * 4
    state = correlation_regime(
        regime_matrix({"AAA": stable, "BBB": other}), params=REGIME_PARAMS
    )
    assert state.delta == pytest.approx(0.0)
    assert state.current_avg < REGIME_PARAMS.converging_level
    assert state.state == STATE_NORMAL
    assert state.reason is None


def test_state_is_elevated_between_the_two_deltas():
    """A jump past ``elevated_delta`` while the CURRENT level stays below
    ``converging_level`` is ELEVATED — diversification weakened, not gone.
    (Revised 2026-08-18: the level alone decides CONVERGING.)"""
    partial_b = OLD_B + [
        -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07,
    ]
    params = CorrelationRegimeParams(
        long_window=24,
        short_window=12,
        stress_quantile=0.5,
        elevated_delta=0.05,
        converging_delta=0.90,  # jump 0.314 stays below -> no "regime shift" note
        converging_level=0.80,
    )
    state = correlation_regime(
        regime_matrix({"AAA": REGIME_A, "BBB": partial_b}), params=params
    )
    assert state.delta >= params.elevated_delta
    assert state.current_avg < params.converging_level
    assert state.state == STATE_ELEVATED
    assert "elevated delta" in (state.reason or "")


def test_a_big_jump_that_lands_below_the_level_is_only_elevated():
    """Contract §7.4 (revised 2026-08-18): CONVERGING = current level at or
    above ``converging_level``; the jump annotates but does not gate.

    A big JUMP that still lands somewhere diversified is only ELEVATED. Here
    BBB tracks AAA on the recent window except for its last two days, which
    is enough to pull the current average down to 0.2160294795025333 while
    the delta (0.3142361404760602 over a normal average of
    -0.09820666097352687) still clears ``converging_delta`` easily. Level,
    not just movement, is what separates the two states.
    """
    partial_b = OLD_B + [
        -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07,  # diverges from AAA exactly here
    ]
    state = correlation_regime(
        regime_matrix({"AAA": REGIME_A, "BBB": partial_b}),
        params=REGIME_PARAMS,
    )
    assert state.normal_avg == pytest.approx(-0.09820666097352687)
    assert state.current_avg == pytest.approx(0.2160294795025333)
    assert state.delta == pytest.approx(0.3142361404760602)
    # delta 0.314 >= converging_delta 0.15, but current 0.216 < level 0.80.
    assert state.delta >= REGIME_PARAMS.converging_delta
    assert state.current_avg < REGIME_PARAMS.converging_level
    assert state.state == STATE_ELEVATED
    assert "regime shift" in (state.reason or "")


def test_persistently_correlated_book_is_converging_without_a_jump():
    """QA finding (2026-08-18): a book that has ALWAYS moved together has no
    diversification — spec §19's failure mode — and must read CONVERGING
    even though ``delta`` ≈ 0. Under the earlier "jump AND level" rule it
    read NORMAL."""
    same = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.012, -0.018] * 4
    scaled = [2.0 * r for r in same]  # perfectly correlated, different scale
    state = correlation_regime(
        regime_matrix({"AAA": same, "BBB": scaled}), params=REGIME_PARAMS
    )
    assert state.current_avg == pytest.approx(1.0)
    assert state.normal_avg == pytest.approx(1.0)
    assert abs(state.delta) < 1e-9
    assert state.state == STATE_CONVERGING
    assert "converging level" in (state.reason or "")


def test_regime_requires_log_returns():
    """Mixing return conventions is malformed input, not a data gap (§1)."""
    with pytest.raises(ValueError, match="requires 'LOG' returns"):
        correlation_regime(regime_matrix(return_type="SIMPLE"), params=REGIME_PARAMS)


def test_regime_unavailable_below_the_short_window():
    matrix = regime_matrix(
        {"AAA": REGIME_A[:8], "BBB": REGIME_B[:8]}
    )
    state = correlation_regime(matrix, params=REGIME_PARAMS)
    assert state.state == STATE_UNAVAILABLE
    assert state.reason == "n=8 < short_window=12"
    assert state.normal_avg is None
    assert state.current_avg is None
    assert state.stress_avg is None
    assert state.delta is None
    assert state.worst_pairs == ()
    assert state.is_available is False


def test_regime_unavailable_with_a_single_ticker():
    state = correlation_regime(
        regime_matrix({"AAA": REGIME_A}), params=REGIME_PARAMS
    )
    assert state.state == STATE_UNAVAILABLE
    assert "n_tickers=1 < 2" in state.reason


def test_regime_unavailable_when_every_pair_has_zero_variance():
    """A flat series has no correlation — honest null, never a fake 0.0."""
    flat = [0.0] * 24
    state = correlation_regime(
        regime_matrix({"AAA": flat, "BBB": list(REGIME_B)}), params=REGIME_PARAMS
    )
    assert state.state == STATE_UNAVAILABLE
    assert "n_pairs=0" in state.reason
    assert state.current_avg is None


def test_worst_pairs_are_the_top_three_by_current_correlation():
    """Display budget: at most three rows, highest current correlation first."""
    base = NEW * 2  # 24 obs
    columns = {
        "AAA": base,
        "BBB": list(base),                       # rho(AAA, BBB) = +1
        "CCC": [-x for x in base],               # rho with AAA/BBB = -1
        "DDD": [x * 2 for x in base],            # rho = +1 with AAA and BBB
    }
    state = correlation_regime(regime_matrix(columns), params=REGIME_PARAMS)
    # 4 tickers -> 6 upper-triangle pairs; only the best 3 are named.
    assert state.n_pairs == 6
    assert len(state.worst_pairs) == 3
    rhos = [rho for _, _, rho in state.worst_pairs]
    assert rhos == sorted(rhos, reverse=True)
    assert all(rho == pytest.approx(1.0) for rho in rhos)
    # Ties are broken by ticker order, deterministically.
    assert state.worst_pairs[0][:2] == ("AAA", "BBB")


def test_regime_averages_the_upper_triangle_only():
    """Each unordered pair counted ONCE: 3 tickers -> 3 pairs, and the mean
    of (+1, -1, -1) is -1/3."""
    base = NEW * 2
    columns = {"AAA": base, "BBB": list(base), "CCC": [-x for x in base]}
    state = correlation_regime(regime_matrix(columns), params=REGIME_PARAMS)
    assert state.n_pairs == 3
    assert state.current_avg == pytest.approx(-1.0 / 3.0)


def test_regime_params_validate_their_inputs():
    for bad in (
        {"long_window": 1},
        {"short_window": 1},
        {"min_stress_obs": 1},
        {"min_pairs": 0},
        {"stress_quantile": 0.0},
        {"stress_quantile": 1.0},
        {"converging_level": 1.5},
        {"long_window": 10, "short_window": 20},  # short must fit inside long
    ):
        with pytest.raises(ValueError):
            CorrelationRegimeParams(**bad)


def test_regime_defaults_are_the_documented_research_values():
    """Every threshold a parameter (house rule); defaults are UNVALIDATED."""
    p = CorrelationRegimeParams()
    assert (p.long_window, p.short_window) == (250, 60)
    assert p.stress_quantile == 0.10
    assert (p.elevated_delta, p.converging_delta) == (0.05, 0.15)
    assert p.converging_level == 0.80
    assert (p.min_pairs, p.min_stress_obs) == (1, 10)


def test_normal_window_uses_all_history_when_shorter_than_long_window():
    """``long_window`` is a MAXIMUM; ``n_obs_long`` says what was really used."""
    params = CorrelationRegimeParams(
        long_window=250, short_window=12, stress_quantile=0.5, min_stress_obs=10
    )
    state = correlation_regime(regime_matrix(), params=params)
    assert state.n_obs_long == 24
    assert state.normal_avg == pytest.approx(NORMAL_AVG)


def test_correlation_module_imports_cleanly_in_both_orders():
    """Contract §7.4 / the import-cycle note: ``correlation`` imports
    ``risk.returns`` and ``risk`` re-exports modules that read correlation
    concepts, so BOTH import orders must work in a fresh interpreter."""
    import subprocess
    import sys

    for first, second in (
        ("libs.trading_core.correlation", "libs.trading_core.risk"),
        ("libs.trading_core.risk", "libs.trading_core.correlation"),
    ):
        code = (
            f"import {first}; import {second}; "
            "from libs.trading_core.correlation import correlation_regime; "
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, f"{first} then {second}: {out.stderr}"
        assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Rolling Spearman (risk spec §18; compliance §3 row 18) — RESEARCH display
#
# Every value below is hand-computed from the rank definition, not read off
# the implementation:
#
#   y = x^3 on x = 1..5 is STRICTLY MONOTONE but NON-LINEAR. Both rank
#   vectors are [1,2,3,4,5], so Spearman is EXACTLY 1.0 while Pearson is
#   0.9431175138077005 (< 1) — the gap is the whole point of §18.
#
#   Ties: x = [1, 2, 2, 4] ranks as [1, 2.5, 2.5, 4] (the tied block shares
#   its average rank), y = [10, 20, 30, 40] ranks as [1, 2, 3, 4]. Then
#   mean_x = mean_y = 2.5, cov = 4.5, var_x = 4.5, var_y = 5.0, so
#   rho = 4.5 / sqrt(4.5 * 5) = 3 / sqrt(10) = 0.9486832980505138.
# ---------------------------------------------------------------------------

from libs.trading_core.correlation import (  # noqa: E402
    _average_ranks,
    _pearson,
    rolling_spearman,
    rolling_spearman_average,
    rolling_spearman_matrix,
    spearman,
)

CUBE_X = [1.0, 2.0, 3.0, 4.0, 5.0]
CUBE_Y = [1.0, 8.0, 27.0, 64.0, 125.0]
CUBE_PEARSON = 0.9431175138077005
TIE_RHO = 3.0 / math.sqrt(10.0)


def test_average_ranks_share_the_average_of_the_tied_block():
    assert _average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]
    # Order of arrival must not matter: the same multiset ranks the same way.
    assert _average_ranks([20.0, 30.0, 10.0, 20.0]) == [2.5, 4.0, 1.0, 2.5]
    # An all-tied series gives every element the same rank -> zero variance.
    assert _average_ranks([7.0, 7.0, 7.0]) == [2.0, 2.0, 2.0]


def test_spearman_is_one_for_a_monotone_nonlinear_relation_while_pearson_is_not():
    """§18's reason to exist: y = x^3 co-moves perfectly but non-linearly."""
    assert spearman(CUBE_X, CUBE_Y) == 1.0  # exact: identical rank vectors
    assert _pearson(CUBE_X, CUBE_Y) == pytest.approx(CUBE_PEARSON)
    assert _pearson(CUBE_X, CUBE_Y) < 1.0


def test_spearman_is_minus_one_for_a_monotone_decreasing_relation():
    assert spearman(CUBE_X, list(reversed(CUBE_Y))) == -1.0


def test_spearman_ties_use_average_ranks_hand_computed():
    """[1,2,2,4] vs [10,20,30,40] -> 3/sqrt(10), NOT the no-ties shortcut."""
    assert spearman([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        TIE_RHO
    )
    # The 1 - 6*sum(d^2)/(n(n^2-1)) shortcut would give a DIFFERENT number
    # here; proving they differ is what pins the general definition.
    d2 = sum((a - b) ** 2 for a, b in zip([1, 2.5, 2.5, 4], [1, 2, 3, 4]))
    shortcut = 1.0 - 6.0 * d2 / (4 * (4**2 - 1))
    assert shortcut != pytest.approx(TIE_RHO)


def test_spearman_is_invariant_to_any_monotone_rescaling():
    """Rank correlation cannot be moved by a monotone transform — Pearson can."""
    warped = [math.exp(y / 50.0) for y in CUBE_Y]
    assert spearman(CUBE_X, warped) == spearman(CUBE_X, CUBE_Y) == 1.0


def test_spearman_is_none_when_a_series_is_constant():
    """All ranks identical -> zero rank variance -> undefined, never 0.0."""
    assert spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None
    assert spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def test_spearman_rejects_malformed_input():
    with pytest.raises(ValueError):
        spearman([1.0, 2.0], [1.0])          # length mismatch
    with pytest.raises(ValueError):
        spearman([1.0], [1.0])               # fewer than two observations


def test_spearman_matches_pearson_exactly_on_already_ranked_data():
    """Rank-transforming ranks is a no-op, so the two must agree exactly."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 1.0, 4.0, 3.0, 5.0]
    assert spearman(a, b) == pytest.approx(_pearson(a, b))


# --- rolling_spearman / matrix / average ------------------------------------


def test_rolling_spearman_mirrors_rolling_correlation_on_a_monotone_pair():
    """A convex monotone map of one close series: Spearman 1, Pearson < 1."""
    base = walk(seed=21, n=61)
    # Strictly increasing map of each CLOSE keeps the return ORDER but bends
    # the magnitudes, so the ranks of the returns are unchanged.
    convex = from_returns([r * abs(r) * 40.0 for r in log_returns(base)])
    assert rolling_spearman(base, convex, window=60) == pytest.approx(1.0)
    pearson = rolling_correlation(base, convex, window=60)
    assert pearson is not None and pearson < 1.0


def test_rolling_spearman_is_none_on_insufficient_history():
    short = walk(seed=3, n=30)
    other = walk(seed=4, n=30)
    assert rolling_spearman(short, other, window=60) is None


def test_rolling_spearman_is_none_on_a_flat_series():
    flat = [100.0] * 61
    assert rolling_spearman(walk(seed=5), flat, window=60) is None


def test_rolling_spearman_window_must_be_at_least_two():
    with pytest.raises(ValueError):
        rolling_spearman(walk(seed=6), walk(seed=7), window=1)


def test_rolling_spearman_matrix_keys_every_unordered_pair_canonically():
    closes = {"CCC": walk(seed=8), "AAA": walk(seed=9), "BBB": walk(seed=10)}
    matrix = rolling_spearman_matrix(closes, window=60)
    assert set(matrix) == {("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")}
    for rho in matrix.values():
        assert rho is not None and -1.0 <= rho <= 1.0


def test_rolling_spearman_matrix_keeps_undefined_pairs_as_honest_nulls():
    """An undefined pair is PRESENT and null — never dropped, never 0.0."""
    closes = {"AAA": walk(seed=11), "FLAT": [100.0] * 61}
    matrix = rolling_spearman_matrix(closes, window=60)
    assert matrix[("AAA", "FLAT")] is None


def test_rolling_spearman_average_skips_undefined_pairs():
    closes = {"AAA": walk(seed=12), "BBB": walk(seed=13), "FLAT": [100.0] * 61}
    matrix = rolling_spearman_matrix(closes, window=60)
    avg, n_pairs = rolling_spearman_average(closes, window=60)
    # Only AAA x BBB is computable: the two FLAT pairs are skipped entirely.
    assert n_pairs == 1
    assert avg == pytest.approx(matrix[("AAA", "BBB")])


def test_rolling_spearman_average_is_none_when_nothing_is_computable():
    closes = {"F1": [100.0] * 61, "F2": [100.0] * 61}
    assert rolling_spearman_average(closes, window=60) == (None, 0)


# --- CorrelationState.current_avg_spearman ----------------------------------


def test_regime_reports_the_spearman_twin_over_the_same_short_window():
    """The canonical fixture's RECENT half has IDENTICAL columns, so both the
    Pearson and the Spearman current averages are exactly 1.0."""
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    assert state.current_avg == 1.0
    assert state.current_avg_spearman == 1.0


def test_spearman_twin_exceeds_pearson_on_a_monotone_nonlinear_book():
    """A book that co-moves monotonically but non-linearly: Spearman EXACTLY
    1.0 while the Pearson average — the number the state uses — is lower."""
    rng = random.Random(77)
    n = 40
    a = [rng.gauss(0.0, 0.02) for _ in range(n)]
    b = [x * abs(x) * 50.0 for x in a]  # strictly monotone, convex
    params = CorrelationRegimeParams(
        long_window=40, short_window=20, min_stress_obs=100
    )
    state = correlation_regime(regime_matrix({"AAA": a, "BBB": b}), params=params)
    assert state.current_avg_spearman == pytest.approx(1.0)
    assert state.current_avg < 1.0
    assert state.current_avg_spearman > state.current_avg


def test_spearman_twin_is_none_when_the_state_is_unavailable():
    """No current window to rank -> honest null, not a fabricated number."""
    state = correlation_regime(
        regime_matrix({"AAA": REGIME_A[:5], "BBB": REGIME_B[:5]}),
        params=REGIME_PARAMS,
    )
    assert state.state == STATE_UNAVAILABLE
    assert state.current_avg_spearman is None


def test_spearman_twin_changes_no_state_decision():
    """§70 SHADOW: the field is display-only. Recomputing the state through
    the SAME inputs with the Spearman value stripped leaves every
    state-bearing field identical — the twin enters no rule."""
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    bare = dataclasses.replace(state)  # a copy without the non-field attribute
    assert bare.current_avg_spearman is None
    assert bare.state == state.state
    assert (bare.normal_avg, bare.current_avg, bare.delta) == (
        state.normal_avg,
        state.current_avg,
        state.delta,
    )


def test_spearman_twin_is_not_a_dataclass_field_so_the_wire_shape_is_unchanged():
    """The §19 wire contract is pinned key-for-key by the gateway's tests.
    This RESEARCH diagnostic must not widen a published API as a side effect
    of being computed, so it is a non-field attribute and ``asdict`` — the
    one serialiser the gateway uses — still yields the SAME eleven keys."""
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    assert state.current_avg_spearman == 1.0  # readable on the object
    assert "current_avg_spearman" not in dataclasses.asdict(state)
    assert {f.name for f in dataclasses.fields(CorrelationState)} == {
        "normal_avg",
        "current_avg",
        "stress_avg",
        "delta",
        "state",
        "n_pairs",
        "n_obs_long",
        "n_obs_short",
        "n_obs_stress",
        "worst_pairs",
        "reason",
    }


def test_with_spearman_rejects_an_impossible_correlation():
    state = correlation_regime(regime_matrix(), params=REGIME_PARAMS)
    for bad in (1.5, -2.0, float("nan"), float("inf"), "0.5"):
        with pytest.raises(ValueError):
            state.with_spearman(bad)
    assert state.with_spearman(None).current_avg_spearman is None
