"""Single-factor (SPY) risk diagnostic — spec §11; compliance §3 row 11.

RESEARCH (spec §70): a display diagnostic. These tests pin the arithmetic,
the honest nulls and — load-bearing — the fact that nothing here can reach a
decision: the module registers no model, derives no ``QuantityCap`` and is
not imported by ``engine``.

Every expected number is hand-derived from the definitions, not read off the
implementation:

    beta = cov(a, f) / var(f)          (sample moments, ddof=1; the n-1
                                        divisors cancel in the ratio)
    r2   = cov^2 / (var_f * var_a)     = the squared Pearson correlation

CONSTRUCTED FACTOR — the exact case. With ``a_t = 2 f_t`` and no noise the
regression is exact: beta = 2 and r2 = 1 to the last bit, because
cov = 2*var_f and cov^2 = 4*var_f^2 = var_f * (4*var_f) = var_f * var_a.

CONSTRUCTED FACTOR + NOISE — the known-R^2 case. Build ``a_t = beta*f_t +
e_t`` from two ORTHOGONAL-BY-CONSTRUCTION series (see ``ORTH_F`` /
``ORTH_E``: their sample covariance is exactly 0), so

    var_a  = beta^2 var_f + var_e     (the cross term vanishes)
    r2     = beta^2 var_f / (beta^2 var_f + var_e)

is an EXACT closed form, not an approximation. Being a RATIO of variances it
is free of the ddof convention entirely: with beta = 2 and var_e = var_f the
share is 4/(4 + 1) = 0.8 exactly, whatever the common divisor.
"""
import dataclasses
import math
import random

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.factor import (
    DEFAULT_FACTOR,
    MODEL_NAME,
    MODEL_VERSION,
    FactorParams,
    FactorRiskResult,
    PositionBeta,
    beta_vs_factor,
    factor_risk_share,
)

#: Enough observations to clear ``min_obs=60`` and the DEGRADED band (120).
N = 200

#: Two series whose SAMPLE covariance is EXACTLY zero, repeated to length N:
#: f cycles (+1,+1,-1,-1) and e cycles (+1,-1,+1,-1). Over each period of 4
#: both have mean 0 and their products sum to (+1)-(1)-(1)+(1) = 0, so the
#: cross term of var(beta*f + e) vanishes identically and r2 has a closed
#: form. Scaled so var_f (ddof=1 over N=200) is a round number.
_F_CYCLE = [1.0, 1.0, -1.0, -1.0]
_E_CYCLE = [1.0, -1.0, 1.0, -1.0]
ORTH_F = [_F_CYCLE[t % 4] for t in range(N)]
ORTH_E = [_E_CYCLE[t % 4] for t in range(N)]


def _sample_var(xs):
    """ddof=1 sample variance — the divisor the module documents."""
    n = len(xs)
    mean = math.fsum(xs) / n
    return math.fsum((x - mean) ** 2 for x in xs) / (n - 1)


def walk(seed: int, n: int = N, sigma: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


# ---------------------------------------------------------------------------
# beta_vs_factor — arithmetic
# ---------------------------------------------------------------------------


def test_orthogonal_fixture_really_is_orthogonal():
    """The closed forms below are only exact because this holds."""
    n = N
    mf = math.fsum(ORTH_F) / n
    me = math.fsum(ORTH_E) / n
    cov = math.fsum((f - mf) * (e - me) for f, e in zip(ORTH_F, ORTH_E)) / (n - 1)
    assert mf == 0.0 and me == 0.0
    assert cov == 0.0  # exactly, not approximately


def test_exact_multiple_gives_beta_two_and_r2_one():
    f = ORTH_F
    a = [2.0 * x for x in f]
    out = beta_vs_factor(a, f)
    assert out["beta"] == pytest.approx(2.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["n"] == N
    assert out["health"] is ModelHealth.ACTIVE
    assert out["reason"] is None


def test_negative_beta_is_reported_with_its_sign():
    f = ORTH_F
    out = beta_vs_factor([-1.5 * x for x in f], f)
    assert out["beta"] == pytest.approx(-1.5)
    assert out["r2"] == pytest.approx(1.0)  # perfectly explained, inversely


def test_known_r2_from_a_constructed_factor_plus_orthogonal_noise():
    """beta=2 with var_e = var_f -> r2 = 4/(4+1) = 0.8 EXACTLY.

    The ddof=1 variance of the +-1 cycle over N=200 is N/(N-1), not 1 — but
    r2 is a RATIO of variances, so that common divisor cancels and the 0.8
    is exact regardless. Both variances are asserted at their true value so
    the arithmetic is stated honestly rather than fudged.
    """
    unit = N / (N - 1)  # ddof=1 variance of a zero-mean +-1 cycle
    f = [x * math.sqrt(2.5) for x in ORTH_F]
    e = [x * math.sqrt(2.5) for x in ORTH_E]
    assert _sample_var(f) == pytest.approx(2.5 * unit)
    assert _sample_var(e) == pytest.approx(2.5 * unit)

    a = [2.0 * f[t] + e[t] for t in range(N)]
    # var_a = 4*var_f + var_e (NO cross term) = 5 * 2.5 * unit.
    assert _sample_var(a) == pytest.approx(5.0 * 2.5 * unit)

    out = beta_vs_factor(a, f)
    assert out["beta"] == pytest.approx(2.0)
    # 4*var_f / (4*var_f + var_e) = 4/(4+1) — the unit cancels.
    assert out["r2"] == pytest.approx(0.8)


def test_r2_is_the_squared_pearson_correlation():
    f = walk(seed=1)
    a = [1.3 * f[t] + w for t, w in enumerate(walk(seed=2, sigma=0.004))]
    out = beta_vs_factor(a, f)
    n = N
    mf, ma = math.fsum(f) / n, math.fsum(a) / n
    cov = math.fsum((x - mf) * (y - ma) for x, y in zip(f, a))
    var_f = math.fsum((x - mf) ** 2 for x in f)
    var_a = math.fsum((y - ma) ** 2 for y in a)
    pearson = cov / math.sqrt(var_f * var_a)
    assert out["r2"] == pytest.approx(pearson**2)


def test_beta_is_invariant_to_the_ddof_convention():
    """The n-1 divisors cancel: beta from population moments is identical."""
    f = walk(seed=3)
    a = [0.7 * x for x in f]
    n = N
    mf, ma = math.fsum(f) / n, math.fsum(a) / n
    pop_beta = math.fsum((y - ma) * (x - mf) for x, y in zip(f, a)) / math.fsum(
        (x - mf) ** 2 for x in f
    )
    assert beta_vs_factor(a, f)["beta"] == pytest.approx(pop_beta)


def test_r2_stays_within_zero_and_one():
    for seed in range(5):
        f = walk(seed=100 + seed)
        a = walk(seed=200 + seed)
        r2 = beta_vs_factor(a, f)["r2"]
        assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# beta_vs_factor — honest nulls and malformed input
# ---------------------------------------------------------------------------


def test_too_few_observations_is_an_honest_null_with_the_real_numbers():
    out = beta_vs_factor(walk(seed=4, n=59), walk(seed=5, n=59))
    assert out["beta"] is None and out["r2"] is None
    assert out["n"] == 59
    assert out["health"] is ModelHealth.UNAVAILABLE
    assert "n=59" in out["reason"] and "min_obs=60" in out["reason"]


def test_thin_sample_is_degraded_but_still_reported():
    """min_obs <= n < 2 x min_obs: the number IS given, and labelled thin."""
    f = [ORTH_F[t] for t in range(80)]
    out = beta_vs_factor([2.0 * x for x in f], f)
    assert out["beta"] == pytest.approx(2.0)  # reported
    assert out["health"] is ModelHealth.DEGRADED
    assert "n=80" in out["reason"]


def test_a_constant_factor_explains_nothing_and_is_unavailable():
    out = beta_vs_factor(walk(seed=6), [0.02] * N)
    assert out["beta"] is None and out["r2"] is None
    assert out["health"] is ModelHealth.UNAVAILABLE
    assert "constant" in out["reason"]


def test_a_constant_asset_has_zero_beta_and_an_undefined_r2():
    """0/0 is not 0: beta is genuinely 0.0, r2 is an honest None."""
    out = beta_vs_factor([0.0] * N, ORTH_F)
    assert out["beta"] == 0.0
    assert out["r2"] is None
    assert out["health"] is ModelHealth.DEGRADED
    assert "0/0" in out["reason"]


def test_length_mismatch_raises_because_it_is_malformed_not_missing():
    with pytest.raises(ValueError, match="same length"):
        beta_vs_factor(walk(seed=7, n=100), walk(seed=8, n=99))


def test_non_finite_input_raises():
    f = walk(seed=9)
    for bad in (float("nan"), float("inf"), None, "x"):
        broken = list(f)
        broken[0] = bad
        with pytest.raises(ValueError):
            beta_vs_factor(broken, f)


def test_min_obs_is_a_parameter_not_a_hardcoded_truth():
    f = [ORTH_F[t] for t in range(40)]
    a = [2.0 * x for x in f]
    assert beta_vs_factor(a, f)["health"] is ModelHealth.UNAVAILABLE
    relaxed = FactorParams(min_obs=20, degraded_multiple=1.0)
    out = beta_vs_factor(a, f, params=relaxed)
    assert out["beta"] == pytest.approx(2.0)
    assert out["health"] is ModelHealth.ACTIVE


def test_factor_params_validate_their_inputs():
    for kwargs in (
        {"min_obs": 2},
        {"min_obs": True},
        {"degraded_multiple": 0.5},
        {"zero_variance": -1.0},
    ):
        with pytest.raises(ValueError):
            FactorParams(**kwargs)


def test_factor_defaults_are_the_documented_research_values():
    p = FactorParams()
    assert (p.min_obs, p.degraded_multiple, p.zero_variance) == (60, 2.0, 1e-18)


# ---------------------------------------------------------------------------
# factor_risk_share — the portfolio number
# ---------------------------------------------------------------------------


def test_share_equals_the_known_r2_of_the_constructed_book():
    """Two positions, both driven by the factor plus ORTHOGONAL noise.

    A = 2f + e, B = 0.5f - e. The book is A + B = 2.5f, exactly — the noise
    cancels — so the factor explains ALL of the book's variance: beta_p =
    2.5 and share = 1.0, even though NEITHER position alone is fully
    explained. That is precisely the portfolio-level fact §11 asks for.
    """
    f = [x * math.sqrt(2.5) for x in ORTH_F]
    e = [x * math.sqrt(2.5) for x in ORTH_E]
    a = [2.0 * f[t] + e[t] for t in range(N)]
    b = [0.5 * f[t] - e[t] for t in range(N)]

    result = factor_risk_share({"A": a, "B": b}, f)
    assert isinstance(result, FactorRiskResult)
    assert result.portfolio_beta == pytest.approx(2.5)
    assert result.explained_variance_share == pytest.approx(1.0)
    assert result.idiosyncratic_share == pytest.approx(0.0)
    # Per position, the individually-known R^2 of 0.8 / (0.25*2.5)/(...)
    betas = {p.label: p.beta for p in result.positions}
    assert betas["A"] == pytest.approx(2.0)
    assert betas["B"] == pytest.approx(0.5)
    r2s = {p.label: p.r2 for p in result.positions}
    assert r2s["A"] == pytest.approx(0.8)          # 4*2.5/(4*2.5+2.5)
    assert r2s["B"] == pytest.approx(0.2)          # 0.25*2.5/(0.25*2.5+2.5)


def test_share_is_the_portfolio_regression_r2_identically():
    """The two routes to the share must agree to the last bit."""
    f = walk(seed=11)
    a = [1.8 * f[t] + w for t, w in enumerate(walk(seed=12, sigma=0.003))]
    b = [-0.4 * f[t] + w for t, w in enumerate(walk(seed=13, sigma=0.003))]
    result = factor_risk_share({"A": a, "B": b}, f)
    total = [a[t] + b[t] for t in range(N)]
    direct = beta_vs_factor(total, f)
    assert result.portfolio_beta == direct["beta"]
    assert result.explained_variance_share == direct["r2"]


def test_share_matches_the_beta_squared_variance_ratio_definition():
    """share = beta_p^2 * var(f) / var(pnl_total), the docstring's formula."""
    f = walk(seed=14)
    a = [1.1 * f[t] + w for t, w in enumerate(walk(seed=15, sigma=0.004))]
    result = factor_risk_share({"A": a}, f)
    total_var = _sample_var(a)
    expected = result.portfolio_beta**2 * _sample_var(f) / total_var
    assert result.explained_variance_share == pytest.approx(expected)


def test_shares_sum_to_one():
    f = walk(seed=16)
    a = [0.9 * f[t] + w for t, w in enumerate(walk(seed=17, sigma=0.005))]
    result = factor_risk_share({"A": a}, f)
    assert (
        result.explained_variance_share + result.idiosyncratic_share
        == pytest.approx(1.0)
    )


def test_a_book_uncorrelated_with_the_factor_has_a_near_zero_share():
    f = walk(seed=18)
    a = walk(seed=19)  # independent
    result = factor_risk_share({"A": a}, f)
    assert result.explained_variance_share < 0.05
    assert result.idiosyncratic_share > 0.95


def test_positions_keep_the_callers_mapping_order():
    f = walk(seed=20)
    book = {t: [0.5 * x for x in f] for t in ("ZZZ", "AAA", "MMM")}
    result = factor_risk_share(book, f)
    assert [p.label for p in result.positions] == ["ZZZ", "AAA", "MMM"]


def test_one_broken_position_does_not_take_the_book_down():
    """A flat position reports its own honest null; the book still computes."""
    f = ORTH_F
    good = [2.0 * x for x in f]
    result = factor_risk_share({"GOOD": good, "FLAT": [0.0] * N}, f)
    by_label = {p.label: p for p in result.positions}
    assert by_label["FLAT"].r2 is None
    assert by_label["FLAT"].beta == 0.0
    assert by_label["GOOD"].beta == pytest.approx(2.0)
    # The book is GOOD + FLAT = GOOD, fully explained.
    assert result.health is ModelHealth.ACTIVE
    assert result.explained_variance_share == pytest.approx(1.0)


def test_constant_factor_makes_the_whole_result_unavailable():
    result = factor_risk_share({"A": walk(seed=21)}, [0.01] * N)
    assert result.portfolio_beta is None
    assert result.explained_variance_share is None
    assert result.idiosyncratic_share is None
    assert result.health is ModelHealth.UNAVAILABLE
    assert "constant" in result.reason
    assert result.is_available is False


def test_too_short_a_history_makes_the_result_unavailable():
    result = factor_risk_share({"A": walk(seed=22, n=30)}, walk(seed=23, n=30))
    assert result.health is ModelHealth.UNAVAILABLE
    assert result.explained_variance_share is None
    assert "min_obs=60" in result.reason


def test_empty_book_and_misaligned_series_raise():
    f = walk(seed=24)
    with pytest.raises(ValueError, match="at least one position"):
        factor_risk_share({}, f)
    with pytest.raises(ValueError, match="positionally aligned"):
        factor_risk_share({"A": walk(seed=25, n=N - 1)}, f)


def test_factor_label_is_recorded_never_assumed():
    f = walk(seed=26)
    default = factor_risk_share({"A": [2.0 * x for x in f]}, f)
    assert default.factor == DEFAULT_FACTOR == "SPY"
    named = factor_risk_share({"A": [2.0 * x for x in f]}, f, factor="QQQ")
    assert named.factor == "QQQ"
    assert named.meta.params["factor"] == "QQQ"


def test_meta_records_everything_needed_to_reproduce_the_number():
    f = walk(seed=27)
    result = factor_risk_share({"A": [2.0 * x for x in f]}, f)
    assert result.meta.model_name == MODEL_NAME == "single_factor_beta"
    assert result.meta.model_version == MODEL_VERSION
    assert result.meta.lookback == N
    assert result.meta.params["min_obs"] == 60


def test_results_are_frozen_and_validate_their_own_invariants():
    f = walk(seed=28)
    result = factor_risk_share({"A": [2.0 * x for x in f]}, f)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.portfolio_beta = 1.0
    # A non-ACTIVE health without a reason is malformed.
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        PositionBeta(
            label="X", beta=None, r2=None, n=0,
            health=ModelHealth.UNAVAILABLE, reason=None,
        )


def test_determinism_same_input_same_number():
    f = walk(seed=29)
    a = [1.4 * f[t] + w for t, w in enumerate(walk(seed=30, sigma=0.004))]
    first = factor_risk_share({"A": a}, f)
    second = factor_risk_share({"A": a}, f)
    assert first.portfolio_beta == second.portfolio_beta
    assert first.explained_variance_share == second.explained_variance_share


# ---------------------------------------------------------------------------
# §70 RESEARCH: this module cannot reach a decision
# ---------------------------------------------------------------------------


def test_factor_module_registers_no_model_and_derives_no_cap():
    """RESEARCH means no seam into the decision path at all: no registered
    model a caller could ``get()``, and no ``QuantityCap`` factory."""
    from libs.trading_core.risk.models import base, factor

    assert "single_factor_beta" not in base.names()
    assert not [n for n in dir(factor) if "cap" in n.lower()]
    assert not [n for n in dir(factor) if n.startswith("register")]


def test_engine_does_not_import_the_factor_module():
    """Tier 0 consults nothing here — the import graph proves it."""
    import ast
    import pathlib

    src = pathlib.Path("libs/trading_core/risk/engine.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("factor" in name for name in imported)


def test_factor_is_exported_from_both_packages_and_is_the_same_object():
    from libs.trading_core import risk
    from libs.trading_core.risk import models

    assert risk.factor_risk_share is models.factor_risk_share is factor_risk_share
    assert risk.beta_vs_factor is models.beta_vs_factor is beta_vs_factor
