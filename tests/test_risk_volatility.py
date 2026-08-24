"""Volatility models — hand-checked (Phase B contract §2.4, invariants §3).

Reference covariance data (3 tickers x 4 dates), used by the covariance
tests below:

    AAA = [1, 2, 3, 4]   mean 2.5
    BBB = [2, 4, 5, 9]   mean 5.0
    CCC = [5, 3, 4, 0]   mean 3.0

Deviations:
    dA = [-1.5, -0.5, 0.5, 1.5]
    dB = [-3.0, -1.0, 0.0, 4.0]
    dC = [ 2.0,  0.0, 1.0, -3.0]

ddof=1 (divide by n-1 = 3):
    cov(A,A) = (2.25+0.25+0.25+2.25)/3 = 5.0/3   = 1.6666666666666667
    cov(A,B) = (4.5+0.5+0+6.0)/3       = 11.0/3  = 3.6666666666666665
    cov(A,C) = (-3.0+0+0.5-4.5)/3      = -7.0/3  = -2.3333333333333335
    cov(B,B) = (9+1+0+16)/3            = 26.0/3  = 8.666666666666666
    cov(B,C) = (-6.0+0+0-12.0)/3       = -18.0/3 = -6.0
    cov(C,C) = (4+0+1+9)/3             = 14.0/3  = 4.666666666666667
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.volatility import (
    DEFAULT_ANNUALIZATION_DAYS,
    DEFAULT_INIT_OBS,
    DEFAULT_LAMBDA,
    DEFAULT_MIN_OBS,
    MODEL_VERSION,
    CovarianceResult,
    ewma_variance,
    ewma_volatility_forecast,
    portfolio_volatility,
    sample_covariance,
    volatility_scaled_pnl,
    volatility_scaling,
)
from libs.trading_core.risk.returns import ReturnMatrix

DATES4 = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8))
TICKERS3 = ("AAA", "BBB", "CCC")
# rows[t][i]: AAA, BBB, CCC on each date
ROWS3x4 = (
    (1.0, 2.0, 5.0),
    (2.0, 4.0, 3.0),
    (3.0, 5.0, 4.0),
    (4.0, 9.0, 0.0),
)


def _matrix(rows=ROWS3x4, tickers=TICKERS3, dates=DATES4) -> ReturnMatrix:
    return ReturnMatrix(
        dates=dates, tickers=tickers, rows=rows, return_type="LOG"
    )


# ---------------------------------------------------------------------------
# Sample covariance
# ---------------------------------------------------------------------------


def test_sample_covariance_3x4_hand_computed():
    r = sample_covariance(_matrix(), min_obs=4)
    assert r.health is ModelHealth.ACTIVE and r.reason is None
    assert r.n == 4 and r.sample_size == 4
    assert r.tickers == TICKERS3
    # Hand values from the module docstring above.
    assert r.entry("AAA", "AAA") == pytest.approx(5.0 / 3.0, abs=1e-15)
    assert r.entry("AAA", "BBB") == pytest.approx(11.0 / 3.0, abs=1e-15)
    assert r.entry("AAA", "CCC") == pytest.approx(-7.0 / 3.0, abs=1e-15)
    assert r.entry("BBB", "BBB") == pytest.approx(26.0 / 3.0, abs=1e-15)
    assert r.entry("BBB", "CCC") == pytest.approx(-6.0, abs=1e-15)
    assert r.entry("CCC", "CCC") == pytest.approx(14.0 / 3.0, abs=1e-15)
    # variance() is the diagonal
    assert r.variance("BBB") == r.entry("BBB", "BBB")


def test_sample_covariance_is_symmetric():
    # Contract 2.4: symmetric within 1e-12 (this implementation mirrors the
    # i<=j cell, so it is bit-for-bit identical).
    r = sample_covariance(_matrix(), min_obs=4)
    for i, a in enumerate(TICKERS3):
        for j, b in enumerate(TICKERS3):
            assert abs(r.matrix[i][j] - r.matrix[j][i]) <= 1e-12
            assert r.matrix[i][j] == r.matrix[j][i]  # exactly, in fact
            assert r.entry(a, b) == r.entry(b, a)


def test_sample_covariance_diagonal_equals_sample_variance():
    # cov(x, x) must equal the ddof=1 variance of that column.
    r = sample_covariance(_matrix(), min_obs=4)
    for t in TICKERS3:
        col = _matrix().column(t)
        mean = math.fsum(col) / len(col)
        var = math.fsum((v - mean) ** 2 for v in col) / (len(col) - 1)
        assert r.variance(t) == pytest.approx(var, abs=1e-15)


def test_sample_covariance_below_min_obs_is_unavailable():
    # Contract 3.6: honest null, real numbers in the reason, empty matrix.
    r = sample_covariance(_matrix())  # default min_obs = 60, n = 4
    assert r.health is ModelHealth.UNAVAILABLE
    assert r.matrix == ()
    assert r.reason and "n=4" in r.reason and "min_obs=60" in r.reason
    assert r.tickers == TICKERS3   # column identity is still reported
    assert not r.is_available
    with pytest.raises(KeyError):
        r.entry("AAA", "BBB")      # no fabricated 0


def test_sample_covariance_meta_and_unknown_ticker():
    r = sample_covariance(_matrix(), min_obs=4)
    assert r.meta.model_name == "sample_covariance"
    assert r.meta.model_version == MODEL_VERSION
    assert r.meta.params["ddof"] == 1
    assert r.meta.return_type == "LOG"
    assert r.meta.as_of == date(2026, 1, 8)
    assert r.diagnostics == {"n": 4, "n_tickers": 3, "ddof": 1}
    with pytest.raises(KeyError):
        r.entry("AAA", "ZZZ")


def test_sample_covariance_scaling_by_k_scales_by_k_squared():
    # Covariance is bilinear: scaling every return by k scales cov by k^2.
    k = 3.0
    scaled_rows = tuple(tuple(k * v for v in row) for row in ROWS3x4)
    base = sample_covariance(_matrix(), min_obs=4)
    got = sample_covariance(_matrix(rows=scaled_rows), min_obs=4)
    assert got.entry("AAA", "BBB") == pytest.approx(
        k * k * base.entry("AAA", "BBB"), rel=1e-12
    )


def test_sample_covariance_rejects_bad_min_obs():
    with pytest.raises(ValueError):
        sample_covariance(_matrix(), min_obs=1)  # ddof=1 needs 2


def test_covariance_result_rejects_inconsistent_construction():
    r = sample_covariance(_matrix(), min_obs=4)
    with pytest.raises(ValueError):
        CovarianceResult(
            tickers=TICKERS3,
            matrix=((1.0, 2.0), (3.0, 4.0)),  # 2x2 for 3 tickers
            n=4,
            health=ModelHealth.ACTIVE,
            reason=None,
            meta=r.meta,
        )
    with pytest.raises(ValueError):
        CovarianceResult(
            tickers=TICKERS3, matrix=(), n=0,
            health=ModelHealth.UNAVAILABLE, reason="",  # non-ACTIVE needs a reason
            meta=r.meta,
        )


# ---------------------------------------------------------------------------
# Portfolio volatility
# ---------------------------------------------------------------------------


def test_portfolio_volatility_equals_sample_stdev_hand_computed():
    # pnl = [10, -10, 20, -20, 0, 0]  =>  mean 0
    # Sum sq dev = 100+100+400+400+0+0 = 1000; var = 1000/5 = 200
    # sigma = sqrt(200) = 14.142135623730951
    pnl = [10.0, -10.0, 20.0, -20.0, 0.0, 0.0]
    r = portfolio_volatility(pnl, min_obs=6)
    assert r.health is ModelHealth.ACTIVE
    assert r.diagnostics["mu"] == pytest.approx(0.0, abs=1e-15)
    assert r.diagnostics["variance"] == pytest.approx(200.0, abs=1e-12)
    assert r.value == pytest.approx(math.sqrt(200.0), abs=1e-12)
    assert r.value == pytest.approx(14.142135623730951, abs=1e-12)
    # annualized_usd = sigma * sqrt(252) = 14.142135623730951 * 15.874507866387544
    assert r.diagnostics["annualized_usd"] == pytest.approx(
        math.sqrt(200.0) * math.sqrt(252), abs=1e-11
    )
    assert r.diagnostics["annualization_days"] == DEFAULT_ANNUALIZATION_DAYS == 252
    assert r.sample_size == 6


def test_portfolio_volatility_matches_statistics_stdev():
    # Cross-check the ddof=1 estimator against the stdlib on a second series.
    import statistics

    pnl = [3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, 6.0]
    r = portfolio_volatility(pnl, min_obs=8)
    assert r.value == pytest.approx(statistics.stdev(pnl), abs=1e-12)


def test_portfolio_volatility_constant_series_is_zero_not_missing():
    # sigma = 0 is a real number (a flat book), not a data gap.
    r = portfolio_volatility([5.0] * 10, min_obs=5)
    assert r.value == 0.0
    assert r.health is ModelHealth.ACTIVE


def test_portfolio_volatility_scales_by_k():
    # Contract 3.4: k * pnl => sigma scales by k.
    pnl = [3.0, -1.0, 4.0, -1.0, 5.0, -9.0]
    k = 2.5
    base = portfolio_volatility(pnl, min_obs=6).value
    got = portfolio_volatility([k * p for p in pnl], min_obs=6).value
    assert got == pytest.approx(k * base, rel=1e-12)


def test_portfolio_volatility_shift_invariant():
    # Adding a constant moves mu, never sigma.
    pnl = [3.0, -1.0, 4.0, -1.0, 5.0, -9.0]
    base = portfolio_volatility(pnl, min_obs=6).value
    got = portfolio_volatility([p + 100.0 for p in pnl], min_obs=6)
    assert got.value == pytest.approx(base, abs=1e-12)
    assert got.diagnostics["mu"] == pytest.approx(
        portfolio_volatility(pnl, min_obs=6).diagnostics["mu"] + 100.0, abs=1e-12
    )


def test_portfolio_volatility_below_min_obs_is_unavailable():
    r = portfolio_volatility([1.0, 2.0, 3.0])  # default min_obs = 60
    assert r.value is None and r.health is ModelHealth.UNAVAILABLE
    assert "n=3" in r.reason and "min_obs=60" in r.reason
    assert DEFAULT_MIN_OBS == 60


def test_portfolio_volatility_rejects_non_finite():
    with pytest.raises(ValueError):
        portfolio_volatility([1.0, float("nan"), 3.0], min_obs=2)


# ---------------------------------------------------------------------------
# EWMA variance — hand-checked recursion
# ---------------------------------------------------------------------------


def test_ewma_variance_three_steps_hand_computed():
    # r = [0.01, -0.02, 0.03, 0.01], lam = 0.94, init_obs = 2.
    # seed  s2 = (0.01^2 + 0.02^2)/2 = (0.0001 + 0.0004)/2 = 0.00025
    # s3 = 0.94*0.00025 + 0.06*(0.03^2) = 0.000235 + 0.000054 = 0.000289
    # s4 = 0.94*0.000289 + 0.06*(0.01^2) = 0.00027166 + 0.000006 = 0.00027766
    #   (s4 is the NEXT-period forecast; out has length len(returns) = 4)
    r = [0.01, -0.02, 0.03, 0.01]
    out = ewma_variance(r, lam=0.94, init_obs=2)
    assert len(out) == 4
    assert out[0] is None and out[1] is None       # warm-up
    assert out[2] == pytest.approx(0.00025, abs=1e-18)
    assert out[3] == pytest.approx(0.000289, abs=1e-18)
    # the next-period forecast is exposed through ewma_volatility_forecast
    f = ewma_volatility_forecast(r, lam=0.94, init_obs=2)
    assert f.diagnostics["variance"] == pytest.approx(0.00027766, abs=1e-18)
    assert f.value == pytest.approx(math.sqrt(0.00027766), abs=1e-15)
    assert f.value == pytest.approx(0.016663132958720576, abs=1e-15)


def test_ewma_variance_walk_forward_sentinel_spike():
    # Contract 3.5 / 2.4: the forecast for index t must use returns[< t]
    # ONLY. Plant a huge spike at index t and check out[t] is unchanged
    # (while out[t+1], which legitimately sees it, does change).
    base = [0.01 if i % 2 == 0 else -0.01 for i in range(12)]
    t = 8
    spiked = list(base)
    spiked[t] = 5.0  # enormous
    a = ewma_variance(base, lam=0.94, init_obs=4)
    b = ewma_variance(spiked, lam=0.94, init_obs=4)
    for i in range(t + 1):
        assert a[i] == b[i], f"forecast for index {i} peeked at returns[{t}]"
    assert b[t + 1] != a[t + 1]  # the day AFTER does see it
    assert b[t + 1] > a[t + 1]


def test_ewma_variance_constant_absolute_returns_is_constant():
    # Contract 3.9: with |r| constant = c, the zero-mean seed is c^2 and
    # c^2 is the recursion's fixed point (0.94*c^2 + 0.06*c^2 = c^2), so
    # every forecast equals c^2 exactly.
    c = 0.02
    r = [c if i % 2 == 0 else -c for i in range(30)]
    out = ewma_variance(r, lam=0.94, init_obs=5)
    for i in range(5, 30):
        assert out[i] == pytest.approx(c * c, abs=1e-18), i
    f = ewma_volatility_forecast(r, lam=0.94, init_obs=5)
    assert f.value == pytest.approx(c, abs=1e-15)


def test_ewma_variance_warmup_entries_are_none():
    r = [0.01] * 10
    out = ewma_variance(r, lam=0.94, init_obs=6)
    assert out[:6] == [None] * 6
    assert all(v is not None for v in out[6:])
    # A series no longer than init_obs yields no forecast at all.
    assert ewma_variance([0.01] * 4, lam=0.94, init_obs=6) == [None] * 4


def test_ewma_variance_seed_is_the_zero_mean_second_moment():
    # seed = Sum r^2 / init_obs (NOT the ddof=1 variance about the mean).
    r = [0.02, 0.04, -0.01, 0.03, 0.05]
    out = ewma_variance(r, lam=0.9, init_obs=3)
    # (0.0004 + 0.0016 + 0.0001)/3 = 0.0021/3 = 0.0007
    assert out[3] == pytest.approx(0.0007, abs=1e-18)
    # next: 0.9*0.0007 + 0.1*(0.03^2) = 0.00063 + 0.00009 = 0.00072
    assert out[4] == pytest.approx(0.00072, abs=1e-18)


def test_ewma_lambda_out_of_range_raises():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ewma_variance([0.01] * 30, lam=bad)
    with pytest.raises(ValueError):
        ewma_variance([0.01] * 30, init_obs=0)
    with pytest.raises(ValueError):
        ewma_variance([0.01, float("inf")] * 15)


def test_ewma_volatility_forecast_health_and_defaults():
    assert DEFAULT_LAMBDA == 0.94 and DEFAULT_INIT_OBS == 20
    short = ewma_volatility_forecast([0.01] * 5, init_obs=20)
    assert short.value is None and short.health is ModelHealth.UNAVAILABLE
    assert "n=5" in short.reason and "init_obs=20" in short.reason
    ok = ewma_volatility_forecast([0.01, -0.01] * 15, init_obs=20)
    assert ok.health is ModelHealth.ACTIVE
    assert ok.diagnostics["lambda"] == 0.94
    # half-life: ln(0.5)/ln(0.94) = -0.6931471805599453 / -0.061875403718087454
    #          = 11.202305583621158 days
    assert ok.diagnostics["half_life_days"] == pytest.approx(
        math.log(0.5) / math.log(0.94), abs=1e-12
    )
    assert ok.diagnostics["half_life_days"] == pytest.approx(11.2023055836, abs=1e-9)


def test_ewma_volatility_forecast_scales_by_k():
    r = [0.01, -0.02, 0.03, 0.01, -0.015, 0.02, 0.005, -0.03]
    k = 4.0
    base = ewma_volatility_forecast(r, lam=0.94, init_obs=3).value
    got = ewma_volatility_forecast([k * x for x in r], lam=0.94, init_obs=3).value
    assert got == pytest.approx(k * base, rel=1e-12)


# ---------------------------------------------------------------------------
# Volatility-scaled P&L (Hull-White filtered historical simulation)
# ---------------------------------------------------------------------------


def test_volatility_scaled_pnl_constant_vol_is_identity_on_the_tail():
    # |pnl| constant => sigma_t == sigma_now for every kept t, ratios are
    # exactly 1, so the output is pnl[init_obs:] unchanged.
    c = 50.0
    pnl = [c if i % 2 == 0 else -c for i in range(20)]
    out = volatility_scaled_pnl(pnl, lam=0.94, init_obs=5)
    assert len(out) == 15
    for got, want in zip(out, pnl[5:]):
        assert got == pytest.approx(want, abs=1e-12)


def test_volatility_scaled_pnl_hand_computed_two_entries():
    # pnl = [10, -10, 30, -20], lam = 0.5, init_obs = 2.
    # seed s2 = (100 + 100)/2 = 100
    # s3 = 0.5*100 + 0.5*(30^2) = 50 + 450 = 500
    # s4 = 0.5*500 + 0.5*(20^2) = 250 + 200 = 450   (= sigma_now^2)
    # sigma_now = sqrt(450) = 21.213203435596427
    # kept t = 2, 3:
    #   t=2: pnl=30, sigma_2 = sqrt(100) = 10  => 30 * sqrt(450)/10 = 63.63961030678928
    #   t=3: pnl=-20, sigma_3 = sqrt(500)      => -20 * sqrt(450)/sqrt(500)
    #        = -20 * sqrt(0.9) = -18.973665961010276
    pnl = [10.0, -10.0, 30.0, -20.0]
    out = volatility_scaled_pnl(pnl, lam=0.5, init_obs=2)
    assert len(out) == 2
    assert out[0] == pytest.approx(30.0 * math.sqrt(450.0) / 10.0, abs=1e-12)
    assert out[0] == pytest.approx(63.63961030678928, abs=1e-12)
    assert out[1] == pytest.approx(-20.0 * math.sqrt(0.9), abs=1e-12)
    assert out[1] == pytest.approx(-18.973665961010276, abs=1e-12)
    # bookkeeping via the detailed view
    s = volatility_scaling(pnl, lam=0.5, init_obs=2)
    assert s.sigma_now == pytest.approx(math.sqrt(450.0), abs=1e-12)
    assert (s.n_input, s.n_used, s.dropped) == (4, 2, 2)


def test_volatility_scaled_pnl_lambda_near_one_approaches_plain_hs():
    # Contract 3.9 limit sanity: as lam -> 1 the EWMA path barely moves, so
    # sigma_now/sigma_t -> 1 and the filtered series -> pnl[init_obs:].
    pnl = [float(((i * 31) % 17) - 8) for i in range(60)]
    out = volatility_scaled_pnl(pnl, lam=0.999, init_obs=10)
    plain = pnl[10:]
    assert len(out) == len(plain)
    for got, want in zip(out, plain):
        assert got == pytest.approx(want, rel=0.05, abs=1e-9)
    # and the agreement tightens as lambda rises towards 1
    def max_gap(lam: float) -> float:
        o = volatility_scaled_pnl(pnl, lam=lam, init_obs=10)
        return max(abs(a - b) for a, b in zip(o, plain))

    assert max_gap(0.9999) < max_gap(0.999) < max_gap(0.99)


def test_volatility_scaled_pnl_walk_forward_uses_only_earlier_data():
    # sigma_t is built from pnl[< t], so a spike at index t rescales that
    # day's own entry (pnl_t changed) but must not change the RATIO applied
    # to any earlier day. Compare the scale factors directly.
    pnl = [10.0 if i % 2 == 0 else -10.0 for i in range(30)]
    t = 20
    spiked = list(pnl)
    spiked[t] = 900.0
    a = volatility_scaling(pnl, lam=0.94, init_obs=5)
    b = volatility_scaling(spiked, lam=0.94, init_obs=5)
    # ratio_t = scaled_t / pnl_t  =  sigma_now / sigma_t; sigma_now differs
    # between the two runs, so normalise it out and compare sigma_t itself.
    va = ewma_variance(pnl, lam=0.94, init_obs=5)
    vb = ewma_variance(spiked, lam=0.94, init_obs=5)
    for i in range(t + 1):
        assert va[i] == vb[i], f"sigma for index {i} peeked at pnl[{t}]"
    assert a.n_used == b.n_used == 25


def test_volatility_scaled_pnl_warmup_shorter_than_init_obs_is_empty():
    assert volatility_scaled_pnl([1.0, 2.0], lam=0.94, init_obs=5) == []
    s = volatility_scaling([1.0, 2.0], lam=0.94, init_obs=5)
    assert s.scaled == () and s.sigma_now is None and s.n_used == 0


def test_volatility_scaled_pnl_all_zero_series_keeps_nothing():
    # Every sigma is 0: there is no volatility information to scale by, so
    # entries are dropped rather than imputed or divided by zero.
    out = volatility_scaled_pnl([0.0] * 20, lam=0.94, init_obs=5)
    assert out == []


def test_volatility_scaled_pnl_scales_by_k():
    # The ratio sigma_now/sigma_t is scale-free, so k*pnl => k*scaled.
    pnl = [float(((i * 13) % 11) - 5) for i in range(40)]
    k = 7.0
    base = volatility_scaled_pnl(pnl, lam=0.94, init_obs=10)
    got = volatility_scaled_pnl([k * p for p in pnl], lam=0.94, init_obs=10)
    assert len(got) == len(base)
    for g, b in zip(got, base):
        assert g == pytest.approx(k * b, rel=1e-12)


def test_volatility_scaled_pnl_rejects_bad_params():
    with pytest.raises(ValueError):
        volatility_scaled_pnl([1.0] * 30, lam=1.0)
    with pytest.raises(ValueError):
        volatility_scaled_pnl([1.0] * 30, init_obs=0)
