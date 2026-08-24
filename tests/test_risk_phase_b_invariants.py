"""Cross-module contract invariants for the Phase B risk library.

Every test here pins a clause of §3 of
``docs/risk-engine-phase-b-design.md`` — the invariants that hold BETWEEN
modules and that no single module's own test file can protect. They are the
reason the contract exists: VaR, ES, volatility and risk contribution must
be mutually coherent (ES ≥ VaR, contributions sum to totals, one sign
convention, ONE quantile estimator shared by ``var_es`` and
``contribution``).

Numbered to the contract:

1.  ES ≥ VaR and monotone in confidence (§3.1, §3.2)
2.  Σ RC^ES == ES exactly; Σ RC^σ == σ_p to 1e-9 relative (§3.3)
3.  Scaling and shift laws (§3.4)
4.  Walk-forward never sees ``pnl[t]`` (§3.5)
5.  Below ``min_obs`` ⇒ honest null, never an exception (§3.6)
6.  ``correlation.log_returns is returns.log_returns`` (§3.7)
7.  ``validate()`` never upgrades health, for every registered model (§3.8)
8.  Filtered HS with λ → 1 collapses to plain HS (§3.9)
9.  The registry holds the four VaR/ES models, all SHADOW (spec §70)
10. An end-to-end mini pipeline: closes → returns → book P&L → every
    estimator → snapshot, asserting the invariants hold on real wiring and
    that every ``ModelResult`` carries reproducible ``ModelMeta``.
"""
from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta

import pytest

from libs.trading_core import correlation
from libs.trading_core.risk import returns as returns_mod
from libs.trading_core.risk.models import base as base_mod
from libs.trading_core.risk.models import contribution as contribution_mod
from libs.trading_core.risk.models import var_es as var_es_mod
from libs.trading_core.risk.models import volatility as volatility_mod
from libs.trading_core.risk.models.base import ModelHealth, ModelMode, ModelResult
from libs.trading_core.risk.models.contribution import (
    es_contributions,
    volatility_contributions,
)
from libs.trading_core.risk.models.diagnostics import distribution_diagnostics
from libs.trading_core.risk.models.drawdown import reconstructed_book_drawdown
from libs.trading_core.risk.models.ensemble import dispersion, model_risk_state
from libs.trading_core.risk.models.var_es import (
    conditional_var,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    tail_size,
)
from libs.trading_core.risk.models.volatility import portfolio_volatility
from libs.trading_core.risk.pnl_series import PositionRiskInput, book_pnl_series
from libs.trading_core.risk.returns import align, returns_from_closes
from libs.trading_core.risk.snapshot import DataQuality, PortfolioRiskSnapshot
from libs.trading_core.risk.validation import walk_forward

SEED = 20260817

#: The four (VaR, ES) estimator pairs the contract's §3.1/§3.2 range over.
VAR_ES_PAIRS = (
    ("historical", historical_var, historical_es),
    ("gaussian", gaussian_var, gaussian_es),
)


@pytest.fixture(autouse=True)
def _registry_populated():
    """The four VaR/ES models must be registered even if another test module's
    ``clear_for_tests()`` fixture emptied the shared registry first."""
    var_es_mod.register_models()
    yield


def _seeded_pnl(seed: int, n: int = 600, mu: float = 40.0, sigma: float = 1800.0) -> list[float]:
    """A deterministic gain-positive P&L series (USD/day)."""
    rnd = random.Random(seed)
    return [rnd.gauss(mu, sigma) for _ in range(n)]


def _seeded_series() -> list[list[float]]:
    """Five independent seeded series from one root seed (contract §3 tests
    ask for 5 random series under ``random.Random(20260817)``)."""
    root = random.Random(SEED)
    return [_seeded_pnl(root.randrange(1_000_000)) for _ in range(5)]


# ---------------------------------------------------------------------------
# §3.1 / §3.2 — ES >= VaR and monotone in confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,var_fn,es_fn", VAR_ES_PAIRS, ids=lambda v: getattr(v, "__name__", v))
def test_es_at_least_var_on_seeded_series(family, var_fn, es_fn):
    """§3.1: ``ES_α ≥ VaR_α`` for the historical AND Gaussian pairs.

    Historical ES is the mean of the k largest losses and historical VaR is
    the k-th (smallest) of exactly those k, so the mean cannot be below it.
    """
    for i, pnl in enumerate(_seeded_series()):
        for alpha in (0.95, 0.99):
            var = var_fn(pnl, alpha)
            es = es_fn(pnl, alpha)
            assert var.value is not None and es.value is not None
            assert es.value >= var.value, f"{family} series {i} @{alpha}: ES {es.value} < VaR {var.value}"


@pytest.mark.parametrize("family,var_fn,es_fn", VAR_ES_PAIRS, ids=lambda v: getattr(v, "__name__", v))
def test_var_and_es_monotone_in_confidence(family, var_fn, es_fn):
    """§3.2: a higher confidence looks further into the tail, so both the
    VaR and the ES at 99% are at least their 95% counterparts."""
    for i, pnl in enumerate(_seeded_series()):
        v95, v99 = var_fn(pnl, 0.95).value, var_fn(pnl, 0.99).value
        e95, e99 = es_fn(pnl, 0.95).value, es_fn(pnl, 0.99).value
        assert v99 >= v95, f"{family} series {i}: VaR99 {v99} < VaR95 {v95}"
        assert e99 >= e95, f"{family} series {i}: ES99 {e99} < ES95 {e95}"


def test_historical_es_equals_var_when_tail_is_one():
    """§3.1 equality case: ``k = 1`` makes ES the mean of a one-element tail,
    which is exactly the k-th largest loss."""
    pnl = _seeded_pnl(1, n=60)
    assert tail_size(60, 0.99) == 1
    var = historical_var(pnl, 0.99, min_obs=60)
    es = historical_es(pnl, 0.99, min_obs=60)
    assert es.value == var.value


# ---------------------------------------------------------------------------
# §3.3 — contributions sum to their totals (the reason ES is a tail average)
# ---------------------------------------------------------------------------


def _seeded_book(seed: int, n_pos: int = 4, n: int = 600) -> dict[str, list[float]]:
    rnd = random.Random(seed)
    return {f"POS#{i}": [rnd.gauss(0.0, 300.0) for _ in range(n)] for i in range(n_pos)}


def test_es_contributions_sum_exactly_to_historical_es():
    """§3.3: ``Σ_i RC^ES_i == ES_α`` EXACTLY — same series, same α, same k.

    This is the invariant that forced the contract to define ES as a plain
    tail average, and it only holds while ``contribution`` and ``var_es``
    agree on which k dates are "the tail". Asserted with ``==``, not
    ``approx``: both sides are the same ``math.fsum`` over the same numbers.
    """
    for seed in (11, 22, 33):
        cols = _seeded_book(seed)
        portfolio = [math.fsum(col[t] for col in cols.values()) for t in range(600)]
        for alpha in (0.95, 0.99):
            rc = es_contributions(cols, confidence=alpha)
            es = historical_es(portfolio, alpha)
            assert rc.total is not None and es.value is not None
            # §3.3 EXACT: the reported total is bit-identical to the ES.
            assert rc.total == es.value
            # Re-summing the rows in a different association order is only
            # guaranteed to 1e-9 relative (fsum of a permutation can differ by
            # an ULP); the contract's tolerance, not equality, applies here.
            resummed = math.fsum(p.contribution for p in rc.per_position)
            assert abs(resummed - es.value) <= 1e-9 * max(1.0, abs(es.value))


def test_vol_contributions_sum_to_portfolio_volatility():
    """§3.3: ``Σ_i RC^σ_i == σ_p`` within ``1e-9 × max(1, |σ_p|)``.

    Only the fsum rounding separates the two: ``Σ_i cov(pnl_i, pnl_p)/σ_p =
    var(pnl_p)/σ_p = σ_p``.
    """
    for seed in (11, 22, 33):
        cols = _seeded_book(seed)
        portfolio = [math.fsum(col[t] for col in cols.values()) for t in range(600)]
        rc = volatility_contributions(cols)
        sigma = portfolio_volatility(portfolio)
        assert rc.total is not None and sigma.value is not None
        total = math.fsum(p.contribution for p in rc.per_position)
        assert total == pytest.approx(sigma.value, rel=1e-9, abs=0.0)
        assert abs(total - sigma.value) <= 1e-9 * max(1.0, abs(sigma.value))


def test_contribution_shares_sum_to_one():
    """§2.5: ``share = contribution/total``, so the shares are a decomposition."""
    cols = _seeded_book(44)
    for rc in (es_contributions(cols, confidence=0.95), volatility_contributions(cols)):
        shares = [p.share for p in rc.per_position]
        assert all(s is not None for s in shares)
        assert math.fsum(shares) == pytest.approx(1.0, rel=1e-12)


def test_one_tail_size_definition_shared_across_modules():
    """§3.3 depends on ONE tail-size function: ``contribution.tail_size`` must
    BE ``var_es.tail_size``, not a second implementation that happens to
    agree on the platform grid.

    A previous ``n − floor(n·α)`` copy in ``contribution`` matched at
    α ∈ {0.95, 0.99} but diverged elsewhere (n=90, α=0.7 ⇒ 27 vs 28), which
    would silently break ``Σ RC^ES == ES`` for any non-grid confidence.
    """
    assert contribution_mod.tail_size is var_es_mod.tail_size
    for n in (1, 8, 90, 100, 250, 600, 1000):
        for alpha in (0.55, 0.7, 0.8, 0.95, 0.99, 0.995):
            assert contribution_mod.tail_size(n, alpha) == var_es_mod.tail_size(n, alpha)


def test_tail_size_matches_contract_worked_examples():
    """§2.3 states the k values by hand; float error in ``1 − α`` must not
    change them (``1 - 0.95 == 0.05000000000000004``)."""
    assert tail_size(600, 0.95) == 30
    assert tail_size(600, 0.99) == 6
    assert tail_size(250, 0.95) == 13
    assert tail_size(250, 0.99) == 3


def test_es_contributions_agree_with_es_off_the_platform_grid():
    """The sum invariant must hold at a confidence nobody tested by hand —
    this is what the shared ``tail_size`` buys."""
    cols = _seeded_book(55, n=90)
    portfolio = [math.fsum(col[t] for col in cols.values()) for t in range(90)]
    for alpha in (0.7, 0.66, 0.88):
        rc = es_contributions(cols, confidence=alpha, min_obs=60)
        es = historical_es(portfolio, alpha, min_obs=60)
        assert rc.total == es.value


# ---------------------------------------------------------------------------
# §3.4 — scaling and shift laws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [2.0, 0.5, 10.0])
def test_scaling_pnl_scales_var_es_and_sigma(factor):
    """§3.4: ``k·pnl`` ⇒ VaR/ES/σ scale by ``k`` (positive homogeneity)."""
    pnl = _seeded_pnl(101)
    scaled = [factor * p for p in pnl]
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        base = fn(pnl, 0.95).value
        assert fn(scaled, 0.95).value == pytest.approx(factor * base, rel=1e-12)
    assert portfolio_volatility(scaled).value == pytest.approx(
        factor * portfolio_volatility(pnl).value, rel=1e-12
    )


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_scaling_pnl_scales_contributions(factor):
    """§3.4 for the decomposition: every RC scales with the book."""
    cols = _seeded_book(202)
    scaled = {k: [factor * v for v in col] for k, col in cols.items()}
    for method, fn in (("ES", lambda c: es_contributions(c, confidence=0.95)),
                       ("VOL", volatility_contributions)):
        base_rc, scaled_rc = fn(cols), fn(scaled)
        assert scaled_rc.total == pytest.approx(factor * base_rc.total, rel=1e-12), method
        for b, s in zip(base_rc.per_position, scaled_rc.per_position):
            assert s.contribution == pytest.approx(factor * b.contribution, rel=1e-12)


def test_constant_gain_shifts_var_and_es_down_by_that_gain():
    """§3.4: adding a constant gain ``c`` to every day shifts a LOSS measure
    by ``−c`` — both the Gaussian pair (via μ) and historical VaR/ES (every
    loss in the tail moves by the same amount, so the tail set is unchanged).
    """
    pnl = _seeded_pnl(303)
    c = 500.0
    shifted = [p + c for p in pnl]
    for fn in (historical_var, historical_es, gaussian_var, gaussian_es):
        assert fn(shifted, 0.95).value == pytest.approx(fn(pnl, 0.95).value - c, rel=1e-9)


def test_constant_shift_leaves_volatility_unchanged():
    """§3.4 corollary: a deterministic drift carries no risk in a σ measure."""
    pnl = _seeded_pnl(404)
    shifted = [p + 777.0 for p in pnl]
    assert portfolio_volatility(shifted).value == pytest.approx(
        portfolio_volatility(pnl).value, rel=1e-9
    )


def test_multi_day_is_sqrt_time_scaled_and_labelled():
    """§1/§2.3: horizon > 1 is √h scaling of the 1-day number, LABELLED
    ``scaling="SQRT_TIME"`` so no reader mistakes it for an estimate."""
    pnl = _seeded_pnl(505)
    one = historical_var(pnl, 0.95, 1)
    five = historical_var(pnl, 0.95, 5)
    assert one.diagnostics["scaling"] == "NONE"
    assert five.diagnostics["scaling"] == "SQRT_TIME"
    assert five.value == pytest.approx(one.value * math.sqrt(5), rel=1e-12)
    assert five.meta.horizon_days == 5


# ---------------------------------------------------------------------------
# §3.5 — walk-forward never touches pnl[t]
# ---------------------------------------------------------------------------


def test_walk_forward_never_sees_the_day_it_forecasts():
    """§3.5, with the REAL ``historical_var`` estimator and a sentinel spike.

    A catastrophic loss is planted on one day. If the forecast for that day
    had seen it, the forecast would jump; it must be identical to the
    forecast produced from the un-spiked series, and it must change on the
    FOLLOWING day (proving the spike is in the data at all).

    The sentinel is ``historical_es``, deliberately NOT ``historical_var``:
    at n=60/α=0.95 the tail is k=3, so one spike only displaces the LARGEST
    loss while VaR reads the 3rd-largest — a VaR sentinel is inert here and
    would pass even against a leaking implementation. ES averages the whole
    tail, so it moves the moment the spike enters the window.
    """
    window = 60
    pnl = _seeded_pnl(606, n=200)
    spike_at = 120
    spiked = list(pnl)
    spiked[spike_at] = -1_000_000.0

    def estimator(sample):
        return historical_es(sample, 0.95, min_obs=window)

    clean = walk_forward(pnl, window=window, confidence=0.95, estimator=estimator)
    dirty = walk_forward(spiked, window=window, confidence=0.95, estimator=estimator)

    idx = clean.indices.index(spike_at)
    assert clean.indices[idx] == spike_at
    assert dirty.forecasts[idx] == clean.forecasts[idx], "forecast for t used pnl[t]"
    assert dirty.forecasts[idx + 1] != clean.forecasts[idx + 1], (
        "the spike never entered the estimation window at all — sentinel is inert"
    )
    # ...and it is a HUGE move once seen, so the guard above is not a rounding
    # coincidence.
    assert dirty.forecasts[idx + 1] > 100 * clean.forecasts[idx + 1]
    # Every forecast is paired with the day it predicts, never consumes it.
    assert dirty.realized[idx] == -1_000_000.0


def test_walk_forward_forecast_matches_a_hand_built_window():
    """§3.5 restated positively: the forecast for ``t`` is exactly the
    estimator applied to ``pnl[t-window:t]``."""
    window = 60
    pnl = _seeded_pnl(707, n=150)

    def estimator(sample):
        return historical_var(sample, 0.95, min_obs=window)

    fs = walk_forward(pnl, window=window, confidence=0.95, estimator=estimator)
    for offset, t in enumerate(range(window, len(pnl))):
        expected = estimator(pnl[t - window : t]).value
        assert fs.forecasts[offset] == expected


# ---------------------------------------------------------------------------
# §3.6 — below min_obs is an honest null, never an exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [historical_var, historical_es, gaussian_var, gaussian_es], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("n", [0, 1, 5, 59])
def test_below_min_obs_is_unavailable_not_an_exception(fn, n):
    """§3.6 / §1 honest nulls: insufficient data ⇒ ``value=None``,
    ``UNAVAILABLE``, a ``reason`` carrying the real numbers. Never a
    fabricated 0, never a raise."""
    pnl = _seeded_pnl(808, n=n)
    result = fn(pnl, 0.95)  # min_obs defaults to 60 at 95%
    assert result.value is None
    assert result.health is ModelHealth.UNAVAILABLE
    assert result.reason and str(n) in result.reason
    assert result.sample_size == n


def test_below_min_obs_is_unavailable_for_the_other_estimators():
    """§3.6 across the rest of the library — one shape for every module."""
    short = _seeded_pnl(909, n=10)
    assert portfolio_volatility(short).health is ModelHealth.UNAVAILABLE
    assert portfolio_volatility(short).value is None
    assert distribution_diagnostics(short).health is ModelHealth.UNAVAILABLE
    rc = es_contributions({"A": short, "B": short}, confidence=0.95)
    assert rc.health is ModelHealth.UNAVAILABLE and rc.total is None
    rcv = volatility_contributions({"A": short, "B": short})
    assert rcv.health is ModelHealth.UNAVAILABLE and rcv.total is None


def test_unavailable_results_always_carry_a_reason_with_numbers():
    """§1: the reason must be actionable (``"n=17 < min_obs=60"``), not a
    bare label."""
    result = historical_var(_seeded_pnl(1010, n=17), 0.95)
    assert result.reason is not None
    assert "n=17" in result.reason and "min_obs=60" in result.reason


# ---------------------------------------------------------------------------
# §3.7 — log_returns moved, not forked
# ---------------------------------------------------------------------------


def test_correlation_log_returns_is_the_returns_module_function():
    """§3.7: ``correlation.py`` RE-EXPORTS the moved function; two copies
    would let the correlation and volatility conventions drift apart."""
    assert correlation.log_returns is returns_mod.log_returns


def test_moved_log_returns_still_matches_hand_vectors():
    """§3.7: byte-identical behaviour on the vectors the old tests used."""
    closes = [100.0, 110.0, 99.0, 105.0]
    got = correlation.log_returns(closes)
    assert got == [
        math.log(110.0 / 100.0),
        math.log(99.0 / 110.0),
        math.log(105.0 / 99.0),
    ]
    with pytest.raises(ValueError):
        correlation.log_returns([100.0, 0.0])


# ---------------------------------------------------------------------------
# §3.8 — validate() never upgrades, for every registered model
# ---------------------------------------------------------------------------


def test_validate_never_upgrades_for_every_registered_model():
    """§3.8 / spec §57: validation is a separate step that may only DOWNGRADE.

    Every registered model is handed a deliberately UNAVAILABLE result; none
    may hand back something healthier.
    """
    assert base_mod.REGISTRY, "registry is empty — models did not register"
    for name, model in sorted(base_mod.REGISTRY.items()):
        meta = model.metadata()
        pessimistic = base_mod.unavailable(meta, "forced for the validate guard", 0)
        after = model.validate(pessimistic)
        assert base_mod.health_rank(after.health) >= base_mod.health_rank(
            pessimistic.health
        ), f"{name}.validate() upgraded {pessimistic.health} -> {after.health}"
        assert after.value is None, f"{name}.validate() conjured a value"


def test_validate_preserves_health_on_a_real_calculation():
    """A healthy result stays healthy — ``validate`` downgrades, it does not
    churn."""
    pnl = _seeded_pnl(1111)
    for name, model in sorted(base_mod.REGISTRY.items()):
        result = model.calculate(pnl)
        after = model.validate(result)
        assert base_mod.health_rank(after.health) >= base_mod.health_rank(result.health)


def test_validate_never_upgrades_guard_rejects_an_upgrade():
    """The guard itself must bite, or §3.8 is unprotected."""
    meta = base_mod.ModelMeta(model_name="m", model_version="1.0.0")
    before = base_mod.unavailable(meta, "no data", 0)
    after = base_mod.active(meta, 1.0, 10)
    with pytest.raises(ValueError, match="never upgrade"):
        base_mod.validate_never_upgrades(before, after)


# ---------------------------------------------------------------------------
# §3.9 — filtered HS collapses to plain HS as lambda -> 1
# ---------------------------------------------------------------------------


def test_filtered_hs_converges_to_plain_hs_as_lambda_approaches_one():
    """§3.9 limit sanity: λ → 1 freezes the EWMA σ path, so every Hull–White
    ratio ``σ_now/σ_t`` → 1 and the conditional VaR tends to the plain
    historical VaR of the post-warm-up tail. Monotone convergence, not just
    a single point."""
    pnl = _seeded_pnl(1212, n=400)
    init_obs = 20
    plain = historical_var(pnl[init_obs:], 0.95).value
    errors = []
    for lam in (0.94, 0.99, 0.999, 0.99999):
        cond = conditional_var(pnl, 0.95, lam=lam, init_obs=init_obs).value
        errors.append(abs(cond - plain) / abs(plain))
    assert errors == sorted(errors, reverse=True), f"not converging: {errors}"
    assert errors[-1] < 1e-3, f"λ=0.99999 still {errors[-1]:.2e} from plain HS"


def test_constant_absolute_pnl_gives_a_flat_ewma_sigma():
    """§3.9 second half: with constant ``|pnl|`` the EWMA σ path is flat, so
    every scaling ratio is exactly 1 and filtering is the identity."""
    pnl = [100.0 if t % 2 else -100.0 for t in range(200)]
    scaled = volatility_mod.volatility_scaled_pnl(pnl, lam=0.94, init_obs=20)
    assert scaled == pytest.approx(pnl[20:], rel=1e-12)


# ---------------------------------------------------------------------------
# spec §70 — the registry is SHADOW-only
# ---------------------------------------------------------------------------


def test_registry_holds_the_four_var_es_models_all_in_shadow_mode():
    """Spec §70 / contract §2.2: these models must be impossible to wire
    into a veto. The engine checks ``mode``; every one of them is SHADOW."""
    for name in ("historical_var", "historical_es", "gaussian_var", "gaussian_es"):
        model = base_mod.get(name)
        assert model.mode is ModelMode.SHADOW, f"{name} is {model.mode}, not SHADOW"
        assert model.name == name
        assert model.version == "1.0.0"


def test_no_registered_model_is_in_production_mode():
    """Phase B ships nothing in PRODUCTION — promotion follows validation
    (spec §68), never a default."""
    for name, model in base_mod.REGISTRY.items():
        assert model.mode is not ModelMode.PRODUCTION, f"{name} is PRODUCTION"


def test_tier_0_engine_is_not_importable_from_the_model_layer():
    """The SHADOW library must not reach into the Tier 0 decision engine —
    that direction of dependency is how a shadow number becomes a veto."""
    import inspect

    for mod in (var_es_mod, volatility_mod, contribution_mod, returns_mod):
        source = inspect.getsource(mod)
        assert "risk.engine" not in source and "from .engine" not in source


# ---------------------------------------------------------------------------
# §3.10 — end-to-end mini pipeline
# ---------------------------------------------------------------------------


def _seeded_closes(seed: int, n: int, start: float) -> list[float]:
    rnd = random.Random(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(round(closes[-1] * (1.0 + rnd.gauss(0.0004, 0.014)), 4))
    return closes


def _pipeline_inputs():
    """3 tickers × 300 daily closes → aligned SIMPLE returns → a 3-position
    book (long stock, short stock, long call ×100 at delta 0.5)."""
    n = 300
    day0 = date(2025, 1, 2)
    dates = [day0 + timedelta(days=i) for i in range(n)]
    specs = {"AAPL": (11, 190.0), "MSFT": (22, 410.0), "NVDA": (33, 120.0)}
    series = [
        returns_from_closes(
            ticker,
            list(zip(dates, _seeded_closes(seed, n, start))),
            return_type="SIMPLE",
        )
        for ticker, (seed, start) in specs.items()
    ]
    matrix = align(series)
    positions = [
        PositionRiskInput("AAPL#1", "AAPL", "STOCK", 300, 1, 190.0, 1.0, 57_000.0),
        PositionRiskInput("MSFT#2", "MSFT", "STOCK", -100, 1, 410.0, 1.0, 41_000.0),
        PositionRiskInput("NVDA#3", "NVDA", "CALL", 10, 100, 120.0, 0.5, 6_000.0),
    ]
    return matrix, positions


def test_end_to_end_pipeline_builds_a_coherent_snapshot():
    """§3.10: the whole library on one book — every invariant that matters
    across modules, asserted on real wiring rather than synthetic inputs."""
    matrix, positions = _pipeline_inputs()
    assert matrix.return_type == "SIMPLE"
    assert matrix.n_obs == 299  # 300 closes ⇒ 299 returns, fully aligned
    assert matrix.tickers == ("AAPL", "MSFT", "NVDA")

    book = book_pnl_series(positions, matrix)
    assert book.method == "DELTA_LINEAR"
    assert book.tickers_missing == () and book.keys_excluded == ()
    assert set(book.per_position) == {"AAPL#1", "MSFT#2", "NVDA#3"}
    # The short leg moves opposite the long: signs are carried, not absolute.
    assert book.per_position["MSFT#2"][0] * matrix.column("MSFT")[0] < 0
    # total is the fsum of the legs, day by day.
    for t in range(book.n_obs):
        assert book.total[t] == math.fsum(s[t] for s in book.per_position.values())

    pnl = book.total
    as_of_date = matrix.as_of

    var_results = {
        "HISTORICAL:0.95:1": historical_var(pnl, 0.95, as_of=as_of_date),
        "HISTORICAL:0.99:1": historical_var(pnl, 0.99, min_obs=250, as_of=as_of_date),
        "GAUSSIAN:0.95:1": gaussian_var(pnl, 0.95, as_of=as_of_date),
    }
    es_results = {
        "HISTORICAL:0.95:1": historical_es(pnl, 0.95, as_of=as_of_date),
        "GAUSSIAN:0.95:1": gaussian_es(pnl, 0.95, as_of=as_of_date),
    }
    sigma = portfolio_volatility(pnl, as_of=as_of_date)
    rc_es = es_contributions(book.per_position, confidence=0.95, as_of=as_of_date)
    rc_vol = volatility_contributions(book.per_position, as_of=as_of_date)
    dist = distribution_diagnostics(pnl, as_of=as_of_date)

    # --- the cross-module invariants, on the real book -----------------
    assert es_results["HISTORICAL:0.95:1"].value >= var_results["HISTORICAL:0.95:1"].value
    assert es_results["GAUSSIAN:0.95:1"].value >= var_results["GAUSSIAN:0.95:1"].value
    assert rc_es.total == es_results["HISTORICAL:0.95:1"].value      # §3.3 exact
    assert math.fsum(p.contribution for p in rc_vol.per_position) == pytest.approx(
        sigma.value, rel=1e-9
    )
    assert rc_es.tail_size == tail_size(book.n_obs, 0.95)

    # --- ensemble views over the coherent set ---------------------------
    views = {
        "historical_var_95": var_results["HISTORICAL:0.95:1"],
        "gaussian_var_95": var_results["GAUSSIAN:0.95:1"],
    }
    disp = dispersion(views)
    assert disp.ratio is not None and disp.ratio >= 1.0
    assert disp.n_views == 2 and disp.n_comparable == 2
    risk = model_risk_state(
        views,
        dispersion_result=disp,
        gaussian_trust=dist.gaussian_trust,
        gaussian_views=("gaussian_var_95",),
        core_views=("historical_var_95",),
        as_of=as_of_date,
    )
    assert risk.state in ("LOW", "ELEVATED", "HIGH")

    dd = reconstructed_book_drawdown(list(zip(book.dates, pnl)), nav_now=250_000.0)
    assert dd.method == "RECONSTRUCTED_CURRENT_BOOK"
    assert dd.current_dd_pct is None or dd.current_dd_pct <= 0.0
    assert dd.max_dd_pct is None or dd.max_dd_pct <= 0.0

    # --- the typed snapshot --------------------------------------------
    snapshot = PortfolioRiskSnapshot(
        as_of=datetime(2025, 10, 28, 16, 0, 0),
        nav=250_000.0,
        cash=50_000.0,
        cash_reserved=0.0,
        gross_exposure=sum(abs(p.exposure) for p in positions),
        delta_adjusted_exposure=math.fsum(p.exposure for p in positions),
        heat_pct=0.31,
        heat_state="NORMAL",
        volatility=sigma,
        var=var_results,
        es=es_results,
        drawdown=dd,
        contributions_vol=rc_vol,
        contributions_es=rc_es,
        distribution=dist,
        dispersion=disp,
        model_risk=risk,
        data_quality=DataQuality(
            as_of=as_of_date,
            oldest_bar=matrix.dates[0],
            newest_bar=matrix.dates[-1],
            n_obs=matrix.n_obs,
            tickers_missing=book.tickers_missing,
        ),
    )
    assert snapshot.risk_state == "NORMAL"  # defaults to heat_state
    assert snapshot.snapshot_version
    summary = snapshot.health_summary()
    for key in ("volatility", "var:HISTORICAL:0.95:1", "es:HISTORICAL:0.95:1",
                "contributions_vol", "contributions_es", "distribution", "dispersion"):
        assert key in summary, f"health_summary missing {key}"
        assert isinstance(summary[key], ModelHealth)
    assert snapshot.overall_health() in tuple(ModelHealth)


def test_every_pipeline_result_carries_reproducible_metadata():
    """Spec §44 / contract §4: a stored number must be reproducible, so every
    ``ModelResult`` names its model, version, confidence, horizon,
    distribution and as_of."""
    matrix, positions = _pipeline_inputs()
    pnl = book_pnl_series(positions, matrix).total
    as_of_date = matrix.as_of

    results = {
        "historical_var": historical_var(pnl, 0.95, as_of=as_of_date),
        "historical_es": historical_es(pnl, 0.95, as_of=as_of_date),
        "gaussian_var": gaussian_var(pnl, 0.95, as_of=as_of_date),
        "gaussian_es": gaussian_es(pnl, 0.95, as_of=as_of_date),
        "historical_var_5d": historical_var(pnl, 0.99, 5, min_obs=250, as_of=as_of_date),
    }
    for label, result in results.items():
        meta = result.meta
        assert isinstance(result, ModelResult)
        assert meta.model_name and meta.model_version == "1.0.0", label
        assert meta.confidence in (0.95, 0.99), label
        assert meta.horizon_days is not None and meta.horizon_days >= 1, label
        assert meta.distribution in ("EMPIRICAL", "NORMAL"), label
        assert meta.as_of == as_of_date, label
        assert meta.lookback == result.sample_size, label
        # params must be enough to re-run the estimator by hand
        for key in ("confidence", "horizon_days", "min_obs", "scaling"):
            assert key in meta.params, f"{label} missing params[{key}]"


def test_pipeline_degrades_honestly_when_a_ticker_is_missing():
    """§2.9: a position whose ticker has no column is EXCLUDED and NAMED —
    never silently priced at zero."""
    matrix, positions = _pipeline_inputs()
    positions = positions + [
        PositionRiskInput("TSLA#9", "TSLA", "STOCK", 50, 1, 250.0, 1.0, 12_500.0)
    ]
    book = book_pnl_series(positions, matrix)
    assert book.tickers_missing == ("TSLA",)
    assert book.keys_excluded == ("TSLA#9",)
    assert "TSLA#9" not in book.per_position
    # DataQuality.valid is the BUILDER's verdict (the dataclass does not
    # infer it), and a builder that reports missing tickers must set it —
    # ``valid=False`` without a reason is refused, so the gap is always named.
    quality = DataQuality(
        as_of=matrix.as_of,
        oldest_bar=matrix.dates[0],
        newest_bar=matrix.dates[-1],
        n_obs=matrix.n_obs,
        tickers_missing=book.tickers_missing,
        valid=False,
        reasons=(f"tickers_missing={book.tickers_missing}",),
    )
    assert quality.valid is False and quality.reasons
    with pytest.raises(ValueError, match="requires at least one reason"):
        DataQuality(as_of=None, oldest_bar=None, newest_bar=None, valid=False)
    # A snapshot carrying invalid data quality can never claim full health.
    snapshot = PortfolioRiskSnapshot(
        as_of=datetime(2025, 10, 28, 16, 0, 0),
        nav=250_000.0, cash=0.0, cash_reserved=0.0,
        gross_exposure=0.0, delta_adjusted_exposure=0.0,
        heat_pct=0.1, heat_state="NORMAL",
        volatility=portfolio_volatility(book.total),
        data_quality=quality,
    )
    assert snapshot.overall_health() is not ModelHealth.ACTIVE
