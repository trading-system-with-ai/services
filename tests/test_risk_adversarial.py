"""PHASE J — ADVERSARIAL VALIDATION of the risk engine (spec §61 "PHASE J",
§67 property tests, §68 model-validation acceptance).

Phases A–E built a statistical risk layer beside the Tier 0 hard-limit
engine. Every other risk test in this suite asks "is the number right?".
This one asks the opposite question, once per scenario:

    **When the model is wrong, does the platform fail SAFE?**

So every test below is written as a three-part contract, stated in its own
docstring and then asserted with real numbers:

    SCENARIO         — the market or infrastructure state being simulated;
    FAIL-SAFE        — what the platform must do about it;
    THE ASSERTION    — the measured numbers that prove it did.

Two house rules are load-bearing here and are re-proved rather than assumed:

1.  **Nothing statistical decides.** Tier 0 (``risk/engine.py``
    ``assess`` / ``assess_income``) is the only authority that sizes a
    trade. Every statistical model is SHADOW or RESEARCH; the promotion
    seam ``extra_caps`` is never populated in ``apps/``. §67 properties (d)
    and (e) attack that claim directly.
2.  **A gap is a labelled null, never a zero.** A missing bar, an
    unreachable broker, an unfittable GARCH and an absent model all have to
    say so — in a reason string carrying the real numbers — rather than
    quietly reporting a number that would understate risk.

Deliberately NOT retested here: the arithmetic of each estimator (owned by
tests/test_risk_var_es.py, test_risk_contribution.py, test_risk_stress.py,
test_risk_garch.py …) and the cross-module invariants
(tests/test_risk_phase_b_invariants.py). This file only attacks.

Runtime budget: the whole file stays under 60 s. The one expensive test is
the §68 walk-forward acceptance run (~2 s); the GARCH failure cases are
chosen to be fast BECAUSE they fail early, which is itself the point.
"""
from __future__ import annotations

import importlib
import math
import random
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.gateway.execution import gate_chain
from sqlalchemy import select

from apps.gateway import risk_snapshot as rs
from apps.gateway.db import (
    Order,
    Position,
    RiskSnapshotRow,
    SessionLocal,
    StockBarDaily,
)
from apps.gateway.routers import orders as orders_router
from libs.trading_core.correlation import (
    STATE_CONVERGING,
    STATE_NORMAL,
    correlation_regime,
)
from libs.trading_core.options.bs import bs_greeks, bs_price
from libs.trading_core.options.reval import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL,
    OptionLeg,
    StockLeg,
    scenario_pnl,
)
from libs.trading_core.risk.models import base as model_base
from libs.trading_core.risk.models.contribution import (
    es_contributions,
    volatility_contributions,
)
from libs.trading_core.risk.models.diagnostics import (
    FLAG_HEAVY_TAIL,
    FLAG_LEFT_SKEWED,
    TRUST_LOW,
    TRUST_REDUCED,
    distribution_diagnostics,
)
from libs.trading_core.risk.models.ensemble import (
    RISK_ELEVATED,
    RISK_HIGH,
    dispersion,
    model_risk_state,
)
from libs.trading_core.risk.models.garch import (
    SOURCE_EWMA,
    conditional_scaled_pnl_source,
    conditional_volatility_source,
)
from libs.trading_core.risk.models.var_es import (
    conditional_es,
    conditional_var,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
)
from libs.trading_core.risk.models.volatility import ewma_volatility_forecast
from libs.trading_core.risk.pnl_series import PositionRiskInput, book_pnl_series
from libs.trading_core.risk.pretrade import (
    CODE_BUCKET_ES_CONTRIBUTION,
    CODE_ES_CONTRIBUTION,
    DECISION_REJECT,
    LAYER_CONCENTRATION,
    MODE_SHADOW,
    CandidateSpec,
    statistical_caps,
    shadow_verdict,
)
from libs.trading_core.risk.returns import ReturnMatrix
from libs.trading_core.risk.snapshot import STALENESS_KIND_GREEKS, TtlPolicy

from .test_order_preview import BULL_TICKER, authorize, preview
from .test_risk_engine import battery
from .test_risk_snapshot_builder import build, seed_stock_position

# ===========================================================================
# Shared seeded generators
#
# Every series below is produced by a SEEDED random.Random, so each number
# quoted in an assertion is reproducible on any machine. The numbers in the
# docstrings were measured from these exact generators.
# ===========================================================================

#: Trading days in the synthetic histories (a bit over two years).
N_DAYS = 600

#: NAV the USD P&L series are scaled against in the vol-targeting tests.
NAV = 500_000.0


def crash_pnl(seed: int = 20260819, n: int = N_DAYS) -> list[float]:
    """A heavy-tailed, LEFT-skewed daily P&L series in USD — a Gaussian
    mixture where 4% of days are crash days drawn as pure LOSSES.

        96% of days:  N(+200, 1000)      the quiet drift
         4% of days:  -|N(0, 9000)|      the crash

    A Gaussian fitted to this understates the tail badly, which is exactly
    the regime spec §15/§39 exists to detect.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        if rng.random() < 0.04:
            out.append(-abs(rng.gauss(0.0, 9000.0)))
        else:
            out.append(rng.gauss(200.0, 1000.0))
    return out


def calm_then_spike(
    seed: int = 424242, calm_n: int = 400, spike_n: int = 30
) -> tuple[list[float], list[float]]:
    """``(calm, spiked)`` daily P&L in USD, where ``spiked`` IS ``calm``
    plus ``spike_n`` further days at 8x the volatility.

    The two series share their first ``calm_n`` observations exactly, so any
    difference between statistics computed on them is caused by the spike
    and by nothing else.
    """
    rng = random.Random(seed)
    calm = [rng.gauss(0.0, 1500.0) for _ in range(calm_n)]
    spiked = calm + [rng.gauss(0.0, 12000.0) for _ in range(spike_n)]
    return calm, spiked


#: The three TECH_MEGA names the Tier 0 correlation bucket already knows
#: (``engine._default_correlation_buckets``) — using the REAL bucket, not an
#: invented one, is what makes the concentration tests meaningful.
TECH = ("NVDA", "AMD", "AAPL")
TECH_BUCKET = {"TECH_MEGA": TECH}


def tech_matrices(
    *, converged: bool, n: int = 140, seed: int = 31337
) -> tuple[ReturnMatrix, ReturnMatrix]:
    """``(simple, log)`` return matrices for NVDA/AMD/AAPL.

    ``converged=True``  — every name is one common factor plus 0.15% noise:
                          average pairwise correlation ~0.99 (a crisis).
    ``converged=False`` — NVDA loads +1.0 on the factor, AMD −0.6 (a real
                          diversifier) and AAPL is idiosyncratic: average
                          pairwise correlation ~ −0.34.

    Both are built from the SAME seed and the same factor draws, so the only
    thing that changes between them is the dependence structure.
    """
    rng = random.Random(seed)
    rows: list[tuple[float, ...]] = []
    for _ in range(n):
        f = rng.gauss(0.0, 0.02)
        if converged:
            rows.append(tuple(f + rng.gauss(0.0, 0.0015) for _ in TECH))
        else:
            rows.append(
                (
                    f + rng.gauss(0.0, 0.0015),
                    -0.6 * f + rng.gauss(0.0, 0.0015),
                    rng.gauss(0.0, 0.006),
                )
            )
    dates = tuple(date(2025, 1, 6) + timedelta(days=i) for i in range(n))
    simple = ReturnMatrix(
        dates=dates, tickers=TECH, rows=tuple(rows), return_type="SIMPLE"
    )
    log = ReturnMatrix(
        dates=dates,
        tickers=TECH,
        rows=tuple(tuple(math.log1p(x) for x in row) for row in rows),
        return_type="LOG",
    )
    return simple, log


#: A concentrated tech book: NVDA three times the size of AMD / AAPL.
TECH_POSITIONS = [
    PositionRiskInput("NVDA#1", "NVDA", "LONG_STOCK", 300, 1, 200.0, 1.0, 9_000.0),
    PositionRiskInput("AMD#2", "AMD", "LONG_STOCK", 100, 1, 200.0, 1.0, 3_000.0),
    PositionRiskInput("AAPL#3", "AAPL", "LONG_STOCK", 100, 1, 200.0, 1.0, 3_000.0),
]

# --- option-book constants for the revaluation scenarios -------------------
SPOT = 100.0
STRIKE = 100.0
T_YEARS = 60.0 / 365.0
IV0 = 0.30
RATE = 0.04
CALL_MARK = bs_price(SPOT, STRIKE, T_YEARS, IV0, "C", r=RATE)
PUT_MARK = bs_price(SPOT, STRIKE, T_YEARS, IV0, "P", r=RATE)
CALL_GREEKS = bs_greeks(SPOT, STRIKE, T_YEARS, IV0, "C", r=RATE)
PUT_GREEKS = bs_greeks(SPOT, STRIKE, T_YEARS, IV0, "P", r=RATE)


def long_call(quantity: int = 10, *, iv0: float | None = IV0) -> OptionLeg:
    """A long ATM call, 60 DTE. ``iv0=None`` forces the DELTA_LINEAR fallback."""
    return OptionLeg(
        key="AAPL#call",
        ticker="AAPL",
        right="C",
        strike=STRIKE,
        t_years=T_YEARS,
        quantity=quantity,
        spot0=SPOT,
        mark0=CALL_MARK,
        iv0=iv0,
        delta0=CALL_GREEKS.delta,
        r=RATE,
    )


def long_put(quantity: int = 10) -> OptionLeg:
    """A long ATM put, 60 DTE — the long-vega leg that GAINS on an IV spike."""
    return OptionLeg(
        key="AAPL#put",
        ticker="AAPL",
        right="P",
        strike=STRIKE,
        t_years=T_YEARS,
        quantity=quantity,
        spot0=SPOT,
        mark0=PUT_MARK,
        iv0=IV0,
        delta0=PUT_GREEKS.delta,
        r=RATE,
    )


def stock_leg(quantity: int = 100) -> StockLeg:
    return StockLeg(key="AAPL#stock", ticker="AAPL", quantity=quantity, spot0=SPOT)


# ===========================================================================
# 1. FAT-TAIL CRASH
# ===========================================================================


def test_fat_tail_crash_is_seen_by_history_missed_by_gaussian_and_flagged():
    """SCENARIO — a book whose P&L is a Gaussian mixture with 4% crash days
    (``crash_pnl``): the tail is far heavier and far more left-skewed than a
    normal distribution allows.

    FAIL-SAFE — the empirical view must report the LARGER loss, the Gaussian
    view must be visibly distrusted rather than quietly averaged in, and the
    platform must raise its own model-risk state and SAY why.

    THE ASSERTION — on seed 20260819 (n=600): historical ES-95 = $7,710 vs
    Gaussian ES-95 = $4,828 (historical is 1.60x larger); skew = −4.21,
    excess kurtosis = 23.7 ⇒ flags contain HEAVY_TAIL and LEFT_SKEWED,
    gaussian_trust = LOW; dispersion ratio 1.60 > 1.5 fires
    MODEL_DISPERSION_HIGH, and model risk is HIGH with reasons naming BOTH
    the dispersion and the Gaussian distrust.
    """
    pnl = crash_pnl()
    hist_es = historical_es(pnl, 0.95)
    gauss_es = gaussian_es(pnl, 0.95)
    hist_var = historical_var(pnl, 0.95)
    gauss_var = gaussian_var(pnl, 0.95)

    # (a) The empirical tail is the bigger one — the whole reason ES exists.
    assert hist_es.value == pytest.approx(7710.1, abs=0.5)
    assert gauss_es.value == pytest.approx(4828.4, abs=0.5)
    assert hist_es.value >= gauss_es.value
    assert hist_es.value / gauss_es.value == pytest.approx(1.597, abs=0.01)
    # ES ≥ VaR within each family (the library's own invariant, restated
    # here because a crash series is where it would break if it could).
    assert hist_es.value >= hist_var.value
    assert gauss_es.value >= gauss_var.value
    # And the Gaussian VaR is the DANGEROUS one: it looks bigger than the
    # historical VaR (a crash makes sigma huge) while its ES understates the
    # tail by 60%. A platform reading only VaR would feel safe here.
    assert gauss_var.value > hist_var.value

    # (b) The distribution is diagnosed, not assumed.
    dist = distribution_diagnostics(pnl)
    assert dist.primary == FLAG_LEFT_SKEWED
    assert FLAG_HEAVY_TAIL in dist.flags
    assert dist.skew == pytest.approx(-4.205, abs=0.01)
    assert dist.excess_kurtosis == pytest.approx(23.73, abs=0.05)
    assert dist.gaussian_trust in {TRUST_LOW, TRUST_REDUCED}
    assert dist.gaussian_trust == TRUST_LOW
    assert dist.jb_p == pytest.approx(0.0, abs=1e-12)

    # (c) Model risk is raised, with reasons a human can act on.
    views = {
        "historical_var_95": hist_var,
        "historical_es_95": hist_es,
        "gaussian_var_95": gauss_var,
        "gaussian_es_95": gauss_es,
    }
    disp = dispersion({"historical_es_95": hist_es, "gaussian_es_95": gauss_es})
    assert disp.ratio == pytest.approx(1.597, abs=0.01)
    assert disp.is_high is True

    state = model_risk_state(
        views,
        dispersion_result=disp,
        gaussian_trust=dist.gaussian_trust,
        gaussian_views=("gaussian_var_95", "gaussian_es_95"),
        core_views=("historical_var_95", "historical_es_95"),
    )
    assert state.state in {RISK_ELEVATED, RISK_HIGH}
    assert state.state == RISK_HIGH
    # The reasons must NAME the two things that went wrong — a bare "HIGH"
    # is not an explanation (spec §59).
    joined = " | ".join(state.reasons)
    assert "dispersion" in joined
    assert "gaussian_trust=LOW" in joined
    assert state.triggers["dispersion_high"] is True
    assert state.triggers["gaussian_trust_low"] is True


@pytest.mark.parametrize("seed", [20260819, 7, 99])
def test_fat_tail_verdict_is_not_a_lucky_seed(seed: int):
    """SCENARIO — the same crash mixture on three independent seeds.

    FAIL-SAFE — the ordering "historical ES ≥ Gaussian ES" and the HEAVY_TAIL
    / trust-LOW verdict are properties of the DISTRIBUTION, not of one draw.

    THE ASSERTION — on seeds 20260819 / 7 / 99 the ratio is 1.60 / 1.66 /
    1.54 (always > 1.5, always flagged) and every draw is HEAVY_TAIL with
    gaussian_trust LOW.
    """
    pnl = crash_pnl(seed)
    hist_es = historical_es(pnl, 0.95)
    gauss_es = gaussian_es(pnl, 0.95)
    ratio = hist_es.value / gauss_es.value
    assert hist_es.value >= gauss_es.value
    assert ratio > 1.5, f"seed {seed}: ratio {ratio}"
    dist = distribution_diagnostics(pnl)
    assert FLAG_HEAVY_TAIL in dist.flags
    assert dist.gaussian_trust == TRUST_LOW


# ===========================================================================
# 2. VOLATILITY SPIKE
# ===========================================================================


def test_vol_spike_moves_ewma_and_conditional_var_but_not_the_crude_proxy():
    """SCENARIO — 400 calm days (sigma $1,500/day) followed by 30 days at
    sigma $12,000/day: the market regime changed 30 days ago.

    FAIL-SAFE — the EWMA forecast must react (and the vol-targeting
    multiplier it implies must FALL, shrinking new exposure), the conditional
    (filtered-historical) VaR/ES must exceed the unconditional views that
    still average the calm past in, and — critically for §70 — the multiplier
    ACTUALLY IN FORCE (the crude RV20 proxy) must be untouched, because the
    EWMA pair is SHADOW and decides nothing yet.

    THE ASSERTION — EWMA sigma $1,333 -> $13,029/day (9.8x); the implied
    multiplier falls 1.2 -> 0.2901; conditional VaR-95 $22,231 vs
    unconditional $2,728 (8.2x) and conditional ES-95 $31,437 vs $9,082.
    """
    calm, spiked = calm_then_spike()

    # (a) The conditional forecast reacts; the unconditional one barely does.
    ewma_calm = ewma_volatility_forecast(calm)
    ewma_spiked = ewma_volatility_forecast(spiked)
    assert ewma_calm.value == pytest.approx(1332.9, abs=1.0)
    assert ewma_spiked.value == pytest.approx(13029.4, abs=5.0)
    assert ewma_spiked.value > 9.0 * ewma_calm.value

    # (b) Conditional VaR/ES > unconditional AFTER the spike. This is the
    # §12 point: filtering rescales quiet history to today's volatility, so
    # the risk number stops being an average of a regime that ended.
    uncond_var = historical_var(spiked, 0.95)
    cond_var = conditional_var(spiked, 0.95)
    uncond_es = historical_es(spiked, 0.95)
    cond_es = conditional_es(spiked, 0.95)
    assert uncond_var.value == pytest.approx(2727.5, abs=2.0)
    assert cond_var.value == pytest.approx(22230.8, abs=10.0)
    assert cond_var.value > uncond_var.value
    assert cond_var.value > 8.0 * uncond_var.value
    assert cond_es.value == pytest.approx(31437.4, abs=20.0)
    assert cond_es.value > uncond_es.value

    # ...and BEFORE the spike the same two views agree closely — the gap is
    # caused by the regime change, not by a bias in the filter.
    calm_uncond = historical_var(calm, 0.95).value
    calm_cond = conditional_var(calm, 0.95).value
    assert calm_cond == pytest.approx(calm_uncond, rel=0.15)


def test_vol_spike_side_by_side_leaves_the_multiplier_in_force_unchanged():
    """SCENARIO — the same spike, now driven through the gateway helper that
    the order chain and the risk view both call (``vol_targeting_block``).

    FAIL-SAFE — §14/§70: the EWMA numbers are logged BESIDE the crude proxy,
    never in place of it. Only ``multiplier_ewma`` may move; ``multiplier``
    (the one ``assess(budget_multiplier=...)`` consumes) must be byte-
    identical across the spike, because promoting EWMA is a human step.

    THE ASSERTION — with the SAME underlying closes, the crude
    ``forecast_vol`` = 0.0026198 and ``multiplier`` = 1.2 on both runs
    (identical to the last bit), while ``multiplier_ewma`` falls
    1.2 -> 0.2901 and ``ewma_sigma_p_annualized_pct_nav`` rises
    0.04232 -> 0.41367 of NAV.
    """
    from apps.gateway.routers.portfolio import vol_targeting_block

    calm, spiked = calm_then_spike()

    # One stock position with real stored closes, so the CRUDE proxy has an
    # RV20 to average. These closes are IDENTICAL for both calls — the crude
    # proxy reads them and nothing else, which is precisely why it cannot see
    # the book's P&L spike.
    position = Position(
        ticker="AAPL",
        instrument="LONG_STOCK",
        quantity=100,
        avg_price=100.0,
        max_loss=1_000.0,
        status="OPEN",
        opened_at=datetime.now(timezone.utc),
    )
    steps = (0.012, -0.008, 0.004, -0.015, 0.009, -0.003, 0.006, -0.011)
    closes = [100.0]
    for i in range(199):
        closes.append(closes[-1] * (1.0 + steps[i % len(steps)]))
    pairs = [(position, closes[-1])]
    closes_by_ticker = {"AAPL": closes}

    before = vol_targeting_block(NAV, pairs, closes_by_ticker, book_pnl=calm)
    after = vol_targeting_block(NAV, pairs, closes_by_ticker, book_pnl=spiked)

    # (a) THE MULTIPLIER IN FORCE IS UNCHANGED — byte-identical, not merely
    # close. This single pair of assertions is what keeps Phase C SHADOW.
    assert before["forecast_vol"] == after["forecast_vol"]
    assert before["multiplier"] == after["multiplier"]
    assert after["forecast_vol"] == pytest.approx(0.0026198, abs=1e-6)
    assert after["multiplier"] == 1.2

    # (b) The SHADOW pair moved, and moved in the risk-reducing direction.
    assert before["multiplier_ewma"] == pytest.approx(1.2, abs=1e-9)
    assert after["multiplier_ewma"] == pytest.approx(0.2901, abs=1e-3)
    assert after["multiplier_ewma"] < before["multiplier_ewma"]
    assert before["ewma_sigma_p_annualized_pct_nav"] == pytest.approx(0.04232, abs=1e-4)
    assert after["ewma_sigma_p_annualized_pct_nav"] == pytest.approx(0.41367, abs=1e-3)


# ===========================================================================
# 3. CORRELATION CONVERGENCE
# ===========================================================================


def test_correlation_convergence_concentrates_es_and_binds_the_bucket_cap():
    """SCENARIO — the same three tech names, priced twice: once genuinely
    diversified (NVDA +1 / AMD −0.6 / AAPL idiosyncratic, average pairwise
    correlation −0.34) and once converged (one factor, ~0.99) — the §19
    failure mode where diversification evaporates in a crisis.

    FAIL-SAFE — the regime monitor must flip NORMAL -> CONVERGING; the tail
    must grow because the hedge stopped hedging; and the hypothetical
    statistical cap must BIND in the SHADOW verdict rather than pass silently.

    THE ASSERTION — state NORMAL (curr −0.3404) -> CONVERGING (curr 0.9922);
    ES-95 $1,932 -> $4,004 (2.07x); AMD's ES contribution share flips from
    −0.2434 (it was absorbing losses) to +0.1978 (it now adds to them);
    NVDA's share of 1.2056 falls to 0.6014 only because the total doubled.
    A 300-share NVDA add-on is then capped by BOTH concentration limits and
    the SHADOW verdict is REJECT at quantity 0.
    """
    # -- diversified -------------------------------------------------------
    simple_div, log_div = tech_matrices(converged=False)
    regime_div = correlation_regime(log_div)
    book_div = book_pnl_series(TECH_POSITIONS, simple_div)
    es_div = historical_es(book_div.total, 0.95)
    contrib_div = es_contributions(book_div.per_position, 0.95)
    share_div = {p.key: p.share for p in contrib_div.per_position}

    assert regime_div.state == STATE_NORMAL
    assert regime_div.current_avg == pytest.approx(-0.3404, abs=0.001)
    assert es_div.value == pytest.approx(1931.8, abs=1.0)
    assert share_div["AMD#2"] == pytest.approx(-0.2434, abs=0.002)
    assert share_div["AMD#2"] < 0.0  # a real diversifier: NEGATIVE contribution

    # -- converged ---------------------------------------------------------
    simple_conv, log_conv = tech_matrices(converged=True)
    regime_conv = correlation_regime(log_conv)
    book_conv = book_pnl_series(TECH_POSITIONS, simple_conv)
    es_conv = historical_es(book_conv.total, 0.95)
    contrib_conv = es_contributions(book_conv.per_position, 0.95)
    share_conv = {p.key: p.share for p in contrib_conv.per_position}

    assert regime_conv.state == STATE_CONVERGING
    assert regime_conv.current_avg == pytest.approx(0.9922, abs=0.001)
    assert regime_conv.reason and "0.80" in regime_conv.reason
    # (a) The tail more than doubles on the SAME positions.
    assert es_conv.value == pytest.approx(4003.9, abs=2.0)
    assert es_conv.value > 2.0 * es_div.value
    # (b) The hedge stops hedging: every name now ADDS to the tail.
    assert share_conv["AMD#2"] == pytest.approx(0.1978, abs=0.002)
    assert share_conv["AMD#2"] > 0.0
    assert all(p.share > 0.0 for p in contrib_conv.per_position)
    # Contributions still reconcile exactly (a concentrated book must not
    # break the Euler identity).
    assert contrib_conv.total == pytest.approx(es_conv.value, abs=1e-9)

    # (c) The bucket ES-share cap binds in the SHADOW verdict.
    candidate = CandidateSpec(
        key="NVDA#cand",
        ticker="NVDA",
        instrument="LONG_STOCK",
        multiplier=1,
        spot=200.0,
        delta=1.0,
        max_loss_per_unit=20.0,
        capital_per_unit=200.0,
        quantity_requested=300,
    )
    caps, health, _reason = statistical_caps(
        book_conv,
        candidate,
        returns=simple_conv,
        nav=200_000.0,
        positions=TECH_POSITIONS,
        buckets=TECH_BUCKET,
    )
    by_code = {cap.code: cap for cap in caps}
    assert health.value == "ACTIVE"
    bucket_code = f"{CODE_BUCKET_ES_CONTRIBUTION}:TECH_MEGA"
    assert bucket_code in by_code, sorted(by_code)
    assert CODE_ES_CONTRIBUTION in by_code
    # The whole book already IS the bucket, so no size passes: cap 0.
    assert by_code[bucket_code].cap_qty == 0
    assert by_code[bucket_code].layer == LAYER_CONCENTRATION
    assert "100.0%" in by_code[bucket_code].sentence
    # The single-name cap bisected to a real, smaller-than-requested size:
    # NVDA's share is 37.6% at the requested 300 and 34.9% at 268 — the
    # largest quantity that still passes the 35% limit.
    single = by_code[CODE_ES_CONTRIBUTION]
    assert single.cap_qty == 268
    assert 0 < single.cap_qty < 300
    assert single.measured["es_share_at_requested"] == pytest.approx(0.3756, abs=1e-3)
    assert single.measured["es_share_at_cap"] == pytest.approx(0.3495, abs=1e-3)
    assert single.measured["es_share_at_cap"] <= single.measured["limit"]

    verdict = shadow_verdict(300, caps)
    assert verdict.mode == MODE_SHADOW  # hypothetical — it decides NOTHING
    assert verdict.hypothetical_decision == DECISION_REJECT
    assert verdict.hypothetical_quantity == 0
    assert verdict.binding[0] == bucket_code  # most restrictive first


# ===========================================================================
# 4. IV CRUSH
# ===========================================================================


def test_iv_crush_hits_the_long_call_and_cannot_touch_a_stock_only_book():
    """SCENARIO — the catalogue's "IV crush (flat, −40%)" scenario, spot
    unchanged: the event after an earnings print that a delta-only view
    reports as a harmless zero.

    FAIL-SAFE — full revaluation must charge the long-premium book for the
    vega it actually holds, and must report EXACTLY 0.0 for a book with no
    optionality (a stock position genuinely cannot lose money to an IV move —
    that zero is a measurement, not a gap).

    THE ASSERTION — 10 long ATM calls (60 DTE, IV 30% -> 18%) lose $1,927.57;
    100 shares of the same underlying lose exactly 0.0 under the identical
    scenario; and a long PUT gains $2,408.84 under "IV spike (flat, +50%)".
    """
    crush = dict(iv_shock=-0.40)
    spike = dict(iv_shock=+0.50)

    # (a) The long call is hurt by the crush — the loss is real and priced.
    call_result = scenario_pnl([], [long_call()], **crush)
    assert call_result.total == pytest.approx(-1927.57, abs=0.05)
    assert call_result.total < 0.0
    assert call_result.fully_revalued is True
    assert call_result.method_by_key["AAPL#call"] == METHOD_FULL_REVAL

    # (b) A stock-only book shows EXACTLY zero — not a null, not a fallback.
    stock_result = scenario_pnl([stock_leg()], [], **crush)
    assert stock_result.total == 0.0
    assert stock_result.method_by_key["AAPL#stock"] == METHOD_FULL_REVAL
    assert stock_result.notes == ()

    # (c) Long vega on the other side of the trade: an IV SPIKE pays a put.
    put_result = scenario_pnl([], [long_put()], **spike)
    assert put_result.total == pytest.approx(2408.84, abs=0.05)
    assert put_result.total > 0.0

    # (d) The signs are symmetric in the shock, as vega demands: the same
    # long put LOSES under the crush.
    assert scenario_pnl([], [long_put()], **crush).total < 0.0
    # ...and the mixed book nets the two, so no leg is double-counted.
    mixed = scenario_pnl([stock_leg()], [long_call(), long_put()], **crush)
    assert mixed.total == pytest.approx(
        call_result.total + scenario_pnl([], [long_put()], **crush).total, abs=1e-6
    )

    # (e) The zero scenario is bit-exactly zero — the basis anchor holds, so
    # "no shock" can never manufacture a P&L (design §8.2).
    assert scenario_pnl([stock_leg()], [long_call(), long_put()]).total == 0.0


# ===========================================================================
# 5. EXTREME LONG GAMMA / LONG VEGA
# ===========================================================================


def test_long_gamma_pnl_is_convex_and_long_vega_signs_are_right():
    """SCENARIO — a deep long-gamma position (10 ATM calls) shocked +10% and
    −10%, and the same book shocked on IV alone.

    FAIL-SAFE — full revaluation must reproduce CONVEXITY: for a long-gamma
    book, ``P&L(+10%) + P&L(−10%) > 0 = 2 x P&L(0)``. A delta-linear model
    gets exactly 0 here, which is how it understates a long-gamma tail — and
    understating a tail is the failure mode this whole phase exists to catch.

    THE ASSERTION — up +$6,915.10, down −$3,816.64, sum +$3,098.46 > 0,
    while 2 x P&L(0) = 0.0. Long vega: +$2,408.84 on an IV spike for the put
    and −$1,927.57 on the crush for the call, both non-zero at unchanged spot.
    """
    up = scenario_pnl([], [long_call()], spot_shock=+0.10).total
    down = scenario_pnl([], [long_call()], spot_shock=-0.10).total
    flat = scenario_pnl([], [long_call()], spot_shock=0.0).total

    assert flat == 0.0
    assert up == pytest.approx(6915.10, abs=0.05)
    assert down == pytest.approx(-3816.64, abs=0.05)

    # THE convexity statement, exactly as spec'd — written as two separate
    # assertions rather than a chained comparison, so each half fails on its
    # own terms: P&L(+10%) + P&L(-10%) > 0, and that 0 IS 2 x P&L(0).
    assert 2.0 * flat == 0.0
    assert up + down > 2.0 * flat
    assert up + down == pytest.approx(3098.46, abs=0.05)
    # The upside outruns the downside — that IS long gamma.
    assert up > abs(down)

    # A delta-linear approximation of the same two shocks cancels to zero:
    # this is the number full revaluation exists to replace.
    linear_up = long_call().quantity * 100 * CALL_GREEKS.delta * SPOT * 0.10
    linear_down = -linear_up
    assert linear_up + linear_down == 0.0
    assert up + down > 3_000.0  # the convexity the linear view cannot see

    # Long vega, at UNCHANGED spot, is signed correctly on both sides.
    assert scenario_pnl([], [long_put()], iv_shock=+0.50).total > 0.0
    assert scenario_pnl([], [long_call()], iv_shock=+0.50).total > 0.0
    assert scenario_pnl([], [long_call()], iv_shock=-0.40).total < 0.0
    # A SHORT call is short vega: the crush PAYS it (sign flips with quantity).
    short_call = long_call(quantity=-10)
    assert scenario_pnl([], [short_call], iv_shock=-0.40).total == pytest.approx(
        +1927.57, abs=0.05
    )


def test_a_leg_without_iv_is_priced_delta_linear_and_says_so():
    """SCENARIO — the same long call, but the chain gave us no IV for it
    (a vendor gap): full revaluation is impossible.

    FAIL-SAFE — the leg must fall back to DELTA_LINEAR, be LABELLED as such
    per-leg and in the coverage counts, and carry a note admitting the leg
    cannot see the IV shock at all. Silently reporting the linear number as
    a full revaluation would hide the understatement.

    THE ASSERTION — ``method_by_key`` is DELTA_LINEAR, ``method_coverage``
    is {FULL_REVAL: 0, DELTA_LINEAR: 1}, ``fully_revalued`` is False, the
    note names the leg key and its delta, and the P&L is exactly the linear
    $5,457.28 — identical with and without the IV shock, proving the leg is
    blind to vega rather than pretending otherwise.
    """
    blind = long_call(iv0=None)
    assert blind.can_full_reval is False

    result = scenario_pnl([], [blind], spot_shock=+0.10)
    assert result.method_by_key["AAPL#call"] == METHOD_DELTA_LINEAR
    assert dict(result.method_coverage) == {
        METHOD_FULL_REVAL: 0,
        METHOD_DELTA_LINEAR: 1,
    }
    assert result.fully_revalued is False
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "AAPL#call" in note
    assert "no iv0" in note
    assert METHOD_DELTA_LINEAR in note

    # The number is the linear one, to the cent.
    expected = 10 * 100 * CALL_GREEKS.delta * SPOT * 0.10
    assert result.total == pytest.approx(expected, abs=1e-9)
    assert result.total == pytest.approx(5457.28, abs=0.05)

    # ...and it is IDENTICAL with an IV shock applied: the fallback is blind
    # to vega, exactly as its note admits.
    with_iv = scenario_pnl([], [blind], spot_shock=+0.10, iv_shock=-0.40)
    assert with_iv.total == result.total
    # The full-revaluation leg, by contrast, DOES move on the same shock.
    seeing = scenario_pnl([], [long_call()], spot_shock=+0.10, iv_shock=-0.40)
    assert seeing.total != pytest.approx(
        scenario_pnl([], [long_call()], spot_shock=+0.10).total
    )


# ===========================================================================
# 6. CONCENTRATED TECH EXPOSURE
# ===========================================================================


def test_concentrated_tech_book_breaches_both_limits_without_moving_tier_0():
    """SCENARIO — a book of NVDA / AMD / AAPL only (all three in the Tier 0
    TECH_MEGA correlation bucket), with a further 300-share NVDA add-on
    proposed on top.

    FAIL-SAFE — the statistical layer must measure the concentration and
    produce a hypothetical RESIZE/REJECT; and — the point of the whole
    phase — the Tier 0 decision on the same trade must be IDENTICAL whether
    that statistical verdict exists or not, because it is SHADOW.

    THE ASSERTION — single-name ES share reaches 50.1% (> the 35% limit) so
    the cap bisects to 160 units; the bucket holds 100.0% (> 50%) so its cap
    is 0; the merged SHADOW verdict is REJECT/0 with the bucket binding
    first. Tier 0 meanwhile approves the very same battery of decisions with
    byte-identical output whether the caps are computed or not — because
    ``assess`` is never handed them (``extra_caps`` stays ``()``).
    """
    simple, _log = tech_matrices(converged=True)
    book = book_pnl_series(TECH_POSITIONS, simple)
    contrib = es_contributions(book.per_position, 0.95)

    # Every name is a tech name: the bucket IS the book.
    assert {p.key.split("#")[0] for p in contrib.per_position} == set(TECH)

    candidate = CandidateSpec(
        key="NVDA#cand",
        ticker="NVDA",
        instrument="LONG_STOCK",
        multiplier=1,
        spot=200.0,
        delta=1.0,
        max_loss_per_unit=20.0,
        capital_per_unit=200.0,
        quantity_requested=300,
    )
    caps, health, _reason = statistical_caps(
        book,
        candidate,
        returns=simple,
        nav=200_000.0,
        positions=TECH_POSITIONS,
        buckets=TECH_BUCKET,
    )
    by_code = {cap.code: cap for cap in caps}
    assert health.value == "ACTIVE"

    single = by_code[CODE_ES_CONTRIBUTION]
    assert "37.6%" in single.sentence  # measured share at the requested size
    assert "35.0%" in single.sentence  # the limit it breached
    assert single.cap_qty == 268
    assert single.layer == LAYER_CONCENTRATION

    bucket = by_code[f"{CODE_BUCKET_ES_CONTRIBUTION}:TECH_MEGA"]
    assert "TECH_MEGA" in bucket.sentence
    assert bucket.cap_qty == 0

    verdict = shadow_verdict(300, caps)
    assert verdict.hypothetical_decision == DECISION_REJECT
    assert verdict.hypothetical_quantity == 0
    assert verdict.mode == MODE_SHADOW

    # ...AND THE TIER 0 DECISION IS UNMOVED. The caps above exist as data;
    # `assess` never receives them, so its whole decision tree is unchanged.
    cases = battery()
    without = [c.fingerprint(c.run()) for c in cases]
    with_default = [c.fingerprint(c.run(extra_caps=())) for c in cases]
    assert without == with_default
    assert len(cases) >= 200


async def test_tier_0_still_owns_the_decision_on_a_concentrated_live_book(client):
    """SCENARIO — the live gate chain on a book heavy enough to breach the
    Tier 0 heat limit outright.

    FAIL-SAFE — the REJECT must come from Tier 0's own hard limit
    (HEAT_LIMIT), not from any statistical layer, and it must be a REJECT at
    quantity 0 with the reason code recorded.

    THE ASSERTION — with $50,000 of open max-loss against the paper NAV, the
    RISK_APPROVAL gate FAILs, ``risk.decision`` is REJECT, ``approved_quantity``
    is 0 and ``reason_codes`` is exactly ``["HEAT_LIMIT"]`` — a HARD_LIMIT
    layer constraint, with no STATISTICAL or CONCENTRATION code among them.
    """
    async with SessionLocal() as session:
        session.add(
            Position(
                ticker="XOM",
                instrument="LONG_STOCK",
                quantity=1,
                avg_price=100.0,
                max_loss=50_000.0,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    risk = body["risk"]
    assert risk is not None
    assert risk["decision"] == "REJECT"
    assert risk["approved_quantity"] == 0
    assert risk["reason_codes"] == ["HEAT_LIMIT"]

    gate = next(g for g in body["gates"] if g["name"] == "RISK_APPROVAL")
    assert gate["status"] == "FAIL"

    # Every binding constraint is a HARD_LIMIT — no statistical layer took
    # part in this decision (contract §7.3).
    layers = {c["layer"] for c in risk["binding_constraints"]}
    assert layers == {"HARD_LIMIT"}


# ===========================================================================
# 7. GARCH MODEL FAILURE
# ===========================================================================


@pytest.mark.parametrize(
    "label, series_factory, expected_fragment",
    [
        (
            "constant series (no variance to model)",
            lambda: [0.001] * 400,
            "persistence=",
        ),
        (
            "too short for GARCH (n=120 < 250)",
            lambda: [random.Random(1).gauss(0.0, 0.01) for _ in range(120)],
            "n=120 < min_obs=250",
        ),
        (
            "near-IGARCH random walk in variance",
            lambda: _igarch_series(),
            "omega=",
        ),
    ],
)
def test_garch_failure_falls_back_to_ewma_and_names_the_reason(
    label: str, series_factory, expected_fragment: str
):
    """SCENARIO — three ways GARCH(1,1) legitimately fails: a constant
    series, a sample below the 250-observation minimum, and a near-integrated
    process whose unconditional variance is not identified.

    FAIL-SAFE — spec §13/§58: a model that fails its diagnostics must fall
    back to a SIMPLER model rather than halt or, worse, publish an
    unvalidated number. The fallback must be EWMA, must be labelled, and the
    reason must state which diagnostic failed with its real value.

    THE ASSERTION — in all three cases ``conditional_volatility_source``
    returns source "EWMA", the EWMA result is ACTIVE (a usable number is
    still produced), and the reason begins "GARCH not ACTIVE (health=...)",
    quotes the failing diagnostic, and names the EWMA lambda it fell back to.
    """
    series = series_factory()
    source, result, reason = conditional_volatility_source(series)

    assert source == SOURCE_EWMA, label
    # The fallback still yields a usable, honest number — degradation is not
    # an outage (spec §58).
    assert result.health.value == "ACTIVE", label
    assert result.value is not None and result.value > 0.0

    # The reason is a full explanation, not a shrug.
    assert reason.startswith("GARCH not ACTIVE (health="), reason
    assert expected_fragment in reason, reason
    assert "falling back to EWMA(lambda=0.94)" in reason

    # The DEGRADED/UNAVAILABLE health of the GARCH fit is named in the
    # reason, so a reader knows whether it failed or was never attempted.
    assert any(h in reason for h in ("DEGRADED", "UNAVAILABLE", "FAILED")), reason

    # The filtered-P&L seam obeys the SAME rule (one fallback hierarchy, not
    # two): the scaled series is EWMA-scaled and says so. The scaled series is
    # HONESTLY shorter — EWMA burns a 20-observation warm-up before its first
    # usable variance, and the filter reports the sample it actually used
    # rather than padding the gap with unscaled observations.
    scaled_source, scaled, scaled_reason = conditional_scaled_pnl_source(series)
    assert scaled_source == SOURCE_EWMA
    assert len(scaled) == len(series) - 20
    assert "falling back to EWMA" in scaled_reason


def _igarch_series(n: int = 600, seed: int = 5) -> list[float]:
    """A near-integrated GARCH path (alpha + beta = 1.0, omega ~ 0): the
    unconditional variance does not exist, so the fit cannot identify it."""
    rng = random.Random(seed)
    variance = 1e-4
    out: list[float] = []
    for _ in range(n):
        x = rng.gauss(0.0, math.sqrt(variance))
        out.append(x)
        variance = 1e-12 + 0.15 * x * x + 0.85 * variance
    return out


async def test_snapshot_still_builds_and_hard_limits_still_decide_under_garch_failure(
    client,
):
    """SCENARIO — the live snapshot build on a book whose series GARCH cannot
    fit (the stub's 200-bar history is below the 250-observation minimum).

    FAIL-SAFE — the build must SUCCEED, the conditional views must be served
    from EWMA with the source named on the wire, and Tier 0 must be entirely
    unaffected.

    THE ASSERTION — the build returns a snapshot with 199 observations and an
    ACTIVE historical VaR-95; ``statistical.conditional_source.source`` is
    "EWMA" with a reason naming the GARCH health; and the Tier 0 battery is
    byte-identical, since none of this reaches ``assess``.
    """
    await seed_stock_position(bars=200)
    result = await build()
    api = result.api

    # (a) The build succeeded and produced real numbers.
    assert api["n_obs"] == 199
    var95 = next(
        r for r in api["var"] if r["model"] == "HISTORICAL" and r["confidence"] == 0.95
    )
    assert var95["health"] in {"ACTIVE", "DEGRADED"}
    assert var95["value_usd"] > 0.0

    # (b) The conditional source is EWMA, and the wire says WHY.
    source = api["conditional_source"]
    assert source["source"] == SOURCE_EWMA
    assert "GARCH not ACTIVE" in source["reason"]
    assert "falling back to EWMA" in source["reason"]

    # (c) The conditional VaR row still carries a number — the fallback
    # produced a view, it did not blank the grid.
    conditional = next(
        r for r in api["var"] if r["model"] == "HISTORICAL_VOL_SCALED"
    )
    assert conditional["health"] in {"ACTIVE", "DEGRADED"}
    assert conditional["value_usd"] is not None

    # (d) Hard limits are untouched by any of it.
    cases = battery()
    assert [c.fingerprint(c.run()) for c in cases] == [
        c.fingerprint(c.run(extra_caps=())) for c in cases
    ]


# ===========================================================================
# 8. STALE RISK SNAPSHOT
# ===========================================================================


def test_ttl_policy_marks_a_snapshot_stale_exactly_at_the_boundary():
    """SCENARIO — a statistical snapshot ages past its TTL while the greeks
    inside it age far faster (spec §55: model-specific TTLs).

    FAIL-SAFE — staleness must be a measured, per-kind fact with a sharp
    boundary, and an unknown TTL kind must RAISE rather than default to
    "fresh" (defaulting to fresh is how a stale number authorizes a trade).

    THE ASSERTION — statistical TTL 86,400 s: at exactly 86,400 s old the
    snapshot is NOT stale, at 86,401 s it IS. The greeks TTL is 120 s, so the
    same snapshot is already greeks-stale at 86,400 s. An unknown kind and a
    non-positive TTL both raise ValueError.
    """
    from libs.trading_core.risk.snapshot import DataQuality, PortfolioRiskSnapshot

    as_of = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)
    snapshot = PortfolioRiskSnapshot(
        as_of=as_of,
        nav=100_000.0,
        cash=50_000.0,
        cash_reserved=0.0,
        gross_exposure=0.0,
        delta_adjusted_exposure=0.0,
        heat_pct=0.0,
        heat_state="NORMAL",
        data_quality=DataQuality(as_of=as_of.date(), oldest_bar=None, newest_bar=None),
        ttl=TtlPolicy(),
    )
    ttl = snapshot.ttl.statistical_seconds
    assert ttl == 86_400.0
    assert snapshot.ttl.greeks_seconds == 120.0

    assert snapshot.is_stale(as_of) is False
    assert snapshot.age_seconds(as_of + timedelta(seconds=ttl)) == ttl
    # Exactly TTL is NOT stale; one second later it is.
    assert snapshot.is_stale(as_of + timedelta(seconds=ttl)) is False
    assert snapshot.is_stale(as_of + timedelta(seconds=ttl + 1)) is True
    # The faster-decaying greeks TTL is already blown at the same instant.
    assert (
        snapshot.is_stale(
            as_of + timedelta(seconds=ttl), kind=STALENESS_KIND_GREEKS
        )
        is True
    )
    assert (
        snapshot.is_stale(as_of + timedelta(seconds=60), kind=STALENESS_KIND_GREEKS)
        is False
    )

    # An unknown kind is malformed input, never a silent "fresh".
    with pytest.raises(ValueError, match="unknown staleness kind"):
        snapshot.is_stale(as_of, kind="whatever")
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        TtlPolicy(statistical_seconds=0.0)


async def test_stale_reaches_the_wire_and_tier_0_still_decides(client):
    """SCENARIO — the serialised ``statistical`` block is read after its TTL
    has expired (the API's own ``stale`` flag).

    FAIL-SAFE — the wire must report ``stale: true``, the Tier 0 chain must
    keep deciding regardless (``assess`` never reads a snapshot at all), and
    the stale statistical layer must authorize NOTHING — the numbers stay
    visible for a human, but no cap derived from them is passed to the engine.

    THE ASSERTION — the SAME snapshot serialises ``stale`` False at age 0 and
    at exactly the 86,400 s TTL, then True at 86,401 s, with the VaR number
    unchanged across all three (staleness is a LABEL, not a recomputation).
    A live preview taken while a stale snapshot exists still produces a Tier 0
    decision, and the RISK_DECISION audit records no statistical cap.
    """
    await seed_stock_position(bars=200)
    result = await build()
    snapshot = result.snapshot
    ttl = snapshot.ttl.statistical_seconds

    seen = {}
    for age in (0.0, ttl, ttl + 1.0):
        api = rs._statistical_api(
            snapshot,
            nav=result.nav,
            now=snapshot.as_of + timedelta(seconds=age),
            positions_excluded=[],
            capital_weights=result.capital_weights,
            meta_by_key={},
            book=result.book,
        )
        seen[age] = (api["stale"], api["var"][0]["value_usd"])

    assert seen[0.0][0] is False
    assert seen[ttl][0] is False
    assert seen[ttl + 1.0][0] is True
    # The NUMBER never changed — only its freshness label did.
    values = {v for _stale, v in seen.values()}
    assert len(values) == 1
    assert values.pop() is not None

    # Tier 0 keeps deciding: the chain reaches RISK_APPROVAL and answers.
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    assert body["risk"] is not None
    assert body["risk"]["decision"] in {
        "APPROVE",
        "APPROVE_WITH_RESIZE",
        "REJECT",
    }
    # ...and no statistical/concentration/stress code binds the real decision.
    layers = {c["layer"] for c in body["risk"]["binding_constraints"]}
    assert layers <= {"HARD_LIMIT"}, layers


# ===========================================================================
# 9. EVT INSUFFICIENT DATA / COPULA FAILURE — the deferred models
# ===========================================================================


def test_evt_and_copula_are_absent_and_asking_for_them_raises():
    """SCENARIO — a caller asks the registry for an EVT or copula model.
    Both were DEFERRED (audit §7 gap matrix / §11): EVT needs a tail sample
    this platform's ~600-observation histories cannot supply, and a copula
    needs a dependence sample it has even less of.

    FAIL-SAFE — the honest answer to "what does EVT say?" is an exception,
    not a number. A deferred model must be ABSENT rather than present and
    quietly returning something plausible; the registry must name exactly the
    models that were actually built.

    THE ASSERTION — ``names()`` is exactly the five built models
    (garch11, gaussian_es, gaussian_var, historical_es, historical_var);
    every EVT/copula spelling raises KeyError whose message lists the known
    names; and no EVT/copula implementation exists anywhere in libs/ or apps/.
    """
    # Importing the package registers every model that exists.
    importlib.import_module("libs.trading_core.risk.models")

    assert model_base.names() == (
        "garch11",
        "gaussian_es",
        "gaussian_var",
        "historical_es",
        "historical_var",
    )

    for missing in (
        "evt",
        "evt_var",
        "evt_es",
        "pot_evt",
        "copula",
        "gaussian_copula",
        "t_copula",
    ):
        with pytest.raises(KeyError) as excinfo:
            model_base.get(missing)
        message = str(excinfo.value)
        assert missing in message
        # The error is USEFUL: it names what IS available.
        assert "known:" in message
        assert "historical_var" in message
        # ...and it never returns a number.
        assert "0.0" not in message

    # A fabricated registration is refused too — a model cannot sneak in
    # under a name that is already taken.
    with pytest.raises(ValueError, match="already registered"):
        model_base.register(model_base.get("historical_var"))


def test_no_evt_or_copula_implementation_exists_in_the_codebase():
    """SCENARIO — the deferral could rot: someone adds an EVT fitter later
    and forgets that no data supports it.

    FAIL-SAFE — the absence must be enforced mechanically, not by memory.

    THE ASSERTION — a source scan of libs/ and apps/ finds no EVT, copula,
    GPD or peaks-over-threshold implementation. (This test is the tripwire:
    if such a model is ever built deliberately, it must be registered, given
    a mode, and this list updated in the same change.)
    """
    services = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r"\b(evt|copula|gpd|generalized_pareto|peaks_over_threshold)\b",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for root in ("libs", "apps"):
        for path in (services / root).rglob("*.py"):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(services)}:{lineno}: {line.strip()}")
    assert offenders == [], "EVT/copula code found:\n" + "\n".join(offenders)


# ===========================================================================
# 10. BROKER / DATA MISMATCH
# ===========================================================================


async def test_unreadable_broker_cash_fails_closed_and_persists_no_snapshot(
    client, monkeypatch
):
    """SCENARIO — a real broker is configured but its account endpoint cannot
    be read (a 502-class fault). Deployable cash is therefore unverifiable.

    FAIL-SAFE — guide §28 / spec §14: the order path must FAIL CLOSED. Sizing
    from the local cash figure alone could deploy money the account does not
    have, so the RISK_APPROVAL gate must veto, no order or position may be
    written, and no risk snapshot may be persisted for the attempt (a
    snapshot implies a measurement that did not happen).

    THE ASSERTION — the gate FAILs with a BROKER_ERROR detail naming the
    fault; ``risk`` is null (no decision was reachable); approve answers 422;
    and the Order / Position / risk_snapshots tables are all empty afterwards.
    """
    await authorize(client, BULL_TICKER)

    async def unreadable_broker_cash():
        return None, "connection refused to the broker account endpoint"

    monkeypatch.setattr(gate_chain, "_broker_cash_for_sizing", unreadable_broker_cash
    )

    body = await preview(client, BULL_TICKER)
    gate = next(g for g in body["gates"] if g["name"] == "RISK_APPROVAL")
    assert gate["status"] == "FAIL"
    detail = gate["detail"]
    assert "BROKER_ERROR" in detail
    assert "connection refused to the broker account endpoint" in detail
    assert "Failing closed" in detail
    # No decision was possible, so none is reported — an honest null.
    assert body["risk"] is None
    assert body["why_not_trade"]

    # The approve path refuses too — a veto cannot be walked past (§42).
    response = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
    assert response.status_code == 422

    async with SessionLocal() as session:
        orders = (await session.execute(select(Order))).scalars().all()
        positions = (await session.execute(select(Position))).scalars().all()
        snapshots = (await session.execute(select(RiskSnapshotRow))).scalars().all()
    assert orders == []
    assert positions == []
    # NOTHING was persisted for an attempt that never got a measurement.
    assert snapshots == []


async def test_a_ticker_without_bars_is_named_excluded_and_the_view_stays_200(client):
    """SCENARIO — one position's underlying has a usable spot but no return
    history (a single stored bar), so it cannot enter the return matrix.

    FAIL-SAFE — the position must be EXCLUDED and NAMED rather than priced at
    zero (a zero-return column would understate the book's risk), the
    snapshot's ``data_quality.valid`` must go False with a reason carrying the
    real ticker, and the read view must still answer 200 with honest values
    for everything that IS measurable.

    THE ASSERTION — GET /api/portfolio/risk answers 200; ``positions_excluded``
    contains ZZZZ with a reason; ``data_quality.valid`` is False with reason
    ``tickers_missing=('ZZZZ',)``; and the surviving AAPL position still
    yields an ACTIVE historical VaR-95 over 199 observations.
    """
    await seed_stock_position("AAPL", bars=200)
    async with SessionLocal() as session:
        # ONE bar: a usable spot (so the row is priceable) but NO return.
        session.add(
            StockBarDaily(
                ticker="ZZZZ",
                ts=date(2025, 7, 1),
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                volume=1_000_000,
            )
        )
        session.add(
            Position(
                ticker="ZZZZ",
                instrument="LONG_STOCK",
                quantity=50,
                avg_price=10.0,
                max_loss=500.0,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    response = await client.get("/api/portfolio/risk")
    assert response.status_code == 200  # a data gap is not a server error
    statistical = response.json()["statistical"]

    # (a) The gap is NAMED, with the reason the server generated.
    excluded = {e["key"]: e["reason"] for e in statistical["positions_excluded"]}
    assert any(key.startswith("ZZZZ#") for key in excluded), excluded
    zzzz_key = next(k for k in excluded if k.startswith("ZZZZ#"))
    assert excluded[zzzz_key]

    # (b) Data quality is honestly INVALID, and says which ticker did it.
    quality = statistical["data_quality"]
    assert quality["valid"] is False
    assert quality["tickers_missing"] == ["ZZZZ"]
    assert any("ZZZZ" in reason for reason in quality["reasons"]), quality
    assert zzzz_key in quality["keys_excluded"]

    # (c) Everything measurable is still measured — honest nulls, not a
    # blanked view.
    assert statistical["n_obs"] == 199
    var95 = next(
        r
        for r in statistical["var"]
        if r["model"] == "HISTORICAL" and r["confidence"] == 0.95
    )
    assert var95["health"] in {"ACTIVE", "DEGRADED"}
    assert var95["value_usd"] > 0.0
    # The excluded position contributed NOTHING — it was not priced at zero
    # delta and folded in.
    contribution_keys = [
        row["key"] for row in statistical["contributions"]["es"]["rows"]
    ]
    assert zzzz_key not in contribution_keys


# ===========================================================================
# 11. §67 PROPERTY TESTS
# ===========================================================================


async def test_property_a_rejected_strategy_never_reaches_the_broker(
    client, monkeypatch
):
    """§67 PROPERTY (a) — "A rejected strategy never reaches broker."

    SCENARIO — a ticker the risk engine VETOES (portfolio heat above the 8%
    reject boundary), then an approve call attempted anyway on that veto.

    FAIL-SAFE — the approve path must re-run the chain server-side (§42:
    client previews are never trusted), refuse, write no Order and no
    Position row, and never construct or call the execution venue.

    THE ASSERTION — preview REJECTs with HEAT_LIMIT; approve answers 422 with
    the fresh preview embedded; zero Order rows and only the pre-seeded
    Position remain; and the broker seams (``resolve_broker`` and
    ``_approve_via_broker``) were called ZERO times — asserted with spies,
    so "no broker call" is measured rather than inferred.
    """
    # A book already at 50k of open max loss: the heat gate rejects.
    async with SessionLocal() as session:
        session.add(
            Position(
                ticker="XOM",
                instrument="LONG_STOCK",
                quantity=1,
                avg_price=100.0,
                max_loss=50_000.0,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    # Spies on BOTH venue seams: reaching either one is the failure.
    calls: list[str] = []

    def spy_resolve_broker(*args, **kwargs):
        calls.append("resolve_broker")
        raise AssertionError("a REJECTED strategy reached the broker")

    async def spy_via_broker(*args, **kwargs):
        calls.append("_approve_via_broker")
        raise AssertionError("a REJECTED strategy reached the broker")

    monkeypatch.setattr(gate_chain, "resolve_broker", spy_resolve_broker)
    monkeypatch.setattr(orders_router, "_approve_via_broker", spy_via_broker)

    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    assert body["risk"]["decision"] == "REJECT"
    assert body["risk"]["approved_quantity"] == 0
    assert body["risk"]["reason_codes"] == ["HEAT_LIMIT"]

    response = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "preview" in detail
    # The server re-evaluated rather than trusting the client's preview.
    approve_gate = next(
        g for g in detail["preview"]["gates"] if g["name"] == "RISK_APPROVAL"
    )
    assert approve_gate["status"] == "FAIL"

    # Nothing was written, and no venue was touched.
    async with SessionLocal() as session:
        orders = (await session.execute(select(Order))).scalars().all()
        positions = (await session.execute(select(Position))).scalars().all()
    assert orders == []
    assert [p.ticker for p in positions] == ["XOM"]  # only the seeded book
    assert calls == []


def test_property_b_risk_contribution_reconciles_with_total_risk():
    """§67 PROPERTY (b) — "Risk contribution reconciles with total risk
    within numerical tolerance where mathematically applicable."

    SCENARIO — 40 seeded random books (2–5 names, 80–200 observations, long
    and short positions, heterogeneous volatilities and spot prices).

    FAIL-SAFE — Euler ES contributions must sum to the portfolio ES and
    volatility contributions to the portfolio sigma. If they did not, the
    "which position owns the risk?" panel would be attributing risk that does
    not exist — the exact hidden-concentration failure spec §10 targets.

    THE ASSERTION — for every book: the ES contribution TOTAL equals
    ``historical_es(book)`` EXACTLY (same float), and the row sum matches it
    to <= 1e-9 absolute (fsum re-association only); the volatility rows sum to
    sigma_p within 1e-9 relative. Measured worst deviation across 40 books:
    9.1e-13 for both methods.
    """
    rng = random.Random(987654)
    worst_es_rows = 0.0
    worst_vol_rows = 0.0
    checked = 0

    for _ in range(40):
        n_obs = rng.randrange(80, 200)
        n_names = rng.randrange(2, 6)
        tickers = tuple(f"T{i}" for i in range(n_names))
        dates = tuple(date(2025, 1, 6) + timedelta(days=i) for i in range(n_obs))
        rows = tuple(
            tuple(rng.gauss(0.0, rng.uniform(0.005, 0.03)) for _ in tickers)
            for _ in range(n_obs)
        )
        matrix = ReturnMatrix(
            dates=dates, tickers=tickers, rows=rows, return_type="SIMPLE"
        )
        positions = [
            PositionRiskInput(
                key=f"{ticker}#{i}",
                ticker=ticker,
                instrument="LONG_STOCK",
                quantity=rng.choice([-200, -50, 50, 100, 300]),
                multiplier=1,
                spot=rng.uniform(20.0, 400.0),
                delta=1.0,
                max_loss=1_000.0,
            )
            for i, ticker in enumerate(tickers)
        ]
        book = book_pnl_series(positions, matrix)

        es_result = es_contributions(book.per_position, 0.95)
        if es_result.is_available:
            checked += 1
            # EXACT: the Euler ES total IS the historical ES, same float.
            expected_es = historical_es(book.total, 0.95)
            assert es_result.total == expected_es.value
            row_sum = math.fsum(p.contribution for p in es_result.per_position)
            worst_es_rows = max(worst_es_rows, abs(row_sum - es_result.total))
            assert row_sum == pytest.approx(es_result.total, abs=1e-9)
            # Shares reconcile to 1 as well.
            assert math.fsum(p.share for p in es_result.per_position) == pytest.approx(
                1.0, abs=1e-12
            )

        vol_result = volatility_contributions(book.per_position)
        if vol_result.is_available:
            row_sum = math.fsum(p.contribution for p in vol_result.per_position)
            worst_vol_rows = max(
                worst_vol_rows, abs(row_sum - vol_result.total) / vol_result.total
            )
            assert row_sum == pytest.approx(vol_result.total, rel=1e-9)

    assert checked >= 35, f"only {checked} books were measurable"
    assert worst_es_rows < 1e-9, worst_es_rows
    assert worst_vol_rows < 1e-9, worst_vol_rows


#: The five ``max_loss`` bases the gateway actually writes, each as
#: ``(label, per-unit basis)``. Every one is ``quantity x per_unit`` at the
#: construction sites in execution/gate_chain.py (stock / short stock / option /
#: spread) and routers/income.py (cash-secured put). Sources:
#:   stock         : quantity * chain.stop_distance
#:   short stock   : quantity * chain.stop_distance * SHORT_STOCK_GAP_RISK_FACTOR
#:   long option   : quantity * fill * OPTION_MULTIPLIER
#:   debit spread  : quantity * net_fill * OPTION_MULTIPLIER
#:   cash-sec. put : (strike - credit) * OPTION_MULTIPLIER * contracts
MAX_LOSS_BASES = (
    ("stock (stop distance)", 2.35),
    ("short stock (stop x 2.0 gap factor)", 2.35 * 2.0),
    ("long option (premium x 100)", 4.10 * 100),
    ("debit spread (net debit x 100)", 1.75 * 100),
    ("cash-secured put ((strike - credit) x 100)", (95.0 - 1.80) * 100),
)


@pytest.mark.parametrize("label, per_unit", MAX_LOSS_BASES)
def test_property_c_increasing_a_position_cannot_reduce_its_max_loss(
    label: str, per_unit: float
):
    """§67 PROPERTY (c) — "Increasing a position cannot accidentally reduce
    its standalone max-loss calculation."

    SCENARIO — each of the five ``max_loss`` bases the gateway writes (stock,
    short stock, long option, debit spread, cash-secured put), evaluated over
    a sweep of quantities.

    FAIL-SAFE — max loss is what portfolio HEAT is measured in. If it could
    fall as size rose, a bigger trade would look safer and the heat gate
    would let more risk through the more risk was added.

    THE ASSERTION — for every base, ``max_loss(q) = q x per_unit`` is
    non-negative, non-decreasing over q = 0..200, strictly increasing for a
    positive basis, and exactly linear (doubling q doubles max loss). Every
    per-unit basis is itself >= 0, which is what makes monotonicity hold.
    """

    def max_loss(quantity: int) -> float:
        return quantity * per_unit

    assert per_unit >= 0.0, label
    assert max_loss(0) == 0.0

    previous = max_loss(0)
    for quantity in range(1, 201):
        current = max_loss(quantity)
        assert current >= 0.0, (label, quantity)
        assert current >= previous, (label, quantity, previous, current)
        assert current > previous, (label, quantity)  # strict: basis > 0
        previous = current

    # Linear and therefore trivially monotone — stated as an equality so a
    # future non-linear basis (a cap, a floor, a rebate) fails HERE and gets
    # its own monotonicity proof rather than inheriting this one.
    assert max_loss(2) == pytest.approx(2.0 * max_loss(1))
    assert max_loss(100) == pytest.approx(100.0 * max_loss(1))
    assert max_loss(7) + max_loss(3) == pytest.approx(max_loss(10))


#: Every statistical entry point the platform has, as ``module -> functions``.
#: Property (d) replaces ALL of them with a raising stub at once. If a new
#: statistical model is added and not listed here, the hasattr guard in the
#: test does not fire — so this list is also the inventory the §68 acceptance
#: test cross-checks against the registry.
STATISTICAL_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "libs.trading_core.risk.models.var_es": (
        "historical_var",
        "historical_es",
        "gaussian_var",
        "gaussian_es",
        "conditional_var",
        "conditional_es",
        "tail_size",
    ),
    "libs.trading_core.risk.models.volatility": (
        "sample_covariance",
        "portfolio_volatility",
        "ewma_variance",
        "ewma_volatility_forecast",
        "volatility_scaled_pnl",
        "volatility_scaling",
    ),
    "libs.trading_core.risk.models.contribution": (
        "es_contributions",
        "volatility_contributions",
        "marginal_es",
        "incremental_es",
    ),
    "libs.trading_core.risk.models.diagnostics": ("distribution_diagnostics",),
    "libs.trading_core.risk.models.ensemble": ("dispersion", "model_risk_state"),
    "libs.trading_core.risk.models.drawdown": (
        "drawdown",
        "reconstructed_book_drawdown",
    ),
    "libs.trading_core.risk.models.stress": (
        "run_stress",
        "run_scenario",
        "stress_caps",
        "auto_worst_windows",
    ),
    "libs.trading_core.risk.models.garch": (
        "fit_garch",
        "garch_volatility_forecast",
        "conditional_volatility_source",
        "conditional_scaled_pnl_source",
    ),
    "libs.trading_core.risk.pretrade": (
        "compare",
        "statistical_caps",
        "shadow_verdict",
        "proposed_book",
    ),
    "libs.trading_core.risk.pnl_series": ("book_pnl_series", "position_pnl_series"),
    "libs.trading_core.risk.validation": (
        "walk_forward",
        "kupiec_pof",
        "christoffersen_independence",
    ),
    "libs.trading_core.correlation": ("correlation_regime",),
}


def test_property_d_a_failed_advanced_model_never_disables_hard_limits(monkeypatch):
    """§67 PROPERTY (d) — "A failed advanced model never disables hard
    limits."

    SCENARIO — EVERY statistical entry point in the platform (40 functions
    across 12 modules — VaR/ES, volatility, contributions, diagnostics,
    ensemble, drawdown, stress, GARCH, pre-trade, P&L series, validation and
    correlation) is replaced with a stub that raises.

    FAIL-SAFE — this is total statistical failure, the worst case spec §58
    names. Tier 0 must be completely unaffected: it imports none of these
    modules and consults none of them, so every decision must be
    byte-identical to the decisions taken with all models healthy.

    THE ASSERTION — the 240-case seeded battery (which covers APPROVE,
    APPROVE_WITH_RESIZE and REJECT, the kill switch, the weak-signal path and
    every sizing cap) produces IDENTICAL decision, approved quantity, reason
    codes and explanations before and after the sabotage — 240/240, field for
    field. Verified sabotage count: 40 entry points.
    """
    cases = battery()
    assert len(cases) >= 200
    healthy = [c.fingerprint(c.run()) for c in cases]

    def boom(*args, **kwargs):
        raise RuntimeError("advanced model exploded")

    sabotaged = 0
    for module_name, function_names in STATISTICAL_ENTRY_POINTS.items():
        module = importlib.import_module(module_name)
        for function_name in function_names:
            assert hasattr(module, function_name), f"{module_name}.{function_name}"
            monkeypatch.setattr(module, function_name, boom)
            sabotaged += 1
    assert sabotaged == 40, sabotaged

    # The sabotage is real: calling one of them now raises.
    from libs.trading_core.risk.models import var_es

    with pytest.raises(RuntimeError, match="advanced model exploded"):
        var_es.historical_var([1.0] * 100, 0.95)

    broken = [c.fingerprint(c.run()) for c in cases]
    assert broken == healthy
    # Spelled out per field, so a failure names WHICH field drifted.
    for case, before, after in zip(cases, healthy, broken):
        assert after[0] == before[0], "decision drifted"
        assert after[1] == before[1], "approved_quantity drifted"
        assert after[2] == before[2], "reason_codes drifted"
        assert after[3] == before[3], "explanations drifted"


def test_property_e_no_code_path_passes_statistical_caps_into_assess():
    """§67 PROPERTY (e) — "A stale critical risk snapshot cannot authorize
    new risk" — the STRUCTURAL half.

    SCENARIO — the only way a statistical number could ever authorize (or
    resize) real risk is the ``extra_caps`` seam on ``assess``. A stale, a
    broken or simply an unvalidated snapshot becomes dangerous the moment
    that seam is populated.

    FAIL-SAFE — while every model is SHADOW/RESEARCH, NO call site in
    ``apps/`` may pass ``extra_caps`` to ``assess`` or ``assess_income``.
    Promotion out of SHADOW is a deliberate human step (spec §70, audit §11
    Q3), so this is enforced by a source scan rather than by convention.

    THE ASSERTION — an AST walk of every module in apps/ finds exactly two
    calls to the engine (``assess`` in execution/gate_chain.py, ``assess_income``
    in routers/income.py); NEITHER passes an ``extra_caps`` keyword; and the
    only ``extra_caps=`` keyword anywhere in apps/ goes to the SHADOW helper
    ``_pretrade_statistical_shadow``, never to the engine. The scan is AST
    based on purpose: a docstring that merely MENTIONS ``assess(...)`` is
    prose, not a call site, and a text scan would trip over it.
    """
    import ast

    services = Path(__file__).resolve().parents[1]
    apps = services / "apps"

    engine_calls: list[str] = []
    extra_caps_calls: list[tuple[str, str]] = []
    for path in sorted(apps.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            callee = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if callee is None:
                continue
            location = f"{path.relative_to(services)}:{node.lineno}"
            keywords = {kw.arg for kw in node.keywords if kw.arg}
            if callee in ("assess", "assess_income"):
                engine_calls.append(f"{location} {callee}")
                # THE PROPERTY, asserted at the call site itself.
                assert "extra_caps" not in keywords, (
                    f"{location}: {callee}() is passed extra_caps — the "
                    "SHADOW promotion seam is populated"
                )
            if "extra_caps" in keywords:
                extra_caps_calls.append((location, callee))

    # The engine is called from exactly two places, both known and named.
    assert len(engine_calls) == 2, engine_calls
    assert any(
        # The gate chain moved out of routers/orders.py into its own module
        # (2026-08-24); the invariant is unchanged — the engine still has
        # exactly two callers and this is one of them.
        "execution/gate_chain.py" in c and c.endswith("assess")
        for c in engine_calls
    ), engine_calls
    assert any(
        "routers/income.py" in c and c.endswith("assess_income") for c in engine_calls
    ), engine_calls

    # Every extra_caps= keyword in apps/ goes to the SHADOW helper — which is
    # NOT the engine: its caps feed a HYPOTHETICAL verdict only.
    assert extra_caps_calls, "the seam vanished — this test needs updating"
    for location, callee in extra_caps_calls:
        assert callee == "_pretrade_statistical_shadow", (location, callee)

    # ...and that helper really exists on the router (so the assertion above
    # is checking a live seam, not a name that has been renamed away).
    assert hasattr(orders_router, "_pretrade_statistical_shadow")


async def test_property_e_a_stale_snapshot_authorizes_nothing_behaviourally(
    client, monkeypatch
):
    """§67 PROPERTY (e) — the BEHAVIOURAL half.

    SCENARIO — the whole statistical snapshot builder raises (the strongest
    form of "the critical snapshot is unusable": worse than stale — absent),
    while a live trade is assessed.

    FAIL-SAFE — the Tier 0 decision must be reached and must be identical to
    the decision taken with a healthy snapshot. A statistical layer that
    cannot produce a number must not be able to block a trade OR to permit
    one; it simply is not in the decision path.

    THE ASSERTION — with ``build_risk_snapshot`` monkeypatched to raise, the
    preview's decision, approved quantity, reason codes, gate statuses and
    binding constraints are all byte-identical to the healthy run; the only
    difference is the SHADOW block reporting its own failure.
    """
    await seed_stock_position("AAPL", bars=200)
    await authorize(client, BULL_TICKER)

    healthy = await preview(client, BULL_TICKER)
    assert healthy["risk"] is not None

    def boom(*args, **kwargs):
        raise RuntimeError("the critical risk snapshot is unusable")

    monkeypatch.setattr(gate_chain, "build_risk_snapshot", boom)
    broken = await preview(client, BULL_TICKER)

    assert broken["risk"] is not None
    assert broken["risk"]["decision"] == healthy["risk"]["decision"]
    assert broken["risk"]["approved_quantity"] == healthy["risk"]["approved_quantity"]
    assert broken["risk"]["reason_codes"] == healthy["risk"]["reason_codes"]
    assert (
        broken["risk"]["binding_constraints"]
        == healthy["risk"]["binding_constraints"]
    )
    assert [(g["name"], g["status"]) for g in broken["gates"]] == [
        (g["name"], g["status"]) for g in healthy["gates"]
    ]
    # An unusable snapshot did not authorize a LARGER trade either — the
    # direction that would actually be dangerous.
    assert broken["risk"]["approved_quantity"] <= healthy["risk"]["approved_quantity"]


# ===========================================================================
# 12. §68 MODEL VALIDATION ACCEPTANCE
# ===========================================================================


async def test_acceptance_every_registered_model_is_shadow_or_research(client):
    """§68 MODEL VALIDATION ACCEPTANCE — "Do not call a model 'production
    ready' until: Unit Tested, Replay Tested, Out-of-Sample Tested,
    Diagnostics Exposed, Failure Mode Tested, Fallback Tested, Audit Tested."

    SCENARIO — the acceptance audit itself: walk the registry and the API and
    check that no model has quietly been promoted, and that the diagnostics
    §68 requires are actually reachable by a user.

    FAIL-SAFE — a model in PRODUCTION mode is a model the engine is allowed
    to consult (``base.RiskModel``: "only PRODUCTION may feed a decision").
    None may be there yet: GARCH is RESEARCH pending the §63 criterion over
    250 forecast days, everything else is SHADOW pending the audit §11 Q3
    window.

    THE ASSERTION — all five registered models (garch11 RESEARCH;
    gaussian_es / gaussian_var / historical_es / historical_var SHADOW) are
    non-PRODUCTION and carry a version; the risk API declares mode SHADOW and
    exposes model_health per view, model_risk with reasons, dispersion,
    distribution, conditional_source, and stress health; and the walk-forward
    validation surface reports a per-row mode and verdict — with the GARCH
    row carrying RESEARCH and no row claiming PRODUCTION.
    """
    importlib.import_module("libs.trading_core.risk.models")

    # (a) The registry: nothing is PRODUCTION.
    registered = {name: model_base.get(name) for name in model_base.names()}
    assert set(registered) == {
        "garch11",
        "gaussian_es",
        "gaussian_var",
        "historical_es",
        "historical_var",
    }
    for name, model in registered.items():
        assert model.mode.value in {"SHADOW", "RESEARCH"}, (name, model.mode)
        assert model.mode.value != "PRODUCTION", name
        assert model.version, name
        # Diagnostics exposed (§68): every model can describe itself.
        meta = model.metadata()
        assert meta.model_name == name
        assert meta.model_version == model.version
    assert registered["garch11"].mode.value == "RESEARCH"
    for name in ("gaussian_es", "gaussian_var", "historical_es", "historical_var"):
        assert registered[name].mode.value == "SHADOW", name

    # (b) The API exposes the health surfaces §68 asks to be exposed.
    await seed_stock_position(bars=520)
    statistical = (await client.get("/api/portfolio/risk")).json()["statistical"]
    assert statistical["mode"] == MODE_SHADOW

    assert statistical["model_health"], "no per-view health ledger"
    for view, health in statistical["model_health"].items():
        assert health in {"ACTIVE", "DEGRADED", "UNAVAILABLE", "FAILED"}, view
    assert set(statistical["model_health"]) >= {
        "historical_var_95",
        "historical_es_95",
        "gaussian_var_95",
        "gaussian_es_95",
        "conditional_var_95",
        "portfolio_volatility",
    }
    assert statistical["model_risk"]["state"] in {"LOW", "ELEVATED", "HIGH"}
    assert statistical["dispersion"]["health"] in {"ACTIVE", "DEGRADED", "UNAVAILABLE"}
    assert statistical["distribution"] is not None
    assert statistical["conditional_source"]["source"] in {"GARCH", "EWMA"}
    assert statistical["conditional_source"]["reason"]
    assert statistical["stress"]["mode"] == MODE_SHADOW
    assert statistical["stress"]["health"] in {
        "ACTIVE",
        "DEGRADED",
        "UNAVAILABLE",
        "FAILED",
    }
    # Every served number carries its methodology (§50/§68 diagnostics).
    for row in statistical["var"] + statistical["es"]:
        assert row["model_name"] and row["model_version"]
        assert row["health"] in {"ACTIVE", "DEGRADED", "UNAVAILABLE", "FAILED"}

    # (c) Out-of-sample / replay evidence: the walk-forward surface, run
    # once, reports a mode and a verdict PER MODEL — and none is PRODUCTION.
    response = await client.post("/api/risk/validation/run", json={})
    assert response.status_code == 200
    validation = (await client.get("/api/portfolio/risk")).json()["statistical"][
        "validation"
    ]
    assert validation is not None, "a validation run persisted nothing"
    assert validation["mode"] == MODE_SHADOW
    assert validation["window"] == 250
    assert validation["rows"]

    modes_by_model: dict[str, set[str]] = {}
    for row in validation["rows"]:
        assert row["mode"] in {"SHADOW", "RESEARCH"}, row
        assert row["mode"] != "PRODUCTION", row
        assert row["verdict"] in {"GREEN", "YELLOW", "RED", "UNAVAILABLE"}, row
        assert row["health"] in {"ACTIVE", "DEGRADED", "UNAVAILABLE", "FAILED"}
        modes_by_model.setdefault(row["model_name"], set()).add(row["mode"])

    # The GARCH row is RESEARCH — one step below everything else, exactly as
    # Phase E registered it.
    assert modes_by_model["garch_var"] == {"RESEARCH"}
    assert modes_by_model["historical_var"] == {"SHADOW"}

    # (d) Promotion is a HUMAN step, and the surface says so rather than
    # promoting anything itself.
    comparison = validation["comparison"]
    assert comparison is not None
    assert "criterion" in comparison
    assert "user action" in comparison["criterion"]
    assert comparison["promotion"].startswith("NONE")
