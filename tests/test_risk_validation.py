"""Tests for ``libs/trading_core/risk/validation.py`` — walk-forward VaR/ES
backtesting and volatility forecast error (risk spec §42, §43, §68; Phase B
design contract §2.10, and contract §3 invariants 5, 6, 10).

Every number below is hand-checked: the arithmetic that produces the
expected value is written out in a comment above the assertion so a reader
can verify it with a calculator, not by re-running the code under test.

Conventions exercised here (contract §1): ``pnl[t] > 0`` is a gain, a loss
is ``L = -pnl``, and VaR/ES forecasts are losses (positive = money lost).

The estimator passed to ``walk_forward`` is a LOCAL toy estimator — this
test must not import ``risk/models/var_es.py`` (the contract lets
``walk_forward`` take any ``Callable[[Sequence[float]], ModelResult]``, and
keeping the dependency out is what makes that generality testable).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from libs.trading_core.risk.models.base import (
    ModelHealth,
    ModelMeta,
    ModelResult,
    active,
    unavailable,
)
from libs.trading_core.risk.validation import (
    BacktestParams,
    BacktestVerdict,
    ExceedanceReport,
    ForecastSeries,
    chi2_1_sf,
    chi2_2_sf,
    christoffersen_independence,
    exceedances,
    kupiec_pof,
    markov_transitions,
    volatility_forecast_error,
    walk_forward,
)

# ---------------------------------------------------------------------------
# Local toy estimators (no var_es import — see module docstring)
# ---------------------------------------------------------------------------


def _meta(name: str = "toy", confidence: float | None = 0.99) -> ModelMeta:
    return ModelMeta(
        model_name=name,
        model_version="1.0.0",
        confidence=confidence,
        horizon_days=1,
        distribution="EMPIRICAL",
    )


def max_loss_estimator(history: Sequence[float]) -> ModelResult:
    """Toy VaR: the largest loss in the window, ``max(-pnl)``, floored at 0.

    Hand-checkable and monotone in the window contents, which is what the
    walk-forward sentinel test needs.
    """
    loss = max((-p for p in history), default=0.0)
    return active(_meta(), max(loss, 0.0), len(history))


def constant_estimator(value: float, *, name: str = "toy", confidence: float | None = 0.99):
    """Estimator returning a fixed forecast (isolates the exceedance logic)."""

    def _est(history: Sequence[float]) -> ModelResult:
        return active(_meta(name, confidence), value, len(history))

    return _est


def _flat_forecasts(var_values, realized, *, confidence: float = 0.99) -> ForecastSeries:
    """A ``ForecastSeries`` built directly (no estimator) for backtest tests."""
    results = tuple(
        active(_meta(confidence=confidence), v, 1)
        if v is not None
        else unavailable(_meta(confidence=confidence), "toy gap", 0)
        for v in var_values
    )
    return ForecastSeries(
        indices=tuple(range(len(var_values))),
        forecasts=tuple(var_values),
        realized=tuple(realized),
        results=results,
        window=1,
        confidence=confidence,
    )


# ===========================================================================
# Walk-forward (spec §43; contract §3 invariant 5)
# ===========================================================================


def test_walk_forward_forecast_count_is_n_minus_window() -> None:
    # len(pnl) = 10, window = 4 -> targets t = 4..9 -> 6 forecasts.
    pnl = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]
    fs = walk_forward(pnl, window=4, estimator=max_loss_estimator)

    assert fs.n_forecasts == len(pnl) - 4 == 6
    assert fs.indices == (4, 5, 6, 7, 8, 9)
    # realized[i] is pnl at the target index, not at the window edge.
    assert fs.realized == (5.0, -6.0, 7.0, -8.0, 9.0, -10.0)
    assert fs.window == 4
    assert fs.n_available == 6
    assert fs.n_unavailable == 0


def test_walk_forward_uses_only_the_preceding_window() -> None:
    # Forecast for t is max(-pnl) over pnl[t-3:t].
    #   t=3: window pnl[0:3] = [1, -2, 3]   -> losses [-1, 2, -3] -> max 2
    #   t=4: window pnl[1:4] = [-2, 3, -4]  -> losses [2, -3, 4]  -> max 4
    #   t=5: window pnl[2:5] = [3, -4, 5]   -> losses [-3, 4, -5] -> max 4
    pnl = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
    fs = walk_forward(pnl, window=3, estimator=max_loss_estimator)

    assert fs.forecasts == (2.0, 4.0, 4.0)


def test_walk_forward_sentinel_spike_never_leaks_into_its_own_forecast() -> None:
    """Contract §3 invariant 5: the forecast for t must not see ``pnl[t]``.

    A huge loss is planted at index 5. Its own forecast (t=5) must be
    unchanged versus the clean series; only the forecasts for t = 6, 7, 8
    (whose windows [3:6], [4:7], [5:8] contain index 5) may change.
    """
    clean = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0]
    spiked = list(clean)
    spiked[5] = -1_000_000.0  # sentinel: a catastrophic loss on day 5

    fs_clean = walk_forward(clean, window=3, estimator=max_loss_estimator)
    fs_spiked = walk_forward(spiked, window=3, estimator=max_loss_estimator)

    # Targets are t = 3..9, i.e. positions 0..6 of the series.
    assert fs_clean.indices == fs_spiked.indices == (3, 4, 5, 6, 7, 8, 9)

    # Position 2 is t=5 — the spike day itself. Its window is pnl[2:5] =
    # [2, -2, 1] in BOTH series (the spike lives at index 5, outside it),
    # so the forecast is max(-[2,-2,1]) = 2 either way.
    assert fs_clean.forecasts[2] == 2.0
    assert fs_spiked.forecasts[2] == 2.0

    # Everything strictly before the spike's window is identical too:
    # positions 0,1,2 -> t=3,4,5 with windows [0:3], [1:4], [2:5].
    assert fs_spiked.forecasts[:3] == fs_clean.forecasts[:3]

    # The realized value at that position DID change (it is pnl[5]).
    assert fs_clean.realized[2] == -1.0
    assert fs_spiked.realized[2] == -1_000_000.0

    # And the spike DOES propagate forward: t=6 window pnl[3:6] contains it.
    #   clean  pnl[3:6] = [-2, 1, -1] -> losses [2, -1, 1] -> max 2
    #   spiked pnl[3:6] = [-2, 1, -1_000_000] -> max 1_000_000
    assert fs_clean.forecasts[3] == 2.0
    assert fs_spiked.forecasts[3] == 1_000_000.0
    # t=9 window pnl[6:9] = [2, -2, 1] no longer contains index 5 -> clean again.
    #   losses [-2, 2, -1] -> max 2 in BOTH series.
    assert fs_spiked.forecasts[6] == fs_clean.forecasts[6] == 2.0


def test_walk_forward_estimator_receives_exactly_window_observations() -> None:
    seen: list[tuple[float, ...]] = []

    def spy(history: Sequence[float]) -> ModelResult:
        seen.append(tuple(history))
        return active(_meta(), 1.0, len(history))

    pnl = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    walk_forward(pnl, window=2, estimator=spy)

    # t = 2..5 -> windows [0:2], [1:3], [2:4], [3:5].
    assert seen == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    assert all(len(h) == 2 for h in seen)


def test_walk_forward_short_series_yields_empty_series_not_an_error() -> None:
    # len(pnl) == window -> range(window, window) is empty; honest empty result.
    fs = walk_forward([1.0, 2.0, 3.0], window=3, estimator=max_loss_estimator)
    assert fs.n_forecasts == 0
    assert fs.forecasts == ()
    assert fs.as_of is None

    fs2 = walk_forward([1.0, 2.0], window=5, estimator=max_loss_estimator)
    assert fs2.n_forecasts == 0


def test_walk_forward_keeps_unavailable_estimator_results_as_none() -> None:
    def flaky(history: Sequence[float]) -> ModelResult:
        # UNAVAILABLE on the first call, ACTIVE afterwards.
        if history[0] == 0.0:
            return unavailable(_meta(), "n=2 < min_obs=60", len(history))
        return active(_meta(), 7.0, len(history))

    fs = walk_forward([0.0, 1.0, 2.0, 3.0, 4.0], window=2, estimator=flaky)
    # t = 2,3,4 -> windows [0:2] (starts 0.0 -> UNAVAILABLE), [1:3], [2:4].
    assert fs.forecasts == (None, 7.0, 7.0)
    assert fs.n_available == 2
    assert fs.n_unavailable == 1
    assert fs.results[0].health is ModelHealth.UNAVAILABLE
    assert fs.results[0].reason == "n=2 < min_obs=60"


def test_walk_forward_infers_confidence_and_name_and_aligns_dates() -> None:
    pnl = [1.0, -1.0, 2.0, -2.0, 3.0]
    d0 = date(2026, 1, 5)
    dates = [d0 + timedelta(days=i) for i in range(5)]
    fs = walk_forward(pnl, window=2, estimator=max_loss_estimator, dates=dates)

    # Confidence comes from the estimator's ModelMeta (0.99 in _meta()).
    assert fs.confidence == 0.99
    assert fs.estimator_name == "toy"
    # Dates align with the TARGET indices t = 2,3,4 -> dates[2:].
    assert fs.dates == (dates[2], dates[3], dates[4])
    assert fs.as_of == dates[4]


def test_walk_forward_malformed_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="window"):
        walk_forward([1.0, 2.0], window=0, estimator=max_loss_estimator)
    with pytest.raises(ValueError, match="finite"):
        walk_forward([1.0, float("nan"), 2.0], window=1, estimator=max_loss_estimator)
    with pytest.raises(ValueError, match="dates length"):
        walk_forward([1.0, 2.0, 3.0], window=1, estimator=max_loss_estimator, dates=[date(2026, 1, 1)])
    with pytest.raises(ValueError, match="ModelResult"):
        walk_forward([1.0, 2.0], window=1, estimator=lambda h: 3.0)  # type: ignore[arg-type,return-value]


# ===========================================================================
# χ² closed forms
# ===========================================================================


def test_chi2_closed_forms() -> None:
    # χ²(1) sf = erfc(sqrt(lr/2)): lr=0 -> erfc(0) = 1.
    assert chi2_1_sf(0.0) == 1.0
    # lr = 3.841458820694124 is the 95th percentile of χ²(1) -> p = 0.05.
    assert chi2_1_sf(3.841458820694124) == pytest.approx(0.05, abs=1e-9)
    # lr = 6.634896601021213 is the 99th percentile -> p = 0.01.
    assert chi2_1_sf(6.634896601021213) == pytest.approx(0.01, abs=1e-9)

    # χ²(2) sf = exp(-lr/2): lr = 5.991464547107979 (95th pct) -> 0.05.
    assert chi2_2_sf(0.0) == 1.0
    assert chi2_2_sf(5.991464547107979) == pytest.approx(0.05, abs=1e-12)
    # exp(-2/2) = exp(-1) = 0.36787944117144233
    assert chi2_2_sf(2.0) == pytest.approx(math.exp(-1.0), abs=1e-15)

    # A materially negative LR is a bug, not rounding.
    with pytest.raises(ValueError):
        chi2_1_sf(-1.0)
    # ...but rounding-scale negatives are clamped to 0.
    assert chi2_1_sf(-1e-12) == 1.0


# ===========================================================================
# Kupiec POF — contract §3 invariant 10 (hand-checked numbers)
# ===========================================================================


def test_kupiec_x_zero_n_250_at_99pct() -> None:
    """x = 0, n = 250, p = 0.01 (contract §3 invariant 10).

    With x = 0 the 0·ln 0 = 0 convention kills both x-terms:
        ll_null = 250·ln(0.99) + 0 = 250·(-0.01005033585350145)
                = -2.5125839633753626
        ll_alt  = 250·ln(1 - 0) + 0 = 0
        LR = -2·(-2.5125839633753626) + 2·0 = 5.025167926750725
           = -500·ln(0.99)                      <- the contract's form
        p  = erfc(sqrt(5.025167926750725 / 2))
           = erfc(1.5851000...) = 0.02498...
    """
    lr, p = kupiec_pof(250, 0, 0.01)
    assert lr == pytest.approx(-500.0 * math.log(0.99), rel=1e-12)
    assert lr == pytest.approx(5.025167926750725, rel=1e-12)
    assert p == pytest.approx(0.024982, abs=5e-7)
    assert p == pytest.approx(math.erfc(math.sqrt(lr / 2.0)), rel=1e-15)
    # 0.01 <= p < 0.05 -> YELLOW band.
    assert 0.01 <= p < 0.05


def test_kupiec_x_two_and_three_n_250_at_99pct_are_green_band() -> None:
    """Expected hits = 250 × 0.01 = 2.5, so x = 2 and x = 3 straddle it.

    x = 2: pi = 2/250 = 0.008
        ll_null = 248·ln(0.99) + 2·ln(0.01)
                = 248·(-0.010050335853501441) + 2·(-4.605170185988091)
                = -2.4924832916683574 - 9.210340371976182 = -11.70282366364454
        ll_alt  = 248·ln(0.992) + 2·ln(0.008)
                = 248·(-0.008032171697338269) + 2·(-4.8283137373023015)
                = -1.9919785808598907 - 9.656627474604603 = -11.648606055464494
        LR = -2·(-11.70282366364454) + 2·(-11.648606055464494) = 0.1084352163600921
        p  = erfc(sqrt(0.1084352163600921/2)) = 0.7419...

    x = 3: pi = 3/250 = 0.012 -> LR = 0.0949398..., p = 0.7580...
    """
    lr2, p2 = kupiec_pof(250, 2, 0.01)
    assert lr2 == pytest.approx(0.108435, abs=5e-7)
    assert p2 == pytest.approx(0.741933, abs=5e-7)
    assert p2 >= 0.05  # GREEN band

    lr3, p3 = kupiec_pof(250, 3, 0.01)
    assert lr3 == pytest.approx(0.094940, abs=5e-7)
    assert p3 == pytest.approx(0.757988, abs=5e-7)
    assert p3 >= 0.05  # GREEN band

    # x = 3 is closer to the expected 2.5 in likelihood terms than x = 2? No:
    # 2.5 sits between them; both LRs are small and x=3's is the smaller.
    assert lr3 < lr2


def test_kupiec_x_ten_n_250_at_99pct_is_red_band() -> None:
    """x = 10, n = 250, p = 0.01 — four times the expected 2.5 hits.

    pi = 10/250 = 0.04
        ll_null = 240·ln(0.99) + 10·ln(0.01)
                = 240·(-0.010050335853501441) + 10·(-4.605170185988091)
                = -2.4120806048403458 - 46.05170185988091 = -48.46378246472126
        ll_alt  = 240·ln(0.96) + 10·ln(0.04)
                = 240·(-0.04082199452025481) + 10·(-3.2188758248682006)
                = -9.797278684861154 - 32.188758248682006 = -41.98603693354316
        LR = -2·(-48.46378246472126) + 2·(-41.98603693354316) = 12.955491062356197
        p  = erfc(sqrt(12.955491062356197/2)) = 3.1898e-4
    """
    lr, p = kupiec_pof(250, 10, 0.01)
    assert lr == pytest.approx(12.955491, abs=5e-7)
    assert p == pytest.approx(0.000319, abs=5e-7)
    assert p < 0.01  # RED band


def test_kupiec_x_equals_n_uses_the_zero_log_zero_convention() -> None:
    """x = n = 5, p = 0.01: every day is an exceedance.

        ll_null = 0·ln(0.99) + 5·ln(0.01) = 5·(-4.605170185988091)
                = -23.025850929940457
        ll_alt  = 0·ln(1-1) + 5·ln(1) = 0        <- 0·ln 0 = 0 keeps it finite
        LR = -2·(-23.025850929940457) + 0 = 46.051701859880914
    """
    lr, p = kupiec_pof(5, 5, 0.01)
    assert lr == pytest.approx(46.051701859880914, rel=1e-12)
    assert p == pytest.approx(math.erfc(math.sqrt(lr / 2.0)), rel=1e-15)
    assert p < 1e-10


def test_kupiec_perfect_coverage_gives_zero_lr() -> None:
    # x/n exactly equals p -> ll_null == ll_alt -> LR = 0 -> p = 1.
    lr, p = kupiec_pof(100, 1, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-12)


def test_kupiec_malformed_input_raises() -> None:
    with pytest.raises(ValueError, match="n must be"):
        kupiec_pof(0, 0, 0.01)
    with pytest.raises(ValueError, match="x must be"):
        kupiec_pof(10, 11, 0.01)
    with pytest.raises(ValueError, match="x must be"):
        kupiec_pof(10, -1, 0.01)
    with pytest.raises(ValueError, match="expected_rate"):
        kupiec_pof(10, 1, 0.0)


# ===========================================================================
# Christoffersen independence
# ===========================================================================


def test_markov_transitions_counts() -> None:
    # hits: F T F F T T -> pairs (F,T)(T,F)(F,F)(F,T)(T,T)
    #   n01 = 2, n10 = 1, n00 = 1, n11 = 1
    hits = [False, True, False, False, True, True]
    assert markov_transitions(hits) == (1, 2, 1, 1)
    # Total transitions = len(hits) - 1 = 5.
    assert sum(markov_transitions(hits)) == len(hits) - 1


def test_christoffersen_no_exceedances_gives_zero_lr() -> None:
    # All-miss: n00 = 9, everything else 0. pi = 0, pi01 = 0, pi11 = 0
    # (undefined, contributes 0 by convention) -> ll_null = ll_alt = 0.
    lr, p, trans = christoffersen_independence([False] * 10)
    assert trans == (9, 0, 0, 0)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-12)


def test_christoffersen_spread_vs_clustered_same_hit_count() -> None:
    """Two 300-day hit sequences with the SAME x = 6, differing only in
    arrangement: evenly spread vs six consecutive days.

    Spread (hits at 25, 75, 125, 175, 225, 275): every hit is isolated, so
    n00 = 287, n01 = 6, n10 = 6, n11 = 0 (299 transitions).
        pi   = (6+0)/299 = 0.020066889632107024
        pi01 = 6/293     = 0.020477815699658702
        pi11 = 0/6       = 0
        ll_null = 293·ln(1-pi) + 6·ln(pi)
                = 293·(-0.020270...) + 6·(-3.908...) = -29.391497186446188
        ll_alt  = 287·ln(1-pi01) + 6·ln(pi01) + 6·ln(1-0) + 0·ln 0
                = -29.26861... (the last two terms are 0)
        LR = 0.24575096565013155  ->  p = erfc(sqrt(LR/2)) = 0.62008...

    Clustered (hits at 100..105): n00 = 292, n01 = 1, n10 = 1, n11 = 5.
        pi   = 6/299,  pi01 = 1/293 = 0.0034129692832764505,  pi11 = 5/6
        ll_null = -29.391497186446188   (same pi, same 293/6 split)
        ll_alt  = 292·ln(1-1/293) + 1·ln(1/293) + 1·ln(1/6) + 5·ln(5/6)
                = -0.99829... - 5.68017... - 1.79175... - 0.91160...
                = -9.381831432860293
        LR = -2·(-29.391497186446188) + 2·(-9.381831432860293)
           = 40.019331507171785  ->  p = 2.5146e-10
    """
    n = 300
    spread = [False] * n
    for i in range(25, n, 50):
        spread[i] = True
    clustered = [False] * n
    for i in range(100, 106):
        clustered[i] = True

    assert sum(spread) == sum(clustered) == 6  # identical unconditional coverage

    lr_s, p_s, trans_s = christoffersen_independence(spread)
    assert trans_s == (287, 6, 6, 0)
    assert lr_s == pytest.approx(0.245751, abs=5e-7)
    assert p_s == pytest.approx(0.620083, abs=5e-7)
    assert p_s > 0.05  # independent -> not clustered

    lr_c, p_c, trans_c = christoffersen_independence(clustered)
    assert trans_c == (292, 1, 1, 5)
    assert lr_c == pytest.approx(40.019332, abs=5e-6)
    assert p_c == pytest.approx(2.5146e-10, rel=1e-3)
    assert p_c < 0.05  # clustered

    # Kupiec cannot tell them apart — that is exactly why we need this test.
    assert kupiec_pof(n, 6, 0.01) == kupiec_pof(n, 6, 0.01)


def test_christoffersen_needs_two_observations() -> None:
    with pytest.raises(ValueError, match=">= 2 observations"):
        christoffersen_independence([True])


# ===========================================================================
# exceedances() — end to end
# ===========================================================================


def _series_with_x_hits(n: int, x: int, *, var: float = 100.0, spacing: int | None = None):
    """n days at a flat VaR forecast of ``var``; ``x`` of them exceed it.

    Non-hit days lose 1.0 (loss 1.0 < 100.0). Hit days lose 150.0.
    Hits are placed evenly (spacing) so they are NOT clustered.
    """
    forecasts = [var] * n
    realized = [-1.0] * n
    if x:
        step = spacing if spacing is not None else n // x
        for j in range(x):
            realized[min(j * step + step // 2, n - 1)] = -150.0
    return forecasts, realized


def test_exceedances_x_zero_n_250_at_99pct_is_yellow() -> None:
    """Contract §3 invariant 10 end to end: n=250, x=0 at 99% -> YELLOW.

    LR = -500·ln(0.99) = 5.025167926750725, p = 0.024982 which is in
    [red_p=0.01, green_p=0.05) -> YELLOW (NOT green: the model is
    over-conservative, and Kupiec is two-sided).
    """
    forecasts = [100.0] * 250
    realized = [-1.0] * 250  # every loss 1.0 << VaR 100 -> no exceedance
    rep = exceedances(forecasts, realized, confidence=0.99)

    assert rep.n == 250
    assert rep.x == 0
    assert rep.rate == 0.0
    assert rep.expected_rate == pytest.approx(0.01)
    assert rep.kupiec_lr == pytest.approx(5.025167926750725, rel=1e-12)
    assert rep.kupiec_p == pytest.approx(0.024982, abs=5e-7)
    assert rep.verdict is BacktestVerdict.YELLOW
    assert rep.health is ModelHealth.ACTIVE
    assert rep.exceedance_positions == ()
    # No hits at all -> Christoffersen LR 0, p 1 -> not clustered.
    assert rep.christoffersen_lr == pytest.approx(0.0, abs=1e-12)
    assert rep.clustered is False
    # Conditional coverage = LR_uc + LR_ind = 5.025168 + 0, χ²(2):
    #   p = exp(-5.025167926750725/2) = exp(-2.5125839633753626) = 0.081092...
    assert rep.conditional_coverage_lr == pytest.approx(5.025167926750725, rel=1e-12)
    assert rep.conditional_coverage_p == pytest.approx(math.exp(-2.5125839633753626), rel=1e-12)


@pytest.mark.parametrize(
    ("x", "expected_lr", "expected_p"),
    [
        (2, 0.108435, 0.741933),  # see test_kupiec_x_two_and_three...
        (3, 0.094940, 0.757988),
    ],
)
def test_exceedances_x_two_or_three_n_250_at_99pct_is_green(x, expected_lr, expected_p) -> None:
    forecasts, realized = _series_with_x_hits(250, x)
    rep = exceedances(forecasts, realized, confidence=0.99)

    assert rep.n == 250
    assert rep.x == x
    assert rep.rate == pytest.approx(x / 250)
    assert rep.kupiec_lr == pytest.approx(expected_lr, abs=5e-7)
    assert rep.kupiec_p == pytest.approx(expected_p, abs=5e-7)
    assert rep.verdict is BacktestVerdict.GREEN
    assert len(rep.exceedance_positions) == x


def test_exceedances_x_ten_n_250_at_99pct_is_red() -> None:
    forecasts, realized = _series_with_x_hits(250, 10)
    rep = exceedances(forecasts, realized, confidence=0.99)

    assert rep.n == 250
    assert rep.x == 10
    assert rep.rate == pytest.approx(0.04)  # 10/250
    assert rep.kupiec_lr == pytest.approx(12.955491, abs=5e-7)
    assert rep.kupiec_p == pytest.approx(0.000319, abs=5e-7)
    assert rep.kupiec_p < 0.01
    assert rep.verdict is BacktestVerdict.RED


def test_exceedances_verdict_bands_follow_params() -> None:
    """The traffic light is p-driven and every threshold is a parameter."""
    forecasts = [100.0] * 250
    realized = [-1.0] * 250  # x = 0 -> p = 0.024982

    # Default params: 0.01 <= 0.024982 < 0.05 -> YELLOW.
    assert exceedances(forecasts, realized, confidence=0.99).verdict is BacktestVerdict.YELLOW
    # Loosen green_p below the p-value -> GREEN.
    loose = BacktestParams(green_p=0.02, red_p=0.005)
    assert exceedances(forecasts, realized, confidence=0.99, params=loose).verdict is BacktestVerdict.GREEN
    # Raise red_p above the p-value -> RED.
    strict = BacktestParams(green_p=0.5, red_p=0.05)
    assert exceedances(forecasts, realized, confidence=0.99, params=strict).verdict is BacktestVerdict.RED


def test_exceedances_is_strict_a_loss_equal_to_var_is_not_a_hit() -> None:
    # Day 0 loses exactly 100.0 = VaR -> NOT an exceedance (strict >).
    # Day 1 loses 100.0000001 -> exceedance.
    forecasts = [100.0, 100.0] + [100.0] * 248
    realized = [-100.0, -100.0000001] + [-1.0] * 248
    rep = exceedances(forecasts, realized, confidence=0.99)
    assert rep.x == 1
    assert rep.exceedance_positions == (1,)


def test_exceedances_clustered_flag_distinguishes_arrangements() -> None:
    """Same x = 6 over n = 300; only the arrangement differs (see the
    Christoffersen test for the full LR arithmetic)."""
    n = 300
    forecasts = [100.0] * n

    spread = [-1.0] * n
    for i in range(25, n, 50):
        spread[i] = -150.0
    rep_spread = exceedances(forecasts, spread, confidence=0.99)

    clustered = [-1.0] * n
    for i in range(100, 106):
        clustered[i] = -150.0
    rep_clustered = exceedances(forecasts, clustered, confidence=0.99)

    assert rep_spread.x == rep_clustered.x == 6
    # Identical Kupiec (unconditional coverage cannot see arrangement).
    assert rep_spread.kupiec_lr == pytest.approx(rep_clustered.kupiec_lr, rel=1e-12)
    assert rep_spread.kupiec_p == pytest.approx(rep_clustered.kupiec_p, rel=1e-12)

    assert rep_spread.transitions == (287, 6, 6, 0)
    assert rep_spread.christoffersen_p == pytest.approx(0.620083, abs=5e-7)
    assert rep_spread.clustered is False

    assert rep_clustered.transitions == (292, 1, 1, 5)
    assert rep_clustered.christoffersen_lr == pytest.approx(40.019332, abs=5e-6)
    assert rep_clustered.christoffersen_p == pytest.approx(2.5146e-10, rel=1e-3)
    assert rep_clustered.clustered is True
    assert rep_clustered.exceedance_positions == (100, 101, 102, 103, 104, 105)


def test_exceedances_es_severity_ratio_arithmetic() -> None:
    """ES severity = mean realized loss on hit days ÷ mean ES forecast on
    those same days.

    n = 250, VaR = 100 everywhere, ES = 120 everywhere.
    Three hit days with losses 150, 180 and 210 (all > 100).
        mean realized loss on hits = (150 + 180 + 210)/3 = 540/3 = 180
        mean ES forecast on hits   = (120 + 120 + 120)/3 = 120
        ratio = 180 / 120 = 1.5      (> 1: the tail was worse than forecast)
    """
    n = 250
    forecasts = [100.0] * n
    es = [120.0] * n
    realized = [-1.0] * n
    realized[40] = -150.0
    realized[120] = -180.0
    realized[200] = -210.0

    rep = exceedances(forecasts, realized, confidence=0.99, es_forecasts=es)

    assert rep.x == 3
    assert rep.es_n == 3
    assert rep.es_severity_ratio == pytest.approx(1.5, rel=1e-12)
    assert rep.verdict is BacktestVerdict.GREEN  # x=3 -> p = 0.757988


def test_exceedances_es_severity_ratio_below_one_when_es_is_conservative() -> None:
    """Two hits, losses 110 and 130; ES forecasts 200 and 300.
        mean loss = (110 + 130)/2 = 120
        mean ES   = (200 + 300)/2 = 250
        ratio     = 120/250 = 0.48   (< 1: ES was comfortably conservative)
    """
    n = 250
    forecasts = [100.0] * n
    es = [200.0] * n
    es[180] = 300.0
    realized = [-1.0] * n
    realized[60] = -110.0
    realized[180] = -130.0

    rep = exceedances(forecasts, realized, confidence=0.99, es_forecasts=es)
    assert rep.es_n == 2
    assert rep.es_severity_ratio == pytest.approx(0.48, rel=1e-12)


def test_exceedances_es_severity_is_none_without_hits_or_without_es() -> None:
    n = 250
    forecasts = [100.0] * n
    realized = [-1.0] * n  # no hits

    # ES supplied but no exceedance days -> nothing to average over.
    rep_no_hits = exceedances(forecasts, realized, confidence=0.99, es_forecasts=[120.0] * n)
    assert rep_no_hits.x == 0
    assert rep_no_hits.es_severity_ratio is None
    assert rep_no_hits.es_n == 0

    # Hits but no ES forecasts at all.
    realized[10] = -150.0
    rep_no_es = exceedances(forecasts, realized, confidence=0.99)
    assert rep_no_es.x == 1
    assert rep_no_es.es_severity_ratio is None


def test_exceedances_below_min_forecasts_is_unavailable() -> None:
    """Contract §3 invariant 6: honest None + UNAVAILABLE + a real reason."""
    forecasts = [100.0] * 30
    realized = [-1.0] * 30
    realized[5] = -150.0
    rep = exceedances(forecasts, realized, confidence=0.99)

    assert rep.verdict is BacktestVerdict.UNAVAILABLE
    assert rep.health is ModelHealth.UNAVAILABLE
    assert rep.reason is not None and "n=30" in rep.reason and "min_forecasts=250" in rep.reason
    # Every inferential statistic is None — never a fabricated 0.
    assert rep.kupiec_lr is None
    assert rep.kupiec_p is None
    assert rep.christoffersen_lr is None
    assert rep.christoffersen_p is None
    assert rep.clustered is None
    assert rep.conditional_coverage_lr is None
    assert rep.transitions is None
    assert rep.is_available is False
    # ...but the raw counts, which are observations rather than inferences,
    # are still reported honestly.
    assert rep.n == 30
    assert rep.x == 1
    assert rep.rate == pytest.approx(1 / 30)
    assert rep.exceedance_positions == (5,)


def test_exceedances_min_forecasts_boundary_is_inclusive() -> None:
    # n == min_forecasts is enough (the gate is n < min_forecasts).
    params = BacktestParams(min_forecasts=50)
    forecasts = [100.0] * 50
    realized = [-1.0] * 50
    rep = exceedances(forecasts, realized, confidence=0.99, params=params)
    assert rep.verdict is not BacktestVerdict.UNAVAILABLE
    assert rep.n == 50

    rep49 = exceedances(forecasts[:49], realized[:49], confidence=0.99, params=params)
    assert rep49.verdict is BacktestVerdict.UNAVAILABLE


def test_exceedances_skips_none_forecasts_and_degrades() -> None:
    """Days with no forecast are neither hits nor misses; they are counted."""
    n = 260
    forecasts: list[float | None] = [100.0] * n
    realized = [-1.0] * n
    for i in range(5):
        forecasts[i] = None
        realized[i] = -9999.0  # would be a hit if it were not skipped
    realized[100] = -150.0
    realized[200] = -150.0

    rep = exceedances(forecasts, realized, confidence=0.99)
    assert rep.n == 255  # 260 - 5 skipped
    assert rep.n_skipped == 5
    assert rep.x == 2  # the 5 huge losses on no-forecast days do NOT count
    assert rep.health is ModelHealth.DEGRADED
    assert rep.reason is not None and "5 of 260" in rep.reason
    # Positions index into the ORIGINAL series (100 and 200), not the compacted one.
    assert rep.exceedance_positions == (100, 200)


def test_exceedances_accepts_a_forecast_series_directly() -> None:
    """A ``ForecastSeries`` carries its own realized P&L and confidence."""
    n = 250
    fs = _flat_forecasts([100.0] * n, [-1.0] * n, confidence=0.99)
    rep = exceedances(fs)  # no realized_pnl, no confidence needed
    assert rep.n == 250
    assert rep.x == 0
    assert rep.confidence == 0.99
    assert rep.verdict is BacktestVerdict.YELLOW


def test_exceedances_end_to_end_through_walk_forward() -> None:
    """walk_forward -> exceedances with a constant estimator, so the whole
    pipe can be hand-checked.

    pnl: 260 days, every day -1.0 except days 60, 130 and 220 which lose
    150.0. window = 10 -> forecasts for t = 10..259, i.e. n = 250 pairs.
    The estimator always forecasts VaR = 100, so the hits are exactly the
    three big-loss days (all at t >= 10) -> x = 3 -> p = 0.757988 -> GREEN.
    """
    pnl = [-1.0] * 260
    for t in (60, 130, 220):
        pnl[t] = -150.0

    fs = walk_forward(pnl, window=10, estimator=constant_estimator(100.0))
    assert fs.n_forecasts == 260 - 10 == 250

    rep = exceedances(fs)
    assert rep.n == 250
    assert rep.x == 3
    assert rep.kupiec_p == pytest.approx(0.757988, abs=5e-7)
    assert rep.verdict is BacktestVerdict.GREEN
    # Positions are offsets into the forecast series: t - window.
    assert rep.exceedance_positions == (60 - 10, 130 - 10, 220 - 10)


def test_exceedances_malformed_input_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        exceedances([100.0, 100.0], [-1.0], confidence=0.99)
    with pytest.raises(ValueError, match="confidence"):
        exceedances([100.0] * 5, [-1.0] * 5, confidence=0.4)
    with pytest.raises(ValueError, match="confidence is required"):
        exceedances([100.0] * 5, [-1.0] * 5)
    with pytest.raises(ValueError, match="realized_pnl is required"):
        exceedances([100.0] * 5, confidence=0.99)
    with pytest.raises(ValueError, match="es_forecasts length"):
        exceedances([100.0] * 5, [-1.0] * 5, confidence=0.99, es_forecasts=[120.0] * 4)
    with pytest.raises(ValueError, match="finite"):
        exceedances([100.0, float("inf")], [-1.0, -1.0], confidence=0.99)


def test_backtest_params_validation() -> None:
    with pytest.raises(ValueError, match="red_p"):
        BacktestParams(green_p=0.01, red_p=0.05)  # red must be < green
    with pytest.raises(ValueError, match="min_forecasts"):
        BacktestParams(min_forecasts=0)
    with pytest.raises(ValueError, match="cluster_p"):
        BacktestParams(cluster_p=1.5)


# ===========================================================================
# Volatility forecast error (spec §42)
# ===========================================================================


def test_volatility_forecast_error_perfect_forecast_gives_zero() -> None:
    """A perfect forecast (σ_t = |r_t|) has MSE = 0 and QLIKE = 0.

    ratio = r²/σ² = 1 for every day, so each QLIKE term is
    1 - ln(1) - 1 = 0, and each squared error is (r² - σ²)² = 0.
    """
    rets = [0.01, -0.02, 0.015, -0.005] * 10  # 40 days, none zero
    sigmas = [abs(r) for r in rets]
    res = volatility_forecast_error(sigmas, rets)

    assert res.n == 40
    assert res.qlike_n == 40
    assert res.mse == pytest.approx(0.0, abs=1e-30)
    assert res.qlike == pytest.approx(0.0, abs=1e-15)
    assert res.health is ModelHealth.ACTIVE
    assert res.reason is None


def test_volatility_forecast_error_hand_checked_mse_and_qlike() -> None:
    """20 days: σ = 2.0 every day; r = 1.0 on 10 days and r = 4.0 on 10.

    σ² = 4.
    r = 1: r² = 1.  squared error = (1 - 4)² = 9.
           ratio = 1/4 = 0.25, term = 0.25 - ln(0.25) - 1
                 = 0.25 + 1.3862943611198906 - 1 = 0.6362943611198906
    r = 4: r² = 16. squared error = (16 - 4)² = 144.
           ratio = 16/4 = 4, term = 4 - ln(4) - 1
                 = 4 - 1.3862943611198906 - 1 = 1.6137056388801094

    MSE   = (10·9 + 10·144)/20 = (90 + 1440)/20 = 1530/20 = 76.5
    QLIKE = (10·0.6362943611198906 + 10·1.6137056388801094)/20
          = (6.362943611198906 + 16.137056388801094)/20
          = 22.5/20 = 1.125          <- ln(0.25) and ln(4) cancel exactly
    """
    rets = [1.0] * 10 + [4.0] * 10
    sigmas = [2.0] * 20
    res = volatility_forecast_error(sigmas, rets)

    assert res.n == 20
    assert res.qlike_n == 20
    assert res.mse == pytest.approx(76.5, rel=1e-12)
    assert res.qlike == pytest.approx(1.125, rel=1e-12)
    assert res.health is ModelHealth.ACTIVE


def test_volatility_forecast_error_qlike_is_non_negative_and_minimised_at_truth() -> None:
    """QLIKE ≥ 0 with equality only when σ² = r²; it grows as σ moves away.

    Same r = 1.0 for 25 days:
      σ = 1.0 -> ratio 1     -> term 0
      σ = 2.0 -> ratio 0.25  -> 0.25 - ln(0.25) - 1 = 0.6362943611198906
      σ = 0.5 -> ratio 4     -> 4 - ln(4) - 1       = 1.6137056388801094
    """
    rets = [1.0] * 25
    q_true = volatility_forecast_error([1.0] * 25, rets).qlike
    q_over = volatility_forecast_error([2.0] * 25, rets).qlike
    q_under = volatility_forecast_error([0.5] * 25, rets).qlike

    assert q_true == pytest.approx(0.0, abs=1e-15)
    assert q_over == pytest.approx(0.6362943611198906, rel=1e-12)
    assert q_under == pytest.approx(1.6137056388801094, rel=1e-12)
    assert q_true < q_over < q_under  # under-forecasting σ is penalised harder


def test_volatility_forecast_error_zero_returns_excluded_from_qlike() -> None:
    """r = 0 makes ln(r²/σ²) undefined; those days leave QLIKE but stay in MSE.

    25 days: 5 with r = 0, 20 with r = 1; σ = 1 throughout (σ² = 1).
      MSE   = (5·(0 - 1)² + 20·(1 - 1)²)/25 = (5·1 + 0)/25 = 5/25 = 0.2
      QLIKE = mean over the 20 non-zero days of (1 - ln 1 - 1) = 0
    """
    rets = [0.0] * 5 + [1.0] * 20
    sigmas = [1.0] * 25
    res = volatility_forecast_error(sigmas, rets)

    assert res.n == 25
    assert res.n_zero_returns == 5
    assert res.qlike_n == 20
    assert res.mse == pytest.approx(0.2, rel=1e-12)
    assert res.qlike == pytest.approx(0.0, abs=1e-15)
    assert res.health is ModelHealth.DEGRADED
    assert res.reason is not None and "5 zero-return days" in res.reason


def test_volatility_forecast_error_skips_none_forecasts() -> None:
    """None σ (e.g. the EWMA burn-in) is a data gap: skipped, counted, DEGRADED.

    25 entries: the first 5 σ are None, the remaining 20 are σ = 2 with
    r = 2 -> ratio 1 -> QLIKE term 0, squared error (4 - 4)² = 0.
    """
    sigmas: list[float | None] = [None] * 5 + [2.0] * 20
    rets = [99.0] * 5 + [2.0] * 20  # the skipped days' returns are ignored
    res = volatility_forecast_error(sigmas, rets)

    assert res.n == 20
    assert res.n_skipped == 5
    assert res.mse == pytest.approx(0.0, abs=1e-30)
    assert res.qlike == pytest.approx(0.0, abs=1e-15)
    assert res.health is ModelHealth.DEGRADED
    assert res.reason is not None and "5 of 25" in res.reason


def test_volatility_forecast_error_below_min_obs_is_unavailable() -> None:
    """Contract §3 invariant 6."""
    res = volatility_forecast_error([1.0] * 5, [1.0] * 5)
    assert res.n == 5
    assert res.mse is None
    assert res.qlike is None
    assert res.health is ModelHealth.UNAVAILABLE
    assert res.reason is not None and "n=5" in res.reason and "min_obs=20" in res.reason

    # min_obs is a parameter, not a magic number.
    ok = volatility_forecast_error([1.0] * 5, [1.0] * 5, min_obs=5)
    assert ok.health is ModelHealth.ACTIVE
    assert ok.mse == pytest.approx(0.0, abs=1e-30)


def test_volatility_forecast_error_malformed_input_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        volatility_forecast_error([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="must be > 0"):
        volatility_forecast_error([1.0] * 19 + [0.0], [1.0] * 20)
    with pytest.raises(ValueError, match="must be > 0"):
        volatility_forecast_error([1.0] * 19 + [-1.0], [1.0] * 20)
    with pytest.raises(ValueError, match="finite"):
        volatility_forecast_error([1.0] * 20, [1.0] * 19 + [float("nan")])
    with pytest.raises(ValueError, match="min_obs"):
        volatility_forecast_error([1.0] * 20, [1.0] * 20, min_obs=0)


# ===========================================================================
# Shape / dataclass guards
# ===========================================================================


def test_forecast_series_rejects_misaligned_fields() -> None:
    r = active(_meta(), 1.0, 1)
    with pytest.raises(ValueError, match="equal length"):
        ForecastSeries(
            indices=(0, 1), forecasts=(1.0,), realized=(0.0, 0.0),
            results=(r, r), window=1,
        )
    with pytest.raises(ValueError, match="dates must align"):
        ForecastSeries(
            indices=(0,), forecasts=(1.0,), realized=(0.0,),
            results=(r,), window=1, dates=(date(2026, 1, 1), date(2026, 1, 2)),
        )
    with pytest.raises(ValueError, match="window"):
        ForecastSeries(
            indices=(0,), forecasts=(1.0,), realized=(0.0,), results=(r,), window=0,
        )


def test_exceedance_report_is_frozen_and_carries_params() -> None:
    rep = exceedances([100.0] * 250, [-1.0] * 250, confidence=0.99)
    assert isinstance(rep, ExceedanceReport)
    assert rep.params.min_forecasts == 250
    assert rep.params.green_p == 0.05
    assert rep.params.red_p == 0.01
    assert rep.version == "1.0.0"
    with pytest.raises(Exception):  # frozen dataclass
        rep.x = 5  # type: ignore[misc]
