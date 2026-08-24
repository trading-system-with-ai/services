"""Stress engine tests (Phase D design §8.3 / §8.7).

The hand-built close path used by the historical tests (one ticker, AAA;
a second, BBB, is added where the per-ticker behaviour matters). Dates are
consecutive business days so the windows are easy to state:

    idx  date        AAA close
     0   2026-01-02  100.0
     1   2026-01-05  101.0
     2   2026-01-06  102.0
     3   2026-01-07  103.0
     4   2026-01-08  104.0
     5   2026-01-09  105.0
     6   2026-01-12   95.0      <- the crash day (-9.5238 %)
     7   2026-01-13   90.0      <- (-5.2632 %)
     8   2026-01-14   93.0
     9   2026-01-15   94.0

Cumulative simple return over a window is close(end)/close(start) - 1:

    2026-01-09 .. 2026-01-13 :  90/105 - 1 = -0.142857142857...
    2026-01-12 .. 2026-01-13 :  90/95  - 1 = -0.052631578947...

Option-leg baseline reused from the reval tests (r = q = 0):

    S0 = K = 100, T0 = 1.0, iv0 = 0.20, C, mark0 = 8.00
    ->  model0 = 7.96556745540580, basis = 0.03443254459420
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from libs.trading_core.options.reval import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL,
    OptionLeg,
    StockLeg,
)
from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.stress import (
    CATALOGUE_VERSION,
    CODE_STRESS_LOSS,
    DEFAULT_HISTORICAL_WINDOWS,
    DEFAULT_HYPOTHETICAL_SCENARIOS,
    IV_SHOCK_SOURCE_RV_PROXY,
    IV_SHOCK_SOURCE_SPECIFIED,
    KIND_HISTORICAL,
    KIND_HYPOTHETICAL,
    KIND_IV_GRID,
    KIND_USER,
    LAYER_STRESS,
    HistoricalShockParams,
    HistoricalWindow,
    Scenario,
    StressLimits,
    auto_worst_windows,
    default_scenarios,
    historical_shocks_from_closes,
    run_scenario,
    run_stress,
    stress_caps,
)

# ---------------------------------------------------------------------------
# Fixtures: the hand-built path and the canonical legs
# ---------------------------------------------------------------------------

DATES = [
    date(2026, 1, 2),
    date(2026, 1, 5),
    date(2026, 1, 6),
    date(2026, 1, 7),
    date(2026, 1, 8),
    date(2026, 1, 9),
    date(2026, 1, 12),
    date(2026, 1, 13),
    date(2026, 1, 14),
    date(2026, 1, 15),
]
AAA_CLOSES = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 95.0, 90.0, 93.0, 94.0]
AAA = list(zip(DATES, AAA_CLOSES))

# BBB rises steadily and never crashes — so a window's per-ticker shocks
# must differ, which is what the per-ticker test pins.
BBB_CLOSES = [50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5]
BBB = list(zip(DATES, BBB_CLOSES))

MARK0 = 8.00


def call(quantity: int = 1, **kw) -> OptionLeg:
    params = dict(
        key="AAA#1",
        ticker="AAA",
        right="C",
        strike=100.0,
        t_years=1.0,
        quantity=quantity,
        spot0=100.0,
        mark0=MARK0,
        iv0=0.20,
        r=0.0,
        q=0.0,
    )
    params.update(kw)
    return OptionLeg(**params)


def flat(name: str = "flat", **kw) -> Scenario:
    params = dict(name=name, kind=KIND_HYPOTHETICAL)
    params.update(kw)
    return Scenario(**params)


# ---------------------------------------------------------------------------
# historical_shocks_from_closes — hand-computed
# ---------------------------------------------------------------------------


def test_historical_shock_is_the_cumulative_window_return() -> None:
    """90/105 - 1 = -0.14285714285714288 over 2026-01-09..2026-01-13."""
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.kind == KIND_HISTORICAL
    assert sc.spot_shock_by_ticker["AAA"] == pytest.approx(
        90.0 / 105.0 - 1.0, abs=1e-15
    )
    assert sc.spot_shock_by_ticker["AAA"] == pytest.approx(-0.142857142857, abs=1e-10)
    assert sc.is_uniform is False  # per-ticker: no beta=1 assumption
    assert sc.validated is False
    assert sc.source == "STORED_CLOSES"


def test_historical_days_forward_is_the_calendar_span() -> None:
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.days_forward == 4.0  # Jan 9 -> Jan 13


def test_historical_shocks_are_per_ticker() -> None:
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA, "BBB": BBB})
    assert sc.spot_shock_by_ticker["AAA"] == pytest.approx(90.0 / 105.0 - 1.0)
    assert sc.spot_shock_by_ticker["BBB"] == pytest.approx(53.5 / 52.5 - 1.0)
    assert sc.spot_shock_by_ticker["AAA"] < 0 < sc.spot_shock_by_ticker["BBB"]


def test_iv_shock_is_labelled_rv_proxy_never_a_measured_iv() -> None:
    """Spec §24: no IV history exists, so the shock MUST say it is a proxy."""
    w = HistoricalWindow("crash", date(2026, 1, 5), date(2026, 1, 15))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.iv_shock_source == IV_SHOCK_SOURCE_RV_PROXY
    assert "RV-ratio proxy" in sc.notes
    assert "no IV history" in sc.notes


def test_iv_proxy_is_zero_with_a_reason_when_there_is_too_little_data() -> None:
    """A 2-bar window has no RV; the shock is 0.0 AND says why."""
    w = HistoricalWindow("tiny", date(2026, 1, 12), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.iv_shock == 0.0
    assert "IV shock 0.0" in sc.reason
    assert "min_rv_obs=5" in sc.reason


def test_iv_proxy_is_clipped_and_the_clip_is_reported() -> None:
    """A violent window against a dead-calm prior would give an absurd
    ratio; the clip keeps a proxy from becoming fiction, and says so."""
    # 25 flat-ish bars then a violent 6-bar window.
    calm_dates = [date(2025, 12, 1) + _bd(i) for i in range(25)]
    calm = [(d, 100.0 + 0.01 * i) for i, d in enumerate(calm_dates)]
    wild_dates = [date(2026, 1, 5) + _bd(i) for i in range(6)]
    wild_closes = [100.25, 120.0, 90.0, 115.0, 85.0, 110.0]
    wild = list(zip(wild_dates, wild_closes))
    bars = calm + wild
    w = HistoricalWindow("wild", wild_dates[0], wild_dates[-1])
    params = HistoricalShockParams()
    sc = historical_shocks_from_closes(w, {"AAA": bars}, params=params)
    assert sc.iv_shock == pytest.approx(params.iv_shock_ceiling)
    assert "clipped" in sc.reason


def _bd(n: int):
    """n days as a timedelta (helper for the clip fixture)."""
    from datetime import timedelta

    return timedelta(days=n)


def test_window_outside_the_stored_history_is_unavailable_not_zero() -> None:
    w = HistoricalWindow("2019", date(2019, 3, 1), date(2019, 3, 8))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.health is ModelHealth.UNAVAILABLE
    assert sc.spot_shock_by_ticker == {}
    assert "no stored history covers" in sc.reason
    assert "2019-03-01" in sc.reason and "2019-03-08" in sc.reason


def test_partially_covered_ticker_is_excluded_and_named() -> None:
    """A ticker whose history starts inside the window is dropped, not
    silently rescaled onto a shorter window."""
    short = AAA[5:]  # starts 2026-01-09
    w = HistoricalWindow("crash", date(2026, 1, 5), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA, "SHORT": short})
    assert "AAA" in sc.spot_shock_by_ticker
    assert "SHORT" not in sc.spot_shock_by_ticker
    assert sc.health is ModelHealth.DEGRADED
    assert "SHORT" in sc.reason


def test_no_closes_at_all_is_unavailable() -> None:
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {})
    assert sc.health is ModelHealth.UNAVAILABLE


def test_historical_window_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError):
        HistoricalWindow("bad", date(2026, 1, 13), date(2026, 1, 9))


# ---------------------------------------------------------------------------
# auto_worst_windows
# ---------------------------------------------------------------------------


def test_auto_worst_1_day_window_is_the_crash_day() -> None:
    """The worst single day is 2026-01-09 -> 2026-01-12 (95/105 - 1)."""
    (w,) = auto_worst_windows({"AAA": AAA}, lengths=(1,))
    assert w.start == date(2026, 1, 9)
    assert w.end == date(2026, 1, 12)
    assert "AUTO worst 1-day" in w.name
    assert "-9.52%" in w.name


def test_auto_worst_2_day_window_spans_both_crash_days() -> None:
    (w,) = auto_worst_windows({"AAA": AAA}, lengths=(2,))
    assert w.start == date(2026, 1, 9)
    assert w.end == date(2026, 1, 13)
    # 90/105 - 1 = -14.29 %
    assert "-14.29%" in w.name


def test_auto_windows_round_trip_through_the_shock_builder() -> None:
    """The window the finder returns must reproduce the move it named."""
    (w,) = auto_worst_windows({"AAA": AAA}, lengths=(2,))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    assert sc.spot_shock_by_ticker["AAA"] == pytest.approx(90.0 / 105.0 - 1.0)


def test_auto_worst_windows_default_lengths() -> None:
    """Defaults are (1, 5, 10). The fixture has 10 bars = 9 returns, so the
    10-day window cannot be formed and is correctly ABSENT — an honest gap,
    not a shortened window presented as a 10-day one."""
    ws = auto_worst_windows({"AAA": AAA})
    names = [w.name for w in ws]
    assert len(ws) == 2
    assert any("1-day" in n for n in names)
    assert any("5-day" in n for n in names)
    assert not any("10-day" in n for n in names)
    # With one more bar the 10-day window appears.
    longer = AAA + [(date(2026, 1, 16), 96.0)]
    assert any("10-day" in w.name for w in auto_worst_windows({"AAA": longer}))


def test_auto_worst_windows_skip_lengths_longer_than_the_history() -> None:
    """An absent window is honest; a fabricated one is not."""
    ws = auto_worst_windows({"AAA": AAA[:4]}, lengths=(1, 5, 10))
    # only 3 returns exist -> only the 1-day window can be formed
    assert len(ws) == 1
    assert "1-day" in ws[0].name


def test_auto_worst_windows_use_the_equal_weight_book() -> None:
    """With a steadily rising BBB the equal-weight worst day is still the
    crash day, but the magnitude is halved."""
    (w,) = auto_worst_windows({"AAA": AAA, "BBB": BBB}, lengths=(1,))
    assert w.start == date(2026, 1, 9)
    assert w.end == date(2026, 1, 12)
    # (-0.095238 + 0.009524) / 2 = -0.042857
    assert "-4.29%" in w.name


def test_auto_worst_windows_inner_join_the_dates() -> None:
    """A ticker missing a date drops it for everyone (no compounding
    across a gap)."""
    gapped = [b for b in BBB if b[0] != date(2026, 1, 12)]
    ws = auto_worst_windows({"AAA": AAA, "BBB": gapped}, lengths=(1,))
    assert ws  # still produces a window
    assert date(2026, 1, 12) not in (ws[0].start, ws[0].end)


def test_auto_worst_windows_empty_and_degenerate_inputs() -> None:
    assert auto_worst_windows({}) == ()
    assert auto_worst_windows({"AAA": []}) == ()
    assert auto_worst_windows({"AAA": AAA[:1]}) == ()


def test_auto_worst_windows_reject_bad_lengths() -> None:
    with pytest.raises(ValueError):
        auto_worst_windows({"AAA": AAA}, lengths=(0,))


def test_auto_worst_windows_reject_non_positive_closes() -> None:
    with pytest.raises(ValueError):
        auto_worst_windows({"AAA": [(DATES[0], 100.0), (DATES[1], 0.0)]})


# ---------------------------------------------------------------------------
# The catalogue — shape and the UNVALIDATED contract
# ---------------------------------------------------------------------------


def test_every_hypothetical_scenario_is_unvalidated() -> None:
    """Spec §24: 'Do not blindly adopt these example numbers.'"""
    assert len(DEFAULT_HYPOTHETICAL_SCENARIOS) == 7
    for sc in DEFAULT_HYPOTHETICAL_SCENARIOS:
        assert sc.validated is False, sc.name
        assert sc.kind in (KIND_HYPOTHETICAL, KIND_IV_GRID)
        assert sc.iv_shock_source == IV_SHOCK_SOURCE_SPECIFIED
        assert sc.notes  # every research row explains itself


def test_hypothetical_catalogue_covers_the_design_grid() -> None:
    names = [sc.name for sc in DEFAULT_HYPOTHETICAL_SCENARIOS]
    assert names == [
        "Equity -5% / IV +20%",
        "Equity -10% / IV +40%",
        "Equity +5% / IV -15%",
        "IV crush (flat, -40%)",
        "IV spike (flat, +50%)",
        "Correlation convergence (all names -8%, IV +30%)",
        "Time decay only (+5 days)",
    ]


def test_hypothetical_scenario_parameters_are_the_documented_ones() -> None:
    by_name = {sc.name: sc for sc in DEFAULT_HYPOTHETICAL_SCENARIOS}
    assert by_name["Equity -5% / IV +20%"].spot_shock == -0.05
    assert by_name["Equity -5% / IV +20%"].iv_shock == 0.20
    assert by_name["Equity -10% / IV +40%"].spot_shock == -0.10
    assert by_name["IV crush (flat, -40%)"].spot_shock == 0.0
    assert by_name["IV crush (flat, -40%)"].iv_shock == -0.40
    assert by_name["Time decay only (+5 days)"].days_forward == 5.0
    assert by_name["Time decay only (+5 days)"].spot_shock == 0.0


def test_uniform_scenarios_declare_the_beta_1_assumption() -> None:
    for sc in DEFAULT_HYPOTHETICAL_SCENARIOS:
        assert sc.is_uniform is True
        assert sc.params()["uniform_beta_1"] is True


def test_default_historical_windows_shape() -> None:
    assert len(DEFAULT_HISTORICAL_WINDOWS) == 2
    names = [w.name for w in DEFAULT_HISTORICAL_WINDOWS]
    assert names == ["2024-08-05 vol spike", "2025-04 tariff drawdown"]


def test_default_scenarios_composes_historical_auto_and_grid() -> None:
    scs = default_scenarios({"AAA": AAA}, auto_lengths=(1, 2))
    kinds = [sc.kind for sc in scs]
    # 2 named windows (UNAVAILABLE here) + 2 auto + 7 grid
    assert len(scs) == 2 + 2 + 7
    assert kinds[:4] == [KIND_HISTORICAL] * 4
    assert [sc.source for sc in scs[2:4]] == ["AUTO_WORST_WINDOW"] * 2
    # The named 2024/2025 windows are outside the 2026 fixture history.
    assert scs[0].health is ModelHealth.UNAVAILABLE
    assert scs[1].health is ModelHealth.UNAVAILABLE


def test_default_scenarios_without_closes_is_the_grid_alone() -> None:
    scs = default_scenarios(None)
    assert len(scs) == len(DEFAULT_HYPOTHETICAL_SCENARIOS)
    assert all(sc.kind in (KIND_HYPOTHETICAL, KIND_IV_GRID) for sc in scs)


def test_scenario_params_are_plain_scalars_for_persistence() -> None:
    p = DEFAULT_HYPOTHETICAL_SCENARIOS[0].params()
    assert set(p) == {
        "spot_shock",
        "spot_shock_by_ticker",
        "iv_shock",
        "iv_shock_source",
        "days_forward",
        "uniform_beta_1",
        "source",
        "validated",
        "notes",
    }


@pytest.mark.parametrize(
    "kw",
    [
        {"name": ""},
        {"kind": "NOPE"},
        {"spot_shock": -1.0},
        {"spot_shock": math.nan},
        {"iv_shock": -1.5},
        {"days_forward": -1.0},
        {"spot_shock_by_ticker": {"A": -1.0}},
        {"spot_shock_by_ticker": {"A": math.inf}},
    ],
)
def test_scenario_rejects_malformed_input(kw: dict) -> None:
    with pytest.raises(ValueError):
        flat(**kw)


# ---------------------------------------------------------------------------
# run_stress
# ---------------------------------------------------------------------------


def test_zero_scenario_gives_exactly_zero_pnl() -> None:
    res = run_stress([], [call(quantity=3)], [flat()], nav=100_000.0)
    assert res.rows[0].pnl_usd == 0.0
    assert res.rows[0].pnl_pct_nav == 0.0
    assert res.worst is res.rows[0]
    assert res.health is ModelHealth.ACTIVE


def test_no_scenarios_is_unavailable_not_an_active_zero() -> None:
    res = run_stress([], [call()], [], nav=100_000.0)
    assert res.rows == ()
    assert res.worst is None
    assert res.min_pnl_usd is None
    assert res.health is ModelHealth.UNAVAILABLE
    assert res.reason == "no scenarios supplied"
    assert res.worst_loss_usd is None


def test_worst_row_is_the_smallest_pnl() -> None:
    scs = [
        flat("mild", spot_shock=-0.02),
        flat("severe", spot_shock=-0.20),
        flat("rally", spot_shock=0.10),
    ]
    res = run_stress(
        [StockLeg(key="S#1", ticker="AAA", quantity=100, spot0=100.0)],
        [],
        scs,
        nav=100_000.0,
    )
    assert res.worst.name == "severe"
    assert res.min_pnl_usd == pytest.approx(-2000.0)
    assert res.worst_loss_usd == pytest.approx(2000.0)  # a LOSS is positive
    assert res.worst_loss_pct_nav == pytest.approx(0.02)


def test_pct_nav_is_a_fraction_and_none_without_nav() -> None:
    stock = [StockLeg(key="S#1", ticker="AAA", quantity=100, spot0=100.0)]
    with_nav = run_stress(stock, [], [flat(spot_shock=-0.10)], nav=50_000.0)
    assert with_nav.rows[0].pnl_pct_nav == pytest.approx(-1000.0 / 50_000.0)
    assert with_nav.rows[0].pnl_pct_nav == pytest.approx(-0.02)
    no_nav = run_stress(stock, [], [flat(spot_shock=-0.10)])
    assert no_nav.rows[0].pnl_pct_nav is None
    assert no_nav.rows[0].pnl_usd == pytest.approx(-1000.0)


def test_unavailable_scenario_becomes_an_unavailable_row() -> None:
    w = HistoricalWindow("2019", date(2019, 3, 1), date(2019, 3, 8))
    bad = historical_shocks_from_closes(w, {"AAA": AAA})
    res = run_stress([], [call()], [bad, flat("ok", spot_shock=-0.05)], nav=1e5)
    assert res.rows[0].pnl_usd is None
    assert res.rows[0].health is ModelHealth.UNAVAILABLE
    assert "no stored history" in res.rows[0].reason
    assert res.rows[0].loss_usd is None
    # The available row still produces the worst.
    assert res.worst.name == "ok"
    # Run-level health is the worst health among the scenarios that PRICED
    # (revised 2026-08-18): a named window outside stored history is an
    # UNAVAILABLE ROW named in ``reason``, not a downgrade of every priced
    # scenario.
    assert res.health is ModelHealth.ACTIVE
    assert "1 of 2 scenarios unavailable" in res.reason


def test_all_scenarios_unavailable_reports_it() -> None:
    w = HistoricalWindow("2019", date(2019, 3, 1), date(2019, 3, 8))
    bad = historical_shocks_from_closes(w, {"AAA": AAA})
    res = run_stress([], [call()], [bad], nav=1e5)
    assert res.worst is None
    assert res.health is ModelHealth.UNAVAILABLE
    assert "no scenario produced a number" in res.reason


def test_delta_linear_fallback_degrades_the_row_health() -> None:
    res = run_stress(
        [], [call(iv0=None, delta0=0.5)], [flat(spot_shock=-0.10)], nav=1e5
    )
    row = res.rows[0]
    assert row.health is ModelHealth.DEGRADED
    assert row.method_coverage == {METHOD_FULL_REVAL: 0, METHOD_DELTA_LINEAR: 1}
    assert "DELTA_LINEAR" in row.reason
    assert res.health is ModelHealth.DEGRADED


def test_method_coverage_is_reported_per_row() -> None:
    legs = [call(key="A#1"), call(key="A#2", iv0=None, delta0=0.4)]
    res = run_stress([], legs, [flat(spot_shock=-0.05)], nav=1e5)
    assert res.rows[0].method_coverage == {
        METHOD_FULL_REVAL: 1,
        METHOD_DELTA_LINEAR: 1,
    }


def test_per_key_pnl_is_carried_on_the_row() -> None:
    stock = [StockLeg(key="S#1", ticker="AAA", quantity=10, spot0=100.0)]
    res = run_stress(stock, [call(key="O#1")], [flat(spot_shock=-0.10)], nav=1e5)
    row = res.rows[0]
    assert set(row.per_key) == {"S#1", "O#1"}
    assert row.per_key["S#1"] == pytest.approx(-100.0)
    assert math.fsum(row.per_key.values()) == pytest.approx(row.pnl_usd)


def test_row_carries_the_scenario_params_for_reproducibility() -> None:
    sc = flat("x", spot_shock=-0.07, iv_shock=0.3, days_forward=2)
    res = run_stress([], [call()], [sc], nav=1e5)
    p = res.rows[0].params
    assert p["spot_shock"] == -0.07
    assert p["iv_shock"] == 0.3
    assert p["days_forward"] == 2


def test_stress_result_carries_the_catalogue_version() -> None:
    res = run_stress([], [call()], [flat()], nav=1e5)
    assert res.catalogue_version == CATALOGUE_VERSION
    assert res.model_version == "1.0.0"


def test_run_scenario_reports_a_failed_row_rather_than_raising() -> None:
    """A malformed book is a FAILED row carrying the real reason — an
    exception must never escape into the caller's snapshot build and take
    the whole risk view down with it."""
    dup_stock = StockLeg(key="D#1", ticker="AAA", quantity=1, spot0=100.0)
    dup_option = call(key="D#1")
    res = run_scenario(
        [dup_stock], [dup_option], Scenario(name="dup", kind=KIND_USER), nav=1e5
    )
    assert res.health is ModelHealth.FAILED
    assert res.pnl_usd is None
    assert res.pnl_pct_nav is None
    assert "revaluation failed" in res.reason
    assert "duplicate leg key" in res.reason
    assert res.name == "dup"
    assert res.kind == KIND_USER


def test_a_malformed_book_fails_every_row_and_says_so() -> None:
    """A duplicate leg key is a defect in the BOOK, not in one scenario, so
    every row fails — and the run reports that honestly instead of
    returning a confident 0.0 for a book it could not price."""
    dup_stock = StockLeg(key="D#1", ticker="AAA", quantity=1, spot0=100.0)
    dup_option = call(key="D#1")
    res = run_stress(
        [dup_stock],
        [dup_option],
        [Scenario(name="dup", kind=KIND_USER), flat("ok", spot_shock=-0.05)],
        nav=1e5,
    )
    assert all(r.health is ModelHealth.FAILED for r in res.rows)
    assert res.worst is None
    assert res.min_pnl_usd is None
    assert res.health is ModelHealth.UNAVAILABLE
    assert "no scenario produced a number" in res.reason


def test_an_unavailable_scenario_alongside_a_good_one_still_yields_a_worst() -> None:
    """Contrast with the malformed-book case: when only the SCENARIO is
    unavailable, the priceable rows still produce a worst-case number."""
    w = HistoricalWindow("2019", date(2019, 3, 1), date(2019, 3, 8))
    bad = historical_shocks_from_closes(w, {"AAA": AAA})
    res = run_stress(
        [StockLeg(key="S#1", ticker="AAA", quantity=100, spot0=100.0)],
        [],
        [bad, flat("ok", spot_shock=-0.05)],
        nav=1e5,
    )
    assert res.rows[0].pnl_usd is None
    assert res.rows[1].pnl_usd == pytest.approx(-500.0)
    assert res.worst.name == "ok"
    assert "1 of 2 scenarios unavailable" in res.reason


def test_income_and_spread_signs_survive_the_stress_run() -> None:
    """Short premium gains in an IV crush; long premium loses."""
    long_leg = call(key="L#1", quantity=1)
    short_leg = call(key="S#1", quantity=-1)
    res = run_stress([], [long_leg, short_leg], [flat(iv_shock=-0.40)], nav=1e5)
    assert res.rows[0].per_key["L#1"] < 0.0
    assert res.rows[0].per_key["S#1"] > 0.0
    assert res.rows[0].pnl_usd == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Historical scenarios end-to-end
# ---------------------------------------------------------------------------


def test_historical_scenario_runs_on_a_stock_book_with_the_exact_shock() -> None:
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    stock = [StockLeg(key="S#1", ticker="AAA", quantity=100, spot0=105.0)]
    res = run_stress(stock, [], [sc], nav=100_000.0)
    expected = 100 * 105.0 * (90.0 / 105.0 - 1.0)
    assert res.rows[0].pnl_usd == pytest.approx(expected, abs=1e-9)
    assert res.rows[0].pnl_usd == pytest.approx(-1500.0, abs=1e-9)


def test_a_ticker_absent_from_the_window_keeps_the_uniform_shock_of_zero() -> None:
    """A leg on a ticker the historical scenario has no shock for is moved
    by the scenario's uniform shock (0.0 for a historical row) — never by
    another ticker's number."""
    w = HistoricalWindow("crash", date(2026, 1, 9), date(2026, 1, 13))
    sc = historical_shocks_from_closes(w, {"AAA": AAA})
    stock = [StockLeg(key="Z#1", ticker="ZZZ", quantity=100, spot0=10.0)]
    res = run_stress(stock, [], [sc], nav=1e5)
    assert res.rows[0].per_key["Z#1"] == 0.0


# ---------------------------------------------------------------------------
# StressLimits
# ---------------------------------------------------------------------------


def test_stress_limits_defaults_are_shadow_and_documented() -> None:
    lim = StressLimits()
    assert lim.max_stress_loss_pct_nav == 0.10
    assert lim.mode == "SHADOW"
    assert lim.is_shadow is True


@pytest.mark.parametrize(
    "kw",
    [
        {"max_stress_loss_pct_nav": 0.0},
        {"max_stress_loss_pct_nav": -0.1},
        {"max_stress_loss_pct_nav": math.inf},
        {"mode": "PROD"},
    ],
)
def test_stress_limits_reject_malformed_input(kw: dict) -> None:
    with pytest.raises(ValueError):
        StressLimits(**kw)


# ---------------------------------------------------------------------------
# stress_caps — bisection vs brute force
# ---------------------------------------------------------------------------

NAV = 100_000.0
DOWN_20 = [flat("down 20%", spot_shock=-0.20)]


def _brute_force_cap(
    candidate_stock, candidate_option, book_stock, book_option, scenarios, requested, nav, limits
) -> int:
    """The answer by exhaustive search — the oracle the bisection must match
    on a monotone limit."""
    budget = limits.max_stress_loss_pct_nav * nav
    best = 0
    for q in range(0, requested + 1):
        res = run_stress(
            [*book_stock, *(l.scaled(q) for l in candidate_stock)],
            [*book_option, *(l.scaled(q) for l in candidate_option)],
            scenarios,
            nav=nav,
        )
        loss = res.worst_loss_usd
        if loss is not None and loss <= budget:
            best = q
    return best


def test_stress_cap_matches_brute_force_on_a_monotone_limit() -> None:
    """One long call per unit; a -20 % move loses ~the whole premium.

    Budget = 10 % of 100 000 = $10 000. Each contract loses roughly
    100 * (mark0 - price_after) with price_after ~ small, so the cap lands
    where the brute-force search lands.
    """
    cand = [call(key="CAND", quantity=1)]
    caps, health, reason = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=50, nav=NAV
    )
    expected = _brute_force_cap([], cand, [], [], DOWN_20, 50, NAV, StressLimits())
    assert len(caps) == 1
    assert caps[0].cap_qty == expected
    assert 0 < caps[0].cap_qty < 50  # the cap actually binds


def test_stress_cap_shape_code_and_layer() -> None:
    cand = [call(key="CAND", quantity=1)]
    caps, health, reason = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=50, nav=NAV
    )
    cap = caps[0]
    assert cap.code == CODE_STRESS_LOSS
    assert cap.layer == LAYER_STRESS
    assert "SHADOW" in cap.sentence
    assert "down 20%" in cap.sentence
    assert "% of NAV limit" in cap.sentence
    assert cap.measured["limit_pct_nav"] == 0.10
    assert cap.measured["budget_usd"] == pytest.approx(10_000.0)
    assert cap.measured["requested_qty"] == 50.0
    assert cap.measured["cap_qty"] == float(cap.cap_qty)
    assert cap.measured["worst_loss_usd_at_requested"] > 10_000.0
    assert cap.measured["worst_loss_usd_at_cap"] <= 10_000.0


def test_the_capped_quantity_actually_satisfies_the_limit() -> None:
    """The invariant that matters: a cap never hands back a breaching qty."""
    cand = [call(key="CAND", quantity=1)]
    limits = StressLimits(max_stress_loss_pct_nav=0.03)
    caps, _, _ = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=100, nav=NAV, limits=limits
    )
    q = caps[0].cap_qty
    at_cap = run_stress([], [cand[0].scaled(q)], DOWN_20, nav=NAV)
    assert at_cap.worst_loss_usd <= 0.03 * NAV
    # ...and one more unit would breach it.
    at_next = run_stress([], [cand[0].scaled(q + 1)], DOWN_20, nav=NAV)
    assert at_next.worst_loss_usd > 0.03 * NAV


def test_no_cap_when_the_limit_is_already_satisfied() -> None:
    cand = [call(key="CAND", quantity=1)]
    caps, health, reason = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=2, nav=NAV
    )
    assert caps == []
    assert health is ModelHealth.ACTIVE


def test_cap_accounts_for_the_existing_book() -> None:
    """A book already near the budget leaves less room for the candidate."""
    cand = [call(key="CAND", quantity=1)]
    empty_caps, _, _ = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=50, nav=NAV
    )
    heavy_book = [call(key="BOOK", quantity=8)]
    loaded_caps, _, _ = stress_caps(
        [], cand, [], heavy_book, DOWN_20, requested_qty=50, nav=NAV
    )
    assert loaded_caps[0].cap_qty < empty_caps[0].cap_qty


def test_cap_can_be_zero_when_even_one_unit_breaches() -> None:
    cand = [call(key="CAND", quantity=1)]
    limits = StressLimits(max_stress_loss_pct_nav=1e-6)  # $0.10 budget
    caps, _, _ = stress_caps(
        [], cand, [], [], DOWN_20, requested_qty=10, nav=NAV, limits=limits
    )
    assert caps[0].cap_qty == 0
    assert "the whole trade would breach it" in caps[0].sentence


def test_stock_candidate_caps_too() -> None:
    cand = [StockLeg(key="CAND", ticker="AAA", quantity=100, spot0=100.0)]
    caps, _, _ = stress_caps(
        cand, [], [], [], DOWN_20, requested_qty=20, nav=NAV
    )
    # each unit = 100 shares * $100 * -20 % = -$2 000; budget $10 000 -> 5
    assert caps[0].cap_qty == 5


# --- fail-open: a missing view NEVER produces a cap (SHADOW) --------------


def test_no_scenarios_produces_no_cap() -> None:
    caps, health, reason = stress_caps(
        [], [call(key="C")], [], [], [], requested_qty=10, nav=NAV
    )
    assert caps == []
    assert health is ModelHealth.UNAVAILABLE
    assert reason == "no scenarios supplied"


def test_non_positive_nav_produces_no_cap() -> None:
    for nav in (0.0, -5.0):
        caps, health, reason = stress_caps(
            [], [call(key="C")], [], [], DOWN_20, requested_qty=10, nav=nav
        )
        assert caps == []
        assert health is ModelHealth.UNAVAILABLE
        assert "not > 0" in reason


def test_zero_requested_quantity_produces_no_cap() -> None:
    caps, health, reason = stress_caps(
        [], [call(key="C")], [], [], DOWN_20, requested_qty=0, nav=NAV
    )
    assert caps == []
    assert health is ModelHealth.UNAVAILABLE


def test_all_unavailable_scenarios_produce_no_cap() -> None:
    w = HistoricalWindow("2019", date(2019, 3, 1), date(2019, 3, 8))
    bad = [historical_shocks_from_closes(w, {"AAA": AAA})]
    caps, health, reason = stress_caps(
        [], [call(key="C")], [], [], bad, requested_qty=10, nav=NAV
    )
    assert caps == []
    assert health is ModelHealth.UNAVAILABLE


def test_stress_caps_reject_a_negative_requested_quantity() -> None:
    with pytest.raises(ValueError):
        stress_caps([], [], [], [], DOWN_20, requested_qty=-1, nav=NAV)


def test_stress_cap_feeds_the_phase_c_shadow_verdict() -> None:
    """The STRESS cap must merge into the SAME verdict shape Phase C emits
    — that is why QuantityCap accepts the STRESS layer."""
    from libs.trading_core.risk.pretrade import shadow_verdict

    cand = [call(key="CAND", quantity=1)]
    caps, _, _ = stress_caps([], cand, [], [], DOWN_20, requested_qty=50, nav=NAV)
    verdict = shadow_verdict(50, caps)
    assert verdict.hypothetical_decision == "APPROVE_WITH_RESIZE"
    assert verdict.hypothetical_quantity == caps[0].cap_qty
    assert verdict.binding == (CODE_STRESS_LOSS,)
    assert verdict.mode == "SHADOW"


# ---------------------------------------------------------------------------
# Monotonicity: |P&L| non-decreasing in |q| (spec §67)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        flat("down", spot_shock=-0.15),
        flat("up", spot_shock=0.15),
        flat("crush", iv_shock=-0.40),
        flat("spike", iv_shock=0.50),
        flat("decay", days_forward=30),
        flat("mixed", spot_shock=-0.10, iv_shock=0.4, days_forward=5),
    ],
)
def test_abs_pnl_is_monotone_non_decreasing_in_abs_quantity(sc: Scenario) -> None:
    """Spec §67 property, for a single leg, both signs of quantity."""
    base = call(key="M#1", quantity=1)
    prev = -1.0
    for q in (0, 1, 2, 3, 5, 8, 13, 21):
        res = run_stress([], [base.scaled(q)], [sc], nav=NAV)
        cur = abs(res.rows[0].pnl_usd)
        assert cur >= prev - 1e-9, f"q={q}: |pnl| {cur} < previous {prev}"
        prev = cur


def test_abs_pnl_monotone_for_a_short_leg_too() -> None:
    base = call(key="M#1", quantity=-1)
    sc = flat("down", spot_shock=-0.15)
    prev = -1.0
    for q in (0, 1, 4, 9, 16):
        res = run_stress([], [base.scaled(q)], [sc], nav=NAV)
        cur = abs(res.rows[0].pnl_usd)
        assert cur >= prev - 1e-9
        prev = cur


def test_abs_pnl_monotone_for_stock() -> None:
    base = StockLeg(key="M#1", ticker="AAA", quantity=-10, spot0=100.0)
    sc = flat("down", spot_shock=-0.15)
    prev = -1.0
    for q in (0, 1, 7, 40):
        res = run_stress([base.scaled(q)], [], [sc], nav=NAV)
        cur = abs(res.rows[0].pnl_usd)
        assert cur >= prev - 1e-9
        prev = cur
