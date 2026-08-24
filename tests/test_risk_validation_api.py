"""Phase E on the wire — VaR/ES model validation (design §9.4).

Four surfaces, all SHADOW/RESEARCH:

- ``apps/gateway/risk_validation.py`` — the walk-forward runner: rows scored
  and persisted, honest UNAVAILABLE below ``MIN_FORECASTS``, the §63
  EWMA-vs-GARCH comparison, and a measured runtime bound;
- ``POST /api/risk/validation/run`` — on demand: 200, rows persisted, NO
  audit event, 422 on an out-of-range window;
- ``GET /api/portfolio/risk`` — ``statistical.validation`` read from the
  NEWEST PERSISTED rows, null before any run, never recomputed on the read;
- the SCHEDULED tick — one validation run per America/New_York day, in the
  snapshot's own transaction.

The house rule this file exists to defend: nothing here may change a Tier 0
decision, no forecast may see the day it forecasts (spec §43), and every gap
is an honest null with a reason — never a fabricated GREEN.
"""
import math
import random
import time
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from apps.gateway import risk_snapshot as rs
from apps.gateway import risk_validation as rv
from apps.gateway.db import AuditEvent, RiskModelBacktestRow, SessionLocal
from libs.trading_core.risk.models.ensemble import EnsembleParams, model_risk_state

from .test_risk_snapshot_builder import build, seed_stock_position

#: Every key the design §9.4 ``statistical.validation`` block carries.
VALIDATION_KEYS = {
    "mode",
    "as_of",
    "window",
    "min_forecasts",
    "n_obs",
    "rows",
    "comparison",
}

#: Every key on one validation row.
ROW_KEYS = {
    "model_name",
    "model_version",
    "distribution",
    "confidence",
    "horizon_days",
    "window",
    "n_forecasts",
    "exceedances",
    "rate",
    "expected_rate",
    "kupiec_p",
    "christoffersen_p",
    "es_severity_ratio",
    "verdict",
    "health",
    "reason",
    "mode",
    # ADDED with the §5 tier taxonomy: a validation row carries the tier of
    # the model it VALIDATES (derived from `model_name`, never stored).
    "tier",
}

#: Every key on the §63 comparison dict.
COMPARISON_KEYS = {
    "ewma_kupiec_p",
    "garch_kupiec_p",
    "garch_christoffersen_p",
    "garch_n_forecasts",
    "preferred",
    "criterion",
    "criterion_met",
    "criterion_unmet_reasons",
    "promotion",
}

#: The grid order the runner produces (design §9.4).
GRID = (
    ("historical_var", 0.95),
    ("historical_var", 0.99),
    ("gaussian_var", 0.95),
    ("gaussian_var", 0.99),
    ("conditional_var", 0.95),
    ("garch_var", 0.95),
)

VERDICTS = {"GREEN", "YELLOW", "RED", "UNAVAILABLE"}


# ---------------------------------------------------------------------------
# Deterministic seeded book P&L
# ---------------------------------------------------------------------------


def seeded_pnl(n: int, *, seed: int = 20260818) -> list[float]:
    """A conditionally-heteroskedastic daily P&L path, seeded and reproducible.

    Volatility clusters (the σ process mean-reverts around $800/day with
    persistent shocks), which is what makes the conditional views and the
    GARCH fit meaningful rather than a coin flip on white noise.
    """
    rng = random.Random(seed)
    sigma = 800.0
    out: list[float] = []
    for _ in range(n):
        sigma = 0.90 * sigma + 0.10 * 800.0 + rng.gauss(0.0, 60.0)
        sigma = max(sigma, 50.0)
        out.append(rng.gauss(0.0, sigma))
    return out


def seeded_dates(n: int) -> list[date]:
    start = date(2024, 1, 2)
    return [start + timedelta(days=i) for i in range(n)]


async def _rows() -> list[RiskModelBacktestRow]:
    async with SessionLocal() as session:
        return list(
            (
                await session.execute(
                    select(RiskModelBacktestRow).order_by(RiskModelBacktestRow.id)
                )
            ).scalars().all()
        )


async def _audit_events() -> list[AuditEvent]:
    async with SessionLocal() as session:
        return list((await session.execute(select(AuditEvent))).scalars().all())


# ---------------------------------------------------------------------------
# (a) The runner on a seeded 400-bar book
# ---------------------------------------------------------------------------


def test_seeded_book_scores_every_view_with_sane_rates():
    """400 observations, window 250 ⇒ 150 forecasts per view: enough to clear
    MIN_FORECASTS, so every row carries real coverage statistics."""
    run = rv.compute_model_backtests(seeded_pnl(400), dates=seeded_dates(400))

    assert [(r.model_name, r.confidence) for r in run.rows] == list(GRID)
    assert run.n_obs == 400
    assert run.window == rv.DEFAULT_WINDOW

    for row in run.rows:
        assert row.horizon_days == 1
        assert row.window_obs == 250
        assert row.verdict in VERDICTS
        assert row.n_forecasts >= rv.MIN_FORECASTS, (row.model_name, row.reason)
        assert row.n_forecasts == 150  # 400 - 250, none skipped
        # A rate is a FRACTION and a count is a count — no fabricated values.
        assert row.rate is not None and 0.0 <= row.rate <= 1.0
        assert row.exceedances == round(row.rate * row.n_forecasts)
        assert math.isclose(row.expected_rate, 1.0 - row.confidence)
        assert row.kupiec_p is not None and 0.0 <= row.kupiec_p <= 1.0
        assert row.christoffersen_p is not None and 0.0 <= row.christoffersen_p <= 1.0
        assert row.kupiec_lr is not None and row.kupiec_lr >= 0.0
        # ES severity comes from the MATCHING ES estimator, so it exists
        # whenever there was at least one exceedance to measure it on.
        if row.exceedances:
            assert row.es_severity_ratio is not None and row.es_severity_ratio > 0.0
        # Methodology travels with every number (spec §44/§50).
        assert row.model_version and row.distribution
        assert row.params["walk_forward"] is True
        assert row.params["window"] == 250

    # A well-behaved series should not be systematically mis-calibrated: the
    # empirical rate lands in the neighbourhood of the promised one.
    for row in run.rows:
        assert abs(row.rate - row.expected_rate) < 0.06, (row.model_name, row.rate)


def test_mode_labels_place_garch_below_shadow():
    """Spec §70: GARCH is RESEARCH, strictly below the SHADOW views."""
    run = rv.compute_model_backtests(seeded_pnl(400))
    modes = {r.model_name: r.mode for r in run.rows}
    assert modes["garch_var"] == "RESEARCH"
    assert modes["historical_var"] == "SHADOW"
    assert modes["gaussian_var"] == "SHADOW"
    assert modes["conditional_var"] == "SHADOW"


def test_distributions_name_the_actual_filter():
    run = rv.compute_model_backtests(seeded_pnl(400))
    by = {(r.model_name, r.confidence): r for r in run.rows}
    assert by[("historical_var", 0.95)].distribution == "EMPIRICAL"
    assert by[("gaussian_var", 0.95)].distribution == "NORMAL"
    assert by[("conditional_var", 0.95)].distribution == "EMPIRICAL_VOL_SCALED"
    assert by[("garch_var", 0.95)].distribution == "EMPIRICAL_GARCH_SCALED"


# ---------------------------------------------------------------------------
# (b) Walk-forward sentinel (spec §43; contract §3 invariant 5)
# ---------------------------------------------------------------------------


def test_a_spike_on_the_last_day_cannot_change_earlier_forecasts():
    """The sentinel that proves no look-ahead — including INSIDE the filters.

    Two runs on the same series, one with a catastrophic loss substituted on
    the FINAL day. Only the final day's realized P&L differs, and no forecast
    is made FOR a day after it, so every earlier forecast — and therefore the
    exceedance count on all but that last day — must be untouched.

    This is the test that would fail if a conditional view were computed by
    filtering the whole series once and slicing it: the end-of-sample σ would
    leak backwards into every window.
    """
    base = seeded_pnl(400)
    spiked = list(base)
    spiked[-1] = -1_000_000.0  # a loss no forecast could have anticipated

    run_a = rv.compute_model_backtests(base)
    run_b = rv.compute_model_backtests(spiked)

    for row_a, row_b in zip(run_a.rows, run_b.rows):
        assert (row_a.model_name, row_a.confidence) == (row_b.model_name, row_b.confidence)
        assert row_a.n_forecasts == row_b.n_forecasts
        # The spike is itself an exceedance on the last day (and only there),
        # so the count moves by exactly one — never by more, which is what a
        # contaminated earlier forecast would produce.
        assert row_b.exceedances - row_a.exceedances == 1, row_a.model_name


def test_walk_forward_forecasts_are_identical_up_to_the_changed_day():
    """The same sentinel, checked forecast-by-forecast on the core view.

    Substituting the last observation must leave every forecast identical
    (the last day is realized, never forecast-from).
    """
    from libs.trading_core.risk.models.var_es import historical_var
    from libs.trading_core.risk.validation import walk_forward

    base = seeded_pnl(400)
    spiked = list(base)
    spiked[-1] = -1_000_000.0

    def est(w):
        return historical_var(w, 0.95, 1, min_obs=250)

    a = walk_forward(base, window=250, estimator=est, confidence=0.95)
    b = walk_forward(spiked, window=250, estimator=est, confidence=0.95)
    assert a.forecasts == b.forecasts
    assert a.realized[:-1] == b.realized[:-1]
    assert a.realized[-1] != b.realized[-1]


# ---------------------------------------------------------------------------
# (c) Honest nulls below MIN_FORECASTS
# ---------------------------------------------------------------------------


def test_too_few_forecasts_produce_unavailable_rows_with_the_real_numbers():
    """300 observations at window 250 ⇒ 50 forecasts < MIN_FORECASTS=60.

    Every row is still PRODUCED (a missing row would read as "never run"),
    with UNAVAILABLE, null statistics and a reason quoting both numbers.
    """
    run = rv.compute_model_backtests(seeded_pnl(300))
    assert len(run.rows) == len(GRID)
    for row in run.rows:
        assert row.verdict == "UNAVAILABLE", row.model_name
        assert row.health == "UNAVAILABLE"
        assert row.reason, row.model_name
        assert row.kupiec_p is None and row.kupiec_lr is None
        assert row.christoffersen_p is None
        # The counts are still real (they are counts, not inferences).
        assert row.n_forecasts <= 50
        # The reason carries the real numbers, never a fixed sentence.
        assert str(rv.MIN_FORECASTS) in row.reason or "window" in row.reason


def test_empty_book_never_raises_and_never_fabricates():
    run = rv.compute_model_backtests([])
    assert len(run.rows) == len(GRID)
    for row in run.rows:
        assert row.verdict == "UNAVAILABLE"
        assert row.n_forecasts == 0
        assert row.rate is None
        assert row.reason


def test_garch_row_is_skipped_with_a_reason_when_no_window_can_be_fitted():
    """Design §9.4: a GARCH view whose fits do not produce a parameterisation
    yields an UNAVAILABLE row with the real numbers — never a silently-missing
    row, and never a forecast from parameters the fit refused to bless.

    A constant series has no variation, so ``fit_garch`` is UNAVAILABLE on
    every window: every day is SKIPPED and the row says so.
    """
    run = rv.compute_model_backtests([0.0] * 400)
    garch = next(r for r in run.rows if r.model_name == "garch_var")
    assert garch.verdict == "UNAVAILABLE"
    assert garch.health == "UNAVAILABLE"
    assert garch.n_forecasts == 0
    assert "skipped=150" in garch.reason      # every day, with the real count
    assert garch.params["n_garch_windows_unfittable"] > 0
    assert "no variation" in garch.params["garch_last_fit_reason"]
    # The stride still bounded the cost: it does NOT refit on every failure.
    assert garch.params["n_garch_fits"] < 40


def test_a_window_shorter_than_the_garch_minimum_is_an_honest_unavailable():
    """The EWMA view stays the conditional forecaster (spec §13/§58 fallback);
    the GARCH row says why it could not run rather than going missing."""
    run = rv.compute_model_backtests(seeded_pnl(519), window=120)
    by = {r.model_name: r for r in run.rows}
    assert by["conditional_var"].verdict in {"GREEN", "YELLOW", "RED"}
    assert by["conditional_var"].n_forecasts == 399
    garch = by["garch_var"]
    assert garch.verdict == "UNAVAILABLE"
    assert f"window=120 < garch_min_obs={rv.GARCH_MIN_OBS}" in garch.reason
    assert "fallback" in garch.reason


# ---------------------------------------------------------------------------
# (d) §63 comparison
# ---------------------------------------------------------------------------


def test_comparison_carries_the_criterion_sentence_verbatim():
    run = rv.compute_model_backtests(seeded_pnl(400))
    comp = run.comparison

    assert set(comp) == COMPARISON_KEYS
    assert comp["criterion"] == rv.COMPARISON_CRITERION
    assert "user action" in comp["criterion"]
    assert comp["preferred"] in {"conditional_var", "garch_var", None}
    assert comp["promotion"].startswith("NONE")
    # 150 forecasts is short of the 250 the criterion demands, so the bar is
    # NOT met and the reason says exactly why.
    assert comp["criterion_met"] is False
    assert any("250" in r for r in comp["criterion_unmet_reasons"])


def test_comparison_preference_is_none_when_a_side_is_missing():
    """A comparison with a missing half is not a preference (honest null)."""
    comp = rv.ewma_vs_garch(
        [r for r in rv.compute_model_backtests(seeded_pnl(300)).rows]
    )
    assert comp["ewma_kupiec_p"] is None
    assert comp["garch_kupiec_p"] is None
    assert comp["preferred"] is None
    assert comp["criterion_met"] is False


# ---------------------------------------------------------------------------
# (e) Runtime bound (design §9.4: "keep runtime bounded — measure")
# ---------------------------------------------------------------------------


def test_runner_on_600_observations_stays_under_three_seconds():
    """The GARCH refit stride is what makes this hold: a per-day MLE over 350
    forecast days would be hundreds of Nelder-Mead runs."""
    pnl = seeded_pnl(600)
    started = time.perf_counter()
    run = rv.compute_model_backtests(pnl)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"validation runner took {elapsed:.2f}s on 600 observations"
    # The run measures itself, and the two agree.
    assert run.seconds <= elapsed + 0.01
    # And the stride actually bounded the fit count: 350 forecast days at a
    # stride of 20, twice (VaR + ES passes), plus the probe fit.
    garch = next(r for r in run.rows if r.model_name == "garch_var")
    assert garch.params["refit_every"] == rv.GARCH_REFIT_EVERY
    assert garch.params["n_garch_fits"] <= 2 * (350 // rv.GARCH_REFIT_EVERY + 2) + 1


# ---------------------------------------------------------------------------
# (f) Persistence
# ---------------------------------------------------------------------------


async def test_run_model_backtests_persists_one_row_per_view(client):
    """Rows are flushed, not committed — the CALLER commits (the
    ``risk_snapshot._persist`` pattern)."""
    async with SessionLocal() as session:
        run = await rv.run_model_backtests(
            session,
            book_pnl=seeded_pnl(400),
            dates=seeded_dates(400),
            nav=100_000.0,
            snapshot_id=None,
        )
        await session.commit()

    rows = await _rows()
    assert len(rows) == len(GRID)
    assert [(r.model_name, r.confidence) for r in rows] == list(GRID)
    for row, result in zip(rows, run.rows):
        assert row.verdict == result.verdict
        assert row.n_forecasts == result.n_forecasts
        assert row.window_obs == 250
        assert row.horizon_days == 1
        assert row.snapshot_id is None       # on-demand: no snapshot behind it
        assert row.params["nav"] == 100_000.0
        assert row.params["n_obs"] == 400
        assert row.as_of is not None
    # The comparison rides on the two conditional rows only.
    with_comparison = [r for r in rows if r.params.get("comparison")]
    assert {r.model_name for r in with_comparison} == {"conditional_var", "garch_var"}


async def test_latest_backtest_rows_returns_only_the_newest_run(client):
    older = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    newer = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    async with SessionLocal() as session:
        await rv.run_model_backtests(
            session, book_pnl=seeded_pnl(400), as_of=older
        )
        await rv.run_model_backtests(
            session, book_pnl=seeded_pnl(400, seed=99), as_of=newer
        )
        await session.commit()

    async with SessionLocal() as session:
        rows = await rv.latest_backtest_rows(session)
    assert len(rows) == len(GRID)
    # SQLite hands the column back NAIVE; the repo convention is that a naive
    # stored instant IS UTC (rv._as_utc, mirroring risk_snapshot).
    assert {rv._as_utc(r.as_of) for r in rows} == {newer}


async def test_latest_backtest_rows_is_empty_before_any_run(client):
    async with SessionLocal() as session:
        assert await rv.latest_backtest_rows(session) == []
        assert await rv.validation_api(session) is None


# ---------------------------------------------------------------------------
# (g) POST /api/risk/validation/run
# ---------------------------------------------------------------------------


async def test_validation_endpoint_runs_persists_and_writes_no_audit(client):
    """200 with the rows, rows persisted, and NO audit event: a measurement
    of past forecasts is not a decision (house rule)."""
    await seed_stock_position(bars=520)
    before = len(await _audit_events())

    r = await client.post("/api/risk/validation/run")
    assert r.status_code == 200, r.text
    body = r.json()

    assert VALIDATION_KEYS <= set(body)
    assert body["mode"] == "SHADOW"
    assert body["window"] == rv.DEFAULT_WINDOW
    assert body["min_forecasts"] == rv.MIN_FORECASTS
    assert len(body["rows"]) == len(GRID)
    for row in body["rows"]:
        assert set(row) == ROW_KEYS
        assert row["verdict"] in VERDICTS
        if row["kupiec_p"] is not None:
            assert 0.0 <= row["kupiec_p"] <= 1.0
    assert set(body["comparison"]) == COMPARISON_KEYS
    assert body["note"].startswith("SHADOW/RESEARCH")
    assert body["seconds"] is not None

    rows = await _rows()
    assert len(rows) == len(GRID)
    assert all(row.snapshot_id is None for row in rows)
    assert all(row.params["trigger"] == "ON_DEMAND" for row in rows)

    # A read writes no audit event — the RISK_DECISION audit stays the one
    # record of the one decision.
    assert len(await _audit_events()) == before


async def test_validation_endpoint_accepts_an_explicit_window(client):
    await seed_stock_position(bars=520)
    r = await client.post("/api/risk/validation/run", json={"window": 120})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["window"] == 120
    assert all(row["window"] == 120 for row in body["rows"])
    # 519 returns - 120 window = 399 forecasts for every view that can run at
    # this window. GARCH cannot (it needs 250 observations) and says so — an
    # honest UNAVAILABLE, not a silently-missing row.
    scored = [r for r in body["rows"] if r["model_name"] != "garch_var"]
    assert len(scored) == len(GRID) - 1
    assert all(r["n_forecasts"] >= rv.MIN_FORECASTS for r in scored)
    garch = next(r for r in body["rows"] if r["model_name"] == "garch_var")
    assert garch["verdict"] == "UNAVAILABLE"
    assert f"garch_min_obs={rv.GARCH_MIN_OBS}" in garch["reason"]


@pytest.mark.parametrize("window", [1, 29, 501, 10_000])
async def test_validation_endpoint_422s_on_an_out_of_range_window(client, window):
    """Out of range is a 422 naming the field, never a clamped run: silently
    validating a different window than the one asked for would be the
    dishonest option."""
    r = await client.post("/api/risk/validation/run", json={"window": window})
    assert r.status_code == 422, r.text
    assert "window" in r.text


async def test_validation_endpoint_on_an_empty_book_is_honest_not_an_error(client):
    """No positions ⇒ no P&L history ⇒ UNAVAILABLE rows with reasons. Never a
    503, and never a fabricated verdict."""
    r = await client.post("/api/risk/validation/run")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["rows"]) == len(GRID)
    for row in body["rows"]:
        assert row["verdict"] == "UNAVAILABLE"
        assert row["reason"]
        assert row["rate"] is None


# ---------------------------------------------------------------------------
# (h) GET /api/portfolio/risk — statistical.validation
# ---------------------------------------------------------------------------


async def test_risk_view_validation_is_null_before_any_run(client):
    """An honest "never validated" — not an empty object that could be
    mistaken for a clean bill of health."""
    await seed_stock_position(bars=520)
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    assert r.json()["statistical"]["validation"] is None


async def test_risk_view_serves_validation_from_the_persisted_rows(client):
    await seed_stock_position(bars=520)
    posted = (await client.post("/api/risk/validation/run")).json()

    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    block = r.json()["statistical"]["validation"]

    assert block is not None
    assert set(block) == VALIDATION_KEYS
    assert block["mode"] == "SHADOW"
    assert block["window"] == rv.DEFAULT_WINDOW
    assert block["min_forecasts"] == rv.MIN_FORECASTS
    assert len(block["rows"]) == len(GRID)
    for row in block["rows"]:
        assert set(row) == ROW_KEYS
    # It is the SAME run, read back — not a fresh one computed on the read.
    assert block["as_of"] == posted["as_of"]
    assert [r_["verdict"] for r_ in block["rows"]] == [
        r_["verdict"] for r_ in posted["rows"]
    ]
    assert block["comparison"]["criterion"] == rv.COMPARISON_CRITERION


async def test_a_page_read_never_writes_a_backtest_row(client):
    """Design §9.4: "never recomputed on a page read". Proven by the ROW
    COUNT — a read that recomputed would either persist or diverge."""
    await seed_stock_position(bars=520)
    await client.post("/api/risk/validation/run")
    after_run = await _rows()

    for _ in range(3):
        assert (await client.get("/api/portfolio/risk")).status_code == 200

    assert len(await _rows()) == len(after_run)


# ---------------------------------------------------------------------------
# (i) The SCHEDULED tick writes once per NY day
# ---------------------------------------------------------------------------


async def test_scheduled_tick_runs_validation_once_per_ny_day(client):
    """One SCHEDULED snapshot per NY day, and the validation run rides with
    it in the SAME transaction — so the second tick of the day writes
    neither a snapshot nor a duplicate set of validation rows."""
    await seed_stock_position(bars=520)

    first = await rs.run_scheduled_snapshot()
    assert first.get("snapshot_id") is not None
    assert first["validation_rows"] == len(GRID)
    assert first["validation_seconds"] is not None

    rows = await _rows()
    assert len(rows) == len(GRID)
    # A SCHEDULED run is attached to the snapshot that triggered it.
    assert all(row.snapshot_id == first["snapshot_id"] for row in rows)
    assert all(row.params["trigger"] == "SCHEDULED" for row in rows)

    second = await rs.run_scheduled_snapshot()
    assert second == {"skipped": "ALREADY_BUILT_TODAY", "day": second["day"]}
    assert len(await _rows()) == len(GRID)


async def test_scheduled_validation_failure_never_fails_the_snapshot(client, monkeypatch):
    """A validation run is a measurement of PAST forecasts: if it explodes,
    the snapshot it accompanies is still a good snapshot."""
    await seed_stock_position(bars=520)

    async def boom(*args, **kwargs):
        raise RuntimeError("synthetic validation fault")

    monkeypatch.setattr(rv, "run_model_backtests", boom)
    result = await rs.run_scheduled_snapshot()

    assert result.get("snapshot_id") is not None
    assert result["validation_rows"] == 0
    assert await _rows() == []


# ---------------------------------------------------------------------------
# (j) backtest_red_triggers feeds model risk (additively)
# ---------------------------------------------------------------------------


def test_backtest_red_triggers_defaults_leave_model_risk_unchanged():
    """ADDITIVE: a caller that passes nothing gets exactly the old answer."""
    from libs.trading_core.risk.models.base import ModelHealth, ModelMeta, ModelResult

    meta = ModelMeta(
        model_name="historical_var", model_version="1.0.0", params={},
        return_type=None, frequency=None, lookback=None, data_source=None,
        as_of=None, confidence=0.95, horizon_days=1, distribution="EMPIRICAL",
    )
    views = {
        "historical_var_95": ModelResult(
            value=1000.0, health=ModelHealth.ACTIVE, reason=None,
            sample_size=500, meta=meta,
        )
    }
    state = model_risk_state(views, core_views=("historical_var_95",))
    assert state.state == "LOW"
    assert state.triggers["backtest_red"] is False
    assert state.diagnostics["backtest_red_count"] == 0


def test_one_red_core_verdict_is_one_model_risk_trigger():
    from libs.trading_core.risk.models.base import ModelHealth, ModelMeta, ModelResult

    meta = ModelMeta(
        model_name="historical_var", model_version="1.0.0", params={},
        return_type=None, frequency=None, lookback=None, data_source=None,
        as_of=None, confidence=0.95, horizon_days=1, distribution="EMPIRICAL",
    )
    views = {
        "historical_var_95": ModelResult(
            value=1000.0, health=ModelHealth.ACTIVE, reason=None,
            sample_size=500, meta=meta,
        )
    }
    state = model_risk_state(
        views,
        core_views=("historical_var_95",),
        backtest_red_count=1,
        backtest_red_reasons=("backtest RED on historical_var @ 0.99: ...",),
    )
    assert state.triggers["backtest_red"] is True
    assert state.state == "ELEVATED"          # exactly one trigger
    assert any("RED backtest verdict" in r for r in state.reasons)

    # The threshold is a PARAMETER: raise it and one RED is no longer enough.
    lenient = model_risk_state(
        views,
        core_views=("historical_var_95",),
        backtest_red_count=1,
        params=EnsembleParams(backtest_red_triggers=2),
    )
    assert lenient.triggers["backtest_red"] is False
    assert lenient.state == "LOW"


def test_red_verdict_count_only_counts_core_views():
    """A RED on the RESEARCH GARCH view must not raise the platform's
    model-risk state: nothing consumes that view."""
    rows = rv.compute_model_backtests(seeded_pnl(400)).rows
    faked = [
        r.__class__(**{**r.__dict__, "verdict": "RED"})
        for r in rows
        if r.model_name in ("historical_var", "garch_var")
    ]
    count, reasons = rv.red_verdict_count(faked)
    assert count == 2                        # both historical rows, not garch
    assert all("historical_var" in reason for reason in reasons)
    assert all("garch_var" not in reason for reason in reasons)


async def test_a_red_core_verdict_reaches_the_risk_view(client):
    """End to end: a persisted RED core row shows up as a model-risk reason
    on the SHADOW risk view — and changes no Tier 0 number."""
    await seed_stock_position(bars=520)
    async with SessionLocal() as session:
        session.add(
            RiskModelBacktestRow(
                as_of=datetime.now(timezone.utc),
                snapshot_id=None,
                model_name="historical_var",
                model_version="1.0.0",
                distribution="EMPIRICAL",
                confidence=0.99,
                horizon_days=1,
                window_obs=250,
                n_forecasts=250,
                exceedances=20,
                rate=0.08,
                expected_rate=0.01,
                kupiec_lr=40.0,
                kupiec_p=0.0000001,
                christoffersen_lr=0.5,
                christoffersen_p=0.48,
                es_severity_ratio=1.4,
                verdict="RED",
                health="ACTIVE",
                reason=None,
                params={"mode": "SHADOW", "min_forecasts": 60, "n_obs": 519},
            )
        )
        await session.commit()

    result = await build()
    model_risk = result.api["model_risk"]
    assert model_risk is not None
    assert any("RED backtest verdict" in r for r in model_risk["reasons"])
    assert model_risk["state"] in {"ELEVATED", "HIGH"}
    # The persisted row is served back unchanged (read, never recomputed).
    assert result.api["validation"]["rows"][0]["verdict"] == "RED"


# ---------------------------------------------------------------------------
# Compliance §65 — telemetry defined on the validation side
# ---------------------------------------------------------------------------


def test_model_health_ordinal_is_a_total_documented_mapping():
    """§65: ACTIVE=0 DEGRADED=1 UNAVAILABLE=2 FAILED=3, higher is worse.

    A DICTIONARY rather than ``list.index`` so adding a health state later
    is a deliberate edit here — a silent renumbering would invalidate every
    existing alert threshold without anything failing.
    """
    from libs.trading_core.risk.models.base import ModelHealth

    assert rv.MODEL_HEALTH_ORDINAL == {
        "ACTIVE": 0,
        "DEGRADED": 1,
        "UNAVAILABLE": 2,
        "FAILED": 3,
    }
    # Total over the enum: no health can reach the gauge unmapped.
    for health in ModelHealth:
        assert str(health) in rv.MODEL_HEALTH_ORDINAL
    # Monotone in severity, which is what makes max() and a `>= 2` alert
    # threshold meaningful.
    order = [rv.MODEL_HEALTH_ORDINAL[str(h)] for h in (
        ModelHealth.ACTIVE, ModelHealth.DEGRADED,
        ModelHealth.UNAVAILABLE, ModelHealth.FAILED,
    )]
    assert order == sorted(order) == [0, 1, 2, 3]


def test_set_model_health_gauge_skips_an_unknown_health():
    """§65 honest nulls: an unrecognised health string is SKIPPED, never
    mapped to a guessed ordinal. A fabricated 0 would read as ACTIVE, which
    is the one answer that could hide a broken model."""
    rv.MODEL_HEALTH_STATE.set(3.0, model="_probe_unknown")
    rv.set_model_health_gauge({"_probe_unknown": "NOT_A_HEALTH"})
    text = rv.REGISTRY.render_prometheus()
    line = next(
        l for l in text.splitlines()
        if l.startswith('model_health_state{model="_probe_unknown"}')
    )
    # Untouched: still the value we set, not overwritten with a guess.
    assert float(line.split()[1]) == 3.0


async def test_validation_run_counts_garch_fit_failures(client):
    """§65: a GARCH MLE that does not yield an ACTIVE fit is counted at the
    VALIDATION seam, labelled by site and resulting health.

    The counter fires where an MLE ACTUALLY RUNS and comes back non-ACTIVE.
    It deliberately does NOT fire when ``window < GARCH_MIN_OBS``, because
    that path short-circuits the whole view before attempting a fit — there
    is no failed fit to count, and inventing one would misreport a
    configuration choice as a model failure.
    """
    text_before = rv.REGISTRY.render_prometheus()
    before = {
        line.split()[0]: float(line.split()[1])
        for line in text_before.splitlines()
        if line.startswith("garch_fit_failures_total{")
    }

    # window=250 == GARCH min_obs, so the view IS attempted and the refit
    # stride runs real MLEs across the pass.
    async with SessionLocal() as session:
        await rv.run_model_backtests(
            session, book_pnl=seeded_pnl(600), window=250,
        )
        await session.commit()

    text_after = rv.REGISTRY.render_prometheus()
    after = {
        line.split()[0]: float(line.split()[1])
        for line in text_after.splitlines()
        if line.startswith("garch_fit_failures_total{")
    }
    validation_keys = [k for k in after if 'site="validation"' in k]
    assert validation_keys, "the validation seam recorded no GARCH fit outcome"
    assert any(
        after[k] > before.get(k, 0.0) for k in validation_keys
    ), "an MLE that came back non-ACTIVE must increment the counter"
    # A counter never goes backwards.
    for key, value in before.items():
        assert after[key] >= value
    # The health label is one of the real ModelHealth spellings (plus the
    # explicit RAISED used only for an exception at the snapshot seam).
    for key in validation_keys:
        health = key.split('health="')[1].split('"')[0]
        assert health in set(rv.MODEL_HEALTH_ORDINAL) | {"RAISED"}


async def test_validation_run_writes_no_audit_event_with_telemetry_on(client):
    """House rule, re-pinned after adding the §65 instruments: a validation
    run is a MEASUREMENT. Counting exceedances must not turn it into a
    decision — still zero audit events."""
    async with SessionLocal() as session:
        before = len((await session.execute(select(AuditEvent))).scalars().all())
    async with SessionLocal() as session:
        await rv.run_model_backtests(
            session, book_pnl=seeded_pnl(400), window=250,
        )
        await session.commit()
    async with SessionLocal() as session:
        after = len((await session.execute(select(AuditEvent))).scalars().all())
    assert after == before


async def test_short_window_does_not_fake_a_garch_fit_failure(client):
    """§65 boundary, pinned deliberately: ``window < GARCH_MIN_OBS`` skips
    the GARCH view BEFORE attempting an MLE, so there is no failed fit to
    count and the counter must NOT move.

    Counting one here would misreport a configuration choice ("we asked for
    a window GARCH cannot fit") as a model failure ("GARCH broke") — two
    different operational conditions that must not share a time series.
    """
    def failures() -> dict[str, float]:
        return {
            line.split()[0]: float(line.split()[1])
            for line in rv.REGISTRY.render_prometheus().splitlines()
            if line.startswith('garch_fit_failures_total{site="validation"')
        }

    before = failures()
    async with SessionLocal() as session:
        run = await rv.run_model_backtests(
            session, book_pnl=seeded_pnl(400), window=120,
        )
        await session.commit()
    after = failures()

    # The row is still produced, UNAVAILABLE, with the real reason — a
    # missing row would read as "never run" (spec §56).
    garch = next(r for r in run.rows if r.model_name == "garch_var")
    assert garch.verdict == "UNAVAILABLE"
    assert "250" in garch.reason  # names GARCH_MIN_OBS
    # ...but no fit was attempted, so nothing was counted.
    assert after == before
