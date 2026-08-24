"""Tests for the statistical risk snapshot BUILDER (apps/gateway/risk_snapshot.py).

Phase B second half, design contract §6. Everything here is SHADOW: the
builder measures, persists and serialises — it never decides. The sibling
file tests/test_risk_snapshot_api.py owns the HTTP contract; this one drives
the builder and its helpers directly.
"""
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from apps.gateway.execution import gate_chain
from sqlalchemy import select

from apps.gateway import risk_snapshot as rs
from apps.gateway.db import (
    AtmIvDailyRow,
    AuditEvent,
    Position,
    RiskContributionRow,
    RiskMetricRow,
    RiskSnapshotRow,
    SessionLocal,
    StockBarDaily,
)

INITIAL_CASH = 100_000.0
#: A synthetic close path with real ups and downs — enough observations that
#: the 95% grid is ACTIVE (contract §2.3: min_obs 60, ACTIVE at 2× that).
_STEPS = (0.012, -0.008, 0.004, -0.015, 0.009, -0.003, 0.006, -0.011)


def _closes(n: int, start: float = 100.0) -> list[float]:
    out = [start]
    for i in range(n - 1):
        out.append(out[-1] * (1.0 + _STEPS[i % len(_STEPS)]))
    return out


async def seed_stock_position(
    ticker: str = "AAPL",
    *,
    bars: int = 200,
    quantity: int = 100,
    max_loss: float = 1_000.0,
) -> int:
    """One OPEN LONG_STOCK position plus ``bars`` stored daily closes."""
    async with SessionLocal() as session:
        start = date(2025, 1, 1)
        for i, close in enumerate(_closes(bars)):
            session.add(
                StockBarDaily(
                    ticker=ticker,
                    ts=start + timedelta(days=i),
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=5_000_000,
                )
            )
        position = Position(
            ticker=ticker,
            instrument="LONG_STOCK",
            quantity=quantity,
            avg_price=100.0,
            max_loss=max_loss,
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
        )
        session.add(position)
        await session.commit()
        return position.id


async def build(*, trigger: str = "ON_DEMAND", cash: float | None = INITIAL_CASH,
                persist: bool = True) -> rs.RiskSnapshotBuild:
    """Run one build in its own committed session (what a caller does)."""
    async with SessionLocal() as session:
        result = await rs.build_risk_snapshot(
            session,
            trigger=trigger,
            cash=cash,
            trading_enabled=False,
            persist=persist,
        )
        await session.commit()
        return result


# ---------------------------------------------------------------------------
# (b) A real book: models ACTIVE, contributions additive, rows persisted
# ---------------------------------------------------------------------------


async def test_seeded_stock_position_produces_active_models(client):
    """≥ 80 bars + one LONG_STOCK ⇒ historical/gaussian VaR & ES computed,
    with the sample and tail sizes that produced them."""
    await seed_stock_position(bars=520)
    result = await build()
    api = result.api

    assert api["mode"] == "SHADOW"
    assert api["pnl_method"] == "DELTA_LINEAR"
    assert api["n_obs"] == 519  # 520 bars -> 519 returns
    assert api["window_start"] is not None and api["window_end"] is not None

    for row in api["var"] + api["es"]:
        assert row["health"] in {"ACTIVE", "DEGRADED"}, row
        assert row["value_usd"] is not None
        # The vol-scaled views burn the EWMA warm-up (init_obs) before their
        # first usable observation, so their sample is honestly SMALLER than
        # the raw one — the number reported is the number used.
        expected = 519 if row["model"] != "HISTORICAL_VOL_SCALED" else 499
        assert row["sample_size"] == expected, row
        assert row["tail_size"] is not None and row["tail_size"] > 0
        # Methodology travels with every number (spec §50).
        assert row["model_name"] and row["model_version"]
        if row["horizon_days"] == 1:
            assert row["scaling"] is None  # 1-day numbers are not scaled
            assert row["mode"] == "SHADOW"
        else:
            # §6/§12: a multi-day row is SCALED, never estimated, and says so
            # on the row itself rather than leaving the reader to infer it.
            assert row["horizon_days"] in (5, 10), row
            assert row["model"] == "HISTORICAL"
            assert row["scaling"] == "SQRT_TIME"
            assert row["mode"] == "RESEARCH"

    # ES ≥ VaR at the same confidence (contract §3 invariant 1) — the shadow
    # layer must not contradict its own library.
    var95 = next(r for r in api["var"] if r["model"] == "HISTORICAL" and r["confidence"] == 0.95)
    es95 = next(r for r in api["es"] if r["model"] == "HISTORICAL" and r["confidence"] == 0.95)
    assert es95["value_usd"] >= var95["value_usd"]

    # pct_nav is a FRACTION of NAV, computed not fabricated.
    assert var95["pct_nav"] == pytest.approx(var95["value_usd"] / result.snapshot.nav)


async def test_contributions_sum_to_total_and_carry_capital_weight(client):
    """Σ contributions == total (contract §3 invariant 3) and each row names
    its CAPITAL weight next to its RISK weight (spec §49)."""
    await seed_stock_position(bars=200)
    api = (await build()).api

    for block in (api["contributions"]["es"], api["contributions"]["vol"]):
        assert block is not None
        assert block["health"] in {"ACTIVE", "DEGRADED"}
        assert block["rows"]
        total = sum(r["contribution_usd"] for r in block["rows"])
        assert total == pytest.approx(block["total_usd"], rel=1e-9, abs=1e-9)
        for row in block["rows"]:
            assert row["ticker"] == "AAPL"
            assert row["instrument"] == "LONG_STOCK"
            assert 0.0 < row["capital_weight"] <= 1.0
    assert api["contributions"]["es"]["confidence"] == 0.95


async def test_persisted_rows_carry_full_model_identity(client):
    """One snapshot row + one metric row per (metric, model, confidence,
    horizon) with the FULL §44 provenance inline, plus contribution rows."""
    await seed_stock_position(bars=200)
    result = await build()

    async with SessionLocal() as session:
        snapshots = (await session.execute(select(RiskSnapshotRow))).scalars().all()
        metrics = (await session.execute(select(RiskMetricRow))).scalars().all()
        contributions = (
            await session.execute(select(RiskContributionRow))
        ).scalars().all()

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.id == result.row_id
    assert snap.trigger == "ON_DEMAND"
    assert snap.snapshot_version == "b.1"
    assert snap.n_obs == 199
    assert snap.n_positions == 1
    assert snap.pnl_method == "DELTA_LINEAR"
    assert snap.nav == pytest.approx(result.snapshot.nav)
    assert snap.data_quality_valid is True
    assert snap.model_health  # the per-model ledger, persisted

    # 5 one-day VaR views + 5 one-day ES views + volatility
    # + the 2 √h-scaled HISTORICAL VaR display rows (h=5, h=10)
    # + the 2 matching ES display rows (§6/§12)
    # + the §34 diversification ratio.
    assert len(metrics) == 16
    by_metric = {}
    for m in metrics:
        by_metric.setdefault(m.metric, []).append(m)
    assert sorted(by_metric) == ["DIVERSIFICATION_RATIO", "ES", "VAR", "VOLATILITY"]
    assert len(by_metric["VAR"]) == 7 and len(by_metric["ES"]) == 7
    assert len(by_metric["DIVERSIFICATION_RATIO"]) == 1
    # The display rows are persisted with their REAL horizon, so a stored
    # 5-day number can never later be misread as a 1-day one.
    assert sorted(m.horizon_days for m in by_metric["VAR"]) == [1, 1, 1, 1, 1, 5, 10]
    assert {m.metric for m in metrics} == {
        "VAR", "ES", "VOLATILITY", "DIVERSIFICATION_RATIO",
    }
    for metric in metrics:
        assert metric.snapshot_id == snap.id
        assert metric.model_name
        assert metric.model_version
        assert metric.health in {"ACTIVE", "DEGRADED", "UNAVAILABLE", "FAILED"}
        # Reproducible without a join (spec §44): the estimator's own params
        # plus the data provenance.
        assert isinstance(metric.params, dict) and metric.params
        assert "return_type" in metric.params
        assert "lookback" in metric.params
        assert isinstance(metric.diagnostics, dict)
    # Exactly the contract §6 grid, in both metric families.
    var_grid = {(m.model_name, m.confidence) for m in metrics if m.metric == "VAR"}
    assert var_grid == {
        ("historical_var", 0.95),
        ("historical_var", 0.99),
        ("gaussian_var", 0.95),
        ("gaussian_var", 0.99),
        ("conditional_var", 0.95),
    }

    # §34: the diversification row carries its own model identity, is a pure
    # RATIO (so no NAV fraction), and has no confidence level — a
    # diversification ratio has no tail to take a quantile of.
    dr = by_metric["DIVERSIFICATION_RATIO"][0]
    assert dr.model_name == "diversification_ratio"
    assert dr.model_version == "1.0.0"
    assert dr.confidence is None
    assert dr.value_pct_nav is None
    assert dr.value is not None and dr.value >= 1.0
    assert dr.params["ddof"] == 1
    assert "stdev" in dr.params["estimator"]

    assert {c.method for c in contributions} == {"ES", "VOL"}
    for contribution in contributions:
        assert contribution.snapshot_id == snap.id
        assert contribution.position_key.startswith("AAPL#")
        assert contribution.capital_weight is not None


async def test_persist_false_writes_nothing(client):
    """``persist=False`` measures without recording — the preview path."""
    await seed_stock_position(bars=200)
    result = await build(persist=False)
    assert result.row_id is None
    assert result.api["snapshot_id"] is None
    async with SessionLocal() as session:
        assert (await session.execute(select(RiskSnapshotRow))).scalars().all() == []
        assert (await session.execute(select(RiskMetricRow))).scalars().all() == []


async def test_builder_writes_no_audit_event(client):
    """A snapshot is a MEASUREMENT, not a decision (house rule: read views
    write no audit events)."""
    await seed_stock_position(bars=200)
    async with SessionLocal() as session:
        before = len((await session.execute(select(AuditEvent))).scalars().all())
    await build()
    async with SessionLocal() as session:
        after = len((await session.execute(select(AuditEvent))).scalars().all())
    assert after == before


# ---------------------------------------------------------------------------
# (c) Not enough history: honest UNAVAILABLE with the real numbers
# ---------------------------------------------------------------------------


async def test_short_history_is_unavailable_with_reasons(client):
    """< 60 observations ⇒ every model UNAVAILABLE with n and min_obs in the
    reason, and data_quality invalid with a reason (never a fabricated 0)."""
    await seed_stock_position(bars=30)
    api = (await build()).api

    assert api["n_obs"] == 29
    dq = api["data_quality"]
    assert dq["valid"] is False
    assert any("n_obs=29" in r and "min_obs=60" in r for r in dq["reasons"]), dq

    for row in api["var"] + api["es"]:
        assert row["health"] == "UNAVAILABLE"
        assert row["value_usd"] is None
        assert row["pct_nav"] is None
        # The reason carries the REAL numbers (contract §1 honest nulls):
        # this view's own sample and the minimum it needed.
        assert f"n={row['sample_size']}" in row["reason"], row
        assert "min_obs=" in row["reason"], row
    assert api["volatility"]["value_usd"] is None
    assert api["volatility"]["health"] == "UNAVAILABLE"
    assert "n=29" in api["volatility"]["reason"]


async def test_no_cash_is_a_data_quality_reason_not_a_crash(client):
    """No account (``cash=None``) is a data fact, not an error."""
    await seed_stock_position(bars=200)
    result = await build(cash=None)
    dq = result.api["data_quality"]
    assert dq["valid"] is False
    assert any("no account" in r for r in dq["reasons"]), dq


# ---------------------------------------------------------------------------
# (d) A position without a delta is EXCLUDED and NAMED
# ---------------------------------------------------------------------------


async def test_position_without_delta_is_excluded_and_named(client):
    """A position whose greeks cannot be known is left OUT of the book P&L
    and listed with the reason — never priced at zero delta."""
    good = await seed_stock_position("AAPL", bars=200)
    async with SessionLocal() as session:
        # An option on a ticker with NO stored bars: no spot, no chain, so
        # portfolio_greeks_read reports data_ok False with its own note.
        orphan = Position(
            ticker="ZZZZ",
            instrument="LONG_CALL",
            quantity=2,
            avg_price=3.0,
            max_loss=600.0,
            multiplier=100,
            opt_expiry="2099-01-15",
            opt_strike=100.0,
            opt_right="C",
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
        )
        session.add(orphan)
        await session.commit()
        orphan_id = orphan.id

    api = (await build()).api
    excluded = {e["key"]: e["reason"] for e in api["positions_excluded"]}
    assert f"ZZZZ#{orphan_id}" in excluded
    assert excluded[f"ZZZZ#{orphan_id}"]  # a real reason, not an empty string
    assert f"AAPL#{good}" not in excluded
    assert api["data_quality"]["keys_excluded"] == [f"ZZZZ#{orphan_id}"]
    # The surviving position still carries the book.
    assert [r["key"] for r in api["contributions"]["es"]["rows"]] == [f"AAPL#{good}"]


# ---------------------------------------------------------------------------
# A failing estimator degrades to FAILED — it never 500s the caller
# ---------------------------------------------------------------------------


async def test_estimator_exception_becomes_failed_not_an_exception(client, monkeypatch):
    """A model that raises yields health FAILED carrying the error text."""
    await seed_stock_position(bars=200)

    def boom(*args, **kwargs):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(rs, "historical_var", boom)
    api = (await build()).api
    row = next(r for r in api["var"] if r["model"] == "HISTORICAL" and r["confidence"] == 0.95)
    assert row["health"] == "FAILED"
    assert row["value_usd"] is None
    assert "estimator exploded" in row["reason"]
    # A FAILED core view is model risk, and it is reported as such (§59).
    assert api["model_risk"]["state"] == "HIGH"


# ---------------------------------------------------------------------------
# (f) The background loop body: ONE SCHEDULED row per NY day
# ---------------------------------------------------------------------------


async def test_scheduled_loop_body_persists_one_row_per_ny_day(client):
    """Calling the tick twice on the same NY day persists exactly one row;
    live drawdown then reads it."""
    await seed_stock_position(bars=200)

    first = await rs.run_scheduled_snapshot()
    second = await rs.run_scheduled_snapshot()

    assert first.get("snapshot_id") is not None
    assert second == {"skipped": "ALREADY_BUILT_TODAY", "day": str(rs.new_york_today())}

    async with SessionLocal() as session:
        scheduled = (
            await session.execute(
                select(RiskSnapshotRow).where(RiskSnapshotRow.trigger == "SCHEDULED")
            )
        ).scalars().all()
    assert len(scheduled) == 1
    assert scheduled[0].nav is not None

    # The drawdown block now reads that SCHEDULED NAV path (one observation
    # is still too few for a drawdown — honest UNAVAILABLE with the numbers).
    drawdown_api = (await build()).drawdown_api
    assert drawdown_api["nav_series"]["n"] == 1
    assert drawdown_api["nav_series"]["source"] == "risk_snapshots SCHEDULED"
    assert drawdown_api["health"] == "UNAVAILABLE"
    assert "2" in drawdown_api["reason"]  # "n=1 < min_obs=2"


async def test_drawdown_reads_the_scheduled_nav_series(client):
    """Two SCHEDULED days ⇒ a real drawdown measured off the persisted path."""
    await seed_stock_position(bars=200)
    await rs.run_scheduled_snapshot()
    # A second, EARLIER day at a higher NAV: the current NAV is then below
    # its peak, so the drawdown is negative and measurable.
    async with SessionLocal() as session:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        session.add(
            RiskSnapshotRow(
                as_of=yesterday,
                snapshot_version="b.1",
                trigger="SCHEDULED",
                nav=1_000_000.0,
                n_positions=0,
            )
        )
        await session.commit()

    drawdown_api = (await build()).drawdown_api
    assert drawdown_api["nav_series"]["n"] == 2
    # Two observations is a measurable but thin path — DEGRADED with a
    # reason is the honest health, not ACTIVE.
    assert drawdown_api["health"] in {"ACTIVE", "DEGRADED"}
    assert drawdown_api["current_pct"] < 0
    assert drawdown_api["peak_nav"] == pytest.approx(1_000_000.0)


async def test_scheduled_tick_skips_without_market_data(unconfigured_client):
    """No provider ⇒ no snapshot (every return series would be missing);
    the skip is named, not a fabricated row."""
    result = await rs.run_scheduled_snapshot()
    assert result["skipped"] == "MARKET_DATA_NOT_CONFIGURED"
    async with SessionLocal() as session:
        assert (await session.execute(select(RiskSnapshotRow))).scalars().all() == []


# ---------------------------------------------------------------------------
# (g) atm_iv_daily upsert
# ---------------------------------------------------------------------------


async def test_record_atm_iv_inserts_then_updates(client):
    """Upsert by (ticker, bar_date) — a second call on the same day updates
    rather than duplicating."""
    day = date(2026, 8, 18)
    async with SessionLocal() as session:
        await rs.record_atm_iv(
            session, "AAPL", bar_date=day, atm_iv=0.31, spot=190.0,
            expiry=date(2026, 9, 18), dte=31, source="alpaca_chain",
        )
        await session.commit()
    async with SessionLocal() as session:
        await rs.record_atm_iv(
            session, "AAPL", bar_date=day, atm_iv=0.28, spot=191.5,
            source="alpaca_chain",
        )
        await session.commit()

    async with SessionLocal() as session:
        rows = (await session.execute(select(AtmIvDailyRow))).scalars().all()
    assert len(rows) == 1
    assert rows[0].atm_iv == pytest.approx(0.28)
    assert rows[0].spot == pytest.approx(191.5)
    assert rows[0].source == "alpaca_chain"


async def test_record_atm_iv_ignores_null_iv_and_never_raises(client):
    """A null IV records nothing (there is nothing to record) and a broken
    session is swallowed — best-effort by contract."""
    async with SessionLocal() as session:
        await rs.record_atm_iv(
            session, "AAPL", bar_date=date(2026, 8, 18), atm_iv=None,
            spot=190.0, source="stub_chain",
        )
        await session.commit()
    async with SessionLocal() as session:
        assert (await session.execute(select(AtmIvDailyRow))).scalars().all() == []

    class _Exploding:
        async def execute(self, *a, **k):
            raise RuntimeError("db is gone")

    # Must NOT raise: the caller is a chain read or an order gate.
    await rs.record_atm_iv(
        _Exploding(), "AAPL", bar_date=date(2026, 8, 18), atm_iv=0.3,
        spot=190.0, source="stub_chain",
    )


async def test_latest_snapshot_row_filters_by_trigger(client):
    """``latest_snapshot_row`` answers honestly when nothing exists."""
    async with SessionLocal() as session:
        assert await rs.latest_snapshot_row(session) is None
    await seed_stock_position(bars=200)
    await build(trigger="ON_DEMAND")
    async with SessionLocal() as session:
        assert (await rs.latest_snapshot_row(session)).trigger == "ON_DEMAND"
        assert await rs.latest_snapshot_row(session, trigger="SCHEDULED") is None
        assert (
            await rs.latest_snapshot_row(session, trigger="ON_DEMAND")
        ).trigger == "ON_DEMAND"


# ---------------------------------------------------------------------------
# Compliance §45 — the declared-but-never-passed correlation_state
# ---------------------------------------------------------------------------


async def test_typed_snapshot_carries_the_correlation_state(client):
    """§45 BUG (compliance Tier C): ``PortfolioRiskSnapshot`` has DECLARED a
    ``correlation_state`` field since Phase B, but the builder never passed
    it — so the typed snapshot always read None while the very same object
    reached the wire through a separate argument. Two surfaces, one number,
    silently disagreeing.

    With two tickers the regime is measurable, and the typed snapshot and
    the API dict must now agree — field by field, not merely both truthy.
    """
    await seed_stock_position("AAPL", bars=200)
    await seed_stock_position("MSFT", bars=200, quantity=50)
    result = await build()

    state = result.snapshot.correlation_state
    assert state is not None, "the builder must pass correlation_state (§45)"
    # The typed field holds the real dataclass, not a stringified summary:
    # the API serialiser needs `worst_pairs`, and a `str` could not carry it.
    assert state.state in {"NORMAL", "ELEVATED", "CONVERGING", "UNAVAILABLE"}
    assert state.n_pairs >= 1

    api_state = result.api["correlation_state"]
    assert api_state is not None
    # The two surfaces agree on every scalar the wire carries.
    assert api_state["state"] == state.state
    assert api_state["n_pairs"] == state.n_pairs
    assert api_state["current_avg"] == state.current_avg
    assert api_state["normal_avg"] == state.normal_avg
    assert api_state["delta"] == state.delta
    assert [tuple(p) for p in api_state["worst_pairs"]] == [
        tuple(p) for p in state.worst_pairs
    ]


async def test_single_ticker_leaves_correlation_state_an_honest_null(client):
    """A correlation needs a PAIR. With one ticker the typed field is None
    and the API is null — the honest null, not a fabricated NORMAL regime."""
    await seed_stock_position("AAPL", bars=200)
    result = await build()
    assert result.snapshot.correlation_state is None
    assert result.api["correlation_state"] is None


# ---------------------------------------------------------------------------
# Compliance §34 — diversification ratio (audit.md:215, P1)
# ---------------------------------------------------------------------------


async def test_diversification_ratio_is_served_with_its_estimator(client):
    """§34: DR = Σ_i σ_i / σ_p on the book P&L series, served as
    ``statistical.diversification_ratio`` with the estimator stated."""
    await seed_stock_position("AAPL", bars=200)
    await seed_stock_position("MSFT", bars=200, quantity=50)
    result = await build()
    block = result.api["diversification_ratio"]

    assert set(block) == {
        "value", "health", "reason", "n_obs",
        "model_name", "model_version", "estimator", "mode",
    }
    assert block["health"] in {"ACTIVE", "DEGRADED"}
    assert block["n_obs"] == 199
    assert block["model_name"] == "diversification_ratio"
    assert block["model_version"] == "1.0.0"
    assert block["mode"] == "SHADOW"  # decides nothing
    # DR >= 1 for any imperfectly correlated book (triangle inequality).
    assert block["value"] >= 1.0
    # The exact convention is on the wire, not left to the reader to guess.
    assert "ddof=1" in block["estimator"]


async def test_diversification_ratio_is_an_honest_null_on_a_short_window(client):
    """Below ``min_obs`` the ratio is UNAVAILABLE with the REAL numbers in
    the reason — never a fabricated 1.0, which would read as "no
    diversification" rather than "not measured"."""
    await seed_stock_position("AAPL", bars=30)
    await seed_stock_position("MSFT", bars=30, quantity=50)
    result = await build()
    block = result.api["diversification_ratio"]
    assert block["value"] is None
    assert block["health"] == "UNAVAILABLE"
    assert "min_obs=60" in block["reason"]


async def test_empty_book_diversification_ratio_says_why(client):
    """No positions ⇒ nothing to diversify. The block is still PRESENT (an
    absent key would read as "not implemented"), with an honest null."""
    result = await build()
    block = result.api["diversification_ratio"]
    assert block["value"] is None
    assert block["health"] == "UNAVAILABLE"
    assert block["reason"]


# ---------------------------------------------------------------------------
# §34 — hand-checked arithmetic on the pure estimator
# ---------------------------------------------------------------------------


def _alternating(n: int) -> list[float]:
    """A deterministic zero-mean-ish series with real dispersion."""
    return [float((-1) ** i * (1 + (i % 7))) for i in range(n)]


def test_perfectly_correlated_book_has_a_diversification_ratio_of_one():
    """HAND CHECK. Let ``b = 2a``. Then

        Σ σ_i = σ_a + σ_b = σ_a + 2σ_a = 3σ_a
        σ_p   = stdev(a + b) = stdev(3a) = 3σ_a
        DR    = 3σ_a / 3σ_a = 1

    Perfect positive correlation ⇒ NO diversification, exactly 1. This is
    the §19 failure mode the ratio exists to make visible.
    """
    from libs.trading_core.risk.snapshot import diversification_ratio

    a = _alternating(120)
    b = [2.0 * x for x in a]
    total = [x + y for x, y in zip(a, b)]

    result = diversification_ratio({"A": a, "B": b}, total)
    assert result.health.name == "ACTIVE"
    assert result.value == pytest.approx(1.0, abs=1e-12)
    assert result.sample_size == 120
    # The diagnostics carry both halves of the quotient, so the number is
    # checkable from the stored row without re-running the estimator. Here
    # numerator and denominator are the SAME 3σ_a, which is why DR is 1.
    import statistics

    sigma_a = statistics.stdev(a)
    assert result.diagnostics["sigma_sum_usd"] == pytest.approx(3.0 * sigma_a)
    assert result.diagnostics["sigma_portfolio_usd"] == pytest.approx(3.0 * sigma_a)
    assert result.diagnostics["n_positions"] == 2


def test_anti_correlated_book_has_a_diversification_ratio_above_one():
    """HAND CHECK. Let ``b = -0.5a`` (imperfectly offsetting). Then

        Σ σ_i = σ_a + 0.5σ_a = 1.5σ_a
        σ_p   = stdev(a - 0.5a) = stdev(0.5a) = 0.5σ_a
        DR    = 1.5σ_a / 0.5σ_a = 3

    Offsetting positions ⇒ DR strictly ABOVE 1, and the exact value is 3.
    """
    from libs.trading_core.risk.snapshot import diversification_ratio

    a = _alternating(120)
    b = [-0.5 * x for x in a]
    total = [x + y for x, y in zip(a, b)]

    result = diversification_ratio({"A": a, "B": b}, total)
    assert result.health.name == "ACTIVE"
    assert result.value == pytest.approx(3.0, rel=1e-12)
    assert result.value > 1.0


def test_exactly_hedged_book_has_no_diversification_ratio():
    """``b = -a`` ⇒ σ_p = 0 ⇒ the quotient is undefined. UNAVAILABLE with
    the reason, NOT an infinity and not a fabricated large number."""
    from libs.trading_core.risk.snapshot import diversification_ratio

    a = _alternating(120)
    b = [-x for x in a]
    total = [x + y for x, y in zip(a, b)]

    result = diversification_ratio({"A": a, "B": b}, total)
    assert result.value is None
    assert result.health.name == "UNAVAILABLE"
    assert "denominator is zero" in result.reason


def test_diversification_ratio_is_scale_invariant():
    """DR is a RATIO of standard deviations: doubling every position's P&L
    doubles numerator and denominator alike and must leave it unchanged
    (contract §3 invariant 4, applied to this estimator)."""
    from libs.trading_core.risk.snapshot import diversification_ratio

    a = _alternating(120)
    b = [0.3 * x + float(i % 5) for i, x in enumerate(a)]
    total = [x + y for x, y in zip(a, b)]

    base = diversification_ratio({"A": a, "B": b}, total)
    scaled = diversification_ratio(
        {"A": [2 * x for x in a], "B": [2 * x for x in b]},
        [2 * x for x in total],
    )
    assert scaled.value == pytest.approx(base.value, rel=1e-12)


def test_diversification_ratio_rejects_a_misaligned_series():
    """A per-position series of a different length than the total means the
    window is not aligned — an honest UNAVAILABLE naming both counts, never
    a ratio computed over mismatched days."""
    from libs.trading_core.risk.snapshot import diversification_ratio

    a = _alternating(120)
    result = diversification_ratio({"A": a, "B": a[:-1]}, a)
    assert result.value is None
    assert result.health.name == "UNAVAILABLE"
    assert "not aligned" in result.reason


# ---------------------------------------------------------------------------
# Compliance §6/§12 — √h display rows and the GARCH term structure
# ---------------------------------------------------------------------------


async def test_display_horizons_are_exactly_sqrt_h_of_the_one_day_row(client):
    """§6/§12: the promised RESEARCH display. ``value_5d == value_1d × √5``
    EXACTLY (the library scales, it does not re-estimate), labelled
    SQRT_TIME, and never presented as an estimated multi-day number."""
    import math

    await seed_stock_position("AAPL", bars=520)
    api = (await build()).api

    for side in ("var", "es"):
        rows = {
            (r["model"], r["confidence"], r["horizon_days"]): r for r in api[side]
        }
        one_day = rows[("HISTORICAL", 0.95, 1)]
        assert one_day["scaling"] is None
        assert one_day["mode"] == "SHADOW"
        for h in (5, 10):
            row = rows[("HISTORICAL", 0.95, h)]
            assert row["value_usd"] == pytest.approx(
                one_day["value_usd"] * math.sqrt(h), rel=1e-12
            ), (side, h)
            assert row["scaling"] == "SQRT_TIME"
            # RESEARCH, not SHADOW: a scaled number is a display, and √h
            # assumes i.i.d. returns that were never validated here.
            assert row["mode"] == "RESEARCH"
            # Same sample and tail as the 1-day number it was scaled from —
            # no extra observation was consulted.
            assert row["sample_size"] == one_day["sample_size"]
            assert row["tail_size"] == one_day["tail_size"]


async def test_display_horizons_do_not_disturb_the_one_day_grid(client):
    """The §6/§12 rows are ADDITIVE. Every 1-day number, and the whole
    1-day array prefix, must be byte-identical to what shipped before."""
    await seed_stock_position("AAPL", bars=520)
    api = (await build()).api

    for side in ("var", "es"):
        prefix = api[side][: len(rs.VIEW_ORDER)]
        assert [(r["model"], r["confidence"]) for r in prefix] == list(rs.VIEW_ORDER)
        assert all(r["horizon_days"] == 1 for r in prefix)
        # And the appended rows are only the HISTORICAL display horizons.
        for row in api[side][len(rs.VIEW_ORDER):]:
            assert (row["model"], row["confidence"]) == ("HISTORICAL", 0.95)
            assert row["horizon_days"] in rs.DISPLAY_HORIZONS


async def test_display_horizons_are_honest_nulls_on_an_empty_book(client):
    """No book ⇒ the display rows are PRESENT and UNAVAILABLE, exactly like
    the 1-day rows beside them. √h × None is not 0."""
    api = (await build()).api
    for side in ("var", "es"):
        for row in api[side]:
            if row["horizon_days"] == 1:
                continue
            assert row["value_usd"] is None
            assert row["health"] == "UNAVAILABLE"
            assert row["reason"]


async def test_conditional_horizon_sigmas_are_null_under_ewma(client):
    """§12: with ~200 bars GARCH is UNAVAILABLE (min_obs 250), so EWMA is the
    live conditional source and the term structure is an honest null naming
    the reason — an EWMA filter has no variance term structure to report."""
    await seed_stock_position("AAPL", bars=200)
    api = (await build()).api

    assert api["conditional_source"]["source"] == "EWMA"
    block = api["conditional_horizon_sigmas"]
    assert set(block) == {"h5_usd", "h10_usd", "source", "reason"}
    assert block["h5_usd"] is None and block["h10_usd"] is None
    assert block["source"] is None
    assert "not GARCH" in block["reason"]


def test_conditional_horizon_sigmas_use_the_garch_term_structure():
    """§12: on the GARCH branch the sigmas are
    ``sqrt(Σ_{k=1..h} σ²_{t+k})`` — the CLOSED-FORM aggregation, labelled
    GARCH_TERM_STRUCTURE so it is never confused with the √h scaling on the
    historical display rows beside it.

    Verified against ``garch_forecast_variance`` directly, and asserted to
    DIFFER from naive √h scaling: GARCH mean-reverts toward the
    unconditional variance, so the two aggregations genuinely disagree.
    """
    import math

    from libs.trading_core.risk.models.garch import (
        fit_garch,
        garch_forecast_variance,
    )

    # A series SIMULATED from a known GARCH(1,1) (omega .05, alpha .10,
    # beta .85 — persistence .95, the textbook equity regime), seeded so the
    # test is deterministic. A real data-generating process is what makes the
    # fit ACTIVE, which is what gives the term structure something to say.
    import random

    rng = random.Random(20260819)
    values: list[float] = []
    sigma2 = 0.05 / (1.0 - 0.10 - 0.85)
    prev = 0.0
    for _ in range(800):
        sigma2 = 0.05 + 0.10 * prev * prev + 0.85 * sigma2
        prev = math.sqrt(sigma2) * rng.gauss(0.0, 1.0)
        values.append(prev)

    fit = fit_garch(values)
    assert fit.health.name == "ACTIVE", fit.reason

    block = rs._conditional_horizon_sigmas(fit, source="GARCH", reason="")
    assert block["source"] == "GARCH_TERM_STRUCTURE"
    assert block["reason"] is None

    variances = garch_forecast_variance(fit, 10)
    for h in (5, 10):
        assert block[f"h{h}_usd"] == pytest.approx(
            math.sqrt(math.fsum(variances[:h])), rel=1e-12
        )
    # Not the same thing as √h scaling of the one-step forecast.
    sigma1 = math.sqrt(variances[0])
    assert block["h10_usd"] != pytest.approx(sigma1 * math.sqrt(10), rel=1e-6)


# ---------------------------------------------------------------------------
# (i) Compliance batch 2 — FULL_REVAL_CONST_IV on the wire (design §10.3/§10.4)
# ---------------------------------------------------------------------------
#
# The builder now feeds the SAME chain-resolved legs it has always given the
# stress engine into the P&L series as well. These tests pin the four things
# that can go wrong: the label lies, the stock-only numbers move, the Euler
# identity breaks on a mixed book, or a chain gap becomes a silent guess.


async def _es95(api) -> dict:
    return next(
        row
        for row in api["es"]
        if row["model"] == "HISTORICAL"
        and row["confidence"] == 0.95
        and row["horizon_days"] == 1
    )


def _numbers_fingerprint(api: dict) -> dict:
    """Every statistical number the API serves, flattened — the object the
    stock-only regression pin compares before and after."""
    return {
        "pnl_method": api["pnl_method"],
        "n_obs": api["n_obs"],
        "window_start": api["window_start"],
        "window_end": api["window_end"],
        "var": [
            (r["model"], r["confidence"], r["horizon_days"], r["value_usd"],
             r["pct_nav"], r["sample_size"], r["tail_size"], r["health"])
            for r in api["var"]
        ],
        "es": [
            (r["model"], r["confidence"], r["horizon_days"], r["value_usd"],
             r["pct_nav"], r["sample_size"], r["tail_size"], r["health"])
            for r in api["es"]
        ],
        "volatility": api["volatility"],
        "distribution": api["distribution"],
        "dispersion": api["dispersion"],
        "diversification_ratio": api["diversification_ratio"],
        "contributions": api["contributions"],
        "model_health": api["model_health"],
    }


async def test_stock_only_build_is_identical_to_the_pinned_pre_change_run(client):
    """§10.4's STRONGEST pin: with no option position, the batch-2 code path
    must serve numbers IDENTICAL to the pre-change builder.

    The pre-change behaviour is captured by running the SAME build on the
    SAME seed with the leg fields absent — `option_leg_fields_by_key`
    returning ``{}`` is exactly what the builder did before this batch, so
    the two runs differ only by the code under test. Compared with ``==``,
    not ``approx``: a stock row must not move by an ulp.
    """
    await seed_stock_position(bars=320)
    after = _numbers_fingerprint((await build(persist=False)).api)

    # The pre-change builder, reproduced: no leg fields reach _position_inputs.
    real = rs.option_leg_fields_by_key
    try:
        rs.option_leg_fields_by_key = lambda option_legs: {}
        before = _numbers_fingerprint((await build(persist=False)).api)
    finally:
        rs.option_leg_fields_by_key = real

    assert after == before
    # ...and the label a stock-only holder sees is unchanged.
    assert after["pnl_method"] == "DELTA_LINEAR"


async def test_stock_only_build_labels_every_row_delta_linear(client):
    await seed_stock_position(bars=200)
    result = await build()
    dq = result.api["data_quality"]

    assert result.api["pnl_method"] == "DELTA_LINEAR"
    assert set(dq["pnl_method_by_key"].values()) == {"DELTA_LINEAR"}
    assert dq["pnl_method_counts"] == {
        "FULL_REVAL_CONST_IV": 0, "DELTA_LINEAR": 1,
    }
    # The persisted column agrees with the served string (design §10.3).
    async with SessionLocal() as session:
        row = (
            await session.execute(select(RiskSnapshotRow).order_by(RiskSnapshotRow.id))
        ).scalars().all()[-1]
        assert row.pnl_method == "DELTA_LINEAR"
        assert set(row.data_quality["pnl_method_by_key"].values()) == {"DELTA_LINEAR"}


async def test_empty_book_serves_the_labelling_keys_with_honest_empties(client):
    """No book ⇒ nothing was priced by anything: an EMPTY map, and both
    counts present at zero rather than the keys being absent."""
    dq = (await build()).api["data_quality"]
    assert dq["pnl_method_by_key"] == {}
    assert dq["pnl_method_counts"] == {
        "FULL_REVAL_CONST_IV": 0, "DELTA_LINEAR": 0,
    }


async def test_seeded_option_position_flips_pnl_method_to_full_reval(client):
    """A LONG_CALL on a real stub-chain contract (which carries an IV) is
    revalued, the book-level summary flips, and the per-key map says which
    row earned the label — while the stock row beside it stays linear."""
    from .test_risk_stress_api import seed_long_call

    stock_id = await seed_stock_position(bars=200)
    option_id = await seed_long_call(client)
    result = await build()
    api = result.api
    dq = api["data_quality"]

    assert api["pnl_method"] == "FULL_REVAL_CONST_IV"
    assert dq["pnl_method_by_key"] == {
        f"AAPL#{stock_id}": "DELTA_LINEAR",
        f"AAPL#{option_id}": "FULL_REVAL_CONST_IV",
    }
    assert dq["pnl_method_counts"] == {
        "FULL_REVAL_CONST_IV": 1, "DELTA_LINEAR": 1,
    }

    # THE SAME CHAIN, ONCE. The leg the P&L series used must be the very leg
    # the stress engine got — same strike, right, tenor, IV and mark.
    leg = next(l for l in result.option_legs if l.key == f"AAPL#{option_id}")
    position = next(p for p in result.positions if p.key == f"AAPL#{option_id}")
    assert (position.strike, position.right, position.t_years,
            position.iv0, position.mark0) == (
        leg.strike, leg.right, leg.t_years, leg.iv0, leg.mark0
    )
    assert leg.iv0 is not None and leg.t_years > 0

    # The persisted row carries the same string (design §10.3).
    async with SessionLocal() as session:
        row = (
            await session.execute(select(RiskSnapshotRow).order_by(RiskSnapshotRow.id))
        ).scalars().all()[-1]
        assert row.pnl_method == "FULL_REVAL_CONST_IV"
        assert row.data_quality["pnl_method_by_key"] == dq["pnl_method_by_key"]


async def test_option_without_chain_iv_stays_delta_linear_and_says_so(client):
    """The honest fallback, through the REAL path: strip IV off today's
    chain and the same seeded call is priced DELTA_LINEAR and labelled —
    the position is NOT dropped and no IV is invented."""
    import dataclasses

    from apps.gateway.routers import options as options_router

    from .test_risk_stress_api import seed_long_call

    option_id = await seed_long_call(client)
    real_chain = options_router.option_chain_or_none

    def _chain_without_iv(ticker: str, spot: float):
        chain = real_chain(ticker, spot)
        if chain is None:
            return None
        return [dataclasses.replace(c, iv=None) for c in chain]

    options_router.option_chain_or_none = _chain_without_iv
    try:
        result = await build()
    finally:
        options_router.option_chain_or_none = real_chain

    api = result.api
    assert api["pnl_method"] == "DELTA_LINEAR"
    assert api["data_quality"]["pnl_method_by_key"] == {
        f"AAPL#{option_id}": "DELTA_LINEAR"
    }
    # The position is still IN the book — the gap degrades the estimator,
    # never the coverage.
    assert api["positions_excluded"] == []
    assert f"AAPL#{option_id}" in result.book.per_position
    # And the leg reached the builder with an honest null IV, not a guess.
    leg = next(l for l in result.option_legs if l.key == f"AAPL#{option_id}")
    assert leg.iv0 is None


async def test_euler_es_contributions_still_sum_exactly_on_a_mixed_book(client):
    """§10.4: the Euler property is method-agnostic — Σ RC == ES holds on a
    book whose rows were priced by DIFFERENT estimators."""
    from .test_risk_stress_api import seed_long_call

    await seed_stock_position(bars=260)
    await seed_long_call(client)
    api = (await build()).api

    assert api["pnl_method"] == "FULL_REVAL_CONST_IV"
    assert set(api["data_quality"]["pnl_method_by_key"].values()) == {
        "DELTA_LINEAR", "FULL_REVAL_CONST_IV",
    }

    for block, total_key in (
        (api["contributions"]["es"], "total_usd"),
        (api["contributions"]["vol"], "total_usd"),
    ):
        rows = block["rows"]
        assert len(rows) == 2
        assert math.fsum(r["contribution_usd"] for r in rows) == pytest.approx(
            block[total_key], rel=1e-9
        )
        assert math.fsum(r["share"] for r in rows) == pytest.approx(1.0, rel=1e-9)

    # The ES the contributions decompose IS the served ES-95 1-day number.
    es95 = await _es95(api)
    assert api["contributions"]["es"]["total_usd"] == pytest.approx(
        es95["value_usd"], rel=1e-12
    )
    # Every row keeps its BARE position key, so ticker/instrument/capital
    # weight still resolve — the reason a spread is not split into two
    # PositionRiskInputs (design §10.3).
    for row in api["contributions"]["es"]["rows"]:
        assert ":" not in row["key"]
        assert row["ticker"] and row["instrument"]
        assert row["capital_weight"] is not None


async def test_option_leg_fields_by_key_skips_spread_legs_and_dead_legs(client):
    """The §10.3 mapping rule, unit-tested on the leg types directly: a
    spread's two suffixed legs are skipped (one position = one book row), an
    IV-less leg is skipped, an expired leg is skipped, and a live single leg
    passes its five fields through unchanged."""
    from libs.trading_core.options.reval import OptionLeg

    def leg(key, **over):
        fields = dict(
            key=key, ticker="AAPL", right="C", strike=180.0, t_years=0.25,
            quantity=1, spot0=200.0, mark0=7.5, iv0=0.35, delta0=0.6,
        )
        fields.update(over)
        return OptionLeg(**fields)

    mapping = rs.option_leg_fields_by_key([
        leg("AAPL#1"),
        leg("AAPL#2:long"),
        leg("AAPL#2:short", quantity=-1, strike=190.0),
        leg("AAPL#3", iv0=None),
        leg("AAPL#4", t_years=0.0),
    ])

    assert set(mapping) == {"AAPL#1"}
    assert mapping["AAPL#1"] == {
        "strike": 180.0, "right": "C", "t_years": 0.25,
        "iv0": 0.35, "mark0": 7.5,
    }


async def test_a_raising_leg_layer_degrades_to_the_old_numbers(client):
    """SHADOW: when the NEW path raises, the build falls back to the
    pre-batch DELTA_LINEAR series — never a 500, never a changed number.

    §40 SCOPE NOTE. ``dispersion`` is deliberately EXCLUDED from the
    comparison since the stress worst loss became one of the views it
    compares (compliance §3 Tier C): the legs that raise here are the same
    legs the stress run needs, so a broken leg layer legitimately removes a
    dispersion view. Asserting it unchanged would assert the §40 wiring
    does not exist. The dependency is pinned POSITIVELY below instead —
    both that the ratio moves and that the STATISTICAL views behind it do
    not — which is the stronger statement.
    """
    await seed_stock_position(bars=200)
    real = rs.stress_legs_from_book

    def _boom(pairs, greeks_rows):
        raise RuntimeError("chain resolution exploded")

    rs.stress_legs_from_book = _boom
    try:
        broken = await build(persist=False)
    finally:
        rs.stress_legs_from_book = real
    healthy = await build(persist=False)

    assert broken.api["pnl_method"] == "DELTA_LINEAR"
    broken_fp = _numbers_fingerprint(broken.api)
    healthy_fp = _numbers_fingerprint(healthy.api)
    assert broken_fp.pop("dispersion") is not None
    healthy_fp.pop("dispersion")
    # Every statistical number the leg layer feeds is byte-identical.
    assert broken_fp == healthy_fp
    # ...and the ONLY thing the broken leg layer cost the dispersion block is
    # the stress view: the statistical min/max it compares are unchanged.
    assert broken.api["dispersion"]["max_model"] != rs.DISPERSION_STRESS_KEY
    assert healthy.api["dispersion"]["max_model"] == rs.DISPERSION_STRESS_KEY


# ---------------------------------------------------------------------------
# (j) Pre-trade candidates carry their selected contract's leg (design §10.3)
# ---------------------------------------------------------------------------


async def _spy_candidate_specs(client, ticker, monkeypatch):
    """Run one preview and return the ``CandidateSpec``s it built."""
    from apps.gateway.routers import orders as orders_router

    seen = []
    real = gate_chain._candidate_spec

    def _spy(*args, **kwargs):
        spec = real(*args, **kwargs)
        seen.append(spec)
        return spec

    monkeypatch.setattr(gate_chain, "_candidate_spec", _spy)
    r = await client.post("/api/orders/preview", json={"ticker": ticker})
    assert r.status_code == 200, r.text
    return r.json(), seen


async def test_option_candidate_carries_the_selected_contract_leg(
    client, monkeypatch
):
    """§10.3: an option candidate's PositionRiskInput gains the SELECTED
    contract's strike/right/tenor/IV/mark, so the incremental ES and the
    ES-share caps see its convexity instead of a straight delta line."""
    from .test_option_execution import CALL_TICKER
    from .test_order_preview import authorize

    await authorize(client, CALL_TICKER)
    body, specs = await _spy_candidate_specs(client, CALL_TICKER, monkeypatch)

    assert body["proposed"]["instrument"] == "LONG_CALL", body["proposed"]
    assert specs, "the preview built no candidate spec"
    spec = specs[-1]
    assert spec.multiplier == 100
    assert spec.strike is not None and spec.strike > 0
    assert spec.right in ("C", "P")
    assert spec.t_years is not None and spec.t_years > 0
    assert spec.iv0 is not None and spec.iv0 > 0
    assert spec.mark0 is not None

    # THE SAME CONTRACT the §9 selector proposed and the stress legs use —
    # one chain read behind the rationale, the stress rows and the P&L.
    contract = body["proposed"]["contract"]
    assert spec.strike == pytest.approx(contract["strike"])
    assert spec.right == contract["right"]
    assert spec.iv0 == pytest.approx(contract["iv"])
    assert spec.mark0 == pytest.approx(contract["mid"])
    assert spec.t_years == pytest.approx(max(contract["dte"], 0) / 365.0)

    # The fields survive into the position the P&L series actually prices,
    # at every quantity the cap search tries.
    for quantity in (0, 1, 7):
        position = spec.position_at(quantity)
        assert position.can_full_reval is True
        assert position.pnl_method == "FULL_REVAL_CONST_IV"
        assert position.quantity == quantity


async def test_stock_candidate_carries_no_leg_fields_and_stays_linear(
    client, monkeypatch
):
    """A stock candidate has no contract, so every leg field stays None and
    its series is the pre-batch DELTA_LINEAR one — byte for byte."""
    from .test_order_preview import BULL_TICKER, authorize

    await authorize(client, BULL_TICKER)
    body, specs = await _spy_candidate_specs(client, BULL_TICKER, monkeypatch)

    assert body["proposed"]["instrument"] == "LONG_STOCK", body["proposed"]
    spec = specs[-1]
    assert (spec.strike, spec.right, spec.t_years, spec.iv0, spec.mark0) == (
        None, None, None, None, None,
    )
    position = spec.position_at(10)
    assert position.can_full_reval is False
    assert position.pnl_method == "DELTA_LINEAR"


async def test_spread_candidate_carries_the_long_leg_under_one_key(
    client, monkeypatch
):
    """The documented single-key choice (design §10.3): a net-debit spread is
    ONE candidate key carrying its LONG leg's contract and the spread's NET
    delta. Two keys would break ``proposed_book`` /
    ``share_of(candidate.key)``, so the short leg's offsetting convexity is
    knowingly unmodelled — an upper bound, the conservative direction."""
    from libs.trading_core.volatility import VolRegimeParams

    from apps.gateway.routers import orders as orders_router
    from .test_spread_execution import authorize as spread_authorize

    monkeypatch.setattr(gate_chain, "VOL_REGIME_PARAMS",
        VolRegimeParams(low_iv=0.0001, high_iv=99.0, extreme_iv=999.0),
    )
    r = await client.put(
        "/api/config/providers", json={"allow_defined_risk_spreads": "true"}
    )
    assert r.status_code == 200
    await spread_authorize(client, "GW")

    body, specs = await _spy_candidate_specs(client, "GW", monkeypatch)
    assert body["proposed"]["instrument"] == "BULL_CALL_SPREAD", body["proposed"]

    spec = specs[-1]
    assert spec.key.endswith("#candidate")  # ONE key, no :long / :short split
    assert ":" not in spec.key
    # The LONG leg's contract, with the spread's NET delta beside it.
    spread = body["proposed"]["spread"]
    assert spec.strike == pytest.approx(spread["long_strike"])
    assert spec.right == spread["right"] == "C"
    assert spec.mark0 == pytest.approx(spread["long_mid"])
    assert spec.t_years == pytest.approx(max(spread["dte"], 0) / 365.0)
    assert spec.iv0 is not None and spec.iv0 > 0
    # ...and NOT the short leg's — the single-key choice is documented, not
    # accidental, and this pins WHICH leg was chosen.
    assert spec.strike != pytest.approx(spread["short_strike"])
    assert spec.mark0 != pytest.approx(spread["short_mid"])
    assert spec.position_at(1).pnl_method == "FULL_REVAL_CONST_IV"
    # The delta is the NET of both legs — strictly smaller than the long
    # leg's own delta, which is what makes it the spread's and not the
    # long call's.
    assert spec.delta == pytest.approx(spread["net_delta"])
    assert 0.0 < spec.delta < 1.0
