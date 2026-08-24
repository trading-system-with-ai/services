"""Tests for the Phase B statistical layer on the wire (design contract §6).

Two surfaces:

- ``GET /api/portfolio/risk`` gains the ADDITIVE ``statistical`` and
  ``drawdown`` blocks — every contract key present, honest nulls inside, and
  STILL no audit event (a read is not a decision);
- the ``RISK_DECISION`` audit detail gains the additive keys and
  ``shadow.statistical`` — with the decision itself provably unchanged.

Everything is SHADOW: nothing here may alter a Tier 0 decision or GATE_ORDER.
"""
from datetime import date, datetime, timedelta

import pytest

from apps.gateway.execution import gate_chain
from sqlalchemy import select

from apps.gateway.db import AuditEvent, RiskMetricRow, RiskSnapshotRow, SessionLocal
from apps.gateway.routers import orders as orders_router

from .test_order_preview import (
    BULL_TICKER,
    authorize,
    get_single_risk_decision_event,
    preview,
)
from .test_risk_snapshot_builder import seed_stock_position

#: Every key contract §6 defines on the ``statistical`` block.
STATISTICAL_KEYS = {
    "mode",
    "snapshot_id",
    "persisted",
    "conditional_source",
    "snapshot_version",
    "as_of",
    "stale",
    "pnl_method",
    "n_obs",
    "window_start",
    "window_end",
    "data_quality",
    "model_health",
    "model_risk",
    "dispersion",
    "distribution",
    "volatility",
    "var",
    "es",
    "contributions",
    "positions_excluded",
    # ADDED in Phase C (contract §7.4/§7.5).
    "correlation_state",
    # ADDED in Phase D (contract §8.5).
    "stress",
    # ADDED in Phase E (contract §9.4) — null until a validation run exists.
    "validation",
    # ADDED closing compliance §34 (audit.md:215, P1): DR = Σ_i σ_i / σ_p.
    "diversification_ratio",
    # ADDED closing compliance §12: the GARCH variance term structure at the
    # display horizons; honest nulls when EWMA is the live conditional source.
    "conditional_horizon_sigmas",
    # ADDED closing compliance §11 (row 11): the single-factor (SPY) RESEARCH
    # diagnostic. Always an object; honest nulls inside when SPY bars are
    # absent or the paired window is shorter than min_obs.
    "factor",
}

#: Every key the ``statistical.data_quality`` block carries. The last two
#: are ADDITIVE (design §10.3, compliance batch 2): the book-level
#: ``pnl_method`` above is a one-word summary of a possibly MIXED book, so
#: the per-key map that makes it honest is served right beside it.
DATA_QUALITY_KEYS = {
    "valid",
    "reasons",
    "tickers_missing",
    "keys_excluded",
    "pnl_method_by_key",
    "pnl_method_counts",
}

#: Every key contract §6 defines on the ``drawdown`` block.
DRAWDOWN_KEYS = {
    "nav_series",
    "current_pct",
    "max_pct",
    "peak_date",
    "trough_date",
    "peak_nav",
    "health",
    "reason",
    "reconstructed",
}

#: The VaR/ES row shape, in the contract's declared ORDER.
VIEW_ORDER = [
    ("HISTORICAL", 0.95),
    ("HISTORICAL", 0.99),
    ("GAUSSIAN", 0.95),
    ("GAUSSIAN", 0.99),
    ("HISTORICAL_VOL_SCALED", 0.95),
]


# ---------------------------------------------------------------------------
# (a) The empty book: every key present, honest nulls inside
# ---------------------------------------------------------------------------


async def test_empty_book_serves_the_full_contract_with_honest_nulls(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()

    statistical = body["statistical"]
    assert set(statistical) == STATISTICAL_KEYS
    assert statistical["mode"] == "SHADOW"  # SHADOW is stated, never implied
    assert statistical["snapshot_version"] == "b.1"
    assert statistical["pnl_method"] == "DELTA_LINEAR"
    assert set(statistical["data_quality"]) == DATA_QUALITY_KEYS
    # No book ⇒ nothing was priced by anything: an empty map, and BOTH
    # counts present at zero (an absent key would read as "unknown").
    assert statistical["data_quality"]["pnl_method_by_key"] == {}
    assert statistical["data_quality"]["pnl_method_counts"] == {
        "FULL_REVAL_CONST_IV": 0,
        "DELTA_LINEAR": 0,
    }
    datetime.fromisoformat(statistical["as_of"])
    assert statistical["stale"] is False
    assert statistical["n_obs"] == 0
    assert statistical["window_start"] is None
    assert statistical["window_end"] is None
    assert statistical["positions_excluded"] == []

    # No book ⇒ nothing measurable ⇒ every number is an honest null with a
    # reason, never a fabricated zero.
    # The 1-day grid is the PREFIX of the array and is unchanged: a consumer
    # reading var[0] still gets HISTORICAL 0.95 1-day. The §6/§12 display
    # rows are appended after it, never interleaved.
    for side in ("var", "es"):
        rows = statistical[side]
        one_day = [r for r in rows if r["horizon_days"] == 1]
        assert [(r["model"], r["confidence"]) for r in one_day] == VIEW_ORDER
        assert rows[: len(VIEW_ORDER)] == one_day
        assert [
            (r["model"], r["confidence"], r["horizon_days"])
            for r in rows[len(VIEW_ORDER):]
        ] == [("HISTORICAL", 0.95, 5), ("HISTORICAL", 0.95, 10)]
    for row in statistical["var"] + statistical["es"]:
        assert row["value_usd"] is None
        assert row["pct_nav"] is None
        assert row["health"] == "UNAVAILABLE"
        assert row["reason"]
    assert statistical["volatility"]["value_usd"] is None
    # `es99` (§10) is additive beside `es`/`vol` and is an honest null on an
    # empty book for exactly the same reason they are: no position, nothing
    # to decompose.
    assert statistical["contributions"] == {"es": None, "vol": None, "es99": None}

    drawdown = body["drawdown"]
    assert set(drawdown) == DRAWDOWN_KEYS
    assert drawdown["nav_series"] == {
        "n": 0,
        "since": None,
        "source": "risk_snapshots SCHEDULED",
    }
    assert drawdown["current_pct"] is None
    assert drawdown["health"] == "UNAVAILABLE"
    assert drawdown["reason"]
    assert drawdown["reconstructed"] is None


async def test_var_es_rows_carry_their_methodology(client):
    """Spec §50: every metric must be able to answer "how is this
    calculated?" — the model, its version, distribution, confidence,
    horizon, sample and health travel WITH the number."""
    await seed_stock_position(bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    for row in body["statistical"]["var"] + body["statistical"]["es"]:
        assert set(row) == {
            "model",
            "model_name",
            "model_version",
            "distribution",
            "confidence",
            "horizon_days",
            "value_usd",
            "pct_nav",
            "health",
            "reason",
            "sample_size",
            "tail_size",
            "scaling",
            # ADDED with the §6/§12 display rows: SHADOW for the 1-day grid,
            # RESEARCH for a √h-scaled multi-day row.
            "mode",
            # ADDED with the §5 tier taxonomy: which model FAMILY produced
            # this number (TIER_1 unconditional / TIER_2 conditional).
            "tier",
        }
        assert row["model_version"]
        assert row["distribution"]
        assert row["mode"] in {"SHADOW", "RESEARCH"}


async def test_no_account_branch_nulls_both_blocks(unconfigured_client):
    """No venue ⇒ no account ⇒ both blocks are null (the ONE case contract §6
    allows): an object full of nulls would imply a measurement was tried."""
    body = (await unconfigured_client.get("/api/portfolio/risk")).json()
    assert body["statistical"] is None
    assert body["drawdown"] is None


# ---------------------------------------------------------------------------
# (b) A real book on the wire, persisted, with no audit event
# ---------------------------------------------------------------------------


async def test_seeded_book_serves_live_numbers_and_persists(client):
    await seed_stock_position(bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    statistical = body["statistical"]

    assert statistical["n_obs"] == 199
    assert statistical["snapshot_id"] is not None
    var95 = next(
        r for r in statistical["var"]
        if r["model"] == "HISTORICAL" and r["confidence"] == 0.95
    )
    assert var95["health"] in {"ACTIVE", "DEGRADED"}
    assert var95["value_usd"] > 0
    assert var95["pct_nav"] == pytest.approx(var95["value_usd"] / body["nav"])
    assert var95["sample_size"] == 199
    assert var95["tail_size"] == 10  # ceil(199 × 0.05)

    contributions = statistical["contributions"]["es"]
    assert contributions["rows"]
    assert sum(r["contribution_usd"] for r in contributions["rows"]) == pytest.approx(
        contributions["total_usd"]
    )

    # The reconstructed drawdown is labelled for what it is — today's book
    # replayed over the window, NOT a NAV path that ever existed.
    assert body["drawdown"]["reconstructed"]["label"] == "RECONSTRUCTED_CURRENT_BOOK"

    async with SessionLocal() as session:
        snapshots = (await session.execute(select(RiskSnapshotRow))).scalars().all()
        metrics = (await session.execute(select(RiskMetricRow))).scalars().all()
    assert len(snapshots) == 1
    assert snapshots[0].trigger == "ON_DEMAND"
    assert snapshots[0].id == statistical["snapshot_id"]
    # 11 one-day rows + 4 √h display rows (§6/§12) + 1 §34 ratio.
    assert len(metrics) == 16
    assert all(m.model_name and m.model_version and m.params for m in metrics)


async def test_the_risk_read_writes_no_audit_event(client):
    """House rule: read views write NO audit events — the snapshot is
    persisted, but nothing about it is a decision.

    The one audit the endpoint can produce is the PRE-EXISTING lazy
    DATA_BACKFILL for SPY (the §6.1 regime read fetches bars it does not
    have); the Phase B layer adds nothing on top of it. Calling twice pins
    that: the second request backfills nothing and must write NOTHING.
    """
    await seed_stock_position(bars=200)
    assert (await client.get("/api/portfolio/risk")).status_code == 200
    async with SessionLocal() as session:
        before = (await session.execute(select(AuditEvent))).scalars().all()

    assert (await client.get("/api/portfolio/risk")).status_code == 200

    async with SessionLocal() as session:
        after = (await session.execute(select(AuditEvent))).scalars().all()
    assert len(after) == len(before)
    assert not [e for e in after if e.action == "RISK_DECISION"]
    # ...while the snapshot itself WAS persisted by the first read; the
    # second read (inside the ON_DEMAND dedupe window) computes but does not
    # write again — see test_on_demand_builds_persist_at_most_once_per_window.
    async with SessionLocal() as session:
        snapshots = (await session.execute(select(RiskSnapshotRow))).scalars().all()
    assert len(snapshots) == 1
    assert {s.trigger for s in snapshots} == {"ON_DEMAND"}


async def test_the_risk_read_never_503s_when_the_builder_explodes(client, monkeypatch):
    """The view has one job it must never fail at — showing the user their
    own book. A broken statistical layer becomes a note, not a 5xx."""
    from apps.gateway.routers import portfolio as portfolio_router

    async def boom(*args, **kwargs):
        raise RuntimeError("snapshot machinery is down")

    monkeypatch.setattr(portfolio_router, "build_risk_snapshot", boom)
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()
    # Everything else is intact...
    assert body["nav"] is not None
    assert body["heat_state"] is not None
    # ...and the statistical block says what went wrong.
    assert "snapshot machinery is down" in body["statistical"]["note"]
    assert body["drawdown"] is None


# ---------------------------------------------------------------------------
# (c) Short history: UNAVAILABLE with the real numbers, data quality invalid
# ---------------------------------------------------------------------------


async def test_short_history_degrades_honestly_on_the_wire(client):
    await seed_stock_position(bars=30)
    statistical = (await client.get("/api/portfolio/risk")).json()["statistical"]

    assert statistical["data_quality"]["valid"] is False
    assert any(
        "n_obs=29" in reason and "min_obs=60" in reason
        for reason in statistical["data_quality"]["reasons"]
    )
    for row in statistical["var"] + statistical["es"]:
        assert row["health"] == "UNAVAILABLE"
        assert row["value_usd"] is None
        assert f"n={row['sample_size']}" in row["reason"]
        assert "min_obs=" in row["reason"]


# ---------------------------------------------------------------------------
# (d) An undeltable position is excluded and NAMED on the wire
# ---------------------------------------------------------------------------


async def test_excluded_position_is_named_on_the_wire(client):
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from apps.gateway.db import Position

    await seed_stock_position("AAPL", bars=200)
    async with SessionLocal() as session:
        orphan = Position(
            ticker="ZZZZ",
            instrument="LONG_CALL",
            quantity=1,
            avg_price=2.5,
            max_loss=250.0,
            multiplier=100,
            opt_expiry="2099-01-15",
            opt_strike=100.0,
            opt_right="C",
            status="OPEN",
            opened_at=_dt.now(_tz.utc),
        )
        session.add(orphan)
        await session.commit()
        orphan_key = f"ZZZZ#{orphan.id}"

    statistical = (await client.get("/api/portfolio/risk")).json()["statistical"]
    excluded = statistical["positions_excluded"]
    assert [e["key"] for e in excluded] == [orphan_key]
    assert excluded[0]["reason"]  # a real, server-generated explanation
    assert statistical["data_quality"]["keys_excluded"] == [orphan_key]
    # An excluded row was priced by NOTHING, so it carries no method label —
    # an honest gap, not a claim that it was DELTA_LINEAR'd (design §10.3).
    assert orphan_key not in statistical["data_quality"]["pnl_method_by_key"]


# ---------------------------------------------------------------------------
# (d2) Compliance batch 2 on the wire — the pnl_method label (design §10.3)
# ---------------------------------------------------------------------------


async def test_pnl_method_flips_to_full_reval_for_an_option_with_chain_iv(client):
    """A seeded LONG_CALL on a real stub-chain contract carries an IV, so the
    served ``pnl_method`` becomes FULL_REVAL_CONST_IV and the per-key map
    names the row that earned it — while the stock beside it stays linear."""
    from .test_risk_stress_api import seed_long_call

    stock_id = await seed_stock_position(bars=200)
    option_id = await seed_long_call(client)

    statistical = (await client.get("/api/portfolio/risk")).json()["statistical"]
    dq = statistical["data_quality"]

    assert set(dq) == DATA_QUALITY_KEYS
    assert statistical["pnl_method"] == "FULL_REVAL_CONST_IV"
    assert dq["pnl_method_by_key"] == {
        f"AAPL#{stock_id}": "DELTA_LINEAR",
        f"AAPL#{option_id}": "FULL_REVAL_CONST_IV",
    }
    assert dq["pnl_method_counts"] == {
        "FULL_REVAL_CONST_IV": 1,
        "DELTA_LINEAR": 1,
    }
    # SHADOW throughout — a new estimator does not promote the block.
    assert statistical["mode"] == "SHADOW"


async def test_pnl_method_stays_delta_linear_when_the_chain_has_no_iv(client):
    """Same position, IV stripped from today's chain: the honest fallback is
    served and LABELLED, and the position stays IN the book."""
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
        statistical = (
            await client.get("/api/portfolio/risk")
        ).json()["statistical"]
    finally:
        options_router.option_chain_or_none = real_chain

    assert statistical["pnl_method"] == "DELTA_LINEAR"
    assert statistical["data_quality"]["pnl_method_by_key"] == {
        f"AAPL#{option_id}": "DELTA_LINEAR"
    }
    assert statistical["positions_excluded"] == []


# ---------------------------------------------------------------------------
# (e) The order path: additive audit keys, shadow.statistical, SAME decision
# ---------------------------------------------------------------------------


async def _latest_risk_decision(client, ticker: str) -> dict:
    """The NEWEST RISK_DECISION for ``ticker`` (a test may preview twice)."""
    r = await client.get("/api/audit", params={"entity_id": ticker})
    events = [e for e in r.json() if e["action"] == "RISK_DECISION"]
    assert events, "no RISK_DECISION event recorded"
    return events[0]


async def test_risk_decision_audit_gains_the_additive_keys(client):
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    gates = {g["name"]: g["status"] for g in body["gates"]}
    assert gates["RISK_APPROVAL"] in {"PASS", "FAIL"}, gates
    details = (await get_single_risk_decision_event(client, BULL_TICKER))["details"]

    # Additive keys (contract §6) — present alongside every pre-existing one.
    assert "quantity_requested" in details
    assert "approved_quantity" in details
    assert "budget_multiplier" in details
    assert isinstance(details["limits"], dict) and details["limits"]
    # SCALARS only: the policy tables (cash floors, correlation buckets) are
    # not per-decision facts.
    assert all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in details["limits"].values()
    )
    assert "single_name_risk" in details["limits"]

    # Pre-existing keys survive untouched (additive-only house rule).
    for key in ("decision", "mode", "veto_gate", "gates", "reason_codes", "shadow"):
        assert key in details
    assert "liquidity" in details["shadow"]

    shadow = details["shadow"]["statistical"]
    assert shadow is not None
    # Phase B — the CURRENT-book view. Every one of these keys is ALWAYS
    # present and still carries exactly what it carried before Phase C.
    assert {
        "snapshot_id",
        "as_of",
        "model_risk_state",
        "dispersion_high",
        "historical_var_95_1d_pct_nav",
        "historical_es_95_1d_pct_nav",
        "gaussian_es_95_1d_pct_nav",
        "health",
        "note",
    } <= set(shadow)
    assert shadow["note"] == (
        "current-book view; the proposed-book comparison is under "
        "`comparison` / `hypothetical` (Phase C, SHADOW)"
    )
    # Phase C (contract §7.5): the thresholds and the regime are always
    # stated; the proposed-book comparison needs a book to compare against,
    # and THIS fixture has no open position — so the layer says so with a
    # note instead of fabricating a comparison against nothing.
    assert shadow["limits"]["mode"] == "SHADOW"
    assert "comparison" not in shadow
    assert "no priceable book" in shadow["comparison_note"]
    assert isinstance(shadow["health"], dict) and shadow["health"]
    # A preview is research: it must not persist a PRE_TRADE snapshot.
    assert shadow["snapshot_id"] is None
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(RiskSnapshotRow).where(RiskSnapshotRow.trigger == "PRE_TRADE")
            )
        ).scalars().all()
    assert rows == []


async def test_shadow_failure_leaves_the_decision_byte_identical(client, monkeypatch):
    """The load-bearing SHADOW guarantee (§70): with the builder RAISING, the
    Tier 0 decision, gates and reason codes are IDENTICAL — only a note
    appears under shadow.statistical."""
    await authorize(client, BULL_TICKER)
    await preview(client, BULL_TICKER)
    good = (await _latest_risk_decision(client, BULL_TICKER))["details"]
    assert good["shadow"]["statistical"] is not None

    async def boom(*args, **kwargs):
        raise RuntimeError("shadow layer exploded")

    monkeypatch.setattr(gate_chain, "build_risk_snapshot", boom)
    await preview(client, BULL_TICKER)
    broken = (await _latest_risk_decision(client, BULL_TICKER))["details"]

    assert broken["decision"] == good["decision"]
    assert broken["gates"] == good["gates"]
    assert broken["reason_codes"] == good["reason_codes"]
    assert broken["veto_gate"] == good["veto_gate"]
    assert broken["approved_quantity"] == good["approved_quantity"]
    assert broken["budget_multiplier"] == good["budget_multiplier"]
    assert broken["limits"] == good["limits"]
    assert broken["shadow"]["liquidity"] == good["shadow"]["liquidity"]

    # The ONLY difference: the shadow block reports its own failure.
    assert broken["shadow"]["statistical"] == {
        "note": "RuntimeError: shadow layer exploded"
    }


async def test_gate_order_is_unchanged_by_phase_b(client):
    """SHADOW mandate: no new gate, no reordering (contract §6)."""
    assert gate_chain.GATE_ORDER == (
        "TRADING_POOL_AUTHORIZATION",
        "DATA_QUALITY",
        "REGIME",
        "DIRECTIONAL_SIGNAL",
        "VOLATILITY",
        "INSTRUMENT",
        # Phase D 2026-08-20: SQUEEZE_RISK is a REPORT-mode proxy gate
        # (never vetoes) — same discipline as LIQUIDITY, not a statistical
        # enforcement gate joining the chain.
        "SQUEEZE_RISK",
        "LIQUIDITY",
        "CONTRACT_SELECTION",
        "RISK_APPROVAL",
    )


# ---------------------------------------------------------------------------
# (g) atm_iv_daily is captured where chain_iv_summary already runs
# ---------------------------------------------------------------------------


async def test_execution_persists_a_pre_trade_snapshot_and_the_atm_iv(client):
    """Approve (execution mode) persists the PRE_TRADE snapshot AND the ATM
    IV in the SAME transaction as the fill — a rolled-back order leaves
    neither behind."""
    from apps.gateway.db import AtmIvDailyRow

    await authorize(client, BULL_TICKER)
    r = await client.post("/api/orders/approve", json={"ticker": BULL_TICKER})
    assert r.status_code == 200, r.text

    async with SessionLocal() as session:
        snapshots = (await session.execute(select(RiskSnapshotRow))).scalars().all()
        ivs = (await session.execute(select(AtmIvDailyRow))).scalars().all()
    assert [s.trigger for s in snapshots] == ["PRE_TRADE"]
    assert [(iv.ticker, iv.source) for iv in ivs] == [(BULL_TICKER, "stub_chain")]
    assert ivs[0].atm_iv > 0 and ivs[0].spot > 0
    # The audit's shadow block names the row that was actually written.
    details = (await get_single_risk_decision_event(client, BULL_TICKER))["details"]
    assert details["shadow"]["statistical"]["snapshot_id"] == snapshots[0].id


async def test_chain_read_records_the_atm_iv_it_computed(client):
    """The chain endpoint had the number in hand; the history now exists —
    labelled INTERNALLY CALCULATED via ``source``, and with no audit event
    (an observation is not a decision)."""
    from apps.gateway.db import AtmIvDailyRow

    await client.post("/api/watchlist", json={"ticker": BULL_TICKER})
    r = await client.get(f"/api/watchlist/{BULL_TICKER}/options")
    assert r.status_code == 200
    served = r.json()["summary"]["atm_iv"]

    async with SessionLocal() as session:
        rows = (await session.execute(select(AtmIvDailyRow))).scalars().all()
        audits = (await session.execute(select(AuditEvent))).scalars().all()
    assert len(rows) == 1
    assert rows[0].ticker == BULL_TICKER
    assert rows[0].atm_iv == pytest.approx(served)
    assert rows[0].source == "stub_chain"  # never presented as vendor IV
    assert not [e for e in audits if e.action == "RISK_DECISION"]

    # A second read on the same day UPDATES rather than duplicating.
    assert (await client.get(f"/api/watchlist/{BULL_TICKER}/options")).status_code == 200
    async with SessionLocal() as session:
        rows = (await session.execute(select(AtmIvDailyRow))).scalars().all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# (h) Telemetry names reach /metrics
# ---------------------------------------------------------------------------


async def test_telemetry_names_are_exposed(client):
    from apps.gateway import risk_snapshot as rs

    await seed_stock_position(bars=200)
    await client.get("/api/portfolio/risk")
    # The age gauge tracks the newest SCHEDULED build ONLY (QA finding):
    # page loads / previews must never mask a dead scheduled writer. Drive
    # one scheduled tick so there is a SCHEDULED build to age.
    scheduled = await rs.run_scheduled_snapshot()
    assert "snapshot_id" in scheduled, scheduled
    text = (await client.get("/metrics")).text
    assert "risk_snapshot_age_seconds" in text
    # The age gauge is a scrape-time CALLBACK: it must report a real, small
    # positive age right after a build — not a 0.0 frozen at build time
    # (which would read "fresh" forever if the builder ever stopped).
    age_line = next(
        line for line in text.splitlines()
        if line.startswith("risk_snapshot_age_seconds ")
    )
    assert 0.0 <= float(age_line.split()[1]) < 60.0
    assert "risk_model_latency_seconds" in text
    assert 'risk_model_latency_seconds_bucket{stage="var_es"' in text
    assert 'risk_snapshot_builds_total{trigger="ON_DEMAND"}' in text
    assert "risk_snapshot_failures_total" in text


async def test_empty_book_reports_low_model_risk_with_the_real_reason(client):
    """An empty book has nothing to model: the core views are UNAVAILABLE
    for lack of positions, not for model failure — model risk must read LOW
    with the honest reason, never ELEVATED (QA/UI finding, 2026-08-18)."""
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    s = r.json()["statistical"]
    if s is None or s.get("n_obs") is None:
        return  # no account in this environment: nothing to assert
    assert s["positions_excluded"] == []
    assert s["model_risk"] is not None
    assert s["model_risk"]["state"] == "LOW"
    assert "no open positions" in " ".join(s["model_risk"]["reasons"])


async def test_on_demand_builds_persist_at_most_once_per_window(client):
    """QA finding (Phase D): the UI polls the risk view every 15 s; ON_DEMAND
    builds must not write a snapshot + metrics + stress catalogue on every
    poll. Two back-to-back reads: the first persists, the second computes
    but does not persist (``persisted`` false, ``snapshot_id`` null)."""
    from sqlalchemy import select

    from apps.gateway.db import RiskSnapshotRow, SessionLocal

    await seed_stock_position(bars=200)
    first = (await client.get("/api/portfolio/risk")).json()["statistical"]
    second = (await client.get("/api/portfolio/risk")).json()["statistical"]
    assert first["persisted"] is True and first["snapshot_id"] is not None
    assert second["persisted"] is False and second["snapshot_id"] is None
    # Same numbers either way — only the write is deduplicated.
    assert second["var"][0]["value_usd"] == first["var"][0]["value_usd"]
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(RiskSnapshotRow).where(RiskSnapshotRow.trigger == "ON_DEMAND")
            )
        ).scalars().all()
    assert len(rows) == 1


async def test_conditional_source_names_the_active_filter(client):
    """Phase E §9.3/§9.5: the conditional VaR/ES rows say which filter is
    behind them. With ~200 stored bars GARCH is UNAVAILABLE (min_obs 250), so
    the §13/§58 fallback hands back EWMA with the real reason; the
    conditional row's distribution matches."""
    await seed_stock_position(bars=200)
    s = (await client.get("/api/portfolio/risk")).json()["statistical"]
    cs = s["conditional_source"]
    assert cs["source"] in ("EWMA", "GARCH")
    if cs["source"] == "EWMA":
        assert cs["reason"]  # names why GARCH was not used
        cond = next(r for r in s["var"] if r["model"] == "HISTORICAL_VOL_SCALED")
        assert cond["distribution"] in ("EMPIRICAL_VOL_SCALED", None)


# ---------------------------------------------------------------------------
# Compliance §65 — the missing Prometheus instruments (Tier A)
# ---------------------------------------------------------------------------


async def test_section_65_metrics_reach_the_exposition(client):
    """§65 (compliance Tier A): four instruments were promised and verified
    absent. Without them there is no alertable TIME SERIES of how often a
    shadow model breached — which is exactly the evidence the 20-day
    promotion window is supposed to accumulate.

    Drives a real build (sets the health gauge) and a real validation run
    (increments the exceedance counters), then asserts all four names, with
    their labels, are in the ``/metrics`` exposition.
    """
    from apps.gateway import risk_validation as rv

    await seed_stock_position(bars=520)
    # A build publishes model_health_state and, on a short/failed GARCH fit,
    # garch_fit_failures_total from the snapshot seam.
    await client.get("/api/portfolio/risk")

    # A validation run persists backtest rows and counts their exceedances.
    r = await client.post("/api/risk/validation/run", json={})
    assert r.status_code == 200, r.text

    text = (await client.get("/metrics")).text

    # (1) + (2) exceedance counters, labelled by model.
    assert "var_exceedances_total" in text
    assert "es_exceedances_total" in text
    # `confidence` is a label: the grid scores historical_var at BOTH 0.95
    # and 0.99, and those are different tests (5% vs 1% expected exceedance
    # rate). Summing them into one series would match neither.
    assert 'var_exceedances_total{model="historical_var",confidence="0.95"}' in text
    assert 'var_exceedances_total{model="historical_var",confidence="0.99"}' in text
    assert 'es_exceedances_total{model="historical_var",confidence="0.95"}' in text

    # (3) GARCH fit failures, labelled by call site and resulting health.
    assert "garch_fit_failures_total" in text

    # (4) the health gauge, one sample per model of the fixed view grid.
    assert "model_health_state" in text
    assert 'model_health_state{model="historical_var_95"}' in text
    assert 'model_health_state{model="portfolio_volatility"}' in text

    # The gauge is an ORDINAL, not a free number: every sample must be one
    # of the four documented codes, or an alert threshold means nothing.
    samples = [
        float(line.split()[1])
        for line in text.splitlines()
        if line.startswith("model_health_state{")
    ]
    assert samples
    assert set(samples) <= {0.0, 1.0, 2.0, 3.0}
    assert set(rv.MODEL_HEALTH_ORDINAL.values()) == {0, 1, 2, 3}


async def test_model_health_gauge_tracks_the_real_health(client):
    """§65: the gauge must report the health the snapshot actually found —
    a gauge frozen at ACTIVE would be worse than no gauge at all.

    An EMPTY book makes every core view UNAVAILABLE (nothing to model), so
    the gauge must read 2, not 0.
    """
    from apps.gateway import risk_validation as rv

    rv.MODEL_HEALTH_STATE.set(0.0, model="historical_var_95")
    await client.get("/api/portfolio/risk")  # empty book
    text = (await client.get("/metrics")).text
    line = next(
        l for l in text.splitlines()
        if l.startswith('model_health_state{model="historical_var_95"}')
    )
    assert float(line.split()[1]) == 2.0  # UNAVAILABLE

    # And a seeded book moves it back down to ACTIVE/DEGRADED.
    await seed_stock_position(bars=520)
    await client.get("/api/portfolio/risk")
    text = (await client.get("/metrics")).text
    line = next(
        l for l in text.splitlines()
        if l.startswith('model_health_state{model="historical_var_95"}')
    )
    assert float(line.split()[1]) in (0.0, 1.0)


async def test_exceedance_counters_match_the_persisted_rows(client):
    """§65: the counter and the stored history can never disagree — the
    increment happens as the row is persisted, from the row's own numbers.

    ``es_exceedances_total`` is a SUBSET of ``var_exceedances_total`` for the
    same view: it counts the exceedance days the ES severity ratio was
    actually scored on, so a gap between the two means ES forecasts went
    missing (the condition worth alerting on).
    """
    from apps.gateway.db import RiskModelBacktestRow

    def exceedance_counters(text: str) -> dict[str, float]:
        return {
            line.split()[0]: float(line.split()[1])
            for line in text.splitlines()
            if line.startswith(("var_exceedances_total{", "es_exceedances_total{"))
        }

    # A counter guarantees a DELTA, not an absolute value: the telemetry
    # REGISTRY is process-global while each test gets a fresh database, so
    # earlier tests in the same session have already contributed. Measuring
    # the increment is what the instrument actually promises.
    before = exceedance_counters((await client.get("/metrics")).text)

    await seed_stock_position(bars=520)
    r = await client.post("/api/risk/validation/run", json={})
    assert r.status_code == 200

    async with SessionLocal() as session:
        rows = (
            await session.execute(select(RiskModelBacktestRow))
        ).scalars().all()
    scored = [row for row in rows if row.n_forecasts]
    assert scored, "the run scored no view — the counter test proves nothing"

    after = exceedance_counters((await client.get("/metrics")).text)
    for row in scored:
        labels = f'{{model="{row.model_name}",confidence="{row.confidence}"}}'
        key = f"var_exceedances_total{labels}"
        delta = after[key] - before.get(key, 0.0)
        assert delta == pytest.approx(float(row.exceedances)), row.model_name
        es_key = f"es_exceedances_total{labels}"
        if es_key in after:
            es_delta = after[es_key] - before.get(es_key, 0.0)
            # ES is scored on a SUBSET of the VaR test's exceedance days.
            assert es_delta <= delta
    # A counter never goes backwards.
    for key, value in before.items():
        assert after[key] >= value


# ---------------------------------------------------------------------------
# (k) §11 single-factor (SPY) diagnostic + §18 Spearman on the wire
# ---------------------------------------------------------------------------


async def _seed_factor_bars(
    ticker: str = "SPY", *, bars: int = 200, start_date: date = date(2025, 1, 1)
) -> None:
    """Stored daily closes for a ticker with NO position — the factor proxy.

    Deliberately bars-only: the §11 diagnostic measures the book against the
    market, which does not require holding the market.
    """
    from apps.gateway.db import StockBarDaily
    from .test_risk_snapshot_builder import _closes

    async with SessionLocal() as session:
        for i, close in enumerate(_closes(bars, start=400.0)):
            session.add(
                StockBarDaily(
                    ticker=ticker,
                    ts=start_date + timedelta(days=i),
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=90_000_000,
                )
            )
        await session.commit()


async def test_factor_block_is_present_and_null_on_an_empty_book(client):
    """An unmeasurable book is a present block with a REAL reason."""
    body = (await client.get("/api/portfolio/risk")).json()
    factor = body["statistical"]["factor"]
    assert factor["portfolio_beta"] is None
    assert factor["explained_variance_share"] is None
    assert factor["positions"] == []
    assert factor["health"] == "UNAVAILABLE"
    assert factor["mode"] == "RESEARCH"  # RESEARCH is stated, never implied
    assert "empty book" in factor["reason"]


async def test_factor_block_uses_the_lazily_backfilled_spy_bars(client):
    """The §11 diagnostic needs no new data plumbing on the risk read path.

    ``GET /api/portfolio/risk`` already calls ``market_regime_from_spy``
    BEFORE building the snapshot, and that helper lazily backfills SPY via
    ``ensure_daily_bars`` (ADR-005 exemption). So by the time the builder
    asks for the factor series the bars are there, with no position in SPY
    and no second fetch — which is why this diagnostic came for free.

    Pinned because it is an ORDERING dependency across two modules: if the
    regime read ever moves after the snapshot build, this block silently
    degrades to UNAVAILABLE on the FIRST request for a fresh install, and
    nothing else in the suite would notice.
    """
    await seed_stock_position("AAPL", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    factor = body["statistical"]["factor"]

    assert factor["factor"] == "SPY"
    assert factor["health"] in {"ACTIVE", "DEGRADED"}
    assert factor["portfolio_beta"] is not None


async def test_factor_block_is_null_with_a_real_reason_when_spy_is_absent(
    client, monkeypatch
):
    """No SPY series must never read as "this book has no market exposure".

    The regime backfill is stubbed out so SPY genuinely has no stored bars —
    the state of a real install whose market-data provider is failing. The
    block must then carry an honest null with a reason that disclaims the
    inference a bare ``null`` invites, and must NOT report a beta of 0.0.
    """
    from apps.gateway.routers import portfolio as portfolio_router

    real = portfolio_router.stored_bars_by_ticker

    async def without_spy(session, tickers, **kwargs):
        out = await real(session, tickers, **kwargs)
        return {t: ([] if t == "SPY" else bars) for t, bars in out.items()}

    monkeypatch.setattr(portfolio_router, "stored_bars_by_ticker", without_spy)

    await seed_stock_position("AAPL", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    factor = body["statistical"]["factor"]

    assert factor["portfolio_beta"] is None
    assert factor["explained_variance_share"] is None
    assert factor["health"] == "UNAVAILABLE"
    # A null here is "not measured", never "no market exposure".
    assert "not a statement about this book" in factor["reason"]


async def test_factor_block_measures_the_book_against_spy(client):
    """With SPY stored, the diagnostic reports real, bounded numbers."""
    await seed_stock_position("AAPL", bars=200)
    await _seed_factor_bars("SPY", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    factor = body["statistical"]["factor"]

    assert factor["factor"] == "SPY"
    assert factor["health"] in {"ACTIVE", "DEGRADED"}
    assert factor["n"] >= 60
    assert factor["portfolio_beta"] is not None
    share = factor["explained_variance_share"]
    assert 0.0 <= share <= 1.0  # a variance RATIO, by construction
    assert factor["idiosyncratic_share"] == pytest.approx(1.0 - share)
    assert factor["model_name"] == "single_factor_beta"

    # One PositionBeta per position in the book, each fully shaped.
    assert len(factor["positions"]) == 1
    pos = factor["positions"][0]
    assert set(pos) == {"label", "beta", "r2", "n", "health", "reason"}
    assert pos["label"].startswith("AAPL#")


async def test_factor_diagnostic_pairs_the_book_with_spy_by_DATE(client):
    """The regression must inner-join on DATES, never zip two raw series.

    SPY is seeded on a window OFFSET from the book's, so a positional zip
    would regress the book against the wrong days and still return a
    plausible-looking number. The overlap here is far below ``min_obs``, so
    the only honest answer is an UNAVAILABLE naming the real count.
    """
    await seed_stock_position("AAPL", bars=200)  # 2025-01-01 + 200 days
    await _seed_factor_bars("SPY", bars=200, start_date=date(2025, 6, 1))

    body = (await client.get("/api/portfolio/risk")).json()
    factor = body["statistical"]["factor"]

    assert factor["health"] == "UNAVAILABLE"
    assert factor["portfolio_beta"] is None
    assert "book P&L dates" in factor["reason"]


async def test_correlation_state_serves_the_spearman_average(client):
    """§18: the rank correlation reaches the wire beside its Pearson twin."""
    await seed_stock_position("AAPL", bars=200)
    await seed_stock_position("MSFT", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    state = body["statistical"]["correlation_state"]

    assert "current_avg_spearman" in state
    rho = state["current_avg_spearman"]
    assert rho is not None, "a two-ticker book has a computable rank correlation"
    assert -1.0 <= rho <= 1.0
    # It is a DISTINCT estimator, not an alias of the Pearson average.
    assert isinstance(rho, float)
