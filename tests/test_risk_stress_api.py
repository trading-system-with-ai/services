"""Phase D on the wire — the stress engine's gateway surfaces (design §8.5).

Four surfaces, all SHADOW:

- ``GET /api/portfolio/risk`` gains ``statistical.stress`` (contract §8.5):
  every key present on an EMPTY book, real numbers and FULL_REVAL coverage on
  a seeded one;
- every snapshot build persists one ``stress_runs`` row per catalogue
  scenario — including the UNAVAILABLE ones, with their reason (spec §56);
- ``POST /api/risk/stress/run`` runs a user hypothesis over the current book:
  happy path, 422 on out-of-range, NO audit event, row persisted;
- the pre-trade ``shadow.statistical.stress`` block, with Tier 0 proved
  byte-identical when the stress layer raises.

The house rule this file exists to defend: nothing here may change a Tier 0
decision, and every gap is an honest null with a reason — never a zero.
"""
from datetime import datetime, timezone

import pytest

from apps.gateway.execution import gate_chain
from sqlalchemy import select

from apps.gateway.db import AuditEvent, Position, SessionLocal, StressRunRow
from apps.gateway.risk_snapshot import worst_scenario_pnl_for_key
from apps.gateway.routers import orders as orders_router
from libs.trading_core.risk.models import stress as stress_models

from .test_order_preview import BULL_TICKER, authorize, preview
from .test_risk_snapshot_builder import build, seed_stock_position

#: Every key design §8.5 defines on the ``statistical.stress`` block.
STRESS_KEYS = {
    "mode",
    # ADDED with the §5 tier taxonomy: stress is a FIRST-tier model in the
    # spec's own list (a deterministic reprice, no fitted parameter).
    "tier",
    "catalogue_version",
    "model_version",
    "health",
    "reason",
    "n_stock_legs",
    "n_option_legs",
    "method_coverage",
    "rows",
    "worst",
    "per_position",
    "positions_excluded",
}

#: Every key on one scenario row.
ROW_KEYS = {
    "name",
    "kind",
    "validated",
    "pnl_usd",
    "pnl_pct_nav",
    "loss_usd",
    "loss_pct_nav",
    "method_coverage",
    "health",
    "reason",
    "params",
}

#: Every key the pre-trade ``shadow.statistical.stress`` block carries.
PRETRADE_STRESS_KEYS = {
    "health",
    "reason",
    "worst_before",
    "worst_after",
    "cap",
    "hypothetical",
    "limits",
    "mode",
    "n_candidate_stock_legs",
    "n_candidate_option_legs",
}

OPTION_TICKER = "AAPL"


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


async def _chain_call_near_atm(client, ticker: str) -> tuple[float, dict]:
    """(spot, contract) for the call closest to ATM in the §9 DTE window.

    Read through the options endpoint, so the bars backfill and the chain are
    EXACTLY the ones the risk view will regenerate the same day — a position
    seeded off this contract is one the stress engine can actually resolve.
    """
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code in (201, 409)
    r = await client.get(f"/api/watchlist/{ticker}/options")
    assert r.status_code == 200
    body = r.json()
    spot = body["spot"]
    calls = [c for c in body["chain"] if c["right"] == "C" and 30 <= c["dte"] <= 90]
    assert calls, "stub chain unexpectedly has no 30-90 DTE calls"
    contract = min(calls, key=lambda c: (abs(c["strike"] - spot), c["dte"], c["strike"]))
    return spot, contract


async def seed_long_call(client, ticker: str = OPTION_TICKER, quantity: int = 2) -> int:
    """One OPEN LONG_CALL on a REAL stub-chain contract (so FULL_REVAL runs).

    The chain read above already backfilled the underlying's daily bars, which
    is what gives the position a spot and the catalogue its historical
    windows.
    """
    _spot, contract = await _chain_call_near_atm(client, ticker)
    async with SessionLocal() as session:
        position = Position(
            ticker=ticker,
            instrument="LONG_CALL",
            quantity=quantity,
            avg_price=contract["mid"],
            max_loss=quantity * contract["mid"] * 100,
            stop_distance=contract["mid"],
            opt_expiry=contract["expiry"],
            opt_strike=contract["strike"],
            opt_right="C",
            multiplier=100,
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
        )
        session.add(position)
        await session.commit()
        return position.id


async def _stress_rows() -> list[StressRunRow]:
    async with SessionLocal() as session:
        return list(
            (
                await session.execute(select(StressRunRow).order_by(StressRunRow.id))
            ).scalars().all()
        )


async def _audit_events() -> list[AuditEvent]:
    async with SessionLocal() as session:
        return list((await session.execute(select(AuditEvent))).scalars().all())


# ---------------------------------------------------------------------------
# (a) statistical.stress on the wire — empty book
# ---------------------------------------------------------------------------


async def test_empty_book_serves_the_full_stress_contract(client):
    """No positions is not "no answer": the catalogue still runs and reports
    a real 0.0 for a book that holds nothing, with every contract key
    present and the research grid honestly flagged UNVALIDATED."""
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    stress = r.json()["statistical"]["stress"]

    assert set(stress) == STRESS_KEYS
    assert stress["mode"] == "SHADOW"  # stated, never implied
    assert stress["catalogue_version"] == stress_models.CATALOGUE_VERSION
    assert stress["model_version"] == stress_models.MODEL_VERSION
    assert stress["n_stock_legs"] == 0
    assert stress["n_option_legs"] == 0

    # With no positions there are no stored closes to derive a historical
    # window from, so the catalogue is the research grid alone — and EVERY
    # row of it is unvalidated (spec §11: never a silent production number).
    assert stress["rows"], "the hypothetical grid must still run on an empty book"
    for row in stress["rows"]:
        assert set(row) == ROW_KEYS
        assert row["validated"] is False
        assert row["kind"] in {"HYPOTHETICAL", "IV_GRID"}
        # An empty book's P&L is a REAL zero, not a missing measurement.
        assert row["pnl_usd"] == 0.0
        assert row["health"] == "ACTIVE"
        assert row["params"]["uniform_beta_1"] is True
    assert stress["worst"] is not None
    assert stress["worst"]["pnl_usd"] == 0.0
    assert stress["per_position"] == {}


# ---------------------------------------------------------------------------
# (b) A seeded stock book: historical windows, worst row, signs
# ---------------------------------------------------------------------------


async def test_seeded_stock_book_prices_every_scenario(client):
    await seed_stock_position("AAPL", bars=200, quantity=100)
    stress = (await build()).api["stress"]

    assert stress["n_stock_legs"] == 1
    assert stress["n_option_legs"] == 0
    kinds = {row["kind"] for row in stress["rows"]}
    assert "HISTORICAL" in kinds, "stored closes must produce historical windows"
    assert "HYPOTHETICAL" in kinds

    # A LONG stock book loses money in a down scenario and makes it in an up
    # one — the sign convention, pinned (pnl_usd is GAIN-positive).
    down = next(r for r in stress["rows"] if r["name"] == "Equity -10% / IV +40%")
    up = next(r for r in stress["rows"] if r["name"] == "Equity +5% / IV -15%")
    assert down["pnl_usd"] < 0 < up["pnl_usd"]
    # `loss_usd` restates it in the VaR/ES sign (positive = money lost).
    assert down["loss_usd"] == pytest.approx(-down["pnl_usd"])

    # Stock has no vega and no theta: an IV-only or time-only scenario moves
    # a pure stock book by exactly nothing.
    crush = next(r for r in stress["rows"] if r["name"] == "IV crush (flat, -40%)")
    decay = next(r for r in stress["rows"] if r["name"] == "Time decay only (+5 days)")
    assert crush["pnl_usd"] == 0.0
    assert decay["pnl_usd"] == 0.0

    # The worst row is the smallest pnl_usd among the priced rows.
    priced = [r["pnl_usd"] for r in stress["rows"] if r["pnl_usd"] is not None]
    assert stress["worst"]["pnl_usd"] == min(priced)
    assert stress["per_position"], "the worst row names its per-leg P&L (spec §52)"


async def test_a_window_outside_the_stored_history_is_an_honest_unavailable_row(client):
    """A named historical window our bars do not cover is a row with a
    REASON — never a fabricated 0 and never a silently dropped scenario."""
    await seed_stock_position("AAPL", bars=200)
    stress = (await build()).api["stress"]

    unavailable = [r for r in stress["rows"] if r["pnl_usd"] is None]
    assert unavailable, "the 2024 window predates the seeded 2025 bars"
    for row in unavailable:
        assert row["health"] == "UNAVAILABLE"
        assert row["reason"], "an unavailable row must say why"
        assert row["pnl_pct_nav"] is None
        assert row["loss_usd"] is None
    # ...and the run as a whole says how many were missing rather than
    # claiming a clean sweep.
    assert stress["reason"] and "unavailable" in stress["reason"]


async def test_stress_gaps_are_reported_separately_from_statistical_gaps(client):
    """The two exclusion lists are different facts and stay apart: a position
    with a known delta is in the STATISTICAL book even when its contract has
    rolled off today's chain and it has no stress leg."""
    # A LONG_CALL whose contract does NOT exist in any chain: greeks report
    # it data_ok=False (statistical exclusion) AND it has no stress leg.
    await seed_stock_position("AAPL", bars=200)
    async with SessionLocal() as session:
        session.add(
            Position(
                ticker="AAPL",
                instrument="LONG_CALL",
                quantity=1,
                avg_price=5.0,
                max_loss=500.0,
                stop_distance=5.0,
                opt_expiry="2099-12-18",   # no chain reaches 2099
                opt_strike=1_000_000.0,
                opt_right="C",
                multiplier=100,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    api = (await build()).api
    stress = api["stress"]
    # The phantom contract has no stress leg, and the stress block says so
    # in ITS OWN list with the real reason.
    assert stress["n_option_legs"] == 0
    assert stress["positions_excluded"], "the missing contract must be named"
    assert any(
        "chain" in e["reason"] for e in stress["positions_excluded"]
    ), stress["positions_excluded"]
    # The stock leg still priced — one bad row never blanks the run.
    assert stress["n_stock_legs"] == 1
    assert stress["worst"]["pnl_usd"] is not None

    # Here BOTH views happen to exclude the same row — but each says so in
    # its OWN words, about its OWN measurement. The statistical view lost a
    # DELTA; the stress view lost a LEG. Merging the lists would have made
    # one reason stand for the other.
    stress_entry = stress["positions_excluded"][0]
    stat_entry = next(
        e for e in api["positions_excluded"] if e["key"] == stress_entry["key"]
    )
    assert stress_entry["reason"] != stat_entry["reason"]
    assert "greeks" in stat_entry["reason"] or "delta" in stat_entry["reason"]
    assert "stress leg" in stress_entry["reason"]


# ---------------------------------------------------------------------------
# (c) An option book: FULL_REVAL actually happens (spec §21/§22)
# ---------------------------------------------------------------------------


async def test_option_position_is_fully_revalued_not_delta_approximated(client):
    """The whole point of Phase D: an option leg with an IV is REPRICED, not
    multiplied by delta. `method_coverage` proves which happened."""
    await seed_long_call(client, quantity=2)
    stress = (await build()).api["stress"]

    assert stress["n_option_legs"] == 1
    assert stress["n_stock_legs"] == 0
    for row in stress["rows"]:
        if row["pnl_usd"] is None:
            continue
        assert row["method_coverage"]["FULL_REVAL"] == 1, row["name"]
        assert row["method_coverage"]["DELTA_LINEAR"] == 0, row["name"]
    assert stress["method_coverage"]["FULL_REVAL"] == 1

    # A long call is long VEGA and short THETA: the IV-only and time-only
    # scenarios move it, which a delta approximation could never show.
    crush = next(r for r in stress["rows"] if r["name"] == "IV crush (flat, -40%)")
    spike = next(r for r in stress["rows"] if r["name"] == "IV spike (flat, +50%)")
    decay = next(r for r in stress["rows"] if r["name"] == "Time decay only (+5 days)")
    assert crush["pnl_usd"] < 0, "an IV crush must hurt a long call"
    assert spike["pnl_usd"] > 0, "an IV spike must help a long call"
    assert decay["pnl_usd"] < 0, "five days of decay must cost a long call"


# ---------------------------------------------------------------------------
# (d) Persistence: one stress_runs row per scenario, per build (spec §56)
# ---------------------------------------------------------------------------


async def test_every_snapshot_build_persists_one_row_per_scenario(client):
    await seed_stock_position("AAPL", bars=200)
    # A PERSISTING build: ON_DEMAND builds dedupe their writes to one per
    # 15-minute window (the risk view is polled), so use the scheduled
    # trigger to exercise the per-scenario persistence contract itself.
    result = await build(trigger="SCHEDULED")
    api_rows = result.api["stress"]["rows"]

    rows = await _stress_rows()
    assert len(rows) == len(api_rows), "one persisted row per catalogue scenario"
    assert {r.snapshot_id for r in rows} == {result.row_id}
    by_name = {r.scenario: r for r in rows}
    for served in api_rows:
        stored = by_name[served["name"]]
        assert stored.kind == served["kind"]
        assert stored.validated == served["validated"]
        assert stored.pnl_usd == served["pnl_usd"]
        assert stored.pnl_pct_nav == served["pnl_pct_nav"]
        assert stored.health == served["health"]
        assert stored.reason == served["reason"]
        assert stored.method_full_reval == served["method_coverage"]["FULL_REVAL"]
        assert (
            stored.method_delta_linear == served["method_coverage"]["DELTA_LINEAR"]
        )
        assert stored.params["spot_shock"] == served["params"]["spot_shock"]

    # UNAVAILABLE rows are kept too: "that window is outside our history" is
    # a fact worth storing — an absent row would later read as "never run".
    assert any(r.pnl_usd is None and r.reason for r in rows)

    # A SECOND persisting build appends a second generation; history accrues
    # (§56). (An ON_DEMAND build inside the 15-minute dedupe window would
    # compute but not write — pinned in test_risk_snapshot_api.)
    await build(trigger="PRE_TRADE")
    assert len(await _stress_rows()) == 2 * len(api_rows)


# ---------------------------------------------------------------------------
# (e) POST /api/risk/stress/run — the user-defined scenario
# ---------------------------------------------------------------------------


async def test_user_scenario_runs_persists_and_writes_no_audit_event(client):
    await seed_long_call(client, quantity=3)
    before = len(await _audit_events())

    r = await client.post(
        "/api/risk/stress/run",
        json={"equity_shock": -0.12, "iv_shock": 0.35, "days_forward": 3,
              "name": "my crash"},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["mode"] == "SHADOW"
    assert body["n_option_legs"] == 1
    scenario = body["scenario"]
    assert set(scenario) == ROW_KEYS
    assert scenario["name"] == "my crash"
    assert scenario["kind"] == "USER"
    assert scenario["validated"] is False, "a user hypothesis is never validated"
    assert scenario["pnl_usd"] is not None
    assert scenario["method_coverage"]["FULL_REVAL"] == 1
    assert scenario["params"]["spot_shock"] == -0.12
    assert scenario["params"]["iv_shock"] == 0.35
    assert scenario["params"]["days_forward"] == 3.0
    assert body["per_position"], "the per-leg P&L travels back (spec §52)"

    # The run is PERSISTED (spec §56) with snapshot_id NULL — it is a read of
    # the book under a hypothesis, not a snapshot build.
    rows = [r for r in await _stress_rows() if r.kind == "USER"]
    assert len(rows) == 1
    assert rows[0].id == body["run_id"]
    assert rows[0].snapshot_id is None
    assert rows[0].scenario == "my crash"
    assert rows[0].pnl_usd == scenario["pnl_usd"]

    # ...and writes NO audit event: a hypothesis is not a decision
    # (house rule — read views write no audit events).
    assert len(await _audit_events()) == before


async def test_user_scenario_on_an_empty_book_is_an_honest_zero(client):
    r = await client.post("/api/risk/stress/run", json={"equity_shock": -0.2})
    assert r.status_code == 200
    body = r.json()
    assert body["n_stock_legs"] == 0 and body["n_option_legs"] == 0
    # An empty book's P&L under any scenario is exactly 0.0 — the true value
    # of a book that holds nothing, and the row says so at ACTIVE health.
    assert body["scenario"]["pnl_usd"] == 0.0
    assert body["scenario"]["health"] == "ACTIVE"
    assert body["per_position"] == {}


async def test_unnamed_user_scenario_gets_a_parameter_derived_name(client):
    r = await client.post(
        "/api/risk/stress/run", json={"equity_shock": -0.05, "iv_shock": 0.2}
    )
    assert r.status_code == 200
    name = r.json()["scenario"]["name"]
    # Two unnamed runs with different shocks must not collide into one label.
    assert "-5.0%" in name and "+20%" in name


@pytest.mark.parametrize(
    "payload",
    [
        {"equity_shock": -0.95},                      # below the -0.9 floor
        {"equity_shock": 2.5},                        # above the +2 ceiling
        {"equity_shock": 0.0, "iv_shock": -0.95},     # below the -0.9 floor
        {"equity_shock": 0.0, "iv_shock": 5.5},       # above the +5 ceiling
        {"equity_shock": 0.0, "days_forward": -1},    # negative decay
        {"equity_shock": 0.0, "days_forward": 400},   # beyond a year
        {},                                            # equity_shock required
    ],
)
async def test_out_of_range_scenario_is_422_and_persists_nothing(client, payload):
    """Out of range is a 422 naming the field — never a silently clamped
    number, which would run a different scenario than the one asked for."""
    r = await client.post("/api/risk/stress/run", json=payload)
    assert r.status_code == 422
    assert await _stress_rows() == []


async def test_range_bounds_are_inclusive(client):
    """The documented bounds are the accepted extremes, not off-by-one."""
    for payload in (
        {"equity_shock": -0.9},
        {"equity_shock": 2.0},
        {"equity_shock": 0.0, "iv_shock": -0.9},
        {"equity_shock": 0.0, "iv_shock": 5.0},
        {"equity_shock": 0.0, "days_forward": 0},
        {"equity_shock": 0.0, "days_forward": 365},
    ):
        r = await client.post("/api/risk/stress/run", json=payload)
        assert r.status_code == 200, payload


# ---------------------------------------------------------------------------
# (f) Pre-trade: shadow.statistical.stress, and Tier 0 byte-identity
# ---------------------------------------------------------------------------


async def test_pretrade_shadow_carries_the_stress_block(client):
    await seed_stock_position("AAPL", bars=200)
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    stress = body["risk"]["shadow_statistical"]["stress"]
    assert set(stress) == PRETRADE_STRESS_KEYS
    assert stress["mode"] == "SHADOW"
    assert stress["limits"]["max_stress_loss_pct_nav"] == 0.10  # research default
    assert stress["limits"]["mode"] == "SHADOW"
    # The candidate is described as LEGS, per unit of quantity.
    assert stress["n_candidate_stock_legs"] + stress["n_candidate_option_legs"] >= 1
    # Before is the CURRENT book (the build already ran the catalogue);
    # after is the same catalogue over book + candidate × approved qty.
    assert stress["worst_before"] is not None
    assert stress["worst_after"] is not None
    assert stress["worst_before"]["scenario"]
    assert stress["hypothetical"]["mode"] == "SHADOW"
    # SHADOW: the hypothetical never raises the quantity Tier 0 approved.
    assert (
        stress["hypothetical"]["quantity"] <= body["risk"]["approved_quantity"]
    )

    # The §46 table gains ONE row, in the before/after/delta shape.
    comparison = body["risk"]["comparison"]
    if comparison is not None:
        row = next(
            r for r in comparison["rows"] if r["metric"] == "worst_stress_loss"
        )
        assert row["layer"] == "STRESS"
        assert row["before_usd"] is not None
        assert row["after_usd"] is not None
        assert row["delta_usd"] == pytest.approx(
            row["after_usd"] - row["before_usd"]
        )
        assert row["before_scenario"] and row["after_scenario"]

    # Mirrored verbatim into the RISK_DECISION audit (SHADOW block).
    r = await client.get("/api/audit", params={"entity_id": BULL_TICKER})
    events = [e for e in r.json() if e["action"] == "RISK_DECISION"]
    assert len(events) == 1
    audited = events[0]["details"]["shadow"]["statistical"]["stress"]
    assert audited["worst_before"] == stress["worst_before"]
    assert audited["hypothetical"] == stress["hypothetical"]


async def test_tier0_is_byte_identical_when_the_stress_layer_raises(
    client, monkeypatch
):
    """§70: a SHADOW model cannot veto — not even by exploding. The Tier 0
    decision, quantity, reason codes and gates are IDENTICAL with the stress
    layer working and with it raising; only a `note` differs."""
    await seed_stock_position("AAPL", bars=200)
    await authorize(client, BULL_TICKER)
    healthy = await preview(client, BULL_TICKER)

    def _boom(*args, **kwargs):
        raise RuntimeError("stress engine exploded")

    monkeypatch.setattr(gate_chain, "run_stress", _boom)
    broken = await preview(client, BULL_TICKER)

    for key in (
        "decision",
        "approved_quantity",
        "signal_strength",
        "risk_budget_pct",
        "trade_risk_usd",
        "reason_codes",
        "explanations",
        "heat_before_pct",
        "heat_after_pct",
        "cash_after_pct",
        "binding_constraints",
    ):
        assert broken["risk"][key] == healthy["risk"][key], key
    assert broken["gates"] == healthy["gates"]

    # The failure is REPORTED, not swallowed — and it is the only difference.
    assert broken["risk"]["shadow_statistical"]["stress"] == {
        "note": "RuntimeError: stress engine exploded"
    }
    # ...and no STRESS cap could have bound, so the hypothetical verdict is
    # the Phase C one alone.
    binding = broken["risk"]["shadow_statistical"]["hypothetical"]["binding"]
    assert "STRESS_LOSS_LIMIT" not in binding


async def test_a_stress_cap_binds_only_in_the_hypothetical_never_in_tier0(client):
    """A STRESS cap that DOES bind resizes the hypothetical quantity and
    names its own layer — while the Tier 0 approved quantity is untouched."""
    await seed_stock_position("AAPL", bars=200)
    await authorize(client, BULL_TICKER)
    # A punishing limit (0.001 % of NAV) guarantees the cap binds.
    punishing = stress_models.StressLimits(max_stress_loss_pct_nav=1e-5)
    original = gate_chain.STRESS_LIMITS
    gate_chain.STRESS_LIMITS = punishing
    try:
        body = await preview(client, BULL_TICKER, quantity=5)
    finally:
        gate_chain.STRESS_LIMITS = original

    risk = body["risk"]
    stress = risk["shadow_statistical"]["stress"]
    if risk["approved_quantity"] == 0:
        pytest.skip("Tier 0 approved nothing here — no cap can bind on 0 units")
    assert stress["cap"] is not None
    assert stress["cap"]["code"] == "STRESS_LOSS_LIMIT"
    assert stress["cap"]["layer"] == "STRESS"
    assert stress["cap"]["sentence"], "the §47 sentence is server-generated"
    assert "SHADOW" in stress["cap"]["sentence"]
    # It joins the ONE shadow verdict Phase C computes, carrying its layer.
    assert "STRESS_LOSS_LIMIT" in risk["shadow_statistical"]["hypothetical"]["binding"]
    # ...and Tier 0's own answer is untouched: no STRESS code among the
    # constraints the ENGINE says bound.
    assert all(
        c["layer"] == "HARD_LIMIT" for c in risk["binding_constraints"]
    )


# ---------------------------------------------------------------------------
# (g) Positions API: the additive option-row fields (spec §52)
# ---------------------------------------------------------------------------


async def test_option_rows_carry_the_additive_risk_fields(client):
    pos_id = await seed_long_call(client, quantity=2)
    await build()  # persists the stress rows the positions view reads

    r = await client.get("/api/positions")
    assert r.status_code == 200
    row = next(p for p in r.json() if p["id"] == pos_id)

    # Premium at risk is a DB fact: quantity × entry premium × multiplier.
    assert row["premium_at_risk"] == pytest.approx(row["max_loss"])
    assert row["dte"] is not None and row["dte"] >= 0
    assert row["dte"] == row["contract"]["dte"]
    assert row["iv0"] is not None and row["iv0"] > 0
    # The worst scenario's P&L for THIS position, with the scenario named.
    assert row["worst_scenario_pnl"] is not None
    assert row["worst_scenario_name"]


async def test_stock_rows_report_nulls_not_zeros_for_the_option_fields(client):
    """A share has no premium at risk, no expiry and no IV — null says that;
    zero would claim a measurement (§44 rule 18)."""
    pos_id = await seed_stock_position("AAPL", bars=200)
    await build()

    r = await client.get("/api/positions")
    row = next(p for p in r.json() if p["id"] == pos_id)
    assert row["premium_at_risk"] is None
    assert row["dte"] is None
    assert row["iv0"] is None
    # ...but its scenario loss IS measurable — a stock leg is revalued too.
    assert row["worst_scenario_pnl"] is not None


async def test_worst_scenario_pnl_is_null_before_any_snapshot_exists(client):
    """Never measured is not zero: with no persisted stress run the field is
    an honest null."""
    pos_id = await seed_stock_position("AAPL", bars=200)
    r = await client.get("/api/positions")
    row = next(p for p in r.json() if p["id"] == pos_id)
    assert row["worst_scenario_pnl"] is None
    assert row["worst_scenario_name"] is None


def test_worst_scenario_pnl_for_key_sums_a_spread_s_two_legs():
    """A spread is ONE position and TWO legs; its row shows the NET."""
    per_key = {"AAPL#7:long": -500.0, "AAPL#7:short": 180.0, "MSFT#9": -20.0}
    assert worst_scenario_pnl_for_key(per_key, "AAPL#7") == pytest.approx(-320.0)
    assert worst_scenario_pnl_for_key(per_key, "MSFT#9") == -20.0
    # A position that contributed no leg was excluded from the run — null,
    # not 0.0 (which would claim it loses nothing).
    assert worst_scenario_pnl_for_key(per_key, "NVDA#3") is None


# ---------------------------------------------------------------------------
# (h) The read view still writes no audit event (house rule)
# ---------------------------------------------------------------------------


async def test_the_risk_view_stress_block_writes_no_audit_event(client):
    """A snapshot is a MEASUREMENT, not a decision: the stress block reaching
    the wire must add no RISK_DECISION (the view may still lazily backfill
    bars, which has its own DATA audit action — that is a data request, not a
    risk decision)."""
    await seed_long_call(client, quantity=1)
    before = [e.action for e in await _audit_events()]
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    assert r.json()["statistical"]["stress"]["rows"]
    after = [e.action for e in await _audit_events()]
    assert after.count("RISK_DECISION") == before.count("RISK_DECISION") == 0


# ---------------------------------------------------------------------------
# (i) Migration parity for the new table
# ---------------------------------------------------------------------------


def test_stress_runs_orm_mirrors_migration_019():
    """The mirror rule, pinned here as well as in test_migration_parity so a
    Phase D reader sees it: the ORM columns ARE the migration's, in order."""
    from pathlib import Path

    from apps.gateway.db import Base
    from tests.test_migration_parity import MIGRATIONS, _sql_columns

    sql = (MIGRATIONS / "019_stress_runs.sql").read_text()
    orm_cols = [c.name for c in Base.metadata.tables["stress_runs"].columns]
    assert _sql_columns(sql, "stress_runs") == orm_cols
    # ...and the file is mounted, so a fresh volume gets the table.
    compose = (Path(MIGRATIONS).parent / "docker-compose.yml").read_text()
    assert "019_stress_runs.sql:/docker-entrypoint-initdb.d/" in compose
