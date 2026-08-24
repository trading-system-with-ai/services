"""Phase C pre-trade portfolio risk on the wire (design contract §7.5), SHADOW.

Three surfaces, one guarantee. The surfaces:

- the preview's ``risk`` block gains ``comparison`` (the spec §46 CURRENT vs
  AFTER TRADE table), ``binding_constraints`` and ``shadow_statistical``;
- the ``RISK_DECISION`` audit detail gains ``requested_quantity`` /
  ``binding_constraints`` and, under ``shadow.statistical``, the comparison,
  the hypothetical caps, the verdict, the limits and the correlation state —
  plus ``shadow.vol_targeting_ewma``, the §14 side-by-side;
- ``GET /api/portfolio/risk`` gains ``statistical.correlation_state`` and the
  two additive ``vol_targeting`` EWMA fields.

The guarantee (§70), which every test here exists to protect: **none of it
changes a Tier 0 decision.** The load-bearing test monkeypatches
``pretrade.compare`` to RAISE and asserts the decision, the approved
quantity, the gates and the reason codes are byte-identical.

Book construction: ``seed_stock_position`` on AAPL gives the book a priceable
position with 200 bars; ``BULL_TICKER`` (GOOGL) is the candidate. Seeding a
one-share GOOGL position too puts the CANDIDATE's ticker in the return matrix,
which is what turns the comparison from UNAVAILABLE into ACTIVE — both states
are pinned below, because an honest null is as much a contract as a number.
"""
from datetime import datetime, timezone

import pytest

from apps.gateway.execution import gate_chain

from apps.gateway.db import Position, SessionLocal
from apps.gateway.routers import orders as orders_router

from .test_order_preview import BULL_TICKER, authorize, preview
from .test_risk_snapshot_builder import seed_stock_position

#: Keys the §46 comparison always carries (contract §7.1 / §7.5).
COMPARISON_KEYS = {
    "quantity",
    "health",
    "reason",
    "n_obs",
    "tail_size_95",
    "tier0_rows",
    "rows",
    "incremental_es_95_usd",
    "incremental_es_95_pct_nav",
    "marginal_es_95_per_unit",
    "candidate_es_share_after",
    "max_single_es_share_before",
    "max_single_es_share_after",
    "bucket_es_share_after",
    "net_delta_notional_before",
    "net_delta_notional_after",
    # ADDED (§8): incremental VaR, first-class beside incremental ES.
    "incremental_var_95_usd",
    "incremental_var_95_pct_nav",
    # ADDED (§46): net vega before/after, $ per one IV point.
    "net_vega_before",
    "net_vega_after",
}

#: The §46 table rows, in the order the UI renders them.
METRIC_ROWS = [
    "var_hist_95",
    "es_hist_95",
    "var_hist_99",
    "es_hist_99",
    "gaussian_es_95",
    "volatility",
]

#: Every row of a metric pair (before / after / delta, each with health).
ROW_KEYS = {
    "metric",
    "before_usd",
    "after_usd",
    "before_pct_nav",
    "after_pct_nav",
    "delta_usd",
    "delta_pct_nav",
    "before_health",
    "after_health",
    "reason",
}


async def _priceable_book() -> None:
    """A book whose ONLY name is AAPL — the candidate GOOGL is NOT in it."""
    await seed_stock_position("AAPL", bars=200)


async def _book_including_candidate() -> None:
    """A book that also holds one share of the candidate's ticker, so GOOGL
    has a column in the return matrix and every statistic is computable.

    The GOOGL position is inserted WITHOUT bars: the ticker's own history is
    the one the chain lazily backfills, and seeding a second synthetic series
    on top of it would both give the candidate returns no real book has and
    starve the DATA_QUALITY gate (which needs the real 200+ bars).
    """
    await seed_stock_position("AAPL", bars=200)
    async with SessionLocal() as session:
        session.add(
            Position(
                ticker=BULL_TICKER,
                instrument="LONG_STOCK",
                quantity=1,
                avg_price=100.0,
                max_loss=10.0,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def _latest_risk_decision(client, ticker: str) -> dict:
    r = await client.get("/api/audit", params={"entity_id": ticker})
    events = [e for e in r.json() if e["action"] == "RISK_DECISION"]
    assert events, "no RISK_DECISION event recorded"
    return events[0]


# ---------------------------------------------------------------------------
# (a) The preview's risk block: the §46 table, before/after rows
# ---------------------------------------------------------------------------


async def test_preview_risk_carries_the_comparison_table(client):
    """The §46 CURRENT vs AFTER TRADE table reaches the preview with real
    before/after numbers on every row, and the Tier 0 rows are Tier 0's."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    risk = body["risk"]
    assert risk is not None, "the chain must reach RISK_APPROVAL for this test"
    comparison = risk["comparison"]
    assert set(comparison) == COMPARISON_KEYS
    assert comparison["health"] == "ACTIVE"
    assert comparison["quantity"] == risk["approved_quantity"]

    # The statistical rows, in the contract's order, each with BOTH sides.
    # Phase D (§8.5) APPENDS a `worst_stress_loss` row with its own shape
    # (it names a scenario and carries a layer) — the Phase C rows and their
    # order are unchanged, which is what this test pins.
    stat_rows = [r for r in comparison["rows"] if r["metric"] in METRIC_ROWS]
    assert [row["metric"] for row in stat_rows] == METRIC_ROWS
    assert comparison["rows"][: len(METRIC_ROWS)] == stat_rows
    for row in stat_rows:
        assert set(row) == ROW_KEYS
        assert row["before_usd"] is not None, row["metric"]
        assert row["after_usd"] is not None, row["metric"]
        # delta = after − before, exactly (contract §7.1).
        assert row["delta_usd"] == pytest.approx(
            row["after_usd"] - row["before_usd"]
        )
        # Adding a long position to a long book cannot REDUCE its loss tail.
        assert row["delta_usd"] > 0, row["metric"]

    # ES ≥ VaR on both sides at the same confidence (contract §3 invariant 1),
    # measured here through the wire rather than in the library.
    by_metric = {row["metric"]: row for row in comparison["rows"]}
    for side in ("before_usd", "after_usd"):
        assert by_metric["es_hist_95"][side] >= by_metric["var_hist_95"][side]
        assert by_metric["es_hist_99"][side] >= by_metric["var_hist_99"][side]

    # Incremental ES IS the ES-95 row's delta — the same two numbers, so the
    # table and the headline can never tell different stories.
    assert comparison["incremental_es_95_usd"] == pytest.approx(
        by_metric["es_hist_95"]["delta_usd"]
    )

    # The two Tier 0 rows are the ENGINE's numbers, carried verbatim.
    tier0 = {row["metric"]: row for row in comparison["tier0_rows"]}
    assert tier0["portfolio_heat_pct"]["before_pct"] == risk["heat_before_pct"]
    assert tier0["portfolio_heat_pct"]["after_pct"] == risk["heat_after_pct"]
    assert tier0["cash_pct"]["after_pct"] == risk["cash_after_pct"]
    assert all(row["layer"] == "HARD_LIMIT" for row in comparison["tier0_rows"])

    # Concentration view: the candidate holds a share of the ES it added, and
    # the delta-notional grows by the candidate's own exposure.
    assert 0.0 < comparison["candidate_es_share_after"] <= 1.0
    assert comparison["net_delta_notional_after"] > comparison[
        "net_delta_notional_before"
    ]


async def test_comparison_is_honest_null_when_the_candidate_has_no_returns(client):
    """The candidate's ticker missing from the book's return matrix is an
    honest UNAVAILABLE with the real reason — never a fabricated zero row."""
    await _priceable_book()  # AAPL only; the candidate is GOOGL
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    comparison = body["risk"]["comparison"]
    assert comparison["health"] == "UNAVAILABLE"
    assert BULL_TICKER in comparison["reason"]
    assert "no returns column" in comparison["reason"]
    # The STATISTICAL rows only: Phase D's appended `worst_stress_loss` row
    # is measured from LEGS, not from a return matrix, so it legitimately
    # still has numbers here — a missing return column disables the
    # statistical view, not the scenario reprice.
    for row in comparison["rows"]:
        if row["metric"] not in METRIC_ROWS:
            continue
        assert row["before_usd"] is None
        assert row["after_usd"] is None
        assert row["delta_usd"] is None
    assert comparison["incremental_es_95_usd"] is None
    assert comparison["candidate_es_share_after"] is None
    # ...while the TIER 0 rows are still real: the hard limits measured fine.
    assert comparison["tier0_rows"][0]["before_pct"] is not None

    # An unavailable statistical view produces NO cap (fail-open in SHADOW,
    # contract §7.2) — and therefore never a hypothetical resize.
    shadow = body["risk"]["shadow_statistical"]
    assert shadow["caps"]["health"] == "UNAVAILABLE"
    assert shadow["caps"]["rows"] == []
    assert shadow["hypothetical"]["binding"] == []
    assert shadow["hypothetical"]["quantity"] == body["risk"]["approved_quantity"]


# ---------------------------------------------------------------------------
# (b) The hypothetical verdict and the caps
# ---------------------------------------------------------------------------


async def test_shadow_statistical_hypothetical_and_caps(client):
    """``hypothetical`` states a decision, a quantity and what bound — and a
    cap that binds does NOT touch the Tier 0 approved quantity."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    risk = body["risk"]
    shadow = risk["shadow_statistical"]
    hypothetical = shadow["hypothetical"]
    assert hypothetical["mode"] == "SHADOW"
    assert hypothetical["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE", "REJECT"}
    assert 0 <= hypothetical["quantity"] <= risk["approved_quantity"]
    assert hypothetical["approved_quantity"] == risk["approved_quantity"]

    # This book is a single-bucket tech book, so the concentration limit binds
    # and the statistical layer alone would have REJECTED...
    assert hypothetical["decision"] == "REJECT"
    assert hypothetical["quantity"] == 0
    assert "BUCKET_ES_CONTRIBUTION_CAP:TECH_MEGA" in hypothetical["binding"]

    # ...and Tier 0 APPROVED anyway. That gap IS the shadow window's finding.
    assert risk["decision"] == "APPROVE"
    assert risk["approved_quantity"] > 0

    caps = shadow["caps"]
    assert caps["health"] in {"ACTIVE", "DEGRADED"}
    assert caps["rows"]
    # The cap search really SEARCHED: this preview auto-sizes (no explicit
    # quantity), which means `quantity_requested` is None on the request —
    # the upper bound must fall back to what Tier 0 sized, never to 0, or
    # every bisection would collapse to an empty [0, 0] interval and report
    # a meaningless cap of 0 for every limit.
    by_code = {cap["code"]: cap for cap in caps["rows"]}
    single = by_code["ES_CONTRIBUTION_CAP"]
    assert 0 < single["cap_qty"] < risk["approved_quantity"]
    assert str(risk["approved_quantity"]) in single["sentence"]
    # Most restrictive first (contract §7.2 ordering).
    binding_caps = [
        by_code[code]["cap_qty"] for code in hypothetical["binding"]
    ]
    assert binding_caps == sorted(binding_caps)
    for cap in caps["rows"]:
        assert cap["layer"] in {"STATISTICAL", "CONCENTRATION"}
        assert cap["cap_qty"] >= 0
        # §47: a server-generated sentence with the REAL numbers in it, which
        # the UI renders verbatim (house rule: audit-exact strings).
        assert cap["sentence"] and "%" in cap["sentence"]
        assert cap["measured"]
    # Every binding code names a cap that was actually emitted.
    codes = {cap["code"] for cap in caps["rows"]}
    assert set(hypothetical["binding"]) <= codes

    # The thresholds are stated with the payload — research defaults, SHADOW.
    assert shadow["limits"]["mode"] == "SHADOW"
    assert shadow["limits"]["max_bucket_es_share"] == 0.50
    assert shadow["limits"]["min_obs"] == 60


async def test_comparison_at_requested_appears_only_when_it_differs(client):
    """A second table at the REQUESTED quantity is present when Tier 0
    resized, and absent (null) when requested == approved — a duplicate row
    set claiming to be a second measurement would be noise."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    # A quantity far above what the risk budget allows forces a resize.
    body = await preview(client, BULL_TICKER, quantity=100_000)
    risk = body["risk"]
    assert risk["approved_quantity"] != 100_000
    at_requested = risk["shadow_statistical"]["comparison_at_requested"]
    assert at_requested is not None
    assert at_requested["quantity"] == 100_000
    # The bigger trade carries the bigger tail — measured, not assumed.
    assert at_requested["incremental_es_95_usd"] > risk["comparison"][
        "incremental_es_95_usd"
    ]


# ---------------------------------------------------------------------------
# (c) THE guarantee: the shadow layer cannot change a decision
# ---------------------------------------------------------------------------


async def test_a_raising_comparison_leaves_the_decision_identical(client, monkeypatch):
    """The load-bearing SHADOW test (§70). With ``pretrade.compare`` RAISING,
    the decision, the approved quantity, the gates and the reason codes are
    IDENTICAL — only a note appears where the comparison would have been."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    good_body = await preview(client, BULL_TICKER)
    good = (await _latest_risk_decision(client, BULL_TICKER))["details"]
    assert good["shadow"]["statistical"]["comparison"] is not None

    def boom(*args, **kwargs):
        raise RuntimeError("pretrade comparison exploded")

    monkeypatch.setattr(gate_chain, "pretrade_compare", boom)
    broken_body = await preview(client, BULL_TICKER)
    broken = (await _latest_risk_decision(client, BULL_TICKER))["details"]

    # The decision, verbatim.
    assert broken["decision"] == good["decision"]
    assert broken["approved_quantity"] == good["approved_quantity"]
    assert broken["requested_quantity"] == good["requested_quantity"]
    assert broken["gates"] == good["gates"]
    assert broken["reason_codes"] == good["reason_codes"]
    assert broken["veto_gate"] == good["veto_gate"]
    assert broken["binding_constraints"] == good["binding_constraints"]
    assert broken["budget_multiplier"] == good["budget_multiplier"]

    # ...and on the wire too.
    assert broken_body["risk"]["decision"] == good_body["risk"]["decision"]
    assert (
        broken_body["risk"]["approved_quantity"]
        == good_body["risk"]["approved_quantity"]
    )
    assert (
        broken_body["risk"]["binding_constraints"]
        == good_body["risk"]["binding_constraints"]
    )

    # The ONLY difference: the Phase C block reports its own failure — under
    # `comparison_note`, leaving the Phase B `note` (which describes the
    # CURRENT-book view, still computed) intact.
    broken_shadow = broken["shadow"]["statistical"]
    assert "comparison" not in broken_shadow
    assert broken_shadow["comparison_note"] == (
        "RuntimeError: pretrade comparison exploded"
    )
    assert broken_shadow["note"] == good["shadow"]["statistical"]["note"]
    assert (
        broken_shadow["historical_es_95_1d_pct_nav"]
        == good["shadow"]["statistical"]["historical_es_95_1d_pct_nav"]
    )
    assert broken_body["risk"]["comparison"] is None


async def test_gate_order_is_unchanged_by_phase_c(client):
    """No STATISTICAL / CONCENTRATION gate joins the chain in SHADOW: the
    gate names and their order are exactly Tier 0's (contract §7.5)."""
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
    # ...and the chain a preview actually runs is a SUBSEQUENCE of it (the
    # research chain drops the pool gate, §15) — no name Phase C invented.
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    assert set(g["name"] for g in body["gates"]) <= set(gate_chain.GATE_ORDER)


# ---------------------------------------------------------------------------
# (d) The RISK_DECISION audit detail
# ---------------------------------------------------------------------------


async def test_risk_decision_carries_the_phase_c_keys(client):
    """``requested_quantity`` / ``binding_constraints`` from the assessment,
    and the full ``shadow.statistical`` Phase C payload."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    details = (await _latest_risk_decision(client, BULL_TICKER))["details"]

    # From the assessment itself (contract §7.3) — a pure re-presentation of
    # the reason codes, each with the LAYER that owns it.
    assert details["requested_quantity"] == details["quantity_requested"]
    for constraint in details["binding_constraints"]:
        assert set(constraint) == {"code", "layer"}
        # Every Tier 0 rule is a HARD_LIMIT; no SHADOW cap may appear here,
        # because none was passed to assess (no `extra_caps`).
        assert constraint["layer"] == "HARD_LIMIT"
    assert [c["code"] for c in details["binding_constraints"]] == list(
        details["reason_codes"]
    )

    shadow = details["shadow"]["statistical"]
    for key in ("comparison", "caps", "hypothetical", "limits", "correlation_state"):
        assert key in shadow, key
    assert shadow["comparison"]["health"] == "ACTIVE"
    assert shadow["hypothetical"]["mode"] == "SHADOW"

    # The audit and the wire tell the SAME story — the preview mirrors the
    # audit's content, so a stored plan and an audit row never disagree.
    assert body["risk"]["shadow_statistical"] == shadow
    assert body["risk"]["comparison"] == shadow["comparison"]
    assert body["risk"]["binding_constraints"] == details["binding_constraints"]

    # §19 correlation regime of the book PLUS the candidate's ticker.
    state = shadow["correlation_state"]
    assert state["state"] in {"NORMAL", "ELEVATED", "CONVERGING", "UNAVAILABLE"}
    assert state["n_pairs"] >= 1
    assert any(BULL_TICKER in pair for pair in state["worst_pairs"])


async def test_risk_decision_carries_the_vol_targeting_ewma_side_by_side(client):
    """§14 side-by-side: the EWMA forecast is logged NEXT TO the multiplier
    that actually scaled the budget, which stays the crude proxy's."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    await preview(client, BULL_TICKER)
    details = (await _latest_risk_decision(client, BULL_TICKER))["details"]

    ewma = details["shadow"]["vol_targeting_ewma"]
    assert ewma["forecast"] is not None and ewma["forecast"] > 0
    assert ewma["multiplier"] is not None
    # THE MULTIPLIER IN FORCE is the top-level one — the proxy's, untouched.
    assert ewma["multiplier_in_force"] == details["budget_multiplier"]
    assert "changed nothing" in ewma["note"]


# ---------------------------------------------------------------------------
# (e) /api/portfolio/risk: correlation state + the EWMA fields
# ---------------------------------------------------------------------------


async def test_portfolio_risk_serves_correlation_state_and_ewma(client):
    await seed_stock_position("AAPL", bars=200)
    await seed_stock_position("MSFT", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()

    state = body["statistical"]["correlation_state"]
    assert set(state) == {
        "normal_avg",
        "current_avg",
        "stress_avg",
        "delta",
        "state",
        "n_pairs",
        "n_obs_long",
        "n_obs_short",
        "n_obs_stress",
        "worst_pairs",
        "reason",
        # ADDED closing compliance §18 (row 18): the rolling Spearman average
        # over the SAME short window as ``current_avg``. Served beside its
        # Pearson twin so a rank-vs-linear divergence is visible.
        "current_avg_spearman",
    }
    assert state["n_pairs"] == 1  # one pair: AAPL × MSFT
    assert state["state"] in {"NORMAL", "ELEVATED", "CONVERGING"}
    assert -1.0 <= state["current_avg"] <= 1.0
    # delta = current − normal, exactly (contract §7.4).
    assert state["delta"] == pytest.approx(
        state["current_avg"] - state["normal_avg"]
    )
    assert state["worst_pairs"] == [["AAPL", "MSFT", state["current_avg"]]]

    vt = body["vol_targeting"]
    assert vt["ewma_sigma_p_annualized_pct_nav"] > 0
    assert vt["multiplier_ewma"] is not None
    # The multiplier IN FORCE is still the crude proxy's (unchanged, §14).
    assert vt["multiplier"] == pytest.approx(
        min(max(0.12 / vt["forecast_vol"], 0.5), 1.2)
    )


async def test_correlation_state_is_null_with_a_single_ticker(client):
    """A correlation needs a pair: one name in the book is an honest null,
    not a state object full of nulls."""
    await seed_stock_position("AAPL", bars=200)
    body = (await client.get("/api/portfolio/risk")).json()
    assert body["statistical"]["correlation_state"] is None


async def test_ewma_fields_are_null_without_a_book(client):
    """No book P&L series ⇒ no EWMA forecast. Honest nulls, never a zero."""
    body = (await client.get("/api/portfolio/risk")).json()
    vt = body["vol_targeting"]
    assert vt["ewma_sigma_p_annualized_pct_nav"] is None
    assert vt["multiplier_ewma"] is None
    assert vt["multiplier"] == 1.0  # unchanged: no adjustment


# ---------------------------------------------------------------------------
# (f) The stored trade plan carries the Phase C keys
# ---------------------------------------------------------------------------


async def test_generated_plan_preview_carries_the_phase_c_risk_keys(client):
    """``plans.py`` stores the preview payload verbatim, so a generated
    plan's ``preview.risk`` must already carry the Phase C keys — nothing in
    plans.py needs to change, and this test is what pins that."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    r = await client.post("/api/plans/generate", json={"ticker": BULL_TICKER})
    assert r.status_code == 201, r.text
    plan_id = r.json()["id"]

    detail = (await client.get(f"/api/plans/{plan_id}")).json()
    risk = detail["preview"]["risk"]
    assert risk is not None
    for key in ("comparison", "binding_constraints", "shadow_statistical"):
        assert key in risk, key
    assert set(risk["comparison"]) == COMPARISON_KEYS
    assert risk["shadow_statistical"]["hypothetical"]["mode"] == "SHADOW"
    # The plan's own decision is Tier 0's, untouched by the shadow layer.
    assert risk["decision"] == detail["preview"]["risk"]["decision"]


# ---------------------------------------------------------------------------
# (g) Phase K: shadow.event joins the SAME audit block — and changes nothing
# ---------------------------------------------------------------------------


async def test_shadow_event_appears_without_moving_the_approved_quantity(client):
    """Phase K (§62-§67) lands ``shadow.event`` beside ``shadow.statistical``
    in the very same RISK_DECISION detail, and the approved quantity is
    BYTE-IDENTICAL with and without an upcoming event in the registry.

    This lives here, next to the Phase C/D shadow tests, on purpose: the
    guarantee is the same one every layer in this file defends (§70 — a SHADOW
    model cannot veto), and the file where a future layer's regression would
    be noticed is the file that already pins the others.

    The planted event is deliberately the §65 worked example — an 8.8% implied
    move 1.3 days out, which the documented classifier table puts at EXTREME —
    so the event layer is not merely PRESENT but at its loudest rung, emitting
    a cap into the hypothetical verdict. The approved quantity still does not
    move, because that cap reaches ``_pretrade_statistical_shadow``'s
    hypothetical ``extra_caps`` and never ``assess``'s.
    """
    from datetime import timedelta

    from apps.gateway.db import EventOptionMetricRow, EventRow
    from libs.trading_core.events.implied_move import BASIS_LIVE, STATUS_OK
    from libs.trading_core.models.enums import (
        EventSession,
        EventSourceKind,
        EventStatus,
        EventType,
    )

    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    before_body = await preview(client, BULL_TICKER)
    before = await _latest_risk_decision(client, BULL_TICKER)
    # Nothing planted yet: the block is PRESENT and says there is no event —
    # an absent key could not be told apart from a missing feature.
    assert before["details"]["shadow"]["event"]["snapshot"] is None

    now = datetime.now(timezone.utc)
    when = now + timedelta(days=1.3)
    async with SessionLocal() as s:
        row = EventRow(
            event_key=f"EARNINGS:{BULL_TICKER}:{when.date().isoformat()}",
            event_type=EventType.EARNINGS.value,
            title=f"{BULL_TICKER} earnings",
            ticker=BULL_TICKER,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=EventSession.AFTER_MARKET.value,
            status=EventStatus.CONFIRMED.value,
            source=EventSourceKind.COMPANY_IR_SEC.value,
            source_name="sec_edgar",
            revision_history=[],
        )
        s.add(row)
        await s.commit()
        event_id = row.id
    async with SessionLocal() as s:
        s.add(
            EventOptionMetricRow(
                event_id=event_id,
                as_of=now,
                basis=BASIS_LIVE,
                implied_move_pct=8.8,
                iv_before=0.62,
                status=STATUS_OK,
                notes={},
            )
        )
        await s.commit()

    after_body = await preview(client, BULL_TICKER)
    after = await _latest_risk_decision(client, BULL_TICKER)

    block = after["details"]["shadow"]["event"]
    assert block["snapshot"]["event_risk_state"] == "EXTREME"
    assert block["enforcement"] == "SHADOW"
    # §64: the sample size travels with the statistics even at n=0 (no prior
    # prints were planted here — only this event's own straddle).
    assert block["snapshot"]["historical"]["n"] == 0
    assert block["verdict"]["would_warn"] is True

    # THE GUARANTEE. Tier 0's decision is untouched on both surfaces.
    assert after_body["risk"]["approved_quantity"] == (
        before_body["risk"]["approved_quantity"]
    )
    assert after_body["risk"]["decision"] == before_body["risk"]["decision"]
    assert after_body["risk"]["reason_codes"] == before_body["risk"]["reason_codes"]
    assert after_body["risk"]["trade_risk_usd"] == (
        before_body["risk"]["trade_risk_usd"]
    )
    assert after["details"]["approved_quantity"] == (
        before["details"]["approved_quantity"]
    )
    assert after_body["gates"] == before_body["gates"]
    # No event reason code reached the engine — the cap bound the HYPOTHETICAL
    # verdict only. (`hypothetical.binding` may name it; `reason_codes` is the
    # engine's own list and must not.)
    assert not any("EVENT" in c for c in after["details"]["reason_codes"])
