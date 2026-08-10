"""Tests for GET /api/watchlist/{ticker}/options (plan §9, §34, §4.2).

The stub chain is deterministic by construction (crc32-seeded per node, keyed
by symbol/day), and the stub bar walk depends only on (symbol, bar count), so
NVDA's spot — and hence the whole chain shape — is stable across runs. Tests
re-check the §9.1 filters against the response numbers independently rather
than trusting the endpoint's own verdicts.
"""
import math
from datetime import date

from libs.trading_core.contracts import SelectorParams

TOP_LEVEL_KEYS = {
    "ticker",
    "as_of",
    "spot",
    "source",
    "direction_used",
    "summary",
    "expiries",
    "chain",
}
SUMMARY_KEYS = {
    "atm_iv",
    "expected_move_pct",
    "rv20",
    "iv_rv_spread",
    "iv_rank",
    "iv_rank_note",
}
ROW_KEYS = {
    "expiry",
    "dte",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "last",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "eligible",
    "fail_reasons",
    "candidate_rank",
    "score",
    "score_components",
}

# Mirrors the bs module defaults used by the stub chain (r=0.04, q=0).
RISK_FREE_RATE = 0.04
IV_FLOOR = 0.05  # stub chain absolute IV floor


async def get_options(client, ticker="NVDA", **params):
    await client.post("/api/watchlist", json={"ticker": ticker})
    r = await client.get(f"/api/watchlist/{ticker}/options", params=params)
    assert r.status_code == 200
    return r.json()


async def test_options_404_for_non_watchlist_ticker(client):
    """Historical data may exist ONLY for Watchlist symbols (plan §4.2)."""
    r = await client.get("/api/watchlist/NVDA/options")
    assert r.status_code == 404
    assert "not on the watchlist" in r.json()["detail"]


async def test_options_rejects_unknown_direction(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.get(
        "/api/watchlist/NVDA/options", params={"direction": "SIDEWAYS"}
    )
    assert r.status_code == 422


async def test_options_contract_shape(client):
    body = await get_options(client)

    assert set(body) == TOP_LEVEL_KEYS
    assert body["ticker"] == "NVDA"
    assert body["source"] == "stub"
    date.fromisoformat(body["as_of"])  # valid ISO-8601 chain snapshot date
    assert isinstance(body["spot"], float) and body["spot"] > 0
    assert body["direction_used"] in ("BULL", "BEAR", None)

    # Non-trivial chain: >= 100 rows, both rights, >= 4 expiries.
    chain = body["chain"]
    assert len(chain) >= 100
    assert {row["right"] for row in chain} == {"C", "P"}
    expiries = body["expiries"]
    assert len(expiries) >= 4
    assert {row["expiry"] for row in chain} == {e["expiry"] for e in expiries}
    # Expiries ascending with consistent, positive DTE.
    assert expiries == sorted(expiries, key=lambda e: e["expiry"])
    for e in expiries:
        assert isinstance(e["dte"], int) and e["dte"] > 0
        assert (
            date.fromisoformat(e["expiry"]) - date.fromisoformat(body["as_of"])
        ).days == e["dte"]

    for row in chain:
        assert set(row) == ROW_KEYS, f"row keys mismatch: {set(row) ^ ROW_KEYS}"
        assert row["right"] in ("C", "P")
        assert isinstance(row["volume"], int) and row["volume"] >= 0
        assert isinstance(row["open_interest"], int) and row["open_interest"] >= 0
        assert isinstance(row["eligible"], bool)
        assert isinstance(row["fail_reasons"], list)
        # Calls have positive delta, puts negative (bs conventions).
        assert (row["delta"] > 0) == (row["right"] == "C")


async def test_options_deterministic_within_a_day(client):
    """Two calls on the same day return byte-identical payloads (plan §22.1)."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r1 = await client.get("/api/watchlist/NVDA/options")
    r2 = await client.get("/api/watchlist/NVDA/options")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()


async def test_options_is_read_only_no_audit_event(client):
    """Chain reads are read-only: no audit event beyond the one bar backfill."""
    # First call: watchlist add + lazy backfill may write their own events.
    await get_options(client)
    before = (await client.get("/api/audit", params={"entity_id": "NVDA"})).json()
    assert [e["action"] for e in before].count("DATA_BACKFILL") == 1
    # Warm chain reads write NOTHING (audit only on state changes).
    await client.get("/api/watchlist/NVDA/options")
    await client.get("/api/watchlist/NVDA/options", params={"direction": "BULL"})
    after = (await client.get("/api/audit", params={"entity_id": "NVDA"})).json()
    assert after == before


async def test_chain_price_sanity(client):
    """OHLC-style sanity on every row (plan §9 stub chain invariants)."""
    body = await get_options(client)
    for row in body["chain"]:
        assert row["dte"] > 0
        assert row["iv"] >= IV_FLOOR
        assert row["mid"] > 0
        assert row["bid"] < row["mid"] < row["ask"]
        assert row["spread_pct"] > 0
        assert math.isclose(
            row["spread_pct"],
            (row["ask"] - row["bid"]) / row["mid"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert row["last"] > 0  # mid perturbed +/-1%, never zero


async def test_eligible_rows_satisfy_section_9_1_filters(client):
    """Independently re-check every §9.1 filter on rows marked eligible."""
    body = await get_options(client, direction="BULL")
    p = SelectorParams()
    eligible = [r for r in body["chain"] if r["eligible"]]
    assert eligible  # the stub chain must produce candidates for NVDA
    for row in eligible:
        assert row["right"] == "C"  # BULL -> calls only (plan §5 long-only)
        assert p.dte_min <= row["dte"] <= p.dte_max
        assert p.abs_delta_min <= abs(row["delta"]) <= p.abs_delta_max
        assert row["open_interest"] >= p.min_open_interest
        assert row["volume"] >= p.min_volume
        assert row["spread_pct"] <= p.max_spread_pct
        assert row["mid"] > 0
        assert abs(row["theta"]) / row["mid"] <= p.max_theta_premium_pct
        assert row["fail_reasons"] == []
        assert isinstance(row["score"], float)
        assert set(row["score_components"]) == {
            "liquidity",
            "theta_burden",
            "delta_fit",
        }
    # Ineligible rows carry explainable reasons and honest nulls (plan §36).
    for row in body["chain"]:
        if not row["eligible"]:
            assert row["fail_reasons"]
            assert row["candidate_rank"] is None


async def test_bull_candidates_are_ranked_eligible_calls(client):
    body = await get_options(client, direction="BULL")
    assert body["direction_used"] == "BULL"
    ranked = [r for r in body["chain"] if r["candidate_rank"] is not None]
    assert ranked  # NVDA's stub chain always has eligible calls
    assert len(ranked) <= SelectorParams().top_n
    assert sorted(r["candidate_rank"] for r in ranked) == list(
        range(1, len(ranked) + 1)
    )
    for row in ranked:
        assert row["eligible"] is True
        assert row["right"] == "C"
        assert isinstance(row["score"], float)


async def test_bear_direction_yields_only_put_candidates(client):
    body = await get_options(client, direction="BEAR")
    assert body["direction_used"] == "BEAR"
    ranked = [r for r in body["chain"] if r["candidate_rank"] is not None]
    assert ranked  # NVDA's stub chain always has eligible puts
    for row in ranked:
        assert row["right"] == "P"
        assert row["eligible"] is True
    # And no call is eligible under BEAR — wrong side (plan §5 long-only).
    assert not any(r["eligible"] for r in body["chain"] if r["right"] == "C")


async def test_auto_direction_is_consistent(client):
    """AUTO resolves via the directional engine; NEUTRAL means NO candidates
    with the §44 rule 18 no-signal reason on every row."""
    body = await get_options(client, direction="AUTO")
    if body["direction_used"] is None:
        for row in body["chain"]:
            assert row["eligible"] is False
            assert row["candidate_rank"] is None
            assert (
                "no directional signal — NO TRADE is a valid output"
                in row["fail_reasons"]
            )
    else:
        side = "C" if body["direction_used"] == "BULL" else "P"
        for row in body["chain"]:
            if row["candidate_rank"] is not None:
                assert row["right"] == side


async def test_put_call_parity_at_the_money(client):
    """C - P = S - K*exp(-rT) at the sampled ATM node (same IV both rights).

    The stub mids are theoretical Black-Scholes prices (q=0), so parity holds
    up to the 4-decimal price rounding; tolerance is kept loose anyway.
    """
    body = await get_options(client)
    spot = body["spot"]
    # Sample the nearest expiry with dte >= 30 (the summary's ATM node).
    expiry = min(e["expiry"] for e in body["expiries"] if e["dte"] >= 30)
    calls = [
        r for r in body["chain"] if r["expiry"] == expiry and r["right"] == "C"
    ]
    atm_call = min(calls, key=lambda r: (abs(r["strike"] - spot), r["strike"]))
    atm_put = next(
        r
        for r in body["chain"]
        if r["expiry"] == expiry
        and r["strike"] == atm_call["strike"]
        and r["right"] == "P"
    )
    assert atm_call["iv"] == atm_put["iv"]  # v0: no skew at a node
    t_years = atm_call["dte"] / 365.0
    parity_rhs = spot - atm_call["strike"] * math.exp(-RISK_FREE_RATE * t_years)
    assert abs((atm_call["mid"] - atm_put["mid"]) - parity_rhs) < 0.01


async def test_summary_block(client):
    body = await get_options(client)
    summary = body["summary"]
    assert set(summary) == SUMMARY_KEYS

    # NVDA's chain always has a 30d+ expiry, and 600 stored bars cover rv20.
    assert isinstance(summary["atm_iv"], float) and summary["atm_iv"] >= IV_FLOOR
    assert (
        isinstance(summary["expected_move_pct"], float)
        and summary["expected_move_pct"] > 0
    )
    assert isinstance(summary["rv20"], float) and summary["rv20"] > 0
    assert math.isclose(
        summary["iv_rv_spread"],
        summary["atm_iv"] - summary["rv20"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )

    # iv_rank is an honest null with the note verbatim (house rule).
    assert summary["iv_rank"] is None
    assert (
        summary["iv_rank_note"] == "requires IV history — arrives with real chain data"
    )

    # expected_move_pct is the ATM straddle over spot: recompute from the chain.
    spot = body["spot"]
    expiry = min(e["expiry"] for e in body["expiries"] if e["dte"] >= 30)
    calls = [
        r for r in body["chain"] if r["expiry"] == expiry and r["right"] == "C"
    ]
    atm_call = min(calls, key=lambda r: (abs(r["strike"] - spot), r["strike"]))
    atm_put = next(
        r
        for r in body["chain"]
        if r["expiry"] == expiry
        and r["strike"] == atm_call["strike"]
        and r["right"] == "P"
    )
    assert summary["atm_iv"] == atm_call["iv"]
    assert math.isclose(
        summary["expected_move_pct"],
        (atm_call["mid"] + atm_put["mid"]) / spot,
        rel_tol=1e-9,
    )
