"""§42 property tests for POST /api/orders/preview (the §10 gate chain).

"No rejected ticker may produce an order" (§42): a symbol failing any gate
must come back with risk null (or REJECT), a non-empty why_not_trade, and the
veto recorded in exactly one SYSTEM RISK_DECISION audit event (§38).

RESEARCH MODE (upgrade 2026-08-12 §15/§16): preview is the RESEARCH chain —
TRADING_POOL_AUTHORIZATION is not a research gate; pool/kill-switch facts are
reported in ``execution_authorization`` and enforced only by the approve path
(pinned in test_orders_execution.py, which stays execution-mode).
"""
import json

# The §16 research gate chain — no TRADING_POOL_AUTHORIZATION (upgrade §15).
GATE_ORDER = [
    "DATA_QUALITY",
    "REGIME",
    "DIRECTIONAL_SIGNAL",
    "VOLATILITY",
    "INSTRUMENT",
    "SQUEEZE_RISK",
    "LIQUIDITY",
    "CONTRACT_SELECTION",
    "RISK_APPROVAL",
]
SKIP_EARLIER_FAIL = "not evaluated: earlier gate failed"
SKIP_STOCK_ORDER = "stock order — no contract selection needed"
# Gate 7 (underlying LIQUIDITY) runs in REPORT mode (risk-engine audit §7.3 /
# B0): PASS with the measured numbers + hypothetical verdict in the detail.
LIQUIDITY_REPORT_PREFIX = "underlying liquidity (REPORT mode, research limits): "

# The stub provider is deterministic per symbol (RNG seeded by the symbol
# name, so the price PATH is identical for any end date): GOOGL yields a bull
# regime with a BULL bias, driving the chain through RISK_APPROVAL.
BULL_TICKER = "GOOGL"


async def authorize(client, ticker, *, enable_symbol=True, resume=True):
    """Watchlist -> Trading Pool (-> per-symbol enable) (-> global resume)."""
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    # acknowledge_risks: the ticker has no stored history/backtest at promote
    # time, so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    if enable_symbol:
        r = await client.post(
            f"/api/trading-pool/{ticker}/trading", json={"enabled": True}
        )
        assert r.status_code == 200
    if resume:
        r = await client.post("/api/trading/resume", json={})
        assert r.status_code == 200


async def preview(client, ticker, quantity=None):
    body = {"ticker": ticker}
    if quantity is not None:
        body["quantity"] = quantity
    r = await client.post("/api/orders/preview", json=body)
    assert r.status_code == 200
    return r.json()


async def get_single_risk_decision_event(client, ticker):
    """Every preview writes exactly ONE SYSTEM RISK_DECISION event (§38)."""
    r = await client.get("/api/audit", params={"entity_id": ticker})
    events = [e for e in r.json() if e["action"] == "RISK_DECISION"]
    assert len(events) == 1
    event = events[0]
    assert event["actor_type"] == "SYSTEM"
    assert event["entity_id"] == ticker
    return event


def assert_contract_shape(body, ticker):
    assert set(body) == {
        "ticker",
        "as_of",
        "mode",
        "execution_authorization",
        "gates",
        "signal",
        "proposed",
        "risk",
        "exit_plan",
        "why_trade",
        "why_not_trade",
    }
    # §24: the exit plan rides on EVERY preview — the user sees how a
    # position would be exited before any Apply.
    exit_plan = body["exit_plan"]
    assert exit_plan["signal_invalidation"]
    assert exit_plan["atr_trail"]
    assert exit_plan["time_stop"]
    assert exit_plan["profit_target"] is None  # V1: honest null, not invented
    assert body["ticker"] == ticker
    assert body["mode"] == "research"
    auth = body["execution_authorization"]
    assert set(auth) == {
        "authorized",
        "in_trading_pool",
        "symbol_trading_enabled",
        "global_trading_enabled",
        "missing",
    }
    # §20: authorized is the conjunction of the three facts, and every
    # missing authorization is named.
    assert auth["authorized"] == (
        auth["in_trading_pool"]
        and auth["symbol_trading_enabled"]
        and auth["global_trading_enabled"]
    )
    assert auth["authorized"] == (auth["missing"] == [])
    assert [g["name"] for g in body["gates"]] == GATE_ORDER
    for g in body["gates"]:
        assert set(g) == {"name", "status", "detail"}
        assert g["status"] in {"PASS", "FAIL", "SKIPPED"}
        assert isinstance(g["detail"], str) and g["detail"]
    assert set(body["signal"]) == {"edge", "bias", "strength"}
    # Option wiring additions: instrument is the §8 matrix verdict (null when
    # the chain vetoed before the matrix ran — honest null), vol_regime the
    # §7 classification, contract the §9 top-ranked candidate (null for
    # stock / no trade).
    assert set(body["proposed"]) == {
        "instrument",
        "vol_regime",
        "instrument_rationale",
        "contract",
        "spread",
        "entry_price",
        "stop_distance",
        "quantity_requested",
    }
    assert body["proposed"]["instrument"] in {
        None,
        "LONG_STOCK",
        "LONG_CALL",
        "LONG_PUT",
        "NO_TRADE",
    }
    assert body["proposed"]["vol_regime"] in {
        None,
        "LOW",
        "NORMAL",
        "HIGH",
        "EXTREME",
    }
    assert isinstance(body["proposed"]["instrument_rationale"], list)
    # §33: both narrative lists are always present.
    assert isinstance(body["why_trade"], list)
    assert isinstance(body["why_not_trade"], list)


async def test_watchlist_only_symbol_gets_full_research_chain(client):
    """Upgrade §15/§45: a Watchlist symbol generates its complete research
    plan — the chain starts at DATA_QUALITY and pool membership is reported
    as an unmet EXECUTION authorization, never a research veto."""
    r = await client.post("/api/watchlist", json={"ticker": "NVDA"})
    assert r.status_code == 201

    body = await preview(client, "NVDA")
    assert_contract_shape(body, "NVDA")

    gates = body["gates"]
    assert gates[0]["name"] == "DATA_QUALITY"
    assert gates[0]["status"] == "PASS"  # research ran — not skipped

    auth = body["execution_authorization"]
    assert auth["authorized"] is False
    assert auth["in_trading_pool"] is False
    assert any("not in the Trading Pool" in m for m in auth["missing"])

    # The research verdict is whatever the market evidence says — but it WAS
    # computed: the signal block carries real numbers, not nulls.
    assert body["signal"]["edge"] is not None
    assert body["signal"]["bias"] is not None

    event = await get_single_risk_decision_event(client, "NVDA")
    assert event["details"]["mode"] == "research"
    assert event["details"]["execution_authorized"] is False
    assert event["details"]["veto_gate"] != "TRADING_POOL_AUTHORIZATION"


async def test_pool_symbol_with_trading_disabled_still_researches(client):
    """Promotion is authorization, not activation (§20): a disabled symbol
    still gets research; the missing enablement is named in the auth block."""
    await authorize(client, "NVDA", enable_symbol=False, resume=True)

    body = await preview(client, "NVDA")
    auth = body["execution_authorization"]
    assert auth["authorized"] is False
    assert auth["in_trading_pool"] is True
    assert auth["symbol_trading_enabled"] is False
    assert any("not enabled for NVDA" in m for m in auth["missing"])
    assert body["gates"][0]["name"] == "DATA_QUALITY"
    assert body["gates"][0]["status"] == "PASS"
    await get_single_risk_decision_event(client, "NVDA")


async def test_global_kill_switch_pauses_execution_not_research(client):
    """Kill switch pauses TRADING system-wide (§18) — research remains
    available; the pause is reported in the authorization block and §43's
    no-bypass rule is enforced by the approve path (execution mode)."""
    await authorize(client, "NVDA", enable_symbol=True, resume=False)

    body = await preview(client, "NVDA")
    auth = body["execution_authorization"]
    assert auth["authorized"] is False
    assert auth["global_trading_enabled"] is False
    assert any("kill switch" in m for m in auth["missing"])
    assert body["gates"][0]["name"] == "DATA_QUALITY"
    assert body["gates"][0]["status"] == "PASS"
    await get_single_risk_decision_event(client, "NVDA")


async def test_fully_authorized_chain_runs_to_risk_or_fails_legitimately(client):
    """Authorized + enabled: gates run through RISK_APPROVAL, or FAIL on a
    legitimate regime/signal veto — structure and audit hold either way."""
    await authorize(client, BULL_TICKER)

    body = await preview(client, BULL_TICKER)
    assert_contract_shape(body, BULL_TICKER)

    gates = body["gates"]
    by_name = {g["name"]: g for g in gates}
    assert body["execution_authorization"]["authorized"] is True
    assert by_name["DATA_QUALITY"]["status"] == "PASS"  # stub data always passes

    first_fail = next((g["name"] for g in gates if g["status"] == "FAIL"), None)
    if first_fail in (None, "RISK_APPROVAL"):
        # Chain reached the risk engine. VOLATILITY is now a REAL §7
        # classification off the stub chain summary (GOOGL deterministically
        # classifies NORMAL and the §8 matrix keeps LONG_STOCK for it);
        # LIQUIDITY is the REPORT-mode underlying check (audit §7.3 / B0):
        # PASS, with ADV20 measured off the stored bars and the hypothetical
        # verdict spelled out; CONTRACT_SELECTION skips for a stock order.
        assert by_name["VOLATILITY"]["status"] == "PASS"
        assert "vol regime" in by_name["VOLATILITY"]["detail"]
        assert by_name["LIQUIDITY"]["status"] == "PASS"
        assert by_name["LIQUIDITY"]["detail"].startswith(LIQUIDITY_REPORT_PREFIX)
        assert "would " in by_name["LIQUIDITY"]["detail"]  # PASS or FAIL, stated
        assert "Massive" not in by_name["LIQUIDITY"]["detail"]
        assert by_name["INSTRUMENT"]["status"] == "PASS"
        assert "§8" in by_name["INSTRUMENT"]["detail"]
        assert by_name["CONTRACT_SELECTION"]["status"] == "SKIPPED"
        assert by_name["CONTRACT_SELECTION"]["detail"] == SKIP_STOCK_ORDER

        assert body["proposed"]["instrument"] == "LONG_STOCK"
        assert body["proposed"]["contract"] is None
        assert body["proposed"]["vol_regime"] is not None
        assert body["proposed"]["instrument_rationale"]
        assert body["signal"]["bias"] == "BULL"
        assert body["proposed"]["entry_price"] > 0
        assert body["proposed"]["stop_distance"] > 0

        risk = body["risk"]
        assert risk is not None
        assert risk["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE", "REJECT"}
        if risk["decision"] == "REJECT":
            assert risk["reason_codes"]
            assert body["why_not_trade"]
        else:
            assert risk["approved_quantity"] > 0
            assert body["why_trade"]
    else:
        # Legitimate veto by the symbol's own regime/signal read (§6.1, §5).
        assert first_fail in {"REGIME", "DIRECTIONAL_SIGNAL"}
        assert body["risk"] is None
        assert body["why_not_trade"]
        idx = GATE_ORDER.index(first_fail)
        for g in gates[idx + 1 :]:
            assert g["status"] == "SKIPPED"
            assert g["detail"] == SKIP_EARLIER_FAIL

    await get_single_risk_decision_event(client, BULL_TICKER)


async def test_quantity_requested_is_honored(client):
    """approved_quantity may never exceed the requested quantity (§12.1)."""
    await authorize(client, BULL_TICKER)

    body = await preview(client, BULL_TICKER, quantity=5)
    assert body["proposed"]["quantity_requested"] == 5
    risk = body["risk"]
    # Deterministic stub: GOOGL is BULL in a bull regime, so risk runs.
    assert risk is not None
    assert risk["approved_quantity"] <= 5


async def test_response_has_no_nan_and_gates_in_exact_order(client):
    await authorize(client, BULL_TICKER)

    r = await client.post("/api/orders/preview", json={"ticker": BULL_TICKER})
    assert r.status_code == 200

    def reject_constant(name):
        raise AssertionError(f"non-finite JSON constant {name!r} in response")

    # json.loads with parse_constant trips on NaN/Infinity anywhere in the body.
    body = json.loads(r.text, parse_constant=reject_constant)
    assert [g["name"] for g in body["gates"]] == GATE_ORDER
