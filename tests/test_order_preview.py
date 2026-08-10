"""§42 property tests for POST /api/orders/preview (the §10 gate chain).

"No rejected ticker may produce an order" (§42): a symbol failing any gate
must come back with risk null (or REJECT), a non-empty why_not_trade, and the
veto recorded in exactly one SYSTEM RISK_DECISION audit event (§38).
"""
import json

GATE_ORDER = [
    "TRADING_POOL_AUTHORIZATION",
    "DATA_QUALITY",
    "REGIME",
    "DIRECTIONAL_SIGNAL",
    "VOLATILITY",
    "INSTRUMENT",
    "LIQUIDITY",
    "CONTRACT_SELECTION",
    "RISK_APPROVAL",
]
SKIP_EARLIER_FAIL = "not evaluated: earlier gate failed"
SKIP_NO_OPTION_DATA = "no option/quote data yet — arrives with the Massive integration"
SKIP_STOCK_ORDER = "stock order — no contract selection needed"

# The stub provider is deterministic per symbol (RNG seeded by the symbol
# name, so the price PATH is identical for any end date): GOOGL yields a bull
# regime with a BULL bias, driving the chain through RISK_APPROVAL.
BULL_TICKER = "GOOGL"


async def authorize(client, ticker, *, enable_symbol=True, resume=True):
    """Watchlist -> Trading Pool (-> per-symbol enable) (-> global resume)."""
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    r = await client.post("/api/trading-pool", json={"ticker": ticker})
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
        "gates",
        "signal",
        "proposed",
        "risk",
        "why_trade",
        "why_not_trade",
    }
    assert body["ticker"] == ticker
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


async def test_watchlist_only_symbol_vetoed_at_gate_one(client):
    """Not in the pool -> gate 1 FAIL, everything later SKIPPED, risk null."""
    r = await client.post("/api/watchlist", json={"ticker": "NVDA"})
    assert r.status_code == 201

    body = await preview(client, "NVDA")
    assert_contract_shape(body, "NVDA")

    gates = body["gates"]
    assert gates[0]["name"] == "TRADING_POOL_AUTHORIZATION"
    assert gates[0]["status"] == "FAIL"
    assert "not in the Trading Pool" in gates[0]["detail"]
    for g in gates[1:]:
        assert g["status"] == "SKIPPED"
        assert g["detail"] == SKIP_EARLIER_FAIL

    assert body["risk"] is None
    assert body["signal"] == {"edge": None, "bias": None, "strength": None}
    assert body["why_not_trade"]  # "No rejected ticker may produce an order" (§42)

    event = await get_single_risk_decision_event(client, "NVDA")
    assert event["details"]["decision"] == "VETOED"
    assert event["details"]["veto_gate"] == "TRADING_POOL_AUTHORIZATION"


async def test_pool_symbol_with_trading_disabled_fails_gate_one(client):
    """Promotion is authorization, not activation: disabled symbol is vetoed."""
    await authorize(client, "NVDA", enable_symbol=False, resume=True)

    body = await preview(client, "NVDA")
    gates = body["gates"]
    assert gates[0]["status"] == "FAIL"
    assert "not enabled for NVDA" in gates[0]["detail"]
    assert body["risk"] is None
    assert body["why_not_trade"]
    await get_single_risk_decision_event(client, "NVDA")


async def test_global_kill_switch_paused_fails_gate_one(client):
    """Kill switch has priority (§18): gate 1 FAIL names the kill switch."""
    await authorize(client, "NVDA", enable_symbol=True, resume=False)

    body = await preview(client, "NVDA")
    gates = body["gates"]
    assert gates[0]["status"] == "FAIL"
    assert "kill switch" in gates[0]["detail"]
    for g in gates[1:]:
        assert g["status"] == "SKIPPED"
        assert g["detail"] == SKIP_EARLIER_FAIL
    assert body["risk"] is None
    assert body["why_not_trade"]
    await get_single_risk_decision_event(client, "NVDA")


async def test_fully_authorized_chain_runs_to_risk_or_fails_legitimately(client):
    """Authorized + enabled: gates run through RISK_APPROVAL, or FAIL on a
    legitimate regime/signal veto — structure and audit hold either way."""
    await authorize(client, BULL_TICKER)

    body = await preview(client, BULL_TICKER)
    assert_contract_shape(body, BULL_TICKER)

    gates = body["gates"]
    by_name = {g["name"]: g for g in gates}
    assert by_name["TRADING_POOL_AUTHORIZATION"]["status"] == "PASS"
    assert by_name["DATA_QUALITY"]["status"] == "PASS"  # stub data always passes

    first_fail = next((g["name"] for g in gates if g["status"] == "FAIL"), None)
    if first_fail in (None, "RISK_APPROVAL"):
        # Chain reached the risk engine. VOLATILITY is now a REAL §7
        # classification off the stub chain summary (GOOGL deterministically
        # classifies NORMAL and the §8 matrix keeps LONG_STOCK for it);
        # LIQUIDITY stays the exact V1 skip and CONTRACT_SELECTION skips for
        # a stock order.
        assert by_name["VOLATILITY"]["status"] == "PASS"
        assert "vol regime" in by_name["VOLATILITY"]["detail"]
        assert by_name["LIQUIDITY"]["status"] == "SKIPPED"
        assert by_name["LIQUIDITY"]["detail"] == SKIP_NO_OPTION_DATA
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
