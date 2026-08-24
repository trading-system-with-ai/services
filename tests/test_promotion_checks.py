"""§4.3 promotion readiness checks on POST /api/trading-pool.

The check ladder for one ticker (NVDA — the stub provider deterministically
yields a backtest with closed trades for it):

1. fresh (no bars, no backtest) -> 422 with MIN_HISTORY / BACKTEST_COMPLETED
   / BACKTEST_TRADES failed (with honest counts) and LIQUIDITY passed in
   REPORT mode (audit §7.3 / B0: ADV20 unmeasurable with 0 bars -> honest
   UNAVAILABLE, never a block);
2. after analysis backfills 600 bars -> MIN_HISTORY passes, backtest checks
   still fail;
3. after a real COMPLETED backtest -> all checks pass -> 201, checks in the
   response AND in the TRADING_POOL_ADD audit details;
4. acknowledge_risks=True overrides failed checks -> 201, and the FAILED
   check results stay permanently visible in the audit details (§4.3, §38).
"""

TICKER = "NVDA"
CHECK_ORDER = ["MIN_HISTORY", "BACKTEST_COMPLETED", "BACKTEST_TRADES", "LIQUIDITY"]
# REPORT-mode LIQUIDITY readiness detail prefix (audit §7.3 / B0) — the shared
# evaluator's audit-exact wording, rendered verbatim by the UI.
LIQUIDITY_REPORT_PREFIX = "underlying liquidity (REPORT mode, research limits): "


def by_name(checks):
    assert [c["name"] for c in checks] == CHECK_ORDER  # evaluated in order
    for c in checks:
        assert set(c) == {"name", "passed", "detail"}
        assert isinstance(c["passed"], bool)
        assert isinstance(c["detail"], str) and c["detail"]
    return {c["name"]: c for c in checks}


async def promote(client, ticker=TICKER, **extra):
    return await client.post("/api/trading-pool", json={"ticker": ticker, **extra})


async def pool_add_audit_details(client, ticker=TICKER):
    r = await client.get("/api/audit", params={"entity_id": ticker})
    events = [e for e in r.json() if e["action"] == "TRADING_POOL_ADD"]
    assert len(events) == 1
    assert events[0]["actor_type"] == "USER"
    return events[0]["details"]


async def test_fresh_ticker_fails_three_checks_with_honest_details(client):
    await client.post("/api/watchlist", json={"ticker": TICKER})
    r = await promote(client)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["message"] == (
        "promotion checks failed — review and acknowledge to proceed"
    )
    checks = by_name(detail["checks"])

    # MIN_HISTORY: no bars stored yet — the detail carries the real count.
    assert checks["MIN_HISTORY"]["passed"] is False
    assert "0 stored daily bars" in checks["MIN_HISTORY"]["detail"]
    assert "200" in checks["MIN_HISTORY"]["detail"]  # RegimeParams.sma_slow

    assert checks["BACKTEST_COMPLETED"]["passed"] is False
    assert "no COMPLETED backtest" in checks["BACKTEST_COMPLETED"]["detail"]

    # BACKTEST_TRADES: 0 trades = no trade evidence.
    assert checks["BACKTEST_TRADES"]["passed"] is False
    assert "0 trades" in checks["BACKTEST_TRADES"]["detail"]

    # LIQUIDITY: REPORT mode (audit §7.3 / B0) — with 0 stored bars nothing
    # is measurable, so the detail says UNAVAILABLE honestly, and the check
    # still passes (never a block until promoted).
    assert checks["LIQUIDITY"]["passed"] is True
    liq = checks["LIQUIDITY"]["detail"]
    assert liq.startswith(LIQUIDITY_REPORT_PREFIX)
    assert "ADV20 n/a" in liq
    assert "verdict UNAVAILABLE" in liq
    assert "0 stored volume(s), need 20" in liq
    assert "Massive" not in liq  # the stale stub text is gone

    # A blocked promotion changes nothing: pool stays empty, no audit row.
    assert (await client.get("/api/trading-pool")).json() == []
    r = await client.get("/api/audit", params={"entity_id": TICKER})
    assert "TRADING_POOL_ADD" not in [e["action"] for e in r.json()]


async def test_backfilled_history_passes_min_history_only(client):
    await client.post("/api/watchlist", json={"ticker": TICKER})
    # Analysis lazily backfills 600 daily bars (plan §4.2).
    r = await client.get(f"/api/watchlist/{TICKER}/analysis")
    assert r.status_code == 200

    r = await promote(client)
    assert r.status_code == 422
    checks = by_name(r.json()["detail"]["checks"])
    assert checks["MIN_HISTORY"]["passed"] is True
    assert "600 stored daily bars" in checks["MIN_HISTORY"]["detail"]
    # Bars alone are not evidence — the backtest checks still fail.
    assert checks["BACKTEST_COMPLETED"]["passed"] is False
    assert checks["BACKTEST_TRADES"]["passed"] is False
    # LIQUIDITY (REPORT mode): with 600 stored bars ADV20 IS measured (the
    # stub's synthetic volume centre is 1,000,000 sh, far above the 100,000
    # research floor) and the hypothetical verdict is stated; the order and
    # spread components stay honestly n/a for a readiness check.
    liq = checks["LIQUIDITY"]["detail"]
    assert checks["LIQUIDITY"]["passed"] is True
    assert liq.startswith(LIQUIDITY_REPORT_PREFIX)
    assert "ADV20 n/a" not in liq
    assert "would PASS" in liq
    assert "order n/a; quote spread n/a" in liq
    assert "(readiness check: no order size, no live quote)" in liq


async def test_completed_backtest_passes_all_checks_and_audits_them(client):
    await client.post("/api/watchlist", json={"ticker": TICKER})
    # A real backtest run (it backfills bars itself, plan §20).
    r = await client.post("/api/backtests", json={"ticker": TICKER})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    backtest_id = body["id"]
    trade_count = body["metrics"]["num_trades"]
    assert trade_count >= 1  # the stub's NVDA series produces trades

    r = await promote(client)
    assert r.status_code == 201
    out = r.json()
    checks = by_name(out["promotion_checks"])
    assert all(c["passed"] for c in checks.values())
    assert str(backtest_id) in checks["BACKTEST_COMPLETED"]["detail"]
    assert str(trade_count) in checks["BACKTEST_TRADES"]["detail"]
    assert out["risks_acknowledged"] is False
    assert out["trading_enabled"] is False  # promotion is never a purchase

    # Audit details ALWAYS carry the checks + acknowledged flag (§4.3, §38).
    details = await pool_add_audit_details(client)
    assert details["risks_acknowledged"] is False
    audited = by_name(details["promotion_checks"])
    assert all(c["passed"] for c in audited.values())


async def test_acknowledged_override_records_failed_checks_in_audit(client):
    await client.post("/api/watchlist", json={"ticker": TICKER})
    r = await promote(client, acknowledge_risks=True)
    assert r.status_code == 201
    out = r.json()
    assert out["risks_acknowledged"] is True
    checks = by_name(out["promotion_checks"])
    assert checks["MIN_HISTORY"]["passed"] is False  # override, not amnesia

    # The override is PERMANENTLY visible: the audit row keeps the failed
    # check results next to the acknowledged flag (§4.3, §38).
    details = await pool_add_audit_details(client)
    assert details["risks_acknowledged"] is True
    audited = by_name(details["promotion_checks"])
    assert audited["MIN_HISTORY"]["passed"] is False
    assert audited["BACKTEST_COMPLETED"]["passed"] is False
    assert audited["BACKTEST_TRADES"]["passed"] is False
    assert audited["LIQUIDITY"]["passed"] is True
