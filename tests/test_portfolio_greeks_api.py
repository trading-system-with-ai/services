"""GET /api/portfolio/risk additions (plan §12.4, §14, §16) + gate wiring.

Covers the §16 ``greeks`` block (hand-computed net delta shares / notionals
against the SAME regenerated chain the endpoint reads), the §14
``vol_targeting`` block, the ``kind`` field on §12.4 bucket rows including a
DYNAMIC bucket built from seeded perfectly-correlated bars, honest
``data_ok: false`` zero contributions, breach reporting on a deliberately
huge option book, and the §16/§14 wiring through the §10 RISK_APPROVAL gate
(greek-limit REJECT + "budget multiplier" detail).

Positions are seeded by DIRECT DB inserts; option rows carry opt_* fields
matching a REAL contract from the watchlisted ticker's chain, so the
endpoint's same-day chain regeneration finds the exact contract.
"""
from datetime import date, datetime, timedelta, timezone

import math

import pytest

from apps.gateway.db import Position, SessionLocal, StockBarDaily
from libs.trading_core.allocation import VolTargetParams
from libs.trading_core.risk import RiskLimits

LIMITS = RiskLimits()
VOL_PARAMS = VolTargetParams()
INITIAL_CASH = 100_000.0

OPTION_TICKER = "GOOGL"  # deterministic BULL ticker used across the suite

GREEKS_KEYS = {
    "net_delta_shares",
    "delta_adjusted_notional_usd",
    "delta_notional_pct_nav",
    "net_gamma",
    "net_theta_usd_per_day",
    "net_vega_usd",
    "limits",
    "breaches",
    "per_position",
}
PER_POSITION_KEYS = {
    "ticker",
    "instrument",
    "equivalent_shares",
    "delta_notional_usd",
    "gamma",
    "theta_usd_per_day",
    "vega_usd",
    "data_ok",
    "note",
}
VOL_TARGETING_KEYS = {
    "target_vol",
    "forecast_vol",
    "multiplier",
    "max_multiplier",
    "note",
}


async def add_watchlist(client, ticker):
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201


async def chain_call_near_atm(client, ticker):
    """(spot, contract-row) for the call closest to ATM in the §9 DTE window.

    Read through the options endpoint so the bars backfill and the chain are
    EXACTLY what the portfolio endpoint will regenerate the same day.
    """
    r = await client.get(f"/api/watchlist/{ticker}/options")
    assert r.status_code == 200
    body = r.json()
    spot = body["spot"]
    calls = [c for c in body["chain"] if c["right"] == "C" and 30 <= c["dte"] <= 90]
    assert calls, "stub chain unexpectedly has no 30-90 DTE calls"
    contract = min(calls, key=lambda c: (abs(c["strike"] - spot), c["dte"], c["strike"]))
    return spot, contract


async def insert_rows(rows):
    async with SessionLocal() as s:
        s.add_all(rows)
        await s.commit()


def option_position(contract, quantity, avg_price=None, max_loss=None):
    """A LONG_CALL Position row matching a real chain contract's identity."""
    premium = contract["mid"] if avg_price is None else avg_price
    return Position(
        ticker=OPTION_TICKER,
        instrument="LONG_CALL",
        quantity=quantity,
        avg_price=premium,
        max_loss=(
            quantity * premium * 100 if max_loss is None else max_loss
        ),
        stop_distance=premium,
        opt_expiry=contract["expiry"],
        opt_strike=contract["strike"],
        opt_right="C",
        multiplier=100,
    )


async def risk_body(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    return r.json()


def weekday_dates(n):
    """The last `n` weekdays ending today (UTC), oldest first."""
    d = datetime.now(timezone.utc).date()
    dates = []
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()
    return dates


# ---------------------------------------------------------------------------
# Fresh book: honest zeros / nulls
# ---------------------------------------------------------------------------


async def test_fresh_book_greeks_and_vol_targeting(client):
    body = await risk_body(client)

    g = body["greeks"]
    assert set(g) == GREEKS_KEYS
    assert g["net_delta_shares"] == 0.0
    assert g["delta_adjusted_notional_usd"] == 0.0
    assert g["delta_notional_pct_nav"] == 0.0
    assert g["net_gamma"] == 0.0
    assert g["net_theta_usd_per_day"] == 0.0
    assert g["net_vega_usd"] == 0.0
    assert g["per_position"] == []
    assert g["breaches"] == []
    assert g["limits"] == {
        "max_delta_notional_pct_nav": LIMITS.max_delta_notional_pct_nav,
        "max_net_theta_pct_nav": LIMITS.max_net_theta_pct_nav,
        "max_net_vega_pct_nav": LIMITS.max_net_vega_pct_nav,
    }

    vt = body["vol_targeting"]
    assert set(vt) == VOL_TARGETING_KEYS
    assert vt["forecast_vol"] is None  # honest null: no open positions
    assert vt["multiplier"] == 1.0
    assert vt["target_vol"] == VOL_PARAMS.target_vol
    assert vt["max_multiplier"] == VOL_PARAMS.max_multiplier
    assert isinstance(vt["note"], str) and vt["note"]

    # Static config buckets all carry their kind; no positions -> no dynamic.
    assert body["buckets"]
    assert all(b["kind"] == "STATIC" for b in body["buckets"])


# ---------------------------------------------------------------------------
# Seeded stock + option book: hand-computed §16 sums against the chain
# ---------------------------------------------------------------------------


async def test_seeded_stock_and_option_greeks_hand_computed(client):
    await add_watchlist(client, OPTION_TICKER)
    spot, contract = await chain_call_near_atm(client, OPTION_TICKER)

    mid = contract["mid"]
    await insert_rows(
        [
            StockBarDaily(
                ticker="IBM",
                ts=date(2026, 8, 7),
                open=49.0,
                high=51.0,
                low=48.0,
                close=50.0,
                volume=1_000.0,
            ),
            Position(ticker="IBM", quantity=10, avg_price=48.0, max_loss=300.0),
            option_position(contract, quantity=2),
        ]
    )

    body = await risk_body(client)
    g = body["greeks"]
    delta = contract["delta"]

    # §16 "Equivalent Shares": 10 * 1 * 1.0 + 2 * 100 * delta.
    assert g["net_delta_shares"] == pytest.approx(10 + 2 * 100 * delta, rel=1e-9)
    # Delta-adjusted notional: 10 * 50 + 2 * 100 * delta * spot.
    expected_notional = 10 * 50.0 + 2 * 100 * delta * spot
    assert g["delta_adjusted_notional_usd"] == pytest.approx(
        expected_notional, rel=1e-9
    )
    # NAV: cash + stock market value + option premium BOOK value (§12.1).
    nav = body["nav"]
    assert nav == pytest.approx(INITIAL_CASH + 500.0 + 2 * mid * 100, rel=1e-9)
    assert g["delta_notional_pct_nav"] == pytest.approx(expected_notional / nav)
    # Gamma / theta / vega scale by qty * multiplier off the SAME chain row.
    assert g["net_gamma"] == pytest.approx(2 * 100 * contract["gamma"], rel=1e-9)
    assert g["net_theta_usd_per_day"] == pytest.approx(
        2 * 100 * contract["theta"], rel=1e-9
    )
    assert g["net_vega_usd"] == pytest.approx(2 * 100 * contract["vega"], rel=1e-9)

    rows = {r["ticker"]: r for r in g["per_position"]}
    assert set(rows) == {"IBM", OPTION_TICKER}
    for row in rows.values():
        assert set(row) == PER_POSITION_KEYS

    ibm = rows["IBM"]
    assert ibm["data_ok"] is True and ibm["note"] is None
    assert ibm["instrument"] == "LONG_STOCK"
    assert ibm["equivalent_shares"] == pytest.approx(10.0)
    assert ibm["delta_notional_usd"] == pytest.approx(500.0)
    assert ibm["gamma"] == 0.0
    assert ibm["theta_usd_per_day"] == 0.0
    assert ibm["vega_usd"] == 0.0

    opt = rows[OPTION_TICKER]
    assert opt["data_ok"] is True and opt["note"] is None
    assert opt["instrument"] == "LONG_CALL"
    assert opt["equivalent_shares"] == pytest.approx(2 * 100 * delta, rel=1e-9)
    assert opt["delta_notional_usd"] == pytest.approx(
        2 * 100 * delta * spot, rel=1e-9
    )

    # A small book breaches nothing (§16 limits are far away).
    assert g["breaches"] == []

    # §14 block is sane: forecast exists (GOOGL RV20 is computable over its
    # backfilled bars) and the multiplier is inside the documented clamp.
    vt = body["vol_targeting"]
    assert vt["forecast_vol"] is not None and vt["forecast_vol"] > 0.0
    assert VOL_PARAMS.min_multiplier <= vt["multiplier"] <= VOL_PARAMS.max_multiplier

    # Every bucket row carries its kind; the static config rows come first.
    assert all(b["kind"] in {"STATIC", "DYNAMIC"} for b in body["buckets"])
    static_names = [b["name"] for b in body["buckets"] if b["kind"] == "STATIC"]
    assert static_names == list(LIMITS.correlation_buckets)


# ---------------------------------------------------------------------------
# Honest zeros: contract missing from today's chain -> data_ok false
# ---------------------------------------------------------------------------


async def test_missing_contract_reports_data_issue_and_contributes_zeros(client):
    await add_watchlist(client, OPTION_TICKER)
    spot, contract = await chain_call_near_atm(client, OPTION_TICKER)

    # An expiry far beyond the stub chain horizon: same-day regeneration can
    # never find this contract -> honest zeros, never a guess (§44 rule 18).
    far_expiry = (datetime.now(timezone.utc).date() + timedelta(days=400)).isoformat()
    ghost = option_position(contract, quantity=3, avg_price=1.0, max_loss=300.0)
    ghost.opt_expiry = far_expiry
    await insert_rows([ghost])

    body = await risk_body(client)
    g = body["greeks"]
    assert g["net_delta_shares"] == 0.0
    assert g["delta_adjusted_notional_usd"] == 0.0
    assert g["net_gamma"] == 0.0
    assert g["net_theta_usd_per_day"] == 0.0
    assert g["net_vega_usd"] == 0.0

    (row,) = g["per_position"]
    assert row["data_ok"] is False
    assert "DATA_ISSUE" in row["note"]
    assert row["equivalent_shares"] == 0.0
    assert row["delta_notional_usd"] == 0.0
    assert row["gamma"] == 0.0
    assert row["theta_usd_per_day"] == 0.0
    assert row["vega_usd"] == 0.0


# ---------------------------------------------------------------------------
# §12.4 dynamic buckets from stored bars
# ---------------------------------------------------------------------------


async def test_dynamic_bucket_from_correlated_stored_bars(client):
    """Two tickers with IDENTICAL daily log returns over > 60 stored bars have
    rolling-60d correlation 1.0 > 0.70 -> one DYNAMIC bucket alongside the
    static config rows, measured against the same bucket_risk cap."""
    dates = weekday_dates(70)
    rows = []
    closes = {"CORA": [], "CORB": []}
    prices = {"CORA": 100.0, "CORB": 40.0}
    for t, ts in enumerate(dates):
        r = 0.02 if t % 2 == 0 else -0.01  # same return path for both
        for ticker in ("CORA", "CORB"):
            prices[ticker] *= math.exp(r)
            close = round(prices[ticker], 4)
            closes[ticker].append(close)
            rows.append(
                StockBarDaily(
                    ticker=ticker,
                    ts=ts,
                    open=close,
                    high=round(close * 1.01, 4),
                    low=round(close * 0.99, 4),
                    close=close,
                    volume=1_000.0,
                )
            )
    rows.append(Position(ticker="CORA", quantity=5, avg_price=100.0, max_loss=250.0))
    rows.append(Position(ticker="CORB", quantity=3, avg_price=40.0, max_loss=150.0))
    await insert_rows(rows)

    body = await risk_body(client)
    dynamic = [b for b in body["buckets"] if b["kind"] == "DYNAMIC"]
    assert len(dynamic) == 1
    bucket = dynamic[0]
    assert bucket["tickers"] == ["CORA", "CORB"]
    assert "DYNAMIC" in bucket["name"]

    nav = body["nav"]
    expected_nav = (
        INITIAL_CASH + 5 * closes["CORA"][-1] + 3 * closes["CORB"][-1]
    )
    assert nav == pytest.approx(expected_nav, rel=1e-9)
    assert bucket["risk_usd"] == pytest.approx(400.0)  # 250 + 150
    assert bucket["risk_pct"] == pytest.approx(400.0 / nav)
    assert bucket["cap_pct"] == LIMITS.bucket_risk
    assert bucket["utilization_pct"] == pytest.approx(
        (400.0 / nav) / LIMITS.bucket_risk
    )

    # The static config rows are still all present, marked STATIC.
    static_names = [b["name"] for b in body["buckets"] if b["kind"] == "STATIC"]
    assert static_names == list(LIMITS.correlation_buckets)


# ---------------------------------------------------------------------------
# Huge option book -> §16 breaches with numeric messages
# ---------------------------------------------------------------------------


async def test_huge_option_book_reports_greek_breaches(client):
    """10,000 contracts of a real chain call (booked at a token premium so
    NAV/heat stay sane) put the CURRENT book far beyond every §16 limit —
    the breaches list must say so with the real numbers."""
    await add_watchlist(client, OPTION_TICKER)
    spot, contract = await chain_call_near_atm(client, OPTION_TICKER)

    await insert_rows(
        [option_position(contract, quantity=10_000, avg_price=0.01, max_loss=500.0)]
    )

    body = await risk_body(client)
    g = body["greeks"]
    delta_notional = 10_000 * 100 * contract["delta"] * spot
    assert g["delta_adjusted_notional_usd"] == pytest.approx(
        delta_notional, rel=1e-9
    )
    breaches = g["breaches"]
    assert breaches, "a 1M-equivalent-share book must breach §16 limits"
    assert any("delta" in b.lower() for b in breaches)
    # Numeric messages: every breach line carries dollar amounts and NAV %.
    for b in breaches:
        assert "$" in b and "NAV" in b


# ---------------------------------------------------------------------------
# Gate-chain wiring (§14 + §16 through RISK_APPROVAL)
# ---------------------------------------------------------------------------


async def test_gate_chain_greek_breach_rejects_with_vol_multiplier_detail(client):
    """With the huge-delta book open, a fresh entry attempt must be REJECTed
    by the §16 post-trade greek limits like any other risk rejection (reason
    codes + explanations, §44 rule 20), and the RISK_APPROVAL detail must
    name the §14 budget multiplier (which is != 1 with positions open)."""
    await add_watchlist(client, OPTION_TICKER)
    spot, contract = await chain_call_near_atm(client, OPTION_TICKER)
    await insert_rows(
        [option_position(contract, quantity=10_000, avg_price=0.01, max_loss=500.0)]
    )

    # Authorize a DIFFERENT deterministic-bull ticker so no single-name cap
    # on the huge book's ticker can zero the quantity before the greek check.
    ticker = "GW"
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    # acknowledge_risks: the ticker has no stored history/backtest at promote
    # time, so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200

    r = await client.post("/api/orders/preview", json={"ticker": ticker})
    assert r.status_code == 200, r.text
    body = r.json()

    risk_gate = next(g for g in body["gates"] if g["name"] == "RISK_APPROVAL")
    assert risk_gate["status"] == "FAIL", risk_gate
    assert "PORTFOLIO_DELTA_LIMIT" in risk_gate["detail"]
    # §14 transparency: the multiplier is reported when it scaled the budget.
    assert "budget multiplier" in risk_gate["detail"]
    assert "(vol targeting)" in risk_gate["detail"]

    risk = body["risk"]
    assert risk["decision"] == "REJECT"
    assert "PORTFOLIO_DELTA_LIMIT" in risk["reason_codes"]
    # Explanations surface like any other risk rejection (§36 numbers).
    assert any("delta" in e.lower() for e in risk["explanations"])
    assert body["why_not_trade"]

    # §42: the vetoed entry can never fill.
    r = await client.post("/api/orders/approve", json={"ticker": ticker})
    assert r.status_code == 422
