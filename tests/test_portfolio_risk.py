"""Tests for GET /api/portfolio/risk (plan §12.5, §13, §35)."""
from datetime import date, datetime

import pytest

from apps.gateway.db import Position, SessionLocal, StockBarDaily
from libs.trading_core.models import MarketRegime
from libs.trading_core.risk import RiskLimits

LIMITS = RiskLimits()
INITIAL_CASH = 100_000.0

CONTRACT_KEYS = {
    "as_of",
    "nav",
    "cash",
    "cash_pct",
    "market_regime",
    "cash_floor_pct",
    "trading_enabled",
    "portfolio_heat_pct",
    "heat_state",
    "max_new_risk_usd",
    "max_new_risk_pct",
    "positions",
    "buckets",
    "limits",
}


async def test_fresh_portfolio_risk(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()

    assert set(body) == CONTRACT_KEYS
    datetime.fromisoformat(body["as_of"])

    # Fresh paper account: NAV == cash == paper_initial_cash, zero heat.
    assert body["nav"] == INITIAL_CASH
    assert body["cash"] == INITIAL_CASH
    assert body["cash_pct"] == 1.0
    assert body["portfolio_heat_pct"] == 0.0
    assert body["heat_state"] == "NORMAL"
    assert body["trading_enabled"] is False  # kill switch default (§18)
    assert body["positions"] == []

    # Regime + matching cash floor (§13) are present and consistent.
    regime = body["market_regime"]
    assert regime in {m.value for m in MarketRegime}
    assert body["cash_floor_pct"] == LIMITS.cash_floors[MarketRegime(regime)]

    # Full heat headroom available: heat_reject * NAV.
    assert body["max_new_risk_usd"] == pytest.approx(LIMITS.heat_reject * INITIAL_CASH)
    assert body["max_new_risk_pct"] == pytest.approx(LIMITS.heat_reject)

    # Buckets present with zero utilization (§12.4).
    buckets = body["buckets"]
    assert [b["name"] for b in buckets] == list(LIMITS.correlation_buckets)
    tech = buckets[0]
    assert tech["name"] == "TECH_MEGA"
    assert tech["tickers"] == list(LIMITS.correlation_buckets["TECH_MEGA"])
    assert tech["risk_usd"] == 0.0
    assert tech["risk_pct"] == 0.0
    assert tech["utilization_pct"] == 0.0
    assert tech["cap_pct"] == LIMITS.bucket_risk

    # Limits echo the RiskLimits defaults (§12).
    assert body["limits"] == {
        "single_name_risk_pct": LIMITS.single_name_risk,
        "single_name_capital_pct": LIMITS.single_name_capital,
        "bucket_risk_pct": LIMITS.bucket_risk,
        "heat_elevated_pct": LIMITS.heat_elevated,
        "heat_high_pct": LIMITS.heat_high,
        "heat_reject_pct": LIMITS.heat_reject,
        "abs_max_trade_risk_pct": LIMITS.abs_max_trade_risk,
    }


async def test_portfolio_risk_view_is_read_only_no_risk_decision_audit(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    r = await client.get("/api/audit")
    actions = {e["action"] for e in r.json()}
    # The SPY lazy backfill may audit DATA_BACKFILL; the read itself decides
    # nothing, so no RISK_DECISION event may exist.
    assert "RISK_DECISION" not in actions


async def test_portfolio_risk_with_seeded_open_positions(client):
    """Hand-computed NAV / heat / bucket numbers for seeded OPEN positions.

    IBM has one stored bar (close 50.0) -> priced honestly; NVDA has no
    stored bars -> market_price/market_value null with a DATA_ISSUE note and
    0 contribution to NAV (plan §44 rule 18).
    """
    async with SessionLocal() as s:
        s.add(
            StockBarDaily(
                ticker="IBM",
                ts=date(2026, 8, 7),
                open=49.0,
                high=51.0,
                low=48.0,
                close=50.0,
                volume=1_000.0,
            )
        )
        s.add(Position(ticker="IBM", quantity=10, avg_price=48.0, max_loss=300.0))
        s.add(Position(ticker="NVDA", quantity=20, avg_price=100.0, max_loss=500.0))
        await s.commit()

    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()

    nav = INITIAL_CASH + 10 * 50.0  # cash + IBM market value; NVDA counts 0
    heat_usd = 300.0 + 500.0
    assert body["nav"] == pytest.approx(nav)
    assert body["cash"] == INITIAL_CASH
    assert body["cash_pct"] == pytest.approx(INITIAL_CASH / nav)
    assert body["portfolio_heat_pct"] == pytest.approx(heat_usd / nav)
    assert body["heat_state"] == "NORMAL"  # 800/100500 ~ 0.80% < 4%
    assert body["max_new_risk_usd"] == pytest.approx(
        LIMITS.heat_reject * nav - heat_usd
    )
    assert body["max_new_risk_pct"] == pytest.approx(
        (LIMITS.heat_reject * nav - heat_usd) / nav
    )

    positions = {p["ticker"]: p for p in body["positions"]}
    assert set(positions) == {"IBM", "NVDA"}

    ibm = positions["IBM"]
    assert ibm["quantity"] == 10
    assert ibm["avg_price"] == 48.0
    assert ibm["market_price"] == 50.0
    assert ibm["market_value"] == pytest.approx(500.0)
    assert ibm["max_loss"] == 300.0
    assert ibm["note"] is None
    datetime.fromisoformat(ibm["opened_at"])

    nvda = positions["NVDA"]
    assert nvda["market_price"] is None  # honest null: no stored bars
    assert nvda["market_value"] is None
    assert nvda["max_loss"] == 500.0
    assert nvda["note"] == "DATA_ISSUE"

    # NVDA is in TECH_MEGA; IBM is not (§12.4).
    tech = body["buckets"][0]
    assert tech["name"] == "TECH_MEGA"
    assert tech["risk_usd"] == pytest.approx(500.0)
    assert tech["risk_pct"] == pytest.approx(500.0 / nav)
    assert tech["cap_pct"] == LIMITS.bucket_risk
    assert tech["utilization_pct"] == pytest.approx((500.0 / nav) / LIMITS.bucket_risk)
