"""THE CORE GUARANTEE: no market data configured means NO NUMBERS INVENTED.

Massive is the only supported source of real market data. Everything this
platform displays is Massive raw data or computed from it. When no provider is
configured — the state a fresh install starts in — the platform must show
NOTHING rather than fall back to something that looks real.

This file is the regression guard for that promise. It exercises the whole API
through the ``unconfigured_client`` fixture (both providers unset) and asserts:

1. every endpoint whose substance is market data answers 503 with the
   documented machine-readable code;
2. the two endpoints that deliberately do NOT 503 — ``GET /api/positions`` and
   ``GET /api/portfolio/risk``, whose substance is real DB rows — still report
   those rows, with every market-derived field an honest null;
3. ``GET /api/config`` reports both providers unconfigured;
4. the background position monitor SKIPS its sweep (no position changes, one
   WARNING) instead of crash-looping;
5. a PROPERTY TEST walks the JSON body of EVERY unconfigured-mode response and
   fails if any price/quote/greek-shaped field is present and non-null.

That last test is the one that matters most long-term: it does not care which
endpoint or which code path reintroduces a synthetic fallback: if a number
shaped like a price ever appears in unconfigured mode again, it fails.
"""
import logging
from datetime import date

import pytest

from apps.gateway import monitor
from apps.gateway.db import Position, SessionLocal, StockBarDaily, WatchlistItem
from apps.gateway.deps import LLM_NOT_CONFIGURED, MARKET_DATA_NOT_CONFIGURED

TICKER = "IBM"


# ---------------------------------------------------------------------------
# The endpoints that MUST refuse to answer (503), and how to call each.
#
# ``(method, path, json_body, expected_code)`` — parametrized so adding a new
# market-facing endpoint without its guard shows up as a missing row here.
# ---------------------------------------------------------------------------
MARKET_DATA_ENDPOINTS = [
    ("GET", "/api/market/overview", None),
    # The watchlist DASHBOARD (every column is a price or computed from one).
    # GET /api/watchlist itself is the curated ticker LIST — real DB rows with
    # no market data in them — and stays 200; see ALL_PROBED_ENDPOINTS.
    ("GET", "/api/watchlist/overview", None),
    ("GET", f"/api/watchlist/{TICKER}/analysis", None),
    ("GET", f"/api/watchlist/{TICKER}/bars", None),
    ("GET", f"/api/watchlist/{TICKER}/options", None),
    ("POST", "/api/orders/preview", {"ticker": TICKER, "quantity": 10}),
    ("POST", "/api/orders/approve", {"ticker": TICKER, "quantity": 10}),
    ("POST", "/api/orders/close", {"ticker": TICKER}),
    ("POST", "/api/backtests", {"ticker": TICKER, "params": {}}),
    ("POST", "/api/positions/check-exits", None),
]

LLM_ENDPOINTS = [
    ("POST", "/api/recommendations/refresh", None),
]


async def _call(client, method: str, path: str, body):
    if method == "GET":
        return await client.get(path)
    return await client.post(path, json=body)


async def _seed_watchlisted_position(ticker: str = TICKER) -> None:
    """One watchlisted ticker with a real OPEN position and a stored bar.

    Deliberately gives every guarded endpoint the best possible chance of
    producing a number: the ticker is on the watchlist (so it passes the §4.2
    gate), it has an OPEN position (so /close and the exit sweep have work to
    do) and it even has a stored daily bar (so a lazy-backfill skip could tempt
    a code path into pricing off it). The 503s below therefore prove the guard
    fires on POLICY, not merely because the database happened to be empty.
    """
    async with SessionLocal() as s:
        s.add(WatchlistItem(ticker=ticker, added_by="test-user"))
        s.add(
            StockBarDaily(
                ticker=ticker,
                ts=date(2026, 8, 7),
                open=49.0,
                high=51.0,
                low=48.0,
                close=50.0,
                volume=1_000.0,
            )
        )
        s.add(
            Position(
                ticker=ticker,
                quantity=10,
                avg_price=48.0,
                max_loss=300.0,
                stop_distance=2.0,
                entry_edge=0.5,
                entry_bar_date="2026-08-07",
            )
        )
        await s.commit()


# ---------------------------------------------------------------------------
# 1. Market-data endpoints: 503 with the documented code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path,body", MARKET_DATA_ENDPOINTS)
async def test_market_endpoints_503_when_unconfigured(
    unconfigured_client, method, path, body
):
    await _seed_watchlisted_position()
    r = await _call(unconfigured_client, method, path, body)

    assert r.status_code == 503, (
        f"{method} {path} returned {r.status_code}, not 503 — an endpoint "
        "serving market data must refuse when no provider is configured"
    )
    detail = r.json()["detail"]
    assert detail["code"] == MARKET_DATA_NOT_CONFIGURED
    # The message must name the missing configuration, not just say "error".
    assert "MARKET_DATA_PROVIDER" in detail["message"]


@pytest.mark.parametrize("method,path,body", LLM_ENDPOINTS)
async def test_llm_endpoints_503_when_unconfigured(
    unconfigured_client, method, path, body
):
    r = await _call(unconfigured_client, method, path, body)

    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == LLM_NOT_CONFIGURED
    assert "LLM_PROVIDER" in detail["message"]


async def test_recommendations_refresh_creates_nothing_when_unconfigured(
    unconfigured_client,
):
    """The 503 is a real refusal, not a cosmetic status on a completed write."""
    r = await unconfigured_client.post("/api/recommendations/refresh")
    assert r.status_code == 503

    listing = await unconfigured_client.get("/api/recommendations?status=ALL")
    assert listing.status_code == 200
    assert listing.json() == []


async def test_market_endpoints_do_not_leak_numbers_in_the_error(
    unconfigured_client,
):
    """A 503 body carries an explanation — never a "best effort" number."""
    await _seed_watchlisted_position()
    for method, path, body in MARKET_DATA_ENDPOINTS:
        r = await _call(unconfigured_client, method, path, body)
        assert_no_market_numbers(r.json(), f"{method} {path}")


# ---------------------------------------------------------------------------
# 2. The endpoints that must NOT 503 — real DB rows, null market fields
# ---------------------------------------------------------------------------

async def test_positions_listed_with_null_price_fields(unconfigured_client):
    """A position is a REAL row: it stays visible, its market fields go null.

    Hiding a position the user actually holds because no quote feed is
    configured would be its own kind of dishonesty — the DB facts (ticker,
    quantity, entry price, open date) are known and are shown.
    """
    await _seed_watchlisted_position()

    r = await unconfigured_client.get("/api/positions")
    assert r.status_code == 200
    (row,) = r.json()

    # Real, DB-owned facts: still reported.
    assert row["ticker"] == TICKER
    assert row["quantity"] == 10
    assert row["avg_price"] == 48.0
    assert row["status"] == "OPEN"

    # Market-derived: every one an honest null.
    for field in (
        "current_price",
        "market_value",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "stop_price",
        "trail_price",
        "current_edge",
        "signal_decay",
        "exit_status",
    ):
        assert row[field] is None, f"positions.{field} must be null, got {row[field]!r}"

    # …and the user is told WHY rather than left to guess (§37, §44 rule 18).
    assert row["exit_reasons"]
    assert any("no market data provider" in reason for reason in row["exit_reasons"])


async def test_portfolio_risk_degrades_instead_of_503(unconfigured_client):
    """NAV/cash come from the DB, so this view answers 200 — with nulls."""
    await _seed_watchlisted_position()

    r = await unconfigured_client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()

    # The situation is stated outright, not left to be inferred from nulls.
    assert body["market_data"]["configured"] is False
    assert "MARKET_DATA_PROVIDER" in body["market_data"]["message"]

    # NAV == cash EXACTLY: the open position contributes nothing rather than a
    # synthetic mark. This is the numeric heart of the guarantee here.
    assert body["nav"] == body["cash"]
    assert body["cash_pct"] == 1.0

    # Regime and its dependent cash floor are computed from SPY bars -> null.
    assert body["market_regime"] is None
    assert body["cash_floor_pct"] is None

    (position,) = body["positions"]
    assert position["ticker"] == TICKER
    assert position["quantity"] == 10  # real DB fact
    assert position["market_price"] is None
    assert position["market_value"] is None
    assert position["note"] == "DATA_ISSUE"

    greeks = body["greeks"]
    for field in (
        "net_delta_shares",
        "delta_adjusted_notional_usd",
        "delta_notional_pct_nav",
        "net_gamma",
        "net_theta_usd_per_day",
        "net_vega_usd",
    ):
        assert greeks[field] is None, f"greeks.{field} must be null when unconfigured"
    assert greeks["breaches"] == []  # no numbers -> no breach can be claimed
    (greeks_row,) = greeks["per_position"]
    assert greeks_row["data_ok"] is False
    for field in (
        "equivalent_shares",
        "delta_notional_usd",
        "gamma",
        "theta_usd_per_day",
        "vega_usd",
    ):
        assert greeks_row[field] is None

    # §14 vol targeting: no forecast, neutral multiplier, and it says so.
    vol = body["vol_targeting"]
    assert vol["forecast_vol"] is None
    assert vol["multiplier"] == 1.0
    assert "no market data provider is configured" in vol["note"]


# ---------------------------------------------------------------------------
# 3. Configuration visibility
# ---------------------------------------------------------------------------

async def test_config_reports_both_providers_unconfigured(unconfigured_client):
    r = await unconfigured_client.get("/api/config")
    assert r.status_code == 200
    providers = r.json()["providers"]

    assert providers["market_data_configured"] is False
    assert providers["llm_configured"] is False
    # Names report "" when unset — never a cosmetic default that would let a
    # UI claim a data source exists.
    assert providers["market_data"] == ""
    assert providers["llm"] == ""


async def test_config_never_claims_a_provider_it_does_not_have(unconfigured_client):
    """Guards the specific bug this change fixes: "stub" must not appear."""
    providers = (await unconfigured_client.get("/api/config")).json()["providers"]
    assert providers["market_data"] != "stub"
    assert providers["llm"] != "stub"


# ---------------------------------------------------------------------------
# 4. The background monitor must skip, not sweep and not crash-loop
# ---------------------------------------------------------------------------

async def test_monitor_sweep_skips_and_warns_when_unconfigured(
    unconfigured_client, caplog
):
    """The sweep sells real positions — it must never run on invented prices."""
    await _seed_watchlisted_position()

    before = (await unconfigured_client.get("/api/positions?status=ALL")).json()
    sweeps_before = monitor.STATE.sweeps_total

    with caplog.at_level(logging.WARNING, logger="position_monitor"):
        result = await monitor.run_sweep_and_update()

    # Skipped, and honest about it.
    assert result["skipped"] == "MARKET_DATA_NOT_CONFIGURED"
    assert result["checked"] == 0
    assert result["exits_triggered"] == []

    # Exactly one WARNING per attempt (no crash loop, no traceback spam).
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and rec.name == "position_monitor"
    ]
    assert len(warnings) == 1
    assert warnings[0].message == "position_monitor_skipped_no_market_data"

    # NOTHING changed: no position was closed, no sweep was counted.
    after = (await unconfigured_client.get("/api/positions?status=ALL")).json()
    assert [p["status"] for p in after] == [p["status"] for p in before]
    assert all(p["status"] == "OPEN" for p in after)
    assert monitor.STATE.sweeps_total == sweeps_before


async def test_monitor_repeated_attempts_stay_quiet_and_harmless(
    unconfigured_client, caplog
):
    """Repeated ticks keep skipping — one line each, never an exception."""
    await _seed_watchlisted_position()

    with caplog.at_level(logging.WARNING, logger="position_monitor"):
        for _ in range(3):
            result = await monitor.run_sweep_and_update()
            assert result["skipped"] == "MARKET_DATA_NOT_CONFIGURED"

    warnings = [
        rec for rec in caplog.records if rec.name == "position_monitor"
    ]
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# 5. PROPERTY TEST — the long-term regression guard
# ---------------------------------------------------------------------------

# Field names that can only ever hold a price, a quote or a greek. If any of
# these appears with a non-null NUMBER while no market data is configured,
# something synthesized it. Matched on the exact key name (a nested key like
# ``limits.max_delta_notional_pct_nav`` is configuration, not a quote, and is
# deliberately absent from this list).
FORBIDDEN_NUMERIC_KEYS = frozenset(
    {
        # Quotes and prices
        "price",
        "current_price",
        "market_price",
        "market_value",
        "bid",
        "ask",
        "mid",
        "current_mid",
        "last",
        "spot",
        "close",
        "open",
        "high",
        "low",
        "change_pct",
        # Volatility
        "iv",
        "atm_iv",
        "iv_rank",
        "iv_rv_spread",
        "rv20",
        "realized_vol20",
        "expected_move_pct",
        "forecast_vol",
        "expected_move",
        # Greeks
        "delta",
        "gamma",
        "theta",
        "vega",
        "net_gamma",
        "net_delta_shares",
        "net_vega_usd",
        "net_theta_usd_per_day",
        "delta_notional_usd",
        "delta_adjusted_notional_usd",
        "delta_notional_pct_nav",
        "equivalent_shares",
        "theta_usd_per_day",
        "vega_usd",
        # Derived marks
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "stop_price",
        "trail_price",
        "premium_pnl_pct",
        "current_edge",
        "signal_decay",
        # Regime / classification derived from price history
        "market_regime",
        "cash_floor_pct",
    }
)


def _walk(node, path=""):
    """Yield ``(json_path, key, value)`` for every key in a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            yield here, key, value
            yield from _walk(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            here = f"{path}[{index}]"
            yield from _walk(value, here)


def assert_no_market_numbers(body, where: str) -> None:
    """Fail if any price/quote/greek-shaped key holds a non-null number.

    Booleans are excluded (``bool`` is an ``int`` in Python, and a flag named
    e.g. ``open`` is not a price). Strings are excluded too: a textual
    ``market_regime`` would be caught by the null assertions in the dedicated
    tests, while this walk is specifically hunting NUMBERS that nobody could
    have known.
    """
    for json_path, key, value in _walk(body):
        if key not in FORBIDDEN_NUMERIC_KEYS:
            continue
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            raise AssertionError(
                f"{where}: {json_path} = {value!r} — a market-data number was "
                "produced while no market data provider is configured. "
                "Something is synthesizing data; there must be no fallback."
            )


# Every endpoint reachable in unconfigured mode, including the ones that 503
# and the ones that answer 200 with degraded content.
ALL_PROBED_ENDPOINTS = MARKET_DATA_ENDPOINTS + LLM_ENDPOINTS + [
    ("GET", "/api/positions", None),
    ("GET", "/api/positions?status=ALL", None),
    ("GET", "/api/portfolio/risk", None),
    ("GET", "/api/config", None),
    ("GET", "/api/watchlist", None),
    ("GET", "/api/recommendations?status=ALL", None),
    ("GET", "/api/health/strategy", None),
    ("GET", "/api/positions/monitor", None),
    ("GET", "/api/alerts", None),
    # The broker surfaces: both must answer in the unconfigured state (they
    # are how a user finds out WHY execution refused) without inventing an
    # account, a position or a price to fill the silence.
    ("GET", "/api/broker/status", None),
    ("GET", "/api/broker/reconcile", None),
]


@pytest.mark.parametrize("method,path,body", ALL_PROBED_ENDPOINTS)
async def test_no_response_contains_a_market_number(
    unconfigured_client, method, path, body
):
    """THE regression guard: walk every response, allow no invented number.

    This is deliberately blunt and endpoint-agnostic. Any future change that
    reintroduces a synthetic fallback — a default provider, a "reasonable"
    zero, a cached last-known price — surfaces here without anyone having to
    remember to write a test for it.
    """
    await _seed_watchlisted_position()
    r = await _call(unconfigured_client, method, path, body)
    assert_no_market_numbers(r.json(), f"{method} {path}")


async def test_property_walker_actually_catches_a_planted_number():
    """The guard above is only worth its runtime if it can fail. Prove it."""
    with pytest.raises(AssertionError, match="market-data number"):
        assert_no_market_numbers(
            {"positions": [{"ticker": "IBM", "market_price": 123.45}]}, "planted"
        )

    # …and that it tolerates the honest shapes it is supposed to allow.
    assert_no_market_numbers(
        {"positions": [{"ticker": "IBM", "market_price": None}], "quantity": 10},
        "honest",
    )
