"""Polymarket adapter contract (Catalyst research upgrade, LOOP 4).

What these tests pin, per the program brief's mandated cases: discovery,
market snapshot, price history, missing optional fields (None never 0),
malformed price, inactive/resolved market, rate limit (one capped retry),
network error (one transient retry then the taxonomy), partial failure
(dead book degrades fields, never the snapshot), and the contact
User-Agent requirement. All HTTP via httpx.MockTransport routed by host —
no real network call anywhere.
"""
import json
from datetime import datetime, timezone

import httpx
import pytest

import libs.prediction_markets.polymarket as pm_module
from libs.prediction_markets import PredictionMarketError
from libs.prediction_markets.polymarket import PolymarketProvider

UA = "test-suite/0.1 (contact@example.com)"
# A window INSIDE the venue's bounded-form span cap, so these tests exercise
# the startTs/endTs shape. The long-history shape (interval=max) has its own
# tests below.
START = datetime(2026, 8, 10, tzinfo=timezone.utc)
END = datetime(2026, 8, 20, tzinfo=timezone.utc)


def gamma_market(**overrides) -> dict:
    base = {
        "id": 501,
        "slug": "fed-cut-september",
        "question": "Will the Fed cut rates in September?",
        "description": "Resolves YES if the FOMC lowers the target range.",
        "endDate": "2026-09-17T18:00:00Z",
        "active": True,
        "closed": False,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.63", "0.37"]),
        "clobTokenIds": json.dumps(["tok-yes-501", "tok-no-501"]),
        "volumeNum": 1_250_000.5,
        "liquidityNum": 84_000.0,
        "events": [{"id": 9001, "title": "September FOMC"}],
    }
    base.update(overrides)
    return base


def make_provider(handler, **kwargs) -> PolymarketProvider:
    kwargs.setdefault("min_request_interval_seconds", 0.0)
    return PolymarketProvider(
        UA, transport=httpx.MockTransport(handler), **kwargs
    )


def test_blank_user_agent_refused_at_construction():
    with pytest.raises(PredictionMarketError) as exc:
        PolymarketProvider("")
    assert "User-Agent" in str(exc.value)


def test_search_parses_discovery_filters_inactive_and_dedups():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/public-search"
        assert request.headers["User-Agent"] == UA
        return httpx.Response(200, json={"events": [
            {"markets": [
                gamma_market(),
                gamma_market(id=502, active=False, closed=True),  # filtered
                gamma_market(),  # duplicate id 501 — folded
                "not-a-dict",    # malformed — costs itself
                gamma_market(id=503, question="Will CPI print above 3%?"),
            ]},
        ]})

    markets = make_provider(handler).search_markets("fed rate cut", limit=10)
    assert [m.market_id for m in markets] == ["501", "503"]
    first = markets[0]
    assert first.provider == "polymarket"
    assert first.question == "Will the Fed cut rates in September?"
    assert first.provider_event_id == "9001"
    assert first.url == "https://polymarket.com/market/fed-cut-september"
    assert [o.price for o in first.outcomes] == [0.63, 0.37]
    assert first.volume == pytest.approx(1_250_000.5)


def test_search_limit_zero_never_touches_network():
    def must_not_call(request: httpx.Request) -> httpx.Response:
        raise AssertionError("limit=0 must not fire an HTTP request")

    assert make_provider(must_not_call).search_markets("q", limit=0) == []


def test_get_market_parses_statuses_and_tolerates_malformed_numbers():
    def handler(request: httpx.Request) -> httpx.Response:
        market_id = request.url.path.rsplit("/", 1)[-1]
        if market_id == "501":
            return httpx.Response(200, json=gamma_market(
                outcomePrices=json.dumps(["not-a-price", "0.37"]),
                volumeNum="garbage", liquidityNum=None,
            ))
        if market_id == "600":
            return httpx.Response(200, json=gamma_market(
                id=600, active=False, closed=True,
                umaResolutionStatus="resolved",
            ))
        return httpx.Response(404, json={"error": "not found"})

    provider = make_provider(handler)
    market = provider.get_market("501")
    # Malformed price / absent depth -> None, NEVER 0 (§44 rule 18).
    assert market.outcomes[0].price is None
    assert market.outcomes[1].price == 0.37
    assert market.volume is None
    assert market.liquidity is None
    assert market.end_date == datetime(2026, 9, 17, 18, tzinfo=timezone.utc)
    resolved = provider.get_market("600")
    assert resolved.status == "RESOLVED"
    with pytest.raises(PredictionMarketError):
        provider.get_market("999")


def test_snapshot_combines_gamma_prices_with_clob_book_and_last_trade():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        if request.url.path == "/book":
            assert request.url.params["token_id"] == "tok-yes-501"
            return httpx.Response(200, json={
                "bids": [{"price": "0.61", "size": "100"},
                         {"price": "0.62", "size": "40"}],
                "asks": [{"price": "0.66", "size": "10"},
                         {"price": "0.64", "size": "55"}],
            })
        if request.url.path == "/last-trade-price":
            return httpx.Response(200, json={"price": "0.63"})
        return httpx.Response(404)

    snap = make_provider(handler).get_market_snapshot("501")
    assert snap.best_bid == 0.62  # best = highest bid
    assert snap.best_ask == 0.64  # best = lowest ask
    assert snap.midpoint == pytest.approx(0.63)
    assert snap.spread == pytest.approx(0.02)
    assert snap.last_trade_price == 0.63
    assert snap.outcome_prices == {"Yes": 0.63, "No": 0.37}
    assert snap.open_interest is None  # never stated by the public APIs
    assert snap.observed_at.tzinfo is not None


def test_snapshot_survives_null_book_sides():
    """A served '"bids": null' has the key PRESENT — it must degrade the
    bid/ask fields exactly like an absent book, never sink the snapshot."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        if request.url.path == "/book":
            return httpx.Response(200, json={"bids": None, "asks": None})
        if request.url.path == "/last-trade-price":
            return httpx.Response(200, json={"price": "0.63"})
        return httpx.Response(404)

    snap = make_provider(handler).get_market_snapshot("501")
    assert snap.best_bid is None and snap.best_ask is None
    assert snap.last_trade_price == 0.63


def test_volume_fallback_is_by_parse_failure_never_truthiness():
    """A provider-STATED 0 is a real number (absent != 0), and garbage in
    the primary field must not block a valid secondary one."""

    def handler(request: httpx.Request) -> httpx.Response:
        market_id = request.url.path.rsplit("/", 1)[-1]
        if market_id == "701":  # stated zero stays zero
            return httpx.Response(200, json=gamma_market(
                id=701, volumeNum=0, liquidityNum=0.0,
            ))
        if market_id == "702":  # garbage primary falls to valid secondary
            return httpx.Response(200, json=gamma_market(
                id=702, volumeNum="garbage", volume="123.45",
            ))
        return httpx.Response(404)

    provider = make_provider(handler)
    stated_zero = provider.get_market("701")
    assert stated_zero.volume == 0.0
    assert stated_zero.liquidity == 0.0
    fallback = provider.get_market("702")
    assert fallback.volume == pytest.approx(123.45)


def test_traversal_shaped_market_id_is_refused_without_network():
    def must_not_call(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a bad id must never reach the URL path")

    provider = make_provider(must_not_call)
    for bad in ("../events", "501/../../x", "", "a b", "x?y=1"):
        with pytest.raises(PredictionMarketError):
            provider.get_market(bad)


def test_non_object_search_body_is_a_fault_not_an_empty_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["a", "list"])

    with pytest.raises(PredictionMarketError):
        make_provider(handler).search_markets("q", limit=5)


def test_snapshot_survives_a_dead_order_book(monkeypatch):
    """Partial failure handling (plan Phase 11): the book failing degrades
    bid/ask/mid/spread to None — the snapshot itself survives on Gamma."""
    monkeypatch.setattr(pm_module.time_module, "sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        if request.url.path == "/book":
            return httpx.Response(500, json={"error": "boom"})
        if request.url.path == "/last-trade-price":
            return httpx.Response(200, json={"price": "0.63"})
        return httpx.Response(404)

    snap = make_provider(handler).get_market_snapshot("501")
    assert snap.best_bid is None and snap.best_ask is None
    assert snap.midpoint is None and snap.spread is None
    assert snap.last_trade_price == 0.63
    assert snap.outcome_prices == {"Yes": 0.63, "No": 0.37}


def test_price_history_parses_skips_malformed_and_sorts():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        if request.url.path == "/prices-history":
            assert request.url.params["market"] == "tok-no-501"
            assert request.url.params["fidelity"] == "60"
            # Raw body: a real server can emit literal NaN, which strict
            # client-side serializers refuse — the parser must not.
            body = (
                '{"history": ['
                '{"t": 1755600000, "p": 0.40},'
                '{"t": 1755500000, "p": "0.38"},'
                '{"t": 1755550000, "p": "not-a-price"},'   # skipped
                '{"p": 0.39},'                              # no ts: skipped
                '"not-a-dict",'                             # skipped
                '{"t": true, "p": 0.41},'      # bool-as-int: no 1970 point
                '{"t": NaN, "p": 0.42},'       # NaN epoch: skipped, no crash
                '{"t": 1e18, "p": 0.43}'       # out-of-range epoch: skipped
                ']}'
            )
            return httpx.Response(
                200, content=body.encode(),
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404)

    points = make_provider(handler).get_price_history(
        "501", outcome="no", start=START, end=END
    )
    assert [p.price for p in points] == [0.38, 0.40]
    assert points[0].ts < points[1].ts
    assert all(p.ts.tzinfo is not None for p in points)


def test_price_history_naive_bounds_are_utc_never_host_local():
    """Naive datetimes are UTC by platform convention — .timestamp() on a
    naive bound would use the HOST timezone and fetch a machine-dependent
    window."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        seen.setdefault("startTs", request.url.params["startTs"])
        seen.setdefault("endTs", request.url.params["endTs"])
        return httpx.Response(200, json={"history": []})

    provider = make_provider(handler)
    provider.get_price_history("501", outcome="Yes", start=START, end=END)
    aware = dict(seen)
    seen.clear()
    provider.get_price_history(
        "501", outcome="Yes",
        start=START.replace(tzinfo=None), end=END.replace(tzinfo=None),
    )
    assert seen == aware  # identical window on ANY host timezone


def test_price_history_unknown_outcome_and_degenerate_window_are_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        raise AssertionError("CLOB must not be called for an unknown outcome")

    provider = make_provider(handler)
    assert provider.get_price_history(
        "501", outcome="Maybe", start=START, end=END
    ) == []
    assert provider.get_price_history(
        "501", outcome="Yes", start=END, end=START
    ) == []


def test_429_retries_once_with_capped_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(pm_module.time_module, "sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "86400"})
        return httpx.Response(200, json=gamma_market())

    market = make_provider(handler).get_market("501")
    assert market.market_id == "501"
    assert calls["n"] == 2
    assert sleeps and all(
        s <= pm_module.MAX_RETRY_AFTER_SECONDS for s in sleeps
    )

    def always_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with pytest.raises(PredictionMarketError) as exc:
        make_provider(always_limited).get_market("501")
    assert "429" in str(exc.value)


def test_transient_fault_retried_once_then_raises(monkeypatch):
    monkeypatch.setattr(pm_module.time_module, "sleep", lambda s: None)
    monkeypatch.setattr(pm_module.random, "random", lambda: 0.0)
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(200, json=gamma_market())

    assert make_provider(flaky).get_market("501").market_id == "501"
    assert calls["n"] == 2

    def always_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    with pytest.raises(PredictionMarketError) as exc:
        make_provider(always_down).get_market("501")
    assert "503" in str(exc.value) or "persisted" in str(exc.value)


def test_capabilities_tri_state_never_raises(monkeypatch):
    monkeypatch.setattr(pm_module.time_module, "sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com":
            return httpx.Response(200, json={"events": []})
        return httpx.Response(503, json={"error": "down"})

    caps = make_provider(handler).capabilities()
    assert caps["market_search"] is True
    assert caps["market_metadata"] is True
    assert isinstance(caps["market_snapshot"], str)  # fault: unknown
    assert caps["price_history"] == caps["market_snapshot"]


def test_registry_serves_polymarket_read_only():
    from libs.prediction_markets import get_provider

    provider = get_provider("polymarket")
    assert provider.name == "polymarket"
    # READ ONLY: the whole public surface is the four Protocol methods.
    for banned in ("place_order", "submit_order", "sign_order", "wallet"):
        assert not hasattr(provider, banned)
    provider.close()


# ---------------------------------------------------------------------------
# LONG HISTORY (probed live 2026-08-23)
#
# The venue offers two shapes of /prices-history and they reach different
# data. `startTs`/`endTs` is SPAN-CAPPED at ~14 days and 400s past it; the
# `interval` form is uncapped and, at daily fidelity, returns a contract's
# whole life (265 points back to first trade on the GDP markets).
#
# Without this the platform could only ever see a fortnight, so a contract
# that repriced from 20c to 60c last quarter looked like it had always been
# at 60c — and "when did the market change its mind, and why" is
# unanswerable, which is the question the history is FOR.
# ---------------------------------------------------------------------------


def test_a_long_window_switches_to_the_uncapped_interval_form():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"history": []})

    provider = make_provider(handler)
    provider.get_price_history(
        "501",
        outcome="Yes",
        start=datetime(2025, 11, 1, tzinfo=timezone.utc),
        end=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )
    # The bounded params would 400 on a span this wide.
    assert "startTs" not in seen and "endTs" not in seen
    assert seen["interval"] == "max"
    # Only daily fidelity returns the full life; hourly silently truncates to
    # the most recent month even under `interval=max`.
    assert seen["fidelity"] == "1440"


def test_a_short_window_keeps_hourly_detail():
    """The long form costs resolution, so it is used ONLY when needed: a
    recent window must still come back hourly."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/markets/"):
            return httpx.Response(200, json=gamma_market())
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"history": []})

    provider = make_provider(handler)
    provider.get_price_history("501", outcome="Yes", start=START, end=END)
    assert seen["fidelity"] == "60"
    assert "startTs" in seen and "interval" not in seen
