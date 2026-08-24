"""Prediction-market registry + stub provider contract (Catalyst research
upgrade).

The registry must clone the platform's provider contract exactly, and the
stub must be DETERMINISTIC while exercising the honesty paths the
interpretation layer depends on: a thin market with unreported depth
(volume/liquidity None — never zero), an outcome with no history (empty
list, never an invented series), and an unknown market id that raises
rather than fabricating a shell.
"""
from datetime import datetime, timedelta, timezone

import pytest

from libs.prediction_markets import (
    CAPABILITY_KEYS,
    MARKET_STATUS_ACTIVE,
    PREDICTION_MARKETS_NOT_CONFIGURED_MESSAGE,
    PredictionMarketError,
    ProviderNotConfigured,
    get_provider,
)

START = datetime(2026, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_empty_name_raises_provider_not_configured():
    with pytest.raises(ProviderNotConfigured) as exc:
        get_provider("")
    assert str(exc.value) == PREDICTION_MARKETS_NOT_CONFIGURED_MESSAGE


def test_unknown_name_raises_value_error_naming_known_providers():
    with pytest.raises(ValueError) as exc:
        get_provider("kalshi")
    assert "kalshi" in str(exc.value)
    assert "stub" in str(exc.value)


def test_search_markets_is_deterministic_and_bounded():
    a = get_provider("stub").search_markets("september fed rate cut", limit=10)
    b = get_provider("stub").search_markets("september fed rate cut", limit=10)
    assert a == b
    assert len(a) == 2
    assert get_provider("stub").search_markets("q", limit=0) == []
    for m in a:
        assert m.provider == "stub"
        assert m.question.startswith("[SYNTHETIC]")
        assert m.status == MARKET_STATUS_ACTIVE
        prices = [o.price for o in m.outcomes]
        assert all(p is not None and 0.0 < p < 1.0 for p in prices)
        assert abs(sum(prices) - 1.0) < 1e-6


def test_unknown_market_id_raises_never_fabricates():
    with pytest.raises(PredictionMarketError):
        get_provider("stub").get_market("condition-0xdeadbeef")


def test_thin_market_reports_absent_depth_as_none_never_zero():
    markets = get_provider("stub").search_markets("GDP above 2.5", limit=10)
    thin = markets[1]
    assert thin.market_id.endswith("-1")
    assert thin.volume is None
    assert thin.liquidity is None
    assert thin.resolution_criteria is None
    deep = markets[0]
    assert deep.volume is not None
    assert deep.resolution_criteria is not None


def test_snapshot_arithmetic_is_internally_consistent():
    stub = get_provider("stub")
    market = stub.search_markets("CPI hot print", limit=1)[0]
    snap = stub.get_market_snapshot(market.market_id)
    assert snap.best_bid is not None and snap.best_ask is not None
    assert snap.best_bid <= snap.midpoint <= snap.best_ask
    assert snap.spread == pytest.approx(snap.best_ask - snap.best_bid)
    assert snap.open_interest is None  # the never-reported field stays absent
    yes = snap.outcome_prices["Yes"]
    no = snap.outcome_prices["No"]
    assert abs(yes + no - 1.0) < 1e-6
    assert snap.observed_at.tzinfo is not None


def test_price_history_is_deterministic_bounded_and_honest_about_absence():
    stub = get_provider("stub")
    market = stub.search_markets("recession 2026", limit=1)[0]
    a = stub.get_price_history(market.market_id, outcome="Yes", start=START, end=END)
    b = stub.get_price_history(market.market_id, outcome="Yes", start=START, end=END)
    assert a == b
    assert len(a) == 24
    assert all(START < p.ts <= END for p in a)
    assert all(0.05 <= p.price <= 0.95 for p in a)
    assert [p.ts for p in a] == sorted(p.ts for p in a)
    # Unknown outcome / degenerate window: an honest empty answer, never an
    # invented series.
    assert stub.get_price_history(
        market.market_id, outcome="Maybe", start=START, end=END
    ) == []
    assert stub.get_price_history(
        market.market_id, outcome="Yes", start=END, end=START
    ) == []


def test_stub_capabilities_report_every_fixed_key():
    caps = get_provider("stub").capabilities()
    assert set(caps) == set(CAPABILITY_KEYS)
    assert all(v is True for v in caps.values())
