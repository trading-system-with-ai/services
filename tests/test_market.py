from datetime import datetime, timezone

from libs.market_data.stub import StubProvider
from libs.trading_core.models import MarketRegime


async def test_market_overview_shape(client):
    r = await client.get("/api/market/overview")
    assert r.status_code == 200
    body = r.json()

    assert body["provider"] == "stub"
    assert body["stale"] is False
    datetime.fromisoformat(body["as_of"])  # valid ISO-8601
    assert body["market_regime"] in {m.value for m in MarketRegime}

    indices = body["indices"]
    assert len(indices) == 3
    symbols = {i["symbol"] for i in indices}
    assert symbols == {"SPY", "QQQ", "VIX"}
    for idx in indices:
        assert idx["price"] > 0
        assert isinstance(idx["change_pct"], float)
        datetime.fromisoformat(idx["ts"])  # valid ISO-8601


def test_stub_provider_deterministic_within_minute():
    provider = StubProvider()
    now = datetime(2026, 8, 10, 14, 30, 12, tzinfo=timezone.utc)
    later_same_minute = now.replace(second=45)
    a = provider.get_quotes(["SPY", "QQQ", "VIX"], now=now)
    b = provider.get_quotes(["SPY", "QQQ", "VIX"], now=later_same_minute)
    assert [(q.symbol, q.price, q.change_pct) for q in a] == [
        (q.symbol, q.price, q.change_pct) for q in b
    ]
