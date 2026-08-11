"""Tests for GET /api/config (plan §44 rule 2 made visible, §28).

The read-only config API must render the REAL engine parameter objects and
must never leak secret material: no key/secret/token/password field names
anywhere, and no configured secret VALUES in the serialized body.
"""
from libs.common.config import get_settings
from libs.trading_core.allocation import VolTargetParams
from libs.trading_core.backtest import BacktestParams
from libs.trading_core.contracts import SelectorParams
from libs.trading_core.exits import ExitParams
from libs.trading_core.risk import RiskLimits
from libs.trading_core.signals import DirectionalParams, RegimeParams
from libs.trading_core.strategies import AccountPermissions

EXPECTED_GROUPS = {
    "environment",
    "providers",
    "account_permissions",
    "risk_limits",
    "exit_params",
    "selector_params",
    "vol_target_params",
    "regime_params",
    "directional_params",
    "backtest_defaults",
    "paper_trading",
    "kill_switch",
}

FORBIDDEN_KEY_FRAGMENTS = ("key", "secret", "token", "password")


async def test_all_top_level_groups_present(client):
    r = await client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == EXPECTED_GROUPS


async def test_values_match_real_dataclasses(client):
    body = (await client.get("/api/config")).json()
    settings = get_settings()

    assert body["environment"] == settings.environment
    assert body["providers"] == {
        "market_data": settings.market_data_provider,
        # Explicit "is there a data source at all" booleans (§44 rule 18);
        # true here because the fixture opts into the stub providers.
        "market_data_configured": True,
        "llm": settings.llm_provider,
        "llm_configured": True,
        "llm_model": settings.llm_model,
        # The execution venue, same shape as the data providers. The fixture
        # opts into BROKER_PROVIDER=simulated (internal fills, dev only) —
        # there is no default venue either (§44 rule 18).
        "broker": settings.broker_provider,
        "broker_configured": True,
    }

    perms = AccountPermissions()
    assert body["account_permissions"] == {
        "long_stock": perms.long_stock,
        "long_call": perms.long_call,
        "long_put": perms.long_put,
        "defined_risk_spreads": perms.defined_risk_spreads,
    }

    limits = RiskLimits()
    rl = body["risk_limits"]
    assert rl["abs_max_trade_risk"] == 0.015 == limits.abs_max_trade_risk
    assert rl["budget_very_strong"] == limits.budget_very_strong
    assert rl["heat_reject"] == limits.heat_reject
    assert rl["max_delta_notional_pct_nav"] == limits.max_delta_notional_pct_nav
    # cash_floors keyed by regime NAME
    assert rl["cash_floors"] == {
        regime.value: floor for regime, floor in limits.cash_floors.items()
    }
    assert rl["cash_floors"]["STRONG_BEAR"] == 0.60
    # correlation_buckets as {name: [tickers]}
    assert rl["correlation_buckets"] == {
        name: list(members) for name, members in limits.correlation_buckets.items()
    }
    assert "NVDA" in rl["correlation_buckets"]["TECH_MEGA"]

    exits = ExitParams()
    assert body["exit_params"]["premium_hard_stop_pct"] == 0.45 == exits.premium_hard_stop_pct
    assert body["exit_params"]["dte_exit_threshold"] == exits.dte_exit_threshold
    assert body["exit_params"]["atr_trail_k"] == exits.atr_trail_k

    sel = SelectorParams()
    assert body["selector_params"]["dte_min"] == sel.dte_min
    assert body["selector_params"]["abs_delta_max"] == sel.abs_delta_max
    assert body["selector_params"]["top_n"] == sel.top_n

    vol = VolTargetParams()
    assert body["vol_target_params"] == {
        "target_vol": vol.target_vol,
        "max_multiplier": vol.max_multiplier,
        "min_multiplier": vol.min_multiplier,
    }

    reg = RegimeParams()
    assert body["regime_params"]["sma_slow"] == reg.sma_slow
    assert body["regime_params"]["extreme_atr_pct"] == reg.extreme_atr_pct

    d = DirectionalParams()
    assert body["directional_params"]["bias_threshold"] == d.bias_threshold
    assert body["directional_params"]["rsi_bull_zone"] == list(d.rsi_bull_zone)

    bt = BacktestParams()
    assert body["backtest_defaults"]["oos_split"] == bt.oos_split
    assert body["backtest_defaults"]["warmup_bars"] == bt.warmup_bars
    assert body["backtest_defaults"]["entry_edge_threshold"] == bt.entry_edge_threshold

    assert body["paper_trading"] == {
        "initial_cash": settings.paper_initial_cash,
        "slippage_bps": settings.paper_slippage_bps,
        "commission_per_share": settings.paper_commission_per_share,
        "commission_per_contract": settings.paper_commission_per_contract,
    }


def _walk_keys(node, path=""):
    """Yield every dict key (with its path) anywhere in a JSON structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield f"{path}.{k}", str(k)
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_keys(v, f"{path}[{i}]")


async def test_secret_absence_property(client, monkeypatch):
    """HARD RULE: no secret names or configured secret VALUES may appear.

    Dummy secrets are injected into the environment and Settings is rebuilt,
    so if any code path dict-dumped Settings into the response, the dummy
    VALUES would show up in the serialized body — proving the rule with real
    material, not just field-name hygiene.
    """
    dummy_llm = "dummy-llm-secret-value-XYZZY-9c1"
    dummy_massive = "dummy-massive-secret-value-PLUGH-7f2"
    monkeypatch.setenv("LLM_API_KEY", dummy_llm)
    monkeypatch.setenv("MASSIVE_API_KEY", dummy_massive)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.llm_api_key == dummy_llm  # the dummies really are configured
        assert settings.massive_api_key == dummy_massive

        r = await client.get("/api/config")
        assert r.status_code == 200

        # 1) no key NAME anywhere contains key/secret/token/password
        for key_path, key in _walk_keys(r.json()):
            lowered = key.lower()
            for fragment in FORBIDDEN_KEY_FRAGMENTS:
                assert fragment not in lowered, (
                    f"forbidden fragment {fragment!r} in response key {key_path}"
                )

        # 2) no configured secret VALUE appears anywhere in the raw body
        assert dummy_llm not in r.text
        assert dummy_massive not in r.text
    finally:
        # drop the poisoned Settings so later tests rebuild from a clean env
        get_settings.cache_clear()


async def test_kill_switch_reflects_pause_round_trip(client):
    body = (await client.get("/api/config")).json()
    assert body["kill_switch"] == {
        "trading_enabled": False,
        "reason": "startup default: trading disabled",
    }

    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200
    body = (await client.get("/api/config")).json()
    assert body["kill_switch"]["trading_enabled"] is True

    r = await client.post("/api/trading/pause", json={"reason": "config api test"})
    assert r.status_code == 200
    body = (await client.get("/api/config")).json()
    assert body["kill_switch"] == {
        "trading_enabled": False,
        "reason": "config api test",
    }