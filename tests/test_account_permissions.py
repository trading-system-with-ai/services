"""§2/§8/§33 account-permission enforcement (cash-account guide).

Four layers, each proven here:

1. The dataclass refuses the forbidden capabilities at CONSTRUCTION:
   AccountPermissions built with any display-and-refuse field True raises
   ValueError citing guide §33 — short stock, naked short options, covered
   calls, cash-secured puts and margin have no code path in this platform
   (no Sell-to-Open exists), so a True flag is a bug, not a configuration.
2. Settings hard-rejects the forbidden ALLOW_* env flags at STARTUP with the
   same §33 citation — an operator learns immediately, not at order time.
3. The real flags flow from the environment through the ONE factory
   (apps.gateway.deps.account_permissions_from_settings) into the §8 matrix:
   ALLOW_LONG_CALL=false degrades the bull LONG_CALL cell to LONG_STOCK.
4. GET /api/config renders ALL TEN fields, forbidden ones always false —
   the restriction is displayed, never lifted (Alpaca Paper capability does
   not override platform permissions, §2).
"""
import pytest
from pydantic import ValidationError

from apps.gateway.deps import account_permissions_from_settings
from libs.common.config import Settings, get_settings
from libs.trading_core.models import DirectionalBias, InstrumentType, IVRegime
from libs.trading_core.strategies import (
    FORBIDDEN_PERMISSION_FIELDS,
    AccountPermissions,
    select_instrument,
)

# All six §2/§33 display-and-refuse dataclass fields (incl. margin).
# Phase 3 (2026-08-17): only the naked shorts remain forbidden — forever
# (broker refusal at every approval level + §4 charter).
FORBIDDEN_FIELDS = (
    "naked_short_call",
    "naked_short_put",
)

# The five forbidden ALLOW_* env flags (guide §8; margin deliberately has NO
# env flag at all — §33 rule 7).
FORBIDDEN_ENV_FLAGS = (
        "ALLOW_NAKED_SHORT_CALL",
    "ALLOW_NAKED_SHORT_PUT",
)

# The complete §2 permission surface GET /api/config must render.
ALL_TEN_FIELDS = {
    "long_stock",
    "long_call",
    "long_put",
    "defined_risk_spreads",
    # Phase 2 unlock (2026-08-17): real flags now, default False.
    "covered_call",
    "cash_secured_put",
    # Phase 3 unlock (2026-08-17): margin-backed short stock.
    "short_stock",
    "margin",
    *FORBIDDEN_FIELDS,
}


# ---------------------------------------------------------------------------
# 1. Dataclass construction (guide §2, §33)
# ---------------------------------------------------------------------------


def test_forbidden_field_registry_matches_the_test_surface():
    """The module's own registry names exactly the six §33 fields."""
    assert set(FORBIDDEN_PERMISSION_FIELDS) == set(FORBIDDEN_FIELDS)


@pytest.mark.parametrize("field", FORBIDDEN_FIELDS)
def test_constructing_forbidden_permission_true_raises_citing_33(field):
    with pytest.raises(ValueError, match="§33") as excinfo:
        AccountPermissions(**{field: True})
    # The error names the offending field and the platform's inability —
    # not just a bare citation.
    message = str(excinfo.value)
    assert field in message
    assert "cannot execute" in message


def test_defaults_construct_and_forbidden_fields_read_false():
    p = AccountPermissions()
    for field in FORBIDDEN_FIELDS:
        assert getattr(p, field) is False
    assert (p.long_stock, p.long_call, p.long_put) == (True, True, True)
    assert p.defined_risk_spreads is False


def test_explicit_false_forbidden_fields_are_accepted():
    """Passing the forbidden fields explicitly False is fine — that is what
    the factory does after the Settings validator has guaranteed them."""
    p = AccountPermissions(
        short_stock=False,
        naked_short_call=False,
        naked_short_put=False,
        covered_call=False,
        cash_secured_put=False,
        margin=False,
    )
    assert p.long_stock is True


def test_defined_risk_spreads_stays_a_real_flag():
    """Spreads are DEFERRED, not forbidden: True constructs fine (§32 phase),
    it simply cannot produce a spread in v1 (see test_instrument_matrix)."""
    p = AccountPermissions(defined_risk_spreads=True)
    assert p.defined_risk_spreads is True


# ---------------------------------------------------------------------------
# 2. Settings startup rejection (guide §8, §33)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_flag", FORBIDDEN_ENV_FLAGS)
def test_settings_hard_rejects_forbidden_allow_flag_true(env_flag, monkeypatch):
    monkeypatch.setenv(env_flag, "true")
    with pytest.raises(ValidationError, match="§33") as excinfo:
        Settings()
    message = str(excinfo.value)
    # The message tells the operator the flag exists to make the restriction
    # explicit, not to lift it.
    assert "cannot execute" in message
    assert "not to lift it" in message


@pytest.mark.parametrize("env_flag", FORBIDDEN_ENV_FLAGS)
def test_settings_accepts_forbidden_allow_flag_false(env_flag, monkeypatch):
    monkeypatch.setenv(env_flag, "false")
    settings = Settings()
    assert getattr(settings, env_flag.lower()) is False


def test_settings_margin_flag_is_real_now():
    """SUPERSEDED (Phase 3): ALLOW_MARGIN exists as a REAL flag — margin
    exists to support shorting; levered long sizing stays off by charter."""
    from libs.common.config import Settings

    assert Settings().allow_margin is False  # opt-in, never a default


def test_settings_permission_defaults_match_guide_8():
    settings = Settings(
        _env_file=None  # defaults only — no developer .env leakage
    )
    assert settings.allow_long_stock is True
    assert settings.allow_long_call is True
    assert settings.allow_long_put is True
    assert settings.allow_defined_risk_spreads is False
    assert settings.allow_short_stock is False
    assert settings.allow_naked_short_call is False
    assert settings.allow_naked_short_put is False
    assert settings.allow_covered_call is False
    assert settings.allow_cash_secured_put is False


# ---------------------------------------------------------------------------
# 3. Environment -> factory -> §8 matrix (guide §8)
# ---------------------------------------------------------------------------


def test_allow_long_call_false_flows_through_factory_into_matrix(monkeypatch):
    """ALLOW_LONG_CALL=false must degrade the bull LONG_CALL cell to
    LONG_STOCK via the ONE factory — configuration IS enforcement."""
    monkeypatch.setenv("ALLOW_LONG_CALL", "false")
    get_settings.cache_clear()
    try:
        perms = account_permissions_from_settings()
        assert perms.long_call is False
        assert perms.long_stock is True  # only the flipped flag changed

        # BULL/STRONG/LOW is the one §8 cell that maps to LONG_CALL.
        d = select_instrument(
            DirectionalBias.BULL, "STRONG", IVRegime.LOW, perms
        )
        assert d.instrument is InstrumentType.LONG_STOCK
        assert d.contract_needed is False
        assert "long calls not permitted" in " ".join(d.rationale)
    finally:
        get_settings.cache_clear()  # never leak the flipped flag onward


def test_factory_defaults_equal_dataclass_defaults():
    """With nothing overridden the factory and the dataclass agree exactly,
    so the §10 gate chain and GET /api/config describe the same account."""
    assert account_permissions_from_settings() == AccountPermissions()


# ---------------------------------------------------------------------------
# 4. GET /api/config shows all ten fields (guide §2 display)
# ---------------------------------------------------------------------------


async def test_config_shows_all_ten_fields_with_forbidden_false(client):
    perms = (await client.get("/api/config")).json()["account_permissions"]

    assert set(perms) == ALL_TEN_FIELDS
    for field in FORBIDDEN_FIELDS:
        assert perms[field] is False, f"{field} must render false, always"
    assert perms["long_stock"] is True
    assert perms["long_call"] is True
    assert perms["long_put"] is True
    assert perms["defined_risk_spreads"] is False


async def test_config_reflects_a_flipped_real_flag(client, monkeypatch):
    """The config view renders the EFFECTIVE permissions (the factory), not a
    frozen default: flipping a real flag shows up; forbidden stay false."""
    monkeypatch.setenv("ALLOW_LONG_PUT", "false")
    get_settings.cache_clear()
    try:
        perms = (await client.get("/api/config")).json()["account_permissions"]
        assert perms["long_put"] is False
        assert perms["long_stock"] is True
        for field in FORBIDDEN_FIELDS:
            assert perms[field] is False
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Phase 2 UNLOCK (2026-08-17): covered_call / cash_secured_put are REAL
# flags now — the collateralized Sell-to-Open chain exists end to end.
# ---------------------------------------------------------------------------


def test_income_permissions_are_real_flags_now():
    from libs.trading_core.strategies.instrument import AccountPermissions

    p = AccountPermissions(covered_call=True, cash_secured_put=True)
    assert p.covered_call is True and p.cash_secured_put is True
    p3 = AccountPermissions(short_stock=True, margin=True)
    assert p3.short_stock is True and p3.margin is True
    # Default remains OFF: the user opts in through Settings.
    d = AccountPermissions()
    assert d.covered_call is False and d.cash_secured_put is False


def test_settings_accepts_income_allow_flags(monkeypatch):
    from libs.common.config import Settings, get_settings

    monkeypatch.setenv("ALLOW_COVERED_CALL", "true")
    monkeypatch.setenv("ALLOW_CASH_SECURED_PUT", "true")
    get_settings.cache_clear()
    try:
        s = Settings()
        assert s.allow_covered_call is True
        assert s.allow_cash_secured_put is True
    finally:
        get_settings.cache_clear()
