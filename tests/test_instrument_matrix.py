"""Tests for Instrument Selection v1 (development plan §8 under §5).

Asserts EVERY cell of the v1 matrix — both §5 degradations (spread cells
-> stock/put, extreme-vol cells -> stock/no-trade), the permission flips,
the NEUTRAL / no-strength NO_TRADE cells, the VERY_STRONG == STRONG
collapsing, honest-null vol handling, contract_needed wiring, and the
rationale content (§8 cell citation AND §5 degradation, plan §37).
"""
import pytest

from libs.trading_core.models import DirectionalBias, InstrumentType, IVRegime
from libs.trading_core.strategies import (
    AccountPermissions,
    InstrumentDecision,
    select_instrument,
)

BULL = DirectionalBias.BULL
BEAR = DirectionalBias.BEAR
NEUTRAL = DirectionalBias.NEUTRAL
LOW, NORMAL, HIGH, EXTREME = (
    IVRegime.LOW,
    IVRegime.NORMAL,
    IVRegime.HIGH,
    IVRegime.EXTREME,
)

STOCK = InstrumentType.LONG_STOCK
CALL = InstrumentType.LONG_CALL
PUT = InstrumentType.LONG_PUT
NO_TRADE = InstrumentType.NO_TRADE


def text(d: InstrumentDecision) -> str:
    return " ".join(d.rationale)


# ---------------------------------------------------------------------------
# Defaults (plan §5: the account constraints ARE the default permissions)
# ---------------------------------------------------------------------------


def test_default_permissions_match_plan_5():
    p = AccountPermissions()
    assert (p.long_stock, p.long_call, p.long_put, p.defined_risk_spreads) == (
        True,
        True,
        True,
        False,
    )


def test_unknown_strength_string_rejected():
    with pytest.raises(ValueError):
        select_instrument(BULL, "STRONGISH", NORMAL)


# ---------------------------------------------------------------------------
# The FULL v1 matrix, cell by cell (plan §8 table degraded per §5; spreads
# unavailable; VERY_STRONG treated as STRONG)
# ---------------------------------------------------------------------------

MATRIX = [
    # --- no directional edge: NO_TRADE is a valid output (§8) -------------
    *[(NEUTRAL, s, v, NO_TRADE) for s in (None, "WEAK", "MODERATE", "STRONG", "VERY_STRONG")
      for v in (LOW, NORMAL, HIGH, EXTREME, None)],
    *[(d, None, v, NO_TRADE) for d in (BULL, BEAR)
      for v in (LOW, NORMAL, HIGH, EXTREME, None)],
    # --- BULL ------------------------------------------------------------
    (BULL, "WEAK", LOW, STOCK),
    (BULL, "WEAK", NORMAL, STOCK),
    (BULL, "WEAK", HIGH, STOCK),
    (BULL, "WEAK", EXTREME, STOCK),
    (BULL, "MODERATE", LOW, STOCK),
    (BULL, "MODERATE", NORMAL, STOCK),
    (BULL, "MODERATE", HIGH, STOCK),
    (BULL, "MODERATE", EXTREME, STOCK),
    (BULL, "STRONG", LOW, CALL),
    (BULL, "STRONG", NORMAL, STOCK),  # spread cell degraded (§5)
    (BULL, "STRONG", HIGH, STOCK),  # spread cell degraded (§5)
    (BULL, "STRONG", EXTREME, STOCK),  # never buy extreme premium (§7)
    (BULL, "VERY_STRONG", LOW, CALL),
    (BULL, "VERY_STRONG", NORMAL, STOCK),
    (BULL, "VERY_STRONG", HIGH, STOCK),
    (BULL, "VERY_STRONG", EXTREME, STOCK),
    # --- BEAR (no short stock, ever — §5) --------------------------------
    (BEAR, "WEAK", LOW, NO_TRADE),
    (BEAR, "WEAK", NORMAL, NO_TRADE),
    (BEAR, "WEAK", HIGH, NO_TRADE),
    (BEAR, "WEAK", EXTREME, NO_TRADE),
    (BEAR, "MODERATE", LOW, PUT),
    (BEAR, "MODERATE", NORMAL, PUT),  # spread cell degraded (§5)
    (BEAR, "MODERATE", HIGH, NO_TRADE),  # expensive premium, no spreads -> pass
    (BEAR, "MODERATE", EXTREME, NO_TRADE),
    (BEAR, "STRONG", LOW, PUT),
    (BEAR, "STRONG", NORMAL, PUT),  # spread cell degraded (§5)
    (BEAR, "STRONG", HIGH, PUT),  # degraded, higher-|delta| preference
    (BEAR, "STRONG", EXTREME, NO_TRADE),  # extreme premium + no short stock
    (BEAR, "VERY_STRONG", LOW, PUT),
    (BEAR, "VERY_STRONG", NORMAL, PUT),
    (BEAR, "VERY_STRONG", HIGH, PUT),
    (BEAR, "VERY_STRONG", EXTREME, NO_TRADE),
]


@pytest.mark.parametrize(("direction", "strength", "vol", "expected"), MATRIX)
def test_matrix_cell(direction, strength, vol, expected):
    d = select_instrument(direction, strength, vol)
    assert d.instrument is expected
    # contract_needed wiring: options need the §9 selector, stock/no-trade
    # do not.
    assert d.contract_needed is (expected in (CALL, PUT))
    # Every decision explains itself (plan §37): a §8 citation and at least
    # one rationale line.
    assert d.rationale
    assert "§8" in text(d)


def test_matrix_covers_every_direction_strength_vol_combination():
    """The parametrized table enumerates all 3*5*5 = 75 combinations except
    the 8 directional (BULL/BEAR × real strength × vol None) ones, which
    test_vol_none_treated_as_normal_with_honest_rationale covers."""
    combos = {(d, s, v) for d, s, v, _ in MATRIX}
    assert len(combos) == len(MATRIX) == 75 - 8


def test_very_strong_treated_as_strong_identically():
    for direction in (BULL, BEAR):
        for vol in (LOW, NORMAL, HIGH, EXTREME):
            strong = select_instrument(direction, "STRONG", vol)
            very = select_instrument(direction, "VERY_STRONG", vol)
            assert very.instrument is strong.instrument
            assert very.contract_needed is strong.contract_needed
    d = select_instrument(BULL, "VERY_STRONG", LOW)
    assert "VERY_STRONG treated as STRONG" in text(d)


def test_vol_none_treated_as_normal_with_honest_rationale():
    for direction, strength in (
        (BULL, "WEAK"),
        (BULL, "MODERATE"),
        (BULL, "STRONG"),
        (BULL, "VERY_STRONG"),
        (BEAR, "WEAK"),
        (BEAR, "MODERATE"),
        (BEAR, "STRONG"),
        (BEAR, "VERY_STRONG"),
    ):
        with_none = select_instrument(direction, strength, None)
        with_normal = select_instrument(direction, strength, NORMAL)
        assert with_none.instrument is with_normal.instrument
        assert "unknown" in text(with_none)
        assert "NORMAL" in text(with_none)


# ---------------------------------------------------------------------------
# Rationale content: every cell cites §8; degradations cite §5/§7 (plan §37)
# ---------------------------------------------------------------------------


def test_no_edge_rationale():
    d = select_instrument(NEUTRAL, "STRONG", LOW)
    assert d.instrument is NO_TRADE
    assert "no directional edge" in text(d)
    d2 = select_instrument(BULL, None, LOW)
    assert d2.instrument is NO_TRADE
    assert "no directional edge" in text(d2)


def test_bull_spread_cell_degradation_rationale():
    d = select_instrument(BULL, "STRONG", NORMAL)
    assert d.instrument is STOCK
    assert "BULL/STRONG/NORMAL" in text(d)
    assert "Bull Call Spread" in text(d)
    assert "spreads not permitted" in text(d)
    assert "LONG_STOCK" in text(d)


def test_bear_spread_cell_degradation_notes_higher_delta():
    d = select_instrument(BEAR, "STRONG", HIGH)
    assert d.instrument is PUT
    assert "BEAR/STRONG/HIGH" in text(d)
    assert "Bear Put Spread" in text(d)
    assert "spreads not permitted" in text(d)
    assert "higher-|delta|" in text(d)


def test_extreme_vol_rationales():
    bull = select_instrument(BULL, "STRONG", EXTREME)
    assert bull.instrument is STOCK
    assert "never buy extreme premium" in text(bull)
    assert "§7" in text(bull)
    bear = select_instrument(BEAR, "STRONG", EXTREME)
    assert bear.instrument is NO_TRADE
    assert "no short stock" in text(bear)
    assert "§5" in text(bear)


def test_bull_weak_cites_stock_no_trade_cell_and_risk_budget():
    d = select_instrument(BULL, "WEAK", NORMAL)
    assert d.instrument is STOCK
    assert "Stock / No Trade" in text(d)
    assert "risk budget" in text(d)


def test_bear_moderate_high_cites_pass_cell():
    d = select_instrument(BEAR, "MODERATE", HIGH)
    assert d.instrument is NO_TRADE
    assert "Higher-delta Long Put / No Trade" in text(d)
    assert "expensive premium" in text(d)


def test_bear_weak_is_explicit_no_trade():
    d = select_instrument(BEAR, "WEAK", LOW)
    assert d.instrument is NO_TRADE
    assert "No Trade" in text(d)


# ---------------------------------------------------------------------------
# Permission flips (§5 configurable constraints, applied last, explained)
# ---------------------------------------------------------------------------


def test_long_call_off_degrades_bull_low_to_stock():
    d = select_instrument(
        BULL, "STRONG", LOW, permissions=AccountPermissions(long_call=False)
    )
    assert d.instrument is STOCK
    assert d.contract_needed is False
    assert "long calls not permitted" in text(d)
    assert "§5" in text(d)


def test_long_put_off_degrades_bear_to_no_trade():
    for vol in (LOW, NORMAL, HIGH):
        d = select_instrument(
            BEAR, "STRONG", vol, permissions=AccountPermissions(long_put=False)
        )
        assert d.instrument is NO_TRADE
        assert d.contract_needed is False
        assert "long puts not permitted" in text(d)
        assert "short stock" in text(d)  # no stock fallback on the bear side


def test_long_stock_off_degrades_stock_cells_to_no_trade():
    perms = AccountPermissions(long_stock=False)
    for direction, strength, vol in (
        (BULL, "MODERATE", NORMAL),
        (BULL, "WEAK", LOW),
        (BULL, "STRONG", NORMAL),  # spread -> stock -> no trade
        (BULL, "STRONG", EXTREME),
    ):
        d = select_instrument(direction, strength, vol, permissions=perms)
        assert d.instrument is NO_TRADE
        assert "long stock not permitted" in text(d)
    # ... but a cell that never needed stock is untouched:
    d = select_instrument(BULL, "STRONG", LOW, permissions=perms)
    assert d.instrument is CALL


def test_long_call_and_stock_off_bull_low_becomes_no_trade():
    d = select_instrument(
        BULL,
        "STRONG",
        LOW,
        permissions=AccountPermissions(long_call=False, long_stock=False),
    )
    assert d.instrument is NO_TRADE
    # Both degradation steps are explained (plan §37).
    assert "long calls not permitted" in text(d)
    assert "long stock not permitted" in text(d)


def test_spreads_permitted_still_degrades_in_v1_with_honest_note():
    """defined_risk_spreads=True cannot produce a spread in v1 —
    InstrumentType has no spread member — so the cell still degrades, and
    the rationale says why (honest, never silent)."""
    d = select_instrument(
        BULL,
        "STRONG",
        NORMAL,
        permissions=AccountPermissions(defined_risk_spreads=True),
    )
    assert d.instrument is STOCK
    assert "not implemented in v1" in text(d)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    a = select_instrument(BEAR, "MODERATE", NORMAL)
    b = select_instrument(BEAR, "MODERATE", NORMAL)
    assert a == b
