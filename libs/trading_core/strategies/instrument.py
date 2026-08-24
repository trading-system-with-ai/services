"""Instrument Selection v1 (development plan §8, constrained by §5).

Pure, deterministic mapping from (directional bias, signal strength,
volatility regime) to the instrument to trade — no DB, no FastAPI, no
market data. The §8 matrix is applied under the §5 account constraints:
NO short stock, NO naked short options, and (in this account today) NO
defined-risk spreads — so every spread cell of the §8 table DEGRADES to
the nearest §5-legal instrument, and the degradation is spelled out in the
rationale. NO_TRADE is a valid output, never a failure (plan §8).

V1 matrix (VERY_STRONG is treated as STRONG; §8 table, §5 degradations):

===========  ============  ==========================================
Direction    Vol regime    WEAK        MODERATE     STRONG/V.STRONG
===========  ============  ==========================================
BULL         LOW           LONG_STOCK  LONG_STOCK   LONG_CALL
BULL         NORMAL/HIGH   LONG_STOCK  LONG_STOCK   LONG_STOCK (spread cell degraded)
BULL         EXTREME       LONG_STOCK  LONG_STOCK   LONG_STOCK (never buy extreme premium, §7)
BEAR         LOW           NO_TRADE    LONG_PUT     LONG_PUT
BEAR         NORMAL        NO_TRADE    LONG_PUT     LONG_PUT (spread cell degraded)
BEAR         HIGH          NO_TRADE    NO_TRADE     LONG_PUT (degraded, higher |delta|)
BEAR         EXTREME       NO_TRADE    NO_TRADE     NO_TRADE (no short stock, §5)
NEUTRAL      any           NO_TRADE    NO_TRADE     NO_TRADE
===========  ============  ==========================================

Every decision carries a rationale citing its §8 matrix cell AND any §5
degradation applied, so the user always sees which cell fired and why the
instrument differs from the ideal one (plan §37).
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.trading_core.models import DirectionalBias, InstrumentType, IVRegime

# Signal-strength tier names as produced by the risk engine's public
# strength_tier() (libs.trading_core.risk.engine, plan §12.2).
_VALID_STRENGTHS = ("WEAK", "MODERATE", "STRONG", "VERY_STRONG")

# The DISPLAY-AND-REFUSE permission fields (guide §2, §33) with the §33 name
# of the capability each one would describe. These are hard account
# constraints, enforced in ALL environments including Alpaca Paper: the
# platform has no code path for any of them (no Sell-to-Open exists anywhere,
# no margin model exists anywhere), so the fields exist only so the
# restriction is EXPLICIT and visible — constructing them True is a
# programming error, refused at construction (see AccountPermissions).
# 2026-08-17 Phase 3 UNLOCK: short_stock / margin moved OUT — the
# margin-backed short chain exists (mirrored exits, stop-based risk with a
# gap factor, broker short/cover orders, backtest leg). ONLY the naked
# shorts remain, PERMANENTLY: Alpaca offers them at no approval level
# (broker refusal) and the §4 charter forbids unbounded premium risk.
FORBIDDEN_PERMISSION_FIELDS: dict[str, str] = {
    "naked_short_call": "naked short calls",
    "naked_short_put": "naked short puts",
}


@dataclass(frozen=True)
class AccountPermissions:
    """What the account is allowed to trade (guide §2/§5, as explicit flags).

    §2/§5 hard constraints are the DEFAULTS here, not assumptions baked into
    logic. Two kinds of field exist:

    - REAL flags (``long_stock``, ``long_call``, ``long_put``,
      ``defined_risk_spreads``): flipping one re-derives the §8 matrix; it
      never edits the matrix itself. ``defined_risk_spreads`` is a genuine
      deferred capability — off until the account is explicitly approved.
    - DISPLAY-AND-REFUSE fields (:data:`FORBIDDEN_PERMISSION_FIELDS` —
      ``short_stock``, ``naked_short_call``, ``naked_short_put``,
      ``covered_call``, ``cash_secured_put``, ``margin``): the guide's §2
      permission block names them so the restriction is explicit, but the
      platform has NO code path for any of them (Sell-to-Open does not exist
      in this system, and neither does a margin model). They therefore
      default False and CANNOT be constructed True: ``__post_init__`` raises
      ``ValueError`` citing guide §33. They are rendered (e.g. by
      GET /api/config) so the constraint is visible, never so it can be
      lifted.

    Alpaca Paper capability does NOT override platform permissions (§2):
    the source of truth is this object, never what Alpaca Paper technically
    permits — paper mode mirrors the intended real cash-account restrictions
    (§23), in all environments.
    """

    long_stock: bool = True
    long_call: bool = True
    long_put: bool = True
    defined_risk_spreads: bool = False
    # Display-and-refuse (guide §2, §33): always False, enforced below.
    short_stock: bool = False
    naked_short_call: bool = False
    naked_short_put: bool = False
    covered_call: bool = False
    cash_secured_put: bool = False
    margin: bool = False

    def __post_init__(self) -> None:
        """Refuse any forbidden permission at construction (guide §33).

        These are non-negotiable: the platform cannot execute them (no
        Sell-to-Open, no margin model), so a True value is not a
        configuration — it is a bug, and it fails loudly here rather than
        pretending a capability exists.
        """
        for name, capability in FORBIDDEN_PERMISSION_FIELDS.items():
            if getattr(self, name):
                raise ValueError(
                    f"AccountPermissions.{name}=True violates guide §33 "
                    f"(non-negotiable rules): this platform cannot execute "
                    f"{capability} — no code path for it exists (no "
                    "Sell-to-Open, no margin). The field exists to make the "
                    "restriction explicit, not to lift it, and Alpaca Paper "
                    "capability does not override platform permissions (§2)."
                )


@dataclass
class InstrumentDecision:
    """One fully explainable instrument decision (plan §8, §37).

    - ``instrument``: the §5-legal instrument to trade (NO_TRADE is a valid
      output, never an error).
    - ``contract_needed``: ``True`` for option instruments — the caller must
      run the §9 contract selector next; ``False`` for stock / no trade.
    - ``rationale``: every decision cites its §8 matrix cell AND any §5 /
      permission degradation applied, with the inputs that selected the cell.
    """

    instrument: InstrumentType
    contract_needed: bool
    rationale: list[str]


def _finalize(
    instrument: InstrumentType,
    rationale: list[str],
    permissions: AccountPermissions,
) -> InstrumentDecision:
    """Apply the §5 permission flags as the LAST word, degrading with an
    explicit rationale line (plan §5: constraints outrank the matrix).

    Degradation ladder: a bull option cell falls back to LONG_STOCK (the
    underlying exposure is still available), then NO_TRADE; a bear option
    cell falls back straight to NO_TRADE — there is NO short stock in this
    system (§5), so no stock fallback exists on the bear side.
    """
    if (
        instrument is InstrumentType.BULL_CALL_SPREAD
        and not permissions.defined_risk_spreads
    ):
        rationale.append(
            "§5 permissions: defined-risk spreads not permitted -> degraded "
            "to LONG_STOCK (bull exposure without the short leg)."
        )
        instrument = InstrumentType.LONG_STOCK
    if (
        instrument is InstrumentType.BEAR_PUT_SPREAD
        and not permissions.defined_risk_spreads
    ):
        rationale.append(
            "§5 permissions: defined-risk spreads not permitted -> degraded "
            "to LONG_PUT, preferring a higher-|delta| (deeper ITM) strike to "
            "cut the premium/theta paid (§9); short stock does not exist in "
            "this system (§5)."
        )
        instrument = InstrumentType.LONG_PUT
    if instrument is InstrumentType.SHORT_STOCK and not (
        permissions.short_stock and permissions.margin
    ):
        rationale.append(
            "§5 permissions: short stock requires BOTH short_stock and "
            "margin enabled -> NO_TRADE."
        )
        instrument = InstrumentType.NO_TRADE
    if instrument is InstrumentType.LONG_CALL and not permissions.long_call:
        rationale.append(
            "§5 permissions: long calls not permitted -> degraded to "
            "LONG_STOCK (bull exposure via the underlying instead)."
        )
        instrument = InstrumentType.LONG_STOCK
    if instrument is InstrumentType.LONG_PUT and not permissions.long_put:
        rationale.append(
            "§5 permissions: long puts not permitted and short stock does "
            "not exist in this system (§5) -> NO_TRADE."
        )
        instrument = InstrumentType.NO_TRADE
    if instrument is InstrumentType.LONG_STOCK and not permissions.long_stock:
        rationale.append(
            "§5 permissions: long stock not permitted -> NO_TRADE."
        )
        instrument = InstrumentType.NO_TRADE
    return InstrumentDecision(
        instrument=instrument,
        contract_needed=instrument
        in (
            InstrumentType.LONG_CALL,
            InstrumentType.LONG_PUT,
            InstrumentType.BULL_CALL_SPREAD,
            InstrumentType.BEAR_PUT_SPREAD,
        ),
        rationale=rationale,
    )


def select_instrument(
    direction: DirectionalBias,
    strength: str | None,
    vol_regime: IVRegime | None,
    permissions: AccountPermissions = AccountPermissions(),
) -> InstrumentDecision:
    """Select the instrument for a signal via the §8 matrix under §5.

    - ``direction`` / ``strength``: the live signal engine's bias and the
      risk engine's tier name (``strength_tier``); ``None`` strength means
      no valid signal (|edge| below the weak threshold, plan §12.2).
    - ``vol_regime``: the §7 classification; ``None`` (chain/IV data
      unavailable — honest null) is treated as NORMAL with an explicit
      rationale line, the matrix's no-information column.
    - ``permissions``: §5 account flags, applied LAST and always explained.

    VERY_STRONG is treated as STRONG (§8 v1: sizing already rewards the
    stronger tier via the §12.2 risk budget; the instrument choice does
    not change). Raises ``ValueError`` on an unknown strength string.

    Returns an :class:`InstrumentDecision`; NO_TRADE is a valid output,
    never a failure (plan §8).
    """
    if strength is not None and strength not in _VALID_STRENGTHS:
        raise ValueError(
            f"strength must be one of {_VALID_STRENGTHS} or None, "
            f"got {strength!r}"
        )

    rationale: list[str] = []

    # --- No directional edge -> NO_TRADE (§8: a valid output) -------------
    if direction is DirectionalBias.NEUTRAL or strength is None:
        why = (
            "bias NEUTRAL"
            if direction is DirectionalBias.NEUTRAL
            else "strength None (|edge| below the weak threshold, §12.2)"
        )
        rationale.append(
            f"§8: {why}: no directional edge — NO TRADE is a valid output."
        )
        return _finalize(InstrumentType.NO_TRADE, rationale, permissions)

    # --- Normalize inputs the matrix keys on -------------------------------
    effective_strength = strength
    if strength == "VERY_STRONG":
        effective_strength = "STRONG"
        rationale.append(
            "§8 v1: VERY_STRONG treated as STRONG (extra conviction is "
            "rewarded by the §12.2 risk budget, not the instrument)."
        )
    if vol_regime is None:
        vol_regime = IVRegime.NORMAL
        rationale.append(
            "§7: vol regime unknown (IV data unavailable — honest null) "
            "-> treated as NORMAL for instrument selection."
        )

    cell = f"{direction.value}/{strength}/{vol_regime.value}"

    # --- EXTREME vol overrides the whole column (§7) ------------------------
    if vol_regime is IVRegime.EXTREME:
        if direction is DirectionalBias.BULL:
            rationale.append(
                f"§8 {cell}: EXTREME IV — never buy extreme premium (§7) "
                "-> LONG_STOCK for bull exposure without paying it."
            )
            return _finalize(InstrumentType.LONG_STOCK, rationale, permissions)
        if permissions.short_stock and permissions.margin:
            rationale.append(
                f"§8 {cell}: EXTREME IV — premium unbuyable (§7); short "
                "stock expresses the bear WITHOUT paying it (Phase 3; "
                "margin-backed, stop-based risk)."
            )
            return _finalize(InstrumentType.SHORT_STOCK, rationale, permissions)
        rationale.append(
            f"§8 {cell}: EXTREME IV — never buy extreme premium (§7), and "
            "short stock is not enabled (§5/Phase 3) -> NO_TRADE."
        )
        return _finalize(InstrumentType.NO_TRADE, rationale, permissions)

    # --- BULL column (§8 table, §5 degradations) ---------------------------
    if direction is DirectionalBias.BULL:
        if effective_strength == "STRONG":
            if vol_regime is IVRegime.LOW:
                rationale.append(
                    f"§8 {cell}: strong bull + cheap premium maps to "
                    "Long Call — premium is worth buying in LOW IV."
                )
                return _finalize(
                    InstrumentType.LONG_CALL, rationale, permissions
                )
            rationale.append(
                f"§8 {cell} maps to Bull Call Spread — defined-risk bull "
                "premium with the vega cost hedged by the short leg."
            )
            return _finalize(
                InstrumentType.BULL_CALL_SPREAD, rationale, permissions
            )
        if effective_strength == "MODERATE":
            rationale.append(
                f"§8 {cell}: moderate bull maps to Stock — not enough "
                "conviction to pay option premium/theta."
            )
            return _finalize(InstrumentType.LONG_STOCK, rationale, permissions)
        # WEAK
        rationale.append(
            f"§8 {cell}: weak bull is the 'Stock / No Trade' cell — stock "
            "chosen; the §12.2 risk budget already scales weak signals down."
        )
        return _finalize(InstrumentType.LONG_STOCK, rationale, permissions)

    # --- BEAR column (§8 table, §5: no short stock, ever) ------------------
    if effective_strength == "STRONG":
        if vol_regime is IVRegime.LOW:
            rationale.append(
                f"§8 {cell}: strong bear + cheap premium maps to Long Put "
                "— the only §5-legal bearish instrument."
            )
            return _finalize(InstrumentType.LONG_PUT, rationale, permissions)
        rationale.append(
            f"§8 {cell} maps to Bear Put Spread — defined-risk bear premium "
            "in richer IV."
        )
        return _finalize(InstrumentType.BEAR_PUT_SPREAD, rationale, permissions)
    if effective_strength == "MODERATE":
        if vol_regime is IVRegime.HIGH:
            if permissions.defined_risk_spreads:
                rationale.append(
                    f"§8 {cell} maps to Bear Put Spread — expensive premium "
                    "made affordable by the short leg."
                )
                return _finalize(
                    InstrumentType.BEAR_PUT_SPREAD, rationale, permissions
                )
            if permissions.short_stock and permissions.margin:
                rationale.append(
                    f"§8 {cell}: expensive premium and no spreads — short "
                    "stock expresses the bear without paying it (Phase 3)."
                )
                return _finalize(
                    InstrumentType.SHORT_STOCK, rationale, permissions
                )
            rationale.append(
                f"§8 {cell}: the 'Higher-delta Long Put / No Trade' cell — "
                "expensive premium with no spreads available and short "
                "stock is not enabled (§5/Phase 3) -> pass; NO TRADE is a "
                "valid output."
            )
            return _finalize(InstrumentType.NO_TRADE, rationale, permissions)
        rationale.append(
            f"§8 {cell} maps to Bear Put Spread — moderate bear expressed "
            "with defined risk."
        )
        return _finalize(InstrumentType.BEAR_PUT_SPREAD, rationale, permissions)
    # WEAK
    rationale.append(
        f"§8 {cell}: weak bear is explicitly No Trade — a weak edge does "
        "not justify paying put premium, and no short stock exists (§5)."
    )
    return _finalize(InstrumentType.NO_TRADE, rationale, permissions)
