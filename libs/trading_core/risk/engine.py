"""Risk Engine v0 (development plan §12, §13, §17).

Pure, deterministic risk decisioning — no DB, no FastAPI, no market data.
The engine is architecturally INDEPENDENT from the strategy engine (plan
§17): it receives a :class:`RiskRequest` (the proposed trade plus the signal
edge already computed elsewhere) and a :class:`PortfolioSnapshot`, and
decides. It never computes signals itself, and risk limits always have
PRIORITY over strategy confidence (plan §44 rule 20) — no confidence score
may override a limit (plan §12.2).

Every threshold is a parameter on :class:`RiskLimits`, never a hardcoded
truth. Decision pipeline (each step cites its plan section):

1. Kill switch (plan §18) — trading disabled rejects everything.
2. Portfolio heat gate (plan §12.5) — heat >= ``heat_reject`` rejects all
   NEW risk; existing positions are untouched.
3. Signal strength tier -> risk budget (plan §12.2), optionally scaled by
   the §14 vol-targeting ``budget_multiplier`` and ALWAYS hard-capped by
   ``abs_max_trade_risk`` (§14 never overrides hard caps).
4. Base sizing from stop distance (plan §12.1).
5. Quantity clamps: single-name risk & capital (plan §12.3), correlation
   bucket (plan §12.4), heat headroom (plan §12.5), regime cash floor
   (plan §13); then, when the caller supplies greeks, the post-trade
   portfolio greek limits (plan §16) at the approved quantity.
6. Decision + human-readable explanations with real numbers (plan §36).

The gateway records every risk decision as an audit event in the same
transaction (house rule; plan §19) — that side effect lives in the gateway,
never here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from libs.trading_core.greeks import PortfolioGreeks, PositionGreeksInput
from libs.trading_core.models import MarketRegime, RiskDecision

# Tolerance used when flooring float ratios into share counts, so that
# exactly-at-the-limit quantities (e.g. 20000.0 / 20.0) are not lost to
# float representation error.
_EPS = 1e-9


def _default_cash_floors() -> dict[MarketRegime, float]:
    """Regime-dependent minimum cash reserve as a fraction of NAV (plan §13)."""
    return {
        MarketRegime.STRONG_BULL: 0.15,
        MarketRegime.MILD_BULL: 0.25,
        MarketRegime.NEUTRAL_RANGE: 0.40,
        MarketRegime.MILD_BEAR: 0.50,
        MarketRegime.STRONG_BEAR: 0.60,
        MarketRegime.TRANSITION: 0.50,
    }


def _default_correlation_buckets() -> dict[str, tuple[str, ...]]:
    """Correlated-underlying buckets sharing one risk cap (plan §12.4)."""
    return {
        "TECH_MEGA": (
            "NVDA",
            "AMD",
            "AVGO",
            "MSFT",
            "GOOGL",
            "META",
            "AAPL",
            "QQQ",
            "SMH",
            "TSLA",
        ),
    }


@dataclass(frozen=True)
class RiskLimits:
    """All risk thresholds as tunable parameters (plan §12, §13).

    - ``budget_*``: per-trade risk budget as a fraction of NAV per signal
      strength tier (plan §12.2).
    - ``abs_max_trade_risk``: absolute per-trade ceiling that NO tier and no
      confidence score may override (plan §12.2).
    - ``single_name_risk`` / ``single_name_capital``: per-underlying caps on
      strategy risk and deployed capital (plan §12.3).
    - ``bucket_risk``: combined risk cap for a correlation bucket (plan §12.4).
    - ``heat_elevated`` / ``heat_high`` / ``heat_reject``: portfolio heat
      state boundaries at 4% / 6% / 8% of NAV (plan §12.5); at or above
      ``heat_reject`` no NEW risk is accepted.
    - ``strength_*``: |directional_edge| thresholds mapping the edge to a
      strength tier; below ``strength_weak`` there is no valid signal.
    - ``cash_floors``: regime -> minimum cash fraction of NAV (plan §13).
    - ``correlation_buckets``: bucket name -> member tickers (plan §12.4).
    - ``max_delta_notional_pct_nav`` / ``max_net_theta_pct_nav`` /
      ``max_net_vega_pct_nav``: portfolio greek limits (plan §16), as
      fractions of NAV, checked post-trade by :func:`assess` when the caller
      supplies portfolio and candidate greeks. Delta: |delta-adjusted
      notional| <= 150% of NAV. Theta: |net theta| per day <= 0.1% of NAV.
      Vega: |net vega| per IV point <= 1% of NAV.
    """

    # Per-tier risk budgets, fraction of NAV (plan §12.2).
    budget_weak: float = 0.005
    budget_moderate: float = 0.0075
    budget_strong: float = 0.01
    budget_very_strong: float = 0.0125
    abs_max_trade_risk: float = 0.015
    # Per-underlying caps (plan §12.3).
    single_name_risk: float = 0.015
    single_name_capital: float = 0.20
    # Correlation bucket cap (plan §12.4).
    bucket_risk: float = 0.03
    # Portfolio heat boundaries (plan §12.5).
    heat_elevated: float = 0.04
    heat_high: float = 0.06
    heat_reject: float = 0.08
    # |edge| -> strength tier thresholds (plan §12.2).
    strength_weak: float = 25.0
    strength_moderate: float = 40.0
    strength_strong: float = 60.0
    strength_very_strong: float = 80.0
    # Regime cash floors (plan §13) and correlation buckets (plan §12.4).
    cash_floors: Mapping[MarketRegime, float] = field(
        default_factory=_default_cash_floors
    )
    correlation_buckets: Mapping[str, tuple[str, ...]] = field(
        default_factory=_default_correlation_buckets
    )
    # Portfolio greek limits as fractions of NAV (plan §16).
    max_delta_notional_pct_nav: float = 1.50
    max_net_theta_pct_nav: float = 0.001
    max_net_vega_pct_nav: float = 0.01


@dataclass
class PositionRisk:
    """One open position's contribution to portfolio risk (plan §12.5).

    ``max_loss`` is the position's defined maximum loss (stop-based for
    stock, premium/width-based for options) — the unit portfolio heat is
    measured in.
    """

    ticker: str
    market_value: float
    max_loss: float


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state the risk engine decides against (§17).

    The strategy engine never builds this; the caller (gateway) does, so the
    risk engine stays independent (plan §17). ``trading_enabled`` mirrors the
    kill switch SystemState (plan §18).
    """

    nav: float
    cash: float
    positions: list[PositionRisk]
    regime: MarketRegime
    trading_enabled: bool


@dataclass
class RiskRequest:
    """A proposed entry the risk engine must approve, resize, or reject.

    ``edge`` is the directional_edge computed UPSTREAM by the signal engine
    (plan §6.2); the risk engine only maps |edge| to a budget tier — it never
    computes signals (plan §17). ``stop_distance`` is the per-share risk
    (entry minus stop for long stock; max loss per contract-share for
    defined-risk options).
    """

    ticker: str
    entry_price: float
    stop_distance: float
    edge: float
    quantity_requested: int | None = None


@dataclass
class RiskAssessment:
    """Full, explainable output of one risk decision (plan §12, §36).

    ``reason_codes`` are machine-readable (``KILL_SWITCH_ACTIVE``,
    ``HEAT_LIMIT``, ``RESIZED_BY_<CAP>`` ...); ``explanations`` are the same
    facts as human-readable sentences with the real numbers (plan §36 style).
    """

    decision: RiskDecision
    approved_quantity: int
    signal_strength: str | None
    risk_budget_pct: float | None
    trade_risk_usd: float
    reason_codes: list[str]
    explanations: list[str]
    heat_before_pct: float
    heat_after_pct: float
    cash_after_pct: float | None


def portfolio_heat(positions: Sequence[PositionRisk], nav: float) -> float:
    """Total open max-loss as a fraction of NAV (plan §12.5)."""
    if nav <= 0:
        raise ValueError(f"nav must be > 0, got {nav}")
    return sum(p.max_loss for p in positions) / nav


def heat_state(heat: float, limits: RiskLimits = RiskLimits()) -> str:
    """Portfolio heat state at the 4% / 6% / 8% boundaries (plan §12.5).

    NORMAL < ``heat_elevated`` <= ELEVATED < ``heat_high`` <= HIGH <
    ``heat_reject`` <= BLOCKED.
    """
    if heat < limits.heat_elevated:
        return "NORMAL"
    if heat < limits.heat_high:
        return "ELEVATED"
    if heat < limits.heat_reject:
        return "HIGH"
    return "BLOCKED"


def strength_tier(edge: float, limits: RiskLimits = RiskLimits()) -> str | None:
    """Map a directional edge to its signal-strength tier name (plan §12.2).

    The SINGLE source of truth for the |edge| -> tier mapping: :func:`assess`
    budgets by it, and the §8 instrument matrix
    (:mod:`libs.trading_core.strategies`) keys its strength column on the
    names returned here. The sign of ``edge`` carries direction, not
    strength, so |edge| is used. Below ``strength_weak`` there is no valid
    signal -> ``None`` (honest null, never a fake "WEAK").
    """
    abs_edge = abs(edge)
    if abs_edge >= limits.strength_very_strong:
        return "VERY_STRONG"
    if abs_edge >= limits.strength_strong:
        return "STRONG"
    if abs_edge >= limits.strength_moderate:
        return "MODERATE"
    if abs_edge >= limits.strength_weak:
        return "WEAK"
    return None


def _tier_budget(strength: str, limits: RiskLimits) -> float:
    """Risk budget (fraction of NAV) for a :func:`strength_tier` name (§12.2)."""
    return {
        "VERY_STRONG": limits.budget_very_strong,
        "STRONG": limits.budget_strong,
        "MODERATE": limits.budget_moderate,
        "WEAK": limits.budget_weak,
    }[strength]


def _floor_qty(numerator: float, denominator: float) -> int:
    """floor(numerator/denominator) with float-representation tolerance, >= 0."""
    return max(math.floor(numerator / denominator + _EPS), 0)


def _early_reject(
    codes: list[str], explanations: list[str], heat_before: float
) -> RiskAssessment:
    """Reject before sizing (kill switch / heat gate / weak signal)."""
    return RiskAssessment(
        decision=RiskDecision.REJECT,
        approved_quantity=0,
        signal_strength=None,
        risk_budget_pct=None,
        trade_risk_usd=0.0,
        reason_codes=codes,
        explanations=explanations,
        heat_before_pct=heat_before,
        heat_after_pct=heat_before,
        cash_after_pct=None,
    )


def assess(
    request: RiskRequest,
    snapshot: PortfolioSnapshot,
    limits: RiskLimits = RiskLimits(),
    *,
    budget_multiplier: float = 1.0,
    portfolio_greeks: PortfolioGreeks | None = None,
    new_position_greeks: PositionGreeksInput | None = None,
) -> RiskAssessment:
    """Approve, resize, or reject a proposed entry (plan §12, §13, §17).

    Deterministic pipeline; risk limits have PRIORITY over strategy
    confidence (plan §44 rule 20). Steps and citations are inline below.

    Optional inputs (all defaults leave behavior EXACTLY as before):

    - ``budget_multiplier`` (plan §14, vol targeting): scales the tier
      budget BEFORE the absolute cap — ``effective_budget = min(tier_budget
      * budget_multiplier, abs_max_trade_risk)`` — so vol targeting can
      never raise a hard cap (§14: hard caps are NEVER overridden). Must be
      > 0; 1.0 means no adjustment.
    - ``portfolio_greeks`` + ``new_position_greeks`` (plan §16): when BOTH
      are supplied, the post-trade net delta notional / theta / vega are
      checked against the ``max_*_pct_nav`` limits AFTER the sizing clamps,
      using the APPROVED quantity (step 5f below). Supplying only one of
      the two runs no greek checks (the post-trade book would be unknowable
      — honest skip, not a guess).
    """
    if budget_multiplier <= 0:
        raise ValueError(
            f"budget_multiplier must be > 0, got {budget_multiplier}"
        )
    nav = snapshot.nav
    heat_before = portfolio_heat(snapshot.positions, nav)  # validates nav > 0

    # ------------------------------------------------------------------
    # Step 1 — kill switch (plan §18): trading disabled rejects ALL new
    # risk, no matter how strong the signal (plan §44 rule 20).
    # ------------------------------------------------------------------
    if not snapshot.trading_enabled:
        return _early_reject(
            ["KILL_SWITCH_ACTIVE"],
            [
                "Kill switch is active: trading is disabled, so the "
                f"{request.ticker} entry is rejected regardless of signal "
                "strength."
            ],
            heat_before,
        )

    # ------------------------------------------------------------------
    # Step 2 — portfolio heat gate (plan §12.5): heat at or above the
    # reject threshold blocks NEW risk; existing positions are untouched.
    # ------------------------------------------------------------------
    if heat_before >= limits.heat_reject:
        return _early_reject(
            ["HEAT_LIMIT"],
            [
                f"Portfolio heat is {heat_before:.2%}, at or above the "
                f"{limits.heat_reject:.2%} reject threshold; no new risk may "
                "be added."
            ],
            heat_before,
        )

    # ------------------------------------------------------------------
    # Step 3 — signal strength tier -> risk budget (plan §12.2). Below the
    # weak threshold there is no valid signal. The budget NEVER exceeds
    # abs_max_trade_risk: "No confidence score may override" (plan §12.2).
    # The §14 vol-targeting multiplier scales the tier budget BEFORE the
    # absolute cap, so it can shrink risk freely but can never push the
    # effective budget above abs_max_trade_risk (§14 never overrides hard
    # caps; with the default multiplier 1.0 this line is unchanged).
    # ------------------------------------------------------------------
    abs_edge = abs(request.edge)
    strength = strength_tier(request.edge, limits)
    if strength is None:
        return _early_reject(
            ["SIGNAL_TOO_WEAK"],
            [
                f"|directional_edge| {abs_edge:.1f} is below the weak-signal "
                f"threshold {limits.strength_weak:.1f}; no valid signal, no "
                "trade."
            ],
            heat_before,
        )
    budget = min(
        _tier_budget(strength, limits) * budget_multiplier,
        limits.abs_max_trade_risk,
    )

    # ------------------------------------------------------------------
    # Step 4 — base sizing (plan §12.1): shares = floor(allowed risk /
    # per-share stop distance), never more than the caller requested.
    # ------------------------------------------------------------------
    if request.stop_distance <= 0:
        raise ValueError(
            f"stop_distance must be > 0, got {request.stop_distance}"
        )
    allowed_risk = nav * budget
    raw_budget_qty = _floor_qty(allowed_risk, request.stop_distance)
    base_qty = raw_budget_qty
    if request.quantity_requested is not None:
        base_qty = min(base_qty, request.quantity_requested)
    base_qty = max(base_qty, 0)
    qty = base_qty

    reason_codes: list[str] = []
    explanations: list[str] = []

    def clamp(current: int, cap_qty: int, code: str, sentence: str) -> int:
        """Apply one cap as a quantity clamp, recording WHY when it binds.

        A cap that reduces the quantity appends ``RESIZED_BY_<code>``; a cap
        that forces zero appends its bare code (the eventual REJECT reason).
        """
        cap_qty = max(cap_qty, 0)
        if cap_qty < current:
            reason_codes.append(code if cap_qty == 0 else f"RESIZED_BY_{code}")
            explanations.append(sentence)
            return cap_qty
        return current

    same_ml = sum(
        p.max_loss for p in snapshot.positions if p.ticker == request.ticker
    )
    same_mv = sum(
        p.market_value for p in snapshot.positions if p.ticker == request.ticker
    )
    total_ml = sum(p.max_loss for p in snapshot.positions)
    sd = request.stop_distance
    entry = request.entry_price

    # ------------------------------------------------------------------
    # Step 5a — single-name strategy risk (plan §12.3): new risk plus the
    # existing same-ticker max loss stays within nav * single_name_risk.
    # ------------------------------------------------------------------
    cap_a = _floor_qty(nav * limits.single_name_risk - same_ml, sd)
    qty = clamp(
        qty,
        cap_a,
        "SINGLE_NAME_RISK_CAP",
        f"{request.ticker} strategy risk would rise from {same_ml / nav:.2%} "
        f"to {(same_ml + qty * sd) / nav:.2%} of NAV, above the "
        f"{limits.single_name_risk:.2%} single-name limit; quantity reduced "
        f"from {qty} to {max(cap_a, 0)}.",
    )

    # ------------------------------------------------------------------
    # Step 5b — single-name capital (plan §12.3): deployed capital plus the
    # existing same-ticker market value stays within nav * single_name_capital.
    # ------------------------------------------------------------------
    cap_b = _floor_qty(nav * limits.single_name_capital - same_mv, entry)
    qty = clamp(
        qty,
        cap_b,
        "SINGLE_NAME_CAPITAL_CAP",
        f"{request.ticker} capital would rise from ${same_mv:,.2f} to "
        f"${same_mv + qty * entry:,.2f}, above the "
        f"${nav * limits.single_name_capital:,.2f} "
        f"({limits.single_name_capital:.0%} of NAV) single-name capital cap; "
        f"quantity reduced from {qty} to {max(cap_b, 0)}.",
    )

    # ------------------------------------------------------------------
    # Step 5c — correlation bucket (plan §12.4): existing bucket max loss
    # plus the new risk stays within nav * bucket_risk.
    # ------------------------------------------------------------------
    for bucket_name, members in limits.correlation_buckets.items():
        if request.ticker not in members:
            continue
        bucket_ml = sum(
            p.max_loss for p in snapshot.positions if p.ticker in members
        )
        cap_c = _floor_qty(nav * limits.bucket_risk - bucket_ml, sd)
        qty = clamp(
            qty,
            cap_c,
            f"BUCKET_LIMIT_{bucket_name}",
            f"{bucket_name} bucket risk would rise from {bucket_ml / nav:.2%} "
            f"to {(bucket_ml + qty * sd) / nav:.2%} of NAV, above the "
            f"{limits.bucket_risk:.2%} bucket limit; quantity reduced from "
            f"{qty} to {max(cap_c, 0)}.",
        )

    # ------------------------------------------------------------------
    # Step 5d — heat headroom (plan §12.5): the trade must leave portfolio
    # heat STRICTLY below the reject threshold.
    # ------------------------------------------------------------------
    cap_d = _floor_qty(nav * limits.heat_reject - total_ml, sd)
    while cap_d > 0 and (total_ml + cap_d * sd) / nav >= limits.heat_reject:
        cap_d -= 1
    qty = clamp(
        qty,
        cap_d,
        "HEAT_LIMIT",
        f"Portfolio heat would rise from {heat_before:.2%} to "
        f"{(total_ml + qty * sd) / nav:.2%}, reaching the "
        f"{limits.heat_reject:.2%} reject threshold; quantity reduced from "
        f"{qty} to {max(cap_d, 0)}.",
    )

    # ------------------------------------------------------------------
    # Step 5e — regime cash floor (plan §13): cash after the purchase must
    # stay at or above cash_floors[regime] of NAV.
    # ------------------------------------------------------------------
    floor_pct = limits.cash_floors[snapshot.regime]
    cap_e = _floor_qty(snapshot.cash - nav * floor_pct, entry)
    qty = clamp(
        qty,
        cap_e,
        "CASH_FLOOR",
        f"Cash would fall from {snapshot.cash / nav:.2%} to "
        f"{(snapshot.cash - qty * entry) / nav:.2%} of NAV, below the "
        f"{floor_pct:.0%} cash floor for regime {snapshot.regime.value}; "
        f"quantity reduced from {qty} to {max(cap_e, 0)}.",
    )

    # ------------------------------------------------------------------
    # Step 5f — portfolio greek limits (plan §16), only when the caller
    # supplied BOTH the current book's greeks and the candidate's per-share
    # greeks. The checks run AFTER the sizing clamps: the candidate's
    # contribution is its PER-SHARE greeks scaled by the APPROVED quantity
    # (qty * multiplier * greek), which is exactly the candidate's
    # full-position greeks scaled by approved qty / requested basis — the
    # ``quantity`` field on ``new_position_greeks`` is the requested basis
    # and is deliberately not used here. A breach REJECTS outright (plan
    # §44 rule 20: limits outrank everything); exactly at a limit passes,
    # strictly above it rejects.
    # ------------------------------------------------------------------
    if portfolio_greeks is not None and new_position_greeks is not None and qty > 0:
        g = new_position_greeks
        scale = qty * g.multiplier
        post_delta_notional = (
            portfolio_greeks.delta_adjusted_notional + scale * g.delta * g.spot
        )
        post_theta = (
            portfolio_greeks.net_theta_per_day + scale * g.theta_per_day
        )
        post_vega = portfolio_greeks.net_vega + scale * g.vega
        breaches: list[tuple[str, str]] = []
        delta_cap = nav * limits.max_delta_notional_pct_nav
        if abs(post_delta_notional) > delta_cap:
            breaches.append((
                "PORTFOLIO_DELTA_LIMIT",
                f"Post-trade |delta-adjusted notional| would be "
                f"${abs(post_delta_notional):,.2f} "
                f"({abs(post_delta_notional) / nav:.2%} of NAV), above the "
                f"{limits.max_delta_notional_pct_nav:.2%}-of-NAV limit "
                f"(${delta_cap:,.2f}); {request.ticker} entry rejected.",
            ))
        theta_cap = nav * limits.max_net_theta_pct_nav
        if abs(post_theta) > theta_cap:
            breaches.append((
                "PORTFOLIO_THETA_LIMIT",
                f"Post-trade |net theta| would be ${abs(post_theta):,.2f}/day "
                f"({abs(post_theta) / nav:.4%} of NAV), above the "
                f"{limits.max_net_theta_pct_nav:.4%}-of-NAV limit "
                f"(${theta_cap:,.2f}/day); {request.ticker} entry rejected.",
            ))
        vega_cap = nav * limits.max_net_vega_pct_nav
        if abs(post_vega) > vega_cap:
            breaches.append((
                "PORTFOLIO_VEGA_LIMIT",
                f"Post-trade |net vega| would be ${abs(post_vega):,.2f} per "
                f"IV point ({abs(post_vega) / nav:.2%} of NAV), above the "
                f"{limits.max_net_vega_pct_nav:.2%}-of-NAV limit "
                f"(${vega_cap:,.2f}); {request.ticker} entry rejected.",
            ))
        if breaches:
            for code, sentence in breaches:
                reason_codes.append(code)
                explanations.append(sentence)
            qty = 0  # step 6 turns this into a REJECT with these reasons

    # ------------------------------------------------------------------
    # Step 6 — decision (plan §12): zero -> REJECT, reduced -> RESIZE,
    # otherwise APPROVE. Every REJECT carries at least one reason code
    # (risk-invariant, plan §42).
    # ------------------------------------------------------------------
    if qty <= 0:
        qty = 0
        if not reason_codes:
            if raw_budget_qty <= 0:
                reason_codes.append("BUDGET_TOO_SMALL")
                explanations.append(
                    f"Risk budget ${allowed_risk:,.2f} ({budget:.2%} of NAV) "
                    f"buys zero shares at ${sd:,.2f} risk per share."
                )
            else:
                reason_codes.append("ZERO_QUANTITY_REQUESTED")
                explanations.append(
                    f"Requested quantity {request.quantity_requested} leaves "
                    "nothing to approve."
                )
        decision = RiskDecision.REJECT
    elif qty < base_qty:
        decision = RiskDecision.APPROVE_WITH_RESIZE
    else:
        decision = RiskDecision.APPROVE

    trade_risk = qty * sd
    if decision is not RiskDecision.REJECT:
        explanations.append(
            f"Approved {qty} shares of {request.ticker} risking "
            f"${trade_risk:,.2f} ({trade_risk / nav:.2%} of NAV) on a "
            f"{strength} signal (|edge| {abs_edge:.1f})."
        )

    return RiskAssessment(
        decision=decision,
        approved_quantity=qty,
        signal_strength=strength,
        risk_budget_pct=budget,
        trade_risk_usd=trade_risk,
        reason_codes=reason_codes,
        explanations=explanations,
        heat_before_pct=heat_before,
        heat_after_pct=(total_ml + qty * sd) / nav,
        cash_after_pct=(snapshot.cash - qty * entry) / nav,
    )
