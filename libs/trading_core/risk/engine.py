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
from typing import Protocol

from libs.trading_core.greeks import PortfolioGreeks, PositionGreeksInput
from libs.trading_core.models import MarketRegime, RiskDecision

# Tolerance used when flooring float ratios into share counts, so that
# exactly-at-the-limit quantities (e.g. 20000.0 / 20.0) are not lost to
# float representation error.
_EPS = 1e-9

#: Stop distance for a LONG_STOCK entry = ATR_STOP_MULTIPLE * ATR14 (plan
#: §12.1). SINGLE SOURCE OF TRUTH: the live §10 gate chain (routers/orders)
#: and the backtest engine both import THIS constant, so live sizing and
#: replayed sizing can never drift apart (user parity mandate 2026-08-16).
ATR_STOP_MULTIPLE = 2.0


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


#: The layer every PRE-EXISTING reason code belongs to (Phase C contract
#: §7.3). Kill switch, heat gate, single-name / bucket / heat caps, the cash
#: floor and the greek limits are all TIER 0 HARD LIMITS — the rules no
#: statistical model may soften and no confidence score may override (plan
#: §44 rule 20; spec §38 "Risk Engine is sovereign"). Phase C caps carry
#: their own layer (``STATISTICAL`` / ``CONCENTRATION``) on the cap itself,
#: which is what lets the UI tell "resized by a hard cap" apart from
#: "resized by a shadow-promoted statistical limit" (spec §47).
LAYER_HARD_LIMIT = "HARD_LIMIT"

#: Prefix the clamp helpers prepend to a cap's code when it REDUCES rather
#: than zeroes the quantity. ``binding_constraints`` strips it so both forms
#: of one cap map to the same layer.
RESIZED_PREFIX = "RESIZED_BY_"


class ExtraCap(Protocol):
    """Structural shape of an additional quantity cap ``assess`` can apply
    (Phase C contract §7.3).

    :class:`~libs.trading_core.risk.pretrade.QuantityCap` satisfies it. The
    shape is STRUCTURAL on purpose: this Tier 0 module must not import the
    statistical library — the dependency runs one way (the statistical
    layer reads Tier 0 decisions, never the reverse), and a Protocol keeps
    ``engine.py`` importable with nothing but ``greeks`` and ``models``
    behind it.

    - ``cap_qty``: the largest quantity this cap allows (``0`` ⇒ REJECT);
    - ``code``: the reason-code stem (``RESIZED_BY_<code>`` when it reduces);
    - ``layer``: the layer reported in ``binding_constraints``;
    - ``sentence``: the §36/§47-style explanation with the real numbers.
    """

    cap_qty: int
    code: str
    layer: str
    sentence: str


@dataclass(frozen=True)
class BindingConstraint:
    """One reason a decision came out the way it did, with its LAYER
    (Phase C contract §7.3; spec §47 "Binding constraints: 1. 99% ES
    contribution, 2. Technology risk concentration ...").

    ``code`` is the reason code as recorded (the bare cap code, or the
    ``RESIZED_BY_<CAP>`` form when the cap only reduced the quantity);
    ``layer`` says which layer owns it — ``"HARD_LIMIT"`` for every Tier 0
    rule, or the ``layer`` the extra cap declared.
    """

    code: str
    layer: str


def _binding_constraints(
    reason_codes: Sequence[str], extra_layers: Mapping[str, str] | None = None
) -> tuple[BindingConstraint, ...]:
    """Map reason codes to ``(code, layer)`` pairs — TOTAL, pure, no
    behaviour change (Phase C contract §7.3).

    Total by construction: every code the engine can emit resolves to
    ``LAYER_HARD_LIMIT`` unless an extra cap declared a layer for it, so a
    code added to the engine later can never produce a KeyError or a
    silently missing row. ``RESIZED_BY_<CAP>`` and the bare ``<CAP>`` are
    the same constraint at different severities and resolve to the same
    layer. Order is the reason-code order — the order the pipeline
    encountered them.
    """
    layers = dict(extra_layers or {})
    out: list[BindingConstraint] = []
    for code in reason_codes:
        stem = code[len(RESIZED_PREFIX):] if code.startswith(RESIZED_PREFIX) else code
        out.append(BindingConstraint(code=code, layer=layers.get(stem, LAYER_HARD_LIMIT)))
    return tuple(out)


@dataclass
class RiskAssessment:
    """Full, explainable output of one risk decision (plan §12, §36).

    ``reason_codes`` are machine-readable (``KILL_SWITCH_ACTIVE``,
    ``HEAT_LIMIT``, ``RESIZED_BY_<CAP>`` ...); ``explanations`` are the same
    facts as human-readable sentences with the real numbers (plan §36 style).

    Phase C adds two OPTIONAL fields with defaults, so every existing
    construction site and every existing test is untouched (contract §7.3):

    - ``requested_quantity``: what the caller asked for, next to
      ``approved_quantity`` — the two numbers spec §47's explanation needs
      side by side ("Requested: 4 contracts / Approved: 2 contracts");
    - ``binding_constraints``: the reason codes paired with the LAYER that
      owns each, populated for EVERY decision (both pipelines) by the pure
      mapping :func:`_binding_constraints`. It is a re-presentation of
      ``reason_codes``, never a second source of truth.
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
    requested_quantity: int | None = None
    binding_constraints: tuple[BindingConstraint, ...] = ()


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
    codes: list[str],
    explanations: list[str],
    heat_before: float,
    requested_quantity: int | None = None,
) -> RiskAssessment:
    """Reject before sizing (kill switch / heat gate / weak signal).

    An early reject is always a Tier 0 hard limit, so its
    ``binding_constraints`` are the codes at ``LAYER_HARD_LIMIT`` (Phase C
    contract §7.3); the decision itself is unchanged.
    """
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
        requested_quantity=requested_quantity,
        binding_constraints=_binding_constraints(codes),
    )


def assess(
    request: RiskRequest,
    snapshot: PortfolioSnapshot,
    limits: RiskLimits = RiskLimits(),
    *,
    budget_multiplier: float = 1.0,
    portfolio_greeks: PortfolioGreeks | None = None,
    new_position_greeks: PositionGreeksInput | None = None,
    extra_caps: Sequence[ExtraCap] = (),
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
    - ``extra_caps`` (Phase C contract §7.3, spec §37): additional quantity
      caps — anything with ``.cap_qty``, ``.code``, ``.layer`` and
      ``.sentence``, i.e. a :class:`~libs.trading_core.risk.pretrade
      .QuantityCap` — applied through the SAME ``clamp`` closure as every
      Tier 0 cap, AFTER step 5e (cash floor) and BEFORE step 5f (greeks),
      in the order given. They can only REDUCE the quantity, exactly like
      the hard caps, and each records ``RESIZED_BY_<code>`` (or the bare
      ``code`` at zero) plus its own ``layer`` in ``binding_constraints``.
      **The default ``()`` leaves this function byte-identical** — a
      statistical cap binds only when a caller passes it in deliberately,
      which is the explicit SHADOW → PRODUCTION promotion step (spec §70).
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
            request.quantity_requested,
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
            request.quantity_requested,
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
            request.quantity_requested,
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
    # Step 5e' — caller-supplied extra caps (Phase C contract §7.3, spec
    # §37 "Final Risk = minimum of: Candidate Risk, Single Name Cap,
    # Portfolio Heat Headroom, Cash Constraint, ..., ES Limit, Risk
    # Contribution Limit"). They run AFTER the cash floor and BEFORE the
    # greek limits, through the SAME clamp closure, so a statistical cap is
    # recorded exactly like a hard cap and can only ever reduce. With the
    # default ``extra_caps=()`` this loop does nothing at all — the
    # byte-identity guarantee (contract §7.3).
    # ------------------------------------------------------------------
    extra_layers: dict[str, str] = {}
    for cap in extra_caps:
        extra_layers[cap.code] = cap.layer
        qty = clamp(qty, cap.cap_qty, cap.code, cap.sentence)

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
        requested_quantity=request.quantity_requested,
        binding_constraints=_binding_constraints(reason_codes, extra_layers),
    )


# ---------------------------------------------------------------------------
# Income opens — covered call / cash-secured put (risk-engine audit §8 item 3,
# §10 Phase B0; spec risk_engine.md §2, §72 "the user must not accidentally
# bypass hard limits"). ADDITIVE: :func:`assess` above is untouched and
# byte-identical in behaviour; income opens have no signal edge and no
# stop-based sizing, so they get their own pipeline that shares the same
# limits, helpers, reason-code vocabulary and explanation style.
# ---------------------------------------------------------------------------

#: The two collateralized short-premium instruments this pipeline sizes.
INCOME_INSTRUMENTS: tuple[str, ...] = ("COVERED_CALL", "CASH_SECURED_PUT")


@dataclass
class IncomeRiskRequest:
    """A proposed income OPEN (short premium) the risk engine must approve,
    resize, or reject (audit §10 Phase B0).

    - ``instrument``: ``COVERED_CALL`` or ``CASH_SECURED_PUT``.
    - ``contracts``: the requested number of contracts (the base quantity —
      there is no edge tier and no stop-based sizing for income opens).
    - ``risk_per_contract``: the RISK basis in $ per contract, the unit
      portfolio heat / single-name risk / bucket risk are measured in. CSP:
      ``(strike − credit) × 100`` (stock to zero, the row's ``max_loss``
      basis). Covered call: ``0.0`` — the stock row already carries the heat
      and the short call adds no incremental defined loss.
    - ``capital_per_contract``: the CAPITAL basis in $ per contract, the
      unit single-name capital and the cash floor are measured in. CSP:
      ``strike × 100`` (the cash reservation). Covered call: ``0.0`` — the
      shares are already deployed capital and are pinned, not bought.

    A ZERO basis means that clamp cannot bind (there is nothing to divide
    by and nothing to add) — it is skipped, never divided by.
    """

    ticker: str
    instrument: str
    contracts: int
    risk_per_contract: float
    capital_per_contract: float


def _greek_breaches(
    ticker: str,
    nav: float,
    qty: int,
    limits: RiskLimits,
    portfolio_greeks: PortfolioGreeks,
    new_position_greeks: PositionGreeksInput,
) -> list[tuple[str, str]]:
    """Post-trade portfolio greek limit breaches at ``qty`` (plan §16).

    The same arithmetic and sentences as step 5f of :func:`assess`: the
    candidate's PER-SHARE greeks scaled by ``qty * multiplier`` are added to
    the current book and each |exposure| is compared with its
    ``max_*_pct_nav`` cap; exactly at a limit passes, strictly above it
    breaches. Returns ``(reason_code, explanation)`` pairs, empty when clean.
    """
    g = new_position_greeks
    scale = qty * g.multiplier
    post_delta_notional = (
        portfolio_greeks.delta_adjusted_notional + scale * g.delta * g.spot
    )
    post_theta = portfolio_greeks.net_theta_per_day + scale * g.theta_per_day
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
            f"(${delta_cap:,.2f}); {ticker} entry rejected.",
        ))
    theta_cap = nav * limits.max_net_theta_pct_nav
    if abs(post_theta) > theta_cap:
        breaches.append((
            "PORTFOLIO_THETA_LIMIT",
            f"Post-trade |net theta| would be ${abs(post_theta):,.2f}/day "
            f"({abs(post_theta) / nav:.4%} of NAV), above the "
            f"{limits.max_net_theta_pct_nav:.4%}-of-NAV limit "
            f"(${theta_cap:,.2f}/day); {ticker} entry rejected.",
        ))
    vega_cap = nav * limits.max_net_vega_pct_nav
    if abs(post_vega) > vega_cap:
        breaches.append((
            "PORTFOLIO_VEGA_LIMIT",
            f"Post-trade |net vega| would be ${abs(post_vega):,.2f} per "
            f"IV point ({abs(post_vega) / nav:.2%} of NAV), above the "
            f"{limits.max_net_vega_pct_nav:.2%}-of-NAV limit "
            f"(${vega_cap:,.2f}); {ticker} entry rejected.",
        ))
    return breaches


def assess_income(
    request: IncomeRiskRequest,
    snapshot: PortfolioSnapshot,
    limits: RiskLimits = RiskLimits(),
    *,
    portfolio_greeks: PortfolioGreeks | None = None,
    new_position_greeks: PositionGreeksInput | None = None,
) -> RiskAssessment:
    """Approve, resize, or reject a covered-call / cash-secured-put OPEN
    (audit §8 item 3, §10 Phase B0; spec §2, §72).

    Deterministic; risk limits have PRIORITY over everything (plan §44 rule
    20). The pipeline mirrors :func:`assess` minus the edge tier and the
    stop-based sizing (income opens carry no directional signal, so
    ``signal_strength`` / ``risk_budget_pct`` are honest ``None``):

    1. Kill switch (plan §18) — ``KILL_SWITCH_ACTIVE``.
    2. Portfolio heat gate (plan §12.5) — ``HEAT_LIMIT``.
    3. Base quantity = ``request.contracts``.
    4. Absolute per-trade risk ceiling (plan §12.2 ``abs_max_trade_risk``,
       which no tier and no confidence may override): ``contracts *
       risk_per_contract <= nav * abs_max_trade_risk`` —
       ``ABS_TRADE_RISK_CAP`` / ``RESIZED_BY_ABS_TRADE_RISK_CAP``.
    5a. Single-name risk (plan §12.3, risk basis).
    5b. Single-name capital (plan §12.3, capital basis).
    5c. Correlation bucket (plan §12.4, risk basis).
    5d. Heat headroom strictly below ``heat_reject`` (plan §12.5, risk basis).
    5e. Regime cash floor (plan §13, capital basis = the cash outlay /
        reservation).
    5f. Portfolio greek limits at the approved quantity (plan §16) exactly
        like :func:`assess` — REJECT on breach; the CALLER passes the short
        leg's greeks already NEGATED (:mod:`libs.trading_core.greeks`
        conventions). Supplying only one of the two greek inputs runs no
        greek check (honest skip).
    6. Decision APPROVE / APPROVE_WITH_RESIZE / REJECT with the same
       reason-code vocabulary and real-number explanations (plan §36).

    A zero basis (covered call: risk 0 / capital 0 — the stock row already
    carries heat and the pinned shares are already deployed) means the
    corresponding clamps CANNOT bind and are skipped; nothing divides by
    zero. For a covered call only the kill switch, the heat gate and the
    greek limits can therefore bind.
    """
    if request.instrument not in INCOME_INSTRUMENTS:
        raise ValueError(
            f"instrument must be one of {INCOME_INSTRUMENTS}, "
            f"got {request.instrument!r}"
        )
    if isinstance(request.contracts, bool) or not isinstance(request.contracts, int):
        raise ValueError(
            f"contracts must be an int, got {type(request.contracts).__name__} "
            f"{request.contracts!r}"
        )
    if request.contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {request.contracts}")
    # Bases must be finite and >= 0: a NaN basis would compare False against
    # every clamp and silently skip them all (QA finding 2026-08-17), so it
    # is refused up front like a negative one — never sized around.
    if not math.isfinite(request.risk_per_contract) or request.risk_per_contract < 0:
        raise ValueError(
            "risk_per_contract must be finite and >= 0, got "
            f"{request.risk_per_contract}"
        )
    if (
        not math.isfinite(request.capital_per_contract)
        or request.capital_per_contract < 0
    ):
        raise ValueError(
            "capital_per_contract must be finite and >= 0, got "
            f"{request.capital_per_contract}"
        )
    nav = snapshot.nav
    heat_before = portfolio_heat(snapshot.positions, nav)  # validates nav > 0
    label = request.instrument.replace("_", " ").lower()

    # Step 1 — kill switch (plan §18).
    if not snapshot.trading_enabled:
        return _early_reject(
            ["KILL_SWITCH_ACTIVE"],
            [
                "Kill switch is active: trading is disabled, so the "
                f"{request.ticker} {label} open is rejected."
            ],
            heat_before,
            request.contracts,
        )

    # Step 2 — portfolio heat gate (plan §12.5).
    if heat_before >= limits.heat_reject:
        return _early_reject(
            ["HEAT_LIMIT"],
            [
                f"Portfolio heat is {heat_before:.2%}, at or above the "
                f"{limits.heat_reject:.2%} reject threshold; no new risk may "
                "be added."
            ],
            heat_before,
            request.contracts,
        )

    # Step 3 — base quantity: the requested contracts (no edge tier).
    base_qty = max(request.contracts, 0)
    qty = base_qty
    rp = request.risk_per_contract  # risk basis, $ per contract
    cp = request.capital_per_contract  # capital basis, $ per contract

    reason_codes: list[str] = []
    explanations: list[str] = []

    def clamp(current: int, cap_qty: int, code: str, sentence: str) -> int:
        """Apply one cap as a quantity clamp, recording WHY when it binds
        (``RESIZED_BY_<code>`` when reduced, the bare code when zeroed)."""
        cap_qty = max(cap_qty, 0)
        if cap_qty < current:
            reason_codes.append(code if cap_qty == 0 else f"RESIZED_BY_{code}")
            explanations.append(sentence)
            return cap_qty
        return current

    same_ml = math.fsum(
        p.max_loss for p in snapshot.positions if p.ticker == request.ticker
    )
    same_mv = math.fsum(
        p.market_value for p in snapshot.positions if p.ticker == request.ticker
    )
    total_ml = math.fsum(p.max_loss for p in snapshot.positions)

    # Step 4 — absolute per-trade risk ceiling (plan §12.2): the whole open
    # may never risk more than nav * abs_max_trade_risk.
    if rp > 0:
        cap_abs = _floor_qty(nav * limits.abs_max_trade_risk, rp)
        qty = clamp(
            qty,
            cap_abs,
            "ABS_TRADE_RISK_CAP",
            f"{request.ticker} {label} would risk ${qty * rp:,.2f} "
            f"({qty * rp / nav:.2%} of NAV), above the absolute per-trade "
            f"ceiling {limits.abs_max_trade_risk:.2%} of NAV "
            f"(${nav * limits.abs_max_trade_risk:,.2f}); contracts reduced "
            f"from {qty} to {max(cap_abs, 0)}.",
        )

    # Step 5a — single-name strategy risk (plan §12.3, risk basis).
    if rp > 0:
        cap_a = _floor_qty(nav * limits.single_name_risk - same_ml, rp)
        qty = clamp(
            qty,
            cap_a,
            "SINGLE_NAME_RISK_CAP",
            f"{request.ticker} strategy risk would rise from "
            f"{same_ml / nav:.2%} to {(same_ml + qty * rp) / nav:.2%} of "
            f"NAV, above the {limits.single_name_risk:.2%} single-name "
            f"limit; contracts reduced from {qty} to {max(cap_a, 0)}.",
        )

    # Step 5b — single-name capital (plan §12.3, capital basis).
    if cp > 0:
        cap_b = _floor_qty(nav * limits.single_name_capital - same_mv, cp)
        qty = clamp(
            qty,
            cap_b,
            "SINGLE_NAME_CAPITAL_CAP",
            f"{request.ticker} capital would rise from ${same_mv:,.2f} to "
            f"${same_mv + qty * cp:,.2f}, above the "
            f"${nav * limits.single_name_capital:,.2f} "
            f"({limits.single_name_capital:.0%} of NAV) single-name capital "
            f"cap; contracts reduced from {qty} to {max(cap_b, 0)}.",
        )

    # Step 5c — correlation bucket (plan §12.4, risk basis).
    if rp > 0:
        for bucket_name, members in limits.correlation_buckets.items():
            if request.ticker not in members:
                continue
            bucket_ml = math.fsum(
                p.max_loss for p in snapshot.positions if p.ticker in members
            )
            cap_c = _floor_qty(nav * limits.bucket_risk - bucket_ml, rp)
            qty = clamp(
                qty,
                cap_c,
                f"BUCKET_LIMIT_{bucket_name}",
                f"{bucket_name} bucket risk would rise from "
                f"{bucket_ml / nav:.2%} to {(bucket_ml + qty * rp) / nav:.2%} "
                f"of NAV, above the {limits.bucket_risk:.2%} bucket limit; "
                f"contracts reduced from {qty} to {max(cap_c, 0)}.",
            )

    # Step 5d — heat headroom (plan §12.5, risk basis): heat after the open
    # must stay STRICTLY below the reject threshold.
    if rp > 0:
        cap_d = _floor_qty(nav * limits.heat_reject - total_ml, rp)
        while cap_d > 0 and (total_ml + cap_d * rp) / nav >= limits.heat_reject:
            cap_d -= 1
        qty = clamp(
            qty,
            cap_d,
            "HEAT_LIMIT",
            f"Portfolio heat would rise from {heat_before:.2%} to "
            f"{(total_ml + qty * rp) / nav:.2%}, reaching the "
            f"{limits.heat_reject:.2%} reject threshold; contracts reduced "
            f"from {qty} to {max(cap_d, 0)}.",
        )

    # Step 5e — regime cash floor (plan §13, capital basis): the cash
    # outlay / reservation must leave cash at or above the regime floor.
    floor_pct = limits.cash_floors[snapshot.regime]
    if cp > 0:
        cap_e = _floor_qty(snapshot.cash - nav * floor_pct, cp)
        qty = clamp(
            qty,
            cap_e,
            "CASH_FLOOR",
            f"Cash would fall from {snapshot.cash / nav:.2%} to "
            f"{(snapshot.cash - qty * cp) / nav:.2%} of NAV, below the "
            f"{floor_pct:.0%} cash floor for regime {snapshot.regime.value}; "
            f"contracts reduced from {qty} to {max(cap_e, 0)}.",
        )

    # Step 5f — portfolio greek limits at the APPROVED quantity (plan §16);
    # the caller supplies the short leg's greeks already negated.
    if portfolio_greeks is not None and new_position_greeks is not None and qty > 0:
        breaches = _greek_breaches(
            request.ticker, nav, qty, limits, portfolio_greeks, new_position_greeks
        )
        if breaches:
            for code, sentence in breaches:
                reason_codes.append(code)
                explanations.append(sentence)
            qty = 0  # step 6 turns this into a REJECT with these reasons

    # Step 6 — decision (plan §12); every REJECT carries a reason (plan §42).
    if qty <= 0:
        qty = 0
        if not reason_codes:
            reason_codes.append("ZERO_QUANTITY_REQUESTED")
            explanations.append(
                f"Requested {request.contracts} contract(s) leaves nothing to "
                "approve."
            )
        decision = RiskDecision.REJECT
    elif qty < base_qty:
        decision = RiskDecision.APPROVE_WITH_RESIZE
    else:
        decision = RiskDecision.APPROVE

    trade_risk = qty * rp
    if decision is not RiskDecision.REJECT:
        explanations.append(
            f"Approved {qty} {label} contract(s) on {request.ticker} risking "
            f"${trade_risk:,.2f} ({trade_risk / nav:.2%} of NAV) with "
            f"${qty * cp:,.2f} of capital reserved (no signal tier: income "
            "opens size by contracts, not by edge)."
        )

    return RiskAssessment(
        decision=decision,
        approved_quantity=qty,
        signal_strength=None,
        risk_budget_pct=None,
        trade_risk_usd=trade_risk,
        reason_codes=reason_codes,
        explanations=explanations,
        heat_before_pct=heat_before,
        heat_after_pct=(total_ml + qty * rp) / nav,
        cash_after_pct=(snapshot.cash - qty * cp) / nav,
        requested_quantity=request.contracts,
        binding_constraints=_binding_constraints(reason_codes),
    )
