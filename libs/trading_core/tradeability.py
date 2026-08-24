"""Tradeability layer v1 (upgrade 2026-08-12 §9/§10) — Layer 2 of the
decision architecture.

Pure, deterministic, direction-AGNOSTIC assessment of whether a symbol's
current ENVIRONMENT permits opening new positions. Directional strength must
not equal permission to trade (§9): a STRONG_BULL read with a BLOCKED
tradeability is a valid, explainable state (§10) — "strong bullish
directional evidence exists, but the environment fails the tradeability
gate" — never a contradiction or an error.

This layer answers only "may the environment be traded?"; it never chooses
an instrument, sizes a position, or consults the LLM. Portfolio/risk gates
and Trading Pool authorization remain sovereign and separate (§12/§43) —
TRADEABLE here is necessary, not sufficient, for execution.

Every threshold is a research parameter on :class:`TradeabilityParams`,
versioned for audit (§6 discipline). All inputs are facts computed upstream
(bar counts, regime classifications, vol regime); this module applies the
documented rules and returns every check it ran, PASS or not, so the verdict
is auditable from the check list alone (same explainability contract as the
directional scorer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from libs.trading_core.models import IVRegime, MarketRegime, TradeabilityState


@dataclass(frozen=True)
class TradeabilityParams:
    """Research thresholds for the §9 tradeability verdict — all tunable,
    never universal truths.

    - ``min_bars``: fewer stored daily bars than this and the slow trend
      structure (sma200) is undefined -> DATA_INSUFFICIENT.
    - ``max_stale_trading_days``: how many WEEKDAYS the newest stored bar may
      lag the last expected trading day before the data is too stale to act
      on -> DATA_INSUFFICIENT (0 = must be current; small tolerance covers
      unmodeled market holidays).
    - ``blocked_regimes``: regime classifications that veto NEW entries for
      the environment regardless of direction (§6.1: TRANSITION defaults to
      NO TRADE). Applied to BOTH the market and the symbol regime.
    - ``version``: identifies this rule set in payloads and audit records.
    """

    min_bars: int = 200
    max_stale_trading_days: int = 1
    blocked_regimes: tuple[MarketRegime, ...] = (MarketRegime.TRANSITION,)
    version: str = "tradeability-v1"


@dataclass
class TradeabilityCheck:
    """One evaluated tradeability input, with the §26 evidence string.

    ``status`` is one of PASS / CONDITION / BLOCK / INSUFFICIENT — the
    verdict-relevant severity of THIS check alone.
    """

    name: str
    status: str
    detail: str


@dataclass
class TradeabilityDecision:
    """The §9 verdict with its full evidence (§13 separate dimension).

    - ``state``: TRADEABLE | CONDITIONAL | BLOCKED | DATA_INSUFFICIENT.
    - ``reasons``: human-readable strings for every non-PASS check — the §10
      "WHY" line renders these verbatim.
    - ``checks``: ALL checks run, passed or not (auditability).
    - ``version``: the :class:`TradeabilityParams` rule-set version.
    """

    state: TradeabilityState
    reasons: list[str] = field(default_factory=list)
    checks: list[TradeabilityCheck] = field(default_factory=list)
    version: str = ""


PASS = "PASS"
CONDITION = "CONDITION"
BLOCK = "BLOCK"
INSUFFICIENT = "INSUFFICIENT"


def assess_tradeability(
    bar_count: int,
    stale_trading_days: int,
    market_regime: MarketRegime | None,
    symbol_regime: MarketRegime | None,
    vol_regime: IVRegime | None,
    vol_unavailable_reason: str | None = None,
    params: TradeabilityParams = TradeabilityParams(),
) -> TradeabilityDecision:
    """Assess environment tradeability from upstream facts (§9).

    Inputs are FACTS computed by their owning engines — this function never
    re-derives them:

    - ``bar_count``: stored daily bars for the symbol.
    - ``stale_trading_days``: WEEKDAYS the newest stored bar lags the last
      expected trading day (0 = current; caller computes it with its own
      calendar arithmetic).
    - ``market_regime`` / ``symbol_regime``: §6.1 classifications (``None``
      when the classifier could not run — counts as insufficient data).
    - ``vol_regime``: §7 classification, or ``None`` when unavailable;
      ``vol_unavailable_reason`` then says WHY (e.g. option data not in the
      current Massive plan). Unknown volatility does not BLOCK — it degrades
      to CONDITIONAL with the stated reason (honest posture: stock
      instruments may still be researched; the §10 execution chain has its
      own volatility gate).

    Verdict precedence (first match wins):

    1. DATA_INSUFFICIENT — any INSUFFICIENT check (bars, staleness, missing
       regime read).
    2. BLOCKED — any BLOCK check (blocked regime on either level, or
       EXTREME volatility).
    3. CONDITIONAL — any CONDITION check (HIGH volatility, unknown
       volatility).
    4. TRADEABLE.
    """
    checks: list[TradeabilityCheck] = []

    # --- Data sufficiency -------------------------------------------------
    if bar_count < params.min_bars:
        checks.append(TradeabilityCheck(
            "DATA_QUALITY", INSUFFICIENT,
            f"{bar_count} stored bars < required {params.min_bars} — slow "
            "trend structure undefined",
        ))
    else:
        checks.append(TradeabilityCheck(
            "DATA_QUALITY", PASS,
            f"{bar_count} stored bars >= required {params.min_bars}",
        ))

    if stale_trading_days > params.max_stale_trading_days:
        checks.append(TradeabilityCheck(
            "DATA_FRESHNESS", INSUFFICIENT,
            f"newest bar lags {stale_trading_days} trading days > allowed "
            f"{params.max_stale_trading_days} — too stale to act on",
        ))
    else:
        checks.append(TradeabilityCheck(
            "DATA_FRESHNESS", PASS,
            f"newest bar lags {stale_trading_days} trading days <= allowed "
            f"{params.max_stale_trading_days}",
        ))

    # --- Regime environment ----------------------------------------------
    for level, regime in (("MARKET_REGIME", market_regime),
                          ("SYMBOL_REGIME", symbol_regime)):
        if regime is None:
            checks.append(TradeabilityCheck(
                level, INSUFFICIENT, "regime could not be classified"))
        elif regime in params.blocked_regimes:
            checks.append(TradeabilityCheck(
                level, BLOCK,
                f"{regime.value} regime defaults to NO TRADE (§6.1)"))
        else:
            checks.append(TradeabilityCheck(level, PASS, regime.value))

    # --- Volatility environment ------------------------------------------
    if vol_regime is None:
        checks.append(TradeabilityCheck(
            "VOLATILITY_REGIME", CONDITION,
            "volatility regime unknown — "
            + (vol_unavailable_reason or "no option data")
            + "; stock instruments only, execution gates still apply",
        ))
    elif vol_regime is IVRegime.EXTREME:
        checks.append(TradeabilityCheck(
            "VOLATILITY_REGIME", BLOCK,
            "EXTREME volatility vetoes new entries (§10)"))
    elif vol_regime is IVRegime.HIGH:
        checks.append(TradeabilityCheck(
            "VOLATILITY_REGIME", CONDITION,
            "HIGH volatility — tradeable with degraded conditions"))
    else:
        checks.append(TradeabilityCheck(
            "VOLATILITY_REGIME", PASS, vol_regime.value))

    # --- Verdict precedence ----------------------------------------------
    if any(c.status == INSUFFICIENT for c in checks):
        state = TradeabilityState.DATA_INSUFFICIENT
    elif any(c.status == BLOCK for c in checks):
        state = TradeabilityState.BLOCKED
    elif any(c.status == CONDITION for c in checks):
        state = TradeabilityState.CONDITIONAL
    else:
        state = TradeabilityState.TRADEABLE

    reasons = [f"{c.name}: {c.detail}" for c in checks if c.status != PASS]
    return TradeabilityDecision(
        state=state, reasons=reasons, checks=checks, version=params.version
    )
