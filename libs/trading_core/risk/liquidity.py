"""Underlying (stock) liquidity gate — REPORT mode (risk-engine audit §7.3
"Liquidity gate" row, §10 Phase B0; spec §2/§5 hard limit, §70 shadow mode).

Pure, deterministic, stdlib-only (house rule): no DB, no market-data calls.
The caller (the §10 gate chain in ``apps/gateway/routers/orders.py`` and the
§4.3 pool readiness check) hands in what it already loaded — stored daily
volumes, the proposed share count and, if one exists, the live stock NBBO —
and receives a :class:`LiquidityReport` with the MEASURED numbers and a
HYPOTHETICAL verdict.

Three components, every one a parameter on :class:`LiquidityLimits`:

- ADV20 = arithmetic mean of the LAST ``adv_window`` daily volumes (share
  count); breach when ``adv20 < min_adv20_shares``;
- order participation = ``order_shares / adv20``; breach when it exceeds
  ``max_order_pct_adv20`` (a fraction: 0.01 = 1 % of ADV20);
- quote spread = ``(ask - bid) / mid`` off the stock NBBO; breach when it
  exceeds ``max_quote_spread_pct`` (0.005 = 0.5 %).

Honest nulls (§44 rule 18): a component that cannot be measured is ``None``
and is named in ``reasons`` as unmeasured; it is NEVER treated as a pass or
a fail. The verdict is ``"UNAVAILABLE"`` only when NOTHING could be
measured, ``"WOULD_FAIL"`` when any MEASURED component breaches, else
``"PASS"``.

REPORT MODE: this module never vetoes anything. Its output is written into
the gate detail and the RISK_DECISION ``shadow.liquidity`` block so the
research defaults can be validated against the watchlist over the Q3 shadow
window before the gate is promoted to a FAIL veto (audit §7.3, §11 Q3).
Option-LEG liquidity (OI / spread / DTE) is enforced separately by the §9
filters in CONTRACT_SELECTION and is untouched here.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Verdict vocabulary (contract-fixed strings; the UI renders them verbatim).
VERDICT_PASS = "PASS"
VERDICT_WOULD_FAIL = "WOULD_FAIL"
VERDICT_UNAVAILABLE = "UNAVAILABLE"

#: The gate's operating mode until promoted (audit §7.3): reported, never
#: enforced.
MODE_REPORT = "REPORT"


@dataclass(frozen=True)
class LiquidityLimits:
    """Underlying liquidity thresholds (audit §7.3 / §10 Phase B0).

    RESEARCH DEFAULTS — UNVALIDATED. They are conventional starting points
    (an ADV floor, ~1 % participation, a half-percent NBBO spread) and NOT
    calibrated against this platform's watchlist, which contains illiquid
    small caps (e.g. RDW, whose §9 rejections are documented in the DEVLOG).
    An un-shadowed hard veto with these guesses would silently block symbols
    that trade today, which is why the gate runs in REPORT mode (verdict
    logged in ``shadow.liquidity``) until the Q3 shadow window has shown
    which watchlist symbols it would have blocked and the numbers have been
    reviewed. Every field is a parameter, never a hardcoded truth (§6.2).

    - ``min_adv20_shares``: minimum average daily volume over ``adv_window``
      sessions, in shares (default 100 000);
    - ``max_order_pct_adv20``: maximum order size as a FRACTION of ADV20
      (default 0.01 = 1 % participation);
    - ``max_quote_spread_pct``: maximum stock NBBO spread as a FRACTION of
      the mid (default 0.005 = 0.5 %);
    - ``adv_window``: number of trailing daily bars averaged (default 20).
    """

    min_adv20_shares: int = 100_000
    max_order_pct_adv20: float = 0.01
    max_quote_spread_pct: float = 0.005
    adv_window: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_adv20_shares) or self.min_adv20_shares < 0:
            raise ValueError("min_adv20_shares must be a finite number >= 0")
        if not (0.0 < self.max_order_pct_adv20 <= 1.0):
            raise ValueError("max_order_pct_adv20 must be in (0, 1]")
        if not (0.0 < self.max_quote_spread_pct <= 1.0):
            raise ValueError("max_quote_spread_pct must be in (0, 1]")
        if self.adv_window < 1:
            raise ValueError("adv_window must be >= 1")


@dataclass(frozen=True)
class LiquidityReport:
    """What was measured, what could not be, and the hypothetical verdict.

    ``adv20`` / ``order_pct_adv20`` / ``quote_spread_pct`` are ``None`` when
    that component could not be measured (honest null — the field name keeps
    ``adv20`` even when ``LiquidityLimits.adv_window`` differs, because it is
    the contract-fixed name the audit rows and the UI use); ``*_pct`` fields
    are FRACTIONS (0.0123 = 1.23 %). ``verdict`` is one of ``"PASS"`` /
    ``"WOULD_FAIL"`` / ``"UNAVAILABLE"``; ``reasons`` are audit-exact
    sentences with the real numbers; ``mode`` is ``"REPORT"`` — the report
    never vetoes.
    """

    adv20: float | None
    order_pct_adv20: float | None
    quote_spread_pct: float | None
    verdict: str
    reasons: tuple[str, ...]
    mode: str = MODE_REPORT
    #: True when at least one component was measured but not all three —
    #: a "PASS" with ``partial`` set is a pass on the measured components
    #: only. When this gate is promoted from REPORT to a veto, partial
    #: measurement must NOT read as a pass (fail closed) — the promotion
    #: design (audit §11 Q3) decides that rule; the flag exists so the
    #: shadow window already records how often it would have applied.
    partial: bool = False


def average_daily_volume(
    volumes: Sequence[float], window: int
) -> float | None:
    """Mean of the LAST ``window`` volumes; ``None`` when fewer than
    ``window`` volumes exist (never a shorter-window substitute — §44 rule 18).
    Non-finite / negative entries make the window unmeasurable too."""
    if window < 1 or len(volumes) < window:
        return None
    raw = list(volumes[-window:])
    if any(v is None or isinstance(v, bool) for v in raw):
        return None
    try:
        tail = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in tail):
        return None
    return math.fsum(tail) / window


def quote_spread_fraction(
    bid: float | None, ask: float | None
) -> float | None:
    """``(ask - bid) / mid`` off a two-sided stock NBBO; ``None`` when either
    side is missing, non-positive, non-finite, or the market is crossed
    (``ask < bid``). A locked market (``ask == bid``) is a measured 0.0."""
    if bid is None or ask is None:
        return None
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return None
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def evaluate_underlying_liquidity(
    volumes: Sequence[float],
    order_shares: int | None,
    bid: float | None,
    ask: float | None,
    limits: LiquidityLimits = LiquidityLimits(),
) -> LiquidityReport:
    """Measure the underlying's liquidity and report the hypothetical verdict
    (audit §7.3, REPORT mode — the caller must not veto on it).

    ``volumes`` are daily share volumes oldest-first (the stored
    ``stock_bars_daily.volume`` series the chain already loaded);
    ``order_shares`` is the proposed SHARE count (``None`` when unknown — e.g.
    an option candidate whose contracts are not translated to shares);
    ``bid`` / ``ask`` the live stock NBBO or ``None`` when no quote exists.
    """
    reasons: list[str] = []
    measured = 0
    breached = 0

    adv20 = average_daily_volume(volumes, limits.adv_window)
    if adv20 is None:
        reasons.append(
            f"ADV{limits.adv_window} unmeasured: {len(volumes)} stored "
            f"volume(s), need {limits.adv_window}"
        )
    else:
        measured += 1
        if adv20 < limits.min_adv20_shares:
            breached += 1
            reasons.append(
                f"ADV{limits.adv_window} {adv20:,.0f} sh < "
                f"{limits.min_adv20_shares:,} minimum"
            )
        else:
            reasons.append(
                f"ADV{limits.adv_window} {adv20:,.0f} sh >= "
                f"{limits.min_adv20_shares:,} minimum"
            )

    order_pct: float | None = None
    if order_shares is None:
        reasons.append("order participation unmeasured: order size unknown")
    elif adv20 is None:
        reasons.append(
            f"order participation unmeasured: {order_shares:,} sh but no "
            f"ADV{limits.adv_window}"
        )
    elif adv20 <= 0.0:
        # A zero ADV can't be a denominator: the ADV component already
        # reports the breach (0 < minimum); participation stays unmeasured.
        reasons.append(
            f"order participation unmeasured: ADV{limits.adv_window} is 0"
        )
    else:
        measured += 1
        order_pct = order_shares / adv20
        if order_pct > limits.max_order_pct_adv20:
            breached += 1
            reasons.append(
                f"order {order_shares:,} sh = {order_pct * 100:.2f}% of "
                f"ADV{limits.adv_window} > {limits.max_order_pct_adv20 * 100:.2f}% "
                "maximum"
            )
        else:
            reasons.append(
                f"order {order_shares:,} sh = {order_pct * 100:.2f}% of "
                f"ADV{limits.adv_window} <= "
                f"{limits.max_order_pct_adv20 * 100:.2f}% maximum"
            )

    spread = quote_spread_fraction(bid, ask)
    if spread is None:
        if bid is None or ask is None:
            reasons.append("quote spread unmeasured: no two-sided stock quote")
        else:
            reasons.append(
                f"quote spread unmeasured: unusable NBBO bid {bid} / ask {ask} "
                "(non-positive or crossed)"
            )
    else:
        measured += 1
        if spread > limits.max_quote_spread_pct:
            breached += 1
            reasons.append(
                f"quote spread {spread * 100:.3f}% (bid {bid} / ask {ask}) > "
                f"{limits.max_quote_spread_pct * 100:.2f}% maximum"
            )
        else:
            reasons.append(
                f"quote spread {spread * 100:.3f}% (bid {bid} / ask {ask}) <= "
                f"{limits.max_quote_spread_pct * 100:.2f}% maximum"
            )

    if measured == 0:
        verdict = VERDICT_UNAVAILABLE
    elif breached > 0:
        verdict = VERDICT_WOULD_FAIL
    else:
        verdict = VERDICT_PASS
    partial = 0 < measured < 3
    if partial and verdict == VERDICT_PASS:
        reasons.append(
            f"partial measurement: {measured} of 3 components measured — "
            "PASS applies to the measured components only"
        )

    return LiquidityReport(
        adv20=adv20,
        order_pct_adv20=order_pct,
        quote_spread_pct=spread,
        verdict=verdict,
        reasons=tuple(reasons),
        mode=MODE_REPORT,
        partial=partial,
    )


def liquidity_report_detail(report: LiquidityReport, limits: LiquidityLimits) -> str:
    """One audit-exact detail line for a gate / readiness check: the measured
    values (``n/a`` when unmeasured) and the hypothetical verdict, prefixed
    with the REPORT-mode disclaimer so nobody reads it as an enforced limit.
    """
    adv = f"{report.adv20:,.0f} sh" if report.adv20 is not None else "n/a"
    pct = (
        f"{report.order_pct_adv20 * 100:.2f}% of ADV{limits.adv_window}"
        if report.order_pct_adv20 is not None
        else "n/a"
    )
    spread = (
        f"{report.quote_spread_pct * 100:.3f}%"
        if report.quote_spread_pct is not None
        else "n/a"
    )
    head = (
        f"underlying liquidity ({report.mode} mode, research limits): "
        f"ADV{limits.adv_window} {adv}; order {pct}; quote spread {spread}"
    )
    if report.verdict == VERDICT_UNAVAILABLE:
        return f"{head} — verdict UNAVAILABLE (nothing measurable): " + "; ".join(
            report.reasons
        )
    if report.verdict == VERDICT_WOULD_FAIL:
        return f"{head} — would FAIL: " + "; ".join(report.reasons)
    return f"{head} — would PASS: " + "; ".join(report.reasons)
