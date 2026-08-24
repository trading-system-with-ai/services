"""Short-squeeze hazard PROXY (auto-strategy program Phase D,
docs/auto-strategy-portfolio-design.md; execution-chains-roadmap.md §33
deferral context).

THIS IS NOT A SHORT-INTEREST SIGNAL. Real squeeze inputs — short interest,
days-to-cover, borrow fee, free float — are unavailable from every
configured provider (audited 2026-08-20; a §33-approved vendor is required
and no proxy substitutes for them, §44 rule 18). What CAN be measured
honestly from stored daily bars is crowding/momentum hazard on the short
side:

- volume z-score over the trailing window (breakout participation),
- proximity to the trailing high (shorts near highs are squeeze fuel),
- overnight gap-up (forced-cover behavior).

The report is consumed by the §10 SQUEEZE_RISK gate in REPORT mode — it
never vetoes; it states its numbers, its verdict, and its proxy nature.
Every threshold is a research parameter (§6.2), never a truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass(frozen=True)
class SqueezeProxyParams:
    """Research defaults — validated against the watchlist before any veto
    is ever proposed (same promotion discipline as the LIQUIDITY gate)."""

    volume_window: int = 20
    volume_z_threshold: float = 2.0
    high_window: int = 252
    high_proximity_pct: float = 0.05  # within 5% of the trailing high
    gap_up_pct: float = 0.05  # overnight open 5% above prior close


@dataclass
class SqueezeProxyReport:
    """Honest nulls throughout: an unmeasurable component is None with its
    reason stated, never zero-filled (§44 rule 18)."""

    volume_z: float | None
    dist_from_high_pct: float | None
    overnight_gap_pct: float | None
    elevated: bool
    reasons: list[str] = field(default_factory=list)


def assess_squeeze_proxy(
    opens: list[float],
    closes: list[float],
    highs: list[float],
    volumes: list[float],
    params: SqueezeProxyParams = SqueezeProxyParams(),
) -> SqueezeProxyReport:
    """Assess short-side crowding hazard from daily bars (latest bar last).

    ``elevated`` fires when (volume z >= threshold AND price within
    ``high_proximity_pct`` of the trailing high) OR the latest overnight
    gap-up exceeds ``gap_up_pct`` — breakout participation near highs, or
    forced-cover behavior, respectively. Unmeasurable components never
    contribute; with everything unmeasurable the verdict is ``False`` with
    the reasons saying so (a proxy that cannot see must not cry wolf).
    """
    n = len(closes)
    reasons: list[str] = []

    volume_z: float | None = None
    if len(volumes) >= params.volume_window + 1:
        window = volumes[-(params.volume_window + 1) : -1]
        mu = mean(window)
        sd = stdev(window)
        if sd > 0.0:
            volume_z = (volumes[-1] - mu) / sd
        else:
            reasons.append(
                f"volume stdev over the prior {params.volume_window} bars is 0 — z unmeasurable"
            )
    else:
        reasons.append(
            f"fewer than {params.volume_window + 1} volume bars — z unmeasurable"
        )

    dist_from_high: float | None = None
    if n >= 2:
        lookback = highs[-min(len(highs), params.high_window):]
        trailing_high = max(lookback)
        if trailing_high > 0.0:
            dist_from_high = (trailing_high - closes[-1]) / trailing_high
        if len(highs) < params.high_window:
            reasons.append(
                f"only {len(highs)} bars for the {params.high_window}-bar high — "
                "proximity measured over the available window"
            )
    else:
        reasons.append("fewer than 2 bars — trailing-high proximity unmeasurable")

    gap_up: float | None = None
    if n >= 2 and len(opens) == n and closes[-2] > 0.0:
        gap_up = (opens[-1] - closes[-2]) / closes[-2]
    else:
        reasons.append("prior close/open unavailable — overnight gap unmeasurable")

    crowded_breakout = (
        volume_z is not None
        and dist_from_high is not None
        and volume_z >= params.volume_z_threshold
        and dist_from_high <= params.high_proximity_pct
    )
    forced_cover = gap_up is not None and gap_up >= params.gap_up_pct
    elevated = bool(crowded_breakout or forced_cover)
    if crowded_breakout:
        reasons.append(
            f"volume z {volume_z:.2f} >= {params.volume_z_threshold:g} within "
            f"{dist_from_high * 100.0:.1f}% of the trailing high"
        )
    if forced_cover:
        reasons.append(
            f"overnight gap-up {gap_up * 100.0:.1f}% >= {params.gap_up_pct * 100.0:g}%"
        )
    if not elevated and not reasons:
        reasons.append("no crowding signature in volume/high-proximity/gap")

    report = SqueezeProxyReport(
        volume_z=volume_z,
        dist_from_high_pct=dist_from_high * 100.0 if dist_from_high is not None else None,
        overnight_gap_pct=gap_up * 100.0 if gap_up is not None else None,
        elevated=elevated,
        reasons=reasons,
    )
    for v in (report.volume_z, report.dist_from_high_pct, report.overnight_gap_pct):
        assert v is None or math.isfinite(v)
    return report
