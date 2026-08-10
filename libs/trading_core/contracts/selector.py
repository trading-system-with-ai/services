"""Contract Selector v0 — filters and ranking heuristic (plan §9).

Pipeline of :func:`select_contracts` (§9.1, §9.2):

1. Side gate: BULL considers calls only, BEAR puts only — this is a
   long-only account (plan §5); the wrong side is ineligible with reason
   ``"wrong side for <DIRECTION> direction"``.
2. Hard filters (§9.1), every failure recorded in ``fail_reasons`` with the
   actual numbers so rejections are explainable (plan §36): DTE window,
   |delta| window, open interest, volume, relative spread, theta burden
   (|theta|/mid per calendar day) and a positive mid.
3. Ranking (§9.2) — v0 HEURISTIC, explicitly scheduled to be replaced by an
   EV-based ranking in Phase 10:

       liquidity    = 1 - min(1, spread_pct / max_spread_pct)
       theta_burden = min(1, (|theta| / mid) / max_theta_premium_pct)
       delta_fit    = 1 - |  |delta| - delta_mid | / half_width
       score        = w_liquidity * liquidity
                      - w_theta * theta_burden
                      + w_delta_fit * delta_fit

   with ``delta_mid = (abs_delta_min + abs_delta_max) / 2`` and
   ``half_width = (abs_delta_max - abs_delta_min) / 2``. The top ``top_n``
   eligible contracts by score receive rank 1..N (ties broken by chain
   order for determinism); the rest keep their score but no rank.

Every input contract is returned, in input order, with its verdict — the UI
needs All / Eligible / Recommended views (plan §34). All thresholds and
weights are :class:`SelectorParams` fields, never hardcoded (house rule),
and per §9.1 they are BACKTEST PARAMETERS to sweep — "do not permanently
select 0.55 delta as a fixed rule".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class SelectorParams:
    """Contract selector thresholds and ranking weights (plan §9.1, §9.2).

    All values are §9.1 backtest parameters — meant to be swept in the
    backtester, not truths ("do not permanently select 0.55 delta as a
    fixed rule"). Filters are inclusive at their boundaries.

    - ``dte_min`` / ``dte_max``: calendar days-to-expiry window.
    - ``abs_delta_min`` / ``abs_delta_max``: |delta| window (right-agnostic).
    - ``min_open_interest`` / ``min_volume``: liquidity floors.
    - ``max_spread_pct``: max relative bid-ask spread (spread / mid).
    - ``max_theta_premium_pct``: max |theta| / mid per calendar day — the
      fraction of the premium that decays away each day.
    - ``top_n``: how many eligible contracts get a rank (§9.2).
    - ``w_liquidity`` / ``w_theta`` / ``w_delta_fit``: §9.2 v0 ranking
      weights (Phase 10 replaces this heuristic with EV-based ranking).
    """

    dte_min: int = 30
    dte_max: int = 90
    abs_delta_min: float = 0.40
    abs_delta_max: float = 0.75
    min_open_interest: int = 100
    min_volume: int = 10
    max_spread_pct: float = 0.10
    max_theta_premium_pct: float = 0.02
    top_n: int = 3
    w_liquidity: float = 1.0
    w_theta: float = 1.0
    w_delta_fit: float = 1.0


@dataclass
class ContractQuote:
    """One option contract snapshot from a chain read (read-only, plan §9).

    ``theta`` is per calendar day and ``vega`` per 1 IV point, matching the
    :mod:`libs.trading_core.options` conventions; ``delta`` is signed
    (calls +, puts -); ``spread_pct`` is (ask - bid) / mid.
    """

    expiry: date
    dte: int
    strike: float
    right: str  # "C" | "P"
    bid: float
    ask: float
    mid: float
    spread_pct: float
    last: float | None
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class ScoredContract:
    """Selector verdict for one contract (plan §9.2, §34).

    ``eligible`` with ``fail_reasons`` explains every rejection with real
    numbers (plan §36). ``score`` and ``components`` (the raw ``liquidity``,
    ``theta_burden`` and ``delta_fit`` terms) are set only for eligible
    contracts — honest nulls otherwise, never NaN (house rule). ``rank`` is
    1..top_n for the recommended contracts, else None.
    """

    contract: ContractQuote
    eligible: bool
    fail_reasons: list[str] = field(default_factory=list)
    score: float | None = None
    rank: int | None = None
    components: dict | None = None


def _required_right(direction: str) -> str:
    """Long-only side for a direction (plan §5): BULL calls, BEAR puts."""
    if direction == "BULL":
        return "C"
    if direction == "BEAR":
        return "P"
    raise ValueError(f'direction must be "BULL" or "BEAR", got {direction!r}')


def _fail_reasons(
    c: ContractQuote, direction: str, params: SelectorParams
) -> list[str]:
    """All §9.1 filter failures for one contract, with the actual numbers."""
    reasons: list[str] = []
    if c.right != _required_right(direction):
        reasons.append(f"wrong side for {direction} direction")
    if not (params.dte_min <= c.dte <= params.dte_max):
        reasons.append(
            f"DTE {c.dte} outside [{params.dte_min}, {params.dte_max}]"
        )
    abs_delta = abs(c.delta)
    if not (params.abs_delta_min <= abs_delta <= params.abs_delta_max):
        reasons.append(
            f"|delta| {abs_delta:.2f} outside "
            f"[{params.abs_delta_min:.2f}, {params.abs_delta_max:.2f}]"
        )
    if c.open_interest < params.min_open_interest:
        reasons.append(
            f"open interest {c.open_interest} < {params.min_open_interest}"
        )
    if c.volume < params.min_volume:
        reasons.append(f"volume {c.volume} < {params.min_volume}")
    if c.spread_pct > params.max_spread_pct:
        reasons.append(
            f"spread_pct {c.spread_pct:.4f} > {params.max_spread_pct:.4f}"
        )
    if c.mid <= 0.0:
        reasons.append(f"mid {c.mid:.4f} <= 0")
    else:
        burden = abs(c.theta) / c.mid
        if burden > params.max_theta_premium_pct:
            reasons.append(
                f"theta burden {burden:.4f}/day > "
                f"{params.max_theta_premium_pct:.4f}"
            )
    return reasons


def _score(c: ContractQuote, params: SelectorParams) -> tuple[float, dict]:
    """§9.2 v0 heuristic score and its component terms (eligible only)."""
    liquidity = 1.0 - min(1.0, c.spread_pct / params.max_spread_pct)
    theta_burden = min(
        1.0, (abs(c.theta) / c.mid) / params.max_theta_premium_pct
    )
    delta_mid = (params.abs_delta_min + params.abs_delta_max) / 2.0
    half_width = (params.abs_delta_max - params.abs_delta_min) / 2.0
    if half_width > 0.0:
        delta_fit = 1.0 - abs(abs(c.delta) - delta_mid) / half_width
    else:
        # Degenerate window (min == max): eligibility already pinned
        # |delta| to delta_mid, so the fit is perfect by construction.
        delta_fit = 1.0
    score = (
        params.w_liquidity * liquidity
        - params.w_theta * theta_burden
        + params.w_delta_fit * delta_fit
    )
    components = {
        "liquidity": liquidity,
        "theta_burden": theta_burden,
        "delta_fit": delta_fit,
    }
    return score, components


def select_contracts(
    chain: list[ContractQuote],
    direction: str,
    params: SelectorParams = SelectorParams(),
) -> list[ScoredContract]:
    """Filter and rank an option chain for a direction (plan §9).

    Returns one :class:`ScoredContract` per input contract, in input order —
    every contract carries its verdict so the UI can show All / Eligible /
    Recommended (plan §34). ``direction`` is ``"BULL"`` (calls) or
    ``"BEAR"`` (puts) — long-only account (plan §5); anything else raises
    ``ValueError``. Deterministic: score ties are broken by chain order.
    """
    _required_right(direction)  # validate direction up front

    scored: list[ScoredContract] = []
    for c in chain:
        reasons = _fail_reasons(c, direction, params)
        if reasons:
            scored.append(
                ScoredContract(contract=c, eligible=False, fail_reasons=reasons)
            )
        else:
            score, components = _score(c, params)
            scored.append(
                ScoredContract(
                    contract=c,
                    eligible=True,
                    fail_reasons=[],
                    score=score,
                    components=components,
                )
            )

    # Rank the top_n eligible by score (§9.2); sorted() is stable, so equal
    # scores keep chain order — deterministic output for identical input.
    eligible = [s for s in scored if s.eligible]
    for rank, s in enumerate(
        sorted(eligible, key=lambda s: -s.score)[: params.top_n], start=1
    ):
        s.rank = rank
    return scored
