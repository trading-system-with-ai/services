"""Stress engine — scenarios, catalogue, results and the hypothetical
STRESS cap (Phase D design §8.3; risk spec §24, §25, §26, §27, §67).

Pure stdlib, deterministic, no I/O (house rule). **SHADOW by construction**
(spec §70): every number here is computed, persisted and displayed, and
NOTHING decides. :class:`StressLimits` records ``mode="SHADOW"`` and
:func:`stress_caps` emits a :class:`~libs.trading_core.risk.pretrade
.QuantityCap` that only binds where a caller deliberately passes it to
``assess(extra_caps=...)`` — an explicit human promotion step. Spec §27
says "The Stress Test must retain veto authority"; that veto is the
PRODUCTION promotion, not this module.

What a scenario is
------------------
:class:`Scenario` is a *state*, not a model: a fractional spot shock (one
uniform number, or per-ticker), a RELATIVE multiplicative IV shock
(``+0.20`` ⇒ ``iv1 = iv0 * 1.20``, design §8.2) and a number of calendar
days forward. The book is revalued under it by
:func:`~libs.trading_core.options.reval.scenario_pnl` — FULL revaluation
of every option leg that has an IV, DELTA_LINEAR (labelled) for the rest.

The uniform ``spot_shock`` applies the SAME fractional move to every
underlying. That is a **beta = 1 assumption** and it is documented as such
on every row: a −10 % uniform shock is not "SPY −10 %", it is "every name
in the book −10 %". Per-ticker shocks (which the historical windows always
produce) carry no such assumption.

Three scenario families
-----------------------
1. **HISTORICAL** (spec §25) — "what would TODAY's book lose if a window
   that actually happened happened now?". Shocks come from stored closes:
   the per-ticker cumulative simple return over the window. The IV shock is
   the honest part: this platform has **no IV history** (spec §24), so the
   IV move is a *proxy* — the ratio of realized vol inside the window to
   realized vol over the prior 20 days, minus 1, clipped to
   ``[-0.5, +2.0]`` — and every row carries
   ``iv_shock_source="RV_PROXY"`` so no reader can mistake it for a
   measured IV move. A window that is not fully inside the stored history
   produces an ``UNAVAILABLE`` row with the real dates, never a partial
   window silently rescaled.

2. **HYPOTHETICAL** (spec §24, §26) — the research grid. Spec §24 is
   explicit: "Do not blindly adopt these example numbers." Every row in
   :data:`DEFAULT_HYPOTHETICAL_SCENARIOS` is therefore
   ``validated=False``, i.e. UNVALIDATED research parameterisation, and the
   API/UI must badge it as such.

3. **USER** (spec §26, §51) — a scenario the operator types. Same shape,
   ``validated=False``.

Plus **AUTO** historical windows found in the stored history itself
(:func:`auto_worst_windows`): the worst 1-day, 5-day and 10-day windows of
the equal-weight book return. These are empirical (spec §24: "derived
empirically where possible") and are the only historical rows guaranteed
to exist for any book with history.

Sign convention (contract §1): ``pnl_usd`` is gain-positive, so a stress
LOSS is NEGATIVE. ``worst`` is the row with the smallest ``pnl_usd``. The
cap compares ``-pnl_usd`` (a loss, positive) against a limit, matching how
VaR/ES are reported everywhere else.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from ...options.reval import (
    METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL,
    OptionLeg,
    StockLeg,
    scenario_pnl,
)
from .base import ModelHealth, ModelTier

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    from ..pretrade import QuantityCap

#: Model name / version recorded on results (Phase B contract §4 — bump
#: MAJOR on any arithmetic change, MINOR on a parameter-default change).
MODEL_NAME = "stress"
MODEL_VERSION = "1.0.0"

#: Spec §5 classification. Stress is listed in §5's FIRST tier: a scenario
#: row is a deterministic reprice of today's book under a stated shock, not
#: a fitted forecast — there is no estimated parameter in it. It is exposed
#: as a module constant rather than a ``ModelMeta.tier`` because this module
#: builds no ``ModelMeta`` at all (its results are ``ScenarioResult`` /
#: ``StressResult``, whose provenance is the catalogue + model version).
MODEL_TIER = ModelTier.TIER_1

#: Catalogue version: bump whenever the DEFAULT_* tuples change, so a
#: persisted stress run stays interpretable (spec §44 reproducibility).
CATALOGUE_VERSION = "d.1"

#: Scenario kinds (design §8.3).
KIND_HISTORICAL = "HISTORICAL"
KIND_HYPOTHETICAL = "HYPOTHETICAL"
KIND_IV_GRID = "IV_GRID"
KIND_USER = "USER"
SCENARIO_KINDS = (KIND_HISTORICAL, KIND_HYPOTHETICAL, KIND_IV_GRID, KIND_USER)

#: Provenance label for the historical IV shock (spec §24 honesty): this
#: platform stores no IV history, so the IV move is inferred from realized
#: vol. Never presented as a measured IV change.
IV_SHOCK_SOURCE_RV_PROXY = "RV_PROXY"
#: The IV shock was given directly by the scenario definition.
IV_SHOCK_SOURCE_SPECIFIED = "SPECIFIED"

#: Cap layer + code for the stress limit (design §8.3).
LAYER_STRESS = "STRESS"
CODE_STRESS_LOSS = "STRESS_LOSS_LIMIT"

#: SHADOW / PRODUCTION mode strings (mirrors ``pretrade``; only PRODUCTION
#: may ever be wired into ``assess``).
MODE_SHADOW = "SHADOW"
MODE_PRODUCTION = "PRODUCTION"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalShockParams:
    """How a historical window is turned into shocks (design §8.3).

    **RESEARCH DEFAULTS — UNVALIDATED.** Every one is a parameter, never a
    hardcoded truth (house rule).

    - ``rv_prior_days`` (20): the comparison window for the realized-vol
      ratio proxy — RV inside the stressed window vs RV over the 20 trading
      days immediately before it;
    - ``iv_shock_floor`` (−0.5) / ``iv_shock_ceiling`` (+2.0): the clip on
      the proxy. A 3-day window can produce an absurd RV ratio; clipping
      keeps a proxy from becoming fiction;
    - ``min_window_obs`` (2): a window needs at least two closes to have a
      return at all;
    - ``min_rv_obs`` (5): below this the RV ratio is not computed and the
      IV shock is reported as 0.0 with the reason on the scenario.
    """

    rv_prior_days: int = 20
    iv_shock_floor: float = -0.5
    iv_shock_ceiling: float = 2.0
    min_window_obs: int = 2
    min_rv_obs: int = 5

    def __post_init__(self) -> None:
        for name in ("rv_prior_days", "min_window_obs", "min_rv_obs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 2:
                raise ValueError(f"{name} must be an int >= 2, got {v!r}")
        if not math.isfinite(self.iv_shock_floor) or not math.isfinite(
            self.iv_shock_ceiling
        ):
            raise ValueError("iv_shock_floor/ceiling must be finite")
        if self.iv_shock_floor <= -1.0:
            raise ValueError(
                f"iv_shock_floor must be > -1 (a -100 % IV shock is not a "
                f"volatility), got {self.iv_shock_floor}"
            )
        if self.iv_shock_ceiling < self.iv_shock_floor:
            raise ValueError(
                f"iv_shock_ceiling {self.iv_shock_ceiling} < floor "
                f"{self.iv_shock_floor}"
            )


@dataclass(frozen=True)
class StressLimits:
    """The stress layer's threshold (design §8.3).

    **RESEARCH DEFAULT — UNVALIDATED** (spec §11: "Do NOT choose arbitrary
    production thresholds silently"). ``max_stress_loss_pct_nav`` (0.10):
    the WORST scenario's loss may not exceed 10 % of NAV. ``mode``
    (``"SHADOW"``) keeps it out of every Tier 0 decision until a human
    promotes it (spec §27's veto authority is that promotion).
    """

    max_stress_loss_pct_nav: float = 0.10
    mode: str = MODE_SHADOW

    def __post_init__(self) -> None:
        v = self.max_stress_loss_pct_nav
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
        ):
            raise ValueError(
                f"max_stress_loss_pct_nav must be a finite number > 0, got {v!r}"
            )
        if self.mode not in (MODE_SHADOW, MODE_PRODUCTION):
            raise ValueError(
                f"mode must be {MODE_SHADOW!r} or {MODE_PRODUCTION!r}, "
                f"got {self.mode!r}"
            )

    @property
    def is_shadow(self) -> bool:
        return self.mode == MODE_SHADOW


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One scenario STATE (design §8.3).

    - ``spot_shock``: fractional, uniform across every underlying — the
      **beta = 1 assumption**, documented on the row;
    - ``spot_shock_by_ticker``: per-ticker override (a ticker absent keeps
      ``spot_shock``). Historical windows always populate this;
    - ``iv_shock``: RELATIVE multiplicative on the IV LEVEL (+0.20 ⇒
      ``iv1 = iv0 * 1.20``);
    - ``days_forward``: calendar days of decay, ≥ 0;
    - ``validated``: True only for a scenario whose parameterisation has
      been validated against data. The whole research grid is False;
    - ``source``: provenance (``"CATALOGUE"``, ``"STORED_CLOSES"``,
      ``"AUTO_WORST_WINDOW"``, ``"USER"``);
    - ``iv_shock_source``: ``"SPECIFIED"`` or ``"RV_PROXY"`` (spec §24);
    - ``health`` / ``reason``: an UNAVAILABLE scenario (e.g. a historical
      window outside the stored history) carries no shocks and produces an
      UNAVAILABLE result row rather than a fabricated 0.
    """

    name: str
    kind: str
    spot_shock: float = 0.0
    spot_shock_by_ticker: Mapping[str, float] = field(default_factory=dict)
    iv_shock: float = 0.0
    days_forward: float = 0.0
    validated: bool = False
    source: str = "CATALOGUE"
    iv_shock_source: str = IV_SHOCK_SOURCE_SPECIFIED
    notes: str | None = None
    health: ModelHealth = ModelHealth.ACTIVE
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if self.kind not in SCENARIO_KINDS:
            raise ValueError(
                f"{self.name}: kind must be one of {SCENARIO_KINDS}, got {self.kind!r}"
            )
        for fname in ("spot_shock", "iv_shock", "days_forward"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{self.name}: {fname} must be a finite number, got {v!r}")
        if self.spot_shock <= -1.0:
            raise ValueError(
                f"{self.name}: spot_shock must be > -1 (a stock cannot go to "
                f"zero or below), got {self.spot_shock}"
            )
        if self.iv_shock <= -1.0:
            raise ValueError(
                f"{self.name}: iv_shock must be > -1, got {self.iv_shock}"
            )
        if self.days_forward < 0.0:
            raise ValueError(
                f"{self.name}: days_forward must be >= 0, got {self.days_forward}"
            )
        shocks = dict(self.spot_shock_by_ticker)
        for tkr, v in shocks.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(
                    f"{self.name}: spot_shock_by_ticker[{tkr!r}] must be finite, got {v!r}"
                )
            if v <= -1.0:
                raise ValueError(
                    f"{self.name}: spot_shock_by_ticker[{tkr!r}] must be > -1, got {v}"
                )
        object.__setattr__(self, "spot_shock_by_ticker", shocks)

    @property
    def is_uniform(self) -> bool:
        """True when the scenario carries the beta = 1 uniform assumption."""
        return not self.spot_shock_by_ticker

    def params(self) -> dict[str, Any]:
        """The scenario's parameters as plain scalars, for persistence and
        the API (spec §44: every number reproducible)."""
        return {
            "spot_shock": self.spot_shock,
            "spot_shock_by_ticker": dict(self.spot_shock_by_ticker),
            "iv_shock": self.iv_shock,
            "iv_shock_source": self.iv_shock_source,
            "days_forward": self.days_forward,
            "uniform_beta_1": self.is_uniform,
            "source": self.source,
            "validated": self.validated,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HistoricalWindow:
    """A named window of real history (design §8.3), inclusive of both
    endpoints. ``start`` must be strictly before ``end``."""

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.start, date) or not isinstance(self.end, date):
            raise ValueError(f"{self.name}: start/end must be datetime.date")
        if self.start >= self.end:
            raise ValueError(
                f"{self.name}: start {self.start} must be < end {self.end}"
            )


# ---------------------------------------------------------------------------
# Historical shocks from stored closes
# ---------------------------------------------------------------------------


def _window_slice(
    bars: Sequence[tuple[date, float]], start: date, end: date
) -> list[tuple[date, float]]:
    """The bars with ``start <= d <= end``, in order."""
    return [(d, c) for (d, c) in bars if start <= d <= end]


def _realized_vol(closes: Sequence[float]) -> float | None:
    """Sample stdev (ddof=1) of the LOG returns of ``closes``.

    ``None`` when fewer than 2 returns exist. This is the plain
    unannualized per-observation figure — the proxy is a RATIO, so any
    common annualization factor cancels.
    """
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i] > 0.0 and closes[i - 1] > 0.0
    ]
    if len(rets) < 2:
        return None
    mean = math.fsum(rets) / len(rets)
    var = math.fsum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def historical_shocks_from_closes(
    window: HistoricalWindow,
    closes_by_ticker: Mapping[str, Sequence[tuple[date, float]]],
    *,
    params: HistoricalShockParams = HistoricalShockParams(),
) -> Scenario:
    """Turn a historical window into a :class:`Scenario` (design §8.3).

    Per ticker, the spot shock is the **cumulative simple return over the
    window**: ``close(last bar in window) / close(first bar in window) − 1``
    (equivalently ``prod(1 + r) − 1``, which is what the design names).

    The IV shock is the RV-ratio proxy, computed on the EQUAL-WEIGHT pooled
    evidence available: ``RV(window) / RV(prior rv_prior_days) − 1``,
    averaged across the tickers that have both, then clipped to
    ``[iv_shock_floor, iv_shock_ceiling]`` and labelled
    ``iv_shock_source="RV_PROXY"``. This platform stores no IV history
    (spec §24) — the label is the honesty.

    ``days_forward`` is the number of CALENDAR days the window spans, so a
    historical scenario decays time exactly as much as the window did.

    A ticker whose stored history does not COVER the window (its bars start
    after ``window.start`` or end before ``window.end``) is excluded and
    named in the reason. If NO ticker covers it, the scenario comes back
    ``health=UNAVAILABLE`` with the real dates and no shocks — never a
    partial window presented as the whole one.
    """
    shocks: dict[str, float] = {}
    rv_ratios: list[float] = []
    excluded: list[str] = []
    reasons: list[str] = []

    for ticker in sorted(closes_by_ticker):
        bars = sorted(closes_by_ticker[ticker], key=lambda b: b[0])
        if not bars:
            excluded.append(ticker)
            continue
        # Coverage: the stored history must span the whole window, else the
        # "cumulative return over the window" would be a different window.
        if bars[0][0] > window.start or bars[-1][0] < window.end:
            excluded.append(ticker)
            continue
        inside = _window_slice(bars, window.start, window.end)
        if len(inside) < params.min_window_obs:
            excluded.append(ticker)
            continue
        first_close = inside[0][1]
        last_close = inside[-1][1]
        if first_close <= 0.0 or last_close <= 0.0:
            raise ValueError(
                f"{ticker}: closes must be > 0 in {window.name} "
                f"({first_close}, {last_close})"
            )
        shocks[ticker] = last_close / first_close - 1.0

        # RV proxy: window RV vs the prior `rv_prior_days` bars' RV.
        prior = [(d, c) for (d, c) in bars if d < window.start][
            -params.rv_prior_days :
        ]
        rv_in = (
            _realized_vol([c for (_, c) in inside])
            if len(inside) >= params.min_rv_obs
            else None
        )
        rv_prior = (
            _realized_vol([c for (_, c) in prior])
            if len(prior) >= params.min_rv_obs
            else None
        )
        if rv_in is not None and rv_prior is not None and rv_prior > 0.0:
            rv_ratios.append(rv_in / rv_prior - 1.0)

    if not shocks:
        return Scenario(
            name=window.name,
            kind=KIND_HISTORICAL,
            validated=False,
            source="STORED_CLOSES",
            iv_shock_source=IV_SHOCK_SOURCE_RV_PROXY,
            health=ModelHealth.UNAVAILABLE,
            reason=(
                f"no stored history covers {window.start}..{window.end} "
                f"(tickers checked: {len(closes_by_ticker)}"
                + (f", excluded: {', '.join(excluded)}" if excluded else "")
                + ")"
            ),
        )

    if rv_ratios:
        raw = math.fsum(rv_ratios) / len(rv_ratios)
        iv_shock = min(max(raw, params.iv_shock_floor), params.iv_shock_ceiling)
        if iv_shock != raw:
            reasons.append(
                f"IV proxy {raw:+.3f} clipped to {iv_shock:+.3f} "
                f"[{params.iv_shock_floor:+g}, {params.iv_shock_ceiling:+g}]"
            )
    else:
        iv_shock = 0.0
        reasons.append(
            f"IV shock 0.0: fewer than min_rv_obs={params.min_rv_obs} bars "
            f"in the window or the prior {params.rv_prior_days} days"
        )
    if excluded:
        reasons.append(
            f"tickers without full coverage of the window, excluded: "
            f"{', '.join(sorted(excluded))}"
        )

    return Scenario(
        name=window.name,
        kind=KIND_HISTORICAL,
        spot_shock=0.0,
        spot_shock_by_ticker=shocks,
        iv_shock=iv_shock,
        days_forward=float((window.end - window.start).days),
        validated=False,
        source="STORED_CLOSES",
        iv_shock_source=IV_SHOCK_SOURCE_RV_PROXY,
        notes=(
            f"per-ticker cumulative return over {window.start}..{window.end}; "
            f"IV shock is an RV-ratio proxy (no IV history — spec §24)"
        ),
        health=ModelHealth.DEGRADED if excluded else ModelHealth.ACTIVE,
        reason="; ".join(reasons) if reasons else None,
    )


def auto_worst_windows(
    closes_by_ticker: Mapping[str, Sequence[tuple[date, float]]],
    *,
    lengths: Sequence[int] = (1, 5, 10),
) -> tuple[HistoricalWindow, ...]:
    """The worst rolling windows of the EQUAL-WEIGHT book (design §8.3).

    The book proxy is the equal-weight average of the tickers' simple
    returns on each date they ALL have (an inner join on dates — the same
    rule ``risk.returns.align`` uses, so no return is ever compounded
    across a gap). For each length ``L`` in ``lengths`` the cumulative
    equal-weight return of every consecutive ``L``-return block is scored
    and the MINIMUM one becomes a window
    ``HistoricalWindow(name, start=date of the bar BEFORE the block,
    end=date of the block's last return)`` — start is the bar the move is
    measured FROM, so :func:`historical_shocks_from_closes` recomputes the
    same move.

    Ties are resolved by the EARLIEST window (stable, deterministic —
    the library-wide tie rule). Lengths for which there are not enough
    returns are skipped silently: an absent window is honest, a fabricated
    one is not. Returns windows in ``lengths`` order, deduplicated by
    (start, end) keeping the first (shortest) name.
    """
    for L in lengths:
        if isinstance(L, bool) or not isinstance(L, int) or L < 1:
            raise ValueError(f"lengths must be ints >= 1, got {L!r}")
    if not closes_by_ticker:
        return ()

    # Inner-join the dates every ticker has.
    per: dict[str, dict[date, float]] = {}
    for ticker, bars in closes_by_ticker.items():
        m: dict[date, float] = {}
        for d, c in bars:
            if c <= 0.0:
                raise ValueError(f"{ticker}: close must be > 0 on {d}, got {c}")
            m[d] = c
        if m:
            per[ticker] = m
    if not per:
        return ()
    common = sorted(set.intersection(*(set(m) for m in per.values())))
    if len(common) < 2:
        return ()

    # Equal-weight simple return per common date (from the previous COMMON
    # date — the inner join means these are the returns the book actually
    # has evidence for).
    ew: list[float] = []
    for i in range(1, len(common)):
        d0, d1 = common[i - 1], common[i]
        rs = [per[t][d1] / per[t][d0] - 1.0 for t in sorted(per)]
        ew.append(math.fsum(rs) / len(rs))

    out: list[HistoricalWindow] = []
    seen: set[tuple[date, date]] = set()
    for L in lengths:
        if L > len(ew):
            continue
        best_i: int | None = None
        best_v = math.inf
        for i in range(0, len(ew) - L + 1):
            # Cumulative (compounded) equal-weight return of the block.
            cum = 1.0
            for r in ew[i : i + L]:
                cum *= 1.0 + r
            v = cum - 1.0
            if v < best_v:  # strict: earliest window wins a tie
                best_v, best_i = v, i
        if best_i is None:
            continue
        start = common[best_i]          # the bar the block is measured FROM
        end = common[best_i + L]        # the block's last return date
        if (start, end) in seen:
            continue
        seen.add((start, end))
        out.append(
            HistoricalWindow(
                name=f"AUTO worst {L}-day ({start}..{end}, {best_v * 100:+.2f}%)",
                start=start,
                end=end,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

#: Named historical windows (design §8.3). Windows outside the stored
#: history produce UNAVAILABLE rows with the real dates — they are kept in
#: the catalogue so the absence is visible rather than silent.
DEFAULT_HISTORICAL_WINDOWS: tuple[HistoricalWindow, ...] = (
    HistoricalWindow(
        name="2024-08-05 vol spike",
        start=date(2024, 7, 31),
        end=date(2024, 8, 5),
    ),
    HistoricalWindow(
        name="2025-04 tariff drawdown",
        start=date(2025, 4, 2),
        end=date(2025, 4, 8),
    ),
)

#: The hypothetical research grid (design §8.3; spec §24, §26).
#: **EVERY ROW IS ``validated=False`` — UNVALIDATED research
#: parameterisation.** Spec §24: "Do not blindly adopt these example
#: numbers." They exist to expose the SHAPE of the book's exposure (long
#: gamma vs short vega vs pure decay), not to predict a market.
DEFAULT_HYPOTHETICAL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="Equity -5% / IV +20%",
        kind=KIND_HYPOTHETICAL,
        spot_shock=-0.05,
        iv_shock=0.20,
        validated=False,
        notes="uniform beta=1 down move with a moderate vol bid",
    ),
    Scenario(
        name="Equity -10% / IV +40%",
        kind=KIND_HYPOTHETICAL,
        spot_shock=-0.10,
        iv_shock=0.40,
        validated=False,
        notes="uniform beta=1 sharp selloff with a vol spike",
    ),
    Scenario(
        name="Equity +5% / IV -15%",
        kind=KIND_HYPOTHETICAL,
        spot_shock=0.05,
        iv_shock=-0.15,
        validated=False,
        notes="uniform beta=1 relief rally; vol bleeds",
    ),
    Scenario(
        name="IV crush (flat, -40%)",
        kind=KIND_IV_GRID,
        spot_shock=0.0,
        iv_shock=-0.40,
        validated=False,
        notes="spot unchanged — isolates long-option vega exposure (spec §26)",
    ),
    Scenario(
        name="IV spike (flat, +50%)",
        kind=KIND_IV_GRID,
        spot_shock=0.0,
        iv_shock=0.50,
        validated=False,
        notes="spot unchanged — isolates short-option vega exposure",
    ),
    Scenario(
        name="Correlation convergence (all names -8%, IV +30%)",
        kind=KIND_HYPOTHETICAL,
        spot_shock=-0.08,
        iv_shock=0.30,
        validated=False,
        notes=(
            "the beta=1 uniform shock IS the correlation-to-1 assumption "
            "(spec §26 'Tech correlation -> 0.9'); diversification is "
            "deliberately switched off here"
        ),
    ),
    Scenario(
        name="Time decay only (+5 days)",
        kind=KIND_HYPOTHETICAL,
        spot_shock=0.0,
        iv_shock=0.0,
        days_forward=5.0,
        validated=False,
        notes="spot and IV unchanged — isolates theta",
    ),
)


def default_scenarios(
    closes_by_ticker: Mapping[str, Sequence[tuple[date, float]]] | None = None,
    *,
    windows: Sequence[HistoricalWindow] = DEFAULT_HISTORICAL_WINDOWS,
    hypothetical: Sequence[Scenario] = DEFAULT_HYPOTHETICAL_SCENARIOS,
    auto_lengths: Sequence[int] = (1, 5, 10),
    params: HistoricalShockParams = HistoricalShockParams(),
) -> tuple[Scenario, ...]:
    """The full catalogue for one book: named historical windows, the AUTO
    worst windows found in the stored history, then the research grid
    (design §8.3). Deterministic order — the API and the persisted rows
    depend on it.

    With no closes the historical half is skipped entirely (there is
    nothing to derive a shock from) and only the research grid comes back.
    """
    out: list[Scenario] = []
    if closes_by_ticker:
        for w in windows:
            out.append(historical_shocks_from_closes(w, closes_by_ticker, params=params))
        for w in auto_worst_windows(closes_by_ticker, lengths=auto_lengths):
            sc = historical_shocks_from_closes(w, closes_by_ticker, params=params)
            out.append(
                Scenario(
                    name=sc.name,
                    kind=sc.kind,
                    spot_shock=sc.spot_shock,
                    spot_shock_by_ticker=sc.spot_shock_by_ticker,
                    iv_shock=sc.iv_shock,
                    days_forward=sc.days_forward,
                    validated=sc.validated,
                    source="AUTO_WORST_WINDOW",
                    iv_shock_source=sc.iv_shock_source,
                    notes=sc.notes,
                    health=sc.health,
                    reason=sc.reason,
                )
            )
    out.extend(hypothetical)
    return tuple(out)


# ---------------------------------------------------------------------------
# Running the stress
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's outcome (design §8.3).

    ``pnl_usd`` is gain-positive, so a stress LOSS is negative.
    ``pnl_pct_nav`` is a FRACTION of NAV (−0.031 = −3.1 %), ``None`` when
    NAV is unknown or ≤ 0. An UNAVAILABLE row carries ``None`` P&L and a
    ``reason`` with the real numbers — never a fabricated 0.
    """

    name: str
    kind: str
    validated: bool
    pnl_usd: float | None
    pnl_pct_nav: float | None
    per_key: Mapping[str, float]
    method_coverage: Mapping[str, int]
    health: ModelHealth
    reason: str | None
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_key", dict(self.per_key))
        object.__setattr__(self, "method_coverage", dict(self.method_coverage))
        object.__setattr__(self, "params", dict(self.params))

    @property
    def loss_usd(self) -> float | None:
        """The scenario LOSS (positive = money lost), the sign VaR/ES use."""
        return None if self.pnl_usd is None else -self.pnl_usd


@dataclass(frozen=True)
class StressResult:
    """The whole stress run (design §8.3).

    - ``rows``: one :class:`ScenarioResult` per scenario, in catalogue
      order;
    - ``worst``: the row with the SMALLEST ``pnl_usd`` among rows that
      produced a number; ``None`` when none did. Ties keep catalogue order;
    - ``min_pnl_usd``: that row's P&L (``None`` when there is no row);
    - ``health``: the worst health across the rows — ACTIVE only when every
      row computed, DEGRADED when some fell back or were excluded,
      UNAVAILABLE when nothing computed;
    - ``catalogue_version`` / ``model_version``: reproducibility (spec §44).
    """

    rows: tuple[ScenarioResult, ...]
    worst: ScenarioResult | None
    health: ModelHealth
    min_pnl_usd: float | None
    reason: str | None = None
    catalogue_version: str = CATALOGUE_VERSION
    model_version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def worst_loss_usd(self) -> float | None:
        """The worst LOSS (positive = money lost); ``None`` when unknown."""
        return None if self.min_pnl_usd is None else -self.min_pnl_usd

    @property
    def worst_loss_pct_nav(self) -> float | None:
        if self.worst is None or self.worst.pnl_pct_nav is None:
            return None
        return -self.worst.pnl_pct_nav


def _health_rank(h: ModelHealth) -> int:
    order = {
        ModelHealth.ACTIVE: 0,
        ModelHealth.DEGRADED: 1,
        ModelHealth.UNAVAILABLE: 2,
        ModelHealth.FAILED: 3,
    }
    return order[h]


def run_scenario(
    stock_legs: Sequence[StockLeg],
    option_legs: Sequence[OptionLeg],
    scenario: Scenario,
    *,
    spot0_by_ticker: Mapping[str, float] | None = None,
    nav: float | None = None,
) -> ScenarioResult:
    """Revalue the book under ONE scenario (design §8.3).

    An UNAVAILABLE scenario (a historical window with no coverage) is
    passed through as an UNAVAILABLE row with its reason, without pricing
    anything. A book with no legs at all yields an exact ``0.0`` P&L —
    that is the true value of an empty book, not a missing one.
    """
    if scenario.health is ModelHealth.UNAVAILABLE:
        return ScenarioResult(
            name=scenario.name,
            kind=scenario.kind,
            validated=scenario.validated,
            pnl_usd=None,
            pnl_pct_nav=None,
            per_key={},
            method_coverage={METHOD_FULL_REVAL: 0, METHOD_DELTA_LINEAR: 0},
            health=ModelHealth.UNAVAILABLE,
            reason=scenario.reason or "scenario unavailable",
            params=scenario.params(),
        )
    try:
        sp = scenario_pnl(
            stock_legs,
            option_legs,
            spot0_by_ticker=spot0_by_ticker,
            spot_shock_by_ticker=scenario.spot_shock_by_ticker,
            spot_shock=scenario.spot_shock,
            iv_shock=scenario.iv_shock,
            days_forward=scenario.days_forward,
        )
    except ValueError as exc:  # malformed leg/scenario combination
        return ScenarioResult(
            name=scenario.name,
            kind=scenario.kind,
            validated=scenario.validated,
            pnl_usd=None,
            pnl_pct_nav=None,
            per_key={},
            method_coverage={METHOD_FULL_REVAL: 0, METHOD_DELTA_LINEAR: 0},
            health=ModelHealth.FAILED,
            reason=f"revaluation failed: {exc}",
            params=scenario.params(),
        )

    pct = None
    if nav is not None and math.isfinite(nav) and nav > 0.0:
        pct = sp.total / nav

    reasons = [r for r in (scenario.reason,) if r]
    reasons.extend(sp.notes)
    health = scenario.health
    if sp.method_coverage.get(METHOD_DELTA_LINEAR, 0) > 0:
        health = (
            ModelHealth.DEGRADED
            if _health_rank(ModelHealth.DEGRADED) > _health_rank(health)
            else health
        )
    return ScenarioResult(
        name=scenario.name,
        kind=scenario.kind,
        validated=scenario.validated,
        pnl_usd=sp.total,
        pnl_pct_nav=pct,
        per_key=sp.per_key,
        method_coverage=sp.method_coverage,
        health=health,
        reason="; ".join(reasons) if reasons else None,
        params=scenario.params(),
    )


def run_stress(
    stock_legs: Sequence[StockLeg],
    option_legs: Sequence[OptionLeg],
    scenarios: Sequence[Scenario],
    *,
    spot0_by_ticker: Mapping[str, float] | None = None,
    nav: float | None = None,
) -> StressResult:
    """Run every scenario over the book (design §8.3).

    ``worst`` is the row with the smallest ``pnl_usd`` among rows that
    produced a number (ties keep catalogue order — the library-wide stable
    tie rule). With zero scenarios the result is an empty, UNAVAILABLE
    :class:`StressResult` with a reason, not an ACTIVE zero.
    """
    rows = tuple(
        run_scenario(
            stock_legs,
            option_legs,
            sc,
            spot0_by_ticker=spot0_by_ticker,
            nav=nav,
        )
        for sc in scenarios
    )
    if not rows:
        return StressResult(
            rows=(),
            worst=None,
            health=ModelHealth.UNAVAILABLE,
            min_pnl_usd=None,
            reason="no scenarios supplied",
        )
    priced = [r for r in rows if r.pnl_usd is not None]
    worst: ScenarioResult | None = None
    for r in priced:
        if worst is None or r.pnl_usd < worst.pnl_usd:  # type: ignore[operator]
            worst = r
    # Run-level health is the WORST health among the scenarios that PRICED
    # (a DELTA_LINEAR-only leg or an excluded position degrades the run).
    # A named historical window that simply lies outside the stored history
    # is an UNAVAILABLE ROW with its real dates — it must not drag every
    # priced scenario down to DEGRADED/UNAVAILABLE (QA finding, Phase D):
    # such rows are counted in ``reason`` instead.
    reason = None
    if not priced:
        health = ModelHealth.UNAVAILABLE
        reason = f"no scenario produced a number ({len(rows)} rows, all unavailable)"
    else:
        health = max((r.health for r in priced), key=_health_rank)
        if len(priced) < len(rows):
            reason = f"{len(rows) - len(priced)} of {len(rows)} scenarios unavailable"
    return StressResult(
        rows=rows,
        worst=worst,
        health=health,
        min_pnl_usd=None if worst is None else worst.pnl_usd,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# The hypothetical STRESS cap (SHADOW)
# ---------------------------------------------------------------------------


def stress_caps(
    candidate_stock_legs: Sequence[StockLeg],
    candidate_option_legs: Sequence[OptionLeg],
    book_stock_legs: Sequence[StockLeg],
    book_option_legs: Sequence[OptionLeg],
    scenarios: Sequence[Scenario],
    *,
    requested_qty: int,
    nav: float,
    spot0_by_ticker: Mapping[str, float] | None = None,
    limits: StressLimits = StressLimits(),
) -> tuple[list[QuantityCap], ModelHealth, str | None]:
    """The hypothetical STRESS cap for a proposed trade (design §8.3).

    The candidate legs are described PER UNIT of quantity (one contract of
    each option leg, one round lot / N shares of stock); a quantity ``q``
    scales every candidate leg by ``q`` (:meth:`OptionLeg.scaled`). For
    each ``q`` the worst scenario is re-evaluated over ``book + candidate*q``
    and the limit is

        ``worst_loss_usd <= max_stress_loss_pct_nav * nav``

    The largest passing ``q`` comes from the SAME verified bisection helper
    Phase C uses (``pretrade._largest_passing``): ≤ 20 probes, every
    non-zero answer CHECKED against the limit, with the step-down guard for
    a non-monotone corner. Reusing it is deliberate — one search, one set
    of guarantees, one place to fix.

    Health/fail-open (design §8.3, same open item as Phase C): when the
    stress view cannot produce a number at the requested quantity —
    ``nav <= 0``, no scenarios, or every scenario UNAVAILABLE — the
    function returns **no caps** with an UNAVAILABLE health and a reason.
    A missing view never produces a cap in SHADOW; the PRODUCTION
    promotion decides the fail-closed rule.

    Returns ``([QuantityCap] or [], health, reason)``; the cap is emitted
    only when it actually binds (``cap_qty < requested_qty``), matching
    Phase C's "a limit that is satisfied produces no cap".

    Import note: ``pretrade`` is imported HERE, inside the function, not at
    module level. ``pretrade`` imports ``models.*`` (it reuses the Phase B
    estimators), so a module-level import back into it would make the
    dependency circular and leave the package's import order load-bearing.
    A function-local import keeps the arrow one-way — ``models`` never
    depends on ``pretrade`` to be importable — which is the same rule the
    package docstring states for ``correlation`` and ``engine``.
    """
    from ..pretrade import QuantityCap as _QuantityCap, _largest_passing

    if isinstance(requested_qty, bool) or not isinstance(requested_qty, int) or requested_qty < 0:
        raise ValueError(f"requested_qty must be an int >= 0, got {requested_qty!r}")
    if not scenarios:
        return [], ModelHealth.UNAVAILABLE, "no scenarios supplied"
    if not math.isfinite(nav) or nav <= 0.0:
        return [], ModelHealth.UNAVAILABLE, f"nav={nav!r} is not > 0"
    if requested_qty == 0:
        return [], ModelHealth.UNAVAILABLE, "requested_qty=0 — nothing to cap"

    budget_usd = limits.max_stress_loss_pct_nav * nav

    def _worst_at(q: int) -> StressResult:
        legs_s = [*book_stock_legs, *(l.scaled(q) for l in candidate_stock_legs)]
        legs_o = [*book_option_legs, *(l.scaled(q) for l in candidate_option_legs)]
        return run_stress(
            legs_s,
            legs_o,
            scenarios,
            spot0_by_ticker=spot0_by_ticker,
            nav=nav,
        )

    at_requested = _worst_at(requested_qty)
    if at_requested.worst is None or at_requested.worst_loss_usd is None:
        return (
            [],
            ModelHealth.UNAVAILABLE,
            at_requested.reason or "no scenario produced a number",
        )

    loss_requested = at_requested.worst_loss_usd
    if loss_requested <= budget_usd:
        return [], at_requested.health, at_requested.reason

    def _passes(q: int) -> bool:
        res = _worst_at(q)
        loss = res.worst_loss_usd
        if loss is None:
            return False
        return loss <= budget_usd

    cap_qty = _largest_passing(requested_qty, _passes)
    at_cap = _worst_at(cap_qty) if cap_qty > 0 else None
    loss_cap = at_cap.worst_loss_usd if at_cap is not None else None

    sentence = (
        f"Stress: worst scenario '{at_requested.worst.name}' loses "
        f"${loss_requested:,.0f} ({loss_requested / nav * 100:.2f}% of NAV) at "
        f"{requested_qty} unit(s), above the "
        f"{limits.max_stress_loss_pct_nav * 100:.1f}% of NAV limit "
        f"(${budget_usd:,.0f}); largest quantity within the limit is {cap_qty}"
        + (
            f" (worst loss ${loss_cap:,.0f})"
            if loss_cap is not None
            else " (the whole trade would breach it)"
        )
        + ". SHADOW — this changes no decision."
    )
    cap = _QuantityCap(
        code=CODE_STRESS_LOSS,
        layer=LAYER_STRESS,
        cap_qty=cap_qty,
        sentence=sentence,
        measured={
            "worst_loss_usd_at_requested": loss_requested,
            "worst_loss_pct_nav_at_requested": loss_requested / nav,
            "worst_loss_usd_at_cap": loss_cap,
            "worst_loss_pct_nav_at_cap": None if loss_cap is None else loss_cap / nav,
            "budget_usd": budget_usd,
            "limit_pct_nav": limits.max_stress_loss_pct_nav,
            "requested_qty": float(requested_qty),
            "cap_qty": float(cap_qty),
        },
    )
    return [cap], at_requested.health, at_requested.reason


__all__ = [
    "CATALOGUE_VERSION",
    "CODE_STRESS_LOSS",
    "DEFAULT_HISTORICAL_WINDOWS",
    "DEFAULT_HYPOTHETICAL_SCENARIOS",
    "IV_SHOCK_SOURCE_RV_PROXY",
    "IV_SHOCK_SOURCE_SPECIFIED",
    "KIND_HISTORICAL",
    "KIND_HYPOTHETICAL",
    "KIND_IV_GRID",
    "KIND_USER",
    "LAYER_STRESS",
    "MODEL_NAME",
    "MODEL_TIER",
    "MODEL_VERSION",
    "MODE_PRODUCTION",
    "MODE_SHADOW",
    "SCENARIO_KINDS",
    "HistoricalShockParams",
    "HistoricalWindow",
    "Scenario",
    "ScenarioResult",
    "StressLimits",
    "StressResult",
    "auto_worst_windows",
    "default_scenarios",
    "historical_shocks_from_closes",
    "run_scenario",
    "run_stress",
    "stress_caps",
]
