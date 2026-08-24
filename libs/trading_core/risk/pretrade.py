"""Pre-trade portfolio risk — CURRENT vs AFTER TRADE, statistical caps and
the hypothetical SHADOW verdict (risk spec §8, §9, §11, §14, §19, §33, §37,
§38, §46, §47, §70; Phase B design contract §7.1–§7.2).

Pure, deterministic, stdlib-only (house rule). SHADOW by construction:
nothing here calls, patches or is called by :func:`~libs.trading_core.risk
.engine.assess` — a caller that wants a statistical cap to BIND must pass
the :class:`QuantityCap` rows explicitly through ``assess(extra_caps=...)``,
and :class:`StatisticalLimits` records ``mode="SHADOW"`` until a human
promotes it (spec §70; audit §11 Q3). Every threshold is a parameter on
:class:`StatisticalLimits` — the defaults are RESEARCH DEFAULTS, UNVALIDATED.

What this module adds on top of Phase B, and what it deliberately does NOT
re-implement: the estimators. VaR/ES/σ/ES-contributions/incremental ES all
come from ``models.var_es``, ``models.volatility`` and
``models.contribution`` — the ONE tail convention (``k = ceil(n(1−α))``,
``tail_size``) and the ONE quantile estimator are theirs, so the "before"
and "after" numbers on one comparison are mutually coherent (contract §3).

The three questions this module answers (spec §8, §9, §46):

1. **What does the book look like WITH the trade?** :func:`proposed_book`
   builds the joined :class:`~libs.trading_core.risk.pnl_series.BookPnl` —
   current per-position series plus the candidate priced DELTA_LINEAR at a
   given quantity, on the SAME dates.
2. **What changes?** :func:`compare` reports before/after pairs for VaR-95,
   ES-95, VaR-99, ES-99, Gaussian ES-95 and σ, plus incremental ES
   (``ES(after) − ES(before)`` — exactly, both recomputed on the joined
   series with the same ``k``), marginal ES per unit (the candidate's Euler
   ES contribution ÷ quantity), the candidate's ES share after, the worst
   single-position share before/after, per-bucket ES shares, and net delta
   notional. Tier 0's own heat/cash numbers are passed in by the caller —
   this module never recomputes a Tier 0 quantity.
3. **How large could the trade be and still respect the statistical
   limits?** :func:`statistical_caps` bisects on quantity per limit and
   returns :class:`QuantityCap` rows; :func:`shadow_verdict` folds them
   into the hypothetical APPROVE / APPROVE_WITH_RESIZE / REJECT the
   statistical layer ALONE would have produced at the Tier 0 approved
   quantity (spec §47).

Health rules (contract §7.2, honest nulls — a missing view is never a
fabricated 0):

- the candidate's ticker has no column in the matrix, or the book has
  ``n_obs < limits.min_obs`` ⇒ the comparison is ``UNAVAILABLE`` with a
  ``reason`` carrying the real numbers, and :func:`statistical_caps`
  returns NO caps. A statistical view that could not be computed must never
  produce a cap: in SHADOW that is fail-open, and it is deliberate — the
  PRODUCTION promotion design decides the fail-closed rules (audit §11 Q3,
  recorded as an open item);
- a pair whose ``before`` or ``after`` is not ACTIVE/DEGRADED reports
  ``delta_usd = delta_pct_nav = None`` rather than a difference of nulls.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .models.base import ModelHealth, ModelResult
from .models.contribution import (
    ContributionParams,
    ContributionResult,
    es_contributions,
    incremental_es,
    marginal_es,
)
from .models.var_es import (
    gaussian_es,
    historical_es,
    historical_var,
    tail_size,
)
from .models.volatility import portfolio_volatility
from .pnl_series import (
    BookPnl,
    PositionRiskInput,
    book_method_summary,
    book_pnl_series,
    position_pnl_series,
)
from .returns import ReturnMatrix

#: The confidence grid this module compares on (contract §7.1). 0.95 drives
#: every limit; 0.99 is reported for the §46 table only.
CONFIDENCE_95 = 0.95
CONFIDENCE_99 = 0.99

#: Cap layer names (contract §7.2 / §7.3, §8.3). ``HARD_LIMIT`` belongs to
#: the Tier 0 engine and is defined there; these are the SHADOW layers.
#: ``STRESS`` is Phase D's (``models.stress`` emits it) and lives here so
#: :class:`QuantityCap` — the ONE cap shape ``assess(extra_caps=...)``
#: understands — validates every layer the platform can produce in one
#: place, rather than each layer's module maintaining its own vocabulary.
LAYER_STATISTICAL = "STATISTICAL"
LAYER_CONCENTRATION = "CONCENTRATION"
LAYER_STRESS = "STRESS"

#: Every layer a :class:`QuantityCap` may carry.
CAP_LAYERS = (LAYER_STATISTICAL, LAYER_CONCENTRATION, LAYER_STRESS)

#: Cap codes (contract §7.2). Bucket caps append ``:<BUCKET>``.
CODE_PORTFOLIO_ES = "PORTFOLIO_ES_LIMIT"
CODE_ES_CONTRIBUTION = "ES_CONTRIBUTION_CAP"
CODE_BUCKET_ES_CONTRIBUTION = "BUCKET_ES_CONTRIBUTION_CAP"
CODE_INCREMENTAL_ES = "INCREMENTAL_ES_CAP"

#: Hypothetical decisions of the statistical layer alone (spec §47).
DECISION_APPROVE = "APPROVE"
DECISION_APPROVE_WITH_RESIZE = "APPROVE_WITH_RESIZE"
DECISION_REJECT = "REJECT"

#: §55: the shadow verdict when the snapshot the caps were measured on is
#: STALE per its own ``TtlPolicy``. Deliberately NOT one of the three Tier 0
#: decision words: a stale statistical view has not decided to approve, to
#: resize OR to reject — it has failed to answer, and saying APPROVE would
#: be the fail-OPEN this vocabulary exists to make visible.
#:
#: SHADOW-ONLY. The Tier 0 ladder never sees this value: `assess()` is not
#: passed `extra_caps` at either production call site, so suppressing shadow
#: caps here removes a hypothetical, not a control. Promoting the statistical
#: layer would make this the fail-CLOSED seam, and choosing that rule is the
#: separate, explicit user decision recorded in design §7.5.
DECISION_UNAVAILABLE_STALE = "UNAVAILABLE_STALE"

#: ``mode`` value that keeps this layer out of every Tier 0 decision (§70).
MODE_SHADOW = "SHADOW"
MODE_PRODUCTION = "PRODUCTION"

#: Bisection budget of the cap search (contract §7.2: "≤ 20 steps").
#: 20 halvings resolve any quantity up to 2^20 = 1 048 576 exactly.
MAX_BISECTION_STEPS = 20


# ---------------------------------------------------------------------------
# The candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """The proposed entry, described PER UNIT of quantity (contract §7.1).

    One "unit" is one share for stock and one contract for options — the
    same unit ``assess()`` approves — so every per-unit field multiplies by
    the quantity under test and the cap search can bisect on an integer.

    - ``key``: the position key the candidate would occupy (e.g.
      ``"AAPL#candidate"``); must not collide with a key already in the
      book (:func:`proposed_book` raises);
    - ``ticker``: the underlying whose returns drive it;
    - ``instrument``: an ``InstrumentType`` value, carried for labelling;
    - ``multiplier``: 1 for stock, 100 for standard equity options;
    - ``spot``: the underlying's last close, > 0;
    - ``delta``: per-share and SIGNED exactly as in
      :class:`~libs.trading_core.risk.pnl_series.PositionRiskInput` — short
      stock −1, a short premium leg already negated by the caller as in
      ``greeks.py``. This module never re-signs anything;
    - ``max_loss_per_unit``: the Tier 0 risk basis per unit (stop × gap for
      stock, premium × 100 for long options, net debit × 100 for a debit
      spread, ``(strike − credit) × 100`` for a CSP, 0 for a covered call);
    - ``capital_per_unit``: the cash outlay / reservation per unit;
    - ``quantity_requested``: what the user asked for, ``>= 0`` — the upper
      end of every cap search.

    OPTIONAL OPTION LEG FIELDS (design §10.1, additive — all default to
    ``None``, and ``None`` anywhere keeps the candidate DELTA_LINEAR exactly
    as before): ``strike``, ``right``, ``t_years``, ``iv0``, ``mark0``,
    passed straight through to :meth:`position_at` so the candidate's
    contribution to the proposed book's VaR/ES/RC is full-revalued on the
    SAME estimator the book's own option rows use. Validation is
    ``PositionRiskInput``'s — a malformed leg is rejected there, not
    silently downgraded.

    ``vega0`` (spec §46, additive) is the per-share vega of the same
    selected contract, used ONLY for the net-vega before/after row — it
    enters no estimator and no cap.

    ONE LEG PER CANDIDATE, by construction. A candidate is ONE
    ``PositionRiskInput`` under ONE key: ``proposed_book`` writes
    ``per_position[candidate.key]``, ``compare`` reads
    ``after_book.per_position[candidate.key]`` and every ES-share cap calls
    ``share_of(candidate.key)``. A two-leg spread representation would need
    two keys and would break all three, so a net-debit spread passes its
    LONG leg's fields (the dominant leg of a debit spread) with the short
    leg's offsetting convexity absent — see :meth:`position_at`.

    ``ValueError`` on malformed input (the same contract as
    ``PositionRiskInput``: multiplier ≥ 1, finite spot > 0, finite delta,
    finite non-negative bases, integral non-negative quantity).
    """

    key: str
    ticker: str
    instrument: str
    multiplier: int
    spot: float
    delta: float
    max_loss_per_unit: float
    capital_per_unit: float
    quantity_requested: int
    # --- optional option leg (design §10.1); None ⇒ DELTA_LINEAR ----------
    strike: float | None = None
    right: str | None = None
    t_years: float | None = None
    iv0: float | None = None
    mark0: float | None = None
    #: PER-SHARE vega (spec §46 net-vega row), signed the same way ``delta``
    #: is: the chain's value AS-IS for a long leg, the SHORT leg already
    #: negated by the caller, and the NET of the two for a spread. ``None``
    #: (stock, or a chain that gave no vega) means "not measurable", and
    #: every net-vega number downstream is then an honest null rather than a
    #: zero — a missing vega is not a vega of zero.
    vega0: float | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be a non-empty string")
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, int) or self.multiplier < 1:
            raise ValueError(
                f"{self.key}: multiplier must be an int >= 1, got {self.multiplier!r}"
            )
        if not math.isfinite(self.spot) or self.spot <= 0:
            raise ValueError(
                f"{self.key}: spot must be a finite number > 0, got {self.spot}"
            )
        if not math.isfinite(self.delta):
            raise ValueError(f"{self.key}: delta must be finite, got {self.delta}")
        for name in ("max_loss_per_unit", "capital_per_unit"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(
                    f"{self.key}: {name} must be finite and >= 0, got {v}"
                )
        if self.vega0 is not None and not math.isfinite(self.vega0):
            raise ValueError(f"{self.key}: vega0 must be finite or None, got {self.vega0}")
        q = self.quantity_requested
        if isinstance(q, bool) or not isinstance(q, int) or q < 0:
            raise ValueError(
                f"{self.key}: quantity_requested must be an int >= 0, got {q!r}"
            )

    def position_at(self, quantity: int) -> PositionRiskInput:
        """The candidate as a :class:`PositionRiskInput` held at ``quantity``
        units — the exact shape ``pnl_series`` prices (contract §2.9).

        ``max_loss`` is ``max_loss_per_unit × quantity`` (carried for the
        snapshot; the estimator does not read it). ``quantity < 0`` is
        malformed: a SHORT candidate carries its sign in ``delta``, as
        everywhere else in this library.

        The optional leg fields ride along unchanged, so a candidate that
        carries them is priced FULL_REVAL_CONST_IV at every quantity the cap
        search tries, and one that does not stays DELTA_LINEAR — the
        dispatch is ``PositionRiskInput``'s and is never re-decided here.

        SPREAD CAVEAT (design §10.1, documented rather than hidden): a
        net-debit spread is ONE candidate key carrying its LONG leg's
        strike/right/tenor/IV/mark and its NET delta. The revaluation
        therefore sees the long leg's convexity but not the short leg's
        offsetting negative convexity, so a spread candidate's modelled
        convexity is an UPPER bound on the true spread's. The direction is
        the conservative one for a risk cap (more measured convexity ⇒ more
        measured tail ⇒ a tighter hypothetical cap), and it is SHADOW
        either way. Splitting the candidate into two keyed legs is what
        would fix it, and that is exactly what ``proposed_book`` /
        ``share_of(candidate.key)`` cannot represent today.
        """
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            raise ValueError(
                f"{self.key}: quantity must be an int >= 0, got {quantity!r} "
                "(a short candidate carries its sign in delta)"
            )
        return PositionRiskInput(
            key=self.key,
            ticker=self.ticker,
            instrument=self.instrument,
            quantity=quantity,
            multiplier=self.multiplier,
            spot=self.spot,
            delta=self.delta,
            max_loss=self.max_loss_per_unit * quantity,
            strike=self.strike,
            right=self.right,
            t_years=self.t_years,
            iv0=self.iv0,
            mark0=self.mark0,
        )

    def exposure_at(self, quantity: int) -> float:
        """Delta-adjusted dollar exposure at ``quantity`` units —
        ``quantity × multiplier × delta × spot`` (the ``greeks.py`` term)."""
        return quantity * self.multiplier * self.delta * self.spot

    def vega_at(self, quantity: int) -> float | None:
        """The candidate's contribution to the book's NET VEGA at
        ``quantity`` units — ``quantity × multiplier × vega0`` (spec §46).

        ``None`` when ``vega0`` is ``None``: a candidate whose vega could
        not be read is UNMEASURED, and returning 0.0 would let a real vega
        exposure disappear into a net that looked complete. Stock carries
        ``vega0 = None`` for the same reason it carries no option leg —
        with one honest exception the caller may make explicitly by passing
        ``vega0=0.0``, which IS a measurement (stock has no vega).

        Units are ``$ per one IV point``, matching
        ``greeks.PortfolioGreeks.net_vega`` exactly, so the two sides of
        the §46 row are the same quantity and their difference is real.
        """
        if self.vega0 is None:
            return None
        return quantity * self.multiplier * self.vega0


def proposed_book(
    book: BookPnl,
    candidate: CandidateSpec,
    quantity: int,
    returns: ReturnMatrix,
) -> BookPnl:
    """The book AS IF the candidate were held at ``quantity`` (contract §7.1).

    The current per-position series are carried through unchanged and the
    candidate's DELTA_LINEAR series
    (``exposure = quantity × multiplier × delta × spot``; ``pnl_t =
    exposure × r_t``) is added under ``candidate.key``, so ``total`` is the
    joined portfolio P&L on the SAME dates.

    LABELLING (design §10.3): ``method_by_key`` is the book's map plus the
    candidate's own label, and ``method`` is :func:`book_method_summary` of
    the joined map — so a DELTA_LINEAR book joined to a full-revalued option
    candidate reports FULL_REVAL_CONST_IV, which is what actually priced the
    series. The candidate's label comes from ``position_at(quantity)
    .pnl_method``, the one dispatch predicate, so the label can never
    disagree with the series beside it.

    Honest gaps (contract §7.2): a candidate whose ticker has no column in
    ``returns`` cannot be priced — it is EXCLUDED, its ticker joins
    ``tickers_missing`` and its key joins ``keys_excluded``, the total is
    the unchanged book, and it carries NO ``method_by_key`` entry (it was
    priced by nothing). :func:`compare` turns that into UNAVAILABLE.

    ``ValueError`` when ``quantity < 0``, when the matrix dates disagree
    with the book's, or when ``candidate.key`` already exists in the book
    (a candidate never silently replaces a position).
    """
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
        raise ValueError(f"quantity must be an int >= 0, got {quantity!r}")
    if tuple(book.dates) != tuple(returns.dates):
        raise ValueError(
            f"book has {len(book.dates)} dates ending {book.as_of}, matrix has "
            f"{returns.n_obs} ending {returns.as_of} — the candidate must be "
            "priced on the same dates as the book"
        )
    if candidate.key in book.per_position:
        raise ValueError(
            f"candidate key {candidate.key!r} already exists in the book; "
            "a candidate never replaces an existing position"
        )

    per_position = {k: list(v) for k, v in book.per_position.items()}
    missing = list(book.tickers_missing)
    excluded = list(book.keys_excluded)
    # The book's own labels are CARRIED THROUGH (its series are carried
    # through unchanged, so its labels must be too) and the candidate adds
    # its own — the estimator that actually priced it, from the SAME
    # `PositionRiskInput.pnl_method` dispatch predicate, never re-derived
    # here (design §10.1/§10.3).
    method_by_key = dict(book.method_by_key)

    if candidate.ticker not in returns.tickers:
        if candidate.ticker not in missing:
            missing.append(candidate.ticker)
        excluded.append(candidate.key)
        total = list(book.total)
    else:
        # `position_pnl_series` enforces the SIMPLE-returns contract.
        position = candidate.position_at(quantity)
        series = position_pnl_series(position, returns)
        per_position[candidate.key] = series
        method_by_key[candidate.key] = position.pnl_method
        total = [b + c for b, c in zip(book.total, series)]

    return BookPnl(
        dates=book.dates,
        per_position=per_position,
        total=total,
        # The §10.3 summary of the JOINED book, not the pre-trade book's
        # stale label: a DELTA_LINEAR book plus a full-revalued option
        # candidate is a FULL_REVAL_CONST_IV book, and reporting it as
        # DELTA_LINEAR would mislabel a series that really was revalued.
        method=book_method_summary(method_by_key),
        tickers_missing=tuple(sorted(missing)),
        keys_excluded=tuple(excluded),
        method_by_key=method_by_key,
    )


# ---------------------------------------------------------------------------
# The comparison (spec §46)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricPair:
    """One §46 table row: the metric BEFORE and AFTER the trade (contract §7.1).

    ``delta_usd = after.value − before.value`` and ``delta_pct_nav =
    delta_usd / nav`` — both ``None`` unless BOTH sides carry a value with
    health ACTIVE or DEGRADED (a difference of nulls is not a number).
    """

    before: ModelResult | None
    after: ModelResult | None
    delta_usd: float | None
    delta_pct_nav: float | None

    @property
    def is_available(self) -> bool:
        return self.delta_usd is not None


def _usable(result: ModelResult | None) -> float | None:
    """The value of a ``ModelResult`` when it is usable, else ``None``.

    Usable = ACTIVE or DEGRADED with a real value; UNAVAILABLE / FAILED are
    honest gaps and never enter an arithmetic difference (contract §7.1).
    """
    value = getattr(result, "value", None)
    health = getattr(result, "health", None)
    if value is None or health not in (ModelHealth.ACTIVE, ModelHealth.DEGRADED):
        return None
    return float(value)


def _pair(
    before: ModelResult | None, after: ModelResult | None, nav: float
) -> MetricPair:
    """Build a :class:`MetricPair`, computing the delta only when both sides
    are usable (contract §7.1)."""
    b = _usable(before) if before is not None else None
    a = _usable(after) if after is not None else None
    if b is None or a is None:
        return MetricPair(before=before, after=after, delta_usd=None, delta_pct_nav=None)
    delta = a - b
    return MetricPair(before=before, after=after, delta_usd=delta, delta_pct_nav=delta / nav)


@dataclass(frozen=True)
class RiskComparison:
    """CURRENT vs AFTER TRADE at ONE quantity (spec §46; contract §7.1).

    Tier 0 numbers (``heat_pct``, ``cash_pct``) are ``(before, after)``
    tuples supplied BY THE CALLER — the statistical layer reads Tier 0, it
    never recomputes it. Everything else is measured here on the book and
    the proposed book.

    - ``incremental_es_95_usd``: ``ES_95(after) − ES_95(before)``, both
      recomputed with the same ``n`` and ``k`` (spec §8 ``ΔES``);
    - ``marginal_es_95_per_unit``: the candidate's Euler ES-95 contribution
      on the JOINED tail ÷ ``quantity`` — "how much ES one more unit adds"
      (spec §9). ``None`` at ``quantity == 0`` (no unit to be marginal in);
    - ``candidate_es_share_after``: the candidate's share of the ES-95
      contributions of the proposed book (spec §11 gate basis);
    - ``max_single_es_share_before`` / ``_after``: the largest single-
      position share on each side (spec §46 "Single-name RC");
    - ``bucket_es_share_after``: ES-95 share of each bucket the candidate
      belongs to, keyed by bucket name, computed on the proposed book;
    - ``incremental_var_95_usd`` / ``_pct_nav``: ``VaR_95(after) −
      VaR_95(before)`` (spec §8 ``ΔVaR``), the same estimator discipline
      as the ES increment beside it;
    - ``net_vega``: ``(before, after)`` book net vega in $ per IV point
      (spec §46); ``before`` is the caller's greeks number and ``after``
      adds the candidate's vega at ``quantity``;
    - ``net_delta_notional``: ``(before, after)`` delta-adjusted notional;
      ``before`` is the caller's Tier 0 / greeks number (``None`` when it
      could not be computed) and ``after`` adds the candidate's exposure at
      ``quantity`` — ``None`` when ``before`` is ``None`` (no guessing);
    - ``health`` / ``reason``: UNAVAILABLE with real numbers when the
      candidate has no returns or ``n_obs < min_obs``; DEGRADED when the
      sample is thin; ACTIVE otherwise.
    """

    quantity: int
    heat_pct: tuple[float, float]
    cash_pct: tuple[float, float]
    var_hist_95: MetricPair
    es_hist_95: MetricPair
    var_hist_99: MetricPair
    es_hist_99: MetricPair
    gaussian_es_95: MetricPair
    volatility: MetricPair
    incremental_es_95_usd: float | None
    incremental_es_95_pct_nav: float | None
    marginal_es_95_per_unit: float | None
    candidate_es_share_after: float | None
    max_single_es_share_before: float | None
    max_single_es_share_after: float | None
    bucket_es_share_after: Mapping[str, float]
    net_delta_notional: tuple[float | None, float | None]
    health: ModelHealth
    reason: str | None
    n_obs: int
    tail_size_95: int | None
    contributions_es_95_after: ContributionResult | None = None
    #: §8 (ADDITIVE): incremental VaR-95, first-class beside incremental ES.
    #: ``VaR_95(after) - VaR_95(before)``, both HISTORICAL on the same ``n``
    #: and the same ``k`` -- the SAME estimator discipline incremental ES
    #: uses, so the two increments are comparable to each other. ``None``
    #: whenever either side is an honest gap.
    incremental_var_95_usd: float | None = None
    incremental_var_95_pct_nav: float | None = None
    #: §46 (ADDITIVE): the book's NET VEGA before and after, in
    #: ``$ per one IV point``. ``before`` is the caller's greeks number
    #: (``None`` when it could not be computed) and ``after`` adds the
    #: candidate's ``vega_at(quantity)``. ``after`` is ``None`` when EITHER
    #: side is unmeasured -- a net that silently drops the candidate's vega
    #: would read as "this trade adds no vol exposure".
    net_vega: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        object.__setattr__(self, "bucket_es_share_after", dict(self.bucket_es_share_after))

    @property
    def is_available(self) -> bool:
        return self.health in (ModelHealth.ACTIVE, ModelHealth.DEGRADED)


def _max_share(result: ContributionResult | None) -> float | None:
    """Largest single-position ``share`` of a contribution result, or ``None``
    when the result is unavailable or every share is ``None`` (total ≤ 0)."""
    if result is None or not result.per_position:
        return None
    shares = [row.share for row in result.per_position if row.share is not None]
    return max(shares) if shares else None


def _bucket_shares(
    result: ContributionResult | None,
    keys_by_bucket: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """ES share of each named bucket = ``Σ contributions of its keys ÷ total``.

    Buckets with no key in the result, and a non-positive total (share is
    meaningless), are omitted rather than reported as 0.0.
    """
    if result is None or result.total is None or result.total <= 0.0:
        return {}
    by_key = {row.key: row.contribution for row in result.per_position}
    shares: dict[str, float] = {}
    for bucket, keys in keys_by_bucket.items():
        members = [by_key[k] for k in keys if k in by_key]
        if members:
            shares[bucket] = math.fsum(members) / result.total
    return shares


def _net_vega_after(
    net_vega_before: float | None,
    candidate: CandidateSpec,
    quantity: int,
) -> float | None:
    """Book net vega AFTER the trade — ``before + candidate.vega_at(q)``.

    Honest null in BOTH directions (spec §46; contract §7.1):

    - ``net_vega_before is None`` (the caller could not read the book's
      greeks) ⇒ ``None``. Reporting the candidate's own vega as the book
      net would claim the rest of the book has none.
    - ``candidate.vega0 is None`` (the chain gave no vega, or this is a
      stock candidate that carries none) ⇒ ``None`` as well, because the
      difference between "before" and an "after" that silently omits the
      candidate would read as "this trade adds no vol exposure" — the one
      wrong answer this row exists to prevent.

    A candidate that genuinely has zero vega says so with ``vega0 = 0.0``,
    which is a measurement and passes straight through.
    """
    if net_vega_before is None:
        return None
    candidate_vega = candidate.vega_at(quantity)
    if candidate_vega is None:
        return None
    return net_vega_before + candidate_vega


def _candidate_buckets(
    candidate: CandidateSpec,
    positions: Sequence[PositionRiskInput],
    buckets: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Position KEYS per bucket the CANDIDATE belongs to (contract §7.1).

    ``buckets`` maps a bucket name to its member TICKERS (the shape of
    ``RiskLimits.correlation_buckets`` and of the dynamic buckets); this
    resolves them to the position keys the contribution rows are keyed on,
    and includes the candidate's own key. Only buckets containing the
    candidate's ticker are reported — a bucket the trade cannot touch has
    no bearing on this trade's caps.
    """
    out: dict[str, list[str]] = {}
    by_ticker: dict[str, list[str]] = {}
    for pos in positions:
        by_ticker.setdefault(pos.ticker, []).append(pos.key)
    for name, members in buckets.items():
        if candidate.ticker not in members:
            continue
        keys: list[str] = []
        for ticker in members:
            keys.extend(by_ticker.get(ticker, ()))
        keys.append(candidate.key)
        out[name] = keys
    return out


def compare(
    book: BookPnl,
    candidate: CandidateSpec,
    quantity: int,
    *,
    returns: ReturnMatrix,
    nav: float,
    heat_before: float,
    heat_after: float,
    cash_before: float,
    cash_after: float,
    positions: Sequence[PositionRiskInput] = (),
    buckets: Mapping[str, Sequence[str]] | None = None,
    delta_notional_before: float | None = None,
    net_vega_before: float | None = None,
    limits: StatisticalLimits | None = None,
    contribution_params: ContributionParams | None = None,
) -> RiskComparison:
    """CURRENT vs AFTER TRADE at ``quantity`` (spec §46; contract §7.1).

    ``before`` metrics are computed on ``book.total``, ``after`` on
    ``proposed_book(...).total`` — same ``min_obs``, same ``k``, same 1-day
    horizon on both sides, so the deltas are comparable by construction.
    The ES-95 contributions AFTER are computed on the joined per-position
    dict (the candidate's key included), which is what makes
    ``Σ_i RC_i == ES_95(after)`` hold exactly (contract §3.3) and the shares
    the §11 gate reads meaningful.

    ``positions`` is the book's :class:`PositionRiskInput` rows — needed
    only to resolve ``buckets`` (ticker members) to the position keys the
    contribution rows carry. ``delta_notional_before`` is the caller's
    current delta-adjusted notional; ``after`` adds
    ``candidate.exposure_at(quantity)``.

    ``net_vega_before`` (spec §46) is the caller's CURRENT book net vega —
    ``greeks.PortfolioGreeks.net_vega``, $ per one IV point. This module
    never aggregates greeks itself: the book's vega is already computed on
    the decision path, and recomputing it here from a different source is
    how two surfaces come to disagree about one number. ``after`` adds
    ``candidate.vega_at(quantity)``, and is an honest ``None`` when either
    side is unmeasured.

    Health: the candidate ticker missing from ``returns`` or
    ``n_obs < limits.min_obs`` ⇒ UNAVAILABLE with the real numbers and every
    pair carrying its honest ``None``s; the per-metric health of each side
    still comes from the estimators themselves (contract §2.3 bands).
    """
    if nav <= 0:
        raise ValueError(f"nav must be > 0, got {nav}")
    limits = limits if limits is not None else StatisticalLimits()
    cparams = contribution_params if contribution_params is not None else ContributionParams()
    buckets = buckets or {}

    after_book = proposed_book(book, candidate, quantity, returns)
    n = book.n_obs
    as_of = book.as_of
    k95 = tail_size(n, CONFIDENCE_95) if n else None

    def _unavailable(reason: str) -> RiskComparison:
        empty = MetricPair(None, None, None, None)
        return RiskComparison(
            quantity=quantity,
            heat_pct=(heat_before, heat_after),
            cash_pct=(cash_before, cash_after),
            var_hist_95=empty,
            es_hist_95=empty,
            var_hist_99=empty,
            es_hist_99=empty,
            gaussian_es_95=empty,
            volatility=empty,
            incremental_es_95_usd=None,
            incremental_es_95_pct_nav=None,
            marginal_es_95_per_unit=None,
            candidate_es_share_after=None,
            max_single_es_share_before=None,
            max_single_es_share_after=None,
            bucket_es_share_after={},
            net_delta_notional=(delta_notional_before, None),
            health=ModelHealth.UNAVAILABLE,
            reason=reason,
            n_obs=n,
            tail_size_95=k95,
            contributions_es_95_after=None,
            incremental_var_95_usd=None,
            incremental_var_95_pct_nav=None,
            # The net-vega row does NOT depend on the return series, so it
            # survives an UNAVAILABLE statistical view: a book with no
            # priceable history still has a real, readable vega today.
            net_vega=(net_vega_before, _net_vega_after(net_vega_before, candidate, quantity)),
        )

    if candidate.key in after_book.keys_excluded:
        return _unavailable(
            f"candidate ticker {candidate.ticker!r} has no returns column "
            f"(matrix covers {list(returns.tickers)}); no statistical view"
        )
    if n < limits.min_obs:
        return _unavailable(f"n={n} < min_obs={limits.min_obs}")

    before_pnl = book.total
    after_pnl = after_book.total
    min_obs = limits.min_obs

    var95 = _pair(
        historical_var(before_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        historical_var(after_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        nav,
    )
    es95 = _pair(
        historical_es(before_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        historical_es(after_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        nav,
    )
    var99 = _pair(
        historical_var(before_pnl, CONFIDENCE_99, min_obs=min_obs, as_of=as_of),
        historical_var(after_pnl, CONFIDENCE_99, min_obs=min_obs, as_of=as_of),
        nav,
    )
    es99 = _pair(
        historical_es(before_pnl, CONFIDENCE_99, min_obs=min_obs, as_of=as_of),
        historical_es(after_pnl, CONFIDENCE_99, min_obs=min_obs, as_of=as_of),
        nav,
    )
    ges95 = _pair(
        gaussian_es(before_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        gaussian_es(after_pnl, CONFIDENCE_95, min_obs=min_obs, as_of=as_of),
        nav,
    )
    vol = _pair(
        portfolio_volatility(before_pnl, min_obs=min_obs, as_of=as_of),
        portfolio_volatility(after_pnl, min_obs=min_obs, as_of=as_of),
        nav,
    )

    # Incremental ES — ES(after) − ES(before), the SAME arithmetic the pair
    # above reports, delegated to the library function so there is one
    # definition (contract §2.5).
    cand_pnl = after_book.per_position[candidate.key]
    inc = incremental_es(
        before_pnl, cand_pnl, CONFIDENCE_95, min_obs=min_obs, params=cparams, as_of=as_of
    )
    inc_usd = inc.delta
    inc_pct = inc_usd / nav if inc_usd is not None else None

    # §8: incremental VaR-95, first-class beside the ES increment. Computed
    # from the SAME two `MetricPair` sides already measured above (`var95`),
    # so it is `VaR(after) - VaR(before)` by construction and can never
    # disagree with the row the table renders. `_pair` already refused the
    # subtraction when either side was UNAVAILABLE/FAILED, which is exactly
    # the honest-null rule the ES increment follows.
    inc_var_usd = var95.delta_usd
    inc_var_pct = inc_var_usd / nav if inc_var_usd is not None else None

    # Marginal ES per unit — the candidate's Euler ES contribution on the
    # joined tail ÷ quantity. Undefined at q = 0 (honest null, not 0.0).
    marginal: float | None = None
    if quantity > 0:
        m = marginal_es(
            cand_pnl,
            before_pnl,
            CONFIDENCE_95,
            quantity,
            min_obs=min_obs,
            params=cparams,
            as_of=as_of,
        )
        marginal = _usable(m)

    contrib_before = es_contributions(
        book.per_position, CONFIDENCE_95, min_obs=min_obs, params=cparams, as_of=as_of
    ) if book.per_position else None
    contrib_after = es_contributions(
        after_book.per_position, CONFIDENCE_95, min_obs=min_obs, params=cparams, as_of=as_of
    )

    keys_by_bucket = _candidate_buckets(candidate, positions, buckets)

    delta_after: float | None = None
    if delta_notional_before is not None:
        delta_after = delta_notional_before + candidate.exposure_at(quantity)

    health = ModelHealth.ACTIVE
    reason: str | None = None
    if es95.before is not None and getattr(es95.before, "health", None) is ModelHealth.DEGRADED:
        health = ModelHealth.DEGRADED
        reason = getattr(es95.before, "reason", None) or f"small sample: n={n}"

    return RiskComparison(
        quantity=quantity,
        heat_pct=(heat_before, heat_after),
        cash_pct=(cash_before, cash_after),
        var_hist_95=var95,
        es_hist_95=es95,
        var_hist_99=var99,
        es_hist_99=es99,
        gaussian_es_95=ges95,
        volatility=vol,
        incremental_es_95_usd=inc_usd,
        incremental_es_95_pct_nav=inc_pct,
        marginal_es_95_per_unit=marginal,
        candidate_es_share_after=contrib_after.share_of(candidate.key),
        max_single_es_share_before=_max_share(contrib_before),
        max_single_es_share_after=_max_share(contrib_after),
        bucket_es_share_after=_bucket_shares(contrib_after, keys_by_bucket),
        net_delta_notional=(delta_notional_before, delta_after),
        health=health,
        reason=reason,
        n_obs=n,
        tail_size_95=k95,
        contributions_es_95_after=contrib_after,
        incremental_var_95_usd=inc_var_usd,
        incremental_var_95_pct_nav=inc_var_pct,
        net_vega=(
            net_vega_before,
            _net_vega_after(net_vega_before, candidate, quantity),
        ),
    )


# ---------------------------------------------------------------------------
# Statistical limits and hypothetical caps (spec §11, §37)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticalLimits:
    """The statistical layer's thresholds (contract §7.2).

    **RESEARCH DEFAULTS — UNVALIDATED.** Spec §11 is explicit: "Do NOT
    choose arbitrary production thresholds silently. Define research
    defaults and validate them." These numbers are starting points for the
    Q3 shadow window (audit §11 Q3), not calibrated limits, and every one of
    them is a parameter — never a hardcoded truth (house rule).

    - ``max_portfolio_es95_pct_nav`` (0.05): ES-95 1D of the WHOLE book
      after the trade ≤ 5 % of NAV;
    - ``max_single_position_es_share`` (0.35): no single position may hold
      more than 35 % of the ES-95 contributions after the trade;
    - ``max_bucket_es_share`` (0.50): no correlation bucket the candidate
      belongs to may hold more than 50 %;
    - ``max_incremental_es95_pct_nav`` (0.015): one trade may add at most
      1.5 % of NAV of ES-95;
    - ``min_obs`` (60): below this the statistical view is UNAVAILABLE and
      produces NO cap;
    - ``mode`` (``"SHADOW"``): only ``"PRODUCTION"`` may be wired into
      ``assess`` (spec §70; promotion is an explicit human action).
    """

    max_portfolio_es95_pct_nav: float = 0.05
    max_single_position_es_share: float = 0.35
    max_bucket_es_share: float = 0.50
    max_incremental_es95_pct_nav: float = 0.015
    min_obs: int = 60
    mode: str = MODE_SHADOW

    def __post_init__(self) -> None:
        for name in (
            "max_portfolio_es95_pct_nav",
            "max_single_position_es_share",
            "max_bucket_es_share",
            "max_incremental_es95_pct_nav",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
        if isinstance(self.min_obs, bool) or not isinstance(self.min_obs, int) or self.min_obs < 2:
            raise ValueError(f"min_obs must be an int >= 2, got {self.min_obs!r}")
        if self.mode not in (MODE_SHADOW, MODE_PRODUCTION):
            raise ValueError(
                f"mode must be {MODE_SHADOW!r} or {MODE_PRODUCTION!r}, got {self.mode!r}"
            )

    @property
    def is_shadow(self) -> bool:
        return self.mode == MODE_SHADOW


@dataclass(frozen=True)
class QuantityCap:
    """One statistical cap in the shape ``assess(extra_caps=...)`` applies
    (contract §7.2 / §7.3).

    - ``code``: the reason-code stem — the engine records ``code`` when the
      cap zeroes the quantity and ``RESIZED_BY_<code>`` when it reduces it,
      exactly like every Tier 0 cap;
    - ``layer``: one of :data:`CAP_LAYERS` (``"STATISTICAL"``,
      ``"CONCENTRATION"`` or Phase D's ``"STRESS"``) — what
      ``binding_constraints`` reports next to the code, so the UI can
      separate a hard-limit resize from a statistical or stress one
      (spec §47);
    - ``cap_qty``: the largest quantity in ``[0, requested]`` that satisfies
      the limit; 0 means "this limit alone would REJECT";
    - ``sentence``: the §47-style explanation with the real numbers;
    - ``measured``: the values behind the sentence, at the requested
      quantity and at ``cap_qty`` (``None`` where a number was unavailable).

    ``ValueError`` on a negative ``cap_qty``, an unknown layer or an empty
    code/sentence — an unexplained cap must never reach a decision.
    """

    code: str
    layer: str
    cap_qty: int
    sentence: str
    measured: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must be a non-empty string")
        if self.layer not in CAP_LAYERS:
            raise ValueError(
                f"layer must be one of {CAP_LAYERS}, got {self.layer!r}"
            )
        if isinstance(self.cap_qty, bool) or not isinstance(self.cap_qty, int) or self.cap_qty < 0:
            raise ValueError(f"cap_qty must be an int >= 0, got {self.cap_qty!r}")
        if not self.sentence:
            raise ValueError(f"{self.code}: sentence must be a non-empty string")
        object.__setattr__(self, "measured", dict(self.measured))


def _es95_at(
    book: BookPnl,
    candidate: CandidateSpec,
    quantity: int,
    returns: ReturnMatrix,
    min_obs: int,
    cparams: ContributionParams,
) -> tuple[ContributionResult, float | None]:
    """``(ES-95 contributions of the proposed book at q, ES_95 total)``.

    One evaluation of the joined series — every limit below reads what it
    needs from it, so the bisection costs one recomputation per step, not
    one per limit (contract §7.2: "cheap on ≤ 600 obs").
    """
    after = proposed_book(book, candidate, quantity, returns)
    contrib = es_contributions(
        after.per_position,
        CONFIDENCE_95,
        min_obs=min_obs,
        params=cparams,
        as_of=book.as_of,
    )
    return contrib, contrib.total


def _largest_passing(requested: int, passes: Callable[[int], bool]) -> int:
    """Largest ``q ∈ [0, requested]`` with ``passes(q)`` true (contract §7.2).

    Bisection assuming monotonicity — a limit that a quantity satisfies is
    assumed to be satisfied by every smaller quantity — in at most
    :data:`MAX_BISECTION_STEPS` steps, followed by the STEP-DOWN GUARD the
    contract requires: the assumption is only an assumption, so the
    candidate answer is VERIFIED and, if the check fails there
    (non-monotone corner), ``q`` is decremented until it passes or reaches
    0. The step-down is bounded by the same budget; if it never finds a
    passing quantity, the honest answer is 0 (the cap rejects) — never an
    unverified quantity.

    ``passes(0)`` is not assumed: an empty trade always satisfies a limit
    about what the trade ADDS, and if it somehow does not, 0 is still the
    smallest thing this cap can return.

    Worked example of the guard. A limit whose only passing quantities are
    ``{1}`` on a request of 10 is not monotone. Bisection probes 10 (fail),
    5 (fail), 2 (fail), 1 (pass) — here ``lo`` IS verified and 1 comes back
    directly. Change the passing set to ``{0, 1, 2, 3, 4, 6, 7}`` and
    bisection probes 10, 5, 2, 3, 4 and returns 4: the true maximum 7 is
    invisible to a bisection that saw 10 and 5 fail, and 4 is a VERIFIED
    quantity that satisfies the limit — under-approving is the safe error,
    approving an unverified quantity is not.

    What the guard prevents is the third shape: bisection can finish with
    ``lo = 0`` never probed (every ``mid`` failed, so only ``hi`` ever
    moved). Returning that 0 as "the answer" would be an unverified claim.
    The guard instead walks DOWN from the largest quantity bisection has
    not ruled out (``hi − 1``), verifying each, and returns the first that
    passes — 0 when the whole remaining interval fails.

    The invariant this buys, and the one the cap search actually needs:
    **every non-zero quantity this function returns has been CHECKED
    against the limit.** A cap that hands back a breaching quantity is
    worse than no cap at all, so the search under-approves rather than
    guess. It does NOT promise the global maximum on a non-monotone limit
    — no ≤ 20-probe search can — and the ``{0,1,2,3,4,6,7}`` example above
    shows it returning 4 where 7 was also legal. Both loops are bounded by
    :data:`MAX_BISECTION_STEPS`.
    """
    if requested <= 0:
        return 0
    if passes(requested):
        return requested
    lo, hi = 0, requested  # target: the answer lies in [lo, hi)
    lo_verified = False
    for _ in range(MAX_BISECTION_STEPS):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        if passes(mid):
            lo, lo_verified = mid, True
        else:
            hi = mid
    if lo_verified:
        return lo
    # Step-down guard (contract §7.2). ``lo`` was never probed (bisection
    # only ever moved ``hi`` down), so returning it would be an unverified
    # answer — exactly the non-monotone corner the contract calls out. Walk
    # DOWN from the largest quantity still in play, verifying each, and
    # return the first that passes; 0 when none does.
    q = min(hi - 1, requested)
    for _ in range(MAX_BISECTION_STEPS):
        if q <= 0:
            return 0
        if passes(q):
            return q
        q -= 1
    return 0


def statistical_caps(
    book: BookPnl,
    candidate: CandidateSpec,
    *,
    returns: ReturnMatrix,
    nav: float,
    positions: Sequence[PositionRiskInput] = (),
    buckets: Mapping[str, Sequence[str]] | None = None,
    limits: StatisticalLimits | None = None,
    contribution_params: ContributionParams | None = None,
) -> tuple[list[QuantityCap], ModelHealth, str | None]:
    """Hypothetical quantity caps from the statistical limits (contract §7.2).

    For each limit: evaluate at ``q = candidate.quantity_requested``; if the
    limit is satisfied there, the limit produces NO cap. Otherwise
    :func:`_largest_passing` bisects on ``q ∈ [0, requested]`` (≤ 20 steps)
    and verifies the answer, stepping down on a non-monotone corner.

    The four limits (spec §11, §37):

    1. ``PORTFOLIO_ES_LIMIT`` (STATISTICAL) — ES-95 of the proposed book ≤
       ``max_portfolio_es95_pct_nav × nav``;
    2. ``ES_CONTRIBUTION_CAP`` (CONCENTRATION) — the candidate's ES-95 share
       ≤ ``max_single_position_es_share``;
    3. ``BUCKET_ES_CONTRIBUTION_CAP:<BUCKET>`` (CONCENTRATION) — each bucket
       the candidate belongs to ≤ ``max_bucket_es_share``;
    4. ``INCREMENTAL_ES_CAP`` (STATISTICAL) — ``ES_95(after) − ES_95(before)``
       ≤ ``max_incremental_es95_pct_nav × nav``.

    Returns ``(caps, health, reason)``. **A statistical view that could not
    be computed NEVER produces a cap**: candidate ticker missing from the
    matrix, or ``n_obs < limits.min_obs``, or an ES-95 that is not
    ACTIVE/DEGRADED ⇒ ``([], UNAVAILABLE, reason)``. In SHADOW that is
    fail-open by design; the PRODUCTION promotion decides the fail-closed
    rules (audit §11 Q3 — open item).

    A share limit is only meaningful when ES-95 is positive: with a
    non-positive tail (a book that gains even in its worst 5 % of days) the
    shares are ``None`` and the two concentration limits report no cap, with
    the reason recorded on the health string.
    """
    limits = limits if limits is not None else StatisticalLimits()
    cparams = contribution_params if contribution_params is not None else ContributionParams()
    buckets = buckets or {}
    if nav <= 0:
        raise ValueError(f"nav must be > 0, got {nav}")

    n = book.n_obs
    requested = candidate.quantity_requested
    if candidate.ticker not in returns.tickers:
        return (
            [],
            ModelHealth.UNAVAILABLE,
            f"candidate ticker {candidate.ticker!r} has no returns column "
            f"(matrix covers {list(returns.tickers)}); no cap can be computed",
        )
    if n < limits.min_obs:
        return [], ModelHealth.UNAVAILABLE, f"n={n} < min_obs={limits.min_obs}"

    min_obs = limits.min_obs
    contrib_req, es_req = _es95_at(book, candidate, requested, returns, min_obs, cparams)
    if contrib_req.health not in (ModelHealth.ACTIVE, ModelHealth.DEGRADED) or es_req is None:
        return (
            [],
            ModelHealth.UNAVAILABLE,
            contrib_req.reason or f"ES-95 contributions unavailable at n={n}",
        )

    es_before_res = historical_es(book.total, CONFIDENCE_95, min_obs=min_obs, as_of=book.as_of)
    es_before = _usable(es_before_res)
    health = (
        ModelHealth.DEGRADED
        if contrib_req.health is ModelHealth.DEGRADED
        else ModelHealth.ACTIVE
    )
    reason = contrib_req.reason if health is ModelHealth.DEGRADED else None
    caps: list[QuantityCap] = []

    def es_total_at(q: int) -> float | None:
        return _es95_at(book, candidate, q, returns, min_obs, cparams)[1]

    def contrib_at(q: int) -> ContributionResult:
        return _es95_at(book, candidate, q, returns, min_obs, cparams)[0]

    # --- 1. Portfolio ES-95 limit (STATISTICAL) ---------------------------
    es_cap_usd = limits.max_portfolio_es95_pct_nav * nav
    if es_req > es_cap_usd:
        def passes_portfolio(q: int) -> bool:
            v = es_total_at(q)
            return v is not None and v <= es_cap_usd

        cap_qty = _largest_passing(requested, passes_portfolio)
        at_cap = es_total_at(cap_qty)
        caps.append(
            QuantityCap(
                code=CODE_PORTFOLIO_ES,
                layer=LAYER_STATISTICAL,
                cap_qty=cap_qty,
                sentence=(
                    f"Portfolio ES-95 (1D) would be ${es_req:,.2f} "
                    f"({es_req / nav:.2%} of NAV) at {requested} unit(s) of "
                    f"{candidate.ticker}, above the "
                    f"{limits.max_portfolio_es95_pct_nav:.2%}-of-NAV limit "
                    f"(${es_cap_usd:,.2f}); quantity reduced from {requested} "
                    f"to {cap_qty}"
                    + (
                        f", where ES-95 is ${at_cap:,.2f} ({at_cap / nav:.2%} of NAV)."
                        if at_cap is not None
                        else "."
                    )
                ),
                measured={
                    "es95_usd_at_requested": es_req,
                    "es95_pct_nav_at_requested": es_req / nav,
                    "es95_usd_at_cap": at_cap,
                    "es95_pct_nav_at_cap": at_cap / nav if at_cap is not None else None,
                    "limit_usd": es_cap_usd,
                },
            )
        )

    # --- 2. Single-position ES share of the CANDIDATE (CONCENTRATION) -----
    share_req = contrib_req.share_of(candidate.key)
    if share_req is not None and share_req > limits.max_single_position_es_share:
        def passes_share(q: int) -> bool:
            s = contrib_at(q).share_of(candidate.key)
            return s is None or s <= limits.max_single_position_es_share

        cap_qty = _largest_passing(requested, passes_share)
        at_cap = contrib_at(cap_qty).share_of(candidate.key)
        caps.append(
            QuantityCap(
                code=CODE_ES_CONTRIBUTION,
                layer=LAYER_CONCENTRATION,
                cap_qty=cap_qty,
                sentence=(
                    f"{candidate.ticker} would hold {share_req:.1%} of the "
                    f"portfolio's ES-95 risk contributions at {requested} "
                    f"unit(s), above the "
                    f"{limits.max_single_position_es_share:.1%} single-position "
                    f"limit; quantity reduced from {requested} to {cap_qty}"
                    + (f", where its share is {at_cap:.1%}." if at_cap is not None else ".")
                ),
                measured={
                    "es_share_at_requested": share_req,
                    "es_share_at_cap": at_cap,
                    "limit": limits.max_single_position_es_share,
                },
            )
        )

    # --- 3. Bucket ES share (CONCENTRATION), one cap per bound bucket -----
    keys_by_bucket = _candidate_buckets(candidate, positions, buckets)
    for bucket in sorted(keys_by_bucket):
        keys = keys_by_bucket[bucket]
        bshare_req = _bucket_shares(contrib_req, {bucket: keys}).get(bucket)
        if bshare_req is None or bshare_req <= limits.max_bucket_es_share:
            continue

        def passes_bucket(q: int, _keys: Sequence[str] = keys, _b: str = bucket) -> bool:
            s = _bucket_shares(contrib_at(q), {_b: _keys}).get(_b)
            return s is None or s <= limits.max_bucket_es_share

        cap_qty = _largest_passing(requested, passes_bucket)
        at_cap = _bucket_shares(contrib_at(cap_qty), {bucket: keys}).get(bucket)
        caps.append(
            QuantityCap(
                code=f"{CODE_BUCKET_ES_CONTRIBUTION}:{bucket}",
                layer=LAYER_CONCENTRATION,
                cap_qty=cap_qty,
                sentence=(
                    f"The {bucket} bucket would hold {bshare_req:.1%} of the "
                    f"portfolio's ES-95 risk contributions at {requested} "
                    f"unit(s) of {candidate.ticker}, above the "
                    f"{limits.max_bucket_es_share:.1%} bucket limit; quantity "
                    f"reduced from {requested} to {cap_qty}"
                    + (f", where the bucket holds {at_cap:.1%}." if at_cap is not None else ".")
                ),
                measured={
                    "bucket_es_share_at_requested": bshare_req,
                    "bucket_es_share_at_cap": at_cap,
                    "limit": limits.max_bucket_es_share,
                },
            )
        )

    # --- 4. Incremental ES-95 (STATISTICAL) -------------------------------
    if es_before is not None:
        inc_cap_usd = limits.max_incremental_es95_pct_nav * nav
        inc_req = es_req - es_before
        if inc_req > inc_cap_usd:
            def passes_incremental(q: int) -> bool:
                v = es_total_at(q)
                return v is not None and (v - es_before) <= inc_cap_usd

            cap_qty = _largest_passing(requested, passes_incremental)
            v_cap = es_total_at(cap_qty)
            inc_at_cap = v_cap - es_before if v_cap is not None else None
            caps.append(
                QuantityCap(
                    code=CODE_INCREMENTAL_ES,
                    layer=LAYER_STATISTICAL,
                    cap_qty=cap_qty,
                    sentence=(
                        f"{candidate.ticker} at {requested} unit(s) would add "
                        f"${inc_req:,.2f} ({inc_req / nav:.2%} of NAV) of ES-95 "
                        f"(from ${es_before:,.2f} to ${es_req:,.2f}), above the "
                        f"{limits.max_incremental_es95_pct_nav:.2%}-of-NAV "
                        f"incremental limit (${inc_cap_usd:,.2f}); quantity "
                        f"reduced from {requested} to {cap_qty}"
                        + (
                            f", adding ${inc_at_cap:,.2f} ({inc_at_cap / nav:.2%} of NAV)."
                            if inc_at_cap is not None
                            else "."
                        )
                    ),
                    measured={
                        "es95_before_usd": es_before,
                        "incremental_usd_at_requested": inc_req,
                        "incremental_pct_nav_at_requested": inc_req / nav,
                        "incremental_usd_at_cap": inc_at_cap,
                        "limit_usd": inc_cap_usd,
                    },
                )
            )

    return caps, health, reason


# ---------------------------------------------------------------------------
# The hypothetical verdict (spec §47, §70)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowVerdict:
    """What the STATISTICAL layer ALONE would have decided (contract §7.2).

    Computed at the Tier 0 APPROVED quantity, so it answers "would the
    statistical layer have cut this further?" — never "what should Tier 0
    have done". ``mode`` is ``"SHADOW"``: this verdict is logged and
    displayed, and changes nothing (spec §70).

    - ``hypothetical_decision``: ``REJECT`` when the quantity falls to 0,
      ``APPROVE_WITH_RESIZE`` when it is reduced but positive, otherwise
      ``APPROVE`` (the Tier 0 vocabulary, so the UI reuses its translations);
    - ``hypothetical_quantity``: ``min(approved_qty, min cap_qty)``;
    - ``binding``: the codes of the caps that actually bind at
      ``approved_qty`` (``cap_qty < approved_qty``), MOST RESTRICTIVE FIRST
      (ascending ``cap_qty``; ties keep the caps' input order, which is the
      deterministic order :func:`statistical_caps` emits);
    - ``caps``: every cap considered, binding or not, in input order;
    - ``reason``: why the verdict reads as it does — set only for
      ``UNAVAILABLE_STALE`` (spec §55).

    ``UNAVAILABLE_STALE`` (spec §55) is the fourth decision: the snapshot the
    caps were measured on is older than its own TTL, so the caps describe a
    book that may no longer exist. They are SUPPRESSED rather than applied,
    and ``hypothetical_quantity`` falls back to the Tier 0 approved quantity
    — which is what actually happened, since this layer changed nothing.
    """

    hypothetical_decision: str
    hypothetical_quantity: int
    binding: tuple[str, ...]
    caps: tuple[QuantityCap, ...]
    mode: str = MODE_SHADOW
    #: §55 (ADDITIVE): why this verdict is what it is, when that needs
    #: saying. Non-empty exactly when the verdict is UNAVAILABLE_STALE —
    #: suppressing caps without a reason would be indistinguishable from
    #: having found none, which is the failure mode §55 is about.
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.hypothetical_decision not in (
            DECISION_APPROVE,
            DECISION_APPROVE_WITH_RESIZE,
            DECISION_REJECT,
            DECISION_UNAVAILABLE_STALE,
        ):
            raise ValueError(
                f"hypothetical_decision must be one of APPROVE / "
                f"APPROVE_WITH_RESIZE / REJECT / UNAVAILABLE_STALE, "
                f"got {self.hypothetical_decision!r}"
            )
        object.__setattr__(self, "binding", tuple(self.binding))
        object.__setattr__(self, "caps", tuple(self.caps))

    @property
    def changes_quantity(self) -> bool:
        """True when the statistical layer WOULD have cut the Tier 0 quantity."""
        return bool(self.binding)


def shadow_verdict(
    approved_qty: int,
    caps: Sequence[QuantityCap],
    *,
    mode: str = MODE_SHADOW,
    stale: bool = False,
    stale_reason: str | None = None,
) -> ShadowVerdict:
    """Fold the caps into the hypothetical statistical verdict (spec §47).

    ``hypothetical_quantity = min(approved_qty, min(cap.cap_qty))`` — a cap
    only ever REDUCES, never raises (the statistical layer cannot grant risk
    Tier 0 refused). With no caps the verdict is APPROVE at ``approved_qty``.

    Hand-check: ``approved_qty=10`` with caps ``[A(cap 7), B(cap 3),
    C(cap 12)]`` ⇒ quantity 3, decision APPROVE_WITH_RESIZE, binding
    ``("B", "A")`` — B first because 3 < 7, and C does not bind at all
    because 12 ≥ 10.

    ``stale`` (spec §55) — THE STALENESS CONSUMER. When the snapshot these
    caps were measured on is older than its own TTL, every cap is
    SUPPRESSED and the verdict is ``UNAVAILABLE_STALE`` at
    ``approved_qty``, carrying ``stale_reason``. The caps are still
    returned in ``caps`` (they were computed; hiding them would lose the
    evidence) but ``binding`` is empty, because nothing bound.

    WHY SUPPRESS RATHER THAN APPLY. A cap derived from a stale book is a
    statement about positions that may have changed — applying it would
    let an out-of-date measurement reduce a quantity, which is a worse
    error than declining to answer. In SHADOW this changes only the logged
    hypothetical. At PROMOTION the same seam is where a fail-CLOSED rule
    would go instead (refuse the trade rather than ignore the caps), and
    that choice is a user decision, not this function's — see design §7.5.
    """
    if isinstance(approved_qty, bool) or not isinstance(approved_qty, int) or approved_qty < 0:
        raise ValueError(f"approved_qty must be an int >= 0, got {approved_qty!r}")
    caps = tuple(caps)
    if stale:
        return ShadowVerdict(
            hypothetical_decision=DECISION_UNAVAILABLE_STALE,
            hypothetical_quantity=approved_qty,
            binding=(),
            caps=caps,
            mode=mode,
            reason=(
                stale_reason
                or "the snapshot these caps were measured on is stale per its "
                "TtlPolicy; caps suppressed rather than applied to a book "
                "that may have changed (spec §55, SHADOW)"
            ),
        )
    binding_caps = [c for c in caps if c.cap_qty < approved_qty]
    # Most restrictive first; ties keep emission order (sorted is stable).
    binding_caps.sort(key=lambda c: c.cap_qty)
    quantity = min([approved_qty, *(c.cap_qty for c in caps)]) if caps else approved_qty
    quantity = max(quantity, 0)
    if quantity <= 0:
        decision = DECISION_REJECT
    elif quantity < approved_qty:
        decision = DECISION_APPROVE_WITH_RESIZE
    else:
        decision = DECISION_APPROVE
    return ShadowVerdict(
        hypothetical_decision=decision,
        hypothetical_quantity=quantity,
        binding=tuple(c.code for c in binding_caps),
        caps=caps,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Sizing v2 — the SHADOW composition of the three unbuilt modifiers, and the
# risk-linked cash floor (spec §36, §37, §59; compliance §3 Tier A)
# ---------------------------------------------------------------------------
#
# THE GAP THIS CLOSES. Today the budget that sizes a trade composes exactly
# two things (`engine.py:490`):
#
#     budget = min(tier_budget(strength) x vol_multiplier, abs_max_trade_risk)
#
# Spec §37 asks for five factors, and `audit.md:226` (P1) committed to
# building the missing three — ES, correlation and model health — in SHADOW.
# Spec §59 separately committed to a model-risk BUDGET EFFECT ("label NOW,
# budget effect SHADOW", `audit.md:212`); only the label shipped, which is
# why the compliance report calls it "the most consequential gap in the
# programme": the 20-day shadow window was accumulating evidence about a
# hypothetical quantity that model risk could not move.
#
# `sizing_v2_shadow` below composes all three, beside the number Tier 0
# actually used, and `risk_linked_cash_floor_pct` composes §36's
# `Risk up -> Floor up` rule over the same inputs. Both are SHADOW by
# construction: this is a pure function of numbers a caller already has. It
# calls nothing, is called by no Tier 0 path, and returns a record. The
# caller LOGS it. Promotion would mean feeding `candidate_budget_pct` into
# `assess(budget_multiplier=...)` and the floor into
# `RiskLimits.cash_floors`, and that is an explicit human step after the
# shadow window (audit §11 Q3), not something this module can do to itself.


#: Correlation regimes (`libs.trading_core.correlation` STATE_* values,
#: restated as plain strings so this stdlib-only module needs no import of
#: the estimator to describe its own parameter table).
CORRELATION_NORMAL = "NORMAL"
CORRELATION_ELEVATED = "ELEVATED"
CORRELATION_CONVERGING = "CONVERGING"

#: Model-risk states (`models.ensemble` RISK_* values, same reasoning).
MODEL_RISK_LOW = "LOW"
MODEL_RISK_ELEVATED = "ELEVATED"
MODEL_RISK_HIGH = "HIGH"


def _default_correlation_modifiers() -> dict[str, float]:
    """§37 correlation modifier per §19 regime — RESEARCH DEFAULTS."""
    return {
        CORRELATION_NORMAL: 1.0,
        CORRELATION_ELEVATED: 0.85,
        CORRELATION_CONVERGING: 0.70,
    }


def _default_model_risk_modifiers() -> dict[str, float]:
    """§59 model-health modifier per model-risk state — RESEARCH DEFAULTS."""
    return {
        MODEL_RISK_LOW: 1.0,
        MODEL_RISK_ELEVATED: 0.85,
        MODEL_RISK_HIGH: 0.70,
    }


def _default_model_risk_floor_addons() -> dict[str, float]:
    """§36 cash-floor addon per model-risk state — RESEARCH DEFAULTS."""
    return {
        MODEL_RISK_LOW: 0.0,
        MODEL_RISK_ELEVATED: 0.05,
        MODEL_RISK_HIGH: 0.10,
    }


@dataclass(frozen=True)
class SizingV2Params:
    """Every threshold sizing v2 uses (house rule: never a hardcoded truth).

    **RESEARCH DEFAULTS — UNVALIDATED.** Not one of these numbers has been
    backtested. They are the starting points the Q3 shadow window measures
    (audit §11 Q3); spec §11 forbids choosing production thresholds silently,
    so they live here, are echoed with every result, and bind nothing.

    Budget side (§37, §59):

    - ``es_target_pct_nav`` (0.03): the ES-95 1-day fraction of NAV the book
      is sized TOWARD. Above it, the ES modifier shrinks the budget in
      proportion; at or below it the modifier is 1.0 (this is a THROTTLE, not
      a leverage rule — a quiet book is never sized UP);
    - ``es_modifier_floor`` (0.5): the ES modifier never cuts by more than
      half, so one noisy ES estimate cannot silently zero the budget;
    - ``correlation_modifiers``: §19 regime -> multiplier;
    - ``model_risk_modifiers``: §59 state -> multiplier. THIS mapping is the
      §59 budget effect the audit promised.

    Cash-floor side (§36):

    - the BASE floor is Tier 0's own ``cash_floors[regime]`` — passed in as
      ``regime_floor_pct``, never re-derived here, so the shadow floor can
      only ever sit ON TOP of the floor in force;
    - ``k_es`` (2.0): floor addon per unit of ES-95 ABOVE
      ``es_target_pct_nav`` (ES 4 % against a 3 % target -> 2.0 x 0.01 =
      0.02);
    - ``k_drawdown`` (0.5): floor addon per unit of |current drawdown|
      (-3 % -> 0.5 x 0.03 = 0.015);
    - ``model_risk_floor_addons``: flat addon per §59 state;
    - ``max_cash_floor_pct`` (0.90): the composed floor is capped, so no
      combination of unvalidated addons can demand a book be 100 % cash.

    ``mode`` (``"SHADOW"``) records that none of this reaches a decision.
    """

    es_target_pct_nav: float = 0.03
    es_modifier_floor: float = 0.5
    correlation_modifiers: Mapping[str, float] = field(
        default_factory=_default_correlation_modifiers
    )
    model_risk_modifiers: Mapping[str, float] = field(
        default_factory=_default_model_risk_modifiers
    )
    k_es: float = 2.0
    k_drawdown: float = 0.5
    model_risk_floor_addons: Mapping[str, float] = field(
        default_factory=_default_model_risk_floor_addons
    )
    max_cash_floor_pct: float = 0.90
    mode: str = MODE_SHADOW

    def __post_init__(self) -> None:
        for name in ("es_target_pct_nav", "k_es", "k_drawdown", "max_cash_floor_pct"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
        if self.es_target_pct_nav <= 0:
            raise ValueError(
                f"es_target_pct_nav must be > 0, got {self.es_target_pct_nav!r}"
            )
        if self.max_cash_floor_pct <= 0 or self.max_cash_floor_pct > 1:
            raise ValueError(
                f"max_cash_floor_pct must be in (0, 1], got {self.max_cash_floor_pct!r}"
            )
        floor = self.es_modifier_floor
        if (
            not isinstance(floor, (int, float))
            or isinstance(floor, bool)
            or not math.isfinite(floor)
            or not 0 < floor <= 1
        ):
            raise ValueError(
                f"es_modifier_floor must be a number in (0, 1], got {floor!r}"
            )
        for name in ("correlation_modifiers", "model_risk_modifiers"):
            table = getattr(self, name)
            frozen = dict(table)
            for state, value in frozen.items():
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or not 0 < value <= 1
                ):
                    raise ValueError(
                        f"{name}[{state!r}] must be a number in (0, 1] "
                        f"(a modifier may only THROTTLE), got {value!r}"
                    )
            object.__setattr__(self, name, frozen)
        addons = dict(self.model_risk_floor_addons)
        for state, value in addons.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"model_risk_floor_addons[{state!r}] must be a number "
                    f">= 0, got {value!r}"
                )
        object.__setattr__(self, "model_risk_floor_addons", addons)
        if self.mode not in (MODE_SHADOW, MODE_PRODUCTION):
            raise ValueError(
                f"mode must be {MODE_SHADOW!r} or {MODE_PRODUCTION!r}, got {self.mode!r}"
            )

    @property
    def is_shadow(self) -> bool:
        return self.mode == MODE_SHADOW


@dataclass(frozen=True)
class SizingV2Shadow:
    """The §37 composition and the §36 floor, beside what Tier 0 really did.

    Every field is either an INPUT echoed back or a number derived by the
    formulas in :func:`sizing_v2_shadow`'s docstring — nothing here is
    fetched, and nothing here is applied.

    - ``es_modifier`` / ``correlation_modifier`` / ``model_health_modifier``:
      the three missing §37 factors, each 1.0 (with a ``note``) when its
      input was unavailable — an unmeasured factor throttles NOTHING, which
      is the honest reading of a missing number in a SHADOW composition;
    - ``candidate_budget_pct``: what the risk budget WOULD be under v2;
    - ``budget_pct_used``: what Tier 0 actually used. The pair is the whole
      point — the shadow window compares these two columns;
    - ``budget_delta_pct`` : ``candidate_budget_pct − budget_pct_used``
      (negative = v2 would have sized SMALLER);
    - ``risk_linked_cash_floor_pct`` / ``risk_linked_cash_floor_binds``: the
      §36 floor and whether it sits ABOVE the regime floor in force;
    - ``health`` / ``reason``: ``ACTIVE`` when every input was present,
      ``DEGRADED`` when one or more was missing (with the real names in the
      reason), never a silent 1.0;
    - ``notes``: one honest sentence per missing input;
    - ``mode``: ``"SHADOW"``.
    """

    es_modifier: float
    correlation_modifier: float
    model_health_modifier: float
    candidate_budget_pct: float
    budget_pct_used: float
    budget_delta_pct: float
    risk_linked_cash_floor_pct: float
    risk_linked_cash_floor_binds: bool
    regime_floor_pct: float
    cash_floor_addons: Mapping[str, float]
    inputs: Mapping[str, object]
    health: str
    reason: str | None
    notes: tuple[str, ...]
    params: SizingV2Params
    mode: str = MODE_SHADOW

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "cash_floor_addons", dict(self.cash_floor_addons))


def sizing_v2_shadow(
    *,
    es95_pct_nav: float | None,
    correlation_state: str | None,
    model_risk_state: str | None,
    drawdown_current_pct: float | None,
    regime_floor_pct: float,
    tier_budget_pct: float,
    vol_multiplier_used: float,
    params: SizingV2Params | None = None,
) -> SizingV2Shadow:
    """Compose the §37 budget and the §36 cash floor — SHADOW, pure.

    **THE FORMULAS**, in full, so any line of this can be re-derived with a
    pocket calculator from the echoed ``inputs``:

    1. **ES modifier** (§37, the missing ES factor)::

           es_mod = 1.0                                   if es95 is None
           es_mod = 1.0                                   if es95 <= es_target
           es_mod = clamp(es_target / es95, floor, 1.0)   otherwise

       Hand-check: ``es95 = 0.04``, ``es_target = 0.03`` ->
       ``0.03 / 0.04 = 0.75``. With ``floor = 0.5``, an ES of 12 % gives
       ``0.03 / 0.12 = 0.25`` -> clamped to ``0.50``.

    2. **Correlation modifier** (§37, §19)::

           corr_mod = correlation_modifiers[correlation_state]   (1.0 if
                      the state is missing or not in the table)

       Defaults: NORMAL 1.0, ELEVATED 0.85, CONVERGING 0.70.

    3. **Model-health modifier** (§59 — the budget effect the audit
       promised)::

           mh_mod = model_risk_modifiers[model_risk_state]       (1.0 if
                    the state is missing or not in the table)

       Defaults: LOW 1.0, ELEVATED 0.85, HIGH 0.70.

    4. **The v2 candidate budget** (§37 composition, multiplicative — the
       same shape Tier 0 already uses for signal x vol, extended by three
       factors)::

           candidate_budget_pct = tier_budget_pct
                                x vol_multiplier_used
                                x es_mod x corr_mod x mh_mod

       Hand-check (the compliance-report example): tier 1 % (0.01), vol
       multiplier 0.8, ES 4 % vs a 3 % target (0.75), CONVERGING (0.70),
       model risk HIGH (0.70)::

           0.01 x 0.8 x 0.75 x 0.70 x 0.70 = 0.00294  = 0.294 % of NAV

       against ``budget_pct_used = tier_budget_pct x vol_multiplier_used``
       ``= 0.01 x 0.8 = 0.008`` = 0.8 % of NAV. NOTE that ``budget_pct_used``
       is reconstructed from the same two Tier 0 inputs and is NOT the
       engine's post-``abs_max_trade_risk`` number — the caller passes the
       tier budget and the multiplier the engine used, and the absolute
       ceiling is a SEPARATE Tier 0 cap applied after this composition would
       be (§14: vol targeting can never override a hard cap).

    5. **The risk-linked cash floor** (§36, ``Risk up -> Floor up``)::

           addon_es    = k_es       x max(0, es95 - es_target)   (0 if None)
           addon_dd    = k_drawdown x |drawdown_current_pct|      (0 if None)
           addon_model = model_risk_floor_addons[model_risk_state] (0 if
                         missing)
           floor = min(max(regime_floor_pct,
                           regime_floor_pct + addon_es + addon_dd
                                            + addon_model),
                       max_cash_floor_pct)
           binds = floor > regime_floor_pct

       The ``max(regime_floor, ...)`` is belt-and-braces: every addon is
       >= 0 by construction (validated on the params), so the composed floor
       can only ever RISE above Tier 0's regime floor, never fall below it.
       The ``min(..., max_cash_floor_pct)`` stops unvalidated addons from
       demanding an all-cash book.

       Hand-check: regime floor 0.40 (NEUTRAL_RANGE), ES 4 % vs 3 % target,
       drawdown −3 %, model risk ELEVATED::

           addon_es    = 2.0 x (0.04 - 0.03) = 0.02
           addon_dd    = 0.5 x 0.03          = 0.015
           addon_model =                       0.05
           floor = 0.40 + 0.02 + 0.015 + 0.05 = 0.485   -> BINDS
                   (0.485 <= 0.90, so the cap does not apply)

    **Honest nulls.** A missing input never becomes a 0 and never becomes a
    fabricated throttle: the corresponding modifier is 1.0 and the
    corresponding floor addon is 0.0, a ``note`` names what was missing, and
    ``health`` degrades to ``DEGRADED``. A composition that silently used
    ``es_mod = 1.0`` without saying the ES was unavailable would read as "ES
    is fine" — which is the one thing this layer must never claim.

    Raises ``ValueError`` on a malformed *number* (negative regime floor,
    non-finite tier budget, negative vol multiplier) — that is a caller bug,
    not missing data. A missing STATE is data and degrades instead.
    """
    p = params if params is not None else SizingV2Params()

    def _number(name: str, value: float, *, allow_zero: bool = True) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        return float(value)

    regime_floor = _number("regime_floor_pct", regime_floor_pct)
    if regime_floor > 1:
        raise ValueError(f"regime_floor_pct must be <= 1, got {regime_floor_pct!r}")
    tier_budget = _number("tier_budget_pct", tier_budget_pct)
    vol_multiplier = _number("vol_multiplier_used", vol_multiplier_used)

    notes: list[str] = []

    # --- (1) ES modifier ------------------------------------------------
    es_value: float | None = None
    if es95_pct_nav is None:
        es_modifier = 1.0
        notes.append(
            "es95_pct_nav unavailable — ES modifier held at 1.0 (an "
            "unmeasured ES throttles nothing; it does NOT mean ES is low)"
        )
    elif (
        isinstance(es95_pct_nav, bool)
        or not isinstance(es95_pct_nav, (int, float))
        or not math.isfinite(es95_pct_nav)
    ):
        es_modifier = 1.0
        notes.append(
            f"es95_pct_nav is not a finite number ({es95_pct_nav!r}) — ES "
            "modifier held at 1.0"
        )
    else:
        es_value = float(es95_pct_nav)
        if es_value <= p.es_target_pct_nav:
            # At or below target the composition is a THROTTLE only: a quiet
            # book is never sized UP by this layer (that would be leverage,
            # which §37 does not ask for and no hard limit would sanction).
            es_modifier = 1.0
        else:
            es_modifier = max(p.es_target_pct_nav / es_value, p.es_modifier_floor)
            es_modifier = min(es_modifier, 1.0)

    # --- (2) correlation modifier ---------------------------------------
    if correlation_state is None:
        correlation_modifier = 1.0
        notes.append(
            "correlation_state unavailable — correlation modifier held at "
            "1.0 (fewer than two priceable names, or no aligned history)"
        )
    elif correlation_state not in p.correlation_modifiers:
        correlation_modifier = 1.0
        notes.append(
            f"correlation_state {correlation_state!r} has no modifier in the "
            "parameter table — held at 1.0 rather than guessed"
        )
    else:
        correlation_modifier = float(p.correlation_modifiers[correlation_state])

    # --- (3) model-health modifier (spec §59) ---------------------------
    if model_risk_state is None:
        model_health_modifier = 1.0
        notes.append(
            "model_risk_state unavailable — model-health modifier held at "
            "1.0 (§59 budget effect not applied)"
        )
    elif model_risk_state not in p.model_risk_modifiers:
        model_health_modifier = 1.0
        notes.append(
            f"model_risk_state {model_risk_state!r} has no modifier in the "
            "parameter table — held at 1.0 rather than guessed"
        )
    else:
        model_health_modifier = float(p.model_risk_modifiers[model_risk_state])

    # --- (4) the composition (§37) --------------------------------------
    budget_pct_used = tier_budget * vol_multiplier
    candidate_budget_pct = (
        tier_budget
        * vol_multiplier
        * es_modifier
        * correlation_modifier
        * model_health_modifier
    )

    # --- (5) the risk-linked cash floor (§36) ---------------------------
    addon_es = 0.0
    if es_value is not None:
        addon_es = p.k_es * max(0.0, es_value - p.es_target_pct_nav)
    addon_dd = 0.0
    if drawdown_current_pct is None:
        notes.append(
            "drawdown_current_pct unavailable — drawdown floor addon 0.0 "
            "(no NAV history yet; it does NOT mean the book is at its peak)"
        )
    elif (
        isinstance(drawdown_current_pct, bool)
        or not isinstance(drawdown_current_pct, (int, float))
        or not math.isfinite(drawdown_current_pct)
    ):
        notes.append(
            f"drawdown_current_pct is not a finite number "
            f"({drawdown_current_pct!r}) — drawdown floor addon 0.0"
        )
    else:
        # `abs`: the drawdown block reports a NEGATIVE fraction (dd <= 0),
        # and a floor addon is a magnitude. A caller that passed a positive
        # number meaning the same magnitude gets the same answer.
        addon_dd = p.k_drawdown * abs(float(drawdown_current_pct))
    addon_model = 0.0
    if model_risk_state is not None and model_risk_state in p.model_risk_floor_addons:
        addon_model = float(p.model_risk_floor_addons[model_risk_state])

    raw_floor = regime_floor + addon_es + addon_dd + addon_model
    floor = min(max(regime_floor, raw_floor), p.max_cash_floor_pct)
    binds = floor > regime_floor

    health = str(ModelHealth.ACTIVE if not notes else ModelHealth.DEGRADED)
    reason = None if not notes else "; ".join(notes)
    return SizingV2Shadow(
        es_modifier=es_modifier,
        correlation_modifier=correlation_modifier,
        model_health_modifier=model_health_modifier,
        candidate_budget_pct=candidate_budget_pct,
        budget_pct_used=budget_pct_used,
        budget_delta_pct=candidate_budget_pct - budget_pct_used,
        risk_linked_cash_floor_pct=floor,
        risk_linked_cash_floor_binds=binds,
        regime_floor_pct=regime_floor,
        cash_floor_addons={
            "es": addon_es,
            "drawdown": addon_dd,
            "model_risk": addon_model,
            "raw_uncapped": raw_floor,
        },
        inputs={
            "es95_pct_nav": es_value,
            "correlation_state": correlation_state,
            "model_risk_state": model_risk_state,
            "drawdown_current_pct": drawdown_current_pct,
            "regime_floor_pct": regime_floor,
            "tier_budget_pct": tier_budget,
            "vol_multiplier_used": vol_multiplier,
        },
        health=health,
        reason=reason,
        notes=tuple(notes),
        params=p,
        mode=p.mode,
    )


def build_book(
    positions: Sequence[PositionRiskInput], returns: ReturnMatrix
) -> BookPnl:
    """Thin convenience re-export of :func:`book_pnl_series` so a Phase C
    caller needs one import (no new arithmetic — the construction is the
    Phase B one, contract §2.9)."""
    return book_pnl_series(positions, returns)


__all__ = [
    "CAP_LAYERS",
    "CODE_BUCKET_ES_CONTRIBUTION",
    "CODE_ES_CONTRIBUTION",
    "CODE_INCREMENTAL_ES",
    "CODE_PORTFOLIO_ES",
    "CONFIDENCE_95",
    "CONFIDENCE_99",
    "CORRELATION_CONVERGING",
    "CORRELATION_ELEVATED",
    "CORRELATION_NORMAL",
    "DECISION_APPROVE",
    "DECISION_APPROVE_WITH_RESIZE",
    "DECISION_REJECT",
    "DECISION_UNAVAILABLE_STALE",
    "LAYER_CONCENTRATION",
    "LAYER_STATISTICAL",
    "LAYER_STRESS",
    "MAX_BISECTION_STEPS",
    "MODE_PRODUCTION",
    "MODE_SHADOW",
    "MODEL_RISK_ELEVATED",
    "MODEL_RISK_HIGH",
    "MODEL_RISK_LOW",
    "CandidateSpec",
    "MetricPair",
    "QuantityCap",
    "RiskComparison",
    "ShadowVerdict",
    "SizingV2Params",
    "SizingV2Shadow",
    "StatisticalLimits",
    "build_book",
    "compare",
    "proposed_book",
    "shadow_verdict",
    "sizing_v2_shadow",
    "statistical_caps",
]
