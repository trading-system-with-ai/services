"""§86 predictiveness MEASUREMENT harness — pure arithmetic (event spec §64,
§85, §86, §92, §101, §102; audit §7.4; Phase L unit U2).

§86 is two sentences long and both of them are prohibitions: *"Do NOT assume
they are predictive. Measure."* This module is the measuring instrument, and
every design decision in it exists to make the instrument incapable of
flattering its own readings.

Pure stdlib, deterministic, **no I/O**. Like the rest of
``libs/trading_core/events/`` it may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4). The gateway
seam (``apps/gateway/event_study.py``) reads STORED rows and hands this
module plain dicts; nothing here knows what a database is.

WHAT IS MEASURED, AND WHY THESE NINE. §86 lists seven candidate features by
name — estimate revision, news materiality, valuation expansion, price
run-up, historical event move, implied move, fundamental change. Two of them
cannot be measured by this platform and say so rather than being quietly
substituted: **estimate revision** needs a consensus feed that the
subscription does not carry (Benzinga answered 403 permanently), and
**valuation expansion** needs a point-in-time multiple history the
fundamentals seam reconstructs only from filings it has. The seven that ARE
measurable are in :data:`FEATURE_SPECS`, each with the spec clause it comes
from and the stored path it is read out of, plus two the platform happens to
hold and §86's spirit clearly wants (``realized_vol_20d``,
``distance_from_52w_high``). :data:`FEATURES_NOT_MEASURABLE` names the two
missing ones IN THE REPORT, because a feature table that silently omits the
features nobody could measure reads as a claim that the list was complete.

RANK CORRELATION, NOT REGRESSION (§92). :func:`spearman_rank_corr` is
Spearman's rho over average ranks. Two reasons it is the only statistic here:
event outcomes have fat tails, so one 22% earnings gap would dominate a
Pearson correlation and the "relationship" reported would be that single
print; and a rank statistic makes no distributional assumption to be wrong
about. Ties take the AVERAGE rank (the textbook correction) rather than
first-seen order — with a feature like "material development count", ties are
the common case, not the exception, and breaking them by row order would make
rho depend on the ORDER THE EVENTS WERE QUERIED IN. That is the single
easiest way for a measurement harness to produce an irreproducible number.

NO P-VALUE, EVER (§92, §102). This module reports ``rho`` and ``n`` and
nothing else. A p-value computed over eleven earnings prints from one
watchlist is not evidence of anything, but it PRINTS like evidence — "p =
0.03" is the exact shape of the fake precision §92 forbids, and it is worse
than a bare rho because it invites the reader to stop thinking. The honest
statement this module can make is "over 14 events the rank correlation was
−0.21", and it makes exactly that.

:data:`MIN_MEANINGFUL_N` is 12 and :attr:`FeatureStat.not_meaningful` fires
below it. The threshold is a HOUSE CONVENTION, not a derived power
calculation, and :data:`CAVEATS` says so in those words. Its job is not to
certify the samples above it — nothing here certifies anything — but to make
the samples below it un-quotable: with n = 4, rho is ±0.8 about as often as
not, and a table that renders that cell the same way it renders an n = 40
cell has already told the reader something false.

DIRECTION IS SIGNED, MAGNITUDE IS ABSOLUTE. Each feature declares an
:attr:`FeatureSpec.outcome_kind`: ``"signed"`` features (run-up, fundamental
momentum, news materiality) are correlated against the SIGNED reaction — the
question is "does a run-up precede a drop"; ``"absolute"`` features (implied
move, historical median move, realized vol, IV) are correlated against
``|reaction|`` — an implied move is a magnitude forecast and has no view on
direction, so correlating it with a signed return would measure nothing and
report a number anyway. Both correlations are computed and BOTH are
returned; ``primary`` names which one the feature's own semantics make
readable, so the UI never has to guess and the other stays visible for
anybody who disagrees with the classification.

CONCLUSIONS ARE NOT PRODUCED HERE, and the omission is deliberate. There is
no ``verdict`` key, no "predictive"/"not predictive" flag and no ranking of
features by |rho|. §86 asks for a measurement; turning a measurement into a
conclusion over a sample this size is the mistake the whole section exists to
prevent, and a conclusion field would be filled in by somebody eventually
just because it was there.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

#: Bumped when the feature list, the rank statistic or the caveat text
#: changes — a stored report from an older version is a different claim.
EVENT_STUDY_MODEL_VERSION = "event_study.v1"

#: Below this many paired observations a rho is reported with
#: :attr:`FeatureStat.not_meaningful` set and MUST NOT be quoted as a
#: finding. A house convention (see the module docstring), not a power
#: calculation — :data:`CAVEATS` states that in the payload.
MIN_MEANINGFUL_N = 12

#: The flag string the payload carries, so the UI matches on a constant
#: rather than on a boolean it has to re-word.
NOT_MEANINGFUL = "NOT_MEANINGFUL"

#: ``rho`` is correlated against the SIGNED reaction.
OUTCOME_SIGNED = "signed"
#: ``rho`` is correlated against ``|reaction|`` — a magnitude claim.
OUTCOME_ABSOLUTE = "absolute"


@dataclass(frozen=True)
class FeatureSpec:
    """One §86 candidate feature: where it comes from and how to read it.

    ``outcome_kind`` is the feature's OWN semantics, not a fitted choice:
    an implied move predicts magnitude and a run-up predicts direction, and
    the harness must not be free to pick whichever correlation looks better
    after seeing the data.
    """

    name: str
    label: str
    spec_clause: str
    outcome_kind: str
    description: str


#: The §86 candidates this platform can actually measure from stored rows.
#: Order is the report's display order and is stable across runs.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="news_materiality",
        label="Material developments in window",
        spec_clause="§26/§86 news materiality",
        outcome_kind=OUTCOME_SIGNED,
        description=(
            "count of §26 material developments in the pre-event news window, "
            "cut on the un-decayed evidence score"
        ),
    ),
    FeatureSpec(
        name="news_evidence_score_max",
        label="Top evidence score",
        spec_clause="§25/§86 news materiality",
        outcome_kind=OUTCOME_SIGNED,
        description=(
            "the highest §25 evidence score in the window — one big story is a "
            "different input from many small ones, and the count above cannot "
            "tell them apart"
        ),
    ),
    FeatureSpec(
        name="price_runup_pct",
        label="Run-up since previous event",
        spec_clause="§32/§86 price run-up",
        outcome_kind=OUTCOME_SIGNED,
        description=(
            "percentage move from the previous comparable event's pre-event "
            "close to the last close before this one"
        ),
    ),
    FeatureSpec(
        name="distance_from_52w_high",
        label="Distance from 52w high",
        spec_clause="§31 positioning",
        outcome_kind=OUTCOME_SIGNED,
        description=(
            "how far below its 52-week high the stock goes into the print; "
            "the positioning half of the §35 expectation proxy"
        ),
    ),
    FeatureSpec(
        name="realized_vol_20d",
        label="Realized vol (20d)",
        spec_clause="§31 positioning",
        outcome_kind=OUTCOME_ABSOLUTE,
        description=(
            "annualised 20-day realized volatility going into the event — a "
            "magnitude input, correlated against |reaction|"
        ),
    ),
    FeatureSpec(
        name="fundamental_momentum_score",
        label="Fundamental momentum",
        spec_clause="§29/§86 fundamental change",
        outcome_kind=OUTCOME_SIGNED,
        description=(
            "the −1..+1 score over the §29 directional metric deltas between "
            "the two most recent point-in-time filings"
        ),
    ),
    FeatureSpec(
        name="implied_move_pct",
        label="Implied move",
        spec_clause="§18/§36/§86 implied move",
        outcome_kind=OUTCOME_ABSOLUTE,
        description=(
            "the stored ATM straddle implied move for this print; a magnitude "
            "forecast with no directional view, so |reaction| is the only "
            "outcome it can be scored against"
        ),
    ),
    FeatureSpec(
        name="historical_median_move",
        label="Historical median |move|",
        spec_clause="§19/§64/§86 historical event move",
        outcome_kind=OUTCOME_ABSOLUTE,
        description=(
            "the median absolute 1D reaction over the last N comparable "
            "prints, as it stood BEFORE this one"
        ),
    ),
    FeatureSpec(
        name="iv_before",
        label="ATM IV before the event",
        spec_clause="§18/§36 implied move",
        outcome_kind=OUTCOME_ABSOLUTE,
        description=(
            "at-the-money implied volatility on the pre-event session, the "
            "input the §37 crush is measured against"
        ),
    ),
)

#: ``{name: spec}`` for the report renderer.
FEATURE_BY_NAME: Mapping[str, FeatureSpec] = {
    spec.name: spec for spec in FEATURE_SPECS
}

#: The ordered feature names — the canonical column order.
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

#: §86 candidates this platform CANNOT measure, named in the report rather
#: than dropped. An omission would read as "the list was complete".
FEATURES_NOT_MEASURABLE: tuple[dict[str, str], ...] = (
    {
        "name": "estimate_revision",
        "spec_clause": "§33/§86 estimate revision",
        "reason": (
            "no consensus/estimate provider in the subscription — Benzinga "
            "returns 403 for this account, and no other configured vendor "
            "carries analyst estimates. There is no proxy for a revision, so "
            "none is substituted (§33)."
        ),
    },
    {
        "name": "valuation_expansion",
        "spec_clause": "§30/§86 valuation expansion",
        "reason": (
            "the §30 valuation context is reconstructed from filings the "
            "platform has stored, so a multiple history long enough to call "
            "an expansion exists only for the tickers backfilled deepest. "
            "Measuring it over that uneven coverage would report a property "
            "of the backfill, not of the market."
        ),
    },
)

#: Fixed caveat block, rendered VERBATIM by the endpoint and the UI. §92/§102
#: language: the sentences a reader needs in order to not over-read the
#: table, stated before the table rather than under it.
CAVEATS: tuple[str, ...] = (
    "MEASURED, NOT ASSUMED (§86). Every number below is a rank correlation "
    "computed over this installation's own stored events. None of it is "
    "evidence that a feature is predictive.",
    "n IS SMALL. This platform tracks a watchlist, not a universe, and event "
    "history goes back as far as the backfill does. Read n before rho, every "
    "time.",
    f"NOT_MEANINGFUL BELOW n = {MIN_MEANINGFUL_N}. That threshold is a house "
    "convention chosen to keep tiny samples un-quotable, not a power "
    "calculation. A rho above it is not thereby meaningful either.",
    "NO P-VALUE IS COMPUTED, DELIBERATELY (§92). |rho| and n are the whole "
    "report. A significance figure over a sample this size would print like "
    "evidence without being any.",
    "NO CLAIM OF PREDICTIVENESS IS MADE ANYWHERE IN THIS PAYLOAD. There is no "
    "verdict field, no ranking and no threshold at which a feature becomes "
    "'validated'; §86 asks for a measurement and this is one.",
    "IN-SAMPLE AND UNCORRECTED. Nine features against two horizons is "
    "eighteen correlations; some of them will look large by chance alone, and "
    "nothing here corrects for that.",
    "THE OUTCOME IS A RAW REACTION, NOT A TRADE. Signed and absolute close-to-"
    "close returns around the print, with no costs, no slippage and no "
    "position sizing. §87 keeps this out of the order path.",
)


@dataclass(frozen=True)
class FeatureRow:
    """One event's feature vector and its realised outcome.

    ``features`` is ``{name: float | None}`` over :data:`FEATURE_NAMES` and
    a missing measurement is ``None``, never ``0.0`` (§44 rule 18, §85).
    Zero is a real value of ``price_runup_pct`` and of
    ``fundamental_momentum_score``; imputing it would move a stock that did
    not move into the middle of the ranking.

    ``outcome_1d`` / ``outcome_5d`` are SIGNED close-to-close reactions as
    fractions (0.043 is +4.3%). ``None`` means the event has not produced
    that horizon yet — a future print, or one whose 5th session has not
    closed — and rows like that are simply absent from the pairings for that
    horizon rather than dropped from the sample entirely: an event can be
    measurable at 1D and not at 5D, and forcing them to agree would throw
    away the 1D observation.
    """

    event_id: int
    event_key: str
    ticker: str | None
    event_date: date | None
    features: Mapping[str, float | None] = field(default_factory=dict)
    outcome_1d: float | None = None
    outcome_5d: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(self.features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_key": self.event_key,
            "ticker": self.ticker,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "features": dict(self.features),
            "outcome_1d": self.outcome_1d,
            "outcome_5d": self.outcome_5d,
        }


@dataclass(frozen=True)
class FeatureStat:
    """One feature × one horizon: rho, n and whether it may be quoted."""

    feature: str
    horizon: str
    rho: float | None
    n: int
    not_meaningful: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "n": self.n,
            "not_meaningful": self.not_meaningful,
            "flag": NOT_MEANINGFUL if self.not_meaningful else None,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Numeric hygiene — the single gate every value passes through
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """``float(value)`` when it is a real finite number, else ``None``.

    Booleans are refused explicitly: ``True`` is an ``int`` in Python and a
    feature column that silently accepted it would rank a flag alongside a
    percentage. NaN and ±inf become ``None`` rather than propagating into a
    rank, where NaN's incomparability would corrupt the whole ordering.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Spearman's rho over AVERAGE ranks (§86)
# ---------------------------------------------------------------------------


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks 1..n with TIES SHARING THEIR AVERAGE RANK.

    ``[10, 20, 20, 30]`` ranks as ``[1, 2.5, 2.5, 4]``. The tie correction is
    not a refinement here, it is a correctness requirement: features like
    "material development count" are integer-valued and mostly tied, and
    assigning ranks by first-seen order would make rho a function of the
    order the caller happened to query events in — the harness would return a
    different answer for the same data on a different day.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0  # 1-based average of the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman_rank_corr(
    xs: Sequence[float | None], ys: Sequence[float | None]
) -> dict[str, Any]:
    """``{"rho", "n", "reason"}`` — Spearman's rho over the PAIRED, finite rows.

    Pairing is the first step and it is strict: an observation counts only if
    BOTH sides are finite numbers. Dropping a pair because the outcome has not
    happened yet is correct; imputing either side would invent an observation.
    ``n`` is therefore the number of pairs actually used, which is the number
    the caller must print beside rho.

    ``rho`` is Pearson's correlation OF THE AVERAGE RANKS — the general
    definition, valid with ties, rather than the ``1 - 6Σd²/(n(n²-1))``
    shortcut, which is only correct when there are none. With integer-valued
    features ties are the normal case here, so the shortcut would be quietly
    wrong on most columns.

    ``rho`` is ``None`` with a ``reason`` when fewer than three pairs survive
    (two points are perfectly correlated by construction) or when either side
    is CONSTANT — a feature that took one value across the sample has no
    ranking to correlate, and the zero-variance division would otherwise be
    the number reported.
    """
    if len(xs) != len(ys):
        raise ValueError(
            f"xs and ys must be the same length, got {len(xs)} and {len(ys)}"
        )
    pairs: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = _finite(raw_x)
        y = _finite(raw_y)
        if x is None or y is None:
            continue
        pairs.append((x, y))

    n = len(pairs)
    if n < 3:
        return {
            "rho": None,
            "n": n,
            "reason": f"needs >= 3 paired observations, have {n}",
        }

    xr = average_ranks([p[0] for p in pairs])
    yr = average_ranks([p[1] for p in pairs])
    mean_x = math.fsum(xr) / n
    mean_y = math.fsum(yr) / n
    dx = [v - mean_x for v in xr]
    dy = [v - mean_y for v in yr]
    var_x = math.fsum(v * v for v in dx)
    var_y = math.fsum(v * v for v in dy)
    if var_x <= 0.0:
        return {"rho": None, "n": n, "reason": "feature is constant across the sample"}
    if var_y <= 0.0:
        return {"rho": None, "n": n, "reason": "outcome is constant across the sample"}
    cov = math.fsum(a * b for a, b in zip(dx, dy))
    rho = cov / math.sqrt(var_x * var_y)
    # Float error can push a perfect correlation a hair outside [-1, 1]; a
    # reported |rho| > 1 would be read as a bug in the statistic, which it is
    # not. Clamp, do not round — rounding would hide real differences.
    rho = max(-1.0, min(1.0, rho))
    return {"rho": rho, "n": n, "reason": None}


# ---------------------------------------------------------------------------
# Row assembly (§86) — from already-gathered per-event evidence
# ---------------------------------------------------------------------------


def _dig(source: Any, *path: str) -> Any:
    """Walk nested mappings, returning ``None`` at the first missing step.

    Stored bundles are honest about absence in several different shapes — a
    section can be missing, present-but-``None``, or present with
    ``available: false`` — and a chain of ``.get()`` calls at nine call sites
    would each have to remember all three.
    """
    node = source
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def features_from_bundle(bundle: Mapping[str, Any] | None) -> dict[str, float | None]:
    """The §86 feature vector read out of ONE stored evidence bundle.

    Reads only — nothing is recomputed. The bundle was assembled as of the
    instant stamped on it and every gate (bars, filings, articles, macro
    releases) was applied then, so reading a feature out of it inherits that
    point-in-time discipline for free. Recomputing any of these from today's
    stored rows would quietly re-introduce the look-ahead the §96 suite
    exists to forbid, in the one module whose whole purpose is to measure
    whether the features were worth anything.

    Every key is present in the returned mapping, ``None`` where the bundle
    could not supply it — a caller must be able to tell "measured as zero"
    from "not measured", and a missing key makes that impossible.
    """
    out: dict[str, float | None] = {name: None for name in FEATURE_NAMES}
    if not isinstance(bundle, Mapping):
        return out

    proxies = _dig(bundle, "expectations_gap_inputs", "expectation_proxies") or {}
    out["news_materiality"] = _finite(proxies.get("material_developments"))
    out["price_runup_pct"] = _finite(proxies.get("run_up_since_previous_event"))
    out["distance_from_52w_high"] = _finite(proxies.get("distance_from_52w_high_pct"))
    out["realized_vol_20d"] = _finite(proxies.get("realized_vol_20d"))
    out["fundamental_momentum_score"] = _finite(
        _dig(bundle, "expectations_gap_inputs", "fundamental_momentum", "score")
    )

    # The run-up and the positioning fields also live on the price section;
    # the proxies block is preferred because it is the one the §35 inputs were
    # actually derived from, but an older bundle may predate it.
    pre_event = _dig(bundle, "price_analysis", "pre_event") or {}
    if out["price_runup_pct"] is None:
        out["price_runup_pct"] = _finite(pre_event.get("run_up_pct"))
    if out["distance_from_52w_high"] is None:
        out["distance_from_52w_high"] = _finite(
            pre_event.get("distance_from_52w_high_pct")
        )
    if out["realized_vol_20d"] is None:
        out["realized_vol_20d"] = _finite(pre_event.get("realized_vol_20d"))

    # The single loudest story in the window. Taken as the MAX over the
    # clusters the bundle carries rather than a mean: §25 scores decay, so a
    # mean over a quiet window full of stale filler would rank a genuine
    # development below a slow news day with many small items.
    clusters = _dig(bundle, "news", "clusters")
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
        scores = [
            value
            for value in (
                _finite(c.get("score")) for c in clusters if isinstance(c, Mapping)
            )
            if value is not None
        ]
        if scores:
            out["news_evidence_score_max"] = max(scores)
    if out["news_materiality"] is None:
        out["news_materiality"] = _finite(_dig(bundle, "news", "counts", "material"))

    # The §19/§64 history strip as it stood BEFORE this print — the §60
    # table's own 1D summary, which the price seam built from as-of-gated
    # bars. ``last8`` is preferred and ``last12``/``last4`` are the fallbacks
    # in that order: the widest window that actually produced a median is the
    # most stable estimate, and ``history_stats`` already returns ``None``
    # rather than a median of one for the windows it could not fill.
    summary = _dig(bundle, "previous_market_reaction", "history_table", "summary")
    horizon = summary.get("1D") if isinstance(summary, Mapping) else None
    if isinstance(horizon, Mapping):
        for key in ("last8", "last12", "last4"):
            window = horizon.get(key)
            if isinstance(window, Mapping):
                median = _finite(window.get("median_abs"))
                if median is not None:
                    out["historical_median_move"] = median
                    break
    return out


def collect_feature_rows(events_with_data: Iterable[Mapping[str, Any]]) -> list[FeatureRow]:
    """``list[FeatureRow]`` from per-event dicts the gateway seam assembled.

    Each input carries ``event_id``, ``event_key``, ``ticker``,
    ``event_date`` (a ``date`` or ISO string), the stored ``bundle``, an
    optional ``option_metrics`` mapping (this event's own stored straddle) and
    the realised ``outcome_1d`` / ``outcome_5d``. Nothing is fetched and
    nothing is recomputed; this function's whole job is to turn several
    honest-absence shapes into one rectangular table with ``None`` in the
    holes.

    Rows with NO usable outcome at either horizon are kept, not dropped. They
    contribute to ``coverage_pct`` — "we have the feature for 30 events and
    the outcome for 11 of them" is the most important sentence a sample this
    small can say about itself, and a table that silently filtered them would
    report 11/11 coverage and look far healthier than it is.

    Output order follows input order; the seam orders by event date so the
    rows a reader scrolls are chronological.
    """
    rows: list[FeatureRow] = []
    for item in events_with_data:
        if not isinstance(item, Mapping):
            continue
        features = features_from_bundle(item.get("bundle"))

        metrics = item.get("option_metrics")
        if isinstance(metrics, Mapping):
            # Stored as FRACTIONS by the Phase I seam; kept as fractions here
            # so every column in this table is the same unit as the outcome it
            # is ranked against. Rank correlation is scale-free, but a mixed
            # table invites a reader to compare two columns by eye.
            implied = _finite(metrics.get("implied_move_pct"))
            if implied is not None:
                out_value = abs(implied)
                features["implied_move_pct"] = out_value
            iv = _finite(metrics.get("iv_before"))
            if iv is not None:
                features["iv_before"] = iv

        raw_date = item.get("event_date")
        event_date: date | None
        if isinstance(raw_date, date):
            event_date = raw_date
        elif isinstance(raw_date, str) and raw_date:
            try:
                event_date = date.fromisoformat(raw_date[:10])
            except ValueError:
                event_date = None
        else:
            event_date = None

        rows.append(
            FeatureRow(
                event_id=int(item.get("event_id") or 0),
                event_key=str(item.get("event_key") or ""),
                ticker=(item.get("ticker") or None),
                event_date=event_date,
                features=features,
                outcome_1d=_finite(item.get("outcome_1d")),
                outcome_5d=_finite(item.get("outcome_5d")),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# The report (§86, §92)
# ---------------------------------------------------------------------------


def _stat(
    rows: Sequence[FeatureRow], spec: FeatureSpec, horizon: str, *, absolute: bool
) -> FeatureStat:
    """One cell of the table: rho over the paired rows at one horizon."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    for row in rows:
        outcome = row.outcome_1d if horizon == "1d" else row.outcome_5d
        if outcome is not None and absolute:
            outcome = abs(outcome)
        xs.append(row.features.get(spec.name))
        ys.append(outcome)
    result = spearman_rank_corr(xs, ys)
    n = int(result["n"])
    return FeatureStat(
        feature=spec.name,
        horizon=horizon,
        rho=result["rho"],
        n=n,
        not_meaningful=n < MIN_MEANINGFUL_N,
        reason=result["reason"],
    )


def feature_report(rows: Sequence[FeatureRow]) -> dict[str, Any]:
    """The whole §86 measurement, ready to serialise.

    Shape::

        {"model_version", "n_events", "outcome_coverage",
         "features": {name: {"label", "spec_clause", "outcome_kind",
                             "primary", "coverage_pct", "n_feature",
                             "signed": {"rho_1d", "rho_5d"},
                             "absolute": {"rho_1d", "rho_5d"}}},
         "not_measurable": [...], "caveats": [...]}

    BOTH correlations are computed for EVERY feature and ``primary`` names
    the one the feature's semantics make readable (see the module docstring).
    Computing only the primary one would leave a reader who disagrees with
    the classification unable to check; computing both and picking the larger
    would be the exact garden-of-forking-paths the caveats warn about, so the
    choice is fixed in :data:`FEATURE_SPECS` before any data is seen.

    ``coverage_pct`` is over the WHOLE row set, not over the rows with an
    outcome: it answers "how often does this platform even have this
    feature", which is a fact about the data pipeline and is the reason
    several columns will read 0.0 on a fresh install.

    ``n_events`` counts rows; ``outcome_coverage`` counts how many of them
    produced each horizon. A table where ``n_events`` is 40 and
    ``outcome_1d`` is 9 is telling the reader that thirty-one of these events
    have not happened yet, and no rho in it can be read without that.
    """
    total = len(rows)
    features: dict[str, Any] = {}
    for spec in FEATURE_SPECS:
        present = sum(1 for r in rows if r.features.get(spec.name) is not None)
        signed = {
            "rho_1d": _stat(rows, spec, "1d", absolute=False).to_dict(),
            "rho_5d": _stat(rows, spec, "5d", absolute=False).to_dict(),
        }
        absolute = {
            "rho_1d": _stat(rows, spec, "1d", absolute=True).to_dict(),
            "rho_5d": _stat(rows, spec, "5d", absolute=True).to_dict(),
        }
        features[spec.name] = {
            "label": spec.label,
            "spec_clause": spec.spec_clause,
            "description": spec.description,
            "outcome_kind": spec.outcome_kind,
            "primary": spec.outcome_kind,
            "n_feature": present,
            "coverage_pct": (100.0 * present / total) if total else 0.0,
            "signed": signed,
            "absolute": absolute,
            # Convenience aliases for the primary view, so a UI cell does not
            # have to branch on outcome_kind to render the headline number.
            "rho_1d": (
                signed["rho_1d"] if spec.outcome_kind == OUTCOME_SIGNED
                else absolute["rho_1d"]
            ),
            "rho_5d": (
                signed["rho_5d"] if spec.outcome_kind == OUTCOME_SIGNED
                else absolute["rho_5d"]
            ),
        }
    return {
        "model_version": EVENT_STUDY_MODEL_VERSION,
        "n_events": total,
        "outcome_coverage": {
            "outcome_1d": sum(1 for r in rows if r.outcome_1d is not None),
            "outcome_5d": sum(1 for r in rows if r.outcome_5d is not None),
        },
        "min_meaningful_n": MIN_MEANINGFUL_N,
        "features": features,
        "feature_order": list(FEATURE_NAMES),
        "not_measurable": [dict(item) for item in FEATURES_NOT_MEASURABLE],
        "caveats": list(CAVEATS),
    }
