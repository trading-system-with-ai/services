"""Macro intelligence arithmetic (event spec §8, §38, §39, §40, §41, §46;
audit §6 macro rows, §11.9; Phase G unit U2).

Pure stdlib, deterministic, **no I/O** — like the rest of
``libs/trading_core/events/`` this module may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4). The gateway
seam (``apps/gateway/event_macro.py``) fetches BLS/BEA series, the Treasury
yield curve and daily bars, and hands this module plain values; nothing here
knows what a provider is.

Five ideas carry the module:

1. **A macro release is an INDEX LEVEL until this module transforms it.**
   BLS publishes ``CUSR0000SA0 = 320.412`` for 2026-07, not "CPI rose
   0.2% MoM". :func:`derive_prints` is the single place a level becomes a
   MoM %, a YoY % or a thousands-change, and every :class:`MacroPrint`
   carries both ``value_raw`` (the level as published) and ``value`` (the
   transform), so a payload can always be checked back against the source.
   The seasonal-adjustment choice is part of the SERIES, not of the
   transform: :class:`SeriesSpec.seasonally_adjusted` names it, because
   ``CUSR0000SA0`` (SA) and ``CUUR0000SA0`` (NSA) give different MoM prints
   from the same month and the difference is not an error.

2. **Point-in-time is a RELEASE-DATE gate, not a period gate** (§14, §96;
   audit §6 "index values only — no release timestamps"). July CPI exists
   as a number long before anyone may use it: it becomes knowable at
   08:30 ET on its release date. :func:`visible_prints` gates on
   ``release_at <= as_of`` and on nothing else. BLS gives us no timestamp,
   so the release instant comes from the SCHEDULE (a :class:`ScheduleRow`
   the calendar adapter parsed off ``bls.gov/schedule``); when no schedule
   row matches, the fallback is period-end + :data:`ESTIMATED_LAG_DAYS`
   and the print is stamped ``release_time_basis =
   ``:data:`RELEASE_BASIS_ESTIMATED` so no reader mistakes a guess for a
   published time.

3. **There is no consensus, and the absence is spelled one way** (§33,
   §38, §98). This platform buys no estimates feed, so every consensus and
   every surprise in a macro packet is :data:`CONSENSUS_UNAVAILABLE_REASON`
   — the same literal string ``replay.py`` uses for the earnings table.
   :func:`build_macro_packet` NEVER computes a surprise, not even when a
   caller passes something that looks like an expectation.

4. **Proxies are labelled as proxies** (§39; audit §6). There is no DXY
   index and no tradable 2Y in this platform's bar store, so
   :data:`ASSET_ROLES` maps ``UUP -> "dxy_proxy"`` and ``SHY -> "2y_proxy"``
   and the role travels into the payload. Actual 2Y/10Y yield changes come
   from the Treasury curve and are reported in BASIS POINTS, which is a
   different unit from the ETF percentage returns and therefore a different
   field (``yields``), never mixed into ``assets``.

5. **Absence is a value** (house rule; §85). Every numeric field is
   ``float | None``, every ``None`` has a companion reason string, and an
   asset with no bars is listed under ``unavailable`` with a reason rather
   than silently dropped — a reaction table missing GLD looks, to a reader
   with no unavailable list, exactly like a release gold ignored.

§41 is deliberately NOT here. "Which component will the market care about
most" is an LLM judgement; this module ships the components (headline,
core, wages, …) as facts and the prompt layer asks the question under an
``LLM ANALYSIS`` label.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from libs.trading_core.events.reaction import (
    DailyBar,
    ReactionResult,
    event_reaction,
)
from libs.trading_core.events.taxonomy import (
    MACRO_EVENT_TYPES,
    UTC,
    eastern_date,
    require_utc,
)
from libs.trading_core.models.enums import EventSession, EventType

__all__ = [
    "ASSET_ROLES",
    "CONSENSUS_UNAVAILABLE_REASON",
    "DEFAULT_MACRO_HORIZONS",
    "DEFAULT_MACRO_CONTEXT_HORIZON_DAYS",
    "ESTIMATED_LAG_DAYS",
    "MACRO_CONTEXT_EVENT_TYPES",
    "MACRO_MODEL_VERSION",
    "MACRO_SERIES",
    "PROXY_ROLES",
    "RELEASE_BASIS_ESTIMATED",
    "RELEASE_BASIS_SCHEDULED",
    "ROLE_ORDER",
    "SURPRISE_UNAVAILABLE_REASON",
    "TENOR_2Y",
    "TENOR_10Y",
    "TRACKED_TENORS",
    "TRANSFORM_CHANGE_K",
    "TRANSFORM_LEVEL",
    "TRANSFORM_MOM_PCT",
    "TRANSFORM_YOY_PCT",
    "TREND_FLAT",
    "TREND_FALLING",
    "TREND_RISING",
    "TREND_TOLERANCE",
    "TREND_WINDOW",
    "TREND_SLOPE_POINTS",
    "AssetReaction",
    "MacroObservation",
    "MacroPacket",
    "MacroPrint",
    "MultiAssetReaction",
    "ScheduleRow",
    "SeriesSpec",
    "YieldCurveRow",
    "YieldChange",
    "build_macro_packet",
    "derive_prints",
    "macro_context_for",
    "multi_asset_reaction",
    "period_end_date",
    "period_sort_key",
    "related_evidence_window",
    "release_at_for_period",
    "series_for",
    "trend_direction",
    "visible_prints",
]

#: Bump when a transform, a role label or the packet shape changes, so a
#: persisted payload still says which arithmetic produced it (same
#: discipline as ``FUNDAMENTALS_MODEL_VERSION`` / ``IMPLIED_MOVE_MODEL_VERSION``).
MACRO_MODEL_VERSION = "macro-v1"

#: §33/§38/§98 — the one spelling of the consensus absence, shared verbatim
#: with :data:`libs.trading_core.events.replay.CONSENSUS_UNAVAILABLE_REASON`.
#: It is re-declared rather than imported so this module keeps its own
#: dependency surface, and the tests pin the two to be equal.
CONSENSUS_UNAVAILABLE_REASON = "CONSENSUS DATA UNAVAILABLE"

#: A surprise is ``actual - consensus``. With no consensus there is no
#: surprise, and the field says so instead of holding a 0.0.
SURPRISE_UNAVAILABLE_REASON = "SURPRISE UNAVAILABLE (no consensus source)"

#: Transform labels. ``level`` publishes the series as-is (an unemployment
#: RATE is already a percent — turning it into a MoM % of a percent would be
#: nonsense); ``change_k`` is the first difference in the series' own unit
#: (payrolls are published as a level in thousands, and the number everyone
#: quotes is the month-over-month CHANGE).
TRANSFORM_MOM_PCT = "mom_pct"
TRANSFORM_YOY_PCT = "yoy_pct"
TRANSFORM_LEVEL = "level"
TRANSFORM_CHANGE_K = "change_k"

#: Release-time provenance (§8; audit §6 "index values only — no release
#: timestamps"). SCHEDULED means an agency schedule page gave us the date and
#: time; ESTIMATED means we derived it from the reference period.
RELEASE_BASIS_SCHEDULED = "SCHEDULED"
RELEASE_BASIS_ESTIMATED = "ESTIMATED"

#: Fallback publication lag when no schedule row matches the period. BLS
#: publishes CPI ~13 days and the Employment Situation ~5 days after the
#: reference month ends; BEA's advance GDP lands ~30 days after the quarter.
#: 45 days is deliberately CONSERVATIVE — a gate that is late hides a print
#: we could legitimately have seen, which is a coverage gap; a gate that is
#: early leaks look-ahead, which is a lie. This is a research parameter, not
#: a published lag, which is exactly why it carries the ESTIMATED stamp.
ESTIMATED_LAG_DAYS = 45

#: §39 asset roles. The value is the ROLE, and every ``*_proxy`` role is a
#: standing admission that this is not the thing itself: there is no DXY
#: index and no 2Y note in the bar store, so UUP and SHY stand in and the UI
#: badges them.
ASSET_ROLES: Mapping[str, str] = {
    "SPY": "equity",
    # The Dow. A macro reader names this index in the same breath as the S&P,
    # and DIA is the fund that tracks it.
    "DIA": "equity_dow_proxy",
    "QQQ": "equity_growth",
    # VOLATILITY, with a caveat that belongs next to the mapping rather than
    # in a footnote: VIXY holds VIX FUTURES, not the index, so roll cost makes
    # it track VIX's direction but drift from its level. It is a proxy for
    # "did fear rise into this print", never a quote of the VIX itself.
    "VIXY": "volatility_proxy",
    "TLT": "long_duration_proxy",
    "IEF": "10y_proxy",
    "SHY": "2y_proxy",
    "GLD": "gold_proxy",
    "USO": "oil_proxy",
    "UUP": "dxy_proxy",
}

#: The roles that are proxies rather than the instrument itself — the UI
#: reads this set to decide which rows get a "proxy" badge, so the judgement
#: lives here next to the mapping rather than in a string search for
#: ``"_proxy"`` in three front-end files.
PROXY_ROLES: frozenset[str] = frozenset(
    role for role in ASSET_ROLES.values() if role.endswith("_proxy")
)

#: Treasury curve tenors tracked for a macro reaction, spelled exactly as the
#: Treasury CSV header spells them.
TENOR_2Y = "2 Yr"
TENOR_10Y = "10 Yr"
TRACKED_TENORS: tuple[str, ...] = (TENOR_2Y, TENOR_10Y)

#: Reaction horizons for a macro release (§39). 1D is the print's own verdict,
#: 5D is whether it stuck.
DEFAULT_MACRO_HORIZONS: tuple[int, ...] = (1, 5)

#: §38 recent trend — how many prints the table shows.
TREND_WINDOW = 6

#: How many of those prints the direction label is computed over. Three is
#: the smallest window that can have a direction at all (two points is a
#: single step, which is noise on a monthly series).
TREND_SLOPE_POINTS = 3

#: Below this absolute slope the direction is FLAT rather than a decimal
#: dust "rising". In the transform's own unit — 0.02 pp of MoM inflation, or
#: 2 thousand payrolls per month, is not a trend.
TREND_TOLERANCE = 0.02

TREND_RISING = "rising"
TREND_FALLING = "falling"
TREND_FLAT = "flat"

#: Role display order inside a release block, so the JSON the model reads is
#: shaped the same whichever series resolved first.
ROLE_ORDER: tuple[str, ...] = (
    "headline",
    "core",
    "rate",
    "level",
    "wages",
)

#: §46 — the macro events an EARNINGS bundle's ``macro_context`` looks ahead
#: for. Not every macro type: a JOLTS print does not reprice a single stock's
#: earnings reaction, while CPI, payrolls and the FOMC decision do.
MACRO_CONTEXT_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.CPI,
        EventType.EMPLOYMENT_REPORT,
        EventType.FOMC_DECISION,
        EventType.PCE,
    }
)

#: §46 default look-ahead for that context.
DEFAULT_MACRO_CONTEXT_HORIZON_DAYS = 14


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSpec:
    """One published series and what this platform does with it.

    ``role`` is the slot the number fills in a release block (``headline``,
    ``core``, ``wages``…), ``transform`` how the level becomes the quoted
    number, and ``seasonally_adjusted`` which of the agency's two parallel
    series this is — SA and NSA are different series with different ids, and
    a MoM comparison across the two is meaningless, so the flag is part of
    the identity rather than a footnote.
    """

    series_id: str
    label: str
    role: str
    transform: str
    seasonally_adjusted: bool
    unit: str
    source: str = "bls"
    note: str | None = None


@dataclass(frozen=True)
class MacroObservation:
    """One published data point, exactly as the agency published it.

    The pure-library mirror of the ``macro_observations`` ORM row and of
    ``libs.event_calendar.macro_data.MacroObservation`` — this layer never
    imports either. ``period`` is ``YYYY-MM`` for a monthly series and
    ``YYYY-Qn`` for a quarterly one.
    """

    series_id: str
    period: str
    value: float | None
    footnotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MacroPrint:
    """One observation after the spec's transform, with its release instant.

    ``value_raw`` is the level as published and ``value`` the transform;
    both ride along so a MoM % can always be re-derived from the source.
    ``prior`` is the input the transform differenced against (the previous
    month's level for a MoM, the year-ago level for a YoY), which is what
    makes the arithmetic checkable by eye.
    """

    series_id: str
    period: str
    value_raw: float | None
    value: float | None
    prior: float | None
    transform: str
    unit: str
    release_at: datetime | None = None
    release_time_basis: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.release_at is not None:
            object.__setattr__(
                self, "release_at", require_utc(self.release_at, name="release_at")
            )


@dataclass(frozen=True)
class ScheduleRow:
    """A row off an agency's release-schedule page (§8).

    ``period`` is the REFERENCE period the release covers ("2026-07"), not
    the release date: that is the join key onto a :class:`MacroPrint`, and
    keying it on the release date instead is how a CPI that slips a day
    silently loses its prints.
    """

    period: str
    release_at_utc: datetime
    basis: str = RELEASE_BASIS_SCHEDULED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "release_at_utc",
            require_utc(self.release_at_utc, name="release_at_utc"),
        )


@dataclass(frozen=True)
class YieldCurveRow:
    """One day of the Treasury par yield curve, tenor label -> percent.

    The pure mirror of ``treasury_yields``; values are PERCENT as the CSV
    publishes them (4.21 means 4.21%), and the reaction converts a change to
    basis points exactly once, in :func:`multi_asset_reaction`.
    """

    curve_date: date
    tenors: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenors", dict(self.tenors))


@dataclass(frozen=True)
class AssetReaction:
    """One asset's move around a macro release (§39)."""

    symbol: str
    role: str
    is_proxy: bool
    basis: str | None = None
    pre_event_close: float | None = None
    pre_event_date: date | None = None
    react_date: date | None = None
    #: ``{k: return}`` as FRACTIONS, exactly as
    #: :class:`reaction.ReactionResult` reports them — 0.012 is +1.2%. The
    #: unit is inherited rather than converted on purpose: a macro reaction
    #: and an earnings reaction must be the same number in the same unit, and
    #: a rescale here would make the two silently incomparable. Note this is
    #: NOT the unit :class:`MacroPrint` uses (a MoM print IS a percent), which
    #: is why the two never share a formatter.
    returns: Mapping[int, float | None] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "returns", dict(self.returns))
        object.__setattr__(self, "reasons", dict(self.reasons))


@dataclass(frozen=True)
class YieldChange:
    """A 2Y/10Y move across the release, in basis points.

    ``before`` is the last curve date strictly BEFORE the release day and
    ``after`` the first on or after it, so a release on a day the Treasury
    did not publish (a holiday) still measures across the gap rather than
    returning nothing.
    """

    tenor: str
    before: float | None = None
    before_date: date | None = None
    after: float | None = None
    after_date: date | None = None
    change_bp: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MultiAssetReaction:
    """The §39 cross-asset table for one release.

    ``assets`` holds only the symbols that had usable bars; everything else
    is in ``unavailable`` WITH a reason. ``yields`` is a separate mapping
    because a basis-point change is not a percentage return and the two must
    never share a column.
    """

    event_at_utc: datetime | None = None
    event_date_et: date | None = None
    session: EventSession | None = None
    horizons: tuple[int, ...] = DEFAULT_MACRO_HORIZONS
    assets: Mapping[str, AssetReaction] = field(default_factory=dict)
    yields: Mapping[str, YieldChange] = field(default_factory=dict)
    unavailable: Mapping[str, str] = field(default_factory=dict)
    model_version: str = MACRO_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", dict(self.assets))
        object.__setattr__(self, "yields", dict(self.yields))
        object.__setattr__(self, "unavailable", dict(self.unavailable))
        object.__setattr__(self, "horizons", tuple(self.horizons))


@dataclass(frozen=True)
class MacroPacket:
    """The §38 macro event packet.

    Four blocks, all always present:

    - ``previous_release`` — the last release VISIBLE at ``as_of``: its
      period, its release instant, and the actual per role. Consensus and
      surprise are the fixed unavailable strings.
    - ``current_release`` — the release this event IS (period + scheduled
      instant), with consensus unavailable. There is no "actual" here: if
      the current print were already out, it would be the previous one.
    - ``recent_trend`` — per series, the last :data:`TREND_WINDOW` visible
      prints and a direction label.
    - ``coverage`` — per series, whether it produced anything and why not.
    """

    event_type: EventType
    as_of: datetime
    previous_release: Mapping[str, object] = field(default_factory=dict)
    current_release: Mapping[str, object] = field(default_factory=dict)
    recent_trend: Mapping[str, object] = field(default_factory=dict)
    coverage: Mapping[str, object] = field(default_factory=dict)
    consensus_status: str = CONSENSUS_UNAVAILABLE_REASON
    model_version: str = MACRO_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", require_utc(self.as_of, name="as_of"))
        object.__setattr__(self, "previous_release", dict(self.previous_release))
        object.__setattr__(self, "current_release", dict(self.current_release))
        object.__setattr__(self, "recent_trend", dict(self.recent_trend))
        object.__setattr__(self, "coverage", dict(self.coverage))


# ---------------------------------------------------------------------------
# §8 — the series catalogue
# ---------------------------------------------------------------------------

#: Series ids are the agency's own (BLS v1 timeseries ids; BEA needs a key and
#: therefore ships EMPTY here — a packet for GDP/PCE marks its actuals
#: unavailable rather than inventing an id we cannot fetch). RETAIL_SALES is
#: Census, for which Phase G ships no adapter, so it is empty for the same
#: reason: an empty list is a documented "no source", a wrong id is a 404 at
#: runtime dressed up as coverage.
MACRO_SERIES: Mapping[EventType, tuple[SeriesSpec, ...]] = {
    EventType.CPI: (
        SeriesSpec(
            series_id="CUSR0000SA0",
            label="CPI-U all items (SA)",
            role="headline",
            transform=TRANSFORM_MOM_PCT,
            seasonally_adjusted=True,
            unit="percent",
            note="MoM % from the seasonally adjusted index — the headline number",
        ),
        SeriesSpec(
            series_id="CUSR0000SA0L1E",
            label="CPI-U less food & energy (SA)",
            role="core",
            transform=TRANSFORM_MOM_PCT,
            seasonally_adjusted=True,
            unit="percent",
        ),
        SeriesSpec(
            series_id="CUUR0000SA0",
            label="CPI-U all items (NSA)",
            role="level",
            transform=TRANSFORM_YOY_PCT,
            seasonally_adjusted=False,
            unit="percent",
            note="YoY % is computed on the NSA index — the SA series is not "
            "meant to be differenced over 12 months",
        ),
    ),
    EventType.PPI: (
        SeriesSpec(
            series_id="WPSFD4",
            label="PPI final demand",
            role="headline",
            transform=TRANSFORM_MOM_PCT,
            seasonally_adjusted=True,
            unit="percent",
        ),
    ),
    EventType.EMPLOYMENT_REPORT: (
        SeriesSpec(
            series_id="CES0000000001",
            label="Total nonfarm payrolls (SA)",
            role="headline",
            transform=TRANSFORM_CHANGE_K,
            seasonally_adjusted=True,
            unit="thousands",
            note="published as a LEVEL in thousands; the quoted number is the "
            "month-over-month change",
        ),
        SeriesSpec(
            series_id="LNS14000000",
            label="Unemployment rate (SA)",
            role="rate",
            transform=TRANSFORM_LEVEL,
            seasonally_adjusted=True,
            unit="percent",
            note="already a rate — published as-is, never differenced into a "
            "percent-of-a-percent",
        ),
        SeriesSpec(
            series_id="CES0500000003",
            label="Average hourly earnings, private (SA)",
            role="wages",
            transform=TRANSFORM_MOM_PCT,
            seasonally_adjusted=True,
            unit="percent",
        ),
    ),
    EventType.JOLTS: (
        SeriesSpec(
            series_id="JTS000000000000000JOL",
            label="Job openings, total nonfarm (SA)",
            role="level",
            transform=TRANSFORM_LEVEL,
            seasonally_adjusted=True,
            unit="thousands",
        ),
        SeriesSpec(
            series_id="JTS000000000000000JOL",
            label="Job openings, month-over-month change",
            role="headline",
            transform=TRANSFORM_CHANGE_K,
            seasonally_adjusted=True,
            unit="thousands",
        ),
    ),
    #: BEA — needs a free API key (settings.bea_api_key). Empty until one is
    #: configured, and the packet says so in ``coverage``.
    EventType.GDP: (),
    EventType.PCE: (),
    #: Census retail sales — no adapter in Phase G.
    EventType.RETAIL_SALES: (),
}

#: Why an event type's catalogue is empty, so ``coverage`` can say more than
#: "nothing here".
_EMPTY_CATALOGUE_REASONS: Mapping[EventType, str] = {
    EventType.GDP: "BEA API key not configured — schedule only, no actuals",
    EventType.PCE: "BEA API key not configured — schedule only, no actuals",
    EventType.RETAIL_SALES: "no Census adapter — schedule only, no actuals",
}


def series_for(event_type: EventType) -> tuple[SeriesSpec, ...]:
    """The catalogue for ``event_type`` (empty tuple when there is no source)."""
    return tuple(MACRO_SERIES.get(event_type, ()))


# ---------------------------------------------------------------------------
# Small numeric helpers — every one of them refuses to fabricate
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """``value`` as a finite float, or ``None`` — NaN and inf never escape."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _pct_change(later: float | None, earlier: float | None) -> float | None:
    """``(later/earlier - 1) * 100``, or ``None`` on a non-positive base."""
    later_f = _finite(later)
    earlier_f = _finite(earlier)
    if later_f is None or earlier_f is None or earlier_f <= 0.0:
        return None
    return _finite((later_f / earlier_f - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# Period arithmetic — ``YYYY-MM`` and ``YYYY-Qn``
# ---------------------------------------------------------------------------


def _parse_period(period: str) -> tuple[int, int, bool] | None:
    """``("2026-07")`` -> ``(2026, 7, False)``; ``("2026-Q2")`` -> ``(2026, 2, True)``."""
    text = (period or "").strip().upper()
    if len(text) < 6 or text[4] != "-":
        return None
    head, tail = text[:4], text[5:]
    if not head.isdigit():
        return None
    year = int(head)
    if tail.startswith("Q"):
        rest = tail[1:]
        if not rest.isdigit():
            return None
        quarter = int(rest)
        if not 1 <= quarter <= 4:
            return None
        return year, quarter, True
    if not tail.isdigit():
        return None
    month = int(tail)
    if not 1 <= month <= 12:
        return None
    return year, month, False


def period_sort_key(period: str) -> tuple[int, int, int]:
    """Sort key that orders monthly and quarterly periods chronologically.

    A quarterly period sorts on its LAST month, so ``2026-Q1`` (March) sits
    after ``2026-02`` and before ``2026-04`` — mixing cadences in one sort is
    otherwise how a quarterly GDP print lands in January.
    Unparseable periods sort last, deterministically, rather than raising:
    a single malformed row from an agency must not take the packet down.
    """
    parsed = _parse_period(period)
    if parsed is None:
        return (9999, 99, 1)
    year, index, quarterly = parsed
    month = index * 3 if quarterly else index
    return (year, month, 0)


def period_end_date(period: str) -> date | None:
    """Last calendar day of ``period`` (``None`` when unparseable)."""
    parsed = _parse_period(period)
    if parsed is None:
        return None
    year, index, quarterly = parsed
    month = index * 3 if quarterly else index
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def release_at_for_period(
    period: str,
    schedule: Mapping[str, datetime] | Sequence[ScheduleRow] | None = None,
    *,
    estimated_lag_days: int = ESTIMATED_LAG_DAYS,
) -> tuple[datetime | None, str | None]:
    """``(release_instant, basis)`` for a reference period (§8).

    The schedule wins when it has the period; otherwise the instant is
    period-end + ``estimated_lag_days`` at midnight UTC, stamped
    :data:`RELEASE_BASIS_ESTIMATED`. The estimate is deliberately LATE (see
    :data:`ESTIMATED_LAG_DAYS`): erring late costs coverage, erring early
    costs the look-ahead guarantee.
    """
    lookup: dict[str, datetime] = {}
    if isinstance(schedule, Mapping):
        lookup = {str(k): v for k, v in schedule.items()}
    elif schedule:
        for row in schedule:
            lookup[row.period] = row.release_at_utc
    hit = lookup.get(period)
    if hit is not None:
        return require_utc(hit, name="release_at"), RELEASE_BASIS_SCHEDULED
    end = period_end_date(period)
    if end is None:
        return None, None
    estimated = datetime(
        end.year, end.month, end.day, tzinfo=UTC
    ) + timedelta(days=int(estimated_lag_days))
    return estimated, RELEASE_BASIS_ESTIMATED


# ---------------------------------------------------------------------------
# §8 — level -> print
# ---------------------------------------------------------------------------


def _prior_period(period: str, *, back: int) -> str | None:
    """The period ``back`` steps earlier on the same cadence."""
    parsed = _parse_period(period)
    if parsed is None:
        return None
    year, index, quarterly = parsed
    if quarterly:
        total = (year * 4 + (index - 1)) - back
        if total < 0:
            return None
        return f"{total // 4:04d}-Q{total % 4 + 1}"
    total = (year * 12 + (index - 1)) - back
    if total < 0:
        return None
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def derive_prints(
    observations: Iterable[MacroObservation],
    spec: SeriesSpec,
    *,
    schedule: Mapping[str, datetime] | Sequence[ScheduleRow] | None = None,
    estimated_lag_days: int = ESTIMATED_LAG_DAYS,
) -> list[MacroPrint]:
    """Turn published levels into the numbers people quote (§8, §38).

    - ``mom_pct`` — ``(x_t / x_{t-1} - 1) * 100`` against the IMMEDIATELY
      preceding period on the same cadence. A gap in the series does not
      silently become a two-month change: the previous period must be
      present by its own label, or the print is ``None`` with a reason.
    - ``yoy_pct`` — the same against the period 12 months (4 quarters) back.
    - ``change_k`` — ``x_t - x_{t-1}`` in the series' own unit.
    - ``level`` — the published value, untouched.

    Observations are sorted chronologically and de-duplicated on period
    (last one wins, which is how an agency REVISION supersedes the first
    print of the same month). Non-finite and missing values propagate as
    ``None`` with a reason; nothing here returns a NaN or a zero stand-in.
    """
    by_period: dict[str, float | None] = {}
    for obs in observations:
        if obs.series_id and obs.series_id != spec.series_id:
            continue
        by_period[obs.period] = _finite(obs.value)

    prints: list[MacroPrint] = []
    for period in sorted(by_period, key=period_sort_key):
        raw = by_period[period]
        release_at, basis = release_at_for_period(
            period, schedule, estimated_lag_days=estimated_lag_days
        )
        prior_value: float | None = None
        value: float | None = None
        reason: str | None = None

        if spec.transform == TRANSFORM_LEVEL:
            value = raw
            if value is None:
                reason = "value_unavailable"
        else:
            if spec.transform == TRANSFORM_YOY_PCT:
                # One YEAR back on the period's own cadence: 12 monthly
                # steps, or 4 quarterly ones.
                parsed = _parse_period(period)
                back = 4 if (parsed is not None and parsed[2]) else 12
            else:
                back = 1
            prior_label = _prior_period(period, back=back)
            prior_value = by_period.get(prior_label) if prior_label else None
            if raw is None:
                reason = "value_unavailable"
            elif prior_label is None or prior_label not in by_period:
                reason = "prior_period_unavailable"
            elif prior_value is None:
                reason = "prior_value_unavailable"
            elif spec.transform == TRANSFORM_CHANGE_K:
                value = _finite(raw - prior_value)
            else:
                value = _pct_change(raw, prior_value)
                if value is None:
                    reason = "prior_value_not_positive"

        prints.append(
            MacroPrint(
                series_id=spec.series_id,
                period=period,
                value_raw=raw,
                value=value,
                prior=prior_value,
                transform=spec.transform,
                unit=spec.unit,
                release_at=release_at,
                release_time_basis=basis,
                reason=reason,
            )
        )
    return prints


# ---------------------------------------------------------------------------
# §14/§96 — the point-in-time gate on RELEASE dates
# ---------------------------------------------------------------------------


def visible_prints(
    prints: Iterable[MacroPrint], as_of: datetime
) -> list[MacroPrint]:
    """The prints published on or before ``as_of`` (§14, §96) — THE macro gate.

    Gating is on ``release_at``, never on the reference period: July CPI is
    a fact about July that did not EXIST until 08:30 ET on 12 August, and a
    period-based filter would hand an as-of-11-August run a number nobody
    had. A print with no ``release_at`` at all is DROPPED — an ungated print
    is exactly the look-ahead this function exists to prevent.
    """
    moment = require_utc(as_of, name="as_of")
    kept = [
        p for p in prints if p.release_at is not None and p.release_at <= moment
    ]
    kept.sort(key=lambda p: period_sort_key(p.period))
    return kept


# ---------------------------------------------------------------------------
# §38 — the trend label
# ---------------------------------------------------------------------------


def trend_direction(
    values: Sequence[float | None],
    *,
    points: int = TREND_SLOPE_POINTS,
    tolerance: float = TREND_TOLERANCE,
) -> tuple[str | None, float | None]:
    """``(direction, slope)`` over the last ``points`` usable values.

    The slope is the plain average step ``(last - first) / (n - 1)`` in the
    series' own unit — not a regression, because a three-point OLS slope on
    a monthly series says nothing an average step does not, and the average
    step is checkable by eye. Fewer than two usable values gives
    ``(None, None)``: a direction over one point is a claim about nothing.
    """
    usable = [v for v in (_finite(v) for v in values) if v is not None]
    tail = usable[-int(points):] if points > 0 else usable
    if len(tail) < 2:
        return None, None
    slope = _finite((tail[-1] - tail[0]) / (len(tail) - 1))
    if slope is None:
        return None, None
    if slope > tolerance:
        return TREND_RISING, slope
    if slope < -tolerance:
        return TREND_FALLING, slope
    return TREND_FLAT, slope


# ---------------------------------------------------------------------------
# §38 — the packet
# ---------------------------------------------------------------------------


def _print_to_dict(item: MacroPrint) -> dict[str, object]:
    return {
        "series_id": item.series_id,
        "period": item.period,
        "value": item.value,
        "value_raw": item.value_raw,
        "prior": item.prior,
        "transform": item.transform,
        "unit": item.unit,
        "release_at": item.release_at,
        "release_time_basis": item.release_time_basis,
        "reason": item.reason,
    }


def build_macro_packet(
    event_type: EventType,
    *,
    as_of: datetime,
    schedule: Sequence[ScheduleRow] = (),
    prints_by_series: Mapping[str, Sequence[MacroPrint]] | None = None,
    current_period: str | None = None,
    current_release_at: datetime | None = None,
    trend_window: int = TREND_WINDOW,
) -> MacroPacket:
    """Assemble the §38 packet for one macro event.

    ``prints_by_series`` is keyed on ``series_id`` and is normally the
    output of :func:`derive_prints` per spec. Everything is gated through
    :func:`visible_prints` FIRST, so no branch below can see a print that
    was not published at ``as_of``.

    The "previous release" is chosen per event type, not per series: the
    latest period for which ANY series in the catalogue has a visible print.
    Picking it per series would let a CPI packet quote July headline next to
    June core because one series revised late — one release, one period.

    Consensus and surprise are :data:`CONSENSUS_UNAVAILABLE_REASON` /
    :data:`SURPRISE_UNAVAILABLE_REASON` in every branch. There is no code
    path in this function that computes a surprise.
    """
    moment = require_utc(as_of, name="as_of")
    specs = series_for(event_type)
    prints_by_series = dict(prints_by_series or {})
    schedule_rows = list(schedule)

    visible_by_spec: dict[int, list[MacroPrint]] = {}
    coverage: dict[str, object] = {}
    series_coverage: dict[str, object] = {}

    for idx, spec in enumerate(specs):
        raw = list(prints_by_series.get(spec.series_id, ()))
        seen = visible_prints(raw, moment)
        visible_by_spec[idx] = seen
        key = f"{spec.series_id}:{spec.role}"
        if not raw:
            series_coverage[key] = {
                "available": False,
                "reason": "no observations supplied",
                "label": spec.label,
                "role": spec.role,
            }
        elif not seen:
            series_coverage[key] = {
                "available": False,
                "reason": "no print released on or before as_of",
                "label": spec.label,
                "role": spec.role,
            }
        else:
            series_coverage[key] = {
                "available": True,
                "reason": None,
                "label": spec.label,
                "role": spec.role,
                "n_visible": len(seen),
                "seasonally_adjusted": spec.seasonally_adjusted,
                "transform": spec.transform,
                "unit": spec.unit,
                "note": spec.note,
            }

    coverage["series"] = series_coverage
    if not specs:
        coverage["actuals"] = {
            "available": False,
            "reason": _EMPTY_CATALOGUE_REASONS.get(
                event_type, "no data source configured for this event type"
            ),
        }
    else:
        any_visible = any(visible_by_spec.get(i) for i in range(len(specs)))
        coverage["actuals"] = {
            "available": bool(any_visible),
            "reason": None
            if any_visible
            else "no print released on or before as_of",
        }
    coverage["consensus"] = {
        "available": False,
        "reason": CONSENSUS_UNAVAILABLE_REASON,
    }
    coverage["schedule"] = {
        "available": bool(schedule_rows),
        "reason": None
        if schedule_rows
        else (
            "no agency schedule rows — release instants estimated at "
            f"period end + {ESTIMATED_LAG_DAYS}d"
        ),
        "n_rows": len(schedule_rows),
    }

    # --- previous release -------------------------------------------------
    latest_period: str | None = None
    for seen in visible_by_spec.values():
        if not seen:
            continue
        candidate = seen[-1].period
        if latest_period is None or period_sort_key(candidate) > period_sort_key(
            latest_period
        ):
            latest_period = candidate

    actual: dict[str, object] = {}
    previous_release_at: datetime | None = None
    previous_basis: str | None = None
    if latest_period is not None:
        for idx, spec in enumerate(specs):
            match = next(
                (p for p in visible_by_spec[idx] if p.period == latest_period),
                None,
            )
            if match is None:
                continue
            actual[spec.role] = {
                **_print_to_dict(match),
                "label": spec.label,
                "seasonally_adjusted": spec.seasonally_adjusted,
                "note": spec.note,
            }
            if match.release_at is not None and (
                previous_release_at is None or match.release_at > previous_release_at
            ):
                previous_release_at = match.release_at
                previous_basis = match.release_time_basis

    previous_release: dict[str, object] = {
        "period": latest_period,
        "release_at": previous_release_at,
        "release_time_basis": previous_basis,
        "actual": {role: actual[role] for role in ROLE_ORDER if role in actual},
        "consensus": CONSENSUS_UNAVAILABLE_REASON,
        "surprise": SURPRISE_UNAVAILABLE_REASON,
        "available": latest_period is not None,
    }

    # --- current release --------------------------------------------------
    resolved_period = current_period
    resolved_at = (
        require_utc(current_release_at, name="current_release_at")
        if current_release_at is not None
        else None
    )
    if resolved_period is None or resolved_at is None:
        upcoming = [
            row
            for row in sorted(schedule_rows, key=lambda r: r.release_at_utc)
            if row.release_at_utc > moment
        ]
        if upcoming:
            resolved_period = resolved_period or upcoming[0].period
            resolved_at = resolved_at or upcoming[0].release_at_utc

    current_release: dict[str, object] = {
        "period": resolved_period,
        "release_at": resolved_at,
        "consensus": CONSENSUS_UNAVAILABLE_REASON,
        "surprise": SURPRISE_UNAVAILABLE_REASON,
        "available": resolved_period is not None or resolved_at is not None,
    }

    # --- recent trend -----------------------------------------------------
    recent_trend: dict[str, object] = {}
    for idx, spec in enumerate(specs):
        seen = visible_by_spec[idx][-int(trend_window):] if trend_window > 0 else []
        direction, slope = trend_direction([p.value for p in seen])
        recent_trend[f"{spec.series_id}:{spec.role}"] = {
            "series_id": spec.series_id,
            "label": spec.label,
            "role": spec.role,
            "transform": spec.transform,
            "unit": spec.unit,
            "seasonally_adjusted": spec.seasonally_adjusted,
            "direction": direction,
            "slope": slope,
            "n_points": len(seen),
            "prints": [_print_to_dict(p) for p in seen],
        }

    return MacroPacket(
        event_type=event_type,
        as_of=moment,
        previous_release=previous_release,
        current_release=current_release,
        recent_trend=recent_trend,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# §39 — the cross-asset reaction
# ---------------------------------------------------------------------------


def _yield_change(
    rows: Sequence[YieldCurveRow], tenor: str, event_day: date
) -> YieldChange:
    """The tenor's move across ``event_day``, in basis points.

    ``before`` is the last curve strictly before the day and ``after`` the
    first on or after it, so a release on a Treasury holiday still measures
    across the gap. The change is ``(after - before) * 100`` because the CSV
    publishes PERCENT and the market quotes macro moves in bp.
    """
    before_row: YieldCurveRow | None = None
    after_row: YieldCurveRow | None = None
    for row in sorted(rows, key=lambda r: r.curve_date):
        value = _finite(row.tenors.get(tenor))
        if value is None:
            continue
        if row.curve_date < event_day:
            before_row = row
        elif after_row is None:
            after_row = row
    before = _finite(before_row.tenors.get(tenor)) if before_row else None
    after = _finite(after_row.tenors.get(tenor)) if after_row else None
    if before is None and after is None:
        return YieldChange(tenor=tenor, reason="no_curve_rows_for_tenor")
    if before is None:
        return YieldChange(
            tenor=tenor,
            after=after,
            after_date=after_row.curve_date if after_row else None,
            reason="no_curve_before_event",
        )
    if after is None:
        return YieldChange(
            tenor=tenor,
            before=before,
            before_date=before_row.curve_date if before_row else None,
            reason="no_curve_on_or_after_event",
        )
    return YieldChange(
        tenor=tenor,
        before=before,
        before_date=before_row.curve_date if before_row else None,
        after=after,
        after_date=after_row.curve_date if after_row else None,
        change_bp=_finite((after - before) * 100.0),
    )


def multi_asset_reaction(
    bars_by_symbol: Mapping[str, Sequence[DailyBar]],
    yields: Sequence[YieldCurveRow] = (),
    *,
    event_at_utc: datetime,
    session: EventSession,
    horizons: Sequence[int] = DEFAULT_MACRO_HORIZONS,
    asset_roles: Mapping[str, str] = ASSET_ROLES,
) -> MultiAssetReaction:
    """The §39 table: how each available asset moved around one release.

    Every symbol goes through :func:`reaction.event_reaction` — the SAME
    window arithmetic an earnings reaction uses, with the same session rules
    and the same ``basis`` label — so a CPI reaction and an earnings reaction
    are comparable numbers rather than two similar-looking ones.

    Only symbols with usable bars appear in ``assets``; a symbol in
    ``asset_roles`` with no bars, or whose reaction could not locate a
    window, is listed in ``unavailable`` with the reason
    ``event_reaction`` gave. Bars are assumed ALREADY as-of gated by the
    caller (``reaction.as_of_bar_filter``) — this layer never sees a bar it
    should not have.

    ``yields`` are handled separately and in BASIS POINTS: a 7bp move in the
    2Y and a 0.4% move in SPY are not the same kind of number and never
    share a column.
    """
    moment = require_utc(event_at_utc, name="event_at_utc")
    event_day = eastern_date(moment)
    horizons = tuple(sorted({int(k) for k in horizons}))

    assets: dict[str, AssetReaction] = {}
    unavailable: dict[str, str] = {}

    for symbol in sorted(asset_roles):
        role = asset_roles[symbol]
        bars = list(bars_by_symbol.get(symbol, ()) or ())
        if not bars:
            unavailable[symbol] = "no stored daily bars"
            continue
        result: ReactionResult = event_reaction(
            bars, event_day, session, horizons=horizons
        )
        if not result.bars_available:
            unavailable[symbol] = result.reasons.get("bars", "reaction_unavailable")
            continue
        assets[symbol] = AssetReaction(
            symbol=symbol,
            role=role,
            is_proxy=role in PROXY_ROLES,
            basis=result.basis,
            pre_event_close=result.pre_event_close,
            pre_event_date=result.pre_event_date,
            react_date=result.react_date,
            returns=dict(result.returns),
            reasons=dict(result.reasons),
        )

    yield_rows = list(yields)
    yield_changes: dict[str, YieldChange] = {}
    for tenor in TRACKED_TENORS:
        if not yield_rows:
            yield_changes[tenor] = YieldChange(
                tenor=tenor, reason="no treasury curve rows supplied"
            )
            continue
        yield_changes[tenor] = _yield_change(yield_rows, tenor, event_day)

    return MultiAssetReaction(
        event_at_utc=moment,
        event_date_et=event_day,
        session=session,
        horizons=horizons,
        assets=assets,
        yields=yield_changes,
        unavailable=unavailable,
    )


# ---------------------------------------------------------------------------
# §46 — macro context for a NON-macro (earnings) bundle
# ---------------------------------------------------------------------------


def _event_time(row: Mapping[str, object]) -> datetime | None:
    value = row.get("scheduled_at")
    if isinstance(value, datetime):
        try:
            return require_utc(value, name="scheduled_at")
        except ValueError:
            return None
    return None


def _event_type(row: Mapping[str, object]) -> EventType | None:
    value = row.get("event_type")
    if isinstance(value, EventType):
        return value
    if isinstance(value, str):
        try:
            return EventType(value)
        except ValueError:
            return None
    return None


def macro_context_for(
    events_upcoming: Sequence[Mapping[str, object]],
    *,
    as_of: datetime,
    horizon_days: int = DEFAULT_MACRO_CONTEXT_HORIZON_DAYS,
    event_types: frozenset[EventType] = MACRO_CONTEXT_EVENT_TYPES,
) -> dict[str, object]:
    """"What macro lands between now and the horizon" (§46), for any bundle.

    The one number an earnings analysis needs from this whole module: a
    company reporting the day before CPI is a different trade from the same
    company reporting the day after, and the bundle should not have to
    reason about it from a raw event list.

    ``events_upcoming`` is a list of plain event dicts (``event_type``,
    ``scheduled_at`` as a tz-aware datetime, optionally ``event_id``,
    ``title``, ``importance``). Rows that are not in ``event_types``, not in
    the window, or missing a usable timestamp are skipped and COUNTED in
    ``skipped`` rather than silently dropped. Output is sorted by time and is
    deterministic for a given input.
    """
    moment = require_utc(as_of, name="as_of")
    horizon = moment + timedelta(days=int(horizon_days))
    items: list[dict[str, object]] = []
    skipped = 0

    for row in events_upcoming:
        etype = _event_type(row)
        when = _event_time(row)
        if etype is None or when is None:
            skipped += 1
            continue
        if etype not in event_types:
            continue
        if not moment <= when <= horizon:
            continue
        delta = when - moment
        items.append(
            {
                "event_id": row.get("event_id") or row.get("id"),
                "event_type": etype.value,
                "title": row.get("title"),
                "scheduled_at": when,
                "days_to": round(delta.total_seconds() / 86400.0, 2),
                "importance": row.get("importance"),
                "status": row.get("status"),
            }
        )

    items.sort(key=lambda item: (item["scheduled_at"], str(item["event_type"])))
    return {
        "available": bool(items),
        "as_of": moment,
        "horizon_days": int(horizon_days),
        "horizon_end": horizon,
        "event_types": sorted(t.value for t in event_types),
        "upcoming": items,
        "next": items[0] if items else None,
        "skipped_unparseable": skipped,
        "reason": None
        if items
        else f"no tracked macro release within {int(horizon_days)}d of as_of",
        "model_version": MACRO_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# §40 — the macro evidence window
# ---------------------------------------------------------------------------


def related_evidence_window(
    previous_release_at: datetime | None,
    as_of: datetime,
    other_events: Sequence[Mapping[str, object]] = (),
    *,
    exclude_event_ids: Iterable[object] = (),
) -> dict[str, object]:
    """Everything that happened BETWEEN the last release and now (§40).

    §40 is explicit that the relevant themes for a CPI print (PPI, wages,
    oil, shelter, Fed speeches…) shift with context and must not come from a
    rigid keyword list. So this function applies no keyword list at all: it
    returns the deterministic FACTUAL set — every other macro/Fed event in
    the window, typed and timestamped — and the LLM layer picks the themes
    out of it. That split is the §40 contract: facts deterministic, themes
    modelled.

    With no ``previous_release_at`` the window is open on the left and says
    so in ``reason`` rather than silently becoming "all of history".
    """
    moment = require_utc(as_of, name="as_of")
    start = (
        require_utc(previous_release_at, name="previous_release_at")
        if previous_release_at is not None
        else None
    )
    excluded = {e for e in exclude_event_ids if e is not None}

    items: list[dict[str, object]] = []
    skipped = 0
    for row in other_events:
        etype = _event_type(row)
        when = _event_time(row)
        if etype is None or when is None:
            skipped += 1
            continue
        identifier = row.get("event_id") or row.get("id")
        if identifier is not None and identifier in excluded:
            continue
        if when > moment:
            continue
        if start is not None and when < start:
            continue
        items.append(
            {
                "event_id": identifier,
                "event_type": etype.value,
                "title": row.get("title"),
                "scheduled_at": when,
                "importance": row.get("importance"),
                "is_macro": etype in MACRO_EVENT_TYPES,
            }
        )

    items.sort(key=lambda item: (item["scheduled_at"], str(item["event_type"])))
    return {
        "available": bool(items),
        "window_start": start,
        "window_end": moment,
        "events": items,
        "n_events": len(items),
        "skipped_unparseable": skipped,
        "reason": None
        if items
        else (
            "no previous release instant — window is open on the left"
            if start is None
            else "no other tracked events between the previous release and as_of"
        ),
        "note": (
            "Deterministic factual set only (§40); relevant THEMES are an LLM "
            "judgement over these events, not a keyword filter applied here."
        ),
        "model_version": MACRO_MODEL_VERSION,
    }
