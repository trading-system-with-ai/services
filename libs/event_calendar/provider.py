"""Event-calendar provider interface (event spec §75; audit §6).

This package is a SEPARATE provider registry from :mod:`libs.market_data`
(audit §6, ADR-008): calendar sources are free primary sources (SEC EDGAR,
the Federal Reserve) as much as they are paid vendors, they fail in
different ways (HTML layout drift, rate limits, contact User-Agent
requirements), and mixing them into the market-data registry would let a
missing ``MARKET_DATA_PROVIDER`` silently disable the event calendar.

The failure taxonomy is REUSED verbatim from
:mod:`libs.market_data.provider` — ``ProviderNotConfigured`` /
``MarketDataError`` / ``CapabilityNotAvailable`` — so a 403 means the same
thing on both sides of the platform and callers need only one except
clause. The audit's §6 "SUBSCRIPTION_DENIED as a first-class verdict" is
implemented by :meth:`EventCalendarProvider.capabilities`, whose FIXED key
set mirrors ``probe_capabilities`` on the market-data adapters: ``True``
(works) / ``False`` (HTTP 403 — proven absence, not in the plan) / the
error string (fault — availability unknown).

NO FABRICATION (§2, §11): when a source cannot answer, the provider returns
FEWER candidates (or none) plus an honest capability verdict. It never
invents a date. Estimated dates exist only as an explicitly
``EventStatus.ESTIMATED`` / ``EventSourceKind.DERIVED`` candidate that the
UI must render as an estimate (§11 "Do not fabricate an exact event date
when only an estimate exists").
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, Sequence

# The exchange clock. Imported from the market-data adapter rather than
# redefined: the Phase B contract forbids a 4th ZoneInfo("America/New_York")
# literal in the codebase.
from libs.market_data.alpaca import EASTERN  # noqa: F401
from libs.market_data.provider import (  # noqa: F401 — re-exported on purpose
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
)

logger = logging.getLogger(__name__)

#: The event timezone string stored alongside every US event (§10).
US_EVENT_TIMEZONE = "America/New_York"

#: The message every unconfigured-event-calendar path reports verbatim.
EVENT_CALENDAR_NOT_CONFIGURED_MESSAGE = (
    "no event calendar provider is configured — SEC EDGAR and the Federal "
    "Reserve need no key, so this means the registry was asked for an empty "
    "provider name"
)

#: The FIXED capability key set every event-calendar provider reports, in the
#: same tri-state shape as ``libs.market_data.*.probe_capabilities`` (audit
#: §6). Fixed so the gateway can merge reports across providers and the UI can
#: render a stable table; a provider that simply does not offer a capability
#: reports ``False`` rather than omitting the key.
CAPABILITY_KEYS: tuple[str, ...] = (
    "earnings_calendar",   # UPCOMING earnings dates as confirmed facts
    "earnings_history",    # PAST earnings release instants
    "market_calendar",     # session open/close per date
    "market_holidays",     # exchange holidays / early closes
    "fed_events",          # FOMC calendar, speeches
    "macro_calendar",      # BLS/BEA/Census release schedule
)

#: Every capability False — the base a provider overrides for what it serves.
NO_CAPABILITIES: dict[str, bool | str] = {k: False for k in CAPABILITY_KEYS}

#: Default US equity session used when no market_calendar row is known.
DEFAULT_SESSION_OPEN = (9, 30)
DEFAULT_SESSION_CLOSE = (16, 0)


class CalendarProviderError(MarketDataError):
    """An event-calendar request could not be answered honestly.

    A subclass of :class:`libs.market_data.provider.MarketDataError` rather
    than a parallel hierarchy: the gateway seam catches ONE exception type
    for "a data source could not answer", and `except MarketDataError`
    written for market data keeps working unchanged for calendars.
    """


def blank_capabilities() -> dict[str, bool | str]:
    """A fresh all-False capability report (never share the module dict)."""
    return dict(NO_CAPABILITIES)


@dataclass(frozen=True)
class MarketDay:
    """One exchange session, normalized to UTC instants (audit §5.2).

    ``open_utc``/``close_utc`` are the REGULAR session; ``session_open_utc``/
    ``session_close_utc`` are the extended-hours window when the source
    reports it (Alpaca does), else ``None`` — an honest absence, never the
    regular times copied across.
    """

    session_date: date
    exchange: str
    open_utc: datetime
    close_utc: datetime
    session_open_utc: datetime | None = None
    session_close_utc: datetime | None = None
    is_early_close: bool = False
    source: str = ""


# ---------------------------------------------------------------------------
# EventCandidate
# ---------------------------------------------------------------------------
# U1 owns the canonical EventCandidate in libs/trading_core/events/models.py.
# It is imported from there when present so the two shapes can NEVER drift
# (the same discipline libs/market_data/provider.py applies to OptionQuote,
# which is an alias of trading_core.contracts.ContractQuote, never a copy).
# Until U1 lands, the identical dataclass is defined here from the Phase B
# contract's field list; the import below takes precedence the moment the
# canonical module exists, so no reconciliation edit is needed in callers.

try:  # pragma: no cover — exercised by whichever half of the branch is live
    from libs.trading_core.events.models import EventCandidate  # noqa: F401
except ImportError:  # pragma: no cover
    from libs.trading_core.models.enums import (
        EventSession,
        EventSourceKind,
        EventStatus,
        EventType,
    )

    @dataclass(frozen=True)
    class EventCandidate:  # type: ignore[no-redef]
        """What a provider emits: an Event minus its database identity.

        Field-for-field the Phase B contract's EventCandidate. ``raw`` keeps
        the provider's own payload for provenance/debugging; it is never used
        as a computation input by the pure analysis layer.
        """

        event_key: str
        event_type: EventType
        title: str
        scheduled_at: datetime
        status: EventStatus
        source: EventSourceKind
        source_name: str
        ticker: str | None = None
        company_id: str | None = None
        event_timezone: str = "America/New_York"
        session: EventSession = EventSession.UNKNOWN
        source_url: str | None = None
        source_event_id: str | None = None
        last_verified_at: datetime | None = None
        importance: int | None = None
        series_id: str | None = None
        agency: str | None = None
        release_period: str | None = None
        fiscal_quarter: int | None = None
        fiscal_year: int | None = None
        speaker: str | None = None
        topic: str | None = None
        raw: dict = field(default_factory=dict)


class EventCalendarProvider(Protocol):
    """Structural interface every event-calendar provider must satisfy.

    Deliberately narrow (the market-data Protocol's lesson): only the two
    methods every consumer needs. Providers may expose more concretely —
    ``SecEdgarProvider.estimate_next_earnings`` is a pure helper, not part of
    the contract.
    """

    name: str

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state report over :data:`CAPABILITY_KEYS` (audit §6).

        ``True`` = probed and works; ``False`` = HTTP 403, i.e. PROVEN
        absence (the subscription does not include it); a string = the fault
        text, i.e. availability unknown. Never raises: a capability probe
        that raised would take the whole ingestion tick down with it.
        """
        ...

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """Candidates this source knows about in ``[start, end]``.

        ``as_of`` is the §14/§96 point-in-time cut: a provider MUST drop
        anything it could not have known at ``as_of`` (a filing accepted
        after it, an RSS item published after it). Passing ``None`` means
        "now" — live ingestion, not a replay.
        """
        ...

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """Exchange sessions in ``[start, end]``.

        Providers that do not serve sessions raise
        :class:`CapabilityNotAvailable` — never an empty list, which would be
        indistinguishable from "the exchange is closed all year".
        """
        ...


def classify_session_et(
    scheduled_at: datetime,
    *,
    open_hm: tuple[int, int] = DEFAULT_SESSION_OPEN,
    close_hm: tuple[int, int] = DEFAULT_SESSION_CLOSE,
):
    """Session bucket for a UTC instant, by its Eastern wall-clock time (§6).

    Providers classify with the default 09:30-16:00 session because they do
    not read the database; the gateway re-classifies against the stored
    ``market_calendar`` row (half-days) via U1's ``classify_session``. Keeping
    both means a provider is never blocked on the calendar table being warm.
    """
    from libs.trading_core.models.enums import EventSession

    local = scheduled_at.astimezone(EASTERN)
    minutes = local.hour * 60 + local.minute
    if minutes < open_hm[0] * 60 + open_hm[1]:
        return EventSession.BEFORE_MARKET
    if minutes >= close_hm[0] * 60 + close_hm[1]:
        return EventSession.AFTER_MARKET
    return EventSession.DURING_MARKET
