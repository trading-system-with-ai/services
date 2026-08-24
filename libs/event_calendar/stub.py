"""Deterministic synthetic event calendar — TESTS AND DEVELOPMENT ONLY.

Every candidate this module emits is INVENTED. It exists so the gateway
ingestion loop, the API shapes and the UI can be exercised without a network,
and it is the direct analogue of :class:`libs.market_data.stub.StubProvider`.

It is NEVER reachable by accident: :func:`libs.event_calendar.configured_providers`
includes it only when the settings name it explicitly, exactly as the
market-data registry keeps its stub opt-in (§44 rule 18 — an unconfigured
install must serve nothing, not plausible-looking fiction).

Determinism is the point: candidates are derived from the requested window
by fixed offsets, so the same call always yields the same rows and an
"ingest twice, create zero the second time" test is meaningful.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

from .provider import (
    EASTERN,
    US_EVENT_TIMEZONE,
    EventCandidate,
    MarketDay,
    blank_capabilities,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "stub"

#: Days from the window start at which the synthetic events are planted.
EARNINGS_OFFSET_DAYS = 3
FOMC_OFFSET_DAYS = 5
CPI_OFFSET_DAYS = 9
ESTIMATED_OFFSET_DAYS = 20

#: Eastern anchors, mirroring the real providers' conventions so a stub row
#: and a real row are interchangeable downstream.
EARNINGS_ET = (16, 5)
FOMC_DECISION_ET = (14, 0)
CPI_ET = (8, 30)


def _et_to_utc(day: date, hm: tuple[int, int]) -> datetime:
    return datetime(
        day.year, day.month, day.day, hm[0], hm[1], tzinfo=EASTERN
    ).astimezone(timezone.utc)


class StubEventCalendarProvider:
    """EventCalendarProvider emitting SYNTHETIC, NON-REAL events."""

    name = SOURCE_NAME

    def __init__(self, session_open_et: tuple[int, int] = (9, 30)) -> None:
        self.session_open_et = session_open_et

    def capabilities(self) -> dict[str, bool | str]:
        """Everything "available" — because everything is fabricated.

        ``earnings_calendar`` is True here and False on every REAL provider:
        a test asserting the honest-absence banner must therefore use a real
        adapter with a 403 fixture, never this one.
        """
        report = blank_capabilities()
        for key in report:
            report[key] = True
        return report

    def fetch_events(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[EventCandidate]:
        """Synthetic candidates inside ``[start, end]``, deterministic."""
        base = start.astimezone(EASTERN).date()
        out: list[EventCandidate] = []

        for ticker in tickers:
            symbol = (ticker or "").strip().upper()
            if not symbol:
                continue
            day = base + timedelta(days=EARNINGS_OFFSET_DAYS)
            out.append(
                EventCandidate(
                    event_key=f"EARNINGS:{symbol}:{day.isoformat()}",
                    event_type=EventType.EARNINGS,
                    title=f"{symbol} earnings release (SYNTHETIC — stub provider)",
                    scheduled_at=_et_to_utc(day, EARNINGS_ET),
                    status=EventStatus.CONFIRMED,
                    source=EventSourceKind.STRUCTURED_PROVIDER,
                    source_name=SOURCE_NAME,
                    ticker=symbol,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=EventSession.AFTER_MARKET,
                    source_event_id=f"stub-earnings-{symbol}-{day.isoformat()}",
                    raw={"synthetic": True},
                )
            )
            est_day = base + timedelta(days=ESTIMATED_OFFSET_DAYS)
            out.append(
                EventCandidate(
                    event_key=f"EARNINGS:{symbol}:{est_day.isoformat()}",
                    event_type=EventType.EARNINGS,
                    title=f"{symbol} earnings (SYNTHETIC estimate — stub provider)",
                    scheduled_at=_et_to_utc(est_day, EARNINGS_ET),
                    status=EventStatus.ESTIMATED,
                    source=EventSourceKind.DERIVED,
                    source_name=SOURCE_NAME,
                    ticker=symbol,
                    event_timezone=US_EVENT_TIMEZONE,
                    session=EventSession.AFTER_MARKET,
                    raw={"synthetic": True},
                )
            )

        fomc_day = base + timedelta(days=FOMC_OFFSET_DAYS)
        out.append(
            EventCandidate(
                event_key=f"FOMC_DECISION:{fomc_day.isoformat()}",
                event_type=EventType.FOMC_DECISION,
                title="FOMC rate decision (SYNTHETIC — stub provider)",
                scheduled_at=_et_to_utc(fomc_day, FOMC_DECISION_ET),
                status=EventStatus.CONFIRMED,
                source=EventSourceKind.FEDERAL_RESERVE,
                source_name=SOURCE_NAME,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.DURING_MARKET,
                agency="Federal Reserve",
                raw={"synthetic": True},
            )
        )

        cpi_day = base + timedelta(days=CPI_OFFSET_DAYS)
        out.append(
            EventCandidate(
                event_key=f"CPI:{cpi_day.strftime('%Y-%m')}",
                event_type=EventType.CPI,
                title="Consumer Price Index (SYNTHETIC — stub provider)",
                scheduled_at=_et_to_utc(cpi_day, CPI_ET),
                status=EventStatus.CONFIRMED,
                source=EventSourceKind.GOVERNMENT_AGENCY,
                source_name=SOURCE_NAME,
                event_timezone=US_EVENT_TIMEZONE,
                session=EventSession.BEFORE_MARKET,
                agency="BLS",
                release_period=cpi_day.strftime("%Y-%m"),
                raw={"synthetic": True},
            )
        )

        return [c for c in out if start <= c.scheduled_at <= end]

    def fetch_market_calendar(self, start: date, end: date) -> list[MarketDay]:
        """A synthetic Mon-Fri 09:30-16:00 session table (no holidays)."""
        days: list[MarketDay] = []
        day = start
        while day <= end:
            if day.weekday() < 5:
                days.append(
                    MarketDay(
                        session_date=day,
                        exchange="US",
                        open_utc=_et_to_utc(day, self.session_open_et),
                        close_utc=_et_to_utc(day, (16, 0)),
                        session_open_utc=_et_to_utc(day, (4, 0)),
                        session_close_utc=_et_to_utc(day, (20, 0)),
                        is_early_close=False,
                        source=SOURCE_NAME,
                    )
                )
            day += timedelta(days=1)
        return days
