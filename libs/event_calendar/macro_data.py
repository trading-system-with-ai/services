"""Macro time-series providers — the ACTUAL numbers behind a print (§38-§41).

The calendar adapters (:mod:`libs.event_calendar.bls`, :mod:`.bea`) answer
*when* a release happens. This module answers *what it said*: the observed
index levels the macro layer turns into MoM/YoY prints.

:class:`MacroDataProvider` is a Protocol in the house style — narrow, one
method, structurally typed — with two implementations:

``BlsMacroDataProvider``
    BLS public API **v1**, which needs NO key:
    ``https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}``. v1 caps
    unregistered use at ~25 requests/day and returns only the latest three
    years — both honest constraints, surfaced as errors rather than worked
    around. Its envelope carries ``status`` (``REQUEST_SUCCEEDED`` /
    ``REQUEST_NOT_PROCESSED``) alongside HTTP 200, so **the body is checked,
    not the status code**: a throttled v1 response is a 200 whose status says
    it did nothing, and treating that as success would look like "the series
    is empty" — i.e. a macro print silently reported as unavailable.

``BeaMacroDataProvider``
    BEA's statistics API DOES require a free key. Without one it raises
    :class:`CapabilityNotAvailable` from every call — the audit §6 verdict for
    proven absence — so GDP/PCE dates still appear on the calendar while
    their actuals are honestly reported as unavailable. There is no
    substitution and no estimate.

SEASONAL ADJUSTMENT IS PART OF THE IDENTITY. ``CUSR0000SA0`` (SA) and
``CUUR0000SA0`` (NSA) are different series that print different MoM numbers;
:attr:`MacroObservation.series_id` is always carried through so a consumer
can never lose track of which one it is holding.

Period codes: ``M01``-``M12`` are months, ``Q01``-``Q04`` quarters, and
``M13``/``Q05`` are ANNUAL AVERAGES — those are skipped, because an annual
average silently mixed into a monthly series would corrupt every MoM.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping, Protocol, Sequence

import httpx

from libs.market_data.provider import (  # noqa: F401 — re-exported taxonomy
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
)

logger = logging.getLogger(__name__)

BLS_API_V1_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}"
BEA_API_URL = "https://apps.bea.gov/api/data"

DEFAULT_TIMEOUT_SECONDS = 20.0

#: BLS v1 serves the latest three years per request and no more.
BLS_V1_MAX_YEARS = 3

#: Annual-average period codes — never a monthly/quarterly observation.
ANNUAL_PERIOD_CODES = frozenset({"M13", "Q05", "A01"})

#: Days after a period ends by which its release has certainly happened, used
#: ONLY when no schedule row is known. Flagged ESTIMATED wherever it is used.
ESTIMATED_RELEASE_LAG_DAYS = 45


class MacroDataError(MarketDataError):
    """A macro series could not be fetched or parsed honestly.

    A subclass of ``MarketDataError`` for the same reason
    ``CalendarProviderError`` is: callers keep ONE except clause for "a data
    source could not answer".
    """


@dataclass(frozen=True)
class MacroObservation:
    """One published observation of a macro series.

    ``period`` is the canonical key: ``"2026-07"`` for July 2026,
    ``"2026-Q2"`` for the second quarter. ``value`` is the number as
    published, in the series' own units (an index level for CPI, thousands of
    persons for payrolls, a percent for the unemployment rate) — this layer
    does NOT transform; :mod:`libs.trading_core.events.macro` does.
    """

    series_id: str
    period: str
    value: float
    year: int
    period_code: str
    footnotes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_quarterly(self) -> bool:
        return self.period_code.upper().startswith("Q")

    @property
    def period_end(self) -> date:
        """Last calendar day of the observation's period.

        The anchor for the ESTIMATED release-time fallback (period end + 45d)
        when no schedule row pins the real release instant.
        """
        code = self.period_code.upper()
        if code.startswith("Q"):
            end_month = min(4, max(1, int(code[1:]))) * 3
        else:
            end_month = min(12, max(1, int(code[1:])))
        if end_month == 12:
            return date(self.year, 12, 31)
        return date(self.year, end_month + 1, 1) - timedelta(days=1)

    @property
    def estimated_release_date(self) -> date:
        """Fallback point-in-time key: period end + 45 days (ESTIMATED)."""
        return self.period_end + timedelta(days=ESTIMATED_RELEASE_LAG_DAYS)


def period_key(year: int, period_code: str) -> str | None:
    """``(2026, "M07")`` -> ``"2026-07"``; ``(2026, "Q02")`` -> ``"2026-Q2"``.

    ``None`` for annual averages and anything unrecognised — the caller skips
    those rather than folding them into a monthly series.
    """
    code = (period_code or "").strip().upper()
    if code in ANNUAL_PERIOD_CODES:
        return None
    if len(code) < 2 or not code[1:].isdigit():
        return None
    number = int(code[1:])
    if code.startswith("M"):
        if not 1 <= number <= 12:
            return None
        return f"{int(year):04d}-{number:02d}"
    if code.startswith("Q"):
        if not 1 <= number <= 4:
            return None
        return f"{int(year):04d}-Q{number}"
    return None


class MacroDataProvider(Protocol):
    """Structural interface for a macro time-series source."""

    name: str

    def get_series(
        self,
        series_id: str,
        *,
        start_year: int,
        end_year: int,
    ) -> list[MacroObservation]:
        """Observations for ``series_id`` in ``[start_year, end_year]``.

        Ascending by period. Raises :class:`MacroDataError` on a fault and
        :class:`CapabilityNotAvailable` when the source is not entitled to
        answer (e.g. BEA without a key) — never returns ``[]`` to mean either,
        because an empty list means "the series has no observations".
        """
        ...


def parse_bls_series_response(payload: object) -> list[MacroObservation]:
    """Parse a BLS v1 ``/timeseries/data`` envelope into observations.

    Raises :class:`MacroDataError` when the envelope's own ``status`` reports
    failure — BLS answers a throttled request with HTTP 200 and
    ``REQUEST_NOT_PROCESSED``, which must NOT read as an empty series.
    """
    if not isinstance(payload, Mapping):
        raise MacroDataError(
            f"BLS returned a {type(payload).__name__}, not a JSON object"
        )
    status = str(payload.get("status") or "").strip()
    if status and status.upper() != "REQUEST_SUCCEEDED":
        messages = payload.get("message") or []
        detail = "; ".join(str(m) for m in messages) if isinstance(messages, Sequence) and not isinstance(messages, str) else str(messages)
        raise MacroDataError(
            f"BLS request was not processed (status {status!r}): "
            f"{detail or 'no detail given'}. The v1 API allows roughly 25 "
            "unregistered requests per day — no fallback series is invented."
        )

    results = payload.get("Results")
    if not isinstance(results, Mapping):
        return []
    series_list = results.get("series")
    if not isinstance(series_list, Sequence) or isinstance(series_list, str):
        return []

    out: list[MacroObservation] = []
    for series in series_list:
        if not isinstance(series, Mapping):
            continue
        series_id = str(series.get("seriesID") or "").strip()
        rows = series.get("data")
        if not series_id or not isinstance(rows, Sequence) or isinstance(rows, str):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("period") or "").strip().upper()
            try:
                year = int(str(row.get("year")).strip())
            except (TypeError, ValueError):
                continue
            key = period_key(year, code)
            if key is None:
                continue  # annual average or an unrecognised code
            raw_value = str(row.get("value") or "").strip().replace(",", "")
            try:
                value = float(raw_value)
            except ValueError:
                logger.warning(
                    "BLS series %s period %s has a non-numeric value %r — dropped",
                    series_id, key, raw_value,
                )
                continue
            notes: list[str] = []
            footnotes = row.get("footnotes")
            if isinstance(footnotes, Sequence) and not isinstance(footnotes, str):
                for note in footnotes:
                    if isinstance(note, Mapping):
                        text = str(note.get("text") or "").strip()
                        if text:
                            notes.append(text)
            out.append(
                MacroObservation(
                    series_id=series_id,
                    period=key,
                    value=value,
                    year=year,
                    period_code=code,
                    footnotes=tuple(notes),
                )
            )
    out.sort(key=lambda obs: (obs.series_id, obs.period))
    return out


class BlsMacroDataProvider:
    """BLS public API v1 — keyless, ~25 requests/day, latest 3 years."""

    name = "bls"

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str = BLS_API_V1_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": user_agent},
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass

    def __del__(self) -> None:  # pragma: no cover — GC-time best effort
        self.close()

    def get_series(
        self,
        series_id: str,
        *,
        start_year: int,
        end_year: int,
    ) -> list[MacroObservation]:
        """Observations for one BLS series, filtered to the year window.

        v1 ignores year parameters and always serves the latest three years,
        so the window is applied client-side and a request reaching further
        back is LOGGED as partially answerable rather than silently truncated.
        """
        key = (series_id or "").strip()
        if not key:
            raise MacroDataError("a BLS series id is required")
        if int(end_year) - int(start_year) + 1 > BLS_V1_MAX_YEARS:
            logger.warning(
                "BLS API v1 serves only the latest %d years; the request for "
                "%s..%s on %s will be answered with what v1 returns",
                BLS_V1_MAX_YEARS, start_year, end_year, key,
            )
        url = self.base_url.format(series_id=key)
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise MacroDataError(
                f"BLS series request failed for {key}: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"api.bls.gov returned HTTP 403 for series {key}: the request "
                f"was refused. There is NO synthetic fallback: {response.text[:300]}"
            )
        if response.status_code == 429:
            raise MacroDataError(
                f"api.bls.gov rate-limited the request for series {key} (HTTP "
                "429). The v1 API allows roughly 25 unregistered requests per "
                "day; retry later — no value is invented in the meantime."
            )
        if response.status_code >= 400:
            raise MacroDataError(
                f"api.bls.gov returned HTTP {response.status_code} for series "
                f"{key}: {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise MacroDataError(
                f"BLS returned a non-JSON body for series {key}: {exc}"
            ) from exc

        observations = parse_bls_series_response(payload)
        low, high = int(start_year), int(end_year)
        return [obs for obs in observations if low <= obs.year <= high]


class BeaMacroDataProvider:
    """BEA statistics API — requires a free key (``settings.bea_api_key``).

    Without a key every call raises :class:`CapabilityNotAvailable`: audit §6's
    "proven absence" verdict. GDP and PCE still appear on the CALENDAR (the
    schedule page is keyless); only their actuals are unavailable, and they
    are reported as unavailable rather than estimated or borrowed.
    """

    name = "bea"

    def __init__(
        self,
        *,
        api_key: str = "",
        user_agent: str = "",
        base_url: str = BEA_API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_series(
        self,
        series_id: str,
        *,
        start_year: int,
        end_year: int,
    ) -> list[MacroObservation]:
        if not self.is_configured:
            raise CapabilityNotAvailable(
                "BEA actuals need a free API key: set BEA_API_KEY (register at "
                "https://apps.bea.gov/API/signup/). GDP and PCE release DATES "
                "are unaffected — they come from the keyless schedule page — "
                "but their values are reported as unavailable, never estimated."
            )
        raise CapabilityNotAvailable(
            "the BEA statistics client is not implemented yet; GDP/PCE actuals "
            "are reported as unavailable rather than fabricated"
        )


def make_bls_macro_data_provider(settings=None) -> BlsMacroDataProvider:
    """Build the keyless BLS macro client with the contact User-Agent."""
    if settings is None:
        from libs.common.config import get_settings

        settings = get_settings()
    user_agent = (getattr(settings, "sec_user_agent", "") or "").strip() or (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )
    return BlsMacroDataProvider(user_agent=user_agent)


def make_bea_macro_data_provider(settings=None) -> BeaMacroDataProvider:
    """Build the BEA macro client; unconfigured without ``bea_api_key``."""
    if settings is None:
        from libs.common.config import get_settings

        settings = get_settings()
    user_agent = (getattr(settings, "sec_user_agent", "") or "").strip() or (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )
    return BeaMacroDataProvider(
        api_key=(getattr(settings, "bea_api_key", "") or "").strip(),
        user_agent=user_agent,
    )


__all__ = [
    "ANNUAL_PERIOD_CODES",
    "BLS_V1_MAX_YEARS",
    "ESTIMATED_RELEASE_LAG_DAYS",
    "BeaMacroDataProvider",
    "BlsMacroDataProvider",
    "CapabilityNotAvailable",
    "MacroDataError",
    "MacroDataProvider",
    "MacroObservation",
    "make_bea_macro_data_provider",
    "make_bls_macro_data_provider",
    "parse_bls_series_response",
    "period_key",
]
