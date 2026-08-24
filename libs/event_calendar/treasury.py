"""US Treasury daily yield curve (spec §39 multi-asset reaction).

The Treasury publishes the par yield curve as a free CSV, one file per year::

    Date,"1 Mo","1.5 Month","2 Mo",…,"2 Yr",…,"10 Yr","20 Yr","30 Yr"
    08/18/2026,3.78,3.78,3.82,…,4.19,…,4.71,5.28,5.28

Why this exists in Phase G: §39 wants the rates leg of a macro print's market
reaction, and the honest way to state it is the change in the 2Y and 10Y par
yield in BASIS POINTS — not an ETF's price move. TLT/IEF/SHY remain in the
reaction table as duration PROXIES (and are labelled as proxies); these are
the actual yields.

Two robustness rules, both learned from the live file:

* **Columns are addressed by header name, never by index.** The live 2026
  file carries a ``"1.5 Month"`` column that did not exist in older years and
  sits in the middle of the row. Positional parsing would silently shift
  every tenor by one and report the 2Y as the 3Y.
* **An empty cell is ``None``, never 0.0.** A missing tenor (the 20Y was not
  published for stretches of the 1990s-2000s, and short tenors are blank on
  some days) must read as ABSENT. Zero would be a fabricated 0% yield and
  would make a bp change calculation wildly wrong.

Rows whose date cannot be parsed are dropped with a warning rather than
guessed at.
"""
import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

import httpx

from libs.market_data.provider import CapabilityNotAvailable, MarketDataError

from .provider import CalendarProviderError

logger = logging.getLogger(__name__)

TREASURY_BASE_URL = "https://home.treasury.gov"
YIELD_CURVE_PATH = (
    "/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
    "/{year}/all"
)

DEFAULT_TIMEOUT_SECONDS = 20.0

SOURCE_NAME = "treasury"

#: The two tenors §39 reports. Kept as constants so the gateway and the UI
#: agree on the exact header spelling used as a dict key.
TENOR_2Y = "2 Yr"
TENOR_10Y = "10 Yr"

#: Canonical tenor keys, in maturity order. Any other column present in the
#: file is still captured — this tuple only fixes the ORDER for display.
TENOR_ORDER: tuple[str, ...] = (
    "1 Mo", "1.5 Month", "2 Mo", "3 Mo", "4 Mo", "6 Mo",
    "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
)

_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d")
_DATE_HEADER_RE = re.compile(r"^\s*date\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class YieldCurveRow:
    """One publication day of the par yield curve.

    ``tenors`` maps the file's own header spelling ("2 Yr", "10 Yr") to a
    yield in PERCENT, e.g. ``4.19``. Tenors absent that day are simply not
    present in the mapping — never present with a zero.
    """

    date: date
    tenors: Mapping[str, float] = field(default_factory=dict)

    def yield_for(self, tenor: str) -> float | None:
        """The yield in percent for ``tenor``, or ``None`` when unpublished."""
        return self.tenors.get(tenor)


def _parse_date(text: str) -> date | None:
    raw = (text or "").strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_yield(text: str) -> float | None:
    """``"4.19"`` -> 4.19; ``""``/``"N/A"`` -> ``None`` (never 0.0)."""
    raw = (text or "").strip()
    if not raw or raw.upper() in {"N/A", "NA", "-", "--"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_yield_curve_csv(text: str) -> list[YieldCurveRow]:
    """Parse the daily yield curve CSV, addressing columns BY HEADER NAME.

    Returns rows sorted ascending by date (the file arrives newest-first).
    Returns ``[]`` when there is no header or no ``Date`` column, so a
    changed layout reads as "no data" rather than as garbage numbers.
    """
    if not (text or "").strip():
        return []
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return []

    date_index: int | None = None
    tenor_columns: list[tuple[int, str]] = []
    for index, raw_name in enumerate(header):
        name = (raw_name or "").strip().strip('"').strip()
        if not name:
            continue
        if date_index is None and _DATE_HEADER_RE.match(name):
            date_index = index
            continue
        tenor_columns.append((index, name))

    if date_index is None:
        logger.warning(
            "Treasury yield curve CSV has no 'Date' column (header: %r) — "
            "refusing to parse positionally", header[:6],
        )
        return []

    rows: list[YieldCurveRow] = []
    for record in reader:
        if not record or len(record) <= date_index:
            continue
        curve_date = _parse_date(record[date_index])
        if curve_date is None:
            logger.warning(
                "Treasury yield curve row has an unparseable date %r — dropped",
                record[date_index],
            )
            continue
        tenors: dict[str, float] = {}
        for index, name in tenor_columns:
            if index >= len(record):
                continue
            value = _parse_yield(record[index])
            if value is not None:  # an empty cell is ABSENT, never 0.0
                tenors[name] = value
        rows.append(YieldCurveRow(date=curve_date, tenors=tenors))

    rows.sort(key=lambda row: row.date)
    return rows


class TreasuryYields:
    """Fetches the Treasury's free daily par yield curve CSV.

    Keyless. A contact User-Agent is sent on every request — the platform's
    rule for every government source.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str = TREASURY_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
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

    def get_yield_curve(self, year: int) -> list[YieldCurveRow]:
        """Every published curve for ``year``, ascending by date.

        Raises :class:`CapabilityNotAvailable` on 403 and
        :class:`CalendarProviderError` on any other fault — the same failure
        taxonomy as every other adapter, so one ``except MarketDataError``
        covers it.
        """
        path = YIELD_CURVE_PATH.format(year=int(year))
        url = f"{self.base_url}{path}"
        params = {
            "type": "daily_treasury_yield_curve",
            "field_tdr_date_value": str(int(year)),
            "_format": "csv",
        }
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise CalendarProviderError(
                f"Treasury request failed for {path}: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise CapabilityNotAvailable(
                f"home.treasury.gov returned HTTP 403 for {path}: the request "
                "was refused (User-Agent or IP block). There is NO synthetic "
                f"fallback: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise CalendarProviderError(
                f"home.treasury.gov returned HTTP {response.status_code} for "
                f"{path}: {response.text[:300]}"
            )
        rows = parse_yield_curve_csv(response.text)
        if not rows:
            logger.warning(
                "Treasury yield curve for %s returned HTTP 200 but no rows "
                "could be parsed — the CSV layout changed", year,
            )
        return rows


def yield_change_bp(before: float | None, after: float | None) -> float | None:
    """Change in basis points between two par yields in percent.

    ``None`` when either side is unpublished — an honest absence, never a
    change computed against a zero that was never a real yield.
    """
    if before is None or after is None:
        return None
    return round((after - before) * 100.0, 2)


__all__ = [
    "TENOR_2Y",
    "TENOR_10Y",
    "TENOR_ORDER",
    "TreasuryYields",
    "YieldCurveRow",
    "parse_yield_curve_csv",
    "yield_change_bp",
    "CapabilityNotAvailable",
    "MarketDataError",
]
