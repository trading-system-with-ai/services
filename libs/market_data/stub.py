"""Deterministic stub market data provider.

STUB ONLY (development plan §22.1): this provider serves synthetic quotes and
synthetic daily bars for local development and tests until the real MASSIVE
market data integration lands. Quote prices are fixed per-symbol base values
plus a small sinusoidal wiggle derived from the current minute — quotes move
from minute to minute, but are fully reproducible within any given minute (and
across processes). Daily bars are a symbol-seeded geometric random walk —
fully reproducible per symbol for a given end date.

All numbers here are parameters, not market truth: base prices, the wiggle
amplitude, and the wiggle period are constructor arguments with defaults.
"""
import math
import random
import zlib
from datetime import date, datetime, timedelta, timezone

from .provider import Bar, Quote

# Default synthetic base prices for the headline symbols. Overridable per instance.
DEFAULT_BASE_PRICES: dict[str, float] = {
    "SPY": 560.0,
    "QQQ": 480.0,
    "VIX": 16.0,
}
DEFAULT_UNKNOWN_BASE_PRICE = 100.0  # base for symbols without a configured price
DEFAULT_WIGGLE_AMPLITUDE_PCT = 0.5  # peak synthetic move, in percent
DEFAULT_WIGGLE_PERIOD_MINUTES = 60  # one full sinusoid cycle per hour

# Daily-bar synthesis parameters (all overridable per instance, never truths).
DEFAULT_DAILY_VOL = 0.015  # stdev of daily log returns (~1.5%/day)
DEFAULT_MAX_DAILY_DRIFT = 0.0008  # |per-day log drift| upper bound, symbol-dependent
DEFAULT_GAP_VOL_FRACTION = 0.25  # overnight gap stdev, as a fraction of daily vol
DEFAULT_RANGE_VOL_FRACTION = 0.5  # intraday high/low extension stdev fraction
DEFAULT_BASE_VOLUME = 1_000_000.0  # synthetic share volume centre
DEFAULT_VOLUME_LOG_VOL = 0.35  # stdev of log volume around the base


class StubProvider:
    """Synthetic, deterministic implementation of MarketDataProvider."""

    def __init__(
        self,
        base_prices: dict[str, float] | None = None,
        unknown_base_price: float = DEFAULT_UNKNOWN_BASE_PRICE,
        wiggle_amplitude_pct: float = DEFAULT_WIGGLE_AMPLITUDE_PCT,
        wiggle_period_minutes: int = DEFAULT_WIGGLE_PERIOD_MINUTES,
        daily_vol: float = DEFAULT_DAILY_VOL,
        max_daily_drift: float = DEFAULT_MAX_DAILY_DRIFT,
        gap_vol_fraction: float = DEFAULT_GAP_VOL_FRACTION,
        range_vol_fraction: float = DEFAULT_RANGE_VOL_FRACTION,
        base_volume: float = DEFAULT_BASE_VOLUME,
        volume_log_vol: float = DEFAULT_VOLUME_LOG_VOL,
    ) -> None:
        self.base_prices = dict(DEFAULT_BASE_PRICES if base_prices is None else base_prices)
        self.unknown_base_price = unknown_base_price
        self.wiggle_amplitude_pct = wiggle_amplitude_pct
        self.wiggle_period_minutes = wiggle_period_minutes
        self.daily_vol = daily_vol
        self.max_daily_drift = max_daily_drift
        self.gap_vol_fraction = gap_vol_fraction
        self.range_vol_fraction = range_vol_fraction
        self.base_volume = base_volume
        self.volume_log_vol = volume_log_vol

    def get_quotes(self, symbols: list[str], now: datetime | None = None) -> list[Quote]:
        """Return synthetic quotes; `now` is injectable for deterministic tests."""
        now = now if now is not None else datetime.now(timezone.utc)
        minute_index = int(now.timestamp() // 60)
        angle = 2 * math.pi * (minute_index % self.wiggle_period_minutes) / self.wiggle_period_minutes

        quotes: list[Quote] = []
        for symbol in symbols:
            base = self.base_prices.get(symbol, self.unknown_base_price)
            # Per-symbol phase offset so symbols don't all move in lockstep;
            # derived from the characters, so it is stable across runs.
            phase = sum(ord(c) for c in symbol)
            change_pct = round(self.wiggle_amplitude_pct * math.sin(angle + phase), 4)
            price = round(base * (1 + change_pct / 100), 4)
            quotes.append(Quote(symbol=symbol, price=price, change_pct=change_pct, ts=now))
        return quotes

    def get_daily_bars(self, symbol: str, days: int, end: date | None = None) -> list[Bar]:
        """Return `days` synthetic daily bars for `symbol`, oldest first.

        STUB ONLY (plan §22.1): deterministic synthetic history until the real
        MASSIVE market data integration lands. The RNG is seeded with a stable
        hash of the symbol (zlib.crc32), so a given symbol always yields the
        exact same path for a given end date — across calls and processes.

        The series is a geometric random walk around the symbol's configured
        base price, with a small symbol-dependent drift (also derived from the
        seed). Weekend dates are skipped; the series ends at `end` (default:
        today, UTC) or the last weekday before it.
        """
        if days <= 0:
            return []
        seed = zlib.crc32(symbol.encode("utf-8"))
        rng = random.Random(seed)
        # Symbol-dependent daily log drift in [-max_daily_drift, +max_daily_drift].
        drift = ((seed % 10001) / 10000.0 - 0.5) * 2.0 * self.max_daily_drift

        # Trading dates: the last `days` weekdays ending at `end` (weekends skipped).
        d = end if end is not None else datetime.now(timezone.utc).date()
        dates: list[date] = []
        while len(dates) < days:
            if d.weekday() < 5:  # Mon..Fri
                dates.append(d)
            d -= timedelta(days=1)
        dates.reverse()

        prev_close = self.base_prices.get(symbol, self.unknown_base_price)
        bars: list[Bar] = []
        for ts in dates:
            open_ = prev_close * math.exp(
                rng.gauss(0.0, self.daily_vol * self.gap_vol_fraction)
            )
            close = open_ * math.exp(rng.gauss(drift, self.daily_vol))
            high = max(open_, close) * (
                1.0 + abs(rng.gauss(0.0, self.daily_vol * self.range_vol_fraction))
            )
            low = min(open_, close) * (
                1.0 - abs(rng.gauss(0.0, self.daily_vol * self.range_vol_fraction))
            )
            volume = self.base_volume * math.exp(rng.gauss(0.0, self.volume_log_vol))
            bars.append(
                Bar(
                    ts=ts,
                    open=round(open_, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close, 4),
                    volume=float(round(volume)),
                )
            )
            prev_close = close
        return bars
