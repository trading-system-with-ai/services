"""Deterministic stub market data provider.

STUB ONLY (development plan §22.1): this provider serves synthetic quotes for
local development and tests until the real MASSIVE market data integration
lands. Prices are fixed per-symbol base values plus a small sinusoidal wiggle
derived from the current minute — quotes move from minute to minute, but are
fully reproducible within any given minute (and across processes).

All numbers here are parameters, not market truth: base prices, the wiggle
amplitude, and the wiggle period are constructor arguments with defaults.
"""
import math
from datetime import datetime, timezone

from .provider import Quote

# Default synthetic base prices for the headline symbols. Overridable per instance.
DEFAULT_BASE_PRICES: dict[str, float] = {
    "SPY": 560.0,
    "QQQ": 480.0,
    "VIX": 16.0,
}
DEFAULT_UNKNOWN_BASE_PRICE = 100.0  # base for symbols without a configured price
DEFAULT_WIGGLE_AMPLITUDE_PCT = 0.5  # peak synthetic move, in percent
DEFAULT_WIGGLE_PERIOD_MINUTES = 60  # one full sinusoid cycle per hour


class StubProvider:
    """Synthetic, deterministic implementation of MarketDataProvider."""

    def __init__(
        self,
        base_prices: dict[str, float] | None = None,
        unknown_base_price: float = DEFAULT_UNKNOWN_BASE_PRICE,
        wiggle_amplitude_pct: float = DEFAULT_WIGGLE_AMPLITUDE_PCT,
        wiggle_period_minutes: int = DEFAULT_WIGGLE_PERIOD_MINUTES,
    ) -> None:
        self.base_prices = dict(DEFAULT_BASE_PRICES if base_prices is None else base_prices)
        self.unknown_base_price = unknown_base_price
        self.wiggle_amplitude_pct = wiggle_amplitude_pct
        self.wiggle_period_minutes = wiggle_period_minutes

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
