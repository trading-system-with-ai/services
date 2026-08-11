"""Deterministic stub market data provider.

NOT MARKET DATA. NEVER A DEFAULT.

Every number this module produces is SYNTHETIC — invented by seeded random
walks and sinusoids, with no relationship whatsoever to any real market. It
exists so local development and the test suite can exercise the full pipeline
deterministically, and it is reachable ONLY by explicitly setting
``MARKET_DATA_PROVIDER=stub``. It must NEVER be used as a default, a fallback,
or a substitute when the real provider is unavailable: MASSIVE is the only
supported source of real market data, and when it is not configured the
platform shows NOTHING (HTTP 503 ``MARKET_DATA_NOT_CONFIGURED``) rather than
numbers that look real but are not (§44 rule 18).

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

from libs.trading_core.options import bs_greeks

from .provider import Bar, OptionQuote, Quote

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

# Option-chain synthesis parameters (plan §9 stub chain; all overridable,
# never truths).
DEFAULT_WEEKLY_EXPIRIES = 2  # next N weekly Fridays
DEFAULT_MONTHLY_EXPIRIES = 4  # next N monthly third-Fridays
DEFAULT_STRIKE_SPAN_PCT = 0.25  # strikes cover spot * (1 +/- span)
# Strike spacing tiers by spot magnitude: (upper_bound_exclusive, spacing).
DEFAULT_STRIKE_SPACING_TIERS: tuple[tuple[float, float], ...] = (
    (50.0, 1.0),
    (100.0, 2.5),
    (250.0, 5.0),
)
DEFAULT_STRIKE_SPACING_ABOVE = 10.0  # spacing for spot >= last tier bound
DEFAULT_IV_BASE_MIN = 0.18  # per-symbol seeded base IV lower bound
DEFAULT_IV_BASE_MAX = 0.45  # per-symbol seeded base IV upper bound
DEFAULT_SMILE_K = 2.0  # smile curvature: iv *= 1 + k * log(K/S)^2
DEFAULT_TERM_K = 0.02  # term-structure slope: iv *= 1 + k*(sqrt(dte/30) - 1)
DEFAULT_IV_FLOOR = 0.05  # absolute IV floor
DEFAULT_MIN_MID = 0.05  # contracts with mid below this are untradeable dust
DEFAULT_HALF_SPREAD_BASE = 0.01  # half-spread_pct intercept
DEFAULT_HALF_SPREAD_MONEYNESS_K = 0.05  # half-spread growth per |log(K/S)|
DEFAULT_HALF_SPREAD_DTE_K = 0.002  # half-spread growth per (dte/30)
DEFAULT_HALF_SPREAD_MIN = 0.01  # half-spread_pct clamp floor
DEFAULT_HALF_SPREAD_MAX = 0.12  # half-spread_pct clamp cap
DEFAULT_CHAIN_BASE_VOLUME = 2_000.0  # ATM same-day contract volume centre
DEFAULT_CHAIN_BASE_OI = 8_000.0  # ATM open-interest centre
DEFAULT_LIQ_MONEYNESS_WIDTH = 0.12  # gaussian width of the ATM liquidity bump
DEFAULT_LIQ_DTE_DECAY_DAYS = 180.0  # e-folding of liquidity with DTE
DEFAULT_CHAIN_VOLUME_LOG_VOL = 0.5  # stdev of log contract volume noise
DEFAULT_CHAIN_OI_LOG_VOL = 0.4  # stdev of log open-interest noise
DEFAULT_LAST_PERTURB_PCT = 0.01  # last = mid perturbed by +/- this fraction
_DAYS_PER_YEAR = 365.0  # calendar-day year for dte -> t_years
_FRIDAY = 4  # date.weekday() value for Friday


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
        weekly_expiries: int = DEFAULT_WEEKLY_EXPIRIES,
        monthly_expiries: int = DEFAULT_MONTHLY_EXPIRIES,
        strike_span_pct: float = DEFAULT_STRIKE_SPAN_PCT,
        strike_spacing_tiers: tuple[tuple[float, float], ...] = DEFAULT_STRIKE_SPACING_TIERS,
        strike_spacing_above: float = DEFAULT_STRIKE_SPACING_ABOVE,
        iv_base_min: float = DEFAULT_IV_BASE_MIN,
        iv_base_max: float = DEFAULT_IV_BASE_MAX,
        smile_k: float = DEFAULT_SMILE_K,
        term_k: float = DEFAULT_TERM_K,
        iv_floor: float = DEFAULT_IV_FLOOR,
        min_mid: float = DEFAULT_MIN_MID,
        half_spread_base: float = DEFAULT_HALF_SPREAD_BASE,
        half_spread_moneyness_k: float = DEFAULT_HALF_SPREAD_MONEYNESS_K,
        half_spread_dte_k: float = DEFAULT_HALF_SPREAD_DTE_K,
        half_spread_min: float = DEFAULT_HALF_SPREAD_MIN,
        half_spread_max: float = DEFAULT_HALF_SPREAD_MAX,
        chain_base_volume: float = DEFAULT_CHAIN_BASE_VOLUME,
        chain_base_oi: float = DEFAULT_CHAIN_BASE_OI,
        liq_moneyness_width: float = DEFAULT_LIQ_MONEYNESS_WIDTH,
        liq_dte_decay_days: float = DEFAULT_LIQ_DTE_DECAY_DAYS,
        chain_volume_log_vol: float = DEFAULT_CHAIN_VOLUME_LOG_VOL,
        chain_oi_log_vol: float = DEFAULT_CHAIN_OI_LOG_VOL,
        last_perturb_pct: float = DEFAULT_LAST_PERTURB_PCT,
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
        self.weekly_expiries = weekly_expiries
        self.monthly_expiries = monthly_expiries
        self.strike_span_pct = strike_span_pct
        self.strike_spacing_tiers = tuple(strike_spacing_tiers)
        self.strike_spacing_above = strike_spacing_above
        self.iv_base_min = iv_base_min
        self.iv_base_max = iv_base_max
        self.smile_k = smile_k
        self.term_k = term_k
        self.iv_floor = iv_floor
        self.min_mid = min_mid
        self.half_spread_base = half_spread_base
        self.half_spread_moneyness_k = half_spread_moneyness_k
        self.half_spread_dte_k = half_spread_dte_k
        self.half_spread_min = half_spread_min
        self.half_spread_max = half_spread_max
        self.chain_base_volume = chain_base_volume
        self.chain_base_oi = chain_base_oi
        self.liq_moneyness_width = liq_moneyness_width
        self.liq_dte_decay_days = liq_dte_decay_days
        self.chain_volume_log_vol = chain_volume_log_vol
        self.chain_oi_log_vol = chain_oi_log_vol
        self.last_perturb_pct = last_perturb_pct

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

    # ------------------------------------------------------------------
    # Option chain (plan §9 — STUB ONLY, deterministic synthetic chain)
    # ------------------------------------------------------------------

    def _expiries(self, as_of: date) -> list[date]:
        """Next `weekly_expiries` weekly Fridays plus the next
        `monthly_expiries` monthly third-Fridays, all strictly after
        `as_of`, deduplicated and sorted (a weekly Friday that IS a third
        Friday appears once)."""
        expiries: set[date] = set()
        # Weekly Fridays, strictly after as_of so DTE is always > 0.
        d = as_of + timedelta(days=1)
        while d.weekday() != _FRIDAY:
            d += timedelta(days=1)
        for i in range(self.weekly_expiries):
            expiries.add(d + timedelta(weeks=i))
        # Monthly third-Fridays, strictly after as_of.
        year, month = as_of.year, as_of.month
        collected = 0
        while collected < self.monthly_expiries:
            first = date(year, month, 1)
            first_friday = first + timedelta(days=(_FRIDAY - first.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            if third_friday > as_of:
                expiries.add(third_friday)
                collected += 1
            month += 1
            if month > 12:
                month, year = 1, year + 1
        return sorted(expiries)

    def _strikes(self, spot: float) -> list[float]:
        """Strike grid covering spot * (1 +/- strike_span_pct), rounded to
        the spacing tier for the spot's magnitude (plan §9 stub chain)."""
        spacing = self.strike_spacing_above
        for bound, tier_spacing in self.strike_spacing_tiers:
            if spot < bound:
                spacing = tier_spacing
                break
        lo_idx = round(spot * (1.0 - self.strike_span_pct) / spacing)
        hi_idx = round(spot * (1.0 + self.strike_span_pct) / spacing)
        return [
            round(i * spacing, 2) for i in range(lo_idx, hi_idx + 1) if i * spacing > 0
        ]

    def _base_iv(self, symbol: str) -> float:
        """Per-symbol base IV in [iv_base_min, iv_base_max], crc32-seeded on
        the symbol alone so it is stable across days."""
        seed = zlib.crc32(f"{symbol}:iv_base".encode("utf-8"))
        u = (seed % 10001) / 10000.0
        return self.iv_base_min + u * (self.iv_base_max - self.iv_base_min)

    def _node_iv(self, base_iv: float, spot: float, strike: float, dte: int) -> float:
        """IV at one (strike, dte) node: symmetric smile in log-moneyness
        plus a mild term structure that is flat at 30 DTE, floored at
        iv_floor. v0: the SAME IV is used for the call and the put at a
        node — no put/call skew yet (arrives with real chain data)."""
        log_m = math.log(strike / spot)
        smile = 1.0 + self.smile_k * log_m * log_m
        term = 1.0 + self.term_k * (math.sqrt(dte / 30.0) - 1.0)
        return max(self.iv_floor, base_iv * smile * term)

    def get_option_chain(
        self, symbol: str, spot: float, as_of: date
    ) -> list[OptionQuote]:
        """Deterministic synthetic option chain for `symbol` (plan §9).

        STUB ONLY (plan §22.1): a Black-Scholes-consistent chain until real
        chain data lands. Every random draw comes from an RNG crc32-seeded
        by the full node key (symbol, as_of, expiry, strike, right) — same
        style as the other stub methods — so the chain is identical across
        calls and processes for a given day.

        Construction:

        - Expiries: next `weekly_expiries` weekly Fridays + next
          `monthly_expiries` monthly third-Fridays (dedup, sorted); DTE
          counted from `as_of` (always > 0).
        - Strikes: tiered spacing by spot magnitude, covering
          +/- `strike_span_pct` of spot on the grid.
        - IV: per-symbol seeded base in [iv_base_min, iv_base_max], smile
          `1 + smile_k * log(K/S)^2`, term structure flat near 30 DTE,
          floored at `iv_floor`. Call and put share the node IV (v0 — no
          skew yet; documented limitation).
        - Mid: the `bs_greeks` theoretical price (theta per calendar day,
          vega per IV point — plan §9 conventions carried through).
          Contracts with mid < `min_mid` are dropped as untradeable dust.
        - bid/ask: half-spread_pct = clamp(base + moneyness_k * |log(K/S)|
          + dte_k * dte/30, min, max), applied symmetrically around mid;
          `spread_pct` is recomputed from the rounded bid/ask/mid so each
          row is self-consistent.
        - volume / open_interest: seeded log-normal noise around a centre
          that decays with distance from ATM (gaussian in log-moneyness)
          and with DTE — near-ATM monthlies land in the hundreds/thousands,
          far wings small.
        - last: mid perturbed by +/- `last_perturb_pct`, seeded.
        """
        if spot <= 0.0:
            raise ValueError(f"spot must be > 0, got {spot}")
        base_iv = self._base_iv(symbol)
        strikes = self._strikes(spot)
        chain: list[OptionQuote] = []
        for expiry in self._expiries(as_of):
            dte = (expiry - as_of).days
            for strike in strikes:
                log_m = math.log(strike / spot)
                iv = round(self._node_iv(base_iv, spot, strike, dte), 6)
                half_spread = min(
                    self.half_spread_max,
                    max(
                        self.half_spread_min,
                        self.half_spread_base
                        + self.half_spread_moneyness_k * abs(log_m)
                        + self.half_spread_dte_k * (dte / 30.0),
                    ),
                )
                # Liquidity centre: gaussian ATM bump, exponential DTE decay.
                liquidity = math.exp(
                    -((log_m / self.liq_moneyness_width) ** 2)
                ) * math.exp(-dte / self.liq_dte_decay_days)
                for right in ("C", "P"):
                    greeks = bs_greeks(spot, strike, dte / _DAYS_PER_YEAR, iv, right)
                    mid = round(greeks.price, 4)
                    if mid < self.min_mid:
                        continue  # untradeable dust
                    bid = round(mid * (1.0 - half_spread), 4)
                    ask = round(mid * (1.0 + half_spread), 4)
                    spread_pct = round((ask - bid) / mid, 6)
                    # Per-node RNG; draw order is fixed: volume, oi, last.
                    key = (
                        f"{symbol}|{as_of.isoformat()}|{expiry.isoformat()}"
                        f"|{strike:.2f}|{right}"
                    )
                    rng = random.Random(zlib.crc32(key.encode("utf-8")))
                    volume = int(
                        round(
                            self.chain_base_volume
                            * liquidity
                            * math.exp(rng.gauss(0.0, self.chain_volume_log_vol))
                        )
                    )
                    open_interest = int(
                        round(
                            self.chain_base_oi
                            * liquidity
                            * math.exp(rng.gauss(0.0, self.chain_oi_log_vol))
                        )
                    )
                    last = round(
                        mid
                        * (
                            1.0
                            + rng.uniform(-self.last_perturb_pct, self.last_perturb_pct)
                        ),
                        4,
                    )
                    chain.append(
                        OptionQuote(
                            expiry=expiry,
                            dte=dte,
                            strike=strike,
                            right=right,
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            spread_pct=spread_pct,
                            last=last,
                            volume=volume,
                            open_interest=open_interest,
                            iv=iv,
                            delta=round(greeks.delta, 6),
                            gamma=round(greeks.gamma, 6),
                            theta=round(greeks.theta, 6),
                            vega=round(greeks.vega, 6),
                        )
                    )
        return chain
