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
import re
import zlib
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Sequence

from libs.trading_core.options import bs_greeks

from .provider import (
    Bar,
    FinancialStatement,
    IntradayBar,
    NewsArticle,
    OptionContractRef,
    OptionQuote,
    Quote,
    require_aware_utc,
)

# Bare OCC option symbol: UNDERLYING + yymmdd + C/P + strike*1000 (8 digits).
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

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
DEFAULT_WEEKLY_EXPIRIES = 7  # next N weekly Fridays — 7 keeps a Friday
# inside ANY 30-45 DTE selection window year-round (weekly granularity 7d
# < window width 16d, coverage to ~49d); at 2 the income tests rotted with
# the calendar (2026-08-20: nearest monthly third-Friday fell at DTE 29).
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
# Fundamentals synthesis parameters (STUB ONLY, all overridable, never truths).
DEFAULT_FIN_BASE_REVENUE = 2.0e10  # quarterly revenue centre, USD
DEFAULT_FIN_GROWTH_QOQ = 0.02  # per-quarter compounding revenue growth
DEFAULT_FIN_SEASONAL_PCT = 0.08  # peak seasonal swing around the trend
DEFAULT_FIN_GROSS_MARGIN = 0.42  # gross_profit / revenues
DEFAULT_FIN_OPEX_PCT = 0.22  # operating_expenses / revenues
DEFAULT_FIN_TAX_RATE = 0.21  # income_tax_expense_benefit / pretax income
DEFAULT_FIN_SHARES = 1.6e9  # diluted average shares outstanding
DEFAULT_FIN_ASSET_TURNOVER = 0.28  # quarterly revenue / total assets
#: Filed this many days after the period ends, at 21:05 UTC — the synthetic
#: ``acceptance_datetime`` every as-of test filters on.
DEFAULT_FIN_FILING_LAG_DAYS = 26
DEFAULT_FIN_ACCEPTANCE_HOUR = 21
DEFAULT_FIN_ACCEPTANCE_MINUTE = 5
#: Calendar quarter ends the synthetic fiscal calendar uses (Q1..Q4).
_FIN_QUARTER_ENDS: tuple[tuple[int, int], ...] = (
    (3, 31), (6, 30), (9, 30), (12, 31),
)

# Phase I historical-option seam (§36) — SYNTHETIC, tests only. A small
# fixed strike ladder around a nominal 100 spot: wide enough that any ATM pick
# lands inside it, narrow enough that a test can assert the whole list.
HIST_STRIKE_MIN = 90.0
HIST_STRIKE_MAX = 110.0
HIST_STRIKE_STEP = 5.0
#: Synthetic premium at zero days elapsed, and the per-session decay, for the
#: fabricated option bars. Parameters, never market truth.
HIST_PREMIUM_BASE = 6.0
HIST_PREMIUM_DECAY_PER_DAY = 0.05
HIST_PREMIUM_FLOOR = 0.5
#: The put's premium is offset from the call's by this much so a straddle
#: built from the pair is not trivially 2x one leg (which would hide a bug
#: that reads the same leg twice).
HIST_PUT_OFFSET = 0.75
#: Synthetic daily contract volume, constant so bar counts are assertable.
HIST_BAR_VOLUME = 250.0

_DAYS_PER_YEAR = 365.0  # calendar-day year for dte -> t_years
_FRIDAY = 4  # date.weekday() value for Friday


#: Fixed origin of every symbol's walk: bar values accumulate from here, so
#: a bar is the same for a given (symbol, date) no matter how it is windowed.
_WALK_EPOCH = date(2022, 1, 3)

#: The exchange clock the synthetic session grid is laid out on. The stub owns
#: its own constant rather than importing the pure library's, keeping
#: libs.market_data free of a trading_core.events dependency; both name the
#: same IANA zone, so DST transitions land identically.
_STUB_EASTERN = ZoneInfo("America/New_York")

# Synthetic intraday session grid (STUB ONLY — a fixed schedule, not a
# calendar: holidays and early closes are NOT modelled, and this must never be
# read as when the market was actually open).
DEFAULT_REGULAR_OPEN_ET = time(9, 30)
DEFAULT_REGULAR_CLOSE_ET = time(16, 0)
DEFAULT_AFTER_HOURS_CLOSE_ET = time(20, 0)
#: After-hours minutes are SPARSE — one bar every N minutes from the close to
#: 20:00 ET — mirroring the real feed, where extended-hours bars only exist
#: for minutes that actually traded. A dense after-hours grid would let a
#: consumer's "no after-hours bars" branch go permanently untested.
DEFAULT_AFTER_HOURS_STEP_MINUTES = 5
#: Intraday synthesis: per-minute log-return stdev, and the volume centre for
#: one regular-session minute.
DEFAULT_MINUTE_VOL = 0.0012
DEFAULT_MINUTE_BASE_VOLUME = 4_000.0
DEFAULT_MINUTE_VOLUME_LOG_VOL = 0.4
#: After-hours minutes are thin: volume centre scaled by this fraction.
DEFAULT_AFTER_HOURS_VOLUME_FRACTION = 0.08


def _default_series_end() -> date:
    """Today (UTC) — or the frozen STUB_ANCHOR_DATE test seam when set."""
    from libs.common.config import get_settings

    anchor = get_settings().stub_anchor_date
    if anchor:
        try:
            return date.fromisoformat(anchor)
        except ValueError:
            pass  # a malformed anchor must never break the provider
    return datetime.now(timezone.utc).date()


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
        regular_open_et: time = DEFAULT_REGULAR_OPEN_ET,
        regular_close_et: time = DEFAULT_REGULAR_CLOSE_ET,
        after_hours_close_et: time = DEFAULT_AFTER_HOURS_CLOSE_ET,
        after_hours_step_minutes: int = DEFAULT_AFTER_HOURS_STEP_MINUTES,
        minute_vol: float = DEFAULT_MINUTE_VOL,
        minute_base_volume: float = DEFAULT_MINUTE_BASE_VOLUME,
        minute_volume_log_vol: float = DEFAULT_MINUTE_VOLUME_LOG_VOL,
        after_hours_volume_fraction: float = DEFAULT_AFTER_HOURS_VOLUME_FRACTION,
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
        fin_base_revenue: float = DEFAULT_FIN_BASE_REVENUE,
        fin_growth_qoq: float = DEFAULT_FIN_GROWTH_QOQ,
        fin_seasonal_pct: float = DEFAULT_FIN_SEASONAL_PCT,
        fin_gross_margin: float = DEFAULT_FIN_GROSS_MARGIN,
        fin_opex_pct: float = DEFAULT_FIN_OPEX_PCT,
        fin_tax_rate: float = DEFAULT_FIN_TAX_RATE,
        fin_shares: float = DEFAULT_FIN_SHARES,
        fin_asset_turnover: float = DEFAULT_FIN_ASSET_TURNOVER,
        fin_filing_lag_days: int = DEFAULT_FIN_FILING_LAG_DAYS,
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
        self.regular_open_et = regular_open_et
        self.regular_close_et = regular_close_et
        self.after_hours_close_et = after_hours_close_et
        self.after_hours_step_minutes = max(1, int(after_hours_step_minutes))
        self.minute_vol = minute_vol
        self.minute_base_volume = minute_base_volume
        self.minute_volume_log_vol = minute_volume_log_vol
        self.after_hours_volume_fraction = after_hours_volume_fraction
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
        self.fin_base_revenue = fin_base_revenue
        self.fin_growth_qoq = fin_growth_qoq
        self.fin_seasonal_pct = fin_seasonal_pct
        self.fin_gross_margin = fin_gross_margin
        self.fin_opex_pct = fin_opex_pct
        self.fin_tax_rate = fin_tax_rate
        self.fin_shares = fin_shares
        self.fin_asset_turnover = fin_asset_turnover
        self.fin_filing_lag_days = fin_filing_lag_days

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
        # Symbol-dependent daily log drift in [-max_daily_drift, +max_daily_drift].
        drift = ((seed % 10001) / 10000.0 - 0.5) * 2.0 * self.max_daily_drift

        # Trading dates: the last `days` weekdays ending at `end` (weekends skipped).
        d = end if end is not None else _default_series_end()
        dates: list[date] = []
        while len(dates) < days:
            if d.weekday() < 5:  # Mon..Fri
                dates.append(d)
            d -= timedelta(days=1)
        dates.reverse()

        # A bar is a PURE FUNCTION OF (symbol, calendar date): each date gets
        # its own RNG stream, and the walk accumulates from a fixed epoch —
        # never from "the start of this request's window". Under the old
        # index-based walk, changing the requested count, trimming a bar, or
        # the calendar rolling a day shifted EVERY served value, so any test
        # characterised against a ticker rotted overnight. Now the same date
        # always carries the same bar, whatever window it is read through.
        walk_dates: list[date] = []
        d = _WALK_EPOCH
        first_needed = dates[0]
        while d <= dates[-1]:
            if d.weekday() < 5:
                walk_dates.append(d)
            d += timedelta(days=1)

        prev_close = self.base_prices.get(symbol, self.unknown_base_price)
        wanted = set(dates)
        by_date: dict[date, Bar] = {}
        for ts in walk_dates:
            rng = random.Random((seed << 32) ^ ts.toordinal())
            open_ = prev_close * math.exp(
                rng.gauss(0.0, self.daily_vol * self.gap_vol_fraction)
            )
            close = open_ * math.exp(rng.gauss(drift, self.daily_vol))
            if ts >= first_needed and ts in wanted:
                high = max(open_, close) * (
                    1.0 + abs(rng.gauss(0.0, self.daily_vol * self.range_vol_fraction))
                )
                low = min(open_, close) * (
                    1.0 - abs(rng.gauss(0.0, self.daily_vol * self.range_vol_fraction))
                )
                volume = self.base_volume * math.exp(
                    rng.gauss(0.0, self.volume_log_vol)
                )
                by_date[ts] = Bar(
                    ts=ts,
                    open=round(open_, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close, 4),
                    volume=float(round(volume)),
                )
            prev_close = close
        # Dates before the epoch (a longer window than the epoch allows)
        # simply do not exist — honest absence, not extrapolation.
        return [by_date[ts] for ts in dates if ts in by_date]

    # ------------------------------------------------------------------
    # Intraday bars (Phase C event replay — STUB ONLY, synthetic minutes)
    # ------------------------------------------------------------------

    def _session_minutes_et(self, day: date) -> list[datetime]:
        """Eastern-local bar starts for one synthetic session `day`.

        The dense regular session ``[open, close)`` at one-minute steps, then
        SPARSE after-hours starts every ``after_hours_step_minutes`` from the
        close up to (not including) ``after_hours_close_et``. Weekends have no
        session at all. Holidays and early closes are NOT modelled: this is a
        fixed grid for exercising consumers, never a claim about when the
        market was open.

        Times are built as Eastern-LOCAL wall clocks and converted by the
        caller, so a DST-transition day naturally yields the right UTC
        instants (09:30 ET is 13:30Z in summer and 14:30Z in winter) instead
        of a fixed UTC offset that would be an hour wrong for half the year.
        """
        if day.weekday() >= 5:  # Sat/Sun — no synthetic session
            return []
        minutes: list[datetime] = []
        cursor = datetime.combine(day, self.regular_open_et, tzinfo=_STUB_EASTERN)
        regular_close = datetime.combine(
            day, self.regular_close_et, tzinfo=_STUB_EASTERN
        )
        while cursor < regular_close:
            minutes.append(cursor)
            cursor += timedelta(minutes=1)
        after_close = datetime.combine(
            day, self.after_hours_close_et, tzinfo=_STUB_EASTERN
        )
        cursor = regular_close
        while cursor < after_close:
            minutes.append(cursor)
            cursor += timedelta(minutes=self.after_hours_step_minutes)
        return minutes

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
    ) -> list[IntradayBar]:
        """Deterministic synthetic minute bars over ``[start, end]``.

        STUB ONLY — NOT MARKET DATA. Every value is a seeded draw with no
        relationship to any real minute.

        Determinism is per ``(symbol, bar instant)``, exactly like
        :meth:`get_daily_bars` is per ``(symbol, date)``: each bar seeds its
        own RNG from the symbol hash and the bar's unix minute, so the SAME
        minute yields the SAME bar regardless of which window it is read
        through, how the window is split, or whether it arrived on the first
        call or the fifth. A consumer that fetches a window twice, or refetches
        an overlapping one, can therefore assert equality — which is the whole
        reason event-replay tests can be written against this provider.

        The day's anchor price is the synthetic DAILY bar's open for that date,
        so intraday and daily series for the same session agree rather than
        telling two different stories about the same day.

        Rejects a naive `start` / `end` with ``ValueError`` — the same refusal
        the real adapters make, so a caller that works against the stub cannot
        break the moment it is pointed at Alpaca. `timeframe` is accepted for
        interface compatibility and only ``"1Min"`` is synthesized; anything
        else returns ``[]`` rather than minute bars mislabelled as another
        resolution.
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )
        if timeframe != "1Min":
            return []  # honest absence, never minutes relabelled as 5Min/1Hour

        seed = zlib.crc32(symbol.encode("utf-8"))
        # Eastern dates the window can touch (a UTC window straddles at most
        # one extra Eastern date on each side).
        first_day = (start_utc.astimezone(_STUB_EASTERN) - timedelta(days=1)).date()
        last_day = (end_utc.astimezone(_STUB_EASTERN) + timedelta(days=1)).date()

        bars: list[IntradayBar] = []
        day = first_day
        while day <= last_day:
            minutes = self._session_minutes_et(day)
            if not minutes:
                day += timedelta(days=1)
                continue
            daily = self.get_daily_bars(symbol, 1, end=day)
            if not daily or daily[0].ts != day:
                day += timedelta(days=1)
                continue  # before the walk epoch: honestly no bars
            price = daily[0].open
            regular_close = datetime.combine(
                day, self.regular_close_et, tzinfo=_STUB_EASTERN
            )
            for minute_et in minutes:
                ts_utc = minute_et.astimezone(timezone.utc)
                # The seed is the bar's own unix minute, NOT its index in this
                # request's window: an index-based walk would move every bar
                # whenever the window changed.
                rng = random.Random((seed << 32) ^ int(ts_utc.timestamp() // 60))
                open_ = price
                close = open_ * math.exp(rng.gauss(0.0, self.minute_vol))
                high = max(open_, close) * (
                    1.0 + abs(rng.gauss(0.0, self.minute_vol * 0.5))
                )
                low = min(open_, close) * (
                    1.0 - abs(rng.gauss(0.0, self.minute_vol * 0.5))
                )
                centre = self.minute_base_volume
                if minute_et >= regular_close:
                    centre *= self.after_hours_volume_fraction
                volume = centre * math.exp(rng.gauss(0.0, self.minute_volume_log_vol))
                price = close
                if start_utc <= ts_utc <= end_utc:
                    bars.append(
                        IntradayBar(
                            ts=ts_utc,
                            open=round(open_, 4),
                            high=round(high, 4),
                            low=round(low, 4),
                            close=round(close, 4),
                            volume=int(round(volume)),
                        )
                    )
            day += timedelta(days=1)
        bars.sort(key=lambda b: b.ts)
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

    def get_option_contracts_window(
        self,
        underlying: str,
        as_of: date,
        dte_min: int,
        dte_max: int,
        spot: float,
        contract_type: str = "call",
    ) -> list[dict]:
        """STUB ONLY: a deterministic synthetic contract universe for the
        options backtest — one expiry in the middle of the DTE window,
        strikes on a 2.5%-of-spot grid within ±25% (pure function of the
        inputs, so tests reproduce exactly). Everything in stub-land is
        synthetic and labeled (plan §22.1)."""
        expiry = as_of + timedelta(days=(dte_min + dte_max) // 2)
        step = max(round(spot * 0.025, 2), 0.5)
        strikes = [
            round(spot + k * step, 2)
            for k in range(-10, 11)
            if spot + k * step > 0
        ]
        cp = "C" if contract_type == "call" else "P"
        return [
            {
                "ticker": (
                    f"{underlying}{expiry.strftime('%y%m%d')}{cp}"
                    f"{int(round(strike * 1000)):08d}"
                ),
                "strike_price": strike,
                "expiration_date": expiry,
            }
            for strike in strikes
        ]

    def get_option_daily_bars(
        self, option_ticker: str, start: date, end: date
    ) -> dict[date, tuple[float, float]]:
        """STUB ONLY: deterministic synthetic option bars — intrinsic value
        off the stub's OWN underlying closes plus a simple decaying time
        value. A pure function of (contract, date); weekends absent, and the
        two weekdays after the underlying seed's day-of-month == 13 are
        absent too so tests can exercise honest-gap handling."""
        m = _OCC_RE.match(option_ticker)
        if m is None:
            return {}
        underlying = m.group(1)
        expiry = datetime.strptime(m.group(2), "%y%m%d").date()
        cp = m.group(3)
        strike = int(m.group(4)) / 1000.0

        span = (end - start).days + 1
        if span <= 0:
            return {}
        bars = self.get_daily_bars(underlying, span + 40, end=end)
        by_date = {b.ts: b.close for b in bars}
        out: dict[date, tuple[float, float]] = {}
        prev_premium: float | None = None
        d = start
        while d <= end:
            close = by_date.get(d)
            if close is not None and d.day != 13:
                dte = max((expiry - d).days, 0)
                intrinsic = max(close - strike, 0.0) if cp == "C" else max(strike - close, 0.0)
                time_value = max(0.02, 0.05 * close * math.sqrt(dte / 365.0))
                premium = round(intrinsic + time_value, 4)
                open_px = prev_premium if prev_premium is not None else premium
                out[d] = (open_px, premium)
                prev_premium = premium
            d += timedelta(days=1)
        return out

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

    # ------------------------------------------------------------------
    # News (Phase 8 — STUB ONLY, synthetic deterministic articles)
    # ------------------------------------------------------------------

    #: Synthetic newsroom: (ticker, catalyst headline fragment). Parameters,
    #: never truths — reachable ONLY under MARKET_DATA_PROVIDER=stub.
    STUB_NEWS_UNIVERSE: tuple[tuple[str, str], ...] = (
        ("NVDA", "raises full-year guidance on datacenter demand"),
        ("AMD", "unveils next-gen accelerator roadmap"),
        ("COST", "reports comparable sales above expectations"),
        ("JPM", "flags rising credit-card delinquencies"),
        ("XOM", "announces additional buyback tranche"),
        ("PFE", "receives FDA fast-track designation"),
    )

    # ------------------------------------------------------------------
    # Free-tier options surface (contracts reference + EOD prev bar) —
    # mirrors the MassiveProvider methods so the EOD options view and its
    # tests run against the same deterministic chain universe.
    # ------------------------------------------------------------------

    @staticmethod
    def _occ_ticker(symbol: str, expiry: date, right: str, strike: float) -> str:
        return (
            f"O:{symbol}{expiry.strftime('%y%m%d')}{right}"
            f"{int(round(strike * 1000)):08d}"
        )

    def get_option_contracts(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        max_pages: int = 2,
    ) -> list[dict]:
        """Contract REFERENCE rows from the deterministic stub chain.

        The chain is generated AS OF the caller's window start (falling back
        to the anchored stub day), so the reference view follows the
        caller's clock exactly like the real endpoint does.
        """
        as_of = expiration_gte or _default_series_end()
        spot = self.get_daily_bars(underlying, 1)[-1].close
        rows: list[dict] = []
        for q in self.get_option_chain(underlying, spot, as_of):
            if expiration_gte is not None and q.expiry < expiration_gte:
                continue
            if expiration_lte is not None and q.expiry > expiration_lte:
                continue
            rows.append(
                {
                    "ticker": self._occ_ticker(underlying, q.expiry, q.right, q.strike),
                    "contract_type": "call" if q.right == "C" else "put",
                    "strike_price": q.strike,
                    "expiration_date": q.expiry,
                    "shares_per_contract": 100.0,
                }
            )
        return rows

    def get_option_prev_bar(self, option_ticker: str) -> dict | None:
        """Deterministic EOD bar for one stub contract.

        Priced directly from the same Black-Scholes helper the stub chain
        uses (spot = anchored last close, per-symbol base IV) — a pure
        function of the OCC ticker, independent of which as_of generated the
        contract list.
        """
        m = re.fullmatch(
            r"O:([A-Z.]{1,6})(\d{6})([CP])(\d{8})", option_ticker.strip()
        )
        if m is None:
            return None
        symbol = m.group(1)
        expiry = datetime.strptime(m.group(2), "%y%m%d").date()
        right, strike = m.group(3), int(m.group(4)) / 1000.0
        as_of = _default_series_end()
        spot = self.get_daily_bars(symbol, 1)[-1].close
        dte = max((expiry - as_of).days, 1)
        iv = max(self._base_iv(symbol), self.iv_floor)
        mid = round(
            max(bs_greeks(spot, strike, dte / _DAYS_PER_YEAR, iv, right).price,
                self.min_mid),
            4,
        )
        prev_day = as_of - timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)
        return {
            "open": round(mid * 0.99, 4),
            "high": round(mid * 1.02, 4),
            "low": round(mid * 0.97, 4),
            "close": mid,
            "volume": 100.0,
            "vwap": mid,
            "date": prev_day.isoformat(),
        }

    # ------------------------------------------------------------------
    # Historical options (Phase I §36) — SYNTHETIC, deterministic, tests only
    # ------------------------------------------------------------------

    def list_option_contracts(
        self,
        underlying: str,
        *,
        expiration_date: date,
        as_of: date,
        right: str | None = None,
        limit: int = 250,
    ) -> list[OptionContractRef]:
        """STUB ONLY: a fixed synthetic strike ladder for the requested expiry.

        NOT MARKET DATA. These contracts were never listed anywhere: for ANY
        underlying and ANY expiry the ladder is calls and puts at strikes
        ``HIST_STRIKE_MIN..HIST_STRIKE_MAX`` step ``HIST_STRIKE_STEP`` (90..110
        by 5, around a nominal 100 spot). It exists so the §36 implied-move
        pipeline and its SQLite tests can run end-to-end without a network,
        and so their assertions can name exact strikes.

        Deliberately a PURE function of ``(underlying, expiration_date,
        right)``: ``as_of`` is accepted to match the Protocol and is NOT read.
        The real point-in-time behaviour — a strike listed after ``as_of``
        being absent — is the server's to model, and faking it here with a
        seeded rule would let a test pass against invented as-of semantics
        that no provider implements.

        ``limit`` is likewise accepted and unused: the ladder is 10 contracts
        and never pages.
        """
        symbol = (underlying or "").strip().upper()
        if not symbol:
            return []
        # Both vocabularies in play: the OCC letter and the reference
        # endpoint's spelled-out word. Anything else filters to nothing rather
        # than silently defaulting to calls — the wrong leg is a wrong number.
        text = (right or "").strip().upper()
        if not text:
            rights: tuple[str, ...] = ("C", "P")
        elif text in ("C", "CALL"):
            rights = ("C",)
        elif text in ("P", "PUT"):
            rights = ("P",)
        else:
            return []

        refs: list[OptionContractRef] = []
        steps = int(round((HIST_STRIKE_MAX - HIST_STRIKE_MIN) / HIST_STRIKE_STEP))
        for i in range(steps + 1):
            strike = round(HIST_STRIKE_MIN + i * HIST_STRIKE_STEP, 2)
            for r in rights:
                refs.append(
                    OptionContractRef(
                        ticker=self._occ_ticker(symbol, expiration_date, r, strike),
                        underlying=symbol,
                        expiry=expiration_date,
                        right=r,
                        strike=strike,
                    )
                )
        refs.sort(key=lambda ref: (ref.strike, ref.right))
        return refs

    def get_option_history_bars(
        self, option_ticker: str, start: date, end: date
    ) -> list[Bar]:
        """STUB ONLY: synthetic daily bars for one contract over ``[start, end]``.

        NOT MARKET DATA — no option ever traded at these premiums. The close
        is a straight-line decay from the contract's own start,
        ``max(HIST_PREMIUM_FLOOR, HIST_PREMIUM_BASE - HIST_PREMIUM_DECAY_PER_DAY
        * day_index)``, with the put offset by ``HIST_PUT_OFFSET`` so a
        straddle is not two copies of one leg, and a strike term so two
        strikes never price identically. ``day_index`` counts WEEKDAYS from
        `start`, making the series a pure function of ``(ticker, start, end)``
        — identical across calls and processes, which is the only property the
        §36 tests need from it.

        Weekends are ABSENT (not zero-filled), so callers exercise the same
        gap handling real data forces on them. An unparseable ticker returns
        ``[]`` — the honest "no bars" answer, matching a provider that does not
        know the symbol. A reversed window raises, matching Massive.
        """
        if end < start:
            raise ValueError(
                f"get_option_history_bars window is reversed: start={start} > "
                f"end={end}; an empty window is expressed as start == end"
            )
        m = re.fullmatch(
            r"O:([A-Z.]{1,6})(\d{6})([CP])(\d{8})", (option_ticker or "").strip()
        )
        if m is None:
            return []
        right = m.group(3)
        strike = int(m.group(4)) / 1000.0

        bars: list[Bar] = []
        day = start
        index = 0
        while day <= end:
            if day.weekday() < 5:  # weekends are absent, never synthesized
                close = max(
                    HIST_PREMIUM_FLOOR,
                    HIST_PREMIUM_BASE
                    - HIST_PREMIUM_DECAY_PER_DAY * index
                    + (HIST_PUT_OFFSET if right == "P" else 0.0)
                    + (strike - 100.0) * 0.01,
                )
                close = round(close, 4)
                bars.append(
                    Bar(
                        ts=day,
                        open=round(close * 1.01, 4),
                        high=round(close * 1.03, 4),
                        low=round(close * 0.97, 4),
                        close=close,
                        volume=HIST_BAR_VOLUME,
                    )
                )
                index += 1
            day += timedelta(days=1)
        return bars

    def get_news(
        self,
        limit: int = 50,
        published_after: datetime | None = None,
    ) -> list[NewsArticle]:
        """Deterministic synthetic articles for dev/tests, newest first.

        Article ids are stable per (ticker, anchor date) so ingestion-dedup
        behaves exactly like it will against real Massive ids. All content is
        SYNTHETIC (see the module docstring) — a stub article can never be
        mistaken for a real citation because its url is a stub:// scheme.
        """
        anchor = _default_series_end()
        articles: list[NewsArticle] = []
        for i, (ticker, headline) in enumerate(self.STUB_NEWS_UNIVERSE[:limit]):
            published = datetime(
                anchor.year, anchor.month, anchor.day, 13, 30, tzinfo=timezone.utc
            ) - timedelta(hours=i)
            if published_after is not None and published <= published_after:
                continue
            articles.append(
                NewsArticle(
                    source_id=f"stub-news-{anchor.isoformat()}-{ticker}",
                    title=f"{ticker} {headline}",
                    publisher="Stub Newswire (synthetic)",
                    published_at=published,
                    url=f"stub://news/{anchor.isoformat()}/{ticker.lower()}",
                    tickers=(ticker,),
                    description=(
                        f"SYNTHETIC test article: {ticker} {headline}. "
                        "Generated by the stub provider for development."
                    ),
                )
            )
        return articles

    #: Synthetic per-day catalyst headlines for the windowed feed. Index
    #: ``day_index % len`` picks the day's story, so a window of N days
    #: produces N different-looking headlines per ticker. Parameters, never
    #: truths — reachable ONLY under MARKET_DATA_PROVIDER=stub.
    STUB_NEWS_WINDOW_HEADLINES: tuple[tuple[str, str], ...] = (
        ("EARNINGS", "{ticker} posts quarterly results above consensus"),
        ("GUIDANCE", "{ticker} raises full-year revenue guidance"),
        ("PRODUCT", "{ticker} launches its next-generation platform"),
        ("CONTRACT", "{ticker} wins a multi-year supply agreement"),
        ("ANALYST_REVISION", "{ticker} upgraded to Buy on margin outlook"),
        ("REGULATION", "{ticker} faces a new regulatory review"),
        ("MANAGEMENT", "{ticker} names a new chief financial officer"),
    )

    #: Publishers the synthetic newsroom cycles through. Deliberately spans
    #: the source-quality tiers the evidence engine grades on, so a consumer
    #: test sees more than one quality band without hand-building fixtures.
    STUB_NEWS_PUBLISHERS: tuple[str, ...] = (
        "Reuters (synthetic)",
        "Benzinga (synthetic)",
        "The Motley Fool (synthetic)",
    )

    #: Every Nth day, the day's article is also emitted by a SECOND publisher
    #: under the same headline — a syndicated copy, which is exactly the
    #: near-duplicate the dedup stage exists to collapse.
    STUB_NEWS_SYNDICATION_EVERY: int = 3

    #: Every Nth day additionally carries an OFF-TOPIC article: tagged with
    #: the ticker by the provider but about the broader tape, so relevance
    #: filtering has something real to exclude.
    STUB_NEWS_OFF_TOPIC_EVERY: int = 4

    def get_news_window(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> list[NewsArticle]:
        """Deterministic synthetic articles for `tickers` in ``[start, end]``.

        STUB ONLY — NOT NEWS. Every headline is generated from the tables
        above and every ``url`` uses the ``stub://`` scheme, so a synthetic
        article can never be mistaken for a real citation.

        Determinism is per ``(ticker, UTC day)``, the news analogue of
        :meth:`get_intraday_bars` being deterministic per ``(symbol, minute)``:
        the same day yields the same articles with the same ``source_id``
        regardless of which window it is read through or how the window is
        split. A consumer can therefore fetch overlapping windows and assert
        that the ingest de-duplicates rather than doubling.

        The generated corpus is deliberately AWKWARD, because the evidence
        engine's job is to handle awkward corpora:

        - **Syndicated duplicates** (every
          :attr:`STUB_NEWS_SYNDICATION_EVERY` days): the same headline from a
          second publisher, minutes apart — the near-duplicate that dedup and
          clustering must collapse into one development.
        - **Off-topic articles** (every :attr:`STUB_NEWS_OFF_TOPIC_EVERY`
          days): tagged with the ticker but about the market at large, so
          relevance scoring has something that SHOULD score low.

        Rejects a naive `start` / `end` with ``ValueError`` — the same refusal
        the real adapters make, so a caller that works against the stub cannot
        break the moment it is pointed at Alpaca or Massive.
        """
        start_utc = require_aware_utc(start, "start")
        end_utc = require_aware_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError(
                f"end ({end_utc.isoformat()}) precedes start "
                f"({start_utc.isoformat()}) — an empty window is expressed as "
                "start == end, never as a reversed one"
            )
        symbols = list(dict.fromkeys(
            t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()
        ))
        if not symbols or limit <= 0:
            return []

        by_id: dict[str, NewsArticle] = {}
        for symbol in symbols:
            day = start_utc.date()
            last_day = end_utc.date()
            while day <= last_day:
                for article in self._stub_news_for_day(symbol, day):
                    if article.published_at < start_utc or article.published_at > end_utc:
                        continue  # the day straddles the window's edges
                    by_id.setdefault(article.source_id, article)
                day += timedelta(days=1)
        ordered = sorted(
            by_id.values(), key=lambda a: (a.published_at, a.source_id), reverse=True
        )
        return ordered[:limit]

    def _stub_news_for_day(self, ticker: str, day: date) -> list[NewsArticle]:
        """The synthetic articles for one ``(ticker, UTC day)``, newest last.

        Split out from :meth:`get_news_window` so the per-day corpus is a
        pure function of its two arguments — that is what makes the same day
        read identically through any window.
        """
        day_index = day.toordinal()
        category, template = self.STUB_NEWS_WINDOW_HEADLINES[
            day_index % len(self.STUB_NEWS_WINDOW_HEADLINES)
        ]
        headline = template.format(ticker=ticker)
        publisher = self.STUB_NEWS_PUBLISHERS[day_index % len(self.STUB_NEWS_PUBLISHERS)]
        stamp = day.isoformat()
        base = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc)

        articles = [
            NewsArticle(
                source_id=f"stub-news-{stamp}-{ticker}-0",
                title=headline,
                publisher=publisher,
                published_at=base,
                url=f"stub://news/{stamp}/{ticker.lower()}/0",
                tickers=(ticker,),
                description=(
                    f"SYNTHETIC test article ({category}): {headline}. "
                    "Generated by the stub provider for development."
                ),
            )
        ]
        if day_index % self.STUB_NEWS_SYNDICATION_EVERY == 0:
            # A syndicated copy: same story, different newsroom, 20 minutes
            # later — a near-duplicate, not a second development.
            other = self.STUB_NEWS_PUBLISHERS[
                (day_index + 1) % len(self.STUB_NEWS_PUBLISHERS)
            ]
            articles.append(
                NewsArticle(
                    source_id=f"stub-news-{stamp}-{ticker}-syndicated",
                    title=headline,
                    publisher=other,
                    published_at=base + timedelta(minutes=20),
                    url=f"stub://news/{stamp}/{ticker.lower()}/syndicated",
                    tickers=(ticker,),
                    description=(
                        f"SYNTHETIC syndicated copy ({category}): {headline}. "
                        "Generated by the stub provider for development."
                    ),
                )
            )
        if day_index % self.STUB_NEWS_OFF_TOPIC_EVERY == 0:
            # Tagged with the ticker by the "provider" but about the tape at
            # large: the article relevance scoring should push aside.
            articles.append(
                NewsArticle(
                    source_id=f"stub-news-{stamp}-{ticker}-offtopic",
                    title="Stocks drift as traders await the next inflation print",
                    publisher="Stub Newswire (synthetic)",
                    published_at=base + timedelta(hours=2),
                    url=f"stub://news/{stamp}/{ticker.lower()}/offtopic",
                    tickers=(ticker,),
                    description=(
                        "SYNTHETIC off-topic article: broad market commentary "
                        "carrying no company-specific development. Generated by "
                        "the stub provider for development."
                    ),
                )
            )
        return articles

    # ------------------------------------------------------------------
    # Fundamentals (Phase E2 — SYNTHETIC statements, tests/dev only)
    # ------------------------------------------------------------------

    def get_financials(
        self, ticker: str, *, timeframe: str = "quarterly", limit: int = 12
    ) -> list[FinancialStatement]:
        """Deterministic synthetic statements for `ticker`, newest first.

        NOT FUNDAMENTALS (see the module docstring): every figure is generated
        from a seeded per-ticker trend, and the ``source_filing_url`` is a
        ``stub://`` scheme so a synthetic statement can never be mistaken for
        a real SEC citation.

        Two properties matter for the tests that consume this:

        1. **Only the fields Massive actually reports exist.** cash, capex,
           depreciation, receivables and interest expense are ABSENT here
           exactly as they are absent from the real provider (contract §3), so
           a metric that needs them resolves to "unavailable + reason" in the
           test suite for the same reason it will in production. Filling them
           in would let a metric pass its tests and then fail live.
        2. **``acceptance_datetime`` is a real instant, later than the period
           end** (``fin_filing_lag_days`` after it), so as-of tests filtering
           on the §85 key have something to bite on: a statement is invisible
           to any as_of before its acceptance.

        ``timeframe`` accepts ``quarterly``, ``annual`` and ``ttm``; ttm and
        annual are the trailing-four-quarter sums of the same series, so the
        blocks stay arithmetically consistent with the quarters.
        """
        symbol = (ticker or "").strip().upper()
        period = (timeframe or "").strip().lower()
        if not symbol or limit <= 0:
            return []
        if period not in ("quarterly", "annual", "ttm"):
            raise ValueError(
                f"unsupported financials timeframe {timeframe!r} — the stub "
                "serves quarterly, annual and ttm"
            )

        anchor = _default_series_end()
        # Quarter ends already ACCEPTED by the anchor date, newest first. The
        # window is over-fetched so ttm/annual sums always have four quarters
        # behind them even for the oldest period returned.
        needed = limit if period == "quarterly" else limit * 4 + 4
        quarter_ends = self._fin_quarter_ends(anchor, needed + 4)
        if not quarter_ends:
            return []

        seed = zlib.crc32(f"fin:{symbol}".encode("utf-8"))
        # Per-ticker scale and margin offsets: stable across processes, so a
        # characterised expectation for a ticker does not rot.
        scale = 0.5 + (seed % 1000) / 1000.0 * 1.5
        margin_offset = ((seed >> 10) % 200) / 1000.0 - 0.1

        quarters = [
            self._fin_quarter(symbol, end, scale, margin_offset)
            for end in quarter_ends
        ]
        if period == "quarterly":
            return quarters[:limit]

        rolled: list[FinancialStatement] = []
        for i in range(len(quarters)):
            window = quarters[i : i + 4]
            if len(window) < 4:
                break  # fewer than four quarters cannot make a trailing year
            rolled.append(self._fin_rollup(window, period))
            if len(rolled) >= limit:
                break
        return rolled

    def _fin_quarter_ends(self, anchor: date, count: int) -> list[date]:
        """The last `count` fiscal quarter ends ACCEPTED on or before `anchor`.

        A quarter that has ended but whose synthetic filing has not been
        accepted yet is excluded — the same shape the real calendar has, and
        the reason an as-of test can find a "not yet filed" gap.
        """
        ends: list[date] = []
        year = anchor.year + 1
        while len(ends) < count and year > anchor.year - 40:
            for month, day in reversed(_FIN_QUARTER_ENDS):
                end = date(year, month, day)
                if end >= anchor:
                    continue
                if self._fin_acceptance(end) > datetime(
                    anchor.year, anchor.month, anchor.day, 23, 59, tzinfo=timezone.utc
                ):
                    continue
                ends.append(end)
                if len(ends) >= count:
                    break
            year -= 1
        return ends

    def _fin_acceptance(self, period_end: date) -> datetime:
        """The synthetic instant the statement for `period_end` became public."""
        filed = period_end + timedelta(days=self.fin_filing_lag_days)
        return datetime(
            filed.year,
            filed.month,
            filed.day,
            DEFAULT_FIN_ACCEPTANCE_HOUR,
            DEFAULT_FIN_ACCEPTANCE_MINUTE,
            tzinfo=timezone.utc,
        )

    def _fin_quarter(
        self, symbol: str, end: date, scale: float, margin_offset: float
    ) -> FinancialStatement:
        """One synthetic quarterly statement — a pure function of (symbol, end)."""
        quarter_index = (end.month - 1) // 3  # 0..3 -> Q1..Q4
        # Quarters since the walk epoch, so the trend is anchored to the
        # calendar and not to "the start of this request's window".
        elapsed = (end.year - _WALK_EPOCH.year) * 4 + quarter_index
        trend = math.exp(math.log1p(self.fin_growth_qoq) * elapsed)
        seasonal = 1.0 + self.fin_seasonal_pct * math.sin(
            2 * math.pi * quarter_index / 4.0
        )
        revenue = self.fin_base_revenue * scale * trend * seasonal

        gross_margin = min(0.9, max(0.05, self.fin_gross_margin + margin_offset))
        gross_profit = revenue * gross_margin
        opex = revenue * self.fin_opex_pct
        operating_income = gross_profit - opex
        pretax = operating_income  # no interest line: the provider reports none
        tax = pretax * self.fin_tax_rate
        net_income = pretax - tax
        shares = self.fin_shares * scale
        assets = revenue / self.fin_asset_turnover
        current_assets = assets * 0.35
        current_liabilities = assets * 0.28
        long_term_debt = assets * 0.22
        liabilities = current_liabilities + long_term_debt
        equity = assets - liabilities
        operating_cash_flow = net_income * 1.25

        values = {
            "income_statement.revenues": revenue,
            "income_statement.cost_of_revenue": revenue - gross_profit,
            "income_statement.gross_profit": gross_profit,
            "income_statement.operating_expenses": opex,
            "income_statement.operating_income_loss": operating_income,
            "income_statement.income_loss_from_continuing_operations_before_tax": pretax,
            "income_statement.income_tax_expense_benefit": tax,
            "income_statement.net_income_loss": net_income,
            "income_statement.basic_earnings_per_share": net_income / shares,
            "income_statement.diluted_earnings_per_share": net_income / (shares * 1.01),
            "income_statement.basic_average_shares": shares,
            "income_statement.diluted_average_shares": shares * 1.01,
            "balance_sheet.assets": assets,
            "balance_sheet.current_assets": current_assets,
            "balance_sheet.noncurrent_assets": assets - current_assets,
            "balance_sheet.liabilities": liabilities,
            "balance_sheet.current_liabilities": current_liabilities,
            "balance_sheet.noncurrent_liabilities": long_term_debt,
            "balance_sheet.long_term_debt": long_term_debt,
            "balance_sheet.inventory": current_assets * 0.18,
            "balance_sheet.equity": equity,
            "balance_sheet.equity_attributable_to_parent": equity,
            "cash_flow_statement.net_cash_flow_from_operating_activities": (
                operating_cash_flow
            ),
            "cash_flow_statement.net_cash_flow_from_investing_activities": (
                -operating_cash_flow * 0.4
            ),
            "cash_flow_statement.net_cash_flow_from_financing_activities": (
                -operating_cash_flow * 0.5
            ),
            "cash_flow_statement.net_cash_flow": operating_cash_flow * 0.1,
        }
        values = {k: round(v, 4) for k, v in values.items()}
        start = date(end.year, end.month - 2, 1)
        acceptance = self._fin_acceptance(end)
        return FinancialStatement(
            ticker=symbol,
            cik=f"{zlib.crc32(symbol.encode('utf-8')) % 10_000_000:010d}",
            timeframe="quarterly",
            fiscal_year=end.year,
            fiscal_period=f"Q{quarter_index + 1}",
            start_date=start,
            end_date=end,
            filing_date=acceptance.date(),
            acceptance_datetime=acceptance,
            source_filing_url=(
                f"stub://filings/{symbol.lower()}/{end.isoformat()}-q{quarter_index + 1}"
            ),
            values=values,
            raw_fields_count=len(values),
        )

    @staticmethod
    def _fin_rollup(
        window: list[FinancialStatement], timeframe: str
    ) -> FinancialStatement:
        """Sum four quarters into a TTM/annual statement.

        Flow lines (income, cash flow) are SUMMED; balance-sheet lines and
        per-share/share-count lines are point-in-time or per-period figures
        that a sum would corrupt, so they are taken from the NEWEST quarter —
        which is what a filer reports on a trailing statement.
        """
        newest = window[0]
        values: dict[str, float] = {}
        for key in newest.values:
            block = key.split(".", 1)[0]
            if block == "balance_sheet" or "share" in key or "per_share" in key:
                values[key] = newest.values[key]
                continue
            total = 0.0
            complete = True
            for statement in window:
                part = statement.values.get(key)
                if part is None:
                    complete = False
                    break
                total += part
            if complete:  # a partial sum would understate the year
                values[key] = round(total, 4)
        oldest = window[-1]
        return FinancialStatement(
            ticker=newest.ticker,
            cik=newest.cik,
            timeframe=timeframe,
            fiscal_year=newest.fiscal_year,
            fiscal_period="TTM" if timeframe == "ttm" else "FY",
            start_date=oldest.start_date,
            end_date=newest.end_date,
            filing_date=newest.filing_date,
            acceptance_datetime=newest.acceptance_datetime,
            source_filing_url=newest.source_filing_url,
            values=values,
            raw_fields_count=len(values),
        )
