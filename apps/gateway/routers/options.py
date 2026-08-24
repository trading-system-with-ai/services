"""Options chain API — chain snapshot + contract selection (plan §9, §34).

Serves `GET /api/watchlist/{ticker}/options`: the full option chain for one
Watchlist symbol with per-contract greeks, an IV summary, and the Contract
Selector verdict merged into every row so the UI can render All / Eligible /
Recommended views (plan §34). OPEN to any ticker like the analysis
endpoint (2026-08-20, §4.2 amended) — and
READ-ONLY: chain reads write no audit events (house rule; audit only on
state changes). The only write this path can trigger is the one-time lazy
bar backfill inside :func:`ensure_daily_bars`, which audits itself.

Direction resolution:

- ``direction=BULL`` / ``BEAR``: the caller's value is used as-is.
- ``direction=AUTO`` (default): the Directional Signal Engine's bias over
  the stored bars decides. A NEUTRAL bias yields ``direction_used: null``
  and NO candidates — every contract is returned ineligible, its own-side
  §9.1 filter verdicts intact plus the no-signal reason, because "NO TRADE
  is a valid output" (§44 rule 18).

All analytics come exclusively from libs.trading_core (options, contracts,
features, signals), so backtest and live share this exact code (plan §21).
"""
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import (
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.contracts import (
    ContractQuote,
    ScoredContract,
    select_contracts,
)
from libs.trading_core.features import realized_vol
from libs.trading_core.models import DirectionalBias
from libs.trading_core.signals import score_direction

from ..db import get_session
from ..deps import market_data_configured, require_market_data_provider
from ..risk_snapshot import NEW_YORK, record_atm_iv
from .analysis import EASTERN as EASTERN_TZ
from .analysis import ensure_daily_bars

router = APIRouter(prefix="/api/watchlist", tags=["options"])

# Tunable parameters (plan §6.2 house rule: parameters, never hardcoded truths).
REALIZED_VOL_PERIOD = 20  # rv20: annualized realized vol window (plan §6)
ATM_MIN_DTE = 30  # summary reads the nearest expiry with at least this DTE

# §44 rule 18: a NEUTRAL bias under AUTO means no directional signal exists,
# so no side is tradeable and no candidate is produced.
NO_SIGNAL_REASON = "no directional signal — NO TRADE is a valid output"

IV_RANK_NOTE = "requires IV history — arrives with real chain data"


def _neutral_verdicts(chain: list[ContractQuote]) -> list[ScoredContract]:
    """Verdicts when AUTO resolves to NEUTRAL (§44 rule 18): NO candidates.

    BOTH sides are evaluated (calls against BULL, puts against BEAR) so each
    contract keeps its own side's honest §9.1 filter verdicts — without the
    meaningless "wrong side" noise — and every contract is forced ineligible
    with :data:`NO_SIGNAL_REASON` appended. Nothing is scored or ranked:
    NO TRADE is a valid output, not an error.
    """
    bull = select_contracts(chain, "BULL")
    bear = select_contracts(chain, "BEAR")
    verdicts: list[ScoredContract] = []
    for contract, bull_scored, bear_scored in zip(chain, bull, bear):
        own_side = bull_scored if contract.right == "C" else bear_scored
        verdicts.append(
            ScoredContract(
                contract=contract,
                eligible=False,
                fail_reasons=[*own_side.fail_reasons, NO_SIGNAL_REASON],
            )
        )
    return verdicts


#: Short chain TTL: bounds provider REST calls when read surfaces poll (the
#: options view, positions, portfolio) — a research read a few seconds old
#: is the same snapshot for practical purposes. EXECUTION paths bypass with
#: ``max_age_seconds=0`` (§21/§42: orders re-run on live data, never cached).
CHAIN_CACHE_TTL_SECONDS = 20.0
# (provider, ticker) -> (fetched_at, as_of, chain)
_chain_cache: dict[tuple[str, str], tuple[datetime, date, list[ContractQuote]]] = {}


def build_option_chain(
    ticker: str, spot: float, *, max_age_seconds: float = CHAIN_CACHE_TTL_SECONDS
) -> tuple[date, list[ContractQuote]]:
    """Today's option chain snapshot for `ticker` around `spot` (plan §9).

    The SHARED chain-build helper: this options view, the §10 order gate
    chain (routers/orders.py) and the position monitor's option reads
    (routers/positions.py) all regenerate the chain through this one
    function, so every consumer sees the identical snapshot — the stub chain
    is keyed by (symbol, day), so two calls on the same day are
    byte-identical (deterministic by construction). Returns ``(as_of,
    chain)`` where ``as_of`` is the snapshot DATE.

    A build fetched within ``max_age_seconds`` is served from the in-process
    cache (same provider+ticker+day) so polling READ surfaces do not multiply
    provider REST calls; EXECUTION callers pass ``max_age_seconds=0`` and
    always hit the provider live (§21/§42 — orders never trust a cache).

    Raises :class:`libs.market_data.ProviderNotConfigured` when no provider is
    configured — a chain is pure market data and is never invented. Callers
    that must keep answering (the positions list, the portfolio risk view) use
    :func:`option_chain_or_none` instead and report honest nulls.
    """
    settings = get_settings()
    # The TRADING day is Eastern: after 20:00 ET the UTC calendar has already
    # rolled to tomorrow, and a UTC date here made the provider's
    # current-state guard reject the request every US evening (observed
    # live). Snapshots are keyed to the exchange's clock, so as_of must be.
    as_of = datetime.now(EASTERN_TZ).date()
    key = (settings.market_data_provider, ticker)
    if max_age_seconds > 0:
        cached = _chain_cache.get(key)
        if cached is not None:
            fetched_at, cached_as_of, cached_chain = cached
            age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
            if cached_as_of == as_of and age <= max_age_seconds:
                return cached_as_of, cached_chain
    provider = get_provider(settings.market_data_provider)
    chain = provider.get_option_chain(ticker, spot, as_of)
    _chain_cache[key] = (datetime.now(timezone.utc), as_of, chain)
    return as_of, chain


def option_chain_or_none(ticker: str, spot: float) -> list[ContractQuote] | None:
    """Today's chain for `ticker`, or ``None`` when market data is unconfigured.

    The degrading variant of :func:`build_option_chain`, for read views that
    stay 200 because their substance is real DB state (positions, portfolio
    risk). ``None`` means "no quote is knowable" and every option-derived
    field the caller would have filled becomes an honest null — never a zero,
    never a synthetic mid (§44 rule 18).
    """
    if not market_data_configured():
        return None
    try:
        _as_of, chain = build_option_chain(ticker, spot)
    except ProviderNotConfigured:
        return None
    return chain


def chain_iv_summary(
    chain: list[ContractQuote], spot: float, closes: list[float]
) -> dict:
    """IV summary block (plan §9): ATM IV, straddle-implied expected move,
    realized vol and the IV-RV spread — honest nulls when undefined.

    SHARED with the §10 order gate chain (routers/orders.py), whose
    VOLATILITY gate classifies the §7 regime off this block's ``atm_iv`` +
    ``rv20`` — one summary implementation, never duplicated (plan §21).

    ``atm_iv`` reads the closest-to-spot strike CALL on the nearest expiry
    with dte >= :data:`ATM_MIN_DTE`; ``expected_move_pct`` is the ATM
    straddle (call mid + put mid) at that node as a fraction of spot.
    ``iv_rank`` is an honest null with :data:`IV_RANK_NOTE` verbatim — it
    needs an IV history the stub chain cannot provide.
    """
    atm_iv: float | None = None
    expected_move_pct: float | None = None
    eligible_expiries = sorted({c.expiry for c in chain if c.dte >= ATM_MIN_DTE})
    if eligible_expiries:
        expiry = eligible_expiries[0]
        # Only contracts the provider actually priced an IV for — greekless
        # rows are real chain data but cannot define the ATM IV.
        calls = [
            c for c in chain
            if c.expiry == expiry and c.right == "C" and c.iv is not None
        ]
        if calls:
            # Deterministic tie-break: closest strike, then the lower one.
            atm_call = min(calls, key=lambda c: (abs(c.strike - spot), c.strike))
            atm_iv = atm_call.iv
            atm_put = next(
                (
                    c
                    for c in chain
                    if c.expiry == expiry
                    and c.strike == atm_call.strike
                    and c.right == "P"
                ),
                None,
            )
            if atm_put is not None:
                expected_move_pct = (atm_call.mid + atm_put.mid) / spot
    rv20 = realized_vol(closes, period=REALIZED_VOL_PERIOD)[-1]
    iv_rv_spread = (
        atm_iv - rv20 if atm_iv is not None and rv20 is not None else None
    )
    return {
        "atm_iv": atm_iv,
        "expected_move_pct": expected_move_pct,
        "rv20": rv20,
        "iv_rv_spread": iv_rv_spread,
        "iv_rank": None,
        "iv_rank_note": IV_RANK_NOTE,
    }


# ---------------------------------------------------------------------------
# EOD options reference view (provider-agnostic: contracts reference +
# previous-day bars — the chain view carries quotes/greeks when available).
# ---------------------------------------------------------------------------

#: Contracts window fetched for the EOD view (front expiries are the useful
#: ones; the reference endpoint pages at 1000 rows).
EOD_EXPIRY_WINDOW_DAYS = 90
#: Nearest-to-ATM strikes whose call+put previous-day bars are fetched.
#: 2 strikes -> <=4 /prev calls; +1 contracts call stays inside the free
#: tier's 5 requests/minute budget for one uncached page load.
EOD_ATM_STRIKES = 2
#: The EOD target expiry: first with at least this DTE (selector-adjacent
#: preference without pretending to select anything).
EOD_MIN_DTE = 14

# (provider, ticker, eastern_date) -> payload. EOD data does not change
# intraday. Provider-keyed so a runtime provider switch can never serve the
# old vendor's numbers; past-day entries are evicted on insert so the cache
# is bounded by the watchlist size.
_eod_cache: dict[tuple[str, str, date], dict] = {}


@router.get("/{ticker}/options/eod")
async def get_symbol_options_eod(
    ticker: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """END-OF-DAY options reference view (provider-agnostic).

    Serves the reference surface every supported provider carries: the
    CONTRACT REFERENCE (which expirations/strikes exist) plus
    PREVIOUS-SESSION EOD bars for the few nearest-to-ATM front-expiry
    contracts. This VIEW contains no quotes, greeks, IV or open interest —
    those live on the Options chain view when the provider serves them;
    nothing here is approximated in their place, and the §9 selector does
    NOT run on this data. Cached per (provider, ticker, Eastern day): EOD
    data is static intraday, and rate-limited plans spend at most 1
    contracts call + {EOD_ATM_STRIKES}×2 bar calls per uncached load.

    503 with code ``OPTION_DATA_NOT_AVAILABLE`` when the plan lacks even the
    reference endpoint. OPEN to any ticker (2026-08-20, §4.2 amended).
    """
    require_market_data_provider()
    ticker = ticker.upper()
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

    settings = get_settings()
    provider = get_provider(settings.market_data_provider)
    if getattr(provider, "get_option_contracts", None) is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "OPTION_DATA_NOT_AVAILABLE",
                "message": (
                    f"provider {settings.market_data_provider!r} does not "
                    "serve option contract reference data"
                ),
            },
        )

    today_eastern = datetime.now(EASTERN_TZ).date()
    cached = _eod_cache.get((settings.market_data_provider, ticker, today_eastern))
    if cached is not None:
        return cached

    bars = await ensure_daily_bars(session, ticker, settings.market_data_provider)
    spot = bars[-1].close

    try:
        contracts = provider.get_option_contracts(
            ticker,
            expiration_gte=today_eastern,
            expiration_lte=today_eastern + timedelta(days=EOD_EXPIRY_WINDOW_DAYS),
        )
    except CapabilityNotAvailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "OPTION_DATA_NOT_AVAILABLE", "message": str(exc)},
        ) from exc
    except MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Expirations summary — pure reference data, grouped.
    by_expiry: dict[date, list[dict]] = {}
    for c in contracts:
        by_expiry.setdefault(c["expiration_date"], []).append(c)
    expirations = [
        {
            "date": expiry.isoformat(),
            "dte": (expiry - today_eastern).days,
            "strikes": len({c["strike_price"] for c in rows_}),
            "calls": sum(1 for c in rows_ if c["contract_type"] == "call"),
            "puts": sum(1 for c in rows_ if c["contract_type"] == "put"),
        }
        for expiry, rows_ in sorted(by_expiry.items())
    ]

    # Front target expiry + nearest-to-ATM contracts, with EOD bars.
    target = next(
        (e for e in sorted(by_expiry) if (e - today_eastern).days >= EOD_MIN_DTE),
        max(by_expiry) if by_expiry else None,
    )
    atm_contracts: list[dict] = []
    if target is not None:
        rows_ = by_expiry[target]
        strikes = sorted({c["strike_price"] for c in rows_}, key=lambda s: abs(s - spot))
        for strike in strikes[:EOD_ATM_STRIKES]:
            for c in rows_:
                if c["strike_price"] != strike:
                    continue
                try:
                    prev = provider.get_option_prev_bar(c["ticker"])
                except (CapabilityNotAvailable, MarketDataError) as exc:
                    prev = None
                    logger_detail = str(exc)
                else:
                    logger_detail = None
                atm_contracts.append(
                    {
                        "ticker": c["ticker"],
                        "contract_type": c["contract_type"],
                        "strike": c["strike_price"],
                        "expiration_date": c["expiration_date"].isoformat(),
                        "dte": (c["expiration_date"] - today_eastern).days,
                        # EOD previous-session bar, or an honest null (an
                        # illiquid contract that did not trade / a fetch
                        # fault — never a zero, never yesterday's guess).
                        "prev_day": prev,
                        "prev_day_error": logger_detail,
                    }
                )

    payload = {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        # §25/§37 provenance: END OF DAY reference data — the UI labels it
        # so and must not style it as a live chain.
        "data_recency": "end_of_day",
        "spot_reference": spot,
        "spot_reference_note": "last stored daily close — not a live quote",
        "expirations": expirations,
        "target_expiry": target.isoformat() if target is not None else None,
        "atm_contracts": atm_contracts,
        # What this EOD VIEW does not contain — a fact about the view (it is
        # reference + previous-session bars), NOT a claim about the plan:
        # under a provider whose chain snapshot serves quotes/greeks (e.g.
        # Alpaca), those live on the Options chain view instead.
        "not_in_this_view": [
            "bid/ask quotes",
            "greeks",
            "implied volatility",
            "open interest",
        ],
        "note": (
            f"End-of-day options reference from provider "
            f"{settings.market_data_provider!r}: contract reference + "
            "previous-session bars only. Contract selection (§9) needs live "
            "quotes and greeks from the chain snapshot endpoint."
        ),
    }
    # Evict past-day entries (bounded by watchlist size), then store under
    # the provider-qualified key.
    for key in [k for k in _eod_cache if k[2] != today_eastern]:
        del _eod_cache[key]
    _eod_cache[(settings.market_data_provider, ticker, today_eastern)] = payload
    return payload


#: IV inversion is unidentifiable when an option trades at (almost) pure
#: intrinsic value or sits at |delta| ~ 1: any tiny premium noise maps to a
#: wild IV. Vendors emit degenerate numbers there (observed live: a $1-strike
#: call on a $13 stock with IV 0.035% and delta 0.99999...). These flags mark
#: such rows so the UI can annotate them — the vendor's number stays visible
#: verbatim, never replaced.
IV_DEGENERATE_ABS_DELTA = 0.98
IV_DEGENERATE_MIN_EXTRINSIC = 0.02  # $/share
IV_DEGENERATE_MIN_IV = 0.02  # 2% annualized — below this, inversion noise


def _chain_row_payload(s: ScoredContract, spot: float) -> dict:
    """One chain row for the API — honest about quote basis and IV quality.

    - ``price_basis == "day_close"`` (quotes-less plan): bid/ask/spread are
      UNKNOWN — serialized as nulls, never 0.00 (a zero bid is a real market
      state; unknown is not it). ``mid`` is the session close, labeled by
      ``price_basis``.
    - ``iv_unreliable``: deep-ITM/OTM rows where premium ≈ intrinsic or
      |delta| ≈ 1 — the vendor's IV is mathematically unidentifiable there.
      The value still renders; the flag lets the UI say why it looks odd.
    """
    c = s.contract
    quoteless = c.price_basis == "day_close"
    intrinsic = max(spot - c.strike, 0.0) if c.right == "C" else max(c.strike - spot, 0.0)
    extrinsic = c.mid - intrinsic
    # Greeks may be honest nulls (provider omits them on deep wings) — the
    # degeneracy flag only applies where an IV exists to flag.
    iv_unreliable = c.iv is not None and (
        (c.delta is not None and abs(c.delta) >= IV_DEGENERATE_ABS_DELTA)
        or extrinsic <= IV_DEGENERATE_MIN_EXTRINSIC
        or c.iv < IV_DEGENERATE_MIN_IV
    )
    return {
        "expiry": c.expiry.isoformat(),
        "dte": c.dte,
        "strike": c.strike,
        "right": c.right,
        "bid": None if quoteless else c.bid,
        "ask": None if quoteless else c.ask,
        "mid": c.mid,
        "spread_pct": None if quoteless else c.spread_pct,
        "price_basis": c.price_basis,
        "last": c.last,
        "volume": c.volume,
        "open_interest": c.open_interest,
        "iv": c.iv,
        "iv_unreliable": iv_unreliable,
        "delta": c.delta,
        "gamma": c.gamma,
        "theta": c.theta,
        "vega": c.vega,
        "eligible": s.eligible,
        "fail_reasons": s.fail_reasons,
        "candidate_rank": s.rank,
        "score": s.score,
        "score_components": s.components,
    }


@router.get("/{ticker}/options")
async def get_symbol_options(
    ticker: str,
    direction: Literal["AUTO", "BULL", "BEAR"] = Query(default="AUTO"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Option chain + selector verdicts for one Watchlist symbol (plan §9).

    OPEN to any ticker (2026-08-20, §4.2 amended). 503
    ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is configured
    — spot, the chain, the greeks and the IV summary are all market data or
    computed from it, so nothing here can be served honestly without it.
    Read-only — no audit event (chain reads are reads). ``as_of`` is the chain
    snapshot DATE: the stub chain is keyed by day, so two calls on the same
    day return byte-identical payloads (deterministic by construction).
    """
    require_market_data_provider()
    ticker = ticker.upper()
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

    settings = get_settings()
    bars = await ensure_daily_bars(session, ticker, settings.market_data_provider)
    closes = [b.close for b in bars]
    spot = closes[-1]  # spot = last stored close (plan §9 v0)

    # Resolve the direction (plan §9): explicit param wins; AUTO defers to
    # the Directional Signal Engine over the same stored bars.
    if direction == "AUTO":
        signal = score_direction(
            closes,
            [b.high for b in bars],
            [b.low for b in bars],
            volumes=[b.volume for b in bars],
        )
        direction_used = (
            signal.bias.value if signal.bias is not DirectionalBias.NEUTRAL else None
        )
    else:
        direction_used = direction

    as_of, chain = build_option_chain(ticker, spot)

    if direction_used is not None:
        scored = select_contracts(chain, direction_used)
    else:
        scored = _neutral_verdicts(chain)

    # atm_iv_daily (spec §24; audit §7.1): this read already computed the ATM
    # IV and today's code throws it away — but empirical IV shocks and IV
    # rank need the HISTORY, and it can only be captured where it is
    # computed. Persist it as an observation and COMMIT it: nothing else is
    # pending on this read's session, so the commit carries exactly this one
    # row and the chain view stays a read of the chain. Best-effort by
    # contract (record_atm_iv never raises, and the commit is guarded), so a
    # storage fault can never turn a chain view into an error. INTERNALLY
    # CALCULATED provenance in `source` — never vendor IV history. Still no
    # audit event: an observation is not a decision.
    summary = chain_iv_summary(chain, spot, closes)
    await record_atm_iv(
        session,
        ticker,
        bar_date=datetime.now(NEW_YORK).date(),
        atm_iv=summary["atm_iv"],
        spot=spot,
        source=f"{settings.market_data_provider}_chain",
    )
    try:
        await session.commit()
    except Exception:  # noqa: BLE001 — best-effort: a chain read must not 5xx
        await session.rollback()

    # Unique expiries with their DTE, ascending.
    expiry_dte = sorted({(c.expiry, c.dte) for c in chain})
    return {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "spot": spot,
        "source": settings.market_data_provider,
        "direction_used": direction_used,
        "summary": summary,
        "expiries": [
            {"expiry": expiry.isoformat(), "dte": dte} for expiry, dte in expiry_dte
        ],
        "chain": [_chain_row_payload(s, spot) for s in scored],
    }
