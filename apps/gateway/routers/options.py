"""Options chain API — chain snapshot + contract selection (plan §9, §34).

Serves `GET /api/watchlist/{ticker}/options`: the full option chain for one
Watchlist symbol with per-contract greeks, an IV summary, and the Contract
Selector verdict merged into every row so the UI can render All / Eligible /
Recommended views (plan §34). Watchlist-gated exactly like the analysis
endpoint — data may exist only for Watchlist symbols (plan §4.2) — and
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
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.contracts import (
    ContractQuote,
    ScoredContract,
    select_contracts,
)
from libs.trading_core.features import realized_vol
from libs.trading_core.models import DirectionalBias
from libs.trading_core.signals import score_direction

from ..db import WatchlistItem, get_session
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


def _summary(chain: list[ContractQuote], spot: float, closes: list[float]) -> dict:
    """IV summary block (plan §9): ATM IV, straddle-implied expected move,
    realized vol and the IV-RV spread — honest nulls when undefined.

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
        calls = [c for c in chain if c.expiry == expiry and c.right == "C"]
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


@router.get("/{ticker}/options")
async def get_symbol_options(
    ticker: str,
    direction: Literal["AUTO", "BULL", "BEAR"] = Query(default="AUTO"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Option chain + selector verdicts for one Watchlist symbol (plan §9).

    404s for tickers not on the Watchlist (plan §4.2). Read-only — no audit
    event (chain reads are reads). ``as_of`` is the chain snapshot DATE:
    the stub chain is keyed by day, so two calls on the same day return
    byte-identical payloads (deterministic by construction).
    """
    ticker = ticker.upper()
    row = await session.execute(
        select(WatchlistItem).where(WatchlistItem.ticker == ticker)
    )
    if row.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{ticker} is not on the watchlist; historical data exists "
                "only for Watchlist symbols"
            ),
        )

    settings = get_settings()
    bars = await ensure_daily_bars(session, ticker, settings.market_data_provider)
    closes = [b.close for b in bars]
    spot = closes[-1]  # spot = last stored close (plan §9 v0)
    as_of = datetime.now(timezone.utc).date()

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

    provider = get_provider(settings.market_data_provider)
    chain = provider.get_option_chain(ticker, spot, as_of)

    if direction_used is not None:
        scored = select_contracts(chain, direction_used)
    else:
        scored = _neutral_verdicts(chain)

    # Unique expiries with their DTE, ascending.
    expiry_dte = sorted({(c.expiry, c.dte) for c in chain})
    return {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "spot": spot,
        "source": settings.market_data_provider,
        "direction_used": direction_used,
        "summary": _summary(chain, spot, closes),
        "expiries": [
            {"expiry": expiry.isoformat(), "dte": dte} for expiry, dte in expiry_dte
        ],
        "chain": [
            {
                "expiry": s.contract.expiry.isoformat(),
                "dte": s.contract.dte,
                "strike": s.contract.strike,
                "right": s.contract.right,
                "bid": s.contract.bid,
                "ask": s.contract.ask,
                "mid": s.contract.mid,
                "spread_pct": s.contract.spread_pct,
                "last": s.contract.last,
                "volume": s.contract.volume,
                "open_interest": s.contract.open_interest,
                "iv": s.contract.iv,
                "delta": s.contract.delta,
                "gamma": s.contract.gamma,
                "theta": s.contract.theta,
                "vega": s.contract.vega,
                "eligible": s.eligible,
                "fail_reasons": s.fail_reasons,
                "candidate_rank": s.rank,
                "score": s.score,
                "score_components": s.components,
            }
            for s in scored
        ],
    }
