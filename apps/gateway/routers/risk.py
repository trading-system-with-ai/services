"""Risk research endpoints — stress scenarios (Phase D) and VaR/ES model
validation (Phase E).

Two POSTs live here, and they are the same KIND of thing: a READ of the
current book (or of its P&L history) that persists its result and writes NO
audit event.

- ``POST /api/risk/stress/run`` — one user-defined hypothetical scenario;
- ``POST /api/risk/validation/run`` — one walk-forward VaR/ES backtest of the
  whole model grid (spec §42/§43; design §9.4).

The module docstring below is about the stress endpoint; see
:func:`run_model_validation` for the validation one.

User-defined stress scenarios — `POST /api/risk/stress/run` (Phase D).

Risk spec §26 (hypothetical stress testing), §51 (stress UI, "allow
user-defined hypothetical scenarios"), §56 (keep history); Phase B/D design
contract §8.5.

WHAT THIS ENDPOINT IS. A READ of the CURRENT book under a hypothesis the user
supplies: "if every name moved x %, IV moved y % relative and z days passed,
what would this book be worth?". It places nothing, changes nothing, and
decides nothing. It is SHADOW by construction — the scenario it runs is a
``USER`` scenario, ``validated=False``, and no Tier 0 number anywhere reads
its output.

TWO CONSEQUENCES OF "IT IS A READ", both deliberate and both tested:

- **No audit event.** House rule: read views write no audit events, and this
  is a read (portfolio.py:44-45). A hypothesis about a book is not a decision
  about it. The RISK_DECISION audit stays the one record of the one decision.
- **It still persists.** Spec §56 says do not store only the latest value: the
  run lands in ``stress_runs`` with ``snapshot_id`` NULL (there is no snapshot
  build behind it) so the history of what the user asked, and what the book
  answered, survives. Persistence is not auditing; one is a measurement
  series, the other is a decision record.

RANGES (§8.5). ``equity_shock`` ∈ [−0.9, 2], ``iv_shock`` ∈ [−0.9, 5],
``days_forward`` ∈ [0, 365] — validated by pydantic, so an out-of-range value
is a 422 with the field named, never a clamped number the user did not ask
for. The bounds are PARAMETERS (module constants below), not magic: −0.9
because a stock cannot go to zero in this model, +2 because a 200 % move is
already far past anything the pricer can be trusted on, 365 days because the
option book's tenors are shorter than that by construction.

The shocks keep the library's conventions exactly (design §8.2):
``equity_shock`` is FRACTIONAL and UNIFORM across every underlying — the
beta = 1 assumption, reported on the row as ``uniform_beta_1`` — and
``iv_shock`` is RELATIVE multiplicative on the IV LEVEL (+0.20 ⇒
``iv1 = iv0 × 1.20``), not "+20 vol points".
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.options.reval import (
    METHOD_DELTA_LINEAR as REVAL_METHOD_DELTA_LINEAR,
    METHOD_FULL_REVAL as REVAL_METHOD_FULL_REVAL,
)
from libs.trading_core.risk.models.stress import (
    KIND_USER,
    IV_SHOCK_SOURCE_SPECIFIED,
    Scenario,
    run_scenario,
)

from ..db import StressRunRow, get_or_create_portfolio, get_session
from ..risk_validation import (
    DEFAULT_WINDOW as VALIDATION_DEFAULT_WINDOW,
    MIN_FORECASTS as VALIDATION_MIN_FORECASTS,
    TRIGGER_ON_DEMAND as VALIDATION_TRIGGER_ON_DEMAND,
    run_model_backtests,
)
from ..deps import (
    broker_configured,
    market_data_configured,
    resolve_broker,
    simulated_broker_mode,
)
from ..risk_snapshot import (
    TRIGGER_ON_DEMAND,
    _scenario_row_api,
    _jsonable,
    build_risk_snapshot,
    stress_legs_from_book,
)

logger = logging.getLogger("apps.gateway.routers.risk")

router = APIRouter(prefix="/api/risk", tags=["risk"])

# --- Request bounds: PARAMETERS, never magic numbers (house rule) ----------

#: A stock cannot reach zero in this model (the pricer needs ``S1 > 0``), so
#: the down bound stops just short of it; the up bound is far past anything a
#: single-scenario reprice should be trusted for, and exists to reject typos
#: (an "equity +200" meaning percent) rather than to bless a +200 % scenario.
EQUITY_SHOCK_MIN = -0.9
EQUITY_SHOCK_MAX = 2.0

#: RELATIVE multiplicative on the IV LEVEL. −0.9 is a 90 % vol collapse;
#: +5.0 is a six-fold vol spike — both already beyond any observed regime.
IV_SHOCK_MIN = -0.9
IV_SHOCK_MAX = 5.0

#: Calendar days of decay. Longer than every tenor the platform trades.
DAYS_FORWARD_MIN = 0
DAYS_FORWARD_MAX = 365

#: Fallback name when the user supplies none — the scenario's own parameters,
#: so two unnamed runs are still distinguishable in the persisted history.
MAX_NAME_LENGTH = 64

#: §26: the most per-ticker overrides one user scenario may carry. Guards
#: the request body, not the model — a book has single-digit names, and an
#: unbounded map is an unbounded parse.
MAX_SPOT_SHOCK_TICKERS = 64

#: §26: longest accepted ticker key in ``spot_shock_by_ticker``.
MAX_TICKER_LENGTH = 12


# --- Validation request bounds (design §9.4) — parameters, never magic ----

#: Smallest rolling estimation window a walk-forward run will accept. Below
#: this the "window" is not a sample: a 95 % tail over 30 observations is one
#: and a half observations, and its VaR is the second-worst day, full stop.
VALIDATION_WINDOW_MIN = 30

#: Largest window. The platform stores ~600 daily bars per underlying, so a
#: window past this leaves fewer forecast days than ``MIN_FORECASTS`` and
#: every row would come back UNAVAILABLE — a 422 naming the field is more
#: useful than a run that cannot say anything.
VALIDATION_WINDOW_MAX = 500


class ValidationRunRequest(BaseModel):
    """Optional overrides for one validation run (design §9.4).

    The whole body is optional — ``POST`` with no body at all runs the
    default 250-observation window. Only the window is exposed: the
    traffic-light cut-offs and the GARCH refit stride are POLICY, and a
    caller who could move them per request could tune a GREEN into existence,
    which is exactly the hindsight calibration spec §43 forbids.
    """

    window: int | None = Field(
        None,
        ge=VALIDATION_WINDOW_MIN,
        le=VALIDATION_WINDOW_MAX,
        description=(
            "Rolling estimation window in OBSERVATIONS. Each forecast uses "
            "exactly this many days strictly before the day it forecasts. "
            f"Defaults to {VALIDATION_DEFAULT_WINDOW}."
        ),
    )


class StressRunRequest(BaseModel):
    """A user-defined hypothetical scenario (spec §26; design §8.5).

    Out-of-range values are 422s, never clamped: silently running a different
    scenario than the one asked for would be the dishonest option.
    """

    equity_shock: float = Field(
        ...,
        ge=EQUITY_SHOCK_MIN,
        le=EQUITY_SHOCK_MAX,
        description=(
            "FRACTIONAL move applied UNIFORMLY to every underlying "
            "(-0.10 = every name down 10 %). This is a beta = 1 assumption "
            "and the result row says so (`uniform_beta_1`)."
        ),
    )
    iv_shock: float = Field(
        0.0,
        ge=IV_SHOCK_MIN,
        le=IV_SHOCK_MAX,
        description=(
            "RELATIVE multiplicative shock on the IV LEVEL (+0.20 => "
            "iv1 = iv0 * 1.20), NOT vol points."
        ),
    )
    days_forward: int = Field(
        0,
        ge=DAYS_FORWARD_MIN,
        le=DAYS_FORWARD_MAX,
        description="Calendar days of time decay applied to every option leg.",
    )
    spot_shock_by_ticker: dict[str, float] | None = Field(
        None,
        description=(
            "PER-TICKER fractional spot shocks that OVERRIDE `equity_shock` "
            "for the tickers they name. A ticker absent from this map keeps "
            "the uniform `equity_shock`, so 'SPY -5% / QQQ -8% / everything "
            "else -3%' is `equity_shock=-0.03` with "
            "`{\"SPY\": -0.05, \"QQQ\": -0.08}`. Keys are uppercased; "
            f"values must lie in [{EQUITY_SHOCK_MIN}, {EQUITY_SHOCK_MAX}] "
            "like the uniform shock. Out of range is a 422, never a clamp."
        ),
    )
    name: str | None = Field(
        None,
        max_length=MAX_NAME_LENGTH,
        description="Optional label; a parameter-derived name is used when absent.",
    )

    @field_validator("spot_shock_by_ticker")
    @classmethod
    def _clean_shocks(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate and normalise the §26 per-ticker overrides.

        OVERRIDE, NOT EXCLUSIVE (the choice this endpoint makes, documented
        here and in the field description): the map is applied ON TOP of
        `equity_shock`, exactly as ``stress.Scenario`` already defines it —
        a ticker present takes its own shock, a ticker absent keeps the
        uniform one. The alternative, rejecting a request that carries
        both, would make the common case ("these two names move
        differently, the rest move together") inexpressible in one
        scenario, which is the case §26's own "SPY -5% / QQQ -8%" example
        is about.

        Rejects, rather than repairs (spec §26: out-of-range is a 422,
        never a clamp): a blank or over-long ticker, a non-finite or
        out-of-range value, more than ``MAX_SPOT_SHOCK_TICKERS`` entries,
        and two keys that collide once uppercased (``spy`` and ``SPY``
        would silently drop one shock).
        """
        if v is None:
            return None
        if len(v) > MAX_SPOT_SHOCK_TICKERS:
            raise ValueError(
                f"spot_shock_by_ticker carries {len(v)} tickers; "
                f"at most {MAX_SPOT_SHOCK_TICKERS} are accepted"
            )
        out: dict[str, float] = {}
        for raw_ticker, value in v.items():
            ticker = str(raw_ticker).strip().upper()
            if not ticker or len(ticker) > MAX_TICKER_LENGTH:
                raise ValueError(
                    f"spot_shock_by_ticker key {raw_ticker!r} is not a ticker "
                    f"(1-{MAX_TICKER_LENGTH} characters after trimming)"
                )
            if ticker in out:
                raise ValueError(
                    f"spot_shock_by_ticker names {ticker!r} twice (keys are "
                    "compared uppercased); one of the two shocks would be lost"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"spot_shock_by_ticker[{ticker!r}] must be a number, got {value!r}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"spot_shock_by_ticker[{ticker!r}] must be finite, got {value}"
                )
            if not (EQUITY_SHOCK_MIN <= value <= EQUITY_SHOCK_MAX):
                raise ValueError(
                    f"spot_shock_by_ticker[{ticker!r}]={value} is outside "
                    f"[{EQUITY_SHOCK_MIN}, {EQUITY_SHOCK_MAX}]"
                )
            out[ticker] = value
        return out

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


def _scenario_name(req: StressRunRequest) -> str:
    """The row's name: the user's, or one built from the parameters.

    A parameter-derived name keeps the persisted history legible — two
    unnamed runs with different shocks never collide into one label.
    """
    if req.name:
        return req.name[:MAX_NAME_LENGTH]
    # §26: a per-ticker run must not share a label with the uniform run of
    # the same headline numbers — the persisted history would then show two
    # different scenarios under one name.
    overrides = req.spot_shock_by_ticker or {}
    suffix = f" / {len(overrides)} per-ticker" if overrides else ""
    return (
        f"User: equity {req.equity_shock:+.1%} / IV {req.iv_shock:+.0%}"
        f" / +{req.days_forward}d{suffix}"
    )[:MAX_NAME_LENGTH]


async def _account_cash(session: AsyncSession) -> float | None:
    """The account's cash, resolved the way every other read view resolves it.

    THE ACCOUNT IS THE BROKER'S — the platform stores no copy. With a real
    broker the live number is fetched; only the dev/test simulator keeps a
    local ledger row. No venue ⇒ ``None``, which makes NAV unknown and every
    percent-of-NAV field an honest null (the USD P&L is still real).
    """
    if not broker_configured():
        return None
    if simulated_broker_mode():
        portfolio = await get_or_create_portfolio(session)
        return portfolio.cash
    try:
        broker = resolve_broker()
        account = await asyncio.to_thread(broker.get_account)
    except Exception as exc:  # noqa: BLE001 — a read view never 5xxs on this
        logger.warning("stress run: broker account unreadable: %s", exc)
        return None
    return account.cash


@router.post("/stress/run")
async def run_user_stress(
    req: StressRunRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Run ONE user-defined scenario over the CURRENT book (spec §26, §51).

    Returns the scenario result in the same shape the risk view's
    ``statistical.stress.rows`` entries use, plus the book context that
    produced it (``nav``, leg counts) and the persisted ``run_id``.

    Persists one ``stress_runs`` row (``kind=USER``, ``snapshot_id`` NULL,
    ``validated=False``) and writes **no audit event** — see the module
    docstring for why both are true at once.

    Never 503: an unconfigured venue or a book with no priceable leg is an
    honest result (P&L 0.0 over zero legs, NAV null) with the gap named in
    ``positions_excluded``, not an error. A genuinely malformed leg is caught
    by the library and reported as an UNAVAILABLE/FAILED row with its reason.
    """
    from .portfolio import (  # local import: the routers import each other
        open_positions_with_prices,
        portfolio_greeks_read,
        position_market_value,
    )

    now = datetime.now(timezone.utc)
    cash = await _account_cash(session)
    pairs = await open_positions_with_prices(session)
    have_market_data = market_data_configured()
    values = [
        position_market_value(pos, price, market_data=have_market_data)
        for pos, price in pairs
    ]
    nav: float | None = None
    if cash is not None:
        nav = cash + sum(v for v in values if v is not None)
        if nav <= 0:
            # A non-positive NAV cannot denominate a percentage; the USD
            # number stays real and pct_nav becomes an honest null.
            nav = None

    _greeks, greeks_rows = portfolio_greeks_read(pairs)
    stock_legs, option_legs, excluded = stress_legs_from_book(pairs, greeks_rows)

    # §26: per-ticker shocks OVERRIDE the uniform one for the tickers they
    # name; every other underlying keeps `equity_shock`. That is exactly
    # `Scenario`'s own documented rule, so this endpoint adds a surface, not
    # a second semantics.
    overrides = dict(req.spot_shock_by_ticker or {})
    scenario = Scenario(
        name=_scenario_name(req),
        kind=KIND_USER,
        spot_shock=req.equity_shock,
        spot_shock_by_ticker=overrides,
        iv_shock=req.iv_shock,
        days_forward=float(req.days_forward),
        validated=False,          # a user hypothesis is never a validated one
        source="USER",
        iv_shock_source=IV_SHOCK_SOURCE_SPECIFIED,
        notes=(
            "user-defined hypothetical (spec §26); the equity shock is "
            "uniform across every underlying (beta = 1)"
            if not overrides
            else (
                "user-defined hypothetical (spec §26); "
                f"{len(overrides)} per-ticker shock(s) OVERRIDE the uniform "
                f"{req.equity_shock:+.1%}, which still applies to every "
                "underlying they do not name (so beta = 1 holds only outside "
                f"{sorted(overrides)})"
            )
        ),
    )
    result = run_scenario(stock_legs, option_legs, scenario, nav=nav)

    # Spec §56: the history is kept even though no audit event is written.
    row = StressRunRow(
        snapshot_id=None,   # a user hypothesis is not a snapshot build
        scenario=result.name,
        kind=result.kind,
        validated=result.validated,
        pnl_usd=result.pnl_usd,
        pnl_pct_nav=result.pnl_pct_nav,
        method_full_reval=int(result.method_coverage.get(REVAL_METHOD_FULL_REVAL, 0)),
        method_delta_linear=int(
            result.method_coverage.get(REVAL_METHOD_DELTA_LINEAR, 0)
        ),
        health=str(result.health),
        reason=result.reason,
        params=_jsonable(result.params),
        per_position=_jsonable(dict(result.per_key)),
        as_of=now,
    )
    session.add(row)
    await session.commit()

    return {
        "mode": "SHADOW",
        "as_of": now.isoformat(),
        "run_id": row.id,
        "nav": nav,
        "n_stock_legs": len(stock_legs),
        "n_option_legs": len(option_legs),
        "positions_excluded": excluded,
        "scenario": _scenario_row_api(result),
        "per_position": dict(result.per_key),
        "note": (
            "SHADOW: a read of the current book under a hypothesis. No order, "
            "no decision and no audit event — the run is persisted to "
            "stress_runs for history (spec §56)."
        ),
    }


@router.post("/validation/run")
async def run_model_validation(
    req: ValidationRunRequest | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Walk-forward backtest the VaR/ES model grid (spec §42, §43; §9.4).

    WHAT IT IS. A READ of the book's own P&L HISTORY, asking one question per
    model: when this estimator said "the 95 % one-day VaR is $X", how often
    did the book actually lose more than $X? Every forecast is produced on the
    ``window`` observations STRICTLY BEFORE the day it scores (spec §43 —
    walk-forward only, no hindsight), and no threshold is tuned to the result.

    WHAT IT IS NOT. It places nothing, changes nothing and decides nothing.
    Its verdicts feed exactly two SHADOW surfaces: the risk view's
    ``statistical.validation`` block and the ``backtest_red_triggers``
    parameter of the model-risk display. The GARCH view is RESEARCH — below
    SHADOW — and its §63 promotion remains a user action.

    Like the stress endpoint, and for the same two reasons:

    - **No audit event.** A measurement of past forecasts is not a decision
      (house rule: read views write no audit events).
    - **It still persists.** One ``risk_model_backtests`` row per view, with
      ``snapshot_id`` NULL (no snapshot build stands behind an on-demand run),
      so the calibration history survives (spec §56). UNAVAILABLE rows are
      persisted too, with their reason — a missing row would later read as
      "never run".

    Never 503. No positions, no stored bars or too short a history are all
    honest results: the rows come back UNAVAILABLE with the real numbers in
    their ``reason``, and ``n_obs`` says how much history there was.

    ``window`` is the only tunable (422 outside
    ``[VALIDATION_WINDOW_MIN, VALIDATION_WINDOW_MAX]``, never clamped). The
    body itself is optional.
    """
    window = VALIDATION_DEFAULT_WINDOW
    if req is not None and req.window is not None:
        window = req.window

    now = datetime.now(timezone.utc)
    cash = await _account_cash(session)

    # The book P&L series comes from the SAME builder the risk view uses, so
    # a verdict can never describe a different book than the numbers it
    # judges. persist=False: this endpoint validates models, it does not add
    # a snapshot to the drawdown NAV series.
    build = await build_risk_snapshot(
        session,
        trigger=TRIGGER_ON_DEMAND,
        cash=cash,
        trading_enabled=None,
        persist=False,
    )
    book_pnl = list(build.book.total) if build.book is not None else []
    book_dates = (
        list(build.book.dates) if build.book is not None and build.book.dates else None
    )

    run = await run_model_backtests(
        session,
        book_pnl=book_pnl,
        dates=book_dates,
        nav=build.snapshot.nav,
        snapshot_id=None,  # an on-demand run is not a snapshot build
        window=window,
        as_of=now,
        trigger=VALIDATION_TRIGGER_ON_DEMAND,
    )
    await session.commit()

    payload = run.api()
    payload.update(
        {
            "window": window,
            "min_forecasts": VALIDATION_MIN_FORECASTS,
            "nav": build.snapshot.nav,
            "seconds": round(run.seconds, 4),
            "note": (
                "SHADOW/RESEARCH: a walk-forward backtest of past forecasts. "
                "No order, no decision and no audit event — the rows are "
                "persisted to risk_model_backtests for history (spec §56). "
                "The GARCH view is RESEARCH; its promotion is a user action "
                "(§63)."
            ),
        }
    )
    return payload
