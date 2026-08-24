# Data Source Architecture

**The single source of truth for provider responsibilities**
(per `prompts/data_source.md`; this document is the §42 deliverable).

Last updated: 2026-08-13. Subscriptions this architecture assumes:

| Subscription | Cost | Role |
|---|---|---|
| Alpaca Algo Trader Plus | $99/mo | ALL market data (stocks, options, news) + trading/brokerage |
| Massive Financials & Ratios | $29/mo | Company fundamentals ONLY (integration pending — see Gaps) |
| **Total** | **$128/mo** | Do not add another paid provider without explicit approval. |

## Non-negotiable principle

For every meaningful number the system must answer *"where did this come
from?"* with exactly one of:

- **ALPACA RAW MARKET DATA** (`source: alpaca` / `price_basis`)
- **MASSIVE RAW FUNDAMENTAL DATA** (future; `source: massive`)
- **INTERNAL DETERMINISTIC CALCULATION** (indicator/score engines; §44 reconciliation)
- **LLM-GENERATED INTERPRETATION** (`llm_model` recorded at generation)

Never blur these categories.

## Provider responsibilities

```
ALPACA (authoritative for market data + execution)
├── Stocks: daily bars (split-adjusted), real-time snapshots (NBBO quote,
│   latest trade, prev close)
├── Options: chain snapshots (NBBO bid/ask, latest trade, day bar,
│   greeks δ/γ/θ/ν/ρ, provider IV) + contract reference (OI, close price)
├── News: real articles (id, headline, source, timestamps, tickers, url)
└── Brokerage (pre-existing, unchanged): account, cash, positions, orders,
    fills — libs/broker/alpaca.py

MASSIVE (fundamentals only — integration pending)
└── Income statements, balance sheets, cash flow, ratios
    (endpoints of the Financials & Ratios plan; NOT market data)

INTERNAL (deterministic, tested, §44-reconciled)
└── SMA/EMA/RSI/MACD/ATR/realized vol, bull/bear scores, edge
    classification, tradeability, liquidity/selector scores, risk sizing,
    mid/spread arithmetic, growth & fundamental scores (future)

LLM (interpretation only; §12/§46 authority boundary)
└── grounded recommendations, catalyst context, plan narratives
```

## Implementation map

| Concern | Where |
|---|---|
| Provider protocol | `libs/market_data/provider.py` (`MarketDataProvider`) |
| Alpaca market data | `libs/market_data/alpaca.py` (`AlpacaMarketDataProvider`) |
| Massive market data (legacy) | `libs/market_data/massive.py` — still registered; superseded for market data |
| Stub (dev/tests only) | `libs/market_data/stub.py` — synthetic, opt-in only |
| Registry / selection | `libs/market_data/__init__.py`; `Settings.market_data_provider` ∈ {"", alpaca, massive, stub} |
| Runtime switch | `PUT /api/config/providers` (`market_data_provider`); DB rows override .env |
| Broker (trading) | `libs/broker/alpaca.py` (paper host structural guard) |

Alpaca market data authenticates with the SAME account keys as the broker
(`alpaca_api_key_id` / `alpaca_api_secret_key`) against
`data.alpaca.markets`; the option-contracts reference endpoint lives on the
trading host (`paper-api.alpaca.markets`, reference data identical across
paper/live). Keys are write-only via the config API and never logged.

## Data source matrix

| Data | Provider / endpoint | Raw or calculated | Cache / freshness |
|---|---|---|---|
| Stock daily bars | Alpaca `GET /v2/stocks/{s}/bars` (1Day, split-adj) | Raw | stored in `stock_bars_daily`; append-only refresh per trading day |
| Stock real-time quote | Alpaca `GET /v2/stocks/snapshots?symbols=` | Raw | no cache (overview refetch ~15s) |
| Option chain (quotes+greeks+IV) | Alpaca `GET /v1beta1/options/snapshots/{u}` (OPRA) + OI merged from `GET /v2/options/contracts` | Raw (greeks/IV = provider values, `source: alpaca`) | rebuilt per read; current-day only |
| Option contract reference / OI | Alpaca Trading `GET /v2/options/contracts` | Raw | per chain read; EOD view day-cached |
| Option EOD prev bar | Alpaca `GET /v1beta1/options/snapshots?symbols=` (prevDailyBar) | Raw | EOD view day-cached |
| News | Alpaca `GET /v1beta1/news` | Raw (verbatim; `source_id` prefixed `alpaca:`) | ingested+deduped at Recommendations refresh |
| SMA/EMA/RSI/MACD/ATR/RV | internal `libs/trading_core/features` over Alpaca bars | **Internal** | recomputed per read from stored bars |
| Bull/Bear/edge/classification | internal signals engines (versioned weights) | **Internal** | per read; §44 exact reconciliation |
| mid / spread / spread_pct | internal from Alpaca NBBO | **Internal** | per read |
| Liquidity / selector score | internal `libs/trading_core/contracts` | **Internal** | per read |
| VWAP (bar) | Alpaca bar `vw` (documented canonical) | Raw | with bars |
| Account / cash / positions / orders | Alpaca Brokerage (live reads; platform stores no cash copy) | Raw | live per read |
| Income/balance/cash-flow/ratios | Massive Financials & Ratios | Raw | **pending integration**; cache 12–24h when built |
| Growth metrics / Fundamental Score | internal over Massive statements | **Internal** | pending |
| Recommendation / catalyst text | LLM (grounded on stored news) | **LLM** | stored rows, model-stamped |

## Fallback policy (§33)

**No silent cross-provider substitution, ever.** If Alpaca market data
fails: data status degrades (503 / DEGRADED capability report) and
decisions requiring fresh market data are prevented. Massive is never
silently used for prices, and a future fundamentals outage must not block
price display — the system degrades BY CAPABILITY
(`GET /api/market/capabilities` probes live entitlements; keys:
`stock_history`, `stock_realtime`, `option_chain`, `option_contracts`,
`news`).

Known honest absences on Alpaca:

- **Indices (VIX/SPX)**: Alpaca has no index feed. VIX is omitted from the
  market overview (skip-with-warning, `DataCapability.UNAVAILABLE`
  posture) — never proxied.
- Rows without greeks/IV (deep wings) are skipped in the chain, never
  zero-filled; rows without a live NBBO price from the session close with
  `price_basis: "day_close"` and a worst-case spread (can only REJECT in
  the §9 selector).

## Point-in-time rules (§30/§31)

- Bars are append-only with `DATA_BACKFILL` audit events recording the
  provider per batch (a provider switch is visible in the audit trail, and
  stored history is never rewritten).
- Plans record `market_data_as_of` + configuration versions (§41); stale
  plans require revalidation (§42).
- Future fundamentals MUST use `filing_date`/`available_date` for backtest
  visibility — never the fiscal period end (look-ahead bias).

## Known provider limitations & gaps (§43F)

1. **Historical options depth**: Alpaca option data begins ~Feb 2024 —
   insufficient for long-horizon option backtests. The consumer-facing
   seam is the provider protocol (`get_option_chain`/`get_option_contracts`
   etc.); backtesting must keep depending on the interface so ThetaData/
   CBOE/ORATS/Massive-Options can slot in later WITHOUT rewriting the
   engine. Do not couple backtests to `AlpacaMarketDataProvider`.
2. **Massive fundamentals integration** (income/balance/cash-flow/ratios +
   Fundamental Score engine) is designed here but NOT yet implemented.
3. VIX unavailable (above). 4. Analyst estimates, insider/institutional
   data, options flow: `UNAVAILABLE` — future providers, not forced into
   current subscriptions.

## Historical options data (verified live 2026-08-17)

Probed against the real Alpaca APIs with the stored Algo Trader Plus
credentials (never assumed from docs). Findings:

| Capability | Verdict | Evidence |
|---|---|---|
| Expired contracts enumerable | ✅ | `/v2/options/contracts?status=inactive` returns 2024-era AAPL contracts (with expiration filters; adjusted contracts appear with a leading digit, e.g. `1AAPL…`, and are NOT valid data-API symbols) |
| Daily bars for expired contracts | ✅ full contract life | `AAPL241220C00250000`: 230 daily bars, 2024-01-18 → expiry 2024-12-20, real volume |
| Earliest availability | ~Feb 2024 (some Jan 2024) | Feb-2024-era contract's first bar 2024-02-12; a contract expiring 2024-01-05 has 0 bars |
| Historical trades (tick) | ✅ | `/v1beta1/options/trades` returns real prints with price/size/exchange |
| Historical NBBO quotes | ❌ endpoint does not exist | `/v1beta1/options/quotes` → 404 route (quotes exist only in live snapshots/stream) |
| Historical greeks / IV | ❌ | snapshot-only; historical IV must be COMPUTED from real prices (that is derivation, not fabrication) |
| Historical open interest | ❌ | contracts endpoint carries only current OI |

**Consequence: the options backtest is DATA-UNBLOCKED** for ~Feb 2024
onward under the fabrication ban — entries/exits can replay against REAL
contract bars. Honest limitations to design around: spreads must stay a
fill-model bps proxy (no quote history); selector filters that need OI
have no historical values; IV/greeks are computed, and the window (~2.5y)
contains no full bear market.
