# Systematic Options Trading Platform — Development Plan

**Document purpose:** engineering + product specification for Claude implementation  
**Target architecture:** two repositories, Dockerized microservices, front end/back end decoupled  
**Primary use case:** watchlist-driven systematic stock/options research, backtesting, paper/live execution, portfolio-level risk management  
**Design principle:** LLM may recommend candidates and summarize information, but **cannot autonomously add symbols to the Watchlist or Trading Pool and cannot directly decide trades**.

---

## 1. Executive Review

The proposed system is strong conceptually because it separates four things that many retail trading systems incorrectly mix together:

1. **Discovery** — finding potentially interesting symbols.
2. **Research** — collecting data and backtesting a user-approved Watchlist.
3. **Authorization** — only user-approved Trading Pool symbols may become executable.
4. **Execution & Risk** — trades require a mechanical signal plus portfolio-level risk approval.

The recommended final workflow is:

```text
News / Market / LLM Discovery
            │
            ▼
   LLM Recommendation Pool
            │
     USER REVIEW REQUIRED
            │
            ▼
         Watchlist
            │
      Historical Data
      Backtesting
      Live Analytics
            │
     USER REVIEW REQUIRED
            │
            ▼
        Trading Pool
            │
        Signal Engine
            │
       Strategy Engine
            │
      Contract Selector
            │
    Portfolio Allocator
            │
        Risk Engine
            │
      APPROVE / REJECT
            │
        Execution
            │
      Position Monitor
            │
         Exit Engine
```

### Core risk-review conclusion

The platform **must never equate model confidence with permission to go all-in**.

Even a 99% model score cannot eliminate:
- gap risk;
- model error;
- stale/bad market data;
- regime shifts;
- news shocks;
- volatility surface dislocations;
- execution slippage;
- correlation concentration.

Confidence may increase risk budget within limits, but **single-name caps, portfolio heat, cash floor, correlation limits, and kill switches remain hard constraints**.

---

# 2. Product Objectives

## 2.1 Primary Objectives

Build a system that can:

- let users maintain a manually approved Watchlist;
- download/store historical stock and option data only for Watchlist symbols;
- backtest mechanical equity/options strategies on Watchlist symbols;
- monitor Watchlist symbols in real time;
- provide LLM-generated candidate recommendations without automatically changing Watchlist;
- allow users to promote approved Watchlist symbols into a Trading Pool;
- execute trades only for Trading Pool symbols;
- automatically select equity/options instruments based on market regime, signal strength, IV regime, liquidity, and account permissions;
- allocate portfolio capital/risk across one or many simultaneous opportunities;
- maintain a dynamic cash reserve;
- enforce portfolio-level risk controls before any order;
- monitor open positions and mechanically generate exits;
- maintain a complete audit trail explaining every recommendation, approval, order, rejection, and exit.

## 2.2 Non-Objectives for V1

Do **not** optimize V1 around:

- HFT;
- sub-millisecond execution;
- every U.S. listed stock;
- fully autonomous LLM trading;
- naked option selling;
- short stock;
- complex exotic options;
- reinforcement learning;
- automatic model retraining in production;
- maximizing trade frequency.

The V1 objective is **robust, explainable, reproducible systematic trading**.

---

# 3. Investment / Quant Philosophy

The system should not attempt to solve discretionary fundamental valuation first.

It should instead answer:

1. What market regime are we in?
2. Is the symbol mechanically bullish, bearish, neutral, or transitional?
3. How strong is the directional edge?
4. Is implied volatility cheap, normal, expensive, or extreme?
5. What instrument best expresses that edge?
6. How much portfolio risk should be allocated?
7. What objective condition invalidates the trade?
8. When should profit be taken?
9. Does the current portfolio permit another correlated position?

The system should treat:

```text
Direction
× Magnitude
× Time
× Volatility
× Liquidity
× Portfolio Context
```

as the core decision vector.

---

# 4. Pool Model

## 4.1 LLM Recommendation Pool

Purpose:

- discover potentially interesting symbols;
- summarize catalysts;
- explain why a symbol may deserve user attention.

Rules:

- LLM recommendations **never** automatically enter Watchlist;
- no historical option download is triggered solely by recommendation;
- no trading signal is executable from this pool;
- every recommendation must show source data, timestamp, rationale, confidence, and expected horizon.

Suggested LLM output schema:

```json
{
  "ticker": "NVDA",
  "sentiment": 0.76,
  "impact": 0.82,
  "novelty": 0.71,
  "source_reliability": 0.95,
  "horizon": "1-5d",
  "catalyst_type": "earnings_guidance",
  "reason_codes": [
    "guidance_raise",
    "sector_positive"
  ],
  "summary": "..."
}
```

LLM output is an **information feature**, not an order signal.

---

## 4.2 Watchlist

Only user actions may add/remove Watchlist symbols.

For Watchlist symbols, the platform may:

- download historical stock data;
- download historical option data;
- maintain live stock market data;
- maintain current option-chain data;
- calculate features;
- run strategy backtests;
- display research;
- calculate opportunity scores.

Watchlist **cannot trade**.

---

## 4.3 Trading Pool

Only user actions may promote a Watchlist symbol to Trading Pool.

Recommended promotion checks:

- minimum historical data exists;
- backtest completed;
- out-of-sample statistics available;
- liquidity requirements met;
- account supports required instrument;
- user acknowledges risk characteristics.

Trading Pool means:

> "This symbol is authorized to trade if the mechanical strategy and risk engine approve it."

It does **not** mean automatic immediate purchase.

---

# 5. Account Constraints

The account permission model must be explicit and configurable.

Current required assumptions:

```text
Short stock:             DISABLED
Naked short call:        DISABLED
Naked short put:         DISABLED
Long stock:              ENABLED
Long call:               ENABLED
Long put:                ENABLED
Sell-to-close:           ENABLED
Defined-risk spreads:    CONFIGURABLE
```

The system must never confuse:

- **Sell to Open** — creates a short option exposure;
- **Sell to Close** — closes a previously purchased option.

If defined-risk spreads are unavailable, the Strategy Engine must degrade gracefully to:

```text
Long Stock
Long Call
Long Put
No Trade
```

---

# 6. Strategy Architecture

## 6.1 Market Regime Engine

Recommended initial features:

- Close vs SMA20 / SMA50 / SMA200;
- SMA slopes;
- ADX(14);
- ATR(14);
- realized volatility;
- SPY / QQQ market direction;
- VIX or equivalent volatility regime;
- breadth if available.

Initial classifications:

```text
STRONG_BULL
MILD_BULL
NEUTRAL_RANGE
MILD_BEAR
STRONG_BEAR
TRANSITION
```

`TRANSITION` should default to **NO TRADE**.

---

## 6.2 Directional Signal Engine

Create mechanical Bull and Bear feature vectors.

Candidate Bull features:

- Price > SMA20 > SMA50 > SMA200;
- positive SMA slopes;
- MACD > signal and > 0;
- RSI continuation zone;
- price above VWAP;
- HH/HL market structure;
- breakout;
- bullish volume expansion;
- market/sector confirmation.

Candidate Bear features:

- Price < SMA20 < SMA50 < SMA200;
- negative SMA slopes;
- MACD < signal and < 0;
- RSI bearish continuation zone;
- price below VWAP;
- LH/LL structure;
- breakdown;
- bearish volume expansion;
- market/sector confirmation.

Initial implementation may use a weighted score, but all weights and thresholds must be parameterized.

Example:

```text
Directional Edge = Bull Score - Bear Score
```

Do not permanently hard-code values such as ADX=25 or RSI=60 as "truth". Treat them as **backtest parameters**.

---

## 6.3 Structure Detection

Market structure must be machine-computable.

Example pivot algorithm:

```text
pivot_window = 5

Pivot High:
high[t] > high[t-1...t-5]
AND
high[t] > high[t+1...t+5]

Pivot Low:
low[t] < low[t-1...t-5]
AND
low[t] < low[t+1...t+5]
```

Then:

```text
HH + HL => bullish structure
LH + LL => bearish structure
```

Avoid subjective chart interpretation.

---

# 7. Volatility Engine

Required metrics:

- ATM IV;
- IV Rank;
- IV Percentile;
- RV20 / RV30;
- IV-RV spread;
- term structure;
- put/call skew;
- expected move;
- ATR-relative implied move.

Suggested regimes:

```text
LOW
NORMAL
HIGH
EXTREME
```

Do not assume `IV > RV` automatically means options are overpriced.

---

# 8. Instrument Selection

The system should choose the instrument only **after** direction and volatility are classified.

Suggested V1 matrix:

| Direction | Strength | IV Regime | Default Instrument |
|---|---|---|---|
| Bull | Strong | Low | Long Call |
| Bull | Strong | Normal/High | Bull Call Spread if permitted |
| Bull | Moderate | Low/Normal | Bull Call Spread if permitted |
| Bull | Moderate | High | Stock / Bull Put Spread if permitted |
| Bull | Weak | Any | Stock / No Trade |
| Neutral | Low | Any | No Trade |
| Neutral | High | High | Iron Condor only if permitted |
| Bear | Weak | Any | No Trade |
| Bear | Moderate | Low/Normal | Bear Put Spread if permitted |
| Bear | Moderate | High | Higher-delta Long Put / No Trade |
| Bear | Strong | Low | Long Put |
| Bear | Strong | Normal/High | Bear Put Spread if permitted |

Because short stock is unavailable:

- bullish weak/moderate views may fall back to stock;
- bearish weak views should usually be **No Trade**;
- moderate bearish views may use longer-DTE, higher-|Delta| long puts when spreads are unavailable.

---

# 9. Option Contract Selector

Do not permanently select "0.55 Delta" as a fixed rule.

Generate candidate contracts and optimize.

## 9.1 Candidate Filters

Example Long Call/Put candidate universe:

```text
DTE:                30-90
Absolute Delta:     0.40-0.75
OI:                 configurable minimum
Option volume:      configurable minimum
Spread / Mid:       configurable maximum
Theta / Premium:    configurable maximum
```

## 9.2 Candidate Ranking

For each contract calculate:

- expected option return;
- expected downside;
- Delta;
- Gamma;
- Theta;
- Vega;
- IV;
- expected move;
- bid/ask cost;
- historical fill assumptions;
- probability-weighted payoff;
- liquidity score.

Select the highest **risk-adjusted expected value**, not the cheapest premium.

---

# 10. Entry Engine

A position should require all applicable gates:

```text
Trading Pool Authorization
        ↓
Data Quality Check
        ↓
Regime Check
        ↓
Directional Signal
        ↓
Volatility Check
        ↓
Instrument Check
        ↓
Liquidity Check
        ↓
Contract Selection
        ↓
Portfolio Allocation
        ↓
Risk Approval
        ↓
Order
```

Examples of entry vetoes:

- stale data;
- missing chain;
- spread too wide;
- insufficient OI;
- abnormal volatility spike;
- unsupported account strategy;
- earnings event conflicting with non-event strategy;
- excessive prior intraday move;
- correlation bucket full;
- cash floor violation;
- portfolio heat violation.

---

# 11. Exit Engine

The default goal is **Buy to Open → Sell to Close**, not exercise.

Exercise should not be part of normal V1 trading logic.

The Exit Engine must evaluate multiple independent exit families.

## 11.1 Alpha / Signal Exit

Exit when original edge materially decays.

Example:

```text
Entry directional score: 82
Current directional score: 44

Signal decay > configured threshold
=> reduce or exit
```

The exit threshold should be easier to trigger than the entry threshold.

---

## 11.2 Underlying Invalidation

Long Call example:

```text
Close < VWAP
AND
break latest pivot low
AND
bullish score deterioration
=> sell to close
```

Long Put is mirrored.

---

## 11.3 Premium Hard Stop

Example research parameters:

```text
warning: -30% to -35%
hard exit: -40% to -45%
```

These values must be backtested.

---

## 11.4 Profit Taking

For multiple contracts:

```text
+40-50% option P&L => reduce
+75-100%            => reduce again
remainder           => trail
```

For one contract:

- use a single mechanical profit target or trailing logic.

---

## 11.5 ATR Trailing Exit

Prefer using underlying price rather than noisy option premium.

Long Call:

```text
trail = highest_underlying_since_entry - k * ATR
```

Long Put:

```text
trail = lowest_underlying_since_entry + k * ATR
```

`k` must be backtested.

---

## 11.6 Time Stop

If the expected move does not occur within the statistical horizon:

```text
holding_days >= N
AND
underlying_move < X * ATR
=> close
```

This prevents Theta from silently turning a correct directional idea into a losing option trade.

---

## 11.7 DTE Exit

Never allow swing positions to drift unintentionally into high-gamma expiry behavior.

Example:

```text
if DTE <= 21:
    close_or_roll_policy()
```

Rolling should be a later-phase feature.

---

## 11.8 Remaining Expected Value Exit

Eventually upgrade exits from fixed profit targets to:

```text
Expected Remaining Upside
vs
Expected Remaining Downside
```

If expected incremental reward is no longer attractive:

```text
EXIT
```

This should become the advanced exit model after V1.

---

# 12. Portfolio Allocation Engine

This is a mandatory independent service/module.

Its job is to decide:

> given 0, 1, or many valid trades, how much risk should each receive?

Never allocate simply by equal capital.

---

## 12.1 Risk-Based Position Sizing

Base formula:

```text
Allowed Trade Risk = NAV × Risk Budget %
```

Then:

```text
Contracts = floor(
    Allowed Trade Risk
    /
    Max Loss per Contract
)
```

For stock:

```text
Shares = floor(
    Allowed Trade Risk
    /
    Stop Distance
)
```

---

## 12.2 Signal-Based Risk Budget

Example initial research schedule:

```text
Weak valid signal      0.50% NAV
Moderate               0.75% NAV
Strong                 1.00% NAV
Very strong            1.25% NAV
Absolute max           1.50% NAV
```

No confidence score may override the absolute maximum.

---

## 12.3 Single-Name Limits

Examples:

```text
Single-name max strategy risk:      1.5% NAV
Single-name stock capital exposure: 20-25% NAV
Long-option premium-at-risk cap:    configurable
```

These are hard risk limits, not recommendations.

---

## 12.4 Correlation Buckets

Calculate rolling correlations and/or predefined factor buckets.

Example:

```text
NVDA
AMD
AVGO
QQQ
SMH
```

may be treated as overlapping technology/Nasdaq exposure.

Possible rule:

```text
if rolling_corr > 0.70:
    shared risk bucket
```

Set:

```text
bucket max risk <= 3% NAV
```

Thresholds require validation.

---

## 12.5 Portfolio Heat

Definition:

```text
Portfolio Heat =
sum(max_loss_active_positions)
/
NAV
```

Suggested states:

```text
0-4%     Normal
4-6%     Elevated
6-8%     High
>8%      Reject new risk
```

Do not blindly implement these exact numbers without testing and risk-profile configuration.

---

# 13. Dynamic Cash Reserve

Cash is an intentional portfolio allocation.

Do not force 100% deployment.

Suggested framework:

| Regime | Example Minimum Cash Floor |
|---|---:|
| Strong Bull | 10-20% |
| Mild Bull | 20-30% |
| Neutral | 35-50% |
| Mild Bear | 40-60% |
| Strong Bear | 50-70% unless valid downside setups exist |

Cash floor may also increase when:

- volatility percentile becomes extreme;
- drawdown increases;
- model health deteriorates;
- cross-asset stress increases;
- strategy expectancy weakens.

Example:

```text
base_cash_floor
+ volatility_adjustment
+ drawdown_adjustment
+ model_health_adjustment
```

No strategy may violate the resulting cash floor.

---

# 14. Volatility Targeting

Advanced allocation layer:

```text
Exposure Multiplier =
Target Portfolio Vol
/
Forecast Portfolio Vol
```

Example:

```text
Target vol = 12%
Forecast vol = 18%

Multiplier = 0.67
```

Cap upward leverage:

```text
max_multiplier = 1.2
```

Do not allow volatility targeting to override hard risk caps.

---

# 15. Kelly Criterion

Optional research feature only.

If enough live/out-of-sample observations exist:

```text
Fractional Kelly = 0.25 × Full Kelly
```

or similar.

Final trade risk:

```text
min(
    Signal Risk Budget,
    Fractional Kelly Limit,
    Single Name Limit,
    Correlation Bucket Limit,
    Portfolio Heat Limit,
    Cash Constraint
)
```

Never use Full Kelly in V1.

---

# 16. Portfolio Greeks

Portfolio risk should include:

- Net Delta;
- Net Gamma;
- Net Theta;
- Net Vega;
- Delta-adjusted notional;
- sector/factor exposure;
- expiry concentration.

Example:

```text
Equivalent Shares =
contracts × 100 × delta
```

Then calculate delta-adjusted dollar exposure.

Risk Engine should veto:

- excessive long beta;
- excessive negative theta;
- excessive long/short vega;
- excessive expiry concentration;
- excessive event concentration.

---

# 17. Risk Engine

The Risk Engine must be architecturally independent from the Strategy Engine.

Strategy Engine may say:

```text
REQUEST:
Buy 5 NVDA calls
```

Risk Engine may respond:

```json
{
  "decision": "APPROVED_WITH_RESIZE",
  "requested_contracts": 5,
  "approved_contracts": 2,
  "reason_codes": [
    "TECH_BUCKET_LIMIT",
    "PORTFOLIO_VEGA_LIMIT"
  ]
}
```

Possible outputs:

```text
APPROVE
APPROVE_WITH_RESIZE
REJECT
PAUSE_STRATEGY
EMERGENCY_EXIT
```

---

# 18. Global Kill Switch

Mandatory.

Trigger examples:

- data feed stale;
- broker unavailable;
- option chain corrupted;
- current loss exceeds daily threshold;
- strategy rolling expectancy below threshold;
- drawdown exceeds hard threshold;
- model drift alert;
- unrealistically large price/IV jump;
- reconciliation mismatch;
- duplicate-order protection triggered.

UI must expose:

```text
TRADING ENABLED / PAUSED
```

with clear reason.

---

# 19. Strategy Health Monitor

Maintain rolling statistics:

```text
Win Rate
Profit Factor
Expected Value / R
Sharpe
Sortino
Max Drawdown
Current Drawdown
Slippage vs Backtest
Fill Rate
Signal Count
Model Drift
```

Example policy:

```text
if rolling_profit_factor < threshold
for N completed trades:
    PAUSE STRATEGY
```

Exact numbers are strategy parameters, not hard-coded assumptions.

---

# 20. Backtesting Architecture

Only Watchlist symbols require historical backtest data.

## 20.1 Research Backtest

Use:

- stock minute bars;
- option minute bars;
- point-in-time contract metadata;
- daily OI;
- reconstructed historical IV/Greeks if necessary.

Purpose:

- feature research;
- signal research;
- instrument selection;
- entry/exit testing;
- parameter robustness.

---

## 20.2 Execution Validation

After research strategy passes:

use historical:

- quotes;
- trades;
- bid/ask;
- slippage model;
- latency assumptions.

Never treat historical mid as guaranteed fill.

Test at least:

```text
Optimistic Fill
Mid-based Conservative Fill
Ask-to-buy / Bid-to-sell Worst Practical Case
```

---

## 20.3 Bias Controls

Mandatory controls:

### No look-ahead bias

If a 10:31 bar closes at 10:31:59, earliest trade is after bar completion.

### Point-in-time option universe

Use only contracts existing at the historical timestamp.

### Survivorship bias

Do not pretend today's winners were known historically.

### News timestamp integrity

LLM may only receive information published before the decision timestamp.

---

# 21. Backtest / Live Code Reuse

Mandatory architectural rule:

> **Signal, feature, strategy, risk, and exit logic must be shared between backtest and live execution.**

Do not build:

```text
research_strategy.py
```

and independently reimplement:

```text
live_strategy.py
```

Instead create shared domain packages.

Example:

```text
trading-core/
  features/
  signals/
  strategies/
  allocation/
  risk/
  exits/
```

Backtest and live services call the same logic.

---

# 22. Data Architecture

## 22.1 Massive

Use Massive for:

- live stock market data;
- live option market data;
- option chain snapshots;
- historical option bars;
- historical quotes/trades when needed;
- contract metadata.

Environment configuration example:

```env
MASSIVE_API_KEY=...
```

Never commit `.env`.

Provide:

```text
.env.example
```

with empty placeholders.

---

## 22.2 LLM

Provider should be abstracted.

Example:

```env
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=...
```

or:

```env
LLM_PROVIDER=bedrock
...
```

LLM service should expose an internal provider-neutral interface.

---

## 22.3 Database Choices

Recommended V1:

### PostgreSQL

Use for:

- users;
- Watchlist;
- Trading Pool;
- strategies;
- configuration;
- orders;
- positions;
- audit events;
- recommendations;
- backtest metadata.

### TimescaleDB extension or ClickHouse

Use for:

- OHLCV;
- features;
- option chain observations;
- backtest time series.

For V1 simplicity, PostgreSQL + TimescaleDB is sufficient.

### Redis

Use for:

- current state;
- hot market snapshots;
- locks;
- rate-limit state;
- pub/sub where appropriate.

### Object Storage

Use S3-compatible storage for:

- Massive flat files;
- historical datasets;
- backtest artifacts;
- model artifacts;
- reports.

Local development may use MinIO.

---

# 23. Event Bus

Recommended:

```text
Kafka / Redpanda
```

Topics might include:

```text
market.stock
market.option
market.option_chain
news.raw
news.enriched
features.updated
signal.generated
risk.request
risk.decision
order.request
order.update
position.update
exit.generated
audit.event
```

For local Docker development, Redpanda may be simpler while preserving Kafka-compatible APIs.

---

# 24. Backend Repository

Suggested repository:

```text
trading-platform-backend/
```

Suggested structure:

```text
trading-platform-backend/
├── docker-compose.yml
├── .env.example
├── README.md
├── Makefile
│
├── libs/
│   ├── trading_core/
│   │   ├── features/
│   │   ├── signals/
│   │   ├── strategies/
│   │   ├── contracts/
│   │   ├── allocation/
│   │   ├── risk/
│   │   ├── exits/
│   │   └── models/
│   └── common/
│       ├── config/
│       ├── logging/
│       ├── events/
│       └── telemetry/
│
├── services/
│   ├── api-gateway/
│   ├── auth-service/
│   ├── watchlist-service/
│   ├── trading-pool-service/
│   ├── market-data-service/
│   ├── historical-data-service/
│   ├── news-ingestor/
│   ├── llm-news-service/
│   ├── feature-engine/
│   ├── signal-engine/
│   ├── backtest-service/
│   ├── strategy-service/
│   ├── contract-selector/
│   ├── portfolio-service/
│   ├── allocation-service/
│   ├── risk-service/
│   ├── execution-service/
│   ├── position-monitor/
│   ├── exit-service/
│   ├── audit-service/
│   └── notification-service/
│
├── migrations/
├── tests/
└── scripts/
```

Do not deploy every directory as an independent service on day one.

Keep **microservice boundaries clear**, but V1 may combine lower-volume services to reduce operational complexity.

---

# 25. Frontend Repository

Suggested repository:

```text
trading-platform-ui/
```

Recommended:

```text
Next.js
TypeScript
React
TanStack Query
WebSocket client
Chart library
```

Suggested structure:

```text
trading-platform-ui/
├── app/
├── components/
│   ├── market/
│   ├── watchlist/
│   ├── recommendations/
│   ├── trading-pool/
│   ├── positions/
│   ├── options/
│   ├── risk/
│   ├── backtest/
│   └── shared/
├── lib/
│   ├── api/
│   ├── websocket/
│   ├── formatting/
│   └── types/
├── hooks/
├── stores/
├── public/
├── Dockerfile
├── .env.example
└── README.md
```

Front end must never directly access Massive, broker, or LLM credentials.

All privileged access goes through backend APIs.

---

# 26. Docker Architecture

Every backend component should have a Dockerfile.

Local `docker-compose.yml` should launch:

```text
frontend
api-gateway
core backend services
postgres/timescaledb
redis
redpanda/kafka
minio
```

Optional observability stack:

```text
Prometheus
Grafana
Loki
Tempo / Jaeger
```

Required health endpoints:

```text
/healthz
/readyz
```

All services should support graceful shutdown.

---

# 27. API Boundary

Front end calls backend only.

Suggested public APIs:

```text
GET    /api/market/overview
GET    /api/recommendations
POST   /api/watchlist
DELETE /api/watchlist/{ticker}
GET    /api/watchlist/{ticker}/analysis
POST   /api/trading-pool
DELETE /api/trading-pool/{ticker}

POST   /api/backtests
GET    /api/backtests/{id}

GET    /api/opportunities
GET    /api/positions
GET    /api/portfolio/risk

POST   /api/orders/preview
POST   /api/orders/approve
POST   /api/trading/pause
POST   /api/trading/resume
```

Use WebSockets/SSE for live updates.

---

# 28. UI / UX Information Architecture

Primary navigation:

```text
Dashboard
Recommendations
Watchlist
Trading Pool
Positions
Backtests
Risk
Activity
Settings
```

---

# 29. Dashboard UX

Dashboard should answer immediately:

1. What is the market regime?
2. What opportunities exist?
3. What do I currently own?
4. How much risk is active?
5. How much cash remains?
6. Is trading enabled?

Recommended layout:

```text
┌────────────────────────────────────────────────────────┐
│ Market Regime | Portfolio NAV | Cash | Heat | Status  │
├────────────────────────────────────────────────────────┤
│ Top Watchlist Opportunities                            │
├────────────────────────────┬───────────────────────────┤
│ Active Positions           │ Risk Exposure             │
│ P&L / Exit status          │ Delta / Vega / Buckets    │
├────────────────────────────┴───────────────────────────┤
│ Alerts / Recent Activity                               │
└────────────────────────────────────────────────────────┘
```

Never bury `Trading Paused` or `Risk Limit Reached`.

They must be visually obvious.

---

# 30. LLM Recommendations UX

Each recommendation card:

```text
Ticker
Company
Direction / sentiment
Catalyst
Impact score
Novelty
Time horizon
Source count
Timestamp
Short rationale
```

Actions:

```text
View Evidence
Dismiss
Add to Watchlist
```

There must be **no "Trade Now" action** directly from Recommendation Pool.

---

# 31. Watchlist UX

Suggested columns:

```text
Ticker
Price
Regime
Directional Score
Bull Score
Bear Score
IV Regime
Expected Move
News Score
Backtest Status
Opportunity Status
```

Status examples:

```text
NO SIGNAL
WATCH
SETUP FORMING
ENTRY READY
DATA ISSUE
BACKTEST FAILED
```

Actions:

```text
Analyze
Run Backtest
View Options
Add to Trading Pool
Remove
```

---

# 32. Trading Pool UX

Trading Pool should look more serious than Watchlist.

Each row should include:

```text
Ticker
Trading Enabled?
Allowed Strategies
Signal
Strategy Candidate
Risk Budget
Current Exposure
Last Decision
```

Actions:

```text
Enable/Disable Trading
View Strategy
View Risk
Remove from Trading Pool
```

A global:

```text
PAUSE ALL TRADING
```

control must exist.

---

# 33. Symbol Analysis Page

Tabs:

```text
Overview
Price
Technical
Options
News
Backtest
Trade Plan
Audit
```

Trade Plan should show:

```text
Signal
Regime
Strategy
Contract
Entry
Maximum Loss
Risk Budget
Position Size
Exit Rules
Why Trade
Why Not Trade
```

Do not display a single opaque "AI Confidence 94%" without showing contributing factors.

---

# 34. Option Chain UX

Required fields:

```text
Expiration
Strike
Call/Put
Bid
Ask
Mid
Spread %
Last
Volume
OI
IV
Delta
Gamma
Theta
Vega
```

Highlight contracts that pass Contract Selector filters.

Allow toggling:

```text
All
Eligible
Recommended Candidate
```

---

# 35. Backtest UX

Backtest configuration:

```text
Ticker
Date range
Strategy version
Timeframe
Fill model
Transaction cost model
Parameter set
```

Results:

```text
Total Return
CAGR
Sharpe
Sortino
Max Drawdown
Win Rate
Profit Factor
Expected Value
Average Trade
Average Hold
Slippage
Number of Trades
```

Must visibly separate:

```text
IN-SAMPLE
VALIDATION
OUT-OF-SAMPLE
```

Include equity curve and drawdown curve.

---

# 36. Risk Dashboard UX

This is a first-class page.

Display:

```text
NAV
Cash %
Portfolio Heat
Max New Risk Available
Net Delta
Net Gamma
Net Theta
Net Vega
Sector Buckets
Correlation Buckets
Expiration Concentration
Current Drawdown
Daily P&L
Kill Switch Status
```

Use clear warnings:

```text
NORMAL
ELEVATED
HIGH
BLOCKED
```

Risk explanations must be human readable.

Example:

> New NVDA position rejected because Technology Bucket would rise from 2.7% to 3.6%, above the configured 3.0% limit.

---

# 37. Position UX

Every position card/page should show:

```text
Entry timestamp
Entry underlying price
Entry option premium
Current premium
P&L
Original signal score
Current signal score
Signal decay
DTE
Greeks
Stop
Profit target
Trailing stop
Time stop
Exit status
```

A user should always know **why the system is still holding the position**.

---

# 38. Audit UX

Every action should be auditable.

Example:

```text
11:32:01 Signal confirmed
11:32:02 Strategy: Long Call
11:32:02 Requested: 3 contracts
11:32:03 Risk resized to 2
11:32:04 User/Auto execution approved
11:32:06 Order submitted
11:32:07 Filled
...
14:18:20 Signal decay threshold reached
14:18:21 Exit generated
14:18:23 Sell to Close filled
```

No black-box state transitions.

---

# 39. UX Safety Principles

- destructive actions require confirmation;
- clearly distinguish Watchlist from Trading Pool;
- clearly distinguish Recommendation from Signal;
- clearly distinguish Signal from Approved Trade;
- show stale-data indicators;
- show market-data timestamp everywhere relevant;
- never hide max loss;
- display cash and Portfolio Heat prominently;
- show why a trade was rejected;
- show whether an option action is Buy-to-Open or Sell-to-Close;
- avoid casino-style UI patterns.

---

# 40. Security

Secrets:

```env
MASSIVE_API_KEY=
LLM_API_KEY=
BROKER_API_KEY=
BROKER_API_SECRET=
DATABASE_URL=
```

Rules:

- `.env` in `.gitignore`;
- commit `.env.example`;
- no browser exposure;
- redact secrets from logs;
- encrypt production secrets;
- use AWS Secrets Manager / similar in deployment;
- implement credential rotation.

---

# 41. Observability

Every service should emit:

- structured JSON logs;
- correlation/request ID;
- metrics;
- traces where useful.

Critical metrics:

```text
market_data_lag_ms
option_chain_age_seconds
signal_generation_latency
risk_decision_latency
order_submission_latency
order_fill_latency
broker_rejections
duplicate_order_blocks
websocket_disconnects
```

---

# 42. Testing Strategy

## Unit Tests

Test:

- indicators;
- signal scores;
- IV/Greek calculations;
- position sizing;
- max loss;
- cash floor;
- correlation caps;
- exit rules.

## Property Tests

Examples:

```text
Position size must never exceed risk cap.
No rejected ticker may produce an order.
No non-Trading-Pool symbol may produce an order.
No naked short position may be generated.
```

## Integration Tests

Test:

```text
Massive → Feature → Signal → Risk → Execution Sandbox
```

## Replay Tests

Replay historical market sessions event-by-event through live code.

This is extremely valuable for production readiness.

---

# 43. Development Phases

## Phase 0 — Architecture & Contracts

Deliverables:

- two repos;
- Docker Compose;
- shared schemas;
- `.env.example`;
- API contracts;
- event schemas;
- database schema;
- CI;
- health endpoints;
- architecture documentation.

Acceptance:

```text
docker compose up
```

starts a working skeleton.

---

## Phase 1 — Watchlist & Market Data

Build:

- Watchlist API/UI;
- Massive adapter;
- stock live data;
- option chain retrieval;
- historical downloader;
- Timescale storage;
- market overview.

Acceptance:

- user adds ticker;
- system starts data lifecycle only after Watchlist addition;
- UI shows price + option chain.

---

## Phase 2 — Feature Engine

Implement:

```text
SMA
MACD
RSI
ATR
ADX
VWAP
Volume features
Pivot structure
RV
IV metrics
Expected move
```

Acceptance:

- deterministic calculation;
- unit-tested;
- same library works in historical/live mode.

---

## Phase 3 — Backtest V1

Build:

- historical replay;
- mechanical signals;
- Long Stock;
- Long Call;
- Long Put;
- Sell-to-Close exits;
- realistic transaction costs.

Do not add LLM yet.

Goal:

**prove whether basic systematic edge exists first.**

---

## Phase 4 — Portfolio / Allocation / Risk

Implement:

- NAV;
- cash;
- risk budgets;
- position sizing;
- Portfolio Heat;
- single-name limits;
- correlation buckets;
- dynamic cash floor;
- delta-adjusted exposure;
- risk veto;
- kill switch.

Acceptance:

Risk Engine can resize/reject trades independently of Strategy Engine.

---

## Phase 5 — Trading Pool

Build:

- Watchlist → Trading Pool user workflow;
- per-symbol trading toggle;
- allowed-strategy configuration;
- authorization enforcement;
- audit events.

Acceptance:

A Watchlist-only symbol can **never** reach execution.

---

## Phase 6 — Paper Execution

Integrate broker paper account.

Implement:

```text
Buy to Open
Sell to Close
Order states
Partial fills
Cancel/replace
Reconciliation
Duplicate prevention
```

Acceptance:

end-to-end paper trading works.

---

## Phase 7 — Exit Engine V2

Implement:

- signal-decay exit;
- pivot invalidation;
- ATR trailing;
- premium hard stop;
- time stop;
- DTE exit;
- staged profit taking.

Replay historical sessions.

---

## Phase 8 — LLM Recommendations

Only now add LLM discovery.

Implement:

- news ingestion;
- deduplication;
- LLM enrichment;
- recommendation score;
- Recommendation Pool;
- user review;
- Add to Watchlist action.

LLM still has zero execution authority.

---

## Phase 9 — Strategy Expansion

If account permits, add:

```text
Bull Call Spread
Bear Put Spread
Bull Put Spread
Bear Call Spread
Iron Condor
```

Every structure must:

- be defined-risk;
- have max-loss calculations;
- have strategy-specific exits;
- be covered by account permission checks.

---

## Phase 10 — Research Upgrade

Upgrade from heuristic score to:

```text
Conditional Forward Return Distribution
vs
Market Implied Distribution
```

Then optimize contract choice on:

```text
Expected Value
Expected Shortfall
Liquidity
Greeks
Slippage
```

This is where the system begins becoming genuinely institutional rather than merely rules-based.

---

## Phase 11 — Live Small-Capital Rollout

Start with:

- small Trading Pool;
- high-liquidity symbols;
- strict risk caps;
- low Portfolio Heat;
- no leverage escalation.

Compare:

```text
Backtest
Paper
Live
```

for slippage and behavior.

---

# 44. Claude Implementation Rules

Claude should follow these rules throughout development:

1. Do not silently change trading rules.
2. Every rule must be configuration-driven.
3. Every calculation must have tests.
4. No signal may bypass Risk Engine.
5. No LLM recommendation may bypass user Watchlist approval.
6. No Watchlist symbol may bypass Trading Pool approval.
7. No naked short options.
8. No short stock.
9. Backtest and live share strategy code.
10. No future data leakage.
11. No assuming midpoint fills without an explicit fill model.
12. Every order decision must generate an audit record.
13. Every service must be Docker runnable.
14. Front end and back end must remain independently deployable.
15. Secrets must come from environment configuration.
16. Do not optimize parameters using the final out-of-sample period.
17. Prefer robust parameter regions over single "best" values.
18. `NO TRADE` is a valid and important system output.
19. Cash is a deliberate portfolio position.
20. Risk limits have priority over strategy confidence.

---

# 45. Definition of Done for V1

V1 is complete when:

- both repos build independently;
- full system runs under Docker Compose;
- user can add/remove Watchlist symbols;
- Massive data is ingested only for appropriate symbols;
- user can run historical backtests;
- mechanical Bull/Bear/Neutral signals exist;
- Long Stock / Long Call / Long Put can be simulated;
- user can promote Watchlist symbols to Trading Pool;
- Portfolio Allocator assigns position size;
- dynamic cash reserve is enforced;
- Risk Engine can reject or resize;
- paper orders execute only for Trading Pool symbols;
- positions are monitored;
- Sell-to-Close exits execute mechanically;
- LLM candidates require explicit user approval before Watchlist;
- all decisions have audit logs;
- UI exposes market state, positions, cash, Portfolio Heat, Greeks, risk warnings, and exit plans.

---

# 46. Final Architectural Principle

The platform should enforce this hierarchy:

```text
LLM proposes.
User curates.
Quant measures.
Strategy selects.
Portfolio allocates.
Risk decides.
Execution obeys.
Exit protects.
Audit explains.
```

That separation is the central design principle of the platform.

The objective is **not** to create a machine that trades as often as possible.

The objective is to create a machine that:

> waits for statistically defensible opportunities, expresses them with the appropriate instrument, sizes them according to portfolio risk, retains adequate cash, and exits mechanically when expected value deteriorates.

