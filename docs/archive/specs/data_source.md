Data Provider Architecture Upgrade

We have decided to use the following paid data subscriptions:

* Alpaca Algo Trader Plus — $99/month
* Massive Financials & Ratios — $29/month
* Total current market/fundamental data cost: approximately $128/month

The purpose of this upgrade is to create a strict, explicit, maintainable data-source architecture.

The most important architectural rule is:

Alpaca is the authoritative source for market data and trading/execution.

Massive is the authoritative source for company fundamentals and financial ratios.

Do NOT arbitrarily mix Alpaca and Massive for the same category of data.

⸻

1. Core Provider Responsibility

The architecture must explicitly define:

ALPACA
│
├── Real-Time Stock Market Data
├── Historical Stock Market Data
├── Real-Time Options Market Data
├── Historical Options Market Data
├── Option Chain
├── Option Greeks
├── Corporate Actions where supported
├── Market Calendar / Trading Status where supported
│
└── Trading / Brokerage
    ├── Account
    ├── Buying Power
    ├── Positions
    ├── Orders
    ├── Fills
    └── Portfolio State
MASSIVE
│
└── Fundamental Data
    ├── Income Statements
    ├── Balance Sheets
    ├── Cash Flow Statements
    ├── TTM Fundamentals
    ├── Annual Fundamentals
    ├── Quarterly Fundamentals
    └── Financial Ratios

There should be NO ambiguity about which provider owns which field.

⸻

2. ALPACA — Authoritative Market Data Provider

Use Alpaca for ALL price-based / market-based information.

2.1 Stock Market Data

The following must come from Alpaca:

Quotes

bid_price
bid_size
ask_price
ask_size
timestamp

Trades

trade_price
trade_size
trade_timestamp
exchange
trade_conditions

Bars / Candles

open
high
low
close
volume
trade_count
vwap
timestamp

Use Alpaca for:

1 minute
5 minute
15 minute
30 minute
1 hour
1 day

and other supported aggregation intervals.

Do NOT fetch these values from Massive.

⸻

3. Technical Indicators Must Be Calculated Internally

Indicators should NOT be treated as authoritative vendor data.

Fetch raw market data from Alpaca and calculate indicators internally.

For example:

Alpaca OHLCV
      ↓
Indicator Engine
      ↓
SMA
EMA
RSI
MACD
ATR
Bollinger Bands
Historical Volatility
Volume Average
Relative Volume
Momentum
Drawdown
Trend Strength

Examples:

SMA20
SMA50
SMA100
SMA200
EMA9
EMA12
EMA20
EMA26
RSI14
MACD
MACD signal
MACD histogram
ATR14
BB upper
BB middle
BB lower

The rule should be:

Vendor provides raw market data.
Our Quant Engine calculates deterministic indicators.

This makes calculations transparent, reproducible, testable and auditable.

⸻

4. Stock Price Features

All price-derived features must use Alpaca data.

Examples:

current_price
previous_close
daily_return
weekly_return
monthly_return
1d_return
5d_return
20d_return
60d_return
120d_return
252d_return
52_week_high
52_week_low
distance_from_52w_high
realized_volatility
ATR
average_volume
relative_volume
VWAP
price_gap
support
resistance
trend
momentum

These should NOT come from Massive.

⸻

5. Real-Time Stock Streaming

Use Alpaca WebSocket for real-time market streaming.

Conceptually:

Alpaca SIP
     ↓
Market Data WebSocket
     ↓
Market Data Service
     ↓
Normalized Market Event
     ↓
Kafka / Internal Event Bus
     ↓
Signal Engine
     ↓
Risk Engine
     ↓
Trading Engine

The system should NOT continuously poll REST endpoints for values that can be streamed.

REST should primarily be used for:

initial state
historical data
snapshots
recovery
backfill

WebSocket should be used for:

real-time quote updates
real-time trades
real-time bars

⸻

6. OPTIONS — Use Alpaca

Alpaca should be the primary provider for options market data.

Do NOT subscribe to Massive Options at this stage.

⸻

7. Option Contract Reference Data

Retrieve available contracts from Alpaca.

Required information includes where available:

option_symbol
underlying_symbol
expiration_date
strike_price
contract_type
    CALL
    PUT
exercise_style
contract_size
tradable

Normalize these into our internal OptionContract model.

⸻

8. Option Chain

Option chains must come from Alpaca.

Input:

underlying = AAPL

Output conceptually:

AAPL
│
├── expiration
│
├── strike
│
├── call / put
│
├── bid
├── ask
├── bid_size
├── ask_size
├── latest_trade
├── underlying_price
│
└── Greeks

The Option Selection Engine should consume our normalized option-chain representation rather than Alpaca-specific JSON directly.

⸻

9. Option Quotes

Use Alpaca for:

bid
ask
bid_size
ask_size
quote_timestamp

Calculate internally:

mid_price = (bid + ask) / 2
spread = ask - bid
spread_pct = spread / mid_price

These values are extremely important for liquidity evaluation.

Never use:

last trade price

alone to determine realistic option entry/exit pricing.

⸻

10. Option Trades

Use Alpaca for:

trade_price
trade_size
timestamp
exchange

These can be used for:

liquidity evaluation
market activity
trade-frequency analysis
execution modeling

⸻

11. Option Greeks

Use Alpaca as the current source for available real-time option Greeks.

Expected normalized fields:

delta
gamma
theta
vega
rho

where available.

Also store:

greeks_timestamp
provider

For example:

{
  "delta": 0.54,
  "gamma": 0.03,
  "theta": -0.12,
  "vega": 0.18,
  "source": "alpaca"
}

Do not silently mix Greeks calculated by different providers.

⸻

12. Implied Volatility

If Alpaca provides implied volatility as part of the relevant response, normalize and use it.

Otherwise calculate IV internally where appropriate.

The architecture must clearly distinguish:

provider_implied_volatility

from:

internally_calculated_implied_volatility

Never combine the two without identifying provenance.

⸻

13. Options Liquidity Metrics

Use Alpaca market data to calculate internally:

bid_ask_spread
spread_pct
volume
open_interest
    if available from an approved source
trade_activity
quote_activity

Then calculate:

Liquidity Score

internally.

For example:

Liquidity Score
    ↓
Bid/Ask Spread
Volume
Open Interest
Quote Activity
Trade Activity

Do not let the LLM invent the Liquidity Score.

⸻

14. Options Selection Pipeline

The system should conceptually work like:

Underlying
    ↓
Alpaca Option Chain
    ↓
Expiration Filter
    ↓
DTE Filter
    ↓
Strike Filter
    ↓
Delta Filter
    ↓
Liquidity Filter
    ↓
IV Filter
    ↓
Risk Filter
    ↓
Option Score
    ↓
Candidate Contracts
    ↓
LLM Explanation

Very important:

Quantitative Filter

happens BEFORE:

LLM Recommendation

The LLM should not scan thousands of raw option contracts and arbitrarily select one.

⸻

15. Historical Options

For now:

Primary Provider = Alpaca

BUT:

Do NOT tightly couple the historical-options backtesting architecture to Alpaca.

This is extremely important.

Alpaca currently has materially less historical option history than we may eventually require for serious long-term option backtesting.

Therefore implement:

HistoricalOptionsProvider

interface.

For example:

HistoricalOptionsProvider
├── AlpacaHistoricalOptionsProvider
│
└── FutureProvider
    ├── Massive
    ├── ThetaData
    ├── CBOE
    └── other institutional source

Do NOT purchase or integrate another provider now.

Just make the architecture replaceable.

⸻

16. Massive — ONLY Fundamental Data

Massive Financials & Ratios should be used only for fundamental/company financial analysis.

Do NOT use Massive for:

stock prices
stock quotes
stock trades
stock OHLCV
option quotes
option trades
option chain
option Greeks
technical indicators

even if a Massive endpoint technically makes some of those available.

We intentionally separate responsibilities.

⸻

17. Massive Income Statement

Use Massive for Income Statement data.

Normalize fields such as:

revenue
cost_of_revenue
gross_profit
operating_expenses
operating_income
interest_expense
pretax_income
income_tax
net_income
EPS
diluted_EPS
shares_outstanding

Use actual field availability and definitions from the API rather than blindly assuming every field exists.

Store:

fiscal_period
fiscal_year
filing_date
period_end_date
timeframe
source

⸻

18. Massive Balance Sheet

Use Massive for fields such as:

cash
cash_and_equivalents
short_term_investments
accounts_receivable
inventory
current_assets
total_assets
accounts_payable
current_liabilities
short_term_debt
long_term_debt
total_debt
total_liabilities
shareholders_equity

Again, use the actual Massive schema as the source of truth.

⸻

19. Massive Cash Flow Statement

Use Massive for:

operating_cash_flow
capital_expenditure
investing_cash_flow
financing_cash_flow
free_cash_flow

If Free Cash Flow is not directly supplied:

calculate internally:

FCF =
Operating Cash Flow
-
Capital Expenditure

and explicitly mark it as:

calculated

rather than:

provider-reported

⸻

20. Massive Financial Ratios

Use Massive Ratios as the provider for available valuation, profitability, liquidity and leverage ratios.

Examples may include:

Valuation

P/E
P/B
P/S
EV/EBITDA

Profitability

ROE
ROA
profit_margin

Liquidity

current_ratio
quick_ratio

Leverage

debt_to_equity
debt ratios

Use the actual current Massive API fields.

Do NOT hallucinate unavailable ratios.

⸻

21. Growth Metrics

Where appropriate, calculate growth ourselves from Massive financial statements rather than depending entirely on precomputed vendor ratios.

Examples:

Revenue YoY Growth
EPS YoY Growth
Net Income Growth
Operating Income Growth
FCF Growth

Calculation:

YoY Growth =
(Current Period - Prior Comparable Period)
/
abs(Prior Comparable Period)

Make sure periods are comparable.

Do NOT compare:

Q1

against:

Q4

and call it YoY.

Use:

Q1 2026
vs
Q1 2025

⸻

22. Fundamental Score

Fundamental Score must use Massive data but be calculated internally.

Example architecture:

Massive Financial Data
          ↓
Fundamental Feature Engine
          ↓
Growth
Profitability
Balance Sheet
Cash Flow
Valuation
          ↓
Fundamental Score

For example:

Fundamental Score
│
├── Growth Score
│   ├── Revenue Growth
│   ├── EPS Growth
│   └── FCF Growth
│
├── Profitability Score
│   ├── ROE
│   ├── Operating Margin
│   └── FCF Margin
│
├── Balance Sheet Score
│   ├── Debt / Equity
│   ├── Current Ratio
│   └── Cash / Debt
│
└── Valuation Score
    ├── P/E
    ├── P/S
    └── EV / EBITDA

Every score must show:

raw value
normalized value
weight
contribution
threshold
data source
calculation formula

The score must NEVER be an unexplained number.

⸻

23. Do NOT Use LLM to Calculate Deterministic Financial Data

LLM should NOT calculate:

SMA
RSI
MACD
ATR
P/E
ROE
ROA
Revenue Growth
EPS Growth
Option Spread
DTE
Portfolio Exposure
Position Size
Risk Score
Fundamental Score
Technical Score
Liquidity Score

Those belong in deterministic code.

LLM should consume the resulting structured data.

⸻

24. LLM Responsibility

LLM is responsible for:

interpretation
context
reasoning
trade thesis generation
risk explanation
conflicting-signal analysis
scenario analysis
human-readable recommendation

Example:

DATA
 ↓
Quant Engine
 ↓
Deterministic Scores
 ↓
LLM
 ↓
Interpretation
 ↓
Trading Plan
 ↓
Human Review

The UI should visibly distinguish:

DATA-DRIVEN
CALCULATED
LLM-GENERATED

content.

⸻

25. Provider Provenance

EVERY normalized data object should contain provider metadata.

For example:

{
    "symbol": "AAPL",
    "price": 220.15,
    "timestamp": "...",
    "source": "alpaca"
}

Financial:

{
    "symbol": "AAPL",
    "revenue": 123456789,
    "fiscal_period": "Q2",
    "fiscal_year": 2026,
    "source": "massive"
}

Calculated:

{
    "symbol": "AAPL",
    "rsi_14": 67.2,
    "source": "internal_calculation",
    "input_source": "alpaca"
}

LLM:

{
    "recommendation": "...",
    "source": "llm",
    "model": "...",
    "generated_at": "..."
}

This provenance architecture is mandatory.

⸻

26. Create Provider Interfaces

Do NOT scatter vendor SDK calls throughout business logic.

Implement abstraction layers.

Conceptually:

MarketDataProvider
HistoricalMarketDataProvider
OptionsMarketDataProvider
FundamentalDataProvider
BrokerProvider

Current implementation:

MarketDataProvider
    └── AlpacaMarketDataProvider
HistoricalMarketDataProvider
    └── AlpacaHistoricalMarketDataProvider
OptionsMarketDataProvider
    └── AlpacaOptionsProvider
FundamentalDataProvider
    └── MassiveFundamentalProvider
BrokerProvider
    └── AlpacaBrokerProvider

Business logic should depend on interfaces, not vendor SDKs.

BAD:

alpaca.get_stock_latest_quote(...)

inside SignalService.

GOOD:

market_data_provider.get_latest_quote(...)

⸻

27. Suggested Service Architecture

Prefer something conceptually similar to:

providers/
│
├── alpaca/
│   ├── market_data.py
│   ├── historical_data.py
│   ├── options.py
│   ├── websocket.py
│   └── broker.py
│
└── massive/
    └── fundamentals.py
services/
│
├── market_data_service.py
├── options_service.py
├── fundamental_service.py
├── indicator_service.py
├── feature_service.py
├── scoring_service.py
├── risk_service.py
├── strategy_service.py
└── trading_service.py

Adapt this to the existing repository architecture rather than blindly creating duplicate services.

⸻

28. API Keys

Use environment variables / secret management.

Example:

ALPACA_API_KEY
ALPACA_API_SECRET
MASSIVE_API_KEY

Do NOT:

hard-code keys
commit keys
print keys in logs
send keys to frontend
send keys to LLM

Production secrets should eventually use AWS Secrets Manager.

⸻

29. Caching Strategy

Different data has different freshness requirements.

Real-time price

Alpaca WebSocket

Do not aggressively cache.

Technical indicators

Update based on timeframe.

Example:

1m indicators → when a new 1m bar closes
5m indicators → when a new 5m bar closes
daily indicators → after daily bar update

Fundamentals

Fundamental data changes slowly.

Cache aggressively.

Example:

Financial Statements:
TTL ~ 12–24 hours
Ratios:
TTL based on provider update behavior and application need

Do NOT call Massive every time the user refreshes a stock page.

⸻

30. Database Provenance

Where appropriate, persisted data should contain:

provider
provider_timestamp
retrieved_at
calculation_version
schema_version

This is particularly important for backtesting.

We need to know:

What did the system know at that point in time?

⸻

31. Point-in-Time Data / Look-Ahead Bias

This is mandatory for backtesting.

When using fundamentals, NEVER simply use the financial period end date.

Example:

Quarter ended:
2026-06-30
Company filed results:
2026-07-30

A backtest running on:

2026-07-15

must NOT have access to those results.

Therefore use:

filing_date
available_date
published_at

where supported.

Prevent look-ahead bias.

⸻

32. Current Data Source Matrix

Implement/document the following matrix:

Data	Provider	Calculation
Stock real-time quote	Alpaca	Raw
Stock trade	Alpaca	Raw
Stock OHLCV	Alpaca	Raw
Historical stock bars	Alpaca	Raw
Stock WebSocket	Alpaca	Raw
VWAP	Alpaca/internal	Prefer documented canonical implementation
SMA	Alpaca OHLCV	Internal
EMA	Alpaca OHLCV	Internal
RSI	Alpaca OHLCV	Internal
MACD	Alpaca OHLCV	Internal
Bollinger Bands	Alpaca OHLCV	Internal
ATR	Alpaca OHLCV	Internal
Historical volatility	Alpaca	Internal
Option contracts	Alpaca	Raw
Option chain	Alpaca	Raw
Option quote	Alpaca	Raw
Option trade	Alpaca	Raw
Option Greeks	Alpaca	Provider value
Option IV	Alpaca/internal	Explicit provenance
Bid/Ask spread	Alpaca	Internal
DTE	Alpaca contract	Internal
Option liquidity score	Alpaca	Internal
Income statement	Massive	Raw
Balance sheet	Massive	Raw
Cash flow	Massive	Raw
Financial ratios	Massive	Raw
Revenue growth	Massive	Internal
EPS growth	Massive	Internal
FCF growth	Massive	Internal
Fundamental Score	Massive	Internal
Technical Score	Alpaca	Internal
Risk Score	Multiple structured inputs	Internal
Recommendation	Structured features	LLM
Trading Plan	Quant + LLM	Hybrid
Account balance	Alpaca Brokerage	Raw
Buying power	Alpaca Brokerage	Raw
Positions	Alpaca Brokerage	Raw
Orders	Alpaca Brokerage	Raw
Execution	Alpaca Brokerage	Raw

This table should become part of the project documentation.

⸻

33. Data Fallback Policy

Do NOT silently substitute Massive market data if Alpaca fails.

If Alpaca market data is unavailable:

Alpaca failure
      ↓
Data status = DEGRADED
      ↓
Prevent decisions requiring fresh market data

Do not:

Alpaca fails
      ↓
silently use Massive

because this creates inconsistent price provenance.

Likewise:

Massive fundamentals unavailable

should not prevent stock-price display.

Instead:

Market Data = AVAILABLE
Fundamentals = TEMPORARILY UNAVAILABLE

The system should degrade by capability.

⸻

34. Data Quality Layer

Every critical input should support validation.

Examples:

timestamp freshness
missing value
NaN
zero price
negative price
crossed market:
bid > ask
stale quote
missing Greeks
unexpected financial period
duplicate financial filing

Bad or stale data must be marked.

Example:

VALID
STALE
PARTIAL
INVALID
UNAVAILABLE

⸻

35. UI Data Transparency

Where useful, UI should allow the user to see:

Market Data
Source: Alpaca SIP
Options Data
Source: Alpaca OPRA
Fundamentals
Source: Massive
Technical Score
Source: Internally calculated
Fundamental Score
Source: Internally calculated from Massive
Recommendation
Source: LLM-assisted analysis

The goal is to make the system trustworthy and auditable.

⸻

36. Important: Do Not Integrate Massive Stocks/Options Market Data

We are intentionally NOT paying for:

Massive Stocks Advanced
Massive Options Advanced

Therefore do not architect the application expecting those subscriptions.

Current subscriptions:

Alpaca Algo Trader Plus
+
Massive Financials & Ratios

Massive should NOT be called for endpoints outside the currently licensed plan unless explicitly verified.

If an endpoint returns authorization failure:

do not work around it.

⸻

37. Do Not Assume Every Desired Dataset Is Already Covered

Some data may require another provider in the future.

Examples may include:

long-history historical options
analyst estimates
earnings consensus
institutional ownership
insider activity
unusual options flow
advanced short-interest data
alternative data
news sentiment

Do NOT force these into Alpaca or Massive if our current subscriptions do not actually provide them.

Instead define:

DataCapability.UNAVAILABLE

and leave an extension point.

⸻

38. Historical Options Warning

Add explicit technical documentation noting:

Alpaca is currently our historical-options provider, but its historical option coverage is shorter than what we may eventually require for institutional-quality option backtests.

Therefore:

DO NOT make Alpaca historical options a permanent architectural dependency.

Backtesting should access:

HistoricalOptionsProvider

not:

AlpacaClient

directly.

This will allow us later to introduce:

Massive Options
ThetaData
CBOE
ORATS
other provider

without rewriting the backtesting engine.

⸻

39. Final Architecture

The final architecture should conceptually be:

                         EXTERNAL DATA
           ┌──────────────────┴─────────────────┐
           │                                    │
        ALPACA                               MASSIVE
           │                                    │
   ┌───────┼────────┐                           │
   │       │        │                           │
Stocks   Options  Broker                  Fundamentals
   │       │        │                           │
   └───────┼────────┘                           │
           │                                    │
           ▼                                    ▼
       Data Normalization / Provider Abstraction
                         │
                         ▼
                     Data Layer
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
        Indicator     Feature      Fundamental
         Engine       Engine         Engine
             │           │           │
             └───────────┼───────────┘
                         ▼
                    Scoring Engine
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Technical      Fundamental       Risk
        Score           Score          Score
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    Decision Engine
                         │
                  Quantitative Data
                         +
                    LLM Reasoning
                         │
                         ▼
                    Trading Plan
                         │
                         ▼
                     HUMAN REVIEW
                         │
                      APPLY
                         │
                         ▼
                    Trading Pool
                         │
                         ▼
                    Risk Controls
                         │
                         ▼
                  Alpaca Execution

⸻

40. Implementation Requirements

Before changing code:

1. Inspect the current frontend and backend repositories.
2. Identify every existing Massive/Polygon/Alpaca market-data integration.
3. Produce an inventory of current data fields and their provider.
4. Identify duplicated provider responsibilities.
5. Identify APIs that are no longer needed.
6. Identify all places where vendor-specific clients leak into business logic.
7. Preserve working functionality.
8. Refactor incrementally rather than rewriting unnecessarily.

Then implement the provider boundaries described above.

⸻

41. Tests

Add unit/integration tests that verify at minimum:

Stock quotes → Alpaca
Stock historical prices → Alpaca
Options chain → Alpaca
Option quotes → Alpaca
Option Greeks → Alpaca
Financial statements → Massive
Financial ratios → Massive
Technical indicators → internal calculation
Fundamental scores → internal calculation
Trading execution → Alpaca

Also verify:

Massive is NOT called for market price
Alpaca is NOT treated as our fundamental provider
LLM is NOT calculating deterministic metrics

⸻

42. Documentation

Create/update a document such as:

docs/data-source-architecture.md

containing:

Provider responsibilities
Data Source Matrix
Data provenance rules
Caching policy
Fallback policy
Point-in-time/backtest rules
Known provider limitations
Subscription assumptions

This should become the single source of truth for provider responsibilities.

⸻

43. Final Deliverables

After implementation provide:

A. Current State Audit

What existed before.

B. Changes Made

Every meaningful architectural/code change.

C. Data Source Matrix

Every important field mapped to:

provider
endpoint
raw/calculated
cache policy
freshness requirement

D. Architecture

Show updated data flow.

E. Tests

What was tested and results.

F. Remaining Gaps

For example:

Historical options before Alpaca coverage
Analyst estimates
Institutional-grade options history
Alternative data

G. Cost

Explicitly document:

Alpaca Algo Trader Plus = $99/month
Massive Financials & Ratios = $29/month
Current expected total = $128/month

Do not add another paid data provider without explicit approval.

⸻

NON-NEGOTIABLE DESIGN PRINCIPLE

The final system should always make it possible to answer:

Where did this number come from?

For every meaningful value, we should be able to identify one of:

ALPACA RAW MARKET DATA
MASSIVE RAW FUNDAMENTAL DATA
INTERNAL DETERMINISTIC CALCULATION
LLM-GENERATED INTERPRETATION

Never blur these categories.

The goal is not merely to make the system work.

The goal is to make every trading decision:

traceable, reproducible, explainable, testable, auditable, and trustworthy.