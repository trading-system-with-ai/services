# Claude Development Guide — Alpaca Paper + Massive Integration Under Real Cash-Account Constraints

## 0. Purpose

This guide defines the **next implementation milestone** for the trading platform.

The immediate goal is not to add more strategies. The goal is to connect the existing quant/risk/exit framework to:

1. **Massive** for real historical and real-time market data;
2. **Alpaca Paper Trading** for broker-side simulated execution;
3. while enforcing the **same restrictions as the user's real cash account**.

Critical rule:

> **Paper trading must never allow strategies that the user's real account cannot execute.**

Even if Alpaca Paper itself allows shorting, naked option selling, or higher option levels, this platform must reject them.

---

## 1. Current State

The existing platform already has:

- Watchlist
- Trading Pool
- Bull/Bear/Neutral mechanical signals
- volatility regime logic
- Long Stock / Long Call / Long Put instrument selection
- option contract selector
- portfolio allocator
- dynamic cash floor
- correlation buckets
- portfolio heat
- portfolio Greeks
- independent Risk Engine
- mechanical Sell-to-Close exits
- position monitor
- kill switch
- audit trail
- UI for recommendations/watchlist/trading pool/backtest/risk/positions
- internal paper simulation

The next milestone replaces internal fake execution/data with real provider adapters.

---

# 2. Hard Account Constraints

These restrictions must be enforced in **all environments**, including Alpaca Paper.

```python
AccountPermissions(
    long_stock=True,
    long_call=True,
    long_put=True,

    short_stock=False,

    naked_short_call=False,
    naked_short_put=False,

    defined_risk_spreads=False,  # keep OFF until explicitly confirmed
    covered_call=False,
    cash_secured_put=False,

    margin=False,
)
```

Alpaca Paper capability does **not** override platform permissions.

The source of truth is:

```text
Platform AccountPermissions
```

not:

```text
What Alpaca Paper technically permits
```

---

# 3. Allowed Order Directions

For V1 broker integration, only:

```text
BUY_TO_OPEN
SELL_TO_CLOSE
```

are allowed for options.

For stock:

```text
BUY
SELL
```

but `SELL` may only reduce/close an existing long stock position.

Explicitly reject:

```text
SELL_SHORT
BUY_TO_COVER
SELL_TO_OPEN
```

unless a later phase explicitly introduces a permitted defined-risk strategy.

---

# 4. Strategies Allowed in This Milestone

## Allowed

```text
LONG_STOCK
LONG_CALL
LONG_PUT
NO_TRADE
```

## Explicitly Forbidden

```text
SHORT_STOCK
NAKED_CALL
NAKED_PUT
COVERED_CALL
CASH_SECURED_PUT
SHORT_STRADDLE
SHORT_STRANGLE
IRON_CONDOR
BULL_PUT_SPREAD
BEAR_CALL_SPREAD
```

## Deferred

```text
BULL_CALL_SPREAD
BEAR_PUT_SPREAD
```

Even though they are defined-risk, keep them disabled until a separate spread phase. Do not activate them merely because Alpaca Paper supports them.

---

# 5. Core Architectural Rule

Create two independent provider abstractions:

```text
MarketDataProvider
BrokerProvider
```

They have different responsibilities.

---

# 6. Source-of-Truth Separation

## Massive = Market/Pricing Source of Truth

Massive drives:

- stock prices
- stock historical bars
- option historical bars
- option quotes
- option trades
- option chain
- option Greeks
- IV
- OI
- technical features
- volatility regime
- contract selection
- signal calculations
- backtests

## Alpaca = Execution/Account Source of Truth

Alpaca drives:

- paper account state
- cash/buying power as broker reports it
- submitted orders
- broker order IDs
- order status
- partial fills
- fills
- cancellations
- rejections
- broker positions

Do not let Alpaca pricing data silently drive the signal engine while Massive drives the rest.

---

# 7. Required Provider Interfaces

## 7.1 MarketDataProvider

```python
class MarketDataProvider(Protocol):
    async def get_stock_quote(...)
    async def get_stock_bars(...)
    async def get_option_contracts(...)
    async def get_option_chain(...)
    async def get_option_quote(...)
    async def get_option_bars(...)
    async def get_option_trades(...)
    async def get_option_quotes(...)
    async def stream_stock_market(...)
    async def stream_option_market(...)
```

Implement:

```text
MassiveProvider
```

No synthetic fallback in production mode.

If unavailable, return explicit states such as:

```text
MARKET_DATA_NOT_CONFIGURED
CAPABILITY_NOT_AVAILABLE
DATA_STALE
```

## 7.2 BrokerProvider

```python
class BrokerProvider(Protocol):
    async def get_account(...)
    async def submit_order(...)
    async def get_order(...)
    async def cancel_order(...)
    async def list_open_orders(...)
    async def list_positions(...)
    async def close_position(...)
    async def reconcile(...)
```

Implement:

```text
InternalPaperBroker
AlpacaPaperBroker
```

The current local simulator becomes `InternalPaperBroker`.

---

# 8. Environment Configuration

```env
# Market data
MARKET_DATA_PROVIDER=massive
MASSIVE_API_KEY=

# LLM
LLM_PROVIDER=openai
LLM_API_KEY=
LLM_MODEL=

# Broker
BROKER_PROVIDER=alpaca
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_PAPER=true

# Account permissions
ALLOW_LONG_STOCK=true
ALLOW_LONG_CALL=true
ALLOW_LONG_PUT=true

ALLOW_SHORT_STOCK=false
ALLOW_NAKED_SHORT_CALL=false
ALLOW_NAKED_SHORT_PUT=false
ALLOW_DEFINED_RISK_SPREADS=false
ALLOW_COVERED_CALL=false
ALLOW_CASH_SECURED_PUT=false
```

Never infer strategy permissions from the Alpaca Paper account itself.

---

# 9. Mandatory Broker Permission Gate

Before every order:

```text
Strategy Candidate
    ↓
Platform Account Permission Gate
    ↓
Order Direction Gate
    ↓
Position-Reducing / Position-Increasing Check
    ↓
Risk Engine
    ↓
Broker Submit
```

Allowed examples:

```text
Buy 1 AAPL call
=> BUY_TO_OPEN
=> ALLOW
```

```text
Close owned AAPL call
=> SELL_TO_CLOSE
=> ALLOW
```

```text
Buy 100 AAPL shares
=> LONG_STOCK
=> ALLOW
```

```text
Sell 100 owned AAPL shares
=> CLOSE LONG
=> ALLOW
```

Forbidden examples:

```text
Sell AAPL call without owning a long call
=> SELL_TO_OPEN
=> REJECT
```

```text
Sell 100 AAPL while holding 0
=> SHORT_STOCK
=> REJECT
```

```text
Sell 2 calls while owning only 1
=> would create 1 short call
=> REJECT ENTIRE ORDER
```

Prefer fail-safe rejection over ambiguous resizing.

---

# 10. Fix the Existing 422 Carefully

Do **not** remove all 422 behavior.

Only fix validation that incorrectly rejects:

```text
LONG_CALL + BUY_TO_OPEN
LONG_PUT + BUY_TO_OPEN
SELL_TO_CLOSE of an owned option
```

Keep valid 422 cases:

- not in Trading Pool;
- trading disabled;
- promotion requirements not met;
- insufficient data;
- unsupported instrument;
- permission violation;
- risk veto;
- stale market data;
- invalid contract;
- cash-floor violation.

Rejected requests should include structured reason information.

---

# 11. Alpaca Paper Execution Lifecycle

Replace fake local fills with broker-driven state:

```text
Order Preview
    ↓
Full Gate Re-run
    ↓
Risk APPROVE / RESIZE
    ↓
Create local order = PENDING_SUBMIT
    ↓
Submit to Alpaca Paper
    ↓
Save Alpaca order_id
    ↓
ORDER_SUBMITTED
    ↓
Poll/stream broker update
    ↓
NEW / PARTIALLY_FILLED / FILLED / CANCELED / REJECTED
    ↓
Update local order
    ↓
Update local position
    ↓
Audit
```

Never assume:

```text
submitted == filled
```

---

# 12. Partial Fill Support

Required.

Example:

```text
requested: 3 contracts
filled: 1
remaining: 2
```

Local state must represent:

```text
filled_qty
remaining_qty
avg_fill_price
broker_status
```

Risk and portfolio views must use actual filled quantity.

---

# 13. Broker Reconciliation

Implement periodic reconciliation:

```text
Alpaca Account
Alpaca Positions
Alpaca Open Orders
        ↓
Compare
        ↓
Local Portfolio / Orders / Positions
```

If mismatch is material:

```text
RECONCILIATION_MISMATCH
    ↓
GLOBAL TRADING PAUSE
```

Do not auto-heal by guessing.

---

# 14. Cash-Account Behavior

The paper system must behave like the intended real cash account, even if Alpaca Paper exposes more functionality.

At minimum:

- no short stock;
- no naked options;
- no margin-dependent position expansion;
- no synthetic leverage through forbidden short legs;
- no order that creates a negative security position;
- no strategy requiring margin permission;
- respect dynamic cash reserve;
- size from deployable cash and risk budget.

Do not size from paper buying power if that buying power includes margin-like capacity beyond the configured platform cash model.

Use conceptually:

```text
usable_capital = min(
    platform_cash_available,
    broker_cash_compatible_amount
)
```

---

# 15. Massive Integration

Implement real Massive data immediately after Alpaca Long Call/Put lifecycle is working.

Required minimum capabilities:

## Historical

- stock daily bars
- stock minute bars
- option minute bars
- option contract metadata
- historical option quotes where plan permits
- historical option trades where plan permits

## Real-time

- stock quotes/trades
- option chain
- option bid/ask
- option trades
- option Greeks
- IV
- OI
- underlying price

Do not fabricate missing values.

---

# 16. Massive Capability Detection

Detect/record available subscription capabilities.

Example:

```json
{
  "stock_realtime": true,
  "stock_history": true,
  "option_chain": true,
  "option_realtime_quotes": true,
  "option_historical_minutes": true,
  "option_historical_quotes": false
}
```

If unavailable:

```text
CAPABILITY_NOT_AVAILABLE
```

No synthetic fallback.

---

# 17. Historical Backtest Scope

Fetch historical data only for:

```text
Watchlist
+
system reference symbols (SPY / QQQ / VIX as configured)
```

LLM recommendations do not trigger historical option downloads.

---

# 18. Trading Scope

Only Trading Pool symbols may generate executable orders.

Required hierarchy:

```text
LLM Recommendation
    ↓ user approval
Watchlist
    ↓ research/backtest
User promotion
    ↓
Trading Pool
    ↓
Signal
    ↓
Risk
    ↓
Broker
```

No shortcut.

---

# 19. Option Trade Lifecycle

Normal lifecycle:

```text
BUY_TO_OPEN
    ↓
hold
    ↓
SELL_TO_CLOSE
```

Do not exercise as normal strategy behavior.
Do not auto-exercise.

Approaching expiry should trigger the DTE exit policy.

---

# 20. Exit Engine Requirements

Keep:

- signal decay
- signal reversal
- premium hard stop
- ATR trailing exit
- time stop
- DTE exit
- profit taking

For options, normal exit action is:

```text
SELL_TO_CLOSE
```

Before submitting an exit verify:

```text
current long quantity >= requested close quantity
```

Never convert an exit into accidental `SELL_TO_OPEN`.

---

# 21. Risk Engine Independence

Correct dependency:

```text
Signal Engine
Strategy Engine
Allocation Engine
Risk Engine
        ↓
BrokerAdapter
```

Do not let core Risk Engine call Alpaca directly.

---

# 22. Portfolio Limits

Keep current hard controls:

- absolute trade-risk cap;
- single-name risk cap;
- capital exposure cap;
- correlation bucket cap;
- portfolio heat cap;
- dynamic cash floor;
- portfolio Greek limits;
- global kill switch.

Paper mode must enforce the same limits intended for real trading.

---

# 23. Paper Mode Philosophy

Paper mode is **not** a sandbox for unavailable strategies.

Paper mode is a rehearsal for real execution.

Therefore:

```text
paper_permissions == intended_live_permissions
```

Do not add a normal UI toggle that bypasses this.

---

# 24. UI Changes

## Settings

Show clearly:

```text
Broker: Alpaca Paper
Account Mode: CASH-CONSTRAINED SIMULATION

Long Stock        ALLOWED
Long Call         ALLOWED
Long Put          ALLOWED
Short Stock       BLOCKED
Naked Call        BLOCKED
Naked Put         BLOCKED
Defined Spreads   BLOCKED
Covered Call      BLOCKED
Cash-Secured Put  BLOCKED
```

Add note:

> Alpaca Paper may technically support additional strategies, but the platform intentionally mirrors the configured real-account restrictions.

---

# 25. Trade Plan UI

Every proposed order should show:

```text
Instrument
Action
Ticker
Option symbol
Expiration
Strike
Call/Put
Quantity
Bid
Ask
Expected entry
Maximum premium at risk
Portfolio risk %
Cash after
Portfolio heat after
Permission status
Risk status
```

For options use explicit action labels:

```text
BUY TO OPEN
SELL TO CLOSE
```

---

# 26. Positions UI

Where useful show:

```text
Internal Position
Broker Position
Reconciliation Status
```

Status:

```text
MATCHED
PENDING_UPDATE
MISMATCH
```

Mismatch must be prominent.

---

# 27. Order Audit

Persist:

```text
internal_order_id
broker_order_id
client_order_id
ticker
instrument
option_symbol
side
requested_qty
approved_qty
filled_qty
avg_fill_price
broker_status
submitted_at
filled_at
rejected_reason
```

Audit every state transition.

---

# 28. Failure Modes

Fail closed.

## Massive unavailable

```text
PAUSE NEW ENTRIES
```

## Alpaca unavailable

```text
PAUSE NEW ENTRIES
retain local monitoring
alert user
```

## Reconciliation mismatch

```text
GLOBAL PAUSE
```

## Stale option quote

```text
REJECT ENTRY
```

## Unsupported strategy

```text
REJECT BEFORE BROKER CALL
```

---

# 29. Testing Requirements

## Permission tests

Must prove:

```text
LONG_CALL BUY_TO_OPEN => allowed
LONG_PUT BUY_TO_OPEN => allowed
SELL_TO_CLOSE owned option => allowed

SELL_TO_OPEN => rejected
SHORT_STOCK => rejected
close_qty > owned_qty => rejected
spreads => rejected while flag=false
```

## Broker tests

Mock Alpaca states:

- accepted
- rejected
- partial fill
- full fill
- canceled
- timeout
- duplicate client_order_id
- stale status
- broker/local mismatch

## Reconciliation test

Force:

```text
local qty = 2
broker qty = 1
```

Expect:

```text
TRADING_PAUSED
RECONCILIATION_MISMATCH
```

## Cash-account tests

Prove the paper system never sizes using margin-only buying power.

---

# 30. End-to-End Acceptance Test

A complete acceptance test must prove:

```text
1. Add AAPL to Watchlist
2. Fetch real Massive historical data
3. Run backtest
4. Promote AAPL to Trading Pool
5. Enable trading
6. Real Massive live signal becomes valid
7. Strategy selects LONG_CALL or LONG_PUT
8. Contract selector chooses a real option contract
9. Risk engine sizes the position
10. Permission gate confirms cash-account compatibility
11. Alpaca Paper receives BUY_TO_OPEN
12. Broker fill is received
13. Local position matches broker position
14. Position monitor continues evaluating exits
15. Exit condition triggers
16. Alpaca Paper receives SELL_TO_CLOSE
17. Broker fill is received
18. Position closes
19. Realized P&L reconciles
20. Full audit trail exists
```

This is the next major milestone.

---

# 31. Implementation Order

## Iteration A — Broker Abstraction

Implement:

- BrokerProvider protocol
- InternalPaperBroker adapter
- AlpacaPaperBroker skeleton
- configuration
- permission gate

No behavior regression.

## Iteration B — Alpaca Long Stock / Long Call / Long Put

Implement real Alpaca Paper:

```text
LONG_STOCK
LONG_CALL
LONG_PUT
```

and:

```text
BUY_TO_OPEN
SELL_TO_CLOSE
```

Fix only the incorrect 422 blocking legitimate long-option paths.

## Iteration C — Order Lifecycle

Implement:

- broker IDs
- partial fills
- rejected/canceled states
- idempotency
- broker polling/streaming
- local synchronization

## Iteration D — Reconciliation

Implement:

- account sync
- order sync
- position sync
- mismatch detection
- fail-safe pause

## Iteration E — Massive Provider

Implement:

- historical stock data
- real-time stock data
- option contracts
- option chain
- IV/Greeks/OI
- option bid/ask
- historical option bars
- capability detection

## Iteration F — Real Data Backtesting

Replace stub-derived research with Massive-derived datasets.

Validate:

- no look-ahead;
- point-in-time contracts;
- realistic fill models;
- spread/slippage filters.

## Iteration G — Full Massive → Signal → Alpaca Paper Loop

Run the full acceptance test.

Only after this is stable should spreads be added.

---

# 32. Deferred Spread Phase

After the long-only system is stable, defined-risk spreads may be considered only after explicit confirmation that the intended real brokerage account supports them.

First candidates:

```text
Bull Call Debit Spread
Bear Put Debit Spread
```

Do not begin with:

```text
Bull Put Credit Spread
Bear Call Credit Spread
Covered Call
CSP
Short Straddle
Short Strangle
```

because they introduce short-option, assignment, collateral, and margin semantics.

---

# 33. Non-Negotiable Rules

Claude must not violate these:

1. Alpaca Paper capabilities do not define platform permissions.
2. Paper mode must mirror intended real cash-account restrictions.
3. No short stock.
4. No naked short call.
5. No naked short put.
6. No SELL_TO_OPEN in V1.
7. No margin-dependent strategy.
8. No spread until separately enabled.
9. No non-Trading-Pool symbol may trade.
10. No order may bypass Risk Engine.
11. No broker submission may bypass permission checks.
12. No synthetic fallback when Massive is missing.
13. Massive is market-data source of truth.
14. Alpaca is broker/execution source of truth.
15. Broker state must reconcile with local state.
16. Reconciliation mismatch pauses trading.
17. Options normally exit via SELL_TO_CLOSE.
18. Do not auto-exercise.
19. Backtest/live signal logic must remain shared.
20. Every decision must remain auditable.

---

# 34. Definition of Done

This milestone is complete only when:

- Massive real data works;
- Alpaca Paper broker adapter works;
- Long Stock / Long Call / Long Put work end-to-end;
- forbidden paper shorting remains impossible;
- no SELL_TO_OPEN can be produced;
- positions reconcile with Alpaca;
- partial fills are handled;
- risk and cash constraints remain enforced;
- automatic exits send SELL_TO_CLOSE to Alpaca;
- all UI states show broker/data source truthfully;
- the full E2E acceptance test passes;
- Docker Compose runs the complete environment.

The goal is not merely "Alpaca accepts the order."

The goal is:

> **The same constrained strategy that can later run in the user's real cash account is rehearsed end-to-end using real Massive data and Alpaca Paper execution, without accidentally enabling unavailable short or margin strategies.**
