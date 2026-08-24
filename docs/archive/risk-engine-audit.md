# Risk Engine Audit — Phase A (spec `prompts/risk_engine.md` §61 Phase A)

**Date:** 2026-08-17  
**Prepared by:** Head of Market Risk + Model Validation Lead (this loop's role)  
**Status:** AUDIT — no implementation in this phase.  
**Inputs:** 12 read-only inspection reports (risk engine; allocation/vol/correlation; portfolio/greeks; backtest; market data; broker/permissions; DB/audit; UI; docs/ADRs; gate chain/plans/LLM; test infra; option pricing/exits/income) plus direct re-reading of every file cited below where a claim was surprising. All `file:line` references are relative to `services/` unless prefixed with `ui/` or `prompts/`.

---

## 1. Executive summary

1. The platform already has a complete, tested **Tier 0 hard-limit engine** (`libs/trading_core/risk/engine.py`; `tests/test_risk_engine.py` collects 45 tests — 35 test functions, two of them parametrized — including the seeded invariant test at `:378`) — kill switch, heat, tier budgets hard-capped at 1.5% NAV, single-name, static correlation bucket, heat headroom, regime cash floor, portfolio greek limits — and a 9-gate live chain that matches the first three tiers of §72 in spirit. **Nothing statistical exists**: no returns layer, no VaR/ES, no portfolio volatility, no drawdown on NAV, no stress, no risk contribution, no model health, no risk persistence (grep-verified across `libs/` and `apps/`).
2. Three **Tier 0 correctness gaps** exist, because §2/§72 say the hard limits are the foundation: (a) income opens (covered call / CSP) bypass `assess()`, the Trading Pool check and the RISK_DECISION audit (`apps/gateway/routers/income.py`) — closed in B0 through an additive `assess_income` seam (§7.3), since `assess()` as written raises on a 0 stop distance and rejects `entry_edge=0.0`; (b) dynamic correlation buckets are displayed but never enforced (`portfolio.py:709-729` vs `engine.py:455-471`) — closed SHADOW-first in B and enforced in C, because the 0.70/60d thresholds are documented unvalidated starting points (`correlation.py:9-12`, `portfolio.py:127-130`) and §11/§70 forbid silent production thresholds; (c) the `orders.side` DB CHECK (`migrations/005_orders.sql:19-20`) was never relaxed although the code now inserts `SELL_TO_OPEN`/`BUY_TO_CLOSE` — a latent Postgres failure that the SQLite test harness cannot see; closed in B0. Also `docker-compose.yml:20-31` mounts only migrations 001–012. A fourth, smaller gap: the LIQUIDITY gate skips **underlying** (stock ADV / quote-spread) liquidity; option-leg liquidity IS enforced per contract inside CONTRACT_SELECTION (`libs/trading_core/contracts/spreads.py:51-52, 186-189`; `selector.py:15`).
3. **Data reality** bounds the model ambition: ~600 daily bars per symbol (history starts ~2024-03), no IV history, no VIX/SPX from any configured provider, current-day option chain only, real option daily bars from ~Feb 2024. Historical VaR/ES at 99% has ~6 tail observations; EVT and copulas cannot be validated; empirical IV-shock calibration is impossible today. Massive is a $29 fundamentals-only plan (`docs/data-source-architecture.md:11`); `libs/market_data/massive.py` is still registered (`libs/market_data/__init__.py:42-45, 70`) but superseded for market data (`data-source-architecture.md:57`), its option-chain and indices endpoints 403 on the plan (`DEVLOG.md:1882-1886`) — it contributes **nothing** to risk modeling today and the stale “arrives with the Massive integration” gate strings should be retired.
4. **Numeric dependency decision — stdlib-first, no numpy now.** The repo is deliberately dependency-free (`pyproject.toml:6-19`; house rule in `libs/trading_core/options/bs.py:3`, `greeks.py:3`, `correlation.py:3`, `libs/common/telemetry.py:16`). Everything in Phases B–E (sorted quantiles, sample/shrunk covariance for ≤ 30 names × 600 obs, EWMA, Nelder-Mead GARCH(1,1) MLE, bisection IV solver, Kupiec LR) is O(N²·T) at most and trivially stdlib; `statistics.NormalDist` supplies z/φ/Φ directly. The one non-trivial stdlib routine is a regularized lower incomplete gamma (~20 lines, series + continued fraction) for χ² p-values in Ljung-Box / Christoffersen — hand-rolled and unit-tested against tabulated values. numpy would be justified only if ALL of: a §62-approved model needs dense linear algebra beyond ~30 assets (Cholesky/eigen for copula MC or constrained QP), a measured snapshot build exceeds a stated latency budget, and the routine is error-prone by hand. If ever triggered: numpy only (no scipy/pandas), via `pyproject.toml` + `Dockerfile` change and an ADR — never silently.
5. **Recommended priority:** Phase B0 (Tier 0 hardening: side vocabulary, income → `assess_income`, liquidity gate in report mode, compose mounts) → Phase B (returns layer, Historical/Gaussian VaR & ES, portfolio vol, drawdown, RC foundation, model health, ensemble dispersion, snapshot table, audit widening, first VaR-exceedance results, dynamic buckets + EWMA vol-target multiplier logged SHADOW — all SHADOW) → Phase C (current-vs-proposed, incremental ES, RC gate in shadow, explainability, dynamic-bucket enforcement and vol-target swap after the Q3 window, plan comparison UI) → Phase D (stress + option full revaluation + IV solver + stress gate) → Phase E (EWMA now, GARCH research) → F/G/H research-only or deferred; UI ships incrementally with each phase, not as a trailing Phase I.
6. Deferred/rejected on §62 grounds for THIS platform (small paper book, daily swing strategies, no IV history): copula/GARCH-copula (REJECT for now), EVT (DEFER until ≥ 1500 obs), MVO/tangency (REJECT), turnover metrics (DEFER, no optimizer in production), Monte-Carlo anything (DEFER).

---

## 2. Current risk architecture

### 2.1 Entry paths and where risk is (and is not) applied

| Entry path | Endpoint / function | Runs the 9-gate chain? | Calls `assess()`? | RISK_DECISION audit? |
|---|---|---|---|---|
| Stock long/short, long call/put, vertical spreads — approve | `POST /api/orders/approve` → `run_gate_chain(mode="execution")` `apps/gateway/routers/orders.py:2257-2259` | Yes | Yes (`orders.py:944-957`) | Yes (`orders.py:1018-1036`) |
| Same instruments — preview | `POST /api/orders/preview` `orders.py:1286-1288` (`mode="research"`) | Yes (pool gate reports, does not veto — same qualifier as plans) | Yes | Yes |
| Trade plans generate/revalidate | `apps/gateway/routers/plans.py:172-174`, `:385` (`mode="research"`) | Yes (pool gate reports, does not veto) | Yes | Yes |
| **Covered call / CSP open** | `POST /api/income/covered-call`, `/cash-secured-put` `income.py:155-296` | **No** | **No** (no `assess`, `RiskLimits`, `TradingPool` reference in file — verified) | **No** (single USER ORDER_FILLED at `income.py:389-410`) |
| Backtests (all 8 legs) | `libs/trading_core/backtest/{engine,options}.py` | No | **No** — imports only `ATR_STOP_MULTIPLE` (`backtest/engine.py:45`, `options.py:46`); sizes by `position_pct`/`option_premium_pct` of cash (`engine.py:580`, `options.py:223,514,777,1016`) | n/a (documented gap `docs/DEVLOG.md:486-487`) |
| Read view | `GET /api/portfolio/risk` `portfolio.py:560-816` | n/a | No (heat/state helpers only) | No (read-only by design `portfolio.py:44-45`) |
| Config echo | `GET /api/config` `config.py:66-82` | n/a | n/a | n/a |

### 2.2 The actual gate order today (execution mode)

`GATE_ORDER` is a contract-fixed tuple at `orders.py:279-289`; tests pin it and the UI renders it verbatim.

| # | Gate | What it checks | Where | Spec §72 layer |
|---|---|---|---|---|
| 1 | TRADING_POOL_AUTHORIZATION | pool membership, per-symbol `trading_enabled`, global kill switch (`SystemState`) — vetoes only in execution mode | `orders.py:447-486` | HARD LIMIT (kill switch, pool) |
| 2 | DATA_QUALITY | `ensure_daily_bars` backfill; bars ≥ `sma_slow` (200); last bar ≤ `MAX_BAR_AGE_DAYS`=5 calendar days | `orders.py:257, 492-524` | DATA QUALITY |
| 3 | REGIME | symbol regime: TRANSITION → FAIL; bear regimes forbid bullish entry | `orders.py:550-576` | (signal) |
| 4 | DIRECTIONAL_SIGNAL | NEUTRAL under AUTO → FAIL; strength tier via `strength_tier()` | `orders.py:584-615` | (signal) |
| 5 | VOLATILITY | live chain, `atm_iv`/`rv20`, `classify_vol_regime`; **§8 matrix + AccountPermissions applied here** (`select_instrument`) | `orders.py:632-687` | ACCOUNT PERMISSION (embedded, degrades rather than vetoes) |
| 6 | INSTRUMENT | matrix NO_TRADE → FAIL | `orders.py:710-724` | (signal/permission) |
| 7 | LIQUIDITY | **always SKIPPED** for the **underlying** — `SKIP_NO_OPTION_DATA` "arrives with the Massive integration"; the code comment says option liquidity is enforced per contract by the §9.1 filters (short-leg OI ≥ 50, spread ≤ 15%: `spreads.py:51-52, 186-189`; spread term in the selector `selector.py:15`) — which is true, so only stock ADV / quote-spread liquidity is missing | `orders.py:297, 712-715` | HARD LIMIT (underlying part missing) |
| 8 | CONTRACT_SELECTION | §9 selector / vertical spread; fail-closed on missing greeks | `orders.py:738-806` | (execution) |
| 9 | RISK_APPROVAL | broker cash (fail-closed), snapshot build, vol-targeting multiplier, greeks, `assess()` | `orders.py:826-1002` | HARD LIMITS |

Observations against §72:
- ACCOUNT PERMISSION is not a standalone first gate; it lives inside gate 5/6 as the `_finalize` degradation ladder (`libs/trading_core/strategies/instrument.py:135-203`) which *substitutes* an instrument rather than vetoing. Functionally safe (permissions cannot be broadened; forbidden flags refused at construction `instrument.py:98-115`, Settings `libs/common/config.py:148-168`), but not explainable as "permission denied".
- Kill switch/pool (hard limits) precede DATA_QUALITY — the reverse of §72's ordering. Every gate is a veto so the outcome is identical; only the recorded order differs.
- STATISTICAL / STRESS / CONCENTRATION / MODEL HEALTH tiers do not exist. `RiskAssessment` (`engine.py:193-207`) has no fields for them.
- The user direction override (`orders.py:590-597`) bypasses the NEUTRAL veto but not the risk engine — acceptable.
- LLM boundary is clean: `recommendations.py` writes recommendation rows only; nothing in `orders.py`/`plans.py` imports it; only USER promote reaches the Watchlist (`migrations/001_initial.sql:54-55`, `006:9-12`).

### 2.3 Diagram — current pipeline vs §2/§72 target

```
CURRENT (execution mode)                          TARGET (§2 / §72)
------------------------                          -----------------
TRADE REQUEST (ticker, qty, direction)            TRADE REQUEST
  │                                                 │
  ▼ gate 1  TRADING_POOL_AUTHORIZATION              ▼ ACCOUNT PERMISSION      (explicit gate 0; today inside gate 5/6)
  │  pool ∧ symbol enabled ∧ kill switch            │
  ▼ gate 2  DATA_QUALITY (≥200 bars, ≤5d old)       ▼ DATA QUALITY            (bars + snapshot as_of/TTL, §55)
  ▼ gate 3-6 REGIME → SIGNAL → VOL → INSTRUMENT     │
  │  (§8 matrix + permission degradation)           ▼ HARD RISK LIMITS        (Tier 0, unchanged: engine.py steps 1-5f)
  ▼ gate 7  LIQUIDITY  ── underlying always SKIPPED  │   + underlying liquidity gate (option legs already filtered)
  ▼ gate 8  CONTRACT_SELECTION                      ▼ STATISTICAL RISK        (Phase B/C: hist/gauss VaR·ES, cond. vol, incremental ES)
  ▼ gate 9  RISK_APPROVAL = assess()                ▼ STRESS RISK             (Phase D: historical + hypothetical, option full reval)
  │   1 kill switch  2 heat gate  3 tier budget      ▼ PORTFOLIO CONCENTRATION (Phase C: RC gate, dynamic buckets enforced)
  │   4 base qty     5a-5e clamps  5f greeks(REJECT) ▼ MODEL HEALTH            (Phase B: ACTIVE/DEGRADED/UNAVAILABLE/FAILED, model-risk state)
  ▼                                                 ▼
APPROVE / APPROVE_WITH_RESIZE / REJECT            APPROVE / RESIZE / REJECT  (+ binding constraints, requested vs approved, before/after)
  │  audit: decision, gates, reason_codes only       │  audit: + snapshot id, limits, model versions/health, sizes (§66)
  ▼                                                 ▼
broker (paper only) / simulated fill              broker

Bypasses today: income opens (no assess, no pool gate, no audit); backtests (no assess).
```

### 2.4 Read-side architecture

`GET /api/portfolio/risk` (`portfolio.py:560-816`) recomputes everything per request from: broker live cash (`portfolio.py:578-593`), DB OPEN positions priced at the last stored daily close (`:236-268`), options at premium **book** value (`:158-198`), heat/state via the shared engine helpers (`:661-663`), static + dynamic buckets (`:690-729`), greeks re-resolved from today's chain (`:289-472`), vol-targeting proxy (`:475-558`). Nothing is persisted; there is no NAV/risk time series anywhere in the schema (all 16 migrations checked).

---

## 3. Existing hard limits (Tier 0) — MUST BE PRESERVED

| Limit | Parameter | Default | Enforced at | Audited? |
|---|---|---|---|---|
| Kill switch rejects all new risk | `PortfolioSnapshot.trading_enabled` ← `SystemState.trading_enabled` (DB default False) | off at startup | `engine.py:325-336`; gate 1 `orders.py:463-486`; income `income.py:116-127` | Yes via RISK_DECISION (chain); TRADING_PAUSED/RESUMED `trading_control.py:62-70`; KILL_SWITCH_TRIGGERED `broker.py:527-549` |
| Portfolio heat gate (pre-trade) | `RiskLimits.heat_reject` | 0.08 | `engine.py:342-352` | Yes (reason `HEAT_LIMIT`) |
| Heat states | `heat_elevated/high/reject` | 0.04/0.06/0.08 | `engine.py:217-229` | display |
| Signal validity | `strength_weak` | 25.0 | `engine.py:363-374` | Yes (`SIGNAL_TOO_WEAK`) |
| Per-trade tier budgets | `budget_weak/moderate/strong/very_strong` | 0.5/0.75/1.0/1.25% NAV | `engine.py:375-378` | Yes |
| Absolute per-trade ceiling | `abs_max_trade_risk` | 1.5% NAV | `engine.py:375-378` (after `budget_multiplier`) | Yes |
| Base sizing | floor(NAV·budget/stop), ≤ requested; `stop_distance ≤ 0` raises | — | `engine.py:382-393` | Yes |
| Single-name strategy risk | `single_name_risk` | 1.5% NAV | `engine.py:424-433` | Yes (`SINGLE_NAME_RISK_CAP`) |
| Single-name capital | `single_name_capital` | 20% NAV | `engine.py:439-450` | Yes |
| Correlation bucket (STATIC only) | `bucket_risk`, `correlation_buckets`={TECH_MEGA: NVDA, AMD, AVGO, MSFT, GOOGL, META, AAPL, QQQ, SMH, TSLA} | 3% NAV | `engine.py:66-81, 455-471` | Yes (`BUCKET_LIMIT_<name>`, f-string over the dict key `:465`) |
| Heat headroom (post-trade, strictly <) | `heat_reject` | 0.08 | `engine.py:477-489` | Yes |
| Regime cash floor | `cash_floors[SPY regime]` | STRONG_BULL .15 / MILD_BULL .25 / NEUTRAL .40 / MILD_BEAR .50 / STRONG_BEAR .60 / TRANSITION .50 | `engine.py:52-62, 495-506` | Yes (`CASH_FLOOR`) |
| Portfolio greek limits (REJECT-only) | `max_delta_notional_pct_nav`, `max_net_theta_pct_nav`, `max_net_vega_pct_nav` | 1.50 / 0.001 / 0.01 NAV | `engine.py:517-560` (step 5f; breach → `qty = 0` → REJECT at `:556-560`) | Yes (`PORTFOLIO_*_LIMIT`) |
| Vol-targeting bounds | `VolTargetParams` min/max/target | 0.25 / 1.2 / 0.12 | `libs/trading_core/allocation.py:38-40, 72-75` | multiplier only in gate detail text (`orders.py:957-962`) |
| Stock stop distance | `ATR_STOP_MULTIPLE` × ATR14 | 2.0 | `engine.py:45-49`; `orders.py:534`; backtests | via ORDER_* |
| Short-stock gap factor | `SHORT_STOCK_GAP_RISK_FACTOR` | 2.0 | `orders.py:276, 846` | via ORDER_* |
| Data quality | `MAX_BAR_AGE_DAYS`, `sma_slow` bars | 5 days / 200 bars | `orders.py:257, 506-517` | Yes (gate FAIL) |
| Broker cash unverifiable → fail closed; usable cash = broker CASH never buying_power | — | — | `orders.py:359-386, 817-865` | Yes |
| Trading pool authorization | pool row, `trading_enabled` | — | `orders.py:447-486` | Yes |
| Account permissions (10 flags; naked×2 locked forever) | `AccountPermissions` via one factory | spreads/covered/CSP/short_stock/margin default False | `instrument.py:52-116, 135-203`; `config.py:148-168`; income `income.py:99-113` | CONFIG_CHANGED on toggle |
| Idempotency, no pyramiding, one in-flight open per ticker, execution lock | `client_order_id` UNIQUE | — | `orders.py:2171-2254`; `migrations/005:17` | Yes |
| Approve re-runs the FULL chain; previews never trusted | — | — | `orders.py:38-40, 2261-2274` | Yes |
| Collateral law (CC ≥ 100 free shares/contract; CSP strike×100 reserved, simulated mode) | — | — | `income.py:173-200, 264-286` | ORDER_FILLED only |
| Liquidity gate | — | **underlying (stock ADV / quote spread) not implemented** — gate always SKIPPED; **option-leg** liquidity enforced per contract (OI ≥ 50, spread ≤ 15%, real NBBO required) | `orders.py:712-715` (SKIP); `spreads.py:51-52, 186-189`, `selector.py:15` (option legs); pool check stub `trading_pool.py:55-58` | option-leg rejections appear in CONTRACT_SELECTION detail only |

Invariants pinned by tests (`tests/test_risk_engine.py:378-451`, seed 20260810): approved risk ≤ NAV·abs cap and ≤ tier budget; heat_after < reject; cash_after ≥ floor; qty ≤ requested; every REJECT has a reason code. `RiskLimits` is only ever constructed with defaults (`orders.py:414`, `portfolio.py:567`, `config.py:73`) — no runtime override, no version field.

---

## 4. Existing statistical models (estimation-like logic)

| Model | Convention | Where | Consumed by |
|---|---|---|---|
| Realized volatility RV20 | sample stdev (ddof=1) of last 20 close-to-close **log** returns × √252; None during warm-up | `libs/trading_core/features/indicators.py:205-233` (verified) | vol regime IV/RV (`options.py:202`), vol-targeting proxy (`portfolio.py:527-537`), watchlist overview |
| Rolling Pearson correlation | last 60 daily **log** returns; `math.fsum`; None on < 61 closes or zero variance | `libs/trading_core/correlation.py:27-76` (verified) | dynamic buckets (display only) |
| Dynamic correlation buckets | union-find components at ρ > 0.70 (strict), singletons dropped | `correlation.py:79-137`; `portfolio.py:129-130, 709-729` | `/api/portfolio/risk` display; **not** `assess()` |
| Vol-targeting "forecast" | NAV-weighted arithmetic mean of per-position RV20; cash and unpriceable positions weigh 0; no covariance, no option delta; multiplier = clamp(0.12/forecast, 0.25, 1.2) | `portfolio.py:475-558` (self-labelled "CRUDE v0 FORECAST PROXY"), `allocation.py:56-75` | `assess(budget_multiplier=…)` `orders.py:884-955` |
| IV regime | LOW/NORMAL/HIGH/EXTREME from ATM IV level (0.20/0.35/0.60) + IV/RV ratio (1.1/1.5/2.0); "PROVISIONAL" pending IV history | `libs/trading_core/volatility.py:29-155` | §8 matrix, tradeability; **not** the risk budget |
| ATM IV / expected move | closest-strike call IV, nearest expiry DTE ≥ 30; straddle mid/spot; `iv_rank` always None | `apps/gateway/routers/options.py:55-62, 159-213` | vol regime |
| Option greeks | provider pass-through (Alpaca snapshot greeks + IV; Massive; stub = Black-Scholes); assumed units theta/day, vega/IV-pt; linear aggregation with builtin `sum()` | `libs/market_data/alpaca.py:513-607`; `libs/trading_core/greeks.py:44-153` | greek limits, risk view |
| Black-Scholes-Merton pricer + greeks | European, `math.erf`, r=0.04 default, q=0; **no IV solver** | `libs/trading_core/options/bs.py:104-199` | stub provider only in production paths |
| Directional edge score | weighted-share technical composite, versioned `score-weights-v1-grouped` | `libs/trading_core/signals/directional.py` | tier budget via `strength_tier` |
| Backtest metrics | Sharpe/Sortino on **simple** daily equity returns, sample stdev, √252, no risk-free; max drawdown = min(equity/running_max − 1) | `libs/trading_core/backtest/engine.py:363-446, 655-660` | backtest records |
| Strategy health | win rate / PF / expectancy / drawdown on **closed-trade realized PnL** (not NAV) | `libs/trading_core/health.py:72-181` | `/api/health/strategy` (read-only, no pause automation) |

Absent (grep-verified `libs/`, `apps/`): VaR, ES, covariance matrix, portfolio σ, GARCH/EWMA, EVT, copula, skew/kurtosis diagnostics, Spearman, risk contribution, NAV drawdown, stress scenarios, Kupiec/Christoffersen, IV solver, Monte Carlo.

Return conventions to standardize (spec §3): log returns are implemented twice (`correlation.py:27-39`, inline `indicators.py:225`); simple returns once on equity (`backtest/engine.py:392-396`); no date alignment (`portfolio.py:271-286 stored_closes_by_ticker` returns unaligned close lists); no metadata record.

---

## 5. Data available for risk modeling

| Dataset | Storage | Depth | Frequency | Gaps / notes |
|---|---|---|---|---|
| Stock daily OHLCV | `stock_bars_daily` (`migrations/007`, `db.py:58-82`), watchlist symbols + SPY (QQQ only if watchlisted; VIX never) | `BACKFILL_DAYS`=600 complete trading days from a symbol's FIRST fetch (`analysis.py:60, 128-216`); live store 601 bars/symbol on 2026-08-13 → history begins ~2024-03/04 (`DEVLOG.md:823`); refresh is append-forward only, no backward extension path (`analysis.py:219-294`) | daily | Alpaca `get_daily_bars` accepts arbitrary `days` (`alpaca.py:318-376`) so a one-off deep backfill is feasible; vendor depth for daily bars is **not verified** in repo. Provider per batch only in DATA_BACKFILL audit; Massive-adjusted and Alpaca-split-adjusted bars coexist in one table (`DEVLOG.md:823-824`). |
| Intraday bars | `stock_bars_1m` hypertable (`migrations/002:18-30`) | **empty** — no ORM, writer or reader (grep-verified) | — | websocket minute bars ignored (`alpaca_stream.py:16`) |
| Index history | SPY yes (stored); QQQ conditional; **VIX/SPX none** | as above | daily | Alpaca declares VIX/^VIX/^SPX/^GSPC unservable (`alpaca.py:99`); Massive indices 403 on plan (`DEVLOG.md:1884-1885`) |
| Massive (inspection item 11) | not used by any risk path | — | — | fundamentals-only $29 plan (`docs/data-source-architecture.md:11`); `libs/market_data/massive.py` still registered (`libs/market_data/__init__.py:42-45, 70`) but superseded for market data (`data-source-architecture.md:57`); option chain + indices 403 (`DEVLOG.md:1882-1886`). Contributes nothing to risk modeling today; a future Fundamental Score is a *signal* input, not a risk input |
| Option chain (current) | not persisted; 20 s in-process cache (`options.py:93`) | today only; `as_of != today` raises (`alpaca.py:467-474`) | per request | NBBO, greeks, IV, OI (day-cached), price_basis; one-sided NBBO → mid = ask/2 (`alpaca.py:542-552`) |
| Option greeks / IV history | **none** | — | — | provider serves none (`docs/data-source-architecture.md:145-147`); no table; `iv_rank` None |
| Option daily bars (historical) | not persisted; refetched per backtest | ~Feb 2024 → now, full contract life, (open, close) only, single un-paginated request (`alpaca.py:803-835`) | daily | no historical NBBO/greeks/OI; enables IV back-out only with a solver (none) |
| Positions | `positions` (`db.py:259-336`) | full history OPEN/CLOSED | event | no entry IV/greeks/spot, no mark history; option MV = book value in risk view |
| Cash / NAV | broker live cash per read; simulator ledger; NAV derived never stored (`migrations/004:4-5`); broker `BrokerAccount.equity` (`libs/broker/provider.py:159`, `alpaca.py:393`) is the broker's own NAV, read per request, never stored | none stored | — | **no NAV time series stored** → live drawdown not computable today. Non-fabricated sources for a series: (i) persist broker equity per daily snapshot going forward (Phase B); (ii) Alpaca's account portfolio-history endpoint — **not integrated** (grep `portfolio_history` in `libs/broker/` empty), would give real equity history for the paper account, DEFERRED behind provider work + provenance labelling; (iii) simulated mode: NAV since deployment reconstructable from `orders` + stored closes (labelled RECONSTRUCTED) |
| Risk decisions | `audit_events` RISK_DECISION details (`orders.py:1018-1036`) | since deployment | per chain run | decision/mode/execution_authorized/veto_gate/gates/reason_codes only; no sizes, snapshot, limits, versions |
| Trade plans | `trade_plans.preview` JSONB (`migrations/013:19`) | per plan | event | only place a full `RiskAssessment` payload is persisted |
| Backtest runs | `backtests` JSONB (`migrations/003`) | per run | — | single-symbol equity curves only |

**Historical stress windows actually usable now (daily, stocks):** 2024-08-05 (vol spike) and 2025-04 (tariff drawdown) are inside the stored 600-bar window for symbols backfilled by 2026-08. **Not stored:** 2020-03, 2022 — reachable only via a deep on-demand backfill (must be verified live and documented per `docs/data-source-architecture.md` before relying on it). **Options:** only 2024-08 and 2025-04 have any option data, daily close only, no IV. **VIX for any window:** unavailable from every configured provider — IV-shock parameterization must be either research-grid (documented as unvalidated) or derived from realized-vol jumps until an ATM-IV daily series has been accumulated in-house.

Frequency decision (§53): all strategies are daily-bar swing strategies (`analysis.py` indicators, `RegimeParams`); a **daily** risk layer is correct. Portfolio greeks already refresh per request (seconds) and satisfy the intraday-greeks tier without bars.

---

## 6. Position & instrument model

| Instrument | max_loss (heat unit) | Market value in NAV (risk view) | Greeks source | Exit engine | Notes |
|---|---|---|---|---|---|
| LONG_STOCK | qty × 2×ATR14 (`orders.py:2436-2444`) | qty × last stored close | (+1, 0, 0, 0) | `evaluate_exit` (`exits/engine.py`) | stop fixed at open, never re-marked |
| SHORT_STOCK | qty × stop × 2.0 gap factor (`orders.py:2426-2434`) | −qty × close (liability) | (−1, 0, 0, 0) | mirrored `evaluate_exit` | unbounded loss represented by an estimate |
| LONG_CALL / LONG_PUT | qty × fill × 100 (full premium) (`orders.py:2376-2392`) | qty × avg_price × 100 (**book**, `portfolio.py:191-193`) | same contract located in today's chain (`portfolio.py:201-216`); missing → zeros + data_ok:false | `evaluate_option_exit` (premium stop −45%, DTE ≤ 21) | premium at risk = max_loss (no field named so); no entry IV stored |
| BULL_CALL_SPREAD / BEAR_PUT_SPREAD | qty × net_debit × 100 (`orders.py:2395-2419`) | book | long − short net (`portfolio.py:357-385`) | `evaluate_option_exit` on net | short leg columns `migrations/015` |
| COVERED_CALL | **0.0** (`income.py:363-374`) — stock row carries heat | −qty × credit × 100 | short leg negated (`portfolio.py:329-356`) | `evaluate_short_premium_exit` | capped upside / assignment risk invisible to heat |
| CASH_SECURED_PUT | (strike − credit) × 100 × qty (`income.py:369-373`) | −qty × credit × 100 | short leg negated | `evaluate_short_premium_exit` | `cash_reserved` not netted from `cash` in view or order-path snapshot |

Option pricer availability for full revaluation (§21–§22): `bs_price`/`bs_greeks` exist (`bs.py:104-199`; European; r=0.04, q=0 defaults; theta/day, vega/pt; put-call parity tested to 2e-13). Missing for revaluation: an implied-vol solver (bisection on `bs_price` is a ~30-line stdlib addition), a per-position baseline bundle (S0, IV0, T0, r, q, right, strike, mark0), a basis adjustment (mark0 − model0) so scenario P&L is anchored to the real mark, and leg-aware revaluation for spreads/income. DTE is calendar days computed at read time (`positions.py:144-150`); `opt_expiry` is stored as ISO string.

---

## 7. Missing statistical risk — gap analysis matrix (spec §62 six questions applied)

Legend — COST: L/M/H (implementation); COMP: computational cost + stdlib feasibility; RISK: model risk; PRIO: P0 (Phase B0/B), P1 (C/D), P2 (E), P3 (research/later). Confidence-level policy for the whole matrix: 95% and 99% are both computed and displayed; **gates use ES 95% first** while the tail sample is ~600 obs (99% ES averages ≈ 6 observations — displayed with its sample size, never gated on alone).

### 7.1 Foundations

| Capability | Current | Missing | Model value | Data | Cost | Comp (stdlib) | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Returns layer (§3) | log returns ×2 (`correlation.py:27`, `indicators.py:225`), simple on equity | shared `simple_returns`/`log_returns`, date-aligned multi-asset return matrix, `ReturnSeries` metadata (return_type, frequency, lookback, as_of, data_source, model_version) | prerequisite for everything | stored closes via `stored_closes_by_ticker` (extend to (ts, close)) | L | trivial | low | P0 | IMPLEMENT NOW | every model below needs aligned returns + provenance |
| Risk model registry (§4) | none | `RiskModel` protocol (calculate/validate/diagnostics/metadata), typed `ModelResult`, `ModelHealth`, registry, mode RESEARCH/SHADOW/PRODUCTION | avoids "everything in engine.py" | — | L–M | trivial | low | P0 | IMPLEMENT NOW | needed before the second model exists |
| PortfolioRiskSnapshot (§45) | `PortfolioSnapshot` = nav/cash/positions/regime/kill (`engine.py:156-169`); untyped dict view | typed snapshot with as_of, exposures, VaR/ES, cond., stress, drawdown, greeks, RC, correlation state, model health, risk state | single source for view + pre-trade + audit | — | M | trivial | low | P0 | IMPLEMENT NOW | §45 explicitly forbids a JSON dumping ground; keep Tier 0 `PortfolioSnapshot` untouched and compose |
| Persistence (§56) | none | `risk_model_runs`, `risk_snapshots`, `risk_metrics`, `risk_contributions`, `atm_iv_daily` (Phase B); `stress_runs` (D). §56 `model_diagnostics` **adopted** as the `diagnostics` JSONB column of `risk_model_runs` (renamed per §56 “may redesign names”); §56 `optimization_runs` **REJECTED** — no optimizer runs in production (§7.7) | audit, drift, exceedance testing, NAV series for drawdown | — | M | trivial | low | P0 | IMPLEMENT NOW | without history there is no drawdown, no VaR backtest, no §44 reproducibility |
| ATM-IV daily accumulation | computed per read, discarded (`options.py:159-213`) | persist ATM IV (+ expiry, spot, source) per underlying per day | unlocks IV rank / empirical IV shocks over time | today's chain | L | trivial | low | P0 | IMPLEMENT NOW | cheapest way to ever get IV history; honest until N ≥ ~120 days |

### 7.2 Tier 1 — core statistical risk (Phase B)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Historical VaR (per position + portfolio, 95/99, 1D) | none | empirical quantile of book P&L series (stock: qty×Δclose; options delta-linear in B, full reval in D; short/income signed) | ordinary downside threshold; §6 | ~600 obs | L | sort 600 floats | medium (small tail; overlapping multi-day) | P0 | IMPLEMENT NOW | Q1 yes (no downside distribution today); Q2 marginal (state N); Q3 exceedance backtest; Q4 shadow → gate; Q5 trivial |
| Gaussian VaR | none | μ,σ of same series; z 1.645/2.326 | comparison + dispersion signal | same | L | trivial | high if trusted alone | P0 | IMPLEMENT NOW | Q6: it *is* the simpler model; used as ensemble member and Gaussian-trust check |
| Historical ES 95/99 | none | mean of losses beyond VaR | first-class per §7 | same | L | trivial | medium | P0 | IMPLEMENT NOW | favored over VaR for approval (§7) |
| Gaussian ES | none | σ·φ(z)/(1−α) | ensemble | same | L | trivial | high | P0 | IMPLEMENT NOW | one line once Gaussian VaR exists |
| Multi-day horizons (5D/10D) | none | √h scaling (labelled) — non-overlapping 10D from 600 obs = 60 samples, too few for empirical; the §12 5D/10D *conditional* volatility outputs are the same √h scaling of EWMA/GARCH σ_t (Phase E), labelled, never an independent forecast | modest | same | L | trivial | high | P1 | IMPLEMENT AS RESEARCH MODE | display "√h-scaled" only; do not gate |
| Portfolio volatility | NAV-weighted RV20 proxy (`portfolio.py:475-558`) mislabelled "forecast" | σ_p = stdev of book P&L series (equivalently √(w'Σw), sample Σ over aligned returns), delta-adjusted for options | candidate replacement for the crude proxy in §14 vol targeting — the *swap* is the §7.5 “Vol targeting v2” row and is SHADOW-first | same | L | n ≤ 30 × 600 | low | P0 | IMPLEMENT NOW | fixes the negative-weight edge case (short/income books drive proxy ≤ 0 → multiplier 1.0 or clamped UP to 1.2, `portfolio.py:534-536`) |
| Live NAV drawdown | none (health drawdown is realized PnL only, `health.py:99-107`) | daily NAV (= broker `BrokerAccount.equity`, or simulator ledger NAV) in `risk_snapshots`; current & max drawdown; plus "reconstructed current-book drawdown" over stored closes labelled RECONSTRUCTED | Tier 1 per §5; kill-switch/cash-floor input later | NAV series from first snapshot; optional real backfill from Alpaca portfolio-history (DEFERRED, see §5) | L | trivial | low | P0 | IMPLEMENT NOW (forward series) / DEFER (broker equity-history backfill) | honest: real drawdown accrues from first snapshot; broker-served equity history is a legitimate, non-fabricated backfill once integrated and provenance-labelled; no synthetic history |
| Risk contribution — volatility | none | component σ: w_i(Σw)_i/σ_p, sums to σ_p | hidden concentration | Σ | L | trivial | low | P0 | IMPLEMENT NOW | Q1 yes — dollar caps ≠ risk weight |
| Risk contribution — ES | none | Euler component ES = mean position P&L on tail days; sums to portfolio ES exactly | concentration under fat tails; §10/§33 | tail days (≈30 at 95%) | L | trivial | medium (noisy at 99%) | P0 | IMPLEMENT NOW | exact additivity → property test "RC sums to total" |
| Distribution diagnostics (§15) | none | skew, excess kurtosis, Jarque-Bera; label NORMAL_LIKE / HEAVY_TAIL / LEFT_SKEWED / UNSTABLE | reduces trust in Gaussian; feeds model-risk state | same | L | trivial | low | P0 | IMPLEMENT NOW | cheap and directly informs which VaR to trust |
| Model ensemble / dispersion (§39-§40) | none | snapshot carries all views; dispersion = max/min across hist/gauss/(EWMA) VaR; MODEL_DISPERSION_HIGH flag | model-risk signal | — | L | trivial | low | P0 | IMPLEMENT NOW | falls out of B |
| Model health + model-risk state (§41, §59) | none | ACTIVE/DEGRADED/UNAVAILABLE/FAILED with reason per model; LOW/ELEVATED/HIGH portfolio model-risk state | fallback logic; §58 | — | L | trivial | low | P0 | IMPLEMENT NOW (label); budget effect SHADOW | hard limits stay active regardless (§58) |
| Rolling Spearman (§18) | Pearson only | rank correlation on same windows | robustness to outliers | same | L | O(n² · w log w) | low | P1 | IMPLEMENT AS RESEARCH MODE (display next to Pearson, Phase C with correlation state) | fails §62 Q4 (affects no decision) and Q6 (Pearson exists); cheap, so it ships with the C correlation-state work rather than in B |
| Correlation regime shift (§19) | single 60d Pearson, values discarded (`correlation.py:119-125`) | normal (long window ≥ 250d) vs current (60d) vs stress-conditioned (worst-10% SPY days); CorrelationState NORMAL/ELEVATED/CONVERGING | diversification decay | same | L–M | trivial | medium (stress window small) | P1 | IMPLEMENT NOW (Phase C) | feeds RC gate/tech concentration; §19 |
| Diversification ratio (§34) | numerator exists (Σ w_i σ_i) | denominator σ_p | diagnostic | Σ | L | trivial | low | P1 | IMPLEMENT NOW (diagnostic only) | one line once Σ exists; not an allocator |

### 7.3 Pre-trade portfolio risk (Phase C)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Current vs proposed portfolio (§46) | heat_before/after, cash_after only (`engine.py:604-606`) | proposed-book P&L series = current + candidate; VaR/ES/σ/stress/greeks/RC before & after | "does this trade damage diversification?" | snapshot + candidate return series | M | recompute on 600 obs | low | P1 | IMPLEMENT NOW | core decision value |
| Incremental VaR/ES (§8) | none | ES(proposed) − ES(current) at requested and approved qty | correlation-aware sizing | same | L | trivial | medium | P1 | IMPLEMENT NOW | Q4: directly changes RESIZE |
| Marginal ES (§9) | none | ∂ES/∂qty via component ES of the candidate | allocation | same | L | trivial | medium | P1 | IMPLEMENT NOW | falls out of Euler RC |
| RC concentration gate (§11) | dollar-based single-name/bucket only | `max_single_position_es_contribution`, `max_bucket_es_contribution` (research defaults e.g. 0.35/0.50 — to be validated); qty cap by bisection on qty; the spec's `max_factor_risk_contribution` — see the factor row below | catches correlated small-premium trades | same | M | ≤ 20 bisection steps × 600 | medium–high (thresholds) | P1 | IMPLEMENT AS SHADOW → PRODUCTION | §11: no silent production thresholds; shadow-log approve/resize for ≥ 20 trading days |
| ES limit (portfolio ES / NAV) (§37) | none | `max_portfolio_es_pct_nav` (research default) | absolute downside budget | same | L | trivial | medium | P1 | IMPLEMENT AS SHADOW | same promotion path |
| Sizing v2 modifiers (§37) | Base × Signal × Vol(proxy) then min(single-name, heat, cash) | ES modifier, correlation modifier, model-health modifier composed into `budget_multiplier`; min over stress/ES/RC caps via extra clamps | coherent sizing | B+C outputs | M | trivial | medium | P1 | IMPLEMENT AS SHADOW | abs cap keeps hard-limit supremacy (`engine.py:375-378`) |
| Explainability (§47) | reason_codes + sentences; greek breach REJECT-only (`engine.py:556-560`) | `requested_quantity`, ordered `binding_constraints` (with layer), before/after deltas without/with resize; greek limits become RESIZE via clamp **only if Q5 is answered yes** | user trust | — | L–M | — | low | P1 | IMPLEMENT NOW (additive fields) / GATED ON Q5 (greek RESIZE) | additive fields on `RiskAssessment`; the greek change is the one non-byte-identical Tier 0 change and stays REJECT until the user decides |
| Risk-linked cash floor (§36) | regime table only (`engine.py:52-62`) | floor = max(regime floor, f(ES%, drawdown, model-risk state)) with research defaults | cash as risk asset | B outputs | L | trivial | medium | P2 | IMPLEMENT AS SHADOW | preserve regime floor as minimum |
| Liquidity gate — **underlying only** (Tier 0 gap) | SKIPPED for the underlying (`orders.py:712-715`); option-leg OI/spread filters already enforced per contract in CONTRACT_SELECTION (`spreads.py:51-52, 186-189`, `selector.py:15`) | frozen `LiquidityLimits` with **research defaults, documented unvalidated**: `min_adv20_shares` = 100 000, `max_order_pct_adv20` = 0.01, `max_quote_spread_pct` = 0.005 (stock NBBO); ADV20 from stored `stock_bars_daily.volume`, spread from the live quote; the pool LIQUIDITY readiness check reports the same numbers | hard limit per §2/§5 | stored bars, quotes | L | trivial | low (thresholds unvalidated) | P0 | IMPLEMENT NOW (B0) in **REPORT mode** — gate returns PASS with the measured ADV20 / spread / order-%-ADV in its detail and a `shadow.liquidity` verdict in RISK_DECISION; becomes a FAIL veto only after the Q3 shadow window and a review of which watchlist symbols it would have blocked | not statistical; the SKIP text is stale; the watchlist contains illiquid small caps (RDW, `DEVLOG.md:818-824`: legitimate §9 DTE/OI/spread rejections) so an un-shadowed hard veto with guessed defaults would silently block currently tradable symbols |
| Dynamic buckets enforced (Tier 0 gap) | display only (`portfolio.py:709-729`, names `DYNAMIC:A+B` at `:720`) | (B) compute dynamic membership from the **date-aligned** return matrix of the Phase B returns layer (today's `stored_closes_by_ticker` lists are unaligned, §4) and log the hypothetical `BUCKET_LIMIT_<name>` clamp in the RISK_DECISION `shadow` block; (C) enforce via `dataclasses.replace(RiskLimits(), correlation_buckets={**static, **dynamic})` in `orders.py` using the **same names the view emits** (`DYNAMIC:NVDA+AMD` → reason code `BUCKET_LIMIT_DYNAMIC:NVDA+AMD`, `engine.py:465` f-string) so gate text and bucket panel match verbatim; membership drifts daily, so no `DYNAMIC_1..n` numbering | closes a shown-but-not-enforced hole | aligned 60d returns | L | trivial | medium (0.70/60d unvalidated) | P0 (shadow) / P1 (enforce) | IMPLEMENT AS SHADOW (B) → PRODUCTION (C, after the Q3 window) | 0.70/60d are documented starting points “requiring validation” (`correlation.py:9-12`, `portfolio.py:127-130`); enforcing them un-shadowed would contradict §11/§70 and Q3; the UI text (`ui/app/risk/page.tsx:463-466`) must say “display only” until promotion |
| Income opens through Tier 0 (gap) | no assess/pool/audit (`income.py:155-296`) | `assess()` **cannot** take income opens as written: a covered call carries 0 incremental risk (`income.py:369-374`) so `stop_distance ≤ 0` raises `ValueError` (`engine.py:382-384`), and income rows have `entry_edge=0.0` (`income.py:376`) so `strength_tier` → None → `SIGNAL_TOO_WEAK` REJECT (`engine.py:363-374`). Seam (additive, `assess` byte-identical): new `assess_income(IncomeRiskRequest, snapshot, limits, portfolio_greeks=None, new_position_greeks=None) -> RiskAssessment` in `engine.py` that skips steps 3–4 (no edge tier, no stop-based sizing; qty = contracts requested, capped by the clamps) and runs kill switch (1), heat gate (2), single-name risk (5a), single-name capital (5b), buckets (5c), heat headroom (5d), cash floor (5e), greeks (5f) with: **CSP** risk basis per contract = (strike − credit)×100 (feeds heat/5a/5c/5d), capital & cash-floor basis per contract = strike×100 (the collateral actually reserved, `income.py:264-286`; the credit is not counted as usable cash), greeks = short put negated; **CC** risk basis = 0 (heat unchanged — the stock row already carries it), capital basis = 0 (shares already owned; the 100-share/contract collateral check stays in `income.py:173-200`), greeks = short call negated. Reason codes reused verbatim; plus Trading Pool authorization and one RISK_DECISION per open | closes a real bypass (§72) | — | M | — | low | P0 | IMPLEMENT NOW (B0) | "user must not accidentally bypass hard limits"; the seam is what makes B0 executable |
| Factor model / `max_factor_risk_contribution` (§11) | none; no factor or sector return data beyond SPY/QQQ | single-factor (SPY-β) risk contribution as a diagnostic; sector RC approximated by the STATIC bucket | marginal on a ≤ 8-name book | SPY stored | L | trivial | medium | P3 | RESEARCH (single-factor diagnostic in C at zero cost) / REJECT (multi-factor model) | no factor data source; §62 Q1 fails for a multi-factor model on this book; static bucket + dynamic correlation cover the sector question |

### 7.4 Stress and options (Phase D)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Historical stress (§25) | none | replay named windows' return paths (2024-08-05, 2025-04 available; 2020-03/2022 after deep backfill) on current book; worst-k-day windows auto-found | "what if it happened now" | stored bars (deep backfill) | M | trivial | medium (window choice) | P1 | IMPLEMENT NOW | needs data step first |
| Hypothetical stress (§26) | none | configurable scenario table (SPY −5/−10%, tech β-scaled, corr→0.9, IV ±%) | policy stress | — | M | trivial | medium (arbitrary numbers) | P1 | IMPLEMENT NOW | research grid, documented as unvalidated until IV history exists |
| Stress loss gate (§27) | none | `max_stress_loss_pct_nav`; veto authority | tail policy | D outputs | L | trivial | medium | P1 | IMPLEMENT AS SHADOW → PRODUCTION | promote after shadow review |
| Option full revaluation (§22) | greeks linear only | IV solver (bisection), baseline bundle per leg, basis-adjusted BS reprice under (S1, IV1, t1), leg-aware for spreads/income | nonlinear tail; long-gamma/vega correctness | current IV from chain, r param | M | trivial | medium (European BS, r/q assumptions) | P1 | IMPLEMENT NOW | pricer already exists; the platform's own implementation per §22 |
| IV shock scenarios (§24) | none | grid (e.g. ±20/40%, crush −40%) + realized-vol-jump-derived shocks; empirical IV calibration once `atm_iv_daily` has ≥ 120 obs | vol-crush exposure | none today | L | trivial | high (uncalibrated) | P1 | IMPLEMENT NOW (grid) / DEFER (empirical) | honest labelling; §24 forbids arbitrary numbers as truth |
| Stock scenario P&L | none | qty × S × r_scenario (signed for shorts) | the stock leg of every historical/hypothetical scenario and the S1 input to option revaluation | stored bars | L | trivial | low | P1 | IMPLEMENT NOW | prerequisite for the whole stress tier; hand-computable test |

### 7.5 Tier 2 — conditional risk (Phase E)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| EWMA conditional volatility (λ=0.94) | none | RiskMetrics-style σ_t; vol-scaled (filtered) historical VaR/ES | volatility clustering with 2 lines of code | 600 obs | L | trivial | low | P2 → pull into B/C | IMPLEMENT NOW (metric) | Q6: simpler model likely ≈ GARCH(1,1) value on this data; its use as the §14 vol-targeting forecast follows the SHADOW path of the “Vol targeting v2” row |
| GARCH(1,1) (§12-§13) | none | MLE via Nelder-Mead (stdlib), Gaussian innovations first, Student-t optional; diagnostics: convergence, α+β<1, Ljung-Box on std. residuals², training window; MODEL_DEGRADED fallback → EWMA | responsiveness | 600 obs (marginal but workable) | M–H | ~600 × few hundred iterations, fine | high (fit instability, small sample) | P2 | IMPLEMENT AS RESEARCH MODE → SHADOW | compare to EWMA per §63; promote only if exceedance backtest beats EWMA |
| Conditional VaR/ES | none | from EWMA/GARCH σ_t × standardized-residual quantiles; 1D native, 5D/10D as labelled √h scaling | vol-scaled tail estimate; ensemble member | same | L | trivial | medium | P2 | IMPLEMENT NOW (EWMA) / RESEARCH (GARCH) | Q1 yes when clustering is present (test via Ljung-Box on r²); enters the ensemble dispersion, never gates alone |
| VaR backtesting (§42) | none | walk-forward exceedance rate, Kupiec POF, Christoffersen independence, ES severity, vol forecast error; per model | validation gate for promotion | ~600 stored obs − 250-day rolling estimation window ≈ **350 walk-forward forecast days available today** for stock-only reconstructed books (≈ 17 expected 95% exceedances, ≈ 3–4 at 99%) | M | trivial | low | P0 (framework **and first results in B** for hist/gauss/EWMA on stock-only books) / P2 (options-inclusive results after D; GARCH in E) | IMPLEMENT NOW | required for SHADOW→PRODUCTION; §42/§43 results are producible now, not deferred to E |
| Vol targeting v2 (§14) | crude proxy (`portfolio.py:475-558`) feeding `budget_multiplier` (`orders.py:885-957`) | forecast = σ_p (EWMA) of the book incl. candidate; multiplier clamps unchanged (`allocation.py:56-75`) | fixes empty-book 1.0 and negative-weight cases | B outputs | L | trivial | low (but changes production quantities) | P1 | IMPLEMENT AS SHADOW (B: both multipliers logged side by side in RISK_DECISION `shadow.budget_multiplier_v2`) → PRODUCTION swap at the `portfolio.py:475` seam in C after the Q3 window | it directly changes the production `budget_multiplier` and hence approved quantities — the same promotion path as every other sizing change; §70 |

### 7.6 Tier 3 — advanced tail / dependence (Phases G/H)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| EVT / POT / GPD (§16-§17) | none | GPD fit on exceedances over threshold, MRL plot data, parameter stability | rare-loss estimate | ~600 obs → ~30 exceedances at 95% threshold: unstable | M–H | trivial | very high | P3 | DEFER (revisit at ≥ 1500 obs post deep-backfill; research metric only) | Q2/Q3 fail; would produce false precision |
| Lower-tail dependence (§35) | none | empirical joint-tail concordance (fraction of joint worst-10% days) per pair | crash co-movement | 600 obs | L | trivial | medium | P3 | IMPLEMENT AS RESEARCH MODE (display) | trivial and informative; not a gate |
| Copula / Student-t / GARCH-copula (§20) | none | marginal GARCH → PIT → copula → MC | tail-dependent portfolio ES | insufficient; book of 2–8 names | H | MC fine in stdlib but pointless | very high | P3 | REJECT for now (revisit only if book > 15 names AND ≥ 1500 obs) | Q1 marginal vs stress + RC; Q3 unvalidatable |
| Minimum tail dependence (§5 Tier 3 list) | none | allocation objective minimizing pairwise lower-tail dependence | allocator variant | copula-grade tail estimates | H | — | very high | P3 | REJECT for now (with copula; no production allocator) | depends on the rejected copula/tail-dependence estimator and on an optimizer the platform does not run; the empirical lower-tail concordance display (row above) covers the informative part |
| Monte Carlo VaR | none | — | — | — | M | — | high | P3 | DEFER | historical + stress cover the need |

### 7.7 Allocation (Phase F — research/benchmark only)

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Markowitz MVO / tangency / frontier (§28) | none | — | benchmark at best | expected returns unstable | M | small solve | very high | P3 | REJECT | spec itself warns; entries are signal-driven single names, not rebalanced weights |
| Global minimum variance (§29) | none | long-only GMV via projected gradient over Σ; cash included; **walk-forward weights harness** (weights persisted per rebalance date over the reconstructed book, compared to the signal-driven book's realized vol/ES/drawdown) | benchmark | Σ + weights history | M | trivial (n ≤ 30) | medium | P3 | DEFER (precondition: the walk-forward comparison harness above; a point-in-time report endpoint is not enough) | spec Phase F says “Compare … Backtest first”; a snapshot endpoint without weights history/replay cannot make that comparison and affects no decision (§62 Q4) — revisit when the book exceeds ~10 names |
| Robust covariance / shrinkage (§30) | none | Ledoit-Wolf constant-correlation shrinkage; sample vs shrunk comparison | stability of RC/σ_p when n approaches T | Σ | L | trivial | low | P2 | IMPLEMENT AS RESEARCH MODE (estimator option, default sample; report the sample-vs-shrunk RC delta as a diagnostic) | with n ≤ 8 names and T ≈ 600 the sample Σ is well conditioned — shrinkage helps when n is comparable to T, not here; §62 Q6 says sample-only until the book is larger, so it stays a research option |
| Turnover stability (§31) | none | weight turnover, allocation change | — | needs weights history | L | trivial | low | P3 | DEFER | no optimizer in production; nothing to measure |
| ERC (§32) | none | iterative ERC weights over Σ, in the same walk-forward harness as GMV | benchmark | Σ + weights history | M | trivial | medium | P3 | DEFER (same precondition as GMV) | benchmark alongside GMV; no comparison without the harness |
| ES budgeting (§33) | none | ES-contribution constraint (= RC gate in 7.3); full ES-budget allocation | — | — | M | trivial | medium | P1 (constraint) / P3 (allocator) | IMPLEMENT NOW (constraint) / DEFER (allocator) | "signal determines candidate, allocation determines size, ES contribution constrains concentration" |

### 7.8 Governance, infrastructure, UI

| Capability | Current | Missing | Model value | Data | Cost | Comp | Model risk | Prio | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Model versioning (§44) | `trade_plans.versions` (config only), no risk-model identity | model_name/version/params/data_window/data_source/as_of/confidence/horizon/distribution/diagnostics on every run | reproducibility | — | L | — | low | P0 | IMPLEMENT NOW | part of `risk_model_runs` |
| Snapshot as_of + TTL (§55) | ad hoc: bar age ≤ 5d, chain 20 s, quotes 60 s | per-metric TTL (greeks seconds, daily models 1 trading day, stress daily); stale critical snapshot → statistical layer UNAVAILABLE, hard limits still run. §58 “if critical data itself is invalid: PAUSE TRADING” — mapped in two layers, **deliberately deviating** from a per-request-only reading: (a) single-candidate invalidity (missing/stale bars, unreadable broker cash) stays the existing per-trade DATA_QUALITY / RISK_APPROVAL fail-closed veto (`orders.py:492-524, 817-865`) — no order reaches the broker; (b) book-level critical-data invalidity (broker unreadable across a snapshot build, or stale bars for the majority of open positions) sets `risk_snapshots.data_quality = INVALID`, which blocks ALL new opens for that snapshot and is proposed to trip the kill switch through the existing reconciliation-style path (`broker.py:527-549` KILL_SWITCH_TRIGGERED precedent, `DEVLOG.md:1596-1601`), closes still allowed. Automatic pausing is a policy change → open question 7 | §58 | — | L | — | low | P0 | IMPLEMENT NOW (a) / GATED ON Q7 (b) | property test "stale snapshot cannot authorize new risk"; a per-trade FAIL is not a pause, so the deviation is stated rather than hidden |
| Fast/slow separation (§54) | everything inline in request | background snapshot writer (monitor-loop pattern `main.py:117-135`); pre-trade reads latest snapshot + computes only the incremental part | latency | — | M | — | low | P1 | IMPLEMENT NOW (light) | keep single-process assumption (ADR-007) |
| Shadow mode (§70) | none | per-model mode flag; hypothetical decision logged in RISK_DECISION details, never alters qty | safe rollout | — | L | — | low | P0 | IMPLEMENT NOW | mandatory path for every new gate |
| Observability (§65) | http/uptime/bar-age/chain-age gauges only (`main.py:66-95`) | `risk_snapshot_age`, `risk_model_latency`, `var_exceedances`, `es_exceedances` (ES-severity breaches: realized loss beyond forecast ES on VaR-exceedance days), `risk_resize_count`, `risk_reject_count`, `stress_limit_blocks`, `model_health_state`, `garch_fit_failures` | ops | — | L | — | low | P1 | IMPLEMENT NOW (with each phase) | REGISTRY pattern exists |
| Audit trail extension (§66) | decision/mode/execution_authorized/veto_gate/gates/reason_codes (`orders.py:1018-1036`) | + snapshot_id, model_run_ids, limits in force, budget_multiplier, requested/approved qty, binding constraints + layer, model health, shadow decisions | reproducibility | — | L | — | low | P0 | IMPLEMENT NOW | additive JSONB keys, no schema change (`alerts.py` predicate depends only on decision/veto_gate) |
| Backtest of risk models / parity (§42-§43, §64) | none; live risk gates not replayed in backtests | walk-forward forecast series from the same lib over reconstructed book P&L; risk-model library **and the book P&L-series construction** (`libs/trading_core/risk/pnl_series.py`, not the gateway builder) imported by replay | validation | — | M | — | low | P1 | IMPLEMENT NOW (VaR backtest harness) / DEFER (replaying `assess()` inside single-symbol engines — separate parity iteration) | replaying Tier 0 sizing changes every historical result; needs its own decision (open question 4) |
| UI (§48-§52) | `ui/app/risk/page.tsx` (inspection item 15): `StatTiles` `:173-272` (Portfolio NAV, Cash + floor, heat, Max New Risk, regime, Trading), `VolTargetingLine` `:273-281`, `GreeksPanel` `:296`, `BucketsPanel` `:456` (STATIC/DYNAMIC legend `:463-466` implies enforcement), `PositionsPanel` `:541`, `LimitsPanel` `:594` (“Hard limits”), `StrategyHealthPanel` `:666`, `RiskDecisionsPanel` `:769`, `ReconciliationPanel` `:856`; types `ui/lib/types.ts:585-598, 676-680`; formatting `ui/lib/risk-format.ts` — nothing statistical, no methodology labels | methodology-labelled tiles ("Historical VaR 99% 1D"), current-vs-after table in Trade Plan, RC bars, stress table, model-health panel, "How is this calculated?" modal (generalize `ScoreExplainerModal`), per-instrument position rows | decision quality | — | M | — | low | P0–P1 incremental | IMPLEMENT NOW incrementally per phase | do not wait for a trailing Phase I |

---

## 8. Technical debt relevant to the upgrade

Correctness (fix in Phase B0):
1. **`orders.side` CHECK constraint** `migrations/005_orders.sql:19-20` allows only BUY_TO_OPEN/SELL_TO_CLOSE; no later migration relaxes it (grep of `migrations/` for SELL_TO_OPEN/BUY_TO_CLOSE/DROP CONSTRAINT is empty), yet `income.py:350, 457` and `orders.py:2340, 2757` (the `Order(...)` rows at `:2336-2342` and `:2729-2760`) insert SELL_TO_OPEN, and `orders.py:1598` (`close_side`, Order row `:1599`) plus `income.py:636, 755` insert BUY_TO_CLOSE. On the Postgres deployment these inserts violate the CHECK; SQLite tests use ORM `create_all` (no CHECK) and cannot detect it. DEVLOG records "live toggle round-trip verified" for income, not a live income fill.
2. `docker-compose.yml:20-31` mounts migrations 001–012 only; 013–016 exist. A fresh volume would lack `recommendations.llm_model` and four `positions` columns (`create_all` cannot ALTER).
3. Income opens bypass `assess()`, Trading Pool and RISK_DECISION audit (`income.py:155-296`); CSP `cash_reserved` is not netted from `cash` in the view (`portfolio.py:583-593`) or the order-path snapshot (`orders.py:855-861`).
4. Dynamic correlation buckets computed for display (`portfolio.py:709-729`) but never passed to `assess()` (`orders.py:414` bare `RiskLimits()`). Fix path is SHADOW (B) → enforce (C), not B0 — see §7.3.
5. LIQUIDITY gate and pool LIQUIDITY check are placeholders with a stale Massive message (`orders.py:297, 712-715`; `trading_pool.py:55-58`) — for the **underlying** only; option-leg OI/spread liquidity is enforced per contract (`spreads.py:51-52, 186-189`, `selector.py:15`).
6. **BEAR_PUT_SPREAD backtest never enters**: `run_spread_backtest` sets `bear` at `backtest/options.py:427` but calls the BULL `_evaluate_entry` at `:643` and requires `short.strike > long.strike` at `:653`, while the router's put resolver returns the short strike BELOW the long (`routers/backtests.py:299-311`). `tests/test_backtests_api.py:391-416` passes vacuously on an empty trade list. (Not a risk-engine defect but a parity/validation defect that would poison any §63 comparison using that leg.)
7. `greeks.py:147-151` and `engine.py:214` use builtin `sum()`; `math.fsum` elsewhere — inconsistent precision convention.

Model-quality debt (addressed by Phases B–D):
8. Vol-targeting proxy: cash weighs 0 (mostly-cash book → multiplier near 1.2), empty book → exactly 1.0 regardless of candidate, negative market values from short/income rows produce negative weights (`portfolio.py:522-537`).
9. Options at book value in NAV/heat (`portfolio.py:158-198`) vs chain mid in `/api/positions` (`positions.py:412-421`) — two valuations of one book; `max_loss` fixed at open, never re-marked.
10. Greek limits REJECT rather than RESIZE (`engine.py:556-560`); no gamma limit; provider greek units assumed, never normalized (`alpaca.py:574-579`).
11. `RiskLimits` default-constructed everywhere, unversioned; `plans.py:62-69` versions omit risk-model/limit versions.
12. `entry_price` unvalidated in `assess()` (only `stop_distance` is checked, `engine.py:382-384`) — 0 raises ZeroDivisionError in `_floor_qty` at `:439/:494`; SHORT_STOCK treated as cash outlay in capital/cash-floor clamps although a short credits cash.
13. `RiskDecision.PAUSE_STRATEGY`/`EMERGENCY_EXIT` never emitted (`models/enums.py:152-157`).
14. `MAX_BAR_AGE_DAYS`/plan staleness disabled suite-wide by the autouse fixture (`tests/conftest.py:143-156`) — staleness tests must opt back in.

Infrastructure debt:
15. No migration runner/version table; migrations never executed in tests or CI; ORM↔SQL mirror unverified mechanically (`README.md:58-73`, `DEVLOG.md:1774-1780`).
16. Correctness rides on one process (in-process locks/caches: `orders.py:222-250`, `alpaca.py:112`, `options.py:93`); ADR-007.
17. Stale docstrings claiming capabilities do not exist that now do (`config.py:131-137`, `instrument.py:1-28`, `deps.py:64-65`, `orders.py:76-81, 202-205`, `provider.py:3-4`, `db.py:3-4` claims option chains live in Timescale — no such migration).
18. `stock_bars_1m` hypertable is dead schema; ADR-005 lists VIX as storable but it is unservable.
19. `README.md:49` says 726 tests (1033 collected, verified); local venv 3.13 vs CI/Docker 3.12; hypothesis absent.
20. UI: `PortfolioGreeks` typed non-null while backend can send null totals (`ui/lib/types.ts:585-598` vs `portfolio.py:763-773`); recent-decisions panel filters 100 unfiltered audit rows client-side and renders `VETOED` as UNKNOWN (`ui/app/risk/page.tsx:142-154, 978`).

---

## 9. House rules & mandates the upgrade must honor

Quoted from code/docs (file:line):
- "Every threshold is a parameter on `RiskLimits`, never a hardcoded truth" — `engine.py:11-12`; plan §6.2 / §44 rule 2.
- "risk limits always have PRIORITY over strategy confidence (plan §44 rule 20) — no confidence score may override a limit" — `engine.py:7-9`; "§14 never overrides hard caps" — `engine.py:17-19`, `allocation.py:9-14`.
- "The gateway records every risk decision as an audit event in the same transaction (house rule; plan §19)" — `engine.py:27-29`; ADR-003 `docs/ARCHITECTURE.md:24-29`; "Records exactly one SYSTEM RISK_DECISION audit event — veto or approval" — `orders.py:397`.
- Additive-only engine changes: "Optional inputs (all defaults leave behavior EXACTLY as before)" — `engine.py:300-301`; "471 prior tests byte-identical" — `DEVLOG.md:2158`.
- "SINGLE SOURCE OF TRUTH: the live §10 gate chain (routers/orders) and the backtest engine both import THIS constant … (user parity mandate 2026-08-16)" — `engine.py:46-49`; "backtest and live logic must be IDENTICAL" — `DEVLOG.md:469-503`.
- "IS/OOS design DELETED by user decision for the manual-tuning era; reintroduce only if/when ML-driven parameter search arrives" — `DEVLOG.md:471-476` (walk-forward for *risk-model validation* per spec §43 must be framed as validation, not parameter search). Spec §68 nevertheless lists “Out-of-Sample Tested” as a PRODUCTION acceptance criterion — the tension is raised as open question 6.
- Dependency-free numerics: "Pure stdlib, deterministic, no numpy/scipy (house rule)" — `libs/trading_core/options/bs.py:3`; `greeks.py:3`; `correlation.py:3`; `allocation.py:3`; `indicators.py:3`; `backtest/engine.py:3`; `libs/common/telemetry.py:16`; `pyproject.toml:6-19`.
- Honest nulls, no synthetic fallback: "NO SYNTHETIC FALLBACK, EVER (§44 rule 18)" — `alpaca.py:11-23`, `provider.py:9-13`; "never a guessed scale-up or -down" — `allocation.py:62-64`; correlation None on insufficient history — `correlation.py:66-68`; "the platform never fabricates a default-cash portfolio for display" — `portfolio.py:571-577`.
- "THE ACCOUNT IS THE BROKER'S — the platform stores NO copy of it"; usable capital = broker CASH, never buying_power — `portfolio.py:571-577`; `orders.py:359-386, 852-865`; `docs/execution-chains-roadmap.md:60-65`.
- Kill switch asymmetry: opens blocked when paused; closes/buybacks/covers always allowed — `orders.py:42-45, 1543-1544`; `income.py:116-127`; `positions.py:29-31`.
- "§18 no auto-correct: reconciliation mismatch pauses trading and reports; ledger alignment is the user's decision" — `broker.py:15-30`; `DEVLOG.md:1596-1601`.
- Paper-only broker by construction; base URL not runtime-configurable — `libs/broker/alpaca.py:10-29, 240-266`; `runtime_config.py:16-18`.
- Permissions: naked_short_call/put locked forever; other flags real, default False, runtime-config only; "a permission toggle unlocks ONLY when its ENTIRE chain exists and is tested" — `docs/execution-chains-roadmap.md:18-25, 89-95`; `config.py:20-23`. Spec §60 "no short stock / no unsupported margin" is superseded in the codebase by the 2026-08-17 mandate (defaults preserved False) — the risk upgrade must not broaden further (open question 1).
- Mirror rule: migrations own the production schema; ORM change ⇒ migration in the same commit ⇒ compose mount line — `README.md:58-73`; `docker-compose.yml:15-19`; hypertables raw-SQL only, not ORM-mapped — `db.py:1-8`, `ARCHITECTURE.md:41-46`.
- Backtests depend on the provider interface, never `AlpacaMarketDataProvider`; no cross-provider fallback; no new paid provider without approval — `docs/data-source-architecture.md:89-98, 122-128`; `DEVLOG.md:834-835`.
- Provenance categories never blur (ALPACA RAW / INTERNAL DETERMINISTIC / LLM); internally calculated IV must be labelled as such — `docs/data-source-architecture.md:14-24`; `prompts/data_source.md` §12.
- "NO TRADE is a valid output" — plan §44 rule 18; every risk read view writes no audit event — `portfolio.py:44-45`.
- "Approve ALWAYS re-runs the full chain server-side; previews are never trusted; no rejected ticker may produce an order" — `orders.py:38-40, 2261-2274`.
- UI: server-generated strings rendered verbatim (audit-exact), enum translation total tables, `*_pct` fields are fractions, no native dialogs — `ui/lib/i18n.tsx:3-11`, `ui/lib/risk-format.ts:132-139`, `ui/scripts/check-native-dialogs.sh`.
- DEVLOG entry after every loop, semantic not vague; §69 fields (Risk hypothesis, Existing capability discovered, Model assumptions, Data used, Validation, Model limitations, Production status RESEARCH/SHADOW/PRODUCTION/DEPRECATED) added to the existing "## YYYY-MM-DD (N) — Title" form (`DEVLOG.md:3`).

---

## 10. Recommended priority & adjusted phase plan

Deviations from the spec's phase boundaries and why:
- **Phase B0 (new)** — Tier 0 hardening precedes statistical work because §2/§72 make hard limits the foundation and several are currently bypassable/inconsistent (§8 items 1–3, 5; item 4 — dynamic buckets — is deliberately SHADOW-first in B because its thresholds are unvalidated). Small, mechanical, high value; nothing in B0 adds a new production veto or changes an approved quantity except by closing the income bypass.
- **Persistence, registry, snapshot type, model health, dispersion, VaR-backtest framework, shadow mode, audit widening move into Phase B** (spec leaves them phase-less in §4/§39-§44/§55-§56/§70). Every later phase depends on them; building them once avoids retrofits.
- **EWMA conditional vol is pulled forward from Phase E into B/C** as the vol-targeting forecast (§62 Q6: simplest model that likely captures most of GARCH's value); GARCH stays research in E.
- **UI ships incrementally with each phase** instead of a trailing Phase I — spec §48 says add only what improves decision quality; a metric without its methodology-labelled tile is not "delivered".
- **Phase F/G/H are research/report-only or deferred/rejected** per §7.6–§7.7; no production allocation change.
- **Backtest replay of `assess()`** (true §64 sizing parity) is split out as its own decision (open question 4) — it changes historical results for every leg and touches six single-position engines.
- **Deliverable shape note:** spec §73 B “Risk Model Matrix” uses columns Purpose / Status / Data / Assumptions / Strength / Weakness / Decision Usage, which differ from the §7 gap-matrix columns here; the end-of-loop deliverable re-cuts §7 into that shape (one row per model actually built), not this document.

Numbering: next migration = **017**. Proposed: `017_orders_side_vocabulary.sql` (relax CHECK to the four sides + compose mounts for 013–017), `018_risk_snapshots.sql` (risk_model_runs, risk_snapshots, risk_metrics, risk_contributions, atm_iv_daily), `019_stress_runs.sql` (Phase D). All ORM-mapped (sqlite-testable), plain tables (daily granularity, same argument as `migrations/007:12-13`), typed scalar columns + JSONB only for diagnostics/params.

### Phase B0 — Tier 0 hardening (first half of the next loop iteration)
- `migrations/017_orders_side_vocabulary.sql`; add 013–017 mounts to `docker-compose.yml`; apply by hand to the live volume (`DEVLOG.md:1779-1780` rule); add a test that the ORM side vocabulary equals the migration's CHECK list (string-level parity test).
- `income.py` opens: build `PortfolioSnapshot` via the shared helpers (`orders.py:849-880` extracted into `apps/gateway/risk_inputs.py`), Trading Pool authorization check, then the **new additive** `engine.assess_income(IncomeRiskRequest(ticker, contracts, risk_per_contract, capital_per_contract), snapshot, limits, portfolio_greeks, new_position_greeks)` — `assess()` untouched and byte-identical; `assess_income` skips the edge tier and stop-based sizing (which is why `assess()` cannot be reused: `stop_distance ≤ 0` raises `engine.py:382-384`; `entry_edge=0.0` → `SIGNAL_TOO_WEAK` `engine.py:363-374`) and runs kill switch, heat gate, 5a–5f with CSP risk basis (strike−credit)×100 and capital/cash-floor basis strike×100 per contract (the reserved collateral), CC risk basis 0 / capital basis 0 (stock row already in heat; share collateral check unchanged), short-leg greeks negated; one RISK_DECISION audit per open with the same reason codes. Net `cash_reserved` out of usable cash in both the view (`portfolio.py:583-593`) and the order-path snapshot (`orders.py:855-861`). Tests: engine-level cases for both instruments (heat, capital, cash floor, greeks, kill switch) + API cases in `tests/test_income_api.py`.
- Dynamic buckets: **not in B0** — SHADOW in Phase B (hypothetical `BUCKET_LIMIT_DYNAMIC:A+B` clamp logged in the RISK_DECISION `shadow` block, computed from the aligned return matrix), enforced in Phase C via `dataclasses.replace(RiskLimits(), correlation_buckets={**static, **dynamic})` with the view's own bucket names (`portfolio.py:720`) after the Q3 window; test then mirrors `tests/test_portfolio_greeks_api.py:287-321` asserting the clamp.
- LIQUIDITY gate (underlying): frozen `LiquidityLimits` (research defaults `min_adv20_shares=100_000`, `max_order_pct_adv20=0.01`, `max_quote_spread_pct=0.005`, documented unvalidated) evaluated against stored volume and the live stock quote; gate emits PASS with measured values in the detail + a `shadow.liquidity` verdict in RISK_DECISION (REPORT mode — no new veto until promoted under Q3); replace `SKIP_NO_OPTION_DATA` for stock candidates; pool LIQUIDITY readiness check reports the same numbers; option-leg filters in CONTRACT_SELECTION unchanged.
- Fix BEAR_PUT_SPREAD replay entry (`backtest/options.py:643, 653`) + engine-level test.
- Tests: `tests/test_income_api.py` (+risk gate cases), `tests/test_risk_engine.py` (`assess_income` cases; existing 45 tests unchanged), `tests/test_liquidity_gate.py` (report-mode: never FAILs, values present, shadow verdict recorded).

### Phase B — Core risk metrics, SHADOW (second half of the next loop iteration; what will be built FIRST)
Order of construction:
1. `libs/trading_core/risk/returns.py` — `simple_returns`, `log_returns`, `align_closes(closes_by_ticker: Mapping[str, Sequence[tuple[date, float]]]) -> ReturnMatrix`, `ReturnSeries` dataclass with metadata (return_type, frequency="1D", lookback, as_of, data_source, model_version). `correlation.log_returns` and `realized_vol` re-point to it (behaviour byte-identical, tests `tests/test_correlation.py:42-63`, `tests/test_indicators.py:145-163` unchanged).
2. `libs/trading_core/risk/models/base.py` — `RiskModel` protocol (`calculate`, `validate`, `diagnostics`, `metadata`), `ModelResult` (value, confidence, horizon_days, sample_size, health, reason), `ModelHealth` enum, `ModelMode` RESEARCH/SHADOW/PRODUCTION; `models/__init__.py` registry.
3. `models/var_es.py` — `HistoricalVaRModel`, `HistoricalESModel`, `GaussianVaRModel`, `GaussianESModel` on a portfolio P&L series (95/99, 1D; √h-scaled 5D/10D labelled). `models/volatility.py` — sample and shrunk covariance, portfolio σ, EWMA σ_t (λ param). `models/diagnostics.py` — skew/kurtosis/JB → distribution label. `models/contribution.py` — component σ and Euler component ES (property: sums to total within 1e-9). `models/ensemble.py` — dispersion + MODEL_DISPERSION_HIGH. `risk/validation.py` — walk-forward exceedances, Kupiec POF, Christoffersen, ES severity (framework; results accrue).
4. `libs/trading_core/risk/snapshot.py` — `PortfolioRiskSnapshot` (typed per §45; `as_of`, TTL policy, `data_quality`, `model_health`, `risk_state`, `model_risk_state`) and `PositionRiskInput` (ticker, instrument, quantity, multiplier, spot, delta, mark, max_loss, return-series key). Tier 0 `PortfolioSnapshot` untouched.
5. `libs/trading_core/risk/pnl_series.py` — pure, stdlib **book P&L-series construction** `book_pnl_series(positions: Sequence[PositionRiskInput], returns: ReturnMatrix) -> PnLSeries` (stock qty×Δprice; options delta-linear ×100 labelled `DELTA_LINEAR` until Phase D; short/income signed) — lives in the library so historical replay imports the identical construction (§64 parity), then `apps/gateway/risk_snapshot.py` — builder shared by `/api/portfolio/risk` and the order path that only fetches inputs (aligned stored closes, positions, broker equity) and calls the library; background writer on the monitor-loop pattern (`main.py:117-135`) once per trading day after the bar refresh, plus on-demand; persists to `risk_snapshots` (incl. NAV = broker equity or simulator NAV)/`risk_model_runs`/`risk_metrics`/`risk_contributions`; `atm_iv_daily` row per open underlying; SHADOW computations logged here and in RISK_DECISION: dynamic-bucket hypothetical clamp, EWMA `budget_multiplier_v2` next to the production multiplier, liquidity verdict; first walk-forward exceedance results (≈ 350 forecast days, stock-only) written to `risk_model_runs.diagnostics`.
6. `migrations/018_risk_snapshots.sql` + ORM classes in `db.py` + compose mount.
7. `portfolio.py`: add `statistical` block (typed, methodology-labelled: model, confidence, horizon, lookback, sample_size, health) and `drawdown` (persisted NAV series + reconstructed-book drawdown, labelled). Never 503; honest nulls when history < minimum sample.
8. `orders.py:1018-1036`: RISK_DECISION details gain `snapshot_id`, `model_run_ids`, `limits` (asdict), `budget_multiplier`, `quantity_requested`, `approved_quantity`, `binding_constraints`, `shadow` block (hypothetical statistical verdict) — no decision change in B.
9. Observability counters/gauges (`risk_snapshot_age`, `risk_model_latency`, `risk_resize_count`, `risk_reject_count`).
10. UI (`ui/app/risk/page.tsx`): tiles "Historical VaR 95%/99% 1D", "Historical ES", "Gaussian VaR/ES" with sample size and health; RC table (capital weight vs risk weight); "How is this calculated?" modal generalized from `ScoreExplainerModal`; glossary keys.
11. Tests: `tests/test_risk_returns.py`, `tests/test_risk_models.py` (hand-computed VaR/ES/RC on a 10-point series; ES ≥ VaR; monotone in confidence; RC additivity; Gaussian formulas), `tests/test_risk_snapshot_api.py` (contract keys, honest nulls, no RISK_DECISION on read, staleness → statistical UNAVAILABLE while hard limits still decide), `tests/test_risk_validation.py` (Kupiec on synthetic exceedance streams).
12. DEVLOG entry with §69 fields; status: everything SHADOW/RESEARCH.

### Phase C — Pre-trade portfolio risk (SHADOW → PRODUCTION after review)
- `libs/trading_core/risk/pretrade.py`: `compare(current: PortfolioRiskSnapshot, candidate: PositionRiskInput, qty) -> RiskComparison` (heat, cash, VaR, ES, σ, stress placeholder, greeks, RC before/after; incremental and marginal ES); `statistical_caps(...) -> list[QuantityCap]` by bisection on qty for `max_portfolio_es_pct_nav`, `max_single_position_es_contribution`, `max_bucket_es_contribution` (research defaults in a new frozen `StatisticalLimits`).
- `assess()` extension (additive): `assess(..., extra_caps: Sequence[QuantityCap] = (), requested_quantity_out: bool = True)`; new optional `RiskAssessment` fields `requested_quantity`, `binding_constraints: list[BindingConstraint(code, layer)]`, `comparison: RiskComparison | None`. Greek limits switched to a clamp (RESIZE) **only if open question 5 is answered yes** — until then the contract-tested REJECT (`engine.py:556-560`) stays byte-identical; this item is gated, not scheduled.
- Correlation regime state (`correlation.py`: long/short/stress-conditioned matrices, Spearman as research display) feeding a correlation modifier in `budget_multiplier` composition (SHADOW); after the Q3 window and a side-by-side review of the logged multipliers, EWMA σ_p replaces the crude proxy at the `portfolio.py:475` seam and dynamic buckets are merged into `RiskLimits.correlation_buckets` with view-consistent names — both are explicit promotion steps, not silent swaps.
- Gate chain: statistical outcome recorded inside RISK_APPROVAL detail while SHADOW; when promoted, add `STATISTICAL_RISK` and `CONCENTRATION` names to `GATE_ORDER` in one coordinated change (tests + `ui/lib/types.ts`).
- UI: Trade Plan "CURRENT vs AFTER TRADE" table (`ui/app/watchlist/[ticker]/page.tsx` proposed-sizing panel), binding-constraints list, requested vs approved.
- Tests: incremental ES arithmetic, cap bisection monotonicity, property "shadow never changes approved_quantity", "failed statistical model never disables hard limits".

### Phase D — Stress engine + option full revaluation
- Data step: deep backfill mode in `analysis.py` (insert bars OLDER than stored oldest, audited DATA_BACKFILL, append-only), watchlist + SPY/QQQ; verify Alpaca depth live and record in `docs/data-source-architecture.md`.
- `libs/trading_core/options/iv.py` (bisection IV solver, labelled internally calculated), `options/reval.py` (leg-aware scenario reprice with basis adjustment; r/q parameters), `risk/models/stress.py` (historical windows incl. auto worst-k-day, hypothetical scenario table, IV grid), `migrations/019_stress_runs.sql`, `POST /api/risk/stress/run`, stress loss gate as `QuantityCap` (SHADOW → PRODUCTION), UI stress table and per-instrument position rows (premium at risk, DTE, vega $, scenario loss).
- Tests: stock and option scenario P&L hand-computed, IV solver round-trip vs `bs_price`, stress gate, "increasing a position cannot reduce its standalone max loss".

### Phase E — Conditional volatility
- `models/garch.py` GARCH(1,1) MLE (Nelder-Mead, stdlib) RESEARCH; diagnostics persisted; MODEL_DEGRADED fallback → EWMA → sample; §63 comparison via `risk/validation.py` exceedance backtest; UI model-health panel; DEVLOG statuses.

### Phases F/G/H — research endpoints only
- F: DEFERRED — GMV/ERC only make sense inside a walk-forward weights harness (weights persisted per rebalance date over the reconstructed book, compared to the signal-driven book); a point-in-time report endpoint would not satisfy “Compare … Backtest first”. Shrinkage stays a research estimator option in `models/volatility.py` (default sample). No production change. G: EVT deferred until ≥ 1500 obs. H: lower-tail concordance display; copula rejected for now.

### Phase J — adversarial validation (continuous, in `tests/test_risk_adversarial.py`)
Fat-tail crash series, vol spike, correlation convergence, IV crush, long gamma/vega, tech concentration, GARCH failure, stale snapshot, EVT insufficient data, broker/data mismatch → assert hard limits unchanged, statistical layer degrades to UNAVAILABLE, no order reaches the broker.

---

## 11. Open questions for the user

1. **§60 vs execution-chains mandate.** Spec §60 says preserve "no short stock / no unsupported margin". The codebase (2026-08-17 mandate) made `allow_short_stock`/`allow_margin` real toggles, default False. Confirm the intended policy is "defaults False, user-toggleable, naked shorts locked forever" so it can be recorded as an ADR; the risk upgrade will not broaden anything either way.
2. **Deep historical backfill.** Approve extending stored daily history for watchlist + SPY/QQQ back to ≥ 2018 via the existing Alpaca provider (no new cost; more storage/API calls) so 2020-03 and 2022 stress windows and longer VaR samples become available. VIX/SPX remain unavailable — accept VIX-free stress design, or approve investigating a free index-history source (a new provider integration, subject to the no-new-provider rule).
3. **Promotion policy for statistical gates.** Proposed: every new gate runs SHADOW for ≥ 20 trading days with logged hypothetical decisions, then a human flips a runtime-config key `risk_statistical_mode=SHADOW|PRODUCTION`. Confirm the shadow duration and that promotion is an explicit USER action.
4. **Backtest parity for sizing.** Should the live `assess()` (and later the statistical gates) be replayed inside the backtest engines? It restores §64 parity but changes every historical backtest result and requires a portfolio-state abstraction in six single-position engines. Recommend a dedicated iteration after Phase C; needs your go/no-go.
5. **Greek limits REJECT → RESIZE.** This is the one proposed change to prior Tier 0 behaviour (breach currently rejects outright, `engine.py:556-560`; every other cap resizes). Confirm it may change (two existing tests updated) or must stay byte-identical. The Phase C item is gated on this answer.
6. **§68 “Out-of-Sample Tested” vs the deleted IS/OOS design.** Spec §68 makes out-of-sample testing a PRODUCTION acceptance criterion; the codebase mandate deleted IS/OOS for the manual-tuning era (`DEVLOG.md:471-476`). Proposed reading: for *risk models* the walk-forward exceedance backtest (§42/§43 — estimate on a rolling window, score on the next day, never re-fit to the score) *is* the out-of-sample test and is validation, not parameter search, so it does not reintroduce IS/OOS parameter splits. Confirm this reading, since it decides what “PRODUCTION” means for promotion (Q3).
7. **§58 automatic pause on invalid critical data.** Per-candidate data invalidity already fails closed per trade. Should *book-level* critical-data invalidity (broker unreadable during a snapshot build, or stale bars for most open positions) automatically trip the kill switch through the existing reconciliation-style path (`broker.py:527-549`), or only mark the snapshot INVALID (blocking new opens) and alert? The audit proposes the former as a SHADOW-logged rule first; needs your policy.
