# Development Log — Backend

Newest entries first. Each loop iteration appends one entry: what was built, key
decisions, test/audit status, and what's next.

---

## 2026-08-10 — Iteration 10: Options in the decision chain — vol regime, §8 matrix, option execution & exits

**Built (pure libs → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/trading_core/volatility.py` — §7 vol regime v0 (LOW/NORMAL/HIGH/
  EXTREME from ATM IV level + IV/RV ratio; provisional until IV history
  enables IV Rank; every threshold a parameter).
- `libs/trading_core/strategies/instrument.py` — the §8 Instrument Selection
  matrix, every cell implemented + documented with §5 degradations (spreads
  unpermitted → stock/single-leg/no-trade): BULL STRONG+LOW → LONG_CALL,
  spread cells degrade, BEAR WEAK → NO_TRADE, EXTREME never buys premium,
  NEUTRAL → NO_TRADE. AccountPermissions configurable. `strength_tier`
  refactored public in risk engine (single source of truth for edge→tier).
- `exits/engine.py` — option exits (§11.3/§11.7): PREMIUM_HARD_STOP (-45%
  research parameter) > DTE_EXIT (≤21) > underlying-driven rules via shared
  internals (bit-identical to stock evaluation, §21). Underlying HARD_STOP
  replaced by the premium stop for options; missing mid reported loudly.
- Gateway: VOLATILITY gate is now a real classification; INSTRUMENT gate is
  the matrix verdict with rationale; CONTRACT_SELECTION proposes the §9
  top-ranked contract; risk sizing for options passes entry=stop=mid×100 so
  approved_quantity counts CONTRACTS with every cap intact (§12.1). Approve/
  close handle ×100 multiplier + per-contract commission ($0.65); close
  regenerates the chain to find the same contract, intrinsic-value fallback
  if expired. Positions evaluate option rows via evaluate_option_exit.
  `migrations/008_option_execution.sql`.

**Verified:** 471/471 green (319 → +152). Exhaustive §8 sweep (1,200
permission-expanded cells): always §5-legal, EXTREME never buys premium,
rationale always present. 5,000-trial option sizing fuzz: contracts×premium
×100 never exceeded the absolute cap nor tier budget; exact boundary check
(100k NAV, STRONG, $250/contract) = exactly 4 contracts. Premium-stop/DTE
boundary arithmetic bit-exact. Live option lifecycle cash-conserved to the
cent.

**Next (iteration 11):**
1. Correlation buckets from returns (§12.4 rolling correlation grouping to
   replace/augment the static TECH_MEGA list) + delta-adjusted exposure and
  portfolio Greeks aggregation (§16) on the Risk page.
2. Volatility targeting layer (§14) as an allocation modifier (capped 1.2x).
3. Replay-style integration test: multi-day loop advancing stub data
   (backfill → signal → preview → approve → monitor → exit) as one test.

**Built (bs+selector libs → chain+API chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/options/bs.py` — pure-stdlib Black-Scholes-Merton
  (math.erf normal CDF): price + Greeks with documented conventions (theta per
  calendar day, vega per IV point, signed delta); intrinsic-value expiry edge.
- `libs/trading_core/contracts/selector.py` — Contract Selector v0 (§9):
  side gate (BULL→calls, BEAR→puts, long-only §5), §9.1 filters (DTE 30–90,
  |Δ| 0.40–0.75, OI/volume/spread/theta-burden — every threshold a parameter,
  every failure a numeric reason), §9.2 v0 heuristic ranking
  (liquidity − theta burden + delta fit, components exposed; Phase 10 upgrades
  to EV-based). All contracts returned with verdicts for the §34 view toggles.
- Stub option chain in libs/market_data: deterministic (crc32-seeded) — weekly
  + monthly expiries, tiered strike grid ±25%, seeded IV smile + term
  structure, BS theoretical mids, moneyness/DTE-dependent spreads, ATM-decaying
  volume/OI. Same-IV-both-rights documented as v0 (no skew yet).
- `GET /api/watchlist/{ticker}/options?direction=AUTO|BULL|BEAR` (§34):
  AUTO resolves via score_direction; NEUTRAL → no candidates ("NO TRADE is a
  valid output"); summary with ATM IV, straddle expected move, RV20, IV−RV
  spread, and iv_rank **honestly null** until real IV history exists.
- Compose: frontend service added (context ../ui, :3000), image built OK.

**Verified:** 319/319 green (259 → +60). Independent quant audit: put-call
parity worst error 2.27e-13 over a 500-point grid; bs_price vs an independent
Simpson risk-neutral integrator agrees to 1.56e-8; finite-difference delta
check passed; no crossed markets; every live API candidate re-satisfies the
§9.1 filters from its own row values.

**Next (iteration 10):**
1. Instrument Selection matrix (§8): direction×strength×IV-regime →
   LONG_STOCK/LONG_CALL/LONG_PUT/NO_TRADE in trading_core; wire into the §10
   INSTRUMENT gate + Trade Plan (show chosen instrument + §9 contract when
   options are selected).
2. Volatility regime classification (§7 LOW/NORMAL/HIGH/EXTREME from stub
   chain IV vs RV) feeding the matrix.
3. Order approve path for LONG_CALL/LONG_PUT paper fills (chain mid ± slippage)
   with per-contract max-loss = premium (§12.1 options sizing in risk engine).

**PHASE 0 ACCEPTANCE: PASS.** `docker compose up --build` boots the real stack
(TimescaleDB pg16 + Redis + gateway) and the smoke test ran end to end through
it: healthz/readyz, watchlist add, full analysis (migrations 001–007 +
Timescale storage + lazy backfill + signals), market overview, portfolio risk
(NAV 100k), audit trail. Stack torn down with `down -v`, nothing left running.

**Two real defects found and fixed by the E2E agent:**
1. docker-compose.yml mounted the whole migrations directory over
   `/docker-entrypoint-initdb.d`, shadowing the timescale image's own init
   scripts (`CREATE EXTENSION timescaledb`) — migration 002's
   `create_hypertable()` would have aborted initdb on a fresh volume. Fixed by
   mounting each migration file individually (rule documented in README).
2. `stock_bars_daily` existed only via ORM `create_all` — added
   `migrations/007_stock_bars_daily.sql` mirroring the ORM exactly, with the
   mirror-in-same-commit rule documented.

**Environmental note:** host port 8000 was occupied by an unrelated container
(roboxai-optimizer) — smoke test used a scratchpad-only override on :8010;
the repo compose keeps the canonical 8000:8000. Free the port to run locally.

**Also built:**
- GitHub Actions CI for both repos (services: py3.12 + pytest; ui: node22 +
  typecheck + build), YAML-validated; `make ci` target.
- `GET /api/watchlist/{ticker}/bars` (watchlist-gated OHLCV series, limit
  10–600) + OHLC-sanity tests.
- Audit filters: `GET /api/audit?action=&actor_type=` (AND semantics, typed
  422 on bad values) + `GET /api/audit/actions` distinct-values endpoint.

**Verified:** 259/259 green before and after Docker work.

**Next (iteration 9):**
1. Option-chain scaffolding (§34): stub chain provider (deterministic strikes/
   expiries/greeks around spot), `GET /api/watchlist/{ticker}/options`,
   Options tab with eligibility highlighting groundwork.
2. Contract Selector v0 (§9): candidate filters + risk-adjusted ranking over
   the stub chain (research-only until real chain data).
3. Docker compose entry for the UI container (frontend joins the stack).

**Built (llm+health libs → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/llm/` — provider abstraction mirroring libs/market_data:
  `RecommendationDraft` validates the §4.1 score schema; deterministic stub
  provider (day-seeded, exclusions honored, evidence timestamps strictly
  before as_of — §20.3 news-timestamp integrity); real Anthropic provider
  (written against the claude-api skill: Messages API structured outputs,
  malformed model output logged-and-skipped, never fires without a key).
  Default provider switched to "stub" — keyless-safe.
- Recommendations API: refresh (LLM-attributed audits, skips watchlisted/
  already-PENDING tickers, **performs zero watchlist/pool/order writes**),
  list by status, dismiss (USER), promote — THE only rec→watchlist path,
  implemented by refactoring watchlist insertion into a shared
  `add_ticker_to_watchlist` helper used by both POST /api/watchlist and
  promote so the paths cannot diverge; WATCHLIST_ADD audited USER with the
  rec id in the note. `migrations/006`. RECOMMENDATION_PROMOTED enum added.
- `libs/trading_core/health.py` — Strategy Health Monitor v0 (§19):
  win rate / profit factor / expectancy / drawdowns over closed-trade PnLs;
  status ladder INSUFFICIENT_DATA (judgement withheld below min sample) /
  HEALTHY / WARNING / PAUSE_RECOMMENDED with numeric explanations;
  `GET /api/health/strategy` read-only report (no pause automation yet).

**Verified:** 249/249 green. Authority-boundary audit (the point of Phase 8):
static — recommendations router's only insert is the Recommendation row,
libs/llm has zero DB references, promote/watchlist share one helper with
hardcoded USER attribution; live — refresh left watchlist/pool/positions
untouched, promote added exactly the approved ticker, double-promote 409'd,
promoted tickers excluded from later drafts. Health math independently
recomputed. UI verifier grep-confirmed no trade action exists on the page.

**Next (iteration 8 — hardening + Phase 0 completion):**
1. Docker Compose end-to-end check (build gateway image, full stack up,
   healthz through the compose network) — Phase 0 acceptance still unproven.
2. CI: GitHub Actions for both repos (pytest / typecheck+build).
3. Watchlist symbol page Price tab (candlestick/volume from stored bars) and
   Activity page action-type filter chips.
4. Begin §34 option-chain scaffolding if time allows (stub chain provider).

**Built (exits lib → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/exits/engine.py` — pure Exit Engine v0 (§11): evaluates
  ALL five rules every call (HARD_STOP → SIGNAL_FLIP → SIGNAL_DECAY →
  ATR_TRAIL → TIME_STOP, backtest-priority order) with numeric reasons; holds
  report "OK:"-prefixed reasons so the user always sees why a position is kept
  (§37). Signal rules degrade to "insufficient data" on short history but a
  data gap can NEVER disable the hard stop. Reuses score_direction (§21).
- Paper execution: `POST /api/orders/approve` re-runs the FULL §10 gate chain
  server-side (client previews never trusted); BUY_TO_OPEN /
  SELL_TO_CLOSE are the only sides (§5, DB CHECK constraint); idempotent
  client_order_id (§42); 409 no-pyramiding; fills at last close ± slippage +
  commission (same model as backtest for comparability); ORDER_REQUESTED →
  ORDER_SUBMITTED → ORDER_FILLED + RISK_DECISION audited in one transaction.
- `POST /api/orders/close`: partial/full, realized-PnL arithmetic, allowed
  while trading is paused (closing reduces risk — §18 risk-priority).
- `GET /api/positions` (§37 contract: stop/trail/edge-decay/time-stop
  countdown/exit status + full reasons) and `POST /api/positions/check-exits`:
  mechanical exits audit EXIT_GENERATED and execute SYSTEM sell-to-close,
  unblocked by the kill switch. `migrations/005_orders.sql`.

**Race conditions found & fixed by the adversarial verifier (live-reproduced):**
concurrent same-client_order_id approves returned [200, 500] via UNIQUE
IntegrityError, and concurrent different-key approves double-filled into two
positions with a double cash decrement. Fixed with a shared per-event-loop
execution lock serializing approve / close / check-exits; two regression
tests added. Cash conservation verified to the cent (partial + full closes).

**Verified:** 202/202 green. Live lifecycle: preview → approve → position
(with hold reasons) → check-exits → forced HARD_STOP exit → cash credited;
watchlist-only approve rejected with zero Order rows; SELL_TO_OPEN absent
from the codebase.

**Next (iteration 7 — Phase 8 + hardening):**
1. LLM Recommendation Pool (§4.1, §30): provider-abstracted llm service (stub
   provider first), recommendations API (PENDING/DISMISSED/PROMOTED lifecycle,
   LLM actor audited, zero execution authority), news-free v0 using
   watchlist-adjacent discovery heuristics as stub input.
2. UI Recommendations page (§30 cards: no Trade Now action; View Evidence /
   Dismiss / Add to Watchlist which routes through the normal USER watchlist
   API).
3. Strategy Health Monitor v0 (§19): rolling stats over closed paper trades.

**Built (risk lib → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/risk/engine.py` — pure, strategy-independent (§17):
  `assess(request, snapshot, limits)` pipeline in spec order: kill switch (§18)
  → heat reject gate (§12.5) → |edge|→strength tier→risk budget hard-capped by
  abs_max_trade_risk (§12.2 "no confidence may override") → base sizing
  floor(nav·budget/stop) (§12.1) → quantity clamps for single-name risk/capital
  (§12.3), correlation bucket (§12.4), strict heat headroom, regime cash floor
  (§13) → APPROVE / APPROVE_WITH_RESIZE / REJECT with machine reason codes +
  §36-style numeric explanations. `portfolio_heat`/`heat_state` helpers
  (NORMAL/ELEVATED/HIGH/BLOCKED at 4/6/8%). All limits in frozen `RiskLimits`.
- 33-test suite incl. hand-computed binders for every cap, regime-dependent
  cash-floor flip (same request: APPROVE in STRONG_BULL → REJECT in
  STRONG_BEAR), and a 200-case seeded property test of the §42 invariants.
- Portfolio singleton (paper cash 100k configurable) + Position ORM +
  `migrations/004_portfolio.sql`; `GET /api/portfolio/risk` (§36 contract:
  NAV/cash/floor/heat/max-new-risk/buckets/limits, honest nulls for bar-less
  positions; read-only, no audit).
- `POST /api/orders/preview` — the §10 gate chain in exact order (pool
  authorization incl. per-symbol + global kill switch → data quality → regime
  (TRANSITION/bear veto for long stock) → directional signal → volatility/
  liquidity SKIPPED with explicit V1 details → instrument → contract-selection
  SKIPPED → risk approval via assess() with stop = 2.0·ATR14). First FAIL
  skips the rest; exactly ONE SYSTEM RISK_DECISION audit event per preview,
  veto or not (§38). why_trade / why_not_trade always both present (§33).

**Verified:** 164/164 green. Independent fuzz (600 seeded cases): approved
risk never exceeded the 1.5% NAV absolute cap nor the tier budget; heat_after
strictly < 8% on every approval; regime cash floors respected; kill switch
always wins; every REJECT carries reason codes. Live boot walked the
watchlist-only → veto, authorized → full chain, paused → gate-1 veto paths.

**Next (iteration 6 — Phase 6 paper execution start):**
1. Order state machine: `POST /api/orders/approve` (from a preview) → paper
   fill at last close, position open/close, cash movement, duplicate-order
   guard; ORDER_* audit chain.
2. Position monitor + Exit Engine v0 wiring for open paper positions
   (signal-decay / ATR-trail / time-stop checks over stored bars, §11).
3. UI Positions page v1 (§37) + order approve flow from Trade Plan.

**Built (engine → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/backtest/engine.py` — pure replay engine, LONG STOCK only
  (option backtesting deferred until real chain data exists — no fabricated
  option prices). §20.3 semantics enforced: signals computed on `[:t+1]` slices
  via the SAME `classify_regime`/`score_direction` used live (§21); decision at
  close of t fills at open of t+1 with slippage bps + per-share commission (§44
  rule 11); exits in priority order SIGNAL_FLIP → SIGNAL_DECAY (§11.1) →
  ATR_TRAIL (§11.5) → TIME_STOP (§11.6) → END_OF_DATA; IS/OOS split with
  per-segment metrics (report-only, §44 rule 16); every division guarded —
  None, never NaN. All knobs in frozen validated `BacktestParams`.
- 26-test quant-integrity suite: the no-look-ahead property (closed trades
  bit-identical between 300-bar prefix and 400-bar runs), hand-computed fill
  arithmetic, costs monotonicity, NO-TRADE honesty, long-only invariants.
- `POST /api/backtests` (watchlist-gated, synchronous V1, params validated
  before any state change), `GET /api/backtests[/{id}]`; records persisted with
  USER BACKTEST_STARTED + SYSTEM BACKTEST_COMPLETED/FAILED audit events in one
  transaction. `migrations/003_backtests.sql`.
- `GET /api/watchlist/overview` — per-symbol price/regime/scores/bias +
  opportunity_status v0 mapping (§31) + latest backtest status.

**Bug found & fixed during review:** the gateway implementation agent spotted
that the engine's fill block never copied `pending_entry` into `entry_reason`
(every trade explained its exit but not its entry — §38 violation). Fixed with
`entry_reason = pending_entry` in the fill branch + a regression test asserting
every trade's entry_reason carries the edge number. Suite now 122 green.

**Verified:** independent verifier re-derived fills to the cent, recomputed
total-return/max-drawdown from the equity array (1e-6 agreement), re-ran the
no-look-ahead experiment with a shock series, and exercised the API live
(404/422 paths, audit pairs, overview status flip).

**Next (iteration 5 — Phase 4 start):**
1. Portfolio state: NAV, cash, positions tables; paper-fill plumbing groundwork.
2. Risk Engine v0 as an independent module (§17): position sizing from risk
   budget (§12.1-12.2), single-name cap, Portfolio Heat, cash floor by regime
   (§13), APPROVE/RESIZE/REJECT decisions with reason codes, audited.
3. `POST /api/orders/preview` returning the full gate-chain evaluation (§10).
4. UI Risk page v1: NAV/cash/heat/limits + latest risk decisions.

**Built (signals lib → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/trading_core/signals/regime.py` — Market Regime Engine v0 (§6.1):
  `classify_regime` with frozen `RegimeParams` (all thresholds backtest parameters).
  Rules ordered: insufficient history → TRANSITION (no-trade posture); ATR/close
  dislocation → TRANSITION; stacked SMAs + fast-slope → STRONG_BULL/BEAR;
  above/below both major SMAs → MILD_*; else NEUTRAL_RANGE. Full explainability
  features dict on every result.
- `libs/trading_core/signals/directional.py` — Directional Signal Engine v0 (§6.2):
  `score_direction` evaluating 8 mirrored bull/bear feature pairs (SMAs, MACD
  cross/zero, RSI continuation zones, pivot HH+HL/LH+LL structure) + optional
  volume expansion. Weighted parameterized scores 0–100, edge = bull − bear,
  bias by threshold; every component listed with numeric human-readable detail.
- Daily bars: `Bar` + `get_daily_bars` in the provider Protocol; StubProvider
  emits deterministic crc32-seeded weekend-skipping walks. `StockBarDaily` table.
- `GET /api/watchlist/{ticker}/analysis` — 404 off-watchlist (§4.2), lazy 600-bar
  backfill with SYSTEM `DATA_BACKFILL` audit (once only), indicators + regime +
  signal + 250-bar chart series in one contract.
- `/api/market/overview` regime now computed from SPY bars. ADR-005: SPY/QQQ/VIX
  are system reference symbols exempt from the watchlist-only data rule.

**Verified:** 85/85 tests green (was 51). Independent verifier booted the app:
contract keys, enum validity, single-backfill audit invariant, 404 path, and
monotonic-series sanity (uptrend → STRONG_BULL/BULL, downtrend → STRONG_BEAR/BEAR)
all confirmed live.

**Next (iteration 4 — Phase 3 start):**
1. Backtest engine v1: bar-by-bar replay over stored daily bars using the SAME
   signals lib (§21); Long Stock entries/exits from directional bias + regime
   gates; explicit fill model (next-bar open) + transaction costs; equity curve,
   drawdown, win rate, profit factor outputs.
2. `POST /api/backtests` + `GET /api/backtests/{id}` with stored results + audit.
3. UI Backtests page v1: config form + results (§35 metrics, IS/OOS split label).
4. Watchlist rows enriched with regime/scores/status from analysis cache.

**Built (via 3 parallel implementation agents + 2 adversarial verify agents):**
- `libs/trading_core/features/indicators.py` — pure, dependency-free, deterministic:
  SMA, EMA (SMA-seeded), RSI (Wilder), True Range/ATR (Wilder), MACD (12/26/9,
  parameterized), realized vol (close-to-close log returns, annualization param),
  pivot highs/lows (§6.3; final `window` bars always unconfirmed — no look-ahead,
  §20.3). All outputs input-length with None warmup padding; all periods parameters.
  34 new tests including hand-computed reference values with arithmetic in comments
  and a backtest/live parity check for pivot stability.
- `libs/market_data/` — provider abstraction (Quote + MarketDataProvider Protocol,
  name-based registry) with deterministic StubProvider (SPY/QQQ/VIX, minute-keyed
  wiggle) until the Massive integration lands. `GET /api/market/overview` returns
  provider/as_of/stale/market_regime (NEUTRAL_RANGE placeholder)/indices.
- Global kill switch (§18): persistent `system_state` singleton row (default:
  trading disabled), `GET /api/trading/status`, `POST /api/trading/pause` (reason
  required), `POST /api/trading/resume`; both mutations USER-attributed and audited
  (TRADING_PAUSED/TRADING_RESUMED) in the same transaction.
- `migrations/002_system_state_and_bars.sql` — system_state seed + `stock_bars_1m`
  Timescale hypertable (Phase 1 groundwork).

**Verified:** full suite 51/51 green (was 11); independent verify agent booted the
app and confirmed overview/status/pause flows + audit records via curl; indicator
spot checks passed. Zero fixes needed.

**Next (iteration 3):**
1. Historical OHLCV ingestion into stock_bars (stub-generated series for dev) and
   `GET /api/watchlist/{ticker}/analysis` computing indicators over stored bars.
2. Market Regime Engine v0 (§6.1) using SPY/QQQ features — replace the
   NEUTRAL_RANGE placeholder in /api/market/overview.
3. Directional signal engine v0 (§6.2): parameterized bull/bear scores over features.
4. UI: symbol analysis page skeleton (tabs per §33) showing computed indicators.

---

## 2026-08-10 — Iteration 1: Phase 0 skeleton + Watchlist/Trading Pool core

**Built:**
- Project skeleton: `pyproject.toml`, `libs/common` (config via pydantic-settings,
  structured JSON logging with secret redaction), `libs/trading_core/models` (domain
  enums: regimes, instruments, risk decisions, audit actions, actor types).
- `apps/gateway` FastAPI modular monolith:
  - `/healthz`, `/readyz` (DB-checked)
  - Watchlist API: `GET/POST /api/watchlist`, `DELETE /api/watchlist/{ticker}`
  - Trading Pool API: `GET/POST /api/trading-pool`, toggle `POST /{ticker}/trading`,
    `DELETE /{ticker}`
  - Audit API: `GET /api/audit` (filter by entity, read-only)
- `migrations/001_initial.sql` — Postgres/Timescale DDL with FK cascade
  (trading_pool → watchlist) and audit indexes.
- `docker-compose.yml` — timescaledb + redis + gateway; migrations auto-applied on
  first db boot. Gateway Dockerfile.
- Test suite: 11 tests, all passing. The important ones encode plan rules:
  - non-Watchlist symbol cannot enter Trading Pool (rule 6);
  - promotion starts with trading disabled (authorization ≠ order);
  - short strategies rejected by account constraints (rules 7/8);
  - every mutation audited with USER attribution (rule 12);
  - Watchlist removal cascades out of Trading Pool, cascade itself audited.

**Verified:** `pytest` 11/11 green; live smoke test via uvicorn+curl confirmed the
422 rejection path, disabled-by-default promotion, and audit trail contents.

**Decisions:** see ARCHITECTURE.md ADR-001…004 (modular monolith, repo naming,
transactional audit, write-path authorization).

**Deliberately deferred:** auth-service (single fixed `local-user` identity for now),
event bus, Massive market data adapter, kill-switch API, recommendations endpoints.

**Next (iteration 2):**
1. Massive market-data adapter interface + stub provider (so UI can show prices
   without a real key), `GET /api/market/overview`.
2. Feature engine start: SMA/RSI/ATR/MACD in `libs/trading_core/features` with
   deterministic unit tests (Phase 2 groundwork).
3. Kill switch API: `POST /api/trading/pause` + `/resume` with audit + UI banner wiring.
4. Historical OHLCV storage schema (Timescale hypertable migration).
