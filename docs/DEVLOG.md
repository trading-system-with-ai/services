# Development Log — Backend

Newest entries first. Each loop iteration appends one entry: what was built, key
decisions, test/audit status, and what's next.

---

## 2026-08-10 — Iteration 4: Backtest Engine V1 + backtest API + watchlist overview (Phase 3)

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
