# Institutional Risk Engine Upgrade — Final Report

**Programme:** `prompts/risk_engine.md` (74 sections), Phases A–E executed.
**Deliverable:** spec §73 (A. Risk Architecture Review · B. Risk Model Matrix · C. Implementation Report · D. Deferred Models · E. Validation Report · F. UI Changes · G. Remaining Model Risk), plus a consolidated appendix of open decisions.
**Date:** 2026-08-18.
**Status vocabulary** (spec §69): PRODUCTION · SHADOW · RESEARCH · DEPRECATED.
**Authoritative sources:** `docs/DEVLOG.md` entries (14)–(19); `docs/risk-engine-audit.md` (Phase A); `docs/risk-engine-phase-b-design.md` §1–§9. All paths are relative to `services/` unless prefixed `ui/` or `prompts/`.

**Headline.** The platform gained a complete statistical, stress, concentration and model-health layer, persisted and displayed, in five phases. **Not one Tier 0 trading decision changed.** `assess()` is byte-identical to the pre-programme engine plus additive seams; `extra_caps` is never populated at any `assess()` call site in `apps/`. Every statistical verdict is hypothetical and logged. The programme also found and fixed eleven real defects (§C.5), one of which was a latent production failure that would have broken the first income short, margin short or cover fill.

**The central caveat, stated once and repeated in §E and §G:** the shadow window has produced **zero trading days of shadow decisions** as of this report. The first SCHEDULED snapshot was written 2026-08-18 14:59 UTC. Drawdown history, ATM-IV history and VaR-backtest history all begin now. Nothing in this report should be read as evidence that a threshold is correct — only that it is computed correctly, labelled honestly, and decides nothing.

---

## A. Risk Architecture Review

### A.1 What existed before the upgrade

From the Phase A audit (`docs/risk-engine-audit.md` §2–§4; DEVLOG (14)):

**Present and genuinely good.** A complete, tested **Tier 0 deterministic hard-limit engine** at `libs/trading_core/risk/engine.py`: kill switch → portfolio heat gate → signal-strength validity → per-trade tier budget hard-capped at 1.5 % NAV → stop-based base sizing → single-name risk cap → single-name capital cap → static correlation-bucket cap → heat headroom (strictly `<`) → regime cash floor → §16 portfolio greek limits (REJECT-only). A nine-gate live chain (`GATE_ORDER`, `apps/gateway/routers/orders.py`) matched the first three tiers of the §72 hierarchy in spirit.

**Estimation-like logic that existed** (audit §4) — none of it a risk model in the institutional sense:

| Existing estimator | Convention | Consumed by |
|---|---|---|
| Realized volatility RV20 | ddof=1 stdev of 20 close-to-close **log** returns × √252 | vol regime, vol-targeting proxy, watchlist |
| Rolling Pearson correlation (60d) | 60 daily log returns, `math.fsum` | dynamic buckets — **display only** |
| Dynamic correlation buckets | union-find at ρ > 0.70 strict | `/api/portfolio/risk` display; **never** `assess()` |
| Vol-targeting "forecast" | NAV-weighted arithmetic mean of per-position RV20; no covariance, no option delta; self-labelled "CRUDE v0 FORECAST PROXY" | `assess(budget_multiplier=…)` |
| IV regime | ATM IV level + IV/RV ratio bands | §8 matrix, tradeability |
| Black–Scholes pricer + greeks | European, `math.erf`, r=0.04, q=0 — **no IV solver** | stub provider |
| Backtest metrics | Sharpe/Sortino on simple equity returns; max drawdown on equity curve | backtest records |

**Absent entirely** (grep-verified across `libs/` and `apps/`): VaR, ES, covariance matrix, portfolio σ, GARCH/EWMA, EVT, copula, skew/kurtosis diagnostics, risk contribution, NAV drawdown, stress scenarios, Kupiec/Christoffersen, IV solver, Monte Carlo, risk persistence of any kind.

**Three Tier 0 correctness gaps, live-verified against the Postgres volume** (audit §1.2, §8):

1. `orders_side_check` (`migrations/005_orders.sql`) permitted only `BUY_TO_OPEN`/`SELL_TO_CLOSE` while the code inserts `SELL_TO_OPEN`/`BUY_TO_CLOSE` (income shorts, margin short stock, covers). The live `orders` table was empty, so the **first real such fill would have failed on INSERT** — invisible to the SQLite test harness.
2. Covered-call / CSP opens bypassed `assess()`, the Trading Pool gate and the RISK_DECISION audit entirely (`apps/gateway/routers/income.py`).
3. Dynamic correlation buckets were computed and displayed but never enforced.

Plus: `docker-compose.yml` mounted only migrations 001–012 (013–016 were hand-applied); the underlying (stock ADV / quote-spread) LIQUIDITY gate was permanently SKIPPED behind a stale "arrives with the Massive integration" string (option-**leg** liquidity *was* enforced per contract in the §9 filters); `BEAR_PUT_SPREAD` backtest replay never entered a trade.

**Data reality bounding all model ambition** (audit §5): ~600 daily bars per symbol (history begins ~2024-03); **no IV history; no VIX/SPX from any configured provider**; current-day option chain only, not persisted; real option daily bars from ~Feb 2024; **no NAV time series stored**. Historical VaR/ES at 99 % has ~6 tail observations. EVT and copulas cannot be validated. Empirical IV-shock calibration was impossible.

**House rule preserved:** stdlib only in `libs/`. Everything built in Phases B–E — sorted quantiles, sample covariance, EWMA, Nelder–Mead GARCH MLE, bisection IV solver, regularized incomplete gamma for χ² p-values, Kupiec LR — is stdlib. No numpy, no scipy, no pandas was added.

### A.2 The pipeline now

Tier 0 is unchanged. Four SHADOW layers were added around it, each computing a hypothetical verdict and logging it where the real decision is recorded.

```
TRADE REQUEST (ticker, qty, direction)
  │
  ▼ gate 1  TRADING_POOL_AUTHORIZATION   pool ∧ symbol enabled ∧ kill switch      [PRODUCTION]
  ▼ gate 2  DATA_QUALITY                 ≥200 bars, last bar ≤5 calendar days     [PRODUCTION]
  ▼ gate 3  REGIME  ─┐
  ▼ gate 4  SIGNAL   ├─ §8 matrix + AccountPermissions degradation ladder         [PRODUCTION]
  ▼ gate 5  VOL      │  (permission lives here, not as a standalone gate 0)
  ▼ gate 6  INSTRUMENT ┘
  ▼ gate 7  LIQUIDITY (underlying)  ADV20 / order-%-ADV / quote spread
  │            ══► measured, PASS-always, hypothetical verdict           ◄══ SHADOW (REPORT)
  ▼ gate 8  CONTRACT_SELECTION      §9 selector; option-LEG OI/spread filters      [PRODUCTION]
  ▼ gate 9  RISK_APPROVAL = assess() / assess_income()
  │            1 kill switch  2 heat gate  3 tier budget  4 base qty
  │            5a-5e clamps   5f greeks (REJECT-only)                              [PRODUCTION]
  │            ── extra_caps=() ALWAYS in apps/ ──
  │
  │  ┌─────────────────────────────────────────────────────────────────┐
  │  │  SHADOW LAYERS — computed at the Tier 0 approved quantity,      │
  │  │  never fed back into assess(); a raise here changes nothing.    │
  │  ├─────────────────────────────────────────────────────────────────┤
  │  │ STATISTICAL RISK   hist/gauss VaR·ES 95/99, σ_p, EWMA σ,        │
  │  │                    conditional (filtered-HS) VaR/ES,            │
  │  │                    incremental & marginal ES, before/after      │
  │  │ STRESS RISK        historical windows, auto-worst windows,      │
  │  │                    hypothetical grid, USER scenarios;           │
  │  │                    options by FULL_REVAL (BS + IV solver)       │
  │  │ CONCENTRATION      Euler ES shares: single-name, bucket;        │
  │  │                    correlation regime NORMAL/ELEVATED/CONVERGING│
  │  │ MODEL HEALTH       ACTIVE/DEGRADED/UNAVAILABLE/FAILED per model,│
  │  │                    ensemble dispersion, model-risk LOW/ELEV/HIGH│
  │  └─────────────────────────────────────────────────────────────────┘
  ▼
APPROVE / APPROVE_WITH_RESIZE / REJECT   ← Tier 0 ALONE decides this
  │
  ▼  audit RISK_DECISION details:
       decision, mode, execution_authorized, veto_gate, gates, reason_codes   [pre-existing]
     + quantity_requested, approved_quantity, budget_multiplier, limits       [added, B]
     + binding_constraints (code, layer)                                      [added, C]
     + shadow.liquidity {measured, verdict, at_approved_quantity}             [added, B0]
     + shadow.statistical {current-book headline}                             [added, B]
         .comparison {rows, tier0_rows}  .caps  .hypothetical  .limits
         .correlation_state                                                   [added, C]
         .stress {worst_before, worst_after, cap, hypothetical}               [added, D]
     + shadow.vol_targeting_ewma {ewma σ_p, multiplier_ewma}                  [added, C]
  ▼
broker (paper only) / simulated fill
```

**Where each layer logs its hypothetical verdict:**

| Layer | Verdict location | Read surface |
|---|---|---|
| Liquidity (underlying) | `RISK_DECISION.details.shadow.liquidity` (+ `at_approved_quantity`); gate-7 detail carries the measured ADV20 / spread / order-%-ADV | Trading Pool readiness LIQUIDITY check reports the same numbers |
| Statistical (current book) | `RISK_DECISION.details.shadow.statistical` | `GET /api/portfolio/risk` → `statistical` |
| Statistical (proposed book) | `shadow.statistical.comparison` / `.caps` / `.hypothetical` / `.limits`; mirrored to the response as `risk.comparison` / `risk.binding_constraints` / `risk.shadow_statistical`, so trade plans persist it verbatim in `trade_plans.preview` | `ui/components/risk/TradeComparison.tsx` in preview and plans |
| Stress | `shadow.statistical.stress`; rows persisted to `stress_runs` | `GET /api/portfolio/risk` → `statistical.stress`; `POST /api/risk/stress/run` |
| Concentration | inside `shadow.statistical` (ES shares, bucket shares) and `shadow.statistical.correlation_state` | `statistical.correlation_state`; correlation pill on `/risk` |
| Model health | `statistical.model_health` / `.model_risk` / `.dispersion` per model row | Statistical Risk panel; `RiskMethodModal` |
| Model validation (backtests) | rows persisted to `risk_model_backtests`; **never recomputed on a read** | `statistical.validation`; `POST /api/risk/validation/run` |
| Vol targeting v2 | `shadow.vol_targeting_ewma` — logged **beside** the crude proxy that remains in force | EWMA line beside the proxy on `/risk` |

Two structural guarantees hold across the whole diagram: (i) a SHADOW layer that raises is caught and recorded as a note — Tier 0 output is byte-identical (proven, §E); (ii) read views (`GET /api/portfolio/risk`) and the two research endpoints write **no** audit event.

---

## B. Risk Model Matrix

One row per model/capability actually present in the code. "Decision Usage — today" is the operative column: **for every statistical, stress, concentration and model-health row it is `none`.**

### B.1 Registered models (`libs/trading_core/risk/models/base.py` REGISTRY — five entries, verified by import)

| Model (registry name) | Purpose | Status | Data | Assumptions | Strength | Weakness | Decision usage today |
|---|---|---|---|---|---|---|---|
| `historical_var` (`models/var_es.py`) | 1-day VaR at 95/99 from the empirical loss distribution | SHADOW | DELTA_LINEAR book P&L from `stock_bars_daily` closes (~600 obs) | i.i.d. draws from the stored window; today's book held over history; simple returns | No distributional assumption; captures realized fat tails; one canonical `tail_size` | 99 % tail ≈ 6 observations (DEGRADED); no weighting of recent data; ~600-obs ceiling | none |
| `historical_es` | Mean of the k largest losses, k = ceil(n(1−α)) | SHADOW | as above | as above | ES ≥ VaR by construction; Euler ES contributions sum **exactly** to ES | same small-tail problem, amplified — ES averages the thinnest part of the sample | none |
| `gaussian_var` | −μ + zσ (ddof=1, `statistics.NormalDist`) | SHADOW | as above | Normal returns | Closed form; stable at any n; the ensemble's disagreement anchor | Normality is wrong for this data — it is a **trust check**, never favoured | none |
| `gaussian_es` | −μ + σφ(z)/(1−α) | SHADOW | as above | Normal returns | Closed form; smooth in α | understates tails exactly where it matters | none |
| `garch11` (`models/garch.py`) | Conditional volatility: GARCH(1,1) Gaussian MLE, ω>0, α,β≥0, α+β<1, init from EWMA | **RESEARCH** (one step *below* the library SHADOW default) | ≥ 250 aligned observations | zero mean model; Gaussian innovations | Responds to volatility clustering; closed-form multi-step forecast (GARCH_TERM_STRUCTURE); diagnostics: convergence, persistence, half-life, Ljung–Box(10) on standardized residuals², ω-at-floor; recovery on seeded data (α=0.08, β=0.90, n=3000): \|Δα\| ≤ 0.016, \|Δβ\| ≤ 0.023 over four seeds; ~0.15 s per fit | Needs n ≥ 250 — **most books today fall back to EWMA**; small-sample fit instability; Student-t deferred; health DEGRADED on non-convergence / persistence ≥ 0.999 / LB p < 0.05 | none |

Promotion of `garch11` RESEARCH → SHADOW requires the §63 criterion over ≥ 250 forecast days **and** an explicit user action.

### B.2 Non-registry estimators

| Capability | Purpose | Status | Data | Assumptions | Strength | Weakness | Decision usage today |
|---|---|---|---|---|---|---|---|
| EWMA volatility (λ=0.94) `models/volatility.py` | RiskMetrics conditional σ; the §13/§58 fallback when GARCH is not ACTIVE | SHADOW | return series | exponential decay, fixed λ | Two-line model, no fit to fail; **it is the conditional forecaster in practice today** (`statistical.conditional_source`) | λ is a convention, not fitted; no mean reversion | none — logged beside the crude proxy in `shadow.vol_targeting_ewma`; the **crude v0 proxy remains the one that sizes trades** |
| Conditional / filtered historical simulation `models/var_es.py`, `volatility.py` | VaR/ES from vol-standardized residuals rescaled to today's σ (FHS) | SHADOW (EWMA-driven) / RESEARCH (GARCH-driven) | return series + conditional σ | standardized residuals are i.i.d. | Volatility clustering without a distributional assumption; λ→1 collapses to plain HS (tested) | inherits the σ model's error; still ~600 obs | none |
| Portfolio σ + sample covariance `models/volatility.py` | Book volatility from Σ | SHADOW | aligned return matrix (inner-join on dates) | stationary Σ over the window; linear position exposures | Exact; vol contributions sum to σ_p (1e-9) | **sample Σ only — there is no shrinkage/robust path at all** (correction, compliance §3 row 30: the earlier "no shrinkage in the *default* path" implied a non-default one exists; none does — see the §30 row in the Deferred Models table) | none |
| Volatility contributions (`cov/σ_p`) `models/contribution.py` | Which position drives book volatility | SHADOW | Σ, weights | Euler homogeneity | sums **exactly** to σ_p | symmetric measure — not a tail measure | none |
| Euler ES contributions `models/contribution.py` | Which position drives the tail; ES shares per name and per bucket | SHADOW | tail-day P&L | the k tail days are representative | Tail-day averages sum **exactly** to ES; the basis of the concentration verdict | thin tail at 99 %; shares unstable when k is small | none — feeds hypothetical single-name (≤ 35 %) and bucket (≤ 50 %) share caps |
| Marginal / incremental ES `models/contribution.py`, `risk/pretrade.py` | ES(after) − ES(before) on the **joined** series (same k, same window); marginal = candidate Euler RC ÷ q | SHADOW | current + candidate series | candidate return series proxies the new position | Incremental ES is **exact to the float**; marginal × qty == candidate contribution (diff 0.0) | option candidates are DELTA_LINEAR until full reval is wired into VaR; needs min_obs 60 | none — feeds the hypothetical incremental-ES cap (≤ 1.5 % NAV) |
| Distribution diagnostics `models/diagnostics.py` | Skewness, excess kurtosis, Jarque–Bera → NORMAL_LIKE / HEAVY_TAIL / LEFT_SKEWED / UNSTABLE, plus `gaussian_trust` | SHADOW | return/P&L series | JB asymptotics | Tells the reader when the Gaussian rows should be distrusted | JB is weak at n ≈ 600 for mild departures | none — advisory label |
| Ensemble dispersion `models/ensemble.py` | Ratio between the widest and narrowest comparable model views → `MODEL_DISPERSION_HIGH` | SHADOW | ≥ 2 comparable `ModelResult`s | models are measuring the same quantity | Model risk made visible instead of hidden by picking one number | a dispersion threshold is itself unvalidated | none |
| Model-risk state `models/ensemble.py` | LOW / ELEVATED / HIGH from a replayable rule table (incl. `backtest_red_triggers`) | SHADOW | health + dispersion + backtest verdicts | the rule table is a policy, not an estimate | Triggers are enumerated and replayable, not a black box | thresholds unvalidated; empty book needed a special case (found post-deploy) | none |
| NAV drawdown `models/drawdown.py` | Live drawdown on the stored SCHEDULED-NAV series + RECONSTRUCTED_CURRENT_BOOK what-if | SHADOW | one SCHEDULED snapshot per NY day | reconstruction assumes today's book held historically | The live series is real, and honestly labelled as starting now | **live history begins 2026-08-18** — currently near-empty; the reconstructed series is explicitly a what-if | none |
| Correlation regime `libs/trading_core/correlation.py::correlation_regime` | normal (250d) vs current (60d) vs stress-conditioned (worst-10 % days) average pairwise Pearson → NORMAL / ELEVATED / CONVERGING | SHADOW | log returns | Pearson linear dependence | Rule **revised after QA**: CONVERGING iff current average ≥ 0.80 — persistent *or* sudden; hand-checked against the textbook formula (≤ 3e-17) | 0.80 / delta parameters unvalidated; Pearson misses tail dependence | none — displayed and logged |
| Stress catalogue `models/stress.py` (version `d.1`) | Historical windows (per-ticker cumulative return from stored closes), `auto_worst_windows` (worst 1/5/10-day of the equal-weight book), named windows 2024-08-05 / 2025-04, a hypothetical grid, and USER scenarios | SHADOW | stored closes; live chain for legs | β=1 uniform equity shocks; IV shock = realized-vol ratio proxy `RV(window)/RV(prior 20d) − 1`, clipped, labelled **RV_PROXY** | `run_stress` never raises; run health = worst among **PRICED** rows; a window outside stored history is an UNAVAILABLE row naming real dates, not a run downgrade | Every grid row carries `validated=False`; **IV shocks are proxies, not history**; named windows before 2024-03 are UNAVAILABLE | none — produces a hypothetical `STRESS_LOSS_LIMIT` cap (`QuantityCap`, layer STRESS) at a research default of 10 % NAV |
| Implied-vol solver `libs/trading_core/options/iv.py` | Bisection on `bs_price`; guards for below-intrinsic, t ≤ 0, σ = 5 ceiling; labelled INTERNALLY CALCULATED | SHADOW | option mark, spot, strike, DTE, r | Black–Scholes European | Round-trip \|Δσ\| < 1e-6 over a grid | BS assumptions; no dividend/early-exercise treatment | none |
| Option full revaluation `libs/trading_core/options/reval.py` | Reprice legs under S / IV / t moves, **basis-anchored** (basis = mark0 − model0, held constant) | SHADOW | chain mid, provider IV, DTE | BS; constant basis; T ≤ 0 ⇒ intrinsic | Zero scenario returns **exactly 0.0**; `DELTA_LINEAR` fallback is labelled per leg, never silently mixed into a FULL_REVAL claim; missing both ⇒ 0 with a note that loss is understated | Used for **stress only** — VaR/ES remain DELTA_LINEAR | none |
| VaR backtests `risk/validation.py`, `apps/gateway/risk_validation.py` | Walk-forward (window 250, min 60 forecasts): Kupiec POF, Christoffersen independence, ES severity, QLIKE → GREEN/YELLOW/RED | SHADOW | stored history for the current book | walk-forward, no re-fit to the score | **No look-ahead at the strongest level** (§E); runs six model rows incl. EWMA- and GARCH-filtered; bounded runtime 0.69 s at n=600; persisted, never recomputed on read | Verdict bands are Basel-style Kupiec p 0.05/0.01 — parameters, not law; **no history has accrued yet** | none — evidence for a future promotion decision |
| Underlying liquidity gate `libs/trading_core/risk/liquidity.py` | ADV20 ≥ 100k sh, order ≤ 1 % ADV20, spread ≤ 0.5 % → PASS / WOULD_FAIL / UNAVAILABLE + `partial` flag | SHADOW (**REPORT mode**) | `stock_bars_daily` volume; fresh in-process NBBO stream cache only | ADV20 proxies tradable depth | Closes an audit gap where the UI implied protection that did not exist; adds **no** new provider call | Thresholds explicitly UNVALIDATED (the watchlist holds illiquid small caps); option candidates: contracts are not translated to shares; a `partial=True` PASS **must fail closed once promoted** | none — gate 7 always PASSes |
| **Tier 0 `assess()`** `libs/trading_core/risk/engine.py` | Kill switch, heat gate, signal validity, tier budget (abs cap 1.5 % NAV), stop sizing, single-name risk & capital, static bucket, heat headroom, regime cash floor, greek limits | **PRODUCTION** | broker live cash (fail-closed), NAV, positions, regime, portfolio + candidate greeks | deterministic; stop distance > 0; static `TECH_MEGA` bucket membership | Byte-identical to the pre-programme engine (proven three ways, §E); every cap resizes except greeks, which REJECT | Static buckets only — the dynamic ones are still display/SHADOW; greek breach rejects rather than resizes (audit §11 Q5) | **Decides every stock, option and spread open**: APPROVE / APPROVE_WITH_RESIZE / REJECT, quantity, and the reason codes |
| **Tier 0 `assess_income()`** `engine.py` (append-only) | Same ladder for covered calls / CSPs, skipping edge tier and stop-based sizing (qty = contracts) | **PRODUCTION** | as above | CSP risk basis = (strike − expected credit) × 100 with credit = mid × (1 − paper slippage), so assessed heat ≥ booked heat; capital basis = strike × 100; CC bases 0/0 | Closes the audit's largest policy hole — income opens now run pool gate → risk gate → exactly one RISK_DECISION | Two policy consequences flagged to the user (§Appendix): covered calls are refused at book heat ≥ 8 % despite a 0 risk basis; one CSP needs NAV ≈ 67 × (strike − credit) × 100 under the 1.5 % ceiling | **Decides every covered-call and CSP open** |

No capability in the codebase is DEPRECATED. The crude v0 vol-targeting proxy (`portfolio.py`) is the closest candidate — it is superseded in quality by EWMA σ_p but remains the one in force, so it is PRODUCTION, not DEPRECATED.

---

## C. Implementation Report

### C.1 Per phase

| Phase | DEVLOG | What was built | Principal files | Migration | Backend tests |
|---|---|---|---|---|---|
| **A** | (14) | Audit only, no code. 12 parallel read-only inspections → synthesis → adversarial critique → revision | `docs/risk-engine-audit.md` | — | 1032 (baseline, unchanged) |
| **B0** | (15) | Tier 0 hardening: side vocabulary, income through Tier 0, one shared snapshot builder, liquidity gate in REPORT mode, bear-spread replay fix | `engine.py` (`IncomeRiskRequest`, `assess_income`), `apps/gateway/risk_inputs.py` (new), `risk/liquidity.py` (new), `routers/income.py`, `tests/test_migration_parity.py` | **017** `orders_side_vocabulary` | 1089 (+57) |
| **B** | (16) | Core statistical library, persistence, API, UI: returns layer, model registry, Historical/Gaussian VaR & ES, portfolio σ + EWMA, Euler contributions, diagnostics, dispersion, model-risk state, NAV drawdown, walk-forward validation harness, typed snapshot | `risk/returns.py`, `pnl_series.py`, `snapshot.py`, `validation.py`, `models/{base,var_es,volatility,contribution,diagnostics,ensemble,drawdown}.py`, `apps/gateway/risk_snapshot.py` | **018** `risk_snapshots` (+ `risk_metrics`, `risk_contributions`, `atm_iv_daily`) | 1420 (+331) |
| **C** | (17) | Pre-trade portfolio risk: current-vs-proposed comparison, incremental/marginal ES, hypothetical statistical caps by bisection, shadow verdict, binding constraints, correlation regime, EWMA vol-target side-by-side | `risk/pretrade.py` (new), `engine.py` (additive `extra_caps` + `binding_constraints`), `correlation.py` (`correlation_regime`), `routers/orders.py` | — | 1507 (+87) |
| **D** | (18) | Stress engine + option full revaluation: IV solver, basis-anchored leg reval, historical/auto/hypothetical/USER scenarios, stress cap, ON_DEMAND persistence dedupe | `options/iv.py` (new), `options/reval.py` (new), `models/stress.py` (new), `routers/risk.py` | **019** `stress_runs` | 1696 (+189) |
| **E** | (19) | GARCH(1,1) RESEARCH + VaR/ES model validation: Nelder–Mead, χ² via incomplete gamma, diagnostics, FHS, conditional-source fallback seam; walk-forward backtests persisted | `risk/optim.py` (new), `models/_chi2.py` (new), `models/garch.py` (new), `apps/gateway/risk_validation.py` (new) | **020** `risk_model_backtests` | **1987 (+291)** |

**Backend suite: 1032 → 1987 passed, 1 skipped** (+955 tests, +92.5 %). **UI component tests: 43 (Phase B) → 62 (C) → 89 (D) → 114 (E)**, across 8 files — the counts DEVLOG (16)–(19) record. The programme brief cited a pre-programme baseline of 41; that figure does not appear in the DEVLOG, so the verified series starts at Phase B's 43.

Per-file test counts for the risk surface (collected 2026-08-18): `test_risk_chi2` 155, `test_risk_stress` 79, `test_risk_garch` 79, `test_risk_engine` 78 (the audit recorded 45 pre-programme; the +33 are the `assess_income` and byte-identity cases added in B0/C — the 45 originals are untouched), `test_options_reval` 57, `test_risk_phase_b_invariants` 52, `test_risk_var_es` 47, `test_risk_validation` 45, `test_risk_pretrade` 42, `test_risk_validation_api` 31, `test_risk_volatility` 31, `test_options_iv` 26, `test_risk_stress_api` 26, `test_liquidity_gate` 25, `test_risk_optim` 25, `test_risk_model_base` 22, `test_risk_returns` 20, `test_risk_contribution` 17, `test_risk_ensemble` 17, `test_risk_snapshot_api` 17, `test_risk_snapshot_builder` 15, `test_risk_pnl_series` 14, `test_risk_drawdown` 13, `test_risk_diagnostics` 11, `test_risk_snapshot` 11, `test_income_risk_gate` 7, `test_migration_parity` 5.

### C.2 Migrations

| # | Table / change | Purpose | Live status |
|---|---|---|---|
| 017 | `orders.side` CHECK = exactly `libs.broker.provider.MLEG_LEG_SIDES` | Closes the latent Postgres INSERT failure; compose mounts extended to 013–017 | APPLIED (verified via `pg_get_constraintdef`) |
| 018 | `risk_snapshots`, `risk_metrics`, `risk_contributions`, `atm_iv_daily` | Risk persistence; the §44 model identity is stored **inline** on `risk_metrics` (the audit's separate `risk_model_runs` folded in) | APPLIED |
| 019 | `stress_runs` | Stress scenario rows per build and per USER run | APPLIED |
| 020 | `risk_model_backtests` | Walk-forward exceedance records | APPLIED |

A mechanical ORM↔SQL column-mirror test guards each new table, and `tests/test_migration_parity.py` pins the CHECK list to the code constant, every migration to a compose mount, and contiguous numbering.

### C.3 Endpoints

| Endpoint | Change | Audit event |
|---|---|---|
| `GET /api/portfolio/risk` | Additive `statistical` block (mode, snapshot_id, as_of, stale, pnl_method, n_obs/window, data_quality, model_health, model_risk, dispersion, distribution, volatility, `var[]`, `es[]` each with full model identity, `contributions{es,vol}`, `positions_excluded`, `correlation_state`, `stress`, `validation`, `conditional_source`) and `drawdown`; plus `cash_reserved_usd` from B0 | **None** — read views write no audit event (probe-verified) |
| `POST /api/orders/preview`, `/approve`; `POST /api/plans/…` | `risk.comparison`, `risk.binding_constraints`, `risk.shadow_statistical` on the response; the full shadow block in RISK_DECISION details | RISK_DECISION (widened) |
| `POST /api/income/covered-call`, `/cash-secured-put` | Now: kill switch → Trading Pool authorization → collateral law → live selection → `assess_income` → fill | Exactly **one** SYSTEM RISK_DECISION (entity `income_open`) in the fill's transaction |
| `POST /api/risk/stress/run` | User-defined equity/IV/days scenario; validated ranges → 422; persists a USER row | **None** |
| `POST /api/risk/validation/run` | Runs the walk-forward backtests on demand | **None** |

Background: `risk_snapshot_loop` (setting `risk_snapshot_interval_seconds` = 1800, 0 disables, off under tests) writes **one SCHEDULED row per NY day** — the live NAV series drawdown is measured on. Model backtests run once per NY day after the SCHEDULED snapshot. Telemetry added: `risk_snapshot_age_seconds` (age of the newest **SCHEDULED** build only), `risk_model_latency_seconds{stage}`, `risk_snapshot_builds_total{trigger}`, `risk_snapshot_failures_total`.

### C.4 Deployment status

All five phases are **deployed** (backend live; UI on the dev server). Migrations 017–020 are applied to the live volume. The scheduled snapshot loop is running; the first SCHEDULED snapshot was written 2026-08-18 14:59 UTC. Live smoke checks after Phase E: `conditional_source` = EWMA (GARCH UNAVAILABLE for today's book length), validation endpoint honest on the empty book, `/risk` renders.

### C.5 Notable defects the programme found and fixed

| # | Defect | Severity | Where found | Fix |
|---|---|---|---|---|
| 1 | **`orders_side_check`** allowed only `BUY_TO_OPEN`/`SELL_TO_CLOSE` while the code inserts `SELL_TO_OPEN`/`BUY_TO_CLOSE` | **Latent production failure** — the first income short, margin short or cover fill would have failed on INSERT; invisible to the SQLite harness | Phase A audit, live-verified against Postgres | Migration 017 pins the CHECK to `MLEG_LEG_SIDES`; parity test added (DEVLOG (15)) |
| 2 | **Income bypass** — covered-call/CSP opens ran with no `assess()`, no Trading Pool gate, no RISK_DECISION | **Policy hole**: an entire trade path outside the risk engine | Phase A audit | Append-only `assess_income()` + pool gate + exactly one RISK_DECISION (DEVLOG (15)) |
| 3 | **BEAR_PUT_SPREAD replay never entered** — the bear branch called the BULL entry evaluator with inverted strike geometry; the API test passed vacuously on an empty trade list | Silent wrong result | Phase A audit | `_evaluate_entry_bear` + put-vertical geometry; bull branch pinned unchanged; PLTR run 0 → 14 trades (DEVLOG (15)) |
| 4 | **`tail_size` divergence** — a *second* implementation agreed on the 95/99 grid but diverged off-grid (34 of 20,000 (n,α) probes) | Would have produced two different ES definitions in one system | Phase B integrator | One canonical function, mutation-verified (DEVLOG (16)) |
| 5 | **UI drift crashes** — unguarded `.toFixed()` on nullable distribution fields when the series is UNSTABLE; nullable `model_risk` | Two crash paths on the risk page | Phase B QA, **before deploy** | Guarded, typed, regression-tested (DEVLOG (16)) |
| 6 | **UI↔gateway drift (Phase C)** — row field names, a separate `tier0_rows`, `caps` as an object: the whole comparison table rendered as dashes | Feature silently non-functional | Phase C QA, **before deploy** | Fixed; fixtures replaced with a real gateway payload; regression-tested (DEVLOG (17)) |
| 7 | **ON_DEMAND persistence flood** — the UI polls `/risk` every 15 s; each poll would have written a snapshot + metrics + contributions + ~12 stress rows | Unbounded table growth in production | Phase D QA, **before it happened live** | ON_DEMAND builds persist at most once per 15 minutes (`statistical.persisted`); verified live True → False on re-read (DEVLOG (18)) |
| 8 | **Correlation CONVERGING rule inverted in effect** — the original "jump AND level" rule read a book that *always* moved at ρ ≈ 1 as NORMAL — the exact §19 failure mode the state exists to catch | Model logic error | Phase C QA | CONVERGING iff the **current** average ≥ 0.80 (persistent *or* sudden); the jump now annotates `reason` (DEVLOG (17)) |
| 9 | **Stress run health over-reported failure** — a named window outside stored history downgraded the whole run | Misleading health | Phase D QA | Run health = worst among **PRICED** rows; an out-of-history window is an UNAVAILABLE row naming real dates (DEVLOG (18)) |
| 10 | **Phase E display defects** — `comparison.preferred` rendered the raw model KEY; LR statistics typed required but not served; per-row `mode` invisible | Misleading UI | Phase E QA, **before deploy** | All three fixed (DEVLOG (19)) |
| 11 | **Empty-book model risk misleading** — the rule table returned ELEVATED on a book with no positions | Misleading state | Phase B post-deploy | Empty-book special case: LOW, "no open positions — nothing to model" (DEVLOG (16)) |

Six of the eleven were caught by QA/adversarial review **before deployment**. Items 1–3 were pre-existing defects the audit surfaced, not regressions introduced by this programme.

---

## D. Deferred Models

Every row below was a deliberate §62 decision recorded in `docs/risk-engine-audit.md` §7, not an omission.

| Model / capability | Audit decision | Why deferred | Re-visit trigger |
|---|---|---|---|
| **EVT / POT / GPD** (§16–§17) | DEFER | ~600 obs → ~30 exceedances at a 95 % threshold: GPD shape parameter is unstable. Would produce **false precision** in exactly the tail the user would trust most | ≥ 1500 observations (i.e. after the deep backfill, audit §11 Q2); research metric only even then |
| **Copula / Student-t / GARCH-copula** (§20) | REJECT for now | Insufficient data and a book of 2–8 names; unvalidatable; marginal value against stress + risk contributions, which already answer the concentration question | Book > 15 names **AND** ≥ 1500 obs |
| **Markowitz MVO / tangency / frontier** (§28) | REJECT | The spec itself warns on MVO. Entries here are signal-driven single names, not rebalanced weights — there is no portfolio for an optimizer to optimize | Only if the platform adopts a weights-based allocation model |
| **GMV** (§29) and **ERC** (§32) | DEFER | Both are *benchmarks*, and a benchmark without a comparison harness is a number with nothing to compare to | **Precondition: the walk-forward weights harness** — weights persisted per rebalance date over the reconstructed book, compared to the signal-driven book's realized vol/ES/drawdown. A point-in-time report endpoint is explicitly not enough |
| **Turnover stability** (§31) | DEFER | No optimizer runs in production — there are no weight changes to measure | Arrives with GMV/ERC and the weights harness |
| **Monte Carlo VaR** | DEFER | Historical simulation + the stress catalogue already cover the need; MC would add simulation error on top of an already-thin sample | Only alongside a copula/EVT model that needs a simulation engine |
| **Student-t GARCH innovations** | DEFER | Gaussian GARCH must first prove it fits and validates on this data; adding a shape parameter to an n≈250-marginal fit compounds instability | GARCH(1,1) Gaussian promoted to SHADOW and stable over ≥ 250 forecast days |
| **Rolling Spearman correlation** | ~~Deferred~~ → **BUILT 2026-08-19** as a RESEARCH display (deferred as a *gate input*, which it remains) | **Correction (compliance §3 row 18).** This row previously described a RESEARCH-mode display as already shipped; it was not — the estimator did not exist, and the audit (`audit.md:213` "IMPLEMENT AS RESEARCH") and design (§7.4 "not built") disagreed with this report. It is now genuinely built: `correlation.py` `spearman` / `rolling_spearman` / `rolling_spearman_matrix` / `rolling_spearman_average`, plus `CorrelationState.current_avg_spearman` over the same short window. Informative next to Pearson; still no evidence it should drive a decision, and it enters no state rule | Promotion to a gate input only if Pearson-based bucket enforcement demonstrably misclassifies during a live regime shift |
| **Empirical IV shocks** for stress | DEFER — RV proxy shipped instead | No IV history existed. Stress IV shocks are the realized-vol ratio `RV(window)/RV(prior 20d) − 1`, clipped and labelled **RV_PROXY** | `atm_iv_daily` ≥ **120** observations. Accumulation began Phase C (AAPL 0.2448 was the first recorded value) — so ≈ 120 trading days from 2026-08-18 |
| **Multi-day empirical horizons** (5D/10D) | DEFER | Overlapping multi-day windows on 600 obs give few independent observations. 1D is estimated natively; longer horizons are shown only as **labelled √h scaling** | Deep backfill (≥ 1500 obs) |
| **Robust / shrinkage covariance** (§30) | DEFER — recorded 2026-08-19 (compliance §3 row 30) | The audit rated this IMPLEMENT AS RESEARCH (`audit.md:271`) but its own argument defers it: shrinkage helps when the name count `n` approaches the sample length `T`, and with **n ≤ 8 names against T ≈ 600** the sample Σ is already well conditioned. Ledoit-Wolf would move the numbers by less than the estimation noise it is meant to control, and §62 Q6 says sample-only until the book is larger. This row is the missing user-facing record: nothing is built, and neither half of the sample-vs-shrunk comparison exists | **Book > 15 names** (where n/T stops being small); revisit alongside the sample-vs-shrunk RC delta as a research diagnostic |
| **Multi-factor risk contribution** | REJECT (unchanged) | No MULTI-factor data is available (no VIX, no factor-return library). Euler ES contributions answer the concentration question with data that exists. *Consistency note, 2026-08-19:* a **single**-factor SPY diagnostic was built this batch (`risk/models/factor.py`, RESEARCH — `beta_vs_factor` / `factor_risk_share`, compliance §3 row 11), which is a beta against ONE proxy series the caller supplies, not a factor model. It derives no cap: §11's `max_factor_...` concentration limit stays REJECT-documented, because a cap needs a validated taxonomy | A factor-return data source, which would be a new provider integration |
| **Deep historical backfill to ≥ 2018** | DEFER — **awaiting the user** (audit §11 Q2) | Not a model decision: it needs user approval (more storage and API calls, no new cost on the existing Alpaca provider). It is the **precondition for EVT, multi-day horizons, and the 2020-03 / 2022 stress windows** | User approval |
| **Backtest replay of `assess()`** (§64 parity) | DEFER — **awaiting the user** (audit §11 Q4) | Restores live/backtest parity, but changes **every** historical backtest result and requires a portfolio-state abstraction in six single-position engines | User go/no-go; a dedicated iteration after Phase C |
| **Greek limits REJECT → RESIZE** | DEFER — **awaiting the user** (audit §11 Q5) | This is the *one* proposed change to prior Tier 0 behaviour. Every other cap resizes; a greek breach rejects outright. Changing it without consent would break the byte-identity guarantee this programme is built on | User confirmation; two existing tests would be updated |

---

## E. Validation Report

Each phase was validated by the implementers **and** an independent adversarial verifier running its own probes rather than re-reading the implementers' tests.

### E.1 Per phase

| Phase | Verification evidence | Suite |
|---|---|---|
| **A** | Critic spot-checked > 30 `file:line` citations; all material claims held. The critique returned REVISE on three items (a test count, drifted line anchors, an over-stated liquidity gap) — corrected before publication | 1032 passed, 1 skipped (baseline; no code changed) |
| **B0** | Three implementers + an adversarial verifier that **diffed the whole change set against a pristine snapshot**. Probes: contracts 0 / huge / NaN NAV, heat exactly at threshold, greeks None, empty/None/NaN volumes, crossed quotes. **7 MINOR findings, all fixed in the same entry** (NAV un-netted vs deployable cash; cash-detail wording naming the pledged amount; None volumes → unmeasurable rather than TypeError; NaN ADV floor rejected; `partial` flag; CSP basis at expected fill; `contracts` must be a real int) and pinned by tests. `assess()` byte-identity: md5 of the first 607 lines equal to baseline; the 45 original engine tests untouched. Live: migration 017 verified with `pg_get_constraintdef` | 1089 passed, 1 skipped |
| **B** | Cross-module invariants pinned in `tests/test_risk_phase_b_invariants.py` (52 tests): ES ≥ VaR, monotone in α, **ES contributions sum == ES exactly**, vol RC sum == σ_p (1e-9), scale/shift laws, **walk-forward never sees `pnl[t]`**, min_obs → None/UNAVAILABLE without exceptions, filtered-HS λ→1 ≈ HS, registry SHADOW modes, end-to-end 3-ticker pipeline. Adversarial probes: **Tier 0 byte-identical when the snapshot builder is monkeypatched to raise** (only `shadow.statistical` differs); `GATE_ORDER` unchanged; API key sets == §6 exactly on empty and seeded books; persistence rows carry model_name/version/params; honest nulls at n < 60 **with reasons**; one SCHEDULED row per NY day; build latency **13 ms on 5×600 (75× inside budget)**. UI: tsc clean, 43 component tests, next build ok | 1420 passed, 1 skipped |
| **C** | **SHADOW proof**: with `compare`/`statistical_caps`/`shadow_verdict` monkeypatched to raise, the decision, approved quantity, gates, reason codes, binding constraints and budget multiplier are all identical. **Byte-identity of `assess()` proven three independent ways**: a 240-case seeded battery in-suite; C-1's reconstruction of the pre-Phase-C engine compared on all ten output fields; QA's independent strip-and-diff confirming pure additions. **Incremental ES exact to the float**; marginal × qty == candidate contribution (diff **0.0**); **cap bisection brute-forced over every integer quantity in 4 constructed cases** (reported cap == true max); correlation regime hand-checked against the textbook Pearson formula (≤ **3e-17**); vol-targeting SHADOW proven by sweeping σ over **5 orders of magnitude** — the applied multiplier stayed pinned while the EWMA multiplier moved; plans store the new keys; read views write no risk audit event. Live after deploy: full chain runs (gate 7 measured a streamed 0.013 % spread), `atm_iv_daily` accumulating, first SCHEDULED snapshot 14:59 UTC | 1507 passed, 1 skipped |
| **D** | IV round-trip grid \|Δσ\| < **1e-6**; zero-scenario **bit-exact 0.0**; expiry intrinsic; spread/income signs; stock linear; historical shocks from a hand-built path; **auto worst windows brute-forced**; **stress cap vs brute force**; \|P&L\| monotone in \|q\|; API stress keys; persistence rows; USER endpoint 422 / no-audit / exactly one row; **Tier 0 byte-identical when the stress layer raises**; ON_DEMAND dedupe. UI tsc clean, 89 component tests. **Caveat: the final adversarial verifier could not run** (API 529 overload, twice) — the orchestrator verified directly instead: migration 019 live, `assess()` called without `extra_caps`, stdlib-only, dedupe live (`persisted` True → False on re-read), USER run persisted (run_id 8), UI `/risk` renders. This phase therefore has one less layer of independent scrutiny than B, C and E | 1696 passed, 1 skipped |
| **E** | **No look-ahead at the strongest available level**: mutating *only the last observation* to −999,999 leaves **every earlier forecast bit-identical across all four estimator families**, including the stateful GARCH window filter. Kupiec / Christoffersen / ES severity recomputed from raw forecasts for a full row. **GARCH recovery on three fresh seeds** (independent of the implementers' four). **χ² identities plus eight published critical values, agreement ≤ 1.5e-3**; df=1/2 closed forms agree with the general incomplete-gamma path to ≤ **4e-16**. Runtime 0.32 / 0.69 / 1.79 s at n = 400 / 600 / 900. **Honest nulls**: 280 observations → six UNAVAILABLE rows reading `n=30 < min_forecasts=60`. No audit event on the endpoint; migration 020 live. UI tsc clean, 114 component tests | **1987 passed, 1 skipped** |

### E.2 Current suite

```
1987 passed, 1 skipped in 64.50s (0:01:04)
```

UI: `Test Files 8 passed (8) / Tests 114 passed (114)`.

### E.3 What is still unvalidated

This is the most important subsection of the report.

1. **Zero trading days of shadow decisions have elapsed.** The programme finished on the same day the first SCHEDULED snapshot was written. The audit §11 Q3 promotion policy asks for ≥ 20 trading days. **The counter stands at 0.**
2. **No backtest history exists.** `risk_model_backtests` accrues from the first SCHEDULED run. The GARCH-vs-EWMA §63 comparison needs ≥ 250 forecast days; none have accrued.
3. **Every threshold in the statistical, stress and liquidity layers is a research default**, documented as UNVALIDATED at its definition site: `StatisticalLimits` (portfolio ES-95 ≤ 5 % NAV, single position ≤ 35 % of ES contributions, bucket ≤ 50 %, incremental ES-95 ≤ 1.5 % NAV, min_obs 60); `StressLimits(max_stress_loss_pct_nav=0.10)`; `LiquidityLimits(min_adv20_shares=100_000, max_order_pct_adv20=0.01, max_quote_spread_pct=0.005, adv_window=20)`; the correlation `converging_level` 0.80 and `converging_delta`; the dispersion ratio threshold; the model-risk rule table; the Kupiec p 0.05/0.01 verdict bands. **What is validated is that they are computed and applied correctly — not that they are the right numbers.**
4. **The hypothetical stress grid is entirely unvalidated by construction** — every row carries `validated=False`.
5. **Phase D's independent adversarial pass did not run** (API overload). Its checks were performed by the orchestrator directly, which is a weaker guarantee than the pattern used in B, C and E.
6. **GARCH has never fitted on a live book.** Live smoke returned `conditional_source` = EWMA because no current book reaches 250 aligned observations. GARCH recovery is proven on **seeded simulations only**.
7. **No live CSP has been booked**, so the B0 assumption that Alpaca cash is not reduced by CSP collateral (options buying power absorbs it) — and therefore that netting once in the shared builder is correct — remains unverified against a real fill.

---

## F. UI Changes

All risk UI ships under `ui/`. Component tests grew 43 → 62 → 89 → **114** across Phases B–E (8 files), re-verified at 114 for this report.

### F.1 Components

| Component / surface | File | Contents |
|---|---|---|
| **Statistical Risk panel** | `ui/components/risk/StatisticalRisk.tsx` | Methodology-labelled tiles ("Historical VaR 95 % 1D", "Historical ES 95 % 1D", "Gaussian VaR 95 % 1D", σ, drawdown, model risk) each with health and n; the full VaR/ES row table; model disagreement (dispersion); the distribution line; model-risk reasons; drawdown block with an honest empty state |
| **Risk Contribution panel** | within `StatisticalRisk.tsx` | Capital weight vs ES-95 risk share as two bars **on one axis**, with totals that reconcile — the visual point being the *gap* between how much capital a position uses and how much tail risk it owns |
| **Stress Scenarios** | `ui/components/risk/StressScenarios.tsx` | Table with a kind badge, an **UNVALIDATED badge on every research-grid row**, P&L in $ and % NAV, method coverage (how many legs were FULL_REVAL vs DELTA_LINEAR), health and reason, worst row highlighted; plus a user-scenario form driven by the **server's** validated ranges with inline errors |
| **Model Validation** | `ui/components/risk/ModelValidation.tsx` | Per-model table (n, exceedances vs expected, rate, Kupiec p, Christoffersen p, ES severity, GREEN/YELLOW/RED badge, health/reason), a **RESEARCH badge on GARCH rows on top of the panel's SHADOW**, the EWMA-vs-GARCH comparison, the promotion-criterion sentence verbatim, a "Run now" control, and an honest empty state |
| **Trade Comparison** | `ui/components/risk/TradeComparison.tsx`, used in `ui/app/watchlist/[ticker]/page.tsx` (preview and plans) | The §46 "CURRENT vs AFTER TRADE" table: **Tier 0 rows first** (heat, cash), then VaR/ES/σ, then concentration (incremental ES-95, this position's ES share, largest single-name ES share, bucket ES shares, net delta notional), then "Worst stress loss"; requested vs approved vs **hypothetical statistical** quantity; binding constraints grouped **HARD_LIMIT → STATISTICAL/CONCENTRATION/STRESS** with §47 cap sentences |
| **Correlation regime pill** | `ui/app/risk/page.tsx` | NORMAL / ELEVATED / CONVERGING with normal vs current vs stress-conditioned ρ and the server's reason string ("regime shift: normal 0.61 → current 0.84"); CONVERGING is the loud tone |
| **EWMA side-by-side** | `ui/app/risk/page.tsx` | "EWMA forecast → multiplier" rendered **beside** the crude v0 proxy line, with the server's note verbatim, and honestly null when EWMA needs more data. The reader can see the two numbers disagree and see which one is in force |
| **Positions option rows** | positions surface | Premium at risk, DTE, IV (with the INTERNALLY CALCULATED label when solved rather than provider-supplied), worst scenario loss and its scenario name |
| **Methodology modal** | `ui/components/shared/RiskMethodModal.tsx` | Reached from "ⓘ How is this calculated?" on every tile: model, confidence, horizon, lookback, distribution, as_of, data source, health, version, and Advanced diagnostics |
| **Glossary** | `ui/lib/glossary.ts` (67 entries), surfaced via `ui/components/shared/Term.tsx` and `ui/app/guide/page.tsx` | Risk terms added across phases: `max_drawdown`, `drawdown_reconstructed`, `model_risk_state`, `incremental_es`, `kupiec_test`, `christoffersen_test`, `garch`, `stress_test`, `full_revaluation`, `iv_crush`, `basis_adjustment` and others |

All of the above is bilingual (English / Simplified Chinese), and the bilingual rendering is itself covered by component tests.

### F.2 How SHADOW is communicated

Deliberately, and in more than one register, because a badge alone is easy to stop seeing:

1. **The badge comes from the server, not the client.** `StatisticalRisk` and `TradeComparison` render the mode string the API sends (`statistical.mode`). If the backend were ever promoted, the UI would follow automatically — and, equally, the UI cannot claim PRODUCTION while the backend is SHADOW.
2. **A sentence, not just a badge.** Each panel carries a one-line disclaimer stating the consequence in plain language — e.g. the ModelValidation panel reads: *"SHADOW — backtest verdicts are computed, persisted and displayed; they alter no trading decision. A RED verdict is a signal to review a model, not an automatic change to any limit."*
3. **Nested status is shown where it differs.** GARCH rows carry **RESEARCH** on top of the panel's SHADOW, because RESEARCH is a weaker claim than SHADOW and collapsing the two would overstate the model.
4. **Three quantities are shown side by side** in TradeComparison — requested, Tier 0 approved, and the hypothetical statistical quantity — so the user can see exactly what the shadow layer *would* have done, and that it did not.
5. **Binding constraints are grouped by layer**, HARD_LIMIT first, so the constraint that actually bound is visually separated from the ones that merely would have.
6. **Unvalidated is labelled at the row level**, not only the panel level: every hypothetical stress-grid row carries its own UNVALIDATED badge.
7. **Empty and degraded states are honest** rather than hidden — "no open positions — nothing to model", `n=30 < min_forecasts=60`, UNAVAILABLE rows naming the real dates of a window outside stored history.

---

## G. Remaining Model Risk

### G.1 What the platform still cannot reliably measure

| # | Limitation | Consequence | Mitigation in place |
|---|---|---|---|
| 1 | **99 % tails on ~600 observations** — the 99 % tail averages ≈ 6 points | The 99 % VaR/ES numbers are displayed but statistically thin; a single unusual day moves them materially | Health label DEGRADED; sample size and `tail_size` shown on every row; policy is that gates would use ES **95 %** first |
| 2 | **VaR/ES are DELTA_LINEAR for options** | Long-gamma and long-vega tails are **understated** — precisely the positions whose tails matter most. Full revaluation exists but is wired into **stress only** | `pnl_method` labelled on the API and in the UI; the Phase D stress layer covers the same book with FULL_REVAL |
| 3 | **Historical IV shocks are RV proxies** (`RV(window)/RV(prior 20d) − 1`, clipped) | Stress IV moves are a plausible analogue of a real IV move, not a measured one | Labelled RV_PROXY at every level; empirical shocks deferred until `atm_iv_daily` ≥ 120 |
| 4 | **No VIX, no SPX** from any configured provider | No market-wide volatility state, no index stress anchor, no beta estimation against a market factor | Documented; VIX-free stress design accepted; a free index-history source would be a new provider integration (audit §11 Q2) |
| 5 | **β = 1 uniform equity shocks** in stress | Every name is shocked identically; a low-beta and a high-beta position produce the same relative loss. Concentration in high-beta names is **understated** | Follows from #4; documented in the catalogue |
| 6 | **Every statistical/stress/liquidity threshold is unvalidated** | If promoted today they would be arbitrary production thresholds — the exact §11 failure the spec forbids | Everything is SHADOW; thresholds are documented UNVALIDATED at their definition sites |
| 7 | **Fail-open on UNAVAILABLE statistical views** | `statistical_caps()` returns **no caps** when a view is UNAVAILABLE. Safe while SHADOW (a missing cap changes nothing); **unsafe the moment it is promoted** | Explicit open item: the PRODUCTION promotion **must** choose the fail-closed rule first. The same applies to a liquidity PASS with `partial=True` |
| 8 | **Drawdown, ATM-IV and backtest histories all start 2026-08-18** | The live NAV drawdown series is near-empty; IV history is one day old; no exceedance record exists | The reconstructed drawdown is labelled RECONSTRUCTED_CURRENT_BOOK and explicitly a what-if; empty states are honest |
| 9 | **Single-process assumptions** (ADR-007) | The snapshot loop, the 15-minute ON_DEMAND dedupe and the in-process NBBO cache all assume one process. A second gateway replica would double-write snapshots and could see a different quote cache | Documented; the dedupe is a mitigation, not a distributed lock |
| 10 | **A bucket limit already breached by non-candidate members resolves to cap 0** | The hypothetical cap says "zero" for a reason the candidate cannot fix | UI wording notes it |
| 11 | **Correlation is Pearson only** | Linear dependence; **tail dependence is not measured** at all (the copula/lower-tail work was rejected/deferred) | The stress catalogue's correlation-convergence scenario (−8 % equity / +30 % IV) is the standing proxy |
| 12 | **The reconstructed-book drawdown is a counterfactual** | It answers "what would this exact book have done", not "what did the account do" | Labelled as a what-if in the UI and the glossary |

### G.2 Promotion path

Per audit §11 Q3, the standing proposal — **still awaiting user confirmation**:

1. **Shadow window: ≥ 20 trading days** with logged hypothetical decisions. **Elapsed: 0.** Every RISK_DECISION already carries the full hypothetical block, so the evidence accrues automatically from here.
2. **Review the logged record** before flipping anything: which trades the statistical layer *would* have resized or rejected, which watchlist symbols the liquidity gate *would* have blocked, how often the stress cap *would* have bound, and whether the correlation state ever read CONVERGING.
3. **A fail-closed rule must be decided before promotion, not after.** Today an UNAVAILABLE statistical view yields no caps and a `partial=True` liquidity measurement passes. In PRODUCTION both must fail closed, and the exact rule is an open decision.
4. **Promotion is an explicit user action** — a runtime-config flip (`risk_statistical_mode=SHADOW|PRODUCTION`), never an import, never a default, never a code deploy that silently changes behaviour.
5. **Per-model promotion, not one switch.** GARCH RESEARCH → SHADOW needs its own §63 criterion (Kupiec p at least EWMA's over ≥ 250 forecast days, Christoffersen p ≥ 0.05, no FAILED diagnostics) — a *lower* bar than any SHADOW → PRODUCTION step.
6. **Mechanically**, promotion means populating `extra_caps` at the `assess()` call site in `apps/gateway/routers/orders.py`. Today that argument is never passed at either engine call site in `apps/` (`orders.py:1744` `assess`, `income.py:333` `assess_income`), so it takes the engine's `()` default — the seam exists precisely so that promotion is a one-line, reviewable, revertible change rather than a refactor.

---

## Appendix — Open decisions for the user

Consolidating `docs/risk-engine-audit.md` §11 Q1–Q7, the retention item raised in Phase D, and the shadow-gate promotions. Each carries the autonomous default adopted so far (DEVLOG (14)), which remains in force until overridden.

| # | Decision | Autonomous default in force | Why it needs you |
|---|---|---|---|
| **Q1** | §60 vs the execution-chains mandate: confirm the policy is "`allow_short_stock` / `allow_margin` default False, user-toggleable, naked shorts locked forever" so it can be recorded as an ADR | Permissions **neither broadened nor reverted** | It is a policy statement about what the account may do — not a modelling choice |
| **Q2** | **Deep historical backfill** to ≥ 2018 for watchlist + SPY/QQQ via the existing Alpaca provider (no new cost; more storage and API calls). Separately: accept a VIX-free stress design, or approve investigating a free index-history source | Backfill **NOT run**, pending approval | It is the single highest-leverage unblock in this report: it gates EVT, multi-day horizons, longer VaR samples, and the 2020-03 / 2022 stress windows |
| **Q3** | **Promotion policy for statistical gates**: ≥ 20 trading days SHADOW, then an explicit user runtime-config flip. Plus: the fail-closed rule for UNAVAILABLE views and `partial=True` liquidity | Proposed policy assumed; **0 of 20 days elapsed**; nothing promoted | Promotion changes real trading quantities. It should be your decision, taken against logged evidence |
| **Q4** | **Backtest parity**: should live `assess()` (and later the statistical gates) be replayed inside the backtest engines? | **Deferred**, pending go/no-go | It changes **every** historical backtest result and needs a portfolio-state abstraction in six single-position engines |
| **Q5** | **Greek limits REJECT → RESIZE** — the one proposed change to prior Tier 0 behaviour | Greek limits **stay REJECT**, byte-identical | It is the only item that would break the byte-identity guarantee the whole programme rests on |
| **Q6** | **§68 "out-of-sample tested" vs the deleted IS/OOS design**: confirm that for *risk models* the walk-forward exceedance backtest **is** the out-of-sample test, and is validation rather than parameter search | Reading adopted | It defines what "PRODUCTION" means for Q3 |
| **Q7** | **§58 automatic pause on invalid critical data**: should *book-level* critical-data invalidity trip the kill switch automatically, or only mark the snapshot INVALID and alert? | **Mark INVALID and SHADOW-log; no automatic kill switch** | An automatic trading halt is a policy decision with real cost when it fires on a false positive |
| **R1** | **`stress_runs` retention policy** beyond the 15-minute ON_DEMAND dedupe (raised in DEVLOG (18)) | Dedupe only; **no retention/pruning policy** | Rows accrue per build and per USER run indefinitely; the right horizon depends on how long you want stress history for review |
| **P1** | **Shadow-gate promotions, individually**: statistical caps (ES limit, single-name share, bucket share, incremental ES) · stress-loss cap · underlying liquidity gate REPORT → FAIL · dynamic correlation buckets display → enforced · EWMA σ_p replacing the crude v0 vol-target proxy · GARCH RESEARCH → SHADOW | **All remain SHADOW/RESEARCH**; the crude proxy remains in force | Each is a separate risk decision with its own evidence requirement. They should not be promoted as a block |

---

**End of report.** Programme phases A–E complete and deployed; F/G/H remain research or deferred per the Phase A audit. The engine is instrumented, persisted, displayed and honest about what it does not know. It has not yet been allowed to decide anything, and the evidence that would justify letting it does not exist yet.
