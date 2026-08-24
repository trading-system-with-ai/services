# Risk Engine Upgrade — Phase B library design contract

**Scope.** The pure, stdlib-only statistical risk library under
`libs/trading_core/risk/` that Phase B adds (spec `prompts/risk_engine.md`
§3–§10, §15, §39–§45, §57–§59; audit `docs/risk-engine-audit.md` §10 Phase B).
Everything here is SHADOW/RESEARCH: nothing in this library alters a Tier 0
decision. The Tier 0 engine (`risk/engine.py`) stays byte-identical.

This document is the *contract* every module follows so VaR, ES, volatility
and risk contribution are mutually coherent (ES ≥ VaR, contributions sum to
totals, one sign convention, one quantile estimator). Deviations require an
edit to this file first.

---

## 1. Conventions

| Item | Convention |
|---|---|
| Numerics | Python stdlib only: `math`, `statistics` (`NormalDist`), `dataclasses`, `enum`. `math.fsum` for every sum of floats that feeds a statistic. Deterministic. |
| Units | P&L, VaR, ES, contributions in **USD per horizon** (default horizon 1 trading day). Percent-of-NAV variants are derived by the caller (snapshot builder), never inside the estimator. |
| Sign | A **P&L series** is `pnl[t] > 0` = gain. A **loss** is `L = -pnl`. VaR/ES are reported as **losses (positive = money lost)**. If the α-tail of losses is negative (portfolio gains even in its worst tail), the raw negative number is reported honestly — no flooring at 0. UI formats. |
| Confidence | `confidence ∈ (0.5, 1)`; the platform grid is `0.95` and `0.99`. |
| Horizon | `horizon_days ≥ 1`. Only 1D is *estimated*; multi-day values are `SQRT_TIME_SCALED` from 1D and labelled so (`scaling="SQRT_TIME"`), never gated on. |
| Return type | `LOG` for correlation / volatility of *returns* (existing `correlation.py` and `realized_vol` conventions — preserved byte-identically). `SIMPLE` for **P&L construction** (`pnl = exposure × r_simple` is exact for stock). A `ReturnMatrix` carries its `return_type`; mixing is a `ValueError`. |
| Sample | `n` = number of observations actually used. Every result carries `sample_size`. Minimums are parameters (`min_obs`), never magic numbers inline. |
| Honest nulls | Insufficient data ⇒ `value=None`, `health=UNAVAILABLE`, `reason` string with the real numbers (`"n=17 < min_obs=60"`). Never a fabricated 0. |
| Time | Dates are `datetime.date`; `as_of` on results is the last observation date (or a `datetime` for live snapshots). |
| Errors | Malformed input (non-positive close, mismatched lengths, α outside range) raises `ValueError`; missing data does *not* raise — it degrades health. |

---

## 2. Modules and public API

### 2.1 `risk/returns.py` — returns layer (spec §3)

```python
ReturnType = Literal["SIMPLE", "LOG"]

def simple_returns(closes: Sequence[float]) -> list[float]     # c[t]/c[t-1] - 1; ValueError on close <= 0
def log_returns(closes: Sequence[float]) -> list[float]        # MOVED here from correlation.py; correlation.py re-exports it (behaviour byte-identical, its tests unchanged)

@dataclass(frozen=True)
class ReturnSeries:               # one ticker
    ticker: str
    dates: tuple[date, ...]       # date of the return (the LATER bar)
    values: tuple[float, ...]
    return_type: ReturnType
    frequency: str = "1D"
    source: str = "stock_bars_daily"
    def window(self, n: int) -> "ReturnSeries"          # last n
    @property n_obs

@dataclass(frozen=True)
class ReturnMatrix:               # date-aligned, several tickers
    dates: tuple[date, ...]
    tickers: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]   # rows[t][i] = return of tickers[i] on dates[t]
    return_type: ReturnType
    frequency: str = "1D"
    source: str = "stock_bars_daily"
    def column(self, ticker: str) -> list[float]
    def window(self, n: int) -> "ReturnMatrix"
    @property n_obs, @property as_of (dates[-1] or None)

def returns_from_closes(ticker, bars: Sequence[tuple[date, float]], *, return_type) -> ReturnSeries
def align(series: Sequence[ReturnSeries]) -> ReturnMatrix
```
Alignment rule: **per-ticker returns are computed first on that ticker's own
consecutive bars, then INNER-JOINED on return dates.** A date missing for any
ticker is dropped for all; returns are never compounded across a gap. Bars
must be strictly increasing by date (else `ValueError`).

Metadata for provenance (spec §3 `return_type, frequency, lookback_window,
timestamp, data_source, model_version`) lives on `ReturnSeries/Matrix` +
`ModelMeta` (§2.2) — never a loose dict.

### 2.2 `risk/models/base.py` — model abstraction & registry (spec §4, §41, §44, §70)

```python
class ModelHealth(StrEnum): ACTIVE, DEGRADED, UNAVAILABLE, FAILED
class ModelMode(StrEnum):   RESEARCH, SHADOW, PRODUCTION

@dataclass(frozen=True)
class ModelMeta:                      # spec §44 — everything needed to reproduce the number
    model_name: str                   # e.g. "historical_var"
    model_version: str                # e.g. "1.0.0" — bump on ANY estimator change
    params: Mapping[str, Any]         # confidence, horizon_days, lookback, lambda, ...
    return_type: ReturnType | None
    frequency: str | None
    lookback: int | None              # observations requested
    data_source: str | None
    as_of: date | datetime | None
    confidence: float | None
    horizon_days: int | None
    distribution: str | None          # "EMPIRICAL" | "NORMAL" | ...

@dataclass(frozen=True)
class ModelResult:
    value: float | None
    health: ModelHealth
    reason: str | None                # why not ACTIVE (real numbers)
    sample_size: int
    meta: ModelMeta
    diagnostics: Mapping[str, Any] = field(default_factory=dict)   # small, typed-ish scalars only

class RiskModel(Protocol):
    name: str
    version: str
    mode: ModelMode
    def calculate(self, *args, **kwargs) -> ModelResult: ...
    def validate(self, result: ModelResult) -> ModelResult: ...   # may downgrade health; never upgrades
    def diagnostics(self, result: ModelResult) -> Mapping[str, Any]: ...
    def metadata(self) -> ModelMeta: ...

REGISTRY: dict[str, RiskModel]      # name -> instance; register(model), get(name)
```
Rules: `validate` is separate from `calculate` (spec §57): calculation never
claims health beyond `ACTIVE`-if-computed; validation may downgrade to
`DEGRADED`/`UNAVAILABLE`. A model in `SHADOW`/`RESEARCH` mode must be
impossible to wire into a veto (the engine consults `mode`).

### 2.3 `risk/models/var_es.py` — VaR & ES (spec §6, §7, §8)

Estimator (the ONE quantile convention for the whole platform):

- Inputs: `pnl: Sequence[float]` (USD/day, gain-positive), `confidence α`, `horizon_days h`.
- Losses `L_t = -pnl_t`, sorted **descending**: `L(1) ≥ L(2) ≥ … ≥ L(n)`.
- Tail size `k = ceil(n·(1−α))` (n=600: 30 @95%, 6 @99%; n=250: 13 @95%, 3 @99%).
- **Historical VaR_α = L(k)** — the k-th largest loss (empirical upper quantile; `P(L ≥ VaR) ≥ 1−α`). Hand-checkable.
- **Historical ES_α = mean(L(1..k))** — average of the k largest losses. ⇒ `ES ≥ VaR` always; equality iff k=1 or ties.
- **Gaussian VaR_α = −μ + z_α·σ**, **Gaussian ES_α = −μ + σ·φ(z_α)/(1−α)**, μ/σ sample (ddof=1) of `pnl`; `z_α = NormalDist().inv_cdf(α)`.
- Multi-day: `value_h = μ-part × h + σ-part × √h` for Gaussian; historical `VaR_h = VaR_1 × √h` (labelled `scaling="SQRT_TIME"`; `distribution="EMPIRICAL"`). Diagnostics record `tail_size=k`, `n`.
- `min_obs` default **60** for 95% and **250** for 99% (k ≥ 3 at 99%): below that ⇒ `UNAVAILABLE` (reason with n and k). Between and 2× min ⇒ `DEGRADED` ("small tail: k=…").

Functions: `historical_var(pnl, confidence, horizon_days=1, *, min_obs=None) -> ModelResult`,
`historical_es(...)`, `gaussian_var(...)`, `gaussian_es(...)`, plus thin
`RiskModel` classes `HistoricalVaRModel`, `HistoricalESModel`,
`GaussianVaRModel`, `GaussianESModel` wrapping them (registered).

### 2.4 `risk/models/volatility.py` — covariance, portfolio σ, EWMA (spec §5 Tier 1, §12/§14 EWMA)

```python
def sample_covariance(matrix: ReturnMatrix, *, min_obs=60) -> CovarianceResult   # ddof=1, fsum; symmetric; ModelResult-like health
def shrunk_covariance(...)   # OUT OF SCOPE for Phase B (audit: P2 research) — do NOT implement now
def portfolio_volatility(pnl: Sequence[float], *, min_obs=60) -> ModelResult      # sample stdev of book P&L, USD/day; diagnostics: annualized_usd = σ·√252
def ewma_variance(returns: Sequence[float], *, lam=0.94, init_obs=20) -> list[float | None]
        # σ²_t = λσ²_{t−1} + (1−λ) r²_{t−1}; σ²_{init_obs} = sample var of first init_obs; None before; forecast for t uses returns < t (walk-forward safe)
def ewma_volatility_forecast(returns, *, lam=0.94, init_obs=20) -> ModelResult   # σ for the NEXT period, USD or return units matching input
def volatility_scaled_pnl(pnl, *, lam=0.94, init_obs=20) -> list[float]         # filtered HS: pnl_t × σ_now/σ_t (Hull-White); used by conditional VaR/ES
```
Conditional VaR/ES (spec §12 outputs) = the historical estimators of §2.3 applied to `volatility_scaled_pnl` — no separate estimator; `distribution="EMPIRICAL_VOL_SCALED"`, `params.lambda`.

### 2.5 `risk/models/contribution.py` — risk contribution (spec §9, §10, §33)

Inputs: `positions_pnl: Mapping[str, Sequence[float]]` (per position P&L series, same dates), portfolio `pnl_t = Σ_i pnl_{i,t}` (assert within 1e-6·scale).

- **Volatility contribution** `RC^σ_i = cov(pnl_i, pnl_p) / σ_p` (ddof=1). Property: `Σ_i RC^σ_i = σ_p` (exactly, up to fsum rounding ≤ 1e-9 relative).
- **ES contribution (Euler)** at α: with the tail set `T` = the k dates of the k largest portfolio losses (same k as §2.3), `RC^ES_i = mean_{t∈T}(−pnl_{i,t})`. Property: `Σ_i RC^ES_i = ES_α` **exactly** (same tail set, same k) — this is why §2.3 fixes ES as a plain tail average.
- **Marginal ES** of a candidate `c` at quantity q: `RC^ES_c(q)/q` per unit; **Incremental ES** `= ES(portfolio + candidate) − ES(portfolio)` recomputed on the joined series (Phase C consumes; the function lives here).
- Outputs: `ContributionResult(total, per_position: tuple[(key, contribution, share)], method="VOL"|"ES", confidence, tail_size, health)`, `share = contribution/total` (None if total ≤ 0). Ties in the tail are resolved by date order (stable, deterministic).

### 2.6 `risk/models/diagnostics.py` — distribution diagnostics (spec §15)

Sample skewness `g1 = m3/m2^{3/2}`, excess kurtosis `g2 = m4/m2² − 3` (population moments about the mean, documented), Jarque–Bera `JB = n/6·(g1² + g2²/4)`, **p-value closed form for χ²(2): `p = exp(−JB/2)`** (no incomplete gamma needed).
Labels (params on `DistributionParams`: `min_obs=60`, `heavy_tail_kurtosis=1.0`, `left_skew=-0.5`, `normal_p=0.05`): flags `NORMAL_LIKE` (p ≥ normal_p and not heavy/skewed), `HEAVY_TAIL` (g2 > heavy_tail_kurtosis), `LEFT_SKEWED` (g1 < left_skew), `UNSTABLE` (n < min_obs or variance ≈ 0). Result carries `primary` (priority UNSTABLE > LEFT_SKEWED > HEAVY_TAIL > NORMAL_LIKE) **and** the full `flags` tuple, plus `gaussian_trust: "HIGH"|"REDUCED"|"LOW"` (LOW if HEAVY_TAIL or LEFT_SKEWED, REDUCED if p < normal_p only, HIGH otherwise) — the spec §15 "reduce trust in Gaussian VaR" signal.

### 2.7 `risk/models/ensemble.py` — model dispersion & model-risk state (spec §39, §40, §59)

`dispersion(views: Mapping[str, ModelResult]) -> DispersionResult(min_name, max_name, ratio = max/min over ACTIVE|DEGRADED positive values, flag "MODEL_DISPERSION_HIGH" if ratio > params.high_ratio (default 1.5), n_views)`. Never averages.
`model_risk_state(inputs) -> ModelRiskState(LOW|ELEVATED|HIGH, reasons)` from a rule table (params): HIGH if any FAILED, or ≥2 of {dispersion high, gaussian_trust LOW with a Gaussian view active, any UNAVAILABLE core view, sample DEGRADED}; ELEVATED if exactly one; LOW otherwise. Reasons list the real triggers.

### 2.8 `risk/models/drawdown.py` — NAV drawdown (spec §5 Tier 1, §45)

`drawdown(nav: Sequence[tuple[date, float]]) -> DrawdownResult(current_dd_pct, max_dd_pct, peak_date, trough_date, peak_nav, current_nav, n_obs, health)`; `dd_t = nav_t / running_max_t − 1` (≤ 0). n < 2 ⇒ UNAVAILABLE. Also `reconstructed_book_drawdown(pnl_series, nav_now)` labelled `"RECONSTRUCTED_CURRENT_BOOK"` (cumulative P&L of today's book over the window, honest label; not a real NAV history).

### 2.9 `risk/pnl_series.py` — book P&L construction (spec §8, §21; DELTA_LINEAR until Phase D)

```python
@dataclass(frozen=True)
class PositionRiskInput:            # what the gateway builder passes (replay-importable; no ORM types)
    key: str                        # unique per position row, e.g. "AAPL#12"
    ticker: str                     # underlying whose returns drive it
    instrument: str                 # InstrumentType value
    quantity: int                   # SIGNED: short legs / short stock negative
    multiplier: int                 # 1 stock, 100 options
    spot: float                     # underlying last close
    delta: float                    # per-share delta (stock 1.0; options from chain; short leg NEGATED by the caller as in greeks.py)
    max_loss: float

def position_pnl_series(pos: PositionRiskInput, returns: ReturnMatrix) -> list[float]
        # exposure = quantity × multiplier × delta × spot; pnl_t = exposure × r_simple_t  (method="DELTA_LINEAR")
def book_pnl_series(positions, returns) -> BookPnl(dates, per_position: dict[key, list[float]], total: list[float], method, tickers_missing: tuple[str,...])
```
A position whose ticker has no column in the matrix is EXCLUDED and named in
`tickers_missing` (honest gap; the snapshot health becomes DEGRADED). Returns
must be `SIMPLE` (else `ValueError`).

### 2.10 `risk/validation.py` — walk-forward VaR/ES backtest (spec §42, §43, §68)

`walk_forward(pnl, *, window, confidence, estimator: Callable[[Sequence[float]], ModelResult]) -> ForecastSeries` (forecast for t uses `pnl[t−window:t]` only — never `pnl[t]`).
`exceedances(forecasts, realized_pnl) -> ExceedanceReport(n, x, rate, expected_rate=1−α, kupiec_lr, kupiec_p, christoffersen_lr, christoffersen_p, clustered: bool, es_severity_ratio (mean realized loss on exceedance days ÷ mean forecast ES; None if no exceedances or no ES))`.
Kupiec POF: `LR = −2·ln[(1−p)^{n−x} p^x] + 2·ln[(1−x/n)^{n−x} (x/n)^x]`, **χ²(1) p-value closed form `p = erfc(sqrt(LR/2))`**. Christoffersen independence: standard 2-state Markov LR, χ²(1), same closed form. Edge cases (x=0 or x=n) handled with the 0·ln0 = 0 convention. Verdicts: `GREEN/YELLOW/RED` by Kupiec p (params 0.05 / 0.01) — Basel-style, documented.

### 2.11 `risk/snapshot.py` — typed portfolio risk snapshot (spec §45, §55)

`PortfolioRiskSnapshot` (frozen dataclass): `as_of: datetime`, `nav`, `cash`, `cash_reserved`, `gross_exposure`, `delta_adjusted_exposure`, `heat_pct`, `heat_state`, `volatility: ModelResult|None`, `var: Mapping[str, ModelResult]` (keys like `"HISTORICAL:0.95:1"`, `"GAUSSIAN:0.99:1"`, `"HISTORICAL_VOL_SCALED:0.95:1"`), `es: Mapping[...]`, `drawdown: DrawdownResult|None`, `greeks: PortfolioGreeks|None`, `contributions_vol`, `contributions_es`, `distribution: DistributionResult|None`, `dispersion`, `model_risk`, `correlation_state: str|None` (Phase C), `data_quality: DataQuality(as_of, oldest_bar, newest_bar, tickers_missing, n_obs, valid: bool, reasons)`, `model_health: Mapping[str, ModelHealth]`, `risk_state: str` (heat state today), `ttl: TtlPolicy(statistical_seconds=86400, greeks_seconds=120)`, `is_stale(now) -> bool`, `snapshot_version: str`.
`Mapping` fields are plain dicts of typed results — no untyped JSON. A `to_api_dict()` serialiser lives in the **gateway** (Phase B second half), not here.

---

## 3. Cross-module invariants (tests must pin these)

1. `ES_α ≥ VaR_α` for the historical pair on any series; Gaussian ES ≥ Gaussian VaR.
2. Monotone in confidence: `VaR_0.99 ≥ VaR_0.95`, `ES_0.99 ≥ ES_0.95` (historical and Gaussian).
3. `Σ_i RC^ES_i == ES_α` and `Σ_i RC^σ_i == σ_p` within `1e-9 × max(1, |total|)`.
4. Scaling: `k·pnl` ⇒ VaR/ES/σ/RC scale by `k`; adding a constant gain shifts Gaussian by it, historical VaR by it.
5. Walk-forward never touches `pnl[t]` when forecasting t (test with a sentinel spike).
6. Below `min_obs` ⇒ `value None`, `health UNAVAILABLE`, `reason` non-empty; no exception.
7. `log_returns` moved from `correlation.py` produces identical output on the existing test vectors; `realized_vol` unchanged (only imports may change).
8. `RiskModel.validate` never upgrades health.
9. Filtered-HS series with λ→1 collapses to plain HS (limit sanity), and with constant returns the EWMA σ is constant.
10. Kupiec: x=0 on n=250 at 99% is GREEN? (expected 2.5; x=0 ⇒ LR≈5.03 → p≈0.025 → YELLOW) — hand-checked numbers in tests, not just sign.

---

## 4. Versioning

Every estimator carries `model_version` starting at `"1.0.0"`. Any change to
an estimator's arithmetic bumps MAJOR; parameter-default changes bump MINOR.
The gateway persists `ModelMeta` with each stored metric (Phase B second
half, migration 018) so historical numbers stay reproducible (spec §44).

---

## 5. What Phase B explicitly does NOT do

No numpy. No shrinkage/robust covariance (P2 research). No GARCH (Phase E,
research). No EVT/copula (deferred/rejected). No production gate changes.
No change to `assess()`. Nothing in this library reads a database or the
network — the gateway builder feeds it plain sequences.

---

## 6. Phase B second half — gateway API contract (SHADOW)

`GET /api/portfolio/risk` gains two ADDITIVE top-level keys. Existing keys keep
their semantics. Both blocks are `null` only when there is no account at all
(the existing "no venue" branch); otherwise they are objects with honest
nulls inside. `*_pct` fields are FRACTIONS (0.0123 = 1.23 %); `*_usd` are USD;
VaR/ES are LOSSES (positive = money lost).

```jsonc
"statistical": {
  "mode": "SHADOW",                       // never alters a Tier 0 decision
  "snapshot_id": 123 | null,              // risk_snapshots.id when persisted (ON_DEMAND builds persist too)
  "snapshot_version": "b.1",
  "as_of": "2026-08-18T14:03:11+00:00",
  "stale": false,                         // TtlPolicy: statistical > 1 trading day old
  "pnl_method": "DELTA_LINEAR",           // FULL_REVAL arrives in Phase D
  "n_obs": 598, "window_start": "2024-03-21", "window_end": "2026-08-14",
  "data_quality": { "valid": true, "reasons": [], "tickers_missing": [], "keys_excluded": [] },
  "model_health": { "historical_var": "ACTIVE", "gaussian_es": "DEGRADED", ... },
  "model_risk": { "state": "LOW" | "ELEVATED" | "HIGH", "reasons": [ ... ] },
  "dispersion": { "ratio": 1.23, "high": false, "min_model": "gaussian_var", "max_model": "historical_var", "n_comparable": 3 } | null,
  "distribution": { "primary": "NORMAL_LIKE", "flags": ["NORMAL_LIKE"], "skew": -0.12, "excess_kurtosis": 0.8, "jarque_bera": 3.1, "jb_p": 0.21, "gaussian_trust": "HIGH", "n": 598 } | null,
  "volatility": { "value_usd": 1234.5, "pct_nav": 0.0124, "annualized_pct_nav": 0.197, "health": "ACTIVE", "reason": null, "sample_size": 598, "model_name": "portfolio_volatility", "model_version": "1.0.0" } | null,
  "var": [                                // one row per (model, confidence, horizon); ORDER: HISTORICAL 0.95, HISTORICAL 0.99, GAUSSIAN 0.95, GAUSSIAN 0.99, HISTORICAL_VOL_SCALED 0.95
    { "model": "HISTORICAL", "model_name": "historical_var", "model_version": "1.0.0", "distribution": "EMPIRICAL",
      "confidence": 0.95, "horizon_days": 1, "value_usd": 754.97, "pct_nav": 0.0076,
      "health": "ACTIVE", "reason": null, "sample_size": 598, "tail_size": 30, "scaling": null }
  ],
  "es":  [ ...same shape, model_name historical_es / gaussian_es / conditional... ],
  "contributions": {
    "es":  { "confidence": 0.95, "total_usd": 919.9, "health": "ACTIVE", "rows": [ { "key": "AAPL#12", "ticker": "AAPL", "instrument": "LONG_STOCK", "contribution_usd": 512.3, "share": 0.557, "capital_weight": 0.31 } ] } | null,
    "vol": { "total_usd": 1234.5, "health": "ACTIVE", "rows": [ ...same row shape ] } | null
  },
  "positions_excluded": [ { "key": "NVDA#3", "reason": "no delta (contract missing from today's chain)" } ],
  "correlation_state": { "normal_avg": 0.61, "current_avg": 0.84, "stress_avg": 0.91, "delta": 0.23, "state": "CONVERGING",
                         "n_pairs": 3, "n_obs_long": 250, "n_obs_short": 60, "n_obs_stress": 25,
                         "worst_pairs": [ ["AAPL", "MSFT", 0.88] ], "reason": null } | null,  // ADDED in Phase C (§7.4/§7.5); null with < 2 tickers
  "stress": {                             // ADDED in Phase D (§8.5). ALWAYS an object; the rows carry the honest nulls.
    "mode": "SHADOW", "catalogue_version": "d.1", "model_version": "1.0.0",
    "health": "ACTIVE" | "DEGRADED" | "UNAVAILABLE", "reason": "2 of 9 scenarios unavailable" | null,
    "n_stock_legs": 1, "n_option_legs": 2,
    "method_coverage": { "FULL_REVAL": 2, "DELTA_LINEAR": 1 },   // summed over the worst row's legs; the per-row counts are on each row
    "rows": [                             // catalogue ORDER: named historical windows, AUTO worst windows, then the research grid
      { "name": "Equity -10% / IV +40%", "kind": "HYPOTHETICAL", "validated": false,
        "pnl_usd": -4210.5, "pnl_pct_nav": -0.042,           // GAIN-POSITIVE: a stress LOSS is negative
        "loss_usd": 4210.5, "loss_pct_nav": 0.042,           // the VaR/ES sign (positive = money lost), derived
        "method_coverage": { "FULL_REVAL": 2, "DELTA_LINEAR": 1 },
        "health": "DEGRADED", "reason": "AAPL#12: no iv0 — priced DELTA_LINEAR (...)",
        "params": { "spot_shock": -0.10, "spot_shock_by_ticker": {}, "iv_shock": 0.40,
                    "iv_shock_source": "SPECIFIED" | "RV_PROXY", "days_forward": 0.0,
                    "uniform_beta_1": true, "source": "CATALOGUE", "validated": false, "notes": "..." } }
    ],
    "worst": { ...the same row shape... } | null,             // smallest pnl_usd among rows that produced a number
    "per_position": { "AAPL#12": -1200.0 },                   // the WORST row's per-leg P&L (spec §52 scenario loss per position)
    "positions_excluded": [ { "key": "NVDA#3", "reason": "contract ... missing from today's chain — no stress leg" } ]
    // NOTE: this is the STRESS view's own gap list, deliberately NOT merged into `statistical.positions_excluded`
    // (a position can be in the statistical book but have no stress leg, and vice versa).
  },
  "validation": {                         // ADDED in Phase E (§9.4). NULL until a validation run has been persisted —
                                          // an honest "never validated". READ from the newest persisted
                                          // `risk_model_backtests` rows; NEVER recomputed on a page read (a walk-forward
                                          // backtest is not a page-load cost, and a read-path number would silently
                                          // differ from the history shown beside it). Written by the SCHEDULED tick
                                          // (once per NY day, in the snapshot's transaction) and by
                                          // POST /api/risk/validation/run (no audit event, snapshot_id NULL).
    "mode": "SHADOW",
    "as_of": "2026-08-18T14:03:11+00:00",
    "window": 250,                        // rolling estimation window in OBSERVATIONS (spec §43: forecasts use only days < t)
    "min_forecasts": 60,                  // fewer usable pairs ⇒ the row is UNAVAILABLE with its reason
    "n_obs": 598,
    "rows": [                             // ORDER: historical 0.95, historical 0.99, gaussian 0.95, gaussian 0.99,
                                          // conditional (EWMA) 0.95, garch 0.95
      { "model_name": "historical_var", "model_version": "1.0.0", "distribution": "EMPIRICAL",
        "confidence": 0.95, "horizon_days": 1, "window": 250,
        "n_forecasts": 348, "exceedances": 20,
        "rate": 0.0575, "expected_rate": 0.05,     // FRACTIONS
        "kupiec_p": 0.52, "christoffersen_p": 0.31,
        "es_severity_ratio": 0.99,                 // mean realized loss ÷ mean forecast ES on exceedance days; > 1 = worse than ES said
        "verdict": "GREEN" | "YELLOW" | "RED" | "UNAVAILABLE",
        "health": "ACTIVE" | "DEGRADED" | "UNAVAILABLE",
        "reason": null,
        "mode": "SHADOW" },                        // "RESEARCH" on the garch row — strictly below SHADOW
      { "model_name": "garch_var", "distribution": "EMPIRICAL_GARCH_SCALED", "confidence": 0.95,
        "n_forecasts": 0, "exceedances": 0, "rate": null, "kupiec_p": null,
        "verdict": "UNAVAILABLE", "health": "UNAVAILABLE",
        "reason": "GARCH fit on the most recent 250-observation window is DEGRADED: ...",
        "mode": "RESEARCH" }
    ],
    "comparison": {                       // §63; null only when neither conditional row was persisted
      "ewma_kupiec_p": 0.29, "garch_kupiec_p": 0.40,
      "garch_christoffersen_p": 0.18, "garch_n_forecasts": 348,
      "preferred": "garch_var" | "conditional_var" | null,   // null when either p-value is missing; ties go to the incumbent EWMA
      "criterion": "RESEARCH: GARCH may move RESEARCH -> SHADOW only if, over at least 250 forecast days, its Kupiec p is at least EWMA's, its Christoffersen p is at least 0.05, and its diagnostics never FAILED in the window. The promotion itself is a user action, never automatic.",
      "criterion_met": false,
      "criterion_unmet_reasons": [ "n_forecasts=348 < 250 required forecast days" ],
      "promotion": "NONE — GARCH stays RESEARCH; promotion is a user action (§63)."
    }
  } | null
},
"drawdown": {
  "nav_series": { "n": 3, "since": "2026-08-18", "source": "risk_snapshots SCHEDULED" },
  "current_pct": -0.012 | null, "max_pct": -0.031 | null, "peak_date": "2026-08-18" | null, "trough_date": ... | null,
  "peak_nav": 100000.0 | null, "health": "ACTIVE" | "UNAVAILABLE", "reason": "n=1 < 2 observations" | null,
  "reconstructed": { "label": "RECONSTRUCTED_CURRENT_BOOK", "current_pct": -0.02, "max_pct": -0.09, "n_obs": 598, "health": "ACTIVE" } | null
}
```

`RISK_DECISION` audit details (orders.py) gain, additively:
`"quantity_requested"`, `"approved_quantity"`, `"budget_multiplier"`,
`"limits": {asdict(RiskLimits) scalars only}`, and under the existing
`"shadow"` dict a `"statistical"` key:
`{ "snapshot_id", "as_of", "model_risk_state", "dispersion_high",
"historical_var_95_1d_pct_nav", "historical_es_95_1d_pct_nav",
"gaussian_es_95_1d_pct_nav", "health": {...}, "note": "current-book view; proposed-book comparison arrives in Phase C" }`
(or `null` with a `note` when the snapshot could not be built). No decision changes.

Persistence: every ON_DEMAND / PRE_TRADE (execution mode) / SCHEDULED build
writes `risk_snapshots` + `risk_metrics` + `risk_contributions` rows in one
transaction; the read view never writes audit events. A background
`risk_snapshot_loop` (main.py pattern, interval setting
`risk_snapshot_interval_seconds`, default 1800, 0 disables — tests keep it
off) persists ONE `SCHEDULED` row per America/New_York calendar day. Live
drawdown reads the SCHEDULED rows' `nav` (last per day). `atm_iv_daily` is
upserted (best-effort, never fails the caller) wherever `chain_iv_summary`
runs with a session in hand.

Observability: gauge `risk_snapshot_age_seconds`, histogram
`risk_model_latency_seconds{stage}`, counters
`risk_snapshot_builds_total{trigger}`, `risk_snapshot_failures_total`.

**Deferred: incremental pre-trade read (spec §54).** `audit.md:282` proposed
that the pre-trade path READ the cached snapshot and compute only the
incremental part, rather than building a full snapshot in-request. It does
not: every `PRE_TRADE` build runs the whole pipeline inline. **This is a
recorded deviation, not an oversight**, and the compliance audit was right
that it had been going unrecorded — this paragraph is that record.

*Why inline compute is acceptable today.* The measured build is **13 ms** on
a 5-ticker × 600-observation book (final report; ~75× inside the latency
budget), and no heavy model runs in-request: GARCH is dormant below its
250-observation minimum, the walk-forward validation harness is driven by
the SCHEDULED tick and by `POST /api/risk/validation/run` (never by a page
load or a preview), and the option revaluation prices one candidate leg.
The correctness argument matters more than the latency one: building in
request means the candidate is compared against **the book Tier 0 just
judged**, on the same dates, the same NAV and the same chain resolution. A
cached-snapshot read would introduce a second book — and a pre-trade
comparison that silently disagrees with the decision beside it is a worse
failure than 13 ms.

*What changes at promotion.* Two triggers, either of which makes the
incremental read load-bearing: (1) a heavier model is promoted into the
pre-trade path (GARCH becoming live on a longer history is the concrete
case — a refit per preview is not a 13 ms operation); or (2) the statistical
caps gain decision authority, at which point the build sits on the critical
path of an order rather than beside it. At that point the design is the one
`audit.md:282` names: read the newest snapshot, verify it against the §55
TTL (see §7.5 below — the staleness rule is already built), and compute only
the candidate's marginal contribution. The §55 consumer landing now is
deliberate: it is the piece the incremental read would depend on.

---

## 7. Phase C — pre-trade portfolio risk (SHADOW → PRODUCTION after the Q3 window)

Spec §8, §9, §11, §14, §19, §33, §37, §38, §46, §47, §70; audit §10 "Phase C".
Everything below is SHADOW: the hypothetical verdicts are computed, logged and
displayed; `assess()` decisions do not change unless a caller explicitly passes
`extra_caps` (production wiring is a separate, user-promoted step).

### 7.1 `risk/pretrade.py`

```python
@dataclass(frozen=True)
class CandidateSpec:            # the proposed entry, per UNIT of quantity
    key: str                    # e.g. "AAPL#candidate"
    ticker: str                 # underlying whose returns drive it
    instrument: str             # InstrumentType value
    multiplier: int             # 1 stock, 100 options
    spot: float
    delta: float                # per share, SIGNED (short stock −1; short premium negated)
    max_loss_per_unit: float    # Tier 0 risk basis per unit (stop×gap for stock, premium×100, net debit×100, (strike−credit)×100, 0 for CC)
    capital_per_unit: float     # cash outlay / reservation per unit
    quantity_requested: int

def proposed_book(book: BookPnl, candidate: CandidateSpec, quantity: int, returns: ReturnMatrix) -> BookPnl
    # current per-position series + candidate series (position_pnl_series on a PositionRiskInput built from the spec × quantity); ValueError if quantity < 0; candidate ticker missing from the matrix -> the returned BookPnl names it in tickers_missing/keys_excluded (caller reports UNAVAILABLE)

@dataclass(frozen=True)
class MetricPair: before: ModelResult | None; after: ModelResult | None; delta_usd: float | None; delta_pct_nav: float | None

@dataclass(frozen=True)
class RiskComparison:           # spec §46 CURRENT vs AFTER TRADE, all at ONE quantity
    quantity: int
    heat_pct: tuple[float, float]                # Tier 0 numbers (caller supplies before/after)
    cash_pct: tuple[float, float]
    var_hist_95: MetricPair; es_hist_95: MetricPair; var_hist_99: MetricPair; es_hist_99: MetricPair
    gaussian_es_95: MetricPair; volatility: MetricPair
    incremental_es_95_usd: float | None          # ES_after − ES_before (same k rule on the joined series)
    incremental_es_95_pct_nav: float | None
    marginal_es_95_per_unit: float | None        # Euler ES contribution of the candidate ÷ quantity
    candidate_es_share_after: float | None       # candidate's share of ES-95 contributions AFTER
    max_single_es_share_before: float | None; max_single_es_share_after: float | None
    bucket_es_share_after: Mapping[str, float]   # per correlation bucket the candidate belongs to (static + dynamic names)
    net_delta_notional: tuple[float | None, float | None]
    health: ModelHealth; reason: str | None      # UNAVAILABLE when the candidate ticker has no returns or n_obs < min_obs

def compare(book: BookPnl, candidate: CandidateSpec, quantity: int, *, returns: ReturnMatrix, nav: float, heat_before: float, heat_after: float, cash_before: float, cash_after: float, buckets: Mapping[str, Sequence[str]], delta_notional_before: float | None) -> RiskComparison
```
Rules: `before` metrics are computed on `book.total`; `after` on `proposed_book(...).total`; both with the SAME `min_obs`, `k`, and horizon (1D). ES contributions AFTER are computed on the joined per-position dict (candidate key included). Ties/health propagate: if either side is not ACTIVE/DEGRADED the pair's deltas are `None`.

### 7.2 Statistical limits and hypothetical caps (spec §11, §27 later, §37)

```python
@dataclass(frozen=True)
class StatisticalLimits:        # RESEARCH DEFAULTS — UNVALIDATED (audit §11 Q3); SHADOW until promoted
    max_portfolio_es95_pct_nav: float = 0.05         # ES-95 1D of the whole book ≤ 5 % NAV
    max_single_position_es_share: float = 0.35       # any one position ≤ 35 % of ES-95 contributions
    max_bucket_es_share: float = 0.50                # any correlation bucket ≤ 50 %
    max_incremental_es95_pct_nav: float = 0.015      # one trade may add ≤ 1.5 % NAV of ES-95
    min_obs: int = 60
    mode: str = "SHADOW"                             # SHADOW | PRODUCTION (only PRODUCTION may be wired into assess)

@dataclass(frozen=True)
class QuantityCap:              # the shape assess(extra_caps=...) understands
    code: str                   # e.g. "PORTFOLIO_ES_LIMIT", "ES_CONTRIBUTION_CAP", "BUCKET_ES_CONTRIBUTION_CAP:TECH_MEGA", "INCREMENTAL_ES_CAP"
    layer: str                  # "STATISTICAL" | "CONCENTRATION"
    cap_qty: int                # largest quantity in [0, requested] that satisfies the limit (0 => would REJECT)
    sentence: str               # real numbers, spec §47 style
    measured: Mapping[str, float | None]   # the values at requested qty and at cap_qty

def statistical_caps(book, candidate, *, returns, nav, buckets, limits=StatisticalLimits()) -> tuple[list[QuantityCap], ModelHealth, str | None]
```
Cap search: for each limit, evaluate at `q = requested`; if satisfied → no cap. Else **bisection on q ∈ [0, requested]** (≤ 20 steps; the metrics are evaluated by recomputing the joined series — cheap on ≤ 600 obs) assuming monotone-in-q; if the check at `cap_qty` still fails (non-monotone corner), step down until it passes or 0. Health UNAVAILABLE (no caps, reason) when the candidate has no returns or `n_obs < min_obs` — a missing statistical view NEVER produces a cap (fail-open in SHADOW; the PRODUCTION promotion design decides fail-closed rules — recorded as an open item).

```python
@dataclass(frozen=True)
class ShadowVerdict:
    hypothetical_decision: str          # APPROVE | APPROVE_WITH_RESIZE | REJECT — what the STATISTICAL layer alone would have done at the Tier 0 approved qty
    hypothetical_quantity: int          # min(approved_qty, min cap_qty)
    binding: tuple[str, ...]            # cap codes that bind at the approved qty, most restrictive first
    caps: tuple[QuantityCap, ...]
    mode: str                           # "SHADOW"
def shadow_verdict(approved_qty: int, caps: Sequence[QuantityCap]) -> ShadowVerdict
```

### 7.3 `risk/engine.py` — additive only

- `assess(..., extra_caps: Sequence[QuantityCap] = ())`: when non-empty, each cap is applied as a `clamp(qty, cap.cap_qty, cap.code, cap.sentence)` AFTER step 5e (cash floor) and BEFORE step 5f (greeks). Defaults ⇒ byte-identical behaviour (existing 45+16 tests untouched). The greek limits stay REJECT (audit §11 Q5 default).
- `RiskAssessment` gains optional fields with defaults: `requested_quantity: int | None = None`, `binding_constraints: tuple[BindingConstraint, ...] = ()` where `BindingConstraint(code, layer)`; layer for existing codes = "HARD_LIMIT" (kill switch / heat / caps / cash floor / greeks), extra caps carry their own layer. Populated for every decision from `reason_codes` (a pure mapping — no behaviour change).

### 7.4 Correlation regime (spec §19) — `libs/trading_core/correlation.py`, additive

`correlation_regime(matrix: ReturnMatrix(LOG), *, params=CorrelationRegimeParams(long_window=250, short_window=60, stress_quantile=0.10, elevated_delta=0.05, converging_delta=0.15, converging_level=0.80, min_pairs=1)) -> CorrelationState(normal_avg: float|None, current_avg: float|None, stress_avg: float|None, delta: float|None, state: "NORMAL"|"ELEVATED"|"CONVERGING"|"UNAVAILABLE", n_pairs: int, n_obs_long: int, n_obs_short: int, worst_pairs: tuple[(a, b, current_rho), ...] (top 3 by current), reason: str|None)`. Averages are over the upper triangle of the pairwise Pearson matrix (existing `_pearson`); `stress_avg` = pairwise correlation over the worst `stress_quantile` days of the equal-weight portfolio return of the same tickers (≥ 10 days else None).

**Rolling Spearman — UPDATED 2026-08-19 (compliance §3 row 18): now BUILT** (this sentence previously read "stays RESEARCH (not built)", which is why the three docs disagreed). It remains RESEARCH in the §70 sense — a display diagnostic that gates nothing — but the estimator exists: `spearman(a, b)` (rank-transform with ties sharing their average rank, then the existing `_pearson`), `rolling_spearman(a_closes, b_closes, window=60)`, `rolling_spearman_matrix(closes_by_ticker, window=60) -> {(a, b): float|None}` and `rolling_spearman_average(...) -> (float|None, n_pairs)`, all in `libs/trading_core/correlation.py`. `CorrelationState` additionally carries `current_avg_spearman: float|None`, computed over the SAME short window with the SAME upper-triangle convention (None on insufficient data). It is a **non-field attribute** of the frozen dataclass (a `ClassVar` set per-instance via `with_spearman`), because `dataclasses.asdict` — the serialiser behind `correlation_state_api` — would otherwise widen the §19 wire contract as a silent side effect of computing a number. **UPDATED (integration, same day): it is now ALSO on the wire**, as the twelfth key of `statistical.correlation_state`. That was the explicit step this paragraph anticipated: `correlation_state_api` reads the attribute directly (`getattr`, since `asdict` cannot see a non-field) and the key-for-key gateway test in `tests/test_orders_shadow_c.py` was updated in the same change. The attribute stays non-field so `==`/`hash`/`repr` and `dataclasses.replace` are unchanged. **No state rule reads it** — `state` is byte-identical with or without it.

### 7.5 Gateway (SHADOW)

- `orders.py` RISK_APPROVAL: build `CandidateSpec` from the chosen instrument (stock: delta ±1, risk basis = risk_stop, capital = entry; option/spread: delta from the selected contract(s) net, risk basis = risk_stop, capital = risk_entry; short stock: delta −1) → `compare()` at the Tier 0 approved qty (and at requested when they differ) → `statistical_caps()` → `shadow_verdict(approved_qty, caps)`. Log under `shadow.statistical`: `comparison` (a §46 table: rows heat/cash/VaR95/ES95/σ/incremental ES/candidate ES share/bucket shares with before/after), `caps`, `hypothetical` (decision, quantity, binding), `limits` (asdict StatisticalLimits), `correlation_state`. Response `risk` block gains `comparison`, `binding_constraints`, `shadow_statistical` (same content) so previews / trade plans carry it. Decision unchanged; a raise → `note`.
- `plans.py`: nothing to change if the preview payload is what it stores — verify; the plan detail UI reads it from `preview.risk`.
- Vol targeting side-by-side (spec §14; audit): `vol_targeting_block` gains additive `ewma_sigma_p_annualized_pct_nav` (EWMA σ of the book P&L ÷ NAV × √252) and `multiplier_ewma` (same `exposure_multiplier` clamps); RISK_DECISION `shadow.vol_targeting_ewma = {forecast, multiplier, note}`; the multiplier actually used is unchanged.
- `/api/portfolio/risk`: `statistical.correlation_state` (the §7.4 object) — additive.

**Pre-promotion staleness rule (spec §55) — CHOSEN AND BUILT.** Until
compliance batch 3, `PortfolioRiskSnapshot.is_stale` was complete, tested to
its boundary (exactly TTL ⇒ fresh, TTL + 1 s ⇒ stale) and **consumed by
nothing but the serialiser**. It now has a consumer inside the shadow layer:

> When the `PRE_TRADE` build's snapshot is stale per its own `TtlPolicy`
> (`statistical` kind — these caps derive from daily-close VaR/ES, not from
> live greeks), the shadow verdict becomes `hypothetical_decision =
> **UNAVAILABLE_STALE**`, every hypothetical cap is **suppressed**, and the
> verdict carries a `reason` naming the real age and TTL.
> `hypothetical_quantity` falls back to the Tier 0 approved quantity — which
> is what actually happened. The caps are still returned in `caps` (they were
> computed; hiding the evidence would defeat the shadow window), but
> `binding` is empty because nothing bound.

`UNAVAILABLE_STALE` is deliberately **not** one of the three Tier 0 decision
words: a stale statistical view has not decided to approve, resize or reject
— it has failed to answer, and reporting `APPROVE` would be exactly the
fail-open this vocabulary exists to make visible.

*Why suppress rather than apply.* A cap derived from a stale book is a
statement about positions that may have changed. Applying it would let an
out-of-date measurement reduce a quantity, which is a worse error than
declining to answer.

**SHADOW-ONLY, and Tier 0 is pinned.** `assess()` receives no `extra_caps` at
either production call site, so suppressing shadow caps removes a
*hypothetical*, never a control. `tests/test_risk_compliance_batch3.py`
pins this with the sabotage pattern: with every snapshot forced to report
itself stale, the whole Tier 0 half of a preview (decision, approved
quantity, gates, binding constraints, reason, Tier 0 comparison rows) is
byte-identical, while the shadow half flips to `UNAVAILABLE_STALE`.

**FAIL-CLOSED AT PROMOTION IS A SEPARATE USER DECISION — NOT MADE HERE.**
The rule above is the *pre-promotion* rule and is safe precisely because it
is shadow-only. Once the statistical caps gain decision authority, the same
seam must be re-answered as a policy question: does a stale snapshot
**refuse the trade** (fail closed) or **let it through un-capped** (fail
open, today's behaviour by omission)? This document does not choose that;
`§3 Tier B` of the compliance report and step 3 of its promotion path list
it, together with the parallel question for `UNAVAILABLE` views, as a
decision the user must make **before** any `extra_caps` line is populated.

### 7.6 UI

- Preview / Trade Plan panel (`ui/app/watchlist/[ticker]/page.tsx` proposed-sizing area, and the plan detail if it renders `preview.risk`): "CURRENT vs AFTER TRADE" table (spec §46) with SHADOW badge, `binding constraints` list (code + layer, HARD_LIMIT vs STATISTICAL/CONCENTRATION), hypothetical statistical verdict with quantity, requested vs approved.
- Risk page: correlation-state pill (normal/current/stress averages, worst pairs) in the buckets panel; vol-targeting line shows the EWMA forecast beside the crude proxy, labelled.

### 7.7 Tests
Hand-computed `compare` on a 3-position × 8-day book + candidate; incremental ES == ES(after) − ES(before) exactly; marginal ES = candidate Euler RC ÷ q; cap bisection finds the largest passing q on a monotone case and steps down on a constructed non-monotone case; UNAVAILABLE ⇒ no caps; `shadow_verdict` ordering; `assess(extra_caps=())` byte-identical (md5 of a decision battery before/after), `extra_caps` clamp + code + layer; `binding_constraints` mapping total; correlation regime hand-checked incl. stress window; API: preview `risk.comparison` + `shadow_statistical` present, decision unchanged with the shadow layer raising; RISK_DECISION shadow keys; vol-targeting side-by-side; UI render tests for the comparison table with nulls.

---

## 8. Phase D — stress engine + option full revaluation (SHADOW)

Spec §21–§27, §51–§52; audit §7.4, §10 "Phase D". SHADOW: stress numbers are
computed, persisted, displayed and logged as a hypothetical STRESS cap;
nothing decides.

### 8.1 `options/iv.py` — implied vol solver (INTERNALLY CALCULATED)
`implied_vol(price, spot, strike, t_years, right, *, r=0.04, q=0.0, lo=1e-4, hi=5.0, tol=1e-8, max_iter=100) -> IVResult(iv: float|None, iterations, converged: bool, reason: str|None, method="BISECTION")` — bisection on `bs_price` (monotone in σ). Guards: `price ≤ intrinsic` or `t_years ≤ 0` ⇒ `iv=None` with reason; `price ≥ bs_price(hi)` ⇒ None ("above the σ=5.0 ceiling"). Labelled internally calculated (provenance rule) — never presented as vendor IV.

### 8.2 `options/reval.py` — leg-aware scenario revaluation, basis-anchored
```python
@dataclass(frozen=True) class OptionLeg: key, ticker, right, strike, t_years, quantity (SIGNED contracts), multiplier=100, mark0 (per share), iv0: float|None, r=0.04, q=0.0
def leg_baseline(leg) -> LegBaseline(model0, basis = mark0 − model0)      # basis held CONSTANT across scenarios so scenario 0 ⇒ P&L exactly 0
def reval_leg(leg, *, spot0, spot1, iv1, days_forward) -> float            # price1 = bs_price(spot1, K, max(T0 − days/365, 0), iv1) + basis; T ≤ 0 ⇒ intrinsic (no basis)
def scenario_pnl(stock_legs, option_legs, *, spot0_by_ticker, spot_shock_by_ticker (fractional), iv_shock (RELATIVE multiplicative on the IV LEVEL: +0.20 ⇒ iv1 = iv0 × 1.20), days_forward) -> ScenarioPnl(total, per_key, method_by_key: FULL_REVAL | DELTA_LINEAR (fallback when iv0 is None — labelled), notes)
```
Stock legs: `qty × spot0 × shock`. Spreads/income are lists of legs (short legs negative qty). Property: P&L is linear in each leg's quantity; scenario (0 shock, 0 IV, 0 days) ⇒ 0.0 exactly.

### 8.3 `risk/models/stress.py` — scenarios, catalogue, results, cap
- `Scenario(name, kind: HISTORICAL|HYPOTHETICAL|IV_GRID|USER, spot_shock (uniform, applied to every underlying — β=1, documented) | spot_shock_by_ticker, iv_shock, days_forward, validated: bool, source: str, notes)`.
- Historical windows: `HistoricalWindow(name, start: date, end: date)`; shocks = per-ticker cumulative simple return over the window from stored closes (product of 1+r); IV shock = realized-vol ratio proxy `RV(window)/RV(prior 20d) − 1` clipped to [−0.5, +2.0], labelled `iv_shock_source="RV_PROXY"` (no IV history — spec §24 honesty). Catalogue: "2024-08-05 vol spike" (2024-07-31→2024-08-05), "2025-04 tariff drawdown" (2025-04-02→2025-04-08), plus AUTO windows found in the stored history: worst 1-day, worst 5-day, worst 10-day of the equal-weight book (`auto_worst_windows`). Windows outside stored history ⇒ UNAVAILABLE row with reason.
- Hypothetical catalogue (research grid, `validated=False`, all UNVALIDATED): "Equity −5% / IV +20%", "Equity −10% / IV +40%", "Equity +5% / IV −15%", "IV crush (flat, −40%)", "IV spike (flat, +50%)", "Correlation convergence (all names −8%, IV +30%)", "Time decay only (+5 days)".
- `run_stress(stock_legs, option_legs, scenarios, *, spot0_by_ticker, closes_by_ticker, nav) -> StressResult(rows: tuple[ScenarioResult(name, kind, validated, pnl_usd, pnl_pct_nav, per_key, method_coverage: {FULL_REVAL: n, DELTA_LINEAR: n}, health, reason, params)], worst: ScenarioResult|None, health, min_pnl_usd)`.
- `StressLimits(max_stress_loss_pct_nav=0.10 — research default UNVALIDATED, mode="SHADOW")`; `stress_caps(candidate legs per unit, book legs, scenarios, requested_qty, nav, limits) -> QuantityCap("STRESS_LOSS_LIMIT", layer="STRESS")` via the same bisection helper style as Phase C (worst scenario re-evaluated at each q). Health UNAVAILABLE ⇒ no cap (SHADOW fail-open, same open item).

### 8.4 Persistence — `migrations/019_stress_runs.sql` + ORM `StressRunRow`
`stress_runs(id, snapshot_id FK→risk_snapshots ON DELETE CASCADE, scenario VARCHAR(64), kind VARCHAR(16), validated BOOLEAN, pnl_usd DOUBLE PRECISION NULL, pnl_pct_nav NULL, method_full_reval INTEGER, method_delta_linear INTEGER, health VARCHAR(16), reason TEXT NULL, params JSONB, per_position JSONB, as_of TIMESTAMPTZ, created_at)`; compose mount; ORM mirror test entry.

### 8.5 Gateway
- `risk_snapshot.py`: build `OptionLeg`s for open option/spread/income positions from the SAME chain resolution the risk view uses (`find_option_contract` mid as `mark0`, provider IV as `iv0`, DTE→t_years; short legs negative qty; income legs negative; spread short leg from `short_strike`/`short_occ_symbol`), stock legs from stock positions (short stock negative qty); run the catalogue on every build; persist rows; API `statistical.stress` = {rows (name, kind, validated, pnl_usd, pnl_pct_nav, method_coverage, health, reason), worst, health, catalogue_version}. Pre-trade (`orders.py`): candidate legs per unit → `shadow.statistical.stress` {worst_before, worst_after (at approved qty), cap, hypothetical} and the STRESS cap merged into the existing shadow verdict/binding list (SHADOW).
- `POST /api/risk/stress/run` (user-defined hypothetical, spec §26/§51): body {equity_shock: float in [−0.9, 2], iv_shock: float in [−0.9, 5], days_forward: int 0..365, name?} → runs on the current book, persists a `USER` row (kind USER, validated False), returns the ScenarioResult; USER actor audit? No — it is a read of the book under a hypothesis; write NO audit event, but DO persist the stress_runs row (history, spec §56).
- Positions API rows for options gain `premium_at_risk`, `dte`, greeks (already), `iv0`, `worst_scenario_pnl` (spec §52) — additive.

### 8.6 UI
Risk page "Stress Scenarios" panel (table: scenario, kind badge, UNVALIDATED badge for the research grid, P&L $, % NAV, method coverage, health/reason; worst highlighted; user-defined scenario form (equity %, IV %, days) → runs `POST /api/risk/stress/run` and appends the row); positions panel option rows show premium at risk / DTE / vega $ / worst scenario loss; TradeComparison gains a "Worst stress loss" row + STRESS layer constraints. Bilingual, glossary (stress_test, full_revaluation, iv_crush, basis_adjustment).

### 8.7 Tests
IV solver round-trip vs `bs_price` across a strike/vol/tenor grid (|Δσ| < 1e-6), intrinsic/ceiling guards; reval: zero scenario ⇒ 0.0 exactly, basis held, expiry ⇒ intrinsic, spread legs net, income short leg sign, stock linear; historical shocks from a hand-built close path; auto worst windows; catalogue rows shape; stress cap bisection vs brute force; monotone: |P&L| non-decreasing in |q| for a single leg (spec §67 property); API `statistical.stress` keys + persistence rows + USER scenario endpoint validation (422 on out-of-range) + no audit; pre-trade `shadow.statistical.stress` and Tier 0 byte-identity when the stress layer raises; UI render tests.

---

## 9. Phase E — conditional volatility (GARCH research) + VaR/ES model validation (SHADOW/RESEARCH)

Spec §12–§14, §42–§43, §57, §59, §63, §68; audit §7.5, §10 "Phase E".

### 9.1 `risk/optim.py` — Nelder–Mead (stdlib, deterministic)
`nelder_mead(f, x0, *, step=0.1, tol=1e-8, max_iter=2000) -> NMResult(x, fval, iterations, converged, reason)` — standard reflection/expansion/contraction/shrink; deterministic; no randomness. Tests: quadratic and Rosenbrock minima.

### 9.2 `risk/models/_chi2.py` — χ² survival function for any df
`regularized_gamma_p(a, x)` (series for x < a+1, continued fraction otherwise — Numerical-Recipes `gammp`/`gammq`), `chi2_sf(x, df) = gammq(df/2, x/2)`. Tests: df=1 vs `erfc(sqrt(x/2))`, df=2 vs `exp(-x/2)`, df=10 known table values (e.g. sf(18.307, 10) ≈ 0.05).

### 9.3 `risk/models/garch.py` — GARCH(1,1), Gaussian innovations, RESEARCH
- `GarchParams(omega, alpha, beta)`; constraints ω>0, α≥0, β≥0, α+β<1 enforced through an unconstrained transform (softplus / logistic) inside the objective; init from EWMA (α=0.06, β=0.90, ω=(1−α−β)·sample var).
- `fit_garch(returns, *, min_obs=250, max_iter=2000) -> GarchFit(params, loglik, converged, iterations, persistence=α+β, unconditional_var, sigma2_series (in-sample conditional variances), std_residuals, diagnostics: {ljung_box_q_sq (m=10 lags on standardized residuals²), ljung_box_p, n, half_life}, health, reason)`. Health: UNAVAILABLE n < min_obs; DEGRADED if not converged, persistence ≥ 0.999, or Ljung–Box p < 0.05 (params); FAILED on numeric error. Never raises for data problems.
- `garch_forecast_variance(fit, h) -> list[float]` closed form σ²_{t+k} = VL + (α+β)^{k−1}(σ²_{t+1} − VL); `garch_volatility_forecast(returns, horizon_days) -> ModelResult` (σ over the horizon = sqrt(Σ_k σ²_{t+k}); model_name "garch11", version 1.0.0, distribution "GAUSSIAN_GARCH", mode RESEARCH); `garch_scaled_pnl(pnl)` (FHS with GARCH σ, same shape as `volatility_scaled_pnl`) so `historical_var/es` over it give conditional VaR/ES (distribution "EMPIRICAL_GARCH_SCALED"). Fallback rule (spec §13/§58): when GARCH health is not ACTIVE, the conditional views come from EWMA (already present) — the snapshot names which one was used.
- Tests: recovers (α, β) on a seeded simulated GARCH series (n=3000; |Δα| < 0.03, |Δβ| < 0.05); constraint respect; persistence/half-life arithmetic; DEGRADED paths; forecast closed form vs iterative; fallback selection.

### 9.4 Validation persistence + run — spec §42/§43 (walk-forward, no hindsight)
- `migrations/020_risk_model_backtests.sql` + ORM `RiskModelBacktestRow`: (id, as_of, snapshot_id FK NULL, model_name, model_version, distribution, confidence, horizon_days, window, n_forecasts, exceedances, rate, expected_rate, kupiec_lr, kupiec_p, christoffersen_lr, christoffersen_p, es_severity_ratio, verdict, health, reason, params JSONB, created_at); compose mount; mirror test.
- `apps/gateway/risk_validation.py`: `run_model_backtests(session, *, book_pnl, snapshot_id) -> list[rows]` — walk-forward (window 250, min 60 forecasts) for historical VaR 95/99, Gaussian VaR 95/99, EWMA-filtered VaR 95, GARCH-filtered VaR 95 (RESEARCH; skipped with reason if GARCH unavailable) using `risk/validation.py`; ES severity from the matching ES estimator; persist rows; called from the SCHEDULED build only (daily) and from `POST /api/risk/validation/run` (on demand, persists, no audit); API `statistical.validation` = {as_of, rows [{model_name, confidence, horizon_days, n_forecasts, exceedances, rate, expected_rate, kupiec_p, christoffersen_p, es_severity_ratio, verdict, health, reason}], comparison: {ewma_vs_garch: {ewma_kupiec_p, garch_kupiec_p, preferred, criterion}} — read from the newest persisted rows (never recomputed on a page read).
- Model-risk rule table gains a param `backtest_red_triggers` (a RED verdict on a core view is one trigger) — additive.
- §63 comparison criterion (documented, research): GARCH may move RESEARCH → SHADOW only if, over ≥ 250 forecast days, its Kupiec p ≥ EWMA's, its Christoffersen p ≥ 0.05, and diagnostics never FAILED in the window; recorded in `comparison.criterion` — the promotion itself is a user action.

### 9.5 UI
Risk page "Model validation (VaR backtests)" panel: table per model (n, exceedances vs expected, rate, Kupiec p, Christoffersen p, ES severity, GREEN/YELLOW/RED badge, health/reason), the EWMA-vs-GARCH comparison line with the criterion sentence, "Run now" button (POST, inline errors), SHADOW/RESEARCH badges; the conditional-vol tile shows which forecaster is active (EWMA / GARCH-research) with its methodology modal.

### 9.6 Tests
optim, chi2, garch (as above); validation runner walk-forward sentinel; persistence rows; API keys + honest nulls (< 60 forecasts ⇒ UNAVAILABLE with reason); no audit on the endpoint; SCHEDULED build writes rows once/day; UI render tests.

---

## 10. Compliance batch 2 — full revaluation in the VaR/ES P&L series (spec §21–§22; SHADOW)

Closes the compliance matrix's §21 gap: portfolio VaR/ES, contributions, incremental/marginal ES and the shadow caps currently price options DELTA_LINEAR; long-gamma/vega convexity is invisible outside stress. This batch makes the BOOK P&L series scenario-driven where the data allows, per leg, labelled — and stays SHADOW.

### 10.1 Estimator (documented, hand-checkable)
For an option leg with (mark0, iv0, T0, K, right, spot0) — the same fields the stress `OptionLeg` carries — the historical 1-day P&L at return r_t is
`pnl_t = [ BS(spot0·(1+r_t), K, T0, iv0) + basis ] − [ BS(spot0, K, T0, iv0) + basis ]`
(basis = mark0 − model0 cancels exactly; T held at T0 — the series measures S-convexity only, CONST_IV documented; iv0 held constant — no IV history yet, labelled). Method per leg: `FULL_REVAL_CONST_IV`; fallback when iv0 is None: `DELTA_LINEAR` (existing), labelled per position. Stock legs unchanged (exact linear). Reuses `options/reval.leg_baseline` + `bs_price` — never a re-implementation.

### 10.2 Library
`risk/pnl_series.py` — additive: `PositionRiskInput` gains optional leg fields (`strike, right, t_years, iv0, mark0` — None ⇒ old behaviour), `position_pnl_series` dispatches on their presence; `BookPnl.method` becomes the book-level summary (`FULL_REVAL_CONST_IV` when ≥1 leg full-revals, with `method_by_key`). Invariants: for a stock leg identical output byte-for-byte; for an option leg with iv0, pnl is CONVEX in r (call: pnl(+r)+pnl(−r) ≥ 0) and `pnl_t → delta·spot·multiplier·qty·r_t` as r→0 (first-order agreement with the old series, test at r=1e-6 rel 1e-3).

### 10.3 Gateway + surfaces
`risk_snapshot.py` already builds stress OptionLegs from the chain — pass the same leg data into the PositionRiskInputs so ES/RC/incremental/caps/validation all inherit the new series automatically (they consume `book.per_position/total`). API: `statistical.pnl_method` becomes `FULL_REVAL_CONST_IV` when any leg full-revals (else DELTA_LINEAR), plus `pnl_method_by_key` in data_quality or positions_excluded-style listing. Persisted rows: `risk_snapshots.pnl_method` reflects it. Pre-trade candidate legs likewise (option candidates get their selected contract's iv/mark). UI: the methodology modal + pnl_method chip already render the served string — verify, add the CONST_IV explainer to the glossary.

### 10.4 Invariants & tests
Stock-only book: every number byte-identical before/after (the strongest regression pin — run the existing seeded API test and diff). Option book: hist ES(full reval) ≥ hist ES(delta-linear) is NOT guaranteed in general — do not assert it; instead assert convexity (long option book's mean pnl over symmetric ±r grid > 0) and that RC still sums exactly to ES (the Euler property is method-agnostic). SHADOW: preview decisions byte-identical when the new path raises (existing sabotage pattern). Latency: builder on 5×600 book with 3 option legs < 1.5×  the delta-linear time (BS is ~µs; 600 obs × legs is trivial).
