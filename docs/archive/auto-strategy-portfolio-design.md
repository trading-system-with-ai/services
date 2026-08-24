# Auto-Strategy & Portfolio Backtest — program design (2026-08-20)

**User mandate (/loop):** score-banded instrument selection (>a calls /
a-b stock / b-c covered calls / c-d short / <d puts), squeeze detection,
risk control — "is this reasonable, what's the gap, implement it"; plus
AUTO decision-making in backtests and PORTFOLIO-level backtests over the
whole watchlist (daily per-symbol allocation + cash %), with instrument
permissions as a user multi-select.

## Verdict on the banded model (quant + architect, code-audited)

The model is sound and ~70% ALREADY EXISTS, with two institutional
refinements the platform already embodies and one it lacks:

1. The continuous score EXISTS: `directional_edge = bull_score −
   bear_score ∈ [−100,+100]` (signals/directional.py) — technicals +
   volume + trend features, weights versioned. The a/b/c/d bands EXIST:
   `strength_tier` at |edge| ≥ 25/40/60/80 (risk/engine.py:125-128).
   The banded instrument map EXISTS: the §8 matrix
   (strategies/instrument.py) = direction × tier × **IV regime** ×
   permissions → one of 8 instruments.
2. REFINEMENT #1 (keep): the matrix conditions option BUYING on IV —
   LONG_CALL only at STRONG+ bull AND LOW IV. Buying calls at HIGH/
   EXTREME IV pays up for volatility and is systematically −EV; a pure
   score-band model without IV awareness would do exactly that.
3. REFINEMENT #2 (keep): bear expression prefers defined-risk (LONG_PUT
   / BEAR_PUT_SPREAD); SHORT_STOCK fires only where premium is
   unbuyable (EXTREME IV, or no spreads). Unbounded loss + borrow +
   squeeze exposure make shorting the fallback, not the default.
4. GENUINE GAP (build): the NEUTRAL band does nothing today — the
   matrix never emits COVERED_CALL/CSP (income is a manual §-endpoint).
   The user's "neutral + holding stock → sell calls" is right and
   becomes the income overlay in AUTO mode.
5. LLM scores stay research-only (§25/§30 verified: no LLM number
   reaches sizing/instrument/execution). Deliberate; revisit only as a
   SHADOW overlay with its own validation.

## The two hard gaps (the build)

- **AUTO instrument switching in backtests** — today the user picks ONE
  leg per run; six engines share signals/exits/fills but each
  hand-rolls its own cash accounting; no engine re-selects the
  instrument as the score moves through bands.
- **Portfolio-level backtest** — nothing multi-symbol exists: single
  ticker per run, private INITIAL_EQUITY per engine, no shared cash, no
  allocation output.

## Squeeze (逼空) — honest data verdict

Real inputs (short interest, days-to-cover, borrow fee, float) are
UNAVAILABLE from every configured provider (audited; the deferral is
already recorded in execution-chains-roadmap.md §33). No proxy
substitution (§33/§44-18). What IS honestly buildable from stored bars:
a **crowding/momentum PROXY** — volume z-score(20d), distance from
252-day high, overnight gap-up — shipped as a REPORT-mode SQUEEZE_RISK
gate before SHORT_STOCK (same pattern as the LIQUIDITY gate), always
labeled a proxy. Veto promotion (live or backtest) is an OPEN USER
DECISION after watchlist validation — no veto shipped. A true detector needs a new §33-approved vendor
(FINRA/Ortex/S3) — recorded as an open user decision.

## Phases

- **A — audit + this design.** DONE 2026-08-20 (4-agent code audit).
- **B — AutoStrategyEngine (single-symbol).**
  `libs/trading_core/backtest/auto.py`: daily loop; signal → tier → vol
  regime → §8 `select_instrument` (same live code, §21); on decision
  change: exit held instrument (shared exit engines), enter new at next
  open (shared fill model). Scope: LONG_STOCK, SHORT_STOCK, LONG_CALL,
  LONG_PUT + COVERED_CALL overlay when NEUTRAL while holding stock.
  `permissions` override = the user's multi-select (validated against
  the real §5 account flags — the backtest may RESTRICT, never exceed).
  API: instrument "AUTO" on POST /api/backtests. Tests incl. §96-style
  look-ahead checks.
- **C — PortfolioBacktestEngine.**
  `libs/trading_core/backtest/portfolio.py`: N tickers, ONE cash
  ledger; per-day: decide per symbol (Phase B engine logic), size by
  the LIVE §12 risk budgets (0.5/0.75/1.0/1.25% of the PREVIOUS bar's
  marked equity per tier — same-morning marks would be look-ahead, ATR-stop-distance sizing — same constants as orders.py), cash
  floor + max concurrent positions; capital contention resolved by
  |edge| priority (documented, deterministic). Outputs: daily
  per-symbol allocation % + cash %, portfolio equity/drawdown/metrics,
  attributed trades. Migration 028 `portfolio_backtests`; POST
  /api/backtests/portfolio (whole watchlist or a subset).
- **D — hardening.** SQUEEZE_RISK proxy gate (live REPORT-mode; veto
  promotion = open user decision); spreads (BULL_CALL/BEAR_PUT) + CSP into AUTO;
  §8 neutral-cell income emission behind a param.
- **E — UI + close.** Portfolio backtest page (allocation stacked-area,
  per-symbol contribution, instrument multi-select checkboxes wired to
  the permissions override), beginner projection, DEVLOG, adversarial
  verification workflow.

## Non-negotiables

§21 one pipeline (AUTO/portfolio replay the SAME signal/§8/exit/sizing
code as live — no parallel math); no fabricated data (real option
contract history only, as today); permissions may restrict but never
exceed account flags; every decision carries its rationale string.
