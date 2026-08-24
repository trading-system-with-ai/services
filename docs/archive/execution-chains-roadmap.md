# Execution Chains Roadmap — every permission genuinely operable

**User mandate (2026-08-17):** the whole platform chain — backtest, live
trading, plan generation, risk — must support every execution chain, so
every account permission is REAL and operable, not a display.

**Broker ground truth (probed live 2026-08-17):** the Alpaca paper account
is a MARGIN account (multiplier 4, shorting_enabled true, options trading
level 3 = spreads approved). The broker permits everything below except
naked short options (Alpaca does not offer them at all) — every lock in
the platform is therefore a PLATFORM construction gap, not a broker
refusal.

**STATUS 2026-08-17: ALL THREE PHASES COMPLETE.** Every §2 permission
except the two permanently-locked naked shorts is a real, tested,
end-to-end toggle. 1032 backend tests green.

## The unlock rule (non-negotiable, §33)

A permission toggle unlocks ONLY when its ENTIRE chain exists and is
tested: §8 matrix → §9 selection → §10 gates → §12/§16 risk → execution →
positions/exits → reconciliation → backtest → UI. Until then the toggle
stays locked with the reason displayed. "Allowed" must always mean
"executable end to end" — a switch with nothing behind it is the lie this
platform exists to never tell.

## Phase 1 — Defined-risk spreads — COMPLETE (2026-08-17)

- [x] InstrumentType members + §8 matrix spread cells; degradation in
      `_finalize`; BEAR/MODERATE/HIGH upgrades from NO_TRADE to spread.
- [x] §9-S spread selector (long leg = §9 rank-1, width-targeted short
      leg, same expiry); net debit/max-loss/break-even in §37 output.
- [x] Risk: max_loss = net debit × contracts (§12.1); §16 NET greeks.
- [x] Execution: Alpaca mleg atomic two-leg order (defined-risk shape
      guard in the adapter); simulated venue net-debit fills.
- [x] Positions/exits: short-leg columns (migration 015); net-premium
      exit semantics; §18 counts the short leg OURS (negative).
- [x] Backtest: run_spread_backtest over two REAL contract-bar series.
- [x] UI + runtime-config toggle `defined_risk_spreads` — UNLOCKED.

## Phase 2 — Covered calls + cash-secured puts — COMPLETE (2026-08-17)

- [x] Sell-to-Open path hard-scoped to collateralized shorts:
      `submit_short_open_order` is OCC-gated + mandatory `covered_by`
      attestation; dedicated `submit_short_close_order` buyback.
- [x] THE COLLATERAL LAW: covered call requires free held shares
      (≥100/contract), pinned until buyback (stock closes + exit sweep
      refuse pinned shares); CSP reserves strike×100 cash (migration 016);
      buyback releases in the same transaction.
- [x] §-standard selection: 30–45 DTE, |Δ| 0.15–0.35, OTM, real NBBO,
      OI≥50, spread≤15%; mechanical management 50% profit capture /
      2× loss stop / 21 DTE / ITM-assignment ADVISORY.
- [x] §16 negated short-leg greeks; §18 short-leg claim; kill-switch
      asymmetry (opens blocked, buybacks always allowed).
- [x] Backtests: buy-write (same-bar unwind, strike assignment, churn
      guard) + CSP (cash-settled assignment approximation, documented).
- [x] UI + runtime-config toggles `covered_call` + `cash_secured_put` —
      UNLOCKED.

## Phase 3 — Margin + short stock — COMPLETE (2026-08-17)

Scope decision (industry standard): margin exists to SUPPORT SHORTING —
the broker enforces buying power/maintenance on its side; levered LONG
sizing is deliberately NOT enabled (§12 sizes from cash, never buying
power). Puts remain the preferred bear expression; the §8 matrix emits
SHORT_STOCK only in the two dead-end bear cells where premium is
unbuyable (STRONG/EXTREME; MODERATE/HIGH without spreads), and only when
BOTH `short_stock` AND `margin` are on.

- [x] Mirrored exits: BEAR hard stop ABOVE entry (close >= entry + stop)
      in the SHARED engine — backtest and live identical (§21).
- [x] Unbounded-loss risk model: stop-based risk × gap factor 2.0
      (`SHORT_STOCK_GAP_RISK_FACTOR`) — sizing halves vs. long stock.
- [x] Adapter: `submit_stock_short_order` (STOCK-gated — an OCC symbol
      raises, so naked short options stay unconstructable forever — +
      mandatory margin attestation) and `submit_stock_cover_order`.
- [x] Gateway: SELL_TO_OPEN approve (simulated proceeds-credit fill +
      broker T1/T2), BUY_TO_CLOSE cover (allowed under the pause),
      liability marking (negative market value), §16 delta −1/share,
      §18 negative-quantity ticker claim, mirrored sweep direction.
- [x] Backtest: run_short_stock_backtest — bear-mirror entries, shared
      mirrored exits, equity = cash − qty×close.
- [x] UI + runtime-config toggles `short_stock` + `margin` — UNLOCKED.
- [ ] DEFERRED (data source needed, §33): borrow availability/fees, HTB
      lists, short-interest squeeze guardrails. Alpaca paper enforces
      locate on its side; a §33-approved data source would let the §10
      chain warn earlier.

## Permanently locked (honest refusals)

- `naked_short_call` / `naked_short_put`: Alpaca does not offer naked
  short options at any level (broker refusal), and they violate the §4
  charter (unbounded risk). The lock note cites BOTH facts. These are the
  ONLY remaining `FORBIDDEN_PERMISSION_FIELDS` / `_FORBIDDEN_ALLOW_FLAGS`
  entries — forever.

## Dependencies / notes

- Historical options data facts that constrain every backtest leg:
  daily bars from ~Feb 2024, no NBBO/greeks/OI history
  (data-source-architecture.md).
- 2026-08-17 latent-bug fix found during Phase 3: `VALID_INSTRUMENTS` in
  libs/trading_core/greeks.py was never extended for Phase 2 — an open
  income position crashed the §16 aggregate view. Fixed + covered.
