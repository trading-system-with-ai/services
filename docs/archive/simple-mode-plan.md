# Simple Mode — design plan (2026-08-20)

**Question asked:** is the system too complicated? Can a freshman use it,
while staying professional and accurate?

## Verdict

Yes — today this is a pro-mode terminal. The complexity is mostly
*essential* (the gate chain, catalyst analysis, and honest backtests ARE
the product), but it is presented all at once, in specialist vocabulary,
with configuration as the very first stage. The fix is **layering, not
lobotomy**: one engine, one set of numbers, two presentations.

## Evidence (measured 2026-08-20)

- UI: 13 pages, 67 components, ~57K LOC; landing page fires ~9 queries
  into ~9 panels; settings 1,513 lines; risk page 1,382 lines.
- Nav: 8 mandatory-feeling stages (Connect → Research → Screen → Validate
  → Authorize → Execute → Risk → Audit) before first value.
- Vocabulary: ~140 glossary terms; enum-first labels (ENTRY_READY,
  expectancy, profit factor, directional edge).
- Backend: 86 endpoints / 21 routers (events alone: 28); ~41 Settings
  fields; 14 runtime-config keys applied at boot.
- Already good (build on, don't duplicate): Guide page w/ first-session
  checklist, FlowNav stage links, glossary cards, why/why-not panels,
  bilingual i18n + useEnumLabel infrastructure.

## Design principles (the 10-persona consensus)

1. **Same engine, same numbers.** Beginner mode is a *projection* of pro
   data, never a second calculation path. (Model Validation veto power.)
2. **Simplify presentation, never controls.** Beginner mode may hide
   detail; it may NOT hide an active warning, skip a gate, or enable
   anything pro mode wouldn't. Gates stay strict in both modes.
3. **Teach the vocabulary, don't amputate it.** Plain label first, pro
   term in tooltip — beginners graduate instead of staying protected.
4. **Value before configuration.** A new user must see something useful
   before being asked for API keys and permission matrices.
5. **Three beginner jobs**, not eight stages: *See ideas → Check one →
   Trade safely.* Risk and audit become ambient (traffic light + feed).

## Build phases

- **P1 — mode toggle + label layer.** Persisted beginner/pro toggle
  (default: beginner for fresh installs). A single mapping table
  plain-label ↔ pro-term ↔ field (one source of truth, testable), built
  on the existing i18n-labels layer.
- **P2 — beginner dashboard.** Landing page in beginner mode answers one
  question: "How am I doing, and is anything worth doing today?"
  Portfolio value + plain-words P&L, ONE risk traffic light derived from
  the existing gate-chain/risk snapshot (same thresholds), top-3
  opportunities with one-line plain-English reasons, single primary CTA.
  UI-side aggregation of existing endpoints first; add /api/simple/summary
  only if query fan-out hurts.
- **P3 — setup wizard + presets.** 3-step connect wizard (provider → key
  → done; paper broker default). Permission presets mapping to the REAL
  §5 flags: Conservative (long stock only), Balanced (+calls),
  Advanced (all). Presets are shorthand for existing flags — no new
  permission semantics.
- **P4 — adversarial QA.** Tests that beginner mode: never suppresses an
  alert severity ≥ warning; renders identical numbers to pro mode for the
  same fields; cannot place any order pro mode would block; label mapping
  is total (no enum falls through to raw).

## Non-goals

- No removal of pro pages/endpoints. No new risk math. No third mode.
