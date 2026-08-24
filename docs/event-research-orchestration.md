# Event Research Orchestration

**How the platform decides what evidence to gather for a catalyst, assembles
it, and hands it to a model — without ever handing it authority.**

Last updated: 2026-08-21 (Catalyst research upgrade, LOOP 6/7/8/10).
Companion documents: `search-architecture.md`,
`prediction-market-architecture.md`.

## The pipeline

The platform's existing philosophy, extended — not replaced — by two new
evidence layers:

```
Data acquisition        news · prices · options · fundamentals · macro
                        + WEB SEARCH · PREDICTION MARKETS      ← new
        ↓
Deterministic normalization   as-of gates, dedup, tiers, features
        ↓
Evidence construction         f1-evidence-v2 bundle
        ↓
LLM interpretation            event-analysis-v2, single-shot, no tools
        ↓
Validation                    number grounding, evidence refs, URL ban
        ↓
Research display              Catalyst tabs
```

There is no ReAct loop, no browser agent and no tool-calling. The model is
called **once**, over evidence the platform has already admitted.

## What the orchestrator is

`EventResearchOrchestrator` is a **function on the existing gateway seam**
(`apps/gateway/event_research.py::run_event_research`), not a standalone
service. Its responsibility is exactly one question:

> What evidence should be gathered for this event?

It does **not** decide whether to trade. It is linear and deterministic —
window, plan, search, evaluate, persist — rather than an agent loop, precisely
so every bound is readable in one pass.

## The research window

For every event:

```
research_as_of = the caller's as_of        # never searched beyond
window_start   = previous comparable event  # "what changed since last time"
```

The existing `previous_comparable()` resolver stays authoritative. When it
returns nothing, the window falls back to a **documented, event-type-specific**
lookback (`TYPE_DEFAULT_LOOKBACK_DAYS`: earnings 98d, CPI-family 45d, GDP 100d,
FOMC 56d, default 30d) — and says so. The returned window always carries:

```
{start, end, basis, previous_event_id, fallback_reason}
```

A fallback window can never masquerade as one anchored on a real previous
event: `previous_event_id` is null and `fallback_reason` is populated. The
orchestrator's candidate pool is **type-general**, so a CPI release finds its
real previous print rather than silently falling back.

## Refresh semantics (Phase 21)

Two actions that mean different things:

| Action | What it does | Cost |
|---|---|---|
| **Refresh Sources** | `POST /research/backfill`, `POST /prediction-markets/backfill` (+ existing news backfill) | Spends search quota / venue calls |
| **Generate Analysis** | `POST /analysis` over the *current* bundle | Spends an LLM call |

Every GET is free and poll-safe. Opening a tab, refreshing a page, or a React
Query poll can never bill the operator — proved by a test that makes provider
construction fatal and then hammers every research GET.

## The Evidence Bundle (f1-evidence-v2)

The version bump is not cosmetic: the contract materially changed, so a v1
analysis and a v2 analysis are not comparable answers.

New sections, in `SECTION_ORDER`:

- **`web_research`** — research window, search plan, counts, source mix, topic
  mix, the bounded ranked evidence set (`safe_title` only, no URLs),
  `suppressed_suspicious`, `skipped`, `run_status`, `retrieved_at`.
- **`prediction_markets`** — matched markets with relation, sanitized question,
  `market_implied_probability`, changes, spread, liquidity/volume, `observed_at`
  and a `data_quality` block of depth facts.

Also fixed in this program: **`options_analysis` is no longer hardcoded
unavailable**. It is populated from the live options seam (implied move, IV
context, implied-vs-actual history) and tiered `QUANT` — the platform's own
arithmetic, not somebody else's fact. Genuine `NO_DATA` statuses still pass
through honestly.

### Digest policy

The analysis cache keys on `(event_id, bundle_digest, prompt_version, model)`.
The digest covers **evidence**, never clocks:

- Prices, features and accepted evidence are digest-relevant — a material
  change must invalidate a cached analysis.
- `retrieved_at`, `observed_at`, `history_end`, `observation_count` and
  `matched_at` are pruned. Re-observing an unchanged market keeps the cached
  answer valid; any price change misses.

Tests pin both directions.

## The analysis contract (event-analysis-v2)

The model synthesizes across layers rather than summarizing providers. The
hierarchy it is told to respect:

```
OFFICIAL / PRIMARY   releases, filings, company IR      ← ground truth
MARKET DATA          prices, options, rates, fundamentals
PROFESSIONAL         journalism, analyst commentary
EXPECTATIONS         consensus · options-implied · prediction markets
```

**Lower layers never override higher ones.** Where they disagree, the model
reports the divergence in `evidence_conflicts` (naming both layers) rather than
averaging incompatible signals into one meaningless view. There is deliberately
**no single "AI score"** anywhere (Phase 23).

Schema additions: `prediction_market_expectations` (nullable — null is the
honest answer when the bundle carries no matched market), `evidence_conflicts[]`,
`web_research_highlights[]`. Confidence stays the enum `HIGH | MODERATE | LOW`
— never a naked number.

## LLM boundaries

The model **may** help with synthesis, ambiguity interpretation and scenario
construction. It **may not** control:

| Forbidden to the model | Owned by |
|---|---|
| Date boundaries | `research_window()` |
| Provider authentication | the adapters |
| Network destinations | the adapters |
| Rate limits | the orchestrator + adapters |
| Source-quality rules | `DOMAIN_TIERS` |
| Evidence persistence | the gateway seams |
| Numeric extraction truth | `fact_index` + validator |
| Execution | nothing — no path exists |

### Validation (Phase 8)

`validate_analysis` enforces, and a violation stores the analysis flagged
`INVALID` rather than serving it silently:

- Every narrative numeral must appear in `numbers_quoted` with a dotted path
  that resolves in the bundle's fact index — including **percentages**, which
  are quantities whatever their magnitude (a market-implied "68%" is exactly
  the invented-probability case the brief names).
- Quoted values must match the bundle within a tight tolerance, and only
  **value-preserving** re-renderings are accepted — quoting `0.63` does not
  license writing `0.6` or `1`.
- Evidence refs must exist: `news:<id>`, `web:<key>`, `pm:<provider>:<id>`, or a
  real bundle path. Grounding authority is minted by the **platform**, never by
  a retrieved document.
- `web_research_highlights` may cite only accepted `web:` documents.
- **URLs are banned from model output** — schemes, bare `www.` hosts, and
  host-with-path shorteners alike.

## Audit and observability

| Audit action | When |
|---|---|
| `EVENT_SEARCH_RUN` | One research backfill, with what it bought |
| `PREDICTION_MARKET_FETCHED` | One market refresh, with candidates/accepted counts |
| `EVENT_ANALYSIS_GENERATED` | (existing) one analysis |

Low-level HTTP GETs are **not** audited individually — that matches the
platform's existing convention and keeps the trail readable.

Metrics (labelled by provider only — never query text or market questions):
`search_requests_total`, `search_results_accepted_total`,
`search_provider_errors_total`, `prediction_market_requests_total`,
`prediction_markets_matched_total`, `prediction_market_provider_errors_total`.

## Failure semantics

Research failure is **degradation, not system failure**. Capability-by-capability:
with no Brave key and no Polymarket, the event calendar, historical comparison,
structured news, fundamentals, price, options, risk and the existing Event
Analysis all keep working. Pinned by
`tests/test_research_e2e_adversarial.py::test_event_surfaces_survive_both_research_providers_unconfigured`.

## Execution isolation

The absolute boundary of this program: nothing here reaches §8 instrument
selection, directional scoring, strength tier, volatility regime, Tier-0
sizing, the gate chain, contract selection, trading-pool authorization, order
approval, execution or exit logic.

Proved two ways:

1. **Structurally** — `tests/test_research_safety_adversarial.py`: an
   import-graph AST test in both directions, plus a word-ban on
   trading-shaped names in the prediction-market package.
2. **Behaviourally** — `tests/test_research_e2e_adversarial.py`: the full
   chain (research → markets → bundle → analysis) driven with an attacker
   controlling the search text and the market question, after which the
   watchlist, trading pool, orders and positions tables are row-diffed and
   must be **unchanged**.
