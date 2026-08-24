# Catalyst & Event Intelligence — Final Report

**Programme:** `prompts/event_analy_system.md` (102 sections), Phases A–L executed.
**Deliverable:** spec §100 (A. Architecture Review · B. Data Coverage Matrix · C. Skill Architecture · D. Event Intelligence Pipeline · E. UI Implementation · F. LLM Architecture · G. Risk Integration · H. Validation · I. Deferred Features), under the §101 FACT/QUANT/ANALYSIS principle and the §102 product question.
**Date:** 2026-08-19.
**Authoritative sources:** `docs/DEVLOG.md` entries (21)–(32); `docs/catalyst-event-audit.md` (Phase A); `docs/ARCHITECTURE.md` ADR-008, ADR-009. All paths are relative to `services/` unless prefixed `ui/` or `prompts/`.

**Headline.** In twelve phases the platform gained a typed event registry fed by authoritative calendars, point-in-time price / fundamentals / news / options / macro / Fed evidence per event, one schema-validated LLM analysis whose every quoted number is checked against the evidence that produced it, an event-risk block inside the existing risk shadow, and a `/catalysts` UI with thirteen tabs. **Not one trading decision changed.** `assess()` is still called with `extra_caps=()` at every call site in `apps/` — grep-pinned and AST-pinned — so the entire event-risk layer decides nothing. The backend suite went from **2020 passed** at the Phase A baseline to **4000 passed, 1 skipped, 1 xfailed**; the UI from 114 to 797 vitest tests.

**The central caveat, stated once and repeated in §H and §I:** no consensus data exists at this subscription tier (Massive `/benzinga/v1/*` is a permanent 403), so EPS/revenue **surprise %** — the number an earnings system is normally built around — is not computable for any period, past or future. Everything the platform says about "expectations" is built from price, positioning and news proxies, and is labelled as such. The §86 predictiveness harness shipped in Phase L is a *measurement instrument*, not a result: it reports |rho| and n and flags `NOT_MEANINGFUL` below n = 12, and the platform does not yet have enough stored event history to say anything with it.

---

## A. Architecture Review

### A.1 What existed before

From the Phase A audit (`docs/catalyst-event-audit.md` §2–§4; DEVLOG (21)):

**Present and reusable.** A modular-monolith FastAPI gateway whose single `lifespan` owned five background loops, each splitting its tick into a directly callable function because httpx `ASGITransport` never runs lifespan under tests (`monitor.run_sweep_and_update`, `risk_snapshot.run_scheduled_snapshot`). A transactional audit helper that adds its row to the caller's session and never commits. ADR-006 alerts as a *classified read over the audit trail* — no alerts table. A market-data provider registry with no default and no cross-provider fallback, plus tri-state capability probing that distinguishes a 403 from a probe fault. A news store deduped on `news_articles.source_id UNIQUE`, and — already in code — the §27/§79 rule that drops any LLM draft citing an unstored article. Daily bars with complete-days-only hygiene. A risk engine with an `extra_caps` seam and an AST tripwire asserting no `apps/` caller uses it. In the UI: `Term.tsx` + a bilingual `glossary.ts`, `.provenance` CSS tiers, and exactly one chart component.

**Absent entirely** (grep-verified across `libs/` and `apps/`): any event entity — no events table, no ORM model, no enum. The nearest thing was `Recommendation.catalyst_type`, a free-form LLM `String(64)`: precisely the generic-string anti-pattern §5 forbids. No calendar provider abstraction and no adapter of any kind. **No fundamentals ingestion whatsoever** — grep for "financials" across `libs/` and `apps/` returned only comments. No as-of primitive and no query-level enforcement of one. No clustering, materiality, novelty or source-quality on news. No event concept anywhere in the risk engine. An existing `stock_bars_1m` Timescale hypertable (migration 002) that **had never been written to**. No token accounting in `libs/llm`. No `/catalysts` nav destination.

**One naming collision found before it could do damage:** `GET /api/analysis/{ticker}/catalyst` already shipped, typed in `ui/lib/types.ts` and pinned by `tests/test_catalyst.py` — read-only LLM interpretation with no event object, no `scheduled_at` and no previous-event linkage. ADR-008 records the decision: keep it, and name the new router `events` (`/api/events`), not `catalyst`.

**Data reality bounding all ambition** (audit §3, from a live entitlement probe of 2026-08-19 ~10:30 ET — not vendor documentation): every Massive `/benzinga/v1/*` endpoint answers **403 "not entitled"**, which removes the earnings *calendar*, consensus, surprise, guidance, ratings and analyst insights in one stroke. No provider at any probed tier sells historical option quotes, greeks or OI. Massive's treasury yields have 1y/5y/10y but **no 2y**. Massive's option snapshot returns `greeks={}` and `implied_volatility=None` on this plan, making Alpaca the only IV source. Stored daily bars begin ~2024-03.

**Structural constraints that shaped every phase:** no migration runner (SQL files execute only on a fresh Postgres volume, so all seven catalyst migrations were applied by hand and every one uses `IF NOT EXISTS`); the test harness is SQLite while production is Postgres+Timescale, so JSONB, GIN and CHECK behaviour is invisible to the suite; ADR-007 assumes one gateway process, so every scheduled write must be idempotent at the *database* level, not in memory.

### A.2 What exists now

Twelve phases, seven migrations (021–027), all applied live and all deployed.

| Phase | DEVLOG | What was built | Principal new files | Migration |
|---|---|---|---|---|
| **A** | (21) | Audit only, no code. Live entitlement probes → 6 parallel read-only inspections → 6 adversarial verifications (47 corrections) → 4-part synthesis | `docs/catalyst-event-audit.md` (603 lines) | — |
| **B0+B** | (22) | Event registry, calendar providers, ingest loop, `/api/events`, T-minus alert, first Catalysts page | `libs/event_calendar/` (registry, `sec_edgar`, `fed`, `alpaca_calendar`, `massive_calendar`, `stub`); `events/{models,taxonomy,importance}.py`; `event_calendar.py`; `routers/events.py` | **021** `events`, `market_calendar`, `event_ingest_state` |
| **E1** | (23) | Point-in-time price context + previous-event market reaction | `events/reaction.py`; `event_price.py` | — |
| **E2** | (24) | Point-in-time fundamentals on `acceptance_datetime` | `events/fundamentals.py`; `fundamentals.py` seam | **022** `fundamental_statements` |
| **C** | (25) | Event replay — intraday reactions, §60 history table, previous-event linkage | `events/replay.py`; `event_replay.py` | — (first writes to `stock_bars_1m`) |
| **D** | (26) | News evidence engine — dedup, clusters, materiality, EvidenceScore | `events/news_intel.py`; `event_news.py` | **023** additive on `news_articles` + tickers GIN |
| **F** | (27) | EvidenceBundle, schema-validated LLM analysis, event memory | `events/evidence.py`; `libs/llm/event_analysis.py`; `event_analysis.py`, `event_evidence.py` | **024** `event_analyses` |
| **I** | (28) | Options / implied move — historical ATM straddle approximation, live chain | `events/implied_move.py`; `event_options.py` | **025** `option_daily_bars`, `event_option_metrics` |
| **J** | (29) | Catalyst UI slices — hero, timeline, evidence/scenarios tabs, card summaries | `event_timeline.py`; 8 UI components | — |
| **G** | (30) | Macro intelligence — BLS/BEA/Treasury adapters, packets, multi-asset reactions | `event_calendar/{bls,bea,treasury,macro_data}.py`; `events/macro.py`; `event_macro.py` | **026** `macro_observations`, `treasury_yields` |
| **H** | (31) | Fed intelligence — statement diff, §43 dimensions, two-window FOMC reaction | `event_calendar/fed_docs.py`; `events/fed_intel.py`; `event_fed.py` | **027** `fed_documents` |
| **K** | (32) | Event risk — SHADOW integration, pre-trade surfacing, Risk tab | `risk/event_risk.py`; `event_risk.py` | — |
| **L** | (33) | §96 adversarial look-ahead suite, §86 measurement harness, this report | `tests/test_event_lookahead.py`; `events/event_study.py`; `event_study.py` | — |

The phase *order* deviates from the spec's alphabetical sequence, and the audit §11.0 explains why: E1/E2 (price, fundamentals) came before C/D because replay and news evidence both consume them, and J (UI) was sliced across phases rather than deferred, so every phase landed with a visible acceptance surface rather than accumulating unverified backend.

**House rule preserved:** stdlib only in `libs/`. Rank correlation, Jaccard shingling, sentence diffing, Black–Scholes bisection, nearest-rank percentiles — all stdlib. No numpy, no scipy, no pandas was added. (Phase L notes the cost of that rule honestly: `scipy` is not installed, so the Spearman implementation could not be cross-checked against `scipy.stats.spearmanr` and was instead property-tested and hand-verified.)

---

## B. Data Coverage Matrix (§100.B) — end state

This updates the Phase A audit's §3 matrix from *probe* to *what actually worked in production*. "PIT-safe?" = does the source carry a publication/acceptance timestamp permitting §14/§85 as-of filtering. **Bold** marks a change from the audit's predicted state.

| Feature | Provider | Endpoint | Status | Historical | Realtime | PIT-safe? | End-state outcome |
|---|---|---|---|---|---|---|---|
| Earnings dates (upcoming) | SEC EDGAR (derived) | cadence estimate from 8-K Item 2.02 | keyless, **works** | n/a | n/a | n/a | **ESTIMATED only, badged, never alerted.** 4 upcoming rows live (HPE 10-14, SMCI 10-22, AAPL 10-29, RDW 11-04), all AMC. Benzinga 403 is permanent |
| Earnings dates (past) | SEC EDGAR | `data.sec.gov/submissions/CIK*.json` | keyless, **works** | full | n/a | Yes (`acceptanceDateTime`) | **12 CONFIRMED past releases per ticker**, Nov-2023→Jul-2026. `cluster_releases` keeps the earliest 2.02 per window; `SEC_USER_AGENT` must carry a contact e-mail or `www.sec.gov` 403s |
| Earnings actuals | Massive | `/vX/reference/financials` | 200 | quarterly + TTM | n/a | Yes | **13 statements stored for AAPL** (12 quarterly + 1 TTM). Provider TTM rows carry no acceptance instant → **derived TTM** from the four newest visible quarters |
| Consensus (EPS/revenue) | — | Massive `/benzinga/v1/earnings` | **403 permanent** | none | none | n/a | Fixed string `CONSENSUS DATA UNAVAILABLE` everywhere. **Surprise % uncomputable for any period.** No code path in the stack computes one |
| Estimate revisions / guidance / ratings | — | `/benzinga/v1/{ratings,guidance,analyst-insights}` | **403 permanent** | none | none | n/a | §34 not deliverable. The §86 harness names `estimate_revision` an **unmeasurable candidate** rather than omitting it |
| News | Massive + Alpaca | `/v2/reference/news`; `/v1beta1/news` | 200 / 200 | yes | yes | Yes (`published_at`) | **Both used together**: a live AAPL window fetched 283 articles (Alpaca 206 + Massive 77), 280 stored, 278 unique. New **windowed** reader `get_news_window` — the recency cursor is useless for a window that closed last week |
| Fundamentals / statements | Massive | `/vX/reference/financials` | 200 | quarterly | n/a | **Yes** (`acceptance_datetime`) | **The point-in-time key the whole as-of contract rests on.** Statements provider ≠ price provider: `fundamentals_provider_name` picks Massive even when market data is Alpaca (the first live call stored 0 rows before this split) |
| Daily bars | Massive / Alpaca | aggs / bars | 200 / 200 | ~2024-03+ | yes | Yes | 604 AAPL bars, SPY 605 at the E1 live run. Reactions before 2024-03-20 are honestly "bars unavailable before …" |
| Intraday bars (1m) | Alpaca | `/v2/stocks/{t}/bars` (iex) | 200 | ~2024+ | yes | Yes | **`stock_bars_1m` written for the first time.** 853 minute bars for one AAPL release; 1 476 for one FOMC window. iex ≠ consolidated tape — labelled |
| Options live IV / greeks | Alpaca | `/v1beta1/options/snapshots` | 200 | n/a | yes | n/a | Only IV source. Basis `LIVE_CHAIN_SNAPSHOT`, served for **upcoming events only** (§85; test-pinned) |
| Options historical bars | Massive | `/v2/aggs/ticker/O:{occ}/range/1/day` | 200 | **~2 years only** | n/a | Yes | **Live limit found, not documented:** aggregates older than ~2 years answer 403 "data timeframe". The 8th AAPL event (2024-08-01) is an honest NO_DATA row surfaced in `coverage.history_attempted_no_data`. Massive also **rate-limited (429)** back-to-back option-bar requests → bounded exponential backoff honouring `Retry-After` |
| Historical IV / greeks / OI | — | none | not sold at any tier | none | n/a | n/a | Reconstructed from ATM call+put daily closes, labelled `HISTORICAL_DAILY_CLOSE_APPROXIMATION`. Event straddles usually expire next day → `iv_after` / IV-crush are NO_DATA for weekly expiries |
| Macro actuals | **BLS v1 (keyless)** / BEA | BLS timeseries API; BEA API | **200 keyless / key-gated** | 3 years (BLS unregistered) | monthly | **Yes, joined** | **BLS works with no credentials at all** (~25 req/day, 3-year window). BEA *actuals* need a free `BEA_API_KEY`; absent → `CapabilityNotAvailable`, never an estimate. The release **instant** is joined at write time from this platform's own stored schedule (basis `SCHEDULED`) else period end + 45 d (basis `ESTIMATED`) — both stored, the as-of gate reads both |
| Macro release calendar | **BLS / BEA schedule pages** | `bls.gov/schedule/*`; `bea.gov/news/schedule` | **200 keyless, both work** | yes | yes | Yes (date + published ET time) | **bls created 52 events, bea 8** on the first live refresh; 60 macro events in the registry. A row whose time will not parse is **dropped, never defaulted** (JOLTS 10:00 ET → DURING_MARKET; the rest 08:30 → BEFORE_MARKET) |
| Fed calendar / documents | Federal Reserve | `fomccalendars.htm`; statement/minutes pages; RSS | 200 keyless | yes | yes | Yes (RSS timestamps) | **29 meetings 2021–2027 + 15 speeches** ingested. Statement/minutes pages parsed live; `released_at` for statements comes from RSS (18:00 UTC); **as-of gating happens BEFORE any HTTP fetch** |
| Treasury yields | **Treasury CSV** | `daily-treasury-rates.csv` | **200 keyless** | full | daily | Yes | **Supplies the 2Y Massive lacks.** 408 yield-curve rows on one backfill; tenor keys read verbatim ("2 Yr", "10 Yr"), a missing tenor is absent, never 0.0 |
| Multi-asset proxies (§39) | Alpaca | daily bars | 200 | full | yes | Yes | `MACRO_REFERENCE_SYMBOLS` (SPY, QQQ, TLT, IEF, SHY, GLD, USO, UUP) as a **second** reference set beside ADR-005's `INDEX_SYMBOLS` (ADR-009). Every one is flagged `is_proxy` — TLT is a fund holding long Treasuries, not "long rates" |
| Fed-funds futures pricing | — | none | not subscribed | none | none | n/a | Market-implied policy pricing honestly **UNAVAILABLE**; the 2Y change is a labelled proxy |
| Retail sales / ISM | — | Census / ISM | **no adapter** | none | n/a | n/a | Release *dates* tracked where a schedule exists; actuals deferred (§I) |
| Trading calendar / holidays | Alpaca | paper-api `/v2/calendar` | 200 | yes | yes | Yes | **552 trading days 2025-07→2027-09, 4 early closes.** Closes the admitted `_last_expected_trading_date` hole |

Live registry at the end of Phase B: **189 events across 9 types**; after Phase G, **60 macro events** added. Previous-event linkage persisted for **104 of 110 rows**.

---

## C. Skill Architecture (§100.C)

### C.1 The decision: no skill framework

The spec (§4) describes five "skills": Event Intelligence Core, Market Reaction Engine, Evidence Engine, News Intelligence Engine, Fundamental Snapshot Engine. The Phase A audit's verdict was explicit — **do not invent a skill framework** — and Phases B–L held to it.

The reasoning is structural, not stylistic. There is no plugin registry, worker or queue anywhere in this codebase. Every unit of work is one of exactly three things: a router, a *gateway seam module* (`risk_snapshot.py`, `order_sync.py`, `monitor.py`), or a pure `libs/trading_core/` module. That two-layer split already satisfies §4's "prefer shared reusable infrastructure": pure, I/O-free computation in `libs/`, and one thin seam per concern in `apps/gateway/` that fetches inputs, calls the pure library, persists, audits and emits metrics. A skill runtime would have been a third concept that no other part of the platform speaks, invented to satisfy a word in the spec.

So the five skills became **capability groupings over one shared core**:

| Spec "skill" (§4) | Realised as | Nature |
|---|---|---|
| Event Intelligence Core | `events/models.py` (Event, `same_event`, `merge`, `previous_comparable`, `SOURCE_RANK`), `events/taxonomy.py` (`event_key`, ET↔UTC, `classify_session`, lifecycle), `events/importance.py` | pure |
| Market Reaction Engine | `events/reaction.py` (E1: windows, abnormal-vs-benchmark, history stats, `as_of_bar_filter`), `events/replay.py` (C: intraday anchors), `events/implied_move.py` (I) | pure |
| Evidence Engine | `events/evidence.py` (F: `EvidenceBundle`, `fact_index`, `bundle_digest`, `compute_expectations_gap_inputs`) | pure |
| News Intelligence Engine | `events/news_intel.py` (D: dedup, leader clustering, materiality, novelty, source quality, decay, `sanitize_for_llm`) | pure |
| Fundamental Snapshot Engine | `events/fundamentals.py` (E2: as-of gate, ratios, §29 deltas, derived TTM) | pure |
| *(not in the spec's five)* | `events/macro.py` (G), `events/fed_intel.py` (H), `risk/event_risk.py` (K), `events/event_study.py` (L) | pure |

with one gateway seam per concern: `event_calendar.py` (the ingest loop), `event_price.py`, `fundamentals.py`, `event_replay.py`, `event_news.py`, `event_evidence.py`, `event_analysis.py`, `event_options.py`, `event_timeline.py`, `event_macro.py`, `event_fed.py`, `event_risk.py`, `event_study.py` — and exactly one router, `routers/events.py`, carrying 28 routes.

### C.2 What the discipline bought

The purity rule is **statically enforced**, not merely intended. Several pure modules carry an AST or tokenized-source scan asserting they import nothing from `apps/` or `libs/market_data` (`test_fed_intel_imports_no_io_layer` is the house name for the pattern; `risk/event_risk.py` additionally forbids any LLM, network or engine import). Phase L's `event_study.py` seam goes further and asserts *no `libs.market_data` import exists at all*, so "DB-only" is a property of construction rather than of current behaviour.

The payoff is visible in the test counts: the pure libraries carry the heavy suites (`test_events_news_intel` 180, `test_events_replay` 112, `test_events_fundamentals` 109, `test_events_implied_move` 89, `test_events_fed_intel` 80) and run in milliseconds with no fixtures beyond dataclasses, because there is nothing to mock.

### C.3 One deliberate deviation, recorded

`GET /api/events/study` is a **static** path added at the end of `routers/events.py`, while `GET /{event_id}` is registered far above it. Starlette matches in registration order, so a plain append answered `422 "study is not a valid integer"`. Rather than move the code (the phase contract required it at the end of the file), one documented line follows the handler:

```python
router.routes.insert(0, router.routes.pop())
```

Only the ordering moves; path, handler and decorator are unchanged, and the other 27 routes keep their original relative order. The rest of the file solves the same problem conventionally, by declaring `/calendar` before `/{event_id}`.

---

## D. Event Intelligence Pipeline (§100.D)

```
  ┌── CALENDAR SOURCES (libs/event_calendar/ — own registry, ADR-008) ──────┐
  │  sec_edgar   8-K Item 2.02 acceptanceDateTime → CONFIRMED past          │
  │              + cadence estimate → ESTIMATED upcoming                    │
  │  fed         fomccalendars.htm → 29 meetings; speeches RSS              │
  │  bls / bea   schedule pages → 60 macro events            [KEYLESS]      │
  │  alpaca_calendar  552 trading days, 4 early closes                      │
  │  massive_calendar holidays; earnings probe = 403 → capability false     │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             ▼
  INGEST TICK  apps/gateway/event_calendar.py::run_calendar_ingest
    per-provider 20 h cadence via `event_ingest_state` (DB, not memory)
    upsert + merge under SOURCE_RANK  (USER < COMPANY_IR_SEC <
      GOVERNMENT_AGENCY = FEDERAL_RESERVE < STRUCTURED_PROVIDER < DERIVED
      < NEWS;  the LLM may NEVER write a date)
    one event = one row (`events.event_key UNIQUE`); EARNINGS reconcile
      by a ±21-day same-ticker window so estimate drift ≠ duplicate card
    relevance POSITION > POOL > WATCHLIST > MARKET_WIDE > OTHER
    exactly-once EVENT_APPROACHING audit row per (event, horizon)
      — checked against the audit table, so restarts cannot double-fire
                             ▼
  ┌──────────────  EVENT REGISTRY  (migration 021)  ──────────────┐
  │  events · market_calendar · event_ingest_state                │
  │  previous_event_id + comparison_reason (never crosses types)  │
  └──────────────────────────┬───────────────────────────────────┘
                             │  as_of = _resolve_as_of(request)  ← ONE seam
                             ▼
  ┌── PER-EVENT SEAMS — every one a pure function of stored rows at as_of ──┐
  │  price-context   E1  reaction.as_of_bar_filter (bar d visible iff       │
  │                      d < as_of ET date, or d == as_of and ≥ 16:00 ET)   │
  │  fundamentals    E2  acceptance_datetime <= as_of ONLY — never period   │
  │                      end; rows without an acceptance instant EXCLUDED   │
  │  replay/history  C   SQL `ts <= as_of` on minute bars                   │
  │  news            D   window = previous comparable event → as_of         │
  │  options         I   stored metrics; LIVE basis only for FUTURE events  │
  │  macro           G   a print is visible iff release_at <= as_of         │
  │  fed             H   SQL bound + a pure gate; as-of BEFORE any fetch    │
  │  timeline        J   every kind's own gate, composed                    │
  │  risk            K   previous prints only                               │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             ▼
  EVIDENCE BUNDLE  events/evidence.py  (`f1-evidence-v1`)
    17 sections in fixed SECTION_ORDER, each tagged tier DATA | QUANT
    fact_index flattens every fact to a dotted path
      — 880 facts / 173 numeric on the live AAPL bundle
    bundle_digest = sha256 of canonical JSON over a PRUNED view
      (volatile timestamps excluded — see §F.4)
    news enters ONLY sanitised; suspicious_instruction articles excluded
    consensus is ALWAYS `CONSENSUS_DATA_UNAVAILABLE`
                             ▼
  LLM  libs/llm/event_analysis.py::analyze_event   [tier: LLM ANALYSIS]
    strict json_schema `event-analysis-v1`; §51 UPSIDE/BASE/DOWNSIDE;
    §52 surprise threshold (confidence NOT_MEANINGFUL allowed);
    §50 invalidation; expectations_gap_regime enum; numbers_quoted[]
                             ▼
  VALIDATOR  validate_analysis(parsed, fact_index)
    every quoted path must EXIST in the bundle and MATCH within 1e-6
    violations are STORED, not hidden → status INVALID
                             ▼
  MEMORY  event_analyses (migration 024)
    bundle JSONB beside the analysis; provider/model/prompt_version/usage/
    latency; PARTIAL unique index (event_id, bundle_digest, prompt_version,
    model) WHERE status='OK' — a retry after FAILED is storable, `force`
    supersedes rather than collides.  Prior analyses re-enter later bundles
    as tier LLM_PRIOR, as-of gated, OK-only, labelled opinions not evidence
                             ▼
  RISK  K   shadow["event"] inside the existing RISK_DECISION shadow dict
            event_risk_caps → QuantityCap rows that join ONLY the
            hypothetical shadow verdict.  assess(extra_caps=()) unchanged
                             ▼
  UI  /catalysts → /catalysts/[eventId] → 13 tabs, DATA/QUANT/AI separated
```

### D.1 What the pipeline actually produced, live

Every number below is from a real run against production data, recorded in the DEVLOG entry named.

| Surface | Live result | DEVLOG |
|---|---|---|
| Registry ingest | 189 events across 9 types; 552 trading days, 4 early closes; SEC gave 12 CONFIRMED past releases per ticker and 4 ESTIMATED upcoming | (22) |
| Price context | AAPL bars through 2026-08-18 (604 bars, SPY 605); run-up since the 2026-07-30 print −7.0 %, RV20 34.7 %, ATR 2.4 %, −10.0 % from the 52-week high; 10 of 12 prior releases measured, 2 honestly "bars unavailable before 2024-03-21" | (23) |
| Fundamentals | 13 statements stored; FQ3'26 (period 2026-06-27, **accepted 2026-07-31 10:01Z**) revenue $109.4 B (+16.4 % YoY), EPS $2.02 (+28.7 % YoY), derived TTM EPS $8.44 → P/E 36.7 against an own-history median of 35.5 (56th pct, n=9) | (24) |
| Replay | 853 minute bars in 0.2 s; after-hours −4.5 %, gap at open −8.6 %, +5m −9.4 %, max first-hour move 1.4 % | (25) |
| News | 283 articles fetched (Alpaca 206 + Massive 77) → 280 stored, 278 unique → **177 clusters**, largest 6 (2.2 %), **104 material**, 14 themes | (26) |
| Evidence + LLM | Bundle in 0.7 s with **880 facts / 173 numeric**; momentum +0.57 (11 improved / 3 weakened of 14), 106 material developments; analysis OK in 51 s, 28 485 in / 3 070 out tokens, **zero validator violations** | (27) |
| Options | 2026-07-30: implied **±3.82 %** vs actual **−7.35 %** → UNDER_PRICED (ratio 1.92); 2026-04-30 3.84 % vs 3.24 % FAIR; 2026-01-29 4.53 % vs 0.46 % OVER_PRICED | (28) |
| Timeline | **99 items** in a 20-day window (97 material NEWS across 14 categories, 1 FILING, 1 ANALYSIS) in 0.28 s | (29) |
| Macro | bls created 52 events, bea 8; one CPI backfill = 90 observations, 408 yield-curve rows, 623 bars in 21 s. **July CPI +0.074 % MoM headline / +0.215 % core, 3.36 % YoY NSA**; previous release: SPY +0.25 %, QQQ +0.73 %, GLD +0.99 %, 2Y −2.0 bp | (30) |
| Fed | 4 documents + 1 476 minute bars in 0.9 s. Statement 2026-07-29: **vote 9–3** (dissenters Hammack/Kashkari/Logan), target 3.50–3.75 % held; diff vs 2026-06-17 = 1 ADDED, 2 CHANGED, 6 UNCHANGED. **The two reaction windows disagree** — statement SPY −0.152 %, TLT −0.072 %; press conference SPY −0.311 %, TLT −1.040 %, GLD +0.270 % — which is exactly why §45 demands they stay separate | (31) |
| Event risk | AAPL T−71 d: LOW, n=8, median \|1.33 %\|, p90 7.35 %, implied None, enforcement SHADOW | (32) |

**Two structural guarantees hold across the whole diagram.** (i) **GET never fetches.** Every read endpoint is pinned by an "exploding provider" test — a provider handle that raises on any call — so a network fetch on a read path fails the suite. All network spend happens on explicit `POST …/backfill`. ADR-009 records why this matters concretely: BLS allows ~25 requests/day unregistered, so a lazily-topping-up read would exhaust the day's budget on one page load and then fail the backfill that could have repaired it. (ii) **The as-of instant is resolved once**, at `routers/events.py::_resolve_as_of`, and threaded down; a future `as_of` is a 422 across eleven endpoint families (test-pinned).

---

## E. UI Implementation (§100.E)

One new nav destination, `/catalysts` (zh 催化剂), added as a `SECTIONS` row beside the existing ten.

**`/catalysts` — the calendar.** Horizon control (today / 7d / 30d / custom), estimated-events toggle, manual refresh, a capability banner naming what the providers cannot serve, and relevance groups (POSITION / POOL / WATCHLIST / MARKET_WIDE / OTHER). `EventCard` carries a status badge, T-minus, an importance ⓘ breakdown showing the components rather than a bare score, a confirm-date dialog, and — opt-in via `GET /api/events?summaries=true` — historical move, implied move and analysis status (READY < 7 d / STALE / NONE). The summaries are opt-in precisely so the default payload stays byte-identical to before, which is pinned.

**`/catalysts/[eventId]` — the event.** `EventHero` shows T−n / T+n, schedule + session + zone, the CONFIRMED/ESTIMATED badge *with its source*, freshness, cost-basis exposure, a risk chip and an implied-move chip that is suppressed on NO_DATA rather than rendered as a zero. Then thirteen tabs, one per phase:

| Tab | Phase | Shows |
|---|---|---|
| Price | E1 | Positioning tiles (run-up, RV20, ATR, SMA distances, % from 52w high), previous-reactions table (gap/1D/3D/5D/10D/abnormal), history strip "Last 8: median \|1D\| … p90 … positive 5/8 — based on 8 events" |
| Fundamentals | E2 | §58 PREVIOUS/CURRENT/CHANGE table with ↑/↓ and bps, valuation tiles vs the company's *own* history, consensus-unavailable banner, freshness line |
| Previous Event | C | Release info, immediate-reaction tiles with reasons, "Load minute bars", subsequent 1D/3D/5D/10D |
| History | C | LAST 4/8/12 toggle, UNAVAILABLE columns named honestly, backfill button |
| News | D | Counts, themes, clusters, evidence table with score components |
| Evidence | F/J | The bundle with DATA/QUANT tiers, coverage, consensus notice |
| Analysis | F | Tier chips DATA / QUANT / **LLM ANALYSIS**, scenario cards, surprise-threshold chip ("not a probability"), invalidation, prior analyses collapsed, INVALID shown *with* its violations banner |
| Scenarios | F/J | Scenario cards + surprise threshold + invalidation, CTA when no analysis exists |
| Options | I | Implied vs realised, IV before/after, inline-SVG `ImpliedVsActualChart`, the §37 disclaimer on every implied-move surface |
| Timeline | J | Rail with anchors (LAST EARNINGS → developments → TODAY → NEXT EARNINGS), kind and category filters, collapsed groups |
| Macro | G | Packet: previous actual, consensus UNAVAILABLE, trend, multi-asset reaction with proxy flags, `MacroReactionChart` |
| Fed | H | Statement diff with counts, dimensions table + "no single hawkish/dovish score by design", **two reaction windows side by side** with a 1m/daily basis badge, speeches |
| Risk | K | State chip + SHADOW badge reading "shadow only — never blocks trades", drivers, caveats with sample size, historical table, implied-vs-historical, options panel |

Plus an EVENT RISK panel on the Trade Plan (§65 layout).

**Three UI conventions were inherited rather than invented:** `Term.tsx` + `glossary.ts` supplies every §90 ⓘ; the `.provenance` CSS tiers gained a third member, `quant-derived`, beside the existing data-driven and llm-generated (§49/§91 in one class name); and the backend returns *structured chart data only* — the UI renders inline SVG, no chart library (§61).

**A recurring failure mode worth recording.** Three separate phases shipped UI that typed the payload from prose rather than from the wire, and `[k: string]: unknown` silenced `tsc` each time. Phase E1's verifier found `"1D"` keys against a `1d` lookup — every reaction cell would have rendered Unavailable. Phase G's verifier found **10** key mismatches in one tab. Phase H's found two more, including diff-count keys that were uppercase on the wire. All were caught by verifiers, none by types. The lesson is in the fix that stuck: tests now pin the **wire spellings**, not the TypeScript interface.

UI tests grew 114 → 180 (B) → 208 (E1) → 256 (E2) → 305 (C) → 338 (D) → 396 (F) → 440 (I) → 581 (J) → 718 (G/H) → **797** (K), across the catalyst components; `tsc` clean throughout.

---

## F. LLM Architecture (§100.F)

### F.1 The §47 contract: the backend calculates, the LLM interprets

Not one number in an analysis is computed by the model. The bundle arrives with 880 facts (173 numeric) already derived by pure libraries, and the model's job is to say what they may imply. The §35 expectations-gap machinery makes the boundary concrete: `compute_expectations_gap_inputs` supplies the **inputs** — fundamental momentum in −1..1 from improved/weakened metric counts, run-up since the previous event, relative return, distance from the 52-week high, realised vol, material +/− development counts — and deliberately emits **no regime label**. The four §35 regimes are an enum in the model's output schema. The platform computes; the model classifies; the two are separately auditable.

### F.2 Schema and validator

`libs/llm/event_analysis.py` defines `EVENT_ANALYSIS_SCHEMA` (name `event_analysis`, `PROMPT_VERSION = "event-analysis-v1"`), a strict JSON schema carrying the §48 sections, §51 UPSIDE/BASE/DOWNSIDE scenarios, a §52 surprise-threshold narrative whose confidence may legitimately be `NOT_MEANINGFUL`, §50 `invalidation`, the `expectations_gap_regime` enum, `evidence_refs`, and `numbers_quoted[{path, value}]`. It is enforced at the provider seam — OpenAI's Responses API with strict `json_schema`, Anthropic's `output_config`, and a deterministic stub that quotes *real* facts from the bundle so the validator is exercised on every test run rather than only in production.

`validate_analysis(parsed, fact_index)` then rejects any quoted path missing from the bundle's `fact_index`, or whose value mismatches beyond 1e-6. **Violations are stored, not hidden**: the row persists with `status='INVALID'` and the UI renders the analysis *with* its violations banner. This is the single most load-bearing guardrail in the LLM stack — it converts "the model said 36.7" from a claim into a checkable assertion about provenance. It does not, and cannot, validate interpretation quality.

### F.3 Prompt-injection isolation (§81)

News is the only untrusted text in the bundle, and it enters exclusively through `sanitize_for_llm`, which strips markup, links and control characters, caps length, and **flags** instruction-shaped lines with `suspicious_instruction=True`. Flagged articles are excluded from the LLM view and counted. The display keeps provider bytes verbatim — nothing is censored, only flagged. Untrusted text never reaches a system role.

### F.4 Cache, digest and memory

`event_analyses` (migration 024) is `TradePlanRow` re-shaped: full payload JSON + versions + as-of + status + self-pointer. The cache key is the `bundle_digest`, and getting that key right required a live fix: the first implementation hashed the wall-clock `as_of`, so a second POST a minute later **missed the cache and spent a second LLM call** — which then hit the provider's 60 s read timeout. The digest now hashes a *pruned* view excluding volatile timestamps, `LLM_ANALYSIS_TIMEOUT_SECONDS` (240 s) is threaded to `analyze_event` only, and `GET /analysis` prefers the latest OK row while reporting `last_attempt` when the newest row failed.

§69 institutional memory works by re-entry: prior analyses are injected into later bundles as `tier: LLM_PRIOR`, as-of gated and OK-only, and the system prompt states they are **opinions, not evidence** (§70).

### F.5 Output language and failure behaviour

Narrative fields honour `Settings.llm_output_language` (`en` | `zh`; the live config is `zh`); machine-read fields — enums, paths, numbers — are never translated, which is why a Chinese headline still hashes and validates identically. Provider failure produces a stored `FAILED` row and **HTTP 200 with the bundle** — never a 500. An unconfigured LLM is a 503 `LLM_NOT_CONFIGURED`, matching the house `deps.py` honest-absence convention.

### F.6 What the live run cost, and what it found

AAPL event 99, OpenAI `gpt-5.6-sol`, zh: status OK in **51 s**, **28 485 input / 3 070 output tokens**, regime `BAD_NEWS_PRICED` at MODERATE confidence, three scenarios, surprise threshold at LOW confidence *citing the missing consensus*, and **zero validator violations**. The first live answer had cited **no** numbers at all — the prompt now lists quotable numeric facts and requires ≥3 `numbers_quoted`.

---

## G. Risk Integration (§100.G)

### G.1 What was added

`libs/trading_core/risk/event_risk.py` (`event-risk-1.0.0`), pure and statically pinned against LLM, network and engine imports:

- `historical_event_risk` — median / p75 / p90 / max of |moves| with **n ALWAYS present** (§64: a tail statistic without its sample size is a claim, not a measurement).
- `classify_event_risk` — a documented threshold table producing LOW / MODERATE / HIGH / EXTREME. Expected move is the implied move if one is stored, else the historical median, and **the basis is recorded** rather than inferred. Imminence bumps within 3 days; exposure share bumps. When nothing is known the state is **UNKNOWN, not LOW**, with a reason — a distinction the whole §64 discipline rests on. Option sensitivity is a *separate axis* (§66), never folded into the state.
- `event_risk_caps` — real `QuantityCap` rows (HIGH → 10 % NAV, EXTREME → 5 %, research defaults) that join **only** the hypothetical shadow verdict.

### G.2 How it reaches the decision path — and does not change it

Five minimal edits in `routers/orders.py`. `shadow["event"]` is computed **in its own try/except**, so a raising seam leaves the order path byte-identical (pinned by test). Its caps merge as `extra_caps=[*stress_caps_shadow, *event_caps_shadow]` into the Phase C *shadow* verdict only. **The real `assess(...)` still takes no `extra_caps`** — grep-pinned in `apps/`, on explicit user mandate, and the pre-existing AST tripwire from the risk programme still passes unchanged.

`_plan_payload` gains an `event_risk` block computed fresh on read, `None` when no event falls within 14 days. `GET /api/events/{id}/risk` serves the §66 options block, including a market-wide FOMC flag within 3 days.

### G.3 The live reading, and the bug that nearly shipped

`GET /api/events/99/risk` (AAPL 2026-10-29, T−71 d): state **LOW**, sensitivity LOW, enforcement **SHADOW**; historical n=8 — median |1.33 %|, p90 **7.35 %**; implied `None` (nothing stored yet for the upcoming event). The drivers and caveats are verbatim honest: *"position exposure unknown — not assumed small"*, *"event date is ESTIMATED"*, *"based on 8 event(s)"*. Trade Plans for RDW correctly carry `event_risk: null` — its next earnings is ~77 days out, beyond the horizon.

The Phase K verifier caught a HIGH-severity silent unit bug that **none of the implementing units saw**: `event_option_metrics` stores moves as **fractions** while the classifier speaks **percent**. Every real event's risk would have been understated **100×** — a HIGH would have read as LOW, silently, forever. It is fixed at one documented boundary with a regression test pinning the conversion. Expected IV crush is honestly `NO_DATA` (no forward surface is subscribed).

### G.4 Promotion path

Promotion out of SHADOW to WARN / RESIZE / REJECT requires two things, in order: a §86 measurement over materially more stored history than exists today, **and** an explicit user decision recorded in `runtime_config`. The thresholds in the table are research defaults and are labelled unvalidated wherever they are displayed. Nothing in this report should be read as evidence that 10 % or 5 % is correct.

---

## H. Validation (§100.H)

### H.1 The §96 adversarial look-ahead suite

`tests/test_event_lookahead.py` (1 405 lines, **46 tests** — 45 passed, 1 xfailed) is the programme's point-in-time proof. Its method: plant a **future artifact** strictly after `as_of` alongside a **past twin**, call the endpoint at `as_of=T`, and recursively scan the entire JSON payload for the future sentinel. The helper (`tests/_lookahead_util.py`, 127 lines) walks dicts, lists and strings — a leak into a nested note string fails exactly as a leak into a numeric field does.

Every case is a **PAIR**: future sentinel absent **and** past twin present. That second half matters more than it looks. A gate that returns nothing at all would satisfy an absence-only assertion perfectly; the paired assertion means a broken endpoint cannot masquerade as a secure one. Five further tests attack the *scanner itself* before any endpoint is trusted.

Coverage — all **12 endpoint families**, listed in the module docstring:

| # | Endpoint | Future artifact planted | Gate under test |
|---|---|---|---|
| 1 | `…/price-context` | daily bar dated after | `as_of_bar_filter` + the pure context's own `as_of_date_et` bound |
| 2 | `…/fundamentals` | statement accepted after | `select_statements_as_of` |
| 3 | `…/replay` | daily + minute bar after | `as_of_bar_filter` / replay window gate |
| 4 | `…/history` | a LATER earnings event row | `_past_comparable_rows` |
| 5 | `…/news` | article published after | `analyze_window` |
| 6 | `…/evidence` | all of the above at once | every seam's gate, composed |
| 7 | `…/analysis`, `…/analyses` | analysis row as-of after | `prior_analyses_for_ticker` |
| 8 | `…/options` | stored metric + later print | `_past_comparable_rows` + §85 basis rule |
| 9 | `…/macro` | observation released after | `macro.visible_prints` |
| 10 | `…/fed` | document released after | SQL bound + `fed_intel._gate` |
| 11 | `…/timeline` | article + filing + event | every kind's gate, composed |
| 12 | `…/risk` | later print + option metric | `event_risk._previous_prints` |

Plus a parametrized future-`as_of` → **422** test across 11 families.

### H.2 Mutation verification — the tests are proven to bite

The house rule from `test_risk_adversarial.py`: a passing test proves nothing unless you have watched it fail for the right reason. **Six** bite tests (the contract asked for ≥4) monkeypatch each gate to a pass-through *inside the test* and assert the sentinel becomes visible. Independently verified: disabling all six monkeypatches makes exactly those six tests fail, so every one is load-bearing. No source file was edited to do this.

### H.3 Defence in depth, mapped — and pinned so a cleanup cannot halve it

Three surfaces carry **two independent gates**: price-context (SQL bar filter + the pure context's own date bound — defeating only the first moves `bars_through` by one day and leaks nothing), news, and fed (SQL bound + pure gate).

For news and fed the bite tests use a **three-step proof**: clean → defeat the SQL bound only, assert *still clean* → defeat the pure gate, sentinel appears. Without that middle step, a future refactor deleting the SQL clause as "redundant" would reopen half the defence with every assertion still green. Fundamentals is **singly** gated by design — the gateway loads all rows unfiltered on purpose — which is exactly why its pure-layer coverage carries more weight.

### H.4 §85 NOT BACKTESTABLE — the live-basis rule

Tested three ways. A `LIVE_CHAIN_SNAPSHOT` row planted on the *past event itself* — newest, addressable, the most tempting row to serve — is still refused: `current.basis` stays `HISTORICAL` and the string `LIVE_CHAIN_SNAPSHOT` appears nowhere in the payload. The **converse** is also asserted (upcoming events may legitimately carry LIVE), so a platform that simply never emitted LIVE could not pass by doing nothing. And `not_backtestable` / `disclaimer` are asserted non-empty, because a limitation the payload does not state is a limitation the UI cannot show.

### H.5 One confirmed defect, pinned rather than patched

The suite found a **real §96 leak** and, being tests-only by contract, pinned it as a `strict=True` xfail that converts to a pass the moment it is fixed.

`apps/gateway/fundamentals.py:899-911` builds the `freshness` block from `rows[0]` — the newest **stored** row by `acceptance_datetime` across *all* rows, with no as-of gate. A filing accepted after `as_of` leaks four fields: `source_filing_url`, `latest_filing_date`, `period_end`, and `acceptance_datetime` — which reports an instant **later than the `as_of` the caller supplied**. The metrics themselves are correctly gated; only this block is not.

It propagates further than the endpoint. `bundle.fundamentals.freshness.source_filing_url` and `bundle.source_metadata[1].source_filing_url` carry it into the LLM evidence bundle — and because `numbers_quoted` validates the analysis *against the bundle*, a poisoned bundle validates perfectly. The leaked **values** are clean today, so no fabricated number reaches the model, but the surface is live. The fix is to build `freshness` from the gated statements; both xfails flip to passes when it lands.

One scaffolding caveat, recorded so it is not mistaken for a gate weakness: two tests needed the future event's `event_key` renamed away from the sentinel, because a past analysis legitimately reports a future event's key via `_analysis_summary`. The original assertion was firing on the test's own scaffolding. The §69 memory gate itself is sound and its bite test confirms it.

### H.6 The §86 measurement harness

`libs/trading_core/events/event_study.py` (728 lines, pure, stdlib) plus the DB-only seam `apps/gateway/event_study.py` (435 lines) and `GET /api/events/study?event_type=&min_n=&as_of=`. **40 tests**, all passing.

It measures nine features against signed 1-day and 5-day outcomes by Spearman rank correlation: `news_materiality`, `news_evidence_score_max`, `price_runup_pct`, `distance_from_52w_high`, `realized_vol_20d`, `fundamental_momentum_score`, `implied_move_pct`, `historical_median_move`, `iv_before`.

Four design decisions carry the honesty of the whole instrument:

1. **Rank correlation is the general Pearson-of-ranks form, not the `1 − 6Σd²/(n(n²−1))` shortcut.** The shortcut is wrong in the presence of ties, and ties are the *normal* case here (materiality counts, momentum scores). With `scipy` unavailable to cross-check, `average_ranks` was property-tested over 300 random tie-heavy vectors and rho hand-computed in four cases — including one tied case where the shortcut gives 0.95 and the correct answer is 0.9486832980505138. That third-digit divergence is precisely what the hand-computed assertion exists to catch.
2. **Features come from the EARLIEST stored bundle per event, never a fresh assembly.** This is the §96 gate for the harness itself. An event re-analysed *after* the print would carry a run-up measured *through* the reaction it is supposed to predict — a leak that would inflate exactly the correlations the report exists to state honestly. Mutation-verified: flipping the `order_by` to descending fails the test.
3. **`LIVE_CHAIN_SNAPSHOT` metrics are excluded** (§85) — a live snapshot is written whenever someone opened the Options tab, possibly days after the print. Mutation-verified.
4. **Outcomes are deliberately measured with hindsight**, from unfiltered bars, because the realised reaction is the thing being predicted. This asymmetry is spelled out in the code so it cannot be misread as a missed gate.

The report emits `rho` and `n` and a `NOT_MEANINGFUL` flag below `MIN_MEANINGFUL_N = 12`, with seven fixed caveats — **no p-value, ever**, because a p-value at these sample sizes would be certainty theatre (§92). `min_n` can only *raise* the flag threshold; the route 422s below 12, and a test asserts that changing it re-labels cells but provably cannot change a rho or the sample. Two §86 candidates are **named as unmeasurable** rather than silently dropped: `estimate_revision` (no consensus vendor) and `valuation_expansion` (filing depth equals backfill depth, so the measurement would describe the backfill). A silently short table would have read as a claim that §86 was fully covered.

**No conclusions are drawn.** The instrument is built, tested and deployed; the history to run it on is not yet there.

### H.7 Test totals

**Backend: 2020 (Phase A baseline) → 4000 passed, 1 skipped, 1 xfailed** in 113 s. The single xfail is the §H.5 defect, pinned `strict=True`.

Per-file counts for the catalyst surface (collected 2026-08-19):

| Suite | Tests | Suite | Tests |
|---|---:|---|---:|
| `test_events_news_intel` | 180 | `test_events_analysis_api` | 41 |
| `test_events_replay` | 112 | `test_events_study` | 40 |
| `test_events_fundamentals` | 109 | `test_events_timeline_api` | 39 |
| `test_events_implied_move` | 89 | `test_events_options_api` | 39 |
| `test_event_calendar_providers` | 87 | `test_events_evidence` | 38 |
| `test_events_fed_intel` | 80 | `test_event_calendar_loop` | 38 |
| `test_events_reaction` | 73 | `test_events_fed_api` | 33 |
| `test_events_models` | 70 | `test_event_risk_api` | 25 |
| `test_risk_event` | 66 | `test_events_macro_api` | 24 |
| `test_events_replay_api` | 62 | `test_events_db` | 20 |
| `test_events_macro` | 60 | `test_catalyst` | 3 |
| `test_fed_docs` | 58 | | |
| `test_events_news_api` | 57 | | |
| `test_event_calendar_macro` | 52 | | |
| `test_llm_event_analysis` | 50 | | |
| `test_events_fundamentals_api` | 47 | | |
| `test_event_lookahead` | **46** | | |
| `test_events_price_api` | 45 | | |
| `test_events_api` | 43 | | |

### H.8 What the suite caught that review did not

Recorded because it is the honest measure of what validation earned:

- **The 100× unit error** (Phase K): fractions stored, percents classified. Every event's risk understated by two orders of magnitude, silently.
- **The `"1D"` vs `1d` key mismatch** (E1): every reaction cell would have rendered Unavailable.
- **10 UI↔wire key mismatches in one tab** (G), and 2 more in H — all hidden from `tsc` by an index signature.
- **Single-link cluster collapse** (D): the first live run fused **269 of 283** AAPL articles into one cluster through Benzinga's daily templated headlines. Only real data exposed it; the fix (leader-based clustering, template-word stripping, a 40 % cap, entity down-weighting above 33 % document frequency) is pinned to an exported live fixture of the same 283 articles.
- **The digest that included wall-clock time** (F): a second POST a minute later spent a whole extra LLM call and then timed out.
- **The provider that stored 0 rows** (E2): the market-data provider was Alpaca, which has no financials — statements and prices need *different* providers.
- **`audit_events.entity_id` VARCHAR(64) overflow** on FED_SPEECH natural keys (B) — invisible in SQLite, fatal in Postgres.

---

## I. Deferred Features (§100.I)

Deferred deliberately, each with the reason it was deferred rather than a promise.

**Blocked by subscription — no engineering will fix these.**

- **Consensus, estimate revisions, guidance, ratings, analyst insights.** Massive `/benzinga/v1/*` is a permanent 403. Consequence: **EPS/revenue surprise % is not computable for any period**, §34 revision trend is not deliverable, and the §52 surprise threshold stays a *narrative* construct. The platform renders the fixed string `CONSENSUS DATA UNAVAILABLE`; `tests/test_no_synthetic_data.py` is the standing guard against anyone later filling these with an estimate. The Benzinga Earnings add-on would also convert every ESTIMATED earnings date to CONFIRMED.
- **Historical option quotes, greeks and OI** — not sold at any probed tier. Historical implied moves remain a daily-close straddle *approximation*, labelled `HISTORICAL_DAILY_CLOSE_APPROXIMATION`; IV crush is NO_DATA for the weekly expiries most event straddles use. Massive's option aggregates additionally stop at ~2 years (found live, 403 "data timeframe").
- **Fed-funds futures pricing.** No source. Market-implied policy expectations are honestly UNAVAILABLE; the 2Y change is a labelled proxy, never presented as a probability of a cut.

**Deferred by scope, buildable when wanted.**

- **Census / ISM adapters** — RETAIL_SALES and ISM actuals. Their release *dates* are tracked where a schedule exists; the actuals need one adapter each.
- **SEP / dot plot ingestion** and **press-conference transcripts** — the transcripts are PDFs, currently linked rather than parsed.
- **The post-event writer (§68).** `event_analyses` already carries `kind=POST_EVENT`, so the memory shape exists; nothing writes the actual-result-and-reaction row yet. This is the single highest-value remaining item, because it is what would let the platform compare what it said with what happened.
- **Event graph (§74)** — cross-event causal linkage.
- **Backtest integration (§87).** The §87 boundary is respected deliberately: nothing in this programme writes into the backtest engine. Event features are not backtestable in the honest sense until enough point-in-time history has accrued *under this platform's own ingestion*, since the vendors do not sell the vintages retroactively.
- **Intraday macro reaction windows** (BEFORE_MARKET prints currently use daily bars); **sector benchmarks** beyond SPY; **price markers on the timeline**; **volatility surface / skew analytics**.

**Deferred pending data, not code — this is the §86 boundary.**

The measurement harness exists and is tested. **Its conclusions do not.** Correlations over n = 8 events are not evidence, which is why the payload flags anything below n = 12 `NOT_MEANINGFUL` and why the event-risk thresholds stay research defaults in SHADOW. Promotion of event risk to any enforcing mode requires both a measurement over materially more history and an explicit user decision. The honest statement today is: *the instrument is calibrated; the sample is not yet there.*

---

## §101 — FACT / QUANT / ANALYSIS, and where the line is drawn

The spec's guiding principle is that three different kinds of claim must never collapse into one opaque AI score. In this platform the separation is not a UI convention — it is enforced at four layers simultaneously:

| Tier | What it is | Where it is enforced |
|---|---|---|
| **FACT** (`tier: DATA`) | What objectively happened, from a source with a publication timestamp | Bundle sections tagged `TIER_DATA`; the as-of gate filters on `acceptance_datetime` / `published_at` / `release_at` / bar `t` — **never** on a period-end date |
| **QUANT** (`tier: QUANT`) | What this platform's deterministic code computes from those facts | Pure `libs/` modules, versioned (`f1-evidence-v1`, `news-intel-v1`, `implied_move-1.0.0`, `fed-intel-v1`, `event-risk-1.0.0`), statically pinned against I/O imports |
| **ANALYSIS** (`tier: LLM`) | What the model believes those may imply | Schema-validated output; every quoted number checked against `fact_index`; violations stored and displayed |
| **PRIOR OPINION** (`tier: LLM_PRIOR`) | What the model said before | A separate tier precisely so yesterday's interpretation cannot be mistaken for today's evidence (§70) |

Three concrete refusals to collapse are worth naming, because each was a decision that could have gone the other way:

- **No aggregate hawkish/dovish score.** `fed_intel` reports eight §43 policy dimensions separately and exports `NO_SINGLE_SCORE_NOTE`; a test asserts mechanically that no aggregate-score export exists.
- **No regime label from the backend.** `compute_expectations_gap_inputs` yields inputs; the four §35 regimes are the model's enum.
- **No probability from a sample of eight.** `history_stats` always carries `n`, `n_available` and `positive_count` — never a probability (§19/§64).

## §102 — What the platform answers today

The spec's final product question, answered honestly per row.

| Question (§102) | Answered by | Honestly? |
|---|---|---|
| What is happening? | Event registry, typed taxonomy (18 `EventType`s) | **Yes** |
| When is it happening? | `scheduled_at` UTC + `event_timezone` + session | **Earnings: ESTIMATED, badged.** Macro/Fed: CONFIRMED from primary sources |
| What happened last time? | Previous-event linkage (104/110 rows) + replay | **Yes** — 12 past releases per ticker |
| How did the market react last time? | E1 daily, C intraday, G multi-asset, H two-window FOMC | **Yes** — to the minute where bars exist (§D.1) |
| What has materially changed since then? | D news engine over the previous-event → as_of window | **Yes** — 177 clusters, 104 material, 14 themes (§D.1) |
| What do the fundamentals say? | E2 point-in-time statements | **Yes, with named absences** — FCF, EBITDA, net debt and quick ratio are "not reported by provider", never 0 |
| What does price behaviour say? | E1 positioning | **Yes** (§D.1) |
| What does the option market appear to be pricing? | I implied move | **Yes, as pricing not forecast.** Across 7 priced prior AAPL events: implied median 4.16 % vs realised median 1.33 % — 5 OVER_PRICED, 2 FAIR |
| What expectations are embedded? | §35 inputs + LLM regime | **Partially.** Price/news proxies only — no consensus exists to compare against |
| What are the unresolved questions? | LLM analysis §48 sections | **Yes**, as interpretation, tiered as such |
| What would be a genuine surprise? | §52 surprise threshold | **Narrative only**, at LOW confidence, *citing the missing consensus* as the reason |
| How much portfolio risk do I have around this event? | K event risk | **Yes, and it decides nothing** — SHADOW, with n on every statistic |

Two rows in that table read "partially" or "narrative only", and both trace to the same 403. That is the shape of the programme's central limitation, and it is stated in the payload, in the UI, and here — rather than papered over with an estimate.

---

## Appendix — programme facts

| | |
|---|---|
| Phases | A, B0+B, E1, E2, C, D, F, I, J, G, H, K, L (13 DEVLOG entries, (21)–(33)) |
| Migrations | 021–027, all applied live by hand (no migration runner), all `IF NOT EXISTS` |
| New pure modules | 12 under `libs/trading_core/events/` (incl. Phase L `event_study.py`) + `risk/event_risk.py` |
| New provider adapters | 9 under `libs/event_calendar/` besides `provider.py` + `stub.py` (4 keyless: `sec_edgar`, `fed`, `bls`, `bea`) |
| New gateway seams | 13 under `apps/gateway/` |
| API routes | 28 on `routers/events.py` |
| UI | 1 nav destination, 2 pages, 13 event tabs, 25 components + 11 format helpers |
| Backend tests | 2020 → **4000 passed, 1 skipped, 1 xfailed** |
| UI tests | 114 → **797** |
| ADRs written | ADR-008 (event calendar registry, source precedence), ADR-009 (keyless government adapters, macro reference symbols) |
| Open defect | 1, pinned `strict=True` xfail — `fundamentals.py` `freshness` block (§H.5) |
