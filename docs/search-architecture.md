# External Web Search Architecture

**How the platform searches the public web for catalyst research, and what it
refuses to do with what it finds.**

Last updated: 2026-08-21 (Catalyst research upgrade, LOOP 2/3/8/10).
Companion documents: `prediction-market-architecture.md`,
`event-research-orchestration.md`.

## The one-sentence version

Search is a **bounded, point-in-time, operator-triggered** retrieval of public
documents about one catalyst, whose results are normalized into deterministic
evidence rows before any model sees them — and which carries **zero execution
authority**.

## Non-negotiable principle

Search does not give the model the internet. It gives the platform controlled
access to external documents, which are then admitted (or refused) by
deterministic code. The model reasons only over what the platform admitted.

```
Brave Search API
      ↓  bounded queries, point-in-time window
raw SearchResult (provider's words, untrusted)
      ↓  canonical URL · dedup · as-of gate · relevance · source tier · topic
EvidenceCandidate (accepted or refused, both stored)
      ↓  safe_* text only
web_research bundle section
      ↓  <untrusted_web_research> fence
LLM
```

## Provider layer

`libs/web_search/` — a third sibling of the existing provider registries
(`libs/market_data`, `libs/event_calendar`, `libs/llm`), deliberately the same
shape rather than a new framework:

| Piece | Responsibility |
|---|---|
| `provider.py` | `WebSearchProvider` Protocol (`search_web`, `search_news`), `SearchResult`, `WebSearchError` |
| `brave.py` | Brave Search API adapter (keyed: `BRAVE_API_KEY`) |
| `stub.py` | Deterministic offline provider — **opt-in only, never a fallback** |

Registry rules, inherited from the house pattern: no default provider, no
cross-provider fallback, `ProviderNotConfigured` for "no key" and `ValueError`
for "unknown name". An unconfigured install reports `NOT_CONFIGURED` and every
unrelated event page keeps working.

### Reliability (Brave adapter)

- One request chokepoint (`_request`), mirroring the Alpaca adapter's shape.
- 429 → **one** retry, honouring `Retry-After` only when it is a finite number
  of seconds, capped at `MAX_RETRY_AFTER_SECONDS`.
- 401 names `BRAVE_API_KEY` **without echoing it**. The key never appears in a
  response, a log line, a metric label or an audit row.
- 403 → `CapabilityNotAvailable` (a plan that does not include this endpoint is
  a capability fact, not an outage).
- Minimum request interval (`MIN_REQUEST_INTERVAL_SECONDS`) under a lock.
- Parsing is bounded by the requested `count` and skips malformed items
  individually — one bad result never sinks a query.

## As-of policy (the point-in-time rule)

Two bounds, both enforced in the pure layer and both re-applied on read:

- `published_at <= as_of` — a document published after the research instant did
  not exist then.
- `published_at >= window_start` — the window opens at the **previous
  comparable event**, because "what changed since last time" is the question.

A document with **no publication time** is not assigned one. It is admissible
only via `retrieved_at <= as_of`, and the exclusion is counted
(`excluded_by_as_of`) rather than hidden. `REJECT_UNPLACEABLE_IN_TIME` is a
real, stored outcome.

The SQL bound on the read path is an **optimisation, never the contract** —
the Python gate re-runs over every stored row, so no storage or replay path
can leak a later document into an earlier instant.

## Source hierarchy

Search rank is **not** evidence reliability. Every admitted document carries a
tier, and the tier is metadata — never a binary truth flag:

| Tier | Meaning | Examples |
|---|---|---|
| `OFFICIAL` | The issuing authority itself | SEC, BLS, BEA, Federal Reserve, Treasury (`.gov` rule) |
| `PRIMARY` | The subject's own publication | company IR, newsroom, filings |
| `HIGH_QUALITY_NEWS` | Professional journalism | Reuters, Bloomberg, WSJ, FT, AP |
| `INDUSTRY` | Trade press | |
| `SECONDARY` | General web | |
| `SOCIAL` | Retail/narrative | reddit.com, twitter.com, x.com |
| `UNKNOWN` | Unclassified domain | the honest default |

The mapping is a maintainable table (`DOMAIN_TIERS`), not a hardcoded belief.

`SOCIAL` is **reachable today**: a general web search can surface a Reddit or
X thread, and when it does the document is admitted wearing its SOCIAL tier
rather than being silently dropped — the tier is how a reader tells it from a
BLS release. What Phase 17 defers is a dedicated Reddit *provider* (an
ingestion path built for social content), not the classification. Nothing
social-specific leaks into this schema, so adding that provider later is new
rows rather than a schema change.

## Cost control

Every bound is a named constant in `libs/trading_core/events/web_research.py`,
so "what can one button press cost?" is answered by reading them:

| Constant | Value | Bounds |
|---|---|---|
| `MAX_QUERIES_PER_EVENT` | 6 | queries issued per press |
| `MAX_RESULTS_PER_QUERY` | 10 | results requested per query |
| `MAX_UNIQUE_DOCUMENTS` | 40 | distinct documents considered |
| `MAX_ACCEPTED_EVIDENCE` | 20 | documents admitted to the bundle |

Plus a per-event throttle (`RESEARCH_ATTEMPT_SECONDS`, 1 hour) so a second
press within the hour costs nothing and reports `RECENTLY_REFRESHED`.

**A GET never searches.** This is the load-bearing cost rule: the Catalyst page
and its React Query polls read stored rows only. `POST
/api/events/{id}/research/backfill` is the sole path that spends, and it
records what it bought (`queries_executed`, `results_considered`,
`results_accepted`) in both the run row and the audit row.

## Query planning

Deterministic and event-type-aware (`RESEARCH_PROFILES`): an earnings event
asks about guidance, margins and analyst revisions; a CPI release asks about
shelter, services and Fed commentary. Queries carry a `purpose` and a
`priority`, are capped, deduped, and **contain no dates** (the window is a
parameter, not prose).

An LLM-assisted planner is permitted by the design but is **not implemented**:
the deterministic planner is the baseline and would remain the fallback. If one
is ever added, deterministic code still enforces the query count, the dates and
the domains.

## Untrusted input

All search text is untrusted. It travels through the same discipline as news:

- `sanitize_for_llm` strips markup and control characters, then **re-strips
  after entity decoding** — an encoded `&lt;/untrusted_web_research&gt;` would
  otherwise reconstitute into a live closing tag and break the prompt fence.
- `suspicious_instruction` flags injection-shaped text. Flagged rows are
  **stored** (diagnostics), **refused** (never admitted), and **counted**
  (`suppressed_suspicious`) — visible to the operator, invisible to the model.
- The bundle section carries `safe_title` only. **No raw titles, no snippets,
  and no URLs** reach the model: a URL in evidence text is an exfiltration
  invitation, and the system prompt separately bans the model from emitting
  one.
- Model-facing web evidence is fenced in `<untrusted_web_research>`.

## Failure semantics

| Situation | Result |
|---|---|
| No API key | 200, `NOT_CONFIGURED` — never an exception |
| Unknown provider name | 200, `NOT_CONFIGURED`, throttle **not** armed |
| Some queries fail | Run stored `PARTIAL`, failures named in `skipped`, good evidence kept |
| Every query fails | Run stored `FAILED`; the read side keeps serving the last good run |
| Throttled | 200, `RECENTLY_REFRESHED`, nothing spent |

A degraded outcome is always a 200 with a named reason. A button press must
report why nothing arrived, never 5xx.

## Execution isolation

`libs/web_search` and the research pure layer import **nothing** from
`libs.broker`, `libs.trading_core.risk`, `libs.trading_core.strategies`,
`libs.trading_core.signals`, or the order/trading-pool routers — and those
modules import nothing back. Enforced by
`tests/test_research_safety_adversarial.py` (import-graph AST, both
directions) and `tests/test_research_e2e_adversarial.py` (the whole chain run
under injection, with the execution tables row-diffed to prove zero writes).
