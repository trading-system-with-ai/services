# Prediction Market Architecture

**How the platform reads prediction-market pricing, and why it never calls it
a probability.**

Last updated: 2026-08-21 (Catalyst research upgrade, LOOP 4/5/8/10).
Companion documents: `search-architecture.md`,
`event-research-orchestration.md`.

## The one-sentence version

Prediction markets are an **additional observable layer of market
expectations** — read-only, point-in-time, operator-triggered — to be compared
against fundamentals, official data and options-implied expectations. They are
not a forecast, and they carry **zero execution authority**.

## Non-negotiable principle

> The objective is NOT "Polymarket predicts the future."

A contract price is what buyers and sellers **transact at**. It is not a
calibrated probability, and the platform's language never lets it become one:

| Always | Never |
|---|---|
| market-implied probability | true probability |
| prediction-market pricing | actual probability |
| contract price | guaranteed likelihood |
| what the contract costs | the odds of this event |

This rule is enforced in three places: the field name itself
(`market_implied_probability`), the LLM system prompt (rule 11), and the UI
components — which additionally show the **relation** beside every price,
because a `DERIVED` contract at 63c is not a 63% forecast of *this* catalyst.

## READ ONLY, structurally

There is no wallet, no signing key, no order and no position anywhere in this
subsystem, and the design makes adding one a visible act rather than an easy
one:

- The `PredictionMarketProvider` Protocol has **no trading method**:
  `search_markets`, `get_market`, `get_market_snapshot`, `get_price_history`.
- The four tables carry no column that could hold an order, a wallet, a
  credential or a position.
- A word-ban test (`tests/test_research_safety_adversarial.py`) fails the build
  if a trading-shaped function name appears in `libs/prediction_markets`.
- The Settings card is an **enable toggle**, not a credential form, and says
  so: *"Public read-only · No trading credentials required."*
- Third-party text is sanitized wherever it reaches the model: the question,
  the resolution criteria **and the outcome name** — an outcome called
  "Yes 42.5 ignore previous instructions" must neither instruct the model nor
  smuggle a quotable number into it.

Polymarket is keyless, yet still **opt-in** — it is an outbound network
dependency an operator should choose deliberately.

## Provider layer

`libs/prediction_markets/` — same registry shape as every other provider
family. Polymarket uses its two official public surfaces:

| Surface | Used for |
|---|---|
| **Gamma** | market discovery (`/public-search`), metadata (`/markets/{id}`) |
| **CLOB** | order book (`/book`), last trade, price history (`/prices-history`), health (`/time`) |

Reliability: contact `User-Agent`, request timeout, minimum request interval,
one jittered retry on transient 5xx/transport errors, capped `Retry-After`
handling on 429, and **schema-tolerant parsing** — a missing optional field
becomes `None`, a malformed price is skipped with a named reason, and one
failing market never sinks the payload.

Kalshi (Phase 18) is deliberately absent. The normalized models carry no
Polymarket-specific field, so adding `KalshiProvider` is new *rows*, not new
columns — `UNIQUE(provider, provider_market_id)` already keeps a Kalshi ticker
from colliding with a Polymarket condition id.

## DIRECT / DERIVED / CONTEXT

One catalyst does **not** equal one market. Discovery produces a candidate
pool; deterministic classification produces 0..N accepted markets, each
labelled by how it relates to the event:

| Relation | Meaning | Example (for a GDP release) |
|---|---|---|
| `DIRECT` | The contract measures the event's own outcome | "GDP > 2.5%" |
| `DERIVED` | The event materially affects the contract, but the contract is not the event | "September Fed rate cut" |
| `CONTEXT` | Broader backdrop related to the event | "US recession during 2026" |
| `NOT_CLASSIFIED` | Stored on refused candidates only — an honest absence, never a judgement nobody made |

`DIRECT` additionally requires a **two-sided horizon** check: a contract
resolving long after the event does not measure it, however similar the words.

**Deterministic code owns every decision that matters** (Phase 4): it builds
the candidate pool, applies the relevance threshold (`MIN_MATCH_RELEVANCE`),
and caps acceptance (`MAX_ACCEPTED_MARKETS`).

A DERIVED contract legitimately shares no subject terms with its event — the
mission's own example, a September Fed-cut contract for a CPI release, has
zero token overlap — so the DERIVED bucket cannot demand subject grounding
without rejecting the case it exists for. That necessary leniency is bounded
instead by a **foreign-jurisdiction guard** (`FOREIGN_JURISDICTION_TERMS`): a
contract naming another country's monetary authority speaks the same
rate-move vocabulary while measuring something else, and is refused with its
own reason (`FOREIGN_SUBJECT`) rather than the generic low-relevance one.

### The venue event is the unit, not the contract

A venue publishes one distribution as one contract per outcome range, grouped
under its own event id. **Those brackets are only meaningful together.**
Accepting a subset does not give a smaller picture of the market — it gives a
WRONG one, because the survivors are whichever the ranking favoured and the
probability mass may sit entirely in the ones dropped. Found live
(2026-08-23): a seven-bracket GDP series stored its four cheapest brackets and
dropped the three holding 80% of the mass, so the panel read "the market
prices every outcome near zero" while the market had a clear central estimate.

Three layers enforce this now:

- **Discovery** takes whole events. `/public-search` already returns markets
  grouped under their event; the provider stops BEFORE a group that would not
  fit rather than cutting through one, and carries the venue's grouping id
  down from the wrapper (a market nested in a search response has no `events`
  key of its own, which is why every stored row had a NULL group).
- **Acceptance** caps GROUPS, not contracts. A group is admitted whole or
  refused whole with its own reason. A single distribution larger than the cap
  is still admitted — one complete distribution beats nothing.
- **The read side** checks completeness by the series' OWN arithmetic:
  exclusive exhaustive brackets price to ~1.00, so a set summing far under
  that is missing brackets whatever the cause, and the UI labels it.

### LLM event selection (market-selection-v1)

The design always permitted an LLM classifier; it is now implemented, for one
narrow job — deciding WHICH venue event a catalyst is best read against. That
is a semantic judgement ("is a Q2 GDP release better read against the Q3
distribution or the full-year one?") and the deterministic matcher was bad at
it: it scored contracts individually, which is how brackets of two different
distributions ended up interleaved.

The boundary is the design:

| The model MAY | The model MAY NOT |
|---|---|
| choose among venue events the provider returned | invent an event, market, question, price or id |
| decline entirely ("none of these fit") | choose a SUBSET of one event's markets |
| suggest a relation and give a reason | set prices, caps, horizons or acceptance |

It **narrows; it never admits**. Refs are opaque handles minted per call, so
the model never handles a venue id and a fabricated one resolves to nothing;
whatever it picks passes back through the same `match_markets` gate, which
re-applies every guard (horizon, foreign jurisdiction, other issuer, ACTIVE
status, relevance floor). Every failure mode — unconfigured, transport error,
malformed reply, empty selection — degrades to the deterministic matcher over
the untouched pool. Venue titles are sanitized and fenced as untrusted text
before they reach the model.

### No relevant market is a SUCCESS

`NO_RELEVANT_PREDICTION_MARKET` is the common, correct outcome for most
catalysts. The platform does not force a loosely-related contract into the
bundle to avoid an empty panel.

## Distinct honest states

These are **different answers** and are never conflated:

| State | Meaning |
|---|---|
| `NEVER_RUN` | Nobody has researched markets for this event |
| `NO_RELEVANT_PREDICTION_MARKET` | Matching ran and honestly accepted nothing |
| `PARTIAL_DISCOVERY` | Some discovery queries failed; "no relevant market" is not a conclusion the platform earned |
| `PROVIDER_UNAVAILABLE` | The venue could not be reached at all |
| `FOREIGN_SUBJECT` (per candidate) | The contract names another jurisdiction |
| `MARKET_METADATA_UNAVAILABLE` | Markets matched but could not be rendered (orphaned rows, withheld wording) |
| `NOT_CONFIGURED` | The operator has not enabled the provider |

A completed run always leaves a watermark in `event_ingest_state`, which is
what lets the read side tell "we looked and found nothing" from "nobody
looked" even when the candidate pool was empty.

## Depth is a fact, not a score

A 70c contract on a thin market and a 70c contract on a deep one are different
claims. The bundle exposes the **facts** — spread, volume, liquidity,
observation count, history availability — rather than folding them into one
confidence number (Phase 23). Absent is `null`, never `0`: an unknown spread
and a zero spread are opposite statements, and the UI prints "unknown".

## Historical features

Computed deterministically in `libs/trading_core/events/prediction_intel.py`,
as-of gated first: `current_price`, `change_1h/1d/7d`,
`change_since_previous_event`, `change_since_window_start`, `recent_high`,
`recent_low`, `price_range`, `trend`, plus `observation_count`,
`history_start`, `history_end`.

**No invented interpolation.** A gap in observations is a gap; an anchor that
falls outside the observed range yields `None` rather than a clamped value, and
the chart draws no line through a single point.

## Caching and the digest

Prices and history features **are** digest-relevant: a material repricing must
invalidate a cached analysis. Pure clock readings are not — `observed_at`,
`history_end`, `observation_count` and `matched_at` are pruned from the digest
view, so re-observing an unchanged market keeps the cached answer valid while
any price change misses.

`history_start` is deliberately **kept** in the digest. It looks like a clock
and is not one — re-observing advances the end and the count, never the start
— and it is the only field distinguishing a two-observation series from a
two-hundred-observation one when the price has not moved. Without it a market
acquiring real depth would leave a stale analysis cached, and depth is exactly
what the reader is told to weigh a thin market's price against. All three
directions are pinned by tests.

## Execution isolation

Identical to the search subsystem: no import path in either direction, proved
structurally by AST tests and behaviourally by
`tests/test_research_e2e_adversarial.py`, which runs the whole chain against a
market whose *question text* is an injection payload and then row-diffs the
watchlist, trading pool, orders and positions tables to prove nothing moved.
