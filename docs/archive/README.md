# Archive

Development history: the log, the audits, and the phase reports written while
the platform was being built.

**These describe how the system got here, not how it works today.** For current
behaviour read the documents one level up:

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — subsystem map and ADRs
- [`../data-source-architecture.md`](../data-source-architecture.md)
- [`../prediction-market-architecture.md`](../prediction-market-architecture.md)
- [`../search-architecture.md`](../search-architecture.md)
- [`../event-research-orchestration.md`](../event-research-orchestration.md)

They are kept because a design decision is easier to evaluate — and to
overturn — when you can see the problem it was answering. `DEVLOG.md` in
particular records what broke and what the fix assumed, which is usually the
missing context when a rule looks arbitrary.

## What is in here

- `DEVLOG.md` — the running development log.
- `CURRENT_PLATFORM_ARCHITECTURE.md` — a full architecture audit, written
  before the search / prediction-market work began.
- `SEARCH_PREDICTION_MARKET_UPGRADE_PLAN.md` — the plan for that upgrade.
- `catalyst-event-*`, `risk-engine-*` — phase audits and completion reports.
- [`specs/`](specs/) — the ORIGINAL DESIGN BRIEFS the platform was built from.
  These are the most useful documents here for a newcomer: they state the
  constraints (point-in-time correctness, honest empty states, human-approved
  execution) that every later rule descends from.

Expect stale details: file paths, counts and provider names in here reflect the
day each was written.
