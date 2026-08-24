# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository instead.

Useful to include: what you can reach, what you had to know or hold to reach
it, and whether it crosses one of the boundaries below.

## What this project treats as a security boundary

Three, in order of severity:

1. **Execution isolation.** Nothing in the research path — web search,
   prediction markets, news, or any LLM output — may reach instrument
   selection, sizing, the gate chain, or order placement. A way to make
   research input move an execution row is the most serious bug this project
   can have. It is defended by structural tests
   (`test_research_safety_adversarial.py`, `test_research_e2e_adversarial.py`)
   and by the absence of any import path between the two.
2. **Credential confinement.** API keys live in the `runtime_config` table.
   They are never logged (the logging layer redacts
   `api_key|secret|password|token|authorization`), never returned by an
   endpoint (the config API answers with booleans), and never sent to the
   browser. A path that echoes one is a vulnerability.
3. **Untrusted third-party text.** Web pages, news articles and
   prediction-market questions are attacker-controlled. They are sanitized and
   fenced before reaching a model. A payload that escapes the fence and
   changes model behaviour is a vulnerability — the sanitizer has already
   needed one fix of exactly this kind (HTML entities that reconstituted a
   closing fence tag after decoding).

## What is not a vulnerability

- **Losing money.** This is research software; see the disclaimer in
  [NOTICE](NOTICE). Bad analysis is a bug, not a security issue.
- **A vendor rate-limiting or blocking you.** Identify yourself honestly in
  `SEC_USER_AGENT` and respect the terms of each source.
- **The default local Postgres password** (`trading`), which is a development
  convenience from `docker-compose.yml`. Change it before exposing the
  database anywhere.

## Running it safely

- Keep `TRADING_ENABLED=false` unless you are deliberately trading.
- Start with a **paper** broker account. Use a live one only after you have
  watched the paper one behave for a while.
- Never commit `.env`. It is gitignored; keep it that way.
- The gateway has no authentication layer — it assumes it is bound to
  localhost. Do not expose it to a network without putting your own
  authentication in front of it.
