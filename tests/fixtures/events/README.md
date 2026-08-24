# Event calendar fixtures

Every fixture here is LIVE-DERIVED: downloaded from the real source with the
contact User-Agent (`settings.sec_user_agent`) and trimmed to the relevant
region, never hand-written. Parsing is pinned to real markup so a layout
change shows up as a failing test rather than as silently wrong dates.

## Phase G — macro (U1), downloaded 2026-08-19

| File | Source URL | Trim |
| --- | --- | --- |
| `bls_schedule_cpi.html` | https://www.bls.gov/schedule/news_release/cpi.htm | `<h2>` heading + `<table class="release-list">` |
| `bls_schedule_ppi.html` | https://www.bls.gov/schedule/news_release/ppi.htm | same |
| `bls_schedule_empsit.html` | https://www.bls.gov/schedule/news_release/empsit.htm | same |
| `bls_schedule_jolts.html` | https://www.bls.gov/schedule/news_release/jolts.htm | same |
| `bea_schedule.html` | https://www.bea.gov/news/schedule | `<table id="release-schedule-table">` only |
| `bls_series_cusr0000sa0.json` | https://api.bls.gov/publicAPI/v1/timeseries/data/CUSR0000SA0 | untrimmed (2.9 KB) — CPI-U all items, SA |
| `treasury_yield_curve_2026.csv` | https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2026/all?type=daily_treasury_yield_curve&field_tdr_date_value=2026&_format=csv | first 60 rows |

Layout facts these fixtures pin (verified live 2026-08-19):

* Every BLS schedule page uses ONE `<table class="release-list">` with the
  columns `Reference Month | Release Date | Release Time`, dates as
  `Feb. 13, 2026` (abbreviated month, sometimes zero-padded day: `Jan. 09, 2026`).
* Release times are NOT uniformly 08:30: CPI/PPI/Employment Situation are
  `08:30 AM`, **JOLTS is `10:00 AM`**. The time is parsed from the page, never
  assumed.
* The BEA table carries the year ONLY in its `<th>Year 2026</th>` header; each
  row's date cell is `<div class="release-date">August 26</div>` plus a
  `<small class="text-muted">8:30 AM</small>`, so the year comes from the header
  and rows that wrap past December roll to the next year.
* BEA titles GDP releases as `GDP (Advance Estimate), 3rd Quarter 2026` — the
  string "Gross Domestic Product" does NOT appear on the page — and PCE as
  `Personal Income and Outlays, July 2026`.

## Phase H — Fed documents (U1), downloaded 2026-08-19

Downloaded with `curl -A "$SEC_USER_AGENT"` (the contact User-Agent
federalreserve.gov requires; anonymous requests are 403ed).

| File | Source URL | Trim |
| --- | --- | --- |
| `fomc_statement_2026-07-29.html` | https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm | `<div id="article">` … matching `</div>` only |
| `fomc_statement_2026-06-17.html` | https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm | same |
| `fomc_minutes_2026-06-17.html` | https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm | same |
| `fed_press_monetary.xml` | https://www.federalreserve.gov/feeds/press_monetary.xml | untrimmed (9.6 KB, 15 items) |

Layout facts these fixtures pin (verified live 2026-08-19):

* A press release wraps everything in `<div id="article">` holding TWO
  columns: `div.heading.col-xs-12.col-sm-8.col-md-8` (`p.article__time`
  "July 29, 2026", `h3.title`, `p.releaseTime` "For release at 2:00 p.m. EDT",
  plus a social-share `<ul>` that is pure navigation) and a sibling
  `div.col-xs-12.col-sm-8.col-md-8` carrying the statement's `<p>` paragraphs.
  The share `<ul>`, the "For media inquiries" line and the
  "Implementation Note issued …" link are stripped; nothing else is.
* The FOMC minutes page uses a DIFFERENT class list on the same id —
  `<div id="article" class="col-xs-12 col-sm-8 col-md-9">` — which is why the
  container is matched by `id` first. Its sections are paragraphs OPENING with
  `<strong>` (`<p><strong>Committee Policy Actions</strong><br/>…`); the
  Secretary's signature is a `<p style="text-align:center">` and is not one.
* The July 2026 vote line is `by a 9 – 3 vote` with an **en dash surrounded by
  spaces**, and the dissenters are `Beth M. Hammack, Neel Kashkari, and
  Lorie K. Logan` — middle initials included, which is why the name-list regex
  must not terminate on `.`. June 2026 is unanimous: `by a 12 – 0 vote`.
* The target range is written as mixed fractions:
  `at 3-1/2 to 3-3/4 percent` → 3.50/3.75.
* `press_monetary.xml` mixes three documents. `Federal Reserve issues FOMC
  statement` links to the statement page. `Minutes of the Federal Open Market
  Committee, July 28–29, 2026` links to a **press-release** page
  (`monetary20260819a.htm`), NOT to `fomcminutes20260729.htm` — so the feed
  supplies the release instant while the canonical minutes URL is still built
  from the meeting's end date. `Minutes of the Board's discount rate meetings
  on …` is a different committee entirely and classifies as `OTHER`.
* Minutes are released ~3 weeks after the meeting: the June 16-17 minutes carry
  `pubDate` `Wed, 08 Jul 2026 18:00:00 GMT`.

## Phase H U2 — FOMC statement text (diff fixtures), downloaded 2026-08-19

Plain-text, not HTML: unit U2 is the PURE diff library and takes paragraph
lists, so these hold the released statement body already split on blank lines
(U1's `.html` fixtures cover the parsing side). Each was downloaded live with
the contact User-Agent and trimmed to the statement paragraphs — the media
contact line, the Implementation Note link and the stray `&nbsp;` paragraph
are dropped, everything else is verbatim (§44: the source document is
authoritative).

| File | Source URL | Trim |
| --- | --- | --- |
| `fomc_statement_2026-07-29.txt` | https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm | statement `<p>` paragraphs of `div.col-xs-12.col-sm-8.col-md-8`, boilerplate dropped |
| `fomc_statement_2026-06-17.txt` | https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm | same |

Facts the pair pins (verified live 2026-08-19), which are what
`tests/test_events_fed_intel.py` asserts the diff counts from:

* The vote line is written `by a 9 – 3 vote:` with an **EN DASH surrounded by
  spaces** (June: `12 – 0`), so the vote parser must not assume a hyphen.
* The target range is a mixed fraction in eighths — `3-1/2 to 3-3/4 percent`
  — never a decimal, and is UNCHANGED across the two meetings (a HOLD).
* June→July changed exactly two sentences (the vote line, and `reaffirmed its
  policy` → `is continuing its policy` of maintaining ample reserves) and
  ADDED one (the three-name dissent paragraph). Six sentences are untouched.
* Dissenter names carry middle initials (`Beth M. Hammack`, `Lorie K. Logan`),
  so sentence splitting must be abbreviation-aware or the single most-read
  paragraph of the statement shatters into fragments.
