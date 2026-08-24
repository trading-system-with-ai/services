# Development Log — Backend

## 2026-08-20 (47) — CRASH FIX: event_status_badge is a struct, not a string

**Runtime error (user report):** the catalyst Evidence tab crashed
rendering `{status, is_estimated, source, source_name, note}` as a React
child. The backend's event date-status block is a STRUCT; EvidenceTab
and AnalysisTab rendered it directly under a `!= null && !== ""` guard —
which a struct always passes, so CONFIRMED events would also have shown
the amber warning chip once rendering worked.

**Fix:** typed the union in lib/types.ts (struct | legacy string |
null); shared `badgeInfo()` helper (components/catalysts/
eventStatusBadge.ts) — chip shows ONLY for is_estimated (a confirmed
date must not cry wolf), text = "ESTIMATED (source)", note as tooltip;
both tabs render through it. 4 helper tests; verified in a real browser
(Playwright: Evidence tab renders, zero pageerrors, CONFIRMED shows no
chip).

**Tests: 901 UI; backend untouched.**

## 2026-08-20 (46) — UI overlap sweep: FlowNav tuck-under made conditional; visual audit via headless Chromium

**User report (screenshot):** the stale-cards explainer on the
Recommendations tab OVERLAPPED the 8-stage FlowNav strip. Root cause:
`.flow-nav { margin-top: -12px }` assumed it always directly follows
`.subtitle`; the explainer inserted between them got pulled under.
STRUCTURAL fix: the tuck-under now applies only via `.subtitle +
.flow-nav`; the explainer moved below the strip. Platform-wide sweep
found no other subtitle/FlowNav insertions; the two remaining negative
margins are safe line-tighteners.

**Visual audit (run skill, Playwright headless Chromium, 9 pages,
zh locale, zero pageerrors):** caught two more — (1) dashboard EdgeBars
was mounted INSIDE the panel's title row (h2 squeezed beside the
chart); repositioned below the header row and re-verified by
screenshot; (2) MILD_BULL/MILD_BEAR missing from ENUM_ZH (the
opportunity table showed "MILD BULL" beside translated peers) — added
温和多头/温和空头. The ticker page's market-regime VALUE stays verbatim
(§26 gate detail).

**Tests: 897 UI; backend untouched (4248).**

## 2026-08-20 (45) — Recommendations: week-old proposals EXPIRE (user: "一周前生成的消息已经没有意义了")

**Backend:** refresh now FIRST marks PENDING rows older than
EXPIRE_AFTER_DAYS=7 (§6.2 parameter) as status=EXPIRED, audited with the
full row list — which also clears the no-duplicate-PENDING block so the
ticker can be re-proposed. LIVE: one refresh expired 9 stale rows (the
user's screenshot set) and immediately created 4 fresh proposals that
the zombies had been blocking. GET stays read-only — the transition
happens only on refresh.

**UI:** the 待处理 view hides ≥7-day PENDING outright with a counted
note ("已隐藏 N 条…下次刷新将标记为已过期"); an all-expired pending view
says so and points at the refresh button; EXPIRED renders as a dim
已过期 badge under 全部. The 48h stale-dimming for 2–7-day cards is
unchanged (timing-decay warning; the user accepted that layer).
Also this session: gray "—/…" placeholders removed from watchlist
research cells (empty cells instead, user request).

**Tests: 4248 backend / 897 UI; deployed.**

## 2026-08-20 (44) — SINGLE MODE + MORE DATAVIZ + BACKTEST REORDER + FULL zh PASS (UI + advice backend)

**User mandate:** with pages consolidated, drop the beginner/pro split
(one professional mode, "更保险更专业"); add more data visualization;
rethink the backtest presentation order; and finish the Chinese pass —
risk advice untranslated, plus accuracy/referent problems.

**Mode removal:** the entire dual-mode system is gone — lib/mode,
beginner-labels overlay, traffic-light, permission-presets,
BeginnerDashboard/Guide, PresetPanel, the forks in
dashboard/guide/settings/trading-hub/Nav/FlowNav, the Simple/Pro toggle.
One professional face; useEnumLabel reverts to the pure bilingual enum
table. (−46 mode tests, all surfaces re-verified.)

**Backtest presentation:** the page is now 单标的回测 | 组合回测 tabs
(HubTabs), and results lead with CHARTS: equity+drawdown and a new
trade-return histogram before the metrics table; the portfolio tab
gained the equity/drawdown chart (EquityChart extracted to
components/backtests/) ahead of the allocation chart.

**Dataviz (dataviz-skill procedure, validator-run):** the allocation
chart's categorical palette was validator-FAILED (CVD ΔE 1.9 on an
adjacent pair, two low-chroma slots) → replaced with a six-color set
that passes ALL checks (#2ea043 #4493f8 #c8851c #c264d9 #f85149
#0e9aa7), >6 symbols fold into "Other", 1.5px surface strokes between
bands; NEW TradeReturnHistogram (bins zero-snapped so no third hue ever
sits at the diverging midpoint); NEW dashboard EdgeBars (per-symbol
directional edge, diverging from a zero baseline, bars are real SVG
link targets).

**Chinese pass:** advice.py text fields are now BILINGUAL {en, zh}
pairs generated from one template (verified live: 最大回撤 −31.9% with
建议/理由 in Chinese); ENUM_ZH completed to full closed sets
(instruments incl. spreads/income, strength tiers, advice
severities+codes, opportunity statuses); both decision tables + status
badges + EdgeBars route through el(); 权益/净值 referent split unified
to 净值 (matching the chart header); BULL_CALL_SPREAD desc's ambiguous
净值 → 净权利金; journal keeps §26-verbatim reason strings but gains a
zh legend for the three reason families; the portfolio panel gained the
three capital-control inputs (max_gross_pct/cash_floor_pct/
max_positions) so the advice's named parameters are actionable in
place.

**Verification: 3 lenses, 16 findings, all addressed** (incl. WRONG:
EdgeBars tooltip link was unclickable under pointer-events:none — bars
are now the click targets; amber status color at the histogram
midpoint; dead beginner schema in Nav; stale comments).

**Tests: 4247 backend / 897 UI (mode tests removed); deployed.**

## 2026-08-20 (43) — IA CONSOLIDATION: 11 nav entries → 7 hubs; pipeline linkage CTAs (UI-only)

**User mandate:** "页面还是过多,即便是 pro mode 也过于复杂 — 合并一些,
或者让前后步骤联系更紧密一些。"

**Hubs (zero content forking):** Research = 推荐+催化剂+自选列表;
Trading = 交易池+持仓 (beginner mode shows Positions only); Oversight =
风控+活动日志. Each hub (app/{research,trading,oversight}/page.tsx)
renders the EXISTING page components as tabs via
components/shared/HubTabs.tsx — no page was split or rewritten; each
tab keeps its own h1 + FlowNav stage marker. Old routes stay fully
reachable (deep links never break); Nav SECTIONS entries carry `extra`
legacy prefixes so /watchlist/AAPL etc. still highlight their hub. Pro
nav: 总览/研究/回测/交易/风控与审计/使用指南/设置 (7). Beginner nav
drops to FIVE (研究 hub absorbs the three research pages).

**Pipeline linkage:** FlowNav stages now land on hub?tab= targets; a
COMPLETED backtest offers "下一步:在交易池授权 →" in-context (with the
authorization-is-explicit disclaimer); the Trading Pool page links
enabled-symbol orders to the Positions tab; guide STAGES/checklist and
all ~30 legacy list-links retargeted to hub form (detail routes
untouched).

**Adversarial verification (2 lenses, 4 findings, all fixed):** WRONG —
HubTabs read ?tab= on mount only, so a same-hub FlowNav link (e.g.
7 Risk ↔ 8 Audit) changed the URL without switching the tab
(reproduced); HubTabs now subscribes via useSearchParams (hubs wrapped
in Suspense) with history.replaceState for local clicks. Plus: the 30
stale links (batch-retargeted), BeginnerGuide's derived "what Pro adds"
now names the Trading-Pool tab explicitly (tab-level gating is not
SECTIONS-derivable — comment updated), guide STAGES hrefs aligned with
FlowNav.

**Tests: 943 UI (+3 HubTabs); backend untouched (4247).**

## 2026-08-20 (42) — PORTFOLIO EXPLAINABILITY: rebalance journal + risk-model advice (DEPLOYED)

**User mandate:** 自动调仓要有可解释性(在什么节点调仓、为什么这么分配);
回测结合风控算法模型对组合给出建议并说明理由。

**Rebalance journal (portfolio.py RebalanceEvent, migration 029
`journal`):** every capital event — ENTER carries the FULL sizing
arithmetic verbatim ("tier MODERATE budget 0.75% × equity $100,000 =
$750 risk ÷ stop $8.53 = 87 sh; caps: position_pct→87, gross→87,
cash→87 @ $242.04"), EXIT carries the shared exit engine's rule, SKIP
names the capital constraint that crowded a selected candidate out
(max_positions / investable / gross budget / zero-size / missing option
bar) — the record that separates "the matrix said no" from "capital
said no". Live run: 143 events (33 ENTER + 33 EXIT + 77 SKIP).

**Risk advice (backtest/advice.py, migration 029 `advice`):**
deterministic findings from the LIVE model libraries (§21): historical
VaR/ES (method-labelled, §6), §2.8 drawdown, Spearman correlation,
concentration, cash drag. Uniform severity rule documented: WARNING =
realized breach, SUGGESTION = estimated breach, INFO = context; every
fired item echoes its warn level in evidence (self-describing record).
Server strings verbatim English (§26). Live: DRAWDOWN WARNING −31.9%
(peak 2025-04-04 → trough 2026-03-06) with actionable parameters named.

**Adversarial verification (3 opus lenses, 13 findings, all fixed):**
4 WRONG — (1) pending-exit was re-evaluated daily so the recorded EXIT
rule was the LAST bar's regenerated text, and a decided exit could be
silently CANCELLED (latched now, in BOTH auto.py and portfolio.py, with
END_OF_DATA handling under the latch); (2) per-symbol return filtering
desynced spearman inputs → uncaught 500 on a zero close (pairs now
date-aligned); (3) a wiped-out book silently LOST its drawdown item
(explicit WARNING now); (4) UI journal cap applied before the skip
filter (first-15-all-SKIP blanked the table). Plus: VaR now runs on the
RETURN series (scale-free — USD÷final-equity misstated risk on any
growing/shrinking book), stalled option exits journal their stall,
concentration reports signed side (SHORT −45% not "45%"), legacy
pre-029 rows say "predates the feature" instead of a false all-clear,
correlation pairs render in the evidence line, AdviceParams dead
min_return_obs dropped, PortfolioBacktest component tests added.

**Tests: 4247 backend / 940 UI; migration 029 applied live; deployed
and live-verified** (return-native VaR 1.13% ACTIVE on 604 samples,
warn levels echoed).

## 2026-08-20 (41) — AUTO-STRATEGY & PORTFOLIO BACKTEST PROGRAM (Phases A–E, DEPLOYED)

**User /loop mandate:** score-banded auto instrument selection (>a calls /
a–b stock / b–c covered calls / c–d short / <d puts), 逼空 detection, risk
control; AUTO decision-making in backtests; PORTFOLIO backtests over the
whole watchlist with daily per-symbol allocation + cash %; instrument
multi-select. Design: docs/auto-strategy-portfolio-design.md.

**Phase A (4-agent audit):** the banded model ≈ existing
directional_edge (±25/40/60/80 tiers) × §8 matrix, which additionally
IV-conditions option buying (kept — never buy premium blind) and prefers
defined-risk bear expression (kept). LLM scores stay research-only
(§25/§30 verified). Squeeze data: ZERO (short interest/borrow/float
unavailable — §33 vendor needed; honest price/volume proxy only).

**Phase B (backtest/auto.py + router):** run_auto_backtest — at every
FLAT moment the LIVE §8 stack picks the instrument (bias × tier ×
vol-regime-from-REAL-stored-ATM-IV-else-NORMAL × permissions); switching
is EXIT-MEDIATED (shared live exit engines only — no parallel exit math,
no tier-flicker churn); stock long/short + LONG_CALL/LONG_PUT via the
same historical-contract resolvers as the single legs; "AUTO"
whitelisted; BacktestRequest.instruments multi-select (restrict-only,
422-by-name); metrics.auto_decisions audit trail. LIVE: real AAPL, 54
decisions, auto stock↔LONG_PUT switch on the 2025-03-12 bear flip.

**Phase C (backtest/portfolio.py + migration 028 + API):**
run_portfolio_backtest — N symbols, ONE cash ledger; LIVE §12 sizing
(tier budget 0.5/0.75/1/1.25% × current equity / ATR stop, capped by
position_pct + free cash); |edge|-priority contention (exits fill before
entries); cash_floor_pct/max_positions; SIGNED daily allocations + cash%
with the identity cash%+Σalloc%≈100 test-pinned; intersection calendar.
POST/GET /api/backtests/portfolio, PortfolioBacktestRecord (migration
028 applied live). LIVE whole watchlist (AAPL/HPE/RDW/SMCI): 605 bars,
124 decisions, 33 trades incl. 8 REAL historical put fills
(AAPL250516P00215000 …), honest −13.27%. (A "LONG_PUT zero fills" scare
was my own verification script counting only tickers — false alarm,
documented.)

**Phase D (squeeze proxy):** risk/squeeze.py assess_squeeze_proxy
(volume z(20), trailing-252d-high proximity, overnight gap-up; honest
nulls; a blind proxy must not cry wolf) + SQUEEZE_RISK gate between
INSTRUMENT and LIQUIDITY — REPORT mode, SHORT_STOCK candidates only,
PROXY disclaimer in-detail, shadow.squeeze in RISK_DECISION. Four
"chain unchanged" test pins updated (a REPORT gate is consistent with
their intent: no statistical VETO joined the chain).

**Phase E (UI):** backtests page AUTO entry (first, "§8 自动决策") +
4-instrument checkbox multi-select + AUTO 决策轨迹 table
(hover=rationale); 组合回测 panel: run-whole-watchlist, metrics,
AllocationChart (page-idiom SVG: positive bands stack up, shorts below
zero, dashed cash line, hover tooltip, legend), decisions table.

**Final adversarial verification (4 opus lenses, 17 findings, all
addressed):** headline catches — (1) WRONG: chained shorts compounded the
book to 662% gross (short proceeds ARE cash, so a cash floor cannot bound
them); fixed with a gross-exposure budget ``max_gross_pct`` (default 1.0,
API-exposed, live-verified 38.8% under an 80% cap) charged per fill;
(2) WRONG: duplicate/unnormalized portfolio tickers created multiple
slots against one ledger and broke the allocation identity on 261/600
bars; router now strips/validates/dedupes; (3) WRONG: the AUTO checkbox
group could 422 on a never-touched disabled instrument (unchecking one
box made the selection explicit incl. default-locked SHORT_STOCK); the
UI now seeds/gates the checkboxes from account permissions; (4) WRONG:
the UI gate stepper didn't know SQUEEZE_RISK and rendered it dead-last —
GateName/GATE_ORDER synced. Plus: cash_floor/max_positions now reachable
through the API and echoed in stored params; audit COMPLETED carries
portfolio_backtest_id; instrument tokens got zh labels (both decision
tables); stale LONG_CALL copy dropped; tooltip follows the chart idiom;
squeeze detail renders n/a not None; docstrings/design doc honesty
(decisions = selection intent, not fills; prior-bar equity sizing; no
squeeze backtest veto shipped — open decision).

**Tests: 4237 backend / 938 UI; migration 028 live; gateway deployed.**

**Open user decisions:** spreads + CSP into AUTO and the neutral-band
covered-call overlay (needs a §8 neutral-cell emission rule — today the
matrix deliberately emits nothing at NEUTRAL); squeeze-gate veto
promotion after watchlist validation; a real short-interest vendor
(§33); historical ATM-IV backfill from real contract prices so AUTO
sees LOW-IV cells before atm_iv_daily accumulates.

## 2026-08-20 (40) — §4.2 AMENDED (user decision): research surfaces OPEN to any ticker; only backtests member-gated (DEPLOYED)

**User mandate:** "加入自选后才解锁研究是不合理的 — 调查好之后再加自选,
只有回测受自选限制。" Look first, add after.

**Backend:** removed the watchlist-membership 404 from six research
surfaces — analysis (overview/technicals), bars, options chain + eod,
catalyst research surface, and trade-plan GENERATION (research §16 chain,
mode="research"). ensure_daily_bars lazily backfills for ANY ticker
(600d + DATA_BACKFILL audit, first-fetch deliberately unthrottled —
documented). Backtests keep the gate (the one member-only surface).
Docs swept: analysis/options module docstrings, StockBarDaily ORM
docstring (bars no longer imply membership, never pruned), ADR-005
superseding note in ARCHITECTURE.md.

**SECURITY CATCH (adversarial verifier, empirically reproduced, FIXED):**
with generation open, POST /api/plans/{id}/apply could insert a
NON-WATCHLIST symbol into the Trading Pool — bypassing the direct
promote endpoint's membership 422. apply_plan now enforces the same
hard membership precondition before the pool insert; acknowledge_risks
cannot bypass it. Pinned by test (generate off-watchlist 201 → apply
w/ acknowledge 422 → pool empty).

**Calendar time-bomb fixed:** 8 income tests failed on 2026-08-20
regardless of this change — stub DEFAULT_WEEKLY_EXPIRIES=2 left no
expiry inside the 30-45 DTE income band once the monthly third-Friday
drifted to DTE 29. Now 7 weeklies (coverage ~49d > band width) —
calendar-independent forever.

**Frontend:** TickerPreview full-page fork retired → all research tabs
render for any symbol; slim NotOnWatchlistBanner (membership = tracking
+ backtest eligibility; research-only add w/ 409 silent recovery;
pending-rec hint, PENDING query no-poll). BacktestTab hides the run
invitation for non-members (explains membership instead); backtests
page translates the run 404. BeginnerGuide watchlist step restated
(research any ticker; the list = tracked daily + rec flow + backtests).

**Second 4-lens verify pass: 13 findings** (incl. the security catch,
5 stale-doc contradictions, ADR-005, banner polling waste, zh
quantifier) — all fixed except one recorded-not-changed item:
acknowledge_risks is a blanket bool that waves through ALL failing
§4.3 checks at once (incl. structurally-unpassable BACKTEST_COMPLETED)
— pre-existing, arguably intended as user override; flagged for a
future decision (per-check acknowledgement).

**Tests: 4208 backend passed / 1 skipped (was 4207); 938 UI; tsc
clean. DEPLOYED and live-verified:** TSLA (non-member)
analysis/bars/options 200; backtest 404; plan apply w/ acknowledge →
422, pool clean.

## 2026-08-20 (39) — Recommendations UX: look first, add after (non-watchlist ticker preview) + stale-card explainer

**User mandate:** clicking a recommended ticker must not REQUIRE adding it
to the watchlist first — review first, decide after. Also asked why some
recommendation cards are dimmed.

**Stale cards:** answered + surfaced — dimming is `PENDING && age ≥ 48h`
(catalyst timing edge decays; the card admits its age instead of posing
as fresh). The Recommendations page now says so in plain words, and adds
the click-through hint (preview needs no membership). RecCard's ticker
link gained a tooltip saying the same.

**Preview (components/watchlist/TickerPreview.tsx):** a non-watchlist
symbol's research page now renders a preview instead of the 404 error
wall: the symbol's recommendation cards (the REAL RecCard — extracted
verbatim to components/recommendations/RecCard.tsx and reused by both
pages), the §29 governance dialog on promote (still the only
recommendation→watchlist path), a clearly-labeled direct-add path
("Add to Watchlist directly") for symbols with no pending rec, the
UNGATED audit trail (served regardless of membership), and an honest
unlock list naming everything membership actually gates (overview,
price history, technicals, options chain, news, backtests, trade
plans). After adding, invalidations flip the page to full research
automatically. Nothing gated is fetched — pinned by test.

**Adversarial 4-lens verification (ultracode, 4 opus agents): 14
findings, all fixed** — 2 WRONG (the old in-body 404 branch became dead
code; unlock list missed overview while audit isn't gated at all),
409-on-add left a stuck modal (now recovers and flips — the goal state
is reached), missing ["audit"] invalidations vs the app-wide
convention, unused imports left by the extraction, zh copy (missing
object in the blurb, 以下 referent collision), overstated cache-sharing
comment, duplicate button labels (now distinct), discoverability of the
ticker link. New tests: no-gated-endpoint sweep, 409 recovery, audit
panel, exact-label button flows.

**Tests: 940 UI.** tsc clean. Backend untouched.

## 2026-08-20 (38) — SIMPLE MODE: mode-aware Guide (beginner handbook ≠ pro handbook)

**User直指:** Simple 与 Pro 的 Guide 不应是同一本。The pro Guide keeps the
8-stage pipeline handbook untouched; Simple mode now gets its own
handbook (components/guide/BeginnerGuide.tsx, forked in app/guide/page.tsx):
what the platform does → set up once (incl. the previously MISSING
watchlist step) → three daily jobs → **how a trade actually happens**
(plan → approve → Trading-Pool authorization as a separate deliberate
step) → the safety light exactly (incl. "unknown is never green" and the
standing-yellow-until-broker warning) → presets (with short-selling/
margin glossed) → Simple↔Pro vocabulary table → paper money & the pause
button (reason-required stated honestly) → when to switch to Pro.

**Single source of truth extended:** preset rows render from PRESETS;
vocabulary from BEGINNER_ENUM + METRIC_LABELS with the pro column
produced by the SAME enumLabel() pro mode calls (zh readers see 可交易,
not the raw token); safety words from LIGHT_WORD (hoisted into
lib/traffic-light.ts, dashboard re-imports); the "what Pro adds" page
list DERIVES from Nav SECTIONS (`!s.beginner`) — counts can't drift.

**Adversarial 4-lens verification (ultracode workflow, 4 opus agents):**
17 findings, all fixed — headline catches: "five more pages incl.
Catalysts" was WRONG (Catalysts is beginner-visible; Pro adds four);
"the light summarizes what the Pro risk page shows" was MISLEADING (the
alert half comes from the Dashboard feed — copy now splits the sources
honestly); zh corruption (都standing显示), 专业版→专业模式 terminology,
run-on conditional, register fixes; GREEN row now states the every-
input-known precondition; jargon glosses added. Findings pinned as 4 new
tests (SECTIONS-derived list excludes Catalysts; zh pro-term via
enumLabel; watchlist step + standing-yellow present; Trading-Pool
authorization panel). Guard kept: no "§" anywhere in the beginner guide.

**Tests: 934 UI.** tsc clean. Backend untouched.

## 2026-08-20 (37) — RISK UI: /risk crash fixed — dispersion block present with a NULL ratio

**Defect.** Runtime TypeError (`null.toFixed`) blanked the whole /risk
page. After batch 3 joined the stress view into the §40 dispersion set,
an empty book serves `dispersion` as a PRESENT block with
`ratio/min_model/max_model = null` (health UNAVAILABLE, reason "only 0
comparable view(s) < min_views=2") — but `ui/lib/types.ts` typed those
fields non-nullable, so tsc could not flag the four unguarded render
sites in `StatisticalRisk.tsx` (the same type-drift-conceals-crash
pattern QA caught for `distribution` in Phase B).

**Fix.** `ModelDispersion.ratio/min_model/max_model` widened to
`| null` (with the regression note in the type docstring); all four
sites guarded — the tile diagnostics and the panel line now show the
server's reason verbatim instead of a fabricated number. Regression
test renders the tiles + panel with the exact live payload and asserts
the reason text appears and no NaN/null× leaks. A SHADOW display bug
only — no backend change, no decision surface involved.

**Validation.** UI 924 component tests green, tsc clean, /risk 200.
## 2026-08-20 (37) — SIMPLE MODE, PHASES 1–4: beginner/pro presentation layer (UI-only, DEPLOYED via dev server)

**Mandate (user /loop):** the platform is pro-mode; make it usable by a
freshman while staying professional and accurate. Plan:
docs/simple-mode-plan.md. Governing principle: **same engine, same
numbers — beginner mode is a projection.** It may relabel and collapse
detail; it may never change a number, hide an active warning, or unlock
anything pro mode gates. Zero backend changes in this program.

**P1 — mode + vocabulary.** ui/lib/mode.tsx ("beginner"|"pro",
localStorage "mode"; app default beginner, bare-test context default pro
so 886 pre-existing tests keep asserting pro vocabulary); Simple/Pro
toggle in Nav (client-only — pinned by test: no backend call, unlike the
lang switch's LLM-language PUT); ui/lib/beginner-labels.ts (24-token
plain-language overlay + METRIC_LABELS pro/beginner names, bilingual);
useEnumLabel is now mode-aware — every enum surface simplifies at once,
pro path byte-identical. Overlay contract tested: total, injective per
language (never merges two server states), exact fallback for unmapped
tokens; estimator families / stress methods deliberately NOT overlaid.

**P2 — beginner dashboard.** app/page.tsx forks below the SHARED
kill-switch banner+dialogs (a beginner sees exactly the warnings a pro
sees). components/dashboard/BeginnerDashboard.tsx: safety light /
account / top-3 worth-a-look / what-next. ui/lib/traffic-light.ts owns
NO thresholds — pure projection of heat_state (NORMAL/ELEVATED/HIGH/
BLOCKED) + 24h alert severities; UNKNOWN inputs (market/broker/risk/
alerts unavailable) force YELLOW, never GREEN. Ranking extracted to
ui/lib/opportunity.ts so both faces rank identically; queries reuse the
pro queryKeys (one cache, two presentations — divergence impossible).

**P3 — permission presets.** ui/lib/permission-presets.ts +
components/settings/PresetPanel.tsx: Conservative (stock-only) /
Balanced (+bought options) / Advanced (+spreads+income) — pure shorthand
for the eight REAL §5 flags via the same runtime-config PUT (strict
"true"/"false"); NO preset ever enables short_stock or margin (pro-mode
acts); exact-match-or-Custom detection; ConfirmDialog lists the exact
flag diffs before writing. Settings page beginner projection =
ConnectionsPanel + PresetPanel; the 1,500-line ConfigView stays pro.

**P4 — adversarial QA (this entry).** Component-level proofs against the
real components with routed fetch stubs: BLOCKED heat + fresh CRITICAL
alert → RED light with BOTH reasons rendered (hazards cannot disappear);
broker-less → YELLOW + honest em-dashes, "All clear" absent; all-normal
→ GREEN with ENTRY_READY rendered only through the overlay (raw token
asserted absent); mode toggle persists and never calls the backend;
fresh-install default is Simple. Order-path claim pinned by grep: the
beginner surfaces contain no api.orders/plans/income/trading calls —
beginner mode adds links and a permissions PUT, nothing executable.

**P5 — beginner nav collapse.** Simple mode lists six destinations
(Dashboard/Guide/Recommendations/Watchlist/Positions/Settings); the pro
pages stay reachable via in-context links (safety light → /risk, audit
deep links); the 8-stage FlowNav strip hides in Simple mode (pro
pedagogy — the Guide teaches the same flow). Pro mode and bare renders
keep all 11 entries + the strip (pinned by tests both ways).

**Tests: 923 UI (was 886 pre-program; +37).** tsc clean. Backend
untouched (4001 backend tests unaffected).

## 2026-08-19 (36) — RISK ENGINE UPGRADE, COMPLIANCE BATCH 3: Tier C sweep (§5/§8/§10/§13/§23/§26/§40/§45/§46/§48/§50/§52/§54/§55/§69)

**Purpose.** Close the last autonomously-actionable compliance rows —
display, nomenclature and structural gaps over data the platform already
computes. After this batch, everything left in the matrix is user-gated
(Tier B) or data-gated (deep backfill / IV history accumulation).

**Implementation.**
- §5 `ModelTier` (TIER_0..3) on ModelMeta/models/API rows/persisted
  params + `tier_for_model_name()`; validation rows carry the underlying
  model's tier on both fresh and persisted paths.
- §8 first-class `incremental_var_95_usd/pct_nav` + comparison row;
  §46 net vega before/after (CandidateSpec.vega0, spread = NET vega,
  same $/IV-pt convention as §16 — unit-mismatch probe passed);
- §10 ES-99 contributions block with the "noisy tail" health warning
  (k<10 parameterized; sums bit-exactly, probed at n=260 and n=1200);
- §13 GARCH fit persisted (COND_VOL_FIT row: ω/α/β/persistence/half-life/
  Ljung–Box) when the conditional source is GARCH;
- §26 `spot_shock_by_ticker` on POST /api/risk/stress/run (OVERRIDE
  semantics, validated 422, hand-verified on a 2-ticker book);
- §40 worst stress loss joined into the dispersion view set (the spec's
  own §40 example); §45 typed `stress` field on PortfolioRiskSnapshot;
- §55 the staleness rule: a stale PRE_TRADE snapshot suppresses shadow
  caps and reports `UNAVAILABLE_STALE` + reason (SHADOW-only; Tier 0
  byte-identical either way — sabotage-pinned; fail-closed-at-promotion
  remains the user's Q3/Q7 call); §54 deviation recorded in design §6;
  §69 DEVLOG entries (17)–(20) backfilled with the missing fields.
- UI: five new dashboard cards (VaR99/ES99/Stress Loss/Net Delta/Net
  Vega), option-row greeks + underlying exposure + vega $, ⓘ modals on
  StressScenarios/ModelValidation, tier chips, per-ticker shock form,
  "local sensitivity, not tail risk" line on Greeks; glossary entries.
- QA: registry test-isolation leak fixed (autouse clear_for_tests
  emptied the process-global registry for later modules); compliance
  matrix §2 rows synced with §3 (doc drift); stale §21 UI note removed.

**Validation.** Backend **4207 passed, 1 skipped**; UI 887 component
tests; adversarial suite 33/33 (AST pin + sabotage battery); verifier
reverted the §55 rule and confirmed the pin fails non-vacuously; all
probes exact (`==`, not approx) where the contract demands. Deployed.

**Production status.** SHADOW/RESEARCH throughout. Compliance tally after
batch 3: see `docs/risk-engine-spec-compliance.md` §1 (matrix + tally
updated in the same change). Remaining rows are ALL user-gated or
data-gated: §27 stress veto promotion (+fail-closed rule), §58 book-level
pause policy, §64 assess() backtest replay, audit Q1–Q7, deep backfill
(unlocks EVT/empirical horizons), atm_iv_daily ≥120 obs (unlocks
empirical IV shocks + IV-path VaR), GMV/ERC weights harness.

## 2026-08-19 (35) — RISK ENGINE UPGRADE, COMPLIANCE BATCH 2: full option revaluation in the VaR/ES P&L series (spec §21–§22; design §10; SHADOW)

**Purpose.** Close the compliance matrix's biggest measurement gap:
portfolio VaR/ES, contributions, marginal/incremental ES and every shadow
cap priced options DELTA-LINEARLY, hiding long-gamma/vega convexity from
exactly the numbers §21 exists to protect.

**Implementation (design contract §10).**
- `pnl_series.py`: `PositionRiskInput` gains five OPTIONAL leg fields
  (strike/right/t_years/iv0/mark0); when present and valid the per-day
  P&L is `qty·mult·[BS(spot·(1+r_t)) − BS(spot)]` (CONST_IV, T held at
  T0, S-convexity only; baseline computed once so r=0 ⇒ exactly 0.0 and
  the basis cancels bit-exactly; r ≤ −1 floors at the discounted terminal
  value). No fields ⇒ the old DELTA_LINEAR expression, byte-identical.
  `BookPnl.method_by_key` + `book_method_summary()`.
- Builder: stress legs are resolved ONCE (new stage d0) and shared by the
  P&L series, greeks and the stress catalogue — no second chain call;
  `statistical.pnl_method` = FULL_REVAL_CONST_IV when ≥1 leg revalues;
  `data_quality.pnl_method_by_key` + counts; persisted on the snapshot
  row. Pre-trade candidates carry the selected contract's leg fields
  (spreads: ONE input with the long leg + documented note — the two-leg
  form breaks every RC key consumer; measured convexity is an upper
  bound, conservative for a cap). All _safe-wrapped: a raise degrades to
  DELTA_LINEAR, never a 500.
- QA fixes: `proposed_book` label/method_by_key propagation (the
  after-book had reported the before-book's stale label), put terminal
  floor now K·e^{−rT} matching bs_price's own limit; UI `PnlMethod` union
  widened.

**Validation.** Suite **4160 passed, 1 skipped**. Verifier probes:
stock-only book deep-equal (bit-exact, `==` not approx); convexity
direction at the worst tail day (−4958 vs −5590 delta-linear on a long
call); Euler RC sums bit-exactly on the mixed book; SHADOW proof (raise ⇒
DELTA_LINEAR degradation, preview byte-identical); adversarial suite
33/33; walk-forward validation consumes the new series with the sentinel
still green. Latency 1.0 ms vs 0.3 ms per build (absolute cost trivial).
Deployed.

**Model limitations.** CONST_IV: vega/theta remain outside VaR/ES until
`atm_iv_daily` accumulates an IV path (~120 trading days from
2026-08-18) — then the same series construction carries IV moves (S-size
residual, recorded in the compliance doc §3 Tier B).

**Production status.** SHADOW. Compliance tally now 16 PROD / 31 SHADOW /
1 RESEARCH / 18 PARTIAL / 8 deferred-or-rejected-documented.

## 2026-08-19 (34) — RISK ENGINE UPGRADE, COMPLIANCE BATCH 1: spec-gap清欠 (§34/§36/§37/§45/§59/§65/§6/§11/§12/§18/§30) — all SHADOW

**Purpose.** A 74-section compliance check of `prompts/risk_engine.md`
against the shipped platform (`docs/risk-engine-spec-compliance.md`,
7 evidence-based checkers) found ~11 items the Phase A audit had
committed to but that never landed or were never recorded. This batch
clears the small/medium ones. Everything remains SHADOW — no Tier 0
decision changes (pinned by the adversarial AST test and a new
sabotage-everything integration test).

**Implementation.**
- §37/§59/§36 — `pretrade.sizing_v2_shadow()` (`SizingV2Params`, research
  defaults, UNVALIDATED): ES modifier (clamp(es_target/es95, floor, 1)),
  correlation modifier (NORMAL/ELEVATED/CONVERGING 1.0/0.85/0.7), model-
  health modifier (LOW/ELEVATED/HIGH 1.0/0.85/0.7 — §59's budget effect,
  SHADOW), candidate budget = tier × vol × the three modifiers, plus the
  §36 risk-linked cash floor (regime floor + k_es·excess ES + k_dd·|dd| +
  model-risk addon, capped 0.90, `binds` flag). Logged under
  `shadow.statistical.sizing_v2` in every RISK_DECISION and mirrored into
  the preview `risk` block; a raise changes nothing (proven).
- §65 — all NINE spec metrics now instrumented (4 pre-existed):
  `var_exceedances_total{model,confidence}`, `es_exceedances_total{…}`,
  `garch_fit_failures_total{site,health}`, `stress_limit_blocks` (the
  would-have-bound evidence series for §27 promotion),
  `risk_resize_count`, `risk_reject_count`, `model_health_state{model}`
  (ordinal gauge). Confidence label added deliberately — 95/99 are
  different tests with different expected rates.
- §6/§12 — HISTORICAL VaR/ES 95% rows at 5D/10D (√h-scaled, labelled
  RESEARCH; `value_5d == value_1d·√5` pinned exactly; 1-day grid is an
  unchanged prefix) and `conditional_horizon_sigmas` from the GARCH term
  structure when the fit is ACTIVE (labelled GARCH_TERM_STRUCTURE).
- §34 — `diversification_ratio()` (Σᵢσᵢ/σ_p, ddof=1) computed each build,
  served + persisted as a risk_metrics row; hand-checked (anti-correlated
  book DR=3; perfect hedge → honest UNAVAILABLE).
- §45 — BUG fixed: `PortfolioRiskSnapshot.correlation_state` was declared
  but never populated by the builder; now typed `CorrelationState | None`
  and passed through (regression-pinned).
- §18 — Spearman built: `spearman()` + rolling average on the regime
  short window, additive `current_avg_spearman` on CorrelationState and
  the wire; three contradictory doc references reconciled.
- §11 — `risk/models/factor.py` single-factor (SPY) RESEARCH diagnostic:
  per-position beta, portfolio beta-explained variance share, date-ALIGNED
  regression (QA caught positional zipping), served as
  `statistical.factor`.
- §30 — robust/shrinkage covariance recorded in the final report's
  Deferred Models table (little value at n≤8 names; revisit >15 names).
- UI: sizing_v2 block in TradeComparison (SHADOW badge), DR + factor
  share lines, Spearman beside Pearson in the correlation pill; glossary
  entries; tsc clean; §47 dialog-gate fixed to ignore string literals in
  test fixtures.

**Validation.** Full suite **4118 passed, 1 skipped** (baseline 4001 —
the Catalyst program landed in parallel, DEVLOG 21–33). Integrator
sabotage test: every new seam raising simultaneously leaves preview
decisions byte-identical; `tests/test_risk_adversarial.py` 33/33; all
nine §65 names asserted at the scrape; DR/sizing arithmetic recomputed by
hand; compliance matrix rows for §6/§11/§12/§18/§30/§34/§36/§37/§45/§59/
§65 flipped with fresh evidence. Deployed; live smoke: 5D/10D rows and DR
block serving with honest nulls on the empty book.

**Model limitations.** All new modifiers/floors are research defaults,
UNVALIDATED — they exist so the 20-day shadow window accumulates the
would-have-bound evidence (§65 counters) promotion needs.

**Production status.** SHADOW/RESEARCH. Remaining from the compliance
check: §21 full revaluation into the VaR/ES P&L series (batch 2, next),
§27 stress veto promotion + audit §11 Q1–Q7 (user decisions).

## 2026-08-19 (33) — CATALYST & EVENT INTELLIGENCE, PHASE L: §96 adversarial look-ahead suite, §86 measurement harness, §100 final report — PROGRAM COMPLETE (DEPLOYED)

**Purpose.** Spec §85/§86/§96/§100: prove — adversarially, with tests that
demonstrably bite — that `analysis(as_of=T)` cannot see anything published
after T on ANY surface; build the harness that will one day say whether the
event features predict anything (measure, never assume); and deliver the
§100 A–I final report.

**Existing capability reused.** Every as-of gate built in phases B–K
(`as_of_bar_filter`, `acceptance_datetime` gate, news/macro/fed release
gates, the §69 memory gate, `digest_view`); the SQLite endpoint harness;
the house adversarial style of `test_risk_adversarial.py`.

**Architecture decision.** (1) `tests/test_event_lookahead.py` (46 tests) +
`tests/_lookahead_util.py`: for each of the 12 endpoint families
(price-context, fundamentals, replay, history, news, evidence,
analysis/analyses, options, macro, fed, timeline, risk) a future sentinel
is planted beside a past twin and a recursive scanner proves the sentinel
appears NOWHERE in the payload while the twin does; 6 mutation-bite tests
monkeypatch a gate to a pass-through and assert the suite would fail;
defence-in-depth is mapped and pinned (news/fed: SQL bound alone removable,
pure gate alone suffices — a 3-step proof so a "cleanup" refactor can't
silently halve the protection); §85: a LIVE_CHAIN_SNAPSHOT row planted on a
past event is refused. (2) The suite FOUND a real §96 leak: the
fundamentals `freshness` block was built from `rows[0]` — the newest STORED
filing, ungated — leaking `source_filing_url`/`acceptance_datetime`/
`latest_filing_date`/`period_end` for a filing accepted after `as_of`, and
propagating into the LLM evidence bundle. Fixed in the same session
(freshness now derives from the as_of-gated rows; `statements_stored`
deliberately stays the store-wide count) and the discovering xfails now
pass outright. (3) §86: pure `libs/trading_core/events/event_study.py`
(Spearman as Pearson-of-average-ranks — the 1−6Σd² shortcut is wrong under
ties and ties are the normal case; `MIN_MEANINGFUL_N = 12`; fixed honesty
caveats) + DB-only `apps/gateway/event_study.py` + `GET /api/events/study`.
Features come from the EARLIEST stored PRE_EVENT bundle per event (a
re-analysis after the print would carry a run-up measured through the
reaction it predicts — mutation-pinned), LIVE-basis metrics excluded,
outcomes deliberately hindsight (that side is the thing being predicted —
documented). Two §86 candidates are named unmeasurable rather than omitted
(estimate_revision: no consensus vendor; valuation_expansion: would measure
the backfill). (4) `docs/catalyst-event-final-report.md` (506 lines):
§100 A–I + §101/§102, every number quoted from DEVLOG 21–33.

**Live.** `GET /api/events/study?event_type=EARNINGS`: 0 paired
observations today (one stored bundle, its event still in the future) →
every feature honestly `NOT_MEANINGFUL` with reason "needs >= 3 paired
observations, have 0" — the §86 answer until event memory accumulates.
Fundamentals endpoint re-verified post-fix.

**Tests.** Look-ahead 46, study 40 + endpoint tests. FINAL SUITE:
**4001 passed, 1 skipped, 0 xfailed** backend; UI 797 vitest; tsc clean.

**Known limitations.** The §86 report is structurally ready but empty until
several events complete with stored pre-event bundles; the look-ahead suite
covers endpoint payloads, not UI rendering; the one skipped test is the
pre-existing environment-dependent skip.

**Production status.** DEPLOYED. **CATALYST & EVENT INTELLIGENCE PROGRAM
(phases A–L, spec §1–§102) COMPLETE**: registry+calendars (B), price
context (E1), fundamentals (E2), replay (C), news evidence (D), LLM
earnings intelligence + event memory (F), options/implied move (I), UI
slices (J), macro (G), Fed (H), risk SHADOW (K), validation + report (L).
Final deliverable: `docs/catalyst-event-final-report.md`.

**Next.** Accumulate event memory; revisit §86 after the next earnings
cycle; user decisions: BEA_API_KEY (free) for GDP/PCE actuals, Benzinga
add-on for consensus, event-risk promotion after shadow validation.

## 2026-08-19 (32) — CATALYST & EVENT INTELLIGENCE, PHASE K: event risk — SHADOW integration with the risk engine, pre-trade surfacing, Risk tab (DEPLOYED)

**Purpose.** Spec §62–§67: give discrete jump risk (an earnings print, an
FOMC decision) a first-class snapshot beside VaR/ES — historical event
moves, implied move, exposure, option sensitivity — classified
LOW/MODERATE/HIGH/EXTREME by a deterministic table (§63: never the LLM),
surfaced in the RISK_DECISION audit, the Trade Plan and the event page, and
kept strictly SHADOW (§65: no enforcement until validated).

**Existing capability reused.** Phase I `event_option_metrics`
(implied/actual moves), the registry's previous-event chain, the Phase C/E1
reaction history, `pretrade.QuantityCap` + the Phase C/D hypothetical
`shadow_verdict` merge point (STRESS caps' pattern), the RISK_DECISION
audit shadow dict, `_plan_payload`.

**Architecture decision.** (1) Pure `libs/trading_core/risk/event_risk.py`
(`event-risk-1.0.0`): `historical_event_risk` (median/p75/p90/max of
|moves| with n ALWAYS present — §64), `classify_event_risk` (documented
threshold table: expected move = implied else historical median, basis
recorded; imminence ≤3 d; exposure share bumps; UNKNOWN — not LOW — when
nothing is known, with reason; sensitivity is a separate options axis,
§66), `event_risk_caps` → real QuantityCap rows (HIGH → 10 % NAV, EXTREME →
5 %, research defaults) that join ONLY the hypothetical shadow verdict.
Statically enforced: no LLM/network/engine import in the module (tokenized
source scan test). (2) Gateway `apps/gateway/event_risk.py` + five minimal
edits in orders.py: `shadow["event"]` computed in its own try/except (a
raising seam leaves the order path byte-identical — pinned), its caps
merged as `extra_caps=[*stress_caps_shadow, *event_caps_shadow]` into the
Phase C SHADOW verdict; **the real `assess(...)` still takes no
`extra_caps`** (grep-pinned; user mandate). `_plan_payload` gains
`event_risk` fresh-on-read (None when no event within 14 d). New `GET
/api/events/{id}/risk` with the §66 options block (event IV, expected IV
crush honestly NO_DATA — no forward surface subscribed, historical crush
when stored, and the long-call explainer). Market-wide FOMC flag within
3 d. (3) UI Risk tab (state chip + SHADOW badge "shadow only — never
blocks trades", drivers, caveats with sample size, historical table,
implied-vs-historical, options panel) + Trade Plan EVENT RISK panel (§65
layout).

**Live.** `GET /api/events/99/risk` (AAPL 2026-10-29, T−71 d): state LOW,
sensitivity LOW, enforcement SHADOW; historical n=8 — median |1.33 %|,
p90 7.35 %; implied None (no stored implied for the upcoming event yet);
drivers and caveats verbatim honest ("position exposure unknown — not
assumed small", "event date is ESTIMATED", "based on 8 event(s)"). Trade
Plans for RDW correctly carry `event_risk: null` (next earnings ~77 d out,
beyond the 14-day horizon).

**Tests.** `test_risk_event.py` (66), `test_event_risk_api.py` (24 + 2
isolation), orders-shadow regression, full orders/risk set 994 passed
(approvals byte-unchanged); UI RiskTab + plan panel tests. Suite: 3915
passed / 1 skipped backend; UI 797 vitest; tsc clean. The verifier caught a
HIGH-severity silent unit bug none of the units saw: `event_option_metrics`
stores moves as FRACTIONS while the classifier speaks PERCENT — every real
event's risk would have been understated 100× — fixed at one documented
boundary with a regression test pinning the conversion.

**Known limitations.** Research thresholds are unvalidated (promotion to
WARN/RESIZE/REJECT only after §86 measurement and an explicit user
runtime-config decision); exposure is cost basis; option greeks wired only
where an existing view exposes them; historical tail stats on n≤8 —
labelled, never certain.

**Production status.** DEPLOYED (no migration; gateway rebuilt).

**Next.** Phase L: §96 adversarial look-ahead suite, §86 measurement
harness, §100 final report — the program's last phase.

## 2026-08-19 (31) — CATALYST & EVENT INTELLIGENCE, PHASE H: Fed intelligence — statement diff, §43 policy dimensions, two-window FOMC reaction, Fed tab (DEPLOYED)

**Purpose.** Spec §9/§42–§45: reconstruct the previous FOMC decision from
primary documents (statement, minutes, subsequent speeches), diff the
statement against the one before it (ADDED/REMOVED/CHANGED/UNCHANGED, the
source document authoritative — §44), report §43 policy dimensions
separately (never one hawkish/dovish score), and quantify the previous
decision's market reaction in TWO separated windows: statement 14:00–14:30
ET vs press conference 14:30–15:30 ET (§45).

**Existing capability reused.** `libs/event_calendar/fed.py` events (the
registry already carries FOMC_MEETING/DECISION/PRESS_CONFERENCE/MINUTES/
FED_SPEECH with dates), press_monetary RSS, minute-bar backfill seam
(`event_replay.ensure_event_window_bars` pattern), Phase G's
`multi_asset_reaction`/Treasury yields for the daily fallback and macro
prints for the data section, evidence bundle + LLM analysis seams.

**Architecture decision.** (1) `libs/event_calendar/fed_docs.py`
(`FedDocumentsProvider`): statement pages
`newsevents/pressreleases/monetary{YYYYMMDD}a.htm`, minutes
`monetarypolicy/fomcminutes{YYYYMMDD}.htm`, speeches; article extraction
survives the Fed's real markup (unclosed `<p class="releaseTime">`, nav
columns that are also col-sm-8 — both found live and regression-pinned);
vote parser (9–3 with dissenters Hammack/Kashkari/Logan live), target-range
parser ("3-1/2 to 3-3/4 percent" → 3.50–3.75), released_at from the RSS
(statements 18:00 UTC; minutes only via RSS); as-of gating happens BEFORE
any HTTP fetch. (2) Pure `libs/trading_core/events/fed_intel.py`
(`fed-intel-v1`): sentence-level diff (difflib, CHANGED at ratio ∈
[0.6, 1)), 8 dimensions tagged by keyword sets, `policy_rate_change` (bp +
CUT/HIKE/HOLD), `vote_dispersion`, `fomc_reaction_windows` (1-minute bars,
windows adjacent and separated) with a labelled daily fallback,
`build_fed_packet`; no aggregate score anywhere (mechanically asserted: the
only matching export is `NO_SINGLE_SCORE_NOTE`). (3) Gateway
`event_fed.py` + migration 027 (`fed_documents`, url-unique upserts), `GET
/api/events/{id}/fed` (DB-only, exploding-provider-pinned) + POST backfill
(documents + SPY/QQQ/TLT/GLD/UUP minute windows around the previous
decision); FOMC evidence bundles gain `macro_context.fed`; the LLM prompt
for FOMC events adds the §44 instruction (explain significance per
dimension, never collapse to one label). Market pricing: honestly
UNAVAILABLE (no fed-funds futures source; 2Y change is a labelled proxy).
(4) UI Fed tab: diff view with counts, dimensions table + "no single
hawkish/dovish score by design" note, two reaction windows side by side
(basis badge 1m/daily), speeches, backfill.

**Live.** FOMC_DECISION:2026-09-16 (event 84) backfill: 4 documents + 1 476
minute bars in 0.9 s. Packet: previous statement 2026-07-29 (vote 9–3,
dissenters Hammack/Kashkari/Logan; target 3.50–3.75 % held); diff vs
2026-06-17: 1 ADDED, 2 CHANGED, 6 UNCHANGED (POLICY_RATE ADDED,
BALANCE_SHEET CHANGED, COMMITTEE_DISPERSION CHANGED, INFLATION/EMPLOYMENT/
GROWTH/RISK_BALANCE UNCHANGED); minutes released 2026-08-19 (today), 8 key
paragraphs; 1 subsequent speech; reaction basis 1m_bars — statement window
SPY −0.152 %, QQQ −0.171 %, TLT −0.072 %, GLD −0.037 %; press-conference
window SPY −0.311 %, TLT −1.040 %, GLD +0.270 %, UUP −0.281 % — the two
windows visibly disagree, which is exactly why §45 wants them separate.

**Tests.** `test_fed_docs.py` (58), `test_events_fed_intel.py` (80),
`test_events_fed_api.py`, parity pins for 027; UI FedTab tests (+15 from
the verifier, incl. two real wire-shape fixes: backfill counts nested under
`counts`, diff-count keys uppercase). Suite: 3823 passed / 1 skipped
backend; UI 718 vitest; tsc clean.

**Known limitations.** Press-conference transcripts are PDFs — linked, not
parsed; minutes released_at only as far back as the RSS reaches; SEP/dot
plot not ingested; fed-funds-futures pricing unavailable at this
subscription tier; speech coverage is the RSS window.

**Production status.** DEPLOYED (migration 027 live, same gateway build as
Phase G).

**Next.** Phase K event-risk SHADOW, then L validation + §100 final report.

## 2026-08-19 (30) — CATALYST & EVENT INTELLIGENCE, PHASE G: macro intelligence — BLS/BEA/Treasury primary-source adapters, macro packets, multi-asset reactions (DEPLOYED)

**Purpose.** Spec §8/§38–§41 with §14/§33: put CPI/PPI/Employment/JOLTS/GDP/
PCE on the same event registry as earnings and FOMC — primary government
sources, not LLM browsing — and give each macro event a packet (previous
actual, consensus honestly UNAVAILABLE, recent trend, previous multi-asset
market reaction) plus the §40 related-evidence window.

**Existing capability reused.** `libs/event_calendar` registry/Protocol
(bls/bea join sec_edgar/fed as KEYLESS_PROVIDERS), the ingest tick,
`reaction.event_reaction` for per-asset reactions, daily-bar storage,
`ensure_daily_bars`-style backfill seams, the evidence bundle
(`macro_context` placeholder from Phase F now filled), migration parity
harness.

**Architecture decision.** (1) Adapters: `bls.py` parses the schedule pages
(cpi/ppi/empsit/jolts; release date + time column — JOLTS releases at 10:00
ET → DURING_MARKET, the rest 08:30 → BEFORE_MARKET; a row whose time won't
parse is dropped, never defaulted), `bea.py` parses bea.gov/news/schedule
(real titles are "GDP (Advance Estimate)…" and "Personal Income and
Outlays…"; year comes from the section header with Dec→Jan rollover),
`treasury.py` parses the daily yield-curve CSV (tenor keys verbatim: "2 Yr",
"10 Yr"; missing tenor absent, never 0.0), `macro_data.py` BLS API v1
(keyless, 3-year window, ~25 req/day — backfill batches conservatively; BEA
actuals need a free key: `BEA_API_KEY` in runtime_config, absent →
CapabilityNotAvailable). Every government request carries the contact
User-Agent. All parsing fixture-pinned to live 2026-08-19 markup. (2) Pure
`libs/trading_core/events/macro.py`: series catalogue per event type
(headline/core/rate/level/wages roles; MoM/YoY/change_k transforms; SA vs
NSA documented), `build_macro_packet` (point-in-time: a print is visible
only if its release_at ≤ as_of; release_at from the stored schedule else
period_end+45 d flagged ESTIMATED), `multi_asset_reaction` over
SPY/QQQ/TLT/IEF/SHY/GLD/USO/UUP (roles; proxies labelled; absent assets
listed under `unavailable`) + Treasury 2Y/10Y changes in bp,
`macro_context_for` (earnings bundles list upcoming macro within 14 d),
`related_evidence_window` (§40 — deterministic list, LLM picks themes).
(3) Gateway `event_macro.py` + migration 026 (`macro_observations`,
`treasury_yields`), `GET /api/events/{id}/macro` (DB-only) and POST
backfill; `MACRO_REFERENCE_SYMBOLS` beside ADR-005's INDEX_SYMBOLS
(ADR-009); evidence bundles tolerate ticker-None events. (4) UI Macro tab +
MacroContextCard.

**Live.** `POST /api/events/refresh`: bls created 52 events, bea 8 (CPI/PPI/
Employment/JOLTS through 2026-12, GDP/PCE). CPI:2026-09-11 (event 227)
backfill: 90 observations, 408 yield-curve rows, 623 bars in 21 s. Packet:
July CPI +0.074 % MoM headline / +0.215 % core, 3.36 % YoY NSA; trend
headline falling, core flat (6 prints); previous-release reaction (8/12):
SPY +0.25 %, QQQ +0.73 %, GLD +0.99 %, 2Y −2.0 bp, 10Y −2.0 bp; consensus
CONSENSUS DATA UNAVAILABLE.

**Tests.** `test_event_calendar_macro.py` (52), `test_events_macro.py`
(60), `test_events_macro_api.py` (24), ingest-tick tests (real adapters over
live fixtures, mutation-verified), parity pins for 026; UI MacroTab tests.
The verifier caught 10 UI↔wire key mismatches (U4 had typed the payload
from prose; `[k: string]: unknown` had silenced tsc) — fixed UI-side and
pinned to the wire spellings. Suite: 3823 passed / 1 skipped backend; UI
718 vitest; tsc clean.

**Known limitations.** No consensus/forecast source → surprise (§38) is
UNAVAILABLE by design; BLS v1 unregistered rate limits bound backfills;
BEA actuals wait on a (free) BEA_API_KEY; RETAIL_SALES/ISM have no adapter
yet (Census/ISM); intraday macro reactions use daily bars (no minute window
for BEFORE_MARKET prints yet).

**Production status.** DEPLOYED (migration 026 live; bls/bea providers
keyless-active; 60 macro events in the registry).

**Next.** Phase H Fed intelligence (same deploy), then K risk SHADOW, L
validation + §100 report.

## 2026-08-19 (29) — CATALYST & EVENT INTELLIGENCE, PHASE J: catalyst UI slices — hero, timeline, evidence/scenarios tabs, card summaries, §60 implied-vs-actual wiring (DEPLOYED)

**Purpose.** Spec §53–§61: the event detail page gets the §56 hero and the
§57 timeline ("LAST EARNINGS → developments → TODAY → NEXT EARNINGS"), the
Evidence and Scenarios tabs stop being placeholders, the §54 calendar cards
carry historical move / implied move / analysis status, and the §60 history
table gains implied-vs-actual columns plus the chart — with the backend
returning structured data and the UI only rendering (§61).

**Existing capability reused.** Phase D news store + `analyze_window`
(timeline NEWS items are the MATERIAL clusters), Phase E2 statements
(FILING items), the registry (EVENT items), Phase F `event_analyses`
(ANALYSIS items, analysis status), Phase I `event_option_metrics` (implied /
historical moves), `EvidenceSections`/`ScenarioCards` from F,
`EventHistoryTable`/`ImpliedVsActualChart`, `EventCard` exposure line, the
`_event_or_404`/`_resolve_as_of` seams.

**Architecture decision.** (1) `apps/gateway/event_timeline.py`:
`build_event_timeline` is DB-only (pinned by an exploding-provider test) and
as-of-safe; window = previous comparable event → as_of (else 120 d); items
NEWS/FILING/EVENT/ANALYSIS sorted by time, capped at 200 with `truncated`,
counts by kind and by materiality category; anchors carry the ESTIMATED
flag. (2) `attach_card_summaries` is OPT-IN (`GET /api/events?summaries=
true`) and the default payload is byte-identical to before (pinned);
`analysis_status` READY (<7 d) / STALE / NONE, implied move from the latest
stored metric (basis + the server's own not-a-forecast note), historical
median |move| over the previous-event chain (≤8) — all DB lookups, no
provider. (3) UI: `EventHero` (T-n / T+n, schedule + session + zone,
CONFIRMED/ESTIMATED badge with source, freshness, COST-BASIS exposure
exactly as the card, Phase K risk chip as an honest placeholder, implied-move
chip suppressed on NO_DATA), `TimelineTab` (rail with anchors, kind and
category filters, collapsed groups), `EvidenceTab` (the F bundle with
DATA/QUANT tiers, coverage, consensus-unavailable notice), `ScenariosTab`
(scenario cards + surprise threshold + invalidation, CTA when no analysis),
card summary lines, `EventHistoryTable.optionsHistory` joined on
`event_key`/`event_id` (never on date — two events can share a day) with
`ImpliedVsActualChart` mounted beneath.

**Live.** `GET /api/events/99/timeline`: 99 items in the 20-day window from
`EARNINGS:AAPL:2026-07-30` (97 material NEWS across 14 categories — PRODUCT
26, EARNINGS 16, ANALYST_REVISION 11, SUPPLY_CHAIN 10 …, 1 FILING, 1
ANALYSIS), 0.28 s. `/catalysts/99` renders hero + tabs on the dev server
(the dev server needed a clean `.next` restart after the day's concurrent
file churn — "Cannot find module './vendor-chunks/next.js'" is a stale dev
cache, not an app error).

**Tests.** `test_events_timeline_api.py` (39), `test_events_api.py` (+3
summaries), UI `EventHero` (24), `TimelineTab` (38), `EvidenceTab` (17),
`ScenariosTab` (24), `EventCard` (+18), `EventHistoryTable` (+16). Suite:
3512 passed / 1 skipped backend; UI 581 vitest; tsc source-clean. The
verifier fixed two UI type-fidelity gaps (undeclared payload keys), no
backend changes.

**Known limitations.** Timeline is news/filings/events/analyses only (no
price markers yet); the Risk tab stays a placeholder until Phase K; card
summaries depend on prior backfills (NONE/null until POSTed); no "Surprise vs
next-day return" chart (§60 optional) because surprises are unavailable
without consensus.

**Production status.** DEPLOYED (gateway rebuilt; UI dev server restarted
with a clean cache).

**Next.** Phase G macro (in flight), H Fed, K risk SHADOW, L validation +
§100 report.

## 2026-08-19 (28) — CATALYST & EVENT INTELLIGENCE, PHASE I: options / implied move — historical ATM straddle approximation, live chain, §60 history stats, Options tab (DEPLOYED)

**Purpose.** Spec §18/§36/§37/§60/§66-prep: show what the option market PRICES
for an event (ATM straddle implied move), compare it with what the stock
actually did at prior events (implied vs realised, IV before/after), and keep
the §37 wording everywhere: "option-market pricing, not a forecast".

**Existing capability reused.** `get_option_chain` (live snapshots with
IV/greeks on Alpaca; Massive chain), `libs/trading_core/options/{bs,iv}.py`
(bisection IV), Phase E1 reaction helpers and `pre_event_close_for`, the
event registry's `previous_event_id` chain, the POST-only backfill pattern,
`fundamentals_provider_name` (Massive for history, market-data provider for
live).

**Architecture decision.** (1) NEW provider capability
`HistoricalOptionProvider` (separate Protocol): `list_option_contracts(
underlying, expiration_date, as_of, right)` → `OptionContractRef` and
`get_option_history_bars(option_ticker, start, end)` → daily `Bar`s —
probed live on the Massive base plan (`/v3/reference/options/contracts?
as_of=` and `/v2/aggs/ticker/O:…/range/1/day`); Alpaca raises
`CapabilityNotAvailable`; the name deliberately differs from the pre-
existing `get_option_daily_bars` (bare-OCC open/close dict used by the
options backtest resolver). (2) Pure `libs/trading_core/events/
implied_move.py` (`implied_move-1.0.0`): `select_event_expiry` (first expiry
on/after the event; AFTER_MARKET → strictly after), `nearest_strike`,
`straddle_implied_move` ((call+put)/spot), `implied_move_from_iv`
(iv·√(dte/365) cross-check), `event_iv` (BS bisection, r=0.04 documented),
`iv_crush`, `implied_vs_realized` (ratio = |actual|/implied; <0.8
OVER_PRICED, >1.2 UNDER_PRICED, else FAIR, with 1e-9 band epsilon),
`historical_move_stats` (median/p90 nearest-rank/max of |moves|, n),
`build_summary` with honest NO_DATA/PARTIAL statuses and notes; every
historical number is labelled `HISTORICAL_DAILY_CLOSE_APPROXIMATION`, live
ones `LIVE_CHAIN_SNAPSHOT`. (3) Seam `apps/gateway/event_options.py`:
`backfill_event_options` (POST-only network: pre-event close → contracts
as-of the pre-event date → event expiry → ATM call+put → daily bars around
the event → straddle → `event_option_metrics`), `backfill_options_history`
(walks the previous-event chain), `live_implied_move` (upcoming events:
current chain, ATM mids, provider IV), `build_event_options_payload`
(current + history + stats + comparison + disclaimer). Migration 025
(`option_daily_bars`, `event_option_metrics`) applied live. Endpoints `GET
/api/events/{id}/options`, `POST …/options/backfill`, `POST
…/options/history/backfill?last=N`. (4) UI Options tab + inline-SVG implied-
vs-actual chart with the §37 disclaimer on every implied-move surface.

**Live.** AAPL `EARNINGS:AAPL:2026-07-30` (AMC): expiry 2026-07-31, ATM
332.5 (spot 333.43), straddle implied ±3.82 %, IV 0.94 (1-DTE), actual
−7.35 % → UNDER_PRICED (ratio 1.92); 2026-04-30 implied 3.84 % vs actual
3.24 % FAIR; 2026-01-29 implied 4.53 % vs 0.46 % OVER_PRICED. The history
backfill stored only 3 of 9 events on the first run because Massive
rate-limited (HTTP 429) the back-to-back option-bar requests and the
provider retried only once → fixed in the same session (bounded exponential
backoff honouring Retry-After, pacing between events, NO_DATA rows persisted
and re-runnable, per-event outcomes in the response). After the fix: 7 of 8 previous AAPL events priced (implied median |4.16 %| / p90 4.53 % vs realised median |1.33 %| / p90 3.74 %; 5 OVER_PRICED, 2 FAIR); the 8th (2024-08-01) is an honest NO_DATA — the Massive base plan does not serve option aggregates older than ~2 years (HTTP 403 "data timeframe"), surfaced verbatim in `coverage.history_attempted_no_data`.

**Tests.** `test_market_data_option_history.py` (33),
`test_events_implied_move.py` (89), `test_events_options_api.py`, parity pins
for 025; UI `OptionsTab.test.tsx`. Suite after Phase I + F fixes: 3457
passed / 1 skipped backend; UI 440 vitest; tsc clean.

**Known limitations.** Historical IV is reconstructed from daily closes (no
quote/greeks history on the base plan) and the event straddle usually
expires the next day, so `iv_after`/IV-crush are NO_DATA for weekly-expiry
events (documented in `notes`); Massive rate limits bound the backfill pace;
live implied move depends on the current chain being open; no volatility-
surface or skew analytics.

**Production status.** DEPLOYED (migration 025 live, gateway rebuilt with
F fixes + I; second rebuild with the 429 fix).

**Next.** Phase J UI slices (hero, timeline, evidence/scenarios tabs, card
summaries — in flight), then G/H macro & Fed, K risk SHADOW, L validation +
§100 report.

## 2026-08-19 (27) — CATALYST & EVENT INTELLIGENCE, PHASE F: earnings intelligence — EvidenceBundle, schema-validated LLM analysis, event memory, Analysis tab (DEPLOYED)

**Purpose.** Spec §16/§33–§35/§46–§52/§69–§71: assemble ONE structured
EventEvidenceBundle from the Phase B–E/C/D seams, hand it to the configured
LLM under §47 ("the backend calculates, the LLM interprets"), validate the
answer against the bundle (every number the model quotes must exist at the
path it cites), persist bundle + analysis as institutional memory (§69), and
render DATA / QUANT / LLM ANALYSIS as visibly separate tiers (§49).

**Existing capability reused.** `build_price_context` (E1),
`build_fundamentals_context` (E2), replay/history (C), `build_event_news` +
`sanitize_for_llm` (D); `libs/llm` provider registry (OpenAI Responses API
with strict `json_schema`, Anthropic `output_config`, deterministic stub —
same factories, same `require_llm_provider` → 503 `LLM_NOT_CONFIGURED`);
audit helper; `_event_or_404` / `_resolve_as_of`.

**Architecture decision.** (1) Pure `libs/trading_core/events/evidence.py`
(`f1-evidence-v1`): `EvidenceBundle` with fixed `SECTION_ORDER`
(event, as_of, previous_event, previous_event_results,
previous_market_reaction, fundamentals, price_analysis, options_analysis,
news, consensus, expectations_gap_inputs, macro_context, peer_context,
prior_analyses, source_metadata, coverage); every section carries
`tier: DATA|QUANT`; `bundle_digest` = sha256 of canonical JSON;
`fact_index` flattens every fact to a dotted path (880 facts / 173 numeric on
the live AAPL bundle); `compute_expectations_gap_inputs` gives §35 INPUTS
(fundamental momentum −1..1 from improved/weakened metric counts, run-up
since previous event, relative return, distance from 52-w high, realised
vol, material +/− development counts) and deliberately NO regime label —
the four §35 regimes are the LLM's enum. Consensus is always
`CONSENSUS_DATA_UNAVAILABLE` (§33; Benzinga add-on not purchased); options /
macro / peer are honest placeholders until Phases I/G. News enters ONLY
sanitised (`safe_title`/`safe_description`); articles flagged
`suspicious_instruction` are excluded from the LLM view and counted. (2)
`libs/llm/event_analysis.py`: strict schema (`event-analysis-v1`) with the
§48 sections, UPSIDE/BASE/DOWNSIDE scenarios (§51), surprise-threshold
narrative with confidence NOT_MEANINGFUL allowed (§52), `invalidation`
(§50), `expectations_gap_regime` enum, `evidence_refs`, and
`numbers_quoted[{path,value}]`; `validate_analysis` rejects any quoted path
missing from `fact_index` or whose value mismatches (1e-6) — violations are
stored, not hidden (status INVALID). `analyze_event(bundle_json, as_of)` on
the provider Protocol (OpenAI/Anthropic capture `usage`; refusal → error;
stub quotes real facts from the bundle so the validator is exercised).
(3) `event_analyses` (migration 024, applied live): bundle JSONB beside the
analysis, `bundle_digest`, provider/model/prompt_version/usage/latency,
`violations`, `status` OK|INVALID|FAILED|SUPERSEDED, PARTIAL unique index
`(event_id, bundle_digest, prompt_version, model) WHERE status='OK'` so a
retry after FAILED is storable and `force` supersedes rather than collides;
`prior_analyses` (§69) are injected as `tier: LLM_PRIOR`, gated on `as_of`
and OK-only, and the system prompt says they are opinions, not evidence
(§70). Provider failure → stored FAILED row, HTTP 200 with the bundle — never
a 500. Audit `EVENT_ANALYSIS_GENERATED` (entity_id = numeric event id).
(4) Endpoints `GET /api/events/{id}/evidence`, `GET|POST
/api/events/{id}/analysis[?as_of&force]`, `GET /api/events/{id}/analyses`.
(5) UI Analysis tab: tier chips DATA / QUANT / LLM ANALYSIS, scenario cards,
surprise-threshold chip ("not a probability"), invalidation, prior analyses
collapsed, Generate/Regenerate (503 → configure-LLM notice), INVALID shown
with its violations banner.

**Live.** AAPL event 99 (ESTIMATED 2026-10-29, badge carried into the
payload): `GET /evidence` 0.7 s, coverage all sections except consensus /
options; fundamental momentum +0.57 "fundamentals_improving" (11 improved /
3 weakened of 14), run-up since previous event −7.0 %, relative return vs
benchmark −10.5 %, 106 material developments. `POST /analysis` with the
live provider (OpenAI `gpt-5.6-sol`, zh): status OK in 51 s, 28 485 input /
3 070 output tokens, regime BAD_NEWS_PRICED at MODERATE confidence, three
scenarios, surprise threshold LOW confidence citing the missing consensus,
zero validator violations. Two findings from the live run were fixed in the
same session: the cache digest included the wall-clock `as_of`, so a second
POST a minute later missed the cache and spent a second LLM call, which then
hit the provider's 60 s read timeout → digest now hashes a pruned view
without volatile timestamps, `LLM_ANALYSIS_TIMEOUT_SECONDS` (240 s) is
threaded to `analyze_event` only, `GET /analysis` prefers the latest OK row
and reports `last_attempt` when the newest row failed, and the prompt now
lists quotable numeric facts and requires ≥3 `numbers_quoted` (the first
live answer cited none).

**Tests.** `test_events_evidence.py` (34), `test_llm_event_analysis.py`
(45), `test_events_analysis_api.py` (30 incl. 5 mutation-verified
end-to-end wiring tests: real bundle → real stub → real validator → router),
parity pins for 024; UI `AnalysisTab.test.tsx` (58, mutation-verified).
Suite after Phase F: 3281 passed / 1 skipped backend; UI 396 vitest; tsc
source-clean. (Post-fix counts in the next entry's baseline.)

**Known limitations.** No consensus/estimate provider → EXPECTATIONS GAP uses
price/news proxies only and `surprise_threshold` is narrative (§52 "LLM-
supported construct"); `numbers_quoted` validation enforces provenance, not
interpretation quality; prior analyses are summaries only; `kind=POST_EVENT`
memory (§69 actual result + reaction) is stored by the same table but no
post-event writer exists yet (Phase L/K); one LLM call ≈ 30 k input tokens.

**Production status.** DEPLOYED (migration 024 live; gateway rebuilt with the
Phase D tuning and Phase F; second rebuild with the live fixes).

**Next.** Phase I options / implied move (historical ATM straddle from Massive
option daily bars + live chain), then J UI slices, G/H macro & Fed, K risk
SHADOW, L validation + §100 report.

## 2026-08-19 (26) — CATALYST & EVENT INTELLIGENCE, PHASE D: news evidence engine — dedup, story clusters, materiality, EvidenceScore, News tab (DEPLOYED)

**Purpose.** Spec §21–§27 with §14/§47 prep: turn the raw `news_articles`
store into EVIDENCE for an event — a window from the previous comparable
event to `as_of`, normalised → deduplicated → story-clustered → ticker
relevance → materiality category (§24) → novelty → source quality → time
decay → one multiplicative EvidenceScore (§25) → §26 counts/themes — with
every number as-of-safe and every article's untrusted text isolated for the
LLM (§27 traceability: evidence ids `news:<provider>:<source_id>`).

**Existing capability reused.** `news_articles` table/ORM (migration 012) and
the providers' news readers; Phase B events + Phase C `previous_event_id`
(window basis `previous_earnings:<event_key>`); Phase C's POST-only backfill
pattern; the `_event_or_404` / `_resolve_as_of` router seams; audit helper.

**Architecture decision.** (1) Providers gained a WINDOWED reader
`get_news_window(tickers, start, end, limit)` (Alpaca symbols CSV +
page_token; Massive per-ticker `published_utc` range + `next_url`; stub) —
the recency cursor `get_news` is useless for a window that closed last week.
(2) Pure `libs/trading_core/events/news_intel.py` (stdlib, model version
`news-intel-v1`): dedup ≥0.8 title-shingle Jaccard; story clustering is
LEADER-based (an article joins the cluster whose canonical title has the
highest Jaccard ≥0.45 within 7 days, or shares ≥2 salient entities —
subject ticker excluded — within 48 h; a cluster may not exceed 40 % of the
window's unique articles; headline template words such as "stocks to watch /
heading into / before the bell" are stripped before shingling) — the first
live run exposed why: single-link transitivity fused 269 of 283 AAPL articles
into one cluster through Benzinga's daily templated headlines. (3) Relevance
1.0 (title/tag) / 0.7 (description only) / 0 ; materiality lexicon with
category weights (GUIDANCE, ANALYST_REVISION, LEGAL, REGULATION, PRODUCT,
SUPPLY_CHAIN, CAPITAL_ALLOCATION, M&A, MACRO, …), zero sentiment vocabulary
by design; novelty = first-in-cluster; source-quality table (unknown 0.5);
14-day half-life decay floored at 0.2. `material` developments are counted on
the NO-DECAY score (relevance×materiality×novelty×source ≥0.25) — an old
development in the window is still a development (§26); ranking uses the
decayed score; both `score` and `score_no_decay` are exposed. (4) Persisted
columns are as-of-INDEPENDENT only (`cluster_id`, `materiality`,
`materiality_score`, `source_quality`, `relevance` JSONB — migration 023
ADDITIVE on `news_articles` + tickers GIN index); novelty/decay/score are
computed per request and never stored. (5) LLM isolation: `sanitize_for_llm`
strips markup/links/control chars, caps length and FLAGS suspicious
instructions (`suspicious_instruction`) — the display keeps provider bytes
verbatim; nothing is censored, only flagged. (6) Seam
`apps/gateway/event_news.py`: `ensure_event_news_window` (POST-only fetch +
upsert) and `build_event_news` (GET never fetches — pinned by an exploding
provider test); endpoints `GET /api/events/{id}/news?as_of=` and
`POST /api/events/{id}/news/backfill`; UI News tab (counts, themes, clusters,
evidence table with score components; keys identical to the API).

**Live.** Migration 023 applied; AAPL `EARNINGS:AAPL:2026-10-29` (event 99)
window 2026-07-29T20:30Z → as_of: 283 articles fetched (Alpaca 206 +
Massive 77), 280 stored, 278 unique. Before tuning: 10 clusters (one of
269), material 1, themes 1. After tuning (fixture-pinned, see Tests):
on the exported live window (`tests/fixtures/events/news_aapl_window_2026-08.json`, the same 283 articles): 177 clusters, largest 6 (2.2 %), material 104, themes 14 — PRODUCT·apple,iphone (29), EARNINGS (19), SUPPLY_CHAIN (10), ANALYST_REVISION (10), MANAGEMENT (8), GUIDANCE/REGULATION/MACRO (4 each), then CAPITAL_ALLOCATION, LEGAL, COMPETITION, M&A, CUSTOMER, CONTRACT; "Needham Reiterates Hold" → ANALYST_REVISION, "Apple launches legal challenge to UK" → LEGAL, "CXMT memory chips" → SUPPLY_CHAIN. Two extra defences the live data forced beyond the contract: provider tag lists longer than 3 symbols are dropped as entity evidence (Benzinga market wraps co-tag 10–45 symbols), and entities present in ≥33 % of the window ("APPLE" was in 144/283) lose linking power; the 40 % cap is floored at 5 so two-article syndication windows still cluster. The tuned engine ships in the same gateway build as Phase F (re-run live after deploy).

**Tests.** `test_market_data_news_window.py`, `test_events_news_intel.py`
(incl. the REAL fixture `tests/fixtures/events/news_aapl_window_2026-08.json`
asserting largest cluster ≤40 %, ≥25 clusters, category hits for the Needham
reiteration / UK legal challenge / CXMT supply-chain stories, determinism),
`test_events_news_api.py` (as-of leak test patches the loader to return
every row; GET-never-fetches), migration parity (`_CREATE_PLUS_ALTER_TABLES`),
UI News tab tests. Suite: 3171 passed / 1 skipped backend after tuning (3150 before); UI 338 vitest, tsc source-clean; an independent verifier re-ran the fixture and the 5 checks with zero fixes.

**Known limitations.** Lexicon-based materiality (no ML, no sentiment by
design); entity extraction is capitalised-token heuristics; source-quality
table is hand-curated; Massive news is per-ticker only; the window basis
falls back to 30 days when no previous event is linked; clustering is
title/entity only (no body text similarity).

**Production status.** DEPLOYED (gateway rebuilt twice: engine, then tuning).
No schema change beyond additive 023.

**Next.** Phase F earnings intelligence (EvidenceBundle → schema-validated
LLM analysis with `numbers_quoted` validation → `event_analyses` memory),
then Phase I options / implied move.

## 2026-08-19 (25) — CATALYST & EVENT INTELLIGENCE, PHASE C: event replay — intraday reactions, §60 history table, previous-event linkage (DEPLOYED)

**Purpose.** Spec §15/§17/§19/§20/§60 with §14/§85/§96: reconstruct a past
event (information before → release → immediate intraday reaction →
subsequent days), keep a LAST 4/8/12 history view, and persist the
previous-comparable link with its `comparison_reason`.

**Existing capability reused.** `stock_bars_1m` hypertable (migration 002,
never written before today), `ensure_daily_bars`/E1 `reaction.py` for daily
windows, Phase B `previous_comparable`, provider transport injection,
market_calendar (Phase B) to find the next session.

**Architecture decision.** Provider layer: `IntradayBar` + `get_intraday_bars
(symbol, start, end, timeframe="1Min")` on the protocol — Alpaca `/v2/stocks/
{sym}/bars` (iex, adjustment=split, page_token loop, extended-hours bars KEPT),
Massive aggs minute range (ms bounds, next_url), stub; aware-UTC required,
ascending, de-duplicated, volume required (no fabricated 0). Pure
`libs/trading_core/events/replay.py`: `intraday_reaction` with session-correct
anchors (AMC: after-hours move vs pre-close, next-session open gap, +5m/+30m/
+60m; BMO: pre-market move then release-day open windows; DURING: release-
anchored; UNKNOWN: assumed-AMC flagged, confidence low), `max_move_first_hour`,
first-30-min volume (ratio only when prior-session bars are supplied),
`EventReplay`/`build_event_replay` (§20 object), `history_table` (§60 rows
with UNAVAILABLE EPS/revenue surprise and implied move), `link_previous_events`
(never crosses types; ESTIMATED → latest CONFIRMED prior; CANCELED excluded).
Gateway `apps/gateway/event_replay.py`: minute-bar windows are fetched ONLY
on explicit `POST …/replay/backfill` / `POST …/history/backfill?last=N`
(bounded, default 4) — never on GET; stored via `StockBar1mRow` (ORM mirror,
composite PK, parity extractor generalised for table-level keys); as-of gate
is SQL `ts <= as_of`; `pre_event_close_for` anchors intraday on the correct
daily pre-close. Endpoints: `GET /api/events/{id}/replay`, `GET …/history`,
the two POST backfills. Registry: the ingest tick now persists
`previous_event_id` + `comparison_reason` for EARNINGS/FOMC_DECISION/
FOMC_MINUTES (idempotent, no audit spam). UI: "Previous Event" tab (release
info, immediate-reaction tiles with reasons, "Load minute bars", subsequent
1D/3D/5D/10D) and a "History" tab (LAST 4/8/12 toggle, UNAVAILABLE columns,
backfill button); zh+en.

**Live.** Linkage 104/110 rows; AAPL 2026-07-30 AMC release (event 94):
backfill 853 minute bars (0.2s, `spans_next_session:market_calendar`),
after-hours −4.5% (22 bars), gap at open −8.6%, +5m −9.4%, max move first
hour 1.4%, first-30-min volume 1.30M; history for event 99: 12 rows, 10 with
daily reactions, intraday_30m honestly "no minute bars stored" until
backfilled; EPS/revenue surprise and implied move UNAVAILABLE with reasons.

**Tests.** Backend **2852 passed, 1 skipped** (+239): test_market_data_
intraday (58: pagination, dedup, ascending, naive/ reversed windows, 403,
ms-vs-ns timestamps), test_events_replay (112: anchors per session incl. DST,
sparse after-hours, windows beyond data, history_table, linkage never crossing
types), test_events_replay_api (62) + 7 linkage tests (future event →
available:false; as-of gate on minute bars; no GET fetch; backfill idempotent;
parity tripwire mutation-checked). UI 305 passed (+49), tsc clean. Verifier:
zero fixes needed.

**Known limitations.** iex minute history reaches ~2024 (older events have no
intraday); after-hours bars are sparse; prior-5-day volume baseline only when
bars supplied (not fetched yet); `previous_event_id` for macro types deferred
to Phase G; no timeline/chart yet (Phase J).

**Production status.** DEPLOYED (gateway rebuilt; no new migration — ORM now
mirrors the existing `stock_bars_1m`).

**Next.** Phase D — News Evidence Engine over the existing `news_articles`
store (normalise → dedup → cluster → relevance → materiality → novelty →
source quality → decay → rank; `(ticker, published_at)` index; window =
previous comparable event → as_of).

## 2026-08-19 (24) — CATALYST & EVENT INTELLIGENCE, PHASE E2: point-in-time fundamentals (DEPLOYED)

**Purpose.** Spec §16/§28/§29/§30/§33/§35 under the §14/§85/§96 as-of
contract: "what are fundamentals now, and what changed since the previous
event", computed by the backend (§47) from filings visible at `as_of`,
with every non-computable ratio an explicit absence (§98).

**Existing capability reused.** `MassiveProvider._request`/`_json` (auth,
401/403/429 taxonomy), `ensure_daily_bars` + `as_of_bar_filter` for the
valuation price, Phase B registry (`previous_comparable`), E1 payload
conventions (provenance, `unavailable[]`, derived-reason fill),
`.provenance` UI classes, glossary ⓘ.

**Architecture decision.** Provider layer: `FinancialStatement` on the
`MarketDataProvider` protocol + `get_financials(ticker, timeframe, limit)`
(Massive `/vX/reference/financials`, values flattened
`statement.field`→float, non-finite skipped, unreported fields ABSENT —
never 0; Alpaca raises `CapabilityNotAvailable`; `financials` probe key on
both). **Statements provider ≠ price provider**: `fundamentals_provider_name`
selects Massive whenever its key is configured (the market-data provider is
Alpaca, which has no financials — live first call stored 0 rows before this
split). Pure library `libs/trading_core/events/fundamentals.py`: the as-of
gate is `acceptance_datetime <= as_of` ONLY (never period end; rows without
an acceptance instant excluded with a reason); newest-first on
`(end_date, acceptance)` so a dual filing/restatement resolves
deterministically; ratios only when inputs exist; P/E None for a loss;
YoY = (later−earlier)/|earlier|; **derived TTM** = sum of the four newest
visible quarters (acceptance = the latest of the four) because Massive's TTM
rows carry no acceptance instant — labelled `notes.ttm_basis`. Table
`fundamental_statements` (migration **022**, applied live) mirrors filings
with `values JSONB`; seam `apps/gateway/fundamentals.py` (`ensure_fundamentals`
20h staleness + 6h attempt throttle, DATA_BACKFILL audit kind=fundamentals;
`build_fundamentals_context`: current snapshot at as_of, previous snapshot
as of the previous print, §29 changes with bps/direction/trend over ≤8
quarters, §30 valuation vs own history (price = last close ≤ as_of),
`fundamental_momentum` label with counts (QUANT, not a signal), consensus
block `CONSENSUS DATA UNAVAILABLE — Benzinga 403`). Endpoint
`GET /api/events/{id}/fundamentals?as_of=`. UI: Fundamentals tab (§58
PREVIOUS/CURRENT/CHANGE table with ↑/↓ and bps, valuation tiles vs own
history, consensus banner, freshness line, DATA/QUANT labels, ⓘ, zh+en).

**Live (AAPL event 99, as_of 2026-08-19).** 13 statements stored (12
quarterly + 1 TTM w/o acceptance); FQ3'26 (period 2026-06-27, accepted
2026-07-31 10:01Z): revenue $109.4B (+16.4% YoY), GM 50.1% (+79 bps q/q),
OM 32.6% (+35 bps), NM 27.2% (+62 bps), EPS $2.02 (+28.7% YoY), OCF $34.4B,
current ratio 1.00, D/E 0.77 (long-term only, noted); derived TTM revenue
$458.4B, EPS $8.44 → P/E 36.7 (own-history median 35.5, 56th pct, n=9),
P/S 9.95 (78th pct), P/B 42.4 (10th pct, n=10), ROE 116%; FCF/capex/cash/
net debt/EBITDA/quick ratio/EV-EBITDA/FCF-yield honestly "not reported by
provider"; momentum `fundamentals_mixed` (6 improved / 3 weakened / 5
unavailable of 14).

**Tests.** Backend **2613 passed, 1 skipped** (+226): test_market_data_
financials (70: parsing, 403, malformed/non-finite, probe keys),
test_events_fundamentals (109: as-of on acceptance incl. the +1h sentinel,
YoY same-quarter, zero-revenue margins, no fabricated zeros (676-case
sweep), change bps/trend, valuation with/without price, derived TTM),
test_events_fundamentals_api (47: provider separation, ORM→pure naive-UTC
adapter, 404/200 semantics). UI 256 passed (+48), tsc clean. Verifier: zero
violations.

**Known limitations / NOT BACKTESTABLE.** Consensus/revisions/guidance
(Benzinga 403) → EPS surprise % impossible; capex/cash/D&A/receivables not in
provider XBRL → FCF, net debt, EBITDA multiples, quick ratio unavailable;
sector/peer multiples deferred (Phase G/J); `total_debt` is long-term only;
provider TTM rows unusable point-in-time (derived TTM used); settings page
shows the new `financials` probe key with a raw label (bilingual label TODO).

**Production status.** DEPLOYED (migration 022 applied, gateway rebuilt).

**Next.** Phase C (event replay: `previous_event_id` linkage, LAST 4/8/12
history view, intraday 5m/30m/1h reactions into `stock_bars_1m` for event
windows) then Phase D (news evidence engine).

## 2026-08-19 (23) — CATALYST & EVENT INTELLIGENCE, PHASE E1: price context + previous-event market reaction (DEPLOYED)

**Purpose.** Spec §17/§19/§20/§31/§32/§64 with §14/§96 as-of discipline:
deterministic, point-in-time-safe price positioning before an event and
the market reaction to every stored previous release, so the LLM (Phase F)
interprets numbers the backend computed (§47) and the UI shows QUANT
blocks distinct from DATA.

**Existing capability reused.** `features/indicators.py` (sma/atr/
realized_vol — never reimplemented, equality-tested), `routers/analysis.py::
ensure_daily_bars` (complete-days-only store, SPY = ADR-005 reference),
Phase B registry rows + `previous_comparable`, `EASTERN` from
`events/taxonomy.py`, `.provenance` UI classes, `Term`/glossary ⓘ.

**Architecture decision.** Pure library `libs/trading_core/events/reaction.py`
(stdlib; AST-verified no apps/market_data imports) + one gateway seam
`apps/gateway/event_price.py` + one endpoint; computed on demand (bars are
local), no new table. Session-correct windows: AMC on D → pre=close(D),
react=next bar; BMO → pre=close(D-1), react=D; DURING flagged
`during_market_same_day`; UNKNOWN flagged `unknown_session_two_day_span`.
Benchmark (SPY) aligned by calendar date via `window_end_dates`, not index.
Run-up anchored on the previous print's **pre-event close** (a BMO print's
own bar already contains its reaction). **As-of gate for prices**:
`as_of_bar_filter` keeps bar d iff d < as_of ET date, or d == as_of date and
as_of ≥ 16:00 ET — tested at 15:59/16:00/16:01 ET across EST/EDT with
literal UTC instants; naive datetimes rejected. Every null carries a reason
(derived-reason fill so tiles never go blank); history stats always carry
`n`/`n_available` + `positive_count` (never a probability, §19/§64).

**New code.** `reaction.py` (DailyBar, ReactionResult, AbnormalResult,
HistoryStats, PriceContext; `first_reaction_index`, `event_reaction`,
`abnormal_vs`, `history_stats` last4/8/12 nearest-rank percentiles,
`pre_event_price_context`, `as_of_bar_filter`); `event_price.py::
build_price_context`; `GET /api/events/{id}/price-context?as_of=` (404
unknown, 200 `available:false reason:no_ticker` for macro/Fed events,
provider unconfigured → bars block unavailable, endpoint still 200; payload
with `provenance {bars: DATA, metrics: QUANT}`, `data_freshness`,
`anchor_event`, `pre_event`, `previous_events[]`, `history_stats {1D,5D}`,
`not_backtestable`, `unavailable[]`). UI: event detail "Price" tab —
positioning tiles with ⓘ, previous-reactions table (gap/1D/3D/5D/10D/
abnormal, "bars unavailable before …" rows, UNKNOWN-session flag), history
strip "Last 8: median |1D| … p90 … positive 5/8 — based on 8 events",
freshness; zh+en.

**Live (AAPL, event 99, as_of 2026-08-19 17:41Z).** bars_through
2026-08-18 (604 bars, SPY 605); run-up since the 2026-07-30 print −7.0%,
RV20 34.7%, ATR 2.4%, SMA20 −2.0% / SMA50 +0.3% / SMA200 +10.4%, −10.0%
from 52w high; 10 of 12 prior releases measured (2023-11/2024-02 honestly
"bars unavailable before 2024-03-21"); e.g. 2024-05-02 AMC gap +7.9%, 1D
+6.0%, 5D +6.7%, abnormal 1D +4.7%; last8 |1D| median 1.3%, p90 7.4%,
positive 4/8.

**Tests.** Backend **2387 passed, 1 skipped** (+118): test_events_reaction
(73: windows AMC/BMO/DURING/UNKNOWN, Friday→Monday, holiday fall-through,
event before first bar / too recent, zero pre-close → no NaN/inf,
benchmark gaps, percentile nearest-rank, as-of boundary + DST),
test_events_price_api (45: look-ahead — as_of before the react bar →
reaction unavailable; 15:59 ET excludes same-day bar; macro event; provider
unconfigured; BMO anchor mutation-checked). UI 208 passed (+28), tsc clean.
Verifier caught and fixed a real seam bug: U2 emitted `"1D"` keys while the
UI looked up `1d` — every reaction cell would have rendered Unavailable.

**Known limitations.** Reactions before 2024-03-20 need a deeper bar
backfill (not run — user decision pending since the risk audit); sector
benchmark not yet mapped (SPY only); intraday 5m/30m/1h windows deferred
to Phase C (`stock_bars_1m` still unused); `volume_trend` needs 80 bars;
`realized_vol_since_anchor` needs ≥3 bars.

**Production status.** DEPLOYED (gateway rebuilt; UI hot-reloaded).

**Next.** Phase E2 — fundamentals: Massive `/vX/reference/financials`
adapter (quarterly + TTM, `filing_date`/`acceptance_datetime` as the as-of
key), `fundamental_snapshots` table (migration 022), §28 ratios, §29
prev-vs-current Δ, consensus fields honestly UNAVAILABLE.

## 2026-08-19 (22) — CATALYST & EVENT INTELLIGENCE, PHASE B0+B: event registry, calendar providers, `/api/events`, first Catalysts page (DEPLOYED)

**Purpose.** Phase B of the Catalyst program per the audit's adjusted plan
(`docs/catalyst-event-audit.md` §11.1): a typed event registry fed by
authoritative calendar providers, an ingestion loop, the `/api/events`
surface, the T-minus alert, and a usable Catalysts page — so every later
phase has a visible acceptance surface. Built as five units (pure domain,
persistence, providers, gateway seam+API, UI) + verifier + fixer, then live
deployment fixes from the first real ingest.

**Existing capability reused.** `risk_snapshot.py` loop shape (NY-day
bucketing, named skips, tick split out for tests, CancelledError
re-raised); transactional `audit.record()`; ADR-006 `ALERT_RULES`;
`market_data` registry contract + httpx transport injection/MockTransport
pattern; migration-parity tripwires; `Term`/`glossary` ⓘ, `.provenance`
classes, `.seg-control`, TanStack Query hooks in the UI.

**Architecture decision.** ADR-008 (new): separate `libs/event_calendar/`
registry; one event = one row with source precedence (LLM never writes
dates); ESTIMATED badged + never alerted; EARNINGS ±21d reconcile window;
SEC provider clusters Item 2.02 filings (earliest = release); audit
`entity_id` = numeric `events.id`; router named `events`.

**Data providers (live, keys from `runtime_config`).** `sec_edgar`
(data.sec.gov submissions JSON: 8-K Item 2.02 `acceptanceDateTime` → 12
CONFIRMED past releases per ticker, AAPL/HPE/RDW/SMCI, Nov-2023→Jul-2026;
cadence estimate → 4 ESTIMATED upcoming, all AMC: HPE 10-14, SMCI 10-22,
AAPL 10-29, RDW 11-04); `fed` (FOMC calendar HTML → 29 meetings
2021-2027 with decision 14:00 ET / presser 14:30 ET / minutes dates;
speeches RSS → 15 FED_SPEECH); `alpaca_calendar` (552 trading days
2025-07→2027-09, 4 early closes); `massive_calendar` (6 holidays;
Benzinga earnings probe = 403 → `earnings_calendar=false`). Live registry:
189 events across 9 types.

**New models / code.** Enums `EventType`(18)/`EventStatus`/`EventSession`/
`EventLifecycle`/`EventSourceKind` + 5 `AuditAction`s; migration **021**
(`events`, `market_calendar`, `event_ingest_state`) applied live + ORM
mirror + parity registration + compose mount; `libs/trading_core/events/`
{models (Event/EventCandidate/`same_event`/`merge`/`previous_comparable`),
taxonomy (event_key, ET↔UTC, `classify_session`, lifecycle), importance
(transparent `ImportanceResult` with components, v1)}; `libs/event_calendar/`
{provider protocol + fixed capability keys, registry, alpaca_calendar,
massive_calendar, sec_edgar (CIK map, look-ahead `as_of`, `cluster_releases`,
`estimate_next_earnings` with roll-forward), fed (markup-keyed FOMC row
extractor + HTMLParser fallback, RSS "Surname, Title" + link-slug speaker
parse), stub}; `apps/gateway/event_calendar.py` (`run_calendar_ingest`,
per-provider 20h cadence via `event_ingest_state`, upsert+merge, relevance
POSITION>POOL>WATCHLIST>MARKET_WIDE>OTHER, exactly-once EVENT_APPROACHING,
`event_calendar_loop`); `routers/events.py` (`GET /api/events` horizon
today/7d/30d/custom + filters, `GET /{id}`, `POST /refresh`,
`POST /{id}/confirm`, `POST /{id}/cancel`, `GET /calendar`); settings
`event_calendar_interval_seconds=3600`, `event_horizon_alert_days=7`,
`sec_user_agent`, `event_calendar_providers`; alert rule for
EVENT_APPROACHING. UI: `/catalysts` (horizon control, estimated toggle,
refresh, capability banner, relevance groups, `EventCard` with status
badge/T-minus/importance ⓘ breakdown/confirm-date dialog), `/catalysts/
[eventId]` hero + previous-event card + placeholder tabs, Nav entry
(Catalysts / 催化剂), zh+en strings, `quant-derived` provenance tier.

**Live-deployment fixes (caught only against Postgres / real sources).**
(1) `audit_events.entity_id` VARCHAR(64) overflow on FED_SPEECH keys → numeric
id; (2) real Fed RSS titles are "Cook, Topic" not "Speaker: Topic"; (3) real
FOMC page layout (`fomc-meeting__month/__date`, "(Released …)") not parsed
→ regex row extractor + live fixture; (4) SEC `www` host 403s without a
contact e-mail in the User-Agent → `SEC_USER_AGENT` set in `.env`,
documented in `.env.example`; (5) follow-up 8-K 2.02 filings within 21 days
were turning CONFIRMED rows into REVISED → `cluster_releases`; (6) a
year-ago cadence anchor already in the past yielded no estimate → roll
forward by median gap; (7) `LOOKBACK_DAYS` 400→1200 so 12 past releases are
retained (§19).

**Tests.** Backend **2269 passed, 1 skipped** (baseline 2020; +249):
test_events_models (63), test_events_db, test_event_calendar_providers
(87 incl. 403→capability false, 5xx→error string, SEC as-of look-ahead,
estimator cases, DST/non-DST, live FOMC fixture, live RSS format,
clustering, roll-forward), test_events_api (40), test_event_calendar_loop
(29: idempotent refresh, exactly-once alert across ticks and a new
session, ESTIMATED never alerts, one provider raising → others ingested).
UI 180 passed (baseline 114), `tsc` clean, `next build` OK, no native
dialogs.

**Known limitations.** Earnings dates are ESTIMATED until confirmed; no
consensus/guidance (Benzinga 403); FOMC minutes for the latest meeting are
ESTIMATED (decision+21d) until the page lists the release; MARKET_HOLIDAY
appears once per exchange (NYSE+NASDAQ); macro (BLS/BEA) calendars not
yet ingested (Phase G); `lifecycle()` POST_EVENT counts weekdays, not
holidays; `npm run lint` is uninitialised project-wide (pre-existing).

**Subscription dependencies.** Massive Benzinga Earnings add-on would turn
ESTIMATED into CONFIRMED and supply consensus; SEC requires a contact UA.

**Production status.** DEPLOYED — migration 021 applied, gateway rebuilt,
loop running hourly, UI live at `/catalysts`.

**Next.** Phase E1 (price context: pre-event run-up / post-event daily
reaction for the 48 stored past releases, macro ETF proxies, as-of) then E2
(fundamentals via Massive `/vX/reference/financials`).

## 2026-08-19 (21) — CATALYST & EVENT INTELLIGENCE, PHASE A: architecture / capability audit (no code)

**Purpose.** Start of the Catalyst & Event Intelligence program
(`prompts/event_analy_system.md`, 102 sections, phases A–L). Spec §1/§93
forbid implementing before inspecting, so this iteration is audit-only:
live provider-entitlement probes (Massive, Alpaca, and the free primary
sources BLS/BEA/Fed/SEC/Census/Treasury) → 6 parallel read-only
inspections (gateway/infra, providers, quant libs, LLM+risk+trade flow,
UI, tests/docs) → 6 adversarial verifications (47 corrections applied:
drifted paths, a non-existent `apps/gateway/pretrade.py`, the existing
`GET /api/analysis/{ticker}/catalyst` naming collision, `anthropic.py`
lacking `enrich()`, no token usage returned by any LLM provider) →
4-part synthesis. Output: **`docs/catalyst-event-audit.md`** (603 lines:
reuse inventory, §100.B data coverage matrix, gap matrix, decomposition,
provider abstraction, as-of design, risk/LLM/UI plans, adjusted phase
plan, open questions, NOT BACKTESTABLE list).

**Existing capability discovered.** Scheduler/seam pattern to copy
(`apps/gateway/risk_snapshot.py`: NY-day bucketing, named skips, tick
split out of lifespan for tests); transactional audit + ADR-006 alert
classification (`alerts.py::ALERT_RULES`); news store with dedup on
`news_articles.source_id` and the grounding-enforcement that drops
uncited LLM drafts (`routers/recommendations.py:286-295` — §27/§79
already in code); shadow-mode precedent in `routers/orders.py` (`"shadow"`
dict at :2075) with an AST tripwire that no `apps/` call passes
`extra_caps` to `assess()` (`tests/test_risk_adversarial.py:1664-1730`);
daily bars via `routers/analysis.py::ensure_daily_bars`; an existing but
**never-written** `stock_bars_1m` hypertable (migration 002); UI
provenance classes `.provenance.data-driven/.llm-generated`
(`ui/app/globals.css:633-639`), `Term.tsx`+`glossary.ts` (65 bilingual
entries) as the ⓘ mechanism, no chart library (inline SVG only);
`TradePlanRow` as the shape for a persisted research package;
migration-parity tests that force `021_*.sql`, compose `:ro` mounts and
`_SINGLE_CREATE_TABLES` registration.

**Data coverage (live probe 2026-08-19, keys from `runtime_config`).**
Massive base plan 200: news, financials (`filing_date` +
`acceptance_datetime` → point-in-time key), tickers, stock + option daily
bars, dividends, IPOs, holidays, related-companies (peers), treasury
yields 1y/5y/10y, CPI series, inflation expectations; options snapshot
carries NO IV/greeks on this plan. Massive **403 "not entitled"**: every
`/benzinga/v1/*` endpoint — **earnings calendar, EPS/revenue
actual/consensus/surprise, ratings, guidance, analyst insights** (the
single blocking external gap). Alpaca Algo Trader Plus 200: trading
calendar, corporate actions (no earnings), Benzinga-sourced news, live
option snapshots WITH IV+greeks, historical 1-minute bars (iex). Free
primary sources all reachable: BLS API v2 + schedule pages, BEA API +
schedule, FOMC calendar HTML + speeches/press RSS, SEC EDGAR submissions
JSON (`acceptanceDateTime`) + 8-K full-text search, Census calendar,
Treasury daily yield-curve CSV (has the 2y Massive lacks).

**Architecture decisions.** No "skill framework": the five spec skills
become capability groupings over one shared core — pure modules under
`libs/trading_core/events/` (models/taxonomy, reaction, evidence,
news_intel, fundamentals) + one gateway seam per concern
(`event_calendar` loop, `event_analysis` builder, router named
**`events`**, not `catalyst`, because `/api/analysis/{ticker}/catalyst`
already exists and stays). New provider registry `libs/event_calendar/`
mirroring `market_data.__init__._PROVIDERS` (no default, no
cross-provider fallback); Benzinga earnings surfaces as a probed
capability `earnings_calendar=false`, never a crash; upcoming earnings
dates are **ESTIMATED** from filing cadence (SEC 8-K Item 2.02
`acceptanceDateTime` / Massive `filing_date`) with a visible badge and
excluded from alerts, until user-confirmed or a subscribed calendar
confirms; optional third-party calendar adapter built disabled behind a
`runtime_config` key, never silently added. As-of: one `AsOf` primitive
threaded end-to-end, filtering on publication timestamps
(`published_at`, `acceptance_datetime`, bar `t`, SEC `acceptanceDateTime`),
never on period-end dates. Risk (K): an event-risk block inside the
existing `shadow` dict only; `extra_caps` stays `()`. LLM: schema-
validated output, news isolated as untrusted evidence, layered
summarization, token usage added to provider return shape. UI: one
`/catalysts` Nav entry, incremental slices per phase, third
`quant-derived` provenance tier beside the existing two.

**Adjusted phase order (deviations explained in audit §11.0).** B0
naming + market-calendar table → B registry + calendar providers (+ first
Catalysts page) → E1 price context → E2 fundamentals → C previous event /
replay (SEC 8-K timestamps) → D news evidence engine → F earnings
intelligence → I options/implied move → J UI slices throughout → G macro
→ H Fed → K risk SHADOW → L replay validation + §86 measurement.

**Validation.** Baseline suite **2020 passed, 1 skipped** in 70s before
any change (this iteration changes no code). All provider claims are
from live HTTP probes, not documentation.

**Known limitations / NOT BACKTESTABLE.** Historical ATM IV / IV crush /
past implied moves (no historical option quotes at any probed tier; daily-
bar straddle reconstruction is an approximation); historical consensus /
revisions / guidance and therefore EPS/revenue surprise % for any period;
confirmed upcoming earnings dates (only ESTIMATED or user-confirmed);
news and intraday bars before the platform's own ingestion window; 2y
yield until the Treasury CSV adapter exists (SHY is a labelled proxy).
Platform constraints: no migration runner (manual live apply, `IF NOT
EXISTS`), SQLite test harness vs Postgres prod, lifespan not run under
tests, single-process ADR-007 (DB-level idempotency required), no token
accounting in `libs/llm`.

**Subscription dependencies.** Massive Benzinga Earnings add-on would
supply authoritative earnings dates + consensus/surprise/guidance/
revisions (§7, §16, §33-34); nothing else requires purchase. Free
primary sources (BLS/BEA/Fed/SEC/Treasury) are rate-limited/HTML and are
treated as failable providers.

**Production status.** AUDIT (nothing new deployed).

**Autonomous defaults adopted for the audit's open questions (user may
override):** Q1 Benzinga add-on NOT purchased — ESTIMATED dates +
"CONSENSUS DATA UNAVAILABLE"; Q2 third-party calendar adapter seam built
but disabled (empty key); Q3 T-minus-7 alert in-app only via
`ALERT_RULES` + `EVENT_APPROACHING` audit; Q4 display tz
`America/New_York` (reuse existing constant), store UTC + event tz; Q5
ESTIMATED dates shown with badge, excluded from alerts, hideable; Q6 keep
`/api/analysis/{ticker}/catalyst` and coexist.

**Next recommendation.** Phase B0+B: `EventType/EventStatus/EventSession`
enums, migration 021 (`events`, `market_calendar`), ORM + parity
registration, `libs/event_calendar/` registry with Alpaca-calendar /
Massive-holiday / SEC-8K-cadence / Fed-FOMC providers, `event_calendar`
loop, `GET /api/events`, T-7 alert rule, first `/catalysts` page.

## 2026-08-18 (20) — RISK ENGINE UPGRADE, PHASE J + FINAL DELIVERABLE: adversarial validation suite, spec §73 report

**Purpose.** Spec §61 "PHASE J" (simulate the failure modes and verify
fail-safe behaviour), §67 property tests, §68 acceptance, and the §73
final deliverable A–G.

**Implementation.**
- `tests/test_risk_adversarial.py` — 33 tests / 275 assertions, 3.4 s:
  fat-tail crash (hist ES ≥ Gaussian ES, HEAVY_TAIL, gaussian_trust
  reduced, model risk names it); vol spike (EWMA multiplier 1.2 → 0.29
  while the crude proxy is byte-identical — the SHADOW boundary); correlation
  convergence (CONVERGING; ES concentrates; bucket cap binds in shadow —
  single-name cap 268 units on the 3:1:1 tech book, 37.6 % → 34.9 %); IV
  crush / long put + spike; long-gamma convexity + labelled DELTA_LINEAR
  fallback; concentrated tech book (hypothetical RESIZE/REJECT, Tier 0
  unchanged); GARCH failure → EWMA with reason, build succeeds; stale
  snapshot (label flips at TTL, numbers identical, nothing authorized);
  EVT/copula absent by construction (registry = exactly the five built
  models; missing names raise cleanly); broker unreadable → fail-closed,
  missing bars → named exclusion + data_quality.valid False, read view
  still 200; §67 (a) rejected strategy never reaches the broker, (b) RC
  reconciles (ES exact, vol 1e-9), (c) max loss monotone in qty for every
  gateway basis, (d) 40 statistical entry points across 12 modules
  sabotaged → 240 Tier 0 decisions byte-identical, (e) AST-level proof
  that no `assess()` call in apps/ passes `extra_caps` (fires even on an
  empty tuple — an accidental promotion cannot land silently); §68 every
  registered model SHADOW/RESEARCH. Mutation-verified: broker gate
  disabled, model promoted to PRODUCTION, DELTA_LINEAR mislabelled,
  `extra_caps=()` injected — each caught by the intended test; repo files
  restored byte-identically.
- `docs/risk-engine-final-report.md` (§73 A–G, ~8.5k words): architecture
  before/after with the SHADOW layers, the risk model matrix (5 registered
  models + every estimator, Status per §69 vocabulary, Decision Usage),
  implementation report per phase incl. the eleven defects found and
  fixed, deferred models with re-visit triggers, validation report,
  UI changes, remaining model risk + promotion path, and an "Open
  decisions for the user" appendix (audit §11 Q1–Q7, stress_runs
  retention, shadow-gate promotions). Verifier spot-checked ≥ 15 claims
  and corrected three (defect count wording, glossary count 67, the
  `extra_caps` phrasing).

**Model assumptions.** This phase adds no estimator, so it inherits every
assumption of A–E rather than making new ones; what it assumes is about
the TESTS. Each adversarial scenario is built from a DETERMINISTIC seeded
series so a failure is reproducible rather than flaky, and each one
asserts a fail-SAFE direction (degrade, label, honest null) rather than a
specific number, so the suite survives a legitimate re-estimation. The
SHADOW boundary is proven by SABOTAGE — statistical entry points are
monkeypatched to raise and the Tier 0 outputs are compared byte-for-byte
— which assumes the sabotage points are the complete set of seams; that
assumption is itself pinned by the AST-level `extra_caps` scan, which
fails on any `assess()` call in `apps/` that passes the argument at all,
including an accidental empty tuple. Absence of EVT/copula is asserted by
a SOURCE SCAN, so the deferral cannot silently rot into a half-wired
model.

**Data used.** Seeded synthetic series constructed per failure mode
(fat-tail crash, vol spike, correlation convergence, IV crush, long-gamma
convexity, concentrated tech book) rather than live market data — a
failure mode has to be summoned on demand, and a real book that happened
to be calm would test nothing. The live store (5 tickers × ~600 daily
bars, real broker cash) was used only for the post-deploy smoke checks
and the §73 report's measured numbers (13 ms build, first SCHEDULED
snapshot and validation run).

**Validation.** Full suite **2020 passed, 1 skipped**. Program totals:
backend 1032 → 2020 tests; UI 41 → 114; migrations 017–020 live; four
gateway builds deployed today; live snapshots, ATM-IV rows and the first
SCHEDULED validation run accruing.

**Production status.** Tier 0 hardening PRODUCTION; every statistical /
stress / concentration / validation layer SHADOW or RESEARCH; zero
shadow trading days elapsed as of this entry — the promotion clock
starts now.

**Program status.** Spec phases A–E and J delivered per the audit's
adjusted plan; F (allocation benchmarks) DEFERRED pending a weights
harness, G (EVT) DEFERRED until ≥ 1500 obs (deep backfill = user Q2),
H copula REJECTED for now / tail concordance + Spearman = RESEARCH
display candidates. Nothing decides until the user promotes it.

## 2026-08-18 (19) — RISK ENGINE UPGRADE, PHASE E: GARCH(1,1) research + VaR/ES model validation (walk-forward, persisted)

**Purpose.** Spec §12–§14, §42–§43, §57, §59, §63, §68; audit §10 "Phase E".
Two things: (1) respond to volatility clustering with a conditional
volatility model that is VALIDATED before it is trusted (GARCH(1,1) in
RESEARCH mode with diagnostics, EWMA as the §13/§58 fallback); (2)
backtest the risk forecasts themselves — walk-forward, no hindsight —
and persist the exceedance record so promotion decisions have evidence.
Everything SHADOW/RESEARCH; nothing decides.

**Implementation.** Contract §9 in `docs/risk-engine-phase-b-design.md`.
- `libs/trading_core/risk/optim.py` — deterministic Nelder–Mead (stdlib).
  `risk/models/_chi2.py` — regularized incomplete gamma (series /
  continued fraction) → `chi2_sf(x, df)` for any df; df=1/2 closed forms
  agree to ≤ 4e-16.
- `risk/models/garch.py` — GARCH(1,1) Gaussian MLE (unconstrained
  transform enforcing ω>0, α,β≥0, α+β<1; init from EWMA), diagnostics
  (convergence, persistence, half-life, Ljung–Box(m=10) on standardized
  residuals² via `chi2_sf`, ω-at-floor), health (UNAVAILABLE n<250;
  DEGRADED on non-convergence / persistence ≥ 0.999 / LB p<0.05 — all
  parameters), closed-form multi-step variance forecast (horizon σ =
  √Σσ²_{t+k}, labelled GARCH_TERM_STRUCTURE), `garch_scaled_pnl` (FHS),
  `Garch11Model` registered as "garch11" in **RESEARCH** mode, and the
  fallback seam `conditional_volatility_source` /
  `conditional_scaled_pnl_source` → GARCH only when ACTIVE, else EWMA
  with the reason. Recovery on seeded simulations (α=0.08, β=0.90,
  n=3000): |Δα| ≤ 0.016, |Δβ| ≤ 0.023 across four seeds; a fit costs
  ~0.15 s.
- Validation persistence: `migrations/020_risk_model_backtests.sql` +
  `RiskModelBacktestRow` (+ mount, mirror test), APPLIED live.
  `apps/gateway/risk_validation.py::run_model_backtests` — walk-forward
  (window 250, min 60 forecasts) for historical VaR 95/99, Gaussian VaR
  95/99, EWMA-filtered VaR 95, GARCH-filtered VaR 95 (RESEARCH; refit on
  a stride, params reused between — bounded runtime 0.69 s at n=600),
  Kupiec POF + Christoffersen independence + ES severity, verdict
  GREEN/YELLOW/RED; rows persisted; runs once per NY day after the
  SCHEDULED snapshot and on `POST /api/risk/validation/run` (no audit);
  `GET /api/portfolio/risk` `statistical.validation` reads the NEWEST
  PERSISTED rows only (never recomputed on a read) with the EWMA-vs-GARCH
  §63 comparison and the promotion criterion sentence; model-risk rule
  table gains `backtest_red_triggers`. `statistical.conditional_source`
  names which filter is behind the conditional VaR/ES rows (GARCH when its
  fit is ACTIVE, else EWMA + reason).
- UI `ui/components/risk/ModelValidation.tsx`: per-model table (n,
  exceedances vs expected, rate, Kupiec p, Christoffersen p, ES severity,
  GREEN/YELLOW/RED badge, health/reason), RESEARCH badge on GARCH rows,
  comparison + criterion line, "Run now", honest empty state; glossary
  entries.

**Model assumptions.** GARCH(1,1) with a ZERO mean model and GAUSSIAN
innovations, fitted by unconstrained-transform MLE (ω > 0, α, β ≥ 0,
α + β < 1) with a deterministic Nelder–Mead and an EWMA-derived start —
Student-t innovations are deferred, so a fat-tailed innovation shows up
as a diagnostic failure rather than being modelled. The variance
recursion is seeded with the window's own second moment, and every
`sigma_t` uses information strictly BEFORE `t`. Multi-step forecasts are
the closed-form variance term structure, not √h. Validation is
WALK-FORWARD by construction: each forecast sees exactly `window`
observations strictly before the day it forecasts, so no run can see
`pnl[t]` — the property is asserted, not asserted-by-convention. Verdict
bands are the Basel-style Kupiec p-values (0.05 / 0.01), parameters and
not truths. A fit that produces no parameters is structurally forbidden
from carrying any (`__post_init__` enforces the honest null).

**Data used.** The book's own P&L series over `stock_bars_daily` closes
(≈600 observations/ticker); GARCH requires ≥ 250 aligned observations, so
on today's books the live `conditional_source` is EWMA and the GARCH
branch is dormant — the smoke test confirmed exactly that. The
walk-forward harness runs on the same series at a default 250-observation
rolling window, driven by the SCHEDULED tick and by
`POST /api/risk/validation/run`, never by a page read.

**Validation.** Backend **1987 passed, 1 skipped** (+291). Adversarial
verifier (independent probes): NO look-ahead at the strongest level —
mutating only the last observation to −999,999 leaves EVERY earlier
forecast bit-identical across all four estimator families incl. the
stateful GARCH window filter; Kupiec/Christoffersen/ES severity
recomputed from raw forecasts for a full row; GARCH recovery on three
fresh seeds; χ² identities + eight published critical values (≤ 1.5e-3);
runtime 0.32/0.69/1.79 s at n=400/600/900; honest nulls (280 obs → six
UNAVAILABLE rows with `n=30 < min_forecasts=60`); no audit on the
endpoint; migration 020 live. UI tsc clean, 114 component tests. Fixed
before deploy: `comparison.preferred` was rendered raw (model KEY, not a
display word), LR statistics typed required but not served, per-row
`mode` invisible. Deployed; live smoke: `conditional_source` EWMA (GARCH
UNAVAILABLE for today's book length), validation endpoint honest on the
empty book.

**Model limitations.** GARCH needs ≥ 250 aligned observations (most
books today ⇒ EWMA fallback until the deep backfill, audit §11 Q2);
Gaussian innovations only (Student-t deferred); backtests accrue from the
first SCHEDULED run — no history yet; verdict bands are the Basel-style
Kupiec p 0.05/0.01 (parameters).

**Production status.** RESEARCH (GARCH), SHADOW (validation surface).
Promotion of GARCH to SHADOW requires the §63 criterion over ≥ 250
forecast days AND a user action.

**Next recommendation.** Phase J adversarial-suite consolidation and the
spec §73 final deliverable (architecture review, risk model matrix,
implementation report, deferred models, validation report, UI changes,
remaining model risk); F/G/H stay research/deferred per the audit.

## 2026-08-18 (18) — RISK ENGINE UPGRADE, PHASE D: stress engine + option full revaluation (SHADOW)

**Purpose.** Spec §21–§27, §51–§52; audit §10 "Phase D". "What if the
model is wrong?" — historical and hypothetical scenarios revalue the
CURRENT book (options by full Black–Scholes revaluation under S / IV / t
moves, basis-anchored to the real mark), persisted, displayed, and logged
as a hypothetical STRESS cap in the pre-trade shadow verdict. SHADOW:
spec §27 veto authority is the PRODUCTION promotion, not this phase.

**Implementation.** Contract §8 in `docs/risk-engine-phase-b-design.md`.
- `libs/trading_core/options/iv.py` — bisection IV solver on `bs_price`
  (guards: below intrinsic, t ≤ 0, σ = 5 ceiling), labelled INTERNALLY
  CALCULATED. `options/reval.py` — `OptionLeg`/`StockLeg`, `leg_baseline`
  (basis = mark0 − model0, held constant), `reval_leg` (T ≤ 0 ⇒ intrinsic),
  `scenario_pnl` (FULL_REVAL, DELTA_LINEAR fallback when IV is missing —
  labelled; neither ⇒ 0 with a note that loss is understated); zero
  scenario ⇒ EXACTLY 0.0 (P&L differenced against the reconstructed
  baseline, not the raw mark — the algebraically-equal form is not
  bit-exact).
- `risk/models/stress.py` — Scenario / HistoricalWindow (per-ticker
  cumulative simple return from stored closes; IV shock = realized-vol
  ratio proxy `RV(window)/RV(prior 20d) − 1`, clipped, labelled RV_PROXY),
  `auto_worst_windows` (worst 1/5/10-day of the equal-weight book), named
  windows 2024-08-05 / 2025-04 (UNAVAILABLE rows with real dates when
  outside stored history), hypothetical grid (Equity −5 %/IV +20 %, −10 %/
  +40 %, +5 %/−15 %, IV crush −40 %, IV spike +50 %, correlation
  convergence −8 %/+30 %, time decay +5d — ALL `validated=False`,
  UNVALIDATED research grid), `run_stress` (never raises; run health =
  worst among PRICED rows — a named window outside history is an
  UNAVAILABLE row named in `reason`, not a run downgrade — revised after
  QA), `StressLimits(max_stress_loss_pct_nav=0.10 research default)`,
  `stress_caps` → `QuantityCap("STRESS_LOSS_LIMIT", layer STRESS)` via the
  Phase C bisection helper (fail-open on UNAVAILABLE, same open item).
- Persistence: `migrations/019_stress_runs.sql` + `StressRunRow` + compose
  mount + mirror test; APPLIED live.
- Gateway: `risk_snapshot.py` builds legs from the same chain resolution
  the view uses (mid → mark0, provider IV → iv0, DTE/365, signed
  quantities), runs the catalogue per build, persists rows, API
  `statistical.stress` {rows (pnl/loss, method_coverage, health, reason,
  params), worst, health, catalogue_version d.1, n_stock/option_legs,
  positions_excluded}; `orders.py` `shadow.statistical.stress`
  {worst_before, worst_after, cap, hypothetical} merged into the shadow
  verdict/binding list + a `worst_stress_loss` comparison row; NEW
  `POST /api/risk/stress/run` (user-defined equity/IV/days scenario,
  validated ranges → 422; persists a USER row; NO audit event); positions
  option rows gain premium_at_risk / dte / iv0 / worst_scenario_pnl /
  worst_scenario_name. **ON_DEMAND builds now persist at most once per 15
  minutes** (`statistical.persisted`) — the UI polls the risk view every
  15 s and would otherwise have written a snapshot + metrics +
  contributions + ~12 stress rows per poll (found by QA before it happened
  live).
- UI: `ui/components/risk/StressScenarios.tsx` (table: kind badge,
  UNVALIDATED badge on the research grid, P&L $ / % NAV, method coverage,
  health/reason, worst row highlighted; user-scenario form with the
  server's ranges and inline errors), positions option rows (premium at
  risk, DTE, IV incl. internally-calculated label, worst scenario loss),
  TradeComparison "Worst stress loss" row + STRESS constraint group,
  glossary (stress_test, full_revaluation, iv_crush, basis_adjustment).

**Model assumptions.** European Black–Scholes with `r = 0.04`, `q = 0.0`,
both stated constants rather than measured. A leg is repriced by moving
S, IV and t TOGETHER and differencing two model prices on a CONSTANT
basis (`basis = mark0 − model0`), so the zero scenario is bit-exactly
`0.0` and the model-vs-mark gap cancels instead of leaking into the P&L.
`T ≤ 0` collapses to intrinsic. A leg whose chain gave no IV falls back
to DELTA_LINEAR and SAYS so per key; a leg with neither IV nor delta
contributes 0 with a note that the loss is understated. Historical
windows use per-ticker cumulative SIMPLE returns from stored closes (so
they are not β = 1), while hypothetical and user scenarios are uniform
β = 1 across underlyings. Historical IV shocks are a REALIZED-vol ratio
proxy (`RV(window)/RV(prior 20d) − 1`, clipped), labelled `RV_PROXY` at
every level — no IV shock in this phase is calibrated from IV history.
The whole hypothetical grid carries `validated=False`.

**Data used.** `stock_bars_daily` closes for the scenario catalogue (the
SAME bars the return matrix was built on, so a window's shock is the real
cumulative return over it, and a window outside the stored history is an
UNAVAILABLE row with its real dates rather than a guess); today's option
chain for strike / right / DTE / IV / mark, resolved ONCE per ticker and
shared with the greeks panel so a stress reprice and the greeks read can
never be anchored to different contracts; broker live cash for NAV.

**Validation.** Backend **1696 passed, 1 skipped** (+189: IV round trip
grid |Δσ| < 1e-6, zero-scenario bit-exact, expiry intrinsic, spread/income
signs, stock linear, historical shocks from a hand-built path, auto worst
windows brute-forced, stress cap vs brute force, |P&L| monotone in |q|,
API stress keys, persistence rows, USER endpoint 422/no-audit/one row,
Tier 0 byte-identical when the stress layer raises, ON_DEMAND dedupe). UI
tsc clean, 89 component tests, native-dialog check ok. The final
adversarial verifier could not run (API 529 overload, twice); orchestrator
verified directly: migration 019 live, `assess()` called without
`extra_caps`, stdlib-only, dedupe live (`persisted` True→False on
re-read), USER run persisted (run_id 8), UI /risk renders.

**Model limitations.** Historical IV shocks are RV proxies (no IV history
until `atm_iv_daily` accumulates); named windows before 2024-03 are
UNAVAILABLE without the deep backfill (audit §11 Q2, still awaiting the
user); the hypothetical grid is unvalidated; β=1 uniform equity shocks;
VaR/ES remain DELTA_LINEAR (full reval is used for stress only).
`stress_runs` retention beyond the dedupe is an open policy item.

**Production status.** SHADOW. Deployed (backend; UI on the dev server).

**Next recommendation.** Phase E — EWMA is already the side-by-side
forecast; run the walk-forward VaR/ES exceedance backtest (Kupiec /
Christoffersen) on the stored history for the current book and persist
results (spec §42), GARCH(1,1) as RESEARCH with diagnostics and the §63
comparison vs EWMA; Phase J adversarial suite consolidation.

## 2026-08-18 (17) — RISK ENGINE UPGRADE, PHASE C: pre-trade portfolio risk (SHADOW) — comparison, incremental/marginal ES, hypothetical caps, correlation regime

**Purpose.** Spec §8/§9/§11/§14/§19/§37/§38/§46/§47/§70; audit §10
"Phase C". Answer "does this trade improve or damage portfolio
diversification, and how large can it safely be?" — computed and shown
before every trade, logged as a hypothetical verdict, **without changing
a single Tier 0 decision** (SHADOW; promotion is an explicit user step,
audit §11 Q3).

**Risk hypothesis.** A small-premium trade can pass every dollar cap and
still lift portfolio ES-95 or concentrate ES in one name/bucket;
incremental ES on the JOINED series (same k, same window) and Euler ES
shares are the decision-grade measure the dollar caps lack.

**Implementation.**
- Design contract §7 in `docs/risk-engine-phase-b-design.md`.
- `libs/trading_core/risk/pretrade.py`: `CandidateSpec` (per-unit delta /
  risk basis / capital basis), `proposed_book` (current + candidate series
  on the same ReturnMatrix), `compare()` → `RiskComparison` (heat, cash,
  hist VaR/ES 95/99, Gaussian ES 95, σ before/after as MetricPairs;
  incremental ES-95 = ES(after) − ES(before) EXACTLY; marginal ES per
  unit = candidate Euler RC ÷ q; candidate ES share; max single-name ES
  share before/after; bucket ES shares; net delta notional),
  `StatisticalLimits` (RESEARCH DEFAULTS, UNVALIDATED: portfolio ES-95 ≤ 5 %
  NAV, single position ≤ 35 % of ES contributions, bucket ≤ 50 %,
  incremental ES-95 ≤ 1.5 % NAV, min_obs 60, mode SHADOW),
  `statistical_caps()` (bisection ≤ 20 steps for the largest passing
  quantity; step-down guard never returns an unverified quantity;
  UNAVAILABLE ⇒ NO caps — fail-open by design while SHADOW),
  `shadow_verdict()` (hypothetical APPROVE / RESIZE / REJECT at the Tier 0
  approved qty, binding codes most-restrictive first).
- `risk/engine.py` — additive: `assess(..., extra_caps=())` applied as
  clamps after the cash floor and before greeks (structural `ExtraCap`
  Protocol — engine never imports the statistical library);
  `RiskAssessment.requested_quantity` + `binding_constraints`
  (BindingConstraint(code, layer); total mapping, HARD_LIMIT for every
  Tier 0 code) populated for every assess/assess_income decision.
  Byte-identity proven three ways: a 240-case seeded battery in-suite; C-1's
  reconstruction of the pre-Phase-C engine compared on all ten output
  fields; QA's independent strip-and-diff (pure additions).
- `libs/trading_core/correlation.py` — additive `correlation_regime()`:
  normal (250d) vs current (60d) vs stress-conditioned (worst-10 % days of
  the equal-weight book) average pairwise Pearson, worst pairs, state
  NORMAL / ELEVATED / CONVERGING. **Rule revised after QA:** CONVERGING iff
  the CURRENT average ≥ `converging_level` (0.80) — persistent or sudden;
  the earlier "jump AND level" rule read a book that always moved at ρ≈1 as
  NORMAL, the exact §19 failure mode. The jump ≥ `converging_delta` now
  annotates `reason` ("regime shift: normal 0.61 → current 0.84").
- Gateway (`orders.py` RISK_APPROVAL, surgical): CandidateSpec from the
  chosen instrument (stock ±1 delta; options/spreads signed per-share net
  delta; risk basis = the same risk_stop Tier 0 used) → compare at the
  approved qty (and requested when different) → caps → shadow verdict →
  correlation regime on the LOG-return matrix of book + candidate.
  Logged under `shadow.statistical` {comparison (rows + tier0_rows), caps,
  hypothetical, limits, correlation_state}; mirrored into the response
  `risk.comparison` / `risk.binding_constraints` / `risk.shadow_statistical`
  (so trade plans store it verbatim). `extra_caps` is NOT passed to
  `assess()` anywhere in apps/ (SHADOW). Vol targeting side-by-side:
  `vol_targeting.ewma_sigma_p_annualized_pct_nav` + `multiplier_ewma`
  logged (`shadow.vol_targeting_ewma`) while the crude proxy stays the one
  used (proven by sweeping σ over 5 orders of magnitude: applied multiplier
  pinned, EWMA multiplier moved). `/api/portfolio/risk`
  `statistical.correlation_state`.
- UI `ui/components/risk/TradeComparison.tsx`: "CURRENT vs AFTER TRADE"
  (spec §46) — Tier 0 rows first (heat, cash), then VaR/ES/σ rows, then
  the concentration rows (incremental ES-95, this position's ES share,
  largest single-name ES share, bucket ES shares, net delta notional);
  requested vs approved vs hypothetical statistical quantity; binding
  constraints grouped HARD_LIMIT → STATISTICAL/CONCENTRATION with cap
  sentences (spec §47); SHADOW badge from the SERVER's mode; correlation
  pill + EWMA-beside-proxy on the risk page; glossary entries; bilingual.
  QA caught three UI↔gateway drifts (row field names, separate
  `tier0_rows`, `caps` as an object) that had rendered the whole table as
  dashes — fixed, fixtures replaced with a real gateway payload,
  regression-tested BEFORE deploy.

**Model assumptions.** The proposed book is the current book's per-position
P&L series PLUS the candidate's, joined on the SAME `ReturnMatrix` dates —
so incremental ES is `ES(after) − ES(before)` at the same `n` and the same
tail `k`, never two differently-sampled estimates differenced. Option
candidates are DELTA_LINEAR until Phase D (labelled). The Euler ES
decomposition is what makes `Σ RC = ES` exact and therefore makes
`marginal × qty == contribution` coherent. Caps are found by bisection on
an INTEGER quantity (≤ 20 steps) with a step-down guard, so a reported cap
is always a verified passing quantity, never an interpolated one. Every
`StatisticalLimits` threshold (5 % / 35 % / 50 % / 1.5 % NAV, min_obs 60)
is a RESEARCH DEFAULT, explicitly UNVALIDATED. Correlation is Pearson on
LOG returns over the book's own dates, with the candidate's ticker
included so the state is the one that would exist AFTER the trade.

**Data used.** `stock_bars_daily` closes (≈600/ticker) via the SAME
`RiskSnapshotBuild` the Tier 0 decision was measured against — never a
rebuilt book that could silently disagree; today's chain delta for the
selected contract (the candidate's per-unit delta, never guessed — a
missing delta raises and becomes an honest note); broker live cash and
NAV; Tier 0's own `heat_before`/`heat_after` and `cash_after`, read from
the assessment rather than recomputed.

**Validation.** Backend **1507 passed, 1 skipped** (+87); UI tsc clean,
62 component tests, native-dialog check ok. QA (independent probes):
SHADOW proof with compare/statistical_caps/shadow_verdict monkeypatched to
raise → decision, approved qty, gates, reason codes, binding constraints,
budget multiplier identical; incremental ES exact to the float; marginal ×
qty == candidate contribution (diff 0.0); cap bisection brute-forced over
every integer quantity in 4 constructed cases (reported cap == true max);
correlation regime hand-checked from the textbook Pearson formula (≤ 3e-17);
plans store the new keys; read views write no risk audit event. Live after
deploy: full chain runs (gate 7 now measures a streamed 0.013 % spread),
`atm_iv_daily` accumulating (AAPL 0.2448), first SCHEDULED snapshot written
14:59 UTC — drawdown history has begun.

**Model limitations / open items.** Fail-open on UNAVAILABLE statistical
views (safe in SHADOW; the PRODUCTION promotion must choose the
fail-closed rule); a bucket limit already breached by non-candidate members
resolves to cap 0 (UI wording notes it); StatisticalLimits and correlation
params are research defaults; DELTA_LINEAR candidates for options until
Phase D full revaluation.

**Production status.** SHADOW throughout (hypothetical verdicts logged in
every RISK_DECISION; nothing decides). Deployed.

**Next recommendation.** Phase D — stress engine (historical windows
2024-08-05 / 2025-04 available now, hypothetical scenario table, IV grid
labelled unvalidated), option full revaluation (bisection IV solver on
`bs_price`, basis-anchored leg-aware reprice, replacing DELTA_LINEAR where
greeks/IV exist), stress-loss hypothetical cap in the same shadow block,
migration 019 stress_runs, stress table + per-instrument rows in the UI.

## 2026-08-18 (16) — RISK ENGINE UPGRADE, PHASE B: core statistical risk layer, SHADOW (library + persistence + API + UI)

**Purpose.** Spec Phase B (`prompts/risk_engine.md` §3–§10, §15,
§39–§45, §55–§56, §65, §70; audit §10 "Phase B"): give the platform its
first statistical view of the book — returns layer, model registry,
Historical/Gaussian VaR & ES, portfolio σ + EWMA, Euler risk
contributions, distribution diagnostics, ensemble dispersion, model
health / model-risk state, NAV drawdown, walk-forward VaR validation —
persisted, exposed with methodology labels, and rendered. Everything is
**SHADOW**: no Tier 0 decision changes (proven, see Validation).

**Risk hypothesis.** Dollar caps cannot see correlated tail risk; a typed,
reproducible VaR/ES/contribution view (with honest health) is the
prerequisite for Phase C's incremental-ES sizing and the §11 RC gate.

**Design contract.** `docs/risk-engine-phase-b-design.md` — ONE quantile
convention platform-wide: losses L=−pnl sorted desc, k = ceil(n(1−α)),
Historical VaR = k-th largest loss, ES = mean of the k largest ⇒ ES ≥ VaR
and Euler ES contributions (tail-day averages) sum EXACTLY to ES;
Gaussian VaR = −μ+zσ, ES = −μ+σφ(z)/(1−α) (ddof=1, `NormalDist`);
SIMPLE returns for P&L, LOG for correlation (existing code byte-identical:
`correlation.log_returns is risk.returns.log_returns`); min_obs 60 @95 /
250 @99; χ² p-values in closed form (χ²₁ = erfc(√(LR/2)), χ²₂ = e^{−x/2})
so no incomplete-gamma; stdlib only.

**Implementation.**
- Library `libs/trading_core/risk/`: `returns.py` (ReturnSeries/Matrix,
  inner-join on return dates, never compound across gaps), `pnl_series.py`
  (PositionRiskInput, DELTA_LINEAR book P&L; excluded positions named),
  `models/base.py` (ModelHealth/ModelMode/ModelMeta/ModelResult, registry,
  validate-never-upgrades), `models/var_es.py` (+ conditional vol-scaled
  VaR/ES, canonical `tail_size`), `models/volatility.py` (sample Σ,
  portfolio σ, EWMA λ=0.94, filtered-HS scaling), `models/contribution.py`
  (vol RC = cov/σ_p; Euler ES RC; marginal/incremental ES),
  `models/diagnostics.py` (skew/kurtosis/JB → NORMAL_LIKE / HEAVY_TAIL /
  LEFT_SKEWED / UNSTABLE + gaussian_trust), `models/ensemble.py`
  (dispersion ratio, MODEL_DISPERSION_HIGH, model-risk LOW/ELEVATED/HIGH
  rule table with replayable triggers), `models/drawdown.py` (NAV drawdown
  + RECONSTRUCTED_CURRENT_BOOK), `validation.py` (walk_forward, Kupiec
  POF, Christoffersen, ES severity, QLIKE), `snapshot.py`
  (PortfolioRiskSnapshot, DataQuality, TtlPolicy, version b.1). 76 names
  exported; cycle-free both import orders. Integrator caught and removed a
  SECOND `tail_size` implementation that agreed on the 95/99 grid but
  diverged off-grid (34/20,000 (n,α) probes) — one canonical function,
  mutation-verified.
- Persistence: `migrations/018_risk_snapshots.sql` (risk_snapshots,
  risk_metrics with the full §44 model identity INLINE — the audit's
  `risk_model_runs` folded in —, risk_contributions, atm_iv_daily) + ORM +
  compose mount + mechanical ORM↔SQL column-mirror test; APPLIED live.
- Gateway `apps/gateway/risk_snapshot.py`: `build_risk_snapshot()` (book
  from the SAME helpers as the risk view; deltas recovered from
  `portfolio_greeks_read` rows so shorts/income/spreads carry the sign
  exactly as §16 does; models under try/except → FAILED health, never a
  5xx; persist = add+flush, CALLER commits), `run_scheduled_snapshot()` +
  `risk_snapshot_loop` (setting `risk_snapshot_interval_seconds`=1800, 0
  off, off under tests; ONE SCHEDULED row per NY day = the live NAV series
  drawdown is measured on), `record_atm_iv()` upsert (VOLATILITY gate +
  chain endpoint, best-effort), telemetry (`risk_snapshot_age_seconds`
  = age of the newest **SCHEDULED** build only, `risk_model_latency_seconds{stage}`,
  `risk_snapshot_builds_total{trigger}`, `risk_snapshot_failures_total`).
- API: `GET /api/portfolio/risk` gains additive `statistical` (mode SHADOW,
  snapshot_id, as_of, stale, pnl_method, n_obs/window, data_quality,
  model_health, model_risk, dispersion, distribution, volatility, var[],
  es[] — every row with model/model_name/model_version/distribution/
  confidence/horizon/value/pct_nav/health/reason/sample_size/tail_size —
  contributions {es, vol} with capital_weight vs share, positions_excluded)
  and `drawdown` (live SCHEDULED-NAV series + reconstructed). Read view
  writes NO audit event (probe-verified). RISK_DECISION audit details gain
  quantity_requested / approved_quantity / budget_multiplier / limits /
  shadow.statistical (current-book headline numbers; proposed-book
  comparison is Phase C).
- UI (`ui/components/risk/StatisticalRisk.tsx`, risk page, types,
  glossary, guide): methodology-labelled tiles ("Historical VaR 95% 1D",
  "Historical ES 95% 1D", "Gaussian VaR 95% 1D", σ, drawdown, model risk)
  with health + n + "ⓘ How is this calculated?" `RiskMethodModal`
  (model, confidence, horizon, lookback, distribution, as_of, data source,
  health, version, Advanced diagnostics); "Statistical Risk (SHADOW)"
  panel (all VaR/ES rows, model disagreement, distribution line, model-risk
  reasons); "Risk Contribution" panel (capital weight vs ES-95 risk share
  as two bars on one axis + totals reconciling); drawdown block with
  honest empty state; bilingual; glossary entries.

**Model assumptions.** Options are DELTA-LINEAR in the P&L series until
Phase D full revaluation (labelled `pnl_method`); today's book under
historical 1-day simple returns; 1D estimated only (√h labelled);
Gaussian views are ensemble members / trust checks, never favoured.

**Data used.** stock_bars_daily closes (≈600/ticker), today's chain deltas
(§16 helper), broker live cash; ATM IV now accumulates daily.

**Validation.** Cross-module invariants pinned
(`tests/test_risk_phase_b_invariants.py`): ES ≥ VaR, monotone in α, ES RC
sum == ES exactly, vol RC sum == σ_p (1e-9), scale/shift laws, walk-forward
never sees pnl[t], min_obs → None/UNAVAILABLE without exceptions,
filtered-HS λ→1 ≈ HS, registry SHADOW modes, end-to-end 3-ticker pipeline.
Adversarial verification (independent probes): Tier 0 byte-identical when
the builder is monkeypatched to raise (only shadow.statistical differs);
GATE_ORDER unchanged; API key sets == §6 exactly on empty and seeded
books; persistence rows carry model_name/version/params; contribution
sums exact; honest nulls at n<60 with reasons; one SCHEDULED row per NY
day; build latency 13 ms on 5×600 (75× inside budget). UI: tsc clean,
43 component tests, next build ok, native-dialog check ok; QA caught two
crash paths (unguarded `.toFixed()` on nullable distribution fields when
the series is UNSTABLE; nullable `model_risk`) — fixed + typed +
regression-tested BEFORE deploy. Post-deploy: live snapshot #2, empty
book → model risk LOW "no open positions — nothing to model" (empty-book
special case added; the ELEVATED rule-table verdict was misleading there).
Full suite **1420 passed, 1 skipped**; deployed; scheduled loop running.

**Model limitations.** ~600 obs ⇒ 99% tails average ~6 points (DEGRADED
label, never gated); DELTA_LINEAR understates long-gamma/vega tails
(Phase D); drawdown history starts today; ATM IV history starts today;
no VIX; thresholds for any future gate remain unvalidated (Phase C shadow
window, audit §11 Q3).

**Production status.** Library + persistence + API + UI = **SHADOW**
(RESEARCH for conditional views). Nothing decides.

**Next recommendation.** Phase C — pre-trade current-vs-proposed
comparison (incremental/marginal ES on the joined series), RC
concentration + ES-limit hypothetical verdicts logged in
`shadow.statistical`, `binding_constraints` explainability, Trade Plan
"CURRENT vs AFTER TRADE" table, EWMA σ_p as the §14 forecast side by side
with the crude proxy (SHADOW), correlation regime state.

## 2026-08-17 (15) — RISK ENGINE UPGRADE, PHASE B0: Tier 0 hardening (income through Tier 0, liquidity REPORT gate, migration 017, bear-spread replay)

**Purpose.** Close the Tier 0 gaps the Phase A audit found BEFORE any
statistical layer exists (spec §2/§72: hard limits are the foundation;
audit §8 items 1–6, §10 "Phase B0").

**Risk hypothesis.** A trade path that bypasses `assess()`/pool/audit,
or a DB CHECK narrower than the ledger vocabulary, is a policy hole no
statistical model can compensate for; and a shown-but-unenforced
liquidity gate implies protection that does not exist.

**Existing capability discovered / verified live.** `orders_side_check`
on the live Postgres allowed only BUY_TO_OPEN/SELL_TO_CLOSE while the
code writes SELL_TO_OPEN/BUY_TO_CLOSE (income shorts, margin short
stock, covers) — live `orders` was empty, so the first real such fill
would have failed on INSERT. Migrations 013–016 had been hand-applied
but never mounted in compose. Income opens ran permission → kill switch
→ collateral law → selection → fill with no risk assessment, no pool
gate, no RISK_DECISION. BEAR_PUT_SPREAD replay called the BULL entry
evaluator with inverted geometry (never entered; API test was vacuous).

**Implementation.**
- `migrations/017_orders_side_vocabulary.sql` (CHECK = exactly
  `libs.broker.provider.MLEG_LEG_SIDES`), compose mounts 013–017,
  ORM docstring; APPLIED to the live volume (verified
  `pg_get_constraintdef`). `tests/test_migration_parity.py` pins the
  CHECK list to the code constant, every migration to a compose mount,
  contiguous numbering, and 005 ⊂ 017.
- `libs/trading_core/risk/engine.py`: **append-only** `IncomeRiskRequest`
  + `assess_income()` — kill switch → heat gate → base qty = contracts
  → ABS_TRADE_RISK_CAP (1.5 % NAV, no tier/edge for income) → single-name
  risk (risk basis) → single-name capital (capital basis) → bucket → heat
  headroom (strictly <) → cash floor (capital basis = the reservation) →
  §16 greek limits at the approved qty (short leg negated by the caller)
  → APPROVE / APPROVE_WITH_RESIZE / REJECT with the same reason-code
  vocabulary and real-number sentences. CSP bases: risk (strike −
  expected credit) × 100 where expected credit = mid × (1 − paper
  slippage) so assessed heat ≥ booked heat; capital strike × 100. CC
  bases 0/0 (the stock row already carries heat; only kill switch, heat
  gate and greeks can bind). `assess()` byte-identical (md5 of the first
  607 lines equal to baseline; 45 original engine tests untouched).
- NEW `apps/gateway/risk_inputs.py::build_portfolio_snapshot()` — the ONE
  builder of the Tier 0 `PortfolioSnapshot` for BOTH write paths
  (orders.py re-pointed; income.py uses it). Deployable cash = account
  cash − Σ open CSP `cash_reserved` (snapshot.cash for the §13 floor);
  NAV stays un-netted (pledged collateral is still an asset; identical to
  the risk view's NAV). Risk view gains additive `cash_reserved_usd`.
- `income.py`: kill switch → **Trading Pool authorization** (gate 1
  semantics; VETOED RISK_DECISION then 422) → collateral law → live
  selection → **risk gate** (broker LIVE cash fail-closed / simulator
  ledger → snapshot → book greeks + negated short leg → `assess_income`;
  REJECT → commit RISK_DECISION + 422 with reason codes; RESIZE → opens
  the approved contracts, CSP reservation recomputed) → fill; exactly ONE
  SYSTEM RISK_DECISION (entity `income_open`) in the fill's transaction.
- LIQUIDITY gate (underlying) in **REPORT mode**: NEW pure
  `libs/trading_core/risk/liquidity.py` (`LiquidityLimits` research
  defaults ADV20 ≥ 100k sh, order ≤ 1 % ADV20, spread ≤ 0.5 % — documented
  UNVALIDATED; `evaluate_underlying_liquidity` → PASS / WOULD_FAIL /
  UNAVAILABLE + `partial` flag + audit-exact reasons). Gate 7 now PASSes
  with measured values and the hypothetical verdict (never vetoes; option
  candidates: contracts not translated to shares; spread only from the
  fresh in-process NBBO stream cache — no new provider call); RISK_DECISION
  details gain `shadow.liquidity` (+ `at_approved_quantity`). Pool
  readiness LIQUIDITY check reports the same numbers. Option-LEG liquidity
  (§9 OI/spread filters) unchanged. GATE_ORDER unchanged.
- BEAR_PUT_SPREAD replay: `_evaluate_entry_bear` + put-vertical geometry
  (short < long); bull branch pinned unchanged (11 trades / −531.2 /
  94,322.8 equity fixture); PLTR API run goes 0 → 14 trades.

**Model assumptions.** None statistical. Income bases are conservative
estimates (broker fills unknowable ahead). Alpaca cash is not reduced by
CSP collateral (options BP absorbs it) — netting once in the builder is
therefore correct; verify on the first live CSP.

**Data used.** stock_bars_daily volume (ADV20), streamed NBBO when fresh,
today's chain mid for CSP credit.

**Validation.** Three implementers + adversarial verifier (diffed the
whole change set against a pristine snapshot; probes: contracts 0 /
huge / NaN NAV, heat exactly at threshold, greeks None, empty/None/NaN
volumes, crossed quotes). 7 MINOR findings, all fixed in this entry
(NAV un-netted vs deployable cash; cash-detail wording names the pledged
amount; None volumes → unmeasurable not TypeError; NaN ADV floor
rejected; `partial` flag for partial-measurement PASS; CSP basis at
expected fill; `contracts` must be a real int) and pinned by tests.
Full suite **1089 passed, 1 skipped** (baseline 1032).

**Model limitations.** Liquidity thresholds unvalidated (watchlist has
illiquid small caps) — REPORT only until the audit-§11-Q3 promotion
window; a PASS with `partial=True` must fail closed once promoted.

**Production status.** Tier 0 hardening = PRODUCTION (deterministic,
additive). Liquidity gate = SHADOW (REPORT). Policy notes for the user:
(1) covered calls are refused when book heat ≥ 8 % even though their
risk basis is 0 (risk-reducing overwrites blocked by the book gate) —
confirm or relax; (2) one CSP needs NAV ≥ ~67× (strike − credit) × 100
under the 1.5 % per-trade ceiling ($72.5 strike ⇒ ≈ $480k) — intended
Tier 0 arithmetic, but it makes CSPs on a $100k paper book REJECT.

**Next recommendation.** Phase B pure library (returns layer, model
registry, Historical/Gaussian VaR & ES, portfolio σ + EWMA, Euler ES /
vol contributions, distribution diagnostics, dispersion + model-risk
state, drawdown, walk-forward Kupiec/Christoffersen validation, typed
`PortfolioRiskSnapshot`) per `docs/risk-engine-phase-b-design.md`
(in flight), then persistence + gateway + UI.

## 2026-08-17 (14) — RISK ENGINE UPGRADE, PHASE A: institutional risk audit (no code)

**Purpose.** Start of the Institutional Risk Engine Upgrade program
(`prompts/risk_engine.md`, 74 sections, phases A–J). Spec §1/§61 forbid
implementing before inspecting, so this iteration is audit-only: 12
parallel read-only inspections (risk engine, allocation/vol/correlation,
portfolio/heat/greeks, backtest, market data, broker/permissions,
DB/audit, UI, docs/ADRs, gate chain/plans, tests/CI, option pricing) →
synthesis → adversarial critique (REVISE: a test count, drifted line
anchors, over-stated liquidity gap) → revision. Output:
**`docs/risk-engine-audit.md`** (executive summary, current architecture
vs §72 target, Tier 0 table, existing statistical logic, data
inventory, instrument model, ~60-row gap matrix with §62 decisions,
tech debt, house rules, adjusted phase plan, open questions).

**Risk hypothesis.** The platform has complete deterministic hard limits
but NO statistical view of the book (no returns layer, VaR/ES, portfolio
σ, NAV drawdown, stress, risk contribution, model health, persistence);
a small-premium trade can therefore pass every dollar cap while
materially raising portfolio tail risk through correlation.

**Existing capability discovered.** Tier 0 is real and tested (45
engine tests incl. seeded invariants): kill switch → heat gate → tier
budget (abs cap 1.5% NAV) → stop sizing → single-name risk/capital,
static TECH_MEGA bucket, heat headroom, regime cash floor → §16 greek
limits (REJECT-only). Estimation-like logic present: RV20 (log, √252),
rolling-60d Pearson buckets (display only), crude NAV-weighted vol
proxy for §14, IV regime, BS pricer (no IV solver), backtest
Sharpe/drawdown on equity. **Three Tier 0 correctness gaps found and
LIVE-VERIFIED against the Postgres volume:** (1) `orders_side_check`
still allows only BUY_TO_OPEN/SELL_TO_CLOSE while code inserts
SELL_TO_OPEN/BUY_TO_CLOSE (short stock, income, covers) — first real
such fill would fail on INSERT; live `orders` is empty so not yet hit;
(2) covered-call/CSP opens bypass `assess()`, the Trading Pool gate and
the RISK_DECISION audit; (3) dynamic correlation buckets are shown but
never enforced. Also: compose mounts only 001–012 (013–016 were
hand-applied to the live volume — verified columns present),
BEAR_PUT_SPREAD backtest never enters (bull entry evaluator + inverted
strike check), underlying (stock) liquidity gate permanently SKIPPED
(option-leg liquidity IS enforced in §9 filters).

**Data used.** Live store: 5 tickers × ~600 daily bars (2024-03-20 →
2026-08-14); no IV history, no VIX/SPX, current-day chain only; expired
option daily bars fetchable from ~Feb 2024. Usable historical stress
windows today: 2024-08-05, 2025-04. 99% ES on 600 obs averages ~6
points → gates will use ES 95% first, 99% displayed with sample size.

**Model assumptions / decisions.** Numerics stay **stdlib-only** (house
rule); numpy only under a documented triple condition via
pyproject+Dockerfile+ADR. Decisions (§62 six questions applied):
IMPLEMENT NOW (all SHADOW) — returns layer, model registry, typed
PortfolioRiskSnapshot, Historical/Gaussian VaR & ES 95/99 1D, portfolio
σ, persisted NAV drawdown, vol & Euler-ES risk contribution,
distribution diagnostics, ensemble dispersion, model health / model-risk
state, VaR-exceedance backtest framework, migration 018 risk tables +
atm_iv_daily, audit widening, methodology-labelled UI tiles.
SHADOW→PRODUCTION (C/D) — current-vs-proposed comparison,
incremental/marginal ES, RC concentration gate, ES limit, sizing v2
modifiers, stress engine + option full revaluation (new IV solver),
stress-loss gate. EWMA pulled forward as the §14 forecast; GARCH(1,1)
research-only (E). DEFER — EVT (≥1500 obs), empirical IV shocks (≥120
ATM-IV days), multi-day empirical horizons, Monte Carlo, turnover,
GMV/ERC (need a weights harness). REJECT for now — copula/GARCH-copula,
MVO/tangency, multi-factor RC.

**Validation.** Critic spot-checked >30 file:line citations; all
material claims held. Baseline suite **1032 passed, 1 skipped** before
any change (this iteration changes no code).

**Model limitations.** Everything statistical will be labelled by
method/confidence/horizon/sample size; nothing gates in production
until the SHADOW window and an explicit user promotion.

**Production status.** AUDIT (nothing new deployed).

**Autonomous defaults adopted for the audit's open questions (user may
override):** Q1 permissions neither broadened nor reverted; Q2 deep
backfill NOT run pending approval; Q3 SHADOW ≥20 trading days, promotion
= explicit user runtime-config action; Q4 backtest replay of `assess()`
deferred pending go/no-go; Q5 greek limits stay REJECT (byte-identical);
Q6 walk-forward exceedance backtest = validation, not IS/OOS parameter
search; Q7 book-level data invalidity marks the snapshot INVALID and is
SHADOW-logged — no automatic kill switch without user policy.

**Next recommendation.** Phase B0 (Tier 0 hardening: migration 017 +
compose mounts, income opens through `assess_income` + pool gate +
audit, underlying liquidity gate in REPORT mode, BEAR_PUT_SPREAD replay
fix), then Phase B (core risk metrics, SHADOW).

## 2026-08-17 (13) — PHASE 3 COMPLETE: margin-backed SHORT STOCK — ALL PERMISSIONS OPERABLE

**The program's final unlock.** short_stock and margin left the forbidden
sets — real flags, default False, runtime-config togglable
(allow_short_stock / allow_margin), Settings UI toggles. ONLY the naked
shorts remain locked, forever (broker refusal + §4 charter). Scope
decision (industry standard): margin exists to SUPPORT SHORTING — the
broker enforces buying power/maintenance; levered LONG sizing stays off
(§12 sizes from cash, never buying power).

**The chain, end to end:**
- §8 matrix: SHORT_STOCK only in the two dead-end bear cells where
  premium is unbuyable (STRONG/EXTREME; MODERATE/HIGH without spreads),
  gated on BOTH flags; puts stay the preferred bear expression; spreads
  outrank shorting where available.
- SHARED exits (§21): the BEAR stock refusal became the mirrored hard
  stop — stop ABOVE entry, breached on close >= entry + stop_distance;
  same engine, backtest and live identical.
- §12.1 risk: stop-based risk × SHORT_STOCK_GAP_RISK_FACTOR (2.0) — an
  overnight gap can blow through any stop, so sizing halves; §16 delta
  −1/share; heat carries the gap-inflated number.
- Adapter mirror gates: submit_stock_short_order is STOCK-gated (an OCC
  option symbol raises "naked short options do not exist... and never
  will") + mandatory margin_attested_by; submit_stock_cover_order
  dedicated BUY_TO_CLOSE. submit_order keeps its exact two-word §5
  vocabulary; the adversarial suite still passes untouched.
- Gateway: simulated SELL_TO_OPEN credits proceeds (liability marked
  daily at −qty×close, NAV moves by exactly the short's P&L); broker
  path T1/T2 via submit_stock_short_and_poll; cover BUY_TO_CLOSE with
  mirrored realized P&L, allowed under the kill switch (§18); in-flight
  guard covers both opening sides; §18 reconciliation claims the ticker
  at NEGATIVE quantity.
- Backtest: run_short_stock_backtest — bear-mirror entries
  (_evaluate_entry_bear), shared mirrored exits, equity = cash −
  qty×close; hand-checked cash mechanics + mirrored-hard-stop tests.

**LATENT BUG KILLED — §16 income greeks crash:** VALID_INSTRUMENTS in
libs/trading_core/greeks.py was never extended for Phase 2, so an open
covered call/CSP CRASHED portfolio_greeks_read (ValueError on
"COVERED_CALL") — the income §16 branch existed but was unreachable.
Vocabulary now carries all 8 instruments; the Phase 3 e2e covers it.

**Tests:** +short-stock e2e (PLTR lands BEAR/MODERATE/HIGH via the vol
seam: preview → SELL_TO_OPEN fill → negative claim/market value → cover
→ cash identity), +adapter mirror gates, +backtests-API permission gate,
+5 engine tests (downtrend profits, no-trade honesty, mirrored stop,
no-look-ahead, hand-checked cash); conftest restores
ALLOW_SHORT_STOCK/ALLOW_MARGIN. Full suite **1032 passed**. UI: Settings
toggles + Phase 3 copy, backtest instrument picker now lists all 8 legs
(COVERED_CALL/CSP were missing too), tsc clean.

## 2026-08-17 (12) — PHASE 2 COMPLETE: covered calls + cash-secured puts UNLOCKED

**The §33 forbidden set shrinks for the first time.** covered_call and
cash_secured_put left _FORBIDDEN_ALLOW_FLAGS (Settings) and
FORBIDDEN_PERMISSION_FIELDS (AccountPermissions) — REAL flags now,
default False, runtime-config togglable (allow_covered_call /
allow_cash_secured_put), Settings UI Enable/Disable, live-verified on the
deployed gateway. Every §33 principle held: the toggles opened only after
the ENTIRE chain existed — selection, collateral locking, both venue
fills, mechanical management, sweep surfacing, §16 negative greeks,
§18 short-leg claim, buy-write + CSP backtests.

**Income real-broker settle (the last chain link):** §11 T1/T2 lifecycle —
durable PENDING_SUBMIT + request audit BEFORE the network, broker's
settled truth after (position at the BROKER's credit, no local cash
mutation); BrokerRejected -> REJECTED row + 422; BrokerError -> durable
row + 502 + reconcile pointer. Buyback identical in reverse. New
broker_exec wrappers (submit_short_open_and_poll /
submit_short_close_and_poll) with adopt-by-client_order_id idempotency.

**LATENT BUG KILLED — compact OCC everywhere:** occ_option_symbol used to
left-pad the root to 6 chars (canonical exchange OCC), but Alpaca is
COMPACT everywhere (chain snapshot keys, GET /v2/positions) — padded
local keys could never string-match broker rows in §18 reconciliation,
and the Phase-1/2 OCC regex gates rejected our own symbols. Found when
the income broker path hit its own adapter gate. One format now; five
test files updated off the padded layout.

**Tests:** income suite now drives the REAL toggles (no seams); +broker
e2e (position at broker's 2.40 credit, buyback at 1.10, realized
arithmetic); permission tests updated (income flags are real; forbidden
registry = short_stock/naked×2/margin). Full suite **1021 passed**;
deployed; live toggle round-trip verified.

**Remaining in the program:** Phase 3 — margin buying-power model + short
stock (the final two locks); naked×2 permanently locked (broker refusal).

## 2026-08-17 (11) — Income backtest legs: buy-write + CSP replays over real bars

**run_covered_call_backtest** — the buy-write replay: V1 stock leg
(same entries, same LIVE evaluate_exit) + a rolling short-call overlay on
the held shares. THE COLLATERAL LAW IN REPLAY FORM: the overlay exists
only against held shares (contracts = shares//100), and a stock exit buys
the call back ON THE SAME FILL BAR (atomic unwind — replay can never
strand a naked short). Overlay managed by the live mechanical standards;
expiry OTM keeps the credit, ITM = shares CALLED AWAY at the strike
(contractual settlement — the capped upside is the strategy's real cost,
and the replay now measures it). CHURN GUARD: an overlay already inside
the 21-DTE zone is never sold (it would be DTE-bought-back immediately,
bleeding slippage per cycle — caught by a characterization trace).

**run_csp_backtest** — bull-signal entries expressed as SOLD puts with
strike×100 reserved per contract (position_pct caps the reserved
fraction); adverse slippage on credit (receive less) and buyback (pay
more); returns measured on the CASH SECURED (the capital actually at
work); expiry ITM settles as CASH P&L = credit − intrinsic — a documented
cash-settled assignment approximation (the wheel comes later).

Router: COVERED_CALL / CASH_SECURED_PUT dispatch with moneyness-based
short-leg selection (5% OTM default — historical greeks do not exist,
stated); permission-gated like everything else (still locked until the
Phase 2 unlock).

**Tests:** +5 engine (profit-capture banking, assignment-at-strike with
same-bar unwind, CSP OTM/managed value, validation ×2) + API e2e with the
permission seam. Full suite **1024 passed**; deployed.

**The unlock now waits on ONE item:** the real-broker settle for income
opens/buybacks (the simulated chain, sweep, §16, §18, backtests and
selection are all done). Next turn: broker settle → UNLOCK covered_call +
cash_secured_put → Phase 3 (margin + short stock).

## 2026-08-17 (10) — Phase 2 wired: the collateral law runs end to end (simulated venue)

**Income selection lib** (strategies/income.py, pure+tested): the
mechanical standards — 30-45 DTE, |Δ| 0.15-0.35 (target 0.25), OTM ONLY
(§4: income, not leverage), sellable liquidity (real NBBO + a real BID,
OI floor, spread cap) — with named blockers and an annualized-yield
rationale line.

**Income router** (/api/income): covered-call and CSP opens + buyback.
THE COLLATERAL LAW enforced in one place:
- covered call: OPEN LONG_STOCK with FREE shares >= 100/contract (free =
  held − pinned under other open CCs); link stored; the STOCK close path
  and the exit sweep both refuse to sell pinned shares (loud HELD, never
  a silent skip) until the call is bought back;
- CSP: strike×100×qty reserved; deployable cash = cash − Σ reserved,
  enforced at open with the amounts named;
- kill switch: opens blocked while paused (obligations), buybacks always
  allowed (risk-reducing, §18 priority);
- buyback releases pin/reservation in the same transaction; realized PnL
  = (credit − buyback)×100×qty − commissions.
Real-broker path: honest 422 (adapter is ready; §26 settle/sweep
integration is the remaining chunk) — never a half-execution.

**Sweep**: income rows evaluated by evaluate_short_premium_exit; a
triggered rule audits EXIT_GENERATED with "action_required: BUY BACK via
POST /api/income/{id}/buyback" (auto-buyback arrives with the unlock).
**§16**: income rows contribute NEGATED short-leg greeks
(zeros-with-note on gaps). **NAV**: short premium booked as a LIABILITY
at credit (documented V1 book approximation). **§18**: income short legs
registered OURS at NEGATIVE quantity.

**Tests:** +9 (4 income-selection; 5 e2e: locked-gate 403, covered-call
full cycle incl. pin/refuse/release, CSP reserve arithmetic, kill-switch
asymmetry, §18 negative claim). Full suite **1019 passed**; deployed.

**Remaining for the covered_call / cash_secured_put unlock:** real-broker
settle+sweep, CC/CSP backtest legs, UI surfaces. Then Phase 3
(margin + short stock).

## 2026-08-17 (9) — Phase 2 foundations: collateralized short premium (covered calls / CSPs)

**User mandate:** implement every remaining locked permission platform-wide.
**Broker truth stated first:** naked short options do not exist at Alpaca
at ANY level — naked_short_call/put become PERMANENTLY locked with the
double refusal (broker + §4 charter) named in the Settings copy. The four
buildable ones proceed: covered calls + CSPs (Phase 2, started now),
short stock + margin (Phase 3, after).

**Foundations landed:**
- InstrumentType: COVERED_CALL / CASH_SECURED_PUT (income overlays, not §8
  directional entries); Position columns collateral_position_id /
  cash_reserved (migration 016, applied).
- Short-premium management in the SHARED exit engine
  (evaluate_short_premium_exit + ShortPremiumState): the most widely cited
  mechanical standards — PROFIT_CAPTURE at 50% of max profit,
  PREMIUM_LOSS_STOP at 2x credit (priority 1), DTE_EXIT 21 (§11.7 applies
  to sellers identically), plus a LOUD ITM assignment ADVISORY that never
  auto-triggers (economics are defined on collateralized positions — the
  human decides). Parameters on ExitParams (§6.2).
- Broker adapter: submit_short_open_order — OCC-REGEX-GATED (a stock
  symbol raises: short stock stays unconstructable until Phase 3 builds it
  deliberately) with a MANDATORY covered_by collateral attestation; and a
  dedicated submit_short_close_order (BUY_TO_CLOSE) so the single-leg
  submit keeps its exact two-word §5 vocabulary — the adversarial
  no-short tests stay meaningful and still pass.

**Tests:** +7 (4 short-premium rule characterizations incl. honest
None-mid; 3 broker: attestation/OCC gates, wire shape sell_to_open /
buy_to_close, live-account refusal). Full suite **1010 passed**; deployed.

**Remaining for the Phase 2 unlock:** gateway open/close paths with REAL
collateral locking (shares pinned under covered calls, cash reserved
under CSPs, stock-close refusal while collateralized), sweep integration,
§16 negative-greek contributions, §18 assignment recognition, covered
backtest legs, UI surfaces — then covered_call/cash_secured_put unlock.

## 2026-08-17 (8) — LIVE BUG FIX: direction-mirrored exits + the bear-side backtest legs

**The bug (confirmed by direct reproduction before fixing):** the shared
exit engine's underlying rules were hard-coded BULL — a LONG_PUT whose
premium had nearly DOUBLED (underlying −20%, bias BEAR, edge −51) returned
``should_exit: True, rule: SIGNAL_FLIP``. Every WINNING live put would have
been closed by the first exit sweep; SIGNAL_DECAY (edge < +10) made it
unconditional. Found while building the bear mirror — exactly the audit
the parity mandate exists for.

**Fix — direction-aware shared engine (industry-standard mirrors):**
``PositionState.direction`` ("BULL" default = byte-identical old behavior;
"BEAR" requires ``lowest_close_since_entry``): SIGNAL_FLIP fires on the
OPPOSING bias; SIGNAL_DECAY on the position-FAVORABLE edge (−edge for
bear) falling below threshold; ATR_TRAIL hangs ABOVE the trough
(trough + k×ATR — the standard short-side trailing stop); TIME_STOP
measures the favorable (downward) move. The stock engine refuses BEAR
(§5: no short stock). Live wiring: positions sweep passes BEAR + trough
for LONG_PUT / BEAR_PUT_SPREAD rows. Verified: the winning put now HOLDS;
a trough+3×ATR rally exits via SIGNAL_FLIP/trail.

**Bear-side backtest legs:** ``_evaluate_entry_bear`` (STRONG/MILD_BEAR +
BEAR bias + edge ≤ −threshold); the single-leg and spread engines are
direction-general (put intrinsic max(K−S,0), put-vertical shape short
strike BELOW long, net intrinsic mirrored, BEAR exit state with trough
tracking); router resolvers fetch put grids with OTM-below-spot targets;
permission gates mirror (allow_long_put / defined_risk_spreads).

**LIVE PROOF (AAPL, real Alpaca history):** LONG_PUT run — 3 trades on
real put contracts entered in the spring-2025 selloff, all stopped
(−14.2% total; the dips recovered too fast) — with the MIRRORED reasons
in every exit line ("BEAR-favorable edge …"). An honest, useful result:
the platform can now compare all five instruments on the same signal.

**Tests:** +5 direction-mirror characterizations (winning put held,
trough-trail rally exit, bear time stop, trough-anchor validation, stock
BEAR refusal) + bear-leg API e2e (PLTR LONG_PUT + BEAR_PUT_SPREAD with
shape assertions + permission mirror). Full suite **1003 passed**;
deployed.

## 2026-08-17 (7) — PHASE 1 COMPLETE: defined-risk spreads fully operable end to end

The unlock rule is satisfied — every link of the chain exists and is
tested, so `defined_risk_spreads` is now a FULL toggle (Settings copy
updated; the deferral override and the §10 spread FAIL guard are gone):

- §10 chain: INSTRUMENT passes spreads; CONTRACT_SELECTION runs the §9-S
  selector over the live chain (fail-closed when short-leg greeks are
  absent — §16 needs the net); RISK sizes on net debit × 100 with NET
  candidate greeks; preview carries a full `proposed.spread` block (§25
  leg identities, net debit = max loss, max profit, breakeven, net
  greeks) and the §24 exit plan quotes the net-debit premium stop.
- Execution: simulated venue fills at NET mid ± slippage with 2-leg
  commission; the real-broker path submits ONE ATOMIC mleg order
  (submit_mleg_and_poll — adopt-by-client_order_id idempotency intact),
  position rows carry both legs, and broker closes go out as the atomic
  {SELL_TO_CLOSE long + BUY_TO_CLOSE short} pair — never leg-by-leg,
  which would strand a naked short between fills.
- Positions/exits: spread rows ride the LIVE option exit engine on NET
  values (net entry debit, live net mid, dte); the sweep closes at net
  mid or the documented net-intrinsic fallback (bounded ≥ 0).
- §16: portfolio greeks contribute NET per-share greeks for spread rows
  (both legs located in the regenerated chain; zeros-with-note when
  either leg or its greeks are missing); greeks vocabulary extended.
- §18: `_local_open_quantities` registers the spread's short leg as OURS
  with NEGATIVE quantity — our own spread can never be mistaken for a
  foreign short and pause trading.

Tests: +3 e2e (preview net sizing + coherent §37 arithmetic;
approve→reconcile→close round trip incl. −qty short-leg claim, net-basis
max_loss, 2-leg close commission; permission-off graceful degradation).
Conftest now restores ALL runtime-config env keys per test (a toggled
permission leaked via os.environ between tests). Full suite **997
passed**; deployed. NOTE: first real-broker mleg round trip pending the
next market session (today is Sunday); the wire shape is pinned by
MockTransport tests.

## 2026-08-17 (6) — MLEG broker layer + spread position model (Phase 1: the money path begins)

**Broker layer:** `BrokerOrderLeg` + mleg-only side vocabulary
(SELL_TO_OPEN / BUY_TO_CLOSE exist ONLY here — the single-leg submit still
has no vocabulary for them) and `AlpacaPaperBroker.submit_mleg_order`: an
ATOMIC two-leg order whose SHAPE GUARD is the §5 safety boundary, enforced
BEFORE any network I/O — exactly two legs, same underlying/expiry/right,
different strikes, 1:1 ratios, pair ∈ {open: BUY_TO_OPEN+SELL_TO_OPEN,
close: SELL_TO_CLOSE+BUY_TO_CLOSE}, and the short strike must sit on the
COVERED side (above the long for calls, below for puts). A lone
sell-to-open is unconstructable through every path. Paper-only layer-2
guard identical to the single-leg path. The spread returns ONE
BrokerOrder: symbol "LONG/SHORT", filled_avg_price = NET per-share (buys −
sells) only once EVERY leg reports a fill — None until then (a
half-filled net would be an invented number).

**Position model:** `short_occ_symbol` / `short_strike` columns
(migration 015, applied) — spread rows carry the long leg in opt_*, the
short leg here; avg_price = net debit/share; max_loss = qty × net × 100
(defined risk, §12.1). Honest NULLs for every non-spread row.

**Tests:** +4 broker suites (atomic payload/intents, half-fill net=None,
the full illegal-pair battery incl. lone sell-to-open and uncovered
strikes, live-account refusal on the close pair). Full suite **994
passed**; deployed.

**Next (the wiring turn):** orders.py approve/close for spreads (mleg +
simulated venue net fills), §16 two-leg net greeks in portfolio risk,
exit sweep on net premium, §18 reconciliation recognizing our short legs
as OURS, then remove the live degradation override = FULL UNLOCK.

## 2026-08-17 (5) — BULL_CALL_SPREAD backtest leg + the spreads toggle goes live (partial scope)

**Engine (backtest/options.py `run_spread_backtest`):** net-debit semantics
throughout — net debit = MAX LOSS, so the LIVE `evaluate_option_exit` runs
verbatim on (net entry debit, current net mid, dte). §20.2 slippage is
ADVERSE ON BOTH LEGS both ways (pay up long + receive less short at entry;
reversed at exit). Honest-gap rules: a day where EITHER leg didn't trade
has no observable net (entries skip, marks carry the last real joint
observation, premium stop reports insufficient data); degenerate fills
(net ≤ 0 or ≥ width) never fill; expiry settles at NET intrinsic bounded
[0, width] off the real underlying close. Router resolver: long leg as the
LONG_CALL pick, short leg = nearest REAL strike to long + spread_width_pct
× spot in the SAME expiry.

**Permission goes togglable with stated partial scope:**
`allow_defined_risk_spreads` joins runtime-config; it gates spread
RESEARCH + BACKTEST now, while the live §10 chain REPLACES spreads off
(dataclasses.replace) with the scope stated in the INSTRUMENT gate detail
— live plans keep degrading to the single leg instead of dead-ending, and
the Settings row copy declares the same scope. Full unlock happens when
mleg execution + leg-paired positions/exits land.

**LIVE PROOF (AAPL, run #4, real Alpaca history):** 4 spread trades on
real leg pairs (e.g. AAPL250417C00245000 / C00255000, width 10), net
fills strictly inside (0, width), exits by SIGNAL_DECAY / DTE_EXIT
(+85.9%) / ATR_TRAIL / PREMIUM_HARD_STOP on the NET. Total −7.07% vs the
single-call leg's +11.35% over the same window — the short leg capped the
big winner: exactly the instrument-level comparison the platform now
enables.

**Tests:** +7 (adverse-slippage net fill hand-computed, net premium stop,
net-intrinsic expiry bounded by width, either-leg-missing skip, degenerate
net skip, validation ×2) + API e2e with permission round-trip. Full suite
**991 passed**; deployed.

## 2026-08-17 (4) — §9-S vertical spread selector (Phase 1 continues)

`libs/trading_core/contracts/spreads.py` — pure, deterministic, §21
composition: the LONG leg is exactly the §9 selector's rank-1 (every
filter and ranking unchanged); the SHORT leg comes from the SAME expiry,
OTM-ward, nearest real strike to `width_pct_target` of spot inside
[min, max] (SpreadParams — §6.2 research parameters, exposed in
/api/config as `spread_params`), with its OWN liquidity gates: real NBBO
only (a day-close mid has an unknown spread and cannot be SOLD —
fail-closed), OI floor, spread cap. Output: SpreadCandidate with the
defined-risk arithmetic (net debit = MAX LOSS, width − debit = max
profit, breakeven) and None-safe NET greeks (either leg unknown → None,
never zero-filled), all §37-explained. Degenerate quotes fail closed
(net debit ≤ 0 anomaly; debit ≥ width = no upside). NO-ELIGIBLE is a
valid output with named blockers incl. §9 top-blocker counts.

Tests: +8 (hand-checked bull math, bear mirror, long-leg §9 blockers
surfaced, width band, short-leg filters, degenerate quotes, None-safe
greeks, determinism+validation). Full suite **984 passed**; deployed.

Phase 1 remaining: §16 net-greeks integration → Alpaca mleg execution +
§26/§18 leg pairing → positions/exits → spread backtest leg → UI →
unlock `defined_risk_spreads`.

## 2026-08-17 (3) — Execution-chains program launched: every permission to become operable

**User mandate:** the full chain (backtest/trading/plans/risk) must
support every execution type so all account permissions are genuinely
operable. **Broker ground truth probed live:** the paper account is
MARGIN (multiplier 4), shorting_enabled, options level 3 (spreads) — every
lock is a platform construction gap, not a broker refusal (except naked
short options, which Alpaca does not offer at all).

**docs/execution-chains-roadmap.md** is the authoritative plan. The
unlock rule: a toggle opens ONLY when its entire chain (§8→§9→§10→risk→
execution→positions/exits→reconciliation→backtest→UI) exists and is
tested. Phase 1 = defined-risk spreads (L3-approved, max loss = net debit
fits §12 as-is); Phase 2 = covered calls + CSPs (first collateralized
Sell-to-Open, assignment handling); Phase 3 = margin/short stock (may be
deliberately skipped — puts already express bears); naked shorts stay
permanently locked (broker refusal + §4 charter).

**Phase 1 research spine landed:** InstrumentType gains BULL_CALL_SPREAD /
BEAR_PUT_SPREAD; the §8 matrix now emits the IDEAL spread from its spread
cells and `_finalize` owns ALL §5 degradation (single place, explained);
BEAR/MODERATE/HIGH upgrades from NO_TRADE to spread when permitted; the
§10 INSTRUMENT gate FAILs spreads honestly ("spread EXECUTION is not
built yet — roadmap Phase 1") so nothing half-executes; the
defined_risk_spreads flag remains env-only/locked until the chain
completes. Tests: +3 spread-cell suites, 1 superseded V1 test updated.
Full suite **976 passed**; gateway redeployed.

## 2026-08-17 (2) — OPTIONS JOIN THE BACKTEST: LONG_CALL leg over real contract bars

**The milestone the fabrication ban was waiting for.** Engine
(backtest/options.py): replays the SAME bull entry signal, expressed by
buying a REAL historical call — contract chosen at the decision date from
the REAL strike/expiry grid (moneyness+DTE targets; greeks/OI have no
history), filled at the contract's REAL next-bar open with a §20.2 bps
spread proxy (no NBBO history exists), marked to REAL closes, exits run
the LIVE evaluate_option_exit (§21: PREMIUM_HARD_STOP §11.3, DTE_EXIT
§11.7, shared signal/trail/time rules). Honest-gap rules: missing fill
bar -> entry SKIPPED; no-trade day -> premium stop says "insufficient
data"; no tradable bar before expiry -> intrinsic settlement off the real
underlying close (contractual arithmetic). NO TRADE valid as ever.

Wiring: BacktestParams.instrument ("LONG_STOCK"|"LONG_CALL") + option
params (DTE window, OTM%, premium budget, per-contract commission, option
slippage/worst); permission gates per leg (allow_long_call for CALL —
same factory as live); provider methods get_option_contracts_window
(both statuses, adjusted 1AAPL… filtered) + get_option_daily_bars on
alpaca, deterministic synthetic equivalents on stub; router resolver runs
the engine in a worker thread (network in the callback); option trades
serialize with contract identity; summaries chip the instrument.

**LIVE PROOF (AAPL, run #3, real Alpaca history):** 6 trades on real
contracts (AAPL250417C00245000 …), +11.35% total, win rate 33%, exits
fired by SIGNAL_DECAY / PREMIUM_HARD_STOP / DTE_EXIT / ATR_TRAIL — the
live rules, on real premiums. LONG_PUT arrives with the bear-side signal
mirror.

**Tests:** +10 (7 engine: hand-computed fill, premium stop, DTE exit,
expiry settlement, missing-bar skip, no-trade, validation; 3 API: e2e via
stub, per-leg permission gates, option-param 422). Full suite **973
passed**.

**Ops:** the user's container fleet expanded — host 3000 now held by
platform-ui, so THIS platform's Next dev server moved to **:3001** (CORS
origin added); gateway stays :8011, db :5433.

## 2026-08-17 — User-level instrument permissions + historical options data verified

**1. Historical options data (probed LIVE, documented in
data-source-architecture.md):** expired contracts enumerable
(status=inactive; adjusted `1AAPL…` symbols are not valid data-API keys);
FULL-LIFE daily bars for expired contracts (AAPL241220C00250000: 230 bars,
2024-01-18 → expiry); earliest ~Feb 2024 (pre-Feb-2024 contract: 0 bars);
tick trades ✅; historical NBBO quotes endpoint does NOT exist (404 route);
greeks/IV/OI have no history (compute IV from real prices; OI filters have
no historical values). **Verdict: the options backtest is DATA-UNBLOCKED
from ~Feb 2024 under the fabrication ban** — next major build.

**2. User-level instrument permission switches:** the three REAL §5 flags
(allow_long_stock/call/put) are now runtime-config settable (STRICT
true/false — "" would make the pydantic bool Settings field, and the whole
app, unconstructable → gated out). One factory
(account_permissions_from_settings) already feeds plan generation, the §10
live gate chain and GET /api/config, so a toggle applies everywhere at
once; NEW: the backtest gate — Engine V1 is stock-only, so
allow_long_stock=false now honestly 422s a backtest instead of replaying
an instrument the user forbade. The six §33 forbidden capabilities
(short/naked/margin/covered/CSP) remain unsettable from every entry point
(unknown request fields are ignored; Settings validator + dataclass
__post_init__ still hard-refuse). defined_risk_spreads stays env-only
until spread execution exists.

**Tests:** +2 (round-trip incl. backtest-gate flip and strict-value 422s;
forbidden-field immunity). Full suite **963 passed**. Live round-trip
verified on the deployed gateway.

**Ops:** gateway host port 8010 → **8011** (8000/8010/8012/8013 all held by
unrelated local containers; 8010 taken by platform-agent's agent-service
today). ui/.env.local updated to http://localhost:8011.

## 2026-08-16 (2) — Backtest↔live parity refactor + IS/OOS removal (user mandate)

**User mandate:** (a) backtest and live logic must be IDENTICAL — same
rules, same parameters, same adjustment paths — "不然回测就没了意义";
(b) DELETE the IS/OOS design for the manual-tuning-only era; reintroduce
only if/when ML-driven parameter search arrives. (b) supersedes plan §20's
IS/OOS reporting and §44 rule 16, and retires yesterday's oos_eval_count.

**Parity review findings (full pass):**
- Signals: SHARED ✓ — engine and live both call classify_regime /
  score_direction with the same defaults (§21 held).
- Entry threshold: backtest default 25.0 == live moderate_edge 25.0 ✓.
- F1 (fixed): exits were REIMPLEMENTED privately in the backtest
  (_evaluate_exit) — identical logic today, guaranteed drift tomorrow.
- F2 (fixed): live stock exits include HARD_STOP (§11.3, priority 1,
  stop = ATR_STOP_MULTIPLE × ATR14 at entry, never widens) — the backtest
  had NO hard stop at all.
- Documented scope gaps that remain (not silently ignored): §10/§12/§16
  risk gates and real fills are execution-layer, not modeled in V1.

**Refactor:** backtest now calls libs.trading_core.exits.evaluate_exit —
the live engine — with ExitParams mapped 1:1 from BacktestParams; entries
additionally require a sizable ATR stop (live refuses unsized stock
entries, replay now refuses identically); ATR_STOP_MULTIPLE moved to
libs/trading_core/risk/engine.py as the SINGLE SOURCE imported by both
routers/orders and the backtest. Exit reasons now carry the live engine's
full §38 lines (incl. HARD_STOP).

**IS/OOS removal:** BacktestParams.oos_split gone; metrics = ONE flat
full-period object; oos_start_date/oos_eval_count dropped from all
payloads. Legacy rows are NOT rewritten — _flat_metrics presents their
stored "full" segment. Trading-pool promotion check OOS_STATS →
BACKTEST_TRADES (≥1 closed trade over the whole period).

**Tests:** engine suite rewritten where needed + 2 new parity tests
(HARD_STOP crash scenario; the three ATR_STOP_MULTIPLE imports are the
same object). Full suite **961 passed**; gateway redeployed; legacy row
verified serving flat metrics live.

**Ops note:** services db host port moved 5432→5433 in docker-compose
(another local project, platform-services, now binds host 5432; in-network
gateway→db traffic unaffected).

## 2026-08-16 — OOS data-snooping accounting (user: "repeated OOS peeking overfits anyway")

**User's (correct) methodology challenge:** an IS/OOS split cannot stop a
user who iterates parameters while reading the OOS column each time —
multiple testing turns held-out data into in-sample data. ML is orthogonal
(it amplifies the search, it doesn't cause or cure the leak); the real
defense is protocol + accounting.

**Implemented the accounting half:** `oos_eval_count` on every backtest
record payload — COMPLETED runs for the same ticker up to and including
that run (`_oos_eval_count`, counted as-of so stored records never change
meaning). The UI banner now displays "OOS evaluation #N", escalating
amber ≥5 / red ≥10, with the honest statement that the split only protects
the first look and that settings picked after many looks are unvalidated
until they survive data arriving after today.

**Deliberately NOT implemented (would need product decisions):**
walk-forward analysis (rolling IS→OOS windows), deflated Sharpe / PBO
(trial-count-corrected significance), and a lockbox third segment. Offered
to the user as candidate next steps.

**Tests:** contract key added; new per-ticker count test (increments per
completed run, per-ticker isolation, as-of stability on detail GET). Full
suite **962 passed**; gateway redeployed.

## 2026-08-15 — LLM output language (Settings.llm_output_language)

**User:** the whole product should read in Simplified Chinese, including
LLM-generated analysis. UI copy is the frontend's job (bilingual t()/glossary
layer); the LLM narrative has to be fixed at the SOURCE — generation — so:

- `Settings.llm_output_language` ("en" default | "zh"), runtime-config
  overridable (`LLM_OUTPUT_LANGUAGE`), enum-gated in ALLOWED_PROVIDERS,
  exposed via GET/PUT /api/config/providers (`llm.output_language`).
- `libs/llm/provider.py::language_instruction()`: a system-prompt addendum,
  appended by BOTH real providers (openai generate+enrich, anthropic
  generate). NARRATIVE fields only (summary, evidence snippets) switch to
  简体中文; machine-read fields (ticker, company, horizon, catalyst_type,
  reason_codes, urls, timestamps) are pinned English — downstream
  filtering/analytics key on them, and a mixed-language enum column is
  silent data corruption. "en"/unknown → no addendum (prompts byte-identical
  to before the feature).
- Records are never rewritten: stored recommendations/catalyst rows keep the
  language they were generated in; only NEW generations follow the setting.
- Runtime config set to "zh" per the user's request.

**Tests:** +5 (language_instruction unit incl. unknown→"", openai zh on both
wire prompts, openai default stays English, anthropic zh system prompt,
runtime-config round-trip + 422 on unknown language). Full suite **961
passed**; gateway redeployed, /api/config/providers serves output_language.

## 2026-08-13 (10) — WS/REST boundary explained + OI day-cache + professional chain visuals

**User: "we have the websocket — why several REST calls for the chain?"**
The honest boundary: Alpaca's OPTIONS stream (OPRA WS) carries trades and
quotes ONLY — **greeks, IV and open interest exist solely in the REST
snapshots**, and a chain view needs them for ~700 contracts. So the chain
rebuild is REST by necessity; the job is to make each rebuild cheap:

- **OI day-cache** (`_OI_DAY_CACHE` in the Alpaca adapter): open interest
  is a once-per-day OCC number (the endpoint even stamps
  `open_interest_date`), yet the contracts call ran on EVERY chain
  rebuild. Now one contracts fetch per (underlying, Eastern day); faults
  are not cached (next rebuild retries); old days evicted; per-test
  isolation in conftest. A chain rebuild is now ONE Alpaca call
  (snapshots) instead of 2-3.
- Combined economics per open Options tab: UI polls 60s → server 20s chain
  cache → 1 snapshots call/min + 1 contracts call/day + the stock
  websocket. (Stock stream stays REST-free.)

**Chain UI/UX (professional pass):** ITM moneyness shading per side (calls
below spot / puts above — the standard cue), a SPOT $X divider row anchored
between the surrounding strikes, sticky two-row header with a 70vh scroll
viewport, full-row hover, bid green / ask red tinting, strike column
emphasized. Candidate/eligible tints kept louder than the moneyness wash.

**Tests:** full suite **956 passed**; tsc clean; §47 scan + component tests
green.

## 2026-08-13 (9) — Chain polling economy (user: "why does the selector strip keep auto-loading?")

**Diagnosis:** the "refreshing…" flicker in the CONTRACT SELECTOR strip was
the options-chain query on the UI's GLOBAL 15s react-query poll — and every
poll made the gateway rebuild the full chain against Alpaca (snapshot pages
+ contracts OI merge, several REST calls each). ~12+ Alpaca calls/min for
one open Options tab, mostly re-reading identical data.

**Fixes:**
- **Server: short chain cache** — `build_option_chain` caches per
  (provider, ticker, Eastern day) for `CHAIN_CACHE_TTL_SECONDS = 20`;
  polling READ surfaces (options view, positions, portfolio) reuse a
  seconds-old build. **Execution paths bypass with max_age_seconds=0**
  (§21/§42: the approve gate chain and close pricing ALWAYS hit the
  provider live — orders never trust a cache). Cache cleared on provider
  switches; per-test isolation in conftest.
- **UI: chain query slowed to 60s** (staleTime 30s) — quotes stream
  server-side and the server holds the short cache, so faster polling only
  re-read the same build. The strip now refreshes once a minute instead of
  every 15s.

**Measured live:** first chain call 300ms (real fetch), second 32ms (cache,
zero Alpaca calls). Tests: cache-bounds + execution-bypass pinned
(monkeypatched provider call counter). Full suite **956 passed**.

## 2026-08-13 (8) — Alpaca WebSocket streaming (data_source.md §5): stop polling REST for streamable values

**Built (user request; §5 of the data-source spec):**
- `libs/market_data/alpaca_stream.py` — the pure half: Alpaca stream
  protocol parsing (JSON arrays, T: q/t/success/error/subscription;
  nanosecond RFC-3339), `QuoteCache` with the freshness contract "stale =
  absent" (a quiet/closed market yields an honestly empty cache, never
  invented ticks), auth/subscribe message builders. Fully unit-tested — no
  sockets in tests.
- `apps/gateway/market_stream.py` — the supervisor task (lifespan-started,
  monitor-loop pattern): ONE websocket to
  wss://stream.data.alpaca.markets/v2/sip, auth → subscribe
  (watchlist tickers + SPY/QQQ, re-diffed every 60s so watchlist edits
  follow; VIX never subscribed — no index feed), bounded-backoff
  reconnect, self-DISABLES whenever the configured provider is not
  "alpaca" (checked every cycle — runtime provider switches need no
  restart). `GET /api/market/stream/status` reports the honest facts.
- Overview integration: REST snapshot quotes (the day-change baseline —
  they carry prev close) are now cached 60s, and a FRESH streamed trade
  (≤30s) supersedes the REST price with the change rebased on the SAME
  baseline. Per-index `transport: "stream" | "rest"` says which path
  served each number. Net effect: Alpaca REST quote calls drop from one
  per UI poll (~4/min) to ≤1/min, with live prices streaming in-session.
  Caches are provider-keyed/cleared on runtime switches.

**Honesty preserved:** stream and REST are the SAME provider — §33 forbids
cross-provider fallback, not transport fallback within one vendor; a
broken stream degrades to REST silently-visibly (transport field), and the
stream never fabricates: only socket-delivered messages enter the cache.

**Tests:** `tests/test_market_stream.py` (6): protocol parsing incl.
garbage frames, cache verbatim application + zero-price refusal, the
freshness contract, message shapes, status endpoint under the stub
(disabled/starting — never a fabricated connection), and the overview
stream-override with rebased change_pct. Full suite **955 passed**.

**LIVE-VERIFIED:** deployed at 22:10 ET — stream `state: connected` on the
SIP feed, subscribed [QQQ, RDW, SMCI, SPY] (watchlist + indices, auto-
diffed), zero messages applied (after-hours silence — honest); overview
serves transport:"rest" until the next session streams trades.

**Deferred (documented):** browser push (SSE/WS gateway→UI) — the UI still
polls the LOCAL gateway; options-chain streaming (OPRA per-contract
channels are heavy; chains stay REST on-demand).

## 2026-08-13 (7) — One-sided NBBO fix (user challenged the "—" cells; raw-snapshot audit)

**User report:** dashes in the T-chain — real or bug? Audited the exact
contracts' RAW Alpaca snapshots:

- greeks/IV dashes: REAL honest nulls — Alpaca serves no greeks for those
  deep/illiquid contracts (verified `greeks: None, impliedVolatility:
  None` on the wire).
- bid/ask dashes: **our bug** — RDW260814P00012500 carried a REAL
  one-sided NBBO (`bp: 0` = OPRA's reported NO-BID state, `ap: 0.05` = a
  real 9-lot offer). The parser demanded a two-sided market and discarded
  the whole quote to the day-close fallback, dropping Alpaca's real ask.

**Fix:** `_parse_chain_row` gained a one-sided branch: ask-only NBBO keeps
`price_basis: "quote"` with bid = the reported 0 (a market fact, not an
unknown), ask preserved verbatim, mid = the canonical no-bid midpoint
(ask/2), spread at the documented worst case so §9 can only ever REJECT.
Pinned by a wire test with the live-observed payload. Full suite **949
passed**. Live-verified: 12.5P and 14.5C now show bid 0.00 / ask 0.05
(quote basis) instead of dashes.

## 2026-08-13 (6) — Chain completeness + Direction visibility (user: "chain incomplete; Direction does nothing visible")

**Chain completeness (backend):** the Alpaca adapter was DROPPING every
contract the feed serves without greeks/IV (deep ITM/OTM wings) — real,
quoted contracts the user should see. `ContractQuote` greeks/IV are now
NULLABLE (honest nulls, never zero-filled): the adapter keeps greekless
rows with their real NBBO quotes; the §9 selector rejects them with the
named reason "greeks/IV not provided for this contract"; `chain_iv_summary`
draws ATM IV only from contracts that HAVE an IV; portfolio greeks
contribute zeros-with-note for a greekless matched contract (same posture
as a missing contract); the degeneracy flag guards nulls. LIVE RESULT: RDW
chain 369 → **723 rows**; the Aug-14 expiry went from 7 near-ATM rows to
**68 rows spanning strikes $1–$21** — full depth, honestly labeled ("—"
for absent greeks).

**UI:**
- **Strike-depth filter** (user request): ±10% / ±25% / All-strikes chips
  (client-side moneyness around spot, tooltips show the dollar window).
- **Direction made visible** (user: clicking AUTO/BULL/BEAR changed
  nothing): an explanation line now states what it does ("Direction drives
  the §9 contract selector, not the chain display: BULL → shopping CALLS —
  only that side can be Eligible/Recommended...") and wrong-side rows DIM
  (opacity) while staying listed — the toggle's effect is now visually
  obvious even when zero contracts are eligible.
- Greeks cells render "—" for provider nulls (never 0.000).

**Tests:** greekless-row test re-pinned to keep-with-nulls + named selector
rejection; full suite **948 passed**; tsc clean; component tests green.

## 2026-08-13 (5) — Adversarial review of the Alpaca migration: 12 confirmed findings, all fixed

A 33-agent review workflow (3 reviewers × correctness/consumer-contracts/
config-UI, then 2 adversarial verifiers per finding) confirmed 12 real
defects (1 rejected). All fixed and re-verified:

1. **get_quotes fabricated change_pct=0.0** when prevDailyBar was missing —
   now SKIPS the symbol with a warning (a 0.00% "unchanged" reading is a
   real market state; unknown is not it — same posture as the Massive
   adapter).
2. **False "chain truncated" warning** when pagination completed exactly on
   the last allowed page — truncation now requires budget exhausted AND a
   live next_page_token (bars loop fixed identically).
3. **VIX warning log spam** — permanent condition now logged once per
   process, not per request.
4. **_eod_cache unbounded growth** — past-day entries evicted on insert
   (bounded by watchlist size).
5. **EOD cache not provider-aware** — key now includes the provider name
   AND `_clear_derived_caches` clears it on any provider switch.
6. **`missing_on_this_plan` untruthful under Alpaca** (quotes/greeks ARE in
   the plan — they live on the chain view) — renamed `not_in_this_view`
   with view-fact semantics; docstrings de-Massived; UI copy follows.
7. **Settings card showed green "Disconnected."** after a Connect that
   saved but failed to configure — result text now distinguishes
   connect/disconnect and states "Saved — NOT configured" honestly.
8. **Disconnect button hidden exactly when the stored selection was
   broken** — now gated on a provider being STORED, not on configured.
9. **Provider select never resynced with server state** — card remounts on
   the server-reported provider (key prop).
10. **httpx.Client never closed** on per-request provider construction —
    close() + GC-time close added (Massive-precedent pattern otherwise).
11. **CAPABILITY_META lacked option_contracts** — Settings capability table
    now labels it.
12. **Ten stale "Massive is the only data source" comments** across the UI
    — updated to provider-neutral truth.

Rejected (2/2 verifiers): "OI merge fetches a subset → liquid contracts
read OI 0" — the unbounded-expiry 2×10,000-row fetch covers any realistic
single-underlying contract count.

**Re-verified after fixes:** backend 948 passed; tsc clean; 9 component
tests green; §47 scan OK; live gateway on alpaca — all 5 capabilities TRUE,
overview SPY/QQQ, RDW chain 369 rows 100% NBBO quote basis, EOD view
serving the renamed honest payload.

## 2026-08-13 (4) — DATA-SOURCE ARCHITECTURE UPGRADE: Alpaca is now the authoritative market-data provider

Executed per `prompts/data_source.md`. The §43 deliverables:

**A. Current state (audited by parallel agents + live probes):** all market
data (stocks bars/quotes, option chain, news) came from MassiveProvider; the
provider layer was already cleanly funneled through
`libs.market_data.get_provider` + two shared helpers (ensure_daily_bars,
build_option_chain) + one guard — no vendor calls in business logic. Broker
was already Alpaca.

**B. Changes made:**
- **New `libs/market_data/alpaca.py` (`AlpacaMarketDataProvider`)** — a full
  drop-in implementing the protocol + every getattr-gated extension. Every
  endpoint LIVE-VERIFIED against the user's Algo Trader Plus account before
  implementation:
  - bars: `GET /v2/stocks/{s}/bars` (1Day, split-adjusted, RFC-3339 → Eastern
    trading date, next_page_token paging);
  - quotes: ONE multi-symbol `GET /v2/stocks/snapshots` call (latestTrade.p,
    change vs prevDailyBar.c). **VIX/indices: Alpaca has no index feed — the
    symbols are skipped pre-request with a warning (honest absence, never a
    proxy)**;
  - option chain: `GET /v1beta1/options/snapshots/{u}` (OPRA) — snapshots
    keyed by BARE OCC symbol with REAL NBBO bid/ask, latestTrade, dailyBar,
    greeks (incl. rho) + provider IV; **open interest merged from the
    Trading-API `GET /v2/options/contracts`** (snapshots don't carry OI; a
    merge failure degrades to OI 0 "none reported", never blocks the chain);
    greekless rows skipped, quoteless rows priced from the session close as
    `price_basis: "day_close"` with worst-case spread; current-day-only
    guard; deterministic sort;
  - contracts + prev-day bar (EOD surface): Trading-API contracts (handles
    Alpaca's STRINGIFIED numerics) + per-contract snapshot prevDailyBar;
  - news: `GET /v1beta1/news`, verbatim mapping, uncitable rows skipped,
    `source_id` prefixed `alpaca:` so the dedup keyspace can never collide
    with stored Massive ids;
  - probe_capabilities with the platform's EXACT §16 keys; 403 →
    CapabilityNotAvailable; keyless construction refused naming the env
    vars; keys never logged.
- Registry: `"alpaca"` factory (reuses the broker's alpaca_api_key_id/secret
  — verified they authenticate against data.alpaca.markets);
  runtime_config ALLOWED_PROVIDERS learned "alpaca"; .env.example rewritten;
  options EOD payload copy made provider-neutral.
- UI: Settings Market Data card is now a PROVIDER CHOICE (Alpaca recommended
  — uses the broker keys, with stored-key awareness; Massive still
  selectable with its key input). All "Massive plan" copy in
  chain/news/capability surfaces made provider-neutral.

**C. Data source matrix + D. architecture:** `docs/data-source-architecture.md`
(new) — the single source of truth: provider responsibilities, per-field
matrix (provider/endpoint/raw-vs-calculated/cache), §33 no-cross-provider-
fallback policy, §30/§31 point-in-time rules, subscription assumptions
($99 Alpaca + $29 Massive F&R = $128/mo), and the §38 historical-options
warning (backtests must depend on the provider interface, never on
AlpacaMarketDataProvider — Alpaca's option history starts ~Feb 2024).

**E. Tests:** new `tests/test_alpaca_market_data.py` (22, MockTransport over
live-verified wire shapes): no-fabrication (VIX absence, greekless skip,
historical-chain refusal), 403 taxonomy, OI merge + degrade, bare-OCC
parsing, news id-prefixing, probe keys, registry + keyless refusal;
runtime_config accepts "alpaca". Full suite **948 passed, 1 skipped**.

**LIVE CUTOVER (verified end to end):** PUT market_data_provider=alpaca →
capabilities ALL TRUE (stock_history/realtime, option_chain,
option_contracts, news); market overview SPY/QQQ real quotes + STRONG_BULL
regime (VIX honestly absent); RDW chain **369 contracts, 100% real NBBO
quotes** (vs day-close basis on the old provider), ATM IV 91.8%, selector
failing candidates for LEGITIMATE §9 reasons (DTE/OI/spread — an illiquid
small-cap's real market); analysis serves source:alpaca over the stored
601 bars (append-only continuity, provider recorded per backfill audit);
news refresh ingested 50 real Alpaca articles → 5 grounded recommendations
(llm_model stamped), zero dedup collisions.

**F. Remaining gaps (documented):** Massive Financials & Ratios fundamentals
integration + Fundamental Score engine (§17–§22 of the spec) not yet built;
VIX unavailable on Alpaca; historical options depth ~Feb 2024; stock
streaming (WebSocket, §5) deferred — REST snapshots serve the current
15s-refresh UI honestly.

**G. Cost:** Alpaca Algo Trader Plus $99 + Massive Financials & Ratios $29
= $128/month. No other paid provider without explicit approval.

## 2026-08-13 (3) — Data-integrity pass on the options chain (user challenged bid/ask=0, delta 1.000, IV oddities)

**Bugs found and fixed:**
1. **Evening timezone bug (the empty chain in the user's screenshot):**
   `build_option_chain` stamped `as_of` with the UTC date; after 20:00 ET
   the UTC calendar has rolled, so the provider's current-state guard
   rejected every request (`as_of=tomorrow != today`) → 500 → the UI showed
   its stale cached 0-row chain. Fixed: the trading day is EASTERN —
   `as_of = datetime.now(EASTERN).date()`. Same one-clock fix applied to
   the positions DTE read (`_option_live_read`), which disagreed with the
   chain paths by one day every US evening (caught by the round-trip test).
2. **bid/ask honesty:** rows priced from the day close (quotes-less plan)
   previously serialized `bid: 0.0, ask: 0.0, spread_pct: 2.0` — 0.00 reads
   as "no bid", a REAL market state, which this is not. `ContractQuote`
   gained `price_basis` ("quote" | "day_close"); the API now serializes
   bid/ask/spread as **nulls** with `price_basis: "day_close"`, and the
   worst-case spread stays INTERNAL to the §9 selector's fail-closed
   filter. The UI renders "—", tags mid with an amber EOD marker, and the
   spread column shows "—".
3. **Degenerate IV flagged, not hidden:** deep-ITM/OTM rows where premium ≈
   intrinsic (or below it, when the option's close and the stock's close
   traded at different instants) or |delta| ≥ 0.98 or IV < 2% now carry
   `iv_unreliable: true` — IV inversion is mathematically unidentifiable
   there, and vendors emit noise (observed live: $1-strike call, IV 0.035%,
   delta 0.9999999). The vendor's number still renders VERBATIM; the UI
   dims it with a ⚠ tooltip explaining why. 59 of 433 live RDW rows flag.
4. **"Direction" tile relabeled** (user: "no professional meaning"): it is
   the PLATFORM's §9 selector direction (BULL → calls, BEAR → puts; AUTO =
   the deterministic signal bias), not a market-data field — now labeled
   "Selector direction" with that explanation in the tile and tooltip.
   Chain footer now prints the snapshot DATE as a date ("US trading day")
   instead of a midnight-UTC timestamp that rendered as "8:00 PM
   yesterday" in ET.

**On the user's specific numbers (verified live):** delta 1.000 was the
$1-strike call on a $13.49 stock — delta ≈ 1 is mathematically correct
there (their 0.9939 was a different contract/model); ATM IV ~95–110% on
1-DTE RDW options is plausible for this name (RV20 alone is 109.9%);
call/put same-strike IV asymmetry comes from closes traded at different
instants — exactly what the new ⚠ flag marks.

**Tests:** ROW_KEYS updated (+price_basis, +iv_unreliable); quoteless
fallback test carries price_basis; full suite **925 passed, 1 skipped**.
Live-verified: chain serves 433 rows again this evening (timezone fix),
ATM rows show `bid/ask: null (day_close)` and flags where warranted.

## 2026-08-13 (2) — Quotes-less chain fix: the upgraded plan's snapshot has greeks/IV/OI but NO last_quote

**User report:** after upgrading the Massive options plan the chain was
still empty. Live diagnosis inside the gateway: `/v3/snapshot/options/RDW`
answers 200 with 250 rows/page carrying `day` bars, `greeks`,
`implied_volatility` (181/250) and `open_interest` — but **zero rows have a
`last_quote` block** (this tier has no NBBO quotes entitlement). Our parser
required bid/ask-or-midpoint and skipped every row → an empty chain that
LOOKED like no data.

**Fix (honest fallback, never fabrication):** `_parse_chain_row` gained a
third pricing branch: when the plan omits `last_quote` entirely, the row
prices from the snapshot's DAY-bar close — a REAL traded session price —
IF that bar is fresh (`DAY_BAR_MAX_AGE_DAYS` = 5; an expired session is
refused). bid/ask stay honest zeros and the spread records the worst case
(`UNQUOTED_SPREAD_PCT`), so unknown quote quality can only ever REJECT in
the §9 selector: the chain is fully VISIBLE with real greeks/IV/OI, and
selection stays gated on real quotes.

**Tests:** new `test_chain_quoteless_plan_falls_back_to_fresh_day_close`
(fresh row priced from day close with worst-case spread + honest zero
bid/ask + untouched IV; stale-session row refused). Full suite **925
passed, 1 skipped**.

**Live-verified:** RDW chain now serves **435 contracts / 12 expiries**
with real IV (ATM IV 94.6%), deltas and OI; the §7 summary populates (ATM
IV 0.9456, expected move ±21.5%, IV−RV −15.4%). Contracts read
`eligible: false` under the worst-case spread — correct until the plan
includes NBBO quotes (or quotes appear in-session), at which point the §9
selector unlocks with no further code change.

## 2026-08-13 — Massive API audit + free-tier EOD options surface (user report: "no options data")

**Audit (every endpoint verified against https://massive.com/docs):**
- In use and CORRECT/current: stocks custom bars `/v2/aggs/.../range/...`
  (all plans), stock snapshot `/v2/snapshot/locale/us/markets/stocks/...`
  (Starter+, user has it), indices `/v3/snapshot/indices`, news
  `/v2/reference/news` (all plans), option chain snapshot
  `/v3/snapshot/options/{u}`.
- ROOT CAUSE of "no options data": not a wrong endpoint — the chain
  snapshot (quotes/greeks/IV/OI) is **not included in Options Basic**
  (Starter+ only; per-tier table in the chain-snapshot doc). But Basic DOES
  include two endpoints the platform never called: contracts reference
  `/v3/reference/options/contracts` and previous-day bar
  `/v2/aggs/ticker/{O:...}/prev` — real EOD options data we were leaving on
  the table.

**Built:**
- `MassiveProvider.get_option_contracts` (reference rows, expiry-window
  filtered, next_url paging) + `get_option_prev_bar` (EOD OHLCV/vwap, None
  when the contract didn't trade — honest absence). `probe_capabilities`
  now probes `option_contracts` SEPARATELY from `option_chain`: a Basic
  plan honestly reads chain=false, contracts=true.
- `GET /api/watchlist/{ticker}/options/eod`: expirations summary (grouped
  reference), front-focus expiry (first ≥14 DTE), nearest-to-ATM
  call+put previous-session bars. Budget-aware: ≤1 contracts call + ≤4
  /prev calls per uncached load, cached per (ticker, Eastern day) — inside
  the Basic tier's 5 requests/minute. Payload states
  `data_recency: end_of_day`, the spot-reference provenance, and
  `missing_on_this_plan` (quotes/greeks/IV/OI) — named, never approximated;
  the §9 selector does NOT run on EOD data.
- Stub equivalents (contracts follow the caller's window; prev bar is a
  pure function of the OCC ticker via the same BS helper).

**Tests:** `tests/test_options_eod.py` (3: watchlist gate, full contract
incl. missing-capability naming and front-focus rule, day-cache) + probe
test updated. Full suite **924 passed, 1 skipped**.

**LIVE-VERIFIED with the user's real Basic key:** capabilities now
`option_chain: false, option_contracts: true`; RDW returned **8 real
expirations** and real ATM prev-session bars (e.g. O:RDW260828C00013500
close $1.10 vol 101; C00013000 close $1.35 vol 202) — the first real
options data the platform has served on this plan.

**Limits (honest):** EOD view is reference + last-session prices only;
live §9 contract selection still requires Options Starter+ for
quotes/greeks/IV/OI — the UI and payload both say exactly that.

## 2026-08-12 (13) — UPGRADE Phase H: closure verification — §52 acceptance review

**Verification runs (all green):**
- Backend suite: **921 passed, 1 skipped**.
- Frontend: §47 native-dialog scan OK · `tsc --noEmit` clean · production
  build clean (dev server stopped first — the .next collision from
  iteration 19 cannot recur) · all 10 routes 200 against the live gateway.
- Docker deployment functional (gateway rebuilt + redeployed every
  iteration today; migrations 013/014 applied to the live DB).
- Repo-root CI workflow added (`.github/workflows/ci.yml`): backend job
  (pytest) + UI job (npm ci → §47 scan → typecheck → production build).
  The services/.github copy remains for a services-rooted checkout.

**§52 acceptance checklist — assessed line by line:**
- ✓ Score formula visible & reproducible (§5 modal; formula with real
  numbers; versions shown)
- ✓ Contributions reconcile EXACTLY (by construction; pinned at lib and
  wire level)
- ✓ Thresholds/classifications visible (§7 bands + §8 legend derived from
  classifier params)
- ✓ Direction ≠ Tradeability (Layer 2; direction-agnostic by signature)
- ✓ Strong Bull → No Trade valid & explained (§10 live-verified on RDW)
- ✓ Quant labeled DATA-DRIVEN / deterministic; ✓ LLM labeled
  LLM-GENERATED; ✓ LLM sources + timestamps (§38 three-line provenance)
- ✓ LLM cannot override quant/risk vetoes (Phase 8 authority-boundary
  tests + read-only catalyst surface)
- ✓ Watchlist symbols generate plans; ✓ pool NOT required for research;
  ✓ Apply promotes to pool; ✓ Apply never trades (order_placed:false
  pinned); ✓ execution needs explicit enablement; ✓ execution re-runs
  live gates (execution-mode chain unchanged)
- ✓ Instrument selection explained (§8 rationale + Decision Summary);
  ✓ contract filters explained (§9/§25 quote + OCC identity)
- ✓ Entry AND exit visible before Apply (§24 exit_plan)
- ✓ Native dialogs gone (§27; CI-enforced §47 scan); ✓ all confirmations
  application-styled (§28–§30 copy)
- ✓ Risk limits unchanged (§43 untouched all fourteen iterations);
  ✓ cash-account restrictions enforced (broker tests green); ✓ full audit
  trail (extended: PLAN_*, mode, execution_authorized, llm_model);
  ✓ Docker functional

**Honest gaps (documented, not blockers):**
- §3/§6: VWAP + market/sector-confirmation features not yet built — scores
  normalize over present groups (stated in code and payloads).
- §47 dialog COMPONENT tests (focus trap/ESC/loading): no UI test framework
  in this repo; covered by the static scan + the shared Modal
  implementation. Adding vitest+RTL is future work.
- §31 Toast system not built (no blocking dialogs are used for success —
  inline notes/banners serve); §32 severity applied via shared badge maps.
- §40 REVIEWED/EXPIRED states have no UI/sweep yet; §14 decision card is
  implemented as stacked panels rather than one literal card (spec allows).
- Vol regime is CONDITION (unknown) in the analysis-view tradeability until
  option data is wired there / the Massive plan unlocks chains.

**The §48 end-to-end scenario** is enabled at every step (LLM recommend →
approval dialog → watchlist → quant+LLM panels → research plan without pool
→ review incl. exit plan → Apply dialog → ACTIVE + pool + audit → separate
trading enablement → execution re-gates live). Steps 1–16 verified across
iterations; 17–19 (enable + live execution) remain user actions on their
account.

## 2026-08-12 (12) — UPGRADE Phase F (backend): §24 exit plan rides on every preview

**Purpose:** §24 "the user should understand how the position will be exited
before applying the plan" — the exit rules become part of the plan payload,
never a surprise after entry.

**What changed:** `run_gate_chain` assembles an `exit_plan` block on EVERY
preview (research and execution): signal invalidation (SIGNAL_FLIP +
SIGNAL_DECAY at the exit edge threshold), the concrete hard stop (stock:
entry − 2×ATR14 with the actual dollar numbers; option: premium hard stop
% with the entry-premium and stop-level dollars), ATR trail rule, time
stop, DTE exit (options only), and `profit_target: null` — V1 has no fixed
profit target and SAYS so rather than inventing one. Every value comes from
`ExitParams` — the SAME engine that monitors open positions — plus the
entry/stop numbers the chain already computed; honest nulls where the chain
vetoed before an entry existed.

**Tests:** preview contract extended (§24 block present, invalidation/trail/
time-stop populated, profit_target honestly null). Full suite **921
passed, 1 skipped**. Live-verified: RDW's preview shows "hard stop at
$11.39 (entry $13.51 − 2×ATR14 $2.12)" from real Massive bars.

**Next:** Phase H — closure verification sweep (§52 checklist).

## 2026-08-12 (11) — UPGRADE Phase E: LLM catalyst surface (§11/§25/§38) — stored interpretation, cited, model-stamped

**Purpose:** give the symbol page a logically separate LLM catalyst context
(§11) with honest provenance (§25) and freshness (§38) — WITHOUT any new
LLM call path: generation remains exclusively the grounded recommendations
refresh (Phase 8); this surface only READS what was stored.

**What changed (semantics):**
- **`Recommendation.llm_model`** (migration 014): each generated
  interpretation now records `provider/model` AT GENERATION TIME. Pre-
  upgrade rows stay "" — an honest unknown, never backfilled from current
  settings (which may have changed since). Serialized on the
  recommendations API too.
- **`GET /api/watchlist/{ticker}/catalyst`** — read-only assembly:
  `llm` = the latest stored interpretation for the ticker (all §11 fields —
  sentiment/impact/novelty/source-reliability/horizon/catalyst type/reason
  codes/summary/evidence — plus `generated_at`, `model`, `status`);
  `articles` = stored news citing the ticker (verbatim provenance fields);
  `latest_source_published_at` = the newest source timestamp across
  citations and cited articles (§38: shown NEXT TO generated_at so an old
  summary can never read as live). Honest empties: `llm: null` /
  `articles: []`. Watchlist-gated 404 like every research surface. The
  payload carries `generated: true` — the §25 label flag; no market-derived
  number appears in this response.

**Risk implications:** none — read-only; the §12/§46 authority boundary is
untouched (this surface can't create, promote, or execute anything).

**Tests:** `tests/test_catalyst.py` (3): watchlist gating; honest empty
state; post-refresh contract with model recorded at generation (stub),
score ranges, parseable timestamps, non-empty evidence. Full suite
**921 passed, 1 skipped**.

**Live-verified:** migration 014 applied; deployed; RDW (no stored
interpretation, no cited articles among the 50 ingested) answers the exact
honest empty state.

**Next:** Phase F — Trade Plan page reorganization (§34/§35 progressive
disclosure; §24 exit plan surfaced in the plan; gate chain → Advanced
Decision Trace).

## 2026-08-12 (10) — UPGRADE Phase B: §6 grouped score weights — the explicit, versioned formula change

**Purpose:** §6 "Do not assume every feature deserves identical weight":
replace the v0 equal weights with the §6 grouped research defaults, AS a
visible versioned change — every §41/§42 mechanism built in earlier
iterations exists precisely so this commit could not happen silently.

**What changed (semantics — this IS a scoring formula change):**
- `DirectionalParams` default weights are now the §6 groups, split evenly
  inside each group: Trend/SMA alignment 20 (three close-vs-SMA components
  at 20/3 each), SMA slope 10, Market structure 20 (pivot_structure), MACD
  10 (cross + zero at 5 each), RSI 5, Volume 10. The §6 VWAP (10) and
  market/sector-confirmation (15) groups have NO engine features yet —
  scores normalize over the weights present, so the implemented groups keep
  §6's relative proportions; documented gap, never faked.
- `weights_version` bumped: "score-weights-v0-equal" →
  **"score-weights-v1-grouped"** (history noted in the field docstring).
- Everything downstream (§41 plan versions, §42 config-drift detection,
  /api/config, analysis payloads, audit) picks the new version up from the
  single source of truth — zero other code changes.

**Semantic shifts observed (why the tests moved):** structure and trend now
dominate; MACD/RSI dilute less when absent. Frozen-stub examples: GOOGL
+44.4 MODERATE → +33.3 WEAK; JPM -44.4 → -20.0 (below the weak floor);
live RDW 77.8/11.1/+66.7 STRONG_BULL → 60.0/26.7/+33.3 MODERATE_BULL.

**Tests (12 updated/re-characterized, all documented in-place):**
- Mechanics tests (weight doubling ×2) now pin an EXPLICIT equal-weight
  parameter set — they verify arithmetic, not defaults.
- Flat-oscillation: exact bull==bear symmetry was an equal-weights artifact
  (different components trigger per side); now asserts NEUTRAL with the
  residual edge far inside the band.
- Exit-engine hand-computed edges re-derived under grouped math (40/75 →
  53.3, 30/75 → 40.0) with the arithmetic in comments.
- Characterization re-picks: broker option/clamp tests moved GOOGL→**VZ**
  (+42.2 MODERATE, NEUTRAL_RANGE, NORMAL vol, ~$25 underlying keeps 2
  contracts inside the MODERATE 0.75% budget); vol-veto cell moved
  JPM→**PLTR** (-55.6 MODERATE, MILD_BEAR). Selection made by scanning the
  frozen stub universe (regime/scores/tier/vol per ticker) — the scan
  method is reproducible from this entry.
- Version pins updated (edge-classification test, §42 drift test now drifts
  to a synthetic v2 string).

**Verified:** full suite **918 passed, 1 skipped**. Deployed; §42 fired
LIVE exactly as designed: plan #1 (generated under v0-equal) now reads
`revalidation_required: true` with `config_changed` naming
v0-equal→v1-grouped — and apply would refuse it until revalidated. The
formula change is visible, auditable, and blocked from silently driving
decisions made under the old formula.

**Risk implications:** scoring layer only; gates/risk limits/permissions
untouched. Weaker signals now budget smaller or fall below the weak floor —
strictly more conservative in the observed cases.

**Known limitations:** VWAP + market/sector-confirmation features (§3
inputs list) still to be built to complete the §6 grouping; backtests over
v0 records remain labeled with their own stored version (reproducibility
preserved).

**Next:** revalidation UI (badge + Revalidate button), or Phase E LLM
catalyst panel separation (§11/§25/§38).

## 2026-08-12 (9) — UPGRADE §42: plan staleness revalidation — stale research cannot become ACTIVE

**Purpose:** §42 "Do not let old plans become orders": compare
plan-generation state vs current state on every read, refuse to APPLY stale
research, and provide the one-click "Recompute".

**What changed (semantics):**
- **`revalidation` block on every plan payload**, computed fresh on EVERY
  read (never stored): `stale_market_data` (the plan's `market_data_as_of`
  lags the last expected trading day by more than
  `PLAN_STALENESS_TOLERANCE_TRADING_DAYS` — configurable, default 1, the
  same unmodeled-holiday tolerance as tradeability; unknown/unparsable
  as-of fails CLOSED), `config_changed` (any §41 version differs from the
  currently active configuration — a formula change invalidates
  reproducibility; per-key plan-vs-current values reported), and
  `revalidation_required` = either.
- **Apply refuses stale plans**: 409 `PLAN_REVALIDATION_REQUIRED` with the
  full revalidation detail. Stale research can no longer transition to the
  ACTIVE plan — the §42 gap between "generated" and "applied" is now closed
  at the plan level too (execution was ALWAYS protected by the §21 live
  chain; this closes the authorization-state path).
- **`POST /api/plans/{id}/revalidate`** — §42 "Recompute": re-runs the
  research chain NOW with the plan's exact parameters, persists a NEW
  GENERATED plan (audited with `revalidated_from`), old plan untouched;
  response includes a previous-vs-fresh comparison surface (old verdict,
  old veto gates). The fresh plan applies via the normal §19 path.
- conftest raises the tolerance suite-wide (same pattern and reason as
  MAX_BAR_AGE_DAYS: the stub universe is frozen); staleness tests restore
  the real tolerance locally.

**Risk implications:** strictly tightening — a previously appliable stale
plan now requires an explicit recompute; no execution-path change.

**Tests:** two new (8 total in test_trade_plans.py): frozen-stub plan with
real tolerance → read reports stale, apply 409s with the code, revalidate
creates a linked fresh plan that applies; version-drift (monkeypatched
current versions) → config_changed named per key, apply refuses. Full suite
**918 passed, 1 skipped**.

**Live-verified:** deployed; RDW plan #1 (as_of 2026-08-11 == last expected
trading day) reads `revalidation_required: false` — an honestly CURRENT
plan is not nagged.

**Known limitations:** UI does not yet render the revalidation state or the
Revalidate button (next UI pass); EXPIRED lifecycle (§40) still unused —
a scheduled expiry sweep could layer on the same staleness arithmetic.

**Next:** Phase B — grouped score weights (§6) as the explicit versioned
formula change (score-weights-v1-grouped + characterization-test updates),
which will also exercise the new config_changed revalidation path for real.

## 2026-08-12 (8) — UPGRADE Phase D: research plan persistence + lifecycle (§19/§40/§41)

**Purpose:** make human review the persisted transition into the executable
universe (§18): Generate → review → Apply → ACTIVE + Trading Pool, with
execution still explicitly disabled.

**What changed (semantics):**
- **`trade_plans` table** (migration 013, `TradePlanRow`): stores the
  COMPLETE §16 research preview exactly as the user reviewed it, plus §41
  version metadata (`score_weight_version` / `edge_classification_version` /
  `tradeability_version` — only versions that exist; nothing invented),
  `market_data_as_of` (the last bar the research saw), lifecycle status,
  `superseded_by` linkage, and who/when.
- **`PlanStatus`** (§40): GENERATED → (REVIEWED) → ACTIVE; SUPERSEDED /
  CANCELLED; DRAFT/APPLIED/EXPIRED reserved. One ACTIVE plan per symbol —
  applying a new plan supersedes the old one, both sides audited.
- **New API `/api/plans`:**
  - `POST /generate` — watchlist-gated, NOT pool-gated (§15); runs the
    research chain and persists GENERATED. A NO TRADE verdict stores too
    (§17). Audited PLAN_GENERATED (USER).
  - `POST /{id}/apply` — the §19 user approval, one transaction: supersede
    old ACTIVE (PLAN_SUPERSEDED) → promote to Trading Pool if absent with
    the SAME §4.3 promotion checks as the direct promote endpoint (failed
    checks 422 unless acknowledged; override permanently audited;
    `trading_enabled=False` always) → plan ACTIVE + applied_at →
    PLAN_APPLIED (USER). Response states the §19 outcome explicitly:
    `trading_pool: true · trading_enabled: false · order_placed: false`.
  - `POST /{id}/cancel`, `GET /api/plans[?ticker=]`, `GET /{id}` with
    lifecycle guards (409 on applying a cancelled/active plan, etc.).
- Audit actions added: PLAN_GENERATED / PLAN_APPLIED / PLAN_SUPERSEDED /
  PLAN_CANCELLED.

**Risk implications:** none to execution authority — apply mutates
AUTHORIZATION state only (pool membership, disabled), never orders;
§4.3 promotion safety runs unchanged inside apply; §21's execution chain
still re-runs live at any order attempt.

**Tests:** `tests/test_trade_plans.py` (6): §45 proofs — generate without
pool membership (research payload + versions stored), apply → pool
disabled + zero Order rows + full USER audit chain, unacknowledged failed
checks 422 and change nothing, supersede linkage both directions, cancel /
double-apply guards, list/get. Full suite **916 passed, 1 skipped**.

**Live-verified:** migration 013 applied to the live DB; deployed gateway
generated plan #1 for RDW from real Massive data — GENERATED, versions
recorded, market_data_as_of 2026-08-11, research verdict NO TRADE (REGIME
TRANSITION veto) honestly stored (§17). Apply deliberately NOT exercised
live — promoting RDW into the user's Trading Pool is their decision.

**Known limitations:** REVIEWED state not yet set by any endpoint (UI
review flow, Phase D UI); §42 plan-staleness revalidation
(PLAN_REVALIDATION_REQUIRED) not yet implemented; exit plan not yet embedded
in the stored plan (§24 — the exit engine exists; surfacing it in the plan
payload is Phase F).

**Next:** Phase D UI — Generate/Apply Plan flow on the symbol page with the
§30 consequence copy in a ConfirmDialog built on the Modal (starts Phase G),
plan status on the Watchlist rows (§33).

## 2026-08-12 (7) — UPGRADE Phase C: research/execution split (§15/§16) — pool membership no longer gates research

**Purpose:** the workflow change at the heart of the upgrade: a Watchlist
symbol must produce a COMPLETE research trade plan; Trading Pool membership
is an execution authorization, not a research prerequisite (§15). Research
approval ≠ execution approval (§20).

**What changed (semantics):**
- `run_gate_chain` gained `mode: "research" | "execution"`.
  - **research** (what `POST /api/orders/preview` now runs): the §16 chain
    starts at DATA_QUALITY — TRADING_POOL_AUTHORIZATION is NOT a gate and
    cannot veto. The pool/per-symbol/kill-switch facts are still evaluated
    and reported in a new `execution_authorization` payload block
    (`authorized` + the three individual facts + `missing[]` naming every
    unmet authorization verbatim).
  - **execution** (the approve path, explicitly pinned): unchanged — gate 1
    vetoes exactly as before. §43 holds: no Trading Pool bypass, no
    kill-switch bypass; §42's "no rejected ticker may produce an order" is
    enforced where orders are made.
- Preview payload gains `mode` + `execution_authorization`; the RISK_DECISION
  audit event now records `mode` and `execution_authorized` (§36 — the audit
  trail distinguishes research reads from execution attempts).
- The kill switch pauses TRADING, not research (§18 intent): with the switch
  engaged, preview still researches; only approve refuses.

**Risk implications:** execution authority is UNCHANGED — the split moves a
check out of the RESEARCH read path only. The approve path re-runs the full
execution chain live (§21/§42) regardless of what any research plan said.

**Tests:** preview property tests rewritten to the §45 contract:
watchlist-only symbol runs the full research chain with real signal numbers
(gate[0]=DATA_QUALITY, veto_gate ≠ TRADING_POOL_AUTHORIZATION, audit
mode=research); disabled pool symbol still researches (authorization block
names the missing enablement); kill-switch-paused still researches; fully
authorized preview shows authorized=true. Alert test moved its veto source
to the approve path (execution mode, where the veto now lives). Execution
tests untouched and green — approve still 422s at gate 1 for
watchlist-only symbols. Full suite **910 passed, 1 skipped**.

**Live-verified:** deployed gateway, RDW (watchlist-only, pool-less, kill
switch engaged): mode=research, execution_authorization honestly lists both
missing authorizations, and the research chain runs to a LEGITIMATE market
verdict — REGIME FAIL (TRANSITION defaults to NO TRADE) — which is §17's
"research plan exists even when action = NO TRADE" in the flesh.

**Known limitations:** UI does not yet render mode/execution_authorization
(next); the plan lifecycle (DRAFT→…→ACTIVE, §40) and Apply Plan (§19) are
Phase D; LLM catalyst context is not yet part of the research chain (§16's
LLM step lands with Phase E).

**Next:** Trade Plan UI — Execution Authorization section (§34) fed from
`execution_authorization`, then Phase D (Apply Plan / plan lifecycle).

## 2026-08-12 (6) — UPGRADE Phase A-2: Tradeability layer (§9/§10) — direction ≠ permission

**Purpose:** Layer 2 of the four-layer decision architecture (upgrade §2).
Directional strength must not equal permission to trade: the environment
gets its own explicit verdict, so "STRONG BULL, but NO TRADE" becomes a
first-class explainable state instead of an apparent contradiction (§10).

**What changed (semantics):**
- New `libs/trading_core/tradeability.py` — pure, deterministic,
  **direction-agnostic** (the function signature contains no direction at
  all, which IS the §9 rule, pinned by test): `assess_tradeability(bar_count,
  stale_trading_days, market_regime, symbol_regime, vol_regime,
  vol_unavailable_reason, params)` → `TradeabilityDecision(state, reasons,
  checks, version)`. States: TRADEABLE / CONDITIONAL / BLOCKED /
  DATA_INSUFFICIENT (`TradeabilityState` in models/enums).
- **Verdict precedence** (first match): INSUFFICIENT (bars < min_bars,
  staleness > tolerance, unclassifiable regime) → BLOCK (TRANSITION regime
  on either market or symbol level, EXTREME vol) → CONDITION (HIGH vol,
  UNKNOWN vol with the reason stated) → TRADEABLE. You cannot call an
  environment BLOCKED on data you don't have — but ALL problems are still
  individually reported in `reasons` (§26 evidence, pinned).
- **Unknown volatility degrades, never blocks:** with option data
  plan-gated, the check reports CONDITION with the honest reason and "stock
  instruments only, execution gates still apply". The §10 execution chain
  keeps its own volatility gate — this layer is research posture, not
  execution authority (§43 untouched).
- `TradeabilityParams` (min_bars 200, max_stale_trading_days 1 — the
  unmodeled-holiday tolerance, blocked_regimes=(TRANSITION,), version
  "tradeability-v1") — research parameters, exposed at `/api/config`.
- **API:** analysis response gains a `tradeability` block (state / reasons /
  all checks with per-check PASS|CONDITION|BLOCK|INSUFFICIENT status /
  version); overview rows gain `tradeability`. New
  `_stale_trading_days` weekday-lag helper beside the existing freshness
  arithmetic. Market-regime read (shared SPY helper, once per overview
  request) is best-effort: a fault degrades the verdict honestly to
  DATA_INSUFFICIENT rather than failing the endpoint or inventing a regime.

**Risk implications:** none to execution — the layer is advisory/research;
order gates unchanged. It cannot AUTHORIZE anything (TRADEABLE is necessary,
not sufficient); it can only explain.

**Tests:** `tests/test_tradeability.py` (12): all four states, precedence
chain, holiday tolerance boundary, §10 RDW scenario (TRANSITION + EXTREME →
BLOCKED with both causes named), custom-params drift. Contract tests updated
(analysis tradeability block incl. reasons==non-PASS-checks invariant,
overview row key, config group). Full suite **910 passed, 1 skipped**
(was 898).

**Live-verified:** deployed gateway reproduces the upgrade doc's own
motivating example on real Massive data — RDW: signal STRONG_BULL +66.7
while tradeability = BLOCKED (SYMBOL_REGIME TRANSITION) + CONDITION
(vol unknown, plan-gated), each with its stated reason.

**Known limitations:** vol_regime is not yet fed from the options path in
the analysis view (honest CONDITION until wired or the Massive plan unlocks
chains); liquidity/event-risk/bid-ask checks are future inputs (§9 "may
include"); UI does not yet render the tradeability verdict.

**Next:** Tradeability UI (context strip verdict + §10 explanation line +
§14 decision-card market-state grid), then Phase C research/execution split
(move TRADING_POOL_AUTHORIZATION out of the research gate chain, §15/§16).

## 2026-08-12 (5) — UPGRADE Phase A-1: Edge classification bands + exact contribution breakdown + versioned score config

**Purpose:** first step of the Decision Intelligence upgrade
(`prompts/upgrade_2026-08-12.md`), Phase A — make the deterministic score
layer classifiable, reconcilable and versioned WITHOUT changing any score
value or execution path.

**What changed (semantics):**
- **§7 classification layer** — new
  `libs/trading_core/signals/classification.py`: `DirectionalEdgeClass`
  enum (STRONG_BULL … STRONG_BEAR, in models/enums) + `classify_edge()` over
  configurable `EdgeClassificationParams` (strong 50 / moderate 25 / weak 15,
  all inclusive, mirrored). STRONG additionally requires the same-side score
  ≥ 70 (§7 minimum-side-score rule); a strong-band edge that fails it
  degrades to MODERATE — documented, tested. Params carry
  `version="edge-class-v1"` and are exposed under `/api/config`
  (`edge_classification_params`).
- **§8 legend from the source of truth** — `edge_legend(params)` derives the
  seven UI bands from the classifier's own params; a test pins that sampling
  each band reproduces its label and that bands tile [-100, +100] gap-free.
  The analysis payload ships it (`signal.edge_legend`) so the UI never
  hardcodes thresholds.
- **§5 contribution breakdown, §44 exact reconciliation** — every
  `SignalComponent` now carries `max_contribution` (weight share in points)
  and `contribution` (that if triggered, else 0). A side's score IS
  `sum(contributions)` — reconciliation holds by construction, not approx.
  Subtlety fixed en route: Python 3.12's builtin `sum()` uses Neumaier
  compensated summation, so the engine now uses builtin `sum()` too —
  a consumer summing the displayed numbers reproduces the score
  bit-for-bit (pinned at both lib and wire level).
- **§6 version metadata** — `DirectionalParams.weights_version =
  "score-weights-v0-equal"` names the CURRENT (equal-weight) configuration;
  it rides into every `DirectionalResult`, the analysis payload, and
  `/api/config`'s `directional_params` (backtests recording params via
  asdict pick it up for free). Grouped §6 weights are a LATER deliberate
  formula change (Phase B) — this iteration only makes versioning exist, so
  that change will be visible instead of silent.
- **API additive only:** `signal` block gains `deterministic: true` (§3
  provenance flag), `classification`, `weights_version`,
  `classification_version`, `edge_legend`, per-component contributions;
  overview rows gain `edge_class` (§33 column). No field removed or
  renamed; scores unchanged (equal weights preserved).

**Risk implications:** none to execution — classification is labeling only;
Tradeability/Risk gates untouched (§43). No weight value changed.

**Tests:** new `tests/test_edge_classification.py` (27 cases: band
boundaries incl. both inclusive edges and the degrade rule, custom-params
drift test, legend/classifier agreement, exact reconciliation on real scorer
output both directions, no-data neutrality). Contract tests extended:
analysis signal block asserts §44 reconciliation AT THE WIRE; config pins
the new group; overview pins `edge_class`. Full suite **898 passed, 1
skipped** (was 871).

**Live-verified** on the deployed gateway (:8010): RDW reads Bull 77.78 /
Bear 11.11 / Edge +66.67 → STRONG_BULL — the upgrade doc's own §4 example —
and the wire contributions sum EXACTLY to the displayed score; legend and
versions present.

**Known limitations:** UI does not yet render classification/legend/
contribution table (next iteration); tradeability layer (§9) not yet built;
weights still equal (grouped weights = Phase B with characterization-test
updates).

**Next:** UI — Symbol Analysis quant panel: DETERMINISTIC label + edge badge
with §8 legend + §5 contribution table behind "How is this calculated?".
Then Phase A-2: Tradeability layer (§9/§10).

## 2026-08-12 (4) — Phase 8: news-grounded LLM recommendations

The original plan's Phase 8 (news ingestion → dedup → LLM enrichment →
recommendation pool → user review) is now implemented — and the enrichment
is GROUNDED, closing the honesty gap where the LLM previously proposed
candidates from its own training memory with invented "evidence":

- **News ingestion (real articles only):** `MassiveProvider.get_news`
  (`GET /v2/reference/news`) parses id/title/publisher/timestamp/url/tickers/
  description VERBATIM; rows missing any citable field are skipped, never
  patched. 403 → CapabilityNotAvailable; `probe_capabilities` gained a
  `news` entry. The stub provider serves synthetic articles (stub:// urls,
  "SYNTHETIC" markers) reachable only under MARKET_DATA_PROVIDER=stub.
- **Dedup:** migration 012 `news_articles` with the provider's own article
  id UNIQUE — re-fetching the feed inserts nothing (pinned by test).
  Ingestion is audited (new NEWS_INGESTED action).
- **Grounded enrichment:** the LLM protocol gained
  `enrich(articles, ...)` — the model receives ONLY the stored articles and
  must cite them by exact url. Implemented for OpenAI (strict JSON-schema
  response, grounded system prompt) and the stub (deterministic, same
  citation contract).
- **Server-side grounding validation (the safety boundary):** the refresh
  route DROPS any draft whose evidence cites a url not in the stored batch,
  or whose ticker does not appear in a cited article's own ticker list —
  both reported in `skipped` with reasons. A fabricated citation cannot
  reach the user even if the model lies (pinned by a LyingProvider test).
- **Honest failure surface:** refresh requires BOTH providers; a plan
  without the news endpoint answers 503 `NEWS_NOT_AVAILABLE` naming the
  gap; zero stored articles is an honest no-op (`created: []`).

**Verified:** full suite **871 passed, 1 skipped** (9 new Phase 8 tests);
gateway deployed. LIVE PROOF the same day: the user reconnected all three
providers through the Settings UI, their Massive plan turned out to INCLUDE
news, and the first real refresh ingested 50 articles (The Motley Fool et
al., same-day) and produced 5 grounded PENDING recommendations from real
OpenAI — every evidence link a real article URL. The live run also exposed a
concurrency race (two simultaneous refreshes both passed the existence check
and collided on the news_articles UNIQUE constraint): fixed with a per-loop
refresh lock + an IntegrityError backstop, pinned by a concurrent-refresh
test.

## 2026-08-12 (3) — ZERO local copy of the account: cash is read live from Alpaca

The user tightened the principle once more: the platform must not even STORE
a copy of the real account — the number on screen must be Alpaca's, live.
Implemented end to end (real-broker mode; the dev/test simulator keeps its
own ledger, which paper_initial_cash now exclusively defines):

- `GET /api/portfolio/risk` reads cash from the broker ON EVERY REQUEST;
  NAV = live cash + platform positions' value. No Portfolio row is ever
  created or written in broker mode — the stored copy was DELETEd from the
  live DB at the user's direction and verified to stay gone across reads.
- Order paths and the order-sync sweep no longer debit/credit any local
  cash: the debit/credit happens in the real account at the broker, and the
  platform's own economic record is the position (avg price, max_loss,
  realized P&L) plus the audit trail's exact incremental numbers.
- §14 sizing uses the broker's LIVE cash directly (fetched fresh, fail-closed
  on fetch failure, never buying power) — the min(local, broker) clamp is
  gone because there is no local number to clamp against.
- Reconciliation no longer compares cash (nothing exists to disagree with the
  broker's number); its job is the POSITION ledger, which the platform does
  own. First-connect/first-sight cash adoption removed — nothing to adopt
  into.

Tests reworked to the new contract: cash-conservation assertions became
realized-P&L + position-economics assertions (the ×100 multiplier round trip
is still pinned to the cent — on `realized_pnl`); new tests pin "no Portfolio
row is ever created in broker mode" and "a broker-side cash change shows on
the very next read". **862 passed, 1 skipped.** Live-verified: portfolio
table 0 rows and STAYS 0 across reads; /api/portfolio/risk == broker/status
cash exactly ($107,778.68, same instant).

## 2026-08-12 (2) — The account IS the venue's: no more default-cash display

User caught the last fabrication: before any provider was connected the
Dashboard still showed NAV/cash — because `GET /api/portfolio/risk` called
`get_or_create_portfolio`, which SEEDED a $100,000 default row just to have
something to display. Fixed at the root:

- **No venue → no account → nulls.** With no broker connected (and not in
  the dev-only simulated mode) the risk view creates NOTHING and reports
  every account number as null, plus a `venue` block naming the reason. A
  stored ledger row without a venue is also not displayed — a number nobody
  can act on reads as an account that does not exist.
- **Real broker, first sight → materialise from the LIVE account.** If the
  broker is connected and no local row exists, the row is created FROM the
  broker's actual cash (audited `FIRST_SIGHT_CASH_ADOPTION`) — never from
  `paper_initial_cash`, which is now exclusively the simulator's spec.
- **Reconciliation reads the ledger, never creates it**: no row → local cash
  null, cash comparison honestly skipped, position comparison still runs.
- **UI**: Dashboard NAV/Cash/Heat tiles render "—" + "broker not connected";
  the Risk page shows a whole-page venue panel when there is no account
  (limits are configuration and stay visible the moment trading resumes).

Tests updated to the new contract (unconfigured install: cash IS null; the
§14 clamp test now sees local == broker == real cash, which is the point).
**862 passed, 1 skipped.** Live-verified the exact complaint: disconnect →
Dashboard numbers vanish with the reason; reconnect → the real $107,778.68
returns from the stored ledger.

## 2026-08-12 — Fresh start on the real ledgers + UI-managed provider config

**Dev residue purged (user-directed).** The user confirmed the local
watchlist/activity/portfolio were development artifacts they never created:
the database volume was reset and re-initialised from migrations — the
platform now starts from the honest "nothing has happened" state, and every
number on screen from here on is Massive data or an Alpaca paper fact.

**Provider configuration moved from .env to the UI (runtime_config layer).**
Migration 011 + `apps/gateway/runtime_config.py`: eight whitelisted keys
(market data / LLM / broker selection + credentials) stored in the DB,
loaded over the environment at startup and on every change, Settings cache
rebuilt — every existing get_settings() call site picks changes up with no
restart. New API: `GET/PUT /api/config/providers`. Secrets are WRITE-ONLY
(presence booleans out, never values; audit records changed KEYS only, via
the new CONFIG_CHANGED action). The Alpaca base URL is deliberately NOT
configurable — the paper-only guard stays structural. .env provider lines
are now blank with a pointer to the UI; infra values stay in .env.

**First-connect cash adoption.** Connecting a real broker while the local
ledger is completely empty (no orders, no positions — the fresh-start case)
adopts the broker's actual cash as the local baseline, audited. Executed
live through the exact UI flow: blank boot (all providers honestly
unconfigured) → API connect → all three providers CONFIGURED from the DB
alone → local cash adopted **$107,778.68** (the real paper account). After
adoption the live reconcile shows CASH MATCHED; the only remaining
mismatches are the user's own Alpaca-side manual positions
(MISSING_LOCALLY ×3), which clear as soon as they flatten them.

**The daily-rot class of test failures is dead.** Overnight the date roll
broke 10 tests: the stub's bar values were INDEX-based (a function of
position-in-window), so any window shift — a new day, a changed fetch count,
a trim — silently moved every value. The stub now generates each bar as a
PURE FUNCTION of (symbol, calendar date) — per-date RNG streams accumulated
from a fixed epoch (2022-01-03) — and the suite anchors the synthetic
universe at STUB_ANCHOR_DATE=2025-11-03 (conftest), where the characterised
tickers hold their documented verdicts (GW strong-bull → LONG_CALL, GOOGL
bull, JPM bear/moderate for the vol-caused NO_TRADE cell, driven through the
documented VOL_REGIME_PARAMS seam). Frozen bars are honestly "old" against
the real clock, so conftest raises the bar-age gate suite-wide (staleness
tests set their own) and the freshness-gauge test now asserts the TRUE age.
Also fixed: an env-isolation hole where "unset broker" harnesses popped the
env var and pydantic silently fell back to the developer's real .env.

**Verified:** full suite **863 passed, 0 skipped/failed**; provider config
survives gateway restarts (DB overrides load in lifespan); live reconcile
cash-matched. UI Connections panel building in a background agent.

## 2026-08-11 (8) — ALPACA PAPER IS LIVE; §34 Definition-of-Done assessment

**Milestone: both real providers are connected and verified.** The user
pasted their Alpaca paper keys; the gateway (recreated with the new env)
read the LIVE paper account: `is_paper: true`, account PA30BVA75LXD, cash
$106,052.68. `/api/config` now reports all three providers configured
(massive / openai / alpaca_paper). First LIVE reconciliation ran and behaved
exactly as designed: it found the real divergences — a local GOOGL 5-share
row (simulator dev-residue, zero broker-backed orders exist), the account's
own manual paper trades (**AAPL −23 short**, NVDA 1) and a $7,052.68 cash
difference — reported all four mismatches, and pulled the §18 kill switch.
The ledger-alignment decision is the USER's (three options presented; an
automated local reset was deliberately not performed — §18: no auto-correct).
Risk note flagged to the user: the AAPL −23 is a naked short in their paper
account; the platform itself can never produce one (§33).

**§34 Definition of Done — formal assessment:**
- Massive real data works — ✅ live (stocks history + realtime; options
  gated on plan, honestly surfaced via /api/market/capabilities)
- Alpaca Paper adapter works — ✅ live (account read, paper-only guard,
  reconcile, kill switch all exercised against the real API)
- Long Stock / Call / Put end-to-end — ✅ §30 acceptance test (real adapter
  over MockTransport; live-data variant pending options-capable plan)
- Forbidden shorting impossible — ✅ three-layer §33 enforcement + tests
- No SELL_TO_OPEN producible — ✅ structural (no code path) + DB CHECK
- Positions reconcile with Alpaca — ✅ live, including options by OCC symbol
- Partial fills handled — ✅ first-class, incremental-delta sweep, tests
- Risk and cash constraints enforced — ✅ §14 min(local, broker) clamp,
  fail-closed on account fetch failure
- Automatic exits send SELL_TO_CLOSE — ✅ exit sweep through the real
  adapter path (acceptance test steps 15–17)
- UI shows broker/data source truthfully — ✅ honest-absence panels,
  capabilities grey-out, PENDING_UPDATE, venue labels
- Full E2E acceptance passes — ✅ automated (deterministic data); LIVE run
  pending the user's ledger decision + options-capable Massive plan
- Docker Compose runs everything — ✅ db/redis/gateway (+frontend service;
  UI currently on the dev server by choice)

**Remaining to close the milestone completely:** the user's ledger-alignment
choice (A: audited local reset + manual flatten; B: Alpaca paper account
reset + local cleanup; C: stay paused), then a LIVE stock round-trip; the
LIVE options leg needs the Massive plan upgrade.

## 2026-08-11 (7) — §30 acceptance test lands (all 20 steps) + options reconcile by OCC symbol

**The §30 End-to-End Acceptance Test now runs green as an automated test**
(`tests/test_acceptance_e2e.py`): Watchlist → data → backtest COMPLETED →
Trading Pool (checks recorded) → enable → BULL signal → LONG_CALL → real
chain contract (server OCC symbol) → risk-sized quantity → INSTRUMENT +
RISK_APPROVAL gates → **BUY_TO_OPEN addressed to the OCC symbol at the real
AlpacaPaperBroker adapter** (MockTransport) → fill at the broker's price →
`GET /api/broker/reconcile` **in_sync with the broker's option holding** →
monitor HOLDS a healthy position → forced DTE condition → exit fires →
SELL_TO_CLOSE → fill → position CLOSED → realized P&L and cash reconcile to
the cent → the full audit chain (DATA_BACKFILL … EXIT_GENERATED) present.
Quantity is whatever the risk engine actually approved — the broker double
echoes submissions, it never scripts sizing. VENUE HONESTY note in the module
docstring: market data is the deterministic stub; the live-Massive variant of
steps 2/6/8 remains a manual acceptance once the plan includes option chains.

**Options now reconcile by OCC symbol (§30.13).** `_local_open_quantities`
had excluded option rows with a stale rationale ("options are not wired at
the broker" — they are, since Iteration B). Local option positions now
compare against the broker's holding of the same OCC contract symbol: equal
quantity is in_sync, an absent contract is MISSING_AT_BROKER and pulls the
§18 kill switch like any other divergence. Both directions pinned by tests.

**Backfill timezone hardening:** a provider series can run one date ahead of
the Eastern trading day (UTC-dated series just after midnight UTC) — the
backfill now fetches two extra bars so the complete-days trim still stores
exactly BACKFILL_DAYS.

**Verified:** full suite **855 passed, 1 skipped**; gateway rebuilt on :8010.
Remaining for the LIVE §30 run: Alpaca paper keys in services/.env (a file
monitor is armed — the loop resumes Iteration G the moment they appear) and
an options-capable Massive plan.

## 2026-08-11 (6) — Daily-bar freshness + second review wave (6 more confirmed, all fixed)

**Daily bars now stay fresh (§15).** `ensure_daily_bars` previously served
whatever the first backfill stored, FOREVER — every signal/regime/backtest
would age with it until the data-quality gate bricked trading. Now: stored
history is APPENDED to when the newest bar is older than the last expected
weekday (Eastern); refreshes are per-symbol throttled (30 min — a holiday
looks exactly like a missing bar); a refresh failure serves the stored real
bars; and **complete trading days only** — a bar dated today (Eastern) is
still forming and is never stored (backfill fetches one extra so the count
stays exact). Audited as DATA_BACKFILL `mode: "refresh"`. 6 new tests.

**Second adversarial review wave (sonnet dimensions: money-math,
state-machine, UI) — 6 findings, ALL verifier-confirmed (two by live
reproduction), all fixed:**

1. **[critical] Price-pending fills were silently dropped, both paths.**
   Alpaca can publish `filled_qty` before `filled_avg_price`; the poll stops
   on quantity alone. The approve/close paths advanced `filled_quantity` to
   the broker's number while booking NOTHING (no price → "zero fill"
   branch), and the sweep's delta (`new_filled − filled_quantity`) then read
   0 forever: a REAL fill invisible to the ledger, permanently — and the
   ticker free to double-buy. Fixed with one invariant:
   **`filled_quantity`/`fill_price` record what has been APPLIED to cash
   and positions.** A quantity-without-price outcome applies nothing, keeps
   the row non-terminal (even against a raw "filled"), audits the truth,
   and the sweep books the fill the moment the price publishes. The sweep
   holds such orders as retries (faults), never mismatches.
2. **[critical] `pending_cancel` mapped to terminal CANCELED.** A cancel
   REQUEST is not a cancel — the order can still fill at Alpaca, and a
   terminal local status dropped it out of the sweep's watch, making any
   last-moment fill invisible. Now maps to ACCEPTED (raw preserved).
3. **[critical, UI] Options positions crashed the Positions page** —
   `stop_price` typed non-null but the backend honestly sends null for
   options; `fmtUsd(null)` threw on render. Type + render guard fixed.
4. **[major, UI] Null `option_symbol` was reconstructed client-side** via
   `??` — fabricating an unvalidated OCC string exactly when the server had
   said "cannot build one". Fallback now fires only when the field is ABSENT.
5. **[minor, UI] `invalidateAfterTrade` didn't refresh `["orders-open"]`**,
   leaving PENDING_UPDATE stale ~15s after a close/exit.

**Verified:** full suite **853 passed, 1 skipped** (17 lifecycle/sweep tests
+ 6 bar-freshness tests among them); UI `tsc --noEmit` clean; gateway
rebuilt on :8010.

## 2026-08-11 (5) — Adversarial review findings: six real defects fixed

An adversarial review (concurrency dimension) surfaced 8 findings; verified
each against the code by hand. Six were real, all now fixed and pinned by
tests (suite: **844 passed, 1 skipped**):

1. **Duplicate protective sells** — every exit-monitor tick re-triggered the
   same exit while the previous sell sat ACCEPTED-unfilled (each tick minted
   a fresh client_order_id). Now: one closing order in flight per position —
   the sell path 409s `CLOSE_ALREADY_IN_FLIGHT` (the exit sweep reports it in
   `exits_failed` and simply waits for the order-sync sweep).
2. **Double-buy on retry** — the no-pyramiding check only saw OPEN positions,
   so an ACCEPTED-zero-fill buy (no position yet) let a retry place a second
   broker order for the same intent. Now: a non-terminal BUY on the ticker
   409s the approve.
3. **Closed-position fill corruption** — a late BUY fill delta was applied to
   its position even if that position had since been CLOSED (quantity
   incremented on a closed row). Now: reported as a §18 mismatch, nothing
   mutated.
4. **False-positive kill switch** — periodic reconciliation could pause
   trading on a divergence that an in-flight order fully explains (broker
   filled, sweep not yet applied). Now: reconciliation runs under the
   execution lock and answers `pause_deferred: true` (mismatches still
   REPORTED) while any order is in flight; the switch fires only when
   nothing in flight can explain the divergence.
5. **Premature orphan rejection** — a PENDING_SUBMIT row was settled REJECTED
   from a single broker 404, but a lookup moments after a lost-response
   submit can 404 transiently (broker-side lag) — and REJECTED licenses a
   re-approve that double-buys when the original quietly fills. Now: orphans
   must age past ORPHAN_GRACE_SECONDS (120s, ≈2 sweep cadences) before
   settling; held orphans are reported in `faults` with the age.
6. **Staged-mutation leak** — when the sweep detected a mismatch AFTER
   staging a broker-id adoption on the session, the next order's commit
   silently carried the un-audited write. Now: mismatch → session.rollback()
   before continuing.

Two findings were accepted tradeoffs, now documented in order_sync.py:
per-process asyncio lock (single-gateway deployment; UNIQUE constraints as
multi-process backstop) and lock-across-broker-RTT in the sweep (correctness
over latency; N bounded by the new one-in-flight guards).

NOTE: the review's money-math / state-machine / UI dimensions did not run
(model usage limit); money-math is separately pinned by the exact
cash-conservation tests, and the six fixes above came from the one dimension
that did run. Re-run the remaining dimensions when capacity allows.

## 2026-08-11 (4) — §25/§26/§27 payloads + open-orders API (+ a live schema gap caught)

**§25 trade-plan payload**: `proposed.contract` now carries the full quote —
`bid`, `ask`, `spread_pct`, `open_interest`, `volume` (all were already on
ContractQuote, just unsurfaced) — plus **server-built `option_symbol`** (the
exact OCC string the broker would be addressed with; null when unbuildable,
never guessed). §27: order/position `contract` blocks carry the same
server-built `option_symbol`, and order payloads now expose `position_id` —
the UI reconstructs nothing.

**§26 PENDING_UPDATE source**: new `GET /api/orders/open` lists every
non-terminal order (PENDING_SUBMIT / ACCEPTED / PARTIALLY_FILLED), local rows
only, never 503s. A position with an in-flight order is honestly "in flux"
rather than MATCHED/MISMATCH; the UI wiring (capabilities grey-out,
PENDING_UPDATE precedence, trade-plan quote fields) is running as a
background agent task.

**Live schema gap caught and closed**: the deployed database predated
migration 009's compose mount, so `orders.broker_order_id/broker_status/
filled_quantity` had never been applied live (tests never saw it — they
create_all from the ORM). Applied 009 manually, then verified EVERY ORM
column exists in the live schema programmatically — clean. Lesson recorded:
initdb-mounted migrations only run on FIRST boot; new migrations must be
applied to existing volumes by hand (as 010 was).

**Verified:** full suite **840 passed, 1 skipped**; live
`/api/orders/open` → `{"orders": []}`, positions/preview payloads healthy.

## 2026-08-11 (3) — Iteration F validated on REAL data + §16 capabilities API

**Iteration F is proven on live Massive data.** Through the running gateway:
AAPL added to the watchlist → first backtest request lazily backfilled **600
real Massive daily bars (2024-03-19 → 2026-08-10)** with the SYSTEM
DATA_BACKFILL audit → engine ran to COMPLETED with honestly unflattering
numbers (9 trades, PF 1.23 full-sample, **negative OOS return** from the
2025-11-19 split — real data does not owe us a good look). The F validation
list: no look-ahead was already pinned by property tests (truncation
invariance + equity-prefix identity); realistic fills by the three-tier
slippage model (fill at NEXT open, slippage both ways); spread filters live
in the option selector. **Point-in-time option contracts are honestly
deferred**: the current Massive plan includes no option data at all, and
nothing in the platform fabricates historical chains.

**§16 capability detection is now an API**: `GET /api/market/capabilities`
probes the live provider (`probe_capabilities`) and reports each capability
as true / false / error-string — verified against the real API, never assumed
from configuration. 300s TTL cache (a probe costs three provider calls),
`?refresh=true` bypass for right-after-a-plan-upgrade. Probeless providers
(stub) answer `capabilities: null` with a message — never a fabricated
all-true. Live verdict today: `stock_history: true, stock_realtime: true,
option_chain: false`. The UI can now grey out options features with the
reason visible instead of letting them 403 deep in a workflow.

**Verified:** full suite **839 passed, 1 skipped**; live endpoint returns the
probed verdict above; backtest record 1 (AAPL, COMPLETED, CONSERVATIVE fill
model) persisted with full audit chain.

**Next:** UI wiring for capabilities + the deferred backend payload gaps
(bid/ask on proposed contracts, server-side option_symbol, order-status
surfacing of PENDING_SUBMIT rows). Iteration G (§30 E2E acceptance) remains
gated on the Alpaca paper keys (user) and an options-capable Massive plan.

## 2026-08-11 (2) — Order lifecycle + order-sync sweep + periodic reconciliation (Iterations C/D)

**PENDING_SUBMIT is real (§11).** Both broker paths now write and COMMIT the
local order row BEFORE the submit leaves the process (migration 010): a crash
or network fault mid-submit leaves a durable PENDING_SUBMIT row instead of an
invisible broker order. A broker rejection settles the row to a stored
REJECTED (terminal, audited); a broker FAULT leaves it PENDING_SUBMIT for the
sweep. The order row also captures approval-time risk context
(`stop_distance`/`entry_edge`/`entry_bar_date`) and its `position_id`, so a
fill that lands after the request returned can still open the position with
the §10 chain's own parameters. The close-path order row now carries `opt_*`
(it previously could not identify the contract it sold — §44 rule 18 gap).

**Order-sync sweep (`apps/gateway/order_sync.py`).** Every non-terminal order
(PENDING_SUBMIT/ACCEPTED/PARTIALLY_FILLED) is looked up at the broker by our
client_order_id and settled to what actually happened: an order the broker
never saw becomes REJECTED `never_reached_broker` (safe to re-approve); a
submit whose response was lost is ADOPTED (broker id + real fills + position +
cash); incremental fills move cash by exactly
`new_avg*new_filled − old_avg*old_filled` (× multiplier), so cash conservation
holds to the cent across any number of partial-fill sweeps — pinned by test.
Sell fills late-credit proceeds and realized P&L and close the position. A
lookup fault changes NOTHING (a fault teaches nothing); a broker reporting
FEWER fills than recorded is reported as a mismatch for §18, never "fixed".
Runs under the shared execution lock; commits per settled order. Wired as a
background loop (ORDER_SYNC_INTERVAL_SECONDS, default 30s; cheap no-op without
a real broker) plus manual `POST /api/broker/sync-orders`.

**Periodic reconciliation (§13 Iteration D).** The reconcile core is extracted
to `run_reconciliation()` — one implementation shared by
`GET /api/broker/reconcile` and a new background loop
(RECONCILIATION_INTERVAL_SECONDS, default 300s). A material mismatch pauses
trading through the same §18 kill switch either way; the loop skips honestly
when there is no broker ledger to compare against.

**Verified:** 9 new tests in `tests/test_order_lifecycle_sync.py` (durability,
orphan settle, adoption, partial-fill cash conservation, late sell fill,
fault-leaves-alone, shrinking-fill mismatch, skip honesty, endpoint); full
suite **836 passed, 1 skipped**. Gateway rebuilt on :8010; both new surfaces
answer honestly in the current unconfigured-broker state
(`{"checked":0,"skipped":"NO_REAL_BROKER"}`).

**Next:** Iteration F (Massive-derived backtests: no look-ahead, point-in-time
contracts, spread/slippage filters) and §30 E2E acceptance — both still gated
on the user pasting Alpaca paper keys into services/.env (broker) and the
Massive plan including option chains (options selection).

## 2026-08-11 — Massive provider live + cash-account guide Iterations A/E landed

**MassiveProvider is real and probed against the live API.** `libs/market_data/
massive.py` (774 lines) implements daily bars (`/v2/aggs`, Eastern-date trim,
oldest-first), stock snapshots, index snapshots (`I:VIX` route), and the paginated
option-chain snapshot (`/v3/snapshot/options`, `next_url` capped at 8 pages, OCC
ticker cross-checked against `details` — mismatch rows are skipped, never guessed).
Auth is header-first with a one-time `apiKey` query fallback on 401; 403 maps to
`CapabilityNotAvailable` (plan-gated ≠ broken), 429 honours `Retry-After` once.
No fabrication anywhere: unquotable/greekless/expired rows are skipped, unknown
symbols are absent-with-warning, historical `as_of` raises. 30 new tests in
`tests/test_massive_provider.py` pin all of this over `httpx.MockTransport`.
One implementation fix while testing: the OCC ticker regex now accepts 8- or
9-digit strikes (Massive doc examples pad to 9; the unit is thousandths
either way, and the parse is only ever a cross-check).

**Live capability probe with the real key (§16):** `stock_history` ✅ (SPY 5 real
bars through 2026-08-10), `stock_realtime` ✅ (SPY 773.03), `option_chain` ❌ and
`indices` ❌ — the current Massive plan does not include those endpoints (HTTP 403
NOT_AUTHORIZED). The platform reports both honestly (VIX absent, chain 403 →
CapabilityNotAvailable); options selection needs a plan upgrade, no fallback.

**Guide §2/§8/§33 permissions + §14 cash clamp (agent-built, verified):**
`AccountPermissions` carries the six forbidden fields (short stock, naked short
call/put, covered call, CSP, margin) refused at construction; forbidden `ALLOW_*`
env flags hard-reject at startup; deliberately no `ALLOW_MARGIN` flag. Sizing in
real-broker mode clamps to `min(local_cash, broker.cash)` — never buying power —
and a broker account fetch failure FAILS the gate closed (§28). `GET /api/config`
renders all ten permission fields.

**Ops:** gateway republished on host **8010** (8000 is held by an unrelated local
container); UI dev server restarted against it via `ui/.env.local`. `.env` gained
the `BROKER_PROVIDER=alpaca_paper` block with empty key slots — broker status
honestly reports which setting is missing until the user pastes the paper keys.

**Verified:** full suite **827 passed, 1 skipped**; live gateway
`/api/market/overview` returns real Massive quotes (SPY/QQQ), `/api/config`
shows massive+openai configured, alpaca_paper unconfigured (honest).

**Next:** order-sync sweep for ACCEPTED/PARTIALLY_FILLED (guide Iter C/D),
periodic reconciliation, PENDING_SUBMIT lifecycle, then §30 E2E acceptance.

## 2026-08-10 — Alpaca paper broker (long stock + long options)

Real execution, paper only. `libs/broker/` mirrors the market-data provider
shape (Protocol + registry + `BrokerNotConfigured`); `AlpacaPaperBroker` talks
Trading API v2 over raw httpx. Gateway wiring in `apps/gateway/broker_exec.py`
+ `deps.py`: `BROKER_PROVIDER` defaults to `""`, approve/close 503, and the
exit sweep SKIPS — no silent fallback to the internal simulator, which stays
reachable only as the explicit `simulated` value.

**Live trading is unreachable by configuration.** Two layers: the constructor
parses the URL and demands scheme `https`, host exactly
`paper-api.alpaca.markets`, port 443 (any accepted spelling is normalised, so
userinfo decoys are discarded rather than carried into requests); then every
`submit_order` re-reads `GET /v2/account` and refuses unless the broker itself
reports `is_paper`. The adversarial verifier attacked this with 18 URL
variants and **found two real holes** — `http://` was accepted (API key in
cleartext) and `:8080` on the real host was accepted. Both fixed and pinned by
an 11-case matrix.

**Options are Level 3 long calls/puts** — the user's account permits them, so
the workflow's original "broker is stock-only, 422 on options" restriction was
removed as soon as that was known. Options ride the same `POST /v2/orders`
endpoint addressed by `occ_option_symbol()` (6-char padded root + YYMMDD +
C/P + strike in thousandths, rounded before scaling so float noise cannot
address a neighbouring strike). Wiring the symbol surfaced three arithmetic
bugs the tests then pinned: the position row was written `LONG_STOCK` with no
`opt_*` fields (so it could never have been closed — the close path rebuilds
the OCC symbol from exactly those columns), and both the buy debit and the
sell credit were missing the ×100 multiplier. An end-to-end round trip now
asserts cash reconciles exactly: 2 contracts 4.20 → 6.50 leaves the account
$460 richer, not $4.60.

Still long-only (§5): no Sell-to-Open exists anywhere, so covered calls,
CSPs and spreads remain out of scope — they need short legs, new max-loss
models and new exit families (Phase 9, not a flag).

Reconciliation (`GET /api/broker/reconcile`) compares broker positions/cash
against local rows, and a mismatch pauses trading per §18 rather than
auto-correcting — a test proves the pause has teeth (the next approve is
blocked by the kill switch).

Full suite: 769 passed, 1 skipped.


## 2026-08-10 — OpenAI LLM provider

The user chose OpenAI for recommendations. `.env.example` alone could not
express that: the registry only knew `stub` and `anthropic`, so configuring
`LLM_PROVIDER=openai` would have failed with "unknown LLM provider".

**Built:** `libs/llm/openai.py` — `OpenAIRecommendationProvider` on the
Responses API (`POST /v1/responses`) with strict `text.format` JSON-schema
structured output, registered as `"openai"`. It mirrors the Anthropic
provider exactly, because the two must be interchangeable by configuration
alone: ProviderError on a missing key at construction (never fires keyless)
or on transport/HTTP failure, and malformed model output logged-and-skipped
so a bad generation degrades to "no candidates" instead of crashing a
request path. Refusal parts and unparseable bodies yield an empty list.
14 new tests against a mocked httpx transport.

**Model id:** `gpt-5.6-sol` — verified against OpenAI's live model docs
rather than recalled (the assistant's training cutoff predates the GPT-5.6
family). `.env.example` lists the flagship/balanced/cheap tiers for both
providers and states that `LLM_MODEL` must match `LLM_PROVIDER` — there is
no cross-provider translation.

**Unchanged:** `llm_provider` still defaults to `""`. Recommendations stay
off until the user sets both `LLM_PROVIDER` and `LLM_API_KEY`; an
unconfigured install still answers 503 LLM_NOT_CONFIGURED.

Full suite: 628 passing.


Newest entries first. Each loop iteration appends one entry: what was built, key
decisions, test/audit status, and what's next.

---

## 2026-08-10 — Iteration 16: V1 DEFINITION-OF-DONE AUDIT — §45 COMPLETE (16/17 PROVEN, 1 PARTIAL)

An adversarial evidence audit walked every §45 bullet against named tests,
endpoints, and recorded live verifications (573/573 tests; two Docker E2E
passes; the 60-day replay test; live authority-boundary and promotion-ladder
audits). Result:

- **PROVEN (16):** independent builds; Docker Compose stack; watchlist CRUD;
  historical backtests; mechanical Bull/Bear/Neutral signals; Long Stock /
  Call / Put paper simulation; Trading Pool promotion (§4.3 checks);
  portfolio allocator sizing; dynamic cash reserve; risk reject/resize;
  pool-only paper orders; position monitoring (automated sweep);
  mechanical Sell-to-Close exits; LLM approval boundary; audit logs on all
  decisions; UI surface (all nine §28 sections live).
- **PARTIAL (1):** "Massive data ingested only for appropriate symbols" —
  the gating architecture is fully tested, but Massive itself is not
  integrated: all market data is deterministic stub. Blocked on
  MASSIVE_API_KEY + a real MassiveProvider.
- **§44 rules 1–20:** no violations. Noted soft spots: rule 9 — the backtest
  engine's exit rules are engine-internal rather than imported from
  libs/trading_core/exits (signals ARE shared; exits consolidation is a
  known follow-up); rule 17 untested until Phase 10 (no parameter
  optimization has occurred yet); LIQUIDITY gates stubbed.

**Blocked on user input (nothing more code-on-stubs can unlock):**
1. MASSIVE_API_KEY → real bars/quotes/chains/IV history (also unlocks IV
   Rank, real liquidity gates, ask/bid WORST fills).
2. LLM_API_KEY → switch llm_provider to "anthropic" for live
   recommendations.
3. git push → the CI in both repos has never run remotely (both 15 commits
   ahead of origin).
4. Free host port 8000 → canonical-port compose bring-up (E2E passes used
   an override port).
5. Broker credentials → Phase 6 real paper account / Phase 11 live rollout.

Docs pass: ADR-006 (alerts as audit view), ADR-007 (in-process monitor),
README refreshes, `make verify` (pytest + compose config + CI YAML) green.

**The development loop pauses here** — V1 per the plan's own definition is
complete on stub data. Phases 9 (spreads — needs account permission
decision), 10 (EV-based research upgrade), and 11 (live rollout) resume when
the blocking inputs above are provided.

**Built:**
- Trading Pool promotion checks (§4.3, closing a long-standing gap):
  MIN_HISTORY (bars ≥ RegimeParams.sma_slow — the real parameter, not a
  duplicated constant), BACKTEST_COMPLETED (latest id cited), OOS_STATS
  (≥ 1 out-of-sample trade — zero trades = no OOS evidence), LIQUIDITY
  (documented stub until Massive). Failures 422 with the structured checks;
  `acknowledge_risks: true` overrides but the TRADING_POOL_ADD audit details
  ALWAYS carry the full checks + acknowledged flag — overrides are
  permanently visible (§4.3 + §38).
- Automated position monitor: `run_exit_sweep` core shared by the POST
  endpoint and `apps/gateway/monitor.py`'s asyncio loop (interval
  `position_monitor_interval_seconds`, default 300, 0=disabled); survives
  transient DB errors; telemetry counters; lifespan start + graceful
  cancel/await shutdown; honest `GET /api/positions/monitor` status.

**Verified:** 573/573 green. Live promotion ladder (fresh → bars-only →
backtested → all-pass 201) exercised through real APIs with honest numeric
details at each rung; override path's audit row inspected directly; with a
1s interval the BACKGROUND task closed a forced hard-stop position with the
full EXIT_GENERATED + SYSTEM ORDER_* chain and the server shut down cleanly.

**Next (iteration 16):**
1. Docs pass: README architecture diagram refresh + DEVLOG index; ARCHITECTURE
   ADR for the alerts-as-audit-view and monitor decisions.
2. Dependency floor pinning + a `make verify` running pytest + docker config
   validation.
3. Review remaining §45 checklist for any unproven claims; consider stopping
   the loop if the plan's V1 surface is fully covered and hardened.

**Built:**
- `apps/gateway/alerts.py`: declarative ALERT_RULES over audit actions —
  alerts are a classified VIEW of the audit trail (no new table; the audit
  log is already the event source of truth). CRITICAL: TRADING_PAUSED /
  KILL_SWITCH_TRIGGERED / ORDER_REJECTED; WARNING: rejected-or-vetoed
  RISK_DECISION (predicate keys off the single detail shape orders.py writes
  for both veto and assess paths), EXIT_GENERATED, BACKTEST_FAILED; INFO:
  ORDER_FILLED / TRADING_RESUMED. Human titles with real numbers; order-
  scoped alerts enrich ticker/qty/price by batch-fetching the Order rows.
- `GET /api/alerts` (limit 1..200, newest-first, read-only).

**Verified:** 565/565 green. Live rule-coverage audit: veto preview alerts
WARNING with the gate named; approving previews (4 audit rows) produce
exactly the 2 genuine alerts; a REAL engine REJECT was reached through the
API (full-size position then re-preview → single-name caps clamp to zero)
and alerted with the engine's actual reason codes; pause/resume/fill/exit
titles all verified; WATCHLIST_ADD/DATA_BACKFILL never surface.

**Next (iteration 15):**
1. Promotion checks on Trading Pool add (§4.3): minimum history + backtest-
   exists + OOS-stats-present validation with an explicit user override
   acknowledging risk (audited) — closes a §4.3 gap.
2. Position monitor cron-style sweep: a lightweight periodic check-exits
   scheduler inside the gateway (asyncio task, interval configurable,
   disabled in tests) so exits stop depending on manual sweeps.
3. UI: promotion-check dialog + monitor status indicator.

**Built:**
- BacktestParams gains `fill_model` (OPTIMISTIC / CONSERVATIVE default /
  WORST) + `worst_slippage_bps` (25): effective slippage 0 / slippage_bps /
  max(slippage, worst) mapped onto daily-bar data (§20.2 "never treat
  historical mid as guaranteed fill"); documented that WORST becomes
  ask-to-buy/bid-to-sell once real quote data lands. CONSERVATIVE proven
  **bit-identical** to the pre-change engine via before/after reference-run
  JSON diff. Monotonicity pinned: OPTIMISTIC ≥ CONSERVATIVE ≥ WORST returns
  (strict when trades exist). API round-trips + summaries carry fill_model.
- UI: three-option segmented control with §20.2 descriptions,
  worst-bps input gated to WORST, fill-model chips on history + results,
  "historical mid is never a guaranteed fill" reminder.

**Docker E2E regression: PASS — and it caught the predicted defect:**
`008_option_execution.sql` was missing from the compose per-file migration
mounts (added in iteration 10 without the matching volume line). Fixed; db
init logs prove 001–008 all ran. Full smoke through the composed stack on an
override port (8000 still squatted by an unrelated container): analysis,
bars, options chain, preview reaching CONTRACT_SELECTION with a LONG_PUT +
contract proposal, backtest with fill_model=WORST echoed, portfolio greeks +
vol targeting, /api/config secret-absence, /metrics route-template labels,
X-Request-ID → audit correlation round-trip, psql schema checks (opt_*
columns, stock_bars_daily, recommendations evidence). Clean teardown.

**Verified:** 557/557 green (544 → +13).

**Next (iteration 14):**
1. Notification/alerts groundwork (§24 notification-service slice): audit-
   driven alert rules (kill switch triggered, EXIT_GENERATED, risk REJECT)
   surfaced as an in-app alerts feed on the Dashboard (no email/push yet).
2. Watchlist symbol page Overview tab: add regime/vol context strip.
3. Housekeeping: address any CI drift; consider pinning dependency versions.

**Built:**
- `libs/common/telemetry.py` — zero-dependency metrics (Counter / Histogram
  with cumulative buckets / Gauge with scrape callbacks) + Prometheus text
  exposition renderer + `request_id_var` ContextVar.
- Request-ID middleware: X-Request-ID honored/generated/echoed; one
  structured log line per request; `http_requests_total` +
  `http_request_duration_ms` labeled by ROUTE TEMPLATE (cardinality control);
  /metrics excluded from its own counters.
- **Correlation closure**: audit.record() now defaults `correlation_id` from
  the ambient request ID — every audit row traces to the exact HTTP request
  that caused it (§38 + §41). Explicit IDs still win.
- `GET /metrics`: uptime, watchlist bar freshness (scrape-time), honest
  option_chain_age_seconds=0 with stub explanation.
- `GET /api/config` — read-only serialization of the REAL engine dataclasses
  (permissions, risk limits, exit/selector/vol-target/signal/backtest params,
  paper fill model, kill switch). Secret-absence enforced by test: dummy env
  secrets planted, recursive walk finds no key/secret/token/password names
  and no secret values anywhere.

**Verified:** 544/544 green. Live audit: 37/37 checks including exact request
accounting in counters, route-template-only labels (no concrete tickers),
verbatim client correlation ID landing in the audit row, FAKELEAK secret hunt
across all endpoints (zero hits), and code-vs-API config drift diff (zero).

**Next (iteration 13):**
1. Docker E2E re-run to cover migrations 007+ and the new endpoints through
   the composed stack (regression of the Phase 0 acceptance).
2. Backtest engine option-aware upgrade research note OR §20.2 fill-model
   variants (optimistic/conservative/worst) as backtest params + UI toggle.
3. Watchlist symbol page News tab groundwork if news ingestion is prioritized.

**Built (pure libs → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/greeks.py` (§16): equivalent-shares aggregation
  (qty×mult×delta), delta-adjusted notional, net gamma/theta/vega with
  per-position contributions.
- `libs/trading_core/correlation.py` (§12.4): rolling Pearson over trailing
  log returns + union-find connected-component dynamic buckets (corr > 0.70,
  60d window — documented as requiring validation).
- `libs/trading_core/allocation.py` (§14): exposure multiplier clamp
  [0.25, 1.2], honest 1.0 on missing forecast.
- Risk engine (additive only, 471 prior tests byte-identical): greek limits
  (delta notional 150% NAV, |theta| 0.1% NAV/day, vega 1% NAV) with
  PORTFOLIO_*_LIMIT reject codes checked at the APPROVED quantity;
  `budget_multiplier` scales tier budget but `min(…, abs_max_trade_risk)`
  keeps §14 subordinate to hard caps.
- `/api/portfolio/risk` gains greeks (chain-resolved option greeks, honest
  data_ok flags), vol_targeting (crude v0 NAV-weighted RV20 forecast proxy,
  documented), and STATIC/DYNAMIC bucket kinds. RISK_APPROVAL gate feeds all
  three into assess(); detail names the multiplier when ≠ 1.
- **§42 replay test**: 60-day bar-by-bar replay through the real HTTP API
  (preview → approve → daily check-exits) asserting single-position
  invariant, gated approves, per-day cash-change == audited fills, complete
  ORDER_* chains, ≥1 entry + ≥1 mechanical ATR_TRAIL exit, and
  final_cash == initial + Σ realized_pnl to the cent. Runs in ~1s.

**Verified:** 519/519 green. Independent hand-recomputation of a mixed
3-position book matched aggregate_greeks exactly; correlation cross-checked;
multiplier fuzz never exceeded the absolute cap; replay test audited as
genuinely asserting (not vacuous).

**Next (iteration 12):**
1. Observability slice (§41): request-ID middleware + structured request
   logs; /metrics endpoint (simple counters/latency histograms, no external
   deps); market_data_lag + option_chain_age surfacing.
2. Settings page v1 (§28): read-only view of account permissions, risk
   limits, exit params from actual config objects (no editing yet).
3. Symbol News tab placeholder→real once news ingestion lands (defer).

**Built (pure libs → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/trading_core/volatility.py` — §7 vol regime v0 (LOW/NORMAL/HIGH/
  EXTREME from ATM IV level + IV/RV ratio; provisional until IV history
  enables IV Rank; every threshold a parameter).
- `libs/trading_core/strategies/instrument.py` — the §8 Instrument Selection
  matrix, every cell implemented + documented with §5 degradations (spreads
  unpermitted → stock/single-leg/no-trade): BULL STRONG+LOW → LONG_CALL,
  spread cells degrade, BEAR WEAK → NO_TRADE, EXTREME never buys premium,
  NEUTRAL → NO_TRADE. AccountPermissions configurable. `strength_tier`
  refactored public in risk engine (single source of truth for edge→tier).
- `exits/engine.py` — option exits (§11.3/§11.7): PREMIUM_HARD_STOP (-45%
  research parameter) > DTE_EXIT (≤21) > underlying-driven rules via shared
  internals (bit-identical to stock evaluation, §21). Underlying HARD_STOP
  replaced by the premium stop for options; missing mid reported loudly.
- Gateway: VOLATILITY gate is now a real classification; INSTRUMENT gate is
  the matrix verdict with rationale; CONTRACT_SELECTION proposes the §9
  top-ranked contract; risk sizing for options passes entry=stop=mid×100 so
  approved_quantity counts CONTRACTS with every cap intact (§12.1). Approve/
  close handle ×100 multiplier + per-contract commission ($0.65); close
  regenerates the chain to find the same contract, intrinsic-value fallback
  if expired. Positions evaluate option rows via evaluate_option_exit.
  `migrations/008_option_execution.sql`.

**Verified:** 471/471 green (319 → +152). Exhaustive §8 sweep (1,200
permission-expanded cells): always §5-legal, EXTREME never buys premium,
rationale always present. 5,000-trial option sizing fuzz: contracts×premium
×100 never exceeded the absolute cap nor tier budget; exact boundary check
(100k NAV, STRONG, $250/contract) = exactly 4 contracts. Premium-stop/DTE
boundary arithmetic bit-exact. Live option lifecycle cash-conserved to the
cent.

**Next (iteration 11):**
1. Correlation buckets from returns (§12.4 rolling correlation grouping to
   replace/augment the static TECH_MEGA list) + delta-adjusted exposure and
  portfolio Greeks aggregation (§16) on the Risk page.
2. Volatility targeting layer (§14) as an allocation modifier (capped 1.2x).
3. Replay-style integration test: multi-day loop advancing stub data
   (backfill → signal → preview → approve → monitor → exit) as one test.

**Built (bs+selector libs → chain+API chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/options/bs.py` — pure-stdlib Black-Scholes-Merton
  (math.erf normal CDF): price + Greeks with documented conventions (theta per
  calendar day, vega per IV point, signed delta); intrinsic-value expiry edge.
- `libs/trading_core/contracts/selector.py` — Contract Selector v0 (§9):
  side gate (BULL→calls, BEAR→puts, long-only §5), §9.1 filters (DTE 30–90,
  |Δ| 0.40–0.75, OI/volume/spread/theta-burden — every threshold a parameter,
  every failure a numeric reason), §9.2 v0 heuristic ranking
  (liquidity − theta burden + delta fit, components exposed; Phase 10 upgrades
  to EV-based). All contracts returned with verdicts for the §34 view toggles.
- Stub option chain in libs/market_data: deterministic (crc32-seeded) — weekly
  + monthly expiries, tiered strike grid ±25%, seeded IV smile + term
  structure, BS theoretical mids, moneyness/DTE-dependent spreads, ATM-decaying
  volume/OI. Same-IV-both-rights documented as v0 (no skew yet).
- `GET /api/watchlist/{ticker}/options?direction=AUTO|BULL|BEAR` (§34):
  AUTO resolves via score_direction; NEUTRAL → no candidates ("NO TRADE is a
  valid output"); summary with ATM IV, straddle expected move, RV20, IV−RV
  spread, and iv_rank **honestly null** until real IV history exists.
- Compose: frontend service added (context ../ui, :3000), image built OK.

**Verified:** 319/319 green (259 → +60). Independent quant audit: put-call
parity worst error 2.27e-13 over a 500-point grid; bs_price vs an independent
Simpson risk-neutral integrator agrees to 1.56e-8; finite-difference delta
check passed; no crossed markets; every live API candidate re-satisfies the
§9.1 filters from its own row values.

**Next (iteration 10):**
1. Instrument Selection matrix (§8): direction×strength×IV-regime →
   LONG_STOCK/LONG_CALL/LONG_PUT/NO_TRADE in trading_core; wire into the §10
   INSTRUMENT gate + Trade Plan (show chosen instrument + §9 contract when
   options are selected).
2. Volatility regime classification (§7 LOW/NORMAL/HIGH/EXTREME from stub
   chain IV vs RV) feeding the matrix.
3. Order approve path for LONG_CALL/LONG_PUT paper fills (chain mid ± slippage)
   with per-contract max-loss = premium (§12.1 options sizing in risk engine).

**PHASE 0 ACCEPTANCE: PASS.** `docker compose up --build` boots the real stack
(TimescaleDB pg16 + Redis + gateway) and the smoke test ran end to end through
it: healthz/readyz, watchlist add, full analysis (migrations 001–007 +
Timescale storage + lazy backfill + signals), market overview, portfolio risk
(NAV 100k), audit trail. Stack torn down with `down -v`, nothing left running.

**Two real defects found and fixed by the E2E agent:**
1. docker-compose.yml mounted the whole migrations directory over
   `/docker-entrypoint-initdb.d`, shadowing the timescale image's own init
   scripts (`CREATE EXTENSION timescaledb`) — migration 002's
   `create_hypertable()` would have aborted initdb on a fresh volume. Fixed by
   mounting each migration file individually (rule documented in README).
2. `stock_bars_daily` existed only via ORM `create_all` — added
   `migrations/007_stock_bars_daily.sql` mirroring the ORM exactly, with the
   mirror-in-same-commit rule documented.

**Environmental note:** host port 8000 was occupied by an unrelated container
(roboxai-optimizer) — smoke test used a scratchpad-only override on :8010;
the repo compose keeps the canonical 8000:8000. Free the port to run locally.

**Also built:**
- GitHub Actions CI for both repos (services: py3.12 + pytest; ui: node22 +
  typecheck + build), YAML-validated; `make ci` target.
- `GET /api/watchlist/{ticker}/bars` (watchlist-gated OHLCV series, limit
  10–600) + OHLC-sanity tests.
- Audit filters: `GET /api/audit?action=&actor_type=` (AND semantics, typed
  422 on bad values) + `GET /api/audit/actions` distinct-values endpoint.

**Verified:** 259/259 green before and after Docker work.

**Next (iteration 9):**
1. Option-chain scaffolding (§34): stub chain provider (deterministic strikes/
   expiries/greeks around spot), `GET /api/watchlist/{ticker}/options`,
   Options tab with eligibility highlighting groundwork.
2. Contract Selector v0 (§9): candidate filters + risk-adjusted ranking over
   the stub chain (research-only until real chain data).
3. Docker compose entry for the UI container (frontend joins the stack).

**Built (llm+health libs → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/llm/` — provider abstraction mirroring libs/market_data:
  `RecommendationDraft` validates the §4.1 score schema; deterministic stub
  provider (day-seeded, exclusions honored, evidence timestamps strictly
  before as_of — §20.3 news-timestamp integrity); real Anthropic provider
  (written against the claude-api skill: Messages API structured outputs,
  malformed model output logged-and-skipped, never fires without a key).
  Default provider switched to "stub" — keyless-safe.
- Recommendations API: refresh (LLM-attributed audits, skips watchlisted/
  already-PENDING tickers, **performs zero watchlist/pool/order writes**),
  list by status, dismiss (USER), promote — THE only rec→watchlist path,
  implemented by refactoring watchlist insertion into a shared
  `add_ticker_to_watchlist` helper used by both POST /api/watchlist and
  promote so the paths cannot diverge; WATCHLIST_ADD audited USER with the
  rec id in the note. `migrations/006`. RECOMMENDATION_PROMOTED enum added.
- `libs/trading_core/health.py` — Strategy Health Monitor v0 (§19):
  win rate / profit factor / expectancy / drawdowns over closed-trade PnLs;
  status ladder INSUFFICIENT_DATA (judgement withheld below min sample) /
  HEALTHY / WARNING / PAUSE_RECOMMENDED with numeric explanations;
  `GET /api/health/strategy` read-only report (no pause automation yet).

**Verified:** 249/249 green. Authority-boundary audit (the point of Phase 8):
static — recommendations router's only insert is the Recommendation row,
libs/llm has zero DB references, promote/watchlist share one helper with
hardcoded USER attribution; live — refresh left watchlist/pool/positions
untouched, promote added exactly the approved ticker, double-promote 409'd,
promoted tickers excluded from later drafts. Health math independently
recomputed. UI verifier grep-confirmed no trade action exists on the page.

**Next (iteration 8 — hardening + Phase 0 completion):**
1. Docker Compose end-to-end check (build gateway image, full stack up,
   healthz through the compose network) — Phase 0 acceptance still unproven.
2. CI: GitHub Actions for both repos (pytest / typecheck+build).
3. Watchlist symbol page Price tab (candlestick/volume from stored bars) and
   Activity page action-type filter chips.
4. Begin §34 option-chain scaffolding if time allows (stub chain provider).

**Built (exits lib → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/exits/engine.py` — pure Exit Engine v0 (§11): evaluates
  ALL five rules every call (HARD_STOP → SIGNAL_FLIP → SIGNAL_DECAY →
  ATR_TRAIL → TIME_STOP, backtest-priority order) with numeric reasons; holds
  report "OK:"-prefixed reasons so the user always sees why a position is kept
  (§37). Signal rules degrade to "insufficient data" on short history but a
  data gap can NEVER disable the hard stop. Reuses score_direction (§21).
- Paper execution: `POST /api/orders/approve` re-runs the FULL §10 gate chain
  server-side (client previews never trusted); BUY_TO_OPEN /
  SELL_TO_CLOSE are the only sides (§5, DB CHECK constraint); idempotent
  client_order_id (§42); 409 no-pyramiding; fills at last close ± slippage +
  commission (same model as backtest for comparability); ORDER_REQUESTED →
  ORDER_SUBMITTED → ORDER_FILLED + RISK_DECISION audited in one transaction.
- `POST /api/orders/close`: partial/full, realized-PnL arithmetic, allowed
  while trading is paused (closing reduces risk — §18 risk-priority).
- `GET /api/positions` (§37 contract: stop/trail/edge-decay/time-stop
  countdown/exit status + full reasons) and `POST /api/positions/check-exits`:
  mechanical exits audit EXIT_GENERATED and execute SYSTEM sell-to-close,
  unblocked by the kill switch. `migrations/005_orders.sql`.

**Race conditions found & fixed by the adversarial verifier (live-reproduced):**
concurrent same-client_order_id approves returned [200, 500] via UNIQUE
IntegrityError, and concurrent different-key approves double-filled into two
positions with a double cash decrement. Fixed with a shared per-event-loop
execution lock serializing approve / close / check-exits; two regression
tests added. Cash conservation verified to the cent (partial + full closes).

**Verified:** 202/202 green. Live lifecycle: preview → approve → position
(with hold reasons) → check-exits → forced HARD_STOP exit → cash credited;
watchlist-only approve rejected with zero Order rows; SELL_TO_OPEN absent
from the codebase.

**Next (iteration 7 — Phase 8 + hardening):**
1. LLM Recommendation Pool (§4.1, §30): provider-abstracted llm service (stub
   provider first), recommendations API (PENDING/DISMISSED/PROMOTED lifecycle,
   LLM actor audited, zero execution authority), news-free v0 using
   watchlist-adjacent discovery heuristics as stub input.
2. UI Recommendations page (§30 cards: no Trade Now action; View Evidence /
   Dismiss / Add to Watchlist which routes through the normal USER watchlist
   API).
3. Strategy Health Monitor v0 (§19): rolling stats over closed paper trades.

**Built (risk lib → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/risk/engine.py` — pure, strategy-independent (§17):
  `assess(request, snapshot, limits)` pipeline in spec order: kill switch (§18)
  → heat reject gate (§12.5) → |edge|→strength tier→risk budget hard-capped by
  abs_max_trade_risk (§12.2 "no confidence may override") → base sizing
  floor(nav·budget/stop) (§12.1) → quantity clamps for single-name risk/capital
  (§12.3), correlation bucket (§12.4), strict heat headroom, regime cash floor
  (§13) → APPROVE / APPROVE_WITH_RESIZE / REJECT with machine reason codes +
  §36-style numeric explanations. `portfolio_heat`/`heat_state` helpers
  (NORMAL/ELEVATED/HIGH/BLOCKED at 4/6/8%). All limits in frozen `RiskLimits`.
- 33-test suite incl. hand-computed binders for every cap, regime-dependent
  cash-floor flip (same request: APPROVE in STRONG_BULL → REJECT in
  STRONG_BEAR), and a 200-case seeded property test of the §42 invariants.
- Portfolio singleton (paper cash 100k configurable) + Position ORM +
  `migrations/004_portfolio.sql`; `GET /api/portfolio/risk` (§36 contract:
  NAV/cash/floor/heat/max-new-risk/buckets/limits, honest nulls for bar-less
  positions; read-only, no audit).
- `POST /api/orders/preview` — the §10 gate chain in exact order (pool
  authorization incl. per-symbol + global kill switch → data quality → regime
  (TRANSITION/bear veto for long stock) → directional signal → volatility/
  liquidity SKIPPED with explicit V1 details → instrument → contract-selection
  SKIPPED → risk approval via assess() with stop = 2.0·ATR14). First FAIL
  skips the rest; exactly ONE SYSTEM RISK_DECISION audit event per preview,
  veto or not (§38). why_trade / why_not_trade always both present (§33).

**Verified:** 164/164 green. Independent fuzz (600 seeded cases): approved
risk never exceeded the 1.5% NAV absolute cap nor the tier budget; heat_after
strictly < 8% on every approval; regime cash floors respected; kill switch
always wins; every REJECT carries reason codes. Live boot walked the
watchlist-only → veto, authorized → full chain, paused → gate-1 veto paths.

**Next (iteration 6 — Phase 6 paper execution start):**
1. Order state machine: `POST /api/orders/approve` (from a preview) → paper
   fill at last close, position open/close, cash movement, duplicate-order
   guard; ORDER_* audit chain.
2. Position monitor + Exit Engine v0 wiring for open paper positions
   (signal-decay / ATR-trail / time-stop checks over stored bars, §11).
3. UI Positions page v1 (§37) + order approve flow from Trade Plan.

**Built (engine → gateway chain + parallel UI, 2 adversarial verifiers):**
- `libs/trading_core/backtest/engine.py` — pure replay engine, LONG STOCK only
  (option backtesting deferred until real chain data exists — no fabricated
  option prices). §20.3 semantics enforced: signals computed on `[:t+1]` slices
  via the SAME `classify_regime`/`score_direction` used live (§21); decision at
  close of t fills at open of t+1 with slippage bps + per-share commission (§44
  rule 11); exits in priority order SIGNAL_FLIP → SIGNAL_DECAY (§11.1) →
  ATR_TRAIL (§11.5) → TIME_STOP (§11.6) → END_OF_DATA; IS/OOS split with
  per-segment metrics (report-only, §44 rule 16); every division guarded —
  None, never NaN. All knobs in frozen validated `BacktestParams`.
- 26-test quant-integrity suite: the no-look-ahead property (closed trades
  bit-identical between 300-bar prefix and 400-bar runs), hand-computed fill
  arithmetic, costs monotonicity, NO-TRADE honesty, long-only invariants.
- `POST /api/backtests` (watchlist-gated, synchronous V1, params validated
  before any state change), `GET /api/backtests[/{id}]`; records persisted with
  USER BACKTEST_STARTED + SYSTEM BACKTEST_COMPLETED/FAILED audit events in one
  transaction. `migrations/003_backtests.sql`.
- `GET /api/watchlist/overview` — per-symbol price/regime/scores/bias +
  opportunity_status v0 mapping (§31) + latest backtest status.

**Bug found & fixed during review:** the gateway implementation agent spotted
that the engine's fill block never copied `pending_entry` into `entry_reason`
(every trade explained its exit but not its entry — §38 violation). Fixed with
`entry_reason = pending_entry` in the fill branch + a regression test asserting
every trade's entry_reason carries the edge number. Suite now 122 green.

**Verified:** independent verifier re-derived fills to the cent, recomputed
total-return/max-drawdown from the equity array (1e-6 agreement), re-ran the
no-look-ahead experiment with a shock series, and exercised the API live
(404/422 paths, audit pairs, overview status flip).

**Next (iteration 5 — Phase 4 start):**
1. Portfolio state: NAV, cash, positions tables; paper-fill plumbing groundwork.
2. Risk Engine v0 as an independent module (§17): position sizing from risk
   budget (§12.1-12.2), single-name cap, Portfolio Heat, cash floor by regime
   (§13), APPROVE/RESIZE/REJECT decisions with reason codes, audited.
3. `POST /api/orders/preview` returning the full gate-chain evaluation (§10).
4. UI Risk page v1: NAV/cash/heat/limits + latest risk decisions.

**Built (signals lib → gateway chain + parallel UI, 2 adversarial verifiers, zero fixes):**
- `libs/trading_core/signals/regime.py` — Market Regime Engine v0 (§6.1):
  `classify_regime` with frozen `RegimeParams` (all thresholds backtest parameters).
  Rules ordered: insufficient history → TRANSITION (no-trade posture); ATR/close
  dislocation → TRANSITION; stacked SMAs + fast-slope → STRONG_BULL/BEAR;
  above/below both major SMAs → MILD_*; else NEUTRAL_RANGE. Full explainability
  features dict on every result.
- `libs/trading_core/signals/directional.py` — Directional Signal Engine v0 (§6.2):
  `score_direction` evaluating 8 mirrored bull/bear feature pairs (SMAs, MACD
  cross/zero, RSI continuation zones, pivot HH+HL/LH+LL structure) + optional
  volume expansion. Weighted parameterized scores 0–100, edge = bull − bear,
  bias by threshold; every component listed with numeric human-readable detail.
- Daily bars: `Bar` + `get_daily_bars` in the provider Protocol; StubProvider
  emits deterministic crc32-seeded weekend-skipping walks. `StockBarDaily` table.
- `GET /api/watchlist/{ticker}/analysis` — 404 off-watchlist (§4.2), lazy 600-bar
  backfill with SYSTEM `DATA_BACKFILL` audit (once only), indicators + regime +
  signal + 250-bar chart series in one contract.
- `/api/market/overview` regime now computed from SPY bars. ADR-005: SPY/QQQ/VIX
  are system reference symbols exempt from the watchlist-only data rule.

**Verified:** 85/85 tests green (was 51). Independent verifier booted the app:
contract keys, enum validity, single-backfill audit invariant, 404 path, and
monotonic-series sanity (uptrend → STRONG_BULL/BULL, downtrend → STRONG_BEAR/BEAR)
all confirmed live.

**Next (iteration 4 — Phase 3 start):**
1. Backtest engine v1: bar-by-bar replay over stored daily bars using the SAME
   signals lib (§21); Long Stock entries/exits from directional bias + regime
   gates; explicit fill model (next-bar open) + transaction costs; equity curve,
   drawdown, win rate, profit factor outputs.
2. `POST /api/backtests` + `GET /api/backtests/{id}` with stored results + audit.
3. UI Backtests page v1: config form + results (§35 metrics, IS/OOS split label).
4. Watchlist rows enriched with regime/scores/status from analysis cache.

**Built (via 3 parallel implementation agents + 2 adversarial verify agents):**
- `libs/trading_core/features/indicators.py` — pure, dependency-free, deterministic:
  SMA, EMA (SMA-seeded), RSI (Wilder), True Range/ATR (Wilder), MACD (12/26/9,
  parameterized), realized vol (close-to-close log returns, annualization param),
  pivot highs/lows (§6.3; final `window` bars always unconfirmed — no look-ahead,
  §20.3). All outputs input-length with None warmup padding; all periods parameters.
  34 new tests including hand-computed reference values with arithmetic in comments
  and a backtest/live parity check for pivot stability.
- `libs/market_data/` — provider abstraction (Quote + MarketDataProvider Protocol,
  name-based registry) with deterministic StubProvider (SPY/QQQ/VIX, minute-keyed
  wiggle) until the Massive integration lands. `GET /api/market/overview` returns
  provider/as_of/stale/market_regime (NEUTRAL_RANGE placeholder)/indices.
- Global kill switch (§18): persistent `system_state` singleton row (default:
  trading disabled), `GET /api/trading/status`, `POST /api/trading/pause` (reason
  required), `POST /api/trading/resume`; both mutations USER-attributed and audited
  (TRADING_PAUSED/TRADING_RESUMED) in the same transaction.
- `migrations/002_system_state_and_bars.sql` — system_state seed + `stock_bars_1m`
  Timescale hypertable (Phase 1 groundwork).

**Verified:** full suite 51/51 green (was 11); independent verify agent booted the
app and confirmed overview/status/pause flows + audit records via curl; indicator
spot checks passed. Zero fixes needed.

**Next (iteration 3):**
1. Historical OHLCV ingestion into stock_bars (stub-generated series for dev) and
   `GET /api/watchlist/{ticker}/analysis` computing indicators over stored bars.
2. Market Regime Engine v0 (§6.1) using SPY/QQQ features — replace the
   NEUTRAL_RANGE placeholder in /api/market/overview.
3. Directional signal engine v0 (§6.2): parameterized bull/bear scores over features.
4. UI: symbol analysis page skeleton (tabs per §33) showing computed indicators.

---

## 2026-08-10 — Iteration 1: Phase 0 skeleton + Watchlist/Trading Pool core

**Built:**
- Project skeleton: `pyproject.toml`, `libs/common` (config via pydantic-settings,
  structured JSON logging with secret redaction), `libs/trading_core/models` (domain
  enums: regimes, instruments, risk decisions, audit actions, actor types).
- `apps/gateway` FastAPI modular monolith:
  - `/healthz`, `/readyz` (DB-checked)
  - Watchlist API: `GET/POST /api/watchlist`, `DELETE /api/watchlist/{ticker}`
  - Trading Pool API: `GET/POST /api/trading-pool`, toggle `POST /{ticker}/trading`,
    `DELETE /{ticker}`
  - Audit API: `GET /api/audit` (filter by entity, read-only)
- `migrations/001_initial.sql` — Postgres/Timescale DDL with FK cascade
  (trading_pool → watchlist) and audit indexes.
- `docker-compose.yml` — timescaledb + redis + gateway; migrations auto-applied on
  first db boot. Gateway Dockerfile.
- Test suite: 11 tests, all passing. The important ones encode plan rules:
  - non-Watchlist symbol cannot enter Trading Pool (rule 6);
  - promotion starts with trading disabled (authorization ≠ order);
  - short strategies rejected by account constraints (rules 7/8);
  - every mutation audited with USER attribution (rule 12);
  - Watchlist removal cascades out of Trading Pool, cascade itself audited.

**Verified:** `pytest` 11/11 green; live smoke test via uvicorn+curl confirmed the
422 rejection path, disabled-by-default promotion, and audit trail contents.

**Decisions:** see ARCHITECTURE.md ADR-001…004 (modular monolith, repo naming,
transactional audit, write-path authorization).

**Deliberately deferred:** auth-service (single fixed `local-user` identity for now),
event bus, Massive market data adapter, kill-switch API, recommendations endpoints.

**Next (iteration 2):**
1. Massive market-data adapter interface + stub provider (so UI can show prices
   without a real key), `GET /api/market/overview`.
2. Feature engine start: SMA/RSI/ATR/MACD in `libs/trading_core/features` with
   deterministic unit tests (Phase 2 groundwork).
3. Kill switch API: `POST /api/trading/pause` + `/resume` with audit + UI banner wiring.
4. Historical OHLCV storage schema (Timescale hypertable migration).
