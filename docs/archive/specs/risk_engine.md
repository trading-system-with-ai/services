Institutional Risk Engine Upgrade — Risk Modeling, Portfolio Risk & Allocation

ROLE

For this loop, act simultaneously as:

* Managing Director / Head of Market Risk at a top-tier global investment bank
* Senior Quantitative Risk Researcher
* Senior Equity & Options Quant
* Portfolio Risk Manager
* Derivatives Risk Specialist
* Senior Software Architect
* Senior Backend Engineer
* Senior Data Engineer
* Senior UI/UX Designer for institutional trading systems
* Adversarial QA / Model Validation Lead

Treat this upgrade as if it were going through an institutional Model Risk Committee.

Your responsibility is NOT simply to add more indicators.

Your responsibility is to determine:

What can go wrong with the current portfolio, how large the losses can become, how the risks interact across positions, whether the statistical model is reliable, and whether a new trade should be allowed.

⸻

1. FIRST RULE — INSPECT BEFORE IMPLEMENTING

Do NOT blindly implement everything described below.

Before changing code:

1. Inspect the complete current backend repository.
2. Inspect the frontend repository.
3. Inspect the current Risk Engine.
4. Inspect Portfolio Allocation.
5. Inspect Portfolio Heat.
6. Inspect cash-floor logic.
7. Inspect correlation buckets.
8. Inspect Portfolio Greeks.
9. Inspect stock and option position models.
10. Inspect backtest infrastructure.
11. Inspect Massive data availability.
12. Inspect Alpaca paper/live broker abstraction.
13. Inspect current supported instruments.
14. Inspect existing DB schemas.
15. Inspect current Risk UI.
16. Read DEVLOG and architecture ADRs.

Then produce an internal gap analysis:

CURRENT CAPABILITY
MISSING CAPABILITY
MODEL VALUE
DATA REQUIREMENTS
IMPLEMENTATION COST
COMPUTATIONAL COST
MODEL RISK
PRIORITY

You have discretion to decide that a proposed model should:

IMPLEMENT NOW
IMPLEMENT AS RESEARCH MODE
DEFER
REJECT

Explain why.

Do not add complexity merely because a model is academically interesting.

⸻

2. CORE DESIGN PRINCIPLE

The existing platform already has rule-based risk controls.

Preserve them.

Examples may include:

Single Trade Max Risk
Single Name Cap
Portfolio Heat
Cash Floor
Correlation Bucket
Portfolio Greeks
Kill Switch
Account Permission Gate
Trading Pool Authorization
Liquidity Gates

These are:

HARD RISK LIMITS

The new statistical models are an additional layer.

They do NOT replace the hard limits.

The new architecture should conceptually become:

                    TRADE REQUEST
                         │
                         ▼
                 ACCOUNT PERMISSIONS
                         │
                         ▼
                   HARD LIMITS
                         │
                         ▼
              STATISTICAL RISK ENGINE
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
   VaR / ES       Conditional Risk     Stress Risk
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                DEPENDENCE / PORTFOLIO
                         │
                         ▼
                 RISK CONTRIBUTION
                         │
                         ▼
                 ALLOCATION ENGINE
                         │
                         ▼
                  APPROVE / RESIZE
                      / REJECT

⸻

3. RISK MODELS MUST USE RETURNS

Risk modeling should operate primarily on:

returns
portfolio P&L
scenario P&L

rather than absolute security prices.

Create a standardized return calculation layer.

Support at minimum:

simple returns
log returns

Use one convention consistently for each model and document it.

Store model metadata including:

return_type
frequency
lookback_window
timestamp
data_source
model_version

⸻

4. BUILD A RISK MODEL REGISTRY

Do not put every calculation inside risk_engine.py.

Create a modular Risk Model abstraction.

Conceptually:

RiskModel
    calculate(...)
    validate(...)
    diagnostics(...)
    metadata(...)

Potential implementations:

HistoricalVaRModel
GaussianVaRModel
HistoricalESModel
GaussianESModel
GARCHRiskModel
EVTTailRiskModel
HistoricalStressModel
HypotheticalStressModel
CovarianceRiskModel
RobustCovarianceModel
CopulaRiskModel
RiskContributionModel

Not every model needs to become production-critical immediately.

⸻

5. MODEL TIER SYSTEM

Classify models.

TIER 0 — HARD LIMITS

Always active.

Examples:

max trade loss
cash floor
single-name cap
portfolio heat
account permissions
liquidity
kill switch

No statistical assumption.

⸻

TIER 1 — CORE STATISTICAL RISK

Production priority.

Implement first if not already available:

Volatility
Historical VaR
Historical ES
Portfolio VaR
Portfolio ES
Risk Contribution
Drawdown
Stress Loss

⸻

TIER 2 — CONDITIONAL RISK

Next priority:

GARCH volatility
Conditional VaR
Conditional ES
Volatility Forecast

⸻

TIER 3 — ADVANCED TAIL / DEPENDENCE

Research-first unless current platform/data clearly supports production use:

EVT / POT / GPD
Copulas
GARCH-Copula
Tail Dependence
Minimum Tail Dependence

⸻

6. VaR — IMPLEMENT BUT DO NOT TRUST ALONE

Implement Value-at-Risk for multiple horizons and confidence levels.

Suggested default research grid:

1 day
5 days
10 days
95%
99%

At minimum calculate:

Historical VaR
Gaussian VaR

The UI should NOT hide the model type.

Show:

Historical VaR 99% 1D
Gaussian VaR 99% 1D

Do not simply display:

VaR = $2,300

without methodology.

⸻

7. EXPECTED SHORTFALL MUST BE FIRST-CLASS

VaR does not tell us the expected severity after the VaR threshold has been breached.

Therefore Expected Shortfall should be at least as prominent as VaR.

Implement:

Historical ES
Gaussian ES

At minimum:

95% ES
99% ES

Conceptually:

VaR:
Where does the bad tail begin?
ES:
Once we are in that tail, how bad is the average loss?

For risk approval, favor ES over VaR when the two give conflicting signals.

⸻

8. PORTFOLIO VaR / ES

Do not calculate risk only position-by-position.

Calculate portfolio-level loss distributions.

Support:

Current Portfolio VaR
Current Portfolio ES
Proposed Portfolio VaR
Proposed Portfolio ES
Incremental VaR
Incremental ES

Before a trade:

Current ES = X
Proposed ES = Y
ΔES = Y - X

A trade with a small standalone max loss may still create a large:

Incremental Portfolio ES

because of correlation or common factor exposure.

⸻

9. MARGINAL RISK

Add:

Marginal VaR
Marginal ES

or an equivalent incremental-risk framework.

The important question is:

How much does adding one more unit of this position increase total portfolio downside risk?

Use this in allocation.

⸻

10. RISK CONTRIBUTION

Create position-level risk contribution.

For position i:

Risk Contribution_i

should answer:

How much of total portfolio risk is attributable to this position?

Support at least:

Volatility Risk Contribution
ES / CVaR Risk Contribution

The UI should be able to show:

NVDA Call        21% of portfolio downside risk
QQQ              18%
AAPL             10%
Cash              ...

Capital weight and risk weight are not the same thing.

This distinction is essential.

⸻

11. RISK-CONTRIBUTION CONCENTRATION GATE

Create a configurable gate such as:

max_single_position_risk_contribution
max_sector_risk_contribution
max_factor_risk_contribution

Do NOT choose arbitrary production thresholds silently.

Define research defaults and validate them.

Example decision:

Requested position:
$800 maximum premium loss
But:
Portfolio ES contribution would rise
from 12% → 31%
Decision:
RESIZE

This is more sophisticated than dollar exposure alone.

⸻

12. GARCH CONDITIONAL VOLATILITY

The current platform should not assume constant volatility.

Assess whether current dependencies support a production GARCH implementation.

Preferred baseline:

GARCH(1,1)

Consider Student-t innovations if supported and justified by diagnostics.

Outputs:

1D conditional volatility
5D conditional volatility
10D conditional volatility
conditional VaR
conditional ES

The important purpose is:

Respond to volatility clustering.

⸻

13. GARCH MODEL VALIDATION

Do not treat a fitted GARCH model as automatically trustworthy.

Persist diagnostics.

Examples:

fit status
parameters
residual diagnostics
standardized residuals
forecast timestamp
training window
distribution assumption

If the model fails diagnostics:

MODEL_DEGRADED

Fallback to a simpler risk model.

Never fabricate GARCH results.

⸻

14. VOLATILITY REGIME SHOULD AFFECT RISK BUDGET

Use conditional volatility forecasts to modify position sizing.

Conceptually:

Base Risk Budget
        ×
Volatility Adjustment
        =
Adjusted Risk Budget

Example:

forecast volatility increases sharply
→ position size decreases

Do not allow:

strong signal confidence

to override a severe conditional-volatility risk increase.

⸻

15. HEAVY-TAILED DISTRIBUTIONS

The platform should not universally assume Normal returns.

Add distribution diagnostics.

At minimum:

skewness
kurtosis
normality diagnostic

Use these diagnostics to label:

NORMAL_LIKE
HEAVY_TAIL
LEFT_SKEWED
UNSTABLE

If returns are heavily non-normal:

reduce trust in Gaussian VaR.

Historical ES / conditional / tail models should receive greater weight.

⸻

16. EVT — EXTREME VALUE THEORY

Assess implementation of:

Peaks Over Threshold
Generalized Pareto Distribution

for the loss tail.

Purpose:

Estimate rare losses that ordinary standard-deviation or Gaussian models may underestimate.

However EVT must NOT become a naive always-on number.

Threshold selection is a major model-risk problem.

Require diagnostics such as:

threshold
number of exceedances
mean residual life stability
parameter stability
sample size
fit quality

If insufficient tail observations exist:

EVT_NOT_RELIABLE

Do not produce false precision.

⸻

17. EVT SHOULD BE RESEARCH-FIRST

Unless sufficient historical data and model validation exist:

implement EVT initially as:

Research Risk Metric
Stress Calibration Input
Tail Warning

not as the only production blocking metric.

The hard Risk Engine should remain operational if EVT is unavailable.

⸻

18. DEPENDENCE — DO NOT RELY ONLY ON LINEAR CORRELATION

The existing correlation buckets are useful.

Do not remove them.

But extend dependency analysis.

Start with:

rolling Pearson correlation
rolling Spearman correlation

and inspect whether current infrastructure can support more advanced dependence modeling.

⸻

19. CORRELATION REGIME SHIFT

Portfolio diversification tends to deteriorate in stressed markets.

Create monitoring for:

normal correlation
current rolling correlation
stress correlation

Example:

NVDA / AMD
normal correlation: 0.61
current:            0.84

This should increase:

Technology concentration risk

even if dollar positions did not change.

⸻

20. COPULA / GARCH-COPULA — ADVANCED MODEL

Evaluate whether GARCH-Copula belongs in the current platform.

Do not implement it merely to satisfy this prompt.

If feasible, architecture should follow:

Individual Return Series
        ↓
Marginal GARCH Models
        ↓
Standardized Residuals
        ↓
Transform to Uniform Variables
        ↓
Copula Dependence Model
        ↓
Monte Carlo Scenarios
        ↓
Portfolio Scenario P&L
        ↓
VaR / ES / Tail Risk

Candidate copula models may include:

Gaussian
Student-t

Student-t deserves consideration because of tail dependence.

But benchmark multiple models.

⸻

21. OPTIONS REQUIRE NONLINEAR RISK TREATMENT

This platform supports:

stocks
long calls
long puts
possibly defined-risk spreads later

Do NOT treat an option as if it were simply stock multiplied by Delta for all risk calculations.

Delta approximation may be acceptable for:

small moves
quick exposure estimates
UI summaries

but portfolio tail risk should preferably use:

FULL REVALUATION

under scenarios.

⸻

22. OPTION SCENARIO REVALUATION

For every simulation / stress scenario:

generate underlying and volatility scenario states.

Then revalue the option.

At minimum incorporate:

Underlying Move
Time Decay
IV Move

Where possible use the platform’s option-pricing implementation.

For a scenario:

S0 → S1
IV0 → IV1
t0 → t1
OptionPrice0
OptionPrice1
Scenario P&L =
OptionPrice1 - OptionPrice0

Then aggregate portfolio P&L.

⸻

23. OPTION GREEKS STILL MATTER

Continue portfolio aggregation for:

Delta
Gamma
Theta
Vega

But interpret Greeks as:

local sensitivity measures

not complete tail-risk measures.

Risk Dashboard should display both:

Portfolio Greeks
AND
Scenario / ES Risk

⸻

24. VOLATILITY SHOCKS FOR OPTIONS

Historical underlying-return simulation alone is insufficient for options.

Design IV stress scenarios.

Examples should be derived empirically where possible, not arbitrary.

Potential scenarios:

Equity -5% / IV +20%
Equity -10% / IV +40%
Equity +5% / IV -15%
Volatility crush
Volatility spike

Claude should determine practical parameterization from available historical data.

Do not blindly adopt these example numbers.

⸻

25. HISTORICAL STRESS TESTING

Create a historical scenario framework.

Examples may include stressed market windows present in available data.

The system should ask:

What would the CURRENT portfolio lose if a historically stressed return/volatility pattern happened now?

For options:

revalue the current option portfolio under the historical underlying and IV shock.

⸻

26. HYPOTHETICAL STRESS TESTING

Support configurable scenarios.

Examples:

SPY -5%
QQQ -8%
Tech correlation → 0.9
IV +30%

or:

Underlying flat
IV -40%

to detect long-option volatility-crush exposure.

Stress tests should become part of pre-trade risk review.

⸻

27. STRESS LOSS GATE

Potential structure:

Max Stress Loss
Stress Loss / NAV

Risk Engine may:

APPROVE
RESIZE
REJECT

based on configured limits.

A position may pass VaR but fail Stress Test.

The Stress Test must retain veto authority.

⸻

28. MARKOWITZ MEAN-VARIANCE — USE CAREFULLY

Implement or preserve Mean-Variance Optimization as:

research
comparison
allocation benchmark

Useful outputs:

Minimum Variance Portfolio
Tangency Portfolio
Efficient Frontier

However do not blindly use maximum Sharpe / sample expected-return optimization for production allocation.

Expected-return estimates are unstable and can produce extreme portfolio weights.

⸻

29. GLOBAL MINIMUM VARIANCE

Evaluate:

Global Minimum Variance

as a more robust baseline than unconstrained maximum-Sharpe allocation.

Use long-only constraints consistent with account permissions.

Also include:

cash

as a legitimate portfolio component where architecture permits.

⸻

30. ROBUST PORTFOLIO OPTIMIZATION

Investigate improvements to ordinary sample estimates.

Possible approaches:

robust location estimation
robust covariance estimation
winsorization/trimming only if justified
covariance shrinkage
robust optimization

Do not automatically implement all.

Compare:

Classical covariance
vs
Robust covariance

and measure portfolio stability.

⸻

31. TURNOVER STABILITY

One risk of unconstrained optimization is unstable portfolio weights.

Add metrics:

weight turnover
allocation change
concentration

Penalize portfolio allocations that change excessively for small input changes.

An “optimal” portfolio that changes radically every day may be operationally inferior.

⸻

32. EQUAL RISK CONTRIBUTION

Implement or evaluate:

ERC — Equal Risk Contribution

The objective is NOT equal capital.

It is roughly:

each position contributes a controlled/equal share of portfolio risk.

This is highly relevant to the platform.

Use it as an allocation benchmark.

⸻

33. CVaR / ES RISK BUDGETING

Evaluate a downside-risk allocation method:

ES Contribution

rather than only volatility contribution.

This is especially important for:

options
fat-tailed stocks
high-volatility assets

A recommended production direction is:

signal determines candidate
allocation determines size
ES contribution constrains concentration

⸻

34. DIVERSIFICATION RATIO

Evaluate whether the:

Most Diversified Portfolio
Diversification Ratio

adds useful information to the existing portfolio allocator.

It may be useful as:

diagnostic
benchmark
allocation candidate

rather than necessarily the default allocator.

⸻

35. TAIL DEPENDENCE

For an advanced phase, investigate:

lower-tail dependence

rather than ordinary correlation alone.

A pair of assets may appear diversified during normal periods but fall together during crashes.

This is exactly the dependence that matters most for risk management.

⸻

36. CASH IS A RISK ASSET ALLOCATION DECISION

Preserve the existing dynamic cash floor.

But investigate linking cash allocation to quantitative risk conditions:

Forecast Volatility
Portfolio ES
Stress Loss
Drawdown
Correlation Regime
Model Health

Conceptually:

Risk ↑
→ Required Cash Floor ↑

Do not let the optimizer automatically force 100% deployment.

⸻

37. RISK-BASED POSITION SIZING V2

Current sizing may use:

NAV × risk budget

Keep this as a hard upper bound.

Add statistical sizing modifiers.

Potential final sizing concept:

Base Risk Budget
× Signal Modifier
× Volatility Modifier
× ES Modifier
× Correlation Modifier
× Model Health Modifier
=
Candidate Risk Budget

Then:

Final Risk =
minimum of:
Candidate Risk
Single Name Cap
Portfolio Heat Headroom
Cash Constraint
Stress Loss Limit
ES Limit
Risk Contribution Limit

⸻

38. DO NOT LET SIGNAL CONFIDENCE DOMINATE RISK

Even an extremely strong signal cannot bypass:

ES Limit
Stress Limit
Tail Risk
Cash Floor
Portfolio Heat
Account Permission
Liquidity
Kill Switch

Trading Signal asks:

Should we want this trade?

Risk Engine asks:

Can the portfolio safely absorb this trade?

Risk Engine is sovereign.

⸻

39. RISK MODEL ENSEMBLE

Avoid a single-model dependency.

Create a Risk Snapshot containing several views:

Historical VaR
Historical ES
Gaussian VaR
Gaussian ES
Conditional Volatility
Conditional VaR
Conditional ES
Stress Loss
EVT Tail Risk if reliable
Portfolio Drawdown
Risk Contribution

Do not average these blindly into one opaque number.

Show model disagreement.

⸻

40. MODEL DISAGREEMENT IS INFORMATION

Example:

Gaussian VaR    $1,200
Historical VaR  $1,800
GARCH VaR       $2,400
Stress Loss      $5,100

This should trigger:

MODEL DISPERSION HIGH

not:

Average Risk = $2,625

Large disagreement often indicates unstable market conditions or model risk.

⸻

41. RISK MODEL HEALTH

Every model should expose:

ACTIVE
DEGRADED
UNAVAILABLE
FAILED

with reason.

Examples:

GARCH:
DEGRADED — convergence issue
EVT:
UNAVAILABLE — insufficient tail observations
Historical ES:
ACTIVE

Risk Engine needs fallback logic.

⸻

42. BACKTEST THE RISK MODELS

Do not merely backtest trading returns.

Backtest risk forecasts.

For VaR:

track exceedances.

Example:

99% VaR

should be breached roughly at the expected frequency over a sufficiently large sample if calibrated.

Measure:

VaR exceedance rate
clustered exceedances
ES realized severity
volatility forecast error

Optionally implement standard coverage tests if appropriate to current dependencies.

Claude has discretion over the exact statistical tests.

⸻

43. WALK-FORWARD ONLY

Risk models must not use future information.

Every historical risk estimate must use only information available at that historical timestamp.

No full-period covariance.

No future volatility.

No future IV.

No hindsight threshold calibration.

⸻

44. MODEL VERSIONING

Every generated risk number must be reproducible.

Store:

model_name
model_version
parameters
data_window
data_source
as_of_timestamp
confidence_level
horizon
distribution
diagnostics

⸻

45. PORTFOLIO RISK SNAPSHOT DOMAIN MODEL

Design a typed object similar conceptually to:

PortfolioRiskSnapshot
NAV
Cash
Gross Exposure
Delta-Adjusted Exposure
Volatility
VaR95
VaR99
ES95
ES99
ConditionalVaR
ConditionalES
StressLoss
CurrentDrawdown
NetDelta
NetGamma
NetTheta
NetVega
PositionRiskContributions
SectorRiskContributions
CorrelationState
ModelHealth
RiskState

Do not create an untyped JSON dumping ground.

⸻

46. PRE-TRADE RISK COMPARISON

Every Trade Plan should eventually show:

                    CURRENT     AFTER TRADE
Portfolio Heat       3.2%          4.0%
Cash                 42%           38%
VaR 99%              $X            $Y
ES 99%               $X            $Y
Stress Loss          $X            $Y
Net Delta            X             Y
Net Vega             X             Y
Tech Risk Contr.     X%            Y%
Single-name RC       X%            Y%

The user should see what the trade does to the entire portfolio.

⸻

47. RISK DECISION EXPLAINABILITY

Risk decisions must be explainable.

Example:

APPROVE WITH RESIZE
Requested:
4 contracts
Approved:
2 contracts
Binding constraints:
1. 99% ES contribution
2. Technology risk concentration
3. Volatility forecast adjustment
Without resize:
Portfolio ES would increase 31%.
After resize:
Portfolio ES increases 14%.

Do not output:

RISK SCORE = 63

without explanation.

⸻

48. RISK DASHBOARD UPGRADE

Inspect existing UI before redesigning.

Add only what improves decision quality.

Recommended top-level cards:

NAV
Cash
Portfolio Heat
1D VaR 99%
1D ES 99%
Conditional Volatility
Stress Loss
Current Drawdown
Net Delta
Net Vega

Then sections:

Risk by Position
Risk by Sector
Risk Contribution
Correlation / Dependence
Stress Scenarios
Model Health
Historical Risk

⸻

49. RISK CONTRIBUTION VISUALIZATION

Create a useful risk-contribution chart/table.

Example:

NVDA         24%
QQQ          19%
AAPL         13%
MSFT         11%
Other        ...

Compare:

Capital Weight
vs
Risk Weight

This distinction must be visible.

⸻

50. RISK MODEL DETAILS UI

Every risk metric must have:

ⓘ How is this calculated?

Show:

Model
Confidence
Horizon
Lookback
Distribution
Last Updated
Data Source
Model Health

For sophisticated models include diagnostics under:

Advanced

⸻

51. STRESS TEST UI

Create scenario cards/table.

Example:

SCENARIO               PORTFOLIO P&L
Historical Stress A       -$X
Historical Stress B       -$Y
Equity Shock              -$Z
IV Spike                  +$A / -$B
IV Crush                  -$C

Allow user-defined hypothetical scenarios later if architecture supports it.

⸻

52. STOCK VS OPTION RISK DISPLAY

For stock:

display:

volatility
VaR / ES
beta/correlation
stress loss
risk contribution

For option:

display additionally:

Premium at Risk
Delta
Gamma
Theta
Vega
DTE
Underlying exposure
Volatility sensitivity
Scenario loss

Do not present them as identical instruments.

⸻

53. DATA FREQUENCY

Claude should inspect current strategy horizons before selecting data frequency.

Avoid one universal risk horizon.

Possible mapping:

intraday strategy → intraday risk layer
swing strategy → daily risk models
portfolio capital risk → daily / multi-day

Do not run computationally expensive GARCH/Copula every tick unless necessary.

⸻

54. COMPUTATIONAL ARCHITECTURE

Risk model calculations should not block live order-request latency unnecessarily.

Separate:

fast pre-trade risk

from:

slow model recalculation

Potential architecture:

Market Data
     ↓
Risk Model Workers
     ↓
Risk Snapshot Cache
     ↓
Fast Risk Engine
     ↓
Trade Decision

Heavy calculations such as:

GARCH fitting
EVT
Copula Monte Carlo
robust optimization

can be asynchronous/background model jobs.

⸻

55. CACHING

Risk Snapshot must include an as_of timestamp.

Risk Engine must reject stale snapshots according to model-specific TTL.

Example:

Portfolio Greeks:
seconds
Daily GARCH:
hours/day
Robust covariance:
daily
EVT:
daily/weekly depending on design

Claude should determine reasonable lifecycle policies.

⸻

56. DATABASE DESIGN

Do not store only the latest value.

Persist enough history to:

audit
backtest
compare model forecasts
measure model drift

Potential entities:

risk_model_runs
risk_snapshots
risk_metrics
stress_runs
risk_contributions
model_diagnostics
optimization_runs

Claude may redesign names to fit existing architecture.

⸻

57. MODEL VALIDATION SERVICE / MODULE

Create a logical separation between:

MODEL OUTPUT

and:

MODEL VALIDATION

Risk calculation code should not automatically claim a model is healthy.

Validation should assess:

data sufficiency
fit
stability
forecast performance
parameter sanity

⸻

58. FALLBACK HIERARCHY

Risk Engine must degrade gracefully.

Example:

GARCH fails
        ↓
Historical ES still available
        ↓
Stress tests still available
        ↓
Hard limits still active

Do not halt all risk control because one advanced model fails.

However if critical data itself is invalid:

PAUSE TRADING

⸻

59. MODEL RISK ITSELF IS A RISK

Create:

Model Risk State

Possible:

LOW
ELEVATED
HIGH

Inputs may include:

model disagreement
forecast instability
failed diagnostics
insufficient data
rapid regime transition

Model Risk can reduce allowable risk budget.

⸻

60. CURRENT ACCOUNT CONSTRAINTS MUST REMAIN

Inspect current configuration and preserve the intended real-account limitations.

The paper environment should continue to mirror real-account permissions.

Do NOT enable strategies simply because Alpaca Paper technically supports them.

Preserve, where currently configured:

No short stock
No naked short call
No naked short put
No unsupported margin strategies

Risk-model upgrade must not broaden permissions.

⸻

61. IMPLEMENTATION PHASES

You have discretion to modify phase boundaries after inspecting the repository, but explain all deviations.

PHASE A — RISK AUDIT

No implementation first.

Output:

Current Risk Architecture
Existing Models
Existing Hard Limits
Missing Statistical Risk
Data Available
Technical Debt
Recommended Priority

⸻

PHASE B — CORE RISK METRICS

Implement:

Historical VaR
Historical ES
Portfolio Volatility
Drawdown
Risk Contribution foundation

Integrate with existing Risk Snapshot.

⸻

PHASE C — PRE-TRADE PORTFOLIO RISK

Implement:

Current vs Proposed Portfolio
Incremental VaR / ES
Marginal Risk
Risk Contribution

Integrate into:

APPROVE
RESIZE
REJECT

⸻

PHASE D — STRESS TEST ENGINE

Implement:

Historical Scenarios
Hypothetical Scenarios
Stock revaluation
Option full revaluation

Add Stress Loss gate.

⸻

PHASE E — CONDITIONAL VOLATILITY

Implement or research:

GARCH(1,1)
conditional volatility
conditional VaR
conditional ES

Add model diagnostics.

⸻

PHASE F — ROBUST ALLOCATION

Compare:

Current Allocation
Minimum Variance
Robust Minimum Variance
ERC
ES-risk budgeting

Do NOT automatically replace current allocation.

Backtest first.

⸻

PHASE G — ADVANCED TAIL RISK

Research:

EVT / POT / GPD

Only promote to production if diagnostics are stable.

⸻

PHASE H — ADVANCED DEPENDENCE

Research:

Copula
Student-t Copula
GARCH-Copula
Tail Dependence

Only productionize if it materially improves risk calibration.

⸻

PHASE I — RISK UI

Upgrade:

Risk Dashboard
Trade Plan Risk Comparison
Risk Contribution
Stress Testing
Model Health
Model Explainability

⸻

PHASE J — ADVERSARIAL VALIDATION

Simulate:

fat-tail crash
volatility spike
correlation convergence
IV crush
extreme long gamma
extreme long vega
concentrated tech exposure
GARCH model failure
stale risk snapshot
EVT insufficient data
copula failure
broker/data mismatch

Verify fail-safe behavior.

⸻

62. IMPORTANT — NO MODEL FOR MODEL’S SAKE

Before implementing each advanced model answer:

1. What risk does this model capture that current system misses?
2. Is sufficient data available?
3. Can the result be validated?
4. Does it affect an actual decision?
5. Is computation appropriate for production?
6. Is there a simpler model producing nearly the same value?

If the answer is weak:

DEFER IT.

⸻

63. REQUIRED MODEL COMPARISON

Whenever practical compare competing approaches.

Example:

Gaussian VaR
Historical VaR
GARCH VaR
EVT VaR

or:

Sample Covariance
Robust Covariance

or:

Current Allocation
GMV
ERC
ES Risk Budget

Measure:

forecast accuracy
exceedance behavior
drawdown
portfolio turnover
concentration
risk-adjusted return
stability

Do not select a model because it is mathematically more sophisticated.

⸻

64. BACKTEST / PRODUCTION PARITY

Use the same risk-model library for:

historical replay
paper trading
live trading

Never implement one calculation in a notebook and a different one in production.

⸻

65. OBSERVABILITY

Add metrics where appropriate:

risk_model_latency
risk_snapshot_age
garch_fit_failures
var_exceedances
es_exceedances
stress_limit_blocks
risk_resize_count
risk_reject_count
model_health_state

⸻

66. AUDIT TRAIL

Every Risk Decision should preserve:

Input Portfolio
Proposed Trade
Hard Limits
Risk Snapshot
Model Versions
Model Health
Binding Constraint
Decision
Requested Size
Approved Size
Timestamp

⸻

67. TESTING REQUIREMENTS

At minimum test:

VaR arithmetic
ES arithmetic
portfolio aggregation
risk contribution sums
incremental risk
stock scenario P&L
option scenario P&L
GARCH fallback
stale snapshot
stress gate
correlation increase
cash floor
portfolio heat
account permission
model failure

Property tests should verify:

A rejected strategy never reaches broker.
Risk contribution reconciles with total risk
within numerical tolerance where mathematically applicable.
Increasing a position cannot accidentally reduce its
standalone max-loss calculation.
A failed advanced model never disables hard limits.
A stale critical risk snapshot cannot authorize new risk.

⸻

68. MODEL VALIDATION ACCEPTANCE

Do not call a model “production ready” until:

Unit Tested
Replay Tested
Out-of-Sample Tested
Diagnostics Exposed
Failure Mode Tested
Fallback Tested
Audit Tested

⸻

69. DEVLOG REQUIREMENT

After every loop append:

Purpose
Risk hypothesis
Existing capability discovered
Implementation
Model assumptions
Data used
Validation
Test results
Model limitations
Production status
Next recommendation

Use statuses:

RESEARCH
SHADOW
PRODUCTION
DEPRECATED

⸻

70. SHADOW MODE

This is highly recommended for new statistical risk models.

A new model can run in:

SHADOW

meaning:

* calculate risk
* display results
* log hypothetical approve/reject
* DO NOT alter trading yet

Then compare it with realized outcomes.

After sufficient validation:

SHADOW → PRODUCTION

Use this especially for:

GARCH
EVT
Copula
Robust Optimization

⸻

71. FINAL RISK ENGINE PRINCIPLE

The target architecture should embody:

Hard Limits prevent catastrophic policy violations.
VaR estimates ordinary downside thresholds.
Expected Shortfall measures severity beyond VaR.
GARCH adapts risk to changing volatility.
EVT studies the extreme tail.
Copulas model dependence beyond simple correlation.
Stress Testing asks what happens when models are wrong.
Risk Contribution identifies hidden concentration.
Robust Optimization avoids unstable allocation estimates.
ERC / ES budgeting allocate risk rather than merely capital.
Greeks describe local derivatives exposure.
Full Revaluation captures nonlinear option tail behavior.
Model Validation determines whether any of these numbers
deserve to influence a trade.

⸻

72. FINAL AUTHORITY HIERARCHY

The production decision hierarchy must remain:

ACCOUNT PERMISSION
        ↓
DATA QUALITY
        ↓
HARD RISK LIMITS
        ↓
STATISTICAL RISK
        ↓
STRESS RISK
        ↓
PORTFOLIO CONCENTRATION
        ↓
MODEL HEALTH
        ↓
APPROVE / RESIZE / REJECT

The trading signal must never override the Risk Engine.

The LLM must never override the Risk Engine.

The user may reduce risk manually.

The user must not accidentally bypass hard limits through the normal trading workflow.

⸻

73. FINAL DELIVERABLE

Do not merely implement code.

At the end of this Claude loop produce:

A. Risk Architecture Review

What existed before the upgrade.

B. Risk Model Matrix

For every model:

Purpose
Status
Data
Assumptions
Strength
Weakness
Decision Usage

C. Implementation Report

What was actually built.

D. Deferred Models

What was intentionally not built and why.

E. Validation Report

Backtest / shadow / stress / failure-mode results.

F. UI Changes

How risk information is exposed to the user.

G. Remaining Model Risk

What the platform still cannot reliably measure.

⸻

74. PRIMARY OBJECTIVE

Do not optimize this project for:

the most sophisticated mathematical model.

Optimize it for:

the best combination of risk coverage, robustness, explainability, computational practicality, and real decision value.

The final system should answer, before every trade:

How much can we lose normally?

How much can we lose in the tail?

What if volatility changes?

What if correlations rise?

What if the statistical model is wrong?

Which position contributes the risk?

Does this trade improve or damage portfolio diversification?

How large can the position safely be?

Only then should the system decide whether capital may be deployed.