# TD-One Presentation: Full Context for Continuing the Work

> **Status: HISTORICAL — point-in-time record, dated 2026-10-20. deck-of-record context; kept with slides/TD-One-Slides.html**

**Purpose of this file.** Everything a person (or an agent working for
them) needs to continue the TD-One final presentation work without
re-deriving it: the verified facts about the codebase, the experiment
verdicts, the exact messaging rules the presenter wants, what has been
produced so far, and what is still open.

**Version note.** This is the second edition, written 2026-10-20. The
first edition (same date, earlier in the day) is superseded. Section 5
lists everything that changed. Ground truth lives in
`td-one/docs/FINAL_RESULTS.md`, `td-one/docs/PHASE2_UPDATE.md`,
`td-one/docs/METHODOLOGY.md`, and `td-one/experiments/README.md`. If
this file and those disagree, those win, and this file should be fixed.

**How to use it.** Read sections 1 to 3 for the situation and the
rules. Section 4 is the factual ground truth about the system; do not
contradict it in slides without re-checking the cited repo files.
Section 5 is what changed in this edition. Sections 6 and 7 are the
state of the deliverables and the open work.

---

## 1. The situation

- TD-One is a Vector Institute Agentic Forecasting Bootcamp project by
  a five-person TD team (Mehrazin Alizadeh, Vandad Davoodnia, Fateme
  Ghadami, Mohammad Hassan Khatami, Nazanin Mehregan). The build
  (`td-one/` in the repo) was done solo by Vandad Davoodnia from
  starting materials prepared by Mehrazin.
- The deliverable is the bootcamp **Final Deck**: a presentation of
  under 10 minutes for a Vector AI Ecosystem audience (companies and
  research community). No proprietary info.
- The presenter did not write the code and is learning the system from
  the outside in. Explanations must be high level first, plain
  language, defined terms, diagrams and concrete numbers over prose.
  This constraint also shapes the slides: the audience should be
  assumed smart but new.

## 2. What has been produced

- **`TD-One-Slides.html` (this directory)** is the deck of record from
  2026-10-20. 8 slides as HTML sections: Executive Summary; Business
  Problem and Opportunity; Our Solution; Technical Approach; Results;
  What Did Not Work; Example Manager Report; Future Work. The
  presenter converts this to slides himself.
- `TD-One-Final-Deck-filled.pptx` and `build_deck2.py` (older PPTX
  attempts) are parked. Do not keep the PPTX in sync; if a native PPTX
  is required for submission, regenerate it from the HTML content,
  respecting section 3.
- The Results slide's three segment panels are inline SVG drawn from a
  real run: the repo simulator (5,000 customers, 66 months, seed 42)
  and the repo segment panels, generated 2026-10-20. Forecast = the
  trailing-6-month pooled switching rate; range = pooled rate plus or
  minus 1.28 times the standard deviation of the last 12 monthly
  rates. Fixed rate: guess 49%, range 24 to 73%. Variable rate: guess
  79%, range 67 to 91%. Highest payment burden: guess 68%, range 41 to
  96%. All actuals land inside all three ranges.

## 3. Messaging and style rules (hard requirements from the presenter)

1. **No em dashes.** They read as machine-generated to him. Use commas,
   periods, or restructure. Write the way a team member would write.
2. **Slides carry shapes, not narration.** Blocks, diagrams, formulas,
   plots, short labels. Anything long belongs in what the presenter
   says, not on the slide.
3. **No jargon without a plain meaning.** Example of a rejected line:
   "enforced in code and audited." Accepted replacement: "a forecast
   for a date uses only information that existed on that date."
4. **Never overpromise external context (H).** The evidence is that
   news, rates, and offers never improved a forecast in this project.
   Slides must present H honestly: tried repeatedly, did not help the
   numbers. News legitimately lives in the report text (with citations)
   and, since exp77, in the range layer only (see section 5).
5. **Keep the test worlds separate in every claim.** World A: the
   synthetic mortgage data built by the team. World B: standard public
   datasets anyone can download (forecasting competitions M3 and M4,
   Tourism, US food prices, US Freddie Mac mortgage data). Real
   Canadian mortgage data (OSFI M4) is a third block and must be
   labeled as such, never folded into either world. The headline wins
   are World B results. Never phrase them so they sound like mortgage
   results.
6. **Claims must match evidence exactly**, including the uncomfortable
   parts (on Tourism a pretrained model alone beats the committee; the
   committee was never run per-segment on the synthetic mortgage book;
   one language model prompt alone beat the full committee on food
   prices). The presenter's trust in this work comes from the honest
   negatives; keep them visible where they belong.
7. Short, dense, high-information text. Define every abbreviation on
   first use. Numbers beat adjectives.

## 4. Ground truth about the system (verified against the repo)

### 4.1 The problem and the presenter's formulation

Forecast, per customer segment s and month m, the distribution of
switchers A given renewals due N (known exactly), past A and N,
customer features X up to time t, and external context H up to time t:
P(A_{s,m} = a | N_{s,m}, past A, past N, X_{<=t}, H_{<=t}).
N is known; A is the unknown. Output per segment: a best guess and an
honest range (an 80% interval in most components). Segments are groups
of customers; see 4.3 for how they are defined.

**Decision (resolved 2026-10-20):** the Solution slide shows both
formulas. The full formula is labeled "Design goal"; the reduced
formula P(A | N, segment history) is labeled "What the built system
uses today", with one line explaining where X and H actually enter.

### 4.2 The data

- **Synthetic mortgage world** (`td-one/src/td_one/data_gen/simulator.py`,
  plus `simulator_v2.py` adding TD and competitor renewal offers):
  10,000 customers x 66 months from Apr 2021, 46 fields per
  customer-month, exactly the schema in the team's data contract.
  Switching is generated by a structural rule involving payment shock,
  rate spread, a family-squeeze term, an OSFI-exemption term, and
  cohort random effects. Planted cohorts (the answer key): `low_rate`
  (fixed, originated 2021, rate at most 3%; verified switching about
  68% vs about 39% for control at seed 42, n=2000), `osfi_exempt`,
  `family_squeeze`, control. Effects other than low_rate are weak at
  segment level.
- **Real public data**: Freddie Mac mortgage leaving rate (63 months),
  US food prices (see provenance note below), Bank of Canada decisions,
  WTI oil, M3/M4/Tourism competition series, OSFI M4 Canadian bank
  runoff (121 months, 4 segments), FRED macro series, and a corpus of
  17,906 dated news articles (544 with full bodies).
- **Provenance correction (2026-10-20):** the series called "Food CPI"
  throughout the repo is **FRED CPIUFDSL**, the US CPI-U for Food,
  monthly, seasonally adjusted, 2010-01 to 2026-09. Earlier docs
  (including the first edition of this file) mislabeled it "Canadian
  food prices". The data and every result are unchanged; only the
  label was wrong. `benchmarks/food_cpi/data.csv` was rebuilt 1:1
  from FRED's CSV after a working-tree deletion. Note the collision:
  OSFI M4 (the Canadian regulator's mortgage balance report) is not
  the M4 forecasting competition.
- **Point-in-time discipline**: every forecast for a date uses only
  information available on that date; outcomes lag 2 months; a leakage
  auditor exists. This is real and central to the project's
  credibility.

### 4.3 Segments: what exists

- **Forecast segment panels** (`src/td_one/segments/discovery.py`):
  15 series total, from four fixed splits: payment-to-income quintiles
  (5), rate type fixed/variable (2), province ON/BC/AB/QC/MB (5), term
  short/medium/long (3). A customer sits in one group per split.
- **Grouping experiments** (exp66 line): hand-written groups (Ontario
  income under/over 100k, BC, everyone else); LLM-proposed groups;
  deterministic KMeans groups with decision-tree rules (exp66c).
- The poster's example segment (income under $100k with 3+ children)
  is NOT a group any method forecasts; it exists only inside the
  generator as the family_squeeze plant.

### 4.4 The current best model (updated: three layers, not one)

Source of truth: `td-one/docs/FINAL_RESULTS.md` and
`td-one/docs/PHASE2_UPDATE.md`. Code:
`td-one/src/td_one/agent/proposed_method.py` (v2),
`td_one/agent/meta_router.py` (Layer-2),
`td_one/uq/` (the range layer).

**Layer 1, Proposed Method v2** (exp43 router, exp47 v1 assembly,
exp55 to exp58 LLM ladder, exp64 v2 unified run). Pipeline for ONE
input series (series-agnostic; it does not know what the numbers
mean):

1. **Features**: about 10 point-in-time-safe statistics of the series
   history (seasonal strength, trend strength, volatility ratio,
   recent-changepoint count, and similar).
2. **Regime labels (hand rules)**: seasonal_strength > 0.5 ->
   "seasonal"; else trend_strength > 0.7 -> "trending"; else "noisy".
   A generic two-way variant: recent changepoint or vol_ratio > 1.0 ->
   "stressed", else "calm". Food prices use calm/volatile.
3. **Router**: a logistic regression trained on past series labeled by
   the hand rules outputs P(regime | features) for a new series. There
   is no hard assignment at forecast time.
4. **Expert pool (exactly four classical models)**: naive (seasonal
   naive), m0 (the project's recent-rate Beta-Binomial baseline), ETS,
   ARIMA. **Per-regime expert selection**: for each regime, the expert
   with the lowest mean CRPS on training records of that regime becomes
   that regime's designated expert (fallback: global best single if a
   regime has under 10 records). On M4 data the outcome was
   seasonal->ARIMA, trending->ETS, noisy->ETS.
5. **Combination**: final quantiles = sum over regimes of
   P(regime) x (that regime's expert's quantiles), averaged at the
   quantile level (Vincentization). There are NO free per-expert
   weights; an expert's effective weight is the sum of the
   probabilities of the regimes it serves.
6. **"Do nothing" safety gate**: the combination is discarded in favor
   of the global best single model only if ALL three hold: top regime
   probability < 0.6, |t-statistic for a level shift| < 3.5, estimated
   shift < 0.03. Rationale: an uncertain mix over a series showing no
   real change invents movement.
7. **Conformal calibration**: the final range is widened using a
   correction computed from training residuals (target 80% coverage),
   with a volatility floor.
8. **Trust flag**: EASY if top regime probability >= 0.7, else HARD.
   Reporting only.
9. **v2 addition, the LLM expert**: Gemini Flash as an extra
   forecaster, gated hard OFF for the datasets tourism/m3/m4 and for
   any series with seasonal_strength >= 0.5; otherwise it joins the
   pool and the router machinery decides per regime. In the final runs
   it changed the result on exactly one dataset (food prices).

**Layer 2, the validation gate** (exp73,
`src/td_one/agent/meta_router.py` plus `layer2_tools.py`). A
deterministic decision stage wrapped around v2 for single-series
targets. It calls a set of analyst tools (series diagnostics,
backtests, changepoint scan, a lock-in meter), then decides per
forecast origin: trust the router, or abstain to the naive forecast.
Every decision emits an auditable trace. Results on OSFI runoff
(pooled, horizon 1, n=300): naive 0.00369, v2 0.00381, gated 0.00374.
The gate's real value is harm repair: it blocked v2's uninsured
segment harm (+10.7%, p=0.030, down to +0.8%, not significant), and at
horizon 3 the gated system beats v2 by 6.8% (p=0.015). exp72
confirmed it also protects the group forecaster (-0.021 to -0.031,
p<1e-8). The lock-in correction arm failed its pre-registered bar
(the lock-in level carries no signal on this target, |r| <= 0.043);
the gate is what ships, not the correction.

**Layer 3, the range layer: CQR** (exp74, `src/td_one/uq/`).
Conformalized quantile regression is now the default way ranges are
sized. On food prices it beat the model's native ranges on error
(CRPS 1.2824 vs 1.3035, p=1.7e-04), put reality inside the 80% range
82.4% of the time and inside the 90% range 90.4% of the time, at a
width cost of 7.8%. ACI (adaptive conformal inference) is a
regime-gated option for volatile subsets (p=3.6e-05 there).
**Caveat that must survive:** exp74 also reported that story features
from news improved the uncertainty model (paired p=2.0e-11). exp76
found a one-month keying offset in exp74's feature construction (a
lookahead). The direction is plausible; the size is being
re-measured. Do not cite the 2.0e-11 figure as settled.

**Headline results (CRPS, lower is better; paired t-tests)**:
M4 Monthly (100 series): v2 0.1451 vs equal weights 0.1507 vs best
single 0.1505 (-3.6%, p=7.8e-11). Tourism (366): 0.1282 vs equal
0.1869 vs best single 0.1387 seasonal naive (-7.6%, p=2.6e-34).
M3 (1,428): 0.1476 vs 0.1641 vs 0.1630 ETS (-9.4%, p=3.5e-35).
Food prices (59 origins): v2 0.5322 vs v1 (classical only) 0.9661
(-45%, p=4.3e-06). BoC decisions (discrete variant): 72.2% accuracy
vs 69.4% best classical on 119 meetings.

**Caveats that must survive any rewrite**:
- On M4, Tourism, M3, and BoC, the LLM expert never fires; v2 = v1
  there. The "agentic" content of the wins is mostly routing, gating,
  and calibration, not the LLM.
- Best single LLM prompt on food prices (prompt V1, standalone)
  scored 0.4262, BETTER than the full v2 hybrid 0.5322. The router
  adds noise when one expert dominates; v2's value is the gating, not
  the mix.
- **Foundation models are not in the pool.** exp50 tested adding
  calibrated Chronos-2 (real weights): selected as per-regime expert
  0% of the time on M4, CRPS identical with and without. Reason:
  winner-take-all per regime, and classical models already capture
  what those M4 series offer at that horizon. As STANDALONE
  forecasters the foundation models are the strongest singles in the
  project: TimesFM-2.5 Tourism CRPS 0.0965 (beats the committee's
  0.1282), Chronos-2 0.0972; on Freddie Mac SMM calibrated TimesFM
  0.000497 vs ETS (-12.2%, p=0.076, not significant). The TSFM-in-pool
  test was never repeated on Tourism.
- **The committee was never run per-segment on the synthetic mortgage
  book.** exp66b designed "full v2 per group"; its runs executed, but
  through `v2_forecaster_fn`, which (exp78) never fitted the router —
  every call returned the point-mass naive fallback, so no v2
  committee forecast ever ran per group. Mortgage segment runs that
  exist used M0, classical, and TSFM models (early experiments exp1
  to exp25). On the one US real
  mortgage series (Freddie Mac SMM), v2 tied its own classical version
  (p=0.69).
- **On OSFI M4 real Canadian runoff (exp70)**, v2 tied naive overall
  (-0.2%, p=0.54) and was significantly WORSE on uninsured mortgages
  (+10.7%, p=0.030). This was recorded as a kill trigger for that
  benchmark. Layer-2 (exp73) then repaired the harm; the honest
  sequence is: committee alone did not win on real Canadian runoff,
  and the gate is what made it safe to deploy there.

### 4.5 Inputs: where X and H actually enter (the input map)

- **Signal (past A and N per segment)**: the only input of every
  method that won. The LLM forecaster's prompt is the dated numbers
  plus a task description, nothing else.
- **X (customer features)**: used exactly once, BEFORE forecasting, to
  define the segments (fixed splits, hand rules, clustering, LLM
  grouper). A static snapshot at grouping time. No forecaster in the
  project consumes features alongside the series. X reappears in words
  in the group explanations.
- **H (external context)**: tried repeatedly as a forecast input;
  never helped on any dataset:
  1. FRED macro as covariates for the TSFM (exp13): apparent benefit
     reversed to a penalty under a point-in-time-safe rerun.
  2. News as a feature / regime augmentation (exp19, exp21, exp25,
     exp46): no effect on monthly SMM; discovered features lag the
     news (-21.6%); news worse by +25% at horizon 3 (p=0.0013); dense
     news definitively negative (exp46); bolting the news module onto
     M4 changed nothing (exp62, fired on 97% of origins).
  3. News shifting the router's regime probabilities (exp44 module,
     inside the proposed method): mechanism proven on synthetic
     (p=0.0068), dead on real monthly data; ships OFF by default.
  4. Competitor offer spread as a covariate (exp71, settled by
     exp71b): in-sample correlation -0.345 with 54 months of real
     broker rates, then failed out of sample (+39.3% error, p=0.019).
     A regime artifact. Closed at aggregate level; the only remaining
     question is at loan level.
  5. Anticipatory LLM overlay reading news to adjust a TSFM forecast
     ahead of events (exp69, exp69b, and exp69c on real episodes):
     +10.2% worse synthetic (p=0.231); even an oracle given the true
     shift was +11.4% worse; real episodes +23.7% worse (win rate 0%).
     The overlay mechanism itself is the limit.
  6. Calibrated macro signals into the router (exp71-MS on SMM, exp77
     on food prices): no effect on either dataset. Mechanism
     diagnosis: every origin in the signal window was "calm", the
     router was degenerate, and both regime experts collapsed to the
     same model, so no signal could express itself. The calibration
     question is untested by this route, not refuted.
  - The one near-exception is prompt formatting, not H: on food
    prices, LLM-alone with numbers only scored 0.5158; adding the SAME
    history restated in words (last value, 12-month and 3-month
    change, last year's high and low) scored 0.4262 (prompt V1).
    Adding true external context on top (BoC policy rate with as-of
    date, prompt V2) scored 0.4599, worse than V1 (p=0.20). Caveat:
    the repo's BoC rate series ends Dec 2023, so later origins
    received stale context, flagged as stale in the prompt.
  - Where H legitimately lives today: the explanation layer (4.6)
    and the **Macro Signal Service** (see 5.3): an LLM reads one
    month of news per category and emits a structured signal
    (direction -2 to +2, risk level, certainty). 725 signals, 22
    categories, 36 months. Calibration is done: 7 of 9 anchored
    categories are live (fx correlation +0.63, holdout accuracy 0.79;
    food prices holdout 0.82; also labour, inflation, housing,
    services costs, wars). The LLM's stated certainty does NOT
    calibrate. Per the user's 2026-10-20 directive, the signals are
    **uncertainty features, not forecast-router inputs**.
    exp76 (in flight) is measuring whether they improve the range
    layer over story-only features.

### 4.6 The explanation layer (the working H consumer)

`src/td_one/agent/risk_reporter.py`, used by the grouping line
(exp66c). Mechanics:

1. Code computes everything numeric: per-group forecast (predicted
   mean attrition + interval) and a risk label LOW/MEDIUM/HIGH from
   training-period tertiles, downgraded one level when the interval
   is wide. The LLM never produces a number or a label.
2. The explanation prompt contains: the group rule, the computed
   statistics (each citable as [stat]), and retrieved news articles
   published before the forecast origin (each citable as [url:N]).
   System contract: every factual sentence must end with a citation;
   never invent numbers or events.
3. A code checker validates citations; uncited factual claims fail.
   exp66c result: 10 deterministic groups, citation pass 10/10, but
   only under the tight contract that every sentence ends with a
   citation marker.
4. **Honest footnote**: in the exp66c run the reporter was invoked
   with `articles=[]`. The 10/10 explanations cite statistics only.
   The news pathway is wired in the prompt and contract but has never
   been run end to end with articles flowing in.

Grouping verdicts overall: LLM-proposed groups were unstable and
forecast worse than random groups (exp66: CRPS 0.01705 vs random
0.00690 vs global 0.00227; pre-registered kill). Deterministic groups
are reproducible and readable but forecasting per group still costs
accuracy vs the global model (+0.0033 CRPS on exp66c's own
window-mean scoring with the exp78-fixed forecaster, seed 42; the
originally published +0.009 to +0.016 was largely the point-mass
artifact — the broken global arm's point mass sat almost exactly on
the window mean, 0.00042, flattering the baseline; see the exp78
item in section 5), and static risk bins
INVERTED on test (LOW bin realized 0.128 vs HIGH 0.087) because
switching is driven by the time-varying rate path, not static traits.
Agreed position: grouping is an explanation layer; the forecaster of
record is global. Planted-cohort recovery was tested (exp66d to
exp66f): the deterministic grouper does not recover the planted
cohorts (ARI 0.01); SHAP-based clustering doubles ARI to 0.091 with
low_rate F1 0.55; still weak. exp66g settled the risk classifier:
LLM plus statistics only, no tool context, is the classifier of
record (beats deterministic bins pooled over 5 seeds, +0.0198, CI
excludes zero); the counterfactual tool's single-seed signal was
seed noise (shuffled-control proof) and stays an explanation device
only. The presenter's original design B (an agent reads news,
proposes at-risk segments, forecaster tests them, scored against the
planted answer key) has never been run in that agent form.

### 4.7 Other methods, one line each

- **M0 recent-rate baseline** (Jeffreys Beta-Binomial on recent
  counts; conformalized variant): the floor everything must beat.
- **Classical ETS/ARIMA/seasonal naive**: solid, in the committee
  pool.
- **TSFM standalone** (Chronos-2, TimesFM-2.5, real weights, with a
  log-scale calibrator): strongest single forecasters; see 4.4.
- **LLM as forecaster** (the ladder exp54 to exp58): great on smooth
  economy-like series (food prices), poor on seasonal ones; prompt
  format matters more than model choice (V1 words block).
- **M2 ReAct agent with tools** (Groq gpt-oss-120b, 13 tools) and
  **M3** (agent adjusts M0 in bounded log-odds), **M5 ensemble
  router**: all killed with evidence. Agent adjusting numbers is
  net-negative. Surviving agent roles are structural: the gated LLM
  expert, the explanation writer, and the macro signal reader (for
  ranges). The regime-label reader (M4 reader: outputs a class +
  confidence, never numbers) survives only as an optional, gated
  sensor in the proposed method.
- **Mini-agent orchestrator** (exp75): 24 factor agents (one per macro
  category) plus 4 diagnostics feeding a deterministic or LLM
  orchestrator. Routing added nothing over the Layer-2 gate on any
  bed; the LLM orchestrator was dominated by a deterministic arm.
  One genuine discovery: arm E, a deterministic signed shift, scored
  -22.8% on food prices at horizon 1 (p=1.9e-14, win rate 0.94), but
  harmed OSFI (+7.9%, p=0.019). It ships opt-in only, gated on a
  validation sign-accuracy test, off by default. Arm D (uncertainty
  mode) survives as a miscalibration rescue tool (food prices horizon
  6: -7.3%, p=3e-19, coverage 0.174 to 0.407). Net lesson: the agents
  are a feature store, not a decision engine.
- **Stress tests** (exp67: 7 synthetic shift types, severity ladder;
  exp68 shift guard): all models break together at extreme shifts;
  Chronos-2 best absolute in 6 of 8; the agentic guard detects shifts
  correctly but acting on detection did not help (+0.5%, p=0.824).
  exp72 ran the shifts against the group forecaster: groups do NOT
  isolate regional shocks (Q1 not supported, p=0.22, oracle geography
  identical); the min_group_n fallback is a working but blunt safety
  valve; Layer-2 protects the group forecaster in the thin-group
  regime (p<1e-8).
- **End-to-end manager report**: NOT built. Components exist (global
  forecaster, deterministic grouper, risk reporter) but no single run
  produces the report. The report slide in the deck is an illustrative
  mock built from a real synthetic run (fixed-rate segment: 65
  renewals due, forecast 48.6% switching, honest range 24.5% to 72.7%,
  so about 32 switchers, range 16 to 47). Do not present it as a
  screenshot of a working pipeline.

### 4.8 Numbers used in the current HTML slides (provenance)

- Segment panels (Results, left): generated 2026-10-20 by running the
  repo simulator (5,000 customers, 66 months, seed 42) and the repo
  segment panels; forecast = trailing-6-month pooled switching rate;
  range = pooled rate plus/minus 1.28 x the standard deviation of the
  last 12 monthly rates. Fixed rate: guess 49%, range 24 to 73%,
  next-6 actuals 52, 55, 65, 46, 52, 52. Variable rate: guess 79%,
  range 67 to 91%, actuals 87, 85, 84, 84, 66, 71. Highest payment
  burden: guess 68%, range 41 to 96%, actuals 83, 77, 67, 78, 64, 68.
  All actuals land inside all three ranges.
- Benchmark bars (Results, right): FINAL_RESULTS.md head-to-head
  table (committee vs best single): 0.145/0.151, 0.128/0.139,
  0.148/0.163, 0.532/0.966. Improvements vs best single: 3.6%, 7.6%,
  9.4%, 45%.
- Real mortgage block (Results, bottom): OSFI sequence from exp70
  (tie, -0.2%, p=0.54; uninsured +10.7%, p=0.030) and exp73 (gate
  repair to +0.8% n.s.; horizon 3 gated beats v2 by 6.8%, p=0.015).
  SMM from exp65 (neutral, p=0.69) and exp18b (TimesFM -12.2% vs ETS,
  p=0.076). CQR row from exp74 (1.2824 vs 1.3035, p=1.7e-04; coverage
  0.824 at 80%, 0.904 at 90%; width 1.078).
- Negatives slide: exp69/69b/69c (+10.2% synthetic, oracle +11.4%,
  real +23.7%); exp71b (out-of-sample +39.3%, p=0.019); exp77 (no
  effect, calm-window mechanism); exp75 (decider win -19.1% beaten by
  fixed rule -22.8%, p=1.9e-14); exp66 (LLM groups worse than random);
  exp78 early numbers (coverage 0% to 0.47 to 0.60 depending on spread
  method, CRPS down about a fifth; in progress, not a result).
- Executive summary stats: 75+ experiments (73 rows in
  experiments/README.md plus exp76 and exp78 in flight). "3 agent
  roles kept" = gated LLM expert, explanation writer, macro signal
  reader (ranges only). Dropped roles include: number adjuster, tool
  agent forecaster, group inventor, anticipatory overlay, mini-agent
  orchestrator, LLM combiner.

## 5. What changed in this edition

1. **Best model is now three layers**: v2 (committee + gated LLM
   expert) + Layer-2 validation gate + CQR range layer. The old
   section 4.4 described v2 alone.
2. **Food CPI provenance corrected**: the series is FRED CPIUFDSL (US
   CPI-U for Food), not Canadian. Fixed in FINAL_RESULTS.md, the
   benchmarks README, METHODOLOGY.md, and this file. Also fixed in
   FINAL_RESULTS.md: the stale Tourism figure 0.1288 in the detail
   table is now 0.1282 (the exp64 rerun value), matching the
   head-to-head table. The News table's 0.1288 there is a different
   number (the news-on run) and stays.
3. **OSFI verdict is now in the deck** (old open decision #3,
   resolved): the Results slide carries the full honest sequence
   (committee tied naive, harmed uninsured, gate repaired it).
4. **Formula decision resolved** (old open decision #1): both
   formulas shown, each labeled.
5. **New honest negatives with verdicts**: exp71b competitor spread
   (settled, regime artifact), exp69c anticipatory on real episodes
   (dead), exp75 mini-agent orchestrator (dead as a decision engine;
   arm E kept opt-in; arm D kept as a rescue tool), exp77 macro
   signals into the router (mechanism-blocked; signals reassigned to
   the range layer by user directive).
6. **Macro Signal Service finished calibration**: 7 of 9 anchored
   categories live. It is no longer "an unfinished side project".
7. **exp74 story-ablation caveat added**: exp76 found a one-month
   keying offset (lookahead) in exp74's story/macro features. The
   p=2.0e-11 figure is under re-measurement; the deck cites the CQR
   result only.
8. **exp72 group-forecaster defect — root-caused and fixed (exp78,
   complete)**: `v2_forecaster_fn` called `predict()` without ever
   calling `fit()`; the `AssertionError` was swallowed by a blanket
   `except`, so 100% of calls returned the last value tiled across
   all five quantile levels — the v2 router never ran in the group
   forecaster (interval coverage 0.000 on monthly actuals, seed 7
   control). exp66b/c comparability is affected (they scored window
   means, where point masses look fine). **exp78** shipped the fix:
   naive center + Gaussian spread from the series' own trailing
   one-step errors, now the default (`quantile_spread=False`
   reproduces the old output byte-for-byte). Validation
   (SimulatorV2, seeds 7/42/123): group CRPS 0.1320 to 0.1042 on
   control (−0.0287, p=4.2e-45, win 0.89), coverage80 0.075 to
   0.714 — much improved but still below the 0.80 target. The deck
   states this plainly on the negatives slide. Restoring the actual
   router in the group forecaster is exp79, in flight.
9. **Classifier of record settled** (exp66g): LLM plus statistics
   only. Counterfactual prompt context is dead (seed noise).
10. **exp76 in flight**: CQR x story fusion plus a UQ sweep across
    Tourism/M3/M4/OSFI/SMM, with corrected feature keying, and the S8
    ablation (story-only vs story+macro vs story+macro-live-7) as a
    first-class result. Clean S8 scope: food price origins 166 to 192
    (27 origins) where both story and macro signals exist.

## 6. Open decisions and open work

1. **Narration notes** still do not exist. The slides are deliberately
   minimal; a one or two sentence spoken cue per slide would help the
   presenter and is a good next task.
2. **exp76 verdict** will change the Future Work slide (range layer
   with news) when it lands: cite it, and retire the "in flight"
   wording.
3. **exp78 verdict** — landed: the grouping cost numbers in 4.6 are
   corrected in this edition (+0.0033 with the fix, seed 42). The
   negatives slide (group ranges row) still needs its update when
   deck work resumes.
4. Vandad's repo norm is direct commits to main; Mehrazin's fork works
   on branch `mehrazin/work` with fork main as a clean mirror of
   upstream. The merge-back path for any code work is undecided.
5. Queued program work that may reach the deck later: BoC MPR summary
   ingestion into the signal service; loan-level (Freddie Mac SFLLD)
   revisit of the competitor-spread question; the unannounced-shock
   synthesis test (exp69b caveat).

## 7. Repo pointers (relative to `td-one/`)

- `docs/slides/TD-One-Slides.html`: the deck of record (this
  directory). `docs/slides/TD-One-Context.md`: this file.
- `docs/FINAL_RESULTS.md`: headline results and architecture diagram.
- `docs/PHASE2_UPDATE.md`: everything since exp63, final + in-flight
  split, mortgage-framed.
- `docs/METHODOLOGY.md`: method spine.
- `docs/MACRO_SIGNAL.md`: the signal service: design, calibration,
  integration verdicts.
- `src/td_one/agent/proposed_method.py`: the flagship implementation
  (regime specs, router, gate constants GATE_TOP_P=0.6,
  GATE_T_STAT=3.5, GATE_SHIFT=0.03, conformal, trust flag).
- `src/td_one/agent/meta_router.py` + `layer2_tools.py`: Layer-2.
- `src/td_one/uq/`: the range layer (CQR, ACI, meta models, fusion).
- `src/td_one/agent/risk_reporter.py`: citation-enforced
  explanations.
- `src/td_one/agent/deterministic_grouper.py`: exp66c grouper.
- `src/td_one/llm_rungs/prompts.py` and `docs/PROMPT_LEAKAGE_AUDIT.md`
 : prompt variants (generic / V1 words / V2 macro context) with
  point-in-time proofs.
- `experiments/README.md`: the one-line verdict index for all
  experiments.
