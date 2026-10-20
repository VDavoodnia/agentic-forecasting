# Loan Attrition: Grouped Forecasting + TSFM Red-Teaming — Build Plan

> **Status: CURRENT — last updated 2026-10-20 (active program doc)**

**Status:** Track 1 built (2026-10-19); exp66 honest negative on CRPS.
**Reframed 2026-10-19:** grouping is for *explainability*, not for beating
CRPS. exp66b reruns with full v2 architecture to measure the explainability
gain vs CRPS cost. **Correction (exp78, 2026-10-20):** the "full v2
architecture" never actually ran in the group forecaster —
`v2_forecaster_fn` called `predict()` without `fit()`, the error was
swallowed by a blanket `except`, and every call returned a point-mass
naive fallback (see the exp78 entry under Track 1 below). Kill criterion is OFF — report cost/benefit honestly.

## exp66 verdict (2026-10-19)

❌ **Honest negative on CRPS.** LLM-discovered groups (CRPS 0.01705) did not
beat random groups (0.00690); global (0.00227) won. **But:** the test was
thin (140 episodes, stub forecaster). Reframed: the question is whether
LLM groups are *explainable* without degrading forecast quality vs
deterministic grouping. See `experiments/exp66_report.md` and exp66b below.

## exp66b: grouping for explainability (in progress)

Full v2 architecture per group. Prompt playground (base / +hints / +news /
+hints+news). Group-count sweep 3/5/8/12. Ensemble arm (LLM selects grouping
per window). Metrics: CRPS (parity check vs hand-written), explanation
quality (citation rate, rule readability), risk-bin calibration.
**No kill criterion** — report "costs X CRPS for Y explainability gain."

### Pivot (from exp66b): deterministic grouping + LLM explanations

LLM-proposed groupings are not reproducible run-to-run — fatal for
explainability. **exp66c** implements the pivot:
- `src/td_one/agent/deterministic_grouper.py` — KMeans (fixed seed) +
  tree-surrogate rules, all passing `validate_rule()`. Same input →
  identical groups (tested).
- LLM keeps the explanation role via citation-enforced `risk_reporter.py`.
- Results: 10 reproducible groups (2–4 conditions each), **10/10 citation
  pass** with a total citation contract; grouping costs CRPS vs global
  (+0.009–0.016 as published — **corrected in exp78:** most of that
  figure was the point-mass bug × window-mean scoring; on exp66c's own
  metric the cost is +0.0033 with the fixed forecaster, seed 42) and
  vs hand-written (+0.006–0.010), beats random by ~40%.
- **Honest negative inside the win:** static-attribute risk bins do not
  calibrate — LOW-bin groups realized *higher* test attrition (0.128)
  than HIGH-bin (0.087). Do not ship static risk bins as predictions.
- See `experiments/exp66c_report.md`.

### exp66d: cohort recovery + LLM risk classifier (exp66c follow-up)

- `src/td_one/data_gen/cohort_labels.py` — pure-function recomputation
  of the simulator's planted cohorts (`low_rate`, `osfi_exempt`,
  `family_squeeze`, `control`).
- `src/td_one/evaluation/cohort_recovery.py` — ARI + per-cohort
  precision/recall of discovered groups vs planted cohorts.
- `src/td_one/agent/llm_risk_classifier.py` — LLM as at-risk classifier
  (HIGH/MEDIUM/LOW + justification), given group stats + PIT-safe news.
- **Recovery honest negative:** ARI 0.01 — the grouper finds real,
  reproducible segments (rate_type ARI 0.24) but not the DGP's causal
  joints. Explanations are true descriptions of the wrong segments for
  forecasting; still useful as segment narratives.
- **LLM classifier directionally better:** monotonic HIGH > MEDIUM > LOW
  on test vs inverted bins — but n=10 groups, info-asymmetric (LLM had
  news). Worth a controlled follow-up before stronger claims.
- **Showcase:** `docs/GROUPING_EXAMPLES.md` — 4 worked examples
  (rule → members → forecast → risk → explanation), including the
  failure case (g3: both reads LOW, test 0.167).
- See `experiments/exp66d_report.md`.

### exp66e: uplift + SHAP tools for cohort recovery (user-directed)

The user's hypothesis: unsupervised clustering failed because it
clusters on feature similarity, not attrition relevance. Tools that
measure *what drives attrition heterogeneity* should recover the
cohorts.

- `src/td_one/evaluation/shap_analysis.py` — GBM attrition model +
  exact TreeSHAP values/interactions, top-interaction discovery,
  planted-pair recall, SHAP-space clustering, per-group driver
  summaries. (`shap>=0.45.0` declared in `pyproject.toml`; uv.lock
  regen pending.)
- `src/td_one/evaluation/uplift.py` — T-learner uplift (variable vs
  fixed rate), decile segmentation. Descriptive heterogeneity
  segmentation, not causal identification (treatment is self-selected).
- **Phase 1 (exp66e):** SHAP clustering doubles ARI (0.091 vs 0.046
  raw; `low_rate` F1 0.55) but stays weak; interaction recall@10 =
  0.000 (top interactions are `term_months` pairs, not planted pairs);
  uplift deciles = 0.002 (chance).
- **Phase 2 (exp66e2):** `llm_grouper.py` gained `tool_context`;
  LLM+tools ARI 0.025 vs LLM-without 0.026 — **honest negative**, tools
  don't fix LLM group discovery. Deterministic grouping pivot stands.
- **exp66f:** controlled classifier ablation (3 seeds) — the LLM-vs-bins
  advantage is +0.154 with CI covering zero; news effect +0.031
  [−0.45, +0.50]; **exp66d's classifier result is downgraded to "no
  measurable advantage"**. Risk tools (`agent/risk_tools.py`: SHAP
  drivers, counterfactual sensitivity, news-risk alignment, uplift
  amenability): only counterfactuals showed a directional signal
  (+0.348, 1 seed) — **superseded by exp66g below**; news alignment
  hurts (−0.174).
- Verdict: SHAP analysis joins the analyst kit (explanation tool);
  uplift stays on the shelf; LLM grouper gets no tool context.
- **exp66g (2026-10-20):** multi-seed (5) confirmation of the
  counterfactual tool as classifier context — **NOT confirmed**:
  B vs A delta −0.0018, CI covers zero; a shuffled-flips control
  arm scores identically (0.0651), proving the content inert. The
  original 1-seed result was seed noise. Five iteration rounds
  (prompt tuning, format, param search, guidance, control)
  converge on inert. Counterfactuals remain a valid analyst
  *explanation* device; dead as classifier prompt context. One
  genuine positive: **LLM + stats (no tool context) is the
  classifier of record** — beats deterministic bins pooled over 5
  seeds (+0.0198 bin error, CI excludes zero).
- **exp78 (2026-10-20):** fixed the structural defect exp72 flagged —
  `v2_forecaster_fn` constructed `ProposedForecasterV2` but never fitted
  it, so `predict` always raised and the blanket `except` returned a
  point mass. Now: naive center + PIT-safe Gaussian spread from trailing
  naive one-step errors (`spread_method="gaussian"` default,
  `quantile_spread=False` reproduces old output). Validation on
  SimulatorV2 (3 seeds): fixed beats point-mass on control (−0.0287
  CRPS, p=4.2e-45, cov80 0.075→0.714) and S1@4σ (−0.0326, p=1.5e-36);
  Gaussian beat the empirical spread ablation in both scopes. The
  exp66c +0.009–0.016 grouping cost was largely a point-mass ×
  window-mean-scoring artifact: on exp66c's own metric it is +0.003
  with the fix (seed 42); grouping = explainability verdict stands.
  Layer-2 smoke clean (traces emit). Default switched; exp75/exp72
  reproduction needs `quantile_spread=False`.
- **exp79 (2026-10-20):** settled the two questions exp78 left open.
  (1) **Center: trailing-6-mean ships as the production default**
  (`center="trailing_mean"`; exp78's naive center stays behind the
  flag). On SimulatorV2 it beats naive on control (−0.0072 CRPS,
  p=0.034 pooled; fresh-seed confirmation −0.0171, p=1.6e-05) and
  under S1@4σ (−0.0243, p=4.7e-07) with *better* shock coverage
  (cov80 0.713 vs 0.602). exp66c grouping cost drops again:
  +0.0033 → +0.0018. (2) **The real fitted v2 router does NOT belong
  inside the group forecaster**: `router_forecaster_fn` (per-series
  rolling-origin fit, experts naive/ETS/ARIMA) is significantly
  *worse* than the naive center on control (+0.0080, p=0.012) and
  loses to the trailing center everywhere (p≤6e-07) — a per-series
  self-fit has ≤13 backtest records vs the hundreds the router was
  designed for. It ships as an available option, not the default,
  with regression tests proving fit() runs and failures raise (the
  exp78 silent-fallback bug class cannot be reintroduced silently).

## Direction

Two tracks, eventually combined:

- **Track 1 (exp66):** LLM proposes borrower groups from metadata (+ news),
  group-level attrition forecasting, LLM risk reports with grounded
  explanations. Forecasting is part of the agent's toolkit.
- **Track 2 (exp67/68):** Red-team Chronos-2/TimesFM-2.5 on loan-domain
  distributional shifts; build an agentic guard; evaluate on held-out shifts.
- **Real data (exp70/71/71b):** OSFI M4 runoff benchmark (exp70) and
  competitor spread covariate (exp71 pilot, exp71b at full coverage:
  BoC posted backbone 622m + Wayback RateHub broker bests 54m) —
  all honest negatives (71b settled: the in-sample spread relation is a
  regime artifact).
- **Layer-2 (exp73):** meta-router over v2 on OSFI runoff — validation
  gating ties naive, repairs v2's harms (partial positive, kept as the
  do-no-harm wrapper); lock-in correction adds nothing (dead). See
  `docs/REAL_DATA_INTEGRATION.md`.
- **Integration (exp72):** Track 2's shift scenarios stress-test Track 1's
  group forecaster. S3 (regional divergence) is the meeting point.
- **Orchestrator (exp75):** mini-agent + orchestrator decision engine over
  all 6 beds (OSFI, Food CPI h=1/h=6, SMM, shifts, SimulatorV2). Routing
  (B) ≡ naive on every bed (agents inert per ablation); LLM orchestrator
  (C) dominated by deterministic shift (E: Food h=1 −22.8%, p=1.9e-14).
  Surviving modules: E (opt-in directional shift, gated on validation
  sign accuracy), D (uncertainty rescue for miscalibrated bases).

## Track 1 — LLM Grouping + Group-Level Forecasting (exp66)

### Data
- **Simulator** (borrower-level, rich metadata): province, age, household
  size, children, employment, income (lognormal), liquid deposits, debt
  payments, rate type, mortgage rate, term, origination date, property
  value, insured flag, occupancy, credit score band, utilization, channel,
  advisor flag, life events, prior refinances (burnout).
- **SMM: NOT usable** — aggregates only (63 rows). Track 1 is synthetic-only.
- **News:** 17,738 articles, PIT-safe accessors. Grouper reads news because
  some news is demographic-specific (e.g., "Ontario first-time buyers").

### Pipeline
1. **`llm_grouper.py`** — LLM sees: attribute distributions (histograms, not
   raw rows), training-period attrition rates by attribute, demographic-
   tagged news summaries (via RAG, see below). Outputs strict JSON:
   executable `pandas.DataFrame.query()` rules + one-sentence rationales.
   Validator rejects bad queries, outcome-column references, tiny groups,
   excessive overlap. **PIT rule:** zero post-origin info in the prompt.
2. **`group_forecaster.py`** — designed as per-group
   `ProposedForecasterV2`; as built, the v2 call was never fitted, so
   every forecast was the point-mass naive fallback (exp78). Since
   exp78: naive center + Gaussian spread. Groups under `min_group_n`
   fall back to global (shrinkage).
3. **`risk_reporter.py`** — deterministic risk bins (LOW/MEDIUM/HIGH from
   training tertiles, widened by conformal intervals); LLM explanations
   with **citation enforcement** — every factual sentence cites a `[stat]`
   or `[url]`, post-hoc checker rejects uncited claims.

### News wiring (added 2026-10-19)
- **Live Perplexity API** for demographic-specific news fetch (user approved
  credit increase). **Cache everything** — every fetch enriches the SQLite
  cache; never re-fetch.
- **RAG upgrade:** current RAG is BM25 → Groq rerank. Upgrade to
  **semantic retrieval + rerank** using local embeddings
  (all-MiniLM-L6-v2, 384-dim — Vector proxy returns 403
  model_access_denied for embeddings, verified 2026-10-19).
  Pipeline: embed query → cosine retrieval over cached articles →
  LLM rerank → top-k to grouper/reporter.
- **Top-k tuning** (2026-10-19, `scripts/tune_rag_k.py`, 20 hand-labeled
  pairs): k=10 maximizes F1 (0.468), but retrieval uses k=50 biasing
  toward recall (0.818) since the reranker filters cheaply; final k=5
  for precision. See `src/td_one/rag/semantic.py` docstring.
- News is PIT-gated via `articles_visible_asof(origin)` everywhere.
- **Article bodies (2026-10-19):** the news DB had titles only — no bodies
  to chunk. Built `scripts/fetch_article_bodies.py` (trafilatura extraction,
  per-host throttle, `article_bodies` table keyed by URL). ~86% fetch success
  on the 500-article validation sample.
- **Chunking ablation** (`src/td_one/rag/chunking.py`,
  `scripts/tune_chunking_v2.py`): title-as-query MRR on 30 bodies —
  fixed-512 (MRR 0.731, 3.5 chunks/article) ≈ paragraph (MRR 0.715).
  **Fixed wins** (simpler, deterministic). Sentence chunking too fragmented
  (13/article); semantic chunking too slow on CPU. Chunk embeddings cached
  to disk keyed by chunk-text hash — never recomputed.
- **Full-vs-chunk ablation** (`scripts/ablate_fast.py`): chunk-only context
  matches full-article keyword coverage (0.818 vs 0.824) at **1.9x fewer
  characters**. **Verdict: pass chunks, not full articles**, to the LLM.
- **Threading fix:** torch/sentence-transformers deadlocks without
  `OMP_NUM_THREADS=1` etc. — set in `EmbeddingClient` and all RAG scripts.

### Evaluation (4-arm ablation)
| Arm | Description |
|---|---|
| A: Global | No grouping |
| B: Random | Same group count, random assignment |
| C: Hand-written | Planted simulator cohorts |
| D: LLM-discovered | This pipeline |

**No kill criterion** (reframed 2026-10-19): grouping is for explainability.
Report CRPS cost vs explainability gain honestly. If the CRPS cost is large,
say so — but don't kill grouping for losing on CRPS alone.

### Build order
1. `src/td_one/agent/grouping.py` — `GroupDefinition`, `validate_rule()`,
   `apply_groups()`, overlap resolution → `tests/test_grouping.py`
2. `src/td_one/rag/semantic.py` — embedding client (Vector proxy),
   semantic retriever, reranker → `tests/test_semantic_rag.py`
3. `src/td_one/agent/llm_grouper.py` — prompt builder (+ news summaries),
   JSON parsing, validation loop → `tests/test_llm_grouper.py` (mock)
4. `src/td_one/agent/group_forecaster.py` — per-group v2 (in practice
   the point-mass naive fallback until the exp78 fix; see the Track-1
   exp78 entry), fallback → `tests/test_group_forecaster.py`
5. `src/td_one/agent/risk_reporter.py` — bins, news retrieval, citation
   checker → `tests/test_risk_reporter.py`
6. `experiments/exp66_grouped_attrition.py` + `exp66_report.md`
7. News fetcher: `scripts/fetch_demographic_news.py` (Perplexity, cached)

Closes Argon Finding 3 (`discover_candidate_groups()`).

## Track 2 — TSFM Red-Teaming (exp67) + Agentic Guard (exp68)

### Attack surface
Chronos-2 and TimesFM-2.5 are strictly univariate: forecast = f(history).
Shifts invisible in the history window but visible in side info (policy
dates, rate levels, borrower mix) are structural blind spots.

### Shift taxonomy (loan domain)
| # | Shift | Why TSFM fails |
|---|---|---|
| S1 | Rate-shock regime | History is pre-shock |
| S2 | Policy intervention (e.g., stress-test change) | Jump predictable from announcement, invisible univariately |
| S3 | Regional divergence | Aggregate averages the move away |
| S4 | Seasonal break | Learned old seasonal pattern |
| S5 | Correlation breakdown (lock-in vs refi) | No covariate access |
| S6 | Volatility regime | Pre-shift spreads too narrow |
| S7 | Composition drift (burnout) | Drift misread as trend |

### Protocol
- Rolling origins across shift onset T (pre / straddling / post).
- Metrics: CRPS on common 19-level grid, 80% coverage, PIT histograms.
- Paired t-test + win rate vs model's own pre-shift baseline AND vs
  classical baselines on identical shifted data.
- Placebo control on no-shift DGP. Severity ladder (0.5σ–4σ) → breaking
  point curves, not binary pass/fail.

### Agentic guard (exp68)
1. Shift detection from covariates (CUSUM on policy rates,
   `events_visible_asof`) — fires before the target moves.
2. Context surgery: truncate TSFM input to post-shift data / switch experts.
3. Shift-aware quantile widening from past shift episodes' residuals.
- LLM classifies shift type from metadata/news → selects response template.
  (LLM outputs structure, never numbers.)
- **Holdout:** dev on S1/S4/S6; eval on S2/S3/S5/S7. Disjoint seeds,
  different magnitude ranges, pre-registered.

### Build order
1. `src/td_one/data_gen/shift_scenarios.py` — `ShiftSpec`,
   `apply_shift()`, seasonal-break + coefficient-schedule hooks
2. `src/td_one/data_gen/group_series.py` — histories → monthly per-group
   series (**shared with Track 1**)
3. `experiments/exp67_tsfm_redteam.py` + report
4. `src/td_one/agent/shift_guard.py` — detector + policies
5. `experiments/exp68_agent_vs_shift.py` + report (pre-registered holdout)

**Order is load-bearing:** red-team numbers before the guard. No guard
without failure evidence.

### exp67 outcome (2026-10-20) — Phase A COMPLETE, Phase B unblocked

- Grid finished: 7 shifts × [1σ, 4σ] × 2 seeds × 9 origins; Chronos-2 /
  TimesFM-2.5 (real weights) vs naive / M0; placebo-controlled; PIT gate
  passes (pre-shift degradation +0% in all 28 cells).
- All models break together at 4σ on S1 (+185–228%), S2 (+65–79%),
  S3 (+30–59%) — TSFMs have no *relative* robustness edge, but Chronos-2 is
  best in *absolute* CRPS on 6/8 shifts. S4/S5 never break (honest negatives).
  Coverage collapses under level shifts (M0 0.61→0.08 at S1/4σ).
### exp68 outcome (2026-10-20) — Phase B COMPLETE: honest negative

- Guard (`src/td_one/agent/shift_guard.py`): CUSUM on policy-rate covariate
  (2-month lead) + PIT-filtered announcements → context truncation +
  conformal widening (dev-calibrated). 18/18 guard tests green.
- **Claim FAILED:** guard+TSFM (Chronos-2) vs TSFM-alone on holdout
  (S2/S3/S5/S7, 2–4σ, seeds 100–101): +0.5% rel., p=0.824, win rate 14.6%.
  Do-no-harm gates pass.
- **Lesson:** context truncation hurts — Chronos-2 adapts better with full
  (contaminated) context than with 9 post-shift points; correct detection
  (S2: 12/12) gained nothing. Per pre-registration: no iteration.
- Full report: `experiments/exp68_report.md`.

### exp69 outcome (2026-10-20) — anticipatory LLM adjustment: honest negative

- Adjuster (`src/td_one/agent/anticipatory_adjuster.py`): LLM reads PIT-safe
  news → structure-only signal (type/direction/bucket/timing); calibrated
  (type, bucket) → level-shift from past episodes (seeds 200–202) applied to
  Chronos-2 quantiles on straddle origins. 21/21 tests green (incl. exp69b driving_trends).
- **Claim FAILED:** B vs A on straddle origins (S1/S2, 2–4σ, seeds 300–302):
  +10.2% rel., p=0.231, win rate 36%. **Oracle ceiling +11.4% (p=0.214)** —
  even perfect anticipation doesn't significantly help; the calibrated median
  shift is blunt vs per-episode noise. LLM ≈ oracle (B vs C, p=0.73).
- Works only at S1/4σ (B +30%, C +35%); hurts at 2σ and on S2. Do-no-harm
  gates pass exactly (no hallucination, no post-onset firing).
- Per pre-registration: no iteration. Full report: `experiments/exp69_report.md`.
- **Variant (exp69b):** broad macro synthesis (semantic RAG over 12+ topics,
  `driving_trends` citations) vs rate-news-only — synthesis *hurts* (D vs B′
  −5.0% rel., win 5.6%): corroborating context dilutes timing precision.
  The unannounced-shock case (converging signals only) is untested, not
  refuted. Full report: `experiments/exp69b_report.md`.
- **Variant (exp69c):** real-data anticipatory adjustment — the user's
  counter-hypothesis that richer real news rescues the idea. Multi-call
  story pipeline (`src/td_one/agent/story_pipeline.py`: retrieve → rank →
  story → classify) with timeline anchors, on real episodes (2022 first
  hike, 2022 100bp super-hike on Food CPI; 2024 first cut on SMM),
  Chronos-2 baseline, calibration from past episodes only (strict PIT,
  absolute |Δ|). Pre-registered kill; per the researcher-mindset
  instruction the kill triggered diagnosis + variants (V_abstain,
  V_timing, V_size, V_timing+size from stored artifacts).
- **exp69c FAILED too:** B vs A +23.7% (p=0.231, win 0%); B vs C −10.5%
  n.s. Failure anatomy: (1) strict-PIT calibration starved 5/12 rows
  (empty tables — T1's past wasn't observable yet); (2) the classifier
  missed the June 2024 cut at 3/4 T3 origins (ranker favored "hold"
  headlines over the one "cut possible" pivot); (3) where it fired,
  direction was right 2/3 but size overshot 2.6× and timing ran +1.5mo
  early; (4) timeline anchors changed 5/12 classifications but 0
  forecasts — unresolved, not refuted. Even with both diagnosed errors
  corrected (V_timing+size, partly post-hoc), the overlay is −4.3%
  (p=0.53) — not a rescue. The news-reading is good; the one-episode
  median size × fragile timing is too blunt an instrument. Standing
  result: no tested intervention beats Chronos-2 alone on shifted
  series — synthetic or real, reactive or anticipatory, single-call or
  scaffolded. Full report: `experiments/exp69c_report.md`.

## Integration (exp72)

Track 2's shift scenarios stress-test Track 1's group forecaster.
S3 (regional divergence) is the natural meeting point: Track 1 groups the
entities, Track 2 shifts the groups.

**Protocol** (pre-registered in `experiments/exp72_report.md`, placebo-
controlled like exp67): MortgageSimulatorV2 portfolio → deterministic
groups fitted on pre-onset metadata only → per-group group forecaster vs
global (portfolio) forecaster vs Layer-2 meta-router, rolling origins
(pre/straddle/post), CRPS diff (shifted − control, same seed/origin/group),
paired tests, member-weighted. Shifts: S1 (rate shock), S3 (regional
divergence via group-level S3 helper `apply_divergence`, regions = median
split on training ON-share), S7 (composition drift via β-schedule and
S7mix = post-onset episode migration donor→recipient), at 1σ and 4σ.

**Questions.** (Q1) Do group-level forecasts degrade less than the global
forecast under S3 (groups isolate the diverging region)? (Q2) Does the
`min_group_n` fallback trigger appropriately as groups shrink/drift under
S7? (Q3) Does Layer-2 gating protect the group forecaster the way it
protected v2 on OSFI?

**Findings (5 iteration rounds; full log in the report).**
- **Q1 — not supported at the pre-registered bar.** 5 seeds:
  S3@4σ group−global degradation diff −0.0021 (p=0.22);
  directionally consistent in every round and both groupers, never
  significant on the headline cell (S1@4σ is significant, p=0.002 —
  a width/noise effect, not isolation). Oracle-pure province groups
  (−0.0053, p=0.15) match tree groups: the limit is the forecasting
  mechanism (trailing-mean has no way to foresee the divergence), not
  the grouping.
- **Q2 — fallback is a working safety valve with a blunt trigger.**
  It helps on thin-but-alive groups (donor CRPS 0.121/0.086 vs
  no-fallback 0.143/0.120) but fires on size not drift, lags collapse
  (trailing-6 window), and is the *wrong* answer for truly empty
  groups (own mean ≈ 0 scores 0.0000 vs fallback 0.0751). Design fix
  noted, not shipped.
- **Q3 — supported in the thin-group regime.** Layer-2: LAYER2Q −
  GROUPQ = −0.021..−0.031 (p<1e-8, win 0.65–0.68) on control and all
  shifted worlds, straddle classes carrying the effect. Edge vanishes
  with 5 big province groups — the protection value lives where the
  base forecaster is most fragile.
- **Methodological flag:** Track-1's `v2_forecaster_fn` emits flat
  point-mass quantiles — CRPS 0.165 vs 0.13 for a proper Gaussian
  group forecaster on identical histories. Affects exp66b/c
  comparability on noisy per-month metrics; flagged here, then
  root-caused and **fixed in exp78** (predict-without-fit; the group
  forecaster now runs naive center + Gaussian spread by default).
- Honest negative on the James–Stein τ sweep (adds nothing).

## Standing constraints
- PIT safety is a gate (no `observed_asof > origin` anywhere).
- Every fetcher hit enriches the cache; commit cache snapshots.
- Perplexity: user-approved budget, resume-safe, cached.
- TDD + regression test per fix; ruff clean; docs with every change.
- Honest negatives with numbers, not softened.
