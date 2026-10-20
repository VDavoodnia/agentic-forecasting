# exp88 — Segment Scout, Scored: an LLM proposes segments from statistics and domain context

**Status:** DONE 2026-10-20. **Verdict: ✅ pre-registered bar PASSED.** With segment-level
statistics plus a generic policy brief and PIT news context (round R3), the LLM scout recovers
SimulatorV2's planted cohorts at **ARI 0.204 / 0.454 / 0.424** on seeds 7 / 42 / 123 (mean
**0.361**) against the bar of ARI > 0.091 on at least 2 of 3 seeds — roughly **4× the SHAP
clustering incumbent (0.091)**, with human-readable rules. The larger "flash" model on the same
information set (R5a) is the most consistent configuration (0.372 / 0.429 / 0.390, mean 0.397)
and on seed 7 reproduces the planted `low_rate` rule **verbatim**. Two honest negatives ride
along: SHAP analyst context *hurts* the scout (R2, mean 0.052), and handing the scout per-cell
training switching rates — the supposed ceiling — is *worse than the bar on 2 of 3 seeds*
(R5b, mean 0.077). The segments are found by the domain prior, not by outcome gradients.

Code: `src/td_one/agent/segment_scout.py` (11 tests, `tests/test_segment_scout.py`),
`experiments/exp88_segment_scout.py`. Raw results: `exp88_results.json`; prompts:
`exp88_prompts.json`; cached LLM replies: `exp88_cache/` (24 calls total).

## Question

exp66e/f established the cohort-recovery league table on SimulatorV2's planted cohorts
(`low_rate`, `osfi_exempt`, `family_squeeze`, `control`): SHAP-guided clustering ARI 0.091
(best), raw-feature KMeans 0.046, LLM grouper 0.025–0.026 (chance), deterministic tree leaves
0.010. The LLM grouper failed *blind* — it proposed rules from attribute summaries with no
domain model of what a mortgage-book segment is. exp88 asks the converse question: **can an
LLM scout that reads only point-in-time-safe feature statistics — plus, in later rounds,
public domain context — propose segment definitions (in the existing `GroupDefinition`
format) that recover the planted cohorts?** This is Future Direction 7 ("segment scout,
scored") from the final delivery.

## Protocol

- **World:** SimulatorV2, 4,000 borrowers × 36 months, seeds 7 / 42 / 123 (exp66d/e
  protocol). Scout input is the last pre-cutoff snapshot per borrower (cutoff = 9th-from-last
  outcome month; 2023-09-01 on all three seeds), with the same engineered columns the SHAP
  incumbent sees (`origination_year`, `n_children` via
  `evaluation/shap_analysis.engineer_features`). Incumbents (raw KMeans k=5, SHAP-cluster
  k=5) are recomputed per seed inside the run; seed 42 reproduces exp66e exactly
  (0.046 / 0.091), so the harness is faithful.
- **Scoring:** proposed groups are applied as a first-match partition (proposal order) plus a
  catch-all; ARI and best-match per-cohort precision/recall/F1 via
  `evaluation/cohort_recovery`, identical to exp66e/f.
- **Validation:** every proposed rule must pass `agent/grouping.py::validate_rule()` with
  `min_group_n=200`; rejections are logged with a categorized reason and fed back for up to
  3 attempts (feedback contains only syntactic/size reasons, never cohort information).
- **Leakage audit (pre-registered):** the scout frame builder *raises* if any outcome column
  is present; prompts are audited for cohort names and planted vocabulary
  (`audit_prompt`) and asserted clean for R1–R3 and R5a before any call. The planted
  parameters (thresholds, vintages, cohort names) appear nowhere in any prompt; the policy
  brief contains only generic public-domain facts about the Canadian mortgage market
  (OSFI B-20 stress test, 2020–21 record-low fixed rates, 2022–23 Bank of Canada hikes,
  childcare-cost pressure, strong-credit mobility). R5b is the single deliberate exception:
  it adds per-cell *training-window switching rates* (outcome aggregates) as a diagnostic
  ceiling probe; cohort names remain banned and asserted absent there too.
- **LLM:** Gemini via the Vector proxy, reasoning effort low, disk-cached, budgets enforced
  (200 flash-lite / 40 flash calls; 24 used in total including retries and R4 samples).

## Rounds (the iteration ladder)

| Round | Information set | Model |
|---|---|---|
| R1 | Feature statistics only: per-feature quantiles/shares + curated joint counts (rate type × vintage × rate band; insured × credit band; income band × children; province × rate type). **No outcome information at all.** | flash-lite |
| R2 | R1 + SHAP analyst context (exp66e's `build_tool_context`: top drivers, interactions, SHAP-segment profiles, uplift summary) | flash-lite |
| R3 | R1 + policy brief + PIT news summaries (semantic index, 4 queries, before cutoff) + macro-signal lines as of the origin | flash-lite |
| R4 | Self-consistency: 3 salted samples of the best R1–R3 information set (R3), merged at min_votes=2 | flash-lite |
| R5a | Capability diagnostic: best information set (R3) on the larger model | flash |
| R5b | Ceiling diagnostic: R1 statistics + per-joint-cell training switching rates (outcome aggregates; **not** a scout configuration) | flash-lite |

## Results — ARI by seed

| Method | seed 7 | seed 42 | seed 123 | mean |
|---|---|---|---|---|
| Raw KMeans (incumbent, recomputed) | 0.039 | 0.046 | 0.033 | 0.039 |
| SHAP clustering (incumbent, recomputed) | 0.070 | 0.091 | 0.089 | 0.083 |
| LLM grouper, exp66f (historical) | — | 0.025 | — | — |
| Deterministic tree leaves, exp66d (historical) | — | 0.010 | — | — |
| **R1 stats only** | **0.149** | 0.085 | **0.257** | 0.164 |
| R2 stats + SHAP context | 0.003 | 0.138 | 0.014 | 0.052 |
| **R3 stats + policy/news context** | **0.204** | **0.454** | **0.424** | **0.361** |
| R4 self-consistency merge of R3 | **0.365** | **0.188** | **0.326** | 0.293 |
| **R5a R3 info set, flash model** | **0.372** | **0.429** | **0.390** | **0.397** |
| R5b ceiling diagnostic (outcome rates) | 0.020 | 0.163 | 0.049 | 0.077 |

Bold = clears the 0.091 bar on that seed. **R3, R4 and R5a clear the bar on all three
seeds; R1 clears it on 2 of 3** (seed 42: 0.085 vs the seed's own SHAP 0.091 — a tie in
substance). The pre-registered bar is met by the R3 configuration.

## Results — per-cohort F1 (mean over 3 seeds, best-match)

| Cohort | SHAP incumbent | R3 | R5a | Reading |
|---|---|---|---|---|
| low_rate | 0.55 | **0.82** | **0.95** | Found. The vintage conjunction is recoverable once the policy brief names the 2020–21 low-rate era; flash pins the exact rate cap. |
| control | 0.53 | **0.69** | 0.61 | Found as the complement of the named segments plus pressure groups. |
| family_squeeze | 0.15 | **0.47** | **0.47** | Partially found: income threshold exact (75k), children threshold proposed at ≥2 vs planted ≥3 (seed 123 R3 used ≥3 but paired it with income <100k). |
| osfi_exempt | 0.19 | **0.49** | 0.43 | Partially found: the exact rule was proposed on seed 42 (F1 0.76); on other seeds the segment is diluted by added clauses (investments, channel) or absorbed into the catch-all. |

## Example proposed rules (verbatim)

R3, seed 42 (ARI 0.454) — all four planted cohorts addressed by four rules:

- `low_rate_lockin_high_impact`: `rate_type == 'fixed' and origination_year >= 2020 and origination_year <= 2021`
- `vulnerable_high_debt_burden`: `n_children >= 2 and household_income_cad < 75000`
- `equity_rich_uninsured`: `insured_flag == False and credit_score_band == '720_plus'` ← the planted `osfi_exempt` rule, exactly
- `life_event_impacted`: `life_event_6m == True and life_event_type != 'none'`

R5a, seed 7 (ARI 0.372) — the planted `low_rate` rule recovered verbatim:

- `ultra_low_fixed_rate_2021_lock_in`: `rate_type == 'fixed' and origination_year == 2021 and mortgage_rate_pct <= 3.0`

R1, seed 42 (ARI 0.085, statistics only) — plausible pressure segments, wrong handles:
`rate_type == 'variable' and mortgage_rate_pct > 5.0`; `td_tenure_months > 80 and
mortgage_rate_pct < 3.0` (low-rate borrowers found via tenure instead of vintage).

R5b, seed 42 (ARI 0.163, outcome rates) — chases switching gradients instead:
`rate_type == 'variable' and credit_utilization_pct > 45.0 and mortgage_rate_pct > 5.0`.

## Validator and call log

Across all rounds and seeds: **0 parse failures**, **28 rejected proposals** (27
`too_small` — below the 200-borrower floor; 1 `empty`), **0** outcome-column, unknown-column
or missing-operator rejections, **0** leakage-audit violations. Every scored partition is
built exclusively from rules that passed `validate_rule()`. 24 LLM calls total (cached).

## Iteration log — what each round taught

1. **R1 (stats only) already beats the old LLM grouper ~6×** (mean 0.164 vs 0.026) and beats
   SHAP clustering on 2 of 3 seeds. Two design differences from exp66f explain most of it:
   the scout sees *joint* feature counts (vintage × rate type × rate band), not just
   per-feature summaries; and rejection feedback with retries keeps every rule valid and
   ≥200 borrowers. But R1's rules grab proxies (tenure for vintage) — statistics alone show
   *that* a low-rate block exists, not *which* borrowers define it.
2. **R2 (SHAP context) hurts — honest negative.** Mean 0.052, at chance on 2 of 3 seeds.
   The SHAP context describes what drives the *outcome model* (dominated by term/payment
   structure), and the scout anchors on those drivers instead of on segment structure.
   This replicates exp66f's finding from the other side: analyst tool context helps
   analysts, not groupers — here it actively misleads the proposer.
3. **R3 (policy/news context) is the unlock.** Mean 0.361, bar cleared on all seeds. The
   policy brief gives the scout *names for the blocks the joint statistics show*: the
   2020–21 vintage becomes "borrowers who locked record-low fixed rates", the
   uninsured × 720_plus cell becomes "stress-test-exempt prime borrowers". The macro-signal
   component was thin at the simulated origins (only the `inflation` category exists at or
   before 2023-09 in the signal store), so the active ingredients are most plausibly the
   brief plus the news summaries; isolating brief-vs-news is a follow-up, not run here.
4. **R4 (self-consistency) is seed-dependent — merge semantics are the problem.** The three
   R3 samples individually score 0.22–0.42; the min_votes=2 union merge *helps* seed 7
   (0.365, collapsing to 2 broad groups) but *fragments* seeds 42/123 into 17–19 groups
   (ARI 0.188 / 0.326, still above bar but below the single R3 draw). Majority merging over
   rule *sets* does not commute with first-match partition scoring; a merge that votes on
   borrower-level assignments instead of rules is the fix to try next.
5. **R5a (capability) helps consistency, not kind.** Flash on the same prompts: mean 0.397,
   tightest range (0.372–0.429), and the exact planted low_rate conjunction on all three
   seeds (rate cap `<= 3.0` or `< 3.0`). The information set is the bottleneck in R3; the
   bigger model mainly sharpens thresholds.
6. **R5b (the "ceiling" that wasn't) — the most informative negative.** Per-cell training
   switching rates are the strongest outcome information legally available, yet the scout
   does *worse* than with statistics alone (0.077 vs 0.164). At a 2.7% base switching rate
   the cell rates are noisy and dominated by outcome-model drivers (variable-rate pressure,
   credit utilization) that cut across the planted cohorts — the cohorts are *feature-defined
   latent groups*, and their signature in outcome space is weak by construction. Implication
   beyond this experiment: segment discovery should be driven by domain structure and
   feature geometry, with outcome rates used to *validate* segments afterwards, not to
   propose them.

## Diagnosis — which cohorts are visible from which information set

- **low_rate** — visible in joint feature statistics (a dense fixed × 2021 × ≤3% block);
  *named* correctly only with the policy brief; thresholds *sharpened* by the larger model.
  Invisible to outcome rates (its members' switching is suppressed, not elevated).
- **osfi_exempt** — a pure feature conjunction with no distinctive marginal signature (its
  members look like control on every single feature); recoverable only when the brief
  supplies the B-20 exemption logic. Exact rule proposed once (seed 42 R3); extra
  "colour" clauses (investments, channel, balance) dilute it on other seeds.
- **family_squeeze** — the income edge (75k) is found reliably; the children edge is
  systematically softened (≥2 proposed vs ≥3 planted) because the joint income × children
  table bands children as 3+ only in aggregate counts the scout weighs against group size —
  the ≥3 cell at income <75k is ~286 borrowers, close to the 200 floor the prompt warns
  about, and the scout trades recall for size safety.
- **control** — recovered as the complement; scores track how many of the other three are
  carved out cleanly.

## Caveats

- The planted cohorts were designed to mimic real Canadian mortgage-market structure, and
  the policy brief describes that same real market. The result therefore reads: *when the
  domain prior matches the segment-generating logic, an LLM scout recovers latent segments
  far better than outcome-driven clustering.* It does not show the scout can recover
  segments whose logic is absent from public domain knowledge.
- R3 bundles brief + news summaries + macro lines; the brief is the suspected active
  ingredient (macro coverage at the origins is one category). A brief-only vs news-only
  ablation is the obvious follow-up.
- ARI is computed against simulator ground truth; on real data there is no planted truth,
  so the transfer path is the exp66c contract: scout proposals as an *explainability and
  hypothesis* layer, validated downstream by the group forecaster's CRPS and calibration,
  with the global model remaining forecaster of record.
- Single prompt design, one model family (Gemini flash-lite / flash via the Vector proxy);
  R4 shows non-trivial sampling variance (individual R3 samples 0.22–0.42).

## Verdict and follow-ups

**The segment scout works — with domain context.** Recommended configuration: R3
information set (feature statistics + policy brief + PIT news summaries), flash-lite for
proposals (flash when threshold precision matters), `validate_rule()` gating with
rejection-feedback retries, min_group_n=200. Follow-ups, in order: (1) brief-only vs
news-only ablation to isolate the active ingredient; (2) borrower-level vote merging for
self-consistency (fix R4's fragmentation); (3) a size-aware children-band joint table to
test the family_squeeze ≥3 recovery; (4) run the scout as the proposal front-end for the
deterministic grouper on the next SimulatorV2 study and score downstream group-forecaster
CRPS, closing the loop with exp66c/exp79.
