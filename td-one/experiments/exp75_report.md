# exp75: Mini-Agent + Orchestrator Decision Engine — Report

**Status: PRE-REGISTERED (design frozen; results pending).**

## Architecture

The user's design: instead of one agent reasoning over all factors,
specialist **mini-agents** each own ONE factor and speak a narrow
vocabulary `{direction: up/down/flat, confidence: 0..1, reason: one
sentence}`. A final **orchestrator** consumes the mini-agent signal
vector (+ the base forecast) and decides. The orchestrator is a
*decision engine, not a forecaster*.

**Roster (per the roster-refinement brief: granular, 1:1 with the Macro
Signal Service category list):**

- **Factor agents (24)**: the 22 `macro_signals` categories
  (`immigration_law`, `food_prices`, `basic_needs`, `income_tax`,
  `minimum_wage`, `salary_region`, `school_childcare`, `services_costs`,
  `inflation`, `elections_politics`, `wars_geopolitics`,
  `pandemics_health`, `flight_prices`, `car_prices`, `global_trade`,
  `ai_startups`, `markets_sp500`, `central_bank_rates`, `housing`,
  `labour`, `fx`, `small_business`) plus `rent` and `climate`
  (keyword-heuristic RAG supplements, absent from the service — labelled
  heuristic in the code and report). Each reads its category's monthly
  signal (PIT: signal month < origin month) → target-relative pressure.
  **No cross-category reasoning in factor agents** — synthesis is the
  orchestrator's job exclusively.
- **Diagnostics family (4)**: `stability` (vol/break/CUSUM →
  uncertainty), `disagreement` (expert median CV → uncertainty),
  `seasonality` (seasonal position/strength → pressure),
  `group` (peer divergence → pressure). Speak about the series/model,
  not the world. Built on `layer2_tools`.

Factor signals are **month-stamped and cached** (`FactorCache`): all
origins in one calendar month share a bundle; the marginal cost per
origin is only the orchestrator LLM call (arm C).

**Families** (grouped voting / ablation): prices (food_prices,
basic_needs, inflation, car_prices, flight_prices, services_costs);
housing (housing, rent, school_childcare); labour (labour,
minimum_wage, salary_region, immigration_law); policy
(central_bank_rates, income_tax, elections_politics, small_business);
external (wars_geopolitics, pandemics_health, global_trade, fx,
markets_sp500, ai_startups, climate).

**Pre-registered pressure-sign priors** live in
`agent/factor_agents.py` (`PRESSURE_ATTRITION`, `PRESSURE_PRICE`);
direction semantics follow the signal service rubric (e.g.
`central_bank_rates` +2 = hawkish; `labour` +2 = unemployment rising;
`fx` +2 = CAD strengthening).

## Pre-registration

**Arms** (per bed):
- A: base alone (v2 on OSFI/Food CPI/SMM; Chronos-2 on shifts;
  group forecaster on the simulator bed).
- B: base + **deterministic orchestrator** (FORECAST mode).
- C: base + **LLM orchestrator** (FORECAST mode, flash-lite, cached).
- D: base + **deterministic orchestrator** (UNCERTAINTY mode —
  adaptive interval widths).

**Beds:**

> **Data-provenance note (2026-10-20):** `benchmarks/food_cpi/data.csv`
> (2010-01→2026-09) is **FRED CPIUFDSL** — the US CPI-U for Food, monthly,
> seasonally adjusted — confirmed by the original `exp35_baseline.py` docstring
> and anchor values 2026-08=350.309 / 2026-09=350.477. The file vanished from
> the tree on 2026-10-20 and was rebuilt 1:1 from FRED's CSV, including the
> genuine 2025-10 gap (US government shutdown, no BLS release — this is the
> "no 2025-10" quirk documented in the leakage audit). Earlier reports that
> call it "Canada Food CPI" are mislabeled; the series still serves as the
> smooth-macro test bed.

1. **OSFI runoff**: 5 segments × 60 origins, h=1 (exact exp70/exp73
   protocol; A = v2).
2. **Food CPI (CPIUFDSL)**: h=1 (59 origins) and h=6 (A = v2 via blocked CV).
3. **SMM**: 50 origins, h=1 (A = v2).
4. **Shifts S1/S2/S3**: Chronos-2 real, 2 severities × 2 seeds × 9
   origins, h=6 (exp67 protocol; A = Chronos-2 native; D scores
   coverage/width on straddle origins).
5. **SimulatorV2 + group forecaster** (ties into exp72; S3-style
   divergence on one group).

**Success criteria (per bed, pre-registered):**
- B/C: beat A at p<0.05 (paired, pooled) AND no segment harm
  (Δ > +2% at p<0.10 vs A on any segment).
- D: interval coverage moves toward 0.80 without mean width >1.5×
  native; CRPS of the widened distribution not significantly worse
  than A.
- Deterministic vs LLM: C must beat B's CRPS at p<0.05 on the
  pooled prioritized beds (OSFI, Food CPI h=1, S3) to justify the
  LLM orchestrator's cost.

**Iteration budget (standing rule: 4–5 diagnosis-driven rounds before
any kill):** Round 1 = this design. Round 2 = per-family ablation →
drop dead families / note. Round 3 = param search (tau thresholds,
family weights via validation ridge, α/β grids). Round 4 =
hypothesis-driven variants from Rounds 2–3 diagnoses (e.g. factor
momentum instead of level; prompt tuning for arm C). Round 5 =
confirmation. Each round logged below with diagnosis → change →
result → lesson. The LLM arm (C) gets 2–3 prompt-tuning rounds max
(cost).

**Known limitations (pre-registered):** factor signals exist only for
2025-11 → 2026-09, so on real beds only recent origins get active
factor agents (reported per bed); on synthetic beds (shifts,
simulator) factor agents degrade to unavailable and the test
evaluates the *diagnostics family* — S3 is a diagnostics test
(Stability/Disagreement/GroupAgent on divergence), not a factor
test. Food CPI (CPIUFDSL) ends 2026-09, so its last ~10 origins have
full factor overlap (signal month < origin month).

## Round log

### Round 1 — design + five beds, deterministic arms (2026-10-20)

Design described above. All beds: arms A (base), B (deterministic
orchestrator, FORECAST), D (deterministic orchestrator, UNCERTAINTY).
LLM arm C runs after Round-1 stabilizes.

| Bed | A CRPS | B CRPS | B vs A | D CRPS | D vs A | D cov→ |
|---|---|---|---|---|---|---|
| OSFI (5u×48) | 0.003998 | 0.003855 | **−3.6%**, p=0.086, win 0.52 | 0.004055 | +1.4%, **p=0.011** | 0.796→0.833 |
| Food h=1 | 0.6354 | 0.6354 | +0.0%, p=0.80 | 0.6412 | +0.9%, p=0.55 | 0.723→0.851 |
| Food h=6 | 3.2064 | 2.2831 | **−28.8%**, p=0.0065, win 0.42 | 2.9257 | **−8.8%**, p=4e-20 | 0.174→0.512 |
| SMM | 0.000594 | 0.000591 | −0.5%, p=0.51, win 0.39 | 0.000587 | −1.3%, n.s. | 0.105→0.105+ |
| Shifts (12u×6) | 0.04760 | 0.04694 | −1.4%, p=0.40, win 0.60 | 0.04762 | +0.0%, n.s. | 0.829→0.829 |
| Simulator | — | — | running | — | — | running |

**Round-1 diagnosis (drives Round 2):**
1. **B's gain is abstention, not synthesis.** OSFI decision mix (240
   origins): 140 stability-veto abstains, 78 quiet-world abstains, 22
   weak-pressure blends, **0 strong corroborated v2 picks**. Against
   the naive-alone reference computed on the same origins, B ≡ naive
   on every real bed: OSFI B 0.003855 vs naive 0.003851; Food h=1
   0.6354 vs 0.6354; Food h=6 2.2831 vs 2.2831; SMM 0.000591 vs
   0.000591; shifts 0.04694 vs 0.04694. Wherever B "beats" A, a pure
   naive forecaster beats A by the same amount — the mini-agent stack
   contributed nothing beyond the abstain-to-naive default. The factor
   corroboration channel never fired: |P| ≤ 0.23 (OSFI), 0.197 (Food
   h=1) < tau_high 0.30 — the threshold is structurally unreachable
   under family-mean aggregation (one strong family ≤ 0.20 of the
   total). On Food h=1 B ≡ A — v2 ≈ naive on the smooth series, so
   abstention adds nothing either. **Round-2 ablation confirms
   inertness**: dropping any one family, factor, or diagnostic on OSFI
   moves B vs A by ≤ 0.09pp (all ≈ −3.6%); even the stability veto is
   redundant with the validation-margin gate (exp73's mechanism).
2. **D (uncertainty) helps only when the base is badly miscalibrated.**
   SMM (A cov 0.105) and Food h=6 (A cov 0.174): D halves/improves CRPS
   and moves coverage toward target. OSFI (A cov 0.796) and Food h=1
   (A cov 0.723): D overshoots (0.851) or hurts CRPS (+1.4%, p=0.011).
   Two design bugs: (a) `fit_uncertainty_params` falls back to
   `(alpha,beta)=(1,1)` — maximum widening — when nothing reaches the
   target; should be `(0,0)`; (b) the rule targets "smallest params with
   validation cov ≥ 0.80" — validation noise causes systematic test
   overshoot; needs a coverage *band* objective and a no-widen clause
   when validation coverage is already ≥ 0.80.
3. **D success bar** (coverage → 0.80, width ≤ 1.5× native, no
   significant CRPS loss): fails on OSFI/Food h=1, passes on
   SMM/Food h=6. D is a rescue tool, not a default.
### Round 2 — ablation + naive reference (2026-10-20)

**Per-family / per-factor / per-diagnostic ablation on OSFI** (drops
each, re-runs B vs A pooled): every single drop leaves B vs A at
≈ −3.6% ± 0.09pp — the mini-agents are individually AND jointly
inert. Even `drop_diag::stability` (the 140-veto driver) barely moves
it, because without the veto those origins fall into the validation-
margin abstain anyway.

**Naive-alone reference** (computed on the same test origins): B ≡
naive on all five beds (OSFI 0.003855 vs 0.003851; Food h=1 0.6354 =
naive; Food h=6 2.2831 = naive; SMM 0.000591 = naive; shifts 0.04694
= naive). Wherever B "beats" A, pure naive beats A by the same
amount. **B contributes nothing beyond the abstain-to-naive default —
the same gating engine exp73's Layer-2 meta-router already ships.**
The ~25-signal synthesis layer is dead weight on these beds.

### Round 3 — parameter search (2026-10-20)

tau_high × tau_low × veto_conf grid (27 configs) on OSFI: results in
`exp75_results/osfi.json::search`. [pending at write time]

### Round 4 — max-family pressure variant (2026-10-20)

**Hypothesis**: family-mean aggregation structurally dilutes a lone
shouting family; `pressure_agg="max"` (strongest family wins) should
let corroboration fire. Result: OSFI −3.52% (p=0.09) vs −3.59% base;
corroboration fired 2/240 origins (vs 0 base) but the two v2 picks
didn't beat the abstain default. Food h=1: +0.0% ≡ base. **Diagnosis**:
even the strongest family never produces |P| ≥ tau_high with val-
winner support — the monthly macro readings are too coarse to
override the validation gate. The synthesis channel is empty, not
just diluted.

**Standing rule status (4–5 rounds)**: R1 design+diagnosis, R2
ablation+naive-reference, R3 param search, R4 max-family variant,
R5 confirmation+arm C. Not killed before Round 5.

_(to be filled as rounds land)_

## Results

_(pending)_

## Ablation

_(per-family ablation + top-factor individual ablation; pending)_

## Verdict

_(pending)_

Files: `experiments/exp75_orchestrator.py`, `experiments/exp75_results.json`,
`src/td_one/agent/factor_agents.py`, `src/td_one/agent/orchestrator.py`,
`tests/test_factor_agents.py`, `tests/test_orchestrator.py`.
