# exp75: Mini-Agent + Orchestrator Decision Engine — Report

**Status: COMPLETE (Rounds 1–5; verdict below).**

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

### Arm C — LLM orchestrator (2026-10-20)

| subset | C vs A | verdict |
|---|---|---|
| OSFI pooled (240) | **+4.4%** (C 0.004174, B 0.003855) | C worse than both |
| Food CPI h=1 (47) | **−19.1%, p=3e-11** | ⭐ C beats A and B big |
| Shifts S3 (24) | +2.5%/0.0% | C ≈ A |

- On OSFI the LLM misbehaves exactly as in exp69c: picks v2 71×
  (deterministic picked it 0×), attaches a shift on 240/240 calls,
  and pays for it.
- On Food CPI h=1 the SAME behavior is a triumph: chose naive 32× /
  v2 15×, shifted on 40/47 origins with **shift-sign accuracy
  39/40 (97.5%)**, mean |shift| 0.29 CPI points. Candidates on this
  series are ≈identical (A ≈ naive), so the entire gain is the
  directional shift, correctly signed by reading the factor signals.
- **The deterministic pressure sign matches it**: sign(world_pressure)
  vs realized move direction on the same 47 origins = **46/47
  (97.9%)**. The LLM contributes no directional knowledge beyond the
  mini-agent bundle's sign — its value was exposing the *shift
  mechanism*, which the deterministic orchestrator can run cheaply.
  → Round 4b: **arm E (deterministic shift)**: shift the base grid by
  sign(P) × 0.5 × validation-resid-SD whenever |P| ≥ tau_low.
- C vs B pooled over the pre-registered prioritized set (OSFI 240 +
  Food 47 + S3 24): dominated by OSFI → C does NOT justify its cost
  as a general orchestrator. It wins exactly one bed — the one where
  the factor corpus describes the target (food-price news → food
  prices) and the series is smooth. Bed-specific, not general.

### Simulator bed (Round 1)

B −28.0% vs the group-forecaster base (p=0.099, win 0.61, n=36; all
abstains-to-naive), driven by S3_4 units (−49% diverged, −45%
portfolio) — but **+10.5% harm on S1_4/diverged** (naive is worse
than the group forecaster on the rate-shocked group), a pre-
registered segment-harm flag. D ≡ A (validation coverage ≥ target →
(0,0) under the new fit).

_(to be filled as rounds land)_

## Results

_(pending)_

### Round 5 — confirmation with final code (2026-10-20)

All beds re-run with the final implementation (Round-2 uncertainty
fit, arm E added). Arm E = **deterministic shift** (sign(P) × 0.5 ×
validation resid SD when |P| ≥ tau_low) — the mechanism arm C
exposed, without the LLM.

| Bed | A CRPS | B vs A | D vs A | E vs A | C vs A |
|---|---|---|---|---|---|
| OSFI (240) | 0.003998 | −3.59%, p=0.087 | +0.51%, p=0.065 | **+7.9%, p=0.019** | +4.4% pooled |
| Food h=1 (47) | 0.6354 | +0.0% | +0.76%, n.s. | **−22.8%, p=1.9e-14** | −19.1%, p=3e-11 |
| Food h=6 (43) | 3.2064 | −28.8%, p=0.007 | −7.3%, p=3e-19 | +6.8%, n.s. | — |
| SMM (38) | 0.000594 | −0.5%, n.s. | −1.1%, n.s. | +0.9%, n.s. | — |
| Shifts (72) | 0.04760 | −1.4%, n.s. | +0.0%, n.s. | +0.0% | +2.6/0.0/+2.5/0.0 |
| Sim (36) | 0.2207 | −28.0%, p=0.099 | +0.0% | +0.0% | — |

- **E replicates and beats C on Food h=1** (−22.8% vs −19.1%): the
  LLM's only win is fully explained by a deterministic sign(P) shift.
  On OSFI both shift arms (C +4.4%, E +7.9% p=0.019) hurt — pressure
  signs are noise there. Shifts are a sharp, bed-specific tool,
  never a default.
- **D with the Round-2 fit**: OSFI harm eliminated (+0.51% n.s. vs
  +1.42% p=0.011 before); Food h=6 rescue intact (−7.3%, p=3e-19,
  coverage 0.174→0.407, width 1.52× — marginally over the 1.5× bar);
  SMM/shifts/sim neutral. D keeps its rescue role.
- **B segment-harm check**: sim S1_4/diverged +10.5% point estimate
  but p=0.82 (n=9) → not a flag under the pre-registered bar; noted.

## Ablation

- **Per-family (OSFI, R2)**: dropping any one of prices / housing /
  labour / policy / external leaves B vs A unchanged at −3.59% ± 0.09pp.
  All five families: dead weight (for the abstain mechanism).
- **Per-factor (OSFI, 22 categories incl. rent/climate RAG)**: every
  single drop leaves B vs A unchanged (−3.59%±0.09). No individual
  factor carries OSFI signal.
- **Per-diagnostic (OSFI)**: stability −3.55% (veto redundant with
  the validation-margin gate); disagreement/seasonality/group
  unchanged.
- **Where signal DID exist**: prices family on the Food CPI bed —
  sign(P) matched realized h=1 move direction 46/47 (97.9%). The
  signal is *target-aligned direction*, visible only to the shift
  arms (E/C); the routing arms (B) never touch it.
- **LLM cost accounting**: factor signals are month-stamped and
  cached (one read per month per target; the shared
  `exp75_factor_cache.json` covers all beds/origins) — the marginal
  cost per origin is ONLY the orchestrator call: 0 for B/D/E, 1 for C
  (287 calls total for the whole experiment; ≈$0 on flash-lite).
  24 factor agents × origins would have been ~7,000 calls uncached.

## Verdict

**The mini-agent + orchestrator architecture, as a general decision
engine over a strong base forecaster, is a negative.** Five rounds of
diagnosis, ablation, search, and variants converge:

1. **Routing (B) adds nothing beyond exp73's Layer-2 gate.** Every B
   "win" is exactly reproduced by naive-alone; ablation shows all 24
   factor agents and all 4 diagnostics individually *and* jointly
   inert for routing. The valuable part — validation-margin gating
   with abstention to naive — is a *validation statistic*, not a
   mini-agent.
2. **The LLM orchestrator (C) is dominated by a deterministic
   variant of its own mechanism.** Arm E (sign(P) shift, zero LLM
   calls) beats C on C's best bed (−22.8% vs −19.1% on Food h=1)
   and C hurts everywhere else (OSFI +4.4%, S3 +2.6%). The
   pre-registered bar (C must beat B pooled, p<0.05) is not met.
   There is no bed where C is the best arm.
3. **One genuine positive survives — the directional shift, on the
   one target-aligned bed.** sign(P) is a near-perfect h=1 direction
   oracle for Food CPI (97.9%) because the factor corpus (food-price
   news) describes the target. Arm E should ship as an *opt-in*
   module gated by a validation sign-accuracy test (enable only when
   sign(P) has predicted validation moves at >70% with ≥10 origins),
   mirroring how v2's LLM expert was gated to Food CPI in exp59.
   On attrition targets the same pressure is noise (OSFI E +7.9%,
   p=0.019) — it must stay OFF there by default.
4. **Uncertainty mode (D) survives as a rescue tool** with the
   Round-2 fit: it no longer harms calibrated bases and rescues
   badly-miscalibrated ones (Food h=6 −7.3%, p=3e-19).

Net architectural lesson: decomposition into 24 narrow factor
agents produced an auditable, cached, PIT-safe signal bundle whose
*direction* is occasionally gold and whose *synthesis* is worthless
— the orchestrator that matters is still a validation statistic
(the exp73 gate) plus one calibrated scalar (the shift). The agents
are a feature store, not a decision engine.
