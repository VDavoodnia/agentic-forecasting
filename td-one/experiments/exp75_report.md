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
1. **OSFI runoff**: 5 segments × 60 origins, h=1 (exact exp70/exp73
   protocol; A = v2).
2. **Food CPI**: h=1 (59 origins) and h=6 (A = v2 via blocked CV).
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
test. Food CPI series ends ~2025 — factor coverage depends on the
overlap.

## Round log

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
