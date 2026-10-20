# exp66g: Multi-Seed Confirmation of the Counterfactual Risk Tool — Report

**Date:** 2026-10-20
**Status:** Pre-registered; run pending. (Results section filled in
after the run — this header was committed before any exp66g number
existed.)

## Why this experiment exists

exp66f's risk-tool ablation ran on **one seed** (42), n=10 groups.
Among four tools given to the LLM risk classifier, only
**counterfactual sensitivity** moved the needle: Spearman +0.348 vs
+0.120 for stats-only, and the only monotonic configuration on that
seed. SHAP drivers (+0.104) and uplift amenability (+0.120) were
neutral; semantic news alignment *hurt* (−0.174). A single-seed
directional signal is a hypothesis, not a result — exp66d's apparent
LLM gain had already been downgraded once for exactly this reason.
exp66g confirms or kills it.

## Design (pre-registered)

- **Protocol:** identical to exp66f's classifier ablation —
  `MortgageSimulatorV2` (4,000 customers, 36 months, `use_fred_macro
  =False`), last 9 months as test, deterministic grouper (5 clusters,
  grouper seed 42, min group n=200), same per-group aggregate stats.
  GBM + SHAP + counterfactual cutoffs fit on the training period
  only (PIT).
- **Seeds:** 42, 123, 7, 2024, 555 (the exp66f three plus two new).
  All seeds run all tool arms (exp66f ran tools on seed 42 only).
- **Arms:**
  - **A** `llm-stats` — LLM + aggregate stats only
  - **B** `llm-stats+counterfactual` — + counterfactual class-flip
    findings (`agent/risk_tools.counterfactual_flips`)
  - **C** `llm-stats+shap` — + SHAP driver summary (control: extra
    tool context of comparable form, different content)
  - **D** `bins` — deterministic training-tertile bins (no LLM)
- **Primary metric:** mean absolute *bin error* —
  |anchor(level) − realized test attrition| — where each level's
  anchor is the mean training-period realized attrition of the
  groups the deterministic bins placed in that level
  (`evaluation/risk_calibration.py`). One anchor mapping per seed,
  identical for all arms, training data only. Pooled over
  (seed, group) units; arm differences get paired bootstrap 95%
  CIs (2,000 reps).
- **Secondary metrics:** calibration monotonicity (per-seed flags,
  fraction of seeds monotonic, pooled level means) and pooled
  Spearman (continuity with exp66f).
- **Confirmation rule (binding):** the counterfactual claim is
  confirmed only if **both** (1) B beats A on pooled MAE with the
  paired bootstrap CI excluding zero, **and** (2) B beats C the
  same way. Condition (2) separates "counterfactual content helps"
  from "any extra context helps". Anything else = not confirmed,
  recorded honestly with a failure diagnosis.

## Results (baseline run, pre-registered protocol)

**Verdict on the pre-registered claim: NOT CONFIRMED.** Neither
confirmation condition holds.

**Pooled MAE (bin error, lower = better; n=50 units):**

| Arm | Pooled MAE | Pooled Spearman | Seeds monotonic | Pooled monotonic |
|---|---|---|---|---|
| A llm-stats | **0.0611** | +0.414 | 3/5 | ✅ |
| B llm-stats+counterfactual | 0.0629 | +0.426 | 2/5 | ❌ (MEDIUM 0.141 > HIGH 0.130) |
| C llm-stats+shap | 0.0672 | +0.394 | 2/5 | ✅ |
| D bins | 0.0810 | +0.299 | 3/5 | ❌ |

**Paired bootstrap 95% CIs (MAE deltas; positive = second arm better):**

| Comparison | Delta | 95% CI | |
|---|---|---|---|
| B − A (cf vs stats) | −0.0018 | [−0.0106, +0.0079] | n.s. — condition (1) fails |
| B − C (cf vs shap) | +0.0043 | [−0.0068, +0.0162] | n.s. — condition (2) fails |
| A − D (stats vs bins) | +0.0198 | [+0.0054, +0.0345] | LLM beats bins |
| B − D (cf vs bins) | +0.0181 | [+0.0022, +0.0346] | |

**Per-seed (MAE / Spearman / monotonic):**

| Seed | A stats | B +counterfactual | C +shap | D bins |
|---|---|---|---|---|
| 42 | 0.0628 / +0.120 / ❌ | 0.0616 / +0.348 / ✅ | 0.0576 / +0.165 / ❌ | 0.0781 / −0.379 / ❌ |
| 123 | 0.0901 / +0.359 / ✅ | 0.0726 / +0.347 / ❌ | 0.1004 / +0.572 / ✅ | 0.1137 / +0.517 / ✅ |
| 7 | 0.0509 / +0.387 / ✅ | 0.0585 / +0.203 / ❌ | 0.0563 / +0.234 / ✅ | 0.0562 / +0.299 / ✅ |
| 2024 | 0.0568 / +0.243 / ✅ | 0.0673 / +0.349 / ✅ | 0.0738 / +0.211 / ❌ | 0.0989 / +0.328 / ❌ |
| 555 | 0.0451 / +0.849 / ❌ | 0.0546 / +0.766 / ❌ | 0.0481 / +0.609 / ❌ | 0.0581 / +0.564 / ✅ |

Seed 42 replicates exp66f exactly (B Spearman +0.348, A +0.120) —
the protocol is faithful; the effect simply does not generalize
across seeds. B's Spearman edge on seeds 42/2024 comes with *worse*
MAE: the counterfactual context sharpens ranking language but
pushes level assignments around without improving calibration.

## Diagnosis of the baseline failure

Four mechanism checks on the 50 pooled units (from
`exp66g_results.json`, justifications + findings):

1. **The content doesn't drive the benefit.** Where counterfactual
   flips actually exist (19/50 units), B is *worse* than A on
   average (Δ = −0.0066). Where the section is "no flips" boilerplate
   (31/50), B is marginally better (+0.0012). Any B edge comes from
   re-reading the prompt, not from the flips.
2. **The LLM mostly ignores the section.** Only 12/50 B
   justifications (24%) reference the flip content. On that subset B
   helps (+0.0102); on the other 38 it hurts (−0.0056).
   (Correlation, not proof of causation — but a clear prompt-
   engagement suspect.)
3. **B's changes are coin flips.** B changed A's level in only 9/50
   units (7 downgrades, 2 upgrades); of those, B was right 4 times,
   A right 5. The downgrade skew matches the pooled MEDIUM>HIGH
   inversion: "fragile" reads as "less risky" to the LLM.
4. **Changes are uncorrelated with flip presence** (3/19 units with
   flips changed level vs 6/31 without) — the section's presence,
   not its content, moves decisions.

**Diagnosis summary:** primarily an *LLM-use* failure (findings 2–4),
with a possible *format* contribution (finding 1: boilerplate noise
on 62% of units). The original exp66f +0.348 was single-seed noise —
seed 42 is simply a seed where B happens to shine. The iteration
rounds below test the fixable causes before the line is closed
(standing rule: 4–5 diagnosis-driven rounds before a kill).

## Iteration log

Five diagnosis-driven rounds (6 variant arms), same 5 seeds × 10
groups, variant MAE compared to stored baseline arms A (0.0611) and
B (0.0629) with paired bootstrap CIs. Script:
`exp66g_iterations.py`; data: `exp66g_iterations_results.json`.

| Round | Hypothesis tested | Result (MAE; vs A) | What it taught |
|---|---|---|---|
| **R1** prompt tuning — engagement requirement (justification must state how the CF changes/confirms the call) | LLM ignores the section (24% citation) | 0.0660; −0.0048 [−0.0116, +0.0016] n.s. Citation rate rose to **100%** | Engagement was NOT the bottleneck. Forcing the LLM to use the content makes decisions no better (point estimate worse). Diagnosis 2 rejected as the cause. |
| **R2** format — omit the CF section when findings are empty | "No flips" boilerplate (62% of units) is pure noise | 0.0663; −0.0052 [−0.0113, −0.0003] — marginally significantly *worse* than A | Boilerplate was not hurting; removing it doesn't rescue the tool. Diagnosis 1 (format noise) rejected. |
| **R3a** tool param — edit size 0.5σ (finer flips) | Flip granularity too coarse | 0.0648; −0.0037 [−0.0101, +0.0023] n.s. | Finer sensitivity doesn't help. |
| **R3b** tool param — edit size 2σ (only large, robust flips) | 1σ flips are noise-level | 0.0616; −0.0005 [−0.0086, +0.0090] n.s.; vs B +0.0013 n.s. | Best variant MAE, still indistinguishable from A and from B. Granularity is not the problem at either end. |
| **R4** prompt tuning — interpretation note ("a flip means SENSITIVE, not lower risk") | LLM misreads fragility as lower risk (downgrade skew) | 0.0623; −0.0012 [−0.0068, +0.0034] n.s.; vs B +0.0006 n.s. | Correcting the interpretation changes nothing measurable. Diagnosis 3 rejected as the cause. |
| **R5** inert-content control — CF section shuffled from a *different* group in the same seed | The content itself is inert; only prompt bulk matters | 0.0651; −0.0040 [−0.0173, +0.0082] n.s.; vs B −0.0022 n.s. | **Decisive.** Wrong-group counterfactuals perform the same as the real ones. The section's content carries no usable signal for this classifier. |

Every variant lands in the MAE band 0.0616–0.0663 with CIs vs A
covering zero (except R2, barely excluding it in the *wrong*
direction). Five independent fixes — engagement, format, two
granularities, interpretation — plus a shuffled-content control all
converge on the same answer.

## Final verdict

**❌ The counterfactual risk tool does not help the LLM risk
classifier. Claim killed with evidence (baseline + 5 iteration
rounds).** exp66f's +0.348 was single-seed noise: seed 42 replicates
exactly (+0.348) and is simply a favourable seed. The failure is not
the tool's output format, not the LLM's engagement, and not the
flip granularity — R5 shows the content is inert for this task at
this scale. `counterfactual_flips` remains a legitimate *analyst*
explanation device (it describes the GBM faithfully); what is dead
is feeding it to the classifier as prompt context. The same applies
to the other exp66f tools as classifier context: SHAP drivers were
neutral-to-worse here (C pooled MAE 0.0672, worst LLM arm), and news
alignment was already negative in exp66f.

**One genuine positive from exp66g:** the stats-only LLM classifier
*does* beat the deterministic bins on the bin-error metric, pooled
over 5 seeds — A vs D delta +0.0198, 95% CI [+0.0054, +0.0345]
(excluding zero). exp66f's Spearman-based version of this comparison
was inconclusive; on calibration error the LLM's edge is real at
this scale. The classifier of record is therefore **LLM + stats,
no tool context**.

## Files

- Code: [risk_calibration.py](sandbox://workspace/agentic-forecasting/td-one/src/td_one/evaluation/risk_calibration.py), [risk_tools.py](sandbox://workspace/agentic-forecasting/td-one/src/td_one/agent/risk_tools.py)
- Experiment: [exp66g_counterfactual_confirm.py](sandbox://workspace/agentic-forecasting/td-one/experiments/exp66g_counterfactual_confirm.py)
- Results: `exp66g_results.json`
- Tests: `tests/test_risk_calibration.py` (9)
