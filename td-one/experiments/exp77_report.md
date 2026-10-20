# exp77: Calibrated Macro Signals → Food CPI

**Date:** 2026-10-20
**Question:** Does a *calibrated* news signal add forecast value where
*uncalibrated* raw news never did (exp44/46/69 all failed)?

## Design (pre-registered)

Food CPI, 33 signal-window origins (idx 166..198, target months
2023-11..2026-07), blocked 5-fold time CV, paired t-test + win rate, CRPS.
All news access via `signals_asof` (PIT: asof = last observed month).

Arms:
- **A:** v2, no news (subset baseline)
- **B:** v2 + live-signal vector (7 calibrated categories, base encoding)
- **C:** v2 + signal vector over ALL 22 categories (gating control)
- **D:** v1 (classical router) + live signals; **D0:** v1 no news
- Reproduction check: v2 on all 59 origins must match exp64.

Kill criterion (exp71-MS bar): beat A at p < 0.05 **and** ≥ 1% relative effect.

## Results

Reproduction: **bit-exact** — v2 all-origins CRPS = 0.5321591236198281,
identical to `exp64_results.json` (diff 0.0). The harness is validated.

| Arm | B | A | diff | p | win | Verdict |
|---|---|---|---|---|---|---|
| R1 B: v2 + live | 0.3697 | 0.3697 | +0.00000 | 0.5517 | 0.33 | NEGLIGIBLE |
| R1 C: v2 + all 22 | 0.3697 | 0.3697 | +0.00000 | 0.4584 | 0.12 | NEGLIGIBLE |
| R1 D: v1 + live | 0.8704 | 0.8704 | +0.00000 | 0.4552 | 0.24 | NEGLIGIBLE |
| R2 B′: v2 + live ex-food_prices | 0.3697 | 0.3697 | +0.00000 | 0.6332 | 0.24 | NEGLIGIBLE |
| R3 F: per-category features (router learns) | 0.3697 | 0.3697 | 0 | nan | 0.00 | NEGLIGIBLE |
| R5 holdout window only (n=13) | 0.4895 | 0.4895 | −0.00000 | 0.7802 | 0.31 | NEGLIGIBLE |
| R6 drop-one ×7 (best: drop-services_costs) | 0.3697 | 0.3697 | −0.00000 | 0.3066 | 0.30 | NEGLIGIBLE |

**Verdict: ❌ honest negative — mechanism-blocked.**

## Diagnosis (mechanism-level)

The signal **does** reach the router (`news_used` 33/33, coverage gate
passes everywhere, mean |stress| = 0.203 — not dilute). It changes
nothing because the router has no lever to pull:

- All 33 signal-window origins label **"calm"** under the generic regime
  spec (no recent changepoints, vol_ratio ≤ 1.0 throughout 2023–2026).
- The router is degenerate (`None`) in **all 5 folds**; per-regime
  experts both collapse to the LLM expert (`{calm: llm, stressed: llm}`,
  global best = llm).
- News modulation shifts ~6pp of mass from calm to stressed — between
  two identical experts. Combined forecasts are identical up to float
  noise (diffs ~1e-16, hence the cosmetic p/win values).

This is the SMM structural block (exp71-MS) in a second flavor: SMM was
single-class labels; Food CPI is single-class labels *plus* expert
collapse. **The service agent's stated hypothesis is refuted:** Food CPI
origins in the signal window do NOT span non-degenerate regimes.

## Iteration log

| Round | Hypothesis tested | Result |
|---|---|---|
| R1 (3 arms) | hand-modulation channel works | no — all ±0.00000 |
| R2 | null is a selection artifact of food_prices (calibrated on Food CPI itself) | no — B′ equally null |
| R3 | router can learn from per-category features | no — router degenerate, p=nan |
| R4 | weight/timing search | **skipped as provably inert**: with both regime experts = llm, no news_weight or lag can move the combination (per-fold evidence above) |
| R5 | null driven by calibration-train overlap | no — holdout window equally null |
| R6 | one category carries a masked effect | no — all drop-one ±0.00000 |

## Answer to the key question

**Untested, not refuted.** The experiment cannot say whether a
calibrated signal beats an uncalibrated one — the router channel was
structurally closed on this window before signal quality could matter.
What *is* settled: exp44 proved the modulation channel works when
regimes are non-degenerate (synthetic p=0.0068); both real datasets
tested (SMM, Food CPI) are uniformly calm in the 2023–2026 signal
window, so the channel is closed on both. The signal store stays live.

## Side observations

- v2 on the 2023–2026 subset (0.3697) ≪ full-run (0.5322): the LLM
  expert dominates this window (active expert llm 31/33).
- v1 on the subset is 0.8704 — the classical router adds nothing here;
  the LLM expert is the entire game on Food CPI.

## Next

1. The natural home for these signals is **uncertainty, not location**:
   exp74 showed story/meta features predict forecast uncertainty
   (p=2.0e-11); exp76's S8 macro re-run should test the calibrated
   signal vector as a UQ feature.
2. Any future *location* test needs a non-calm window — none exists in
   the current signal coverage for Food CPI or SMM.

Files: `experiments/exp77_foodcpi_macro_integration.py`,
`experiments/exp77_results.json`, `tests/test_exp77_macro_foodcpi.py`.
