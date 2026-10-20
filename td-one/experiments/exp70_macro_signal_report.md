# exp70: Macro Signal Service — calibration

**Date:** 2026-10-20
**Question:** Does a dedicated news-reading LLM produce categorical macro signals that track realized data?

## Design (pre-registered)

22 macro categories × 36 months (2023-10..2026-09). Per (month, category), Gemini Flash reads the month's articles (PIT: pub_date within the month) and outputs strict JSON `{direction: -2..+2, risk, certainty: 0..1, summary citing titles}` → stored in `macro_signals` (725 cells).

Calibration: each signal vs FRED anchors on train 2023-10..2025-06, holdout 2025-07..2026-09. Gate: directional accuracy > 55% on train → live.

**Enforced iteration rule (Vandad 2026-10-19):** a failed gate is a tripwire, not an execution. 5 diagnosis-driven rounds per category: R1 base → R2 timing offsets (leads/lags) → R3 certainty-weighting → R4 strong-only (|dir|≥2) → R5 2-month smoothing. A category still failing after R5 is dead *with its iteration log*.

## Results

**7 live categories:**

| Category | R1 | Rounds (R1/R2/R3/R4/R5) | corr | Holdout | Live via |
|---|---|---|---|---|---|
| fx | 0.70 (n=20) | .70/.70/.71/1.00/.67 | +0.63 | 0.79 (n=14) | R1 base |
| food_prices | 0.62 (n=13) | .62/.67/.60/.67/.67 | +0.38 | **0.82 (n=11)** | R1 base |
| labour | 0.65 (n=17) | .65/.65/.63/nan/.61 | +0.32 | 0.00 (n=2) ⚠️ | R1 base |
| inflation | 0.56 (n=16) | .56/**.93**/.55/.50/.46 | +0.17 | n=0 | R1 base |
| housing | 0.75 (n=4) ⚠️ | .75/.75/.71/.00/.60 | +0.35 | n=0 | R1 base |
| services_costs | 0.47 (n=15) | .47/.50/.49/**.57**/.41 | +0.26 | n=0 | R4 strong-only |
| wars_geopolitics | 0.52 (n=21) | .52/.55/.51/.40/**.56** | +0.17 | 0.50 (n=14) | R5 smoothing |

**2 dead (honest negatives, 5 rounds each):**

- **basic_needs** — R1 0.35, corr −0.28: *anti-correlated with the same CPI anchor inflation tracks* (R2–R5 max 0.47). Diagnosis: the extractor's sign convention for this category is inverted vs the rubric (affordability framing). A sign-flip would reach ~0.65 on train but is **not applied**: a post-hoc sign flip risks overfitting, and the holdout is empty (anchor currency). Recorded as a hypothesis, not a rescue.
- **central_bank_rates** — n=1: the extractor says direction=0 for 20/21 months (BoC held rates; delta≈0). Structurally silent: the category has no signal most months, not a miscalibrated one.

**13 unanchored** (ai_startups, car_prices, elections_politics, flight_prices, global_trade, immigration_law, income_tax, markets_sp500, minimum_wage, pandemics_health, salary_region, school_childcare, small_business) — no FRED anchor; research artifacts.

**Certainty does not calibrate:** high-certainty (≥0.7) accuracy is not consistently above medium (fx 0.69 vs 0.75; food_prices 0.55 vs 1.00; wars 0.52). The LLM's self-reported certainty is not a reliable filter.

## Notable round findings

- **inflation R2=0.93**: timing-offset analysis finds the signal aligns at a 1-month offset — CPI news in month M anticipates the release (which lands mid-M+1). The signal is *forward-looking*, correctly captured by the gate (R1=0.56 passed).
- **fx R4=1.00**: strong signals (|dir|≥2) are perfect on train — magnitude carries information.
- **housing**: live on n=4 — flagged thin; treat as provisional.

## Coverage (news volume, honest)

792 cells fetched, 725 signals (67 zero-article cells — mostly salary_region, labour, car_prices). Coverage is well below the 20 articles/month target: category means range 1.7 (salary_region, 22 zero-months) to 8.0 (wars). Dense-enough for the core macro categories; thin categories fail structurally (coverage → silence).

## Conclusion

The Macro Signal Service **passes calibration**: 7/9 anchored categories go live over 5 rounds, with the two strongest (fx, food_prices) validating on holdout (0.79/0.82). The reusable store is live for downstream use; per-category `live_via` records the timing/aggregation variant each signal ships with.
