# exp71: Macro Signal integration on SMM

**Date:** 2026-10-20
**Question:** Do the live macro signals (exp70) improve SMM forecasts when fed to ProposedForecasterV2?

## Design (pre-registered)

Arms compared on the 28 SMM origins with signal coverage (2023-11..2026-03), blocked 5-fold time CV, paired t-test + win rate, CRPS:
- Arm A: ProposedForecasterV2(dataset="smm", use_llm_expert=False, use_news=False)
- Arm B: identical + live-signal vector (7 categories from exp70)

Kill criterion (researcher-mindset): B beats A at p<0.05 **and** ≥1% relative effect. The effect-size floor exists because an earlier run produced a p=0.049 "HELPS" on a mean diff of ~1e-7 — correctly rejected as no material effect.

**Enforced iteration rule:** 4–5+ diagnosis-driven rounds. Completed: R1 base encoding → R2 certainty-weighted → R3 lead-1 timing → R4 news_weight param search (0.1/0.5/0.7) → **R5/R6 signal-as-features** (composite 3-dim, per-category 7-dim — the task-sanctioned alternative mechanism: let the router learn from the signals) → R7 drop-one-category ablation.

## Results

All 7 rounds: **NEGLIGIBLE**. Representative: R1 base B=A=0.000687, diff +0.000000, p=0.4890; R6 percat p=nan (zero variance — features changed nothing); R7 all drop-one arms NEGLIGIBLE.

**Verdict: ❌ honest negative — the signal stays a research artifact, not wired in.**

## Diagnosis (why — mechanism-level, not a vague kill)

1. **Signal dilution:** the 7 live directions mostly cancel — mean |stress| = 0.167 → ~5pp regime shift → CRPS moves by ~1e-7. The modulation lever is too weak for a composite signal.
2. **Single-regime window:** all 28 SMM origins in the signal window label 'calm' (the stressed months were the 2022 rate-shock period, pre-coverage). The router is degenerate, and both regime experts collapse to the same global-best model — so *no* signal, however good, can express itself through routing on SMM in this window. The learned-features arm (R5/R6) fails for the same reason: features irrelevant when the router has one class.

This is a mechanism limit (same shape as exp69's oracle-ceiling finding: +11.4% ceiling, +10.2% achieved — the mechanism, not the input, binds), not a verdict on the signal store, which calibrates well (exp70: 7/9 live, holdout 0.79–0.82 on fx/food_prices).

## Open, not closed

- The signal store itself is live and reusable (exp70).
- **Food CPI integration is the natural next test**: its origins span regimes where routing is non-degenerate — the blocked-SMM channel may not apply there. Not run here (out of the sanctioned rounds); recorded as the next hypothesis.
