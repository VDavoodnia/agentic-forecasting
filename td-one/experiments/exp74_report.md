# exp74: Predict Chronos's Uncertainty, Not Its Forecast — Report

**Date:** 2026-10-20. **Status:** COMPLETE — 2 wins, 1 conditional, story-ablation positive.
**Data:** `experiments/exp74_results.json` (14,718 case-results; 223 Chronos forecast origins:
S1@1σ/4σ, S2@4σ, S3@4σ, control ×2 seeds + 111 Food CPI origins), `experiments/exp74_story_signals.json`
(60 story-pipeline classifications, 2021–2025). Code: `src/td_one/uq/`, `experiments/exp74_uq_battery.py`.
**Background:** docs/UQ_SURVEY.md (Phase 1, 2024+ literature).

## Why this exists

exp68/69/69b/69c all failed the same way: correcting the forecast's *location*
needs timing precision we don't have (exp69c: direction right 2/3, but size
overshot 2.6×, timing +1.5mo early — and even the corrected-size/timing
variants stayed n.s.). The pivot, from the user: **don't move Chronos's
forecast — predict its uncertainty** at each origin (recent volatility, recent
error history, regime diagnostics, narrative macro signals) and adapt the
intervals. The uncertainty features are all observable pre-origin; no timing
precision is required.

## Pre-registered design (locked in the survey doc before running)

**Arms** (S-codes from the survey): S1 native Chronos (baseline), S2
fixed-width expansion, S3 rolling-residual conformal (EnbPI-style), S4 ACI,
S5 weighted conformal (recency × volatility-similarity), S6 CQR on Chronos
quantiles, S7 error-percentile GBM meta-model (numeric features), S8 =
S7 + story/macro features (the ablation). Variants added in-round:
S3b/S4c width-relative-score conformal (conformal_rel, aci_rel), S4b
bias-corrected ACI.

**Mechanics.** Every arm is expressed as side-aware scale factors on the
native 21-level quantile grid around the median (q′ = m + r·(q − m)), so
coverage, width, and CRPS are directly comparable. The study driver is
structurally PIT-safe: at each origin the arm sees only the *bank* of cases
with (origin′ + h′ − 1) < origin, verified by a prefix-vs-full regression
test (`test_study_pit_prefix_equals_full`).

**Pre-registered success criterion** (from the survey doc): a method wins if
it moves coverage toward nominal **without width ratio > 1.25 vs native**
on pooled test origins — **or** improves CRPS significantly.

**Researcher rule in force:** 4–5 diagnosis-driven iteration rounds before
any negative is final (user standing instruction 2026-10-19/20). The log:

| Round | What was done | Diagnosis / lesson |
|---|---|---|
| 1 | Base battery (9 arms), synth + food | Native Chronos is already ~well-calibrated *pooled* (synth cov80 0.790); headroom is in the shock/volatile regimes. Meta_numeric *shrinks* intervals (wr 0.93) — it learned Chronos is conservative in calm times. |
| 2 | Added width-relative conformal arms (conformal_rel, aci_rel); raw case dumps | At S1@4σ straddle origins, ALL residual-based arms under-widen (cov80 best 0.44): the past is pre-shift, so no residual history can predict onset width. Relative scores transfer only marginally better. |
| 3 | Deep slices: post-shift, per-shift, food story subset | Post-shift, ACI over-widens (wr up to 2.06) while CQR stays sane (1.13). On the volatile food subset, **aci CRPS beats native (p=3.6e-05, win 0.61, wr 0.965)** and **meta_full beats meta_numeric (paired p=2e-11)**. |
| 4 | ACI param search (γ × window, 15 configs) on food story subset | Every config beats native CRPS significantly; defaults (γ=0.05, w=24) are near-optimal on the CRPS/coverage trade (cov80 0.783, wr 0.965). Tuned numbers flagged as in-sample; defaults already good. |
| 5 | Conformalized-meta variant + permutation importances | Conformalization overcorrects (CRPS 1.834 > native 1.723; diagnosed: GBM's shrinkage makes the multiplicative needed-scale overshoot). `st_magnitude` ranks 5th of 27 features (+0.447), ahead of vol_6 — the narrative channel is real. |

## Results

### CQR (S6): ✅ WIN — permanent place in the stack

| Scope | cov80 | cov90 | width ratio | CRPS vs native |
|---|---|---|---|---|
| Pooled synth (n=672) | **0.811** (0.790) | **0.946** (0.936) | 1.034 | 0.0449 tie (p=0.69) |
| Food CPI (n=666) | **0.824** (0.799) | **0.904** (0.884) | 1.078 | **1.2824 vs 1.3035, p=1.7e-04** |
| Control (n=192) | 0.818 (0.812) | 0.969 (0.943) | 1.024 | tie (p=0.63) |

Meets the pre-registered bar on both scopes: coverage moves toward nominal at
both levels, width never exceeds +8%, and food CRPS improves significantly.
It is the only arm that is never harmful anywhere (win rates low but means
tie-or-better). Asymmetric tails are its edge over symmetric conformal.

### ACI (S4): ✅ conditional win — regime-dependent, keep as the option

On the volatile food story subset (n=360): CRPS 1.6614 vs native 1.7231,
**p=3.6e-05, win 0.61, wr 0.965**. On pooled synth, CRPS is slightly worse
(0.0467 vs 0.0449). Reading: ACI wins where Chronos is systematically
miscalibrated (real volatile regimes — it shrinks *and* widens adaptively),
loses where native is calibrated. Verdict: ACI behind a regime flag (high
recent miss streak / volatility), not always-on.

### Meta-full vs meta-numeric (S8 vs S7): ✅ the story ablation is real

On the food story subset, **meta_full CRPS 1.6758 vs meta_numeric 1.7383,
paired t=−6.92, p=2.0e-11, win 0.56**, with *better* coverage too
(0.681 vs 0.644). Pooled food: meta_full 1.2795 (p=0.0138) vs meta_numeric
1.3150 (p=0.326, n.s.). **Narrative/trend features predict Chronos's
uncertainty beyond what volatility and error-history capture** — the user's
hypothesis, confirmed with a paired test. `st_magnitude` is the #5 feature
overall. Caveat: both meta arms under-cover (0.65–0.68 vs 0.80 nominal)
because they shrink toward Chronos's calm-times conservatism; they are
CRPS-improving uncertainty *signals*, not coverage-guarantee layers.

### Weighted conformal (S5): directional positive with a coverage cost

Best food CRPS of the battery (1.2782, p=2e-05, win 0.56) but cov80 drops to
0.728. Same trade as the meta arms — do-no-harm on coverage fails the
pre-registered bar despite the CRPS win.

### Honest negatives (with mechanisms)

- **Straddle/onset coverage:** no residual-based method rescues the S1@4σ
  straddle slice (best cov80 0.444 vs target 0.80). Mechanism: the bank is
  pre-shift, so every residual score is small — the method is *structurally*
  blind before onset. Only exogenous (story) features fire pre-onset, which
  is exactly why the meta ablation matters.
- **aci_bias (S4b):** persistently worse (food CRPS 1.50 vs native 1.30,
  p=7e-31). The EWM bias estimate chases the regime instead of anchoring the
  interval — BC-ACI's trick needs per-horizon care we didn't get right.
  Diagnosed, not adopted.
- **conformalized-meta:** overcorrects past native (diagnosed: multiplicative
  needed-scale double-counts GBM shrinkage). Future variant: ratio-target
  calibration at a lower quantile level, or CQR applied to the meta
  half-width.
- **meta-numeric alone:** no win pooled (food p=0.326) — numeric diagnostics
  alone don't beat Chronos's native calibration.

## Verdict for the forecasting stack

1. **CQR on the native quantile grid becomes the default UQ layer** —
   honest calibration with no CRPS cost, never harmful, +3–8% width.
2. **ACI (defaults γ=0.05, w=24) becomes the regime-conditional option**,
   gated by recent miss-streak/volatility.
3. **The story-feature meta-model is the CRPS-improving uncertainty
   signal** — it belongs in the stack as a *signal*, feeding the meta-router
   (exp73's analyst toolbox), not as a replacement for conformal intervals.
   Next: fuse S6 (CQR) with S8's signal — calibrate CQR's quantile level on
   the meta-predicted uncertainty (the conformalized-meta idea, done right).
4. Macro-signal features: the interface is wired (`macro_features`,
   `signals_asof`), but the service's backfill only covers 2026-04→2026-09,
   so the ablation's "macro" half is under-powered. Re-run the S8 ablation
   once the 2021–2025 backfill lands — logged as the one open follow-up.

## Engineering

- `src/td_one/uq/`: `methods.py` (11 arms + finite/weighted quantile
  helpers), `features.py` (numeric/story/macro extractors), `evaluate.py`
  (PIT-safe study driver, paired-test summaries). 18 tests green, ruff +
  format clean. Two real bugs caught by tests during development: ACI
  warm-up misses, and None-valued story fields.
- Story signals: 60 origins × 3 LLM calls (flash-lite, cached;
  `exp74_story_llm_cache.json` local-only per convention; parsed signals
  committed).
- PIT safety: structural (bank rule) + regression test; story/macro
  features keyed by origin month via `signals_asof` / pre-origin news only.

Files: [exp74_report.md](sandbox://workspace/agentic-forecasting/td-one/experiments/exp74_report.md),
[docs/UQ_SURVEY.md](sandbox://workspace/agentic-forecasting/td-one/docs/UQ_SURVEY.md),
`src/td_one/uq/`, `experiments/exp74_uq_battery.py`, `experiments/exp74_results.json`,
`experiments/exp74_story_signals.json`, `tests/test_uq.py`.
