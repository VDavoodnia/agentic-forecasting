# Experiment 60 — Full ladder on Tourism Monthly (Monash)

**Date:** 2026-10-18 · **Status:** complete (classical ladder); TSFM quantiles in progress

## Question

Does the proposed method (exp47: regime-probability router + gates) generalize
beyond M4 to a second standard benchmark — and how do our numbers compare with
the published literature on that benchmark?

## Dataset

**Tourism Monthly**, Monash Time Series Forecasting Archive (Godahewa et al.,
NeurIPS 2021). 366 monthly series from the Kaggle tourism forecasting
competition (Athanasopoulos et al., IJF 2011). Lengths 91–333 obs. Standard
evaluation: fixed test window = last **h=24** observations. Strong, stable
seasonality — the regime structure (seasonal / trending / noisy) the router
was designed for.

## Method

Same ladder as the M4 experiments (exp42–47), one origin per series (Monash
standard split):
- **Classical:** seasonal naive (lag-12), naive, m0 (3-pt moving avg),
  ETS (non-seasonal, repo config), ARIMA(1,1,1)
- **TSFM:** Chronos-2, TimesFM-2.5 (real weights, local snapshot; quantiles
  interpolated to the 9-level grid) — *in progress*
- **Combination:** equal weights, proposed method (ProposedForecaster,
  m4 regime spec, 5-fold CV by series, exp47 protocol)

Per-series PIT-safe min-max normalization → CRPS in normalized space.
MASE (seasonal-naive scaler, m=12) and sMAPE on raw scale for literature
comparability.

## Results — CRPS (normalized)

| model | mean CRPS |
|---|---|
| snaive | 0.1387 |
| ets | 0.2025 |
| arima | 0.2025 |
| m0 | 0.2107 |
| naive | 0.2182 |
| equal weights | 0.1869 |
| **proposed** | **0.1282** |

Proposed vs equal weights: Δ = −0.0586, **t = −21.9, p = 1.9e-68**,
win rate **94.8%** (n=366).
Proposed vs best single (snaive): Δ = −0.0102, **t = −13.6, p = 2.6e-34**,
win rate 91.5%.
Proposed vs router-only ablation: Δ = −0.0064, t = −17.1, p = 1.7e-48
(gates + conformal add value beyond the router).
Gate fire rate 3.6%, trust-EASY rate 94.8%.

![CRPS ladder](../docs/plots/monash_crps_ladder.png)

## Results — 7-model ladder (classical + TSFM zero-shot)

| model | mean CRPS | mean MASE |
|---|---|---|
| timesfm25 | **0.0965** | **0.996** |
| chronos2 | 0.0972 | 1.012 |
| **proposed** | **0.0974** | — |
| snaive | 0.1387 | 1.080 |
| equal weights (7) | 0.1525 | — |
| ets / arima | 0.2025 | 2.281 |

Proposed vs equal weights (7): Δ = −0.0552, **t = −25.2, p = 8.2e-82**,
win rate 93.4%.
Proposed vs best single (timesfm25, selected on train folds, no hindsight):
Δ = −0.0003, t = −0.81, **p = 0.42**, win rate 47.8% — a tie.
Proposed vs router-only: p = 0.18 — also a tie.

Honest reading: the TSFMs dominate the classical zoo on this seasonal
dataset (30% better than snaive), and the proposed method correctly
identifies the ex-ante best model and matches it — but adds nothing beyond
it. The method's value here is robustness (matching the best model without
knowing which it is in advance), not beating a dominant model. Contamination
check: Tourism Monthly is in neither Chronos-2's nor TimesFM-2.5's published
training corpora (see EVALUATION_INTEGRITY.md), so these are clean
zero-shot numbers. Note: our zero-shot TimesFM-2.5 MASE 0.996 beats every
published Monash baseline (best DeepAR 1.409). Context: the table's
TimesFM-ZS (2.273) is TimesFM **1.0** (Nov 2024 update); the TimeFound
paper (arXiv 2503.04118) reports Tourism-Monthly MASE 1.918 for TimesFM
(likely 2.0), 1.819 for Chronos-Large, 1.610 for TimeFound-Large. Our
numbers (TimesFM-2.5 0.996, Chronos-2 1.012) reflect ~19 months of
frontier progress on the same protocol — i.e. a 2025 model beating
2021–2024 baselines, which is real but expected. The benchmark table is
maintained (Nov 2025 update) but its baselines lag the frontier, and the
TSFM literature has largely moved to GIFT-Eval. Eyeballed forecasts
confirm the gains are genuine seasonal tracking, not a pipeline bug.

## Results — MASE vs published literature

Monash archive published MASE (same split, same scaler):

| model | Monash published | ours |
|---|---|---|
| seasonal naive | SES 3.306 (non-seasonal!) | **1.080** |
| ETS | 1.526 (auto-ETS w/ seasonality) | 2.281 (non-seasonal repo config) |
| ARIMA | 1.589 | 2.286 |
| Theta | 1.649 | — |
| DeepAR (best published) | **1.409** | — |
| TimesFM zero-shot | 2.273 (v1.0, Nov 2024) | **0.996 (v2.5)** |
| Chronos zero-shot | — | **1.012 (v2)** |

Notes:
- Our seasonal naive (1.080) beats every published MASE on this dataset —
  tourism monthly is dominated by stable seasonality, and the literature's
  SES baseline is non-seasonal, which flatters the comparison. The honest
  read: snaive is the model to beat here, and **the proposed method beats it
  significantly** (CRPS p = 2.6e-34) on the classical zoo.
- Our ETS/ARIMA look worse than the published ones because ours are the
  deliberately simple repo configs (non-seasonal ETS, ARIMA(1,1,1)), kept
  identical to the M4 ladder for comparability — not the best possible ETS.
- The published TimesFM-ZS 2.273 turned out to be a **version artifact**:
  it's TimesFM 1.0 (Nov 2024 update). TimeFound (Mar 2025) reports 1.918
  for TimesFM (2.0-era). Our 0.996 is TimesFM 2.5 (Sep 2025) — ~19 months
  of frontier progress on the identical protocol, i.e. a 2025 model beating
  2021–2024 baselines. Real, but expected; not a methodological breakthrough.
  The Monash table is maintained (Nov 2025 update) but its baselines lag the
  frontier, and the TSFM literature has largely moved to GIFT-Eval.

![Ours vs published MASE](../docs/plots/monash_mase_vs_literature.png)

## Regime structure (full-data fit)

| regime | n series | expert chosen |
|---|---|---|
| seasonal | 296 | snaive |
| noisy | 64 | ets |
| trending | 6 | snaive |

The router correctly identifies seasonal naive as the expert for the 296
seasonal series — matching the CRPS table where snaive is the best single.

## News re-eval (exp44 mechanism, 2026-10-18)

With the 1979–2017 Perplexity backfill complete (17,738 articles), the news
module was turned on (12-month PIT-safe window → coverage + stress signal;
gate: coverage ≥ 0.5, then P(regime) shifts toward noisy/stressed):

| | news off | news on | Δ |
|---|---|---|---|
| proposed CRPS | 0.09737 | 0.09716 | −0.00021 |
| news fire rate | — | 1.000 | — |

The gate fires on every series (coverage is complete for the 1992–2005
origins), but the CRPS impact is negligible (~0.2%). The news signal is
generic macro stress (series are anonymized), not per-series information —
consistent with exp44's "real-data blocked" finding, now tested rather than
blocked: **the mechanism works, the signal doesn't move the needle here.**

## Verdict

**Positive, with the story corrected.** The proposed method's M4 result
generalizes: on a second, independent benchmark with a longer horizon
(h=24), it beats equal weights (p ≈ 1e-68) *and* the best single classical
model (p ≈ 1e-34). The "constrained structural priors beat flexible
combination" claim now rests on three datasets (M4, tourism, M3), not one.

Two corrections to the naive story:
1. The TSFM results (TimesFM-2.5 MASE 0.996, "beating" the published best
   1.409) are **version progress, not method progress** — a 2025 foundation
   model vs 2021–2024 baselines on a maintained-but-lagging table. The
   frontier is measured on GIFT-Eval now, not Monash.
2. Against a dominant zero-shot model, the proposed method **ties but does
   not beat** the ex-ante best single (p = 0.42). Its value there is
   robustness — matching the best model without knowing which it is — not
   outperformance. The outperformance claim lives and dies on the classical
   zoo, where no single model dominates.

Open question (exp63): whether agentic components (LLM selection, news)
add anything on top of the statistical router + strong TSFMs.

## Integrity

Leakage audit, protocol adherence, and pretraining-contamination check:
[docs/EVALUATION_INTEGRITY.md](../docs/EVALUATION_INTEGRITY.md).
Pipeline clean at every stage; Monash standard protocol; neither
Chronos-2 nor TimesFM-2.5 was trained on Tourism Monthly
(M4 Monthly was — see the doc for the contamination matrix).

## Repro

`experiments/exp60_tourism.py` (`--tsfm` for TSFM quantiles, `--eval-only`
to re-run eval). Records + results: `experiments/exp60_results.json`.
Shared machinery: `experiments/monash_common.py`. Literature baselines:
`experiments/monash_literature.json`.
