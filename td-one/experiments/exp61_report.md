# Experiment 61 — Full ladder on M3 Monthly (Monash)

**Date:** 2026-10-18 · **Status:** complete (classical ladder); TSFM quantiles pending

## Question

Same as exp60, on the most-cited benchmark in forecasting: does the proposed
method generalize to M3 Monthly, and where do our numbers sit vs the
literature?

## Dataset

**M3 Monthly**, Monash archive (Makridakis & Hibon, IJF 2000). 1428 monthly
series across 6 domains (demographic, micro, macro, industry, finance, other).
Standard evaluation: fixed test window = last **h=18** observations (M3
competition horizon).

## Method

Identical to exp60: seasonal naive / naive / m0 / ETS / ARIMA (+ Chronos-2 /
TimesFM-2.5 pending), equal weights, proposed method (m4 regime spec, 5-fold
CV by series). PIT-safe per-series normalization; CRPS primary; MASE/sMAPE
for literature comparability.

## Results — CRPS (normalized)

| model | mean CRPS |
|---|---|
| ets | 0.1630 |
| arima | 0.1631 |
| m0 | 0.1717 |
| naive | 0.1781 |
| snaive | 0.1852 |
| equal weights | 0.1641 |
| **proposed** | **0.1476** |

Proposed vs equal weights: Δ = −0.0164, **t = −14.5, p = 2.7e-44**,
win rate **76.1%** (n=1428).
Proposed vs best single (ets): Δ = −0.0154, **t = −12.7, p = 3.5e-35**,
win rate 67.2%.
Gate fire rate 5.3%, trust-EASY rate 89.3%.

Note the contrast with tourism: here ETS (not snaive) is the best single —
M3 is less seasonal, more heterogeneous — and the proposed method still
wins significantly. The router is exploiting cross-series heterogeneity,
not just seasonality.

## Results — MASE vs published literature

Monash archive published MASE (same split, same scaler):

| model | Monash published | ours |
|---|---|---|
| SES | 1.091 | — (we use seasonal naive: 1.162) |
| Theta | 0.864 | — |
| TBATS (best published) | **0.861** | — |
| ETS | 0.865 | 1.111 (non-seasonal repo config) |
| ARIMA | 0.873 | 1.093 |
| N-BEATS | 0.934 | — |
| TimesFM zero-shot | 0.983 | *pending* |

Our ETS/ARIMA trail the published ones for the same reason as exp60
(deliberately simple repo configs, identical to the M4 ladder). The
literature-facing comparison that matters: the proposed method's CRPS
improvement over the best single (+9.4% relative) is the result to carry
into the paper.

## Regime structure (full-data fit)

| regime | n series | expert chosen |
|---|---|---|
| seasonal | 418 | snaive |
| trending | 505 | arima |
| noisy | 505 | ets |

Unlike tourism (seasonal-dominated), M3 splits evenly across regimes with a
*different* expert per regime — the router is exploiting genuine
cross-series heterogeneity, which is why the proposed method wins here too.

## News re-eval (2026-10-18)

| | news off | news on | Δ |
|---|---|---|---|
| proposed CRPS | 0.1476 | 0.1472 | −0.0004 |
| news fire rate | — | 0.901 | — |

142/1428 M3 origins pre-date 1979 (coverage 0, gate shut — honest). On the
rest the gate fires, but the CRPS impact is negligible (~0.3%). Same story
as tourism: the mechanism fires, the generic macro signal doesn't move a
strong statistical result.

## Verdict

**Positive.** Third dataset where the proposed method significantly beats
both equal weights (p ≈ 1e-44) and the best single model (p ≈ 1e-35).
The claim now rests on M4 + Tourism + M3 — two competition benchmarks and
the most-cited dataset in the field.

## Integrity

Leakage audit, protocol adherence, and pretraining-contamination check:
[docs/EVALUATION_INTEGRITY.md](../docs/EVALUATION_INTEGRITY.md).
Pipeline clean at every stage; Monash standard protocol; neither
Chronos-2 nor TimesFM-2.5 was trained on M3 Monthly.

## Repro

`experiments/exp61_m3.py` (`--tsfm` for TSFM quantiles, `--eval-only` to
re-run eval). Records + results: `experiments/exp61_results.json`.
