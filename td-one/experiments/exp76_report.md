# exp76: Fuse CQR with the Meta Uncertainty Signal + UQ Benchmark Sweep — Report

**Date:** 2026-10-20. **Status:** IN PROGRESS.
**Code:** `src/td_one/uq/fusion.py`, `experiments/exp76_fusion_battery.py`,
`experiments/exp76_generate_data.py`, `experiments/exp76_sweep.py`.
**Tests:** `tests/test_uq_fusion.py` (10, green).

## Why this exists

exp74's two wins: **CQR** became the default UQ layer (Food CPI CRPS
1.2824 vs 1.3035, p=1.7e-04, never harmful) and the **story/meta
uncertainty signal** beat numeric-only decisively (paired p=2.0e-11) but
under-covers (0.65–0.68) — a CRPS-improving signal, not a coverage
layer. The stated next step: fuse CQR's coverage honesty with the
meta signal's CRPS gain, then sweep the UQ layer across more
benchmarks.

## PIT correction vs exp74

During this work I found exp74 keyed story/macro features by the **first
forecast month** `ym_of_index(idx)` for origin `idx` (context
`series[:idx]`) — one month of lookahead, since the story pipeline's
`before_ymd` and `signals_asof` both include that month. exp76 keys all
narrative features by the **last observed month**
`month_before(ym_of_index(idx))` (see `td_one.uq.fusion.month_before`).
All S7/S8 comparisons here use the corrected keying; exp74's published
story-ablation numbers carry the one-month-lookahead caveat.

## Pre-registered design

**Fusion arms** (both reuse exp74's GBM error-percentile model; details in
`src/td_one/uq/fusion.py`):
- **F1 `MetaLevelCqr` (meta-informed conformal level):** rank the current
  case's predicted error scale among the bank's same-horizon predictions
  (predicted-error percentile); modulate the CQR conformal level around
  0.8 by `k*(2*rank-1)`, with the scalar `k` picked from a grid by CRPS
  on a chronological replay of the most recent 48 bank cases. At rank
  0.5 or `k=0` it is exactly plain CQR.
- **F2 `ConditionalCqr` (feature-conditional CQR):** two GBM quantile
  regressors predict the CQR exceedance scores `(s_lo, s_hi)` conditional
  on the case features; the predicted conditional score quantile is the
  per-origin offset. This is exp74's "conformalized-meta, done right":
  the model predicts the *conformal score*, not the raw error, so the
  double-counted-shrinkage failure (round 5) cannot recur. Falls back to
  plain CQR on degenerate banks.
- **Regime-gated ACI:** ACI runs only when the last-6 same-horizon bank
  shows ≥2 coverage misses or the case's vol_12 exceeds 1.5× the bank
  median; native otherwise.

**Success criterion:** a fused arm wins if it beats plain CQR on CRPS
(paired p<0.05) with coverage80 at/above CQR's (within 1pp), pooled on
Food CPI and/or the exp67 synth scenarios.

**Part B sweep:** native / CQR / gated-ACI / meta-numeric / both fusion
arms in numeric-feature form, on Tourism (20 series, h=24), M3 (20, h=18),
M4 (20, h=18), OSFI runoff (4 segments, h=6), SMM (h=6).

**Program stance (2026-10-20, confirmed): macro signals are an
uncertainty feature, not a forecast-routing feature.** exp77 showed no
signal can express itself through the regime router when origins are
calm (mechanism-level result); exp74 showed narrative features carry a
real uncertainty signal (story vs numeric p=2.0e-11). So S8 is a
first-class result here, not an optional extra.

**S8 macro ablation:** `macro_signals` now covers 2023-10→2026-09
(36 months, 725 rows, 22 categories). With corrected keying the overlap
with Food CPI origins (feature month ≤ 2026-09, story keys 2021-01..2025-12)
is origins 166–197 (32 origins); the *clean* S8 scope with both signals
present is 166–192 (27 origins). Arms: `meta_story` (numeric + 7 story
features) vs `meta_full` (numeric + story + macro aggregates over all
22 categories) vs `meta_full_live` (numeric + story + macro aggregates
over the **7 live categories** — inflation, services_costs,
food_prices, labour, fx, housing, wars_geopolitics, from the exp70
calibration gate). All macro access is PIT-safe via `signals_asof`.
If macro adds nothing over story-only, that is a reported finding —
per-category diagnosis ships with it (see below), not a reason to drop
the feature.

## Iteration log (researcher rule: 4–5 diagnosis-driven rounds)

| Round | What was done | Diagnosis / lesson |
|---|---|---|
| 1 | Fusion battery on Food CPI + exp67 synth (corrected keying); story-ablation recomputed; S8 made first-class with the live-7 macro arm (per user directive 2026-10-20: macro is officially an uncertainty feature, not a routing feature) | … |
| 2 | Benchmark sweep (5 families) | … |
| 3 | … | … |
| 4 | … | … |

## Results

### Part A — fusion verdict

…

### Part B — benchmark sweep

…

### S8 macro follow-up (Part C)

…

## Verdict for the forecasting stack

…
