# exp76: Fuse CQR with the Meta Uncertainty Signal + UQ Benchmark Sweep — Report

**Date:** 2026-10-20. **Status:** COMPLETE.
**Code:** `src/td_one/uq/fusion.py`, `experiments/exp76_fusion_battery.py`,
`experiments/exp76_generate_data.py`, `experiments/exp76_sweep.py`,
`experiments/exp76_analyze.py`, `experiments/exp76_macro_diagnosis.py`.
**Tests:** `tests/test_uq_fusion.py` (14, green).

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
| 1 | Fusion battery on Food CPI + exp67 synth with corrected PIT keying; story ablation recomputed under the fix; S8 made first-class with the live-7 macro arm (program stance, 2026-10-20: macro is an uncertainty feature, not a routing feature) | **F1 wins on Food CPI** (−1.7% CRPS vs CQR, p=1.8e-04, cov80 0.856 vs 0.824) — pre-registered success met. F2 fails (+1.0%, p=6.5e-04). Story signal **survives the PIT fix**: meta_story vs meta_numeric −2.8%, p=1.9e-06 (exp74's leaky figure was p=2.0e-11). Story ablations hold; the lookahead inflated, but did not create, the effect |
| 2 | Mechanism diagnosis of F1's two regressions (food h≤2: +1.2%, p=0.019; synth S1@4σ: +1.1%, p=0.001) and F2's uniform failure; per-category macro diagnosis | F1's food win is a **long-horizon phenomenon** (h≥5: −2.5%, p=2.2e-04); at h≤2 the meta rank is uninformative and modulation is pure harm. Under S1@4σ the bank is all-extreme → rank saturates. Macro diagnosis: only **inflation |direction|** tracks residual error (Spearman ρ=+0.408, p=0.028, uncorrected, n=29 origins); all other live categories ≈ 0 |
| 3 | F1 variants: h3 gate (min_h=3) + widen-only; F2 shrink=0.5; S8 inflation-only arm (meta_story_inf) | **h3-gated F1 is the champion**: −1.8% vs CQR, p=2.0e-05, and −0.2% vs plain F1 (p=0.0195). Widen-only is *worse* than full F1 (+0.2%, p=0.0017) — the h≤2 harm is modulation noise, not narrowing per se. Shrunk F2 still fails (+1.1%, p=5.0e-05) → conditional-score noise dominates even at 50% pooling. Inflation-only macro: −0.3% vs story-only (p=0.072) — direction matches the diagnosis, not significant |
| 4 | F2 final iteration: shrink=0.25 (stronger pooling) | Still fails (+1.27%, p=1.1e-05) — and *more* shrinkage is slightly worse, which kills the variance hypothesis: F2's offsets are **biased, not just noisy**. Mechanism found: F2 pools exceedance scores across all horizons for both training and its global fallback, while plain CQR conditions on h; longer-h scores are larger, so pooled offsets systematically over-widen short horizons. A per-h F2 would need banks ≫ what a single series provides. **F2 closed** after 3 variants + mechanism; revisit only with cross-series pooled banks |
| 5 | Benchmark sweep (Tourism/M3/M4/OSFI/SMM) with the champion + plain CQR + gated ACI | The fusion gain does **not** transfer (details in Part B): with per-series banks of 8–12 origins the UQ machinery is starved — M4 arms are an exact no-op. CQR itself generalizes as a *coverage* fix, not a CRPS win: pooled ≈ neutral (+0.2%, p=0.057), genuinely helpful on OSFI (−1.3%, p=0.001), CRPS-costly on Tourism (+0.8%, p=5.7e-08) while buying coverage (0.770→0.782) |

## Results

### Part A — fusion verdict

**F1 (MetaLevelCqr) fuses successfully; F2 (ConditionalCqr) does not.**

Food CPI, pooled (666 cases, paired vs plain CQR):

| Arm | CRPS | Δ vs CQR | paired p | cov80 | verdict |
|---|---|---|---|---|---|
| native | 1.3035 | +1.6% | — | 0.799 | — |
| **plain CQR** | 1.2824 | — | — | 0.824 | exp74 default |
| F1 meta-level (story feats) | 1.2612 | −1.7% | 1.8e-04 | 0.856 | **wins** |
| **F1 h3-gated (champion)** | **1.2587** | **−1.8%** | **2.0e-05** | 0.854 | **best** |
| F1 widen-only | 1.2632 | −1.5% | 6.4e-04 | 0.856 | wins, < F1 |
| F1 meta-level (numeric only) | 1.2724 | −0.8% | 0.032 | 0.853 | wins, weaker |
| F2 conditional (story) | 1.2957 | +1.0% | 6.5e-04 | 0.827 | **fails** |
| F2 conditional (numeric) | 1.3048 | +1.7% | 6.0e-07 | 0.821 | **fails** |
| F2 conditional shrunk 0.5 | 1.2970 | +1.1% | 5.0e-05 | 0.818 | **fails** |
| F2 conditional shrunk 0.25 | 1.2987 | +1.3% | 1.1e-05 | — | **fails** |

Mechanism: modulating the conformal level by predicted-error
percentile pays exactly where uncertainty is *differentiable* — long
horizons on Food CPI (h≥5, −2.5%). The meta rank carries no signal at
h≤2 (+1.2% harm → the h3 gate removes it) and saturates under
bank-wide regime shift (S1@4σ, +1.1%; on synth pooled all F1 variants
run +0.4–0.5% vs CQR). Win rate is low (0.22–0.23): F1 loses most
origins slightly and wins a few by a lot — it is buying tail
insurance, correctly priced.

F2's failure is mechanistic, not parametric: GBM quantile regression
on the exceedance score produces offsets noisier than the global
conformal quantile; shrinkage does not rescue it (0.5: +1.1%,
p=5.0e-05; 0.25: +1.27%, p=1.1e-05 — *worse* with more shrinkage).
Round 4 found the actual flaw: F2 pools scores across horizons while
CQR conditions on h, so its offsets are biased upward at short
horizons (see iteration log). F2 is closed; a per-h conditional
design needs cross-series banks this protocol doesn't have.

**Gated ACI**: regime gating earns its keep as designed — on synth it
beats raw ACI (−1.5%, p=1.4e-08) and on Food CPI it is a wash
(+0.1%, p=0.84) with better coverage (0.803 vs 0.769). Ship the gated
form as the ACI variant.

**Story ablation, corrected keying** (food_story_subset, 360 cases):
meta_story beats meta_numeric by −2.8% (p=1.9e-06, win 0.54) and
beats native (−1.9%, p=0.043), while meta_numeric alone does not beat
native (+0.9%, p=0.47). exp74's headline narrative-signal result
survives its one-month-lookahead correction, at reduced strength.

### Part B — benchmark sweep

Same UQ arms in numeric-feature form (no story pipeline on these
benchmarks), Chronos native forecasts generated fresh
(`experiments/exp76_forecasts_{a,b}.json`, 778 rows; split for the
push size limit, values at 4 significant figures). CRPS is in raw series
units — compare within a row family only. Δ and p are paired vs
native; cov80 native→CQR.

| Family (cases) | native CRPS | CQR Δ (p) | F1-h3 Δ (p) | F2-num Δ (p) | gated-ACI Δ (p) | meta-num Δ (p) | cov80 nat→CQR | CQR width |
|---|---|---|---|---|---|---|---|---|---|
| Tourism (5760) | 852.64 | +0.75% (5.7e-08) | +1.02% (9.2e-10) | +0.60% (5.7e-06) | +1.38% (7.8e-12) | +1.80% (2.2e-14) | 0.770→0.782 | 1.046 |
| M3 (3600) | 383.93 | −0.03% (0.71) | +0.03% (0.73) | +0.11% (0.18) | +0.39% (2.2e-06) | +1.52% (2.9e-10) | 0.759→0.765 | 1.027 |
| M4 (2880) | 417.34 | 0.00% (—) | 0.00% (—) | +0.72% (8.6e-11) | 0.00% (—) | +3.06% (3.7e-09) | 0.767→0.767 | 1.000 |
| OSFI (864) | 1732.00 | **−1.28% (0.0012)** | −0.92% (0.012) | −1.04% (1.0e-04) | +1.48% (6.5e-05) | +0.99% (0.0018) | 0.749→0.796 | 1.154 |
| SMM (204) | ~0.0008 | ≈0% (0.48) | ≈0% (0.63) | ≈0% (0.53) | ≈0% (0.49) | ≈0% (0.31) | 0.725→0.745 | 1.038 |
| **Pooled (13308)** | 675.66 | +0.19% (0.057) | +0.41% (2.3e-04) | +0.27% (0.0022) | +1.06% (7.1e-17) | +1.79% (3.9e-29) | 0.764→0.775 | 1.038 |

Where the UQ layer generalizes, where it breaks:

- **Bank size is the binding constraint.** The study driver banks
  *within series* (PIT rule), so sweep series contribute 8–12 origins
  each. On M4 (8 origins) every conformal arm is an **exact no-op**
  (identical CRPS, width 1.000) — there is no bank to learn an offset
  from. Food CPI's 111-origin bank is the favorable regime; the sweep
  measures the starved regime. A production UQ layer on many-series
  benchmarks needs cross-series pooled banks (same h, similar scale)
  — flagged as the follow-up design, not tested here.
- **CQR is a coverage repair, not a CRPS engine, outside Food CPI.**
  It lifts cov80 toward nominal everywhere it has a bank (OSFI
  0.749→0.796 is the standout, with the only real CRPS win, −1.3%)
  but costs a little CRPS where Chronos is already calibrated
  (Tourism +0.75%). Pooled: +0.19%, p=0.057 — statistically a wash.
  exp74's "never harmful" needs this correction: *never very harmful,
  occasionally helpful, coverage-positive*.
- **The F1 fusion does not transfer in numeric form.** h3-F1 ≈ plain
  F1 ≈ slightly worse than CQR on Tourism/M3/M4, slightly better
  than CQR only on OSFI (where plain CQR is best anyway). The Food
  CPI fusion win needs (a) the story features and (b) a long bank —
  both absent here.
- **meta_numeric is harmful everywhere in the sweep** (under-covers
  0.71–0.75, CRPS +1.0–3.1%): without the story signal, the raw meta
  model should not ship in any form tested.
- **Gated ACI** helps only where ACI itself helps (synth battery);
  on real sweep families it is neutral-to-harmful (+1.5% OSFI).

### S8 macro follow-up (Part C)

**Verdict: macro features add nothing measurable over story-only on
the available overlap — reported as a finding, with diagnosis.**

Paired CRPS (case-level; only 27 independent origins in the clean
scope, so treat p-values as optimistic):

| Scope | Comparison | Δ | p | win |
|---|---|---|---|---|
| overlap 166–192 (162 cases) | meta_full_live vs meta_story | +0.1% | 0.885 | 0.36 |
| overlap 166–192 | meta_full (all-22) vs meta_story | −0.2% | 0.565 | 0.38 |
| overlap 166–192 | live-7 vs all-22 | +0.2% | 0.463 | 0.33 |
| overlap 166–192 | meta_story_inf vs meta_story | −0.3% | 0.072 | 0.31 |
| macro 166–197 (174 cases) | meta_full_live vs meta_story | +0.2% | 0.572 | 0.36 |

Three compounding reasons, in decreasing confidence:

1. **The overlap window is signal-quiet.** In 2023-10→ origins the
   story effect itself vanishes (meta_story vs meta_numeric −0.2%,
   p=0.83, vs −2.8% on the full story subset): these are calm-regime
   origins where Chronos is already near-calibrated (native cov80
   0.99). There is little residual uncertainty left for *any*
   narrative feature to explain.
2. **Aggregation dilutes the one live wire.** Per-category diagnosis
   (`experiments/exp76_macro_diagnosis.py`, 29 origins): inflation
   |direction| correlates with native residual magnitude at ρ=+0.408
   (p=0.028, uncorrected); services_costs, food_prices, labour, fx,
   housing, wars_geopolitics are all |ρ|≤0.19, ns. The aggregate
   encodings average one informative category with six silent ones.
   The inflation-only arm is the best macro arm (−0.3%, p=0.072) —
   consistent with the diagnosis, not conclusive.
3. **Power.** 27 origins × 6 horizons is a thin basis for a ±0.3%
   effect; case-level p-values overstate independence.

Program stance stands (macro = uncertainty feature); within the UQ
meta model the macro half is **kept but flagged**: the encoding to
carry forward is story + standalone inflation magnitude, to be
re-tested when the backfill covers a higher-volatility window.

## Verdict for the forecasting stack

1. **Default UQ layer stays CQR**, with the exp76 correction to its
   claim: it is a coverage-repair layer — pooled across five new
   benchmark families it is CRPS-neutral (+0.19%, p=0.057), lifts
   coverage toward nominal wherever it has a bank, and wins outright
   on OSFI (−1.3%). "Never harmful" becomes "never very harmful".
2. **Ship the F1 fusion (h3-gated MetaLevelCqr) where its two inputs
   exist**: a long single-series bank (≥ ~40 origins) and the story
   feature pipeline. On Food CPI it is the best UQ arm tested
   (−1.8% vs CQR, p=2.0e-05, coverage up). Do not ship its
   numeric-only form on many-series benchmarks — no transfer.
3. **Do not ship** the raw meta arm in numeric form (harmful
   everywhere in the sweep), F2 conditional CQR (closed after 3
   variants; h-pooling bias), or ungated ACI. Gated ACI remains the
   conditional option for volatile regimes (synth evidence).
4. **Macro signals: uncertainty feature, kept, flagged.** No
   measurable add-on over story-only in the quiet 2023-10+ window;
   the encoding to carry is story + standalone inflation magnitude,
   retest pending a higher-volatility window or a longer backfill.
5. **Next design problem (new):** cross-series pooled conformal
   banks. The sweep's binding constraint was bank size, not arm
   design — M4's exact no-op is the tell. Pooling same-h cases across
   series (scale-normalized scores) is the unlock for UQ on
   many-series benchmarks.
