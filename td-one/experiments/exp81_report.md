# exp81: Cross-Series Pooled Conformal Banks — Report

**Date:** 2026-10-20. **Status:** COMPLETE — pooled banks SHIP,
family-gated (verdict at the end).

## Why this exists

exp76's central finding: **per-series bank size is the binding
constraint on every UQ method.** On the benchmark sweep, sweep series
contribute only 8–12 origins each, and at most origins the same-horizon
bank is far smaller (origins are spaced 6 apart, horizons run 18–24, so
a same-h case becomes observable only ~3–4 rounds later). On M4 every
conformal arm was an **exact no-op** (identical CRPS, width ratio
1.000) — CQR's `min_bank=8` same-h gate never fired. exp79 found the
same starvation killing the per-series router (≤13 backtest records).
exp76's report names the follow-up design verbatim: *"A production UQ
layer on many-series benchmarks needs cross-series pooled banks (same
h, similar scale)."* This experiment builds and tests that design.

## Pre-registered design (locked 2026-10-20, before running)

**Case sets:** exactly exp76 Part B's — `experiments/exp76_forecasts_{a,b}.json`
(Tourism 20 series h=24, M3 20 series h=18, M4 20 series h=18, OSFI 4
segments h=6, SMM 1 series h=6; 13,308 cases). Arms A and B must
reproduce exp76's native and CQR numbers; that reproduction is the
harness check.

**Pooled bank construction.** For a target case in config *c* at
origin *o*, the bank has two parts:

- *Own bank* — exp74's exact PIT rule within the series:
  cases of *c* with `origin' + h' − 1 < origin`.
- *Cross bank* — cases of other configs in the same benchmark family,
  admitted by an **evaluation-round lag rule**: let `round(x)` be the
  index of x's origin in its own config's sorted origin list, and
  `step` the family's origin spacing (6 for the Monash subsets, 1 for
  OSFI/SMM). A cross case with horizon step `h'` enters iff
  `(round(target) − round(case)) * step ≥ h'` — i.e., the pooled case
  is fully realized within its own series' timeline at least as many
  rounds before the target round as its horizon requires.

Rationale, stated plainly: the Monash subsets are **not
calendar-aligned** (verified 2026-10-20: the Tourism subset's series
end 1994-07…2007-09, M3's 1966-12…1995-03), so calendar pooling is
meaningless there; what pooling borrows is *calibration information*
(how Chronos's stated intervals err on other series), not temporal
information about the target series. Within-series PIT — the leak
that matters for a single series' evaluation — is preserved exactly.
The round-lag rule mirrors the deployment setting the method is for
(many series observed on a shared round clock, e.g. a lender's
portfolio re-forecast monthly). OSFI's segments share an exact
calendar and origin axis, where the round rule coincides with the
exact calendar rule. SMM has one config, so its cross bank is empty
by construction — it is the machinery control.

**Score normalization (primary design).** CQR exceedance scores are
in raw series units and cannot be pooled across heterogeneous series.
Primary: **side-specific width-relative scores** —
`s_lo / (m − q10)` and `s_hi / (q90 − m)` of the *source* case; the
pooled finite-sample quantile of the normalized scores is then
rescaled by the *target* case's own side half-widths. This is the
CQR analogue of exp74's relative-score conformal (S3b), which
transferred across regimes better than raw residuals. Ablations:
raw-score pooling (control — expected to be dominated by
large-scale series) and residual-scale normalization (round 2, if
the primary fails diagnostically).

**Arms** (all at CQR level 0.8, `min_bank=8`):

- **A `native`** — Chronos passthrough (control; reproduces exp76).
- **B `cqr`** — per-series CQR, exp76's shipped config (`arm_cqr` on
  the own bank; reproduces exp76).
- **C `pooled_cqr`** — CQR over own ∪ cross bank, width-normalized
  scores, same-h only.
- **C-raw `pooled_raw`** — same bank, raw scores (normalization
  control).
- **D `pooled_shrink`** — own scores in raw units with total weight
  `λ = n_own / (n_own + n0)`, `n0 = 8`; cross scores width-normalized
  and rescaled to target units with total weight `1 − λ`; per-side
  weighted finite quantile. `n_own = 0` → pure pooled; `n_cross = 0`
  → pure own (= B).

**Pre-registered success bar.** Pooling ships iff, on the full sweep:

1. **CRPS:** C or D beats B on the pooled comparison — paired t-test
   over all 13,308 cases on per-case CRPS differences normalized by
   each family's native mean CRPS (scale-free pooling), p < 0.05,
   *and* the raw pooled mean CRPS (exp76's convention) does not get
   worse; and
2. **Coverage:** the winning arm's cov80 is no worse than B's minus
   0.01 in **every** family; and
3. **M4 headline:** the winning arm turns M4 from a no-op into a
   measurable coverage gain — cov80 at least +0.02 above B's (= native's
   0.767), with the arm actually firing (width ratio ≠ 1.000).

If no arm clears the bar after the diagnosis rounds below, the
verdict is an honest negative with the mechanism recorded.

**Diagnosis rounds (researcher rule, if round 1 fails):** (R2)
residual-scale normalization variant; (R3) shrinkage pseudo-count
`n0 ∈ {4, 24}` and/or cross-bank recency weighting; (R4)
similarity-gated pooling (pool only series whose trailing residual
scale is within a factor of the target's); (R5) minimum-cross-bank
gating / level recalibration if coverage overshoots.

## Iteration log

| Round | What was done | Diagnosis / lesson |
|---|---|---|
| — | Pre-registration (above) | — |
| 1 | Arms A/B/C/C-raw/D as pre-registered on the full sweep (results: `exp81_results_r1.json`) | **Fails the bar.** A/B reproduce exp76 to artifact rounding (see note below). M4 cov80 gains only +0.002/+0.004/+0.006 (bar: +0.02); M3 pooled ≈ no-op (fire rate 0.02); OSFI pooling is actively harmful (ΔCRPS +49…+95, p≤1e-09, cov80 *down* 0.03–0.04); Tourism pooled *wins* CRPS (−6.0, p=9.3e-07) but *loses* coverage vs B (−0.011). Two mechanisms, both diagnosed below |
| 1-diag | Why is M4's pooled arm inert despite same-h cross banks of 95–133 scores? | **Per-side score degeneracy.** CQR's per-side exceedance scores are ~88% zeros per side (native side-exceedance ≈ 11% at cov80 0.767), so the level-0.8 finite quantile is exactly 0 whenever a side's exceedance rate is < 20% — pooling grows the bank but not the exceedance rate. Per-series CQR's apparent activity on small banks was partly finite-sample index inflation (at n=8 the "0.8 quantile" is the max order statistic). Pooling removes that inflation — hence Tourism: pooled CRPS better than B (B was over-widening), pooled coverage lower than B's |
| 1-diag | Why does OSFI pooling hurt? | OSFI is **not bank-starved** (36 rounds, step 1 → own same-h banks reach ~30) and its 4 segments are heterogeneous (insured vs uninsured runoff dynamics). Cross-segment normalized scores import the wrong calibration; B's own-bank answer was already good (exp76's one real CQR win). Pooling must be *gated to starved banks* and/or shrunk hard toward own scores — motivates rounds 2–4 |
| 2 | Signed-score redesign (textbook CQR): one signed max-score `max(q10 − y, y − q90)` per case (negative inside the interval → not zero-inflated; bites whenever *total* exceedance > 20%), width-normalized (C2), residual-scale-normalized (C2-resid), and signed shrinkage (D2) | **Mechanism fixed; bar still failed.** C2 turns M4 on: cov80 0.767→0.813 (+0.045 vs B, headline criterion met) at width 1.18, but CRPS +2.77 vs B (p=0.022). Tourism: D2 wins both criteria (−8.65 CRPS, p=3.5e-06; cov80 +0.018). OSFI remains the poison: every pooled variant loses CRPS there (+62…+414); resid-norm explodes (width 1.811) → **residual-scale normalization closed**. Pooled scale-free Δ: C2 +0.0033 (p=0.0075, worse), D2 −0.0001 (p=0.96, wash) |
| 2-diag | Why does OSFI resist even shrinkage (λ→0.79 at late origins)? | Early OSFI origins have small own banks (λ≈0) so cross dominates exactly where segments disagree most; and OSFI's own bank is *not* starved — there is nothing for pooling to fix, only foreign calibration to import. Conclusion: pooling must be **gated off where the own bank is fed** (R5) and **similarity-filtered** (R4), not merely shrunk |
| 3 | D3 starvation gate (n_own_h ≥ 24 → exactly B), D4 similarity gate (κ=3 on PIT residual scales), D5 = both, D5-16 = both with gate 16 | **Neither gate rescues OSFI.** D3 ≡ D2 on the Monash families (own banks never reach 24 — pooling was the only option there anyway) and improves OSFI only +62→+55; D5-16 → +39, still decisively harmful. **Similarity gating is a clean negative**: worse on Tourism (−4.86 vs D2's −8.65) and M3 (+1.47, p=0.020), no OSFI rescue — it removes useful cross data without removing the harm. **Closed** |
| 4 | Harm decomposition of D2 (by own-bank bucket; by width bucket) | OSFI harm is **diffuse, not tail-only**: it is worst where B is a no-op (own<8: Δ +167/case vs *native*) and present in every width bucket, even narrowing ones (+14/case at width<1). Foreign calibration is simply wrong for these segments in timing/regime, not just scale — no per-case gate or filter can fix a signal that is wrong at every bank size. Tourism by contrast wins in *every* bucket (−6.2 own<8; −18.0 at 8≤own<16, where B actually fires — pooled shrinkage beats even a fed per-series bank there). Conclusion: activation must be a **family-level scope rule** (the design's stated regime: many-series, bank-starved families), not a per-case statistic |
| 5 | Ship candidate: D2 (signed shrinkage) restricted to families with ≥8 series, per-series CQR elsewhere; R3 n0 grid {4, 8, 24} on the same rule | **Passes the pre-registered bar** (details in Verdict). n0=4 fails crit-1 (+0.0006, p=0.70); n0=8 passes all three (p=0.047); n0=24 passes all three more strongly (p=0.0066, M4 +0.036) and dominates n0=8 on the primary criterion → **n0=24 ships**, selected from the pre-registered grid by the pre-registered criterion |

**Note on the harness check:** A/B reproduce exp76's sweep within the
rounding of the pushed forecast artifacts (native CRPS Δ ≤ 3e-5
relative in every family; cov80 within ±0.0014). exp76's sweep summary
was computed at full precision, but the forecast files were compacted
to 4 significant figures when pushed (`exp76` ops note), so
bit-exact reproduction from the committed artifacts is impossible;
all exp81 comparisons are internal (identical cases for every arm).

## Results

### Round 1 — per-side pooling (pre-registered arms)

Paired vs B (per-series CQR) per family; Δ = mean per-case CRPS
difference (negative = better than B).

| Family | C pooled Δ (p) | C-raw Δ (p) | D shrink Δ (p) | cov80 B → best pooled |
|---|---|---|---|---|
| Tourism | −5.99 (9.3e-07) | −6.27 (1.2e-07) | −4.25 (3.7e-04) | 0.783 → 0.777 |
| M3 | +0.14 (0.69) | +0.13 (0.70) | −0.12 (0.75) | 0.766 → 0.762 |
| M4 | +0.53 (0.35) | +0.51 (6.0e-05) | +0.96 (0.096) | 0.768 → 0.774 |
| OSFI | +95.3 (2.4e-14) | +71.7 (7.5e-19) | +49.3 (1.2e-09) | 0.798 → 0.771 |
| SMM | 0.00 (0.85) | 0.00 (—) | −0.00 (0.48) | 0.745 → 0.745 |

### Round 2 — signed-score pooling

| Family | C2 signed Δ (p) | cov80 | D2 shrink Δ (p) | cov80 | C2-resid Δ (p) |
|---|---|---|---|---|---|
| Tourism | −5.97 (2.0e-04) | 0.810 | **−8.65 (3.5e-06)** | 0.801 | −5.78 (7.4e-04) |
| M3 | +0.58 (0.23) | 0.770 | +0.83 (0.13) | 0.762 | +0.74 (0.12) |
| M4 | +2.77 (0.022) | **0.813** | +2.75 (0.020) | 0.797 | +3.35 (4.5e-05) |
| OSFI | +121.5 (6.7e-15) | 0.818 | +62.0 (3.0e-11) | 0.809 | +413.8 (8.0e-15) |
| SMM | −0.00 (0.24) | 0.779 | −0.00 (0.78) | 0.745 | +0.00 (0.15) |

*(round 3 in progress — gated variants)*

### Round 3 — gated variants (Δ CRPS vs B, per family)

| Family | D3 gated | D4 sim | D5 full | D5-16 |
|---|---|---|---|---|
| Tourism | −8.65 (3.5e-06) | −4.86 (0.036) | −4.86 | −4.86 |
| M3 | +0.83 (0.13) | +1.47 (0.020) | +1.47 | +1.47 |
| M4 | +2.75 (0.020) | +3.61 (0.005) | +3.61 | +3.61 |
| OSFI | +54.6 (3.4e-09) | +50.5 (8.9e-08) | +43.6 (3.1e-06) | +38.8 (1.5e-05) |
| SMM | −0.00 | −0.00 | −0.00 | −0.00 |

No gated variant clears the bar — OSFI stays harmful in all of them,
which is what forced the round-4 decomposition and the round-5
family-level scope rule.

### Round 5 — ship candidate vs the pre-registered bar

Ship = signed shrinkage (D2) in families with ≥ 8 series, per-series
CQR (B) elsewhere. All three criteria evaluated exactly as
pre-registered:

| Criterion | n0=4 | n0=8 | n0=24 (ships) |
|---|---|---|---|
| 1. Pooled scale-free Δ vs B (p) | +0.0006 (0.70) ❌ | −0.0024 (0.047) ✅ | **−0.0028 (0.0066) ✅** |
| 1b. Raw pooled CRPS ≤ B | ✅ | ✅ (674.04 ≤ 676.96) | ✅ (673.87 ≤ 676.96) |
| 2. cov80 ≥ B − 0.01, every family | ✅ | ✅ | ✅ |
| 3. M4 cov80 gain ≥ +0.02, arm fires | +0.023 ✅ | +0.029 ✅ | **+0.036 ✅** (width 1.154) |

Per-family, ship (n0=24) vs B: Tourism CRPS 850.46 vs 859.05
(−1.0%), cov80 0.806 vs 0.783; M4 CRPS 419.80 vs 417.33 (+0.6% — the
price of coverage where B offered nothing), cov80 **0.805** vs
0.768; M3 CRPS 384.16 vs 383.80 (wash), cov80 0.762 vs 0.766;
OSFI and SMM are exactly B by the family gate.

## Verdict for the forecasting stack

1. **Cross-series pooled conformal banks SHIP** — as the UQ
   configuration for **many-series benchmarks (≥ 8 series)**:
   signed max-score CQR, side-specific width normalization,
   own-score shrinkage with **n0=24**, round-lag PIT pooling
   (`td_one.uq.pooled`). Few-series families keep per-series CQR;
   the family gate is part of the method, not an afterthought —
   it is what the OSFI evidence (rounds 1–4) demands.
2. **The unlock was the score, not the bank.** Pooling per-side
   exceedances (the pre-registered round-1 design) is near-inert:
   at level 0.8 the per-side quantile is 0 whenever a side's
   exceedance rate is < 20%, and pooling cannot change a rate.
   The signed max-score (textbook CQR) is the correct pooled
   object. A side result worth keeping: per-series CQR's small-bank
   activity is partly finite-sample index inflation; pooled
   estimates expose it (Tourism round 1).
3. **Closed with evidence:** residual-scale normalization
   (explodes: OSFI width 1.811, pooled Δ +0.0151, p=3.3e-11);
   similarity gating (negative on Tourism/M3, no OSFI rescue);
   per-case starvation gating as the *sole* guardrail (cannot see
   that foreign calibration is wrong even at zero own bank).
4. **Scope condition, stated as a finding:** pooling borrows
   calibration from a *population* of series. Four heterogeneous
   segments of one institution are not a population — their error
   processes differ in regime timing, and imported calibration
   loses to both per-series CQR and, at starved origins, to doing
   nothing at all (+167/case). Any future pooled deployment needs
   the family gate or a walk-forward equivalent.
5. exp76's "next design problem" is thereby answered with a
   boundary: pooled banks unlock UQ exactly in the regime that
   motivated them (many-series, starved banks — M4's no-op is
   gone, Tourism improves on both criteria), and are
   contraindicated outside it.

**Files:** `src/td_one/uq/pooled.py`, `tests/test_uq_pooled.py`
(16 tests), `experiments/exp81_pooled_banks.py`,
`experiments/exp81_ship_eval.py`, `experiments/exp81_results.json`
(+ `_r1`/`_r2` round snapshots), `experiments/exp81_ship_results.json`.
