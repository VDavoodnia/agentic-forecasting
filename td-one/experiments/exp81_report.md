# exp81: Cross-Series Pooled Conformal Banks — Report

**Date:** 2026-10-20. **Status:** IN PROGRESS (pre-registration locked
before any sweep run).

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
| — | Pre-registration (this section) | — |

## Results

*(to be filled after the runs)*
