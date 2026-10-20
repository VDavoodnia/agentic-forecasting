# exp84 — Competitor spread at loan level (Future Direction 3)

**Verdict: the pre-registered bar is MET — decisively.** The origination
spread (loan rate minus contemporaneous Freddie PMMS) predicts 24-month
prepayment with the **same positive sign in all three eras** (+0.56 /
+0.49 / +1.06 logit points per 100bp, all p < 1e-150), and training on
one era and testing on another, the spread model **beats the no-spread
baseline on log-loss in 5 of 6 era pairs** (mean Δ +0.056; the sixth is a
−0.0001 tie) and on AUC in all 6. This is the loan-level revisit that
exp71b prescribed after closing the aggregate broker-spread channel as a
regime artifact: at the level where the mechanism actually lives — the
individual borrower's refinance incentive — the spread signal is real,
era-stable, and out-of-sample useful. The aggregate series failed
because aggregation (and forward-fill) destroyed exactly this
cross-sectional variation.

## Step 1 — Inventory: true loan-level SFLLD data IS present

`td-one/data/freddie/raw/` holds the Freddie Mac Single-Family Loan-Level
Dataset as delivered: **21 origination files + 21 performance files,
2021Q1–2026Q1, ~37GB total** (origination ~18–153MB each; performance up
to 7.5GB for 2021Q1), plus derived SMM extracts
(`smm_YYYYQX.parquet`, `smm_monthly_2021_2026.csv`) and 7z staging parts.
The repo loader (`src/td_one/freddie/`) defines the full origination
schema (credit score, first payment date, LTV/CLTV, DTI, UPB, **note
rate**, channel, product, purpose, term) and performance schema with
zero-balance codes — everything the design needs, no fallback to
coarser granularity required.

Built panel (`exp84_build_panel.py`, per-vintage parts, raw files never
committed): **8,888,789 loans**, of which **7,320,258 FRM 30-year**
(350–365mo term) form the analysis population. Termination outcomes
from performance files (first zero-balance event per loan; follow-up
ends 2026-03 in every vintage):

| Event | Loans |
|---|---|
| Voluntary prepayment (zb 01) | 1,582,734 |
| Other termination (zb 96: 25,412; 02: 1,540; 09: 1,859; 16: 1,298; 15: 803; 03: 306) | 31,218 |
| Still active at 2026-03 | 7,274,837 |

Market reference: `data/fred/MORTGAGE30US.parquet` — Freddie Mac PMMS
30-yr fixed, weekly 1971→2026-10 (2,899 obs). The BoC posted backbone
was rejected (Canadian rates for US loans); PMMS is the dataset's own
survey rate. Spread uses the PMMS monthly mean of the month **before**
first payment (closing-month proxy) — knowable at origination, PIT-safe.

Spread distribution (bps, FRM30): p10 −73, p25 −44, median −4, p75 +19,
p90 +60, range −500…+323.

## Step 2 — Design (pre-registered before fitting)

- **Spread:** `spread_bps = (orig_rate − PMMS_{fp−1mo}) × 100`.
- **Outcome:** prepay within H months of first payment (zb 01);
  **complete windows only** (follow-up ≥ H); loans with a competing
  termination inside H excluded (competing-risk convention, as in the
  repo's SMM work). Primary H=24; robustness H=12, H=36.
- **Eras (first-payment date):** E1 2021 · E2 2022–23 · E3 2024+.
  E3 at H=24 is necessarily early-2024 cohorts (later cohorts lack 24
  observable months by 2026-03); H=12 gives the fuller E3.
- **Model:** logit of prepay on spread/100 + controls — FICO bands
  (<680, 680–719, 720–759, 760+), LTV bands (≤80, 80–90, 90–95, >95),
  DTI bands (≤36, 36–45, >45, missing), log UPB, purpose dummies
  (P purchase / C cash-out refi / N no-cash-out refi).
- **Bar:** (a) spread slope same sign in all 3 eras at H=24; (b) pooled
  OOS (all 6 ordered era pairs) mean Δlog-loss > 0 vs no-spread.
  Mechanism prediction: slope > 0 (above-market originators gain most
  from refinancing later).

## Results

### R1 — Primary era fits (H=24; population n; fits on ≤200k/era samples)

| Era | n | Base prepay rate | Coef /100bp | SE | OR /100bp | p |
|---|---|---|---|---|---|---|
| E1 2021 | 2,619,685 | 0.103 | **+0.563** | 0.021 | 1.76 | 6.7e-161 |
| E2 2022–23 | 2,510,042 | 0.111 | **+0.487** | 0.012 | 1.63 | ≈0 |
| E3 2024+ | 172,954 | 0.303 | **+1.055** | 0.010 | 2.87 | ≈0 |

Same sign everywhere; magnitude grows in the high-rate era, exactly as
the refinance-incentive mechanism predicts (a 2024 borrower 100bp above
market had the most to gain when rates eased in late 2024–25 — base
prepay rate 30%).

### R1-OOS — Cross-era transfer (train era A → test era B, ≤400k/era)

| Pair | Δlog-loss (spread − no-spread) | ΔAUC |
|---|---|---|
| E1→E2 | +0.0004 | +0.058 |
| E1→E3 | +0.0682 | +0.155 |
| E2→E1 | −0.0001 | +0.027 |
| E2→E3 | +0.0882 | +0.126 |
| E3→E1 | +0.0794 | +0.035 |
| E3→E2 | +0.1018 | +0.069 |
| **Mean** | **+0.0563** | **+0.078 (6/6 positive)** |

Bar (b) met. Transfer gains are largest into/out of E3, mirroring the
era slope pattern.

### Iteration log (researcher protocol — diagnose, don't kill)

- **R2 — spread deciles (shape):** prepay rate rises monotonically with
  spread decile in every era (E1 7.6%→15.4% across decile means −43bp→
  +81bp, one mid-range dip 0.104→0.085; E2 6.6%→17.9%, clean; E3
  11.4%→45.8%, clean and steep). No U-shape or threshold artifact; a
  linear-in-logit spread is an adequate form.
- **R3a — refinance-incentive transform (log rate ratio):** coefs
  +1.83 / +2.53 / +7.60 by era, all p≈0. Sign-stable under the ratio
  transform as well.
- **R3b — time-varying hazard (150k loans, ~3.9M loan-months):** the
  *current* incentive (PMMS_t − orig_rate) enters at −0.226 / −0.325 /
  −1.118 per 100bp (all hugely significant) — as market rates fall below
  the note rate, monthly prepay hazard rises, in every era: the
  mechanism, live. Conditional on that path, the *origination* spread
  stays positive in E1 (+0.367) and E2 (+0.173) but flips in E3 (−0.203,
  p=3.3e-06). Reading: in E3 the unconditional slope works **through**
  the realized incentive path; conditional on the path, a high
  origination spread marks rate-priced-for-risk borrowers who refinance
  *less* (credit/LLPA constraints bind). The primary (origination-time,
  decision-relevant) specification is unaffected.
- **R4 — interactions (pooled, era dummies):** spread main +0.827
  (p=7e-107); spread×PMMS-level −0.012 (p=0.052, negligible); 
  spread×refi-purpose −0.140 (p=3e-15) — purchase borrowers respond
  ~0.14 more per 100bp than refinance borrowers. Main sign untouched.
- **R5 — robustness battery:**
  - H=12: +0.689 / +0.494 / +0.934 (E3 now the fuller 2024–25 cohorts,
    base rate 9.9%) — sign-stable.
  - H=36: E1 +0.506, E2 +0.363; E3 not estimable (no 2024+ cohort has
    36 complete months by 2026-03) — a data limit, stated not hidden.
  - PMMS timing variants (fp month / fp−2mo): +0.56…+0.61 (E1),
    +0.48…+0.58 (E2), +0.90…+1.00 (E3) — not a timing artifact.
  - **Own-mean reference** (loan rate minus the dataset's own
    same-month mean rate, no PMMS): +0.747 / +0.700 / +1.021 — the
    signal is not manufactured by the PMMS reference choice.
  - **Vintage-quarter fixed effects** (within-vintage identification):
    +0.670 / +0.596 / +1.055 — survives absorbing all vintage-level
    rate environment differences.

### Design corrections made en route (recorded, per protocol)

1. **Censoring fix:** the first horizon filter also admitted
   event-≤H loans with follow-up <H, selecting fast prepayers from late
   cohorts while excluding their active peers — it inflated E3's base
   rate to 0.70 and crushed its slope to +0.03. Complete-windows-only
   is the correct binary-outcome design; E3 then reads base 0.303,
   slope +1.055. (Had we stopped at the first pass, we would have
   mis-diagnosed E3 as "signal dies in 2024+".)
2. **Purpose codes:** SFLLD refinances are C (cash-out) + N (no-cash-
   out) — there is no "R". An early encoder lumped N into "other";
   fixing it moved the R1 slopes by ≤0.02 (controls were categorical
   either way) and enabled the R4 refi interaction.

## Caveats

- **Associational, not causal.** Origination spread partly prices
  borrower risk and channel; controls + vintage FE shrink but do not
  eliminate confounding. The claim is predictive: spread is a stable,
  transferable loan-level feature — which is what exp71b asked.
- E3 at H=24 covers first payments Jan–Mar 2024 only (complete-window
  requirement); H=12 (fp through Mar 2025) corroborates at +0.934.
- Scope: FRM 30-yr only, 2021–2026Q1 vintages, PMMS as market
  reference. Nothing here resurrects the *aggregate* broker-spread
  series — exp71b's negative stands at aggregate level. The correct
  home for this signal is the loan-level attrition/SMM stack (e.g. as
  a per-loan covariate in segment prepay models), where the
  cross-section exists.

## Files

- `experiments/exp84_build_panel.py` — per-vintage panel build
  (restart-resilient parts; raw data gitignored, never committed)
- `experiments/exp84_make_frame.py` — compact numeric frame encoder
- `experiments/exp84_make_samples.py` — per-horizon analysis samples
  (the shared 7GB VM OOMs on full-panel rounds; fits were always
  ≤200–400k/era-capped, samples are seeded and rate-preserving)
- `experiments/exp84_loan_level_spread.py` — staged rounds r1/oos/r2/
  r3a/r3b/r4/r5, results merged into `exp84_results.json`
- `experiments/exp84_inventory.json` — per-vintage inventory
- `tests/test_exp84_panel.py` — 3 tests (date arithmetic, dtype
  regression), green
