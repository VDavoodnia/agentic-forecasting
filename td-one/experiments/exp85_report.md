# exp85: Shift-Proof Conformal Calibration — Changepoint Banks + Bias-Corrected ACI

**Status: IN PROGRESS (2026-10-20).** Pre-registration below is FROZEN
before any exp85 run. Results and the iteration log are appended as
rounds complete.

## Why

Conformal banks are structurally blind at shift onset — found twice:

- **exp74** (Chronos-2 UQ on the exp67 shift battery): at S1@4σ
  straddle origins, *every* residual-based arm under-covers (best
  cov80 **0.444** vs the 0.80 target). Mechanism: the bank is
  pre-shift, so every score in it is small; no offset computed from it
  can anticipate onset width.
- **exp80a** (CQR conformal on the group forecaster): the winning
  conditional variant cleared the bar on protocol seeds but failed
  the fresh-seed S1 replication (cov80 **0.652** vs the 0.77 floor),
  and an aggressive-tuning round (level 0.95, vol-clamp 3×) produced
  coverage identical to four decimals — bank offsets are bounded by
  the worst *pre-shock* miss. The gap is structural, not tuning.

ACI (exp74) was only a regime-gated partial win: it reacts to misses
online but adapts only the level — it cannot move the interval
center, so under a persistent level shift it must widen by ~2|bias|
to hold coverage. The UQ survey's **BC-ACI** entry (2026 preprint)
names the fix: estimate the bias online (per-horizon EWM) and correct
the scores *before* conformal/ACI, so the interval shifts instead of
only widening. exp74's in-house `aci_bias` attempt failed (food CRPS
1.50 vs native 1.30, p=7e-31; diagnosis: the EWM estimate chased the
regime, and the correction was applied to symmetric residual ACI,
not to per-side CQR scores with shrinkage).

exp85 tests the two adaptation mechanisms the predecessors lacked:

- **(a) Changepoint-triggered bank surgery.** Detect the break from
  data visible at the origin (Layer-2 `changepoint_scan` as the
  gate, `shift_guard.cusum_detect` for the onset estimate — both
  PIT-pure), then either **reset** the bank (drop pre-break scores)
  or **decay** it (exponential down-weighting by age, with an extra
  penalty factor on pre-break scores). This attacks the *dilution*
  mechanism directly: post-break scores are the only ones that know
  the new regime's error scale, and a plain quantile over a mostly
  pre-break bank dilutes them away.
- **(b) Bias-corrected conformal / BC-ACI.** Per-horizon (testbed 1)
  / per-series (testbed 2) online EWM bias of the signed forecast
  error, estimated causally (bias-before-case track), shrunk toward
  0 by bank size (`b * m/(m+n0)`) and clipped to the native
  half-width. Scores are the asymmetric CQR exceedances of the
  *bias-shifted* interval; the current interval is shifted by the
  shrunk bias and widened by the corrected scores' conformal
  quantile (BC-CQR), optionally with ACI level adaptation on top
  (BC-ACI). Differences vs exp74's failed `aci_bias`: per-side CQR
  scores (not symmetric |residuals|), causal bias track for scoring
  (exp74 used it too, but combined with an unshrunk final bias for
  centering), shrinkage + clipping of the applied bias, and ACI
  applied to corrected-score exceedances rather than raw residuals.

## Testbeds

1. **Chronos-2 UQ (exp74 harness).** Cached forecasts
   `exp74_forecasts.json` (dev seeds 0/1: S1@1σ, S1@4σ ×2 seeds,
   S2@4σ, S3@4σ, control ×2, Food CPI) + **fresh seeds 2/3**
   (`exp85_forecasts.json`: S1@4σ ×2, control ×2; same generator,
   origins 12–42 step 2, h=6, onset 30). Arms are (case, bank)
   functions in `src/td_one/uq/adaptive.py`, run through exp74's
   PIT-safe study driver. References: native, CQR (shipped default),
   ACI (regime option).
2. **Group forecaster (exp80a harness).** SimulatorV2, seeds
   7/42/123 (protocol) + 2024/555/9001 (fresh), worlds control +
   S1@4σ, `min_group_n=30`, member-weighted pooled CRPS + cov80,
   paired cells vs arm A (shipped exp79 default). The adaptive modes
   extend `v2_forecaster_fn`'s conformal layer
   (`conformal_adaptive=...`); reference arm C90 = exp80's winner
   (conditional conformal, level 0.90, default off).

## Arms

Testbed 1 (`td_one.uq.adaptive`): `cqr` (reference), `cqr_reset`,
`cqr_decay`, `cqr_cp` (decay + pre-break penalty), `bc_cqr`,
`bc_aci`, `cp_bc` (reset filter, then BC-CQR on the surviving bank).
Testbed 2 (`conformal_adaptive`): `"reset"`, `"decay"`, `"cp"`,
`"bc"`, `"cp_bc"` — same mechanisms over the per-series backtest
bank; detection runs on the series history itself.

## Pre-registered bar (FROZEN)

An arm **clears the bar** in a testbed iff, on **fresh seeds**:

- S1@4σ cov80 ≥ **0.77** (testbed 2: member-weighted pooled, the
  exact exp80a replication setup; testbed 1: pooled over all cases
  of the fresh S1@4σ configs, with the straddle slice
  27 ≤ origin ≤ 33 reported as the headline diagnostic), **and**
- control CRPS harm ≤ **+2%** vs the testbed reference (testbed 1:
  native; testbed 2: arm A), **and**
- control cov80 ∈ **[0.77, 0.85]**.

## Ship rule (FROZEN)

- **Testbed 2:** an adaptive mode clearing the bar becomes the group
  forecaster's conformal default (`conformal=True,
  conformal_adaptive=<mode>` in `v2_forecaster_fn`'s defaults);
  pre-exp85 behavior stays reproducible via `conformal=False` /
  `conformal_adaptive="none"`. If several modes clear, ship the one
  with the largest fresh-S1 cov80 margin, tie-broken by least
  control CRPS harm.
- **Testbed 1:** an arm clearing the fresh bar and not harming Food
  CPI pooled CRPS by > +2% vs CQR ships as the stack's documented
  shift-regime UQ option in `td_one.uq` (alongside ACI). If it also
  matches-or-beats CQR on pooled synth + food CRPS, the report
  recommends it as the CQR successor; the default flip itself is
  recorded as a recommendation with numbers, not done silently.
- **No kills.** Designs that miss the bar are reported with their
  measured ceiling and the mechanism that caps it (researcher rule:
  4–5 diagnosis rounds before any negative is final).

## PIT safety

Bank rule unchanged (`origin' + h' - 1 < origin`). Changepoint
detection reads only the current case's context (testbed 1:
`series[:origin]`) or the forecaster's history argument (testbed 2:
`series[:origin]` by contract) — never realized values at or after
the origin. Bias tracks are causal (bias-before-case). A
prefix-vs-full regression test pins every new arm, mirroring exp74's
`test_study_pit_prefix_equals_full`.

## Iteration log

(Appended per round as runs complete.)

### Round 1 — defaults on both testbeds (2026-10-20)

Testbed 1 (Chronos battery, `exp85_chronos_r1.json`; 17 arms,
29,274 case-results). Headline cells:

| scope | native | CQR | ACI | best adaptive |
|---|---|---|---|---|
| fresh S1@4σ pooled cov80 (n=192) | 0.833 | 0.833 | 0.812 | cp_bc / cqr_reset 0.839 |
| fresh S1@4σ pooled CRPS | 0.0533 | 0.0538 | 0.0569 | bc_cqr 0.0534 |
| fresh straddle cov80 (n=36) | 0.556 | 0.556 | 0.556 | bc_lam90 0.611 |
| dev straddle cov80 (n=36) | 0.278 | 0.389 | 0.417 | bc_aci 0.444 |
| fresh control cov80 / CRPS | 0.828 / 0.0342 | 0.833 / 0.0341 | 0.823 / 0.0348 | all BC ≈ native |
| food CRPS (n=666) | 1.3035 | 1.2824 | 1.2724 | **bc_aci 1.2106** |

Three findings drove round 2:

1. **The surgery arms were inert on this battery.** `cqr_reset`
   reproduced plain CQR *exactly* on both straddle slices, and
   `cp_bc` ≡ `bc_cqr` everywhere. Diagnosis (detection probe on the
   fresh contexts): the Layer-2 scan gate **never fires** on the
   exp74 synth series — S1 there is a 6-month *ramp* into +4σ on a
   noisy (σ≈0.35 + wave) series, and the scan's triple gate (level
   z>3 AND innovation z>3) essentially cannot trip on a ramp.
   Seed 2: no detection at any origin 24–42; seed 3: fires only from
   origin 32, onset ≈27–31. All fresh-S1 gains in round 1 came from
   the bias-correction mechanism, which needs no detector.
2. **The straddle slice is information-starved by construction.**
   Straddle origins are 28/30/32 (onset 30): at origin 28 the shift
   is entirely in the forecast period; at origin 30 the context is
   still 100% pre-shift. No causal bank mechanism can cover those
   cases — only unconditional width could, at large CRPS cost. The
   pooled fresh-S1 number (all origins) is therefore the honest
   headline; the straddle slice measures the irreducible part.
3. **Bias correction is a large, real win on Food CPI.** Every BC
   variant beat plain CQR's food CRPS (1.2824) by 4–6%: bc_aci
   1.2106, bc_lam90_n02 1.2113, bc_cqr 1.2232 — and beat exp76's
   shipped fusion champion (MetaLevelCqr, 1.2587) by ~4%. Food CPI's
   Chronos forecasts carry a persistent drift bias; exp74's
   `aci_bias` failed at exactly this because it corrected symmetric
   residual ACI with an unshrunk final-bank bias. The BC design
   (per-side CQR scores + causal bias track + shrinkage + clipping)
   fixes each of those choices. Paired arm-vs-arm tests (round-2
   analysis): bc_aci vs CQR −5.6% (p=7.1e-22, win 0.65); bc_cqr vs
   CQR −4.6% (p=5.6e-32); bc_aci vs bc_cqr −1.0% (p=9.6e-05).

### Round 2 — detectors that engage + faster bias tracks (testbed 1)

Changes: `detect_break_onset(gate="cusum")` (CUSUM alone decides;
on the fresh contexts it fires from origin 32 with onset 30/27 on
seeds 2/3 and stays silent on control — regression-tested against
exp74's exact generator); `arm_cqr_reset_pool` (width-normalized
scores pooled across horizon steps, the exp81 move, to defeat
per-h post-break bank starvation); BC grid λ ∈ {0.85, 0.90}, n0 ∈
{2, 6}, bc_aci γ=0.10 and λ=0.90 variants. `exp85_chronos_r2.json`
(28 arms, 46,494 case-results).

- **Fresh S1 pooled:** bc_lam90_n02 has the best CRPS of all 28
  arms (0.0529 vs native 0.0533, CQR 0.0538; paired vs CQR −1.6%,
  p=2.9e-04) at cov80 0.828. All BC variants clear the bar cell
  (cov80 0.828–0.839 ≥ 0.77; control cov80 0.828–0.839 ∈
  [0.77, 0.85]; control CRPS within ±0.1% of native).
- **Fresh straddle:** BC ceiling 0.611 (λ≤0.90 variants) vs 0.556
  native/CQR — a real but bounded gain, consistent with the
  information-starvation diagnosis: origins 28/30 cannot adapt,
  origin 32 partially can.
- **Dev straddle:** bc_aci_lam90 0.472 and bc_lam90_n02 0.472 —
  the first arms to exceed exp74's 0.444 wall on the dev seeds,
  by +0.028 (one case in 36; not a breakthrough, a crack).
- **Pooled reset is a dead end on this battery (honest negative).**
  On food it is *worse* than per-h reset (1.294–1.303 vs 1.254);
  on dev straddle it never engages (wr = 1.000). Cross-h pooling
  dilutes horizon-specific scale structure — exp81's pooling
  worked across *series at fixed h* with signed scores, which is
  a different object. Recorded; not pursued further.
- **Food:** bc_aci_g10 1.2062 (cov80 0.856) is the single best
  food number in the UQ program (vs CQR 1.2824, vs exp76 champion
  1.2587); γ=0.10 vs 0.05 is only −0.4% (p=0.029), so the shipped
  default stays γ=0.05.

**Testbed-1 verdict (pre-registered rule):** bc_aci and bc_cqr
clear the fresh bar with food CRPS *better* than CQR by 4.6–5.6%
(paired p ≤ 7e-22) → the BC family ships as the stack's documented
shift-regime UQ option in `td_one.uq.adaptive`, recommended over
plain CQR wherever a bias track is meaningful (persistent-drift
series). The straddle ceiling (~0.61 fresh / ~0.47 dev) is
documented as information-theoretic for origins whose context
contains no post-shift observation, not as a tuning failure.

