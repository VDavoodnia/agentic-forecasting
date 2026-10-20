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

### Round 1 — group forecaster, defaults (testbed 2, FRESH seeds)

Harness fidelity first: arm A reproduces exp80's fresh-seed base
exactly (S1 group cov80 **0.565**; exp80: 0.5652) and C90
reproduces exp80's failed-bar number exactly (**0.652**). The
exp85 arms at their defaults (level 0.80, no conditional scaling,
scan detector):

| fresh, group scope | A (ref) | C90 | RST | DEC | CP | BC | CPBC |
|---|---|---|---|---|---|---|---|
| S1@4σ cov80 | 0.565 | 0.652 | 0.594 | 0.594 | 0.594 | 0.582 | 0.582 |
| S1@4σ CRPS | 0.16822 | 0.16283 | 0.16589 | 0.16610 | 0.16610 | 0.16622 | 0.16622 |
| control cov80 | 0.767 | 0.824 | 0.796 | 0.781 | 0.781 | 0.770 | 0.770 |
| control CRPS | 0.08657 | 0.08655 | 0.08629 | 0.08647 | 0.08647 | 0.08637 | 0.08637 |

No arm clears the 0.77 bar; the adaptive arms sit between A and
C90. The identities CP ≡ DEC and CPBC ≡ BC (every digit) say the
changepoint machinery never engaged — the same signature as
testbed 1 round 1, but the mechanism is different (round 4).

### Round 2 — group forecaster, level/conditional matched to C90

Adaptive arms at level 0.90 + conditional vol scaling (C90's own
settings, isolating the adaptive contribution): **RST ≡ C90
exactly** (the reset arm degenerates to plain conditional CQR —
detector still silent), DEC = C90 on S1 cov80 (0.652) but worse on
control (0.804 vs 0.824), BC slightly *below* C90 on S1 (0.639).
At matched level/conditional, no adaptive mechanism adds anything
on this testbed yet — because none of them are actually running.

### Round 3 — group forecaster, CUSUM detector

`--detector cusum` at level 0.90 + conditional: results **identical
to round 2 in every digit**. A detector probe
(`exp85_probe_detect.py`) on the real harness series settles why:
no detector — scan, CUSUM at h=5, even CUSUM at h=3 — fires on any
group or portfolio series at any post-onset origin. Two stacked
causes, both structural:

1. **Length starvation.** The exp72/80 worlds are 29 months long;
   forecast histories are 12–21 points. `detect_break_onset`'s
   length guard (baseline 24 + 3) returns None *unconditionally*
   at those lengths. The surgery arms were never eligible to run
   on testbed 2 — rounds 1–3 measured their fallback paths only.
2. **Quantization noise.** The group series are small-n binomial
   rates (0, 0.25, 0, 0.2, …): baseline MAD-of-diffs is as large
   as the shift itself, so even a length-fitted series-level
   detector has per-step SNR ≈ 1.

### Round 4 — errors-surface detection + short-half-life decay (testbed 2)

Design change (the diagnosis applied): detect on the bank's
**signed-error sequence** (`error_onset_index`: CUSUM over the
chronological one-step errors the conformal bank already computes;
the forecaster's center lags the new level after a break, so
post-break errors are persistently one-signed even when the raw
series is too quantized to read). Plus: min_post 5 → 2 (post-onset
origins in this harness have at most ~2 post-break bank cases —
a count gate of 5 is a permanent disable), decay half-life 6 → 3.
Series-level gates in the group path also got length-fitted
windows so the length guard can no longer silently disable them.

**Infra note (round 4).** The heavy harness rebuilds the
simulator per process (peak RSS ~2.2 GB) and the shared box was
OOM-starved for hours by sibling jobs. Fix: `exp85_build_worlds.py`
pickles the built worlds once (48 KB total; local artifacts —
regenerate per seed with the builder, ~1 min each on an unloaded
machine);
`exp85_group_light.py` replays exp80's forecast loop verbatim on
the pickles (same exp72/exp78 functions, ~3 s per unit).
Fidelity check: on fresh seed 2024 the light rows reproduce the
heavy rows **bit-identically for arms A/C90/DEC/BC/RST/CPBC**;
only CP differs, and only because round 4's length-fitted window
intentionally changes CP's detector input (in round 1's heavy run
CP's detector was length-starved to None; now it can fire).

Results (fresh, level 0.90 + conditional, detector=errors,
hl 3, min_post 2; S1 group scope):

| arm | A | C90 | RST | DEC | CP | BC | CPBC |
|---|---|---|---|---|---|---|---|
| S1 cov80 | 0.565 | 0.652 | **0.658** | 0.652 | 0.652 | 0.639 | 0.646 |
| S1 CRPS | 0.16822 | 0.16283 | 0.16289 | 0.16296 | 0.16296 | 0.16297 | 0.16289 |
| control cov80 | 0.767 | 0.824 | 0.824 | 0.789 | 0.789 | 0.798 | 0.798 |
| control CRPS | 0.08657 | 0.08655 | 0.08655 | 0.08706 | 0.08706 | 0.08659 | 0.08655 |

The errors-surface reset now engages (RST edges C90, +0.006) —
but the bar (0.77) is nowhere in reach. The scan-detector variant
of the same config is identical (RST 0.658): with length-fitted
windows both surfaces fire, and the binding constraint is
downstream of detection entirely.

**Per-origin decomposition (the ceiling, measured).**
Member-weighted cov80 by origin, fresh S1, group scope:

| origin | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| A | .89 | .99 | .86 | .48 | .50 | .34 | .35 | .09 | .56 |
| C90 | .89 | .99 | .86 | .48 | .50 | .44 | .51 | .49 | .72 |
| RST | .89 | .99 | .86 | .48 | .50 | .44 | .51 | .57 | .72 |
| BC | .89 | .99 | .86 | .48 | .43 | .44 | .46 | .49 | .72 |

Three structural facts, in order of damage:

1. **Origins 12–15 are conformal-free by construction** (bank
   minimum 8 cases from t=8 → first engagement at origin 16),
   and the S1 ramp already bites at origins 15–17 (coverage
   0.34–0.50 for every arm). No bank method can act there.
2. **Origins 16–18 have banks with ≤1 post-shift case** — the
   scores that exist are pre-shift, and offsets are bounded by
   them (the original wall, now localized to three origins).
   The conditional vol-scaling is the only lever that responds
   (C90 vs A at o17: +0.10, o18: +0.16) because trailing vol
   rises as the ramp enters the history.
3. **Where post-shift evidence exists, the reset works**:
   origin 19 coverage 0.09 (A) → 0.49 (C90) → **0.57 (RST)**;
   origin 20 reaches 0.72 for all conformal arms.

Roughly 55% of the pooled shortfall vs 0.77 sits at origins
15–17, where a causal bank is empty of post-shift evidence by
construction. This is the same information-theoretic wall as
testbed 1's straddle slice, but on this harness it covers a
third of all origins instead of three cases in 36.

**Protocol cross-check (same config):** S1 group cov80 — A 0.731,
C90 0.776, RST 0.776, BC 0.756. Protocol worlds are gentler (the
0.77 line is crossed there by plain C90); fresh seeds remain the
binding set, and RST adds +0.006 fresh / +0.000 protocol over C90.
BC is *worse* than C90 on both seed sets here (fresh 0.639;
protocol CRPS harm vs A significant, p=0.013): the group
forecaster's trailing-mean center re-centers within a month or
two, so its errors carry no persistent bias for a BC track to
correct — the exact opposite of Food CPI's Chronos bias, which
is why BC wins on testbed 1 and not here.

### Round 5 — aggression sweep (testbed 2): saturation

Two maximal-aggression variants (level 0.95; hl 2 + min_post 1):
pooled numbers move by ≤0.001 and **row-level diffs vs round 4
are 1–2 cells out of 1008**. The conformal output on banks of
4–12 scores with a large atom at 0 is effectively binary — the
finite-sample quantile at any level in [0.80, 0.95] is the same
order statistic (usually the max score), decay weights cannot
move a weighted quantile off that atom, and count gates of 1
vs 2 select the same post sets. Tuning has no travel left on
this testbed; the ceiling is the bank's information content,
not its hyperparameters. (Same saturation exp80 hit in its
rounds 2/4, now confirmed for the adaptive variants.)

### Testbed-2 verdict (pre-registered rule)

**Bar NOT cleared on fresh seeds by any arm in five rounds.**
Best adaptive arm: errors-surface reset, S1 cov80 **0.658**
(C90 0.652, A 0.565); control constraints satisfied everywhere
(cov80 0.824 ∈ [0.77, 0.85], CRPS harm ≈ 0). Per the frozen ship
rule there is **no default flip**: the group forecaster keeps
`conformal=False` by default (exp80's verdict stands), with the
adaptive machinery shipped as documented options
(`conformal_adaptive` ∈ {reset, decay, cp, bc, cp_bc},
`conformal_detector` ∈ {scan, cusum, errors}). What five rounds
bought: the detector stack now actually runs on short series
(length-fitted windows + errors surface), the reset is worth
+0.006 pooled and +0.08 at the worst origin when evidence
exists, and the residual gap is attributed — origins 15–17
need a mechanism that widens *without* bank evidence of the
shift (exogenous triggers, e.g. the Macro Signal Service's
risk channel, are the named successor direction; untested here).

## Ship summary

| testbed | ships | default change |
|---|---|---|
| 1 (Chronos-2 UQ) | `td_one.uq.adaptive` BC family (bc_aci primary, bc_cqr, bc variants) as the documented shift-regime UQ option; beats CQR on food CRPS by 4.6–5.6% (p ≤ 7e-22) and clears the fresh bar | none (option alongside CQR; CQR stays the general default) |
| 2 (group forecaster) | adaptive conformal modes + errors detector as options in `v2_forecaster_fn`; reset is the best adaptive arm | none (bar not cleared; conformal stays opt-in per exp80) |

Tests: `tests/test_uq_adaptive.py` (19) +
`tests/test_group_forecaster_exp85.py` (7) cover detection on
both surfaces, reset/decay/BC mechanics, detector plumbing,
invalid-parameter rejection, and prefix-vs-full PIT safety on
both testbeds' code paths.


