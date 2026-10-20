# exp86: Chronos-2 as the Segment Engine

**Status: PRE-REGISTERED (2026-10-20) — design and decision rule
frozen below before any exp86 run.** Results and the iteration log
are appended under "Results" as rounds complete.
**COMPLETED 2026-10-20: ❌ honest negative — no arm clears the
frozen bar; nothing ships. See Results.** (Run completed by a
recovery agent after the original run's agent died mid-analysis;
all per-cell rows and the Chronos disk cache survived and were
reused — no forecast was recomputed for the protocol seeds.)

## Why

exp67 red-teamed TSFMs on shifted SimulatorV2 *portfolio-level*
series: in absolute terms Chronos-2 ranked #1 in 6 of 8 shift cells
(S1, S2, S3, S3ctrl, S6, calm) — it degrades like everything else at
4σ, but it lands softest. Yet the production *segment* (group)
forecaster is a purely local statistical device: trailing-mean center
+ Gaussian spread (exp79) + thin/dead fallback (exp80). exp67's
Chronos evidence is about univariate series at h=6 on the aggregate;
whether a zero-shot TSFM beats the tuned local device **per segment,
at h=1, inside the production fallback semantics** has never been
tested. That is this experiment. A second question rides along:
exp81's pooled conformal banks fixed calibration starvation on
many-series benchmarks — the segment setting is also many-series
(≈8 groups per seed), so Chronos + the exp81 pooled machinery is a
natural arm.

## Harness (identical to exp79/exp80)

SimulatorV2 via `exp72.build_seed_data(seed, "tree")` /
`exp72.build_world`; origins `range(12, n-8)`; `TRAIN_WINDOW=6`;
`GroupForecaster(min_group_n=30)` with production defaults
(`dead_zero_months=2`); member weights = the control world's
`n_at_risk` at the origin (exp79's convention); scoring = pinball
CRPS on the 9-level grid interpolated from the 5-level forecast
(`exp72.interp9`) + cov80/cov90 (`exp78._coverage`); paired t-test +
win rate on seed×origin×group cells. Worlds: **control, S1@4σ
(rate shock — the universal killer), S3@4σ (regional divergence)**.
Protocol seeds **7/42/123**; any arm clearing the bar is re-checked
on fresh seeds **2024/555/9001** (the exp66g/exp79/exp80 seed-noise
guard).

## Arms (frozen)

| Arm | Definition |
|---|---|
| A — CHAMPION | Production today: `v2_forecaster_fn()` defaults (trailing-mean center, Gaussian spread, conformal off) inside `GroupForecaster` defaults. Sanity gate: must reproduce exp80's arm A (control group CRPS 0.08849 / cov80 0.7618; S1@4σ 0.11763 / 0.7312). |
| B — CHRONOS | `Chronos2Wrapper` zero-shot per segment series, h=1, native 21-level quantiles interpolated onto the production 5-level grid, clipped + monotonized. Same `GroupForecaster` wrapper — fallback semantics (thin → portfolio forecast, dead → own series) are identical; only the `forecaster_fn` changes. Fixed torch seed per call; forecasts disk-cached by exact series bytes (resume-safe). PIT: the fn sees only `series[:origin]`. |
| C — CHRONOS+POOL | Post-pass conformal on B's issued group-scope forecasts, using exp81's **shipped** score design: signed max-score `max(q10−y, y−q90)`, width-normalized by the binding side's half-width, own/cross mixture with shrinkage `λ = n_own/(n_own+n0)`, **n0=24**, min bank 8, level 0.8, weighted finite quantile (`td_one.uq.methods.weighted_quantile`). Banks pool across the segment groups within the same (seed, world): own bank = the target group's scored cases at origins `o' < o`; cross bank = other groups' cases at `o' < o` (round-lag rule at step 1, h=1: a bank case is fully realized before the target origin iff `o' < o`). Application = canonical CQR outer-pair adjustment (as exp80): `q10 −= q·(m−q10)`, `q90 += q·(q90−m)`; a negative q narrows (textbook CQR, as exp81 allows); then clip + monotone. **Global scope: unchanged from B** — the portfolio is a single aggregate series, and exp81's family gate (pooling only for ≥8-series families; OSFI contraindication) forbids pooling it against its own segments. |
| D — GATE (diagnostic) | exp73-style validation gate between A and B, computed post-pass: per (seed, world, group), walk-forward — at origin o, if the group has ≥4 scored validation cells at origins < o (same world) and mean CRPS(B) ≤ 0.98 × mean CRPS(A) over them, issue B's forecast, else A's (the incumbent wins ties/abstentions). Global scope: the same gate on the portfolio cell's history. D is a bonus arm: ship consideration only if it beats **both** A and the best Chronos arm on the primary endpoint. |

Per-cell stored artifacts: the issued 5-level quantile vectors for A
and B (group + global scope), the champion's Gaussian spread-offset
vector, fallback flags, realized y, member weight. Every post-pass
arm (C, D, and all diagnosis variants) is computed from these stored
vectors — iteration rounds cost zero additional Chronos calls.

## Pre-registered bar (frozen)

A Chronos arm (B or C) **ships** iff all three hold:

1. **Control win:** group scope, control world — pooled
   member-weighted CRPS strictly below A's, with paired t-test vs A
   at **p < 0.05**.
2. **Degradation:** its pooled ΔCRPS = CRPS(S1@4σ) − CRPS(control)
   is **smaller** than A's Δ.
3. **Replication:** on fresh seeds 2024/555/9001, the control
   contrast (arm − A, pooled) stays negative with paired p < 0.10,
   and the S1 degradation ordering (arm Δ < A Δ) holds.

Ship form: an **option** in `group_forecaster.py` behind the flag
style (`chronos_forecaster_fn`, default stays A) — unless the margin
is large (≥10% relative pooled CRPS improvement over A on **both**
control and S1@4σ, replicated), in which case a default flip is
eligible and will be argued explicitly in the verdict. If C is the
winner, its pooled-conformal adjustment ships as part of the option
(a stateful two-pass forecaster class if the plain `forecaster_fn`
contract cannot carry it — decided at ship time, with tests either
way). S3@4σ carries no bar (reported as a secondary endpoint; a
catastrophic S3 loss vs A would be flagged in the verdict).

If no arm clears the bar: honest negative, nothing ships, and the
diagnosis rounds below are still run and reported.

## Diagnosis rounds (planned, researcher rule — no kills)

- **R1 decomposition:** per-seed and per-world B-vs-A contrasts;
  center-vs-width diagnosis (cov80 + mean interval width vs A).
- **R2 center/width surgery:** BC arm = Chronos q50 as center + the
  champion's Gaussian spread offsets (post-pass); also a
  native-width-rescaled variant if R1 shows a pure width failure.
- **R3 blends:** quantile averages `(1−w)·A + w·B`,
  w ∈ {0.25, 0.5, 0.75} (post-pass), paired vs A and vs B.
- **R4 context variants:** Chronos on truncated contexts (last 12 /
  last 18 points), one extra cached pass each.
- **R5 gating deep-dive:** if B and A win in different regimes,
  examine D's per-cell choices (what the validation gate saw) for a
  PIT-honest selection signal.

## Runtime discipline

Chronos-2 weights are local (`models/weights/chronos2`, MANIFEST
status ok). Smoke test 2026-10-20: load ≈ 25 s once per process,
then 0.09–0.28 s per h=1 call on contexts of 12–21 points (CPU,
`OMP_NUM_THREADS=1` etc.). Unique (series, origin) pairs across
3 seeds × 3 worlds ≈ 650 worst case (worlds share bit-identical
pre-onset prefixes; the bytes cache dedups them) → one main pass is
minutes, and the disk cache makes every later round free. Runtime
will be reported with the results.

## Results

### Sanity gate — passed exactly

Arm A on protocol seeds reproduces exp80's arm A to the printed
digit: control group CRPS **0.08849** / cov80 **0.7618**; S1@4σ
**0.11763** / **0.7312**. The harness is faithful; every contrast
below is against the true production champion.

### Protocol seeds 7/42/123 (702 cells) — group scope

Member-weighted (mw) CRPS is the pre-registered primary endpoint;
"unw" is the unweighted paired contrast vs A (mean diff, paired
t-test, win rate).

| World | A (champion) | B Chronos | C Chronos+pool | D gate | BL25 |
|---|---|---|---|---|---|
| control mw | **0.08849** (cov80 .762) | 0.09024 (.724) | 0.09043 (.831) | 0.08727 (.776) | 0.08755 (.696) |
| S1@4σ mw | **0.11763** (.731) | 0.12382 (.667) | 0.12210 (.804) | 0.11873 (.705) | 0.11695 (.671) |
| S3@4σ mw | **0.12665** (.705) | 0.13061 (.659) | 0.12914 (.819) | 0.12599 (.664) | 0.12657 (.606) |
| control unw vs A | — | −0.00526 (p=0.026, win .57) | −0.00691 (p=0.0023, win .62) | −0.00130 (p=0.36) | −0.00180 (p=0.0030) |
| Δ S1−control (mw) | **+0.02914** | +0.03358 | +0.03167 | +0.03146 | +0.02940 |

The central tension: **Chronos wins the typical cell but loses the
book.** B and C beat A on the unweighted paired contrast (C at
p=0.0023) yet lose the member-weighted endpoint in every world,
because member weight is extremely concentrated — the top weight
tercile of groups holds **83%** of total weight, and there Chronos
loses (see R1). Global scope is the mirror image: B ties A on
control (0.03624 vs 0.03583) and the portfolio is where Chronos's
exp67 strength lives.

**Bar check (frozen):** B fails criterion 1 (mw 0.09024 > A
0.08849) and criterion 2 (Δ +0.03358 > A +0.02914). C fails both
(0.09043; +0.03167). **Neither Chronos arm ships.** The fresh-seed
replication was still run, as diagnosis (below).

### Fresh seeds 2024/555/9001 (729 cells) — replication as diagnosis

| World | A | B | C | D | BL25 |
|---|---|---|---|---|---|
| control mw | 0.08657 (cov80 .767) | 0.08782 (.734) | 0.08724 (.819) | **0.08561** (.768) | 0.08615 (.697) |
| S1@4σ mw | 0.16822 (.565) | 0.16370 (.598) | **0.15996** (.694) | 0.16297 (.540) | 0.16531 (.543) |
| S3@4σ mw | 0.12809 | 0.12127 | **0.11990** | 0.12012 | 0.12502 |
| control unw vs A | — | +0.00200 (p=0.357) | +0.00025 (p=0.904) | −0.00051 (p=0.655) | +0.00025 (p=0.653) |

Three protocol-seed patterns **do not replicate**: B's unweighted
control win flips sign (+0.00200, n.s.); BL25's significant
unweighted win vanishes (+0.00025, p=0.653); per-seed control diffs
for B range −0.0016 / −0.0008 / **+0.0084** (seed 9001). The S1
degradation ordering also flips (fresh Δ: A +0.08165 > B +0.07588
> C +0.07271 — Chronos degrades *less* here, opposite of protocol).
What **does** replicate, in both seed sets and every world: **arm
C's coverage repair** (control cov80 0.831/0.819 vs A 0.762/0.767;
S1 0.804/0.694 vs 0.731/0.565) and **Chronos's portfolio-level
strength** (fresh global control B 0.03976 vs A 0.04408; S3 global
0.06436 vs 0.07175). Bar criterion 3 would have failed anyway
(control contrast not negative on fresh seeds for B or C).

### Iteration log (researcher rule — diagnose, don't kill)

- **R1 decomposition.** Weight terciles (protocol): B/C win the
  low and mid terciles in all worlds (mw diffs −0.004..−0.010) and
  lose the high tercile (+0.003..+0.007) — the tercile holding 83%
  of weight. On fresh seeds the size pattern is **unstable** (low
  tercile flips to +0.013/+0.028 against Chronos; mid wins; high
  wins only in shifted worlds): there is no stable size rule.
  Fallback split (protocol, control) is the cleanest mechanism:
  on **fallback cells** (wrapper substitutes the portfolio series;
  125 of 207 cells) Chronos *wins* (B −0.0029, C −0.0069 mw); on
  **own-series cells** it loses (B +0.0034, C +0.0050). Per-group
  swings are large in both directions (g4: B +0.0238 mw; g7:
  −0.0165, p=0.008). Reading: Chronos reads smooth aggregate
  series better than the champion and short noisy segment series
  worse — exactly the exp67 (portfolio) vs exp86 (segment) split.
- **R2 center/width surgery.** BC (Chronos q50 center + champion
  Gaussian spread) is worse than **both** parents everywhere
  (protocol control mw 0.09347; fresh S1 unw +0.01421, p=0.0137).
  Neither Chronos's center nor its width is separately salvageable;
  the width fix that works is arm C's conformal (for coverage,
  at a small CRPS cost). Closed.
- **R3 blends.** BL25 is the best diagnosis variant on protocol
  (mw below A in all three worlds; unw p=0.0030) — and does not
  replicate on fresh seeds (above); its cov80 also degrades to
  ~0.70 (quantile averaging narrows intervals). A blend was never
  a frozen ship arm; on this evidence it does not earn a
  pre-registration of its own without a new idea attached.
- **R4 context variants.** Last-12 context improves B slightly
  (control mw 0.08889 vs full-context 0.09024) but still loses to
  A (0.08849), its unweighted win weakens (p=0.054), and S1 is
  unchanged (0.12351 vs A 0.11763). Last-18 is bracketed by the
  last-12 and full results (both fail the bar) and was not run.
  Closed: context length is not the binding constraint — at
  origins 12–21 the history simply is short.
- **R5 gating deep-dive.** The gate's picks carry real signal:
  where D picks B, B beats A (protocol control −0.0038; fresh
  S1 −0.0233, S3 −0.0230) — but it picks B on only 19–34% of
  cells (≥4 validation cells per group is a hard budget at these
  history lengths), so D's pooled gain is small and never
  significant, and D fails its ship condition (beat both A and
  the best Chronos arm). The signal is real but too sparse to
  harvest at segment scale with h=1 walk-forward validation.

### Verdict

❌ **Honest negative. Nothing ships; `group_forecaster.py` is
unchanged.** A zero-shot TSFM does not beat the tuned local
forecaster per segment at h=1 inside production fallback
semantics — on the pre-registered member-weighted endpoint it
loses on protocol seeds and ties-to-loses on fresh seeds, and its
two apparent protocol-seed wins were seed noise, caught by the
fresh-seed guard. Durable findings:

1. **TSFM strength is scale-dependent.** Chronos-2 is the best
   portfolio-level forecaster in this program (exp67; global
   scope here, both seed sets) and a losing segment-level one.
   exp67's result does not transfer down a level of aggregation.
2. **exp81's pooled conformal calibrates a TSFM too.** Arm C
   repairs Chronos's native coverage on 6/6 seed-world sets at a
   small CRPS cost — a calibration result, not a forecasting
   one, and the natural UQ pairing if a TSFM is ever deployed
   at portfolio level here.
3. **Follow-up hypothesis (untested, needs its own
   pre-registration):** selection with more validation data per
   group (longer books / pooled validation), or Chronos restricted
   to fallback (portfolio-substituted) cells, where R1 shows it
   genuinely wins. Not shipped on this evidence.

### Runtime

Protocol pass: 702 cells, 194 unique full-context Chronos calls
(rest cache hits), ≈35 min wall under multi-agent contention.
Fresh pass: 729 cells, completed after two OOM kills and a
retry-driver resume (7 GB box shared with six other experiment
tracks; the disk cache made every retry cheap). R4 (last-12):
702 cells, 185 model calls + 223 cache hits, 920 s. All post-pass
arms (C, D, BC, blends, all diagnosis) cost zero model calls.
Artifacts: `exp86_results_{main,fresh,ctx12,smoke}.json`,
`exp86_rows_{main,fresh,ctx12,smoke}.json`,
`exp86_diagnosis_{main,fresh}.json`, `exp86_diagnose.py`.
