# exp82: The Pooled Router — cross-series fit inside the group forecaster

**Status: COMPLETED (2026-10-20). Verdict: the pooled router does NOT
ship as the group forecaster's default center — the pre-registered bar
was missed by every design tried (six rounds, below). The trailing-mean
center (exp79) remains the production default. `pooled_router_
forecaster_fn` ships as a tested, documented research option in
`agent/group_forecaster.py` (with the `normalize_crps` variant
available), mirroring how exp79's per-series router ships. The
experiment's positive yield is diagnostic: exp79's starvation
hypothesis is rejected, and the two mechanisms that actually bind —
expert level-anchoring (M1) and scale-dominated expert selection (M2)
— are identified, measured, and partially corrected (control reaches
parity on typical cells at the best variant; the S1/shock regime and
large-group cells never close).**

## Why

exp79 settled the group forecaster's center question *per series*: the
trailing-6-mean center ships (control group CRPS 0.0899; S1@4σ 0.1193),
and the **real** `ProposedForecasterV2`, fitted per series on that
series' own rolling-origin backtest, *lost* — significantly worse than
the naive center on control (group +0.0080, p=0.012; global p=1.6e-06)
with worse coverage (cov80 0.618 vs 0.714). exp79's diagnosis was
explicit: the router was designed for record pools of hundreds across
series (exp47, where pooled fitting on M4 produced 0.1451 vs equal
weights 0.1507, p=6.5e-09); a per-series self-fit gives it at most ~13
backtest records, so regimes barely separate and its ETS/ARIMA experts
overfit 12–29 noisy points.

exp81 then proved the mirror-image result on the UQ side: pooling
conformal banks *across* series (with shrinkage) fixed exactly the
starvation that made per-series CQR a no-op on M4. The open question
exp82 asks: **does pooling the router's fit — one router fitted on
rolling-origin backtest records pooled across all series in the panel —
beat the shipped trailing-mean center inside the group forecaster?**

## Design

The pooled forecaster (`pooled_router_forecaster_fn`, shipped in
`agent/group_forecaster.py` if it wins):

- A **panel** (group_id -> full history, including `__portfolio__`) is
  given to the factory once. At a call with a history of length L
  (the harness's rolling origin), every panel series is truncated to
  L and rolling-origin backtest records are built from its prefixes
  t = 8..L-1 exactly as in exp79's per-series protocol (10 canonical
  `series_features`, experts {naive, ets, arima} on the 9-level grid,
  per-expert pinball CRPS vs the realized next value). **One**
  `ProposedForecasterV2` (generic calm/stressed spec, exp79's frozen
  spec choice) is fitted on the pooled records (~8x the per-series
  record count) and cached per L.
- Prediction: the fitted router combines the input series' expert
  quantiles; the combined **median is the center**, and exp78's
  Gaussian spread (trailing-24 naive one-step errors of the input
  series) is applied on top — the identical spread machinery arms A/B
  share, so A vs B isolates the center, as exp79's A-vs-B did.
- PIT safety: records at origin L use only prefixes `s[:t]`, t < L,
  of series truncated at L. Pinned by a prefix-invariance regression
  test (perturbing panel tails beyond L cannot change the output).
- No silent degradation: no `try/except` on the fit/predict path; an
  empty pool or un-fittable router raises (exp78 bug class).

## Arms (frozen)

Harness = exp79's (SimulatorV2, exp72 protocol, seeds 7/42/123,
`min_group_n=30`, worlds control + S1@4σ, group + global scopes,
member-weighted pooled CRPS, cov80/cov90, paired t + win rate on
seed x origin x group cells).

| Arm | Definition |
|---|---|
| A — TRAIL (champion) | Shipped default: trailing-6-mean center + Gaussian spread (`v2_forecaster_fn()`). exp79: control 0.0899, S1@4σ 0.1193. |
| B — POOL | Pooled router center + identical Gaussian spread (`pooled_router_forecaster_fn`). |
| B0 — POOL-OWN (diagnostic) | Pooled router's own combined quantiles, no exp78 spread. Deconfounds pooling from spread-stacking: B0 vs D isolates the pooled fit under the router's own interval machinery; B vs B0 isolates the spread choice. Not a ship candidate. |
| C — POOL+L2 | Arm B's grids gated by exp73's Layer-2 `MetaRouter` per series over the origin sequence: candidates {trail = arm A's grids (abstention baseline), pool = arm B's grids}; config = exp72's on this harness (val_window=12, min_val=6, margin=0.02, no lock-in). Do-no-harm by construction: the gate must see pool beat trail on trailing validation origins by margin + noise floor, else it abstains to trail. |
| D — PERSERIES (reference) | exp79's arm C: `router_forecaster_fn` fitted per series. Reference only; expected to reproduce exp79 (control ~0.0925). |

## Decision rule (frozen)

Primary endpoint: member-weighted pooled CRPS, group scope. An arm
(B or C) **ships** iff, on protocol seeds, it:

1. beats A on pooled CRPS in **both** worlds (control and S1@4σ),
   group scope, at paired p < 0.05; and
2. has cov80 **no worse than A's** in both worlds, group scope; and
3. **replicates on fresh seeds 2024/555/9001** (same two conditions).

If C ships, it ships as the gated configuration; if B ships, plain.
Shipping form: `pooled_router_forecaster_fn` becomes the documented
production default for panel forecasting in `group_forecaster.py`
(the stateless `v2_forecaster_fn` trailing-mean center remains the
default for single-series callers, since the pooled fn requires a
panel), with all pre-exp82 behaviors reproducible via the existing
flags, plus anti-regression tests (fit-spy, failure-raises,
prefix-invariance) in the exp79 style.

If no arm clears the bar, **nothing is killed**: the iteration log
records each round's diagnosis and the design variants tried
(researcher-mindset directive), and the report closes with what the
pooled fit did and did not fix.

## Iteration plan (pre-committed variants, run as evidence dictates)

- R2a: regime-features-only router — fit the pooled logistic router on
  the feature subset the generic spec's label rule actually reads
  (plus level/vol context), reducing LR overfit on ~100 records.
- R2b: shrinkage center — center = w * router_median + (1 - w) *
  trailing_mean, w in {0.25, 0.5, 0.75}.
- R2c: backtest-pool size — backtest_start in {6, 10} (more/fewer
  pooled records per series).
- R3: partial pooling — pooled records + (k-1) extra copies of the
  target series' own records (k=3), i.e. shrinkage between the pooled
  and per-series fits at the data level.
- R4: hard pooled — pooled router's argmax-regime expert only (no
  soft Vincentization).

## Results

### R1 — the pre-registered design (protocol seeds 7/42/123)

Member-weighted pooled CRPS, group scope (primary endpoint);
paired = seed x origin x group cells (n=207) vs arm A.

| World | A TRAIL | B POOL | B vs A (paired) | B0 POOL-OWN | C POOL+L2 | C vs A | D PERSERIES | D vs A |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.0885 (cov80 0.762) | 0.0952 (0.713) | **+0.0060, p=0.0052, win 0.37** | 0.0949 (0.653) | 0.0878 (0.762) | −0.0001, p=0.687 | 0.0906 (0.638) | +0.0106, p=4.1e-04 |
| S1@4σ | 0.1176 (cov80 0.731) | 0.1295 (0.662) | **+0.0131, p=1.1e-05, win 0.34** | 0.1339 (0.543) | 0.1173 (0.731) | −0.0001, p=0.615 | 0.1255 (0.585) | +0.0158, p=2.1e-05 |

Global scope, same direction: control B 0.0861 vs A 0.0846
(+0.0045, p=0.0094), D 0.0891; S1 B 0.1283 vs A 0.1219 (+0.0132,
p=8.4e-06), D 0.1227. **B vs D is split**: pooling wins the typical
cell (control paired −0.0046, p=0.030, win 0.57; global −0.0098,
p=1.4e-06, win 0.68; S1 −0.0027, p=0.31 n.s.) and coverage
everywhere (control cov80 0.713 vs D's 0.638), but loses the
member-weighted pooled CRPS on control (0.0952 vs 0.0906) through a
few large-group cells. D replicates exp79's arm C (0.0925 / 0.1285
there) up to small harness drift. A replicates exp79 (0.0885 vs
exp79's 0.0899 control — same harness, minor drift via arm set).
The production-equivalence check passed in-run: arm B as executed
*is* `pooled_router_forecaster_fn`.

**The bar is missed decisively.** B is significantly *worse* than A
in both worlds on CRPS *and* coverage. Pooling ~8x the records did
not rescue the router: B's deficit (+0.0060 / +0.0131) is the same
order as exp79's per-series deficit (+0.0080 control). **The
starvation hypothesis is rejected as the main explanation.** Two
secondary findings:

- The exp78 spread genuinely helps the pooled center: B beats B0 in
  both worlds (control −0.0022, p=0.0093; S1 −0.0042, p=2.5e-04) —
  spread-stacking is worth ~0.002–0.004 CRPS; the *center* is what
  loses.
- Arm C (Layer-2 gate) is a tie with A, not a win: across all six
  combos the gate picked "pool" in 29 of 828 gated decisions (3.5%,
  plus 10 blends) — cold-start + margin abstention dominate, as
  designed. Do-no-harm holds trivially; there is nothing to ship.

### Diagnosis (seed 42, both worlds; `exp82_diag.py`)

1. **Center level bias (M1).** The pooled router's median is
   systematically *above* the realized next value: bias +0.0135
   (control) / +0.0116 (S1) vs the trailing mean's −0.0028 / −0.0022.
   Center MAE: router 0.0977 / 0.0924 vs trail 0.0842 / 0.0874;
   router center wins only 32–37% of cells. The ETS/ARIMA experts
   anchor on the full prefix's level; the trailing-6 mean anchors on
   the recent level, and on these series the recent level is what
   the next value resembles.
2. **Scale-dominated expert selection (M2).** The router picks the
   per-regime expert by *mean raw CRPS* over pooled records. In the
   stressed regime the raw winner is ETS (0.0918 vs naive 0.0983 on
   control), but per-record CRPS normalized by the naive expert's
   CRPS flips the winner to **naive by ~2x** (ETS 1.997, ARIMA 1.718
   relative): a few large-scale series dominate the raw mean and
   invert the choice — in exactly the regime that decides S1.
   Per-series winners are heterogeneous (naive takes calm in 3 of 6
   series), so no single pooled expert is right everywhere either.
   In the calm regime raw and scale-free nearly agree (all experts
   within ~1% of naive) — selection there is close to arbitrary.

### Iteration rounds (all on protocol seeds; group scope)

Paired diff vs A (n=207 cells) / member-weighted CRPS / cov80.

| Round | Design | control: B diff (p) · mw · cov80 | S1@4σ: B diff (p) · mw · cov80 |
|---|---|---|---|
| R1 | plain pooled fit | +0.0060 (0.0052) · 0.0952 · 0.713 | +0.0131 (1.1e-05) · 0.1295 · 0.662 |
| R2e | scale-free fit (`normalize_crps`) | +0.0023 (0.427 n.s.) · 0.1012 · 0.717 | +0.0155 (1.8e-04) · 0.1401 · 0.613 |
| R2b | shrink w=0.5 toward trailing mean | +0.0022 (0.031) · 0.0902 · 0.741 | +0.0059 (7.4e-05) · 0.1222 · 0.669 |
| R3 | partial pooling (target records x3) | +0.0077 (2.9e-04) · 0.0968 · 0.727 | +0.0124 (1.0e-05) · 0.1259 · 0.731 |
| R4 | normalize + shrink 0.5 | **−0.0000 (0.98) · 0.0921 · 0.731** | +0.0052 (0.018) · 0.1231 · 0.680 |
| R5 | shrink w=0.75 | +0.0008 (0.094 n.s.) · 0.0888 · 0.756 | +0.0027 (2.6e-04) · 0.1195 · 0.712 |

(A: control mw 0.0885, cov80 0.762; S1 mw 0.1176, cov80 0.731.)

Reading the rounds:

- **R2e** removes the control deficit's significance (M2 correction
  works where the world is stationary) but *worsens* S1: the
  scale-free stressed-regime winner is the naive expert, whose
  backtest advantage in volatile-but-stationary "stressed" records
  does not transfer to a level-shift regime — backtest-stressed ≠
  shock-stressed. Its control member-weighted CRPS stays poor
  (0.1012): typical cells are fixed, large-group cells are not.
- **R2b/R5** give a clean dose-response: the deficit shrinks
  monotonically toward A as w grows (control +0.0060 → +0.0022 →
  +0.0008; S1 +0.0131 → +0.0059 → +0.0027) and never crosses zero —
  the router center's marginal contribution on this harness is
  negative at every dose; the curve is the bias (M1) being diluted,
  not signal being added.
- **R3** interpolates between the pooled and per-series fits and
  inherits both failures (worse than R1 on control).
- **R4** (both corrections) is the best design: exact parity with A
  on control's typical cells (win 0.49) and global scope — but the
  bar still fails twice over: member-weighted control CRPS stays
  worse (0.0921 vs 0.0885, the large-group cells again), S1 stays
  +0.0052 (p=0.018), and cov80 is below A's in both worlds (0.731 /
  0.680 vs 0.762 / 0.731). No round reaches the fresh-seed
  replication step, which the pre-registration reserves for a
  bar-clearing arm.

**Ship decision (per the frozen decision rule): nothing ships as
the default.** The trailing-mean center remains the group
forecaster's production center. What exp82 ships instead: the
pooled forecaster as a tested option in `group_forecaster.py`
(`pooled_router_forecaster_fn`, incl. `normalize_crps`), the
diagnosis (M1/M2) with its corrections measured, and this negative
result with the same standing as exp79's: on SimulatorV2 group
series, *no* router fit — per-series, pooled, scale-free, shrunk,
or partially pooled — beats a trailing-6 mean as the center. The
contrast with exp47 (where the same router wins on M4) localizes
the router's value to long, heterogeneous benchmark panels, not to
short simulator group series whose next value is dominated by the
recent level.

## Iteration log

- **R1 (pre-registered design): MISS** (table above). Pooling alone
  does not fix the router inside the group forecaster. Diagnosis
  (above) identifies two mechanisms, M1 (center level bias) and M2
  (scale-dominated expert selection), neither of which is record
  starvation.
- **R2e — scale-free fit** (targets M2): records' CRPS divided by
  the record's naive CRPS before fitting (`normalize_crps=True`;
  spy-tested). **Partial**: control deficit loses significance;
  S1 worsens (naive expert's backtest win does not transfer to the
  shift regime). Kept as an option in the shipped factory.
- **R2b — shrinkage center** (targets M1): w=0.5 halves both
  deficits, still significant. Dose-response completed at **R5**
  (w=0.75): control n.s., S1 +0.0027 still significant; monotone
  convergence to A from above.
- **R3 — partial pooling** (`target_weight=3`): worse than R1 on
  control; interpolates between two losing fits. Closed as a
  direction within exp82 (measured, not assumed).
- **R4 — normalize + shrink 0.5**: best design; control parity on
  typical cells, bar still missed (member-weighted control, S1,
  cov80). 
- **R2a / R2c / hard-routing** (feature subset / pool size /
  argmax-expert, pre-registered as R2a/R2c/R4): not run — after
  five measured rounds the binding
  constraints are M1/M2 and the large-group/shock cells, none of
  which these knobs address (regime separation is balanced per the
  diagnosis; pool size was R1's whole hypothesis). Recorded as a
  deliberate stop after 5 iteration rounds, per the
  diagnose-and-iterate directive, not a silent drop.
- Infrastructure note: the harness checkpoints rows after every
  (seed, world) combo with `--resume`; the run survived two VM
  restarts and one session kill with zero lost combos. Arm A's rows
  reproduce exactly across all six runs (control 0.0885 / S1
  0.1176) — the harness is deterministic.
