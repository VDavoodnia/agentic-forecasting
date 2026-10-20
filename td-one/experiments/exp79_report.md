# exp79: Group-Forecaster Center — naive vs trailing-mean vs the real v2 router vs M0

**Status: COMPLETE (2026-10-20).** Pre-registered below before any
exp79 run; decision rule and arm definitions were frozen and are
unchanged. **Verdict: trailing-mean center ships as the production
default; the real v2 router was evaluated honestly and loses — it does
not ship as the default.** Results under "Results".

## Why

exp78 root-caused and fixed the production group forecaster's
point-mass defect: `v2_forecaster_fn` built a `ProposedForecasterV2`,
never called `fit()`, `predict()` asserted, a blanket `except`
swallowed it, and 100% of calls returned the last value tiled across
quantile levels (coverage 0.000). exp78 shipped the minimal honest fix —
naive last-value center + Gaussian spread from trailing one-step errors
(control CRPS 0.1320 → 0.1042, p=4.2e-45) — and left two questions open:

1. exp78's reference arm **G5** (trailing-6-mean center + Gaussian
   spread) beat the shipped fix on CRPS in both worlds (control 0.0872
   vs 0.1042). Its edge was attributed to the *center*, not the spread.
   Is the trailing-mean center genuinely better in production?
2. The router whose absence caused the defect was never actually
   evaluated inside the group forecaster. Does the **real**
   `ProposedForecasterV2` — fitted properly per series — beat both
   naive-center variants?

## Arms (frozen)

All arms share exp78's harness (SimulatorV2, exp72 protocol, seeds
7/42/123, `min_group_n=30`, worlds control + S1@4σ) and, where a spread
is applied, exp78's exact Gaussian machinery (`_spread_offsets`,
trailing-24 naive one-step errors).

| Arm | Definition |
|---|---|
| A — GAU (baseline) | exp78 shipped default: naive last-value center + Gaussian spread (`v2_forecaster_fn()`). |
| B — TRAIL | Trailing-6-mean center + the *identical* Gaussian spread as A (`v2_forecaster_fn(center="trailing_mean")`). A vs B isolates the center: nothing else differs. |
| B′ — G5X (reference) | exp72's `gauss5_forecaster_fn` verbatim (trailing-6 mean + sd of level-residuals). Connects exp79 to exp78's published G5 numbers; not itself a production candidate (its spread differs from exp78's). |
| C — ROUTER | The real `ProposedForecasterV2` (`router_forecaster_fn`, generic calm/stressed regime spec — see below), fitted **per series** on rolling-origin backtest records built from that series' own history, experts {naive, ets, arima} (the expert set named in the original pre-exp78 production docstring). Output = the router's own conformal-calibrated combined quantiles, interpolated from its 9-level grid (LEVELS 0.1–0.9) onto the production 5-level grid. **No exp78 spread is added on top**: the router's conformal layer is its own interval machinery; stacking both would double-count width. |
| C-M4 — ROUTER-M4 (diagnostic) | Same as C but with the `m4` regime spec (the `dataset=None` default). Diagnostic only: on 12–29-point monthly series the m4 spec's STL-based strengths are degenerate (STL needs ≥ 24 points), so C-M4 is expected to collapse toward a single expert; it is run to document that, not as a candidate. |
| D — M0 (reference) | The project's M0 baseline (`models/baseline.HistoricalBaseline`, Jeffreys Beta-Binomial, trailing-6 counts): posterior predictive rate quantiles at the group's last-month exposure, used as-is (M0's own spread mechanism). Reference arm: it needs counts, which the `forecaster_fn` contract does not carry, so it is harness-wired, not a production candidate. |

**Arm C spec choice (pre-registered justification).** The original
production call was `ProposedForecasterV2(h, dataset=None)`, i.e. the
m4 spec. But group attrition series here are 12–29 points; the m4
spec's regime features (`seasonal_strength`, `trend_strength`) come
from an STL decomposition that needs ≥ 2×12 points and is meaningless
below that, so the m4 router cannot express its design on this data.
The generic calm/stressed spec is the same production method's SMM
configuration and labels from changepoint/volatility features that are
well-defined at these lengths. C therefore uses the generic spec as the
primary router arm; C-M4 documents the dataset=None default's behavior.

**Arm C fitting protocol (frozen).** For a history `s` of length L:
backtest origins t = 8..L−1; each record = (10 canonical
`series_features` of `s[:t]`, per-expert (9,1) quantiles from `s[:t]`,
per-expert pinball CRPS vs `s[t]`, actual `s[t]`). Experts: naive =
last value + Gaussian diff-spread on the 9-level grid; ets =
statsmodels damped-trend ETS, quantiles = forecast ± z·(in-sample
residual sd); arima = statsmodels ARIMA(1,1,1), same construction.
Then `fit(records, FEATURE_NAMES)` and `predict(s, expert quantiles on
full s, features of full s)` with gate inputs at their defaults
(t_stat=0, shift=0 — the do-nothing gate's designed defaults).
**No blanket except anywhere in this path: any failure raises.**
Histories shorter than 9 points raise `ValueError` from the router fn
(callers must use a center-based fn there; the harness never does
this — origins start at 12).

## Decision rule (frozen)

Primary endpoint: member-weighted pooled CRPS, group scope, control
world; paired t-test + win rate on seed×origin×group cells vs A.
Guardrail: the winner must not lose cov80 on S1@4σ vs A by more than
0.05 (absolute) — a CRPS win bought by collapsing intervals under the
universal killer does not ship.

- The arm with the lowest control group-scope CRPS that clears the
  guardrail **and** beats A at paired p < 0.05 becomes the production
  default in `group_forecaster.py`, behind exp78's flag style
  (`center="trailing_mean"` / default flip / `router_forecaster_fn`
  wired as the default fn factory), with all previous behaviors still
  reproducible via flags.
- If ROUTER (C) wins, it ships **only** with regression tests proving
  `fit()` is actually called on the router path and that a router-path
  failure raises instead of silently degrading (the exp78 bug class
  must be impossible to reintroduce silently).
- If no arm beats A at p < 0.05, A stays the default and the verdict
  is an honest negative for both open questions.
- The exp66c comparability cell (seed 42, exp66c's own window-mean
  scoring) is re-run for the winning arm and reported next to exp78's
  OLD/NEW cells.

## Results

**Run:** 2026-10-20, `experiments/exp79_group_forecaster_center.py`,
deterministic (two full runs, identical numbers). Sanity: A and G5X
reproduce exp78's published cells exactly (A control group 0.1042 /
cov80 0.714; G5X 0.0872), so the harness replication is faithful.

### Main run: seeds 7/42/123, min_group_n=30

Pooled = member-weighted CRPS. Paired cells = seed × origin × group
(n=207). Guardrail: winner must beat A at paired p<0.05 on control
**and** not lose >0.05 cov80 on S1@4σ.

| world/scope | A (naive c.) | B (trailing-6 c.) | C (router) | D (M0) |
|---|---|---|---|---|
| control/group | 0.1042, cov80 0.714 | **0.0899**, cov80 0.744, vs A −0.0072 **p=0.034**, win 0.49 | 0.0925, cov80 0.618, vs A +0.0080 **p=0.012 (worse)** | 0.0862, cov80 0.709, vs A p=0.30 n.s. |
| control/global | 0.0875, cov80 0.623 | 0.0846, vs A p=0.22 n.s. | 0.0891, vs A **p=1.6e-06 (worse)** | 0.0869, p=0.31 n.s. |
| S1@4σ/group | 0.1456, cov80 0.602 | **0.1193**, cov80 **0.713**, vs A −0.0243 **p=4.7e-07**, win 0.58 | 0.1285, cov80 0.565, vs A p=0.69 n.s. | 0.1214, cov80 0.550, p=1.5e-04 |
| S1@4σ/global | 0.1350, cov80 0.435 | **0.1219**, cov80 0.444, vs A −0.0190 **p=2.2e-05** | 0.1227, cov80 0.396, p=0.85 n.s. | 0.1249, p=0.0037 |

Reference/diagnostic arms: G5X (exp72's trailing-6 + level-residual
spread) 0.0872 control / 0.1182 S1; CM4 (router, m4 spec) ≡ C on S1
(0.1285) and ~C on control (0.0922), as predicted in the pre-registration.

**Decision (per the pre-registered rule): B ships.**
The trailing-mean center wins pooled CRPS on control vs A at paired
p<0.05 **and** clears the guardrail decisively — B's S1 cov80 is
*better* than A's (0.713 vs 0.602, +0.111), not worse. The default in
`v2_forecaster_fn` is now `center="trailing_mean"` (exp78's naive center
remains available as `center="naive"`; `quantile_spread=False` still
reproduces the pre-exp78 point mass byte-for-byte).

**The router loses — honest negative for question 2.** C is
*significantly worse* than A on control in both scopes (group +0.0080,
p=0.012; global p=1.6e-06) and loses to B everywhere (C−B +0.0152,
p=6e-07 group control; p=7e-09 under S1; win rate for B 0.65). Its
coverage is worse than A's (0.618 vs 0.714 control). CM4 ≈ C, consistent
with the pre-registered expectation that the m4 spec's STL features are
degenerate on 12–29-point series. Diagnosis: the router was designed
for record pools of hundreds across series (exp47); a per-series
self-fit gives at most ~13 backtest records, so regimes barely separate
and ETS/ARIMA experts overfit 12–29 noisy points. `router_forecaster_fn`
stays available (now real, fitted, and covered by anti-silent-fallback
tests) but is not the default.

**G5X nuance (reported, not shipped):** G5X's pooled control CRPS is
lower than B's (0.0872 vs 0.0899) because its spread treats heavy cells
better under member weighting — but B beats G5X on the paired cells
(mean −0.0024, p=1.7e-06, win 0.83) and B's spread was already chosen
over G5X's in exp78's ablation. Same for D (M0): lowest pooled (0.0862)
but n.s. vs A (p=0.30) and unshippable — it needs counts the
`forecaster_fn` contract does not carry. A vs B is the comparison the
protocol was built to isolate (identical spread, center-only delta),
and it settles the question.

### Diagnosis rounds (researcher mindset: the pooled win was seed-7-driven)

Round 1 (per-seed, control/group): seed 7 B−A −0.0231 (p=0.0024),
seed 42 +0.0023 (p=0.64), seed 123 +0.0002 (p=0.97). Under S1: B wins
on seeds 7 (−0.0508, p=1.2e-06) and 123 (−0.0181, p=0.013), ties 42.
Round 2 (mechanism): per-cell trailing volatility does NOT explain the
edge (Spearman ρ≈0.01, p=0.84; low- and high-vol halves both ≈−0.007
n.s.) — the honest update is that level-estimation noise reduction,
not cell volatility, is the plausible mechanism; the mechanism is
reported as unresolved rather than overclaimed. Round 3 (fresh seeds
2024/555/9001, arms A/B only — the exp66g seed-noise guard): B beats A
on control pooled −0.0171 (**p=1.6e-05**, win 0.64; all three seeds
individually p≤0.048) and on S1 pooled −0.0355 (p=1.0e-04). Across six
seeds B never loses significantly anywhere. The decision to ship B
rests on the replication, not on the pooled p=0.034 alone.

### exp66c comparability cell (winner's arm, seed 42)

`--exp66c-arm B`: A_global 0.01272, G_deterministic 0.01447,
grouping cost **+0.00176** — vs +0.00333 for the naive-center fix
(exp78 NEW) and +0.01637 pre-fix (OLD). The trailing-mean center nearly
halves the residual grouping cost on exp66c's own metric.

### What ships in `src/td_one/agent/group_forecaster.py`

1. `v2_forecaster_fn` default `center="trailing_mean"` (window 6);
   `center="naive"` reproduces exp78; `quantile_spread=False`
   reproduces the pre-exp78 point mass.
2. `router_forecaster_fn` (generic/m4 specs): the real
   `ProposedForecasterV2` fitted per series on its own rolling-origin
   backtest, no exp78 spread stacked on its conformal intervals, **no
   try/except anywhere on the router path** — failures raise. Wired in
   as an available option, not the default, per the honest negative.
3. Anti-regression tests (`tests/test_group_forecaster_center.py`):
   fit() is actually called before predict(); a fit/predict failure
   raises instead of degrading; prefix-invariance (PIT); center
   selection and grid validity; unknown center/spec raise. The exp78
   bug class (silent fallback to a point mass) is impossible to
   reintroduce without failing these tests.
4. `ruff check` + `ruff format` + strict `mypy` clean (following the
   codebase's `# type: ignore[import-untyped]` convention for
   statsmodels); 29 tests green.

