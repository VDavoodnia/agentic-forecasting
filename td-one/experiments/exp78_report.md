# exp78: Quantile-Spread Fix for the Production Group Forecaster

**Status: COMPLETE — structural defect fixed, default switched, validated.**

## The defect (flagged by exp72, root-caused here)

`agent/group_forecaster.py::v2_forecaster_fn` emitted flat point-mass
quantiles. exp72 framed it as "single expert = tiled last value"; the
deeper mechanism, verified by direct execution:

1. `_forecast` constructed a fresh `ProposedForecasterV2` and called
   `predict()` **without ever calling `fit()`**. v1's `predict` starts
   with `assert self._fitted` → `AssertionError` on every call.
2. The blanket `except Exception` swallowed it and returned
   `naive_q[0]`.
3. `naive_q = np.tile(last, (h, 5))` is itself flat across quantile
   levels — and shaped `(h, 5)`, not the router's expected 9-level
   `(9, H)` grid, so the router could not have produced spread even if
   fitted.

**Net effect: the "router" in the production group forecaster never ran
once. 100% of calls returned the last value tiled across all five
quantile levels.** Measured here (seed 7, control): OLD 80%/90%
interval coverage = **0.000** — the interval never contains a per-month
actual.

## The fix

`v2_forecaster_fn(h=1, *, quantile_spread=True, spread_method="gaussian")`:

- Center stays the naive last value (minimal change; the point forecast
  is untouched).
- Quantile offsets come from the series' own trailing naive one-step
  errors (`np.diff`, last 24) — strictly history, so PIT-safe.
  `"gaussian"`: Normal z-scores × sample sd of the errors (sd floor
  1e-4). `"empirical"`: sample quantiles of the errors.
- Output clipped to [0, 1] with monotone accumulation — always a valid
  quantile vector.
- The dead router call is removed (documented in the docstring);
  `quantile_spread=False` reproduces the exact pre-fix point-mass
  output byte-for-byte for reproducibility of exp66b/c/d and exp72
  (their scripts pass no flag, so re-running them now uses the fix;
  pass `quantile_spread=False` to reproduce their published numbers).
- Fallback logic (`min_group_n` → portfolio forecast) untouched; the
  portfolio forecast is spread by the same code path.

Method choice (Gaussian over empirical) was decided by ablation, not
preference: Gaussian beat empirical on CRPS and coverage in **both**
scopes and **both** worlds (below). The empirical quantiles of ~24
trailing diffs are too noisy at the 0.1/0.9 tails.

## Validation (SimulatorV2, exp72 harness, seeds 7/42/123, min_group_n=30)

Member-weighted pooled CRPS; paired t / win rate on 207 seed×origin×group
cells per world; coverage of the [q10, q90] interval (cov90 via
constant-slope extrapolation to 0.05/0.95).

### Control

| Arm | Group CRPS | cov80 | cov90 | vs OLD |
|---|---|---|---|---|
| OLD (point mass) | 0.1320 | 0.075 | 0.075 | — |
| EMP (spread, empirical) | 0.1125 | 0.617 | 0.694 | −0.0222, p=3.1e-45, win 0.89 |
| **GAU (spread, Gaussian) — shipped default** | **0.1042** | **0.714** | **0.726** | **−0.0287, p=4.2e-45, win 0.89** |
| G5 (exp72 trailing-mean Gaussian, reference) | 0.0872 | 0.614 | 0.642 | −0.0334, p=1.5e-13, win 0.67 |

Global scope (portfolio forecast applied to every group): OLD 0.1057
(cov80 0.053) → GAU 0.0875 (−0.0266, p=2.6e-67, win 0.92), EMP 0.0899,
G5 0.0858.

### S1 rate shock @ 4σ

| Arm | Group CRPS | cov80 | cov90 | vs OLD |
|---|---|---|---|---|
| OLD | 0.1841 | 0.092 | 0.092 | — |
| EMP | 0.1560 | 0.513 | 0.602 | −0.0233, p=2.5e-31, win 0.87 |
| **GAU** | **0.1456** | **0.602** | **0.673** | **−0.0326, p=1.5e-36, win 0.88** |
| G5 | 0.1182 | 0.517 | 0.604 | −0.0547, p=1e-16, win 0.69 |

### Honest caveats

- **Calibration improved massively but is not nominal**: cov80 0.71
  (control) / 0.60 (S1@4σ) vs the 0.80 target. Diff-based spread around
  a naive center still under-covers noisy group series. The fix removes
  the *structural* defect; full calibration is a further step
  (conformal layer, cf. exp74's CQR).
- **G5 still wins on CRPS.** Its edge over GAU is the *center*
  (trailing-6 mean vs naive last value), not the spread — a separate,
  larger change to the production forecaster's point forecast, out of
  scope for this fix and worth its own experiment.

## exp66c comparability cell (seed 42, exp66c's own window-mean scoring)

| | A_global | G_deterministic | Grouping cost (G − A) |
|---|---|---|---|
| OLD | 0.00042 | 0.01679 | **+0.01637** |
| NEW | 0.01185 | 0.01517 | **+0.00333** |

Δ cost = **−0.0130**. Read carefully: exp66c scores a single forecast
against the *mean* of the test window. OLD's global arm scored 0.0004 —
a point mass that happened to sit almost exactly on the window mean, an
artifact, not skill. The spread forecast is properly penalized for
width against a point target (0.0118), while the group arm genuinely
improves (0.0168 → 0.0152). Conclusion: **most of exp66c's reported
+0.009–0.016 grouping cost was the point-mass bug interacting with
window-mean scoring.** On exp66c's own metric the cost is ≈ +0.003
with the fix; on per-month scoring (this experiment) grouping questions
should be re-asked with the fixed forecaster. exp66c's qualitative
verdict (grouping = explainability layer, global stays forecaster of
record) is not overturned by this — the group arm still does not beat
the global arm under either scoring.

## Layer-2 smoke

`MetaRouter` over {naive, fixed-v2, fixed-global} candidates (seed 7,
control, 2 groups × 9 origins): runs clean, emits one trace per origin
with the full key set (`choice`, `evidence`, `layer2_crps`,
`lockin_shift`, `origin_index`, `origin_pos`, `reason`); gating chose
`naive` at all 18 origins — conservative, consistent with exp73.

## Tests

`tests/test_group_forecaster_spread.py` (10 tests, written RED first):
non-flat spread, exact old-behavior reproduction under the flag,
monotonicity/bounds, noise-scaling of width, constant/short-series
safety, prefix-invariance (PIT), both methods, unknown-method error,
and end-to-end `GroupForecaster` fallback with spread. Full suite:
green apart from the known pre-existing failures
(`test_decision_policy.py::test_gate_constants_sane`, flagged by
exp66g; `test_fetch_fred.py` mock-interface, excluded as before).

## Default switched

**Yes.** `v2_forecaster_fn()` now defaults to
`quantile_spread=True, spread_method="gaussian"`. ⚠️ Coordination note:
exp75's script calls `v2_forecaster_fn(1)` with defaults — if its runs
restart, its "v2 baseline" arm becomes the *fixed* forecaster. exp72
reproduction likewise needs `quantile_spread=False`.
