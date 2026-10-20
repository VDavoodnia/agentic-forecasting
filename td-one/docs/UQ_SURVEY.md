# Uncertainty Quantification for Time-Series Foundation Models — Literature Survey (exp74, Phase 1)

**Date:** 2026-10-20. Papers from 2024 onward (classics EnbPI 2021 / ACI 2021 cited where built upon).

## Why this survey exists

exp68 (reactive guard), exp69/69b (anticipatory, synthetic), exp69c (anticipatory,
real data) all failed on the same mechanism: **correcting the forecast's *location*
needs timing precision we don't have** (exp69c diagnosis: direction right 2/3,
but size overshot 2.6×, timing +1.5mo early). The pivot: stop trying to move
Chronos-2's point forecast; instead **predict its uncertainty** at each origin —
recent volatility, recent forecast-error history, model disagreement,
regime-instability diagnostics, narrative macro signals — and adapt prediction
intervals accordingly. If we can't say *which way* it will miss, maybe we can
say *how far* it might miss, and price that honestly.

## 1. Conformal prediction for time series

The classical split-conformal guarantee needs exchangeability, which time series
violate. The literature splits into four philosophies (alphaXiv taxonomy,
`2511.13608v1`):

### 1.1 Refreshing methods — rolling-residual (EnbPI-style)

- **EnbPI** (Xu & Xie 2021). Wraps any forecaster; maintains a sliding window
  of recent nonconformity scores (absolute residuals, from leave-one-out /
  rolling-origin forecasts), and forms the interval as point ± empirical
  $(1-\alpha)$ quantile of the window. Core idea: replace exchangeability with
  a *stationary strongly-mixing* error assumption; the sliding window forgets
  stale errors. Target: interval width/level. Features: past residuals only.
  Tested on real data, not TSFMs specifically.
- **What we take:** the mandatory baseline. It is exactly "widen/narrow by
  recent |residuals|" — the simplest possible answer to "predict uncertainty".

### 1.2 Adaptive-coverage methods — ACI and its 2024+ children

- **ACI** (Gibbs & Candès 2021): $\alpha_{t+1}=\alpha_t+\gamma(\alpha-err_t)$.
  After misses, $\alpha$ drops and intervals widen; after hits they narrow.
  Long-run coverage converges to nominal under *arbitrary* shift. Target:
  coverage rate; interval width is the consequence. Features: past coverage
  errors only. Our repo already has `AdaptiveConformalM0` (ACI on M0), so the
  machinery is proven in-house.
- **Dt-ACI** (Gibbs & Candès 2024): aggregates multiple learning rates
  $\gamma$ to get *strongly adaptive* regret — reacts to recent shifts without
  hand-tuning $\gamma$.
- **Automatic ACI** (Makhin & Panov, AIST 2024 / Springer 2025): replaces the
  constant $\gamma$ with an adaptive one, explicitly to avoid the width
  explosions ACI shows after big shifts.
- **BC-ACI** (2026 preprint, arxiv 2604.13253): the key insight for *us*.
  ACI only adapts the quantile threshold — it **cannot shift the interval
  center**, so under persistent bias (a level shift the base model doesn't
  track) it must widen symmetrically by $2|b|$ to hold coverage. BC-ACI adds
  per-horizon online EWM bias estimation and corrects scores first: up to
  −32% Winkler score under compound shift, provably neutral when no bias.
  **This matches our exp69c failure mode exactly** — bias accumulating faster
  than width.
- **TCP / TCP-RM** (arXiv 2507.05470): quantile forecaster + rolling
  split-conformal + one Robbins–Monro offset steering coverage in real time;
  bands visibly expand/contract around March-2020-style volatility. Robust to
  window/step-size choice.
- **AcMCP** (Wang & Hyndman 2024): multi-step CP that *forecasts future
  nonconformity scores* — adds a predicted residual term per horizon to the
  ACI update. Relevant because our horizon is h=6, not h=1.
- **Two-Timescale CP** (Texas A&M, 2026): decouples fast calibration
  adaptation from slow model refitting; 3.4× shorter intervals than ACI on
  synthetic, −17–71% width on real data (p=0.002). Degenerates to ACI when
  refitting can't help.

### 1.3 Reweighting methods — weighted / locally-adaptive conformal

- **WCP family** (Tibshirani et al. 2019; Guan 2023; Ghosh 2023; Mao et al.
  2024): weight calibration residuals by relevance to current conditions —
  exponential recency decay $w_i=\beta^{n-i}$, sliding windows, or
  content-based similarity weights. Interval = weighted $(1-\alpha)$
  quantile. Core idea: old residuals from a different regime should not set
  today's width. Features: residuals + a similarity kernel. Tested on
  anomaly detection with TSFMs (arxiv 2604.20122: weighted adaptive
  conformal quantiles + TSFM scores remain calibrated across time).
- **ECP/EACP** (Kasa et al., UAI 2025): adjust the CP *score function* by the
  base model's own uncertainty estimate — e.g. weight residuals by the
  model's native interval width. Unlabeled test data only; nearly matches
  fully-supervised adaptation. For us: Chronos's own quantile spread as the
  relevance weight.

### 1.4 Conformalized quantile regression (CQR)

- (Romano et al. 2019; Nixtla-CP implements a rolling-window variant.)
  Chronos already emits quantiles. CQR calibrates them: on past origins,
  score $s=\max(q_{lo}-y,\,y-q_{hi},\,0)$; widen the native $[q_{10},q_{90}]$
  by the $(1-\alpha)$ quantile of $s$. Non-symmetric, preserves Chronos's
  distributional shape, needs only past forecasts + realizeds. No
  distribution assumption at all.

## 2. Conformal + TSFMs specifically

- **Achour et al. 2025** (arXiv 2507.08858, "Foundation models for time
  series forecasting: Application in conformal prediction"): Chronos,
  Lag-Llama, TimesFM on ERCOT/NN5/M3 with split conformal. Two findings that
  directly motivate our battery: (1) in data-scarce settings TSFMs give
  *more reliable* conformal intervals than classical models, because zero-shot
  means nearly all data can go to *calibration* instead of training; (2)
  calibration is more stable with TSFMs. All our code reproduces the
  finding. Tested on TSFMs; not under explicit shift.
- **Eki.Lab (Ekimetrics, 2025)**: same setup on ERCOT hourly; TSFMs
  (Chronos, Chronos-Bolt, TimesFM) beat LightGBM/StatisticalEnsemble on
  MCR/MSIW/MCR especially in the low-data window — again, zero-shot frees
  data for calibration.
- **"Beyond Accuracy: Are TSFMs Well-Calibrated?" (ICLR 2026)**: 5 TSFMs vs
  2 baselines; TSFMs are consistently *better calibrated* than baselines and
  tend not to be systematically over/under-confident — unlike typical deep
  models. Implication: our adjustment layer starts from an already-decent
  base; we are fixing *conditional* miscalibration (under shift), not
  wholesale overconfidence.
- **Independent practitioners**: TimesFM-2.5 80% coverage 0.64 on a retail
  series (Konrad substack, 2025); Chronos-2 under-covers 1–4pp under sensor
  degradation while TimesFM-2.5/TiRex stay within 2pp (MDPI industrial study,
  2026). Coverage quality is model- and regime-dependent — exactly the
  conditionality a UQ layer should capture.

## 3. Predicting the error itself (meta-models of uncertainty)

This is the philosophical pivot of exp74 — *learn a function from pre-origin
observables to expected |error|*, rather than assuming recent residuals
suffice.

- **Selective forecasting via metalearning** (arXiv 2606.23448): predicts the
  *empirical percentile* of the forecast error from structural features of
  recent lags — a scale-invariant statistic — and abstains on high-risk
  samples. Core idea: decouple the uncertainty decision from the forecast
  itself; percentile prediction transfers across heterogeneous series.
  Target: error percentile → abstention/rejection. Features: lag structure.
  Tested in-domain and transfer. **This is the closest published match to
  our "predict uncertainty, not location" pivot.**
- **Residual postprocessing / second-model correction** (NWP literature:
  Bremnes et al. 2023 postprocess deterministic forecasts with a second
  model; a 2025 load-forecasting paper gates a lightweight residual module
  by "mutation intensity" features). Target: residual mean/variance.
  Features: model inputs + recent errors. Lesson: the second model works
  best when gated by regime features — not always-on.
- **Heteroscedastic scale prediction**: predict a per-origin scale factor
  for the base forecast's spread (Gaussian NLL / CRPS training). Classical,
  CPU-trivial, and directly CRPS-evaluable.
- **Deep ensembles / MC dropout for TSFMs** (NeurIPS 2025 TTM paper): work,
  hyperparameter ensembles beat MC dropout, but they require either many
  forward passes or model access we don't exploit zero-shot — heavy for our
  CPU budget and orthogonal to our feature-based approach. Noted, not
  shortlisted.

## 4. Aleatoric vs epistemic decomposition

Standard decomposition (law of total variance) into data noise
(aleatoric) vs model ignorance (epistemic), via ensembles/BNNs/hypernetworks
(HyperDM 2024, HybridFlow). For a frozen zero-shot Chronos the epistemic
term is not directly accessible without sampling tricks; the practical
translation for us is: **Chronos's native interval width ≈ its learned
aleatoric estimate**, and our layer estimates the *residual miscalibration*
(epistemic-ish: when is the model wrong about its own uncertainty?).
We implement the decomposition behaviorally — as features (native width,
past miscoverage) — not as weight-space inference.

## Shortlist for exp74 (Phase 2)

| # | Method | Family | Target | Features needed | TSFM-tested? | Shift-tested? |
|---|---|---|---|---|---|---|
| S1 | Native Chronos intervals | baseline | — | — | yes | yes (exp67) |
| S2 | Fixed-width expansion | baseline | width ratio | past width/coverage | — | — |
| S3 | Rolling-residual conformal (EnbPI-style) | conformal | interval | past rolling |error| | — | partial |
| S4 | ACI (+ bias-corrected variant) | conformal | coverage | past coverage errors | no | yes |
| S5 | Weighted conformal (recency × volatility-similarity) | conformal | interval | past residuals + similarity kernel | partial | yes |
| S6 | CQR on Chronos quantiles | conformal | calibrated quantiles | past quantile exceedances | yes (Achour) | partial |
| S7 | Error-percentile meta-model (GBM, numeric features) | meta-model | |error| percentile → width | volatility, past errors, diagnostics | no | no |
| S8 | S7 + story/macro features (ablation of S7) | meta-model | same | + story classifications, macro signals | no | no |

Why these six (S3–S8): each has a distinct failure-mode hypothesis for our
setting. S3 assumes recent residuals are representative (breaks at shift
onset — that's the test). S4 reacts to coverage misses online (slow to
fire — the exp68 lesson). S5 discounts stale regimes. S6 keeps Chronos's
shape and only fixes the edges. S7 learns *when* Chronos is wrong from
observables (the pivot). S8 tests whether narrative/trend features add
anything beyond numbers (the user's hypothesis).

Success (pre-registered in exp74 report header): a method wins if it moves
coverage toward nominal **without widening more than 25% vs native** on
pooled test origins — or improves CRPS significantly. Width alone is not a
win; coverage alone bought with 3× width is not a win.
