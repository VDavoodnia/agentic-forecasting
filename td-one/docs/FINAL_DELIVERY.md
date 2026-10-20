# TD-One: Final Delivery

> **Status: CURRENT — 2026-10-20**

## What this package is

TD-One is the completed TD-One research program: **Agentic Solutions for
Loan Attrition Prediction**, built and tested end to end for the Vector
Agentic Forecasting Bootcamp. The package is this note plus the HTML
final report (8 exec slides and a full paper: architecture, datasets,
models, per-dataset final results, visualizations, and appendices
covering ablations, prompts and parameters, a glossary, and the
failed-experiments register), the final-results document, the Phase-2
update, the loan-attrition plan, the methodology reference, and the
experiment registry (exp1 to exp81 plus the Macro Signal Service line,
every experiment with a verdict and a report). All experiments have
landed; nothing in the package is in flight.

## The final architecture, in one paragraph

The shipped system is **Proposed Method v2** (a regime-probability
router over classical experts, with a trust flag, a do-nothing gate,
and a gated LLM expert, Gemini Flash with the V1 prompt, admitted only
where it has proven to help) wrapped by the **Layer-2 validation gate**
on single-series targets (a deterministic analyst toolbox plus
walk-forward gating that vetoes to the simple option when the past
record cannot support the forecast, and mechanically repairs harm).
Around it sits the **uncertainty layer**: per-series conformalized
quantile regression (CQR) by default; **pooled conformal banks**
(signed scores, shrunk toward each series' own scores, n0=24) on
families with at least 8 series; the h3-gated story-feature fusion
where story features and a long history exist; and adaptive conformal
(ACI) as a regime-gated option. On the mortgage book, deterministic
segment rules feed the **group forecaster** (trailing six-month mean
center, Gaussian spread from each segment's own past errors, and the
thin/dead fallback split) for forecasts, while grouping itself serves
explainability through citation-enforced risk narratives. The Macro
Signal Service's calibrated news signals are uncertainty features only;
news never moves the best guess.

## Headline results (CRPS, lower is better, unless noted)

| Benchmark | Ours | Baseline | Evidence |
|---|---|---|---|
| M4 Monthly | 0.1451 | best single 0.1505, equal 0.1507 | p=7.8e-11 vs best single |
| Tourism Monthly | 0.1282 | best single (seasonal naive) 0.1387 | p=2.6e-34; zero-shot TSFMs reach 0.0965 and the router ties them |
| M3 Monthly | 0.1476 | best single (ETS) 0.1630 | p=3.5e-35 |
| Food CPI | 0.5322 (v2) | v1 classical 0.9661 | p=4.3e-06; the LLM expert alone (0.4262) is the best single forecaster here |
| BoC decisions (accuracy) | 72.2% | climatology 69.4% | 119 meetings, corrected set |
| OSFI M4 runoff | naive is the bar (0.00369) | v2 ungated 0.00381 | Layer-2 gate: do-no-harm holds; uninsured harm +10.7% (p=0.030) repaired to +0.8% (n.s.); h=3 beats v2 by 6.8% (p=0.015) |
| Freddie Mac SMM | 0.000508 (v2) | v1 0.000512 | p=0.69, neutral: the honest real-data attrition benchmark |
| Uncertainty layer, Food CPI | CQR 1.2824; story fusion 1.2587 | native ranges 1.3035 | p=1.7e-04 and p=2.0e-05; coverage 0.824 and 0.854 at the 80% level |
| Pooled banks (exp81) | Tourism CRPS 850.46, cov80 0.806 | per-series CQR 859.05, cov80 0.783 | pooled scale-free difference −0.0028, p=0.0066; M4 coverage 0.768 → 0.805 (was an exact no-op) |
| Group forecaster (synthetic book) | 0.0899 control | naive center 0.1042; pre-fix defect 0.1320 | p=0.034; coverage 0.075 → 0.714; grouping cost +0.0018 |

## What shipped

- Proposed Method v2 (router + gates + gated LLM expert) — the forecaster of record.
- Layer-2 validation gate — deployment wrapper on single-series targets; harm repair proven on OSFI.
- CQR range layer — the default uncertainty layer (coverage repair at ~1.03–1.08× width).
- Pooled conformal banks (n0=24, family-gated ≥8 series) — end the M4 conformal no-op; Tourism wins on error and coverage.
- Story-feature fusion (F1, horizon-gated) — best range method where story features and a long bank exist (Food CPI).
- ACI — regime-gated option for volatile subsets.
- Group forecaster — trailing-mean center + Gaussian spread + thin/dead fallback split; the fitted router stays an option, not the default.
- Deterministic grouping + citation-enforced explanations — the explainability layer; LLM+statistics is the risk classifier of record.
- Macro Signal Service — calibrated signal store (7 of 9 anchored categories live), used as uncertainty features.
- SHAP-space clustering — best planted-cohort recovery method (ARI 0.091), in the analyst kit.

## What was killed (each on evidence, with the number in the report)

- News as a forecast input — five tests, no gain anywhere (exp19, exp21, exp25, exp46, exp60–62 re-eval).
- Anticipatory and reactive LLM overlays — even the oracle arm fails; timing precision is the limit (exp68, exp69/69b/69c).
- LLM as combiner, calibrator, grouper, and orchestrator — each dominated by a deterministic alternative (exp63, exp56, exp66, exp75).
- The fitted v2 router as the group forecaster's center — at most 13 backtest records per series (exp79).
- Group conformal layer as a default — failed the fresh-seed shock bar; available behind a flag (exp80a).
- Conditional CQR (F2) — cross-horizon score pooling bias (exp76).
- Macro signals in the forecast router — mechanism-blocked on calm-regime data (exp71-MS, exp77); as uncertainty features they added nothing in the calm window tested (exp76 S8).
- Competitor-rate spread at aggregate level — a 2022–23 regime artifact (exp71b); lock-in correction — no signal (exp73).
- Learned combination weights, learned regimes, kitchen-sink ensembling, "high" reasoning effort (exp42, exp45, exp51, exp48).
- Pooling variants inside exp81: per-side scores, residual-scale normalization, similarity gating — each diagnosed and closed in the exp81 report.

## Where everything lives

- **Final report (HTML, deck of record):** [slides/TD-One-Slides.html](slides/TD-One-Slides.html), with its context doc [slides/TD-One-Context.md](slides/TD-One-Context.md)
- **Final results:** [FINAL_RESULTS.md](FINAL_RESULTS.md)
- **Phase-2 update (everything since exp63):** [PHASE2_UPDATE.md](PHASE2_UPDATE.md)
- **Loan-attrition program plan and track verdicts:** [LOAN_ATTRITION_PLAN.md](LOAN_ATTRITION_PLAN.md)
- **Methodology (the reference spine):** [METHODOLOGY.md](METHODOLOGY.md)
- **Literature and SOTA surveys:** [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) and [UQ_SURVEY.md](UQ_SURVEY.md); the 2026-10-19 agentic-forecasting SOTA survey (23 papers, 11 repos) was delivered as the artifact page "agentic-forecasting-sota-survey"
- **Macro Signal Service:** [MACRO_SIGNAL.md](MACRO_SIGNAL.md)
- **Experiment registry (every experiment, verdict, report):** [../experiments/README.md](../experiments/README.md)
- **Self-audit (multiple comparisons, development-on-test caveats):** [SELF_AUDIT.md](SELF_AUDIT.md)
