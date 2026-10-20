# TD-One: Final Delivery

> **Status: CURRENT — 2026-10-20**
>
> **Update (2026-10-20):** the future-directions wave (exp82 to exp88)
> has landed and is folded into this note, the final results, and the
> report's Section 6. All seven directions are answered below; two
> follow-ups remain open and are named at the end.

## What this package is

TD-One is the completed TD-One research program: **Agentic Solutions for
Loan Attrition Prediction**, built and tested end to end for the Vector
Agentic Forecasting Bootcamp. The package is this note plus the HTML
final report (8 exec slides and a full paper: architecture, datasets,
models, per-dataset final results, visualizations, and appendices
covering ablations, prompts and parameters, a glossary, and the
failed-experiments register), the final-results document, the Phase-2
update, the loan-attrition plan, the methodology reference, and the
experiment registry (exp1 to exp88 plus the Macro Signal Service line,
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
where story features and a long history exist (its feature set is now
story features plus the inflation macro signal's signed direction
wherever the signal store has coverage, exp83); adaptive conformal
(ACI) as a regime-gated option; and the **bias-corrected family
(BC-ACI)** as the documented shift-regime option (exp85). On the mortgage book, deterministic
segment rules feed the **group forecaster** (trailing six-month mean
center, Gaussian spread from each segment's own past errors, and the
thin/dead fallback split) for forecasts, while grouping itself serves
explainability through citation-enforced risk narratives — which have
now run end to end with real news citations (exp87). Segment discovery
has two tested routes: an **LLM segment scout** that recovers planted
cohorts about four times better than clustering when a matching domain
prior exists (exp88), and SHAP-space clustering as the prior-free
method. The Macro
Signal Service's calibrated news signals are uncertainty features only;
news never moves the best guess. Its store now spans 2021-01 to
2026-09 (899 signals).

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
| Macro signal, stormy window (exp83) | champion + signed inflation 2.2179 (stormy scope, cov80 0.854) | champion without macro 2.2297 | −0.53%, bootstrap CI [−0.91%, −0.20%]; meta family −1.10%, CI [−2.09%, −0.25%]; quiet control null |
| Shift-proof ranges (exp85) | BC family on Food CPI 1.2062–1.2106 | plain CQR 1.2824; fusion champion 1.2587 | −4.6 to −5.6%, p ≤ 7e-22; fresh-seed shock coverage 0.828–0.839 vs the 0.77 bar |
| Loan-level spread (exp84) | +0.563 / +0.487 / +1.055 per 100bp (2021 / 2022–23 / 2024+) | no-spread model | all eras p≈0; cross-era OOS ΔAUC positive 6/6 (mean +0.078); associational, not causal |
| Segment scout (exp88) | ARI 0.361 (flash variant 0.397) | SHAP clustering 0.091 | planted rules recovered verbatim on some seeds; wins when the domain prior matches the segment logic |
| Manager reports (exp87) | citation contract 100%; news citations in 100% of flagged explanations | pre-registered bars: contract 100%, news citations ≥ 50% | judged relevance 20/20 on the full-corpus round; point-in-time leakage audit clean |

## What shipped

- Proposed Method v2 (router + gates + gated LLM expert) — the forecaster of record.
- Layer-2 validation gate — deployment wrapper on single-series targets; harm repair proven on OSFI.
- CQR range layer — the default uncertainty layer (coverage repair at ~1.03–1.08× width).
- Pooled conformal banks (n0=24, family-gated ≥8 series) — end the M4 conformal no-op; Tourism wins on error and coverage.
- Story-feature fusion (F1, horizon-gated) — best range method where story features and a long bank exist (Food CPI).
- ACI — regime-gated option for volatile subsets.
- Group forecaster — trailing-mean center + Gaussian spread + thin/dead fallback split; the fitted router stays an option, not the default.
- Deterministic grouping + citation-enforced explanations — the explainability layer; LLM+statistics is the risk classifier of record.
- Manager risk reporter, run end to end (exp87) — real monthly reports on OSFI segments and Freddie Mac SMM with enforced news citations (contract 100%, news citations in 100% of flagged explanations, leakage audit clean).
- BC-ACI family (bias-corrected conformal, exp85) — the documented shift-regime range option in `td_one.uq.adaptive`; the group forecaster's adaptive conformal modes ship as options alongside it. No default changed.
- Pooled router (exp82) — tested research option in the group forecaster (`pooled_router_forecaster_fn`); the trailing-mean center stays the default.
- LLM segment scout (exp88) — segment-discovery front-end when a matching domain prior exists (`src/td_one/agent/segment_scout.py`); SHAP clustering remains the prior-free method.
- Loan-level origination spread (exp84) — validated loan-level attrition feature: same positive sign in all three eras, cross-era out-of-sample gains.
- Macro Signal Service — calibrated signal store (7 of 9 anchored categories live; 899 signals spanning 2021-01 to 2026-09), used as uncertainty features; the signed inflation signal is in the champion's range feature set (exp83).
- SHAP-space clustering — best prior-free planted-cohort recovery method (ARI 0.091), in the analyst kit; the exp88 scout beats it when a domain prior matches.

## What was killed (each on evidence, with the number in the report)

- News as a forecast input — five tests, no gain anywhere (exp19, exp21, exp25, exp46, exp60–62 re-eval).
- Anticipatory and reactive LLM overlays — even the oracle arm fails; timing precision is the limit (exp68, exp69/69b/69c).
- LLM as combiner, calibrator, blind grouper, and orchestrator — each dominated by a deterministic alternative (exp63, exp56, exp66, exp75). The context-fed segment scout is a different configuration and ships (exp88, above).
- The fitted v2 router as the group forecaster's center — at most 13 backtest records per series (exp79).
- The pooled router as that center — pooling rejected the starvation hypothesis (about 8x the records) and still lost in both worlds; level anchoring and scale-dominated expert selection bind instead (exp82). Research option only.
- Chronos-2 as the segment engine — loses the pre-registered member-weighted endpoint and degrades more under the shock; its unweighted win was seed noise. Its strength is portfolio level (exp86).
- Group conformal layer as a default — failed the fresh-seed shock bar; available behind a flag (exp80a).
- Conditional CQR (F2) — cross-horizon score pooling bias (exp76).
- Macro signals in the forecast router — mechanism-blocked on calm-regime data (exp71-MS, exp77). As uncertainty features the calm-window test was null (exp76 S8), but the stormy-window retest passed for the signed inflation direction (exp83): the router kill stands, the uncertainty use is validated in volatile windows.
- Competitor-rate spread at aggregate level — a 2022–23 regime artifact (exp71b); lock-in correction — no signal (exp73).
- Learned combination weights, learned regimes, kitchen-sink ensembling, "high" reasoning effort (exp42, exp45, exp51, exp48).
- Pooling variants inside exp81: per-side scores, residual-scale normalization, similarity gating — each diagnosed and closed in the exp81 report.

## Future directions — answered by the 2026-10-20 wave (exp82 to exp88; full text in the report, Section 6)

- **The pooled router — answered, no (exp82).** One router fitted on pooled cross-series backtests lost to the trailing-mean center in both worlds (control +0.0060, p=0.0052; rate shock +0.0131, p=1.1e-05). Starvation rejected; it ships as a research option only.
- **Macro signals on a stormy window — answered, yes (exp83).** Store backfilled into 2021–2023 (899 signals total). Story + signed inflation direction beats story-only by −1.10% CRPS on shock origins (CI excludes zero) and improves the shipped champion by −0.53%; quiet-window control null. The feature ships in the champion's set.
- **Competitor spread at loan level — answered, yes (exp84).** In the 7.32M-loan Freddie Mac cross-section the origination spread predicts 24-month prepayment with the same positive sign in all three eras and positive cross-era out-of-sample AUC gains in all 6 pairs. The aggregate-level kill stands; the signal's home is loan level.
- **Shift-proof conformal — answered, split (exp85).** On the Chronos-2 battery the bias-corrected family clears the fresh-seed shock bar and beats CQR on Food CPI by 4.6–5.6%: it ships as the documented shift-regime option. On the group forecaster the bar was not cleared (best 0.658 vs 0.77): the ceiling is the bank's information content at origins 15–17, not tuning. No defaults changed.
- **Chronos-2 as segment engine — answered, no (exp86).** The local champion wins the member-weighted endpoint and degrades least; Chronos-2's apparent win was seed noise. Pooled conformal over Chronos-2 does repair its coverage (0.82–0.83) — a portfolio-level note, not a segment ship.
- **The manager report, run end to end — answered, yes (exp87).** The news-citation path ran end to end on real monthly reports: contract 100%, news citations in 100% of flagged explanations, judged relevance 20/20 on the full-corpus round, leakage audit clean.
- **Segment scout, scored — answered, yes, with the caveat (exp88).** The scout recovers planted cohorts at mean ARI 0.361 (flash 0.397), about 4x the SHAP incumbent, with planted rules recovered verbatim on some seeds — when the domain prior matches the segment logic. SHAP context hurts it (0.052) and outcome rates fail as a proposer signal (0.077).

**Still open (the only two):**
- **Exogenous widening triggers** for the group forecaster's bank-empty origins (exp85's named successor): widen ranges from an outside signal, e.g. the Macro Signal Service's risk channel, where a causal bank holds no post-shift evidence.
- **Borrower-level vote merging** for the segment scout's self-consistency (exp88 follow-up): merge sampled rule sets by voting on borrower assignments, not on rules, to stop the fragmentation seen in the merge round.

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
