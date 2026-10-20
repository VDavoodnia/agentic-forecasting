# TD-One Experiments

End-to-end results for the mortgage attrition forecasting ladder. Every
experiment is PIT-clean (no lookahead) and reproducible — see
`REPRODUCIBILITY.md` for the full reproduction steps.

## The ladder

| Rung | Model | What it is |
|---|---|---|
| M0 | Historical baseline | Trailing-6 Jeffreys Beta-Binomial on segment panels |
| M1 | Zero-shot TSFMs | Chronos-2 / TimesFM-2.5 on panel switching-rate series |
| M2 | ReAct LLM agent | Groq-driven tool loop (data_segment → evidence → forecasting) |
| M3 | Hybrid | M0 + bounded log-odds adjustment Δ ∈ [-0.75, +0.75] proposed by the agent |
| M4 | Regime reader | LLM reads news → regime label → model selection |
| M5 | Ensemble router | LLM routes among experts (REATS-style) |

Primary metric everywhere is **CRPS** (proper score on count distributions);
secondary: MAE and 80% interval coverage.

Verdict key: ✅ positive · ⚖️ partial/mixed · ❌ honest negative · 🔨 build/scaffolding · ▶️ running

## Experiment index

Numbering jumps from exp25 to exp40 (exp26–exp39 were never assigned).

| Exp | Question | Verdict | Report |
|---|---|---|---|
| exp1 | M0 vs M1 (mock TSFMs), 500×24 synthetic | ✅ M1a 3.26 / M1b 3.32 vs M0 3.50 MAE | [report](exp1_report.md) |
| exp2 | M0 vs M1 (real weights), 3 scenarios × 5 seeds | ✅ Chronos-2 MAE 4.00 (−20% vs M0); TimesFM-2.5 4.62 (−8%) | [report](exp2_report.md) |
| exp3 | M2 vs M0, 3 pilot segments, mock + real Groq | ⚖️ seg-1 win (0.82 vs 1.33), seg-0 worse, seg-2 never finalized | [report](exp3_report.md) |
| exp4 | M3 vs M0, 6 macro scenarios, mock + real Groq | ⚖️ 3/6 beat M0 — LLM directionally right but over-adjusts | [report](exp4_report.md) |
| exp5 | Better classical baselines vs M0 | 🔨 classical-track scaffolding | `exp5_results.json` |
| exp6 | M3 + shrinkage/gating vs M3/M0 | 🔨 decision policy built: James-Stein shrinkage + corroboration gate | `exp6_results.json` |
| exp7/7b | Regime-classifier hybrid (M3c), prototype + real-LLM test | 🔨 prototype → fed M4 regime-reader design | — |
| exp8 | Macro-grounded DGP: does macro structure reward macro-aware models? | ❌ macro track eventually closed (see exp10/13/15) | — |
| exp9 | Richer DGP (simulator v2): do conclusions change? | 🔨 DGP robustness check | — |
| exp10 | Classical track: promoted baselines under one roof | ❌ 0.678 CRPS gain was 100% z-score leak | `exp10_results.json` |
| exp11 | M4 regime-reader agent vs M0 | 🔨 led to regime-reader hardening (see `../docs/AGENT_AUDIT_M4.md`) | `exp11_results.json` |
| exp12/12b | M5 ensemble router v1, then critic-constrained v2 | ❌ v1 failed (see `../docs/AGENT_AUDIT_M5.md`); v2 tested | `exp12_results.json`, `exp12b_results.json` |
| exp13 | TSFM + FRED rematch — the fair fight | ❌ PIT-safe rerun reversed macro benefit to penalty | `exp13_tsfm_macro_results.json` |
| exp14 | M4 Monthly real-data benchmark | 🔨 real-data baseline for the agentic track | `exp14_m4_results.json` |
| exp15 | Macro overlay with corrected sign (negative lock-in) | ❌ neutralized — no evidence macro helps | `exp15_macro_corrected_results.json` |
| exp16 | Freddie Mac real-data SMM forecasting (PoC) | 🔨 real-data proof of concept | `exp16_freddie_smm_results.json` |
| exp17 | News-augmented regime reader, end-to-end | ⚖️ mechanism probe | `exp17_news_regime_results.json` |
| exp18/18b | Full 63-month Freddie SMM benchmark; fair ablation + ARIMA/ETS | 🔨 calibrated-TSFM real-data benchmark | `exp18_smm_full_results.json`, `exp18b_fair_smm_results.json` |
| exp19 | News-regime-augmented SMM forecasting | ❌ frozen — news doesn't move SMM | `exp19_news_smm_results.json` |
| exp20 | Event-augmented simulator validation | 🔨 build complete — 84% news-leading events (20 seeds) | [build summary](exp20_BUILD_SUMMARY.md) |
| exp21 | LLM feature discovery on event DGP | ❌ −21.6%: discovered features lag the news | [report](exp21_report.md) |
| exp22 | M2 v2 (decision policy) vs M0 | ⚖️ +6.6% when active, high variance | [report](exp22_report.md) |
| exp23 | Rule-based model selection | ❌ = random (p=0.80) | [report](exp23_report.md) |
| exp24 | Regime-aware selection (news-signal regimes) | ❌ honest wash | [report](exp24_report.md) |
| exp25 | h=3 horizon: does news help further out? | ❌ news worse by +25% | [report](exp25_report.md) |
| exp40 | Calibrated rerun of the event track | ⚖️ calibration cuts false discoveries; transfer problem remains | [report](exp40_report.md) |
| exp41 | Meta-learning (FFORMA-style) for model selection | ⚖️ learned beats hand-written; both lose to ARIMA | [report](exp41_report.md) |
| exp42 | FFORMA meta-learner vs equal weights | ❌ honest negative — doesn't beat equal weights | [report](exp42_report.md) |
| exp43 | Constrained regime router (3 regimes + fixed experts) | ✅ first to beat equal weights: M4 CRPS 0.1282 vs 0.1295, p=0.018 | [report](exp43_report.md) |
| exp44 | News as model selector (synthetic → real) | ⚖️ mechanism proven on synthetic (p=0.0068); real blocked by news sparsity | [report](exp44_report.md) |
| exp45 | Learned regimes (flexible) | ❌ honest negative — flexibility loses to the constrained prior | [report](exp45_report.md) |
| exp46 | Dense-news regime/selection | ❌ definitive negative — dense news doesn't help | [report](exp46_report.md) |
| exp47 | Proposed method (exp43 + exp44 combined) | ✅ beats equal weights (0.1451 vs 0.1507, p=6.5e-09), best single, and exp43 alone | [report](exp47_report.md) |
| exp48 | Reasoning-effort sweep (low/med/high × 3 tasks × 3 seeds) | ❌ "high" never helps: 3.5–6× slower, sometimes hurts (1.000→0.333 on BoC discovery). Default to "low" for diagnostics | [report](exp48_report.md) |
| exp49 | Final model run with updated toolset (13 tools) | ✅ reproduces exp47 exactly (0.1451/0.1507/0.1505) — toolset refactor changed nothing | [report](exp49_report.md) |
| exp50 | 4 open extensions (TSFM expert, BoC discrete, loss regimes, learned gates) | ⚖️ 1/4 helps (BoC discrete ✅); 3 honest negatives | [report](exp50_report.md) |
| exp51 | Kitchen sink: all nonsignificant positives combined | ❌ sink worse than proposed on M4 (+0.00073, p=0.25) and SMM (+0.00166, win 0.26); validates the significance filter | [report](exp51_report.md) |
| exp52 | Proposed vs Vector's `LastValuePredictor` (their floor) | ✅ proposed wins 11.8% (1.0662 vs 1.2091, p=7.1e-04, 50/59) — weak baseline, superseded by exp53 | [report](exp52_report.md) |
| exp53 | Vector's `QuantileGridLLMPredictor` (gpt-oss-120b) vs proposed, same 59 origins | ❌ **LLM wins decisively**: 0.5688 vs 1.0339, p=5.4e-05, 43/58 origins. Genuine skill (no memorization pattern); better calibration on smooth series | [report](exp53_report.md) |
| exp54 | Rung 1: dataset registry — exp49 Food CPI + M4 rerun with registry-wired forecaster | ✅ reproduces exp49 exactly (max abs diff 1.8e-15 on M4, 0.0 on Food CPI) — expected null; value is the infrastructure | [report](exp54_report.md) |
| exp55 | Rung 2: LLM as expert in the regime router | ✅ proposed+llm 0.5057 beats proposed 0.9661 (p=9.8e-07) and LLM alone 0.5158 (p=0.037); router picks llm in both regimes — the gain is the LLM expert, not regime selection | [report](exp55_report.md) |
| exp56 | Rung 3: LLM as calibrator of classical 9-quantile grids | ❌ honest negative: calibrated grids 1.1391 alone; proposed+calibrated 1.1432 vs proposed 0.9661 (p=0.32, n.s., directionally worse); router picks naive everywhere | [report](exp56_report.md) |
| exp57 | Rung 4: per-dataset prompts V1 (history stats in words) / V2 (+ PIT-safe macro) | ✅ V1 0.4262 directionally beats generic 0.5158 (p=0.057); V2's extra macro context adds nothing (vs V1 p=0.20); mini-router picks V1 every fold — selection adds nothing | [report](exp57_report.md) |
| exp58 | Rung 5: hybrid regime router over {4 classical + 4 LLM experts} | ⚠️ 0.4289 beats Vector 0.5688 by 24.6% (paired p=0.009) BUT loses to llm-promptV1 alone (0.4262, p=0.058) — router adds no value when one expert dominates; guardrails reproduce exp49 bit-exact (M4/SMM/BoC/Synthetic) | [report](exp58_report.md) |
| exp59 | Rung 6: LLM generalizability on Tourism/M3 (30 series each, h=1) | ❌ clean negative: LLM 64% worse on Tourism, 36% worse on M3 — Food CPI win was dataset-specific | [report](exp59_report.md) |
| exp60 | Full ladder on Tourism Monthly (Monash, 366 series, h=24) | ✅ proposed 0.1282 beats equal 0.1869 (p=1.9e-68) AND best single snaive 0.1387 (p=2.6e-34); 7-model: timesfm25 0.0965, proposed ties best (p=0.42); news-on: negligible Δ | [report](exp60_report.md) |
| exp61 | Full ladder on M3 Monthly (Monash, 1428 series, h=18) | ✅ proposed 0.1476 beats equal 0.1641 (p=2.7e-44) AND best single ets 0.1630 (p=3.5e-35); 3rd dataset confirming the claim; news-on: negligible Δ | [report](exp61_report.md) |
| exp62 | M4-100 with news module on | ✅ reproduces exp47 exactly (0.1451 vs 0.1507, p=7.2e-09); news fire 0.97, no material change — 3rd dataset where news adds nothing | [report](exp62_report.md) |
| exp63 | LLM as forecast combiner vs statistical router (60 tourism series) | ❌ honest negative: LLM 0.1169 significantly WORSE than router 0.0992 (p=1.7e-08) and best-single 0.1005 — LLM subtracts value | [report](exp63_report.md) |
| exp64 | Proposed method v2: unified router + gated LLM expert, all 6 benchmarks | ✅ v2 wins/ties everywhere: Food CPI 0.5322 vs v1 0.9661 (p=4.3e-06), beats Vector LLM by 6.4%; M4/Tourism/M3/BoC unchanged (gate off = v1) | [report](exp64_report.md) |
| exp65 | SMM with LLM expert (v2, Gemini Flash, 50 origins) | ⚖️ neutral: v2 0.000508 vs v1 0.000512 (p=0.69); LLM alone 0.000492 ties TimesFM-2.5 (0.000497) | [report](exp65_report.md) |
| exp66 | LLM grouping + group-level attrition forecasting (4-arm ablation) | ❌ honest negative: LLM groups 0.01705 worse than random 0.00690; global 0.00227 wins — kill criterion triggered | [report](exp66_report.md) |
| exp66b | Grouping for explainability (full v2, prompt playground, LLM-as-selector) | ⚠️ pivot: LLM grouper works in isolation (5/5) but flaky in full pipeline — groupings not reproducible run-to-run; recommendation: deterministic groups + LLM explanations | [report](exp66b_report.md) |
| exp66c | Deterministic grouping (kmeans + tree-surrogate rules) + LLM explanations | ✅ pivot validated: 10 reproducible groups, ≤4 plain-English conditions, 10/10 citation pass; grouping costs CRPS vs global (+0.009–0.016); static risk bins do NOT calibrate (inverted) | [report](exp66c_report.md) |
| exp66d | Cohort recovery + LLM risk classifier (exp66c follow-up) | ❌ recovery honest negative: ARI 0.01 — grouper finds real segments, not the planted cohorts; LLM classifier monotonic on test vs inverted bins (n=10, directional) | [report](exp66d_report.md) |
| exp66e | Uplift + SHAP tools for cohort recovery (Phase 1: tool-based recovery; Phase 2: LLM grouper with tool context) | ⚠️ mixed: SHAP clustering doubles ARI (0.091 vs 0.046, `low_rate` F1 0.55) but interaction recall@10 = 0, uplift = 0.002; Phase 2 negative — tool context doesn't help LLM grouper (0.025 vs 0.026) | [report](exp66e_report.md) |
| exp66f | LLM risk classifier ablation (LLM vs news, 3 seeds) + risk-identification tools | ❌ no measurable LLM advantage over bins (pooled +0.154, CI covers 0); news effect +0.031 [−0.45, +0.50]; counterfactual tool directional only (+0.348, 1 seed) | [report](exp66e_report.md) |
| exp66g | Multi-seed confirmation of the counterfactual risk tool (5 seeds) + 5 diagnosis-driven iteration rounds | ❌ NOT CONFIRMED: B vs A MAE delta −0.0018 [−0.0106, +0.0079], B vs C n.s.; shuffled-CF control proves content inert; seed-42 was a favourable seed. Genuine positive: stats-only LLM beats bins on bin error (+0.0198, CI excludes 0) | [report](exp66g_report.md) |
| exp67 | TSFM red-team on 7 loan-attrition shifts (real Chronos-2/TimesFM-2.5 weights) | ⚖️ all models break together at 4σ S1 (+185–228%); S4/S5 never break (honest negative); Chronos-2 best absolute in 6/8 shifts; coverage collapses (m0 0.61→0.08) | [report](exp67_report.md) |
| exp68 | Agentic shift guard (CUSUM + truncation + widening) vs TSFM-alone, pre-registered holdout | ❌ honest negative: guard +0.5% (p=0.824), wins 14.6% — context truncation hurts; correct detection ≠ benefit | [report](exp68_report.md) |
| exp69 | Anticipatory LLM shift adjustment (LLM reads PIT-safe news → calibrated level-shift overlay on Chronos-2), pre-registered | ❌ honest negative: overlay +10.2% (p=0.231), wins 36%; oracle ceiling +11.4% (p=0.214) — mechanism is the limit, not the LLM; works only at S1/4σ (+30%) | [report](exp69_report.md) |
| exp69b | Broad-synthesis variant: LLM + semantic RAG over 12+ macro topics vs rate-news-only | ❌ synthesis hurts: D vs B′ −5.0% (win 5.6%) — corroborating context dilutes timing; unannounced-shock synthesis untested, not refuted | [report](exp69b_report.md) |
| exp69c | Real-data anticipatory adjustment: multi-call story pipeline (retrieve→rank→story→classify + timeline anchors) on 2022 hikes / 2024 cut, Food CPI + SMM, pre-registered | ❌ honest negative: B vs A +23.7% (p=0.231, win 0%); B vs C −10.5% n.s. — pipeline reads news well (direction 2/3) but calibrated size overshoots 2.6× and timing runs +1.5mo early; size+timing variants still n.s. (−4.3%) | [report](exp69c_report.md) |
| exp70 | OSFI M4 real Canadian runoff benchmark: v2 vs naive, 121 months, 4 segments + total | ❌ honest negative (kill triggered): v2 ties naive on total (−0.2%, p=0.54), significantly worse on uninsured (+10.7%, p=0.030) — aggregate runoff is near-random-walk; regime machinery adds variance | [report](exp70_report.md) |
| exp71 | Competitor spread (Perplexity broker rates) as refi-pressure covariate for runoff | ❌ honest negative: zero effect as router feature; in-sample r=−0.85 does not survive out-of-sample (−295% MSE, n=7); Perplexity rates too sparse (8/21 months) | [report](exp71_report.md) |
| exp71b | Competitor spread re-test at full coverage (BoC 622m backbone + Wayback RateHub 54m) | ❌ settled negative: in-sample r shrinks to −0.345 on real months (pilot's −0.85 was a forward-fill artifact); OOS slope hurts (+39%, p=0.019); short-window variant +322%; router still Δ=0.0. Regime artifact, not a data problem | [report](exp71b_report.md) |
| exp72 | Integration: Track-2 shift scenarios (S1/S3/S7, placebo-controlled) stress-test Track-1's group forecaster, 5 seeds + province-oracle ablation, 5 iteration rounds | Mixed, pre-registered: Q1 not supported — S3@4σ group−global degradation diff −0.0021 (p=0.22, 5 seeds; oracle geography identical); Q2 fallback is a working safety valve with a blunt trigger (helps thin groups, lags collapse, wrong for dead groups); Q3 Layer-2 protects in the thin-group regime (LAYER2Q−GROUPQ −0.021..−0.031, p<1e-8 on all worlds, vanishes with 5 big province groups). Honest negative on James–Stein τ-sweep. Flagged: Track-1 v2_forecaster_fn emits flat point-mass quantiles (CRPS 0.165 vs 0.13 for Gaussian wiring) | [report](exp72_report.md) |
| exp73 | Layer-2 meta-router (deterministic analyst toolbox + walk-forward gating) over v2 on OSFI runoff | ❌ pre-registered bar: lock-in correction adds nothing (D +10.7% vs naive, p=0.006). Partial positive: validation gating ties naive and repairs v2's harms (uninsured +10.7%→+0.8%; h=3 C beats B −6.8%, p=0.015) — gating kept as the do-no-harm wrapper | [report](exp73_report.md) |
| exp70-MS | Macro Signal Service calibration: LLM-read monthly categorical signals (22 categories × 36 months) vs FRED anchors, train/holdout, >55% gate, 5 enforced iteration rounds | ✅ 7/9 anchored categories live (fx corr +0.63 holdout 0.79; food_prices holdout 0.82; inflation via timing R2=0.93; wars via smoothing; services via strong-only). Dead: basic_needs (anti-correlated), central_bank_rates (silent). Certainty does not calibrate | [report](exp70_macro_signal_report.md) |
| exp71-MS | Macro Signal integration: live signal vector into ProposedForecasterV2 on SMM, 28 origins, 5-fold CV, ≥1% effect floor | ❌ honest negative after 7 rounds (effect ~1e-7 CRPS): signal dilution (mean |stress| 0.167) + structural block — all 28 origins label 'calm' (stressed months 2022, pre-coverage), router degenerate, experts collapse. Signal store stays live; Food CPI integration settled by exp77 (same block) | [report](exp71_macro_signal_report.md) |
| exp74 | Predict Chronos's uncertainty, not its forecast: 11 UQ arms (conformal/ACI/CQR/meta-GBM incl. story+macro features) on exp67 shifts + Food CPI | ✅ CQR wins pooled (synth cov80 0.811 @ wr 1.03; food CRPS 1.282 vs 1.304, p=1.7e-04); ACI wins on volatile food subset (p=3.6e-05); story ablation real: meta+story beats meta-numeric (paired p=2e-11, `st_magnitude` #5 feature). Onset coverage honest negative — no residual method sees the shift coming | [report](exp74_report.md) |
| exp77 | Macro Signal integration on Food CPI: live calibrated signals as regime feature for v2 (and v1), 33 signal-window origins, 5-fold CV, arms A/B/C/D + 6 diagnostic rounds | ❌ honest negative, mechanism-blocked: all 33 origins label 'calm', router degenerate in all 5 folds, regime experts collapse to LLM expert → modulation identical up to float noise (diff ~1e-16). Calibration story untested, not refuted: channel closed by uniformly calm window on both real datasets. Reproduction check bit-exact (0.5321591 = exp64) | [report](exp77_report.md) |

## Supporting reports

| Report | What it is |
|---|---|
| [multiseed_report.md](multiseed_report.md) | Multi-seed aggregation (macro track closure: 5-seed, no evidence macro helps) |
| [r2_comparison.md](r2_comparison.md) | R2 PIT-safe rerun comparison |
| [scale_up_report.md](scale_up_report.md) | Scale-up plumbing (10k×66, mock) |
| [scale_up_report_real-chronos2.md](scale_up_report_real-chronos2.md) / [-timesfm25](scale_up_report_real-timesfm25.md) | Scale-up with real TSFM weights |
| [scale_up_report_seed1…4-*.md](scale_up_report_seed1-chronos2.md) | Per-seed scale-up reports: [seed1](scale_up_report_seed1-chronos2.md) [seed2](scale_up_report_seed2-chronos2.md) [seed3](scale_up_report_seed3-chronos2.md) [seed4](scale_up_report_seed4-chronos2.md) (chronos2) · [seed1](scale_up_report_seed1-timesfm25.md) [seed2](scale_up_report_seed2-timesfm25.md) [seed3](scale_up_report_seed3-timesfm25.md) [seed4](scale_up_report_seed4-timesfm25.md) (timesfm25) · [seed1](scale_up_report_seed1-m0w3.md) [seed2](scale_up_report_seed2-m0w3.md) [seed3](scale_up_report_seed3-m0w3.md) [seed4](scale_up_report_seed4-m0w3.md) (m0w3) · [seed2](scale_up_report_seed2-m0w6.md) [seed3](scale_up_report_seed3-m0w6.md) [seed4](scale_up_report_seed4-m0w6.md) (m0w6) · [verify](scale_up_report_verify-m0w3.md) |
| [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md) | Phase-3 adjustments from exp21–23 learnings (news-leading DGP, decision policy) |
| [exp21_BUILD_SUMMARY.md](exp21_BUILD_SUMMARY.md) | exp21 infrastructure build notes |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, weights, and commands to reproduce any experiment |

Ablations (`abl_m0_window.py`, `abl_m3_bound.py`) and `collect_multiseed.py` are
utilities; their outputs feed the reports above. `evidence_corpus/` holds the
labeled evidence used by the news-track experiments — see its
[README](evidence_corpus/README.md).

## The honest-negatives ledger

Killed tracks stay killed only with evidence. The closed ones: R2 macro
(5-seed, no signal), M2/M3/M5 output-wrapper agents, FFORMA (exp42), flexible
regimes (exp45), dense news (exp46), news-for-SMM (exp19), h=3 news (exp25),
kitchen-sink combination (exp51: nonsignificant positives combined are
directionally worse — validates the significance filter).
See [REDESIGN.md](../docs/REDESIGN.md) for what each kill taught us and
[POSITIVE_LIFT_AUDIT.md](../docs/POSITIVE_LIFT_AUDIT.md) for what survived
into the proposed method.
