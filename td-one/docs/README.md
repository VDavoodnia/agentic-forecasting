# TD-One Documentation Index

Start here. This page maps every document in `td-one/docs/` — what it is, who it's for, and where to go next. Everything under `td-one/` is self-contained: no document here depends on anything outside `td-one/`.

**Cleanup 2026-10-20:** every doc now carries a status stamp — `CURRENT` (living, trust it), `HISTORICAL` (dated point-in-time record, kept in place for provenance), or `ARCHIVED` (superseded, frozen in [archive/](archive/)). If a doc has no stamp, it predates the cleanup — treat its claims accordingly.

## Start with these (5 min)

| Doc | What you get |
|---|---|
| [FINAL_DELIVERY.md](FINAL_DELIVERY.md) | **The final delivery note (2026-10-20).** What the finished package is, the final architecture in one paragraph, the headline results, what shipped vs what was killed, and pointers to everything below. |
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | Final numbers only — no experiment codes, human-readable names, benchmarks in tables. Includes Layer-2 (exp73), the uncertainty layer through pooled banks (exp74–exp81), and the settled group forecaster (exp78–exp80). |
| [PHASE2_UPDATE.md](PHASE2_UPDATE.md) | Since exp63: Layer-2 gating, CQR uncertainty, red-team, OSFI benchmark, Macro Signal Service — mortgage-framed. All experiments landed (through exp81). |
| [METHODOLOGY.md](METHODOLOGY.md) | The full methodology: problem, glossary, all tracks (exp1–exp63 + LLM ladder). The reference spine. |
| [PROPOSED_METHOD.md](PROPOSED_METHOD.md) | The exp47 method: architecture, implementation (`src/td_one/agent/proposed_method.py`), verdict. |
| [MACRO_SIGNAL.md](MACRO_SIGNAL.md) | Macro Signal Service: LLM-read monthly categorical macro signals, calibrated against realized data. |

## Presentations

| Doc | What you get |
|---|---|
| [slides/TD-One-Slides.html](slides/TD-One-Slides.html) | The Final Report (HTML, deck of record): 8 exec slides + the full paper (general architecture, datasets, models, per-dataset final results, visualizations) + Appendices A–D (ablations, prompts and parameters, glossary, failed experiments). Final 2026-10-20 — all experiments landed. |
| [slides/TD-One-Context.md](slides/TD-One-Context.md) | Full context for continuing the deck work: ground truth, verdicts, messaging rules, what changed, open work. |

## Current program docs

| Doc | What you get |
|---|---|
| [LOAN_ATTRITION_PLAN.md](LOAN_ATTRITION_PLAN.md) | Active — loan attrition program: Track 1 (grouping), Track 2 (TSFM red-team exp67 ✅ + guard exp68 ❌ + anticipatory exp69/69b/69c ❌), integration (exp72), with verdicts appended per track. |
| [REAL_DATA_INTEGRATION.md](REAL_DATA_INTEGRATION.md) | Real-data integration: OSFI M4 runoff benchmark (exp70 ❌), competitor spread (exp71 ❌, exp71b ❌ settled at full coverage: BoC 622m + Wayback 54m), FRED adds, Layer-2 meta-router (exp73: gating partial positive, lock-in ❌). |
| [REDTEAM_TSF.md](REDTEAM_TSF.md) | TSFM red-team protocol (exp67): 7-shift loan taxonomy, breaking-point design, results summary. Status: complete (2026-10-20). |
| [UQ_SURVEY.md](UQ_SURVEY.md) | Uncertainty quantification for TSFMs (2024+): conformal-for-time-series families, TSFM calibration, error-prediction meta-models; shortlist behind exp74. |
| [SELF_AUDIT.md](SELF_AUDIT.md) | Adversarial self-review: multiple comparisons, M4 development bias, BoC-discrete downgrade, PIT re-verification, CRPS verification. Headlines survive; margins disclosed. |
| [DIAGRAMS.md](DIAGRAMS.md) | Architecture and experiment diagrams (mermaid — renders on GitHub). |
| [../experiments/README.md](../experiments/README.md) | Full experiment index (exp1–exp81) with verdicts and report links. |
| [../experiments/REPRODUCIBILITY.md](../experiments/REPRODUCIBILITY.md) | Reproduce any experiment: environment, weights, commands. |

## The story (dated finals — 2026-10-19 close-outs)

| Doc | What you get |
|---|---|
| [reports/MANAGER_REPORT.md](reports/MANAGER_REPORT.md) | Plain-language project summary with glossary, diagrams, and tables. **Final, 2026-10-19.** Phase-2 continuation: [PHASE2_UPDATE.md](PHASE2_UPDATE.md). |
| [reports/FINAL_SYNTHESIS.md](reports/FINAL_SYNTHESIS.md) | Technical synthesis across all datasets and tracks (exp1–exp63). **Final, 2026-10-19.** Phase-2 continuation: [PHASE2_UPDATE.md](PHASE2_UPDATE.md). |

## Historical records (dated, kept in place for provenance)

Point-in-time audits, surveys, and design records. Each is stamped `HISTORICAL` with its date; none is the current reference for anything — they are the evidence trail behind the current docs.

| Doc | Dated | What it is |
|---|---|---|
| [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) | 2026-10-16 | TSFM + agentic forecasting literature with verified citations; base for the redesign. |
| [CITATION_AUDIT.md](CITATION_AUDIT.md) | 2026-10-16 | Literature claims in the survey checked against primary sources. |
| [CRITIC_SYNTHESIS.md](CRITIC_SYNTHESIS.md) | 2026-10-15 | 6-angle critic panel findings, rounds 1–2. |
| [DATA_AUDIT.md](DATA_AUDIT.md) | 2026-10-16 | Data engineering audit: 2 P0, 5 P1, 5 P2 + fixes. |
| [REDESIGN.md](REDESIGN.md) | 2026-10-16 | Methodological redesign: what we keep, what we kill, what we try next (era: exp5–exp11). Current direction: PHASE2_UPDATE.md. |
| [REAL_DATA_SURVEY.md](REAL_DATA_SURVEY.md) | 2026-10-16 | Real dataset candidates for validation. Acquisition record: REAL_DATA_INTEGRATION.md. |
| [RAG_DESIGN.md](RAG_DESIGN.md) | 2026-10-16 | News RAG design: offline-first KB, BM25 → Groq rerank → abstain; recall@k/precision@k eval. |
| [SCALE_UP_RUNBOOK.md](SCALE_UP_RUNBOOK.md) | 2026-10-16 | Early pipeline runbook (simulate → PIT → segments → backtest). Reproduction now: experiments/REPRODUCIBILITY.md. |
| [AGENTIC_ROADMAP.md](AGENTIC_ROADMAP.md) | 2026-10-17 | The approved roadmap from output-layer wrappers to true agents. Trilogy complete (exp42–44), proposed method shipped (exp47). |
| [AGENT_AUDIT_M4.md](AGENT_AUDIT_M4.md) | 2026-10-16 | Regime reader (M4) hardening audit: v2 prompt, ABSTAIN, counterargument. |
| [AGENT_AUDIT_M5.md](AGENT_AUDIT_M5.md) | 2026-10-16 | Router (M5) post-mortem: why v1 failed, v2 design, kill decision. |
| [ARGON_REVIEW_2.md](ARGON_REVIEW_2.md) | 2026-10-17 | Second-pass review of the external auditor's 6 findings: what's addressed, where it's mistaken, what we missed. |
| [POSITIVE_LIFT_AUDIT.md](POSITIVE_LIFT_AUDIT.md) | 2026-10-18 | Inventory of every positive result (exp1–exp46 + benchmarks) used to design the proposed method. |
| [NONSIGNIFICANT_POSITIVES.md](NONSIGNIFICANT_POSITIVES.md) | 2026-10-18 | Audit of every positive-but-nonsignificant effect (exp1–exp50) with combinability verdicts; evidence base for exp51. |
| [VECTOR_METHODOLOGY_AUDIT.md](VECTOR_METHODOLOGY_AUDIT.md) | 2026-10-18 | Every methodological difference vs Vector's stack (CRPS scorer, quantile grids, LLM config, baselines); evidence base for exp52/exp53. |
| [TOOL_AUDIT.md](TOOL_AUDIT.md) | 2026-10-18 | Tool call frequency, patterns, failures across exp21–exp47 and benchmarks. |
| [TOOL_OPTIMIZATION.md](TOOL_OPTIMIZATION.md) | 2026-10-18 | Tool recommendations from the audit: which tools to add, drop, fix. |
| [TOOL_USAGE_DATA.md](TOOL_USAGE_DATA.md) | 2026-10-18 | Runtime tool-call data from `@tracked()` instrumentation. |
| [PROMPT_LEAKAGE_AUDIT.md](PROMPT_LEAKAGE_AUDIT.md) | 2026-10-18 | Point-in-time proofs for the LLM prompt variants (generic / V1 words / V2 macro context). |
| [LLM_INTEGRATION_LADDER.md](LLM_INTEGRATION_LADDER.md) | 2026-10-19 | The 6-rung LLM integration ladder (exp54–59 + exp64): design and verdicts. Results: FINAL_RESULTS.md. |
| [GROUPING_EXAMPLES.md](GROUPING_EXAMPLES.md) | 2026-10-19 | Worked examples for the grouping track (exp66d): 4 cohorts end-to-end — rule → members → forecast → risk → citation-grounded explanation, incl. the failure case. |

## Plans

| Doc | Status |
|---|---|
| [LOAN_ATTRITION_PLAN.md](LOAN_ATTRITION_PLAN.md) | Active (see Current program docs) |
| [plans/](plans/) | Folder of dated plans; see its [index](plans/README.md). Includes [plans/MULTI_DATASET_PLAN.md](plans/MULTI_DATASET_PLAN.md) (2026-10-17: real-data benchmarking plan). |

## Archive (superseded — frozen, do not revive)

[archive/](archive/) holds docs whose content is fully covered by newer docs. Kept for provenance only:

| Doc | Why archived |
|---|---|
| [archive/RESULTS.md](archive/RESULTS.md) | R2-era results registry (status 2026-10-16). Superseded by FINAL_RESULTS.md + PHASE2_UPDATE.md. |
| [archive/RESULTS_ARCHIVE.md](archive/RESULTS_ARCHIVE.md) | exp1–exp15 results (already self-marked superseded). Current results: FINAL_RESULTS.md. |
| [archive/FULL_STORY.md](archive/FULL_STORY.md) | 2026-10-17 narrative capstone. Superseded by reports/FINAL_SYNTHESIS.md + PHASE2_UPDATE.md. |
| [archive/MANAGER_REPORT.md](archive/MANAGER_REPORT.md) | Stale 2026-10-17 draft. The final is reports/MANAGER_REPORT.md (2026-10-19). |
| [archive/FINAL_SYNTHESIS.md](archive/FINAL_SYNTHESIS.md) | Stale 2026-10-17 living-draft. The final is reports/FINAL_SYNTHESIS.md (2026-10-19). |

## Operations

| Doc | What you get |
|---|---|
| [AGENTS.md](AGENTS.md) | Working conventions for agents in `td-one/`: TDD, PIT-safety gate, metrics, honesty policy. |
| [GUIDELINES.md](GUIDELINES.md) | Contribution guidelines for this project. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute: branches, commits, PRs. |

## Data

| Asset | Location | Notes |
|---|---|---|
| News corpus | `../data/news/` (gitignored) | Articles 2025–2026, FTS5 SQLite. Enriched on every fetcher hit. |
| SMM series | `../data/freddie/smm_monthly_2021_2026.csv` | 63 months of Freddie Mac prepayment rates. |
| FRED cache | `../data/fred/` (gitignored) | 19 series. Rebuild via `../scripts/fetch_fred.py`. |
| M4 data | `../data/m4/` | M4 competition series used by the agentic track. |
| Plots | [plots/](plots/) | Regenerated, embedded figures — see its README for sources. |
