# TD-One: Agentic Mortgage Attrition Forecasting

Solo implementation for the Vector Agentic Forecasting Bootcamp by Vandad Davoodnia.

We forecast borrower attrition at Canadian mortgage renewal: for each portfolio segment and month, how many renewing borrowers switch lenders vs stay? We compare a 4-rung modeling ladder — Bayesian baseline (M0), zero-shot TSFMs (M1), ReAct LLM agent (M2), hybrid (M3) — plus agentic extensions (M4 regime reader, M5 router) under strict point-in-time evaluation.

**Start here:** [docs/README.md](docs/README.md) (doc index) → [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md)

**Original plan:** [PLAN_SOLO.md](PLAN_SOLO.md) (the solo-build plan this implementation follows).

## Quickstart

```bash
cd td-one
uv sync
uv run pytest tests/ -q
```

Run an experiment:
```bash
uv run python experiments/exp16_freddie_smm.py
```

## Setup & installation

**Prerequisites:** Python 3.12+, `uv`.

```bash
# 1. Clone and enter
git clone https://github.com/VDavoodnia/agentic-forecasting.git
cd agentic-forecasting/td-one

# 2. Install dependencies
uv sync

# 3. Download TSFM weights (never committed; ~1.3GB)
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
uv run --with huggingface_hub python scripts/bootstrap_models.py

# 4. (Optional) Fetch FRED data — needs API key via vault connector
uv run python scripts/fetch_fred.py

# 5. (Optional) Build news corpus — GDELT is free, Perplexity needs credit
uv run python scripts/fetch_news_gdelt.py --year 2025
uv run python scripts/fetch_news_perplexity.py --year 2025

# 6. Verify
uv run pytest tests/ -q
```

See [experiments/REPRODUCIBILITY.md](experiments/REPRODUCIBILITY.md) for full reproduction steps.

## Repository map

| Path | What it is |
|---|---|
| [docs/](docs/) | All documentation. Start with [docs/README.md](docs/README.md). |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Methodology reference (problem, ladder, PIT, metrics). |
| [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) | Final results: all tracks, exp1–exp88 era. |
| [docs/REDESIGN.md](docs/REDESIGN.md) | Design decisions: what we keep/kill/try next. |
| [docs/DIAGRAMS.md](docs/DIAGRAMS.md) | Architecture diagrams. |
| [docs/plots/](docs/plots/) | Result plots — regenerate via `docs/plots/make_plots.py`. |
| [experiments/](experiments/) | All experiments (exp1–exp88). See [experiments/README.md](experiments/README.md). |
| [src/td_one/](src/td_one/) | Main package (see structure below; each subpackage has a README). |
| [scripts/](scripts/) | Data fetchers, model bootstrap, pipelines. |
| [configs/](configs/) | Run configs (reserved). |
| [tests/](tests/) | TDD test suite. |
| [benchmarks/](benchmarks/) | Real-data benchmarks: BoC, WTI, food CPI, M4, SMM. |
| [audits/](audits/) | News leakage + content audits. |
| [models/](models/) | Weight manifest + weights dir (gitignored). |
| [data/](data/) | Data dirs (gitignored) + per-dir READMEs. |

## Package structure

- `src/td_one/`
  - `data_gen/` — synthetic mortgage simulator (history + structural outcome DGP)
  - `data_service/` — point-in-time DuckDB store
  - `freddie/` — Freddie Mac loan-level loader (SMM computation)
  - `fred/` — FRED live client + vintage-safe loader
  - `news/` — news store (SQLite FTS5), RAG (BM25 → Groq rerank → abstain), Perplexity client
  - `tools/` — agent tools: data_segment, evidence, forecasting
  - `models/` — M0 baseline, TSFM wrappers (+ calibration), macro overlays, hybrid readout
  - `agent/` — ReAct loop, regime reader (M4), ensemble router (M5), Groq client, retry
  - `evaluation/` — metrics (CRPS/MAE/coverage), leakage auditor, backtest runner
  - `segments/` — segment discovery (PIT-safe panels)

## Key design decisions

- **LLM**: Groq `openai/gpt-oss-120b` (via vault surrogate, never in code). The LLM never outputs numbers — only structure (classes, weights, confidences).
- **FRED**: Live via vault connector (`custom.fred`); bulk cache in `data/fred/` via `scripts/fetch_fred.py`.
- **TSFMs**: Chronos-2 / TimesFM-2.5 (CPU-compatible). [LogScaleCalibrator](src/td_one/models/tsfm_calibration.py) fixes near-zero miscalibration.
- **News**: Offline-first SQLite FTS5 KB. GDELT (free breadth) + Perplexity (depth). RAG: BM25 → Groq rerank → abstain.
- **Point-in-time**: All queries enforce `record_asof <= run_date`; leakage auditor verifies 0%.

## Current status

| Track | Verdict |
|---|---|
| Proposed method (exp47) | ✅ Alive — beats equal weights (0.1451 vs 0.1507, p=6.5e-09), best single, and exp43 router; confirmed on Tourism (exp60, p=1.9e-68) and M3 (exp61, p=2.7e-44) |
| LLM integration ladder (exp54–58) | ⚠️ Partial — rung 5 hybrid 0.4289 beats Vector's LLM 0.5688 by 24.6% (paired p=0.009) on Food CPI, but loses to its own best LLM expert (llm-promptV1 0.4262, p=0.058); rung 3 calibrator ❌ killed; guardrails reproduce exp49 bit-exact |
| exp43 regime router | ✅ Partial positive — first to beat equal weights (M4 0.1282 vs 0.1295, p=0.018) |
| Conformal M0 | ✅ Strong baseline — nothing beats it cleanly on synthetic |
| Uncertainty layer (exp74–76) | ✅ CQR on the TSFM quantile grid is the default UQ layer (coverage→nominal, no CRPS cost); ACI regime-conditional; story-feature uncertainty meta-model real (exp76 PIT-corrected re-measurement: −2.8% vs numeric-only, p=1.9e-06 — exp74's p=2e-11 was leak-inflated and is retired) |
| Pooled conformal banks (exp81) | ✅ Ships, family-gated — signed-score shrinkage banks (n0=24) activate on families with ≥8 series: M4 conformal no-op → cov80 0.805; Tourism CRPS −1.0% and cov80 0.806. Pooling contraindicated on OSFI (4 heterogeneous segments) — per-series CQR kept there |
| Segment scout (exp88) | ✅ Bar passed — LLM proposes segment rules from feature statistics + policy/news context: planted-cohort recovery ARI 0.204/0.454/0.424 (seeds 7/42/123) vs SHAP-clustering incumbent 0.091; planted low_rate rule recovered verbatim by the flash variant. SHAP context hurts the proposer (0.052) and outcome-rate aggregates fail (0.077) — segments come from the domain prior |
| Manager risk reports (exp87) | ✅ Bar passed — reporter runs end-to-end with real news: OSFI + Freddie SMM monthly reports, citation contract 100%, news citations in 100% of flagged explanations, judged relevance 100% (19 RELEVANT + 1 PARTIAL / 20); retrieval density + jurisdiction-scoped queries are the operative levers |
| Competitor spread, loan level (exp84) | ✅ Bar passed — the exp71b aggregate negative reverses in the SFLLD cross-section: origination spread vs PMMS predicts 24m prepayment, same positive sign in all 3 eras (+0.56/+0.49/+1.06 per 100bp), pooled cross-era OOS Δlog-loss +0.056; robust across horizons, references, and vintage FE. Signal's home is loan-level attrition/SMM features |
| Integration (exp72) | ⚠️ Mixed — Track-2 shifts stress-test Track-1 groups: Q1 regional-isolation not supported (p=0.22, 5 seeds; oracle geography identical — the limit is the forecaster); Q2 fallback a working but blunt safety valve; Q3 Layer-2 protects in the thin-group regime (p<1e-8). Flagged: v2_forecaster_fn emits flat point-mass quantiles |
| Group forecaster (exp78–80) | ⚠️ Fixed + settled — point-mass defect root-caused and fixed (exp78: Gaussian spread, cov80 0.075→0.714); trailing-mean center ships, the real fitted v2 router loses inside the group forecaster (exp79); thin/dead fallback split ships (exp80b: dead groups use their own mean, pooled CRPS improves in every world, p≤0.008); CQR conformal range layer available but NOT the default (exp80a: nominal control coverage, S1 fresh-seed replication failed its pre-registered bar — structural bank-blindness) |
| Chronos-2 as segment engine (exp86) | ❌ Honest negative — zero-shot Chronos-2 per segment loses the member-weighted endpoint to the shipped device (protocol 0.09024 vs 0.08849; fresh seeds tie-to-lose; its unweighted protocol win was seed noise). It wins only on aggregate-smooth series (portfolio scope, fallback cells): TSFM strength is scale-dependent. exp81 pooled conformal over Chronos replicates as a coverage repair (cov80 0.82–0.83 vs 0.76–0.77) |
| TSFMs | ✅ Beat m0 30% on 63-mo real SMM (exp18, p=0.0012); beat classical ~20% on M4 |
| Macro (FRED overlay) | ❌ Closed — 5-seed, no evidence macro helps (classical or TSFM) |
| M5 router | ❌ Dead — loses to inverse-CRPS even constrained |
| M2 ReAct / M3 hybrid | ❌ Dead — LLM numerical output is net-negative |
| News as feature (monthly) | ❌ Dead — exp46 definitive negative; exp19 frozen |
| Reasoning effort | ❌ "High" never helps — default to "low" for diagnostics (exp48) |

See [docs/REDESIGN.md](docs/REDESIGN.md) for the full kill/keep list with evidence.
