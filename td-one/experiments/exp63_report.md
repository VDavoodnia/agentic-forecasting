# Experiment 63 — LLM as forecast combiner vs statistical router

**Date:** 2026-10-18 | **Status:** complete (honest negative)

## Question

Does an LLM (gpt-oss-120b, low reasoning) add anything on top of the
statistical regime router when combining the 7-model tourism zoo?

## Method

- 60 tourism series subsample (fixed seed 63), 5-fold CV by series.
- Per test series, the LLM receives the **union** of what the router sees
  (10 diagnostic features) and what best-single sees (validation-fold mean
  CRPS per model) — the most generous setup, so a null result is conclusive.
- LLM outputs JSON weights over the 7 models (json_mode, 0 parse failures
  in 60 calls). Parse failures would fall back to best-single.
- PIT-safe: LLM sees train-window features, validation CRPS, and test-origin
  quantile forecasts only. Never test actuals.

## Results — mean CRPS (n=60)

| method | mean CRPS |
|---|---|
| proposed (statistical router) | **0.0992** |
| best single (argmin validation CRPS) | 0.1005 |
| **LLM-weighted** | **0.1169** |
| equal weights | 0.1586 |

- LLM vs proposed: Δ = +0.0177, t = +6.52, **p = 1.7e-08 against the LLM**,
  win rate 13.3%.
- LLM vs best single: Δ = +0.0165, t = +6.08, **p = 9.4e-08 against the LLM**,
  win rate 15.0%.

## Verdict

**Honest negative.** The LLM is significantly *worse* than both the
statistical router and the naive "pick the validation winner" baseline,
despite having strictly more information than either. It does not merely
fail to add value — it subtracts it, likely by over-thinking: spreading
weight across models instead of trusting the validation signal.

This extends the agentic-track honest negatives (exp23: model selection =
random; exp24: regime = wash; exp25: news worse at h=3). The statistical
router's constrained priors beat LLM judgment on this task.

## Repro

`.venv/bin/python experiments/exp63_llm_selector.py` (60 Groq calls, low
reasoning; restart-safe via `exp63_partial.jsonl`). Results:
`experiments/exp63_results.json`.
