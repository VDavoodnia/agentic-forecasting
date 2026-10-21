# TD-One Plots

> **Status: CURRENT — living plot index (generator: `make_plots.py`)**

Figures for the docs. **Every plot is generated, never hand-drawn** — rerun the
generator after any result it depicts changes.

**Repo state (2026-10-20):** 3 of the 6 generated plots are committed here
(`rag_recall_precision.png`, `agent_m4_m5.png`, `smm_2021_2026.png`).
`trilogy_m4.png`, `reasoning_sweep.png`, and `tool_usage.png` are NOT
currently in the repo — rerun the generator below to recreate them. Until
then, the embeds of those three in `docs/TOOL_AUDIT.md` and
`docs/archive/RESULTS.md` (both frozen records) render as broken images.

## Regenerate

From `td-one/`:

```
.venv/bin/python docs/plots/make_plots.py
```

Shared style (dpi=150, tight bbox, grid, source caption on every figure) is set
in `make_plots.py::setup()`. Keep it that way — consistency is the point.

## The plots

| File | Shows | Source |
|---|---|---|
| `rag_recall_precision.png` | RAG recall@k / precision@k (BM25, n=100) | `experiments/rag_eval_results.json` |
| `agent_m4_m5.png` | M4 regime-reader beats M0; M5 router loses to inv-CRPS | `exp11`/`exp12` result JSONs |
| `smm_2021_2026.png` | Freddie Mac SMM, 63 months | `data/freddie/smm_monthly_2021_2026.csv` |
| `trilogy_m4.png` | exp42/43/45/47 on M4 vs equal weights vs best single | `exp42/43/45/47_results.json` |
| `reasoning_sweep.png` | exp48: accuracy + latency by reasoning level | `experiments/exp48_report.md` |
| `tool_usage.png` | Tool call frequency vs latency (138 calls) | `docs/TOOL_USAGE_DATA.md` |

## Where they're embedded

- `trilogy_m4.png` → `docs/archive/RESULTS.md` (agentic track section)
- `rag_recall_precision.png` → `docs/RAG_DESIGN.md`
- `agent_m4_m5.png` → `docs/archive/RESULTS.md` (M4/M5 section)
- `smm_2021_2026.png` → `docs/archive/RESULTS.md` (real-data section)
- `reasoning_sweep.png` → `experiments/exp48_report.md`
- `tool_usage.png` → `docs/TOOL_AUDIT.md`
