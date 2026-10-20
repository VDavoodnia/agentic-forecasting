# exp73: Layer-2 Meta-Router (Analyst Toolbox) — Report

**Result: ❌ HONEST NEGATIVE on the pre-registered bar; partial positive for the gating layer.**
Layer-2's validation gating ties naive and repairs v2's documented
harms, but the exogenous lock-in correction adds nothing through any of
three tested variants. The proposed method's real-data problem stands;
the analyst layer is the do-no-harm wrapper, not the fix.

## Background

exp70: v2 ties naive on OSFI runoff (total −0.2%, p=0.54) and is
*significantly worse* on the uninsured segment (+10.7%, p=0.030); the
v2 router has no do-no-harm gate and no exogenous channel. exp71:
competitor spread, zero effect. exp73 builds the Layer-2 meta-router the
user asked for: a deterministic **analyst with tools**, not just a gating
rule.

## What was built

**`src/td_one/agent/layer2_tools.py`** — deterministic, PIT-safe,
JSON-traceable analyst toolbox (reusable by other tracks):
`series_diagnostics`, `cusum_break_pvalue`, `changepoint_scan`
(random-walk-safe: autocorrelation-corrected level-shift z + jump and
trend channels in innovations), `expert_backtest` / `summarize_backtest`,
`router_audit`, `interval_compare`, `lockin_level` / `OSFILockInMeter`
(as-of M+2 lock-in proxy from OSFI new-lending rates), `macro_context`
(Macro Signal Service signals, competitor spread, house-price momentum —
gracefully degrades when a source is absent).

**`src/td_one/agent/meta_router.py`** — walk-forward Layer-2 stage over
{naive, snaive, ets, arima, v2, blend(v2, naive)}. Pre-registered policy:
cold start / changepoint / volatility abstention → naive; validation
winner needs a 2% relative margin AND to beat the paired noise floor;
v2-distrust if its validation override hit rate < 0.5; coverage veto.
Lock-in arm: ridge (intercept + one slope) of gated-forecast step-1
residuals on lock-in, fit on validation only, clipped at ±2 resid SD.
Every origin emits a machine-readable decision trace. 32/32 new tests
green; ruff + strict mypy clean.

## Experiment

OSFI runoff-rate, 5 segment-units × 60 rolling origins (36m train),
h=1 primary (exact exp70 records/folds), h=3/6 direct-step secondary.
Arm B replication verified: B matches exp70's table to the 5th decimal
on every segment (total 0.00207, uninsured 0.00262/0.00236, ...).

### Pooled h=1 (n=300 segment-origins)

| Arm | CRPS | vs A | vs B |
|---|---|---|---|
| A naive | 0.00369 | — | — |
| B v2 | 0.00381 | +3.2% (n.s.) | — |
| C gating | 0.00374 | +1.2% (p=0.081) | −1.9% (p=0.31) |
| D lock-in | 0.00409 | **+10.7% (p=0.006)** | +7.3% (p=0.15) |
| D2 gated lock-in | 0.00384 | +4.0% (n.s.) | +0.8% (p=0.71) |

Per segment h=1 (C vs A / D2 vs B): total −0.3%/+1.2%, insured
+1.8%/−1.5%, uninsured +0.8%/−5.4%, variable_insured +1.2%/+1.7%,
variable_uninsured +2.1%/+6.4% — none significant.

### Horizons (pooled)

- **h=3: C beats B significantly** (−6.8%, p=0.015, win 0.46); D2 vs B
  −4.7% (p=0.10). The per-horizon gating repaired a real v2 deficit.
- h=6: C 0.00589 vs A 0.00578 (+1.8%, n.s.) — exp70's +24% v2 blow-up
  is gone; Layer-2 abstained its way out of it.

## Verdicts

**Pre-registered success bar: FAIL.** D does not beat B (D +7.3% vs B,
p=0.15; worse, D violates do-no-harm against A at +10.7%, p=0.006).

**Gating layer (C): partial positive, kept.** It does what it was built
for — it ties naive (no segment violates the harm bar) and *repairs*
v2's documented harms: uninsured +10.7%→+0.8%, the h=6 blow-up→+1.8%
(n.s.), and it significantly beats v2 at h=3. The verdict: Layer-2
gating is the do-no-harm wrapper that makes v2 deployable on
single-series targets — it earns its keep as *safety*, not as accuracy.

**Lock-in correction: dead on this target.** The unconditional ridge
hurts (fits noise: in-sample |r(lock-in, innovation)| ≤ 0.043 on all
segments, n=83). The researcher-pass variant D2 (evidence-gated:
correction applied only if it beats the no-correction baseline on a
held-out half of validation) correctly stays silent (shifts on 11/60
origins) and lands at D2≈C≪D. The momentum_3m variant (D3, weak
in-sample r=−0.166) also fails to beat C (+1.6%, p=0.10). Combined with
exp71's competitor-spread collapse, **two independent exogenous
channels are now empty on aggregate monthly runoff** — the signal is
not there, not the design.

## Worked traces (h=1)

**1. Validation winner — total, 2022-08 (arm C picks arima).**
Validation CRPS: naive 0.00168, arima 0.00162, v2 0.00168; rel. gain
3.6% > 2% margin, above noise floor → arima. Layer-1 audit: no overrides
in window (hit rate n/a). No changepoint.

**2. Harm blocked — uninsured, 2022-08 (margin_gate).** v2's grid was
non-naive (it would have been the +10.7%-type pick), but validation
says naive 0.00234 < v2 0.00234±noise → the 2% margin is not met →
naive. This is the exact failure exp70 diagnosed, now caught
mechanically.

**3. Evidence gate trade-off — total, 2022-08 (D vs D2).** D's
validation fit gave slope −0.000374, shift −0.00109; D2's held-out
check (mse_with 7.8e-06 ≥ mse_without 6.07e-06) blocked it. On this one
origin the shift would have helped (D 0.00084 vs D2 0.00142) — the gate
correctly optimises for the average case at the cost of occasional
forgone wins.

Full traces: `exp73_traces.json` (h=1, 5 segments × 60 origins × C/D/D2).

## Verdict on Layer-2 as permanent architecture

**Yes for the analyst-gating layer, no for the exogenous correction.**
The toolbox + walk-forward gating is a genuine architectural addition:
it converts v2's "occasionally much worse than naive" into "≈naive,
never much worse" on single-series targets, and it preserves v2's
upside where validation evidence is real (h=3). The lock-in correction
stays a null layer until an exogenous proxy with in-sample signal
appears. The `OSFILockInMeter` and `macro_context` tools remain useful
evidence sources for the analyst bundle regardless.

Files: `exp73_metarouter.py`, `exp73_results.json`, `exp73_traces.json`,
`src/td_one/agent/layer2_tools.py`, `src/td_one/agent/meta_router.py`,
`tests/test_layer2_tools.py`, `tests/test_meta_router.py` (32 new).
