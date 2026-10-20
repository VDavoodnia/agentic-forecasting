"""exp73: Layer-2 meta-router (analyst toolbox) over ProposedForecasterV2.

On the OSFI M4 runoff benchmark where exp70 failed.

========================================================================
PRE-REGISTRATION (written before any exp73 number was produced)
========================================================================

**Question.** exp70 showed v2 ties naive on OSFI runoff overall and is
significantly WORSE on the uninsured segment (+10.7%, p=0.030), because
(a) the router picks bad experts with no do-no-harm gate, (b) the router
has no channel for exogenous features (rate-level lock-in changed
forecasts by exactly 0.0%), and (c) behaviour is horizon-dependent
(h=3 faint edge p=0.081, h=6 significantly worse). Can a Layer-2
meta-router — a deterministic analyst with a toolbox
(``agent/layer2_tools.py``) and a decision layer
(``agent/meta_router.py``) — fix all three?

**Protocol.** Same benchmark as exp70: OSFI M4 runoff *rate* per segment
(insured / uninsured / variable_insured / variable_uninsured / total),
last 60 rolling origins, 36-month train window. Primary horizon h=1
(exact exp70 comparability: same records, same expert pipeline, v2 via
the same 5-fold blocked CV); secondary h=3 and h=6 with direct per-step
expert grids (the h=1 construction generalised: naive lattice from
``naive_forecasts``, snaive seasonal copies with sqrt-year sigma growth,
exp52 ets/arima direct-step fits). Layer-2 decisions are walk-forward:
the decision at origin o uses only origins < o and data visible at o.

**Arms.**
- A: naive (reference).
- B: v2 alone (exp70 replication — must reproduce exp70's table within
  rounding, else the run is invalid).
- C: v2 + Layer-2 analyst gating only (validation-margin gate +
  changepoint/volatility/coverage vetoes + router-trust audit).
- D: C + lock-in correction (ridge on validation residuals of the gated
  forecast vs the OSFI lock-in level, PIT as-of M+2).
- D2 (researcher-pass variant, added after the smoke test showed the
  unconditional lock-in correction hurts by fitting noise: in-sample
  |r(lock-in, innovation)| <= 0.04 on all segments): D with the
  correction itself evidence-gated — it is applied only if it beats the
  no-correction baseline on a held-out half of the validation window.

**Layer-2 policy (frozen config, ``MetaRouterConfig``).**
val_window=24, min_val=12 (cold start -> naive), margin=2% relative
CRPS, noise_k=1.0 (gain must exceed 1 paired SE), coverage floor 0.5,
vol abstain at vol_ratio>3, router-distrust if v2's validation override
hit rate < 0.5, lock-in ridge lambda=1.0, shift clipped at +-2 resid SD.

**Success bar (pre-registered).**
1. D beats B significantly (paired t, p<0.05) on the pooled h=1
   segment-origins; AND
2. do-no-harm: neither C nor D loses to A on any segment at h=1 by more
   than noise (no segment with Delta > +2% at p<0.10 against).
Secondary: D vs C isolates the lock-in layer; per-horizon tables test
whether Layer-2's per-horizon gating repairs the h=6 blow-up.

**PIT.** OSFI month M observed as-of M+2 (lock-in meter enforces it);
validation windows strictly precede test origins; the leakage-sensitive
paths are unit-tested (tests/test_layer2_tools.py, test_meta_router.py).

Files: exp73_results.json (arm tables + stats), exp73_traces.json
(h=1 decision traces for arms C, D, D2).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp52_vector_baselines as e52  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402
from exp70_osfi_benchmark import (  # noqa: E402
    N_ORIGINS,
    SEGMENTS,
    TRAIN_WINDOW,
    load_runoff_rates,
)

from td_one.agent.layer2_tools import OSFILockInMeter, pinball_crps  # noqa: E402
from td_one.agent.meta_router import MetaRouter, MetaRouterConfig  # noqa: E402
from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import naive_forecasts, series_features  # noqa: E402
from td_one.llm_rungs.combine import _time_block_folds  # noqa: E402

HORIZONS = [1, 3, 6]
Z9 = lfc.Z
DATA_DIR = ROOT / "data" / "osfi"


def expert_grids(train: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """(9, H) expert grids; h=1 is exactly exp70's pipeline."""
    if horizon == 1:
        return lfc.origin_quantiles(train)
    nf = naive_forecasts(train, horizon, period=12)
    naive = np.asarray(nf.naive, dtype=float).T  # (9, H)
    resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
    sigma12 = float(np.std(resid)) if len(resid) > 1 else 1.0
    cols = []
    for s in range(1, horizon + 1):
        val = train[-12 + ((s - 1) % 12)] if len(train) >= 12 else train[-1]
        sig = sigma12 * float(np.sqrt(np.ceil(s / 12)))
        cols.append(val + Z9 * sig)
    snaive = np.stack(cols, axis=1)
    ets = np.stack([e52.fit_ets_like(train, s)[:, 0] for s in range(1, horizon + 1)], axis=1)
    arima = np.stack([e52.fit_arima_like(train, s)[:, 0] for s in range(1, horizon + 1)], axis=1)
    return {"naive": naive, "snaive": snaive, "ets": ets, "arima": arima}


def build_records(values: np.ndarray, horizon: int) -> tuple[list[dict], list[str]]:
    """Per-origin records for one segment at one horizon (exp70-style)."""
    feature_names = lfc.feature_names_for(values)
    records: list[dict] = []
    start = len(values) - N_ORIGINS
    for o in range(start, len(values)):
        if o + horizon > len(values):
            continue
        train = values[o - TRAIN_WINDOW : o]
        if len(train) < TRAIN_WINDOW:
            continue
        actual = values[o : o + horizon]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = series_features(train, period=12)
            qm = expert_grids(train, horizon)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        records.append(
            {
                "origin": int(o),
                "features": [float(fd.get(fn, 0.0)) for fn in feature_names],
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {m: pinball_crps(q, actual) for m, q in qm.items()},
                "actual": actual.tolist(),
            }
        )
    return records, feature_names


def v2_grids(records: list[dict], feature_names: list[str], horizon: int) -> list[np.ndarray]:
    """v2 forecasts per origin via exp70's 5-fold blocked CV (fit per fold)."""
    out: list[np.ndarray | None] = [None] * len(records)

    def factory() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(h=horizon, use_llm_expert=False)
        fc.models = list(lfc.CLASSICAL_MODELS)
        return fc

    for tr_idx, te_idx in _time_block_folds(records, 5):
        fc = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc.fit([records[i] for i in tr_idx], feature_names)
            for i in te_idx:
                r = records[i]
                qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in fc.models}
                feats = dict(zip(feature_names, r["features"], strict=True))
                out_i = fc.predict(np.zeros(30), qm, feats, feature_names)
                out[i] = np.asarray(out_i.quantiles, dtype=float)
    assert all(g is not None for g in out)
    return [g for g in out if g is not None]


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    t, p = ttest_rel(a, b)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_pct": float(100 * (np.mean(a) - np.mean(b)) / np.mean(b)),
        "p": float(p),
        "win_rate": float(np.mean(a < b)),
        "n": int(len(a)),
    }


def run_segment_horizon(
    name: str,
    series: pd.Series,
    horizon: int,
    meter: OSFILockInMeter,
    *,
    d3_momentum: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    values = series.to_numpy(dtype=float)
    months = series.index
    records, feature_names = build_records(values, horizon)
    grids_v2 = v2_grids(records, feature_names, horizon)
    origins = [r["origin"] for r in records]
    actuals = [np.asarray(r["actual"], dtype=float) for r in records]
    candidates: dict[str, list[np.ndarray]] = {
        m: [np.asarray(r["quantiles"][m], dtype=float) for r in records]
        for m in lfc.CLASSICAL_MODELS
    }
    candidates["v2"] = grids_v2

    # PIT lock-in per origin (meter enforces the M+2 publication lag).
    # D3 variant (researcher pass): momentum_3m instead of the level.
    feat_key = "momentum_3m" if d3_momentum else "lockin"
    lockin_by_origin: dict[int, float | None] = {}
    for o in origins:
        f = meter.features(name, pd.Timestamp(months[o]))
        v = f[feat_key] if f["available"] else None
        lockin_by_origin[o] = float(v) if v is not None else None

    res_c = MetaRouter(MetaRouterConfig(use_lockin=False)).run(values, origins, candidates, actuals)
    res_d = MetaRouter(
        MetaRouterConfig(use_lockin=True),
        lockin_fn=lambda o: lockin_by_origin[o],
    ).run(values, origins, candidates, actuals)
    # D2: evidence-gated lock-in (researcher-pass variant — the
    # correction must beat the no-correction baseline on a held-out
    # half of the validation window before it is applied).
    res_d2 = MetaRouter(
        MetaRouterConfig(use_lockin=True, lockin_evidence_gate=True),
        lockin_fn=lambda o: lockin_by_origin[o],
    ).run(values, origins, candidates, actuals)

    arms = {
        "A_naive": np.array(res_c.candidate_crps["naive"]),
        "B_v2": np.array(res_c.candidate_crps["v2"]),
        "C_gated": np.array(res_c.crps["layer2"]),
        "D_gated_lockin": np.array(res_d.crps["layer2"]),
        "D2_gated_lockin_gated": np.array(res_d2.crps["layer2"]),
    }
    stats: dict[str, object] = {
        "segment": name,
        "horizon": horizon,
        "n": len(records),
        "mean_crps": {k: float(np.mean(v)) for k, v in arms.items()},
        "comparisons": {
            "C_vs_A": _paired(arms["C_gated"], arms["A_naive"]),
            "C_vs_B": _paired(arms["C_gated"], arms["B_v2"]),
            "D_vs_A": _paired(arms["D_gated_lockin"], arms["A_naive"]),
            "D_vs_B": _paired(arms["D_gated_lockin"], arms["B_v2"]),
            "D_vs_C": _paired(arms["D_gated_lockin"], arms["C_gated"]),
            "D2_vs_C": _paired(arms["D2_gated_lockin_gated"], arms["C_gated"]),
            "D2_vs_B": _paired(arms["D2_gated_lockin_gated"], arms["B_v2"]),
        },
        "choices_C": res_c.choices,
        "choices_D": res_d.choices,
        "choices_D2": res_d2.choices,
    }
    traces_c = [
        {**t, "month": str(pd.Timestamp(months[o]).strftime("%Y-%m"))}
        for t, o in zip(res_c.traces, origins, strict=True)
    ]
    traces_d = [
        {**t, "month": str(pd.Timestamp(months[o]).strftime("%Y-%m"))}
        for t, o in zip(res_d.traces, origins, strict=True)
    ]
    traces_d2 = [
        {**t, "month": str(pd.Timestamp(months[o]).strftime("%Y-%m"))}
        for t, o in zip(res_d2.traces, origins, strict=True)
    ]
    # keep pooled CRPS vectors for the headline stats
    stats["_arms"] = {k: v.tolist() for k, v in arms.items()}
    return stats, traces_c, traces_d, traces_d2


def main() -> None:
    series_map = load_runoff_rates()
    balances = pd.read_parquet(DATA_DIR / "m4_balances.parquet")
    meter = OSFILockInMeter(balances, pub_lag_months=2)

    results: dict[str, object] = {"segments": {}, "pooled": {}}
    all_traces: dict[str, dict] = {}
    pooled: dict[int, dict[str, list[float]]] = {
        h: {
            "A_naive": [],
            "B_v2": [],
            "C_gated": [],
            "D_gated_lockin": [],
            "D2_gated_lockin_gated": [],
        }
        for h in HORIZONS
    }
    for name in ["total", *SEGMENTS]:
        for h in HORIZONS:
            print(f"[exp73] {name} h={h} ...", flush=True)
            stats, tr_c, tr_d, tr_d2 = run_segment_horizon(name, series_map[name], h, meter)
            results["segments"][f"{name}|h{h}"] = stats  # type: ignore[index]
            if h == 1:
                all_traces[name] = {"C": tr_c, "D": tr_d, "D2": tr_d2}
            for k, v in stats["_arms"].items():  # type: ignore[attr-defined]
                pooled[h][k].extend(v)
            mc = stats["mean_crps"]  # type: ignore[attr-defined]
            print(
                f"    A={mc['A_naive']:.5f} B={mc['B_v2']:.5f} "
                f"C={mc['C_gated']:.5f} D={mc['D_gated_lockin']:.5f} "
                f"D2={mc['D2_gated_lockin_gated']:.5f}",
                flush=True,
            )
    for h in HORIZONS:
        pa = {k: np.array(v) for k, v in pooled[h].items()}
        results["pooled"][f"h{h}"] = {  # type: ignore[index]
            "mean_crps": {k: float(np.mean(v)) for k, v in pa.items()},
            "D_vs_B": _paired(pa["D_gated_lockin"], pa["B_v2"]),
            "D_vs_A": _paired(pa["D_gated_lockin"], pa["A_naive"]),
            "D2_vs_B": _paired(pa["D2_gated_lockin_gated"], pa["B_v2"]),
            "D2_vs_A": _paired(pa["D2_gated_lockin_gated"], pa["A_naive"]),
            "D2_vs_C": _paired(pa["D2_gated_lockin_gated"], pa["C_gated"]),
            "D_vs_C": _paired(pa["D_gated_lockin"], pa["C_gated"]),
            "C_vs_B": _paired(pa["C_gated"], pa["B_v2"]),
            "C_vs_A": _paired(pa["C_gated"], pa["A_naive"]),
        }

    out_path = Path(__file__).parent / "exp73_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    traces_path = Path(__file__).parent / "exp73_traces.json"
    with open(traces_path, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)
    print(f"\nwrote {out_path} and {traces_path}")

    h1 = results["pooled"]["h1"]  # type: ignore[index]
    d_vs_b = h1["D_vs_B"]
    print(
        f"[exp73] pooled h=1: D {h1['mean_crps']['D_gated_lockin']:.5f} vs "
        f"B {h1['mean_crps']['B_v2']:.5f} "
        f"({d_vs_b['diff_pct']:+.1f}%, p={d_vs_b['p']:.3g})"
    )
    ok = d_vs_b["p"] < 0.05 and d_vs_b["mean_a"] < d_vs_b["mean_b"]
    print(f"[exp73] success bar part 1 (D beats B, p<0.05): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
