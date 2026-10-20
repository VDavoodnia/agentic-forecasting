"""exp82 diagnosis: why the pooled router's center loses.

R1 (pre-registered arms) showed the pooled router center loses to the
trailing mean by about the same margin as exp79's per-series router —
so record starvation is not the mechanism. This script interrogates
the fitted pooled router on one seed's panels:

1. Regime/expert structure: which regimes fire, which expert is
   selected per regime, and does the do-nothing gate trigger?
2. Center quality: pooled-router median vs trailing-6 mean as pure
   point forecasts (MAE / bias vs the realized next value), by world.
3. Scale domination check: the router picks the per-regime expert by
   MEAN raw CRPS over pooled records. Records come from series with
   heterogeneous levels, so large-scale series dominate that mean.
   Compare the pooled winner per regime against (a) the winner under
   per-record CRPS normalized by the naive expert's CRPS (scale-free)
   and (b) the per-series majority winner.

Usage:
    .venv/bin/python experiments/exp82_diag.py [--seed 42]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.group_forecaster import _PooledRouter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
MODELS = ("naive", "ets", "arima")


def diagnose(seed: int, world: str) -> None:
    sd = exp72.build_seed_data(seed, "tree")
    groups = exp72.build_world(sd, world)
    gids = [g for g in groups if g != "__portfolio__"]
    panel = {g: np.array(groups[g].attrition_rate, dtype=float) for g in [*gids, "__portfolio__"]}
    core = _PooledRouter(panel)
    n = len(sd.months)
    origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))

    print(f"\n== seed {seed} world {world} ==")
    # --- 1. regime / expert structure at the last origin ---------------
    last = origins[-1]
    fc = core.router(panel["__portfolio__"][:last])
    print("expert_per_regime (pooled raw-CRPS winner):", fc._expert_per_regime)
    print(
        "global_best_expert:",
        fc._global_best,
        "| gate fires (backtest):",
        getattr(fc, "_gate_fired_backtest", "n/a"),
    )

    records, by_series = core._records(last)
    labels = [
        fc.regime_spec.label_fn(dict(zip(core._feature_names, r["features"], strict=True)))
        for r in records
    ]
    print("record regime mix:", dict(Counter(labels)), f"(n={len(records)})")

    # scale-free expert winners per regime: normalize each record's
    # CRPS by its naive CRPS, then average.
    for regime in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == regime]
        raw = {m: float(np.mean([records[i]["crps"][m] for i in idx])) for m in MODELS}
        norm = {
            m: float(
                np.mean(
                    [records[i]["crps"][m] / max(records[i]["crps"]["naive"], 1e-12) for i in idx]
                )
            )
            for m in MODELS
        }
        print(
            f"  regime {regime} (n={len(idx)}): raw winner "
            f"{min(raw, key=raw.get)} { {k: round(v, 4) for k, v in raw.items()} } | "
            f"scale-free winner {min(norm, key=norm.get)} "
            f"{ {k: round(v, 3) for k, v in norm.items()} }"
        )

    # per-series winners (raw, own records only)
    per_series_winners: Counter = Counter()
    for _key_bytes, recs in by_series.items():
        labs = [
            fc.regime_spec.label_fn(dict(zip(core._feature_names, r["features"], strict=True)))
            for r in recs
        ]
        for regime in set(labs):
            idx = [i for i, lab in enumerate(labs) if lab == regime]
            raw = {m: float(np.mean([recs[i]["crps"][m] for i in idx])) for m in MODELS}
            per_series_winners[(regime, min(raw, key=raw.get))] += 1
    print("per-series (regime, winner) counts:", dict(per_series_winners))

    # --- 2. center quality over origins --------------------------------
    rows = []
    for o in origins:
        for _g, s_full in panel.items():
            s = s_full[:o]
            y = float(s_full[o])
            c_router = core.center(s)
            c_trail = float(np.mean(s[-6:]))
            rows.append((c_router - y, c_trail - y, abs(c_router - y), abs(c_trail - y)))
    arr = np.array(rows)
    print(
        f"center over {len(rows)} cells: router bias {arr[:, 0].mean():+.4f} "
        f"MAE {arr[:, 2].mean():.4f} | trail bias {arr[:, 1].mean():+.4f} "
        f"MAE {arr[:, 3].mean():.4f} | router wins {(arr[:, 2] < arr[:, 3]).mean():.2f}"
    )
    # regime probs actually used at prediction time
    probs_seen: Counter = Counter()
    for o in origins[::3]:
        for s_full in panel.values():
            s = s_full[:o]
            vec, _ = core._artifacts(s)
            feats = dict(zip(core._canonical, vec, strict=True))
            probs, _, _ = core.router(s)._regime_probs(feats, core._feature_names)
            top = max(probs, key=probs.get)
            probs_seen[top] += 1
    print("argmax regime at prediction (sampled origins):", dict(probs_seen))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for world in ("control", "S1_4.0"):
        diagnose(args.seed, world)


if __name__ == "__main__":
    main()
