"""exp78: validation of the quantile-spread fix for ``v2_forecaster_fn``.

Root cause (see ``agent/group_forecaster.py`` docstring): the production
group forecaster never actually ran its router — ``ProposedForecasterV2``
was constructed but never fitted, ``predict`` raised, and the blanket
``except`` returned a point mass (last value tiled across all quantile
levels). exp72 measured the damage at CRPS 0.165 (control) vs 0.13 for a
proper Gaussian wiring on identical histories.

This experiment asks, on SimulatorV2 portfolios via exp72's harness:

1. Does the fixed forecaster (OLD=point-mass vs NEW=spread, empirical and
   Gaussian variants; G5=exp72's trailing-mean Gaussian as reference)
   improve CRPS and interval coverage on control and S1@4-sigma, through
   the real ``GroupForecaster`` (fallback included) AND as a global
   forecaster?
2. Does the exp66c grouping-cost cell change? (Re-run of exp66c seed 42,
   arms A_global vs G_deterministic, OLD vs NEW — exp66c scores against
   test-window means, the setting where point-masses looked fine.)
3. Layer-2 smoke: does ``MetaRouter`` still emit per-origin traces when
   fed the fixed forecaster's candidates?

Metrics: pinball CRPS on the 9-level grid (member-weighted pooled,
paired t + win rate on seed x origin x group cells); coverage of the
80% central interval [q10, q90] (native) and a 90% interval obtained by
constant-slope extrapolation of the outermost 5-level segments to
0.05/0.95 (for a point mass both collapse to P(y == last value)).

Usage:
    .venv/bin/python experiments/exp78_group_forecaster_spread.py
    .venv/bin/python experiments/exp78_group_forecaster_spread.py --seeds 7 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn  # noqa: E402
from td_one.agent.layer2_tools import pinball_crps  # noqa: E402
from td_one.agent.meta_router import MetaRouter, MetaRouterConfig  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
MIN_GROUP_N = 30  # exp72 round-1+ setting (fallback fires, but not for all)


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered pre-exec
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")

ARMS = {
    "OLD": lambda: v2_forecaster_fn(quantile_spread=False),
    "EMP": lambda: v2_forecaster_fn(quantile_spread=True, spread_method="empirical"),
    "GAU": lambda: v2_forecaster_fn(quantile_spread=True, spread_method="gaussian"),
    "G5": lambda: exp72.gauss5_forecaster_fn(),
}


def _coverage(q5: np.ndarray, y: float) -> tuple[float, float]:
    """(cov80, cov90) indicators for one 5-level forecast and actual."""
    q = np.asarray(q5, dtype=float)
    cov80 = float(q[0] <= y <= q[4])
    lo = q[0] - (q[1] - q[0]) * (0.05 / 0.15)
    hi = q[4] + (q[4] - q[3]) * (0.05 / 0.15)
    cov90 = float(lo <= y <= hi)
    return cov80, cov90


def run_worlds(seeds: list[int], worlds: list[str]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        sd = exp72.build_seed_data(seed, "tree")
        for world in worlds:
            groups = exp72.build_world(sd, world)
            gids = [g for g in groups if g != "__portfolio__"]
            n = len(sd.months)
            origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
            ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids}
            fns = {name: exp72.CachedForecaster(mk()) for name, mk in ARMS.items()}
            gfs = {name: GroupForecaster(fn, min_group_n=MIN_GROUP_N) for name, fn in fns.items()}
            for o in origins:
                trunc = {g: exp72._truncate(groups[g], o) for g in gids}
                trunc["__portfolio__"] = exp72._truncate(groups["__portfolio__"], o)
                ph = np.array(trunc["__portfolio__"].attrition_rate, dtype=float)
                for name, gf in gfs.items():
                    fc = gf.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
                    global_q = fns[name](ph)
                    for g in gids:
                        y = float(groups[g].attrition_rate[o])
                        w = float(ctrl_counts[g][o])
                        for scope, q5 in (
                            ("group", fc[g].quantiles),
                            ("global", global_q),
                        ):
                            grid = exp72.interp9(np.asarray(q5, dtype=float))
                            c80, c90 = _coverage(q5, y)
                            rows.append(
                                {
                                    "seed": seed,
                                    "world": world,
                                    "origin": o,
                                    "group": g,
                                    "arm": name,
                                    "scope": scope,
                                    "crps": pinball_crps(grid.reshape(9, 1), np.array([y])),
                                    "cov80": c80,
                                    "cov90": c90,
                                    "weight": w,
                                    "fallback": bool(fc[g].used_fallback),
                                }
                            )
            print(f"  seed {seed} world {world}: done", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    worlds = sorted({r["world"] for r in rows})
    for world in worlds:
        out[world] = {}
        for scope in ("group", "global"):
            sub = [r for r in rows if r["world"] == world and r["scope"] == scope]
            cell: dict[tuple, dict[str, float]] = {}
            for r in sub:
                key = (r["seed"], r["origin"], r["group"])
                cell.setdefault(key, {})[r["arm"]] = r["crps"]
            entry: dict = {}
            for arm in ARMS:
                arows = [r for r in sub if r["arm"] == arm]
                wsum = sum(r["weight"] for r in arows)
                entry[arm] = {
                    "crps_mw": float(sum(r["crps"] * r["weight"] for r in arows) / wsum),
                    "cov80": float(sum(r["cov80"] * r["weight"] for r in arows) / wsum),
                    "cov90": float(sum(r["cov90"] * r["weight"] for r in arows) / wsum),
                }
                if arm != "OLD":
                    diffs = np.array(
                        [cell[k][arm] - cell[k]["OLD"] for k in cell if arm in cell[k]]
                    )
                    t = stats.ttest_1samp(diffs, 0.0)
                    entry[arm]["vs_old"] = {
                        "mean_diff": float(np.mean(diffs)),
                        "t": float(t.statistic),
                        "p": float(t.pvalue),
                        "win_rate": float(np.mean(diffs < 0)),
                        "n_cells": int(len(diffs)),
                    }
            out[world][scope] = entry
    return out


def layer2_smoke() -> dict:
    """MetaRouter over fixed-forecaster candidates: traces must emit."""
    sd = exp72.build_seed_data(7, "tree")
    groups = exp72.build_world(sd, "control")
    gids = [g for g in groups if g != "__portfolio__"][:2]
    n = len(sd.months)
    origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
    fn_new = exp72.CachedForecaster(v2_forecaster_fn())
    gf = GroupForecaster(fn_new, min_group_n=MIN_GROUP_N)
    result: dict = {"groups": {}}
    for g in gids:
        cand: dict[str, list[np.ndarray]] = {"naive": [], "v2new": [], "globalnew": []}
        actuals: list[np.ndarray] = []
        for o in origins:
            trunc = {gg: exp72._truncate(groups[gg], o) for gg in gids}
            trunc["__portfolio__"] = exp72._truncate(groups["__portfolio__"], o)
            fc = gf.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
            hist = np.array(trunc[g].attrition_rate, dtype=float)
            ph = np.array(trunc["__portfolio__"].attrition_rate, dtype=float)
            cand["naive"].append(exp72.naive_grid9(hist).reshape(9, 1))
            cand["v2new"].append(exp72.interp9(fc[g].quantiles).reshape(9, 1))
            cand["globalnew"].append(exp72.interp9(fn_new(ph)).reshape(9, 1))
            actuals.append(np.array([groups[g].attrition_rate[o]]))
        y = np.array(groups[g].attrition_rate, dtype=float)
        res = MetaRouter(MetaRouterConfig()).run(y, origins, cand, actuals)
        assert len(res.traces) == len(origins), "trace count mismatch"
        assert len(res.choices) == len(origins), "choice count mismatch"
        assert all(isinstance(t, dict) and t for t in res.traces), "empty trace"
        result["groups"][g] = {
            "n_origins": len(origins),
            "choices": {c: res.choices.count(c) for c in sorted(set(res.choices))},
            "trace_keys": sorted(res.traces[0].keys()),
        }
    result["ok"] = True
    return result


def run_exp66c_cell() -> dict:
    """exp66c seed-42 grouping cost (G_deterministic - A_global), OLD vs NEW."""
    exp66c = _load("exp66c_deterministic_groups", "exp66c_deterministic_groups.py")
    out: dict = {}
    for label, factory in (
        ("OLD", lambda: v2_forecaster_fn(quantile_spread=False)),
        ("NEW", lambda: v2_forecaster_fn()),
    ):
        exp66c.v2_forecaster_fn = factory  # run_seed builds its forecaster from this
        res = exp66c.run_seed(42)
        arms = res["arms"]
        out[label] = {
            "A_global": arms["A_global"]["weighted_crps"],
            "G_deterministic": arms["G_deterministic"]["weighted_crps"],
            "grouping_cost": arms["G_deterministic"]["weighted_crps"]
            - arms["A_global"]["weighted_crps"],
        }
    out["delta_cost"] = out["NEW"]["grouping_cost"] - out["OLD"]["grouping_cost"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    ap.add_argument("--worlds", nargs="+", default=["control", "S1_4.0"])
    ap.add_argument("--skip-66c", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows = run_worlds(args.seeds, args.worlds)
    payload: dict = {"worlds_summary": summarize(rows)}
    if not args.skip_smoke:
        payload["layer2_smoke"] = layer2_smoke()
    if not args.skip_66c:
        payload["exp66c_cell"] = run_exp66c_cell()
    payload["runtime_s"] = round(time.time() - t0, 1)
    out_path = EXP_DIR / "exp78_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path} in {payload['runtime_s']}s")


if __name__ == "__main__":
    main()
