"""exp80: conformal calibration (Part 1) + thin/dead fallback (Part 2).

Part 1 — conformal range calibration (exp80a), pre-registered in
exp80_report.md. SimulatorV2, exp72/78/79 harness, seeds 7/42/123,
min_group_n=30, worlds control + S1@4-sigma:

- A  BASE  : shipped exp79 default (trailing-mean center + Gaussian spread)
- B  CONF  : A + per-group CQR conformal offsets on [q10, q90]
- C  COND  : B + trailing-volatility scaling of the offsets

Part 2 — thin vs dead fallback (exp80b). Worlds control, S7mix @
migration 0.06/0.25, S1@4-sigma:

- CURR : GroupForecaster with pre-exp80 size-only fallback
         (dead_zero_months=0)
- NEW  : GroupForecaster with the thin/dead split (default)

Metrics: member-weighted pooled CRPS + cov80/cov90, paired t / win
rate on seed x origin x group cells vs the baseline arm.

Usage:
    .venv/bin/python experiments/exp80_conformal_fallback.py [--part a|b]
    .venv/bin/python experiments/exp80_conformal_fallback.py --part a --seeds 2024 555 9001 --arms B
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

EXP_DIR = Path(__file__).resolve().parent
MIN_GROUP_N = 30


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
exp78 = _load("exp78_group_forecaster_spread", "exp78_group_forecaster_spread.py")


def _aggressive_fn():
    import td_one.agent.group_forecaster as gf_mod

    gf_mod.COND_SCALE_MAX = 3.0
    return v2_forecaster_fn(conformal=True, conformal_conditional=True, conformal_level=0.95)


ARM_NAMES_A = ["A", "B", "C"]
ARM_NAMES_B = ["CURR", "NEW"]


def run_worlds(
    seeds: list[int],
    worlds: list[tuple[str, float | None]],
    arm_names: list[str],
    factories: dict,
) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        sd = exp72.build_seed_data(seed, "tree")
        for world, mig in worlds:
            groups = (
                exp72.build_world(sd, world, migration_rate=mig)
                if mig is not None
                else exp72.build_world(sd, world)
            )
            gids = [g for g in groups if g != "__portfolio__"]
            n = len(sd.months)
            origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
            ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids}
            fns = {name: exp72.CachedForecaster(mk()) for name, mk in factories.items()}
            gfs = {}
            for name, fn in fns.items():
                if name == "CURR":  # pre-exp80 size-only fallback
                    gfs[name] = GroupForecaster(fn, min_group_n=MIN_GROUP_N, dead_zero_months=0)
                elif name == "NEW":  # exp80 thin/dead split
                    gfs[name] = GroupForecaster(fn, min_group_n=MIN_GROUP_N, dead_zero_months=2)
                else:  # part-1 arms: fallback semantics held at the new default
                    gfs[name] = GroupForecaster(fn, min_group_n=MIN_GROUP_N)
            world_key = world if mig is None else f"{world}@{mig}"
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
                            c80, c90 = exp78._coverage(q5, y)
                            rows.append(
                                {
                                    "seed": seed,
                                    "world": world_key,
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
            print(f"  seed {seed} world {world_key}: done", flush=True)
    return rows


def _paired(cell: dict, arm: str, ref: str) -> dict:
    diffs = np.array(
        [cell[k][arm] - cell[k][ref] for k in cell if arm in cell[k] and ref in cell[k]]
    )
    t = stats.ttest_1samp(diffs, 0.0)
    return {
        "mean_diff": float(np.mean(diffs)),
        "t": float(t.statistic),
        "p": float(t.pvalue),
        "win_rate": float(np.mean(diffs < 0)),
        "n_cells": int(len(diffs)),
    }


def summarize(rows: list[dict], ref: str) -> dict:
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
            for arm in sorted({r["arm"] for r in sub}):
                arows = [r for r in sub if r["arm"] == arm]
                wsum = sum(r["weight"] for r in arows)
                entry[arm] = {
                    "crps_mw": float(sum(r["crps"] * r["weight"] for r in arows) / wsum),
                    "cov80": float(sum(r["cov80"] * r["weight"] for r in arows) / wsum),
                    "cov90": float(sum(r["cov90"] * r["weight"] for r in arows) / wsum),
                    "fallback_rate": float(sum(r["fallback"] for r in arows)) / max(len(arows), 1),
                }
                if arm != ref:
                    entry[arm]["vs_ref"] = _paired(cell, arm, ref)
            out[world][scope] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], default="a")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--rows-out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.part == "a":
        worlds: list[tuple[str, float | None]] = [("control", None), ("S1_4.0", None)]
        factories = {
            "A": lambda: v2_forecaster_fn(),
            "B": lambda: v2_forecaster_fn(conformal=True),
            "C": lambda: v2_forecaster_fn(conformal=True, conformal_conditional=True),
            # Diagnosis rounds (level sweep): how conservative must the
            # offset quantile be to cover post-shock exceedances?
            "B85": lambda: v2_forecaster_fn(conformal=True, conformal_level=0.85),
            "B90": lambda: v2_forecaster_fn(conformal=True, conformal_level=0.90),
            "C90": lambda: v2_forecaster_fn(
                conformal=True, conformal_conditional=True, conformal_level=0.90
            ),
            # Diagnosis round 4: deliberately aggressive variant
            # (level 0.95, vol-scale clamp 3x) — tests whether the
            # fresh-seed S1 coverage gap is tuning or structural.
            "CX": _aggressive_fn,
        }
        names = args.arms or ARM_NAMES_A
        factories = {k: v for k, v in factories.items() if k in names}
        rows = run_worlds(args.seeds, worlds, names, factories)
        payload = {
            "part": "a",
            "seeds": args.seeds,
            "worlds_summary": summarize(rows, "A"),
        }
        default_out = "exp80a_conformal_results.json"
    else:
        worlds = [
            ("control", None),
            ("S7mix", 0.06),
            ("S7mix", 0.25),
            ("S1_4.0", None),
        ]
        mk_base = lambda: v2_forecaster_fn()  # noqa: E731
        factories = {"CURR": mk_base, "NEW": mk_base}
        rows = run_worlds(args.seeds, worlds, ARM_NAMES_B, factories)
        payload = {
            "part": "b",
            "seeds": args.seeds,
            "worlds_summary": summarize(rows, "CURR"),
        }
        default_out = "exp80b_fallback_results.json"
    payload["runtime_s"] = round(time.time() - t0, 1)
    if args.rows_out:
        (EXP_DIR / args.rows_out).write_text(json.dumps(rows))
        print(f"wrote rows to {args.rows_out}")
    out_path = EXP_DIR / (args.out or default_out)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path} in {payload['runtime_s']}s")
    print(json.dumps(payload["worlds_summary"], indent=2)[:4000])


if __name__ == "__main__":
    main()
