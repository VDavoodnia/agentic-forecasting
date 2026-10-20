"""exp85 testbed 2 (light): the exp80 harness loop without rebuilds.

Replicates exp80_conformal_fallback.run_worlds line-for-line, except
the simulator seed data / worlds come from exp85_build_worlds.py
pickles instead of a per-process rebuild (peak RSS ~2.2 GB -> a few
hundred MB; the forecast math, truncation, caching, and scoring are
exp72/exp78's own functions, unchanged). Rows are written in
exp80's format and merge with exp85_group_conformal.py --merge.

Fidelity: validated against the heavy harness on fresh seed 2024
(r1 configuration) — pooled arm numbers identical (see
exp85_report.md, round 4 infra note).

Usage:
    .venv/bin/python experiments/exp85_group_light.py --setname fresh \
        --seed-one 2024 --world-one control --tag r4 --level 0.90 \
        --cond --detector errors --hl 3 --minpost 2 \
        --rows-out experiments/exp85_rows_fresh_r4_2024_control.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e80 = _load("exp80_conformal_fallback", "exp80_conformal_fallback.py")
exp72 = e80.exp72
exp78 = e80.exp78
GroupForecaster = e80.GroupForecaster
v2_forecaster_fn = e80.v2_forecaster_fn
pinball_crps = e80.pinball_crps

PROTOCOL = [7, 42, 123]
FRESH = [2024, 555, 9001]
WORLDS = ["control", "S1_4.0"]


def run_light(setname: str, seed: int, world: str, factories: dict) -> list[dict]:
    """exp80.run_worlds' inner loop, one (seed, world), pickled inputs."""
    path = HERE / "exp85_worlds" / setname / f"{seed}_{world}.pkl"
    bundle = pickle.loads(path.read_bytes())
    groups = bundle["groups"]
    ctrl_counts = bundle["base_counts"]
    gids = [g for g in groups if g != "__portfolio__"]
    n = len(bundle["months"])
    origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
    fns = {name: exp72.CachedForecaster(mk()) for name, mk in factories.items()}
    gfs = {name: GroupForecaster(fn, min_group_n=e80.MIN_GROUP_N) for name, fn in fns.items()}
    rows: list[dict] = []
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
                for scope, q5 in (("group", fc[g].quantiles), ("global", global_q)):
                    grid = exp72.interp9(np.asarray(q5, dtype=float))
                    c80, c90 = exp78._coverage(q5, y)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setname", choices=["protocol", "fresh"], required=True)
    ap.add_argument("--seed-one", type=int, required=True)
    ap.add_argument("--world-one", choices=WORLDS, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--rows-out", type=str, required=True)
    ap.add_argument("--hl", type=float, default=6.0)
    ap.add_argument("--pref", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--n0", type=float, default=12.0)
    ap.add_argument("--minpost", type=int, default=5)
    ap.add_argument("--level", type=float, default=0.8)
    ap.add_argument("--detector", type=str, default="scan", choices=["scan", "cusum", "errors"])
    ap.add_argument("--cond", action="store_true")
    args = ap.parse_args()

    common = {
        "conformal": True,
        "conformal_conditional": args.cond,
        "conformal_level": args.level,
        "conformal_half_life": args.hl,
        "conformal_pre_factor": args.pref,
        "conformal_min_post": args.minpost,
        "conformal_bias_lambda": args.lam,
        "conformal_bias_n0": args.n0,
        "conformal_detector": args.detector,
    }

    def adaptive(mode: str):
        return lambda: v2_forecaster_fn(**common, conformal_adaptive=mode)  # noqa: E731

    factories = {
        "A": lambda: v2_forecaster_fn(),
        "C90": lambda: v2_forecaster_fn(
            conformal=True, conformal_conditional=True, conformal_level=0.90
        ),
        "RST": adaptive("reset"),
        "DEC": adaptive("decay"),
        "CP": adaptive("cp"),
        "BC": adaptive("bc"),
        "CPBC": adaptive("cp_bc"),
    }
    rows = run_light(args.setname, args.seed_one, args.world_one, factories)
    Path(args.rows_out).write_text(json.dumps({"rows": rows}))
    print(f"wrote {len(rows)} rows -> {args.rows_out}")


if __name__ == "__main__":
    main()
