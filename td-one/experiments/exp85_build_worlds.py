"""exp85 infra: build exp72 worlds once and pickle them (light).

exp80's harness rebuilds the simulator seed data in every process
(peak RSS ~2.2 GB — the OOM bottleneck under multi-agent box
contention). The built worlds themselves are tiny (29-month group
series). This builder runs ONE seed per process and pickles
{groups, months, base n_at_risk counts} per world under
experiments/exp85_worlds/<setname>/, so exp85_group_light.py can
run the identical forecast loop at a few hundred MB.

Usage:
    .venv/bin/python experiments/exp85_build_worlds.py \
        --setname fresh --seed-one 2024
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "exp85_worlds"


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setname", required=True)
    ap.add_argument("--seed-one", type=int, required=True)
    args = ap.parse_args()

    e80 = _load("exp80_conformal_fallback", "exp80_conformal_fallback.py")
    exp72 = e80.exp72
    out_dir = OUT_DIR / args.setname
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = exp72.build_seed_data(args.seed_one, "tree")
    base_counts = {g: sd.base_series[g].n_at_risk for g in sd.base_series}
    for world in ("control", "S1_4.0"):
        path = out_dir / f"{args.seed_one}_{world}.pkl"
        if path.exists():
            print(f"skip {path} (exists)", flush=True)
            continue
        groups = exp72.build_world(sd, world)
        bundle = {"groups": groups, "months": sd.months, "base_counts": base_counts}
        path.write_bytes(pickle.dumps(bundle))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
