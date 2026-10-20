"""exp80 cell-level analysis (offline, from saved rows JSON).

Part A: per-seed cov80/CRPS for the conformal winner vs A.
Part B: donor-group post-empty cells on S7mix@0.25 — pooled CRPS for
CURR vs NEW on cells where the donor group's recent-2 n_at_risk sum is
0 (the cells the dead branch targets), plus fallback rates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[0] / "src"))


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")


def _mw(rows: list[dict], key: str) -> float:
    w = sum(r["weight"] for r in rows)
    return float(sum(r[key] * r["weight"] for r in rows) / w) if w else float("nan")


def part_a(rows_path: str) -> None:
    rows = json.loads((EXP_DIR / rows_path).read_text())
    for world in sorted({r["world"] for r in rows}):
        print(f"--- {world} (group scope) ---")
        for seed in sorted({r["seed"] for r in rows}):
            line = [f"seed {seed}"]
            for arm in ("A", "C90"):
                sub = [
                    r
                    for r in rows
                    if r["world"] == world
                    and r["seed"] == seed
                    and r["arm"] == arm
                    and r["scope"] == "group"
                ]
                line.append(f"{arm}: crps {_mw(sub, 'crps'):.4f} cov80 {_mw(sub, 'cov80'):.3f}")
            print("  " + " | ".join(line))


def _donor_and_dead(rows: list[dict]) -> None:
    """Identify the donor per seed (S7mix@0.25) and its dead cells."""
    for seed in sorted({r["seed"] for r in rows}):
        sd = exp72.build_seed_data(seed, "tree")
        groups = exp72.build_world(sd, "S7mix", migration_rate=0.25)
        gids = [g for g in groups if g != "__portfolio__"]
        # donor = group whose late-window n_at_risk collapses to 0
        dead_cells: dict[str, list[int]] = {}
        for g in gids:
            n_risk = groups[g].n_at_risk
            origins = list(range(exp72.ORIGIN_START, len(sd.months) - exp72.ORIGIN_STOP_TAIL))
            dead = [o for o in origins if sum(n_risk[max(0, o - 2) : o]) == 0]
            if dead:
                dead_cells[g] = dead
        print(f"seed {seed}: groups with dead cells: {dead_cells}")
        for g, dead_origins in dead_cells.items():
            for arm in ("CURR", "NEW"):
                sub = [
                    r
                    for r in rows
                    if r["world"] == "S7mix@0.25"
                    and r["seed"] == seed
                    and r["group"] == g
                    and r["arm"] == arm
                    and r["scope"] == "group"
                    and r["origin"] in dead_origins
                ]
                if sub:
                    print(
                        f"  {g} {arm}: dead-cell crps (unweighted mean) "
                        f"{float(np.mean([r['crps'] for r in sub])):.4f} over {len(sub)} cells; "
                        f"fallback rate {float(np.mean([r['fallback'] for r in sub])):.2f}"
                    )


def part_b(rows_path: str) -> None:
    rows = json.loads((EXP_DIR / rows_path).read_text())
    _donor_and_dead(rows)
    print("--- fallback rates (group scope) ---")
    for world in sorted({r["world"] for r in rows}):
        for arm in ("CURR", "NEW"):
            sub = [
                r for r in rows if r["world"] == world and r["arm"] == arm and r["scope"] == "group"
            ]
            print(
                f"  {world} {arm}: fallback {float(np.mean([r['fallback'] for r in sub])):.3f} "
                f"cov80 {_mw(sub, 'cov80'):.3f}"
            )


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "a":
        part_a(sys.argv[2])
    else:
        part_b(sys.argv[2])
