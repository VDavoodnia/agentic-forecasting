"""exp86 diagnosis: where Chronos wins and loses vs the champion.

Post-pass only (zero Chronos calls). Slices the stored per-cell
forecast vectors from a rows file by member-weight tercile, group,
and fallback status, and dissects the gate arm's picks. Motivated by
the main run's central tension: arms B/C beat A on the *unweighted*
paired contrast (typical cell) yet lose on the pre-registered
*member-weighted* pooled CRPS (the big groups dominate).

Usage:
    .venv/bin/python experiments/exp86_diagnose.py \
        --rows exp86_rows_main.json --out exp86_diagnosis_main.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXP_DIR = Path(__file__).resolve().parent


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e86 = _load("exp86_chronos_segments", "exp86_chronos_segments.py")

ARMS = ["B", "C", "D", "BL25", "BL50"]


def _arm_quantiles(c_map: dict, d_choice: dict, c: dict, arm: str) -> np.ndarray:
    qa = np.array(c["qA"], dtype=float)
    qb = np.array(c["qB"], dtype=float)
    key = (c["seed"], c["world"], c["origin"], c["group"])
    if arm == "A":
        return qa
    if arm == "B":
        return qb
    if arm == "C":
        return c_map[key] if c["scope"] == "group" else qb
    if arm == "D":
        return qb if d_choice[key] == "B" else qa
    w = {"BL25": 0.25, "BL50": 0.5}[arm]
    return np.clip(np.maximum.accumulate((1.0 - w) * qa + w * qb), 0.0, 1.0)


def _contrast(diffs: list[float], weights: list[float]) -> dict:
    d = np.array(diffs)
    w = np.array(weights)
    t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else (float("nan"), float("nan"))
    return {
        "n": int(len(d)),
        "mean_diff_unweighted": float(np.mean(d)) if len(d) else float("nan"),
        "p_unweighted": float(t.pvalue),
        "crps_a_mw": float("nan"),  # filled by caller when raw scores given
        "mean_diff_mw": float(np.sum(w * d) / np.sum(w)) if np.sum(w) > 0 else float("nan"),
        "win_rate": float(np.mean(d < 0)) if len(d) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="exp86_rows_main.json")
    ap.add_argument("--out", default="exp86_diagnosis_main.json")
    args = ap.parse_args()

    cells = json.loads((EXP_DIR / args.rows).read_text())["cells"]
    group_cells = [c for c in cells if c["scope"] == "group"]
    c_map = e86._arm_c(cells)
    d_choice = e86._arm_d_choice(cells)

    # per-cell CRPS for A and each arm
    crps: dict[tuple, dict[str, float]] = {}
    for c in group_cells:
        key = (c["seed"], c["world"], c["origin"], c["group"])
        crps[key] = {"A": e86._score(np.array(c["qA"], dtype=float), c["y"])[0]}
        for arm in ARMS:
            crps[key][arm] = e86._score(_arm_quantiles(c_map, d_choice, c, arm), c["y"])[0]

    out: dict = {"weight_terciles": {}, "per_group_control": {}, "fallback_control": {}, "gate": {}}

    # --- weight terciles per world (terciles over that world's cells) ---
    for world in e86.WORLDS:
        sub = [c for c in group_cells if c["world"] == world]
        ws = np.array([c["weight"] for c in sub])
        qs = np.quantile(ws, [1 / 3, 2 / 3])
        out["weight_terciles"][world] = {}
        for name, mask in (
            ("low", ws <= qs[0]),
            ("mid", (ws > qs[0]) & (ws <= qs[1])),
            ("high", ws > qs[1]),
        ):
            sel = [c for c, m in zip(sub, mask, strict=True) if m]
            entry = {"weight_share": float(sum(c["weight"] for c in sel) / np.sum(ws))}
            for arm in ARMS:
                diffs = [
                    crps[(c["seed"], c["world"], c["origin"], c["group"])][arm]
                    - crps[(c["seed"], c["world"], c["origin"], c["group"])]["A"]
                    for c in sel
                ]
                entry[arm] = _contrast(diffs, [c["weight"] for c in sel])
            out["weight_terciles"][world][name] = entry

    # --- per-group, control world, pooled across seeds ---
    groups = sorted({c["group"] for c in group_cells})
    for g in groups:
        sel = [c for c in group_cells if c["world"] == "control" and c["group"] == g]
        if not sel:
            continue
        entry = {"total_weight": float(sum(c["weight"] for c in sel))}
        for arm in ARMS:
            diffs = [
                crps[(c["seed"], c["world"], c["origin"], c["group"])][arm]
                - crps[(c["seed"], c["world"], c["origin"], c["group"])]["A"]
                for c in sel
            ]
            entry[arm] = _contrast(diffs, [c["weight"] for c in sel])
        out["per_group_control"][g] = entry

    # --- fallback vs own-forecast cells, control ---
    for name, want in (("fallback", True), ("own", False)):
        sel = [c for c in group_cells if c["world"] == "control" and c["fallback"] == want]
        entry = {}
        for arm in ARMS:
            diffs = [
                crps[(c["seed"], c["world"], c["origin"], c["group"])][arm]
                - crps[(c["seed"], c["world"], c["origin"], c["group"])]["A"]
                for c in sel
            ]
            entry[arm] = _contrast(diffs, [c["weight"] for c in sel])
        out["fallback_control"][name] = entry

    # --- gate dissection: pick rate + conditional diffs, per world ---
    for world in e86.WORLDS:
        sub = [c for c in group_cells if c["world"] == world]
        picks = [d_choice[(c["seed"], c["world"], c["origin"], c["group"])] for c in sub]
        d_diffs = [
            crps[(c["seed"], c["world"], c["origin"], c["group"])]["D"]
            - crps[(c["seed"], c["world"], c["origin"], c["group"])]["A"]
            for c in sub
        ]
        b_diffs = [
            crps[(c["seed"], c["world"], c["origin"], c["group"])]["B"]
            - crps[(c["seed"], c["world"], c["origin"], c["group"])]["A"]
            for c in sub
        ]
        picked_b = [bd for bd, p in zip(b_diffs, picks, strict=True) if p == "B"]
        out["gate"][world] = {
            "pick_B_rate": float(np.mean([p == "B" for p in picks])),
            "D_mean_diff": float(np.mean(d_diffs)),
            "B_mean_diff_when_picked": float(np.mean(picked_b)) if picked_b else float("nan"),
            "n_picked": int(len(picked_b)),
        }

    (EXP_DIR / args.out).write_text(json.dumps(out, indent=2))
    # compact printout
    for world in e86.WORLDS:
        print(f"== {world} weight terciles (share | B mw-diff | C mw-diff | BL25 mw-diff) ==")
        for name in ("low", "mid", "high"):
            e = out["weight_terciles"][world][name]
            print(
                f"  {name:4s} share {e['weight_share']:.2f}  "
                f"B {e['B']['mean_diff_mw']:+.5f}  C {e['C']['mean_diff_mw']:+.5f}  "
                f"BL25 {e['BL25']['mean_diff_mw']:+.5f}"
            )
    print("== gate ==")
    for world in e86.WORLDS:
        g = out["gate"][world]
        print(
            f"  {world}: pick-B {g['pick_B_rate']:.2f} ({g['n_picked']} cells), "
            f"D diff {g['D_mean_diff']:+.5f}, "
            f"B diff when picked {g['B_mean_diff_when_picked']:+.5f}"
        )
    print(f"wrote {EXP_DIR / args.out}")


if __name__ == "__main__":
    main()
