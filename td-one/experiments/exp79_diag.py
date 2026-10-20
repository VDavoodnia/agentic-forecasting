"""exp79 diagnosis rounds: A vs B mechanism + fresh-seed confirmation.

Round 2 (mechanism): per-cell (B - A) CRPS vs the cell's trailing
volatility (sd of trailing-12 diffs at the origin). Hypothesis from
the per-seed split: the naive center's weakness grows with noise (one
noisy last month poisons it); the trailing mean smooths it, costing
nothing in calm cells.

Round 3 (confirmation): arms A/B only on fresh seeds never used in the
main protocol (2024, 555, 9001), same harness, to check B >= A
out-of-sample (the exp66g seed-noise trap guard).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn  # noqa: E402
from td_one.agent.layer2_tools import pinball_crps  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
exp78 = _load("exp78_group_forecaster_spread", "exp78_group_forecaster_spread.py")
MIN_GROUP_N = 30


def run_ab(seeds: list[int], worlds: list[str]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        sd = exp72.build_seed_data(seed, "tree")
        for world in worlds:
            groups = exp72.build_world(sd, world)
            gids = [g for g in groups if g != "__portfolio__"]
            n = len(sd.months)
            origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
            ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids}
            fns = {
                "A": exp72.CachedForecaster(v2_forecaster_fn()),
                "B": exp72.CachedForecaster(v2_forecaster_fn(center="trailing_mean")),
            }
            gfs = {k: GroupForecaster(fn, min_group_n=MIN_GROUP_N) for k, fn in fns.items()}
            for o in origins:
                trunc = {g: exp72._truncate(groups[g], o) for g in gids}
                trunc["__portfolio__"] = exp72._truncate(groups["__portfolio__"], o)
                for name, gf in gfs.items():
                    fc = gf.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
                    for g in gids:
                        hist = np.array(trunc[g].attrition_rate, dtype=float)
                        diffs = np.diff(hist[-13:])
                        vol = float(np.std(diffs)) if len(diffs) > 1 else 0.0
                        y = float(groups[g].attrition_rate[o])
                        q5 = np.asarray(fc[g].quantiles, dtype=float)
                        grid = exp72.interp9(q5)
                        rows.append(
                            {
                                "seed": seed,
                                "world": world,
                                "origin": o,
                                "group": g,
                                "arm": name,
                                "crps": pinball_crps(grid.reshape(9, 1), np.array([y])),
                                "weight": float(ctrl_counts[g][o]),
                                "vol": vol,
                            }
                        )
            print(f"  seed {seed} world {world}: done", flush=True)
    return rows


def paired(rows: list[dict]) -> dict:
    cell: dict[tuple, dict[str, float]] = {}
    for r in rows:
        cell.setdefault((r["seed"], r["origin"], r["group"]), {})[r["arm"]] = r["crps"]
    d = np.array([c["B"] - c["A"] for c in cell.values() if "B" in c and "A" in c])
    t = stats.ttest_1samp(d, 0.0)
    return {
        "n": int(len(d)),
        "mean_diff": float(np.mean(d)),
        "p": float(t.pvalue),
        "win": float(np.mean(d < 0)),
    }


def main() -> None:
    out: dict = {}

    print("round 3: fresh seeds, A vs B")
    fresh = run_ab([2024, 555, 9001], ["control", "S1_4.0"])
    for world in ("control", "S1_4.0"):
        sub = [r for r in fresh if r["world"] == world]
        out[f"fresh_{world}"] = paired(sub)
        for seed in (2024, 555, 9001):
            out[f"fresh_{world}_seed{seed}"] = paired([r for r in sub if r["seed"] == seed])
    (EXP_DIR / "exp79_diag_fresh_rows.json").write_text(json.dumps(fresh))

    print("round 2: mechanism (volatility split) on protocol seeds")
    proto = run_ab([7, 42, 123], ["control"])
    cell: dict[tuple, dict[str, float]] = {}
    vol: dict[tuple, float] = {}
    for r in proto:
        key = (r["seed"], r["origin"], r["group"])
        cell.setdefault(key, {})[r["arm"]] = r["crps"]
        vol[key] = r["vol"]
    keys = [k for k in cell if "A" in cell[k] and "B" in cell[k]]
    diffs = np.array([cell[k]["B"] - cell[k]["A"] for k in keys])
    vols = np.array([vol[k] for k in keys])
    med = float(np.median(vols))
    out["vol_corr"] = {
        "spearman_rho": float(stats.spearmanr(vols, diffs).statistic),
        "spearman_p": float(stats.spearmanr(vols, diffs).pvalue),
        "median_vol": med,
    }
    for label, mask in (("low_vol", vols <= med), ("high_vol", vols > med)):
        d = diffs[mask]
        t = stats.ttest_1samp(d, 0.0)
        out[f"control_{label}"] = {
            "n": int(len(d)),
            "mean_diff": float(np.mean(d)),
            "p": float(t.pvalue),
            "win": float(np.mean(d < 0)),
        }
    (EXP_DIR / "exp79_diag_proto_rows.json").write_text(json.dumps(proto))
    print(json.dumps(out, indent=2))
    (EXP_DIR / "exp79_diag.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
