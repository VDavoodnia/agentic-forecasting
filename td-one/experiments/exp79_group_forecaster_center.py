"""exp79: which center (and whether the real router) for the group forecaster.

exp78 fixed the point-mass defect with a naive last-value center +
Gaussian spread, but its G5 reference arm (trailing-mean center) still
won on CRPS, and the *real* fitted v2 router was never evaluated inside
the group forecaster. This experiment settles both questions on
SimulatorV2 via exp72/exp78's harness (seeds 7/42/123, min_group_n=30,
control + S1@4-sigma):

- A  GAU    : exp78 shipped default (naive center + Gaussian spread)
- B  TRAIL  : trailing-6-mean center + the identical exp78 spread
- B' G5X    : exp72's gauss5 verbatim (reference, connects to exp78)
- C  ROUTER : production ``router_forecaster_fn`` — ProposedForecasterV2
              fitted per series on its own rolling-origin backtest
              (generic calm/stressed spec; experts naive/ETS/ARIMA)
- C' ROUTM4 : same router with the m4 spec (diagnostic)
- D  M0     : HistoricalBaseline (Jeffreys Beta-Binomial) posterior
              predictive rate quantiles (reference; needs counts, so it
              is wired via a per-world counts registry, not the plain
              forecaster_fn contract)

Metrics and decision rule are pre-registered in exp79_report.md:
member-weighted pooled CRPS + cov80/cov90, group and global scopes,
paired t + win rate on seed x origin x group cells; the winner must
beat A at p<0.05 on control group CRPS without losing >0.05 cov80 on
S1@4sigma.

Usage:
    .venv/bin/python experiments/exp79_group_forecaster_center.py
    .venv/bin/python experiments/exp79_group_forecaster_center.py --seeds 42
    .venv/bin/python experiments/exp79_group_forecaster_center.py --exp66c-arm B
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

from td_one.agent.group_forecaster import (  # noqa: E402
    QUANTILE_LEVELS,
    GroupForecaster,
    router_forecaster_fn,
    v2_forecaster_fn,
)
from td_one.agent.layer2_tools import pinball_crps  # noqa: E402
from td_one.data_gen.group_series import GroupSeries  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
MIN_GROUP_N = 30  # exp72 round-1+ / exp78 setting


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered pre-exec
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
exp78 = _load("exp78_group_forecaster_spread", "exp78_group_forecaster_spread.py")

ARM_NAMES = ["A", "B", "G5X", "C", "CM4", "D"]


def _counts_registry(
    groups: dict[str, GroupSeries], origins: list[int]
) -> tuple[dict[bytes, tuple[np.ndarray, np.ndarray]], int]:
    """Map truncated-rate-series bytes -> truncated (n_at_risk, n_switched).

    Arm D (M0) needs counts the forecaster_fn contract does not carry.
    The harness only ever passes truncated series at known origins, so
    an exact bytes-keyed registry is complete and PIT-safe (truncated
    counts only).

    Collision rule (indistinguishability pooling): two groups can
    share a bit-identical rate history with different exposures (seen
    in practice: all-zero prefixes). A rate-only forecaster cannot
    distinguish them, so their counts are pooled (elementwise sum) —
    the maximum-information M0 answer available under the contract.
    Returns (registry, n_pooled_collisions).
    """
    reg: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    n_pooled = 0
    for o in origins:
        for gs in groups.values():
            rates = np.array(gs.attrition_rate[:o], dtype=float)
            counts = (
                np.array(gs.n_at_risk[:o], dtype=float),
                np.array(gs.n_switched[:o], dtype=float),
            )
            key = rates.tobytes()
            if key in reg and not (
                np.array_equal(reg[key][0], counts[0]) and np.array_equal(reg[key][1], counts[1])
            ):
                reg[key] = (reg[key][0] + counts[0], reg[key][1] + counts[1])
                n_pooled += 1
            else:
                reg[key] = counts
    return reg, n_pooled


def _m0_forecaster_fn(registry: dict[bytes, tuple[np.ndarray, np.ndarray]]):
    """M0 arm: HistoricalBaseline posterior predictive rate quantiles."""

    def _forecast(series: np.ndarray) -> np.ndarray:
        key = np.asarray(series, dtype=float).tobytes()
        n_risk, n_sw = registry[key]
        m0 = HistoricalBaseline(trailing_window=6)
        m0.fit(n_risk, n_sw)
        n_up = max(1, int(round(float(n_risk[-1])))) if len(n_risk) else 1
        q = m0.predict_rate_quantiles(n_up, QUANTILE_LEVELS)
        return np.array([q[float(lv)] for lv in QUANTILE_LEVELS], dtype=float)

    return _forecast


def _arm_factories(registry: dict[bytes, tuple[np.ndarray, np.ndarray]]) -> dict:
    return {
        "A": lambda: v2_forecaster_fn(),
        "B": lambda: v2_forecaster_fn(center="trailing_mean"),
        "G5X": lambda: exp72.gauss5_forecaster_fn(),
        "C": lambda: router_forecaster_fn(),
        "CM4": lambda: router_forecaster_fn(regime_spec="m4"),
        "D": lambda: _m0_forecaster_fn(registry),
    }


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
            registry, n_pooled = _counts_registry(groups, origins)
            if n_pooled:
                print(
                    f"  seed {seed} world {world}: M0 registry pooled "
                    f"{n_pooled} indistinguishable rate histories",
                    flush=True,
                )
            factories = _arm_factories(registry)
            fns = {name: exp72.CachedForecaster(mk()) for name, mk in factories.items()}
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
            for arm in ARM_NAMES:
                arows = [r for r in sub if r["arm"] == arm]
                wsum = sum(r["weight"] for r in arows)
                entry[arm] = {
                    "crps_mw": float(sum(r["crps"] * r["weight"] for r in arows) / wsum),
                    "cov80": float(sum(r["cov80"] * r["weight"] for r in arows) / wsum),
                    "cov90": float(sum(r["cov90"] * r["weight"] for r in arows) / wsum),
                }
                if arm != "A":
                    entry[arm]["vs_A"] = _paired(cell, arm, "A")
            entry["contrasts"] = {
                "C_vs_B": _paired(cell, "C", "B"),
                "B_vs_G5X": _paired(cell, "B", "G5X"),
                "C_vs_G5X": _paired(cell, "C", "G5X"),
            }
            out[world][scope] = entry
    return out


def run_exp66c_cell(arm: str) -> dict:
    """exp66c seed-42 grouping cost with one exp79 arm as the forecaster."""
    exp66c = _load("exp66c_deterministic_groups", "exp66c_deterministic_groups.py")
    registry_free = {
        "A": lambda: v2_forecaster_fn(),
        "B": lambda: v2_forecaster_fn(center="trailing_mean"),
        "C": lambda: router_forecaster_fn(),
    }
    if arm not in registry_free:
        raise ValueError(f"exp66c cell supports arms {sorted(registry_free)}, got {arm!r}")
    exp66c.v2_forecaster_fn = registry_free[arm]
    res = exp66c.run_seed(42)
    arms = res["arms"]
    return {
        "arm": arm,
        "A_global": arms["A_global"]["weighted_crps"],
        "G_deterministic": arms["G_deterministic"]["weighted_crps"],
        "grouping_cost": arms["G_deterministic"]["weighted_crps"]
        - arms["A_global"]["weighted_crps"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    ap.add_argument("--worlds", nargs="+", default=["control", "S1_4.0"])
    ap.add_argument("--exp66c-arm", type=str, default=None)
    ap.add_argument("--out", type=str, default="exp79_results.json")
    ap.add_argument("--rows-out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    payload: dict = {}
    if args.exp66c_arm:
        payload["exp66c_cell"] = run_exp66c_cell(args.exp66c_arm)
    else:
        rows = run_worlds(args.seeds, args.worlds)
        payload["worlds_summary"] = summarize(rows)
        if args.rows_out:
            rows_path = EXP_DIR / args.rows_out
            rows_path.write_text(json.dumps(rows))
            print(f"wrote {rows_path}")
    payload["runtime_s"] = round(time.time() - t0, 1)
    out_path = EXP_DIR / args.out
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path} in {payload['runtime_s']}s")


if __name__ == "__main__":
    main()
