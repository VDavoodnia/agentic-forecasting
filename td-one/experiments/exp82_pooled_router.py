"""exp82: the pooled router inside the group forecaster.

exp79 proved the real v2 router, fitted **per series**, loses to the
trailing-mean center: each series gives it at most ~13 backtest
records. exp79's own diagnosis: the router was designed for record
pools of hundreds across series (exp47). exp82 tests the pooled fit —
one ``ProposedForecasterV2`` fitted on rolling-origin backtest records
pooled across all SimulatorV2 series in the panel (production
``pooled_router_forecaster_fn``), center = router's combined median,
intervals = exp78's Gaussian spread.

Arms (pre-registered in exp82_report.md):
- A  TRAIL    : shipped default (trailing-6-mean center + Gaussian spread)
- B  POOL     : pooled router center + identical Gaussian spread
- B0 POOLOWN  : pooled router's own combined quantiles (diagnostic)
- C  POOL+L2  : Layer-2 MetaRouter gate over {trail=A, pool=B} grids
- D  PERSERIES: exp79's arm C (per-series router), reference

Iteration-round variants (same script, flags): --features (R2a,
regime-feature subset), --shrink (R2b, center shrinkage toward the
trailing mean), --backtest-start (R2c, pool size), --target-weight
(R3, partial pooling), --hard (R4, argmax-regime expert only).

Usage:
    .venv/bin/python experiments/exp82_pooled_router.py
    .venv/bin/python experiments/exp82_pooled_router.py --seeds 2024 555 9001 --out exp82_fresh.json
    .venv/bin/python experiments/exp82_pooled_router.py --shrink 0.5 --out exp82_r2b.json
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
    _PooledRouter,
    _spread_offsets,
    pooled_router_forecaster_fn,
    router_forecaster_fn,
    v2_forecaster_fn,
)
from td_one.agent.layer2_tools import pinball_crps  # noqa: E402
from td_one.agent.meta_router import MetaRouter, MetaRouterConfig  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
MIN_GROUP_N = 30  # exp72 round-1+ / exp78/79 setting
L9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

#: R2a: the features the generic regime rule reads, plus its close kin.
REGIME_FEATURES = (
    "n_changepoints",
    "n_recent_cps",
    "vol_ratio",
    "trend_strength",
    "seasonal_strength",
    "hurst",
)


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
exp78 = _load("exp78_group_forecaster_spread", "exp78_group_forecaster_spread.py")

ARM_NAMES = ["A", "B", "B0", "C", "D"]

_EQUIV_CHECKED = False


class Variant:
    """Iteration-round knobs for the pooled arms (defaults = R1 design)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.features: tuple[str, ...] | None = REGIME_FEATURES if args.features else None
        self.shrink: float = args.shrink
        self.backtest_start: int = args.backtest_start
        self.target_weight: int = args.target_weight
        self.hard: bool = args.hard
        self.normalize: bool = args.normalize

    @property
    def is_plain(self) -> bool:
        """True iff arm B is exactly the production pooled fn."""
        return (
            self.features is None
            and self.shrink == 0.0
            and self.backtest_start == 8
            and self.target_weight == 1
            and not self.hard
            and not self.normalize
        )

    def describe(self) -> dict:
        return {
            "features": list(self.features) if self.features else "all-10",
            "shrink": self.shrink,
            "backtest_start": self.backtest_start,
            "target_weight": self.target_weight,
            "hard": self.hard,
            "normalize": self.normalize,
        }


def _core(panel: dict[str, np.ndarray], variant: Variant) -> _PooledRouter:
    return _PooledRouter(
        panel,
        backtest_start=variant.backtest_start,
        features=variant.features,
        normalize_crps=variant.normalize,
    )


def _variant_center(core: _PooledRouter, s: np.ndarray, variant: Variant) -> float:
    """Center for the variant arms, built on the shared pooled core."""
    if variant.hard:
        fc = core.router(s, target_weight=variant.target_weight)
        vec, qs = core._artifacts(s)
        feats = dict(zip(core._canonical, vec, strict=True))
        probs, _, _ = fc._regime_probs(feats, core._feature_names)
        regime = max(probs, key=probs.get)
        expert = fc._expert_per_regime[regime]
        return float(np.asarray(qs[expert], dtype=float)[4, 0])
    c = core.center(s, target_weight=variant.target_weight)
    if variant.shrink > 0.0:
        trail = float(np.mean(s[-6:])) if len(s) else 0.0
        c = (1.0 - variant.shrink) * c + variant.shrink * trail
    return c


def _pooled_fns(panel: dict[str, np.ndarray], variant: Variant):
    """(arm B fn, arm B0 fn) for one (seed, world) panel.

    Both pooled arms share ONE ``_PooledRouter`` core: the prefix
    artifacts (series_features + ETS/ARIMA fits) dominate the runtime
    and are identical for B and B0. Arm B's fn is the production
    computation plumbed through the shared core (router center +
    exp78 Gaussian spread); :func:`_check_production_equivalence`
    verifies it against the literal production factory.
    """
    core = _core(panel, variant)

    def fn_b0(series: np.ndarray) -> np.ndarray:
        q9 = core.combined_q9(np.asarray(series, dtype=float), target_weight=variant.target_weight)
        q5 = np.interp(QUANTILE_LEVELS, L9, q9)
        return np.clip(np.maximum.accumulate(q5), 0.0, 1.0)

    def fn_b(series: np.ndarray) -> np.ndarray:
        s = np.asarray(series, dtype=float)
        q = _variant_center(core, s, variant) + _spread_offsets(s, "gaussian")
        return np.clip(np.maximum.accumulate(q), 0.0, 1.0)

    return fn_b, fn_b0


def _check_production_equivalence(variant: Variant) -> None:
    """Arm B (shared core) must equal the production factory, exactly.

    Only meaningful for the plain R1 variant; other variants have no
    production counterpart. Checked on a small synthetic panel: what
    could drift is the plumbing (center + spread), not the data.
    Raises AssertionError on any mismatch so a plumbing drift can
    never silently contaminate the ship decision.
    """
    if not variant.is_plain:
        return
    rng = np.random.default_rng(82)
    panel = {f"s{i}": 0.05 + np.cumsum(rng.normal(0.0, 0.004, 30)) for i in range(3)}
    prod = pooled_router_forecaster_fn(panel)
    mine, _ = _pooled_fns(panel, variant)
    for series in panel.values():
        for length in (15, 21):
            a = prod(series[:length])
            b = mine(series[:length])
            if not np.allclose(a, b, rtol=0, atol=1e-12):
                raise AssertionError(f"arm B != production pooled fn at length {length}")
    print("  production-equivalence check: arm B == pooled_router_forecaster_fn", flush=True)


def _layer2_rows(
    gate_inputs: dict[tuple, dict],
    cells: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Arm C: MetaRouter gate over {trail, pool} grids.

    ``gate_inputs`` maps (seed, world, scope, key) -> {"series": full
    series, "origins": [...], "A": [q5...], "B": [q5...]}. ``cells``
    carries one entry per scored cell (seed, world, origin, group,
    scope, y, weight) so gated grids expand to rows comparable with
    the other arms (global scope = one gated portfolio grid scored
    against every group, exactly as arms A/B/D score it).
    """
    cfg = MetaRouterConfig(
        val_window=12,
        min_val=6,
        margin=0.02,
        use_lockin=False,
        naive_name="trail",
        layer1_name="pool",
    )
    gated: dict[tuple, np.ndarray] = {}
    choices: dict[str, int] = {}
    choice_by_cell: dict[tuple, str] = {}
    for (seed, world, scope, key), gi in sorted(gate_inputs.items()):
        origins = gi["origins"]
        cand = {
            "trail": [exp72.interp9(q).reshape(9, 1) for q in gi["A"]],
            "pool": [exp72.interp9(q).reshape(9, 1) for q in gi["B"]],
        }
        actuals = [np.array([gi["series"][o]]) for o in origins]
        res = MetaRouter(cfg).run(gi["series"], origins, cand, actuals)
        for i, o in enumerate(origins):
            gated[(seed, world, scope, key, o)] = np.asarray(res.grids[i], dtype=float).ravel()
            choice_by_cell[(seed, world, scope, key, o)] = res.choices[i]
            choices[res.choices[i]] = choices.get(res.choices[i], 0) + 1
    rows: list[dict] = []
    for cell in cells:
        seed, world, scope = cell["seed"], cell["world"], cell["scope"]
        key = cell["group"] if scope == "group" else "__portfolio__"
        grid9 = gated[(seed, world, scope, key, cell["origin"])]
        q5 = np.interp(QUANTILE_LEVELS, L9, grid9)
        c80, c90 = exp78._coverage(q5, cell["y"])
        rows.append(
            {
                "seed": seed,
                "world": world,
                "origin": cell["origin"],
                "group": cell["group"],
                "arm": "C",
                "scope": scope,
                "crps": pinball_crps(grid9.reshape(9, 1), np.array([cell["y"]])),
                "cov80": c80,
                "cov90": c90,
                "weight": cell["weight"],
                "fallback": False,
                "choice": choice_by_cell[(seed, world, scope, key, cell["origin"])],
            }
        )
    return rows, choices


def _dump_rows(rows: list[dict], rows_path: Path) -> None:
    tmp = rows_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rows))
    tmp.replace(rows_path)


def run_worlds(
    seeds: list[int],
    worlds: list[str],
    variant: Variant,
    rows_path: Path | None = None,
    resume: bool = False,
    arms: tuple[str, ...] = ("A", "B", "B0", "C", "D"),
) -> list[dict]:
    """Run all (seed, world) combos, checkpointing rows after each one.

    A combo counts as complete once its completion-arm rows exist
    (arm C when requested — computed per combo, not at the end — else
    the last requested arm); with ``resume`` those combos are skipped,
    so a killed run loses at most the in-flight combo. ``arms`` allows
    splitting the expensive reference arm D into its own pass.
    """
    if "C" in arms and not {"A", "B"} <= set(arms):
        raise ValueError("arm C (Layer-2 gate) requires arms A and B")
    completion_arm = "C" if "C" in arms else arms[-1]
    rows: list[dict] = []
    completed: set[tuple[int, str]] = set()
    if resume and rows_path is not None and rows_path.exists():
        rows = json.loads(rows_path.read_text())
        completed = {(r["seed"], r["world"]) for r in rows if r["arm"] == completion_arm}
        print(f"resume: {len(rows)} rows loaded, combos done: {sorted(completed)}", flush=True)
    for seed in seeds:
        sd = exp72.build_seed_data(seed, "tree")
        for world in worlds:
            if (seed, world) in completed:
                print(f"  seed {seed} world {world}: skipped (checkpointed)", flush=True)
                continue
            gate_inputs: dict[tuple, dict] = {}
            cells: list[dict] = []
            groups = exp72.build_world(sd, world)
            gids = [g for g in groups if g != "__portfolio__"]
            n = len(sd.months)
            origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
            ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids}
            panel = {
                g: np.array(groups[g].attrition_rate, dtype=float) for g in [*gids, "__portfolio__"]
            }
            all_factories = {
                "A": lambda: v2_forecaster_fn(),
                "D": lambda: router_forecaster_fn(),
            }
            if {"B", "B0"} & set(arms):
                fn_b, fn_b0 = _pooled_fns(panel, variant)
                global _EQUIV_CHECKED
                if not _EQUIV_CHECKED:
                    _check_production_equivalence(variant)
                    _EQUIV_CHECKED = True
                all_factories["B"] = lambda fn_b=fn_b: fn_b
                all_factories["B0"] = lambda fn_b0=fn_b0: fn_b0
            factories = {k: v for k, v in all_factories.items() if k in arms}
            fns = {name: exp72.CachedForecaster(mk()) for name, mk in factories.items()}
            gfs = {name: GroupForecaster(fn, min_group_n=MIN_GROUP_N) for name, fn in fns.items()}
            for key in [*gids, "__portfolio__"]:
                for scope in ("group", "global"):
                    if (key == "__portfolio__") != (scope == "global"):
                        continue
                    gate_inputs[(seed, world, scope, key)] = {
                        "series": panel[key],
                        "origins": origins,
                        "A": [],
                        "B": [],
                    }
            for o in origins:
                trunc = {g: exp72._truncate(groups[g], o) for g in gids}
                trunc["__portfolio__"] = exp72._truncate(groups["__portfolio__"], o)
                ph = np.array(trunc["__portfolio__"].attrition_rate, dtype=float)
                fc_by_arm = {}
                global_by_arm = {}
                for name, gf in gfs.items():
                    fc_by_arm[name] = gf.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
                    global_by_arm[name] = fns[name](ph)
                for g in gids:
                    y = float(groups[g].attrition_rate[o])
                    w = float(ctrl_counts[g][o])
                    for scope in ("group", "global"):
                        cells.append(
                            {
                                "seed": seed,
                                "world": world,
                                "origin": o,
                                "group": g,
                                "scope": scope,
                                "y": y,
                                "weight": w,
                            }
                        )
                    if "C" in arms:
                        gate_inputs[(seed, world, "group", g)]["A"].append(
                            np.asarray(fc_by_arm["A"][g].quantiles, dtype=float)
                        )
                        gate_inputs[(seed, world, "group", g)]["B"].append(
                            np.asarray(fc_by_arm["B"][g].quantiles, dtype=float)
                        )
                    for name in factories:
                        for scope, q5 in (
                            ("group", fc_by_arm[name][g].quantiles),
                            ("global", global_by_arm[name]),
                        ):
                            q5 = np.asarray(q5, dtype=float)
                            grid = exp72.interp9(q5)
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
                                    "fallback": bool(fc_by_arm[name][g].used_fallback),
                                }
                            )
                if "C" in arms:
                    gate_inputs[(seed, world, "global", "__portfolio__")]["A"].append(
                        np.asarray(global_by_arm["A"], dtype=float)
                    )
                    gate_inputs[(seed, world, "global", "__portfolio__")]["B"].append(
                        np.asarray(global_by_arm["B"], dtype=float)
                    )
            print(f"  seed {seed} world {world}: arms {sorted(factories)} done", flush=True)
            if "C" in arms:
                c_rows, choices = _layer2_rows(gate_inputs, cells)
                rows.extend(c_rows)
                print(f"  seed {seed} world {world}: arm C done {choices}", flush=True)
            if rows_path is not None:
                _dump_rows(rows, rows_path)
            print(f"  seed {seed} world {world}: checkpointed", flush=True)
    return rows


def _paired(cell: dict, arm: str, ref: str) -> dict:
    diffs = np.array(
        [cell[k][arm] - cell[k][ref] for k in cell if arm in cell[k] and ref in cell[k]]
    )
    if len(diffs) == 0:
        return {
            "mean_diff": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "win_rate": float("nan"),
            "n_cells": 0,
        }
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
                if not arows:
                    continue
                wsum = sum(r["weight"] for r in arows)
                entry[arm] = {
                    "crps_mw": float(sum(r["crps"] * r["weight"] for r in arows) / wsum),
                    "cov80": float(sum(r["cov80"] * r["weight"] for r in arows) / wsum),
                    "cov90": float(sum(r["cov90"] * r["weight"] for r in arows) / wsum),
                }
                if arm != "A":
                    entry[arm]["vs_A"] = _paired(cell, arm, "A")
            entry["contrasts"] = {
                "B_vs_D": _paired(cell, "B", "D"),
                "B_vs_B0": _paired(cell, "B", "B0"),
                "C_vs_B": _paired(cell, "C", "B"),
            }
            out[world][scope] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    ap.add_argument("--worlds", nargs="+", default=["control", "S1_4.0"])
    ap.add_argument("--features", action="store_true", help="R2a: regime-feature subset")
    ap.add_argument(
        "--shrink", type=float, default=0.0, help="R2b: shrink center toward trailing mean"
    )
    ap.add_argument("--backtest-start", type=int, default=8, help="R2c: pool size knob")
    ap.add_argument("--target-weight", type=int, default=1, help="R3: partial pooling weight")
    ap.add_argument("--hard", action="store_true", help="R4: argmax-regime expert only")
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="R2e: scale-free (naive-normalized) CRPS in the pooled fit",
    )
    ap.add_argument("--out", type=str, default="exp82_results.json")
    ap.add_argument("--rows-out", type=str, default="exp82_rows.json")
    ap.add_argument(
        "--resume", action="store_true", help="skip (seed, world) combos already checkpointed"
    )
    ap.add_argument(
        "--arms",
        type=str,
        default="A,B,B0,C,D",
        help="comma-separated arms to run (C requires A,B); D can run as its own pass",
    )
    args = ap.parse_args()

    variant = Variant(args)
    t0 = time.time()
    rows_path = EXP_DIR / args.rows_out
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    rows = run_worlds(
        args.seeds, args.worlds, variant, rows_path=rows_path, resume=args.resume, arms=arms
    )
    choices: dict[str, int] = {}
    for r in rows:
        if r["arm"] == "C":
            choices[r["choice"]] = choices.get(r["choice"], 0) + 1
    payload: dict = {
        "variant": variant.describe(),
        "l2_choices": choices,
        "worlds_summary": summarize(rows),
    }
    print(f"rows checkpointed at {rows_path}")
    payload["runtime_s"] = round(time.time() - t0, 1)
    out_path = EXP_DIR / args.out
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path} in {payload['runtime_s']}s")


if __name__ == "__main__":
    main()
