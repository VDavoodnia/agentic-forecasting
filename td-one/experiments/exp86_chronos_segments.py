"""exp86: Chronos-2 as the segment engine.

Pre-registered in exp86_report.md. Question: does a zero-shot TSFM
beat the tuned local group forecaster (exp79 trailing-mean center +
Gaussian spread + exp80 thin/dead fallback) per segment, at h=1,
inside the production fallback semantics? And does exp81's pooled
conformal machinery (signed-score shrinkage, n0=24) improve Chronos's
ranges when pooled across segment groups?

Arms (all post-pass arms derive from stored per-cell forecast
vectors, so diagnosis rounds cost zero extra Chronos calls):

- A  CHAMPION : production v2_forecaster_fn() + GroupForecaster defaults
- B  CHRONOS  : Chronos-2 zero-shot forecaster_fn, same wrapper/fallback
- C  CHRONOS+POOL : B + exp81-style pooled signed-score conformal
- D  GATE     : exp73-style validation gate between A and B (diagnostic)
- variants (diagnosis): BC (Chronos q50 center + champion spread),
  blends (1-w)*A + w*B for w in {0.25, 0.5, 0.75}

Phases:
    run     : compute per-cell A/B forecast vectors (Chronos disk-cached)
    analyze : score all arms + variants from a rows file

Usage:
    .venv/bin/python experiments/exp86_chronos_segments.py run \
        --seeds 7 42 123 --rows-out exp86_rows_main.json
    .venv/bin/python experiments/exp86_chronos_segments.py analyze \
        --rows exp86_rows_main.json --out exp86_results_main.json
    # context-length variant pass (arm B only differs):
    .venv/bin/python experiments/exp86_chronos_segments.py run \
        --seeds 7 42 123 --context 12 --rows-out exp86_rows_ctx12.json
"""

from __future__ import annotations

import argparse
import hashlib
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
    _spread_offsets,
    v2_forecaster_fn,
)
from td_one.agent.layer2_tools import pinball_crps  # noqa: E402
from td_one.uq.methods import weighted_quantile  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
TD_DIR = EXP_DIR.parent
CACHE_DIR = EXP_DIR / "exp86_chronos_cache"
MIN_GROUP_N = 30  # exp72/79/80 setting
WORLDS = ["control", "S1_4.0", "S3_4.0"]
POOL_N0 = 24.0  # exp81 shipped shrinkage pseudo-count
POOL_LEVEL = 0.8
POOL_MIN_BANK = 8
GATE_MARGIN = 0.98  # exp73: challenger must win validation by >= 2%
GATE_MIN_VAL = 4
_EPS = 1e-12


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp72 = _load("exp72_integration", "exp72_integration.py")
exp78 = _load("exp78_group_forecaster_spread", "exp78_group_forecaster_spread.py")


# ---------------------------------------------------------------------------
# Chronos forecaster_fn with a persistent disk cache
# ---------------------------------------------------------------------------


class ChronosDiskForecaster:
    """Chronos-2 zero-shot forecaster_fn; native quantiles cached to disk.

    Cache key = sha256(variant tag + exact context bytes). A cache hit
    never touches the model, so diagnosis rounds and reruns are free
    and bit-reproducible. ``context`` truncates the history to its
    last N points (R4 context-length variants).
    """

    def __init__(self, context: int | None = None) -> None:
        from td_one.models.tsfm import Chronos2Wrapper

        self.context = context
        self.variant = "full" if context is None else f"last{context}"
        self.cache_dir = CACHE_DIR / self.variant
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wrapper = Chronos2Wrapper(
            prediction_length=1,
            weights_dir=str(TD_DIR / "models" / "weights" / "chronos2"),
        )
        self.n_model_calls = 0
        self.n_cache_hits = 0

    def _native21(self, ctx: np.ndarray) -> np.ndarray:
        key = hashlib.sha256(self.variant.encode() + ctx.tobytes()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.n_cache_hits += 1
            return np.array(json.loads(path.read_text())["native"], dtype=float)
        levels, vals = self.wrapper.forecast_quantiles(ctx, horizon=1, seed=42)
        native = np.asarray(vals, dtype=float)[:, 0]
        assert native.shape == (len(levels),)
        path.write_text(json.dumps({"levels": levels.tolist(), "native": native.tolist()}))
        self.n_model_calls += 1
        return native

    def __call__(self, series: np.ndarray) -> np.ndarray:
        from td_one.models.tsfm import CHRONOS2_LEVELS

        s = np.asarray(series, dtype=float)
        if len(s) == 0:
            raise ValueError("chronos forecaster needs a non-empty history")
        ctx = s if self.context is None else s[-self.context :]
        native = self._native21(ctx)
        q5 = np.interp(QUANTILE_LEVELS, CHRONOS2_LEVELS, native)
        return np.clip(np.maximum.accumulate(q5), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Phase 1: per-cell forecast vectors
# ---------------------------------------------------------------------------


def run_phase(seeds: list[int], rows_out: str, context: int | None) -> None:
    rows_path = EXP_DIR / rows_out
    done: list[str] = []
    cells: list[dict] = []
    if rows_path.exists():
        payload = json.loads(rows_path.read_text())
        done = payload["done"]
        cells = payload["cells"]
        print(f"resuming: {len(done)} (seed, world) combos already done", flush=True)

    chronos = ChronosDiskForecaster(context=context)
    champ_fn = v2_forecaster_fn()
    t0 = time.time()
    for seed in seeds:
        sd = exp72.build_seed_data(seed, "tree")
        gids_all = [g for g in sd.base_series if g != "__portfolio__"]
        ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids_all}
        n = len(sd.months)
        origins = list(range(exp72.ORIGIN_START, n - exp72.ORIGIN_STOP_TAIL))
        for world in WORLDS:
            tag = f"{seed}:{world}"
            if tag in done:
                continue
            groups = exp72.build_world(sd, world)
            gids = [g for g in groups if g != "__portfolio__"]
            gf_a = GroupForecaster(champ_fn, min_group_n=MIN_GROUP_N)
            gf_b = GroupForecaster(chronos, min_group_n=MIN_GROUP_N)
            for o in origins:
                trunc = {g: exp72._truncate(groups[g], o) for g in groups}
                ph = np.array(trunc["__portfolio__"].attrition_rate, dtype=float)
                fc_a = gf_a.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
                fc_b = gf_b.forecast(trunc, train_months=exp72.TRAIN_WINDOW)
                glob_a = champ_fn(ph)
                glob_b = chronos(ph)
                off_port = _spread_offsets(ph, "gaussian")
                cells.append(
                    {
                        "seed": seed,
                        "world": world,
                        "origin": o,
                        "group": "__portfolio__",
                        "scope": "global",
                        "y": float(groups["__portfolio__"].attrition_rate[o]),
                        "weight": 0.0,
                        "fallback": False,
                        "qA": glob_a.tolist(),
                        "qB": glob_b.tolist(),
                        "off": off_port.tolist(),
                    }
                )
                for g in gids:
                    own_hist = np.array(trunc[g].attrition_rate, dtype=float)
                    cells.append(
                        {
                            "seed": seed,
                            "world": world,
                            "origin": o,
                            "group": g,
                            "scope": "group",
                            "y": float(groups[g].attrition_rate[o]),
                            "weight": float(ctrl_counts[g][o]),
                            "fallback": bool(fc_b[g].used_fallback),
                            "qA": np.asarray(fc_a[g].quantiles, dtype=float).tolist(),
                            "qB": np.asarray(fc_b[g].quantiles, dtype=float).tolist(),
                            "off": (
                                off_port
                                if fc_b[g].used_fallback
                                else _spread_offsets(own_hist, "gaussian")
                            ).tolist(),
                        }
                    )
            done.append(tag)
            rows_path.write_text(json.dumps({"done": done, "cells": cells}))
            print(
                f"  seed {seed} world {world}: done "
                f"(chronos calls {chronos.n_model_calls}, cache hits {chronos.n_cache_hits}, "
                f"{time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
    print(
        f"run phase complete: {len(cells)} cells, {chronos.n_model_calls} model calls, "
        f"{chronos.n_cache_hits} cache hits, {time.time() - t0:.0f}s"
    )


# ---------------------------------------------------------------------------
# Phase 2: arms + scoring
# ---------------------------------------------------------------------------


def _score(q5: np.ndarray, y: float) -> tuple[float, float, float]:
    grid = exp72.interp9(np.asarray(q5, dtype=float))
    crps = pinball_crps(grid.reshape(9, 1), np.array([y]))
    c80, c90 = exp78._coverage(np.asarray(q5, dtype=float), y)
    return float(crps), float(c80), float(c90)


def _signed(q5: np.ndarray, y: float) -> float:
    return max(float(q5[0]) - y, y - float(q5[4]))


def _arm_c(cells: list[dict]) -> dict[tuple, np.ndarray]:
    """exp81-style pooled signed-score conformal on arm B (group scope)."""
    out: dict[tuple, np.ndarray] = {}
    by_sw: dict[tuple[int, str], list[dict]] = {}
    for c in cells:
        if c["scope"] == "group":
            by_sw.setdefault((c["seed"], c["world"]), []).append(c)
    for group_cells in by_sw.values():
        group_cells.sort(key=lambda c: (c["origin"], c["group"]))
        for c in group_cells:
            q = np.array(c["qB"], dtype=float)
            own = [
                b
                for b in group_cells
                if b["group"] == c["group"] and b["origin"] < c["origin"]
            ]
            cross = [
                b
                for b in group_cells
                if b["group"] != c["group"] and b["origin"] < c["origin"]
            ]
            n_own, n_cross = len(own), len(cross)
            if n_own + n_cross < POOL_MIN_BANK:
                out[(c["seed"], c["world"], c["origin"], c["group"])] = q
                continue
            m, q10, q90 = float(q[2]), float(q[0]), float(q[4])
            mean_half = max((q90 - q10) / 2.0, _EPS)
            lam = n_own / (n_own + POOL_N0) if n_own > 0 else 0.0
            vals: list[np.ndarray] = []
            wts: list[np.ndarray] = []
            if n_own:
                own_s = np.array([_signed(np.array(b["qB"], dtype=float), b["y"]) for b in own])
                vals.append(own_s / mean_half)
                wts.append(np.full(n_own, lam / n_own))
            if n_cross:
                crs = []
                for b in cross:
                    qb = np.array(b["qB"], dtype=float)
                    s = _signed(qb, b["y"])
                    lo_ex = float(qb[0]) - b["y"]
                    hi_ex = b["y"] - float(qb[4])
                    half = (
                        (float(qb[2]) - float(qb[0]))
                        if lo_ex >= hi_ex
                        else (float(qb[4]) - float(qb[2]))
                    )
                    crs.append(s / max(half, _EPS))
                vals.append(np.array(crs))
                wts.append(np.full(n_cross, (1.0 - lam) / n_cross))
            adj = weighted_quantile(np.concatenate(vals), np.concatenate(wts), POOL_LEVEL)
            q = q.copy()
            q[0] = q10 - adj * max(m - q10, _EPS)
            q[4] = q90 + adj * max(q90 - m, _EPS)
            out[(c["seed"], c["world"], c["origin"], c["group"])] = np.clip(
                np.maximum.accumulate(q), 0.0, 1.0
            )
    return out


def _arm_d_choice(cells: list[dict]) -> dict[tuple, str]:
    """Pick A or B per (seed, world, group-key) by walk-forward validation."""
    scored: dict[tuple, dict[str, float]] = {}
    for c in cells:
        key = (c["seed"], c["world"], c["origin"], c["group"])
        scored[key] = {
            "A": _score(np.array(c["qA"], dtype=float), c["y"])[0],
            "B": _score(np.array(c["qB"], dtype=float), c["y"])[0],
        }
    choice: dict[tuple, str] = {}
    series_keys = sorted({(c["seed"], c["world"], c["group"]) for c in cells})
    for seed, world, g in series_keys:
        origins = sorted(
            {c["origin"] for c in cells if (c["seed"], c["world"], c["group"]) == (seed, world, g)}
        )
        for o in origins:
            prior = [
                scored[(seed, world, oo, g)]
                for oo in origins
                if oo < o and (seed, world, oo, g) in scored
            ]
            pick = "A"
            if len(prior) >= GATE_MIN_VAL:
                ma = float(np.mean([p["A"] for p in prior]))
                mb = float(np.mean([p["B"] for p in prior]))
                if mb <= GATE_MARGIN * ma:
                    pick = "B"
            choice[(seed, world, o, g)] = pick
    return choice


def analyze(rows_file: str, out_file: str) -> dict:
    payload = json.loads((EXP_DIR / rows_file).read_text())
    cells: list[dict] = payload["cells"]
    c_map = _arm_c(cells)
    d_choice = _arm_d_choice(cells)

    arm_names = ["A", "B", "C", "D", "BC", "BL25", "BL50", "BL75"]

    def arm_q(c: dict, arm: str) -> np.ndarray:
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
        if arm == "BC":
            center = float(qb[2])
            q = center + np.array(c["off"], dtype=float)
            return np.clip(np.maximum.accumulate(q), 0.0, 1.0)
        w = {"BL25": 0.25, "BL50": 0.5, "BL75": 0.75}[arm]
        return np.clip(np.maximum.accumulate((1.0 - w) * qa + w * qb), 0.0, 1.0)

    # score every arm on every cell
    scored: dict[tuple, dict[str, tuple[float, float, float]]] = {}
    for c in cells:
        key = (c["seed"], c["world"], c["origin"], c["group"], c["scope"])
        scored[key] = {arm: _score(arm_q(c, arm), c["y"]) for arm in arm_names}

    def skey(c: dict) -> tuple:
        return (c["seed"], c["world"], c["origin"], c["group"], c["scope"])

    def paired(world: str, scope: str, arm: str, ref: str = "A") -> dict:
        diffs = []
        for c in cells:
            if c["world"] == world and c["scope"] == scope:
                key = (c["seed"], c["world"], c["origin"], c["group"], c["scope"])
                diffs.append(scored[key][arm][0] - scored[key][ref][0])
        d = np.array(diffs)
        t = stats.ttest_1samp(d, 0.0)
        return {
            "mean_diff": float(np.mean(d)),
            "t": float(t.statistic),
            "p": float(t.pvalue),
            "win_rate": float(np.mean(d < 0)),
            "n_cells": int(len(d)),
        }

    summary: dict = {"arms": {}, "degradation": {}, "per_seed_control_group": {}}
    for world in WORLDS:
        summary["arms"][world] = {}
        for scope in ("group", "global"):
            sub = [c for c in cells if c["world"] == world and c["scope"] == scope]
            entry: dict = {}
            for arm in arm_names:
                if scope == "group":
                    wsum = sum(c["weight"] for c in sub)
                    entry[arm] = {
                        "crps_mw": float(
                            sum(
                                scored[skey(c)][arm][0] * c["weight"] for c in sub
                            )
                            / wsum
                        ),
                        "cov80": float(
                            sum(
                                scored[skey(c)][arm][1] * c["weight"] for c in sub
                            )
                            / wsum
                        ),
                        "cov90": float(
                            sum(
                                scored[skey(c)][arm][2] * c["weight"] for c in sub
                            )
                            / wsum
                        ),
                        "width80": float(
                            np.mean([arm_q(c, arm)[4] - arm_q(c, arm)[0] for c in sub])
                        ),
                    }
                else:
                    entry[arm] = {
                        "crps": float(
                            np.mean(
                                [
                                    scored[skey(c)][arm][0] for c in sub
                                ]
                            )
                        ),
                        "cov80": float(
                            np.mean(
                                [
                                    scored[skey(c)][arm][1] for c in sub
                                ]
                            )
                        ),
                    }
                if arm != "A":
                    entry[arm]["vs_A"] = paired(world, scope, arm)
            summary["arms"][world][scope] = entry
    for arm in arm_names:
        summary["degradation"][arm] = {
            "group": summary["arms"]["S1_4.0"]["group"][arm]["crps_mw"]
            - summary["arms"]["control"]["group"][arm]["crps_mw"],
            "global": summary["arms"]["S1_4.0"]["global"][arm]["crps"]
            - summary["arms"]["control"]["global"][arm]["crps"],
        }
    # R1: per-seed control group-scope B-vs-A and C-vs-A diffs
    for seed in sorted({c["seed"] for c in cells}):
        row: dict = {}
        for arm in ("B", "C", "D"):
            diffs = [
                scored[skey(c)][arm][0] - scored[skey(c)]["A"][0]
                for c in cells
                if c["seed"] == seed and c["world"] == "control" and c["scope"] == "group"
            ]
            d = np.array(diffs)
            row[arm] = {"mean_diff": float(np.mean(d)), "n": int(len(d))}
        summary["per_seed_control_group"][str(seed)] = row

    (EXP_DIR / out_file).write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary: dict) -> None:
    for world in WORLDS:
        print(f"== {world} / group ==")
        for arm, e in summary["arms"][world]["group"].items():
            vs = e.get("vs_A", {})
            print(
                f"  {arm:5s} CRPS {e['crps_mw']:.5f} cov80 {e['cov80']:.3f} "
                f"cov90 {e['cov90']:.3f} width {e['width80']:.4f} "
                + (
                    f"vsA {vs['mean_diff']:+.5f} p={vs['p']:.3g} win {vs['win_rate']:.2f}"
                    if vs
                    else "(champion)"
                )
            )
        print(f"== {world} / global ==")
        for arm, e in summary["arms"][world]["global"].items():
            print(f"  {arm:5s} CRPS {e['crps']:.5f} cov80 {e['cov80']:.3f}")
    print("== degradation (S1 - control, group) ==")
    for arm, d in summary["degradation"].items():
        print(f"  {arm:5s} {d['group']:+.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "analyze"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--rows-out", type=str, default="exp86_rows_main.json")
    ap.add_argument("--rows", type=str, default="exp86_rows_main.json")
    ap.add_argument("--out", type=str, default="exp86_results_main.json")
    args = ap.parse_args()
    if args.phase == "run":
        run_phase(args.seeds, args.rows_out, args.context)
    else:
        summary = analyze(args.rows, args.out)
        _print_summary(summary)
        print(f"wrote {EXP_DIR / args.out}")


if __name__ == "__main__":
    main()
