"""exp72: Integration — Track 2 shifts stress-test Track 1's group forecaster.

Design (pre-registered in experiments/exp72_report.md BEFORE round 0):

1. MortgageSimulatorV2 portfolios (deterministic seed per world-family).
2. Deterministic groups fit on PRE-ONSET metadata only (PIT); membership
   is fixed across the window (except S7mix, where membership drift is
   itself the shift).
3. Per-group monthly series via build_group_series. Shifts from Track 2's
   machinery applied at the group-series level:
   - S1 (rate shock): apply_shift, all groups.
   - S3 (regional divergence): apply_divergence — groups whose modal
     province is ON shift up, all others down.
   - S7 (composition drift): (a) S7rate = apply_shift logit drift;
     (b) S7mix = post-onset episode migration donor->recipient group.
4. Rolling origins (h=1), exp67-style origin classes (pre/straddle/post).
   Arms: GROUP (GroupForecaster incl. min_group_n fallback), GLOBAL
   (portfolio forecast applied to every group), LAYER2 (MetaRouter over
   candidates {naive, v2=group, global}), plus NOFB (group forecaster
   with min_group_n=1) for the fallback question.
5. Placebo control: CRPS diffs (shifted - control, same seed/origin/
   group) — pre-onset diffs must be exactly 0 (bit-identical histories).
6. Metrics: CRPS (pinball, 9-level grid, layer2_tools.pinball_crps),
   member-weighted pooled diffs, paired t-test + win rate.

Usage:
    .venv/bin/python experiments/exp72_integration.py --seeds 7 --worlds control,S1_4.0
    .venv/bin/python experiments/exp72_integration.py            # full round 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.deterministic_grouper import fit_deterministic_groups  # noqa: E402
from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn  # noqa: E402
from td_one.agent.grouping import GroupDefinition  # noqa: E402
from td_one.agent.layer2_tools import LEVELS_9, pinball_crps  # noqa: E402
from td_one.agent.meta_router import MetaRouter, MetaRouterConfig  # noqa: E402
from td_one.data_gen.group_series import GroupSeries, build_group_series, portfolio_totals  # noqa: E402
from td_one.data_gen.shift_scenarios import ShiftSpec, apply_divergence, apply_shift  # noqa: E402

# --- pre-registered config (round 0) -------------------------------------
SEEDS = [7, 42]
N_CUSTOMERS = 10000
N_MONTHS = 48
N_CLUSTERS = 4
GROUPER_SEED = 42
MIN_GROUP_N = 150
TRAIN_WINDOW = 6
ORIGIN_START = 12
ORIGIN_STOP_TAIL = 8  # stop this many months before the last outcome month
SEVERITIES = (1.0, 4.0)
STRADDLE_LEN = 6  # straddle = [onset, onset + STRADDLE_LEN)
MIGRATION_RATE = 0.06  # S7mix: donor-episode migration prob, +6pp/month
Q5_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
L9 = np.array(LEVELS_9)

ALL_WORLDS = (
    ["control"]
    + [f"S1_{s}" for s in SEVERITIES]
    + [f"S3_{s}" for s in SEVERITIES]
    + [f"S7_{s}" for s in SEVERITIES]
    + ["S7mix"]
)


# --- small helpers ---------------------------------------------------------


class CachedForecaster:
    """Memoize a forecaster_fn on the exact history bytes.

    Worlds share bit-identical pre-onset histories, so the cache pays
    off across arms and worlds.
    """

    def __init__(self, fn) -> None:
        self.fn = fn
        self.cache: dict[bytes, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, series: np.ndarray) -> np.ndarray:
        key = np.asarray(series, dtype=float).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        out = self.fn(np.asarray(series, dtype=float))
        self.cache[key] = np.asarray(out, dtype=float)
        return self.cache[key]


def interp9(q5: np.ndarray) -> np.ndarray:
    """Interp a 5-level quantile vector onto the 9-level grid, monotone."""
    q = np.asarray(q5, dtype=float).ravel()
    if len(q) < len(Q5_LEVELS):
        q = np.pad(q, (0, len(Q5_LEVELS) - len(q)), mode="edge")
    vals = np.interp(L9, Q5_LEVELS, q[: len(Q5_LEVELS)])
    return np.clip(np.maximum.accumulate(vals), 0.0, 1.0)


def naive_grid9(ctx: np.ndarray) -> np.ndarray:
    """Trailing-mean + Gaussian residual quantiles on the 9-level grid."""
    ctx = np.asarray(ctx, dtype=float)
    w = min(6, len(ctx))
    mu = float(np.mean(ctx[-w:])) if len(ctx) else 0.0
    tail = ctx[-min(24, len(ctx)) :]
    resid = tail - np.mean(tail) if len(tail) else np.array([0.01])
    sd = max(float(np.std(resid)) if len(resid) > 1 else 0.01, 1e-4)
    z = stats.norm.ppf(L9)
    return np.clip(mu + z * sd, 0.0, 1.0)


def assign_by_rules(df: pd.DataFrame, groups: list[GroupDefinition]) -> dict[str, str]:
    """Vectorized rule application; each row must match exactly one rule."""
    assign: dict[str, str] = {}
    for g in groups:
        for mid in df.query(g.rule)["mortgage_id"].astype(str):
            if mid in assign:
                raise ValueError(f"mortgage {mid} matches multiple groups")
            assign[mid] = g.group_id
    missing = set(df["mortgage_id"].astype(str)) - set(assign)
    if missing:
        raise ValueError(f"{len(missing)} rows match no group")
    return assign


def origin_class(o: int, onset: int, n_months: int) -> str:
    if o < onset:
        return "pre"
    if o < onset + STRADDLE_LEN:
        return "straddle"
    return "post"


def _truncate(gs: GroupSeries, o: int) -> GroupSeries:
    return GroupSeries(
        group_id=gs.group_id,
        months=gs.months[:o],
        n_at_risk=gs.n_at_risk[:o],
        n_switched=gs.n_switched[:o],
    )


@dataclass
class SeedData:
    seed: int
    months: list[date]
    onset: int
    groups: list[GroupDefinition]
    region_of: dict[str, str]
    donor: str
    recipient: str
    base_series: dict[str, GroupSeries]  # includes __portfolio__
    out_df: pd.DataFrame
    group_of: dict[str, str]
    train_rates: dict[str, float]


def build_seed_data(seed: int) -> SeedData:
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2

    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed, use_fred_macro=False
    )
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    out_df = pd.DataFrame(
        {
            "mortgage_id": [o.mortgage_id for o in outcomes],
            "outcome_observed_at": [o.outcome_observed_at for o in outcomes],
            "switched_lender_flag": [bool(o.switched_lender_flag) for o in outcomes],
        }
    )
    out_df["month"] = pd.to_datetime(out_df["outcome_observed_at"]).dt.date.apply(
        lambda d: date(d.year, d.month, 1)
    )
    months = sorted(out_df["month"].unique())
    if len(months) < 18:
        raise RuntimeError(f"seed {seed}: only {len(months)} outcome months")
    onset = len(months) // 2
    cutoff = months[onset]

    hist_df = pd.DataFrame([r.model_dump() for r in history])
    hist_df["snapshot_date"] = pd.to_datetime(hist_df["snapshot_date"])
    meta_train = (
        hist_df[hist_df["snapshot_date"] < pd.Timestamp(cutoff)]
        .sort_values("snapshot_date")
        .groupby("mortgage_id")
        .last()
        .reset_index()
    )
    det = fit_deterministic_groups(
        meta_train, n_clusters=N_CLUSTERS, seed=GROUPER_SEED, min_group_n=MIN_GROUP_N
    )
    out_mids = set(out_df["mortgage_id"].astype(str))
    meta_out = meta_train[meta_train["mortgage_id"].astype(str).isin(out_mids)].copy()
    group_of = assign_by_rules(meta_out, det.groups)
    # Borrowers with outcomes but no pre-onset snapshot: cannot assign
    # PIT-cleanly; drop their episodes (counted).
    keep = out_df["mortgage_id"].astype(str).isin(group_of)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  seed {seed}: dropped {n_dropped} unassignable episodes")
    out_df = out_df[keep].copy()

    series = build_group_series(out_df, group_of)
    series["__portfolio__"] = portfolio_totals(series)

    # Region map for S3: modal province per group (from training meta).
    meta_out = meta_out.copy()
    meta_out["group_id"] = meta_out["mortgage_id"].astype(str).map(group_of)
    modal_prov = meta_out.groupby("group_id")["province"].agg(
        lambda s: s.mode().iloc[0] if len(s) else "?"
    )
    region_of = {gid: ("A" if p == "ON" else "B") for gid, p in modal_prov.items()}
    if "A" not in region_of.values():  # fallback: highest ON share -> A
        on_share = (
            meta_out.assign(is_on=meta_out["province"] == "ON").groupby("group_id")["is_on"].mean()
        )
        region_of[str(on_share.idxmax())] = "A"

    # Donor/recipient for S7mix by pre-onset realised attrition.
    train_rates = {
        gid: float(np.mean(gs.attrition_rate[:onset]))
        for gid, gs in series.items()
        if gid != "__portfolio__"
    }
    donor = max(train_rates, key=train_rates.get)  # type: ignore[arg-type]
    recipient = min(train_rates, key=train_rates.get)  # type: ignore[arg-type]
    return SeedData(
        seed=seed,
        months=months,
        onset=onset,
        groups=det.groups,
        region_of=region_of,
        donor=donor,
        recipient=recipient,
        base_series=series,
        out_df=out_df,
        group_of=group_of,
        train_rates=train_rates,
    )


def _to_shift_dict(gs: GroupSeries) -> dict[str, np.ndarray]:
    return {
        "n_renewals": np.array(gs.n_at_risk, dtype=float),
        "n_switches": np.array(gs.n_switched, dtype=float),
        "true_rates": np.array(gs.attrition_rate, dtype=float),
        "attrition_rate": np.array(gs.attrition_rate, dtype=float),
    }


def _from_shift_dict(gid: str, months: list[date], d: dict[str, np.ndarray]) -> GroupSeries:
    return GroupSeries(
        group_id=gid,
        months=list(months),
        n_at_risk=[int(round(x)) for x in d["n_renewals"]],
        n_switched=[int(round(x)) for x in d["n_switches"]],
    )


def build_world(sd: SeedData, world: str) -> dict[str, GroupSeries]:
    """Build the per-group series for one world (control or shifted)."""
    base = {g: gs for g, gs in sd.base_series.items() if g != "__portfolio__"}
    if world == "control":
        groups = dict(base)
    elif world == "S7mix":
        rng = np.random.default_rng(50_000 + sd.seed)
        df = sd.out_df.copy()
        df["group_id"] = df["mortgage_id"].astype(str).map(sd.group_of)
        df["midx"] = df["month"].apply(sd.months.index)
        mask = (df["group_id"] == sd.donor) & (df["midx"] >= sd.onset)
        p = np.where(
            mask,
            np.minimum(1.0, MIGRATION_RATE * (df["midx"] - sd.onset + 1)),
            0.0,
        )
        move = rng.random(len(df)) < p
        df.loc[move, "group_id"] = sd.recipient
        new_group_of = dict(zip(df["mortgage_id"].astype(str), df["group_id"], strict=True))
        groups = build_group_series(df, new_group_of)
        for gid in base:  # a group that vanished entirely
            if gid not in groups:
                groups[gid] = GroupSeries(
                    gid, list(sd.months), [0] * len(sd.months), [0] * len(sd.months)
                )
    else:
        shift, sev_s = world.split("_")
        sev = float(sev_s)
        spec = ShiftSpec(shift, sd.onset, sev)
        rng = np.random.default_rng(60_000 + sd.seed + int(sev * 10))
        if shift == "S3":
            dicts = {g: _to_shift_dict(gs) for g, gs in base.items()}
            shifted = apply_divergence(dicts, sd.region_of, spec, rng)
            groups = {g: _from_shift_dict(g, sd.months, shifted[g]) for g in base}
        else:
            groups = {}
            for i, (g, gs) in enumerate(base.items()):
                r = np.random.default_rng(rng.integers(0, 2**31) + i)
                groups[g] = _from_shift_dict(g, sd.months, apply_shift(_to_shift_dict(gs), spec, r))
    out = dict(groups)
    out["__portfolio__"] = portfolio_totals(groups)
    return out


# --- arms -------------------------------------------------------------------


def run_world(
    sd: SeedData,
    world: str,
    groups: dict[str, GroupSeries],
    fcaster: CachedForecaster,
    l2_cfg: MetaRouterConfig,
) -> list[dict]:
    """Rolling-origin forecasts for all arms in one (seed, world)."""
    months = sd.months
    n = len(months)
    origins = list(range(ORIGIN_START, n - ORIGIN_STOP_TAIL))
    gids = [g for g in groups if g != "__portfolio__"]
    rows: list[dict] = []
    gf_default = GroupForecaster(fcaster, min_group_n=MIN_GROUP_N)
    gf_nofb = GroupForecaster(fcaster, min_group_n=1)

    # Pre-compute per-origin group/global grids for LAYER2 candidates.
    grids_by_origin: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    actuals_by_group: dict[str, list[np.ndarray]] = {g: [] for g in gids}
    for o in origins:
        trunc_p = _truncate(groups["__portfolio__"], o)
        global_q5 = fcaster(np.array(trunc_p.attrition_rate, dtype=float))
        global_grid = interp9(global_q5)
        per_group: dict[str, dict[str, np.ndarray]] = {}
        trunc_groups = {g: _truncate(groups[g], o) for g in gids}
        trunc_groups["__portfolio__"] = trunc_p
        fc = gf_default.forecast(trunc_groups, train_months=TRAIN_WINDOW)
        fc_nofb = gf_nofb.forecast(trunc_groups, train_months=TRAIN_WINDOW)
        for g in gids:
            hist = np.array(trunc_groups[g].attrition_rate, dtype=float)
            v2_grid = interp9(fc[g].quantiles)
            per_group[g] = {
                "naive": naive_grid9(hist),
                "v2": v2_grid,
                "global": global_grid,
                "_fb": np.array([1.0 if fc[g].used_fallback else 0.0]),
                "_nofb_grid": interp9(fc_nofb[g].quantiles),
            }
            actuals_by_group[g].append(np.array([groups[g].attrition_rate[o]]))
        grids_by_origin[o] = per_group

    # LAYER2: one MetaRouter run per group over the origin sequence.
    l2_grids: dict[str, list[np.ndarray]] = {}
    l2_choices: dict[str, list[str]] = {}
    for g in gids:
        y = np.array(groups[g].attrition_rate, dtype=float)
        cand = {
            name: [grids_by_origin[o][g][name] for o in origins]
            for name in ("naive", "v2", "global")
        }
        cand = {k: [v.reshape(9, 1) for v in vs] for k, vs in cand.items()}
        res = MetaRouter(l2_cfg).run(y, origins, cand, actuals_by_group[g])
        l2_grids[g] = [np.asarray(gr).ravel() for gr in res.grids]
        l2_choices[g] = list(res.choices)

    ctrl_counts = {g: sd.base_series[g].n_at_risk for g in gids}
    for oi, o in enumerate(origins):
        oclass = origin_class(o, sd.onset, n)
        for g in gids:
            actual = groups[g].attrition_rate[o]
            w = float(ctrl_counts[g][o])
            pg = grids_by_origin[o][g]
            arm_grids = {
                "GROUP": pg["v2"],
                "GLOBAL": pg["global"],
                "LAYER2": l2_grids[g][oi],
                "NOFB": pg["_nofb_grid"],
            }
            for arm, grid in arm_grids.items():
                rows.append(
                    {
                        "seed": sd.seed,
                        "world": world,
                        "origin": o,
                        "origin_class": oclass,
                        "group_id": g,
                        "arm": arm,
                        "crps": pinball_crps(np.asarray(grid).reshape(9, 1), np.array([actual])),
                        "weight": w,
                        "fallback": bool(pg["_fb"][0]) if arm in ("GROUP", "LAYER2") else None,
                        "l2_choice": l2_choices[g][oi] if arm == "LAYER2" else None,
                        "n_at_risk": int(groups[g].n_at_risk[o]),
                    }
                )
    return rows


# --- analysis ---------------------------------------------------------------


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of a vs b (reports a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {
            "n": int(len(d)),
            "mean_diff": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "win_rate": float("nan"),
        }
    t, p = stats.ttest_1samp(d, 0.0)
    return {
        "n": int(len(d)),
        "mean_diff": float(np.mean(d)),
        "t": float(t),
        "p": float(p),
        "win_rate": float(np.mean(d < 0)),
    }


def analyze(rows: list[dict], onset_by_seed: dict[int, int]) -> dict:
    df = pd.DataFrame(rows)
    out: dict = {}
    ctrl = df[df.world == "control"].set_index(["seed", "origin", "group_id", "arm"])["crps"]
    worlds = [w for w in sorted(df.world.unique()) if w != "control"]
    deg: dict[str, dict] = {}
    for world in worlds:
        sub = df[df.world == world]
        for arm in ("GROUP", "GLOBAL", "LAYER2", "NOFB"):
            s = sub[sub.arm == arm].set_index(["seed", "origin", "group_id", "arm"])["crps"]
            idx = s.index.intersection(ctrl.index)
            diff = (s.loc[idx] - ctrl.loc[idx]).to_numpy()
            wts = (
                sub[sub.arm == arm]
                .set_index(["seed", "origin", "group_id", "arm"])["weight"]
                .loc[idx]
                .to_numpy()
            )
            classes = (
                sub[sub.arm == arm]
                .set_index(["seed", "origin", "group_id", "arm"])["origin_class"]
                .loc[idx]
                .to_numpy()
            )
            entry: dict = {
                "all": {
                    "n": int(len(diff)),
                    "mean_diff": float(np.mean(diff)) if len(diff) else float("nan"),
                    "p": float(stats.ttest_1samp(diff, 0.0).pvalue)
                    if len(diff) > 1
                    else float("nan"),
                    "win_rate": float(np.mean(diff < 0)) if len(diff) else float("nan"),
                    "wmean_diff": float(np.average(diff, weights=wts))
                    if len(diff)
                    else float("nan"),
                }
            }
            for cls in ("pre", "straddle", "post", "straddle_post"):
                m = (classes == cls) if cls != "straddle_post" else (classes != "pre")
                if m.sum() > 1:
                    entry[cls] = {
                        "n": int(m.sum()),
                        "mean_diff": float(np.mean(diff[m])),
                        "p": float(stats.ttest_1samp(diff[m], 0.0).pvalue),
                        "win_rate": float(np.mean(diff[m] < 0)),
                        "wmean_diff": float(np.average(diff[m], weights=wts[m])),
                    }
            deg[f"{world}/{arm}"] = entry
    out["degradation"] = deg

    # Q1: S3@4 GROUP vs GLOBAL degradation (diff of diffs), straddle+post.
    def _diff_frame(world: str, arm: str) -> pd.DataFrame:
        sub = df[(df.world == world) & (df.arm == arm)].set_index(["seed", "origin", "group_id"])
        c = df[(df.world == "control") & (df.arm == arm)].set_index(["seed", "origin", "group_id"])
        idx = sub.index.intersection(c.index)
        d = (sub.loc[idx, "crps"] - c.loc[idx, "crps"]).rename("diff")
        cls = sub.loc[idx, "origin_class"].rename("cls")
        return pd.concat([d, cls], axis=1)

    for world in ("S3_4.0", "S3_1.0", "S1_4.0", "S1_1.0"):
        try:
            dg = _diff_frame(world, "GROUP")
            dl = _diff_frame(world, "GLOBAL")
            idx = dg.index.intersection(dl.index)
            both = pd.concat(
                [
                    dg.loc[idx, "diff"].rename("group"),
                    dl.loc[idx, "diff"].rename("glob"),
                    dg.loc[idx, "cls"],
                ],
                axis=1,
            )
            sp = both[both.cls != "pre"]
            out[f"Q1_{world}"] = {
                "group_minus_global_deg": _paired(sp["group"].to_numpy(), sp["glob"].to_numpy()),
                "n": int(len(sp)),
            }
        except Exception as e:  # noqa: BLE001
            out[f"Q1_{world}"] = {"error": str(e)}

    # Q3: LAYER2 vs GROUP within shifted worlds (and control do-no-harm).
    for world in ["control"] + worlds:
        try:
            a = df[(df.world == world) & (df.arm == "LAYER2")].set_index(
                ["seed", "origin", "group_id"]
            )["crps"]
            b = df[(df.world == world) & (df.arm == "GROUP")].set_index(
                ["seed", "origin", "group_id"]
            )
            cls = b["origin_class"]
            b = b["crps"]
            idx = a.index.intersection(b.index)
            cc = cls.loc[idx].to_numpy()
            entry = {"all": _paired(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())}
            for cl in ("pre", "straddle", "post"):
                m = cc == cl
                if m.sum() > 1:
                    entry[cl] = _paired(a.loc[idx].to_numpy()[m], b.loc[idx].to_numpy()[m])
            out[f"Q3_{world}"] = entry
        except Exception as e:  # noqa: BLE001
            out[f"Q3_{world}"] = {"error": str(e)}

    # Fallback rates.
    fb = (
        df[df.arm == "GROUP"]
        .groupby(["world", "origin_class"])["fallback"]
        .mean()
        .round(3)
        .to_dict()
    )
    out["fallback_rates"] = {f"{k[0]}/{k[1]}": float(v) for k, v in fb.items()}
    # Placebo gate: max |pre diff| across worlds/arms.
    mx = 0.0
    for _key, e in deg.items():
        if "pre" in e:
            mx = max(mx, abs(e["pre"]["mean_diff"]))
    out["placebo_max_abs_pre_diff"] = float(mx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--worlds", default=",".join(ALL_WORLDS))
    ap.add_argument("--val-window", type=int, default=12)
    ap.add_argument("--min-val", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--out", default="experiments/exp72_results.json")
    ap.add_argument("--tag", default="round0")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    worlds = [w for w in args.worlds.split(",") if w]
    l2_cfg = MetaRouterConfig(
        val_window=args.val_window, min_val=args.min_val, margin=args.margin, use_lockin=False
    )
    fcaster = CachedForecaster(v2_forecaster_fn())
    all_rows: list[dict] = []
    meta: dict = {"config": vars(args), "seeds_info": {}}
    t0 = time.time()
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        sd = build_seed_data(seed)
        print(
            f"  months={len(sd.months)} onset={sd.onset} groups={len(sd.groups)} "
            f"regions={sd.region_of} donor={sd.donor} recip={sd.recipient}",
            flush=True,
        )
        meta["seeds_info"][str(seed)] = {
            "n_months": len(sd.months),
            "onset": sd.onset,
            "n_groups": len(sd.groups),
            "region_of": sd.region_of,
            "donor": sd.donor,
            "recipient": sd.recipient,
            "rules": [{"group_id": g.group_id, "rule": g.rule} for g in sd.groups],
            "train_rates": sd.train_rates,
        }
        for world in worlds:
            groups = build_world(sd, world)
            rows = run_world(sd, world, groups, fcaster, l2_cfg)
            all_rows.extend(rows)
            print(
                f"  world {world}: {len(rows)} rows "
                f"({time.time() - t0:.0f}s, cache hits={fcaster.hits} miss={fcaster.misses})",
                flush=True,
            )
    summary = analyze(all_rows, {})
    out = {"meta": meta, "summary": summary, "rows": all_rows}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {args.out} ({len(all_rows)} rows, {time.time() - t0:.0f}s)")
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
