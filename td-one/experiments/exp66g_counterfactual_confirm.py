"""exp66g: multi-seed confirmation of the counterfactual risk tool.

exp66f (seed 42 only, n=10 groups) found the counterfactual
sensitivity tool was the only risk-identification tool that moved the
LLM risk classifier: Spearman +0.348 vs +0.120 for stats-only, and
the only monotonic config. One seed proves nothing. This experiment
repeats the exp66f classifier protocol across 5 seeds.

Arms (same deterministic group sets, same stats for every group):
  A  llm-stats                — LLM + aggregate stats only
  B  llm-stats+counterfactual — LLM + stats + counterfactual flips
  C  llm-stats+shap           — LLM + stats + SHAP drivers (control:
                                "any tool context", not the content)
  D  bins                     — deterministic training-tertile bins

Primary metric: mean absolute *bin error* — |anchor(level) -
realized test attrition|, anchors from training data only
(``evaluation.risk_calibration``), pooled over (seed, group) units
with paired bootstrap 95% CIs. Secondary: calibration monotonicity
(per-seed flags + pooled level means) and Spearman (continuity with
exp66f).

**Pre-registered confirmation rule:** the counterfactual claim is
confirmed only if BOTH (1) B beats A on pooled MAE with the paired
bootstrap 95% CI excluding zero, AND (2) B beats C the same way.
Otherwise the exp66f signal is recorded as not confirmed.

PIT: simulator outcomes split at the training cutoff; the GBM, SHAP
values, counterfactual cutoffs and bin anchors are fit on
training-period data only.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import _group_train_meta  # noqa: E402

from td_one.agent.deterministic_grouper import fit_deterministic_groups  # noqa: E402
from td_one.agent.llm_risk_classifier import classify_risk_llm  # noqa: E402
from td_one.agent.risk_reporter import compute_risk_bins  # noqa: E402
from td_one.agent.risk_tools import (  # noqa: E402
    counterfactual_flips,
    format_counterfactuals,
    format_shap_drivers,
    training_proba_cutoffs,
)
from td_one.evaluation.risk_calibration import (  # noqa: E402
    LEVEL_RANK,
    bin_errors,
    level_anchors,
    level_realized_means,
    mean_bin_error,
    monotonic_check,
    paired_bootstrap_mae_delta,
)
from td_one.evaluation.shap_analysis import (  # noqa: E402
    base_attribute,
    expected_log_odds,
    fit_attrition_model,
    group_shap_driver_summary,
    prepare_features,
    shap_values,
)

SEEDS = [42, 123, 7, 2024, 555]
N_CUSTOMERS = 4000
N_MONTHS = 36
TEST_MONTHS = 9
MIN_GROUP_N = 200
N_CLUSTERS = 5
GROUPER_SEED = 42
N_BOOTSTRAP = 2000

ARM_STATS = "llm-stats"
ARM_CF = "llm-stats+counterfactual"
ARM_SHAP = "llm-stats+shap"
ARM_BINS = "bins"
ARMS = [ARM_STATS, ARM_CF, ARM_SHAP, ARM_BINS]


def _simulate(seed: int):
    """Mirror exp66f's simulate() (same simulator settings)."""
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2

    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed, use_fred_macro=False
    )
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    hist_df = pd.DataFrame([r.model_dump() for r in history])
    hist_df["snapshot_date"] = pd.to_datetime(hist_df["snapshot_date"])
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
    cutoff = months[-TEST_MONTHS]
    return hist_df, out_df, cutoff


def _training_label(train_meta: pd.DataFrame, out_df: pd.DataFrame, cutoff: date) -> np.ndarray:
    train_out = out_df[out_df["month"] < cutoff]
    switched = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"].astype(str))
    return train_meta["mortgage_id"].astype(str).isin(switched).to_numpy(dtype=int)


def _spearman(levels: dict[str, str], realized: dict[str, float]) -> float:
    gids = [g for g in levels if g in realized]
    if len(gids) < 4:
        return float("nan")
    r, _ = spearmanr([LEVEL_RANK[levels[g]] for g in gids], [realized[g] for g in gids])
    return float(r)


def run_seed(seed: int, llm) -> dict:
    """Run one seed: fit groups + tools, classify under each arm."""
    print(f"\n=== seed {seed} ===", flush=True)
    hist_df, out_df, cutoff = _simulate(seed)
    train_meta = _group_train_meta(hist_df, cutoff)
    y = _training_label(train_meta, out_df, cutoff)
    test_out = out_df[out_df["month"] >= cutoff]
    train_out = out_df[out_df["month"] < cutoff]

    det = fit_deterministic_groups(
        train_meta, n_clusters=N_CLUSTERS, seed=GROUPER_SEED, min_group_n=MIN_GROUP_N
    )
    gids = [g.group_id for g in det.groups]
    assign = np.asarray(det.assignment.values).astype(str)
    print(f"  {len(gids)} groups", flush=True)

    mid_to_group = dict(zip(train_meta["mortgage_id"].astype(str), assign, strict=True))
    test_assign = {
        str(mid): mid_to_group[str(mid)]
        for mid in test_out["mortgage_id"].astype(str).unique()
        if str(mid) in mid_to_group
    }
    group_test = test_out.copy()
    group_test["group_id"] = group_test["mortgage_id"].astype(str).map(test_assign)
    test_realized = {
        g: float(v)
        for g, v in group_test.groupby("group_id")["switched_lender_flag"].mean().items()
    }
    test_n = group_test.groupby("group_id").size().to_dict()

    tg = train_out.copy()
    tg["group_id"] = tg["mortgage_id"].astype(str).map(mid_to_group)
    train_realized = {
        g: float(tg.groupby("group_id")["switched_lender_flag"].mean().get(g, 0.0))
        for g in gids
    }

    # ---- deterministic bins (arm D) + level anchors (training only) ----
    bins = compute_risk_bins(train_realized, list(train_realized.values()))
    anchors = level_anchors(train_realized, train_bins=bins)

    # ---- model + tools (training period only) ----
    x, feature_names, base_map = prepare_features(train_meta)
    model = fit_attrition_model(x, y, seed=seed)
    sv = shap_values(model, x)
    ev = expected_log_odds(model)
    proba = model.predict_proba(x)[:, 1]
    cutoffs = training_proba_cutoffs(proba)
    feature_std = {
        base_map[c]: float(x[c].std())
        for c in x.columns
        if base_map[c] == c and pd.api.types.is_numeric_dtype(x[c])
    }

    def tool_sections(g: str) -> tuple[dict[str, str], list[str]]:
        """SHAP + counterfactual prompt sections and raw CF findings."""
        mask = assign == g
        group_mean = ev + sv[mask].sum(axis=1).mean()
        summary = group_shap_driver_summary(
            sv, feature_names, mask, ev, group_mean, top_k=5, base_map=base_map
        )
        sections = {"shap": format_shap_drivers(summary)}
        mean_shap = sv[mask].mean(axis=0)
        top_attrs: list[str] = []
        for i in np.argsort(-np.abs(mean_shap))[:12]:
            a = base_attribute(feature_names[i], base_map)
            if a not in top_attrs:
                top_attrs.append(a)
            if len(top_attrs) == 5:
                break
        findings = counterfactual_flips(
            model, x.iloc[mask], top_attrs, base_map, feature_std, cutoffs
        )
        sections["counterfactual"] = format_counterfactuals(findings)
        return sections, findings

    def classify(g: str, rule: str, stats: dict, extra: list[str]) -> tuple[str, str]:
        try:
            out = classify_risk_llm(llm, g, rule, stats, [], extra_sections=extra or None)
            return out.level, out.justification
        except Exception as e:  # noqa: BLE001 — one bad group must not kill the run
            print(f"    LLM failed for {g}: {e}; defaulting MEDIUM", flush=True)
            return "MEDIUM", ""

    levels: dict[str, dict[str, str]] = {}
    justifications: dict[str, dict[str, str]] = {}
    cf_findings: dict[str, list[str]] = {}
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = assign
    for g in gids:
        gd = next(gg for gg in det.groups if gg.group_id == g)
        members = train_meta_g[train_meta_g["group_id"] == g]
        stats = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(train_realized.get(g, 0.0)), 4),
            "predicted_mean_attrition": round(float(proba[assign == g].mean()), 4),
            "n_test_episodes": int(test_n.get(g, 0)),
        }
        sec, findings = tool_sections(g)
        cf_findings[g] = findings
        cfg_levels: dict[str, str] = {}
        cfg_just: dict[str, str] = {}
        cfg_levels[ARM_BINS] = bins.get(g, "MEDIUM")
        cfg_just[ARM_BINS] = ""
        for arm, extra in [
            (ARM_STATS, []),
            (ARM_CF, [sec["counterfactual"]]),
            (ARM_SHAP, [sec["shap"]]),
        ]:
            lvl, just = classify(g, gd.rule, stats, extra)
            cfg_levels[arm] = lvl
            cfg_just[arm] = just
        levels[g] = cfg_levels
        justifications[g] = cfg_just
        print(
            f"  {g}: "
            + " ".join(f"{k}={v}" for k, v in cfg_levels.items())
            + f"  cf_flips={len(findings)}",
            flush=True,
        )

    metrics: dict[str, dict] = {}
    errors: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        lv = {g: levels[g][arm] for g in gids}
        errs = bin_errors(lv, anchors, test_realized)
        errors[arm] = {f"{seed}:{g}": e for g, e in errs.items()}
        metrics[arm] = {
            "mae": mean_bin_error(errs) if errs else float("nan"),
            "spearman": _spearman(lv, test_realized),
            "monotonic": monotonic_check(lv, test_realized),
            "n_groups": len(errs),
        }
    print(f"  metrics: {json.dumps(metrics, indent=2)}", flush=True)
    return {
        "seed": seed,
        "cutoff": str(cutoff),
        "anchors": anchors,
        "levels": levels,
        "justifications": justifications,
        "cf_findings": cf_findings,
        "metrics": metrics,
        "errors": errors,
        "test_realized": test_realized,
    }


def main() -> None:
    from td_one.llm_rungs.vector_client import VectorProxyClient

    t0 = time.time()
    out_dir = Path(__file__).resolve().parent
    checkpoint_path = out_dir / "exp66g_checkpoint.json"
    llm = VectorProxyClient(model="flash-lite", call_budget=400)

    # Resume-safe: completed seeds are checkpointed after each seed, so
    # a killed run (daemon restarts kill background processes) loses at
    # most the in-flight seed.
    seed_results: list[dict] = []
    done_seeds: set[int] = set()
    if checkpoint_path.exists():
        seed_results = json.loads(checkpoint_path.read_text())
        done_seeds = {sr["seed"] for sr in seed_results}
        print(f"resuming from checkpoint: seeds done {sorted(done_seeds)}", flush=True)
    for seed in SEEDS:
        if seed in done_seeds:
            continue
        seed_results.append(run_seed(seed, llm))
        checkpoint_path.write_text(json.dumps(seed_results, indent=2))
    seed_results.sort(key=lambda sr: SEEDS.index(sr["seed"]))

    # ---- pooled errors + MAE ----
    pooled_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for sr in seed_results:
        for arm in ARMS:
            pooled_errors[arm].update(sr["errors"][arm])
    pooled_mae = {arm: mean_bin_error(pooled_errors[arm]) for arm in ARMS}

    # ---- paired bootstrap deltas (delta > 0 means second arm better) ----
    comparisons = {}
    for a, b, name in [
        (ARM_STATS, ARM_CF, "cf_vs_stats"),
        (ARM_SHAP, ARM_CF, "cf_vs_shap"),
        (ARM_BINS, ARM_STATS, "stats_vs_bins"),
        (ARM_BINS, ARM_CF, "cf_vs_bins"),
    ]:
        delta, lo, hi = paired_bootstrap_mae_delta(
            pooled_errors[a], pooled_errors[b], n=N_BOOTSTRAP, seed=0
        )
        comparisons[name] = {"mae_delta": delta, "ci_lo": lo, "ci_hi": hi}
        print(f"  {name}: MAE delta={delta:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]", flush=True)

    # ---- pooled monotonicity + level means + Spearman (continuity) ----
    pooled_levels: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    pooled_realized: dict[str, float] = {}
    for sr in seed_results:
        for g, rl in sr["test_realized"].items():
            pooled_realized[f"{sr['seed']}:{g}"] = rl
        for g, arms in sr["levels"].items():
            for arm in ARMS:
                pooled_levels[arm][f"{sr['seed']}:{g}"] = arms[arm]
    pooled_level_means = {
        arm: level_realized_means(pooled_levels[arm], pooled_realized) for arm in ARMS
    }
    pooled_mono = {
        arm: monotonic_check(pooled_levels[arm], pooled_realized) for arm in ARMS
    }
    mono_fraction = {
        arm: float(np.mean([sr["metrics"][arm]["monotonic"] for sr in seed_results]))
        for arm in ARMS
    }
    pooled_spearman = {
        arm: _spearman(pooled_levels[arm], pooled_realized) for arm in ARMS
    }

    confirmed = (
        comparisons["cf_vs_stats"]["ci_lo"] > 0 and comparisons["cf_vs_shap"]["ci_lo"] > 0
    )
    print(f"\n  pooled MAE: {json.dumps(pooled_mae, indent=2)}", flush=True)
    verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED"
    print(f"  pre-registered confirmation: {verdict}", flush=True)

    summary = {
        "config": {
            "seeds": SEEDS,
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "test_months": TEST_MONTHS,
            "n_clusters": N_CLUSTERS,
            "grouper_seed": GROUPER_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "arms": ARMS,
        },
        "seeds": [
            {
                "seed": sr["seed"],
                "cutoff": sr["cutoff"],
                "anchors": sr["anchors"],
                "metrics": sr["metrics"],
                "levels": sr["levels"],
                "justifications": sr["justifications"],
                "cf_findings": sr["cf_findings"],
                "test_realized": sr["test_realized"],
            }
            for sr in seed_results
        ],
        "pooled": {
            "mae": pooled_mae,
            "comparisons": comparisons,
            "monotonic_fraction_seeds": mono_fraction,
            "monotonic_pooled": pooled_mono,
            "level_realized_means": pooled_level_means,
            "spearman": pooled_spearman,
            "confirmed": confirmed,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "exp66g_results.json").write_text(json.dumps(summary, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    print(f"\nwrote exp66g_results.json ({summary['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
