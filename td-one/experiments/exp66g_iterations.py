"""exp66g iterations: diagnosis-driven rounds for the counterfactual tool.

The pre-registered baseline (exp66g_counterfactual_confirm.py) came
back NOT CONFIRMED: pooled MAE B=0.0629 vs A=0.0611 (CI covers 0),
B vs C also n.s. Mechanism diagnosis (see exp66g_report.md):

1. B is *worse* where flips exist (-0.0066) and marginally better
   where the section is "no flips" boilerplate (+0.0012) — the
   content is not driving any benefit.
2. Only 24% of B justifications cite the flip content; B helps on
   that subset (+0.0102), hurts elsewhere (-0.0056).
3. B changed levels in 9/50 units (7 down, 2 up); B right 4, A right 5.
4. Changes are uncorrelated with flip presence.

Rounds (one hypothesis each, same 5 seeds x 10 groups):
  R1 prompt tuning — engagement requirement appended to the CF
     section ("justify how the counterfactual changes/confirms your
     call"). Tests diagnosis 2.
  R2 format — omit the CF section entirely when findings are empty.
     Tests diagnosis 1 (boilerplate noise).
  R3 tool param search — counterfactual edit size 0.5 sigma (R3a) and
     2 sigma (R3b) instead of 1 sigma. Tests whether flip granularity
     is the problem.
  R4 prompt tuning — interpretation note ("a flip means SENSITIVE,
     not lower risk; classify on current conditions"). Tests
     diagnosis 3 (downgrade skew).
  R5 inert-content control — CF section taken from a *different*
     group in the same seed. If R5 ~= B ~= A, the content is
     provably inert and the line can close with evidence.

Comparisons: per-unit bin errors vs the stored baseline arms
(anchors / realized / A and B levels loaded from
exp66g_results.json), paired bootstrap 95% CIs. Data prep is
asserted to reproduce the baseline's test_realized per seed before
any variant level is accepted.

Per-seed checkpointing: a killed run loses at most one seed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import _group_train_meta  # noqa: E402
from exp66g_counterfactual_confirm import _simulate, _training_label  # noqa: E402

from td_one.agent.deterministic_grouper import fit_deterministic_groups  # noqa: E402
from td_one.agent.llm_risk_classifier import classify_risk_llm  # noqa: E402
from td_one.agent.risk_tools import (  # noqa: E402
    counterfactual_flips,
    format_counterfactuals,
    training_proba_cutoffs,
)
from td_one.evaluation.risk_calibration import (  # noqa: E402
    bin_errors,
    mean_bin_error,
    paired_bootstrap_mae_delta,
)
from td_one.evaluation.shap_analysis import (  # noqa: E402
    base_attribute,
    fit_attrition_model,
    prepare_features,
    shap_values,
)

SEEDS = [42, 123, 7, 2024, 555]
MIN_GROUP_N = 200
N_CLUSTERS = 5
GROUPER_SEED = 42
N_BOOTSTRAP = 2000

ARM_A = "llm-stats"
ARM_B = "llm-stats+counterfactual"
VARIANTS = [
    "R1_prompt_cite",
    "R2_skip_empty",
    "R3a_std0.5",
    "R3b_std2.0",
    "R4_guidance",
    "R5_shuffled",
]

ENGAGE_NOTE = (
    "\nEngagement requirement: your JUSTIFICATION must explicitly state "
    "how the counterfactual sensitivity above changes or confirms your "
    "risk call; if it does not apply, say why in one clause."
)
INTERPRET_NOTE = (
    "\nHow to read the counterfactuals: a class flip means this group's "
    "risk is SENSITIVE to that driver — it does NOT mean the risk is "
    "lower. Unless the change has already happened, classify on current "
    "conditions."
)


def _prep_seed(seed: int):
    """Reproduce exp66g's per-seed data prep (deterministic)."""
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

    x, feature_names, base_map = prepare_features(train_meta)
    model = fit_attrition_model(x, y, seed=seed)
    sv = shap_values(model, x)
    proba = model.predict_proba(x)[:, 1]
    cutoffs = training_proba_cutoffs(proba)
    feature_std = {
        base_map[c]: float(x[c].std())
        for c in x.columns
        if base_map[c] == c and pd.api.types.is_numeric_dtype(x[c])
    }

    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = assign
    groups = {}
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
        mask = assign == g
        mean_shap = sv[mask].mean(axis=0)
        top_attrs: list[str] = []
        for i in np.argsort(-np.abs(mean_shap))[:12]:
            a = base_attribute(feature_names[i], base_map)
            if a not in top_attrs:
                top_attrs.append(a)
            if len(top_attrs) == 5:
                break

        def flips_at(scale: float, _mask=mask, _top_attrs=top_attrs) -> list[str]:
            scaled = {k: v * scale for k, v in feature_std.items()}
            return counterfactual_flips(
                model, x.iloc[_mask], _top_attrs, base_map, scaled, cutoffs
            )

        groups[g] = {
            "rule": gd.rule,
            "stats": stats,
            "findings": {1.0: flips_at(1.0), 0.5: flips_at(0.5), 2.0: flips_at(2.0)},
        }
    return gids, test_realized, groups


def _classify(llm, g: str, rule: str, stats: dict, extra: list[str]) -> tuple[str, str]:
    try:
        out = classify_risk_llm(llm, g, rule, stats, [], extra_sections=extra or None)
        return out.level, out.justification
    except Exception as e:  # noqa: BLE001
        print(f"    LLM failed for {g}: {e}; defaulting MEDIUM", flush=True)
        return "MEDIUM", ""


def run_seed_variants(seed: int, llm, baseline_seed: dict) -> dict:
    """Classify every group under all round variants for one seed."""
    print(f"\n=== seed {seed} (iterations) ===", flush=True)
    gids, test_realized, groups = _prep_seed(seed)
    base_real = baseline_seed["test_realized"]
    assert set(test_realized) == set(base_real) and all(
        abs(test_realized[g] - base_real[g]) < 1e-9 for g in base_real
    ), f"seed {seed}: data prep diverged from baseline (test_realized mismatch)"

    # R5: deranged findings assignment within the seed
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(len(gids)))
    for i in range(len(gids)):
        if perm[i] == i:
            j = (i + 1) % len(gids)
            perm[i], perm[j] = perm[j], perm[i]
    shuffled = {gids[i]: groups[gids[perm[i]]]["findings"][1.0] for i in range(len(gids))}

    levels: dict[str, dict[str, str]] = {v: {} for v in VARIANTS}
    justifications: dict[str, dict[str, str]] = {v: {} for v in VARIANTS}
    for g in gids:
        info = groups[g]
        f1 = info["findings"][1.0]
        sec1 = format_counterfactuals(f1)
        variant_sections = {
            "R1_prompt_cite": [sec1 + ENGAGE_NOTE],
            "R2_skip_empty": [sec1] if f1 else [],
            "R3a_std0.5": [format_counterfactuals(info["findings"][0.5])],
            "R3b_std2.0": [format_counterfactuals(info["findings"][2.0])],
            "R4_guidance": [sec1 + INTERPRET_NOTE],
            "R5_shuffled": [format_counterfactuals(shuffled[g])],
        }
        for v in VARIANTS:
            lvl, just = _classify(llm, g, info["rule"], info["stats"], variant_sections[v])
            levels[v][g] = lvl
            justifications[v][g] = just
        print(
            f"  {g}: " + " ".join(f"{v}={levels[v][g]}" for v in VARIANTS), flush=True
        )
    return {"seed": seed, "levels": levels, "justifications": justifications}


def main() -> None:
    from td_one.llm_rungs.vector_client import VectorProxyClient

    t0 = time.time()
    out_dir = Path(__file__).resolve().parent
    baseline = json.loads((out_dir / "exp66g_results.json").read_text())
    base_by_seed = {sr["seed"]: sr for sr in baseline["seeds"]}
    checkpoint_path = out_dir / "exp66g_iterations_checkpoint.json"

    llm = VectorProxyClient(model="flash-lite", call_budget=900)
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        done = json.loads(checkpoint_path.read_text())
        print(f"resuming iterations: seeds done {sorted(int(s) for s in done)}", flush=True)
    for seed in SEEDS:
        if str(seed) in done:
            continue
        done[str(seed)] = run_seed_variants(seed, llm, base_by_seed[seed])
        checkpoint_path.write_text(json.dumps(done, indent=2))

    # ---- score variants against stored baseline arms ----
    def unit_errors(levels_by_seed: dict[int, dict[str, str]]) -> dict[str, float]:
        errs: dict[str, float] = {}
        for sr in baseline["seeds"]:
            s = sr["seed"]
            e = bin_errors(levels_by_seed[s], sr["anchors"], sr["test_realized"])
            errs.update({f"{s}:{g}": v for g, v in e.items()})
        return errs

    base_levels = {
        arm: {
            sr["seed"]: {g: sr["levels"][g][arm] for g in sr["levels"]}
            for sr in baseline["seeds"]
        }
        for arm in (ARM_A, ARM_B)
    }
    errors_a = unit_errors(base_levels[ARM_A])
    errors_b = unit_errors(base_levels[ARM_B])

    results: dict[str, dict] = {}
    for v in VARIANTS:
        lv = {int(s): done[s]["levels"][v] for s in done}
        errs = unit_errors(lv)
        d_vs_a, lo_a, hi_a = paired_bootstrap_mae_delta(errors_a, errs, n=N_BOOTSTRAP, seed=0)
        d_vs_b, lo_b, hi_b = paired_bootstrap_mae_delta(errors_b, errs, n=N_BOOTSTRAP, seed=0)
        results[v] = {
            "mae": mean_bin_error(errs),
            "vs_stats": {"delta": d_vs_a, "ci_lo": lo_a, "ci_hi": hi_a},
            "vs_baseline_cf": {"delta": d_vs_b, "ci_lo": lo_b, "ci_hi": hi_b},
        }
        print(
            f"  {v}: MAE={results[v]['mae']:.4f} vs A {d_vs_a:+.4f} [{lo_a:+.4f},{hi_a:+.4f}]"
            f" | vs B {d_vs_b:+.4f} [{lo_b:+.4f},{hi_b:+.4f}]",
            flush=True,
        )

    summary = {
        "config": {"seeds": SEEDS, "variants": VARIANTS, "n_bootstrap": N_BOOTSTRAP},
        "baseline_mae": baseline["pooled"]["mae"],
        "variants": results,
        "levels": {s: done[s]["levels"] for s in done},
        "justifications": {s: done[s]["justifications"] for s in done},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "exp66g_iterations_results.json").write_text(json.dumps(summary, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    print(f"\nwrote exp66g_iterations_results.json ({summary['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
