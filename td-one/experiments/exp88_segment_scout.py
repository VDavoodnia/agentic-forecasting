"""exp88: the segment scout, scored.

An LLM scout proposes borrower segment definitions (GroupDefinition
rules) from aggregate statistics only, and we score the proposals
against SimulatorV2's planted cohorts (ARI + per-cohort F1) on seeds
7/42/123 — the exp66d/e recovery protocol (4,000 customers, 36 months,
last pre-cutoff snapshot per borrower).

Incumbents (exp66e, seed 42): SHAP-cluster KMeans ARI 0.091 (best),
raw-feature KMeans 0.046, LLM grouper 0.026, deterministic tree leaves
0.010. Incumbents are recomputed per seed here for a fair table.

Rounds (information-set ablation; the scout never sees labels):
  R1 stats only (feature quantiles/shares + joint feature counts)
  R2 stats + SHAP analyst context (exp66e2's tool context)
  R3 stats + policy brief + PIT news summaries + macro signals asof
  R4 self-consistency: 3 samples of the best R1-R3 info set, merged
  R5 diagnostics (chosen by pre-registered rule, see report):
     R5a capability — best info set on the larger model (flash)
     R5b ceiling — stats + per-cell TRAINING switching rates (outcome
         aggregates; a deliberate diagnostic, not a scout config)

Pre-registered bar: ARI > 0.091 with readable rules on >= 2 of 3 seeds.
No kills: a chance-level scout ships as a per-cohort invisibility
diagnosis, not a buried negative.

LLM calls are disk-cached under exp88_cache/ (keyed by model+prompt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import _group_train_meta, cohort_labels_for  # noqa: E402
from exp66e2_llm_tool_context import build_tool_context  # noqa: E402

from td_one.agent.segment_scout import (  # noqa: E402
    audit_prompt,
    build_feature_stats,
    build_scout_frame,
    build_scout_prompt,
    categorical_levels_of,
    eligible_rule_columns,
    merge_samples,
    partition_from_groups,
    proposals_to_json,
    propose_segments,
)
from td_one.data_gen.simulator_v2 import MortgageSimulatorV2  # noqa: E402
from td_one.evaluation.cohort_recovery import (  # noqa: E402
    adjusted_rand_index,
    best_match_per_cohort,
)
from td_one.evaluation.shap_analysis import (  # noqa: E402
    cluster_on_shap,
    fit_attrition_model,
    prepare_features,
    shap_values,
)

N_CUSTOMERS = 4000
N_MONTHS = 36
TEST_MONTHS = 9
MIN_GROUP_N = 200
SEEDS = (7, 42, 123)
CACHE_DIR = Path(__file__).resolve().parent / "exp88_cache"
NEWS_DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"


def simulate_seed(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, date]:
    """exp66d.simulate() with a seed parameter (identical protocol)."""
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


def training_label(train_meta: pd.DataFrame, out_df: pd.DataFrame, cutoff: date) -> np.ndarray:
    """Switched during the training window (exp66e's label)."""
    train_out = out_df[out_df["month"] < cutoff]
    switched = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"].astype(str))
    return train_meta["mortgage_id"].astype(str).isin(switched).to_numpy(dtype=int)


class CachedClient:
    """Disk-cached wrapper around VectorProxyClient (bounded, resume-safe)."""

    def __init__(self, model: str, call_budget: int = 200) -> None:
        from td_one.llm_rungs.vector_client import VectorProxyClient

        self._inner = VectorProxyClient(model=model, call_budget=call_budget, max_tokens=4096)
        self.model = model
        self.real_calls = 0
        self.cache_hits = 0
        CACHE_DIR.mkdir(exist_ok=True)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        salt: str = "",
    ) -> str:
        key = hashlib.sha256(f"{self.model}|{salt}|{system or ''}|{prompt}".encode()).hexdigest()[
            :24
        ]
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return str(json.loads(path.read_text())["text"])
        text = self._inner.complete(prompt, system=system, json_mode=json_mode)
        self.real_calls += 1
        path.write_text(json.dumps({"model": self.model, "salt": salt, "text": text}))
        return text


class SaltedClient:
    """Fixed-salt view of a CachedClient (self-consistency samples)."""

    def __init__(self, base: CachedClient, salt: str) -> None:
        self._base = base
        self._salt = salt

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        return self._base.complete(prompt, system=system, json_mode=json_mode, salt=self._salt)


def score_partition(scout_df: pd.DataFrame, groups: list, true: list[str]) -> dict:
    """ARI + per-cohort F1 for a proposal list (first-match partition)."""
    part = partition_from_groups(scout_df, groups)
    pred = [part[m] for m in scout_df["mortgage_id"].astype(str)]
    return {
        "ari": adjusted_rand_index(true, pred),
        "per_cohort": best_match_per_cohort(true, pred),
        "n_groups": len(groups),
        "catch_all_share": float(np.mean([p == "__catch_all__" for p in pred])),
    }


def incumbents(train_meta: pd.DataFrame, y: np.ndarray, true: list[str], seed: int) -> dict:
    """Raw KMeans + SHAP-cluster ARI on this seed (exp66e protocol)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    x, feature_names, base_map = prepare_features(train_meta)
    x_std = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    raw = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(x_std)
    model = fit_attrition_model(x, y, seed=seed)
    sv = shap_values(model, x)
    shap_labels = cluster_on_shap(sv, n_clusters=5, seed=seed)
    return {
        "raw_kmeans_ari": adjusted_rand_index(true, [str(v) for v in raw]),
        "shap_cluster_ari": adjusted_rand_index(true, [str(v) for v in shap_labels]),
        "shap_per_cohort": best_match_per_cohort(true, [str(v) for v in shap_labels]),
        "_model": model,
        "_sv": sv,
        "_x": x,
        "_feature_names": feature_names,
        "_base_map": base_map,
    }


def news_context(cutoff: date) -> tuple[list[str], list[str]]:
    """PIT-safe news summaries + macro signal lines as of the origin."""
    summaries: list[str] = []
    try:
        from td_one.rag.semantic import SemanticNewsIndex

        index = SemanticNewsIndex()
        queries = [
            "Canada mortgage renewal interest rates households",
            "OSFI mortgage stress test uninsured borrowers",
            "Canada families cost of living childcare expenses",
            "Bank of Canada rate hikes fixed mortgage borrowers",
        ]
        seen: set[str] = set()
        for q in queries:
            for a in index.search(q, before=cutoff, top_k=3):
                if a.title not in seen:
                    seen.add(a.title)
                    summaries.append(f"[{a.pub_date}] {a.title}")
    except Exception as e:  # noqa: BLE001 — news is context, not a gate
        print(f"  news retrieval failed ({e}); continuing without")
    macro_lines: list[str] = []
    try:
        from td_one.news.macro_signal import signals_asof

        con = sqlite3.connect(str(NEWS_DB))
        sigs = signals_asof(con, cutoff.strftime("%Y-%m"))
        for cat, s in sorted(sigs.items()):
            macro_lines.append(
                f"{cat} ({s.month}): direction {s.direction:+d}, risk {s.risk}, "
                f"certainty {s.certainty} — {s.summary}"
            )
        con.close()
    except Exception as e:  # noqa: BLE001
        print(f"  macro signals failed ({e}); continuing without")
    return summaries, macro_lines


def cell_rate_block(scout_df: pd.DataFrame, y: np.ndarray) -> str:
    """R5b diagnostic: the scout's joint tables + training switching rates.

    Deliberately outcome-bearing — a ceiling diagnostic, NOT a scout
    configuration. Cohort names never appear.
    """
    d = scout_df.copy()
    d["_y"] = y
    lines = ["Joint shares with TRAINING switching rate per cell (diagnostic):"]
    specs: list[tuple[str, pd.Series]] = []
    d["rate_band"] = pd.cut(
        d["mortgage_rate_pct"],
        bins=[0, 2.0, 3.0, 4.0, 5.0, 99],
        labels=["<=2", "2-3", "3-4", "4-5", ">5"],
    )
    specs.append(
        (
            "rate_type x origination_year x rate band",
            d.groupby(["rate_type", "origination_year", "rate_band"], observed=True)["_y"],
        )
    )
    specs.append(
        (
            "insured x credit band",
            d.groupby(["insured_flag", "credit_score_band"], observed=True)["_y"],
        )
    )
    d["income_band"] = pd.cut(
        d["household_income_cad"],
        bins=[0, 50000, 75000, 100000, 150000, 10**9],
        labels=["<50k", "50-75k", "75-100k", "100-150k", ">=150k"],
    )
    d["children_band"] = pd.cut(
        d["n_children"], bins=[-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"]
    )
    specs.append(
        ("income band x children", d.groupby(["income_band", "children_band"], observed=True)["_y"])
    )
    for label, grp in specs:
        agg = grp.agg(["mean", "count"])
        parts = [
            f"{' & '.join(str(v) for v in idx)}: rate={r['mean']:.3f} (n={int(r['count'])})"
            for idx, r in agg.iterrows()
            if r["count"] >= 30
        ]
        lines.append(f"- {label}: " + "; ".join(parts))
    return "\n".join(lines)


def run_seed(seed: int, lite: CachedClient, rounds: set[str]) -> dict:
    """All per-seed work except R4/R5 (which need cross-seed results)."""
    print(f"\n=== seed {seed}: simulating ===", flush=True)
    hist_df, out_df, cutoff = simulate_seed(seed)
    train_meta = _group_train_meta(hist_df, cutoff)
    labels = cohort_labels_for(hist_df)
    true = [labels[m] for m in train_meta["mortgage_id"].astype(str)]
    y = training_label(train_meta, out_df, cutoff)
    print(f"  cutoff {cutoff}; borrowers {len(train_meta)}; train switch rate {y.mean():.3f}")

    print("  incumbents (KMeans + SHAP)...", flush=True)
    inc = incumbents(train_meta, y, true, seed)
    print(f"  raw KMeans ARI {inc['raw_kmeans_ari']:.4f}; SHAP ARI {inc['shap_cluster_ari']:.4f}")

    scout_df = build_scout_frame(train_meta)
    stats = build_feature_stats(scout_df)
    cols = eligible_rule_columns(scout_df)
    levels = categorical_levels_of(scout_df)

    shap_ctx = build_tool_context(
        train_meta,
        inc["_x"],
        inc["_feature_names"],
        inc["_base_map"],
        inc["_model"],
        inc["_sv"],
        y,
    )
    news_summaries, macro_lines = ([], [])
    if "r3" in rounds or "r5b" in rounds:
        news_summaries, macro_lines = news_context(cutoff)
        print(f"  news: {len(news_summaries)} summaries; macro: {len(macro_lines)} lines")

    prompts: dict[str, tuple[str, str]] = {}
    prompts["r1"] = build_scout_prompt(stats, cols, categorical_levels=levels)
    prompts["r2"] = build_scout_prompt(
        stats, cols, categorical_levels=levels, shap_context=shap_ctx
    )
    prompts["r3"] = build_scout_prompt(
        stats,
        cols,
        categorical_levels=levels,
        policy_brief=True,
        news_summaries=news_summaries,
        macro_lines=macro_lines,
    )
    if "r5b" in rounds:
        prompts["r5b"] = build_scout_prompt(
            stats + "\n\n" + cell_rate_block(scout_df, y),
            cols,
            categorical_levels=levels,
        )

    out: dict = {
        "seed": seed,
        "cutoff": str(cutoff),
        "cohort_counts": pd.Series(true).value_counts().to_dict(),
        "incumbents": {
            "raw_kmeans_ari": inc["raw_kmeans_ari"],
            "shap_cluster_ari": inc["shap_cluster_ari"],
            "shap_per_cohort": inc["shap_per_cohort"],
        },
        "rounds": {},
        "prompts": {k: {"system": s, "user": u} for k, (s, u) in prompts.items()},
        "_ctx": {"prompts": prompts, "scout_df": scout_df, "true": true},
    }
    for rnd in ("r1", "r2", "r3"):
        if rnd not in rounds:
            continue
        system, user = prompts[rnd]
        violations = audit_prompt(system, user)
        assert not violations, f"{rnd} prompt leakage: {violations}"
        print(f"  {rnd}: proposing (audit clean)...", flush=True)
        res = propose_segments(lite, system, user, scout_df, min_group_n=MIN_GROUP_N)
        scored = (
            score_partition(scout_df, res.groups, true)
            if res.groups
            else {"ari": None, "per_cohort": {}, "n_groups": 0, "catch_all_share": 1.0}
        )
        print(
            f"  {rnd}: ARI={scored['ari']} groups={scored['n_groups']} "
            f"rejections={len(res.rejections)} parse_failures={res.parse_failures}"
        )
        out["rounds"][rnd] = {
            **scored,
            "groups": proposals_to_json(res.groups),
            "rejections": res.rejections,
            "parse_failures": res.parse_failures,
            "attempts": res.attempts,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--rounds", default="r1,r2,r3,r4,r5a,r5b")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rounds = set(args.rounds.split(","))
    t0 = time.time()
    out_dir = Path(__file__).resolve().parent

    lite = CachedClient("flash-lite", call_budget=200)
    per_seed = [run_seed(seed, lite, rounds) for seed in seeds]

    # ---- R4: self-consistency on the best R1-R3 info set ----
    if "r4" in rounds:
        means = {}
        for rnd in ("r1", "r2", "r3"):
            aris = [
                s["rounds"][rnd]["ari"]
                for s in per_seed
                if rnd in s["rounds"] and s["rounds"][rnd]["ari"] is not None
            ]
            if aris:
                means[rnd] = float(np.mean(aris))
        best_rnd = max(means, key=lambda r: (means[r], r)) if means else "r1"
        print(f"\nR4: best info set = {best_rnd} (means {means})", flush=True)
        for s in per_seed:
            ctx = s["_ctx"]
            system, user = ctx["prompts"][best_rnd]
            samples = []
            sample_scores = []
            for i in range(3):
                res = propose_segments(
                    SaltedClient(lite, f"r4-{s['seed']}-{i}"),
                    system,
                    user,
                    ctx["scout_df"],
                    min_group_n=MIN_GROUP_N,
                )
                samples.append(res.groups)
                if res.groups:
                    sample_scores.append(
                        score_partition(ctx["scout_df"], res.groups, ctx["true"])["ari"]
                    )
            merged = merge_samples(samples, min_votes=2)
            scored = score_partition(ctx["scout_df"], merged, ctx["true"])
            print(f"  seed {s['seed']}: samples ARI={sample_scores} merged ARI={scored['ari']:.4f}")
            s["rounds"]["r4"] = {
                **scored,
                "info_set": best_rnd,
                "sample_aris": sample_scores,
                "groups": proposals_to_json(merged),
            }

    # ---- R5a: capability — best info set on the larger model ----
    if "r5a" in rounds:
        means = {}
        for rnd in ("r1", "r2", "r3", "r4"):
            aris = [
                s["rounds"][rnd]["ari"]
                for s in per_seed
                if rnd in s["rounds"] and s["rounds"].get("ari") is not None
            ]
            if aris:
                means[rnd] = float(np.mean(aris))
        info_rnd = max((r for r in means if r != "r4"), key=lambda r: (means[r], r), default="r1")
        print(f"\nR5a: flash model on {info_rnd} prompts", flush=True)
        flash = CachedClient("flash", call_budget=40)
        for s in per_seed:
            ctx = s["_ctx"]
            system, user = ctx["prompts"][info_rnd]
            res = propose_segments(flash, system, user, ctx["scout_df"], min_group_n=MIN_GROUP_N)
            scored = (
                score_partition(ctx["scout_df"], res.groups, ctx["true"])
                if res.groups
                else {"ari": None, "per_cohort": {}, "n_groups": 0, "catch_all_share": 1.0}
            )
            print(f"  seed {s['seed']}: flash ARI={scored['ari']}")
            s["rounds"]["r5a"] = {
                **scored,
                "info_set": info_rnd,
                "model": "flash",
                "groups": proposals_to_json(res.groups),
                "rejections": res.rejections,
                "parse_failures": res.parse_failures,
            }

    # ---- R5b: information ceiling (outcome aggregates, diagnostic) ----
    if "r5b" in rounds:
        print("\nR5b: ceiling diagnostic (cell-level training rates)", flush=True)
        for s in per_seed:
            ctx = s["_ctx"]
            if "r5b" not in ctx["prompts"]:
                continue
            system, user = ctx["prompts"]["r5b"]
            # cohort names must still never appear, even in the diagnostic
            assert not [
                t
                for t in ("low_rate", "osfi_exempt", "family_squeeze")
                if t in (system + user).lower()
            ]
            res = propose_segments(lite, system, user, ctx["scout_df"], min_group_n=MIN_GROUP_N)
            scored = (
                score_partition(ctx["scout_df"], res.groups, ctx["true"])
                if res.groups
                else {"ari": None, "per_cohort": {}, "n_groups": 0, "catch_all_share": 1.0}
            )
            print(f"  seed {s['seed']}: ceiling ARI={scored['ari']}")
            s["rounds"]["r5b"] = {
                **scored,
                "diagnostic": "outcome aggregates (not a scout config)",
                "groups": proposals_to_json(res.groups),
                "rejections": res.rejections,
                "parse_failures": res.parse_failures,
            }

    prompts_out = {}
    for s in per_seed:
        prompts_out[str(s["seed"])] = s.pop("prompts")
        s.pop("_ctx")
    results = {
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "test_months": TEST_MONTHS,
            "min_group_n": MIN_GROUP_N,
            "seeds": seeds,
            "bar": "ARI > 0.091 on >= 2 of 3 seeds",
        },
        "per_seed": per_seed,
        "llm": {"real_calls": lite.real_calls, "cache_hits": lite.cache_hits},
        "elapsed_s": round(time.time() - t0, 1),
    }
    # summary table
    all_rounds = ["r1", "r2", "r3", "r4", "r5a", "r5b"]
    summary = {}
    for rnd in all_rounds:
        aris = [
            s["rounds"][rnd]["ari"]
            for s in per_seed
            if rnd in s["rounds"] and s["rounds"][rnd]["ari"] is not None
        ]
        if aris:
            summary[rnd] = {"aris": aris, "mean": float(np.mean(aris))}
    results["summary"] = summary
    print("\n=== exp88 summary (ARI) ===")
    for rnd, v in summary.items():
        print(f"  {rnd}: {[f'{a:.4f}' for a in v['aris']]} mean={v['mean']:.4f}")
    (out_dir / "exp88_results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "exp88_prompts.json").write_text(json.dumps(prompts_out, indent=2))
    print(f"wrote exp88_results.json + exp88_prompts.json ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
