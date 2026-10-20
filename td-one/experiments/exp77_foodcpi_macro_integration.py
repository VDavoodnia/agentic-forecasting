"""exp77: calibrated Macro Signal Service -> Food CPI.

Question: does a *calibrated* news signal add forecast value where
*uncalibrated* raw news never did (exp44/46/69 all failed)?

Protocol:
  1. Reproduction check: v2 (no news) on all 59 Food CPI origins must
     reproduce exp64's 0.5322 bit-for-bit (same code path).
  2. Arms on the 33 signal-window origins (origin idx >= 166, asof month
     >= 2023-10), blocked 5-fold time CV, paired t-test + win rate:
       A : v2, no news (subset baseline)
       B : v2 + live-signal feature (7 categories, base encoding)
       C : v2 + signal feature over ALL 22 categories (gating control)
       D : v1 + live-signal feature (classical router)
       D0: v1, no news (baseline for D)
  3. Diagnosis-driven iteration rounds (4-5+, standing rule):
       R2: B' = live minus food_prices (selection-on-target diagnostic;
           food_prices was calibrated on Food CPI itself, exp70)
       R3: per-category 7-dim signal-as-features (router learns; the
           exp71-MS R5/R6 mechanism alternative)
       R4: news_weight search + lead-1 timing on the best design
       R5: holdout-window origins only (target >= 2025-07, n=13) on the
           best design — the calibration-train overlap check
       R6: drop-one-category ablation on the best design

Verdict (same bar as exp71-MS): beats A at p < 0.05 AND >= 1% relative
effect. All news access goes through signals_asof (PIT: asof = dates[o-1],
the last observed month).

Usage:
    cd td-one && .venv/bin/python experiments/exp77_foodcpi_macro_integration.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp71_macro_signal_integration as e71ms  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method import ProposedForecaster  # noqa: E402
from td_one.agent.proposed_method_v2 import (  # noqa: E402
    LLM_EXPERT,
    ProposedForecasterV2,
    v2_for_dataset,
)
from td_one.llm_rungs.cache import load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import (  # noqa: E402
    blocked_cv_predict,
    build_records_with_llm,
)
from td_one.news.macro_signal import signals_asof  # noqa: E402

DB = ROOT / "data" / "news" / "news_2026.db"
EXP70_JSON = ROOT / "experiments" / "exp70_macro_signal_results.json"
CACHE = ROOT / "experiments" / "exp57_promptV1_cache.json"
EXP64_V2 = 0.5322  # canonical v2 CRPS on Food CPI (exp64, all 59 origins)


def load_data():
    """Food CPI values, dates, origins, features, classical + LLM records."""
    values = lfc.load_food_cpi_values()
    dates = pd.date_range("2010-01-01", periods=len(values), freq="MS")
    origins = lfc.origins_59(values)
    feature_names = lfc.feature_names_for(values)
    classical = lfc.build_classical_records(values, origins, feature_names)
    cache = load_grid_cache(CACHE)
    records = build_records_with_llm(
        classical, cache, feature_names, expert_name=LLM_EXPERT
    )
    first_sig = next(o for o in origins if dates[o - 1].strftime("%Y-%m") >= "2023-10")
    subset = [r for r in records if r["origin"] >= first_sig]
    classical_subset = lfc.aligned_classical(classical, subset)
    return {
        "values": values,
        "dates": dates,
        "feature_names": feature_names,
        "records_all": records,
        "records": subset,
        "classical_subset": classical_subset,
        "first_sig_origin": first_sig,
    }


def v2_factory(news: bool = False, weight: float = 0.3) -> ProposedForecasterV2:
    fc = v2_for_dataset("food_cpi", h=1)
    fc.models = [*lfc.CLASSICAL_MODELS, LLM_EXPERT]
    fc.use_news = news
    fc.news_weight = weight
    return fc


def v1_factory(news: bool = False, weight: float = 0.3) -> ProposedForecaster:
    fc = ProposedForecaster(h=1, dataset="food_cpi")
    fc.models = list(lfc.CLASSICAL_MODELS)
    fc.use_news = news
    fc.news_weight = weight
    return fc


def _paired(b: list[float], a: list[float]) -> dict:
    from scipy.stats import ttest_rel

    ba, aa = np.asarray(b), np.asarray(a)
    t, p = ttest_rel(ba, aa)
    rel = float(np.mean(ba - aa) / np.mean(aa)) if np.mean(aa) else 0.0
    if p < 0.05 and rel <= -0.01:
        verdict = "SIGNAL HELPS"
    elif p < 0.05 and rel >= 0.01:
        verdict = "SIGNAL HURTS"
    else:
        verdict = "NEGLIGIBLE"
    return {
        "mean_B": float(np.mean(ba)),
        "mean_A": float(np.mean(aa)),
        "mean_diff": float(np.mean(ba - aa)),
        "rel": rel,
        "p": float(p),
        "win_rate": float(np.mean(ba < aa)),
        "n": len(ba),
        "verdict": verdict,
    }


def run(
    design: str,
    records: list[dict],
    feature_names: list[str],
    method_factory,
    baseline_factory,
    news_fn=None,
) -> dict:
    res = blocked_cv_predict(
        records,
        feature_names,
        method_factory,
        n_folds=5,
        baseline=baseline_factory,
        fold_mode="time",
        news_features_fn=news_fn,
    )
    stats = _paired(res["crps"], res["baseline_crps"])
    print(
        f"[R] {design:58s} B={stats['mean_B']:.4f} A={stats['mean_A']:.4f} "
        f"diff={stats['mean_diff']:+.5f} ({stats['rel']:+.2%}) "
        f"p={stats['p']:.4f} win={stats['win_rate']:.2f} -> {stats['verdict']}",
        flush=True,
    )
    return {"design": design, **stats}


def run_two_cv(
    design: str,
    records_b: list[dict],
    feats_b: list[str],
    records_a: list[dict],
    feats_a: list[str],
    factory,
) -> dict:
    """Paired comparison across two record sets (signal-as-features arm)."""
    res_b = blocked_cv_predict(
        records_b, feats_b, factory, n_folds=5, baseline="equal", fold_mode="time"
    )
    res_a = blocked_cv_predict(
        records_a, feats_a, factory, n_folds=5, baseline="equal", fold_mode="time"
    )
    stats = _paired(res_b["crps"], res_a["crps"])
    print(
        f"[R] {design:58s} B={stats['mean_B']:.4f} A={stats['mean_A']:.4f} "
        f"diff={stats['mean_diff']:+.5f} ({stats['rel']:+.2%}) "
        f"p={stats['p']:.4f} win={stats['win_rate']:.2f} -> {stats['verdict']}",
        flush=True,
    )
    return {"design": design, **stats}


def build_signal_features(
    records: list[dict],
    feature_names: list[str],
    cats: list[str],
    dates: pd.DatetimeIndex,
) -> tuple[list[dict], list[str]]:
    """Per-category direction features (PIT-safe), 7-dim for the live set."""
    import sqlite3

    con = sqlite3.connect(DB)
    extra_names = [f"sig_{c}" for c in cats]
    out = []
    for r in records:
        o = int(r["origin"])
        asof_ym = dates[o - 1].strftime("%Y-%m")
        sigs = signals_asof(con, asof_ym)
        extra = [float(sigs[c].direction / 2.0) if c in sigs else 0.0 for c in cats]
        out.append({**r, "features": [*r["features"], *extra]})
    con.close()
    return out, [*feature_names, *extra_names]


def diagnose(
    records: list[dict],
    feature_names: list[str],
    cats: list[str],
    dates: pd.DatetimeIndex,
) -> dict:
    """Mechanism diagnostics: news_used rate, stress distribution, expert flips.

    Fit v2 (no news) once on all origins, predict each origin with/without
    the live-signal vector, tabulate what the signal actually did.
    """
    import sqlite3

    fc = v2_factory(news=True)
    fc.fit(records, feature_names)
    con = sqlite3.connect(DB)
    n_used = n_flip = 0
    stresses: list[float] = []
    news_experts: list[str] = []
    for r in records:
        o = int(r["origin"])
        asof_ym = dates[o - 1].strftime("%Y-%m")
        sigs = [s for c, s in signals_asof(con, asof_ym).items() if c in cats]
        dirs = np.array([s.direction for s in sigs], float)
        nf = {
            "coverage": float(len(sigs) / len(cats)),
            "stress_signal": float(np.mean(dirs) / 2.0),
        }
        stresses.append(float(nf["stress_signal"]))
        qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in fc.models}
        feats = dict(zip(feature_names, r["features"], strict=True))
        out_n = fc.predict(np.zeros(30), qm, feats, feature_names, news_features=nf)
        out_0 = fc.predict(np.zeros(30), qm, feats, feature_names)
        n_used += int(out_n.news_used)
        n_flip += int(out_n.active_expert != out_0.active_expert)
        news_experts.append(out_n.active_expert)
    con.close()
    stresses_a = np.abs(np.asarray(stresses))
    from collections import Counter

    print(
        f"[D] news_used {n_used}/{len(records)}, expert flips {n_flip}/{len(records)}, "
        f"mean|stress|={float(np.mean(stresses_a)):.3f}, "
        f"experts: {dict(Counter(news_experts))}",
        flush=True,
    )
    return {
        "news_used_frac": n_used / len(records),
        "expert_flip_frac": n_flip / len(records),
        "mean_abs_stress": float(np.mean(stresses_a)),
        "experts_with_news": dict(Counter(news_experts)),
    }


def main() -> None:
    live = e71ms.load_live_categories()
    assert len(live) == 7, f"expected 7 live categories, got {live}"
    all_cats = sorted(
        json.loads(EXP70_JSON.read_text())["categories"].keys()
    )
    print(f"live ({len(live)}): {live}\nall categories: {len(all_cats)}", flush=True)

    data = load_data()
    dates, fns = data["dates"], data["feature_names"]
    records, classical_subset = data["records"], data["classical_subset"]
    print(
        f"Food CPI: {len(data['values'])} months; subset origins "
        f"{len(records)} (idx {data['first_sig_origin']}..198)",
        flush=True,
    )

    # --- Reproduction check: v2 no-news on ALL 59 origins -------------------
    rep = blocked_cv_predict(
        data["records_all"],
        fns,
        lambda: v2_factory(news=False),
        n_folds=5,
        baseline="best_single",
        fold_mode="time",
    )
    rep_crps = float(np.mean(rep["crps"]))
    exp64_raw = float(
        json.loads((ROOT / "experiments" / "exp64_results.json").read_text())["food_cpi"][
            "v2_crps"
        ]
    )
    print(f"[repro] v2 all-origins CRPS={rep_crps!r} (exp64 raw: {exp64_raw!r})", flush=True)
    reproduced = rep_crps == exp64_raw
    if not reproduced:
        print("[repro] WARNING: baseline not reproduced — investigate before trusting arms")

    def news_fn(cats):
        return e71ms.news_features_fn(live, DB, dates, encoding="base", lag=0, cats=cats)

    rounds: list[dict] = []

    # R1: primary arms
    rounds.append(
        {
            **run(
                "R1 arm B: v2 + live signals",
                records,
                fns,
                lambda: v2_factory(news=True),
                lambda: v2_factory(news=False),
                news_fn(live),
            ),
            "round": 1,
        }
    )
    rounds.append(
        {
            **run(
                "R1 arm C: v2 + ALL 22 categories",
                records,
                fns,
                lambda: v2_factory(news=True),
                lambda: v2_factory(news=False),
                news_fn(all_cats),
            ),
            "round": 1,
        }
    )
    rounds.append(
        {
            **run(
                "R1 arm D: v1 + live signals",
                classical_subset,
                fns,
                lambda: v1_factory(news=True),
                lambda: v1_factory(news=False),
                news_fn(live),
            ),
            "round": 1,
        }
    )

    # R2: selection-on-target diagnostic (live minus food_prices)
    live_no_fp = [c for c in live if c != "food_prices"]
    rounds.append(
        {
            **run(
                "R2 arm B': v2 + live ex-food_prices",
                records,
                fns,
                lambda: v2_factory(news=True),
                lambda: v2_factory(news=False),
                news_fn(live_no_fp),
            ),
            "round": 2,
            "taught": "does the effect survive removing the target-calibrated category?",
        }
    )

    # R3: per-category signal-as-features (router learns, not hand modulation)
    recs_b, feats_b = build_signal_features(records, fns, live, dates)
    rounds.append(
        {
            **run_two_cv(
                "R3 arm F: v2 + per-category signal features (router learns)",
                recs_b,
                feats_b,
                records,
                fns,
                lambda: v2_factory(news=False),
            ),
            "round": 3,
            "taught": "learned use of the signal via the router vs fixed modulation",
        }
    )

    # mechanism diagnostics on the live design
    diag = diagnose(records, fns, live, dates)

    # R4: weight search + lead timing on the best arm so far
    best = min(rounds, key=lambda r: r["mean_diff"])
    if best["design"] in ("R1 arm B: v2 + live signals", "R2 arm B': v2 + live ex-food_prices"):
        cats = live if "ex-food_prices" not in best["design"] else live_no_fp
        for w in (0.1, 0.5, 0.7):
            rounds.append(
                {
                    **run(
                        f"R4 weight: news_weight={w}",
                        records,
                        fns,
                        lambda w=w: v2_factory(news=True, weight=w),
                        lambda: v2_factory(news=False),
                        news_fn(cats),
                    ),
                    "round": 4,
                    "taught": "modulation strength parameter search",
                }
            )
        rounds.append(
            {
                **run(
                    "R4 timing: lead-1 (asof-1 month)",
                    records,
                    fns,
                    lambda: v2_factory(news=True),
                    lambda: v2_factory(news=False),
                    e71ms.news_features_fn(live, DB, dates, encoding="base", lag=1, cats=cats),
                ),
                "round": 4,
                "taught": "whether signals lag the release calendar",
            }
        )

    # R5: holdout-window origins only (target >= 2025-07)
    hold = [r for r in records if dates[int(r["origin"])].strftime("%Y-%m") >= "2025-07"]
    rounds.append(
        {
            **run(
                f"R5 holdout-window origins only (n={len(hold)})",
                hold,
                fns,
                lambda: v2_factory(news=True),
                lambda: v2_factory(news=False),
                news_fn(live),
            ),
            "round": 5,
            "taught": "removes the calibration-train overlap (2023-10..2025-06)",
        }
    )

    # R6: drop-one-category ablation on the live design
    drop = [
        run(
            f"R6 drop-{c}",
            records,
            fns,
            lambda: v2_factory(news=True),
            lambda: v2_factory(news=False),
            news_fn([x for x in live if x != c]),
        )
        for c in live
    ]
    best_d = min(drop, key=lambda r: r["mean_diff"])
    rounds.append({**best_d, "round": 6, "taught": "which category drives the result"})

    helped = any(r["verdict"] == "SIGNAL HELPS" for r in rounds)
    print(
        f"\nfinal: {'SIGNAL HELPS' if helped else 'honest negative'} "
        f"after {len(rounds)} runs",
        flush=True,
    )
    out = ROOT / "experiments" / "exp77_results.json"
    out.write_text(
        json.dumps(
            {
                "date": datetime.now(UTC).isoformat(),
                "reproduction": {"v2_all_origins": rep_crps, "exp64": EXP64_V2,
                                 "exp64_raw_match": reproduced, "reproduced": reproduced},
                "live": live,
                "n_origins": len(records),
                "diagnostics": diag,
                "rounds": rounds,
                "final": "HELPS" if helped else "NEGATIVE",
            },
            indent=2,
        )
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
