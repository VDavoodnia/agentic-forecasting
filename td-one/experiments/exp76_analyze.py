"""exp76 analysis: arm-vs-arm paired tests the battery summary cannot do.

``summarize`` only pairs each arm against native. exp76 needs:
  - fused arms vs plain CQR (the pre-registered success criterion),
  - meta_story vs meta_numeric (S7 story ablation, corrected PIT keying),
  - meta_full vs meta_story (S8 macro add-on, corrected PIT keying),
  - aci_gated vs aci (regime gating check).
Reads experiments/exp76_fusion_results.json, prints markdown tables and
writes experiments/exp76_fusion_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "exp76_fusion_results.json"
OUT = HERE / "exp76_fusion_analysis.json"

SCOPES = {
    "pooled_synth": lambda r: r["config_id"].startswith("synth:"),
    "pooled_synth_S1_4sigma": lambda r: r["config_id"].startswith("synth:S1:4.0"),
    "pooled_synth_control": lambda r: r["config_id"].startswith("synth:none"),
    "pooled_food": lambda r: r["config_id"] == "food_cpi",
    "food_story_subset": lambda r: r["config_id"] == "food_cpi" and 133 <= r["origin"] <= 192,
    "food_macro_subset": lambda r: r["config_id"] == "food_cpi" and 166 <= r["origin"] <= 197,
    # Clean S8 scope: both story (<=192) and macro (>=166) present.
    "food_overlap": lambda r: r["config_id"] == "food_cpi" and 166 <= r["origin"] <= 192,
    "pooled_all": lambda r: True,
}

COMPARISONS = [
    ("pooled_food", "cqr_meta_level", "cqr"),
    ("pooled_food", "cqr_meta_level_num", "cqr"),
    ("pooled_food", "cqr_conditional", "cqr"),
    ("pooled_food", "cqr_conditional_num", "cqr"),
    ("pooled_synth", "cqr_meta_level", "cqr"),
    ("pooled_synth", "cqr_meta_level_num", "cqr"),
    ("pooled_synth", "cqr_conditional", "cqr"),
    ("pooled_synth", "cqr_conditional_num", "cqr"),
    ("pooled_food", "aci_gated", "aci"),
    ("pooled_synth", "aci_gated", "aci"),
    ("pooled_all", "aci_gated", "aci"),
    ("food_story_subset", "meta_story", "meta_numeric"),
    ("food_story_subset", "meta_story", "native"),
    ("food_story_subset", "meta_numeric", "native"),
    ("food_macro_subset", "meta_full", "meta_story"),
    ("food_macro_subset", "meta_full", "meta_numeric"),
    ("food_macro_subset", "meta_full", "native"),
    # First-class S8: story+macro(live-7) vs story-only, live-7 vs all-22.
    ("food_overlap", "meta_full_live", "meta_story"),
    ("food_overlap", "meta_full", "meta_story"),
    ("food_overlap", "meta_full_live", "meta_full"),
    ("food_overlap", "meta_story", "meta_numeric"),
    ("food_overlap", "meta_full_live", "meta_numeric"),
    ("food_overlap", "meta_full_live", "native"),
    ("food_macro_subset", "meta_full_live", "meta_story"),
    ("food_macro_subset", "meta_full_live", "meta_full"),
    ("food_macro_subset", "meta_full_live", "meta_numeric"),
    ("food_macro_subset", "meta_full_live", "native"),
    ("pooled_food", "cqr", "native"),
    ("pooled_synth", "cqr", "native"),
    # Round-2 variants vs plain CQR and vs the round-1 fused arms.
    ("pooled_food", "cqr_meta_level_h3", "cqr"),
    ("pooled_food", "cqr_meta_level_wide", "cqr"),
    ("pooled_food", "cqr_conditional_shrunk", "cqr"),
    ("pooled_food", "cqr_conditional_shrunk25", "cqr"),
    ("pooled_all", "cqr_conditional_shrunk25", "cqr"),
    ("pooled_food", "cqr_meta_level_h3", "cqr_meta_level"),
    ("pooled_food", "cqr_meta_level_wide", "cqr_meta_level"),
    ("pooled_food", "cqr_conditional_shrunk", "cqr_conditional"),
    ("pooled_synth", "cqr_meta_level_h3", "cqr"),
    ("pooled_synth", "cqr_meta_level_wide", "cqr"),
    ("pooled_synth", "cqr_conditional_shrunk", "cqr"),
    # S8 round 2: inflation-only macro feature (diagnosis-led).
    ("food_overlap", "meta_story_inf", "meta_story"),
    ("food_overlap", "meta_story_inf", "meta_full_live"),
    ("food_macro_subset", "meta_story_inf", "meta_story"),
    ("food_macro_subset", "meta_story_inf", "meta_full_live"),
]


def paired(cases: list[dict], scope, arm_a: str, arm_b: str) -> dict:
    """Paired CRPS comparison arm_a vs arm_b over (config, origin, h) keys."""
    by: dict[tuple, dict[str, dict]] = {}
    for r in cases:
        if not scope(r):
            continue
        key = (r["config_id"], r["origin"], r["h"])
        by.setdefault(key, {})[r["arm"]] = r
    pairs = [(v[arm_a], v[arm_b]) for v in by.values() if arm_a in v and arm_b in v]
    if len(pairs) < 8:
        return {"n": float(len(pairs))}
    a = np.array([p[0]["crps"] for p in pairs])
    b = np.array([p[1]["crps"] for p in pairs])
    t, p = stats.ttest_rel(a, b)
    def _side_mean(side: int, key: str) -> float:
        vals = [p_[side].get(key) for p_ in pairs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": float(len(pairs)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta_pct": float(100.0 * (a.mean() - b.mean()) / max(b.mean(), 1e-12)),
        "t": float(t),
        "p": float(p),
        "win_rate": float(np.mean(a < b)),
        "cov80_a": _side_mean(0, "coverage80"),
        "cov80_b": _side_mean(1, "coverage80"),
        "cov90_a": _side_mean(0, "coverage90"),
        "cov90_b": _side_mean(1, "coverage90"),
        "width_a": _side_mean(0, "width_ratio"),
        "width_b": _side_mean(1, "width_ratio"),
    }


def main() -> None:
    payload = json.loads(RESULTS.read_text())
    if "cases" in payload:
        cases = payload["cases"]
    else:
        # Compact committed format: CRPS pivot only (coverage/width live
        # in the battery summary, not per case).
        cases = []
        for key, per_arm in payload["crps_pivot"].items():
            cfg, origin, h = key.rsplit("|", 2)
            for arm, crps in per_arm.items():
                cases.append(
                    {
                        "config_id": cfg,
                        "origin": int(origin),
                        "h": int(h),
                        "arm": arm,
                        "crps": crps,
                    }
                )
    out: dict[str, dict] = {}
    print("| scope | A | B | n | CRPS A | CRPS B | Δ% | p | win | cov80 A/B |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for scope_name, arm_a, arm_b in COMPARISONS:
        res = paired(cases, SCOPES[scope_name], arm_a, arm_b)
        out[f"{scope_name}::{arm_a}_vs_{arm_b}"] = res
        if res.get("n", 0) < 8 or "p" not in res:
            print(
                f"| {scope_name} | {arm_a} | {arm_b} | {res.get('n', 0):.0f} "
                "| — | — | — | — | — | — |"
            )
            continue
        print(
            f"| {scope_name} | {arm_a} | {arm_b} | {res['n']:.0f} | {res['mean_a']:.4f} "
            f"| {res['mean_b']:.4f} | {res['delta_pct']:+.1f}% | {res['p']:.3g} "
            f"| {res['win_rate']:.2f} | {res['cov80_a']:.3f}/{res['cov80_b']:.3f} |"
        )
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
