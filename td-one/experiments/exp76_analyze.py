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
    ("pooled_food", "cqr", "native"),
    ("pooled_synth", "cqr", "native"),
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
    return {
        "n": float(len(pairs)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta_pct": float(100.0 * (a.mean() - b.mean()) / max(b.mean(), 1e-12)),
        "t": float(t),
        "p": float(p),
        "win_rate": float(np.mean(a < b)),
        "cov80_a": float(np.mean([p[0]["coverage80"] for p in pairs])),
        "cov80_b": float(np.mean([p[1]["coverage80"] for p in pairs])),
        "cov90_a": float(np.mean([p[0]["coverage90"] for p in pairs])),
        "cov90_b": float(np.mean([p[1]["coverage90"] for p in pairs])),
        "width_a": float(np.mean([p[0]["width_ratio"] for p in pairs])),
        "width_b": float(np.mean([p[1]["width_ratio"] for p in pairs])),
    }


def main() -> None:
    payload = json.loads(RESULTS.read_text())
    cases = payload["cases"]
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
