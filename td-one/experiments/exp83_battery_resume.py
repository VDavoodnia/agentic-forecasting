"""exp83 RECOVERY runner — checkpointed per-arm execution of the battery.

Written after two VM reboots killed the monolithic
``exp83_stormy_battery.py`` run (2026-10-20; the second at ~1000 CPU-sec
with no output, since the original writes results only at the end).

Why this is the identical protocol, not a fork: ``run_study`` calls each
arm independently per case with the same PIT bank (see
``td_one.uq.evaluate.run_study``), so running the arms one at a time and
concatenating the case results is exactly equivalent to the single
11-arm call. Arms, feature sets, scopes, paired tests and the bootstrap
RNG are all imported / copied verbatim from ``exp83_stormy_battery``.
The only change is operational: each arm's case-results are checkpointed
to ``experiments/exp83_checkpoints/<arm>.json`` the moment the arm
finishes, so a reboot loses at most one arm.

Usage: .venv/bin/python experiments/exp83_battery_resume.py
Output: experiments/exp83_stormy_results.json (same schema as the
original battery) + printed paired tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import exp76_fusion_battery as b76  # noqa: E402
import exp83_stormy_battery as sb  # noqa: E402

from td_one.uq import methods as M  # noqa: E402, N812
from td_one.uq.evaluate import (  # noqa: E402
    CaseResult,
    build_cases,
    run_study,
    summarize,
)

CKPT = HERE / "exp83_checkpoints"


def build_arms() -> dict:
    """Identical arm construction to exp83_stormy_battery.main()."""
    MetaArm = b76.MetaArm  # noqa: N806
    return {
        "native": M.arm_native,
        "cqr": M.arm_cqr,
        "meta_numeric": MetaArm(sb.FEATS_NUM),
        "meta_story": MetaArm(sb.FEATS_STORY),
        "meta_full_live": MetaArm(sb.FEATS_FULL_LIVE),
        "meta_story_inf": MetaArm(sb.FEATS_STORY + ["mac_inflation"]),
        "meta_macro_live": MetaArm(sb.FEATS_MACRO_LIVE),
        "meta_story_persign": MetaArm(sb.FEATS_PERSIGN),
        "meta_story_perabs": MetaArm(sb.FEATS_PERABS),
        "meta_story_inter": MetaArm(sb.FEATS_INTER),
        "meta_story_gated": MetaArm(sb.FEATS_GATED),
    }


def main() -> None:
    doc = json.loads(b76.FORECASTS.read_text())
    rows = [r for r in doc["rows"] if r["config_id"] == "food_cpi"]
    extra = sb.extra_features_83()
    cases_by_config = build_cases(rows, extra_features=extra)

    arms = build_arms()
    CKPT.mkdir(exist_ok=True)
    case_dicts: list[dict] = []
    results_all = []
    for name, fn in arms.items():
        f = CKPT / f"{name}.json"
        if f.exists():
            saved = json.loads(f.read_text())
            case_dicts += saved
            # summarize() reads only these fields (never width80).
            results_all += [
                CaseResult(width80=0.0, **d) for d in saved
            ]
            print(f"[ckpt] {name}: loaded {len(saved)} cases", flush=True)
            continue
        t0 = time.time()
        results = run_study(cases_by_config, {name: fn})
        dicts = [
            {
                "config_id": r.config_id,
                "origin": r.origin,
                "h": r.h,
                "arm": r.arm,
                "coverage80": r.coverage80,
                "coverage90": r.coverage90,
                "width_ratio": r.width_ratio,
                "crps": r.crps,
                "crps_native": r.crps_native,
            }
            for r in results
        ]
        f.write_text(json.dumps(dicts))
        case_dicts += dicts
        results_all += results
        print(
            f"[done] {name}: {len(dicts)} cases in {time.time() - t0:.0f}s",
            flush=True,
        )

    # ---- analysis: verbatim from exp83_stormy_battery.main() ----
    def scope_keys(lo: int, hi: int) -> list[tuple[int, int]]:
        return [(o, h) for o in range(lo, hi + 1) for h in range(1, 7)]

    scopes = {
        "stormy_clean_133_164": scope_keys(133, 164),
        "stormy_all_132_164": scope_keys(132, 164),
        "stormy_feat2021": scope_keys(133, 144),
        "stormy_feat2022": scope_keys(145, 156),
        "stormy_feat2023": scope_keys(157, 164),
        "quiet_clean_166_192": scope_keys(166, 192),
    }
    macro_arms = [
        "meta_full_live",
        "meta_story_inf",
        "meta_macro_live",
        "meta_story_persign",
        "meta_story_perabs",
        "meta_story_inter",
        "meta_story_gated",
        "meta_numeric",
    ]
    analysis: dict[str, dict[str, dict[str, float]]] = {}
    for name, keys in scopes.items():
        analysis[name] = {
            arm: sb.paired_vs_story(case_dicts, arm, keys) for arm in macro_arms
        }
    summary = {
        "stormy_clean": summarize(
            results_all,
            lambda r: r.config_id == "food_cpi" and 133 <= r.origin <= 164,
        ),
        "quiet_clean": summarize(
            results_all,
            lambda r: r.config_id == "food_cpi" and 166 <= r.origin <= 192,
        ),
    }
    sb.OUT.write_text(
        json.dumps(
            {
                "config": {"arms": list(arms), "pit_keying": "last_observed_month"},
                "summary": summary,
                "paired_vs_story": analysis,
                "cases": case_dicts,
            },
            indent=1,
        )
    )
    print(f"wrote {sb.OUT} ({len(case_dicts)} case-results)")

    for name in scopes:
        print(f"\n== paired vs meta_story: {name} ==")
        for arm in macro_arms:
            a = analysis[name][arm]
            if a.get("n", 0) < 8:
                print(f"{arm:22s} n too small")
                continue
            print(
                f"{arm:22s} rel={a['rel_pct']:+.2f}%  boot95=[{a['boot_rel_lo']:+.2f}%,"
                f"{a['boot_rel_hi']:+.2f}%]  t_p={a['t_p']:.3g}  win={a['win']:.2f}  "
                f"n={int(a['n'])}"
            )
    for name, s in summary.items():
        print(f"\n== summarize: {name} ==")
        for arm in ("native", "cqr", "meta_numeric", "meta_story", "meta_full_live"):
            if arm in s:
                v = s[arm]
                print(
                    f"{arm:22s} cov80={v['coverage80']:.3f} cov90={v['coverage90']:.3f} "
                    f"width={v['width_ratio']:.3f} crps={v['crps']:.4f} "
                    f"p_vs_native={v['crps_paired_p']:.3g} win={v['crps_win_rate']:.2f}"
                )


if __name__ == "__main__":
    main()
