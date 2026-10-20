"""exp83: S8 rerun on the stormy window (Food CPI origins 2021-01..2023-09).

exp76's S8 was inconclusive: its clean scope (origins 166-192, feature
months 2023-10+) sat in a signal-quiet window where even the story
effect vanished (meta_story vs meta_numeric -0.2%, p=0.83). exp83
backfilled the Macro Signal store into 2021-01..2023-09 (the inflation
shock) and reruns the identical protocol here: same cases
(exp74_forecasts.json), same corrected PIT keying (feature month = last
observed month), same MetaArm GBM, same feature encodings (imported from
exp76_fusion_battery so the protocol cannot drift).

Arms:
  native, cqr                      anchors (window difficulty context)
  meta_numeric                     numeric only
  meta_story                       numeric + story            (baseline)
  meta_full_live                   + live-7 macro aggregates  (headline)
  meta_story_inf                   + mac_inflation standalone
  meta_macro_live                  numeric + live-7, no story (macro-only)
  meta_story_persign               + per-category SIGNED directions (live-7)
  meta_story_perabs                + per-category |directions| (live-7)
  meta_story_inter                 + |inflation dir| x story magnitude
  meta_story_gated                 + live-7 aggregates, signals >3mo dropped

Pre-registered bar: a macro arm beats meta_story on CRPS with the
origin-cluster bootstrap 95% CI excluding zero, on the clean stormy
scope (origins 133-164; feature months 2021-01..2023-08, where story
features exist for all 32 origins and macro is keyed PIT-safely).

Output: experiments/exp83_stormy_results.json (case-level, re-analyzable)
plus printed paired tables (t-test + origin-cluster bootstrap CIs).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import exp76_fusion_battery as b76  # noqa: E402  (protocol: do not fork it)

from td_one.news.macro_signal import signals_asof  # noqa: E402
from td_one.uq import methods as uq_methods  # noqa: E402
from td_one.uq.evaluate import build_cases, run_study, summarize  # noqa: E402

M = uq_methods
OUT = HERE / "exp83_stormy_results.json"
DB_PATH = b76.DB_PATH
LIVE_CATS = b76.LIVE_CATS

RNG = np.random.default_rng(20261020)


def _month_diff(a: str, b: str) -> int:
    """Months from b to a (both YYYY-MM)."""
    return (int(a[:4]) - int(b[:4])) * 12 + (int(a[5:7]) - int(b[5:7]))


def extra_features_83() -> dict[tuple[str, int], dict[str, float]]:
    """exp76 extra features + per-category / gated / interaction features."""
    extra = b76.build_extra_features()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for idx in range(84, 198):
            ym = b76.feature_month(idx)
            sig = signals_asof(con, ym)
            feats = extra.setdefault(("food_cpi", idx), {})
            # per-category signed / abs directions (live-7)
            for cat in LIVE_CATS:
                s = sig.get(cat)
                d = float(s.direction) if s is not None else 0.0
                feats[f"mpl_{cat}"] = d
                feats[f"mpa_{cat}"] = abs(d)
            # gated live-7 aggregates: drop signals older than 3 months
            fresh = {
                c: s
                for c, s in sig.items()
                if c in LIVE_CATS and _month_diff(ym, s.month) <= 3
            }
            from td_one.uq.features import macro_features  # noqa: PLC0415

            gated = macro_features(fresh or None)
            feats.update({"gcl_" + k[4:]: v for k, v in gated.items()})
            # staleness of the inflation signal at this origin (months)
            inf = sig.get("inflation")
            feats["stl_inflation"] = (
                float(_month_diff(ym, inf.month)) if inf is not None else 99.0
            )
            # interaction: |inflation direction| x story magnitude
            st_mag = feats.get("st_magnitude", 0.0)
            feats["x_infabs_stmag"] = feats.get("mpa_inflation", 0.0) * st_mag
    finally:
        con.close()
    return extra


_MAC_KEYS = [k for k in M.META_FEATURES_STORY if k.startswith("mac_")]
FEATS_NUM = list(M.META_FEATURES_NUMERIC)
FEATS_STORY = FEATS_NUM + list(M.META_FEATURES_STORY[:7])
FEATS_FULL_LIVE = FEATS_STORY + ["mcl_" + k[4:] for k in _MAC_KEYS]
FEATS_MACRO_LIVE = FEATS_NUM + ["mcl_" + k[4:] for k in _MAC_KEYS]
FEATS_PERSIGN = FEATS_STORY + [f"mpl_{c}" for c in LIVE_CATS]
FEATS_PERABS = FEATS_STORY + [f"mpa_{c}" for c in LIVE_CATS]
FEATS_INTER = FEATS_STORY + ["x_infabs_stmag", "stl_inflation"]
FEATS_GATED = FEATS_STORY + ["gcl_" + k[4:] for k in _MAC_KEYS]


def paired_vs_story(
    cases: list[dict], arm: str, scope: list[tuple[int, int]]
) -> dict[str, float]:
    """Paired CRPS comparison arm vs meta_story over an (origin,h) scope.

    Returns mean diff, relative diff, paired t p, win rate, and an
    origin-cluster bootstrap 95% CI for the mean diff (origins resampled
    with replacement; cases of a resampled origin enter with multiplicity).
    """
    by: dict[tuple[int, int], dict[str, float]] = {}
    for c in cases:
        if c["config_id"] != "food_cpi":
            continue
        by.setdefault((c["origin"], c["h"]), {})[c["arm"]] = c["crps"]
    keys = [k for k in scope if arm in by.get(k, {}) and "meta_story" in by.get(k, {})]
    if len(keys) < 8:
        return {"n": float(len(keys))}
    diffs = np.array([by[k][arm] - by[k]["meta_story"] for k in keys])
    story = np.array([by[k]["meta_story"] for k in keys])
    t, p = stats.ttest_rel(
        [by[k][arm] for k in keys], [by[k]["meta_story"] for k in keys]
    )
    origins = sorted({k[0] for k in keys})
    per_origin: dict[int, list[float]] = {o: [] for o in origins}
    per_origin_story: dict[int, list[float]] = {o: [] for o in origins}
    for k, d, s in zip(keys, diffs, story, strict=True):
        per_origin[k[0]].append(float(d))
        per_origin_story[k[0]].append(float(s))
    boot = []
    for _ in range(4000):
        draw = RNG.choice(origins, size=len(origins), replace=True)
        d = np.concatenate([per_origin[o] for o in draw])
        s = np.concatenate([per_origin_story[o] for o in draw])
        boot.append(d.mean() / max(s.mean(), 1e-12))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": float(len(keys)),
        "n_origins": float(len(origins)),
        "diff": float(diffs.mean()),
        "rel_pct": float(100.0 * diffs.mean() / story.mean()),
        "t_p": float(p),
        "win": float(np.mean(diffs < 0)),
        "boot_rel_lo": float(100.0 * lo),
        "boot_rel_hi": float(100.0 * hi),
    }


def main() -> None:
    doc = json.loads(b76.FORECASTS.read_text())
    rows = [r for r in doc["rows"] if r["config_id"] == "food_cpi"]
    extra = extra_features_83()
    cases_by_config = build_cases(rows, extra_features=extra)

    MetaArm = b76.MetaArm  # noqa: N806
    arms = {
        "native": M.arm_native,
        "cqr": M.arm_cqr,
        "meta_numeric": MetaArm(FEATS_NUM),
        "meta_story": MetaArm(FEATS_STORY),
        "meta_full_live": MetaArm(FEATS_FULL_LIVE),
        "meta_story_inf": MetaArm(FEATS_STORY + ["mac_inflation"]),
        "meta_macro_live": MetaArm(FEATS_MACRO_LIVE),
        "meta_story_persign": MetaArm(FEATS_PERSIGN),
        "meta_story_perabs": MetaArm(FEATS_PERABS),
        "meta_story_inter": MetaArm(FEATS_INTER),
        "meta_story_gated": MetaArm(FEATS_GATED),
    }
    results = run_study(cases_by_config, arms)
    case_dicts = [
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

    def scope_keys(lo: int, hi: int) -> list[tuple[int, int]]:
        return [
            (o, h) for o in range(lo, hi + 1) for h in range(1, 7)
        ]

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
            arm: paired_vs_story(case_dicts, arm, keys) for arm in macro_arms
        }
    summary = {
        "stormy_clean": summarize(
            results, lambda r: r.config_id == "food_cpi" and 133 <= r.origin <= 164
        ),
        "quiet_clean": summarize(
            results, lambda r: r.config_id == "food_cpi" and 166 <= r.origin <= 192
        ),
    }
    OUT.write_text(
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
    print(f"wrote {OUT} ({len(case_dicts)} case-results)")

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
