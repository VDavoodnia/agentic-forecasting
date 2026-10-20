"""exp83 round 2 — does macro improve the SHIPPED champion, and other arms.

Round 1 (exp83_stormy_battery.py, run via exp83_battery_resume.py)
cleared the pre-registered bar on the stormy clean scope: story +
standalone inflation (meta_story_inf) beat story-only by -1.10% CRPS
with the origin-cluster bootstrap 95% CI excluding zero [-2.09%, -0.25%].
But every meta arm in that battery modulates the NATIVE forecast and
still loses to plain CQR on the stormy window (2.3496 vs 2.3142). The
shipped design (exp76) is F1 — MetaLevelCqr, whose h3-gated variant is
the Food CPI champion — so the decision-relevant question is whether
the inflation feature improves F1 itself.

Arms (checkpointed per arm under experiments/exp83_checkpoints_r2/,
same run_study protocol as round 1):
  f1_story          MetaLevelCqr(story)
  f1_story_inf      MetaLevelCqr(story + mac_inflation)
  f1_h3_story       MetaLevelCqr(story, min_h=3)        (exp76 champion)
  f1_h3_story_inf   MetaLevelCqr(story + mac_inflation, min_h=3)
  meta_story_housing  MetaArm(story + mpl_housing)  (diagnosis wire:
                      housing signed dir vs signed residual rho=+0.783)
  meta_story_signinf  MetaArm(story + mpl_inflation) (raw signed
                      encoding vs the mac_inflation encoding that won)

Analyses (paired CRPS, origin-cluster bootstrap, seed 20261021):
  - each round-2 arm vs its own no-macro baseline on round-1's scopes
  - round-1 story_inf vs story split by horizon (h<=2 vs h>=3)
  - round-1 story_inf vs story split by story magnitude (median)
Output: experiments/exp83_round2_results.json + printed tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import exp76_fusion_battery as b76  # noqa: E402
import exp83_stormy_battery as sb  # noqa: E402

from td_one.uq import fusion as uqf  # noqa: E402
from td_one.uq.evaluate import build_cases, run_study  # noqa: E402

CKPT = HERE / "exp83_checkpoints_r2"
OUT = HERE / "exp83_round2_results.json"
RNG = np.random.default_rng(20261021)


def build_arms() -> dict:
    MetaArm = b76.MetaArm  # noqa: N806
    return {
        "f1_story": uqf.MetaLevelCqr(sb.FEATS_STORY),
        "f1_story_inf": uqf.MetaLevelCqr(sb.FEATS_STORY + ["mac_inflation"]),
        "f1_h3_story": uqf.MetaLevelCqr(sb.FEATS_STORY, min_h=3),
        "f1_h3_story_inf": uqf.MetaLevelCqr(
            sb.FEATS_STORY + ["mac_inflation"], min_h=3
        ),
        "meta_story_housing": MetaArm(sb.FEATS_STORY + ["mpl_housing"]),
        "meta_story_signinf": MetaArm(sb.FEATS_STORY + ["mpl_inflation"]),
    }


def paired(
    cases: list[dict], arm: str, base: str, scope: list[tuple[int, int]]
) -> dict[str, float]:
    """paired_vs_story generalized to an arbitrary baseline arm."""
    by: dict[tuple[int, int], dict[str, float]] = {}
    for c in cases:
        if c["config_id"] != "food_cpi":
            continue
        by.setdefault((c["origin"], c["h"]), {})[c["arm"]] = c["crps"]
    keys = [k for k in scope if arm in by.get(k, {}) and base in by.get(k, {})]
    if len(keys) < 8:
        return {"n": float(len(keys))}
    diffs = np.array([by[k][arm] - by[k][base] for k in keys])
    basev = np.array([by[k][base] for k in keys])
    t, p = stats.ttest_rel([by[k][arm] for k in keys], [by[k][base] for k in keys])
    origins = sorted({k[0] for k in keys})
    per_origin: dict[int, list[float]] = {o: [] for o in origins}
    per_origin_base: dict[int, list[float]] = {o: [] for o in origins}
    for k, d, s in zip(keys, diffs, basev, strict=True):
        per_origin[k[0]].append(float(d))
        per_origin_base[k[0]].append(float(s))
    boot = []
    for _ in range(4000):
        draw = RNG.choice(origins, size=len(origins), replace=True)
        d = np.concatenate([per_origin[o] for o in draw])
        s = np.concatenate([per_origin_base[o] for o in draw])
        boot.append(d.mean() / max(s.mean(), 1e-12))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": float(len(keys)),
        "n_origins": float(len(origins)),
        "diff": float(diffs.mean()),
        "rel_pct": float(100.0 * diffs.mean() / basev.mean()),
        "t_p": float(p),
        "win": float(np.mean(diffs < 0)),
        "boot_rel_lo": float(100.0 * lo),
        "boot_rel_hi": float(100.0 * hi),
    }


def fmt(tag: str, a: dict[str, float]) -> str:
    if a.get("n", 0) < 8:
        return f"{tag:44s} n too small"
    return (
        f"{tag:44s} rel={a['rel_pct']:+.2f}%  boot95=[{a['boot_rel_lo']:+.2f}%,"
        f"{a['boot_rel_hi']:+.2f}%]  t_p={a['t_p']:.3g}  win={a['win']:.2f}  "
        f"n={int(a['n'])}"
    )


def main() -> None:
    doc = json.loads(b76.FORECASTS.read_text())
    rows = [r for r in doc["rows"] if r["config_id"] == "food_cpi"]
    extra = sb.extra_features_83()
    cases_by_config = build_cases(rows, extra_features=extra)

    arms = build_arms()
    CKPT.mkdir(exist_ok=True)
    new_cases: list[dict] = []
    for name, fn in arms.items():
        f = CKPT / f"{name}.json"
        if f.exists():
            new_cases += json.loads(f.read_text())
            print(f"[ckpt] {name}: loaded", flush=True)
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
        new_cases += dicts
        print(f"[done] {name}: {len(dicts)} cases in {time.time() - t0:.0f}s", flush=True)

    main_cases = json.loads(sb.OUT.read_text())["cases"]
    cases = main_cases + new_cases

    def scope_keys(lo: int, hi: int, hs=range(1, 7)) -> list[tuple[int, int]]:
        return [(o, h) for o in range(lo, hi + 1) for h in hs]

    scopes = {
        "stormy_clean_133_164": scope_keys(133, 164),
        "stormy_feat2021": scope_keys(133, 144),
        "stormy_feat2022": scope_keys(145, 156),
        "stormy_feat2023": scope_keys(157, 164),
        "quiet_clean_166_192": scope_keys(166, 192),
    }
    out: dict[str, dict] = {"vs_baseline": {}, "splits": {}}
    pairs = [
        ("f1_story_inf", "f1_story"),
        ("f1_h3_story_inf", "f1_h3_story"),
        ("f1_h3_story", "meta_story"),
        ("f1_story", "meta_story"),
        ("meta_story_housing", "meta_story"),
        ("meta_story_signinf", "meta_story"),
        ("meta_story_inf", "f1_h3_story_inf"),
    ]
    for sname, keys in scopes.items():
        out["vs_baseline"][sname] = {}
        for arm, base in pairs:
            r = paired(cases, arm, base, keys)
            out["vs_baseline"][sname][f"{arm} vs {base}"] = r
            print(fmt(f"{sname}: {arm} vs {base}", r))

    # horizon split of the round-1 headline (story_inf vs story)
    for tag, hs in (("h_le2", range(1, 3)), ("h_ge3", range(3, 7))):
        keys = scope_keys(133, 164, hs)
        r = paired(cases, "meta_story_inf", "meta_story", keys)
        out["splits"][f"stormy_inf_{tag}"] = r
        print(fmt(f"split {tag}: meta_story_inf vs meta_story", r))
        r2 = paired(cases, "f1_h3_story_inf", "f1_h3_story", keys)
        out["splits"][f"stormy_f1h3inf_{tag}"] = r2
        print(fmt(f"split {tag}: f1_h3_story_inf vs f1_h3_story", r2))

    # story-magnitude split (median over stormy origins)
    mag = {}
    for idx in range(133, 165):
        feats = extra.get(("food_cpi", idx), {})
        mag[idx] = float(feats.get("st_magnitude", 0.0))
    med = float(np.median(list(mag.values())))
    for tag, sel in (
        ("stmag_low", [o for o, v in mag.items() if v <= med]),
        ("stmag_high", [o for o, v in mag.items() if v > med]),
    ):
        keys = [(o, h) for o in sel for h in range(1, 7)]
        r = paired(cases, "meta_story_inf", "meta_story", keys)
        out["splits"][f"stormy_inf_{tag}"] = r
        print(fmt(f"split {tag}: meta_story_inf vs meta_story", r))

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
