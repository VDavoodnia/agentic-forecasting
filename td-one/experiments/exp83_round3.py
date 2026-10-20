"""exp83 round 3 — the risk-flag wires the diagnosis exposed.

Round-2 diagnosis (exp83_macro_diagnosis.py, stormy origins 133-164):
the wars_geopolitics and food_prices HIGH-RISK flags anti-track native
exceedance (Spearman rho=-0.754, p=6.3e-07 and rho=-0.527, p=0.0019) —
when the flag is high, realized values fall INSIDE the native band more
than usual (the TSFM's intervals already price those news regimes, or
the flagged risk does not materialize as a price surprise). A meta
model given the flags can learn to narrow there. Round 1's arms never
saw per-category risk (only directions).

Arms (checkpointed under experiments/exp83_checkpoints_r3/, same
run_study protocol; risk features keyed PIT via signals_asof at the
feature month, exactly like the direction features):
  meta_story_inf_risk   MetaArm(story + mac_inflation
                              + mrisk_wars_geopolitics + mrisk_food_prices)
  f1_h3_story_inf_risk  MetaLevelCqr(same features, min_h=3)

Comparisons: each vs its no-risk counterpart (meta_story_inf /
f1_h3_story_inf) on stormy_clean + per-year scopes.
Output: experiments/exp83_round3_results.json + printed tables.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import exp76_fusion_battery as b76  # noqa: E402
import exp83_round2 as r2  # noqa: E402
import exp83_stormy_battery as sb  # noqa: E402

from td_one.news.macro_signal import signals_asof  # noqa: E402
from td_one.uq import fusion as uqf  # noqa: E402
from td_one.uq.evaluate import build_cases, run_study  # noqa: E402

CKPT = HERE / "exp83_checkpoints_r3"
OUT = HERE / "exp83_round3_results.json"
RISK_CATS = ["wars_geopolitics", "food_prices"]
RISK_FEATS = [f"mrisk_{c}" for c in RISK_CATS]


def extra_with_risk() -> dict:
    extra = sb.extra_features_83()
    con = sqlite3.connect(f"file:{sb.DB_PATH}?mode=ro", uri=True)
    try:
        for idx in range(84, 198):
            ym = b76.feature_month(idx)
            sig = signals_asof(con, ym)
            feats = extra.setdefault(("food_cpi", idx), {})
            for cat in RISK_CATS:
                s = sig.get(cat)
                feats[f"mrisk_{cat}"] = (
                    1.0 if (s is not None and s.risk == "high") else 0.0
                )
    finally:
        con.close()
    return extra


def main() -> None:
    doc = json.loads(b76.FORECASTS.read_text())
    rows = [r for r in doc["rows"] if r["config_id"] == "food_cpi"]
    extra = extra_with_risk()
    cases_by_config = build_cases(rows, extra_features=extra)

    MetaArm = b76.MetaArm  # noqa: N806
    feats = sb.FEATS_STORY + ["mac_inflation"] + RISK_FEATS
    arms = {
        "meta_story_inf_risk": MetaArm(feats),
        "f1_h3_story_inf_risk": uqf.MetaLevelCqr(feats, min_h=3),
    }
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

    cases = json.loads(sb.OUT.read_text())["cases"]
    for f in sorted((HERE / "exp83_checkpoints_r2").glob("*.json")):
        cases += json.loads(f.read_text())
    cases += new_cases

    def scope_keys(lo: int, hi: int) -> list[tuple[int, int]]:
        return [(o, h) for o in range(lo, hi + 1) for h in range(1, 7)]

    scopes = {
        "stormy_clean_133_164": scope_keys(133, 164),
        "stormy_feat2021": scope_keys(133, 144),
        "stormy_feat2022": scope_keys(145, 156),
        "stormy_feat2023": scope_keys(157, 164),
        "quiet_clean_166_192": scope_keys(166, 192),
    }
    pairs = [
        ("meta_story_inf_risk", "meta_story_inf"),
        ("f1_h3_story_inf_risk", "f1_h3_story_inf"),
        ("f1_h3_story_inf_risk", "f1_h3_story"),
        ("meta_story_inf_risk", "meta_story"),
    ]
    out: dict[str, dict] = {}
    for sname, keys in scopes.items():
        out[sname] = {}
        for arm, base in pairs:
            r = r2.paired(cases, arm, base, keys)
            out[sname][f"{arm} vs {base}"] = r
            print(r2.fmt(f"{sname}: {arm} vs {base}", r))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
