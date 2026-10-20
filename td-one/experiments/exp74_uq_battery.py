"""exp74 Phase 2c: run the UQ methods battery on cached Chronos forecasts.

Loads ``exp74_forecasts.json`` (from exp74_generate_data.py), merges
story-pipeline + macro-signal features for Food CPI origins, runs all arms
through the PIT-safe study driver, and writes ``exp74_results.json``.

Arms (S-codes from docs/UQ_SURVEY.md):
  native (S1), fixed_expansion (S2), rolling_conformal (S3), aci (S4),
  aci_bias (S4b), weighted_conformal (S5), cqr (S6),
  meta_numeric (S7), meta_full (S8 = S7 + story/macro features).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from td_one.uq import methods as uq_methods
from td_one.uq.evaluate import run_study, summarize
from td_one.uq.features import macro_features, story_features

M = uq_methods

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORECASTS = HERE / "exp74_forecasts.json"
STORY_SIGNALS = HERE / "exp74_story_signals.json"
DB_PATH = ROOT / "data" / "news" / "news_2026.db"
OUT = HERE / "exp74_results.json"

FOOD_BASE = date(2010, 1, 1)  # index 0 of benchmarks/food_cpi/data.csv


def ym_of_index(idx: int) -> str:
    y = FOOD_BASE.year + (FOOD_BASE.month - 1 + idx) // 12
    m = (FOOD_BASE.month - 1 + idx) % 12 + 1
    return f"{y:04d}-{m:02d}"


def macro_by_origin() -> dict[int, dict[str, float]]:
    """Macro Signal Service features per Food CPI origin index."""
    from td_one.news.macro_signal import signals_asof  # noqa: PLC0415

    out: dict[int, dict[str, float]] = {}
    if not DB_PATH.exists():
        return out
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for idx in range(84, 198):
            ym = ym_of_index(idx)
            try:
                sig = signals_asof(con, ym)
            except sqlite3.Error:
                sig = {}
            out[idx] = macro_features(sig if sig else None)
    finally:
        con.close()
    return out


def build_extra_features() -> dict[tuple[str, int], dict[str, float]]:
    extra: dict[tuple[str, int], dict[str, float]] = {}
    story: dict[str, dict] = {}
    if STORY_SIGNALS.exists():
        story = json.loads(STORY_SIGNALS.read_text())
    macro = macro_by_origin()
    for idx in range(84, 198):
        ym = ym_of_index(idx)
        feats: dict[str, float] = {}
        feats.update(story_features(story.get(ym)))
        feats.update(macro.get(idx, {}))
        extra[("food_cpi", idx)] = feats
    return extra


class MetaArm:
    """arm_meta with the fitted GBM cached per (origin, bank).

    The model is pooled over horizon steps, so refitting per case would
    be waste.
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self._bank_key: tuple | None = None
        self._model = None

    def __call__(self, case, bank):
        key = (case.origin, len(bank))
        if key != self._bank_key:
            self._bank_key = key
            self._model = M.fit_meta_model(bank, self.features)
        if self._model is None:
            return M.arm_rolling_conformal(case, bank)
        pred = float(self._model.predict(M.meta_matrix([case], self.features))[0])
        pred = max(pred, 1e-12)
        m, q10, q90 = case.interval_edges()
        return max(pred / max(m - q10, 1e-12), 0.0), max(pred / max(q90 - m, 1e-12), 0.0)


def main() -> None:
    doc = json.loads(FORECASTS.read_text())
    rows = doc["rows"]
    from td_one.uq.evaluate import build_cases  # noqa: PLC0415

    extra = build_extra_features()
    cases_by_config = build_cases(rows, extra_features=extra)

    arms = {
        "native": M.arm_native,
        "fixed_expansion": M.arm_fixed_expansion,
        "rolling_conformal": M.arm_rolling_conformal,
        "conformal_rel": M.arm_conformal_rel,
        "aci": M.arm_aci,
        "aci_bias": M.arm_aci_bias,
        "aci_rel": M.arm_aci_rel,
        "weighted_conformal": M.arm_weighted_conformal,
        "cqr": M.arm_cqr,
        "meta_numeric": MetaArm(M.META_FEATURES_NUMERIC),
        "meta_full": MetaArm(M.META_FEATURES_NUMERIC + M.META_FEATURES_STORY),
    }
    results = run_study(cases_by_config, arms)

    scopes = {
        "pooled_synth": lambda r: r.config_id.startswith("synth:"),
        "pooled_synth_S1_4sigma": lambda r: r.config_id.startswith("synth:S1:4.0"),
        "pooled_synth_S1_4sigma_straddle": lambda r: (
            r.config_id.startswith("synth:S1:4.0") and 27 <= r.origin <= 33
        ),
        "pooled_synth_control": lambda r: r.config_id.startswith("synth:none"),
        "pooled_food": lambda r: r.config_id == "food_cpi",
        "food_story_subset": lambda r: r.config_id == "food_cpi" and 132 <= r.origin <= 191,
        "pooled_all": lambda r: True,
    }
    summary = {name: summarize(results, scope) for name, scope in scopes.items()}
    per_config = {
        cid: summarize(results, lambda r, c=cid: r.config_id == c)
        for cid in sorted(cases_by_config)
    }
    payload = {
        "config": {"n_rows": len(rows), "arms": list(arms)},
        "summary": summary,
        "per_config": per_config,
        "cases": [
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
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({len(results)} case-results)")

    for scope_name in ("pooled_synth_S1_4sigma_straddle", "pooled_synth", "pooled_food"):
        print(f"\n== {scope_name} ==")
        for arm, s in summary[scope_name].items():
            print(
                f"{arm:20s} cov80={s['coverage80']:.3f} cov90={s['coverage90']:.3f} "
                f"width_ratio={s['width_ratio']:.3f} crps={s['crps']:.4f} "
                f"p={s['crps_paired_p']:.3g} win={s['crps_win_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
