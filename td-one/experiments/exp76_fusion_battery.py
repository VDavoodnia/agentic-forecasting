"""exp76 Part A: fuse CQR with the meta uncertainty signal (exp74's two wins).

Loads exp74_forecasts.json (cached Chronos forecasts), merges story +
macro features with CORRECTED PIT keying (features for origin idx use the
last OBSERVED month ym_of_index(idx-1), not the first forecast month —
exp74 keyed one month ahead; see exp76_report.md), and runs the fusion
arms through the PIT-safe study driver.

Arms:
  native, cqr (plain S6), aci (S4), aci_gated (regime-gated ACI),
  meta_numeric (S7, corrected keying), meta_story (S7 + story, corrected),
  meta_full (S7 + story + macro, corrected),
  cqr_meta_level (F1 fusion, numeric+story), cqr_meta_level_num (F1, numeric),
  cqr_conditional (F2 fusion, numeric+story), cqr_conditional_num (F2, numeric).
Note: ``M.META_FEATURES_STORY`` already bundles the 8 mac_* keys; the
S8 split is reproduced here with an explicit story-only feature list.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from td_one.uq import fusion as uqf
from td_one.uq import methods as uq_methods
from td_one.uq.evaluate import run_study, summarize
from td_one.uq.features import macro_features, story_features
from td_one.uq.fusion import month_before

M = uq_methods

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORECASTS = HERE / "exp74_forecasts.json"
STORY_SIGNALS = HERE / "exp74_story_signals.json"
DB_PATH = ROOT / "data" / "news" / "news_2026.db"
OUT = HERE / "exp76_fusion_results.json"

FOOD_BASE = date(2010, 1, 1)  # index 0 of benchmarks/food_cpi/data.csv


def ym_of_index(idx: int) -> str:
    y = FOOD_BASE.year + (FOOD_BASE.month - 1 + idx) // 12
    m = (FOOD_BASE.month - 1 + idx) % 12 + 1
    return f"{y:04d}-{m:02d}"


def feature_month(idx: int) -> str:
    """PIT-safe feature month for a forecast origin.

    Origin ``idx`` has context series[:idx], i.e. last observed month is
    ``ym_of_index(idx - 1)``. Story/macro features must come from that
    month, never from the first forecast month.
    """
    return month_before(ym_of_index(idx))


def macro_by_origin() -> dict[int, dict[str, float]]:
    """Macro Signal Service features per Food CPI origin (PIT-safe keying)."""
    from td_one.news.macro_signal import signals_asof  # noqa: PLC0415

    out: dict[int, dict[str, float]] = {}
    if not DB_PATH.exists():
        return out
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for idx in range(84, 198):
            ym = feature_month(idx)
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
        ym = feature_month(idx)
        feats: dict[str, float] = {}
        feats.update(story_features(story.get(ym)))
        feats.update(macro.get(idx, {}))
        extra[("food_cpi", idx)] = feats
    return extra


class MetaArm:
    """arm_meta with the fitted GBM cached per (origin, bank length)."""

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


FEATS_NUM = M.META_FEATURES_NUMERIC
# META_FEATURES_STORY = 7 st_* keys + 8 mac_* keys; split for the S8 ablation.
FEATS_STORY = M.META_FEATURES_NUMERIC + M.META_FEATURES_STORY[:7]
FEATS_FULL = M.META_FEATURES_NUMERIC + M.META_FEATURES_STORY


def main() -> None:
    doc = json.loads(FORECASTS.read_text())
    rows = doc["rows"]
    from td_one.uq.evaluate import build_cases  # noqa: PLC0415

    extra = build_extra_features()
    cases_by_config = build_cases(rows, extra_features=extra)

    arms = {
        "native": M.arm_native,
        "cqr": M.arm_cqr,
        "aci": M.arm_aci,
        "aci_gated": uqf.arm_aci_gated,
        "meta_numeric": MetaArm(FEATS_NUM),
        "meta_story": MetaArm(FEATS_STORY),
        "meta_full": MetaArm(FEATS_FULL),
        "cqr_meta_level": uqf.MetaLevelCqr(FEATS_STORY),
        "cqr_meta_level_num": uqf.MetaLevelCqr(FEATS_NUM),
        "cqr_conditional": uqf.ConditionalCqr(FEATS_STORY),
        "cqr_conditional_num": uqf.ConditionalCqr(FEATS_NUM),
    }
    results = run_study(cases_by_config, arms)

    scopes = {
        "pooled_synth": lambda r: r.config_id.startswith("synth:"),
        "pooled_synth_S1_4sigma": lambda r: r.config_id.startswith("synth:S1:4.0"),
        "pooled_synth_control": lambda r: r.config_id.startswith("synth:none"),
        "pooled_food": lambda r: r.config_id == "food_cpi",
        # Corrected keying: story keys 2021-01..2025-12 map to origins 133..192.
        "food_story_subset": lambda r: r.config_id == "food_cpi" and 133 <= r.origin <= 192,
        # Macro signals now start 2023-10; with corrected keying origins 166..197
        # have macro features (was 175..197 when backfill started 2024-07).
        "food_macro_subset": lambda r: r.config_id == "food_cpi" and 166 <= r.origin <= 197,
        "pooled_all": lambda r: True,
    }
    summary = {name: summarize(results, scope) for name, scope in scopes.items()}
    payload = {
        "config": {"n_rows": len(rows), "arms": list(arms), "pit_keying": "last_observed_month"},
        "summary": summary,
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

    for scope_name in ("food_story_subset", "food_macro_subset", "pooled_food", "pooled_synth"):
        print(f"\n== {scope_name} ==")
        for arm, s in summary[scope_name].items():
            print(
                f"{arm:22s} cov80={s['coverage80']:.3f} cov90={s['coverage90']:.3f} "
                f"width_ratio={s['width_ratio']:.3f} crps={s['crps']:.4f} "
                f"p={s['crps_paired_p']:.3g} win={s['crps_win_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
