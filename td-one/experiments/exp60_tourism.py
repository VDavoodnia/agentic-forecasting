"""Experiment 60: full ladder on Tourism Monthly (Monash archive).

366 monthly series (Kaggle tourism competition, Athanasopoulos et al. 2011),
horizon h=24 (Monash standard). Ladder: snaive / naive / m0 / ets / arima /
Chronos-2 / TimesFM-2.5 / equal weights / proposed method (regime router +
gates, m4 regime spec). Primary metric: CRPS (paired t-test vs equal
weights). MASE/sMAPE reported for literature comparability.

Usage:
    .venv/bin/python experiments/exp60_tourism.py            # classical + eval
    .venv/bin/python experiments/exp60_tourism.py --tsfm      # + TSFM inference (slow)
    .venv/bin/python experiments/exp60_tourism.py --eval-only # rerun eval from JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import monash_common as mc  # noqa: E402

DATASET = "tourism_monthly"
TSF = ROOT / "data" / "tourism" / "tourism_monthly_dataset.tsf"
RESULTS = ROOT / "experiments" / "exp60_results.json"
TSFM_JSON = ROOT / "experiments" / "exp60_tsfm.json"
NEWS_DB = ROOT / "data" / "news" / "news_2026.db"


def compute_origins(series: list) -> dict[str, str]:
    """{series_id: origin_ym} from the .tsf start timestamps."""
    starts = mc.parse_tsf_starts(TSF)
    return {sid: mc.origin_ym(starts[sid], len(vals), 24)
            for sid, vals in series if sid in starts}


def main() -> int:
    do_tsfm = "--tsfm" in sys.argv
    eval_only = "--eval-only" in sys.argv
    use_news = "--news" in sys.argv

    series, horizon = mc.parse_tsf(TSF)
    print(f"{DATASET}: {len(series)} series, h={horizon}")
    assert horizon == 24, horizon

    if eval_only:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        records = data["records"]
        if use_news:  # backfill origins into pre-news records
            origins = compute_origins(series)
            for r in records:
                r["origin_ym"] = origins.get(r["series_id"])
    else:
        origins = compute_origins(series)
        records = mc.generate_records(series, horizon, dataset=DATASET,
                                      origins=origins)
        print(f"classical records: {len(records)}")
        if do_tsfm:
            tsfm_data = mc.run_tsfm_quantiles(series, horizon, TSFM_JSON)
            records = mc.merge_tsfm(records, tsfm_data)
            print(f"records with TSFM: {len(records)}")
        # strip bulky fields before saving
        for r in records:
            r.pop("train_norm", None)
        RESULTS.write_text(json.dumps({
            "dataset": DATASET, "horizon": horizon,
            "records": records,
        }), encoding="utf-8")
        print(f"saved {RESULTS}")

    models = [m for m in mc.ALL_MODELS if all(m in r["crps"] for r in records)]
    print("models:", models)
    ladder = mc.evaluate_ladder(records, models)
    proposed = mc.run_proposed_cv(records, models, use_news=use_news,
                                  news_db=NEWS_DB if use_news else None)
    if use_news:
        print(f"news fire rate: {proposed['news_fire_rate']:.3f}")

    out = {
        "dataset": DATASET, "horizon": horizon, "n_series": len(records),
        "models": models,
        "ladder": ladder,
        "proposed": proposed,
    }
    # always persist eval (eval-only included); keeps records
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    data.update({k: v for k, v in out.items() if k != "records"})
    data["use_news"] = use_news
    RESULTS.write_text(json.dumps(data), encoding="utf-8")

    print("\n=== mean CRPS ===")
    for m in models:
        print(f"  {m:12s} {ladder['mean_crps'][m]:.4f}")
    print(f"  {'equal_weights':12s} {ladder['mean_crps']['equal_weights']:.4f}")
    print("\n=== proposed vs equal ===")
    print(" ", json.dumps(proposed["proposed_vs_equal"], indent=1))
    print("\n=== mean MASE ===")
    for m in models:
        print(f"  {m:12s} {ladder['mean_mase'][m]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
