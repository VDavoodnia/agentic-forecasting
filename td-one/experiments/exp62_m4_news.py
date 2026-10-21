"""Experiment 62: M4-100 with the news module ON (exp44, gated).

Reuses exp47's M4-100 records (f42.generate_records, 100 series x 6 origins)
and attaches calendar origin months via monash_common.m4_origin_ym, then runs
the proposed CV with use_news=True against the news DB.

The news DB needs the 1979-2017 historical backfill for this to fire;
without it every origin gets coverage=0 and the news gate stays shut
(equivalent to the no-news run -- useful as a smoke test).

Usage:
    .venv/bin/python experiments/exp62_m4_news.py            # with news
    .venv/bin/python experiments/exp62_m4_news.py --no-news  # smoke test
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402
import monash_common as mc  # noqa: E402

NEWS_DB = ROOT / "data" / "news" / "news_2026.db"
RESULTS = ROOT / "experiments" / "exp62_results.json"


def load_starts() -> dict[str, str]:
    with open(ROOT / "data" / "m4" / "M4-info.csv", encoding="utf-8") as f:
        return {r["M4id"]: r["StartingDate"] for r in csv.DictReader(f)}


def main() -> int:
    use_news = "--no-news" not in sys.argv
    print("[exp62] building M4-100 records (exp47 protocol)...", flush=True)
    records = f42.generate_records(f42.load_m4(n=100), 48, 6, 6, "m4")
    print(f"[exp62] {len(records)} records", flush=True)

    starts = load_starts()
    missing = 0
    for r in records:
        sd = starts.get(r["series_id"])
        r["origin_ym"] = mc.m4_origin_ym(sd, r["origin"]) if sd else None
        if sd is None:
            missing += 1
    print(f"[exp62] origins mapped, {missing} missing dates", flush=True)
    assert missing == 0, "every M4 record needs a calendar origin for news"

    models = f42.MODELS
    out = mc.run_proposed_cv(records, models, use_news=use_news,
                             news_db=NEWS_DB if use_news else None)
    if use_news:
        print(f"[exp62] news fire rate: {out['news_fire_rate']:.3f}")

    RESULTS.write_text(json.dumps({
        "dataset": "m4_100", "n_series": 100, "use_news": use_news,
        "models": models, "proposed": out,
    }), encoding="utf-8")
    print(f"[exp62] saved {RESULTS}")

    print("\n=== proposed vs equal ===")
    print(" ", json.dumps(out["proposed_vs_equal"], indent=1))
    print("=== proposed vs best single ===")
    print(" ", json.dumps(out["proposed_vs_best"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
