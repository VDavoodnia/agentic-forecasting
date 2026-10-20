"""exp83: extract macro signals for the 2021-01..2023-09 stormy backfill.

Wraps ``td_one.news.macro_signal.extract_pending`` for the backfill window
and the 7 live categories only (exp70 calibration gate set). Two modes:

  --existing-first : require_fetched=False — squeeze whatever categorized
                     articles the corpus already holds (Step 1 of exp83:
                     no Perplexity spend before this pass).
  (default)        : require_fetched=True — only cells the exp83 fetcher
                     logged complete.

Extraction LLM: Vector proxy Gemini Flash-Lite (no Perplexity credits).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: E402
from td_one.news.macro_signal import extract_pending  # noqa: E402

LIVE_CATS = [
    "inflation",
    "services_costs",
    "food_prices",
    "labour",
    "fx",
    "housing",
    "wars_geopolitics",
]

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "news" / "news_2026.db"


def window_months() -> list[str]:
    out = []
    y, m = 2021, 1
    while (y, m) <= (2023, 9):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing-first", action="store_true")
    ap.add_argument("--model", default="flash-lite")
    args = ap.parse_args()

    months = window_months()
    require_fetched = not args.existing_first
    print(
        f"exp83 extract: {len(months)} months x {len(LIVE_CATS)} live cats, "
        f"require_fetched={require_fetched}",
        flush=True,
    )
    client = VectorProxyClient(model=args.model, call_budget=400)
    counts = extract_pending(
        str(DB_PATH), client, months, LIVE_CATS, require_fetched=require_fetched
    )
    print(f"extracted this pass: {counts}", flush=True)
    print(f"vector calls used: {client._calls}", flush=True)
    con = sqlite3.connect(DB_PATH)
    n = con.execute(
        "SELECT COUNT(*) FROM macro_signals WHERE month BETWEEN '2021-01' AND '2023-09'"
    ).fetchone()[0]
    per_cat = con.execute(
        "SELECT category, COUNT(*) FROM macro_signals "
        "WHERE month BETWEEN '2021-01' AND '2023-09' GROUP BY category ORDER BY category"
    ).fetchall()
    con.close()
    print(f"macro_signals in window now: {n}; per category: {per_cat}", flush=True)


if __name__ == "__main__":
    main()
