"""exp83: Perplexity backfill of Macro Signal cells for 2021-01..2023-09.

Step 1 of exp83 found the categorized corpus effectively EMPTY for the
stormy window (exactly 1 categorized article: inflation 2021-05; the
existing-articles extraction pass yielded exactly 1 signal). This driver
therefore fetches the empty cells via the established Perplexity path
(``scripts/fetch_macro_signal_news.py`` machinery: month-named queries,
in-month pub_date filter, fetch_log resume), restricted to the 7 live
categories (exp70 gate set) and a HARD search-call budget.

Budget discipline: the exp70-MS run spent ~$8.70 over ~906 Perplexity
searches (~$0.0096/search, observed). exp83's cap is ~$3 => hard stop at
300 searches (~$2.88 estimated). Every search is counted and logged to
experiments/exp83_fetch_log.json with the running estimate.

Priority (spend allocation only; no feature ever reads this ordering):
inflation first (all 33 months — the one live wire from exp76's S8
diagnosis), then the other live-6 categories, months ordered by
|MoM log change| of the Food CPI benchmark series, most volatile first.

Usage:
    .venv/bin/python experiments/exp83_fetch_backfill.py [--search-cap 300]
        [--only-cat inflation] [--only-month 2022-06] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_macro_signal_news import (  # noqa: E402
    fetch_cell,
    init_article_categories,
    init_fetch_log,
    log_fetch,
    store,
    was_fetched,
)
from td_one.news import init_db  # noqa: E402
from td_one.news.perplexity import PerplexityClient  # noqa: E402

LIVE_CATS = [
    "inflation",
    "services_costs",
    "food_prices",
    "labour",
    "fx",
    "housing",
    "wars_geopolitics",
]
OTHER_LIVE = [c for c in LIVE_CATS if c != "inflation"]

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "news" / "news_2026.db"
LOG_PATH = Path(__file__).resolve().parent / "exp83_fetch_log.json"
COST_PER_SEARCH = 0.0096  # observed: ~$8.70 / ~906 searches, exp70-MS run


def window_months() -> list[str]:
    out = []
    y, m = 2021, 1
    while (y, m) <= (2023, 9):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_volatility() -> dict[str, float]:
    """|MoM log change| of the Food CPI series per month (fetch order only)."""
    path = ROOT / "benchmarks" / "food_cpi" / "data.csv"
    rows = list(csv.DictReader(path.open()))
    vals = {r["date"][:7]: float(r["value"]) for r in rows}
    months = sorted(vals)
    out: dict[str, float] = {}
    for i in range(1, len(months)):
        out[months[i]] = abs(math.log(vals[months[i]] / vals[months[i - 1]]))
    return out


class CountingClient:
    """Wraps PerplexityClient.search to count (and log) every search call."""

    def __init__(self, inner: PerplexityClient, cap: int, log: list[dict]) -> None:
        self.inner = inner
        self.cap = cap
        self.log = log
        self.n = 0
        self.cell: tuple[str, str] = ("", "")

    def search(self, query: str, before, limit: int = 10, after=None):  # noqa: ANN001, ANN202
        if self.n >= self.cap:
            raise RuntimeError("exp83 search cap reached")
        self.n += 1
        rec = {
            "n": self.n,
            "category": self.cell[0],
            "month": self.cell[1],
            "est_cost_usd": round(self.n * COST_PER_SEARCH, 4),
        }
        self.log.append(rec)
        LOG_PATH.write_text(json.dumps({"searches": self.log}, indent=1))
        return self.inner.search(query, before=before, limit=limit, after=after)


def prioritized_cells(months: list[str], vol: dict[str, float]) -> list[tuple[str, str]]:
    """(category, month) cells in spend-priority order."""
    by_vol = sorted(months, key=lambda ym: -vol.get(ym, 0.0))
    cells: list[tuple[str, str]] = [("inflation", ym) for ym in by_vol]
    for ym in by_vol:
        for cat in OTHER_LIVE:
            cells.append((cat, ym))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-cap", type=int, default=300)
    ap.add_argument("--only-cat", default=None)
    ap.add_argument("--only-month", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    months = window_months()
    vol = month_volatility()
    cells = prioritized_cells(months, vol)
    if args.only_cat:
        cells = [c for c in cells if c[0] == args.only_cat]
    if args.only_month:
        cells = [c for c in cells if c[1] == args.only_month]

    init_db(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    init_fetch_log(con)
    init_article_categories(con)
    todo = [c for c in cells if not was_fetched(con, c[1], c[0])]
    print(
        f"exp83 fetch: {len(cells)} cells in scope, {len(todo)} todo "
        f"({len(cells) - len(todo)} already logged), search cap {args.search_cap} "
        f"(~${args.search_cap * COST_PER_SEARCH:.2f} est.)",
        flush=True,
    )
    if args.dry_run:
        for cat, ym in todo[:12]:
            print(f"  WOULD fetch {cat} {ym} (vol {vol.get(ym, 0):.4f})")
        return

    log: list[dict] = []
    if LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text()).get("searches", [])
    client = CountingClient(PerplexityClient(preset="fast"), args.search_cap, log)
    client.n = len(log)
    total_new = 0
    done_cells = 0
    coverage: dict[str, list[int]] = {}
    for cat, ym in todo:
        if client.n >= args.search_cap:
            print(f"search cap reached ({client.n}); stopping — rerun to resume", flush=True)
            break
        client.cell = (cat, ym)
        month_idx = months.index(ym)
        arts = None
        for attempt in range(3):
            try:
                arts = fetch_cell(client, cat, ym, month_idx)  # type: ignore[arg-type]
                break
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                if "429" in str(e):
                    wait = 60 * (attempt + 1)
                    print(f"  429 {cat} {ym}: backing off {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  !! {cat} {ym}: {e}", flush=True)
                break
        if arts is None:
            print(f"  !! {cat} {ym}: skipped (not logged)", flush=True)
            continue
        n = store(con, arts, cat)
        log_fetch(con, ym, cat, n)
        total_new += n
        done_cells += 1
        coverage.setdefault(cat, []).append(len(arts))
        if done_cells % 20 == 0:
            print(
                f"  [{done_cells} cells] {total_new} new articles, {client.n} searches, "
                f"~${client.n * COST_PER_SEARCH:.2f}",
                flush=True,
            )
        time.sleep(2)

    con.close()
    print(
        f"\nDone. {done_cells} cells fetched, {total_new} new articles, "
        f"{client.n} searches total, est. spend ${client.n * COST_PER_SEARCH:.2f}."
    )
    print("Coverage (in-month articles per fetched cell):")
    for cat in LIVE_CATS:
        vals = coverage.get(cat, [])
        if vals:
            zeros = sum(1 for v in vals if v == 0)
            print(
                f"  {cat:20s} cells={len(vals):3d} mean={sum(vals) / len(vals):5.1f} "
                f"zero_cells={zeros}"
            )


if __name__ == "__main__":
    main()
