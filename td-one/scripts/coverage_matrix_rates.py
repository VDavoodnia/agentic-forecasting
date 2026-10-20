"""Print the month x source coverage matrix for competitor_rates.

Usage: .venv/bin/python scripts/coverage_matrix_rates.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

# (label, source tag, product predicate)
SOURCES = [
    ("BoC-posted", "boc-valet", "posted_5yr_conventional"),
    ("bank-posted", "perplexity", "5yr_fixed_posted"),
    ("bank-3yr", "perplexity", "posted_3yr_fixed"),
    ("broker-5yr-P", "perplexity", "broker_discount_5yr"),
    ("broker-3yr-P", "perplexity", "broker_discount_3yr"),
    ("var-disc-P", "perplexity", "variable_discount"),
    ("monoline-P", "perplexity", "monoline_5yr_fixed"),
    ("broker-5yr-W", "wayback-ratehub", "broker_discount_5yr"),
    ("broker-3yr-W", "wayback-ratehub", "broker_discount_3yr"),
    ("var-disc-W", "wayback-ratehub", "variable_discount"),
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-month", default="2022-01",
                    help="first month row to print (default 2022-01; "
                         "BoC backbone extends to 1975)")
    ap.add_argument("--to-month", default=None)
    args = ap.parse_args()
    con = sqlite3.connect(DB_PATH)
    months = [r[0] for r in con.execute(
        "SELECT DISTINCT month FROM competitor_rates ORDER BY month")]
    months = [m for m in months if m >= args.from_month]
    if args.to_month:
        months = [m for m in months if m <= args.to_month]
    print(f"coverage rows: {len(months)} months "
          f"({months[0]} -> {months[-1]})")
    header = "month   " + "".join(f" {s[0]:>10}" for s in SOURCES)
    print(header)
    filled = {s[0]: 0 for s in SOURCES}
    any_broker = 0
    for m in months:
        cells = []
        for label, src, prod in SOURCES:
            n = con.execute(
                "SELECT COUNT(*) FROM competitor_rates WHERE month=? "
                "AND source=? AND product=?", (m, src, prod)).fetchone()[0]
            cells.append("X" if n else ".")
            if n:
                filled[label] += 1
        if any(c == "X" for c in cells[3:]):
            any_broker += 1
        print(f"{m}  " + "".join(f" {c:>10}" for c in cells))
    print("\nfilled months per source:")
    for label, _src, _prod in SOURCES:
        print(f"  {label:>12}: {filled[label]}")
    print(f"\nmonths with ANY broker/discount data: {any_broker}")
    # Broker 5yr coverage (the exp71 spread series), source-combined.
    con.close()


if __name__ == "__main__":
    main()
