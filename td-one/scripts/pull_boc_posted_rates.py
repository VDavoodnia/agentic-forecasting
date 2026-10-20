"""Pull the Bank of Canada's posted chartered-bank rates (the backbone).

The BoC publishes weekly (Wednesday) posted rates for the major chartered
banks (group CHARTERED_BANK_INTEREST, via its free Valet API — no key).
This is GROUND TRUTH, not LLM retrieval: conventional mortgage 1/3/5-year
and prime, weekly since 1975.

We aggregate weekly -> monthly using the LAST Wednesday in the month (the
rate as-published for that month; PIT-safe for experiments that filter
``month <= origin``).

Storage: the same ``competitor_rates`` table in data/news/news_2026.db as
the Perplexity-sourced rates, distinguished by ``source='boc-valet'`` and
lender='Bank of Canada'.

Usage:
    .venv/bin/python scripts/pull_boc_posted_rates.py
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
VALET_URL = (
    "https://www.bankofcanada.ca/valet/observations/"
    "V80691333,V80691334,V80691335,V80691311/json"
)
GROUP_URL = "https://www.bankofcanada.ca/rates/banking-and-financial-statistics/posted-interest-rates-offered-by-chartered-banks/"

SERIES_MAP = {
    "V80691333": "posted_1yr_conventional",
    "V80691334": "posted_3yr_conventional",
    "V80691335": "posted_5yr_conventional",
    "V80691311": "prime_rate",
}


def monthly_from_weekly(obs: list[dict]) -> list[dict]:
    """Aggregate weekly BoC observations to monthly (last Wednesday wins)."""
    out: dict[tuple[str, str], dict] = {}
    for o in obs:
        day = str(o["d"])[:10]
        month = day[:7]
        for sid, product in SERIES_MAP.items():
            cell = o.get(sid)
            if not cell or "v" not in cell:
                continue
            try:
                rate = float(cell["v"])
            except (TypeError, ValueError):
                continue
            key = (month, sid)
            # Observations arrive in date order; last write in a month wins.
            out[key] = {
                "month": month,
                "series": sid,
                "product": product,
                "rate_pct": rate,
                "asof_date": day,
            }
    return sorted(out.values(), key=lambda r: (r["month"], r["product"]))


def fetch_observations() -> list[dict]:
    req = urllib.request.Request(VALET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["observations"]


def main() -> None:
    obs = fetch_observations()
    rows = monthly_from_weekly(obs)
    con = sqlite3.connect(DB_PATH)
    # Schema migration: add the source tag column (idempotent).
    cols = {r[1] for r in con.execute("PRAGMA table_info(competitor_rates)")}
    if "source" not in cols:
        con.execute(
            "ALTER TABLE competitor_rates ADD COLUMN source TEXT "
            "NOT NULL DEFAULT 'perplexity'"
        )
    n = 0
    for r in rows:
        con.execute(
            "INSERT OR REPLACE INTO competitor_rates "
            "(month, lender, product, rate_pct, source_url, source) "
            "VALUES (?,?,?,?,?,?)",
            (r["month"], "Bank of Canada", r["product"], r["rate_pct"],
             GROUP_URL, "boc-valet"),
        )
        n += 1
    con.commit()
    months = sorted({r["month"] for r in rows})
    print(f"boc-valet backbone: {n} rows, {len(months)} months, "
          f"{months[0]} -> {months[-1]}")
    con.close()


if __name__ == "__main__":
    main()
