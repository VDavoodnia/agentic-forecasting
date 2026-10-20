"""Fetch competitor mortgage rates via Perplexity for refi-pressure analysis.

The refi-spread trigger (competitor rate - borrower's current rate) is a
core DGP driver in SimulatorV2 that we currently simulate. This pulls REAL
major-bank posted rates (5yr/3yr fixed, variable) and broker/monoline
discount rates, monthly, back to --start (default 2022-01).

Two phrasings per month: P1 = Big-6 posted rates; P2 = broker + monoline
(MCAP / First National / Merix / RMG) discount rates. Rows are stored with
source='perplexity' alongside the BoC posted backbone (source='boc-valet')
and Wayback best rates (source='wayback-ratehub').

Storage: competitor_rates table in data/news/news_2026.db
(schema: month, lender, product, rate_pct, source_url, fetched_at, source).
PIT-safe by construction: each row is as-of its month; experiments must
filter month <= origin.

Usage:
    .venv/bin/python scripts/fetch_competitor_rates.py [--start 2022-01]

Credits: two Perplexity calls per month (one per phrasing) plus a retry
phrasing for months that come back empty. Resume-safe via comp_rates_log
(month, phrasing).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.news.perplexity import PerplexityClient

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

LENDERS = ["TD Bank", "RBC", "BMO", "Scotiabank", "CIBC"]
# P1: Big-6 posted rates. P2: broker + monoline discount rates.
PRODUCTS_BANK = ["5yr_fixed_posted", "posted_3yr_fixed", "variable_posted"]
PRODUCTS_BROKER = [
    "broker_discount_5yr", "broker_discount_3yr", "variable_discount",
    "monoline_5yr_fixed",
]


def _ensure_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS competitor_rates (
            month TEXT NOT NULL,
            lender TEXT NOT NULL,
            product TEXT NOT NULL,
            rate_pct REAL NOT NULL,
            source_url TEXT DEFAULT '',
            fetched_at TEXT DEFAULT (datetime('now')),
            source TEXT NOT NULL DEFAULT 'perplexity',
            PRIMARY KEY (month, lender, product)
        )"""
    )
    cols = {r[1] for r in con.execute("PRAGMA table_info(competitor_rates)")}
    if "source" not in cols:
        con.execute(
            "ALTER TABLE competitor_rates ADD COLUMN source TEXT "
            "NOT NULL DEFAULT 'perplexity'"
        )
    # (month, phrasing) resume log. Migrate legacy single-phrasing rows.
    cols = {r[1] for r in con.execute("PRAGMA table_info(comp_rates_log)")}
    if "phrasing" not in cols:
        con.execute("ALTER TABLE comp_rates_log RENAME TO comp_rates_log_old")
        con.execute(
            """CREATE TABLE comp_rates_log (
                month TEXT NOT NULL,
                phrasing TEXT NOT NULL DEFAULT 'single',
                n_rates INTEGER,
                fetched_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (month, phrasing)
            )"""
        )
        con.execute(
            "INSERT INTO comp_rates_log (month, n_rates, fetched_at) "
            "SELECT month, n_rates, fetched_at FROM comp_rates_log_old"
        )
        con.execute("DROP TABLE comp_rates_log_old")


def _month_range(n: int) -> list[str]:
    today = datetime.now(UTC).date()
    out = []
    y, m = today.year, today.month - 1  # last complete month
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return sorted(out)


def _month_list(start: str) -> list[str]:
    """All months from ``start`` (YYYY-MM) up to the last complete month."""
    today = datetime.now(UTC).date()
    out = []
    y, m = int(start[:4]), int(start[5:7])
    while (y, m) < (today.year, today.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def parse_rates(text: str) -> list[dict]:
    """Parse a JSON array of rate objects from an LLM response.

    Tolerant of fence wrapping and surrounding prose. Each object must
    have lender, product, and rate_pct (or rate) keys.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        lender = str(it.get("lender", "")).strip()
        product = str(it.get("product", "")).strip()
        rate = it.get("rate_pct", it.get("rate"))
        try:
            rate_f = float(rate)
        except (TypeError, ValueError):
            continue
        if lender and product and 0.5 <= rate_f <= 15.0:
            out.append(
                {"lender": lender, "product": product, "rate_pct": rate_f,
                 "source_url": str(it.get("source_url", ""))[:500]}
            )
    return out


def _next_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    m += 1
    if m == 13:
        m, y = 1, y + 1
    return f"{y:04d}-{m:02d}"


def fetch_month_bank(client: PerplexityClient, month: str) -> list[dict]:
    """P1: Big-6 posted rates for one month (one Perplexity call)."""
    next_m = _next_month(month)
    prompt = (
        f"What were the posted mortgage rates in Canada in {month} "
        f"(published before {next_m}-01)? For each of these lenders: "
        f"{', '.join(LENDERS)} — return a JSON array of objects with keys: "
        f"lender, product (one of: 5yr_fixed_posted, posted_3yr_fixed, "
        f"variable_posted), rate_pct (number), source_url. "
        f"Return ONLY the JSON array."
    )
    data = client._call(prompt)
    return parse_rates(PerplexityClient._extract_text(data))


def fetch_month_broker(client: PerplexityClient, month: str) -> list[dict]:
    """P2: broker + monoline discount rates for one month (one call)."""
    next_m = _next_month(month)
    prompt = (
        f"What were the best mortgage broker discount rates in Canada in "
        f"{month} (published before {next_m}-01)? Include: the best broker "
        f"discount rate for 5-year fixed, for 3-year fixed, and for "
        f"5-year variable (discount to prime); plus the best 5-year fixed "
        f"rates offered by monoline lenders MCAP, First National, Merix "
        f"and RMG where reported. Return a JSON array of objects with "
        f"keys: lender (use 'Broker' for broker-channel bests, or the "
        f"monoline lender name), product (one of: broker_discount_5yr, "
        f"broker_discount_3yr, variable_discount, monoline_5yr_fixed), "
        f"rate_pct (number), source_url. Return ONLY the JSON array."
    )
    data = client._call(prompt)
    return parse_rates(PerplexityClient._extract_text(data))


def fetch_month_retry(client: PerplexityClient, month: str) -> list[dict]:
    """Alternate phrasing for months that came back empty."""
    next_m = _next_month(month)
    prompt = (
        f"In {month}, what was the lowest advertised 5-year fixed mortgage "
        f"rate in Canada (published before {next_m}-01)? Check rate "
        f"comparison sites like Ratehub, rates.ca, WowA, NerdWallet Canada. "
        f"Return a JSON array with one object: lender 'Broker', product "
        f"'broker_discount_5yr', rate_pct, source_url. Return ONLY JSON."
    )
    data = client._call(prompt)
    return parse_rates(PerplexityClient._extract_text(data))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--months", type=int, default=None,
                    help="limit to N most recent months (default: all)")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    _ensure_tables(con)
    done = {(r[0], r[1]) for r in con.execute(
        "SELECT month, phrasing FROM comp_rates_log")}
    months = _month_list(args.start)
    if args.months:
        months = months[-args.months:]
    client = PerplexityClient(preset="fast")
    calls, total = 0, 0

    def store(month: str, phrasing: str, rates: list[dict]) -> None:
        nonlocal total
        for r in rates:
            con.execute(
                "INSERT OR REPLACE INTO competitor_rates "
                "(month, lender, product, rate_pct, source_url, source) "
                "VALUES (?,?,?,?,?,?)",
                (month, r["lender"], r["product"], r["rate_pct"],
                 r["source_url"], "perplexity"),
            )
        con.execute(
            "INSERT OR REPLACE INTO comp_rates_log (month, phrasing, n_rates) "
            "VALUES (?,?,?)",
            (month, phrasing, len(rates)),
        )
        con.commit()
        total += len(rates)

    for month in months:
        for phrasing, fn in (("bank", fetch_month_bank),
                             ("broker", fetch_month_broker)):
            if (month, phrasing) in done:
                continue
            try:
                rates = fn(client, month)
                calls += 1
            except Exception as e:  # noqa: BLE001 — resume-safe
                print(f"FAILED {month}/{phrasing}: {e}")
                continue
            store(month, phrasing, rates)
            print(f"{month}/{phrasing}: {len(rates)} rates")
        # Retry phrasing for months with no broker data at all.
        n_broker = con.execute(
            "SELECT COUNT(*) FROM competitor_rates WHERE month=? "
            "AND product LIKE 'broker%'", (month,)).fetchone()[0]
        if n_broker == 0 and (month, "retry") not in done:
            try:
                rates = fetch_month_retry(client, month)
                calls += 1
            except Exception as e:  # noqa: BLE001 — resume-safe
                print(f"FAILED {month}/retry: {e}")
                continue
            store(month, "retry", rates)
            print(f"{month}/retry: {len(rates)} rates")
    con.close()
    print(f"done: {total} new rates cached, {calls} Perplexity calls")


if __name__ == "__main__":
    main()
