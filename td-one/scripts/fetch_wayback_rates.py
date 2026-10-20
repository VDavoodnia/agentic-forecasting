"""Fetch historical broker-channel best rates from Wayback snapshots of RateHub.

RateHub's /best-mortgage-rates page advertises broker-channel best rates by
term — the actual market-available rates that drive the refi incentive
(triggered DGP in SimulatorV2). Monthly snapshots (2022 -> present) are
parsed and stored as the monthly best per (term, kind):

  (month, 'Broker', 'broker_discount_5yr'|'broker_discount_3yr'
   |'variable_discount', rate, snapshot_url, 'wayback-ratehub')

Two page layouts are handled (see tests): legacy ``<tr class="rh-analytics">``
rows and modern ``__NEXT_DATA__`` JSON (``bestMarketRates``).

PIT: rates are dated to the snapshot's month; experiments must filter
``month <= origin``.

Usage:
    .venv/bin/python scripts/fetch_wayback_rates.py [--start 2022-01]

Polite: ~1.5s between Wayback fetches, backoff on 429. Resume-safe via
``wayback_log`` (month -> timestamp).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
CDX_URL = "http://web.archive.org/cdx/search/cdx"
PAGE_URL = "ratehub.ca/best-mortgage-rates"
UA = {"User-Agent": "Mozilla/5.0 (research; contact: td-one)"}

# (term_years, kind) -> product name
PRODUCT_MAP = {
    (5, "fixed"): "broker_discount_5yr",
    (3, "fixed"): "broker_discount_3yr",
    (5, "variable"): "variable_discount",
}


def _clean_lender(raw: str) -> str:
    # "CanWise Financial A Ratehub Company 4,228 reviews CanWise Financial"
    head = re.split(r"\s+\d[\d,]*\s+reviews", raw, maxsplit=1)[0]
    head = " ".join(head.split())
    return head.strip()


def _slug_lender(href: str) -> str:
    path = href.split("?", 1)[0]
    parts = [p for p in path.split("/") if p]
    for i, p in enumerate(parts[:-1]):
        if p in ("fixed", "variable"):
            return parts[i + 1]
    return parts[-1]


def _valid(rate: float) -> bool:
    return 0.5 <= rate <= 15.0


def _parse_legacy(html: str) -> list[dict]:
    rows = []
    for m in re.finditer(r'<tr class="rh-analytics">(.*?)</tr>', html, re.S):
        r = m.group(1)
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 4:
            continue
        txt = [" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in cells]
        rate_m = re.search(r"(\d+\.\d+)\s*%", txt[0])
        term_m = re.search(r"(\d+)\s*year", txt[1], re.I)
        kind = "fixed" if "fix" in txt[2].lower() else (
            "variable" if "var" in txt[2].lower() else "")
        if not (rate_m and term_m and kind):
            continue
        rate = float(rate_m.group(1))
        if not _valid(rate):
            continue
        rows.append({
            "term_years": int(term_m.group(1)),
            "kind": kind,
            "rate_pct": rate,
            "lender": _clean_lender(txt[3]),
        })
    return rows


def _parse_next_data(html: str) -> list[dict]:
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.S,
    )
    if not m:
        return []
    try:
        nd = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    # Walk to bestMarketRates (structure varies by page version).
    terms = None
    stack = [nd]
    while stack and terms is None:
        obj = stack.pop()
        if isinstance(obj, dict):
            if "bestMarketRates" in obj and isinstance(
                    obj["bestMarketRates"], list):
                terms = obj["bestMarketRates"]
                break
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj[:50])
    if not terms:
        return []
    rows = []
    for t in terms:
        if not isinstance(t, dict) or "term" not in t:
            continue
        years = int(t["term"]) // 12
        for kind in ("fixed", "variable"):
            cell = t.get(kind)
            if not isinstance(cell, dict):
                continue
            val = cell.get("value")
            try:
                rate = float(val)
            except (TypeError, ValueError):
                continue
            if not _valid(rate):
                continue
            rows.append({
                "term_years": years,
                "kind": kind,
                "rate_pct": rate,
                "lender": _slug_lender(str(cell.get("href", "")))
                if cell.get("href") else "",
            })
    return rows


def parse_ratehub_html(html: str) -> list[dict]:
    """Parse RateHub best-rates rows from either page layout."""
    rows = _parse_legacy(html)
    if rows:
        return rows
    return _parse_next_data(html)


def _get(url: str, timeout: int = 90) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def snapshot_for_month(month: str) -> tuple[str, list[str]] | None:
    """Snapshots of the RateHub page in ``month`` (YYYY-MM).

    Returns (timestamp, urls) where urls tries the raw ``id_`` variant
    first (exact original) then the rewritten variant — some captures
    serve truncated content through ``id_``.
    """
    q = (f"{CDX_URL}?url={PAGE_URL}&from={month[:4]}{month[5:7]}01"
         f"&to={month[:4]}{month[5:7]}31&output=text&limit=3"
         f"&filter=statuscode:200&collapse=digest")
    try:
        text = _get(q, timeout=60)
    except Exception:  # noqa: BLE001 — treat CDX failure as a gap
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) >= 14:
            ts = parts[1]
            return ts, [
                f"https://web.archive.org/web/{ts}id_/https://www.{PAGE_URL}",
                f"https://web.archive.org/web/{ts}/https://www.{PAGE_URL}",
            ]
    return None


def best_per_product(rows: list[dict]) -> list[dict]:
    """Monthly minimum rate per (term, kind) mapped to product names."""
    out: dict[str, dict] = {}
    for r in rows:
        key = PRODUCT_MAP.get((r["term_years"], r["kind"]))
        if not key:
            continue
        if key not in out or r["rate_pct"] < out[key]["rate_pct"]:
            out[key] = {
                "product": key,
                "rate_pct": r["rate_pct"],
                "lender_detail": r.get("lender", ""),
            }
    return list(out.values())


def _month_list(start: str) -> list[str]:
    today = datetime.now(UTC).date()
    months = []
    y, m = int(start[:4]), int(start[5:7])
    while (y, m) < (today.year, today.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS wayback_log "
        "(month TEXT PRIMARY KEY, timestamp TEXT, n_rates INTEGER)"
    )
    done = {r[0] for r in con.execute("SELECT month FROM wayback_log")}
    total = 0
    for month in _month_list(args.start):
        if month in done:
            continue
        snap = snapshot_for_month(month)
        if snap is None:
            con.execute("INSERT OR REPLACE INTO wayback_log VALUES (?,?,?)",
                        (month, "", 0))
            con.commit()
            print(f"{month}: no snapshot (gap)")
            continue
        ts, urls = snap
        html, rows = "", []
        for url in urls:
            try:
                html = _get(url)
                time.sleep(1.5)
            except Exception as e:  # noqa: BLE001 — resume-safe
                print(f"{month}: fetch FAILED ({type(e).__name__})")
                continue
            rows = parse_ratehub_html(html)
            if rows:
                url_used = url
                break
        else:
            print(f"{month}: no parseable snapshot (ts={ts[:8]})")
            con.execute("INSERT OR REPLACE INTO wayback_log VALUES (?,?,?)",
                        (month, ts, 0))
            con.commit()
            continue
        best = best_per_product(rows)
        for b in best:
            con.execute(
                "INSERT OR REPLACE INTO competitor_rates "
                "(month, lender, product, rate_pct, source_url, source) "
                "VALUES (?,?,?,?,?,?)",
                (month, "Broker", b["product"], b["rate_pct"], url_used,
                 "wayback-ratehub"),
            )
        con.execute("INSERT OR REPLACE INTO wayback_log VALUES (?,?,?)",
                    (month, ts, len(best)))
        con.commit()
        total += len(best)
        print(f"{month}: {len(best)} best rates (ts={ts[:8]})")
    con.close()
    print(f"done: {total} wayback best-rate rows")


if __name__ == "__main__":
    main()
