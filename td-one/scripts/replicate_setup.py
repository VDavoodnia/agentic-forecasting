#!/usr/bin/env python3
"""Replicate the full TD-One data/model setup on a fresh checkout.

Runs, in order, every *automatic* step that rebuilds TD-One's local
data state, and prints the *manual* steps that need Vandad's accounts,
paid credit, or his shared datasets (class S). Idempotent: a step whose
sentinel output already exists is skipped unless ``--force`` is given.

    cd td-one
    .venv/bin/python scripts/replicate_setup.py            # full setup
    .venv/bin/python scripts/replicate_setup.py --list     # asset map only

The master asset table lives in ``docs/DATA_REPLICATION.md``; this
script is its executable half. Produced artifacts (class P) come back
from the committed multipart-zip snapshots via
``scripts/restore_dist.py``; downloaded assets (class D) are re-fetched
from their public sources; shared assets (class S) are never fetched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class Step(TypedDict):
    """One automatic setup step."""

    name: str
    sentinel: Path  # step is skipped when this exists (unless --force)
    command: list[str]
    note: str


AUTO_STEPS: list[Step] = [
    {
        "name": "Restore news store snapshot (class P)",
        "sentinel": ROOT / "data/news/news_2026.db",
        "command": [PY, "scripts/restore_dist.py", "data/news_dist", "--target", "data/news"],
        "note": "21,234-article SQLite store + embeddings from data/news_dist/.",
    },
    {
        "name": "Restore exp84 analysis population (class P)",
        "sentinel": ROOT / "data/freddie/exp84_panel.parquet",
        "command": [PY, "scripts/restore_dist.py", "data/freddie_exp84_dist", "--target", "data/freddie"],
        "note": "exp84 panel/frame/horizon parquets from data/freddie_exp84_dist/.",
    },
    {
        "name": "Rebuild scale-up PIT store (class P)",
        "sentinel": ROOT / "data/scale_up/pit.duckdb",
        "command": [PY, "scripts/scale_up.py"],
        "note": "Seeded generator (defaults: seed 0, 10,000 customers, 66 months) regenerates the simulated histories/outcomes + DuckDB PIT store. Not archived in git (decision 2026-10-20: script-rebuildable artifacts are rebuilt, not snapshotted).",
    },
    {
        "name": "Bootstrap TSFM weights (class D)",
        "sentinel": ROOT / "models/weights/chronos2",
        "command": [PY, "scripts/bootstrap_models.py"],
        "note": "Chronos-2 + TimesFM-2.5 at the revisions pinned in the script and models/MANIFEST.json. Needs HF_TOKEN for gated repos.",
    },
    {
        "name": "Fetch FRED series (class D)",
        "sentinel": ROOT / "data/fred/MORTGAGE30US.parquet",
        "command": [PY, "scripts/fetch_fred.py"],
        "note": "17 curated series (incl. MORTGAGE30US, CPIUFDSL) via the vault FRED client.",
    },
    {
        "name": "Pull OSFI M4 (class D)",
        "sentinel": ROOT / "data/osfi/m4_balances.parquet",
        "command": [PY, "scripts/pull_osfi_m4.py"],
        "note": "StatCan table 10100006 -> balances + runoff parquets.",
    },
    {
        "name": "Pull BoC posted rates (class D)",
        "sentinel": ROOT / "data/news/news_2026.db",
        "command": [PY, "scripts/pull_boc_posted_rates.py"],
        "note": "BoC Valet posted rates, 1975-2026 (622 months). Sentinel is shared with the news store; use --force to re-pull on its own.",
    },
    {
        "name": "Fetch Monash/M4 benchmarks (class D)",
        "sentinel": ROOT / "data/m4/Monthly-train.csv",
        "command": [PY, "scripts/fetch_monash_benchmarks.py"],
        "note": "Tourism + M3 (Zenodo) and M4 (Mcompetitions GitHub), size-pinned.",
    },
]

MANUAL_STEPS: list[str] = [
    "News corpus rebuild (approximate only; the snapshot above is the copy of record): "
    "scripts/fetch_news_gdelt.py (free; GDELT persistently throttles this IP), "
    "scripts/fetch_news_perplexity.py (PAID - needs Perplexity credit via the vault connector), "
    "scripts/fetch_article_bodies.py, scripts/fetch_news_gdelt_gapfill.py.",
    "Macro Signal Service: scripts/fetch_macro_signal_news.py (Perplexity, paid) then "
    "scripts/run_macro_signal_extraction.py (Vector-proxy Gemini). The extracted signals "
    "already live inside the archived news DB.",
    "Competitor rates: scripts/fetch_competitor_rates.py and scripts/fetch_wayback_rates.py "
    "(Wayback Machine RateHub snapshots) refresh the competitor_rates table in the news DB.",
    "Freddie Mac SMM monthly series (data/freddie/smm_monthly_2021_2026.csv, committed): derived "
    "from Vandad's SFLLD raw (class S) via the src/td_one/freddie/ loader. Cannot be re-fetched "
    "by the agent; requires his data.",
    "exp84 rebuild from scratch: experiments/exp84_build_panel.py + exp84_make_frame.py + "
    "exp84_make_samples.py, all requiring data/freddie/raw/ (class S). The archived snapshot "
    "above makes this unnecessary for analysis.",
    "Food CPI benchmark CSV (benchmarks/food_cpi/data.csv, committed): rebuilt 1:1 from FRED "
    "CPIUFDSL (scripts/fetch_fred.py) if ever lost.",
]


def print_asset_map() -> None:
    """Print the automatic + manual step map without running anything."""
    print("AUTOMATIC STEPS (run in order; sentinels skip finished steps):")
    for i, step in enumerate(AUTO_STEPS, 1):
        state = "present" if step["sentinel"].exists() else "missing"
        print(f"  {i}. {step['name']} [{state}]\n     {step['note']}")
    print("\nMANUAL STEPS (accounts / paid credit / class-S data required):")
    for i, text in enumerate(MANUAL_STEPS, 1):
        print(f"  {i}. {text}")
    print("\nFull table: docs/DATA_REPLICATION.md")


def main(argv: list[str] | None = None) -> int:
    """Run the automatic setup sequence. Returns exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the asset map and exit")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if sentinels exist")
    args = parser.parse_args(argv)

    if args.list:
        print_asset_map()
        return 0

    for step in AUTO_STEPS:
        if step["sentinel"].exists() and not args.force:
            print(f"SKIP (present): {step['name']}")
            continue
        print(f"RUN: {step['name']}")
        result = subprocess.run(step["command"], cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED: {step['name']} (exit {result.returncode}). Fix and re-run; finished steps are skipped.")
            return result.returncode
    print("\nAutomatic setup complete. Remaining manual steps:")
    for i, text in enumerate(MANUAL_STEPS, 1):
        print(f"  {i}. {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
