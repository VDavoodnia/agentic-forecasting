"""exp84 panel builder: loan-level Freddie SFLLD panel for the spread test.

Extracts, per vintage quarter (2021Q1..2026Q1):
  - origination rows (via the repo loader's schema), and
  - the first zero-balance (termination) event per loan from the
    multi-GB performance files. Perf files are streamed with awk —
    only rows with a non-empty zb_code (field 9) survive, so the
    extract is ~1% of the raw bytes.

Derived fields:
  - market_pmms: Freddie PMMS 30-yr FRM monthly mean (FRED MORTGAGE30US)
    in the month BEFORE first payment (closing-month proxy).
  - spread_bps: (orig_rate - market_pmms) * 100, knowable at origination.
  - months_to_event / event_type from the first zb row.
  - followup_months: months from first payment to the panel end
    (2026-03, the last period in every vintage's perf file).

Output: data/freddie/exp84_panel.parquet (gitignored data dir) +
exp84_inventory.json next to this script's results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.freddie import load_orig  # noqa: E402

TD_ONE = Path(__file__).resolve().parents[1]
RAW = TD_ONE / "data" / "freddie" / "raw"
PANEL_END = 202603  # last reporting period present in every perf file
VINTAGES = [f"{y}Q{q}" for y in range(2021, 2027) for q in range(1, 5)]
VINTAGES = [v for v in VINTAGES if v <= "2026Q1"]

ORIG_KEEP = [
    "loan_seq",
    "credit_score",
    "first_payment_date",
    "oltv",  # combined LTV
    "oltv_orig",  # original LTV
    "dti",
    "orig_upb",
    "orig_rate",
    "channel",
    "product_type",
    "occupancy",
    "loan_purpose",
    "orig_term",
    "num_borrowers",
    "harp_indicator",
    "super_conforming",
    "mi_pct",
]


def yyyymm_add(yyyymm: pd.Series, delta: int) -> pd.Series:
    """Add delta months to a YYYYMM int series."""
    y = yyyymm // 100
    m = yyyymm % 100
    idx = y * 12 + (m - 1) + delta
    return (idx // 12) * 100 + (idx % 12) + 1


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole months from start YYYYMM to end YYYYMM."""
    return (end // 100 - start // 100) * 12 + (end % 100 - start % 100)


def extract_terminations(vintage: str) -> pd.DataFrame:
    """First zb event per loan from perf_<vintage>.txt via awk stream."""
    perf = RAW / f"perf_{vintage}.txt"
    # cut (C speed) takes fields 1,2,9,10; awk keeps zb_code != "" rows.
    cmd = (
        f"cut -d'|' -f1,2,9,10 '{perf}' | awk -F'|' '$3!=\"\"'"
    )
    out = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=True
    ).stdout
    df = pd.read_csv(
        pd.io.common.StringIO(out),
        sep="|",
        header=None,
        names=["loan_seq", "period", "zb_code", "zb_date"],
        dtype={"loan_seq": str, "zb_code": str},
    )
    df["zb_date"] = pd.to_numeric(df["zb_date"], errors="coerce")
    df = df.dropna(subset=["zb_date"])
    df["zb_date"] = df["zb_date"].astype(int)
    # First event per loan = min zb_date.
    df = df.sort_values("zb_date").drop_duplicates("loan_seq", keep="first")
    return df[["loan_seq", "zb_code", "zb_date"]]


def build_vintage(vintage: str, pmms: pd.Series) -> pd.DataFrame:
    orig = load_orig(RAW / f"orig_{vintage}.txt")
    orig = orig[ORIG_KEEP].copy()
    # Loader leaves first_payment_date as str; the panel needs YYYYMM ints.
    orig["first_payment_date"] = pd.to_numeric(
        orig["first_payment_date"], errors="coerce"
    ).astype("Int64")
    term = extract_terminations(vintage)
    df = orig.merge(term, on="loan_seq", how="left")
    df["vintage"] = vintage
    # Market rate at origination: PMMS of the month before first payment.
    df["pmms_month"] = yyyymm_add(df["first_payment_date"], -1)
    df["market_pmms"] = df["pmms_month"].map(pmms)
    df["spread_bps"] = (df["orig_rate"] - df["market_pmms"]) * 100.0
    df["followup_months"] = months_between(
        df["first_payment_date"], pd.Series(PANEL_END, index=df.index)
    )
    has_event = df["zb_date"].notna()
    df["months_to_event"] = np.where(
        has_event,
        months_between(
            df["first_payment_date"], df["zb_date"].fillna(PANEL_END).astype(int)
        ),
        np.nan,
    )
    df["event_type"] = np.where(
        df["zb_code"] == "01",
        "prepay",
        np.where(has_event, "other_term", "active"),
    )
    return df


def main() -> None:
    # Optional argv: vintage range like 2022Q1:2024Q4 (inclusive) to
    # bound peak memory when sibling jobs share the VM.
    if len(sys.argv) > 1 and ":" in sys.argv[1]:
        lo, hi = sys.argv[1].split(":")
        globals()["VINTAGES"] = [v for v in VINTAGES if lo <= v <= hi]
    pmms_raw = pd.read_parquet(TD_ONE / "data" / "fred" / "MORTGAGE30US.parquet")
    pmms_raw["date"] = pd.to_datetime(pmms_raw["date"])
    monthly = pmms_raw.set_index("date")["value"].resample("MS").mean()
    pmms = pd.Series(
        monthly.values,
        index=[d.year * 100 + d.month for d in monthly.index],
        name="pmms",
    )

    frames = []
    inventory: dict[str, object] = {"vintages": {}, "panel_end": PANEL_END}
    parts_dir = TD_ONE / "data" / "freddie" / "exp84_parts"
    parts_dir.mkdir(exist_ok=True)
    for v in VINTAGES:
        part_path = parts_dir / f"panel_{v}.parquet"
        if part_path.exists():
            # Restart-resilient: per-vintage parts persist on disk.
            df = pd.read_parquet(part_path)
            print(f"{v}: cached ({len(df):,} loans)", flush=True)
        else:
            df = build_vintage(v, pmms)
            df.to_parquet(part_path)
        frames.append(df)
        inventory["vintages"][v] = {
            "n_loans": int(len(df)),
            "n_prepay_events": int((df["event_type"] == "prepay").sum()),
            "n_other_term": int((df["event_type"] == "other_term").sum()),
            "zb_code_counts": {
                str(k): int(n)
                for k, n in df.loc[df["zb_code"].notna(), "zb_code"]
                .value_counts()
                .items()
            },
        }
        print(f"{v}: {len(df):,} loans", flush=True)

    # Assemble from ALL parts on disk (chunk runs may build a subset).
    all_parts = sorted(parts_dir.glob("panel_*.parquet"))
    panel = pd.concat(
        [pd.read_parquet(pp) for pp in all_parts], ignore_index=True
    )
    out_path = TD_ONE / "data" / "freddie" / "exp84_panel.parquet"
    panel.to_parquet(out_path)
    inventory["total_loans"] = int(len(panel))
    inventory["columns"] = list(panel.columns)
    inventory["parts_present"] = [pp.stem.replace("panel_", "") for pp in all_parts]
    inv_path = Path(__file__).with_name("exp84_inventory.json")
    inv_path.write_text(json.dumps(inventory, indent=1))
    print(f"panel: {len(panel):,} loans -> {out_path}")


if __name__ == "__main__":
    main()
