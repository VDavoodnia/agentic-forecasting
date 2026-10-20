"""exp84 frame encoder: panel parts -> compact all-numeric analysis frame.

The string-heavy loan panel (8.89M rows) OOMs pandas on this shared
7GB VM, so rounds consume this compact frame instead: FRM 30-yr only,
categoricals as int codes, float32 numerics (~350MB in memory, 58MB
on disk).

Purpose codes follow the SFLLD spec: P=purchase, C=cash-out refi,
N=no-cash-out refi, other/missing=3. (An early throwaway encoder
mapped a nonexistent "R" code and lumped N into "other"; the R1 fits
from that pass are superseded — see exp84_report.md.)
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TD_ONE = Path(__file__).resolve().parents[1]

COLS = [
    "credit_score", "first_payment_date", "oltv_orig", "dti", "orig_upb",
    "orig_rate", "market_pmms", "spread_bps", "followup_months",
    "months_to_event", "event_type", "loan_purpose", "product_type",
    "orig_term",
]


def main() -> None:
    writer = None
    total = 0
    parts = sorted(glob.glob(str(TD_ONE / "data/freddie/exp84_parts/panel_*.parquet")))
    for pp in parts:
        t = pq.read_table(pp, columns=COLS)
        d = t.to_pandas(types_mapper=lambda x: pd.ArrowDtype(x))
        m = (
            (d["product_type"] == "FRM")
            & (d["orig_term"].between(350, 365))
            & d["spread_bps"].notna()
            & d["orig_rate"].notna()
        )
        d = d[m]
        out = pa.table(
            {
                "credit_score": pa.array(
                    d["credit_score"].to_numpy("float32", na_value=np.nan)
                ),
                "fp": pa.array(d["first_payment_date"].to_numpy("int32")),
                "oltv": pa.array(d["oltv_orig"].to_numpy("float32", na_value=np.nan)),
                "dti": pa.array(d["dti"].to_numpy("float32", na_value=np.nan)),
                "log_upb": pa.array(
                    np.log(d["orig_upb"].clip(lower=1).to_numpy("float64")).astype("float32")
                ),
                "orig_rate": pa.array(d["orig_rate"].to_numpy("float32")),
                "market_pmms": pa.array(d["market_pmms"].to_numpy("float32")),
                "spread_bps": pa.array(d["spread_bps"].to_numpy("float32")),
                "followup": pa.array(d["followup_months"].to_numpy("int16")),
                "months_to_event": pa.array(
                    d["months_to_event"].to_numpy("float32", na_value=np.nan)
                ),
                "event_code": pa.array(
                    d["event_type"]
                    .map({"active": 0, "prepay": 1, "other_term": 2})
                    .to_numpy("int8")
                ),
                "purpose_code": pa.array(
                    d["loan_purpose"]
                    .map({"P": 0, "C": 1, "N": 2})
                    .fillna(3)
                    .to_numpy("int8")
                ),
            }
        )
        total += out.num_rows
        if writer is None:
            writer = pq.ParquetWriter(
                TD_ONE / "data" / "freddie" / "exp84_frame.parquet", out.schema
            )
        writer.write_table(out)
    writer.close()
    print(f"compact frame rows: {total:,}")


if __name__ == "__main__":
    main()
