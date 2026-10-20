"""exp84 sampler: lean per-horizon analysis samples from the compact frame.

The full-panel rounds OOM on this shared 7GB VM (sibling jobs hold
~4-5GB). Random samples preserve rates/slopes: 400k/era at H=24 and
250k/era at H=12/H=36 give slope SEs of ~0.01-0.02 — the population
fits in exp84_results.json (R1) already used 200k caps for inference.

Output: data/freddie/exp84_s_h{12,24,36}.parquet with the same schema
the analysis rounds expect (bands as categoricals, era, prepay set by
horizon_sample downstream from event_code/months_to_event/followup).
Complete windows only (followup >= h); other-termination loans within
the horizon excluded (competing event), as in the analysis design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TD_ONE = Path(__file__).resolve().parents[1]
SEED = 84

BANDS = {
    "fico_band": ("credit_score", [0, 680, 720, 760, 900],
                  ["<680", "680-719", "720-759", "760+"]),
    "ltv_band": ("oltv_orig", [0, 80, 90, 95, 200],
                 ["<=80", "80-90", "90-95", ">95"]),
    "dti_band": ("dti", [0, 36, 45, 100], ["<=36", "36-45", ">45"]),
}


def build(h: int, cap: int, frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.rename(
        columns={"fp": "first_payment_date", "oltv": "oltv_orig",
                 "followup": "followup_months"}
    )
    d["event_type"] = pd.Categorical.from_codes(
        d["event_code"], categories=["active", "prepay", "other_term"]
    )
    d["purpose"] = pd.Categorical.from_codes(
        d["purpose_code"], categories=["P", "C", "N", "other"]
    )
    for name, (col, bins, labels) in BANDS.items():
        cats = pd.cut(d[col], bins=bins, labels=labels)
        if name == "dti_band":
            cats = cats.cat.add_categories(["missing"]).fillna("missing")
        d[name] = cats
    d["era"] = np.where(
        d["first_payment_date"] < 202200, "E1_2021",
        np.where(d["first_payment_date"] < 202400, "E2_2022_23", "E3_2024p"),
    )
    other_in_h = (d["event_type"] == "other_term") & (d["months_to_event"] <= h)
    d = d[(d["followup_months"] >= h) & ~other_in_h]
    keep = [
        "credit_score", "first_payment_date", "oltv_orig", "dti", "log_upb",
        "orig_rate", "market_pmms", "spread_bps", "followup_months",
        "months_to_event", "event_type", "purpose", "fico_band", "ltv_band",
        "dti_band", "era",
    ]
    d = d[keep]
    parts = []
    for _, g in d.groupby("era"):
        parts.append(g.sample(min(cap, len(g)), random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    frame = pd.read_parquet(TD_ONE / "data" / "freddie" / "exp84_frame.parquet")
    for h, cap in [(24, 400_000), (12, 250_000), (36, 250_000)]:
        s = build(h, cap, frame)
        out = TD_ONE / "data" / "freddie" / f"exp84_s_h{h}.parquet"
        s.to_parquet(out)
        print(f"H{h}: {len(s):,} rows -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
