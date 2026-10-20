"""exp76 Part B: Chronos-2 per-origin forecasts for the benchmark sweep.

Tourism (subset), M3 (subset), M4 (subset), OSFI runoff segments, SMM —
everything the UQ sweep needs, computed once and cached in
``experiments/exp76_forecasts.json``.

Resume-safe: appends rows and skips (config, origin) pairs already present.

Usage:
    .venv/bin/python experiments/exp76_generate_data.py [--only tourism|m3|m4|osfi|smm]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "exp76_forecasts.json"


def load_tsf(path: Path) -> list[tuple[str, np.ndarray]]:
    """Parse a Monash .tsf file -> [(series_name, values)]."""
    out = []
    in_data = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line == "@data":
            in_data = True
            continue
        if not in_data or not line:
            continue
        # name:start_timestamp:values  (timestamp itself contains colons)
        name, rest = line.split(":", 1)
        # rest = "1979-01-01 00-00-00:1.0,2.0,..."
        _, _, vals = rest.partition(":")
        arr = np.array([float(v) for v in vals.split(",") if v.strip()], dtype=float)
        if np.isfinite(arr).all() and len(arr) > 0:
            out.append((name, arr))
    return out


def pick_subset(
    series: list[tuple[str, np.ndarray]], n: int, min_len: int
) -> list[tuple[str, np.ndarray]]:
    """Deterministic subset: filter by length, take evenly spaced indices."""
    ok = [s for s in series if len(s[1]) >= min_len]
    if len(ok) <= n:
        return ok
    idx = [round(i * (len(ok) - 1) / (n - 1)) for i in range(n)]
    return [ok[i] for i in idx]


def origins_for(n_vals: int, horizon: int, n: int, step: int) -> list[int]:
    """Last ``n`` origins, ``step`` apart, each with ``horizon`` realized."""
    last = n_vals - horizon
    return [last - (n - 1 - i) * step for i in range(n)]


def load_existing() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"config": {}, "rows": []}


def save(doc: dict) -> None:
    OUT.write_text(json.dumps(doc))


def run_forecasts(
    model, doc: dict, config_id: str, series: np.ndarray, origins: list[int], horizon: int
) -> None:
    done = {(r["config_id"], r["origin"]) for r in doc["rows"]}
    for origin in origins:
        if (config_id, origin) in done:
            continue
        if origin + horizon > len(series):
            continue
        t0 = time.time()
        levels, values = model.forecast_quantiles(
            series[:origin], horizon=horizon, validate_input=False
        )
        doc["rows"].append(
            {
                "config_id": config_id,
                "origin": int(origin),
                "levels": [float(x) for x in levels],
                "values": [[float(x) for x in row] for row in values],
                "realized": [float(x) for x in series[origin : origin + horizon]],
                "context": [float(x) for x in series[:origin]],
                "elapsed_s": round(time.time() - t0, 2),
            }
        )
        save(doc)
        print(f"{config_id} origin={origin} done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tourism", "m3", "m4", "osfi", "smm"], default=None)
    args = ap.parse_args()

    from td_one.models.tsfm import Chronos2Wrapper  # noqa: PLC0415

    doc = load_existing()
    model = Chronos2Wrapper(weights_dir=str(ROOT / "models" / "weights" / "chronos2"))

    if args.only in (None, "tourism"):
        series = load_tsf(ROOT / "data" / "tourism" / "tourism_monthly_dataset.tsf")
        for name, vals in pick_subset(series, 20, 144):
            run_forecasts(
                model, doc, f"tourism:{name}", vals, origins_for(len(vals), 24, 12, 6), 24
            )

    if args.only in (None, "m3"):
        series = load_tsf(ROOT / "data" / "m3mon" / "m3_monthly_dataset.tsf")
        for name, vals in pick_subset(series, 20, 120):
            run_forecasts(model, doc, f"m3:{name}", vals, origins_for(len(vals), 18, 10, 6), 18)

    if args.only in (None, "m4"):
        with open(ROOT / "data" / "m4" / "Monthly-train.csv") as fh:
            rows = list(csv.reader(fh))
        series = []
        for r in rows[1:]:
            vals = np.array([float(v) for v in r[1:] if v.strip()], dtype=float)
            if np.isfinite(vals).all() and len(vals) >= 80:
                series.append((r[0], vals))
        for name, vals in pick_subset(series, 20, 80):
            run_forecasts(model, doc, f"m4:{name}", vals, origins_for(len(vals), 18, 8, 6), 18)

    if args.only in (None, "osfi"):
        import pandas as pd  # noqa: PLC0415

        df = pd.read_parquet(ROOT / "data" / "osfi" / "m4_runoff.parquet")
        for seg, g in df.groupby("segment"):
            vals = np.asarray(g.sort_values("month")["runoff_m"], dtype=float)
            origins = list(range(len(vals) - 36 - 6, len(vals) - 6))
            run_forecasts(model, doc, f"osfi:{seg}", vals, origins, 6)

    if args.only in (None, "smm"):
        vals = np.genfromtxt(
            ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv",
            delimiter=",",
            skip_header=1,
            usecols=3,
        )
        origins = list(range(24, len(vals) - 6 + 1))
        run_forecasts(model, doc, "smm", np.asarray(vals, dtype=float), origins, 6)

    print(f"total rows: {len(doc['rows'])}", flush=True)


if __name__ == "__main__":
    main()
