"""exp74 Phase 2a: generate Chronos-2 per-origin forecasts for the UQ battery.

Stores, per (config, origin): the native 21-level quantile grid (K x H),
the realized values (H,), and the series context up to the origin — everything
the uncertainty methods in ``td_one.uq`` need, computed once and cached so
method iteration is free.

Configs:
  * synthetic exp67 scenarios (same seeds/protocol as exp67_tsfm_redteam):
    S1 @1.0/4.0, S2 @4.0, S3 @4.0, control — rolling origins every 2 months
    from 12 to 42, h=6 (onset=30, n_months=48).
  * real: Food CPI monthly level (benchmarks/food_cpi/data.csv), rolling
    origins monthly from 2017-01, h=6.

Resume-safe: appends rows and skips (config, origin) pairs already present.

Usage:
    .venv/bin/python experiments/exp74_generate_data.py [--only synth|food]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from td_one.data_gen.shift_scenarios import (
    ShiftSpec,
    apply_regional_divergence,
    apply_shift,
    generate_base_series,
)
from td_one.models.tsfm import Chronos2Wrapper

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp74_forecasts.json"
HORIZON = 6
N_MONTHS = 48
ONSET = 30
SYNTH_ORIGINS = list(range(12, 43, 2))
SYNTH_CONFIGS: list[tuple[str, float, int]] = [
    ("S1", 1.0, 0),
    ("S1", 4.0, 0),
    ("S1", 4.0, 1),
    ("S2", 4.0, 0),
    ("S3", 4.0, 0),
    ("none", 0.0, 0),
    ("none", 0.0, 1),
]


def build_shifted_series(shift: str, severity: float, seed: int) -> np.ndarray:
    """Rebuild an exp67 series exactly as exp67_tsfm_redteam.run_combo did."""
    rng = np.random.default_rng(1000 + seed)
    if shift == "S3":
        ra = generate_base_series(N_MONTHS, seed=2000 + seed)
        rb = generate_base_series(N_MONTHS, seed=3000 + seed)
        series = apply_regional_divergence(ra, rb, ShiftSpec("S3", ONSET, severity), 0.5, rng)
    elif shift == "none":
        series = generate_base_series(N_MONTHS, seed=4000 + seed)
    else:
        base = generate_base_series(N_MONTHS, seed=4000 + seed)
        series = apply_shift(base, ShiftSpec(shift, ONSET, severity), rng)
    return np.asarray(series["attrition_rate"], dtype=float)


def load_existing() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"config": {"horizon": HORIZON}, "rows": []}


def save(doc: dict) -> None:
    OUT.write_text(json.dumps(doc))


def run_forecasts(
    model: Chronos2Wrapper,
    doc: dict,
    config_id: str,
    series: np.ndarray,
    origins: list[int],
    seed: int,
    validate_input: bool = True,
) -> None:
    done = {(r["config_id"], r["origin"]) for r in doc["rows"]}
    for origin in origins:
        if (config_id, origin) in done:
            continue
        if origin + HORIZON > len(series):
            continue
        t0 = time.time()
        levels, values = model.forecast_quantiles(
            series[:origin], horizon=HORIZON, seed=seed, validate_input=validate_input
        )
        doc["rows"].append(
            {
                "config_id": config_id,
                "origin": int(origin),
                "levels": [float(x) for x in levels],
                "values": [[float(x) for x in row] for row in values],
                "realized": [float(x) for x in series[origin : origin + HORIZON]],
                "context": [float(x) for x in series[:origin]],
                "elapsed_s": round(time.time() - t0, 2),
            }
        )
        save(doc)
        print(f"{config_id} origin={origin} done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["synth", "food"], default=None)
    args = ap.parse_args()

    doc = load_existing()
    weights_dir = HERE.parent / "models" / "weights" / "chronos2"
    model = Chronos2Wrapper(weights_dir=str(weights_dir))

    if args.only in (None, "synth"):
        for shift, sev, seed in SYNTH_CONFIGS:
            cid = f"synth:{shift}:{sev}:seed{seed}"
            series = build_shifted_series(shift, sev, seed)
            run_forecasts(model, doc, cid, series, SYNTH_ORIGINS, seed)

    if args.only in (None, "food"):
        import csv

        path = HERE.parent / "benchmarks" / "food_cpi" / "data.csv"
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        series = np.array([float(r["value"]) for r in rows], dtype=float)
        # Monthly origins from 2017-01 (index 84) to the last with h realized.
        origins = list(range(84, len(series) - HORIZON + 1))
        # Raw CPI levels are not rates in [0, 1]: Chronos normalizes context
        # internally (same convention as exp69c's SMM arm).
        run_forecasts(model, doc, "food_cpi", series, origins, seed=0,
                      validate_input=False)

    print(f"total rows: {len(doc['rows'])}", flush=True)


if __name__ == "__main__":
    main()
