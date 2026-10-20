"""exp85: generate FRESH-seed Chronos-2 forecasts for the UQ battery.

exp74's battery used synth seeds 0/1. exp85's pre-registered bar is
evaluated on fresh seeds 2/3 (never touched by exp74/76/81), so this
script mirrors ``exp74_generate_data.py`` exactly — same series
builder, origins, horizon — but writes to ``exp85_forecasts.json``
(exp74's committed cache is not mutated) and only builds the configs
the bar needs: S1 @4.0 and control (none), seeds 2 and 3.

Resume-safe: skips (config, origin) pairs already present.

Usage:
    .venv/bin/python experiments/exp85_generate_fresh.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.models.tsfm import Chronos2Wrapper  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp85_forecasts.json"
HORIZON = 6
SYNTH_ORIGINS = list(range(12, 43, 2))
FRESH_CONFIGS: list[tuple[str, float, int]] = [
    ("S1", 4.0, 2),
    ("S1", 4.0, 3),
    ("none", 0.0, 2),
    ("none", 0.0, 3),
]


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen74 = _load("exp74_generate_data", "exp74_generate_data.py")


def main() -> None:
    doc: dict = (
        json.loads(OUT.read_text())
        if OUT.exists()
        else {"config": {"horizon": HORIZON}, "rows": []}
    )
    done = {(r["config_id"], r["origin"]) for r in doc["rows"]}
    weights_dir = HERE.parent / "models" / "weights" / "chronos2"
    model = Chronos2Wrapper(weights_dir=str(weights_dir))
    for shift, sev, seed in FRESH_CONFIGS:
        cid = f"synth:{shift}:{sev}:seed{seed}"
        series = gen74.build_shifted_series(shift, sev, seed)
        for origin in SYNTH_ORIGINS:
            if (cid, origin) in done or origin + HORIZON > len(series):
                continue
            t0 = time.time()
            levels, values = model.forecast_quantiles(series[:origin], horizon=HORIZON, seed=seed)
            doc["rows"].append(
                {
                    "config_id": cid,
                    "origin": int(origin),
                    "levels": [float(x) for x in levels],
                    "values": [[float(x) for x in row] for row in np.asarray(values)],
                    "realized": [float(x) for x in series[origin : origin + HORIZON]],
                    "context": [float(x) for x in series[:origin]],
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            OUT.write_text(json.dumps(doc))
            print(f"{cid} origin={origin} done in {time.time() - t0:.1f}s", flush=True)
    print(f"total rows: {len(doc['rows'])}", flush=True)


if __name__ == "__main__":
    main()
