"""exp85 probe: detector firing check on the harness's S1 series.

Builds one fresh seed's S1 world (exp72 machinery) and runs
detect_break_onset (scan + cusum gates) on the portfolio series and
each group's series at post-onset origins.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e80 = _load("exp80_conformal_fallback", "exp80_conformal_fallback.py")
exp72 = e80.exp72

from td_one.uq.adaptive import detect_break_onset  # noqa: E402

sd = exp72.build_seed_data(2024, "tree")
groups = exp72.build_world(sd, "S1_4.0")
gids = [g for g in groups if g != "__portfolio__"]
print("groups:", gids)
for gid in ["__portfolio__", *gids]:
    s = np.array(groups[gid].attrition_rate, dtype=float)
    for o in (34, 38, 42, 46):
        ctx = s[:o]
        print(
            f"{gid:16s} origin={o} scan={detect_break_onset(ctx)} "
            f"cusum={detect_break_onset(ctx, gate='cusum')} "
            f"cusum3={detect_break_onset(ctx, gate='cusum', cusum_threshold=3.0)}"
        )
