"""exp85 probe 2: errors-detector engagement check in the real harness.

Builds one fresh seed's S1 world and calls the group forecaster's
conformal offsets directly on truncated group/portfolio series at
post-onset origins, comparing detectors (scan / cusum / errors)
under the R4 configuration (level 0.90, conditional).
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

from td_one.agent import group_forecaster as gf  # noqa: E402

sd = exp72.build_seed_data(2024, "tree")
groups = exp72.build_world(sd, "S1_4.0")
for gid in ("__portfolio__", "g0", "g1", "g2"):
    s = np.array(groups[gid].attrition_rate, dtype=float)
    for o in (18, 19, 20, 21):
        ctx = s[:o]
        row = [f"{gid} o={o}"]
        for det in ("scan", "cusum", "errors"):
            off = gf._conformal_offsets(
                ctx,
                center="trailing_mean",
                center_window=6,
                spread_method="gaussian",
                min_bank=8,
                conditional=True,
                level=0.90,
                adaptive="reset",
                min_post=2,
                detector=det,
            )
            row.append(f"{det}={off[0]:.3f}/{off[1]:.3f}")
        print("  ".join(row), flush=True)
