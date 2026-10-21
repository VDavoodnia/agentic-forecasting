"""Resume Tourism TSFM inference into exp60_tsfm.json (survives restarts).

run_tsfm_quantiles resumes from series already in the output JSON, so this
is safe to re-run. Writes ONLY exp60_tsfm.json -- never touches the
exp49_results.json that belongs to the main chat.

Usage:
    .venv/bin/python experiments/exp60_tsfm_resume.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import monash_common as mc  # noqa: E402

TSF = ROOT / "data" / "tourism" / "tourism_monthly_dataset.tsf"
OUT = ROOT / "experiments" / "exp60_tsfm.json"


def main() -> int:
    series, horizon = mc.parse_tsf(TSF)
    assert horizon == 24, horizon
    done = mc.run_tsfm_quantiles(series, horizon, OUT)
    print(f"final: {len(done)} series in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
