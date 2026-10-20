"""exp85 analysis: paired CRPS tests between UQ arms (food + synth).

summarize() only pairs arms against native; the ship decision needs
arm-vs-arm pairs (BC family vs the shipped CQR). Deterministic on
the cached forecasts.
"""

import functools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.uq import adaptive as uq_adaptive  # noqa: E402
from td_one.uq import methods as uq_methods  # noqa: E402
from td_one.uq.evaluate import build_cases, run_study  # noqa: E402

A = uq_adaptive
M = uq_methods

HERE = Path(__file__).resolve().parent
p = functools.partial
ARMS = {
    "cqr": M.arm_cqr,
    "bc_cqr": A.arm_bc_cqr,
    "bc_aci": A.arm_bc_aci,
    "bc_aci_g10": p(A.arm_bc_aci, gamma=0.10),
    "bc_lam90_n02": p(A.arm_bc_cqr, bias_lambda=0.90, bias_n0=2.0),
}


def paired(results, a: str, b: str, scope) -> None:
    da = {(r.config_id, r.origin, r.h): r.crps for r in results if r.arm == a and scope(r)}
    db = {(r.config_id, r.origin, r.h): r.crps for r in results if r.arm == b and scope(r)}
    keys = sorted(set(da) & set(db))
    x = np.array([da[k] for k in keys])
    y = np.array([db[k] for k in keys])
    t, pv = stats.ttest_rel(x, y)
    print(
        f"{a:14s} vs {b:14s}: n={len(keys)} mean {x.mean():.4f} vs {y.mean():.4f} "
        f"delta={x.mean() - y.mean():+.4f} ({(x.mean() / y.mean() - 1) * 100:+.1f}%) "
        f"p={pv:.3g} win={(x < y).mean():.2f}"
    )


def main() -> None:
    rows: list[dict] = []
    for fname in ("exp74_forecasts.json", "exp85_forecasts.json"):
        rows.extend(json.loads((HERE / fname).read_text())["rows"])
    results = run_study(build_cases(rows), ARMS)
    food = lambda r: r.config_id == "food_cpi"  # noqa: E731

    def fresh_s1(r) -> bool:
        return r.config_id.startswith("synth:S1:4.0") and (
            ":seed2" in r.config_id or ":seed3" in r.config_id
        )

    print("== food ==")
    for a, b in (
        ("bc_cqr", "cqr"),
        ("bc_aci", "cqr"),
        ("bc_aci", "bc_cqr"),
        ("bc_aci_g10", "bc_aci"),
        ("bc_lam90_n02", "cqr"),
    ):
        paired(results, a, b, food)
    print("== fresh S1@4sigma ==")
    for a, b in (("bc_cqr", "cqr"), ("bc_aci", "cqr"), ("bc_lam90_n02", "cqr")):
        paired(results, a, b, fresh_s1)


if __name__ == "__main__":
    main()
