"""exp85 testbed 1: shift-adaptive conformal arms on Chronos-2 UQ.

Loads exp74's cached forecasts (dev seeds 0/1) + exp85's fresh-seed
forecasts (seeds 2/3, exp85_forecasts.json) and runs the exp74
PIT-safe study driver with the exp85 arms from td_one.uq.adaptive
plus the shipped references (native, CQR, ACI).

Pre-registered bar (exp85_report.md): on FRESH seeds, S1@4sigma
pooled cov80 >= 0.77, control CRPS harm vs native <= +2%, control
cov80 in [0.77, 0.85].

Usage:
    .venv/bin/python experiments/exp85_chronos_uq.py [--out NAME]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.uq import adaptive as uq_adaptive  # noqa: E402
from td_one.uq import methods as uq_methods  # noqa: E402
from td_one.uq.evaluate import build_cases, run_study, summarize  # noqa: E402

A = uq_adaptive
M = uq_methods

HERE = Path(__file__).resolve().parent


def build_arms() -> dict:
    p = functools.partial
    return {
        "native": M.arm_native,
        "cqr": M.arm_cqr,
        "aci": M.arm_aci,
        "cqr_reset": A.arm_cqr_reset,
        "cqr_decay_hl3": p(A.arm_cqr_decay, half_life=3.0),
        "cqr_decay_hl6": p(A.arm_cqr_decay, half_life=6.0),
        "cqr_decay_hl12": p(A.arm_cqr_decay, half_life=12.0),
        "cqr_cp": p(A.arm_cqr_cp, half_life=6.0, pre_factor=0.25),
        "cqr_cp_pf01": p(A.arm_cqr_cp, half_life=6.0, pre_factor=0.1),
        "cqr_cp_pf05": p(A.arm_cqr_cp, half_life=6.0, pre_factor=0.5),
        "bc_cqr": A.arm_bc_cqr,
        "bc_lam90": p(A.arm_bc_cqr, bias_lambda=0.90),
        "bc_lam98": p(A.arm_bc_cqr, bias_lambda=0.98),
        "bc_n0_4": p(A.arm_bc_cqr, bias_n0=4.0),
        "bc_n0_24": p(A.arm_bc_cqr, bias_n0=24.0),
        "bc_aci": A.arm_bc_aci,
        "cp_bc": A.arm_cp_bc,
        # --- exp85 round 2: faster bias tracks + detectors that engage ---
        "bc_lam85": p(A.arm_bc_cqr, bias_lambda=0.85),
        "bc_lam90_n02": p(A.arm_bc_cqr, bias_lambda=0.90, bias_n0=2.0),
        "bc_lam90_n06": p(A.arm_bc_cqr, bias_lambda=0.90, bias_n0=6.0),
        "bc_aci_lam90": p(A.arm_bc_aci, bias_lambda=0.90),
        "bc_aci_g10": p(A.arm_bc_aci, gamma=0.10),
        "cqr_reset_cusum": p(A.arm_cqr_reset, detector="cusum"),
        "cqr_reset_cusum3": p(A.arm_cqr_reset, detector="cusum", cusum_threshold=3.0),
        "cqr_reset_pool_cusum": p(A.arm_cqr_reset_pool, detector="cusum"),
        "cqr_reset_pool_scan": A.arm_cqr_reset_pool,
        "cp_bc_cusum": p(A.arm_cp_bc, detector="cusum"),
    }


def _fresh(r) -> bool:
    return ":seed2" in r.config_id or ":seed3" in r.config_id


def _dev(r) -> bool:
    return r.config_id.startswith("synth:") and not _fresh(r)


SCOPES = {
    "dev_synth": _dev,
    "dev_S1_4sigma": lambda r: _dev(r) and r.config_id.startswith("synth:S1:4.0"),
    "dev_S1_4sigma_straddle": lambda r: (
        _dev(r) and r.config_id.startswith("synth:S1:4.0") and 27 <= r.origin <= 33
    ),
    "dev_control": lambda r: _dev(r) and r.config_id.startswith("synth:none"),
    "fresh_S1_4sigma": lambda r: _fresh(r) and r.config_id.startswith("synth:S1:4.0"),
    "fresh_S1_4sigma_straddle": lambda r: (
        _fresh(r) and r.config_id.startswith("synth:S1:4.0") and 27 <= r.origin <= 33
    ),
    "fresh_control": lambda r: _fresh(r) and r.config_id.startswith("synth:none"),
    "food": lambda r: r.config_id == "food_cpi",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="exp85_chronos_results.json")
    args = ap.parse_args()

    rows: list[dict] = []
    for fname in ("exp74_forecasts.json", "exp85_forecasts.json"):
        path = HERE / fname
        if path.exists():
            rows.extend(json.loads(path.read_text())["rows"])
    cases_by_config = build_cases(rows)
    arms = build_arms()
    t0 = time.time()
    results = run_study(cases_by_config, arms)
    summary = {name: summarize(results, scope) for name, scope in SCOPES.items()}
    payload = {
        "n_rows": len(rows),
        "arms": list(arms),
        "runtime_s": round(time.time() - t0, 1),
        "summary": summary,
    }
    (HERE / args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out} ({len(results)} case-results, {payload['runtime_s']}s)")
    for scope in (
        "fresh_S1_4sigma",
        "fresh_S1_4sigma_straddle",
        "fresh_control",
        "dev_S1_4sigma_straddle",
        "food",
    ):
        print(f"\n== {scope} ==")
        for arm, s in summary[scope].items():
            print(
                f"{arm:16s} cov80={s['coverage80']:.3f} cov90={s['coverage90']:.3f} "
                f"wr={s['width_ratio']:.3f} crps={s['crps']:.4f} p={s['crps_paired_p']:.3g} "
                f"win={s['crps_win_rate']:.2f} n={int(s['n'])}"
            )


if __name__ == "__main__":
    main()
