"""exp81: cross-series pooled conformal banks — benchmark sweep.

Loads exp76's Chronos forecast rows (experiments/exp76_forecasts_{a,b}.json),
runs A native / B per-series CQR / C pooled-normalized CQR / C-raw pooled
raw-score control / D pooled + own-score shrinkage, and compares against
the pre-registered success bar (see exp81_report.md).

Harness check: arms A and B must reproduce exp76's native/CQR numbers on
the identical case sets.

Usage:
    .venv/bin/python experiments/exp81_pooled_banks.py [--only m4]
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from exp76_generate_data import load_all_rows  # noqa: E402
from scipy import stats  # noqa: E402

from td_one.uq import methods as uq_methods  # noqa: E402
from td_one.uq import pooled as uq_pooled  # noqa: E402
from td_one.uq.evaluate import CaseResult, build_cases, summarize  # noqa: E402

P = uq_pooled

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp81_results.json"
EXP76_SUMMARY = HERE / "exp76_sweep_results.json"


def family_of(config_id: str) -> str:
    return config_id.split(":")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tourism", "m3", "m4", "osfi", "smm"], default=None)
    args = ap.parse_args()

    rows = load_all_rows()
    cases_all = build_cases(rows)
    by_family: dict[str, dict[str, list]] = {}
    for cid, cases in cases_all.items():
        fam = family_of(cid)
        if args.only is not None and fam != args.only:
            continue
        by_family.setdefault(fam, {})[cid] = cases

    arms: dict[str, P.PooledArmFn] = {
        "native": P.local_arm(uq_methods.arm_native),
        "cqr": P.local_arm(uq_methods.arm_cqr),
        "pooled_cqr": P.arm_pooled_cqr,
        "pooled_raw": P.arm_pooled_raw,
        "pooled_shrink": P.arm_pooled_shrink,
        "pooled_cqr2": P.arm_pooled_cqr_signed,
        "pooled_resid2": P.arm_pooled_cqr_resid,
        "pooled_shrink2": P.arm_pooled_shrink_signed,
        "pooled_gated": P.arm_pooled_shrink_gated,
        "pooled_sim": P.arm_pooled_shrink_sim,
        "pooled_full": P.arm_pooled_shrink_full,
        "pooled_full16": partial(P.arm_pooled_shrink_full, gate_n=16),
    }
    results = P.run_study_pooled(by_family, arms)

    families = sorted(by_family)
    summary = {
        fam: summarize(results, lambda r, f=fam: family_of(r.config_id) == f) for fam in families
    }
    summary["pooled_all"] = summarize(results)

    # ---- harness check: A and B vs exp76 ----
    repro = json.loads(EXP76_SUMMARY.read_text())["summary_by_family"]
    print("== reproduction check vs exp76 ==")
    for fam in families:
        for arm in ("native", "cqr"):
            a, b = summary[fam][arm], repro[fam][arm]
            print(
                f"{fam:8s} {arm:6s} crps={a['crps']:.6f}/{b['crps']:.6f} "
                f"cov80={a['coverage80']:.4f}/{b['coverage80']:.4f} "
                f"n={int(a['n'])}/{int(b['n'])}"
            )

    # ---- paired comparisons vs B (per-series CQR) ----
    by_key: dict[tuple, dict[str, CaseResult]] = {}
    for r in results:
        by_key.setdefault((r.config_id, r.origin, r.h), {})[r.arm] = r
    native_mean = {fam: summary[fam]["native"]["crps"] for fam in families}

    pooled_arms = [a for a in arms if a not in ("native", "cqr")]
    vs_b: dict[str, dict[str, dict]] = {}
    for fam in families:
        keys = [k for k in by_key if family_of(k[0]) == fam]
        vs_b[fam] = {}
        for arm in pooled_arms:
            d_raw = np.array([by_key[k][arm].crps - by_key[k]["cqr"].crps for k in keys])
            t, p = stats.ttest_rel(
                np.array([by_key[k][arm].crps for k in keys]),
                np.array([by_key[k]["cqr"].crps for k in keys]),
            )
            wins = np.array([by_key[k][arm].crps < by_key[k]["cqr"].crps for k in keys])
            vs_b[fam][arm] = {
                "n": len(keys),
                "delta_raw_mean": float(np.mean(d_raw)),
                "delta_raw_p": float(p),
                "win_rate": float(np.mean(wins)),
                "cov80": summary[fam][arm]["coverage80"],
                "cov80_vs_b": summary[fam][arm]["coverage80"] - summary[fam]["cqr"]["coverage80"],
                "width_ratio": summary[fam][arm]["width_ratio"],
                "firing_rate": float(
                    np.mean([abs(by_key[k][arm].width_ratio - 1.0) > 1e-9 for k in keys])
                ),
            }
    # scale-free pooled comparison over every case, per pooled arm
    pooled_stats: dict[str, dict] = {}
    for arm in pooled_arms:
        d_all = np.array(
            [
                (dd[arm].crps - dd["cqr"].crps) / max(native_mean[family_of(k[0])], 1e-12)
                for k, dd in by_key.items()
            ]
        )
        tt = stats.ttest_1samp(d_all, 0.0)
        pooled_stats[arm] = {
            "n": int(len(d_all)),
            "norm_delta_mean": float(np.mean(d_all)),
            "norm_delta_t": float(tt.statistic),
            "norm_delta_p": float(tt.pvalue),
        }

    print("\n== pooled (scale-free) delta vs B, per arm ==")
    for arm, s in pooled_stats.items():
        print(
            f"{arm:16s} mean={s['norm_delta_mean']:+.5f} t={s['norm_delta_t']:+.2f} "
            f"p={s['norm_delta_p']:.3g} n={s['n']}"
        )
    print("\n== per-family vs B ==")
    for fam in families:
        print(f"-- {fam} --")
        for arm, s in vs_b[fam].items():
            print(
                f"{arm:14s} Δcrps={s['delta_raw_mean']:+.6f} p={s['delta_raw_p']:.3g} "
                f"win={s['win_rate']:.2f} cov80={s['cov80']:.3f} "
                f"(vsB {s['cov80_vs_b']:+.3f}) width={s['width_ratio']:.3f} "
                f"fire={s['firing_rate']:.2f} n={s['n']}"
            )

    configs = sorted({r.config_id for r in results})
    ci = {c: i for i, c in enumerate(configs)}
    arm_names = list(arms)
    seen: dict[tuple[int, int, int], list] = {}
    for r in results:
        key = (ci[r.config_id], r.origin, r.h)
        row = seen.setdefault(key, [key[0], key[1], key[2], [None] * len(arm_names)])
        row[3][arm_names.index(r.arm)] = float(f"{r.crps:.6g}")
    payload = {
        "config": {"n_cases": len(by_key), "arms": arm_names},
        "summary_by_family": summary,
        "vs_b": vs_b,
        "pooled_vs_b": pooled_stats,
        "configs": configs,
        "crps_rows": [seen[k] for k in sorted(seen)],
        "crps_rows_format": "[config_idx, origin, h, [crps per config.arms order]]",
    }
    OUT.write_text(json.dumps(payload))
    print(f"\nwrote {OUT} ({len(results)} case-results)")


if __name__ == "__main__":
    main()
