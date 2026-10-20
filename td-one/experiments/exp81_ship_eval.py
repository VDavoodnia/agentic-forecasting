"""exp81 round 5: the ship candidate — family-gated pooled shrinkage.

Design (from the round 1-4 diagnoses): cross-series pooled signed
shrinkage (arm D2, n0=8) is active only in many-series families
(>= 8 configs — the Monash regime the design targets, where per-series
banks are starved). Few-series families (OSFI's 4 heterogeneous
segments, SMM's single series) keep per-series CQR: pooling there was
harmful in every form tested, including where the per-series arm is
itself a no-op. Includes the R3 n0 sensitivity (4 / 8 / 24) on the
same family rule.

Output: experiments/exp81_ship_results.json + the pre-registered bar
evaluation printed at the end.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
from exp76_generate_data import load_all_rows  # noqa: E402
from scipy import stats

from td_one.uq import methods as uq_methods  # noqa: E402
from td_one.uq import pooled as uq_pooled  # noqa: E402
from td_one.uq.evaluate import CaseResult, build_cases, summarize  # noqa: E402

P = uq_pooled
HERE = Path(__file__).resolve().parent
OUT = HERE / "exp81_ship_results.json"
MIN_SERIES = 8


def family_of(config_id: str) -> str:
    return config_id.split(":")[0]


def main() -> None:
    rows = load_all_rows()
    cases_all = build_cases(rows)
    by_family: dict[str, dict[str, list]] = {}
    for cid, cases in cases_all.items():
        by_family.setdefault(family_of(cid), {})[cid] = cases

    results: list[CaseResult] = []
    for fam, fam_cases in sorted(by_family.items()):
        many = len(fam_cases) >= MIN_SERIES
        arms: dict[str, P.PooledArmFn] = {
            "native": P.local_arm(uq_methods.arm_native),
            "cqr": P.local_arm(uq_methods.arm_cqr),
        }
        if many:
            arms["ship"] = P.arm_pooled_shrink_signed
            arms["ship_n04"] = partial(P.arm_pooled_shrink_signed, n0=4.0)
            arms["ship_n024"] = partial(P.arm_pooled_shrink_signed, n0=24.0)
        else:
            arms["ship"] = P.local_arm(uq_methods.arm_cqr)
            arms["ship_n04"] = P.local_arm(uq_methods.arm_cqr)
            arms["ship_n024"] = P.local_arm(uq_methods.arm_cqr)
        results.extend(P.run_study_pooled({fam: fam_cases}, arms))
        print(f"{fam}: {'pooled' if many else 'per-series'} (n_configs={len(fam_cases)})")

    families = sorted(by_family)
    summary = {
        fam: summarize(results, lambda r, f=fam: family_of(r.config_id) == f) for fam in families
    }
    summary["pooled_all"] = summarize(results)
    by_key: dict[tuple, dict[str, CaseResult]] = {}
    for r in results:
        by_key.setdefault((r.config_id, r.origin, r.h), {})[r.arm] = r
    native_mean = {fam: summary[fam]["native"]["crps"] for fam in families}

    ship_arms = ["ship", "ship_n04", "ship_n024"]
    bar: dict[str, dict] = {}
    for arm in ship_arms:
        d_all = np.array(
            [
                (dd[arm].crps - dd["cqr"].crps) / max(native_mean[family_of(k[0])], 1e-12)
                for k, dd in by_key.items()
            ]
        )
        tt = stats.ttest_1samp(d_all, 0.0)
        raw_ship = float(np.mean([dd[arm].crps for dd in by_key.values()]))
        raw_b = float(np.mean([dd["cqr"].crps for dd in by_key.values()]))
        cov_ok = all(
            summary[fam][arm]["coverage80"] >= summary[fam]["cqr"]["coverage80"] - 0.01
            for fam in families
        )
        m4_gain = summary["m4"][arm]["coverage80"] - summary["m4"]["cqr"]["coverage80"]
        m4_width = summary["m4"][arm]["width_ratio"]
        bar[arm] = {
            "norm_delta_mean": float(np.mean(d_all)),
            "norm_delta_p": float(tt.pvalue),
            "raw_pooled_crps_ship": raw_ship,
            "raw_pooled_crps_b": raw_b,
            "crit1_crps": bool(np.mean(d_all) < 0 and tt.pvalue < 0.05 and raw_ship <= raw_b),
            "crit2_coverage_all_families": bool(cov_ok),
            "crit3_m4_cov80_gain": float(m4_gain),
            "crit3_m4_width": float(m4_width),
            "crit3_m4": bool(m4_gain >= 0.02 and abs(m4_width - 1.0) > 1e-9),
        }
        b = bar[arm]
        print(
            f"\n{arm}: norm Δ={b['norm_delta_mean']:+.5f} (p={b['norm_delta_p']:.3g}) "
            f"raw {raw_ship:.4f} vs B {raw_b:.4f}"
        )
        print(
            f"  bar: crit1={b['crit1_crps']} crit2={b['crit2_coverage_all_families']} "
            f"crit3={b['crit3_m4']} (M4 cov80 gain {m4_gain:+.3f}, width {m4_width:.3f})"
        )
        for fam in families:
            s, sb = summary[fam][arm], summary[fam]["cqr"]
            print(
                f"  {fam:8s} crps={s['crps']:.4f} (B {sb['crps']:.4f}) "
                f"cov80={s['coverage80']:.3f} (B {sb['coverage80']:.3f}) "
                f"width={s['width_ratio']:.3f}"
            )

    payload = {"summary_by_family": summary, "bar": bar, "min_series": MIN_SERIES}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
