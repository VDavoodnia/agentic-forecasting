"""exp76 Part B: benchmark sweep for the UQ layer.

Loads experiments/exp76_forecasts_{a,b}.json (Chronos forecasts for Tourism,
M3, M4 subsets + OSFI runoff segments + SMM) and runs the best UQ
configurations from Part A: native, plain CQR, regime-gated ACI, the
meta model (numeric), and both fusion designs in their numeric-feature
form (no story pipeline exists for these benchmarks, so story features
are all-absent — the fusion arms fall back to the numeric signal).

Output: experiments/exp76_sweep_results.json + per-benchmark tables.
"""

from __future__ import annotations

import json
from pathlib import Path

from exp76_generate_data import load_all_rows  # noqa: E402

from td_one.uq import fusion as uqf
from td_one.uq import methods as uq_methods
from td_one.uq.evaluate import build_cases, run_study, summarize

M = uq_methods

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp76_sweep_results.json"

FEATS_NUM = M.META_FEATURES_NUMERIC


class MetaArm:
    """arm_meta with the fitted GBM cached per (origin, bank length)."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self._bank_key: tuple | None = None
        self._model = None

    def __call__(self, case, bank):
        key = (case.origin, len(bank))
        if key != self._bank_key:
            self._bank_key = key
            self._model = M.fit_meta_model(bank, self.features)
        if self._model is None:
            return M.arm_rolling_conformal(case, bank)
        pred = float(self._model.predict(M.meta_matrix([case], self.features))[0])
        pred = max(pred, 1e-12)
        m, q10, q90 = case.interval_edges()
        return max(pred / max(m - q10, 1e-12), 0.0), max(pred / max(q90 - m, 1e-12), 0.0)


def family_of(config_id: str) -> str:
    return config_id.split(":")[0]


def main() -> None:
    rows = load_all_rows()
    cases_by_config = build_cases(rows)

    arms = {
        "native": M.arm_native,
        "cqr": M.arm_cqr,
        "aci_gated": uqf.arm_aci_gated,
        "meta_numeric": MetaArm(FEATS_NUM),
        "cqr_meta_level_num": uqf.MetaLevelCqr(FEATS_NUM),
        "cqr_conditional_num": uqf.ConditionalCqr(FEATS_NUM),
        # Round-3 champion on Food CPI in numeric-feature form.
        "cqr_meta_level_h3_num": uqf.MetaLevelCqr(FEATS_NUM, min_h=3),
    }
    results = run_study(cases_by_config, arms)

    families = sorted({family_of(cid) for cid in cases_by_config})
    summary = {
        fam: summarize(results, lambda r, f=fam: family_of(r.config_id) == f) for fam in families
    }
    summary["pooled_all"] = summarize(results)
    per_config = {
        cid: summarize(results, lambda r, c=cid: r.config_id == c)
        for cid in sorted(cases_by_config)
    }
    # Per-case payload is a factored CRPS array (config index + per-arm
    # values): the full row dump exceeds the push connector's body-size
    # limit; coverage/width aggregates live in the summaries above.
    configs = sorted(cases_by_config)
    ci = {c: i for i, c in enumerate(configs)}
    arm_names = list(arms)
    seen: dict[tuple[int, int, int], list] = {}
    for r in results:
        key = (ci[r.config_id], r.origin, r.h)
        row = seen.setdefault(key, [key[0], key[1], key[2], [None] * len(arm_names)])
        row[3][arm_names.index(r.arm)] = float(f"{r.crps:.6g}")
    crps_rows = [seen[k] for k in sorted(seen)]
    payload = {
        "config": {"n_rows": len(rows), "arms": arm_names},
        "summary_by_family": summary,
        "per_config": per_config,
        "configs": configs,
        "crps_rows": crps_rows,
        "crps_rows_format": "[config_idx, origin, h, [crps per config.arms order]]",
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({len(results)} case-results)")

    for fam in families + ["pooled_all"]:
        print(f"\n== {fam} ==")
        for arm, s in summary[fam].items():
            print(
                f"{arm:22s} cov80={s['coverage80']:.3f} cov90={s['coverage90']:.3f} "
                f"width_ratio={s['width_ratio']:.3f} crps={s['crps']:.4f} "
                f"p={s['crps_paired_p']:.3g} win={s['crps_win_rate']:.2f} n={int(s['n'])}"
            )


if __name__ == "__main__":
    main()
