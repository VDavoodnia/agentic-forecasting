"""exp74 study driver: PIT-safe rolling evaluation of UQ arms.

Builds (origin, horizon-step) cases from cached Chronos forecasts, then for
each origin recomputes the *bank* of cases whose realized values were
strictly observable before that origin (origin' + h' - 1 < origin) and runs
every arm against that bank. Any arm output therefore depends only on
pre-origin information — the PIT gate is structural, not conventional.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from td_one.evaluation.real_benchmark import crps_from_quantiles
from td_one.uq.features import numeric_features
from td_one.uq.methods import ForecastCase

ArmFn = Callable[[ForecastCase, list[ForecastCase]], tuple[float, float]]


@dataclass
class CaseResult:
    config_id: str
    origin: int
    h: int
    arm: str
    coverage80: float
    coverage90: float
    width80: float
    width_ratio: float
    crps: float
    crps_native: float


def _edge(q: np.ndarray, levels: np.ndarray, target: float) -> float:
    return float(q[int(np.argmin(np.abs(levels - target)))])


def known_bank(all_cases: list[ForecastCase], origin: int) -> list[ForecastCase]:
    """Cases strictly observable before ``origin`` (the PIT rule)."""
    return [c for c in all_cases if c.origin + c.h - 1 < origin]


def build_cases(
    rows: list[dict],
    extra_features: dict[tuple[str, int], dict[str, float]] | None = None,
) -> dict[str, list[ForecastCase]]:
    """Forecast rows -> {(config): [cases grouped per origin structure]}.

    Cases are returned flat per config; the driver re-derives the bank per
    origin. Features are computed in a second pass (they need the bank).
    """
    out: dict[str, list[ForecastCase]] = {}
    extra_features = extra_features or {}
    for row in rows:
        cid = row["config_id"]
        origin = int(row["origin"])
        levels = np.asarray(row["levels"], dtype=float)
        values = np.asarray(row["values"], dtype=float)
        realized = np.asarray(row["realized"], dtype=float)
        context = np.asarray(row["context"], dtype=float)
        horizon = values.shape[1]
        width_1 = float(values[int(np.argmin(np.abs(levels - 0.9))), 0]) - float(
            values[int(np.argmin(np.abs(levels - 0.1))), 0]
        )
        config_cases = out.setdefault(cid, [])
        for h in range(1, horizon + 1):
            feats = dict(extra_features.get((cid, origin), {}))
            width_h = float(values[int(np.argmin(np.abs(levels - 0.9))), h - 1]) - float(
                values[int(np.argmin(np.abs(levels - 0.1))), h - 1]
            )
            feats["width_growth"] = width_h / max(width_1, 1e-12)
            config_cases.append(
                ForecastCase(
                    origin=origin,
                    h=h,
                    levels=levels,
                    q=values[:, h - 1].copy(),
                    y=float(realized[h - 1]),
                    context=context,
                    feats=feats,
                )
            )
    return out


def fill_numeric_features(cases: list[ForecastCase]) -> None:
    """Second pass: numeric features need the bank, so compute per origin."""
    origins = sorted({c.origin for c in cases})
    for origin in origins:
        bank = known_bank(cases, origin)
        for case in cases:
            if case.origin != origin:
                continue
            nf = numeric_features(case.context, case.h, case.levels, case.q, bank)
            # Preserve width_growth (computed from the full grid) + any
            # story/macro features merged earlier.
            wg = case.feats.get("width_growth", 1.0)
            case.feats.update(nf)
            case.feats["width_growth"] = wg
            case.feats = {k: float(v) for k, v in case.feats.items() if isinstance(v, (int, float))}


def run_study(
    cases_by_config: dict[str, list[ForecastCase]],
    arms: dict[str, ArmFn],
) -> list[CaseResult]:
    """Run every arm over every case with PIT-safe banks."""
    from td_one.uq.methods import scale_grid  # noqa: PLC0415

    results: list[CaseResult] = []
    for cid, cases in sorted(cases_by_config.items()):
        fill_numeric_features(cases)
        for case in sorted(cases, key=lambda c: (c.origin, c.h)):
            bank = known_bank(cases, case.origin)
            native_w = max(
                _edge(case.q, case.levels, 0.9) - _edge(case.q, case.levels, 0.1),
                1e-12,
            )
            crps_native = float(crps_from_quantiles(case.levels, case.q, np.array([case.y])))
            for arm_name, arm in arms.items():
                r_lo, r_hi = arm(case, bank)
                q_adj = scale_grid(case.levels, case.q, r_lo, r_hi)
                lo80 = _edge(q_adj, case.levels, 0.1)
                hi80 = _edge(q_adj, case.levels, 0.9)
                lo90 = _edge(q_adj, case.levels, 0.05)
                hi90 = _edge(q_adj, case.levels, 0.95)
                w80 = hi80 - lo80
                results.append(
                    CaseResult(
                        config_id=cid,
                        origin=case.origin,
                        h=case.h,
                        arm=arm_name,
                        coverage80=1.0 if lo80 <= case.y <= hi80 else 0.0,
                        coverage90=1.0 if lo90 <= case.y <= hi90 else 0.0,
                        width80=w80,
                        width_ratio=w80 / native_w,
                        crps=float(crps_from_quantiles(case.levels, q_adj, np.array([case.y]))),
                        crps_native=crps_native,
                    )
                )
    return results


def summarize(
    results: list[CaseResult],
    scope: Callable[[CaseResult], bool] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-arm pooled metrics + paired tests vs the native arm."""
    sel = [r for r in results if scope is None or scope(r)]
    out: dict[str, dict[str, float]] = {}
    by_arm: dict[str, list[CaseResult]] = {}
    for r in sel:
        by_arm.setdefault(r.arm, []).append(r)
    native = {(r.config_id, r.origin, r.h): r for r in by_arm.get("native", [])}
    for arm, rows in sorted(by_arm.items()):
        cov80 = float(np.mean([r.coverage80 for r in rows]))
        cov90 = float(np.mean([r.coverage90 for r in rows]))
        wr = float(np.mean([r.width_ratio for r in rows]))
        crps = float(np.mean([r.crps for r in rows]))
        paired = [
            (r.crps, native[(r.config_id, r.origin, r.h)].crps)
            for r in rows
            if (r.config_id, r.origin, r.h) in native
        ]
        if arm != "native" and len(paired) >= 8:
            a = np.array([p[0] for p in paired])
            b = np.array([p[1] for p in paired])
            t, p = stats.ttest_rel(a, b)
            win = float(np.mean(a < b))
        else:
            t, p, win = float("nan"), float("nan"), float("nan")
        out[arm] = {
            "n": float(len(rows)),
            "coverage80": cov80,
            "coverage90": cov90,
            "width_ratio": wr,
            "crps": crps,
            "crps_paired_t": float(t),
            "crps_paired_p": float(p),
            "crps_win_rate": win,
        }
    return out
