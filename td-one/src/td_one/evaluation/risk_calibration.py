"""Calibration metrics for categorical risk-level classifiers (exp66g).

A risk level (LOW/MEDIUM/HIGH) is only falsifiable if it maps to a
number. Each level is assigned an *anchor*: the mean training-period
realized attrition of the groups the deterministic tertile bins
placed in that level. Anchors are computed from training data only
(PIT), with one identical mapping for every arm, so arms differ only
through their level assignments.

Primary metric: per-group absolute *bin error*
``|anchor(level) - realized_test_attrition|``, averaged (MAE) over
groups and seeds, with paired bootstrap CIs on arm differences.
Secondary: monotonicity — mean realized attrition non-decreasing
across LOW -> MEDIUM -> HIGH.
"""

from __future__ import annotations

import numpy as np

from td_one.agent.risk_reporter import compute_risk_bins

LEVEL_ORDER = ("LOW", "MEDIUM", "HIGH")
LEVEL_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def level_anchors(
    train_realized: dict[str, float],
    train_bins: dict[str, str] | None = None,
) -> dict[str, float]:
    """Attrition anchor per risk level, from training data only.

    ``train_bins`` defaults to the deterministic tertile bins of
    ``train_realized`` (the same bins the deterministic arm uses).
    Levels with no groups fall back to the overall training mean.
    """
    if not train_realized:
        raise ValueError("train_realized must be non-empty")
    bins = (
        train_bins
        if train_bins is not None
        else compute_risk_bins(train_realized, list(train_realized.values()))
    )
    overall = float(np.mean(list(train_realized.values())))
    out: dict[str, float] = {}
    for lvl in LEVEL_ORDER:
        vals = [train_realized[g] for g, b in bins.items() if b == lvl and g in train_realized]
        out[lvl] = float(np.mean(vals)) if vals else overall
    return out


def bin_errors(
    levels: dict[str, str],
    anchors: dict[str, float],
    realized: dict[str, float],
) -> dict[str, float]:
    """Absolute bin error per group: |anchor(level) - realized|."""
    return {
        g: abs(anchors[lvl] - realized[g])
        for g, lvl in levels.items()
        if g in realized and lvl in anchors
    }


def mean_bin_error(errors: dict[str, float]) -> float:
    """Mean absolute bin error over groups."""
    if not errors:
        raise ValueError("errors must be non-empty")
    return float(np.mean(list(errors.values())))


def level_realized_means(levels: dict[str, str], realized: dict[str, float]) -> dict[str, float]:
    """Mean realized attrition per level (levels present in the data)."""
    agg: dict[str, list[float]] = {}
    for g, lvl in levels.items():
        if g in realized:
            agg.setdefault(lvl, []).append(realized[g])
    return {lvl: float(np.mean(v)) for lvl, v in agg.items() if v}


def monotonic_check(levels: dict[str, str], realized: dict[str, float]) -> bool:
    """Return True if mean realized attrition is non-decreasing by level."""
    means = level_realized_means(levels, realized)
    present = [lvl for lvl in LEVEL_ORDER if lvl in means]
    return all(means[present[i]] <= means[present[i + 1]] for i in range(len(present) - 1))


def paired_bootstrap_mae_delta(
    errors_a: dict[str, float],
    errors_b: dict[str, float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for MAE(A) - MAE(B), paired on shared units.

    Positive delta means arm B has *lower* MAE than arm A. Units are
    the keys (e.g. ``"seed:group"``) present in both error dicts.
    """
    keys = sorted(set(errors_a) & set(errors_b))
    if len(keys) < 4:
        raise ValueError(f"need >= 4 aligned units, got {len(keys)}")
    a = np.array([errors_a[k] for k in keys], dtype=float)
    b = np.array([errors_b[k] for k in keys], dtype=float)
    delta = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    m = len(keys)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return delta, float(lo), float(hi)
