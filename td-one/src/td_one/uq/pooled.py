"""exp81: cross-series pooled conformal banks for the UQ layer.

exp76's sweep found per-series bank size is the binding constraint on
every conformal UQ arm: sweep series contribute 8–12 origins each and,
with origins spaced 6 apart and horizons of 18–24, the same-horizon
bank at most origins is far smaller still — on M4, CQR's ``min_bank``
gate never fired and every conformal arm was an exact no-op.

This module pools CQR exceedance scores **across series within a
benchmark family**, with side-specific width normalization so
heterogeneous series can share a bank:

    s_lo_norm = s_lo / (m - q10),   s_hi_norm = s_hi / (q90 - m)

(the CQR analogue of exp74's relative-score conformal). The pooled
finite-sample quantile of the normalized scores is rescaled by the
target case's own side half-widths.

PIT safety has two parts, both structural:

- *Own bank* — exp74's exact rule within the target series: only cases
  with ``origin' + h' - 1 < origin``.
- *Cross bank* — an evaluation-round lag rule. ``round(x)`` is the
  index of x's origin in its own config's sorted origin list and
  ``step`` is the family's origin spacing. A cross case enters iff
  ``(round(target) - round(case)) * step >= h'``: the pooled case is
  fully realized within its own series' timeline at least as many
  rounds before the target round as its horizon requires. The Monash
  sweep subsets are not calendar-aligned across series (their series
  end in different decades), so what pooling borrows is calibration
  information — how Chronos's intervals err on other series — never
  temporal information about the target series, whose own PIT rule is
  preserved exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from td_one.evaluation.real_benchmark import crps_from_quantiles
from td_one.uq.evaluate import CaseResult, known_bank
from td_one.uq.methods import (
    ForecastCase,
    arm_cqr,
    finite_quantile,
    scale_grid,
    scales_from_interval,
    weighted_quantile,
)

_EPS = 1e-12

#: A pooled arm sees the target case plus its own-series bank and its
#: cross-series bank, kept separate so shrinkage arms can weight them.
PooledArmFn = Callable[[ForecastCase, list[ForecastCase], list[ForecastCase]], tuple[float, float]]


def local_arm(fn: Callable[[ForecastCase, list[ForecastCase]], tuple[float, float]]) -> PooledArmFn:
    """Adapt a per-series ``(case, bank)`` arm to the pooled signature."""

    def _arm(
        case: ForecastCase, own: list[ForecastCase], cross: list[ForecastCase]
    ) -> tuple[float, float]:
        return fn(case, own)

    return _arm


def _rounds_of(cases: list[ForecastCase]) -> dict[int, int]:
    return {o: i for i, o in enumerate(sorted({c.origin for c in cases}))}


def _family_step(cases_by_config: dict[str, list[ForecastCase]]) -> int:
    """Origin spacing shared by every config in a family (asserted)."""
    steps: set[int] = set()
    n_origins: set[int] = set()
    for cases in cases_by_config.values():
        origins = sorted({c.origin for c in cases})
        n_origins.add(len(origins))
        steps.update(int(b - a) for a, b in zip(origins, origins[1:], strict=False))
    if len(steps) != 1 or len(n_origins) != 1:
        raise ValueError(
            f"pooled families need a uniform origin grid; steps={steps}, counts={n_origins}"
        )
    return steps.pop()


class PooledBankIndex:
    """Precomputed own/cross banks for one benchmark family.

    All configs in the family must share the same origin grid shape
    (same number of rounds, same spacing) — true by construction for
    the exp76 sweep case sets, asserted here.
    """

    def __init__(self, cases_by_config: dict[str, list[ForecastCase]]) -> None:
        self.step = _family_step(cases_by_config)
        self.cases_by_config = cases_by_config
        self._rounds = {cid: _rounds_of(cases) for cid, cases in cases_by_config.items()}
        # First target round (in any config) at which each cross case is
        # admissible: round(case) + ceil(h / step).
        self._avail: dict[str, list[tuple[int, ForecastCase]]] = {}
        for cid, cases in cases_by_config.items():
            rmap = self._rounds[cid]
            self._avail[cid] = sorted(
                ((rmap[c.origin] + -(-c.h // self.step), c) for c in cases),
                key=lambda t: t[0],
            )
        self._cache: dict[tuple[str, int], tuple[list[ForecastCase], list[ForecastCase]]] = {}

    def banks(self, config_id: str, origin: int) -> tuple[list[ForecastCase], list[ForecastCase]]:
        """(own bank, cross bank) for a target at ``origin`` in ``config_id``."""
        key = (config_id, origin)
        if key in self._cache:
            return self._cache[key]
        own = known_bank(self.cases_by_config[config_id], origin)
        r = self._rounds[config_id][origin]
        cross: list[ForecastCase] = []
        for ocid, avail in self._avail.items():
            if ocid == config_id:
                continue
            cross.extend(c for a, c in avail if a <= r)
        out = (own, cross)
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _raw_scores(bank: list[ForecastCase], h: int) -> tuple[np.ndarray, np.ndarray]:
    """CQR exceedance scores (s_lo, s_hi) of bank cases at horizon h."""
    hb = [c for c in bank if c.h == h]
    s_lo = np.array([max(c.interval_edges()[1] - c.y, 0.0) for c in hb])
    s_hi = np.array([max(c.y - c.interval_edges()[2], 0.0) for c in hb])
    return s_lo, s_hi


def _norm_scores(bank: list[ForecastCase], h: int) -> tuple[np.ndarray, np.ndarray]:
    """Width-normalized scores: exceedance / the source case's side half-width."""
    hb = [c for c in bank if c.h == h]
    s_lo = np.array(
        [
            max(c.interval_edges()[1] - c.y, 0.0) / max(c.median - c.interval_edges()[1], _EPS)
            for c in hb
        ]
    )
    s_hi = np.array(
        [
            max(c.y - c.interval_edges()[2], 0.0) / max(c.interval_edges()[2] - c.median, _EPS)
            for c in hb
        ]
    )
    return s_lo, s_hi


def _scales(case: ForecastCase, off_lo: float, off_hi: float) -> tuple[float, float]:
    _m, q10, q90 = case.interval_edges()
    return scales_from_interval(case, q10 - off_lo, q90 + off_hi)


def _signed_scores(bank: list[ForecastCase], h: int, mode: str) -> np.ndarray:
    """Signed CQR max-score ``max(q10 - y, y - q90)`` per bank case.

    Unlike the per-side exceedances above, the signed score is negative
    when the realized value lands inside the interval, so its
    distribution is not zero-inflated and its (1-alpha) quantile is
    informative even when exceedance rates are low (round-1 diagnosis:
    per-side quantiles at level 0.8 are exactly 0 whenever a side's
    exceedance rate is under 20%).

    ``mode``: "raw" (series units), "width" (divided by the binding
    side's half-width), or "resid" (divided by the source case's
    trailing residual scale, ``feats["resid_scale_h"]`` — see
    :func:`fill_resid_scales`).
    """
    out = []
    for c in bank:
        if c.h != h:
            continue
        m, q10, q90 = c.interval_edges()
        s = max(q10 - c.y, c.y - q90)
        if mode == "raw":
            out.append(s)
        elif mode == "width":
            half = (m - q10) if (q10 - c.y) >= (c.y - q90) else (q90 - m)
            out.append(s / max(half, _EPS))
        elif mode == "resid":
            scale = c.feats.get("resid_scale_h", (q90 - q10) / 2.0)
            out.append(s / max(scale, _EPS))
        else:
            raise ValueError(f"unknown signed-score mode: {mode}")
    return np.array(out)


def fill_resid_scales(cases_by_config: dict[str, list[ForecastCase]]) -> None:
    """Per-case trailing residual scale, PIT-safe within each series.

    ``feats["resid_scale_h"]`` = median |y - median| over the config's
    same-h cases observable before the case's origin (exp74's rule);
    falls back to all-h observable cases, then to the case's own mean
    half-width. Used by the residual-normalized pooled arm.
    """
    for cases in cases_by_config.values():
        for case in cases:
            prior = [c for c in cases if c.h == case.h and c.origin + c.h - 1 < case.origin]
            if not prior:
                prior = [c for c in cases if c.origin + c.h - 1 < case.origin]
            if prior:
                scale = float(np.median([abs(c.y - c.median) for c in prior]))
            else:
                _m, q10, q90 = case.interval_edges()
                scale = (q90 - q10) / 2.0
            case.feats["resid_scale_h"] = max(scale, _EPS)


# ---------------------------------------------------------------------------
# Pooled arms
# ---------------------------------------------------------------------------


def arm_pooled_cqr(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
) -> tuple[float, float]:
    """C: CQR over own ∪ cross bank with width-normalized scores."""
    s_lo, s_hi = _norm_scores(own + cross, case.h)
    if len(s_lo) < min_bank:
        return 1.0, 1.0
    m, q10, q90 = case.interval_edges()
    off_lo = finite_quantile(s_lo, level) * max(m - q10, _EPS)
    off_hi = finite_quantile(s_hi, level) * max(q90 - m, _EPS)
    return _scales(case, off_lo, off_hi)


def arm_pooled_raw(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
) -> tuple[float, float]:
    """C-raw: pooled CQR on raw scores (normalization control arm)."""
    return arm_cqr(case, own + cross, level=level, min_bank=min_bank)


def arm_pooled_shrink(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    n0: float = 8.0,
) -> tuple[float, float]:
    """D: pooled bank shrunk toward the series' own scores by bank size.

    Own scores enter in raw units with total weight
    ``lambda = n_own / (n_own + n0)``; cross scores are
    width-normalized, rescaled to the target's units, and carry total
    weight ``1 - lambda``. ``n_own = 0`` gives the pure pooled arm;
    ``n_cross = 0`` gives (up to the weighted-quantile convention) the
    per-series arm.
    """
    own_lo, own_hi = _raw_scores(own, case.h)
    crs_lo, crs_hi = _norm_scores(cross, case.h)
    n_own, n_cross = len(own_lo), len(crs_lo)
    if n_own + n_cross < min_bank:
        return 1.0, 1.0
    lam = n_own / (n_own + n0) if n_own > 0 else 0.0
    m, q10, q90 = case.interval_edges()

    def _mix(own_s: np.ndarray, cross_norm: np.ndarray, half: float) -> float:
        vals: list[np.ndarray] = []
        wts: list[np.ndarray] = []
        if n_own:
            vals.append(own_s)
            wts.append(np.full(n_own, lam / n_own))
        if n_cross:
            vals.append(cross_norm * max(half, _EPS))
            wts.append(np.full(n_cross, (1.0 - lam) / n_cross))
        return weighted_quantile(np.concatenate(vals), np.concatenate(wts), level)

    off_lo = _mix(own_lo, crs_lo, m - q10)
    off_hi = _mix(own_hi, crs_hi, q90 - m)
    return _scales(case, off_lo, off_hi)


# ---------------------------------------------------------------------------
# Round 2 arms: signed-score pooling (round-1 diagnosis fix)
# ---------------------------------------------------------------------------


def arm_pooled_cqr_signed(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
) -> tuple[float, float]:
    """C2: pooled CQR on signed max-scores, width-normalized.

    The pooled quantile (in binding-half-width units) is applied per
    side with the target's own side half-widths; a negative quantile
    narrows the interval, as textbook CQR allows.
    """
    s = _signed_scores(own + cross, case.h, "width")
    if len(s) < min_bank:
        return 1.0, 1.0
    q = finite_quantile(s, level)
    m, q10, q90 = case.interval_edges()
    return _scales(case, q * max(m - q10, _EPS), q * max(q90 - m, _EPS))


def arm_pooled_cqr_resid(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
) -> tuple[float, float]:
    """C2-resid: pooled signed CQR, residual-scale normalization (R2 ablation)."""
    s = _signed_scores(own + cross, case.h, "resid")
    if len(s) < min_bank:
        return 1.0, 1.0
    q = finite_quantile(s, level)
    _m, q10, q90 = case.interval_edges()
    scale = case.feats.get("resid_scale_h", (q90 - q10) / 2.0)
    return _scales(case, q * max(scale, _EPS), q * max(scale, _EPS))


def arm_pooled_shrink_signed(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    n0: float = 8.0,
) -> tuple[float, float]:
    """D2: signed-score shrinkage — own scores vs pooled scores by bank size.

    Everything is expressed in units of the target's mean half-width:
    own signed scores are divided by it, cross scores are already
    width-normalized. Own total weight ``lambda = n_own/(n_own+n0)``,
    cross the remainder; the mixed quantile is applied per side with
    the target's side half-widths.
    """
    own_s = _signed_scores(own, case.h, "raw")
    crs_s = _signed_scores(cross, case.h, "width")
    n_own, n_cross = len(own_s), len(crs_s)
    if n_own + n_cross < min_bank:
        return 1.0, 1.0
    m, q10, q90 = case.interval_edges()
    mean_half = max((q90 - q10) / 2.0, _EPS)
    lam = n_own / (n_own + n0) if n_own > 0 else 0.0
    vals: list[np.ndarray] = []
    wts: list[np.ndarray] = []
    if n_own:
        vals.append(own_s / mean_half)
        wts.append(np.full(n_own, lam / n_own))
    if n_cross:
        vals.append(crs_s)
        wts.append(np.full(n_cross, (1.0 - lam) / n_cross))
    q = weighted_quantile(np.concatenate(vals), np.concatenate(wts), level)
    return _scales(case, q * max(m - q10, _EPS), q * max(q90 - m, _EPS))


# ---------------------------------------------------------------------------
# Round 3/4 arms: starvation gating + similarity gating (pre-registered
# diagnosis rounds R4/R5 — pooling hurt exactly where the own bank was
# already fed (OSFI) and where cross series are dissimilar in scale).
# ---------------------------------------------------------------------------


def arm_pooled_shrink_gated(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    n0: float = 8.0,
    gate_n: int = 24,
) -> tuple[float, float]:
    """D3: signed shrinkage, but only where the own bank is starved.

    When the series' own same-h bank reaches ``gate_n`` cases, the
    per-series answer (arm B) is already reliable — pooling can only
    import foreign miscalibration (round-1/2 OSFI diagnosis) — so the
    arm returns B exactly. Below the gate it is D2.
    """
    n_own_h = sum(1 for c in own if c.h == case.h)
    if n_own_h >= gate_n:
        return arm_cqr(case, own, level=level, min_bank=min_bank)
    return arm_pooled_shrink_signed(case, own, cross, level=level, min_bank=min_bank, n0=n0)


def _similar_cross(
    case: ForecastCase, cross: list[ForecastCase], kappa: float
) -> list[ForecastCase]:
    """Cross cases whose source series' residual scale is within ``kappa``.

    Uses each case's PIT-safe trailing residual scale
    (``feats["resid_scale_h"]``, filled by :func:`fill_resid_scales`).
    """
    _m, q10, q90 = case.interval_edges()
    target_scale = case.feats.get("resid_scale_h", (q90 - q10) / 2.0)
    if target_scale <= _EPS:
        return cross
    out = []
    for c in cross:
        s = c.feats.get("resid_scale_h")
        if s is None or s <= _EPS:
            out.append(c)
            continue
        ratio = s / target_scale
        if 1.0 / kappa <= ratio <= kappa:
            out.append(c)
    return out


def arm_pooled_shrink_sim(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    n0: float = 8.0,
    kappa: float = 3.0,
) -> tuple[float, float]:
    """D4: signed shrinkage over similarity-gated cross banks (R4)."""
    return arm_pooled_shrink_signed(
        case, own, _similar_cross(case, cross, kappa), level=level, min_bank=min_bank, n0=n0
    )


def arm_pooled_shrink_full(
    case: ForecastCase,
    own: list[ForecastCase],
    cross: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    n0: float = 8.0,
    gate_n: int = 24,
    kappa: float = 3.0,
) -> tuple[float, float]:
    """D5: starvation gate (R5) + similarity gate (R4) + signed shrinkage."""
    n_own_h = sum(1 for c in own if c.h == case.h)
    if n_own_h >= gate_n:
        return arm_cqr(case, own, level=level, min_bank=min_bank)
    return arm_pooled_shrink_sim(
        case, own, cross, level=level, min_bank=min_bank, n0=n0, kappa=kappa
    )


# ---------------------------------------------------------------------------
# Study runner (mirrors evaluate.run_study, with pooled banks)
# ---------------------------------------------------------------------------


def _edge(q: np.ndarray, levels: np.ndarray, target: float) -> float:
    return float(q[int(np.argmin(np.abs(levels - target)))])


def run_study_pooled(
    cases_by_family: dict[str, dict[str, list[ForecastCase]]],
    arms: dict[str, PooledArmFn],
) -> list[CaseResult]:
    """Run pooled arms over every case with PIT-safe own + cross banks."""
    results: list[CaseResult] = []
    for _family, cases_by_config in sorted(cases_by_family.items()):
        index = PooledBankIndex(cases_by_config)
        fill_resid_scales(cases_by_config)
        for cid, cases in sorted(cases_by_config.items()):
            for case in sorted(cases, key=lambda c: (c.origin, c.h)):
                own, cross = index.banks(cid, case.origin)
                native_w = max(
                    _edge(case.q, case.levels, 0.9) - _edge(case.q, case.levels, 0.1),
                    1e-12,
                )
                crps_native = float(crps_from_quantiles(case.levels, case.q, np.array([case.y])))
                for arm_name, arm in arms.items():
                    r_lo, r_hi = arm(case, own, cross)
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
