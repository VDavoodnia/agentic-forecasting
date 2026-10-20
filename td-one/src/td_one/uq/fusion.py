"""exp76: fuse CQR with the meta uncertainty signal (exp74's two wins).

exp74 showed CQR is the coverage-honest UQ layer and the story/meta
signal improves CRPS but under-covers (it shrinks toward Chronos's
calm-times conservatism). The two fusion designs here keep CQR's
conformal machinery and let the meta signal steer *how much* conformal
correction a given origin gets:

F1 ``MetaLevelCqr`` — meta-informed conformal level. The meta GBM's
  predicted error scale ranks the current origin among past origins
  (predicted-error percentile); the CQR level is modulated around the
  nominal level by that percentile, with the modulation strength k
  tuned on the bank (single scalar, CRPS on a chronological replay).
F2 ``ConditionalCqr`` — feature-conditional CQR. Two GBM quantile
  regressors predict the CQR exceedance scores (s_lo, s_hi) conditional
  on the case's features; the predicted conditional score quantile is
  the per-origin conformal offset. Unlike exp74's conformalized-meta
  (which scaled the interval by the *raw* predicted error and
  double-counted the GBM's shrinkage), here the model predicts the
  conformal score itself.

Plus ``arm_aci_gated``: exp74's conditional ACI win made deployable —
ACI runs only when recent evidence says Chronos is miscalibrating
(miss streak or a volatility spike), native otherwise.

PIT safety is inherited from the study driver: the bank contains only
cases observable before the origin, and the meta/score models are fit
on that bank alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from td_one.uq.methods import (
    MIN_META_BANK,
    ForecastCase,
    arm_aci,
    arm_cqr,
    arm_native,
    finite_quantile,
    fit_meta_model,
    meta_matrix,
    scale_grid,
    scales_from_interval,
)

_EPS = 1e-12


def month_before(ym: str) -> str:
    """Previous ``YYYY-MM`` — the PIT-safe feature month for an origin.

    A forecast origin whose first forecast month is ``ym`` has last
    *observed* month ``month_before(ym)``; story/macro features must be
    keyed to that month, never to ``ym`` itself.
    """
    y, m = int(ym[:4]), int(ym[5:7])
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def modulated_level(rank_pct: float, level: float, k: float) -> float:
    """CQR level modulated by predicted-error percentile.

    rank 0.5 (median predicted uncertainty) -> the nominal level;
    rank 1 -> level + k; rank 0 -> level - k. Clipped to [0.5, 0.95].
    """
    return float(min(max(level + k * (2.0 * rank_pct - 1.0), 0.5), 0.95))


def _exceedance_scores(bank: list[ForecastCase], h: int) -> tuple[np.ndarray, np.ndarray]:
    """(s_lo, s_hi) CQR exceedance scores of the bank at horizon h."""
    hbank = [c for c in bank if c.h == h]
    s_lo = np.array([max(c.interval_edges()[1] - c.y, 0.0) for c in hbank])
    s_hi = np.array([max(c.y - c.interval_edges()[2], 0.0) for c in hbank])
    return s_lo, s_hi


def _scales_from_scores(
    case: ForecastCase, s_lo: np.ndarray, s_hi: np.ndarray, level: float
) -> tuple[float, float]:
    """CQR scales from precomputed bank score arrays at ``level``."""
    if len(s_lo) == 0:
        return 1.0, 1.0
    m, q10, q90 = case.interval_edges()
    lo_adj = q10 - finite_quantile(s_lo, level)
    hi_adj = q90 + finite_quantile(s_hi, level)
    return scales_from_interval(case, lo_adj, hi_adj)


class MetaLevelCqr:
    """F1: CQR whose conformal level is modulated by the meta signal.

    The meta model (exp74's GBM error-percentile model) is fit on the
    bank; the current case's predicted error scale is ranked among the
    bank's same-horizon predictions to get a predicted-error
    percentile, which modulates the CQR level (see
    :func:`modulated_level`). The modulation strength k is picked from
    ``k_grid`` by CRPS on a chronological replay of the most recent
    bank cases (their scores come from the same bank — a single scalar
    tuned in-sample, deliberately small). Falls back to plain CQR when
    the bank is too small for the meta model.
    """

    def __init__(
        self,
        features: list[str],
        level: float = 0.8,
        k_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25),
        min_meta_bank: int = MIN_META_BANK,
        replay_n: int = 48,
        min_h: int = 0,
        widen_only: bool = False,
    ) -> None:
        self.features = features
        self.level = level
        self.k_grid = k_grid
        self.min_meta_bank = min_meta_bank
        self.replay_n = replay_n
        # Round-2 variants (exp76 diagnosis): the level modulation's
        # Food CPI win concentrates at longer horizons and its synth
        # harm in shift scenarios comes from narrowing already-tight
        # intervals; min_h gates short horizons to plain CQR and
        # widen_only floors the modulated level at the nominal level.
        self.min_h = min_h
        self.widen_only = widen_only
        self._key: tuple[int, int] | None = None
        self._model: Any = None
        self._preds: np.ndarray | None = None
        self._scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._k: float = 0.0

    def _refresh(self, case: ForecastCase, bank: list[ForecastCase]) -> None:
        key = (case.origin, len(bank))
        if key == self._key:
            return
        self._key = key
        self._scores = {h: _exceedance_scores(bank, h) for h in {c.h for c in bank}}
        self._model = None
        self._preds = None
        self._k = 0.0
        if len(bank) < self.min_meta_bank:
            return
        model = fit_meta_model(bank, self.features)
        if model is None:
            return
        self._model = model
        self._preds = np.asarray(self._model.predict(meta_matrix(bank, self.features)), dtype=float)
        self._k = self._tune_k(bank)

    def _rank(self, pred: float, h: int, bank: list[ForecastCase]) -> float:
        assert self._preds is not None
        same_h = self._preds[[i for i, c in enumerate(bank) if c.h == h]]
        if len(same_h) == 0:
            return 0.5
        return float(np.mean(same_h <= pred))

    def _tune_k(self, bank: list[ForecastCase]) -> float:
        if len(self.k_grid) == 1:
            return float(self.k_grid[0])
        from td_one.evaluation.real_benchmark import crps_from_quantiles  # noqa: PLC0415

        assert self._preds is not None
        replay = bank[-self.replay_n :]
        best_k, best_crps = float(self.k_grid[0]), float("inf")
        for k in self.k_grid:
            total = 0.0
            for j, b in enumerate(replay):
                i = len(bank) - len(replay) + j
                r = self._rank(float(self._preds[i]), b.h, bank)
                lvl = modulated_level(r, self.level, float(k))
                s_lo, s_hi = self._scores.get(b.h, (np.array([]), np.array([])))
                r_lo, r_hi = _scales_from_scores(b, s_lo, s_hi, lvl)
                grid = scale_grid(b.levels, b.q, r_lo, r_hi)
                total += float(crps_from_quantiles(b.levels, grid, np.array([b.y])))
            if total < best_crps:
                best_crps, best_k = total, float(k)
        return best_k

    def __call__(self, case: ForecastCase, bank: list[ForecastCase]) -> tuple[float, float]:
        if case.h < self.min_h:
            return arm_cqr(case, bank, level=self.level)
        self._refresh(case, bank)
        if self._model is None:
            return arm_cqr(case, bank, level=self.level)
        pred = float(self._model.predict(meta_matrix([case], self.features))[0])
        r = self._rank(pred, case.h, bank)
        lvl = modulated_level(r, self.level, self._k)
        if self.widen_only:
            lvl = max(lvl, self.level)
        s_lo, s_hi = self._scores.get(case.h, (np.array([]), np.array([])))
        if len(s_lo) < 8:
            return arm_cqr(case, bank, level=lvl)
        return _scales_from_scores(case, s_lo, s_hi, lvl)


def _fit_score_model(
    bank: list[ForecastCase], features: list[str], targets: np.ndarray, quantile: float
) -> Any:
    """GBM quantile regressor for a CQR score target; None if degenerate."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped] # noqa: PLC0415

    if len(bank) < MIN_META_BANK or len(np.unique(targets)) < 2:
        return None
    x = meta_matrix(bank, features)
    if not np.isfinite(x).all():
        return None
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=200,
        early_stopping=True,
        random_state=0,
    )
    model.fit(x, targets)
    return model


class ConditionalCqr:
    """F2: CQR with feature-conditional exceedance-score quantiles.

    Fits two GBM quantile regressors on the bank — one for the downside
    exceedance score s_lo, one for the upside s_hi — conditional on the
    case features (numeric + story/macro when present). The predicted
    conditional (level)-quantile of each score is the per-origin
    conformal offset. Falls back to plain CQR when the bank is too
    small or both score targets are degenerate (no exceedances yet).
    """

    def __init__(
        self,
        features: list[str],
        level: float = 0.8,
        min_bank: int = MIN_META_BANK,
        shrink: float = 1.0,
    ) -> None:
        self.features = features
        self.level = level
        self.min_bank = min_bank
        # Round-2 variant: shrink<1 partially pools the conditional
        # offset toward the global (unconditional) CQR offset —
        # off = global + shrink * (pred - global). shrink=1.0 is the
        # round-1 pure-conditional design; shrink=0.0 is plain CQR.
        self.shrink = shrink
        self._key: tuple[int, int] | None = None
        self._lo: Any = None
        self._hi: Any = None
        self._g_lo = 0.0
        self._g_hi = 0.0

    def _refresh(self, case: ForecastCase, bank: list[ForecastCase]) -> None:
        key = (case.origin, len(bank))
        if key == self._key:
            return
        self._key = key
        self._lo = self._hi = None
        if len(bank) < self.min_bank:
            return
        s_lo = np.array([max(c.interval_edges()[1] - c.y, 0.0) for c in bank])
        s_hi = np.array([max(c.y - c.interval_edges()[2], 0.0) for c in bank])
        self._g_lo = finite_quantile(s_lo, self.level)
        self._g_hi = finite_quantile(s_hi, self.level)
        if len(np.unique(s_lo)) < 2 and len(np.unique(s_hi)) < 2:
            return
        self._lo = _fit_score_model(bank, self.features, s_lo, self.level)
        self._hi = _fit_score_model(bank, self.features, s_hi, self.level)

    def _offset(self, model: Any, x: np.ndarray, global_q: float) -> float:
        if model is None:
            return global_q if self.shrink < 1.0 else 0.0
        pred = max(float(model.predict(x)[0]), 0.0)
        return global_q + self.shrink * (pred - global_q)

    def __call__(self, case: ForecastCase, bank: list[ForecastCase]) -> tuple[float, float]:
        self._refresh(case, bank)
        if self._lo is None and self._hi is None:
            return arm_cqr(case, bank, level=self.level)
        x = meta_matrix([case], self.features)
        off_lo = self._offset(self._lo, x, self._g_lo)
        off_hi = self._offset(self._hi, x, self._g_hi)
        _m, q10, q90 = case.interval_edges()
        return scales_from_interval(case, q10 - off_lo, q90 + off_hi)


def arm_aci_gated(
    case: ForecastCase,
    bank: list[ForecastCase],
    miss_threshold: int = 2,
    vol_ratio: float = 1.5,
    min_bank: int = 8,
    alpha: float = 0.2,
    gamma: float = 0.05,
    window: int = 24,
) -> tuple[float, float]:
    """Regime-gated ACI (exp74 conditional win, deployable form).

    Runs ACI only when the recent bank says Chronos is miscalibrating:
    at least ``miss_threshold`` of the last 6 same-horizon cases missed
    their native 80% interval, or the case's vol_12 exceeds
    ``vol_ratio`` x the bank's median vol_12. Native otherwise — the
    gate is what keeps ACI from hurting on calibrated scopes, where
    exp74 showed it slightly loses.
    """
    hbank = [c for c in bank if c.h == case.h]
    if len(hbank) < min_bank:
        return arm_native(case, bank)
    recent = hbank[-6:]
    misses = sum(1.0 for c in recent if c.y < c.interval_edges()[1] or c.y > c.interval_edges()[2])
    vols = np.array([c.feats.get("vol_12", 0.0) for c in hbank])
    med_vol = float(np.median(vols)) if len(vols) else 0.0
    vol_hot = med_vol > _EPS and case.feats.get("vol_12", 0.0) > vol_ratio * med_vol
    if misses >= miss_threshold or vol_hot:
        return arm_aci(case, bank, alpha=alpha, gamma=gamma, window=window)
    return arm_native(case, bank)
