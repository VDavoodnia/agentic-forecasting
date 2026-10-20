"""Uncertainty-quantification methods for TSFM forecasts (exp74).

Every method takes a *case* — one (origin, horizon-step) Chronos forecast —
plus the *bank* of past cases whose realizeds were strictly observable before
the current origin, and returns an adjusted quantile grid. All methods are
expressible as side-aware scale factors on the native grid around the median:

    q'_k = m + r_lo * (q_k - m)   for levels < 0.5
    q'_k = m + r_hi * (q_k - m)   for levels > 0.5

so interval coverage, interval width, and CRPS are directly comparable across
arms. PIT safety is enforced by the study driver, not by the methods: the
bank passed in contains only cases with (origin' + h' - 1) < origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Floor to avoid division by zero on degenerate (zero-width) intervals.
_EPS = 1e-12


@dataclass
class ForecastCase:
    """One (origin, horizon-step) forecast case."""

    origin: int  # origin index (series position; context = series[:origin])
    h: int  # horizon step, 1-based
    levels: np.ndarray  # (K,) quantile levels
    q: np.ndarray  # (K,) native quantiles at this step
    y: float  # realized value
    context: np.ndarray  # series[:origin]
    feats: dict[str, float] = field(default_factory=dict)

    @property
    def median(self) -> float:
        i = int(np.argmin(np.abs(self.levels - 0.5)))
        return float(self.q[i])

    def interval_edges(self) -> tuple[float, float, float]:
        """(median, q10, q90) on the native grid (closest levels)."""
        i10 = int(np.argmin(np.abs(self.levels - 0.1)))
        i90 = int(np.argmin(np.abs(self.levels - 0.9)))
        return self.median, float(self.q[i10]), float(self.q[i90])


def finite_quantile(x: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: ceil((m+1)*level)/m, 1-indexed."""
    xs = np.sort(np.asarray(x, dtype=float))
    m = len(xs)
    if m == 0:
        raise ValueError("finite_quantile needs at least one value")
    k = int(np.ceil((m + 1) * level)) - 1
    return float(xs[min(max(k, 0), m - 1)])


def weighted_quantile(x: np.ndarray, w: np.ndarray, level: float) -> float:
    """Weighted quantile: smallest x with cumulative weight >= level."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) == 0 or bool((w < 0).any()) or not bool((w > 0).any()):
        raise ValueError("weighted_quantile needs non-negative weights, some positive")
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) / ws.sum()
    return float(xs[int(np.searchsorted(cum, level, side="left"))])


def scale_grid(
    levels: np.ndarray,
    q: np.ndarray,
    r_lo: float,
    r_hi: float,
) -> np.ndarray:
    """Scale native quantiles around the median; re-enforce monotonicity."""
    lv = np.asarray(levels, dtype=float)
    qv = np.asarray(q, dtype=float)
    m = float(qv[int(np.argmin(np.abs(lv - 0.5)))])
    out = np.where(lv < 0.5, m + r_lo * (qv - m), m + r_hi * (qv - m))
    return np.maximum.accumulate(out)


def scales_from_interval(case: ForecastCase, lo_adj: float, hi_adj: float) -> tuple[float, float]:
    """Convert an adjusted [lo, hi] interval into (r_lo, r_hi) scale factors."""
    m, q10, q90 = case.interval_edges()
    r_lo = (m - lo_adj) / max(m - q10, _EPS)
    r_hi = (hi_adj - m) / max(q90 - m, _EPS)
    return max(r_lo, 0.0), max(r_hi, 0.0)


# ---------------------------------------------------------------------------
# Method arms: each takes (case, bank) and returns (r_lo, r_hi).
# ``bank`` is chronological and contains only PIT-safe known cases.
# ---------------------------------------------------------------------------


def arm_native(case: ForecastCase, bank: list[ForecastCase]) -> tuple[float, float]:
    """S1: passthrough baseline."""
    return 1.0, 1.0


def _needed_scales(case: ForecastCase) -> float:
    """Scale that would have put the realized value on the interval edge."""
    m, q10, q90 = case.interval_edges()
    if case.y >= m:
        return (case.y - m) / max(q90 - m, _EPS)
    return (m - case.y) / max(m - q10, _EPS)


def arm_fixed_expansion(
    case: ForecastCase, bank: list[ForecastCase], level: float = 0.8
) -> tuple[float, float]:
    """S2: single global scale = level-quantile of past needed scales."""
    needed = np.array([_needed_scales(c) for c in bank])
    if len(needed) < 8:
        return 1.0, 1.0
    r = finite_quantile(needed, level)
    return r, r


def _residuals_at_h(bank: list[ForecastCase], h: int) -> np.ndarray:
    return np.array([abs(c.y - c.median) for c in bank if c.h == h])


def arm_rolling_conformal(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    window: int = 24,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S3: EnbPI-style — symmetric interval from recent |residual| quantile."""
    res = _residuals_at_h(bank, case.h)[-window:]
    if len(res) < min_bank:
        return 1.0, 1.0
    qhat = finite_quantile(res, level)
    m = case.median
    return scales_from_interval(case, m - qhat, m + qhat)


def _aci_alpha_path(
    bank: list[ForecastCase],
    h: int,
    alpha: float,
    gamma: float,
    window: int,
) -> tuple[float, list[float]]:
    """Replay ACI over the chronological bank; return (current alpha, path)."""
    res = _residuals_at_h(bank, h)
    a_t = alpha
    path = [a_t]
    for i in range(len(res)):
        past = res[max(0, i - window) : i] if i > 0 else res[:0]
        if len(past) < 2:
            # Warm-up: no evidence yet — do not score a spurious miss.
            path.append(a_t)
            continue
        qhat = finite_quantile(past, 1.0 - a_t)
        err = 1.0 if res[i] > qhat else 0.0
        a_t = min(max(a_t + gamma * (alpha - err), 0.01), 0.99)
        path.append(a_t)
    return a_t, path


def arm_aci(
    case: ForecastCase,
    bank: list[ForecastCase],
    alpha: float = 0.2,
    gamma: float = 0.05,
    window: int = 24,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S4: Adaptive Conformal Inference — alpha adapts to coverage misses."""
    a_now, _ = _aci_alpha_path(bank, case.h, alpha, gamma, window)
    res = _residuals_at_h(bank, case.h)[-window:]
    if len(res) < min_bank:
        return 1.0, 1.0
    qhat = finite_quantile(res, 1.0 - a_now)
    m = case.median
    return scales_from_interval(case, m - qhat, m + qhat)


def arm_aci_bias(
    case: ForecastCase,
    bank: list[ForecastCase],
    alpha: float = 0.2,
    gamma: float = 0.05,
    window: int = 24,
    min_bank: int = 8,
    bias_lambda: float = 0.9,
) -> tuple[float, float]:
    """S4b: ACI + online EWM bias estimate shifting the interval center.

    BC-ACI (2026) insight: ACI cannot move the center, so under persistent
    bias it must widen by 2|b|. This arm estimates the bias online and
    re-centers before scoring — an *interval calibration*, not a forecast
    correction from external signals.
    """
    hbank = [c for c in bank if c.h == case.h]
    if len(hbank) < min_bank:
        return 1.0, 1.0
    b = 0.0
    for c in hbank:
        b = bias_lambda * b + (1.0 - bias_lambda) * (c.median - c.y)
    m_shifted = case.median + b
    res = np.array([abs(c.y - (c.median + b_now)) for c, b_now in _bias_track(hbank, bias_lambda)])
    res = res[-window:]
    a_now, _ = _aci_alpha_path(bank, case.h, alpha, gamma, window)
    qhat = finite_quantile(res, 1.0 - a_now)
    lo, hi = m_shifted - qhat, m_shifted + qhat
    # Express as scales on the native grid via the shifted center.
    m = case.median
    r_lo = (m - lo) / max(m - case.interval_edges()[1], _EPS)
    r_hi = (hi - m) / max(case.interval_edges()[2] - m, _EPS)
    return max(r_lo, 0.0), max(r_hi, 0.0)


def _bias_track(hbank: list[ForecastCase], bias_lambda: float) -> list[tuple[ForecastCase, float]]:
    """Chronological (case, bias-before-case) pairs for the EWM bias track."""
    out: list[tuple[ForecastCase, float]] = []
    b = 0.0
    for c in hbank:
        out.append((c, b))
        b = bias_lambda * b + (1.0 - bias_lambda) * (c.median - c.y)
    return out


def arm_weighted_conformal(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    rho: float = 0.97,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S5: residuals weighted by recency x volatility-similarity."""
    hbank = [c for c in bank if c.h == case.h]
    if len(hbank) < min_bank:
        return 1.0, 1.0
    vols = np.array([c.feats.get("vol_12", 0.0) for c in hbank])
    now_vol = case.feats.get("vol_12", 0.0)
    vol_scale = float(np.std(vols)) + _EPS
    ages = np.arange(len(hbank))[::-1].astype(float)  # 0 = most recent
    w = (rho**ages) / (1.0 + np.abs(vols - now_vol) / vol_scale)
    res = np.array([abs(c.y - c.median) for c in hbank])
    qhat = weighted_quantile(res, w, level)
    m = case.median
    return scales_from_interval(case, m - qhat, m + qhat)


def arm_cqr(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S6: conformalized quantile regression on the native [q10, q90].

    Exceedance scores s = max(q10 - y, y - q90, 0); widen by the
    (1-alpha) quantile of s. Asymmetric: each tail gets its own score.
    """
    hbank = [c for c in bank if c.h == case.h]
    if len(hbank) < min_bank:
        return 1.0, 1.0
    s_lo = np.array([max(c.interval_edges()[1] - c.y, 0.0) for c in hbank])
    s_hi = np.array([max(c.y - c.interval_edges()[2], 0.0) for c in hbank])
    m, q10, q90 = case.interval_edges()
    lo_adj = q10 - finite_quantile(s_lo, level)
    hi_adj = q90 + finite_quantile(s_hi, level)
    return scales_from_interval(case, lo_adj, hi_adj)


def _relative_scores(bank: list[ForecastCase], h: int) -> np.ndarray:
    """|y - m| / native half-width — scale-free nonconformity scores."""
    out = []
    for c in bank:
        if c.h != h:
            continue
        m, q10, q90 = c.interval_edges()
        out.append(abs(c.y - m) / max((q90 - q10) / 2.0, _EPS))
    return np.array(out)


def arm_conformal_rel(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    window: int = 24,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S3b (variant): rolling conformal on width-relative scores.

    Chronos's own width already grows with horizon/regime; the relative
    score isolates *how often reality exceeds its own stated uncertainty*,
    which transfers across regimes better than raw residuals.
    """
    s = _relative_scores(bank, case.h)[-window:]
    if len(s) < min_bank:
        return 1.0, 1.0
    s_star = finite_quantile(s, level)
    m, q10, q90 = case.interval_edges()
    half = s_star * max((q90 - q10) / 2.0, _EPS)
    return scales_from_interval(case, m - half, m + half)


def arm_aci_rel(
    case: ForecastCase,
    bank: list[ForecastCase],
    alpha: float = 0.2,
    gamma: float = 0.05,
    window: int = 24,
    min_bank: int = 8,
) -> tuple[float, float]:
    """S4c (variant): ACI on width-relative scores."""
    hbank = [c for c in bank if c.h == case.h]
    if len(hbank) < min_bank:
        return 1.0, 1.0
    s = _relative_scores(bank, case.h)
    a_t = alpha
    for i in range(len(s)):
        past = s[max(0, i - window) : i]
        if len(past) < 2:
            continue
        qhat = finite_quantile(past, 1.0 - a_t)
        err = 1.0 if s[i] > qhat else 0.0
        a_t = min(max(a_t + gamma * (alpha - err), 0.01), 0.99)
    recent = s[-window:]
    s_star = finite_quantile(recent, 1.0 - a_t)
    m, q10, q90 = case.interval_edges()
    half = s_star * max((q90 - q10) / 2.0, _EPS)
    return scales_from_interval(case, m - half, m + half)


# ---------------------------------------------------------------------------
# Error-percentile meta-models (S7 numeric, S8 numeric+story/macro).
# GBM with quantile loss predicts the 80th-percentile |error|; the
# prediction becomes the interval half-width.
# ---------------------------------------------------------------------------

META_FEATURES_NUMERIC = [
    "h",
    "log_width",
    "width_rel",
    "vol_6",
    "vol_12",
    "err_mean3",
    "err_mean6",
    "err_count3",
    "err_count6",
    "miss_streak",
    "regime_shift",
    "slope_6",
    "width_growth",
]

META_FEATURES_STORY = [
    "st_available",
    "st_shift",
    "st_direction",
    "st_magnitude",
    "st_months_from_now",
    "st_timing_conf",
    "st_trend_count",
    "mac_available",
    "mac_n_high",
    "mac_max_abs_dir",
    "mac_mean_cert",
    "mac_inflation",
    "mac_rates",
    "mac_food",
    "mac_labour",
]

MIN_META_BANK = 40


def _meta_matrix(cases: list[ForecastCase], features: list[str]) -> np.ndarray:
    return np.array([[c.feats.get(f, 0.0) for f in features] for c in cases])


def _fit_meta(
    bank: list[ForecastCase], features: list[str], quantile: float = 0.8
) -> object | None:
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    if len(bank) < MIN_META_BANK:
        return None
    x = _meta_matrix(bank, features)
    z = np.array([abs(c.y - c.median) for c in bank])
    if not np.isfinite(x).all() or len(np.unique(z)) < 2:
        return None
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=200,
        early_stopping=True,
        random_state=0,
    )
    model.fit(x, z)
    return model


def fit_meta_model(
    bank: list[ForecastCase], features: list[str], quantile: float = 0.8
) -> object | None:
    """Fit the GBM error-percentile model on a PIT-safe bank.

    Returns None when the bank is too small or degenerate (callers fall
    back to rolling conformal).
    """
    return _fit_meta(bank, features, quantile)


def meta_matrix(cases: list[ForecastCase], features: list[str]) -> np.ndarray:
    """Public feature-matrix builder for fitted meta-models."""
    return _meta_matrix(cases, features)


def arm_meta(
    case: ForecastCase,
    bank: list[ForecastCase],
    features: list[str] = META_FEATURES_NUMERIC,
    quantile: float = 0.8,
) -> tuple[float, float]:
    """S7/S8: predict the quantile-|error| as the interval half-width."""
    model = fit_meta_model(bank, features, quantile)
    if model is None:
        # Fallback: rolling conformal on the bank.
        return arm_rolling_conformal(case, bank)
    pred = float(model.predict(meta_matrix([case], features))[0])
    pred = max(pred, _EPS)
    m, q10, q90 = case.interval_edges()
    r_lo = pred / max(m - q10, _EPS)
    r_hi = pred / max(q90 - m, _EPS)
    return max(r_lo, 0.0), max(r_hi, 0.0)
