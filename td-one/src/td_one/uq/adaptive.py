"""Shift-adaptive conformal arms (exp85).

exp74 and exp80a hit the same wall twice: a conformal bank is
structurally blind at shift onset, because every score in the bank
was realized *pre-shift* and is therefore small. These arms make the
bank adapt when the world changes, via two mechanisms:

(a) **Changepoint-triggered bank surgery** — detect the break from
    data visible at the origin (Layer-2 ``changepoint_scan`` as the
    gate, ``shift_guard.cusum_detect`` for the onset estimate), then
    reset the bank to post-break scores (``arm_cqr_reset``),
    exponentially down-weight old scores (``arm_cqr_decay``), or
    both (``arm_cqr_cp``: decay + an extra penalty on pre-break
    scores).

(b) **Bias-corrected conformal (BC-ACI family)** — an online EWM
    estimate of the signed forecast error, tracked causally
    (bias-before-case), corrects the CQR exceedance scores and
    shifts the current interval, so a persistent level shift is met
    by *moving* the interval instead of only widening it
    (``arm_bc_cqr``; ``arm_bc_aci`` adds ACI level adaptation on the
    corrected scores; ``arm_cp_bc`` combines reset + BC).

All arms take ``(case, bank)`` like the exp74 methods and return
``(r_lo, r_hi)`` scale factors on the native grid. PIT safety: the
bank is pre-filtered by the study driver (``origin' + h' - 1 <
origin``); detection reads only ``case.context`` (the series up to
the origin); bias tracks never use a case's own realized value.
"""

from __future__ import annotations

import numpy as np

from td_one.uq.methods import (
    ForecastCase,
    finite_quantile,
    scales_from_interval,
    weighted_quantile,
)

_EPS = 1e-12


def detect_break_onset(
    context: np.ndarray,
    *,
    lookback: int = 12,
    baseline: int = 24,
    cusum_threshold: float = 5.0,
    gate: str = "scan",
) -> int | None:
    """Estimated first post-shift index within ``context``, or None.

    Gate ``"scan"`` (default): the Layer-2 ``changepoint_scan`` over
    the trailing ``baseline + lookback`` window must fire; the onset
    is then ``cusum_detect``'s estimate on the same window (its
    baseline is the window's first ``baseline`` points), falling
    back to the start of the lookback window when CUSUM does not
    confirm. Gate ``"cusum"`` (exp85 round 2): CUSUM alone decides —
    the scan's triple gate (level z AND innovation z AND vol/trend)
    proved too conservative for gradual (ramp) shifts on noisy
    series, where it essentially never fired; the CUSUM gate trades
    some false-positive risk for actually engaging. PIT-pure either
    way: only ``context`` (data up to the origin) is read.
    """
    from td_one.agent.layer2_tools import changepoint_scan  # noqa: PLC0415
    from td_one.agent.shift_guard import GuardConfig, cusum_detect  # noqa: PLC0415

    if gate not in ("scan", "cusum"):
        raise ValueError(f"gate must be 'scan' or 'cusum', got {gate!r}")
    x = np.asarray(context, dtype=float).ravel()
    if len(x) < baseline + 3:
        return None
    window = x[-(baseline + lookback) :]
    offset = len(x) - len(window)
    if gate == "scan":
        scan = changepoint_scan(x, lookback=lookback, baseline=baseline)
        if not bool(scan["break_flag"]):
            return None
    det = cusum_detect(
        window,
        asof=len(window) - 1,
        config=GuardConfig(baseline_window=baseline, cusum_threshold_h=cusum_threshold),
    )
    if det.detected and det.onset_idx is not None:
        return offset + int(det.onset_idx)
    return offset + baseline if gate == "scan" else None


def _hbank(bank: list[ForecastCase], h: int) -> list[ForecastCase]:
    """Chronological bank cases at one horizon step."""
    return sorted((c for c in bank if c.h == h), key=lambda c: c.origin)


def _realized_time(case: ForecastCase) -> int:
    return case.origin + case.h - 1


def _side_scores(cases: list[ForecastCase]) -> tuple[np.ndarray, np.ndarray]:
    """CQR exceedance scores (s_lo, s_hi) for a list of cases."""
    s_lo = np.array([max(c.interval_edges()[1] - c.y, 0.0) for c in cases])
    s_hi = np.array([max(c.y - c.interval_edges()[2], 0.0) for c in cases])
    return s_lo, s_hi


def _cqr_scales(case: ForecastCase, cases: list[ForecastCase], level: float) -> tuple[float, float]:
    """Plain CQR offsets from ``cases`` expressed as grid scales."""
    if not cases:
        return 1.0, 1.0
    s_lo, s_hi = _side_scores(cases)
    _, q10, q90 = case.interval_edges()
    lo_adj = q10 - finite_quantile(s_lo, level)
    hi_adj = q90 + finite_quantile(s_hi, level)
    return scales_from_interval(case, lo_adj, hi_adj)


def arm_cqr_reset(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    min_post: int = 5,
    lookback: int = 12,
    baseline: int = 24,
    detector: str = "scan",
    cusum_threshold: float = 5.0,
) -> tuple[float, float]:
    """CQR on post-break scores only, when a break is detected.

    A bank case counts as post-break when its *realized* time is at
    or after the estimated onset (a forecast made pre-break but
    realized post-break has a large, informative score — it stays).
    Falls back to the full bank when the post-break bank is smaller
    than ``min_post``, and to plain CQR when no break is detected.
    """
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    onset = detect_break_onset(
        case.context,
        lookback=lookback,
        baseline=baseline,
        gate=detector,
        cusum_threshold=cusum_threshold,
    )
    if onset is not None:
        post = [c for c in hb if _realized_time(c) >= onset]
        if len(post) >= min_post:
            hb = post
    return _cqr_scales(case, hb, level)


def arm_cqr_reset_pool(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    min_post: int = 5,
    lookback: int = 12,
    baseline: int = 24,
    detector: str = "scan",
    cusum_threshold: float = 5.0,
) -> tuple[float, float]:
    """Reset with scores pooled across horizon steps (exp85 round 2).

    Per-horizon banks starve the reset exactly at the straddle: at
    h=6 only a handful of cases are realized post-break when the
    forecast is made. Pooling width-normalized exceedances
    (``s / (q90 - q10)`` of the scoring case) across all h — the
    exp81 pooled-bank move — multiplies the post-break evidence by
    the number of horizon steps; the offsets are de-normalized by
    the current case's own width.
    """
    if len(bank) < min_bank:
        return 1.0, 1.0
    cases = list(bank)
    onset = detect_break_onset(
        case.context,
        lookback=lookback,
        baseline=baseline,
        gate=detector,
        cusum_threshold=cusum_threshold,
    )
    if onset is not None:
        post = [c for c in cases if _realized_time(c) >= onset]
        if len(post) >= min_post:
            cases = post
    n_lo: list[float] = []
    n_hi: list[float] = []
    for c in cases:
        _, q10, q90 = c.interval_edges()
        wdt = max(q90 - q10, _EPS)
        n_lo.append(max(q10 - c.y, 0.0) / wdt)
        n_hi.append(max(c.y - q90, 0.0) / wdt)
    _, q10, q90 = case.interval_edges()
    wdt = max(q90 - q10, _EPS)
    lo_adj = q10 - finite_quantile(np.array(n_lo), level) * wdt
    hi_adj = q90 + finite_quantile(np.array(n_hi), level) * wdt
    return scales_from_interval(case, lo_adj, hi_adj)


def _decay_weights(
    case: ForecastCase,
    hb: list[ForecastCase],
    half_life: float,
    onset: int | None,
    pre_factor: float,
) -> np.ndarray:
    ages = np.array([case.origin - _realized_time(c) for c in hb], dtype=float)
    w = np.power(0.5, ages / half_life)
    if onset is not None:
        pre = np.array([_realized_time(c) < onset for c in hb])
        w = np.where(pre, w * pre_factor, w)
    return w


def _weighted_cqr_scales(
    case: ForecastCase, hb: list[ForecastCase], w: np.ndarray, level: float
) -> tuple[float, float]:
    s_lo, s_hi = _side_scores(hb)
    _, q10, q90 = case.interval_edges()
    lo_adj = q10 - weighted_quantile(s_lo, w, level)
    hi_adj = q90 + weighted_quantile(s_hi, w, level)
    return scales_from_interval(case, lo_adj, hi_adj)


def arm_cqr_decay(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    half_life: float = 6.0,
) -> tuple[float, float]:
    """CQR with exponentially decaying score weights (age half-life)."""
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    w = _decay_weights(case, hb, half_life, onset=None, pre_factor=1.0)
    return _weighted_cqr_scales(case, hb, w, level)


def arm_cqr_cp(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    half_life: float = 6.0,
    pre_factor: float = 0.25,
    lookback: int = 12,
    baseline: int = 24,
    detector: str = "scan",
    cusum_threshold: float = 5.0,
) -> tuple[float, float]:
    """Decay-weighted CQR + extra down-weighting of pre-break scores."""
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    onset = detect_break_onset(
        case.context,
        lookback=lookback,
        baseline=baseline,
        gate=detector,
        cusum_threshold=cusum_threshold,
    )
    w = _decay_weights(case, hb, half_life, onset=onset, pre_factor=pre_factor)
    return _weighted_cqr_scales(case, hb, w, level)


# ---------------------------------------------------------------------------
# Bias correction (BC-ACI family)
# ---------------------------------------------------------------------------


def _bias_track(hb: list[ForecastCase], lam: float) -> tuple[list[float], float]:
    """Causal EWM bias track: (bias-before-each-case, bias-after-all).

    The bias estimates E[y - median] from signed forecast errors,
    updated only *after* each case is scored, so a case never sees
    its own realized value.
    """
    track: list[float] = []
    b = 0.0
    for c in hb:
        track.append(b)
        b = lam * b + (1.0 - lam) * (c.y - c.median)
    return track, b


def _corrected_scores(hb: list[ForecastCase], lam: float) -> tuple[np.ndarray, np.ndarray, float]:
    """CQR exceedances of the bias-shifted interval + current bias.

    Each bank case's interval is shifted by the bias estimated
    *before* that case; scores are the exceedances of the realized
    value against the shifted [q10, q90].
    """
    track, b_now = _bias_track(hb, lam)
    s_lo = np.array(
        [max((c.interval_edges()[1] + b) - c.y, 0.0) for c, b in zip(hb, track, strict=True)]
    )
    s_hi = np.array(
        [max(c.y - (c.interval_edges()[2] + b), 0.0) for c, b in zip(hb, track, strict=True)]
    )
    return s_lo, s_hi, b_now


def _shrunk_bias(b_now: float, n: int, n0: float, case: ForecastCase, clip_frac: float) -> float:
    """Shrink the raw EWM bias toward 0 and clip to the native width."""
    b_eff = b_now * n / (n + n0) if n + n0 > 0 else 0.0
    _, q10, q90 = case.interval_edges()
    half = (q90 - q10) / 2.0
    if half > _EPS:
        b_eff = float(np.clip(b_eff, -clip_frac * half, clip_frac * half))
    return float(b_eff)


def _bc_scales(
    case: ForecastCase,
    hb: list[ForecastCase],
    level: float,
    bias_lambda: float,
    bias_n0: float,
    clip_frac: float,
) -> tuple[float, float]:
    s_lo, s_hi, b_now = _corrected_scores(hb, bias_lambda)
    b_eff = _shrunk_bias(b_now, len(hb), bias_n0, case, clip_frac)
    _, q10, q90 = case.interval_edges()
    lo_adj = q10 + b_eff - finite_quantile(s_lo, level)
    hi_adj = q90 + b_eff + finite_quantile(s_hi, level)
    return scales_from_interval(case, lo_adj, hi_adj)


def arm_bc_cqr(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    bias_lambda: float = 0.95,
    bias_n0: float = 12.0,
    clip_frac: float = 1.0,
) -> tuple[float, float]:
    """Bias-corrected CQR.

    Shift by the shrunk EWM bias; widen by the conformal quantile of
    bias-corrected exceedance scores.
    """
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    return _bc_scales(case, hb, level, bias_lambda, bias_n0, clip_frac)


def arm_bc_aci(
    case: ForecastCase,
    bank: list[ForecastCase],
    alpha: float = 0.2,
    gamma: float = 0.05,
    min_bank: int = 8,
    bias_lambda: float = 0.95,
    bias_n0: float = 12.0,
    clip_frac: float = 1.0,
) -> tuple[float, float]:
    """BC-ACI: ACI level adaptation replayed on bias-corrected scores.

    The ACI loop scores each bank case against the interval built
    from the *past* corrected scores at the running level (with the
    causal bias shift), exactly as ACI replays on residuals — but
    the object being calibrated is the bias-corrected CQR interval,
    so adaptation moves both the width (via the level) and the
    center (via the bias).
    """
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    track, b_now = _bias_track(hb, bias_lambda)
    s_lo, s_hi, _ = _corrected_scores(hb, bias_lambda)
    a_t = alpha
    past_lo: list[float] = []
    past_hi: list[float] = []
    for i, c in enumerate(hb):
        if len(past_lo) >= 2:
            off_lo = finite_quantile(np.array(past_lo), 1.0 - a_t)
            off_hi = finite_quantile(np.array(past_hi), 1.0 - a_t)
            _, q10, q90 = c.interval_edges()
            err = 1.0 if (c.y < q10 + track[i] - off_lo or c.y > q90 + track[i] + off_hi) else 0.0
            a_t = min(max(a_t + gamma * (alpha - err), 0.01), 0.99)
        past_lo.append(float(s_lo[i]))
        past_hi.append(float(s_hi[i]))
    b_eff = _shrunk_bias(b_now, len(hb), bias_n0, case, clip_frac)
    _, q10, q90 = case.interval_edges()
    lo_adj = q10 + b_eff - finite_quantile(s_lo, 1.0 - a_t)
    hi_adj = q90 + b_eff + finite_quantile(s_hi, 1.0 - a_t)
    return scales_from_interval(case, lo_adj, hi_adj)


def arm_cp_bc(
    case: ForecastCase,
    bank: list[ForecastCase],
    level: float = 0.8,
    min_bank: int = 8,
    min_post: int = 5,
    bias_lambda: float = 0.95,
    bias_n0: float = 12.0,
    clip_frac: float = 1.0,
    lookback: int = 12,
    baseline: int = 24,
    detector: str = "scan",
    cusum_threshold: float = 5.0,
) -> tuple[float, float]:
    """Hybrid bank surgery + bias correction.

    Reset the bank at a detected break, then BC-CQR on the surviving
    (post-break) bank. Without a detected break this is exactly
    ``arm_bc_cqr``.
    """
    hb = _hbank(bank, case.h)
    if len(hb) < min_bank:
        return 1.0, 1.0
    onset = detect_break_onset(
        case.context,
        lookback=lookback,
        baseline=baseline,
        gate=detector,
        cusum_threshold=cusum_threshold,
    )
    if onset is not None:
        post = [c for c in hb if _realized_time(c) >= onset]
        if len(post) >= min_post:
            hb = post
    return _bc_scales(case, hb, level, bias_lambda, bias_n0, clip_frac)
