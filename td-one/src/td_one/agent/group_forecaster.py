"""Group-level attrition forecasting with fallback to global.

For each borrower group, forecasts the group's monthly attrition series
using the injected ``forecaster_fn``. Groups with fewer than
``min_group_n`` at-risk borrowers in the training window fall back to
the global (portfolio) forecast — shrinkage toward global, same principle
as the decision policy's James-Stein.

The ``forecaster_fn`` contract: ``(series: np.ndarray) -> np.ndarray``
mapping a 1D history of attrition rates to a quantile forecast vector.
In production this is :func:`v2_forecaster_fn` (trailing-mean center +
Gaussian spread, exp78/exp79; an exp80 conformal layer is available
behind a flag, not the default — exp80's S1 replication failed its
pre-registered bar); the real fitted v2 router is available
as :func:`router_forecaster_fn` (exp79: evaluated, not the default),
and a cross-series **pooled** router fit as
:func:`pooled_router_forecaster_fn` (exp82: evaluated across six
designs — pooling, scale-free expert selection, shrinkage, partial
pooling — and it does not beat the trailing-mean center on the
SimulatorV2 harness; available as a research option, not the
default). In tests a stub.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from td_one.data_gen.group_series import GroupSeries

DEFAULT_MIN_GROUP_N = 200

#: Quantile levels every forecaster_fn in this module returns.
QUANTILE_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

#: Trailing naive one-step errors used to build the quantile spread.
SPREAD_WINDOW = 24

#: Floor on the Gaussian spread's sd (degenerate / constant histories).
MIN_SPREAD_SD = 1e-4

#: The proposed-method router's 9-level quantile grid (LEVELS in
#: ``agent/proposed_method.py``).
_L9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

#: Trailing window for the "trailing_mean" center (exp79).
CENTER_WINDOW = 6

#: First inner origin for the router's rolling-origin backtest. Expert
#: fits (ETS/ARIMA) and features are computed on prefixes >= this long.
BACKTEST_START = 8

#: exp80 conformal layer: first backtest origin for the per-series bank
#: (same minimum-history floor as the router protocol).
CONFORMAL_START = 8

#: exp80 conformal layer: minimum bank size before any offset applies.
CONFORMAL_MIN_BANK = 8

#: exp80 conformal layer: calibration level of the [q10, q90] interval.
CONFORMAL_LEVEL = 0.8

#: exp80 conditional conformal: trailing window for volatility scaling.
COND_VOL_WINDOW = 12

#: exp80 conditional conformal: clamp on the volatility scale factor.
COND_SCALE_MIN = 0.5
COND_SCALE_MAX = 2.0


@dataclass
class GroupForecast:
    group_id: str
    quantiles: np.ndarray  # forecast quantile vector
    used_fallback: bool
    n_at_risk_train: int


class GroupForecaster:
    """Forecast each group's attrition series, with global fallback.

    Fallback split (exp80b, from exp72's Q2 finding): a group below
    ``min_group_n`` whose recent ``n_at_risk`` has collapsed to 0 is
    **dead** — the portfolio forecast is the wrong answer for it (its
    realized rate is 0; exp72 measured fallback CRPS 0.0751 vs 0.0000
    for its own trailing mean) — so it forecasts from its own series.
    A group below ``min_group_n`` with recent observations is **thin
    but alive** and keeps the portfolio fallback (exp72: fallback helps
    there, donor CRPS 0.121/0.086 with vs 0.143/0.120 without).
    ``dead_zero_months=0`` disables the dead branch and reproduces the
    pre-exp80 size-only fallback exactly.
    """

    def __init__(
        self,
        forecaster_fn: Callable[[np.ndarray], np.ndarray],
        min_group_n: int = DEFAULT_MIN_GROUP_N,
        dead_zero_months: int = 2,
    ) -> None:
        if min_group_n < 1:
            raise ValueError(f"min_group_n must be >= 1, got {min_group_n}")
        if dead_zero_months < 0:
            raise ValueError(f"dead_zero_months must be >= 0, got {dead_zero_months}")
        self.forecaster_fn = forecaster_fn
        self.min_group_n = min_group_n
        self.dead_zero_months = dead_zero_months

    def forecast(
        self,
        groups: dict[str, GroupSeries],
        train_months: int | None = None,
    ) -> dict[str, GroupForecast]:
        """Forecast every group.

        Parameters
        ----------
        groups:
            ``group_id -> GroupSeries``. Must include the portfolio total
            under key ``"__portfolio__"`` (see
            ``td_one.data_gen.group_series.portfolio_totals``) for fallback.
        train_months:
            History length used for the ``min_group_n`` check (sum of
            ``n_at_risk`` over the last ``train_months``). None = all.

        """
        if "__portfolio__" not in groups:
            raise ValueError("groups must include '__portfolio__' total")
        portfolio = groups["__portfolio__"]
        global_q = self.forecaster_fn(np.array(portfolio.attrition_rate, dtype=float))
        out: dict[str, GroupForecast] = {}
        for gid, gs in groups.items():
            if gid == "__portfolio__":
                continue
            n_risk = gs.n_at_risk
            if train_months is not None:
                n_risk = n_risk[-train_months:]
            total_risk = int(sum(n_risk))
            if total_risk < self.min_group_n and not self._is_dead(n_risk):
                out[gid] = GroupForecast(
                    group_id=gid,
                    quantiles=global_q.copy(),
                    used_fallback=True,
                    n_at_risk_train=total_risk,
                )
            else:
                q = self.forecaster_fn(np.array(gs.attrition_rate, dtype=float))
                out[gid] = GroupForecast(
                    group_id=gid,
                    quantiles=q,
                    used_fallback=False,
                    n_at_risk_train=total_risk,
                )
        return out

    def _is_dead(self, n_risk_window: list[int]) -> bool:
        """Detect whether the group's recent at-risk count collapsed to 0.

        ``n_risk_window`` is the (possibly window-truncated) training
        window. A group is dead when its last ``dead_zero_months``
        months are all 0; chronic thinness (small but positive recent
        counts) is *not* death. Disabled when ``dead_zero_months == 0``.
        """
        if self.dead_zero_months == 0 or len(n_risk_window) < self.dead_zero_months:
            return False
        return int(sum(n_risk_window[-self.dead_zero_months :])) == 0


def _spread_offsets(series: np.ndarray, method: str) -> np.ndarray:
    """Quantile offsets around the naive point forecast.

    Uses the series' own naive one-step errors (``np.diff``) — strictly
    history, so PIT-safe. ``"empirical"`` takes the sample quantiles of
    the trailing errors; ``"gaussian"`` scales the Normal quantiles by
    their sample sd. A short or constant history yields a near-zero
    spread (correct: no observed variation).
    """
    diffs = np.diff(series) if len(series) >= 2 else np.empty(0)
    if len(diffs) == 0:
        return np.zeros(len(QUANTILE_LEVELS))
    tail = diffs[-SPREAD_WINDOW:]
    if method == "empirical":
        return np.asarray(np.quantile(tail, QUANTILE_LEVELS), dtype=float)
    sd = float(np.std(tail)) if len(tail) > 1 else MIN_SPREAD_SD
    return np.asarray(norm.ppf(QUANTILE_LEVELS) * max(sd, MIN_SPREAD_SD), dtype=float)


def _center_value(series: np.ndarray, center: str, window: int) -> float:
    """Point-forecast center: naive last value or trailing mean."""
    if len(series) == 0:
        return 0.0
    if center == "naive":
        return float(series[-1])
    return float(np.mean(series[-window:]))


def _base_quantiles(
    series: np.ndarray, center: str, center_window: int, spread_method: str
) -> np.ndarray:
    """Compute the shipped exp78/79 forecast: center + spread."""
    q = _center_value(series, center, center_window) + _spread_offsets(series, spread_method)
    return np.clip(np.maximum.accumulate(q), 0.0, 1.0)


def _trailing_vol(series: np.ndarray) -> float:
    """Sd of the trailing one-step differences (volatility proxy)."""
    tail = series[-COND_VOL_WINDOW:]
    diffs = np.diff(tail)
    return float(np.std(diffs)) if len(diffs) > 1 else 0.0


def _conformal_offsets(  # noqa: PLR0913 - parameter object would obscure the exp80/85 knobs
    series: np.ndarray,
    *,
    center: str,
    center_window: int,
    spread_method: str,
    min_bank: int,
    conditional: bool,
    level: float = CONFORMAL_LEVEL,
    adaptive: str = "none",
    half_life: float = 6.0,
    pre_factor: float = 0.25,
    min_post: int = 5,
    bias_lambda: float = 0.95,
    bias_n0: float = 12.0,
    detector: str = "scan",
) -> tuple[float, float]:
    """CQR-style conformal offsets (lo, hi) for the [q10, q90] interval.

    Bank = rolling backtest origins ``t`` of this same series, from
    ``CONFORMAL_START`` to ``len(series) - 1``: the base forecaster's
    forecast from ``series[:t]`` vs the realized ``series[t]``. A bank
    case enters only when its realized value was observable before
    the forecast origin (``t + h - 1 < origin`` with ``h = 1`` — exp74's
    bank rule); since this function only ever sees ``series[:origin]``,
    every bank case satisfies that by construction (pinned by the
    prefix-invariance regression test).

    Scores are exp74 ``arm_cqr``'s asymmetric exceedances
    ``s_lo = max(q10 - y, 0)``, ``s_hi = max(y - q90, 0)``; offsets are
    their finite-sample conformal quantiles at ``CONFORMAL_LEVEL``, so
    they are widen-only. Fewer than ``min_bank`` cases -> ``(0, 0)``.

    ``conditional`` scales both offsets by the trailing-volatility
    ratio ``clamp(vol_now / median(vol_bank), 0.5, 2.0)`` (for the
    bias modes below, the scale applies to the quantile parts only,
    not to the center shift).

    ``adaptive`` (exp85) makes the bank shift-aware; ``"none"`` is
    exactly the exp80 behavior above:

    - ``"reset"``: if a changepoint is detected in ``series``
      (``td_one.uq.adaptive.detect_break_onset`` — Layer-2 scan gate
      + CUSUM onset, both reading only this history), scores from
      bank cases realized before the onset are dropped when at
      least ``min_post`` post-break cases remain.
    - ``"decay"``: score weights decay exponentially with age
      (half-life ``half_life`` months); offsets are weighted
      quantiles (no finite-sample correction — documented exp85
      design call).
    - ``"cp"``: ``"decay"`` plus an extra ``pre_factor`` multiplier
      on pre-break scores when a break is detected.
    - ``"bc"``: bias-corrected scores (BC-ACI family): a causal EWM
      track of the signed one-step errors shifts each bank case's
      interval before scoring; the returned offsets encode the
      shrunk, half-width-clipped current bias as a center shift
      (``off_lo = Q_lo - b``, ``off_hi = Q_hi + b``), so this mode
      can *move* the interval, not only widen it.
    - ``"cp_bc"``: ``"bc"`` restricted to the post-break bank when
      a break is detected (same ``min_post`` rule as ``"reset"``).

    ``detector`` selects the changepoint detector used by the
    surgery modes: ``"scan"`` (Layer-2 triple gate), ``"cusum"``
    (CUSUM alone on the series; engages on gradual shifts, some
    false-positive risk), or ``"errors"`` (CUSUM over the bank's
    signed errors — exp85 round 4: the only surface that engages
    on the short, noisy, quantized histories this forecaster sees;
    series-level gates starve there, first by a length guard —
    histories of 12–21 points vs a 24+12 default window — and then
    by quantization noise). Series-level detectors use a
    length-fitted window (baseline = 2/3 of the history).
    """
    from td_one.uq.methods import finite_quantile, weighted_quantile

    ts: list[int] = []
    lo_t: list[float] = []
    hi_t: list[float] = []
    err_t: list[float] = []
    s_lo: list[float] = []
    s_hi: list[float] = []
    vols: list[float] = []
    for t in range(CONFORMAL_START, len(series)):
        qb = _base_quantiles(series[:t], center, center_window, spread_method)
        y = float(series[t])
        ts.append(t)
        lo_t.append(float(qb[0]))
        hi_t.append(float(qb[-1]))
        err_t.append(y - float(qb[2]))
        s_lo.append(max(float(qb[0]) - y, 0.0))
        s_hi.append(max(y - float(qb[-1]), 0.0))
        vols.append(_trailing_vol(series[:t]))
    if len(s_lo) < min_bank:
        return 0.0, 0.0
    s_lo_a = np.array(s_lo)
    s_hi_a = np.array(s_hi)
    vols_a = np.array(vols)
    shift = 0.0

    if adaptive == "none":
        off_lo = finite_quantile(s_lo_a, level)
        off_hi = finite_quantile(s_hi_a, level)
    elif adaptive in ("reset", "decay", "cp", "bc", "cp_bc"):
        from td_one.uq.adaptive import detect_break_onset, error_onset_index

        onset: int | None = None
        if adaptive in ("reset", "cp", "cp_bc"):
            if detector == "errors":
                # CUSUM over the bank's signed errors (chronological);
                # the onset is a series time via the bank case's t.
                onset_idx = error_onset_index(np.array(err_t))
                onset = ts[onset_idx] if onset_idx is not None and onset_idx < len(ts) else None
            elif len(series) >= 12:
                # Length-fitted window: the exp74 defaults (24 + 12)
                # starve on the short series this forecaster sees
                # (exp85 round 4: harness histories are 12-21 long,
                # so the default gate returned None unconditionally).
                det_baseline = min(24, max(6, (2 * len(series)) // 3))
                det_lookback = min(12, len(series) - det_baseline)
                if det_lookback >= 3:
                    onset = detect_break_onset(
                        series, gate=detector, baseline=det_baseline, lookback=det_lookback
                    )
        post_mask = np.array([t >= onset for t in ts]) if onset is not None else None
        if adaptive == "reset":
            if post_mask is not None and int(post_mask.sum()) >= min_post:
                s_lo_a, s_hi_a, vols_a = s_lo_a[post_mask], s_hi_a[post_mask], vols_a[post_mask]
            off_lo = finite_quantile(s_lo_a, level)
            off_hi = finite_quantile(s_hi_a, level)
        elif adaptive in ("decay", "cp"):
            ages = np.array([len(series) - 1 - t for t in ts], dtype=float)
            w = np.power(0.5, ages / half_life)
            if adaptive == "cp" and post_mask is not None:
                w = np.where(post_mask, w, w * pre_factor)
            off_lo = weighted_quantile(s_lo_a, w, level)
            off_hi = weighted_quantile(s_hi_a, w, level)
        else:  # "bc" / "cp_bc"
            idx = list(range(len(ts)))
            if adaptive == "cp_bc" and post_mask is not None and int(post_mask.sum()) >= min_post:
                idx = [i for i in idx if bool(post_mask[i])]
                vols_a = vols_a[idx]
            b = 0.0
            corr_lo: list[float] = []
            corr_hi: list[float] = []
            for i in idx:
                y = float(series[ts[i]])
                corr_lo.append(max((lo_t[i] + b) - y, 0.0))
                corr_hi.append(max(y - (hi_t[i] + b), 0.0))
                b = bias_lambda * b + (1.0 - bias_lambda) * err_t[i]
            n = len(idx)
            b_eff = b * n / (n + bias_n0) if n + bias_n0 > 0 else 0.0
            qb_now = _base_quantiles(series, center, center_window, spread_method)
            half = (float(qb_now[-1]) - float(qb_now[0])) / 2.0
            if half > 1e-12:
                b_eff = float(np.clip(b_eff, -half, half))
            shift = float(b_eff)
            off_lo = finite_quantile(np.array(corr_lo), level) - shift
            off_hi = finite_quantile(np.array(corr_hi), level) + shift
    else:
        raise ValueError(f"unknown conformal adaptive mode {adaptive!r}")

    if conditional:
        vol_bank = float(np.median(vols_a)) if len(vols_a) else 0.0
        vol_now = _trailing_vol(series)
        if vol_bank > 0.0:
            scale = min(max(vol_now / vol_bank, COND_SCALE_MIN), COND_SCALE_MAX)
            off_lo = (off_lo + shift) * scale - shift
            off_hi = (off_hi - shift) * scale + shift
    return off_lo, off_hi


def v2_forecaster_fn(
    h: int = 1,
    *,
    quantile_spread: bool = True,
    spread_method: str = "gaussian",
    center: str = "trailing_mean",
    center_window: int = CENTER_WINDOW,
    conformal: bool = False,
    conformal_conditional: bool = False,
    conformal_min_bank: int = CONFORMAL_MIN_BANK,
    conformal_level: float = CONFORMAL_LEVEL,
    conformal_adaptive: str = "none",
    conformal_half_life: float = 6.0,
    conformal_pre_factor: float = 0.25,
    conformal_min_post: int = 5,
    conformal_bias_lambda: float = 0.95,
    conformal_bias_n0: float = 12.0,
    conformal_detector: str = "scan",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a group-series forecaster_fn: center + PIT-safe spread.

    The 5-level quantile vector spreads around a point-forecast center
    with offsets from the series' own trailing naive one-step errors
    (see :func:`_spread_offsets`). The spread is what the production
    forecaster was missing.

    **Default center (exp79): ``"trailing_mean"``** — the mean of the
    last ``center_window`` (6) values. exp79 settled the center question
    exp78 left open: on SimulatorV2 the trailing-mean center beat the
    naive last-value center on control (pooled −0.0072 CRPS, p=0.034;
    confirmed on three fresh seeds at −0.0171, p=1.6e-05) and under the
    S1@4σ shock (−0.0243, p=4.7e-07), with *better* S1 coverage (cov80
    0.713 vs 0.602). The naive center remains available as
    ``center="naive"`` (the exp78 behavior). The real fitted v2 router
    (:func:`router_forecaster_fn`) was also evaluated in exp79 and lost
    to both centers on control — it is not the default.

    History (exp78 root-cause): this function previously constructed a
    ``ProposedForecasterV2`` and called ``predict`` on it *without ever
    calling fit*, so ``assert self._fitted`` raised and the blanket
    ``except`` returned ``naive_q[0]`` — the last value tiled across
    every quantile level, a point mass. Additionally the single naive
    expert's quantiles were themselves tiled (``(h, 5)`` shape, not the
    router's expected 9-level grid), so the router could never have
    produced spread even if it had run. The dead router call was
    removed in exp78; the *real*, fitted router is available as
    :func:`router_forecaster_fn` (exp79).

    Parameters
    ----------
    h:
        Kept for call-site compatibility; the fn returns the 5-level
        next-step quantile vector.
    quantile_spread:
        ``False`` reproduces the exact pre-exp78 point-mass output
        (tiled last value, center ignored) for reproducibility of old
        results.
    spread_method:
        ``"gaussian"`` (default) or ``"empirical"`` offsets. exp78's
        ablation picked Gaussian: on SimulatorV2 control it beat the
        empirical offsets on CRPS and coverage in both group and global
        scopes (the empirical quantiles of ~24 trailing diffs are too
        noisy at the 0.1/0.9 tails). Other values raise ``ValueError``.
    center:
        ``"trailing_mean"`` (default, exp79) or ``"naive"`` (last
        value, exp78 behavior). Other values raise ``ValueError``.
    center_window:
        Trailing window for the ``"trailing_mean"`` center.
    conformal:
        exp80: add the CQR-style conformal layer
        (:func:`_conformal_offsets`) — per-series backtest-bank
        exceedance offsets widening [q10, q90]. **Default ``False`` —
        exp80's verdict:** the conditional variant (level 0.90)
        reached nominal control coverage on protocol *and* fresh
        seeds (cov80 0.804 / 0.824, CRPS-neutral) and improved S1@4σ
        coverage materially (+0.045 / +0.086), but failed the
        pre-registered S1 bar on fresh seeds (0.652 vs 0.77 floor) and
        an aggressive-tuning round showed the gap is structural (bank
        offsets are bounded by the worst pre-shock miss). Available
        and tested; not the default.
    conformal_conditional:
        exp80 arm C: scale the conformal offsets by the
        trailing-volatility ratio (clamped to [0.5, 2.0]).
    conformal_min_bank:
        Minimum bank size before any conformal offset applies.
    conformal_level:
        Calibration level for the offset quantile (default
        ``CONFORMAL_LEVEL`` = the [q10, q90] interval's own 0.8).
    conformal_adaptive:
        exp85 shift-adaptation mode for the conformal bank:
        ``"none"`` (default; exact exp80 behavior), ``"reset"``
        (drop pre-break scores at a detected changepoint),
        ``"decay"`` (exponential age down-weighting), ``"cp"``
        (decay + pre-break penalty), ``"bc"`` (bias-corrected
        scores + center shift, BC-ACI family), ``"cp_bc"`` (reset,
        then bias correction). Requires ``conformal=True``. See
        :func:`_conformal_offsets`.
    conformal_half_life:
        Age half-life (months) for the ``"decay"``/``"cp"`` weights.
    conformal_pre_factor:
        Extra weight multiplier on pre-break scores in ``"cp"``.
    conformal_min_post:
        Minimum post-break bank size before a reset applies.
    conformal_bias_lambda:
        EWM decay for the bias track in the ``"bc"``/``"cp_bc"``
        modes.
    conformal_bias_n0:
        Shrinkage prior count for the applied bias
        (``b * n / (n + n0)``).
    conformal_detector:
        Changepoint detector for the surgery modes (exp85):
        ``"scan"`` (default), ``"cusum"``, or ``"errors"`` — see
        :func:`_conformal_offsets`.

    """
    if spread_method not in ("empirical", "gaussian"):
        raise ValueError(f"spread_method must be 'empirical' or 'gaussian', got {spread_method!r}")
    if center not in ("naive", "trailing_mean"):
        raise ValueError(f"center must be 'naive' or 'trailing_mean', got {center!r}")
    if center_window < 1:
        raise ValueError(f"center_window must be >= 1, got {center_window}")
    if conformal_min_bank < 1:
        raise ValueError(f"conformal_min_bank must be >= 1, got {conformal_min_bank}")
    if not 0.0 < conformal_level < 1.0:
        raise ValueError(f"conformal_level must be in (0, 1), got {conformal_level}")
    if conformal and not quantile_spread:
        raise ValueError("conformal=True requires quantile_spread=True (no point-mass bank)")
    if conformal_adaptive not in ("none", "reset", "decay", "cp", "bc", "cp_bc"):
        raise ValueError(
            "conformal_adaptive must be one of 'none', 'reset', 'decay', "
            f"'cp', 'bc', 'cp_bc', got {conformal_adaptive!r}"
        )
    if conformal_adaptive != "none" and not conformal:
        raise ValueError("conformal_adaptive != 'none' requires conformal=True")
    if conformal_half_life <= 0:
        raise ValueError(f"conformal_half_life must be > 0, got {conformal_half_life}")
    if not 0.0 < conformal_pre_factor <= 1.0:
        raise ValueError(f"conformal_pre_factor must be in (0, 1], got {conformal_pre_factor}")
    if conformal_min_post < 1:
        raise ValueError(f"conformal_min_post must be >= 1, got {conformal_min_post}")
    if not 0.0 < conformal_bias_lambda < 1.0:
        raise ValueError(f"conformal_bias_lambda must be in (0, 1), got {conformal_bias_lambda}")
    if conformal_bias_n0 < 0:
        raise ValueError(f"conformal_bias_n0 must be >= 0, got {conformal_bias_n0}")
    if conformal_detector not in ("scan", "cusum", "errors"):
        raise ValueError(
            f"conformal_detector must be 'scan', 'cusum' or 'errors', got {conformal_detector!r}"
        )

    def _forecast(series: np.ndarray) -> np.ndarray:
        s = np.asarray(series, dtype=float)
        if not quantile_spread:
            # Exact pre-exp78 behavior: tiled last value (point mass).
            last = float(s[-1]) if len(s) else 0.0
            return np.full(len(QUANTILE_LEVELS), last)
        q = _base_quantiles(s, center, center_window, spread_method)
        if conformal:
            off_lo, off_hi = _conformal_offsets(
                s,
                center=center,
                center_window=center_window,
                spread_method=spread_method,
                min_bank=conformal_min_bank,
                conditional=conformal_conditional,
                level=conformal_level,
                adaptive=conformal_adaptive,
                half_life=conformal_half_life,
                pre_factor=conformal_pre_factor,
                min_post=conformal_min_post,
                bias_lambda=conformal_bias_lambda,
                bias_n0=conformal_bias_n0,
                detector=conformal_detector,
            )
            q = q.copy()
            q[0] -= off_lo
            q[-1] += off_hi
            q = np.clip(np.maximum.accumulate(q), 0.0, 1.0)
        return q

    return _forecast


# ---------------------------------------------------------------------------
# exp79: the real v2 router as a group forecaster_fn
# ---------------------------------------------------------------------------

_ROUTER_MODELS = ("naive", "ets", "arima")


def _naive_q9(hist: np.ndarray) -> np.ndarray:
    """Naive expert (9,) quantiles: last value + Gaussian diff-spread."""
    last = float(hist[-1])
    diffs = np.diff(hist)
    sd = float(np.std(diffs)) if len(diffs) > 1 else MIN_SPREAD_SD
    q: np.ndarray = np.asarray(last + norm.ppf(_L9) * max(sd, MIN_SPREAD_SD), dtype=float)
    out: np.ndarray = np.clip(q, 0.0, 1.0)
    return out


def _statmodel_q9(hist: np.ndarray, model: str) -> np.ndarray:
    """ETS / ARIMA expert (9,) quantiles: forecast +- z * residual sd."""
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore[import-untyped]

    x = np.asarray(hist, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "ets":
            res = ExponentialSmoothing(x, trend="add", damped_trend=True).fit(
                optimized=True, use_brute=True
            )
        elif model == "arima":
            res = ARIMA(x, order=(1, 1, 1)).fit()
        else:
            raise ValueError(f"unknown statmodel expert {model!r}")
        mu = float(np.asarray(res.forecast(1), dtype=float).ravel()[0])
        resid = np.asarray(res.resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    sd = float(np.std(resid)) if len(resid) > 1 else MIN_SPREAD_SD
    q: np.ndarray = np.asarray(mu + norm.ppf(_L9) * max(sd, MIN_SPREAD_SD), dtype=float)
    out: np.ndarray = np.clip(q, 0.0, 1.0)
    return out


def _expert_q9(model: str, hist: np.ndarray) -> np.ndarray:
    """(9,) one-step quantiles for one router expert on ``hist``."""
    if model == "naive":
        return _naive_q9(hist)
    return _statmodel_q9(hist, model)


def router_forecaster_fn(
    h: int = 1,
    *,
    regime_spec: str = "generic",
    models: tuple[str, ...] = _ROUTER_MODELS,
    backtest_start: int = BACKTEST_START,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a group-series forecaster_fn around the *real* v2 router.

    This is what the pre-exp78 production code intended and never did:
    a :class:`~td_one.agent.proposed_method_v2.ProposedForecasterV2`
    that is actually **fitted** before it predicts. Per call, the
    router is fitted on rolling-origin backtest records built from the
    given series' own history (PIT-safe: record t uses only ``s[:t]``):

    - features: the canonical 10-feature ``series_features`` bundle;
    - experts: ``models`` (default naive / ETS / ARIMA — the set named
      in the original production docstring), each emitting (9, 1)
      quantiles on the router's 9-level grid;
    - per-expert CRPS vs the realized next value.

    ``predict`` then combines the experts' full-history quantiles per
    the fitted regime router (+ conformal calibration), and the (9, 1)
    output is interpolated onto the production 5-level grid. The
    router's own conformal layer is its interval machinery, so no
    exp78 spread is stacked on top.

    **exp79 verdict: evaluated, not the production default.** Fitted
    per-series on SimulatorV2, the router *lost* to the naive-center
    baseline on control (+0.0080 CRPS, p=0.012) and to the trailing-mean
    center everywhere (p<=6e-07), with worse coverage (cov80 0.618 vs
    0.714). Diagnosis: a per-series self-fit has at most ~13 backtest
    records (vs hundreds pooled across series in the benchmarks the
    router was designed for), so regimes barely separate and ETS/ARIMA
    experts overfit 12–29 noisy points. The function stays available
    for series long enough to fit honestly, and for research.

    **No silent degradation:** there is deliberately no ``try/except``
    around fit/predict. If the router cannot fit or predict, the call
    raises — the exp78 defect (unfitted router -> swallowed assert ->
    point mass) must be impossible to reintroduce silently. Histories
    shorter than ``backtest_start + 1`` raise ``ValueError``; callers
    with tiny histories should use :func:`v2_forecaster_fn` instead.

    Parameters
    ----------
    h:
        Kept for call-site compatibility; returns the next-step vector.
    regime_spec:
        ``"generic"`` (calm/stressed; the method's SMM configuration,
        well-defined on short series) or ``"m4"`` (seasonal/trending/
        noisy; needs long series for its STL-based features to be
        meaningful). Other values raise ``ValueError``.
    models:
        Expert names; subset of {"naive", "ets", "arima"}.
    backtest_start:
        First inner backtest origin (prefix length).

    """
    if regime_spec not in ("generic", "m4"):
        raise ValueError(f"regime_spec must be 'generic' or 'm4', got {regime_spec!r}")
    unknown = [m for m in models if m not in _ROUTER_MODELS]
    if unknown:
        raise ValueError(f"unknown router experts {unknown}; allowed {_ROUTER_MODELS}")
    if backtest_start < 5:
        raise ValueError(f"backtest_start must be >= 5, got {backtest_start}")

    # Prefix cache: features/expert-quantiles for a history prefix are
    # pure functions of that prefix, and rolling outer origins revisit
    # the same prefixes repeatedly. Keyed by exact bytes (PIT-safe).
    artifact_cache: dict[bytes, tuple[list[float], dict[str, np.ndarray]]] = {}

    def _artifacts(
        hist: np.ndarray, feature_names: list[str]
    ) -> tuple[list[float], dict[str, np.ndarray]]:
        from td_one.agent.tools.data import series_features

        key = np.asarray(hist, dtype=float).tobytes()
        if key not in artifact_cache:
            feats = series_features(hist).as_dict()
            vec = [float(feats[k]) for k in feature_names]
            qs = {m: _expert_q9(m, hist).reshape(9, 1) for m in models}
            artifact_cache[key] = (vec, qs)
        return artifact_cache[key]

    def _forecast(series: np.ndarray) -> np.ndarray:
        from td_one.agent.layer2_tools import pinball_crps
        from td_one.agent.proposed_method import generic_regime_spec, m4_regime_spec
        from td_one.agent.proposed_method_v2 import ProposedForecasterV2
        from td_one.agent.tools.data import SeriesFeatures

        s = np.asarray(series, dtype=float)
        if len(s) < backtest_start + 1:
            raise ValueError(
                f"router needs a history of at least {backtest_start + 1} points "
                f"to fit its backtest records, got {len(s)}"
            )
        feature_names = list(SeriesFeatures.model_fields)

        records: list[dict[str, Any]] = []
        for t in range(backtest_start, len(s)):
            vec, qs = _artifacts(s[:t], feature_names)
            records.append(
                {
                    "features": vec,
                    "quantiles": qs,
                    "crps": {m: pinball_crps(qs[m], np.array([s[t]])) for m in models},
                    "actual": np.array([s[t]]),
                }
            )

        spec = generic_regime_spec() if regime_spec == "generic" else m4_regime_spec()
        fc = ProposedForecasterV2(
            h=h,
            regime_spec=spec,
            models=list(models),
            use_llm_expert=False,
        )
        fc.fit(records, feature_names)
        vec, qs = _artifacts(s, feature_names)
        feats = dict(zip(feature_names, vec, strict=True))
        out = fc.predict(s, qs, feats, feature_names)
        q9 = np.asarray(out.quantiles, dtype=float)[:, 0]
        q5 = np.interp(QUANTILE_LEVELS, _L9, q9)
        return np.clip(np.maximum.accumulate(q5), 0.0, 1.0)

    return _forecast


# ---------------------------------------------------------------------------
# exp82: the pooled router — one fit across the whole panel
# ---------------------------------------------------------------------------


def _normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    """Copy a backtest record with its CRPS made scale-free (exp82 R2e).

    Each expert's CRPS is divided by the record's naive-expert CRPS
    (or the record's mean expert CRPS when naive is not among the
    experts), so pooled expert selection weighs every series equally
    instead of being dominated by the largest-scale series. Per-record
    argmin labels are invariant to the scaling.
    """
    crps = record["crps"]
    denom = float(crps.get("naive", 0.0))
    if denom <= 0.0:
        denom = float(np.mean(list(crps.values()))) if crps else 0.0
    if denom <= 0.0:
        return record
    return {**record, "crps": {m: float(c) / denom for m, c in crps.items()}}


class _PooledRouter:
    """One v2 router fitted on backtest records pooled across a panel.

    exp79's diagnosis of the per-series router (:func:`router_forecaster_fn`)
    was starvation: a series' own rolling-origin backtest yields at most
    ~13 records, while the router was designed for the pooled record
    sets of exp47 (hundreds of records across series). This core fits
    **one** :class:`ProposedForecasterV2` per forecast-origin length L
    on the records of *every* panel series truncated to L — the same
    record protocol as exp79 (prefixes t = ``backtest_start``..L-1,
    10 canonical features, experts {naive, ets, arima}, pinball CRPS
    vs the realized next value), pooled.

    PIT safety: records at origin L use only prefixes ``s[:t]`` with
    t < L of panel series truncated to L, so nothing at or after the
    origin enters the fit (pinned by a prefix-invariance test).

    ``target_weight`` > 1 gives partial pooling (exp82 R3): the target
    series' own records are repeated ``target_weight`` times in the
    fit — shrinkage between the pooled and per-series fits at the data
    level. The default 1 is the pure pool.
    """

    def __init__(
        self,
        panel: Mapping[str, np.ndarray],
        *,
        regime_spec: str = "generic",
        models: tuple[str, ...] = _ROUTER_MODELS,
        backtest_start: int = BACKTEST_START,
        features: tuple[str, ...] | None = None,
        normalize_crps: bool = False,
    ) -> None:
        if regime_spec not in ("generic", "m4"):
            raise ValueError(f"regime_spec must be 'generic' or 'm4', got {regime_spec!r}")
        unknown = [m for m in models if m not in _ROUTER_MODELS]
        if unknown:
            raise ValueError(f"unknown router experts {unknown}; allowed {_ROUTER_MODELS}")
        if backtest_start < 5:
            raise ValueError(f"backtest_start must be >= 5, got {backtest_start}")
        from td_one.agent.tools.data import SeriesFeatures

        canonical = list(SeriesFeatures.model_fields)
        if features is not None:
            bad = [f for f in features if f not in canonical]
            if bad or len(features) == 0:
                raise ValueError(
                    f"features must be a non-empty subset of {canonical}, got {features!r}"
                )
            self._feature_names = list(features)
            self._feature_idx: list[int] | None = [canonical.index(f) for f in features]
        else:
            self._feature_names = canonical
            self._feature_idx = None
        self._canonical = canonical
        self._normalize_crps = normalize_crps
        self._panel = {str(k): np.asarray(v, dtype=float) for k, v in panel.items()}
        self._regime_spec = regime_spec
        self._models = models
        self._backtest_start = backtest_start
        # Prefix artifacts are pure functions of the prefix; rolling
        # origins and pooled series revisit the same prefixes often.
        self._artifact_cache: dict[bytes, tuple[list[float], dict[str, np.ndarray]]] = {}
        self._records_cache: dict[
            int, tuple[list[dict[str, Any]], dict[bytes, list[dict[str, Any]]]]
        ] = {}
        self._routers: dict[tuple[int, bytes | None], Any] = {}

    # -- record building --------------------------------------------------

    def _artifacts(self, hist: np.ndarray) -> tuple[list[float], dict[str, np.ndarray]]:
        from td_one.agent.tools.data import series_features

        key = np.asarray(hist, dtype=float).tobytes()
        if key not in self._artifact_cache:
            feats = series_features(hist).as_dict()
            vec = [float(feats[k]) for k in self._canonical]
            qs = {m: _expert_q9(m, hist).reshape(9, 1) for m in self._models}
            self._artifact_cache[key] = (vec, qs)
        return self._artifact_cache[key]

    def _records(
        self, length: int
    ) -> tuple[list[dict[str, Any]], dict[bytes, list[dict[str, Any]]]]:
        """Pooled backtest records for origin ``length`` (+ per-series split).

        Returns (all records, {truncated-series bytes: that series'
        records}) — the split exists for partial pooling.
        """
        from td_one.agent.layer2_tools import pinball_crps

        if length in self._records_cache:
            return self._records_cache[length]
        pooled: list[dict[str, Any]] = []
        by_series: dict[bytes, list[dict[str, Any]]] = {}
        for s_full in self._panel.values():
            s = s_full[:length]
            if len(s) < self._backtest_start + 1:
                continue
            recs: list[dict[str, Any]] = []
            for t in range(self._backtest_start, len(s)):
                vec, qs = self._artifacts(s[:t])
                if self._feature_idx is not None:
                    vec = [vec[i] for i in self._feature_idx]
                recs.append(
                    {
                        "features": vec,
                        "quantiles": qs,
                        "crps": {m: pinball_crps(qs[m], np.array([s[t]])) for m in self._models},
                        "actual": np.array([s[t]]),
                    }
                )
            by_series[np.asarray(s, dtype=float).tobytes()] = recs
            pooled.extend(recs)
        self._records_cache[length] = (pooled, by_series)
        return pooled, by_series

    # -- fitting / prediction ----------------------------------------------

    def router(self, series: np.ndarray, *, target_weight: int = 1) -> Any:
        """Return the pooled router fitted for ``len(series)`` (cached).

        With ``target_weight`` > 1 the fit pool additionally repeats
        the target series' own records ``target_weight - 1`` times.
        Raises ``ValueError`` when the pool is empty at this length.
        """
        s = np.asarray(series, dtype=float)
        length = len(s)
        key = (length, s.tobytes() if target_weight > 1 else None)
        if key not in self._routers:
            from td_one.agent.proposed_method import generic_regime_spec, m4_regime_spec
            from td_one.agent.proposed_method_v2 import ProposedForecasterV2

            pooled, by_series = self._records(length)
            if not pooled:
                raise ValueError(
                    f"pooled router: empty backtest pool at length {length} "
                    f"(panel series shorter than backtest_start + 1 = "
                    f"{self._backtest_start + 1}, or empty panel)"
                )
            pool = list(pooled)
            if target_weight > 1:
                pool += by_series.get(s.tobytes(), []) * (target_weight - 1)
            if self._normalize_crps:
                pool = [_normalized_record(r) for r in pool]
            spec = generic_regime_spec() if self._regime_spec == "generic" else m4_regime_spec()
            fc = ProposedForecasterV2(
                h=1,
                regime_spec=spec,
                models=list(self._models),
                use_llm_expert=False,
            )
            fc.fit(pool, self._feature_names)
            self._routers[key] = fc
        return self._routers[key]

    def combined_q9(self, series: np.ndarray, *, target_weight: int = 1) -> np.ndarray:
        """Return the pooled router's combined (9,) quantiles for ``series``."""
        s = np.asarray(series, dtype=float)
        if len(s) < self._backtest_start + 1:
            raise ValueError(
                f"pooled router needs a history of at least {self._backtest_start + 1} "
                f"points, got {len(s)}"
            )
        fc = self.router(s, target_weight=target_weight)
        vec, qs = self._artifacts(s)
        feats = dict(zip(self._canonical, vec, strict=True))
        out = fc.predict(s, qs, feats, self._feature_names)
        return np.asarray(out.quantiles, dtype=float)[:, 0]

    def center(self, series: np.ndarray, *, target_weight: int = 1) -> float:
        """Return the pooled router's combined median — the forecast center."""
        return float(self.combined_q9(series, target_weight=target_weight)[4])


def pooled_router_forecaster_fn(
    panel: Mapping[str, np.ndarray],
    h: int = 1,
    *,
    regime_spec: str = "generic",
    models: tuple[str, ...] = _ROUTER_MODELS,
    backtest_start: int = BACKTEST_START,
    features: tuple[str, ...] | None = None,
    normalize_crps: bool = False,
    spread_method: str = "gaussian",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a group-series forecaster_fn around the **pooled** v2 router.

    exp82's candidate center: one :class:`ProposedForecasterV2` fitted
    on rolling-origin backtest records **pooled across every series in
    ``panel``** (the exp47 recipe), instead of exp79's per-series fit
    (~13 records) that starved the router. **exp82's verdict: the
    pooled router does not beat the shipped trailing-mean center on
    the SimulatorV2 group harness** (control +0.0060 CRPS, p=0.005;
    S1@4σ +0.0131, p=1.1e-05; five further designs — scale-free
    selection, shrinkage, partial pooling, combinations — close but
    never clear the gap; see experiments/exp82_report.md). This
    factory remains available as a research option; it is **not** the
    production default. The router's combined median
    is the point-forecast center; exp78's Gaussian spread (the input
    series' trailing naive one-step errors) supplies the intervals —
    the same spread machinery as :func:`v2_forecaster_fn`, so pooled
    vs trailing-mean comparisons isolate the center.

    ``panel`` maps series keys (group ids, conventionally including
    ``"__portfolio__"``) to their **full** histories; at each call the
    panel is truncated to the input's length, so only data strictly
    before the forecast origin enters the fit (PIT-safe; pinned by a
    prefix-invariance regression test). The input series need not be a
    panel member — the router generalizes through the features — but
    in the group-forecaster harness it always is.

    **No silent degradation:** there is deliberately no ``try/except``
    around fit/predict. An empty pool (panel series all shorter than
    ``backtest_start + 1``) raises ``ValueError``; any router failure
    raises. The exp78 defect class must be impossible to reintroduce
    silently.

    Parameters
    ----------
    panel:
        Fit pool: series key -> full history of attrition rates.
    h:
        Kept for call-site compatibility; returns the next-step vector.
    regime_spec:
        ``"generic"`` (default; exp79's frozen choice for short series)
        or ``"m4"``. Other values raise ``ValueError``.
    models:
        Expert names; subset of {"naive", "ets", "arima"}.
    backtest_start:
        First inner backtest origin (prefix length).
    features:
        Optional subset of the 10 canonical ``series_features`` names
        for the router's logistic fit (exp82 R2a variant); the regime
        label rule always sees the full feature dict. Unknown names
        raise ``ValueError``.
    normalize_crps:
        exp82 R2e variant (default off): divide each pooled record's
        per-expert CRPS by that record's naive CRPS before fitting, so
        expert selection is scale-free across heterogeneous series
        (the raw pooled mean is dominated by the largest-scale series;
        exp82's diagnosis found it inverts the stressed-regime expert).
    spread_method:
        ``"gaussian"`` (default) or ``"empirical"`` — as in
        :func:`v2_forecaster_fn`.

    """
    if spread_method not in ("empirical", "gaussian"):
        raise ValueError(f"spread_method must be 'empirical' or 'gaussian', got {spread_method!r}")
    core = _PooledRouter(
        panel,
        regime_spec=regime_spec,
        models=models,
        backtest_start=backtest_start,
        features=features,
        normalize_crps=normalize_crps,
    )

    def _forecast(series: np.ndarray) -> np.ndarray:
        s = np.asarray(series, dtype=float)
        q = core.center(s) + _spread_offsets(s, spread_method)
        return np.clip(np.maximum.accumulate(q), 0.0, 1.0)

    return _forecast
