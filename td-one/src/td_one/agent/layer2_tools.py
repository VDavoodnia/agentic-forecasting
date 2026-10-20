"""Deterministic analyst toolbox for the Layer-2 meta-router (exp73).

The Layer-2 meta-router is an *analyst with tools*, not a bare gating
rule: at every origin it assembles an evidence bundle from these tools
before deciding which forecaster to trust. Every tool here is

- **deterministic** (no LLM, no randomness),
- **PIT-safe by construction** — each tool only sees the data slice the
  caller passes (series up to the origin, rates with
  ``month <= asof - publication_lag``), and
- **JSON-traceable** — returns plain dicts of floats/strings/bools so
  the full evidence bundle can be dumped into a decision trace.

Tools
-----
``series_diagnostics``  — is this segment stable enough that naive wins?
``cusum_break_pvalue``  — OLS-CUSUM parameter-stability p-value.
``changepoint_scan``    — did the ground move in the last ``lookback`` months?
``expert_backtest``     — rolling-origin CRPS/bias of candidate callables.
``summarize_backtest``  — the same table from precomputed CRPS arrays.
``router_audit``        — when Layer-1 overrode naive historically, was it right?
``interval_compare``    — interval widths + empirical calibration per candidate.
``lockin_level`` / ``OSFILockInMeter`` — refi-incentive (lock-in) proxy from
                          OSFI M4 new-lending rates, as-of dated.
``macro_context``       — Macro Signal Service signals, competitor spread,
                          house-price momentum (all strictly as-of filtered;
                          missing sources degrade to ``available=False``).

Other tracks (Macro Signal Service consumers, exp69c) can import these
directly — they have no dependency on the meta-router itself.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

#: The 9-quantile grid used across the repo (exp47+ convention).
LEVELS_9: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_LEVELS_ARR = np.array(LEVELS_9)

#: A candidate forecaster: (train window, horizon) -> (9, H) quantile grid.
CandidateFn = Callable[[np.ndarray, int], np.ndarray]


def _f(x: float) -> float:
    """JSON-safe float (NaN/inf -> nan stays float for json with default)."""
    return float(x)


def pinball_crps(quantiles: np.ndarray, actual: np.ndarray) -> float:
    """CRPS of a (9, H) quantile grid vs an (H,) actual vector.

    ``2 * mean pinball`` over levels and steps — bit-identical to
    ``llm_rungs.combine._mean_crps`` and, at H=1, to
    ``llm_rungs.cache.grid_to_crps``.
    """
    qs = np.asarray(quantiles, dtype=float)
    y = np.asarray(actual, dtype=float).ravel()
    if qs.shape != (len(LEVELS_9), y.size):
        raise ValueError(f"quantiles shape {qs.shape} != (9, {y.size})")
    losses = np.where(
        y[None, :] >= qs,
        (y[None, :] - qs) * _LEVELS_ARR[:, None],
        (qs - y[None, :]) * (1.0 - _LEVELS_ARR[:, None]),
    )
    return 2.0 * float(np.mean(losses))


def q50_of(grid: np.ndarray) -> float:
    """Step-1 median (the 0.5 level) of a (9, H) quantile grid."""
    return float(np.asarray(grid, dtype=float)[4, 0])


# ---------------------------------------------------------------------------
# Tool 1: series diagnostics
# ---------------------------------------------------------------------------


def cusum_break_pvalue(y: np.ndarray) -> float:
    """OLS-CUSUM parameter-stability p-value for ``y ~ 1 + t``.

    Statistic: sup of the centred cumulative residual sum, standardised
    by ``sigma * sqrt(n)``; under H0 it follows the sup of a Brownian
    bridge, whose CDF is the Kolmogorov distribution
    (``scipy.stats.kstwobign``). Small p -> the series' level/trend
    parameters are unstable.
    """
    from scipy.stats import kstwobign

    x = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 10:
        return float("nan")
    t = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ beta
    sigma = float(np.std(resid, ddof=2))
    if sigma < 1e-12:
        return 1.0
    cusum = np.cumsum(resid - resid.mean())
    stat = float(np.max(np.abs(cusum)) / (sigma * np.sqrt(n)))
    return float(1.0 - kstwobign.cdf(stat))


def series_diagnostics(y: np.ndarray, period: int = 12, recent_k: int = 12) -> dict[str, Any]:
    """Stability diagnostics for one series (visible data only).

    Returns trend/seasonal strength (Wang–Smith–Hyndman definitions via
    an additive STL-style decomposition), residual variance, AC1 of the
    levels, recent-vs-historical volatility ratio (on first
    differences), a last-k stability score in (0, 1], and the
    OLS-CUSUM break p-value.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore[import-untyped]

    x = np.asarray(y, dtype=float).ravel()
    if x.size < 36:
        raise ValueError(f"series_diagnostics needs at least 36 points, got {x.size}")
    decomp = seasonal_decompose(x, model="additive", period=period, extrapolate_trend="period")
    resid = np.asarray(decomp.resid, dtype=float)
    seasonal = np.asarray(decomp.seasonal, dtype=float)
    deseasonalised = x - seasonal
    var_resid = float(np.nanvar(resid))
    var_ds = float(np.nanvar(deseasonalised))
    var_sr = float(np.nanvar(seasonal + resid))
    trend_strength = max(0.0, 1.0 - var_resid / var_ds) if var_ds > 0 else 0.0
    seasonal_strength = max(0.0, 1.0 - var_resid / var_sr) if var_sr > 0 else 0.0

    ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.size > 2 else float("nan")
    diffs = np.diff(x)
    recent_vol = float(np.std(diffs[-recent_k:])) if recent_k >= 2 else float("nan")
    hist_vol = float(np.std(diffs[:-recent_k])) if diffs.size > recent_k else float("nan")
    vol_ratio = recent_vol / hist_vol if hist_vol and hist_vol > 0 else float("nan")
    sd = float(np.std(x))
    mean_shift = abs(float(np.mean(x[-recent_k:])) - float(np.mean(x[:-recent_k])))
    stability = float(np.exp(-mean_shift / sd)) if sd > 0 else 0.0
    return {
        "n": int(x.size),
        "ac1": _f(ac1),
        "trend_strength": _f(trend_strength),
        "seasonal_strength": _f(seasonal_strength),
        "resid_var": _f(var_resid),
        "vol_recent": _f(recent_vol),
        "vol_hist": _f(hist_vol),
        "vol_ratio": _f(vol_ratio),
        "stability": _f(stability),
        "cusum_p": _f(cusum_break_pvalue(x)),
    }


# ---------------------------------------------------------------------------
# Tool 6: changepoint scan (recent window)
# ---------------------------------------------------------------------------


def changepoint_scan(y: np.ndarray, lookback: int = 12, baseline: int = 24) -> dict[str, Any]:
    """Detect a regime shift whose onset falls in the last ``lookback`` points.

    Three channels, all computed on the trailing ``baseline + lookback``
    window (PIT-pure — only ``y`` as passed is read). A break is a
    *concentrated* event, which is what separates it from random-walk
    drift (diffuse by construction):

    - **Jump break**: level-shift z > 3 AND the largest standardised
      innovation inside the recent window exceeds 3 (a genuine level
      break has one or two outsized innovations at its onset; drift
      does not).
    - **Trend break**: the mean of recent innovations differs from the
      baseline innovations' mean by > 3 robust SEs.
    - **Volatility break**: ``sd(diff(recent)) / sd(diff(baseline))``
      above 3.

    The level-shift z uses the *innovation* SD with an autocorrelation
    correction (effective sample size ``n (1 - rho) / (1 + rho)``, with
    ``rho`` the level AC1 estimated on the *baseline* window only, so a
    break cannot inflate it) — this keeps highly persistent series
    (runoff rates, random walks) from false-firing. All innovation
    scales are MAD-robust so a single jump cannot inflate its own
    yardstick. The level CUSUM statistic is reported for the trace.
    ``onset_from_end`` is a coarse "within the lookback window" when a
    channel fires.
    """
    from td_one.agent.shift_guard import cusum_detect

    x = np.asarray(y, dtype=float).ravel()
    out: dict[str, Any] = {
        "break_flag": False,
        "onset_from_end": None,
        "level_shift_z": 0.0,
        "trend_shift_z": 0.0,
        "max_innov_z": 0.0,
        "vol_ratio": float("nan"),
        "cusum_stat": 0.0,
    }
    if x.size < baseline + 3:
        return out
    window = x[-(baseline + lookback) :]
    base = window[:baseline]
    recent = window[baseline:]
    innov = np.diff(window)
    innov_base = innov[: baseline - 1]
    innov_recent = innov[baseline - 1 :]
    med = float(np.median(innov_base))
    mad = float(np.median(np.abs(innov_base - med)))
    sigma_rob = 1.4826 * mad if mad > 0 else float(np.std(innov_base))
    if sigma_rob <= 0:
        return out
    base_sd = float(np.std(innov_base))
    recent_sd = float(np.std(innov_recent))
    out["vol_ratio"] = _f(recent_sd / base_sd) if base_sd > 0 else float("nan")
    # Level-shift z with autocorrelation-corrected effective sample sizes.
    rho = float(np.corrcoef(base[:-1], base[1:])[0, 1])
    rho = min(max(rho, -0.99), 0.99)
    n_eff_l = max(1.0, lookback * (1.0 - rho) / (1.0 + rho))
    n_eff_b = max(1.0, baseline * (1.0 - rho) / (1.0 + rho))
    se_level = sigma_rob * float(np.sqrt(1.0 / n_eff_l + 1.0 / n_eff_b))
    z_level = abs(float(np.mean(recent)) - float(np.mean(base))) / se_level if se_level > 0 else 0.0
    out["level_shift_z"] = _f(z_level)
    # Trend channel: innovation-mean shift (innovations ~iid by design).
    se_diff = sigma_rob * float(np.sqrt(1.0 / innov_recent.size + 1.0 / innov_base.size))
    z_diff = (
        abs(float(np.mean(innov_recent)) - float(np.mean(innov_base))) / se_diff
        if se_diff > 0
        else 0.0
    )
    out["trend_shift_z"] = _f(z_diff)
    # Jump channel: largest standardised innovation in the recent window.
    out["max_innov_z"] = _f(float(np.max(np.abs(innov_recent - med))) / sigma_rob)
    det = cusum_detect(window, asof=len(window) - 1)
    out["cusum_stat"] = _f(det.statistic)
    vol = out["vol_ratio"]
    fired = (
        (z_level > 3.0 and out["max_innov_z"] > 3.0)
        or z_diff > 3.0
        or (isinstance(vol, float) and vol > 3.0)
    )
    if fired:
        out["break_flag"] = True
        out["onset_from_end"] = lookback
    return out


# ---------------------------------------------------------------------------
# Tool 3: expert backtest
# ---------------------------------------------------------------------------


def summarize_backtest(
    crps_by_candidate: Mapping[str, Sequence[float]],
    bias_by_candidate: Mapping[str, Sequence[float]] | None = None,
    naive_name: str = "naive",
    recent_k: int = 12,
) -> dict[str, Any]:
    """Evidence table from per-origin CRPS (and optional bias) arrays.

    Per candidate: mean CRPS, mean bias (if given), win rate vs the
    naive reference, and the paired SE of the CRPS difference vs naive.
    Plus ``best`` (lowest mean CRPS) and ``recent_winner`` (best over
    the last ``recent_k`` origins).
    """
    if naive_name not in crps_by_candidate:
        raise ValueError(f"naive reference {naive_name!r} not in candidates")
    naive = np.asarray(crps_by_candidate[naive_name], dtype=float)
    table: dict[str, Any] = {}
    for name, vals in crps_by_candidate.items():
        c = np.asarray(vals, dtype=float)
        entry: dict[str, Any] = {"mean_crps": _f(np.mean(c)), "n": int(c.size)}
        if bias_by_candidate is not None and name in bias_by_candidate:
            entry["mean_bias"] = _f(np.mean(bias_by_candidate[name]))
        if name == naive_name:
            entry["win_rate_vs_naive"] = None
            entry["se_vs_naive"] = 0.0
        else:
            entry["win_rate_vs_naive"] = _f(np.mean(c < naive))
            diffs = naive - c
            entry["se_vs_naive"] = (
                _f(np.std(diffs, ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else float("inf")
            )
        table[name] = entry
    k = min(recent_k, naive.size)
    recent_means = {
        name: float(np.mean(np.asarray(v, dtype=float)[-k:]))
        for name, v in crps_by_candidate.items()
    }
    names = [m for m in table if isinstance(table[m], dict)]
    table["best"] = min(names, key=lambda m: table[m]["mean_crps"])
    table["recent_winner"] = min(recent_means, key=lambda m: recent_means[m])
    return table


def expert_backtest(
    y: np.ndarray,
    candidates: Mapping[str, CandidateFn],
    origins: Sequence[int],
    horizon: int = 1,
    train_window: int = 36,
    naive_name: str = "naive",
) -> dict[str, Any]:
    """Rolling-origin backtest of candidate forecaster callables on ``y``.

    Each candidate is called as ``fn(train, horizon) -> (9, H)`` on the
    ``train_window`` points strictly before each origin; scored with
    :func:`pinball_crps` against ``y[o:o+horizon]``; bias is the mean
    over steps of (step median − actual). Returns the
    :func:`summarize_backtest` table (which includes per-candidate
    ``mean_crps`` / ``mean_bias`` / ``win_rate_vs_naive``).
    """
    x = np.asarray(y, dtype=float).ravel()
    crps: dict[str, list[float]] = {name: [] for name in candidates}
    bias: dict[str, list[float]] = {name: [] for name in candidates}
    for o in origins:
        train = x[o - train_window : o]
        if train.size < train_window:
            raise ValueError(f"origin {o}: train window {train.size} < {train_window}")
        actual = x[o : o + horizon]
        if actual.size < horizon:
            raise ValueError(f"origin {o}: fewer than horizon={horizon} actuals")
        for name, fn in candidates.items():
            grid = np.asarray(fn(train, horizon), dtype=float)
            crps[name].append(pinball_crps(grid, actual))
            med = np.asarray(grid, dtype=float)[4, :]
            bias[name].append(float(np.mean(med - actual)))
    return summarize_backtest(crps, bias, naive_name=naive_name)


# ---------------------------------------------------------------------------
# Tool 4: router audit
# ---------------------------------------------------------------------------


def router_audit(
    layer1_crps: Sequence[float],
    naive_crps: Sequence[float],
    override: Sequence[bool],
    regimes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Did Layer-1's non-naive overrides pay off historically.

    ``override[i]`` marks origins where Layer-1's forecast differed
    materially from naive. Returns the override rate, the hit rate
    (fraction of overrides where Layer-1 beat naive), the mean CRPS
    delta on overrides (negative = overrides helped), and the same
    breakdown per regime when ``regimes`` is given.
    """
    l1 = np.asarray(layer1_crps, dtype=float)
    nv = np.asarray(naive_crps, dtype=float)
    ov = np.asarray(override, dtype=bool)
    if not (l1.size == nv.size == ov.size):
        raise ValueError("layer1_crps, naive_crps, override must align")

    def _stats(mask: np.ndarray) -> dict[str, Any]:
        n_ov = int(np.sum(mask))
        if n_ov == 0:
            return {"n_override": 0, "hit_rate": float("nan"), "mean_delta": float("nan")}
        return {
            "n_override": n_ov,
            "hit_rate": _f(np.mean(l1[mask] < nv[mask])),
            "mean_delta": _f(np.mean(l1[mask] - nv[mask])),
        }

    out: dict[str, Any] = {
        "n": int(l1.size),
        "override_rate": _f(np.mean(ov)) if ov.size else 0.0,
        **_stats(ov),
    }
    if regimes is not None:
        reg = np.asarray([str(r) for r in regimes])
        out["by_regime"] = {r: _stats(ov & (reg == r)) for r in sorted(set(reg.tolist()))}
    return out


# ---------------------------------------------------------------------------
# Tool 7: interval comparison
# ---------------------------------------------------------------------------


def interval_compare(
    grids: Mapping[str, np.ndarray],
    actuals: np.ndarray,
) -> dict[str, Any]:
    """Interval widths + 80% coverage per candidate.

    ``grids[name]`` is an (n, 9, H) array, ``actuals`` an (n, H) array.
    Width = mean(q90 − q10); coverage = fraction of (origin, step) pairs
    with q10 <= actual <= q90. Feeds Layer-2's do-no-harm logic (a
    candidate whose intervals badly under-cover on validation is
    vetoed even if its CRPS looks good).
    """
    act = np.asarray(actuals, dtype=float)
    out: dict[str, Any] = {}
    for name, g in grids.items():
        arr = np.asarray(g, dtype=float)
        if arr.ndim != 3 or arr.shape[0] != act.shape[0]:
            raise ValueError(f"{name}: grids must be (n, 9, H) aligned to actuals")
        width = float(np.mean(arr[:, 8, :] - arr[:, 0, :]))
        cover = float(np.mean((act >= arr[:, 0, :]) & (act <= arr[:, 8, :])))
        out[name] = {"mean_width": _f(width), "coverage_80": _f(cover)}
    return out


# ---------------------------------------------------------------------------
# Tool 2: lock-in meter
# ---------------------------------------------------------------------------


def lockin_level(market_rate: float, portfolio_proxy: float) -> float:
    """Refi-incentive proxy: market rate minus the book's average rate.

    Positive = borrowers are locked in below market (refi incentive is
    low, runoff should soften); negative = the book sits above market
    (refi incentive high).
    """
    return float(market_rate) - float(portfolio_proxy)


class OSFILockInMeter:
    """As-of lock-in features from OSFI M4 balances (StatCan 10100006).

    The balances frame carries ``rate_pct`` = the new-lending rate per
    segment (StatCan publishes rates only for funds-advanced
    components). The portfolio-rate proxy is the trailing 60-month mean
    of new-lending rates (the book is dominated by loans originated in
    the last 5 years). Renewal wall = funds advanced 60 months ago /
    current outstanding (the 5-year-term cohort coming due).

    PIT: ``features(segment, asof)`` only uses months
    ``<= asof - pub_lag_months`` (OSFI month M is observed as-of M+2).
    """

    def __init__(self, balances: pd.DataFrame, pub_lag_months: int = 2) -> None:
        bal = balances.copy()
        bal["month"] = pd.to_datetime(bal["month"]).dt.to_period("M").dt.to_timestamp()
        self._pub_lag = pub_lag_months
        self._frames: dict[str, pd.DataFrame] = {}
        for seg, grp in bal.groupby("segment"):
            self._frames[str(seg)] = grp.sort_values("month").reset_index(drop=True)
        # Synthetic "total": balance-weighted across segments.
        tot_rows = []
        for mo, grp in bal.groupby("month"):
            tot_rows.append(self._total_row(cast("pd.Timestamp", mo), grp))
        tot = pd.DataFrame(tot_rows)
        self._frames["total"] = tot.sort_values("month").reset_index(drop=True)

    @staticmethod
    def _total_row(month: pd.Timestamp, grp: pd.DataFrame) -> dict[str, Any]:
        adv = grp["funds_advanced_m"].fillna(0.0)
        w = adv / adv.sum() if adv.sum() > 0 else None
        rate = float(np.average(grp["rate_pct"], weights=w)) if w is not None else np.nan
        return {
            "month": month,
            "segment": "total",
            "outstanding_m": float(grp["outstanding_m"].sum()),
            "funds_advanced_m": float(adv.sum()),
            "rate_pct": rate,
        }

    def features(self, segment: str, asof: pd.Timestamp) -> dict[str, Any]:
        """Lock-in features for ``segment`` as of month ``asof``."""
        empty: dict[str, Any] = {
            "available": False,
            "asof_used_month": None,
            "market_rate": None,
            "portfolio_proxy": None,
            "lockin": None,
            "momentum_3m": None,
            "momentum_12m": None,
            "renewal_wall": None,
        }
        frame = self._frames.get(segment)
        if frame is None:
            return empty
        cutoff = asof - pd.DateOffset(months=self._pub_lag)
        usable = frame[frame["month"] <= cutoff].dropna(subset=["rate_pct"])
        if len(usable) < 6:
            return empty
        rates = usable["rate_pct"].to_numpy(dtype=float)
        months = usable["month"]
        market = float(rates[-1])
        proxy = float(np.mean(rates[-60:]))
        last_month = months.iloc[-1]

        def _rate_months_ago(k: int) -> float | None:
            target = last_month - pd.DateOffset(months=k)
            hit = usable[usable["month"] == target]
            return float(hit["rate_pct"].iloc[0]) if len(hit) else None

        r3 = _rate_months_ago(3)
        r12 = _rate_months_ago(12)
        wall: float | None = None
        adv_target = last_month - pd.DateOffset(months=60)
        hit = usable[usable["month"] == adv_target]
        if len(hit) and float(usable["outstanding_m"].iloc[-1]) > 0:
            wall = float(hit["funds_advanced_m"].iloc[0]) / float(usable["outstanding_m"].iloc[-1])
        return {
            "available": True,
            "asof_used_month": last_month.strftime("%Y-%m"),
            "market_rate": _f(market),
            "portfolio_proxy": _f(proxy),
            "lockin": _f(lockin_level(market, proxy)),
            "momentum_3m": _f(market - r3) if r3 is not None else None,
            "momentum_12m": _f(market - r12) if r12 is not None else None,
            "renewal_wall": _f(wall) if wall is not None else None,
        }


# ---------------------------------------------------------------------------
# Tool 5: macro context
# ---------------------------------------------------------------------------


def macro_context(
    asof_month: str,
    *,
    news_db: str | Path | None = None,
    fred_dir: str | Path | None = None,
    balances: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Outside-world context as of ``asof_month`` ("YYYY-MM").

    Sources (each independently degradable to ``available=False``):

    - **macro_signals**: latest Macro Signal Service signal per category
      with ``month < asof_month`` (a signal *for* month M synthesises
      news published within M, so it is only usable for later origins).
    - **competitor**: latest competitor rate with ``month <= asof_month``
      and, when OSFI balances are passed, the spread vs the OSFI
      uninsured new-lending rate of the same month.
    - **house_prices**: FRED QCAR628BIS latest quarter ending on or
      before ``asof_month`` and its 4-quarter momentum.
    """
    out: dict[str, Any] = {
        "asof_month": asof_month,
        "macro_signals": {"available": False, "rows": []},
        "competitor": {"available": False},
        "house_prices": {"available": False},
    }
    if news_db is not None and Path(news_db).exists():
        try:
            con = sqlite3.connect(str(news_db))
            try:
                tables = {
                    r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if "macro_signals" in tables:
                    rows = con.execute(
                        "SELECT category, month, direction, risk, certainty "
                        "FROM macro_signals WHERE month < ? ORDER BY month",
                        (asof_month,),
                    ).fetchall()
                    latest: dict[str, dict[str, Any]] = {}
                    for cat, mo, direction, risk, certainty in rows:
                        latest[str(cat)] = {
                            "category": str(cat),
                            "month": str(mo),
                            "direction": int(direction),
                            "risk": str(risk),
                            "certainty": _f(certainty),
                        }
                    sig_rows = [latest[k] for k in sorted(latest)]
                    out["macro_signals"] = {
                        "available": bool(sig_rows),
                        "rows": sig_rows,
                    }
                if "competitor_rates" in tables:
                    row = con.execute(
                        "SELECT month, lender, product, rate_pct FROM "
                        "competitor_rates WHERE month <= ? "
                        "ORDER BY month DESC LIMIT 1",
                        (asof_month,),
                    ).fetchone()
                    if row is not None:
                        comp: dict[str, Any] = {
                            "available": True,
                            "month": str(row[0]),
                            "lender": str(row[1]),
                            "product": str(row[2]),
                            "rate_pct": _f(row[3]),
                            "spread_vs_osfi": None,
                        }
                        if balances is not None:
                            bal = balances.copy()
                            bal["month"] = pd.to_datetime(bal["month"]).dt.strftime("%Y-%m")
                            hit = bal[
                                (bal["segment"] == "uninsured") & (bal["month"] == str(row[0]))
                            ]
                            if len(hit):
                                comp["spread_vs_osfi"] = _f(
                                    float(hit["rate_pct"].iloc[0]) - float(row[3])
                                )
                        out["competitor"] = comp
            finally:
                con.close()
        except sqlite3.Error:
            pass  # context tool must never crash the caller
    if fred_dir is not None:
        hp_path = Path(fred_dir) / "QCAR628BIS.parquet"
        if hp_path.exists():
            try:
                hp = pd.read_parquet(hp_path)
                hp["date"] = pd.to_datetime(hp["date"])
                cutoff = pd.Timestamp(asof_month + "-01") + pd.offsets.MonthEnd(0)
                hp = hp[hp["date"] <= cutoff].sort_values("date")
                if len(hp) >= 5:
                    last = float(hp["value"].iloc[-1])
                    year_ago = float(hp["value"].iloc[-5])
                    out["house_prices"] = {
                        "available": True,
                        "quarter": hp["date"].iloc[-1].strftime("%Y-%m"),
                        "value": _f(last),
                        "momentum_4q_pct": _f(100.0 * (last / year_ago - 1.0)),
                    }
            except (OSError, ValueError, KeyError):
                pass
    return out
