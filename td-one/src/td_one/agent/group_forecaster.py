"""Group-level attrition forecasting with fallback to global.

For each borrower group, forecasts the group's monthly attrition series
using the injected ``forecaster_fn``. Groups with fewer than
``min_group_n`` at-risk borrowers in the training window fall back to
the global (portfolio) forecast — shrinkage toward global, same principle
as the decision policy's James-Stein.

The ``forecaster_fn`` contract: ``(series: np.ndarray) -> np.ndarray``
mapping a 1D history of attrition rates to a quantile forecast vector.
In production this is :func:`v2_forecaster_fn` (spread-naive, exp78);
in tests a stub.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class GroupForecast:
    group_id: str
    quantiles: np.ndarray  # forecast quantile vector
    used_fallback: bool
    n_at_risk_train: int


class GroupForecaster:
    """Forecast each group's attrition series, with global fallback."""

    def __init__(
        self,
        forecaster_fn: Callable[[np.ndarray], np.ndarray],
        min_group_n: int = DEFAULT_MIN_GROUP_N,
    ) -> None:
        if min_group_n < 1:
            raise ValueError(f"min_group_n must be >= 1, got {min_group_n}")
        self.forecaster_fn = forecaster_fn
        self.min_group_n = min_group_n

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
            if total_risk < self.min_group_n:
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


def v2_forecaster_fn(
    h: int = 1,
    *,
    quantile_spread: bool = True,
    spread_method: str = "gaussian",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a group-series forecaster_fn: naive center + PIT-safe spread.

    Center is the last observed value (naive); the 5-level quantile
    vector spreads around it with offsets from the series' own trailing
    naive one-step errors (see :func:`_spread_offsets`). The spread is
    what the production forecaster was missing.

    History (exp78 root-cause): this function previously constructed a
    ``ProposedForecasterV2`` and called ``predict`` on it *without ever
    calling fit*, so ``assert self._fitted`` raised and the blanket
    ``except`` returned ``naive_q[0]`` — the last value tiled across
    every quantile level, a point mass. Additionally the single naive
    expert's quantiles were themselves tiled (``(h, 5)`` shape, not the
    router's expected 9-level grid), so the router could never have
    produced spread even if it had run. The dead router call is
    intentionally gone; what remains is honest spread-naive.

    Parameters
    ----------
    h:
        Kept for call-site compatibility; the fn returns the 5-level
        next-step quantile vector.
    quantile_spread:
        ``False`` reproduces the exact pre-exp78 point-mass output
        (tiled last value) for reproducibility of old results.
    spread_method:
        ``"gaussian"`` (default) or ``"empirical"`` offsets. exp78's
        ablation picked Gaussian: on SimulatorV2 control it beat the
        empirical offsets on CRPS and coverage in both group and global
        scopes (the empirical quantiles of ~24 trailing diffs are too
        noisy at the 0.1/0.9 tails). Other values raise ``ValueError``.

    """
    if spread_method not in ("empirical", "gaussian"):
        raise ValueError(f"spread_method must be 'empirical' or 'gaussian', got {spread_method!r}")

    def _forecast(series: np.ndarray) -> np.ndarray:
        s = np.asarray(series, dtype=float)
        last = float(s[-1]) if len(s) else 0.0
        if not quantile_spread:
            # Exact pre-exp78 behavior: tiled last value (point mass).
            return np.full(len(QUANTILE_LEVELS), last)
        q = last + _spread_offsets(s, spread_method)
        # Attrition rates live in [0, 1]; clipping is monotone so the
        # vector stays a valid quantile function.
        return np.clip(np.maximum.accumulate(q), 0.0, 1.0)

    return _forecast
