"""Tests for the quantile-spread fix in ``v2_forecaster_fn`` (exp78).

Defect (flagged by exp72): the production group forecaster emitted flat
point-mass quantiles — the router was constructed but never fitted, so
``predict`` raised ``AssertionError`` and the blanket ``except`` returned
``naive_q[0]``, the last value tiled across all quantile levels. These
tests pin the fixed behavior: PIT-safe spread from naive one-step errors,
exact reproduction of the old output under ``quantile_spread=False``,
and intact ``GroupForecaster`` fallback semantics.
"""

import numpy as np
import pytest

from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn
from td_one.data_gen.group_series import GroupSeries


def _noisy_series(n: int = 40, sd: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.05 + np.cumsum(rng.normal(0.0, sd, n))


def test_spread_default_is_nonflat_on_noisy_series() -> None:
    series = _noisy_series()
    q = v2_forecaster_fn()(series)
    assert q.shape == (5,)
    assert q[-1] - q[0] > 0.002  # q90-q10 width is real, not a point mass
    assert np.all(np.diff(q) > 0)  # strictly increasing on noisy history


def test_spread_false_reproduces_old_point_mass() -> None:
    series = _noisy_series()
    q = v2_forecaster_fn(quantile_spread=False)(series)
    assert q == pytest.approx(np.full(5, series[-1]))


def test_spread_monotone_and_within_unit_interval() -> None:
    for seed in range(5):
        q = v2_forecaster_fn()(_noisy_series(seed=seed, sd=0.01))
        assert np.all(np.diff(q) >= 0)
        assert np.all(q >= 0.0) and np.all(q <= 1.0)


def test_spread_widens_with_noise() -> None:
    quiet = _noisy_series(sd=0.0005, seed=1)
    loud = _noisy_series(sd=0.01, seed=1)
    w_quiet = np.ptp(v2_forecaster_fn()(quiet))
    w_loud = np.ptp(v2_forecaster_fn()(loud))
    assert w_loud > 3.0 * w_quiet


def test_spread_constant_series_is_valid_and_degenerate() -> None:
    q = v2_forecaster_fn()(np.full(20, 0.03))
    assert np.all(np.isfinite(q))
    # Effectively degenerate (within the numerical sd floor), but valid.
    assert np.ptp(q) < 1e-3
    assert q == pytest.approx(np.full(5, 0.03), abs=1e-3)


def test_spread_short_series_is_valid() -> None:
    q1 = v2_forecaster_fn()(np.array([0.04]))
    assert q1 == pytest.approx(np.full(5, 0.04))
    q2 = v2_forecaster_fn()(np.array([0.04, 0.05]))
    assert np.all(np.isfinite(q2)) and np.all(np.diff(q2) >= 0)


def test_spread_uses_only_given_history() -> None:
    # PIT safety: the forecast for a truncated history must not depend on
    # values after the truncation point (callers pass history <= origin).
    a = _noisy_series(seed=3)
    b = a.copy()
    b[25:] = 0.5  # wildly different future
    assert v2_forecaster_fn()(a[:25]) == pytest.approx(v2_forecaster_fn()(b[:25]))


def test_spread_methods_both_nonflat() -> None:
    series = _noisy_series()
    for method in ("empirical", "gaussian"):
        q = v2_forecaster_fn(spread_method=method)(series)
        assert q[-1] - q[0] > 0.002, method
        assert np.all(np.diff(q) >= 0), method


def test_spread_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="spread_method"):
        v2_forecaster_fn(spread_method="bootstrap")


def _gs(gid: str, n_risk: list[int], n_sw: list[int]) -> GroupSeries:
    from datetime import date

    months = [date(2024, m, 1) for m in range(1, len(n_risk) + 1)]
    return GroupSeries(group_id=gid, months=months, n_at_risk=n_risk, n_switched=n_sw)


def test_group_forecaster_end_to_end_with_spread() -> None:
    # 12 months of noisy group history; big group forecasts itself, small
    # group falls back to the (also spread) portfolio forecast.
    rng = np.random.default_rng(7)
    n = 12
    big_sw = np.maximum(1, (900 * (0.05 + rng.normal(0, 0.004, n))).astype(int))
    groups = {
        "big": _gs("big", [900] * n, big_sw.tolist()),
        "small": _gs("small", [5] * n, [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]),
        "__portfolio__": _gs("__portfolio__", [905] * n, (big_sw + 1).tolist()),
    }
    gf = GroupForecaster(v2_forecaster_fn(), min_group_n=200)
    out = gf.forecast(groups)
    assert not out["big"].used_fallback
    assert out["small"].used_fallback
    assert out["big"].quantiles[-1] > out["big"].quantiles[0]  # spread, not flat
    assert out["small"].quantiles[-1] > out["small"].quantiles[0]
