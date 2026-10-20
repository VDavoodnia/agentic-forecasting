"""Tests for the exp82 pooled router in ``group_forecaster``.

exp79 showed the real v2 router, fitted **per series**, loses inside
the group forecaster: each series gives it at most ~13 backtest
records. exp82's hypothesis is that a **pooled** fit — one router
fitted on rolling-origin backtest records pooled across every series
in the panel, the exp47 recipe — fixes the starvation.

These tests pin the contract:

- the pooled path actually calls ``fit()`` (on the pooled record set,
  which must be larger than any single series' records) before
  ``predict()``;
- a failure on that path **raises**, never silently degrades (the
  exp78 bug class);
- prefix invariance (PIT): panel data after the forecast origin cannot
  change the output;
- the returned quantiles are the router's combined median plus the
  exp78 Gaussian spread.
"""

import numpy as np
import pytest

from td_one.agent import group_forecaster
from td_one.agent.group_forecaster import (
    QUANTILE_LEVELS,
    _PooledRouter,
    pooled_router_forecaster_fn,
)
from td_one.agent.proposed_method_v2 import ProposedForecasterV2


def _noisy_series(n: int = 40, sd: float = 0.004, seed: int = 0, level: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.normal(0.0, sd, n))


def _panel(n: int = 40) -> dict[str, np.ndarray]:
    return {
        "g1": _noisy_series(n, seed=1, level=0.04),
        "g2": _noisy_series(n, seed=2, level=0.06),
        "g3": _noisy_series(n, seed=3, level=0.05),
        "__portfolio__": _noisy_series(n, seed=4, level=0.05),
    }


def test_pooled_factory_is_public() -> None:
    assert callable(pooled_router_forecaster_fn)
    assert hasattr(group_forecaster, "pooled_router_forecaster_fn")


def test_pooled_output_is_valid_nonflat_quantiles() -> None:
    panel = _panel()
    fn = pooled_router_forecaster_fn(panel)
    q = fn(panel["g1"])
    assert q.shape == (len(QUANTILE_LEVELS),)
    assert np.all(np.diff(q) >= 0)  # monotone
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert float(np.max(q) - np.min(q)) > 0.0  # spread, not a point mass


def test_pooled_actually_fits_on_the_pool_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    calls: list[str] = []
    record_counts: list[int] = []
    orig_fit = ProposedForecasterV2.fit
    orig_predict = ProposedForecasterV2.predict

    def spy_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        calls.append("fit")
        record_counts.append(len(records))
        return orig_fit(self, records, feature_names)

    def spy_predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("predict")
        return orig_predict(self, *args, **kwargs)

    monkeypatch.setattr(ProposedForecasterV2, "fit", spy_fit)
    monkeypatch.setattr(ProposedForecasterV2, "predict", spy_predict)

    fn = pooled_router_forecaster_fn(panel)
    fn(panel["g1"][:25])
    assert calls[0] == "fit"
    assert "predict" in calls
    # Pooled: 4 panel series x (25 - 8) backtest origins each.
    assert record_counts[0] == 4 * (25 - 8)
    # ...strictly more than any per-series fit could see (25 - 8).
    assert record_counts[0] > 25 - 8


def test_pooled_fit_failure_raises_not_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(ProposedForecasterV2, "fit", boom_fit)
    fn = pooled_router_forecaster_fn(_panel())
    with pytest.raises(RuntimeError, match="fit exploded"):
        fn(_panel()["g1"])


def test_pooled_predict_failure_raises_not_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom_predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("predict exploded")

    monkeypatch.setattr(ProposedForecasterV2, "predict", boom_predict)
    fn = pooled_router_forecaster_fn(_panel())
    with pytest.raises(RuntimeError, match="predict exploded"):
        fn(_panel()["g1"])


def test_pooled_prefix_invariance() -> None:
    """PIT: perturbing panel tails beyond the origin changes nothing."""
    panel = _panel()
    target = panel["g1"][:22]
    q_before = pooled_router_forecaster_fn(panel)(target)

    perturbed = {k: v.copy() for k, v in panel.items()}
    for v in perturbed.values():
        v[22:] = 0.99  # wild future values, must be invisible at origin 22
    q_after = pooled_router_forecaster_fn(perturbed)(target)
    np.testing.assert_array_equal(q_before, q_after)


def test_pooled_median_is_router_center_plus_zero_spread_offset() -> None:
    panel = _panel()
    core = _PooledRouter(panel)
    series = panel["g2"][:30]
    center = core.center(series)
    q = pooled_router_forecaster_fn(panel)(series)
    # Gaussian spread: the 0.5-level offset is exactly 0, so q50 == center.
    assert q[2] == pytest.approx(center)


def test_pooled_combined_q9_median_matches_center() -> None:
    panel = _panel()
    core = _PooledRouter(panel)
    series = panel["g3"][:28]
    q9 = core.combined_q9(series)
    assert q9.shape == (9,)
    assert core.center(series) == pytest.approx(float(q9[4]))


def test_pooled_unknown_spec_raises() -> None:
    with pytest.raises(ValueError, match="regime_spec"):
        pooled_router_forecaster_fn(_panel(), regime_spec="atlantis")


def test_pooled_empty_panel_raises_on_call() -> None:
    fn = pooled_router_forecaster_fn({})
    with pytest.raises(ValueError, match="pool"):
        fn(_noisy_series(20, seed=9))


def test_pooled_too_short_panel_raises_on_call() -> None:
    short = {"g1": _noisy_series(8, seed=5)}  # < backtest_start + 1
    fn = pooled_router_forecaster_fn(short)
    with pytest.raises(ValueError, match="pool"):
        fn(short["g1"])


def test_pooled_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="features"):
        pooled_router_forecaster_fn(_panel(), features=("not_a_feature",))


def test_pooled_normalize_crps_makes_fit_scale_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """exp82 R2e: fit() receives per-record scale-free CRPS.

    With normalize_crps, every record handed to fit() has its CRPS
    divided by that record's naive CRPS (scale-free expert selection —
    the pooled raw mean is otherwise dominated by the largest-scale
    series).
    """
    seen: list[list[dict]] = []
    orig_fit = ProposedForecasterV2.fit

    def spy_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        seen.append(records)
        return orig_fit(self, records, feature_names)

    monkeypatch.setattr(ProposedForecasterV2, "fit", spy_fit)
    fn = pooled_router_forecaster_fn(_panel(), normalize_crps=True)
    fn(_panel()["g1"][:25])
    assert seen, "fit was never called"
    for rec in seen[0]:
        assert rec["crps"]["naive"] == pytest.approx(1.0)


def test_pooled_normalize_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[dict]] = []
    orig_fit = ProposedForecasterV2.fit

    def spy_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        seen.append(records)
        return orig_fit(self, records, feature_names)

    monkeypatch.setattr(ProposedForecasterV2, "fit", spy_fit)
    fn = pooled_router_forecaster_fn(_panel())
    fn(_panel()["g1"][:25])
    assert seen
    assert any(rec["crps"]["naive"] != pytest.approx(1.0) for rec in seen[0])
