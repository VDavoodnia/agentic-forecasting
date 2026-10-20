"""Tests for the exp79 center/router options in ``group_forecaster``.

exp79 asks two questions exp78 left open: (1) is a trailing-mean center
better than the naive last-value center, and (2) does the *real*
``ProposedForecasterV2`` — fitted properly — belong inside the group
forecaster?

The router tests pin the anti-regression contract for the exp78 bug
class: the router path must actually call ``fit()`` before ``predict()``,
and a failure on that path must **raise**, never silently degrade to a
naive point mass (that silent degradation is exactly how the exp78
defect shipped).
"""

import numpy as np
import pytest

from td_one.agent import group_forecaster
from td_one.agent.group_forecaster import router_forecaster_fn, v2_forecaster_fn


def _noisy_series(n: int = 40, sd: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.05 + np.cumsum(rng.normal(0.0, sd, n))


# --- center selection on v2_forecaster_fn ---------------------------------


def test_center_trailing_mean_is_default_and_median_is_trailing_mean() -> None:
    # exp79: the trailing-mean center is the production default.
    series = _noisy_series()
    q = v2_forecaster_fn()(series)
    # Gaussian spread: the 0.5-level offset is exactly 0, so q50 == center.
    assert q[2] == pytest.approx(float(np.mean(series[-6:])))


def test_center_naive_median_is_last_value() -> None:
    # exp78 behavior stays available behind the flag.
    series = _noisy_series()
    q = v2_forecaster_fn(center="naive")(series)
    assert q[2] == pytest.approx(series[-1])


def test_center_trailing_mean_window_is_configurable() -> None:
    series = _noisy_series()
    q = v2_forecaster_fn(center="trailing_mean", center_window=3)(series)
    assert q[2] == pytest.approx(float(np.mean(series[-3:])))


def test_center_unknown_raises() -> None:
    with pytest.raises(ValueError, match="center"):
        v2_forecaster_fn(center="median_of_universe")


def test_center_trailing_mean_short_series_uses_what_exists() -> None:
    series = np.array([0.04, 0.06])
    q = v2_forecaster_fn(center="trailing_mean")(series)
    assert q[2] == pytest.approx(0.05)


# --- router_forecaster_fn ---------------------------------------------------


def test_router_output_is_valid_nonflat_quantiles() -> None:
    series = _noisy_series(n=29, seed=5)
    q = router_forecaster_fn()(series)
    assert q.shape == (5,)
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q) >= 0)
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert q[-1] - q[0] > 0.0005  # real spread, not the exp78 point mass


def test_router_actually_fits_before_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    from td_one.agent.proposed_method_v2 import ProposedForecasterV2

    calls: list[str] = []
    orig_fit = ProposedForecasterV2.fit
    orig_predict = ProposedForecasterV2.predict

    def spy_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        calls.append("fit")
        return orig_fit(self, records, feature_names)

    def spy_predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("predict")
        return orig_predict(self, *args, **kwargs)

    monkeypatch.setattr(ProposedForecasterV2, "fit", spy_fit)
    monkeypatch.setattr(ProposedForecasterV2, "predict", spy_predict)

    router_forecaster_fn()(_noisy_series(n=29, seed=5))
    assert "fit" in calls, "router path never called fit() — exp78 bug class"
    assert "predict" in calls
    assert calls.index("fit") < calls.index("predict")


def test_router_fit_failure_raises_not_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from td_one.agent.proposed_method_v2 import ProposedForecasterV2

    def boom_fit(self, records, feature_names):  # type: ignore[no-untyped-def]
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(ProposedForecasterV2, "fit", boom_fit)
    with pytest.raises(RuntimeError, match="fit exploded"):
        router_forecaster_fn()(_noisy_series(n=29, seed=5))


def test_router_predict_failure_raises_not_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from td_one.agent.proposed_method_v2 import ProposedForecasterV2

    def boom_predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("predict exploded")

    monkeypatch.setattr(ProposedForecasterV2, "predict", boom_predict)
    with pytest.raises(RuntimeError, match="predict exploded"):
        router_forecaster_fn()(_noisy_series(n=29, seed=5))


def test_router_short_series_raises_value_error() -> None:
    with pytest.raises(ValueError, match="history"):
        router_forecaster_fn()(np.array([0.05, 0.06, 0.04]))


def test_router_uses_only_given_history() -> None:
    # PIT safety: forecast on a prefix must not depend on later values.
    a = _noisy_series(n=40, seed=3)
    b = a.copy()
    b[29:] = 0.5
    fn = router_forecaster_fn()
    assert fn(a[:29]) == pytest.approx(fn(b[:29]))


def test_router_m4_spec_variant_runs() -> None:
    q = router_forecaster_fn(regime_spec="m4")(_noisy_series(n=29, seed=5))
    assert q.shape == (5,)
    assert np.all(np.diff(q) >= 0)


def test_router_unknown_spec_raises() -> None:
    with pytest.raises(ValueError, match="regime_spec"):
        router_forecaster_fn(regime_spec="astrology")


def test_router_factory_is_public() -> None:
    # The experiment and docs import it from the module, not a private.
    assert callable(group_forecaster.router_forecaster_fn)
