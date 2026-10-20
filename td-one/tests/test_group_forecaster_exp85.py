"""exp85 tests: shift-adaptive conformal modes of ``v2_forecaster_fn``.

The adaptive modes (reset / decay / cp / bc / cp_bc) must be PIT-safe
(prefix invariance), must leave ``conformal_adaptive="none"`` output
bit-identical to the exp80 layer, must validate their parameters,
and must actually react to a level shift in the series history.
"""

import numpy as np
import pytest

from td_one.agent.group_forecaster import v2_forecaster_fn


def _shifted_series(n_pre: int = 30, n_post: int = 14, seed: int = 5) -> np.ndarray:
    """Attrition-rate-like series with a level shift at index n_pre."""
    rng = np.random.default_rng(seed)
    pre = 0.05 + 0.004 * rng.standard_normal(n_pre)
    post = 0.16 + 0.01 * rng.standard_normal(n_post)
    return np.clip(np.concatenate([pre, post]), 0.0, 1.0)


def _flat_series(n: int = 40, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(0.05 + 0.004 * rng.standard_normal(n), 0.0, 1.0)


def test_adaptive_none_reproduces_exp80_exactly() -> None:
    s = _shifted_series()
    fn_default = v2_forecaster_fn(conformal=True)
    fn_none = v2_forecaster_fn(conformal=True, conformal_adaptive="none")
    assert fn_none(s) == pytest.approx(fn_default(s))


def test_adaptive_modes_run_and_stay_valid() -> None:
    s = _shifted_series()
    for mode in ("reset", "decay", "cp", "bc", "cp_bc"):
        q = v2_forecaster_fn(conformal=True, conformal_adaptive=mode)(s)
        assert np.all(np.diff(q) >= 0), mode  # monotone quantile vector
        assert np.all((q >= 0.0) & (q <= 1.0)), mode


def test_adaptive_modes_pit_prefix_invariance() -> None:
    """Forecasts on a shared prefix must not depend on later data."""
    s = _shifted_series(30, 16)
    s_alt = np.concatenate([s[:36], np.full(10, 0.9)])
    for mode in ("reset", "decay", "cp", "bc", "cp_bc"):
        fn = v2_forecaster_fn(conformal=True, conformal_adaptive=mode)
        for o in (28, 32, 36):
            assert fn(s[:o]) == pytest.approx(fn(s_alt[:o])), (mode, o)


def test_bc_mode_shifts_interval_up_after_level_shift() -> None:
    """After an upward level shift the base center lags.

    The bias-corrected interval's low edge must sit above the plain
    conformal layer's.
    """
    s = _shifted_series()
    q_plain = v2_forecaster_fn(conformal=True)(s)
    q_bc = v2_forecaster_fn(conformal=True, conformal_adaptive="bc")(s)
    assert q_bc[0] > q_plain[0]


def test_cusum_detector_modes_valid_and_pit() -> None:
    """exp85 round 3: the CUSUM detector gate.

    Runs through the group forecaster, stays a valid quantile
    vector, and is PIT-safe.
    """
    s = _shifted_series(30, 16)
    s_alt = np.concatenate([s[:36], np.full(10, 0.9)])
    for det in ("cusum", "errors"):
        for mode in ("reset", "cp", "cp_bc"):
            fn = v2_forecaster_fn(conformal=True, conformal_adaptive=mode, conformal_detector=det)
            q = fn(s)
            assert np.all(np.diff(q) >= 0), (det, mode)
            for o in (28, 32, 36):
                assert fn(s[:o]) == pytest.approx(fn(s_alt[:o])), (det, mode, o)


def test_adaptive_noop_below_min_bank() -> None:
    s = _flat_series(12)  # bank = 12 - 8 = 4 < 8
    fn_b = v2_forecaster_fn(conformal=False)
    for mode in ("reset", "decay", "cp", "bc", "cp_bc"):
        fn = v2_forecaster_fn(conformal=True, conformal_adaptive=mode)
        assert fn(s) == pytest.approx(fn_b(s)), mode


def test_adaptive_bad_params_raise() -> None:
    with pytest.raises(ValueError, match="conformal_adaptive"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="bogus")
    with pytest.raises(ValueError, match="requires conformal"):
        v2_forecaster_fn(conformal_adaptive="bc")
    with pytest.raises(ValueError, match="conformal_detector"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="reset", conformal_detector="bogus")
    with pytest.raises(ValueError, match="conformal_half_life"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="decay", conformal_half_life=0.0)
    with pytest.raises(ValueError, match="conformal_pre_factor"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="cp", conformal_pre_factor=1.5)
    with pytest.raises(ValueError, match="conformal_min_post"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="reset", conformal_min_post=0)
    with pytest.raises(ValueError, match="conformal_bias_lambda"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="bc", conformal_bias_lambda=1.0)
    with pytest.raises(ValueError, match="conformal_bias_n0"):
        v2_forecaster_fn(conformal=True, conformal_adaptive="bc", conformal_bias_n0=-1.0)
