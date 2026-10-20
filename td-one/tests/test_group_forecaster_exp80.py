"""exp80 tests: conformal calibration layer + thin/dead fallback split.

Part 1 (conformal): the layer must be PIT-safe (prefix invariance —
the exp74 bank-rule regression pattern), widen-only, a no-op below the
minimum bank size, and validated on widening under-covering histories.

Part 2 (fallback): below ``min_group_n``, a group whose recent
``n_at_risk`` is 0 ("dead") forecasts from its own series; a group
with recent observations ("thin but alive") still falls back to the
portfolio.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn
from td_one.data_gen.group_series import GroupSeries


def _gs(gid: str, n_risk: list[int], n_sw: list[int]) -> GroupSeries:
    start = date(2023, 1, 1)
    months = [start + timedelta(days=31 * i) for i in range(len(n_risk))]
    return GroupSeries(group_id=gid, months=months, n_at_risk=n_risk, n_switched=n_sw)


def _stub_fn(series: np.ndarray) -> np.ndarray:
    m = float(np.mean(series)) if len(series) else 0.0
    return np.full(5, m)


# ---------------------------------------------------------------------------
# Part 1 — conformal layer
# ---------------------------------------------------------------------------


def _undercovering_series(n: int = 28) -> np.ndarray:
    """Flat-ish history with repeated jumps the base spread under-covers."""
    rng = np.random.default_rng(7)
    base = 0.05 + 0.002 * rng.standard_normal(n)
    jumps = np.zeros(n)
    idx = [i for i in (10, 14, 18, 22) if i < n]
    jumps[idx] = 0.06
    return np.clip(base + np.cumsum(jumps) * 0.15, 0.0, 1.0)


def test_conformal_noop_below_min_bank() -> None:
    s = _undercovering_series(12)  # bank size 12 - 8 = 4 < 8
    fn_c = v2_forecaster_fn(conformal=True)
    fn_b = v2_forecaster_fn(conformal=False)
    assert fn_c(s) == pytest.approx(fn_b(s))


def test_conformal_widens_undercovering_history() -> None:
    s = _undercovering_series()
    fn_c = v2_forecaster_fn(conformal=True)
    fn_b = v2_forecaster_fn(conformal=False)
    qc, qb = fn_c(s), fn_b(s)
    assert qc[0] <= qb[0] + 1e-12
    assert qc[4] >= qb[4] - 1e-12
    assert qc[0] < qb[0] or qc[4] > qb[4]  # at least one tail strictly wider


def test_conformal_never_narrows() -> None:
    rng = np.random.default_rng(3)
    fn_c = v2_forecaster_fn(conformal=True)
    fn_b = v2_forecaster_fn(conformal=False)
    for _ in range(20):
        s = np.clip(0.08 + 0.03 * rng.standard_normal(26), 0.0, 1.0)
        qc, qb = fn_c(s), fn_b(s)
        assert qc[0] <= qb[0] + 1e-12
        assert qc[4] >= qb[4] - 1e-12
        assert np.all(np.diff(qc) >= 0)  # still a valid quantile vector
        assert np.all((qc >= 0.0) & (qc <= 1.0))
        assert qc[2] == pytest.approx(qb[2])  # median untouched


def test_conformal_pit_prefix_invariance() -> None:
    """Forecasts on a shared prefix must not depend on later data.

    The exp74 bank-rule regression pattern, adapted to a stateless fn:
    the bank for origin o is built only from series[:o], so two series
    sharing a prefix give identical forecasts on that prefix, whatever
    follows it.
    """
    s = _undercovering_series(30)
    s_alt = np.concatenate([s[:24], [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]])
    fn = v2_forecaster_fn(conformal=True)
    for o in (16, 20, 24):
        assert fn(s[:o]) == pytest.approx(fn(s_alt[:o]))


def test_conformal_conditional_scales_with_volatility() -> None:
    """Conditional arm: a volatility spike at the end widens >= plain CQR."""
    rng = np.random.default_rng(11)
    calm = 0.05 + 0.004 * rng.standard_normal(20)
    wild = 0.05 + 0.05 * rng.standard_normal(8)
    s = np.clip(np.concatenate([calm, wild]), 0.0, 1.0)
    fn_b = v2_forecaster_fn(conformal=True)
    fn_c = v2_forecaster_fn(conformal=True, conformal_conditional=True)
    width_b = fn_b(s)[4] - fn_b(s)[0]
    width_c = fn_c(s)[4] - fn_c(s)[0]
    assert width_c >= width_b - 1e-12


def test_conformal_bad_params_raise() -> None:
    with pytest.raises(ValueError, match="conformal_min_bank"):
        v2_forecaster_fn(conformal_min_bank=0)


# ---------------------------------------------------------------------------
# Part 2 — thin vs dead fallback
# ---------------------------------------------------------------------------


def _groups_dead() -> dict[str, GroupSeries]:
    return {
        # n: 8 months of 4/mo, then emptied (last 2 months 0)
        "dead": _gs("dead", [4, 4, 4, 4, 4, 4, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0]),
        # thin but alive: small n every month
        "thin": _gs("thin", [4, 4, 4, 4, 4, 4, 4, 4], [1, 0, 1, 0, 1, 0, 1, 0]),
        "__portfolio__": _gs(
            "__portfolio__",
            [400, 400, 400, 400, 400, 400, 400, 400],
            [40, 44, 36, 40, 44, 36, 40, 44],
        ),
    }


def test_dead_group_uses_own_forecast() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    out = gf.forecast(_groups_dead())
    assert not out["dead"].used_fallback
    own_rate = np.array(_groups_dead()["dead"].attrition_rate)
    assert out["dead"].quantiles == pytest.approx(np.full(5, float(np.mean(own_rate))))


def test_thin_alive_group_still_falls_back() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    out = gf.forecast(_groups_dead())
    assert out["thin"].used_fallback
    port_rate = np.array(_groups_dead()["__portfolio__"].attrition_rate)
    assert out["thin"].quantiles == pytest.approx(np.full(5, float(np.mean(port_rate))))


def test_dead_branch_disabled_reproduces_old_behavior() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200, dead_zero_months=0)
    out = gf.forecast(_groups_dead())
    assert out["dead"].used_fallback


def test_bad_dead_zero_months_raises() -> None:
    with pytest.raises(ValueError, match="dead_zero_months"):
        GroupForecaster(_stub_fn, dead_zero_months=-1)


def test_dead_detection_uses_recent_window_not_total() -> None:
    """A group that was big but emptied recently is dead, not big."""
    groups = {
        "collapsed": _gs(
            "collapsed",
            [500, 500, 500, 500, 500, 500, 0, 0],
            [50, 50, 50, 50, 50, 50, 0, 0],
        ),
        "__portfolio__": _gs(
            "__portfolio__",
            [400, 400, 400, 400, 400, 400, 400, 400],
            [40, 44, 36, 40, 44, 36, 40, 44],
        ),
    }
    # Total (3000) >= min_group_n: never in the fallback branch at all.
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    out = gf.forecast(groups)
    assert not out["collapsed"].used_fallback
