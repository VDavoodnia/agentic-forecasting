"""Tests for td_one.uq: methods, features, and the PIT-safe study driver."""

import numpy as np
import pytest

from td_one.uq import methods as uq_methods
from td_one.uq.evaluate import build_cases, known_bank
from td_one.uq.features import macro_features, numeric_features, story_features

M = uq_methods

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def _case(origin: int, h: int, m: float = 0.0, half: float = 1.0, y: float = 0.0):
    q = np.array(
        [
            m - 3 * half,
            m - 2 * half,
            m - half,
            m - 0.5 * half,
            m,
            m + 0.5 * half,
            m + half,
            m + 2 * half,
            m + 3 * half,
        ]
    )
    return M.ForecastCase(
        origin=origin,
        h=h,
        levels=LEVELS,
        q=q,
        y=y,
        context=np.linspace(-1, 1, 30),
    )


def _bank(n: int, resid: float, h: int = 1, start_origin: int = 0):
    # Bank cases whose realizeds are known long before later origins.
    return [
        _case(origin=start_origin + i, h=h, m=0.0, half=1.0, y=resid if i % 2 == 0 else -resid)
        for i in range(n)
    ]


def test_finite_quantile_correction():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # ceil((4+1)*0.8)/4 -> k=4 -> max
    assert M.finite_quantile(x, 0.8) == 4.0
    assert M.finite_quantile(x, 0.5) == 3.0  # ceil(2.5)-1=2 -> x[2]
    with pytest.raises(ValueError):
        M.finite_quantile(np.array([]), 0.8)


def test_weighted_quantile():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.0, 0.0, 1.0])
    assert M.weighted_quantile(x, w, 0.8) == 3.0
    w2 = np.array([0.9, 0.05, 0.05])
    assert M.weighted_quantile(x, w2, 0.8) == 1.0
    with pytest.raises(ValueError):
        M.weighted_quantile(x, np.zeros(3), 0.5)


def test_scale_grid_preserves_median_and_monotonicity():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    out = M.scale_grid(LEVELS, q, 2.0, 0.5)
    assert out[4] == pytest.approx(5.0)  # median fixed
    assert bool((np.diff(out) > 0).all())
    assert out[2] == pytest.approx(5.0 + 2.0 * (3.0 - 5.0))
    assert out[6] == pytest.approx(5.0 + 0.5 * (7.0 - 5.0))


def test_scales_from_interval():
    case = _case(0, 1, m=0.0, half=1.0)
    r_lo, r_hi = M.scales_from_interval(case, -2.0, 3.0)
    assert r_lo == pytest.approx(2.0)
    assert r_hi == pytest.approx(3.0)


def test_arm_native_and_fixed_expansion_fallback():
    case = _case(10, 1, y=0.5)
    assert M.arm_native(case, []) == (1.0, 1.0)
    # Bank too small -> fallback to 1.0
    assert M.arm_fixed_expansion(case, _bank(3, 0.5)) == (1.0, 1.0)


def test_arm_fixed_expansion_scales_to_cover():
    # Past residuals 2x the native half-width -> r* should be >= ~2
    bank = _bank(20, 2.0)
    case = _case(100, 1, y=0.0)
    r_lo, r_hi = M.arm_fixed_expansion(case, bank)
    assert r_lo >= 1.9 and r_hi >= 1.9


def test_arm_rolling_conformal_widens_on_large_residuals():
    bank = _bank(20, 3.0)
    case = _case(100, 1, y=0.0)
    r_lo, r_hi = M.arm_rolling_conformal(case, bank)
    assert r_lo > 2.5 and r_hi > 2.5
    # Tiny bank -> fallback
    assert M.arm_rolling_conformal(case, _bank(2, 3.0)) == (1.0, 1.0)


def test_arm_aci_reacts_to_misses():
    # Deteriorating regime: residuals ramp up, past-quantile always lags ->
    # persistent misses -> alpha must drop (intervals widen).
    ramp = np.linspace(1.0, 5.0, 20)
    bank = [_case(i, 1, m=0.0, half=0.1, y=float(r)) for i, r in enumerate(ramp)]
    a_now, path = M._aci_alpha_path(bank, 1, alpha=0.2, gamma=0.05, window=24)
    assert a_now < 0.2
    assert len(path) == 21
    # Improving regime: residuals decay -> covered -> alpha rises.
    decay = np.linspace(5.0, 1.0, 20)
    bank2 = [_case(i, 1, m=0.0, half=10.0, y=float(r)) for i, r in enumerate(decay)]
    a_now2, _ = M._aci_alpha_path(bank2, 1, alpha=0.2, gamma=0.05, window=24)
    assert a_now2 > 0.2


def test_arm_cqr_asymmetric():
    # Downside misses only -> r_lo > r_hi
    bank = [_case(i, 1, m=0.0, half=1.0, y=-3.0) for i in range(20)]
    case = _case(100, 1, y=0.0)
    r_lo, r_hi = M.arm_cqr(case, bank)
    assert r_lo > 2.5
    assert r_hi == pytest.approx(1.0)  # no upside misses -> minimal widening


def test_arm_weighted_conformal_prefers_similar_vol():
    bank = _bank(20, 1.0)
    for c in bank:
        c.feats["vol_12"] = 5.0  # all stale, high vol
    bank.append(_case(50, 1, m=0.0, half=1.0, y=0.5))
    bank[-1].feats["vol_12"] = 0.1  # one recent, calm case
    case = _case(100, 1, y=0.0)
    case.feats["vol_12"] = 0.1
    r_lo, _ = M.arm_weighted_conformal(case, bank, rho=0.5)
    # Weighted toward the calm case (resid 0.5) -> modest widening
    assert r_lo < 1.5


def test_arm_conformal_rel_scales():
    # Bank cases with |y-m| = 2x their native half-width -> s* ~ 2
    bank = [_case(i, 1, m=0.0, half=1.0, y=2.0) for i in range(20)]
    case = _case(100, 1, y=0.0)
    r_lo, r_hi = M.arm_conformal_rel(case, bank)
    assert r_lo == pytest.approx(2.0, abs=0.3)
    assert M.arm_conformal_rel(case, _bank(2, 1.0)) == (1.0, 1.0)


def test_arm_aci_rel_reacts_to_regime():
    # Relative scores ramp 0.5 -> 3: ACI alpha drops -> s* from high quantile
    ys = np.linspace(0.5, 3.0, 20)
    bank = [_case(i, 1, m=0.0, half=1.0, y=float(v)) for i, v in enumerate(ys)]
    case = _case(100, 1, y=0.0)
    r_lo, _ = M.arm_aci_rel(case, bank)
    assert r_lo > 1.5
    assert M.arm_aci_rel(case, _bank(2, 0.5)) == (1.0, 1.0)


def test_arm_meta_fallback_matches_conformal_with_tiny_bank():
    bank = _bank(10, 1.0)
    case = _case(100, 1, y=0.0)
    for c in bank + [case]:
        c.feats.update(dict.fromkeys(M.META_FEATURES_NUMERIC, 0.0))
    assert M.arm_meta(case, bank) == M.arm_rolling_conformal(case, bank)


def test_known_bank_pit_rule():
    # Case at origin 10, h=6 -> realized at index 15; known only from origin 16.
    cases = [_case(10, 6, y=1.0), _case(5, 1, y=0.5)]
    assert len(known_bank(cases, 15)) == 1  # only the (5,1) case
    assert len(known_bank(cases, 16)) == 2


def test_build_cases_structure():
    rows = [
        {
            "config_id": "x",
            "origin": 12,
            "levels": [0.1, 0.5, 0.9],
            "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "realized": [3.5, 4.5],
            "context": [1.0] * 12,
        }
    ]
    out = build_cases(rows)
    assert len(out["x"]) == 2
    assert out["x"][0].h == 1 and out["x"][0].y == 3.5
    assert out["x"][1].q.tolist() == [2.0, 4.0, 6.0]


def test_numeric_features_pit_safe():
    ctx = np.linspace(0, 1, 30)
    bank = _bank(10, 0.5)
    feats = numeric_features(ctx, 1, LEVELS, _case(0, 1).q, bank)
    for key in M.META_FEATURES_NUMERIC:
        assert key in feats, key
        assert np.isfinite(feats[key])
    assert feats["err_count3"] == 3.0
    assert feats["err_count6"] == 6.0


def test_story_and_macro_features_missing_safe():
    sf = story_features(None)
    assert sf["st_available"] == 0.0
    mf = macro_features(None)
    assert mf["mac_available"] == 0.0
    sf2 = story_features(
        {
            "shift_type": "rate",
            "direction": "up",
            "magnitude_bucket": "medium",
            "months_from_now": 2,
            "timing_confidence": 0.7,
            "driving_trends": ["a", "b"],
        }
    )
    assert sf2["st_available"] == 1.0 and sf2["st_shift"] == 1.0
    assert sf2["st_direction"] == 1.0 and sf2["st_trend_count"] == 2.0
    # None-valued fields (no shift detected) -> neutral fills, no crash
    sf3 = story_features(
        {
            "shift_type": "none",
            "direction": "none",
            "magnitude_bucket": "none",
            "months_from_now": None,
            "timing_confidence": None,
            "driving_trends": [],
        }
    )
    assert sf3["st_months_from_now"] == -1.0 and sf3["st_timing_conf"] == 0.0
    mf2 = macro_features(
        {"inflation": {"category": "inflation", "direction": 2, "risk": "high", "certainty": 0.8}}
    )
    assert mf2["mac_available"] == 1.0 and mf2["mac_inflation"] == 2.0
    assert mf2["mac_n_high"] == 1.0


def test_study_pit_prefix_equals_full():
    """Arms produce identical output on prefix cases.

    Whether or not future cases exist must not change past output — the
    PIT gate is structural.
    """
    from td_one.uq.evaluate import run_study  # noqa: PLC0415

    rng = np.random.default_rng(7)
    cases_full: list = []
    for origin in range(0, 30, 2):
        ctx = rng.normal(0, 1, origin + 20)
        q = np.quantile(ctx[-5:], [0.1, 0.5, 0.9])
        for h in (1, 2):
            cases_full.append(
                M.ForecastCase(
                    origin=origin,
                    h=h,
                    levels=np.array([0.1, 0.5, 0.9]),
                    q=q.copy(),
                    y=float(rng.normal()),
                    context=ctx.copy(),
                )
            )
    arms = {"native": M.arm_native, "roll": M.arm_rolling_conformal}
    res_full = run_study({"c": cases_full}, arms)
    res_pre = run_study({"c": [c for c in cases_full if c.origin < 14]}, arms)

    def key(r):
        return (r.origin, r.h, r.arm)

    d_full = {key(r): r for r in res_full}
    d_pre = {key(r): r for r in res_pre}
    for k, r in d_pre.items():
        assert d_full[k].crps == pytest.approx(r.crps)
        assert d_full[k].coverage80 == r.coverage80
