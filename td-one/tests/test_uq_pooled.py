"""Tests for td_one.uq.pooled: cross-series pooled conformal banks (exp81)."""

import numpy as np
import pytest

from td_one.uq import methods as uq_methods
from td_one.uq import pooled as uq_pooled

M = uq_methods
P = uq_pooled

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
    return M.ForecastCase(origin=origin, h=h, levels=LEVELS, q=q, y=y, context=np.zeros(30))


def _family(n_configs=3, rounds=6, step=6, horizon=18, half=1.0, y=0.0):
    """Synthetic family on a uniform origin grid, like the exp76 sweep."""
    out = {}
    for ci in range(n_configs):
        cases = []
        for r in range(rounds):
            origin = 100 + r * step
            for h in range(1, horizon + 1):
                cases.append(_case(origin, h, half=half, y=y))
        out[f"fam:s{ci}"] = cases
    return out


def test_cross_bank_round_lag_rule():
    fam = _family(rounds=6, step=6, horizon=18)
    idx = P.PooledBankIndex(fam)
    # Target: config s0, round 2 (origin 112), h=18.
    own, cross = idx.banks("fam:s0", 112)
    # Own bank: exact PIT rule — origins 100 (h<=11 observable... origin'
    # + h' - 1 < 112 -> h' <= 11) only; round 1 (origin 106) needs h' <= 5.
    assert {c.origin for c in own} == {100, 106}
    # Cross bank: (2 - k) * 6 >= h'. Round 0 admits h' <= 12; round 1
    # admits h' <= 6; round 2 admits nothing.
    cross_h = sorted({(c.origin, c.h) for c in cross})
    assert all(o == 100 and h <= 12 or o == 106 and h <= 6 for o, h in cross_h)
    assert len(cross) == 2 * (12 + 6)  # 2 other configs
    # Round 5 target: all earlier rounds admissible at h'=18 from k<=2.
    _, cross5 = idx.banks("fam:s0", 130)
    assert any(c.h == 18 for c in cross5)


def test_cross_bank_excludes_own_config_and_future():
    fam = _family(rounds=4, step=1, horizon=6)
    idx = P.PooledBankIndex(fam)
    own, cross = idx.banks("fam:s1", 101)  # round 1
    assert all(c.origin + c.h - 1 < 101 for c in own)
    # Cross cases must come from other configs and satisfy the lag rule.
    assert cross  # round-0 cases with h'<=1... step=1: (1-0)*1 >= h' -> h'=1
    assert all(c.h == 1 for c in cross)


def test_family_step_rejects_nonuniform_grid():
    fam = _family(n_configs=2, rounds=4, step=6, horizon=6)
    fam["fam:s1"] = [c for c in fam["fam:s1"] if c.origin != 106]
    with pytest.raises(ValueError, match="uniform origin grid"):
        P.PooledBankIndex(fam)


def test_norm_scores_are_scale_free():
    # Source case: half-width 2, realized 1 beyond q90 -> s_hi=1, norm 0.5.
    src = [_case(0, 1, half=2.0, y=3.0)]
    s_lo, s_hi = P._norm_scores(src, 1)
    assert s_lo[0] == 0.0
    assert s_hi[0] == pytest.approx(0.5)
    # A source with half the scale and half the exceedance -> same norm.
    src2 = [_case(0, 1, half=1.0, y=1.5)]
    _, s_hi2 = P._norm_scores(src2, 1)
    assert s_hi2[0] == pytest.approx(0.5)


def test_arm_pooled_cqr_rescales_to_target_units():
    # 20 cross cases, all exceeding by 0.5 of their half-width (half=1).
    cross = [_case(i, 1, half=1.0, y=1.5) for i in range(20)]
    case = _case(100, 1, half=4.0)  # target half-width 4 -> offset ~2
    r_lo, r_hi = P.arm_pooled_cqr(case, [], cross)
    assert r_hi == pytest.approx(1.5, abs=0.05)  # (4 + 2) / 4
    assert r_lo == pytest.approx(1.0)  # no downside exceedances


def test_arm_pooled_raw_matches_plain_cqr_on_combined_bank():
    own = [_case(i, 1, y=2.0) for i in range(10)]
    cross = [_case(50 + i, 1, y=-2.0) for i in range(10)]
    case = _case(100, 1)
    assert P.arm_pooled_raw(case, own, cross) == M.arm_cqr(case, own + cross)


def test_arm_pooled_min_bank_fallback():
    case = _case(100, 1)
    assert P.arm_pooled_cqr(case, [], [_case(0, 1, y=3.0)] * 3) == (1.0, 1.0)
    assert P.arm_pooled_shrink(case, [], [_case(0, 1, y=3.0)] * 3) == (1.0, 1.0)


def test_arm_pooled_shrink_limits():
    own = [_case(i, 1, y=3.0) for i in range(20)]  # big own exceedances
    cross = [_case(50 + i, 1, y=0.0) for i in range(20)]  # none
    case = _case(100, 1)
    # No cross bank -> shrink reduces to own-bank answer (approx: the
    # weighted-quantile convention differs slightly from finite_quantile).
    r_shrink = P.arm_pooled_shrink(case, own, [])
    r_own = M.arm_cqr(case, own)
    assert r_shrink[1] == pytest.approx(r_own[1], abs=0.3)
    # No own bank -> shrink equals pooled-normalized on cross, rescaled.
    r_pool = P.arm_pooled_shrink(case, [], cross)
    r_c = P.arm_pooled_cqr(case, [], cross)
    assert r_pool == pytest.approx(r_c)
    # Mixed: the mixture's offset sits weakly between the two pure arms
    # (weighted quantiles are step functions, so equality at a step is
    # legitimate) ...
    r_mix = P.arm_pooled_shrink(case, own, cross)
    assert r_c[1] <= r_mix[1] <= r_shrink[1]
    # ... and shrinkage visibly bites as n0 grows: with n0=1000 the own
    # bank's total weight is ~0.02, so the all-zero cross bank drives
    # the offset to ~0 despite the own bank's large exceedances.
    r_mix_far = P.arm_pooled_shrink(case, own, cross, n0=1000.0)
    assert r_mix_far[1] == pytest.approx(1.0)
    assert r_mix_far[1] < r_mix[1]


def test_pooled_study_pit_prefix_equals_full():
    """Dropping later-round cases must not change earlier outputs.

    The pooled PIT gate is structural: a target's banks contain only
    own-series cases observable before its origin and cross-series
    cases that satisfy the round-lag rule.
    """
    rng = np.random.default_rng(3)
    fam_full = {}
    for ci in range(3):
        cases = []
        for r in range(8):
            origin = 100 + r * 6
            for h in (1, 6, 12, 18):
                cases.append(_case(origin, h, half=1.0 + 0.1 * ci, y=float(rng.normal(0, 2))))
        fam_full[f"fam:s{ci}"] = cases
    arms = {
        "native": P.local_arm(M.arm_native),
        "cqr": P.local_arm(M.arm_cqr),
        "pooled_cqr": P.arm_pooled_cqr,
        "pooled_shrink": P.arm_pooled_shrink,
    }
    res_full = P.run_study_pooled({"fam": fam_full}, arms)
    fam_pre = {
        cid: [c for c in cases if c.origin < 124]  # rounds 0..3 only
        for cid, cases in fam_full.items()
    }
    res_pre = P.run_study_pooled({"fam": fam_pre}, arms)

    def key(r):
        return (r.config_id, r.origin, r.h, r.arm)

    d_full = {key(r): r for r in res_full}
    for r in res_pre:
        f = d_full[key(r)]
        assert f.crps == pytest.approx(r.crps)
        assert f.coverage80 == r.coverage80


def test_pooled_study_future_y_perturbation_is_invisible():
    """Changing a later case's realized value cannot move earlier scales."""
    fam = _family(n_configs=2, rounds=5, step=6, horizon=18, y=1.0)
    arms = {"pooled_cqr": P.arm_pooled_cqr}
    res_a = P.run_study_pooled({"fam": fam}, arms)
    fam2 = {
        cid: [_case(c.origin, c.h, y=99.0) if c.origin >= 118 else c for c in cases]
        for cid, cases in fam.items()
    }
    res_b = P.run_study_pooled({"fam": fam2}, arms)
    early_a = [r for r in res_a if r.origin < 112]
    early_b = [r for r in res_b if r.origin < 112]
    assert len(early_a) == len(early_b) > 0
    for a, b in zip(early_a, early_b, strict=True):
        assert (a.config_id, a.origin, a.h) == (b.config_id, b.origin, b.h)
        assert a.crps == pytest.approx(b.crps)


def test_signed_scores_not_zero_inflated():
    # 12 upside + 12 downside exceedances: 12% per-side rate (per-side
    # CQR offset is exactly 0) but 24% total (signed CQR bites).
    bank = (
        [_case(i, 1, half=1.0, y=0.0) for i in range(76)]
        + [_case(100 + i, 1, half=1.0, y=5.0) for i in range(12)]
        + [_case(200 + i, 1, half=1.0, y=-5.0) for i in range(12)]
    )
    case = _case(500, 1)
    s_per = P._signed_scores(bank, 1, "width")
    assert P.finite_quantile(s_per, 0.8) > 0
    r_lo, r_hi = P.arm_pooled_cqr_signed(case, [], bank)
    assert r_lo > 1.0 and r_hi > 1.0
    # The per-side variant is provably inert on this bank.
    assert P.arm_pooled_cqr(case, [], bank) == (1.0, 1.0)


def test_signed_arm_narrows_on_overcovered_bank():
    # All realizeds inside the interval -> signed scores negative ->
    # the pooled quantile is negative -> narrowing (textbook CQR).
    bank = [_case(i, 1, half=1.0, y=0.0) for i in range(20)]
    case = _case(100, 1)
    r_lo, r_hi = P.arm_pooled_cqr_signed(case, [], bank)
    assert r_lo < 1.0 and r_hi < 1.0


def test_shrink_signed_limits():
    own = [_case(i, 1, half=1.0, y=5.0) for i in range(20)]
    cross = [_case(50 + i, 1, half=1.0, y=0.0) for i in range(20)]
    case = _case(100, 1)
    # No cross -> own signed scores only.
    r_mix = P.arm_pooled_shrink_signed(case, own, [])
    assert r_mix[1] > 1.5
    # No own -> equals pooled-signed on cross (quantile-convention slack).
    r_cross = P.arm_pooled_shrink_signed(case, [], cross)
    r_dir = P.arm_pooled_cqr_signed(case, [], cross)
    assert r_cross[0] == pytest.approx(r_dir[0], abs=0.2)
    assert r_cross[1] == pytest.approx(r_dir[1], abs=0.2)


def test_fill_resid_scales_pit_safe():
    # scale at a case uses only cases observable before its origin.
    rng = np.random.default_rng(11)
    cases = []
    for origin in (100, 112, 124):
        for h in (1, 6):
            cases.append(_case(origin, h, y=float(rng.normal(0, 3))))
    P.fill_resid_scales({"c": cases})
    early = [c for c in cases if c.origin == 100]
    late = [c for c in cases if c.origin == 124]
    for c in early:
        assert c.feats["resid_scale_h"] == pytest.approx(1.0)  # no prior -> own half-width
    # Later cases get a real residual scale from prior cases.
    assert late[0].feats["resid_scale_h"] > 0
    s0 = {id(c): c.feats["resid_scale_h"] for c in early}
    # Perturbing a LATER realized must not move EARLIER scales.
    for c in late:
        c.y = 999.0
    P.fill_resid_scales({"c": cases})
    for c in early:
        assert c.feats["resid_scale_h"] == pytest.approx(s0[id(c)])


def test_gated_arm_returns_b_when_bank_fed():
    own = [_case(i, 1, half=1.0, y=2.0) for i in range(30)]  # 30 >= gate 24
    cross = [_case(100 + i, 1, half=1.0, y=-9.0) for i in range(30)]
    case = _case(500, 1)
    assert P.arm_pooled_shrink_gated(case, own, cross) == M.arm_cqr(case, own)
    # Below the gate it must use the pool (different answer than B here).
    own_small = own[:5]
    assert P.arm_pooled_shrink_gated(case, own_small, cross) != M.arm_cqr(case, own_small)


def test_similarity_gate_filters_far_scale_series():
    case = _case(500, 1)
    case.feats["resid_scale_h"] = 1.0
    near = _case(0, 1, y=5.0)
    near.feats["resid_scale_h"] = 2.0  # within kappa=3
    far = _case(1, 1, y=5.0)
    far.feats["resid_scale_h"] = 50.0  # outside kappa=3
    kept = P._similar_cross(case, [near, far], 3.0)
    assert kept == [near]
    # Missing scale information -> kept (never silently drop data).
    bare = _case(2, 1, y=5.0)
    assert P._similar_cross(case, [bare], 3.0) == [bare]
