"""Tests for td_one.uq.fusion: CQR+meta fusion arms and gated ACI (exp76)."""

import numpy as np
import pytest

from td_one.uq import fusion as _fusion
from td_one.uq import methods as _methods

F = _fusion
M = _methods

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def _case(
    origin: int,
    h: int,
    m: float = 0.0,
    half: float = 1.0,
    y: float = 0.0,
    feats: dict | None = None,
):
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
    c = M.ForecastCase(
        origin=origin,
        h=h,
        levels=LEVELS,
        q=q,
        y=y,
        context=np.linspace(-1, 1, 30),
    )
    for name in M.META_FEATURES_NUMERIC:
        c.feats[name] = 0.0
    if feats:
        c.feats.update(feats)
    return c


def _volatile_bank(n: int = 90, h: int = 1):
    """Bank where |error| grows smoothly with the vol_12 feature.

    y = 1.3 * vol (one-sided misses), vol cycling over 8 graded values,
    so exceedance scores are graded and both the meta model and the
    conditional score model can learn the vol -> error mapping.
    """
    vols = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    bank = []
    for i in range(n):
        vol = vols[i % len(vols)]
        bank.append(_case(i, h, y=float(1.3 * vol), feats={"vol_12": vol}))
    return bank


def test_modulated_level_monotone_in_percentile():
    assert F.modulated_level(0.5, 0.8, 0.2) == pytest.approx(0.8)
    assert F.modulated_level(1.0, 0.8, 0.2) == pytest.approx(0.95)  # clipped
    assert F.modulated_level(0.0, 0.8, 0.2) == pytest.approx(0.6)
    # clips to [0.5, 0.95]
    assert F.modulated_level(1.0, 0.9, 0.4) == pytest.approx(0.95)
    assert F.modulated_level(0.0, 0.6, 0.4) == pytest.approx(0.5)


def test_cqr_meta_level_widens_for_high_uncertainty_case():
    bank = _volatile_bank()
    hi = _case(500, 1, y=0.0, feats={"vol_12": 3.0})
    lo = _case(500, 1, y=0.0, feats={"vol_12": 0.2})
    arm = F.MetaLevelCqr(M.META_FEATURES_NUMERIC, k_grid=(0.2,))  # fixed k
    r_hi_case = arm(hi, bank)
    r_lo_case = arm(lo, bank)
    # High predicted uncertainty -> wider than the low-uncertainty case.
    assert (r_hi_case[0] + r_hi_case[1]) > (r_lo_case[0] + r_lo_case[1])


def test_cqr_meta_level_k_zero_equals_plain_cqr():
    bank = _volatile_bank()
    case = _case(500, 1, y=0.0, feats={"vol_12": 3.0})
    arm = F.MetaLevelCqr(M.META_FEATURES_NUMERIC, k_grid=(0.0,))
    assert arm(case, bank) == pytest.approx(M.arm_cqr(case, bank))


def test_cqr_meta_level_fallback_small_bank():
    bank = _volatile_bank(10)  # below MIN_META_BANK
    case = _case(500, 1, y=0.0, feats={"vol_12": 3.0})
    arm = F.MetaLevelCqr(M.META_FEATURES_NUMERIC, k_grid=(0.2,))
    assert arm(case, bank) == pytest.approx(M.arm_cqr(case, bank))


def test_conditional_cqr_responds_to_feature():
    bank = _volatile_bank()
    hi = _case(500, 1, y=0.0, feats={"vol_12": 3.0})
    lo = _case(500, 1, y=0.0, feats={"vol_12": 0.2})
    arm = F.ConditionalCqr(M.META_FEATURES_NUMERIC)
    # High-vol case must get a strictly larger upside scale than low-vol.
    r_hi = arm(hi, bank)
    r_lo = arm(lo, bank)
    assert r_hi[1] > r_lo[1]
    assert r_hi[1] > 1.5  # and it must actually widen (bank has 4x misses)


def test_conditional_cqr_fallback_no_exceedances():
    # Calm bank: everything inside the interval -> plain CQR fallback.
    bank = [_case(i, 1, y=0.0, feats={"vol_12": 1.0}) for i in range(60)]
    case = _case(500, 1, y=0.0, feats={"vol_12": 1.0})
    arm = F.ConditionalCqr(M.META_FEATURES_NUMERIC)
    assert arm(case, bank) == pytest.approx(M.arm_cqr(case, bank))


def test_aci_gated_calm_bank_is_native():
    bank = [_case(i, 1, y=0.0, feats={"vol_12": 1.0}) for i in range(30)]
    case = _case(500, 1, y=0.0, feats={"vol_12": 1.0})
    assert F.arm_aci_gated(case, bank) == (1.0, 1.0)


def test_aci_gated_miss_streak_uses_aci():
    bank = [_case(i, 1, y=5.0, feats={"vol_12": 1.0}) for i in range(30)]
    case = _case(500, 1, y=0.0, feats={"vol_12": 1.0})
    gated = F.arm_aci_gated(case, bank)
    plain_aci = M.arm_aci(case, bank)
    assert gated == pytest.approx(plain_aci)
    assert gated != (1.0, 1.0)  # and it actually reacted


def test_aci_gated_volatility_spike_uses_aci():
    # No misses, but the case's vol_12 is 5x the bank median -> gate fires.
    bank = [_case(i, 1, y=0.0, feats={"vol_12": 1.0}) for i in range(30)]
    case = _case(500, 1, y=0.0, feats={"vol_12": 5.0})
    gated = F.arm_aci_gated(case, bank, vol_ratio=1.5)
    assert gated == pytest.approx(M.arm_aci(case, bank))


def test_month_before_pit_keying():
    assert F.month_before("2021-01") == "2020-12"
    assert F.month_before("2024-07") == "2024-06"
    assert F.month_before("2026-09") == "2026-08"
