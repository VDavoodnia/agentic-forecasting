"""Tests for td_one.uq.adaptive (exp85).

Changepoint bank surgery + bias-corrected conformal arms, and their
PIT contracts.
"""

import numpy as np
import pytest

from td_one.uq import adaptive as uq_adaptive
from td_one.uq import methods as uq_methods

A = uq_adaptive
M = uq_methods

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def _case(
    origin: int,
    h: int,
    m: float = 0.0,
    half: float = 1.0,
    y: float = 0.0,
    context: np.ndarray | None = None,
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
    return M.ForecastCase(
        origin=origin,
        h=h,
        levels=LEVELS,
        q=q,
        y=y,
        context=np.linspace(-1, 1, 40) if context is None else context,
    )


def _shifted_context(n_pre: int = 30, n_post: int = 12, jump: float = 4.0, seed: int = 5):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [0.1 * rng.standard_normal(n_pre), jump + 0.1 * rng.standard_normal(n_post)]
    )


def _flat_context(n: int = 42, seed: int = 5):
    rng = np.random.default_rng(seed)
    return 0.1 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detect_break_on_shifted_context():
    ctx = _shifted_context()
    onset = A.detect_break_onset(ctx)
    assert onset is not None
    # The true onset is index 30; CUSUM estimates are coarse but local.
    assert 24 <= onset <= 36


def test_detect_no_break_on_flat_context():
    assert A.detect_break_onset(_flat_context()) is None
    assert A.detect_break_onset(np.zeros(42)) is None


def test_detect_short_context_returns_none():
    assert A.detect_break_onset(np.array([1.0, 2.0, 3.0])) is None


def test_detect_bad_gate_raises():
    with pytest.raises(ValueError, match="gate"):
        A.detect_break_onset(np.zeros(42), gate="bogus")


def _s1_series(seed: int = 2, shift: bool = True) -> np.ndarray:
    """exp74's S1@4sigma series (or its control).

    Built exactly as exp74_generate_data.build_shifted_series does.
    """
    from td_one.data_gen.shift_scenarios import (  # noqa: PLC0415
        ShiftSpec,
        apply_shift,
        generate_base_series,
    )

    base = generate_base_series(48, seed=4000 + seed)
    if not shift:
        return np.asarray(base["attrition_rate"], dtype=float)
    rng = np.random.default_rng(1000 + seed)
    out = apply_shift(base, ShiftSpec("S1", 30, 4.0), rng)
    return np.asarray(out["attrition_rate"], dtype=float)


def test_cusum_gate_catches_ramp_the_scan_misses():
    """exp85 round-1 finding.

    On exp74's noisy ramped S1 series the scan gate never fires; the
    CUSUM gate fires with onset == 30.
    """
    ctx = _s1_series()[:36]
    assert A.detect_break_onset(ctx) is None
    assert A.detect_break_onset(ctx, gate="cusum") == 30


def test_cusum_gate_quiet_on_control():
    ctx = _s1_series(shift=False)[:42]
    assert A.detect_break_onset(ctx, gate="cusum") is None
    assert A.detect_break_onset(ctx, gate="cusum", cusum_threshold=3.0) is None


def test_reset_pool_uses_cross_horizon_evidence():
    """Per-h post-break banks are too small to reset.

    The pooled bank (all h) is not — pooled reset must widen more.
    """
    ctx = _shifted_context()
    case = _case(origin=40, h=1, context=ctx)
    bank = []
    for h in (1, 2, 3, 4, 5, 6):
        bank.append(_case(origin=10, h=h, m=0.0, half=1.0, y=0.1, context=ctx))
        for o in (32, 34):
            bank.append(_case(origin=o, h=h, m=0.0, half=1.0, y=3.0, context=ctx))
    r_pool = A.arm_cqr_reset_pool(case, bank, min_post=5)
    r_perh = A.arm_cqr_reset(case, bank, min_post=5)
    assert r_pool[1] > r_perh[1]


def test_error_onset_index_finds_sign_flip():
    rng = np.random.default_rng(3)
    errs = np.concatenate([0.05 * rng.standard_normal(14), 2.0 + 0.05 * rng.standard_normal(8)])
    idx = A.error_onset_index(errs)
    assert idx is not None and 10 <= idx <= 18
    calm = 0.05 * rng.standard_normal(22)
    assert A.error_onset_index(calm) is None


def test_reset_with_errors_detector_engages():
    """Bank errors flip sign at the break.

    The errors detector must find the post-break scores even when
    the context is uninformative. Post-break cases are a minority
    (3/22), so the plain full-bank quantile still lands on a zero
    score — the dilution the reset exists to fix.
    """
    flat = _flat_context()
    case = _case(origin=40, h=1, context=flat)
    bank = []
    for i in range(19):  # pre-break: centered, cover (s_hi = 0)
        y = 0.05 if i % 2 == 0 else 0.15
        bank.append(_case(origin=10 + i, h=1, m=0.0, half=1.0, y=y, context=flat))
    for i in range(3):  # post-break: persistent +3 misses
        bank.append(_case(origin=29 + i, h=1, m=0.0, half=1.0, y=3.0, context=flat))
    r_err = A.arm_cqr_reset(case, bank, min_post=3, detector="errors")
    r_scan = A.arm_cqr_reset(case, bank, min_post=3, detector="scan")
    assert r_err[1] > r_scan[1]


# ---------------------------------------------------------------------------
# Reset / decay arms
# ---------------------------------------------------------------------------


def _mixed_bank(ctx: np.ndarray, n_pre: int = 17, n_post: int = 3):
    """Bank at h=1: pre-break cases cover (small scores), post-break miss.

    Post-break cases are a small minority (3/20 < 20%), so the plain
    0.8-quantile over the full bank still lands on a zero score —
    exactly the dilution the reset exists to fix.
    """
    bank = []
    for i in range(n_pre):
        bank.append(_case(origin=10 + i, h=1, m=0.0, half=1.0, y=0.1, context=ctx))
    for i in range(n_post):
        # Realized far above q90=1.0 -> s_hi = 2.0 each.
        bank.append(_case(origin=31 + i, h=1, m=0.0, half=1.0, y=3.0, context=ctx))
    return bank


def test_reset_uses_post_break_scores():
    ctx = _shifted_context()
    case = _case(origin=40, h=1, context=ctx)
    bank = _mixed_bank(ctx)
    r_reset = A.arm_cqr_reset(case, bank, min_post=3)
    r_plain = M.arm_cqr(case, bank)
    # Post-break-only scores are all large on the hi side -> wider hi scale.
    assert r_reset[1] > r_plain[1]


def test_reset_falls_back_when_post_too_small():
    ctx = _shifted_context()
    case = _case(origin=40, h=1, context=ctx)
    bank = _mixed_bank(ctx)
    r_reset = A.arm_cqr_reset(case, bank, min_post=99)
    r_plain = M.arm_cqr(case, bank)
    assert r_reset == pytest.approx(r_plain)


def test_reset_no_break_equals_plain_cqr():
    ctx = _flat_context()
    case = _case(origin=40, h=1, context=ctx)
    bank = _mixed_bank(ctx)
    assert A.arm_cqr_reset(case, bank) == pytest.approx(M.arm_cqr(case, bank))


def test_decay_weights_recent_more():
    case = _case(origin=40, h=1)
    bank = [_case(origin=o, h=1, m=0.0, half=1.0, y=3.0) for o in range(10, 30)]
    w = A._decay_weights(case, A._hbank(bank, 1), half_life=6.0, onset=None, pre_factor=1.0)
    assert w[-1] > w[0]  # most recent case weighs most
    assert w[-1] == pytest.approx(0.5 ** ((40 - 29) / 6.0))


def test_min_bank_gate():
    case = _case(origin=40, h=1)
    bank = [_case(origin=o, h=1) for o in range(3)]
    for arm in (
        A.arm_cqr_reset,
        A.arm_cqr_decay,
        A.arm_cqr_cp,
        A.arm_bc_cqr,
        A.arm_bc_aci,
        A.arm_cp_bc,
    ):
        assert arm(case, bank) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


def test_bias_track_is_causal():
    hb = [_case(origin=o, h=1, m=0.0, y=2.0) for o in range(5)]
    track, b_now = A._bias_track(hb, lam=0.5)
    assert track[0] == 0.0  # first case sees zero bias
    assert track[1] == pytest.approx(1.0)  # 0.5*0 + 0.5*2
    assert b_now == pytest.approx(0.5 * track[-1] + 0.5 * 2.0)


def test_bc_shifts_interval_toward_persistent_bias():
    """Forecasts under-predict by +2 every time: BC must shift up."""
    case = _case(origin=40, h=1, m=0.0, half=1.0)
    bank = [_case(origin=o, h=1, m=0.0, half=1.0, y=2.0) for o in range(10, 30)]
    r_bc = A.arm_bc_cqr(case, bank)
    r_cqr = M.arm_cqr(case, bank)
    q_bc = M.scale_grid(case.levels, case.q, *r_bc)
    q_cq = M.scale_grid(case.levels, case.q, *r_cqr)
    i10 = int(np.argmin(np.abs(LEVELS - 0.1)))
    # The whole BC interval sits above the plain-CQR interval's low edge.
    assert q_bc[i10] > q_cq[i10]


def test_bc_no_bias_no_shift():
    """Unbiased bank (alternating errors): BC stays close to plain CQR."""
    case = _case(origin=40, h=1, m=0.0, half=1.0)
    bank = [
        _case(origin=o, h=1, m=0.0, half=1.0, y=0.2 if o % 2 == 0 else -0.2) for o in range(10, 30)
    ]
    r_bc = A.arm_bc_cqr(case, bank)
    assert r_bc == pytest.approx(M.arm_cqr(case, bank), abs=0.05)


def test_cp_bc_no_break_equals_bc_cqr():
    ctx = _flat_context()
    case = _case(origin=40, h=1, context=ctx)
    bank = [_case(origin=o, h=1, m=0.0, half=1.0, y=0.3, context=ctx) for o in range(10, 30)]
    assert A.arm_cp_bc(case, bank) == pytest.approx(A.arm_bc_cqr(case, bank))


# ---------------------------------------------------------------------------
# PIT: prefix-vs-full (the exp74 regression pattern, new arms)
# ---------------------------------------------------------------------------


def test_adaptive_arms_pit_prefix_equals_full():
    from td_one.uq.evaluate import run_study  # noqa: PLC0415

    rng = np.random.default_rng(11)
    cases_full: list = []
    for origin in range(0, 30, 2):
        ctx = np.concatenate([rng.normal(0, 1, origin + 20), rng.normal(3, 1, 5)])
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
    arms = {
        "cqr_reset": A.arm_cqr_reset,
        "cqr_reset_pool": A.arm_cqr_reset_pool,
        "cqr_reset_cusum": lambda c, b: A.arm_cqr_reset(c, b, detector="cusum"),
        "cqr_decay": A.arm_cqr_decay,
        "cqr_cp": A.arm_cqr_cp,
        "bc_cqr": A.arm_bc_cqr,
        "bc_aci": A.arm_bc_aci,
        "cp_bc": A.arm_cp_bc,
    }
    res_full = run_study({"c": cases_full}, arms)
    res_pre = run_study({"c": [c for c in cases_full if c.origin < 14]}, arms)

    def key(r):
        return (r.origin, r.h, r.arm)

    d_full = {key(r): r for r in res_full}
    for r in res_pre:
        assert d_full[key(r)].crps == pytest.approx(r.crps)
        assert d_full[key(r)].coverage80 == r.coverage80
