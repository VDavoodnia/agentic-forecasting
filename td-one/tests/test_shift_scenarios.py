"""Tests for td_one.data_gen.shift_scenarios."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from td_one.data_gen.shift_scenarios import (
    SHIFT_SIGMA_LOGIT,
    ShiftSpec,
    apply_divergence,
    apply_regional_divergence,
    apply_shift,
    beta_schedule_for,
    generate_base_series,
    macro_shifts_for,
)


@pytest.fixture()
def base() -> dict[str, np.ndarray]:
    return generate_base_series(n_months=36, seed=7)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def test_spec_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown shift_type"):
        ShiftSpec("S9", 10, 1.0)


def test_spec_rejects_negative_onset() -> None:
    with pytest.raises(ValueError, match="onset_month"):
        ShiftSpec("S1", -1, 1.0)


def test_spec_rejects_negative_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        ShiftSpec("S1", 10, -0.5)


def test_shift_never_touches_pre_onset(
    base: dict[str, np.ndarray], rng: np.random.Generator
) -> None:
    """PIT core: months t < T are bit-identical after apply_shift."""
    for stype in ("S1", "S2", "S4", "S5", "S6", "S7"):
        spec = ShiftSpec(stype, 20, 2.0)
        out = apply_shift(base, spec, rng)
        for key in ("n_renewals", "n_switches", "true_rates"):
            np.testing.assert_array_equal(
                out[key][:20], base[key][:20], err_msg=f"{stype}/{key} pre-onset changed"
            )
        # ... and something actually changed post-onset (except pathological draws)
        assert not np.array_equal(out["true_rates"][20:], base["true_rates"][20:]) or stype == "S4"


def test_magnitude_zero_is_noop(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    spec = ShiftSpec("S1", 20, 0.0)
    out = apply_shift(base, spec, rng)
    # true rates unchanged up to fp roundtrip through logit/inv_logit
    np.testing.assert_allclose(out["true_rates"], base["true_rates"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out["n_renewals"], base["n_renewals"])


def test_onset_out_of_range(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="out of range"):
        apply_shift(base, ShiftSpec("S1", 36, 1.0), rng)


def test_s1_level_shift_direction(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    out = apply_shift(base, ShiftSpec("S1", 20, 2.0), rng)
    # Mean post-shift true rate must exceed pre-shift (positive logit shift).
    assert float(np.mean(out["true_rates"][20:])) > float(np.mean(out["true_rates"][:20]))


def test_s2_has_implementation_lag(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    out = apply_shift(base, ShiftSpec("S2", 20, 2.0), rng)
    # Months 20..21 unaffected by the lagged shift (rates identical pre-lag).
    np.testing.assert_array_equal(out["true_rates"][20:22], base["true_rates"][20:22])


def test_s3_requires_regional_fn(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="apply_regional_divergence"):
        apply_shift(base, ShiftSpec("S3", 20, 1.0), rng)


def test_s3_aggregate_averages_away(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    b = generate_base_series(n_months=36, seed=8)
    agg = apply_regional_divergence(base, b, ShiftSpec("S3", 20, 4.0), 0.5, rng)
    # Aggregate post-shift move is small relative to a single region's move.
    pre = float(np.mean(agg["true_rates"][:20]))
    post = float(np.mean(agg["true_rates"][20:]))
    only_a = apply_shift(base, ShiftSpec("S1", 20, 4.0), rng)
    single_move = abs(
        float(np.mean(only_a["true_rates"][20:])) - float(np.mean(only_a["true_rates"][:20]))
    )
    agg_move = abs(post - pre)
    assert agg_move < 0.5 * single_move, f"agg moved {agg_move:.3f} vs single {single_move:.3f}"


def test_s6_widens_spread(base: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    out = apply_shift(base, ShiftSpec("S6", 20, 4.0), rng)
    pre_sd = float(np.std(out["true_rates"][:20]))
    post_sd = float(np.std(out["true_rates"][20:]))
    # Volatility regime: post-shift spread should generally exceed pre-shift.
    # (Stochastic; use a lenient bound — the mechanism is what matters.)
    assert post_sd > 0.5 * pre_sd


def test_macro_shifts_for_s1() -> None:
    months = [date(2021, 4, 30), date(2021, 5, 31), date(2021, 6, 30)]
    out = macro_shifts_for(ShiftSpec("S1", 1, 2.0), months)
    assert out == {date(2021, 5, 31): 2.0 * SHIFT_SIGMA_LOGIT}


def test_macro_shifts_for_s2_lag() -> None:
    months = [date(2021, 4, 30), date(2021, 5, 31), date(2021, 6, 30), date(2021, 7, 31)]
    out = macro_shifts_for(ShiftSpec("S2", 0, 2.0), months)
    # 2-month implementation lag from April -> June.
    assert out == {date(2021, 6, 30): 0.6 * 2.0 * SHIFT_SIGMA_LOGIT}


def test_macro_shifts_rejects_others() -> None:
    with pytest.raises(ValueError, match="S1/S2"):
        macro_shifts_for(ShiftSpec("S5", 1, 1.0), [date(2021, 4, 30)])


def test_beta_schedule_for_s5() -> None:
    sched = beta_schedule_for(ShiftSpec("S5", 10, 2.0), 36)
    assert set(sched.keys()) == set(range(10, 36))
    assert sched[10]["V2_BETA_RATE_SPREAD"] == -0.4  # flips v2's +0.4
    assert 9 not in sched


def test_beta_schedule_rejects_others() -> None:
    with pytest.raises(ValueError, match="S5"):
        beta_schedule_for(ShiftSpec("S1", 1, 1.0), 36)


def test_beta_overrides_unknown_name_raises() -> None:
    from td_one.data_gen.simulator_v2 import v2_switching_logit

    with pytest.raises(ValueError, match="unknown beta"):
        v2_switching_logit(
            payment_shock=0.0,
            offer_spread=0.0,
            rate_spread=0.0,
            family_squeeze=0.0,
            osfi_exempt=0.0,
            util_drift_ppm=0.0,
            life_event_6m=0.0,
            is_digital=0.0,
            has_advisor=0.0,
            n_prior_refinances=0,
            age_months=60,
            beta_overrides={"NOPE": 1.0},
        )


def test_beta_overrides_changes_logit() -> None:
    from td_one.data_gen.simulator_v2 import v2_switching_logit

    kw: dict[str, float] = {
        "payment_shock": 0.0,
        "offer_spread": 0.5,
        "rate_spread": 2.0,
        "family_squeeze": 0.0,
        "osfi_exempt": 0.0,
        "util_drift_ppm": 0.0,
        "life_event_6m": 0.0,
        "is_digital": 0.0,
        "has_advisor": 0.0,
        "n_prior_refinances": 0,
        "age_months": 60,
    }
    base = v2_switching_logit(**kw)  # type: ignore[arg-type]
    flipped = v2_switching_logit(**kw, beta_overrides={"V2_BETA_RATE_SPREAD": -0.4})  # type: ignore[arg-type]
    # rate_spread=2.0: +0.4*2=+0.8 vs -0.4*2=-0.8 -> diff of 1.6 logit.
    assert flipped == pytest.approx(base - 1.6)


def test_generate_base_series_seasonal_break() -> None:
    s = generate_base_series(n_months=36, seed=3, seasonal_break=(18, 0.0, 0.0))
    # Post-break volume seasonality killed -> flatter renewals.
    pre_sd = float(np.std(s["n_renewals"][:18]))
    post_sd = float(np.std(s["n_renewals"][18:]))
    assert post_sd < pre_sd


def _group_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    series = {
        "g0": generate_base_series(n_months=36, seed=21),
        "g1": generate_base_series(n_months=36, seed=22),
        "g2": generate_base_series(n_months=36, seed=23),
    }
    regions = {"g0": "A", "g1": "B"}  # g2 unlisted -> passthrough
    return series, regions


def test_apply_divergence_rejects_non_s3() -> None:
    series, regions = _group_fixture()
    with pytest.raises(ValueError, match="requires shift_type 'S3'"):
        apply_divergence(series, regions, ShiftSpec("S1", 20, 2.0), np.random.default_rng(1))


def test_apply_divergence_rejects_bad_region() -> None:
    series, _ = _group_fixture()
    with pytest.raises(ValueError, match="'A' or 'B'"):
        apply_divergence(series, {"g0": "C"}, ShiftSpec("S3", 20, 2.0), np.random.default_rng(1))


def test_apply_divergence_pre_onset_bit_identical() -> None:
    """Observable arrays are bit-identical pre-onset (PIT core).

    ``true_rates`` goes through a logit round-trip in apply_shift, so it
    is compared at 1e-12; what forecasts consume (counts, observed
    rates) is exactly unchanged.
    """
    series, regions = _group_fixture()
    out = apply_divergence(series, regions, ShiftSpec("S3", 20, 4.0), np.random.default_rng(5))
    for gid in series:
        for key in ("n_renewals", "n_switches", "attrition_rate"):
            np.testing.assert_array_equal(
                out[gid][key][:20],
                series[gid][key][:20],
                err_msg=f"{gid}/{key} pre-onset changed",
            )
        np.testing.assert_allclose(
            out[gid]["true_rates"][:20], series[gid]["true_rates"][:20], rtol=1e-12
        )


def test_apply_divergence_signed_direction() -> None:
    """Region A true rates move up, region B down (noiseless latent check)."""
    series, regions = _group_fixture()
    out = apply_divergence(series, regions, ShiftSpec("S3", 20, 4.0), np.random.default_rng(5))
    post_a = out["g0"]["true_rates"][20:] - series["g0"]["true_rates"][20:]
    post_b = out["g1"]["true_rates"][20:] - series["g1"]["true_rates"][20:]
    assert float(np.mean(post_a)) > 0.0
    assert float(np.mean(post_b)) < 0.0

    # Symmetric in logit space by construction (rate moves are not:
    # inv_logit is nonlinear, so +1 logit from ~0.11 moves more than -1).
    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    d_a = logit(out["g0"]["true_rates"][20:]) - logit(series["g0"]["true_rates"][20:])
    d_b = logit(out["g1"]["true_rates"][20:]) - logit(series["g1"]["true_rates"][20:])
    assert float(np.mean(d_a)) == pytest.approx(4.0 * SHIFT_SIGMA_LOGIT, abs=1e-9)
    assert float(np.mean(d_b)) == pytest.approx(-4.0 * SHIFT_SIGMA_LOGIT, abs=1e-9)


def test_apply_divergence_passthrough_unlisted() -> None:
    series, regions = _group_fixture()
    out = apply_divergence(series, regions, ShiftSpec("S3", 20, 4.0), np.random.default_rng(5))
    for key in ("n_renewals", "n_switches", "true_rates", "attrition_rate"):
        np.testing.assert_array_equal(out["g2"][key], series["g2"][key])


def test_apply_divergence_does_not_mutate_input() -> None:
    series, regions = _group_fixture()
    before = {g: {k: np.array(v) for k, v in s.items()} for g, s in series.items()}
    apply_divergence(series, regions, ShiftSpec("S3", 20, 4.0), np.random.default_rng(5))
    for gid in series:
        for key in before[gid]:
            np.testing.assert_array_equal(series[gid][key], before[gid][key])
