"""Tests for risk-level calibration metrics (exp66g)."""

import numpy as np
import pytest

from td_one.evaluation import risk_calibration as rc


def _train_data():
    # 6 groups with clearly increasing training attrition.
    train_realized = {
        "g0": 0.02,
        "g1": 0.04,
        "g2": 0.06,
        "g3": 0.10,
        "g4": 0.14,
        "g5": 0.20,
    }
    return train_realized


def test_level_anchors_are_training_tertile_means():
    train = _train_data()
    anchors = rc.level_anchors(train)
    # tertiles of [0.02,0.04,0.06,0.10,0.14,0.20]: LOW gets the two
    # smallest, HIGH the two largest (quantile cutoffs fall between).
    assert anchors["LOW"] == pytest.approx(np.mean([0.02, 0.04]))
    assert anchors["HIGH"] == pytest.approx(np.mean([0.14, 0.20]))
    assert anchors["LOW"] < anchors["MEDIUM"] < anchors["HIGH"]


def test_level_anchors_fallback_for_empty_level():
    # All groups identical -> every bin collapses to one level; the
    # other levels fall back to the overall training mean.
    train = {"g0": 0.10, "g1": 0.10, "g2": 0.10}
    anchors = rc.level_anchors(train)
    for lvl in ("LOW", "MEDIUM", "HIGH"):
        assert anchors[lvl] == pytest.approx(0.10)


def test_level_anchors_use_passed_bins():
    train = _train_data()
    custom = dict.fromkeys(train, "LOW")
    anchors = rc.level_anchors(train, train_bins=custom)
    assert anchors["LOW"] == pytest.approx(np.mean(list(train.values())))
    assert anchors["HIGH"] == pytest.approx(np.mean(list(train.values())))


def test_bin_errors_absolute():
    anchors = {"LOW": 0.03, "MEDIUM": 0.08, "HIGH": 0.17}
    levels = {"g0": "LOW", "g1": "HIGH"}
    realized = {"g0": 0.05, "g1": 0.10, "gX": 0.99}
    errs = rc.bin_errors(levels, anchors, realized)
    assert errs["g0"] == pytest.approx(0.02)
    assert errs["g1"] == pytest.approx(0.07)
    assert "gX" not in errs  # no level assigned -> excluded


def test_mean_bin_error():
    errs = {"a": 0.1, "b": 0.3}
    assert rc.mean_bin_error(errs) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        rc.mean_bin_error({})


def test_monotonic_check():
    realized = {"g0": 0.02, "g1": 0.05, "g2": 0.12}
    assert rc.monotonic_check({"g0": "LOW", "g1": "MEDIUM", "g2": "HIGH"}, realized)
    assert not rc.monotonic_check({"g0": "HIGH", "g1": "MEDIUM", "g2": "LOW"}, realized)
    # levels missing on one side still check the present order
    assert rc.monotonic_check({"g0": "LOW", "g2": "HIGH"}, realized)


def test_paired_bootstrap_identical_arms_zero_delta():
    errs = {f"u{i}": 0.05 for i in range(10)}
    delta, lo, hi = rc.paired_bootstrap_mae_delta(errs, dict(errs), n=200, seed=1)
    assert delta == 0.0
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_clear_winner_excludes_zero():
    a = {f"u{i}": 0.10 for i in range(20)}
    b = {f"u{i}": 0.02 for i in range(20)}
    delta, lo, hi = rc.paired_bootstrap_mae_delta(a, b, n=200, seed=1)
    assert delta == pytest.approx(0.08)
    assert lo > 0.0  # B better, CI excludes zero


def test_paired_bootstrap_alignment_and_min_units():
    a = {"u1": 0.1, "u2": 0.1, "u3": 0.1}
    b = {"u1": 0.0, "u2": 0.0, "u3": 0.0}
    with pytest.raises(ValueError):
        rc.paired_bootstrap_mae_delta(a, b)
