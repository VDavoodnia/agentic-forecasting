"""Tests for the Layer-2 analyst toolbox (exp73, user-directed).

The toolbox is deterministic, PIT-safe (as-of dating enforced by the
caller slicing data), and JSON-traceable. These tests pin the contracts
before the implementation exists.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from td_one.agent.layer2_tools import (
    LEVELS_9 as L9,
)
from td_one.agent.layer2_tools import (
    OSFILockInMeter,
    changepoint_scan,
    cusum_break_pvalue,
    expert_backtest,
    interval_compare,
    lockin_level,
    macro_context,
    pinball_crps,
    q50_of,
    router_audit,
    series_diagnostics,
    summarize_backtest,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def random_walk(rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.normal(0, 1, 120))


@pytest.fixture()
def seasonal(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(120)
    return 10 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 120)


# --- pinball_crps / q50_of -------------------------------------------------


def test_pinball_crps_matches_grid_to_crps(random_walk: np.ndarray) -> None:
    """pinball_crps on a (9,1) grid agrees with the canonical grid_to_crps."""
    from td_one.llm_rungs.cache import grid_to_crps

    q = np.sort(np.random.default_rng(0).normal(0, 1, 9)).reshape(9, 1)
    actual = np.array([0.3])
    ours = pinball_crps(q, actual)
    canon = grid_to_crps(q[:, 0], list(L9), 0.3)
    assert ours == pytest.approx(canon, rel=1e-9)


def test_q50_of_picks_median_level() -> None:
    q = np.arange(9).reshape(9, 1) * 1.0
    assert q50_of(q) == pytest.approx(4.0)


# --- series_diagnostics -----------------------------------------------------


def test_diagnostics_seasonal_series_is_seasonal(seasonal: np.ndarray) -> None:
    d = series_diagnostics(seasonal)
    assert d["seasonal_strength"] > 0.5
    assert d["trend_strength"] < 0.5
    assert 0.0 <= d["stability"] <= 1.0
    assert d["vol_ratio"] > 0


def test_diagnostics_trending_series_has_trend() -> None:
    y = np.arange(120, dtype=float) + np.random.default_rng(1).normal(0, 1, 120)
    d = series_diagnostics(y)
    assert d["trend_strength"] > 0.7


def test_diagnostics_random_walk_stable(random_walk: np.ndarray) -> None:
    d = series_diagnostics(random_walk)
    assert d["ac1"] > 0.8  # levels of a random walk are highly persistent
    assert d["seasonal_strength"] < 0.3


def test_diagnostics_vol_ratio_detects_regime_change(rng: np.random.Generator) -> None:
    calm = rng.normal(0, 1, 108)
    storm = rng.normal(0, 6, 12)
    d = series_diagnostics(np.concatenate([calm, storm]))
    assert d["vol_ratio"] > 3.0


def test_diagnostics_requires_minimum_length() -> None:
    with pytest.raises(ValueError, match="at least"):
        series_diagnostics(np.ones(20))


def test_cusum_break_pvalue_flags_planted_shift(rng: np.random.Generator) -> None:
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 20)])
    assert cusum_break_pvalue(y) < 0.05


def test_cusum_break_pvalue_clean_on_stable() -> None:
    y = np.random.default_rng(7).normal(0, 1, 120)  # iid noise: no break
    assert cusum_break_pvalue(y) > 0.01


# --- changepoint_scan --------------------------------------------------------


def test_changepoint_scan_finds_recent_break(rng: np.random.Generator) -> None:
    y = np.concatenate([rng.normal(0, 1, 108), rng.normal(4, 1, 12)])
    out = changepoint_scan(y)
    assert out["break_flag"] is True
    assert out["onset_from_end"] is not None and out["onset_from_end"] <= 13


def test_changepoint_scan_quiet_on_stable() -> None:
    y = np.random.default_rng(11).normal(0, 1, 120)  # iid noise: no break
    out = changepoint_scan(y)
    assert out["break_flag"] is False


def test_changepoint_scan_is_pit_pure(rng: np.random.Generator) -> None:
    """Future data must not change the scan result at a fixed as-of."""
    y = rng.normal(0, 1, 120)
    asof = y[:100].copy()
    before = changepoint_scan(asof)
    asof2 = np.concatenate([asof, rng.normal(10, 1, 20)])  # future shock
    after = changepoint_scan(asof2[:100])
    assert before == after


# --- expert_backtest / summarize_backtest ------------------------------------


def _naive_fn(train: np.ndarray, horizon: int) -> np.ndarray:
    last = float(train[-1])
    return np.tile(last, (9, horizon))


def _mean_fn(train: np.ndarray, horizon: int) -> np.ndarray:
    m = float(np.mean(train))
    return np.tile(m, (9, horizon))


def test_expert_backtest_naive_wins_on_random_walk(random_walk: np.ndarray) -> None:
    origins = list(range(60, 120))
    out = expert_backtest(
        random_walk,
        {"naive": _naive_fn, "mean": _mean_fn},
        origins,
        horizon=1,
        train_window=36,
    )
    assert out["naive"]["mean_crps"] < out["mean"]["mean_crps"]
    assert out["mean"]["win_rate_vs_naive"] < 0.3  # luck wins only
    assert out["recent_winner"] == "naive"


def test_summarize_backtest_from_precomputed_arrays() -> None:
    n = 20
    crps = {
        "naive": np.full(n, 1.0),
        "v2": np.concatenate([np.full(10, 0.8), np.full(10, 1.1)]),
    }
    out = summarize_backtest(crps)
    assert out["best"] == "v2"
    assert out["v2"]["mean_crps"] == pytest.approx(0.95)
    assert out["v2"]["win_rate_vs_naive"] == pytest.approx(0.5)


# --- router_audit ------------------------------------------------------------


def test_router_audit_perfect_overrides() -> None:
    n = 30
    out = router_audit(
        layer1_crps=np.full(n, 0.8),
        naive_crps=np.full(n, 1.0),
        override=np.ones(n, dtype=bool),
    )
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["override_rate"] == pytest.approx(1.0)
    assert out["mean_delta"] == pytest.approx(-0.2)


def test_router_audit_never_overrides() -> None:
    n = 30
    out = router_audit(
        layer1_crps=np.full(n, 1.2),
        naive_crps=np.full(n, 1.0),
        override=np.zeros(n, dtype=bool),
    )
    assert out["override_rate"] == pytest.approx(0.0)
    assert np.isnan(out["hit_rate"])


def test_router_audit_per_regime() -> None:
    n = 40
    out = router_audit(
        layer1_crps=np.full(n, 0.9),
        naive_crps=np.full(n, 1.0),
        override=np.ones(n, dtype=bool),
        regimes=np.array(["A"] * 20 + ["B"] * 20),
    )
    assert set(out["by_regime"]) == {"A", "B"}
    assert out["by_regime"]["A"]["hit_rate"] == pytest.approx(1.0)


# --- interval_compare ---------------------------------------------------------


def test_interval_compare_wide_vs_tight() -> None:
    n = 50
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 1, n)
    wide_grid = np.tile(np.array([-5, -4, -3, -2, 0, 2, 3, 4, 5.0]).reshape(9, 1), (n, 1, 1))
    tight_grid = np.tile(np.array([-0.1] * 4 + [0.0] + [0.1] * 4).reshape(9, 1), (n, 1, 1))
    grids = {"wide": wide_grid, "tight": tight_grid}
    out = interval_compare(grids, actual.reshape(n, 1))
    assert out["wide"]["coverage_80"] == pytest.approx(1.0)
    assert out["tight"]["coverage_80"] < 1.0
    assert out["wide"]["mean_width"] > out["tight"]["mean_width"]


# --- lockin ------------------------------------------------------------------


def test_lockin_level_sign() -> None:
    assert lockin_level(6.0, 3.0) == pytest.approx(3.0)  # locked in below market
    assert lockin_level(3.0, 6.0) == pytest.approx(-3.0)


def _balances_frame() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    rows = []
    for seg in ("insured", "uninsured"):
        for i, m in enumerate(months):
            rows.append(
                {
                    "month": m,
                    "segment": seg,
                    "outstanding_m": 500000.0 + i * 1000,
                    "funds_advanced_m": 10000.0,
                    "rate_pct": 2.0 + 0.05 * i,
                    "prev_outstanding_m": 500000.0 + (i - 1) * 1000 if i else np.nan,
                    "runoff_m": 9000.0,
                }
            )
    return pd.DataFrame(rows)


def test_lockin_meter_respects_pub_lag() -> None:
    bal = _balances_frame()
    meter = OSFILockInMeter(bal, pub_lag_months=2)
    asof = pd.Timestamp("2022-06-01")
    f1 = meter.features("insured", asof)
    # Mutate rates AFTER the usable window (month asof-1, asof-0) — must not change.
    bal2 = bal.copy()
    mask = (bal2["segment"] == "insured") & (bal2["month"] >= pd.Timestamp("2022-05-01"))
    bal2.loc[mask, "rate_pct"] = 99.0
    meter2 = OSFILockInMeter(bal2, pub_lag_months=2)
    f2 = meter2.features("insured", asof)
    assert f1 == f2
    assert f1["available"] is True
    assert f1["lockin"] == pytest.approx(f1["market_rate"] - f1["portfolio_proxy"])


def test_lockin_meter_unavailable_when_no_history() -> None:
    bal = _balances_frame()
    meter = OSFILockInMeter(bal)
    f = meter.features("insured", pd.Timestamp("2018-01-01"))
    assert f["available"] is False


# --- macro_context ------------------------------------------------------------


def _news_db_with(tmp: Path, with_signals: bool) -> Path:
    p = tmp / "news.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE competitor_rates (month TEXT, lender TEXT, product TEXT,"
        " rate_pct REAL, source_url TEXT DEFAULT '')"
    )
    con.executemany(
        "INSERT INTO competitor_rates VALUES (?,?,?,?,'')",
        [("2024-12", "broker", "5yr_fixed", 4.5), ("2025-01", "broker", "5yr_fixed", 4.4)],
    )
    if with_signals:
        con.execute(
            "CREATE TABLE macro_signals (month TEXT, category TEXT, direction INT,"
            " risk TEXT, certainty REAL)"
        )
        con.executemany(
            "INSERT INTO macro_signals VALUES (?,?,?,?,?)",
            [
                ("2025-01", "rates", 1, "med", 0.8),
                ("2025-02", "rates", -1, "high", 0.9),
            ],  # future row: must be excluded
        )
    con.commit()
    con.close()
    return p


def test_macro_context_pit_filtering(tmp_path: Path) -> None:
    db = _news_db_with(tmp_path, with_signals=True)
    out = macro_context("2025-02", news_db=db, fred_dir=None, balances=None)
    sigs = out["macro_signals"]
    assert sigs["available"] is True
    assert sigs["rows"][0]["month"] == "2025-01"  # the 2025-02 row is future
    assert out["competitor"]["available"] is True


def test_macro_context_degrades_without_signals_table(tmp_path: Path) -> None:
    db = _news_db_with(tmp_path, with_signals=False)
    out = macro_context("2025-02", news_db=db, fred_dir=None, balances=None)
    assert out["macro_signals"]["available"] is False
    assert out["competitor"]["available"] is True
    assert out["house_prices"]["available"] is False  # no fred_dir
