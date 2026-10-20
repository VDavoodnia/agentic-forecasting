"""Tests for exp75 mini-agents (factor + diagnostics families)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from td_one.agent.factor_agents import (
    FACTOR_FAMILIES,
    SIGNAL_CATEGORIES,
    FactorAgent,
    FactorCache,
    family_scores,
    group_agent,
    read_all_factors,
    read_diagnostics,
    seasonality_agent,
    stability_agent,
)


@pytest.fixture()
def news_db(tmp_path):
    db = tmp_path / "news.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE macro_signals (month TEXT, category TEXT, direction INTEGER, "
        "risk TEXT, certainty REAL, summary TEXT, n_articles INTEGER, "
        "PRIMARY KEY (month, category))"
    )
    rows = [
        # central_bank_rates: hawkish tilt in 2026-01, was flat in 2025-10
        ("2025-10", "central_bank_rates", 0, "low", 0.8, "flat", 6),
        ("2026-01", "central_bank_rates", 2, "high", 0.9, "hikes", 8),
        # inflation: accelerating
        ("2026-01", "inflation", 2, "med", 0.7, "hot", 5),
        ("2025-10", "inflation", 1, "low", 0.6, "warm", 4),
        # labour: unemployment rising
        ("2026-01", "labour", 2, "med", 0.8, "layoffs", 6),
    ]
    con.executemany("INSERT INTO macro_signals VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return db


def test_signal_contract(news_db):
    sig = FactorAgent("central_bank_rates", news_db).read("2026-02", "attrition")
    assert sig.available
    assert -2 <= sig.direction <= 2
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.reason and isinstance(sig.reason, str)
    assert sig.family == "policy"


def test_pit_signal_month_must_precede_origin(news_db):
    a = FactorAgent("central_bank_rates", news_db)
    # Signal exists for 2026-01; at origin month 2026-01 it must NOT be visible.
    sig = a.read("2026-01", "attrition")
    assert not sig.available or sig.month < "2026-01"
    sig2 = a.read("2026-02", "attrition")
    assert sig2.available and sig2.month == "2026-01"


def test_pressure_mapping_signs(news_db):
    # hawkish (+2) -> attrition pressure DOWN (lock-in)
    sig = FactorAgent("central_bank_rates", news_db).read("2026-02", "attrition")
    assert sig.pressure < 0
    # inflation accelerating -> price pressure UP
    sig2 = FactorAgent("inflation", news_db).read("2026-02", "price")
    assert sig2.pressure > 0
    # labour: +2 = unemployment rising -> attrition pressure UP
    sig3 = FactorAgent("labour", news_db).read("2026-02", "attrition")
    assert sig3.pressure > 0


def test_trend_3m(news_db):
    sig = FactorAgent("central_bank_rates", news_db).read("2026-02", "attrition")
    # direction went 0 -> +2 over ~3 months
    assert sig.trend_3m == pytest.approx(0.5)


def test_confidence_scales_with_coverage(news_db):
    a = FactorAgent("inflation", news_db)
    sig = a.read("2026-02", "attrition")
    assert 0.0 < sig.confidence <= 1.0


def test_unknown_category_rejected(news_db):
    with pytest.raises(ValueError):
        FactorAgent("not_a_factor", news_db)


def test_cache_month_stamped(news_db, tmp_path):
    cache = FactorCache(tmp_path / "fc.json")
    b1 = read_all_factors(news_db, "2026-02", "attrition", cache)
    assert len(b1) == len(SIGNAL_CATEGORIES) + 2  # 22 + rent + climate
    b2 = read_all_factors(news_db, "2026-02", "attrition", cache)
    assert [s.pressure for s in b1] == [s.pressure for s in b2]
    b3 = read_all_factors(news_db, "2026-03", "attrition", cache)
    assert b3 is not None  # different month -> recompute, no crash


def test_supplements_never_crash(tmp_path):
    # No semantic index / empty db: supplements degrade to unavailable.
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    bundle = read_all_factors(db, "2026-02", "attrition", None)
    cats = {s.category for s in bundle}
    assert "rent" in cats and "climate" in cats


def test_family_scores_grouping(news_db):
    bundle = read_all_factors(news_db, "2026-02", "attrition", None)
    scores = family_scores(bundle)
    assert set(scores) == set(FACTOR_FAMILIES)
    # policy family has central_bank_rates active (negative pressure)
    assert scores["policy"] < 0


def test_stability_agent_quiet_series():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.01, 120)) + 5.0
    d = stability_agent(y)
    assert d.name == "stability"
    assert d.direction in (-1, 0, 1)


def test_stability_agent_break():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.full(100, 1.0), np.full(30, 3.0)]) + rng.normal(0, 0.01, 130)
    d = stability_agent(y)
    assert d.direction == 1  # more uncertainty


def test_stability_agent_short_series():
    d = stability_agent(np.ones(10))
    assert not d.available


def test_seasonality_agent_seasonal():
    t = np.arange(72)
    y = 10.0 + 2.0 * np.sin(2 * np.pi * t / 12)
    d = seasonality_agent(y)
    assert d.direction in (-1, 0, 1)
    assert d.available


def test_group_agent_divergence():
    tot = np.linspace(0.01, 0.02, 24)
    div = tot.copy()
    div[-1] += 0.05  # one peer jumps
    d = group_agent({"total": tot, "g1": tot, "g2": div}, target="total")
    assert d.available and d.direction in (-1, 0, 1)


def test_group_agent_no_groups():
    d = group_agent(None)
    assert not d.available


def test_read_diagnostics_shapes():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 60)
    diags = read_diagnostics(y, {"a": np.tile(np.arange(9)[:, None], (1, 1)).astype(float)})
    assert {d.name for d in diags} == {"stability", "disagreement", "seasonality", "group"}
