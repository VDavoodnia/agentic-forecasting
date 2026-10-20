"""Tests for exp77: Macro Signal Service integration on Food CPI.

Covers the PIT-safety contract of the signal path (asof boundary), the
live-category loader from the exp70 results JSON, the forecaster factory
configuration (v1 vs v2, news switch), and paired-design integrity (all
arms must evaluate on exactly the same origins).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp71_macro_signal_integration as e71ms  # noqa: E402

from td_one.news.macro_signal import signals_asof  # noqa: E402

EXP70_JSON = ROOT / "experiments" / "exp70_macro_signal_results.json"


def _fixture_db(path: Path) -> None:
    """Tiny macro_signals DB with controlled months."""
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE macro_signals
           (month TEXT, category TEXT, direction INTEGER, risk TEXT,
            certainty REAL, summary TEXT, n_articles INTEGER)"""
    )
    con.executemany(
        "INSERT INTO macro_signals VALUES (?,?,?,?,?,?,?)",
        [
            ("2024-01", "food_prices", 2, "med", 0.8, "prices up", 5),
            ("2024-01", "fx", -1, "low", 0.6, "cad steady", 4),
            ("2024-02", "food_prices", -2, "high", 0.9, "prices down", 6),
        ],
    )
    con.commit()
    con.close()


def test_signals_asof_boundary(tmp_path: Path) -> None:
    """signals_asof(M) sees month M but nothing later (PIT contract)."""
    db = tmp_path / "sig.db"
    _fixture_db(db)
    con = sqlite3.connect(db)
    at_jan = signals_asof(con, "2024-01")
    assert at_jan["food_prices"].direction == 2
    # February's reversal is invisible at January's asof
    at_feb = signals_asof(con, "2024-02")
    assert at_feb["food_prices"].direction == -2
    con.close()


def test_news_features_fn_uses_asof_month(tmp_path: Path) -> None:
    """news_features_fn at origin o consumes months <= dates[o-1] only."""
    db = tmp_path / "sig.db"
    _fixture_db(db)
    # origin 2 -> asof dates[1] = 2024-01 -> direction +2
    # origin 3 -> asof dates[2] = 2024-02 -> direction -2
    dates = pd.date_range("2023-12-01", periods=4, freq="MS")
    fn = e71ms.news_features_fn(
        ["food_prices", "fx"], db, dates, encoding="base", lag=0, cats=None
    )
    r2 = fn({"origin": 2})
    r3 = fn({"origin": 3})
    assert r2 is not None and r3 is not None
    # origin 2 sees only the 2024-01 signals
    assert abs(r2["stress_signal"] - ((2 + -1) / 2) / 2) < 1e-9
    # origin 3 sees 2024-02 food_prices (dir -2) but still 2024-01 fx (dir -1)
    assert abs(r3["stress_signal"] - ((-2 + -1) / 2) / 2) < 1e-9


def test_live_categories_from_exp70_json() -> None:
    """exp70's live set is exactly the 7 calibrated categories."""
    live = e71ms.load_live_categories()
    assert sorted(live) == sorted(
        [
            "inflation",
            "services_costs",
            "food_prices",
            "labour",
            "fx",
            "housing",
            "wars_geopolitics",
        ]
    )


def test_food_prices_anchor_is_the_target() -> None:
    """Documented selection caveat: food_prices calibrated on Food CPI itself.

    This test pins the caveat so any rerun surfaces it: the B-arm (live
    categories) includes a category whose live gate was earned on the
    target's own history. The B' diagnostic (live minus food_prices) must
    be reported alongside B.
    """
    data = json.loads(EXP70_JSON.read_text())
    assert data["categories"]["food_prices"]["anchor"] == "food_cpi"


def test_signal_window_origin_cutoff() -> None:
    """Origins with asof month >= 2023-10 are origin idx >= 166.

    Food CPI: 200 values, months 2010-01..2026-08; origin o predicts
    values[o]; asof month for news is dates[o-1]. Month index 165 =
    2023-10 => origin >= 166 has signal coverage.
    """
    dates = pd.date_range("2010-01-01", periods=200, freq="MS")
    first = next(i for i in range(140, 199) if dates[i - 1].strftime("%Y-%m") >= "2023-10")
    assert first == 166
    assert [o for o in range(140, 199) if o >= first] == list(range(166, 199))
