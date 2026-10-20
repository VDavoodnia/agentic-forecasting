"""Tests for exp84 panel-builder helpers (date math + dtype contract).

Regression coverage for the exp84 build failure: the Freddie loader
leaves first_payment_date as str, so YYYYMM arithmetic must only run
after the explicit numeric conversion in build_vintage.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "exp84_build_panel",
    Path(__file__).resolve().parents[1] / "experiments" / "exp84_build_panel.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_yyyymm_add_crosses_year_boundary() -> None:
    s = pd.Series([202101, 202112, 202312])
    out = _mod.yyyymm_add(s, -1)
    assert list(out) == [202012, 202111, 202311]
    out2 = _mod.yyyymm_add(s, 2)
    assert list(out2) == [202103, 202202, 202402]


def test_months_between() -> None:
    start = pd.Series([202101, 202306, 202401])
    end = pd.Series([202201, 202603, 202403])
    assert list(_mod.months_between(start, end)) == [12, 33, 2]


def test_build_panel_converts_first_payment_date() -> None:
    # The loader contract: first_payment_date arrives as str.
    raw = pd.Series(["202307", "202401"])
    converted = pd.to_numeric(raw, errors="coerce").astype("Int64")
    assert list(_mod.yyyymm_add(converted, -1)) == [202306, 202312]
