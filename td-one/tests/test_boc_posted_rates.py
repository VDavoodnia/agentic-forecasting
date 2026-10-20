"""Tests for the BoC posted-rates puller (mock-only, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pull_boc_posted_rates import SERIES_MAP, monthly_from_weekly


def _obs(week: str, **vals: str | None) -> dict:
    o: dict = {"d": week}
    for k, v in vals.items():
        if v is not None:
            o[k] = {"v": v}
    return o


def test_monthly_uses_last_wednesday_in_month() -> None:
    # 2025-01 had Wednesdays on the 1st, 8th, 15th, 22nd, 29th.
    obs = [
        _obs("2025-01-01", V80691335="6.50"),
        _obs("2025-01-08", V80691335="6.49"),
        _obs("2025-01-22", V80691335="6.45"),
        _obs("2025-01-29", V80691335="6.40"),
        _obs("2025-02-05", V80691335="6.39"),
    ]
    rows = monthly_from_weekly(obs)
    jan = [r for r in rows if r["month"] == "2025-01" and r["series"] == "V80691335"]
    assert len(jan) == 1
    assert jan[0]["rate_pct"] == 6.40  # last Wednesday, not first or mean
    feb = [r for r in rows if r["month"] == "2025-02" and r["series"] == "V80691335"]
    assert feb[0]["rate_pct"] == 6.39


def test_monthly_skips_series_missing_all_month() -> None:
    # Series present in the week dict but with no 'v' key must not crash.
    obs = [
        _obs("2025-01-08", V80691335=None, V80691311="4.95"),
    ]
    rows = monthly_from_weekly(obs)
    s35 = [r for r in rows if r["series"] == "V80691335"]
    assert s35 == []
    s11 = [r for r in rows if r["series"] == "V80691311"]
    assert len(s11) == 1 and s11[0]["rate_pct"] == 4.95


def test_series_map_covers_expected_products() -> None:
    # 1yr / 3yr / 5yr conventional + prime — the posted-rate backbone.
    prods = {p for _, p in SERIES_MAP.items()}
    assert prods == {"posted_1yr_conventional", "posted_3yr_conventional",
                     "posted_5yr_conventional", "prime_rate"}
