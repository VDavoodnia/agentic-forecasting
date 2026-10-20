"""exp87 regression tests: PIT gate, leakage audit, contract predicate.

These guard the experiment's hard gate (no article/signal/history may
postdate the origin) and the citation-contract acceptance rule. Pure
functions only — no LLM or DB access.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "exp87_manager_reports",
    Path(__file__).resolve().parents[1] / "experiments" / "exp87_manager_reports.py",
)
assert _SPEC is not None and _SPEC.loader is not None
exp87 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp87  # dataclass resolution needs the module registered
_SPEC.loader.exec_module(exp87)


def _art(pub_date: str, url: str = "https://x.example/a", score: float = 0.5) -> dict:
    return {"pub_date": pub_date, "title": "t", "url": url, "score": score}


def test_filter_pit_drops_post_origin_and_pre_window() -> None:
    origin = "2026-07"  # cutoff 2026-08-01, 3-month window starts 2026-05-01
    arts = [
        _art("2026-07-31", "https://x.example/in"),
        _art("2026-08-01", "https://x.example/post"),  # == cutoff: excluded
        _art("2026-04-30", "https://x.example/pre"),  # before window: excluded
        _art("2026-05-01", "https://x.example/edge"),  # window start: included
    ]
    kept = exp87.filter_pit(arts, origin, window_months=3)
    urls = {a["url"] for a in kept}
    assert urls == {"https://x.example/in", "https://x.example/edge"}


def test_filter_pit_dedupes_by_url_keeping_best_score() -> None:
    arts = [
        _art("2026-07-01", "https://x.example/dup", score=0.3),
        _art("2026-07-02", "https://x.example/dup", score=0.9),
    ]
    kept = exp87.filter_pit(arts, "2026-07", window_months=3)
    assert len(kept) == 1 and kept[0]["score"] == 0.9


def test_audit_catches_planted_violations() -> None:
    good = {
        "segment": "insured",
        "origin": "2026-07",
        "window_months": 3,
        "articles": [_art("2026-06-15")],
        "signal_months": ["2026-07"],
        "history_max_month": "2026-07",
    }
    assert exp87.audit_run_log([good]) == []
    bad = dict(good, articles=[_art("2026-08-02")], signal_months=["2026-08"])
    violations = exp87.audit_run_log([bad])
    assert len(violations) == 2


def test_contract_requires_url_citation_when_articles_exist() -> None:
    stat_only = "The forecast median is 0.0123 [stat]. Risk is HIGH [stat]."
    ok, reasons = exp87.contract_ok(stat_only, n_articles=3)
    assert not ok and any("url" in r for r in reasons)
    with_news = stat_only + " Rates stayed high this month [url:0]."
    ok, _ = exp87.contract_ok(with_news, n_articles=3)
    assert ok
    ok, _ = exp87.contract_ok(stat_only, n_articles=0)
    assert ok  # no articles provided: stat-only is acceptable


def test_contract_rejects_uncited_sentences() -> None:
    text = "The forecast median is 0.0123 [stat]. Borrowers are struggling badly."
    ok, _ = exp87.contract_ok(text, n_articles=0)
    assert not ok


def test_month_helpers() -> None:
    assert exp87.shift_month("2026-01", -1) == "2025-12"
    assert exp87.month_before_cutoff("2025-12") == "2026-01-01"
    assert exp87.window_start("2026-07", 3) == "2026-05-01"
    assert exp87.window_start("2026-03", 6) == "2025-10-01"
