"""Tests for the exp88 segment scout (agent/segment_scout.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from td_one.agent.grouping import GroupDefinition
from td_one.agent.segment_scout import (
    POLICY_BRIEF,
    audit_prompt,
    build_feature_stats,
    build_scout_frame,
    build_scout_prompt,
    categorize_rejection,
    eligible_rule_columns,
    merge_samples,
    partition_from_groups,
    propose_segments,
)


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(n)],
            "customer_id": [f"c{i}" for i in range(n)],
            "snapshot_date": pd.to_datetime("2023-08-31"),
            "origination_date": pd.to_datetime(
                rng.choice(["2019-05-15", "2020-06-15", "2021-03-15", "2022-01-15"], size=n)
            ),
            "province": rng.choice(["ON", "BC", "AB"], size=n),
            "rate_type": rng.choice(["fixed", "variable"], size=n),
            "mortgage_rate_pct": rng.uniform(1.5, 6.0, size=n),
            "household_income_cad": rng.uniform(30_000, 200_000, size=n),
            "children_under6": rng.integers(0, 3, size=n),
            "children_6to17": rng.integers(0, 3, size=n),
            "insured_flag": rng.random(n) < 0.5,
            "credit_score_band": rng.choice(["below_600", "660_719", "720_plus"], size=n),
        }
    )


class _FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def test_scout_frame_rejects_outcome_columns() -> None:
    df = _frame()
    df["switched"] = False
    with pytest.raises(ValueError, match="outcome"):
        build_scout_frame(df)


def test_scout_frame_engineers_columns() -> None:
    out = build_scout_frame(_frame())
    assert "origination_year" in out.columns
    assert "n_children" in out.columns
    assert set(out["origination_year"].unique()) <= {2019.0, 2020.0, 2021.0, 2022.0}


def test_eligible_columns_exclude_ids_and_dates() -> None:
    cols = eligible_rule_columns(build_scout_frame(_frame()))
    for banned in ("mortgage_id", "customer_id", "snapshot_date", "origination_date"):
        assert banned not in cols
    assert "origination_year" in cols and "n_children" in cols


def test_feature_stats_have_no_outcome_content() -> None:
    stats = build_feature_stats(build_scout_frame(_frame()))
    for token in ("switched", "attrition rate", "low_rate", "cohort"):
        assert token not in stats.lower()


def test_prompt_audit_clean_for_all_context_variants() -> None:
    df = build_scout_frame(_frame())
    stats = build_feature_stats(df)
    cols = eligible_rule_columns(df)
    for kwargs in (
        {},
        {"policy_brief": True},
        {"shap_context": ["Top attrition drivers: mortgage_rate_pct (0.4)"]},
        {"news_summaries": ["Mortgage renewals rise as rates stay high"]},
        {"macro_lines": ["inflation: direction +1, risk high"]},
    ):
        system, user = build_scout_prompt(stats, cols, **kwargs)
        assert audit_prompt(system, user) == []


def test_policy_brief_has_no_planted_vocabulary() -> None:
    low = POLICY_BRIEF.lower()
    for token in ("low_rate", "osfi_exempt", "family_squeeze", "75000", "75,000"):
        assert token not in low


def test_propose_segments_validates_and_logs_rejections() -> None:
    df = build_scout_frame(_frame())
    reply = (
        '{"groups": ['
        '{"group_id": "a", "rule": "rate_type == \'fixed\'", "rationale": "x"},'
        '{"group_id": "b", "rule": "switched == True", "rationale": "x"},'
        '{"group_id": "c", "rule": "nonsense_col > 3", "rationale": "x"},'
        '{"group_id": "d", "rule": "mortgage_rate_pct", "rationale": "x"}'
        "]}"
    )
    llm = _FakeLLM([reply])
    res = propose_segments(llm, "sys", "user", df, min_group_n=100)
    assert [g.group_id for g in res.groups] == ["a"]
    reasons = {r["reason"] for r in res.rejections}
    assert reasons == {"outcome_column", "unknown_column", "missing_operator"}


def test_propose_segments_retries_after_parse_failure() -> None:
    df = build_scout_frame(_frame())
    good = '{"groups": [{"group_id": "a", "rule": "insured_flag == False", "rationale": "x"}]}'
    llm = _FakeLLM(["not json", "```json\n" + good + "\n```"])
    res = propose_segments(llm, "sys", "user", df, min_group_n=100)
    assert res.parse_failures == 1
    assert len(res.groups) == 1


def test_categorize_rejection_too_small() -> None:
    df = build_scout_frame(_frame(n=300))
    reason = categorize_rejection("mortgage_rate_pct > 5.99", set(df.columns), df)
    assert reason in ("too_small", "empty")


def test_partition_first_match_and_catch_all() -> None:
    df = build_scout_frame(_frame())
    groups = [
        GroupDefinition("g1", "rate_type == 'fixed'"),
        GroupDefinition("g2", "rate_type == 'fixed' and insured_flag == True"),
    ]
    part = partition_from_groups(df, groups)
    assert set(part.values()) <= {"g1", "g2", "__catch_all__"}
    # g2 is shadowed by g1 (first match wins)
    assert "g2" not in set(part.values())
    assert set(part) == set(df["mortgage_id"].astype(str))


def test_merge_samples_majority() -> None:
    a = GroupDefinition("a", "rate_type == 'fixed'")
    b = GroupDefinition("b", "insured_flag == False")
    c = GroupDefinition("c", "household_income_cad < 50000")
    merged = merge_samples([[a, b], [a, c], [a, b]], min_votes=2)
    assert [g.group_id for g in merged] == ["a", "b"]
