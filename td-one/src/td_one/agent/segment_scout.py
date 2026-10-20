"""Segment scout: an LLM proposes borrower segment rules from statistics.

exp88's subject. The scout sees **aggregate statistics only** — never
individual borrower rows and (in the base configuration) never any
outcome information:

- Per-feature quantiles / level shares of the borrower metadata
  cross-section (training period, ``observed_asof <= origin``).
- Joint feature counts for a curated set of attribute pairs/triples —
  pure feature aggregates, so conjunction-shaped structure is visible
  as concrete cells with sizes.
- Optionally: analyst-tool context (SHAP drivers / interactions /
  segment profiles, exp66e), PIT-safe news summaries + macro signals,
  and a generic policy brief (public-domain facts, no DGP parameters).

Output is strict JSON proposals of :class:`GroupDefinition` rules,
validated by :func:`td_one.agent.grouping.validate_rule` (the PIT
guard: metadata columns only, comparison operator per clause, minimum
size). Every rejection is categorized and logged.

Leakage contract (tested): prompts never contain cohort names, planted
DGP parameters, or outcome columns. The statistics builder refuses a
frame that carries outcome columns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from td_one.agent.grouping import OUTCOME_COLUMNS, GroupDefinition, validate_rule
from td_one.agent.llm_grouper import parse_group_proposals
from td_one.evaluation.shap_analysis import engineer_features

logger = logging.getLogger(__name__)

# Columns never eligible for rules or statistics: identifiers and dates
# (dates are represented by the engineered ``origination_year``).
ID_COLUMNS = ("mortgage_id", "customer_id", "household_id", "renewal_episode_id")
DATE_COLUMNS = (
    "snapshot_date",
    "origination_date",
    "term_start_date",
    "maturity_date",
    "credit_score_asof",
    "property_value_asof",
)

DEFAULT_MIN_GROUP_N = 200
DEFAULT_MAX_GROUPS = 8

# Generic Canadian mortgage-policy facts (public domain). Deliberately
# free of any simulator parameter: no thresholds, no vintage cutoffs
# beyond what any practitioner would recite from memory.
POLICY_BRIEF = """\
Canadian mortgage context (general knowledge):
- Since 2018, the OSFI B-20 stress test requires UNINSURED borrowers
  (down payment >= 20%) to qualify at the higher of contract rate + 2%
  or the benchmark rate. Insured mortgages (smaller down payments) are
  default-insured and underwritten under separate insurer rules, so
  insured and uninsured borrowers face different renewal constraints.
- Fixed rates in 2020-2021 fell to historic lows; the Bank of Canada
  then raised its policy rate from 0.25% to 5.00% during 2022-2023.
  Borrowers who locked low fixed rates early in that window face a
  large payment increase if they renew at current rates, which makes
  them reluctant to move lenders but very rate-sensitive at renewal.
- Variable-rate borrowers felt the hikes immediately; fixed-rate
  borrowers feel them only at renewal.
- Households with several young children face high childcare costs and
  have less slack to absorb payment increases; lower-income households
  have the least slack of all.
- Borrowers with strong credit have the most lender choice at renewal;
  weaker-credit borrowers are closer to captive."""


def build_scout_frame(train_meta: pd.DataFrame) -> pd.DataFrame:
    """Engineer the scout's view of the borrower cross-section.

    Adds ``origination_year`` + ``n_children`` (same engineering the
    SHAP incumbent used, exp66e) and refuses frames carrying outcome
    columns — the scout's information set is metadata only.
    """
    leaked = OUTCOME_COLUMNS & set(train_meta.columns)
    if leaked:
        raise ValueError(f"scout frame carries outcome columns: {sorted(leaked)}")
    return engineer_features(train_meta)


def eligible_rule_columns(df: pd.DataFrame) -> list[str]:
    """Columns a proposed rule may reference (no ids, no raw dates)."""
    banned = set(ID_COLUMNS) | set(DATE_COLUMNS) | set(OUTCOME_COLUMNS)
    return [c for c in df.columns if c not in banned and df[c].nunique(dropna=False) >= 2]


def _fmt_num(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _joint_table(df: pd.DataFrame, cols: list[str], label: str) -> str:
    """Share-of-book table for a feature conjunction (counts only)."""
    g = df.groupby(cols, dropna=False, observed=True).size()
    g = g[g > 0].sort_values(ascending=False)
    total = len(df)
    parts = [
        f"{' & '.join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,)))}: "
        f"{n / total:.1%}"
        for idx, n in g.items()
    ]
    return f"- {label}: " + "; ".join(parts)


def build_feature_stats(df: pd.DataFrame) -> str:
    """Feature-only statistics block for the scout prompt.

    Marginals (quantiles / level shares) + joint counts for the
    conjunctions a segment scout would naturally inspect. Contains NO
    outcome information by construction (the frame carries none).
    """
    lines: list[str] = []
    n = len(df)
    lines.append(f"Book: {n:,} borrowers (single cross-section, training period).")
    if "province" in df.columns:
        prov = df["province"].value_counts(normalize=True).head(6)
        lines.append("Province mix: " + ", ".join(f"{k} {v:.0%}" for k, v in prov.items()) + ".")
    lines.append("")
    lines.append("Per-feature distributions:")
    for col in eligible_rule_columns(df):
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            lines.append(f"- {col}: True {s.mean():.1%} / False {1 - s.mean():.1%}")
        elif pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            lines.append(
                f"- {col}: p10={_fmt_num(q.iloc[0])}, p25={_fmt_num(q.iloc[1])}, "
                f"median={_fmt_num(q.iloc[2])}, p75={_fmt_num(q.iloc[3])}, "
                f"p90={_fmt_num(q.iloc[4])}"
            )
        else:
            vc = s.astype(str).value_counts(normalize=True)
            shown = ", ".join(f"{k} {v:.0%}" for k, v in vc.head(8).items())
            lines.append(f"- {col}: {shown}")
    lines.append("")
    lines.append("Joint shares of book (feature conjunctions):")
    if {"rate_type", "origination_year"} <= set(df.columns):
        lines.append(
            _joint_table(df, ["rate_type", "origination_year"], "rate_type x origination_year")
        )
    if {"rate_type", "origination_year", "mortgage_rate_pct"} <= set(df.columns):
        d = df.copy()
        d["rate_band"] = pd.cut(
            d["mortgage_rate_pct"],
            bins=[0, 2.0, 3.0, 4.0, 5.0, 99],
            labels=["<=2", "2-3", "3-4", "4-5", ">5"],
        )
        lines.append(
            _joint_table(
                d,
                ["rate_type", "origination_year", "rate_band"],
                "rate_type x origination_year x rate band (%)",
            )
        )
    if {"insured_flag", "credit_score_band"} <= set(df.columns):
        lines.append(
            _joint_table(df, ["insured_flag", "credit_score_band"], "insured x credit band")
        )
    if {"household_income_cad", "n_children"} <= set(df.columns):
        d = df.copy()
        d["income_band"] = pd.cut(
            d["household_income_cad"],
            bins=[0, 50000, 75000, 100000, 150000, 10**9],
            labels=["<50k", "50-75k", "75-100k", "100-150k", ">=150k"],
        )
        d["children_band"] = pd.cut(
            d["n_children"], bins=[-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"]
        )
        lines.append(_joint_table(d, ["income_band", "children_band"], "income band x children"))
    if {"province", "rate_type"} <= set(df.columns):
        lines.append(_joint_table(df, ["province", "rate_type"], "province x rate_type"))
    return "\n".join(lines)


def build_scout_prompt(
    stats: str,
    columns: list[str],
    *,
    categorical_levels: dict[str, list[str]] | None = None,
    shap_context: list[str] | None = None,
    news_summaries: list[str] | None = None,
    macro_lines: list[str] | None = None,
    policy_brief: bool = False,
    max_groups: int = DEFAULT_MAX_GROUPS,
) -> tuple[str, str]:
    """Build (system, user) for one scout proposal call."""
    system = (
        "You are a segment scout for a mortgage retention team. You propose "
        "borrower segments as executable rules. Output STRICT JSON only: "
        '{"groups": [{"group_id": "...", "rule": "...", "rationale": "..."}]}. '
        "Each rule must be a valid pandas DataFrame.query() expression over "
        "the listed columns only, combining comparisons with and/or/not. "
        "Every clause must contain a comparison operator (==, !=, <, <=, >, "
        ">=, in). String values must be quoted, e.g. rate_type == 'fixed'. "
        "Boolean values are True/False, e.g. insured_flag == False. "
        "Keep each rationale to one sentence with no numbers."
    )
    lines = [
        "Below are aggregate statistics for a mortgage book. You see NO",
        "individual borrowers and NO labels. Your job: propose",
        "segments of borrowers whose CIRCUMSTANCES differ in ways that",
        "plausibly drive different renewal behaviour (switching lender,",
        "negotiating, staying). Think in conjunctions: a segment is usually",
        "a combination of contract terms, household situation, and credit",
        "position - not a single attribute, and not a risk score.",
        "",
        f"Propose at most {max_groups} segments, ordered by how confident",
        "you are that each is a real, distinct segment. Each segment must",
        "cover at least ~5% of the book. Segments should be mutually",
        "exclusive in intent (the first matching rule wins). Borrowers",
        "matching no rule form the residual segment automatically - do",
        "not propose a catch-all.",
        "",
        "Columns you may use in rules:",
        ", ".join(columns),
        "",
    ]
    if categorical_levels:
        lines.append("Categorical levels:")
        for col, levels in categorical_levels.items():
            lines.append(f"- {col}: {', '.join(levels)}")
        lines.append("")
    lines.append(stats)
    if shap_context:
        lines += [
            "",
            "Analyst findings from an attrition model fit on this book",
            "(which attributes drive switching, and how borrowers cluster",
            "by their drivers - use as hints, not as segment definitions):",
        ]
        lines += [f"- {s}" for s in shap_context]
    if policy_brief:
        lines += ["", POLICY_BRIEF]
    if macro_lines:
        lines += ["", "Macro signals as of the forecast origin (direction -2..+2):"]
        lines += [f"- {s}" for s in macro_lines]
    if news_summaries:
        lines += ["", "Recent news summaries (all published before the origin):"]
        lines += [f"- {s}" for s in news_summaries]
    lines += ["", "Propose the segments now."]
    return system, "\n".join(lines)


def categorical_levels_of(df: pd.DataFrame) -> dict[str, list[str]]:
    """Level lists for non-numeric rule columns (prompt grounding)."""
    out: dict[str, list[str]] = {}
    for col in eligible_rule_columns(df):
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            vals = sorted({str(v) for v in s.dropna().unique()})
            if 1 < len(vals) <= 12:
                out[col] = vals
    return out


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def categorize_rejection(rule: str, columns: set[str], df: pd.DataFrame) -> str:
    """Why did validate_rule reject this rule? (diagnostic mirror)."""
    lowered = rule.lower()
    for oc in OUTCOME_COLUMNS:
        if re.search(rf"(?<![a-z0-9_]){re.escape(oc)}(?![a-z0-9_])", lowered):
            return "outcome_column"
    dequoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", rule)
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", dequoted)) - {
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "None",
    }
    if tokens - columns:
        return "unknown_column"
    clauses = re.split(r"\band\b|\bor\b", rule, flags=re.IGNORECASE)
    for clause in clauses:
        c = clause.strip().strip("()")
        if not c or c.lower() == "not":
            continue
        if c.lower().startswith("not"):
            c = c[3:].strip().strip("()")
        if not re.search(r"==|!=|<=|>=|<|>|\bin\b|\bis\b", c, flags=re.IGNORECASE):
            return "missing_operator"
    try:
        matched = df.query(rule)
    except Exception:  # noqa: BLE001 — query failure is the signal
        return "query_error"
    if len(matched) == 0:
        return "empty"
    return "too_small"


@dataclass
class ScoutResult:
    """Outcome of one scout proposal loop."""

    groups: list[GroupDefinition] = field(default_factory=list)
    rejections: list[dict[str, object]] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    parse_failures: int = 0
    attempts: int = 0


def propose_segments(
    llm: object,  # .complete(prompt, system=..., json_mode=...) -> str
    system: str,
    user: str,
    df: pd.DataFrame,
    *,
    min_group_n: int = DEFAULT_MIN_GROUP_N,
    max_attempts: int = 3,
) -> ScoutResult:
    """Propose -> parse -> validate, with rejection-feedback retries.

    Retry feedback lists only syntactic/size rejection reasons — never
    anything about outcomes or scoring.
    """
    result = ScoutResult()
    columns = set(df.columns)
    feedback = ""
    for attempt in range(max_attempts):
        result.attempts = attempt + 1
        text = llm.complete(user + feedback, system=system, json_mode=True)  # type: ignore[attr-defined]
        result.raw_texts.append(text)
        proposals = parse_group_proposals(_strip_fences(text))
        if proposals is None:
            result.parse_failures += 1
            logger.warning("scout: unparseable JSON (attempt %d)", attempt)
            feedback = "\n\nYour previous reply was not valid JSON. Reply with strict JSON only."
            continue
        valid: list[GroupDefinition] = []
        attempt_rejections: list[dict[str, object]] = []
        seen_rules: set[str] = set()
        for p in proposals:
            rule = str(p.get("rule", "")).strip()
            if rule in seen_rules:
                attempt_rejections.append({"rule": rule, "reason": "duplicate", "attempt": attempt})
                continue
            seen_rules.add(rule)
            gd = validate_rule(rule, columns, min_group_n=min_group_n, df=df)
            if gd is None:
                reason = categorize_rejection(rule, columns, df)
                attempt_rejections.append({"rule": rule, "reason": reason, "attempt": attempt})
            else:
                valid.append(
                    GroupDefinition(
                        group_id=str(p.get("group_id", f"s{len(valid)}")),
                        rule=gd.rule,
                        rationale=str(p.get("rationale", "")),
                    )
                )
        result.rejections.extend(attempt_rejections)
        if valid:
            result.groups = valid
            return result
        reasons = sorted({str(r["reason"]) for r in attempt_rejections}) or ["no_proposals"]
        feedback = (
            "\n\nAll proposed rules were rejected by the validator "
            f"(reasons: {', '.join(reasons)}). Rules must reference only the "
            f"listed columns, include a comparison operator in every clause, "
            f"and match at least {min_group_n} borrowers. Propose again."
        )
        logger.warning("scout: no valid groups (attempt %d): %s", attempt, reasons)
    return result


def partition_from_groups(df: pd.DataFrame, groups: list[GroupDefinition]) -> dict[str, str]:
    """First-match partition {mortgage_id: group_id} + ``__catch_all__``."""
    mids = df["mortgage_id"].astype(str)
    assignment: dict[str, str] = dict.fromkeys(mids, "__catch_all__")
    claimed: set[str] = set()
    for g in groups:
        try:
            hits = set(df.query(g.rule)["mortgage_id"].astype(str))
        except Exception:  # noqa: BLE001 — validated already; defensive
            continue
        for m in sorted(hits - claimed):
            assignment[m] = g.group_id
            claimed.add(m)
    return assignment


def _normalize_rule(rule: str) -> str:
    return re.sub(r"\s+", " ", rule.strip().lower())


def merge_samples(
    samples: list[list[GroupDefinition]], min_votes: int = 2
) -> list[GroupDefinition]:
    """Self-consistency merge: keep rules proposed in >= ``min_votes`` samples.

    Ordered by votes (desc), then first appearance. If nothing reaches
    ``min_votes``, fall back to the single most-voted rule set ordering:
    all rules sorted the same way (the caller sees the vote counts in
    the returned groups' order; the experiment logs the fallback).
    """
    votes: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    exemplar: dict[str, GroupDefinition] = {}
    order = 0
    for sample in samples:
        seen_in_sample: set[str] = set()
        for g in sample:
            key = _normalize_rule(g.rule)
            if key not in first_seen:
                first_seen[key] = order
                exemplar[key] = g
            order += 1
            if key not in seen_in_sample:
                votes[key] = votes.get(key, 0) + 1
                seen_in_sample.add(key)
    ranked = sorted(votes, key=lambda k: (-votes[k], first_seen[k]))
    kept = [exemplar[k] for k in ranked if votes[k] >= min_votes]
    if not kept:
        kept = [exemplar[k] for k in ranked]
    return kept


def audit_prompt(system: str, user: str) -> list[str]:
    """Return leakage violations found in a prompt (empty = clean).

    Checks for cohort names and planted-DGP vocabulary. Generic numbers
    (bin edges, quantiles) are statistics, not parameters, and are not
    flagged; the strings below can only appear via leakage.
    """
    text = f"{system}\n{user}".lower()
    violations = []
    for token in ("low_rate", "osfi_exempt", "family_squeeze", "planted", "cohort"):
        if token in text:
            violations.append(token)
    # "attrition" / "outcome" are the project's ordinary English for the
    # task itself; the data-bearing identifiers are what must never leak.
    # (The hard guard is build_scout_frame refusing outcome columns.)
    specific = OUTCOME_COLUMNS - {"attrition", "outcome"}
    for oc in specific:
        if re.search(rf"(?<![a-z0-9_]){re.escape(oc)}(?![a-z0-9_])", text):
            violations.append(f"outcome:{oc}")
    return violations


def proposals_to_json(groups: list[GroupDefinition]) -> list[dict[str, str]]:
    """Serialize proposals for results files."""
    return [{"group_id": g.group_id, "rule": g.rule, "rationale": g.rationale} for g in groups]


__all__ = [
    "POLICY_BRIEF",
    "ScoutResult",
    "audit_prompt",
    "build_feature_stats",
    "build_scout_frame",
    "build_scout_prompt",
    "categorical_levels_of",
    "categorize_rejection",
    "eligible_rule_columns",
    "merge_samples",
    "partition_from_groups",
    "proposals_to_json",
    "propose_segments",
]
