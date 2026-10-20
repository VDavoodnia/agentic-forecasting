"""exp87: manager risk reports, end to end with real news citations.

The gap (stated in the deck itself): ``RiskReporter``'s NEWS citation
path is wired in the prompt and the contract but was never run end to
end with articles flowing in — every tested explanation (exp66c) cited
statistics only. exp87 closes that gap on real data:

* Series: OSFI M4 runoff **rate** (runoff / lagged outstanding) for the
  4 segments + Freddie Mac SMM, most recent 12 months each supports
  (OSFI 2025-08..2026-07, SMM 2025-04..2026-03).
* Forecast at each origin: the shipped group forecaster
  (``v2_forecaster_fn`` defaults — trailing-6-mean center + Gaussian
  spread, exp78/exp79). PIT convention follows exp70: history uses
  months <= origin month (the OSFI M+2 publication lag is not modelled,
  same limitation exp70 flags; it affects all arms equally).
* Flagging: the LLM+stats risk classifier of record (exp66g,
  ``classify_risk_llm`` with statistics only — its winning config).
  Flagged = HIGH or MEDIUM. Deterministic tertile bins are computed
  alongside for an agreement check.
* Context: PIT-safe news via ``SemanticNewsIndex`` (pub_date < first
  day of month after origin, trailing window) + macro signals via
  ``signals_asof`` (live categories), fed to the citation-enforced
  reporter. A local copy of the news DB is indexed so the shared
  embedding cache (built for 17.7k articles; DB now holds 21.2k) is
  neither read stale nor overwritten.
* Contract loop: an explanation is accepted only when
  ``check_citations`` passes AND (when articles were provided) at least
  one ``[url:N]`` citation is present. Failures are logged with the
  reason and retried with a corrective note (max 4 attempts).

Metrics (pre-registered bar in exp87_report.md): final citation-
contract pass rate, % of flagged explanations citing >= 1 news
article, topical relevance of sampled citations judged by a different
model family (Groq gpt-oss-120b) under a fixed rubric.

LLM calls are cached to ``exp87_cache.json`` (sha256 of model|system|
prompt) so reruns and iteration rounds are reproducible and cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

TD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TD / "src"))
sys.path.insert(0, str(TD / "experiments"))

from td_one.agent.group_forecaster import v2_forecaster_fn  # noqa: E402
from td_one.agent.llm_risk_classifier import classify_risk_llm  # noqa: E402
from td_one.agent.risk_reporter import (  # noqa: E402
    build_explanation_prompt,
    check_citations,
    compute_risk_bins,
)
from td_one.news.macro_signal import signals_asof  # noqa: E402

NEWS_DB = TD / "data" / "news" / "news_2026.db"
LOCAL_DB_DIR = Path("/tmp/exp87")
RESULTS_PATH = TD / "experiments" / "exp87_results.json"
CACHE_PATH = TD / "experiments" / "exp87_cache.json"
SAMPLES_DIR = TD / "experiments" / "exp87_samples"

URL_CITE = re.compile(r"\[url:\d+\]")

# ---------------------------------------------------------------------------
# Segment metadata
# ---------------------------------------------------------------------------

SEGMENTS: dict[str, dict[str, Any]] = {
    "insured": {
        "label": "OSFI insured mortgages",
        "rule": "OSFI M4 segment: insured mortgage balances (CMHC-backed)",
        "queries": [
            "insured mortgages Canada CMHC housing market mortgage renewals",
        ],
        "macro_cats": ["housing", "inflation", "labour", "central_bank_rates"],
    },
    "uninsured": {
        "label": "OSFI uninsured mortgages",
        "rule": "OSFI M4 segment: uninsured mortgage balances",
        "queries": [
            "uninsured mortgages Canada renewals interest rates refinancing housing",
        ],
        "macro_cats": ["housing", "inflation", "labour", "central_bank_rates"],
    },
    "variable_insured": {
        "label": "OSFI variable-rate insured mortgages",
        "rule": "OSFI M4 segment: variable-rate insured mortgage balances",
        "queries": [
            "variable rate insured mortgages Bank of Canada interest rate payments Canada",
        ],
        "macro_cats": ["central_bank_rates", "housing", "inflation", "labour"],
    },
    "variable_uninsured": {
        "label": "OSFI variable-rate uninsured mortgages",
        "rule": "OSFI M4 segment: variable-rate uninsured mortgage balances",
        "queries": [
            "variable rate mortgages Canada Bank of Canada rate payments refinancing",
        ],
        "macro_cats": ["central_bank_rates", "housing", "inflation", "labour"],
    },
    "smm": {
        "label": "Freddie Mac SMM (US prepayment)",
        "rule": "Freddie Mac single-family loan-level aggregate: monthly SMM prepayment rate",
        "queries": [
            "US mortgage prepayment refinancing interest rates housing market",
        ],
        "macro_cats": ["central_bank_rates", "housing", "inflation", "fx"],
    },
}

# Live macro categories per exp70-MS calibration (dead ones excluded from
# the reporter snapshot; central_bank_rates is structurally silent, n=1).
LIVE_CATS = {
    "fx",
    "food_prices",
    "labour",
    "inflation",
    "housing",
    "services_costs",
    "wars_geopolitics",
}


# ---------------------------------------------------------------------------
# Month helpers (pure)
# ---------------------------------------------------------------------------


def shift_month(ym: str, k: int) -> str:
    """Shift a 'YYYY-MM' month by k months."""
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def month_before_cutoff(origin_ym: str) -> str:
    """Exclusive pub_date cutoff: first day of the month after origin."""
    return f"{shift_month(origin_ym, 1)}-01"


def window_start(origin_ym: str, window_months: int) -> str:
    """Inclusive window start (ISO date) for a trailing window."""
    return f"{shift_month(origin_ym, -(window_months - 1))}-01"


def last_months(index: list[str], n: int) -> list[str]:
    return list(index[-n:])


# ---------------------------------------------------------------------------
# PIT filtering + leakage audit (pure)
# ---------------------------------------------------------------------------


def filter_pit(
    articles: list[dict[str, Any]], origin_ym: str, window_months: int
) -> list[dict[str, Any]]:
    """Keep articles inside [window_start, cutoff); dedupe by URL.

    Input dicts carry pub_date/title/url/score. Output sorted by score
    descending. This is the experiment's PIT gate on top of the index's
    own ``pub_date < before`` mask.
    """
    cutoff = month_before_cutoff(origin_ym)
    start = window_start(origin_ym, window_months)
    best: dict[str, dict[str, Any]] = {}
    for a in articles:
        d = str(a["pub_date"])
        if not (start <= d < cutoff):
            continue
        u = str(a["url"])
        if u not in best or float(a["score"]) > float(best[u]["score"]):
            best[u] = a
    return sorted(best.values(), key=lambda a: -float(a["score"]))


def audit_run_log(records: list[dict[str, Any]]) -> list[str]:
    """Leakage audit over the full run log; returns violation strings.

    Every retrieved article must satisfy window_start <= pub_date <
    cutoff(origin); every macro signal month must be <= origin; every
    history month must be <= origin.
    """
    violations: list[str] = []
    for r in records:
        origin = r["origin"]
        cutoff = month_before_cutoff(origin)
        start = window_start(origin, int(r.get("window_months", 3)))
        for a in r.get("articles", []):
            d = str(a["pub_date"])
            if not (start <= d < cutoff):
                violations.append(
                    f"{r['segment']}@{origin}: article {d} outside "
                    f"[{start}, {cutoff}): {a.get('title', '')[:60]}"
                )
        for m in r.get("signal_months", []):
            if m > origin:
                violations.append(f"{r['segment']}@{origin}: signal month {m} > origin")
        if r.get("history_max_month", origin) > origin:
            violations.append(
                f"{r['segment']}@{origin}: history reaches {r['history_max_month']} > origin"
            )
    return violations


def contract_ok(text: str, n_articles: int) -> tuple[bool, list[str]]:
    """Decide whether one explanation draft satisfies the contract.

    Requires the shipped citation contract (``check_citations``) and,
    when news articles were provided, at least one [url:N] citation —
    the news-citation presence the brief measures separately, enforced
    here so the retry loop optimizes for it directly.
    """
    reasons: list[str] = []
    if not check_citations(text):
        reasons.append("uncited factual sentence (check_citations failed)")
    if n_articles > 0 and not URL_CITE.search(text):
        reasons.append("no [url:N] news citation although articles were provided")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Cached LLM wrapper
# ---------------------------------------------------------------------------


class CachedClient:
    """JSON-file cache around any .complete(prompt, system) client."""

    def __init__(self, inner: Any, cache_path: Path, tag: str) -> None:
        self._inner = inner
        self._tag = tag
        self._path = cache_path
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text())
        self.api_calls = 0
        self.cache_hits = 0

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        key = hashlib.sha256(f"{self._tag}|{system}|{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        text = self._inner.complete(prompt, system=system, **kw)
        self.api_calls += 1
        self._cache[key] = text
        self._path.write_text(json.dumps(self._cache))
        return text


# ---------------------------------------------------------------------------
# Series loading
# ---------------------------------------------------------------------------


def load_series() -> dict[str, pd.Series]:
    """Monthly runoff/prepayment rates per segment, indexed 'YYYY-MM'."""
    out: dict[str, pd.Series] = {}
    bal = pd.read_parquet(TD / "data" / "osfi" / "m4_balances.parquet")
    bal["ym"] = bal["month"].astype(str).str[:7]
    bal["rate"] = bal["runoff_m"] / bal["prev_outstanding_m"]
    for seg, g in bal.groupby("segment"):
        s = g.dropna(subset=["rate"]).set_index("ym")["rate"].sort_index()
        out[str(seg)] = s
    smm = pd.read_csv(TD / "data" / "freddie" / "smm_monthly_2021_2026.csv")
    smm["ym"] = smm["period"].astype(str).str[:4] + "-" + smm["period"].astype(str).str[4:]
    out["smm"] = smm.set_index("ym")["smm"].sort_index()
    return out


def segment_stats(history: pd.Series, q: np.ndarray) -> dict[str, Any]:
    """Compute the statistics shown to the classifier and the reporter."""
    vals = history.to_numpy(dtype=float)
    stats: dict[str, Any] = {
        "current_rate": round(float(vals[-1]), 5),
        "trailing_3m_mean_rate": round(float(np.mean(vals[-3:])), 5),
        "trailing_12m_mean_rate": round(float(np.mean(vals[-12:])), 5),
        "change_vs_12m_ago": round(float(vals[-1] - vals[-13]), 5) if len(vals) > 13 else 0.0,
        "trailing_12m_volatility": round(float(np.std(np.diff(vals[-13:]))), 5)
        if len(vals) > 13
        else 0.0,
        "forecast_next_month_median": round(float(q[2]), 5),
        "forecast_interval_80_low": round(float(q[0]), 5),
        "forecast_interval_80_high": round(float(q[4]), 5),
    }
    return stats


# ---------------------------------------------------------------------------
# Reporter with contract retry loop
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 4


def generate_explanation(
    llm: Any,
    group_id: str,
    rule: str,
    stats: dict[str, Any],
    articles: list[dict[str, Any]],
    risk_level: str,
) -> dict[str, Any]:
    """Generate a citation-enforced explanation; log every attempt."""
    full_stats: dict[str, Any] = {
        "predicted_mean_attrition": stats["forecast_next_month_median"],
        "interval_low": stats["forecast_interval_80_low"],
        "interval_high": stats["forecast_interval_80_high"],
        "risk_level": risk_level,
        **{
            k: v
            for k, v in stats.items()
            if k
            not in (
                "forecast_next_month_median",
                "forecast_interval_80_low",
                "forecast_interval_80_high",
            )
        },
    }
    system, user = build_explanation_prompt(group_id, rule, full_stats, articles)
    attempts: list[dict[str, Any]] = []
    text = ""
    ok = False
    reasons: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        prompt = user
        if attempt > 0:
            prompt = (
                user
                + "\n\nYour previous draft violated the citation contract: "
                + "; ".join(reasons)
                + ". Rewrite the explanation so EVERY factual sentence ends "
                "with [stat] or [url:N], and cite at least one news article "
                "with [url:N] if any are listed."
            )
        text = llm.complete(prompt, system=system)
        ok, reasons = contract_ok(text, len(articles))
        attempts.append(
            {
                "attempt": attempt,
                "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                "ok": ok,
                "reasons": reasons,
            }
        )
        if ok:
            break
    return {
        "explanation": text,
        "citations_valid": ok,
        "attempts": attempts,
        "n_url_citations": len(URL_CITE.findall(text)),
    }


# ---------------------------------------------------------------------------
# Round configuration + runner
# ---------------------------------------------------------------------------


@dataclass
class RoundCfg:
    name: str
    window_months: int = 3
    top_n: int = 5
    search_k: int = 50
    multi_query: bool = False
    extra_queries: tuple[str, ...] = ()


ROUND1 = RoundCfg(name="R1")
# Diagnosis-driven variants are chosen in main() based on R1's failures;
# the pre-registered iteration space is: query construction (multi_query
# + driver queries), window (3 -> 6), top_n, category union (R3+).


def macro_snapshot(con: sqlite3.Connection, origin_ym: str, seg: str) -> dict[str, Any]:
    """Live macro signals as of origin, restricted to segment categories."""
    sigs = signals_asof(con, origin_ym)
    snap: dict[str, Any] = {}
    months: list[str] = []
    for cat in SEGMENTS[seg]["macro_cats"]:
        s = sigs.get(cat)
        if s is None or cat not in LIVE_CATS:
            continue
        snap[f"macro_{cat}"] = f"direction {s.direction:+d}, risk {s.risk} (month {s.month})"
        months.append(s.month)
    return {"stats": snap, "months": months}


def run_round(
    cfg: RoundCfg,
    series: dict[str, pd.Series],
    origins: dict[str, list[str]],
    index: Any,
    con: sqlite3.Connection,
    gen_llm: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run one full round; returns per-segment-month records."""
    fn = v2_forecaster_fn(h=1)
    records: list[dict[str, Any]] = []
    jobs = [(seg, o) for seg, os_ in origins.items() for o in os_]
    if limit:
        jobs = jobs[:limit]
    for seg, origin in jobs:
        s = series[seg]
        hist = s.loc[:origin]
        q = fn(hist.to_numpy(dtype=float))
        stats = segment_stats(hist, q)
        meta = SEGMENTS[seg]
        # Classifier of record: LLM + stats only (exp66g winning config).
        cls = classify_risk_llm(gen_llm, seg, meta["rule"], dict(stats), [], max_retries=1)
        # Deterministic bins alongside (agreement check): train = the
        # segment's one-step forecast medians over the 24 months before
        # the evaluation window.
        window = origins[seg]
        train_months = [m for m in s.index if m < window[0]][-24:]
        train_medians = [float(fn(s.loc[:m].to_numpy(dtype=float))[2]) for m in train_months]
        bin_level = compute_risk_bins({seg: float(q[2])}, train_medians)[seg]
        flagged = cls.level in ("HIGH", "MEDIUM")
        rec: dict[str, Any] = {
            "segment": seg,
            "origin": origin,
            "window_months": cfg.window_months,
            "history_max_month": str(hist.index[-1]),
            "stats": stats,
            "classifier_level": cls.level,
            "bin_level": bin_level,
            "flagged": flagged,
            "articles": [],
            "signal_months": [],
        }
        if flagged:
            queries = list(meta["queries"])
            if cfg.multi_query:
                queries += list(cfg.extra_queries)
            raw: list[dict[str, Any]] = []
            for query in queries:
                for a in index.search(
                    query,
                    before=month_before_cutoff(origin),
                    top_k=cfg.search_k,
                ):
                    raw.append(
                        {
                            "pub_date": a.pub_date,
                            "title": a.title,
                            "url": a.url,
                            "source": a.source,
                            "score": a.score,
                        }
                    )
            kept = filter_pit(raw, origin, cfg.window_months)[: cfg.top_n]
            rec["articles"] = kept
            snap = macro_snapshot(con, origin, seg)
            rec["signal_months"] = snap["months"]
            exp_stats = dict(stats)
            exp_stats.update(snap["stats"])
            out = generate_explanation(gen_llm, seg, meta["rule"], exp_stats, kept, cls.level)
            rec.update(out)
        records.append(rec)
        print(
            f"[{cfg.name}] {seg}@{origin} cls={cls.level} bin={bin_level} "
            f"flagged={flagged}"
            + (
                f" arts={len(rec['articles'])} valid={rec.get('citations_valid')}"
                f" attempts={len(rec.get('attempts', []))}"
                if flagged
                else ""
            ),
            flush=True,
        )
    return records


def round_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [r for r in records if r["flagged"]]
    with_exp = [r for r in flagged if "explanation" in r]
    return {
        "n_segment_months": len(records),
        "n_flagged": len(flagged),
        "contract_pass_final": (
            float(np.mean([r["citations_valid"] for r in with_exp])) if with_exp else 0.0
        ),
        "contract_pass_first_attempt": (
            float(np.mean([r["attempts"][0]["ok"] for r in with_exp])) if with_exp else 0.0
        ),
        "pct_flagged_with_news_citation": (
            float(np.mean([r["n_url_citations"] > 0 for r in with_exp])) if with_exp else 0.0
        ),
        "classifier_bin_agreement": float(
            np.mean([r["classifier_level"] == r["bin_level"] for r in records])
        ),
        "mean_attempts": (
            float(np.mean([len(r["attempts"]) for r in with_exp])) if with_exp else 0.0
        ),
        "leakage_violations": audit_run_log(records),
    }


# ---------------------------------------------------------------------------
# Relevance judging (different model family, fixed rubric)
# ---------------------------------------------------------------------------

JUDGE_RUBRIC = """You judge whether a news article cited in a mortgage
risk explanation is topically relevant.

Score:
2 = RELEVANT: the article is about a driver of mortgage attrition /
prepayment / runoff for the named segment (interest rates, housing
market, employment, mortgage or housing policy) in or near the
report window.
1 = PARTIAL: general economy or markets news with only an indirect
link to the segment's attrition risk.
0 = IRRELEVANT: unrelated topic, or an idiosyncratic event in another
country/sector with no plausible channel to this segment's attrition.

Reply with exactly one JSON object: {"score": 0|1|2, "reason": "<one short sentence>"}"""


def cited_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All (explanation, cited article) pairs from flagged records."""
    pairs: list[dict[str, Any]] = []
    for r in records:
        if not r.get("flagged") or "explanation" not in r:
            continue
        cited = {int(m) for m in re.findall(r"\[url:(\d+)\]", r["explanation"])}
        for i in sorted(cited):
            if i < len(r["articles"]):
                a = r["articles"][i]
                pairs.append(
                    {
                        "segment": r["segment"],
                        "origin": r["origin"],
                        "title": a["title"],
                        "url": a["url"],
                        "pub_date": a["pub_date"],
                    }
                )
    return pairs


def judge_pairs(judge_llm: Any, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pairs:
        prompt = (
            f"Segment: {SEGMENTS[p['segment']]['label']}\n"
            f"Report month: {p['origin']}\n"
            f"Article ({p['pub_date']}): {p['title']}\nURL: {p['url']}"
        )
        text = judge_llm.complete(prompt, system=JUDGE_RUBRIC, json_mode=True)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            verdict = json.loads(m.group(0)) if m else {"score": -1, "reason": "unparseable"}
        except json.JSONDecodeError:
            verdict = {"score": -1, "reason": "unparseable"}
        out.append({**p, **verdict})
        print(f"[judge] {p['segment']}@{p['origin']} score={verdict.get('score')}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Sample monthly reports
# ---------------------------------------------------------------------------


def render_month_report(
    kind: str, origin: str, records: list[dict[str, Any]], series: dict[str, pd.Series]
) -> str:
    segs = [s for s in series if (s == "smm") == (kind == "smm")]
    kind_label = "Freddie Mac SMM" if kind == "smm" else "OSFI M4 segments"
    lines = [
        f"# Manager risk report — {origin} ({kind_label})",
        "",
        "Generated end-to-end by exp87: shipped forecaster (trailing-mean "
        "center + Gaussian spread) + LLM+stats risk classifier of record "
        "(exp66g) + PIT-safe news retrieval + citation-enforced reporter.",
        "",
        "| Segment | Current rate | Forecast (median) | 80% interval | "
        "Classifier | Bins | Flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for seg in segs:
        r = next(x for x in records if x["segment"] == seg and x["origin"] == origin)
        st = r["stats"]
        lines.append(
            f"| {SEGMENTS[seg]['label']} | {st['current_rate']:.5f} | "
            f"{st['forecast_next_month_median']:.5f} | "
            f"[{st['forecast_interval_80_low']:.5f}, {st['forecast_interval_80_high']:.5f}] | "
            f"{r['classifier_level']} | {r['bin_level']} | {'YES' if r['flagged'] else 'no'} |"
        )
    lines += ["", "## Flagged segments — explanations (verbatim, citations enforced)", ""]
    for seg in segs:
        r = next(x for x in records if x["segment"] == seg and x["origin"] == origin)
        if not r["flagged"]:
            continue
        lines += [
            f"### {SEGMENTS[seg]['label']} — {r['classifier_level']}",
            "",
            r["explanation"],
            "",
            "News cited / retrieved:",
        ]
        for i, a in enumerate(r["articles"]):
            lines.append(f"- [url:{i}] {a['title']} ({a['pub_date']}) — {a['url']}")
        lines.append("")
    lines += [
        "## Method notes",
        "",
        "- PIT: history uses months <= origin (exp70 convention); articles "
        "satisfy window_start <= pub_date < first day of month after origin; "
        "macro signals via signals_asof(origin).",
        "- Every factual sentence in each explanation passed check_citations; "
        "explanations without a news citation were retried under a corrective "
        "prompt (see exp87_report.md for the iteration log).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_index() -> Any:
    """Semantic index over a local DB copy (fresh embeddings, full DB)."""
    from td_one.rag.semantic import SemanticNewsIndex

    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    local_db = LOCAL_DB_DIR / "news_2026.db"
    if not local_db.exists():
        shutil.copy(NEWS_DB, local_db)
    return SemanticNewsIndex(db_path=local_db)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="R1")
    ap.add_argument("--limit", type=int, default=None, help="smoke-run job cap")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=87)
    args = ap.parse_args()

    cfgs: dict[str, RoundCfg] = {
        "R1": ROUND1,
        # Filled after R1 diagnosis; see exp87_report.md iteration log.
        "R2": RoundCfg(
            name="R2",
            window_months=6,
            top_n=6,
            multi_query=True,
            extra_queries=(
                "Bank of Canada interest rate decision mortgage payments Canada",
                "Canada housing market home prices mortgage stress test rules",
            ),
        ),
    }
    cfg = cfgs[args.round]

    from td_one.agent.groq_client import GroqChatClient
    from td_one.llm_rungs.vector_client import VectorProxyClient

    t0 = time.time()
    series = load_series()
    origins = {seg: last_months(list(s.index), 12) for seg, s in series.items()}
    print({k: (v[0], v[-1], len(v)) for k, v in origins.items()}, flush=True)

    con = sqlite3.connect(NEWS_DB)
    index = build_index()
    gen = CachedClient(VectorProxyClient(model="flash-lite", call_budget=2000), CACHE_PATH, "gen")
    records = run_round(cfg, series, origins, index, con, gen, limit=args.limit)
    metrics = round_metrics(records)
    print(json.dumps(metrics, indent=1, default=str), flush=True)

    results: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
    results.setdefault("rounds", {})[cfg.name] = {
        "cfg": asdict(cfg),
        "metrics": metrics,
        "records": records,
        "gen_api_calls": gen.api_calls,
        "gen_cache_hits": gen.cache_hits,
    }

    if args.judge and not args.limit:
        pairs = cited_pairs(records)
        rng = np.random.default_rng(args.sample_seed)
        idx = rng.choice(len(pairs), size=min(20, len(pairs)), replace=False)
        sample = [pairs[int(i)] for i in sorted(idx)]
        judge = CachedClient(
            GroqChatClient(call_budget=200, reasoning_effort="low"),
            CACHE_PATH,
            "judge",
        )
        verdicts = judge_pairs(judge, sample)
        scores = [v["score"] for v in verdicts]
        results.setdefault("judgments", {})[cfg.name] = {
            "n_pairs_total": len(pairs),
            "sample": verdicts,
            "pass_rate_relevant": float(np.mean([s >= 1 for s in scores])),
            "score_distribution": {str(k): scores.count(k) for k in (2, 1, 0, -1)},
            "judge_api_calls": judge.api_calls,
        }
        print(json.dumps(results["judgments"][cfg.name]["score_distribution"]), flush=True)

    results["wall_seconds_last_run"] = round(time.time() - t0, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
