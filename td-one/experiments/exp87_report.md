# exp87: Manager risk reports, end to end with real news

**Status: COMPLETE 2026-10-20. Pre-registered bar PASSED on all three
criteria (final configuration R3; the R1 baseline already passed,
R2–R3 removed its two weaknesses; R4 re-verified the final
configuration on a fresh full-corpus index — 20/20 RELEVANT).** Samples:
[exp87_samples/osfi_2026-07.md](exp87_samples/osfi_2026-07.md),
[exp87_samples/osfi_2025-12.md](exp87_samples/osfi_2025-12.md),
[exp87_samples/smm_2026-03.md](exp87_samples/smm_2026-03.md).
Raw records + judgments: `exp87_results.json`; LLM cache:
`exp87_cache.json`.

## The gap (now closed)

`agent/risk_reporter.py`'s NEWS citation path was wired in the prompt
and the contract (`[url:N]` citations, `check_citations` enforcement)
but had never run end to end with articles flowing in — every tested
explanation (exp66c, 10/10 pass) cited statistics only. exp87 runs the
full stack on real data: shipped forecaster → classifier of record →
PIT-safe news retrieval → citation-enforced reporter → relevance
judging, for the OSFI M4 segments and Freddie SMM over their most
recent 12 months.

## Verdict vs the pre-registered bar (R3, final configuration)

| Criterion | Bar | R3 result |
|---|---|---|
| Citation-contract pass (final) | 100% | **100%** (45/45 flagged explanations) |
| Flagged explanations with ≥1 news citation | ≥ 50% | **100%** (45/45) |
| Sampled news citations topically relevant | ≥ 80% | **100%** (20/20 score ≥ 1: 19 RELEVANT, 1 PARTIAL, 0 irrelevant) |
| Leakage audit | 0 violations | **0** (full run log, all rounds) |

## Protocol (as pre-registered)

- **Series/origins:** OSFI M4 runoff rate (runoff / lagged
  outstanding), 4 segments, origins 2025-08..2026-07; Freddie SMM,
  origins 2025-04..2026-03 — 60 segment-months. PIT convention follows
  exp70 (history months ≤ origin; OSFI M+2 publication lag not
  modelled, exp70's flagged limitation).
- **Forecast:** shipped `v2_forecaster_fn` defaults (trailing-6-mean
  center + Gaussian spread, exp78/exp79), h=1.
- **Flagging:** LLM+stats classifier of record (exp66g's winning
  config, `classify_risk_llm`, statistics only). Flagged = HIGH (16)
  or MEDIUM (29); 15 LOW. Deterministic tertile bins computed
  alongside: **agreement only 0.283** — the bins call HIGH almost
  everywhere on these trending series (they saturate), while the
  classifier discriminates; another real-data instance of exp66c's
  bins critique and of why exp66g replaced them.
- **Retrieval:** `SemanticNewsIndex` (shipped), articles gated to
  window_start ≤ pub_date < first day of month after origin; macro
  signals via `signals_asof(origin)`, live categories only, injected
  as citable `[stat]` facts.
- **Contract loop:** acceptance requires `check_citations` AND ≥1
  `[url:N]` when articles exist; failures logged (attempt, sha,
  reasons) and retried with a corrective note, max 4 attempts. Across
  all rounds the only failure mode observed was a first draft with no
  news citation (3 instances, all fixed on retry); `check_citations`
  itself never failed on a final draft.
- **Judge:** Groq gpt-oss-120b (different family from the flash-lite
  generator), reasoning low, fixed rubric: 2 = article is about a
  driver of this segment's attrition (rates, housing, employment,
  mortgage policy) in/near the window; 1 = general economy, indirect
  link; 0 = unrelated or wrong-jurisdiction idiosyncratic. Pass =
  score ≥ 1. 20 pairs sampled per round (seed 87). Raw judgments in
  `exp87_results.json`.

## Iteration log (diagnose → change → result)

**R1 — baseline (shipped tooling as-is).** Segment query only,
semantic top-50, 3-month window, top-5. Contract 100% final AND
first-attempt; news citations 84.4%; judge 20/20 RELEVANT. Bar passed
already — but diagnosis found the weak spot: **7/45 flagged
segment-months retrieved ZERO articles** (all 7 are exactly the
news-citation misses), concentrated in variable_uninsured and 2026
origins. Cause: the top-50 is global over all history, so the
3-month window filter discarded nearly everything (in-window
survivors: 0–5 per origin).

**R2 — density fix.** search_k 50 → 1000, window 3 → 6 months,
top_n 6, plus two macro-driver queries unioned in. Result: every
flagged segment-month got 6 articles; news citations **100%**;
contract 100% (2 first drafts lacked a `[url]`, fixed on retry).
BUT judge pass fell to 0.90: both score-0 citations were the **same
failure mode — SMM (US prepayment) explanations citing a Bank of
Canada article**. The globally-unioned Canada driver queries polluted
the one US series in a Canada-heavy corpus.

**R3 — per-segment drivers + jurisdiction scoping.** Driver queries
moved into per-segment metadata (SMM gets Federal Reserve / US
housing queries); SMM retrieval drops titles naming Canadian
institutions (deterministic filter; 526 candidates excluded across
the run, counted in the records). Result: contract 100%, news
citations 100%, judge **19 RELEVANT + 1 PARTIAL + 0 irrelevant**.
The residual PARTIAL is itself instructive: a Canada-focused
"Mortgage Digest" article whose title doesn't say "Canada" — a
title-only filter can't catch it; segment-level corpus partitioning
would be the next step if SMM reporting were productized.

**R4 — index-coverage robustness (R3 config, fresh full-corpus
index).** The shipped embedding cache covers only 17,738 of 21,234
articles (75–80% in-window). A fresh cache over all articles dated
≥ 2024-01 (10,514; complete in-window coverage) changed 48/270
retrieved articles and grew cited pairs 71 → 80 — and every metric
stayed at ceiling: contract 100%, news citations 100%, judge
**20/20 RELEVANT**, 0 leakage violations. The R1–R3 conclusions do
not depend on the stale cache.

**What the rounds taught:** (1) the citation contract + a cheap
corrective retry is sufficient — the generator wants to cite and
usually does so on attempt 1 (first-attempt pass 95.6–100% per
round); (2) retrieval *density* (search depth × window) is what
starves or feeds the reporter, not prompt wording; (3) query unions
must be jurisdiction-aware — a global driver query silently
cross-contaminates the one foreign series.

## Cost / runtime

Generator (Vector flash-lite): 182 API calls across R1–R4 (classifier
60 + reporter drafts/retries; prompts cached — later rounds reused
60/98/75 cached completions). Judge (Groq): 54 API calls (later
rounds' samples largely re-judged identical pairs from cache). Wall time: R1 ≈ 28 min
(includes two multi-minute I/O stalls on the shared 2-core box), R2
≈ 6.5 min, R3 ≈ 2.3 min (mostly cache-warm).

## Caveats (honest limits)

- **Index coverage:** R1–R3 ran on the shipped embedding cache
  (17,738 of 21,234 articles; in-window coverage 75–80% — the cache
  predates the macro backfill). R4 re-ran the final configuration on
  a fresh full-corpus cache and reproduced every headline metric
  exactly (see iteration log), so the caveat is retired empirically,
  not just argued away.
- **Judge scope:** one judge model, titles only (~650 of 21k articles
  have bodies; both generator and judge saw titles, so the comparison
  is fair), 20-pair samples. The judge's R1 perfection survived my
  manual read of the same 20 titles (BoC decisions, renewal/payment
  shock, delinquency, refinance — all on-driver).
- **Flag rate:** the classifier flags 45/60 segment-months (MEDIUM is
  a watchlist level by design); a production report would likely
  restrict narrative generation to HIGH + MEDIUM-with-wide-intervals.
- One explanation cited a March-2026 labour signal in a July-2026
  report: that is `signals_asof` returning the latest available month
  (signal-store lag), working as designed — the citation is dated in
  the sample report's stats, not hidden.
- Minor retrieval wart (visible in the SMM sample): http/https twins
  of the same CBS story occupy 4 of 6 slots — dedupe is by exact URL
  string; canonical-URL dedupe is a trivial follow-up fix.

## Files

`exp87_manager_reports.py` (pipeline; rounds R1–R3 in `cfgs`),
`exp87_results.json`, `exp87_cache.json`, `exp87_samples/` (3 monthly
reports), `tests/test_exp87_manager_reports.py` (PIT filter, leakage
audit with planted violations, contract predicate — 6 tests).
