# exp87: Manager risk reports, end to end with real news

**Status: PRE-REGISTERED 2026-10-20 (protocol + bar fixed before the run;
results and iteration log appended below as rounds complete).**

## The gap

`agent/risk_reporter.py`'s NEWS citation path is wired in the prompt and
the contract (`[url:N]` citations, `check_citations` enforcement), but
every tested explanation so far (exp66c, 10/10 pass) cited **statistics
only** — no run ever fed retrieved articles through the reporter on real
data. The deck states this gap explicitly. exp87 closes it.

## Protocol (fixed in advance)

- **Series:** OSFI M4 runoff rate (runoff / lagged outstanding), 4
  segments, origins 2025-08..2026-07; Freddie Mac SMM, origins
  2025-04..2026-03. 60 segment-months. PIT convention follows exp70
  (history months <= origin; OSFI M+2 lag not modelled — exp70's flagged
  limitation, equal across arms).
- **Forecast:** shipped `v2_forecaster_fn` defaults (trailing-6-mean
  center + Gaussian spread, exp78/exp79), h=1.
- **Flagging:** LLM+stats classifier of record (exp66g's winning config:
  `classify_risk_llm`, statistics only). Flagged = HIGH or MEDIUM.
  Deterministic tertile bins computed alongside; agreement reported.
- **Retrieval (PIT gate):** `SemanticNewsIndex` over a local copy of the
  news DB (fresh embeddings over all 21,234 articles; the shared cache
  covers only 17,738 and is left untouched). Articles must satisfy
  window_start(origin, W) <= pub_date < first day of month after origin.
  Macro signals via `signals_asof(origin)`, live categories only
  (exp70-MS).
- **Reporting:** citation-enforced reporter. Acceptance requires
  `check_citations` AND >= 1 `[url:N]` when articles were provided;
  failures logged (attempt, hash, reasons) and retried with a corrective
  note, max 4 attempts.
- **Relevance:** 20 (explanation, cited article) pairs sampled with a
  fixed seed, judged by a different model family (Groq gpt-oss-120b,
  reasoning low) under a fixed rubric (2 = relevant driver, 1 = partial
  / indirect, 0 = irrelevant). Pass = score >= 1. Rubric and raw
  judgments ship with the results.
- **Leakage audit:** the full run log is audited mechanically (article
  dates vs window/cutoff, signal months, history months); unit tests
  cover the PIT filter, the audit (planted violations), and the
  contract predicate.
- **LLM budget:** generator = Vector flash-lite (exp66c's client),
  cached to `exp87_cache.json`; judge = Groq. All calls counted.

## Pre-registered bar

1. Final citation-contract pass rate = **100%** of flagged explanations.
2. News citations present in **>= 50%** of flagged explanations.
3. **>= 80%** of the 20 sampled news citations topically relevant
   (judge score >= 1).

## Iteration space (researcher rule: diagnose, reiterate 4-5 rounds, no kills)

Query construction (segment-only -> + macro-driver queries), retrieval
window (3 -> 6 months), top_n, category-union retrieval
(`article_categories`), prompt tightening (dated articles, cite-the-
driver instruction), body/summary enrichment for the ~650 articles with
bodies. Each round logs: what changed, which metric moved, what was
learned.

## Results

_(appended as rounds complete)_

## Iteration log

_(appended as rounds complete)_
