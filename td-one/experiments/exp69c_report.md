# exp69c: Real-Data Anticipatory LLM Shift Adjustment — Report

**Date:** 2026-10-20
**Status:** COMPLETE — honest negative (pre-registered kill triggered; failure diagnosed; variants tested).
**Data:** `experiments/exp69c_results.json` (19 rows, 0 errors). Code:
`experiments/exp69c_real_data.py`, `src/td_one/agent/story_pipeline.py`.
92 real LLM calls (flash-lite), cached in `exp69c_llm_cache.json` (local, not committed).

## Why this exists

exp69 failed on synthetic data — and the *oracle* failed too (+11.4%,
p=0.214), so the mechanism itself was the limit, not the LLM. exp69b
showed broad synthesis *blurs timing* (−5.0%, win 5.6%). The user's
counter-hypothesis: real shocks have richer, uneven, real textual buildup
that a single LLM call on a bag of chunks cannot exploit. exp69c tests
the anticipatory idea **on real data, with better scaffolding**:

1. **Real shock episodes, real series:** the 2022 BoC hiking cycle
   (first hike 2022-03; 100bp super-hike 2022-07), the 2020 COVID shock,
   and the 2021 inflation build — on Food CPI year-over-year inflation
   (2010–2026) and Freddie Mac SMM attrition (2021–2026). News: the real
   `data/news/news_2026.db`, PIT-safe by `pub_date`.
2. **Timing reminders:** the classifier gets explicit timeline anchors —
   "today is {date}, the forecast window is {months}, key dated events:
   ..." (exp69/69b's fragile component, now scaffolded).
3. **News as story:** a chronological narrative (ordered dated events +
   "what's building") instead of a bag of chunks.
4. **Multi-call pipeline:** retrieve (semantic RAG) → rank (LLM, top-12
   with reasons) → story (LLM) → classify (LLM, ShiftSignal). One focused
   job per call.

A sibling agent is building a Macro Signal Service in parallel; the
retrieval function exposes an `extra_articles` seam for its outputs, but
exp69c does not block on it.

## Pre-registered design (locked before running)

**Series & targets.** Food CPI → year-over-year inflation *fraction*
(`value/100`, in [0, 1] so the existing overlay machinery applies
unchanged). SMM → the rate itself. Horizon h=6. Baseline forecaster A =
Chronos-2 zero-shot quantiles (the same A as exp69/exp67).

**Episodes.**

| ID | Series | Onset | Event | Role |
|---|---|---|---|---|
| C1a | Food CPI | 2020-03 | COVID shock | calibration (sparse news: 3 pre-onset articles — expected "none") |
| C1b | Food CPI | 2021-10 | inflation build (YoY 4.6→6.3%) | calibration |
| C2 | SMM | 2022-03 | first BoC hike | calibration |
| T1 | Food CPI | 2022-03 | first hike | test (origins 2021-11…2022-02) |
| T2 | Food CPI | 2022-07 | 100bp super-hike | test (origins 2022-03…2022-06) |
| T3 | SMM | 2024-06 | first cut | test (origins 2024-02…2024-05) |

**Calibration.** Per (series, test origin): classify each calibration
episode with the story pipeline at origin = onset−1 (news only, PIT-safe);
measure `|median(post[onset:onset+6]) − median(pre[onset−6:onset])|`;
table maps `magnitude_bucket → |level change|`, median over episodes.
An episode contributes only if `onset+6 ≤ origin` (realized post window
fully observed — strict PIT). **Absolute value** (new `absolute=True`
flag): the LLM owns *direction*; statistics own *size*. The signed
version would flip an opposite-direction test (e.g. hiking calibration →
cutting test) the wrong way — regression-tested.

**Arms** (per straddle origin):
- **A** = Chronos-2 alone.
- **B** = A + overlay from the multi-call pipeline *with* timeline anchors.
- **B_notl** = A + overlay from the multi-call pipeline *without* timeline
  anchors (ablation: does the timeline earn its keep?).
- **C** = A + overlay from a single-call baseline: top-30 retrieved chunks
  → `AnticipatoryClassifier` (replicates exp69b's approach on real data;
  isolates the scaffolding effect). Same calibration table as B (the arms
  differ only in test-time signal generation — controlled comparison).

**Do-no-harm.** Placebo origins (Food CPI 2018-06/07, SMM 2023-06/07 —
calm windows, no shock in the following 6 months) and post origins
(onset+1 per test episode): expect ΔCRPS = 0 (classifier abstains / the
overlay's own origin ≥ onset guard fires).

**Primary claims & kill criteria.**
1. B beats A on pooled straddle origins (paired t-test on CRPS, p<0.05).
2. B beats C on pooled straddle origins (p<0.05) — the scaffolding must
   earn its keep.
3. Secondary: B vs B_notl (timeline ablation); per-episode B vs A
   (2022 hikes vs 2024 cut may differ — reported honestly).

If (1) or (2) fails: **dead, no iteration.**

**Known limitations (stated upfront).**
- n ≈ 12 straddle rows (3 episodes × 4 origins) — weak power; a null
  result cannot distinguish "no effect" from "underpowered".
- Food CPI tests depend on C1b's classification (C1a's news is too sparse
  to classify); if both yield "none", Food CPI rows no-op by construction.
- SMM's calibrated |Δ| is small (~0.0003) relative to monthly noise
  (~±0.001) — the overlay may be correctly sized yet undetectable.
- Chunks are article *titles* (no bodies for 2022/2024 articles) — the
  story is built from dated headlines.

## Results

**Calibration classifications (the pipeline reads real news well):**
- C1a (2020-03 COVID): `none` — as pre-registered, the 3 pre-onset
  articles carry no signal. Contributes nothing (honest).
- C1b (2021-10 build): `policy/up/medium`, mfn=2, trends: "persistent
  inflationary pressure", "monetary policy tightening expectations".
  |Δ| = 0.031 (YoY fraction).
- C2 (2022-03 first hike): `rate/down/medium`, mfn=1, high confidence,
  trends: "rate hike expected", "mortgage lock-in effect" — direction
  *correct* for SMM (hikes suppress prepayment).

**Primary tests (pooled straddle, n=12):**

| Test | Result |
|---|---|
| **Kill 1: B vs A** | mean ΔCRPS **+23.7% rel., p=0.231, win rate 0%** → **FAIL** |
| **Kill 2: B vs C** | B better by −10.5% rel., **p=0.721**, win 8% → **FAIL** (not significant) |
| Timeline ablation: B vs B_notl | Δ = 0.0000 exactly (see diagnosis §4) |
| Per-episode B vs A | T1: 0.0% (never fired, by construction); T2: **+75.5%** (p=0.274); T3: +8.9% (p=0.391) |
| Do-no-harm: placebo | B Δ −4.6% (p=0.203) PASS; C Δ 0.0 PASS |
| Do-no-harm: post | B +53.4% (p=0.199) PASS by criterion; C +181% (p=0.423) PASS by criterion |

Both kill criteria triggered → the pre-registration says dead. Per the
standing researcher-mindset instruction, the kill is a tripwire for
investigation — the failure anatomy follows, with variants.

## Failure diagnosis (why it failed, component by component)

**1. The forecast gate, not the news-reading, starved the test (5/12 + 1/12 rows).**
T1's four origins all had *empty* calibration tables (C1a = none; C1b's
post window, onset+6 = 2022-04, is after every T1 origin — strict PIT
working as designed). B's signals there were sensible (`policy/up`,
`policy/down`) but could not fire. One more row (T2 2022-05) died on a
*bucket miss*: the LLM said "large", the table only has "medium" — the
exp69 keying gap, one level down. Net: only 6/12 rows could ever fire.

**2. The classifier missed the June 2024 cut at 3 of 4 T3 origins.**
At 2024-03/04/05 the pipeline said `none`. The corpus *did* contain
"Bank of Canada says June rate cut possible if inflation keeps easing"
(2024-04-10), but the ranker's top slots went to five "BoC holds at 5%"
stories, and the story's synthesis extrapolated the hold ("lock-in
effect") instead of the pivot. The single ambiguous cut headline lost
to the louder hold narrative. The one T3 origin that fired (2024-02)
said `rate/down` — wrong direction for SMM (cuts raise prepayment;
realized +0.0015).

**3. Where it fired, size and timing were the failure — not direction.**
3 fired rows: direction right 2/3 (both T2 rows "up", realized up). But:
- **Size overshot ~2.6×**: calibrated |Δ| (C1b) = 0.031 vs realized
  window changes of +0.0145 / +0.0048. Median realized/calibrated ratio
  = **0.39**.
- **Timing ran early**: calibration episodes show a +1.5-month bias
  (LLM predicts onset earlier than realized). At T2 2022-06 the overlay
  shifted YoY *up* from August onward — the exact month inflation peaked
  (11.4%) and turned. CRPS 0.0047 → 0.0221 (4.7× worse) on that row.

**4. Timeline anchors: changed classifications, changed zero forecasts.**
B vs B_notl signals differed on 5/12 rows, but CRPS was bit-identical:
either neither arm fired (gate), or both fired with the same bucket/
direction/onset (the table is bucket-keyed, so type differences don't
propagate). The timeline question is **unresolved by this test, not
refuted** — the anchors' effect never reached a forecast.

**5. The single-call baseline (C) was worse where it mattered.**
At the shared 2022-04 origin, C said `rate/down` (realized: up) and
produced CRPS 0.0465 vs A's 0.0106; B said `up` (right direction, wrong
size). The story pipeline's one measurable edge over single-call is
direction — real, but not significant (B vs C p=0.72).

**6. Post-origin firing is a premise mismatch, reported honestly.**
B fired at all 3 post origins — not a guard violation: the guard blocks
the *same* shock re-firing, and the LLM was anticipating the *next* one
(e.g. T3-post 2024-07 `rate/up`, which helped). Net post effect +53%
rel, n.s. — future designs should score post origins as "next-shift
anticipation", not do-no-harm.

## Variants tested (from stored artifacts, no new LLM calls)

| Variant | Fix attempted | Result |
|---|---|---|
| V_abstain (fire rate < 50% trigger) | "low"-confidence allowed to fire | Identical to B (+23.7%) — abstention was at the *classifier* level, not the overlay; nothing to relax |
| V_timing (cal bias +1.5mo) | shift predicted onset by calibration bias | +15.5% (p=0.29) — better than B, still worse than A |
| V_size (post-hoc scale 0.39) | shrink table by median realized/calibrated ratio | −3.7% (p=0.60), win 8% — sign flips on one big row; not a rescue |
| V_timing+size | both corrections | −4.3% (p=0.53), win 8% — same story |
| V_oracle_dir | not triggered (direction accuracy 0.67 ≥ 0.5) | — |

Even with both diagnosed errors corrected (one of them post-hoc, which
is generous), the overlay does **not** significantly beat Chronos-2
alone. The information the news adds is smaller than the residual
calibration/timing error at n=12 — the same mechanism limit exp69's
oracle exposed on synthetic data, now replicated on real data.

## Verdict

**Real data does not rescue the anticipatory idea in this form.** The
user's hypothesis was tested fairly: the news is rich, the pipeline
reads it well (calibration classifications are genuinely good —
`rate/down` + "mortgage lock-in effect" for the 2022 hikes is exactly
right), and the scaffolding (story, ranking, timeline) measurably
improves the *signal* over single-call (direction 2/3 vs C's reversal).
The failure is downstream of the LLM: a one-episode median size, applied
as a step at an LLM-predicted onset ±1.5 months, is too blunt an
instrument — it converts good classification into bad forecasts.

What would have to change for a future attempt (recorded, not pursued):
(a) many more calibration episodes so the size is a distribution, not a
point — the Macro Signal Service's monthly categorical signals are the
natural source; (b) the overlay should shrink toward zero with timing
uncertainty instead of a confident step; (c) ranking must surface the
*pivot* headline over the *regime* narrative (the T3 miss). Standing
result across exp67→exp69c: **no tested intervention beats Chronos-2
alone on shifted series** — synthetic or real, reactive or anticipatory,
single-call or scaffolded.
