# Real-Data Integration Plan

**Date:** 2026-10-20. **Purpose:** our synthetic machinery (SimulatorV2,
red-team shifts, macro signals) is strong but our real data is thin
(SMM = 63 aggregate rows). Three real sources now ground it.

## Acquired sources (Phase 1 — done)

| Source | Contents | Rows | Range | Location |
|---|---|---|---|---|
| OSFI M4 (StatCan 10100006) | Outstanding balances, funds advanced, new-lending rates by segment (insured/uninsured × fixed/variable) + implied runoff | 484 balances / 480 runoff | 2016-07 → 2026-07 (121 months) | `data/osfi/m4_{balances,runoff}.parquet` via `scripts/pull_osfi_m4.py` |
| FRED 10Y yield (IRLTLT01CAM156N) | Canada 10Y gov bond yield | 861 | 1955-01 → 2026-09 | `data/fred/` (already in fetch list — verified) |
| FRED BIS house prices (QCAR628BIS) | Real residential property prices, Canada, quarterly | 225 | 1970 → 2026-Q1 | `data/fred/` (already in fetch list — verified) |
| Competitor rates (Perplexity) | Big-5 posted 5yr fixed / variable + broker discount | ~24 months × ~6 rates | 2024-10 → 2026-09 | `competitor_rates` table in `data/news/news_2026.db` via `scripts/fetch_competitor_rates.py` |

**Implied runoff identity (OSFI):**
`runoff[t] = outstanding[t-1] + funds_advanced[t] − outstanding[t]`
— the balance change not explained by new lending = prepayments +
non-renewed maturities + defaults. Portfolio-level real attrition.
Sanity: 7/480 negative months (restatement noise), mean uninsured runoff
~23.1B CAD/mo, insured ~9.75B CAD/mo. Publication lag ≈ 2 months —
experiments must respect `observed_asof = month + 2` (conservative).

## Integration designs (Phase 2)

### I1 ⭐ OSFI runoff as a real-data benchmark (built first)

**Question:** can ProposedForecasterV2 beat naive on real Canadian
mortgage runoff — the real-data test our synthetic work has never faced?
- **Target:** monthly runoff per segment (4 segments), plus total.
- **Protocol:** rolling origins over the last 60 months, h = 1 and h = 3;
  runoff-rate series (runoff / outstanding[t-1]) to normalize scale drift.
- **Arms:** naive (last value), seasonal-naive, ETS, v2 (router over
  naive/ETS/ARIMA experts; LLM off — monthly portfolio series is not a
  macro-like series for the gate).
- **Metric:** CRPS (primary), MAE, 80% coverage; paired t-test + win rate.
- **Kill criterion:** if v2 does not beat naive at p < 0.10 on total
  runoff h=1, the proposed method has a real-data problem — record it as
  such; do not tune on this target.

### I2 ⭐ Competitor spread as a real refi-pressure covariate (built second)

**Question:** does adding the real competitor spread (broker discount
rate − OSFI new-lending variable rate) improve runoff forecasts vs the
standalone series — i.e., does the DGP's simulated refi-spread mechanism
show up in real data?
- **Covariate:** spread[t], lagged 0–3 months (PIT: rates are as-of
  month t; runoff at t+k only).
- **Protocol:** same origins as I1. Arm A = best I1 model; Arm B = Arm A
  + spread as a routing feature (router sees spread regime) or as a
  post-hoc level adjustment calibrated on training folds only.
- **Metric:** CRPS, paired t-test.
- **Kill criterion:** no improvement at p < 0.10 → the simulated spread
  mechanism does not transfer to aggregate real data (plausible:
  aggregate runoff mixes refi with moves/sales/defaults). Honest negative.

### I3 FRED 10Y + house prices as macro context (deferred)

- 10Y yield already feeds the macro signal categories; house prices are
  quarterly (coarse) — candidate router feature for the Food CPI/SMM
  benchmarks rather than OSFI (monthly mismatch).
- **Deferred** because I1/I2 answer the load-bearing questions first;
  revisit if I1 shows the router responds to real macro features at all.

## PIT rules for all integrations

1. OSFI month M data is `observed_asof` M+2 (publication lag).
2. Competitor rates are as-of month M; use only months ≤ origin − 0.
3. FRED series: use live-client start/end filtering or parquet
   `date <= origin`; vintage revisions are NOT modeled (flagged
   limitation — FRED values are latest-vintage).
4. Run `evaluation/leakage.py` audit on any new data path.

## Competitor-rate coverage (expanded 2026-10-20, exp71b)

The `competitor_rates` table now has three tagged sources (`source`
column) and a monthly coverage matrix generator
(`scripts/coverage_matrix_rates.py`):

| Source | Coverage |
|---|---|
| `boc-valet` — BoC Valet, chartered-bank posted (exact, free) | **622 months**, 1975-01 → 2026-10: 1/3/5yr conventional + prime (`scripts/pull_boc_posted_rates.py`) |
| `wayback-ratehub` — Wayback snapshots of RateHub best-rates page | **54 months** 5yr-fixed best, 44 months 3yr, 54 months variable, 2022-01 → 2026-09 (`scripts/fetch_wayback_rates.py`) |
| `perplexity` — Big-6 posted + broker/monoline (LLM retrieval) | 16 months bank-posted, ~10 broker — sparse; the LLM cannot pin historical rates (`scripts/fetch_competitor_rates.py`, dual phrasing, `--start 2022-01`) |

Remaining honest gaps: 2026-07 has no RateHub snapshot; 2022-12 and
2024-09 broker bests are missing. Pre-2022 broker-discount archives are
too sparse to trust — not fabricated. 56 of the 57 target months
(2022-01 → 2026-09) have some broker/discount data; **54 raw broker-5yr
months** feed the exp71b spread series.

With coverage solved, exp71b re-ran the spread test: **❌ settled
negative** (in-sample r halves to −0.345 on real months; OOS slope
hurts). See `experiments/exp71b_report.md`.
