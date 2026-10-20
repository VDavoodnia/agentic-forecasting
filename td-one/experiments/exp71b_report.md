# exp71b: Competitor Spread Re-Test at Full Coverage — Report

**Result: ❌ HONEST NEGATIVE, stronger evidence than exp71.** With 54 raw
broker months (vs 8 in the pilot), the competitor spread still does not
improve runoff forecasts — and the re-test reveals the pilot's in-sample
strength was inflated by forward-filling.

## Coverage (new this round)

| Source | Months | Range | Notes |
|---|---|---|---|
| BoC posted (Valet, exact) | 622 | 1975-01 → 2026-10 | 1/3/5yr conventional + prime; ground truth backbone |
| Wayback RateHub best rates | 54 (5yr), 44 (3yr), 54 (var) | 2022-01 → 2026-09 | monthly snapshots, best advertised broker-channel rate |
| Perplexity Big-6 posted | 16 | sporadic | 8 legacy + 8 new dual-phrasing |
| Perplexity broker/monoline | ~10 | sporadic | mostly zeros — the LLM cannot pin historical rates |

Honest gaps: 2026-07 has no RateHub snapshot; 2022-12, 2024-09 missing.
Pre-2022 broker rates don't exist on comparison sites (RateHub's
page format changes by 2020 are parseable, but broker-discount archives
before 2022 are too sparse to trust — not fabricated).

Broker series: `broker_discount_5yr` = monthly min, Wayback preferred
(exact), Perplexity as supplement. `competitor_rates` table now carries a
`source` tag: `boc-valet` / `wayback-ratehub` / `perplexity`.

## Tests (spread = OSFI uninsured new rate − broker 5yr best; PIT: lag −1)

| Test | Result |
|---|---|
| T1 in-sample r(spread, next-month runoff) | **−0.345, p=0.011, n=53** (pilot: −0.85 on 8 filled months) |
| T2 rolling OOS slope adjustment (spread level) | **+39.3% MSE, p=0.019, win 0.31** — hurts |
| T3 rolling OOS slope adjustment (Δbroker) | +2.6%, p=0.38 — null |
| T4 router feature re-run | Δ = 0.0% — identical forecasts (same architectural limit) |

**Researcher-pass variant:** short-window (12m) slope — **+322% MSE,
p=0.0024**. Slope-stability check: the expanding slope is stable and
always negative (mean −0.0017, std 0.0002, no sign flips), so the failure
is not noise in the fit.

## Diagnosis: the in-sample relationship is a regime artifact

The stable negative slope fits the 2022–23 hiking-cycle lock-in
(wide spread ↔ low runoff) and shrinks forecasts whenever spreads are
wide. In 2024–26 spreads narrowed while runoff stayed low (origination
stock still underwater) — the proportional relationship doesn't transfer.
Aggregate runoff mixes refi with moves/sales/defaults and term-structure
effects; the spread has no stable linear effect at this level.

The pilot's −0.85 came from 8 months forward-filled to 20 — with real
coverage it halves to −0.345. exp71's negative was *not* just a data
problem.

## Verdict

❌ **Honest negative, settled.** The refi-spread mechanism in SimulatorV2
does not transfer to aggregate real runoff data with any tested design
(spread level, spread change, short/long slope, router feature). The
Wayback best-rate series is kept as a data asset (it feeds the Macro
Signal Service and validates posted-rate curves); the covariate line is
closed at this level of aggregation. Revisit only at the *loan-level*
(when Freddie Mac SFLLD is available): the spread effect may exist in
cross-section (which borrowers refi) even if it cancels in aggregate.

Files: `exp71b_spread_rerun.py`, `exp71b_results.json`,
`scripts/pull_boc_posted_rates.py`, `scripts/fetch_wayback_rates.py`,
`scripts/fetch_competitor_rates.py` (dual-phrasing, `--start`),
`scripts/coverage_matrix_rates.py`.
