# exp59 Report — Rung 6: Generalizability on Tourism/M3

**Date:** 2026-10-19
**Status:** Complete. Honest negative.

## Summary

The rung-4 V1 prompt + Gemini Flash, which dominated on Food CPI (0.4262 vs 0.9661 classical), **does not generalize** to Tourism Monthly or M3 Monthly. On both datasets, classical methods beat the LLM decisively.

## Results (h=1, 30 series each)

### Tourism Monthly

| Method | Mean CRPS |
|--------|-----------|
| Classical best | **1030.05** |
| LLM (V1 + Gemini Flash) | 1691.17 |

- LLM win rate: 0.47 (14/30 series)
- Mean diff (LLM - classical): +661.12 (LLM 64% worse)

### M3 Monthly

| Method | Mean CRPS |
|--------|-----------|
| Classical best | **147.20** |
| LLM (V1 + Gemini Flash) | 199.53 |

- LLM win rate: 0.20 (6/30 series)
- Mean diff (LLM - classical): +52.33 (LLM 36% worse)

## Interpretation

**Why the LLM fails here:** Tourism and M3 are multi-series benchmarks with strong, regular seasonality. The classical methods (especially seasonal naive) exploit this structure directly. The LLM, given only a single series history and a generic prompt, cannot match the precision of a seasonal naive forecast on strongly seasonal data.

**Contrast with Food CPI:** Food CPI is a single smooth macro series where the LLM's semantic context ("Canada Food CPI") and calibration skill matter. Tourism/M3 series are diverse, seasonal, and lack the semantic richness that helps the LLM.

**The Food CPI win was dataset-specific.** The LLM integration does not generalize to multi-series seasonal benchmarks. This is an honest negative for rung 6.

## Implications for the ladder

1. The hybrid (exp58) beats Vector's method on Food CPI, but this does not transfer.
2. For Tourism/M3, the proposed method (classical router) remains the best approach.
3. The LLM is a specialist tool for smooth, context-rich series — not a general forecaster.

## Success criteria

- [ ] LLM/hybrid wins on Tourism/M3 — **FAILED**
- [x] Honest reporting of the negative result

## Conclusion

Rung 6 is a clean negative. The LLM integration is valuable for Food CPI-like series but does not generalize to seasonal multi-series benchmarks. The proposed classical method stands on Tourism/M3.
