# Experiment 62 — M4-100 with the news module on

**Date:** 2026-10-18 | **Status:** complete

## Question

Does the news module (exp44 mechanism) change the proposed method's M4-100
result?

## Method

M4-100 records built with the exact exp47 protocol
(`f42.generate_records(f42.load_m4(n=100), 48, 6, 6, "m4")`, 600 records),
calendar origins attached via `monash_common.m4_origin_ym` (handles the
mixed DD-MM-YY / YYYY-MM-DD StartingDate formats in M4-info.csv), then
`run_proposed_cv` with `use_news=True` against the 1979–2017 backfill.

Smoke test (`--no-news`) reproduces exp47 exactly: 0.1451 vs 0.1507,
p = 7.2e-09.

## Results

| | news off | news on | Δ |
|---|---|---|---|
| proposed CRPS | 0.1451 | 0.1450 | ≈0 |
| news fire rate | — | 0.97 | — |

- Proposed vs equal (news on): p = 4.2e-09.
- Proposed vs best single (news on): p = 5.4e-11.

## Verdict

**No material change.** The headline exp47 result is unaffected by the news
module: the gate fires on 97% of origins, but generic macro stress adds
nothing on top of the regime router on M4-100. Third dataset confirming the
pattern: the news mechanism is sound (exp44 synthetic proof), the real-data
signal is negligible.

## Repro

`.venv/bin/python experiments/exp62_m4_news.py` (add `--no-news` for the
exp47-reproduction smoke test). Results: `experiments/exp62_results.json`.
