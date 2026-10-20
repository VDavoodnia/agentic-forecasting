# exp83 — Macro signals on a stormy window: the exp76 S8 question, answered

> **Status: COMPLETE — 2026-10-20. Recovery run** (the original exp83 agent
> died in a daemon restart mid-fetch; this run inventoried and resumed its
> partial work — see §1).

## Verdict

**✅ The pre-registered bar is met — twice, in two independent arm
families.** exp76's S8 ablation was inconclusive because its only overlap
window (origins 166–192, feature months 2023-10+) was signal-quiet: even
the story effect vanished there. exp76's own carry-forward encoding —
**story features + the standalone inflation macro feature** — was named
the design to retest on a volatile window. With the Macro Signal store
backfilled into 2021-01→2023-09 and the identical protocol rerun on Food
CPI origins 133–164 (the inflation shock), the designated arm wins:

| Family | Comparison (stormy clean scope, 32 origins, 192 cases) | Δ CRPS | Origin-cluster bootstrap 95% CI | Paired t p |
|---|---|---|---|---|
| meta-on-native | `meta_story_inf` vs `meta_story` | **−1.10%** | **[−2.09%, −0.25%]** | 1.8e-04 |
| F1 (shipped design) | `f1_story_inf` vs `f1_story` | **−0.55%** | **[−0.97%, −0.18%]** | 2.3e-04 |
| F1 h3-gated (champion config) | `f1_h3_story_inf` vs `f1_h3_story` | **−0.53%** | **[−0.91%, −0.20%]** | 1.1e-04 |

The best overall configuration on the stormy window is the champion with
the inflation feature added: **CRPS 2.2179, cov80 0.854** (native
2.3845 / 0.693; plain CQR 2.3142 / 0.755; champion without macro
2.2297 / 0.849). The quiet-window control (origins 166–192, the exp76
scope) shows **no macro effect** (+0.15% meta family, +0.40% F1 family,
both n.s.) — the gain is specific to volatile origins, exactly the
pattern exp76's diagnosis predicted. **Ship decision:** the F1 h3-gated
champion's feature set becomes story + `mac_inflation` wherever macro
signal coverage exists (recorded in `src/td_one/uq/README.md`). The
encoding is encoding-free, in one sense: raw signed inflation direction
reproduces the win exactly (−1.10%, CI [−2.06%, −0.25%]) — the active
ingredient is inflation's direction itself.

What did **not** work, with equal prominence: macro aggregates over all
7 live categories (−0.63%, CI includes zero — dilution again), per-category
|direction| features (**+0.38%**, harmful direction), macro-without-story
(−0.04%, nothing), and the diagnosis's risk-flag features (round 6:
+0.01–0.08%, inert — §6).

## 1. Recovery log (what the dead agent left, and what this run did)

| Item | State found | Action |
|---|---|---|
| Fetch (Perplexity) | 121/231 cells logged in DB; `exp83_fetch_log.json` at 188 searches, est. **$1.80** | Resumed with the same script and cap (300 searches); died at the cap mid-cell with an unhandled `RuntimeError` (the in-cell retry search tripped the driver's own cap check — that cell stayed unlogged and was simply never extracted). Final: **300 searches, est. $2.88 total for the whole experiment**, 193/231 cells fetched |
| Extraction | Only 1 signal in-window (from the existing-corpus pass; the categorized corpus was effectively empty for 2021–2023 before fetching) | Two resume passes over fetched cells (112 + 61 Vector-proxy calls): **174 signals** in-window |
| Battery | Written (`exp83_stormy_battery.py`) but never run; no results | Run by this agent — see the reboot saga below |
| Diagnosis | Written, never run | Run first (§4) |

The battery itself survived two VM reboots badly: the monolithic run
writes results only at the end, and both attempts (one at ~1000 CPU-sec)
were lost. This run therefore executed the identical protocol through a
**checkpointed per-arm runner** (`experiments/exp83_battery_resume.py`):
`run_study` calls each arm independently per case with the same PIT
bank, so per-arm execution + concatenation is exactly equivalent to the
single 11-arm call, and each arm's case-results were checkpointed to
disk on completion. Mid-incident, a duplicate runner briefly duplicated
one arm (identical deterministic output; the duplicate was killed). The
quiet-scope control reproduces exp76's published story-vs-numeric figure
exactly (−0.2%, p=0.83 in both runs), which cross-validates the runner.

## 2. Backfill coverage (2021-01 → 2023-09, live-7 categories)

Spend discipline: inflation first (all 33 months — exp76's one live
wire), then the other live categories ordered by Food CPI month
volatility. 727 categorized article-links landed in-window.

| Category | Cells fetched | Signals | Notes |
|---|---|---|---|
| inflation | 33/33 | **33** | complete |
| fx | 27/33 | 26 | |
| wars_geopolitics | 26/33 | 26 | |
| services_costs | 27/33 | 25 | |
| housing | 26/33 | 25 | |
| food_prices | 27/33 | 20 | thinnest — several fetched cells returned zero in-month articles |
| labour | 27/33 | 19 | |
| **Total** | **193/231** | **174** | missing-signal months cluster at 2021-01..03 and 2023-03/06/09 |

Per-origin availability through `signals_asof` at the 32 stormy origins
(§4): inflation 1.00, all other live categories 0.84–0.91, median
staleness 0 months — carry-forward fills the gaps, so the battery's
feature matrix is dense where it matters.

## 3. Protocol (identical to exp76 S8, new window)

Same cases (`exp74_forecasts.json`, Food CPI), same corrected PIT keying
(feature month = last observed month), same MetaArm GBM, same feature
encodings imported from `exp76_fusion_battery.py` (not forked). Arms:
native, CQR, meta_numeric, meta_story (baseline), meta_full_live
(+ live-7 aggregates), meta_story_inf (+ `mac_inflation` standalone),
meta_macro_live (macro only), + per-category signed / |direction| /
interaction / staleness-gated variants. Clean scope: origins 133–164
(feature months 2021-01..2023-08), where story features exist for all
origins and macro is keyed PIT-safely. Bar (pre-registered): a macro arm
beats meta_story on CRPS with the origin-cluster bootstrap 95% CI
excluding zero.

## 4. Round 2 output first: per-category diagnosis on the stormy window

32 origins, native residuals from `exp83_macro_diagnosis.py`
(Spearman ρ, uncorrected p; ~35 pairs tested, so read p≈0.01–0.05 as
suggestive, p≤0.002 as solid):

| Category | avail | ρ \|dir\| ~ err | ρ dir ~ **signed** resid | ρ cert ~ err | ρ high-risk ~ exceed |
|---|---|---|---|---|---|
| inflation | 1.00 | +0.286 (p=0.113) | **+0.520 (p=0.0023)** | +0.137 | −0.142 |
| housing | 0.91 | −0.345 (p=0.053) | **+0.783 (p=1.2e-07)** | −0.172 | −0.217 |
| fx | 0.91 | −0.174 | **+0.439 (p=0.012)** | −0.153 | −0.166 |
| food_prices | 0.84 | −0.247 | −0.390 (p=0.027) | −0.317 (p=0.077) | **−0.527 (p=0.0019)** |
| wars_geopolitics | 0.91 | +0.013 | +0.046 | −0.261 | **−0.754 (p=6.3e-07)** |
| labour | 0.91 | −0.057 | +0.136 | −0.093 | const |
| services_costs | 0.91 | −0.005 | −0.039 | +0.004 | +0.025 |

Two structural differences from exp76's quiet window: (1) on stormy
origins the **sign** of direction carries more signal than its magnitude
(inflation's |dir| wire weakens to +0.286 n.s. while its signed wire is
+0.520); (2) housing's signed direction is the single strongest wire in
either window (+0.783). The risk-flag anti-correlations (wars −0.754)
say the TSFM's own intervals already price flagged-risk regimes —
tested as features in round 6, inert there (§6).

## 5. Round 1: the pre-registered battery (meta-on-native family)

Paired CRPS vs meta_story, stormy clean scope (192 cases):

| Arm | Δ | bootstrap 95% CI | t p | win |
|---|---|---|---|---|
| **meta_story_inf** | **−1.10%** | **[−2.09%, −0.25%]** | 1.8e-04 | 0.47 |
| meta_story_persign (per-cat signed) | −0.82% | [−1.86%, +0.10%] | 0.0079 | 0.40 |
| meta_full_live (live-7 aggregates) | −0.63% | [−1.62%, +0.33%] | 0.027 | 0.41 |
| meta_story_gated (≤3mo staleness) | −0.63% | [−1.65%, +0.27%] | 0.027 | 0.41 |
| meta_macro_live (no story) | −0.04% | [−1.32%, +1.43%] | 0.93 | 0.40 |
| meta_story_inter (\|inf\| × story mag) | +0.01% | [−0.20%, +0.23%] | 0.92 | 0.24 |
| meta_story_perabs (per-cat \|dir\|) | **+0.38%** | [−0.14%, +1.09%] | 0.029 | 0.29 |
| meta_numeric (no story, no macro) | +3.78% | [+1.87%, +5.72%] | 2.0e-07 | 0.24 |

Per-year for the designated arm (meta_story_inf): 2021 −0.54%,
2022 −0.93%, 2023 −5.43% — same sign all three years, growing with the
window's violence (2023 is only 8 origins; its CI is wide). The gated
arm is numerically identical to full_live: no in-scope signal was stale
enough for the gate to bind. Absolute levels on the stormy scope:
native CRPS 2.3845 (cov80 0.693 — the TSFM badly under-covers the
shock), CQR 2.3142 (0.755), meta_story 2.3757 (0.625), meta_story_inf
2.3496 (0.615). The meta family wins CRPS *within itself* but still
trails plain CQR and under-covers — which is why round 3 matters.

**Quiet control (origins 166–192):** meta_story_inf +0.15% (p=0.86),
meta_full_live −0.40% (p=0.66) — nothing clears, replicating exp76's
verdict on its own scope. (Point estimates on individual null arms
differ from exp76's table by up to ~0.5pp between the two runs' GBM
fits; the shared anchor — story vs numeric — matches exactly.)

## 6. Rounds 3–6: transfer, ablations, splits, risk flags

**Round 3 — does macro improve the shipped F1 design?** (§0 table:
yes, −0.55% / −0.53%, both CIs excluding zero; quiet scope +0.40% /
+0.39%, n.s.) F1-family absolutes on stormy: f1_story 2.2348 (cov80
0.854), f1_story_inf 2.2226 (0.865), f1_h3_story 2.2297 (0.849),
**f1_h3_story_inf 2.2179 (0.854)** — the only family that holds
near-nominal coverage on stormy origins, and macro improves it.
Caveat on cross-family comparisons: the F1-vs-meta gap flips sign by
year (2021: F1 −17.3%; 2023: meta −38% on 8 origins) — pooled, F1 wins
CRPS and coverage, but neither family dominates every sub-window. All
macro conclusions above are within-family paired and unaffected.

**Round 4 — ablations.** Raw signed inflation direction as the single
added feature (`mpl_inflation`) reproduces the designated arm exactly:
−1.10%, CI [−2.06%, −0.25%], same t p — the effect is the direction,
not the `mac_inflation` encoding's details. Housing signed direction
alone: −0.24% pooled (CI includes zero), but **−0.62% [−1.22%, −0.27%]
in 2022** — the diagnosis's strongest wire fires in the rate-shock year
and is silent in 2021 (+0.13%). Exploratory; a candidate feature for a
future round, not part of the ship decision.

**Round 5 — splits.** The win is a long-horizon phenomenon in both
families, mirroring exp76's F1 lesson: meta family h≥3 −1.22%
[−2.46%, −0.26%] vs h≤2 −0.16% n.s.; F1-h3 family h≥3 −0.62%
[−1.04%, −0.23%] and exactly 0.00 at h≤2 (the gate working as built).
The story-magnitude split is degenerate in this window (30/32 origins
sit in the low half — story magnitude is near zero at most stormy
origins), and the low half alone still shows the full effect (−1.14%
[−2.28%, −0.24%]): the macro win is **not** confined to origins where
the story features are loud. The F1 win rates look paradoxical (0.15)
until the mechanism is stated: at most origins the inflation direction
is 0 or the modulation doesn't move, so most cases are *identical*;
the CRPS gain concentrates in the minority of cases where a strong
inflation signal coincides with a mispriced interval.

**Round 6 — risk-flag features (diagnosis-driven, honest negative).**
Adding the wars/food high-risk flags (the two anti-correlated wires of
§4) to the winning arms changes nothing: meta family +0.08%
[−0.04%, +0.24%], F1 family +0.01% [−0.02%, +0.03%]; in 2021 the arms
are bit-identical to their no-flag counterparts. Mechanism: the flags
are rare, collinear with the |direction| features already present, and
their univariate correlation is with the *native* exceedance the meta
model already conditions away. The §4 anti-correlation remains a true
statement about the TSFM (its intervals price flagged risk), not a
usable meta feature. Recorded, not pursued further this round.

## 7. Iteration log

| Round | What was done | Lesson |
|---|---|---|
| 0 | Recovery: inventoried the dead agent's fetch/extract state, resumed within the original $3 cap, rebuilt the battery as a checkpointed per-arm runner after two reboot losses | Resume state must be on disk per arm, not per run, on this VM; the local corpus held ~nothing for 2021–2023 — the backfill had to be fetched, at $2.88 total |
| 1 | Pre-registered 11-arm battery on the stormy clean scope | **Bar met** by the designated arm (−1.10%, CI excl. 0). Aggregates dilute (−0.63% n.s.); per-category \|dir\| harms (+0.38%); macro without story is nothing (−0.04%) |
| 2 | Per-category diagnosis (availability, staleness, signed vs magnitude wires) | Stormy windows flip the wire structure: **sign** > magnitude (inflation +0.520 signed); housing signed is the strongest single wire (+0.783); risk flags anti-track exceedance |
| 3 | F1-family transfer (MetaLevelCqr ± inflation, ± h3 gate) | Macro improves the shipped design too (−0.55% / −0.53%, CIs excl. 0); F1+inf is the best stormy configuration overall (2.2179, cov80 0.854); quiet scope inert (+0.4% n.s.) |
| 4 | Encoding + category ablations | Raw signed direction ≡ the `mac_inflation` encoding (−1.10% both) — the ingredient is inflation's direction. Housing is a 2022-only wire (−0.62%, CI excl. 0, that year only) |
| 5 | Horizon / year / story-magnitude splits | Win concentrates at h≥3 in both families (mirrors exp76); sign-consistent across all 3 years; not confined to loud-story origins |
| 6 | Risk-flag features on both winning arms | ❌ inert (+0.01–0.08%): rare, collinear, and already conditioned away. Diagnostic truth about the TSFM, not a feature |

## 8. Verdict and what ships

exp76 left the macro question as *"untested, not refuted"* — the quiet
window could not express the effect. exp83 removes the confound:
**backfilled to the inflation shock, the designated macro feature
improves the uncertainty layer on volatile origins, in both arm
families, with cluster-bootstrap CIs excluding zero, sign-consistent
across three years, and a same-run quiet control showing no effect.**
Effect size is modest (−0.5 to −1.1% CRPS) and concentrated in few
high-leverage cases — this is a real incremental feature, not a
breakthrough, and it is reported as such.

- **Ships:** F1 h3-gated champion feature set = story + `mac_inflation`
  where macro coverage exists (uq README updated). On calm origins the
  feature is inert-to-noise (+0.4% n.s.), so there is no regime gate to
  maintain — the GBM sees direction 0 and does nothing.
- **Store asset:** `macro_signals` now spans 2021-01→2026-09 for the
  live-7 (899 rows total). Any future backtest over the inflation shock
  can consume it PIT-safely via `signals_asof` at zero marginal cost.
- **Stays flagged/exploratory:** housing signed direction (2022-only
  evidence), per-category signed sets (persign −0.82%, CI includes
  zero). **Closed with mechanism:** per-category |direction| features,
  staleness gating (never binds in-scope), risk-flag features.
- **Open follow-up (not run):** replication on a second stormy series —
  the sweep families (Tourism/M3/M4/OSFI/SMM) lack story features, so
  this needs a story pipeline for one of them or a second long-bank
  macro series; and a 2024–2026 holdout once a fresh volatile window
  exists (the current store's later months are the quiet control, not
  a holdout).

## 9. Files and reproducibility

- Backfill: `experiments/exp83_fetch_backfill.py` (resume-safe; log:
  `experiments/exp83_fetch_log.json` — 300 searches, est. $2.88),
  `experiments/exp83_extract_backfill.py` (173 recovery-pass Vector
  calls; signals live in the local news DB, never committed).
- Diagnosis: `experiments/exp83_macro_diagnosis.py` →
  `experiments/exp83_macro_diagnosis.json`.
- Round 1: `experiments/exp83_stormy_battery.py` (protocol) executed via
  `experiments/exp83_battery_resume.py` (checkpointed runner) →
  `experiments/exp83_stormy_results.json` (7,326 case-results).
- Round 2: `experiments/exp83_round2.py` → `exp83_round2_results.json`.
- Round 3: `experiments/exp83_round3.py` → `exp83_round3_results.json`.
- Per-arm checkpoints (`experiments/exp83_checkpoints*/`) are local
  resume state; the results JSONs above contain every case result.
