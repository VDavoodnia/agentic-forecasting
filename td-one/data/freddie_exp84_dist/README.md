# exp84 Analysis Population — Distribution Snapshot

> **Status: CURRENT — snapshot of 2026-10-20**

Multipart-zip snapshot of exp84's loan-level analysis population,
preserved in git because the parquets are gitignored build outputs and
this population is the analysis set of record behind exp84's published
era coefficients.

## Contents

- `exp84_panel_data.zip.001.b64` … `.072.b64` — base64 text parts of
  `exp84_panel_data.zip` (each part is the base64 encoding of one
  3 MiB slice of the zip; see `MANIFEST.json` for sha256 of every part
  slice, the archive, and each original file).
- The archive contains (all derived from Freddie Mac SFLLD raw —
  Vandad's shared dataset, class S, never committed):
  - `exp84_panel.parquet` — 8,888,789 loans, full panel.
  - `exp84_frame.parquet` — 7,320,258 FRM 30-yr analysis population.
  - `exp84_s_h12.parquet`, `exp84_s_h24.parquet`, `exp84_s_h36.parquet`
    — horizon samples (H=12/24/36 complete windows).

## Restore

```bash
cd td-one
.venv/bin/python scripts/restore_dist.py data/freddie_exp84_dist --target data/freddie
```

## Rebuild

```bash
.venv/bin/python experiments/exp84_build_panel.py     # needs data/freddie/raw/ (class S)
.venv/bin/python experiments/exp84_make_frame.py
.venv/bin/python experiments/exp84_make_samples.py
```

Intermediate chunks (`data/freddie/exp84_parts/`, 137 MB local) are
regenerable by the same scripts and are intentionally not archived.
See `docs/DATA_REPLICATION.md` for the full asset map.
