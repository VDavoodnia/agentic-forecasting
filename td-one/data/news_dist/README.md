# News Corpus Distribution

> **Status: CURRENT — snapshot of 2026-10-20** (supersedes the 2021-era
> snapshot previously described here; those parts are no longer on the
> remote).

Multipart-zip snapshot of TD-One's produced news store, preserved in git
because the store itself is gitignored and exists nowhere else.

## Contents

- `news_store.zip.001.b64` … `.009.b64` — base64 text parts of
  `news_store.zip` (the push channel is text-only, so each part is the
  base64 encoding of one 3 MiB slice of the zip; see `MANIFEST.json`
  for sizes and sha256 of every part slice, the archive, and each
  original file).
- The archive contains:
  - `news_2026.db` (15,368,192 bytes) — SQLite + FTS5 news store,
    **21,234 articles** (2021–2026; GDELT + Perplexity + macro-signal
    backfill), PIT-safe (`pub_date` ≤ fetch date), including the
    `article_bodies` and `macro_signals` tables.
  - `news_embeddings_v1.npz` (22,640,635 bytes) — chunk embeddings for
    the semantic RAG layer (`all-MiniLM-L6-v2`).
  - `gdelt_gaps_remaining.json` — the 227 known GDELT coverage gaps
    still open when the snapshot was taken.

## Restore

```bash
cd td-one
.venv/bin/python scripts/restore_dist.py data/news_dist --target data/news
```

The script decodes and joins the parts, verifies the archive sha256
against `MANIFEST.json`, extracts, and verifies every restored file.
Manual fallback: `for f in news_store.zip.*.b64; do base64 -d "$f";
done > news_store.zip`, check the sha256 against `MANIFEST.json`,
then unzip into `data/news/`.

## Rebuild from scratch (approximate)

The store is the accumulation of every fetcher run, so a rebuild is
not bit-identical; the snapshot above is the copy of record.

```bash
.venv/bin/python scripts/fetch_news_gdelt.py --year 2021        # free; this IP is persistently throttled by GDELT
.venv/bin/python scripts/fetch_news_perplexity.py --year 2021   # paid; needs Perplexity credit via the vault connector
.venv/bin/python scripts/fetch_article_bodies.py                # fills article_bodies via trafilatura
```

See `docs/DATA_REPLICATION.md` for the full asset map and
`docs/RAG_DESIGN.md` for the schema and PIT guarantees.
