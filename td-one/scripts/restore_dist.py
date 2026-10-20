#!/usr/bin/env python3
"""Restore a TD-One multipart-zip distribution archive.

Each ``td-one/data/<group>_dist/`` directory holds a large produced
artifact as a zip split into numbered parts. Because the git push
channel is text-only, each part file (``<archive>.zip.001.b64``, ...)
is the base64 encoding of one 3 MiB slice of the zip. A
``MANIFEST.json`` records the sha256 of every original file, of the
reassembled zip, and of each binary part slice.

Usage:
    cd td-one
    .venv/bin/python scripts/restore_dist.py data/news_dist --target data/news
    .venv/bin/python scripts/restore_dist.py data/freddie_exp84_dist --target data/freddie
    .venv/bin/python scripts/restore_dist.py data/scale_up_dist --target data/scale_up

The restore decodes and joins the parts in manifest order, verifies
the zip's sha256, extracts it into the target directory, and verifies
every extracted file against the manifest. Any mismatch aborts with a
nonzero exit code and nothing is left half-verified silently.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Join, verify, and extract one dist archive. Returns exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="Directory holding MANIFEST.json + parts")
    parser.add_argument("--target", type=Path, required=True, help="Directory to extract into")
    args = parser.parse_args(argv)

    manifest_path = args.dist_dir / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"ERROR: no MANIFEST.json in {args.dist_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    archive_name = manifest["archive"]["name"]
    joined = args.dist_dir / archive_name
    print(f"Decoding and joining {len(manifest['parts'])} parts -> {joined}")
    with joined.open("wb") as out:
        for part in manifest["parts"]:
            part_path = args.dist_dir / part["name"]
            if not part_path.exists():
                print(f"ERROR: missing part {part_path}", file=sys.stderr)
                return 2
            raw = base64.b64decode(part_path.read_bytes())
            if len(raw) != part["binary_size"] or hashlib.sha256(raw).hexdigest() != part["binary_sha256"]:
                print(f"ERROR: part failed verification after base64 decode: {part['name']}", file=sys.stderr)
                return 1
            out.write(raw)

    if joined.stat().st_size != manifest["archive"]["size"]:
        print("ERROR: joined archive size mismatch vs MANIFEST", file=sys.stderr)
        return 1
    digest = sha256_of(joined)
    if digest != manifest["archive"]["sha256"]:
        print("ERROR: joined archive sha256 mismatch vs MANIFEST", file=sys.stderr)
        return 1
    print(f"Archive verified: {archive_name} sha256={digest[:16]}...")

    args.target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(joined) as zf:
        zf.extractall(args.target)

    ok = True
    for entry in manifest["files"]:
        extracted = args.target / entry["name"]
        if not extracted.exists():
            print(f"ERROR: expected file missing after extract: {entry['name']}", file=sys.stderr)
            ok = False
            continue
        if extracted.stat().st_size != entry["size"] or sha256_of(extracted) != entry["sha256"]:
            print(f"ERROR: extracted file failed verification: {entry['name']}", file=sys.stderr)
            ok = False
        else:
            print(f"  verified {entry['name']} ({entry['size']} bytes)")
    joined.unlink()
    print("Restore complete." if ok else "Restore finished WITH ERRORS.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
