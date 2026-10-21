#!/usr/bin/env python3
"""Download the Monash / M4 public benchmark datasets into ``data/``.

TD-One's public-benchmark results (exp47, exp59–exp65, exp76) run on
three public datasets whose files are gitignored. This script restores
the exact local layout the loaders expect:

- Tourism Monthly (Monash Time Series Forecasting Archive):
  ``data/tourism/tourism_monthly_dataset.zip`` + extracted
  ``tourism_monthly_dataset.tsf`` (366 series, h=24; exp60).
  Source: Zenodo record 4656096.
- M3 Monthly (Monash archive): ``data/m3mon/m3_monthly_dataset.zip`` +
  extracted ``m3_monthly_dataset.tsf`` (1,428 series, h=18; exp61).
  Source: Zenodo record 4656298.
- M4 Monthly (M4 competition): ``data/m4/Monthly-train.csv``,
  ``Monthly-test.csv``, ``M4-info.csv`` (flat layout; exp47, exp62).
  Source: the official Mcompetitions/M4-methods GitHub repository.

Every file's expected byte size (verified against the canonical URLs
on 2026-10-20) is pinned below; a download whose size differs is
rejected. The script is idempotent: files already present with the
right size are skipped unless ``--force`` is given.

Usage:
    cd td-one
    .venv/bin/python scripts/fetch_monash_benchmarks.py [--only tourism|m3|m4] [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import TypedDict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USER_AGENT = "td-one-fetch-monash/1.0"


class RemoteFile(TypedDict):
    """One downloadable file and where it lands."""

    url: str
    dest: Path
    size: int


class Dataset(TypedDict):
    """A benchmark dataset: files to fetch, plus a zip to unpack."""

    files: list[RemoteFile]
    zip_to_extract: Path | None
    extract_names: list[str]


DATASETS: dict[str, Dataset] = {
    "tourism": {
        "files": [
            {
                "url": "https://zenodo.org/records/4656096/files/tourism_monthly_dataset.zip",
                "dest": DATA_DIR / "tourism" / "tourism_monthly_dataset.zip",
                "size": 199_791,
            }
        ],
        "zip_to_extract": DATA_DIR / "tourism" / "tourism_monthly_dataset.zip",
        "extract_names": ["tourism_monthly_dataset.tsf"],
    },
    "m3": {
        "files": [
            {
                "url": "https://zenodo.org/records/4656298/files/m3_monthly_dataset.zip",
                "dest": DATA_DIR / "m3mon" / "m3_monthly_dataset.zip",
                "size": 336_324,
            }
        ],
        "zip_to_extract": DATA_DIR / "m3mon" / "m3_monthly_dataset.zip",
        "extract_names": ["m3_monthly_dataset.tsf"],
    },
    "m4": {
        "files": [
            {
                "url": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
                "dest": DATA_DIR / "m4" / "Monthly-train.csv",
                "size": 91_655_432,
            },
            {
                "url": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv",
                "dest": DATA_DIR / "m4" / "Monthly-test.csv",
                "size": 7_942_698,
            },
            {
                "url": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv",
                "dest": DATA_DIR / "m4" / "M4-info.csv",
                "size": 4_335_598,
            },
        ],
        "zip_to_extract": None,
        "extract_names": [],
    },
}


def download(url: str, dest: Path, expected_size: int) -> None:
    """Download ``url`` to ``dest`` (via a temp file), verifying size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1024 * 1024):
            fh.write(chunk)
    actual = tmp.stat().st_size
    if actual != expected_size:
        tmp.unlink()
        raise RuntimeError(f"{url}: downloaded {actual} bytes, expected {expected_size}")
    tmp.rename(dest)
    print(f"  downloaded {dest} ({actual} bytes)")


def fetch_dataset(name: str, spec: Dataset, force: bool) -> None:
    """Fetch one dataset's files and unpack its zip if it has one."""
    print(f"[{name}]")
    for remote in spec["files"]:
        dest = remote["dest"]
        if dest.exists() and dest.stat().st_size == remote["size"] and not force:
            print(f"  present, skipped: {dest} ({remote['size']} bytes)")
            continue
        download(remote["url"], dest, remote["size"])
    zip_path = spec["zip_to_extract"]
    if zip_path is not None:
        for member in spec["extract_names"]:
            target = zip_path.parent / member
            if target.exists() and not force:
                print(f"  present, skipped: {target}")
                continue
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(member, zip_path.parent)
            print(f"  extracted {target} ({target.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    """Fetch the selected Monash/M4 datasets. Returns exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args(argv)

    names = list(DATASETS) if args.only == "all" else [args.only]
    for name in names:
        fetch_dataset(name, DATASETS[name], args.force)
    print("Monash/M4 benchmark data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
