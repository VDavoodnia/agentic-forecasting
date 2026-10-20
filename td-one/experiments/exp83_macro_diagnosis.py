"""exp83 per-category macro diagnosis on the stormy window.

Mirror of experiments/exp76_macro_diagnosis.py, re-scoped to Food CPI
origins 133-164 (feature months 2021-01..2023-08, corrected PIT keying)
and extended with the two diagnostics the stormy window needs:

- availability + staleness per category: how many of the 32 origins have
  a signal at all via signals_asof, and how old the freshest signal is
  (the backfill is thin, so asof carry-forward stretches further here
  than in exp76's dense window);
- signed direction vs SIGNED residual: separates "magnitude carries the
  signal" (exp76's finding) from "sign carries the signal".

Output: printed table + experiments/exp83_macro_diagnosis.json.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORECASTS = HERE / "exp74_forecasts.json"
DB_PATH = ROOT / "data" / "news" / "news_2026.db"
OUT = HERE / "exp83_macro_diagnosis.json"

FOOD_BASE = date(2010, 1, 1)
LIVE_CATS = [
    "inflation",
    "services_costs",
    "food_prices",
    "labour",
    "fx",
    "housing",
    "wars_geopolitics",
]
ORIGINS = range(133, 165)


def ym_of_index(idx: int) -> str:
    y = FOOD_BASE.year + (FOOD_BASE.month - 1 + idx) // 12
    m = (FOOD_BASE.month - 1 + idx) % 12 + 1
    return f"{y:04d}-{m:02d}"


def month_diff(a: str, b: str) -> int:
    return (int(a[:4]) - int(b[:4])) * 12 + (int(a[5:7]) - int(b[5:7]))


def origin_errors() -> dict[int, dict[str, float]]:
    """Per-origin native residual magnitude + signed residual (mean over h)."""
    rows = json.loads(FORECASTS.read_text())["rows"]
    out: dict[int, list[tuple[float, float, float]]] = {}
    for r in rows:
        if r["config_id"] != "food_cpi":
            continue
        levels = r["levels"]
        i10, i50, i90 = levels.index(0.1), levels.index(0.5), levels.index(0.9)
        vals, real = r["values"], r["realized"]
        for h in range(len(real)):
            q = [vals[k][h] for k in range(len(levels))]
            half = max((q[i90] - q[i10]) / 2.0, 1e-9)
            err_norm = abs(real[h] - q[i50]) / half
            exceed = max(q[i10] - real[h], real[h] - q[i90], 0.0) / half
            signed = (real[h] - q[i50]) / half
            out.setdefault(r["origin"], []).append((err_norm, exceed, signed))
    return {
        idx: {
            "err_norm": float(np.mean([e for e, _, _ in v])),
            "exceed": float(np.mean([x for _, x, _ in v])),
            "signed": float(np.mean([s for _, _, s in v])),
        }
        for idx, v in out.items()
    }


def main() -> None:
    from td_one.news.macro_signal import signals_asof  # noqa: PLC0415

    errs = origin_errors()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cats = sorted(
        {r[0] for r in con.execute("select distinct category from macro_signals")}
    )
    per_cat: dict[str, dict[str, list[float]]] = {
        c: {"direction": [], "abs_dir": [], "certainty": [], "high_risk": []}
        for c in cats
    }
    avail: dict[str, list[float]] = {c: [] for c in cats}
    stale: dict[str, list[float]] = {c: [] for c in cats}
    targets: dict[str, list[float]] = {"err_norm": [], "exceed": [], "signed": []}
    origins: list[int] = []
    try:
        for idx in ORIGINS:
            ym = ym_of_index(idx - 1)  # last observed month (PIT-safe)
            sig = signals_asof(con, ym)
            if idx not in errs:
                continue
            origins.append(idx)
            for t in targets:
                targets[t].append(errs[idx][t])
            for c in cats:
                s = sig.get(c)
                if s is None:
                    d, cert, risk, av, st = 0.0, 0.0, "low", 0.0, 99.0
                else:
                    d = float(s.direction)
                    cert = float(s.certainty)
                    risk = s.risk
                    av = 1.0
                    st = float(month_diff(ym, s.month))
                per_cat[c]["direction"].append(d)
                per_cat[c]["abs_dir"].append(abs(d))
                per_cat[c]["certainty"].append(cert)
                per_cat[c]["high_risk"].append(1.0 if risk == "high" else 0.0)
                avail[c].append(av)
                stale[c].append(st)
    finally:
        con.close()

    print(f"origins: n={len(origins)} ({origins[0]}..{origins[-1]})")
    print(
        f"{'category':20s} {'live':4s} {'avail':>5s} {'stale_med':>9s} "
        f"{'rho|dir|~err':>12s} {'p':>8s} {'rho dir~signed':>14s} {'p':>8s} "
        f"{'rho cert~err':>12s} {'p':>8s} {'rho risk~exc':>12s} {'p':>8s}"
    )
    results: dict[str, dict] = {}
    for c in cats:
        row: dict[str, float] = {}
        for feat, tgt in (
            ("abs_dir", "err_norm"),
            ("certainty", "err_norm"),
            ("high_risk", "exceed"),
            ("direction", "err_norm"),
            ("direction", "signed"),
        ):
            x = np.array(per_cat[c][feat])
            y = np.array(targets[tgt])
            if np.std(x) < 1e-12:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = stats.spearmanr(x, y)
            row[f"rho_{feat}_{tgt}"] = float(rho)
            row[f"p_{feat}_{tgt}"] = float(p)
        row["avail_rate"] = float(np.mean(avail[c])) if avail[c] else 0.0
        got = [s for s, a in zip(stale[c], avail[c], strict=True) if a > 0]
        row["staleness_median"] = float(np.median(got)) if got else 99.0
        row["staleness_max"] = float(np.max(got)) if got else 99.0
        results[c] = row
        print(
            f"{c:20s} {'YES' if c in LIVE_CATS else '':4s} "
            f"{row['avail_rate']:5.2f} {row['staleness_median']:9.0f} "
            f"{row['rho_abs_dir_err_norm']:12.3f} {row['p_abs_dir_err_norm']:8.3g} "
            f"{row['rho_direction_signed']:14.3f} {row['p_direction_signed']:8.3g} "
            f"{row['rho_certainty_err_norm']:12.3f} {row['p_certainty_err_norm']:8.3g} "
            f"{row['rho_high_risk_exceed']:12.3f} {row['p_high_risk_exceed']:8.3g}"
        )
    OUT.write_text(json.dumps({"origins": origins, "per_category": results}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
