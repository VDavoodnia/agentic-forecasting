"""exp76 S8 diagnosis of which macro categories track residual forecast error.

For every Food CPI origin with macro coverage (idx >= 166, corrected
PIT keying: feature month = last observed month), pull the per-category
signals via ``signals_asof`` and correlate (Spearman) each category's
direction / |direction| / certainty / high-risk flag with the native
forecast's residual error magnitude at that origin (mean over horizons
of |y - median| / native 80%-interval half-width, plus the CQR
exceedance score). A category that genuinely carries uncertainty signal
should show |rho| materially above 0 with the error magnitude.

Runs standalone on experiments/exp74_forecasts.json (no battery needed).
Output: printed table + experiments/exp76_macro_diagnosis.json.
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
OUT = HERE / "exp76_macro_diagnosis.json"

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


def ym_of_index(idx: int) -> str:
    y = FOOD_BASE.year + (FOOD_BASE.month - 1 + idx) // 12
    m = (FOOD_BASE.month - 1 + idx) % 12 + 1
    return f"{y:04d}-{m:02d}"


def origin_error_magnitudes() -> dict[int, dict[str, float]]:
    """Per-origin native-forecast residual error magnitude (mean over h)."""
    rows = json.loads(FORECASTS.read_text())["rows"]
    out: dict[int, list[tuple[float, float]]] = {}
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
            out.setdefault(r["origin"], []).append((err_norm, exceed))
    return {
        idx: {
            "err_norm": float(np.mean([e for e, _ in v])),
            "exceed": float(np.mean([x for _, x in v])),
        }
        for idx, v in out.items()
    }


def main() -> None:
    from td_one.news.macro_signal import signals_asof  # noqa: PLC0415

    errs = origin_error_magnitudes()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cats = sorted(
        {r[0] for r in con.execute("select distinct category from macro_signals")}
    )
    per_cat: dict[str, dict[str, list[float]]] = {
        c: {"direction": [], "abs_dir": [], "certainty": [], "high_risk": []}
        for c in cats
    }
    targets: dict[str, list[float]] = {"err_norm": [], "exceed": []}
    origins: list[int] = []
    try:
        for idx in range(166, 198):
            ym = ym_of_index(idx - 1)  # last observed month (PIT-safe)
            sig = signals_asof(con, ym)
            if not sig or idx not in errs:
                continue
            origins.append(idx)
            for t in targets:
                targets[t].append(errs[idx][t])
            for c in cats:
                s = sig.get(c)
                if s is None:
                    d, cert, risk = 0.0, 0.0, "low"
                elif isinstance(s, dict):
                    d = float(s.get("direction", 0.0))
                    cert = float(s.get("certainty", 0.0))
                    risk = s.get("risk", "low")
                else:
                    d = float(getattr(s, "direction", 0.0))
                    cert = float(getattr(s, "certainty", 0.0))
                    risk = getattr(s, "risk", "low")
                per_cat[c]["direction"].append(d)
                per_cat[c]["abs_dir"].append(abs(d))
                per_cat[c]["certainty"].append(cert)
                per_cat[c]["high_risk"].append(1.0 if risk == "high" else 0.0)
    finally:
        con.close()

    print(f"origins: n={len(origins)} ({origins[0]}..{origins[-1]})")
    print(f"{'category':20s} {'live':4s} {'rho|dir|~err':>12s} {'p':>8s} "
          f"{'rho cert~err':>12s} {'p':>8s} {'rho risk~exc':>12s} {'p':>8s}")
    results: dict[str, dict] = {}
    for c in cats:
        row: dict[str, float] = {}
        for feat, tgt in (
            ("abs_dir", "err_norm"),
            ("certainty", "err_norm"),
            ("high_risk", "exceed"),
            ("direction", "err_norm"),
        ):
            x = np.array(per_cat[c][feat])
            y = np.array(targets[tgt])
            if np.std(x) < 1e-12:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = stats.spearmanr(x, y)
            row[f"rho_{feat}_{tgt}"] = float(rho)
            row[f"p_{feat}_{tgt}"] = float(p)
        results[c] = row
        print(
            f"{c:20s} {'YES' if c in LIVE_CATS else '':4s} "
            f"{row['rho_abs_dir_err_norm']:12.3f} {row['p_abs_dir_err_norm']:8.3g} "
            f"{row['rho_certainty_err_norm']:12.3f} {row['p_certainty_err_norm']:8.3g} "
            f"{row['rho_high_risk_exceed']:12.3f} {row['p_high_risk_exceed']:8.3g}"
        )
    OUT.write_text(json.dumps({"origins": origins, "per_category": results}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
