"""exp71b: re-test the competitor spread with expanded coverage.

exp71 (pilot) found: strong in-sample spread/runoff correlation (r=-0.85)
that collapsed out-of-sample on 8 raw broker months. Coverage has since
expanded: exact Wayback best rates (RateHub, 2022-01->), dual-phrasing
Perplexity pulls, and the BoC posted backbone.

Designs (PIT: signals for origin month M use rates as-of M-1):
  T1  in-sample correlation: spread vs next-month total runoff (expanded).
  T2  rolling OOS slope adjustment on naive with spread LEVEL.
  T3  rolling OOS slope adjustment on naive with spread CHANGE (Delta
      broker 5yr) — the variant exp71's report suggested.
  T4  router feature re-run: v2 + spread vs v2 (same as exp71).

Slopes in T2/T3 are fit on training history only (expanding window), so
no look-ahead. Spread = OSFI uninsured new rate - broker 5yr best.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp70_osfi_benchmark as e70  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402

MIN_TRAIN = 24  # months of history before the first OOS origin


def load_series() -> pd.DataFrame:
    """Monthly broker-5yr (wayback preferred), OSFI uninsured rate, BoC 5yr."""
    con = sqlite3.connect(ROOT / "data" / "news" / "news_2026.db")
    cr = pd.read_sql(
        "SELECT month, product, rate_pct, source FROM competitor_rates", con
    )
    con.close()
    cr["month"] = pd.to_datetime(cr["month"])
    pref = {"wayback-ratehub": 0, "perplexity": 1}
    b5 = cr[cr["product"] == "broker_discount_5yr"].copy()
    b5["prio"] = b5["source"].map(pref).fillna(2)
    broker = b5.sort_values("prio").drop_duplicates("month").set_index("month")[
        "rate_pct"].sort_index()
    boc = (cr[(cr["source"] == "boc-valet")
               & (cr["product"] == "posted_5yr_conventional")]
           .drop_duplicates("month").set_index("month")["rate_pct"].sort_index())
    bal = pd.read_parquet(ROOT / "data" / "osfi" / "m4_balances.parquet")
    osfi = (bal[bal["segment"] == "uninsured"].set_index("month")
            .sort_index()["rate_pct"])
    osfi.index = pd.to_datetime(osfi.index)
    idx = pd.date_range(min(broker.index.min(), osfi.index.min()),
                        max(broker.index.max(), osfi.index.max()), freq="MS")
    broker = broker.reindex(idx)
    osfi_r = osfi.reindex(idx).ffill()
    spread = (osfi_r - broker)
    d_broker = broker.diff()
    raw = broker.notna()
    return pd.DataFrame({
        "broker": broker, "spread": spread, "d_broker": d_broker,
        "osfi_rate": osfi_r, "boc5": boc.reindex(idx).ffill(),
        "broker_raw": raw,
    })


def rolling_slope_oos(values: np.ndarray, sig_lag: np.ndarray,
                       min_train: int = MIN_TRAIN) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Naive vs naive + slope*sig on raw-signal months only.

    ``sig_lag[t]`` is the signal known at month t (as of t-1). Forecast
    ``values[t+1] = naive + slope * sig_lag[t]``; slope refit each origin
    on pairs (sig_lag[s], values[s+1]) with s < t (train-only, PIT).
    Origins with no raw signal are skipped, never forward-filled.
    """
    err_a, err_b = [], []
    n = len(values)
    for t in range(min_train, n - 1):
        xs, ys = [], []
        for s in range(t):
            if not np.isnan(sig_lag[s]):
                xs.append(sig_lag[s])
                ys.append(values[s + 1])
        if len(xs) < 12 or np.isnan(sig_lag[t]):
            continue
        slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
        base = values[t]
        adj = base + slope * float(sig_lag[t])
        err_a.append(values[t + 1] - base)
        err_b.append(values[t + 1] - adj)
    return np.asarray(err_a), np.asarray(err_b)


def _paired(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    ma, mb = float(np.mean(a ** 2)), float(np.mean(b ** 2))
    t, p = ttest_rel(a ** 2, b ** 2)
    return {
        "n": len(a),
        "mse_A_naive": ma,
        "mse_B_adj": mb,
        "diff_pct": 100 * (mb - ma) / ma,
        "p": float(p),
        "win_rate_B": float(np.mean(np.abs(b) < np.abs(a))),
    }


def router_feature_test(values: np.ndarray, months: pd.DatetimeIndex,
                        spread: pd.Series) -> dict:
    """exp71 T4 re-run: v2 + spread feature vs v2."""
    fn = lfc.feature_names_for(values)
    base, aug = [], []
    for o in range(36, len(values)):
        train = values[o - 36:o]
        actual = float(values[o])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = series_features(train, period=12)
            qm = lfc.origin_quantiles(train)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        vec = [float(fd.get(f, 0.0)) for f in fn]
        sp = spread.reindex([months[o] - pd.offsets.MonthBegin(1)])
        spv = float(sp.iloc[0]) if sp.notna().any() else np.nan
        rec = {
            "origin": o,
            "quantiles": {m: q.tolist() for m, q in qm.items()},
            "crps": {m: grid_to_crps(q[:, 0], LEVELS_9, actual)
                     for m, q in qm.items()},
            "actual": [actual],
        }
        base.append({**rec, "features": vec})
        aug.append({**rec, "features": [*vec, spv]})
    aug = [r for r in aug if not np.isnan(r["features"][-1])]
    base = [b for b in base if b["origin"] in {r["origin"] for r in aug}]

    def factory() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(h=1, use_llm_expert=False)
        fc.models = list(lfc.CLASSICAL_MODELS)
        return fc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = blocked_cv_predict(base, fn, factory, n_folds=5,
                                baseline="best_single")
        rb = blocked_cv_predict(aug, [*fn, "spread_lag1"], factory,
                                n_folds=5, baseline="best_single")
    a = np.asarray(ra["crps"])
    b = np.asarray(rb["crps"])
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    t, p = ttest_rel(a, b)
    return {
        "n": int(len(a)),
        "crps_A": float(np.mean(a)),
        "crps_B": float(np.mean(b)),
        "diff_pct": float(100 * (np.mean(b) - np.mean(a)) / np.mean(a)),
        "p": float(p),
    }


def main() -> None:
    df = load_series()
    runoff = e70.load_runoff_rates()["total"]
    common = df.index.intersection(runoff.index)
    df, runoff = df.loc[common], runoff.loc[common]

    # Restrict the analysis window to months with a raw broker observation.
    df = df[df["broker_raw"]].copy()
    runoff = runoff.reindex(df.index).dropna()
    df = df.reindex(runoff.index)
    n_raw = len(df)
    print(f"[exp71b] broker months raw: {n_raw} "
          f"({df.index.min().date()} -> {df.index.max().date()})")

    # T1: in-sample correlation on the real-coverage window.
    sub = pd.DataFrame({"spread": df["spread"],
                        "next": runoff.shift(-1)}).dropna()
    r, p = pearsonr(sub["spread"], sub["next"])
    print(f"[exp71b] T1 in-sample r(spread, next runoff) = {r:.3f} "
          f"(p={p:.4g}, n={len(sub)})")

    # T2/T3: rolling OOS slope adjustments. sig_lag[t] = signal as of t-1.
    values = runoff.to_numpy(dtype=float)
    sig_lag_sp = df["spread"].shift(1).to_numpy()
    sig_lag_d = df["d_broker"].shift(1).fillna(0).to_numpy()
    ea, eb = rolling_slope_oos(values, sig_lag_sp)
    res_t2 = _paired("spread_level", ea, eb)
    ea, eb = rolling_slope_oos(values, sig_lag_d)
    res_t3 = _paired("d_broker", ea, eb)
    print("[exp71b] T2 spread-level:", json.dumps(res_t2))
    print("[exp71b] T3 d_broker:", json.dumps(res_t3))

    # T4: router feature.
    res_t4 = router_feature_test(values, runoff.index, df["spread"])
    print("[exp71b] T4 router:", json.dumps(res_t4))

    out = {
        "broker_months_raw": n_raw,
        "window": [str(df.index.min().date()), str(df.index.max().date())],
        "T1_insample": {"r": r, "p": float(p), "n": len(sub)},
        "T2_spread_level_oos": res_t2,
        "T3_d_broker_oos": res_t3,
        "T4_router_feature": res_t4,
    }
    path = Path(__file__).parent / "exp71b_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
