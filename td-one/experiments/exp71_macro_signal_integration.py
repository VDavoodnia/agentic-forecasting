"""exp71: Macro Signal integration on SMM.

Feeds the live macro-signal vector (exp70) into ProposedForecasterV2's
router news-modulation path (``news_features``) and compares against the
identical forecaster without news, on SMM origins where signals exist.

Design (pre-registered):
  Arm A: ProposedForecasterV2(dataset="smm", use_llm_expert=False, use_news=False)
  Arm B: same factory with use_news=True; per-origin news_features from
         live signals (exp70_macro_signal_results.json):
             coverage    = fraction of live categories with a signal at asof
             stress      = mean(direction of live signals) / 2  in [-1, 1]
  PIT: origin at month M uses signals_asof(M-1) (last observed month).
  Origins: SMM idx 34..61 (first with signal coverage) — 28 origins,
           blocked 5-fold time CV, paired t-test + win rate.

Kill criterion (researcher-mindset version): B must beat A at p < 0.05.
On failure, diagnose and try ONE alternative before concluding:
  alt: coverage = mean certainty, stress = certainty-weighted mean
       direction / 2 (encoding variant).
If that also fails, the signal stays a research artifact — documented,
not wired in.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp52_vector_baselines as e52  # noqa: E402

from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import naive_forecasts, series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.news.macro_signal import signals_asof  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "news" / "news_2026.db"
SMM_CSV = ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv"
EXP70_JSON = ROOT / "experiments" / "exp70_macro_signal_results.json"
LEVELS_9 = np.array([0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95])
Z = np.array([-1.645, -1.282, -0.674, -0.253, 0.0, 0.253, 0.674, 1.282, 1.645])
CLASSICAL = ["naive", "snaive", "ets", "arima"]


def load_smm() -> tuple[np.ndarray, pd.DatetimeIndex]:
    df = pd.read_csv(SMM_CSV)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    assert len(df) == 63, f"expected 63 months, got {len(df)}"
    return df["smm"].to_numpy(dtype=float), pd.DatetimeIndex(df["date"])


def feature_names_for(values: np.ndarray) -> list[str]:
    feats = series_features(values[:30], period=12)
    fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
    return sorted(fd.keys())


def origin_quantiles(train: np.ndarray) -> dict[str, np.ndarray]:
    qm: dict[str, np.ndarray] = {}
    nq = naive_forecasts(train, 1, period=12)
    q = np.asarray(nq.naive)
    qm["naive"] = q.T if q.shape == (1, 9) else q
    sp = train[-12] if len(train) >= 12 else train[-1]
    resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    qm["snaive"] = (sp + Z * sigma).reshape(9, 1)
    qm["ets"] = e52.fit_ets_like(train, 1)
    qm["arima"] = e52.fit_arima_like(train, 1)
    return qm


def build_records(values: np.ndarray, origins: list[int], feature_names: list[str]) -> list[dict]:
    records = []
    for o in origins:
        train = values[:o]
        actual = float(values[o])
        feats = series_features(train, period=12)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        qm = origin_quantiles(train)
        records.append(
            {
                "origin": int(o),
                "features": [float(fd.get(fn, 0.0)) for fn in feature_names],
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {m: grid_to_crps(q[:, 0], LEVELS_9, actual) for m, q in qm.items()},
                "actual": [actual],
            }
        )
    return records


def load_live_categories() -> list[str]:
    if not EXP70_JSON.exists():
        print(f"{EXP70_JSON.name} missing — run exp70 first")
        return []
    data = json.loads(EXP70_JSON.read_text())
    return [c for c, r in data["categories"].items() if r.get("live")]


def news_features_fn(
    live: list[str],
    db_path: Path,
    dates: pd.DatetimeIndex,
    encoding: str = "base",
    lag: int = 0,
    cats: list[str] | None = None,
):
    """Build the per-record news_features closure (PIT-safe via asof month).

    ``lag`` shifts the asof month back by ``lag`` months (timing variant).
    ``cats`` restricts the signal vector to a subset of live categories
    (ablation variant).
    """
    con = sqlite3.connect(db_path)
    use = cats if cats is not None else live

    def fn(r: dict) -> dict | None:
        o = int(r["origin"])
        asof = dates[o - 1 - lag]
        asof_ym = asof.strftime("%Y-%m")
        sigs = [s for c, s in signals_asof(con, asof_ym).items() if c in use]
        if not sigs:
            return None
        dirs = np.array([s.direction for s in sigs], float)
        certs = np.array([s.certainty for s in sigs], float)
        if encoding == "base":
            return {
                "coverage": float(len(sigs) / len(use)),
                "stress_signal": float(np.mean(dirs) / 2.0),
            }
        # alt: certainty-weighted encoding
        return {
            "coverage": float(np.mean(certs)),
            "stress_signal": float(np.average(dirs, weights=certs) / 2.0),
        }

    return fn


def build_signal_records(
    records: list[dict],
    feature_names: list[str],
    live: list[str],
    dates: pd.DatetimeIndex,
    db_path: Path,
    mode: str,
) -> tuple[list[dict], list[str]]:
    """Extend records with PIT-safe signal features (router-features arm).

    mode="composite": [mean_dir/2, mean_cert, frac_positive] (3 dims).
    mode="percat": one direction dim per live category (0 when missing).
    """
    con = sqlite3.connect(db_path)
    extra_names = (
        ["sig_mean_dir", "sig_mean_cert", "sig_frac_pos"]
        if mode == "composite"
        else [f"sig_{c}" for c in live]
    )
    out = []
    for r in records:
        o = int(r["origin"])
        asof_ym = dates[o - 1].strftime("%Y-%m")
        sigs = signals_asof(con, asof_ym)
        if mode == "composite":
            vals = [s for c, s in sigs.items() if c in live]
            if vals:
                dirs = np.array([s.direction for s in vals], float)
                extra = [
                    float(np.mean(dirs) / 2.0),
                    float(np.mean([s.certainty for s in vals])),
                    float(np.mean(dirs > 0)),
                ]
            else:
                extra = [0.0, 0.0, 0.0]
        else:
            extra = [float(sigs[c].direction / 2.0) if c in sigs else 0.0 for c in live]
        out.append({**r, "features": [*r["features"], *extra]})
    return out, [*feature_names, *extra_names]


def _paired(a: list[float], b: list[float]) -> dict:
    a_, b_ = np.asarray(a), np.asarray(b)
    t, p = ttest_rel(a_, b_)
    return {
        "mean_diff": float(np.mean(a_ - b_)),
        "p": float(p),
        "win_rate": float(np.mean(a_ < b_)),
        "n": len(a_),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--alt", action="store_true", help="legacy flag; rounds 2+ now run automatically on failure"
    )
    ap.parse_args()

    live = load_live_categories()
    if not live:
        print(
            "no live categories from exp70 — integration skipped; signal stays a research artifact"
        )
        return
    print(f"live categories ({len(live)}): {live}", flush=True)

    values, dates = load_smm()
    # first origin with signal coverage: asof month (dates[o-1]) >= 2023-10
    first = next(i for i, d in enumerate(dates) if d.strftime("%Y-%m") == "2023-11")
    origins = list(range(first, 62))
    print(f"SMM: {len(values)} months; origins {len(origins)} (idx {first}..61)", flush=True)

    feature_names = feature_names_for(values)
    records = build_records(values, origins, feature_names)

    def make(news: bool, weight: float = 0.3):
        def f() -> ProposedForecasterV2:
            fc = ProposedForecasterV2(
                dataset="smm", use_llm_expert=False, use_news=news, news_weight=weight
            )
            fc.models = [*CLASSICAL]
            return fc

        return f

    def _verdict(stats: dict, mean_a: float) -> str:
        # Effect-size floor: a p<0.05 on a <1% effect is not a win.
        rel = stats["mean_diff"] / mean_a if mean_a else 0.0
        if stats["p"] < 0.05 and rel <= -0.01:
            return "SIGNAL HELPS"
        if stats["p"] < 0.05 and rel >= 0.01:
            return "SIGNAL HURTS"
        return "NEGLIGIBLE"

    def run(design: str, encoding: str, lag: int, weight: float, cats: list[str] | None) -> dict:
        res = blocked_cv_predict(
            records,
            feature_names,
            make(True, weight),
            n_folds=5,
            baseline=make(False),
            fold_mode="time",
            news_features_fn=news_features_fn(live, DB, dates, encoding, lag, cats),
        )
        a, b = res["crps"], res["baseline_crps"]
        stats = _paired(b, a)  # B(with news) vs A(without)
        verdict = _verdict(stats, float(np.mean(a)))
        print(
            f"[R] {design:40s} B={np.mean(b):.6f} A={np.mean(a):.6f} "
            f"diff={stats['mean_diff']:+.6f} p={stats['p']:.4f} "
            f"win={stats['win_rate']:.2f} -> {verdict}",
            flush=True,
        )
        return {
            "design": design,
            "mean_B": float(np.mean(b)),
            "mean_A": float(np.mean(a)),
            "paired": stats,
            "verdict": verdict,
        }

    def run_features(design: str, mode: str) -> dict:
        """Signal-as-features arm: extend record features.

        Compare against the same forecaster on original features (both CV,
        paired by origin).
        """
        recs_b, feats_b = build_signal_records(records, feature_names, live, dates, DB, mode)
        cv_b = blocked_cv_predict(
            recs_b, feats_b, make(False), n_folds=5, baseline="equal", fold_mode="time"
        )
        cv_a = blocked_cv_predict(
            records,
            feature_names,
            make(False),
            n_folds=5,
            baseline="equal",
            fold_mode="time",
        )
        b, a = cv_b["crps"], cv_a["crps"]
        stats = _paired(b, a)
        verdict = _verdict(stats, float(np.mean(a)))
        print(
            f"[R] {design:40s} B={np.mean(b):.6f} A={np.mean(a):.6f} "
            f"diff={stats['mean_diff']:+.6f} p={stats['p']:.4f} "
            f"win={stats['win_rate']:.2f} -> {verdict}",
            flush=True,
        )
        return {
            "design": design,
            "mean_B": float(np.mean(b)),
            "mean_A": float(np.mean(a)),
            "paired": stats,
            "verdict": verdict,
        }

    # 4-5 diagnosis-driven rounds (standing rule, enforced).
    rounds: list[dict] = []
    rounds.append(
        {
            **run("base encoding, news_weight=0.3", "base", 0, 0.3, None),
            "round": 1,
            "taught": "raw integration quality",
        }
    )
    if rounds[0]["verdict"] != "SIGNAL HELPS":
        rounds.append(
            {
                **run("alt encoding: certainty-weighted", "alt", 0, 0.3, None),
                "round": 2,
                "taught": "whether weighting by certainty helps",
            }
        )
    if not any(r["verdict"] == "SIGNAL HELPS" for r in rounds):
        rounds.append(
            {
                **run("lead-1 timing: signal from asof-1 month", "base", 1, 0.3, None),
                "round": 3,
                "taught": "whether signals lag the release calendar",
            }
        )
    if not any(r["verdict"] == "SIGNAL HELPS" for r in rounds):
        w_results = [run(f"news_weight={w}", "base", 0, w, None) for w in (0.1, 0.5, 0.7)]
        best_w = min(w_results, key=lambda r: r["paired"]["mean_diff"])
        rounds.append(
            {
                **best_w,
                "round": 4,
                "taught": f"param search over news_weight {sorted([0.1, 0.5, 0.7])}",
            }
        )
    if not any(r["verdict"] == "SIGNAL HELPS" for r in rounds):
        # R5: different mechanism — signals as router FEATURES (learned),
        # not a hand-set modulation. Task Phase 4 allows both.
        rounds.append(
            {
                **run_features(
                    "signal-as-features: composite [mean_dir, mean_cert, frac_pos]", "composite"
                ),
                "round": 5,
                "taught": "learned use of the signal via the router vs fixed modulation",
            }
        )
    if not any(r["verdict"] == "SIGNAL HELPS" for r in rounds) and len(live) >= 2:
        rounds.append(
            {
                **run_features("signal-as-features: per-category directions", "percat"),
                "round": 6,
                "taught": "whether the router can pick useful categories from 7 dims",
            }
        )
    if not any(r["verdict"] == "SIGNAL HELPS" for r in rounds) and len(live) >= 2:
        drop_results = [run(f"drop-{c}", "base", 0, 0.3, [x for x in live if x != c]) for c in live]
        best_d = min(drop_results, key=lambda r: r["paired"]["mean_diff"])
        rounds.append(
            {
                **best_d,
                "round": 7,
                "taught": "drop-one-category ablation: which category drives the result",
            }
        )

    helped = any(r["verdict"] == "SIGNAL HELPS" for r in rounds)
    print(
        f"\nfinal verdict after {len(rounds)} rounds: "
        f"{'SIGNAL HELPS' if helped else 'signal stays a research artifact'}",
        flush=True,
    )
    results = {
        "date": datetime.now(UTC).isoformat(),
        "live": live,
        "origins": len(origins),
        "rounds": rounds,
        "final": "HELPS" if helped else "ARTIFACT",
    }

    out = ROOT / "experiments" / "exp71_macro_signal_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
