"""Experiment 63: does an LLM add anything on top of the statistical router?

LLM (gpt-oss-120b, low reasoning) combines the 7-model tourism zoo per
series from diagnostics + validation-fold CRPS. Compared (paired, same
test origins) against:
  - proposed: the statistical regime router (run_proposed_cv)
  - best_single: argmin validation CRPS (train folds, no hindsight)
  - equal: equal weights

The LLM gets the UNION of what the router sees (features) and what
best-single sees (validation CRPS) -- the most generous setup, so a null
result is conclusive. Parse failures fall back to best-single (recorded).

PIT: LLM sees train-window features, validation CRPS, and test-origin
quantile forecasts only. Never test actuals.

Usage:
    .venv/bin/python experiments/exp63_llm_selector.py            # no news
    .venv/bin/python experiments/exp63_llm_selector.py --news    # + news context
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import exp42_fforma as f42  # noqa: E402
import monash_common as mc  # noqa: E402
from td_one.agent.proposed_method import ProposedForecaster  # noqa: E402
from td_one.agent.groq_client import GroqChatClient  # noqa: E402

RESULTS = ROOT / "experiments" / "exp63_results.json"
PARTIAL = ROOT / "experiments" / "exp63_partial.jsonl"
NEWS_DB = ROOT / "data" / "news" / "news_2026.db"
N_SUBSAMPLE = 60
SEED = 63

SYSTEM = (
    "You are a forecasting expert combining point-forecast models. "
    "Reply with a single JSON object only."
)


def build_prompt(feats: dict, val_crps: dict, models: list[str],
                 news: dict | None) -> str:
    feat_str = ", ".join(f"{k}={feats[k]:.3f}" for k in f42.FEATURE_NAMES)
    crps_str = ", ".join(f"{m}={val_crps[m]:.4f}" for m in models)
    lines = [
        "Combine 7 forecasting models for one monthly tourism series.",
        f"Series diagnostics: {feat_str}.",
        f"Mean CRPS on validation folds (lower is better): {crps_str}.",
    ]
    if news:
        lines.append(
            f"News context for the forecast origin month: article coverage "
            f"{news['coverage']:.2f} (fraction of prior 12 months with news), "
            f"macro stress signal {news['stress']:.2f} (0=calm, 1=stressed)."
        )
    lines += [
        'Output JSON: {"weights": {model: weight, ...}, "pick": "model_name", '
        '"rationale": "one sentence"}.',
        "Weights must be >= 0 and sum to 1. Favor models with low validation "
        "CRPS; adjust for diagnostics (strong seasonality favors seasonal "
        "models; high entropy/noise favors robust smoothers).",
    ]
    return "\n".join(lines)


def parse_weights(text: str, models: list[str]) -> dict[str, float] | None:
    try:
        d = json.loads(text)
        w = d.get("weights", {})
        vals = np.array([float(w.get(m, 0.0)) for m in models])
        if (vals < 0).any() or vals.sum() <= 0:
            return None
        vals = vals / vals.sum()
        return dict(zip(models, vals.tolist()))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    use_news = "--news" in sys.argv
    data = json.loads((ROOT / "experiments" / "exp60_results.json").read_text())
    records = data["records"]
    models = [m for m in mc.ALL_MODELS if all(m in r["crps"] for r in records)]
    print(f"[exp63] {len(records)} records, models={models}, news={use_news}",
          flush=True)

    # stratified-ish subsample: fixed seed, shuffled series ids
    rng = np.random.default_rng(SEED)
    sids = sorted({r["series_id"] for r in records})
    rng.shuffle(sids)
    sub = set(sids[:N_SUBSAMPLE])
    print(f"[exp63] subsample: {len(sub)} series", flush=True)

    news_monthly = mc.build_news_monthly(NEWS_DB) if use_news else {}
    client = GroqChatClient(call_budget=N_SUBSAMPLE + 10, max_tokens=1500,
                            reasoning_effort="low")

    folds = f42.kfold_by_series(records, n_folds=5, seed=0)
    llm_c, prop_c, best_c, eq_c = [], [], [], []
    parse_fail = 0
    llm_calls = 0

    # restart-safe: skip series already in the partial log
    done: set[str] = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["series_id"] + "|" + json.loads(line)["fold"])
    part_f = open(PARTIAL, "a", encoding="utf-8")  # noqa: PTH123

    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr = [records[i] for i in tr_idx]
        te = [records[i] for i in te_idx if records[i]["series_id"] in sub]
        if not te:
            continue
        val_crps = {m: float(np.mean([r["crps"][m] for r in tr])) for m in models}
        best = min(val_crps, key=val_crps.get)

        fc = ProposedForecaster(h=1, regime_spec=mc.m4_regime_spec(), models=models)
        fc.fit(tr, f42.FEATURE_NAMES)

        for r in te:
            key = r["series_id"] + "|" + str(fi)
            actual = np.asarray(r["actual"], dtype=float)
            qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in models}
            feats = dict(zip(f42.FEATURE_NAMES, r["features"], strict=True))

            # proposed (statistical router)
            out = fc.predict(np.zeros(30), qm, feats, f42.FEATURE_NAMES)
            prop_c.append(f42._crps_quantile(actual, out.quantiles))
            # best single / equal
            best_c.append(f42._crps_quantile(actual, qm[best]))
            q_eq = np.mean([qm[m] for m in models], axis=0)
            eq_c.append(f42._crps_quantile(actual, q_eq))

            if key in done:
                # replay LLM result from partial log (no re-call)
                for line in PARTIAL.read_text().splitlines():
                    d = json.loads(line)
                    if d["series_id"] + "|" + d["fold"] == key:
                        llm_c.append(d["llm_crps"])
                        parse_fail += d.get("parse_fail", 0)
                        break
                continue

            # LLM combination
            news = None
            if use_news and r.get("origin_ym"):
                news = mc.news_features_for_origin(news_monthly, r["origin_ym"])
            prompt = build_prompt(feats, val_crps, models, news)
            pf = 0
            try:
                text = client.complete(prompt, system=SYSTEM, json_mode=True)
                llm_calls += 1
            except Exception as e:  # noqa: BLE001
                print(f"[exp63] LLM call failed for {r['series_id']}: {e}")
                text = ""
            w = parse_weights(text, models)
            if w is None:
                pf = 1
                parse_fail += 1
                w = {m: (1.0 if m == best else 0.0) for m in models}
            q_llm = sum(w[m] * qm[m] for m in models)
            crps_llm = float(f42._crps_quantile(actual, q_llm))
            llm_c.append(crps_llm)
            part_f.write(json.dumps({"series_id": r["series_id"], "fold": str(fi),
                                     "llm_crps": crps_llm,
                                     "prop_crps": float(prop_c[-1]),
                                     "best_crps": float(best_c[-1]),
                                     "eq_crps": float(eq_c[-1]),
                                     "parse_fail": pf}) + "\n")
            part_f.flush()
            time.sleep(0.5)
    part_f.close()

    llm_c, prop_c = np.array(llm_c), np.array(prop_c)
    best_c, eq_c = np.array(best_c), np.array(eq_c)
    print(f"[exp63] n={len(llm_c)}, llm calls={llm_calls}, "
          f"parse failures={parse_fail}", flush=True)

    def cmp(a: np.ndarray, b: np.ndarray) -> dict:
        t, p = ttest_rel(a, b)
        return {"mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "mean_diff": float(np.mean(a - b)), "t": float(t),
                "p": float(p), "win_rate": float(np.mean(a < b))}

    out = {
        "dataset": "tourism_monthly", "n_series": len(sub), "n": len(llm_c),
        "use_news": use_news, "models": models,
        "llm_calls": llm_calls, "parse_failures": parse_fail,
        "mean_crps": {"llm": float(np.mean(llm_c)),
                      "proposed": float(np.mean(prop_c)),
                      "best_single": float(np.mean(best_c)),
                      "equal": float(np.mean(eq_c))},
        "llm_vs_proposed": cmp(llm_c, prop_c),
        "llm_vs_best": cmp(llm_c, best_c),
        "llm_vs_equal": cmp(llm_c, eq_c),
    }
    RESULTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[exp63] saved {RESULTS}")
    print(json.dumps(out["mean_crps"], indent=1))
    print("llm vs proposed:", json.dumps(out["llm_vs_proposed"], indent=1))
    print("llm vs best_single:", json.dumps(out["llm_vs_best"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
