"""exp84: competitor spread at loan level — era-split sign stability test.

Pre-registered design (written before any model was fit):

  Sample: Freddie SFLLD originations 2021Q1-2026Q1, product FRM,
  term 350-365 months (30-yr), orig_rate and PMMS both observed.
  Spread: spread_bps = (orig_rate - PMMS month before first payment)*100.
  Outcome: prepay_H = voluntary payoff (zb 01) within H months of first
  payment; loans with another termination (zb != 01) within H are
  excluded (competing event); loans are in the H-sample only if they
  have H months of follow-up or an event within H.
  Eras (by first-payment date): E1 2021, E2 2022-2023, E3 2024+.
  Primary horizon H=24; robustness H=12, H=36.

  BAR: (a) spread slope (logit coef per 100bp) has the same sign in all
  three eras at H=24; (b) pooled out-of-sample prediction (train era A,
  test era B, all ordered pairs) improves: mean delta-logloss > 0 for
  the with-spread model vs the no-spread baseline, same controls.

  Mechanism-consistent prediction (falsifiable): slope > 0 in every era
  (borrowers who originated above market prepay more).

Rounds (researcher-mindset iteration log, all reported):
  R1 primary spec (linear spread, logit, controls).
  R2 nonlinear spread bins (deciles) — shape / monotonicity.
  R3 transforms: rate ratio; time-varying refi-incentive hazard
     (discrete-time, loan sample, incentive_t = PMMS_t - orig_rate).
  R4 interactions: spread x market level; spread x loan purpose.
  R5 robustness: H=12/36, PMMS timing (fp / fp-1 / fp-2), reference =
     dataset's own monthly mean rate, vintage-quarter fixed effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

TD_ONE = Path(__file__).resolve().parents[1]
RESULTS: dict[str, object] = {"rounds": {}}

CONTROL_BANDS = ["fico_band", "ltv_band", "dti_band", "purpose"]


def band(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels).astype(str)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    # Mutates in place (callers pass frames they own) — full-frame
    # copies of a 7M-row panel OOM this shared 7GB VM.
    df["fico_band"] = band(
        df["credit_score"], [0, 680, 720, 760, 900],
        ["<680", "680-719", "720-759", "760+"],
    )
    df["ltv_band"] = band(
        df["oltv_orig"], [0, 80, 90, 95, 200], ["<=80", "80-90", "90-95", ">95"]
    )
    df["dti_band"] = band(
        df["dti"], [0, 36, 45, 100], ["<=36", "36-45", ">45"]
    ).replace("nan", "missing")
    df["era"] = np.where(
        df["first_payment_date"] < 202200,
        "E1_2021",
        np.where(df["first_payment_date"] < 202400, "E2_2022_23", "E3_2024p"),
    )
    return df



def maybe_sample(d: pd.DataFrame, cap: int = 200_000, seed: int = 84) -> pd.DataFrame:
    """Cap inference fits: at n>cap a random subsample keeps SEs tiny
    and statsmodels fast. Full data is used for sklearn OOS metrics."""
    if len(d) <= cap:
        return d
    return d.sample(cap, random_state=seed)


def horizon_sample(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Loans observable for horizon h (or with an event inside it)."""
    d = df
    d["prepay"] = (
        (d["event_type"] == "prepay") & (d["months_to_event"] <= h)
    ).astype(int)
    other_in_h = (d["event_type"] == "other_term") & (d["months_to_event"] <= h)
    # Complete windows only: followup >= h. Admitting event<=h loans
    # with shorter followup (an earlier draft did) selects fast
    # prepayers from late cohorts while excluding their active peers —
    # it inflated E3's base rate to 0.70. Standard binary-outcome design:
    # the full window must be observable for every included loan.
    observable = d["followup_months"] >= h
    return d[observable & ~other_in_h]


def design(d: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = ["spread_bps", "log_upb"] + (extra or [])
    x = pd.get_dummies(d[CONTROL_BANDS], drop_first=True, dtype=float)
    base = pd.DataFrame(
        {"spread_c": d["spread_bps"] / 100.0, "log_upb": d["log_upb"]}, index=d.index
    )
    if extra:
        base = pd.concat([base, d[extra]], axis=1)
    out = pd.concat([base, x], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    # Guard: a dummy level unobserved in a subsample (e.g. purpose
    # 'other') is an all-zero column -> singular Hessian in Logit.
    out = out.loc[:, out.nunique() > 1]
    return sm.add_constant(out, has_constant="add")


def fit_logit(x: pd.DataFrame, y: pd.Series) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    return sm.Logit(y, x).fit(disp=0, maxiter=200)


def era_fits(d: pd.DataFrame, x_cols_from: pd.DataFrame | None = None) -> dict:
    out = {}
    for era, g0 in d.groupby("era"):
        g = maybe_sample(g0)
        x = design(g)
        res = fit_logit(x, g["prepay"])
        out[era] = {
            "n": int(len(g0)),
            "n_fit": int(len(g)),
            "base_rate": float(g["prepay"].mean()),
            "spread_coef_per100bp": float(res.params["spread_c"]),
            "spread_se": float(res.bse["spread_c"]),
            "spread_p": float(res.pvalues["spread_c"]),
            "spread_or": float(np.exp(res.params["spread_c"])),
        }
    return out


def pooled_oos(d: pd.DataFrame) -> dict:
    """Train era A -> test era B with sklearn (fast); delta logloss/AUC.

    Eras are capped at 400k loans (seeded): full-matrix sklearn on 5M+
    rows OOMs this shared VM, and log-loss/AUC deltas are precise
    at 400k with 10-30% base rates.
    """
    d = pd.concat(
        [maybe_sample(g, cap=400_000, seed=7) for _, g in d.groupby("era")],
        ignore_index=True,
    )
    feats_full = ["spread_bps", "log_upb"]
    eras = sorted(d["era"].unique())
    cat = pd.get_dummies(d[CONTROL_BANDS], drop_first=True, dtype=float)
    x_all = pd.concat(
        [d[["spread_bps", "log_upb"]].reset_index(drop=True), cat.reset_index(drop=True)],
        axis=1,
    ).astype("float32")
    x_all["spread_bps"] = x_all["spread_bps"] / 100.0
    pairs = {}
    d = d.reset_index(drop=True)
    for a in eras:
        tr = d["era"] == a
        for b in eras:
            if a == b:
                continue
            te = d["era"] == b
            ytr, yte = d.loc[tr, "prepay"], d.loc[te, "prepay"]
            res_pair = {}
            for name, cols in [
                ("full", list(x_all.columns)),
                ("no_spread", [c for c in x_all.columns if c != "spread_bps"]),
            ]:
                m = LogisticRegression(max_iter=1000, C=1e6)
                m.fit(x_all.loc[tr, cols], ytr)
                p = m.predict_proba(x_all.loc[te, cols])[:, 1]
                res_pair[name] = {
                    "logloss": float(log_loss(yte, p)),
                    "auc": float(roc_auc_score(yte, p)),
                }
            res_pair["delta_logloss"] = (
                res_pair["no_spread"]["logloss"] - res_pair["full"]["logloss"]
            )
            res_pair["delta_auc"] = res_pair["full"]["auc"] - res_pair["no_spread"]["auc"]
            pairs[f"{a}->{b}"] = res_pair
    deltas = [v["delta_logloss"] for v in pairs.values()]
    return {
        "pairs": pairs,
        "mean_delta_logloss": float(np.mean(deltas)),
        "share_pairs_positive": float(np.mean(np.array(deltas) > 0)),
    }


def r2_bins(d: pd.DataFrame) -> dict:
    """Prepay rate by within-era spread decile (shape check)."""
    out = {}
    dd = d.copy()
    dd["bin"] = dd.groupby("era")["spread_bps"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
    )
    for era, g in dd.groupby("era"):
        t = g.groupby("bin").agg(
            n=("prepay", "size"),
            rate=("prepay", "mean"),
            spread_mean=("spread_bps", "mean"),
        )
        out[era] = {
            str(int(i)): {
                "n": int(r["n"]),
                "rate": float(r["rate"]),
                "spread_mean_bps": float(r["spread_mean"]),
            }
            for i, r in t.iterrows()
        }
    return out


def r3_hazard(panel: pd.DataFrame, seed: int = 84) -> dict:
    """Discrete-time hazard on a loan sample: incentive_t = PMMS_t - rate.

    Expands loans to loan-months up to event/censoring at 36 months
    (vectorized via np.repeat). Uses monthly PMMS from FRED weekly.
    """
    pmms_raw = pd.read_parquet(TD_ONE / "data" / "fred" / "MORTGAGE30US.parquet")
    pmms_raw["date"] = pd.to_datetime(pmms_raw["date"])
    monthly = pmms_raw.set_index("date")["value"].resample("MS").mean()
    pmms = {d.year * 100 + d.month: float(v) for d, v in monthly.items()}

    # panel = compact frame (fp / event_code / ...), adapted in place.
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(panel), size=min(150_000, len(panel)), replace=False)
    d = panel.iloc[np.sort(pos)].copy()
    d = d.rename(columns={"fp": "first_payment_date",
                          "followup": "followup_months"})
    d["event_type"] = pd.Categorical.from_codes(
        d["event_code"], categories=["active", "prepay", "other_term"]
    )
    d["era"] = np.where(
        d["first_payment_date"] < 202200, "E1_2021",
        np.where(d["first_payment_date"] < 202400, "E2_2022_23", "E3_2024p"),
    )

    event_m = np.where(
        d["event_type"] == "active",
        np.inf,
        d["months_to_event"].fillna(np.inf),
    )
    last_m = np.minimum.reduce(
        [np.minimum(event_m, 36), d["followup_months"].to_numpy(dtype=float)]
    )
    counts = np.maximum(last_m, 0).astype(int)
    keep = counts > 0
    d = d[keep]
    counts = counts[keep]
    rep = np.repeat(np.arange(len(d)), counts)
    age = np.concatenate([np.arange(1, c + 1) for c in counts])
    start = d["first_payment_date"].to_numpy()[rep]
    idx = (start // 100) * 12 + (start % 100 - 1) + age
    ym = (idx // 12) * 100 + (idx % 12) + 1
    pmms_t = np.array([pmms.get(int(v), np.nan) for v in ym])
    evt = d["event_type"].to_numpy()[rep]
    etm = d["months_to_event"].fillna(-1).to_numpy()[rep]
    hz = pd.DataFrame(
        {
            "era": d["era"].to_numpy()[rep],
            "incentive_bps": (pmms_t - d["orig_rate"].to_numpy()[rep]) * 100.0,
            "loan_age": age,
            "spread_c": d["spread_bps"].to_numpy()[rep] / 100.0,
            "log_upb": d["log_upb"].to_numpy()[rep],
            "prepay": ((evt == "prepay") & (etm == age)).astype(int),
        }
    ).dropna(subset=["incentive_bps"])
    hz["age_band"] = band(hz["loan_age"], [0, 6, 12, 24, 37],
                          ["1-6", "7-12", "13-24", "25-36"])
    out = {}
    for era, g in hz.groupby("era"):
        x = pd.get_dummies(g[["age_band"]], drop_first=True, dtype=float)
        core = g[["incentive_bps", "spread_c", "log_upb"]].reset_index(drop=True)
        core["incentive_bps"] = core["incentive_bps"] / 100.0
        x = sm.add_constant(
            pd.concat([core, x.reset_index(drop=True)], axis=1),
            has_constant="add",
        )
        res = fit_logit(x, g["prepay"].reset_index(drop=True))
        out[era] = {
            "n_loan_months": int(len(g)),
            "incentive_coef_per100bp": float(res.params["incentive_bps"]),
            "incentive_p": float(res.pvalues["incentive_bps"]),
            "orig_spread_coef": float(res.params["spread_c"]),
            "orig_spread_p": float(res.pvalues["spread_c"]),
        }
    return out


def load_base() -> pd.DataFrame:
    """Load the compact numeric frame built from the panel (see
    exp84_build_panel.py + frame encoder): FRM30 only, all-numeric,
    ~350MB in memory — the string-heavy panel OOMs this shared VM."""
    d = pd.read_parquet(TD_ONE / "data" / "freddie" / "exp84_frame.parquet")
    d = d.rename(
        columns={
            "fp": "first_payment_date",
            "oltv": "oltv_orig",
            "followup": "followup_months",
        }
    )
    d["event_type"] = pd.Categorical.from_codes(
        d["event_code"], categories=["active", "prepay", "other_term"]
    )
    d["purpose"] = pd.Categorical.from_codes(
        d["purpose_code"], categories=["P", "C", "N", "other"]
    )
    return prep(d)


def population_h24_stats() -> dict:
    """Population era n / base prepay rate at H=24, computed leanly
    from the compact frame (four columns only)."""
    f = pd.read_parquet(
        TD_ONE / "data" / "freddie" / "exp84_frame.parquet",
        columns=["fp", "event_code", "months_to_event", "followup"],
    )
    ok = (f["followup"] >= 24) & ~(
        (f["event_code"] == 2) & (f["months_to_event"] <= 24)
    )
    f = f[ok]
    prepay = ((f["event_code"] == 1) & (f["months_to_event"] <= 24)).astype(float)
    era = np.where(
        f["fp"] < 202200, "E1_2021",
        np.where(f["fp"] < 202400, "E2_2022_23", "E3_2024p"),
    )
    g = pd.DataFrame({"era": era, "prepay": prepay}).groupby("era")["prepay"]
    return {
        e: {"n_population": int(g.count()[e]), "base_rate": float(g.mean()[e])}
        for e in ["E1_2021", "E2_2022_23", "E3_2024p"]
    }


def round_r1(base: pd.DataFrame) -> None:
    # base = H24 complete-window sample (fits cap at 200k/era anyway);
    # population n / base rates come from population_h24_stats().
    h24 = horizon_sample(base, 24)
    RESULTS["population_h24"] = population_h24_stats()
    fr = pd.read_parquet(
        TD_ONE / "data" / "freddie" / "exp84_frame.parquet",
        columns=["spread_bps"],
    )
    RESULTS["sample"] = {
        "frm30_loans": int(len(fr)),
        "spread_bps_deciles": {
            str(k): float(v)
            for k, v in fr["spread_bps"].quantile(np.linspace(0, 1, 11)).items()
        },
    }
    RESULTS["rounds"]["R1_primary_H24"] = {
        "era_fits": era_fits(h24),
        "n": int(len(h24)),
    }


def round_oos(base: pd.DataFrame) -> None:
    h24 = horizon_sample(base, 24)
    RESULTS.setdefault("rounds", {}).setdefault("R1_primary_H24", {})
    RESULTS["rounds"]["R1_primary_H24"]["pooled_oos"] = pooled_oos(h24)


def round_r2(base: pd.DataFrame) -> None:
    RESULTS["rounds"]["R2_bins_H24"] = r2_bins(horizon_sample(base, 24))


def round_r3a(base: pd.DataFrame) -> None:
    h24 = horizon_sample(base, 24)
    out_rr = {}
    for era, g0 in h24.groupby("era"):
        g = maybe_sample(g0)
        x = design(g)
        x["spread_c"] = np.log(g["orig_rate"] / g["market_pmms"])
        res = fit_logit(x, g["prepay"])
        out_rr[era] = {
            "logratio_coef": float(res.params["spread_c"]),
            "p": float(res.pvalues["spread_c"]),
        }
    RESULTS["rounds"]["R3a_rate_ratio_H24"] = out_rr


def round_r3b(base: pd.DataFrame) -> None:
    RESULTS["rounds"]["R3b_hazard"] = r3_hazard(base)


def round_r4(base: pd.DataFrame) -> None:
    d = horizon_sample(base, 24)
    d["spread_x_pmms"] = (d["spread_bps"] / 100.0) * d["market_pmms"]
    d["is_refi"] = d["purpose"].isin(["C", "N"]).astype(float)
    d["spread_x_refi"] = (d["spread_bps"] / 100.0) * d["is_refi"]
    d = maybe_sample(d, cap=400_000)
    x = design(d, extra=["spread_x_pmms", "spread_x_refi"])
    x = pd.concat(
        [x, pd.get_dummies(d["era"], drop_first=True, dtype=float)], axis=1
    )
    res4 = fit_logit(x, d["prepay"])
    RESULTS["rounds"]["R4_interactions_H24"] = {
        "spread_main": float(res4.params["spread_c"]),
        "spread_main_p": float(res4.pvalues["spread_c"]),
        "spread_x_pmms": float(res4.params["spread_x_pmms"]),
        "spread_x_pmms_p": float(res4.pvalues["spread_x_pmms"]),
        "spread_x_refi": float(res4.params["spread_x_refi"]),
        "spread_x_refi_p": float(res4.pvalues["spread_x_refi"]),
    }


def round_r5(_base: pd.DataFrame) -> None:
    """Robustness on the lean per-horizon samples (exp84_make_samples).

    H12/H36 era fits use their own complete-window samples; the
    PMMS-timing / own-mean / vintage-FE variants run on the H24 sample
    (mutate-and-restore, no frame copies). Own-mean reference means are
    estimated on the sample (400k/era) rather than the population.
    """
    rob: dict[str, object] = {}
    for h in (12, 36):
        hh = pd.read_parquet(TD_ONE / "data" / "freddie" / f"exp84_s_h{h}.parquet")
        rob[f"H{h}"] = {"era_fits": era_fits(horizon_sample(hh, h)),
                        "n": int(len(hh))}
        del hh
    base = pd.read_parquet(TD_ONE / "data" / "freddie" / "exp84_s_h24.parquet")
    pmms_raw = pd.read_parquet(TD_ONE / "data" / "fred" / "MORTGAGE30US.parquet")
    pmms_raw["date"] = pd.to_datetime(pmms_raw["date"])
    monthly = pmms_raw.set_index("date")["value"].resample("MS").mean()
    pmms_map = pd.Series(
        monthly.values, index=[dt.year * 100 + dt.month for dt in monthly.index]
    )

    def add_months(s_: pd.Series, k: int) -> pd.Series:
        idx = (s_ // 100) * 12 + (s_ % 100 - 1) + k
        return (idx // 12) * 100 + (idx % 12) + 1

    orig_spread = base["spread_bps"].copy()
    orig_pmms = base["market_pmms"].copy()
    for label, k in [("pmms_fp_month", 0), ("pmms_fp_minus2", -2)]:
        base["market_pmms"] = add_months(base["first_payment_date"], k).map(pmms_map)
        base["spread_bps"] = (base["orig_rate"] - base["market_pmms"]) * 100.0
        rob[label] = {"era_fits": era_fits(horizon_sample(base, 24))}
    own_mean = base.groupby("first_payment_date")["orig_rate"].transform("mean")
    base["spread_bps"] = (base["orig_rate"] - own_mean) * 100.0
    rob["own_mean_reference"] = {"era_fits": era_fits(horizon_sample(base, 24))}
    base["spread_bps"] = orig_spread
    base["market_pmms"] = orig_pmms
    h24 = horizon_sample(base, 24)
    vfe = {}
    for era, g0 in h24.groupby("era"):
        g = maybe_sample(g0)
        x = design(g)
        q = pd.get_dummies(
            (g["first_payment_date"] // 100) * 10 + (g["first_payment_date"] % 100 - 1) // 3,
            prefix="q", drop_first=True, dtype=float,
        )
        x = pd.concat([x, q.set_index(x.index)], axis=1)
        res = fit_logit(x, g["prepay"])
        vfe[era] = {
            "spread_coef_per100bp": float(res.params["spread_c"]),
            "spread_p": float(res.pvalues["spread_c"]),
        }
    rob["vintage_quarter_FE"] = vfe
    RESULTS["rounds"]["R5_robustness"] = rob


ROUNDS = {
    "r1": round_r1,
    "oos": round_oos,
    "r2": round_r2,
    "r3a": round_r3a,
    "r3b": round_r3b,
    "r4": round_r4,
    "r5": round_r5,
}


def main() -> None:
    """Staged runner: one round per process (shared 7GB VM — a single
    all-rounds process OOMs). Usage: exp84_loan_level_spread.py r1|oos|...
    Each round merges its output into exp84_results.json."""
    import sys

    out = Path(__file__).with_name("exp84_results.json")
    if out.exists():
        RESULTS.update(json.loads(out.read_text()))
    which = sys.argv[1] if len(sys.argv) > 1 else "r1"
    if which in ("r1", "oos", "r2", "r3a", "r4"):
        # Lean H24 complete-window sample (exp84_make_samples.py).
        base = pd.read_parquet(
            TD_ONE / "data" / "freddie" / "exp84_s_h24.parquet"
        )
    elif which == "r3b":
        base = pd.read_parquet(
            TD_ONE / "data" / "freddie" / "exp84_frame.parquet",
            columns=["fp", "orig_rate", "spread_bps", "log_upb",
                     "followup", "months_to_event", "event_code"],
        )
    elif which == "r5":
        base = pd.DataFrame()  # round_r5 loads its own samples
    else:
        base = load_base()
    ROUNDS[which](base)
    out.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"round {which} done")


if __name__ == "__main__":
    main()
