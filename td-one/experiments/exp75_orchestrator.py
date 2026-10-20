r"""exp75: Mini-agent + orchestrator decision engine — experiment.

========================================================================
PRE-REGISTRATION (see experiments/exp75_report.md for the full text)
========================================================================
Arms: A base alone (v2 on OSFI/Food CPI/SMM; Chronos-2 on shifts;
group forecaster on the simulator bed) / B deterministic orchestrator
(FORECAST) / C LLM orchestrator (FORECAST, flash-lite, cached) /
D deterministic orchestrator (UNCERTAINTY).

Beds: OSFI (5 segments x 60 origins, h=1); Food CPI (h=1, h=6); SMM
(50 origins, h=1); shifts S1/S2/S3 (Chronos-2 real, sev 1/4, seeds 0/1,
9 origins, h=6); SimulatorV2 + group forecaster (exp72 worlds).

Success bar per bed: B/C beat A at p<0.05 pooled AND no segment harm
(>+2% at p<0.10); D moves coverage toward 0.80 without width >1.5x
native and no significant CRPS loss. C must beat B (p<0.05, pooled
OSFI+FoodCPI h=1+S3) to justify the LLM orchestrator.

Iteration budget (standing rule): 4-5 diagnosis-driven rounds.
Candidate grids are cached per bed (exp75_grids_cache/) so rounds are
cheap; the log lives in exp75_report.md.

Usage:
    .venv/bin/python experiments/exp75_orchestrator.py --bed osfi
    .venv/bin/python experiments/exp75_orchestrator.py --bed food --horizon 1
    .venv/bin/python experiments/exp75_orchestrator.py --bed shifts
    .venv/bin/python experiments/exp75_orchestrator.py --bed all --llm
    .venv/bin/python experiments/exp75_orchestrator.py --bed osfi --ablate
    .venv/bin/python experiments/exp75_orchestrator.py --bed osfi --search
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp52_vector_baselines as e52  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402
from exp70_osfi_benchmark import (  # noqa: E402
    TRAIN_WINDOW,
    load_runoff_rates,
)

from td_one.agent.factor_agents import (  # noqa: E402
    FACTOR_FAMILIES,
    FactorCache,
    FactorSignal,
    read_all_factors,
    read_diagnostics,
)
from td_one.agent.layer2_tools import pinball_crps, summarize_backtest  # noqa: E402
from td_one.agent.orchestrator import (  # noqa: E402
    DeterministicOrchestrator,
    OrchestratorConfig,
    fit_uncertainty_params,
    world_pressure,
)

NEWS_DB = ROOT / "data" / "news" / "news_2026.db"
GRID_CACHE = ROOT / "experiments" / "exp75_grids_cache"
FACTOR_CACHE_PATH = ROOT / "experiments" / "exp75_factor_cache.json"
LLM_CACHE_PATH = ROOT / "experiments" / "exp75_llm_cache.json"
RESULTS_JSON = ROOT / "experiments" / "exp75_results.json"
BURNIN = 12  # origins of pure validation history before decisions start
VAL_WINDOW = 24


# ---------------------------------------------------------------------------
# Candidate-grid builders (cached per bed)
# ---------------------------------------------------------------------------


@dataclass
class BedData:
    """One bed = several units (segments); each unit has aligned grids."""

    bed: str
    target_kind: str  # attrition | price
    units: dict[str, dict[str, Any]]  # unit -> {values, origins, months, candidates, actuals}
    base_name: str = "v2"
    burnin: int = BURNIN
    group_series_by_unit: dict[str, dict[int, dict[str, np.ndarray]]] | None = None


def _cached_bed(name: str) -> BedData | None:
    path = GRID_CACHE / f"{name}.pkl"
    if not path.exists():
        return None
    import pickle  # noqa: PLC0415

    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 — own cache file


def _store_bed(name: str, bed: BedData) -> None:
    import pickle  # noqa: PLC0415

    GRID_CACHE.mkdir(parents=True, exist_ok=True)
    with (GRID_CACHE / f"{name}.pkl").open("wb") as fh:
        pickle.dump(bed, fh)


def bed_osfi() -> BedData:
    hit = _cached_bed("osfi")
    if hit is not None:
        return hit
    from exp73_metarouter import SEGMENTS, build_records, v2_grids  # noqa: PLC0415

    rates = load_runoff_rates()
    units: dict[str, dict[str, Any]] = {}
    for seg in ["total", *SEGMENTS]:
        series = rates[seg]
        values = series.to_numpy(dtype=float)
        months = series.index
        records, feature_names = build_records(values, 1)
        grids_v2 = v2_grids(records, feature_names, 1)
        candidates: dict[str, list[np.ndarray]] = {
            m: [np.asarray(r["quantiles"][m], dtype=float) for r in records]
            for m in lfc.CLASSICAL_MODELS
            if m != "v2"
        }
        candidates["v2"] = grids_v2
        units[seg] = {
            "values": values,
            "origins": [r["origin"] for r in records],
            "months": [str(pd.Timestamp(months[r["origin"]]).strftime("%Y-%m")) for r in records],
            "candidates": candidates,
            "actuals": [np.asarray(r["actual"], dtype=float) for r in records],
        }
    bed = BedData("osfi", "attrition", units)
    _store_bed("osfi", bed)
    return bed


def _expert_grids_h(train: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Classical grids at horizon > 1 (exp73's construction)."""
    from td_one.agent.tools.data import naive_forecasts  # noqa: PLC0415

    nf = naive_forecasts(train, horizon, period=12)
    naive = np.asarray(nf.naive, dtype=float).T  # (9, H)
    resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
    sigma12 = float(np.std(resid)) if len(resid) > 1 else 1.0
    cols = []
    for s in range(1, horizon + 1):
        val = train[-12 + ((s - 1) % 12)] if len(train) >= 12 else train[-1]
        sig = sigma12 * float(np.sqrt(np.ceil(s / 12)))
        cols.append(val + lfc.Z * sig)
    snaive = np.stack(cols, axis=1)
    ets = np.stack([e52.fit_ets_like(train, s)[:, 0] for s in range(1, horizon + 1)], axis=1)
    arima = np.stack([e52.fit_arima_like(train, s)[:, 0] for s in range(1, horizon + 1)], axis=1)
    return {"naive": naive, "snaive": snaive, "ets": ets, "arima": arima}


def _v2_blocked_cv(
    records: list[dict[str, Any]], feature_names: list[str], horizon: int
) -> list[np.ndarray]:
    from exp73_metarouter import v2_grids  # noqa: PLC0415

    return v2_grids(records, feature_names, horizon)


def bed_food(horizon: int) -> BedData:
    name = f"food_h{horizon}"
    hit = _cached_bed(name)
    if hit is not None:
        return hit
    from td_one.agent.tools.data import series_features  # noqa: PLC0415

    df = pd.read_csv(lfc.DATA_CSV, parse_dates=["date"])
    values = df["value"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(df["date"])
    origins = lfc.origins_59(values)
    feature_names = lfc.feature_names_for(values)
    records: list[dict[str, Any]] = []
    for o in origins:
        train = values[o - TRAIN_WINDOW : o]
        if len(train) < TRAIN_WINDOW or o + horizon > len(values):
            continue
        actual = values[o : o + horizon]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = series_features(train, period=12)
            qm = lfc.origin_quantiles(train) if horizon == 1 else _expert_grids_h(train, horizon)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        records.append(
            {
                "origin": int(o),
                "features": [float(fd.get(fn, 0.0)) for fn in feature_names],
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {m: pinball_crps(q, actual) for m, q in qm.items()},
                "actual": actual.tolist(),
            }
        )
    grids_v2 = _v2_blocked_cv(records, feature_names, horizon)
    candidates: dict[str, list[np.ndarray]] = {
        m: [np.asarray(r["quantiles"][m], dtype=float) for r in records]
        for m in ("naive", "snaive", "ets", "arima")
    }
    candidates["v2"] = grids_v2
    units = {
        "food_cpi": {
            "values": values,
            "origins": [r["origin"] for r in records],
            "months": [str(dates[r["origin"]].strftime("%Y-%m")) for r in records],
            "candidates": candidates,
            "actuals": [np.asarray(r["actual"], dtype=float) for r in records],
        }
    }
    bed = BedData(name, "price", units)
    _store_bed(name, bed)
    return bed


def bed_smm() -> BedData:
    hit = _cached_bed("smm")
    if hit is not None:
        return hit
    from exp65_smm_llm import build_classical_records, load_smm  # noqa: PLC0415

    values, dates = load_smm()
    n_origins = 50
    origins = list(range(len(values) - n_origins, len(values)))
    feature_names = lfc.feature_names_for(values)
    records = build_classical_records(values, origins, feature_names)
    norm: list[dict[str, Any]] = []
    for r in records:
        qm = {m: np.asarray(q, dtype=float) for m, q in r["quantiles"].items()}
        norm.append(
            {
                "origin": int(r["origin"]),
                "features": [float(x) for x in r["features"]],
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {
                    m: pinball_crps(q, np.array([float(r["actual"][0])])) for m, q in qm.items()
                },
                "actual": [float(r["actual"][0])],
            }
        )
    grids_v2 = _v2_blocked_cv(norm, feature_names, 1)
    candidates = {
        m: [np.asarray(r["quantiles"][m], dtype=float) for r in norm]
        for m in ("naive", "snaive", "ets", "arima")
    }
    candidates["v2"] = grids_v2
    units = {
        "smm": {
            "values": values,
            "origins": [r["origin"] for r in norm],
            "months": [str(dates[r["origin"]].strftime("%Y-%m")) for r in norm],
            "candidates": candidates,
            "actuals": [np.asarray(r["actual"], dtype=float) for r in norm],
        }
    }
    bed = BedData("smm", "attrition", units)
    _store_bed("smm", bed)
    return bed


def bed_shifts() -> BedData:
    """S1/S2/S3 shift scenarios (exp67 protocol), base = Chronos-2 real.

    Units: {shift}/sev{sev}/seed{seed}; 9 rolling origins, h=6. Synthetic
    months -> factor agents degrade (by design); the diagnostics family
    is what this bed tests.
    """
    hit = _cached_bed("shifts")
    if hit is not None:
        return hit
    from td_one.data_gen.shift_scenarios import (  # noqa: PLC0415
        ShiftSpec,
        apply_regional_divergence,
        apply_shift,
        generate_base_series,
    )
    from td_one.evaluation.real_benchmark import interp_quantiles_to_grid  # noqa: PLC0415
    from td_one.llm_rungs.prompts import LEVELS_9  # noqa: PLC0415
    from td_one.models.tsfm import Chronos2Wrapper  # noqa: PLC0415

    horizon = 6
    origins = [18, 21, 24, 27, 30, 33, 36, 39, 42]
    onset = 30
    wrapper = Chronos2Wrapper(
        prediction_length=horizon,
        use_mock=False,
        weights_dir=str(ROOT / "models" / "weights" / "chronos2"),
    )
    units: dict[str, dict[str, Any]] = {}
    for shift in ("S1", "S2", "S3"):
        for sev in (1.0, 4.0):
            for seed in (0, 1):
                rng = np.random.default_rng(1000 + seed)
                if shift == "S3":
                    ra = generate_base_series(48, seed=2000 + seed)
                    rb = generate_base_series(48, seed=3000 + seed)
                    series = apply_regional_divergence(
                        ra, rb, ShiftSpec("S3", onset, sev), 0.5, rng
                    )
                else:
                    base = generate_base_series(48, seed=4000 + seed)
                    series = apply_shift(base, ShiftSpec(shift, onset, sev), rng)
                rate = np.asarray(series["attrition_rate"], dtype=float)
                chronos_g: list[np.ndarray] = []
                naive_g: list[np.ndarray] = []
                actuals: list[np.ndarray] = []
                for o in origins:
                    ctx = rate[:o]
                    levels, vals = wrapper.forecast_quantiles(ctx, horizon=horizon, seed=seed)
                    chronos_g.append(interp_quantiles_to_grid(levels, vals, np.array(LEVELS_9)))
                    from exp72_integration import naive_grid9  # noqa: PLC0415

                    g1 = naive_grid9(ctx)  # (9,) step-1 style; tile to horizon
                    naive_g.append(np.tile(g1[:, None], (1, horizon)))
                    actuals.append(rate[o : o + horizon])
                units[f"{shift}/sev{sev}/seed{seed}"] = {
                    "values": rate,
                    "origins": origins,
                    "months": None,
                    "candidates": {"chronos2": chronos_g, "naive": naive_g},
                    "actuals": actuals,
                }
                print(f"  shifts unit {shift}/sev{sev}/seed{seed} done", flush=True)
    bed = BedData("shifts", "attrition", units, base_name="chronos2", burnin=3)
    _store_bed("shifts", bed)
    return bed


def bed_sim() -> BedData:
    """SimulatorV2 + group forecaster bed (exp72 machinery, seed 7).

    Units: S3_4 and S1_4 worlds x {diverged group, portfolio}. Base =
    v2 group forecaster on the unit's series; GroupAgent sees all groups.
    Stored per-unit group series (truncated on the fly in run_unit via
    the ``group_series`` payload keyed by unit).
    """
    hit = _cached_bed("sim")
    if hit is not None:
        return hit
    from exp72_integration import (  # noqa: PLC0415
        CachedForecaster,
        build_seed_data,
        build_world,
        naive_grid9,
    )

    from td_one.agent.group_forecaster import v2_forecaster_fn  # noqa: PLC0415

    sd = build_seed_data(7)
    fcaster = CachedForecaster(v2_forecaster_fn(1))
    units: dict[str, dict[str, Any]] = {}
    group_payloads: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for world in ("S3_4", "S1_4"):
        groups = build_world(sd, world)
        ctrl = build_world(sd, "control")
        # Diverged group = largest |drift| vs control over post-onset months.
        drift: dict[str, float] = {}
        for gid, gs in groups.items():
            if gid == "__portfolio__":
                continue
            a = np.asarray(gs.attrition_rate, dtype=float)
            b = np.asarray(ctrl[gid].attrition_rate, dtype=float)
            drift[gid] = float(np.max(np.abs(a[sd.onset :] - b[sd.onset :])))
        div_gid = max(drift, key=lambda g: drift[g])
        for label, gid in (("diverged", div_gid), ("portfolio", "__portfolio__")):
            gs = groups[gid]
            values = np.asarray(gs.attrition_rate, dtype=float)
            origins = list(range(24, 47))
            v2_g: list[np.ndarray] = []
            naive_g: list[np.ndarray] = []
            actuals: list[np.ndarray] = []
            for o in origins:
                train = values[:o]
                v2_g.append(np.asarray(fcaster(train), dtype=float).reshape(9, 1))
                naive_g.append(naive_grid9(train)[:, None])
                actuals.append(values[o : o + 1])
            unit_name = f"{world}/{label}"
            units[unit_name] = {
                "values": values,
                "origins": origins,
                "months": None,
                "candidates": {"v2": v2_g, "naive": naive_g},
                "actuals": actuals,
            }
            # Truncated group series per origin for the GroupAgent.
            per_origin: dict[int, dict[str, np.ndarray]] = {}
            for o in origins:
                per_origin[o] = {
                    g: np.asarray(gseries.attrition_rate, dtype=float)[:o]
                    for g, gseries in groups.items()
                }
                # GroupAgent keys the portfolio as "total".
                per_origin[o]["total"] = per_origin[o].pop("__portfolio__")
            group_payloads[unit_name] = per_origin
    bed = BedData(
        "sim",
        "attrition",
        units,
        base_name="v2",
        burnin=12,
        group_series_by_unit=group_payloads,
    )
    _store_bed("sim", bed)
    return bed


# ---------------------------------------------------------------------------
# Generic walk-forward orchestrator evaluation
# ---------------------------------------------------------------------------


def _blend(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    return w * np.asarray(a, dtype=float) + (1.0 - w) * np.asarray(b, dtype=float)


def _val_winner(
    cand_crps: dict[str, list[float]], margin: float = 0.02, noise_k: float = 1.0
) -> tuple[str, bool]:
    """Pick the validation winner; check it beats naive by margin & noise floor."""
    table = summarize_backtest(cand_crps, naive_name="naive")
    best = str(table["best"])
    if best == "naive":
        return "naive", False
    naive_mean = float(table["naive"]["mean_crps"])
    gain = naive_mean - float(table[best]["mean_crps"])
    rel = gain / naive_mean if naive_mean > 0 else 0.0
    beats = rel >= margin and gain > noise_k * float(table[best]["se_vs_naive"])
    return best, bool(beats)


@dataclass
class UnitResults:
    arm_crps: dict[str, list[float]]
    arm_cov: dict[str, list[float]]
    arm_width: dict[str, list[float]]
    traces_b: list[dict[str, Any]]
    factor_avail_rate: float
    mean_abs_p: float


def _coverage_width(grid: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    lo = np.minimum(grid[0], grid[8])
    hi = np.maximum(grid[0], grid[8])
    cov = float(np.mean((actual >= lo) & (actual <= hi)))
    return cov, float(np.mean(hi - lo))


def run_unit(
    unit_name: str,
    u: dict[str, Any],
    bed_name: str,
    target_kind: str,
    base_name: str = "v2",
    config: OrchestratorConfig | None = None,
    drop_families: set[str] | None = None,
    drop_factors: set[str] | None = None,
    drop_diags: set[str] | None = None,
    factor_cache: FactorCache | None = None,
    group_series_by_index: dict[int, dict[str, np.ndarray]] | None = None,
    arms: tuple[str, ...] = ("A", "B", "D"),
    burnin: int = BURNIN,
) -> UnitResults:
    """Walk-forward evaluation of arms A/B/D for one bed unit."""
    cfg = config or OrchestratorConfig()
    orch = DeterministicOrchestrator(cfg)
    values = np.asarray(u["values"], dtype=float)
    origins: list[int] = list(u["origins"])
    months: list[str] | None = u.get("months")
    cands: dict[str, list[np.ndarray]] = u["candidates"]
    actuals: list[np.ndarray] = [np.asarray(a, dtype=float) for a in u["actuals"]]
    naive_g = cands["naive"]
    base_g = cands[base_name]
    blend_g = [_blend(base_g[i], naive_g[i], cfg.blend_weight) for i in range(len(origins))]
    cand_all = {"naive": naive_g, base_name: base_g, "blend": blend_g}
    n = len(origins)

    # Precompute per-origin CRPS of every candidate (for validation tables).
    cand_crps: dict[str, list[float]] = {
        name: [pinball_crps(grids[i], actuals[i]) for i in range(n)]
        for name, grids in cand_all.items()
    }
    # Precompute signals + diagnostics per origin (walk-forward visible data).
    sigs_all: list[list[FactorSignal]] = []
    diags_all: list[list] = []
    n_fired = 0
    ps: list[float] = []
    for i, o in enumerate(origins):
        visible = values[:o]
        if months is not None:
            bundle = read_all_factors(NEWS_DB, months[i], target_kind, factor_cache)
        else:
            bundle = []
        if drop_factors:
            bundle = [s for s in bundle if s.category not in drop_factors]
        if drop_families:
            bundle = [s for s in bundle if s.family not in drop_families]
        gs = group_series_by_index.get(o) if group_series_by_index else None
        cand_at = {name: grids[i] for name, grids in cand_all.items()}
        diags = read_diagnostics(visible, cand_at, gs)
        if drop_diags:
            diags = [d for d in diags if d.name not in drop_diags]
        sigs_all.append(bundle)
        diags_all.append(diags)
        fired = any(s.available for s in bundle)
        n_fired += int(fired)
        p, _ = world_pressure(bundle, cfg.family_weights)
        ps.append(abs(p))

    arm_crps: dict[str, list[float]] = {a: [] for a in ("A", "B", "C", "D")}
    arm_cov: dict[str, list[float]] = {a: [] for a in ("A", "B", "C", "D")}
    arm_width: dict[str, list[float]] = {a: [] for a in ("A", "B", "C", "D")}
    traces_b: list[dict[str, Any]] = []

    for i in range(burnin, n):
        o = origins[i]
        act = actuals[i]
        val_idx = list(range(max(0, i - VAL_WINDOW), i))
        val_crps = {name: [cand_crps[name][j] for j in val_idx] for name in cand_crps}
        winner, beats = _val_winner(val_crps, margin=cfg.val_margin)
        base_grid = base_g[i]
        naive_grid = naive_g[i]
        naive_q50 = float(naive_grid[4, 0])
        base_q50 = float(base_grid[4, 0])

        def _emit(arm: str, grid: np.ndarray, act: np.ndarray) -> None:
            arm_crps[arm].append(pinball_crps(grid, act))
            c, w = _coverage_width(grid, act)
            arm_cov[arm].append(c)
            arm_width[arm].append(w)

        _emit("A", base_grid, act)
        # Arm B: deterministic orchestrator, FORECAST mode.
        if "B" in arms:
            trace = orch.decide_forecast(
                sigs_all[i], diags_all[i], winner, beats, base_q50, naive_q50
            )
            grid_b = {"naive": naive_grid, "v2": base_grid, "blend": blend_g[i]}[trace.choice]
            _emit("B", grid_b, act)
            traces_b.append(
                {
                    "unit": unit_name,
                    "origin": int(o),
                    "month": months[i] if months else None,
                    "choice": trace.choice,
                    "reason": trace.reason,
                    "world_pressure": trace.world_pressure,
                    "veto": trace.veto,
                    "family_scores": trace.family_scores,
                }
            )
        # Arm D: deterministic orchestrator, UNCERTAINTY mode on the base.
        if "D" in arms:
            alpha, beta = fit_uncertainty_params(
                [actuals[j] for j in val_idx],
                [base_g[j] for j in val_idx],
                [diags_all[j] for j in val_idx],
                [sigs_all[j] for j in val_idx],
                coverage_target=cfg.coverage_target,
            )
            m, _t = orch.decide_uncertainty(diags_all[i], sigs_all[i], alpha, beta)
            _emit("D", orch.apply_multiplier(base_grid, m), act)

    used = [a for a in ("A", "B", "C", "D") if arm_crps[a]]
    return UnitResults(
        arm_crps={a: arm_crps[a] for a in used},
        arm_cov={a: arm_cov[a] for a in used},
        arm_width={a: arm_width[a] for a in used},
        traces_b=traces_b,
        factor_avail_rate=(n_fired / n if n else 0.0),
        mean_abs_p=float(np.mean(ps)) if ps else 0.0,
    )


def run_unit_llm(
    unit_name: str,
    u: dict[str, Any],
    bed_name: str,
    target_kind: str,
    base_name: str,
    burnin: int,
    llm_orch: LLMOrchestrator,
    factor_cache: FactorCache | None = None,
    group_series_by_index: dict[int, dict[str, np.ndarray]] | None = None,
) -> dict[str, list[float]]:
    """Arm C: LLM orchestrator (FORECAST mode) over one bed unit."""
    values = np.asarray(u["values"], dtype=float)
    origins: list[int] = list(u["origins"])
    months: list[str] | None = u.get("months")
    cands: dict[str, list[np.ndarray]] = u["candidates"]
    actuals: list[np.ndarray] = [np.asarray(a, dtype=float) for a in u["actuals"]]
    naive_g, base_g = cands["naive"], cands[base_name]
    blend_g = [_blend(base_g[i], naive_g[i]) for i in range(len(origins))]
    cand_all = {"naive": naive_g, base_name: base_g, "blend": blend_g}
    cand_crps = {
        name: [pinball_crps(grids[i], actuals[i]) for i in range(len(origins))]
        for name, grids in cand_all.items()
    }
    crps_c: list[float] = []
    for i in range(burnin, len(origins)):
        o = origins[i]
        val_idx = list(range(max(0, i - VAL_WINDOW), i))
        val_crps = {name: [cand_crps[name][j] for j in val_idx] for name in cand_crps}
        winner, beats = _val_winner(val_crps)
        table = summarize_backtest(val_crps, naive_name="naive")
        val_summary = "; ".join(
            f"{name} val CRPS {table[name]['mean_crps']:.6f}"
            for name in ("naive", base_name, "blend")
        )
        resid = np.array(
            [float(base_g[j][4, 0]) - float(actuals[j][0]) for j in val_idx], dtype=float
        )
        bundle = (
            read_all_factors(NEWS_DB, months[i], target_kind, factor_cache)
            if months is not None
            else []
        )
        gs = group_series_by_index.get(o) if group_series_by_index else None
        diags = read_diagnostics(values[:o], {n: g[i] for n, g in cand_all.items()}, gs)
        decision, shift = llm_orch.decide(
            bed_name,
            unit_name,
            months[i] if months else "synthetic",
            int(np.asarray(actuals[i]).size),
            bundle,
            diags,
            float(naive_g[i][4, 0]),
            float(base_g[i][4, 0]),
            val_summary,
            float(np.std(resid)) if resid.size else 0.0,
        )
        grid = {"naive": naive_g[i], "v2": base_g[i], "blend": blend_g[i]}[decision["choice"]]
        if shift:
            grid = grid + shift
        crps_c.append(pinball_crps(grid, actuals[i]))
    return {"C": crps_c}


def run_llm_arms(
    bed: BedData,
    client: Any,
    factor_cache: FactorCache | None = None,
    units: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Run arm C over (a subset of) a bed's units with the LLM client."""
    cache = LLMOrchestratorCache(LLM_CACHE_PATH)
    llm_orch = LLMOrchestrator(client, cache)
    out: dict[str, dict[str, list[float]]] = {}
    for unit, u in bed.units.items():
        if units is not None and unit not in units:
            continue
        gs = bed.group_series_by_unit.get(unit) if bed.group_series_by_unit else None
        out[unit] = run_unit_llm(
            unit, u, bed.bed, bed.target_kind, bed.base_name, bed.burnin,
            llm_orch, factor_cache, gs,
        )
        print(f"  arm C unit {unit} done", flush=True)
    return out


def paired(a: list[float], b: list[float]) -> dict[str, Any]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) < 3:
        return {
            "mean_a": float(np.mean(x)),
            "mean_b": float(np.mean(y)),
            "diff_pct": float("nan"),
            "p": float("nan"),
            "win_rate": float("nan"),
            "n": int(len(x)),
        }
    t, p = ttest_rel(x, y)
    return {
        "mean_a": float(np.mean(x)),
        "mean_b": float(np.mean(y)),
        "diff_pct": float(100 * (np.mean(x) - np.mean(y)) / np.mean(y))
        if np.mean(y)
        else float("nan"),
        "p": float(p),
        "win_rate": float(np.mean(x < y)),
        "n": int(len(x)),
    }


def bed_report(bed: BedData, results: dict[str, UnitResults]) -> dict[str, Any]:
    pooled: dict[str, list[float]] = {}
    per_unit: dict[str, Any] = {}
    for unit, r in results.items():
        for arm, crps in r.arm_crps.items():
            pooled.setdefault(arm, []).extend(crps)
        per_unit[unit] = {
            "_armA_test_crps": r.arm_crps.get("A"),
            "mean_crps": {a: float(np.mean(v)) for a, v in r.arm_crps.items()},
            "coverage": {a: float(np.mean(v)) for a, v in r.arm_cov.items()},
            "width": {a: float(np.mean(v)) for a, v in r.arm_width.items()},
            "B_vs_A": paired(r.arm_crps["B"], r.arm_crps["A"]) if "B" in r.arm_crps else None,
            "D_vs_A": paired(r.arm_crps["D"], r.arm_crps["A"]) if "D" in r.arm_crps else None,
            "factor_avail_rate": r.factor_avail_rate,
            "mean_abs_p": r.mean_abs_p,
            "choices_B": pd.Series([t["choice"] for t in r.traces_b]).value_counts().to_dict()
            if r.traces_b
            else {},
        }
    out: dict[str, Any] = {"bed": bed.bed, "per_unit": per_unit}
    if "A" in pooled:
        out["pooled"] = {
            "mean_crps": {a: float(np.mean(v)) for a, v in pooled.items()},
            "coverage": {},
            "B_vs_A": paired(pooled["B"], pooled["A"]) if "B" in pooled else None,
            "D_vs_A": paired(pooled["D"], pooled["A"]) if "D" in pooled else None,
        }
        cov_pool: dict[str, list[float]] = {}
        wid_pool: dict[str, list[float]] = {}
        for r in results.values():
            for arm in r.arm_cov:
                cov_pool.setdefault(arm, []).extend(r.arm_cov[arm])
                wid_pool.setdefault(arm, []).extend(r.arm_width[arm])
        out["pooled"]["coverage"] = {a: float(np.mean(v)) for a, v in cov_pool.items()}
        out["pooled"]["width"] = {a: float(np.mean(v)) for a, v in wid_pool.items()}
    return out


# ---------------------------------------------------------------------------
# Ablation + parameter search (Rounds 2-3)
# ---------------------------------------------------------------------------


def run_bed_arms(
    bed: BedData,
    arms: tuple[str, ...] = ("A", "B", "D"),
    config: OrchestratorConfig | None = None,
    drop_families: set[str] | None = None,
    drop_factors: set[str] | None = None,
    drop_diags: set[str] | None = None,
    factor_cache: FactorCache | None = None,
) -> dict[str, UnitResults]:
    out = {}
    for unit, u in bed.units.items():
        out[unit] = run_unit(
            unit,
            u,
            bed.bed,
            bed.target_kind,
            config=config,
            drop_families=drop_families,
            drop_factors=drop_factors,
            drop_diags=drop_diags,
            factor_cache=factor_cache,
            arms=arms,
        )
    return out


def pooled_b_minus_a(report: dict[str, Any]) -> dict[str, Any] | None:
    return report.get("pooled", {}).get("B_vs_A")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bed",
        choices=["osfi", "food", "smm", "shifts", "sim", "real", "all"],
        default="osfi",
    )
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Run arm C (LLM orchestrator) on the prioritized subset "
        "(OSFI + Food CPI h=1 + S3 shifts).",
    )
    args = ap.parse_args()

    fcache = FactorCache(FACTOR_CACHE_PATH)
    results: dict[str, Any] = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())

    beds: list[BedData] = []
    if args.bed in ("osfi", "real", "all"):
        beds.append(bed_osfi())
    if args.bed in ("food", "real", "all"):
        beds.append(bed_food(args.horizon))
    if args.bed in ("smm", "real", "all"):
        beds.append(bed_smm())
    if args.bed in ("shifts", "all"):
        beds.append(bed_shifts())
    if args.bed in ("sim", "all"):
        beds.append(bed_sim())

    for bed in beds:
        print(f"== bed {bed.bed} ==", flush=True)
        res = run_bed_arms(bed, factor_cache=fcache)
        rep = bed_report(bed, res)
        results[bed.bed] = rep
        print(json.dumps(rep["pooled"], indent=1), flush=True)
        if args.ablate:
            abl: dict[str, Any] = {}
            for fam in FACTOR_FAMILIES:
                r2 = run_bed_arms(bed, drop_families={fam}, factor_cache=fcache)
                abl[f"drop_family::{fam}"] = pooled_b_minus_a(bed_report(bed, r2))
            for diag in ("stability", "disagreement", "seasonality", "group"):
                r2 = run_bed_arms(bed, drop_diags={diag}, factor_cache=fcache)
                abl[f"drop_diag::{diag}"] = pooled_b_minus_a(bed_report(bed, r2))
            # Per-factor: individual categories by mean |pressure| contribution.
            from td_one.agent.factor_agents import SIGNAL_CATEGORIES  # noqa: PLC0415

            for cat in SIGNAL_CATEGORIES:
                r2 = run_bed_arms(bed, drop_factors={cat}, factor_cache=fcache)
                abl[f"drop_factor::{cat}"] = pooled_b_minus_a(bed_report(bed, r2))
            results[f"{bed.bed}::ablation"] = abl
            print("ablation done", flush=True)
        if args.search:
            search: dict[str, Any] = {}
            for tau_high in (0.2, 0.3, 0.45):
                for tau_low in (0.08, 0.12, 0.2):
                    for veto in (0.6, 0.7, 0.8):
                        cfg = OrchestratorConfig(tau_high=tau_high, tau_low=tau_low, veto_conf=veto)
                        r2 = run_bed_arms(bed, config=cfg, factor_cache=fcache)
                        key = f"th{tau_high}_tl{tau_low}_v{veto}"
                        search[key] = pooled_b_minus_a(bed_report(bed, r2))
            results[f"{bed.bed}::search"] = search
            print("search done", flush=True)

    RESULTS_JSON.write_text(json.dumps(results, indent=1))
    print(f"wrote {RESULTS_JSON}")

    if args.llm:
        from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: PLC0415

        client = VectorProxyClient(model="flash-lite", call_budget=600)
        for bed in beds:
            subset: list[str] | None = None
            if bed.bed == "shifts":
                subset = [u for u in bed.units if u.startswith("S3/")]
            if bed.bed == "sim":
                continue  # arm C not on the simulator bed (Round 1)
            print(f"== arm C bed {bed.bed} ==", flush=True)
            cres = run_llm_arms(bed, client, fcache, subset)
            # Merge: align with arm A on the same test origins.
            merged: dict[str, Any] = {}
            for unit, cres_u in cres.items():
                a_crps = results[bed.bed]["per_unit"][unit].get("_armA_test_crps")
                merged[unit] = {"C_vs_A": None, "mean_crps_C": float(np.mean(cres_u["C"]))}
                if a_crps is not None and len(a_crps) == len(cres_u["C"]):
                    merged[unit]["C_vs_A"] = paired(cres_u["C"], a_crps)
            results[f"{bed.bed}::armC"] = merged
            RESULTS_JSON.write_text(json.dumps(results, indent=1))
            print(f"wrote arm C for {bed.bed}", flush=True)


if __name__ == "__main__":
    main()
