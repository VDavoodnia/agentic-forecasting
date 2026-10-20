"""Layer-2 meta-router (exp73): an analyst with tools over Layer-1 forecasters.

exp70's diagnosis of the proposed method on OSFI runoff was architectural:
the v1/v2 router (a) picks bad experts with no do-no-harm gate, (b) has no
channel for exogenous features, and (c) behaves differently by horizon.
Layer-2 is a *second decision stage* over any set of candidate forecasters
(here: naive / snaive / ets / arima / v2 / blend(v2, naive)) that fixes all
three without touching Layer-1:

1. **Validation-gated expert choice, per segment and horizon.** At each
   origin, candidates are scored on a trailing validation window of
   *strictly earlier* origins (:func:`td_one.agent.layer2_tools
   .summarize_backtest`). The validation winner replaces naive only if it
   beats naive by a pre-set relative margin AND the paired improvement
   exceeds the noise floor (``noise_k`` x SE). Otherwise Layer-2 abstains
   to naive — do-no-harm by construction.
2. **Analyst vetoes.** Before the margin gate, the toolbox is consulted:
   a fresh changepoint (:func:`changepoint_scan`), a volatility explosion
   or a badly under-covering winner on validation all force abstention;
   if the winner is Layer-1 (v2), its historical override hit rate
   (:func:`router_audit`) must be at least 0.5.
3. **Exogenous lock-in correction.** Optionally (arm D), a tiny ridge
   (intercept + one slope) maps the as-of lock-in level
   (:class:`td_one.agent.layer2_tools.OSFILockInMeter`) to the gated
   forecast's step-1 residual, fit on validation origins only, clipped to
   +-``lockin_clip_sd`` validation residual SDs, and shifts the whole
   quantile grid. This gives exogenous features a channel that Layer-1
   lacks — as a *correction layer*, not a second model.

Everything is deterministic and walk-forward: the decision at origin
``o`` uses only origins ``< o`` and ``y[:o]``. Each origin emits a
JSON-able **decision trace** (evidence bundle + rule fired + adjustment
applied) so a human can audit the "reasoning" end to end.

The router consumes precomputed candidate grids, so it is agnostic to
how Layer-1 produced them (blocked-CV folds, live fits, caches).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from td_one.agent.layer2_tools import (
    changepoint_scan,
    interval_compare,
    pinball_crps,
    q50_of,
    router_audit,
    series_diagnostics,
    summarize_backtest,
)

#: Name of the synthesised blend candidate (Vincent average of the two grids).
BLEND_NAME = "blend_v2_naive"


@dataclass(frozen=True)
class MetaRouterConfig:
    """Pre-registered Layer-2 policy parameters (exp73)."""

    val_window: int = 24  # trailing validation origins per decision
    min_val: int = 12  # below this: cold start -> naive
    margin: float = 0.02  # required relative CRPS gain vs naive
    noise_k: float = 1.0  # gain must exceed noise_k * paired SE
    coverage_floor: float = 0.5  # winner's validation 80% coverage floor
    vol_abstain: float = 3.0  # vol_ratio above this -> abstain
    ridge_lambda: float = 1.0  # lock-in slope penalty
    lockin_clip_sd: float = 2.0  # clip shift to +-k * validation resid SD
    use_lockin: bool = False
    lockin_evidence_gate: bool = False  # shift must win on held-out validation
    naive_name: str = "naive"
    layer1_name: str = "v2"


@dataclass
class MetaResult:
    """Per-origin outputs of a Layer-2 run (aligned to ``origins``)."""

    traces: list[dict[str, Any]]
    choices: list[str]
    grids: list[np.ndarray]  # final Layer-2 grids
    crps: dict[str, list[float]]  # {"layer2": [...]} + candidates
    candidate_crps: dict[str, list[float]]


@dataclass
class _History:
    """Sequential state: everything Layer-2 has observed so far."""

    crps: dict[str, list[float]] = field(default_factory=dict)
    grids: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actuals: list[np.ndarray] = field(default_factory=list)
    base_chosen_q50: list[float] = field(default_factory=list)
    base_chosen_grid: list[np.ndarray] = field(default_factory=list)
    lockin: list[float | None] = field(default_factory=list)


class MetaRouter:
    """Walk-forward Layer-2 decision stage over candidate forecasters.

    Args:
        config: policy parameters (see :class:`MetaRouterConfig`).
        lockin_fn: optional callable mapping a series index (origin) to
            the as-of lock-in level (PIT-filtered by the caller, e.g. via
            :class:`OSFILockInMeter`), or ``None`` when unavailable.

    """

    def __init__(
        self,
        config: MetaRouterConfig | None = None,
        lockin_fn: Callable[[int], float | None] | None = None,
    ) -> None:
        self.cfg = config or MetaRouterConfig()
        self._lockin_fn = lockin_fn

    # -- public ----------------------------------------------------------

    def run(
        self,
        y: np.ndarray,
        origins: Sequence[int],
        candidate_grids: Mapping[str, Sequence[np.ndarray]],
        actuals: Sequence[np.ndarray],
    ) -> MetaResult:
        """Run Layer-2 sequentially over ``origins`` (time-ordered).

        ``candidate_grids[name][i]`` is the (9, H) grid of candidate
        ``name`` for origin ``origins[i]``; ``actuals[i]`` the (H,)
        realised values. A ``blend_v2_naive`` candidate is synthesised
        when both parents are present. Returns choices, final grids,
        per-origin CRPS for Layer-2 and every candidate, and the full
        decision traces.
        """
        cfg = self.cfg
        series = np.asarray(y, dtype=float).ravel()
        grids = self._with_blend(candidate_grids)
        if cfg.naive_name not in grids:
            raise ValueError(f"candidates must include {cfg.naive_name!r}")
        n = len(origins)
        if len(actuals) != n or any(len(g) != n for g in grids.values()):
            raise ValueError("candidate_grids/actuals must align with origins")

        hist = _History(
            crps={name: [] for name in grids},
            grids={name: [] for name in grids},
        )
        cand_crps: dict[str, list[float]] = {name: [] for name in grids}
        layer2_crps: list[float] = []
        choices: list[str] = []
        final_grids: list[np.ndarray] = []
        traces: list[dict[str, Any]] = []

        for i, o in enumerate(origins):
            act = np.asarray(actuals[i], dtype=float).ravel()
            origin_grids = {name: np.asarray(g[i], dtype=float) for name, g in grids.items()}
            for name, g in origin_grids.items():
                cand_crps[name].append(pinball_crps(g, act))

            lockin_val = self._lockin_fn(int(o)) if self._lockin_fn else None
            choice, reason, evidence, val_idx = self._decide(series, int(o), hist)
            base_grid = origin_grids[choice]
            # The lock-in correction adjusts WHATEVER the gate chose
            # (arm D): it is a correction layer over the gated forecast.
            shift, lockin_info = self._apply_lockin(hist, val_idx, lockin_val)
            evidence["lockin"] = lockin_info
            final = base_grid + shift if shift else base_grid

            layer2_crps.append(pinball_crps(final, act))
            choices.append(choice)
            final_grids.append(final)
            traces.append(
                {
                    "origin_index": int(o),
                    "origin_pos": i,
                    "choice": choice,
                    "reason": reason,
                    "lockin_shift": float(shift),
                    "layer2_crps": float(layer2_crps[-1]),
                    "evidence": evidence,
                }
            )
            # Append observations AFTER deciding (walk-forward purity).
            for name, g in origin_grids.items():
                hist.crps[name].append(cand_crps[name][-1])
                hist.grids[name].append(g)
            hist.actuals.append(act)
            hist.base_chosen_q50.append(q50_of(base_grid))
            hist.base_chosen_grid.append(base_grid)
            hist.lockin.append(lockin_val)

        return MetaResult(
            traces=traces,
            choices=choices,
            grids=final_grids,
            crps={"layer2": layer2_crps, **cand_crps},
            candidate_crps=cand_crps,
        )

    # -- internals ---------------------------------------------------------

    def _with_blend(
        self, candidate_grids: Mapping[str, Sequence[np.ndarray]]
    ) -> dict[str, list[np.ndarray]]:
        grids = {name: list(g) for name, g in candidate_grids.items()}
        cfg = self.cfg
        if BLEND_NAME not in grids and cfg.layer1_name in grids and cfg.naive_name in grids:
            grids[BLEND_NAME] = [
                0.5 * (np.asarray(a, dtype=float) + np.asarray(b, dtype=float))
                for a, b in zip(grids[cfg.layer1_name], grids[cfg.naive_name], strict=True)
            ]
        return grids

    def _apply_lockin(
        self,
        hist: _History,
        val_idx: list[int] | None,
        lockin_val: float | None,
    ) -> tuple[float, dict[str, Any] | None]:
        """Apply the lock-in correction to the gated forecast (arm D)."""
        cfg = self.cfg
        if not cfg.use_lockin or lockin_val is None or not val_idx:
            return 0.0, None
        if len(val_idx) < cfg.min_val:
            return 0.0, {"n_fit": 0, "reason": "insufficient lockin history"}
        return self._lockin_shift(hist, val_idx, lockin_val)

    def _decide(
        self,
        series: np.ndarray,
        origin: int,
        hist: _History,
    ) -> tuple[str, str, dict[str, Any], list[int] | None]:
        """One origin's decision. Returns (choice, reason, evidence, val_idx)."""
        cfg = self.cfg
        naive = cfg.naive_name
        evidence: dict[str, Any] = {"val_n": 0, "val_crps": {}, "lockin": None}

        # Analyst tools on data visible at this origin.
        visible = series[:origin]
        try:
            diag: dict[str, Any] = series_diagnostics(visible)
        except ValueError:
            diag = {"available": False}
        evidence["diagnostics"] = diag
        evidence["changepoint"] = changepoint_scan(visible)

        val_idx = list(range(max(0, len(hist.actuals) - cfg.val_window), len(hist.actuals)))
        evidence["val_n"] = len(val_idx)
        if len(val_idx) < cfg.min_val:
            return naive, "cold_start", evidence, None
        if evidence["changepoint"]["break_flag"]:
            return naive, "changepoint_abstain", evidence, val_idx
        if isinstance(diag.get("vol_ratio"), float) and diag["vol_ratio"] > cfg.vol_abstain:
            return naive, "volatility_abstain", evidence, val_idx

        val_crps = {name: [hist.crps[name][j] for j in val_idx] for name in hist.crps}
        table = summarize_backtest(val_crps, naive_name=naive)
        evidence["val_crps"] = {name: table[name]["mean_crps"] for name in val_crps}
        evidence["backtest"] = table

        # Router audit: has Layer-1 earned trust on this segment?
        l1 = cfg.layer1_name
        if l1 in hist.grids:
            override = [
                bool(np.max(np.abs(hist.grids[l1][j] - hist.grids[naive][j])) > 1e-9)
                for j in val_idx
            ]
            evidence["router_audit"] = router_audit(
                [hist.crps[l1][j] for j in val_idx],
                [hist.crps[naive][j] for j in val_idx],
                override,
            )
        # Interval calibration of every candidate on validation.
        val_actuals = np.stack([hist.actuals[j] for j in val_idx])
        evidence["intervals"] = interval_compare(
            {name: np.stack([hist.grids[name][j] for j in val_idx]) for name in hist.grids},
            val_actuals,
        )

        best = str(table["best"])
        if best == naive:
            return naive, "naive_best", evidence, val_idx
        if best == l1 and "router_audit" in evidence:
            hit = evidence["router_audit"]["hit_rate"]
            if isinstance(hit, float) and hit == hit and hit < 0.5:
                return naive, "router_distrust", evidence, val_idx
        naive_mean = float(table[naive]["mean_crps"])
        gain = naive_mean - float(table[best]["mean_crps"])
        rel_gain = gain / naive_mean if naive_mean > 0 else 0.0
        evidence["margin"] = {
            "best": best,
            "rel_gain": float(rel_gain),
            "gain": float(gain),
            "se": float(table[best]["se_vs_naive"]),
        }
        if rel_gain < cfg.margin:
            return naive, "margin_gate", evidence, val_idx
        if gain <= cfg.noise_k * float(table[best]["se_vs_naive"]):
            return naive, "noise_abstain", evidence, val_idx
        cov = evidence["intervals"][best]["coverage_80"]
        if cov < cfg.coverage_floor:
            return naive, "coverage_veto", evidence, val_idx

        return best, "validation_winner", evidence, val_idx

    def _lockin_shift(
        self,
        hist: _History,
        val_idx: list[int],
        lockin_val: float | None,
    ) -> tuple[float, dict[str, Any] | None]:
        """Ridge correction of the gated forecast from validation residuals.

        Fits ``resid ~ a + b * lockin`` (b ridge-penalised, a free) on
        the validation origins' step-1 residuals of the *base* gated
        forecasts, then shifts by the prediction at this origin's
        lock-in level, clipped to +-``lockin_clip_sd`` residual SDs.
        Returns (shift, trace-info); shift 0 when disabled/unavailable.

        With ``lockin_evidence_gate`` on, the fit must first prove
        itself out-of-sample *within* the validation window: fit on
        the first half, and the correction is applied only if it beats
        the no-correction baseline on the second half's residual MSE.
        An unconditionally-applied correction fits noise on targets
        whose innovations carry no exogenous signal (exp73 diagnosis:
        lock-in vs runoff innovations |r| <= 0.04); the gate makes the
        correction earn its place like every other Layer-2 override.
        """
        cfg = self.cfg
        if not cfg.use_lockin or lockin_val is None:
            return 0.0, None
        xs: list[float] = []
        rs: list[float] = []
        for j in val_idx:
            lj = hist.lockin[j]
            if lj is None:
                continue
            xs.append(lj)
            rs.append(float(hist.actuals[j][0]) - hist.base_chosen_q50[j])
        info: dict[str, Any] = {"n_fit": len(xs), "shift": 0.0}
        if len(xs) < cfg.min_val:
            info["reason"] = "insufficient lockin history"
            return 0.0, info
        x = np.asarray(xs)
        r = np.asarray(rs)

        def _fit(xf: np.ndarray, rf: np.ndarray) -> tuple[float, float]:
            x_bar, r_bar = float(np.mean(xf)), float(np.mean(rf))
            denom = float(np.sum((xf - x_bar) ** 2)) + cfg.ridge_lambda
            slope = float(np.sum((xf - x_bar) * (rf - r_bar)) / denom) if denom > 0 else 0.0
            return r_bar - slope * x_bar, slope

        if cfg.lockin_evidence_gate:
            half = len(xs) // 2
            a1, b1 = _fit(x[:half], r[:half])
            held = r[half:]
            mse_with = float(np.mean((held - (a1 + b1 * x[half:])) ** 2))
            mse_without = float(np.mean(held**2))
            info["gate_mse_with"] = mse_with
            info["gate_mse_without"] = mse_without
            if mse_with >= mse_without:
                info["applied"] = False
                info["reason"] = "lockin_no_validation_gain"
                return 0.0, info
            info["applied"] = True
        intercept, slope = _fit(x, r)
        raw = intercept + slope * float(lockin_val)
        sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        clip = cfg.lockin_clip_sd * sd
        shift = float(np.clip(raw, -clip, clip)) if clip > 0 else 0.0
        info.update(
            {
                "slope": slope,
                "intercept": intercept,
                "raw_shift": float(raw),
                "clip": float(clip),
                "shift": shift,
                "lockin": float(lockin_val),
            }
        )
        return shift, info
