"""Orchestrator: decision engine over mini-agent signals (exp75).

The user-directed architecture: specialist mini-agents (factor agents +
diagnostics family, see :mod:`td_one.agent.factor_agents`) each speak a
narrow vocabulary ``{direction, confidence, reason}``; the orchestrator is
a *decision engine, not a forecaster*. Two decision modes:

- **FORECAST mode**: choose expert (naive / v2 / blend) from the
  corroboration of the factor agents' world-pressure with the validation
  evidence, with vetoes from the diagnostics family (do-no-harm).
- **UNCERTAINTY mode**: scale the base forecast's interval width by a
  multiplier learned on validation to hit nominal coverage.

Two policies: deterministic (primary, auditable) and an LLM orchestrator
variant (arm C) where the LLM reads the mini-agent signals and decides —
structure only, never numbers. Every decision emits a JSON-traceable trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from td_one.agent.factor_agents import (
    FACTOR_FAMILIES,
    DiagnosticSignal,
    FactorSignal,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Pre-registered orchestrator policy parameters (exp75, Round 1)."""

    # FORECAST mode
    tau_high: float = 0.30  # |P| >= this -> strong world-pressure
    tau_low: float = 0.12  # |P| >= this -> weak world-pressure
    blend_weight: float = 0.5  # blend(v2, naive) mixing weight
    veto_conf: float = 0.7  # stability/disagreement conf >= this -> abstain
    val_margin: float = 0.02  # validation winner must beat naive by this
    family_weights: tuple[float, ...] = (1.0,) * 5  # prices,housing,labour,policy,external
    pressure_agg: str = "mean"  # mean | max (Round-4 variant: strongest family)
    # UNCERTAINTY mode
    width_cap: float = 2.5  # multiplier clip
    coverage_target: float = 0.80
    # LLM orchestrator
    shift_sd_frac: float = 0.5  # fixed shift size = frac * validation resid SD


@dataclass
class OrchestratorTrace:
    """Machine-readable decision trace for one origin."""

    mode: str
    choice: str  # naive | v2 | blend
    world_pressure: float
    family_scores: dict[str, float]
    veto: str | None
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


def world_pressure(
    signals: Sequence[FactorSignal],
    family_weights: Sequence[float] | None = None,
    pressure_agg: str = "mean",
) -> tuple[float, dict[str, float]]:
    """Aggregate factor pressures into one world-pressure score P.

    Grouped voting: mean signed pressure per family, weighted mean over
    families (``pressure_agg="mean"``), or the single strongest family's
    score (``pressure_agg="max"`` — a Round-4 variant that avoids
    dilution when one family shouts and the rest are silent).
    ``family_weights`` order: prices, housing, labour, policy,
    external (matches :data:`FACTOR_FAMILIES` insertion order).
    """
    fams = list(FACTOR_FAMILIES)
    acc: dict[str, list[float]] = {f: [] for f in fams}
    for s in signals:
        if s.available and s.family in acc:
            acc[s.family].append(s.pressure)
    scores = {f: float(sum(v) / len(v)) if v else 0.0 for f, v in acc.items()}
    w = list(family_weights) if family_weights is not None else [1.0] * len(fams)
    if pressure_agg == "max":
        best_i = max(range(len(fams)), key=lambda i: abs(scores[fams[i]]))
        p = float(scores[fams[best_i]] * w[best_i] / (abs(w[best_i]) or 1.0))
        return p, scores
    total = float(sum(abs(x) for x in w)) or 1.0
    p = float(sum(scores[f] * w[i] for i, f in enumerate(fams)) / total)
    return p, scores


def _diag(name: str, diags: Sequence[DiagnosticSignal]) -> DiagnosticSignal | None:
    for d in diags:
        if d.name == name and d.available:
            return d
    return None


class DeterministicOrchestrator:
    """Auditable decision engine: mini-agent signals -> decision."""

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.cfg = config or OrchestratorConfig()

    # -- FORECAST mode ----------------------------------------------------

    def decide_forecast(
        self,
        signals: Sequence[FactorSignal],
        diags: Sequence[DiagnosticSignal],
        val_winner: str,
        val_winner_beats_naive: bool,
        v2_q50: float,
        naive_q50: float,
    ) -> OrchestratorTrace:
        """Choose naive / v2 / blend from corroboration + vetoes.

        Policy (pre-registered, exp75 Round 1):
        1. Veto: stability or disagreement says MORE uncertainty with
           confidence >= veto_conf, or a changepoint fired -> naive.
        2. Strong world-pressure (|P| >= tau_high) corroborated by the
           validation winner pointing the same way as P -> v2.
        3. Strong pressure contradicting the validation winner -> blend
           (hedge the disagreement).
        4. Strong pressure with no validation support -> naive (do-no-harm:
           exp73 D's lesson — no overrides on vibes alone).
        5. Weak pressure + validation winner -> blend; else naive.
        """
        cfg = self.cfg
        p, fam_scores = world_pressure(signals, cfg.family_weights, cfg.pressure_agg)
        veto: str | None = None
        stab = _diag("stability", diags)
        dis = _diag("disagreement", diags)
        if stab is not None and stab.direction == 1 and stab.confidence >= cfg.veto_conf:
            veto = f"stability_veto(conf={stab.confidence:.2f})"
        if dis is not None and dis.direction == 1 and dis.confidence >= cfg.veto_conf:
            veto = f"disagreement_veto(conf={dis.confidence:.2f})"
        v2_dir = 1 if v2_q50 > naive_q50 else (-1 if v2_q50 < naive_q50 else 0)
        p_dir = 1 if p > 0 else (-1 if p < 0 else 0)

        if veto is not None:
            return OrchestratorTrace("forecast", "naive", p, fam_scores, veto, f"abstain: {veto}")
        if abs(p) >= cfg.tau_high:
            if val_winner_beats_naive and v2_dir != 0 and v2_dir == p_dir:
                return OrchestratorTrace(
                    "forecast",
                    "v2",
                    p,
                    fam_scores,
                    None,
                    f"corroborated: |P|={abs(p):.2f}>=tau_high, v2 points with P",
                )
            if val_winner_beats_naive and v2_dir != 0 and v2_dir != p_dir:
                return OrchestratorTrace(
                    "forecast",
                    "blend",
                    p,
                    fam_scores,
                    None,
                    f"hedge: |P|={abs(p):.2f} strong but contradicts v2 direction",
                )
            return OrchestratorTrace(
                "forecast",
                "naive",
                p,
                fam_scores,
                None,
                "do-no-harm: strong world pressure but no validation support",
            )
        if abs(p) >= cfg.tau_low and val_winner_beats_naive:
            return OrchestratorTrace(
                "forecast",
                "blend",
                p,
                fam_scores,
                None,
                f"weak pressure |P|={abs(p):.2f} with validation winner -> blend",
            )
        return OrchestratorTrace(
            "forecast",
            "naive",
            p,
            fam_scores,
            None,
            f"quiet world (|P|={abs(p):.2f}<tau_low) -> naive",
        )

    # -- UNCERTAINTY mode --------------------------------------------------

    def decide_uncertainty(
        self,
        diags: Sequence[DiagnosticSignal],
        signals: Sequence[FactorSignal],
        alpha: float,
        beta: float,
    ) -> tuple[float, OrchestratorTrace]:
        """Interval-width multiplier from uncertainty agents + news risk.

        m = 1 + alpha * U + beta * R, clipped to [1, width_cap], where
        U = mean(stability, disagreement) up-confidence and R = max
        high-risk factor confidence. (alpha, beta) are fit on validation
        origins to hit the coverage target.
        """
        cfg = self.cfg
        u_vals: list[float] = []
        for name in ("stability", "disagreement"):
            d = _diag(name, diags)
            if d is not None:
                u_vals.append(d.confidence if d.direction == 1 else 0.0)
        u = float(sum(u_vals) / len(u_vals)) if u_vals else 0.0
        r = float(max([s.confidence for s in signals if s.available and s.risk == "high"] or [0.0]))
        m = float(min(cfg.width_cap, max(1.0, 1.0 + alpha * u + beta * r)))
        trace = OrchestratorTrace(
            "uncertainty",
            "v2",
            0.0,
            {},
            None,
            f"m={m:.2f} (U={u:.2f}, R={r:.2f}, alpha={alpha}, beta={beta})",
            extra={"multiplier": m, "U": u, "R": r},
        )
        return m, trace

    @staticmethod
    def apply_multiplier(grid: np.ndarray, m: float) -> np.ndarray:
        """Expand a (9, H) grid around q50 by factor m."""
        g = np.asarray(grid, dtype=float)
        q50 = g[4:5, :]
        return np.asarray(q50 + m * (g - q50), dtype=float)


def fit_uncertainty_params(
    val_actuals: Sequence[np.ndarray],
    val_base_grids: Sequence[np.ndarray],
    val_diags: Sequence[Sequence[DiagnosticSignal]],
    val_signals: Sequence[Sequence[FactorSignal]],
    coverage_target: float = 0.80,
    max_width_ratio: float = 1.5,
) -> tuple[float, float]:
    """Grid-search (alpha, beta) on validation.

    Round-2 redesign (diagnosed in exp75 Round 1): the original rule
    picked the smallest params with validation coverage >= target and
    fell back to (1, 1) — maximum widening — when nothing qualified;
    both choices caused systematic test-time overshoot. New rule:

    1. If the *native* validation coverage already meets the target,
       return (0, 0) — never widen a calibrated base.
    2. Otherwise pick the params whose validation coverage is *closest*
       to the target (band objective) among mean widths within the cap,
       breaking ties toward smaller params.
    3. If nothing widens at all (quiet validation diagnostics), return
       (0, 0) — do no harm.
    """
    if not len(val_actuals):
        return (0.0, 0.0)
    orch = DeterministicOrchestrator()
    native_widths: list[float] = []
    native_cov: float = 0.0
    for act, grid in zip(val_actuals, val_base_grids, strict=True):
        g = np.asarray(grid, dtype=float)
        lo, hi = np.minimum(g[0], g[8]), np.maximum(g[0], g[8])
        native_cov += float(np.mean((np.asarray(act, dtype=float) >= lo)
                                    & (np.asarray(act, dtype=float) <= hi)))
        native_widths.append(float(np.mean(hi - lo)))
    native_cov /= len(val_actuals)
    native_mean_width = float(np.mean(native_widths))
    if native_cov >= coverage_target:
        return (0.0, 0.0)
    best: tuple[float, float] | None = None
    best_key: tuple[float, float] | None = None
    for alpha in (0.0, 0.25, 0.5, 1.0):
        for beta in (0.0, 0.25, 0.5, 1.0):
            covs: list[float] = []
            widths: list[float] = []
            for act, grid, diags, sigs in zip(
                val_actuals, val_base_grids, val_diags, val_signals, strict=True
            ):
                m, _ = orch.decide_uncertainty(diags, sigs, alpha, beta)
                g = orch.apply_multiplier(grid, m)
                lo, hi = np.minimum(g[0], g[8]), np.maximum(g[0], g[8])
                covs.append(float(np.mean((act >= lo) & (act <= hi))))
                widths.append(float(np.mean(hi - lo)))
            cov = float(np.mean(covs))
            wr = float(np.mean(widths) / (native_mean_width + 1e-12))
            if wr > max_width_ratio:
                continue
            key = (abs(cov - coverage_target), alpha + beta)
            if best_key is None or key < best_key:
                best_key, best = key, (alpha, beta)
    return best if best is not None else (0.0, 0.0)


# ---------------------------------------------------------------------------
# LLM orchestrator variant (arm C): the LLM reads the mini-agent signals.
# ---------------------------------------------------------------------------

_LLM_SCHEMA = (
    '{"choice": "naive" | "v2" | "blend", '
    '"shift": "none" | "up" | "down", '
    '"confidence": "low" | "medium" | "high", '
    '"reason": "one sentence"}'
)


def build_llm_prompt(
    bed: str,
    segment: str,
    asof_month: str,
    horizon: int,
    signals: Sequence[FactorSignal],
    diags: Sequence[DiagnosticSignal],
    naive_q50: float,
    v2_q50: float,
    val_summary: str,
) -> str:
    """Prompt for the LLM orchestrator: signals in, structure out."""
    fam_lines: dict[str, list[str]] = {}
    for s in signals:
        if s.available and abs(s.pressure) > 0.01:
            fam_lines.setdefault(s.family, []).append(
                f"- {s.category}: direction {s.direction:+d}/2, "
                f"pressure {s.pressure:+.2f}, conf {s.confidence:.2f}. {s.reason}"
            )
    fam_block = (
        "\n".join(f"[{fam}]\n" + "\n".join(lines) for fam, lines in fam_lines.items())
        or "(no active factor signals)"
    )
    diag_block = "\n".join(
        f"- {d.name}: direction {d.direction:+d}, conf {d.confidence:.2f}. {d.reason}"
        for d in diags
    )
    return (
        f"You are the ORCHESTRATOR of a forecasting decision engine, not a forecaster.\n"
        f"Bed: {bed} / segment {segment}. Origin month: {asof_month}, horizon: {horizon}.\n"
        f"Base forecasts (step-1 medians): naive={naive_q50:.6f}, v2={v2_q50:.6f}.\n"
        f"Validation evidence: {val_summary}\n\n"
        f"FACTOR AGENT SIGNALS (each owns one factor; synthesis is YOUR job):\n{fam_block}\n\n"
        f"DIAGNOSTIC AGENT SIGNALS (about the series/model):\n{diag_block}\n\n"
        f"Decide: choose naive (safe), v2 (trust the statistical router), or blend "
        f"(hedge). Optionally request a small shift of the chosen forecast's quantiles "
        f"up/down (the shift SIZE is fixed by calibration, never by you).\n"
        f"Prefer naive when signals are noisy, contradictory, or the world is unstable.\n"
        f"Reply with ONLY this JSON object, no other text:\n{_LLM_SCHEMA}"
    )


def parse_llm_decision(text: str) -> dict[str, Any]:
    """Validate the LLM's JSON into a decision; malformed -> safe default."""
    default = {
        "choice": "naive",
        "shift": "none",
        "confidence": "low",
        "reason": "LLM output malformed; abstaining to naive",
    }
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        d = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return dict(default)
    if not isinstance(d, dict):
        return dict(default)
    choice = str(d.get("choice", "naive")).lower()
    shift = str(d.get("shift", "none")).lower()
    conf = str(d.get("confidence", "low")).lower()
    if choice not in ("naive", "v2", "blend"):
        return dict(default)
    if shift not in ("none", "up", "down"):
        shift = "none"
    if conf not in ("low", "medium", "high"):
        conf = "low"
    return {
        "choice": choice,
        "shift": shift,
        "confidence": conf,
        "reason": str(d.get("reason", ""))[:300],
    }


class LLMOrchestratorCache:
    """File cache keyed by SHA-256 of the prompt.

    LLM calls are the expensive part; factor signals are month-stamped so
    reruns are free.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def get(self, prompt: str) -> dict[str, Any] | None:
        return self._data.get(self._key(prompt))

    def put(self, prompt: str, decision: Mapping[str, Any]) -> None:
        self._data[self._key(prompt)] = dict(decision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1))


class LLMOrchestrator:
    """LLM variant of the orchestrator (arm C).

    The LLM reads the mini-agent signal bundle and decides
    {choice, shift}; the shift SIZE comes from validation-residual
    calibration, never from the LLM.
    """

    def __init__(
        self,
        client: Any,
        cache: LLMOrchestratorCache,
        shift_sd_frac: float = 0.5,
    ) -> None:
        self.client = client
        self.cache = cache
        self.shift_sd_frac = shift_sd_frac

    def decide(
        self,
        bed: str,
        segment: str,
        asof_month: str,
        horizon: int,
        signals: Sequence[FactorSignal],
        diags: Sequence[DiagnosticSignal],
        naive_q50: float,
        v2_q50: float,
        val_summary: str,
        val_resid_sd: float,
    ) -> tuple[dict[str, Any], float]:
        """Return (decision dict, shift size). Cached by prompt hash."""
        prompt = build_llm_prompt(
            bed,
            segment,
            asof_month,
            horizon,
            signals,
            diags,
            naive_q50,
            v2_q50,
            val_summary,
        )
        hit = self.cache.get(prompt)
        if hit is not None:
            decision = dict(hit)
        else:
            decision = parse_llm_decision(self.client.complete(prompt))
            self.cache.put(prompt, decision)
        shift = 0.0
        if decision["shift"] == "up":
            shift = self.shift_sd_frac * val_resid_sd
        elif decision["shift"] == "down":
            shift = -self.shift_sd_frac * val_resid_sd
        return decision, float(shift)
