"""Tests for the exp75 orchestrator (decision engine)."""

from __future__ import annotations

import numpy as np
import pytest

from td_one.agent.factor_agents import DiagnosticSignal, FactorSignal
from td_one.agent.orchestrator import (
    DeterministicOrchestrator,
    LLMOrchestrator,
    LLMOrchestratorCache,
    build_llm_prompt,
    fit_uncertainty_params,
    parse_llm_decision,
    world_pressure,
)


def _fs(category, family, pressure, conf=0.8, direction=2):
    return FactorSignal(
        category=category,
        family=family,
        month="2026-01",
        direction=direction,
        confidence=conf,
        pressure=pressure,
        trend_3m=0.0,
        risk="med",
        reason="test",
        available=True,
    )


def _ds(name, direction=0, conf=0.6):
    return DiagnosticSignal(
        name=name, direction=direction, confidence=conf, pressure=0.0, reason="test", available=True
    )


def test_world_pressure_max_agg_picks_strongest_family():
    # One family shouts (+0.9), the rest are silent: mean aggregation
    # dilutes it to ~0.18; max aggregation keeps 0.9.
    sigs = [
        FactorSignal("inflation", "prices", "2026-01", 1, 0.9, 0.9, 0.0, "high", "x", True),
    ]
    p_mean, _ = world_pressure(sigs)
    p_max, _ = world_pressure(sigs, pressure_agg="max")
    assert abs(p_mean) < 0.3
    assert p_max == pytest.approx(0.9)



    sigs = [
        _fs("inflation", "prices", 0.8),
        _fs("food_prices", "prices", 0.4),
        _fs("central_bank_rates", "policy", -0.9),
    ]
    p, scores = world_pressure(sigs)
    assert scores["prices"] == pytest.approx(0.6)
    assert scores["policy"] == pytest.approx(-0.9)
    assert scores["housing"] == 0.0
    # equal weights over 5 families
    assert p == pytest.approx((0.6 - 0.9) / 5.0)


def test_veto_forces_naive():
    orch = DeterministicOrchestrator()
    sigs = [_fs("inflation", "prices", 0.9)]
    diags = [_ds("stability", direction=1, conf=0.9), _ds("disagreement", 0, 0.2)]
    t = orch.decide_forecast(
        sigs, diags, val_winner="v2", val_winner_beats_naive=True, v2_q50=1.0, naive_q50=0.0
    )
    assert t.choice == "naive" and t.veto is not None


def _strong_sigs(pressure=1.0):
    """Signals across 3 families -> |P| = 0.6 >= tau_high."""
    return [
        _fs("inflation", "prices", pressure),
        _fs("central_bank_rates", "policy", pressure),
        _fs("labour", "labour", pressure),
    ]


def test_corroborated_strong_pressure_picks_v2():
    orch = DeterministicOrchestrator()
    sigs = _strong_sigs()
    p, _ = world_pressure(sigs)
    assert abs(p) >= 0.30  # 3 families at full pressure, equal weights
    diags = [_ds("stability", 0, 0.3), _ds("disagreement", 0, 0.2)]
    t = orch.decide_forecast(
        sigs, diags, val_winner="v2", val_winner_beats_naive=True, v2_q50=1.5, naive_q50=1.0
    )
    assert t.choice == "v2" and "corroborated" in t.reason


def test_contradicting_pressure_hedges_to_blend():
    orch = DeterministicOrchestrator()
    sigs = _strong_sigs()  # P up, strong (0.6)
    diags = [_ds("stability", 0, 0.3), _ds("disagreement", 0, 0.2)]
    t = orch.decide_forecast(
        sigs, diags, val_winner="v2", val_winner_beats_naive=True, v2_q50=0.5, naive_q50=1.0
    )  # v2 points DOWN
    assert t.choice == "blend" and "hedge" in t.reason


def test_strong_pressure_no_validation_support_abstains():
    orch = DeterministicOrchestrator()
    sigs = _strong_sigs()
    diags = [_ds("stability", 0, 0.3), _ds("disagreement", 0, 0.2)]
    t = orch.decide_forecast(
        sigs, diags, val_winner="naive", val_winner_beats_naive=False, v2_q50=1.5, naive_q50=1.0
    )
    assert t.choice == "naive"  # do-no-harm: no overrides on vibes alone


def test_quiet_world_abstains():
    orch = DeterministicOrchestrator()
    sigs = [_fs("inflation", "prices", 0.05)]
    diags = [_ds("stability", 0, 0.3), _ds("disagreement", 0, 0.2)]
    t = orch.decide_forecast(
        sigs, diags, val_winner="v2", val_winner_beats_naive=True, v2_q50=1.5, naive_q50=1.0
    )
    assert t.choice == "naive"


def test_uncertainty_multiplier_bounds():
    orch = DeterministicOrchestrator()
    diags = [_ds("stability", 1, 1.0), _ds("disagreement", 1, 1.0)]
    m, t = orch.decide_uncertainty(diags, [], alpha=1.0, beta=1.0)
    assert 1.0 <= m <= 2.5
    m0, _ = orch.decide_uncertainty([_ds("stability", 0, 0.0)], [], 1.0, 1.0)
    assert m0 == 1.0


def test_apply_multiplier_identity_and_expand():
    g = np.tile(np.linspace(0, 8, 9)[:, None], (1, 3)).astype(float)
    out = DeterministicOrchestrator.apply_multiplier(g, 1.0)
    assert np.allclose(out, g)
    out2 = DeterministicOrchestrator.apply_multiplier(g, 2.0)
    assert np.allclose(out2[4], g[4])  # q50 fixed
    assert np.allclose(out2[8] - out2[0], 2 * (g[8] - g[0]))


def test_fit_uncertainty_params_returns_pair():
    acts = [np.array([0.0]) for _ in range(20)]
    grids = []
    for _ in range(20):
        g = np.tile(np.linspace(-1, 1, 9)[:, None], (1, 1))
        grids.append(g)
    diags = [[_ds("stability", 1, 0.9), _ds("disagreement", 0, 0.1)] for _ in range(20)]
    sigs = [[] for _ in range(20)]
    a, b = fit_uncertainty_params(acts, grids, diags, sigs)
    assert isinstance(a, float) and isinstance(b, float)


def test_fit_uncertainty_never_widens_calibrated_base():
    # Native coverage = 1.0 (actual 0 inside [-1, 1] grids) >= 0.80 target:
    # even with noisy diagnostics the fit must return (0, 0).
    acts = [np.array([0.0]) for _ in range(20)]
    grids = [np.tile(np.linspace(-1, 1, 9)[:, None], (1, 1)) for _ in range(20)]
    diags = [[_ds("stability", 1, 0.9), _ds("disagreement", 1, 0.9)] for _ in range(20)]
    a, b = fit_uncertainty_params(acts, grids, diags, [[] for _ in range(20)])
    assert (a, b) == (0.0, 0.0)


def test_fit_uncertainty_quiet_validation_do_no_harm():
    # Undercovering base (actual 2.0 outside [-1, 1]) but validation
    # diagnostics are quiet (U = R = 0): no (alpha, beta) can widen, so
    # the fit must return (0, 0) — not the old (1, 1) fallback.
    acts = [np.array([2.0]) for _ in range(20)]
    grids = [np.tile(np.linspace(-1, 1, 9)[:, None], (1, 1)) for _ in range(20)]
    diags = [[_ds("stability", 0, 0.0), _ds("disagreement", 0, 0.0)] for _ in range(20)]
    a, b = fit_uncertainty_params(acts, grids, diags, [[] for _ in range(20)])
    assert (a, b) == (0.0, 0.0)


def test_fit_uncertainty_rescues_undercovering_base():
    # Undercovering base + active stability diagnostics: some widening
    # must be chosen (params not both zero). Actual 1.2 is reachable:
    # U = 0.45 (mean of stability 0.9, disagreement 0) and alpha = 0.5
    # gives m = 1.225 -> q90 = 1.225 > 1.2.
    acts = [np.array([1.2]) for _ in range(20)]
    grids = [np.tile(np.linspace(-1, 1, 9)[:, None], (1, 1)) for _ in range(20)]
    diags = [[_ds("stability", 1, 0.9), _ds("disagreement", 0, 0.0)] for _ in range(20)]
    a, b = fit_uncertainty_params(acts, grids, diags, [[] for _ in range(20)])
    assert a + b > 0.0


def test_parse_llm_decision_valid():
    d = parse_llm_decision('{"choice": "v2", "shift": "up", "confidence": "high", "reason": "x"}')
    assert d["choice"] == "v2" and d["shift"] == "up"


def test_parse_llm_decision_malformed_defaults_safe():
    d = parse_llm_decision("not json at all")
    assert d["choice"] == "naive" and d["shift"] == "none"
    d2 = parse_llm_decision('{"choice": "cheat"}')
    assert d2["choice"] == "naive"


def test_llm_prompt_contains_signals():
    sigs = [_fs("inflation", "prices", 0.5)]
    p = build_llm_prompt(
        "osfi",
        "total",
        "2026-01",
        1,
        sigs,
        [_ds("stability")],
        1.0,
        1.1,
        "v2 beats naive on validation",
    )
    assert "inflation" in p and "naive" in p and "ONLY this JSON" in p


class _FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.text


def test_llm_orchestrator_cache(tmp_path):
    cache = LLMOrchestratorCache(tmp_path / "c.json")
    client = _FakeClient('{"choice": "blend", "shift": "none", "confidence": "med", "reason": "r"}')
    orch = LLMOrchestrator(client, cache)
    sigs = [_fs("inflation", "prices", 0.5)]
    kw = {
        "bed": "osfi",
        "segment": "total",
        "asof_month": "2026-01",
        "horizon": 1,
        "signals": sigs,
        "diags": [_ds("stability")],
        "naive_q50": 1.0,
        "v2_q50": 1.1,
        "val_summary": "v2 wins",
        "val_resid_sd": 0.01,
    }
    d1, _ = orch.decide(**kw)
    d2, _ = orch.decide(**kw)
    assert d1["choice"] == "blend" and d2["choice"] == "blend"
    assert client.calls == 1  # second call served from cache


def test_llm_orchestrator_shift_size_from_validation():
    cache = LLMOrchestratorCache.__new__(LLMOrchestratorCache)
    cache._data = {}
    cache.path = None  # type: ignore[assignment]
    import types

    def put(self, prompt, decision):
        pass

    cache.put = types.MethodType(put, cache)  # type: ignore[method-assign]
    client = _FakeClient('{"choice": "v2", "shift": "up", "confidence": "high", "reason": "r"}')
    orch = LLMOrchestrator(client, cache, shift_sd_frac=0.5)
    d, shift = orch.decide(
        "osfi", "total", "2026-01", 1, [], [], 1.0, 1.1, "v2 wins", val_resid_sd=0.04
    )
    assert shift == pytest.approx(0.02)  # 0.5 * 0.04; never from the LLM
