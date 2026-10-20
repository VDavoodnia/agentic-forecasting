"""Tests for the Layer-2 meta-router (exp73).

Contract: deterministic, sequential (walk-forward) decisions over
precomputed candidate quantile grids. Do-no-harm by construction:
naive is the default and any override must clear a validation margin.
Every decision emits a JSON-able trace. PIT: a decision at origin o
depends only on origins < o and on y[:o].
"""

from __future__ import annotations

import numpy as np

from td_one.agent.meta_router import MetaRouter, MetaRouterConfig

Z9 = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28])


def _grid(center: float, sigma: float = 0.1) -> np.ndarray:
    return (center + Z9 * sigma).reshape(9, 1)


def _scenario(
    n: int = 40,
    naive_bias: float = 0.0,
    v2_bias: float = 0.0,
    lockin_effect: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[int], dict[str, list[np.ndarray]], list[np.ndarray], np.ndarray]:
    """Synthetic segment whose candidates are grids around the last value.

    Optionally the actual deviates from the naive centre by
    ``lockin_effect * lockin`` (the exogenous signal the base forecast
    cannot see).
    """
    rng = np.random.default_rng(seed)
    lockin = rng.normal(0, 1, n + 40)
    y = np.zeros(n + 40)
    for i in range(1, n + 40):
        # Lock-in effect is a stationary part of the DGP from the start —
        # no artificial volatility break at the eval-window boundary.
        y[i] = y[i - 1] + lockin_effect * lockin[i] + rng.normal(0, 0.05)
    origins = list(range(40, 40 + n))
    grids: dict[str, list[np.ndarray]] = {"naive": [], "v2": []}
    actuals: list[np.ndarray] = []
    for o in origins:
        grids["naive"].append(_grid(y[o - 1] + naive_bias))
        grids["v2"].append(_grid(y[o - 1] + v2_bias))
        actuals.append(np.array([y[o]]))
    return y, origins, grids, actuals, lockin


def test_cold_start_defaults_to_naive() -> None:
    y, origins, grids, actuals, _ = _scenario(v2_bias=-0.5)
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res = router.run(y, origins, grids, actuals)
    first = res.traces[:12]
    assert all(t["choice"] == "naive" for t in first)
    assert all(t["reason"] == "cold_start" for t in first)


def test_override_when_candidate_dominates() -> None:
    """v2 unbiased vs naive biased +0.5: Layer-2 must switch to v2.

    After the cold start, the validation winner is v2 by a wide margin.
    """
    y, origins, grids, actuals, _ = _scenario(naive_bias=0.5, v2_bias=0.0)
    router = MetaRouter(MetaRouterConfig(min_val=12, margin=0.02))
    res = router.run(y, origins, grids, actuals)
    late = res.traces[20:]
    assert sum(t["choice"] == "v2" for t in late) >= int(0.8 * len(late))
    assert np.mean(res.crps["layer2"]) < np.mean(res.candidate_crps["naive"])


def test_do_no_harm_when_naive_is_best() -> None:
    y, origins, grids, actuals, _ = _scenario(naive_bias=0.0, v2_bias=0.4)
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res = router.run(y, origins, grids, actuals)
    assert all(t["choice"] == "naive" for t in res.traces[12:])


def test_trace_is_complete_and_jsonable() -> None:
    import json

    y, origins, grids, actuals, _ = _scenario()
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res = router.run(y, origins, grids, actuals)
    t = res.traces[-1]
    for key in ("origin_index", "choice", "reason", "evidence"):
        assert key in t
    ev = t["evidence"]
    for key in ("val_crps", "val_n", "changepoint", "diagnostics", "router_audit", "intervals"):
        assert key in ev
    json.dumps(res.traces)  # must not raise


def test_pit_future_corruption_does_not_change_past_decisions() -> None:
    y, origins, grids, actuals, lockin = _scenario(seed=3)
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res1 = router.run(y, origins, grids, actuals)
    # Corrupt everything strictly after origin index 25.
    grids2 = {k: [g.copy() for g in v] for k, v in grids.items()}
    actuals2 = [a.copy() for a in actuals]
    y2 = y.copy()
    for i in range(26, len(origins)):
        grids2["v2"][i] = _grid(999.0)
        actuals2[i] = np.array([999.0])
        y2[origins[i]] = 999.0
    res2 = router.run(y2, origins, grids2, actuals2)
    assert [t["choice"] for t in res1.traces[:26]] == [t["choice"] for t in res2.traces[:26]]


def test_lockin_correction_recovers_exogenous_effect() -> None:
    """Actual = naive centre + 2*lockin + noise.

    Arm C (no lock-in) cannot see it; arm D (lock-in) must score better.
    """
    y, origins, grids, actuals, lockin = _scenario(lockin_effect=2.0, seed=5)
    lockin_by_origin = {o: float(lockin[o]) for o in origins}
    cfg_c = MetaRouterConfig(min_val=12, use_lockin=False)
    cfg_d = MetaRouterConfig(min_val=12, use_lockin=True)
    res_c = MetaRouter(cfg_c).run(y, origins, grids, actuals)
    res_d = MetaRouter(cfg_d, lockin_fn=lambda o: lockin_by_origin[o]).run(
        y, origins, grids, actuals
    )
    assert np.mean(res_d.crps["layer2"]) < np.mean(res_c.crps["layer2"])
    shifts = [t["evidence"]["lockin"]["shift"] for t in res_d.traces[15:]]
    assert np.mean(np.abs(shifts)) > 0.5  # corrections actually fire


def test_lockin_evidence_gate_blocks_noise_correction() -> None:
    """Gated lock-in must stay silent on noise and fire on signal.

    With lock-in values unrelated to the outcome, the gated variant
    (D2) must stay at ~zero shift — while the ungated variant (D) fits
    noise and wanders.
    """
    y, origins, grids, actuals, lockin = _scenario(lockin_effect=0.0, seed=8)
    lockin_by_origin = {o: float(lockin[o]) for o in origins}
    res_g = MetaRouter(
        MetaRouterConfig(min_val=12, use_lockin=True, lockin_evidence_gate=True),
        lockin_fn=lambda o: lockin_by_origin[o],
    ).run(y, origins, grids, actuals)
    shifts = [abs(t["lockin_shift"]) for t in res_g.traces if t["evidence"]["lockin"]]
    assert shifts, "gate info should be present"
    assert float(np.mean([s > 0 for s in shifts])) < 0.5  # mostly silent
    # And where the effect is real, the gate still lets it through.
    y2, origins2, grids2, actuals2, lockin2 = _scenario(lockin_effect=2.0, seed=5)
    by_origin2 = {o: float(lockin2[o]) for o in origins2}
    res_g2 = MetaRouter(
        MetaRouterConfig(min_val=12, use_lockin=True, lockin_evidence_gate=True),
        lockin_fn=lambda o: by_origin2[o],
    ).run(y2, origins2, grids2, actuals2)
    res_c2 = MetaRouter(MetaRouterConfig(min_val=12)).run(y2, origins2, grids2, actuals2)
    assert np.mean(res_g2.crps["layer2"]) < np.mean(res_c2.crps["layer2"])


def test_changepoint_abstention() -> None:
    """A level break inside the last 12 months forces naive.

    Even when v2 dominated the validation window.
    """
    rng = np.random.default_rng(9)
    y = np.concatenate(
        [
            np.cumsum(rng.normal(0, 0.05, 70)),
        ]
    )
    y = y.copy()
    origins = list(range(40, 70))
    grids: dict[str, list[np.ndarray]] = {"naive": [], "v2": []}
    actuals: list[np.ndarray] = []
    for o in origins:
        grids["naive"].append(_grid(y[o - 1] + 0.5))
        grids["v2"].append(_grid(y[o]))
        actuals.append(np.array([y[o]]))
    # Plant the break at the final origin's window: shift y[58:] up.
    y[58:] += 3.0
    for i, o in enumerate(origins):
        if o >= 58:
            grids["naive"][i] = _grid(y[o - 1] + 0.5)
            grids["v2"][i] = _grid(y[o])
            actuals[i] = np.array([y[o]])
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res = router.run(y, origins, grids, actuals)
    last = res.traces[-1]
    assert last["choice"] == "naive"
    assert last["reason"] == "changepoint_abstain"


def test_blend_candidate_available() -> None:
    y, origins, grids, actuals, _ = _scenario(naive_bias=0.3, v2_bias=-0.3)
    router = MetaRouter(MetaRouterConfig(min_val=12))
    res = router.run(y, origins, grids, actuals)
    assert "blend_v2_naive" in res.candidate_crps
    # The blend centres between the two biased candidates -> near-actual.
    assert np.mean(res.candidate_crps["blend_v2_naive"]) <= np.mean(res.candidate_crps["naive"])
