"""Distributional shift scenarios for TSFM red-teaming (Track 2, exp67).

A shift is a structural break at a known onset month T. Everything with
``t < T`` is bit-identical to the no-shift series (PIT-safe by construction:
the shift function never reads post-onset values to determine pre-onset
ones).

Severity ladder
---------------
``magnitude`` is in units of ``SHIFT_SIGMA_LOGIT`` (0.25 logit). At the base
attrition rate (~0.11, logit ~ -2.09), one unit moves the rate to ~0.136 —
a meaningful but not extreme move. The experiment sweeps
0.5 / 1 / 2 / 4 units ("0.5σ … 4σ") and reports breaking-point curves.

Two application layers
----------------------
*Series-level* (S1, S2, S4, S5-series, S6, S7): applied directly to a
monthly ``(n_renewals, n_switches)`` series via :func:`apply_shift`.
Fast, fully controlled — the primary red-team surface.

*DGP-level* (S2, S3, S5): applied inside the borrower simulator so the
shift propagates through the true mechanism:
- :func:`macro_shifts_for` converts S1/S2 specs to the simulator's
  ``macro_shifts`` dict (month-end date -> logit shift).
- :func:`beta_schedule_for` converts an S5 spec to a coefficient-override
  schedule consumed by ``MortgageSimulatorV2.generate_outcomes``
  (``beta_schedule`` hook).
- S3 (regional divergence) needs two regional series; see
  :func:`apply_regional_divergence`, which reuses the per-region
  aggregation from ``td_one.data_gen.group_series`` conceptually
  (portfolio = weighted mean of regional rates).
- S3 at the *group* level (exp72): :func:`apply_divergence` applies the
  same signed level shift group-by-group (region "A" up, region "B"
  down) so per-group series can be stress-tested individually.

Shift taxonomy (loan domain)
----------------------------
S1 rate shock:          level shift at T (history is pre-shock).
S2 policy intervention: partial jump with 2-month implementation lag
                        (mirrors events.py POLICY_SHOCK transmission).
S3 regional divergence: two regions shift in opposite directions; the
                        aggregate averages the move away.
S4 seasonal break:      renewal-volume seasonality changes amplitude
                        (and optionally phase) at T.
S5 correlation breakdown: the DGP's rate-sensitivity mechanism changes.
                        DGP-level: flips the sign of V2_BETA_RATE_SPREAD
                        via beta_schedule. Series-level proxy: the AR(1)
                        persistence of the logit noise flips sign
                        (dynamics break invisible to a fixed window).
S6 volatility regime:   residual scale inflates and counts overdisperse
                        at T (tests the conformal wrapper too).
S7 composition drift:    slow linear drift post-T (burnout/selection),
                        misread as a trend by univariate extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

#: One severity unit, in logit space.
SHIFT_SIGMA_LOGIT = 0.25

#: Valid shift-type codes.
SHIFT_TYPES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")

#: Base-series DGP constants (mirror scenarios.py's proven pattern).
_BASE_RATE = 0.11
_SEASONAL_AMP = 0.25
_AR_RHO = 0.5
_LOGIT_NOISE_SD = 0.08
_OVERDISPERSION = 30.0
_BASE_N = 100

#: S2: fraction of the portfolio affected by a targeted policy change.
_S2_COVERAGE = 0.6
#: S2: implementation lag in months (matches POLICY_SHOCK transmission).
_S2_LAG_MONTHS = 2


@dataclass(frozen=True)
class ShiftSpec:
    """Specification of a single distributional shift.

    Attributes:
        shift_type: One of "S1".."S7" (see module docstring).
        onset_month: 0-based month index where the shift takes effect.
            Months ``t < onset_month`` are never modified.
        magnitude: Severity in units of SHIFT_SIGMA_LOGIT (documented
            per type in :func:`apply_shift`).
        target: Optional cohort/region selector (e.g. "ON", "variable").
            ``None`` = portfolio-wide.
        params: Type-specific extras, e.g. ``{"new_amp": 0.0}`` for S4.

    """

    shift_type: str
    onset_month: int
    magnitude: float
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shift_type not in SHIFT_TYPES:
            raise ValueError(
                f"unknown shift_type {self.shift_type!r}; expected one of {SHIFT_TYPES}"
            )
        if self.onset_month < 0:
            raise ValueError("onset_month must be >= 0")
        if self.magnitude < 0:
            raise ValueError("magnitude must be >= 0")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)))


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-x)))


def generate_base_series(
    n_months: int,
    seed: int,
    base_rate: float = _BASE_RATE,
    seasonal_amp: float = _SEASONAL_AMP,
    ar_rho: float = _AR_RHO,
    logit_noise_sd: float = _LOGIT_NOISE_SD,
    overdispersion: float = _OVERDISPERSION,
    seasonal_break: tuple[int, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Generate a monthly (n_renewals, n_switches) series, no shift.

    Args:
        n_months: Series length.
        seed: Random seed.
        base_rate: Mean attrition rate.
        seasonal_amp: Renewal-volume seasonal amplitude.
        ar_rho: AR(1) coefficient of the logit noise.
        logit_noise_sd: Std of the logit-noise innovations.
        overdispersion: Beta concentration (lower = more overdispersed).
        seasonal_break: Optional ``(break_month, new_amp, phase_shift)`` —
            the volume seasonality switches amplitude/phase at
            ``break_month`` (the S4 mechanism, built into the base).

    Returns:
        Dict with ``n_renewals``, ``n_switches``, ``true_rates``,
        ``attrition_rate`` (observed switches/renewals).

    """
    if n_months < 1:
        raise ValueError("n_months must be >= 1")
    rng = np.random.default_rng(seed)
    t = np.arange(n_months, dtype=float)

    # Seasonal volume, with optional break.
    amp = np.full(n_months, seasonal_amp)
    phase = np.zeros(n_months)
    if seasonal_break is not None:
        b, new_amp, dphase = seasonal_break
        if not 0 <= b < n_months:
            raise ValueError("seasonal break month out of range")
        amp[b:] = new_amp
        phase[b:] = dphase
    seasonal = 1.0 + amp * np.sin(2 * np.pi * t / 12 + phase)
    ar_vol = np.zeros(n_months)
    for i in range(1, n_months):
        ar_vol[i] = 0.5 * ar_vol[i - 1] + rng.normal(0, 8)
    n_renewals = np.maximum(20, _BASE_N * seasonal + ar_vol)

    # True rate + AR(1) logit noise.
    logit_noise = np.zeros(n_months)
    for i in range(1, n_months):
        logit_noise[i] = ar_rho * logit_noise[i - 1] + rng.normal(0, logit_noise_sd)
    true_rates = _inv_logit(_logit(np.full(n_months, base_rate)) + logit_noise)

    # Beta-Binomial draws.
    alpha = true_rates * overdispersion
    beta = (1 - true_rates) * overdispersion
    p_drawn = rng.beta(alpha, beta)
    n_switches = rng.binomial(np.round(n_renewals).astype(int), p_drawn).astype(float)

    return {
        "n_renewals": n_renewals,
        "n_switches": n_switches,
        "true_rates": true_rates,
        "attrition_rate": n_switches / np.maximum(n_renewals, 1),
    }


def _redraw_switches(
    true_rates: np.ndarray,
    n_renewals: np.ndarray,
    months: np.ndarray,
    rng: np.random.Generator,
    overdispersion: float,
) -> np.ndarray:
    """Redraw switch counts for ``months`` from the (shifted) true rates.

    Returns an array aligned with ``months`` (length ``len(months)``).
    """
    out = np.zeros(len(months))
    for j, m in enumerate(months):
        p = float(np.clip(true_rates[m], 1e-6, 1 - 1e-6))
        a = p * overdispersion
        b = (1 - p) * overdispersion
        p_drawn = rng.beta(a, b)
        out[j] = rng.binomial(int(round(n_renewals[m])), p_drawn)
    return out


def apply_shift(
    series: dict[str, np.ndarray],
    spec: ShiftSpec,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Apply a series-level shift at ``spec.onset_month``.

    Only months ``t >= onset_month`` are modified; ``t < onset_month`` are
    returned bit-identical (PIT-safe by construction).

    Magnitude semantics per type (units of SHIFT_SIGMA_LOGIT = 0.25 logit):
    - S1: level shift of ``magnitude`` units at T.
    - S2: ``0.6 * magnitude`` units at T+2 (targeted cohort, 2-month lag).
    - S4: seasonality amplitude scales to ``max(0, 1 - magnitude/2)`` of
      the original at T (magnitude=2 kills it, 4 inverts it); optional
      ``params["phase_shift"]`` rotates the phase.
    - S5: AR(1) persistence flips sign at T (dynamics break); plus a small
      concurrent level shift of ``0.1 * magnitude`` units so the severity
      ladder varies something observable.
    - S6: logit-noise sd × (1 + 0.5·m) and overdispersion ÷ (1 + 0.5·m)
      for t >= T.
    - S7: linear drift of ``0.02 * magnitude`` logit per month for t >= T.
    - S3: not series-level — use :func:`apply_regional_divergence`.

    Args:
        series: Dict with ``n_renewals``, ``n_switches``, ``true_rates``.
        spec: Shift specification.
        rng: Random generator (for redrawing counts).

    Returns:
        New series dict; the input is not mutated.

    """
    n = len(series["true_rates"])
    if not 0 <= spec.onset_month < n:
        raise ValueError(f"onset_month {spec.onset_month} out of range [0, {n})")
    if spec.shift_type == "S3":
        raise ValueError("S3 needs two regions; use apply_regional_divergence")

    t_idx = np.arange(n)
    post = t_idx >= spec.onset_month
    m = spec.magnitude

    n_renewals = series["n_renewals"].copy()
    true_rates = series["true_rates"].copy()
    overdisp = _OVERDISPERSION
    logit_r = _logit(true_rates)

    if spec.shift_type == "S1":
        logit_r[post] += m * SHIFT_SIGMA_LOGIT
    elif spec.shift_type == "S2":
        lagged = t_idx >= spec.onset_month + _S2_LAG_MONTHS
        logit_r[lagged] += _S2_COVERAGE * m * SHIFT_SIGMA_LOGIT
    elif spec.shift_type == "S4":
        # Seasonal break: amplitude scales by (1 - m/2); m=2 kills the
        # seasonal pattern, m=4 inverts it. Optional phase_shift rotates.
        amp_factor = 1.0 - m / 2.0
        dphase = float(spec.params.get("phase_shift", 0.0))
        seasonal_post = 1.0 + (_SEASONAL_AMP * amp_factor) * np.sin(
            2 * np.pi * t_idx[post] / 12 + dphase
        )
        # Rescale renewals to the new seasonal shape, preserving the
        # pre-break mean level.
        if (~post).any():
            pre_mean = float(np.mean(n_renewals[~post]))
        else:
            pre_mean = float(np.mean(n_renewals))
        shape_mean = float(np.mean(seasonal_post))
        n_renewals[post] = np.maximum(
            20, _BASE_N * (seasonal_post / max(shape_mean, 1e-9)) * (pre_mean / _BASE_N)
        )
    elif spec.shift_type == "S5":
        # Dynamics break: re-simulate the logit noise post-T with flipped
        # persistence, anchored at the pre-break level.
        anchor = float(logit_r[spec.onset_month - 1]) if spec.onset_month > 0 else float(logit_r[0])
        noise = np.zeros(n)
        for i in range(spec.onset_month, n):
            prev = noise[i - 1] if i > spec.onset_month else 0.0
            noise[i] = -_AR_RHO * prev + rng.normal(0, _LOGIT_NOISE_SD)
        base_logit = _logit(np.full(n, _BASE_RATE))
        logit_r[post] = (
            base_logit[post] + (anchor - float(base_logit[spec.onset_month])) + noise[post]
        )
        logit_r[post] += 0.1 * m * SHIFT_SIGMA_LOGIT
    elif spec.shift_type == "S6":
        overdisp = _OVERDISPERSION / (1.0 + 0.5 * m)
        # Widen residuals: add extra mean-zero noise post-T.
        extra = rng.normal(0, 0.5 * m * _LOGIT_NOISE_SD, size=n)
        logit_r[post] += extra[post]
    elif spec.shift_type == "S7":
        drift = 0.02 * m * (t_idx[post] - spec.onset_month)
        logit_r[post] += drift
    else:  # pragma: no cover — validated in ShiftSpec
        raise ValueError(f"unhandled shift_type {spec.shift_type}")

    true_rates = _inv_logit(logit_r)
    n_switches = series["n_switches"].copy()
    n_switches[post] = _redraw_switches(true_rates, n_renewals, t_idx[post], rng, overdisp)
    return {
        "n_renewals": n_renewals,
        "n_switches": n_switches,
        "true_rates": true_rates,
        "attrition_rate": n_switches / np.maximum(n_renewals, 1),
    }


def apply_regional_divergence(
    series_a: dict[str, np.ndarray],
    series_b: dict[str, np.ndarray],
    spec: ShiftSpec,
    weight_a: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """S3: two regions shift in opposite directions; return the aggregate.

    Region A gets ``+magnitude`` units, region B ``-magnitude`` units at T.
    The portfolio aggregate (weighted mean of rates) averages the move away
    — the structural blind spot under test.

    Args:
        series_a: Per-region series dict for region A (shifts up).
        series_b: Per-region series dict for region B (shifts down).
        spec: ShiftSpec with shift_type "S3".
        weight_a: Portfolio weight of region A (0..1).
        rng: Random generator.

    Returns:
        Aggregate series dict (portfolio-level).

    """
    if spec.shift_type != "S3":
        raise ValueError("apply_regional_divergence requires shift_type 'S3'")
    if len(series_a["true_rates"]) != len(series_b["true_rates"]):
        raise ValueError("regional series must have equal length")
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    spec_a = ShiftSpec("S1", spec.onset_month, spec.magnitude, target="A", params=dict(spec.params))
    # Region A shifts up (S1 mechanics); region B shifts down.
    shifted_a = apply_shift(series_a, spec_a, rng)
    n = len(series_b["true_rates"])
    t_idx = np.arange(n)
    post = t_idx >= spec.onset_month
    logit_b = _logit(series_b["true_rates"].copy())
    logit_b[post] -= spec.magnitude * SHIFT_SIGMA_LOGIT
    true_b = _inv_logit(logit_b)
    n_sw_b = series_b["n_switches"].copy()
    n_sw_b[post] = _redraw_switches(
        true_b, series_b["n_renewals"], t_idx[post], rng, _OVERDISPERSION
    )
    shifted_b = {
        "n_renewals": series_b["n_renewals"].copy(),
        "n_switches": n_sw_b,
        "true_rates": true_b,
        "attrition_rate": n_sw_b / np.maximum(series_b["n_renewals"], 1),
    }
    n_r = weight_a * shifted_a["n_renewals"] + (1 - weight_a) * shifted_b["n_renewals"]
    n_s = weight_a * shifted_a["n_switches"] + (1 - weight_a) * shifted_b["n_switches"]
    true_r = weight_a * shifted_a["true_rates"] + (1 - weight_a) * shifted_b["true_rates"]
    return {
        "n_renewals": n_r,
        "n_switches": n_s,
        "true_rates": true_r,
        "attrition_rate": n_s / np.maximum(n_r, 1),
    }


def apply_divergence(
    series_by_group: dict[str, dict[str, np.ndarray]],
    region_of: dict[str, str],
    spec: ShiftSpec,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """S3 at the group level: signed level shift per group by region.

    Groups mapped to region ``"A"`` shift UP by ``spec.magnitude`` logit
    units at ``spec.onset_month``; groups mapped to ``"B"`` shift DOWN by
    the same amount (the S1 mechanics of :func:`apply_shift`, signed).
    Groups absent from ``region_of`` pass through unchanged (same
    arrays). The portfolio aggregate (sum over groups) averages the two
    moves toward each other — the structural blind spot under test in
    exp72 — while each group's own series shows its full move.

    Only months ``t >= onset_month`` are modified; ``t < onset_month``
    are bit-identical (PIT-safe by construction). Inputs are not mutated.

    Args:
        series_by_group: ``group_id -> series dict`` (the
            :func:`generate_base_series` shape).
        region_of: ``group_id -> "A" | "B"``.
        spec: ShiftSpec with shift_type "S3".
        rng: Random generator (for redrawing counts).

    Returns:
        New ``group_id -> series dict``.

    """
    if spec.shift_type != "S3":
        raise ValueError("apply_divergence requires shift_type 'S3'")
    bad = sorted(set(region_of.values()) - {"A", "B"})
    if bad:
        raise ValueError(f"region values must be 'A' or 'B', got {bad}")
    out: dict[str, dict[str, np.ndarray]] = {}
    for gid, series in series_by_group.items():
        region = region_of.get(gid)
        if region is None:
            out[gid] = {k: np.array(v, dtype=float) for k, v in series.items()}
            continue
        n = len(series["true_rates"])
        if not 0 <= spec.onset_month < n:
            raise ValueError(f"onset_month {spec.onset_month} out of range [0, {n})")
        if region == "A":
            out[gid] = apply_shift(
                series,
                ShiftSpec("S1", spec.onset_month, spec.magnitude, target=gid),
                rng,
            )
            continue
        # Region B: S1 mechanics with the opposite sign.
        t_idx = np.arange(n)
        post = t_idx >= spec.onset_month
        true_rates = series["true_rates"].copy()
        logit_r = _logit(true_rates)
        logit_r[post] -= spec.magnitude * SHIFT_SIGMA_LOGIT
        true_rates = _inv_logit(logit_r)
        n_switches = series["n_switches"].copy()
        n_switches[post] = _redraw_switches(
            true_rates, series["n_renewals"], t_idx[post], rng, _OVERDISPERSION
        )
        out[gid] = {
            "n_renewals": series["n_renewals"].copy(),
            "n_switches": n_switches,
            "true_rates": true_rates,
            "attrition_rate": n_switches / np.maximum(series["n_renewals"], 1),
        }
    return out


def macro_shifts_for(spec: ShiftSpec, month_dates: list[date]) -> dict[date, float]:
    """Convert an S1/S2 spec to the simulator's ``macro_shifts`` dict.

    Maps month-end dates -> logit shift. S1 applies at the onset month;
    S2 applies at onset + 2 (implementation lag), scaled by coverage.

    Args:
        spec: ShiftSpec with shift_type "S1" or "S2".
        month_dates: Simulator month-end dates (from ``_month_dates()``).

    Returns:
        Dict suitable for ``generate_outcomes(macro_shifts=...)``.

    """
    if spec.shift_type not in ("S1", "S2"):
        raise ValueError("macro_shifts_for supports S1/S2 only")
    if not 0 <= spec.onset_month < len(month_dates):
        raise ValueError("onset_month out of range for month_dates")
    if spec.shift_type == "S1":
        months = {spec.onset_month: spec.magnitude * SHIFT_SIGMA_LOGIT}
    else:
        lagged = min(spec.onset_month + _S2_LAG_MONTHS, len(month_dates) - 1)
        months = {lagged: _S2_COVERAGE * spec.magnitude * SHIFT_SIGMA_LOGIT}
    return {month_dates[i]: v for i, v in months.items()}


def beta_schedule_for(spec: ShiftSpec, n_months: int) -> dict[int, dict[str, float]]:
    """Convert an S5 spec to a DGP coefficient-override schedule.

    Flips the sign of ``V2_BETA_RATE_SPREAD`` (the rate-sensitivity
    mechanism) for all months ``t >= onset_month`` — the correlation
    breakdown. Consumed by ``MortgageSimulatorV2.generate_outcomes``
    via its ``beta_schedule`` hook.

    Args:
        spec: ShiftSpec with shift_type "S5".
        n_months: Total months (keys are 0-based month indices).

    Returns:
        ``{month_idx: {"V2_BETA_RATE_SPREAD": -0.4}}`` for idx >= onset.

    """
    if spec.shift_type != "S5":
        raise ValueError("beta_schedule_for supports S5 only")
    if not 0 <= spec.onset_month < n_months:
        raise ValueError("onset_month out of range")
    # Import here to avoid a hard dependency at module import time.
    from td_one.data_gen.simulator_v2 import V2_BETA_RATE_SPREAD

    return {
        i: {"V2_BETA_RATE_SPREAD": -V2_BETA_RATE_SPREAD} for i in range(spec.onset_month, n_months)
    }
