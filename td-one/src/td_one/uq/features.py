"""PIT-safe feature extraction for the exp74 uncertainty study.

Numeric features come only from the pre-origin context and the PIT-safe
bank of known cases. Story features come from exp74's story-pipeline runs
(keyed by origin month); macro features from the Macro Signal Service's
``signals_asof``. Missing story/macro features are filled with neutral
values and ``*_available`` flags — the meta-model learns what the flags
mean.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12

SHIFT_TYPE_MAP = {"none": 0, "rate": 1, "policy": 2, "other": 3}

#: Macro categories surfaced individually; the rest enter as aggregates.
KEY_MACRO_CATS = ("inflation", "central_bank_rates", "food_prices", "labour")

RISK_MAP = {"low": 0, "med": 1, "high": 2}


def numeric_features(
    context: np.ndarray,
    h: int,
    levels: np.ndarray,
    q: np.ndarray,
    bank_known: list,
) -> dict[str, float]:
    """Pre-origin numeric features for one (origin, h) case.

    Args:
        context: Series values strictly before the origin.
        h: Horizon step (1-based).
        levels: Quantile levels of the native grid.
        q: Native quantile values at this step.
        bank_known: PIT-safe known cases, chronological.

    """
    ctx = np.asarray(context, dtype=float)
    i10 = int(np.argmin(np.abs(levels - 0.1)))
    i90 = int(np.argmin(np.abs(levels - 0.9)))
    q10, q90 = float(q[i10]), float(q[i90])
    width = max(q90 - q10, _EPS)

    d1 = np.diff(ctx)
    vol_6 = float(np.std(d1[-6:])) if len(d1) >= 2 else 0.0
    vol_12 = float(np.std(d1[-12:])) if len(d1) >= 2 else 0.0
    scale = float(np.std(ctx[-24:])) + _EPS if len(ctx) >= 2 else 1.0

    same_h = [c for c in bank_known if c.h == h]
    err3 = [abs(c.y - c.median) for c in same_h[-3:]]
    err6 = [abs(c.y - c.median) for c in same_h[-6:]]
    recent = bank_known[-6:]
    miss_streak = sum(
        1.0 for c in recent if c.y < c.interval_edges()[1] or c.y > c.interval_edges()[2]
    )

    if len(ctx) >= 12:
        regime_shift = (float(np.mean(ctx[-6:])) - float(np.mean(ctx[-12:-6]))) / scale
        x = np.arange(6, dtype=float)
        y6 = ctx[-6:]
        slope_6 = float(np.polyfit(x, y6, 1)[0] / (scale + _EPS) if np.std(y6) > _EPS else 0.0)
    else:
        regime_shift, slope_6 = 0.0, 0.0

    # width growth across horizons needs the full grid; caller passes it via
    # feats merge — default to 1.0 here.
    return {
        "h": float(h),
        "log_width": float(np.log(width)),
        "width_rel": float(width / scale),
        "vol_6": float(vol_6),
        "vol_12": float(vol_12),
        "err_mean3": float(np.mean(err3)) if err3 else 0.0,
        "err_mean6": float(np.mean(err6)) if err6 else 0.0,
        "err_count3": float(len(err3)),
        "err_count6": float(len(err6)),
        "miss_streak": float(miss_streak),
        "regime_shift": float(regime_shift),
        "slope_6": float(slope_6),
        "width_growth": 1.0,  # overwritten by the driver when the grid is known
    }


def story_features(signal: dict | None) -> dict[str, float]:
    """Story-pipeline classification -> features.

    ``signal`` is the raw signal dict from exp74_story_signals.json, or
    None if unavailable.
    """
    if not signal:
        return {
            "st_available": 0.0,
            "st_shift": 0.0,
            "st_direction": 0.0,
            "st_magnitude": 0.0,
            "st_months_from_now": -1.0,
            "st_timing_conf": 0.0,
            "st_trend_count": 0.0,
        }
    direction = {"down": -1.0, "up": 1.0}.get(str(signal.get("direction", "")), 0.0)
    mag = {"none": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}.get(
        str(signal.get("magnitude_bucket", "none")), 0.0
    )
    trends = signal.get("driving_trends") or []
    mfn = signal.get("months_from_now")
    conf = signal.get("timing_confidence")
    return {
        "st_available": 1.0,
        "st_shift": float(SHIFT_TYPE_MAP.get(str(signal.get("shift_type", "none")), 3)),
        "st_direction": direction,
        "st_magnitude": mag,
        "st_months_from_now": float(mfn) if isinstance(mfn, (int, float)) else -1.0,
        "st_timing_conf": float(conf) if isinstance(conf, (int, float)) else 0.0,
        "st_trend_count": float(len(trends)),
    }


def macro_features(signals: dict | None) -> dict[str, float]:
    """Macro Signal Service ``signals_asof`` output -> aggregate features.

    ``signals`` maps category -> MacroSignal (or a plain dict with the same
    fields). None means no signals were available as of this origin.
    """
    base = {
        "mac_available": 0.0,
        "mac_n_high": 0.0,
        "mac_max_abs_dir": 0.0,
        "mac_mean_cert": 0.0,
        "mac_inflation": 0.0,
        "mac_rates": 0.0,
        "mac_food": 0.0,
        "mac_labour": 0.0,
    }
    if not signals:
        return base
    items = list(signals.values())
    if not items:
        return base

    def _get(s: object, k: str, default: float = 0.0) -> float:
        v = s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)
        return float(v) if isinstance(v, (int, float)) else default

    dirs = [_get(s, "direction") for s in items]
    certs = [_get(s, "certainty") for s in items]
    risks = [
        (s.get("risk", "low") if isinstance(s, dict) else getattr(s, "risk", "low")) for s in items
    ]
    base.update(
        {
            "mac_available": 1.0,
            "mac_n_high": float(sum(1 for r in risks if r == "high")),
            "mac_max_abs_dir": float(max((abs(d) for d in dirs), default=0.0)),
            "mac_mean_cert": float(np.mean(certs)) if certs else 0.0,
        }
    )
    cat_map = {
        "inflation": "mac_inflation",
        "central_bank_rates": "mac_rates",
        "food_prices": "mac_food",
        "labour": "mac_labour",
    }
    sig_by_cat: dict[str, object] = {}
    for s in items:
        c = s.get("category", "") if isinstance(s, dict) else getattr(s, "category", "")
        sig_by_cat[str(c)] = s
    for cat, feat in cat_map.items():
        if cat in sig_by_cat:
            base[feat] = _get(sig_by_cat[cat], "direction")
    return base
