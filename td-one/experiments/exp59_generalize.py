"""exp59 — rung 6: generalizability of the LLM integration on Tourism/M3.

Tests whether the rung-4 V1 prompt + Gemini Flash (the winner from exp58)
generalizes to the two new benchmark datasets.

Design (pilot, 60 series total):
- 30 Tourism Monthly series (h=24) + 30 M3 Monthly series (h=18)
- For each series: LLM predicts h=1 (first test step) via V1 prompt
- Compare LLM CRPS vs classical best CRPS (from exp60/exp61 records, h=1)
- Hybrid = per-series best of {LLM, classical best} (oracle upper bound)

If LLM wins here too, the integration generalizes. If not, report honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.llm_rungs.prompts import (  # noqa: E402
    LEVELS_9,
    serialize_history,
    system_prompt_forecaster,
    user_prompt_generic,
    words_stats_block,
)
from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
TD_ONE = Path(__file__).resolve().parents[1]

N_SERIES_PER_DATASET = 30


def parse_tsf(path: Path) -> dict[str, tuple[str, list[float]]]:
    """Parse .tsf file; return {series_id: (start_date, values)}."""
    series = {}
    in_data = False
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line == "@data":
            in_data = True
            continue
        if not in_data or not line or line.startswith("#"):
            continue
        # Format: ID:start_timestamp:values
        parts = line.split(":")
        sid = parts[0]
        # start_timestamp is parts[1] + parts[2] (date + time with colons)
        # values start after the timestamp
        # Find where values begin: after "YYYY-MM-DD HH-MM-SS"
        # Simpler: split by ":" and find the part with commas
        for i, p in enumerate(parts):
            if "," in p:
                start = ":".join(parts[1:i])
                values = [float(x) for x in p.split(",")]
                # Handle remaining parts (shouldn't be any)
                series[sid] = (start, values)
                break
    return series


def build_v1_prompt(
    series_desc: str,
    values: list[float],
    dates: list[str],
    forecast_date: str,
) -> tuple[str, str]:
    """Build rung-4 V1 prompt (metadata + history stats in words)."""
    n = len(values)
    last_value = values[-1]
    last_date = dates[-1]
    # 12m and 3m changes
    if n >= 13:
        ref_12m_val = values[-13]
        if ref_12m_val != 0:
            change_12m_pct = 100 * (last_value - ref_12m_val) / abs(ref_12m_val)
        else:
            change_12m_pct = 0.0
        ref_12m = dates[-13][:7]
    else:
        change_12m_pct, ref_12m = 0.0, last_date[:7]
    if n >= 4:
        ref_3m_val = values[-4]
        if ref_3m_val != 0:
            change_3m_pct = 100 * (last_value - ref_3m_val) / abs(ref_3m_val)
        else:
            change_3m_pct = 0.0
        ref_3m = dates[-4][:7]
    else:
        change_3m_pct, ref_3m = 0.0, last_date[:7]
    window_12m = values[-12:] if n >= 12 else values
    max_12m = max(window_12m)
    min_12m = min(window_12m)

    words_block = words_stats_block(
        last_value=last_value,
        last_date=last_date,
        change_12m_pct=change_12m_pct,
        ref_12m=ref_12m,
        change_3m_pct=change_3m_pct,
        ref_3m=ref_3m,
        max_12m=max_12m,
        min_12m=min_12m,
        as_of=last_date,
    )
    history_str = serialize_history(values, dates, precision=2)
    user = user_prompt_generic(
        task_line=f"{series_desc} 1-month-ahead forecast.",
        series_desc=series_desc,
        units="index",
        frequency="MS",
        history_str=history_str,
        forecast_start=forecast_date,
        forecast_end=forecast_date,
        n_steps=1,
        suffix="\n\nRecent behavior (in words):\n" + words_block,
    )
    return system_prompt_forecaster(), user


def parse_llm_grid(text: str) -> np.ndarray | None:
    """Parse LLM JSON response to (11,) grid."""
    try:
        data = json.loads(text)
        fc = data["forecasts"][0]
        levels = ["q05", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90", "q95"]
        return np.array([float(fc[k]) for k in levels])
    except Exception:  # noqa: BLE001
        return None


def crps_quantile(quantiles, actual, levels):
    return float(
        np.mean(
            [
                2.0 * ((actual - q) * lv - (actual - q) * (actual < q))
                for q, lv in zip(quantiles, levels, strict=True)
            ]
        )
    )


def month_add(ym: str, months: int) -> str:
    """Add months to YYYY-MM string."""
    y, m = map(int, ym.split("-")[:2])
    m += months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def run_dataset(
    name: str,
    tsf_path: Path,
    results_path: Path,
    horizon: int,
    series_desc_template: str,
    client: VectorProxyClient,
) -> dict:
    """Run LLM on subset of series; return per-series results."""
    print(f"\n=== {name} ===")
    series = parse_tsf(tsf_path)
    print(f"  Loaded {len(series)} series")

    # Load classical records for comparison
    exp_records = json.loads(results_path.read_text())["records"]
    rec_by_id = {r["series_id"]: r for r in exp_records}

    # Select subset (first N for determinism)
    sids = sorted(series.keys())[:N_SERIES_PER_DATASET]
    print(f"  Using {len(sids)} series")

    results = []
    for i, sid in enumerate(sids):
        start_date, values = series[sid]
        if len(values) <= horizon:
            continue
        # Split: train = all except last horizon, test_h1 = first of last horizon
        train_vals = values[:-horizon]
        actual_h1 = values[-horizon]
        # Build date strings
        start_ym = start_date[:7]
        dates = [month_add(start_ym, j) + "-01" for j in range(len(train_vals))]
        forecast_ym = month_add(start_ym, len(train_vals))

        # Build V1 prompt
        series_desc = series_desc_template.format(sid=sid)
        system, user = build_v1_prompt(series_desc, train_vals, dates, forecast_ym)

        # Call LLM
        try:
            text = client.complete(user, system=system, json_mode=True, timeout_s=120.0)
            grid11 = parse_llm_grid(text)
            if grid11 is None:
                print(f"  [{i+1}/{len(sids)}] {sid}: parse failed")
                continue
            # Subset to 9 levels (0.10..0.90 are indices 1..9)
            grid9 = grid11[1:10]
            llm_crps = crps_quantile(grid9, actual_h1, np.array(LEVELS_9))
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(sids)}] {sid}: LLM failed: {e}")
            continue

        # Classical best (h=1 from exp records)
        # NOTE: exp60/61 records store NORMALIZED quantiles/actuals.
        # Denormalize to raw scale for fair comparison with LLM.
        rec = rec_by_id.get(sid)
        if rec is None:
            continue
        norm_lo = float(rec.get("norm_lo", 0.0))
        norm_span = float(rec.get("norm_span", 1.0))
        classical_crps_h1 = {}
        for m, q in rec["quantiles"].items():
            q_arr = np.asarray(q, dtype=float)
            # Denormalize: raw = norm * span + lo
            q_raw = q_arr[:, 0] * norm_span + norm_lo
            classical_crps_h1[m] = crps_quantile(q_raw, actual_h1, np.array(LEVELS_9))
        best_classical = (
            min(classical_crps_h1.values()) if classical_crps_h1 else float("nan")
        )
        best_model = (
            min(classical_crps_h1, key=lambda m: classical_crps_h1[m])
            if classical_crps_h1
            else "none"
        )

        results.append({
            "series_id": sid,
            "llm_crps": llm_crps,
            "best_classical_crps": best_classical,
            "best_classical_model": best_model,
            "classical_all": classical_crps_h1,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sids)}] done")

    return {
        "dataset": name,
        "n_series": len(results),
        "results": results,
    }


def main():
    client = VectorProxyClient(model="flash-lite", max_tokens=2048)

    tourism = run_dataset(
        "tourism_monthly",
        TD_ONE / "data" / "tourism" / "tourism_monthly_dataset.tsf",
        EXP_DIR / "exp60_results.json",
        horizon=24,
        series_desc_template="Tourism Monthly series {sid}, monthly visitor counts",
        client=client,
    )
    m3 = run_dataset(
        "m3_monthly",
        TD_ONE / "data" / "m3mon" / "m3_monthly_dataset.tsf",
        EXP_DIR / "exp61_results.json",
        horizon=18,
        series_desc_template="M3 Monthly series {sid}, monthly economic time series",
        client=client,
    )

    output = {"tourism": tourism, "m3": m3}
    out_path = EXP_DIR / "exp59_results.json"
    # Don't store full classical_all to keep file small
    for ds in output.values():
        for r in ds["results"]:
            r.pop("classical_all", None)
    out_path.write_text(json.dumps(output, indent=1))
    print(f"\nWrote {out_path}")

    for ds_name, ds in [("tourism", tourism), ("m3", m3)]:
        res = ds["results"]
        if not res:
            print(f"\n{ds_name}: no results")
            continue
        llm = np.array([r["llm_crps"] for r in res])
        clas = np.array([r["best_classical_crps"] for r in res])
        print(f"\n=== {ds_name} (n={len(res)}) ===")
        print(f"  LLM mean CRPS: {np.mean(llm):.4f}")
        print(f"  Classical best mean CRPS: {np.mean(clas):.4f}")
        print(f"  LLM win rate: {np.mean(llm < clas):.2f}")
        print(f"  Mean diff (LLM - classical): {np.mean(llm - clas):.4f}")


if __name__ == "__main__":
    main()
