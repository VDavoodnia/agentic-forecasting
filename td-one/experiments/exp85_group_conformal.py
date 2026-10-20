"""exp85 testbed 2: shift-adaptive conformal on the group forecaster.

Reuses exp80's harness (run_worlds / summarize) verbatim. Arms:

  A     shipped exp79 default (trailing-mean center + Gaussian spread
        + exp80b thin/dead fallback), conformal OFF — reference
  C90   exp80's winner, as a reference: conditional conformal @0.90
  RST   conformal @level + adaptive "reset"
  DEC   conformal @level + adaptive "decay" (half-life --hl)
  CP    conformal @level + adaptive "cp" (half-life --hl, pre --pref)
  BC    conformal @level + adaptive "bc" (lambda --lam, n0 --n0)
  CPBC  conformal @level + adaptive "cp_bc"

Pre-registered bar (exp85_report.md): on FRESH seeds (2024/555/9001),
S1@4sigma member-weighted pooled "group" cov80 >= 0.77 with control
CRPS harm vs A <= +2% and control cov80 in [0.77, 0.85].

Usage:
    .venv/bin/python experiments/exp85_group_conformal.py \
        --seeds fresh --tag r1 [--hl 6] [--pref 0.25] [--lam 0.95] \
        [--n0 12] [--minpost 5] [--level 0.8] [--cond]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e80 = _load("exp80_conformal_fallback", "exp80_conformal_fallback.py")

sys.path.insert(0, str(HERE.parent / "src"))
import td_one.agent.group_forecaster as gf  # noqa: E402

PROTOCOL = [7, 42, 123]
FRESH = [2024, 555, 9001]
WORLDS = [("control", None), ("S1_4.0", None)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["protocol", "fresh"], required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--seed-one", type=int, default=None, help="run a single seed only")
    ap.add_argument("--world-one", type=str, default=None, help="run a single world only")
    ap.add_argument("--rows-out", type=str, default=None, help="write raw rows JSON here")
    ap.add_argument("--merge", nargs="*", default=None, help="row JSONs to merge + summarize")
    ap.add_argument("--hl", type=float, default=6.0)
    ap.add_argument("--pref", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--n0", type=float, default=12.0)
    ap.add_argument("--minpost", type=int, default=5)
    ap.add_argument("--level", type=float, default=0.8)
    ap.add_argument("--detector", type=str, default="scan", choices=["scan", "cusum", "errors"])
    ap.add_argument("--cond", action="store_true", help="conditional vol scaling on adaptive arms")
    args = ap.parse_args()

    arm_names = ["A", "C90", "RST", "DEC", "CP", "BC", "CPBC"]
    if args.merge is not None:
        rows: list[dict] = []
        for path in args.merge:
            rows.extend(json.loads(Path(path).read_text())["rows"])
        summary = e80.summarize(rows, "A")
        out = {"params": vars(args), "summary": summary}
        out_name = f"exp85_group_results_{args.seeds}_{args.tag}.json"
        (HERE / out_name).write_text(json.dumps(out, indent=1))
        print(f"merged {len(rows)} rows -> {out_name}")
        _print_summary(summary, arm_names, args.seeds)
        return

    seeds = PROTOCOL if args.seeds == "protocol" else FRESH
    if args.seed_one is not None:
        seeds = [args.seed_one]
    common = {
        "conformal": True,
        "conformal_conditional": args.cond,
        "conformal_level": args.level,
        "conformal_half_life": args.hl,
        "conformal_pre_factor": args.pref,
        "conformal_min_post": args.minpost,
        "conformal_bias_lambda": args.lam,
        "conformal_bias_n0": args.n0,
        "conformal_detector": args.detector,
    }

    def adaptive(mode: str):
        return lambda: gf.v2_forecaster_fn(**common, conformal_adaptive=mode)  # noqa: E731

    factories = {
        "A": lambda: gf.v2_forecaster_fn(),
        "C90": lambda: gf.v2_forecaster_fn(
            conformal=True, conformal_conditional=True, conformal_level=0.90
        ),
        "RST": adaptive("reset"),
        "DEC": adaptive("decay"),
        "CP": adaptive("cp"),
        "BC": adaptive("bc"),
        "CPBC": adaptive("cp_bc"),
    }
    arm_names = list(factories)
    rows = e80.run_worlds(
        seeds,
        [w for w in WORLDS if args.world_one is None or w[0] == args.world_one],
        arm_names,
        factories,
    )
    if args.rows_out:
        Path(args.rows_out).write_text(json.dumps({"rows": rows}))
        print(f"wrote {len(rows)} rows -> {args.rows_out}")
        return
    summary = e80.summarize(rows, "A")
    out = {
        "params": vars(args),
        "summary": summary,
    }
    out_name = f"exp85_group_results_{args.seeds}_{args.tag}.json"
    (HERE / out_name).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_name} ({len(rows)} cell-results)")
    _print_summary(summary, arm_names, args.seeds)


def _print_summary(summary: dict, arm_names: list[str], label: str) -> None:
    for world in ("control", "S1_4.0"):
        if world not in summary:
            continue
        for scope in ("group", "global"):
            entry = summary[world][scope]
            print(f"\n== {label} / {world} / {scope} scope (vs A) ==")
            for arm in arm_names:
                if arm not in entry:
                    continue
                s = entry[arm]
                line = (
                    f"  {arm:5s} crps={s['crps_mw']:.5f} cov80={s['cov80']:.3f} "
                    f"cov90={s['cov90']:.3f} fb={s['fallback_rate']:.2f}"
                )
                if "vs_ref" in s:
                    v = s["vs_ref"]
                    line += f"  d={v['mean_diff']:+.5f} p={v['p']:.3g} win={v['win_rate']:.2f}"
                else:
                    line += "  (ref)"
                print(line)


if __name__ == "__main__":
    main()
