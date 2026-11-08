"""Command line.

  python -m fiscalsim baseline                  deterministic baseline and its fit to the Trustees
  python -m fiscalsim run [--sims N] [--seed S] [--policies k1,k2] [--solvency 0.8] [--out DIR]
  python -m fiscalsim list                      the named policy options
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np


def _baseline() -> int:
    from .calibrate import baseline_model, baseline_paths, calibrate
    from .metrics import actuarial_balance

    _, fit = calibrate()
    print("Fit to the 2026 Trustees (model vs target):")
    for k, v in fit.items():
        if isinstance(v, tuple):
            unit, suffix = (1, "%") if "rate" in k else (1e12, "T")
            print(f"  {k:28s} {v[0] / unit:9.2f}{suffix}  {v[1] / unit:9.2f}{suffix}")
        elif "year" in k:
            print(f"  {k:28s} {v:.2f}")
        else:
            print(f"  {k:28s} {v:.4g}" if isinstance(v, float) else f"  {k:28s} {v}")
    r = baseline_model().run(baseline_paths())
    ab = actuarial_balance(r)
    print(f"\n75-year actuarial balance: {ab['actuarial_balance'][0]:.2f}% of taxable payroll "
          f"(income {ab['income_rate'][0]:.2f}%, cost {ab['cost_rate'][0]:.2f}%)")
    print(f"Combined OASDI reserves run out: {r.ss_depletion_year[0]:.1f}; HI: {r.hi_depletion_year[0]:.1f}")
    return 0


def _list() -> int:
    from .scenarios import library

    for k, p in library().items():
        print(f"{k:22s} {p.name}: {p.description}")
    return 0


def _run(args) -> int:
    from .assumptions import trustees_2026
    from .report import write_report
    from .runner import Run, run_policies
    from .scenarios import library, solvency_packages
    from .stochastic import draw_paths
    from .calibrate import baseline_model
    from .metrics import summarize

    lib = library()
    keys = list(lib) if args.policies == "all" else ["current_law"] + args.policies.split(",")
    unknown = [k for k in keys if k not in lib]
    if unknown:
        print(f"unknown policies: {', '.join(unknown)} (see `python -m fiscalsim list`)")
        return 2
    t = time.time()
    runs = run_policies({k: lib[k] for k in keys}, sims=args.sims, seed=args.seed)
    det = run_policies({"current_law": lib["current_law"]}, sims=1)
    print(f"{len(runs)} policies x {args.sims:,} paths in {time.time() - t:.1f}s")
    solvency = {}
    if args.solvency:
        paths = draw_paths(trustees_2026(), args.sims, seed=args.seed)
        base = runs["current_law"].results
        for k, pol in solvency_packages(paths, args.solvency).items():
            r = baseline_model(pol).run(paths)
            solvency[k] = Run(k, pol, r, summarize(r, discount=base.discount_rate))
        print(f"solvency packages sized in {time.time() - t:.1f}s")
    out = Path(args.out)
    path = write_report(runs, det, solvency, out, args.sims, args.seed)
    print(f"wrote {path} and {out / 'summary.csv'}")
    if args.dashboard:
        from .dashboard import export

        paths = draw_paths(trustees_2026(), args.sims, seed=args.seed)
        files = export(runs, solvency, paths, Path(args.dashboard), args.sims, args.seed)
        print(f"wrote {len(files)} dashboard datasets to {args.dashboard}")
    return 0


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")
    ap = argparse.ArgumentParser(prog="python -m fiscalsim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("--sims", type=int, default=2000)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--policies", default="all", help="comma-separated keys, or 'all'")
    run.add_argument("--solvency", type=float, default=0.8,
                     help="size packages that stay solvent through 2100 in this share of paths (0 to skip)")
    run.add_argument("--out", default="out")
    run.add_argument("--dashboard", default="", help="also write the dashboard's JSON datasets to this folder")
    args = ap.parse_args()
    if args.cmd == "baseline":
        return _baseline()
    if args.cmd == "list":
        return _list()
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
