"""Datasets for the shareable dashboard (dashboard/index.html).

`python -m fiscalsim run --dashboard DIR` writes one JSON file of row objects
per dataset. Every figure the page shows comes from these files; labels that
are text on the page (a year or "after 2100") are written here as text so the
page shows the data's own value.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .calibrate import baseline_model, baseline_paths, calibrate
from .metrics import actuarial_balance
from .runner import Run

GROUPS = {
    "current_law": "Current law",
    "payroll_plus_1": "Revenue", "payroll_plus_2": "Revenue", "payroll_phased_3": "Revenue",
    "cap_90": "Revenue", "cap_none_credit": "Revenue", "cap_none_no_credit": "Revenue", "donut_400k": "Revenue",
    "fra_69": "Benefits", "fra_70": "Benefits", "chained_cpi": "Benefits", "progressive_cut": "Benefits",
    "cut_all_now": "Benefits",
    "gr_borrow": "General revenue", "gr_income_tax": "General revenue",
    "immigration_plus_50": "Immigration",
    "package_revenue": "Package", "package_benefits": "Package", "package_balanced": "Package",
}
PCTS = (5, 10, 25, 50, 75, 90, 95)


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _year_label(y: float) -> str:
    return "after 2100" if not np.isfinite(y) or y >= 2101 else str(int(np.floor(y)))


def _bands(option: str, series: str, years: np.ndarray, arr: np.ndarray, digits: int = 2) -> list[dict]:
    q = np.percentile(arr, PCTS, axis=0)
    return [
        {"option": option, "series": series, "year": int(y), **{f"p{p}": _r(q[i, j], digits) for i, p in enumerate(PCTS)}}
        for j, y in enumerate(years)
    ]


def _option_row(key: str, run: Run, base: Run, base_balance: np.ndarray) -> dict:
    r, s = run.results, run.summary
    bal = actuarial_balance(r, base.results.discount_rate)["actuarial_balance"]
    change = bal - base_balance
    dep = np.where(np.isfinite(r.ss_depletion_year), r.ss_depletion_year, 2101.0)
    yrs = r.years

    def at(arr, y):
        return arr[:, int(y - yrs[0])]

    p10, p50, p90 = np.percentile(dep, [10, 50, 90])
    return {
        "key": key,
        "name": run.policy.name,
        "group": GROUPS.get(key, "Package"),
        "description": run.policy.description,
        "balance_p10": _r(np.percentile(bal, 10)),
        "balance_p50": _r(np.percentile(bal, 50)),
        "balance_p90": _r(np.percentile(bal, 90)),
        "change_p50": _r(np.percentile(change, 50)),
        "depletion_p10": _r(p10), "depletion_p50": _r(p50), "depletion_p90": _r(p90),
        "runs_out": _year_label(p50), "runs_out_p10": _year_label(p10), "runs_out_p90": _year_label(p90),
        "solvent_2050": _r(s["share_solvent_through_2050"] * 100, 0),
        "solvent_2100": _r(s["share_solvent_through_2100"] * 100, 0),
        "payable_2035": _r(np.median(at(r.payable_ratio, 2035)) * 100, 0),
        "payable_2075": _r(np.median(at(r.payable_ratio, 2075)) * 100, 0),
        "debt_2050": _r(np.median(at(r.debt, 2050) / at(r.gdp, 2050)) * 100, 0),
        "debt_2050_p10": _r(np.percentile(at(r.debt, 2050) / at(r.gdp, 2050), 10) * 100, 0),
        "debt_2050_p90": _r(np.percentile(at(r.debt, 2050) / at(r.gdp, 2050), 90) * 100, 0),
    }


def export(runs: dict[str, Run], solvency: dict[str, Run], paths, out_dir: Path, sims: int, seed: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = runs["current_law"]
    base_balance = actuarial_balance(base.results)["actuarial_balance"]
    det = baseline_model().run(baseline_paths())

    run_info = [{
        "sims": sims, "seed": seed, "generated": date.today().isoformat(),
        "baseline": "2026 Trustees intermediate assumptions", "options": len(runs) - 1,
        "baseline_runs_out": round(float(det.ss_depletion_year[0]), 2),
        "baseline_balance": _r(actuarial_balance(det)["actuarial_balance"][0]),
    }]

    options = [_option_row(k, run, base, base_balance) for k, run in runs.items()]

    rows = []
    for key, run in runs.items():
        r = run.results
        yrs = r.years
        ratio = r.ss_assets[:, :-1] / r.ss_cost[:, 1:] * 100
        rows += _bands(key, "trust_fund_ratio", yrs[1:], ratio, 1)
        rows += _bands(key, "payable_share", yrs[1:], r.payable_ratio[:, 1:] * 100, 1)
        sel = yrs <= 2075
        rows += _bands(key, "debt_gdp", yrs[sel], (r.debt / r.gdp * 100)[:, sel], 1)
    r = base.results
    rows += _bands("current_law", "cost_rate", r.years[1:], (r.ss_cost / r.taxable_payroll * 100)[:, 1:], 2)
    rows += _bands("current_law", "income_rate", r.years[1:],
                   ((r.ss_payroll_tax + r.ss_benefit_tax_scheduled) / r.taxable_payroll * 100)[:, 1:], 2)

    depletion = []
    for key, run in runs.items():
        dep = run.results.ss_depletion_year
        n = len(dep)
        late = ~np.isfinite(dep) | (dep >= 2101)
        early = dep[~late]
        if len(early):
            edges = np.arange(np.floor(early.min()), np.floor(early.max()) + 0.5, 0.25)
            counts = np.histogram(early, bins=np.append(edges, edges[-1] + 0.25))[0]
            depletion += [{"option": key, "bin": float(e), "share": _r(c / n * 100, 2)}
                          for e, c in zip(edges, counts) if c > 0]
        if late.any():
            depletion.append({"option": key, "bin": 2101.0, "share": _r(late.mean() * 100, 2)})

    packages = []
    for key, run in solvency.items():
        p = run.policy
        if p.all_benefit_cut > 0:
            lever, value, label = "Cut to every benefit", p.all_benefit_cut * 100, f"{p.all_benefit_cut * 100:.1f}%"
        else:
            lever, value = "Payroll tax rise", p.payroll_rate_increase * 100
            label = f"+{value:.2f} points"
        s = run.summary
        packages.append({
            "key": key, "name": p.name, "description": p.description.split(" Solved:")[0],
            "lever": lever, "required": _r(value), "required_label": label,
            "solvent_2100": _r(s["share_solvent_through_2100"] * 100, 0),
            "debt_2050": _r(s["debt_pct_gdp_2050"][1], 0),
        })

    _, fit = calibrate()
    fit_rows = [
        {"measure": "OASDI summarized income rate, 2026-2100", "unit": "% of taxable payroll",
         "model": _r(fit["summarized_income_rate_pct"][0]), "trustees": _r(fit["summarized_income_rate_pct"][1])},
        {"measure": "OASDI summarized cost rate, 2026-2100", "unit": "% of taxable payroll",
         "model": _r(fit["summarized_cost_rate_pct"][0]), "trustees": _r(fit["summarized_cost_rate_pct"][1])},
        {"measure": "Present value of HI taxable payroll", "unit": "$ trillions",
         "model": _r(fit["pv_hi_payroll"][0] / 1e12, 1), "trustees": _r(fit["pv_hi_payroll"][1] / 1e12, 1)},
        {"measure": "Present value of GDP", "unit": "$ trillions",
         "model": _r(fit["pv_gdp"][0] / 1e12, 1), "trustees": _r(fit["pv_gdp"][1] / 1e12, 1)},
        {"measure": "Year HI reserves run out", "unit": "year",
         "model": int(np.floor(fit["hi_depletion_year"])), "trustees": 2033},
    ]

    years = np.arange(2026, 2101)
    sl = slice(1, None)
    det_paths = baseline_paths()

    def avg(a):
        return a[:, sl].mean(axis=1)

    drivers = []
    for name, unit, arr, d, scale, digits in [
        ("Fertility in 2050", "children per woman", paths.tfr[:, 25], det_paths.tfr[0, 25], 1, 2),
        ("Death rates, average yearly decline", "%", avg(paths.mortality_decline), avg(det_paths.mortality_decline)[0], 100, 2),
        ("Net immigration, yearly average", "thousands", avg(paths.net_immigration), avg(det_paths.net_immigration)[0], 1e-3, 0),
        ("Inflation (CPI-W), yearly average", "%", avg(paths.cpi), avg(det_paths.cpi)[0], 100, 2),
        ("Real wage growth, yearly average", "%", avg(paths.real_wage), avg(det_paths.real_wage)[0], 100, 2),
        ("Real interest rate, yearly average", "%", avg(paths.real_interest), avg(det_paths.real_interest)[0], 100, 2),
        ("Unemployment, yearly average", "%", avg(paths.unemployment), avg(det_paths.unemployment)[0], 100, 2),
    ]:
        drivers.append({"driver": name, "unit": unit, "baseline": _r(d * scale, digits),
                        "p10": _r(np.percentile(arr, 10) * scale, digits),
                        "p90": _r(np.percentile(arr, 90) * scale, digits)})
    del years

    files = {
        "run_info.json": run_info, "options.json": options, "paths.json": rows, "depletion.json": depletion,
        "packages.json": packages, "fit.json": fit_rows, "drivers.json": drivers,
    }
    written = []
    for name, data in files.items():
        p = out_dir / name
        p.write_text(json.dumps(data, separators=(",", ":")))
        written.append(p)
    return written
