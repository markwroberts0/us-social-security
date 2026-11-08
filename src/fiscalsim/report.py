"""Charts and a self-contained HTML report for a set of Monte Carlo runs."""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibrate import calibrate  # noqa: E402
from .runner import Run  # noqa: E402

INK = "#1f2a24"
SOFT = "#5b6660"
ACCENT = "#2f6f4f"
WARN = "#b5532c"
BAND = "#2f6f4f"
GRID = "#d9dfdb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": SOFT, "axes.labelcolor": INK, "xtick.color": SOFT,
    "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 110,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fan(ax, years, arr, color, label, median_style="-"):
    for lo, hi, a in ((5, 95, 0.12), (10, 90, 0.18), (25, 75, 0.28)):
        ax.fill_between(years, np.percentile(arr, lo, axis=0), np.percentile(arr, hi, axis=0),
                        color=color, alpha=a, linewidth=0)
    ax.plot(years, np.median(arr, axis=0), median_style, color=color, linewidth=1.8, label=label)


def chart_trust_fund(run: Run, det: Run | None) -> str:
    r = run.results
    yrs = r.years
    ratio = np.zeros_like(r.ss_assets)
    ratio[:, 1:] = r.ss_assets[:, :-1] / r.ss_cost[:, 1:] * 100
    ratio[:, 0] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    _fan(ax, yrs[1:], ratio[:, 1:], ACCENT, "median")
    ax.set_xlim(2025, 2045)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.set_ylabel("Reserves at start of year, % of annual cost")
    ax.set_title("Trust fund ratio, current law", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    dep = r.ss_depletion_year[np.isfinite(r.ss_depletion_year)]
    ax.hist(dep, bins=np.arange(np.floor(dep.min()), np.ceil(dep.max()) + 0.25, 0.25), color=ACCENT, alpha=0.85)
    ax.set_title("Year reserves run out", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("Paths")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if det is not None:
        d0 = det.results.ss_depletion_year[0]
        ax.axvline(d0, color=WARN, linewidth=1.2)
        ax.text(d0, ax.get_ylim()[1] * 0.95, f" baseline {d0:.1f}", color=WARN, va="top", fontsize=9)
    return _png(fig)


def chart_rates(run: Run) -> str:
    r = run.results
    yrs = r.years[1:]
    cost = r.ss_cost[:, 1:] / r.taxable_payroll[:, 1:] * 100
    income = (r.ss_payroll_tax + r.ss_benefit_tax_scheduled)[:, 1:] / r.taxable_payroll[:, 1:] * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    _fan(ax, yrs, cost, WARN, "cost")
    _fan(ax, yrs, income, ACCENT, "income")
    ax.set_ylabel("% of taxable payroll")
    ax.set_title("Scheduled cost and income", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    _fan(ax, yrs, r.payable_ratio[:, 1:] * 100, ACCENT, "median")
    ax.set_ylim(40, 105)
    ax.set_ylabel("% of scheduled benefits")
    ax.set_title("Benefits payable under current law", loc="left", fontsize=11, color=INK)
    return _png(fig)


def chart_policies(runs: dict[str, Run], keys: list[str]) -> str:
    base_ab = np.array(runs["current_law"].summary["actuarial_balance"])
    labels, med, lo, hi = [], [], [], []
    for k in keys:
        s = runs[k].summary
        labels.append(runs[k].policy.name)
        med.append(s["actuarial_balance"][1])
        lo.append(s["actuarial_balance"][0])
        hi.append(s["actuarial_balance"][2])
    order = np.argsort(med)
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(keys) + 1.2))
    y = np.arange(len(keys))
    ax.hlines(y, np.array(lo)[order], np.array(hi)[order], color=ACCENT, alpha=0.5, linewidth=6)
    ax.plot(np.array(med)[order], y, "o", color=ACCENT)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvline(base_ab[1], color=WARN, linewidth=1, linestyle="--")
    ax.text(base_ab[1], len(keys) - 0.4, " current law", color=WARN, fontsize=9)
    ax.set_xlabel("75-year actuarial balance, % of taxable payroll (dot: median; bar: 10th-90th percentile)")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def chart_debt(runs: dict[str, Run], keys: list[str]) -> str:
    """Current law as a fan; the alternatives as their median paths only."""
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    styles = [(ACCENT, "-"), (WARN, "-"), ("#6a5acd", "--"), ("#8a8f3c", "-")]
    for i, (k, (col, ls)) in enumerate(zip(keys, styles)):
        r = runs[k].results
        sel = r.years <= 2060
        ratio = (r.debt / r.gdp * 100)[:, sel]
        if i == 0:
            _fan(ax, r.years[sel], ratio, col, runs[k].policy.name)
        else:
            ax.plot(r.years[sel], np.median(ratio, axis=0), ls, color=col, linewidth=1.8, label=runs[k].policy.name)
    ax.set_ylabel("Debt held by the public, % of GDP")
    ax.set_title("Federal debt", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper left")
    return _png(fig)


def _fmt_range(v, fmt="{:.1f}") -> str:
    return f"{fmt.format(v[1])} <span class=r>({fmt.format(v[0])} to {fmt.format(v[2])})</span>"


def summary_table(runs: dict[str, Run]) -> pd.DataFrame:
    rows = []
    base = runs["current_law"].summary["actuarial_balance"][1]
    for k, run in runs.items():
        s = run.summary
        rows.append({
            "key": k, "policy": run.policy.name, "description": run.policy.description,
            "balance_median": s["actuarial_balance"][1], "balance_p10": s["actuarial_balance"][0],
            "balance_p90": s["actuarial_balance"][2], "balance_change_median": s["actuarial_balance"][1] - base,
            "depletion_median": s["depletion_year"][1], "depletion_p10": s["depletion_year"][0],
            "depletion_p90": s["depletion_year"][2],
            "p_solvent_2050": s["share_solvent_through_2050"], "p_solvent_2075": s["share_solvent_through_2075"],
            "p_solvent_2100": s["share_solvent_through_2100"],
            "payable_2075_median": s["payable_ratio_2075"][1],
            "debt_gdp_2050_median": s["debt_pct_gdp_2050"][1],
        })
    return pd.DataFrame(rows)


CSS = """
:root{--paper:#f7f8f6;--ink:#1f2a24;--soft:#5b6660;--hair:#d9dfdb;--accent:#2f6f4f}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--paper:#141a17;--ink:#e6ece8;--soft:#a3ada7;--hair:#2c3530}}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--hair);padding-top:20px}
.sub{color:var(--soft);margin:0 0 18px}
img{max-width:100%;background:#fff;border-radius:6px;border:1px solid var(--hair)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--hair);vertical-align:top}
th{color:var(--soft);font-weight:600}td.n{text-align:right;white-space:nowrap}
.r{color:var(--soft);font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:14px 0}
.kpi{border:1px solid var(--hair);border-radius:8px;padding:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--soft);font-size:13px}
.note{color:var(--soft);font-size:13px}
.wrap{overflow-x:auto}
"""


def write_report(runs: dict[str, Run], det: dict[str, Run], solvency: dict[str, Run], out_dir: Path,
                 sims: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, fit = calibrate()
    table = summary_table(runs)
    table.to_csv(out_dir / "summary.csv", index=False)
    if solvency:
        summary_table({"current_law": runs["current_law"], **solvency}).to_csv(out_dir / "solvency.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {k: v.summary for k, v in runs.items()}, indent=1, default=float))

    cl = runs["current_law"].summary
    dcl = det["current_law"]
    keys = [k for k in runs if k != "current_law"]
    img_tf = chart_trust_fund(runs["current_law"], dcl)
    img_rates = chart_rates(runs["current_law"])
    img_pol = chart_policies(runs, keys)
    debt_keys = [k for k in ("current_law", "gr_borrow", "gr_income_tax", "package_revenue") if k in runs]
    img_debt = chart_debt(runs, debt_keys)

    def row(k: str, run: Run) -> str:
        s = run.summary
        return (f"<tr><td><b>{html.escape(run.policy.name)}</b><br><span class=r>{html.escape(run.policy.description)}"
                f"</span></td><td class=n>{_fmt_range(s['actuarial_balance'], '{:.2f}')}</td>"
                f"<td class=n>{_fmt_range(s['depletion_year'], '{:.0f}')}</td>"
                f"<td class=n>{s['share_solvent_through_2050']:.0%}</td><td class=n>{s['share_solvent_through_2100']:.0%}</td>"
                f"<td class=n>{s['debt_pct_gdp_2050'][1]:.0f}%</td></tr>")

    head = ("<tr><th>Option</th><th>75-year balance, % of payroll</th><th>Reserves run out</th>"
            "<th>Solvent through 2050</th><th>Through 2100</th><th>Debt 2050</th></tr>")
    pol_rows = "".join(row(k, runs[k]) for k in runs)
    solv_rows = "".join(row(k, v) for k, v in solvency.items())
    fit_rows = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td class=n>{v[0] / (1 if 'rate' in k else 1e12):.2f}</td>"
        f"<td class=n>{v[1] / (1 if 'rate' in k else 1e12):.2f}</td></tr>"
        for k, v in fit.items() if isinstance(v, tuple))
    body = f"""
<main>
<h1>Social Security, Medicare and the debt: a Monte Carlo outlook</h1>
<p class=sub>Baseline: the 2026 Trustees' intermediate assumptions, rebuilt from public data. {sims:,} simulated futures (seed {seed}). Generated {date.today():%B %-d, %Y}.</p>
<div class=kpis>
<div class=kpi><b>{cl['depletion_year'][1]:.1f}</b><span>median year combined OASDI reserves run out (80% range {cl['depletion_year'][0]:.1f} to {cl['depletion_year'][2]:.1f})</span></div>
<div class=kpi><b>{cl['actuarial_balance'][1]:.2f}%</b><span>75-year actuarial balance, % of taxable payroll (80% range {cl['actuarial_balance'][0]:.2f} to {cl['actuarial_balance'][2]:.2f})</span></div>
<div class=kpi><b>{cl['payable_ratio_2035'][1]:.0%}</b><span>share of scheduled benefits payable in 2035; {cl['payable_ratio_2075'][1]:.0%} in 2075</span></div>
<div class=kpi><b>{cl['debt_pct_gdp_2050'][1]:.0f}%</b><span>debt held by the public in 2050, % of GDP (current law, benefits cut at depletion)</span></div>
</div>
<h2>Current law</h2>
<img src="data:image/png;base64,{img_tf}" alt="Trust fund ratio fan chart and histogram of depletion years">
<img src="data:image/png;base64,{img_rates}" alt="Cost and income rates and payable benefit share">
<p class=note>Bands show the 5-95, 10-90 and 25-75 percentile ranges across simulated futures; the line is the median.</p>
<h2>Policy options</h2>
<img src="data:image/png;base64,{img_pol}" alt="Actuarial balance by policy option">
<div class=wrap><table>{head}{pol_rows}</table></div>
<p class=note>Ranges are 10th to 90th percentiles. "Reserves run out" shows 2101 when they last past 2100. Every option runs on the same simulated futures and is valued at current-law interest rates.</p>
{"<h2>What it takes to stay solvent through 2100</h2><p>Each package is sized so reserves stay positive through 2100 in the stated share of futures.</p><div class=wrap><table>" + head + solv_rows + "</table></div>" if solvency else ""}
<h2>Federal debt</h2>
<img src="data:image/png;base64,{img_debt}" alt="Debt held by the public as a share of GDP">
<p class=note>Current law shows the median and percentile bands; other lines are medians. Spending and revenue outside Social Security and Medicare follow simple current-law rules from FY2025 actuals, not CBO's long-term projections. Read the debt paths as directions and relative sizes, not forecasts.</p>
<h2>How the baseline matches the Trustees</h2>
<div class=wrap><table><tr><th>Measure</th><th>Model</th><th>2026 Trustees</th></tr>{fit_rows}</table></div>
<p class=note>Rates are percent of taxable payroll; dollar amounts are trillions, present value at January 1, 2026. Targets from the 2026 Medicare Trustees Report, Tables V.F1 and V.F2.</p>
</main>"""
    page = f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Social Security Outlook</title><style>{CSS}</style></head><body>{body}</body></html>"
    path = out_dir / "report.html"
    path.write_text(page)
    return path
