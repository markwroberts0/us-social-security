"""Summary measures, defined as in the Trustees Report where one exists.

Actuarial balance (75 years, 2026-2100), percent of taxable payroll:
  summarized income rate = (reserves at the start + PV of non-interest income) / PV of taxable payroll
  summarized cost rate   = (PV of cost + PV of an ending fund equal to the next year's cost) / PV of taxable payroll
  actuarial balance      = income rate - cost rate
Present values discount with each path's own trust fund discount rate (see model.py). Income counts
taxes on scheduled benefits, as the Trustees do, and any general revenue the
policy dedicates (but not the "fill the gap" transfers, which would make the
balance zero by definition).
"""

from __future__ import annotations

import numpy as np

from .model import Results


def _disc(rates: np.ndarray) -> np.ndarray:
    """Mid-year discount factors to January 1, 2026, for 2026-2100: shape (N, 75)."""
    y = rates[:, 1:]
    return np.cumprod(1 / (1 + y), axis=1) * np.sqrt(1 + y)


def actuarial_balance(r: Results, discount: np.ndarray | None = None) -> dict:
    """`discount`: yearly rates to discount with, (N, T). Pass the current-law
    run's rates when comparing policies, so every option is valued at the same
    interest rates, as the Social Security actuaries score provisions."""
    rates = r.discount_rate if discount is None else discount
    d = _disc(rates)
    pv = lambda v: (v[:, 1:] * d).sum(axis=1)  # noqa: E731
    gr = r.ss_general_revenue if r.policy.general_revenue == "share_of_payroll" else 0 * r.ss_general_revenue
    income = r.ss_payroll_tax + r.ss_benefit_tax_scheduled + gr
    pv_pay = pv(r.taxable_payroll)
    start_assets = r.ss_assets[:, 0]
    end_disc = d[:, -1] / np.sqrt(1 + rates[:, -1])  # end of 2100
    next_cost = r.ss_cost[:, -1] * (r.ss_cost[:, -1] / r.ss_cost[:, -2])
    sir = (start_assets + pv(income)) / pv_pay
    scr = (pv(r.ss_cost) + next_cost * end_disc) / pv_pay
    return {
        "income_rate": sir * 100,
        "cost_rate": scr * 100,
        "actuarial_balance": (sir - scr) * 100,
        "pv_taxable_payroll": pv_pay,
        "unfunded_obligation": pv(r.ss_cost) - pv(income) - start_assets,
    }


def pct(x: np.ndarray, qs=(10, 50, 90)) -> list[float]:
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, q)) for q in qs]


def at(r: Results, arr: np.ndarray, year: int) -> np.ndarray:
    return arr[:, int(year - r.years[0])]


def summarize(r: Results, discount: np.ndarray | None = None) -> dict:
    """Headline numbers for one policy: medians and 10th-90th percentile ranges."""
    ab = actuarial_balance(r, discount)
    dep = r.ss_depletion_year
    never = ~np.isfinite(dep)
    out = {
        "policy": r.policy.name,
        "sims": r.n,
        "depletion_year": pct(np.where(never, 2101, dep)),
        "share_solvent_through_2050": float((np.where(never, 9999, dep) >= 2051).mean()),
        "share_solvent_through_2075": float((np.where(never, 9999, dep) >= 2076).mean()),
        "share_solvent_through_2100": float(never.mean()),
        "actuarial_balance": pct(ab["actuarial_balance"]),
        "unfunded_obligation_trillions": pct(ab["unfunded_obligation"] / 1e12),
        "hi_depletion_year": pct(np.where(np.isfinite(r.hi_depletion_year), r.hi_depletion_year, 2101)),
    }
    for y in (2035, 2050, 2075, 2100):
        out[f"payable_ratio_{y}"] = pct(at(r, r.payable_ratio, y))
        out[f"cost_rate_{y}"] = pct(at(r, r.ss_cost / r.taxable_payroll, y) * 100)
    for y in (2035, 2050, 2075):
        g = at(r, r.gdp, y)
        out[f"debt_pct_gdp_{y}"] = pct(at(r, r.debt, y) / g * 100)
        out[f"net_interest_pct_gdp_{y}"] = pct(at(r, r.net_interest, y) / g * 100)
        out[f"ss_outlays_pct_gdp_{y}"] = pct(at(r, r.paid_benefits, y) / g * 100)
    return out
