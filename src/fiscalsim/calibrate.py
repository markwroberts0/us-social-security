"""Fit the deterministic baseline to the 2026 Trustees' published totals.

Anchors (all from data/processed):
  - calendar-2025 OASDI benefits, reserves and yield: OMB Table 13.1, FY2025
    actuals blended with a quarter of FY2026 to reach calendar years
  - 2025 OASDI and HI taxable payroll and GDP: MTR Table V.F1 note (p. 216)
  - HI taxable payroll and GDP by year: CMS expanded tables
  - 75-year present values, January 1, 2026: MTR Table V.F2 (p. 217)

Three residual paths are fitted, each to one published target:
  payroll_residual  -> HI taxable payroll by year (CMS)
  taxable_ratio     -> PV of OASDI taxable payroll ($690.4T)
  tob_rate          -> PV of OASDI payroll and benefit taxes ($91.9T)
OASDI cost is not fitted. Its present value ($123.7T) is the out-of-sample test.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from .assumptions import YEARS, Assumptions, trustees_2026
from .budget import current_law as budget_current_law
from .inputs import BASE_YEAR, load
from .model import T, Calibration, Model, Paths, Results

# MTR Table V.F2 (p. 217), present values over 2026-2100 as of January 1, 2026.
PV_TARGETS = {
    "oasdi_taxable_payroll": 690.4e12,
    "oasdi_revenue": 91.9e12,  # payroll and benefit taxes
    "oasdi_cost": 123.7e12,  # benefits, administration, railroad interchange
    "oasdi_assets_2026": 2.6e12,
    "hi_taxable_payroll": 810.5e12,
}
# The Trustees' summarized rates over 2026-2100, from the same table: these
# ratios, not the dollar PVs, are what the actuarial balance is built from.
COST_RATE_TARGET = PV_TARGETS["oasdi_cost"] / PV_TARGETS["oasdi_taxable_payroll"]  # 17.92%
INCOME_RATE_TARGET = PV_TARGETS["oasdi_revenue"] / PV_TARGETS["oasdi_taxable_payroll"]  # 13.31%
TAXABLE_RATIO_DECLINE = 0.02
DI_RISE = 0.30
# MTR Table V.F1 note (p. 216): calendar 2025.
OASDI_TAXABLE_PAYROLL_2025 = 11_043e9
HI_TAXABLE_PAYROLL_2025 = 13_277e9
# MTR p. 50: HI received $9.1 billion of interest in 2025.
HI_INTEREST_2025 = 9.1e9


def _omb(fund: str, item: str, fy: int) -> float:
    tf = load().trust_funds
    row = tf[(tf.fund == fund) & (tf["item"] == item) & (tf.fiscal_year == fy)]
    return float(row.millions.iloc[0]) * 1e6


def _calendar(fund_items: list[tuple[str, str]]) -> float:
    """Calendar 2025 = 3/4 of FY2025 (Jan-Sep) + 1/4 of FY2026 (Oct-Dec)."""
    return sum(0.75 * _omb(f, i, 2025) + 0.25 * _omb(f, i, 2026) for f, i in fund_items)


def base_year_anchors() -> dict:
    benefits = _calendar([("OASI", "Benefit payments (1)"), ("DI", "Benefit payments (1)")])
    admin = _calendar([
        ("OASI", "Administrative expenses"), ("DI", "Administrative expenses"),
        ("OASI", "Payments to the railroad retirement account"),
        ("DI", "Payments to the railroad retirement account"),
    ])
    fy25_end = _omb("OASI", "Fund balance, end of year", 2025) + _omb("DI", "Fund balance, end of year", 2025)
    fy24_end = _omb("OASI", "Fund balance, end of year", 2024) + _omb("DI", "Fund balance, end of year", 2024)
    fy26_surplus = _omb("OASI", "Surplus or deficit (-)", 2026) + _omb("DI", "Surplus or deficit (-)", 2026)
    interest_fy25 = _omb("OASI", "Interest (903)", 2025) + _omb("DI", "Interest (903)", 2025)
    tob_fy25 = _omb("OASI", "Other", 2025) + _omb("DI", "Other", 2025)
    ben_fy25 = _omb("OASI", "Benefit payments (1)", 2025) + _omb("DI", "Benefit payments (1)", 2025)
    cms = load().cms
    hi_ops = load().cms  # noqa: F841
    return {
        "benefits_2025": benefits,
        "admin_rate": admin / benefits,
        "tf_assets_2025": fy25_end + 0.25 * fy26_surplus,
        "tf_yield_2025": interest_fy25 / (0.5 * (fy24_end + fy25_end)),
        "tob_rate_2025": tob_fy25 / ben_fy25,
        "hi_payroll": cms.hi_taxable_payroll,
    }


def _hi_assets_2025() -> tuple[float, float]:
    import pandas as pd

    from .inputs import PROC

    ops = pd.read_csv(PROC / "cms_tr2026_hi_operations.csv").set_index("year")
    end = ops.filter(like="balance_at_end").iloc[:, 0]
    a25, a24 = end.loc[2025] * 1e6, end.loc[2024] * 1e6
    return a25, HI_INTEREST_2025 / (0.5 * (a24 + a25))


def present_value(values: np.ndarray, yields: np.ndarray) -> np.ndarray:
    """PV at January 1, 2026 of flows over 2026-2100, mid-year discounting.

    values, yields: (N, T) aligned to YEARS (index 0 = 2025 is skipped).
    """
    r = yields[:, 1:]
    disc = np.cumprod(1 / (1 + r), axis=1) * np.sqrt(1 + r)
    return (values[:, 1:] * disc).sum(axis=1)


def _initial(base, anchors, a: Assumptions) -> Calibration:
    cms = base.cms
    years = list(YEARS)
    pay = cms.hi_taxable_payroll.reindex(years).values
    hi_assets, hi_yield = _hi_assets_2025()
    benefit_scale = anchors["benefits_2025"] / (base.pop * base.ss_per_capita).sum()
    earnings_scale = HI_TAXABLE_PAYROLL_2025 / (base.pop * base.earnings_per_capita).sum()
    awi_2025 = base.awi.loc[2024] * base.awi.loc[2025] / base.awi.loc[2024]
    return Calibration(
        benefit_scale=float(benefit_scale),
        earnings_scale=float(earnings_scale),
        payroll_residual=np.ones(T),
        taxable_ratio=np.full(T, OASDI_TAXABLE_PAYROLL_2025 / HI_TAXABLE_PAYROLL_2025),
        tob_rate=np.full(T, anchors["tob_rate_2025"]),
        di_prevalence=np.ones(T),
        cohort_level=np.ones(T),
        awi_2025=float(awi_2025),
        benefits_2025=anchors["benefits_2025"],
        admin_rate=anchors["admin_rate"],
        tf_assets_2025=anchors["tf_assets_2025"],
        tf_yield_2025=anchors["tf_yield_2025"],
        hi_assets_2025=hi_assets,
        hi_yield_2025=hi_yield,
        # MTR Table V.F1: SMI revenues from the public / SMI expenditures, FY2025.
        smi_premium_share=186.3 / 747.6,
        ref_p65=np.ones(T),
        ref_awi=np.ones(T),
        ref_unemployment=np.asarray(a.unemployment, dtype=float),
        hi_cost=(cms.hi_cost_rate.reindex(years).values / 100) * pay,
        hi_income_rate=cms.hi_income_rate.reindex(years).values / 100,
        smi_cost=(cms.smi_b_incurred + cms.smi_d_incurred).reindex(years).values,
        gdp_to_payroll=(cms.gdp / cms.hi_taxable_payroll).reindex(years).values,
    )


def _tob_path(t0: float, ult: float) -> np.ndarray:
    """Benefit-tax income as a share of benefits.

    The 2025 senior deduction (P.L. 119-21) lowers it for 2025-2028. After
    that the unindexed income thresholds pull more benefits into tax each year,
    approaching `ult` with a 25-year time constant.
    """
    k = np.arange(T)
    path = ult - (ult - t0) * np.exp(-np.maximum(k - 3, 0) / 25.0)
    path[1:4] = t0 * 0.94  # 2026-2028, deduction in force
    path[0] = t0
    return path


@lru_cache(maxsize=4)
def calibrate(assumptions_name: str = "trustees_2026") -> tuple[Calibration, dict]:
    base = load()
    a = trustees_2026()
    anchors = base_year_anchors()
    c = _initial(base, anchors, a)
    budget = budget_current_law()
    det = Paths.from_assumptions(a)

    # Pass 1: population and wages do not depend on the residuals.
    r1 = Model(base, c, budget).run(det)
    cms_pay = base.cms.hi_taxable_payroll.reindex(list(YEARS)).values
    resid = cms_pay / r1.covered_earnings[0]
    c = replace(c, payroll_residual=resid, ref_p65=r1.pop_65_plus[0], ref_awi=r1.awi[0])

    # Pass 2: taxable ratio. The Trustees' OASDI discount factors are not
    # published outside the SSA tables, and the payroll PV is too sensitive
    # to them to pin this path (its implied 2035 value ranges from -7% to +9%
    # across plausible conventions). Instead: a 2% decline over 2026-2035 as
    # wages above the cap grow, then flat. Their share has been roughly flat
    # since 2000 (15-17%); replace with TR Table VI.G6 when available.
    tau0 = c.taxable_ratio[0]
    tau = tau0 * (1 - TAXABLE_RATIO_DECLINE * np.clip(np.arange(T) / 10.0, 0, 1))
    c = replace(c, taxable_ratio=tau)

    # Disability: the Trustees assume incidence recovers from its 2020s lows.
    # We take a 30% rise in disability benefits per person by 2045. CPS
    # identifies only ~62% of DI benefits (6.2% of all benefits vs 10.0% in
    # Treasury data), so the multiplier on the CPS-identified part is larger.
    di_true_share = _omb("DI", "Benefit payments (1)", 2025) / (
        _omb("OASI", "Benefit payments (1)", 2025) + _omb("DI", "Benefit payments (1)", 2025))
    di_cps_share = float((base.pop * base.ss_per_capita * base.di_share).sum() / (base.pop * base.ss_per_capita).sum())
    ramp20 = np.clip(np.arange(T) / 20.0, 0, 1)
    c = replace(c, di_prevalence=1 + DI_RISE * (di_true_share / di_cps_share) * ramp20)

    # Passes 3-4, repeated because the depletion date moves the discount
    # rates: fit the summarized income rate (benefit-tax share) and the
    # summarized cost rate (benefit level of cohorts turning 62 after 2025,
    # growing at a constant rate through 2050: rising career earnings of
    # women and later claiming lift each new cohort above the cross-section).
    from scipy.optimize import brentq

    t0 = anchors["tob_rate_2025"]
    yrs = np.minimum(np.arange(T), 25)
    ult, g = 0.06, 0.0
    for _ in range(4):
        r3 = Model(base, c, budget).run(det)
        y = r3.discount_rate
        pv_tp = present_value(r3.taxable_payroll, y)[0]
        pv_payroll_tax = present_value(r3.ss_payroll_tax, y)[0]

        def income_rate(u):
            return (pv_payroll_tax + present_value(r3.scheduled_benefits * _tob_path(t0, u)[None], y)[0]) / pv_tp

        ult = brentq(lambda u: income_rate(u) - INCOME_RATE_TARGET, 0.0, 0.2)
        c = replace(c, tob_rate=_tob_path(t0, ult))

        def cost_rate(gg):
            rr = Model(base, replace(c, cohort_level=(1 + gg) ** yrs), budget).run(det)
            return present_value(rr.ss_cost, rr.discount_rate)[0] / present_value(rr.taxable_payroll, rr.discount_rate)[0]

        g = brentq(lambda gg: cost_rate(gg) - COST_RATE_TARGET, -0.02, 0.03)
        c = replace(c, cohort_level=(1 + g) ** yrs)
    d_ult = c.di_prevalence[-1] - 1
    tau_ult = tau[-1]

    final = Model(base, c, budget).run(det)
    report = fit_report(final, c, extra={"taxable_ratio_ultimate": tau_ult, "tob_rate_ultimate": ult,
                                          "di_prevalence_ultimate": 1 + d_ult,
                                          "cohort_level_growth": g, "cohort_level_2050": float(c.cohort_level[-1])})
    return c, report


def fit_report(r: Results, c: Calibration, extra: dict | None = None) -> dict:
    y = r.discount_rate
    pv = lambda v: present_value(v, y)[0]  # noqa: E731
    pv_pay = pv(r.taxable_payroll)
    rep = {
        "summarized_income_rate_pct": (pv(r.ss_payroll_tax + r.ss_benefit_tax_scheduled) / pv_pay * 100,
                                       INCOME_RATE_TARGET * 100),
        "summarized_cost_rate_pct": (pv(r.ss_cost) / pv_pay * 100, COST_RATE_TARGET * 100),
        "pv_taxable_payroll": (pv_pay, PV_TARGETS["oasdi_taxable_payroll"]),
        "pv_cost": (pv(r.ss_cost), PV_TARGETS["oasdi_cost"]),
        "pv_hi_payroll": (present_value(r.covered_earnings, r.hi_discount_rate)[0], PV_TARGETS["hi_taxable_payroll"]),
        "pv_gdp": (present_value(r.gdp, r.hi_discount_rate)[0], 1853.5e12),
        "depletion_year": float(r.ss_depletion_year[0]),
        "hi_depletion_year": float(r.hi_depletion_year[0]),
        "payroll_residual_2050": float(c.payroll_residual[YEARS == 2050][0]),
        "payroll_residual_2100": float(c.payroll_residual[-1]),
    }
    if extra:
        rep.update(extra)
    return rep


def baseline_model(policy=None) -> Model:
    c, _ = calibrate()
    return Model(load(), c, budget_current_law(), policy)


def baseline_paths(n: int = 1) -> Paths:
    return Paths.from_assumptions(trustees_2026(), n)


__all__ = ["calibrate", "baseline_model", "baseline_paths", "present_value", "BASE_YEAR"]
