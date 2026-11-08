"""The rest of the federal budget, as shares of GDP.

Social Security and Medicare come from their own modules, and net interest is
computed from the debt. Everything else follows these paths, which start from
FY2025 actuals (OMB Historical Tables 2.3 and 8.4, FY2027 Budget) and move by
simple current-law rules stated next to each one. They stand in for CBO's
long-term budget outlook until data/manual/cbo/ltbo_2026.xlsx is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assumptions import YEARS
from .inputs import load


def _glide(start: float, end: float, end_year: int, start_year: int = 2025) -> np.ndarray:
    return np.interp(YEARS, [start_year, end_year], [start, end])


@dataclass
class BudgetPaths:
    individual_income_tax: np.ndarray
    corporate_income_tax: np.ndarray
    other_revenue: np.ndarray  # excise, customs, Fed remittances, estate and gift, misc.
    other_social_insurance: np.ndarray  # unemployment insurance, railroad retirement, federal employee contributions
    discretionary: np.ndarray
    medicaid: np.ndarray
    other_mandatory: np.ndarray  # other means-tested and other mandatory, net of offsetting receipts
    debt_held_by_public_2025: float  # end of calendar 2025, Treasury Debt to the Penny
    debt_rate_2025: float  # average rate on marketable debt, December 2025
    notes: dict


def current_law() -> BudgetPaths:
    b = load()
    fy = b.budget_pct_gdp.loc[2025]
    pct = lambda v: float(v) / 100  # noqa: E731
    dtp = b.debt_to_penny.set_index("record_date")
    rates = b.treasury_rates
    rate_dec = rates[(rates.record_date == "2025-12-31") & (rates.security_desc == "Total Marketable")]
    return BudgetPaths(
        # Real bracket creep: brackets index to chained CPI while incomes grow
        # with wages, so receipts rise faster than GDP; capped at 10% of GDP.
        individual_income_tax=np.minimum(pct(fy.individual_income) + 0.0002 * (YEARS - 2025), 0.10),
        corporate_income_tax=_glide(pct(fy.corporate_income), 0.014, 2030),
        # FY2025 actual 0.9% + excise 0.3%; customs receipts roughly doubled in
        # FY2026 (OMB estimate 1.5% "other"), held at 1.4% after.
        other_revenue=_glide(pct(fy.other_receipts) + pct(fy.excise), 0.014, 2027),
        other_social_insurance=np.full(len(YEARS), 0.003),
        # Appropriations grow with inflation through 2035, so their GDP share
        # falls, then they keep pace with GDP.
        discretionary=_glide(pct(fy.discretionary), 0.056, 2035),
        # 2025 Reconciliation Act reductions through 2034, then growth with
        # health costs and aging.
        medicaid=np.where(YEARS <= 2034, _glide(pct(fy.medicaid), 0.020, 2034),
                          0.020 + 0.00015 * (YEARS - 2034)),
        other_mandatory=_glide(pct(fy.other_means_tested) + pct(fy.other_mandatory) + pct(fy.offsetting_receipts),
                               0.026, 2035),
        debt_held_by_public_2025=float(dtp.loc["2025-12-31", "debt_held_public_amt"]),
        debt_rate_2025=float(rate_dec.avg_interest_rate_amt.iloc[0]) / 100,
        notes={"source": "OMB FY2027 Historical Tables (FY2025 actual); drift rules are this model's"},
    )
