"""Policy levers. The defaults are current law.

Each lever names the year it starts and, where it phases in, over how many
years. Scoring is static, as the Social Security actuaries do for individual
provisions: people claim at the same ages and earn the same wages as under
current law.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Policy:
    name: str = "Current law"
    description: str = "Scheduled taxes; benefits cut to what income covers once reserves run out."

    # --- Payroll tax
    payroll_rate_increase: float = 0.0  # added to the 12.4% combined rate (0.01 = one point)
    payroll_rate_start: int = 2027
    payroll_rate_phase_years: int = 1

    # --- Taxable maximum: multiple of the current-law cap (np.inf removes it)
    taxable_max_multiple: float = 1.0
    taxable_max_start: int = 2027
    taxable_max_phase_years: int = 1
    benefit_credit_factor: float = 0.0  # PIA factor on newly taxed earnings; 0 = no credit

    # --- Benefits
    cola_adjustment: float = 0.0  # added to every COLA (-0.003 is roughly chained CPI)
    cola_start: int = 2027
    fra_target_years: float = 67.0  # full retirement age reached by later cohorts
    fra_start_year: int = 2027  # first year in which a cohort turning 62 sees a higher FRA
    fra_months_per_year: float = 2.0
    new_award_cut: float = 0.0  # proportional cut to initial benefits (0.05 = 5%)
    new_award_cut_start: int = 2030
    new_award_cut_phase_years: int = 10
    new_award_cut_progressive: bool = False  # spare low earners, as progressive indexing would

    # --- General revenue
    general_revenue: str = "none"  # "none" | "fill_gap" | "share_of_payroll"
    general_revenue_share: float = 0.0  # of taxable payroll, when "share_of_payroll"
    general_revenue_start: int = 2027

    # --- Rest of the budget (percent of GDP, positive = more revenue / less spending)
    income_tax_change: float = 0.0
    spending_cut: float = 0.0
    budget_change_start: int = 2027
    budget_change_phase_years: int = 5

    # --- Demography
    immigration_multiplier: float = 1.0

    def with_(self, **kw) -> "Policy":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        if np.isinf(d["taxable_max_multiple"]):
            d["taxable_max_multiple"] = "none"
        return d


def ramp(years: np.ndarray, start: int, phase_years: int) -> np.ndarray:
    """0 before `start`, rising linearly to 1 over `phase_years`."""
    n = max(phase_years, 1)
    return np.clip((years - start + 1) / n, 0.0, 1.0)


# --- Full retirement age and claiming ------------------------------------------

# Share of retired-worker claims by age at entitlement. Roughly the recent
# pattern (heavy at 62, the FRA, and 70); an assumption of this model until
# SSA's Annual Statistical Supplement table 6.B5 is added.
CLAIM_AGE_SHARES = {62: 0.24, 63: 0.08, 64: 0.09, 65: 0.13, 66: 0.13, 67: 0.17, 68: 0.05, 69: 0.03, 70: 0.08}


def current_law_fra_months(birth_year: int) -> float:
    """Full retirement age in months, Social Security Act section 216(l)."""
    if birth_year <= 1937:
        return 65 * 12
    if birth_year <= 1942:
        return 65 * 12 + 2 * (birth_year - 1937)
    if birth_year <= 1954:
        return 66 * 12
    if birth_year <= 1959:
        return 66 * 12 + 2 * (birth_year - 1954)
    return 67 * 12


def drc_per_year(birth_year: int) -> float:
    """Delayed retirement credit, section 202(w)."""
    if birth_year >= 1943:
        return 0.08
    return max(0.03, 0.03 + 0.005 * ((birth_year - 1923) // 2))


def claim_factor(claim_months: float, fra_months: float, drc: float) -> float:
    early = fra_months - claim_months
    if early > 0:
        return 1 - (5 / 900) * min(early, 36) - (5 / 1200) * max(early - 36, 0)
    late = min(claim_months, 70 * 12) - fra_months
    return 1 + drc / 12 * max(late, 0)


def average_claim_factor(fra_months: float, birth_year: int) -> float:
    drc = drc_per_year(birth_year)
    return sum(s * claim_factor(age * 12, fra_months, drc) for age, s in CLAIM_AGE_SHARES.items())


def policy_fra_months(birth_year: int, policy: Policy) -> float:
    fra = current_law_fra_months(birth_year)
    if policy.fra_target_years <= 67:
        return fra
    year_turns_62 = birth_year + 62
    steps = max(0, year_turns_62 - policy.fra_start_year + 1)
    return min(policy.fra_target_years * 12, 67 * 12 + steps * policy.fra_months_per_year)


def fra_factor(birth_year: int, policy: Policy) -> float:
    """Average benefit relative to an FRA-67 cohort, with claiming ages held fixed."""
    ref = average_claim_factor(67 * 12, 1960)
    return average_claim_factor(policy_fra_months(birth_year, policy), birth_year) / ref
