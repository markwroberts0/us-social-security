"""Named policy options, single provisions and packages.

Start years assume enactment in 2026 with changes from 2027, except where a
phase-in gives people time to plan. "Credit" means newly taxed wages also
count toward benefits.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .calibrate import baseline_model, baseline_paths
from .earnings import current_law_cap_rel, wage_distribution
from .metrics import actuarial_balance
from .policy import Policy


def cap_multiple_for_coverage(share_taxed: float) -> float:
    """Multiple of the current cap at which `share_taxed` of all wages is below it."""
    d, k = wage_distribution(), current_law_cap_rel()
    return brentq(lambda m: d.share_below(k * m) - share_taxed, 1.0, 400.0)


def library() -> dict[str, Policy]:
    cap90 = cap_multiple_for_coverage(0.90)
    p = Policy
    return {
        "current_law": p(),
        # --- Revenue
        "payroll_plus_1": p(name="Payroll tax +1 point", payroll_rate_increase=0.01,
                            description="12.4% combined rate rises to 13.4% in 2027 (half on workers, half on employers)."),
        "payroll_plus_2": p(name="Payroll tax +2 points", payroll_rate_increase=0.02,
                            description="12.4% rises to 14.4% in 2027."),
        "payroll_phased_3": p(name="Payroll tax +3 points, phased", payroll_rate_increase=0.03,
                              payroll_rate_phase_years=20,
                              description="Rate rises 0.15 points a year from 2027 to 15.4% in 2046."),
        "cap_90": p(name="Cover 90% of wages", taxable_max_multiple=cap90, taxable_max_phase_years=10,
                    benefit_credit_factor=0.15,
                    description=f"Taxable maximum raised to about {cap90:.1f}x its current-law level over 2027-2036, "
                                "restoring the 1983 coverage of 90% of wages; credit at 15%."),
        "cap_none_credit": p(name="No taxable maximum, with credit", taxable_max_multiple=np.inf,
                             taxable_max_phase_years=5, benefit_credit_factor=0.15,
                             description="Every dollar of wages taxed by 2031; new earnings credited at 15%."),
        "cap_none_no_credit": p(name="No taxable maximum, no credit", taxable_max_multiple=np.inf,
                                taxable_max_phase_years=5,
                                description="Every dollar of wages taxed by 2031; benefits unchanged."),
        "donut_400k": p(name="Tax wages above $400k", tax_above_threshold=400_000, benefit_credit_factor=0.02,
                        description="Wages above $400,000 (not indexed) taxed from 2027; the gap closes as the cap "
                                    "grows past $400,000. Credit at 2%, as in the Social Security 2100 Act."),
        # --- Benefits
        "fra_69": p(name="Retirement age 69", fra_target_years=69.0,
                    description="Full retirement age rises 2 months a year for people turning 62 from 2027, to 69."),
        "fra_70": p(name="Retirement age 70", fra_target_years=70.0,
                    description="Full retirement age rises 2 months a year from 2027, to 70."),
        "chained_cpi": p(name="Chained CPI for COLAs", cola_adjustment=-0.003,
                         description="Each COLA from December 2027 is 0.3 points lower."),
        "progressive_cut": p(name="Slow benefit growth for higher earners", new_award_cut=0.20,
                             new_award_cut_start=2030, new_award_cut_phase_years=30, new_award_cut_progressive=True,
                             description="Initial benefits grow more slowly for middle and high earners, "
                                         "reaching 20% below scheduled (15% overall) for those turning 62 in 2059."),
        "cut_all_now": p(name="Cut all benefits now", all_benefit_cut=0.22,
                         description="Every benefit 22% lower from 2027 instead of after depletion."),
        # --- General revenue
        "gr_borrow": p(name="Pay full benefits by borrowing", general_revenue="fill_gap",
                       description="General revenue covers whatever the trust fund cannot, financed by debt."),
        "gr_income_tax": p(name="Pay full benefits from income taxes", general_revenue="fill_gap",
                           general_revenue_financing="income_tax",
                           description="Same transfers, raised through income taxes each year."),
        # --- Demography
        "immigration_plus_50": p(name="50% more immigration", immigration_multiplier=1.5,
                                 description="Net immigration 50% above the Trustees' path every year."),
        # --- Packages
        "package_revenue": p(name="Revenue package", taxable_max_multiple=np.inf, taxable_max_phase_years=5,
                             benefit_credit_factor=0.15, payroll_rate_increase=0.012, payroll_rate_phase_years=10,
                             description="No taxable maximum (credit at 15%) plus a 1.2-point rate rise over "
                                         "2027-2036."),
        "package_benefits": p(name="Benefit package", fra_target_years=69.0, cola_adjustment=-0.003,
                              new_award_cut=0.20, new_award_cut_start=2030, new_award_cut_phase_years=30,
                              new_award_cut_progressive=True,
                              description="Retirement age 69, chained CPI, and slower initial benefit growth "
                                          "for higher earners."),
        "package_balanced": p(name="Balanced package", taxable_max_multiple=cap90, taxable_max_phase_years=10,
                              benefit_credit_factor=0.15, payroll_rate_increase=0.01, payroll_rate_phase_years=10,
                              cola_adjustment=-0.003, fra_target_years=68.0,
                              description="Cover 90% of wages, +1 point payroll over a decade, chained CPI, "
                                          "retirement age 68."),
    }


def solve(param: str, start: dict | None = None, lo: float = 0.0, hi: float = 0.1) -> float:
    """Smallest value of one Policy field that brings the deterministic 75-year balance to zero.

    Every candidate is valued at the current-law discount rates, as run_policies
    scores options, so the answer also scores zero in the report."""
    paths = baseline_paths()
    base_rates = baseline_model().run(paths).discount_rate

    def balance(v: float) -> float:
        pol = Policy(name="solve", **(start or {}), **{param: v})
        r = baseline_model(pol).run(paths)
        return float(actuarial_balance(r, base_rates)["actuarial_balance"][0])

    return brentq(balance, lo, hi, xtol=1e-6)


def solve_probability(
    template: Policy,
    param: str,
    target: float,
    paths,
    lo: float,
    hi: float,
    until: int = 2100,
    tol: float = 1e-4,
) -> tuple[float, float]:
    """Smallest value of `param` at which reserves last through `until` in at
    least `target` of the simulated paths. Paths are fixed, so the share is
    monotone in the lever and bisection applies. Returns (value, share)."""

    def share(v: float) -> float:
        r = baseline_model(template.with_(**{param: v})).run(paths)
        dep = r.ss_depletion_year
        return float(((~np.isfinite(dep)) | (dep >= until + 1)).mean())

    if share(hi) < target:
        return float("nan"), share(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if share(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi, share(hi)


def solvency_packages(paths, target: float = 0.8) -> dict[str, Policy]:
    """Four ways to keep reserves positive through 2100 in `target` of paths,
    each sized by solving for its last lever."""
    cap90 = cap_multiple_for_coverage(0.90)
    templates = {
        "solvent_payroll_only": (
            Policy(name="Payroll tax only", description="Rate rise from 2027, nothing else."),
            "payroll_rate_increase", 0.0, 0.12,
        ),
        "solvent_cut_only": (
            Policy(name="Benefit cuts only", description="Every benefit cut by the same share from 2027."),
            "all_benefit_cut", 0.0, 0.6,
        ),
        "solvent_no_cap_plus_rate": (
            Policy(name="No taxable maximum, then rate", taxable_max_multiple=np.inf, taxable_max_phase_years=5,
                   benefit_credit_factor=0.15, description="All wages taxed by 2031 with credit, plus a rate rise."),
            "payroll_rate_increase", 0.0, 0.10,
        ),
        "solvent_balanced": (
            Policy(name="Balanced mix", taxable_max_multiple=cap90, taxable_max_phase_years=10,
                   benefit_credit_factor=0.15, cola_adjustment=-0.003, fra_target_years=68.0,
                   description="Cover 90% of wages, chained CPI, retirement age 68, plus a rate rise."),
            "payroll_rate_increase", 0.0, 0.10,
        ),
    }
    out = {}
    for key, (tpl, param, lo, hi) in templates.items():
        value, share = solve_probability(tpl, param, target, paths, lo, hi)
        label = f"{value * 100:.2f} points" if param == "payroll_rate_increase" else f"{value:.1%}"
        out[key] = tpl.with_(**{param: value}, description=f"{tpl.description} Solved: {param.replace('_', ' ')} = "
                                                            f"{label} ({share:.0%} of paths solvent through 2100).")
    return out
