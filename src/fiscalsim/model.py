"""The projection engine. One code path serves a single deterministic run and a
Monte Carlo run of thousands of paths: every state array carries a leading
simulation axis of length N.

Each year, in order:
  1. Population: deaths, births and net immigration, by single year of age and sex.
  2. Economy: average wage index, covered earnings, taxable payroll and GDP.
  3. Social Security: benefits by age from cohort indexes (wage-indexed at
     entitlement, then COLAs), payroll and benefit-tax income, trust fund.
  4. Medicare: HI trust fund and SMI general revenue, scaled from the
     Trustees' paths by the simulated enrollment, wages and health costs.
  5. Budget: other revenue and spending as GDP shares, net interest, debt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assumptions import YEARS, Assumptions
from .budget import BudgetPaths
from .earnings import current_law_cap_rel, wage_distribution
from .inputs import BASE_YEAR, BaseData
from .policy import Policy, fra_factor, ramp

T = len(YEARS)
OASDI_RATE = 0.124
ENTITLEMENT_AGE = 62
YOUNG_TURNOVER = 0.15  # share of the under-62 benefit stock newly awarded each year
# Share of the gap between the trust funds' portfolio yield and the rate on new
# special issues that closes each year. Fitted (0.042) to OMB's projected OASI
# interest and balances for FY2026-2031 (Table 13.1): with bonds of up to 15
# years and a shrinking fund that buys few new ones, the yield moves slowly.
TF_YIELD_ADJUST = 0.042
# Once reserves are gone, the Trustees' hypothetical fund (which still pays
# scheduled benefits for actuarial measures) borrows at new-issue rates, so
# its yield, the discount rate for present values, converges to them quickly.
POST_DEPLETION_ADJUST = 0.5
DEBT_RATE_ADJUST = 0.20  # same for marketable federal debt
PAYROLL_UNEMPLOYMENT_ELASTICITY = 1.2  # % payroll lost per point of unemployment above baseline
DEBT_RATE_SENSITIVITY = 0.0002  # 2 basis points per point of debt-to-GDP above 2025 (CBO rule of thumb)
MAX_DEBT_PREMIUM = 0.03  # the rule of thumb is not meant for debt far outside history
TREASURY_SPREAD = 0.005  # new marketable borrowing (bills and coupons) vs the trust fund's 4+ year rate
PIA_TO_AIME = 0.42  # aggregate PIA / AIME for new retirees, for benefit credit on newly taxed wages


@dataclass
class Paths:
    """Economic and demographic drivers, each shaped (N, T)."""

    tfr: np.ndarray
    mortality_decline: np.ndarray
    net_immigration: np.ndarray
    cpi: np.ndarray
    real_wage: np.ndarray
    unemployment: np.ndarray
    real_interest: np.ndarray
    health_excess: np.ndarray
    disability: np.ndarray | None = None  # multiplier on disability prevalence, (N, T); 1 = assumption

    @property
    def n(self) -> int:
        return self.tfr.shape[0]

    @classmethod
    def from_assumptions(cls, a: Assumptions, n: int = 1) -> "Paths":
        tile = lambda x: np.tile(np.asarray(x, dtype=float), (n, 1))  # noqa: E731
        return cls(
            tfr=tile(a.tfr), mortality_decline=tile(a.mortality_decline), net_immigration=tile(a.net_immigration),
            cpi=tile(a.cpi), real_wage=tile(a.real_wage), unemployment=tile(a.unemployment),
            real_interest=tile(a.real_interest), health_excess=tile(a.health_excess),
        )


@dataclass
class Calibration:
    """Base-year anchors and the residual paths that make the deterministic
    baseline reproduce the Trustees' published totals. Built by calibrate.py."""

    benefit_scale: float  # CPS-reported Social Security -> administrative total
    earnings_scale: float  # CPS earnings -> HI taxable payroll
    payroll_residual: np.ndarray  # (T,) labor-force and coverage trends the age profile misses
    taxable_ratio: np.ndarray  # (T,) OASDI taxable payroll / covered earnings
    tob_rate: np.ndarray  # (T,) OASDI income from taxing benefits / benefits
    di_prevalence: np.ndarray  # (T,) disability benefits per person relative to 2025, by year
    cohort_level: np.ndarray  # (T,) benefit level of the cohort turning 62 in each year, relative to the 2025 cross-section
    awi_2025: float
    benefits_2025: float  # calendar 2025 OASDI benefits
    admin_rate: float  # administration and railroad interchange, share of benefits
    tf_assets_2025: float
    tf_yield_2025: float
    hi_assets_2025: float
    hi_yield_2025: float
    smi_premium_share: float
    ref_p65: np.ndarray  # (T,) baseline population 65+, to scale Medicare enrollment
    ref_awi: np.ndarray  # (T,) baseline AWI, to scale Medicare per-enrollee costs
    ref_unemployment: np.ndarray
    hi_cost: np.ndarray  # (T,) Trustees' HI cost, dollars
    hi_income_rate: np.ndarray  # (T,) Trustees' HI non-interest income / HI payroll
    smi_cost: np.ndarray  # (T,) Trustees' SMI B + D incurred cost, dollars
    gdp_to_payroll: np.ndarray  # (T,) Trustees' GDP / HI taxable payroll
    notes: dict = field(default_factory=dict)


@dataclass
class Results:
    years: np.ndarray
    policy: Policy
    # (N, T) arrays
    population: np.ndarray
    pop_65_plus: np.ndarray
    pop_20_64: np.ndarray
    births: np.ndarray
    awi: np.ndarray
    covered_earnings: np.ndarray
    taxable_payroll: np.ndarray
    gdp: np.ndarray
    cola: np.ndarray
    scheduled_benefits: np.ndarray
    paid_benefits: np.ndarray
    payable_ratio: np.ndarray
    ss_payroll_tax: np.ndarray
    ss_benefit_tax: np.ndarray  # on benefits actually paid
    ss_benefit_tax_scheduled: np.ndarray  # on scheduled benefits, as the Trustees' actuarial balance counts it
    ss_general_revenue: np.ndarray
    ss_interest: np.ndarray
    ss_cost: np.ndarray  # scheduled benefits plus administration
    ss_assets: np.ndarray  # end of year
    tf_yield: np.ndarray  # portfolio yield, earned on reserves
    discount_rate: np.ndarray  # yield of the Trustees' hypothetical fund, which borrows at new-issue rates once reserves are gone
    hi_income: np.ndarray
    hi_cost: np.ndarray
    hi_assets: np.ndarray
    hi_payable_ratio: np.ndarray
    hi_yield: np.ndarray
    hi_discount_rate: np.ndarray
    smi_cost: np.ndarray
    revenues: np.ndarray
    outlays: np.ndarray
    net_interest: np.ndarray
    deficit: np.ndarray
    debt: np.ndarray
    debt_rate: np.ndarray
    # (N,) arrays
    ss_depletion_year: np.ndarray  # fractional year; inf if never within the window
    hi_depletion_year: np.ndarray

    @property
    def n(self) -> int:
        return self.population.shape[0]


class Model:
    def __init__(self, base: BaseData, calib: Calibration, budget: BudgetPaths, policy: Policy | None = None):
        self.base = base
        self.c = calib
        self.budget = budget
        self.policy = policy or Policy()
        self._prepare()

    # --- setup ----------------------------------------------------------------------------

    def _prepare(self) -> None:
        b, p = self.base, self.policy
        ages = np.arange(101)
        birth_years = BASE_YEAR - ages
        awi = b.awi
        cola_hist = b.cola

        # Cohort index in 2025 for ages 62+: wage level at 60, COLAs since 62,
        # and the cohort's full retirement age. Theta is the age-sex shape that
        # turns that index into the CPS per-capita benefit.
        x0 = np.ones(101)
        for a in range(ENTITLEMENT_AGE, 101):
            c = BASE_YEAR - a
            level = awi.loc[c + 60] / awi.loc[2023]
            k = np.prod([1 + cola_hist.loc[y] / 100 for y in range(c + 62, BASE_YEAR)])
            x0[a] = level * k * fra_factor(c, Policy())
        self.x0 = x0
        ss = b.ss_per_capita * self.c.benefit_scale
        self.theta = ss / x0[None, :]
        self.theta_di = self.theta * b.di_share

        # FRA factor for each future cohort under the policy, relative to current law.
        self.fra_adj = {}
        for k, y in enumerate(YEARS):
            c = y - ENTITLEMENT_AGE
            self.fra_adj[k] = fra_factor(c, p) / fra_factor(c, Policy())

        # Mortality improvement pattern, normalized each year inside run().
        self.mort_pattern = b.mort_improvement

        # Policy schedules over YEARS.
        self.payroll_rate = OASDI_RATE + p.payroll_rate_increase * ramp(YEARS, p.payroll_rate_start,
                                                                        p.payroll_rate_phase_years)
        dist = wage_distribution()
        cap_rel = current_law_cap_rel()
        # The calibrated taxable ratio falls over the first decade. Read that as
        # wages above the cap growing (a wider top tail), scaling the IRS 2023
        # tail by `tail`, so cap options tax what the baseline leaves untaxed.
        coverage = self.c.taxable_ratio[0] / dist.share_below(cap_rel)  # OASDI-covered share of HI payroll
        tail = (1 - self.c.taxable_ratio / coverage) / dist.excess_share(cap_rel)
        self.tail = tail

        def below(k_rel: float, t: int) -> float:
            return 1.0 - tail[t] * dist.excess_share(k_rel)

        m = p.taxable_max_multiple
        phase = ramp(YEARS, p.taxable_max_start, p.taxable_max_phase_years)
        if m == 1.0:
            mult = np.ones(T)
        elif np.isinf(m):
            mult = 1.0 + (1.0 / np.array([below(cap_rel, t) for t in range(T)]) - 1.0) * phase
        else:
            mult = np.array([below(cap_rel * (1 + (m - 1) * phase[t]), t) / below(cap_rel, t) for t in range(T)])
        self.taxable_mult = mult
        # Donut: tax wages above a dollar threshold, evaluated per path inside
        # run() because the threshold's real value depends on simulated wages.
        self.donut_grid = None
        if p.tax_above_threshold > 0:
            grid = np.geomspace(0.5, 400, 600)
            self.donut_grid = (np.log(grid), np.array([dist.excess_share(g) for g in grid]), cap_rel)
        # Benefit credit on newly taxed wages, by cohort reaching 62 in each
        # year: the average extra taxable share over the 35 years before 62.
        # For the donut, use its extra share along the baseline wage path.
        credit_mult = mult
        if p.tax_above_threshold > 0:
            awi0 = self.c.ref_awi[0]
            thr = np.where(p.threshold_indexed, p.tax_above_threshold * self.c.ref_awi / awi0, p.tax_above_threshold)
            k_rel = np.maximum(thr / self.c.ref_awi, cap_rel)
            extra_det = np.array([tail[t] * dist.excess_share(x) / below(cap_rel, t) for t, x in enumerate(k_rel)])
            credit_mult = 1 + extra_det * (YEARS >= p.taxable_max_start)
        self.credit_adj = np.ones(T)
        if p.benefit_credit_factor > 0 and (m != 1.0 or p.tax_above_threshold > 0):
            hist = {int(y): v for y, v in zip(YEARS, credit_mult)}
            for k, y in enumerate(YEARS):
                career = [hist.get(int(yy), 1.0) for yy in range(y - 35, y)]
                extra = np.mean(career) - 1.0
                self.credit_adj[k] = 1.0 + p.benefit_credit_factor * extra / PIA_TO_AIME
        self.cola_adj = np.where(YEARS >= p.cola_start, p.cola_adjustment, 0.0)
        cut = p.new_award_cut * ramp(YEARS, p.new_award_cut_start, p.new_award_cut_phase_years)
        if p.new_award_cut_progressive:
            # Progressive price indexing spares roughly the bottom 30% of earners
            # and cuts the top more; on aggregate benefits it bites about 75% as hard.
            cut = cut * 0.75
        self.new_award_adj = 1.0 - cut
        self.all_cut = 1.0 - p.all_benefit_cut * (YEARS >= p.all_benefit_cut_start)
        self.budget_phase = ramp(YEARS, p.budget_change_start, p.budget_change_phase_years)
        self.gr_on = YEARS >= p.general_revenue_start

    # --- run ----------------------------------------------------------------------------------

    def run(self, paths: Paths) -> Results:
        b, c, p, bud = self.base, self.c, self.policy, self.budget
        N = paths.n
        out = {name: np.zeros((N, T)) for name in Results.__dataclass_fields__
               if name not in ("years", "policy", "ss_depletion_year", "hi_depletion_year")}
        ss_dep = np.full(N, np.inf)
        hi_dep = np.full(N, np.inf)

        # State
        P = np.broadcast_to(b.pop, (N, 2, 101)).copy()
        mx = np.broadcast_to(b.mx, (N, 2, 101)).copy()
        X = np.broadcast_to(self.x0, (N, 101)).copy()  # cohort benefit index, ages 62+
        Xy = np.ones(N)  # under-62 benefit stock index
        awi = np.full(N, c.awi_2025)
        awi_hist = {BASE_YEAR - 1: np.full(N, b.awi.loc[BASE_YEAR - 1]), BASE_YEAR: awi.copy()}
        A = np.full(N, c.tf_assets_2025)
        tf_yield = np.full(N, c.tf_yield_2025)
        A_hi = np.full(N, c.hi_assets_2025)
        hi_yield = np.full(N, c.hi_yield_2025)
        disc = np.full(N, c.tf_yield_2025)
        hi_disc = np.full(N, c.hi_yield_2025)
        debt = np.full(N, bud.debt_held_by_public_2025)
        debt_rate = np.full(N, bud.debt_rate_2025)
        health = np.ones(N)  # cumulative excess health cost factor
        cola_prev = np.full(N, b.cola.loc[BASE_YEAR] / 100)  # effective Dec 2025, paid from Jan 2026
        known_cola = {int(y): v / 100 for y, v in b.cola.items() if y >= BASE_YEAR}

        e_profile = b.earnings_per_capita * c.earnings_scale
        nim_shape = b.nim_shape
        asfr = b.asfr_shape
        std = b.std_pop
        imm_mult = p.immigration_multiplier
        dist_debt_gdp0 = None

        di_paths = paths.disability if paths.disability is not None else np.ones((N, T))
        self._donut_extra = np.zeros((N, T))
        for k, year in enumerate(YEARS):
            paths_di = di_paths[:, k]
            # ---- 1. Population -------------------------------------------------------
            if k > 0:
                r = paths.mortality_decline[:, k]
                w = std[None] * mx
                pat = self.mort_pattern[None]
                norm = (w * pat).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
                mx = mx * np.exp(-(r / norm)[:, None, None] * pat)
                surv = np.exp(-0.5 * (mx[:, :, :-1] + mx[:, :, 1:]))
                aged = np.zeros_like(P)
                aged[:, :, 1:] = P[:, :, :-1] * surv
                aged[:, :, 100] += P[:, :, 100] * np.exp(-mx[:, :, 100])
                women = 0.5 * (P[:, 1, :] + aged[:, 1, :])
                births = paths.tfr[:, k] * (women * asfr[None]).sum(axis=1)
                aged[:, 0, 0] = births * b.male_birth_share * np.exp(-0.3 * mx[:, 0, 0])
                aged[:, 1, 0] = births * (1 - b.male_birth_share) * np.exp(-0.3 * mx[:, 1, 0])
                aged += (paths.net_immigration[:, k] * imm_mult)[:, None, None] * nim_shape[None]
                P_prev, P = P, np.maximum(aged, 0.0)
                out["births"][:, k] = births
            else:
                P_prev = P
                out["births"][:, k] = np.nan

            # ---- 2. Economy -----------------------------------------------------------
            if k > 0:
                wage_growth = (1 + paths.cpi[:, k]) * (1 + paths.real_wage[:, k]) - 1
                awi = awi * (1 + wage_growth)
                awi_hist[year] = awi.copy()
            cyc = 1 - PAYROLL_UNEMPLOYMENT_ELASTICITY * (paths.unemployment[:, k] - c.ref_unemployment[k])
            ce = (P * e_profile[None]).sum(axis=(1, 2)) * (awi / c.awi_2025) * c.payroll_residual[k] * cyc
            tp = ce * c.taxable_ratio[k] * self.taxable_mult[k]
            if self.donut_grid is not None and year >= p.taxable_max_start:
                lg, ex, cap_rel = self.donut_grid
                thr = p.tax_above_threshold * ((awi / awi_hist[2025]) if p.threshold_indexed else 1.0)
                k_rel = np.maximum(thr / awi, cap_rel)  # never below the current-law cap
                below_cap = 1.0 - self.tail[k] * np.interp(np.log(cap_rel), lg, ex)
                extra = self.tail[k] * np.interp(np.log(k_rel), lg, ex) / below_cap
                tp = tp * (1 + extra)
                self._donut_extra[:, k] = extra
            gdp = ce * c.gdp_to_payroll[k]

            # ---- 3. Social Security ------------------------------------------------------
            if k > 0:
                cola = known_cola.get(int(year) - 1)
                cola_eff = (np.full(N, cola) if cola is not None else paths.cpi[:, k - 1]) + self.cola_adj[k - 1]
                cola_eff = np.maximum(cola_eff, 0.0)  # COLAs are never negative
                Xn = np.empty_like(X)
                Xn[:, 63:] = X[:, 62:-1] * (1 + cola_eff)[:, None]
                # 100+ is an open group: blend the survivors from 99 and from 100+.
                w99 = P_prev[:, :, 99].sum(axis=1) * np.exp(-mx[:, :, 99].mean(axis=1))
                w100 = P_prev[:, :, 100].sum(axis=1) * np.exp(-mx[:, :, 100].mean(axis=1))
                Xn[:, 100] = (w99 * X[:, 99] + w100 * X[:, 100]) / np.maximum(w99 + w100, 1) * (1 + cola_eff)
                lvl = awi_hist[year - 2] / b.awi.loc[2023] if (year - 2) in awi_hist else b.awi.loc[year - 2] / b.awi.loc[2023]
                Xn[:, 62] = lvl * self.fra_adj[k] * self.credit_adj[k] * self.new_award_adj[k] * c.cohort_level[k]
                Xn[:, :62] = 0.0
                X = Xn
                Xy = (1 - YOUNG_TURNOVER) * Xy * (1 + cola_eff) + YOUNG_TURNOVER * lvl * self.credit_adj[k] * self.new_award_adj[k]
                out["cola"][:, k - 1] = cola_eff
            theta = self.theta + self.theta_di * (c.di_prevalence[k] * paths_di[:, None, None] - 1.0) \
                if k > 0 else self.theta[None]
            per_capita = theta * np.concatenate(
                [np.broadcast_to(Xy[:, None], (N, 62)), X[:, 62:]], axis=1)[:, None, :]
            sched = (P * per_capita).sum(axis=(1, 2)) * self.all_cut[k]
            cost = sched * (1 + c.admin_rate)

            payroll_tax = self.payroll_rate[k] * tp
            tob = c.tob_rate[k]
            gr = np.zeros(N)
            if p.general_revenue == "share_of_payroll" and self.gr_on[k]:
                gr = p.general_revenue_share * tp
            nonint = payroll_tax + tob * sched + gr
            # Trust fund yield moves toward the rate on new special issues.
            new_issue = (1 + paths.cpi[:, k]) * (1 + paths.real_interest[:, k]) - 1
            if k > 0:
                tf_yield = tf_yield + TF_YIELD_ADJUST * (new_issue - tf_yield)
            A_start = A.copy()
            interest = tf_yield * np.maximum(A_start + 0.5 * (nonint - cost), 0.0)
            A_end = A_start + nonint + interest - cost
            if k == 0:
                # Reserves are given at the end of 2025; record 2025 flows only.
                interest = c.tf_yield_2025 * A_start
                A_end = A_start.copy()
            if p.general_revenue == "fill_gap" and self.gr_on[k]:
                fill = np.maximum(-A_end, 0.0)
                gr = gr + fill
                A_end = A_end + fill
            short = A_end < 0
            payable = np.ones(N)
            if short.any():
                # Reserves run out this year: pay what income and remaining reserves cover.
                avail = A_start + payroll_tax + gr + np.maximum(interest, 0) * 0.5
                ratio = avail / (sched * (1 + c.admin_rate - tob))
                payable = np.where(short, np.clip(ratio, 0.0, 1.0), 1.0)
                newly = short & np.isinf(ss_dep)
                gap = cost - nonint - interest
                frac = np.clip(A_start / np.maximum(gap, 1.0), 0.0, 0.999)
                ss_dep = np.where(newly, year + frac, ss_dep)
                A_end = np.where(short, 0.0, A_end)
            A = A_end
            paid = sched * payable
            out["scheduled_benefits"][:, k] = sched
            out["paid_benefits"][:, k] = paid
            out["payable_ratio"][:, k] = payable
            out["ss_payroll_tax"][:, k] = payroll_tax
            out["ss_benefit_tax"][:, k] = tob * paid
            out["ss_benefit_tax_scheduled"][:, k] = tob * sched
            out["ss_general_revenue"][:, k] = gr
            out["ss_interest"][:, k] = interest
            out["ss_cost"][:, k] = cost
            out["ss_assets"][:, k] = A
            out["tf_yield"][:, k] = tf_yield
            if k > 0:
                depleted = np.isfinite(ss_dep) & (ss_dep < year + 1)
                disc = np.where(depleted, disc + POST_DEPLETION_ADJUST * (new_issue - disc), tf_yield)
            out["discount_rate"][:, k] = disc

            # ---- 4. Medicare -----------------------------------------------------------
            p65 = P[:, :, 65:].sum(axis=(1, 2))
            if k > 0:
                health = health * np.exp(paths.health_excess[:, k])
            scale = (p65 / c.ref_p65[k]) * (awi / c.ref_awi[k]) * health
            hi_cost = c.hi_cost[k] * scale
            hi_income = c.hi_income_rate[k] * ce
            if k > 0:
                hi_yield = hi_yield + TF_YIELD_ADJUST * (new_issue - hi_yield)
            hi_int = hi_yield * np.maximum(A_hi + 0.5 * (hi_income - hi_cost), 0.0)
            A_hi_end = A_hi + hi_income + hi_int - hi_cost if k > 0 else A_hi.copy()
            hi_pay = np.ones(N)
            hshort = A_hi_end < 0
            if hshort.any():
                hi_pay = np.where(hshort, np.clip((A_hi + hi_income) / hi_cost, 0, 1), 1.0)
                newly = hshort & np.isinf(hi_dep)
                gap = hi_cost - hi_income - hi_int
                frac = np.clip(A_hi / np.maximum(gap, 1.0), 0.0, 0.999)
                hi_dep = np.where(newly, year + frac, hi_dep)
                A_hi_end = np.where(hshort, 0.0, A_hi_end)
            A_hi = A_hi_end
            smi = c.smi_cost[k] * scale
            out["hi_income"][:, k] = hi_income
            out["hi_cost"][:, k] = hi_cost
            out["hi_assets"][:, k] = A_hi
            out["hi_payable_ratio"][:, k] = hi_pay
            out["hi_yield"][:, k] = hi_yield
            if k > 0:
                hdep = np.isfinite(hi_dep) & (hi_dep < year + 1)
                hi_disc = np.where(hdep, hi_disc + POST_DEPLETION_ADJUST * (new_issue - hi_disc), hi_yield)
            out["hi_discount_rate"][:, k] = hi_disc
            out["smi_cost"][:, k] = smi

            # ---- 5. Budget --------------------------------------------------------------
            ph = self.budget_phase[k]
            revenues = (
                gdp * (bud.individual_income_tax[k] + bud.corporate_income_tax[k] + bud.other_revenue[k]
                       + bud.other_social_insurance[k] + p.income_tax_change * ph)
                + payroll_tax + hi_income * 0.89  # HI payroll taxes; benefit taxes are in individual income tax
            )
            if p.general_revenue_financing == "income_tax":
                revenues = revenues + gr  # the transfer is raised, not borrowed
            medicare_net = hi_cost * hi_pay + smi * (1 - c.smi_premium_share)
            program = (
                paid * (1 + c.admin_rate) + medicare_net
                + gdp * (bud.discretionary[k] + bud.medicaid[k] + bud.other_mandatory[k] - p.spending_cut * ph)
            )
            primary = program - revenues
            if dist_debt_gdp0 is None:
                dist_debt_gdp0 = debt / gdp
            premium = np.minimum(DEBT_RATE_SENSITIVITY * 100 * np.maximum(debt / gdp - dist_debt_gdp0, 0.0),
                                 MAX_DEBT_PREMIUM)
            market = new_issue - TREASURY_SPREAD + premium
            if k > 0:
                debt_rate = debt_rate + DEBT_RATE_ADJUST * (market - debt_rate)
                net_interest = debt_rate * (debt + 0.5 * primary)
                deficit = primary + net_interest
                debt = debt + deficit
            else:
                net_interest = debt_rate * debt
                deficit = primary + net_interest
            out["revenues"][:, k] = revenues
            out["outlays"][:, k] = program + net_interest
            out["net_interest"][:, k] = net_interest
            out["deficit"][:, k] = deficit
            out["debt"][:, k] = debt
            out["debt_rate"][:, k] = debt_rate

            # ---- bookkeeping -------------------------------------------------------------
            out["population"][:, k] = P.sum(axis=(1, 2))
            out["pop_65_plus"][:, k] = p65
            out["pop_20_64"][:, k] = P[:, :, 20:65].sum(axis=(1, 2))
            out["awi"][:, k] = awi
            out["covered_earnings"][:, k] = ce
            out["taxable_payroll"][:, k] = tp
            out["gdp"][:, k] = gdp

        out["cola"][:, -1] = np.nan
        return Results(years=YEARS, policy=p, ss_depletion_year=ss_dep, hi_depletion_year=hi_dep, **out)
