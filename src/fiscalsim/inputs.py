"""Load data/processed into the arrays the model runs on.

Ages run 0..100 (100 means 100 and over); sex index 0 is male, 1 female.
Everything here is the base year (2025) or a published path the baseline is
calibrated to. Nothing in this module is an assumption about the future except
the age *patterns* borrowed from the Census Bureau's 2023 projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import DATA

PROC = DATA / "processed"
AGES = np.arange(101)
SEX = ("male", "female")
BASE_YEAR = 2025

# Statutory anchor: SSA's national average wage index for 2024, the value that
# set the 2026 taxable maximum ($184,500) and bend points ($1,286 / $7,749).
# Replace from data/manual/ssa/AWI.html when it is available.
AWI_2024 = 69_846.57


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PROC / name)


def _by_sex_age(df: pd.DataFrame, value: str, age_col: str = "age") -> np.ndarray:
    out = np.zeros((2, 101))
    for s, sex in enumerate(SEX):
        sub = df[df.sex == sex].set_index(age_col)[value]
        out[s, sub.index.values] = sub.values
    return out


def _cps_to_model_ages(prof: pd.DataFrame, col: str) -> np.ndarray:
    """CPS ages are at the March 2026 interview; income is for calendar 2025.

    A person aged A in March 2026 was A-1 on July 1, 2025 with probability
    ~0.71 (8.5 of 12 months), else A. Ages 80-84 and 85+ are published as
    groups; the group value is used for each single year.
    """
    out = np.zeros((2, 101))
    for s, sex in enumerate(SEX):
        p = prof[prof.sex == sex].set_index("age_at_interview")[col]
        at_interview = np.zeros(102)
        for a in range(102):
            src = 80 if 80 <= a < 85 else (85 if a >= 85 else a)
            at_interview[a] = p.get(src, np.nan)
        at_interview = pd.Series(at_interview).interpolate().values
        out[s] = 0.71 * at_interview[1:102] + 0.29 * at_interview[0:101]
    return out


@dataclass
class BaseData:
    pop: np.ndarray  # (2, 101) July 1, 2025 resident population
    pop_prev: np.ndarray  # (2, 101) July 1, 2024
    mx: np.ndarray  # (2, 101) central death rates for 2025, level-calibrated
    mort_improvement: np.ndarray  # (2, 101) relative age pattern of annual improvement, mean-normalized
    asfr_shape: np.ndarray  # (101,) fertility by mother's age, sums to 1
    tfr_2025: float  # total fertility implied by Census births
    male_birth_share: float
    nim_shape: np.ndarray  # (2, 101) age-sex mix of net migrants, sums to 1
    nim_2025: float
    std_pop: np.ndarray  # (2, 101) standard population for age-sex-adjusted death rates
    ss_per_capita: np.ndarray  # (2, 101) CPS Social Security per person, model ages
    di_share: np.ndarray  # (2, 101) share of those benefits paid for disability (reason code 2)
    earnings_per_capita: np.ndarray  # (2, 101) CPS earnings per person, model ages
    cms: pd.DataFrame  # Trustees' Medicare and payroll paths by calendar year
    awi: pd.Series  # average wage index by year (history; proxy before 2024)
    cola: pd.Series  # COLA in percent, effective December of each year
    fred: pd.DataFrame
    fred_ytd: pd.DataFrame  # current partial year: average to date vs same months a year earlier
    budget_pct_gdp: pd.DataFrame
    debt: pd.DataFrame
    trust_funds: pd.DataFrame
    wage_bins: pd.DataFrame
    treasury_rates: pd.DataFrame
    debt_to_penny: pd.DataFrame
    notes: dict = field(default_factory=dict)


def _death_rates(pop_prev: np.ndarray, pop: np.ndarray, deaths_target: float) -> tuple[np.ndarray, float]:
    """Census-projected 2025 death rates, scaled so deaths match the estimate."""
    mx = _csv("census_np2023_mx.csv")
    m25 = _by_sex_age(mx[mx.year == 2025], "mx")
    exposure = 0.5 * (pop_prev + pop)
    implied = (m25 * exposure).sum()
    scale = deaths_target / implied
    return m25 * scale, scale


# Relative rates of decline in death rates by age, shaped like the Trustees'
# ultimate assumptions (fastest for children, about 1% at working ages, slower
# past 85). Only the shape matters: the model scales it each year so the
# age-sex-adjusted death rate falls at the assumed aggregate rate. Replace
# with the Trustees' age-group rates when the SSA tables are available.
# The Census 2023 projection's own pattern was tried first and rejected: it
# implies 2-2.5% a year at ages 20-60 and pushed the 65+ population ~7% above
# the Trustees' Medicare enrollment path by 2075.
IMPROVEMENT_SHAPE = {0: 2.2, 14: 2.2, 15: 1.3, 49: 1.3, 50: 1.2, 64: 1.2, 65: 1.0, 84: 1.0, 85: 0.7, 94: 0.55, 100: 0.35}


def _improvement_pattern() -> np.ndarray:
    xs = np.array(sorted(IMPROVEMENT_SHAPE))
    shape = np.interp(np.arange(101), xs, [IMPROVEMENT_SHAPE[x] for x in xs])
    return np.vstack([shape, shape])


@lru_cache(maxsize=1)
def load() -> BaseData:
    cpop = _csv("census_population.csv")
    pop = _by_sex_age(cpop[cpop.year == 2025], "pop")
    pop_prev = _by_sex_age(cpop[cpop.year == 2024], "pop")
    std_pop = _by_sex_age(cpop[cpop.year == 2020], "pop")
    comp = _csv("census_components.csv").set_index("year")

    mx, mort_scale = _death_rates(pop_prev, pop, comp.loc[2025, "deaths"])

    asfr = _csv("census_np2023_asfr.csv")
    a25 = asfr[asfr.year == 2025].set_index("age").asfr
    shape = np.zeros(101)
    shape[a25.index.values] = a25.values
    shape = shape / shape.sum()
    women = 0.5 * (pop_prev[1] + pop[1])
    tfr_2025 = comp.loc[2025, "births"] / (shape * women).sum()
    births = _csv("census_np2023_births.csv").set_index("year").loc[2025]
    male_share = births.male / (births.male + births.female)

    nim = _csv("census_np2023_nim.csv")
    n25 = _by_sex_age(nim[nim.year == 2025], "nim")
    nim_shape = n25 / n25.sum()

    prof = _csv("cps_asec2026_profiles.csv")
    ss = _cps_to_model_ages(prof, "ss_per_capita")
    di = _cps_to_model_ages(prof, "ss_per_capita_disabled")
    di_share = np.where(ss > 0, di / np.maximum(ss, 1e-9), 0.0)
    di_share[:, 67:] = 0.0  # disabled-worker benefits convert to retirement at the FRA
    earn = _cps_to_model_ages(prof, "earnings_per_capita")

    cms = _csv("cms_tr2026_medicare.csv").set_index("year")

    fred = _csv("fred_annual.csv").set_index("year")
    proxy = ((fred.A576RC1 * 1e9) / (fred.PAYEMS * 1e3)).dropna()
    # Pandemic job losses were concentrated in low-wage work, which inflates
    # wages per job in 2020-2021 far more than SSA's index moved. Bridge those
    # years geometrically between 2019 and 2022.
    g = (proxy.loc[2022] / proxy.loc[2019]) ** (1 / 3)
    proxy.loc[2020] = proxy.loc[2019] * g
    proxy.loc[2021] = proxy.loc[2019] * g * g
    awi = proxy / proxy.loc[2024] * AWI_2024

    cola = _csv("cola_history.csv").set_index("year").cola_pct

    return BaseData(
        pop=pop,
        pop_prev=pop_prev,
        mx=mx,
        mort_improvement=_improvement_pattern(),
        asfr_shape=shape,
        tfr_2025=float(tfr_2025),
        male_birth_share=float(male_share),
        nim_shape=nim_shape,
        nim_2025=float(comp.loc[2025, "net_international_migration"]),
        std_pop=std_pop,
        ss_per_capita=ss,
        di_share=di_share,
        earnings_per_capita=earn,
        cms=cms,
        awi=awi,
        cola=cola,
        fred=fred,
        fred_ytd=_csv("fred_year_to_date.csv").set_index("series"),
        budget_pct_gdp=_csv("omb_budget_pct_gdp.csv").set_index("fiscal_year"),
        debt=_csv("omb_debt.csv").set_index("fiscal_year"),
        trust_funds=_csv("omb_trust_funds.csv"),
        wage_bins=_csv("irs_w2_2023_wage_bins.csv"),
        treasury_rates=_csv("treasury_avg_interest_rates.csv"),
        debt_to_penny=_csv("treasury_debt_to_penny.csv"),
        notes={
            "mortality_level_scale": float(mort_scale),
            "awi_history": "wages and salaries per payroll job (FRED A576RC1 / PAYEMS), scaled to SSA's 2024 AWI",
        },
    )
