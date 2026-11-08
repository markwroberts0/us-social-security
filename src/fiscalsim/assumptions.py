"""Baseline assumptions: the 2026 Trustees' intermediate path, year by year.

Sources are cited inline. "MTR" is the 2026 Medicare Trustees Report
(data/raw/cms/2026-medicare-trustees-report.pdf); its economic and demographic
assumptions are the OASDI Trustees' intermediate assumptions (MTR p. 14).
Where the Trustees publish only an ultimate value, the path to it is an
assumption of this model and says so. The SSA single-year tables
(data/manual/ssa) would replace those paths with the Trustees' own.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .inputs import BASE_YEAR, load

LAST_YEAR = 2100
YEARS = np.arange(BASE_YEAR, LAST_YEAR + 1)


def _path(points: dict[int, float]) -> np.ndarray:
    """Linear interpolation between anchor years, flat after the last one."""
    xs = np.array(sorted(points))
    ys = np.array([points[x] for x in xs])
    return np.interp(YEARS, xs, ys)


@dataclass
class Assumptions:
    name: str
    tfr: np.ndarray
    mortality_decline: np.ndarray  # annual reduction in the age-sex-adjusted death rate
    net_immigration: np.ndarray  # persons per year
    cpi: np.ndarray  # annual CPI-W growth
    real_wage: np.ndarray  # growth of the average covered wage minus CPI growth
    unemployment: np.ndarray
    real_interest: np.ndarray  # real yield on newly issued trust fund securities
    health_excess: np.ndarray = field(default_factory=lambda: np.zeros(len(YEARS)))
    notes: dict = field(default_factory=dict)

    def with_(self, **kw) -> "Assumptions":
        return replace(self, **kw)

    @property
    def nominal_wage(self) -> np.ndarray:
        return (1 + self.cpi) * (1 + self.real_wage) - 1

    @property
    def nominal_interest(self) -> np.ndarray:
        return (1 + self.cpi) * (1 + self.real_interest) - 1


def trustees_2026() -> Assumptions:
    base = load()
    fred, ytd = base.fred, base.fred_ytd
    cpi_2026 = ytd.loc["CWUR0000SA0", "avg_to_date"] / ytd.loc["CWUR0000SA0", "prior_year_same_months"] - 1
    # MTR Table II.D1: CPI averages 3.2% over 2026-2035, 2.4% after (Table II.C1).
    # The shape inside 2026-2035 is ours, holding the decade average at 3.2%.
    cpi_anchor = {2025: 0.029, 2026: round(cpi_2026, 4)}
    near = [0.034, 0.033, 0.033, 0.032, 0.032, 0.031, 0.031, 0.031, 0.030]
    near = list(np.array(near) + (0.032 * 10 - cpi_anchor[2026] - sum(near)) / 9)
    for i, v in enumerate(near):
        cpi_anchor[2027 + i] = v
    cpi_anchor[2036] = 0.024

    unemp_2026 = ytd.loc["UNRATE", "avg_to_date"] / 100

    return Assumptions(
        name="2026 Trustees intermediate (as reconstructed)",
        # Ultimate 1.75 (MTR Table II.C1, p. 21). 2025 is the rate implied by
        # Census Vintage 2025 births. Reaching the ultimate in 2050 follows the
        # Trustees' practice in recent reports; the 2026 timing is unconfirmed.
        tfr=_path({BASE_YEAR: base.tfr_2025, 2050: 1.75}),
        # 0.73% a year (MTR Table II.C1).
        mortality_decline=np.full(len(YEARS), 0.0073),
        # Ultimate 788,000 LPR + 389,000 other (MTR Table II.C1). The 2026-2029
        # ramp is ours: Census measured 1.26M for mid-2024 to mid-2025, and
        # enforcement since then points lower in the near term.
        net_immigration=_path({BASE_YEAR: base.nim_2025, 2026: 400_000, 2027: 700_000, 2028: 900_000,
                               2029: 1_050_000, 2030: 1_177_000}),
        cpi=_path(cpi_anchor),
        # 1.14% (MTR Table II.C1); the Trustees' near-term deviations are not
        # in the Medicare report, so the ultimate value is used throughout.
        real_wage=np.full(len(YEARS), 0.0114),
        # 4.5% is the Trustees' long-standing ultimate rate; the glide from
        # the 2026 average is ours.
        unemployment=_path({BASE_YEAR: fred.UNRATE.get(2025, 4.2) / 100, 2026: unemp_2026, 2030: 0.045}),
        # 2.3% ultimate (MTR Table II.C1), reached by year 15 (MTR p. 74). The
        # 2025 starting point (1.1%) is fitted: it is the value at which this
        # model's HI yields reproduce both present values in MTR Table V.F2
        # (HI taxable payroll $810.5T and GDP $1,853.5T).
        real_interest=_path({BASE_YEAR: 0.011, 2040: 0.023}),
        notes={
            "tfr": "1.58 in 2025 (Census births) rising linearly to 1.75 in 2050",
            "immigration": "0.4M in 2026 rising to 1.177M by 2030",
            "cpi": f"{cpi_2026:.1%} in 2026 (CPI-W to date), decade average 3.2%, 2.4% from 2036",
        },
    )
