"""Monte Carlo paths around the baseline.

Each simulated path is  baseline + long-run level offset (ramped in) + AR noise.

  * The annual noise for inflation, real wage growth, unemployment and the
    real interest rate is estimated from US history (FRED, 1983-2025): AR(1)
    for each, AR(2) for unemployment, with correlated innovations.
  * Fertility persistence and volatility are estimated from 1976-2024.
  * Each path also draws its own long-run level for every assumption. The
    Trustees' low-cost and high-cost ultimate values (MTR Table II.C1) are
    treated as roughly the 2.5th and 97.5th percentiles, so the level's
    standard deviation is a quarter of that range. Draws are independent
    across variables, unlike the Trustees' low/high scenarios, which move
    every assumption in the same direction at once.
  * Mortality, immigration, health-cost and disability noise have no long
    clean US series in the data folder, so their annual noise is set by
    judgment, stated below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assumptions import YEARS, Assumptions
from .inputs import load
from .model import T, Paths

# MTR Table II.C1 (p. 21): (low-cost, high-cost) ultimate values.
TRUSTEES_RANGE = {
    "tfr": (2.10, 1.40),
    "mortality_decline": (0.0027, 0.0124),
    "real_wage": (0.0174, 0.0053),
    "cpi": (0.030, 0.018),
    "real_interest": (0.028, 0.018),
    "net_immigration": (1_624_000, 749_000),
}


@dataclass
class Process:
    phi: float
    sigma: float
    level_sd: float
    ramp_years: int  # years over which the long-run level offset phases in
    phi2: float = 0.0


@dataclass
class StochasticSpec:
    processes: dict
    corr: np.ndarray  # innovations of (cpi, real_wage, unemployment, real_interest)
    first_year_scale: float = 0.15  # 2026 is mostly observed as of November 2026
    notes: dict = field(default_factory=dict)


def _ar1(x: pd.Series) -> tuple[float, float, pd.Series]:
    y, yl = x.values[1:], x.values[:-1]
    X = np.c_[np.ones_like(yl), yl]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(b[1]), float(e.std(ddof=2)), pd.Series(e, index=x.index[1:])


def estimate(start: int = 1983, end: int = 2025) -> StochasticSpec:
    b = load()
    f = b.fred
    cpi = f.CWUR0000SA0.pct_change(fill_method=None)
    wage = b.awi.pct_change(fill_method=None)
    real_wage = (1 + wage) / (1 + cpi) - 1
    unemp = f.UNRATE / 100
    real_rate = f.GS10 / 100 - cpi.shift(1).rolling(3).mean()  # ex-ante: 10-year yield less trailing inflation

    phi_c, s_c, e_c = _ar1(cpi.loc[start:end].dropna())
    phi_w, s_w, e_w = _ar1(real_wage.loc[start:end].dropna())
    phi_r, s_r, e_r = _ar1(real_rate.loc[start:end].dropna())
    u = unemp.loc[1960:end].dropna().values
    X = np.c_[np.ones(len(u) - 2), u[1:-1], u[:-2]]
    bu, *_ = np.linalg.lstsq(X, u[2:], rcond=None)
    e_u = pd.Series(u[2:] - X @ bu, index=unemp.loc[1960:end].dropna().index[2:])
    s_u = float(e_u.std(ddof=3))
    E = pd.concat({"cpi": e_c, "real_wage": e_w, "unemployment": e_u, "real_interest": e_r}, axis=1).dropna()
    corr = E.corr().values

    tfr = f.SPDYNTFRTINUSA.loc[1976:].dropna()
    phi_f, s_f, _ = _ar1(tfr)

    rng = lambda k: abs(TRUSTEES_RANGE[k][0] - TRUSTEES_RANGE[k][1]) / 4  # noqa: E731
    processes = {
        # Inflation: history since the Volcker disinflation.
        "cpi": Process(phi=phi_c, sigma=s_c, level_sd=rng("cpi"), ramp_years=10),
        "real_wage": Process(phi=phi_w, sigma=s_w, level_sd=rng("real_wage"), ramp_years=10),
        # Unemployment: AR(2) business cycle around the baseline; the Trustees
        # do not publish a range for the ultimate rate, 0.25 points is ours.
        "unemployment": Process(phi=float(bu[1]), phi2=float(bu[2]), sigma=s_u, level_sd=0.0025, ramp_years=10),
        "real_interest": Process(phi=phi_r, sigma=s_r, level_sd=rng("real_interest"), ramp_years=15),
        # Fertility is close to a random walk historically; mean reversion
        # is capped at 0.97 so paths drift without wandering off for good.
        "tfr": Process(phi=min(phi_f, 0.97), sigma=s_f, level_sd=rng("tfr"), ramp_years=25),
        # Annual noise in the rate of mortality improvement: 1 point a year
        # with little persistence (judgment; pandemics are larger, see notes).
        "mortality_decline": Process(phi=0.3, sigma=0.010, level_sd=rng("mortality_decline"), ramp_years=1),
        # Net immigration: 250,000 a year of noise, fairly persistent (judgment).
        "net_immigration": Process(phi=0.6, sigma=250_000, level_sd=rng("net_immigration"), ramp_years=5),
        # Excess growth of Medicare costs per enrollee beyond the Trustees'
        # path: 1 point a year of noise, 0.3 points of long-run level (judgment).
        "health_excess": Process(phi=0.5, sigma=0.010, level_sd=0.003, ramp_years=10),
        # Disability prevalence, as a log multiplier (judgment).
        "disability": Process(phi=0.8, sigma=0.03, level_sd=0.10, ramp_years=15),
    }
    return StochasticSpec(
        processes=processes,
        corr=corr,
        notes={"estimation_window": f"{start}-{end}", "tfr_window": "1976-2024", "corr_vars": list(E.columns)},
    )


BOUNDS = {
    "tfr": (1.0, 2.8),
    "mortality_decline": (-0.05, 0.05),
    "net_immigration": (-500_000, 4_000_000),
    "cpi": (-0.02, 0.15),
    "real_wage": (-0.06, 0.06),
    "unemployment": (0.025, 0.15),
    "real_interest": (-0.02, 0.08),
    "health_excess": (-0.05, 0.05),
}


def draw_paths(a: Assumptions, n: int, seed: int = 0, spec: StochasticSpec | None = None) -> Paths:
    spec = spec or estimate()
    rng = np.random.default_rng(seed)
    base = Paths.from_assumptions(a, n)
    k = np.arange(T)
    first = np.where(k == 1, spec.first_year_scale, 1.0)  # dampen 2026
    first[0] = 0.0  # 2025 is history

    def ar(proc: Process, eps: np.ndarray) -> np.ndarray:
        x = np.zeros((n, T))
        for t in range(1, T):
            prev2 = x[:, t - 2] if t >= 2 else 0.0
            x[:, t] = proc.phi * x[:, t - 1] + proc.phi2 * prev2 + proc.sigma * eps[:, t] * first[t]
        return x

    def level(proc: Process) -> np.ndarray:
        ramp = np.clip(k / max(proc.ramp_years, 1), 0, 1)
        return rng.standard_normal(n)[:, None] * proc.level_sd * ramp[None]

    P = spec.processes
    econ = ["cpi", "real_wage", "unemployment", "real_interest"]
    L = np.linalg.cholesky(spec.corr)
    z = rng.standard_normal((n, T, len(econ))) @ L.T
    out = {}
    for i, name in enumerate(econ):
        out[name] = getattr(base, name) + level(P[name]) + ar(P[name], z[:, :, i])
    for name in ["tfr", "mortality_decline", "net_immigration", "health_excess"]:
        out[name] = getattr(base, name) + level(P[name]) + ar(P[name], rng.standard_normal((n, T)))
    for name, (lo, hi) in BOUNDS.items():
        out[name] = np.clip(out[name], lo, hi)
    dis = P["disability"]
    disability = np.exp(level(dis) + ar(dis, rng.standard_normal((n, T))))
    return Paths(disability=disability, **out)
