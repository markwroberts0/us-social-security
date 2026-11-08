"""How wages are spread across workers, from IRS W-2 statistics (tax year 2023).

Used to answer one question: what share of all wages falls below a cap K.
That gives the taxable payroll under any taxable maximum, and the earnings
exposed to a threshold that is not indexed (the 0.9% Medicare surtax).

Within each IRS size-of-wages bin, wages follow a truncated Pareto density
whose shape is solved so the bin's mean matches the published total; the open
top bin is an ordinary Pareto tail. Caps are expressed relative to mean wages,
so the shape carries forward as the average wage grows.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .inputs import load


def _trunc_pareto_mean(alpha: float, lo: float, hi: float) -> float:
    if abs(alpha - 1) < 1e-9:
        return lo * hi * np.log(hi / lo) / (hi - lo)
    r = lo / hi
    num = alpha * lo * (1 - r ** (alpha - 1)) / (alpha - 1)
    return num / (1 - r**alpha)


def _solve_alpha(mean: float, lo: float, hi: float) -> float:
    """Shape whose truncated mean equals `mean`; negative alpha tilts mass upward."""
    target = np.clip(mean, lo * (1 + 1e-6), hi * (1 - 1e-6))
    a_lo, a_hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (a_lo + a_hi)
        if abs(mid) < 1e-9 or abs(mid - 1) < 1e-9:
            mid += 1e-6
        if _trunc_pareto_mean(mid, lo, hi) > target:
            a_lo = mid
        else:
            a_hi = mid
    return 0.5 * (a_lo + a_hi)


def _partial_above(alpha: float, lo: float, hi: float, k: float) -> float:
    """E[(w - k)+] for one person drawn from the bin's truncated Pareto."""
    if k <= lo:
        return _trunc_pareto_mean(alpha, lo, hi) - k
    if k >= hi:
        return 0.0
    # density f(w) = C w^(-alpha-1) on [lo, hi]
    c = alpha * lo**alpha / (1 - (lo / hi) ** alpha)

    def integral_w(a, b):  # integral of w f(w)
        if abs(alpha - 1) < 1e-9:
            return c * np.log(b / a)
        return c * (a ** (1 - alpha) - b ** (1 - alpha)) / (alpha - 1)

    def integral_1(a, b):  # integral of f(w)
        return c * (a ** (-alpha) - b ** (-alpha)) / alpha

    return integral_w(k, hi) - k * integral_1(k, hi)


class WageDistribution:
    def __init__(self) -> None:
        bins = load().wage_bins
        self.n = bins.taxpayers.values.astype(float)
        self.total = bins.wages.values.astype(float)
        self.lo = bins.lower.values.astype(float)
        self.hi = bins.upper.values.astype(float)
        self.mean_wage = self.total.sum() / self.n.sum()
        self.alpha = np.empty(len(self.n))
        for i in range(len(self.n)):
            m = self.total[i] / self.n[i]
            if np.isinf(self.hi[i]):
                self.alpha[i] = m / (m - self.lo[i])
            else:
                self.alpha[i] = _solve_alpha(m, self.lo[i], self.hi[i])

    def excess_share(self, cap_rel: float) -> float:
        """Share of all wages above a cap expressed as a multiple of mean wages."""
        if np.isinf(cap_rel):
            return 0.0
        k = cap_rel * self.mean_wage
        above = 0.0
        for i in range(len(self.n)):
            if np.isinf(self.hi[i]):
                if k <= self.lo[i]:
                    above += self.total[i] - self.n[i] * k
                else:
                    a = self.alpha[i]
                    # Pareto tail: E[(w-k)+] = (lo/k)^a * k / (a - 1)
                    above += self.n[i] * (self.lo[i] / k) ** a * k / (a - 1)
            else:
                above += self.n[i] * _partial_above(self.alpha[i], self.lo[i], self.hi[i], k)
        return above / self.total.sum()

    def share_below(self, cap_rel: float) -> float:
        return 1.0 - self.excess_share(cap_rel)


@lru_cache(maxsize=1)
def wage_distribution() -> WageDistribution:
    return WageDistribution()


# The 2023 OASDI contribution and benefit base, the cap in force for the IRS
# data year. Public law parameter (SSA, "Contribution and Benefit Base").
CAP_2023 = 160_200.0


def current_law_cap_rel() -> float:
    """Current-law cap relative to mean W-2 wages; constant because both index to wages."""
    return CAP_2023 / wage_distribution().mean_wage
