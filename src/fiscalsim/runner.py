"""Run policies over the same simulated futures and collect results.

Every policy sees the same random paths (common random numbers), so the
differences between them come from the policy, not from sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assumptions import trustees_2026
from .calibrate import baseline_model, baseline_paths
from .metrics import summarize
from .model import Paths, Results
from .policy import Policy
from .stochastic import draw_paths


@dataclass
class Run:
    key: str
    policy: Policy
    results: Results
    summary: dict


def run_policies(policies: dict[str, Policy], sims: int = 2000, seed: int = 2026) -> dict[str, Run]:
    """Monte Carlo when sims > 1; the deterministic baseline when sims == 1."""
    paths: Paths = baseline_paths() if sims <= 1 else draw_paths(trustees_2026(), sims, seed=seed)
    base = baseline_model().run(paths)
    out = {"current_law": Run("current_law", Policy(), base, summarize(base))}
    for key, pol in policies.items():
        if key == "current_law":
            continue
        r = baseline_model(pol).run(paths)
        out[key] = Run(key, pol, r, summarize(r, discount=base.discount_rate))
    return out
