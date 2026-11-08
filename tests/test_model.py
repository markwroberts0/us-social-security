import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fiscalsim import DATA
from fiscalsim.assumptions import trustees_2026
from fiscalsim.calibrate import COST_RATE_TARGET, INCOME_RATE_TARGET, baseline_model, baseline_paths, calibrate
from fiscalsim.earnings import CAP_2023, wage_distribution
from fiscalsim.inputs import load
from fiscalsim.metrics import actuarial_balance
from fiscalsim.scenarios import library
from fiscalsim.stochastic import draw_paths, estimate

warnings.filterwarnings("ignore", category=RuntimeWarning)

PUBLISHED_COLA = {  # SSA, effective December of each year
    1984: 3.5, 1985: 3.1, 1986: 1.3, 1987: 4.2, 1988: 4.0, 1989: 4.7, 1991: 3.7, 1992: 3.0, 1993: 2.6,
    1994: 2.8, 1995: 2.6, 1996: 2.9, 1997: 2.1, 1998: 1.3, 1999: 2.5, 2000: 3.5, 2001: 2.6, 2002: 1.4,
    2003: 2.1, 2004: 2.7, 2005: 4.1, 2006: 3.3, 2007: 2.3, 2008: 5.8, 2009: 0.0, 2010: 0.0, 2011: 3.6,
    2012: 1.7, 2013: 1.5, 2014: 1.7, 2015: 0.0, 2016: 0.3, 2017: 2.0, 2018: 2.8, 2019: 1.6, 2020: 1.3,
    2021: 5.9, 2022: 8.7, 2023: 3.2, 2024: 2.5, 2025: 2.8,
}


def test_colas_recomputed_from_cpiw_match_published():
    # Every COLA since the 1983 amendments except 1990, which recomputes as
    # 5.3 (published 5.4), from FRED's CPI-W and SSA's rule.
    cola = load().cola
    assert {y: cola.loc[y] for y in PUBLISHED_COLA} == PUBLISHED_COLA


def test_wage_distribution_reproduces_irs_totals():
    bins = pd.read_csv(DATA / "processed" / "irs_w2_2023_wage_bins.csv")
    d = wage_distribution()
    # Wages above $200,000, straight from the bins that start there.
    top = bins[bins.lower >= 200_000]
    above_200k = (top.wages.sum() - top.taxpayers.sum() * 200_000) / bins.wages.sum()
    assert d.excess_share(200_000 / d.mean_wage) == pytest.approx(above_200k, abs=1e-4)
    # IRS Table 1.H: 15.2% of 2023 wages were above the $160,200 cap.
    assert d.excess_share(CAP_2023 / d.mean_wage) == pytest.approx(0.1518, abs=5e-4)


def test_base_year_population_is_census_2025():
    assert load().pop.sum() == pytest.approx(341_784_857, rel=1e-6)


def test_calibration_hits_trustees_ratios():
    _, fit = calibrate()
    assert fit["summarized_income_rate_pct"][0] == pytest.approx(INCOME_RATE_TARGET * 100, abs=0.01)
    assert fit["summarized_cost_rate_pct"][0] == pytest.approx(COST_RATE_TARGET * 100, abs=0.01)
    assert fit["pv_hi_payroll"][0] == pytest.approx(fit["pv_hi_payroll"][1], rel=2e-3)
    assert fit["pv_gdp"][0] == pytest.approx(fit["pv_gdp"][1], rel=2e-3)
    assert int(fit["hi_depletion_year"]) == 2033  # MTR: HI reserves depleted in 2033
    assert 2033 <= fit["depletion_year"] < 2036


def test_base_year_matches_administrative_totals():
    c, _ = calibrate()
    r = baseline_model().run(baseline_paths())
    assert r.scheduled_benefits[0, 0] == pytest.approx(c.benefits_2025, rel=1e-6)
    assert r.ss_assets[0, 0] == pytest.approx(c.tf_assets_2025, rel=1e-9)
    assert r.taxable_payroll[0, 0] == pytest.approx(11_043e9, rel=1e-3)


def _balance(policy):
    return float(actuarial_balance(baseline_model(policy).run(baseline_paths()),
                                   baseline_model().run(baseline_paths()).discount_rate)["actuarial_balance"][0])


def test_levers_move_the_balance_the_right_way():
    lib = library()
    base = _balance(lib["current_law"])
    gain = {k: _balance(lib[k]) - base for k in
            ["payroll_plus_1", "payroll_plus_2", "cap_90", "cap_none_credit", "cap_none_no_credit",
             "fra_69", "fra_70", "chained_cpi", "gr_borrow"]}
    assert 0.85 < gain["payroll_plus_1"] < 1.05
    assert gain["payroll_plus_2"] == pytest.approx(2 * gain["payroll_plus_1"], rel=0.05)
    assert 0 < gain["cap_90"] < gain["cap_none_credit"] < gain["cap_none_no_credit"]
    assert 0 < gain["fra_69"] < gain["fra_70"]
    assert 0.3 < gain["chained_cpi"] < 0.8
    assert gain["gr_borrow"] == pytest.approx(0.0, abs=1e-9)  # transfers are not income in the balance


def test_zero_noise_monte_carlo_equals_deterministic():
    spec = estimate()
    quiet = replace(spec, processes={k: replace(p, sigma=0.0, level_sd=0.0) for k, p in spec.processes.items()})
    mc = baseline_model().run(draw_paths(trustees_2026(), 3, seed=1, spec=quiet))
    det = baseline_model().run(baseline_paths())
    assert np.allclose(mc.ss_assets, det.ss_assets[[0, 0, 0]])
    assert np.allclose(mc.debt, det.debt[[0, 0, 0]])


def test_same_seed_same_futures():
    a = draw_paths(trustees_2026(), 50, seed=7)
    b = draw_paths(trustees_2026(), 50, seed=7)
    assert np.array_equal(a.cpi, b.cpi) and np.array_equal(a.tfr, b.tfr)


def test_paying_full_benefits_by_borrowing_adds_debt():
    lib = library()
    base = baseline_model().run(baseline_paths())
    gr = baseline_model(lib["gr_borrow"]).run(baseline_paths())
    tax = baseline_model(lib["gr_income_tax"]).run(baseline_paths())
    assert np.isinf(gr.ss_depletion_year[0])
    assert gr.debt[0, 25] > base.debt[0, 25] * 1.05
    assert tax.debt[0, 25] == pytest.approx(base.debt[0, 25], rel=0.02)


def test_unreachable_solvency_target_is_left_out():
    from fiscalsim.scenarios import solvency_packages

    paths = draw_paths(trustees_2026(), 20, seed=3)
    with pytest.warns(UserWarning, match="left out"):
        packages = solvency_packages(paths, target=1.01)  # no lever reaches more than every path
    assert packages == {}


def test_depletion_year_flows_reconcile():
    # In the year reserves run out, what came in (reserves, taxes, interest)
    # equals what went out (benefits paid and administration).
    r = baseline_model().run(baseline_paths())
    k = int(np.floor(r.ss_depletion_year[0])) - int(r.years[0])
    inflow = (r.ss_assets[0, k - 1] + r.ss_payroll_tax[0, k] + r.ss_general_revenue[0, k]
              + r.ss_benefit_tax[0, k] + r.ss_interest[0, k])
    outflow = r.paid_benefits[0, k] * r.ss_cost[0, k] / r.scheduled_benefits[0, k]
    assert r.ss_assets[0, k] == 0
    assert inflow == pytest.approx(outflow, rel=1e-9)


def test_depletion_histogram_counts_every_path(tmp_path):
    import json

    from fiscalsim.dashboard import export
    from fiscalsim.runner import futures, run_policies

    lib = library()
    paths = futures(200, seed=11)
    runs = run_policies({k: lib[k] for k in ["current_law", "payroll_plus_2", "cut_all_now"]}, paths=paths)
    export(runs, {}, paths, tmp_path, 200, 11)
    rows = json.loads((tmp_path / "depletion.json").read_text())
    for key in runs:
        total = sum(r["share"] for r in rows if r["option"] == key)
        assert total == pytest.approx(100, abs=0.05), key
