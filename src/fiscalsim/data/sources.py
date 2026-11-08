"""Every external file the model reads, and where it comes from.

`access="auto"` sources download with `python -m fiscalsim.data fetch`.
`access="manual"` sources sit on sites whose bot protection refuses scripted
requests (ssa.gov, cbo.gov, bls.gov). Download them in a browser from
`landing` and save them under data/manual/ with the given path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    path: str
    publisher: str
    title: str
    access: str = "auto"
    landing: str = ""


CENSUS = "https://www2.census.gov/programs-surveys"
OMB = "https://www.whitehouse.gov/wp-content/uploads/2026/04"
IRS = "https://www.irs.gov/pub/irs-soi"
FISCALDATA = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

# FRED mirrors BLS, BEA and Treasury series; fredgraph.csv needs no API key.
FRED_SERIES = {
    "CWUR0000SA0": "CPI-W, all items, not seasonally adjusted (drives the COLA)",
    "CPIAUCSL": "CPI-U, all items, seasonally adjusted",
    "UNRATE": "Civilian unemployment rate",
    "CIVPART": "Labor force participation rate",
    "GS10": "10-year Treasury constant maturity yield",
    "TB3MS": "3-month Treasury bill rate",
    "GDP": "Nominal GDP, quarterly SAAR",
    "GDPC1": "Real GDP, chained 2017 dollars",
    "GDPDEF": "GDP implicit price deflator",
    "COMPNFB": "Nonfarm business compensation per hour (index)",
    "OPHNFB": "Nonfarm business output per hour (index)",
    "A576RC1": "Wages and salaries, all workers (monthly SAAR)",
    "PAYEMS": "Total nonfarm payroll employment",
    "SPDYNTFRTINUSA": "Total fertility rate, United States (World Bank)",
    "FYGFGDQ188S": "Federal debt held by the public, percent of GDP",
    "A091RC1Q027SBEA": "Federal government interest payments",
}


def _fred(series_id: str, title: str) -> Source:
    return Source(
        key=f"fred_{series_id.lower()}",
        url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        path=f"fred/{series_id}.csv",
        publisher="Federal Reserve Bank of St. Louis (FRED)",
        title=title,
    )


SOURCES: list[Source] = [
    # --- Medicare Trustees Report (CMS). Released with the OASDI report and built
    # on the same intermediate economic and demographic assumptions.
    Source(
        "cms_mtr2026_report",
        "https://www.cms.gov/files/document/2026-medicare-trustees-report.pdf",
        "cms/2026-medicare-trustees-report.pdf",
        "CMS Office of the Actuary",
        "2026 Medicare Trustees Report",
    ),
    Source(
        "cms_mtr2026_tables",
        "https://www.cms.gov/files/zip/2026-expanded-supplementary-tables-figures.zip",
        "cms/2026-expanded-supplementary-tables-figures.zip",
        "CMS Office of the Actuary",
        "2026 Medicare Trustees Report, expanded and supplementary tables",
    ),
    # --- Population (Census Bureau)
    Source(
        "census_est2025_agesex",
        f"{CENSUS}/popest/datasets/2020-2025/national/asrh/nc-est2025-agesex-res.csv",
        "census/nc-est2025-agesex-res.csv",
        "U.S. Census Bureau",
        "Resident population by single year of age and sex, July 1 2020-2025 (Vintage 2025)",
    ),
    *[
        Source(
            f"census_np2023_{f}",
            f"{CENSUS}/popproj/datasets/2023/2023-popproj/np2023_{f}.csv",
            f"census/np2023_{f}.csv",
            "U.S. Census Bureau",
            f"2023 National Population Projections, file {f}",
        )
        for f in ["a1", "a2", "a3", "a4", "a5", "d1_mid", "d2_mid", "d3_mid", "d4_mid", "d5_mid"]
    ],
    # --- Federal budget history and current estimates (OMB, FY2027 Budget)
    *[
        Source(
            f"omb_{t}",
            f"{OMB}/{t}_fy2027.xlsx",
            f"omb/{t}_fy2027.xlsx",
            "Office of Management and Budget",
            title,
        )
        for t, title in [
            ("hist01z2", "Table 1.2: receipts, outlays, surpluses as percentages of GDP"),
            ("hist02z1", "Table 2.1: receipts by source"),
            ("hist02z3", "Table 2.3: receipts by source as percentages of GDP"),
            ("hist03z1", "Table 3.1: outlays by superfunction and function"),
            ("hist07z1", "Table 7.1: federal debt at the end of year"),
            ("hist08z4", "Table 8.4: outlays by budget enforcement category, percent of GDP"),
            ("hist10z1", "Table 10.1: gross domestic product and deflators"),
            ("hist13z1", "Table 13.1: Social Security and Medicare trust fund income, outgo, balances"),
        ]
    ],
    # --- Earnings and income distributions (IRS Statistics of Income)
    Source(
        "irs_w2_2023_t1",
        f"{IRS}/23in01w2all.xlsx",
        "irs/23in01w2all.xlsx",
        "IRS Statistics of Income",
        "Form W-2 statistics TY2023, Table 1: wage income by size and by multiple of the OASDI wage limit",
    ),
    Source(
        "irs_soi_2023_t11",
        f"{IRS}/23in11si.xls",
        "irs/23in11si.xls",
        "IRS Statistics of Income",
        "Individual returns TY2023, Table 1.1: income and tax by size of AGI",
    ),
    Source(
        "irs_soi_2023_t14",
        f"{IRS}/23in14ar.xls",
        "irs/23in14ar.xls",
        "IRS Statistics of Income",
        "Individual returns TY2023, Table 1.4: sources of income by size of AGI",
    ),
    # --- Debt and interest (Treasury Fiscal Data API)
    Source(
        "treasury_debt_to_penny",
        f"{FISCALDATA}/v2/accounting/od/debt_to_penny?sort=-record_date&page%5Bsize%5D=400&format=csv",
        "treasury/debt_to_penny.csv",
        "U.S. Treasury, Bureau of the Fiscal Service",
        "Debt to the Penny, most recent 400 business days",
    ),
    Source(
        "treasury_avg_interest_rates",
        f"{FISCALDATA}/v2/accounting/od/avg_interest_rates?sort=-record_date&page%5Bsize%5D=600&format=csv",
        "treasury/avg_interest_rates.csv",
        "U.S. Treasury, Bureau of the Fiscal Service",
        "Average interest rates on Treasury securities, most recent months",
    ),
    *[_fred(sid, title) for sid, title in FRED_SERIES.items()],
    # --- Manual downloads. ssa.gov, cbo.gov and bls.gov refuse scripted requests.
    Source(
        "ssa_tr2026_single_year",
        "https://www.ssa.gov/oact/TR/2026/SingleYearTRTables_TR2026.xlsx",
        "ssa/SingleYearTRTables_TR2026.xlsx",
        "SSA Office of the Chief Actuary",
        "2026 OASDI Trustees Report, supplemental single-year tables (Excel)",
        access="manual",
        landing="https://www.ssa.gov/oact/TR/2026/lrIndex.html",
    ),
    Source(
        "ssa_tr2026_report",
        "https://www.ssa.gov/oact/TR/2026/tr2026.pdf",
        "ssa/tr2026.pdf",
        "SSA Office of the Chief Actuary",
        "2026 OASDI Trustees Report (PDF)",
        access="manual",
        landing="https://www.ssa.gov/oact/TR/2026/",
    ),
    Source(
        "ssa_awi",
        "https://www.ssa.gov/oact/COLA/AWI.html",
        "ssa/AWI.html",
        "SSA Office of the Chief Actuary",
        "National average wage index series (save the page as HTML)",
        access="manual",
        landing="https://www.ssa.gov/oact/COLA/AWI.html",
    ),
    Source(
        "ssa_solvency_provisions",
        "https://www.ssa.gov/oact/solvency/provisions/index.html",
        "ssa/solvency_provisions/",
        "SSA Office of the Chief Actuary",
        "Estimated effects of individual solvency provisions (save each category page as HTML)",
        access="manual",
        landing="https://www.ssa.gov/oact/solvency/provisions/index.html",
    ),
    Source(
        "cbo_ltbo_2026",
        "https://www.cbo.gov/data/budget-economic-data",
        "cbo/ltbo_2026.xlsx",
        "Congressional Budget Office",
        "Long-Term Budget Outlook supplemental data, latest edition (Excel)",
        access="manual",
        landing="https://www.cbo.gov/data/budget-economic-data",
    ),
]

BY_KEY = {s.key: s for s in SOURCES}
