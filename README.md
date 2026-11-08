# us-social-security

A Monte Carlo model of Social Security, Medicare, federal taxes and the national debt, built to compare ways of keeping Social Security solvent under uncertain economic and demographic futures.

The baseline is the 2026 Trustees' intermediate projection, rebuilt from public data and calibrated to the Trustees' published totals. Around it, the model simulates thousands of futures for fertility, mortality, immigration, inflation, wages, unemployment, interest rates, health costs and disability. It then scores policy options on every one of them.

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m fiscalsim baseline          # deterministic baseline and its fit to the Trustees
.venv/bin/python -m fiscalsim list              # named policy options
.venv/bin/python -m fiscalsim run --sims 2000   # Monte Carlo over every option; writes out/report.html
.venv/bin/python -m pytest
```

A full run (19 options and 4 solved packages over 2,000 futures) takes about a minute on a laptop. The processed data is committed, so none of this needs a network connection. To refresh the data, run `python -m fiscalsim.data fetch` and then `python -m fiscalsim.data build`.

## What it finds (seed 2026, 2,000 futures)

**Current law.** Combined Social Security (OASDI) reserves run out in **2034.7** (median). In 80% of simulated futures that happens between 2033.7 and 2036.0. The near-term date is this certain because the next decade's retirees and workers are already born. After depletion, income covers about 81% of scheduled benefits in 2035, 65% in 2075 and 62% in 2100. The 75-year actuarial deficit is **4.48% of taxable payroll** (80% range 3.31 to 5.98), larger than the 2025 Trustees' 3.82%. The main reason is the Trustees' cut in assumed fertility from 1.9 to 1.75. Medicare's HI trust fund runs out in 2033.

**Single options** (median change in the 75-year balance, percent of taxable payroll):

| Option | Balance change | Reserves run out (median) |
|---|---:|---:|
| Payroll tax +1 point (12.4% to 13.4%) in 2027 | +0.98 | 2038 |
| Payroll tax +2 points | +1.96 | 2047 |
| Cover 90% of wages (cap about 1.55x higher, over 10 years) | +0.81 | 2036 |
| No taxable maximum, newly taxed wages credited | +2.19 | 2051 |
| No taxable maximum, no credit | +2.72 | 2053 |
| Tax wages above $400,000 (not indexed) | +2.49 | 2043 |
| Retirement age to 69 (2 months a year from 2027) | +1.48 | 2035 |
| Retirement age to 70 | +2.06 | 2035 |
| Chained CPI for COLAs | +0.52 | 2035 |
| Slower initial benefits for higher earners (15% overall by 2059) | +1.39 | 2035 |
| 50% more immigration | +0.62 | 2035 |

Benefit-side changes that phase in over decades barely move the 2034 date, even when they close a third to nearly half of the 75-year gap. Revenue changes that start in 2027 move it immediately. Paying full benefits by borrowing adds about 26 points of GDP to the debt by 2050. Raising the same amount through income taxes leaves the debt where current law has it.

**What it takes to stay solvent through 2100 in 80% of futures** (each package solved for its last lever):

| Package | Required |
|---|---|
| Payroll tax only | +5.53 points from 2027 (12.4% to 17.9%) |
| Benefit cuts only | every benefit 30.4% lower from 2027 |
| No taxable maximum (with credit), then rate | +3.20 points |
| Cover 90% of wages, chained CPI, retirement age 68, then rate | +3.26 points |

These exceed what closes the deterministic 75-year balance (a 4.51-point rate rise from 2027, or a 26.1% cut to every benefit). Costs keep rising after 2075, and the requirement is solvency in most futures, not on average.

The debt projections are the weakest part of the model (see Limitations). Under current law, debt held by the public reaches about 170% of GDP in 2050, with an 80% range of 117% to 250%.

## How it works

Annual steps from 2025 to 2100, with a simulation axis on every array:

1. **Population.** A cohort-component projection by single year of age and sex. It starts from Census Vintage 2025, uses Census age patterns for fertility, mortality and migration, and scales them to the Trustees' totals: fertility 1.75, death rates falling 0.73% a year, net immigration 1.18M.
2. **Economy.** Covered earnings combine the population, a CPS age-sex earnings profile and the wage index. A residual makes them follow the Trustees' HI taxable payroll path. Taxable payroll applies the cap. Taxable-maximum options use a wage distribution fitted to IRS W-2 data.
3. **Social Security.** Per-person benefits by age and sex come from the March 2026 CPS, scaled to 2025 Treasury totals. They are projected with cohort indexes: wage-indexed at 62 (including the full retirement age schedule), then raised by COLAs. Payroll and benefit-tax income feed a combined OASDI trust fund. At depletion, benefits fall to what income covers.
4. **Medicare.** HI and SMI follow the Trustees' cost paths, scaled by the simulated 65+ population, wages and excess health cost growth. HI has its own trust fund; SMI's general revenue share goes to the budget.
5. **Budget and debt.** Other revenue and spending are GDP shares from FY2025 actuals with simple current-law drift. Net interest is computed from the debt, with a rate that adjusts as debt rolls over and rises 2 basis points per point of debt-to-GDP.

The Monte Carlo adds two things to each assumption: a long-run level offset, sized so the Trustees' low- and high-cost values sit near ±2 standard deviations, and annual AR noise estimated from history (FRED 1983-2025 for the economy, with correlated shocks).

See [docs/methodology.md](docs/methodology.md) for the equations, calibration and every judgment call.

## Fit to the 2026 Trustees

From the Medicare Trustees Report, which shares the OASDI report's assumptions and quotes its present values:

| Measure | Model | Trustees |
|---|---:|---:|
| OASDI summarized income rate, 2026-2100 | 13.31% | 13.31% |
| OASDI summarized cost rate, 2026-2100 | 17.92% | 17.92% |
| PV of HI taxable payroll | $810.8T | $810.5T |
| PV of GDP | $1,853.5T | $1,853.5T |
| HI reserves depleted | 2033 | 2033 |

Several fitted pieces sit behind these fits; each is described in the methodology: the near-term real interest rate, the benefit-tax share, a 30% rise in disability benefits by 2045, and a 0.14%-a-year rise in new retirees' benefit levels. Dollar present values for OASDI are not matched, because they depend on discount factors published only in SSA's tables.

## Data

All sources are listed in `src/fiscalsim/data/sources.py` and logged with hashes in `data/raw/MANIFEST.json`:

- **Census:** Vintage 2025 population, births, deaths and migration; 2023 national projections; March 2026 CPS ASEC microdata.
- **CMS:** 2026 Medicare Trustees Report and its tables.
- **OMB:** FY2027 Historical Tables.
- **IRS:** SOI W-2 and individual income tables for 2023.
- **Treasury:** Fiscal Data API.
- **FRED:** BLS and BEA series.

**Manual downloads.** ssa.gov, cbo.gov and bls.gov refuse scripted requests, and this project does not work around that. Saving these files by hand under `data/manual/` would replace several stand-ins (`python -m fiscalsim.data status` lists them with links):

- SSA 2026 Trustees Report, single-year tables (`ssa/SingleYearTRTables_TR2026.xlsx`): the Trustees' own year-by-year cost, income, taxable payroll, effective interest rates and demographic paths. These would replace the 2026-2029 immigration ramp, the fertility path, the taxable-ratio path and the disability and cohort-level fits.
- SSA average wage index page (`ssa/AWI.html`): replaces a wage-per-job proxy for the historical wage index.
- SSA solvency provisions pages: the actuaries' own scores for each option, as a check on the table above.
- CBO Long-Term Budget Outlook data: would replace the simple budget paths behind the debt projections.

## Limitations

- **Static scoring.** People do not change when they claim, how much they work or what they earn in response to a policy. This matches how the Social Security actuaries score individual provisions.
- **Benefits are modeled per person by age and sex, not by earnings history.** Formula changes are approximated: "slower growth for higher earners" is an aggregate cut that bites about 75% as hard as a flat one. Benefit credit for newly taxed wages uses an aggregate PIA/AIME ratio of 0.42.
- **OASI and DI are combined,** as the hypothetical combined fund. OASI alone runs out about a year earlier.
- **The budget outside Social Security and Medicare is a set of GDP shares,** not a model of the tax code or appropriations. Read the debt paths as directions and relative sizes. Past about 2060, the interest feedback makes current-law debt explode, as it does in CBO's own extended projections.
- **Uncertainty is sized by judgment in places.** The long-run levels come from the Trustees' low/high range. Annual mortality, immigration, health-cost and disability noise are judgment calls, stated in `stochastic.py`.
