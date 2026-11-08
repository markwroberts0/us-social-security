# Methodology

Everything here is implemented in `src/fiscalsim/`. Years run 2025 (base, history) to 2100. Ages run 0 to 100, with 100 meaning 100 and over. Every state array carries a leading simulation axis, so one code path serves the deterministic baseline (one path) and the Monte Carlo (thousands).

"MTR" is the 2026 Medicare Trustees Report (`data/raw/cms/2026-medicare-trustees-report.pdf`). Its economic and demographic assumptions are the OASDI Trustees' intermediate assumptions (MTR p. 14), and it quotes the OASDI report's 75-year present values (Table V.F2, p. 217).

## 1. Population (`model.py`, step 1)

A cohort-component projection by single year of age and sex, with July 1 populations:

- **Base:** Census Vintage 2025 resident population, July 1, 2025 (341.8M).
- **Mortality.** Central death rates start from the Census 2023 projection's rates for 2025. They are scaled by 1.043 so deaths from mid-2024 to mid-2025 match the Census estimate (3.10M). Each year they fall by `exp(-r_t · s(a) / n_t)`:
  - `r_t` is the assumed decline in the age-sex-adjusted death rate (0.73% a year, MTR Table II.C1).
  - `s(a)` is a relative age shape: 2.2 for ages 0-14, 1.3 for 15-49, 1.2 for 50-64, 1.0 for 65-84, 0.7 at 85, 0.55 at 94 and 0.35 at 100. It is shaped like the Trustees' ultimate rates by age group.
  - `n_t` renormalizes the decline so the adjusted death rate, weighted by the 2020 population, falls by exactly `r_t`.

  The Census projection's own age pattern was tried first and rejected. It implies 2-2.5% annual improvement at ages 20-60. Implied period life expectancy at 65 rises from 18.6 (men) and 21.2 (women) in 2025 to 22.8 and 25.0 in 2100.
- **Fertility.** Births are `TFR_t · Σ_a f(a) · W(a)`, where `f` is the Census 2025 fertility-by-age shape normalized to 1 and `W` is the average female population at each age over the year. The TFR is 1.583 in 2025, the value implied by Census births (3.62M). It rises linearly to 1.75 in 2050 (MTR Table II.C1); the year the ultimate is reached is an assumption. The male share of births is 51.16% (Census).
- **Immigration.** Net immigrants are distributed with the Census 2025 age-sex pattern of net migration. The total is 1.26M in 2025 (Census, mid-2024 to mid-2025). It drops to 0.4M in 2026, then rises through 0.7M, 0.9M and 1.05M to the Trustees' ultimate 1.177M in 2030 (788,000 LPR plus 389,000 other, MTR Table II.C1). The 2026-2029 ramp is ours.

## 2. Economy (step 2)

- **Average wage index:** `AWI_t = AWI_{t-1} · (1 + CPI_t)(1 + RWD_t)`.
  - CPI-W averages 3.2% over 2026-2035 (MTR Table II.D1) and is 2.4% after (Table II.C1). 2026 uses CPI-W observed through September (3.4%); the shape within the decade is ours, holding the average.
  - The real-wage differential is 1.14% (Table II.C1). The ultimate nominal wage growth is 3.57%, which matches the MTR's average wage in covered employment.
- **Wage index history.** The 2024 AWI is SSA's $69,846.57 (the value behind the 2026 taxable maximum). Earlier years use wages per payroll job (FRED A576RC1/PAYEMS) scaled to that anchor, with 2020-21 bridged geometrically because pandemic job losses distorted the proxy.
- **Covered earnings:** `CE_t = Σ P_t(s,a) · e(s,a) · AWI_t/AWI_2025 · R_t · (1 − 1.2 · (u_t − ū_t))`.
  - `e` is CPS earnings per person by age and sex (calendar 2025), scaled so 2025 equals HI taxable payroll ($13,277B).
  - `R_t` is a residual fitted so the deterministic CE equals the Trustees' HI taxable payroll path (CMS tables). It falls to 0.95 by 2050 and absorbs labor-force and coverage trends the fixed age profile misses.
  - The last factor is the cyclical effect of unemployment above its baseline path.
- **Taxable payroll:** `TP_t = CE_t · τ_t`. The ratio starts at 0.8317 (OASDI $11,043B / HI $13,277B in 2025, MTR p. 216) and declines 2% over 2026-2035 as wages above the cap grow, then holds. This path is an assumption (see §8).
- **GDP:** `CE_t` times the Trustees' ratio of GDP to HI taxable payroll for that year.

## 3. Social Security benefits (step 3)

Per-person benefits by age and sex in 2025 come from the March 2026 CPS ASEC. They are calendar-2025 Social Security income divided by the population at each age.

- Interview ages are shifted to mid-2025 ages: a person aged A in March was A−1 on July 1 with probability 0.71.
- The 80-84 and 85+ groups are spread flat.
- The profile is scaled by 1.154 so the total equals calendar-2025 OASDI benefits. That total is $1,592B, three quarters of FY2025 plus one quarter of FY2026 (OMB Table 13.1). CPS under-reports Social Security, as surveys do.

**Projection uses cohort indexes.** For ages 62 and over, the cohort born in year c has index `X = AWI_{c+60}/AWI_2023 · Π COLA · F(c)`:

- the wage level at 60, which sets the PIA bend points and indexing;
- every COLA since 62;
- `F(c)`, the cohort's average claiming factor relative to a full retirement age of 67.

`F` uses the statutory FRA schedule, the actuarial reduction and delayed credits, and a fixed claiming distribution: 24% at 62, 8% at 63, 9% at 64, 13% at 65, 13% at 66, 17% at 67, 5% at 68, 3% at 69 and 8% at 70. That distribution is an assumption close to recent patterns.

The age-sex shape is `θ(s,a) = b_2025(s,a) / X_2025(a)`. Each later year:

- cohorts age one year and gain that December's COLA;
- the cohort turning 62 enters at `AWI_{t−2}/AWI_2023 · F · (policy factors) · cohort_level_t`;
- benefits under 62 (disability, young survivors) follow a stock index in which 15% turns over each year at current wage levels and the rest gains the COLA.

The COLA for December 2025 (2.8%) and December 2026 (3.6%) are taken from actual CPI-W. Later COLAs equal assumed CPI. COLAs are recomputed from Q3 CPI-W with SSA's rule and match every published COLA since 1984 except 1990, which is off by 0.1.

Two fitted trends adjust `θ` (§8):

- disability benefits per person rise 30% by 2045, as incidence recovers;
- each new cohort's benefit level rises 0.14% a year through 2050.

Administration and the railroad interchange add 0.83% of benefits.

## 4. Trust fund (step 3, continued)

- **Income** is `12.4% · TP` (plus policy changes) and benefit taxes, `tob_t · scheduled benefits`. `tob` is 3.83% in 2025 (OMB, FY2025). It dips 6% while the 2025 senior deduction is in force (2026-2028), then approaches 5.9% with a 25-year time constant (fitted, §8).
- **Interest** is earned at the portfolio yield `y_t = y_{t−1} + 0.042 (i_t − y_{t−1})`. `i_t` is the new special-issue rate. The 0.042 is fitted to OMB's projected OASI interest and balances for FY2026-2031, so the yield rises slowly from 2.6% (FY2025 actual) as old bonds mature. Starting reserves are $2,577B at the end of 2025.
- **Depletion.** When reserves cannot cover a year's cost, the depletion date is the year plus the fraction of it that reserves last. Benefits for the year fall to what income and remaining reserves cover, and to income alone after that.
- **Discount rate for present values.** This follows the portfolio yield while reserves last, then converges halfway each year to new-issue rates. After depletion the Trustees' hypothetical fund, which still pays scheduled benefits for actuarial measures, borrows at those rates. When comparing policies, every policy is valued at the current-law path's discount rates, as the actuaries score provisions.
- **Real interest rate:** 1.0% in 2025, rising linearly to 2.3% in 2040 (Table II.C1, "by year 15"). The starting value is fitted: at 1.0%, the HI discount path reproduces both MTR present values, HI payroll ($810.5T) and GDP ($1,853.5T), with one parameter. It implies a 4.5% new-issue rate in 2026.

**Actuarial balance** (`metrics.py`), 2026-2100:

- summarized income rate: (reserves + PV of payroll and benefit taxes) / PV of taxable payroll;
- summarized cost rate: (PV of cost + PV of an ending fund equal to the next year's cost) / PV of taxable payroll.

Dedicated general revenue counts as income; "fill the gap" transfers do not.

## 5. Medicare (step 4)

- **HI cost and income rates** follow the Trustees' paths (CMS tables). Simulated cost is the Trustees' cost scaled by three factors: simulated 65+ population over baseline, simulated over baseline wage index, and the cumulative excess health-cost shock.
- **HI income** is the Trustees' income rate times simulated covered earnings.
- **HI reserves** start at $255.7B (end of 2025). The HI yield starts at 3.7% (MTR p. 50: $9.1B of interest in 2025). Depletion works as for OASDI.
- **SMI cost** is the Trustees' Parts B and D incurred cost under the same scaling. Premiums cover 24.9% (FY2025, MTR Table V.F1); the rest is general revenue.

## 6. Budget and debt (step 5, `budget.py`)

Revenue is individual income taxes, corporate taxes, other receipts and other social insurance, all as GDP shares, plus modeled OASDI payroll taxes and 89% of HI non-interest income (the payroll part). Benefit taxes are inside individual income taxes.

Outlays are Social Security benefits paid plus administration, HI cost, SMI net of premiums, and discretionary, Medicaid and other mandatory spending as GDP shares, plus net interest.

Starting shares are FY2025 actuals (OMB Tables 2.3 and 8.4). The drift rules:

- individual income taxes rise 0.02 points of GDP a year (real bracket creep), capped at 10%;
- corporate taxes go to 1.4% by 2030;
- other receipts go to 1.4% by 2027, since tariffs roughly doubled customs receipts;
- discretionary spending falls to 5.6% by 2035 (appropriations grow with inflation), then holds;
- Medicaid falls to 2.0% by 2034 (2025 reconciliation act), then grows 0.015 points a year;
- other mandatory spending falls to 2.6% by 2035.

**Debt** starts at $30.85T held by the public (Treasury, December 31, 2025). Its average rate starts at 3.36% (Treasury, December 2025) and closes 20% of the gap each year to the market rate on new borrowing. That market rate is the new-issue rate minus 0.5 points, plus 2 basis points per point of debt-to-GDP above 2025, a premium capped at 3 points.

These rules stand in for CBO's long-term outlook and are the weakest part of the model.

## 7. Uncertainty (`stochastic.py`)

Each path is the baseline plus a long-run level offset, ramped in over the years shown, plus annual AR noise.

| Driver | Annual noise | Long-run level s.d. | Ramp |
|---|---|---|---|
| CPI-W | AR(1), φ 0.42, σ 1.43 pts (1983-2025) | 0.30 pts | 10 y |
| Real wage growth | AR(1), φ 0.21, σ 1.39 pts (1983-2025) | 0.30 pts | 10 y |
| Unemployment | AR(2), φ 0.95 / −0.26, σ 1.06 pts (1960-2025) | 0.25 pts | 10 y |
| Real interest | AR(1), φ 0.86, σ 1.16 pts (ex-ante 10-year, 1983-2025) | 0.25 pts | 15 y |
| Fertility | AR(1), φ 0.97 (capped), σ 0.037 (1976-2024) | 0.175 | 25 y |
| Mortality decline | AR(1), φ 0.3, σ 1.0 pt (judgment) | 0.24 pts | immediate |
| Net immigration | AR(1), φ 0.6, σ 250k (judgment) | 219k | 5 y |
| Excess health cost growth | AR(1), φ 0.5, σ 1.0 pt (judgment) | 0.30 pts | 10 y |
| Disability prevalence (log) | AR(1), φ 0.8, σ 0.03 (judgment) | 0.10 | 15 y |

- **Correlated shocks.** Innovations to CPI, real wages, unemployment and the real rate are correlated as estimated from 1984-2025 residuals. For example, CPI with real wages is −0.58, and CPI with unemployment is −0.49.
- **Long-run levels.** Level standard deviations are a quarter of the Trustees' low-to-high range (MTR Table II.C1), so their low/high values sit near the 2.5th and 97.5th percentiles. Draws are independent across variables, unlike the Trustees' scenarios, which move every assumption the same way at once.
- **First year.** 2026 noise is scaled by 0.15, since the year is mostly observed.
- **Bounds:** fertility 1.0-2.8; unemployment 2.5%-15%; net immigration −0.5M to 4M.

## 8. Calibration (`calibrate.py`)

**Targets** are the Trustees' 75-year summarized rates from MTR Table V.F2:

- income: $91.9T / $690.4T = 13.31% of taxable payroll;
- cost: $123.7T / $690.4T = 17.92%.

The model matches ratios rather than dollar present values. Dollar PVs depend on OASDI discount factors that appear only in SSA's tables, and the payroll PV is too sensitive to them: across plausible conventions, the implied 2035 taxable ratio ranges from −7% to +9%.

**Fitted, one target each:**

| Parameter | Value | Target |
|---|---|---|
| Payroll residual `R_t` | 1.00 (2025), 0.95 (2050), 0.945 (2100) | HI taxable payroll path (CMS) |
| Real interest rate in 2025 | 1.0% | PV of HI payroll and of GDP (MTR Table V.F2) |
| Benefit-tax share, long run | 5.9% of benefits | summarized income rate |
| New-cohort benefit level growth | 0.14% a year to 2050 (3.6% total) | summarized cost rate |

**Set by judgment:**

- disability rise: 30% by 2045, in line with the Trustees' assumed recovery in incidence. CPS identifies only 62% of disability benefits, so the multiplier on the CPS-identified part is 1.48;
- taxable ratio decline: 2% over the decade;
- claiming distribution;
- mortality age shape;
- 2026-2029 immigration;
- fertility timing.

**Results.**

- **OASDI:** reserves deplete in 2034.7; the 75-year balance is −4.43% of payroll.
- **HI:** depletes in 2033.4; the Trustees say "second quarter of 2033".

**What the SSA tables would replace.** SSA's single-year tables (`data/manual/ssa/`) carry the Trustees' own paths: taxable payroll, effective yields, cost, income, fertility, immigration and disability. With them, the judgment calls above become data, the two cost and income fits become checks, and the depletion year can be compared with the Trustees' directly.

## 9. Policy levers (`policy.py`, `scenarios.py`)

Every option is scored statically: claiming ages, work and earnings are unchanged.

- **Payroll rate.** Added to 12.4%, with an optional phase-in.
- **Taxable maximum.** Set as a multiple of the current-law cap, or removed. The extra taxable share comes from the IRS 2023 W-2 wage distribution (a truncated Pareto inside each published bin). The upper tail grows with the taxable ratio's decline, so the options tax what the baseline leaves untaxed. Benefit credit raises each cohort's initial benefit by `credit factor × (average extra taxable share over the 35 years before 62) / 0.42`.
- **Donut.** Wages above a fixed dollar threshold are taxed. The gap closes as the indexed cap passes the threshold.
- **Retirement age.** FRA rises 2 months a year from the cohort turning 62 in the start year. Benefits fall through the actuarial reduction at fixed claiming ages.
- **COLA adjustment.** Added to every COLA from the start year (chained CPI is −0.3).
- **New-award cut.** Proportional and phased in. The "progressive" version bites 75% as hard in aggregate, approximating progressive price indexing.
- **Across-the-board cut.** Applies to every benefit from the start year.
- **General revenue.**
  - "fill the gap" pays scheduled benefits, either by borrowing or with a same-size income tax;
  - "share of payroll" dedicates a fixed transfer.
- **Rest of the budget.** Income tax changes and spending cuts, as shares of GDP.
- **Immigration.** A multiplier on the path.

**Solving.**

- `scenarios.solve` finds the value of one lever that zeroes the deterministic balance.
- `scenarios.solve_probability` finds the smallest value that keeps reserves positive through 2100 in a target share of paths. It bisects over fixed paths, where the share is monotone in the lever.
