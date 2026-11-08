"""Turn raw downloads into tidy CSVs under data/processed.

Run with `python -m fiscalsim.data build`. The outputs are small and committed,
so the model never parses Excel, PDF or zip files at run time.
"""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from .. import DATA

RAW = DATA / "raw"
OUT = DATA / "processed"
SEXES = {1: "male", 2: "female"}


def _num(v) -> float:
    """'1,234' -> 1234.0, '4.5%' -> 4.5, dashes and '*' -> nan."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "").replace("%", "")
    if s in ("", "—", "-", "*", "nan", "n/a") or set(s) <= {"."}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _year(v) -> int | None:
    m = re.match(r"^\s*(\d{4})", str(v))
    return int(m.group(1)) if m else None


def _write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, float_format="%.10g")
    print(f"  wrote {name} ({len(df)} rows)")


# --- Census -------------------------------------------------------------------


def population() -> None:
    """July 1 resident population by single year of age (100 = 100+) and sex."""
    df = pd.read_csv(RAW / "census/nc-est2025-agesex-res.csv")
    df = df[df.SEX.isin(SEXES) & (df.AGE != 999)]
    long = df.melt(
        id_vars=["SEX", "AGE"],
        value_vars=[f"POPESTIMATE{y}" for y in range(2020, 2026)],
        var_name="year",
        value_name="pop",
    )
    long["year"] = long.year.str[-4:].astype(int)
    long["sex"] = long.SEX.map(SEXES)
    _write(long[["year", "sex", "AGE", "pop"]].rename(columns={"AGE": "age"}), "census_population.csv")


def components() -> None:
    """National births, deaths and net international migration, July-June years."""
    df = pd.read_csv(RAW / "census/NST-EST2025-ALLDATA.csv")
    us = df[df.NAME == "United States"].iloc[0]
    rows = [
        {
            "year": y,
            "pop": us[f"POPESTIMATE{y}"],
            "births": us[f"BIRTHS{y}"],
            "deaths": us[f"DEATHS{y}"],
            "net_international_migration": us[f"INTERNATIONALMIG{y}"],
        }
        for y in range(2021, 2026)
    ]
    _write(pd.DataFrame(rows), "census_components.csv")


def census_projection_inputs() -> None:
    """Age patterns from the 2023 National Population Projections (middle series).

    The model takes shapes from these files (fertility by mother's age, death
    rates by age, the age-sex mix of migrants) and scales them to the Trustees'
    levels, which differ from the Census Bureau's.
    """
    a1 = pd.read_csv(RAW / "census/np2023_a1.csv")
    a1 = a1[a1.GROUP == 0].drop(columns="GROUP")
    asfr = a1.melt(id_vars="YEAR", var_name="age", value_name="asfr")
    asfr["age"] = asfr.age.str[5:].astype(int)
    _write(asfr.rename(columns={"YEAR": "year"}).sort_values(["year", "age"]), "census_np2023_asfr.csv")

    a2 = pd.read_csv(RAW / "census/np2023_a2.csv")
    a2 = a2[(a2.NATIVITY == 0) & (a2.GROUP == 0) & a2.SEX.isin(SEXES)]
    mx = a2.drop(columns=["NATIVITY", "GROUP"]).melt(id_vars=["SEX", "YEAR"], var_name="age", value_name="mx")
    mx["age"] = mx.age.str[5:].astype(int)
    mx["sex"] = mx.SEX.map(SEXES)
    _write(
        mx[["YEAR", "sex", "age", "mx"]].rename(columns={"YEAR": "year"}).sort_values(["year", "sex", "age"]),
        "census_np2023_mx.csv",
    )

    d4 = pd.read_csv(RAW / "census/np2023_d4_mid.csv")
    d4 = d4[(d4.RACE_HISP == 0) & d4.SEX.isin(SEXES)].drop(columns=["RACE_HISP", "TOTAL_NIM"])
    nim = d4.melt(id_vars=["SEX", "YEAR"], var_name="age", value_name="nim")
    nim["age"] = nim.age.str[4:].astype(int)
    nim["sex"] = nim.SEX.map(SEXES)
    _write(
        nim[["YEAR", "sex", "age", "nim"]].rename(columns={"YEAR": "year"}).sort_values(["year", "sex", "age"]),
        "census_np2023_nim.csv",
    )

    d2 = pd.read_csv(RAW / "census/np2023_d2_mid.csv")
    d2 = d2[(d2.RACE_HISP == 0) & d2.SEX.isin(SEXES)]
    births = d2.pivot(index="YEAR", columns="SEX", values="BIRTHS").rename(columns=SEXES).reset_index()
    births.columns.name = None
    _write(births.rename(columns={"YEAR": "year"}), "census_np2023_births.csv")


def cps_profiles() -> None:
    """Social Security receipt and earnings by age and sex, CPS ASEC March 2026.

    Amounts are calendar-2025 income. Ages are as of the March 2026 interview,
    with 80 meaning 80-84 and 85 meaning 85 and over. Reason codes (RESNSS1):
    1 retired, 2 disabled, 3 widowed, 4 spouse, 5 surviving child,
    6 dependent child, 7 on behalf of children, 8 other.
    """
    cols = [
        "A_AGE", "A_SEX", "A_MARITL", "MARSUPWT", "SS_YN", "SS_VAL", "RESNSS1",
        "WSAL_VAL", "SEMP_VAL", "FRSE_VAL", "PEARNVAL", "WKSWORK",
    ]
    with zipfile.ZipFile(RAW / "census/asecpub26csv.zip") as z:
        with z.open("pppub26.csv") as f:
            p = pd.read_csv(f, usecols=cols)
    p["w"] = p.MARSUPWT / 100.0
    p["ss"] = np.where(p.SS_YN == 1, p.SS_VAL, 0.0)
    p["earn"] = p.WSAL_VAL.clip(lower=0) + p.SEMP_VAL.clip(lower=0) + p.FRSE_VAL.clip(lower=0)
    reasons = {1: "retired", 2: "disabled", 3: "widowed", 4: "spouse", 5: "child", 6: "child", 7: "child", 8: "other"}
    p["reason"] = p.RESNSS1.map(reasons).fillna("none")

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g.w
        out = {
            "weighted_n": w.sum(),
            "sample_n": len(g),
            "ss_recipient_share": (w * (g.SS_YN == 1)).sum() / w.sum(),
            "ss_per_capita": (w * g.ss).sum() / w.sum(),
            "earner_share": (w * (g.earn > 0)).sum() / w.sum(),
            "earnings_per_capita": (w * g.earn).sum() / w.sum(),
            "widowed_share": (w * (g.A_MARITL == 4)).sum() / w.sum(),
        }
        for r in ["retired", "disabled", "widowed", "spouse", "child", "other"]:
            out[f"ss_per_capita_{r}"] = (w * g.ss * (g.reason == r)).sum() / w.sum()
        return pd.Series(out)

    prof = p.groupby(["A_SEX", "A_AGE"]).apply(agg).reset_index()
    prof["sex"] = prof.A_SEX.map(SEXES)
    prof = prof.rename(columns={"A_AGE": "age_at_interview"}).drop(columns="A_SEX")
    _write(prof[["sex", "age_at_interview"] + [c for c in prof.columns if c not in ("sex", "age_at_interview")]],
           "cps_asec2026_profiles.csv")
    totals = pd.DataFrame(
        [{
            "weighted_persons": p.w.sum(),
            "ss_recipients": (p.w * (p.SS_YN == 1)).sum(),
            "ss_total_dollars": (p.w * p.ss).sum(),
            "earners": (p.w * (p.earn > 0)).sum(),
            "earnings_total_dollars": (p.w * p.earn).sum(),
        }]
    )
    _write(totals, "cps_asec2026_totals.csv")


# --- IRS ----------------------------------------------------------------------


def irs_wage_bins() -> None:
    """W-2 wage income by size of wages, tax year 2023 (Table 1.B)."""
    df = pd.read_excel(RAW / "irs/23in01w2all.xlsx", sheet_name="Table 1.B", header=None)
    rows = []
    for _, r in df.iterrows():
        label = str(r[0]).strip()
        m = re.match(r"^\$([\d,]+) (under \$([\d,]+)|and over)$", label)
        if not m:
            continue
        lo = float(m.group(1).replace(",", ""))
        hi = float(m.group(3).replace(",", "")) if m.group(3) else np.inf
        rows.append({
            "lower": lo, "upper": hi,
            "taxpayers": _num(r[1]), "wages": _num(r[2]) * 1000,
            "women": _num(r[3]), "women_wages": _num(r[4]) * 1000,
            "men": _num(r[5]), "men_wages": _num(r[6]) * 1000,
        })
    _write(pd.DataFrame(rows), "irs_w2_2023_wage_bins.csv")


# --- FRED ---------------------------------------------------------------------


def _fred(series: str) -> pd.Series:
    df = pd.read_csv(RAW / f"fred/{series}.csv")
    date_col = df.columns[0]
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def fred_annual() -> None:
    """Calendar-year averages of every FRED series, plus Q3 CPI-W for COLAs."""
    from .sources import FRED_SERIES

    cols = {}
    for sid in FRED_SERIES:
        s = _fred(sid)
        cols[sid] = s.groupby(s.index.year).mean()
    df = pd.DataFrame(cols)
    df.index.name = "year"
    # Count only complete calendar years for monthly and quarterly series. A
    # past year may miss one month: the October 2025 shutdown left no October
    # CPI or jobs report, and an 11-month average is still the best estimate.
    for sid in FRED_SERIES:
        s = _fred(sid)
        counts = s.groupby(s.index.year).size()
        freq = counts.median()
        current = s.index.max().year
        short = counts[(counts < freq - (1 if freq == 12 else 0)) | (counts.index == current) & (counts < freq)]
        df.loc[short.index, sid] = np.nan
    cpiw = _fred("CWUR0000SA0")
    q3 = cpiw[cpiw.index.month.isin([7, 8, 9])]
    q3_counts = q3.groupby(q3.index.year).size()
    df["CPIW_Q3"] = q3.groupby(q3.index.year).mean()[q3_counts == 3]
    _write(df.reset_index(), "fred_annual.csv")

    # The current, incomplete year: average to date against the same months a
    # year earlier, so the baseline can use 2026 as observed so far.
    rows = []
    for sid in FRED_SERIES:
        s = _fred(sid)
        last = s.index.max()
        cur = s[s.index.year == last.year]
        prev = s[(s.index.year == last.year - 1) & (s.index.month <= last.month)]
        rows.append({
            "series": sid, "year": last.year, "through": last.strftime("%Y-%m-%d"),
            "avg_to_date": cur.mean(), "prior_year_same_months": prev.mean(),
        })
    _write(pd.DataFrame(rows), "fred_year_to_date.csv")


def cola_history() -> None:
    """Social Security COLAs computed the way SSA does, from Q3 CPI-W.

    The COLA effective in December of year y is the rise in average Q3 CPI-W
    since the last year a COLA was paid, rounded to 0.1 percent. No COLA is
    paid when that rise is under 0.1 percent, and the base quarter then stays.
    """
    fa = pd.read_csv(OUT / "fred_annual.csv").set_index("year")
    q3 = fa.CPIW_Q3.dropna()
    # Before 2007 BLS published CPI-W to one decimal. Rounding the Q3 average
    # to two decimals then (three since) reproduces every published COLA from
    # 1984 to 2025 except 1990 (5.3 here, 5.4 published), likely a later
    # revision to the 1989-90 index values.
    q3 = pd.Series([round(v, 2 if y < 2007 else 3) for y, v in q3.items()], index=q3.index)
    rows, base = [], None
    for y, v in q3.items():
        if y < 1975:
            continue
        if base is None:
            base = q3.loc[y - 1]
        pct = round((v / base - 1) * 100 + 1e-9, 1)
        cola = pct if pct >= 0.1 else 0.0
        rows.append({"year": int(y), "cola_pct": cola})
        if cola > 0:
            base = v
    _write(pd.DataFrame(rows), "cola_history.csv")


# --- CMS (2026 Medicare Trustees Report tables) ---------------------------------


def _cms_csv(member: str) -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "cms/2026-expanded-supplementary-tables-figures.zip") as z:
        raw = z.read(f"CSV/{member}").decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw), skiprows=1, header=0, dtype=str)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df[df.iloc[:, 0].map(_year).notna()].copy()
    df.insert(0, "year", df.iloc[:, 0].map(_year))
    return df.drop(columns=df.columns[1])


def cms() -> None:
    pay = _cms_csv("Ratio of HI Taxable Payroll to GDP.csv")
    out = pd.DataFrame({
        "year": pay.year,
        "hi_taxable_payroll": pay.iloc[:, 1].map(_num) * 1e6,
        "gdp": pay.iloc[:, 2].map(_num) * 1e6,
    })
    rates = _cms_csv("HI Cost and Income Rates.csv")
    rates = pd.DataFrame({"year": rates.year, "hi_cost_rate": rates.iloc[:, 1].map(_num),
                          "hi_income_rate": rates.iloc[:, 2].map(_num)})
    enr = _cms_csv("Medicare Enrollment by Part and in Total.csv")
    enr = pd.DataFrame({"year": enr.year, "enroll_a": enr.iloc[:, 1].map(_num) * 1e6,
                        "enroll_b": enr.iloc[:, 2].map(_num) * 1e6, "enroll_d": enr.iloc[:, 3].map(_num) * 1e6,
                        "enroll_total": enr.iloc[:, 4].map(_num) * 1e6})
    gdp_share = _cms_csv("HI and SMI Incurred Expenditures as a Percentage of the Gross Domestic Product.csv")
    gdp_share = pd.DataFrame({
        "year": gdp_share.year,
        "hi_incurred": gdp_share.iloc[:, 1].map(_num) * 1e6,
        "smi_b_incurred": gdp_share.iloc[:, 2].map(_num) * 1e6,
        "smi_d_incurred": gdp_share.iloc[:, 3].map(_num) * 1e6,
    })
    per_ben = _cms_csv("HI and SMI Average per Beneficiary Costs.csv")
    per_ben = pd.DataFrame({"year": per_ben.year, "hi_per_beneficiary": per_ben.iloc[:, 1].map(_num),
                            "smi_b_per_beneficiary": per_ben.iloc[:, 2].map(_num),
                            "smi_d_per_beneficiary": per_ben.iloc[:, 3].map(_num)})
    tfr = _cms_csv("Ratio of Assets at the Beginning of the Year to Expenditures During the Year for the HI Trust Fund.csv")
    tfr = pd.DataFrame({"year": tfr.year, "hi_trust_fund_ratio": tfr.iloc[:, 1].map(_num)})
    merged = out
    for t in (rates, enr, gdp_share, per_ben, tfr):
        merged = merged.merge(t, on="year", how="outer")
    _write(merged.sort_values("year"), "cms_tr2026_medicare.csv")

    ops = _cms_csv("Operations of the Hospital Insurance (HI) Trust Fund For Calendar Years.csv")
    ops_cols = {c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in ops.columns if c != "year"}
    ops = ops.rename(columns=ops_cols)
    for c in ops_cols.values():
        ops[c] = ops[c].map(_num)
    ops = ops.dropna(axis=1, how="all")
    _write(ops, "cms_tr2026_hi_operations.csv")


# --- OMB (FY2027 Budget historical tables) --------------------------------------


def _omb_rows(table: str, ncols: int, names: list[str]) -> pd.DataFrame:
    df = pd.read_excel(RAW / f"omb/{table}_fy2027.xlsx", header=None)
    rows = []
    for _, r in df.iterrows():
        y = _year(r[0])
        if y is None:
            continue
        rec = {"fiscal_year": y, "estimate": "estimate" in str(r[0]).lower()}
        rec.update({n: _num(r[i + 1]) for i, n in enumerate(names[:ncols])})
        rows.append(rec)
    return pd.DataFrame(rows)


def omb() -> None:
    summary = _omb_rows("hist01z2", 4, ["gdp_billions", "receipts", "outlays", "surplus"])
    receipts = _omb_rows(
        "hist02z3", 10,
        ["individual_income", "corporate_income", "social_insurance", "social_insurance_on_budget",
         "social_insurance_off_budget", "excise", "other_receipts", "total_receipts", "on_budget_receipts",
         "off_budget_receipts"],
    ).drop(columns=["estimate"])
    outlays = _omb_rows(
        "hist08z4", 14,
        ["total_outlays", "discretionary", "defense", "nondefense", "mandatory_and_interest", "mandatory",
         "programmatic", "social_security", "medicare", "medicaid", "other_means_tested", "other_mandatory",
         "offsetting_receipts", "net_interest"],
    ).drop(columns=["estimate"])
    pct = summary.merge(receipts, on="fiscal_year", how="outer").merge(outlays, on="fiscal_year", how="outer")
    _write(pct.sort_values("fiscal_year"), "omb_budget_pct_gdp.csv")

    debt = _omb_rows(
        "hist07z1", 10,
        ["gross_debt_millions", "held_by_government_accounts_millions", "held_by_public_millions",
         "held_by_federal_reserve_millions", "held_by_others_millions", "gross_debt_pct_gdp",
         "held_by_government_accounts_pct_gdp", "held_by_public_pct_gdp", "held_by_federal_reserve_pct_gdp",
         "held_by_others_pct_gdp"],
    )
    _write(debt, "omb_debt.csv")

    tf = pd.read_excel(RAW / "omb/hist13z1_fy2027.xlsx", header=None)
    years = [_year(v) for v in tf.iloc[2]]
    labels = {
        "OASI": (3, 27), "DI": (27, 51), "HI": (51, 81),
    }
    rows = []
    for fund, (start, end) in labels.items():
        for i in range(start, end):
            item = str(tf.iloc[i, 0]).strip()
            if item in ("nan", "") or tf.iloc[i, 1:].isna().all():
                continue
            for j, y in enumerate(years):
                if y is None:
                    continue
                rows.append({"fund": fund, "item": item, "fiscal_year": y, "millions": _num(tf.iloc[i, j])})
    tf_long = pd.DataFrame(rows).dropna(subset=["millions"])
    _write(tf_long, "omb_trust_funds.csv")


# --- Treasury -------------------------------------------------------------------


def treasury() -> None:
    rates = pd.read_csv(RAW / "treasury/avg_interest_rates.csv")
    rates = rates[["record_date", "security_type_desc", "security_desc", "avg_interest_rate_amt"]]
    _write(rates.sort_values(["record_date", "security_type_desc", "security_desc"]), "treasury_avg_interest_rates.csv")
    debt = pd.read_csv(RAW / "treasury/debt_to_penny.csv")
    debt = debt[["record_date", "debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"]]
    _write(debt.sort_values("record_date"), "treasury_debt_to_penny.csv")


STEPS = [
    population, components, census_projection_inputs, cps_profiles, irs_wage_bins,
    fred_annual, cola_history, cms, omb, treasury,
]


def build_all() -> None:
    for step in STEPS:
        print(f"{step.__name__}:")
        step()
