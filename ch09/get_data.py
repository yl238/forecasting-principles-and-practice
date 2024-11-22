import pandas as pd
import rdatasets

google = (
    pd.read_csv("../data/tsibbledata/gafa_stock.csv")
    .query("Symbol == 'GOOG'")
    .assign(Date=lambda df: pd.to_datetime(df["Date"], format="%Y-%m-%d"))
    .set_index("Date")
)

strikes = pd.read_excel("../data/external/strikes-annual-listing.xlsx", skiprows=2)
strikes = strikes.loc[:71, strikes.columns[:2]].astype(int)
strikes.columns = "Year", "Count"

housing = (
    pd.read_csv(
        "../data/external/enigma-us.gov_.census.eits_.ressales-"
        "e7513e56d76050c05caf638306055c98_1.csv"
    )
    .query(
        "dt_code == 'TOTAL' and cat_code == 'SOLD' and geo_desc == 'United States'"
    )
    .rename(columns={"per_name": "date", "val": "Count"})
    .assign(date=lambda df: pd.to_datetime(df.date))
    .set_index("date")[["Count"]]
)

us_econ = (
    pd.read_csv("../data/tsibbledata/global_economy.csv")
    .query("Code == 'USA'")
    .assign(date=lambda df: pd.to_datetime(df.Year, format="%Y"))
    .set_index("date")
)

eggs = pd.read_csv("../data/external/eggs.csv")
eggs.columns = "date", "EggPrice"
eggs = (
    eggs.set_index(pd.to_datetime(eggs.date))
    .join(us_econ.resample("1MS").max().fillna(method="ffill"))
    .dropna()
    .assign(price_per_cpi=lambda df: df["EggPrice"] / df["CPI"])
)

vic_pigs = (
    pd.read_csv("../data/tsibbledata/aus_livestock.csv")
    .assign(date=lambda df: pd.to_datetime(df.Month, format="%Y %b"))
    .query("Animal == 'Pigs' and State == 'Victoria'")
    .groupby("date Animal".split())
    .Count.sum()
    .reset_index()
    .set_index("date")
)

lynx = (
    rdatasets.data("lynx")
    .assign(date=lambda df: pd.to_datetime(df.time, format="%Y"))
    .set_index("date")
)

aus_production = (
    pd.read_csv("../data/tsibbledata/aus_production.csv")
    .assign(Date=lambda df: pd.to_datetime(df.Quarter.str.replace(" ", "")))
    .set_index("Date")
)
recent_production = aus_production.query("Date.dt.year >= 1992")

PBS = pd.read_csv("../data/tsibbledata/PBS.csv", parse_dates=["Month"])
a10 = (
    PBS.query('ATC2=="A10"')
    .groupby("Month", sort=False)
    .agg({"Cost": "sum"})
    .assign(Cost=lambda df: df["Cost"] / 1.0e6)
    .reset_index()
    .set_index("Month", drop=False)
)
