import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv(override=True)

FMP_API_KEY = os.getenv("FMP_API_KEY")

print(f"FMP_API_KEY={FMP_API_KEY!= None}")

all_stocks_file = "all_stocks.csv"


# Prepare list of stocks for training

all_stock_set = set()
stock_files = [
    "IBD50.csv",
    "IBD250.csv",
    "ibdlive_picks.csv",
    "russell2000_iwm_holdings.csv",
    "sp500_ivv_holdings.csv",
    "nasdaq100_cndx_holdings.csv",
    all_stocks_file,
]
for f in stock_files:
    fp = f"data/data-3rd-party/{f}"
    if Path(fp).is_file():
        stocks = pd.read_csv(fp)
        print(f"loaded {len(stocks)} symbols from {fp}")
        stock_set = set(stocks["Symbol"])
        print(f"{len(stock_set)} symbols in stock set")
        all_stock_set |= stock_set
        print(f"total symbols loaded: {len(all_stock_set)}")
    else:
        print(f"{fp} not found.")


len(all_stock_set), all_stock_set

stocks_ticker_set = all_stock_set

growth_stocks_df = pd.DataFrame()
growth_stocks_df["Symbol"] = list(stocks_ticker_set)
growth_stocks_df = growth_stocks_df.set_index(["Symbol"])
growth_stocks_df.index = growth_stocks_df.index.drop_duplicates()
# drop known junk symbols from the data feed
junk = [
    "MSFUT",
    "GEFB",
    "METCV",
    "SGAFT",
    "NQH4",
    "XTSLA",
    "-",
    "PDLI",
    "ADRO",
    "ICSUAGD",
    "BFB",
    "GTXI",
    "P5N994",
    "LGFB",
    "MLIFT",
    "ESH4",
    "LGFA",
    "MOGA",
    "PBRA",
    "BRKB",
    "RTYH4",
    "\xa0",
    "CRDA",
]
growth_stocks_df.index = growth_stocks_df.index.drop(junk)
growth_stocks_df

growth_stocks_df.to_csv(f"data/data-3rd-party/{all_stocks_file}")


## Prepare data for broad market indicies

# Capture S&P500, NASDAQ100 and Russell 200 indecies and their equal weighted counter parts
# As well as VIX volatility index, DYX US Dollar index, TNX US 12 Weeks Treasury Yield, 5 Years Treasury Yield and 10 Year Treasuries Yield
broad_market_indicies = (
    "^SPX ^SPXEW ^NDX ^NDXE ^RUT ^R2ESC ^VIX DX-Y.NYB ^IRX ^FVX ^TNX"
)

broad_market = yf.download(broad_market_indicies, period="max", group_by="tickers")
broad_market

bm_file = "data/data-3rd-party/broad_market.parquet"
broad_market.to_parquet(bm_file)

bm = pd.read_parquet(bm_file)

bm

sector_indicies = "XLE ^SP500-15 ^SP500-20 ^SP500-25 ^SP500-30 ^SP500-35 ^SP500-40 ^SP500-45 ^SP500-50 ^SP500-55 ^SP500-60"

sectors = yf.download(sector_indicies, period="max")
sectors

sectors_file = "data/data-3rd-party/sectors.parquet"
sectors.to_parquet(sectors_file)

tmp_s = pd.read_parquet(sectors_file)
tmp_s

tmp_s = pd.read_parquet(sectors_file)
tmp_s

stock_price_data = yf.download(
    all_stock_set, period="max", group_by="tickers", interval=price_interval
)
stock_price_data

stock_price_data.columns.levels

stock_price_data.tail(20)

len(stock_price_data)

stock_price_data = stock_price_data.dropna(how="all")

len(stock_price_data)

# price_hist_file = f'data/all_stocks_price_hist_{price_interval}.csv.bz2'
price_hist_file = f"data/data-3rd-party/all_stocks_price_hist_{price_interval}.parquet"

df = stock_price_data_loaded.copy()

df2 = df.stack(level=0)
df2

df2.index

df2.index.names = ["Date", "Symbol"]

df2 = df2.index.swaplevel(0)

df2 = df2.sort_index()
df2

# stock_price_data.to_csv(price_hist_file, index='Date')
df2.to_parquet(price_hist_file)

# stock_price_data_loaded = pd.read_csv(price_hist_file, header=[0, 1], index_col=0)
stock_price_data_loaded = pd.read_parquet(
    price_hist_file
)  # , filters=[("Symbol", "in", ['AEP', 'AAPL'])])
stock_price_data_loaded

stock_price_data_loaded.dropna()

list(stock_price_data_loaded.index.levels[0])

## Prepare earnings and sales data

import fmpsdk

# Company Valuation Methods
symbol: str = "AAPL"
symbols: ["AAPL", "CSCO", "QQQQ"]
exchange: str = "NYSE"
exchanges: ["NYSE", "NASDAQ"]
query: str = "AA"
limit: int = 3
period: str = "quarter"
download: bool = True
market_cap_more_than: int = 1000000000
beta_more_than: int = 1
volume_more_than: int = 10000
sector: str = "Technology"
dividend_more_than: int = 0
industry: str = "Software"
filing_type: str = "10-K"
print(f"Company Profile: {fmpsdk.company_profile(apikey=FMP_API_KEY, symbol=symbol)=}")


earnings_all_df = pd.DataFrame()
for ticker in stocks_ticker_set:  # ['AAON']: #
    earnings = fmpsdk.historical_earning_calendar(
        apikey=FMP_API_KEY, symbol=ticker, limit=-1
    )
    if earnings is not None and len(earnings) > 0:
        edf = pd.DataFrame(earnings)
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf.set_index(["symbol", "date"])
        # edf = edf.pivot(columns='symbol')
        # edf.swaplevel(i=0,j=1, axis=0)
        # edf.drop(columns=['symbol'])
        earnings_all_df = pd.concat([earnings_all_df, edf])
        n_earnings = len(earnings)
        # print(f"Total earnings reports for {ticker}: {n_earnings}")
#    earliest_earn = earnings[-1] if len(earnings > 0 else 'None')
#    print(f"Earliest earnings report for {ticker}: {earliest_earn}")


earnings

aaon = earnings_all_df.loc[["AAON"]]

len(earnings_all_df)

earnings_all_df


len(earnings_all_df.index.levels[0])

# earnings_file = 'data/earnings_calendar.csv.bz2'
earnings_file = "data/data-3rd-party/earnings_calendar.parquet"

# earnings_all_df.to_csv(earnings_file)
earnings_all_df.to_parquet(earnings_file)

### Read back data and verify it

import pandas as pd

tmp_earn_df = pd.read_parquet(earnings_file)
tmp_earn_df

# earnings_loaded_df = pd.read_csv('data/earnings_calendar.csv.bz2', index_col=['symbol', 'date'])
# print(earnings_loaded_df)

tmp_earn_df.index.names = ["Symbol", "Date"]

## Prepare historical dividends
#  * This is secondary information since growth stocks usually do not have dividends and rarely have splits
#  * Additionally the dividends and split information is partially reflected in Adj Close of price history data


def fetch_dividends_history():
    divs_hist_all_df = pd.DataFrame()
    for ticker in stocks_ticker_set:  # ['AAON']:
        divs_hist = fmpsdk.historical_stock_dividend(apikey=FMP_API_KEY, symbol=ticker)
        # print(f"Loaded historical dividends for {ticker}: \n{divs_hist}")
        print(
            f"Loaded {len(divs_hist['historical'])} historical dividends for {ticker}"
        )
        if divs_hist["historical"] is not None and len(divs_hist["historical"]) > 0:
            dh_df_tmp = pd.DataFrame.from_dict(data=divs_hist["historical"])
            # print(f"Historical dividends for {ticker} dataframe: \n{dh_df_tmp.head()}")
            dh_df_tmp["symbol"] = ticker
            dh_df = dh_df_tmp
            # print(f"Historical dividends for {ticker} dataframe: \n{dh_df_tmp.head()}")
            # print(f"Historical dividends for {ticker} full dataframe: \n{dh_df.head()}")
            dh_df["date"] = pd.to_datetime(dh_df["date"])
            dh_df = dh_df.set_index(["symbol", "date"])
            n_divs_hist = len(dh_df)
            print(f"Total dividends history reports for {ticker}: {n_divs_hist}")
            # print(f"Historical dividends for {ticker} full dataframe: \n{dh_df}")
            divs_hist_all_df = pd.concat([divs_hist_all_df, dh_df])
    return divs_hist_all_df


## Prepare key metrics data for company fundamentals


keymetrics_all_df = pd.DataFrame()
for ticker in stocks_ticker_set:
    kms = fmpsdk.key_metrics(
        apikey=FMP_API_KEY, symbol=ticker, period="quarter", limit=-1
    )
    if kms is not None and len(kms) > 0:
        kms_df = pd.DataFrame(kms)
        kms_df["date"] = pd.to_datetime(kms_df["date"])
        kms_df = kms_df.set_index(["symbol", "date"])
        # print(f"Key metrics for {ticker} sample: \n{kms_df.columns}")
        keymetrics_all_df = pd.concat([keymetrics_all_df, kms_df])
        # print(f"Key metrics concatenated {ticker}: \n{keymetrics_all_df.columns}")
        n_kms = len(kms_df)
        print(f"Total key metrics reports for {ticker}: {n_kms}")
    else:
        print(f"No {ticker} key metrics reports: kms={kms}")


keymetrics_all_df

keymetrics_all_df.dtypes

len(keymetrics_all_df)

index, row = next(keymetrics_all_df.iterrows())

row["averagePayables"]

# prevent parquet serialization issues
keymetrics_all_df["averagePayables"] = pd.to_numeric(
    keymetrics_all_df["averagePayables"], dtype_backend="pyarrow"
)

keymetrics_all_df.index.names = ["Symbol", "Date"]
keymetrics_all_df.index.names

keymetrics_all_df = keymetrics_all_df.sort_index()
keymetrics_all_df

# kms_file = 'data/keymetrics_history.csv.bz2'
# keymetrics_all_df.to_csv(kms_file)

kms_file = "data/data-3rd-party/keymetrics_history.parquet"


keymetrics_all_df.to_parquet(kms_file, engine="pyarrow")

temp_kms_df = pd.read_parquet(kms_file)
temp_kms_df

## Prepare institutional ownership data


from fmpsdk.settings import DEFAULT_LIMIT, SEC_RSS_FEEDS_FILENAME, BASE_URL_v3
from fmpsdk.url_methods import __return_json_v4
import typing


def institutional_symbol_ownership(
    apikey: str,
    symbol: str,
    limit: int,
    includeCurrentQuarter: bool = False,
) -> typing.Optional[typing.List[typing.Dict]]:
    """
    Query FMP /institutional-ownership/ API.

    :param apikey: Your API key.
    :param symbol: Company ticker.
    :param limit: up to how many quarterly reports to return.
    :param includeCurrentQuarter: Whether to include any available data in the current quarter.
    :return: A list of dictionaries.
    """
    path = f"institutional-ownership/symbol-ownership"
    query_vars = {
        "symbol": symbol,
        "apikey": apikey,
        "includeCurrentQuarter": includeCurrentQuarter,
        "limit": limit,
    }
    return __return_json_v4(path=path, query_vars=query_vars)


inst_ownership_all_df = pd.DataFrame()
for ticker in stocks_ticker_set:
    inst_ownership = institutional_symbol_ownership(
        apikey=FMP_API_KEY, symbol=ticker, limit=-1, includeCurrentQuarter=False
    )
    # print("inst_ownership: ", inst_ownership)
    if inst_ownership is not None and len(inst_ownership) > 0:
        inst_ownership_df = pd.DataFrame(inst_ownership)
        inst_ownership_df["date"] = pd.to_datetime(inst_ownership_df["date"])
        inst_ownership_df = inst_ownership_df.set_index(["symbol", "date"])
        # print(f"Institutional ownership for {ticker} # columns: \n{len(inst_ownership_df.columns)}")
        n_iown = len(inst_ownership_df)
        print(f"Total institutional ownership reports for {ticker}: {n_iown}")
        inst_ownership_all_df = pd.concat([inst_ownership_all_df, inst_ownership_df])
        # print(f"Institutional ownership concatenated {ticker} # columns: \n{inst_ownership_all_df.columns}")
    else:
        print(
            f"No {ticker} institutional ownership reports: inst_ownership={inst_ownership}"
        )


inst_own_df = inst_ownership_all_df.copy()

inst_own_df

# prevent parquet serialization issues
inst_own_df["totalInvestedChange"] = pd.to_numeric(
    inst_own_df["totalInvestedChange"], dtype_backend="pyarrow", downcast="integer"
)


def find_bad_cell():
    for index, row in inst_own_df.iterrows():
        try:
            x = row["totalPutsChange"]
            assert isinstance(x, int)
        except Exception as e:
            print(
                f"Unable to convert to numeric type value:({x}), type({type(x)}), index({index}, \nerror:{e}\nrow:{row})"
            )
            break


find_bad_cell()

type(inst_own_df["totalInvestedChange"][0])

# clean up bad data from the third party source feed
inst_own_df["totalInvestedChange"] = inst_own_df["totalInvestedChange"].astype(
    "float64"
)
inst_own_df["totalInvestedChange"] = inst_own_df["totalInvestedChange"].astype("int64")


inst_own_df["cik"] = inst_own_df["cik"].replace("", -1)
inst_own_df["cik"] = inst_own_df["cik"].astype("int64")


inst_own_df["totalPutsChange"] = inst_own_df["totalPutsChange"].astype("float64")
inst_own_df["totalPutsChange"] = inst_own_df["totalPutsChange"].astype("int64")
# inst_own_df["totalPutsChange"] = pd.to_numeric(inst_own_df["totalPutsChange"], dtype_backend="pyarrow", downcast = 'integer')

inst_own_df["totalCallsChange"] = inst_own_df["totalCallsChange"].astype("float64")
inst_own_df["totalCallsChange"] = inst_own_df["totalCallsChange"].astype("int64")


inst_own_df.dtypes

inst_own_df = inst_own_df.sort_index()
inst_own_df

inst_own_df.index.names = ["Symbol", "Date"]
inst_own_df.index.names

inst_ownership_file = "data/data-3rd-party/institutional_symbol_ownership.parquet"


inst_own_df.to_parquet(inst_ownership_file, engine="pyarrow")

tmp_int_own_df = pd.read_parquet(inst_ownership_file)
tmp_int_own_df

## Prepare forward looking analyst estimates to be used as future covariates

DEFAULT_LIMIT = -1
import typing
from fmpsdk.url_methods import __return_json_v3, __validate_period


def analyst_estimates(
    apikey: str, symbol: str, period: str = "annual", limit: int = DEFAULT_LIMIT
) -> typing.Optional[typing.List[typing.Dict]]:
    """
    Query FMP /analyst-estimates/ API.

    :param apikey: Your API key.
    :param symbol: Company ticker.
    :param period: 'annual' or 'quarter'
    :param limit: Number of rows to return.
    :return: A list of dictionaries.
    """
    path = f"/analyst-estimates/{symbol}"
    query_vars = {
        "apikey": apikey,
        "symbol": symbol,
        "period": __validate_period(value=period),
        "limit": limit,
    }
    return __return_json_v3(path=path, query_vars=query_vars)


def fetch_estimates(period=None):
    assert period in ["quarter", "annual"]
    estimates_all_df = pd.DataFrame()
    for ticker in stocks_ticker_set:  # ['ALTR']:
        est = analyst_estimates(
            apikey=FMP_API_KEY, symbol=ticker, period=period, limit=-1
        )
        # print('est:', est)
        if est is not None and len(est) > 0:
            est_df = pd.DataFrame(est)
            est_df["date"] = pd.to_datetime(est_df["date"])
            est_df = est_df.set_index(["symbol", "date"])
            # print(f"Analyst estimates for {ticker} sample: \n{est_df.columns}")
            estimates_all_df = pd.concat([estimates_all_df, est_df])
            # print(f"Key metrics concatenated {ticker}: \n{estimates_all_df.columns}")
            n_est = len(est_df)
            print(f"{n_est} total {ticker} {period} analyst estimates reports")
        else:
            print(f"No {ticker} {period} analyst estimates reports: est={est}")

    return estimates_all_df


for p in ["annual", "quarter"]:
    est_file_name = f"data/data-3rd-party/analyst_estimates_{p}.parquet"
    estimates_all_df = fetch_estimates(p)
    estimates_all_df = estimates_all_df.sort_index()
    estimates_all_df.index.names = ["Symbol", "Date"]
    # est_file_name= f'data/analyst_estimates_{p}.csv.bz2'
    # estimates_all_df.to_csv(est_file_name)
    estimates_all_df.to_parquet(est_file_name)
    print(f"all {p} estimates count:", len(estimates_all_df.index))


tmp_est_dict = {}
for p in ["annual", "quarter"]:
    est_file_name = f"data/data-3rd-party/analyst_estimates_{p}.parquet"
    tmp_est_dict[p] = pd.read_parquet(est_file_name)
    print(f"all {p} estimates count:", len(tmp_est_dict[p]))


estimates_all_df = tmp_est_dict["quarter"]


estimates_all_df

## Upload all gathered data from 3rd party sources to hf hub

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, upload_folder, create_repo
from canswim.hfhub import HFHub

# prefix for HF Hub dataset repo
repo_id = "ivelin/canswim"
private = True

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")

print(f"HF_TOKEN={HF_TOKEN!= None}")

# Create repo if not existing yet
repo_info = create_repo(
    repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=HF_TOKEN
)
print(f"repo_info: ", repo_info)
data_path = Path("data")
upload_folder(
    repo_id=repo_id,
    # path_in_repo="data-3rd-party",
    repo_type="dataset",
    folder_path=data_path,
    token=HF_TOKEN,
)
