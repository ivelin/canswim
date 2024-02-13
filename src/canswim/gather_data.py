"""
Gather stock market data from 3rd party sources
"""

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import os
from pathlib import Path
import fmpsdk
from fmpsdk.settings import DEFAULT_LIMIT, SEC_RSS_FEEDS_FILENAME, BASE_URL_v3
from fmpsdk.url_methods import __return_json_v4
import typing
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, upload_folder, create_repo
from canswim.hfhub import HFHub
import typing
from fmpsdk.url_methods import __return_json_v3, __validate_period

class MarketDataGatherer:

    def __init__(self) -> None:
        load_dotenv(override=True)
        self.FMP_API_KEY = os.getenv("FMP_API_KEY")
        logger.info(f"FMP_API_KEY found: {self.FMP_API_KEY!= None}")
        self.all_stocks_file = "all_stocks.csv"
        self.price_frequency = "1d" # "1wk"

    def gather_stock_tickers(self):
        # Prepare list of stocks for training
        all_stock_set = set()
        stock_files = [
            "IBD50.csv",
            "IBD250.csv",
            "ibdlive_picks.csv",
            "russell2000_iwm_holdings.csv",
            "sp500_ivv_holdings.csv",
            "nasdaq100_cndx_holdings.csv",
            self.all_stocks_file,
        ]
        logger.info("Compiling list of stocks to train on from: {files}", files=stock_files)
        for f in stock_files:
            fp = f"data/data-3rd-party/{f}"
            if Path(fp).is_file():
                stocks = pd.read_csv(fp)
                logger.info(f"loaded {len(stocks)} symbols from {fp}")
                stock_set = set(stocks["Symbol"])
                logger.info(f"{len(stock_set)} symbols in stock set")
                all_stock_set |= stock_set
                logger.info(f"total symbols loaded: {len(all_stock_set)}")
            else:
                logger.info(f"{fp} not found.")

        logger.info(f"Total size of stock set: {len(all_stock_set)}")
        logger.info(f"All stocks set: {all_stock_set}")
        # Drop invalid symbols with more than 10 ticker characters.
        # Almost all stocks have less than 5 characters.
        self.stocks_ticker_set = [x for x in set(all_stock_set) if len(x) < 10]
        # drop known junk symbols from the data feed
        junk_tickers = set(
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
        )
        self.stocks_ticker_set = self.stocks_ticker_set-junk_tickers
        stocks_df = pd.DataFrame()
        stocks_df["Symbol"] = list(self.stocks_ticker_set)
        stocks_df = stocks_df.set_index(["Symbol"])
        stocks_df.index = stocks_df.index.drop_duplicates()
        stocks_df
        stocks_file = f"data/data-3rd-party/{self.all_stocks_file}"
        stocks_df.to_csv(stocks_file)
        logger.info(f"Saved stock set to {stocks_file}")

    def gather_broad_market_data(self):
        ## Prepare data for broad market indicies

        # Capture S&P500, NASDAQ100 and Russell 200 indecies and their equal weighted counter parts
        # As well as VIX volatility index, DYX US Dollar index, TNX US 12 Weeks Treasury Yield, 5 Years Treasury Yield and 10 Year Treasuries Yield
        broad_market_indicies = (
            "^SPX ^SPXEW ^NDX ^NDXE ^RUT ^R2ESC ^VIX DX-Y.NYB ^IRX ^FVX ^TNX"
        )
        broad_market = yf.download(broad_market_indicies, period="max", group_by="tickers")
        logger.info("Broad market data gathered. Sample: {bm}", bm=broad_market)
        bm_file = "data/data-3rd-party/broad_market.parquet"
        broad_market.to_parquet(bm_file)
        logger.info(f"Saved broad market data to {bm_file}")
        bm = pd.read_parquet(bm_file)
        logger.info(f"Sanity check passed for broad market data. Loaded OK from {bm_file}")

        sector_indicies = "XLE ^SP500-15 ^SP500-20 ^SP500-25 ^SP500-30 ^SP500-35 ^SP500-40 ^SP500-45 ^SP500-50 ^SP500-55 ^SP500-60"
        sectors = yf.download(sector_indicies, period="max")
        logger.info(f"Sector indicies data gathered. Sample: {sectors}")
        sectors_file = "data/data-3rd-party/sectors.parquet"
        sectors.to_parquet(sectors_file)
        logger.info(f"Saved sectors data to {sectors_file}")
        tmp_s = pd.read_parquet(sectors_file)
        logger.info(f"Sanity check passed for sector data. Loaded OK from {sectors_file}")

    def gather_stock_price_data(self):
        stock_price_data = yf.download(
            self.all_stock_set, period="max", group_by="tickers", interval=self.price_frequency
        )
        logger.info(f"Stock price data downloaded. Sample:\n {stock_price_data}")
        logger.info(f"stock_price_data.columns.levels: {stock_price_data.columns.levels}")
        logger.info(f"len(stock_price_data): {len(stock_price_data)}")
        # drop rows where all values are NaN
        stock_price_data = stock_price_data.dropna(how="all")
        len(stock_price_data)
        logger.info(f"len(stock_price_data) after dropna: {len(stock_price_data)}")
        price_hist_file = f"data/data-3rd-party/all_stocks_price_hist_{self.price_frequency}.parquet"
        stock_price_data.to_parquet(price_hist_file)
        logger.info(f"Stock price data saved to {price_hist_file}")
        stock_price_data_loaded = pd.read_parquet(
            price_hist_file
        )  # , filters=[("Symbol", "in", ['AEP', 'AAPL'])])
        logger.info(f"Sanity check passed for price history data. Loaded OK from {price_hist_file}")
        stock_price_data_loaded.dropna()
        logger.info(f"list(stock_price_data_loaded.index.levels[0]): {list(stock_price_data_loaded.index.levels[0])}")


    def gather_earnings_data(self):
        ## Prepare earnings and sales data
        earnings_all_df = pd.DataFrame()
        for ticker in self.stocks_ticker_set:  # ['AAON']: #
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
                # logger.info(f"Total earnings reports for {ticker}: {n_earnings}")
        #    earliest_earn = earnings[-1] if len(earnings > 0 else 'None')
        #    logger.info(f"Earliest earnings report for {ticker}: {earliest_earn}")


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
        # logger.info(earnings_loaded_df)

        tmp_earn_df.index.names = ["Symbol", "Date"]

        ## Prepare historical dividends
        #  * This is secondary information since growth stocks usually do not have dividends and rarely have splits
        #  * Additionally the dividends and split information is partially reflected in Adj Close of price history data


    def gather_stock_key_metrics(self):
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
                # logger.info(f"Key metrics for {ticker} sample: \n{kms_df.columns}")
                keymetrics_all_df = pd.concat([keymetrics_all_df, kms_df])
                # logger.info(f"Key metrics concatenated {ticker}: \n{keymetrics_all_df.columns}")
                n_kms = len(kms_df)
                logger.info(f"Total key metrics reports for {ticker}: {n_kms}")
            else:
                logger.info(f"No {ticker} key metrics reports: kms={kms}")


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


    def gather_institutional_stock_ownership(self):
        ## Prepare institutional ownership data
        inst_ownership_all_df = pd.DataFrame()
        for ticker in stocks_ticker_set:
            inst_ownership = institutional_symbol_ownership(
                apikey=FMP_API_KEY, symbol=ticker, limit=-1, includeCurrentQuarter=False
            )
            # logger.info("inst_ownership: ", inst_ownership)
            if inst_ownership is not None and len(inst_ownership) > 0:
                inst_ownership_df = pd.DataFrame(inst_ownership)
                inst_ownership_df["date"] = pd.to_datetime(inst_ownership_df["date"])
                inst_ownership_df = inst_ownership_df.set_index(["symbol", "date"])
                # logger.info(f"Institutional ownership for {ticker} # columns: \n{len(inst_ownership_df.columns)}")
                n_iown = len(inst_ownership_df)
                logger.info(f"Total institutional ownership reports for {ticker}: {n_iown}")
                inst_ownership_all_df = pd.concat([inst_ownership_all_df, inst_ownership_df])
                # logger.info(f"Institutional ownership concatenated {ticker} # columns: \n{inst_ownership_all_df.columns}")
            else:
                logger.info(
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
                    logger.info(
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



    DEFAULT_LIMIT = -1


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
            # logger.info('est:', est)
            if est is not None and len(est) > 0:
                est_df = pd.DataFrame(est)
                est_df["date"] = pd.to_datetime(est_df["date"])
                est_df = est_df.set_index(["symbol", "date"])
                # logger.info(f"Analyst estimates for {ticker} sample: \n{est_df.columns}")
                estimates_all_df = pd.concat([estimates_all_df, est_df])
                # logger.info(f"Key metrics concatenated {ticker}: \n{estimates_all_df.columns}")
                n_est = len(est_df)
                logger.info(f"{n_est} total {ticker} {period} analyst estimates reports")
            else:
                logger.info(f"No {ticker} {period} analyst estimates reports: est={est}")

        return estimates_all_df


    def gather_analyst_estimates(self):
    ## Prepare forward looking analyst estimates to be used as future covariates

        for p in ["annual", "quarter"]:
            est_file_name = f"data/data-3rd-party/analyst_estimates_{p}.parquet"
            estimates_all_df = fetch_estimates(p)
            estimates_all_df = estimates_all_df.sort_index()
            estimates_all_df.index.names = ["Symbol", "Date"]
            # est_file_name= f'data/analyst_estimates_{p}.csv.bz2'
            # estimates_all_df.to_csv(est_file_name)
            estimates_all_df.to_parquet(est_file_name)
            logger.info(f"all {p} estimates count:", len(estimates_all_df.index))


        tmp_est_dict = {}
        for p in ["annual", "quarter"]:
            est_file_name = f"data/data-3rd-party/analyst_estimates_{p}.parquet"
            tmp_est_dict[p] = pd.read_parquet(est_file_name)
            logger.info(f"all {p} estimates count:", len(tmp_est_dict[p]))


        estimates_all_df = tmp_est_dict["quarter"]


        estimates_all_df


    def upload_data_to_hfhub(self):
        ## Upload all gathered data from 3rd party sources to hf hub
        # prefix for HF Hub dataset repo
        repo_id = "ivelin/canswim"
        private = True

        load_dotenv(override=True)

        HF_TOKEN = os.getenv("HF_TOKEN")

        logger.info(f"HF_TOKEN={HF_TOKEN!= None}")

        # Create repo if not existing yet
        repo_info = create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=HF_TOKEN
        )
        logger.info(f"repo_info: ", repo_info)
        data_path = Path("data")
        upload_folder(
            repo_id=repo_id,
            # path_in_repo="data-3rd-party",
            repo_type="dataset",
            folder_path=data_path,
            token=HF_TOKEN,
        )
