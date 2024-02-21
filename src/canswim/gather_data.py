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
from canswim.hfhub import HFHub
import typing
from fmpsdk.url_methods import __return_json_v3, __validate_period
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def get_latest_date(datetime_col):
    """Return the latest datetime value in a datetime formatted dataframe column"""
    assert len(datetime_col) > 0
    assert is_datetime(datetime_col)
    latest_saved_record = datetime_col.max()
    return latest_saved_record


FMP_API_DEFAULT_LIMIT = -1


def institutional_symbol_ownership(
    apikey: str,
    symbol: str,
    limit: int = FMP_API_DEFAULT_LIMIT,
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


def analyst_estimates(
    apikey: str, symbol: str, period: str = "annual", limit: int = FMP_API_DEFAULT_LIMIT
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


class MarketDataGatherer:

    def __init__(self) -> None:
        load_dotenv(override=True)
        self.FMP_API_KEY = os.getenv("FMP_API_KEY")
        logger.info(f"FMP_API_KEY found: {self.FMP_API_KEY!= None}")
        self.all_stocks_file = "all_stocks.csv"
        self.price_frequency = "1d"  # "1wk"
        self.min_start_date = os.getenv("train_date_start", "1991-01-01")
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")

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
        logger.info(
            "Compiling list of stocks to train on from: {files}", files=stock_files
        )
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
        # logger.info(f"All stocks set: {all_stock_set}")
        # Drop invalid symbols with more than 10 ticker characters.
        # Almost all stocks have less than 5 characters.
        slist = [x for x in set(all_stock_set) if len(x) < 10]
        self.stocks_ticker_set = set(slist)
        # drop known junk symbols from the data feed
        junk_tickers = set(
            [
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
                "GBP",
                "ESMT",
                "MRTX",
                "RAIN",
                "CAD",
                "EUR",
            ]
        )
        self.stocks_ticker_set = self.stocks_ticker_set - junk_tickers
        stocks_df = pd.DataFrame()
        stocks_df["Symbol"] = list(self.stocks_ticker_set)
        stocks_df = stocks_df.set_index(["Symbol"])
        stocks_df.index = stocks_df.index.drop_duplicates()
        stocks_df
        stocks_file = f"data/data-3rd-party/{self.all_stocks_file}"
        stocks_df.to_csv(stocks_file)
        logger.info(f"Saved stock set to {stocks_file}")

    def _gather_yfdata_date_index(self, data_file: str = None, tickers: str = None):
        start_date = self.min_start_date
        old_df = None
        try:
            old_df = pd.read_parquet(data_file)
            logger.info("Loaded saved data. Sample: \n{df}", df=old_df)
            start_date = get_latest_date(old_df.index) - pd.Timedelta(1, "d")
            logger.info(f"Columns: \n{old_df.columns}")
            logger.info(f"Latest saved record is after {start_date}")
        except Exception as e:
            logger.exception(f"Could not load data from file: {data_file}. Error: {e}")
        new_df = yf.download(tickers, start=start_date, group_by="tickers", period="1d")
        new_df = new_df.dropna(how="all")
        logger.info("New data gathered. Sample: \n{bm}", bm=new_df)
        logger.info(f"Columns: \n{new_df.columns}")
        if old_df is not None:
            assert sorted(old_df.columns) == sorted(new_df.columns)
            merged_df = pd.concat([old_df, new_df], axis=0)
            # logger.info(f"bm_df concat\n {merged_df}")
            assert sorted(merged_df.columns) == sorted(old_df.columns)
            assert len(merged_df) == len(old_df) + len(new_df)
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
        else:
            merged_df = new_df
        logger.info("Updated data ready. Sample: \n{bm}", bm=merged_df)
        assert merged_df.index.is_unique
        merged_df.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")
        _bm = pd.read_parquet(data_file)
        assert sorted(_bm.columns) == sorted(merged_df.columns)
        assert len(_bm) == len(merged_df)
        logger.info(f"Sanity check passed for saved data. Loaded OK from {data_file}")

    def gather_broad_market_data(self):
        ## Prepare data for broad market indicies
        # Capture S&P500, NASDAQ100 and Russell 200 indexes and their equal weighted counter parts
        # As well as VIX volatility index, DYX US Dollar index, TNX US 12 Weeks Treasury Yield, 5 Years Treasury Yield and 10 Year Treasuries Yield
        broad_market_indicies = (
            "^SPX ^SPXEW ^NDX ^NDXE ^RUT ^R2ESC ^VIX DX-Y.NYB ^IRX ^FVX ^TNX"
        )
        data_file = "data/data-3rd-party/broad_market.parquet"
        self._gather_yfdata_date_index(
            data_file=data_file, tickers=broad_market_indicies
        )

    def gather_sectors_data(self):
        """Gather historic price and volume data for key market sectors"""
        sector_indicies = "XLE ^SP500-15 ^SP500-20 ^SP500-25 ^SP500-30 ^SP500-35 ^SP500-40 ^SP500-45 ^SP500-50 ^SP500-55 ^SP500-60"
        data_file = "data/data-3rd-party/sectors.parquet"
        self._gather_yfdata_date_index(data_file=data_file, tickers=sector_indicies)

    def gather_subindustries_data(self):
        """
        WARN: YFinance does not provide rich historical price and volume data for Industries and Sub-industries
        the way it does for Sectors.
        Do not use this method with YFinance.
        """
        return  # See warning message above.
        """
        Gather historic price and volume data for S&P 1500 GICS subindustries indexes.
        S&P 1500 includes S&P 400, S&P 500, S&P 600 and overall about 90% of the US stock market capitalization.
        The dataset has 163 GICS sub-industry indexes active as of 2023 plus 7 that were removed in 2023.
        https://www.msci.com/documents/1296102/11185224/GICS+Map+2023.xlsx/82cc6504-9919-29e5-9789-a24fc039d0a5?t=1679087572540
        The goal of these covariates is to provide the model with a more granural breakdown of stock grouping by industry.
        Since stocks usually move together with their group, the model can learn the patterns how an individual stock trend
        relates to its group price and volume action.
        """
        subindustry_indicies = []
        gics_file = f"{self.data_dir}/{self.data_3rd_party}/GICS2023.csv"
        gics_df = pd.read_csv(gics_file)
        logger.info(f"Loaded {len(gics_df)} GICS records. Columns: {gics_df}")
        code_col_name = "Sub-Industry Code"
        subindustry_codes = gics_df[
            pd.to_numeric(gics_df[code_col_name], errors="coerce").notnull()
        ]
        subindustry_codes = subindustry_codes[code_col_name].unique()
        logger.info(
            f"Loaded {len(subindustry_codes)} subindustry codes: {subindustry_codes}"
        )
        sub_prefix = "^sp1500-"
        subindustry_symbols = []
        for c in subindustry_codes:
            subindustry_symbols.append(f"{sub_prefix}{c}")
        logger.info(
            f"Prepared list of {len(subindustry_symbols)} S&P1500 subindustry symbols: {subindustry_symbols}"
        )
        data_file = "data/data-3rd-party/subindustries.parquet"
        self._gather_yfdata_date_index(data_file=data_file, tickers=subindustry_symbols)
        logger.info("Finished gathering subindisty data")

    def gather_stock_price_data(self):
        data_file = (
            f"data/data-3rd-party/all_stocks_price_hist_{self.price_frequency}.parquet"
        )
        start_date = self.min_start_date
        old_df = None
        try:
            old_df = pd.read_parquet(data_file)
            logger.info("Loaded saved data. Sample: \n{df}", df=old_df)
            start_date = get_latest_date(
                old_df.index.get_level_values("Date")
            ) - pd.Timedelta(1, "d")
            logger.info(f"Columns: \n{old_df.columns}")
            logger.info(f"Latest saved record is after {start_date}")
        except Exception as e:
            logger.exception(f"Could not load data from file: {data_file}. Error: {e}")
        new_df = yf.download(
            self.stocks_ticker_set, start=start_date, group_by="tickers"
        )
        new_df = new_df.dropna(how="all")
        new_df = new_df.stack(level=0)
        new_df.index.names = ["Date", "Symbol"]
        new_df.index = new_df.index.swaplevel(0)
        logger.info(f"Stock price data index: {new_df.index.names}")
        logger.info("New data gathered. Sample: \n{df}", df=new_df)
        logger.info(f"Columns: \n{new_df.columns}")
        if old_df is not None:
            assert sorted(old_df.columns) == sorted(new_df.columns)
            merged_df = pd.concat([old_df, new_df], axis=0, join="inner")
            # logger.info(f"merged_df concat\n {merged_df}")
            assert sorted(merged_df.columns) == sorted(old_df.columns)
            assert len(merged_df) == len(old_df) + len(new_df)
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
        else:
            merged_df = new_df
        logger.info("Updated data ready. Sample: \n{df}", df=merged_df)
        # df=merged_df.loc[merged_df.index.get_level_values("Symbol") == 'TEAM'])
        assert merged_df.index.is_unique
        merged_df.to_parquet(data_file)
        logger.info(f"Saved broad market data to {data_file}")
        _bm = pd.read_parquet(data_file)
        assert sorted(_bm.columns) == sorted(merged_df.columns)
        assert len(_bm) == len(merged_df)
        logger.info(
            f"Sanity check passed for broad market data. Loaded OK from {data_file}"
        )

    def gather_earnings_data(self):
        logger.info("Gathering  earnings and sales data...")
        earnings_all_df = pd.DataFrame()
        for ticker in self.stocks_ticker_set:  # ['AAON']: #
            earnings = fmpsdk.historical_earning_calendar(
                apikey=self.FMP_API_KEY, symbol=ticker, limit=-1
            )
            if earnings is not None and len(earnings) > 0:
                edf = pd.DataFrame(earnings)
                edf["date"] = pd.to_datetime(edf["date"])
                edf = edf.set_index(["symbol", "date"])
                # edf = edf.pivot(columns='symbol')
                # edf.swaplevel(i=0,j=1, axis=0)
                # edf.drop(columns=['symbol'])
                earnings_all_df = pd.concat([earnings_all_df, edf])
                # n_earnings = len(earnings)
                # logger.info(f"Total earnings reports for {ticker}: {n_earnings}")
        #    earliest_earn = earnings[-1] if len(earnings > 0 else 'None')
        #    logger.info(f"Earliest earnings report for {ticker}: {earliest_earn}")

        logger.info(f"Earnings sample report for {ticker}: \n{earnings}")
        earnings_all_df.index.names = ["Symbol", "Date"]
        earnings_all_df = earnings_all_df.sort_index()
        logger.info(
            f"Total number of earnings records for all stocks: \n{len(earnings_all_df)}"
        )
        logger.info(f"earnings_all_df: \n{earnings_all_df}")
        logger.info(
            f"len(earnings_all_df.index.levels[0]): \n{len(earnings_all_df.index.levels[0])}"
        )
        earnings_file = "data/data-3rd-party/earnings_calendar.parquet"
        earnings_all_df.to_parquet(earnings_file)
        ### Read back data and verify it
        tmp_earn_df = pd.read_parquet(earnings_file)
        logger.info(f"tmp_earn_df: \n{tmp_earn_df}")
        assert tmp_earn_df.index.names == ["Symbol", "Date"]
        assert len(tmp_earn_df) == len(earnings_all_df)
        assert sorted(tmp_earn_df.columns) == sorted(earnings_all_df.columns)
        logger.info(
            f"Sanity check passed for earnings data. Loaded OK from file: {earnings_file}"
        )

    def gather_stock_key_metrics(self):
        logger.info("Gathering key metrics data with company fundamentals...")
        keymetrics_all_df = pd.DataFrame()
        for ticker in self.stocks_ticker_set:
            kms = fmpsdk.key_metrics(
                apikey=self.FMP_API_KEY, symbol=ticker, period="quarter", limit=-1
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

        logger.info(f"keymetrics_all_df: \n{keymetrics_all_df}")
        logger.info(f"keymetrics_all_df.dtypes: \n{keymetrics_all_df.dtypes}")
        logger.info(f"len(keymetrics_all_df): \n{len(keymetrics_all_df)}")
        # index, row = next(keymetrics_all_df.iterrows())
        # row["averagePayables"]
        # prevent parquet serialization issues
        keymetrics_all_df["averagePayables"] = pd.to_numeric(
            keymetrics_all_df["averagePayables"], dtype_backend="pyarrow"
        )
        keymetrics_all_df.index.names = ["Symbol", "Date"]
        keymetrics_all_df = keymetrics_all_df.sort_index()
        kms_file = "data/data-3rd-party/keymetrics_history.parquet"
        keymetrics_all_df.to_parquet(kms_file, engine="pyarrow")
        ### Read back data and verify it
        temp_kms_df = pd.read_parquet(kms_file)
        logger.info(f"temp_kms_df: \n{temp_kms_df}")
        assert temp_kms_df.index.names == ["Symbol", "Date"]
        assert len(temp_kms_df) == len(keymetrics_all_df)
        assert sorted(temp_kms_df.columns) == sorted(keymetrics_all_df.columns)
        logger.info(
            f"Sanity check passed for key metrics data. Loaded OK from file: {kms_file}"
        )

    def gather_institutional_stock_ownership(self):
        logger.info("Gathering institutional ownership data...")
        inst_ownership_all_df = pd.DataFrame()
        for ticker in self.stocks_ticker_set:
            inst_ownership = institutional_symbol_ownership(
                apikey=self.FMP_API_KEY,
                symbol=ticker,
                limit=-1,
                includeCurrentQuarter=False,
            )
            # logger.info("inst_ownership: ", inst_ownership)
            if inst_ownership is not None and len(inst_ownership) > 0:
                inst_ownership_df = pd.DataFrame(inst_ownership)
                inst_ownership_df["date"] = pd.to_datetime(inst_ownership_df["date"])
                inst_ownership_df = inst_ownership_df.set_index(["symbol", "date"])
                # logger.info(f"Institutional ownership for {ticker} # columns: \n{len(inst_ownership_df.columns)}")
                n_iown = len(inst_ownership_df)
                logger.info(
                    f"Total institutional ownership reports for {ticker}: {n_iown}"
                )
                inst_ownership_all_df = pd.concat(
                    [inst_ownership_all_df, inst_ownership_df]
                )
                # logger.info(f"Institutional ownership concatenated {ticker} # columns: \n{inst_ownership_all_df.columns}")
            else:
                logger.info(
                    f"No {ticker} institutional ownership reports: inst_ownership={inst_ownership}"
                )
        inst_own_df = inst_ownership_all_df.copy()
        # prevent parquet serialization issues
        inst_own_df["totalInvestedChange"] = pd.to_numeric(
            inst_own_df["totalInvestedChange"],
            dtype_backend="pyarrow",
            downcast="integer",
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
        # type(inst_own_df["totalInvestedChange"][0])
        # clean up bad data from the third party source feed
        inst_own_df["totalInvestedChange"] = inst_own_df["totalInvestedChange"].astype(
            "float64"
        )
        inst_own_df["totalInvestedChange"] = inst_own_df["totalInvestedChange"].astype(
            "int64"
        )
        inst_own_df["cik"] = inst_own_df["cik"].replace("", -1)
        inst_own_df["cik"] = inst_own_df["cik"].astype("int64")
        inst_own_df["totalPutsChange"] = inst_own_df["totalPutsChange"].astype(
            "float64"
        )
        inst_own_df["totalPutsChange"] = inst_own_df["totalPutsChange"].astype("int64")
        # inst_own_df["totalPutsChange"] = pd.to_numeric(inst_own_df["totalPutsChange"], dtype_backend="pyarrow", downcast = 'integer')
        inst_own_df["totalCallsChange"] = inst_own_df["totalCallsChange"].astype(
            "float64"
        )
        inst_own_df["totalCallsChange"] = inst_own_df["totalCallsChange"].astype(
            "int64"
        )
        # inst_own_df.dtypes
        # inst_own_df
        inst_own_df.index.names = ["Symbol", "Date"]
        inst_own_df = inst_own_df.sort_index()
        # inst_own_df.index.names
        inst_ownership_file = (
            "data/data-3rd-party/institutional_symbol_ownership.parquet"
        )
        inst_own_df.to_parquet(inst_ownership_file, engine="pyarrow")

        ### Read back data and verify it
        tmp_int_own_df = pd.read_parquet(inst_ownership_file)
        logger.info(f"tmp_int_own_df: \n{tmp_int_own_df}")
        assert tmp_int_own_df.index.names == ["Symbol", "Date"]
        assert len(tmp_int_own_df) == len(inst_own_df)
        assert sorted(tmp_int_own_df.columns) == sorted(inst_own_df.columns)
        logger.info(
            f"Sanity check passed for institutional ownership data. Loaded OK from file: {inst_ownership_file}"
        )

    def gather_analyst_estimates(self):

        def _fetch_estimates(period=None):
            assert period in ["quarter", "annual"]
            estimates_all_df = pd.DataFrame()
            for ticker in self.stocks_ticker_set:  # ['ALTR']:
                est = analyst_estimates(
                    apikey=self.FMP_API_KEY, symbol=ticker, period=period, limit=-1
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
                    logger.info(
                        f"{n_est} total {ticker} {period} analyst estimates reports"
                    )
                else:
                    logger.info(
                        f"No {ticker} {period} analyst estimates reports: est={est}"
                    )

            return estimates_all_df

        logger.info("Gathering analyst estimates...")
        est_file_name_template = "data/data-3rd-party/analyst_estimates_{p}.parquet"
        for p in ["annual", "quarter"]:
            est_file_name = est_file_name_template.format(p=p)
            estimates_all_df = _fetch_estimates(p)
            estimates_all_df.index.names = ["Symbol", "Date"]
            estimates_all_df = estimates_all_df.sort_index()
            # est_file_name= f'data/analyst_estimates_{p}.csv.bz2'
            # estimates_all_df.to_csv(est_file_name)
            estimates_all_df.to_parquet(est_file_name)
            logger.info(f"Saved analyst estimates to: {est_file_name}")
            logger.info(f"all {p} estimates count: {len(estimates_all_df)}")
            # logger.info(f"{p} estimates sample:\n{estimates_all_df}")
            tmp_est_df = pd.read_parquet(est_file_name)
            assert tmp_est_df.index.names == ["Symbol", "Date"]
            assert len(tmp_est_df) == len(estimates_all_df)
            # logger.info(f"tmp_est_df.columns: {tmp_est_df.columns}")
            # logger.info(f"estimates_all_df.columns: {estimates_all_df.columns}")
            assert sorted(tmp_est_df.columns) == sorted(estimates_all_df.columns)
            logger.info(
                f"Sanity check passed for analyst estimates data. Loaded OK from file: {est_file_name}"
            )


# main function
def main():
    hfhub = HFHub()
    hfhub.download_data()
    g = MarketDataGatherer()
    g.gather_broad_market_data()
    g.gather_sectors_data()
    g.gather_stock_tickers()
    g.gather_stock_price_data()
    g.gather_earnings_data()
    g.gather_stock_key_metrics()
    g.gather_institutional_stock_ownership()
    g.gather_analyst_estimates()
    hfhub.upload_data()


if __name__ == "__main__":
    main()
