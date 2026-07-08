"""
Gather stock market data from 3rd party sources
"""

import yfinance as yf
from requests import Session
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import os
from pathlib import Path
import fmpsdk
from fmpsdk.url_methods import __return_json_v4
import typing
from canswim.hfhub import HFHub, _env_bool
from canswim.paths import data_3rd_party_dir, resolve_symbol_csv, symbol_lists_dir
from canswim.eligibility import is_valid_ticker_symbol
from canswim.market_data_io import (
    coerce_ohlcv_numeric,
    normalize_earnings_dataframe,
    normalize_key_metrics_dataframe,
    yfinance_multi_to_symbol_date,
)
from fmpsdk.url_methods import __return_json_v3, __validate_period
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class RateLimitedSession(LimiterMixin, Session):
    """Rate-limited session without multi-GB SQLite cache (default)."""

    pass

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
        logger.info(f"FMP_API_KEY found: {self.FMP_API_KEY != None}")
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        self.all_stocks_file = "all_stocks.csv"
        self.price_frequency = "1d"  # "1wk"
        self.min_start_date = os.getenv("train_date_start", "1991-01-01")
        # Optional scope: gather only tickers from this CSV (under symbol_lists/)
        self.stock_tickers_list = os.getenv("stock_tickers_list")
        # Never use multi-GB SQLite yfinance cache unless explicitly enabled
        self.use_yfinance_cache = _env_bool("YFINANCE_USE_CACHE", default=False)

    def _yfinance_session(self):
        """Rate-limited session. Avoids SQLite cache hang by default."""
        if self.use_yfinance_cache:
            try:
                from requests_cache import CacheMixin, SQLiteCache

                class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
                    pass

                logger.warning(
                    "YFINANCE_USE_CACHE=1: using SQLite cache (can be very large/slow)"
                )
                return CachedLimiterSession(
                    limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
                    bucket_class=MemoryQueueBucket,
                    backend=SQLiteCache("yfinance.cache"),
                )
            except Exception as e:
                logger.warning(f"Could not enable yfinance cache ({e}); uncached session")
        return RateLimitedSession(
            limiter=Limiter(RequestRate(5, Duration.SECOND * 1)),
            bucket_class=MemoryQueueBucket,
        )

    def _data_file(self, name: str) -> str:
        d = data_3rd_party_dir()
        d.mkdir(parents=True, exist_ok=True)
        return str(d / name)

    def gather_stock_tickers(self):
        # Prepare list of stocks for training from checked-in symbol_lists/
        all_stock_set = set()
        # Source lists only (do not feed derived all_stocks.csv back into itself)
        stock_files = [
            "IBD50.csv",
            "IBD250.csv",
            "ibdlive_picks.csv",
            "russell2000_iwm_holdings.csv",
            "sp500_ivv_holdings.csv",
            "nasdaq100_cndx_holdings.csv",
            "watchlist.csv",
            "vti_total_market_stocks.csv",
            "ITB_holdings.csv",
            "IYM_holdings.csv",
            "test_stocks.csv",
            "few_stocks.csv",
        ]
        # Optional single-list scope for quick/local gathers
        if self.stock_tickers_list:
            stock_files = [self.stock_tickers_list]
            logger.info(f"Scoped gather tickers from stock_tickers_list={self.stock_tickers_list}")

        logger.info(
            "Compiling list of stocks from symbol lists: {files}", files=stock_files
        )
        for f in stock_files:
            try:
                fp = resolve_symbol_csv(f)
            except FileNotFoundError:
                logger.info(f"{f} not found in symbol_lists or data-3rd-party.")
                continue
            logger.info(f"loading symbols from {fp}...")
            try:
                stocks = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
            except Exception as e:
                logger.warning(f"Failed to read {fp}: {e}")
                continue
            logger.info(f"loaded {len(stocks)} rows from {fp}")
            if "Symbol" not in stocks.columns:
                logger.warning(f"No Symbol column in {fp}; columns={list(stocks.columns)}")
                continue
            raw = stocks["Symbol"].dropna().astype(str).tolist()
            stock_set = {s.strip().upper() for s in raw if is_valid_ticker_symbol(s)}
            dropped = len(raw) - len(stock_set)
            if dropped:
                logger.warning(
                    f"{fp}: dropped {dropped} invalid Symbol values "
                    f"(numeric/price residue from unquoted CSV fields)"
                )
            all_stock_set |= stock_set
            logger.info(f"total symbols loaded: {len(all_stock_set)}")

        logger.info(f"Total size of stock set: {len(all_stock_set)}")
        slist = [
            x
            for x in set(all_stock_set)
            if is_valid_ticker_symbol(x) and len(str(x)) < 10
        ]
        self.stocks_ticker_set = set(slist)
        try:
            junk_path = resolve_symbol_csv("junk_tickers.csv")
            junk_tickers = set(pd.read_csv(junk_path)["Symbol"].dropna().astype(str))
            self.stocks_ticker_set = sorted(list(self.stocks_ticker_set - junk_tickers))
        except FileNotFoundError:
            logger.warning("junk_tickers.csv not found; keeping full ticker set")
            self.stocks_ticker_set = sorted(list(self.stocks_ticker_set))
        stocks_df = pd.DataFrame()
        stocks_df["Symbol"] = self.stocks_ticker_set
        stocks_df = stocks_df.set_index(["Symbol"])
        stocks_df.index = stocks_df.index.str.upper()
        stocks_df.index = stocks_df.index.drop_duplicates()
        stocks_df.index = stocks_df.index.dropna()
        # Persist to both symbol_lists and runtime data dir
        for out in (symbol_lists_dir() / self.all_stocks_file, Path(self._data_file(self.all_stocks_file))):
            out.parent.mkdir(parents=True, exist_ok=True)
            stocks_df.to_csv(out)
            logger.info(f"Saved stock set to {out}")

    def gather_fund_tickers(self):
        all_funds_set = set()
        fund_files = ["ibdfunds.csv", "industry_funds.csv"]
        logger.info("Compiling list of funds: {files}", files=fund_files)
        for f in fund_files:
            try:
                fp = resolve_symbol_csv(f)
            except FileNotFoundError:
                logger.error(f"{f} not found.")
                continue
            funds = pd.read_csv(fp)
            logger.info(f"loaded {len(funds)} fund symbols from {fp}")
            fund_set = set(funds["Symbol"].dropna().astype(str))
            all_funds_set |= fund_set
            logger.info(f"total fund symbols loaded: {len(all_funds_set)}")

        logger.info(f"Loaded fund tickers: \n{sorted(list(all_funds_set))}")
        return all_funds_set

    def _gather_yfdata_date_index(self, data_file: str = None, tickers: str = None):
        start_date = self.min_start_date
        old_df = None
        try:
            old_df = pd.read_parquet(data_file)
            logger.info("Loaded saved data. Sample: \n{df}", df=old_df)
            try:
                existing = set(old_df.columns.get_level_values(0))
            except Exception:
                existing = set()
            new_tickers = set(tickers) - existing
            if not new_tickers and len(old_df) > 0:
                start_date = get_latest_date(old_df.index) - pd.Timedelta(1, "d")
                logger.info(f"Latest saved record is after {start_date}")
            else:
                logger.info(f"Missing tickers vs saved data: {new_tickers}")
        except Exception as e:
            logger.warning(f"Could not load data from file: {data_file}. Error: {e}")
        session = self._yfinance_session()
        try:
            new_df = yf.download(
                list(tickers),
                start=start_date,
                group_by="tickers",
                auto_adjust=False,
                threads=True,
                progress=True,
                timeout=30,
                session=session,
            )
        except TypeError:
            # older yfinance without timeout kwarg
            new_df = yf.download(
                list(tickers),
                start=start_date,
                group_by="tickers",
                auto_adjust=False,
                threads=True,
                progress=True,
                session=session,
            )
        except Exception as e:
            logger.error(f"yfinance download failed for {data_file}: {e}")
            new_df = pd.DataFrame()
        if new_df is None or new_df.empty:
            logger.warning(f"Empty yfinance result for {data_file}; keeping prior data")
            return
        new_df = new_df.dropna(how="all")
        logger.info("New data gathered. Sample: \n{bm}", bm=new_df)
        logger.info(f"Columns: \n{new_df.columns}")
        if old_df is not None and not new_df.empty:
            # Align column multiindex if possible
            try:
                common = old_df.columns.intersection(new_df.columns)
                if len(common) == 0 and isinstance(new_df.columns, pd.MultiIndex):
                    new_df.columns = new_df.columns.swaplevel(0, 1)
                    new_df = new_df.sort_index(axis=1)
                    common = old_df.columns.intersection(new_df.columns)
                if len(common) > 0:
                    merged_df = pd.concat([old_df[common], new_df[common]], axis=0)
                    merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
                else:
                    logger.warning("No common columns; writing new download only")
                    merged_df = new_df
            except Exception as e:
                logger.warning(f"Merge failed ({e}); writing new download only")
                merged_df = new_df
        else:
            merged_df = new_df if not new_df.empty else old_df
        if merged_df is None or (hasattr(merged_df, "empty") and merged_df.empty):
            logger.warning(f"No data to save for {data_file}")
            return
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
        logger.info("Updated data ready. Sample: \n{bm}", bm=merged_df)
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")

    def gather_broad_market_data(self):
        ## Prepare data for broad market indicies
        # Capture S&P500, NASDAQ100 and Russell 200 indexes and their equal weighted counter parts
        # As well as VIX volatility index, DYX US Dollar index, TNX US 12 Weeks Treasury Yield, 5 Years Treasury Yield and 10 Year Treasuries Yield
        # Note: ^R2ESC is often unavailable from yfinance and can hang downloads
        broad_market_indicies = [
            "^SPX",
            "^SPXEW",
            "^NDX",
            "^NDXE",
            "^RUT",
            "^VIX",
            "DX-Y.NYB",
            "^IRX",
            "^FVX",
            "^TNX",
        ]
        data_file = self._data_file("broad_market.parquet")
        self._gather_yfdata_date_index(
            data_file=data_file, tickers=broad_market_indicies
        )

    def gather_sectors_data(self):
        """Gather historic price and volume data for key market sectors"""
        sector_indicies = "XLE ^SP500-15 ^SP500-20 ^SP500-25 ^SP500-30 ^SP500-35 ^SP500-40 ^SP500-45 ^SP500-50 ^SP500-55 ^SP500-60".split()
        data_file = self._data_file("sectors.parquet")
        self._gather_yfdata_date_index(data_file=data_file, tickers=sector_indicies)

    def gather_industry_fund_data(self):
        """
        Gather historic price and volume data for key industry ETFs.
        The goal of these covariates is to provide the model with a more granural breakdown of stock grouping by industry.
        Since stocks usually move together with their group, the model can learn which group(s) a stock moves with from these covariates.
        """
        fund_tickers = self.gather_fund_tickers()
        data_file = self._data_file("industry_funds.parquet")
        self._gather_yfdata_date_index(data_file=data_file, tickers=list(fund_tickers))

    def gather_stock_price_data(self):
        data_file = self._data_file(
            f"all_stocks_price_hist_{self.price_frequency}.parquet"
        )
        start_date = self.min_start_date
        old_df = None
        try:
            old_df = pd.read_parquet(data_file)
            logger.info("Loaded saved data. Sample: \n{df}", df=old_df)
            all_tickers = set(self.stocks_ticker_set)
            old_tickers = set(old_df.index.get_level_values("Symbol"))
            new_tickers = all_tickers - old_tickers
            if not new_tickers:
                start_date = get_latest_date(
                    old_df.index.get_level_values("Date")
                ) - pd.Timedelta(1, "d")
                logger.info(f"Columns: \n{old_df.columns}")
                logger.info(f"Latest saved record is after {start_date}")
            else:
                logger.info(
                    f"Not all tickers found in saved data. Missing: {new_tickers}"
                )
        except Exception as e:
            logger.warning(f"Could not load data from file: {data_file}. Error: {e}")
        session = self._yfinance_session()
        try:
            raw = yf.download(
                list(self.stocks_ticker_set),
                start=start_date,
                group_by="tickers",
                auto_adjust=False,
                threads=True,
                progress=True,
                timeout=30,
                session=session,
            )
        except TypeError:
            raw = yf.download(
                list(self.stocks_ticker_set),
                start=start_date,
                group_by="tickers",
                auto_adjust=False,
                threads=True,
                progress=True,
                session=session,
            )
        new_df = yfinance_multi_to_symbol_date(raw, tickers=list(self.stocks_ticker_set))
        new_df = coerce_ohlcv_numeric(new_df)
        # Keep only ground-truth complete bars
        ohlcv = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in new_df.columns]
        if ohlcv:
            new_df = new_df.dropna(subset=ohlcv, how="any")
        logger.info(f"Stock price data index: {new_df.index.names}")
        logger.info("New data gathered. Sample: \n{df}", df=new_df)
        if old_df is not None and not new_df.empty:
            cols = [c for c in old_df.columns if c in new_df.columns]
            if not cols:
                cols = list(new_df.columns)
            merged_df = pd.concat([old_df[cols] if set(cols) <= set(old_df.columns) else old_df,
                                   new_df[cols] if set(cols) <= set(new_df.columns) else new_df],
                                  axis=0, join="inner")
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
        else:
            merged_df = new_df if not new_df.empty else old_df
        if merged_df is None or merged_df.empty:
            raise RuntimeError("No stock price data gathered; aborting")
        assert merged_df.index.is_unique
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")

    def gather_earnings_data(self):
        logger.info("Gathering earnings and sales data...")
        earnings_all_df = None
        last_raw = None
        for ticker in self.stocks_ticker_set:
            earnings = fmpsdk.historical_earning_calendar(
                apikey=self.FMP_API_KEY, symbol=ticker, limit=-1
            )
            last_raw = earnings
            if earnings is not None and len(earnings) > 0:
                try:
                    edf = normalize_earnings_dataframe(earnings)
                    if not edf.empty and "symbol" in edf.columns and "date" in edf.columns:
                        edf = edf.set_index(["symbol", "date"])
                        earnings_all_df = pd.concat([earnings_all_df, edf])
                        logger.info(f"Total earnings reports for {ticker}: {len(edf)}")
                    else:
                        logger.warning(f"Skipping {ticker} due to lack of data")
                except ValueError as e:
                    logger.warning(
                        f"Skipping {ticker} due to error: {type(e)}, {e}, \n{earnings} "
                    )
            else:
                logger.warning(f"Skipping {ticker} due to lack of data")

        if earnings_all_df is None or earnings_all_df.empty:
            logger.warning("No earnings data gathered")
            return
        logger.info(f"Earnings sample report for last ticker: \n{last_raw}")
        earnings_all_df.index.names = ["Symbol", "Date"]
        earnings_all_df = earnings_all_df.sort_index()
        logger.info(
            f"Total number of earnings records for all stocks: \n{len(earnings_all_df)}"
        )
        earnings_file = self._data_file("earnings_calendar.parquet")
        # Remove broken symlink if present
        p = Path(earnings_file)
        if p.is_symlink() and not p.exists():
            p.unlink()
        earnings_all_df.to_parquet(earnings_file)
        tmp_earn_df = pd.read_parquet(earnings_file)
        pd.testing.assert_frame_equal(tmp_earn_df, earnings_all_df)
        logger.info(
            f"Sanity check passed for earnings data. Loaded OK from file: {earnings_file}"
        )

    def gather_stock_key_metrics(self):
        logger.info("Gathering key metrics data with company fundamentals...")
        keymetrics_all_df = None
        for ticker in self.stocks_ticker_set:
            kms = fmpsdk.key_metrics(
                apikey=self.FMP_API_KEY, symbol=ticker, period="quarter", limit=-1
            )
            if kms is not None and len(kms) > 0:
                try:
                    kms_df = normalize_key_metrics_dataframe(kms)
                    if not kms_df.empty and "symbol" in kms_df.columns and "date" in kms_df.columns:
                        kms_df = kms_df.set_index(["symbol", "date"])
                        keymetrics_all_df = pd.concat([keymetrics_all_df, kms_df])
                        logger.info(f"Total key metrics reports for {ticker}: {len(kms_df)}")
                    else:
                        logger.warning(f"Skipping {ticker} due to lack of data")
                except ValueError as e:
                    logger.warning(
                        f"Skipping {ticker} due to error: {type(e)}, {e}, \n{kms} "
                    )
            else:
                logger.info(f"No {ticker} key metrics reports: kms={kms}")

        if keymetrics_all_df is None or keymetrics_all_df.empty:
            logger.warning("No key metrics data gathered")
            return
        kms_num_cols = list(set(keymetrics_all_df.columns) - {"calendarYear", "period"})
        keymetrics_all_df[kms_num_cols] = keymetrics_all_df[kms_num_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        keymetrics_all_df.index.names = ["Symbol", "Date"]
        keymetrics_all_df = keymetrics_all_df.sort_index()
        kms_file = self._data_file("keymetrics_history.parquet")
        p = Path(kms_file)
        if p.is_symlink() and not p.exists():
            p.unlink()
        keymetrics_all_df.to_parquet(kms_file, engine="pyarrow")
        temp_kms_df = pd.read_parquet(kms_file)
        pd.testing.assert_frame_equal(temp_kms_df, keymetrics_all_df)
        logger.info(
            f"Sanity check passed for key metrics data. Loaded OK from file: {kms_file}"
        )

    def gather_stock_dividends(self):
        logger.info("Gathering stock dividends data...")
        all_df = None
        for ticker in self.stocks_ticker_set:
            logger.info(f"Gathering report for {ticker}")
            raw = fmpsdk.historical_stock_dividend(
                apikey=self.FMP_API_KEY, symbol=ticker
            )
            # skip symbols without any data
            if raw is not None:
                raw = raw.get("historical")
            if raw is not None and len(raw) > 0:
                # logger.info(f"Sample raw report for {ticker}: \n{raw}")
                df = pd.DataFrame(raw)
                # logger.debug(f"df for {ticker}: \n{df}")
                df = df.dropna(how="all")
                df["date"] = pd.to_datetime(df["date"])
                df["symbol"] = ticker
                df = df.set_index(["symbol", "date"])
                all_df = pd.concat([all_df, df])
                logger.debug(f"Total reports for {ticker}: {len(df)}")

        if all_df is not None:
            logger.debug(f"Sample report for {ticker}: \n{df}")
            all_df.index.names = ["Symbol", "Date"]
            all_df = all_df.sort_index()
            logger.info(f"Total number of records for all stocks: \n{len(all_df)}")
            logger.debug(f"all_df: \n{all_df}")
            logger.info(f"len(all_df.index.levels[0]): \n{len(all_df.index.levels[0])}")
            file = self._data_file("stock_dividends.parquet")
            all_df.to_parquet(file)
            ### Read back data and verify it
            tmp_df = pd.read_parquet(file)
            logger.debug(f"tmp_df: \n{tmp_df}")
            pd.testing.assert_frame_equal(tmp_df, all_df)
            logger.info(f"Sanity check passed. Data loaded OK from file: {file}")
        logger.info("Finished gathering stock dividends data.")

    def gather_stock_splits(self):
        logger.info("Gathering stock splits data...")
        all_df = None
        for ticker in self.stocks_ticker_set:
            logger.info(f"Gathering report for {ticker}")
            raw = fmpsdk.historical_stock_split(apikey=self.FMP_API_KEY, symbol=ticker)
            # skip symbols without any data
            if raw is not None:
                raw = raw.get("historical")
            if raw is not None and len(raw) > 0:
                # logger.debug(f"Sample raw report for {ticker}: \n{raw}")
                df = pd.DataFrame(raw)
                # logger.debug(f"df for {ticker}: \n{df}")
                df = df.dropna(how="all")
                df["date"] = pd.to_datetime(df["date"])
                df["symbol"] = ticker
                df = df.set_index(["symbol", "date"])
                all_df = pd.concat([all_df, df])
                logger.debug(f"Total reports for {ticker}: {len(df)}")

        if all_df is not None:
            logger.debug(f"Sample report for {ticker}: \n{df}")
            all_df.index.names = ["Symbol", "Date"]
            all_df = all_df.sort_index()
            logger.info(f"Total number of records for all stocks: \n{len(all_df)}")
            logger.debug(f"all_df: \n{all_df}")
            logger.info(f"len(all_df.index.levels[0]): \n{len(all_df.index.levels[0])}")
            file = self._data_file("stock_splits.parquet")
            all_df.to_parquet(file)
            ### Read back data and verify it
            tmp_df = pd.read_parquet(file)
            logger.debug(f"tmp_df: \n{tmp_df}")
            pd.testing.assert_frame_equal(tmp_df, all_df)
            logger.info(f"Sanity check passed. Data loaded OK from file: {file}")
        logger.info("Finished gathering stock splits data.")

    def gather_institutional_stock_ownership(self):
        logger.info("Gathering institutional ownership data...")
        inst_ownership_all_df = None
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
            self._data_file("institutional_symbol_ownership.parquet")
        )
        inst_own_df.to_parquet(inst_ownership_file, engine="pyarrow")

        ### Read back data and verify it
        tmp_int_own_df = pd.read_parquet(inst_ownership_file)
        logger.info(f"tmp_int_own_df: \n{tmp_int_own_df}")
        pd.testing.assert_frame_equal(tmp_int_own_df, inst_own_df)
        logger.info(
            f"Sanity check passed for institutional ownership data. Loaded OK from file: {inst_ownership_file}"
        )

    def gather_analyst_estimates(self):

        def _fetch_estimates(period=None):
            assert period in ["quarter", "annual"]
            estimates_all_df = None
            for ticker in self.stocks_ticker_set:  # ['ALTR']:
                est = analyst_estimates(
                    apikey=self.FMP_API_KEY, symbol=ticker, period=period, limit=-1
                )
                if est is not None and len(est) > 0:
                    try:
                        # logger.debug(f"{ticker} estimates: {est}")
                        est_df = pd.DataFrame(est)
                        est_df = est_df.dropna(how="all")
                        est_df = est_df.fillna(-1)
                        if not est_df.empty:
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
                            logger.warning(f"Skipping {ticker} due to lack of data")
                    except ValueError as e:
                        logger.warning(
                            f"Skipping {ticker} due to error: {type(e)}, {e}, \n{est} "
                        )
                else:
                    logger.info(
                        f"No {ticker} {period} analyst estimates reports: est={est}"
                    )

            return estimates_all_df

        logger.info("Gathering analyst estimates...")
        for p in ["annual", "quarter"]:
            est_file_name = self._data_file(f"analyst_estimates_{p}.parquet")
            estimates_all_df = _fetch_estimates(p)
            if estimates_all_df is None or estimates_all_df.empty:
                logger.warning(f"No analyst estimates for period={p}")
                continue
            estimates_all_df.index.names = ["Symbol", "Date"]
            estimates_all_df = estimates_all_df.sort_index()
            # est_file_name= f'data/analyst_estimates_{p}.csv.bz2'
            # estimates_all_df.to_csv(est_file_name)
            estimates_all_df.to_parquet(est_file_name)
            logger.info(f"Saved analyst estimates to: {est_file_name}")
            logger.info(f"all {p} estimates count: {len(estimates_all_df)}")
            # logger.info(f"{p} estimates sample:\n{estimates_all_df}")
            tmp_est_df = pd.read_parquet(est_file_name)
            pd.testing.assert_frame_equal(tmp_est_df, estimates_all_df)
            logger.info(
                f"Sanity check passed for analyst estimates data. Loaded OK from file: {est_file_name}"
            )


# main function
def main():
    """Local-first gather: refresh market data from FMP/yfinance into local parquet.

    Does **not** download/upload the full Hugging Face dataset unless
    ``hfhub_sync=True``. Optional one-shot CSV bootstrap via
    ``SYNC_SYMBOL_LISTS=1``.
    """
    load_dotenv(override=True)
    # Optional light CSV pull only (not historical parquet)
    if _env_bool("SYNC_SYMBOL_LISTS", default=False):
        try:
            HFHub(api_key=os.getenv("HF_TOKEN")).download_symbol_list_csvs()
        except Exception as e:
            logger.warning(f"Symbol list sync skipped: {e}")

    # HF full-dataset sync is opt-in and runs only if explicitly enabled
    hfhub = None
    if _env_bool("hfhub_sync", default=False):
        hfhub = HFHub()
        logger.warning(
            "hfhub_sync=True: performing optional HF dataset download (can be slow)"
        )
        hfhub.download_data()
    else:
        logger.info(
            "Local-first gather (hfhub_sync off): no HF dataset download/upload"
        )

    g = MarketDataGatherer()
    g.gather_stock_tickers()
    g.gather_broad_market_data()
    g.gather_sectors_data()
    g.gather_industry_fund_data()
    g.gather_stock_price_data()
    g.gather_stock_dividends()
    g.gather_stock_splits()
    g.gather_earnings_data()
    g.gather_stock_key_metrics()
    g.gather_institutional_stock_ownership()
    g.gather_analyst_estimates()

    if hfhub is not None:
        hfhub.upload_data()
    logger.info("Gather finished (local data dir updated).")


if __name__ == "__main__":
    main()
