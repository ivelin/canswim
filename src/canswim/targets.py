from typing import Union
from loguru import logger
from darts.dataprocessing.transformers import MissingValuesFiller
from darts import TimeSeries
import pandas as pd


class Targets:
    def __init__(self) -> None:
        self.target_series = {}

    def load_data(
        self,
        stock_tickers: set = None,
        min_samples: int = -1,
        start_date: pd.Timestamp = None,
    ):
        self.__start_date = start_date
        self.__load_tickers = stock_tickers
        self.min_samples = min_samples
        self.load_stock_prices()

    @property
    def pyarrow_filters(self):
        return [
            ("Symbol", "in", self.__load_tickers),
            ("Date", ">=", self.__start_date),
        ]

    def load_stock_prices(self):
        stocks_price_file = "data/data-3rd-party/all_stocks_price_hist_1d.parquet"
        logger.info(
            f"Loading data from: {stocks_price_file} with filter: {self.pyarrow_filters}"
        )
        # load into a dataframe with valid market calendar days
        # stocks_df = pd.read_csv(
        #     stocks_price_file, header=[0, 1], index_col=0, on_bad_lines="warn"
        # )
        stocks_df = pd.read_parquet(
            stocks_price_file,
            filters=self.pyarrow_filters,
            dtype_backend="numpy_nullable",
        )
        logger.info("filtered data loaded")
        stocks_df = stocks_df.dropna()
        # stocks_df.index = pd.to_datetime(stocks_df.index)
        stock_price_dict = {}
        # stock_tickers = self.__get_stock_tickers(stocks_df)
        tickers = list(stocks_df.index.levels[0])
        logger.info(f"price history loaded for {len(tickers)} stocks: \n{tickers}")
        for t in tickers:
            # logger.info(f"validating price data for {t}")
            stock_full_hist = stocks_df.loc[[t]]
            if len(stock_full_hist.index) >= self.min_samples:
                stock_full_hist = stock_full_hist.droplevel("Symbol")
                stock_full_hist.index = pd.to_datetime(stock_full_hist.index)
                # Drop Adj Close because yfiance changes its values retroactively at future dates
                # after stock dividend or split dates, which makes training data less stable
                # Ref: https://help.yahoo.com/kb/adjusted-close-sln28256.html
                stock_price_dict[t] = stock_full_hist.drop(columns=["Adj Close"])
                # logger.info(f'ticker: {t}')
                # logger.info(f'ticker historic data: {ticker_dict[t]}')
            else:
                logger.info(
                    f"Skipping {t} from price series. Not enough samples for model training."
                )
        self.stock_price_dict = stock_price_dict

    def prepare_data(
        self, stock_price_series: dict = None, target_columns: Union[str, list] = None
    ):
        def drop_non_target_columns(series):
            cols = series.columns
            non_target_columns = list(set(cols) - set(target_columns))
            new_series = series.drop_columns(col_names=non_target_columns)
            # logger.info(f'dropped non-target columns: {non_target_columns}')
            return new_series

        if isinstance(target_columns, list) and len(target_columns) == 1:
            target_columns = target_columns[0]
            logger.info(f"Single target column selected: {target_columns}")

        if isinstance(target_columns, str):
            # prepare target univariate series for Close price
            target_series = {
                t: stock_price_series[t].univariate_component(target_columns)
                for t in stock_price_series.keys()
            }
            logger.info(f"Preparing univariate target series: {target_columns}")
        else:
            # prepare target multivariate series for Open, Close and Volume
            target_series = {
                t: drop_non_target_columns(s) for t, s in stock_price_series.items()
            }
            logger.info(f"Preparing multivariate target series: {target_columns}")
        self.target_series = target_series

    def prepare_stock_price_series(self, train_date_start: pd.Timestamp = None):
        loaded_tickers = self.stock_price_dict.keys()
        logger.info(
            f"Preparing ticker series for {len(loaded_tickers)} stocks: \n{loaded_tickers}"
        )
        stock_price_series = {
            t: TimeSeries.from_dataframe(self.stock_price_dict[t], freq="B")
            for t in loaded_tickers
        }
        logger.info("Ticker series dict created.")
        filler = MissingValuesFiller()
        for t, series in stock_price_series.items():
            # gaps = series.gaps(mode="any")
            # logger.info(f'ticker: {t} gaps: \n {gaps}')
            series_filled = filler.transform(series)
            # check for any data gaps
            price_gaps = series_filled.gaps(mode="any")
            assert len(price_gaps) == 0
            # logger.info(f'ticker: {t} gaps after filler: \n {any_price_gaps}')
            stock_price_series[t] = series_filled
        logger.info("Filled missing values in ticker series.")
        for t, series in stock_price_series.items():
            stock_price_series[t] = series.slice(train_date_start, series.end_time())
            # logger.info(f'ticker: {t} , {ticker_series[t]}')
        # add holidays as future covariates
        logger.info("Aligned ticker series dict with train start date.")
        logger.info("Ticker series prepared.")
        return stock_price_series
