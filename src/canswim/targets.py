from typing import Union
from loguru import logger
from darts import TimeSeries
import pandas as pd

from canswim.eligibility import filter_eligible_price_dict, price_history_is_eligible


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
        stocks_df = pd.read_parquet(
            stocks_price_file,
            filters=self.pyarrow_filters,
            dtype_backend="numpy_nullable",
        )
        logger.info("filtered data loaded")
        # Drop rows missing ground-truth OHLCV (do not invent)
        ohlcv = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in stocks_df.columns]
        stocks_df = stocks_df.dropna(subset=ohlcv, how="any")
        stock_price_dict = {}
        tickers = list(stocks_df.index.levels[0])
        logger.info(f"price history loaded for {len(tickers)} stocks: \n{tickers}")
        for t in tickers:
            stock_full_hist = stocks_df.loc[[t]]
            if len(stock_full_hist.index) >= self.min_samples:
                stock_full_hist = stock_full_hist.droplevel("Symbol")
                stock_full_hist.index = pd.to_datetime(stock_full_hist.index)
                # Drop Adj Close (retroactively adjusted by vendor)
                if "Adj Close" in stock_full_hist.columns:
                    stock_full_hist = stock_full_hist.drop(columns=["Adj Close"])
                ok, reason = price_history_is_eligible(
                    stock_full_hist, min_samples=self.min_samples
                )
                if ok:
                    stock_price_dict[t] = stock_full_hist
                else:
                    logger.info(
                        f"Skipping {t} from price series (ground-truth check): {reason}"
                    )
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
            return new_series

        if isinstance(target_columns, list) and len(target_columns) == 1:
            target_columns = target_columns[0]
            logger.info(f"Single target column selected: {target_columns}")

        if isinstance(target_columns, str):
            target_series = {
                t: stock_price_series[t].univariate_component(target_columns)
                for t in stock_price_series.keys()
            }
            logger.info(f"Preparing univariate target series: {target_columns}")
        else:
            target_series = {
                t: drop_non_target_columns(s) for t, s in stock_price_series.items()
            }
            logger.info(f"Preparing multivariate target series: {target_columns}")
        self.target_series = target_series

    def prepare_stock_price_series(self, train_date_start: pd.Timestamp = None):
        """Build price TimeSeries from ground-truth bars only (no MissingValuesFiller).

        Tickers with internal missing trading-day OHLC are excluded rather than
        having invented prices interpolated.
        """
        loaded_tickers = list(self.stock_price_dict.keys())
        logger.info(
            f"Preparing ticker series for {len(loaded_tickers)} stocks: \n{loaded_tickers}"
        )
        # Re-filter eligibility after any upstream mutation
        self.stock_price_dict = filter_eligible_price_dict(
            self.stock_price_dict, min_samples=max(self.min_samples, 1)
        )
        stock_price_series = {}
        # NYSE trading-day frequency: no Mon–Fri holiday placeholders / invented bars
        try:
            import pandas_market_calendars as mcal

            _holidays = mcal.get_calendar("NYSE").holidays().holidays
            trading_freq = pd.offsets.CustomBusinessDay(holidays=_holidays)
        except Exception as e:
            logger.warning(f"NYSE CustomBusinessDay unavailable ({e}); using 'B'")
            trading_freq = "B"

        for t, df in self.stock_price_dict.items():
            try:
                # Eligibility already required complete OHLCV on NYSE sessions.
                # Use exchange calendar freq so darts does not invent holiday bars
                # via MissingValuesFiller / naive 'B' reindex.
                series = TimeSeries.from_dataframe(
                    df.sort_index(),
                    fill_missing_dates=True,
                    freq=trading_freq,
                )
                # Any remaining NaNs mean a missing true session → exclude (no fill)
                if series.pd_dataframe().isna().any().any():
                    logger.info(
                        f"Skipping {t}: nulls after calendar align; "
                        "refusing to invent missing market bars"
                    )
                    continue
                if train_date_start is not None:
                    series = series.slice(train_date_start, series.end_time())
                if len(series) < self.min_samples:
                    logger.info(
                        f"Skipping {t}: only {len(series)} samples after slice "
                        f"(need >= {self.min_samples})"
                    )
                    continue
                stock_price_series[t] = series
            except Exception as e:
                logger.warning(f"Skipping {t} while building TimeSeries: {type(e)}: {e}")
        logger.info(
            f"Prepared {len(stock_price_series)} ground-truth price series "
            f"(no MissingValuesFiller / no synthetic OHLC)."
        )
        return stock_price_series
