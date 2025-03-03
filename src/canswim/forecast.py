"""
Forecast stock price movement and upload results to HF Hub
"""

import pandas as pd
from loguru import logger
from canswim.model import CanswimModel
from canswim.hfhub import HFHub
from pandas.tseries.offsets import BDay
from loguru import logger
import os
from canswim import constants
from typing import List
import duckdb
from datetime import datetime, timedelta
import pandas_market_calendars as mcal


class CanswimForecaster:
    def __init__(self):
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        self.stock_tickers_list = os.getenv("stock_tickers_list", "all_stocks.csv")
        logger.info(f"Stocks train list: {self.stock_tickers_list}")
        self.n_stocks = int(os.getenv("n_stocks", 50))
        logger.info(f"n_stocks: {self.n_stocks}")
        self.forecast_subdir = os.getenv("forecast_subdir", "forecast/")
        logger.info(f"Forecast data path: {self.forecast_subdir}")
        self.canswim_model = CanswimModel(forecast_only=True)
        self.hfhub = HFHub()

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model()
        logger.info("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def load_model(self):
        """Load model from local storage"""
        self.canswim_model.load()
        logger.info("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data()
        # load raw data from hf hub
        self.start_date = pd.Timestamp.now() - BDay(
            # min_samples is not sufficient as it does not account for all market days without price data
            # such as holidays. Also for some sparse data sets such as analyst estimates, there are periods of
            # 2 or more years without data.
            # Add a sufficiently big sample pad to include all market off days.
            n=self.canswim_model.min_samples
            * 2
        )

    def get_forecast(self, forecast_start_date: pd.Timestamp = None):
        logger.info("Forecast start. Calling model predict().")
        forecasted_tickers = []
        target_sliced_list = []
        past_cov_list = []
        future_cov_list = []
        tickers_list = self.canswim_model.targets_ticker_list
        # trim end of targets to specified forecast start date
        if forecast_start_date is not None:
            logger.debug(
                f"Dropping target samples after forecast start date, included: {forecast_start_date}"
            )
            for i, ts in enumerate(self.canswim_model.targets_list):
                try:
                    logger.debug(
                        f"Target {tickers_list[i]} start date, end date, sample count: {ts.start_time()}, {ts.end_time()}, {len(ts)}"
                    )
                    # if forecast start date is after the end of the target time series,
                    # then we can still forecast if the target series ends on the open market business day before the forecast start date
                    cutoff_forecast_start_date = get_next_open_market_day(after_date=ts.end_time())
                    if forecast_start_date == cutoff_forecast_start_date:
                        target_sliced = ts
                    if forecast_start_date < cutoff_forecast_start_date:
                        target_sliced = ts.drop_after(forecast_start_date)
                    # we can only provide forecasts
                    # when forecast start date is immediately after target series end date
                    # and there are sufficient number of historical data samples
                    if (
                        forecast_start_date <= cutoff_forecast_start_date
                        and len(ts) >= self.canswim_model.min_samples
                    ):
                        forecasted_tickers.append(tickers_list[i])
                        target_sliced_list.append(target_sliced)
                        past_cov_list.append(self.canswim_model.past_cov_list[i])
                        future_cov_list.append(self.canswim_model.future_cov_list[i])
                    else:
                        logger.info(
                            f"Skipping {tickers_list[i]} for forecast start date {forecast_start_date} due to lack of historical data. Min {self.canswim_model.min_samples} samples needed. Historical timeseries has {len(ts)} samples and ends on {ts.end_time()}."
                        )
                except ValueError as e:
                    logger.warning(
                        f"Skipping {tickers_list[i]} for forecast start date {forecast_start_date} due to error: {type(e)}: {e}"
                    )
        if len(target_sliced_list) > 0:
            canswim_forecast = self.canswim_model.predict(
                target=target_sliced_list,
                past_covariates=past_cov_list,
                future_covariates=future_cov_list,
            )
            logger.info("Forecast finished.")
            forecasts = dict(zip(forecasted_tickers, canswim_forecast))
            return forecasts
        else:
            logger.warning(
                "No stocks have enough data in this batch. Skipping forecast."
            )
            return None

    def _get_stocks_without_forecast(self, stocks_df=None, forecast_start_date=None):
        if forecast_start_date is not None:
            dt = pd.Timestamp(forecast_start_date)
        else:
            dt = pd.Timestamp.now()
        # align date to closest business day
        # leave as is if dt is a business day,
        # otherwise move forward to next business day
        bd = dt + 0 * BDay()
        y = bd.year
        m = bd.month
        d = bd.day
        # logger.debug(f"Forecast start date, year, month, day: {bd}, {y}, {m}, {d}")
        # logger.debug(f"len(stocks_df): {len(stocks_df)}")
        # logger.debug(f"stocks_df: {stocks_df}")
        df = duckdb.sql(
            f"""--sql
            CREATE OR REPLACE TABLE stock_group AS SELECT Symbol from stocks_df;
            SELECT symbol, count(*), forecast_start_year, forecast_start_month, forecast_start_day
            FROM read_parquet('{self.data_dir}/{self.forecast_subdir}**/*.parquet', hive_partitioning = 1) as f
            SEMI JOIN stock_group
            ON f.symbol = stock_group.symbol
            GROUP BY f.symbol, forecast_start_year, forecast_start_month, forecast_start_day
            HAVING 
                forecast_start_year={y} AND
                forecast_start_month={m} AND
                forecast_start_day={d} AND
                count(*) >= {self.canswim_model.pred_horizon}
            """
        ).df()
        # logger.debug(f"sql result: {df}")
        stocks_with_saved_forecast = set(df["symbol"])
        logger.debug(
            f"""These stocks already have a saved forecast: {stocks_with_saved_forecast}"""
        )
        stocks_without_forecast = set(stocks_df["Symbol"]) - stocks_with_saved_forecast
        stocks_without_forecast = sorted(list(stocks_without_forecast))
        logger.debug(
            f"""These stocks do not have a saved forecast yet: {stocks_without_forecast}"""
        )
        return stocks_without_forecast

    def prep_next_stock_group(self, forecast_start_date=None):
        """Generator which iterates over all stocks and prepares them in groups."""
        stocks_file = f"{self.data_dir}/{self.data_3rd_party}/{self.stock_tickers_list}"
        logger.info(f"Loading stock tickers from {stocks_file}.")
        all_stock_tickers = pd.read_csv(stocks_file)
        # trim any duplicate tickers
        all_stock_tickers.index = all_stock_tickers.index.drop_duplicates()
        # drop any empty symbols/tickers
        all_stock_tickers = all_stock_tickers[all_stock_tickers.index.notnull()]
        logger.info(f"Loaded {len(all_stock_tickers)} symbols in total")
        stock_list = self._get_stocks_without_forecast(
            stocks_df=all_stock_tickers, forecast_start_date=forecast_start_date
        )
        logger.info(
            f"{len(all_stock_tickers)-len(stock_list)} stock already have forecast saved."
        )
        logger.info(f"{len(stock_list)} stock tickers candidates for new forecast.")
        if self.n_stocks < 0 or self.n_stocks > len(stock_list):
            self.n_stocks = len(stock_list)
        # group tickers in workable sample sizes for each forecast pass
        # credit ref: https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks
        for pos in range(0, len(stock_list), self.n_stocks):
            stock_group = stock_list[pos : pos + self.n_stocks]
            self.canswim_model.load_data(
                stock_tickers=stock_group, start_date=self.start_date
            )
            # prepare timeseries for forecast
            self.canswim_model.prepare_forecast_data(start_date=self.start_date)
            logger.info(f"Prepared forecast data for {len(stock_group)}: {stock_group}")
            yield pos

    def save_forecast(self, forecasts: dict = None):
        """Saves forecast data to local database"""

        def _list_to_df(forecasts: dict = None):
            """Format list of forecasts as a dataframe to be saved as a partitioned parquet dir"""
            forecast_df = pd.DataFrame()
            for t, ts in forecasts.items():
                pred_start = ts.start_time()
                # logger.debug(f"Next forecast timeseries: {ts}")
                # normalize name of target series column if needed (e.g. "Adj Close" -> "Close")
                if self.canswim_model.target_column != "Close":
                    ts = ts.with_columns_renamed(
                        self.canswim_model.target_column, "Close"
                    )
                # convert probabilistic forecast into a dataframe with quantile samples as columns Close_01, Close_02...
                df = ts.pd_dataframe()
                # save commonly used quantiles
                for q in constants.quantiles:
                    qseries = df.quantile(q=q, axis=1)
                    qname = f"close_quantile_{q}"
                    df[qname] = qseries
                df["symbol"] = t
                df["forecast_start_year"] = pred_start.year
                df["forecast_start_month"] = pred_start.month
                df["forecast_start_day"] = pred_start.day
                # logger.debug(f"Next forecast sample: {df}")
                forecast_df = pd.concat([forecast_df, df])
            return forecast_df

        assert forecasts is not None and len(forecasts) > 0
        forecast_df = _list_to_df(forecasts)
        logger.info(
            f"Saving forecast_df with {len(forecast_df.columns)} columns, {len(forecast_df)} rows: {forecast_df}"
        )
        forecast_df.to_parquet(
            f"{self.data_dir}/{self.forecast_subdir}",
            partition_cols=[
                "symbol",
                "forecast_start_year",
                "forecast_start_month",
                "forecast_start_day",
            ],
        )
        logger.info(f"Saved forecast data to: {self.forecast_subdir}")

    def upload_data(self):
        self.hfhub.upload_data()

def get_next_open_market_day(after_date=None):
    """Get the date of the next open market day after a given date: after_date when provided or after today otherwise."""
    # Get calendar for NYSE
    nyse = mcal.get_calendar('NYSE')
    
    if after_date is None:
        # Get today's date
        today = datetime.now().date()
        after_date = today
    
    # Look for the next valid trading day within a reasonably big window of 20 regular business days
    valid_days = nyse.valid_days(start_date=after_date+BDay(1), end_date=after_date + BDay(20), tz=None)
    
    next_trading_day = None

    if valid_days is not None and len(valid_days) > 0:
        next_trading_day = valid_days[0]

    if next_trading_day is not None:
        logger.debug(f"The next open stock market date is: {next_trading_day}")
    else:
        logger.info("No open market day found within the next 30 days.")

    # If we can't find a next valid day within 30 days (which shouldn't happen for NYSE), return None
    return next_trading_day


# main function
def main(forecast_start_date: str = None):
    logger.info("Running forecast on stocks and uploading results to HF Hub...")
    if forecast_start_date is not None:
        logger.info(f"forecast_start_date: {forecast_start_date}")
        forecast_start_date = pd.Timestamp(forecast_start_date, tz=None)
    else:
        # get next open stock market date
        # Example usage
        forecast_start_date = get_next_open_market_day()

    logger.info(f"Forecast start date set to: {forecast_start_date}")

    cf = CanswimForecaster()
    cf.download_model()
    cf.download_data()
    ## loop in groups over all stocks
    # next(cf.prep_next_stock_group())
    for pos in cf.prep_next_stock_group(forecast_start_date=forecast_start_date):
        forecasts = cf.get_forecast(forecast_start_date=forecast_start_date)
        ## save new or update existing data file
        if forecasts:
            cf.save_forecast(forecasts)
    cf.upload_data()
    logger.info("Finished forecast task.")


if __name__ == "__main__":
    main()
