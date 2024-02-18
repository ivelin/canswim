"""
Forecast stock price movement and upload results to HF Hub
"""

import pandas as pd
from loguru import logger
from canswim.model import CanswimModel
import pandas as pd
from pandas import DataFrame
from canswim.hfhub import HFHub
from matplotlib import pyplot
from pandas.tseries.offsets import BDay
from loguru import logger
from pandas.tseries.offsets import BDay
import os


class CanswimForecaster:
    def __init__(self):
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        self.stock_train_list = os.getenv("stocks_train_list", "all_stocks.csv")
        logger.info(f"Stocks train list: {self.stock_train_list}")
        self.n_stocks = int(os.getenv("n_stocks", 50))
        logger.info(f"n_stocks: {self.n_stocks}")
        self.forecast_data_file = os.getenv(
            "forecast_data_file", "forecast_data.parquet"
        )
        logger.info(f"Forecast data file: {self.forecast_data_file}")
        self.canswim_model = CanswimModel()
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
            # such as holidays. Apparently there are about 15 non-weekend days when the market is off
            # and price data is not available on such days.
            # Add a sufficiently big sample pad to include all market off days.
            n=self.canswim_model.min_samples + self.canswim_model.train_history
        )

    def get_forecast(self):
        logger.info("Forecast start. Calling model predict().")
        canswim_forecast = self.canswim_model.predict(
            target=self.canswim_model.targets_list,
            past_covariates=self.canswim_model.past_cov_list,
            future_covariates=self.canswim_model.future_cov_list,
        )
        logger.info("Forecast finished.")
        return canswim_forecast

    def prep_next_stock_group(self):
        """Generator which iterates over all stocks and prepares them in groups."""
        stocks_file = f"{self.data_dir}/{self.data_3rd_party}/{self.stock_train_list}"
        logger.info(f"Loading stock tickers from {stocks_file}")
        all_stock_tickers = pd.read_csv(stocks_file)
        logger.info(f"Loaded {len(all_stock_tickers)} symbols in total")
        stock_list = list(set(all_stock_tickers["Symbol"]))
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

    def _list_to_df(self, forecast_list: [] = None):
        """Format list of forecasts as a dataframe to be saved as a partitioned parquet dir"""
        forecast_df = pd.DataFrame(
            columns=[
                "symbol",
                "forecast_start_year",
                "forecast_start_month",
                "forecast_start_day",
                "target_date",
                "target_close",
            ]
        )
        forecast_df.set_index(
            [
                "symbol",
                "forecast_start_year",
                "forecast_start_month",
                "forecast_start_day",
            ]
        )
        for i, t in enumerate(self.canswim_model.targets_ticker_list):
            df = forecast_list[i].pd_dataframe()
            df["symbol"] = t
            df["forecast_start_year"] = self.start_date.year
            df["forecast_start_month"] = self.start_date.month
            df["forecast_start_day"] = self.start_date.day
            forecast_df = pd.concat([forecast_df, df])
        return forecast_df

    def save_forecast(self, forecast_list: [] = None):
        """Saves forecast data to local database"""
        assert forecast_list is not None and len(forecast_list) > 0
        forecast_df = self._list_to_df(forecast_list)
        logger.info(
            f"Saving forecast_df with {len(forecast_df.columns)} columns, {len(forecast_df)} rows: {forecast_df}"
        )
        forecast_df.to_parquet(
            f"{self.data_dir}/forecast",
            partition_cols=[
                "symbol",
                "forecast_start_year",
                "forecast_start_month",
                "forecast_start_day",
            ],
        )
        logger.info(f"Saved forecast data to: {self.forecast_data_file}")

    def upload_data(self):
        self.hfhub.upload_data()


# main function
def main():
    logger.info("Running forecast on stocks and uploading results to HF Hub...")
    cf = CanswimForecaster()
    cf.download_model()
    # cf.load_model()
    cf.download_data()
    # loop in groups over all stocks
    for pos in cf.prep_next_stock_group():
        forecast = cf.get_forecast()
        # save new or update existing data file
        cf.save_forecast(forecast)
    cf.upload_data()
    logger.info("Finished forecast and uploaded results to HF Hub.")


if __name__ == "__main__":
    main()
