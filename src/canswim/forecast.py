"""
Forecast stock price movement and upload results to HF Hub
"""

import yfinance as yf
import pandas as pd
from loguru import logger
from canswim.model import CanswimModel
import pandas as pd
from canswim.hfhub import HFHub
from matplotlib import pyplot
from pandas.tseries.offsets import BDay
from loguru import logger
from pandas.tseries.offsets import BDay


class CanswimForecaster:

    def __init__(self):
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model()
        logger.info("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data()
        # load raw data from hf hub
        start_date = pd.Timestamp.now() - BDay(n=self.canswim_model.min_samples)
        self.canswim_model.load_data(start_date=start_date)
        # prepare timeseries for forecast
        self.canswim_model.prepare_forecast_data(start_date=start_date)
        self.tickers = list(self.canswim_model.targets.target_series.keys())

    def get_forecast(self):
        logger.info("Forecast start. Running model predict().")
        canswim_forecast = self.canswim_model.predict(
            target=self.canswim_model.targets_list,
            past_covariates=self.canswim_model.past_cov_list,
            future_covariates=self.canswim_model.future_cov_list,
        )
        logger.info(f"Forecast finished.")
        return canswim_forecast


# main function
def main():
    logger.info("Running forecast on stocks and uploading results to HF Hub...")
    f = CanswimForecaster()
    f.download_model()
    # ...loop in batches over all stocks
    for stocks in next_batch():
        f.download_data()
        f.get_forecast()
        # save new or update existing data file
        f.save_forecast()
    f.upload_data()
    logger.info("Finished forecast and uploaded results to HF Hub.")


if __name__ == "__main__":
    main()
