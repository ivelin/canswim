"""CANSWIM Playground. A gradio app intended to be deployed on HF Hub."""

from canswim.model import CanswimModel
import pandas as pd
import gradio as gr
from canswim.hfhub import HFHub
from loguru import logger
from canswim.dashboard.charts import ChartTab
from canswim.dashboard.scans import ScanTab
from canswim.dashboard.advanced import AdvancedTab
from pandas.tseries.offsets import BDay
import duckdb
import os

# Note: It appears that gradio Plot ignores the backend plot lib setting
# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

repo_id = "ivelin/canswim"

class CanswimPlayground:

    def __init__(self):
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()
        data_dir = os.getenv("data_dir", "data")
        forecast_subdir = os.getenv(
            "forecast_subdir", "forecast/"
        )
        self.forecast_path = f"{data_dir}/{forecast_subdir}"
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        price_data = os.getenv("price_data", "all_stocks_price_hist_1d.parquet")
        self.stocks_price_path = f"{data_dir}/{self.data_3rd_party}/{price_data}"


    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model(repo_id=repo_id)
        logger.info(f"trainer params {self.canswim_model.torch_model.trainer_params}")
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data(repo_id=repo_id)
        # load raw data from hf hub
        start_date = pd.Timestamp.now() - BDay(
            n=self.canswim_model.min_samples + self.canswim_model.train_history
        )
        self.canswim_model.load_data(start_date=start_date)
        # prepare timeseries for forecast
        self.canswim_model.prepare_forecast_data(start_date=start_date)


    def initdb(self):
        logger.info(f"Forecast path: {self.forecast_path}")
        tickers_str = "'"+"','".join(self.canswim_model.targets_ticker_list)+"'"
        duckdb.sql(f"CREATE TABLE forecast AS SELECT date, symbol, forecast_start_year, forecast_start_month, forecast_start_day, COLUMNS(\"close_quantile_\d+\.\d+\") FROM read_parquet('{self.forecast_path}/**/*.parquet', hive_partitioning = 1) WHERE symbol in ({tickers_str})")
        duckdb.sql(f"CREATE TABLE close_price AS SELECT Date, Symbol, Close FROM read_parquet('{self.stocks_price_path}') WHERE symbol in ({tickers_str})")
        duckdb.sql('SET enable_external_access = false; ')


def main():

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        CANSWIM Playground for CANSLIM style investors.
        * __NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK!__
        * Model trainer source repo [here](https://github.com/ivelin/canswim). Feedback welcome via github issues.
        """
        )

        canswim_playground = CanswimPlayground()
        canswim_playground.download_model()
        canswim_playground.download_data()
        canswim_playground.initdb()

        with gr.Tab("Charts"):
            charts_tab = ChartTab(canswim_playground.canswim_model, forecast_path=canswim_playground.forecast_path)
        with gr.Tab("Scans"):
            ScanTab(canswim_playground.canswim_model)
        with gr.Tab("Advanced Queries"):
            AdvancedTab(canswim_playground.canswim_model)

        demo.load(
            fn=charts_tab.plot_forecast,
            inputs=[charts_tab.tickerDropdown, charts_tab.lowq],
            outputs=[charts_tab.plotComponent],
            queue=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()
