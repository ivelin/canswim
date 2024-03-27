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
        forecast_subdir = os.getenv("forecast_subdir", "forecast/")
        self.forecast_path = f"{data_dir}/{forecast_subdir}"
        data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        price_data = os.getenv("price_data", "all_stocks_price_hist_1d.parquet")
        self.stocks_price_path = f"{data_dir}/{data_3rd_party}/{price_data}"
        stock_tickers_list = os.getenv("stock_tickers_list", "all_stocks.csv")
        self.stock_tickers_path = f"{data_dir}/{data_3rd_party}/{stock_tickers_list}"

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        logger.info(f"Downloading model from remote repo: {repo_id}")
        self.canswim_model.download_model(repo_id=repo_id)
        if self.canswim_model.torch_model is None:
            self.canswim_model.build()
        logger.info(f"trainer params {self.canswim_model.torch_model.trainer_params}")
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        # download raw data from hf hub
        logger.info(f"Downloading data from remote repo: {repo_id}")
        self.hfhub.download_data(repo_id=repo_id)

    def initdb(self):
        logger.info(f"Forecast path: {self.forecast_path}")
        duckdb.sql(
            f"""
            CREATE VIEW stock_tickers 
            AS SELECT * FROM read_csv('{self.stock_tickers_path}', header=True)
            """
        )
        # df_tickers = duckdb.sql("SELECT * from stock_tickers").df()
        # logger.info(f"stock ticker list:\n {df_tickers}")
        duckdb.sql(
            f"""
            CREATE VIEW forecast 
            AS SELECT date, symbol, forecast_start_year, forecast_start_month, forecast_start_day, COLUMNS(\"close_quantile_\d+\.\d+\") 
            FROM read_parquet('{self.forecast_path}/**/*.parquet', hive_partitioning = 1) as f
            SEMI JOIN stock_tickers
            ON f.symbol = stock_tickers.symbol;
            """
        )
        duckdb.sql(
            f"""--sql
            CREATE TABLE latest_forecast AS
                SELECT symbol, max(make_date(forecast_start_year, forecast_start_month, forecast_start_day)) as date
                FROM forecast
                GROUP BY symbol
                ;
            """
        )
        duckdb.sql(
            f"""--sql
            CREATE VIEW close_price
            AS SELECT Date, Symbol, "{self.canswim_model.target_column}" as Close
            FROM read_parquet('{self.stocks_price_path}') as cp
            SEMI JOIN stock_tickers
            ON cp.symbol = stock_tickers.symbol;
            """
        )
        duckdb.sql(
            f"""--sql
            CREATE VIEW backtest_error 
            AS SELECT f.symbol, mean(abs(log(greatest(f."close_quantile_0.5", 0.01)/cp.Close))) as mal_error
            FROM forecast as f, close_price as cp
            WHERE cp.symbol = f.symbol AND cp.date = f.date
            GROUP BY f.symbol, cp.symbol
            HAVING cp.symbol = f.symbol
            """
        )

        # access protected via read only remote access tokebs
        # restricting access prevents sql views from working
        # duckdb.sql("SET enable_external_access = false; ")


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
            charts_tab = ChartTab(
                canswim_playground.canswim_model,
                forecast_path=canswim_playground.forecast_path,
            )
        with gr.Tab("Scans"):
            ScanTab(canswim_playground.canswim_model)
        with gr.Tab("Advanced Queries"):
            AdvancedTab(canswim_playground.canswim_model)

        demo.load(
            fn=charts_tab.plot_forecast,
            inputs=[charts_tab.tickerDropdown, charts_tab.lowq],
            outputs=[charts_tab.plotComponent, charts_tab.rrTable],
        )

    demo.queue().launch()


if __name__ == "__main__":
    main()
