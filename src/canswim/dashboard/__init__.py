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

    def __init__(self, same_data=False):
        self.same_data = same_data
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()
        data_dir = os.getenv("data_dir", "data")
        db_file = os.getenv("db_file", "local.duckdb")
        self.db_path = f"{data_dir}/{db_file}"
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
        with duckdb.connect(self.db_path) as db_con:
            create_new_db = True
            if self.same_data:
                result = db_con.table("stock_tickers").fetchone()
                if result is not None:
                    create_new_db = False
            if not create_new_db:
                logger.info("Reusing search database")
            else:
                logger.info("Creating search optimized database")
                db_con.sql(
                    f"""--sql
                    SET enable_progress_bar = true;        
                    """
                )
                logger.info("Creating stock_tickers table")
                db_con.sql(
                    f"""--sql
                    CREATE OR REPLACE TABLE stock_tickers 
                    AS SELECT * FROM read_csv('{self.stock_tickers_path}', header=True)
                    """
                )
                db_con.table("stock_tickers").show()
                db_con.sql(
                    f"""--sql
                    CREATE UNIQUE INDEX stock_tickers_sym_idx ON stock_tickers (symbol)
                    """
                )
                logger.info(
                    """
                    Creating forecast tables optimized for search. May take a few minutes.
                    Use --help to see all dashboard launch options.
                    """
                )
                db_con.sql(
                    f"""--sql
                    CREATE OR REPLACE TABLE forecast 
                    AS SELECT date, symbol, make_date(forecast_start_year, forecast_start_month, forecast_start_day) as start_date, COLUMNS(\"close_quantile_\d+\.\d+\") 
                    FROM read_parquet('{self.forecast_path}/**/*.parquet', hive_partitioning = 1) as f
                    SEMI JOIN stock_tickers
                    ON f.symbol = stock_tickers.symbol
                    """
                )
                db_con.table("forecast").show()
                db_con.sql(
                    f"""--sql
                    CREATE UNIQUE INDEX forecast_symd_idx
                    ON forecast (symbol, start_date, date)
                    """
                )
                logger.info("Creating latest_forecast table")
                db_con.sql(
                    f"""--sql
                    CREATE OR REPLACE TABLE latest_forecast AS
                        SELECT symbol, max(start_date) as date
                        FROM forecast as f
                        SEMI JOIN stock_tickers
                        ON f.symbol = stock_tickers.symbol                
                        GROUP BY symbol
                    """
                )
                db_con.table("latest_forecast").show()
                db_con.sql(
                    f"""--sql
                    CREATE UNIQUE INDEX latest_forecast_symd_idx
                    ON latest_forecast (symbol, date)
                    """
                )
                logger.info("Creating close_price table")
                db_con.sql(
                    f"""--sql
                    CREATE OR REPLACE TABLE close_price
                    AS SELECT Date, Symbol, "{self.canswim_model.target_column}" as Close
                    FROM read_parquet('{self.stocks_price_path}') as cp
                    SEMI JOIN stock_tickers
                    ON cp.symbol = stock_tickers.symbol;
                    """
                )
                db_con.table("close_price").show()
                db_con.sql(
                    f"""--sql
                    CREATE UNIQUE INDEX close_price_symd_idx
                    ON close_price (symbol, date)
                    """
                )
                logger.info("Creating backtest_error table")
                db_con.sql(
                    f"""--sql
                    CREATE OR REPLACE TABLE backtest_error 
                    AS SELECT f.symbol, mean(abs(log(greatest(f."close_quantile_0.5", 0.01)/cp.Close))) as mal_error
                    FROM forecast as f, close_price as cp
                    WHERE cp.symbol = f.symbol AND cp.date = f.date
                    GROUP BY f.symbol, cp.symbol
                    HAVING cp.symbol = f.symbol
                    """
                )
                db_con.table("backtest_error").show()
                db_con.sql(
                    f"""--sql
                    CREATE UNIQUE INDEX backtest_error_sym_idx
                    ON backtest_error (symbol)
                    """
                )

    def launch(self):
        self.download_model()
        self.download_data()
        self.initdb()
        with gr.Blocks(theme=gr.themes.Soft(), title="CANSWIM") as demo:
            gr.Markdown(
                """
            CANSWIM Playground for CANSLIM style investors.
            * __NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK!__
            * Model trainer source repo [here](https://github.com/ivelin/canswim). Feedback welcome via github issues.
            """
            )
            with gr.Tab("Charts"):
                charts_tab = ChartTab(
                    canswim_model=self.canswim_model,
                    db_path=self.db_path,
                )
            with gr.Tab("Scans"):
                ScanTab(
                    self.canswim_model,
                    db_path=self.db_path,
                )
            with gr.Tab("Advanced Queries"):
                AdvancedTab(
                    self.canswim_model,
                    db_path=self.db_path,
                )

            demo.load(
                fn=charts_tab.plot_forecast,
                inputs=[charts_tab.tickerDropdown, charts_tab.lowq],
                outputs=[charts_tab.plotComponent, charts_tab.rrTable],
            )

        demo.queue().launch()


def main(same_data=False):

    canswim_playground = CanswimPlayground(same_data=same_data)
    canswim_playground.launch()


if __name__ == "__main__":
    main()
