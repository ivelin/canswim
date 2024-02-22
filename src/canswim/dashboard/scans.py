from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb
import os
class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None, forecast_path: str = None):
        self.canswim_model = canswim_model
        self.forecast_path = forecast_path
        tickers_str = "'"+"','".join(self.canswim_model.targets_ticker_list)+"'"
        self.db = duckdb.sql(f"CREATE TABLE forecast AS SELECT * FROM read_parquet('{forecast_path}/**/*.parquet', hive_partitioning = 1) WHERE symbol in ({tickers_str})")
        with gr.Row():
            self.queryBox = gr.TextArea(value="select * from forecast")
            self.scanBtn = gr.Button(value="Scan")
            self.scanResult = gr.Dataframe()

            sorted_tickers = sorted(self.canswim_model.targets_ticker_list)
            logger.info("Dropdown tickers: ", sorted_tickers)
            self.scanBtn.click(
                fn=self.scan_forecasts,
                inputs=[self.queryBox, ],
                outputs=[self.scanResult],
                queue=False,
            )
            self.queryBox.submit(
                fn=self.scan_forecasts,
                inputs=[self.queryBox, ],
                outputs=[self.scanResult],
                queue=False,
            )

    def scan_forecasts(self, query):
        return duckdb.sql(query).df()
