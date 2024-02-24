from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb

class AdvancedTab:

    def __init__(self, canswim_model: CanswimModel = None):
        self.canswim_model = canswim_model
        with gr.Row():
            self.queryBox = gr.TextArea(value="""
                SELECT f.symbol, min(f.date) as forecast_start_date, max(c.date) as prior_close_date, arg_max(c.close, c.date) as prior_close_price, min("close_quantile_0.2") as forecast_low_quantile, max("close_quantile_0.5") as forecast_mean_quantile
                FROM forecast f, close_price c
                WHERE f.symbol = c.symbol
                GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol
                HAVING prior_close_date < forecast_start_date AND forecast_mean_quantile > prior_close_price AND (forecast_low_quantile > prior_close_price OR (forecast_mean_quantile - prior_close_price)/(prior_close_price-forecast_low_quantile) > 3)
            """)

        with gr.Row():
            self.runBtn = gr.Button(value="Run Query", variant="primary")

        with gr.Row():
            self.queryResult = gr.Dataframe()

        self.runBtn.click(
            fn=self.scan_forecasts,
            inputs=[self.queryBox, ],
            outputs=[self.queryResult],
            queue=False,
        )
        self.queryBox.submit(
            fn=self.scan_forecasts,
            inputs=[self.queryBox, ],
            outputs=[self.queryResult],
            queue=False,
        )

    def scan_forecasts(self, query):
        # only run select queries
        if query.strip().upper().startswith('SELECT'):
          return duckdb.sql(query).df()
