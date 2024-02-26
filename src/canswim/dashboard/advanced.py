from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb


class AdvancedTab:

    def __init__(self, canswim_model: CanswimModel = None):
        self.canswim_model = canswim_model
        with gr.Row():
            self.queryBox = gr.TextArea(
                value="""
                SELECT 
                    f.symbol, 
                    min(f.date) as forecast_start_date, 
                    max(c.date) as prior_close_date, 
                    arg_max(c.close, c.date) as prior_close_price, 
                    min("close_quantile_0.2") as forecast_low_quantile, 
                    max("close_quantile_0.5") as forecast_mean_quantile, 
                    ROUND(100*(forecast_mean_quantile - prior_close_price) / prior_close_price) as reward_percent, 
                    ROUND((forecast_mean_quantile - prior_close_price)/GREATEST(prior_close_price-forecast_low_quantile, 0.01),2) as reward_risk 
                FROM forecast f, close_price c
                WHERE f.symbol = c.symbol
                GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol
                HAVING prior_close_date < forecast_start_date AND forecast_mean_quantile > prior_close_price 
                    AND reward_risk> 3 AND reward_percent >= 20
            """
            )

        with gr.Row():
            self.runBtn = gr.Button(value="Run Query", variant="primary")

        with gr.Row():
            self.queryResult = gr.Dataframe()

        self.runBtn.click(
            fn=self.scan_forecasts,
            inputs=[
                self.queryBox,
            ],
            outputs=[self.queryResult],
        )
        self.queryBox.submit(
            fn=self.scan_forecasts,
            inputs=[
                self.queryBox,
            ],
            outputs=[self.queryResult],
        )

    def scan_forecasts(self, query):
        # only run select queries
        if query.strip().upper().startswith("SELECT"):
            return duckdb.sql(query).df()
