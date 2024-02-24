from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb
class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None):
        self.canswim_model = canswim_model

        with gr.Row():
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Choose confidence percentage",
            )
            self.rr = gr.Radio(
                choices=[3, 5, 8, 11],
                value=3,
                label="Probabilistic Reward to Risk ratio between price increase and price drop within the forecast period.",
                info="Choose confidence percentage",
            )


        with gr.Row():
            self.scanBtn = gr.Button(value="Scan", variant="primary")

        with gr.Row():
            self.scanResult = gr.Dataframe()

        self.scanBtn.click(
            fn=self.scan_forecasts,
            inputs=[self.lowq, self.rr],
            outputs=[self.scanResult],
            queue=False,
        )

    def scan_forecasts(self, lowq, rr):
        lq = (100 - lowq)/100
        quantile_col = f'close_quantile_{lq}'
        mean_col = 'close_quantile_0.5'
        return duckdb.sql(f"""
            SELECT f.symbol, min(f.date) as forecast_start_date, max(c.date) as prior_close_date, arg_max(c.close, c.date) as prior_close_price, min("{quantile_col}") as forecast_low_quantile, max("{mean_col}") as forecast_mean_quantile
            FROM forecast f, close_price c
            WHERE f.symbol = c.symbol
            GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol
            HAVING prior_close_date < forecast_start_date AND forecast_mean_quantile > prior_close_price AND (forecast_low_quantile > prior_close_price OR (forecast_mean_quantile - prior_close_price)/(prior_close_price-forecast_low_quantile) > {rr})
            """).df()
            # we need a join with the close table that provides actual closing prices prior to forecast start date
            # having (max("close_quantile_0.5") - close_prior_to_forecast) / (min("close_quantile_0.2") - close_prior_to_forecast) > RR
