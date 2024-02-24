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
                label="Minimum Reward to Risk ratio between worst case relative price drop and expected price increase within forecast period.",
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
            select symbol, forecast_start_year, forecast_start_month, forecast_start_day, min("{quantile_col}") , max("{mean_col}")
            from forecast
            group by symbol, forecast_start_year, forecast_start_month, forecast_start_day
            """).df()
            # we need a join with the close table that provides actual closing prices prior to forecast start date
            # having (max("close_quantile_0.5") - close_prior_to_forecast) / (min("close_quantile_0.2") - close_prior_to_forecast) > RR
