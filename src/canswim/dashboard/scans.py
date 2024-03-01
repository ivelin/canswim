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
                info="Choose low price confidence percentage",
            )
            self.reward = gr.Radio(
                choices=[10, 15, 20, 25],
                value=20,
                label="Minimum probabilistic price gain.",
                info="Choose reward percentage",
            )
            self.rr = gr.Radio(
                choices=[3, 5, 10],
                value=3,
                label="Probabilistic Reward to Risk ratio between price increase and price drop within the forecast period.",
                info="Choose R/R ratio",
            )

        with gr.Row():
            self.scanBtn = gr.Button(value="Scan", variant="primary")

        with gr.Row():
            self.scanResult = gr.Dataframe()

        self.scanBtn.click(
            fn=self.scan_forecasts,
            inputs=[self.lowq, self.reward, self.rr],
            outputs=[self.scanResult],
        )

    def scan_forecasts(self, lowq, reward, rr):
        lq = (100 - lowq) / 100
        low_quantile_col = f"close_quantile_{lq}"
        mean_col = "close_quantile_0.5"
        return duckdb.sql(
            f"""--sql
            SELECT 
                f.symbol, 
                min(f.date) as forecast_start_date, 
                max(c.date) as prior_close_date, 
                round(arg_max(c.close, c.date), 2) as prior_close_price, 
                round(min("{low_quantile_col}"), 2) as forecast_close_low, 
                round(max("{mean_col}"), 2) as forecast_close_high, 
                round(100*(forecast_close_high / prior_close_price - 1), 2) as reward_percent, 
                round((forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01),2) as reward_risk,
                round(max(e.mal_error), 4) as backtest_error
            FROM forecast f, close_price c, backtest_error as e
            WHERE f.symbol = c.symbol and f.symbol = e.symbol
            GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol, e.symbol
            HAVING prior_close_date < forecast_start_date AND forecast_close_high > prior_close_price 
                AND reward_risk> {rr} AND reward_percent >= {reward}
            """
        ).df()
