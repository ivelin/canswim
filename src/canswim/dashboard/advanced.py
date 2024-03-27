"""
Class representing an advanced tab for querying and scanning forecasts.

Attributes:
    canswim_model (CanswimModel): An instance of the CanswimModel class.

Methods:
    __init__(canswim_model): Initializes the AdvancedTab class.
    scan_forecasts(query): Scans the forecasts based on the provided query.

"""

from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb


class AdvancedTab:

    def __init__(self, canswim_model: CanswimModel = None):
        self.canswim_model = canswim_model
        with gr.Row():
            self.queryBox = gr.TextArea(
                value="""SELECT 
                    f.symbol, 
                    min(f.date) as forecast_start_date, 
                    max(c.date) as prior_close_date, 
                    round(arg_max(c.close, c.date), 2) as prior_close_price, 
                    round(min("close_quantile_0.2"), 2) as forecast_close_low, 
                    round(max("close_quantile_0.5"), 2) as forecast_close_high, 
                    round(100*(forecast_close_high / prior_close_price - 1), 2) as reward_percent, 
                    round((forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01),2) as reward_risk,
                    round(max(e.mal_error), 4) as backtest_error
                FROM forecast f, close_price c, backtest_error as e
                WHERE f.symbol = c.symbol and f.symbol = e.symbol
                GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol, e.symbol
                HAVING prior_close_date < forecast_start_date AND forecast_close_high > prior_close_price 
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
        try:
            # only run select queries
            if query.strip().upper().startswith("SELECT"):
                sql_result = duckdb.sql(query)
                logger.debug(f"SQL Result: \n{sql_result}")
                df = sql_result.df()
                return df
        except Exception as e:
            gr.Error("An error occurred while running the query:", e)
            return None
