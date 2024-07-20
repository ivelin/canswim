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

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        with gr.Row():
            self.queryBox = gr.TextArea(
                value="""SELECT 
                    f.symbol, 
                    f.start_date as forecast_start_date, 
                    arg_max(c.close, c.date) as prior_close_price, 
                    min("close_quantile_0.2") as forecast_close_low, 
                    max("close_quantile_0.5") as forecast_close_high,
                    100*(forecast_close_high / prior_close_price - 1) as reward_percent,
                    (forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01) as reward_risk,
                    max(e.mal_error) as backtest_error,
                    max(c.date) as prior_close_date,
                FROM forecast f, close_price c, backtest_error as e, latest_forecast as lf
                WHERE f.symbol = lf.symbol AND
                    f.symbol = c.symbol AND f.symbol = e.symbol AND c.date < lf.date
                GROUP BY f.symbol, f.start_date, c.symbol, e.symbol, lf.symbol, lf.date
                HAVING forecast_close_high > prior_close_price AND
                    f.start_date = lf.date AND
                    reward_risk> 3 AND reward_percent >= 20
                    AND forecast_start_date = (select max(date) from latest_forecast)
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
                with duckdb.connect(self.db_path) as db_con:
                    sql_result = db_con.sql(query)
                    logger.info(f"SQL Result: \n{sql_result}")
                    df = sql_result.df()
                    df["prior_close_date"] = df["prior_close_date"].dt.strftime(
                        "%Y-%m-%d"
                    )
                    df["forecast_start_date"] = df["forecast_start_date"].dt.strftime(
                        "%Y-%m-%d"
                    )
                    return df
        except Exception as e:
            gr.Error("An error occurred while running the query:", e)
            return None
