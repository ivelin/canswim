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
        sql_result = duckdb.sql(
            f"""--sql
            SELECT 
                f.symbol, 
                min(f.date) as forecast_start_date, 
                arg_max(c.close, c.date) as prior_close_price, 
                min("{low_quantile_col}") as forecast_close_low, 
                max("{mean_col}") as forecast_close_high, 
                100*(forecast_close_high / prior_close_price - 1) as reward_percent, 
                (forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01) as reward_risk,
                max(e.mal_error) as backtest_error,
                max(c.date) as prior_close_date, 
            FROM forecast f, close_price c, backtest_error as e, latest_forecast as lf
            WHERE f.symbol = lf.symbol AND 
                f.symbol = c.symbol AND f.symbol = e.symbol AND c.date < lf.date
            GROUP BY f.symbol, f.forecast_start_year, f.forecast_start_month, f.forecast_start_day, c.symbol, e.symbol, lf.symbol, lf.date
            HAVING forecast_close_high > prior_close_price AND
                make_date(f.forecast_start_year, f.forecast_start_month, f.forecast_start_day) = lf.date AND
                reward_risk> {rr} AND reward_percent >= {reward}                
            """
        )
        logger.debug(f"SQL Result: \n{sql_result}")
        df = sql_result.df()
        dateformat = lambda d: d.strftime("%d %b, %Y")
        df_styler = df.style.format(
            {"prior_close_date": dateformat, "forecast_start_date": dateformat},
            precision=2,
            thousands=",",
            decimal=".",
        )
        # .format(...date columns format ... .format(na_rep='PASS', precision=2, subset=[1, 2]))
        return df_styler
