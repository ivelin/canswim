from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
from canswim.db import (
    list_forecast_start_dates,
    scan_forecasts as db_scan_forecasts,
)


class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        # Choices filled dynamically via refresh_start_dates() (page load / refresh)
        with gr.Row():
            self.forecastStart = gr.Dropdown(
                choices=[],
                value=None,
                label="Forecast as-of date (backtest origin)",
                info=(
                    "Loaded from the search DB (newest first). "
                    "Default is the most recent forecast start. "
                    "Scan scores stocks for that origin only."
                ),
                allow_custom_value=False,
            )
            self.refreshStartsBtn = gr.Button(
                value="Refresh dates",
                scale=0,
            )
        with gr.Row():
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Choose low price confidence percentage",
            )
            self.reward = gr.Radio(
                choices=[5, 10, 15, 20, 25],
                value=10,
                label="Minimum probabilistic price gain.",
                info="Choose reward percentage",
            )
            self.rr = gr.Radio(
                choices=[1, 1.5, 3, 5, 10],
                value=1.5,
                label="Probabilistic Reward to Risk ratio between price increase and price drop within the forecast period.",
                info="Choose R/R ratio",
            )

        with gr.Row():
            self.scanBtn = gr.Button(value="Scan", variant="primary")

        with gr.Row():
            self.scanResult = gr.Dataframe()

        self.refreshStartsBtn.click(
            fn=self.refresh_start_dates,
            inputs=[],
            outputs=[self.forecastStart],
        )
        self.scanBtn.click(
            fn=self.scan_forecasts,
            inputs=[self.lowq, self.reward, self.rr, self.forecastStart],
            outputs=[self.scanResult],
        )

    def refresh_start_dates(self):
        """Query DB for distinct start dates; default selection = most recent."""
        choices = list_forecast_start_dates(self.db_path)
        default = choices[0] if choices else None
        logger.info(
            f"Scan start-date picker: {len(choices)} origins; default={default}"
        )
        return gr.Dropdown(choices=choices, value=default)

    def scan_forecasts(self, lowq, reward, rr, forecast_start_date=None):
        # If UI has no selection yet, resolve to newest via scan_forecasts(None)
        df = db_scan_forecasts(
            self.db_path,
            lowq=lowq,
            reward=reward,
            rr=rr,
            forecast_start_date=forecast_start_date,
        )
        logger.info(
            f"Scan as-of={forecast_start_date} lowq={lowq} reward={reward} rr={rr} "
            f"hits={0 if df is None else len(df)}"
        )
        logger.debug(f"df types: {df.dtypes if df is not None else None}")
        if df is None or df.empty:
            return df
        return df.style.format(
            precision=2,
            thousands=",",
            decimal=".",
        )
