from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
from canswim.db import (
    list_forecast_start_date_choices,
    scan_forecasts as db_scan_forecasts,
)


class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        # Horizon for open-vs-completed labels (TiDE output_chunk_length when loaded)
        try:
            self._pred_horizon = int(self.canswim_model.pred_horizon)
        except Exception:
            self._pred_horizon = 42

        # Choices filled dynamically via refresh_start_dates() (page load / refresh)
        with gr.Row():
            self.forecastStart = gr.Dropdown(
                choices=[],
                value=None,
                label="Forecast start date (as-of origin)",
                info=(
                    "Queried from the search DB (newest first). Default = most recent. "
                    "Includes the latest run, which may still be an **open-horizon live "
                    "forecast** (not a finished backtest) if today is inside the "
                    "prediction window. Older dates with a fully elapsed horizon are "
                    "**completed backtests**. Scan scores only that origin."
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
                value=5,
                label="Minimum probabilistic price gain (%)",
                info="Median forecast high vs prior close. Lower if the table is empty.",
            )
            self.rr = gr.Radio(
                choices=[1, 1.5, 3, 5, 10],
                value=1,
                label="Minimum reward / risk ratio",
                info="Upside to downside (low-confidence quantile). Lower if empty.",
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
        choices = list_forecast_start_date_choices(
            self.db_path, pred_horizon_bdays=self._pred_horizon
        )
        # Gradio: list of (label, value); value is ISO date
        default = choices[0][1] if choices else None
        logger.info(
            f"Scan start-date picker: {len(choices)} origins; default={default}"
        )
        return gr.Dropdown(choices=choices, value=default)

    def scan_forecasts(self, lowq, reward, rr, forecast_start_date=None):
        df = db_scan_forecasts(
            self.db_path,
            lowq=lowq,
            reward=reward,
            rr=rr,
            forecast_start_date=forecast_start_date,
        )
        n = 0 if df is None else len(df)
        logger.info(
            f"Scan as-of={forecast_start_date} lowq={lowq} reward={reward} rr={rr} "
            f"hits={n}"
        )
        if df is None or df.empty:
            gr.Warning(
                f"No symbols met reward ≥ {reward}% and R/R ≥ {rr} for as-of "
                f"{forecast_start_date or 'latest'}. "
                f"Try lower thresholds (e.g. 5% / 1.0) or another start date. "
                f"Latest open-horizon forecasts are often milder than past backtests."
            )
            return df
        return df.style.format(
            precision=2,
            thousands=",",
            decimal=".",
        )
