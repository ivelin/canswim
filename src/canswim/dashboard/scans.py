from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
from canswim.db import scan_forecasts as db_scan_forecasts


class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
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
        df = db_scan_forecasts(self.db_path, lowq=lowq, reward=reward, rr=rr)
        logger.debug(f"df types: {df.dtypes}")
        df_styler = df.style.format(
            precision=2,
            thousands=",",
            decimal=".",
        )
        return df_styler
