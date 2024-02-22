"""CANSWIM Playground. A gradio app intended to be deployed on HF Hub."""

from canswim.model import CanswimModel
import pandas as pd
import gradio as gr
from canswim.hfhub import HFHub
from loguru import logger
from canswim.dashboard.charts import ChartsTab
from pandas.tseries.offsets import BDay

# Note: It appears that gradio Plot ignores the backend plot lib setting
# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

repo_id = "ivelin/canswim"

class CanswimPlayground:

    def __init__(self):
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model(repo_id=repo_id)
        logger.info(f"trainer params {self.canswim_model.torch_model.trainer_params}")
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data(repo_id=repo_id)
        # load raw data from hf hub
        start_date = pd.Timestamp.now() - BDay(
            n=self.canswim_model.min_samples + self.canswim_model.train_history
        )
        self.canswim_model.load_data(start_date=start_date)
        # prepare timeseries for forecast
        self.canswim_model.prepare_forecast_data(start_date=start_date)



def main():


    with gr.Blocks() as demo:
        gr.Markdown(
            """
        CANSWIM Playground for CANSLIM style investors.
        * __NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK!__
        * Model trainer source repo [here](https://github.com/ivelin/canswim). Feedback welcome via github issues.
        """
        )

        canswim_playground = CanswimPlayground()
        canswim_playground.download_model()
        canswim_playground.download_data()

        with gr.Tab("Charts"):
            charts_tab = ChartsTab(canswim_playground.canswim_model)
        with gr.Tab("Scans"):
            pass
        with gr.Tab("Advanced Queries"):
            pass

        demo.load(
            fn=charts_tab.plot_forecast,
            inputs=[charts_tab.tickerDropdown, charts_tab.lowq],
            outputs=[charts_tab.plotComponent],
            queue=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()
