"""CANSWIM Playground. A gradio app intended to be deployed on HF Hub."""

from canswim.model import CanswimModel
import gradio as gr
from canswim.hfhub import HFHub
from loguru import logger
from canswim.dashboard.charts import ChartTab
from canswim.dashboard.scans import ScanTab
from canswim.dashboard.advanced import AdvancedTab
from canswim.dashboard.run_tab import RunTab
from canswim.db import DataPaths, init_search_db

# Note: It appears that gradio Plot ignores the backend plot lib setting
# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

repo_id = "ivelin/canswim"


class CanswimPlayground:

    def __init__(self, same_data=False):
        self.same_data = same_data
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()
        paths = DataPaths.from_env()
        self.db_path = paths.db_path
        self.forecast_path = paths.forecast_path
        self.stocks_price_path = paths.stocks_price_path
        self.stock_tickers_path = paths.stock_tickers_path

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        logger.info(f"Downloading model from remote repo: {repo_id}")
        self.canswim_model.download_model(repo_id=repo_id)
        if self.canswim_model.torch_model is None:
            self.canswim_model.build()
        logger.info(f"trainer params {self.canswim_model.torch_model.trainer_params}")
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        # download raw data from hf hub
        logger.info(f"Downloading data from remote repo: {repo_id}")
        self.hfhub.download_data(repo_id=repo_id)

    def initdb(self):
        init_search_db(
            self.db_path,
            same_data=self.same_data,
            target_column=self.canswim_model.target_column,
            stock_tickers_path=self.stock_tickers_path,
            forecast_path=self.forecast_path,
            stocks_price_path=self.stocks_price_path,
        )

    def launch(self):
        self.download_model()
        self.download_data()
        self.initdb()
        with gr.Blocks(theme=gr.themes.Soft(), title="CANSWIM") as demo:
            gr.Markdown(
                """
            CANSWIM Playground for CANSLIM style investors.
            * __NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK!__
            * Model trainer source repo [here](https://github.com/ivelin/canswim). Feedback welcome via github issues.
            """
            )
            with gr.Tab("Charts"):
                charts_tab = ChartTab(
                    canswim_model=self.canswim_model,
                    db_path=self.db_path,
                )
            with gr.Tab("Scans"):
                scans_tab = ScanTab(
                    self.canswim_model,
                    db_path=self.db_path,
                )
            with gr.Tab("Run"):
                RunTab(
                    self.canswim_model,
                    db_path=self.db_path,
                )
            with gr.Tab("Advanced Queries"):
                AdvancedTab(
                    self.canswim_model,
                    db_path=self.db_path,
                )

            def _on_load(ticker, lowq):
                """Page load: chart + scan dates + auto-scan with defaults."""
                chart_out = charts_tab.plot_forecast(ticker, lowq)
                start_dd, start_iso, scan_status, scan_table = (
                    scans_tab.initial_scan()
                )
                if isinstance(chart_out, dict):
                    plot_val = chart_out.get(charts_tab.plotComponent)
                    rr_val = chart_out.get(charts_tab.rrTable)
                    return (
                        plot_val,
                        rr_val,
                        start_dd,
                        start_iso,
                        scan_status,
                        scan_table,
                    )
                return (
                    chart_out[0],
                    chart_out[1],
                    start_dd,
                    start_iso,
                    scan_status,
                    scan_table,
                )

            demo.load(
                fn=_on_load,
                inputs=[charts_tab.tickerDropdown, charts_tab.lowq],
                outputs=[
                    charts_tab.plotComponent,
                    charts_tab.rrTable,
                    scans_tab.forecastStart,
                    scans_tab.selectedStart,
                    scans_tab.scanStatus,
                    scans_tab.scanResult,
                ],
            )

        demo.queue().launch()


def main(same_data=False):

    canswim_playground = CanswimPlayground(same_data=same_data)
    canswim_playground.launch()


if __name__ == "__main__":
    main()
