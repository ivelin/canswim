from canswim.model import CanswimModel
import pandas as pd
import gradio as gr
from darts.models import ExponentialSmoothing
from canswim.hfhub import HFHub
from matplotlib import pyplot
import random

# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

repo_id = "ivelin/canswim"

class CanswimPlayground():

    def __init__(self):
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()
        # cache backtest and forecast predictions per ticker
        self.forecast_cache = {}

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model(repo_id=repo_id)
        print("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data(repo_id=repo_id)
        # load raw data from hf hub
        start_date = pd.Timestamp("2019-10-21")
        self.canswim_model.load_data(start_date=start_date)
        # prepare timeseries for forecast
        self.canswim_model.prepare_forecast_data(start_date=start_date)
        global tickers
        tickers = list(self.canswim_model.targets.target_series.keys())


    def plot_forecast(self, ticker: str = None, lowq: int = 0.2):
        lq = lowq/100
        fig, axes = pyplot.subplots(figsize=(20, 12))
        self.target.plot(label=f"{ticker} Close actual")
        self.baseline_forecast.plot(label=f"{ticker} Close baseline forecast")
        self.canswim_forecast.plot(label=f"{ticker} Close CANSWIM forecast", low_quantile=lq, high_quantile=0.95)
        pyplot.legend()
        return fig


    def get_forecast(self, ticker: str = None, lowq: int = 20):
        self.target = self.canswim_model.targets.target_series[ticker]
        self.past_covariates = self.canswim_model.covariates.past_covariates[ticker]
        self.future_covariates = self.canswim_model.covariates.future_covariates[ticker]
        print('target {ticker} start, end: ', self.target.start_time(), self.target.end_time())
        print('future covariates start, end: ', self.future_covariates.start_time(), self.future_covariates.end_time())
        baseline_model = ExponentialSmoothing()
        baseline_model.fit(self.target)
        self.baseline_forecast = baseline_model.predict(self.canswim_model.pred_horizon, num_samples=500)
        cached_canswim_pred = self.forecast_cache.get(ticker)
        if cached_canswim_pred is not None:
            print(f"{ticker} forecast found in cache.")
            self.canswim_forecast = cached_canswim_pred
        else:
            print(f"{ticker} forecast not in cache. Running model predict().")
            self.canswim_forecast = self.canswim_model.predict(target=[self.target], past_covariates=[self.past_covariates], future_covariates=[self.future_covariates])[0]
            self.forecast_cache[ticker] = self.canswim_forecast
        fig = self.plot_forecast(lowq=lowq)
        return fig

    def pick_ticker(self, ticker: str = None, lowq: int = 20):
        self.download_data(ticker)
        return self.get_forecast(ticker)

with gr.Blocks() as demo:
    gr.Markdown(
    """
    CANSWIM Playground for CANSLIM style investors.
    * __NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK!__
    * Model trainer source repo [here](https://github.com/ivelin/canswim). Feedback welcome via github issues.
    """)

    canswim_playground = CanswimPlayground()
    canswim_playground.download_model()
    canswim_playground.download_data()

    plotComponent = gr.Plot()

    with gr.Row():
        ticker = gr.Dropdown(sorted(tickers), label="Stock Symbol", value=random.sample(tickers, 1)[0])
        ## time = gr.Dropdown(["3 months", "6 months", "9 months", "12 months"], label="Downloads over the last...", value="12 months")
        lowq = gr.Slider(5, 80, value=20, label="Forecast probability low threshold", info="Choose from 5% to 80%")

    ticker.change(fn=canswim_playground.get_forecast, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)
    lowq.change(fn=canswim_playground.plot_forecast, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)
    ## time.change(get_forecast, [lib, time], plt, queue=False)
    demo.load(fn=canswim_playground.get_forecast, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)

demo.launch()
