"""CANSWIM Playground. A gradio app intended to be deployed on HF Hub."""
from canswim.model import CanswimModel
import pandas as pd
import gradio as gr
from darts.models import ExponentialSmoothing
from darts import TimeSeries
from canswim.hfhub import HFHub
from matplotlib import pyplot
import random
from pandas.tseries.offsets import BDay

# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

repo_id = "ivelin/canswim"

class CanswimPlayground():


    def get_target(self, ticker):
        target = self.canswim_model.targets.target_series[ticker]
        print('target {ticker} start, end: ', target.start_time(), target.end_time())
        return target

    def get_past_covariates(self, ticker):
        past_covariates = self.canswim_model.covariates.past_covariates[ticker]
        print('past covariates start, end: ', past_covariates.start_time(), past_covariates.end_time())
        return past_covariates

    def get_future_covariates(self, ticker):
        future_covariates = self.canswim_model.covariates.future_covariates[ticker]
        print('future covariates start, end: ', future_covariates.start_time(), future_covariates.end_time())
        return future_covariates


    def __init__(self):
        self.canswim_model = CanswimModel()
        self.hfhub = HFHub()
        # cache backtest and forecast predictions per ticker
        self.forecast_cache = {}
        self.backtest_cache = {}

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
        self.tickers = list(self.canswim_model.targets.target_series.keys())


    def plot_forecast(self, ticker: str = None, lowq: int = 0.2):
        target = self.get_target(ticker)
        baseline_forecast, canswim_forecast = self.get_forecast(ticker)
        backtest_forecasts = self.backtest(ticker)
        lq = lowq/100
        fig, axes = pyplot.subplots(figsize=(20, 12))
        target.plot(label=f"{ticker} Close actual")
        baseline_forecast.plot(label=f"{ticker} Close baseline forecast")
        canswim_forecast.plot(label=f"{ticker} Close CANSWIM forecast", low_quantile=lq, high_quantile=0.95)
        for b in backtest_forecasts:
            b.plot(label=f"{ticker} Close CANSWIM backtest", low_quantile=lq, high_quantile=0.95)
        pyplot.legend()
        return fig


    def get_forecast(self, ticker: str = None):
        target = self.get_target(ticker)
        past_covariates = self.get_past_covariates(ticker)
        future_covariates = self.get_future_covariates(ticker)
        baseline_model = ExponentialSmoothing()
        baseline_model.fit(target)
        baseline_forecast = baseline_model.predict(self.canswim_model.pred_horizon, num_samples=500)
        cached_canswim_pred = self.forecast_cache.get(ticker)
        if cached_canswim_pred is not None:
            print(f"{ticker} forecast found in cache.")
            canswim_forecast = cached_canswim_pred
        else:
            print(f"{ticker} forecast not in cache. Running model predict().")
            canswim_forecast = self.canswim_model.predict(target=[target], past_covariates=[past_covariates], future_covariates=[future_covariates])[0]
            self.forecast_cache[ticker] = canswim_forecast
        print(f"{ticker} get_forecast() finished.")
        return baseline_forecast, canswim_forecast

    def backtest(self, ticker: str = None):
        cached_backtest = self.backtest_cache.get(ticker)
        if cached_backtest is not None:
            print(f"{ticker} backtest found in cache.")
            backtest_forecasts = cached_backtest
        else:
            print(f"{ticker} backtest not in cache. Running model predict().")
            target = self.get_target(ticker)
            past_covariates = self.get_past_covariates(ticker)
            future_covariates = self.get_future_covariates(ticker)
            end_date = target.end_time()
            earnings_df = self.canswim_model.covariates.earnings_loaded_df
            print("earnings_df.columns", earnings_df.columns)
            mask = (earnings_df.index.get_level_values('Symbol') == ticker) & (earnings_df.index.get_level_values('Date') < end_date - BDay(n=10))
            earnings_dates = earnings_df.loc[mask]
            print(f"{ticker} earnings dates: {earnings_dates}")
            earnings_dates_unique = earnings_dates.index.get_level_values('Date').unique()
            assert len(earnings_dates_unique) >= 2
            target1 = target.drop_after(earnings_dates_unique[-1])
            target2 = target.drop_after(earnings_dates_unique[-2])
            backtest_forecasts = self.canswim_model.predict(target=[target1, target2], past_covariates=[past_covariates, past_covariates], future_covariates=[future_covariates, future_covariates])
            print(f"{ticker} backtest finished.\n", backtest_forecasts)
            self.backtest_cache[ticker] = backtest_forecasts
        return backtest_forecasts


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
        sorted_tickers = sorted(canswim_playground.tickers)
        print("Dropdown tickers: ", sorted_tickers)
        tickerDropdown = gr.Dropdown(choices=sorted_tickers, label="Stock Symbol", value=random.sample(sorted_tickers, 1)[0])
        ## time = gr.Dropdown(["3 months", "6 months", "9 months", "12 months"], label="Downloads over the last...", value="12 months")
        lowq = gr.Slider(5, 80, value=20, label="Forecast probability low threshold", info="Choose from 5% to 80%")

    tickerDropdown.change(fn=canswim_playground.plot_forecast, inputs=[tickerDropdown, lowq], outputs=[plotComponent], queue=False)
    lowq.change(fn=canswim_playground.plot_forecast, inputs=[tickerDropdown, lowq], outputs=[plotComponent], queue=False)
    ## time.change(get_forecast, [lib, time], plt, queue=False)
    demo.load(fn=canswim_playground.plot_forecast, inputs=[tickerDropdown, lowq], outputs=[plotComponent], queue=False)

demo.launch()
