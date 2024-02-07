from canswim.model import CanswimModel
from darts import TimeSeries
import pandas as pd
import gradio as gr
from darts.models import ExponentialSmoothing
from canswim.hfhub import HFHub
from matplotlib import pyplot

# pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"
# pd.options.plotting.backend = "hvplot"

hfhub = HFHub()
repo_id = "ivelin/canswim"
canswim_model = CanswimModel()


global target
global past_covariates
global future_covariates
global baseline_prediction
global canswim_prediction

# st.set_page_config(page_title="CANSWIM dashboard", page_icon=":dart:", layout="wide")

# """
# # CANSWIM dashboard

# A research tool for CANSLIM style investors.
# """
# with st.expander("What is this?"):
#     """CANSWIM is a research tool for CANSLIM style investors."""


def download_model():
    """Load model from HF Hub"""
    # download model from hf hub
    canswim_model.download_model(repo_id=repo_id) #, hparams_file="hparams.yaml")
    m = canswim_model.torch_model
    print("trainer params", canswim_model.torch_model.trainer_params)
    canswim_model.torch_model.trainer_params["logger"] = False


def download_data(ticker: str = None):
    """Prepare time series for model forecast"""
    hfhub.download_data(repo_id=repo_id)
    # load raw data from hf hub
    canswim_model.load_data(stock_tickers=[ticker])
    # prepare timeseries for forecast
    canswim_model.prepare_forecast_data(start_date=pd.Timestamp("2015-10-21"))

def plot_forecast(ticker: str = None, lowq: int = 0.2):
    lq = lowq/100
    fig, axes = pyplot.subplots(figsize=(20, 12))
    global target
    target.plot(label=f"{ticker} Close actual")
    global baseline_prediction
    baseline_prediction.plot(label=f"{ticker} Close baseline forecast")
    global canswim_prediction
    canswim_prediction.plot(label=f"{ticker} Close CANSWIM forecast", low_quantile=lq, high_quantile=0.95)
    pyplot.legend()
    return fig


def get_forecast(ticker: str = None, lowq: int = 20):
    global target
    target = canswim_model.targets.target_series[ticker]
    global past_covariates
    past_covariates = canswim_model.covariates.past_covariates[ticker]
    global future_covariates
    future_covariates = canswim_model.covariates.future_covariates[ticker]

    print('target start, end: ', target.start_time(), target.end_time())
    print('future covariates start, end: ', future_covariates.start_time(), future_covariates.end_time())

    baseline_model = ExponentialSmoothing()
    baseline_model.fit(target)
    global baseline_prediction
    baseline_prediction = baseline_model.predict(canswim_model.pred_horizon, num_samples=500)

    global canswim_prediction
    canswim_prediction = canswim_model.predict(target=[target], past_covariates=[past_covariates], future_covariates=[future_covariates])[0]

    fig = plot_forecast(lowq=lowq)
    return fig

def pick_ticker(ticker: str = None, lowq: int = 20):
    download_data(ticker)
    return get_forecast(ticker)

with gr.Blocks() as demo:
    gr.Markdown(
    """
    CANSWIM Playground for CANSLIM style investors.
    """)

    download_model()

    plotComponent = gr.Plot()

    with gr.Row():
        ticker = gr.Dropdown(["AAON", "MSFT", "NVDA"], label="Stock Symbol", value="AAON")
        # time = gr.Dropdown(["3 months", "6 months", "9 months", "12 months"], label="Downloads over the last...", value="12 months")
        lowq = gr.Slider(5, 80, value=20, label="Forecast probability low threshold", info="Choose from 5% to 80%")

    ticker.change(fn=pick_ticker, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)
    lowq.change(fn=plot_forecast, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)
    # time.change(get_forecast, [lib, time], plt, queue=False)
    demo.load(pick_ticker, inputs=[ticker, lowq], outputs=[plotComponent], queue=False)

demo.launch()
