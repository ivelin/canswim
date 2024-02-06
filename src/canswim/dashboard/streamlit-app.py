from inspect import isclass
import streamlit as st
from canswim.model import CanswimModel
from darts.models import TiDEModel
from darts import TimeSeries
from canswim.hfhub import HFHub
import pandas as pd

canswim_model = CanswimModel()

st.set_page_config(page_title="CANSWIM dashboard", page_icon=":dart:", layout="wide")

"""
# CANSWIM dashboard

A research tool for CANSLIM style investors.
"""
with st.expander("What is this?"):
    """CANSWIM is a research tool for CANSLIM style investors."""


"## Load model from HF Hub:"
with st.echo():
    hfhub = HFHub()
    repo_id = "ivelin/canswim"
    # download model from hf hub
    canswim_model.download_model(repo_id=repo_id)



ticker_picker = 'AAON'

"## Prepare time series for model forecast"
with st.echo():
    def load_data(ticker: str = None):
        hfhub.download_data(repo_id=repo_id)
        # load raw data from hf hub
        canswim_model.load_data(stock_tickers=[ticker])
        # prepare timeseries for forecast
        canswim_model.prepare_forecast_data(start_date=pd.Timestamp("2015-10-21"))

    load_data(ticker_picker)


def show_series(series, label):
    with st.expander(label):
        st.write("Type:", type(series))
        st.write("First 5 values:", series[:5].values())
        st.write("Length:", len(series))
        attributes = [x for x in series.columns]
        st.write("Components: ", attributes)


target: TimeSeries = canswim_model.targets.target_series[ticker_picker]
show_series(target, "Forecast Target Time Series")

past_covariates = canswim_model.covariates.past_covariates[ticker_picker]

show_series(past_covariates, "Past Covariates")

future_covariates = canswim_model.covariates.future_covariates[ticker_picker]

show_series(past_covariates, "Future Covariates")


with st.expander("Target Dataframe View"):
    st.dataframe(target.pd_dataframe())

"## Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series' duration:"

with st.echo():
    from darts.models import ExponentialSmoothing

    model = ExponentialSmoothing()
    model.fit(target)
    prediction = model.predict(canswim_model.pred_horizon, num_samples=500)

with st.expander("Prediction Time Series"):
    st.write("Type:", type(prediction))
    st.write("First 5 values:", prediction[:5].values())
    st.write("Length:", len(prediction))

"## Plot the median, 5th and 95th percentiles:"

with st.echo():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    target.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    st.pyplot(fig)

"## Interact with Training and Plotting:"


with st.echo("below"):
    interactive_fig = plt.figure()
    target.plot()

    st.subheader("Training Controls")
    num_periods = st.slider(
        "Number of validation months",
        min_value=2,
        max_value=len(target) - 24,
        value=36,
        help="How many months worth of datapoints to exclude from training",
    )
    num_samples = st.number_input(
        "Number of prediction samples",
        min_value=1,
        max_value=10000,
        value=1000,
        help="Number of times a prediction is sampled for a probabilistic model",
    )
    st.subheader("Plotting Controls")
    low_quantile = st.slider(
        "Lower Percentile",
        min_value=0.01,
        max_value=0.99,
        value=0.05,
        help="The quantile to use for the lower bound of the plotted confidence interval.",
    )
    high_quantile = st.slider(
        "High Percentile",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        help="The quantile to use for the upper bound of the plotted confidence interval.",
    )

    train, val = target[:-num_periods], target[-num_periods:]
    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=num_samples)
    prediction.plot(
        label="forecast", low_quantile=low_quantile, high_quantile=high_quantile
    )

    plt.legend()
    st.pyplot(interactive_fig)
