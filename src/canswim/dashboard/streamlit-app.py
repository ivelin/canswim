from inspect import isclass
import streamlit as st
from darts.datasets import AirPassengersDataset

st.set_page_config(page_title="CANSWIM dashboard", page_icon=":dart:", layout="wide")

"""
# Streamlit + Darts

Exploring the "Time Series Made Easy in Python" library [Darts](https://unit8co.github.io/darts/).
Adding interactive web elements with [Streamlit](https://streamlit.io)
"""
with st.expander("What is this?"):
    """\
A Demo blending Darts and Streamlit, made with :heart: from [Gar's Bar](https://tech.gerardbentley.com/)
"""
"""
## Install

You can install darts using pip or conda

```sh
pip install darts
# OR (recommended for dependencies)
conda install -c conda-forge -c pytorch u8darts-all
```
"""
with st.expander("If that doesn't work..."):
    """\
Install it without all of the models with pip (from [docs](https://unit8co.github.io/darts/#id3)):
```sh
# Install core only (without neural networks, Prophet or AutoARIMA):
pip install u8darts

# Install core + neural networks (PyTorch):
pip install u8darts[torch]

# Install core + Facebook Prophet:
pip install u8darts[prophet]

# Install core + AutoARIMA:
pip install u8darts[pmdarima]
```

Install it without all of the models with conda:

```sh
# Install core only (without neural networks, Prophet or AutoARIMA):
conda install -c conda-forge u8darts

# Install core + neural networks (PyTorch):
conda install -c conda-forge -c pytorch u8darts-torch
```
"""

"## Create a TimeSeries object from a Pandas DataFrame, and split it in train/validation series:"
with st.echo():
    import pandas as pd
    from darts import TimeSeries

    # Create a TimeSeries, specifying the time and value columns
    series = series = AirPassengersDataset().load()

    # Set aside the last 36 months as a validation series
    train, val = series[:-36], series[-36:]

with st.expander("Train Time Series"):
    st.write("Type:", type(train))
    st.write("First 5 values:", train[:5].values())
    st.write("Length:", len(train))
    attributes = [x for x in dir(train) if not x.startswith("_")]
    st.write("Non-private Attributes:", attributes)

with st.expander("Dataframe View"):
    st.dataframe(series.pd_dataframe())

"## Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series' duration:"

with st.echo():
    from darts.models import ExponentialSmoothing

    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1000)

with st.expander("Prediction Time Series"):
    st.write("Type:", type(prediction))
    st.write("First 5 values:", prediction[:5].values())
    st.write("Length:", len(prediction))

"## Plot the median, 5th and 95th percentiles:"

with st.echo():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    st.pyplot(fig)

"## Interact with Training and Plotting:"


with st.echo("below"):
    interactive_fig = plt.figure()
    series.plot()

    st.subheader("Training Controls")
    num_periods = st.slider(
        "Number of validation months",
        min_value=2,
        max_value=len(series) - 24,
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

    train, val = series[:-num_periods], series[-num_periods:]
    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=num_samples)
    prediction.plot(
        label="forecast", low_quantile=low_quantile, high_quantile=high_quantile
    )

    plt.legend()
    st.pyplot(interactive_fig)

"""## Go Wild!\

Use your own csv data that has a well formed time series and plot some forecasts!

Or use one of the example Darts [datasets](https://github.com/unit8co/darts/tree/master/datasets)

(Limited to ExponentialSmoothing with single variable. Resampling time period will only perform summation. For now...)
"""

import darts.datasets as ds

all_datasets = {
    x: ds.__getattribute__(x)
    for x in dir(ds)
    if isclass(ds.__getattribute__(x))
    and x not in ("DatasetLoaderMetadata", "DatasetLoaderCSV")
}
with st.expander("More info on Darts Datasets"):
    for name, dataset in all_datasets.items():
        st.write(f"#### {name}\n\n{dataset.__doc__}")

with st.echo("below"):
    use_example = st.checkbox("Use example dataset")
    if use_example:
        dataset_choice = st.selectbox("Example Dart Dataset", all_datasets, index=5)
        with st.spinner("Fetching Dataset"):
            dataset = all_datasets[dataset_choice]()
            timeseries = dataset.load()
            custom_df = timeseries.pd_dataframe()
            custom_df["Period"] = custom_df.index.to_series()
            custom_df = custom_df[["Period", *custom_df.columns[:-1]]]
    else:
        csv_data = st.file_uploader("New Timeseries csv")
        delimiter = st.text_input(
            "CSV Delimiter",
            value=",",
            max_chars=1,
            help="How your CSV values are separated",
        )

        if csv_data is None:
            st.warning("Upload a CSV to analyze")
            st.stop()

        custom_df = pd.read_csv(csv_data, sep=delimiter)
    with st.expander("Show Raw Data"):
        st.dataframe(custom_df)

    columns = list(custom_df.columns)
    with st.expander("Show all columns"):
        st.write(" | ".join(columns))

    time_col = st.selectbox(
        "Time Column",
        columns,
        help="Name of the column in your csv with time period data",
    )
    value_cols = st.selectbox(
        "Values Column(s)",
        columns,
        1,
        help="Name of column(s) with values to sample and forecast",
    )
    options = {
        "Monthly": ("M", 12),
        "Weekly": ("W", 52),
        "Yearly": ("A", 1),
        "Daily": ("D", 365),
        "Hourly": ("H", 365 * 24),
        "Quarterly": ("Q", 8),
    }
    sampling_period = st.selectbox(
        "Time Series Period",
        options,
        help="How to define samples. Pandas will sum entries between periods to create a well-formed Time Series",
    )

    custom_df[time_col] = pd.to_datetime(custom_df[time_col])
    freq_string, periods_per_year = options[sampling_period]
    custom_df = custom_df.set_index(time_col).resample(freq_string).sum()
    with st.expander("Show Resampled Data"):
        st.write("Number of samples:", len(custom_df))
        st.dataframe(custom_df)

    custom_series = TimeSeries.from_dataframe(custom_df, value_cols=value_cols)
    st.subheader("Custom Training Controls")
    max_periods = len(custom_series) - (2 * periods_per_year)
    default_periods = min(10, max_periods)
    num_periods = st.slider(
        "Number of validation periods",
        key="cust_period",
        min_value=2,
        max_value=max_periods,
        value=default_periods,
        help="How many periods worth of datapoints to exclude from training",
    )
    num_samples = st.number_input(
        "Number of prediction samples",
        key="cust_sample",
        min_value=1,
        max_value=10000,
        value=1000,
        help="Number of times a prediction is sampled for a probabilistic model",
    )

    st.subheader("Custom Plotting Controls")
    low_quantile = st.slider(
        "Lower Percentile",
        key="cust_low",
        min_value=0.01,
        max_value=0.99,
        value=0.05,
        help="The quantile to use for the lower bound of the plotted confidence interval.",
    )
    high_quantile = st.slider(
        "High Percentile",
        key="cust_high",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        help="The quantile to use for the upper bound of the plotted confidence interval.",
    )

    train, val = custom_series[:-num_periods], custom_series[-num_periods:]
    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=num_samples)

    custom_fig = plt.figure()
    custom_series.plot()

    prediction.plot(
        label="forecast", low_quantile=low_quantile, high_quantile=high_quantile
    )

    plt.legend()
    st.pyplot(custom_fig)
