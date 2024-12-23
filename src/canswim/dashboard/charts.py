from loguru import logger
from typing import Union, Optional, Sequence
import matplotlib
from canswim.model import CanswimModel
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import pandas as pd
import duckdb
from pandas.tseries.offsets import BDay

# Note: It appears that gradio Plot ignores the backend plot lib setting
# pd.options.plotting.backend = 'hvplot'


class ChartTab:

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        self.plotComponent = gr.Plot()
        with gr.Row():
            sorted_tickers = sorted(self.get_tickers())
            logger.info(f"Sorted selection tickers: {sorted_tickers}")
            self.tickerDropdown = gr.Dropdown(
                choices=sorted_tickers,
                label="Stock Symbol",
                value=random.sample(sorted_tickers, 1)[0],
            )
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Choose low price confidence percentage",
            )
        with gr.Row():
            self.rrTable = gr.Dataframe(
                label="Reward / Risk metrics (only for uptrending forecasts)"
            )
        self.tickerDropdown.change(
            fn=self.plot_forecast,
            inputs=[self.tickerDropdown, self.lowq],
            outputs=[self.plotComponent, self.rrTable],
        )
        self.lowq.change(
            fn=self.plot_forecast,
            inputs=[self.tickerDropdown, self.lowq],
            outputs=[self.plotComponent, self.rrTable],
        )

    def get_tickers(self):
        logger.info(f"Loading stock tickers from stock_tickers table")
        with duckdb.connect(self.db_path) as db_con:
            tickers_df = db_con.sql(
                "SELECT symbol FROM stock_tickers WHERE symbol IS NOT NULL ORDER BY symbol"
            ).df()
            logger.info(f"Loaded {len(tickers_df)} symbols in total")
            stock_list = list(set(tickers_df["Symbol"]))
            # logger.debug(f"Loaded symbols: {stock_list}")
            stock_list = sorted(stock_list)
            # logger.debug(f"Sorted symbols: {stock_list}")
            return stock_list

    def get_target(self, ticker):
        try:
            target = self.canswim_model.targets.target_series[ticker]
            logger.info(
                f"target {ticker} start, end: {target.start_time()}, {target.end_time()}"
            )
            return target
        except KeyError as e:
            logger.warning(f"No target series for {ticker}")

    def get_past_covariates(self, ticker):
        past_covariates = self.canswim_model.covariates.past_covariates[ticker]
        logger.info(
            f"past covariates start, end: {past_covariates.start_time()}, {past_covariates.end_time()}"
        )
        return past_covariates

    def get_future_covariates(self, ticker):
        future_covariates = self.canswim_model.covariates.future_covariates[ticker]
        logger.info(
            f"future covariates start, end: {future_covariates.start_time()}, {future_covariates.end_time()}"
        )
        return future_covariates

    def get_rr(self, ticker=None, lq=None):
        assert ticker is not None
        assert lq is not None and lq >= 0
        low_quantile_col = f"close_quantile_{lq}"
        # logger.debug(f"lowq_min: {lowq_min}")
        mean_col = "close_quantile_0.5"
        rr = 1.01  # minimum reward/risk ratio
        reward = 1  # minimum reward percent
        with duckdb.connect(self.db_path) as db_con:
            sql_result = db_con.sql(
                f"""--sql
                SELECT 
                    f.symbol, 
                    f.start_date as forecast_start_date, 
                    arg_max(c.close, c.date) as prior_close_price,
                    min("{low_quantile_col}") as forecast_close_low, 
                    max("{mean_col}") as forecast_close_high, 
                    100*(forecast_close_high / prior_close_price - 1) as reward_percent, 
                    (forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01) as reward_risk,
                    max(e.mal_error) as backtest_error,
                    max(c.date) as prior_close_date, 
                FROM forecast f, close_price c, backtest_error as e, latest_forecast as lf
                WHERE f.symbol = $ticker AND f.symbol = lf.symbol AND 
                    f.symbol = c.symbol AND f.symbol = e.symbol AND c.date < lf.date
                GROUP BY f.symbol, f.start_date, c.symbol, e.symbol, lf.symbol, lf.date
                HAVING forecast_close_high > prior_close_price AND
                    f.start_date = lf.date AND
                    reward_risk> $rr AND reward_percent >= $reward
                """,
                params={
                    "ticker": ticker,
                    "rr": rr,
                    "reward": reward,
                },
            )
            logger.debug(f"sqlresult: \n{sql_result}")
            df = sql_result.df()
            logger.info(f"rr table df: {df}")
            df["prior_close_date"] = df["prior_close_date"].dt.strftime("%Y-%m-%d")
            df["forecast_start_date"] = df["forecast_start_date"].dt.strftime(
                "%Y-%m-%d"
            )
            # logger.debug(f"df types: {df.dtypes}")
            # dateformat = lambda d: d.strftime("%d %b, %Y")
            df_styler = df.style.format(
                # {"prior_close_date": dateformat, "forecast_start_date": dateformat},
                precision=2,
                thousands=",",
                decimal=".",
            )
            return df_styler

    def plot_quantiles_df(
        self,
        df: pd.DataFrame = None,
        new_plot: bool = False,
        central_quantile: Union[float, str] = 0.5,
        low_quantile: Optional[float] = 0.05,
        high_quantile: Optional[float] = 0.95,
        default_formatting: bool = True,
        label: Optional[Union[str, Sequence[str]]] = "",
        max_nr_components: int = 10,
        ax: Optional[matplotlib.axes.Axes] = None,
        *args,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the series saved as parquet via dataframe.
        This method recreates the darts.TimeSeries.plot() method, which does not currently support restoring stochastic timeseries after saving.

        This is a wrapper method around :func:`xarray.DataArray.plot()`.

        Parameters
        ----------
        new_plot
            Whether to spawn a new axis to plot on. See also parameter `ax`.
        central_quantile
            The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
            it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
            of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
            median of each component. `central_quantile` can also be set to 'mean'.
        low_quantile
            The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `confidence_low_quantile` is None (default 0.05).
        high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `high_quantile` is None (default 0.95).
        default_formatting
            Whether to use the darts default scheme.
        label
            Can either be a string or list of strings. If a string and the series only has a single component, it is
            used as the label for that component. If a string and the series has multiple components, it is used as
            a prefix for each component name. If a list of strings with length equal to the number of components in
            the series, the labels will be mapped to the components in order.
        max_nr_components
            The maximum number of components of a series to plot. -1 means all components will be plotted.
        ax
            Optionally, an axis to plot on. If `None`, and `new_plot=False`, will use the current axis. If
            `new_plot=True`, will create a new axis.
        alpha
             Optionally, set the line alpha for deterministic series, or the confidence interval alpha for
            probabilistic series.
        args
            some positional arguments for the `plot()` method
        kwargs
            some keyword arguments for the `plot()` method

        Returns
        -------
        matplotlib.axes.Axes
            Either the passed `ax` axis, a newly created one if `new_plot=True`, or the existing one.
        """

        if df is None or len(df) == 0:
            # logger.debug("Dataframe is empty. Nothing to plot.")
            return

        alpha_confidence_intvls = 0.25

        if central_quantile != "mean":
            assert (
                isinstance(central_quantile, float) and 0.0 <= central_quantile <= 1.0
            ), 'central_quantile must be either "mean", or a float between 0 and 1.'

        if high_quantile is not None and low_quantile is not None:
            assert (
                0.0 <= low_quantile <= 1.0 and 0.0 <= high_quantile <= 1.0
            ), "confidence interval low and high quantiles must be between 0 and 1."

        if new_plot:
            fig, ax = plt.subplots()
        else:
            if ax is None:
                ax = plt.gca()

        if not any(lw in kwargs for lw in ["lw", "linewidth"]):
            kwargs["lw"] = 2

        if label is not None:
            assert isinstance(label, str), "The label argument should be a string"
            custom_labels = True
        else:
            custom_labels = False

        if central_quantile == "mean":
            # central_series = df.mean(axis=1)
            central_series = df[f"close_quantile_0.5"]
        else:
            # central_series = df.quantile(q=central_quantile, axis=1)
            central_series = df[f"close_quantile_{central_quantile}"]
            alpha = kwargs["alpha"] if "alpha" in kwargs else None
            if custom_labels:
                label_to_use = label
            else:
                if label == "":
                    label_to_use = "Value"
                else:
                    label_to_use = label
            kwargs["label"] = label_to_use

            if len(central_series) > 1:
                p = ax.plot(
                    df.index.values, central_series.values, "--", *args, **kwargs
                )
            # empty TimeSeries
            elif len(central_series) == 0:
                p = ax.plot(
                    [],
                    [],
                    *args,
                    **kwargs,
                )
                ax.set_xlabel(df.index.name)
            else:
                p = ax.plot(
                    [df.index.min()],
                    central_series.values[0],
                    "o",
                    *args,
                    **kwargs,
                )
            color_used = p[0].get_color() if default_formatting else None

            # Optionally show confidence intervals
            if low_quantile is not None and high_quantile is not None:
                # low_series = df.quantile(q=low_quantile, axis=1)
                # high_series = df.quantile(q=high_quantile, axis=1)
                low_series = df[f"close_quantile_{low_quantile}"]
                high_series = df[f"close_quantile_{high_quantile}"]
                if len(low_series) > 1:
                    ax.fill_between(
                        df.index,
                        low_series,
                        high_series,
                        color=color_used,
                        alpha=(alpha if alpha is not None else alpha_confidence_intvls),
                    )
                else:
                    ax.plot(
                        [df.index.min(), df.index.min()],
                        [low_series.iloc[0], high_series.iloc[0]],
                        "-+",
                        color=color_used,
                        lw=2,
                    )

        ax.legend()
        return ax

    def plot_forecast(self, ticker: str = None, lowq: int = 0.2):
        load_start_date = pd.Timestamp.now() - BDay(
            n=self.canswim_model.min_samples + self.canswim_model.train_history
        )
        self.canswim_model.load_data(stock_tickers=[ticker], start_date=load_start_date)
        # prepare timeseries for forecast
        self.canswim_model.prepare_stock_price_data(start_date=load_start_date)
        fig, axes = plt.subplots(figsize=(20, 12))
        target = self.get_target(ticker)
        # reward risk df
        rr_df = None
        if target is None:
            msg = f"Not enough data available to forecast {ticker}"
            gr.Info(msg)
            plt.text(0.5, 0.5, msg, fontsize="large")
        else:
            plot_start_date = pd.Timestamp.now() - BDay(
                self.canswim_model.train_history
            )
            visible_target = target.drop_before(plot_start_date)
            saved_forecast_df_list = self.get_saved_forecasts(ticker=ticker, start_date=plot_start_date)
            lq = (100 - lowq) / 100
            visible_target.plot(label=f"{ticker} Close actual")
            # logger.debug(f"Plotting saved forecast: {saved_forecast_df_list}")
            if saved_forecast_df_list is not None and len(saved_forecast_df_list) > 0:
                for forecast in saved_forecast_df_list:
                    self.plot_quantiles_df(
                        df=forecast,
                        low_quantile=lq,
                        high_quantile=0.95,
                        label=f"{ticker} Close forecast",
                    )
                # fetch Reward Risk data for latest forecast
                latest_forecast_df = saved_forecast_df_list[-1]
                if latest_forecast_df is not None and not latest_forecast_df.empty:
                    rr_df = self.get_rr(ticker=ticker, lq=lq)
            # Set the locator
            major_locator = mdates.YearLocator()  # every year
            minor_locator = mdates.MonthLocator()  # every month
            # Specify the format - %b gives us Jan, Feb...
            major_fmt = mdates.DateFormatter('"%Y"')
            minor_fmt = mdates.DateFormatter("%b")
            X = plt.gca().xaxis
            X.set_major_locator(major_locator)
            X.set_minor_locator(minor_locator)
            # Specify formatter
            X.set_major_formatter(major_fmt)
            X.set_minor_formatter(minor_fmt)
            # plt.show()
            # plt.grid(gridOn=True, which='major', color='b', linestyle='-')
            plt.grid(which="minor", color="lightgrey", linestyle="--")
            plt.legend()
        return {self.plotComponent: fig, self.rrTable: rr_df}

    def get_saved_forecasts(self, ticker: str = None, start_date = None):
        """Load forecasts from storage to a list of individual forecast series with quantile sampling"""
        # load parquet partition for stock
        logger.info(f"Loading saved forecast for {ticker} starting after: {start_date}")
        with duckdb.connect(self.db_path) as db_con:
            sql_result = db_con.sql(
                f"""--sql
                SELECT *
                FROM forecast as f
                WHERE f.symbol = $ticker AND f.start_date >= $start_date
                ORDER BY f.symbol, f.start_date, f.date
                """,
                params={"ticker": ticker, "start_date": start_date},
            )
            df = sql_result.df()
            # filters = [("symbol", "=", ticker)]
            # df = pd.read_parquet(
            #     self.forecast_path,
            #     filters=filters,
            # )
            logger.debug(f"df columns count: {len(df.columns)}")
            logger.debug(f"df row count: {len(df)}")
            logger.debug(f"df columns: {df.columns}")
            logger.debug(f"df column types: \n{df.dtypes}")
            # logger.debug(f"df row sample: {df}")
            df = df.drop(columns=["symbol"])
            df.sort_index(ascending=True, inplace=True)
            df_list = []
            for d in df["start_date"].unique():
                single_forecast = df.loc[df["start_date"] == d]
                single_forecast = single_forecast.drop(columns=["start_date"])
                single_forecast = single_forecast.set_index("Date")
                single_forecast.index = pd.to_datetime(single_forecast.index)
                if not single_forecast.empty:
                    df_list.append(single_forecast)
                    logger.debug(
                        f"Loaded forecast series with {len(single_forecast)} samples, start date: {d}"  # , sample:\n {single_forecast}"  # , column types:\n{single_forecast.dtypes}"
                    )
            return df_list
