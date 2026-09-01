from loguru import logger
from typing import Union, Optional, Sequence
import matplotlib
from canswim.model import CanswimModel
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import pandas as pd
from pandas.tseries.offsets import BDay
from canswim.db import (
    list_tickers,
    get_reward_risk,
    get_saved_forecasts_as_series,
    get_company_profile,
    format_company_profile_markdown,
    list_forecast_start_dates,
    load_ibd50_symbols,
    FORECAST_MODEL_TIDE,
    FORECAST_MODEL_TIMESFM3,
)
from canswim.dashboard.url_state import CHART_URL_SYNC_JS

# Note: It appears that gradio Plot ignores the backend plot lib setting
# pd.options.plotting.backend = 'hvplot'


class ChartTab:

    def __init__(
        self, canswim_model: CanswimModel = None, db_path=None, timesfm3_ui=False
    ):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        self.timesfm3_ui = bool(timesfm3_ui)
        self.plotComponent = gr.Plot()
        with gr.Row():
            sorted_tickers = sorted(self.get_tickers())
            logger.info(f"Sorted selection tickers: {sorted_tickers}")
            default_ticker = (
                random.sample(sorted_tickers, 1)[0] if sorted_tickers else None
            )
            self.sorted_tickers = sorted_tickers
            self.default_ticker = default_ticker
            self.tickerDropdown = gr.Dropdown(
                choices=sorted_tickers,
                label="Stock Symbol",
                value=default_ticker,
            )
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Choose low price confidence percentage",
            )
        self.modelPicker = None
        self.originPicker = None
        if self.timesfm3_ui:
            origin_choices = ["all"] + list_forecast_start_dates(self.db_path)
            with gr.Row():
                self.modelPicker = gr.Radio(
                    choices=["both", FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3],
                    value="both",
                    label="Timeseries model",
                    info="Side-by-side IBD50 comparison: tide (TiDE) vs timesfm3 (research)",
                )
                self.originPicker = gr.Dropdown(
                    choices=origin_choices,
                    value="all",
                    label="Backtest origin",
                    info="Chart a single monthly/live origin, or all saved starts",
                    allow_custom_value=False,
                    filterable=True,
                )
        self.companyInfo = gr.Markdown(
            value=self._company_md(default_ticker),
            label="Company",
        )
        with gr.Row():
            self.rrTable = gr.Dataframe(
                label="Reward / Risk metrics (only for uptrending forecasts)"
            )
        plot_inputs = [self.tickerDropdown, self.lowq]
        if self.timesfm3_ui and self.modelPicker is not None:
            plot_inputs.extend([self.modelPicker, self.originPicker])
        self.tickerDropdown.change(
            fn=self.plot_forecast,
            inputs=plot_inputs,
            outputs=[self.plotComponent, self.rrTable, self.companyInfo],
        )
        self.lowq.change(
            fn=self.plot_forecast,
            inputs=plot_inputs,
            outputs=[self.plotComponent, self.rrTable, self.companyInfo],
        )
        if self.timesfm3_ui and self.modelPicker is not None:
            self.modelPicker.change(
                fn=self.plot_forecast,
                inputs=plot_inputs,
                outputs=[self.plotComponent, self.rrTable, self.companyInfo],
            )
            self.originPicker.change(
                fn=self.plot_forecast,
                inputs=plot_inputs,
                outputs=[self.plotComponent, self.rrTable, self.companyInfo],
            )
        # Live shareable URL (?ticker=&lowq=) — separate from plot handlers so
        # Gradio js= does not rewrite plot_forecast inputs.
        self.tickerDropdown.change(
            fn=None,
            inputs=[self.tickerDropdown, self.lowq],
            js=CHART_URL_SYNC_JS,
        )
        self.lowq.change(
            fn=None,
            inputs=[self.tickerDropdown, self.lowq],
            js=CHART_URL_SYNC_JS,
        )

    def _company_md(self, ticker: str | None) -> str:
        if not ticker:
            return ""
        try:
            return format_company_profile_markdown(
                get_company_profile(self.db_path, ticker)
            )
        except Exception as e:
            logger.debug(f"company profile blurb: {e}")
            return "_Company profile unavailable._"

    def get_tickers(self):
        all_t = list_tickers(self.db_path)
        if not self.timesfm3_ui:
            return all_t
        ibd = set(load_ibd50_symbols())
        if not ibd:
            return all_t
        preferred = [t for t in all_t if t in ibd]
        # Keep IBD50 names even if not yet in stock_tickers (still chartable
        # from close_price / parquet).
        extra = sorted(ibd - set(preferred))
        return preferred + extra

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

    def get_rr(self, ticker=None, lq=None, model=None):
        assert ticker is not None
        assert lq is not None and lq >= 0
        # Chart plots all backtest starts in range — table must match (not latest-only).
        # Still filter to uptrending rows (label); prior close is as-of each start.
        kwargs = dict(
            symbol=ticker,
            low_quantile=lq,
            latest_only=False,
            uptrending_only=True,
        )
        if model in (FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3):
            kwargs["model"] = model
        df = get_reward_risk(self.db_path, **kwargs)
        logger.info(f"rr table df: {df}")
        if df is None or df.empty:
            return df
        from canswim.dashboard.formatting import format_forecast_metrics_table

        return format_forecast_metrics_table(df)

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

    def plot_forecast(
        self,
        ticker: str = None,
        lowq: int = 0.2,
        model: str = None,
        origin: str = None,
    ):
        # Chart / forecast display only needs forecast-horizon history, not full train min_samples
        # (train min_samples ≈ lookback*2 + horizons; scoped gather keeps ~2y of prices).
        try:
            th = int(self.canswim_model.train_history)
            ph = int(self.canswim_model.pred_horizon)
            have_tide = getattr(self.canswim_model, "torch_model", None) is not None
        except Exception:
            th, ph, have_tide = 252, 42, False
        chart_min_samples = th + ph + ph  # same floor as forecast_only mode
        load_start_date = pd.Timestamp.now() - BDay(n=chart_min_samples + th)
        if have_tide:
            self.canswim_model.load_data(
                stock_tickers=[ticker],
                start_date=load_start_date,
                min_samples=chart_min_samples,
            )
            # prepare timeseries for forecast
            self.canswim_model.prepare_stock_price_data(start_date=load_start_date)
        model = (model or ("both" if self.timesfm3_ui else FORECAST_MODEL_TIDE)).lower()
        origin = origin or "all"
        side_by_side = self.timesfm3_ui and model == "both"
        if side_by_side:
            fig, axes_arr = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
        else:
            fig, axes_arr = plt.subplots(figsize=(20, 12))
            axes_arr = [plt.gca()]
        target = self.get_target(ticker) if have_tide else None
        rr_df = None
        plot_start_date = pd.Timestamp.now() - BDay(th)
        lq = (100 - lowq) / 100
        models_to_plot = (
            [FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3]
            if side_by_side
            else [
                model
                if model in (FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3)
                else FORECAST_MODEL_TIDE
            ]
        )
        close_s = None
        if target is None:
            from canswim.db import get_close_prices

            try:
                close_df = get_close_prices(
                    self.db_path,
                    ticker,
                    start=plot_start_date.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                logger.warning(f"close_price fallback failed for {ticker}: {e}")
                close_df = None
            if close_df is not None and len(close_df) > 0:
                date_col = "date" if "date" in close_df.columns else "Date"
                close_col = "close" if "close" in close_df.columns else "Close"
                close_s = close_df.set_index(date_col)[close_col].sort_index()
                close_s.index = pd.to_datetime(close_s.index)

        for ax, mname in zip(axes_arr, models_to_plot):
            plt.sca(ax)
            if target is not None:
                visible_target = target.drop_before(plot_start_date)
                visible_target.plot(label=f"{ticker} Close actual")
            elif close_s is not None:
                ax.plot(close_s.index, close_s.values, label=f"{ticker} Close actual")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Not enough local price history for {ticker} to chart.",
                    fontsize="large",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            saved = self.get_saved_forecasts(
                ticker=ticker,
                start_date=plot_start_date,
                model=mname if self.timesfm3_ui else None,
                origin=origin,
            )
            if saved:
                for forecast in saved:
                    self.plot_quantiles_df(
                        df=forecast,
                        low_quantile=lq,
                        high_quantile=0.95,
                        label=f"{ticker} {mname} forecast",
                        ax=ax,
                    )
            ax.set_title(f"{ticker} · {mname}")
            ax.legend()
            ax.grid(True, which="major", linestyle="-", alpha=0.3)
            ax.grid(True, which="minor", color="lightgrey", linestyle="--")
            major_locator = mdates.YearLocator()
            minor_locator = mdates.MonthLocator()
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_minor_locator(minor_locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('"%Y"'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

        rr_model = None if side_by_side else (
            model if model in (FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3) else None
        )
        if target is not None or close_s is not None:
            rr_df = self.get_rr(ticker=ticker, lq=lq, model=rr_model)
        company_md = self._company_md(ticker)
        return {
            self.plotComponent: fig,
            self.rrTable: rr_df,
            self.companyInfo: company_md,
        }

    def get_saved_forecasts(
        self, ticker: str = None, start_date=None, model=None, origin=None
    ):
        """Load forecasts from storage to a list of individual forecast series with quantile sampling"""
        start_eq = None
        if origin and str(origin).strip().lower() not in ("", "all", "none"):
            start_eq = str(origin).strip()[:10]
        return get_saved_forecasts_as_series(
            self.db_path,
            ticker=ticker,
            start_date=start_date if start_eq is None else None,
            model=model,
            start_date_eq=start_eq,
        )
