"""
Forecast stock price movement and upload results to HF Hub
"""

import glob
import os
from datetime import datetime, timedelta
from typing import List

import duckdb
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger
from pandas.tseries.offsets import BDay

from canswim import constants
from canswim.hfhub import HFHub
from canswim.model import CanswimModel


def validate_forecast_dataframe(
    forecast_df: pd.DataFrame,
    *,
    expected_horizon: int | None = None,
    quantiles: list[float] | None = None,
    drop_bad_symbols: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Sanity-check forecast rows before they hit parquet / search DB (issue #32).

    Checks:
    - required partition / quantile columns present
    - no empty frame
    - unique (symbol, forecast date) rows
    - median quantile finite and positive
    - quantile columns non-decreasing across configured quantiles (per row)
    - optional per-symbol row count near ``expected_horizon``

    Returns
    -------
    cleaned_df, errors
        ``cleaned_df`` may drop entire symbols that fail hard checks when
        ``drop_bad_symbols`` is True. ``errors`` is a list of human-readable issues.
    """
    errors: list[str] = []
    if forecast_df is None or forecast_df.empty:
        return forecast_df if forecast_df is not None else pd.DataFrame(), [
            "forecast dataframe is empty"
        ]

    qlist = list(quantiles if quantiles is not None else constants.quantiles)
    qcols = [f"close_quantile_{q}" for q in qlist]
    required = ["symbol", "forecast_start_year", "forecast_start_month", "forecast_start_day"]
    missing = [c for c in required + qcols if c not in forecast_df.columns]
    if missing:
        return forecast_df, [f"missing required columns: {missing}"]

    df = forecast_df.copy()
    # Normalize time index to a column for uniqueness
    if not isinstance(df.index, pd.RangeIndex) or df.index.name is not None:
        time_col = df.index.name or "time"
        if time_col not in df.columns:
            df = df.reset_index()
            # reset may name index "index" or "time"
            if "index" in df.columns and time_col not in df.columns:
                df = df.rename(columns={"index": "time"})
                time_col = "time"
            elif df.columns[0] != "symbol" and time_col not in df.columns:
                # first column often the former index
                time_col = str(df.columns[0])
        else:
            time_col = "time"
    else:
        df = df.reset_index()
        time_col = "time" if "time" in df.columns else str(df.columns[0])

    if time_col not in df.columns:
        # last resort: use positional date from index values already reset
        for cand in ("time", "Date", "date", "level_0"):
            if cand in df.columns:
                time_col = cand
                break
    if time_col not in df.columns:
        return forecast_df, ["could not identify forecast date/time column"]

    df[time_col] = pd.to_datetime(df[time_col]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Uniqueness
    dup_mask = df.duplicated(subset=["symbol", time_col], keep=False)
    if dup_mask.any():
        bad_syms = sorted(df.loc[dup_mask, "symbol"].unique().tolist())
        errors.append(
            f"duplicate (symbol, date) rows for: {', '.join(bad_syms)}"
        )
        if drop_bad_symbols:
            df = df[~df["symbol"].isin(bad_syms)]

    # Drop rows with non-finite median
    med_col = "close_quantile_0.5"
    if med_col in df.columns:
        bad_med = ~np.isfinite(df[med_col].astype(float)) | (df[med_col].astype(float) <= 0)
        if bad_med.any():
            bad_syms = sorted(df.loc[bad_med, "symbol"].unique().tolist())
            errors.append(
                f"non-finite or non-positive median quantile for: {', '.join(bad_syms)}"
            )
            if drop_bad_symbols:
                df = df[~df["symbol"].isin(bad_syms)]

    # Quantile monotonicity (weak: allow tiny float noise)
    present_q = [c for c in qcols if c in df.columns]
    if len(present_q) >= 2 and not df.empty:
        arr = df[present_q].astype(float).to_numpy()
        # each row: diffs >= -eps
        diffs = np.diff(arr, axis=1)
        row_bad = (diffs < -1e-6).any(axis=1)
        if row_bad.any():
            bad_syms = sorted(df.loc[row_bad, "symbol"].unique().tolist())
            errors.append(
                f"quantile order violated (expected non-decreasing) for: "
                f"{', '.join(bad_syms)}"
            )
            if drop_bad_symbols:
                df = df[~df["symbol"].isin(bad_syms)]

    # Horizon length check
    if expected_horizon is not None and expected_horizon > 0 and not df.empty:
        counts = df.groupby("symbol", sort=False).size()
        for sym, n in counts.items():
            if abs(int(n) - int(expected_horizon)) > 2:
                errors.append(
                    f"{sym}: forecast rows={n}, expected ~{expected_horizon}"
                )
                if drop_bad_symbols:
                    df = df[df["symbol"] != sym]

    if df.empty and not forecast_df.empty:
        errors.append("all symbols failed forecast sanity checks")

    # Keep only partition + quantile columns (+ date) for hive parquet compatibility
    keep = (
        required
        + qcols
        + [c for c in ("date", "time", time_col) if c in df.columns]
    )
    keep = list(dict.fromkeys(keep))  # stable unique
    df = df[[c for c in keep if c in df.columns]]

    # Prefer column name "date" (production search-DB / sync path)
    if time_col in df.columns and time_col != "date":
        df = df.rename(columns={time_col: "date"})
        time_col = "date"
    if "date" in df.columns:
        df = df.set_index("date")
        df.index.name = "date"

    return df, errors


class CanswimForecaster:
    def __init__(self):
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        self.stock_tickers_list = os.getenv("stock_tickers_list", "all_stocks.csv")
        logger.info(f"Stocks train list: {self.stock_tickers_list}")
        self.n_stocks = int(os.getenv("n_stocks", 50))
        logger.info(f"n_stocks: {self.n_stocks}")
        self.forecast_subdir = os.getenv("forecast_subdir", "forecast/")
        logger.info(f"Forecast data path: {self.forecast_subdir}")
        self.canswim_model = CanswimModel(forecast_only=True)
        self.hfhub = HFHub()

    def download_model(self):
        """Load model from HF Hub"""
        # download model from hf hub
        self.canswim_model.download_model()
        logger.info("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def load_model(self):
        """Load model from local storage"""
        self.canswim_model.load()
        logger.info("trainer params", self.canswim_model.torch_model.trainer_params)
        self.canswim_model.torch_model.trainer_params["logger"] = False

    def download_data(self):
        """Prepare time series for model forecast"""
        self.hfhub.download_data()
        # load raw data from hf hub
        self.start_date = pd.Timestamp.now() - BDay(
            # min_samples is not sufficient as it does not account for all market days without price data
            # such as holidays. Also for some sparse data sets such as analyst estimates, there are periods of
            # 2 or more years without data.
            # Add a sufficiently big sample pad to include all market off days.
            n=self.canswim_model.min_samples
            * 2
        )

    @staticmethod
    def _truncate_series_before(series, asof: pd.Timestamp):
        """Keep only timestamps strictly before ``asof`` (no as-of leakage)."""
        from canswim.eligibility import timeseries_from_observed_df

        asof = pd.Timestamp(asof).tz_localize(None).normalize()
        df = series.pd_dataframe()
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
        df = df[df.index < asof]
        if df.empty:
            raise ValueError(f"no samples strictly before {asof.date()}")
        return timeseries_from_observed_df(df)

    def get_forecast(self, forecast_start_date: pd.Timestamp = None):
        logger.info("Forecast start. Calling model predict().")
        forecasted_tickers = []
        target_sliced_list = []
        past_cov_list = []
        future_cov_list = []
        tickers_list = self.canswim_model.targets_ticker_list
        # trim end of targets (and past covs) for historical / as-of forecast starts
        if forecast_start_date is not None:
            fsd = pd.Timestamp(forecast_start_date).tz_localize(None).normalize()
            logger.info(
                f"As-of forecast start date {fsd.date()}: truncating targets and "
                f"past covariates to timestamps < start (no leakage)"
            )
            for i, ts in enumerate(self.canswim_model.targets_list):
                try:
                    logger.debug(
                        f"Target {tickers_list[i]} start date, end date, sample count: "
                        f"{ts.start_time()}, {ts.end_time()}, {len(ts)}"
                    )
                    # Latest start allowed = next open market day after last ground-truth bar
                    cutoff_forecast_start_date = get_next_open_market_day(
                        after_date=ts.end_time()
                    )
                    if fsd > cutoff_forecast_start_date:
                        # Pull start back to the latest legal origin for this series
                        # (common when live week default is after cov-aligned end)
                        logger.info(
                            f"{tickers_list[i]}: forecast start {fsd.date()} is after "
                            f"next open after last bar ({cutoff_forecast_start_date}; "
                            f"series ends {ts.end_time()}). Using {cutoff_forecast_start_date}."
                        )
                        fsd = pd.Timestamp(cutoff_forecast_start_date).normalize()

                    if fsd == pd.Timestamp(cutoff_forecast_start_date).normalize():
                        # Forecast from end of available history (no truncation)
                        target_sliced = ts
                        past_sliced = self.canswim_model.past_cov_list[i]
                    else:
                        # Historical backtest: history strictly before start
                        target_sliced = self._truncate_series_before(ts, fsd)
                        past_sliced = self._truncate_series_before(
                            self.canswim_model.past_cov_list[i], fsd
                        )
                        # Align past cov end to target end
                        if past_sliced.end_time() > target_sliced.end_time():
                            past_sliced = self._truncate_series_before(
                                past_sliced,
                                target_sliced.end_time() + pd.Timedelta(days=1),
                            )

                    n_hist = len(target_sliced)
                    if n_hist < self.canswim_model.min_samples:
                        logger.info(
                            f"Skipping {tickers_list[i]}: only {n_hist} pre-start samples "
                            f"(need >= {self.canswim_model.min_samples}); "
                            f"as-of {fsd.date()}, hist ends {target_sliced.end_time().date()}."
                        )
                        continue

                    forecasted_tickers.append(tickers_list[i])
                    target_sliced_list.append(target_sliced)
                    past_cov_list.append(past_sliced)
                    # Future covs may extend through the prediction horizon (known futures)
                    future_cov_list.append(self.canswim_model.future_cov_list[i])
                    logger.info(
                        f"Eligible {tickers_list[i]}: hist_end={target_sliced.end_time().date()}, "
                        f"n={n_hist}, forecast_start={fsd.date()}"
                    )
                except (ValueError, KeyError, AssertionError) as e:
                    logger.warning(
                        f"Skipping {tickers_list[i]} for forecast start date "
                        f"{forecast_start_date} due to error: {type(e)}: {e}"
                    )
        if len(target_sliced_list) > 0:
            canswim_forecast = self.canswim_model.predict(
                target=target_sliced_list,
                past_covariates=past_cov_list,
                future_covariates=future_cov_list,
            )
            logger.info("Forecast finished.")
            forecasts = dict(zip(forecasted_tickers, canswim_forecast))
            return forecasts
        else:
            logger.warning(
                "No stocks have enough data in this batch. Skipping forecast."
            )
            return None

    def _get_stocks_without_forecast(self, stocks_df=None, forecast_start_date=None):
        if forecast_start_date is not None:
            dt = pd.Timestamp(forecast_start_date)
        else:
            dt = pd.Timestamp.now()
        # align date to closest business day
        # leave as is if dt is a business day,
        # otherwise move forward to next business day
        bd = dt + 0 * BDay()
        y = bd.year
        m = bd.month
        d = bd.day
        # logger.debug(f"Forecast start date, year, month, day: {bd}, {y}, {m}, {d}")
        # logger.debug(f"len(stocks_df): {len(stocks_df)}")
        # logger.debug(f"stocks_df: {stocks_df}")
        forecast_glob = f"{self.data_dir}/{self.forecast_subdir}**/*.parquet"
        # Empty/missing forecast tree: all symbols need a forecast (do not crash DuckDB)
        if not glob.glob(forecast_glob, recursive=True):
            logger.info(
                f"No existing forecast parquet under {self.data_dir}/{self.forecast_subdir}; "
                "all listed symbols are candidates."
            )
            stocks_without_forecast = sorted(list(set(stocks_df["Symbol"])))
            return stocks_without_forecast
        try:
            df = duckdb.sql(
                f"""--sql
                CREATE OR REPLACE TABLE stock_group AS SELECT Symbol from stocks_df;
                SELECT symbol, count(*), forecast_start_year, forecast_start_month, forecast_start_day
                FROM read_parquet('{forecast_glob}', hive_partitioning = 1) as f
                SEMI JOIN stock_group
                ON f.symbol = stock_group.symbol
                GROUP BY f.symbol, forecast_start_year, forecast_start_month, forecast_start_day
                HAVING 
                    forecast_start_year={y} AND
                    forecast_start_month={m} AND
                    forecast_start_day={d} AND
                    count(*) >= {self.canswim_model.pred_horizon}
                """
            ).df()
        except duckdb.IOException as e:
            logger.warning(
                f"Could not read existing forecasts ({e}); treating all as candidates."
            )
            return sorted(list(set(stocks_df["Symbol"])))
        # logger.debug(f"sql result: {df}")
        stocks_with_saved_forecast = set(df["symbol"])
        logger.debug(
            f"""These stocks already have a saved forecast: {stocks_with_saved_forecast}"""
        )
        stocks_without_forecast = set(stocks_df["Symbol"]) - stocks_with_saved_forecast
        stocks_without_forecast = sorted(list(stocks_without_forecast))
        logger.debug(
            f"""These stocks do not have a saved forecast yet: {stocks_without_forecast}"""
        )
        return stocks_without_forecast

    def prep_next_stock_group(self, forecast_start_date=None):
        """Generator which iterates over all stocks and prepares them in groups."""
        stocks_file = f"{self.data_dir}/{self.data_3rd_party}/{self.stock_tickers_list}"
        logger.info(f"Loading stock tickers from {stocks_file}.")
        all_stock_tickers = pd.read_csv(stocks_file)
        # trim any duplicate tickers
        all_stock_tickers.index = all_stock_tickers.index.drop_duplicates()
        # drop any empty symbols/tickers
        all_stock_tickers = all_stock_tickers[all_stock_tickers.index.notnull()]
        logger.info(f"Loaded {len(all_stock_tickers)} symbols in total")
        stock_list = self._get_stocks_without_forecast(
            stocks_df=all_stock_tickers, forecast_start_date=forecast_start_date
        )
        logger.info(
            f"{len(all_stock_tickers)-len(stock_list)} stock already have forecast saved."
        )
        logger.info(f"{len(stock_list)} stock tickers candidates for new forecast.")
        if not stock_list:
            # Idempotent success: all symbols already have a partition for this start
            self.all_already_saved = True
            logger.info(
                "All listed symbols already have saved forecasts for this start date; nothing to do."
            )
            return
        self.all_already_saved = False
        if self.n_stocks < 1 or self.n_stocks > len(stock_list):
            self.n_stocks = len(stock_list)
        # group tickers in workable sample sizes for each forecast pass
        # credit ref: https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks
        for pos in range(0, len(stock_list), self.n_stocks):
            stock_group = stock_list[pos : pos + self.n_stocks]
            self.canswim_model.load_data(
                stock_tickers=stock_group, start_date=self.start_date
            )
            # prepare timeseries for forecast
            self.canswim_model.prepare_forecast_data(start_date=self.start_date)
            logger.info(f"Prepared forecast data for {len(stock_group)}: {stock_group}")
            yield pos

    def save_forecast(self, forecasts: dict = None, asof_start: pd.Timestamp = None):
        """Saves forecast data to local hive-partitioned parquet.

        Partition keys use ``asof_start`` when provided (CLI / backtest date) so
        skip-if-exists and monthly inventories match the requested start even if
        the first predicted bar falls on an adjacent session due to holidays.
        """

        def _list_to_df(forecasts: dict = None):
            """Format list of forecasts as a dataframe to be saved as a partitioned parquet dir"""
            forecast_df = pd.DataFrame()
            for t, ts in forecasts.items():
                pred_start = ts.start_time()
                partition_start = (
                    pd.Timestamp(asof_start).tz_localize(None).normalize()
                    if asof_start is not None
                    else pd.Timestamp(pred_start).tz_localize(None).normalize()
                )
                # logger.debug(f"Next forecast timeseries: {ts}")
                # normalize name of target series column if needed (e.g. "Adj Close" -> "Close")
                if self.canswim_model.target_column != "Close":
                    ts = ts.with_columns_renamed(
                        self.canswim_model.target_column, "Close"
                    )
                # convert probabilistic forecast into a dataframe with quantile samples as columns Close_01, Close_02...
                df = ts.pd_dataframe()
                # save commonly used quantiles only (drop raw MC sample columns)
                out = pd.DataFrame(index=df.index)
                for q in constants.quantiles:
                    qseries = df.quantile(q=q, axis=1)
                    qname = f"close_quantile_{q}"
                    out[qname] = qseries
                out["symbol"] = t
                out["forecast_start_year"] = partition_start.year
                out["forecast_start_month"] = partition_start.month
                out["forecast_start_day"] = partition_start.day
                # logger.debug(f"Next forecast sample: {out}")
                forecast_df = pd.concat([forecast_df, out])
            return forecast_df

        assert forecasts is not None and len(forecasts) > 0
        forecast_df = _list_to_df(forecasts)
        horizon = None
        try:
            horizon = int(self.canswim_model.pred_horizon)
        except Exception:
            try:
                horizon = int(self.canswim_model.torch_model.output_chunk_length)
            except Exception:
                horizon = None
        forecast_df, sanity_errors = validate_forecast_dataframe(
            forecast_df,
            expected_horizon=horizon,
            quantiles=list(constants.quantiles),
            drop_bad_symbols=True,
        )
        for msg in sanity_errors:
            logger.warning(f"Forecast sanity: {msg}")
        if forecast_df is None or forecast_df.empty:
            raise ValueError(
                "Forecast sanity check failed; nothing left to save. "
                + ("; ".join(sanity_errors) if sanity_errors else "")
            )
        logger.info(
            f"Saving forecast_df with {len(forecast_df.columns)} columns, "
            f"{len(forecast_df)} rows: {forecast_df}"
        )
        forecast_df.to_parquet(
            f"{self.data_dir}/{self.forecast_subdir}",
            partition_cols=[
                "symbol",
                "forecast_start_year",
                "forecast_start_month",
                "forecast_start_day",
            ],
        )
        logger.info(f"Saved forecast data to: {self.forecast_subdir}")

    def upload_data(self):
        self.hfhub.upload_data()

def get_next_open_market_day(after_date=None):
    """Get the date of the next open market day after a given date: after_date when provided or after today otherwise."""
    # Get calendar for NYSE
    nyse = mcal.get_calendar('NYSE')
    
    if after_date is None:
        # Get today's date
        today = datetime.now().date()
        after_date = today
    
    # Look for the next valid trading day within a reasonably big window of 20 regular business days
    valid_days = nyse.valid_days(start_date=after_date+BDay(1), end_date=after_date + BDay(20), tz=None)
    
    next_trading_day = None

    if valid_days is not None and len(valid_days) > 0:
        next_trading_day = valid_days[0]

    if next_trading_day is not None:
        logger.debug(f"The next open stock market date is: {next_trading_day}")
    else:
        logger.info("No open market day found within the next 30 days.")

    # If we can't find a next valid day within 30 days (which shouldn't happen for NYSE), return None
    return next_trading_day


# main function
def main(forecast_start_date: str = None):
    logger.info("Running forecast on stocks (local data; HF upload only if hfhub_sync)...")
    explicit_start = forecast_start_date is not None
    if explicit_start:
        logger.info(f"forecast_start_date: {forecast_start_date}")
        forecast_start_date = pd.Timestamp(forecast_start_date, tz=None)
        logger.info(f"Forecast start date set to: {forecast_start_date}")
    else:
        # Default is resolved after data prep: next open market day after the
        # latest ground-truth bar (not calendar "today"), so we do not skip
        # symbols when local prices lag the wall clock by one session.
        forecast_start_date = None
        logger.info(
            "Forecast start date: auto (next open market day after latest ground-truth bar)"
        )

    cf = CanswimForecaster()
    cf.all_already_saved = False
    cf.download_model()
    # Local-first: download_data is a no-op unless hfhub_sync=True
    cf.download_data()
    any_saved = False
    any_group = False
    for pos in cf.prep_next_stock_group(forecast_start_date=forecast_start_date):
        any_group = True
        fsd = forecast_start_date
        if fsd is None and cf.canswim_model.targets_list:
            latest_bar = max(ts.end_time() for ts in cf.canswim_model.targets_list)
            fsd = get_next_open_market_day(after_date=latest_bar)
            logger.info(
                f"Auto forecast start date {fsd} (next open after latest bar {latest_bar})"
            )
        forecasts = cf.get_forecast(forecast_start_date=fsd)
        if forecasts:
            # Drop any all-NaN probabilistic outputs (not ground-truth usable)
            clean = {}
            for t, ts in forecasts.items():
                try:
                    qdf = ts.quantile_df(0.5) if hasattr(ts, "quantile_df") else ts.pd_dataframe()
                    if qdf.isna().all().all():
                        logger.warning(
                            f"Skipping save for {t}: forecast quantiles are all NaN"
                        )
                        continue
                except Exception:
                    pass
                clean[t] = ts
            if clean:
                cf.save_forecast(clean, asof_start=fsd)
                any_saved = True
        else:
            logger.warning(
                "No eligible tickers with complete ground-truth history in this batch"
            )
    if not any_group:
        if getattr(cf, "all_already_saved", False):
            logger.info(
                "Finished forecast task (all symbols already had forecasts for this start)."
            )
            return
        raise RuntimeError(
            "Forecast aborted: no stock groups prepared. Check stock list and local price data."
        )
    if not any_saved:
        raise RuntimeError(
            "Forecast aborted: no forecasts saved. Missing ground-truth market data "
            "for requested symbols/window (no synthetic prices used)."
        )
    cf.upload_data()
    logger.info("Finished forecast task.")


if __name__ == "__main__":
    main()
